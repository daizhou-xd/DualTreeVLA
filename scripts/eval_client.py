"""
DualTreeVLA — LIBERO WebSocket Simulation Client
=================================================

Drives the LIBERO simulation environment and queries the DualTreeVLA
inference server (``eval_server.py``) over WebSocket for action predictions.

Architecture
------------
  eval_server.py  ←─── WebSocket ───→  eval_client.py
  (GPU node, model)                    (CPU/GPU, MuJoCo simulation)

This split mirrors Evo-1's server/client design so the heavy GPU model and
the MuJoCo physics simulation can run on separate machines or processes.

Protocol
--------
Client sends JSON:
  {"type": "reset"}                           → server resets HMT trees
  {"type": "infer",
   "image":       [H×W×3 uint8 flat list],    → agentview (flipped, RGB)
   "state":       [8 floats, physical scale], → eef_pos + axis_angle + gripper
   "instruction": "task description"}

Server replies JSON:
  {"status": "ok"}                            ← for reset
  {"actions": [[7 floats, physical] × H_a]}  ← for infer

Usage
-----
  # Evaluate libero_10, 10 episodes per task
  python scripts/eval_client.py \\
      --server ws://127.0.0.1:9000 \\
      --suite  libero_10 \\
      --num_episodes 10 \\
      --out results/libero10_server_eval.json

  # Evaluate all 4 suites sequentially
  python scripts/eval_client.py \\
      --server ws://127.0.0.1:9000 \\
      --suites libero_spatial libero_object libero_goal libero_10 \\
      --num_episodes 10 \\
      --out results/all_suites_eval.json \\
      --save_video

  # Quick sanity check (2 tasks, 3 episodes, save video)
  python scripts/eval_client.py \\
      --server ws://127.0.0.1:9000 \\
      --suite  libero_10 \\
      --num_episodes 3 \\
      --max_task_id  2 \\
      --save_video
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import pathlib
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root + LIBERO path ──────────────────────────────────────────
_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
_LIBERO_PKG_ROOT = _PROJ_ROOT / "dataset" / "LIBERO"
if _LIBERO_PKG_ROOT.is_dir():
    sys.path.insert(0, str(_LIBERO_PKG_ROOT))

import websockets  # pip install websockets>=12.0

# ── LIBERO imports ───────────────────────────────────────────────────────
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    _LIBERO_OK = True
except ImportError as _e:
    _LIBERO_OK = False
    _LIBERO_ERR = str(_e)


# ================================================================
#  Max steps per suite (same defaults as eval_libero_sim.py)
# ================================================================

_DEFAULT_MAX_STEPS: Dict[str, int] = {
    "libero_10":      600,
    "libero_spatial": 400,
    "libero_object":  400,
    "libero_goal":    400,
}

# Warm-up dummy actions to settle physics (matches Evo-1)
_N_WARMUP = 10
_DUMMY_ACTION = [0.0] * 7


# ================================================================
#  CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DualTreeVLA LIBERO client — queries eval_server.py over WebSocket"
    )
    p.add_argument("--server",  default="ws://127.0.0.1:9000",
                   help="WebSocket URL of eval_server.py, e.g. ws://0.0.0.0:9000")

    # Task suite selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--suite",
                   choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                   default=None,
                   help="Single task suite to evaluate")
    g.add_argument("--suites", nargs="+",
                   choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                   default=None,
                   help="One or more suites to evaluate sequentially")

    p.add_argument("--num_episodes", type=int, default=10,
                   help="Episodes per task (capped by provided init states)")
    p.add_argument("--max_task_id",  type=int, default=None,
                   help="Evaluate tasks 0 … max_task_id-1 only (quick testing)")
    p.add_argument("--max_steps",    type=int, default=None,
                   help="Max env steps per episode (suite default if omitted)")
    p.add_argument("--horizon",      type=int, default=16,
                   help="Action prediction horizon H_a; execute this many steps "
                        "before issuing the next server request")
    p.add_argument("--img_size",     type=int, default=224,
                   help="Camera resolution AND model input size (must match server)")

    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     default=None, help="JSON file to write results")
    p.add_argument("--log_file",default=None, help="Optional log file (appended)")

    p.add_argument("--save_video",  action="store_true",
                   help="Save per-episode MP4s to --video_dir")
    p.add_argument("--video_dir",   default="results/videos_server",
                   help="Directory for saved episode videos")

    p.add_argument("--no_image_flip", action="store_true",
                   help="Do NOT flip agentview vertically (default: flip to match training)")
    p.add_argument("--gripper_thresh", type=float, default=0.0,
                   help="Threshold on denormalized action[6] for gripper open/close "
                        "(>thresh → open +1, else close -1)")
    return p.parse_args()


# ================================================================
#  Logging
# ================================================================

def setup_logger(log_file: Optional[str]) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        d = os.path.dirname(log_file)
        if d:
            os.makedirs(d, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = handlers,
        force  = True,
    )
    return logging.getLogger("eval_client")


# ================================================================
#  Helpers
# ================================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """[x,y,z,w] quaternion → axis-angle (3,). No abs(w)."""
    q = quat.copy().astype(np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = q[:3] / s
    return (axis * angle).astype(np.float32)


def encode_image(img: np.ndarray) -> list:
    """(H, W, 3) uint8 → nested list for JSON encoding."""
    return img.astype(np.uint8).tolist()


def obs_to_state(obs: dict) -> np.ndarray:
    """Extract 8-dim proprioceptive state from LIBERO observation."""
    eef_pos = obs["robot0_eef_pos"].astype(np.float32)          # (3,)
    eef_aa  = quat2axisangle(obs["robot0_eef_quat"])             # (3,)
    gripper = obs["robot0_gripper_qpos"].astype(np.float32)[:2]  # (2,)
    return np.concatenate([eef_pos, eef_aa, gripper])            # (8,)


def build_infer_msg(obs: dict, instruction: str,
                    img_size: int, flip: bool) -> str:
    """Build the JSON message for one model-step request."""
    img = obs["agentview_image"]
    if flip:
        img = img[::-1]    # robosuite renders upside-down; match training convention
    if img.shape[0] != img_size or img.shape[1] != img_size:
        import cv2
        img = cv2.resize(img, (img_size, img_size))
    state = obs_to_state(obs).tolist()
    msg = {
        "type":        "infer",
        "image":       encode_image(img),
        "state":       state,
        "instruction": instruction,
    }
    return json.dumps(msg)


def save_video(frames: List[np.ndarray], path: str, fps: int = 20):
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {path}  ({len(frames)} frames)")
    except ImportError:
        print(f"  [WARN] cv2 unavailable — cannot save video: {path}")


# ================================================================
#  Single-suite evaluation loop
# ================================================================

async def eval_suite(
    ws,
    suite_name:   str,
    num_episodes: int,
    max_steps:    int,
    horizon:      int,
    img_size:     int,
    seed:         int,
    flip:         bool,
    gripper_thresh: float,
    save_vid:     bool,
    video_dir:    str,
    max_task_id:  Optional[int],
    log:          logging.Logger,
) -> dict:
    """
    Evaluate one task suite.  Returns a dict with per-task and overall stats.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[suite_name]()
    n_tasks        = task_suite.n_tasks
    task_limit     = n_tasks if max_task_id is None else min(max_task_id, n_tasks)

    W = 70
    COL = 50
    log.info(f"\n{'=' * W}")
    log.info(f"  Suite: {suite_name.upper()}  |  tasks: {task_limit}  "
             f"|  ep/task: {num_episodes}  |  max_steps: {max_steps}")
    log.info(f"  {'#':<3}  {'Task':<{COL}}  {'SR':>6}  {'n':>4}")
    log.info(f"  {'-' * (W - 2)}")

    per_task: List[dict] = []
    total_ok = total_ep = 0

    for task_id in range(task_limit):
        task          = task_suite.get_task(task_id)
        init_states   = task_suite.get_task_init_states(task_id)
        task_desc     = task.language
        bddl_file     = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": img_size,
            "camera_widths":  img_size,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        n_ep = min(num_episodes, len(init_states))
        n_ok = 0

        log.info(f"\n  Task {task_id + 1:>2}/{task_limit}: {task_desc}")

        for ep in range(n_ep):
            # ── Send reset → server clears HMT ────────────────────────
            await ws.send(json.dumps({"type": "reset"}))
            resp = json.loads(await ws.recv())
            assert resp.get("status") == "ok", f"Unexpected reset reply: {resp}"

            # ── LIBERO env reset + warm-up ─────────────────────────────
            env.reset()
            obs = env.set_init_state(init_states[ep])
            for _ in range(_N_WARMUP):
                obs, _, _, _ = env.step(_DUMMY_ACTION)

            success    = False
            terminated = False
            frames: List[np.ndarray] = []
            a_chunk_np: Optional[np.ndarray] = None

            for model_step in range(max_steps):
                if terminated:
                    break

                # ── Query server ───────────────────────────────────────
                msg = build_infer_msg(obs, task_desc, img_size, flip)
                t0  = time.time()
                await ws.send(msg)
                reply = json.loads(await ws.recv())
                latency_ms = (time.time() - t0) * 1000

                if "actions" not in reply:
                    log.error(f"    [ERROR] unexpected server reply: {reply}")
                    break

                a_chunk_np = np.array(reply["actions"], dtype=np.float32)  # (H_a, 7)
                if model_step == 0:
                    log.info(f"    ep {ep+1:>2} | step 0 latency={latency_ms:.0f}ms "
                             f"| a[0]={np.round(a_chunk_np[0], 3).tolist()}")

                # ── Execute H_a steps in simulation ───────────────────
                for h in range(min(horizon, a_chunk_np.shape[0])):
                    a_raw = a_chunk_np[h].copy()   # (7,) already denormalized

                    # Gripper binarization: follow Evo-1 convention
                    gripper = 1.0 if a_raw[6] > gripper_thresh else -1.0
                    action7 = a_raw[:6].tolist() + [gripper]

                    try:
                        obs, reward, done, info = env.step(action7)
                    except ValueError as e:
                        if "terminated" in str(e).lower():
                            terminated = True
                        else:
                            log.warning(f"    env.step error: {e}")
                        break

                    if save_vid:
                        frame = obs["agentview_image"]
                        if flip:
                            frame = frame[::-1]
                        frames.append(frame.copy())

                    if reward > 0 or info.get("success", False):
                        success    = True
                        terminated = True

                    if done:
                        terminated = True

                    if terminated:
                        break

            n_ok += int(success)
            status = "✅" if success else "❌"
            log.info(f"    ep {ep+1:>2}/{n_ep}  {status}")

            if save_vid:
                vid_path = os.path.join(
                    video_dir, suite_name,
                    f"task{task_id + 1:02d}_ep{ep + 1:02d}.mp4"
                )
                save_video(frames, vid_path)

        env.close()

        sr = n_ok / n_ep if n_ep > 0 else 0.0
        log.info(f"  {task_id+1:>3}  {task_desc:<{COL}}  {sr:>5.0%}  {n_ep:>4}")
        log.info(f"       ↳ {n_ok}/{n_ep} successful")
        per_task.append({
            "task_id":     task_id,
            "description": task_desc,
            "success":     n_ok,
            "episodes":    n_ep,
            "sr":          sr,
        })
        total_ok += n_ok
        total_ep += n_ep

    overall_sr = total_ok / total_ep if total_ep > 0 else 0.0
    log.info(f"\n{'=' * W}")
    log.info(f"  {suite_name.upper()} OVERALL: {total_ok}/{total_ep} = {overall_sr:.1%}")
    log.info(f"{'=' * W}\n")

    return {
        "suite":      suite_name,
        "total_ok":   total_ok,
        "total_ep":   total_ep,
        "overall_sr": overall_sr,
        "per_task":   per_task,
    }


# ================================================================
#  Main
# ================================================================

async def run(args: argparse.Namespace, log: logging.Logger):
    if not _LIBERO_OK:
        log.error(f"LIBERO import failed: {_LIBERO_ERR}")
        log.error("  pip install robosuite  &&  cd dataset/LIBERO && pip install -r requirements.txt")
        sys.exit(1)

    # Resolve suite list
    if args.suites:
        suites = args.suites
    elif args.suite:
        suites = [args.suite]
    else:
        # Default: all 4 suites
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    flip = not args.no_image_flip

    all_results = []

    async with websockets.connect(
        args.server,
        max_size=50_000_000,
        open_timeout=30,
    ) as ws:
        log.info(f"Connected to server: {args.server}")

        for suite in suites:
            max_steps = args.max_steps or _DEFAULT_MAX_STEPS.get(suite, 400)
            result = await eval_suite(
                ws           = ws,
                suite_name   = suite,
                num_episodes = args.num_episodes,
                max_steps    = max_steps,
                horizon      = args.horizon,
                img_size     = args.img_size,
                seed         = args.seed,
                flip         = flip,
                gripper_thresh = args.gripper_thresh,
                save_vid     = args.save_video,
                video_dir    = args.video_dir,
                max_task_id  = args.max_task_id,
                log          = log,
            )
            all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    if len(all_results) > 1:
        log.info("\n" + "=" * 50)
        log.info("  ACROSS ALL SUITES")
        log.info(f"  {'Suite':<20}  {'SR':>8}")
        log.info("  " + "-" * 30)
        grand_ok = grand_ep = 0
        for r in all_results:
            log.info(f"  {r['suite']:<20}  {r['overall_sr']:>7.1%}")
            grand_ok += r["total_ok"]
            grand_ep += r["total_ep"]
        grand_sr = grand_ok / grand_ep if grand_ep > 0 else 0.0
        log.info("  " + "-" * 30)
        log.info(f"  {'GRAND TOTAL':<20}  {grand_sr:>7.1%}  ({grand_ok}/{grand_ep})")
        log.info("=" * 50)

    # ── Write JSON output ─────────────────────────────────────────────
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": all_results}, f, indent=2)
        log.info(f"Results saved: {args.out}")


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    np.random.seed(args.seed)
    random.seed(args.seed)
    log = setup_logger(args.log_file)
    asyncio.run(run(args, log))


if __name__ == "__main__":
    main()
