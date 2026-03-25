"""
LIBEROEvaluator: WebSocket client that evaluates MemoryTreeVLA on LIBERO benchmarks.

Connects to a running MTVLA server (MTVLA/scripts/MTVLA_server.py) and drives
the LIBERO OffScreenRenderEnv through all four task suites:
  LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long.

Usage (via eval_libero.py)::

    # 1. Start the MTVLA inference server
    python MTVLA/scripts/MTVLA_server.py --ckpt_dir outputs/stage3/ --port 9000

    # 2. Run evaluation across all suites
    python LIBERO_evaluation/eval_libero.py \\
        --server_url ws://localhost:9000 \\
        --config LIBERO_evaluation/configs/libero_eval.yaml

    # 3. Single suite
    python LIBERO_evaluation/eval_libero.py \\
        --server_url ws://localhost:9000 \\
        --suite LIBERO-Goal \\
        --num_episodes 20
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np

try:
    import websockets
except ImportError:
    raise ImportError("websockets is required: pip install websockets>=12.0")

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MTVLA.utils import setup_logger, compute_success_rate

logger = setup_logger("LIBERO-Eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps evaluator suite name → (libero benchmark key, default max_steps)
_SUITE_REGISTRY: Dict[str, Tuple[str, int]] = {
    "LIBERO-Spatial": ("libero_spatial", 25),
    "LIBERO-Object":  ("libero_object",  25),
    "LIBERO-Goal":    ("libero_goal",    25),
    "LIBERO-Long":    ("libero_10",      95),   # libero_10 = long-horizon
}

_WARMUP_STEPS = 10       # dummy no-op steps before each episode
_DUMMY_ACTION = [0.0] * 7


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to axis-angle (3-D vector)."""
    quat = quat.copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den


def _encode_image(obs: dict, key: str = "agentview_image") -> list:
    """Return a uint8 H×W×3 image from obs as a nested Python list."""
    img = np.ascontiguousarray(obs[key][::-1, ::-1])  # flip to standard orientation
    return img.astype(np.uint8).tolist()


def _encode_state(obs: dict) -> list:
    """Build a flat state vector from LIBERO observation dict.

    Returns an 8-dimensional vector:
        eef_pos (3) + eef_quat→axis-angle (3) + gripper_qpos (2)

    The MTVLA server pads shorter state vectors to model.state_dim (15).
    """
    return np.concatenate([
        obs["robot0_eef_pos"],                          # (3,)
        _quat_to_axisangle(obs["robot0_eef_quat"]),     # (3,)
        obs["robot0_gripper_qpos"],                     # (2,)
    ]).tolist()


def _process_action(raw_action: list) -> list:
    """Post-process a single action step from the model.

    MTVLA action_dim=7: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
    LIBERO convention: gripper > 0 → close (−1), gripper ≤ 0 → open (+1).
    The model outputs a continuous gripper signal; we binarise it here.
    """
    action = list(raw_action)
    action[6] = -1.0 if action[6] > 0.0 else 1.0
    return action[:7]


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class LIBEROEvaluator:
    """Evaluate MemoryTreeVLA on LIBERO task suites via the MTVLA WebSocket server.

    Args:
        server_url  : WebSocket URL of MTVLA_server.py  (e.g. "ws://localhost:9000").
        cfg         : Evaluation config object or argparse Namespace.
                      Recognised fields: image_size (int), horizon (int),
                      record_video (bool), video_dir (str), max_steps (dict).
        result_dir  : Directory to write results.json.
        seed        : RNG seed for environment initialisation.
    """

    TASK_SUITES: List[str] = list(_SUITE_REGISTRY.keys())

    def __init__(
        self,
        server_url: str = "ws://localhost:9000",
        cfg=None,
        result_dir: str = "results/libero",
        seed: int = 42,
    ) -> None:
        if not _LIBERO_AVAILABLE:
            raise RuntimeError(
                "LIBERO is not installed. "
                "See https://github.com/Lifelong-Robot-Learning/LIBERO for instructions."
            )
        self.server_url  = server_url
        self.cfg         = cfg or {}
        self.result_dir  = Path(result_dir)
        self.seed        = seed
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Config helpers ─────────────────────────────────────────────────────
        def _get(key, default):
            if hasattr(self.cfg, key):
                return getattr(self.cfg, key)
            if isinstance(self.cfg, dict):
                return self.cfg.get(key, default)
            return default

        self.image_size    = _get("image_size",    448)
        self.horizon       = _get("horizon",       14)   # action steps executed per query
        self.record_video  = _get("record_video",  False)
        self.video_dir     = _get("video_dir",     "results/libero/videos")
        self.update_tree_every = _get("update_tree_every", 0)  # 0 = never

        # Per-suite max_steps overrides (dict) ───────────────────────────────
        self._max_steps_override: Dict[str, int] = _get("max_steps", {}) or {}

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_suite(
        self,
        suite_name: str,
        num_episodes: int = 50,
    ) -> Dict:
        """Run evaluation on a single LIBERO task suite.

        Blocks until all episodes are done.
        Returns a result dict with success_rate and per-task breakdown.
        """
        if suite_name not in _SUITE_REGISTRY:
            raise ValueError(f"Unknown suite '{suite_name}'. Valid: {self.TASK_SUITES}")
        return asyncio.run(self._async_evaluate_suite(suite_name, num_episodes))

    def evaluate_all(self, num_episodes: int = 50) -> Dict:
        """Evaluate on all four LIBERO task suites sequentially."""
        return asyncio.run(self._async_evaluate_all(num_episodes))

    # ── async internals ───────────────────────────────────────────────────────

    async def _async_evaluate_all(self, num_episodes: int) -> Dict:
        all_results: Dict = {}
        # Share a single WebSocket connection across all suites
        async with websockets.connect(
            self.server_url,
            max_size=100_000_000,
            open_timeout=30,
        ) as ws:
            for suite in self.TASK_SUITES:
                logger.info("=" * 60)
                logger.info("Suite: %s", suite)
                logger.info("=" * 60)
                all_results[suite] = await self._run_suite(ws, suite, num_episodes)

        self._save_results(all_results)
        self._log_summary(all_results)
        return all_results

    async def _async_evaluate_suite(self, suite_name: str, num_episodes: int) -> Dict:
        async with websockets.connect(
            self.server_url,
            max_size=100_000_000,
            open_timeout=30,
        ) as ws:
            result = await self._run_suite(ws, suite_name, num_episodes)
        self._save_results({suite_name: result})
        return result

    async def _run_suite(
        self,
        ws,
        suite_name: str,
        num_episodes: int,
    ) -> Dict:
        """Drive all tasks in one suite over an open WebSocket connection."""
        bench_key, default_max_steps = _SUITE_REGISTRY[suite_name]
        max_steps = self._max_steps_override.get(suite_name, default_max_steps)

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[bench_key]()
        num_tasks   = task_suite.n_tasks

        logger.info("%s — %d tasks | %d episodes each | max_steps=%d",
                    suite_name, num_tasks, num_episodes, max_steps)

        suite_results: Dict = {}
        total_success = 0
        total_episodes = 0

        for task_id in range(num_tasks):
            task           = task_suite.get_task(task_id)
            init_states    = task_suite.get_task_init_states(task_id)
            env, task_desc = self._make_env(task)

            n_ep = min(num_episodes, len(init_states))
            logger.info("  Task %02d/%02d: %s (%d episodes)",
                        task_id + 1, num_tasks, task_desc, n_ep)

            successes: List[bool] = []

            for ep in range(n_ep):
                success = await self._run_episode(
                    ws         = ws,
                    env        = env,
                    init_state = init_states[ep],
                    task_desc  = task_desc,
                    max_steps  = max_steps,
                    task_id    = task_id,
                    ep_idx     = ep,
                    suite_name = suite_name,
                )
                successes.append(success)
                status = "✅" if success else "❌"
                logger.info("    [%s] Episode %02d/%02d", status, ep + 1, n_ep)

            env.close()

            sr = compute_success_rate(successes)
            suite_results[f"task_{task_id + 1:02d}"] = {
                "description":  task_desc,
                "num_episodes": n_ep,
                "success_rate": sr,
                "successes":    int(sum(successes)),
            }
            total_success  += sum(successes)
            total_episodes += n_ep
            logger.info("    Task %02d success rate: %.1f%%", task_id + 1, sr * 100)

        overall_sr = total_success / max(total_episodes, 1)
        return {
            "suite":         suite_name,
            "num_episodes":  total_episodes,
            "success_rate":  overall_sr,
            "successes":     total_success,
            "tasks":         suite_results,
        }

    async def _run_episode(
        self,
        ws,
        env,
        init_state,
        task_desc: str,
        max_steps: int,
        task_id: int,
        ep_idx: int,
        suite_name: str,
    ) -> bool:
        """Run a single episode. Returns True on success."""
        # ── tell server to start new episode ─────────────────────────────────
        await ws.send(json.dumps({"cmd": "reset", "task": task_desc}))
        reset_resp = json.loads(await ws.recv())
        if reset_resp.get("status") != "ok":
            logger.warning("Server reset failed: %s", reset_resp)

        # ── reset environment ─────────────────────────────────────────────────
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(_WARMUP_STEPS):
            obs, _, _, _ = env.step(_DUMMY_ACTION)

        frames: List[np.ndarray] = []
        episode_success = False

        for step in range(max_steps):
            update_tree = (
                self.update_tree_every > 0
                and step % self.update_tree_every == 0
                and step > 0
            )

            # ── query server ─────────────────────────────────────────────────
            send_data = {
                "cmd":         "act",
                "image":       _encode_image(obs),
                "state":       _encode_state(obs),
                "update_tree": update_tree,
            }
            await ws.send(json.dumps(send_data))
            raw = await ws.recv()

            try:
                action_chunk = json.loads(raw)   # (horizon, action_dim)
            except Exception as e:
                logger.error("Action parse error at step %d: %s", step, e)
                break

            if isinstance(action_chunk, dict) and "error" in action_chunk:
                logger.error("Server error: %s", action_chunk["error"])
                break

            # ── execute action chunk in env ───────────────────────────────────
            for i, raw_action in enumerate(action_chunk[: self.horizon]):
                action = _process_action(raw_action)
                try:
                    obs, reward, done, info = env.step(action)
                except ValueError as e:
                    logger.warning("Invalid action at step %d sub %d: %s", step, i, e)
                    done = False

                if self.record_video:
                    frame = np.hstack([
                        np.rot90(obs["agentview_image"], 2),
                        np.rot90(obs["robot0_eye_in_hand_image"], 2),
                    ])
                    frames.append(frame)

                if done:
                    episode_success = True
                    break

            if episode_success:
                break

        if self.record_video and frames:
            self._save_video(frames, suite_name, task_id, ep_idx)

        return episode_success

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_env(self, task) -> Tuple["OffScreenRenderEnv", str]:
        """Instantiate a LIBERO OffScreenRenderEnv for the given task."""
        task_desc = task.language
        bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
        )
        env.seed(self.seed)
        return env, task_desc

    def _save_video(
        self,
        frames: List[np.ndarray],
        suite_name: str,
        task_id: int,
        ep_idx: int,
    ) -> None:
        video_dir = Path(self.video_dir) / suite_name
        video_dir.mkdir(parents=True, exist_ok=True)
        out_path = video_dir / f"task{task_id + 1:02d}_ep{ep_idx + 1:02d}.mp4"
        imageio.mimsave(str(out_path), frames, fps=30)
        logger.debug("Video saved: %s (%d frames)", out_path, len(frames))

    def _save_results(self, results: Dict) -> None:
        out_path = self.result_dir / "results.json"
        # Merge with existing results if the file already exists
        existing: Dict = {}
        if out_path.exists():
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update(results)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Results saved → %s", out_path)

    @staticmethod
    def _log_summary(all_results: Dict) -> None:
        logger.info("\n%s", "=" * 60)
        logger.info("  LIBERO Evaluation Summary")
        logger.info("=" * 60)
        for suite, res in all_results.items():
            sr = res.get("success_rate", 0.0)
            n  = res.get("num_episodes", 0)
            logger.info("  %-20s  %.1f%%  (%d episodes)", suite, sr * 100, n)
        logger.info("=" * 60)

