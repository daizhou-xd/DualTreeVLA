"""
DualTreeVLA — LIBERO Simulation Evaluation (Success Rate)
==========================================================

Runs the model in the LIBERO MuJoCo simulation environment and measures
task success rate.  Each task is evaluated for `num_episodes` episodes;
results are broken down per task and per suite.

Supported suites
----------------
  libero_10       LIBERO-LONG  (10 long-horizon tasks, max_steps=600)
  libero_spatial  LIBERO-SPATIAL (10 tasks)
  libero_object   LIBERO-OBJECT  (10 tasks)
  libero_goal     LIBERO-GOAL    (10 tasks)

Usage
-----
  # LIBERO-10, 10 episodes per task
  python scripts/eval_libero_sim.py \\
      --ckpt  checkpoints/runs/phase2/phase2_best.pt \\
      --config configs/train_phase2.yaml \\
      --suite libero_10 \\
      --num_episodes 10 \\
      --out results/libero10_sim_eval.json

  # Quick test: 3 episodes, first 2 tasks only, save videos
  python scripts/eval_libero_sim.py \\
      --ckpt  checkpoints/runs/phase2/phase2_best.pt \\
      --config configs/train_phase2.yaml \\
      --suite libero_10 \\
      --num_episodes 3 \\
      --max_task_id 2 \\
      --save_video \\
      --out results/libero10_quick.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── Project root on sys.path ─────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
# LIBERO package lives at dataset/LIBERO/libero — add its parent so
# `import libero` resolves correctly regardless of editable-install state.
_LIBERO_PKG_ROOT = _PROJ_ROOT / "dataset" / "LIBERO"
if _LIBERO_PKG_ROOT.is_dir():
    sys.path.insert(0, str(_LIBERO_PKG_ROOT))

# ── LIBERO imports ───────────────────────────────────────────────────────
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    _LIBERO_AVAILABLE = True
except ImportError as _e:
    _LIBERO_AVAILABLE = False
    _LIBERO_IMPORT_ERR = str(_e)

# ── Optional video writer ────────────────────────────────────────────────
try:
    import imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False


# ================================================================
#  Logging
# ================================================================

def _setup_logger(log_file: Optional[str]) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("eval_libero_sim")


# ================================================================
#  CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DualTreeVLA LIBERO simulation success-rate eval")

    p.add_argument("--ckpt",   required=True,  help="Path to .pt model checkpoint")
    p.add_argument("--config", default="configs/train_phase2.yaml",
                   help="Training config YAML (used to reconstruct model arch)")

    p.add_argument("--suite",
                   choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                   default="libero_10",
                   help="Which LIBERO task suite to evaluate")
    p.add_argument("--num_episodes", type=int, default=10,
                   help="Episodes per task (max = number of provided init states)")
    p.add_argument("--max_task_id", type=int, default=None,
                   help="Only evaluate tasks 0..max_task_id-1 (for quick testing)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Max env steps per episode "
                        "(defaults: libero_10=600, others=400)")

    p.add_argument("--img_size", type=int, default=224,
                   help="Image resolution fed to the model (env renders at this size)")
    p.add_argument("--horizon", type=int, default=16,
                   help="Action chunk horizon H_a; the model predicts H_a steps, "
                        "we execute all of them before the next model call")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)

    p.add_argument("--save_video", action="store_true",
                   help="Save per-episode MP4 videos under results/videos/")
    p.add_argument("--video_dir", default="results/videos",
                   help="Directory to save episode videos")

    p.add_argument("--out", default=None,
                   help="JSON file to write aggregated results")
    p.add_argument("--log_file", default=None,
                   help="Optional log file (appended)")

    p.add_argument("--data_root", default=None,
                   help="Path to dataset root containing meta/stats.json "
                        "(e.g. dataset/datasets/libero_10). Used to de-normalise "
                        "model outputs and normalise proprioception inputs. "
                        "If omitted the script tries to find it automatically.")

    p.add_argument("--no_image_flip", action="store_true",
                   help="Do NOT flip the agentview image vertically. "
                        "By default the image is flipped (robosuite renders "
                        "upside-down; lerobot v3 stores right-side-up). "
                        "If your lerobot dataset was built WITHOUT flipping, "
                        "pass this flag so inference matches training.")

    p.add_argument("--debug_first_ep", action="store_true",
                   help="Print state/action values for the first episode's "
                        "first step to help verify normalization. "
                        "Auto-saves that episode's video to results/videos/<suite>/debug_task00_ep00_*.mp4.")

    p.add_argument("--record_fail", action="store_true",
                   help="Auto-save the first failed episode video per task "
                        "(lightweight alternative to --save_video for debugging).")

    return p.parse_args()


# ================================================================
#  Action normalization / denormalization
# ================================================================

class ActionNorm:
    """
    Wraps the stats.json normalization used by LiberoDataset.
    Used to
      (a) normalize proprioceptive state  before passing to the model
      (b) denormalize predicted actions   before sending to the env
    """
    def __init__(self, stats_path: Optional[str]):
        self._a_mean: Optional[np.ndarray] = None
        self._a_std:  Optional[np.ndarray] = None
        self._s_mean: Optional[np.ndarray] = None
        self._s_std:  Optional[np.ndarray] = None
        if stats_path and os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            if "action" in stats:
                self._a_mean = np.array(stats["action"]["mean"], dtype=np.float32)
                self._a_std  = np.array(stats["action"]["std"],  dtype=np.float32)
                self._a_std  = np.where(self._a_std < 1e-6, 1.0, self._a_std)
            _sk = "observation.state" if "observation.state" in stats else "state"
            if _sk in stats:
                self._s_mean = np.array(stats[_sk]["mean"], dtype=np.float32)
                self._s_std  = np.array(stats[_sk]["std"],  dtype=np.float32)
                self._s_std  = np.where(self._s_std < 1e-6, 1.0, self._s_std)
            print(f"  [Norm] Loaded stats from {stats_path}")
        else:
            print("  [WARN] No stats.json found — actions will NOT be denormalised. "
                  "Pass --data_root to fix this.")

    @property
    def available(self) -> bool:
        return self._a_mean is not None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize proprioceptive state to match training distribution."""
        if self._s_mean is None:
            return state
        d = min(state.shape[0], self._s_mean.shape[0])
        out = state.copy()
        out[:d] = (state[:d] - self._s_mean[:d]) / self._s_std[:d]
        return out

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Reverse the training normalization applied to actions."""
        if self._a_mean is None:
            return action
        d = min(action.shape[0], self._a_mean.shape[0])
        out = action.copy()
        out[:d] = action[:d] * self._a_std[:d] + self._a_mean[:d]
        return out


def _find_stats_json(suite: str, data_root: Optional[str]) -> Optional[str]:
    """Try to locate meta/stats.json automatically."""
    candidates = []
    if data_root:
        candidates.append(os.path.join(data_root, "meta", "stats.json"))
        candidates.append(os.path.join(data_root, "stats.json"))
    # Auto-detection heuristic
    proj = str(_PROJ_ROOT)
    suite_dir_map = {
        "libero_10":      "libero_10",
        "libero_spatial": "libero_spatial",
        "libero_object":  "libero_object",
        "libero_goal":    "libero_goal",
    }
    sub = suite_dir_map.get(suite, suite)
    for base in ["dataset/datasets", "dataset/LIBERO", "data"]:
        candidates.append(os.path.join(proj, base, sub, "meta", "stats.json"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ================================================================
#  Default max_steps per suite
# ================================================================

_DEFAULT_MAX_STEPS: Dict[str, int] = {
    "libero_10":      600,
    "libero_spatial": 400,
    "libero_object":  400,
    "libero_goal":    400,
}


# ================================================================
#  Observation → model tensors
# ================================================================

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x,y,z,w] → axis-angle (3,).
    Matches robosuite / lerobot convention exactly: NO abs(w).
    (lerobot libero_10_image stores axis-angle computed this way —
     confirmed by stats.json showing max component > π, which only occurs
     when w < 0 and abs() is NOT applied.)
    """
    q = quat.copy().astype(np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))        # scalar part, keep sign!
    angle = 2.0 * np.arccos(w)                 # ∈ [0, 2π]
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = q[:3] / s
    return (axis * angle).astype(np.float32)


def obs_to_tensors(
    obs: dict,
    device: torch.device,
    img_size: int,
    d_q: int,
    action_norm: Optional["ActionNorm"] = None,
    flip_image: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a LIBERO observation dict to (image, state) tensors.

    Returns
    -------
    img_t   : (1, 3, img_size, img_size) float32 in [0, 1]
    state_t : (1, d_q) float32
    """
    try:
        from PIL import Image as _PILImage
        raw = obs["agentview_image"]
        if flip_image:
            raw = raw[::-1]          # robosuite renders upside-down; flip to right-side-up
        pil = _PILImage.fromarray(raw.astype(np.uint8))
        if pil.size != (img_size, img_size):
            pil = pil.resize((img_size, img_size), _PILImage.BILINEAR)
        img_np = np.array(pil, dtype=np.float32) / 255.0
    except Exception:
        import cv2
        raw = obs["agentview_image"]
        if flip_image:
            raw = raw[::-1]
        if raw.shape[:2] != (img_size, img_size):
            raw = cv2.resize(raw, (img_size, img_size))
        img_np = raw.astype(np.float32) / 255.0

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    # Proprioception: eef_pos (3) + eef_axisangle (3) + gripper (2)
    eef_pos   = obs["robot0_eef_pos"].astype(np.float32)          # (3,)
    eef_aa    = _quat2axisangle(obs["robot0_eef_quat"])            # (3,)
    gripper   = obs["robot0_gripper_qpos"].astype(np.float32)[:2]  # (2,)
    state_raw = np.concatenate([eef_pos, eef_aa, gripper])         # (8,)

    # Pad / truncate to d_q
    if state_raw.shape[0] < d_q:
        state_raw = np.pad(state_raw, (0, d_q - state_raw.shape[0]))
    else:
        state_raw = state_raw[:d_q]

    # Normalize state to match training distribution
    if action_norm is not None:
        state_raw = action_norm.normalize_state(state_raw)

    state_t = torch.from_numpy(state_raw).unsqueeze(0).to(device)  # (1, d_q)
    return img_t, state_t


# ================================================================
#  Model loading  (reused from eval.py)
# ================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    """Load DualTreeVLA from a training checkpoint (.pt file)."""
    from dual_tree_vla.model import DualTreeVLA

    m_cfg = cfg.get("model", {})
    model = DualTreeVLA(
        llm_path        = m_cfg.get("llm_path",        "checkpoints/Qwen2.5-0.5B"),
        clip_model_name = m_cfg.get("clip_model_name", None),
        d          = m_cfg.get("d",          256),
        d_a        = m_cfg.get("d_a",        7),
        d_q        = m_cfg.get("d_q",        84),
        d_visual   = m_cfg.get("d_visual",   256),
        d_ssm      = m_cfg.get("d_ssm",      256),
        d_state    = m_cfg.get("d_state",    16),
        patch_size = m_cfg.get("patch_size", 16),
        H_a        = m_cfg.get("H_a",        16),
        n_ode      = m_cfg.get("n_ode",      20),
        theta_fuse = m_cfg.get("theta_fuse", 0.35),
        K_elev     = m_cfg.get("K_elev",     4),
        delta_w    = m_cfg.get("delta_w",    0.1),
        tau        = m_cfg.get("tau",        0.1),
        freeze_llm = False,
    )

    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = (
        state.get("model")
        or state.get("model_state_dict")
        or state.get("module")
        or state
    )

    model_sd = model.state_dict()
    sd_clean = {k: v for k, v in sd.items()
                if k not in model_sd or v.shape == model_sd[k].shape}
    skipped = len(sd) - len(sd_clean)
    if skipped:
        print(f"[WARN] Skipped {skipped} shape-mismatched keys (ZeRO-3 placeholders)")

    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"[WARN] Missing  ({len(missing)}): {missing[:4]}{'...' if len(missing)>4 else ''}")
    if unexpected:
        print(f"[WARN] Unexpect ({len(unexpected)}): {unexpected[:4]}{'...' if len(unexpected)>4 else ''}")

    model.to(device).eval()
    print(f"  Model loaded. sem_proj norm={sum(p.norm().item() for p in model.sem_proj.parameters()):.3f}")
    return model


# ================================================================
#  Language cache  (same trick as eval.py)
# ================================================================

def _patch_lang_cache(model):
    """Replace model._encode_language with a memoised version."""
    import types
    _cache: dict = {}
    _orig = model._encode_language.__func__

    def _cached(self, instructions, dev):
        key = tuple(instructions)
        if key not in _cache:
            with torch.no_grad():
                h, g = _orig(self, instructions, dev)
            _cache[key] = (h.detach().cpu(), g.detach().cpu())
        h, g = _cache[key]
        return h.to(dev), g.to(dev)

    model._encode_language = types.MethodType(_cached, model)


# ================================================================
#  Video save
# ================================================================

def _save_video(frames: List[np.ndarray], path: str, fps: int = 20):
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if _IMAGEIO_AVAILABLE:
        imageio.mimsave(path, frames, fps=fps)
        print(f"  Video saved: {path}  ({len(frames)} frames)")
        return

    # Fallback: cv2 VideoWriter
    try:
        import cv2 as _cv2
        h, w = frames[0].shape[:2]
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        writer = _cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(_cv2.cvtColor(f, _cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved (cv2): {path}  ({len(frames)} frames)")
        return
    except ImportError:
        pass

    print(f"  [WARN] Neither imageio nor cv2 available — cannot save video {path}")
    print(f"         Install with:  pip install imageio[ffmpeg]  or  pip install opencv-python")


def _debug_step(
    step_idx: int,
    state_t: torch.Tensor,
    a_chunk: torch.Tensor,
    action_norm: Optional[ActionNorm],
):
    """
    Print state and first action values for the given step index.
    Called for each of the first `debug_steps` model-call steps.
    """
    state_np = state_t[0].cpu().numpy()      # (d_q,) — normalized
    a_norm   = a_chunk[0, 0].cpu().numpy()   # (d_a,) — normalized (first action of chunk)
    a_raw    = action_norm.denormalize_action(a_norm.copy()) if action_norm else a_norm.copy()

    np.set_printoptions(precision=4, suppress=True)
    print(f"  [DEBUG step {step_idx:>3}]")
    if step_idx == 0:
        print(f"    state eef_pos  (norm): {state_np[:3]}")
        print(f"    state axis_ang (norm): {state_np[3:6]}")
        print(f"    state gripper  (norm): {state_np[6:8]}")
    print(f"    action norm    : {a_norm}")
    print(f"    action denorm  : {a_raw}")
    if action_norm is None or action_norm._a_mean is None:
        print("    [WARN] no stats — action NOT denormalized")


def _debug_episode_summary(all_denorm: List[np.ndarray]):
    """Print per-dimension statistics over a full episode's denormed actions."""
    if not all_denorm:
        return
    arr = np.stack(all_denorm, axis=0)   # (T, d_a)
    np.set_printoptions(precision=4, suppress=True)
    print("\n  [DEBUG] Episode action statistics (denormed, per dim):")
    print(f"    mean : {arr.mean(0)}")
    print(f"    std  : {arr.std(0)}")
    print(f"    min  : {arr.min(0)}")
    print(f"    max  : {arr.max(0)}")
    print()


# ================================================================
#  Single-episode rollout
# ================================================================

@torch.no_grad()
def run_episode(
    model,
    env,
    task_description: str,
    initial_state,
    max_steps: int,
    horizon: int,
    img_size: int,
    d_q: int,
    device: torch.device,
    save_video: bool = False,
    action_norm: Optional[ActionNorm] = None,
    flip_image: bool = True,
    debug_steps: int = 0,
) -> Tuple[bool, List[np.ndarray]]:
    """
    Roll out one episode.

    Returns
    -------
    success : bool
    frames  : list of RGB uint8 arrays (empty when save_video=False and debug_steps=0)
    """
    # LIBERO dummy warm-up (settle physics)
    DUMMY = [0.0] * 7
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY)

    model.reset_trees(batch_size=1)
    a_prev = None
    frames: List[np.ndarray] = []
    record_frames = save_video or (debug_steps > 0)   # collect frames whenever we might want a video
    success    = False
    terminated = False
    all_denorm: List[np.ndarray] = []   # for debug summary

    for step in range(max_steps):
        if terminated:
            break

        img_t, state_t = obs_to_tensors(obs, device, img_size, d_q, action_norm,
                                         flip_image=flip_image)

        # Model predicts a chunk of H_a actions
        a_chunk = model.step(img_t, task_description, state_t, a_prev)  # (1, H_a, d_a)

        if step < debug_steps:
            _debug_step(step, state_t, a_chunk, action_norm)

        for h in range(horizon):
            raw_action = a_chunk[0, h].cpu().numpy()  # (d_a,) — still in normalised space

            # Denormalise action back to env's physical scale
            if action_norm is not None:
                raw_action = action_norm.denormalize_action(raw_action)

            # Gripper binarisation: follow Evo-1 convention
            # positive logit → open (1.0), negative → close (-1.0)
            gripper = 1.0 if raw_action[6] > 0.0 else -1.0
            action_7 = raw_action[:6].tolist() + [gripper]

            try:
                obs, reward, done, info = env.step(action_7)
            except ValueError as e:
                # "executing action in terminated episode" means the env's
                # internal horizon was reached; stop stepping silently.
                if "terminated" in str(e).lower():
                    terminated = True
                else:
                    print(f"    [WARN] env.step error at step={step} h={h}: {e}")
                break

            if record_frames:
                frame = obs["agentview_image"]
                if flip_image:
                    frame = frame[::-1]
                frames.append(frame.copy())

            if debug_steps > 0:
                all_denorm.append(raw_action.copy())

            # Success: positive reward OR explicit flag in info
            if reward > 0 or info.get("success", False):
                success    = True
                terminated = True

            # done=True may be timeout (env horizon) — mark terminated but
            # don't override success; success was set above if reward > 0.
            if done:
                terminated = True

            if terminated:
                break

        # Feed last action of chunk as a_prev for continuity
        a_prev = a_chunk[0, -1].unsqueeze(0)   # (1, d_a)

    if all_denorm:
        _debug_episode_summary(all_denorm)

    return success, frames


# ================================================================
#  Main evaluation loop
# ================================================================

def main():
    args = parse_args()
    log  = _setup_logger(args.log_file)

    if not _LIBERO_AVAILABLE:
        print(f"[ERROR] LIBERO or its dependencies not available: {_LIBERO_IMPORT_ERR}")
        print("  Fix:")
        print("    pip install robosuite")
        print("    cd dataset/LIBERO && pip install -r requirements.txt")
        sys.exit(1)

    os.environ.setdefault("MUJOCO_GL", "osmesa")     # headless rendering
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device     = torch.device(args.device)
    max_steps  = args.max_steps or _DEFAULT_MAX_STEPS.get(args.suite, 400)
    cfg        = load_config(args.config)
    d_q        = cfg.get("model", {}).get("d_q", 84)
    img_size   = args.img_size
    horizon    = args.horizon

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, cfg, device)
    _patch_lang_cache(model)

    # ── Load action normalization stats ───────────────────────────────
    stats_path  = _find_stats_json(args.suite, args.data_root)
    action_norm = ActionNorm(stats_path)

    # ── Load task suite ───────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[args.suite]()
    n_tasks        = task_suite.n_tasks
    task_limit     = args.max_task_id if args.max_task_id is not None else n_tasks
    task_limit     = min(task_limit, n_tasks)

    log.info(f"Suite: {args.suite}  |  tasks: {task_limit}/{n_tasks}  "
             f"|  episodes/task: {args.num_episodes}  |  max_steps: {max_steps}")

    # ── Per-task evaluation ───────────────────────────────────────────
    W = 70
    print(f"\n{'=' * W}")
    print(f"  LIBERO SIMULATION EVAL — {args.suite.upper()}")
    print(f"{'=' * W}")
    COL = 50
    print(f"  {'#':<3}  {'Task':<{COL}}  {'SR':>6}  {'n':>4}")
    print(f"  {'-' * (W - 2)}")

    per_task_results: List[dict] = []
    total_success = 0
    total_episodes = 0

    for task_id in range(task_limit):
        task            = task_suite.get_task(task_id)
        init_states     = task_suite.get_task_init_states(task_id)
        task_desc       = task.language
        bddl_file       = pathlib.Path(get_libero_path("bddl_files")) \
                          / task.problem_folder / task.bddl_file
        env_args        = {
            "bddl_file_name":  str(bddl_file),
            "camera_heights":  img_size,
            "camera_widths":   img_size,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        n_ep     = min(args.num_episodes, len(init_states))
        n_ok     = 0
        ep_data: List[dict] = []

        log.info(f"\n{'─' * W}")
        log.info(f"  Task {task_id+1:>2}/{task_limit}: {task_desc}")

        t0 = time.time()
        first_fail_saved = False
        for ep in range(n_ep):
            is_debug_ep  = args.debug_first_ep and task_id == 0 and ep == 0
            want_fail_vid = args.record_fail and not first_fail_saved
            # Tell run_episode to collect frames if any video path will be needed
            collect = args.save_video or is_debug_ep or want_fail_vid
            success, frames = run_episode(
                model            = model,
                env              = env,
                task_description = task_desc,
                initial_state    = init_states[ep],
                max_steps        = max_steps,
                horizon          = horizon,
                img_size         = img_size,
                d_q              = d_q,
                device           = device,
                save_video       = collect,
                action_norm      = action_norm,
                flip_image       = not args.no_image_flip,
                debug_steps      = 10 if is_debug_ep else 0,
            )
            n_ok += int(success)
            status = "✅" if success else "❌"
            log.info(f"    ep {ep+1:>2}/{n_ep}  {status}")
            ep_data.append({"episode": ep, "success": success})

            # Determine which videos to actually write
            if frames:
                to_save: List[Tuple[str, str]] = []   # (vid_path, reason)
                if args.save_video:
                    to_save.append((
                        os.path.join(args.video_dir, args.suite,
                                     f"task{task_id:02d}_ep{ep:02d}_{'ok' if success else 'fail'}.mp4"),
                        "save_video"
                    ))
                elif is_debug_ep:
                    to_save.append((
                        os.path.join(args.video_dir, args.suite,
                                     f"debug_task{task_id:02d}_ep{ep:02d}_{'ok' if success else 'fail'}.mp4"),
                        "debug"
                    ))
                elif want_fail_vid and not success:
                    to_save.append((
                        os.path.join(args.video_dir, args.suite,
                                     f"fail_task{task_id:02d}_ep{ep:02d}.mp4"),
                        "record_fail"
                    ))
                    first_fail_saved = True
                for vid_path, reason in to_save:
                    _save_video(frames, vid_path)

        elapsed = time.time() - t0
        sr = n_ok / n_ep
        task_name = task_desc[:COL] if len(task_desc) > COL else task_desc
        print(f"  {task_id+1:<3}  {task_name:<{COL}}  {sr:>6.2%}  {n_ep:>4}   ({elapsed:.0f}s)")
        log.info(f"  Task {task_id+1} summary: {n_ok}/{n_ep} ({sr:.2%})")

        per_task_results.append({
            "task_id":     task_id,
            "task_desc":   task_desc,
            "n_episodes":  n_ep,
            "n_success":   n_ok,
            "success_rate": sr,
            "episodes":    ep_data,
        })
        total_success  += n_ok
        total_episodes += n_ep

        env.close()

    # ── Overall summary ───────────────────────────────────────────────
    overall_sr = total_success / max(total_episodes, 1)
    print(f"\n{'=' * W}")
    print(f"  OVERALL  {args.suite.upper()}")
    print(f"  Success rate : {overall_sr:.2%}  ({total_success}/{total_episodes})")
    print(f"{'=' * W}")
    log.info(f"\n{'=' * W}")
    log.info(f"OVERALL {args.suite}: {total_success}/{total_episodes} = {overall_sr:.2%}")

    # ── Save JSON ─────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint":    args.ckpt,
            "config":        args.config,
            "suite":         args.suite,
            "num_episodes":  args.num_episodes,
            "max_steps":     max_steps,
            "horizon":       horizon,
            "seed":          args.seed,
            "overall_success_rate": overall_sr,
            "total_success":        total_success,
            "total_episodes":       total_episodes,
            "per_task":      per_task_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
