"""
MTVLA_server.py — WebSocket inference server for MemoryTreeVLA.

Reference architecture adapted from Evo-1 (MINT-SJTU/Evo-1).

Protocol (JSON over WebSocket)
───────────────────────────────────────────────────────────────────────────────
Reset / start new episode
  Client → Server:
    {
      "cmd":       "reset",
      "task":      "<natural language task description>",
      "tree_json": "<optional initial Tree.json string>"   // omit to auto-generate
    }
  Server → Client:
    {"status": "ok", "tree": "<generated or echoed Tree.json string>"}

Action inference (called every control step)
  Client → Server:
    {
      "cmd":         "act",
      "image":       [[[R,G,B], ...], ...],   // H×W×3 uint8 list
      "state":       [f1, f2, ..., f15],      // state_dim floats (raw / normalised)
      "update_tree": false                    // true → run Tree LLM update this step
    }
  Server → Client:
    [[a0, a1, ..., a6], ...]                  // (horizon, action_dim) float list

───────────────────────────────────────────────────────────────────────────────

Usage examples:

  # Basic start
  python MTVLA/scripts/MTVLA_server.py \\
      --ckpt_dir outputs/stage3/ \\
      --port 9000

  # With normalisation stats (recommended for real robots)
  python MTVLA/scripts/MTVLA_server.py \\
      --ckpt_dir outputs/stage3/ \\
      --norm_stats dataset/norm_stats.json \\
      --port 9000

  # CPU-only debug mode (inference only, no CUDA required)
  python MTVLA/scripts/MTVLA_server.py \\
      --ckpt_dir outputs/stage3/ \\
      --device cpu \\
      --port 9000

Checkpoint directory layout expected by --ckpt_dir:
  <ckpt_dir>/
    config.yaml            # model config (ModelConfig block)
    stage3_best.pth        # or stage2_best.pth / any .pth — first one found
    norm_stats.json        # optional; can also be passed via --norm_stats

norm_stats.json format:
  {
    "state": {"min": [...], "max": [...]},
    "action": {"min": [...], "max": [...]}
  }
  All arrays must have length == state_dim (15) and action_dim (7) respectively.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── resolve project root ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from MTVLA.configs.config import ModelConfig, MTVLAConfig, load_config
from MTVLA.models import MemoryTreeVLA
from MTVLA.models.memory_tree import MemoryTree
from MTVLA.utils.logger import setup_logger

try:
    import websockets
except ImportError:
    raise ImportError("websockets is required: pip install websockets>=12.0")

logger = setup_logger("MTVLA-Server")


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helper  (mirrors Evo-1 Normalizer)
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """Min-max normaliser / de-normaliser for proprioceptive state and actions.

    State is normalised to [-1, 1] before being fed to the model.
    Model output actions are de-normalised back to raw joint-space.

    norm_stats.json expected format::

        {
          "state":  {"min": [...], "max": [...]},
          "action": {"min": [...], "max": [...]}
        }

    Array lengths must match state_dim (default 15) and action_dim (default 7).
    """

    def __init__(self, stats_or_path: "str | dict") -> None:
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def _t(x: list) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32)

        self.state_min  = _t(stats["state"]["min"])
        self.state_max  = _t(stats["state"]["max"])
        self.action_min = _t(stats["action"]["min"])
        self.action_max = _t(stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        lo = self.state_min.to(state.device, dtype=state.dtype)
        hi = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2.0 * (state - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, horizon, action_dim) or (B, action_dim)."""
        lo = self.action_min.to(action.device, dtype=action.dtype)
        hi = self.action_max.to(action.device, dtype=action.dtype)
        return (action + 1.0) / 2.0 * (hi - lo + 1e-8) + lo


# ─────────────────────────────────────────────────────────────────────────────
# Model / checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_checkpoint(ckpt_dir: Path) -> Path:
    """Return the best available checkpoint inside *ckpt_dir*, preferring
    stage3 > stage2 > stage1, then any other .pth file."""
    priority = [
        "stage3_best.pth", "stage3_latest.pth",
        "stage2_best.pth", "stage2_latest.pth",
        "stage1_best.pth", "stage1_latest.pth",
    ]
    for name in priority:
        p = ckpt_dir / name
        if p.exists():
            return p
    # Any .pth
    ptfiles = sorted(ckpt_dir.glob("*.pth"))
    if ptfiles:
        return ptfiles[-1]
    raise FileNotFoundError(
        f"No .pth checkpoint found in {ckpt_dir}. "
        "Train the model first or pass --ckpt_dir pointing to the checkpoint."
    )


def load_model(
    ckpt_dir: str,
    device: torch.device,
    config_override: Optional[str] = None,
) -> "tuple[MemoryTreeVLA, MTVLAConfig]":
    """Load MemoryTreeVLA from checkpoint directory.

    Args:
        ckpt_dir        : directory containing config.yaml + *.pth.
        device          : target device.
        config_override : optional path to a different YAML config.

    Returns:
        Model in eval mode, on *device*.
    """
    ckpt_path_dir = Path(ckpt_dir)

    # ── config ───────────────────────────────────────────────────────────────
    cfg_file = config_override or str(ckpt_path_dir / "config.yaml")
    if not Path(cfg_file).exists():
        # Fall back to default
        default_cfg = str(_REPO_ROOT / "MTVLA" / "configs" / "default.yaml")
        logger.warning(
            "No config.yaml in ckpt_dir; using default config: %s", default_cfg
        )
        cfg_file = default_cfg

    cfg: MTVLAConfig = load_config(cfg_file)

    # ── build model ──────────────────────────────────────────────────────────
    logger.info("Building MemoryTreeVLA …")
    model = MemoryTreeVLA(cfg)

    # ── load weights ─────────────────────────────────────────────────────────
    ckpt_pth = _find_checkpoint(ckpt_path_dir)
    logger.info("Loading checkpoint: %s", ckpt_pth)
    ckpt = torch.load(ckpt_pth, map_location="cpu")

    # Support both raw state_dict and train.py's wrapped format
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "module" in ckpt:
        # DeepSpeed checkpoint
        state_dict = ckpt["module"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])

    model = model.to(device).eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded — %.2fM parameters", total)
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

_TO_TENSOR = transforms.ToTensor()

def decode_image(img_list: list, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert a raw uint8 H×W×3 list to a normalised (1, 3, H, W) tensor.

    Args:
        img_list   : nested list [H][W][3] or flat list of the same.
        image_size : target spatial resolution (square crop).
        device     : target device.

    Returns:
        Float32 tensor of shape ``(1, 3, image_size, image_size)`` in [0, 1].
    """
    arr = np.array(img_list, dtype=np.uint8)
    if arr.ndim == 1:
        # flat encoding → try to infer HxW from length
        side = int(round(arr.size ** 0.5 / 3) ** 0.5)
        arr = arr.reshape(side, side, 3)

    # resize to model's expected spatial size
    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # BGR (OpenCV default) → RGB  (skip if already RGB from client)
    # Clients should send RGB; leave the conversion optional.
    pil = Image.fromarray(arr)
    tensor = _TO_TENSOR(pil).unsqueeze(0).to(device)   # (1, 3, H, W)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# MemoryTree helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_minimal_tree(task: str) -> MemoryTree:
    """Create a single-node MemoryTree from a raw task description string.

    Used when no Tree.json is provided at episode reset.  The Tree LLM will
    elaborate the tree once ``update_tree=True`` is sent in an "act" message.
    """
    minimal_dict = {
        "task_description": task,
        "root": {
            "id":          "0",
            "description": task,
            "status":      "not_started",
            "children":    [],
        },
    }
    return MemoryTree.from_dict(minimal_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode inference context
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeContext:
    """Holds per-episode mutable state shared across WebSocket messages.

    One context is created at server start and reset between episodes via
    the "reset" command.
    """

    def __init__(self, model: MemoryTreeVLA, cfg: MTVLAConfig, device: torch.device) -> None:
        self.model  = model
        self.cfg    = cfg
        self.device = device

        self.task_description: str = ""
        self.step_count: int = 0

    def reset(self, task: str, tree_json: Optional[str] = None) -> str:
        """Start a new episode.

        Args:
            task      : natural language task description.
            tree_json : optional pre-built Tree.json string.

        Returns:
            Tree.json string (generated or echoed).
        """
        self.model.reset()
        self.task_description = task
        self.step_count = 0

        if tree_json:
            # Parse and attach the provided tree
            try:
                tree = MemoryTree.from_json_string(tree_json)
            except Exception as e:
                logger.warning("Could not parse provided tree_json: %s", e)
                tree = _make_minimal_tree(task)
        else:
            # Build a minimal tree from the task description (no vision yet)
            tree = _make_minimal_tree(task)

        self.model.set_memory_tree(tree)
        logger.info("Episode reset — task: %s | tree nodes: %d", task, len(tree))
        return str(tree)

    @torch.no_grad()
    def act(
        self,
        image_list: list,
        state_list: list,
        normalizer: Optional[Normalizer],
        update_tree: bool = False,
    ) -> list:
        """Run inference for one control step.

        Args:
            image_list  : H×W×3 uint8 list (from client JSON).
            state_list  : proprioceptive state floats.
            normalizer  : optional Normalizer for state / action.
            update_tree : whether to run Tree LLM update first.

        Returns:
            Nested list ``(horizon, action_dim)`` of float action values.
        """
        image_size = self.cfg.model.image_size
        horizon    = self.cfg.model.action_horizon
        action_dim = self.cfg.model.action_dim

        # ── image ────────────────────────────────────────────────────────────
        image = decode_image(image_list, image_size, self.device)  # (1,3,H,W)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        image = image.to(dtype=dtype)

        # ── state ────────────────────────────────────────────────────────────
        state_dim = self.cfg.model.state_dim
        state_raw = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        if state_raw.ndim == 1:
            state_raw = state_raw.unsqueeze(0)                     # (1, state_dim)

        # Pad or truncate to model's expected state_dim
        if state_raw.shape[1] < state_dim:
            pad = torch.zeros(1, state_dim - state_raw.shape[1],
                              device=self.device)
            state_raw = torch.cat([state_raw, pad], dim=1)
        elif state_raw.shape[1] > state_dim:
            state_raw = state_raw[:, :state_dim]

        state = (normalizer.normalize_state(state_raw) if normalizer else state_raw
                 ).to(dtype=dtype)

        # ── optional: update memory tree via Tree LLM ─────────────────────────
        if update_tree:
            logger.debug("Step %d — updating memory tree …", self.step_count)
            new_tree_text = self.model.update_tree(
                image,
                self.task_description,
                max_new_tokens=256,
            )
            try:
                new_tree = MemoryTree.from_json_string(new_tree_text)
                self.model.set_memory_tree(new_tree)
                logger.debug("Tree updated: %d nodes", len(new_tree))
            except Exception as e:
                logger.warning("Tree LLM output could not be parsed: %s", e)

        # ── inference ────────────────────────────────────────────────────────
        autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=autocast_dtype)

        with autocast_ctx:
            actions_flat = self.model.act(images=image, state=state)
            # (B=1, horizon * action_dim)  →  (1, horizon, action_dim)
            actions = actions_flat.reshape(1, horizon, action_dim).float()

        if normalizer:
            actions = normalizer.denormalize_action(actions)

        self.step_count += 1
        return actions[0].cpu().tolist()   # (horizon, action_dim)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket request handler
# ─────────────────────────────────────────────────────────────────────────────

async def handle_client(
    websocket,
    ctx: EpisodeContext,
    normalizer: Optional[Normalizer],
) -> None:
    """Handle one WebSocket client connection (one evaluation episode)."""
    remote = websocket.remote_address
    logger.info("Client connected: %s", remote)

    try:
        async for raw_message in websocket:
            try:
                data: dict = json.loads(raw_message)
            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({"error": f"JSON parse error: {e}"}))
                continue

            cmd = data.get("cmd", "act")

            # ── reset ─────────────────────────────────────────────────────────
            if cmd == "reset":
                task      = data.get("task", "")
                tree_json = data.get("tree_json", None)
                if not task:
                    await websocket.send(json.dumps({"error": "'task' field required for reset"}))
                    continue
                tree_str = ctx.reset(task=task, tree_json=tree_json)
                await websocket.send(json.dumps({"status": "ok", "tree": tree_str}))

            # ── act ───────────────────────────────────────────────────────────
            elif cmd == "act":
                if "image" not in data or "state" not in data:
                    await websocket.send(
                        json.dumps({"error": "'image' and 'state' are required for act"})
                    )
                    continue

                update_tree = bool(data.get("update_tree", False))

                try:
                    actions = ctx.act(
                        image_list=data["image"],
                        state_list=data["state"],
                        normalizer=normalizer,
                        update_tree=update_tree,
                    )
                except Exception as exc:
                    logger.exception("Inference error: %s", exc)
                    await websocket.send(json.dumps({"error": str(exc)}))
                    continue

                await websocket.send(json.dumps(actions))

            # ── unknown command ───────────────────────────────────────────────
            else:
                await websocket.send(json.dumps({"error": f"Unknown cmd '{cmd}'"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected: %s", remote)
    except Exception as exc:
        logger.exception("Unexpected error for client %s: %s", remote, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MemoryTreeVLA WebSocket inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Directory containing config.yaml + *.pth checkpoint(s)."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Override path to model config YAML (default: <ckpt_dir>/config.yaml)."
    )
    parser.add_argument(
        "--norm_stats", type=str, default=None,
        help="Path to norm_stats.json for state / action normalisation. "
             "If omitted, also checks <ckpt_dir>/norm_stats.json."
    )
    parser.add_argument(
        "--port", type=int, default=9000,
        help="WebSocket port (default: 9000)."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device — 'cuda', 'cuda:1', 'cpu', etc. "
             "Defaults to 'cuda' if available, else 'cpu'."
    )
    parser.add_argument(
        "--max_size", type=int, default=100_000_000,
        help="Maximum WebSocket message size in bytes (default: 100 MB)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Loading MemoryTreeVLA from %s …", args.ckpt_dir)
    model, cfg = load_model(args.ckpt_dir, device, config_override=args.config)

    # ── normaliser (optional) ─────────────────────────────────────────────────
    normalizer: Optional[Normalizer] = None
    norm_path = args.norm_stats or str(Path(args.ckpt_dir) / "norm_stats.json")
    if Path(norm_path).exists():
        logger.info("Loading normalisation stats: %s", norm_path)
        normalizer = Normalizer(norm_path)
    else:
        logger.info(
            "No norm_stats.json found — running without state/action normalisation."
        )

    # ── episode context ───────────────────────────────────────────────────────
    ctx = EpisodeContext(model=model, cfg=cfg, device=device)

    # ── WebSocket server ──────────────────────────────────────────────────────
    async def _serve() -> None:
        logger.info(
            "MTVLA server running at ws://%s:%d  (max_msg=%d bytes)",
            args.host, args.port, args.max_size,
        )
        async with websockets.serve(
            lambda ws: handle_client(ws, ctx, normalizer),
            args.host,
            args.port,
            max_size=args.max_size,
        ):
            await asyncio.Future()   # run forever

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
