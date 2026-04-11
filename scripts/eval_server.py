"""
DualTreeVLA — WebSocket Inference Server
=========================================

Loads the trained DualTreeVLA model and serves action predictions over a
WebSocket connection.  The paired simulation client is ``eval_client.py``.

Protocol (JSON over WebSocket)
-------------------------------
Client → Server:
  {"type": "reset"}
      Reset the hierarchical memory tree for a new episode.
      Server replies: {"status": "ok"}

  {"type": "infer",
   "image":       [H×W×3 uint8 flat list],   # agentview, already flipped
   "state":       [8 floats, raw physical],
   "instruction": "<task string>"}
      Run one model.step() call.
      Server replies: {"actions": [[7 floats raw] × H_a]}
      Actions are already denormalized to physical scale.

Usage
-----
  # On the GPU server:
  python scripts/eval_server.py \\
      --ckpt  checkpoints/runs/phase2/phase2_best.pt \\
      --config configs/train_phase2.yaml \\
      --stats dataset/datasets/libero_10/meta/stats.json \\
      --port  9000

  # Then launch the client in another terminal:
  python scripts/eval_client.py --server ws://127.0.0.1:9000 ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# ── Project root ─────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

import websockets  # pip install websockets>=12.0


# ================================================================
#  Action normalization  (z-score, same as LiberoDataset)
# ================================================================

class ActionNorm:
    """Load stats.json and expose normalize_state / denormalize_action."""

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
            sk = "observation.state" if "observation.state" in stats else "state"
            if sk in stats:
                self._s_mean = np.array(stats[sk]["mean"], dtype=np.float32)
                self._s_std  = np.array(stats[sk]["std"],  dtype=np.float32)
                self._s_std  = np.where(self._s_std < 1e-6, 1.0, self._s_std)
            print(f"[Server] Loaded norm stats: {stats_path}")
        else:
            print("[Server][WARN] No stats.json — actions will NOT be denormalized. "
                  "Pass --stats to fix.")

    @property
    def available(self) -> bool:
        return self._a_mean is not None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self._s_mean is None:
            return state
        d = min(len(state), len(self._s_mean))
        out = state.copy()
        out[:d] = (state[:d] - self._s_mean[:d]) / self._s_std[:d]
        return out

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._a_mean is None:
            return action
        d = min(len(action), len(self._a_mean))
        out = action.copy()
        out[:d] = action[:d] * self._a_std[:d] + self._a_mean[:d]
        return out


# ================================================================
#  Model loading
# ================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    from dual_tree_vla.model import DualTreeVLA

    m = cfg.get("model", {})
    model = DualTreeVLA(
        llm_path        = m.get("llm_path",        "checkpoints/Qwen2.5-0.5B"),
        clip_model_name = m.get("clip_model_name", None),
        d           = m.get("d",           256),
        d_a         = m.get("d_a",         7),
        d_q         = m.get("d_q",         8),
        d_visual    = m.get("d_visual",    256),
        d_ssm       = m.get("d_ssm",       256),
        d_state     = m.get("d_state",     16),
        patch_size  = m.get("patch_size",  16),
        H_a         = m.get("H_a",         16),
        n_ode       = m.get("n_ode",       20),
        theta_fuse  = m.get("theta_fuse",  0.35),
        K_elev      = m.get("K_elev",      4),
        delta_w     = m.get("delta_w",     0.1),
        tau         = m.get("tau",         0.1),
        freeze_llm  = False,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    sd = (
        state.get("model")
        or state.get("model_state_dict")
        or state.get("module")
        or state
    )
    model_sd = model.state_dict()
    sd_clean = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = len(sd) - len(sd_clean)
    if skipped:
        print(f"[Server][WARN] Skipped {skipped} shape-mismatched keys")

    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"[Server][WARN] Missing ({len(missing)}): {missing[:4]}")
    if unexpected:
        print(f"[Server][WARN] Unexpected ({len(unexpected)}): {unexpected[:4]}")

    model.to(device).eval()
    print(f"[Server] Model loaded on {device}. "
          f"sem_proj norm={sum(p.norm().item() for p in model.sem_proj.parameters()):.3f}")
    return model


# ================================================================
#  Language caching (per server lifetime)
# ================================================================

def patch_lang_cache(model):
    """Memoize _encode_language to avoid repeated LLM forward passes."""
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
#  WebSocket handler
# ================================================================

async def handle_client(
    websocket,
    model,
    action_norm: ActionNorm,
    device: torch.device,
    img_size: int,
    d_q: int,
):
    """Handle one WebSocket connection (one evaluation run)."""
    import cv2

    print("[Server] Client connected")
    try:
        async for raw_msg in websocket:
            data = json.loads(raw_msg)
            msg_type = data.get("type", "infer")

            # ── Reset ─────────────────────────────────────────────────
            if msg_type == "reset":
                model.reset_trees(batch_size=1)
                await websocket.send(json.dumps({"status": "ok"}))
                print("[Server] Trees reset for new episode")
                continue

            # ── Infer ─────────────────────────────────────────────────
            # Decode image: client sends (H, W, 3) uint8 as nested list
            img_arr = np.array(data["image"], dtype=np.uint8)   # (H, W, 3) RGB
            if img_arr.shape[0] != img_size or img_arr.shape[1] != img_size:
                img_arr = cv2.resize(img_arr, (img_size, img_size))
            # [0,1] float tensor — CLIP normalization is handled inside
            # CLIPPatchExtractor.forward() so we just scale to [0,1]
            img_t = (
                torch.from_numpy(img_arr.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )  # (1, 3, H, W)

            # Decode state: raw physical values (8-dim for LIBERO)
            state_raw = np.array(data["state"], dtype=np.float32)
            if len(state_raw) < d_q:
                state_raw = np.pad(state_raw, (0, d_q - len(state_raw)))
            else:
                state_raw = state_raw[:d_q]
            state_norm = action_norm.normalize_state(state_raw)
            state_t = torch.from_numpy(state_norm).unsqueeze(0).to(device)  # (1, d_q)

            instruction = data["instruction"]

            # Model inference
            a_chunk = model.step(img_t, instruction, state_t)   # (1, H_a, d_a)
            a_np = a_chunk[0].cpu().float().numpy()             # (H_a, d_a)

            # Denormalize
            actions_out = []
            for h in range(a_np.shape[0]):
                a_raw = action_norm.denormalize_action(a_np[h].copy())
                actions_out.append(a_raw.tolist())

            await websocket.send(json.dumps({"actions": actions_out}))

    except websockets.exceptions.ConnectionClosed:
        print("[Server] Client disconnected")
    except Exception as e:
        import traceback
        print(f"[Server][ERROR] {e}")
        traceback.print_exc()


# ================================================================
#  Entry point
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DualTreeVLA WebSocket inference server")
    p.add_argument("--ckpt",   required=True,  help="Path to .pt checkpoint")
    p.add_argument("--config", default="configs/train_phase2.yaml",
                   help="Training config YAML")
    p.add_argument("--stats",  default=None,
                   help="Path to stats.json for z-score de-normalization. "
                        "Auto-detected from config if omitted.")
    p.add_argument("--port",   type=int, default=9000, help="WebSocket port")
    p.add_argument("--host",   default="0.0.0.0",      help="Bind address")
    p.add_argument("--img_size", type=int, default=224, help="Model input resolution")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _find_stats(cfg: dict, explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    # Auto-detect from data.root in config
    data_root = cfg.get("data", {}).get("root", "")
    cands = [
        os.path.join(str(_PROJ_ROOT), data_root, "meta", "stats.json"),
        os.path.join(str(_PROJ_ROOT), data_root, "stats.json"),
        os.path.join(data_root, "meta", "stats.json"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"[Server] Loading config: {args.config}")
    cfg = load_config(args.config)
    d_q = cfg.get("model", {}).get("d_q", 8)

    print(f"[Server] Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, cfg, device)
    patch_lang_cache(model)

    stats_path  = _find_stats(cfg, args.stats)
    action_norm = ActionNorm(stats_path)

    async def serve():
        handler = lambda ws: handle_client(
            ws, model, action_norm, device, args.img_size, d_q
        )
        async with websockets.serve(
            handler, args.host, args.port,
            max_size=50_000_000,    # 50 MB — enough for 224×224 images
        ):
            print(f"[Server] Listening on ws://{args.host}:{args.port}")
            print("[Server] Waiting for eval_client.py …")
            await asyncio.Future()  # run forever

    asyncio.run(serve())


if __name__ == "__main__":
    main()
