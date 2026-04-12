# DualTreeVLA — WebSocket Inference Server
# Based on Evo-1's Evo1_server.py
#
# Usage:
#   python scripts/eval_server.py
#   (edit ckpt_dir / config_path / port below)

import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_tree_vla.model import DualTreeVLA


# ========= Normalizer (z-score, matches LiberoDataset) =========
class Normalizer:
    def __init__(self, stats_path):
        self._a_mean = self._a_std = self._s_mean = self._s_std = None
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
            print(f"Loaded norm stats: {stats_path}")
        else:
            print("⚠️  No stats.json found — actions will NOT be denormalized.")

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


# ========= Model loading =========
def load_model_and_normalizer(ckpt_path, config_path, stats_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

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
    ).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = (
        ckpt.get("model")
        or ckpt.get("model_state_dict")
        or ckpt.get("module")
        or ckpt
    )
    model_sd = model.state_dict()
    sd_clean = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = len(sd) - len(sd_clean)
    if skipped:
        print(f"⚠️  Skipped {skipped} shape-mismatched checkpoint keys")
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"⚠️  Missing  ({len(missing)}): {missing[:4]}")
    if unexpected:
        print(f"⚠️  Unexpected ({len(unexpected)}): {unexpected[:4]}")

    model = model.to("cuda")

    # Resolve stats.json path
    if stats_override and os.path.isfile(stats_override):
        stats_path = stats_override
    else:
        data_root = cfg.get("data", {}).get("root", "")
        proj = Path(__file__).parent.parent
        stats_path = None
        for cand in [
            proj / data_root / "meta" / "stats.json",
            proj / data_root / "stats.json",
            Path(data_root) / "meta" / "stats.json",
        ]:
            if cand.is_file():
                stats_path = str(cand)
                break

    normalizer = Normalizer(stats_path)
    d_q = m.get("d_q", 8)
    return model, normalizer, d_q


# ========= Language cache (avoid repeated LLM forward) =========
def patch_lang_cache(model):
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


# ========= Decode image from JSON list =========
def decode_image_from_list(img_list, img_size=224):
    img_array = np.array(img_list, dtype=np.uint8)   # (H, W, 3) RGB
    if img_array.shape[0] != img_size or img_array.shape[1] != img_size:
        img_array = cv2.resize(img_array, (img_size, img_size))
    img_t = (
        torch.from_numpy(img_array.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda")
    )   # (1, 3, H, W)
    return img_t


# ========= Inference from JSON dict =========
def infer_from_json_dict(data: dict, model, normalizer, d_q: int):
    img_t   = decode_image_from_list(data["image"])

    state_raw = np.array(data["state"], dtype=np.float32)
    if len(state_raw) < d_q:
        state_raw = np.pad(state_raw, (0, d_q - len(state_raw)))
    else:
        state_raw = state_raw[:d_q]
    state_norm = normalizer.normalize_state(state_raw)
    state_t = torch.from_numpy(state_norm).unsqueeze(0).to("cuda")   # (1, d_q)

    prompt = data["prompt"]

    with torch.no_grad():
        a_chunk = model.step(img_t, prompt, state_t)   # (1, H_a, d_a)
    a_np = a_chunk[0].cpu().float().numpy()            # (H_a, d_a)

    # Debug: print raw z-score vs denormalized action for first slot
    a0_denorm = normalizer.denormalize_action(a_np[0].copy())
    print(f"  [dbg] raw z-score a[0]: {np.round(a_np[0], 3).tolist()}")
    print(f"  [dbg] denormed   a[0]: {np.round(a0_denorm, 3).tolist()}")

    actions_out = []
    for h in range(a_np.shape[0]):
        a_raw = normalizer.denormalize_action(a_np[h].copy())
        actions_out.append(a_raw.tolist())

    return actions_out


# ========= WebSocket handler =========
async def handle_request(websocket, model, normalizer, d_q):
    print("Client connected")
    try:
        async for message in websocket:
            json_data = json.loads(message)

            # Reset HMT trees for new episode
            if json_data.get("type") == "reset":
                model.reset_trees(batch_size=1)
                await websocket.send(json.dumps({"status": "ok"}))
                print("Trees reset for new episode")
                continue

            # Inference
            print("Received observation")
            actions = infer_from_json_dict(json_data, model, normalizer, d_q)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


# ================================================================
#  Entry point  (edit these paths or pass --ckpt --config --port)
# ================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   default="checkpoints/runs/phase2/phase2_best.pt")
    p.add_argument("--config", default="configs/train_phase2.yaml")
    p.add_argument("--stats",  default=None,
                   help="Path to stats.json.  If omitted, auto-detected from config's data.root.")
    p.add_argument("--port",   type=int, default=9000)
    p.add_argument("--host",   default="0.0.0.0")
    args = p.parse_args()

    port = args.port

    print("Loading DualTreeVLA model...")
    model, normalizer, d_q = load_model_and_normalizer(args.ckpt, args.config, args.stats)
    patch_lang_cache(model)

    async def main():
        print(f"DualTreeVLA server running at ws://{args.host}:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, d_q),
            args.host, port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
