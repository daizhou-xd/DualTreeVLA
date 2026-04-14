"""
LIBERO Dataset Loader (LeRobot parquet format) — CONSTRUCTION.md §6.1

Supports the HuggingFace LeRobot-format LIBERO dataset:
    git clone https://huggingface.co/datasets/shihao1895/libero-rlds
    (See §6.5 of CONSTRUCTION.md for download instructions)

Expected directory structure (LeRobot parquet):
    {root}/
        train/
            episode_XXXXXXXXX.parquet   (one file per episode)
        meta/
            stats.json                  (normalization statistics)

Each parquet row = one timestep with columns (standard LeRobot schema):
    observation.images.image  : bytes (JPEG/PNG encoded)
    observation.state         : list[float]
    action                    : list[float]
    language_instruction      : str

Backward-compatible fallback: if HDF5 files are found instead, they are
loaded via the legacy LIBEROHdf5Dataset class at the bottom of this file.

__getitem__ returns a dict matching RoboCerebra's interface:
    frames      : (T, 3, H, W) float32 [0,1]
    actions     : (T, d_a)     float32
    states      : (T, d_q)     float32
    instruction : str
"""
from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


# ================================================================
#  Image decoding
# ================================================================

def _decode_image_bytes(raw: bytes, h: int, w: int) -> np.ndarray:
    if _PIL_AVAILABLE:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), PILImage.BILINEAR)
        return np.array(img, dtype=np.uint8)
    elif _CV2_AVAILABLE:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        return img
    raise RuntimeError("PIL or cv2 required for image decoding.")


def _extract_image_raw(cell) -> bytes:
    """
    Extract JPEG/PNG bytes from lerobot v3 image cell.

    lerobot v3 stores images as dicts:  {"bytes": b"...", "path": None}
    Older formats store raw bytes directly.
    PIL Image objects (HuggingFace datasets library) are also handled.
    """
    if isinstance(cell, bytes):
        return cell
    if isinstance(cell, dict):
        return cell.get("bytes") or cell.get("data") or b""
    if _PIL_AVAILABLE and isinstance(cell, PILImage.Image):
        buf = io.BytesIO()
        cell.save(buf, format="JPEG")
        return buf.getvalue()
    return b""


def _load_video_frames(video_path: Path, h: int, w: int) -> np.ndarray:
    """
    Decode all frames from an mp4 video file.
    Returns uint8 array of shape (T, H, W, 3) in RGB order.
    Tries PyAV first, falls back to cv2.VideoCapture.
    """
    if _AV_AVAILABLE:
        frames = []
        with _av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                if img.shape[:2] != (h, w):
                    if _PIL_AVAILABLE:
                        img = np.array(
                            PILImage.fromarray(img).resize((w, h), PILImage.BILINEAR)
                        )
                    elif _CV2_AVAILABLE:
                        img = cv2.resize(img[:, :, ::-1], (w, h))[:, :, ::-1]
                frames.append(img)
        return np.stack(frames, 0).astype(np.uint8) if frames else np.zeros((0, h, w, 3), np.uint8)

    if _CV2_AVAILABLE:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            frames.append(frame)
        cap.release()
        return np.stack(frames, 0).astype(np.uint8) if frames else np.zeros((0, h, w, 3), np.uint8)

    raise RuntimeError(
        "PyAV or cv2 required for video decoding. "
        "Install with:  pip install av  or  pip install opencv-python"
    )


# ================================================================
#  Normalization statistics
# ================================================================

class _NormStats:
    def __init__(self, stats_path: Optional[str]):
        self.stats: Dict = {}
        if stats_path and os.path.isfile(stats_path):
            with open(stats_path) as f:
                self.stats = json.load(f)

    def normalize(self, key: str, arr: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            return arr
        mean = np.array(self.stats[key]["mean"], dtype=np.float32)
        std  = np.array(self.stats[key]["std"],  dtype=np.float32)
        std  = np.where(std < 1e-6, 1.0, std)
        return (arr - mean) / std


# ================================================================
#  Main parquet-based Dataset
# ================================================================

class LiberoDataset(Dataset):
    """
    LIBERO LeRobot parquet dataset (main training dataset for Phase 1+2).

    Supports two storage layouts:

    Layout A — Inline bytes (LeRobot v1 / custom format)
        {root}/train/*.parquet  or  {root}/*.parquet
        Each parquet row has an image column with JPEG/PNG bytes.

    Layout B — LeRobot v2 (official lerobot org datasets)
        {root}/data/chunk-XXX/episode_XXXXXX.parquet  (state + action + task_index)
        {root}/videos/chunk-XXX/{camera}/episode_XXXXXX.mp4 (image frames)
        {root}/meta/tasks.jsonl           (task_index → instruction)
        {root}/meta/stats.json            (normalization stats, optional)

    The format is detected automatically from the directory layout.
    Multiple dataset roots can be combined by passing `roots: list[str]`.

    Args
    ----
    root        : path to a single dataset root (mutually exclusive with roots)
    roots       : list of dataset roots to merge (e.g. all 4 LIBERO subsets)
    img_h/img_w : target frame resolution
    d_q         : target proprioceptive dim (pad/truncate)
    d_a         : target action dim
    max_seqlen  : truncate long episodes
    normalize   : apply stats.json normalization
    """

    def __init__(
        self,
        root: Optional[str]        = None,
        roots: Optional[List[str]] = None,
        img_h: int                 = 224,
        img_w: int                 = 224,
        d_q: int                   = 84,
        d_a: int                   = 7,
        H_a: int                   = 16,     # action horizon — defines per-step action chunk size
        max_seqlen: Optional[int]  = None,   # kept for backward-compat; ignored when step_level=True
        normalize: bool            = True,
        step_level: bool           = True,   # Evo-1 style: one sample per timestep, no truncation
    ):
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas + pyarrow required: pip install pandas pyarrow"
            )
        if root is None and roots is None:
            raise ValueError("Provide either root or roots.")
        if root is not None and roots is not None:
            raise ValueError("Provide either root or roots, not both.")

        self.img_h      = img_h
        self.img_w      = img_w
        self.d_q        = d_q
        self.d_a        = d_a
        self.H_a        = H_a
        self.max_seqlen = max_seqlen
        self.step_level = step_level
        self._step_records: List[Tuple[int, int]] = []  # (record_idx, frame_t)
        # LRU episode cache: keys are evicted when size exceeds _MAX_CACHE_EPISODES.
        # Each worker process has its own copy; limit prevents OOM with many workers.
        # 64 episodes × ~3 GB worst-case JPEG bytes per worker is acceptable.
        self._MAX_CACHE_EPISODES: int = 64
        self._episode_cache: OrderedDict = OrderedDict()  # record_idx → cached dict

        all_roots = [Path(root)] if root is not None else [Path(r) for r in roots]  # type: ignore[arg-type]

        # Merge episodes from all roots into a flat list.
        # Each entry: (record_data, root_path, layout)
        # layout "A" = inline bytes, one parquet per episode
        # layout "B" = lerobot v2, per-episode parquet + video files
        # layout "C" = lerobot v3, per-chunk parquet, images inline as dict
        self._records: List[Tuple] = []
        self._tasks_map: Dict[str, Dict[int, str]] = {}  # root_str → {task_idx: task}
        self._norms: Dict[str, _NormStats] = {}

        for root_p in all_roots:
            # ── Detect layout ───────────────────────────────────────
            info_path = root_p / "meta" / "info.json"
            if info_path.exists():
                _info = json.loads(info_path.read_text())
                _tmpl = _info.get("data_path", "")
                if "chunk_index" in _tmpl and "file_index" in _tmpl:
                    layout: str = "C"   # lerobot v3 per-chunk
                else:
                    layout = "B"        # lerobot v2 per-episode
            else:
                layout = "A"            # custom inline-bytes format

            norm = _NormStats(str(root_p / "meta" / "stats.json") if normalize else None)
            rk   = str(root_p)
            self._norms[rk]     = norm
            self._tasks_map[rk] = self._load_tasks(root_p)

            if layout == "C":
                # lerobot v3: group parquet rows by episode_index
                data_dir = root_p / "data"
                all_pqs  = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
                if not all_pqs:
                    raise FileNotFoundError(
                        f"No .parquet files found in {root_p}/data.\n"
                        "Download LIBERO datasets:  python scripts/download_data.py --libero"
                    )
                # Read only episode_index column for fast indexing
                # Also count frames per episode for step_level index building
                ep_to_files: Dict[int, List[Path]] = {}
                ep_to_count: Dict[int, int]        = {}
                for pf in all_pqs:
                    _edf = pd.read_parquet(pf, columns=["episode_index"])
                    for _ep in _edf["episode_index"].unique():
                        ep_to_files.setdefault(int(_ep), []).append(pf)
                    for _ep, cnt in _edf["episode_index"].value_counts().items():
                        ep_to_count[int(_ep)] = ep_to_count.get(int(_ep), 0) + int(cnt)
                for ep_idx in sorted(ep_to_files.keys()):
                    self._records.append((
                        {"ep_idx": ep_idx, "files": ep_to_files[ep_idx]},
                        root_p, "C",
                    ))
                    if self.step_level:
                        ri = len(self._records) - 1
                        ep_len = ep_to_count.get(ep_idx, 0)
                        for t in range(max(0, ep_len - self.H_a + 1)):
                            self._step_records.append((ri, t))

            elif layout == "B":
                data_dir = root_p / "data"
                eps = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
                if not eps:
                    raise FileNotFoundError(
                        f"No .parquet files found in {root_p}/data.\n"
                        "Download LIBERO datasets:  python scripts/download_data.py --libero"
                    )
                for ep in eps:
                    self._records.append((ep, root_p, "B"))
                    if self.step_level:
                        ri = len(self._records) - 1
                        try:
                            _ep_len = len(pd.read_parquet(ep, columns=["frame_index"]))
                        except Exception:
                            _ep_len = 0
                        for t in range(max(0, _ep_len - self.H_a + 1)):
                            self._step_records.append((ri, t))

            else:
                # Layout A: try train/, then data/, then root itself
                eps_a: List[Path] = []
                for sub in ("train", "data", ""):
                    search_dir = root_p / sub if sub else root_p
                    eps_a = sorted(search_dir.rglob("*.parquet")) if search_dir.is_dir() else []
                    if eps_a:
                        break
                if not eps_a:
                    raise FileNotFoundError(
                        f"No .parquet files found in {root_p}.\n"
                        "Download LIBERO datasets:  python scripts/download_data.py --libero"
                    )
                for ep in eps_a:
                    self._records.append((ep, root_p, "A"))
                    if self.step_level:
                        ri = len(self._records) - 1
                        try:
                            _ep_len = len(pd.read_parquet(ep, columns=["frame_index"]))
                        except Exception:
                            _ep_len = 0
                        for t in range(max(0, _ep_len - self.H_a + 1)):
                            self._step_records.append((ri, t))

        if self.step_level:
            print(f"LiberoDataset [step_level]: {len(self._step_records):,} step samples "
                  f"from {len(self._records)} episodes (H_a={self.H_a})")

    # ─── Task metadata ──────────────────────────────────────────────

    @staticmethod
    def _load_tasks(root_p: Path) -> Dict[int, str]:
        tasks: Dict[int, str] = {}
        tasks_file = root_p / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # lerobot v2: {"task_index": 0, "task": "..."}  or  {"index": 0, "task": "..."}
                    idx  = entry.get("task_index", entry.get("index", None))
                    text = entry.get("task", entry.get("instruction", ""))
                    if idx is not None:
                        tasks[int(idx)] = str(text)
        return tasks

    # ─── Video frame loader (lerobot v2) ────────────────────────────

    def _load_frames_from_video(self, parquet_path: Path, root_p: Path) -> np.ndarray:
        """
        Locate and decode the episode video corresponding to a parquet file.

        lerobot v2 path pattern:
            parquet: {root}/data/chunk-{C:03d}/episode_{E:06d}.parquet
            video:   {root}/videos/chunk-{C:03d}/{camera}/episode_{E:06d}.mp4

        Falls back to a directory scan if the expected path does not exist.
        """
        # Derive chunk and episode stem from parquet path
        episode_stem = parquet_path.stem          # e.g. "episode_000017"
        chunk_folder = parquet_path.parent.name   # e.g. "chunk-000"

        videos_root = root_p / "videos"
        if not videos_root.is_dir():
            return np.zeros((0, self.img_h, self.img_w, 3), np.uint8)

        chunk_video_dir = videos_root / chunk_folder
        if not chunk_video_dir.is_dir():
            # Try any chunk directory (some datasets use flat structure)
            chunk_video_dir = videos_root

        # Find the first camera directory that has the episode mp4
        video_path: Optional[Path] = None
        for cam_dir in sorted(chunk_video_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            candidate = cam_dir / f"{episode_stem}.mp4"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            # Fallback: search anywhere under videos_root
            candidates = list(videos_root.rglob(f"{episode_stem}.mp4"))
            video_path = candidates[0] if candidates else None

        if video_path is None:
            return np.zeros((0, self.img_h, self.img_w, 3), np.uint8)

        return _load_video_frames(video_path, self.img_h, self.img_w)

    # ─── Core API ───────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.step_level:
            return len(self._step_records)
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        """
        step_level=True  (Evo-1 style, default):
            Returns one timestep: single frame + next H_a actions.
            _step_records maps idx -> (record_idx, frame_t).

        step_level=False  (episode mode):
            Returns a full episode sequence (T, ...). Used for visualization.
        """
        if self.step_level:
            record_idx, frame_t = self._step_records[idx]
        else:
            record_idx = idx
            frame_t    = None

        return self._load_episode(record_idx, frame_t)

    def load_episode(self, record_idx: int) -> Dict:
        """Always return the full episode (used by visualize_epoch)."""
        return self._load_episode(record_idx, frame_t=None)

    def _load_episode(self, record_idx: int, frame_t: Optional[int]) -> Dict:
        """
        Core loading logic with episode-level cache.

        First access to each episode: reads parquet once, stores JPEG bytes +
        actions/states in self._episode_cache (per worker process in RAM).
        Subsequent accesses (same worker, same epoch or later epochs): dict lookup only.

        For step_level (frame_t is not None): decodes only ONE JPEG frame instead of
        all T frames in the episode  — the main speedup vs the old implementation.
        """
        # ── Episode-level cache lookup (LRU) ──────────────────────────────
        if record_idx in self._episode_cache:
            # Move to end (most-recently-used)
            self._episode_cache.move_to_end(record_idx)
        else:
            self._episode_cache[record_idx] = self._cache_episode(record_idx)
            self._episode_cache.move_to_end(record_idx)
            # Evict least-recently-used when over limit
            while len(self._episode_cache) > self._MAX_CACHE_EPISODES:
                self._episode_cache.popitem(last=False)
        cached = self._episode_cache[record_idx]

        img_bytes: List[bytes] = cached["img_bytes"]
        a_raw: np.ndarray      = cached["a_raw"]    # (T, d_a) float32
        s_raw: np.ndarray      = cached["s_raw"]    # (T, d_q) float32
        instruction: str       = cached["instruction"]
        T = len(a_raw)

        # ── Step-level return (Evo-1 style) ─────────────────────────
        if frame_t is not None:
            # Decode only ONE frame — O(1) JPEG decode instead of O(T)
            if img_bytes and frame_t < len(img_bytes):
                frm_np = _decode_image_bytes(img_bytes[frame_t], self.img_h, self.img_w)
            else:
                frm_np = np.zeros((self.img_h, self.img_w, 3), np.uint8)
            fr_t = torch.from_numpy(frm_np.astype(np.float32) / 255.0).permute(2, 0, 1)

            t_end   = min(frame_t + self.H_a, T)
            a_chunk = torch.from_numpy(a_raw[frame_t:t_end].copy())
            if a_chunk.shape[0] < self.H_a:
                a_chunk = torch.cat(
                    [a_chunk, a_chunk[-1:].expand(self.H_a - a_chunk.shape[0], -1)]
                )
            return {
                "frames":      fr_t.unsqueeze(0),                         # (1, 3, H, W)
                "actions":     a_chunk,                                   # (H_a, d_a)
                "states":      torch.from_numpy(s_raw[frame_t].copy()).unsqueeze(0),  # (1, d_q)
                "instruction": instruction,
                "episode_id":  int(record_idx),
                "frame_idx":   int(frame_t),
            }

        # ── Episode-level return (viz / backward-compat) ────────────
        if img_bytes:
            frames_np = np.stack(
                [_decode_image_bytes(b, self.img_h, self.img_w) for b in img_bytes]
            )
        else:
            frames_np = np.zeros((T, self.img_h, self.img_w, 3), np.uint8)
        fr = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        ac = torch.from_numpy(a_raw[:T].copy())
        st = torch.from_numpy(s_raw[:T].copy())
        return {"frames": fr, "actions": ac, "states": st, "instruction": instruction}

    def _cache_episode(self, record_idx: int) -> Dict:
        """
        Load episode from disk ONCE and return a cacheable dict:
            img_bytes : List[bytes]  — one JPEG blob per frame (NOT decoded)
            a_raw     : np.ndarray   — (T, d_a) float32, normalized
            s_raw     : np.ndarray   — (T, d_q) float32, normalized
            instruction: str

        Storing JPEG bytes (not decoded uint8) keeps RAM usage ~50× lower:
        379 episodes × 277 frames × ~30 KB/JPEG ≈ 3 GB  vs  ~14 GB decoded.
        """
        record, root_p, layout = self._records[record_idx]
        rk   = str(root_p)
        norm = self._norms[rk]

        # ── Load DataFrame ──────────────────────────────────────────
        if layout == "C":
            ep_idx: int = record["ep_idx"]
            # Pass filters= so pyarrow skips row-groups for other episodes
            dfs = [
                pd.read_parquet(f, filters=[("episode_index", "==", ep_idx)])
                for f in record["files"]
            ]
            df = pd.concat(dfs, ignore_index=True).sort_values(
                "frame_index", ignore_index=True
            )
        else:
            df = pd.read_parquet(record)   # record is Path for A/B

        # ── Actions ─────────────────────────────────────────────────
        a_col = self._col(df, ["action", "actions"])
        if a_col:
            a_raw = np.stack(df[a_col].tolist(), 0).astype(np.float32)
            a_raw = norm.normalize("action", a_raw)
        else:
            a_raw = np.zeros((len(df), self.d_a), np.float32)
        a_raw = self._pad_dim(a_raw, self.d_a)

        # ── States ──────────────────────────────────────────────────
        s_col = self._col(df, ["observation.state", "state"])
        if s_col:
            s_raw = np.stack(df[s_col].tolist(), 0).astype(np.float32)
            s_raw = norm.normalize("observation.state", s_raw)
        else:
            s_raw = np.zeros((len(df), self.d_q), np.float32)
        s_raw = self._pad_dim(s_raw, self.d_q)

        # ── Language instruction ─────────────────────────────────────
        instruction = ""
        if layout in ("B", "C"):
            ti_col = self._col(df, ["task_index", "task_idx"])
            if ti_col is not None:
                task_idx = int(df[ti_col].iloc[0])
                instruction = self._tasks_map[rk].get(task_idx, "")
            if not instruction:
                i_col = self._col(df, ["language_instruction", "instruction", "task"])
                instruction = str(df[i_col].iloc[0]) if i_col else ""
        else:
            i_col = self._col(df, ["language_instruction", "instruction", "task"])
            instruction = str(df[i_col].iloc[0]) if i_col else ""

        # ── Image bytes (store compressed; decode per-step in _load_episode) ─
        _im_candidates = ["observation.images.image", "observation.image", "image"]
        img_bytes: List[bytes] = []
        if layout == "B":
            # lerobot v2: decode from video, then re-compress to JPEG bytes
            frames_np = self._load_frames_from_video(record, root_p)
            if frames_np.shape[0] > 0:
                if _PIL_AVAILABLE:
                    for frm in frames_np:
                        buf = io.BytesIO()
                        PILImage.fromarray(frm).save(buf, format="JPEG", quality=85)
                        img_bytes.append(buf.getvalue())
                else:
                    # cv2 fallback: store raw RGB bytes (larger but avoids PIL dep)
                    img_bytes = [frm.tobytes() for frm in frames_np]
            else:
                im_col = self._col(df, _im_candidates)
                if im_col:
                    img_bytes = [_extract_image_raw(r) for r in df[im_col].tolist()]
        else:
            # Layout A / C: images already inline as JPEG bytes
            im_col = self._col(df, _im_candidates)
            if im_col:
                img_bytes = [_extract_image_raw(r) for r in df[im_col].tolist()]

        return {
            "img_bytes":   img_bytes,   # List[bytes], one JPEG blob per frame
            "a_raw":       a_raw,       # (T, d_a) float32
            "s_raw":       s_raw,       # (T, d_q) float32
            "instruction": instruction,
        }

    # ─── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _col(df: "pd.DataFrame", candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        for c in candidates:
            for col in df.columns:
                if c in col:
                    return col
        return None

    @staticmethod
    def _pad_dim(arr: np.ndarray, target: int) -> np.ndarray:
        d = arr.shape[1]
        if d < target:
            arr = np.pad(arr, ((0, 0), (0, target - d)))
        else:
            arr = arr[:, :target]
        return arr


# ================================================================
#  Collate function
# ================================================================

def libero_collate(batch: List[Dict]) -> Dict:
    """
    Pad batch to equal length.

    step_level mode : frames=(1,C,H,W), actions=(H_a,d_a), states=(1,d_q)
    episode mode    : frames=(T,C,H,W), actions=(T,d_a),   states=(T,d_q)

    We pad frames/states to T_frames_max and actions to T_actions_max
    independently, so both modes work with the same collate.
    """
    T_frames  = max(b["frames"].shape[0]  for b in batch)
    T_actions = max(b["actions"].shape[0] for b in batch)
    B         = len(batch)
    C, H, W   = batch[0]["frames"].shape[1:]
    d_a       = batch[0]["actions"].shape[1]
    d_q       = batch[0]["states"].shape[1]

    frames  = torch.zeros(B, T_frames,  C, H, W)
    actions = torch.zeros(B, T_actions, d_a)
    states  = torch.zeros(B, T_frames,  d_q)
    episode_ids = torch.full((B,), -1, dtype=torch.long)
    frame_indices = torch.full((B,), -1, dtype=torch.long)
    instructions: List[str] = []

    for i, s in enumerate(batch):
        Tf = s["frames"].shape[0]
        Ta = s["actions"].shape[0]
        frames[i, :Tf]  = s["frames"]
        actions[i, :Ta] = s["actions"]
        states[i, :Tf]  = s["states"]
        if Tf < T_frames:
            frames[i, Tf:] = s["frames"][-1:].expand(T_frames - Tf, -1, -1, -1)
        if "episode_id" in s:
            episode_ids[i] = int(s["episode_id"])
        if "frame_idx" in s:
            frame_indices[i] = int(s["frame_idx"])
        instructions.append(s["instruction"])

    return {
        "frames": frames,
        "actions": actions,
        "states": states,
        "instructions": instructions,
        "episode_ids": episode_ids,
        "frame_indices": frame_indices,
    }
