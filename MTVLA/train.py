"""
train.py — Three-stage training entry point for MemoryTreeVLA.

Three-stage training paradigm (CONSTRUCTION.md §4):

  Stage 1  Tree LLM + Multimodal Mamba pre-training
           Dataset : RoboCerebra_trainset  (fixed — tree annotation required)
           Trains  : tree_llm, tree_llm_proj, multimodal_mamba
           Freezes : vision_mamba, task_tree_mamba, action_llm, action_head

  Stage 2  Fusion module + Action head joint training
           Dataset : configurable — LIBERO / CALVIN / RoboMME (see YAML)
           Trains  : multimodal_mamba, action_llm_proj, action_condition,
                     action_head
           Freezes : vision_mamba, task_tree_mamba, action_llm, tree_llm

  Stage 3  End-to-end fine-tuning
           Dataset : configurable — same pool as Stage 2
           Trains  : action_llm (lr × 0.1), multimodal_mamba,
                     action_condition, action_head
           Freezes : vision_mamba

Usage examples:

  # Full 3-stage run
  python MTVLA/train.py --config MTVLA/configs/default.yaml

  # Multi-GPU with DeepSpeed
  deepspeed --num_gpus=4 MTVLA/train.py \\
      --config MTVLA/configs/default.yaml \\
      --deepspeed MTVLA/configs/ds_config.json

  # Skip Stage 1 (already trained), start from Stage 2 with a checkpoint
  python MTVLA/train.py \\
      --config MTVLA/configs/default.yaml \\
      --start_stage 2 \\
      --stage2_ckpt outputs/stage1/best.pth

  # Only run Stage 3 fine-tuning
  python MTVLA/train.py \\
      --config MTVLA/configs/default.yaml \\
      --start_stage 3 \\
      --stage3_ckpt outputs/stage2/best.pth \\
      --end_stage 3
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Project imports — resolve relative to repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MTVLA.configs.config import (
    DatasetConfig,
    MTVLAConfig,
    StageTrainConfig,
    TrainingPipelineConfig,
    load_config,
)
from MTVLA.models import MemoryTreeVLA
from MTVLA.utils.logger import setup_logger
from MTVLA.utils.metrics import AverageMeter

logger = setup_logger("MTVLA-Train")


# ---------------------------------------------------------------------------
# Dataset stubs
# Each real integration should implement a proper Dataset class that:
#   - reads task_description.json (RoboCerebra) or equivalent task definitions
#   - loads observations from demo.hdf5 / LIBERO / CALVIN / RoboMME APIs
#   - returns the dict expected by _batch_to_device()
# ---------------------------------------------------------------------------

class _DummyEpisodeDataset(Dataset):
    """Placeholder dataset used when the real benchmark is not installed.

    Returns random tensors with the correct shapes so the training loop
    can be exercised without any benchmark installed.
    """

    def __init__(self, cfg: DatasetConfig, model_cfg) -> None:
        self.n_samples = 64
        self.image_size = cfg.image_size
        self.state_dim  = model_cfg.state_dim
        self.action_dim = model_cfg.action_dim
        self.horizon    = model_cfg.action_horizon
        self.node_vocab = model_cfg.node_vocab_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        N = random.randint(3, 9)   # number of tree nodes
        return {
            "image":    torch.randn(3, self.image_size, self.image_size),
            "state":    torch.randn(self.state_dim),
            "actions":  torch.randn(self.horizon, self.action_dim),
            "node_ids": torch.randint(0, self.node_vocab, (N,)),
            # parent_map is per-episode and handled as metadata outside the tensor batch
        }


class RoboCerebraDataset(Dataset):
    """RoboCerebra training dataset.

    Structure expected on disk:
        <data_root>/<scene>/<case_id>/
            task_description.json
            demo.hdf5

    Replace the placeholder body below with your actual HDF5 loader.
    """

    def __init__(self, cfg: DatasetConfig, model_cfg) -> None:
        from MTVLA.models.memory_tree import MemoryTree
        import glob, json

        self.cfg        = cfg
        self.model_cfg  = model_cfg
        self.image_size = cfg.image_size

        pattern = str(Path(cfg.data_root) / "*" / "*" / "task_description.json")
        self.json_paths = sorted(glob.glob(pattern))
        if not self.json_paths:
            logger.warning(
                "RoboCerebraDataset: no task_description.json found under %s. "
                "Using dummy data.", cfg.data_root
            )
        logger.info("RoboCerebraDataset: %d episodes found.", len(self.json_paths))

    def __len__(self) -> int:
        return max(len(self.json_paths), 1)

    def __getitem__(self, idx: int) -> Dict:
        if not self.json_paths:
            return _DummyEpisodeDataset(self.cfg, self.model_cfg)[0]

        from MTVLA.models.memory_tree import MemoryTree

        json_path = self.json_paths[idx % len(self.json_paths)]
        tree = MemoryTree.from_robocerebra_file(json_path)
        hdf5_path = Path(json_path).parent / "demo.hdf5"

        # --- Load a random subsequence from demo.hdf5 ---
        # Adapt this block to your actual HDF5 schema.
        try:
            import h5py  # optional dependency
            with h5py.File(hdf5_path, "r") as f:
                # Expected keys (iGibson / OmniGibson convention):
                #   "observations/rgb"   : (T, H, W, 3)  uint8
                #   "actions"            : (T, action_dim)
                #   "robot_state"        : (T, state_dim)
                rgb    = torch.from_numpy(f["observations/rgb"][:].astype("float32") / 255.0)
                acts   = torch.from_numpy(f["actions"][:].astype("float32"))
                state  = torch.from_numpy(f["robot_state"][:].astype("float32"))
        except Exception:
            # Fall back to dummy data if HDF5 is missing or malformed
            return _DummyEpisodeDataset(self.cfg, self.model_cfg)[0]

        T  = rgb.shape[0]
        H  = self.model_cfg.action_horizon
        t0 = random.randint(0, max(0, T - H - 1))

        image  = rgb[t0].permute(2, 0, 1)             # (3, H, W)
        # Simple resize; swap for transforms.Resize in a real loader
        if image.shape[-1] != self.image_size:
            image = nn.functional.interpolate(
                image.unsqueeze(0), size=self.image_size, mode="bilinear",
                align_corners=False
            ).squeeze(0)

        action_chunk = acts[t0 : t0 + H]              # (H, action_dim)
        if action_chunk.shape[0] < H:
            action_chunk = torch.cat([
                action_chunk,
                action_chunk[-1:].expand(H - action_chunk.shape[0], -1)
            ], 0)

        s = state[t0]
        if s.shape[0] < self.model_cfg.state_dim:
            s = torch.cat([s, torch.zeros(self.model_cfg.state_dim - s.shape[0])])
        elif s.shape[0] > self.model_cfg.state_dim:
            s = s[:self.model_cfg.state_dim]

        N        = len(tree)
        node_ids = torch.arange(N, dtype=torch.long)

        return {
            "image":       image,
            "state":       s,
            "actions":     action_chunk,
            "node_ids":    node_ids,
            "parent_map":  tree.to_parent_map(),   # Dict[str, Optional[str]]
            "tree_text":   str(tree),
        }


class LiberoDataset(_DummyEpisodeDataset):
    """LIBERO dataset stub.  Replace body with actual libero.envs loader."""


class CalvinDataset(_DummyEpisodeDataset):
    """CALVIN dataset stub.  Replace body with actual calvin_agent loader."""


class RoboMMEDataset(_DummyEpisodeDataset):
    """RoboMME dataset stub.  Replace body with actual RoboMME loader."""


_DATASET_REGISTRY = {
    "robocerebra": RoboCerebraDataset,
    "libero":      LiberoDataset,
    "calvin":      CalvinDataset,
    "robomme":     RoboMMEDataset,
}


def build_dataset(cfg: DatasetConfig, model_cfg) -> Dataset:
    """Instantiate the correct Dataset class for a given DatasetConfig."""
    cls = _DATASET_REGISTRY.get(cfg.name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown dataset '{cfg.name}'. "
            f"Available: {list(_DATASET_REGISTRY)}"
        )
    return cls(cfg, model_cfg)


def build_dataloader(
    stage_cfg: StageTrainConfig,
    model_cfg,
) -> DataLoader:
    """Build a (possibly multi-dataset) DataLoader for one training stage.

    Multiple datasets are combined with ConcatDataset.
    """
    if not stage_cfg.datasets:
        raise ValueError("stage_cfg.datasets is empty — add at least one dataset.")

    datasets = [build_dataset(d, model_cfg) for d in stage_cfg.datasets]
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    num_workers = max(d.num_workers for d in stage_cfg.datasets
                      if hasattr(d, "num_workers"))

    return DataLoader(
        combined,
        batch_size=stage_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate: stack tensors; keep variable-length node_ids padded."""
    images  = torch.stack([b["image"]   for b in batch])
    states  = torch.stack([b["state"]   for b in batch])
    actions = torch.stack([b["actions"] for b in batch])

    # Pad node_ids to the longest tree in the batch
    max_N = max(b["node_ids"].shape[0] for b in batch)
    node_ids = torch.stack([
        torch.cat([b["node_ids"],
                   torch.zeros(max_N - b["node_ids"].shape[0], dtype=torch.long)])
        for b in batch
    ])

    out: Dict[str, torch.Tensor] = {
        "images":    images,
        "states":    states,
        "actions":   actions,
        "node_ids":  node_ids,
    }
    return out


def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def build_optimizer(
    model: MemoryTreeVLA,
    stage_cfg: StageTrainConfig,
) -> torch.optim.Optimizer:
    """Build AdamW with optional per-module learning-rate multipliers."""
    base_lr = stage_cfg.lr
    mults   = stage_cfg.lr_multipliers   # e.g. {"action_llm": 0.1}

    param_groups: List[Dict] = []
    assigned: set = set()

    for name, mult in mults.items():
        module = getattr(model, name, None)
        if module is None:
            logger.warning("lr_multipliers: sub-module '%s' not found, skipping.", name)
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": base_lr * mult, "name": name})
            assigned.update(id(p) for p in params)

    # Remaining trainable params → base lr
    rest = [p for p in model.parameters()
            if p.requires_grad and id(p) not in assigned]
    if rest:
        param_groups.append({"params": rest, "lr": base_lr, "name": "default"})

    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=1e-4)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage_cfg: StageTrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps  = stage_cfg.epochs * steps_per_epoch
    warmup_steps = min(stage_cfg.warmup_steps, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: MemoryTreeVLA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    stage: int,
    output_dir: Path,
    tag: str = "latest",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"stage{stage}_{tag}.pth"
    torch.save(
        {
            "stage":       stage,
            "epoch":       epoch,
            "global_step": global_step,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved → %s", ckpt_path)
    return ckpt_path


def load_checkpoint(
    model: MemoryTreeVLA,
    optimizer: Optional[torch.optim.Optimizer],
    ckpt_path: str,
) -> Dict:
    logger.info("Loading checkpoint from %s …", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)
    return ckpt


# ---------------------------------------------------------------------------
# Single-stage training loop
# ---------------------------------------------------------------------------

def train_one_stage(
    stage: int,
    stage_cfg: StageTrainConfig,
    pipeline_cfg: TrainingPipelineConfig,
    model: MemoryTreeVLA,
    device: torch.device,
    resume_ckpt: Optional[str],
    model_cfg=None,
) -> Path:
    """Run one complete training stage.

    Args:
        stage      : 1, 2, or 3.
        stage_cfg  : hyper-parameters + dataset list for this stage.
        pipeline_cfg: global pipeline settings (output_dir, intervals, …).
        model      : the MemoryTreeVLA instance (already on device).
        device     : torch device.
        resume_ckpt: path to checkpoint to resume from (overrides stage_cfg.resume_from).

    Returns:
        Path to the best checkpoint produced by this stage.
    """
    logger.info("=" * 70)
    logger.info("  Starting Stage %d / 3", stage)
    logger.info("=" * 70)

    # 1. Apply parameter freeze policy
    model.set_stage(stage)

    # 2. Build DataLoader
    loader = build_dataloader(stage_cfg, model_cfg)
    steps_per_epoch = len(loader)

    # 3. Build optimizer & scheduler
    optimizer = build_optimizer(model, stage_cfg)
    scheduler = build_scheduler(optimizer, stage_cfg, steps_per_epoch)

    # 4. Optionally resume
    ckpt_to_load = resume_ckpt or stage_cfg.resume_from
    start_epoch  = 0
    global_step  = 0
    if ckpt_to_load:
        meta = load_checkpoint(model, optimizer, ckpt_to_load)
        start_epoch  = meta.get("epoch", 0) + 1
        global_step  = meta.get("global_step", 0)

    # 5. Output directory for this stage
    stage_dir = Path(pipeline_cfg.output_dir) / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # 6. Metrics
    loss_meter = AverageMeter()
    best_loss  = float("inf")
    best_ckpt  = stage_dir / f"stage{stage}_best.pth"

    t0 = time.time()

    for epoch in range(start_epoch, stage_cfg.epochs):
        model.train()
        loss_meter.reset()

        for batch_idx, batch in enumerate(loader):
            batch = _batch_to_device(batch, device)

            # --- Forward pass ---
            out = model(
                images=batch["images"],
                node_ids=batch["node_ids"],
                state=batch["states"],
                actions_gt=batch["actions"],
            )
            loss = out["loss"]

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            if stage_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    stage_cfg.grad_clip,
                )
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item())
            global_step += 1

            # Logging
            if global_step % pipeline_cfg.log_interval == 0:
                elapsed = time.time() - t0
                cur_lr  = optimizer.param_groups[0]["lr"]
                logger.info(
                    "[Stage %d | Epoch %02d/%02d | Step %06d] "
                    "loss=%.4f (avg=%.4f)  lr=%.2e  elapsed=%.0fs",
                    stage, epoch + 1, stage_cfg.epochs,
                    global_step, loss.item(), loss_meter.avg,
                    cur_lr, elapsed,
                )

            # Periodic checkpoint
            if global_step % pipeline_cfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step, stage,
                    stage_dir, tag=f"step{global_step:07d}"
                )

        # End-of-epoch checkpoint + best-model tracking
        save_checkpoint(model, optimizer, epoch, global_step, stage,
                        stage_dir, tag="latest")
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, global_step, stage,
                            stage_dir, tag="best")
            best_ckpt  = stage_dir / f"stage{stage}_best.pth"
            logger.info("  ↑ New best loss=%.4f at epoch %d", best_loss, epoch + 1)

        logger.info(
            "[Stage %d] Epoch %d/%d done — avg_loss=%.4f",
            stage, epoch + 1, stage_cfg.epochs, loss_meter.avg,
        )

    logger.info(
        "Stage %d finished.  Best loss=%.4f  Best ckpt: %s",
        stage, best_loss, best_ckpt
    )
    return best_ckpt


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MemoryTreeVLA three-stage training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default="MTVLA/configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--start_stage", type=int, default=1, choices=[1, 2, 3],
        help="Start from this stage (skip earlier stages).  "
             "Requires a matching --stageN_ckpt when > 1."
    )
    parser.add_argument(
        "--end_stage", type=int, default=3, choices=[1, 2, 3],
        help="Stop after this stage."
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, default=None,
        help="Checkpoint to resume / initialise Stage 1 from."
    )
    parser.add_argument(
        "--stage2_ckpt", type=str, default=None,
        help="Checkpoint to resume / initialise Stage 2 from "
             "(typically the Stage 1 best checkpoint)."
    )
    parser.add_argument(
        "--stage3_ckpt", type=str, default=None,
        help="Checkpoint to resume / initialise Stage 3 from "
             "(typically the Stage 2 best checkpoint)."
    )
    parser.add_argument(
        "--deepspeed", type=str, default=None,
        help="Override deepspeed_config path from pipeline config."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override pipeline.output_dir."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override pipeline.seed."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # ---- Load config ----
    cfg: MTVLAConfig = load_config(args.config)
    pc: TrainingPipelineConfig = cfg.pipeline

    # CLI overrides
    if args.output_dir:
        pc.output_dir = args.output_dir
    if args.seed is not None:
        pc.seed = args.seed
    if args.deepspeed:
        pc.deepspeed_config = args.deepspeed

    set_seed(pc.seed)
    Path(pc.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Build model ----
    logger.info("Building MemoryTreeVLA …")
    model = MemoryTreeVLA(cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Total parameters: %.2fM", total_params)

    # ---- Resolve stage checkpoints (CLI > YAML) ----
    s1_ckpt = args.stage1_ckpt or pc.stage1.resume_from
    s2_ckpt = args.stage2_ckpt or pc.stage2.resume_from
    s3_ckpt = args.stage3_ckpt or pc.stage3.resume_from

    # When starting from Stage 2/3, a preceding checkpoint must be loaded
    # to initialise the model with the upstream stage weights.
    if args.start_stage >= 2 and s2_ckpt:
        logger.info("Pre-loading Stage 2 init checkpoint: %s", s2_ckpt)
        load_checkpoint(model, None, s2_ckpt)
    elif args.start_stage >= 3 and s3_ckpt:
        logger.info("Pre-loading Stage 3 init checkpoint: %s", s3_ckpt)
        load_checkpoint(model, None, s3_ckpt)

    # ---- Run stages ----
    last_ckpt: Optional[Path] = None

    stage_map = {
        1: (pc.stage1, s1_ckpt),
        2: (pc.stage2, s2_ckpt),
        3: (pc.stage3, s3_ckpt),
    }

    for stage_idx in range(args.start_stage, args.end_stage + 1):
        stage_cfg, resume_ckpt = stage_map[stage_idx]

        if not stage_cfg.enabled:
            logger.info("Stage %d is disabled in config, skipping.", stage_idx)
            continue

        # Stage 1 dataset is always RoboCerebra (enforced here)
        if stage_idx == 1:
            if not stage_cfg.datasets:
                logger.warning(
                    "Stage 1 datasets list is empty — adding default RoboCerebra config."
                )
                from MTVLA.configs.config import DatasetConfig
                stage_cfg.datasets = [DatasetConfig(
                    name="robocerebra",
                    data_root="dataset/RoboCerebra/RoboCerebra_trainset",
                )]
            non_rc = [d for d in stage_cfg.datasets if d.name.lower() != "robocerebra"]
            if non_rc:
                logger.warning(
                    "Stage 1 only supports RoboCerebra (tree annotation required). "
                    "Ignoring non-RoboCerebra entries: %s",
                    [d.name for d in non_rc],
                )
                stage_cfg.datasets = [d for d in stage_cfg.datasets
                                       if d.name.lower() == "robocerebra"]

        # Stages 2/3 must have at least one configurable dataset
        if stage_idx in (2, 3) and not stage_cfg.datasets:
            logger.error(
                "Stage %d has no datasets configured. "
                "Add at least one entry under pipeline.stage%d.datasets in the YAML.",
                stage_idx, stage_idx,
            )
            raise ValueError(f"Stage {stage_idx} datasets not configured.")

        # If stage N-1 produced a checkpoint and stage N has no explicit ckpt,
        # automatically chain them (progressive training).
        if last_ckpt is not None and resume_ckpt is None:
            logger.info(
                "Auto-chaining: loading Stage %d init from previous stage best: %s",
                stage_idx, last_ckpt,
            )
            load_checkpoint(model, None, str(last_ckpt))

        last_ckpt = train_one_stage(
            stage=stage_idx,
            stage_cfg=stage_cfg,
            pipeline_cfg=pc,
            model=model,
            device=device,
            resume_ckpt=resume_ckpt,
            model_cfg=cfg.model,
        )

    logger.info("Training complete.  Final checkpoint: %s", last_ckpt)


if __name__ == "__main__":
    main()

