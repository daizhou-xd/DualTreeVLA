"""
MemoryTreeVLA 训练脚本 — Phase 1 / Phase 2（参考 Evo-1 train.py 风格）

═══════════════════════════════════════════════════════════════════
  Phase 1 — FlowMatching 热身（LIBERO，LLM 冻结）
═══════════════════════════════════════════════════════════════════
  损失: 仅 L_flow（不混合任何语义损失）
  可训练: CrossModalFusion, FlowMatchingActionHead
  冻结:   LLM backbone + 全部预训练模块（从 pretrain_best.pt 加载）
  推荐:   accelerate launch ... train.py --config configs/train_phase1.yaml --phase 1

═══════════════════════════════════════════════════════════════════
  Phase 2 — 全量微调（LIBERO）
═══════════════════════════════════════════════════════════════════
  损失: 仅 L_flow（主），可选 L_prog（权重很低）
  可训练: 全部（可选 LoRA LLM）
  推荐:   accelerate launch ... train.py --config configs/train_phase2.yaml --phase 2

用法:
  Phase 1 (8 GPU):
    accelerate launch --config_file configs/accelerate_zero2.yaml \\
        train.py --config configs/train_phase1.yaml --phase 1
  Phase 2 (8 GPU):
    accelerate launch --config_file configs/accelerate_zero3.yaml \\
        train.py --config configs/train_phase2.yaml --phase 2
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    _ACCELERATE = True
except ImportError:
    _ACCELERATE = False

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tree_vla.model import MemoryTreeVLA
from memory_tree_vla.dataset import LiberoDataset, libero_collate


# ================================================================
#  工具函数（与 Evo-1 train.py 保持一致风格）
# ================================================================

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_main(accel=None) -> bool:
    if accel is not None:
        return accel.is_main_process
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log_msg(msg: str, accel=None):
    if is_main(accel):
        print(msg, flush=True)


def get_lr_lambda(warmup_steps: int, total_steps: int, resume_step: int = 0):
    """Cosine decay with linear warmup（与 Evo-1 保持一致）。"""
    def lr_lambda(current_step: int) -> float:
        current_step += resume_step
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def inspect_named_modules(model: MemoryTreeVLA, accel=None):
    """打印各模块参数统计（直接对标 Evo-1 inspect_named_submodules）。"""
    groups = {
        "LLM backbone":     model.llm,
        "SGMTS":            model.sgmts,
        "s_proj":           model.s_proj,
        "JumpAwareHead":    model.jump_head,
        "TreeSSMReadout":   model.tree_ssm,
        "MLPElevation":     model.mlp_elev,
        "CrossModalFusion": model.fusion,
        "FlowMatchingHead": model.action_head,
    }
    log_msg("\n── 参数统计 ──────────────────────────────────────────", accel)
    total_all, train_all = 0, 0
    for name, mod in groups.items():
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        log_msg(f"  {name:<22}: {total/1e6:6.2f}M  trainable={train/1e6:6.2f}M"
                + ("  [TRAIN]" if train > 0 else "  [frozen]"), accel)
        total_all += total
        train_all  += train
    log_msg(f"  TOTAL: {total_all/1e6:.2f}M  trainable={train_all/1e6:.2f}M\n", accel)


# ================================================================
#  冻结策略
# ================================================================

def freeze_phase1(model: MemoryTreeVLA):
    """
    Phase 1：冻结 LLM + 全部预训练模块，只训 CrossModalFusion + FlowMatchingHead。
    """
    for p in model.parameters():
        p.requires_grad = False

    for m in [model.fusion, model.action_head]:
        for p in m.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def unfreeze_phase2(model: MemoryTreeVLA):
    """
    Phase 2：全量解冻（LLM 也解冻，但 LR 极低）。
    """
    for p in model.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


# ================================================================
#  Checkpoint 保存
# ================================================================

def save_ckpt(model, optimizer, epoch, step, path: Path, accel=None):
    if accel is not None and hasattr(accel, "unwrap_model"):
        state = accel.unwrap_model(model).state_dict()
    else:
        state = model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step}, path)
    print(f"[train] Saved → {path}", flush=True)


# ================================================================
#  主训练
# ================================================================

def train(cfg: dict, phase: int):
    assert phase in (1, 2), f"phase must be 1 or 2, got {phase}"

    # ── Accelerator ─────────────────────────────────────────────────
    if _ACCELERATE:
        accel = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=cfg.get("grad_accum", 1),
        )
        device = accel.device
    else:
        accel  = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main(accel):
        if _ACCELERATE:
            set_seed(cfg.get("seed", 42))

    tag = f"[phase{phase}]"
    log_msg(f"{tag} device={device}  mixed_precision={cfg.get('mixed_precision','bf16')}", accel)

    # ── Dataset ──────────────────────────────────────────────────────
    dc = cfg["data"]
    dataset = LiberoDataset(
        root       = dc["root"],
        img_h      = dc.get("img_h", 224),
        img_w      = dc.get("img_w", 224),
        d_q        = dc.get("d_q", 84),
        d_a        = dc.get("d_a", 7),
        max_seqlen = dc.get("max_seqlen", 64),
        normalize  = dc.get("normalize", True),
    )
    log_msg(f"{tag} Dataset: {len(dataset)} episodes", accel)

    loader = DataLoader(
        dataset,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["train"].get("num_workers", 4),
        collate_fn  = libero_collate,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── 构建模型 ────────────────────────────────────────────────────
    mc = cfg["model"]
    model = MemoryTreeVLA(
        llm_path   = mc["llm_path"],
        d          = mc.get("d", 256),
        d_a        = mc.get("d_a", 7),
        d_q        = mc.get("d_q", 84),
        d_visual   = mc.get("d_visual", 256),
        d_ssm      = mc.get("d_ssm", 256),
        d_state    = mc.get("d_state", 16),
        patch_size = mc.get("patch_size", 16),
        H_a        = mc.get("H_a", 16),
        n_ode      = mc.get("n_ode", 20),
        theta_fuse = mc.get("theta_fuse", 0.35),
        K_elev     = mc.get("K_elev", 4),
        delta_w    = mc.get("delta_w", 0.1),
        tau        = mc.get("tau", 0.1),
        freeze_llm = (phase == 1),
    )

    # 从预训练 / Phase1 ckpt 初始化
    tc     = cfg["train"]
    init_ckpt = tc.get("init_from")
    resume    = tc.get("resume_from")
    if init_ckpt and os.path.isfile(str(init_ckpt)):
        ckpt = torch.load(init_ckpt, map_location="cpu")
        missing, unexp = model.load_state_dict(ckpt["model"], strict=False)
        log_msg(f"{tag} 加载 {init_ckpt}  missing={len(missing)}  unexpected={len(unexp)}", accel)
    elif resume and os.path.isfile(str(resume)):
        ckpt = torch.load(resume, map_location="cpu")
        missing, unexp = model.load_state_dict(ckpt["model"], strict=False)
        log_msg(f"{tag} 恢复自 {resume}  missing={len(missing)}  unexpected={len(unexp)}", accel)

    # 冻结设置
    if phase == 1:
        n_train, n_total = freeze_phase1(model)
        mode_str = "phase1"
    else:
        n_train, n_total = unfreeze_phase2(model)
        mode_str = "phase2"

    log_msg(f"{tag} 可训练参数: {n_train:,} / {n_total:,}", accel)
    inspect_named_modules(model, accel)

    # ── 优化器（Evo-1 风格：分组 weight decay）────────────────────────
    lr = float(tc.get("lr", 1e-4 if phase == 1 else 3e-5))
    wd = float(tc.get("weight_decay", 1e-4))

    # Phase 2：LLM 使用更低 LR
    if phase == 2:
        llm_params    = [p for p in model.llm.parameters()  if p.requires_grad]
        other_params  = [p for name, p in model.named_parameters()
                         if p.requires_grad and not name.startswith("llm.")]
        param_groups  = [
            {"params": other_params, "lr": lr,       "weight_decay": wd},
            {"params": llm_params,   "lr": lr * 0.1, "weight_decay": wd},
        ]
    else:
        decay_p, no_decay_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("norm", "bias", "A_log")):
                no_decay_p.append(p)
            else:
                decay_p.append(p)
        param_groups = [
            {"params": decay_p,    "weight_decay": wd},
            {"params": no_decay_p, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    total_steps  = tc["epochs"] * len(loader)
    warmup_steps = int(tc.get("warmup_ratio", 0.05) * total_steps)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(warmup_steps, total_steps)
    )

    # ── Accelerate 包装 ──────────────────────────────────────────────
    if _ACCELERATE and accel is not None:
        model, optimizer, loader, scheduler = accel.prepare(
            model, optimizer, loader, scheduler
        )

    model.train()

    # ── W&B ──────────────────────────────────────────────────────────
    project_name = cfg.get("wandb_project", f"MemoryTreeVLA-phase{phase}")
    if _WANDB and is_main(accel):
        wandb.init(
            project=project_name,
            name=cfg.get("wandb_run", f"phase{phase}_{int(time.time())}"),
            config=cfg,
            mode="offline",
        )

    # ── 训练循环 ─────────────────────────────────────────────────────
    ckpt_dir    = Path(tc["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every  = tc.get("save_every", 5)
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, tc["epochs"] + 1):
        epoch_loss_sum = 0.0

        for batch in loader:
            if not _ACCELERATE:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            frames  = batch["frames"].to(device)          # (B, T, C, H, W)
            actions = batch["actions"].to(device)         # (B, T, d_a)
            states  = batch["states"].to(device)          # (B, T, d_q)
            instructions: List[str] = batch["instructions"]

            # ── 前向 + 损失（仅 L_flow）────────────────────────────
            losses = model(
                images=frames,
                instructions=instructions,
                states=states,
                actions=actions,
                mode=mode_str,
            )
            loss = losses["total"]

            # 数值稳定性检查
            if not torch.isfinite(loss):
                log_msg(f"{tag} step={global_step} loss=NaN/inf, 跳过本批", accel)
                optimizer.zero_grad()
                continue

            if _ACCELERATE and accel is not None:
                accel.backward(loss)
                accel.clip_grad_norm_(model.parameters(), 1.0)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            epoch_loss_sum += loss.item()

            if global_step % 50 == 0 and is_main(accel):
                lr_now = scheduler.get_last_lr()[0]
                log_msg(
                    f"{tag} ep={epoch}/{tc['epochs']}  step={global_step}"
                    f"  L_flow={loss.item():.4f}  lr={lr_now:.2e}",
                    accel,
                )
                if _WANDB:
                    wandb.log({"train/L_flow": loss.item(), "lr": lr_now},
                              step=global_step)

        avg_loss = epoch_loss_sum / max(len(loader), 1)
        log_msg(f"{tag} Epoch {epoch}/{tc['epochs']}  avg_L_flow={avg_loss:.4f}", accel)

        if is_main(accel) and epoch % save_every == 0:
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / f"phase{phase}_ep{epoch:03d}.pt", accel)

        if is_main(accel) and avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / f"phase{phase}_best.pt", accel)

    log_msg(f"{tag} 训练完成。", accel)


# ================================================================
#  入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase",  type=int, required=True, choices=[1, 2])
    args = parser.parse_args()
    cfg  = load_cfg(args.config)
    train(cfg, phase=args.phase)


if __name__ == "__main__":
    main()

