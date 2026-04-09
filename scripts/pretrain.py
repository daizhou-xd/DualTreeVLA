"""
DualTreeVLA 预训练脚本 — 语义边界结构学习

预训练阶段目标（RoboCerebra）:
  1. 训练 JumpAwareHead：纯动作突变 → 分支点检测（L_boundary）
  2. 检验分支点语义是否接近真实子任务描述（L_sem，InfoNCE）
  3. SGMTS 视觉扫描的语义引导能力同步优化

冻结: LLM backbone, CrossModalFusion, FlowMatchingActionHead
可训练: SGMTS, s_proj, JumpAwareHead, TreeSSMReadout, MLPElevation

损失（完全不包含 FlowMatching）:
  L_boundary  — 动作突变边界 BCE（自监督或 RoboCerebra 标注）
  L_sem       — 分支点语义 InfoNCE（需 RoboCerebra 子任务描述标注）
  L_elev      — 语义提升一致性

用法:
  单卡: python pretrain.py --config configs/pretrain.yaml
  多卡: accelerate launch --config_file configs/accelerate_zero2.yaml \\
            pretrain.py --config configs/pretrain.yaml
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

from dual_tree_vla.model import DualTreeVLA
from dual_tree_vla.dataset import RoboCerebraDataset, robocerebra_collate


# ================================================================
#  工具函数
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


# ================================================================
#  冻结策略 — 预训练阶段
# ================================================================

def freeze_for_pretrain(model: DualTreeVLA):
    """
    冻结: LLM backbone, CrossModalFusion, FlowMatchingActionHead
    可训练: SGMTS, s_proj, JumpAwareHead, TreeSSMReadout, MLPElevation
    """
    # 冻结 LLM（init 时 freeze_llm=True 已冻结，再显式确认一次）
    for p in model.llm.parameters():
        p.requires_grad = False

    # 冻结 FlowMatching 动作头和跨模态融合
    for m in [model.action_head, model.fusion]:
        for p in m.parameters():
            p.requires_grad = False

    # 可训练模块
    trainable_modules = [
        model.sgmts,
        model.sem_proj,
        model.jump_head,
        model.tree_ssm,
        model.mlp_elev,
    ]
    for m in trainable_modules:
        for p in m.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def inspect_parameters(model: DualTreeVLA, accel=None):
    log_msg("\n── 参数统计 ──────────────────────────────", accel)
    groups = {
        "LLM backbone":         model.llm,
        "SGMTS encoder":        model.sgmts,
        "sem_proj":             model.sem_proj,
        "JumpAwareHead":        model.jump_head,
        "TreeSSMReadout":       model.tree_ssm,
        "MLPElevation":         model.mlp_elev,
        "CrossModalFusion":     model.fusion,
        "FlowMatchingHead":     model.action_head,
    }
    total_all, train_all = 0, 0
    for name, mod in groups.items():
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        status = "TRAIN" if train > 0 else "frozen"
        log_msg(f"  {name:<24}: {total/1e6:6.2f}M total  {train/1e6:6.2f}M trainable  [{status}]", accel)
        total_all += total
        train_all  += train
    log_msg(f"  {'─'*60}", accel)
    log_msg(f"  {'TOTAL':<24}: {total_all/1e6:6.2f}M total  {train_all/1e6:6.2f}M trainable", accel)
    log_msg("", accel)


# ================================================================
#  单步训练（一条轨迹）
# ================================================================

def pretrain_step(
    model: DualTreeVLA,
    batch: Dict,
    device: torch.device,
    loss_cfg: dict,
) -> Dict[str, torch.Tensor]:
    """
    调用 model(..., mode='pretrain')：仅计算 L_boundary + L_sem + L_elev。
    完全不触发 FlowMatching 动作头。
    """
    frames      = batch["frames"].to(device)         # (B, T, 3, H, W)
    actions     = batch["actions"].to(device)        # (B, T, d_a)
    states      = batch["states"].to(device)         # (B, T, d_q)
    subtask_ids = batch.get("subtask_ids")
    if subtask_ids is not None:
        subtask_ids = subtask_ids.to(device)
    instructions: List[str] = batch["instructions"]

    # 预训练 forward：L_boundary + L_sem + L_elev（模型内部完成）
    losses = model(
        images=frames,
        instructions=instructions,
        states=states,
        actions=actions,
        subtask_ids=subtask_ids,
        subtask_descs=batch.get("subtask_descs"),
        mode="pretrain",
        w_boundary=loss_cfg.get("w_boundary", 1.0),
        w_sem=loss_cfg.get("w_sem", 0.5),
        w_elev=loss_cfg.get("w_elev", 0.2),
        tau_sem=loss_cfg.get("tau_sem", 0.07),
    )

    return {
        "loss":       losses.get("total", torch.zeros((), device=device)),
        "L_boundary": losses.get("L_boundary", torch.zeros((), device=device)).detach(),
        "L_sem":      losses.get("L_sem", torch.zeros((), device=device)).detach(),
        "L_elev":     losses.get("L_elev", torch.zeros((), device=device)).detach(),
    }


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
    print(f"[pretrain] Saved checkpoint → {path}", flush=True)


# ================================================================
#  主训练循环
# ================================================================

def train(cfg: dict):
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

    log_msg(f"[pretrain] device={device}  mixed_precision={cfg.get('mixed_precision', 'bf16')}", accel)

    # ── Dataset ──────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    dataset  = RoboCerebraDataset(
        root       = data_cfg["root"],
        scenes     = data_cfg.get("scenes"),
        img_h      = data_cfg.get("img_h", 224),
        img_w      = data_cfg.get("img_w", 224),
        subsample  = data_cfg.get("subsample", 4),
        max_seqlen = data_cfg.get("max_seqlen", 64),
    )
    log_msg(f"[pretrain] Dataset: {len(dataset)} 条轨迹", accel)

    loader = DataLoader(
        dataset,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["train"].get("num_workers", 4),
        collate_fn  = robocerebra_collate,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── 构建模型 ────────────────────────────────────────────────────
    mc = cfg["model"]
    model = DualTreeVLA(
        llm_path        = mc["llm_path"],
        d               = mc.get("d", 256),
        d_a             = mc.get("d_a", 7),
        d_q             = mc.get("d_q", 84),
        d_visual        = mc.get("d_visual", 256),
        d_ssm           = mc.get("d_ssm", 256),
        d_state         = mc.get("d_state", 16),
        patch_size      = mc.get("patch_size", 16),
        H_a             = mc.get("H_a", 16),
        n_ode           = mc.get("n_ode", 20),
        theta_fuse      = mc.get("theta_fuse", 0.65),
        K_elev          = mc.get("K_elev", 4),
        delta_w         = mc.get("delta_w", 0.1),
        tau             = mc.get("tau", 0.1),
        clip_model_name = mc.get("clip_model_name"),   # None → PatchCNN fallback
        freeze_llm      = True,
    )

    n_train, n_total = freeze_for_pretrain(model)
    log_msg(f"[pretrain] 可训练参数: {n_train:,} / {n_total:,}", accel)
    inspect_parameters(model, accel)

    # 可选：从已有断点恢复
    tc = cfg["train"]
    resume = tc.get("resume_from")
    if resume and os.path.isfile(str(resume)):
        ckpt = torch.load(resume, map_location="cpu")
        missing, unexp = model.load_state_dict(ckpt["model"], strict=False)
        log_msg(f"[pretrain] 恢复自 {resume}  missing={len(missing)}  unexpected={len(unexp)}", accel)

    # ── 优化器（参考 Evo-1 train.py 风格）────────────────────────────
    lr = float(tc.get("lr", 3e-4))
    wd = float(tc.get("weight_decay", 1e-4))

    decay_p, no_decay_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("norm", "bias", "A_log")):
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": wd},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=lr,
    )

    total_steps  = tc["epochs"] * len(loader)
    warmup_steps = int(tc.get("warmup_ratio", 0.05) * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Accelerate 包装 ──────────────────────────────────────────────
    if _ACCELERATE and accel is not None:
        model, optimizer, loader, scheduler = accel.prepare(
            model, optimizer, loader, scheduler
        )

    model.train()

    # ── W&B 初始化 ────────────────────────────────────────────────────
    if _WANDB and is_main(accel):
        wandb.init(
            project=cfg.get("wandb_project", "DualTreeVLA-pretrain"),
            name=cfg.get("wandb_run",    f"pretrain_{int(time.time())}"),
            config=cfg,
            mode="offline",
        )

    # ── 训练循环 ─────────────────────────────────────────────────────
    ckpt_dir    = Path(tc["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every  = tc.get("save_every", 5)
    global_step = 0
    best_loss   = float("inf")
    loss_cfg    = cfg.get("loss", {})

    for epoch in range(1, tc["epochs"] + 1):
        epoch_sums: Dict[str, float] = {}

        for batch in loader:
            if not _ACCELERATE:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            step_losses = pretrain_step(model, batch, device, loss_cfg)
            loss = step_losses["loss"]

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

            for k, v in step_losses.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v))

            if global_step % 50 == 0 and is_main(accel):
                lr_now = scheduler.get_last_lr()[0]
                detail = "  ".join(f"{k}={v:.4f}" for k, v in step_losses.items() if k != "loss")
                log_msg(
                    f"[pretrain] ep={epoch}/{tc['epochs']}  step={global_step}"
                    f"  loss={loss.item():.4f}  lr={lr_now:.2e}  {detail}",
                    accel,
                )
                if _WANDB:
                    wandb.log({"train/"+k: (v.item() if isinstance(v, torch.Tensor) else v)
                               for k, v in step_losses.items()}, step=global_step)
                    wandb.log({"lr": lr_now}, step=global_step)

        n_batch  = max(len(loader), 1)
        avg_loss = epoch_sums.get("loss", 0.0) / n_batch
        log_msg(f"[pretrain] Epoch {epoch}/{tc['epochs']}  avg_loss={avg_loss:.4f}", accel)

        if is_main(accel) and epoch % save_every == 0:
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / f"pretrain_ep{epoch:03d}.pt", accel)

        if is_main(accel) and avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / "pretrain_best.pt", accel)

    log_msg("[pretrain] 完成。", accel)


# ================================================================
#  入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pretrain.yaml")
    args = parser.parse_args()
    cfg  = load_cfg(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

