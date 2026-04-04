"""
MemoryTreeVLA 预训练脚本 — 语义边界结构学习

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

from memory_tree_vla.model import MemoryTreeVLA
from memory_tree_vla.model.memory_tree.operations import semantic_elevation
from memory_tree_vla.losses import l_sem, l_elev, pretrain_loss
from memory_tree_vla.dataset import RoboCerebraDataset, robocerebra_collate


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

def freeze_for_pretrain(model: MemoryTreeVLA):
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


def inspect_parameters(model: MemoryTreeVLA, accel=None):
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
    model: MemoryTreeVLA,
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

    # 预训练 forward：仅 JumpAwareHead + L_boundary
    losses = model(
        images=frames,
        instructions=instructions,
        states=states,
        actions=actions,
        subtask_ids=subtask_ids,
        mode="pretrain",
    )

    # L_sem：从树中提取分支节点语义，与子任务文本对比
    w_sem = loss_cfg.get("w_sem", 0.5)
    L_sem_val = torch.zeros((), device=device)
    if w_sem > 0 and subtask_ids is not None:
        L_sem_val = _compute_tree_sem_loss(model, batch, device, loss_cfg)

    # L_elev：语义提升一致性
    w_elev = loss_cfg.get("w_elev", 0.2)
    L_elev_val = torch.zeros((), device=device)
    if w_elev > 0:
        L_elev_val = _compute_tree_elev_loss(model, frames.shape[0], device, model.mlp_elev)

    # 合并总损失（L_boundary 已在 losses["total"] 中）
    L_boundary = losses.get("L_boundary", torch.zeros((), device=device))
    w_boundary = loss_cfg.get("w_boundary", 1.0)
    total = w_boundary * L_boundary + w_sem * L_sem_val + w_elev * L_elev_val

    return {
        "loss":       total,
        "L_boundary": L_boundary.detach(),
        "L_sem":      L_sem_val.detach(),
        "L_elev":     L_elev_val.detach(),
    }


@torch.no_grad()
def _encode_subtask_text(model: MemoryTreeVLA, descs: List[str], device) -> torch.Tensor:
    """用 LLM 编码子任务描述，返回均值池化语义向量 (N, d_lang)。"""
    enc = model.tokenizer(
        descs, return_tensors="pt", padding=True, truncation=True, max_length=64
    ).to(device)
    out  = model.llm(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)


def _compute_tree_sem_loss(model: MemoryTreeVLA, batch: Dict, device, loss_cfg: dict) -> torch.Tensor:
    """
    L_sem：遍历树中抽象节点（已提升的节点），做子任务语义 InfoNCE。

    abstract node.s (d,) 经 sem_proj → d_lang，与子任务描述的语言嵌入对齐。
    跳过所有叶节点（s=None）。
    """
    s_nodes_list, s_text_list = [], []
    subtask_ids   = batch.get("subtask_ids")
    subtask_descs = batch.get("subtask_descs")
    instructions  = batch["instructions"]
    B = len(instructions)

    w_dtype = next(model.sem_proj.parameters()).dtype

    for b in range(B):
        tree = model.get_tree(b)
        if tree.size() == 0:
            continue
        descs = (subtask_descs[b] if subtask_descs and b < len(subtask_descs)
                 else [instructions[b]])
        if not descs:
            continue
        g_sub = _encode_subtask_text(model, descs, device)  # (S, d_lang)

        for nid, node in tree.nodes.items():
            # 只处理抽象节点（已提升，s != None）
            if node.is_leaf() or node.s is None:
                continue
            t_idx = min(nid, subtask_ids.shape[1] - 1) if subtask_ids is not None else 0
            sid   = int(subtask_ids[b, t_idx].item()) if subtask_ids is not None else 0
            sid   = max(0, min(sid, len(descs) - 1))
            # 将 node.s (d,) 投影到 d_lang 空间
            s_proj_out = model.sem_proj(node.s.to(device=device, dtype=w_dtype))  # (d_lang,)
            s_nodes_list.append(s_proj_out)
            s_text_list.append(g_sub[sid].to(device=device, dtype=w_dtype))

    if len(s_nodes_list) < 2:
        return torch.zeros((), device=device)

    S_nodes = torch.stack(s_nodes_list, dim=0)   # (N, d_lang)
    S_text  = torch.stack(s_text_list,  dim=0)   # (N, d_lang)
    return l_sem(S_nodes, S_text, temperature=loss_cfg.get("tau_sem", 0.07))


def _compute_tree_elev_loss(model: MemoryTreeVLA, B: int, device, mlp_elev) -> torch.Tensor:
    """
    L_elev：对每个抽象节点，验证其 s 与子节点加权语义的一致性。

    子节点可以是：
      - 抽象节点 → 直接使用 node.s
      - 叶节点   → 通过 mlp_elev(node.z_v) 投影到语义空间
    """
    elev_losses = []
    dtype = next(mlp_elev.parameters()).dtype
    for b in range(B):
        tree = model.get_tree(b)
        for nid, node in tree.nodes.items():
            # 只处理抽象节点（有子节点且 s 有效）
            if node.is_leaf() or node.s is None:
                continue
            ch_ids = node.children_ids
            if len(ch_ids) < 1:
                continue
            s_ch_list = []
            w_ch_list = []
            for c in ch_ids:
                child = tree.nodes[c]
                w_ch_list.append(child.w)
                if child.is_leaf():
                    # 叶节点：用 mlp_elev(z_v) 获得语义代理
                    z_v_c = child.z_v.to(device=device, dtype=dtype)
                    with torch.no_grad():
                        s_proxy = mlp_elev(z_v_c.unsqueeze(0)).squeeze(0)
                    s_ch_list.append(s_proxy)
                else:
                    if child.s is None:
                        continue
                    s_ch_list.append(child.s.to(device=device, dtype=dtype))
            if len(s_ch_list) < 1:
                continue
            # 在梯度计算下重新计算 s_abs（让 mlp_elev 有梯度流向）
            # 收集所有叶子的 z_v 做加权池化（与 operations.py 保持一致）
            leaf_zv = [tree.nodes[c].z_v.to(device=device, dtype=dtype)
                       for c in ch_ids if tree.nodes[c].is_leaf()]
            if leaf_zv:
                lw = [tree.nodes[c].w for c in ch_ids if tree.nodes[c].is_leaf()]
                lw_t = torch.tensor(lw, device=device, dtype=dtype)
                lw_t = lw_t / lw_t.sum().clamp(min=1e-6)
                z_pool = sum(z * w for z, w in zip(leaf_zv, lw_t))
                s_abs = mlp_elev(z_pool.unsqueeze(0)).squeeze(0)
            else:
                s_abs = node.s.to(device=device, dtype=dtype)
            elev_losses.append(l_elev(s_abs, s_ch_list, w_ch_list))
    if not elev_losses:
        return torch.zeros((), device=device)
    return torch.stack(elev_losses).mean()


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
        theta_fuse = mc.get("theta_fuse", 0.65),
        K_elev     = mc.get("K_elev", 4),
        delta_w    = mc.get("delta_w", 0.1),
        tau        = mc.get("tau", 0.1),
        freeze_llm = True,
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
            project=cfg.get("wandb_project", "MemoryTreeVLA-pretrain"),
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

