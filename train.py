"""
Training script for MemoryTreeVLA — with DeepSpeed + Flash Attention.

Supports:
  - Single GPU:   python train.py --config configs/default.yaml --phase 1
  - 8-GPU (DS):   deepspeed --num_gpus 8 train.py --deepspeed \
                      --config configs/default.yaml --phase 1

3-Phase training curriculum:
    Phase 1 — Visual warm-up       (SGMTS + s_proj + tree_ssm;  L_recon)
    Phase 2 — Action head          (+ fusion + prog_head + action_head;  L_flow + L_prog)
    Phase 3 — Joint fine-tuning    (+ LLM;  all losses)
      Optionally unfreeze LLM LoRA layers + all modules
      Loss: sum of all active losses

Usage:
    python train.py --config configs/default.yaml --phase 1
  - 8-GPU DeepSpeed:
    deepspeed --num_gpus 8 train.py --deepspeed --config configs/default.yaml --phase 1
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

# ── Optional DeepSpeed ─────────────────────────────────────────────
try:
    import deepspeed
    _DS_AVAILABLE = True
except ImportError:
    _DS_AVAILABLE = False

# ── Optional Weights & Biases ──────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models import MemoryTreeVLA
from models.memory_tree.operations import reinforce, prune
from losses import tree_loss, l_prog
from dataset import RoboCerebraDataset, robocerebra_collate


# ================================================================
#  Utilities
# ================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


_wandb_run = None    # module-level handle so train_epoch can log to it


def wandb_log(metrics: dict, step: int):
    """Log a dict of scalars to wandb (no-op if wandb is disabled)."""
    if _WANDB_AVAILABLE and _wandb_run is not None and is_main_process():
        _wandb_run.log(metrics, step=step)


def set_trainable(model: MemoryTreeVLA, phase: int) -> List[torch.nn.Parameter]:
    """
    Freeze all params then unfreeze the modules active at this phase.
    Phase 1: SGMTS + s_proj + tree_ssm + recon_decoder
    Phase 2: + fusion + prog_head + action_head
    Phase 3: + llm
    """
    for p in model.parameters():
        p.requires_grad_(False)

    train_modules: list = [model.sgmts, model.s_proj, model.tree_ssm]
    if phase == 1 or phase == 3:
        train_modules.append(model.recon_decoder)
    if phase >= 2:
        train_modules.extend([model.fusion, model.prog_head])
        train_modules.append(model.action_head)
    if phase >= 3:
        train_modules.append(model.llm)

    trainable, seen = [], set()
    for m in train_modules:
        for p in m.parameters():
            if id(p) not in seen:
                p.requires_grad_(True)
                trainable.append(p)
                seen.add(id(p))
    return trainable


# ================================================================
#  Training epoch
# ================================================================

def train_epoch(
    model_engine,                                        # DS engine or nn.Module
    loader: DataLoader,
    phase: int,
    cfg: dict,
    device: torch.device,
    epoch: int,
    use_deepspeed: bool,
    plain_optimizer: Optional[torch.optim.Optimizer],
    global_step_offset: int = 0,
) -> Dict[str, float]:

    model: MemoryTreeVLA = model_engine.module if use_deepspeed else model_engine
    model.train()
    total_losses: Dict[str, float] = {}
    n_steps = 0

    pbar = tqdm(
        loader,
        desc     = f"Epoch {epoch:03d}",
        disable  = not is_main_process(),
        dynamic_ncols = True,
        leave    = False,
    )
    for batch_idx, batch in enumerate(pbar):
        frames      = batch["frames"].to(device)          # (B, T, C, H, W)
        actions     = batch["actions"].to(device)         # (B, T, d_a)
        states      = batch["states"].to(device)          # (B, T, d_q)
        subtask_ids = batch["subtask_ids"].to(device)
        instructions: List[str] = batch["instructions"]
        B = frames.shape[0]

        if use_deepspeed:
            model_engine.zero_grad()
        else:
            plain_optimizer.zero_grad()

        # ── Forward ─────────────────────────────────────────────────
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = model(
                images=frames,
                instructions=instructions,
                states=states,
                actions=actions,
                subtask_ids=subtask_ids,
                compute_flow=(phase >= 2),
            )
            L_flow     = loss_dict.get("L_flow",  torch.tensor(0.0, device=device))
            L_recon_val = loss_dict.get("L_recon", torch.tensor(0.0, device=device))
            L_prog_val  = loss_dict.get("L_prog",  torch.tensor(0.0, device=device))

            # ── Combined loss ────────────────────────────────────────────
            # Phase 1: L_flow=0 (no action head), L_prog=None (not yet)
            _zero = torch.zeros((), device=device, dtype=L_flow.dtype)
            loss = tree_loss(
                L_flow   = L_flow       if phase >= 2 else _zero,
                L_recon  = L_recon_val  if phase == 1 or phase == 3 else None,
                L_prog   = L_prog_val   if phase >= 2 else None,
                w_flow   = cfg["loss"]["w_flow"],
                w_recon  = cfg["loss"]["w_recon"],
                w_prog   = cfg["loss"]["w_prog"],
            )

        # ── Backward + optimiser step ────────────────────────────────
        if use_deepspeed:
            model_engine.backward(loss)
            model_engine.step()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            plain_optimizer.step()

        # ── Metrics ──────────────────────────────────────────────────
        step_metrics = {
            "L_total": loss.item(),
            "L_flow":  L_flow.item(),
            "L_recon": L_recon_val.item(),
            "L_prog":  L_prog_val.item(),
        }
        for k, v in step_metrics.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_steps += 1
        global_step = global_step_offset + n_steps

        # Update progress bar postfix every step (tqdm handles rate-limiting)
        pbar.set_postfix(
            loss  = f"{loss.item():.4f}",
            recon = f"{L_recon_val.item():.4f}",
            flow  = f"{L_flow.item():.4f}",
            prog  = f"{L_prog_val.item():.4f}",
        )

        if is_main_process() and (batch_idx % cfg["log_every"] == 0):
            log(
                f"  Epoch {epoch:03d} | Step {batch_idx:04d}/{len(loader)} | "
                f"loss={loss.item():.4f}  flow={L_flow.item():.4f}  "
                f"recon={L_recon_val.item():.4f}  prog={L_prog_val.item():.4f}"
            )
            # wandb step-level scalars
            wandb_log(
                {
                    "train/step_loss":  loss.item(),
                    "train/step_flow":  L_flow.item(),
                    "train/step_recon": L_recon_val.item(),
                    "train/step_prog":  L_prog_val.item(),
                },
                step=global_step,
            )

    return {k: v / max(n_steps, 1) for k, v in total_losses.items()}


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           default="configs/default.yaml")
    parser.add_argument("--phase",            type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--epochs",           type=int, default=None,
                        help="Override train.epochs in the config file")
    parser.add_argument("--resume",           type=str, default=None)
    # DeepSpeed args
    parser.add_argument("--deepspeed",        action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank",       type=int, default=-1,
                        help="Injected automatically by deepspeed launcher")
    # Wandb args
    parser.add_argument("--wandb",            action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",    type=str, default="MemoryTreeVLA")
    parser.add_argument("--wandb_entity",     type=str, default=None,
                        help="W&B team/entity (optional)")
    parser.add_argument("--wandb_name",       type=str, default=None,
                        help="Run name (default: auto phase+timestamp)")
    parser.add_argument("--wandb_tags",       nargs="*", default=None,
                        help="Space-separated list of tags, e.g. --wandb_tags phase1 a6000")
    args, _ = parser.parse_known_args()

    cfg   = load_config(args.config)
    phase = args.phase

    # Command-line --epochs overrides the value in the config file
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    # ── Distributed init ─────────────────────────────────────────────
    use_deepspeed = args.deepspeed or cfg.get("deepspeed", {}).get("enabled", False)
    if use_deepspeed and not _DS_AVAILABLE:
        raise RuntimeError("DeepSpeed not installed.  Run: pip install deepspeed")

    if use_deepspeed:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Enable Flash Attention SDPA backend globally
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    log(f"[Phase {phase}]  device={device}  deepspeed={use_deepspeed}")

    # ── Weights & Biases init ────────────────────────────────────────
    global _wandb_run
    use_wandb = args.wandb and _WANDB_AVAILABLE and is_main_process()
    if args.wandb and not _WANDB_AVAILABLE:
        log("[WARN] --wandb passed but wandb is not installed.  Run: pip install wandb")
    if use_wandb:
        import datetime
        run_name = args.wandb_name or f"phase{phase}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        tags = (args.wandb_tags or []) + [f"phase{phase}"]
        _wandb_run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            tags    = tags,
            config  = {
                **cfg.get("model", {}),
                **cfg.get("train", {}),
                **{f"loss/{k}": v for k, v in cfg.get("loss", {}).items()},
                "phase":       phase,
                "deepspeed":   use_deepspeed,
            },
            resume  = "allow" if args.resume else None,
        )
        log(f"W&B run: {_wandb_run.url}")
    elif not use_wandb:
        _wandb_run = None

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = RoboCerebraDataset(
        root       = cfg["data"]["root"],
        scenes     = cfg["data"].get("scenes"),
        img_h      = cfg["data"]["img_h"],
        img_w      = cfg["data"]["img_w"],
        subsample  = cfg["data"]["subsample"],
        max_seqlen = cfg["data"]["max_seqlen"],
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_deepspeed else None
    loader  = DataLoader(
        dataset,
        batch_size  = cfg["train"]["batch_size"],
        sampler     = sampler,
        shuffle     = sampler is None,
        num_workers = cfg["train"]["num_workers"],
        collate_fn  = robocerebra_collate,
        pin_memory  = True,
        drop_last   = True,
    )
    log(f"Dataset: {len(dataset)} trajectories  |  {len(loader)} steps/epoch")

    # ── Model ────────────────────────────────────────────────────────
    model = MemoryTreeVLA(
        llm_path   = cfg["model"]["llm_path"],
        d          = cfg["model"]["d"],
        d_a        = cfg["model"]["d_a"],
        d_q        = cfg["model"]["d_q"],
        d_visual   = cfg["model"]["d_visual"],
        d_ssm      = cfg["model"]["d_ssm"],
        d_state    = cfg["model"]["d_state"],
        patch_size = cfg["model"]["patch_size"],
        H_a        = cfg["model"]["H_a"],
        n_ode      = cfg["model"]["n_ode"],
        theta_fuse = cfg["model"]["theta_fuse"],
        K_elev     = cfg["model"]["K_elev"],
        delta_w    = cfg["model"]["delta_w"],
        tau        = cfg["model"]["tau"],
        freeze_llm = (phase < 3),
    ).to(device)

    if phase >= 3 and hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        if hasattr(model.llm, "config") and hasattr(model.llm.config, "use_cache"):
            model.llm.config.use_cache = False
        log("LLM gradient checkpointing enabled")

    # ── Trainable params ─────────────────────────────────────────────
    trainable = set_trainable(model, phase)
    log(f"Trainable: {sum(p.numel() for p in trainable) / 1e6:.2f}M params")

    # ── DeepSpeed or plain optimiser init ────────────────────────────
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    plain_optimizer = None
    plain_scheduler = None

    if use_deepspeed:
        ds_cfg_path = (
            args.deepspeed_config
            or cfg.get("deepspeed", {}).get("config", "configs/ds_zero2.json")
        )
        with open(ds_cfg_path) as f:
            ds_config = json.load(f)

        # Sync batch size with yaml
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        grad_accum = ds_config.get("gradient_accumulation_steps", 1)
        ds_config["train_micro_batch_size_per_gpu"] = cfg["train"]["batch_size"]
        ds_config["train_batch_size"] = cfg["train"]["batch_size"] * grad_accum * world

        if "optimizer" in ds_config:
            ds_config["optimizer"]["params"]["lr"]           = cfg["train"]["lr"]
            ds_config["optimizer"]["params"]["weight_decay"] = cfg["train"]["weight_decay"]

        model_engine, _, _, _ = deepspeed.initialize(
            model            = model,
            model_parameters = trainable,
            config           = ds_config,
        )
    else:
        model_engine    = model
        plain_optimizer = torch.optim.AdamW(
            trainable,
            lr           = cfg["train"]["lr"],
            weight_decay = cfg["train"]["weight_decay"],
        )
        plain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            plain_optimizer,
            T_max   = cfg["train"]["epochs"],
            eta_min = cfg["train"]["lr"] * 1e-2,
        )

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 1
    best_metric = float("inf")
    best_epoch = 0
    if args.resume:
        if use_deepspeed:
            _, client_sd = model_engine.load_checkpoint(args.resume)
            start_epoch  = (client_sd or {}).get("epoch", 0) + 1
            best_metric = (client_sd or {}).get("best_metric", float("inf"))
            best_epoch = (client_sd or {}).get("best_epoch", 0)
        else:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            plain_optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_metric = ckpt.get("best_metric", float("inf"))
            best_epoch = ckpt.get("best_epoch", 0)
        log(f"Resumed from epoch {start_epoch - 1}")

    # ── Training loop ────────────────────────────────────────────────
    steps_per_epoch = len(loader)
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0      = time.time()
        metrics = train_epoch(
            model_engine      = model_engine,
            loader            = loader,
            phase             = phase,
            cfg               = cfg,
            device            = device,
            epoch             = epoch,
            use_deepspeed     = use_deepspeed,
            plain_optimizer   = plain_optimizer,
            global_step_offset = global_step,
        )
        global_step += steps_per_epoch

        if plain_scheduler is not None:
            plain_scheduler.step()

        if is_main_process():
            elapsed = time.time() - t0
            log(
                f"Epoch {epoch:03d}  [{elapsed:.0f}s]  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
            # ── wandb epoch-level logging ──────────────────────────
            current_lr = (
                plain_optimizer.param_groups[0]["lr"]
                if plain_optimizer is not None
                else cfg["train"]["lr"]
            )
            epoch_log = {
                "epoch":         epoch,
                "train/epoch_loss":  metrics.get("L_total", 0),
                "train/epoch_flow":  metrics.get("L_flow",  0),
                "train/epoch_recon": metrics.get("L_recon", 0),
                "train/epoch_prog":  metrics.get("L_prog",  0),
                "train/lr":          current_lr,
                "train/epoch_time_s": elapsed,
            }
            # Log gradient histogram of trainable params every save_every epochs
            if (
                use_wandb
                and _wandb_run is not None
                and epoch % cfg["train"]["save_every"] == 0
            ):
                grad_tensors = [
                    p.grad.detach() for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if grad_tensors:
                    # Stack norms and call .item() once to avoid per-param GPU sync
                    all_norms = torch.stack([g.norm() for g in grad_tensors])
                    epoch_log["train/grad_norm_max"]  = all_norms.max().item()
                    epoch_log["train/grad_norm_mean"] = all_norms.mean().item()
            wandb_log(epoch_log, step=global_step)

        if epoch % cfg["train"]["save_every"] == 0:
            tag = f"phase{phase}_epoch{epoch:04d}"
            log(f"  ⏳ Saving checkpoint {tag} ...")
            t_ckpt = time.time()
            if use_deepspeed:
                # All ranks must call save_checkpoint together: DeepSpeed ZeRO
                # gathers sharded optimizer states via collective communication.
                # Wrapping in is_main_process() causes a cross-rank deadlock.
                model_engine.save_checkpoint(
                    str(ckpt_dir),
                    tag          = tag,
                    client_state = {"epoch": epoch, "metrics": metrics},
                )
            elif is_main_process():
                torch.save(
                    {
                        "epoch":     epoch,
                        "phase":     phase,
                        "model":     model.state_dict(),
                        "optimizer": plain_optimizer.state_dict(),
                        "metrics":   metrics,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                    ckpt_dir / f"{tag}.pt",
                )
            log(f"  ✓ Checkpoint saved: {ckpt_dir}/{tag}  ({time.time()-t_ckpt:.1f}s)")

        # Always keep one rolling best checkpoint per phase (by L_total)
        cur_metric_local = metrics.get("L_total", float("inf")) if is_main_process() else 0.0
        improved_local = False
        if is_main_process() and cur_metric_local < best_metric:
            best_metric = cur_metric_local
            best_epoch = epoch
            improved_local = True

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor([1 if improved_local else 0], device=device, dtype=torch.int32)
            torch.distributed.broadcast(flag, src=0)
            improved = bool(flag.item())

            best_metric_t = torch.tensor([best_metric if is_main_process() else 0.0], device=device, dtype=torch.float32)
            best_epoch_t = torch.tensor([best_epoch if is_main_process() else 0], device=device, dtype=torch.int32)
            torch.distributed.broadcast(best_metric_t, src=0)
            torch.distributed.broadcast(best_epoch_t, src=0)
            best_metric = float(best_metric_t.item())
            best_epoch = int(best_epoch_t.item())
        else:
            improved = improved_local

        if improved:
            best_tag = f"phase{phase}_best"
            log(f"  ⏳ Saving best checkpoint {best_tag} (epoch={best_epoch}, L_total={best_metric:.4f}) ...")
            t_best = time.time()
            if use_deepspeed:
                model_engine.save_checkpoint(
                    str(ckpt_dir),
                    tag=best_tag,
                    client_state={
                        "epoch": epoch,
                        "metrics": metrics,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                )
            elif is_main_process():
                torch.save(
                    {
                        "epoch":     epoch,
                        "phase":     phase,
                        "model":     model.state_dict(),
                        "optimizer": plain_optimizer.state_dict(),
                        "metrics":   metrics,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                    ckpt_dir / f"{best_tag}.pt",
                )
            log(f"  ✓ Best checkpoint saved: {ckpt_dir}/{best_tag}  ({time.time()-t_best:.1f}s)")

    log("Training complete.")
    if use_wandb and _wandb_run is not None:
        _wandb_run.finish()


if __name__ == "__main__":
    main()
