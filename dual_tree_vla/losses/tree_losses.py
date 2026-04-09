"""
DualTreeVLA loss functions.

Pretrain:
  - l_boundary
  - l_sem
  - l_elev
  - pretrain_loss

Phase 1/2:
  - L_flow is computed inside action head.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeReconDecoder(nn.Module):
    """Optional semantic reconstruction decoder used in some ablations."""

    def __init__(self, d: int, d_patch: int = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        weight_dtype = next(self.parameters()).dtype
        return self.mlp(z_v.to(dtype=weight_dtype))


def l_boundary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary boundary classification loss with automatic positive reweighting."""
    labels = labels.to(logits.device).float()
    if pos_weight is None:
        n_neg = (labels == 0).sum().clamp(min=1).float()
        n_pos = (labels == 1).sum().clamp(min=1).float()
        pos_weight = (n_neg / n_pos).unsqueeze(0)
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight.to(logits.device)
    )


def l_sem(
    s_nodes: torch.Tensor,
    s_text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between node semantics and text semantics."""
    if s_nodes.shape[0] == 0:
        return s_nodes.new_zeros(1).squeeze()
    s_n = F.normalize(s_nodes, dim=-1)
    s_t = F.normalize(s_text, dim=-1)
    logits = (s_n @ s_t.T) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def l_elev(
    s_abs: torch.Tensor,
    s_children: List[torch.Tensor],
    w_children: List[float],
) -> torch.Tensor:
    """Elevation consistency loss using raw MSE (per CONSTRUCTION)."""
    if not s_children:
        return torch.tensor(0.0, device=s_abs.device)
    w_tensor = torch.tensor(w_children, device=s_abs.device, dtype=s_abs.dtype)
    w_tensor = w_tensor / w_tensor.sum().clamp(min=1e-6)
    s_stack = torch.stack(s_children, dim=0)
    s_target = (w_tensor.unsqueeze(1) * s_stack).sum(0)
    return F.mse_loss(s_abs.unsqueeze(0), s_target.unsqueeze(0))


def pretrain_loss(
    logits_boundary: torch.Tensor,
    labels_boundary: torch.Tensor,
    s_branch: Optional[torch.Tensor],
    s_text: Optional[torch.Tensor],
    s_abs_list: Optional[List[torch.Tensor]] = None,
    s_children_list: Optional[List[List[torch.Tensor]]] = None,
    w_children_list: Optional[List[List[float]]] = None,
    w_boundary: float = 1.0,
    w_sem: float = 0.5,
    w_elev: float = 0.2,
    tau_sem: float = 0.07,
) -> dict:
    """Convenience wrapper for weighted pretrain losses."""
    loss_boundary = l_boundary(logits_boundary, labels_boundary)

    loss_sem = torch.tensor(0.0, device=logits_boundary.device)
    if s_branch is not None and s_text is not None and s_branch.shape[0] > 0:
        loss_sem = l_sem(s_branch, s_text, temperature=tau_sem)

    loss_elev = torch.tensor(0.0, device=logits_boundary.device)
    if s_abs_list:
        for s_abs, s_ch, w_ch in zip(s_abs_list, s_children_list, w_children_list):
            loss_elev = loss_elev + l_elev(s_abs, s_ch, w_ch)
        loss_elev = loss_elev / max(len(s_abs_list), 1)

    total = w_boundary * loss_boundary + w_sem * loss_sem + w_elev * loss_elev
    return {
        "boundary": loss_boundary,
        "sem": loss_sem,
        "elev": loss_elev,
        "total": total,
    }


def l_recon(
    decoder: NodeReconDecoder,
    z_v_batch: torch.Tensor,
    patch_targets: torch.Tensor,
) -> torch.Tensor:
    """Optional reconstruction loss (ablation use)."""
    recon = decoder(z_v_batch)
    return F.mse_loss(recon, patch_targets)


def l_prog(
    s_nodes: dict,
    pairs: List[tuple],
    s_goal: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Optional progression monotonicity margin loss."""
    if not pairs:
        return torch.tensor(0.0, device=s_goal.device)
    loss = torch.tensor(0.0, device=s_goal.device)
    count = 0
    for anc_id, desc_id in pairs:
        if anc_id not in s_nodes or desc_id not in s_nodes:
            continue
        s_a = F.normalize(s_nodes[anc_id].unsqueeze(0), dim=-1)
        s_d = F.normalize(s_nodes[desc_id].unsqueeze(0), dim=-1)
        s_g = F.normalize(s_goal.unsqueeze(0), dim=-1)
        dist_a = 1.0 - (s_a * s_g).sum()
        dist_d = 1.0 - (s_d * s_g).sum()
        loss = loss + F.relu(dist_d - dist_a + gamma)
        count += 1
    return loss / max(count, 1)


def l_align(*args, **kwargs) -> torch.Tensor:
    """Deprecated compatibility stub."""
    return torch.tensor(0.0)


def tree_loss(*args, **kwargs) -> torch.Tensor:
    """Deprecated compatibility stub."""
    return torch.tensor(0.0)
