"""
MemoryTreeVLA 损失函数 — 预训练与训练完全分离

═══════════════════════════════════════════════════════════════════════
  预训练阶段（RoboCerebra）— 语义结构学习
═══════════════════════════════════════════════════════════════════════
  数据集: RoboCerebra（含子任务边界标注）
  可训练: JumpAwareHead, SGMTS, s_proj, TreeSSMReadout
  冻结:   LLM backbone, CrossModalFusion, FlowMatchingActionHead

  损失:
    L_boundary  — 动作突变点边界二分类（纯动作信号，自监督或 RoboCerebra 标注）
    L_sem       — 分支点语义对齐：分支时刻的 s_t 应与真实子任务描述接近（InfoNCE）
    L_elev      — 语义提升一致性（抽象节点是其子节点的语义概括）

  注意：FlowMatching 损失（L_flow）在此阶段不计算也不载入。

═══════════════════════════════════════════════════════════════════════
  训练第一阶段（LIBERO）— FlowMatching 热身
═══════════════════════════════════════════════════════════════════════
  冻结: LLM backbone（及预训练模块）
  可训练: CrossModalFusion, FlowMatchingActionHead
  损失: 仅 L_flow（来自 action_head 内部）

═══════════════════════════════════════════════════════════════════════
  训练第二阶段（LIBERO）— 全量微调
═══════════════════════════════════════════════════════════════════════
  可训练: 全部参数（可选 LoRA 微调 LLM）
  损失: L_flow（主）
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
#  NodeReconDecoder（被 MemoryTreeVLA 持有，在预训练中作为辅助解码器）
# ==================================================================

class NodeReconDecoder(nn.Module):
    """
    语义重建解码器: s_p (d,) → ŝ_children_mean (d,)

    预训练中可选使用，验证语义提升后抽象节点的表达质量。
    """

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


# ==================================================================
#  预训练损失集合 — 仅用于 RoboCerebra 预训练阶段
# ==================================================================

def l_boundary(
    logits: torch.Tensor,       # (B*T,) — JumpAwareHead 输出 logit
    labels: torch.Tensor,       # (B*T,) — 二值边界标签（0/1）
    pos_weight: Optional[torch.Tensor] = None,  # 正样本重加权因子
) -> torch.Tensor:
    """
    动作突变点边界二分类损失。

    标签生成（自监督）: y_t = 1 iff ||a_t - ā_act|| > γ · σ_act
    RoboCerebra 提供真实子任务边界标注时直接使用。

    正负样本比例严重不均衡（边界帧 ≤ 5%），用 pos_weight 补偿。
    """
    labels = labels.to(logits.device).float()
    if pos_weight is None:
        n_neg = (labels == 0).sum().clamp(min=1).float()
        n_pos = (labels == 1).sum().clamp(min=1).float()
        pos_weight = (n_neg / n_pos).unsqueeze(0)
    return F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=pos_weight.to(logits.device),
    )


def l_sem(
    s_nodes: torch.Tensor,     # (N, d)  — 分支点语义嵌入
    s_text: torch.Tensor,      # (N, d)  — 对应子任务的语言嵌入
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    分支点语义对齐损失（InfoNCE / NT-Xent）。

    分支点时刻的视觉语义嵌入 s_t 应与真实子任务描述的语言嵌入接近。
    仅在预训练阶段使用，不与 FlowMatching 混合。
    """
    if s_nodes.shape[0] == 0:
        return s_nodes.new_zeros(1).squeeze()
    s_n  = F.normalize(s_nodes, dim=-1)   # (N, d)
    s_t  = F.normalize(s_text,  dim=-1)   # (N, d)
    logits  = (s_n @ s_t.T) / temperature  # (N, N)
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits,   targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) * 0.5


def l_elev(
    s_abs: torch.Tensor,
    s_children: List[torch.Tensor],
    w_children: List[float],
) -> torch.Tensor:
    """
    语义提升一致性损失。

    抽象节点 s_abs 应接近其子节点的加权平均语义，确保语义提升是真实概括。
    """
    if not s_children:
        return torch.tensor(0.0, device=s_abs.device)
    w_tensor = torch.tensor(w_children, device=s_abs.device, dtype=s_abs.dtype)
    w_tensor = w_tensor / w_tensor.sum().clamp(min=1e-6)
    s_stack  = torch.stack(s_children, dim=0)           # (K, d)
    s_target = (w_tensor.unsqueeze(1) * s_stack).sum(0) # (d,)
    return F.mse_loss(
        F.normalize(s_abs.unsqueeze(0), dim=-1),
        F.normalize(s_target.unsqueeze(0), dim=-1),
    )


def pretrain_loss(
    logits_boundary: torch.Tensor,        # (N,)
    labels_boundary: torch.Tensor,        # (N,)
    s_branch: Optional[torch.Tensor],     # (M, d) — branch-point semantics
    s_text:   Optional[torch.Tensor],     # (M, d) — sub-task text embeddings
    s_abs_list: Optional[List[torch.Tensor]]    = None,
    s_children_list: Optional[List[List[torch.Tensor]]] = None,
    w_children_list: Optional[List[List[float]]]         = None,
    w_boundary: float = 1.0,
    w_sem:      float = 0.5,
    w_elev:     float = 0.2,
    tau_sem:    float = 0.07,
) -> dict:
    """
    预训练总导损失。仅在 RoboCerebra 预训练阶段调用，不依赖 FlowMatching。

    Returns
    -------
    losses : dict 含 total, boundary, sem, elev
    """
    loss_boundary = l_boundary(logits_boundary, labels_boundary)
    losses = {"boundary": loss_boundary}

    loss_sem = torch.tensor(0.0, device=logits_boundary.device)
    if s_branch is not None and s_text is not None and s_branch.shape[0] > 0:
        loss_sem = l_sem(s_branch, s_text, temperature=tau_sem)
    losses["sem"] = loss_sem

    loss_elev = torch.tensor(0.0, device=logits_boundary.device)
    if s_abs_list:
        for s_abs, s_ch, w_ch in zip(s_abs_list, s_children_list, w_children_list):
            loss_elev = loss_elev + l_elev(s_abs, s_ch, w_ch)
        loss_elev = loss_elev / max(len(s_abs_list), 1)
    losses["elev"] = loss_elev

    total = w_boundary * loss_boundary + w_sem * loss_sem + w_elev * loss_elev
    losses["total"] = total
    return losses


# ==================================================================
#  训练损失 — Phase 1 / Phase 2（仅 FlowMatching，不混合语义损失）
# ==================================================================

def l_recon(
    decoder: NodeReconDecoder,
    z_v_batch: torch.Tensor,
    patch_targets: torch.Tensor,
) -> torch.Tensor:
    """节点视觉重建 MSE（预训练阶段可选辅助损失）。"""
    recon = decoder(z_v_batch)
    return F.mse_loss(recon, patch_targets)


def l_prog(
    s_nodes: dict,
    pairs: List[tuple],
    s_goal: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    进度单调损失（仅施加于祖先-后代对，不跨分支约束）。
    Phase 2 全量微调时可选加入，权重设低。
    """
    if not pairs:
        return torch.tensor(0.0)
    loss  = torch.tensor(0.0, device=s_goal.device)
    count = 0
    for anc_id, desc_id in pairs:
        if anc_id not in s_nodes or desc_id not in s_nodes:
            continue
        s_a = F.normalize(s_nodes[anc_id].unsqueeze(0),  dim=-1)
        s_d = F.normalize(s_nodes[desc_id].unsqueeze(0), dim=-1)
        s_g = F.normalize(s_goal.unsqueeze(0),            dim=-1)
        dist_a = 1.0 - (s_a * s_g).sum()
        dist_d = 1.0 - (s_d * s_g).sum()
        loss = loss + F.relu(dist_d - dist_a + gamma)
        count += 1
    return loss / max(count, 1)


# 兼容旧导入（tree_loss / pretrain_loss 旧签名保留为 pretrain_loss）
def l_align(*args, **kwargs) -> torch.Tensor:
    """μ_t ↔ p_jump 对齐损失（已弃用，保留接口不报错）。"""
    return torch.tensor(0.0)


def tree_loss(*args, **kwargs) -> torch.Tensor:
    """旧版统一树损失（已弃用，保留接口不报错）。"""
    return torch.tensor(0.0)

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
#  Node visual reconstruction loss
# ==================================================================

class NodeReconDecoder(nn.Module):
    """
    Semantic reconstruction decoder: s_p (d,) → ŝ_children_mean (d,).

    CONSTRUCTION.md Section 3.6①:
        L_recon = Σ_{v_p has children} ||Dec_sem(s_p) - (1/|ch|) Σ s_i||²
    """

    def __init__(self, d: int, d_patch: int = None):
        """d_patch is accepted but ignored (kept for backwards compatibility)."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        # Cast input to match model weight dtype (e.g. bfloat16 under DeepSpeed)
        weight_dtype = next(self.parameters()).dtype
        return self.mlp(z_v.to(dtype=weight_dtype))


def l_recon(
    decoder: NodeReconDecoder,
    z_v_batch: torch.Tensor,       # (N, d)  — node visual embeddings
    patch_targets: torch.Tensor,   # (N, d_patch) — target mean-patch pixels
) -> torch.Tensor:
    """MSE reconstruction loss between decoded z_v and mean patch target."""
    recon = decoder(z_v_batch)
    return F.mse_loss(recon, patch_targets)


# ==================================================================
#  Semantic alignment loss (InfoNCE)
# ==================================================================

def l_sem(
    s_nodes: torch.Tensor,     # (N, d)  — node semantic embeddings
    s_text: torch.Tensor,      # (N, d)  — corresponding subtask embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / NT-Xent loss between visual node semantics and
    language subtask semantics.
    """
    s_n  = F.normalize(s_nodes, dim=-1)   # (N, d)
    s_t  = F.normalize(s_text,  dim=-1)   # (N, d)

    logits = (s_n @ s_t.T) / temperature   # (N, N)
    targets = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) * 0.5


# ==================================================================
#  Progression ordering loss
# ==================================================================

def l_prog(
    s_nodes: dict,            # {node_id: s_tensor (d,)}
    pairs: List[tuple],       # [(anc_id, desc_id), ...]
    s_goal: torch.Tensor,     # (d,)  — mean of leaf-node semantics
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Margin loss: distance(ancestor, goal) > distance(descendant, goal) + γ
    i.e., as we go deeper in the tree we should be closer to the task goal.
    """
    if not pairs:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0, device=s_goal.device)
    count = 0
    for anc_id, desc_id in pairs:
        if anc_id not in s_nodes or desc_id not in s_nodes:
            continue
        s_a  = F.normalize(s_nodes[anc_id].unsqueeze(0),  dim=-1)
        s_d  = F.normalize(s_nodes[desc_id].unsqueeze(0), dim=-1)
        s_g  = F.normalize(s_goal.unsqueeze(0),            dim=-1)

        # Distance = 1 – cosine similarity
        dist_a = 1.0 - (s_a * s_g).sum()
        dist_d = 1.0 - (s_d * s_g).sum()

        # Hinge: ancestor should be farther from goal than descendant
        loss = loss + F.relu(dist_d - dist_a + gamma)
        count += 1

    return loss / max(count, 1)


# ==================================================================
#  Elevation consistency loss
# ==================================================================

def l_elev(
    s_abs: torch.Tensor,          # (d,) — abstract node semantic
    s_children: List[torch.Tensor],
    w_children: List[float],
) -> torch.Tensor:
    """
    s_abs should match the weighted-mean of children semantics.
    MSE loss in normalised embedding space.
    """
    if not s_children:
        return torch.tensor(0.0)

    w_tensor = torch.tensor(w_children, device=s_abs.device)
    w_tensor = w_tensor / w_tensor.sum().clamp(min=1e-6)
    s_stack  = torch.stack(s_children, dim=0)               # (K, d)
    s_target = (w_tensor.unsqueeze(1) * s_stack).sum(0)     # (d,)

    return F.mse_loss(
        F.normalize(s_abs.unsqueeze(0), dim=-1),
        F.normalize(s_target.unsqueeze(0), dim=-1),
    )


# ==================================================================
#  Combined tree loss
# ==================================================================

def tree_loss(
    L_flow: torch.Tensor,
    L_recon: Optional[torch.Tensor] = None,
    L_sem: Optional[torch.Tensor] = None,
    L_prog: Optional[torch.Tensor] = None,
    L_elev_val: Optional[torch.Tensor] = None,
    w_flow: float = 1.0,
    w_recon: float = 0.5,
    w_sem: float = 0.5,
    w_prog: float = 0.3,
    w_elev: float = 0.2,
) -> torch.Tensor:
    total = w_flow * L_flow
    if L_recon is not None:
        total = total + w_recon * L_recon
    if L_sem is not None:
        total = total + w_sem * L_sem
    if L_prog is not None:
        total = total + w_prog * L_prog
    if L_elev_val is not None:
        total = total + w_elev * L_elev_val
    return total


# ==================================================================
#  Pre-training losses (boundary classification + alignment)
# ==================================================================

def l_boundary(
    logits: torch.Tensor,           # (N,) raw logits from JumpAwareHead
    y_act: torch.Tensor,            # (N,) float binary labels in [0,1]
    pos_weight: float = 5.0,        # upweight rare boundary frames
) -> torch.Tensor:
    """
    BCE loss for boundary classifier (CONSTRUCTION §6.3 Pre-training L_boundary).

    y_t^act = 1  iff  ||a_t − ā_act|| > γ · σ_act   (γ=1.5)
    """
    pw = logits.new_tensor([pos_weight])
    return F.binary_cross_entropy_with_logits(logits, y_act, pos_weight=pw)


def l_align(
    mu_t: torch.Tensor,    # (N,) soft-gate values ∈ [0,1]  (stop-gradient)
    p_jump: torch.Tensor,  # (N,) predicted jump probability ∈ [0,1]
) -> torch.Tensor:
    """
    Consistency loss: μ_t = 1 − p_jump.
    L_align = mean((μ_t.detach() + p_jump − 1)²)
    """
    return ((mu_t.detach() + p_jump - 1.0) ** 2).mean()


def pretrain_loss(
    L_boundary_val: torch.Tensor,
    L_sem_val: Optional[torch.Tensor]    = None,
    L_recon_val: Optional[torch.Tensor]  = None,
    L_elev_val: Optional[torch.Tensor]   = None,
    L_align_val: Optional[torch.Tensor]  = None,
    w_boundary: float = 1.0,
    w_sem: float      = 0.5,
    w_recon: float    = 0.3,
    w_elev: float     = 0.2,
    w_align: float    = 0.1,
) -> torch.Tensor:
    """Weighted sum of all pre-training losses."""
    total = w_boundary * L_boundary_val
    if L_sem_val is not None:
        total = total + w_sem * L_sem_val
    if L_recon_val is not None:
        total = total + w_recon * L_recon_val
    if L_elev_val is not None:
        total = total + w_elev * L_elev_val
    if L_align_val is not None:
        total = total + w_align * L_align_val
    return total

