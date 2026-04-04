"""
JumpAwareHead — 纯动作突变检测跳变感知头

设计原则：
    动作突变点 = 分支点。JumpAwareHead 只消费动作序列，不接触任何语义特征。
    语义对齐是预训练阶段独立的 L_sem 损失负责，与跳变判决完全解耦。

Architecture
------------
1. 动作序列编码器（1-layer Mamba SSM）：
     seq = [a_0, ..., a_{L-1}, a_new]   shape (B, L+1, d_a)
     → Linear embed → d_inner
     → 选择性 SSM 顺序扫描（ZOH 离散化）
     → h_ctx = 末尾隐状态              shape (B, d_inner)

2. 轻量线性分类头（直接作用于 h_ctx，无语义/语言特征）：
     logit = W_cls · h_ctx + b_cls       shape (B, 1)

整体参数量约 0.05M。

Inputs
------
A_act  (B, L, d_a)  — 活跃节点动作历史（尾部截断 ≤ max_len）
a_new  (B, d_a)     — 当前时刻动作

Output
------
p_jump (B,)   ∈ [0,1]  — 跳变概率（≥0.5 → 创建分支节点）
logit  (B,)            — 原始 logit（供 BCEWithLogitsLoss 使用）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpAwareHead(nn.Module):
    """
    跳变感知头 — 纯动作 Mamba SSM + 轻量线性分类头。

    不接收任何语义/语言输入；语义对齐由预训练阶段的 L_sem 独立处理。
    纯动作突变点 = 分支点。

    Args
    ----
    d_a      : action dim（默认 7）
    d_inner  : Mamba inner dim（默认 64）
    d_state  : SSM state dim（默认 16）
    max_len  : 活跃节点历史最大长度（尾部截断，默认 64）
    """

    def __init__(
        self,
        d_a: int     = 7,
        d_inner: int = 64,
        d_state: int = 16,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.max_len  = max_len

        # ── Action embedding ─────────────────────────────────────────
        self.act_embed = nn.Linear(d_a, d_inner)

        # ── Mamba selective-SSM parameters (1 layer, sequential) ────
        self.W_delta  = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj   = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj   = nn.Linear(d_inner, d_state, bias=False)
        self.D        = nn.Parameter(torch.ones(d_inner))

        # Fixed A — S4D real init
        A_log_init = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float)
            .unsqueeze(0).expand(d_inner, -1)
        )
        self.A_log = nn.Parameter(A_log_init.clone())   # (d_inner, d_state)

        self.ssm_norm = nn.LayerNorm(d_inner)

        # ── 轻量线性分类头（直接作用于 h_ctx，无语义特征）─────────────
        self.classifier = nn.Linear(d_inner, 1)

        self._init_weights()

    # ---------------------------------------------------------------- #

    def _init_weights(self):
        """Initialize Δ bias so softplus(b) falls in [dt_min, dt_max]."""
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.W_delta.bias.copy_(inv_dt)

    # ---------------------------------------------------------------- #

    def _mamba_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        1-layer Mamba selective SSM sequential scan.

        x      : (B, L, d_inner)
        returns: (B, L, d_inner)  — same shape, final token = h_ctx
        """
        B, L, _ = x.shape
        dtype = x.dtype

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative-definite

        delta = F.softplus(self.W_delta(x))              # (B, L, d_inner)
        Bm    = self.B_proj(x)                           # (B, L, d_state)
        Cm    = self.C_proj(x)                           # (B, L, d_state)

        # ZOH discretize: Ā_t = exp(Δ_t · A),  B̄_t = Δ_t · B_t
        # shapes after einsum: (B, L, d_inner, d_state)
        A_f    = A.to(delta.dtype)
        dA     = torch.einsum("bld,ds->blds", delta, A_f)  # (B,L,d_inner,d_state)
        A_bar  = torch.exp(dA)
        Bm_f   = Bm.to(delta.dtype)
        B_bar  = torch.einsum("bld,bls->blds", delta, Bm_f)  # (B,L,d_inner,d_state)

        # Sequential scan (causal — h_t depends only on past)
        h   = x.new_zeros(B, self.d_inner, self.d_state)
        ys  = []
        D_f = self.D.to(delta.dtype)
        x_f = x.to(delta.dtype)
        for t in range(L):
            h   = A_bar[:, t] * h + B_bar[:, t] * x_f[:, t, :, None]  # (B,d_inner,d_state)
            # y_t = Σ_s C_s * h_s + D * x_t
            y_t = (Cm[:, t, None, :].to(delta.dtype) * h).sum(-1)      # (B, d_inner)
            y_t = y_t + D_f * x_f[:, t]
            ys.append(y_t)

        out = torch.stack(ys, dim=1).to(dtype)   # (B, L, d_inner)
        return self.ssm_norm(out)

    # ---------------------------------------------------------------- #

    def forward(
        self,
        A_act: torch.Tensor,       # (B, L, d_a)   active-node action history
        a_new: torch.Tensor,       # (B, d_a)       incoming new action
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        仅消费动作序列，检测动作模式是否发生结构性突变。

        Returns
        -------
        p_jump : (B,) ∈ [0,1]  — 跳变概率（≥0.5 → 创建分支节点）
        logit  : (B,)          — 原始 logit（用于 BCEWithLogitsLoss）
        """
        w_dtype = next(self.parameters()).dtype

        # 尾部截断：只保留最近 max_len 步历史
        if A_act.shape[1] > self.max_len:
            A_act = A_act[:, -self.max_len:]

        # 构建序列：[历史动作, 当前动作] → (B, L+1, d_a)
        seq = torch.cat([A_act, a_new.unsqueeze(1)], dim=1).to(w_dtype)

        # 动作嵌入 → (B, L+1, d_inner)
        x = self.act_embed(seq)

        # Mamba 扫描 → 末尾隐状态作为轨迹上下文
        out   = self._mamba_scan(x)
        h_ctx = out[:, -1, :]                           # (B, d_inner)

        # 纯线性分类（不涉及任何语义特征）
        logit = self.classifier(h_ctx).squeeze(-1)      # (B,)
        return torch.sigmoid(logit), logit


# 向后兼容别名
SemanticJumpHead = JumpAwareHead
