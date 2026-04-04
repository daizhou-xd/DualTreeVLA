"""
Flow Matching Action Head — CONSTRUCTION.md Section 5.4.

Adapted from Evo-1 (CVPR 2026) for single-manipulator (Franka, d_a=7).

Training : learn a velocity field v_θ(a_t, t, c) that transports
           noise → action by minimising the conditional flow matching
           objective: L_flow = E[‖v_θ(a_t, t, c) − (a_1 − a_0)‖²]
           where a_t = (1-t)a_0 + t·a_1,  a_0 ~ N(0,I),  a_1 = a_gt.

Inference : integrate a_t from t=0→1 with N_ode Euler steps.

Context c : concatenation of fused latent tokens from CrossModalFusion
            (shape B × N_ctx × d_ctx).  The head reads these via
            cross-attention inside each Transformer block.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_tree_vla.model.attn import FlashMHA


# ================================================================
#  Sinusoidal time embedding
# ================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) or (B,H_a) float in [0,1] → (B, d_model) or (B,H_a,d_model)"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        flat_t  = t.view(-1, 1) * freqs.view(1, -1)   # (..., half)
        emb     = torch.cat([flat_t.sin(), flat_t.cos()], dim=-1)
        emb     = self.mlp(emb)
        return emb.view(*t.shape, self.d_model)


# ================================================================
#  Causal DiT block with cross-attention context
# ================================================================

class FlowBlock(nn.Module):
    """Single Transformer block of the action-head DiT."""

    def __init__(self, d_model: int, n_heads: int, d_ctx: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn  = FlashMHA(d_model, n_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = FlashMHA(d_model, n_heads, dropout=dropout, d_kv=d_ctx)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        # AdaLN scale & shift from time embedding
        self.adaln_proj = nn.Linear(d_model, 6 * d_model)   # 3×(shift, scale) for norm1,2,3

    def forward(
        self,
        x: torch.Tensor,         # (B, H_a, d_model)
        t_emb: torch.Tensor,     # (B, d_model)
        ctx: torch.Tensor,       # (B, N_ctx, d_ctx)
    ) -> torch.Tensor:
        # AdaLN
        adaln = self.adaln_proj(t_emb).unsqueeze(1).chunk(6, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = adaln

        def modulate(norm, inp, shift, scale):
            return norm(inp) * (1 + scale) + shift

        # Self-attention (causal for action sequence)
        h  = modulate(self.norm1, x, shift1, scale1)
        x  = x + self.self_attn(h, h, h, is_causal=True)

        # Cross-attention to fused context
        h  = modulate(self.norm2, x, shift2, scale2)
        x  = x + self.cross_attn(h, ctx, ctx)

        # Feed-forward
        h  = modulate(self.norm3, x, shift3, scale3)
        x  = x + self.ff(h)
        return x


# ================================================================
#  Flow Matching Action Head
# ================================================================

class FlowMatchingActionHead(nn.Module):
    """
    Parameters
    ----------
    d_a      : action dimension (7 for Franka)
    H_a      : action prediction horizon
    d_model  : internal DiT dim
    n_layers : number of FlowBlock layers
    n_heads  : attention heads
    d_ctx    : context token dim (from CrossModalFusion)
    N_ode    : Euler integration steps at inference
    """

    def __init__(
        self,
        d_a: int = 7,
        H_a: int = 16,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ctx: int = 256,
        N_ode: int = 20,
    ):
        super().__init__()
        self.d_a    = d_a
        self.H_a    = H_a
        self.N_ode  = N_ode

        # Project noisy action sequence to d_model
        self.a_in   = nn.Linear(d_a, d_model)
        self.pos_emb = nn.Embedding(H_a, d_model)

        self.t_emb   = TimestepEmbedding(d_model)

        self.blocks  = nn.ModuleList([
            FlowBlock(d_model, n_heads, d_ctx) for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_a)

        # causal masking is handled natively by FlashMHA via is_causal=True

    # -------------------------------------------------------------- #
    #  Training: return velocity prediction                            #
    # -------------------------------------------------------------- #

    def forward(
        self,
        a_noisy: torch.Tensor,   # (B, H_a, d_a) — interpolated a_t
        t: torch.Tensor,         # (B,) float in [0,1]
        ctx: torch.Tensor,       # (B, N_ctx, d_ctx)
    ) -> torch.Tensor:
        """Returns predicted velocity v_θ(a_t, t, c) of shape (B, H_a, d_a)."""
        B, H, _ = a_noisy.shape
        positions = torch.arange(H, device=a_noisy.device)

        x     = self.a_in(a_noisy) + self.pos_emb(positions).unsqueeze(0)
        t_e   = self.t_emb(t)      # (B, d_model)

        for blk in self.blocks:
            x = blk(x, t_e, ctx)

        x = self.out_norm(x)
        v = self.out_proj(x)      # (B, H_a, d_a)
        return v

    # -------------------------------------------------------------- #
    #  Inference: Euler ODE integration                                #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,       # (B, N_ctx, d_ctx)
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Integrate from t=0 (pure noise) to t=1 (clean action).
        Returns a_1 of shape (B, H_a, d_a).
        """
        B = ctx.shape[0]
        if device is None:
            device = ctx.device

        # a^(0) ~ Uniform[-1, 1]  (CONSTRUCTION Section 5.4.8)
        a_t = torch.empty(B, self.H_a, self.d_a, device=device).uniform_(-1.0, 1.0)
        dt  = 1.0 / self.N_ode

        for step in range(self.N_ode):
            t_val = step / self.N_ode
            t     = torch.full((B,), t_val, device=device)
            v     = self(a_t, t, ctx)
            a_t   = a_t + dt * v

        return a_t   # (B, H_a, d_a)

    # -------------------------------------------------------------- #
    #  Training loss                                                   #
    # -------------------------------------------------------------- #

    def flow_loss(
        self,
        a_gt: torch.Tensor,      # (B, H_a, d_a)
        ctx: torch.Tensor,       # (B, N_ctx, d_ctx)
    ) -> torch.Tensor:
        """
        Conditional flow matching loss — CONSTRUCTION.md Section 5.4.7.

        Time sampling  : τ ~ Beta(2, 2) clipped to [0.02, 0.98]
        Noise          : ε ~ Uniform[-1, 1]^{H_a × d_a}
        Interpolation  : a_τ = (1 - τ)·ε + τ·a*
        Velocity target: v* = a* - ε
        Loss           : E[‖v_θ(a_τ, τ, C) - v*‖²]
        """
        B = a_gt.shape[0]
        device = a_gt.device

        # τ ~ Beta(2, 2) ∩ [0.02, 0.98]
        beta_dist = torch.distributions.Beta(
            torch.tensor(2.0, device=device),
            torch.tensor(2.0, device=device),
        )
        t = beta_dist.sample((B,)).clamp(0.02, 0.98)   # (B,)

        # ε ~ Uniform[-1, 1]
        a_noise  = torch.empty_like(a_gt).uniform_(-1.0, 1.0)
        t_expand = t.view(B, 1, 1)

        a_t  = (1.0 - t_expand) * a_noise + t_expand * a_gt
        v_gt = a_gt - a_noise

        v_pred = self(a_t, t, ctx)                       # (B, H_a, d_a)
        return F.mse_loss(v_pred, v_gt)
