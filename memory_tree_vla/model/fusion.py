"""
CrossModalFusion — CONSTRUCTION.md Section 5.3.

Fuses the three information streams:
  - Z_M : memory tree readout     (N_M, d_ssm)
  - Z_V : SGMTS visual tokens     (P, d_visual)
  - g   : language CLS embedding  (d_lang,)
  - q   : proprioceptive state    (d_q,)

Output: Z_fused ∈ R^{N_ctx × d}  (context tokens for the action head)

Architecture:
  1. Project all streams to d (shared dim)
  2. Concatenate [Z_M_proj ; Z_V_proj] → Z_all  (N_M+P, d)
  3. Two rounds of cross-attention:
       Q=Z_all, K=V=Z_all  (self-attn to mix streams)
  4. Append proprioceptive token (projected q) as extra head token
  5. LayerNorm → output Z_fused
"""
import torch
import torch.nn as nn

from memory_tree_vla.model.attn import FlashMHA


# ================================================================
#  Pre-norm Fusion Block (self-attention + FFN)
# ================================================================

class FusionBlock(nn.Module):
    """Pre-norm Transformer block using FlashMHA for best GPU utilisation."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = FlashMHA(d, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class CrossModalFusion(nn.Module):
    """
    Args:
        d_ssm    : memory SSM output dim
        d_visual : SGMTS output dim
        d_lang   : LLM embedding dim
        d_q      : proprioceptive state dim
        d        : unified fusion dim (= output dim)
        n_heads  : number of attention heads
        n_layers : depth of self-attention stacks
        dropout  : dropout probability
    """

    def __init__(
        self,
        d_ssm: int = 256,
        d_visual: int = 256,
        d_lang: int = 896,
        d_q: int = 84,
        d: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # ── Input projections ────────────────────────────────────────
        self.mem_proj   = nn.Linear(d_ssm,   d)
        self.vis_proj   = nn.Linear(d_visual, d)
        self.lang_proj  = nn.Linear(d_lang,  d)   # language CLS → 1 token
        self.prop_proj  = nn.Linear(d_q,     d)   # proprioception → 1 token

        # ── Self-attention blocks (Flash Attention) ───────────────────
        self.blocks = nn.ModuleList([
            FusionBlock(d, n_heads, dropout) for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d)

    def forward(
        self,
        Z_M: torch.Tensor,      # (B, N_M, d_ssm)
        Z_V: torch.Tensor,      # (B, P,   d_visual)
        lang_g: torch.Tensor,   # (B, d_lang)
        q: torch.Tensor,        # (B, d_q)
    ) -> torch.Tensor:
        """Returns Z_fused : (B, N_M + P + 2, d)"""
        B = Z_M.shape[0]

        z_m  = self.mem_proj(Z_M)               # (B, N_M, d)
        z_v  = self.vis_proj(Z_V)               # (B, P,   d)
        z_l  = self.lang_proj(lang_g).unsqueeze(1)  # (B, 1, d)
        z_p  = self.prop_proj(q).unsqueeze(1)       # (B, 1, d)

        # Concatenate all tokens
        Z = torch.cat([z_l, z_m, z_v, z_p], dim=1)   # (B, 2+N_M+P, d)

        for blk in self.blocks:
            Z = blk(Z)
        Z = self.out_norm(Z)
        return Z   # (B, N_ctx, d)
