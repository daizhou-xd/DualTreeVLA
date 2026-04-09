"""
CrossModalFusion — CONSTRUCTION.md Section 3.4.

Gated fusion of visual, memory, language, and proprioceptive streams.

Gate formula (CONSTRUCTION §3.4):
    g = σ(W_g[z_v; m_ctx; q; g_lang] + b_g) ∈ [0,1]^d
    f_fused = g ⊙ W_1[z_v; m_ctx] + (1−g) ⊙ W_2[q; g_lang] ∈ R^d

Inputs:
    z_v   : (B, d_visual) — mean-pooled SGMTS visual features
    m_ctx : (B, d_ssm)   — last token of TreeSSMReadout (m_ctx[-1])
    g_lang: (B, d_lang)  — LLM mean-pooled language embedding
    q     : (B, d_q)     — proprioceptive state

Output: f_fused ∈ R^{B × 1 × d}  (unsqueezed seq dim for action-head cross-attn)
"""
import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Gated single-vector fusion as specified in CONSTRUCTION.md §3.4.

    Args:
        d_ssm    : memory SSM output dim
        d_visual : SGMTS output dim
        d_lang   : LLM embedding dim
        d_q      : proprioceptive state dim
        d        : unified fusion dim (= output dim)
        n_heads  : kept for API compatibility (unused)
        n_layers : kept for API compatibility (unused)
        dropout  : kept for API compatibility (unused)
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

        gate_in = d_visual + d_ssm + d_q + d_lang
        self.W_g = nn.Linear(gate_in, d, bias=True)
        self.W_1 = nn.Linear(d_visual + d_ssm, d, bias=False)
        self.W_2 = nn.Linear(d_q + d_lang, d, bias=False)

    def forward(
        self,
        z_v: torch.Tensor,      # (B, d_visual) — mean-pooled visual features
        m_ctx: torch.Tensor,    # (B, d_ssm)    — TreeSSM last token
        g_lang: torch.Tensor,   # (B, d_lang)   — language embedding
        q: torch.Tensor,        # (B, d_q)      — proprioceptive state
    ) -> torch.Tensor:
        """Returns f_fused : (B, 1, d)  (seq dim kept for action-head cross-attn)"""
        gate_in = torch.cat([z_v, m_ctx, q, g_lang], dim=-1)   # (B, gate_in)
        g  = torch.sigmoid(self.W_g(gate_in))                   # (B, d)
        f1 = self.W_1(torch.cat([z_v, m_ctx], dim=-1))         # (B, d)
        f2 = self.W_2(torch.cat([q, g_lang], dim=-1))          # (B, d)
        f_fused = g * f1 + (1.0 - g) * f2                      # (B, d)
        return f_fused.unsqueeze(1)                              # (B, 1, d)

