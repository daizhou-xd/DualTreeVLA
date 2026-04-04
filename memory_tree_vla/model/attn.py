"""
Unified multi-head attention with automatic Flash Attention backend selection.

Priority (highest to lowest):
  1. flash_attn package     — fastest; requires separate CUDA compilation
                              (pip install flash-attn --no-build-isolation)
  2. F.scaled_dot_product_attention  — PyTorch ≥ 2.0 built-in; automatically
                              selects Flash Attention kernel on sm_80+ (A6000)
  3. Manual softmax attention — CPU / fallback

FlashMHA supports both self-attention and cross-attention.
Use is_causal=True instead of passing an explicit causal mask (faster).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Backend detection ────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_PKG = True
except ImportError:
    _flash_attn_func = None
    _FLASH_PKG = False

_SDPA = hasattr(F, "scaled_dot_product_attention")


def flash_attn_available() -> bool:
    return _FLASH_PKG


def sdpa_available() -> bool:
    return _SDPA


# ── Attention module ─────────────────────────────────────────────

class FlashMHA(nn.Module):
    """
    Drop-in multi-head attention with automatic Flash Attention selection.

    Args:
        d_model  : query / output embedding dim
        n_heads  : number of attention heads
        dropout  : attention dropout probability (only during training)
        d_kv     : key/value source dim (set for cross-attention; default = d_model)
        bias     : include bias in QKV / output projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        d_kv: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        d_kv = d_kv or d_model

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_kv,    d_model, bias=bias)
        self.v_proj   = nn.Linear(d_kv,    d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        q: torch.Tensor,                          # (B, Sq, d_model)
        k: torch.Tensor,                          # (B, Sk, d_kv)
        v: torch.Tensor,                          # (B, Sk, d_kv)
        attn_mask: Optional[torch.Tensor] = None, # additive mask (for SDPA / manual)
        is_causal: bool = False,                  # use causal masking (Flash-native)
    ) -> torch.Tensor:                            # (B, Sq, d_model)
        B, Sq = q.shape[:2]
        Sk    = k.shape[1]
        dp    = self.dropout if self.training else 0.0

        Q = self.q_proj(q).view(B, Sq, self.n_heads, self.head_dim)
        K = self.k_proj(k).view(B, Sk, self.n_heads, self.head_dim)
        V = self.v_proj(v).view(B, Sk, self.n_heads, self.head_dim)

        # ── Backend 1: flash_attn package ──────────────────────────
        if _FLASH_PKG and attn_mask is None and q.is_cuda:
            orig = Q.dtype
            if orig == torch.float32:
                Q, K, V = Q.bfloat16(), K.bfloat16(), V.bfloat16()
            # flash_attn_func: (B, S, nh, hd) → (B, S, nh, hd)
            out = _flash_attn_func(Q, K, V, dropout_p=dp, causal=is_causal)
            out = out.to(orig)                            # (B, Sq, nh, hd)

        # ── Backend 2: PyTorch SDPA (auto Flash Attn on Ampere+) ───
        elif _SDPA:
            # SDPA expects (B, nh, S, hd)
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask  = attn_mask,
                dropout_p  = dp,
                is_causal  = is_causal,
            )
            out = out.transpose(1, 2)                     # (B, Sq, nh, hd)

        # ── Backend 3: manual fallback ──────────────────────────────
        else:
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            scale = self.head_dim ** -0.5
            att = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, nh, Sq, Sk)
            if attn_mask is not None:
                att = att + attn_mask
            if is_causal:
                cm  = torch.triu(torch.ones(Sq, Sk, device=Q.device, dtype=torch.bool), diagonal=1)
                att = att.masked_fill(cm, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and dp > 0.0:
                att = F.dropout(att, p=dp)
            out = torch.matmul(att, V).transpose(1, 2)      # (B, Sq, nh, hd)

        out = out.contiguous().view(B, Sq, -1)               # (B, Sq, d_model)
        return self.out_proj(out)
