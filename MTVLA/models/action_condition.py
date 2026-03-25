"""
Action Condition Builder for MemoryTreeVLA.

Fuses Action-LLM output tokens with robot proprioceptive state tokens into a
unified condition sequence consumed by ``FlowMatchingActionHead``.

Design inspired by Evo-1 ``InternVL3Embedder``
(https://github.com/MINT-SJTU/Evo-1/blob/main/Evo_1/model/internvl3/internvl3_embedder.py):
  * InternVL3Embedder  : ViT patches + text prompt  → LLM hidden states
  * ActionConditionBuilder: LLM hidden states + proprioceptive state
                            → action-condition tokens

Data-flow
---------
                ┌─────────────────────────────────────────┐
  LLM tokens    │  (B, L, D_llm)                          │
  ─────────────►│  LLMTokenProjector  → (B, L, D)         │
                │                            ↓             │
                │  + segment embedding [0…0] ↓             │
                │                            │             │
  Robot state   │  (B, state_dim)            │             │
  ─────────────►│  RobotStateEncoder → (B, N_s, D)        │
                │                            ↓             │
                │  + segment embedding [1…1] ↓             │
                │                            │             │
                │  cat( LLM tokens, state tokens ) → ctx   │
                │  (B, L + N_s, D)                         │
                └─────────────────────────────────────────┘

Segment embeddings let the cross-attention inside ``FlowMatchingActionHead``
distinguish "language context" from "proprioceptive state" — analogous to how
InternVL3Embedder uses image/text segment masks inside InternVL3.

Typical dims for MemoryTreeVLA:
  D_llm  = 896   (Qwen2.5-0.5B hidden size)
  D      = 512   (action head embed_dim)
  state  = 15    (7 joint pos + 7 joint vel + 1 gripper)  or 8 (pos + gripper)
  N_s    = 1..4  num_state_tokens   (default 1)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sinusoidal(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return ``(1, seq_len, dim)`` sinusoidal positional encodings."""
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float, device=device)
        * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, dim)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class LLMTokenProjector(nn.Module):
    """Projects LLM hidden states from ``D_llm`` to the action head's ``embed_dim``.

    Mirrors the "connector" in VLA models that bridges the vision encoder and
    the LLM (e.g. a linear or MLP projector).  Here it bridges the LLM output
    and the flow-matching action head.

    Args:
        llm_dim:    Hidden dimension of the language model (e.g. 896 for
                    Qwen2.5-0.5B).
        embed_dim:  Target dimension for the action head (default 512).
        use_mlp:    If ``True``, use a 2-layer GELU MLP; otherwise a single
                    linear layer (cheaper, good enough when dimensions are
                    already well-aligned).
    """

    def __init__(self, llm_dim: int, embed_dim: int, use_mlp: bool = True) -> None:
        super().__init__()
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(llm_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        else:
            self.proj = nn.Linear(llm_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, llm_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llm_tokens: ``(B, L, D_llm)`` – last hidden layer of the action LLM.

        Returns:
            ``(B, L, embed_dim)``
        """
        return self.norm(self.proj(llm_tokens))


class RobotStateEncoder(nn.Module):
    """Encodes a flat proprioceptive state vector into ``num_tokens`` dense tokens.

    Inspired by how InternVL3Embedder projects ViT cls/patch features through an
    MLP before fusing with text tokens — we do the same for grip/joint state.

    The state is first linearly expanded to ``num_tokens * embed_dim``, then
    reshaped to ``(B, num_tokens, embed_dim)`` so the action head can perform
    cross-attention over multiple state tokens rather than a single squeezed
    vector.

    Args:
        state_dim:   Dimension of the proprioceptive state vector.  Common
                     choices for a 7-DoF arm:

                     * 8  = 7 joint positions + 1 gripper
                     * 14 = 7 joint pos + 7 joint vel
                     * 15 = 7 joint pos + 7 joint vel + 1 gripper

        embed_dim:   Token embedding dimension (should match the action head).
        num_tokens:  How many tokens the state expands into.  More tokens give
                     the action head finer-grained access; 1–4 is typical.
        hidden_dim:  Hidden layer size in the 3-layer MLP encoder.  Defaults
                     to ``2 * embed_dim``.
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        num_tokens: int = 1,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        _hidden = hidden_dim or embed_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, _hidden),
            nn.SiLU(),
            nn.Linear(_hidden, _hidden),
            nn.SiLU(),
            nn.Linear(_hidden, num_tokens * embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: ``(B, state_dim)`` proprioceptive state.

        Returns:
            ``(B, num_tokens, embed_dim)``
        """
        B = state.size(0)
        out = self.encoder(state)                    # (B, num_tokens * embed_dim)
        out = out.view(B, self.num_tokens, self.embed_dim)  # (B, N_s, D)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class ActionConditionBuilder(nn.Module):
    """Builds the condition sequence for ``FlowMatchingActionHead``.

    Concatenates projected LLM output tokens with encoded proprioceptive state
    tokens.  Two learnable **segment embeddings** (``seg_llm`` and
    ``seg_state``) are broadcast-added to tell the action head which tokens
    come from the language branch and which from the proprioception branch —
    exactly analogous to how InternVL3Embedder uses image/text masking to
    preserve modality identity through the LLM layers.

    A sinusoidal positional encoding is applied *within* each segment before
    adding the segment embedding.

    Args:
        llm_dim:        Hidden dim of the action LLM (e.g. 896 for Qwen2.5-0.5B).
        state_dim:      Dimension of the proprioceptive state vector.
        embed_dim:      Shared embedding dimension for the action head (default 512).
        num_state_tokens: Number of tokens the state vector is split into (default 1).
        state_hidden_dim: Hidden size for ``RobotStateEncoder`` MLP.
        use_mlp_proj:   If ``True``, use a 2-layer MLP for the LLM projector.
        llm_seq_pool:   If not ``None``, pool LLM tokens down to this many tokens
                        *before* concatenating (reduces context length).  Valid
                        values: ``"mean"`` (global average pool → 1 token) or an
                        integer K (keep last K tokens, mimicking the Evo-1
                        practice of using the last hidden state).

    Shapes (forward):
        llm_tokens  : ``(B, L, D_llm)``
        state       : ``(B, state_dim)``
        → context   : ``(B, L_ctx, embed_dim)``
          where ``L_ctx = L_pool + num_state_tokens``
          and    ``L_pool = 1 if llm_seq_pool=='mean' else
                            K if isinstance(llm_seq_pool, int) else L``
    """

    def __init__(
        self,
        llm_dim: int,
        state_dim: int,
        embed_dim: int = 512,
        num_state_tokens: int = 1,
        state_hidden_dim: Optional[int] = None,
        use_mlp_proj: bool = True,
        llm_seq_pool: Optional[int | str] = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.llm_seq_pool = llm_seq_pool

        # ---- LLM projector  (D_llm → embed_dim) ----
        self.llm_proj = LLMTokenProjector(
            llm_dim=llm_dim,
            embed_dim=embed_dim,
            use_mlp=use_mlp_proj,
        )

        # ---- Proprioceptive state encoder  (state_dim → N_s × embed_dim) ----
        self.state_encoder = RobotStateEncoder(
            state_dim=state_dim,
            embed_dim=embed_dim,
            num_tokens=num_state_tokens,
            hidden_dim=state_hidden_dim,
        )

        # ---- Segment embeddings  (one per modality) ----
        # Shape: (1, 1, embed_dim) — broadcast over batch and sequence dims.
        self.seg_llm   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seg_state = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.seg_llm,   std=0.02)
        nn.init.trunc_normal_(self.seg_state, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool_llm(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply optional sequence pooling to LLM tokens.

        Args:
            tokens: ``(B, L, embed_dim)`` projected LLM tokens.

        Returns:
            ``(B, L_pool, embed_dim)``
        """
        p = self.llm_seq_pool
        if p is None:
            return tokens
        if p == "mean":
            return tokens.mean(dim=1, keepdim=True)           # (B, 1, D)
        if isinstance(p, int):
            return tokens[:, -p:, :]                          # (B, K, D)
        raise ValueError(f"llm_seq_pool must be None, 'mean', or an int; got {p!r}")

    @staticmethod
    def _add_pos_enc(tokens: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to a token sequence."""
        pe = _make_sinusoidal(tokens.size(1), tokens.size(2), tokens.device)
        return tokens + pe.to(tokens.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        llm_tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Build the action-condition sequence.

        Args:
            llm_tokens: ``(B, L, D_llm)`` – last hidden layer of the action
                LLM (Qwen2.5-0.5B) *after* processing the fused multimodal
                context.  This typically comes from
                ``action_llm(..., output_hidden_states=True).hidden_states[-1]``.
            state: ``(B, state_dim)`` – proprioceptive robot state vector.
                For a 7-DoF arm: ``[q_1..q_7, dq_1..dq_7, gripper]`` or any
                subset thereof; must match ``state_dim`` given at construction.

        Returns:
            ``(B, L_ctx, embed_dim)`` unified condition tokens ready to be
            passed to ``FlowMatchingActionHead`` as ``fused_tokens``.
        """
        # 1. Project LLM tokens: (B, L, D_llm) → (B, L, D)
        llm_emb = self.llm_proj(llm_tokens)

        # 2. Optional sequence pooling
        llm_emb = self._pool_llm(llm_emb)              # (B, L_pool, D)

        # 3. Encode proprioceptive state: (B, state_dim) → (B, N_s, D)
        state_emb = self.state_encoder(state)           # (B, N_s, D)

        # 4. Add intra-segment positional encodings
        llm_emb   = self._add_pos_enc(llm_emb)
        state_emb = self._add_pos_enc(state_emb)

        # 5. Add segment embeddings to preserve modality identity
        llm_emb   = llm_emb   + self.seg_llm            # broadcast (1,1,D)
        state_emb = state_emb + self.seg_state

        # 6. Concatenate along sequence dimension → (B, L_pool + N_s, D)
        ctx = torch.cat([llm_emb, state_emb], dim=1)
        return ctx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Embedding dimension of the output context sequence."""
        return self.embed_dim

    def context_length(self, llm_seq_len: int) -> int:
        """Return the total context length for a given LLM sequence length.

        Useful for pre-allocating attention masks or KV caches.
        """
        p = self.llm_seq_pool
        if p is None:
            l_pool = llm_seq_len
        elif p == "mean":
            l_pool = 1
        elif isinstance(p, int):
            l_pool = min(p, llm_seq_len)
        else:
            l_pool = llm_seq_len
        return l_pool + self.state_encoder.num_tokens
