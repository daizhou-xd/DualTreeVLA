"""
Flow Matching Action Head for MemoryTreeVLA.

Adapted from Evo-1 (MINT-SJTU/Evo-1, NeurIPS 2024).
Reference: https://github.com/MINT-SJTU/Evo-1/tree/main/Evo_1/model/action_head

Architecture overview:
  1. Context tokens  : Z_fused (B, L, D) from MultimodalMamba + optional
                       proprioceptive state embedding
  2. Action tokens   : noisy action sequence encoded via MultiEmbodimentActionEncoder
                       (B, horizon, D)
  3. Denoiser        : N × BasicTransformerBlock  – action tokens cross-attend
                       to context tokens conditioned on flow-time embedding
  4. Output head     : CategorySpecificMLP → predicted velocity field

Training loss (Conditional Flow Matching – CFM):
  Sample t ~ Beta(2,2) clipped to [0.02, 0.98].
  Interpolate: a_t = (1 - t) * noise + t * a_gt
  Target velocity: v* = a_gt - noise   (straight-path CFM)
  Loss: MSE(v_pred, v*)

Inference (Euler integration):
  Start from noise a_0 ~ Uniform[-1, 1].
  For i in range(N_steps): a += (1/N) * v_pred(a, t=i/N)

Differences from Evo-1:
  * Input is Z_fused (B, L, D) from MultimodalMamba.
  * Default per_action_dim=7  (Δx, Δy, Δz, Δrx, Δry, Δrz, gripper).
  * Default horizon=16.
  * Debug print statements removed.
  * All lazy module creation moved into __init__ for cleaner forward logic.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used for flow-time embeddings.

    Pre-computes a (1, max_len, dim) buffer and returns a slice of it.
    Grows dynamically when ``seq_len`` exceeds ``max_len``.
    """

    def __init__(self, dim: int, max_len: int = 1000) -> None:
        super().__init__()
        self.register_buffer("pe", self._make_pe(dim, max_len))

    @staticmethod
    def _make_pe(dim: int, length: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return ``(1, seq_len, dim)`` positional encodings."""
        if seq_len > self.pe.size(1):  # type: ignore[attr-defined]
            dim = self.pe.size(2)       # type: ignore[attr-defined]
            new_pe = self._make_pe(dim, seq_len)
            self.pe = new_pe.to(self.pe.device)  # type: ignore[attr-defined]
        return self.pe[:, :seq_len, :]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Category-specific (multi-embodiment) linear / MLP
# ---------------------------------------------------------------------------

class CategorySpecificLinear(nn.Module):
    """Linear layer with optional per-embodiment weights.

    When ``num_categories <= 1`` this reduces to a standard ``nn.Linear``.
    Otherwise, a separate weight matrix is maintained for each category and
    selected at runtime via ``category_id``.

    Args:
        in_dim         : input feature dimension.
        out_dim        : output feature dimension.
        num_categories : number of embodiment categories.
    """

    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1) -> None:
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor) -> torch.Tensor:
        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        if category_id.dim() == 0:
            cid = int(category_id.item())
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
            cat_ids = category_id.view(-1)
            weight_sel = self.weight[cat_ids]    # (B, in, out)
            bias_sel = self.bias[cat_ids]        # (B, out)
            out = torch.bmm(x_flat.unsqueeze(1), weight_sel).squeeze(1) + bias_sel

        return out.view(*orig_shape[:-1], out.shape[-1])


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights (ReLU activation)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_categories: int = 1,
    ) -> None:
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x, category_id)), category_id)


# ---------------------------------------------------------------------------
# Multi-embodiment action encoder
# ---------------------------------------------------------------------------

class MultiEmbodimentActionEncoder(nn.Module):
    """Encode a chunk of actions into transformer-compatible tokens.

    Projects each per-step action vector to ``embed_dim`` via a 3-layer MLP,
    then adds sinusoidal position encodings over the horizon.

    Args:
        action_dim     : per-step action dimension (``per_action_dim``).
        embed_dim      : output embedding dimension.
        hidden_dim     : MLP hidden dimension.
        horizon        : action chunk length.
        num_categories : number of embodiment categories.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
        horizon: int,
        num_categories: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim

        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        action_seq: torch.Tensor,        # (B, horizon, action_dim)
        category_id: torch.LongTensor,   # (B,) or scalar
    ) -> torch.Tensor:                   # (B, horizon, embed_dim)
        B, H, D = action_seq.shape
        assert H == self.horizon

        x = action_seq.reshape(B * H, D)
        cat_ids = (
            category_id.repeat(H * B)
            if category_id.dim() == 0
            else category_id.unsqueeze(1).repeat(1, H).reshape(B * H)
        )

        x = self.activation(self.W1(x, cat_ids))

        pos = self.pos_enc(H).to(x.device).repeat(B, 1, 1).reshape(B * H, -1)
        x = x + pos
        x = self.activation(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x.view(B, H, self.embed_dim)


# ---------------------------------------------------------------------------
# Transformer denoising block
# ---------------------------------------------------------------------------

class BasicTransformerBlock(nn.Module):
    """Cross-attention transformer block for action denoising.

    Action tokens attend to context tokens (Z_fused + optional state embedding)
    conditioned on the flow-time embedding via additive bias.

    Args:
        embed_dim  : token dimension.
        num_heads  : number of attention heads.
        hidden_dim : feed-forward hidden dimension.
        dropout    : attention dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,   # (B, H, D) – queries
        context_tokens: torch.Tensor,  # (B, L, D) – keys & values
        time_emb: torch.Tensor,        # (B, D)    – flow-time bias
    ) -> torch.Tensor:                 # (B, H, D)
        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)
        x = action_tokens + attn_out

        x2 = self.norm2(x)
        x2 = x2 + time_emb.unsqueeze(1)   # broadcast over horizon
        return x + self.ff(x2)


# ---------------------------------------------------------------------------
# FlowMatchingActionHead
# ---------------------------------------------------------------------------

class FlowMatchingActionHead(nn.Module):
    """Conditional Flow Matching action head for MemoryTreeVLA.

    Consumes the fused multimodal representation ``Z_fused`` (output of
    MultimodalMamba) and optionally a proprioceptive state vector, then
    generates a de-noised action chunk via learned flow matching.

    Args:
        embed_dim              : dimension of ``Z_fused`` tokens (must match
                                 MultimodalMamba output dim).
        hidden_dim             : MLP / transformer hidden dim.
        per_action_dim         : action dimension per time-step (default 7:
                                 Δx, Δy, Δz, Δrx, Δry, Δrz, gripper).
        horizon                : action chunk length.
        num_heads              : transformer attention heads.
        num_layers             : number of BasicTransformerBlock layers.
        dropout                : attention dropout.
        num_inference_timesteps: Euler integration steps at inference.
        num_categories         : number of embodiment categories (>1 activates
                                 multi-embodiment mode).
        state_dim              : proprioceptive state dimension; ``None``
                                 disables the state encoder.
        state_hidden_dim       : hidden dim of the state MLP encoder.

    Inputs (forward / training):
        fused_tokens   : ``(B, L, D)`` – MultimodalMamba output.
        state          : ``(B, state_dim)`` – optional proprio state.
        actions_gt     : ``(B, horizon, per_action_dim)`` – ground-truth actions
                         for training.  Pass ``None`` to run inference.
        embodiment_id  : ``(B,)`` – category indices (default: all zeros).
        action_mask    : ``(B, horizon, per_action_dim)`` float mask – zero out
                         inactive action dimensions.

    Returns (training):
        pred_velocity  : ``(B, action_dim)`` – predicted velocity for CFM loss.
        noise          : ``(B, horizon, per_action_dim)`` – sampled noise.

    Returns (inference, actions_gt=None):
        actions        : ``(B, action_dim)`` – predicted flat action vector.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        per_action_dim: int = 7,
        horizon: int = 16,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.0,
        num_inference_timesteps: int = 20,
        num_categories: int = 1,
        state_dim: Optional[int] = None,
        state_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        action_dim = per_action_dim * horizon
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_dim = action_dim

        self.config = SimpleNamespace(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            per_action_dim=per_action_dim,
            action_dim=action_dim,
            horizon=horizon,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_inference_timesteps=num_inference_timesteps,
            num_categories=num_categories,
            state_dim=state_dim,
            state_hidden_dim=state_hidden_dim or embed_dim,
        )

        # ---- time / noise embedding ----
        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        # ---- action encoder ----
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=per_action_dim,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            horizon=horizon,
            num_categories=num_categories,
        )

        # ---- denoising transformer ----
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=embed_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)

        # ---- pooling + output MLP ----
        self.seq_pool_proj = nn.Linear(horizon * embed_dim, embed_dim)
        self.mlp_head = CategorySpecificMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_categories=num_categories,
        )

        # ---- optional state encoder ----
        self.state_encoder: Optional[CategorySpecificMLP] = None
        if state_dim is not None:
            self.state_encoder = CategorySpecificMLP(
                input_dim=state_dim,
                hidden_dim=state_hidden_dim or embed_dim,
                output_dim=embed_dim,
                num_categories=num_categories,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _default_embodiment_id(self, B: int, device: torch.device) -> torch.LongTensor:
        return torch.zeros(B, dtype=torch.long, device=device)  # type: ignore[return-value]

    def _build_context(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor],
        embodiment_id: torch.LongTensor,
    ) -> torch.Tensor:
        """Concatenate Z_fused with optional state embedding."""
        ctx = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)  # (B,1,D)
            ctx = torch.cat([ctx, state_emb], dim=1)
        return ctx

    def _denoise(
        self,
        action_tokens: torch.Tensor,   # (B, H, D)
        context_tokens: torch.Tensor,  # (B, L+1?, D)
        time_emb: torch.Tensor,        # (B, D)
    ) -> torch.Tensor:                 # (B, action_dim)
        """Run transformer blocks and pool to an action prediction."""
        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)                         # (B, H, D)
        x_flat = x.reshape(x.size(0), -1)            # (B, H*D)
        return self.seq_pool_proj(x_flat)            # (B, D)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        actions_gt: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.LongTensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        In training mode (``actions_gt`` provided):
            Returns ``(pred_velocity, noise)`` for CFM loss computation.
        In inference mode (``actions_gt=None``):
            Returns the predicted flat action vector ``(B, action_dim)``.
        """
        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id,
                                   action_mask=action_mask)

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = self._default_embodiment_id(B, device)

        context_tokens = self._build_context(fused_tokens, state, embodiment_id)

        # ---- sample flow time t ~ Beta(2,2) clipped to [0.02, 0.98] ----
        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device=device,
                                                                               dtype=self.dtype)
        time_idx = (t * 1000).long()
        time_emb = self.time_pos_enc(1000)[:, time_idx, :].squeeze(0)  # (B, D)

        # ---- sample noise and interpolate ----
        noise = torch.rand_like(actions_gt) * 2 - 1                    # uniform [-1, 1]
        if action_mask is not None:
            noise = noise * action_mask.to(dtype=noise.dtype, device=device)

        t_bc = t.view(B, 1, 1)
        a_t = (1 - t_bc) * noise + t_bc * actions_gt                   # (B, H, per_dim)

        # ---- encode noisy action sequence ----
        action_tokens = self.action_encoder(a_t, embodiment_id)         # (B, H, D)

        # ---- predict velocity ----
        x_pooled = self._denoise(action_tokens, context_tokens, time_emb)
        pred_velocity = self.mlp_head(x_pooled, embodiment_id)          # (B, action_dim)

        return pred_velocity, noise

    # ------------------------------------------------------------------
    # Inference (Euler integration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.LongTensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate actions via Euler integration of the flow.

        Args:
            fused_tokens   : ``(B, L, D)`` context tokens from MultimodalMamba.
            state          : ``(B, state_dim)`` optional proprio state.
            embodiment_id  : ``(B,)`` category indices.
            action_mask    : ``(B, per_action_dim)`` or
                             ``(B, horizon, per_action_dim)`` float mask.

        Returns:
            Predicted flat action vector ``(B, action_dim)``.
        """
        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = self._default_embodiment_id(B, device)

        context_tokens = self._build_context(fused_tokens, state, embodiment_id)

        # broadcast / validate action_mask
        if action_mask is not None:
            if action_mask.dim() == 2:
                # (B, per_action_dim) → (B, horizon, per_action_dim)
                action_mask = action_mask.unsqueeze(1).expand(B, self.horizon, self.per_action_dim)
            action_mask = action_mask.to(dtype=fused_tokens.dtype, device=device)

        # ---- initialise from uniform noise ----
        a = torch.rand(B, self.horizon, self.per_action_dim, device=device,
                       dtype=self.dtype) * 2 - 1
        if action_mask is not None:
            a = a * action_mask

        N = int(self.config.num_inference_timesteps)
        dt = 1.0 / N
        for i in range(N):
            t_val = i / N
            time_idx = int(t_val * 1000)
            time_emb = (
                self.time_pos_enc(1000)[:, time_idx, :]
                    .to(device=device, dtype=self.dtype)
                    .expand(B, -1)
            )  # (B, D)

            action_tokens = self.action_encoder(a, embodiment_id)  # (B, H, D)
            x_pooled = self._denoise(action_tokens, context_tokens, time_emb)
            pred = self.mlp_head(x_pooled, embodiment_id)           # (B, action_dim)

            a = a + dt * pred.view(B, self.horizon, self.per_action_dim)
            if action_mask is not None:
                a = a * action_mask

        return a.reshape(B, self.action_dim)

    # ------------------------------------------------------------------
    # CFM loss (convenience method for training loop)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        fused_tokens: torch.Tensor,
        actions_gt: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.LongTensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Conditional Flow Matching MSE loss.

        The velocity target in straight-path CFM is:
            v* = a_gt − noise

        Args:
            fused_tokens : ``(B, L, D)`` MultimodalMamba output.
            actions_gt   : ``(B, horizon, per_action_dim)`` target actions.
            state        : optional proprio state ``(B, state_dim)``.
            embodiment_id: optional embodiment category ``(B,)``.
            action_mask  : optional float mask ``(B, horizon, per_action_dim)``.

        Returns:
            Scalar MSE loss.
        """
        pred_velocity, noise = self.forward(                   # type: ignore[misc]
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            embodiment_id=embodiment_id,
            action_mask=action_mask,
        )
        # Flatten targets to match pred_velocity shape (B, action_dim)
        target = (actions_gt - noise).reshape(pred_velocity.shape)

        if action_mask is not None:
            mask_flat = action_mask.reshape(pred_velocity.shape)
            loss = F.mse_loss(pred_velocity * mask_flat, target * mask_flat)
        else:
            loss = F.mse_loss(pred_velocity, target)
        return loss

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


__all__ = [
    "FlowMatchingActionHead",
    "BasicTransformerBlock",
    "MultiEmbodimentActionEncoder",
    "CategorySpecificMLP",
    "CategorySpecificLinear",
    "SinusoidalPositionalEncoding",
]
