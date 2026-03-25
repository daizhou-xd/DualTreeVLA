"""
MemoryTreeVLA – Full Architecture Implementation.

Data-flow (one inference step):
    Image (B,3,H,W)
        │
        ▼  Vision Mamba (Tree-scan SSM, GrootVL-style)
    Z_v  (B, L_v, D)
        │
        │              Tree.json / MemoryTree
        │                    │
        │                    ▼  Task Tree Mamba
        │               Z_t  (B, N, D)
        │                    │
        └─────────┬──────────┘
                  ▼  Multimodal Mamba  (cross-modal gated SSM)
              Z_fused  (B, L_v+N, D)
                  │
                  ▼  Action-LLM projector  (D → D_llm)
              Z_prefix (B, L_v+N, D_llm)    ← used as inputs_embeds
                  │
                  ▼  Action LLM (Qwen2.5-0.5B, last hidden layer)
              llm_out  (B, L_v+N, D_llm)
                  │
                  ▼  ActionConditionBuilder
                       llm_out + robot state (B, state_dim)
              ctx      (B, L_ctx, D)
                  │
                  ▼  FlowMatchingActionHead (CFM)
              action   (B, action_dim)   at inference
              velocity (B, H, action_dim) at training

Tree updates (low-frequency, ≤1 Hz):
    Z_v → tree_v_proj → tree_v_tokens (B, L_v, D_tree_llm)
    + Tree.json text → Tree LLM (Qwen2.5-1.5B-Instruct)
    → updated Tree.json (decoded text → MemoryTree.from_robocerebra())
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]  # transformers >= 4.x
except ImportError:  # very old transformers (< 3.x) used by other projects
    try:
        from transformers import AutoModelWithLMHead as AutoModelForCausalLM  # type: ignore[no-redef]
        from transformers import AutoTokenizer                                  # type: ignore[assignment]
    except ImportError:
        AutoModelForCausalLM = None  # type: ignore[assignment,misc]
        AutoTokenizer = None         # type: ignore[assignment]

from .memory_tree import MemoryTree
from .tree_scan import VisionMamba, TaskTreeMamba
from .action_condition import ActionConditionBuilder
from .action_head import FlowMatchingActionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multimodal Mamba  (pure-PyTorch, no CUDA extension required)
# ---------------------------------------------------------------------------

class _MambaLikeSSM(nn.Module):
    """Lightweight SSM approximation using linear-recurrence GRU cells.

    Serves as a drop-in stand-in when ``mamba_ssm`` is not installed.
    For production use, replace with ``mamba_ssm.Mamba`` for ~10× speed-up.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        # Factored into input projection + GRU-style gate
        self.in_proj  = nn.Linear(d_model, d_model * 2, bias=False)
        self.x_proj   = nn.Linear(d_model, d_state + d_state + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.d_state  = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(B, L, D)`` → ``(B, L, D)``."""
        B, L, D = x.shape
        # Gate-and-scan (simplified parameter-efficient version)
        xz = self.in_proj(x)                     # (B, L, 2D)
        x_in, z = xz.chunk(2, dim=-1)            # each (B, L, D)
        x_in = torch.tanh(x_in)
        z    = torch.sigmoid(z)
        # Simple exponential-decay scan along L (O(L), causal)
        dt_raw = self.x_proj(x_in)[..., :1]      # (B, L, 1)
        dt = torch.sigmoid(self.dt_proj(dt_raw))  # (B, L, D)
        # a = exp(-softplus(dt))  – decay factor
        a = torch.exp(-torch.nn.functional.softplus(dt))
        # Parallel prefix scan via chunk-scans (length-L sequential fallback)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = a[:, t] * h + (1 - a[:, t]) * x_in[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1)              # (B, L, D)
        return self.norm(y * z) + x               # residual


def _try_import_mamba(d_model: int, d_state: int, d_conv: int = 4, expand: int = 2):
    """Try to import ``mamba_ssm.Mamba``; fall back to _MambaLikeSSM."""
    try:
        from mamba_ssm import Mamba  # type: ignore[import]
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    except (ImportError, Exception):
        return _MambaLikeSSM(d_model=d_model, d_state=d_state)


class MultimodalMamba(nn.Module):
    """Cross-modal SSM fusion: Z_v × Z_t → Z_fused.

    Implements the gated-fusion strategy from CONSTRUCTION.md §3.2:
      1. Project Z_v and Z_t to a common dimension via linear layers.
      2. Tree-conditioned gate: task context weight-gates visual features.
      3. Concatenate gated Z_v with Z_t: Z_concat (B, L_v+N, D).
      4. Stack of Mamba SSM layers with residual connections.
      5. LayerNorm output → Z_fused.

    Args:
        d_model   : Feature dimension (must match embed_dim of Z_v and Z_t).
        d_state   : SSM state dimension.
        n_layers  : Number of Mamba layers.
    """

    def __init__(self, d_model: int, d_state: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_t = nn.Linear(d_model, d_model)
        self.gate   = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.mamba_layers = nn.ModuleList([
            _try_import_mamba(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        Z_v: torch.Tensor,   # (B, L_v, D)
        Z_t: torch.Tensor,   # (B, N,   D)
    ) -> torch.Tensor:       # (B, L_v+N, D)
        Z_v = self.proj_v(Z_v)
        Z_t = self.proj_t(Z_t)

        # Task-conditioned gate: global tree context modulates visual tokens
        task_ctx = Z_t.mean(dim=1, keepdim=True).expand_as(Z_v)   # (B, L_v, D)
        gate_w   = self.gate(torch.cat([Z_v, task_ctx], dim=-1))   # (B, L_v, D)
        Z_v_gated = Z_v * gate_w + Z_v                             # residual gate

        Z_concat = torch.cat([Z_v_gated, Z_t], dim=1)             # (B, L_v+N, D)
        x = Z_concat
        for layer in self.mamba_layers:
            x = layer(x)
        return self.norm(x)                                        # (B, L_v+N, D)


# ---------------------------------------------------------------------------
# Helper: truncate LLM to N layers and strip LM head (same as InternVL3Embedder)
# ---------------------------------------------------------------------------

def _truncate_llm(model: Any, n_layers: int) -> Any:
    """Keep only the first ``n_layers`` transformer layers and replace lm_head
    with Identity (we use hidden states, not logits)."""
    inner: Any = model.model if hasattr(model, "model") else model

    if hasattr(inner, "layers"):
        inner.layers = nn.ModuleList(list(inner.layers)[:n_layers])  # type: ignore[arg-type]
    elif hasattr(inner, "h"):  # GPT-2 style
        inner.h = nn.ModuleList(list(inner.h)[:n_layers])  # type: ignore[arg-type]

    if hasattr(model, "lm_head"):
        model.lm_head = nn.Identity()

    return model


# ---------------------------------------------------------------------------
# MemoryTreeVLA  (main model)
# ---------------------------------------------------------------------------

class MemoryTreeVLA(nn.Module):
    """MemoryTreeVLA: long-horizon robot manipulation VLA with hierarchical
    task-tree memory.

    Architecture:
      Vision Mamba  → Z_v
      Task Tree Mamba → Z_t          (driven by MemoryTree)
      Multimodal Mamba → Z_fused
      Z_fused → Action-LLM projector → Action LLM (Qwen2.5-0.5B) → llm_out
      ActionConditionBuilder(llm_out, robot_state) → ctx
      FlowMatchingActionHead(ctx) → action chunk

    Tree-management path (low-frequency):
      Z_v → Tree-LLM projector → Tree LLM (Qwen2.5-1.5B-Instruct) → Tree.json
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        m = cfg.model
        D = m.embed_dim

        # ------------------------------------------------------------------ #
        # 1. Visual encoder: Vision Mamba (GrootVL tree-scan)                #
        # ------------------------------------------------------------------ #
        # vision_layers is split evenly across two stages; each stage gets
        # vision_layers // 2 layers (minimum 1).  E.g. vision_layers=4 → [2,2].
        _half = max(1, m.vision_layers // 2)
        self.vision_mamba = VisionMamba(
            in_chans=3,
            channels=m.vision_channels,
            depths=[_half, _half],
            out_dim=D,
        )

        # ------------------------------------------------------------------ #
        # 2. Task-tree encoder: Task Tree Mamba                              #
        # ------------------------------------------------------------------ #
        # Node-id → embedding lookup (learnable, vocab covers all task nodes)
        self.node_embedding = nn.Embedding(m.node_vocab_size, D)
        self.task_tree_mamba = TaskTreeMamba(
            d_model=D,
            num_layers=m.tree_layers,
        )

        # ------------------------------------------------------------------ #
        # 3. Multimodal Mamba fusion                                         #
        # ------------------------------------------------------------------ #
        self.multimodal_mamba = MultimodalMamba(
            d_model=D,
            d_state=m.mm_d_state,
            n_layers=m.mm_mamba_layers,
        )

        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers >= 4.37 is required for MemoryTreeVLA. "
                "Install with: pip install transformers>=4.37"
            )
        if AutoTokenizer is None:
            raise ImportError("transformers AutoTokenizer is not available.")

        # ------------------------------------------------------------------ #
        # 4. Action LLM (Qwen2.5-0.5B) – feature extractor mode             #
        # ------------------------------------------------------------------ #
        self.action_llm_proj = nn.Linear(D, m.action_llm_dim)   # D → D_llm
        logger.info("Loading Action LLM from %s …", m.action_llm_path)
        _action_llm = AutoModelForCausalLM.from_pretrained(
            m.action_llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _action_llm = _truncate_llm(_action_llm, m.action_llm_layers)
        self.action_llm: nn.Module = _action_llm
        if m.freeze_action_llm:
            for p in self.action_llm.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # 5. Action condition builder                                        #
        # ------------------------------------------------------------------ #
        self.action_condition = ActionConditionBuilder(
            llm_dim=m.action_llm_dim,
            state_dim=m.state_dim,
            embed_dim=D,
            num_state_tokens=m.num_state_tokens,
            llm_seq_pool=m.llm_seq_pool,
        )

        # ------------------------------------------------------------------ #
        # 6. Flow-matching action head                                       #
        # ------------------------------------------------------------------ #
        self.action_head = FlowMatchingActionHead(
            embed_dim=D,
            hidden_dim=D * 2,
            per_action_dim=m.action_dim,
            horizon=m.action_horizon,
            num_heads=m.action_head_heads,
            num_layers=m.action_head_layers,
            num_inference_timesteps=m.num_inference_timesteps,
        )

        # ------------------------------------------------------------------ #
        # 7. Tree LLM (Qwen2.5-1.5B-Instruct) – tree management             #
        # ------------------------------------------------------------------ #
        self.tree_llm_proj = nn.Linear(D, m.tree_llm_dim)  # D → D_tree_llm
        logger.info("Loading Tree LLM from %s …", m.tree_llm_path)
        self.tree_llm = AutoModelForCausalLM.from_pretrained(
            m.tree_llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.tree_tokenizer = AutoTokenizer.from_pretrained(
            m.tree_llm_path, trust_remote_code=True
        )
        if m.freeze_tree_llm:
            for p in self.tree_llm.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # 8. Episode-level memory tree                                       #
        # ------------------------------------------------------------------ #
        self.memory_tree: Optional[MemoryTree] = None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → Z_v (B, L_v, D)."""
        return self.vision_mamba(images)

    def _encode_tree(
        self,
        node_ids: torch.Tensor,
        parent_map: Optional[Dict],
    ) -> torch.Tensor:
        """node_ids (B, N) + parent_map  → Z_t (B, N, D)."""
        node_feats = self.node_embedding(node_ids)   # (B, N, D)
        return self.task_tree_mamba(
            node_feats, parent_map=parent_map
        )  # (B, N, D)

    def _run_action_llm(self, Z_fused: torch.Tensor) -> torch.Tensor:
        """Z_fused (B, L, D) → llm_out (B, L, D_llm) [last hidden state]."""
        inputs_embeds = self.action_llm_proj(Z_fused)     # (B, L, D_llm)
        inputs_embeds = inputs_embeds.to(self.action_llm.dtype
                                         if hasattr(self.action_llm, 'dtype')
                                         else torch.bfloat16)
        out = self.action_llm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1].to(torch.float32)    # (B, L, D_llm)

    # ---------------------------------------------------------------------- #
    # Forward (training)                                                      #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        images: torch.Tensor,
        node_ids: torch.Tensor,
        state: torch.Tensor,
        actions_gt: Optional[torch.Tensor] = None,
        parent_map: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            images     : ``(B, 3, H, W)`` RGB images.
            node_ids   : ``(B, N)`` integer task-tree node indices.
            state      : ``(B, state_dim)`` proprioceptive state.
            actions_gt : ``(B, horizon, action_dim)`` ground-truth actions.
                         When provided, returns CFM loss.
            parent_map : ``{node_id: parent_id}`` tree topology for TreeMamba.
                         Can be ``None`` if topology was already set.

        Returns:
            Dict with keys:
              ``"Z_fused"``       – fused representation (B, L, D)
              ``"llm_out"``       – Action LLM hidden states (B, L, D_llm)
              ``"ctx"``           – action condition tokens (B, L_ctx, D)
              ``"loss"``          – CFM training loss (if ``actions_gt`` given)
              ``"pred_velocity"`` – predicted velocity field (if training)
        """
        # 1. Visual encoding
        Z_v = self._encode_visual(images)                    # (B, L_v, D)

        # 2. Task-tree encoding
        Z_t = self._encode_tree(node_ids, parent_map)        # (B, N, D)

        # 3. Multimodal fusion
        Z_fused = self.multimodal_mamba(Z_v, Z_t)           # (B, L_v+N, D)

        # 4. Action LLM feature extraction
        llm_out = self._run_action_llm(Z_fused)             # (B, L_v+N, D_llm)

        # 5. Build action condition
        ctx = self.action_condition(llm_out, state)          # (B, L_ctx, D)

        result: Dict[str, torch.Tensor] = {
            "Z_fused": Z_fused,
            "llm_out": llm_out,
            "ctx": ctx,
        }

        # 6. Action head (training loss or inference)
        if actions_gt is not None:
            loss = self.action_head.compute_loss(
                fused_tokens=ctx,
                actions_gt=actions_gt,
            )
            result["loss"] = loss
        else:
            action = self.action_head.get_action(fused_tokens=ctx)
            result["action"] = action

        return result

    # ---------------------------------------------------------------------- #
    # Inference step                                                          #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        parent_map: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Single-step inference: image + state → action chunk.

        Args:
            images     : ``(B, 3, H, W)``
            state      : ``(B, state_dim)``
            parent_map : optional topology override; if ``None`` the cached
                         topology from the last ``set_tree()`` call is used.

        Returns:
            ``(B, action_dim)`` flat action vector (first step of horizon chunk).
        """
        Z_v = self._encode_visual(images)

        # Use cached MemoryTree node ids if available, else zeros
        if self.memory_tree is not None:
            N = len(self.memory_tree)
            node_ids = torch.arange(N, device=images.device).unsqueeze(0).expand(
                images.size(0), -1
            )
            pm = self.memory_tree.to_parent_map()
        else:
            node_ids = torch.zeros(images.size(0), 1, dtype=torch.long, device=images.device)
            pm = parent_map

        Z_t     = self._encode_tree(node_ids, pm)
        Z_fused = self.multimodal_mamba(Z_v, Z_t)
        llm_out = self._run_action_llm(Z_fused)
        ctx     = self.action_condition(llm_out, state)
        return self.action_head.get_action(fused_tokens=ctx)

    # ---------------------------------------------------------------------- #
    # Tree LLM: update Tree.json from visual observation                     #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def update_tree(
        self,
        images: torch.Tensor,
        current_tree_text: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Low-frequency tree update (called by the orchestrator, not every step).

        Encodes the current visual observation, prepends the projected visual
        tokens to the tree description, and generates an updated Tree.json.

        Args:
            images           : ``(B, 3, H, W)`` – typically B=1 at inference.
            current_tree_text: Serialised current Tree.json or task description.
            max_new_tokens   : Generation budget.

        Returns:
            Raw text output from Tree LLM (expected to be valid JSON).
        """
        device = images.device

        # Visual encoding → project to Tree LLM dim
        Z_v = self._encode_visual(images)                          # (B, L_v, D)
        vis_embeds = self.tree_llm_proj(Z_v)                       # (B, L_v, D_tlm)
        vis_embeds = vis_embeds.to(self.tree_llm.dtype
                                   if hasattr(self.tree_llm, 'dtype')
                                   else torch.bfloat16)

        # Tokenise tree description
        tree_inputs = self.tree_tokenizer(
            current_tree_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        text_embeds = self.tree_llm.get_input_embeddings()(
            tree_inputs["input_ids"]
        )                                                          # (B, L_t, D_tlm)

        # Concat visual prefix + tree text
        inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        attn_mask = torch.ones(
            inputs_embeds.size(0), inputs_embeds.size(1),
            dtype=torch.long, device=device,
        )

        out_ids = self.tree_llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tree_tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        n_prefix = inputs_embeds.size(1)
        new_ids  = out_ids[0, n_prefix:]
        return self.tree_tokenizer.decode(new_ids, skip_special_tokens=True)

    # ---------------------------------------------------------------------- #
    # Episode management                                                      #
    # ---------------------------------------------------------------------- #

    def set_memory_tree(self, tree: MemoryTree) -> None:
        """Attach a MemoryTree for the current episode."""
        self.memory_tree = tree
        # MemoryTree.to_parent_map() returns Dict[str, Optional[str]];
        # convert keys/values to int for TaskTreeMamba which expects int node ids.
        str_map = tree.to_parent_map()
        int_map: Dict[int, Optional[int]] = {
            int(k): (int(v) if v is not None else None)
            for k, v in str_map.items()
        }
        self.task_tree_mamba.set_tree(int_map)

    def reset(self) -> None:
        """Reset all episode-level state."""
        self.memory_tree = None
        self.task_tree_mamba.reset()

    # ---------------------------------------------------------------------- #
    # Convenience: freeze / unfreeze sub-modules by training stage           #
    # ---------------------------------------------------------------------- #

    def set_stage(self, stage: int) -> None:
        """Apply per-stage parameter freeze policy from CONSTRUCTION.md §4.

        Stage 1 – Tree LLM + Multimodal Mamba pre-training:
          Train: tree_llm, tree_llm_proj, multimodal_mamba
          Freeze: everything else

        Stage 2 – Fusion + action head joint training:
          Train: multimodal_mamba, action_condition, action_head, action_llm_proj
          Freeze: vision_mamba, task_tree_mamba, action_llm, tree_llm

        Stage 3 – End-to-end fine-tuning:
          Train: action_llm (lr×0.1), multimodal_mamba, action_condition, action_head
          Freeze: vision_mamba (keep general visual features)
        """
        def _set(module: nn.Module, trainable: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(trainable)

        if stage == 1:
            _set(self, False)
            _set(self.tree_llm, True)
            _set(self.tree_llm_proj, True)
            _set(self.multimodal_mamba, True)

        elif stage == 2:
            _set(self, False)
            _set(self.multimodal_mamba, True)
            _set(self.action_llm_proj, True)
            _set(self.action_condition, True)
            _set(self.action_head, True)

        elif stage == 3:
            _set(self, True)
            _set(self.vision_mamba, False)          # keep frozen in stage 3

        else:
            raise ValueError(f"Unknown training stage: {stage}. Expected 1, 2, or 3.")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            "Stage %d: trainable %.2fM / %.2fM parameters",
            stage, trainable / 1e6, total / 1e6,
        )

