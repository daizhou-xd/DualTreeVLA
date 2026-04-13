"""
DualTreeVLA — 主模型 (CONSTRUCTION.md §2.1)

数据流 (单帧):
  task_desc → LLM → g_task (SGMTS) + task_tokens
  HMT top-2 abstract nodes → s_top → SGMTS (β-blended)
  SGMTS(image, g_sem) → Z_v (B, P, d_visual)
    TreeSSM(HMT) → m_ctx_last (B, d_ssm)
    CrossModalFusion(z_v_mean, m_ctx_last, g_lang, q) → Z_fused
  FlowMatchingHead(Z_fused) → â
  JumpAwareHead(A_act_hist, â_1) → p_jump → HMT.insert(z_v_mean, â_1, p_jump>=0.5)

三阶段训练:
    pretrain  : L_boundary + L_sem (pretrain.py 调用)
  phase1    : L_flow only (LLM+预训练模块冻结)
  phase2    : L_flow only (全量微调)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from transformers import AutoModel, AutoTokenizer

from .memory_tree import (
    HierarchicalMemoryTree,
    MLPElevation,
    TreeSSMReadout,
    propagate_elevation_to_root,
)
from .sgmts import SGMTSEncoder
from .fusion import CrossModalFusion
from .action_head import FlowMatchingActionHead
from .semantic_jump_head import JumpAwareHead


class DualTreeVLA(nn.Module):
    """
    Parameters
    ----------
    llm_path    : Qwen2.5 checkpoint directory
    d           : unified embedding dim (HMT nodes, d_hidden)
    d_a         : action dim (default 7)
    d_q         : proprioceptive state dim (default 84)
    d_visual    : SGMTS output dim
    d_ssm       : TreeSSM output dim
    d_state     : SSM hidden-state size
    patch_size  : image patch side length
    H_a         : action prediction horizon
    n_ode       : Euler ODE steps at inference
    K_elev      : Branch K_elev param (kept for compat; elevation now triggered by branch)
    delta_w     : weight increment per merge
    freeze_llm  : freeze LLM on init
    """

    def __init__(
        self,
        llm_path: str,
        d: int = 256,
        d_a: int = 7,
        d_q: int = 84,
        d_visual: int = 256,
        d_ssm: int = 256,
        d_state: int = 16,
        patch_size: int = 16,
        H_a: int = 16,
        n_ode: int = 20,
        K_elev: int = 4,
        delta_w: float = 0.1,
        mount_tau: float = 0.4,
        max_tree_depth: int = 4,
        freeze_llm: bool = True,
        clip_model_name: Optional[str] = None,
        # legacy / compat params (unused)
        theta_fuse: float = 0.4,
        tau: float = 0.1,
    ):
        super().__init__()
        self.d         = d
        self.d_a       = d_a
        self.d_q       = d_q
        self.d_visual      = d_visual
        self.H_a           = H_a
        self.max_tree_depth = max_tree_depth

        # ── LLM backbone (language encoding only) ────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        _can_flash = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        )
        _attn_impl   = "flash_attention_2" if _can_flash else "eager"
        _torch_dtype = torch.bfloat16       if _can_flash else torch.float32
        _device_map  = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        try:
            self.llm = AutoModel.from_pretrained(
                llm_path,
                trust_remote_code=True,
                attn_implementation=_attn_impl,
                torch_dtype=_torch_dtype,
                device_map=_device_map,
            )
        except Exception:
            self.llm = AutoModel.from_pretrained(
                llm_path, trust_remote_code=True, device_map=_device_map,
            )
        d_lang = self.llm.config.hidden_size

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # ── SGMTS visual encoder ─────────────────────────────────────
        self.sgmts = SGMTSEncoder(
            d_f=d_visual,
            d_lang=d_lang,
            d_hidden=d,        # HMT node embedding dim
            d_visual=d_visual,
            patch_size=patch_size,
            d_state=d_state,
            clip_model_name=clip_model_name,  # None → PatchCNN；非 None → 冻结 CLIP ViT
        )

        # ── JumpAwareHead ─────────────────────────────────────────────
        self.jump_head = JumpAwareHead(d_a=d_a)

        # ── TreeSSMReadout ────────────────────────────────────────────
        self.tree_ssm = TreeSSMReadout(
            d_node=d,
            d_ssm=d_ssm,
            d_state=d_state,
        )

        # ── MLPElevation ──────────────────────────────────────────────
        self.mlp_elev = MLPElevation(d=d)

        # ── CrossModalFusion ─────────────────────────────────────────
        self.fusion = CrossModalFusion(
            d_ssm=d_ssm,
            d_visual=d_visual,
            d_lang=d_lang,
            d_q=d_q,
            d=d,
        )

        # ── FlowMatchingActionHead ────────────────────────────────────
        self.action_head = FlowMatchingActionHead(
            d_a=d_a,
            H_a=H_a,
            d_model=d,
            d_ctx=d,
            N_ode=n_ode,
        )

        # ── Semantic → language projection (for L_sem InfoNCE in pretrain) ─
        self.sem_proj = nn.Linear(d, d_lang, bias=False)

        # ── Tree pool ─────────────────────────────────────────────────
        self._tree_pool: Dict[int, HierarchicalMemoryTree] = {}
        self._tree_cfg = dict(d=d, d_a=d_a, K_elev=K_elev, delta_w=delta_w, mount_tau=mount_tau)

    # ---------------------------------------------------------------- #
    #  Tree management                                                  #
    # ---------------------------------------------------------------- #

    def get_tree(self, batch_idx: int) -> HierarchicalMemoryTree:
        if batch_idx not in self._tree_pool:
            self._tree_pool[batch_idx] = HierarchicalMemoryTree(**self._tree_cfg)
        return self._tree_pool[batch_idx]

    def reset_trees(self, batch_size: int = 1):
        self._tree_pool.clear()
        for i in range(batch_size):
            self._tree_pool[i] = HierarchicalMemoryTree(**self._tree_cfg)

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _get_A_act_tensor(
        self,
        active_node,
        device: torch.device,
    ) -> torch.Tensor:
        """(1, L, d_a) float tensor of active node action history. L=0 if empty."""
        if active_node is None or not active_node.a_hist:
            return torch.zeros(1, 0, self.d_a, device=device)
        stacked = torch.stack(active_node.a_hist, dim=0).to(device)   # (L, d_a)
        return stacked.unsqueeze(0)                                     # (1, L, d_a)

    def _get_s_top(
        self,
        tree: HierarchicalMemoryTree,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Return mean of the top-2 abstract nodes' s embeddings, or None.
        Top = BFS-depth ≤ 1 non-leaf nodes with s != None.
        """
        bfs = tree.bfs_order()
        abstracts = []
        for nid in bfs:
            node = tree.nodes[nid]
            if not node.is_leaf() and node.s is not None:
                abstracts.append(node.s)
            if len(abstracts) == 2:
                break
        if not abstracts:
            return None
        return torch.stack(abstracts).mean(0).to(device=device, dtype=dtype)

    def _compute_beta(self, tree: HierarchicalMemoryTree) -> float:
        """β linearly decays from 1.0 to 0.3 as tree gains depth."""
        if tree.root_id is None or tree.size() < 3:
            return 1.0
        max_depth = max(tree.depth(nid) for nid in tree.nodes)
        # decay: β = max(0.3, 1.0 - 0.7 * depth/5)
        return max(0.3, 1.0 - 0.14 * max_depth)

    def _encode_language(
        self,
        instructions: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lang_hidden : (B, L, d_lang)
            g_lang      : (B, d_lang)  mean-pooled
        """
        enc = self.tokenizer(
            instructions, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        llm_grad = any(p.requires_grad for p in self.llm.parameters())
        with torch.set_grad_enabled(llm_grad):
            out = self.llm(**enc, use_cache=False)
        lang_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        g_lang = (lang_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return lang_hidden, g_lang

    def _compute_flow_loss(
        self,
        all_Z_fused: List[torch.Tensor],
        actions: torch.Tensor,   # (B, T, d_a)
        device: torch.device,
    ) -> torch.Tensor:
        B, T, _ = actions.shape
        losses = []
        for t, Z_t in enumerate(all_Z_fused):
            t_end = min(t + self.H_a, T)
            a_gt  = actions[:, t:t_end]
            if a_gt.shape[1] < self.H_a:
                a_gt = torch.cat(
                    [a_gt, a_gt[:, -1:].expand(-1, self.H_a - a_gt.shape[1], -1)], dim=1
                )
            losses.append(self.action_head.flow_loss(a_gt, Z_t))
        return torch.stack(losses).mean()

    def _encode_text_descs(self, descs: List[str], device: torch.device) -> torch.Tensor:
        """Encode a list of subtask descriptions to mean-pooled language embeddings (S, d_lang)."""
        enc = self.tokenizer(
            descs, return_tensors="pt", padding=True, truncation=True, max_length=64,
        ).to(device)
        llm_grad = any(p.requires_grad for p in self.llm.parameters())
        with torch.set_grad_enabled(llm_grad):
            out = self.llm(**enc, use_cache=False)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)

    def _compute_boundary_labels(
        self,
        actions: torch.Tensor,   # (B, T, d_a)
        device: torch.device,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Self-supervised boundary labels: y_t = 1[||a_t - ā|| > γ·σ]. Returns (B*T,)."""
        B, T, _ = actions.shape
        a_mean = actions.mean(dim=1, keepdim=True)
        gap    = (actions - a_mean).norm(dim=-1)           # (B, T)
        sigma  = gap.std(dim=1, keepdim=True).clamp(min=1e-6)
        y      = (gap > gamma * sigma).float()
        return y.reshape(-1).to(device)

    # ---------------------------------------------------------------- #
    #  Inference: single step                                           #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def step(
        self,
        image: torch.Tensor,           # (1, C, H, W)
        instruction: str,
        q: torch.Tensor,               # (1, d_q)
        a_prev: Optional[torch.Tensor] = None,  # (1, d_a) — previous GT or predicted action
    ) -> torch.Tensor:                 # (1, H_a, d_a)
        device = image.device
        tree   = self.get_tree(0)

        # 1. Language
        _, g_lang = self._encode_language([instruction], device)   # (1, d_lang)

        # 2. s_top from HMT
        s_top  = self._get_s_top(tree, device, g_lang.dtype)
        beta_v = self._compute_beta(tree)

        # 3. SGMTS
        Z_v     = self.sgmts(image, g_lang, [s_top], [beta_v])   # (1, P, d_visual)
        z_v_mean = Z_v.mean(1).squeeze(0)                         # (d_visual,)

        # 4. TreeSSM readout（取最后一个语义token作为 m_ctx）
        Z_M = self.tree_ssm(tree, device=device)                   # (N_M, d_ssm)
        m_ctx = Z_M[-1].unsqueeze(0)                               # (1, d_ssm)
        pad_q = q
        if pad_q.shape[1] < self.d_q:
            pad_q = F.pad(pad_q, (0, self.d_q - pad_q.shape[1]))
        elif pad_q.shape[1] > self.d_q:
            pad_q = pad_q[:, :self.d_q]

        # 5. Fusion
        Z_fused = self.fusion(z_v_mean.unsqueeze(0), m_ctx, g_lang, pad_q)  # (1, 1, d)
        # Broaden context: [fused_vec | SGMTS patch tokens] → (1, 1+P, d)
        ctx = torch.cat([Z_fused, Z_v], dim=1)                              # (1, 1+P, d)

        # 6. Flow matching prediction
        a_pred  = self.action_head.sample(ctx, device=device)        # (1, H_a, d_a)
        a_hat_1 = a_pred[:, 0]                                     # (1, d_a)

        # 7. JumpAwareHead → p_jump
        active_node = tree.nodes.get(tree.active_id)
        A_act = self._get_A_act_tensor(active_node, device)        # (1, L, d_a)
        p_jump, _ = self.jump_head(A_act, a_hat_1)
        force_branch = bool(p_jump.item() >= 0.5)

        # 8. Update HMT
        # 先用 mlp_elev 计算当前帧语义供挂载点搜索（no_grad，仅用于树结构决策）
        s_current = self.mlp_elev(
            z_v_mean.unsqueeze(0).float()
        ).squeeze(0).detach().cpu()
        tree.insert(z_v_mean.detach(), a_hat_1.squeeze(0).detach(), force_branch,
                    s_current=s_current)
        if tree.elevation_pending_parent is not None:
            propagate_elevation_to_root(tree, tree.elevation_pending_parent, self.mlp_elev, device)
            tree._prune_to_max_depth(self.max_tree_depth)
            tree.elevation_pending_parent = None

        return a_pred

    # ---------------------------------------------------------------- #
    #  Training forward                                                 #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        images: torch.Tensor,           # (B, T, C, H, W)
        instructions: List[str],
        states: torch.Tensor,           # (B, T, d_q)
        actions: torch.Tensor,          # (B, T, d_a)
        subtask_ids: Optional[torch.Tensor] = None,
        subtask_descs: Optional[List[List[str]]] = None,
        mode: str = "phase1",
        w_boundary: float = 1.0,
        w_sem: float = 0.5,
        tau_sem: float = 0.07,
    ) -> Dict[str, torch.Tensor]:
        """
        mode='pretrain' : L_boundary + L_sem
        mode='phase1'   : L_flow only
        mode='phase2'   : L_flow only
        """
        B, T, C, H, W = images.shape
        device = images.device

        compute_flow = (mode in ("phase1", "phase2"))
        compute_jump = (mode == "pretrain")

        self.reset_trees(B)

        # Shared language encoding
        _, g_lang = self._encode_language(instructions, device)    # (B, d_lang)

        all_Z_fused: List[torch.Tensor] = []
        jump_logits: List[torch.Tensor] = []

        for t in range(T):
            imgs_t = images[:, t]          # (B, C, H, W)
            q_t    = states[:, t]          # (B, T, d_q)
            a_t    = actions[:, t]         # (B, d_a)

            # Pad/truncate q
            if q_t.shape[1] < self.d_q:
                q_t = F.pad(q_t, (0, self.d_q - q_t.shape[1]))
            elif q_t.shape[1] > self.d_q:
                q_t = q_t[:, :self.d_q]

            # s_top and β per sample
            s_top_list = []
            beta_list  = []
            for b in range(B):
                tree_b = self.get_tree(b)
                s_top_b = self._get_s_top(tree_b, device, g_lang.dtype)
                s_top_list.append(s_top_b)
                beta_list.append(self._compute_beta(tree_b))

            # SGMTS
            if self.training:
                Z_v_t = grad_ckpt(
                    self.sgmts, imgs_t, g_lang, s_top_list, beta_list,
                    use_reentrant=False,
                )
            else:
                Z_v_t = self.sgmts(imgs_t, g_lang, s_top_list, beta_list)   # (B, P, d_visual)

            z_v_mean_t = Z_v_t.mean(1)   # (B, d_visual)

            # Per-sample: JumpAwareHead + tree update
            m_ctx_list = []
            for b in range(B):
                tree_b      = self.get_tree(b)
                active_node = tree_b.nodes.get(tree_b.active_id)
                A_act_b     = self._get_A_act_tensor(active_node, device)  # (1, L, d_a)

                # During training always use GT action for jump + insert (teacher forcing)
                p_j, logit_j = self.jump_head(A_act_b, a_t[b:b+1])
                if compute_jump:
                    jump_logits.append(logit_j)

                # 预训练阶段：优先用 GT subtask_ids 强制分支，保证树结构与标注一致
                if compute_jump and subtask_ids is not None:
                    gt_branch = False
                    if t > 0:
                        gt_branch = bool(subtask_ids[b, t].item() != subtask_ids[b, t - 1].item())
                    force_branch = gt_branch
                else:
                    force_branch = bool(p_j.detach().item() >= 0.5)
                # 计算当前帧语义供挂载点搜索（no_grad，仅用于树结构决策）
                with torch.no_grad():
                    s_current_b = self.mlp_elev(
                        z_v_mean_t[b].unsqueeze(0).float()
                    ).squeeze(0).cpu()
                tree_b.insert(z_v_mean_t[b].detach(), a_t[b].detach(), force_branch,
                              s_current=s_current_b)

                if tree_b.elevation_pending_parent is not None:
                    propagate_elevation_to_root(tree_b, tree_b.elevation_pending_parent,
                                               self.mlp_elev, device)
                    tree_b._prune_to_max_depth(self.max_tree_depth)
                    tree_b.elevation_pending_parent = None

                Z_M_b = self.tree_ssm(tree_b, device=device)   # (N_M, d_ssm)
                # When tree has no abstract nodes (step_level training: tree reset
                # each forward, T=1 → only 1 leaf, s=None → TreeSSMReadout returns
                # zeros), fall back to z_v_mean as a meaningful memory proxy so
                # CrossModalFusion and tree_ssm parameters receive useful gradients.
                m_ctx_b = Z_M_b[-1]
                if tree_b.size() <= 1:
                    # Project visual mean into d_ssm space via tree_ssm's abs_in_proj:
                    # use a detached proxy so the fallback doesn't interfere with the
                    # tree_ssm output gradient path.
                    m_ctx_b = z_v_mean_t[b].detach()[:self.tree_ssm.d_ssm]
                    if m_ctx_b.shape[0] < self.tree_ssm.d_ssm:
                        m_ctx_b = F.pad(m_ctx_b, (0, self.tree_ssm.d_ssm - m_ctx_b.shape[0]))
                m_ctx_list.append(m_ctx_b)

            m_ctx_t = torch.stack(m_ctx_list, dim=0)           # (B, d_ssm)
            Z_fused_t = self.fusion(z_v_mean_t, m_ctx_t, g_lang, q_t)   # (B, 1, d)
            # Broaden context: [fused_vec | SGMTS patch tokens] → (B, 1+P, d)
            ctx_t = torch.cat([Z_fused_t, Z_v_t], dim=1)                # (B, 1+P, d)
            if not compute_flow:
                ctx_t = ctx_t.detach()
            all_Z_fused.append(ctx_t)

        torch.cuda.empty_cache()

        # ── Phase 1 / 2: flow loss ────────────────────────────────────
        if compute_flow:
            L_flow = self._compute_flow_loss(all_Z_fused, actions, device)
            return {"L_flow": L_flow, "total": L_flow}

        # ── Pretrain: boundary + semantic losses ─────────────────────
        if compute_jump and jump_logits:
            from dual_tree_vla.losses.tree_losses import l_boundary, l_sem

            logits_t = torch.cat(jump_logits, dim=0)          # (B*T,)
            y_act    = self._compute_boundary_labels(actions, device)
            L_bnd    = l_boundary(logits_t, y_act)

            # L_sem: abstract node semantics vs subtask text embeddings
            L_sem = torch.zeros((), device=device)
            if subtask_ids is not None:
                s_nodes_list, s_text_list = [], []
                sem_dtype = next(self.sem_proj.parameters()).dtype
                for b in range(B):
                    tree_b = self.get_tree(b)
                    if tree_b.size() == 0:
                        continue
                    descs = ([instructions[b]]
                             if subtask_descs is None or b >= len(subtask_descs) or not subtask_descs[b]
                             else subtask_descs[b])
                    g_sub = self._encode_text_descs(descs, device).to(dtype=sem_dtype)  # (S, d_lang)
                    for nid, node in tree_b.nodes.items():
                        if node.is_leaf() or node.s is None:
                            continue
                        t_idx = min(nid, subtask_ids.shape[1] - 1)
                        sid = int(subtask_ids[b, t_idx].item())
                        sid = max(0, min(sid, g_sub.shape[0] - 1))
                        s_proj_out = self.sem_proj(node.s.to(device=device, dtype=sem_dtype))
                        s_nodes_list.append(s_proj_out)
                        s_text_list.append(g_sub[sid])
                if len(s_nodes_list) >= 2:
                    S_nodes = torch.stack(s_nodes_list, dim=0)
                    S_text  = torch.stack(s_text_list, dim=0)
                    L_sem = l_sem(S_nodes, S_text, temperature=tau_sem)

            total = w_boundary * L_bnd + w_sem * L_sem
            return {
                "L_boundary": L_bnd,
                "L_sem": L_sem,
                "total": total,
            }

        return {"total": torch.zeros((), device=device)}
