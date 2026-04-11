"""
SGMTS — Semantic-Guided Mamba Tree Scan (CONSTRUCTION.md §3.2)

整体架构：
  输入图像
      ↓
  [CLIP Vision Encoder] ──────→ patch 特征 [B, P, d_f] ────┐
      │                                                       ▼
  [CLIP Text Encoder] ←── 文本/类别提示 ───────→ [语义引导树构建器]
                                                      │
                                              语义重要性图 + 树拓扑
                                                      │
                                              [MambaTree扫描层]
                                                      │
                                              语义增强视觉特征
                                                      │
                                              [下游任务头]

步骤:
  1. CLIP Vision Encoder 提取视觉特征 F ∈ R^{P × d_f}（CLIPPatchExtractor 或 PatchCNN）
  2. 语义引导向量构造: g_sem = β·g_task + (1-β)·s̄_top（CLIP Text Encoder 输出）
  3. 语义引导树构建器:
       ① 语义重要性图 σ_i = cos(p_i, W_g·g_sem)
       ② 语义根: r* = argmax_i σ_i
       ③ 语义加权 MST: w_ij = cos(p_i,p_j) + α·σ_i·σ_j
  4. MambaTree 扫描层: X_i = p_i + σ_i·W_g'·g_sem → BFS 序 Tree-SSM
  5. 输出: Z_v ∈ R^{P × d_visual} → 下游任务头

视觉骨干选择:
  CLIPPatchExtractor (默认，clip_model_name 非 None):
    使用冻结的 CLIP ViT 提取多尺度语义 patch 特征
    依赖: transformers >= 4.40.0（已在 requirements.txt 中）
  PatchCNN (fallback，clip_model_name=None):
    轻量级随机初始化 patch 投影器，适合快速实验。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPPatchExtractor(nn.Module):
    """
    使用冻结 CLIP Vision Transformer 提取 patch 级视觉特征。

    直接利用 CLIP 的跨模态预训练语义，无需在 mini-imagenet 上从头训练。
    CLIP 权重完全冻结，仅 adapter（Linear + LayerNorm）参与训练。

    Args:
        model_name : HuggingFace 模型 ID，例如 "openai/clip-vit-base-patch16"
        d_f        : 输出特征维度（适配 SGMTS 内部维度）
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", d_f: int = 256):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPVisionConfig

        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        clip_dim = self.clip.config.hidden_size
        self.patch_size = self.clip.config.patch_size

        # 轻量 adapter：将 CLIP 特征投影到 SGMTS 内部维度 d_f
        self.adapter = nn.Sequential(
            nn.Linear(clip_dim, d_f),
            nn.LayerNorm(d_f),
        )

        # CLIP 标准图像归一化常数（OpenAI CLIP ViT-B/16）
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x : (B, C, H, W) 图像，值域 [-1, 1]（CLIP 归一化）或 [0, 1]
        Returns:
            feats : (B, P, d_f)  patch 特征
            nH    : int  patch 行数
            nW    : int  patch 列数
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        nH, nW = H // ps, W // ps

        # 将 [0,1] 图像归一化为 CLIP 标准 (mean/std) 格式
        x = (x.float() - self.clip_mean) / self.clip_std

        with torch.no_grad():
            out = self.clip(pixel_values=x)
            # last_hidden_state: (B, 1 + nH*nW, clip_dim)，index 0 为 [CLS]
            patch_feats = out.last_hidden_state[:, 1:, :]   # (B, P, clip_dim)

        feats = self.adapter(patch_feats)   # (B, P, d_f)
        return feats, nH, nW


class PatchCNN(nn.Module):
    """轻量 ViT-style patch 投影器（随机初始化 fallback）。"""

    def __init__(self, patch_size: int = 16, d_f: int = 256, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        d_patch = in_channels * patch_size * patch_size
        self.proj = nn.Sequential(
            nn.Linear(d_patch, d_patch // 2),
            nn.GELU(),
            nn.Linear(d_patch // 2, d_f),
            nn.LayerNorm(d_f),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        ps = self.patch_size
        nH, nW = H // ps, W // ps
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.reshape(B * nH * nW, C * ps * ps)
        feats = self.proj(patches).view(B, nH * nW, -1)
        return feats, nH, nW


def _kruskal_mst_max(edge_src, edge_dst, edge_w, num_nodes):
    """Kruskal MAX spanning tree (path-compressed union-by-rank). Returns selected edge indices."""
    sorted_idx = torch.argsort(edge_w, descending=True).tolist()
    parent = list(range(num_nodes))
    rank   = [0] * num_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    mst_edges = []
    src_list = edge_src.tolist()
    dst_list = edge_dst.tolist()
    for idx in sorted_idx:
        if union(int(src_list[idx]), int(dst_list[idx])):
            mst_edges.append(idx)
        if len(mst_edges) == num_nodes - 1:
            break
    return mst_edges


class SGMTSEncoder(nn.Module):
    """
    Semantic-Guided Mamba Tree Scan encoder (CONSTRUCTION.md §3.2).

    架构流程:
      [CLIP Vision Encoder] → 单尺度 patch 特征 (B, P, d_f)
      [CLIP Text Encoder]  → 文本/类别提示语义向量 g_sem
      [语义引导树构建器]   → 语义重要性图 σ + 树拓扑（MST + r*）
      [MambaTree扫描层]    → 语义增强视觉特征 Z_v → [下游任务头]

    Innovations over GrootV:
      1. 语义重要性图: σ_i = cos(p_i, W_g·g_sem)（空间语义热图）
      2. Dynamic semantic root: r* = argmax σ_i
      3. Semantic-biased MST: w_ij = cos(p_i,p_j) + α·σ_i·σ_j
      4. Semantic-enhanced SSM input: X_i = p_i + σ_i·W_g'·g_sem
      5. β-schedule: g_sem = β·g_task + (1-β)·s̄_top

    视觉骨干 (clip_model_name):
      非 None → CLIPPatchExtractor（冻结 CLIP ViT，无需 mini-imagenet 预训练）
      None    → PatchCNN（轻量随机初始化 fallback）
    """

    def __init__(
        self,
        d_f: int = 256,
        d_lang: int = 896,
        d_hidden: int = 256,
        d_visual: int = 256,
        patch_size: int = 16,
        d_state: int = 16,
        alpha: float = 0.5,
        connectivity: int = 4,
        clip_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.d_f      = d_f
        self.d_state  = d_state
        self.alpha    = alpha
        self.connectivity = connectivity

        # ── 视觉骨干 ──────────────────────────────────────────────────
        if clip_model_name is not None:
            # CLIP 语义特征：冻结权重 + 轻量 adapter，替代 mini-imagenet 预训练
            self.patch_cnn = CLIPPatchExtractor(model_name=clip_model_name, d_f=d_f)
        else:
            # PatchCNN：随机初始化 fallback（适合快速实验）
            self.patch_cnn = PatchCNN(patch_size=patch_size, d_f=d_f)
        self.lang_gate  = nn.Linear(d_lang, d_f, bias=False)
        self.s_top_proj = nn.Linear(d_hidden, d_lang, bias=False)
        self.W_g_prime  = nn.Linear(d_lang, d_f, bias=False)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_f, -1)
        self.A_log   = nn.Parameter(torch.log(A_init.clone()))
        self.D       = nn.Parameter(torch.ones(d_f))
        self.B_proj  = nn.Linear(d_f, d_state)
        self.C_proj  = nn.Linear(d_f, d_state)
        self.W_delta = nn.Linear(d_f, d_f, bias=True)

        self.out_proj = nn.Linear(d_f, d_visual)
        self.out_norm = nn.LayerNorm(d_visual)

        # Grid edge cache keyed by (nH, nW, connectivity) — built once, reused every frame
        self._grid_edge_cache: Dict[tuple, tuple] = {}

        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_f) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.W_delta.bias.copy_(inv_dt)

    def _get_grid_edges(self, nH: int, nW: int) -> tuple:
        """Return cached (edge_src, edge_dst, src_list, dst_list) for the grid topology.

        Edges are built once per unique (nH, nW, connectivity) and reused across all
        subsequent frames, eliminating the O(P) Python loop from the hot path.
        """
        key = (nH, nW, self.connectivity)
        if key not in self._grid_edge_cache:
            src_list, dst_list = [], []
            for i in range(nH):
                for j in range(nW):
                    u = i * nW + j
                    if j + 1 < nW:
                        src_list.append(u); dst_list.append(i * nW + (j + 1))
                    if i + 1 < nH:
                        src_list.append(u); dst_list.append((i + 1) * nW + j)
                    if self.connectivity == 8:
                        if i + 1 < nH and j + 1 < nW:
                            src_list.append(u); dst_list.append((i + 1) * nW + (j + 1))
                        if i + 1 < nH and j - 1 >= 0:
                            src_list.append(u); dst_list.append((i + 1) * nW + (j - 1))
            self._grid_edge_cache[key] = (
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long),
                src_list,
                dst_list,
            )
        return self._grid_edge_cache[key]

    def forward(
        self,
        images: torch.Tensor,
        g_task: torch.Tensor,
        s_top: Optional[List[Optional[torch.Tensor]]] = None,
        beta: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        images : (B, C, H, W)
        g_task : (B, d_lang)
        s_top  : list[B] of (d_hidden,) tensors or None
        beta   : list[B] of floats in [0.3, 1.0]; None = all 1.0
        Returns: Z_v (B, P, d_visual)
        """
        B = images.shape[0]
        device = images.device

        feats, nH, nW = self.patch_cnn(images)   # (B, P, d_f)

        g_sem_list = []
        for b in range(B):
            b_val = 1.0 if (beta is None) else float(beta[b])
            g_t   = g_task[b]
            if s_top is not None and s_top[b] is not None:
                s_t_proj = self.s_top_proj(s_top[b].to(device=device, dtype=g_t.dtype))
                g_sem_b  = b_val * g_t + (1.0 - b_val) * s_t_proj
            else:
                g_sem_b = g_t
            g_sem_list.append(g_sem_b)
        g_sem = torch.stack(g_sem_list, dim=0)   # (B, d_lang)

        all_Y = [self._scan_one(feats[b], g_sem[b], nH, nW, device) for b in range(B)]
        return torch.stack(all_Y, dim=0)   # (B, P, d_visual)

    def _scan_one(self, f, g_sem, nH, nW, device):
        P = nH * nW

        # ── 1. Grid edges (cached, O(0) after first call) ───────────────
        edge_src, edge_dst, src_list, dst_list = self._get_grid_edges(nH, nW)

        # ── 2. Semantic scores on CPU ────────────────────────────────────
        g_gate_cpu = self.lang_gate(g_sem).detach().cpu().float()
        f_cpu      = f.detach().cpu().float()
        f_norm     = F.normalize(f_cpu, dim=1)
        g_norm     = F.normalize(g_gate_cpu.unsqueeze(0), dim=1)

        cos_patch = (f_norm[edge_src] * f_norm[edge_dst]).sum(dim=1)
        sem_score = (f_norm @ g_norm.T).squeeze(-1)
        w_ij = cos_patch + self.alpha * sem_score[edge_src] * sem_score[edge_dst]

        # ── 3. Kruskal MST ───────────────────────────────────────────────
        mst_idx = _kruskal_mst_max(edge_src, edge_dst, w_ij, P)

        # ── 4. BFS with deque (O(1) popleft vs O(P) list.pop(0)) ────────
        adj = [[] for _ in range(P)]
        for idx in mst_idx:
            u, v = int(src_list[idx]), int(dst_list[idx])
            adj[u].append(v); adj[v].append(u)

        root      = int(sem_score.argmax().item())
        bfs_order = []
        parent_of = [-1] * P
        level_of  = [0]  * P
        visited   = [False] * P
        q = deque([root])
        visited[root] = True
        while q:
            node = q.popleft()
            bfs_order.append(node)
            lv = level_of[node]
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb]   = True
                    parent_of[nb] = node
                    level_of[nb]  = lv + 1
                    q.append(nb)

        # ── 5. Batched SSM on GPU ────────────────────────────────────────
        A = -torch.exp(self.A_log.float()).to(device)       # (d_f, d_state)
        bfs_t_cpu = torch.tensor(bfs_order, dtype=torch.long)
        bfs_t    = bfs_t_cpu.to(device)
        f_sorted = f[bfs_t]

        g_prime = self.W_g_prime(g_sem.to(device))
        sem_bfs = sem_score[bfs_t_cpu].to(device)
        X = (f_sorted + sem_bfs.unsqueeze(1) * g_prime.unsqueeze(0)).to(f.dtype)

        # Pre-compute all SSM matrices in one batched pass (no per-step exp/proj)
        delta  = F.softplus(self.W_delta(X))                               # (P, d_f)
        B_proj = self.B_proj(X)                                            # (P, d_state)
        C_proj = self.C_proj(X)                                            # (P, d_state)
        A_bar_all = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0))         # (P, d_f, d_state)
        Bx_all    = (delta * X).unsqueeze(2) * B_proj.unsqueeze(1)         # (P, d_f, d_state)

        # Build parent index tensor in BFS order
        orig2bfs = [0] * P
        for bfs_i, orig_i in enumerate(bfs_order):
            orig2bfs[orig_i] = bfs_i
        parent_bfs = [
            orig2bfs[parent_of[bfs_order[i]]] if parent_of[bfs_order[i]] != -1 else -1
            for i in range(P)
        ]

        # Level-parallel tree SSM: O(depth) Python iterations instead of O(P)
        # Within each BFS level all nodes are independent → process as a batch
        level_bfs = [level_of[bfs_order[i]] for i in range(P)]
        max_lv    = max(level_bfs)
        lv_groups: List[List[int]] = [[] for _ in range(max_lv + 1)]
        for bfs_i, lv in enumerate(level_bfs):
            lv_groups[lv].append(bfs_i)

        H = X.new_zeros(P, self.d_f, self.d_state)
        for lv_nodes in lv_groups:
            idx   = torch.tensor(lv_nodes, dtype=torch.long, device=device)   # (L,)
            p_idx = torch.tensor([parent_bfs[i] for i in lv_nodes],
                                  dtype=torch.long, device=device)             # (L,)
            mask  = p_idx >= 0
            h_par = H.new_zeros(len(lv_nodes), self.d_f, self.d_state)
            if mask.any():
                h_par[mask] = H[p_idx[mask]]
            H[idx] = A_bar_all[idx] * h_par + Bx_all[idx]

        # Batch Y computation (was per-node inside the loop)
        Y = (H * C_proj.unsqueeze(1)).sum(dim=2) + self.D.to(X.dtype) * X    # (P, d_f)

        # Scatter back to raster order (fancy index assignment, no Python loop)
        Y_raster = Y.new_zeros(P, self.d_f)
        Y_raster[bfs_t] = Y

        return self.out_norm(self.out_proj(Y_raster))   # (P, d_visual)
