"""
SGMTS — Semantic-Guided Mamba Tree Scan (CONSTRUCTION.md §3.2)

步骤:
  1. Patch CNN → patch features F ∈ R^{P × d_f}
  2. 语义引导向量构造: g_sem = β·g_task + (1-β)·s̄_top
  3. 语义根节点选择: r* = argmax_i cos(p_i, g_sem)
  4. 语义加权 MST: w_ij = cos(p_i,p_j) + α·cos(p_i,g_sem)·cos(p_j,g_sem)
  5. 从 r* BFS → 扫描序列
  6. Tree-SSM 前向
  7. 输出: Z_v ∈ R^{P × d_visual}
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNN(nn.Module):
    """ViT-style patch projector."""

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
    """Kruskal MAX spanning tree. Returns selected edge indices."""
    sorted_idx = torch.argsort(edge_w, descending=True).tolist()
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
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

    Innovations over GrootV:
      1. Dynamic semantic root: r* = argmax cos(p_i, g_sem)
      2. Semantic-biased MST: w_ij = cos(p_i,p_j) + α·cos(p_i,g_sem)·cos(p_j,g_sem)
      3. β-schedule: g_sem = β·g_task + (1-β)·s̄_top
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
    ):
        super().__init__()
        self.d_f      = d_f
        self.d_state  = d_state
        self.alpha    = alpha
        self.connectivity = connectivity

        self.patch_cnn  = PatchCNN(patch_size=patch_size, d_f=d_f)
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
        src_list, dst_list = [], []
        for i in range(nH):
            for j in range(nW):
                u = i * nW + j
                if j + 1 < nW:   src_list.append(u); dst_list.append(i * nW + (j + 1))
                if i + 1 < nH:   src_list.append(u); dst_list.append((i + 1) * nW + j)
                if self.connectivity == 8:
                    if i + 1 < nH and j + 1 < nW:
                        src_list.append(u); dst_list.append((i + 1) * nW + (j + 1))
                    if i + 1 < nH and j - 1 >= 0:
                        src_list.append(u); dst_list.append((i + 1) * nW + (j - 1))

        edge_src = torch.tensor(src_list, dtype=torch.long)
        edge_dst = torch.tensor(dst_list, dtype=torch.long)

        g_gate_cpu = self.lang_gate(g_sem).detach().cpu().float()
        f_cpu      = f.detach().cpu().float()
        f_norm     = F.normalize(f_cpu, dim=1)
        g_norm     = F.normalize(g_gate_cpu.unsqueeze(0), dim=1)

        cos_patch  = (f_norm[edge_src] * f_norm[edge_dst]).sum(dim=1)
        sem_score  = (f_norm @ g_norm.T).squeeze(-1)
        w_ij = cos_patch + self.alpha * sem_score[edge_src] * sem_score[edge_dst]

        mst_idx = _kruskal_mst_max(edge_src, edge_dst, w_ij, P)

        adj = [[] for _ in range(P)]
        for idx in mst_idx:
            u, v = int(src_list[idx]), int(dst_list[idx])
            adj[u].append(v); adj[v].append(u)

        root = int(sem_score.argmax().item())

        bfs_order, visited, parent_of = [], [False] * P, [-1] * P
        queue = [root]; visited[root] = True
        while queue:
            node = queue.pop(0); bfs_order.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True; parent_of[nb] = node; queue.append(nb)

        A = -torch.exp(self.A_log.float()).to(device)
        bfs_t   = torch.tensor(bfs_order, dtype=torch.long, device=device)
        f_sorted = f[bfs_t]

        g_prime  = self.W_g_prime(g_sem.to(device))
        sem_bfs  = sem_score[bfs_t].to(device)
        X = (f_sorted + sem_bfs.unsqueeze(1) * g_prime.unsqueeze(0)).to(f.dtype)

        delta  = F.softplus(self.W_delta(X))
        B_proj = self.B_proj(X)
        C_proj = self.C_proj(X)

        orig2bfs = [0] * P
        for bfs_i, orig_i in enumerate(bfs_order):
            orig2bfs[orig_i] = bfs_i

        H = X.new_zeros(P, self.d_f, self.d_state)
        Y = X.new_zeros(P, self.d_f)
        for bfs_i in range(P):
            d_i   = delta[bfs_i]
            A_bar = torch.exp(d_i.unsqueeze(1) * A)
            B_bar = d_i.unsqueeze(1) * B_proj[bfs_i].unsqueeze(0)
            par_orig = parent_of[bfs_order[bfs_i]]
            h_par = H[orig2bfs[par_orig]] if par_orig != -1 else H.new_zeros(self.d_f, self.d_state)
            h_i   = A_bar * h_par + B_bar * X[bfs_i].unsqueeze(1)
            H[bfs_i] = h_i
            Y[bfs_i] = (h_i * C_proj[bfs_i].unsqueeze(0)).sum(dim=1) + self.D.to(X.dtype) * X[bfs_i]

        Y_raster = Y.new_zeros(P, self.d_f)
        for bfs_i, orig_i in enumerate(bfs_order):
            Y_raster[orig_i] = Y[bfs_i]

        return self.out_norm(self.out_proj(Y_raster))   # (P, d_visual)
