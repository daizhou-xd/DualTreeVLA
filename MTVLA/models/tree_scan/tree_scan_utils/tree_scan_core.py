"""
Tree Scan Core Utilities: Minimum Spanning Tree construction and BFS traversal.

Adapted from GrootVL (MambaTree, NeurIPS 2024 Spotlight).
Reference: https://github.com/EasonXiao-888/MambaTree

Provides pure-PyTorch/NumPy CPU fallback when the compiled `tree_scan` CUDA
extension is not available.  When installed, the extension is used automatically
for GPU-accelerated MST and BFS.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional CUDA extension
# ---------------------------------------------------------------------------
try:
    from tree_scan import _C as _tree_scan_C  # type: ignore[import]  # compiled CUDA extension

    _CUDA_EXT = True
except ImportError:
    _tree_scan_C = None
    _CUDA_EXT = False


# ---------------------------------------------------------------------------
# Pure-Python / NumPy MST (Kruskal) – batch-wise fallback
# ---------------------------------------------------------------------------

def _kruskal_mst_single(
    edge_index: torch.Tensor,  # (E, 2) int32
    edge_weight: torch.Tensor,  # (E,) float
    n_vertices: int,
) -> torch.Tensor:
    """Kruskal's MST for a single graph (CPU, NumPy-free union-find)."""
    parent = list(range(n_vertices))
    rank = [0] * n_vertices

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    order = torch.argsort(edge_weight)
    edges_cpu = edge_index.cpu().numpy()

    mst: list[list[int]] = []
    for idx in order.tolist():
        u, v = int(edges_cpu[idx, 0]), int(edges_cpu[idx, 1])
        if union(u, v):
            mst.append([u, v])
            if len(mst) == n_vertices - 1:
                break

    return torch.tensor(mst, dtype=torch.int64, device=edge_index.device)


# ---------------------------------------------------------------------------
# BFS traversal (CPU)
# ---------------------------------------------------------------------------

def bfs_traversal(
    mst_edges: torch.Tensor,  # (E, 2) int64  -- E = n_nodes-1
    n_nodes: int,
    root: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BFS traversal of an undirected tree.

    Args:
        mst_edges : ``(n_nodes-1, 2)`` edge index tensor.
        n_nodes   : total number of nodes.
        root      : BFS root (default 0).

    Returns:
        sorted_index  : ``(n_nodes,)`` node indices in BFS order.
        sorted_parent : ``(n_nodes,)`` BFS *position* of each node's parent;
                         −1 for the root node.
    """
    edges_np = mst_edges.cpu().tolist()
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges_np:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    visited = [False] * n_nodes
    visited[root] = True
    queue: deque[int] = deque([root])

    bfs_order: list[int] = []
    node_to_pos: dict[int, int] = {root: 0}
    parent_pos: list[int] = [-1] * n_nodes  # parent BFS position, -1 = root

    while queue:
        node = queue.popleft()
        bfs_order.append(node)
        for nb in sorted(adj[node]):
            if not visited[nb]:
                visited[nb] = True
                node_to_pos[nb] = len(bfs_order)
                parent_pos[nb] = node_to_pos[node]
                queue.append(nb)

    sorted_parent_list = [parent_pos[bfs_order[i]] for i in range(n_nodes)]

    device = mst_edges.device
    return (
        torch.tensor(bfs_order, dtype=torch.int64, device=device),
        torch.tensor(sorted_parent_list, dtype=torch.int64, device=device),
    )


# ---------------------------------------------------------------------------
# MinimumSpanningTree module
# ---------------------------------------------------------------------------

class MinimumSpanningTree(nn.Module):
    """Build a Minimum Spanning Tree from an image feature map patch grid.

    Uses cosine-similarity (or L2) between spatially adjacent patches as edge
    weights, then runs Kruskal's algorithm per sample.

    Args:
        distance_func : ``"Cosine"`` (default) or ``"L2"``.
        mapping_func  : optional monotone scalar transform applied to raw
                        similarity before MST (e.g. ``torch.exp``).

    Adapted from GrootVL ``tree_scan_utils/tree_scan_core.py``.
    """

    def __init__(
        self,
        distance_func: str = "Cosine",
        mapping_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_edge_index(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Build ``(E, 2)`` grid edge-index for an H×W patch grid."""
        idx = torch.arange(H * W, dtype=torch.int64, device=device).reshape(H, W)
        row_edges = torch.stack([idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)], dim=1)
        col_edges = torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)], dim=1)
        return torch.cat([row_edges, col_edges], dim=0)  # (E, 2)

    def _edge_weights(self, fm: torch.Tensor, max_tree: bool) -> torch.Tensor:
        """Return per-edge weights ``(B, E)`` suitable for *minimum* spanning tree."""
        B, C, H, W = fm.shape
        if self.distance_func == "Cosine":
            row_sim = F.cosine_similarity(
                fm[:, :, :-1, :].reshape(B, C, -1),
                fm[:, :, 1:, :].reshape(B, C, -1),
                dim=1,
            )  # (B, (H-1)*W)
            col_sim = F.cosine_similarity(
                fm[:, :, :, :-1].reshape(B, C, -1),
                fm[:, :, :, 1:].reshape(B, C, -1),
                dim=1,
            )  # (B, H*(W-1))
            sim = torch.cat([row_sim, col_sim], dim=1)  # (B, E)
            if self.mapping_func is not None:
                sim = self.mapping_func(sim)
            # higher cosine sim → closer nodes → lower MST edge weight
            weight = sim if max_tree else -sim
        else:  # L2
            row_d = ((fm[:, :, :-1, :] - fm[:, :, 1:, :]) ** 2).sum(1).reshape(B, -1)
            col_d = ((fm[:, :, :, :-1] - fm[:, :, :, 1:]) ** 2).sum(1).reshape(B, -1)
            weight = torch.cat([row_d, col_d], dim=1)
            if self.mapping_func is not None:
                weight = self.mapping_func(weight)
            if max_tree:
                weight = -weight
        return weight  # (B, E)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self, fm: torch.Tensor, max_tree: bool = False
    ) -> Tuple[list[torch.Tensor], list[tuple]]:
        """Compute per-sample MSTs and BFS traversal orders.

        Args:
            fm       : ``(B, C, H, W)`` feature map.
            max_tree : build maximum spanning tree instead.

        Returns:
            bfs_indices : list of B tensors, each ``(L,)`` BFS node order.
            bfs_parents : list of B tensors, each ``(L,)`` BFS parent positions
                          (``−1`` for root).
        """
        B, C, H, W = fm.shape
        L = H * W

        with torch.no_grad():
            edge_idx = self._grid_edge_index(H, W, fm.device)  # (E, 2)
            weights = self._edge_weights(fm, max_tree)  # (B, E)

            bfs_indices, bfs_parents = [], []
            for b in range(B):
                mst = _kruskal_mst_single(edge_idx, weights[b], L)  # (L-1, 2)
                si, sp = bfs_traversal(mst, L, root=0)
                bfs_indices.append(si)
                bfs_parents.append(sp)

        return bfs_indices, bfs_parents
