"""
Tree Mamba Encoder – task-tree hierarchy encoder for MemoryTreeVLA.

Unlike VisionMamba (which derives the tree topology from image-feature similarity
via MST), TreeMamba works with *explicit* task-tree topologies parsed from
``Tree.json`` provided by the Tree LLM.

Design:
  * Each task-tree node is embedded by a learnable embedding layer (or an LLM
    token projection layer from the Tree LLM output).
  * The tree structure (parent→child edges) is converted to a BFS sequence once
    per episode and cached.
  * A stack of ``TreeMambaLayer`` blocks runs the tree-SSM recurrence along the
    explicit topology, producing contextual node representations ``Z_t``.

Output format:
  Z_t : ``(B, N, D)`` – N contextual node embeddings consumed by
         MultimodalMamba fusion together with Z_v.

Usage example::

    encoder = TaskTreeMamba(node_vocab_size=128, embed_dim=256, num_layers=2)
    # tree_adj: list of (child_id, parent_id) tuples, or None for root
    Z_t = encoder(node_ids, tree_adj)   # (B, N, D)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat  # type: ignore[import]

from .tree_scan_utils.tree_scan_core import bfs_traversal


# ---------------------------------------------------------------------------
# Tree topology helpers
# ---------------------------------------------------------------------------

def build_bfs_from_adj(
    parent_map: Dict[int, Optional[int]],
    root: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build BFS order from an explicit parent-map.

    Args:
        parent_map : ``{node_id: parent_id}``; root has ``parent_id = None``.
        root       : root node id.

    Returns:
        sorted_index  : ``(N,)`` node ids in BFS order.
        sorted_parent : ``(N,)`` BFS *position* of parent; ``−1`` for root.
    """
    from collections import defaultdict, deque

    children: Dict[int, list] = defaultdict(list)
    for node, par in parent_map.items():
        if par is not None:
            children[par].append(node)

    n = len(parent_map)
    bfs_order, node_to_pos = [], {}
    par_pos = {}
    queue: deque = deque([root])
    node_to_pos[root] = 0

    while queue:
        node = queue.popleft()
        bfs_order.append(node)
        for child in sorted(children[node]):
            node_to_pos[child] = len(bfs_order)
            par_pos[child] = node_to_pos[node]
            queue.append(child)

    sorted_par = [-1] + [par_pos[bfs_order[i]] for i in range(1, n)]

    sorted_index = torch.tensor(bfs_order, dtype=torch.int64)
    sorted_parent = torch.tensor(sorted_par, dtype=torch.int64)
    return sorted_index, sorted_parent


def tree_json_to_adj(tree_json: dict) -> Dict[int, Optional[int]]:
    """Convert a ``Tree.json`` dict to a ``{node_id: parent_id}`` map.

    Expected Tree.json schema (from CONSTRUCTION.md)::

        {
          "task": "...",
          "nodes": [
            {"id": 0, "description": "...", "parent": null, ...},
            {"id": 1, "description": "...", "parent": 0, ...},
            ...
          ]
        }
    """
    return {
        node["id"]: node.get("parent")
        for node in tree_json.get("nodes", [])
    }


# ---------------------------------------------------------------------------
# TreeMamba SSM layer (1-D sequence along BFS order)
# ---------------------------------------------------------------------------

class TreeMambaLayer(nn.Module):
    """Single-layer Tree SSM operating on *explicit* tree topology.

    Differences from ``Tree_SSM`` (vision case):
    * The tree topology (BFS order) is passed in explicitly – no MST needed.
    * Input is a 1-D node sequence ``(B, N, D)`` rather than a 2-D spatial grid.
    * Works on sequences of variable length N.

    Internally the same d_state=1 scalar recurrence is used::

        h[i] = dA[i] · h[parent[i]] + dB[i] · x[i]
        y[i] = h[i] · C[i] + D · x[i]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_state != 1:
            raise ValueError("TreeMambaLayer requires d_state=1.")

        d_inner = int(ssm_ratio * d_model)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # in-proj (gated)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act = nn.SiLU()

        # x_proj: x → (dt, B_ssm, C_ssm) per token
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Δ projection
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self._init_dt_proj(self.dt_proj, self.dt_rank, d_inner, dt_min, dt_max)

        # A and D
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _init_dt_proj(
        proj: nn.Linear, dt_rank: int, d_inner: int,
        dt_min: float = 0.001, dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias._no_reinit = True  # type: ignore[attr-defined]

    def forward(
        self,
        x: torch.Tensor,              # (B, N, D)
        sorted_index: torch.Tensor,   # (N,) BFS node order
        sorted_parent: torch.Tensor,  # (N,) BFS parent positions (−1 = root)
    ) -> torch.Tensor:
        """
        Args:
            x             : ``(B, N, D)`` node feature sequence.
            sorted_index  : ``(N,)`` node ids in BFS order.
            sorted_parent : ``(N,)`` BFS positions of parents; ``−1`` = root.

        Returns:
            ``(B, N, D)`` updated node features.
        """
        B, N, D = x.shape

        # gate
        xz = self.in_proj(x)                       # (B, N, 2*D_inner)
        x_inner, z = xz.chunk(2, dim=-1)            # each (B, N, D_inner)
        z = self.act(z)

        # project to SSM params
        x_dbl = self.x_proj(x_inner)               # (B, N, dt_rank+2)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, 1, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))           # (B, N, D_inner)

        As = -torch.exp(self.A_log[:, 0].float())   # (D_inner,) negative
        Ds = self.D.float()                          # (D_inner,)
        B_ssm = B_ssm.squeeze(-1)                   # (B, N) scalar per token
        C_ssm = C_ssm.squeeze(-1)                   # (B, N) scalar per token

        # rearrange to (B, D, N) for recurrence
        x_seq = x_inner.permute(0, 2, 1).contiguous()   # (B, D, N)
        dt_seq = dt.permute(0, 2, 1).contiguous()        # (B, D, N)
        B_seq = B_ssm.unsqueeze(1).expand(B, self.d_inner, N)  # (B, D, N)
        C_seq = C_ssm.unsqueeze(1).expand(B, self.d_inner, N)  # (B, D, N)

        # discretise
        dA = torch.exp(dt_seq * As.unsqueeze(0).unsqueeze(-1))  # (B, D, N)
        dBx = dt_seq * B_seq * x_seq                             # (B, D, N)

        # expand BFS tensors to batch
        si = sorted_index.unsqueeze(0).expand(B, -1)   # (B, N)
        sp = sorted_parent.unsqueeze(0).expand(B, -1)  # (B, N)

        # tree SSM recurrence
        from .tree_scanning import _tree_ssm_recurrence
        h = _tree_ssm_recurrence(dBx, dA, si, sp)      # (B, D, N)

        # read-out
        y = h * C_seq + Ds.unsqueeze(0).unsqueeze(-1) * x_seq  # (B, D, N)
        y = self.out_norm(y.permute(0, 2, 1).contiguous())      # (B, N, D_inner)
        y = y * z                                                # gated
        return self.dropout(self.out_proj(y))                    # (B, N, D)


# ---------------------------------------------------------------------------
# TaskTreeMamba  –  full N-layer tree encoder
# ---------------------------------------------------------------------------

class TaskTreeMamba(nn.Module):
    """Encode a hierarchical task tree into contextual node embeddings Z_t.

    Processes the Tree LLM's output (node token embeddings from
    ``Qwen2.5-1.5B-Instruct``) through a shallow stack of ``TreeMambaLayer``s
    to produce Z_t that captures the parent→child execution context.

    The BFS traversal order is computed once per task tree and can be cached
    across time steps of the same episode.

    Args:
        d_model       : I/O embedding dimension.
        num_layers    : number of Tree SSM layers.
        ssm_ratio     : inner-dim expansion ratio (default 2.0).
        dropout       : dropout rate per layer.
        max_nodes     : maximum number of nodes in the task tree (for sanity
                        checks only; the encoder handles variable-length trees).

    Input:
        node_features : ``(B, N, D)`` – node embeddings (e.g. projected LLM
                         hidden states or learnable status embeddings).
        parent_map    : ``{node_id: parent_id | None}`` describing the tree.
                        The root node has ``parent_id = None``.
                        **Must be the same for all items in the batch.**

    Output:
        Z_t : ``(B, N, D)`` – contextual node embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        ssm_ratio: float = 2.0,
        dropout: float = 0.0,
        max_nodes: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes

        self.layers = nn.ModuleList([
            TreeMambaLayer(d_model=d_model, ssm_ratio=ssm_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # cache for current episode's BFS topology
        self._cached_sorted_index: Optional[torch.Tensor] = None
        self._cached_sorted_parent: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # topology helpers
    # ------------------------------------------------------------------

    def set_tree(self, parent_map: Dict[int, Optional[int]], root: int = 0) -> None:
        """Pre-compute and cache BFS topology for the current task tree.

        Call this once when the Tree LLM emits a new ``Tree.json``.

        Args:
            parent_map : ``{node_id: parent_id | None}``.
            root       : root node id (default 0).
        """
        si, sp = build_bfs_from_adj(parent_map, root=root)
        self._cached_sorted_index = si
        self._cached_sorted_parent = sp

    def set_tree_from_json(self, tree_json: dict, root: int = 0) -> None:
        """Convenience wrapper: parse ``Tree.json`` dict and call :meth:`set_tree`."""
        adj = tree_json_to_adj(tree_json)
        self.set_tree(adj, root=root)

    def reset(self) -> None:
        """Clear cached topology (call at episode start)."""
        self._cached_sorted_index = None
        self._cached_sorted_parent = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,               # (B, N, D)
        parent_map: Optional[Dict[int, Optional[int]]] = None,
        sorted_index: Optional[torch.Tensor] = None,   # (N,), override cache
        sorted_parent: Optional[torch.Tensor] = None,  # (N,), override cache
    ) -> torch.Tensor:
        """
        Args:
            node_features : ``(B, N, D)`` input node embeddings.
            parent_map    : if provided, recomputes BFS topology.
            sorted_index  : explicit BFS ordering; overrides ``parent_map``.
            sorted_parent : explicit BFS parent positions; overrides cache.

        Returns:
            Z_t : ``(B, N, D)`` contextual node embeddings.
        """
        # resolve BFS topology
        si: torch.Tensor
        sp: torch.Tensor
        if sorted_index is not None and sorted_parent is not None:
            si, sp = sorted_index, sorted_parent
        elif parent_map is not None:
            self.set_tree(parent_map)
            assert self._cached_sorted_index is not None
            assert self._cached_sorted_parent is not None
            si, sp = self._cached_sorted_index, self._cached_sorted_parent
        elif self._cached_sorted_index is not None and self._cached_sorted_parent is not None:
            si, sp = self._cached_sorted_index, self._cached_sorted_parent
        else:
            raise ValueError(
                "TaskTreeMamba.forward(): no tree topology provided. "
                "Call set_tree() first or pass parent_map / sorted_index."
            )

        device = node_features.device
        si = si.to(device)
        sp = sp.to(device)

        x = node_features
        for layer in self.layers:
            x = x + layer(x, si, sp)  # residual

        return self.norm(x)  # (B, N, D)


__all__ = ["TaskTreeMamba", "TreeMambaLayer", "build_bfs_from_adj", "tree_json_to_adj"]
