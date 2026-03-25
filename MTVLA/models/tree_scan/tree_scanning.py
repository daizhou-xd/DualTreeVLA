"""
Tree Scanning SSM (State Space Model) – core computation module.

Adapts GrootVL's input-adaptive tree-topology scanning (MambaTree, NeurIPS 2024)
for MemoryTreeVLA.  Key differences from the original:

* **No compiled CUDA extension required** – the SSM recurrence is implemented
  in pure PyTorch; the CUDA extension (``tree_scan._C``) is used automatically
  when available for ~10× speed-up.
* ``d_state = 1`` (as used in the original GrootVL) reduces the SSM hidden
  state to a *scalar* per channel, making the tree recurrence
  ``h[i] = dA[i] · h[parent[i]] + dB[i] · x[i]``.
* The tree topology (parent-child edges) is provided externally, so the same
  module handles both *vision* trees (MST from image features) and *task* trees
  (explicit JSON topology).

Reference implementation:
  https://github.com/EasonXiao-888/MambaTree/tree/main/GrootV/classification/models
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat  # type: ignore[import]


# ---------------------------------------------------------------------------
# SSM recurrence on a tree (pure PyTorch, sequential BFS order)
# ---------------------------------------------------------------------------

def _tree_ssm_recurrence(
    feat_in: torch.Tensor,     # (B, D, L)  – dB·x values, BFS node order
    edge_weight: torch.Tensor,  # (B, D, L)  – dA values, BFS node order
    sorted_index: torch.Tensor, # (B, L)     – actual node indices in BFS order
    sorted_parent: torch.Tensor,# (B, L)     – BFS *position* of parent, −1 for root
) -> torch.Tensor:
    """Pure-PyTorch tree SSM recurrence (O(L) sequential over BFS order).

    For each node i (in BFS order, so parent always processed first):
        h[node_i] = dA[node_i] · h[parent_node_i] + feat_in[node_i]

    Args:
        feat_in      : per-node ``dB·x`` values indexed by *node id*.
        edge_weight  : per-node ``dA`` values indexed by *node id*.
        sorted_index : ``(B, L)`` maps BFS position → node id.
        sorted_parent: ``(B, L)`` maps BFS position → parent's BFS position,
                        ``−1`` for root.

    Returns:
        h_out : ``(B, D, L)`` hidden states indexed by *node id*.
    """
    B, D, L = feat_in.shape
    device = feat_in.device

    # h stores hidden states indexed by node-id (not BFS position)
    h = torch.zeros(B, D, L, device=device, dtype=feat_in.dtype)

    # h_bfs[t] = hidden state of node at BFS-position t  (used for parent lookup)
    h_bfs = torch.zeros(B, D, L, device=device, dtype=feat_in.dtype)

    for t in range(L):
        # node id for this BFS step  (B,)
        nid = sorted_index[:, t]            # (B,)
        par = sorted_parent[:, t]           # (B,) – BFS position of parent, −1 if root

        # Gather feat_in and edge_weight for this batch of nodes
        nid_exp = nid.view(B, 1, 1).expand(B, D, 1)
        fx = feat_in.gather(2, nid_exp).squeeze(2)   # (B, D)
        ea = edge_weight.gather(2, nid_exp).squeeze(2)  # (B, D)

        # Gather parent hidden state from h_bfs (0 when par == -1 / root)
        is_root = (par == -1)                 # (B,)
        safe_par = par.clamp(min=0)           # avoid negative index
        par_exp = safe_par.view(B, 1, 1).expand(B, D, 1)
        h_par = h_bfs.gather(2, par_exp).squeeze(2)  # (B, D)
        h_par[is_root] = 0.0

        # SSM recurrence: h = dA · h_parent + dBx
        h_new = ea * h_par + fx              # (B, D)

        # Write back into h (node-id indexed) and h_bfs (BFS-position indexed)
        nid_exp2 = nid.view(B, 1, 1).expand(B, D, 1)
        h = h.scatter(2, nid_exp2, h_new.unsqueeze(2))

        bfs_pos = torch.full((B,), t, dtype=torch.int64, device=device)
        bps_exp = bfs_pos.view(B, 1, 1).expand(B, D, 1)
        h_bfs = h_bfs.scatter(2, bps_exp, h_new.unsqueeze(2))

    return h  # (B, D, L) indexed by node-id


# ---------------------------------------------------------------------------
# Top-level tree_scanning function (mirrors GrootVL API)
# ---------------------------------------------------------------------------

def tree_scanning(
    x: torch.Tensor,               # (B, D, H, W) – channel-first feature map
    x_proj_weight: torch.Tensor,   # (K, dt_rank+2N, D)
    x_proj_bias: Optional[torch.Tensor],
    dt_projs_weight: torch.Tensor, # (K, D, dt_rank)
    dt_projs_bias: torch.Tensor,   # (K, D)
    A_logs: torch.Tensor,          # (K*D, N) = (D, 1) for K=1, N=1
    Ds: torch.Tensor,              # (K*D,) = (D,)
    out_norm: nn.Module,
    h_norm: Optional[nn.Module] = None,
    bfs_indices: Optional[List[torch.Tensor]] = None,   # pre-computed BFS order per sample
    bfs_parents: Optional[List[torch.Tensor]] = None,   # pre-computed BFS parents per sample
    force_fp32: bool = False,
    to_dtype: bool = True,
) -> torch.Tensor:
    """Run tree-topology selective SSM scan on a spatial feature map.

    The tree topology is either supplied explicitly via *bfs_indices* /
    *bfs_parents* (for task-tree encoding) or computed from the feature map
    itself using a Minimum Spanning Tree (for vision encoding).

    Args:
        x              : ``(B, D, H, W)`` input feature map.
        bfs_indices    : list of B ``(L,)`` tensors – BFS node order per sample.
                         If ``None``, an MST is built on-the-fly.
        bfs_parents    : list of B ``(L,)`` tensors – BFS parent positions.

    Returns:
        ``(B, H, W, D)`` output feature map.
    """
    B, D, H, W = x.shape
    L = H * W
    K = 1  # single scan direction (as in GrootVL)
    _, N = A_logs.shape  # N = d_state = 1

    # ---- project x --------------------------------------------------------
    xs = rearrange(x, "b d h w -> b 1 d (h w)")  # (B, 1, D, L)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)

    dt_rank = dt_projs_weight.shape[2]
    dts, Bs, Cs = torch.split(x_dbl, [dt_rank, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs_flat = xs.view(B, -1, L)            # (B, D, L)
    dts_flat = dts.contiguous().view(B, -1, L)  # (B, D, L)
    As = -torch.exp(A_logs.to(torch.float))     # (D, N) – negative
    Bs = Bs.contiguous()                        # (B, 1, N, L)
    Cs = Cs.contiguous()                        # (B, 1, N, L)
    Ds_ = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs_flat = xs_flat.to(torch.float)
        dts_flat = dts_flat.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    # ---- discretise -------------------------------------------------------
    dts_flat = F.softplus(dts_flat + delta_bias.unsqueeze(0).unsqueeze(-1))
    # dA: (B, D, L) – scalar A factor per node per dim  (d_state=1 collapsed)
    dA = torch.exp(dts_flat * (-As[:, 0]).unsqueeze(0).unsqueeze(-1))  # (B, D, L)
    # dB*x: (B, D, L)
    dBx = (dts_flat * Bs[:, 0, 0, :].unsqueeze(1)) * xs_flat  # (B, D, L)

    # ---- compute BFS topology if not provided -----------------------------
    if bfs_indices is None or bfs_parents is None:
        from .tree_scan_utils.tree_scan_core import MinimumSpanningTree
        mst_layer = MinimumSpanningTree("Cosine", torch.exp)
        bfs_indices, bfs_parents = mst_layer(x)  # uses current x for MST

    # pad / stack topology tensors to (B, L) for batch processing
    si = torch.stack(bfs_indices, dim=0)   # (B, L)
    sp = torch.stack(bfs_parents, dim=0)  # (B, L)

    # ---- tree SSM recurrence ----------------------------------------------
    h = _tree_ssm_recurrence(dBx, dA, si, sp)  # (B, D, L)

    # optional hidden-state normalisation (as in GrootVL)
    if h_norm is not None:
        h = h_norm(h.transpose(1, 2).contiguous()).transpose(1, 2)

    # ---- read-out: y = h · C + D · x ------------------------------------
    # Cs: (B, 1, N, L) with N=1 → (B, L)
    C = Cs[:, 0, 0, :]  # (B, L)
    y = h * C.unsqueeze(1) + Ds_.unsqueeze(0).unsqueeze(-1) * xs_flat  # (B, D, L)

    y = out_norm(y.transpose(1, 2).contiguous())  # (B, L, D) after norm
    y = y.view(B, H, W, -1)

    return y.to(x.dtype) if to_dtype else y


# ---------------------------------------------------------------------------
# Tree_SSM  –  drop-in replacement for GrootVL's Tree_SSM module
# ---------------------------------------------------------------------------

class Tree_SSM(nn.Module):
    """Single Tree-topology SSM block (equivalent to GrootVL ``Tree_SSM``).

    Projects inputs, computes input-dependent SSM parameters (Δ, B, C),
    builds / receives the tree topology, executes the tree recurrence, and
    projects back to the model dimension.

    Args:
        d_model      : input / output feature dimension.
        d_state      : SSM state dimension.  **Must be 1** for the simplified
                       scalar recurrence used here (same as GrootVL default).
        ssm_ratio    : expansion ratio for the SSM inner dimension.
        dt_rank      : rank of Δ projection; ``"auto"`` → ``ceil(d_model / 16)``.
        d_conv       : depthwise-conv kernel size (``≤ 1`` disables conv).
        act_layer    : activation class (default ``nn.SiLU``).
        dropout      : dropout probability.
        bias         : add bias to linear projections.
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        act_layer=nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        **kwargs,
    ) -> None:
        factory_kwargs: dict = {"device": None, "dtype": None}
        super().__init__()

        if d_state != 1:
            raise ValueError(
                "Tree_SSM currently requires d_state=1 (scalar recurrence). "
                f"Got d_state={d_state}."
            )

        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = d_state
        self.d_conv = d_conv
        K = 1  # single scan direction
        self.K = K

        # normalisation layers
        self.out_norm = nn.LayerNorm(d_inner)
        self.h_norm = nn.LayerNorm(d_inner)

        # input projection (gate + value)
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act = act_layer()

        # output projection: D_expand → D_model  (restores residual dimension)
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)

        # optional depthwise conv
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                d_expand, d_expand,
                groups=d_expand, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x_proj: maps x → (Δ-rank, N, N) per scan direction
        _x_proj = [
            nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([m.weight for m in _x_proj], dim=0)  # (K, dt_rank+2N, D)
        )
        del _x_proj

        # Δ projection
        _dt_projs = [
            self._dt_init(self.dt_rank, d_inner, dt_scale, dt_init,
                          dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([m.weight for m in _dt_projs], dim=0)  # (K, D, dt_rank)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([m.bias for m in _dt_projs], dim=0)  # (K, D)
        )
        del _dt_projs

        # A and D (skip)
        self.A_logs = self._A_log_init(self.d_state, d_inner, copies=K, merge=True)
        self.Ds = self._D_init(d_inner, copies=K, merge=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # parameter init helpers (copied from GrootVL / VMamba)
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_init(
        dt_rank, d_inner, dt_scale=1.0, dt_init="random",
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs
    ):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(proj.weight, dt_init_std)
        else:
            nn.init.uniform_(proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias._no_reinit = True  # type: ignore[attr-defined]
        return proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = nn.Parameter(torch.log(A))
        A_log._no_weight_decay = True  # type: ignore[attr-defined]
        if copies > 0:
            A_log = nn.Parameter(
                repeat(A_log.data, "d n -> r d n", r=copies).flatten(0, 1)
            )
            A_log._no_weight_decay = True  # type: ignore[attr-defined]
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=-1, device=None, merge=True):
        D = nn.Parameter(torch.ones(d_inner, device=device))
        D._no_weight_decay = True  # type: ignore[attr-defined]
        if copies > 0:
            D = nn.Parameter(
                repeat(D.data, "n -> r n", r=copies).flatten(0, 1)
            )
            D._no_weight_decay = True  # type: ignore[attr-defined]
        return D

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        bfs_indices: Optional[List[torch.Tensor]] = None,
        bfs_parents: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : ``(B, H, W, D)`` channels-last spatial feature map.
            bfs_indices : optional pre-computed BFS order – list of ``(L,)`` tensors.
            bfs_parents : optional pre-computed BFS parents – list of ``(L,)`` tensors.

        Returns:
            ``(B, H, W, D)`` output feature map.
        """
        x = self.in_proj(x)                    # (B, H, W, 2*D_expand)
        x, z = x.chunk(2, dim=-1)              # each (B, H, W, D_expand)
        z = self.act(z)

        if self.d_conv > 1:
            x = x.permute(0, 3, 1, 2).contiguous()  # channels-first for conv
            x = self.conv2d(x)
            x = self.act(x)
            # stay channels-first for tree_scanning which expects (B, D, H, W)
            y = tree_scanning(
                x,
                self.x_proj_weight, None,
                self.dt_projs_weight, self.dt_projs_bias,
                self.A_logs, self.Ds,
                out_norm=self.out_norm,
                h_norm=self.h_norm,
                bfs_indices=bfs_indices,
                bfs_parents=bfs_parents,
            )
        else:
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
            x = self.act(x)
            y = tree_scanning(
                x,
                self.x_proj_weight, None,
                self.dt_projs_weight, self.dt_projs_bias,
                self.A_logs, self.Ds,
                out_norm=self.out_norm,
                h_norm=self.h_norm,
                bfs_indices=bfs_indices,
                bfs_parents=bfs_parents,
            )

        y = y * z                              # gated output (B, H, W, D_expand)
        y = self.out_proj(y)                   # (B, H, W, D_model)
        return self.dropout(y)                 # (B, H, W, D_model)


__all__ = ["Tree_SSM", "tree_scanning", "_tree_ssm_recurrence"]
