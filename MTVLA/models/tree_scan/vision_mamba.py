"""
Vision Mamba Encoder – visual feature extractor for MemoryTreeVLA.

Replaces the ViT visual backbone with a hierarchical tree-topology SSM encoder
based on GrootVL (MambaTree, NeurIPS 2024 Spotlight).

Architecture:
  Input image (B, 3, H_img, W_img)
      │
  StemLayer  → (B, C, H/4, W/4)  [channels-last after stem]
      │
  N × VisionMambaLayer  (Tree_SSM + MLP, no downsampling)
      │
  Optional DownsampleLayer → (B, 2C, H/8, W/8)  [for hierarchical staging]
      │
  Output: Z_v  (B, L, D)  – flattened spatial sequence

For MemoryTreeVLA we use a *lightweight* single-stage config that yields a
fixed-size token sequence Z_v consumed by the MultimodalMamba fusion module.

Reference:
  https://github.com/EasonXiao-888/MambaTree/blob/main/GrootV/classification/models/grootv.py
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import DropPath, trunc_normal_  # type: ignore[import]  # timm < 0.9
except ImportError:
    from timm.layers import DropPath, trunc_normal_  # type: ignore[import]  # timm >= 0.9

from .tree_scanning import Tree_SSM
from .tree_scan_utils.tree_scan_core import MinimumSpanningTree


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

def _build_norm(dim: int, norm_type: str = "LN", eps: float = 1e-6) -> nn.Module:
    if norm_type == "LN":
        return nn.LayerNorm(dim, eps=eps)
    if norm_type == "BN":
        return nn.BatchNorm2d(dim)
    raise ValueError(f"Unknown norm: {norm_type}")


class StemLayer(nn.Module):
    """Patch-embed equivalent: 2× stride-2 convolutions → 1/4 resolution.

    Output is in *channels-last* format ``(B, H, W, C)``.
    """

    def __init__(self, in_chans: int = 3, out_chans: int = 96) -> None:
        super().__init__()
        mid = out_chans // 2
        self.conv1 = nn.Conv2d(in_chans, mid, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, out_chans, 3, stride=2, padding=1)
        self.ln = nn.LayerNorm(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)                          # (B, C, H/4, W/4)
        return self.ln(x.permute(0, 2, 3, 1))      # (B, H/4, W/4, C)


class DownsampleLayer(nn.Module):
    """Stride-2 conv downsampling + LayerNorm (channels-last I/O)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False)
        self.ln = nn.LayerNorm(channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 3, 1, 2))      # channels-first for conv
        return self.ln(x.permute(0, 2, 3, 1))      # back to channels-last


class MLPLayer(nn.Module):
    """Standard 2-layer feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# VisionMambaLayer  (= GrootVLayer)
# ---------------------------------------------------------------------------

class VisionMambaLayer(nn.Module):
    """Pre-norm Tree_SSM + MLP residual block.

    Computes MST on-the-fly from the current feature map so the tree topology
    is *input-adaptive* (key feature of GrootVL).

    Args:
        channels    : feature dimension.
        mlp_ratio   : MLP hidden-dim multiplier.
        drop_path   : stochastic depth rate.
        layer_scale : if not ``None``, initialise layer-scale gammas to this value.
        reuse_tree  : if ``True``, re-use the MST computed in Tree_SSM for the
                      MLP pass (no effect on current impl; kept for API compat).
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
        **ssm_kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ssm = Tree_SSM(d_model=channels, d_state=1, ssm_ratio=2.0, **ssm_kwargs)
        self.mlp = MLPLayer(channels, int(channels * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            _ls = float(layer_scale)  # type: ignore[arg-type]
            self.gamma1 = nn.Parameter(_ls * torch.ones(channels))
            self.gamma2 = nn.Parameter(_ls * torch.ones(channels))

        # shared MST builder for this layer
        self._mst = MinimumSpanningTree("Cosine", torch.exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : ``(B, H, W, C)`` channels-last feature map.
        Returns:
            ``(B, H, W, C)``
        """
        B, H, W, C = x.shape

        # Build MST from current features (input-adaptive topology)
        with torch.no_grad():
            fm = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            bfs_idx, bfs_par = self._mst(fm)

        if self.layer_scale:
            x = x + self.drop_path(
                self.gamma1 * self.ssm(self.norm1(x), bfs_idx, bfs_par)
            )
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ssm(self.norm1(x), bfs_idx, bfs_par))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# VisionMambaBlock  (= GrootVBlock)
# ---------------------------------------------------------------------------

class VisionMambaBlock(nn.Module):
    """Stack of ``VisionMambaLayer``s with optional downsampling."""

    def __init__(
        self,
        channels: int,
        depth: int,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        layer_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            VisionMambaLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layer_scale=layer_scale,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(channels)
        self.downsample = DownsampleLayer(channels) if downsample else None

    def forward(
        self, x: torch.Tensor, return_before_downsample: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pre = x
        if self.downsample is not None:
            x = self.downsample(x)
        if return_before_downsample:
            return x, pre
        return x


# ---------------------------------------------------------------------------
# VisionMamba  –  full hierarchical encoder
# ---------------------------------------------------------------------------

class VisionMamba(nn.Module):
    """Hierarchical tree-topology Mamba vision encoder.

    Produces a spatial feature sequence **Z_v** ``(B, L, D)`` consumed by the
    MultimodalMamba fusion module in MemoryTreeVLA.

    Args:
        in_chans      : input image channels (default 3).
        channels      : base channel width (default 64).
        depths        : list of layer depths per stage.
        out_dim       : output projection dimension.  ``None`` → last-stage dim.
        drop_path_rate : stochastic depth budget.
        layer_scale   : layer-scale init value; ``None`` disables layer-scale.

    Example::

        encoder = VisionMamba(channels=64, depths=[2, 2], out_dim=256)
        Z_v = encoder(images)   # (B, H*W//4 or similar, 256)
    """

    def __init__(
        self,
        in_chans: int = 3,
        channels: int = 64,
        depths: List[int] = (2, 2),  # type: ignore[assignment]
        out_dim: Optional[int] = None,
        drop_path_rate: float = 0.1,
        layer_scale: Optional[float] = None,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_stages = len(depths)
        self.stem = StemLayer(in_chans, channels)

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            stage = VisionMambaBlock(
                channels=int(channels * 2 ** i),
                depth=depth,
                downsample=(i < self.num_stages - 1),
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]): sum(depths[:i + 1])],
                layer_scale=layer_scale,
            )
            self.stages.append(stage)

        final_dim = int(channels * 2 ** (self.num_stages - 1))
        self.out_proj = (
            nn.Linear(final_dim, out_dim) if out_dim and out_dim != final_dim else nn.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : ``(B, 3, H, W)`` RGB image batch.

        Returns:
            Z_v : ``(B, L, D)`` visual token sequence.
        """
        x = self.stem(x)                        # (B, H', W', C)
        for stage in self.stages:
            x = stage(x)                         # (B, H'', W'', C')

        B, H, W, C = x.shape
        z_v = x.reshape(B, H * W, C)            # (B, L, C)
        return self.out_proj(z_v)               # (B, L, D)


__all__ = ["VisionMamba", "VisionMambaBlock", "VisionMambaLayer", "StemLayer"]
