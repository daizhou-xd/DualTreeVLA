from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MemoryNode:
    """
    HMT 节点 — 叶子节点与抽象节点存储完全不同的字段。

    叶子节点（is_leaf() == True）
    ─────────────────────────────
      z_v    : (d,)  视觉嵌入（Welford 在线均值）
      a_hist : List[(d_a,)]  动作历史（供 JumpAwareHead 使用）
      w      : float  节点重要性权重
      s      : None

    抽象节点（is_leaf() == False，由 MLPElevation 创建）
    ─────────────────────────────────────────────────────
      s      : (d,)  语义嵌入（MLPElevation(z_pool) 的输出）
      w      : float  节点重要性权重
      z_v    : None
      a_hist : []（空列表）
    """

    node_id: int
    w: float = 1.0
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # ── 叶子节点字段（抽象节点中为 None / 空）──────────────────────────
    z_v:    Optional[torch.Tensor]  = None       # (d,) 视觉嵌入
    a_hist: List[torch.Tensor]      = field(default_factory=list)  # 动作序列

    # ── 抽象节点字段（叶子节点中为 None）──────────────────────────────
    s: Optional[torch.Tensor] = None             # (d,) 语义嵌入

    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def a_last(self) -> Optional[torch.Tensor]:
        """动作历史中最近的一帧（供 TreeSSMReadout 使用）。"""
        return self.a_hist[-1] if self.a_hist else None

    @property
    def a_mean(self) -> Optional[torch.Tensor]:
        """动作历史的均值（d_a,）。"""
        if not self.a_hist:
            return None
        return torch.stack(self.a_hist).mean(0)

    @property
    def sigma_act(self) -> Optional[torch.Tensor]:
        """动作模长的标准差（标量 tensor），供自监督边界标签计算。"""
        if len(self.a_hist) < 2:
            return self.a_hist[0].new_ones(1) if self.a_hist else None
        stacked = torch.stack(self.a_hist)   # (n, d_a)
        norms   = stacked.norm(dim=-1)       # (n,)
        return norms.std().unsqueeze(0).clamp(min=1e-6)
