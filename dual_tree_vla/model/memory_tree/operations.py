"""
Tree operations: Reinforcement, Semantic Elevation, Pruning.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .node import MemoryNode
from .tree import HierarchicalMemoryTree


# ======================================================================= #
#  Operation ① — Memory Reinforcement                                      #
# ======================================================================= #

def merge(
    tree: HierarchicalMemoryTree,
    z_v: torch.Tensor,
    a: torch.Tensor,
):
    """Merge operation wrapper: Welford update on active leaf (delegates to tree)."""
    if tree.active_id is None:
        tree.insert(z_v=z_v, a=a, force_branch=False, s_current=None)
        return
    tree._merge_update(tree.nodes[tree.active_id], z_v, a)


def branch(
    tree: HierarchicalMemoryTree,
    z_v: torch.Tensor,
    a: torch.Tensor,
    s_current: Optional[torch.Tensor] = None,
):
    """Branch operation wrapper: semantic-aware split (delegates to tree)."""
    if tree.root_id is None:
        tree.insert(z_v=z_v, a=a, force_branch=True, s_current=s_current)
        return
    tree._branch_split(z_v=z_v, a=a, s_current=s_current)

def reinforce(
    tree: HierarchicalMemoryTree,
    grad_norms: Dict[int, float],
    eta: float = 0.01,
    theta_grad: float = 0.1,
):
    """
    梯度驱动的节点权重更新:
        w_i ← w_i + η · ‖∇L‖₂   if  ‖∇L‖₂ > θ_grad

    只更新权重，不修改嵌入（叶子/抽象节点的嵌入分别由 Welford 和 MLPElevation 管理）。
    """
    for node_id, grad_norm in grad_norms.items():
        if node_id not in tree.nodes:
            continue
        if grad_norm > theta_grad:
            tree.nodes[node_id].w += eta * grad_norm


# ======================================================================= #
#  Operation ② — Semantic Elevation                                        #
# ======================================================================= #

class MLPElevation(nn.Module):
    """
    语义提升 MLP：对叶子子节点的 z_v 加权池化后生成抽象父节点的语义嵌入 s_abs。

    输入:  z_pool (d,)  — 叶子子节点 z_v 的加权均值
    输出:  s_abs  (d,)  — 抽象节点语义嵌入

    注意：输入为 d 维（仅视觉），不再拼接语义嵌入（叶子节点不存 s）。
    """

    def __init__(self, d: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d * 2
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.LayerNorm(d),
        )

    def forward(self, z_pool: torch.Tensor) -> torch.Tensor:
        """z_pool: (d,) → s_abs: (d,)"""
        return self.net(z_pool)


def semantic_elevation(
    tree: HierarchicalMemoryTree,
    parent_id: int,
    mlp_elev: MLPElevation,
    device: torch.device = torch.device("cpu"),
) -> Optional[int]:
    """
    在 parent_id 节点上触发语义提升。

    从其叶子子节点中按权重选出 top-⌊K/2⌋ 个作为组 G，
    用 MLPElevation(z_pool) 生成抽象节点 v_abs，插入在 parent 与 G 之间。

    返回 v_abs 的 node_id，若跳过则返回 None。
    """
    v_p = tree.nodes[parent_id]
    children = [nid for nid in v_p.children_ids
                if tree.nodes[nid].is_leaf()]       # 只提升叶子子节点

    if len(children) < 2:
        return None

    K = len(children)
    group_size = max(2, K // 2)

    # 按权重降序排列，选 top group_size
    sorted_children = sorted(children, key=lambda nid: tree.nodes[nid].w, reverse=True)
    G = sorted_children[:group_size]

    # 加权池化叶子节点的 z_v
    ws = torch.tensor([tree.nodes[nid].w for nid in G], dtype=torch.float, device=device)
    ws = ws / ws.sum()

    z_pool = sum(ws[i] * tree.nodes[nid].z_v.to(device) for i, nid in enumerate(G))

    with torch.no_grad():
        s_abs = mlp_elev(z_pool.float())

    w_abs = sum(tree.nodes[nid].w for nid in G)

    # 创建抽象节点 v_abs — 只存 s 和 w，不存 z_v 或 a_hist
    abs_id = tree.alloc_id()
    v_abs = MemoryNode(
        node_id=abs_id,
        s=s_abs.detach().cpu(),
        w=w_abs,
        parent_id=parent_id,
        children_ids=list(G),
    )
    tree.add_node(v_abs)

    # 重连：parent 的 children 列表中用 v_abs 替换 G
    v_p.children_ids = [nid for nid in v_p.children_ids if nid not in G]
    v_p.children_ids.append(abs_id)

    # 更新 G 成员的 parent 指针
    for nid in G:
        tree.nodes[nid].parent_id = abs_id

    return abs_id


def propagate_elevation_to_root(
    tree: HierarchicalMemoryTree,
    start_id: int,
    mlp_elev: MLPElevation,
    device: torch.device = torch.device("cpu"),
):
    """
    从 start_id 开始沿父链向上，逐节点更新每个抽象节点的语义嵌入 s，直至根节点。

    每个抽象节点的 s 由其**所有直接子节点**的嵌入加权池化后经 MLPElevation 所得：
      - 叶子子节点 → 提供 z_v
      - 抽象子节点 → 提供 s（已由下层向上更新过的最新值）

    自下而上保证每一层用的都是已更新的子层嵌入，避免使用过期语义做高层概括。
    """
    current_id = start_id
    while current_id is not None:
        node = tree.nodes[current_id]
        if node.is_leaf() or not node.children_ids:
            current_id = node.parent_id
            continue

        embeds: list = []
        weights: list = []
        for cid in node.children_ids:
            if cid not in tree.nodes:
                continue
            child = tree.nodes[cid]
            if child.is_leaf():
                if child.z_v is not None:
                    embeds.append(child.z_v.float())
                    weights.append(child.w)
            else:
                if child.s is not None:
                    embeds.append(child.s.float())
                    weights.append(child.w)

        if embeds:
            stacked = torch.stack(embeds)
            wt = torch.tensor(weights, dtype=torch.float, device=stacked.device)
            wt = wt / wt.sum()
            z_pool = (stacked * wt.unsqueeze(1)).sum(0).to(device)
            with torch.no_grad():
                node.s = mlp_elev(z_pool.float()).detach().cpu()

        current_id = node.parent_id


# ======================================================================= #
#  Operation ③ — Pruning                                                   #
# ======================================================================= #

def prune(
    tree: HierarchicalMemoryTree,
    theta_w: float = 0.3,
) -> List[int]:
    """
    删除权重 w_i < theta_w 的叶子节点（级联删除）。
    返回已删除的 node_id 列表。
    """
    pruned: List[int] = []

    changed = True
    while changed:
        changed = False
        for node_id in list(tree.nodes.keys()):
            node = tree.nodes[node_id]
            if node.is_leaf() and node.w < theta_w and not node.is_root():
                par = tree.nodes[node.parent_id]
                par.children_ids.remove(node_id)
                if tree.active_id == node_id:
                    tree.active_id = node.parent_id
                del tree.nodes[node_id]
                pruned.append(node_id)
                changed = True

    return pruned
