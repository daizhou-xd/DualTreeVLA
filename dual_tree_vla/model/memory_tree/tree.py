from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .node import MemoryNode


class HierarchicalMemoryTree:
    """
    Hierarchical Memory Tree (HMT).

    节点类型:
      叶子节点  — 存储视觉嵌入 z_v、动作历史 a_hist、权重 w
      抽象节点  — 由语义提升（semantic_elevation）创建，只存语义嵌入 s、权重 w

    每次调用 insert() 推进树一帧:
      force_branch=False → 合并（Decision A）: Welford 更新活跃叶子的 z_v
      force_branch=True  → 分支（Decision B）: 创建新叶子，附着到活跃节点的父节点
    """

    def __init__(
        self,
        d: int,              # embedding dim
        d_a: int,            # action dim
        theta_fuse: float = 0.4,   # 保留供外部访问，不再用于内部决策
        K_elev: int = 4,           # 保留供外部访问（已不再作为 elevation 阈值）
        delta_w: float = 0.1,
        tau: float = 0.1,
        mount_tau: float = 0.4,    # 语义挂载点搜索阈值：余弦距离 >= 此值视为「差异大」
    ):
        self.d = d
        self.d_a = d_a
        self.theta_fuse = theta_fuse
        self.K_elev = K_elev
        self.delta_w = delta_w
        self.tau = tau
        self.mount_tau = mount_tau

        self._nodes: Dict[int, MemoryNode] = {}
        self._next_id: int = 0
        self.root_id: Optional[int] = None
        self.active_id: Optional[int] = None
        self.elevation_pending_parent: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Read-only helpers                                                   #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> Dict[int, MemoryNode]:
        return self._nodes

    def size(self) -> int:
        return len(self._nodes)

    def depth(self, node_id: int) -> int:
        return len(self.get_ancestors(node_id))

    def get_ancestors(self, node_id: int) -> List[int]:
        """Ordered from parent → root."""
        ancestors: List[int] = []
        current = self._nodes[node_id]
        while current.parent_id is not None:
            ancestors.append(current.parent_id)
            current = self._nodes[current.parent_id]
        return ancestors

    def bfs_order(self) -> List[int]:
        """Return all node ids in breadth-first order."""
        if self.root_id is None:
            return []
        result, queue = [], [self.root_id]
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            queue.extend(self._nodes[nid].children_ids)
        return result

    def bfs_order_up_to_depth(self, max_depth: int) -> List[int]:
        """
        Return node ids in BFS order, restricted to depth <= max_depth.

        Because depth(parent) = depth(child) - 1, any included node's parent
        is guaranteed to also be included, so hidden-state parent lookups in
        tree_ssm.py remain consistent without extra linking logic.
        """
        if self.root_id is None:
            return []
        result: List[int] = []
        queue = [(self.root_id, 0)]
        while queue:
            nid, d = queue.pop(0)
            if d > max_depth:
                continue
            result.append(nid)
            for cid in self._nodes[nid].children_ids:
                queue.append((cid, d + 1))
        return result

    def ancestor_descendant_pairs(self) -> List[Tuple[int, int]]:
        """All (v_i, v_j) where v_i is a strict ancestor of v_j — for L_prog."""
        pairs: List[Tuple[int, int]] = []
        for nid in self._nodes:
            for anc in self.get_ancestors(nid):
                pairs.append((anc, nid))
        return pairs

    # ------------------------------------------------------------------ #
    #  Mutation API                                                        #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear the tree for a new episode."""
        self._nodes.clear()
        self._next_id = 0
        self.root_id = None
        self.active_id = None
        self.elevation_pending_parent = None

    def insert(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        force_branch: bool,
        s_current: Optional[torch.Tensor] = None,
    ) -> int:
        """
        插入一帧观测，推进记忆树。

        Parameters
        ----------
        z_v          : (d,)   当前帧视觉嵌入
        a            : (d_a,) 当前帧动作
        force_branch : bool   JumpAwareHead 判定的分支决策
        s_current    : (d,) optional — 当前帧语义（由 MLPElevation(z_v) 计算），
                        供分支时在树中向上搜索挂载点；None 时回退到直接挂载父节点。

        Returns
        -------
        active_id : int — 当前活跃叶子节点的 id
        """
        self.elevation_pending_parent = None

        if self.root_id is None:
            # 第一帧：视为跳变点，直接建立两层结构
            #   抽象根（语义探针，s = s_current）
            #     └── 叶子（当前帧）
            probe_s = (s_current.detach().clone().cpu() if s_current is not None
                       else torch.zeros(self.d, dtype=torch.float))
            abs_id = self.alloc_id()
            abs_root = MemoryNode(
                node_id=abs_id,
                s=probe_s,
                w=1.0,
                children_ids=[],
                parent_id=None,
            )
            self._nodes[abs_id] = abs_root
            self.root_id = abs_id
            leaf = self._new_leaf(z_v, a, parent_id=abs_id)
            abs_root.children_ids.append(leaf.node_id)
            self.active_id = leaf.node_id
            self.elevation_pending_parent = abs_id
            return leaf.node_id

        if not force_branch:
            # ── Decision A: 合并，Welford 更新活跃叶子 ──────────────
            self._merge_update(self._nodes[self.active_id], z_v, a)
        else:
            # ── Decision B: 分支，语义感知挂载 ──────────────────────
            self._branch_split(z_v, a, s_current)

        return self.active_id

    def add_node(self, node: MemoryNode):
        """Register an externally constructed node (used by elevation)."""
        self._nodes[node.node_id] = node
        if node.node_id >= self._next_id:
            self._next_id = node.node_id + 1

    def alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _new_leaf(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        parent_id: Optional[int],
    ) -> MemoryNode:
        """创建叶子节点：只存 z_v, a_hist, w。"""
        node = MemoryNode(
            node_id=self._next_id,
            z_v=z_v.detach().clone(),
            a_hist=[a.detach().clone()],
            w=1.0,
            parent_id=parent_id,
        )
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node

    def _merge_update(
        self,
        v_act: MemoryNode,
        z_v: torch.Tensor,
        a: torch.Tensor,
    ):
        """Welford 在线均值更新叶子节点的 z_v，追加动作到 a_hist。"""
        n = len(v_act.a_hist) + 1
        v_act.z_v = (v_act.z_v + (z_v.detach() - v_act.z_v) / n).clone()
        v_act.a_hist.append(a.detach().clone())
        v_act.w += self.delta_w

    def _classify_mount(
        self,
        s_current: Optional[torch.Tensor],
        first_abs_id: int,
    ) -> Tuple[str, int]:
        """
        对即将发生的分支操作进行语义分类，决定挂载策略。

        分类规则（d_k = 1 - cos(s_cur, s_k)）：

          Case A 'first'        : d_first >= τ → 第一个抽象节点就满足，直接挂叶子
          Case B 'intermediate' : d_first < τ 但爬升中某节点 d_k >= τ → 在该节点下
                                  插入语义探针抽象层，再挂叶子（两层）
          Case C 'root_exceeded': 爬升至根仍无 d_k >= τ → 语义已超出整棵树，
                                  创建超级根，将旧根和新叶同挂其下（两层）

        Returns
        -------
        (case_str, node_id) : Tuple[str, int]
        """
        if s_current is None:
            return ('first', first_abs_id)

        s_curr = F.normalize(s_current.cpu().float().unsqueeze(0), dim=-1).squeeze(0)

        def _cos_dist(s: Optional[torch.Tensor]) -> Optional[float]:
            if s is None or s.norm().item() < 1e-6:
                return None
            s_n = F.normalize(s.float().unsqueeze(0), dim=-1).squeeze(0)
            return 1.0 - (s_curr * s_n).sum().item()

        # 检查第一个抽象节点（最低语义）
        d_first = _cos_dist(self._nodes[first_abs_id].s)
        if d_first is not None and d_first >= self.mount_tau:
            return ('first', first_abs_id)

        # 向上爬升
        current_id = self._nodes[first_abs_id].parent_id
        while current_id is not None:
            d_k = _cos_dist(self._nodes[current_id].s)
            if d_k is not None and d_k >= self.mount_tau:
                return ('intermediate', current_id)
            current_id = self._nodes[current_id].parent_id

        # 爬升至根仍未触发 → 语义超出整棵树
        return ('root_exceeded', self.root_id)

    def _branch_split(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        s_current: Optional[torch.Tensor] = None,
    ):
        """
        分支操作，根据语义爬升结果分 3 种情况挂载。

        Case A（最低抽象节点已满足）: 直接挂叶子于第一抽象层（1层新增）
        Case B（中间某节点满足）    : 先挂语义探针抽象节点，再挂叶子（2层新增）
        Case C（爬升超出根）        : 创建超级根，旧根挂其下；语义探针+新叶再下一层（3层新增）

        初始化时树已保证至少存在一层抽象节点，此方法无需处理零抽象层特殊情况。
        每次分支后设置 elevation_pending_parent，由调用方触发
        propagate_elevation_to_root 完成全路径语义更新。
        """
        first_abs_id = self._nodes[self.active_id].parent_id
        case, node_id = self._classify_mount(s_current, first_abs_id)

        if case == 'first':
            # ── Case A：直接在第一抽象节点下挂叶子 ──────────────────
            new_leaf = self._new_leaf(z_v, a, parent_id=node_id)
            self._nodes[node_id].children_ids.append(new_leaf.node_id)
            self.active_id = new_leaf.node_id
            self.elevation_pending_parent = node_id

        elif case == 'intermediate':
            # ── Case B：在 v_k 下插入语义探针抽象层，再挂叶子 ────────
            probe_s = (s_current.detach().clone().cpu() if s_current is not None
                       else torch.zeros(self.d, dtype=torch.float))
            probe_id = self.alloc_id()
            probe = MemoryNode(
                node_id=probe_id,
                s=probe_s,
                w=1.0,
                children_ids=[],
                parent_id=node_id,
            )
            self._nodes[probe_id] = probe
            self._nodes[node_id].children_ids.append(probe_id)
            new_leaf = self._new_leaf(z_v, a, parent_id=probe_id)
            probe.children_ids.append(new_leaf.node_id)
            self.active_id = new_leaf.node_id
            self.elevation_pending_parent = probe_id

        else:  # 'root_exceeded'
            # ── Case C：创建超级根，旧根挂超级根下，语义探针+新叶再下一层 ──
            super_id = self.alloc_id()
            old_root = self._nodes[self.root_id]
            super_root = MemoryNode(
                node_id=super_id,
                s=torch.zeros(self.d, dtype=torch.float),  # 由 propagate 立即更新
                w=old_root.w,
                children_ids=[self.root_id],
                parent_id=None,
            )
            self._nodes[super_id] = super_root
            old_root.parent_id = super_id
            self.root_id = super_id
            # 在超级根下先挂语义探针抽象节点（代表新子任务分支）
            probe_s = (s_current.detach().clone().cpu() if s_current is not None
                       else torch.zeros(self.d, dtype=torch.float))
            probe_id = self.alloc_id()
            probe = MemoryNode(
                node_id=probe_id,
                s=probe_s,
                w=1.0,
                children_ids=[],
                parent_id=super_id,
            )
            self._nodes[probe_id] = probe
            super_root.children_ids.append(probe_id)
            # 再在语义探针下挂新叶子
            new_leaf = self._new_leaf(z_v, a, parent_id=probe_id)
            probe.children_ids.append(new_leaf.node_id)
            self.active_id = new_leaf.node_id
            self.elevation_pending_parent = probe_id

    def _prune_to_max_depth(self, max_depth: int = 4):
        """
        删除所有深度超过 max_depth 的节点（根节点深度 = 0）。

        树倾向于左侧（旧记忆）积累更深，剪枝相当于遗忘远期历史。
        从最深层开始删除，确保先扫清子节点再删父节点。
        """
        if self.root_id is None:
            return
        from collections import deque
        depth_map: Dict[int, int] = {self.root_id: 0}
        q: deque = deque([self.root_id])
        while q:
            nid = q.popleft()
            for cid in list(self._nodes[nid].children_ids):
                if cid in self._nodes:
                    depth_map[cid] = depth_map[nid] + 1
                    q.append(cid)

        to_prune = sorted(
            [nid for nid, d in depth_map.items() if d > max_depth],
            key=lambda x: depth_map[x], reverse=True,
        )
        for nid in to_prune:
            if nid not in self._nodes:
                continue
            node = self._nodes[nid]
            if node.parent_id is not None and node.parent_id in self._nodes:
                parent = self._nodes[node.parent_id]
                if nid in parent.children_ids:
                    parent.children_ids.remove(nid)
            del self._nodes[nid]

        if self.active_id not in self._nodes:
            self.active_id = self._find_rightmost_leaf(self.root_id)

    def _find_rightmost_leaf(self, node_id: int) -> int:
        """返回以 node_id 为根的子树中最右叶子（最新记忆）。"""
        node = self._nodes[node_id]
        if node.is_leaf():
            return node_id
        return self._find_rightmost_leaf(node.children_ids[-1])
