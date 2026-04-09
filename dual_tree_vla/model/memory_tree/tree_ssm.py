"""
Tree-SSM Readout — BFS-order Mamba-style tree recurrence over abstract nodes only.

**只对抽象节点（非叶子节点）执行 SSM 扫描**：

  抽象节点   输入: [s, log_w]   dim = d + 1   → abs_in_proj → d_ssm

叶子节点不参与 Readout，其语义信息已在 MLPElevation 阶段被归纳入父抽象节点的 s 中。
投影到 d_ssm 后，所有节点共享同一组 SSM 参数（Δ, A, B, C, D）。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree import HierarchicalMemoryTree


class TreeSSMReadout(nn.Module):
    """
    Input  : HierarchicalMemoryTree
    Output : Y ∈ R^{N_abs × d_ssm}  (BFS 顺序，仅抽象节点)

    只扫描抽象节点（abstract nodes，即 is_leaf()==False）：
      抽象节点  → abs_in_proj([s; log_w])   → x_i (d_ssm,)

    抽象节点的父节点必然也是抽象节点（或 None=根），父子链在抽象层级内自洽。

    树 SSM 递推（ZOH 离散化，父节点传播）:
      Δ_i = softplus(W_Δ x_i) ⊙ σ(W_w log_w_i)
      Ā_i = exp(Δ_i · A)
      B̄_i = Δ_i · B(x_i)
      h_i = Ā_i ⊙ h_{par(i)} + B̄_i ⊙ x_i
      y_i = C(x_i)ᵀ h_i + D ⊙ x_i
    """

    def __init__(
        self,
        d_node: int,           # 语义嵌入维度 (= d)
        d_ssm: int,            # SSM 内部维度
        d_state: int = 16,
        max_depth: Optional[int] = None,
    ):
        super().__init__()
        self.d_ssm   = d_ssm
        self.d_state = d_state
        self.max_depth = max_depth

        # ── 抽象节点输入投影：[s, log_w] ────────────────────────────
        self.abs_in_proj = nn.Linear(d_node + 1, d_ssm)

        # ── 权重自适应时间步 ─────────────────────────────────────────
        self.W_delta = nn.Linear(d_ssm, d_ssm, bias=True)
        self.W_w     = nn.Linear(1, d_ssm, bias=True)

        # ── SSM 参数（S4D real init，A 负实数以保证稳定性）──────────
        A_init = torch.arange(1, d_state + 1, dtype=torch.float) \
                      .unsqueeze(0).expand(d_ssm, -1)
        self.A_log  = nn.Parameter(torch.log(A_init.clone()))  # (d_ssm, d_state)
        self.D      = nn.Parameter(torch.ones(d_ssm))

        # ── 选择性 B, C 投影 ────────────────────────────────────────
        self.B_proj = nn.Linear(d_ssm, d_state)
        self.C_proj = nn.Linear(d_ssm, d_state)

        # ── 输出归一化 ───────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_ssm)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.abs_in_proj.weight)
        nn.init.zeros_(self.abs_in_proj.bias)
        nn.init.zeros_(self.W_delta.bias)
        # delta bias 初始化到合理时间步范围
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_ssm) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.W_delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # ------------------------------------------------------------------ #

    def _node_input(
        self,
        node,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """将抽象节点字段投影为 x_i (d_ssm,)。仅处理抽象节点。"""
        log_w = torch.log(torch.tensor([node.w + 1e-6], device=device, dtype=dtype))
        s = node.s
        if s is None:
            s = torch.zeros(self.abs_in_proj.in_features - 1,
                            device=device, dtype=dtype)
        else:
            s = s.to(device=device, dtype=dtype)
        feat = torch.cat([s, log_w], dim=0)
        return self.abs_in_proj(feat)

    def forward(
        self,
        tree: HierarchicalMemoryTree,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        返回 Y ∈ R^{N_sem × d_ssm}，N_sem 为 BFS 顺序中的语义节点数。
        仅扫描语义节点（node.s is not None）。

        注意：剪枝后语义节点可能因子节点被删而变为叶子（is_leaf()==True），
        但只要节点持有语义嵌入 s，就仍需参与 Readout — 因此以 s is not None
        作为判断标准，而非 is_leaf()。
        """
        if self.max_depth is not None:
            all_ids = tree.bfs_order_up_to_depth(self.max_depth)
        else:
            all_ids = tree.bfs_order()

        # 只保留语义节点（s is not None，无论是否有子节点）
        bfs_ids = [nid for nid in all_ids if tree.nodes[nid].s is not None]

        if not bfs_ids:
            dev = next(self.parameters()).device if device is None else device
            return torch.zeros(1, self.d_ssm, device=dev)

        device    = next(self.parameters()).device
        wt_dtype  = self.abs_in_proj.weight.dtype

        # ── 构建投影后的输入矩阵 X_p ────────────────────────────────
        rows = [self._node_input(tree.nodes[nid], device, wt_dtype)
                for nid in bfs_ids]
        X_p  = torch.stack(rows, dim=0)          # (N, d_ssm)

        # ── 权重自适应时间步 ─────────────────────────────────────────
        log_ws = torch.tensor(
            [[math.log(tree.nodes[nid].w + 1e-6)] for nid in bfs_ids],
            device=device, dtype=wt_dtype,
        )                                         # (N, 1)
        delta = F.softplus(self.W_delta(X_p)) \
              * torch.sigmoid(self.W_w(log_ws))   # (N, d_ssm)

        # ── SSM 参数 ─────────────────────────────────────────────────
        A = -torch.exp(self.A_log.to(dtype=wt_dtype))   # (d_ssm, d_state)
        B = self.B_proj(X_p)                             # (N, d_state)
        C = self.C_proj(X_p)                             # (N, d_state)

        # ── BFS 顺序树递推（预先批量计算 A_bar/Bx，减少逐节点 GPU kernel 启动）──
        N        = len(bfs_ids)
        node2idx = {nid: i for i, nid in enumerate(bfs_ids)}

        # 批量预计算：A_bar[i] = exp(Δ[i]·A)，Bx[i] = (Δ[i]⊙x[i])[:,None]·B[i][None,:]
        A_bar_all = torch.exp(delta.unsqueeze(2) * A)                      # (N, d_ssm, d_state)
        Bx_all    = (delta * X_p).unsqueeze(2) * B.unsqueeze(1)            # (N, d_ssm, d_state)

        H = X_p.new_zeros(N, self.d_ssm, self.d_state)
        for i, nid in enumerate(bfs_ids):
            par_id = tree.nodes[nid].parent_id
            h_par  = (H[node2idx[par_id]]
                      if par_id is not None and par_id in node2idx
                      else H.new_zeros(self.d_ssm, self.d_state))
            H[i]   = A_bar_all[i] * h_par + Bx_all[i]

        # 批量 Y 计算（脱离循环，单次 GPU kernel）
        Y = (H * C.unsqueeze(1)).sum(dim=2) + self.D * X_p                # (N, d_ssm)

        return self.out_norm(Y)   # (N, d_ssm)
