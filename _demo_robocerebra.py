"""
记忆树真实数据演示 —— 使用 RoboCerebra coffee_table/case1 的一条轨迹。

参数初始化策略（仅初始化阶段使用标注 JSON）：
  1. 读取子任务边界标注，在全量动作序列上统计余弦距离分布，
     确定 ACTION_TAU（边界 p25 与 within p85 的均值）。
  2. 在全量数据上模拟分支检测，对检测到的分支点按子任务分类，
     统计同子任务相邻分支的动作窗口均值余弦距离 vs 跨子任务对应距离，
     取 (intra_p75 + inter_p25) / 2 作为 MOUNT_TAU。
     训练时 MLPElevation 输出替代动作窗口均值，校准思路相同：
     令同阶段语义距离 < MOUNT_TAU，跨阶段距离 > MOUNT_TAU。

语义向量计算（运行时不使用标注）：
  s_current = normalize([mean_action_window_7d, zeros_77d])
  动作历史窗口均值方向映射到 D 维空间（仅前 D_A 维有效）

  此设计保证：
    - 同阶段连续分支 → 动作方向相近 → d < MOUNT_TAU → Case B/C（树向上攀爬）
    - 跨阶段分支   → 动作方向大变 → d >= MOUNT_TAU → Case A（挂载到当前层）

检测循环不使用任何标注信息，完全依赖动作变化信号。

输出：
  logs/demo_robocerebra_log.json  —— 超参（含校准统计）、分支快照（树形状）、汇总
"""
import sys, json, time, types
from collections import deque
from pathlib import Path

sys.path.insert(0, ".")

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dual_tree_vla.model.memory_tree import HierarchicalMemoryTree

# ─────────────────────────── 固定超参 ────────────────────────────────────
CASE_DIR        = Path("dataset/RoboCerebra/RoboCerebra_trainset/coffee_table/case1")
SUBSAMPLE       = 20      # 每隔 20 帧取 1 帧
D               = 84      # states / 语义向量维度
D_A             = 7       # actions 维度
A_WIN           = 6       # 动作历史窗口大小（demo 帧数）
MAX_DEPTH       = 32      # 树最大深度（demo 中足够大以保留全部历史）
LOG_SAVE        = Path("logs/demo_robocerebra_log.json")
# ACTION_TAU / MOUNT_TAU / ACTION_COOLDOWN 由校准阶段确定
torch.manual_seed(0)

# ─────────────────────────── 加载数据 ────────────────────────────────────
print("loading data ...")
with h5py.File(CASE_DIR / "demo.hdf5", "r") as f:
    demo         = f["data/demo_1"]
    actions_full = torch.from_numpy(demo["actions"][:].astype(np.float32))
    states_full  = torch.from_numpy(demo["states" ][:].astype(np.float32))
    T_full       = actions_full.shape[0]

states_sub  = states_full [::SUBSAMPLE]
actions_sub = actions_full[::SUBSAMPLE]
T           = states_sub.shape[0]
orig_indices = list(range(0, SUBSAMPLE * T, SUBSAMPLE))
print(f"   original frames: {T_full}  ->  subsampled: {T}")

with open(CASE_DIR / "task_description.json") as fp:
    td    = json.load(fp)
steps = td["steps"]

# ─────────────────────────── 语义向量辅助 ────────────────────────────────
def action_window_sem(buf: deque, D: int = D, D_A: int = D_A, min_norm: float = 0.05):
    """
    用近期动作窗口均值构造 D 维语义向量。

    取 buf 中 L2 范数 > min_norm 的动作，计算均值后做单位化，
    填入 D 维零向量的前 D_A 个位置（已为单位向量）。

    Returns: (D,) tensor，若无有效动作则返回 None。
    """
    valid = [a for a in buf if a.float().norm().item() > min_norm]
    if not valid:
        return None
    mean_a = torch.stack(valid).float().mean(0)  # (D_A,)
    if mean_a.norm().item() < 1e-4:
        return None
    s = torch.zeros(D)
    s[:D_A] = F.normalize(mean_a, dim=0)
    return s   # 单位向量（||s|| = 1，前 D_A 维为 mean_action 方向，余为 0）

# ─────────────────────────── 校准阶段 ────────────────────────────────────
def calibrate(actions_full, steps, T_full, T_demo, SUBSAMPLE, A_WIN, D, D_A):
    """
    使用 GT 子任务边界在全量数据上统计分布，校准三个超参。

    ACTION_TAU  = ( within-sub p85 + boundary p25 ) / 2
        在"正常动作波动上界"与"真实切换下界"之间取均值。
        排除幅度近零的动作帧（帧幅度 < 1e-3 时归一化无意义）。

    MOUNT_TAU   = (intra_adj_p75 + inter_adj_p25) / 2
        在全量序列上模拟 ActionChangeDetector，找出所有分支点。
        将相邻分支对按"同子任务 / 跨子任务"分类，统计动作窗口均值余弦距离。
        intra_adj_p75 使同阶段内大部分分支对触发 Case B/C；
        inter_adj_p25 保证跨阶段分支对触发 Case A。

    COOLDOWN    = max(2, 采样后每子任务平均帧数 // 6)
        减小至 //6 提高分支频率。
    """
    # ── 子任务边界标注 ────────────────────────────────────────────────────
    sub_ids = np.zeros(T_full, dtype=np.int64)
    for s in steps:
        s_idx = int(s["step_number"]) - 1
        start = int(s["timestep"]["start"])
        end   = min(int(s["timestep"]["end"]) + 1, T_full)
        sub_ids[start:end] = s_idx

    # ── ACTION_TAU（排除近零动作帧） ─────────────────────────────────────
    a_raw  = actions_full.float()
    valid  = a_raw.norm(dim=1) > 1e-3
    a_n    = torch.zeros_like(a_raw)
    a_n[valid] = F.normalize(a_raw[valid], dim=1)
    both_v = valid[1:] & valid[:-1]
    adj_d  = (1.0 - (a_n[1:] * a_n[:-1]).sum(dim=1))[both_v]
    is_b   = torch.from_numpy((sub_ids[1:] != sub_ids[:-1]))[both_v]
    w_d    = adj_d[~is_b]
    b_d    = adj_d[is_b]
    p85w   = float(w_d.quantile(0.85)) if len(w_d) > 0 else 0.5
    p25b   = float(b_d.quantile(0.25)) if len(b_d) > 0 else p85w
    action_tau = (p85w + p25b) / 2.0

    n_sub    = int(sub_ids.max()) + 1
    cooldown = max(2, int((T_full / SUBSAMPLE) / max(n_sub, 1) // 6))

    # ── MOUNT_TAU（基于动作窗口均值语义代理） ────────────────────────────
    # 在子采样序列上模拟 ActionChangeDetector，记录分支点及其动作窗口语义向量
    actions_demo = actions_full[::SUBSAMPLE][:T_demo]
    sub_demo     = sub_ids[::SUBSAMPLE][:T_demo]

    sim_buf      = deque(maxlen=A_WIN)
    sim_branches = []   # (t_demo, subtask_id, s_vec)
    prev_an      = None
    cd           = 0
    for t in range(T_demo):
        a    = actions_demo[t]
        a_n_t= F.normalize(a.float(), dim=0)
        sim_buf.append(a.clone())
        force = False
        if prev_an is None:
            force = True
        elif cd == 0 and float(1.0 - (a_n_t * prev_an).sum()) >= action_tau:
            force = True
            cd    = cooldown
        prev_an = a_n_t
        if cd > 0:
            cd -= 1
        if force:
            sv = action_window_sem(sim_buf, D, D_A)
            if sv is not None:
                sim_branches.append((t, int(sub_demo[t]), sv))

    # 计算相邻分支对的动作窗口余弦距离，按同/跨子任务分类
    intra_adj = []
    inter_adj = []
    for i in range(1, len(sim_branches)):
        si = sim_branches[i - 1][2]
        sj = sim_branches[i][2]
        d  = float(1.0 - (si * sj).sum())
        if sim_branches[i - 1][1] == sim_branches[i][1]:
            intra_adj.append(d)
        else:
            inter_adj.append(d)

    if intra_adj and inter_adj:
        ip75 = float(np.percentile(intra_adj, 75))
        ep25 = float(np.percentile(inter_adj, 25))
        mount_tau = (ip75 + ep25) / 2.0
    elif intra_adj:
        mount_tau = float(np.percentile(intra_adj, 90)) * 1.5
    else:
        mount_tau = 0.30   # fallback

    stats = {
        "action_within_p50":   round(float(w_d.quantile(0.50)) if len(w_d) else 0, 6),
        "action_within_p85":   round(p85w, 6),
        "action_boundary_p25": round(p25b, 6),
        "action_boundary_p50": round(float(b_d.quantile(0.50)) if len(b_d) else 0, 6),
        "n_sim_branches":      len(sim_branches),
        "intra_adj_n":         len(intra_adj),
        "intra_adj_p75":       round(float(np.percentile(intra_adj, 75)) if intra_adj else 0, 6),
        "inter_adj_n":         len(inter_adj),
        "inter_adj_p25":       round(float(np.percentile(inter_adj, 25)) if inter_adj else 0, 6),
        "mount_tau_final":     round(mount_tau, 6),
    }
    return action_tau, mount_tau, cooldown, stats


print("calibrating parameters ...")
ACTION_TAU, MOUNT_TAU, ACTION_COOLDOWN, calib_stats = calibrate(
    actions_full, steps, T_full, T, SUBSAMPLE, A_WIN, D, D_A
)
print(f"   action dist: within-p85={calib_stats['action_within_p85']:.4f}"
      f"  boundary-p25={calib_stats['action_boundary_p25']:.4f}")
print(f"   -> ACTION_TAU={ACTION_TAU:.4f}  COOLDOWN={ACTION_COOLDOWN}")
print(f"   action-window sem: intra_adj_p75={calib_stats['intra_adj_p75']:.4f}"
      f"  inter_adj_p25={calib_stats['inter_adj_p25']:.4f}")
print(f"   -> MOUNT_TAU={MOUNT_TAU:.4f}")

# ─────────────────────────── 动作变化检测器 ──────────────────────────────
class ActionChangeDetector:
    def __init__(self, threshold, cooldown):
        self.threshold = threshold
        self.cooldown  = cooldown
        self._prev_a   = None
        self._cd       = 0

    def update(self, a):
        a_n = F.normalize(a.float(), dim=0)
        if self._prev_a is None:
            self._prev_a = a_n
            return True
        if self._cd > 0:
            self._cd -= 1
            self._prev_a = a_n
            return False
        cos_dist = float(1.0 - (a_n * self._prev_a).sum())
        self._prev_a = a_n
        if cos_dist > self.threshold:
            self._cd = self.cooldown
            return True
        return False


# ─────────────────────────── 树形状序列化 ────────────────────────────────
def tree_shape(tree):
    def _node(nid):
        n = tree.nodes[nid]
        e = {"id": nid, "w": round(float(n.w), 4)}
        if n.s is not None:
            e["type"]   = "SEM"
            e["s_norm"] = round(float(n.s.norm()), 4)
        else:
            e["type"]     = "LEAF"
            e["z_norm"]   = round(float(n.z_v.norm()), 4)
            e["hist_len"] = len(n.a_hist)
            e["active"]   = (nid == tree.active_id)
        e["children"] = [_node(c) for c in n.children_ids]
        return e
    return _node(tree.root_id) if tree.root_id is not None else {}


def count_leaves(tree):
    return sum(1 for n in tree.nodes.values() if n.is_leaf())


# ─────────────────────────── 初始化 ──────────────────────────────────────
detector     = ActionChangeDetector(ACTION_TAU, ACTION_COOLDOWN)
tree         = HierarchicalMemoryTree(d=D, d_a=D_A, mount_tau=MOUNT_TAU)
action_buf   = deque(maxlen=A_WIN)   # 动作历史窗口

branch_cases = []
_CASE_MAP    = {"first": "A", "intermediate": "B", "root_exceeded": "C"}
_orig_cls    = tree._classify_mount.__func__

def _patched_cls(self, s_current, first_abs_id):
    result = _orig_cls(self, s_current, first_abs_id)
    branch_cases.append(_CASE_MAP.get(result[0], result[0]))
    return result

tree._classify_mount = types.MethodType(_patched_cls, tree)

# ─────────────────────────── 主循环 ──────────────────────────────────────
print("building memory tree ...")
t0        = time.time()
snapshots = []

for t_idx in range(T):
    z_v    = states_sub[t_idx]
    a      = actions_sub[t_idx]
    t_orig = orig_indices[t_idx]

    action_buf.append(a.clone())
    force_branch = detector.update(a)

    # 语义向量：动作历史窗口均值方向（运行时不使用任何标注）
    s_cur = None
    if force_branch:
        s_cur = action_window_sem(action_buf, D, D_A)

    tree.insert(z_v, a, force_branch, s_current=s_cur)

    if tree.elevation_pending_parent is not None:
        # demo 中不调用 propagate_elevation_to_root，
        # 直接裁剪深度，节点 s 保持 insert 时写入的 s_cur 值。
        tree._prune_to_max_depth(MAX_DEPTH)
        tree.elevation_pending_parent = None

    if force_branch:
        case = branch_cases[-1] if branch_cases else "?"
        snap = {
            "t_idx":       t_idx,
            "t_orig":      t_orig,
            "branch_case": case,
            "nodes_total": tree.size(),
            "leaves":      count_leaves(tree),
            "tree":        tree_shape(tree),
        }
        snapshots.append(snap)
        print(f"  t={t_idx:4d} (orig={t_orig:5d}) [BRANCH Case {case}]"
              f"  nodes={tree.size()}, leaves={snap['leaves']}")

elapsed  = time.time() - t0
n_branch = len(snapshots)
n_merge  = T - n_branch
case_cnt = {c: branch_cases.count(c) for c in sorted(set(branch_cases)) if c != "?"}

print(f"\n=== summary ===")
print(f"  subsampled frames: {T}")
print(f"  branch: {n_branch}  merge: {n_merge}")
print(f"  case distribution: {case_cnt}")
print(f"  final nodes: {tree.size()}  leaves: {count_leaves(tree)}")

# ─────────────────────────── 保存 JSON ───────────────────────────────────
LOG_SAVE.parent.mkdir(parents=True, exist_ok=True)

log = {
    "case": "coffee_table/case1",
    "hyperparams": {
        "subsample":        SUBSAMPLE,
        "d":                D,
        "d_a":              D_A,
        "a_win":            A_WIN,
        "action_tau":       round(ACTION_TAU, 6),
        "action_cooldown":  ACTION_COOLDOWN,
        "mount_tau":        round(MOUNT_TAU, 6),
        "max_depth":        MAX_DEPTH,
        "calibration":      calib_stats,
    },
    "summary": {
        "total_frames_orig":  T_full,
        "total_frames_demo":  T,
        "n_branch":           n_branch,
        "n_merge":            n_merge,
        "case_distribution":  case_cnt,
        "final_nodes":        tree.size(),
        "final_leaves":       count_leaves(tree),
        "elapsed_s":          round(elapsed, 3),
    },
    "branch_snapshots": snapshots,
    "final_tree": tree_shape(tree),
}

with open(LOG_SAVE, "w", encoding="utf-8") as fp:
    json.dump(log, fp, indent=2, ensure_ascii=False)

print(f"\nlog saved -> {LOG_SAVE}")
print(f"elapsed: {elapsed:.2f}s")