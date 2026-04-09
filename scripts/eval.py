"""
DualTreeVLA — Offline Evaluation Script
==========================================

Benchmarks
----------
  robocerebra       RoboCerebra trainset (long-horizon, offline)
  robocerebra_bench RoboCerebraBench — 6 task-type subsets, structured evaluation
                    (Ideal / Memory_Execution / Memory_Exploration /
                     Mix / Observation_Mismatching / Random_Disturbance)
  libero            LIBERO-SPATIAL / OBJECT / GOAL / LONG (NeurIPS 2023)

Metrics
-------
  Action quality (all benchmarks)
    action_l1            Mean per-step L1 error: |a_pred[0] − a_gt|
    action_l2            Mean per-step L2 error: ||a_pred[0] − a_gt||₂

  Memory tree quality (all benchmarks)
    tree_nodes           Avg node count at trajectory end
    tree_depth           Avg max-depth at trajectory end
    tree_branches        Avg branch-creation events per trajectory
    tree_elevations      Avg elevation events per trajectory

  Subtask / progress  (RoboCerebra / RoboCerebraBench)
    subtask_boundary_f1  Branch-creation F1 vs GT subtask boundaries (±tol)
    prog_monotone_rate   Fraction of (ancestor, descendant) pairs where
                         ancestor is farther from the task-goal embedding
    subtask_sr           Fraction of GT boundaries with ≥1 matched branch

Usage
-----
  # RoboCerebraBench — all 6 task types
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_best \\
      --config configs/default.yaml \\
      --dataset robocerebra_bench \\
      --bench_root dataset/RoboCerebra/RoboCerebraBench \\
      --out results/robocerebra_bench.json

  # RoboCerebraBench — specific task types only
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_best \\
      --dataset robocerebra_bench \\
      --bench_root dataset/RoboCerebra/RoboCerebraBench \\
      --task_types Ideal Random_Disturbance \\
      --out results/bench_ideal_rd.json

  # RoboCerebra trainset
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_best \\
      --dataset robocerebra \\
      --data_root dataset/RoboCerebra/RoboCerebra_trainset \\
      --out results/robocerebra_train_eval.json

  # LIBERO-LONG evaluation
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_best \\
      --dataset libero \\
      --data_root dataset/LIBERO \\
      --libero_split long \\
      --out results/libero_long_eval.json

  # Quick test: limit trajectories, custom GPU
  python eval.py --ckpt ... --dataset robocerebra_bench \\
      --bench_root dataset/RoboCerebra/RoboCerebraBench \\
      --device cuda:0 --subsample 4 --max_traj 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml


# ================================================================
#  CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DualTreeVLA offline evaluation")

    # Checkpoint & config
    p.add_argument("--ckpt",    required=True,  help="Path to .pt model checkpoint")
    p.add_argument("--config",  default="configs/default.yaml",
                   help="YAML config used during training")

    # Dataset
    p.add_argument("--dataset",
                   choices=["robocerebra", "robocerebra_bench", "libero"],
                   required=True, help="Which benchmark to evaluate")
    p.add_argument("--data_root", default=None,
                   help="Root directory of the dataset (robocerebra / libero)")
    # RoboCerebraBench-specific arguments
    p.add_argument("--bench_root",
                   default="dataset/RoboCerebra/RoboCerebraBench",
                   help="Root directory of RoboCerebraBench "
                        "(used when --dataset robocerebra_bench)")
    p.add_argument("--task_types", nargs="*", default=None,
                   metavar="TASK_TYPE",
                   help="(robocerebra_bench) Task-type subsets to evaluate. "
                        "Choices: Ideal Memory_Execution Memory_Exploration "
                        "Mix Observation_Mismatching Random_Disturbance. "
                        "Default: all six.")
    p.add_argument("--libero_split",
                   choices=["spatial", "object", "goal", "long"],
                   default="long",
                   help="LIBERO subset (only used when --dataset libero)")
    p.add_argument("--scenes",   nargs="*", default=None,
                   help="(RoboCerebra trainset) scene sub-folders to include, e.g. "
                        "coffee_table kitchen_table")
    p.add_argument("--subsample", type=int, default=None,
                   help="Temporal subsample (overrides config)")
    p.add_argument("--max_traj",  type=int, default=None,
                   help="Limit number of trajectories evaluated (for quick testing)")
    p.add_argument("--max_seqlen", type=int, default=None,
                   help="Truncate trajectories to this many steps (overrides config)")

    # Runtime
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of trajectories per evaluation batch "
                        "(set >1 only if GPU memory allows)")

    # Boundary detection tolerance (subtask-boundary F1 / SR)
    p.add_argument("--boundary_tol", type=int, default=5,
                   help="Timestep window for branch-vs-GT-boundary matching "
                        "(RoboCerebra only)")
    p.add_argument("--theta_fuse", type=float, default=None,
                   help="Override the memory-tree merge threshold at eval time. "
                        "Lower values (e.g. 0.05–0.15) cause more branching. "
                        "Default: use value from config (typically 0.35).")

    # Output
    p.add_argument("--out", default=None,
                   help="JSON file to write results (optional)")
    p.add_argument("--print_tree", action="store_true",
                   help="Print the memory-tree ASCII structure after each "
                        "trajectory (useful for verifying semantic elevation)")

    return p.parse_args()


# ================================================================
#  Config & model loading
# ================================================================

import sys
import os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent))

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    """
    Load DualTreeVLA from checkpoint.
    Handles plain .pt (state_dict or full training ckpt) and
    DeepSpeed tag-file directory checkpoints.
    """
    from dual_tree_vla.model import DualTreeVLA

    m_cfg = cfg.get("model", {})
    model = DualTreeVLA(
        llm_path   = m_cfg.get("llm_path", "checkpoints/Qwen2.5-1.5B-Instruct"),
        d          = m_cfg.get("d", 256),
        d_a        = m_cfg.get("d_a", 7),
        d_q        = m_cfg.get("d_q", 84),
        d_visual   = m_cfg.get("d_visual", 256),
        d_ssm      = m_cfg.get("d_ssm", 256),
        d_state    = m_cfg.get("d_state", 16),
        patch_size = m_cfg.get("patch_size", 16),
        H_a        = m_cfg.get("H_a", 16),
        n_ode      = m_cfg.get("n_ode", 20),
        theta_fuse = m_cfg.get("theta_fuse", 0.35),
        K_elev     = m_cfg.get("K_elev", 4),
        delta_w    = m_cfg.get("delta_w", 0.1),
        tau        = m_cfg.get("tau", 0.1),
        freeze_llm = False,   # load full weights regardless
    )

    ckpt_path = str(ckpt_path)

    # DeepSpeed checkpoint directory
    if os.path.isdir(ckpt_path):
        # Try DeepSpeed's official consolidation API first.
        # This correctly handles both ZeRO-2 and ZeRO-3 (sharded params).
        try:
            from deepspeed.utils.zero_to_fp32 import (  # type: ignore[import]
                get_fp32_state_dict_from_zero_checkpoint,
            )
            print(f"[INFO] Consolidating DeepSpeed ZeRO checkpoint from {ckpt_path} …")
            sd = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        except Exception as ds_err:
            # Fallback: ZeRO-2 stores full params in rank-0 shard
            import glob
            print(f"[WARN] DeepSpeed consolidation failed ({ds_err}); "
                  "falling back to rank-0 shard loading (ZeRO-2 only).")
            shard_files = sorted(glob.glob(
                os.path.join(ckpt_path, "**", "*model_states.pt"), recursive=True
            ))
            if not shard_files:
                raise FileNotFoundError(
                    f"No model_states.pt found in DeepSpeed checkpoint dir: {ckpt_path}"
                )
            state = torch.load(shard_files[0], map_location="cpu")
            sd = state.get("module", state)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        # Training checkpoint wraps state_dict under 'model_state_dict'
        sd = (
            state.get("model_state_dict")
            or state.get("module")
            or state
        )

    # ZeRO-3 stores non-local shard params as shape-[0] placeholders.
    # Trying to load them into the model raises a "size mismatch" RuntimeError
    # even with strict=False.  Drop any key whose shape doesn't match the
    # current model so those parameters keep their initialised values.
    model_sd = model.state_dict()
    shape_skipped = []
    sd_clean: dict = {}
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            shape_skipped.append(f"{k}: ckpt{list(v.shape)} vs model{list(model_sd[k].shape)}")
        else:
            sd_clean[k] = v
    if shape_skipped:
        print(f"[WARN] Skipping {len(shape_skipped)} shape-mismatched keys "
              f"(ZeRO-3 placeholders): {shape_skipped[:5]}"
              f"{'...' if len(shape_skipped) > 5 else ''}")

    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device)
    model.eval()

    # ── sem_proj health check ────────────────────────────────────────
    sem_norms = [p.data.norm().item() for p in model.sem_proj.parameters()]
    if sem_norms:
        mean_n = sum(sem_norms) / len(sem_norms)
        print(f"  [DIAG] sem_proj weight norms: mean={mean_n:.3f}")
        if mean_n < 0.1:
            print("  [WARN] sem_proj weights are near-zero — L_sem alignment may collapse.")

    return model


# ================================================================
#  Metrics helpers
# ================================================================

def tree_max_depth(tree) -> int:
    """Maximum node depth across all nodes."""
    if tree.root_id is None:
        return 0
    return max(tree.depth(nid) for nid in tree.nodes)


def tree_elevation_events(tree) -> int:
    """Count nodes that are 'abstract' (inserted by elevation).
    These are identified as non-leaf nodes whose children were re-parented
    — approximated here by counting nodes with node_id gaps, or by the
    elevation_counter attribute if tracked externally.
    Falls back to the tree's own counter if available."""
    return getattr(tree, "_elevation_count", 0)


def compute_prog_monotone_rate(tree) -> float:
    """
    Fraction of (ancestor, descendant) pairs where ancestor is farther from
    the mean leaf-node semantic embedding (= task goal proxy).
    Section 3.6: L_prog corresponds to dist(ancestor, goal) > dist(descendant, goal).
    """
    if tree.root_id is None or len(tree.nodes) < 2:
        return 1.0

    # Collect leaf z_v as task-goal proxy (visual embedding of terminal states)
    leaf_s = [
        tree.nodes[nid].z_v.float()
        for nid in tree.nodes
        if tree.nodes[nid].is_leaf() and tree.nodes[nid].z_v is not None
    ]
    if not leaf_s:
        return 1.0
    goal = F.normalize(torch.stack(leaf_s).mean(0), dim=-1)   # (d,)

    pairs = tree.ancestor_descendant_pairs()
    if not pairs:
        return 1.0

    correct = 0
    for anc_id, desc_id in pairs:
        if anc_id not in tree.nodes or desc_id not in tree.nodes:
            continue
        node_a = tree.nodes[anc_id]
        node_d = tree.nodes[desc_id]
        # Use z_v for leaves, s for abstract nodes (project to same space)
        vec_a = node_a.z_v.float() if node_a.is_leaf() else (
            node_a.s.float() if node_a.s is not None else None)
        vec_d = node_d.z_v.float() if node_d.is_leaf() else (
            node_d.s.float() if node_d.s is not None else None)
        if vec_a is None or vec_d is None:
            continue
        # Truncate/pad to goal dimension
        d_goal = goal.shape[0]
        if vec_a.shape[0] != d_goal:
            vec_a = F.pad(vec_a[:d_goal], (0, max(0, d_goal - vec_a.shape[0])))
        if vec_d.shape[0] != d_goal:
            vec_d = F.pad(vec_d[:d_goal], (0, max(0, d_goal - vec_d.shape[0])))
        s_a = F.normalize(vec_a, dim=-1)
        s_d = F.normalize(vec_d, dim=-1)
        dist_a = 1.0 - (s_a * goal).sum().item()
        dist_d = 1.0 - (s_d * goal).sum().item()
        if dist_a >= dist_d - 1e-4:
            correct += 1

    return correct / len(pairs)


def compute_boundary_matches(
    branch_steps: List[int],
    gt_boundaries: List[int],
    tol: int,
) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN for boundary detection with ±tol tolerance.

    Returns (tp, fp, fn).
    """
    matched_gt  = set()
    matched_pred = set()

    for bi, gt in enumerate(gt_boundaries):
        for pi, pr in enumerate(branch_steps):
            if abs(pr - gt) <= tol and bi not in matched_gt and pi not in matched_pred:
                matched_gt.add(bi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(branch_steps) - len(matched_pred)
    fn = len(gt_boundaries) - tp
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    denom     = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


# ================================================================
#  Tree visualisation
# ================================================================

def format_tree(
    tree,
    branch_steps: List[int],
    header: str = "",
) -> str:
    """
    Render the HierarchicalMemoryTree as an ASCII tree.

    Each node line shows:
        <prefix><marker> [id:<nid>] n=<merge_cnt> w=<weight:.2f>
                         |s|=<||s||:.3f>  elev=<yes/no>

    Elevation nodes are identified as non-leaf nodes that were NOT in the
    original BFS sequence before their children (i.e. abstract parent added
    by semantic_elevation).  We detect them by checking whether the node's
    children count > 0 AND the node has a parent (elevated nodes are always
    interior, non-root).

    branch_steps  : timesteps at which a branch was created (from evaluate_trajectory)
    """
    if tree.root_id is None:
        return "<empty tree>"

    lines: List[str] = []
    if header:
        lines.append(header)

    n_nodes     = tree.size()
    n_elevations = getattr(tree, "_elevation_count", 0)
    max_depth   = tree_max_depth(tree)
    lines.append(
        f"  nodes={n_nodes}  depth={max_depth}  "
        f"branches={len(branch_steps)}  elevations={n_elevations}"
    )
    lines.append("")

    def _render(nid: int, prefix: str, is_last: bool) -> None:
        node = tree.nodes[nid]

        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Identify elevated nodes: interior (has children) and non-root
        is_elev = (
            not node.is_leaf()
            and not node.is_root()
            and node.parent_id is not None
        )
        # Identify active node
        is_active = (nid == tree.active_id)

        # Merge count / visit count: leaf nodes track a_hist length; abstract track children count
        if node.is_leaf():
            n_merge = len(node.a_hist) if node.a_hist else 0
        else:
            n_merge = len(node.children_ids)
        # Semantic norm: leaf nodes have z_v (visual), abstract have s (semantic)
        if node.is_leaf():
            s_norm = node.z_v.float().norm().item() if node.z_v is not None else 0.0
        else:
            s_norm = node.s.float().norm().item() if node.s is not None else 0.0
        # Importance weight
        w = node.w
        # Children count
        n_ch = len(node.children_ids)

        tags: List[str] = []
        if node.is_root():
            tags.append("ROOT")
        if is_elev:
            tags.append("ELEV")
        if is_active:
            tags.append("ACTIVE")
        if node.is_leaf():
            tags.append("leaf")
        tag_str = " [" + ",".join(tags) + "]" if tags else ""

        lines.append(
            f"{prefix}{connector}[{nid:>3}]{tag_str}"
            f"  n={n_merge:>3}  w={w:.2f}  |s|={s_norm:.3f}  children={n_ch}"
        )

        children = node.children_ids
        for i, cid in enumerate(children):
            _render(cid, child_prefix, is_last=(i == len(children) - 1))

    _render(tree.root_id, prefix="", is_last=True)

    # Branch timeline
    if branch_steps:
        lines.append("")
        lines.append(f"  Branch timesteps: {branch_steps}")

    return "\n".join(lines)


# ================================================================
#  Single-trajectory evaluation
# ================================================================

@torch.no_grad()
def evaluate_trajectory(
    model,
    frames: torch.Tensor,     # (T, 3, H, W)
    actions: torch.Tensor,    # (T, d_a)  — ground-truth actions
    states: torch.Tensor,     # (T, d_q)
    instruction: str,
    device: torch.device,
    has_subtask_labels: bool = False,
    subtask_ids: Optional[torch.Tensor] = None,   # (T,) int
    boundary_tol: int = 5,
    print_tree: bool = False,
    traj_label: str = "",
) -> Dict:
    """
    Run one trajectory through model.step(), collecting per-step metrics.
    Returns a dict of scalar metric values for this trajectory.
    """
    T, d_a = actions.shape
    if T == 0:
        return {}

    # Pad proprioception states to the model's expected d_q if necessary
    # (e.g. RoboCerebraBench states are dim-71 while training used dim-84)
    try:
        model_dq: int = model.fusion.prop_proj.weight.shape[1]
    except AttributeError:
        model_dq = states.shape[-1]
    if states.shape[-1] < model_dq:
        pad = torch.zeros(T, model_dq - states.shape[-1], dtype=states.dtype)
        states = torch.cat([states, pad], dim=-1)
    elif states.shape[-1] > model_dq:
        states = states[:, :model_dq]

    model.reset_trees(batch_size=1)
    tree = model.get_tree(0)
    # Attach elevation counter (incremented here, not inside step())
    tree._elevation_count = 0
    # Monkey-patch insert() to record each timestep's cosine distance d_t.
    # This is the single number that decides merge vs. branch — logging it
    # lets us check whether theta_fuse needs tuning.
    tree._dt_log = []
    _orig_insert = tree.insert.__func__   # unbound method
    def _patched_insert(self, z_v, a, force_branch: bool):
        # Snapshot d_t (cosine distance) before calling original insert
        active_node = self.nodes.get(self.active_id)
        if active_node is not None and not active_node.is_leaf() and active_node.s is not None:
            import torch.nn.functional as _F
            import torch as _torch
            # Use z_v distance as a proxy for semantic change
            s_before = active_node.s.float()
            d_cosine = (1.0 - _F.cosine_similarity(
                z_v.float().unsqueeze(0),
                s_before.unsqueeze(0),
            ).item())
            self._dt_log.append(d_cosine)
        return _orig_insert(self, z_v, a, force_branch)
    import types
    tree.insert = types.MethodType(_patched_insert, tree)

    # Helper to snapshot active-node state before each insert (kept for compat)
    def _snapshot_s(t):
        pass  # no-op: new node design doesn't store s on leaves

    l1_errors, l2_errors = [], []
    branch_steps: List[int] = []   # timesteps where a real branch was created
    merge_count: int = 0           # timesteps that triggered a merge (no new node)
    dt_values: List[float] = []    # cosine distances between consecutive s embeddings
    a_prev = None

    for t in range(T):
        img_t = frames[t].unsqueeze(0).to(device)     # (1, 3, H, W)
        q_t   = states[t].unsqueeze(0).to(device)     # (1, d_q)

        # Snapshot active-node s before model.step() calls insert() internally
        _snapshot_s(t)

        # Snapshot tree size before step
        size_before = tree.size()

        # Predict action chunk (also updates the memory tree internally)
        a_chunk = model.step(img_t, instruction, q_t, a_prev)   # (1, H_a, d_a)

        # ── Track tree structural changes ──────────────────────────
        size_after = tree.size()
        delta = size_after - size_before

        # Recover the d_t that insert() just computed from the patched list.
        if hasattr(tree, "_dt_log") and tree._dt_log:
            dt_values.append(tree._dt_log[-1])

        if size_before == 0:
            # Root creation — not a real branch event, skip.
            pass
        elif delta == 0:
            # Merge: active node updated in-place, no new node.
            merge_count += 1
        else:
            # delta >= 1: a new leaf was created (branch)
            branch_steps.append(t)
            if delta >= 2:
                # semantic_elevation ran and added an abstract parent node too.
                # model.step() clears elevation_pending_parent after handling,
                # so we detect it purely via the extra +1 size jump.
                tree._elevation_count += 1

        # Action L1 / L2  (compare first predicted step vs GT action)
        a_pred_first = a_chunk[0, 0].cpu()    # (d_a,)
        a_gt_t       = actions[t]              # (d_a,)
        l1_errors.append((a_pred_first - a_gt_t).abs().mean().item())
        l2_errors.append((a_pred_first - a_gt_t).norm().item())

        # Teacher-force: feed GT action as a_prev
        a_prev = actions[t].unsqueeze(0).to(device)

    # --- Tree stats -------------------------------------------------------
    # Summarise d_t distribution for diagnosing theta_fuse tuning needs.
    dt_summary: Dict[str, float] = {}
    if dt_values:
        dt_sorted = sorted(dt_values)
        n_dt = len(dt_sorted)
        dt_summary = {
            "dt_mean":   sum(dt_values) / n_dt,
            "dt_max":    dt_sorted[-1],
            "dt_p90":    dt_sorted[int(0.9 * n_dt)],
            "dt_p50":    dt_sorted[n_dt // 2],
        }

    # s_embed_spread: mean pairwise cosine distance between s vectors sampled
    # across the trajectory.  Near 0 = collapse; healthy = 0.1 ~ 1.0.
    s_embed_spread = 0.0
    s_log = getattr(tree, "_s_log", [])
    if len(s_log) > 1:
        n_sample = min(len(s_log), 64)
        step = max(1, len(s_log) // n_sample)
        s_mat = torch.stack(s_log[::step][:n_sample])  # (N, d)
        s_mat_n = F.normalize(s_mat.float(), dim=-1)
        sim_mat = s_mat_n @ s_mat_n.T                  # (N, N)
        N = sim_mat.shape[0]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        s_embed_spread = float((1.0 - sim_mat[mask]).mean().item())

    result: Dict = {
        "action_l1":        float(sum(l1_errors) / max(len(l1_errors), 1)),
        "action_l2":        float(sum(l2_errors) / max(len(l2_errors), 1)),
        "tree_nodes":       float(tree.size()),
        "tree_depth":       float(tree_max_depth(tree)),
        "tree_branches":    float(len(branch_steps)),
        "tree_elevations":  float(tree._elevation_count),
        "merge_rate":       float(merge_count / max(T - 1, 1)),
        "s_embed_spread":   s_embed_spread,
        **dt_summary,
    }

    # --- Subtask-specific metrics (RoboCerebra) ---------------------------
    if has_subtask_labels and subtask_ids is not None:
        # GT subtask boundaries: timesteps where subtask label changes
        gt_bounds: List[int] = []
        for t in range(1, T):
            if subtask_ids[t].item() != subtask_ids[t - 1].item():
                gt_bounds.append(t)

        # Boundary F1
        tp, fp, fn = compute_boundary_matches(branch_steps, gt_bounds, tol=boundary_tol)
        result["subtask_boundary_f1"] = f1_from_counts(tp, fp, fn)

        # Subtask success rate: fraction of GT boundaries with ≥1 matched branch
        sr = tp / max(len(gt_bounds), 1)
        result["subtask_sr"] = float(sr)

        # Progress monotonicity
        result["prog_monotone_rate"] = compute_prog_monotone_rate(tree)

    # ── Optional tree dump ────────────────────────────────────────────
    if print_tree:
        label = traj_label or instruction[:60]
        header = f"\n{'─' * 64}\n  Tree after: {label}"
        print(format_tree(tree, branch_steps, header=header))
        print(f"{'─' * 64}")

    return result


# ================================================================
#  Per-trajectory runner (shared by all dataset modes)
# ================================================================

def _eval_trajectories(
    model,
    ds,
    n_traj: int,
    device: torch.device,
    has_subtask_labels: bool,
    boundary_tol: int,
    show_task_type: bool = False,
    print_tree: bool = False,
) -> List[Dict]:
    """Iterate over dataset, run evaluate_trajectory, return per-traj results."""
    all_results: List[Dict] = []
    t0 = time.time()

    for idx in range(n_traj):
        sample = ds[idx]

        frames      = sample["frames"]           # (T, 3, H, W)
        actions     = sample["actions"]          # (T, 7)
        states      = sample["states"]           # (T, d_q)
        instruction = sample["instruction"]
        subtask_ids = sample.get("subtask_ids")  # (T,) or None

        if show_task_type:
            traj_label = (
                f"{sample.get('task_type', '')} / "
                f"{sample.get('case_name', '')}  "
                f"{instruction[:40]}"
            )
        else:
            traj_label = f"traj {idx}  {instruction[:55]}"
        traj_result = evaluate_trajectory(
            model              = model,
            frames             = frames,
            actions            = actions,
            states             = states,
            instruction        = instruction,
            device             = device,
            has_subtask_labels = has_subtask_labels,
            subtask_ids        = subtask_ids,
            boundary_tol       = boundary_tol,
            print_tree         = print_tree,
            traj_label         = traj_label,
        )
        traj_result["trajectory_idx"] = idx
        traj_result["instruction"]    = instruction
        if show_task_type:
            traj_result["task_type"] = sample.get("task_type", "")
            traj_result["case_name"] = sample.get("case_name", "")
        all_results.append(traj_result)

        if (idx + 1) % 10 == 0 or idx == n_traj - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / max(elapsed, 1e-6)
            tt_tag = (
                f"[{traj_result.get('task_type', '')}] "
                if show_task_type else ""
            )
            print(
                f"  [{idx+1:>4}/{n_traj}] {tt_tag}"
                f"L1={traj_result.get('action_l1', 0):.4f}  "
                f"L2={traj_result.get('action_l2', 0):.4f}  "
                f"nodes={traj_result.get('tree_nodes', 0):.1f}  "
                f"depth={traj_result.get('tree_depth', 0):.1f}  "
                f"({fps:.2f} traj/s)"
            )

    return all_results


def _print_summary_table(title: str, summary: Dict[str, float]) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'-' * w}")
    print(f"  {'Metric':<33} {'Value':>10}")
    print(f"{'-' * w}")
    # Show mean metrics first (skip *_std lines)
    for k, v in summary.items():
        if not k.endswith("_std"):
            std = summary.get(f"{k}_std", None)
            std_str = f"  ±{std:.4f}" if std is not None else ""
            print(f"  {k:<33} {v:>10.4f}{std_str}")
    print(f"{'=' * w}")

def _print_semantic_diagnosis(summary: Dict[str, float]) -> None:
    """Print a clear diagnosis when semantic embedding collapse is detected."""
    dt_mean      = summary.get("dt_mean",       1.0)
    dt_max       = summary.get("dt_max",        1.0)
    s_spread     = summary.get("s_embed_spread", 1.0)
    merge_rate   = summary.get("merge_rate",     0.0)
    theta_fuse   = 0.35   # informational only

    if dt_max >= 0.05:
        return   # looks healthy, no diagnosis needed

    print("\n" + "!" * 64)
    print("  SEMANTIC EMBEDDING COLLAPSE DETECTED")
    print("!" * 64)
    print(f"  dt_mean={dt_mean:.5f}  dt_max={dt_max:.5f}  "
          f"s_embed_spread={s_spread:.4f}")
    print(f"  theta_fuse={theta_fuse}  →  tree branches NEVER (merge_rate={merge_rate:.3f})")
    print()
    print("  The model's 's' embeddings are nearly identical across all frames.")
    print("  Likely causes (check in order):")
    print("  1. Phase 1 L_sem loss was zero / too small → s_proj never learned")
    print("     contrast.  Check training logs for 'L_sem' column.")
    print("  2. Checkpoint has MISSING s_proj weights (loaded as random init).")
    print("     Check '[WARN] Missing keys' above — if 's_proj' appears there,")
    print("     the Phase 3 ZeRO-3 checkpoint did not save it.")
    print("  3. Phase 3 LLM joint fine-tuning destroyed semantic alignment.")
    print("     Try an earlier checkpoint to compare:")
    print("       --ckpt checkpoints/runs/phase1_best")
    print("       --ckpt checkpoints/runs/phase2_best")
    print("  4. s_proj weight norms are near-zero (see [DIAG] lines above).")
    print("!" * 64)

# ================================================================
#  Main evaluation loop
# ================================================================

def run_evaluation(args: argparse.Namespace):
    cfg    = load_config(args.config)
    device = torch.device(args.device)

    # Override config with CLI args
    data_cfg = cfg.get("data", {})
    if args.subsample  is not None: data_cfg["subsample"]  = args.subsample
    if args.max_seqlen is not None: data_cfg["max_seqlen"] = args.max_seqlen

    subsample  = data_cfg.get("subsample", 4)
    max_seqlen = data_cfg.get("max_seqlen", 256)

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt, cfg, device)

    # ── Override tree theta_fuse if requested ────────────────────────
    if args.theta_fuse is not None:
        old_theta = model._tree_cfg.get("theta_fuse", "?")
        model._tree_cfg["theta_fuse"] = args.theta_fuse
        print(f"[INFO] theta_fuse overridden: {old_theta} → {args.theta_fuse}")

    # ── Dispatch per dataset mode ────────────────────────────────────
    if args.dataset == "robocerebra_bench":
        return _run_bench_evaluation(args, cfg, model, device,
                                     data_cfg, subsample, max_seqlen)

    # ── Build dataset (robocerebra trainset or libero) ────────────────
    print(f"Building dataset ({args.dataset}) ...")
    has_subtask_labels = (args.dataset == "robocerebra")

    if args.dataset == "robocerebra":
        if args.data_root is None:
            raise ValueError("--data_root is required for dataset=robocerebra")
        from dual_tree_vla.dataset import RoboCerebraDataset
        ds = RoboCerebraDataset(
            root       = args.data_root,
            scenes     = args.scenes,
            img_h      = data_cfg.get("img_h", 224),
            img_w      = data_cfg.get("img_w", 224),
            subsample  = subsample,
            max_seqlen = max_seqlen,
        )
    else:   # libero
        if args.data_root is None:
            raise ValueError("--data_root is required for dataset=libero")
        from dual_tree_vla.dataset.libero import LiberoDataset
        ds = LiberoDataset(
            root       = args.data_root,
            img_h      = data_cfg.get("img_h", 224),
            img_w      = data_cfg.get("img_w", 224),
            max_seqlen = max_seqlen,
            d_q        = cfg.get("model", {}).get("d_q", 84),
        )

    n_traj = len(ds)
    if args.max_traj is not None:
        n_traj = min(n_traj, args.max_traj)
    print(f"Evaluating {n_traj} trajectories ...")

    all_results = _eval_trajectories(
        model, ds, n_traj, device,
        has_subtask_labels=has_subtask_labels,
        boundary_tol=args.boundary_tol,
        print_tree=args.print_tree,
    )

    summary = aggregate_results(all_results, has_subtask_labels)
    _print_summary_table("RESULTS", summary)
    _print_semantic_diagnosis(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config":         args.config,
            "checkpoint":     args.ckpt,
            "dataset":        args.dataset,
            "libero_split":   args.libero_split if args.dataset == "libero" else None,
            "n_trajectories": n_traj,
            "summary":        summary,
            "per_trajectory": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return summary


# ================================================================
#  RoboCerebraBench structured evaluation
# ================================================================

def _run_bench_evaluation(
    args: argparse.Namespace,
    cfg: dict,
    model,
    device: torch.device,
    data_cfg: dict,
    subsample: int,
    max_seqlen: int,
) -> Dict:
    """
    Evaluate on RoboCerebraBench with per-task-type breakdown.

    Follows the task-type categorisation in the original RoboCerebra eval
    (eval_openvla.py):
      Ideal                  — standard long-horizon tasks
      Memory_Execution       — tests memory-guided execution
      Memory_Exploration     — tests memory-guided exploration
      Mix                    — dynamic disturbance + observation shift
      Observation_Mismatching— shifted observation description
      Random_Disturbance     — random object disturbances during execution

    Since DualTreeVLA is evaluated offline (no simulation), we measure:
      action_l1 / action_l2          — action-prediction quality
      tree_nodes / tree_depth /
      tree_branches / tree_elevations — memory-tree structure
      subtask_boundary_f1 / subtask_sr
      prog_monotone_rate              — subtask-aware metrics
    """
    from dual_tree_vla.dataset.robocerebra_bench import (
        RoboCerebraBenchDataset,
        BENCH_TASK_TYPES,
    )

    bench_root = args.bench_root
    task_types = args.task_types  # None → all six

    print(f"Building RoboCerebraBench dataset ...")
    print(f"  Root      : {bench_root}")
    print(f"  Task types: {task_types or BENCH_TASK_TYPES}")

    full_ds = RoboCerebraBenchDataset(
        root       = bench_root,
        task_types = task_types,
        img_h      = data_cfg.get("img_h", 224),
        img_w      = data_cfg.get("img_w", 224),
        subsample  = subsample,
        max_seqlen = max_seqlen,
    )

    n_total = len(full_ds)
    if args.max_traj is not None:
        n_total = min(n_total, args.max_traj)

    if n_total == 0:
        print("[WARN] No cases found. Check --bench_root and --task_types.")
        return {}

    print(f"Evaluating {n_total} cases ...")

    # ── Per-trajectory evaluation ─────────────────────────────────────
    all_results = _eval_trajectories(
        model, full_ds, n_total, device,
        has_subtask_labels=True,   # bench always has step annotations
        boundary_tol=args.boundary_tol,
        show_task_type=True,        print_tree=args.print_tree,    )

    # ── Per-task-type aggregation ─────────────────────────────────────
    effective_task_types = task_types or BENCH_TASK_TYPES
    per_type_results: Dict[str, Dict] = {}

    print(f"\n{'=' * 70}")
    print("  PER TASK-TYPE RESULTS  (RoboCerebraBench)")
    print(f"{'=' * 70}")

    for tt in effective_task_types:
        tt_rows = [r for r in all_results if r.get("task_type") == tt]
        if not tt_rows:
            continue
        tt_summary = aggregate_results(tt_rows, has_subtask_labels=True)
        per_type_results[tt] = {
            "n_cases": len(tt_rows),
            "metrics": tt_summary,
        }
        # Print compact row
        print(
            f"  {tt:<30} n={len(tt_rows):>3}  "
            f"L1={tt_summary.get('action_l1', 0):.4f}  "
            f"L2={tt_summary.get('action_l2', 0):.4f}  "
            f"F1={tt_summary.get('subtask_boundary_f1', 0):.4f}  "
            f"SR={tt_summary.get('subtask_sr', 0):.4f}  "
            f"mono={tt_summary.get('prog_monotone_rate', 0):.4f}"
        )

    # ── Overall summary ───────────────────────────────────────────────
    overall = aggregate_results(all_results, has_subtask_labels=True)
    _print_summary_table("OVERALL SUMMARY", overall)
    _print_semantic_diagnosis(overall)

    # ── Save ──────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config":          args.config,
            "checkpoint":      args.ckpt,
            "dataset":         "robocerebra_bench",
            "bench_root":      bench_root,
            "task_types":      task_types or BENCH_TASK_TYPES,
            "n_cases":         n_total,
            "boundary_tol":    args.boundary_tol,
            "overall_summary": overall,
            "per_task_type":   per_type_results,
            "per_case":        all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return overall


# ================================================================
#  Result aggregation
# ================================================================

def aggregate_results(
    results: List[Dict],
    has_subtask_labels: bool,
) -> Dict[str, float]:
    """Compute mean ± std for all numeric metrics across trajectories."""
    if not results:
        return {}

    # Collect numeric keys (exclude string & index fields)
    ignore = {"trajectory_idx", "instruction"}
    numeric_keys = [
        k for k in results[0]
        if k not in ignore and isinstance(results[0][k], (int, float))
    ]

    summary: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [r[k] for r in results if k in r and not math.isnan(r[k])]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        summary[k]           = mean
        summary[f"{k}_std"]  = std

    return summary


# ================================================================
#  Entry point
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
