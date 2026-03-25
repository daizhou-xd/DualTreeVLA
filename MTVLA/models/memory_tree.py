"""
MemoryTree: Hierarchical task-tree memory for MemoryTreeVLA.

Implements the Tree.json data structure defined in CONSTRUCTION.md §8.
Each node represents a sub-task with explicit status tracking, stored action
token sequences, completion evidence, backtrack pointers and failure counters.

State transition rules (CONSTRUCTION.md §8):
  pending       → in_progress  : parent activated & all prior siblings completed
  in_progress   → completed    : Tree LLM confidence > threshold
  in_progress   → failed       : failure_count exceeds max_retries
  failed        → in_progress  : backtrack pointer activated (reset & retry)

The tree serialises to / deserialises from the Tree.json dict that the Tree LLM
(Qwen2.5-1.5B-Instruct) reads and writes each episode step.

Key public methods:
  MemoryTree.from_dict(d)          – build from a Tree.json dict
  tree.to_dict()                   – serialise to Tree.json
  tree.get_active_node()           – currently executing primitive leaf
  tree.advance(node, evidence)     – mark node completed with visual evidence
  tree.fail(node)                  – increment failure count / trigger fail
  tree.backtrack(node)             – follow backtrack_pointer and reset subtree
  tree.to_parent_map()             – {node_id: parent_id} for TaskTreeMamba
  tree.node_list()                 – flat list in BFS order (for embedding lookup)
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Node status literals
# ---------------------------------------------------------------------------

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"

# Node types (matching CONSTRUCTION.md §8)
SEQUENCE = "sequence"   # composite: children executed in order
PRIMITIVE = "primitive"  # leaf: directly mapped to action sequence


# ---------------------------------------------------------------------------
# MemoryNode dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    """A single node in the hierarchical task tree (Tree.json spec).

    Attributes:
        node_id          : unique string identifier (e.g. ``"subtask_1"``).
        description      : natural-language sub-task description.
        node_type        : ``"primitive"`` (leaf) or ``"sequence"`` (composite).
        status           : one of ``pending | in_progress | completed | failed``.
        step_number      : 1-based step index from the RoboCerebra annotation
                           (``0`` for the root / non-primitive nodes).
        timestep         : ``{"start": int, "end": int}`` ground-truth frame range
                           from the RoboCerebra dataset annotation; used as
                           supervision signal for the Tree LLM completion detector.
        related_objects  : list of object names involved in this sub-task
                           (from ``task_description.json`` ``related_objects`` field).
        token_sequence   : stored action token ids from the Action LLM when
                           this sub-task was last executed; ``None`` if not yet
                           executed.
        completion_evidence : dict with ``frame_idx`` and ``confidence`` set by
                           Tree LLM when it detects task completion.
        failure_count    : number of consecutive execution failures.
        backtrack_pointer: ``node_id`` of the node to revert to on failure;
                           ``None`` means no backtrack target.
        children         : ordered list of child ``MemoryNode``s.
        parent           : reference to parent node (``None`` for root).
    """

    node_id: str
    description: str
    node_type: str = PRIMITIVE
    status: str = PENDING
    step_number: int = 0
    timestep: Optional[Dict[str, int]] = None          # {"start": int, "end": int}
    related_objects: List[str] = field(default_factory=list)
    token_sequence: Optional[List[int]] = None
    completion_evidence: Optional[Dict[str, Any]] = None
    failure_count: int = 0
    backtrack_pointer: Optional[str] = None
    children: List["MemoryNode"] = field(default_factory=list)
    parent: Optional["MemoryNode"] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ #

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_completed(self) -> bool:
        return self.status == COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == FAILED

    @property
    def is_active(self) -> bool:
        return self.status == IN_PROGRESS

    def add_child(self, child: "MemoryNode") -> None:
        child.parent = self
        self.children.append(child)

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise to Tree.json-compatible dict."""
        d: Dict[str, Any] = {
            "id": self.node_id,
            "description": self.description,
            "type": self.node_type,
            "status": self.status,
            "step_number": self.step_number,
            "timestep": self.timestep,
            "related_objects": self.related_objects,
            "token_sequence": self.token_sequence,
            "completion_evidence": self.completion_evidence,
            "failure_count": self.failure_count,
            "backtrack_pointer": self.backtrack_pointer,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], parent: Optional["MemoryNode"] = None) -> "MemoryNode":
        """Recursively deserialise from a Tree.json node dict."""
        node = cls(
            node_id=d["id"],
            description=d.get("description", ""),
            node_type=d.get("type", PRIMITIVE),
            status=d.get("status", PENDING),
            step_number=d.get("step_number", 0),
            timestep=d.get("timestep"),
            related_objects=d.get("related_objects", []),
            token_sequence=d.get("token_sequence"),
            completion_evidence=d.get("completion_evidence"),
            failure_count=d.get("failure_count", 0),
            backtrack_pointer=d.get("backtrack_pointer"),
            parent=parent,
        )
        for child_dict in d.get("children", []):
            child = cls.from_dict(child_dict, parent=node)
            node.children.append(child)
        return node


# ---------------------------------------------------------------------------
# MemoryTree
# ---------------------------------------------------------------------------

class MemoryTree(nn.Module):
    """Hierarchical task-tree memory for long-horizon robot manipulation.

    Wraps the Tree.json structure and exposes Pythonic state-management APIs
    used by the Tree LLM integration and the training loop.

    Args:
        max_retries : number of failures allowed before a node transitions
                      to ``failed`` and backtracking is triggered.
        completion_threshold : minimum Tree LLM confidence required to
                               accept a completion claim.
    """

    def __init__(
        self,
        max_retries: int = 3,
        completion_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.max_retries = max_retries
        self.completion_threshold = completion_threshold

        self.task_id: str = ""
        self.task_description: str = ""
        self.scene: str = ""          # e.g. "coffee_table", "kitchen_table", "study_table"
        self.case_id: str = ""         # e.g. "case1"
        self.root: Optional[MemoryNode] = None
        self._id_index: Dict[str, MemoryNode] = {}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(
        cls,
        tree_json: Dict[str, Any],
        max_retries: int = 3,
        completion_threshold: float = 0.8,
    ) -> "MemoryTree":
        """Build a MemoryTree from a Tree.json dict (output of Tree LLM).

        Supports both the flat ``"nodes"`` list format (used by TaskTreeMamba)
        and the nested ``"root"`` format (canonical Tree.json).

        Args:
            tree_json : parsed Tree.json dict.
        """
        tree = cls(max_retries=max_retries, completion_threshold=completion_threshold)
        tree.task_id = tree_json.get("task_id", "")
        tree.task_description = tree_json.get("task_description", "")
        tree.scene = tree_json.get("scene", "")
        tree.case_id = tree_json.get("case_id", "")

        if "root" in tree_json:
            tree.root = MemoryNode.from_dict(tree_json["root"])
        elif "nodes" in tree_json:
            # flat list format: build nested tree from parent pointers
            nodes_raw: List[Dict] = tree_json["nodes"]
            id_map: Dict[str, MemoryNode] = {}
            for nd in nodes_raw:
                id_map[nd["id"]] = MemoryNode.from_dict(nd)
            root_node: Optional[MemoryNode] = None
            for nd in nodes_raw:
                par_id = nd.get("parent")
                node = id_map[nd["id"]]
                if par_id is None:
                    root_node = node
                else:
                    id_map[par_id].add_child(node)
            tree.root = root_node
        else:
            raise ValueError("tree_json must contain either 'root' or 'nodes' key.")

        tree._rebuild_index()
        return tree

    @classmethod
    def from_json_string(cls, json_str: str, **kwargs) -> "MemoryTree":
        """Parse a JSON string emitted by the Tree LLM."""
        return cls.from_dict(json.loads(json_str), **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the current tree state to a Tree.json-compatible dict."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "scene": self.scene,
            "case_id": self.case_id,
            "root": self.root.to_dict() if self.root else None,
        }

    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ------------------------------------------------------------------ #
    # Index helpers
    # ------------------------------------------------------------------ #

    def _rebuild_index(self) -> None:
        """Re-build flat ``{node_id: MemoryNode}`` index (call after any tree mutation)."""
        self._id_index = {}
        for node in self._bfs_iter():
            self._id_index[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self._id_index.get(node_id)

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def _bfs_iter(self) -> List[MemoryNode]:
        """Return all nodes in BFS order."""
        if self.root is None:
            return []
        result: List[MemoryNode] = []
        queue: deque[MemoryNode] = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)
        return result

    def node_list(self) -> List[MemoryNode]:
        """All nodes in BFS order (used for embedding lookup in TaskTreeMamba)."""
        return self._bfs_iter()

    def get_active_node(self) -> Optional[MemoryNode]:
        """Return the deepest currently ``in_progress`` primitive leaf node.

        Traversal priority: left-to-right DFS so that sequential children are
        activated in order.
        """
        return self._find_active(self.root)

    def _find_active(self, node: Optional[MemoryNode]) -> Optional[MemoryNode]:
        if node is None or node.status in (COMPLETED, FAILED):
            return None
        # For a sequence node, activate children in order
        if node.children:
            for child in node.children:
                result = self._find_active(child)
                if result is not None:
                    return result
            return None
        # Leaf node
        return node if node.status == IN_PROGRESS else None

    # ------------------------------------------------------------------ #
    # State transition API
    # ------------------------------------------------------------------ #

    def activate(self, node_id: str) -> bool:
        """Transition a node from ``pending`` → ``in_progress``.

        Returns ``True`` if the transition succeeded.
        """
        node = self.get_node(node_id)
        if node is None or node.status != PENDING:
            return False
        node.status = IN_PROGRESS
        return True

    def advance(
        self,
        node_id: str,
        frame_idx: int,
        confidence: float,
        token_sequence: Optional[List[int]] = None,
    ) -> bool:
        """Mark a node as completed with visual evidence from Tree LLM.

        Transitions ``in_progress → completed`` when confidence ≥ threshold.
        Also stores the action token sequence for potential backtrack replay.

        Args:
            node_id       : target node.
            frame_idx     : video frame where completion was detected.
            confidence    : Tree LLM completion confidence score.
            token_sequence: Action LLM token ids to store for replay.

        Returns:
            ``True`` if the node was successfully completed.
        """
        if confidence < self.completion_threshold:
            return False
        node = self.get_node(node_id)
        if node is None or node.status != IN_PROGRESS:
            return False

        node.status = COMPLETED
        node.completion_evidence = {"frame_idx": frame_idx, "confidence": confidence}
        if token_sequence is not None:
            node.token_sequence = token_sequence

        # Propagate completion upward: mark sequence parent as completed
        # when ALL its children are completed.
        self._propagate_completion(node.parent)
        return True

    def _propagate_completion(self, node: Optional[MemoryNode]) -> None:
        if node is None:
            return
        if node.node_type == SEQUENCE and all(c.is_completed for c in node.children):
            node.status = COMPLETED
            self._propagate_completion(node.parent)

    def fail(self, node_id: str) -> bool:
        """Increment failure counter and optionally transition to ``failed``.

        Returns ``True`` if the node transitioned to ``failed`` (max retries
        exceeded), ``False`` if it is still ``in_progress`` (retrying).
        """
        node = self.get_node(node_id)
        if node is None or node.status != IN_PROGRESS:
            return False
        node.failure_count += 1
        if node.failure_count >= self.max_retries:
            node.status = FAILED
            return True
        return False

    def backtrack(self, node_id: str) -> Optional[str]:
        """Follow the node's backtrack_pointer and reset the subtree.

        Resets the backtrack target node (and its entire subtree) to
        ``pending`` / ``in_progress`` so execution can restart from there.

        Args:
            node_id: the ``failed`` node whose ``backtrack_pointer`` to follow.

        Returns:
            The ``node_id`` of the reset target node, or ``None`` if no
            backtrack pointer is set (caller should propagate failure upward).
        """
        node = self.get_node(node_id)
        if node is None:
            return None
        target_id = node.backtrack_pointer
        if target_id is None:
            return None

        target = self.get_node(target_id)
        if target is None:
            return None

        # Reset the backtrack target and everything below it
        self._reset_subtree(target)
        target.status = IN_PROGRESS
        target.failure_count = 0
        return target_id

    def _reset_subtree(self, node: MemoryNode) -> None:
        node.status = PENDING
        node.token_sequence = None
        node.completion_evidence = None
        node.failure_count = 0
        for child in node.children:
            self._reset_subtree(child)

    # ------------------------------------------------------------------ #
    # Advance to next pending child (called after a sibling completes)
    # ------------------------------------------------------------------ #

    def activate_next(self) -> Optional[str]:
        """Activate the next ``pending`` node in DFS-left order.

        Returns the activated node's ``node_id``, or ``None`` if all nodes
        are completed / failed.
        """
        for node in self._bfs_iter():
            if node.status == PENDING:
                node.status = IN_PROGRESS
                return node.node_id
        return None

    # ------------------------------------------------------------------ #
    # Interfaces for TaskTreeMamba
    # ------------------------------------------------------------------ #

    def to_parent_map(self) -> Dict[str, Optional[str]]:
        """Return ``{node_id: parent_id | None}`` for TaskTreeMamba topology.

        The root node has ``parent_id = None``.
        """
        result: Dict[str, Optional[str]] = {}
        for node in self._bfs_iter():
            result[node.node_id] = node.parent.node_id if node.parent else None
        return result

    def node_descriptions(self) -> List[str]:
        """Ordered list of node descriptions in BFS order (for tokenisation)."""
        return [n.description for n in self._bfs_iter()]

    def node_status_flags(self) -> List[int]:
        """Integer status codes per node (BFS order).

        Encoding: pending=0, in_progress=1, completed=2, failed=3.
        """
        _map = {PENDING: 0, IN_PROGRESS: 1, COMPLETED: 2, FAILED: 3}
        return [_map.get(n.status, 0) for n in self._bfs_iter()]

    # ------------------------------------------------------------------ #
    # RoboCerebra dataset integration
    # ------------------------------------------------------------------ #

    @classmethod
    def from_robocerebra(
        cls,
        annotation: Dict[str, Any],
        scene: str = "",
        case_id: str = "",
        max_retries: int = 3,
        completion_threshold: float = 0.8,
    ) -> "MemoryTree":
        """Build a ``MemoryTree`` directly from a RoboCerebra ``task_description.json``.

        The RoboCerebra annotation schema is::

            {
              "high_level_instruction": "...",
              "steps": [
                {
                  "step_number": 1,
                  "subtask_description": "...",
                  "timestep": {"start": int, "end": int},
                  "related_objects": ["obj1", "obj2"]
                },
                ...
              ]
            }

        Each step becomes a ``primitive`` leaf node.  All steps are collected
        under a single ``sequence`` root node.  The backtrack pointer of each
        step (except step 1) is automatically set to the preceding step, which
        matches the recover-to-previous-subtask convention described in
        CONSTRUCTION.md §5.2.

        Args:
            annotation  : parsed ``task_description.json`` dict.
            scene       : scene name, e.g. ``"coffee_table"``.
            case_id     : case folder name, e.g. ``"case1"``.

        Returns:
            A freshly built ``MemoryTree`` ready for episode execution.
        """
        tree = cls(max_retries=max_retries, completion_threshold=completion_threshold)
        tree.scene = scene
        tree.case_id = case_id
        task_desc: str = annotation.get("high_level_instruction", "")
        tree.task_description = task_desc
        tree.task_id = f"{scene}__{case_id}" if scene and case_id else case_id

        # Root sequence node
        root = MemoryNode(
            node_id="root",
            description=task_desc,
            node_type=SEQUENCE,
            status=PENDING,
        )
        tree.root = root

        steps: List[Dict[str, Any]] = annotation.get("steps", [])
        prev_id: Optional[str] = None
        for step in steps:
            sn: int = step["step_number"]
            node_id = f"subtask_{sn}"
            node = MemoryNode(
                node_id=node_id,
                description=step.get("subtask_description", ""),
                node_type=PRIMITIVE,
                status=PENDING,
                step_number=sn,
                timestep=step.get("timestep"),
                related_objects=step.get("related_objects", []),
                backtrack_pointer=prev_id,   # recover to previous step
            )
            root.add_child(node)
            prev_id = node_id

        # Activate root and first leaf
        root.status = IN_PROGRESS
        if root.children:
            root.children[0].status = IN_PROGRESS

        tree._rebuild_index()
        return tree

    @classmethod
    def from_robocerebra_file(
        cls,
        json_path: str,
        max_retries: int = 3,
        completion_threshold: float = 0.8,
    ) -> "MemoryTree":
        """Load a ``MemoryTree`` from a ``task_description.json`` file path.

        Automatically infers ``scene`` and ``case_id`` from the directory
        structure ``<scene>/<case_id>/task_description.json``.

        Args:
            json_path : absolute or relative path to ``task_description.json``.
        """
        import os
        path = os.path.abspath(json_path)
        case_id = os.path.basename(os.path.dirname(path))
        scene = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, encoding="utf-8") as f:
            annotation = json.load(f)
        return cls.from_robocerebra(
            annotation,
            scene=scene,
            case_id=case_id,
            max_retries=max_retries,
            completion_threshold=completion_threshold,
        )

    def get_step_frame_range(self, node_id: str) -> Optional[Dict[str, int]]:
        """Return the ground-truth frame range ``{start, end}`` for a node.

        Used as supervision labels for the Tree LLM completion detector.
        Returns ``None`` if the node has no timestep annotation.
        """
        node = self.get_node(node_id)
        return node.timestep if node else None

    def get_step_objects(self, node_id: str) -> List[str]:
        """Return the list of related objects for a sub-task node."""
        node = self.get_node(node_id)
        return node.related_objects if node else []

    def reset(self) -> None:
        """Reset the tree to a clean state (call at episode start)."""
        self.task_id = ""
        self.task_description = ""
        self.scene = ""
        self.case_id = ""
        self.root = None
        self._id_index = {}

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._id_index)

    def __repr__(self) -> str:
        n = len(self)
        active = self.get_active_node()
        active_id = active.node_id if active else "—"
        return f"MemoryTree(task='{self.task_id}', nodes={n}, active='{active_id}')"
