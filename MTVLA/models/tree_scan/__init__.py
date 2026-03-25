"""
MTVLA tree_scan sub-package.

Provides the tree-topology State Space Model (SSM) components for
MemoryTreeVLA:

  VisionMamba     – hierarchical image encoder using MST-based tree scanning
                    (adapted from GrootVL, MambaTree NeurIPS 2024).
  TaskTreeMamba   – task-tree hierarchy encoder consuming Tree.json topology
                    from the Tree LLM.
  Tree_SSM        – single Tree-SSM block (shared primitive).

Reference:
  GrootVL: Tree Topology is All You Need in State Space Model
  https://github.com/EasonXiao-888/MambaTree
"""

from .vision_mamba import VisionMamba, VisionMambaBlock, VisionMambaLayer
from .tree_mamba import TaskTreeMamba, TreeMambaLayer, build_bfs_from_adj, tree_json_to_adj
from .tree_scanning import Tree_SSM, tree_scanning

__all__ = [
    # Vision encoder
    "VisionMamba",
    "VisionMambaBlock",
    "VisionMambaLayer",
    # Task-tree encoder
    "TaskTreeMamba",
    "TreeMambaLayer",
    "build_bfs_from_adj",
    "tree_json_to_adj",
    # Core SSM primitive
    "Tree_SSM",
    "tree_scanning",
]
