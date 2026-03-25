"""
MTVLA action_head sub-package.

Provides the flow-matching action head used in MemoryTreeVLA.

  FlowMatchingActionHead – Conditional Flow Matching policy head
                           (adapted from Evo-1, NeurIPS 2024).
                           Consumes Z_fused from MultimodalMamba and
                           generates action chunks of shape (B, action_dim).

Reference:
  Evo-1: Pushing the Boundaries of Open-Vocabulary Embodied Action Recognition
  https://github.com/MINT-SJTU/Evo-1
"""

from .flow_matching import (
    FlowMatchingActionHead,
    BasicTransformerBlock,
    MultiEmbodimentActionEncoder,
    CategorySpecificMLP,
    CategorySpecificLinear,
    SinusoidalPositionalEncoding,
)

__all__ = [
    "FlowMatchingActionHead",
    "BasicTransformerBlock",
    "MultiEmbodimentActionEncoder",
    "CategorySpecificMLP",
    "CategorySpecificLinear",
    "SinusoidalPositionalEncoding",
]
