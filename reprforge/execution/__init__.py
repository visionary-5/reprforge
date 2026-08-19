"""Execution primitives and collection compiler."""

from .assignment import PageAssignment, assign_topology_anchored, topology_anchors
from .coalescing import CoalescedState, apply_assignment, coalesce_hidden_states
from .compiler import CompilerConfig, ReprForgeCompiler

__all__ = [
    "CoalescedState",
    "CompilerConfig",
    "PageAssignment",
    "ReprForgeCompiler",
    "apply_assignment",
    "assign_topology_anchored",
    "coalesce_hidden_states",
    "topology_anchors",
]
