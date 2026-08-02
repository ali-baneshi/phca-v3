"""
World Model — G' (Probabilistic Graph) with VSA integration (Phase 3.2+).

Phase 3.1:
    - G' Bayesian network with discrete CPDs (graph.py)
    - Exact junction tree inference via pgmpy (inference.py)
    - k-NN similarity search replaces VSA (similarity.py)

Phase 3.2:
    - V = python/phca/world_model/vsa.py if similarity insufficient
    - Sampling inference for |V| > 100
    - Dual-model ensemble (G' + V)

Cross-ref: v3.0 §2.2, §4, §D.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phca.world_model.ensemble import WorldModelMLPEnsemble
from phca.world_model.mlp import WorldModelMLP

if TYPE_CHECKING:
    from phca.world_model.graph import StateNode, TemporalEdge, WorldModelGPrime

__all__ = [
    "WorldModelGPrime",
    "StateNode",
    "TemporalEdge",
    "WorldModelMLP",
    "WorldModelMLPEnsemble",
    "HybridGraphMLP",
]

_GRAPH_EXPORTS = frozenset({"WorldModelGPrime", "StateNode", "TemporalEdge"})


def __getattr__(name: str):
    if name in _GRAPH_EXPORTS:
        from phca.world_model import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
