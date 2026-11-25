"""World model (G').

- ``WorldModelMLP`` (``mlp.py``): pure-numpy MLP. Default for every builder.
- ``WorldModelGPrime`` (``graph.py``): discrete/Gaussian Bayesian network.
  Opt-in with ``use_mlp=False``. Discrete inference uses pgmpy when installed
  (``pip install phca[graph]``) and a dependency-free fallback otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phca.world_model.mlp import WorldModelMLP

if TYPE_CHECKING:
    from phca.world_model.graph import StateNode, TemporalEdge, WorldModelGPrime

__all__ = [
    "WorldModelGPrime",
    "StateNode",
    "TemporalEdge",
    "WorldModelMLP",
]

_GRAPH_EXPORTS = frozenset({"WorldModelGPrime", "StateNode", "TemporalEdge"})


def __getattr__(name: str):
    if name in _GRAPH_EXPORTS:
        from phca.world_model import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
