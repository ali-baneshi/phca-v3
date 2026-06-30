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

from phca.world_model.graph import WorldModelGPrime, StateNode, TemporalEdge
from phca.world_model.inference import forward_inference, infer_next_state
from phca.world_model.similarity import knn_similarity
from phca.world_model.mlp import WorldModelMLP

__all__ = [
    "WorldModelGPrime",
    "StateNode",
    "TemporalEdge",
    "forward_inference",
    "infer_next_state",
    "knn_similarity",
    "WorldModelMLP",
]
