"""
World Model — G' (Probabilistic Graph) with VSA integration (Phase 3.2+).

Phase 3.1: No VSA component. G' similarity search (k-NN) replaces VSA
            for analogical retrieval. See v3.0 §4 Implementation Phasing.
Phase 3.2: V = python/phca/world_model/vsa.py if similarity insufficient.
"""

# Phase 3.1: WorldModelGPrime lives in graph.py, similarity.py
# Phase 3.2: VSA lives in vsa.py (conditional — see v3.0 §4.3)
