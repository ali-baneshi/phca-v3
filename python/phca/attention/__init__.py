"""Precision-Weighted Sparse Attention — k-WTA with Gumbel noise.

Phase 3.1: Attention stub — select() returns all WM chunks unchanged.
Phase 3.2+: k-WTA sparse selection with Gumbel noise, precision-weighted.
"""

from phca.attention.attention import Attention

__all__ = [
    "Attention",
]
