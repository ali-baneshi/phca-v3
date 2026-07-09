"""Precision-Weighted Sparse Attention — k-WTA with Gumbel noise.

Phase 3.2+: k-WTA sparse selection with Gumbel noise, precision-weighted,
            drive-dependent top-down biasing.
"""

from phca.attention.attention import Attention

__all__ = [
    "Attention",
]
