"""
M2 Working Memory — v3.0 §3.1 Memory Hierarchy.

Holds 7±2 chunks (the "magic number" from cognitive science).
Supports attention-gated read/write and oldest-eviction on overflow.

Concurrency: Single-writer, no reads during write (atomic swap between cycles).

Cross-ref: v3.0 §3.1 Table, Blueprint §B.1
"""

from __future__ import annotations

from phca.logging import logger, _log
from ..config import StateVector


class Chunk:
    """
    A working memory chunk — the atomic unit of M2.

    Each chunk wraps a StateVector with metadata.
    """

    def __init__(self, state: StateVector, chunk_id: int = 0, salience: float = 0.0):
        self.state = state
        self.chunk_id = chunk_id
        self.salience = salience  # attention weight
        self.age = 0  # cycles since creation

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, dim={self.state.dim}, salience={self.salience:.3f}, age={self.age})"


class M2WorkingMemory:
    """
    M2 Working Memory — bounded store with 7±2 chunks.

    Attributes:
        capacity: Maximum number of chunks (default 7).
        chunks: List of currently held chunks.
    """

    def __init__(self, capacity: int = 7, min_capacity: int = 5, max_capacity: int = 9):
        """
        Initialize working memory.

        Args:
            capacity: Default capacity (must be 5-9, default 7).
            min_capacity: Minimum capacity (5).
            max_capacity: Maximum capacity (9).
        """
        assert min_capacity <= capacity <= max_capacity, \
            f"Capacity {capacity} must be between {min_capacity} and {max_capacity} (7±2)"
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.chunks: list[Chunk] = []
        self._next_id = 0

        _log(logger, "info", "m2.init", capacity=capacity)

    def write(self, state: StateVector, salience: float = 0.0) -> Chunk:
        """
        Write a state vector into working memory.

        If the buffer is full, evict the chunk with lowest salience.

        Args:
            state: The state vector to store.
            salience: Initial salience (attention weight).

        Returns:
            The created chunk.
        """
        chunk = Chunk(state=state, chunk_id=self._next_id, salience=salience)
        self._next_id += 1

        if len(self.chunks) >= self.capacity:
            # Evict the chunk with lowest salience
            evict_idx = min(range(len(self.chunks)), key=lambda i: self.chunks[i].salience)
            evicted = self.chunks.pop(evict_idx)
            _log(logger, "debug", "m2.evict", chunk_id=evicted.chunk_id, salience=evicted.salience)

        self.chunks.append(chunk)

        # Age all chunks
        for c in self.chunks:
            c.age += 1

        _log(logger, "debug", "m2.write", chunk_id=chunk.chunk_id, salience=salience, size=len(self.chunks))
        return chunk


