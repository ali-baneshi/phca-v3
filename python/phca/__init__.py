"""
PHCA v3.0 — Predictive Hierarchical Cognitive Architecture.

A formally specified, resource-bounded cognitive architecture for
continual learning, intrinsic motivation, and self-regulated autonomous agents.

See docs/ for the architectural specification and implementation plan.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phca")
except PackageNotFoundError:
    __version__ = "0+unknown"
