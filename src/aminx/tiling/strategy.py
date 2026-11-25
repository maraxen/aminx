"""Axis iteration strategy types for BatchPlanner.

Strategies control how a mapped axis is iterated:
- Vmap: jax.vmap — fully parallel, materializes the full axis (xtrax.tiling.Vmap).
- ChunkedMap: jax.lax.map with tile chunking — stateless, memory-bounded (xtrax.tiling.ChunkedMap).
- Scan: jax.lax.scan with carry — sequential with rolling cross-talk (xtrax.tiling.Scan).
- DedupGather: in-trace dedup-gather — deduplicate K from N, iterate K, scatter to N
  (xtrax.DedupGather).

AxisStrategy is a sealed union; use isinstance() guards, not if/elif on strings.
"""

from __future__ import annotations

from xtrax import DedupGather
from xtrax.tiling import ChunkedMap, Scan, Vmap

# Sealed union — all four are concrete and exhaustive.
AxisStrategy = Vmap | ChunkedMap | Scan | DedupGather

__all__ = ["AxisStrategy", "ChunkedMap", "DedupGather", "Scan", "Vmap"]
