"""Axis iteration strategy types for BatchPlanner.

Strategies control how a mapped axis is iterated:
- Vmap: jax.vmap — fully parallel, materializes the full axis (xtrax.tiling.Vmap).
- ChunkedMap: jax.lax.map with tile chunking — stateless, memory-bounded (xtrax.tiling.ChunkedMap).
- Scan: jax.lax.scan with carry — sequential with rolling cross-talk (xtrax.tiling.Scan).
- DedupGather: in-trace dedup-gather — deduplicate K elements from N, iterate K, scatter to N.

AxisStrategy is a sealed union; use isinstance() guards, not if/elif on strings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from xtrax.tiling import ChunkedMap, Scan, Vmap

if TYPE_CHECKING:
  from aminx.types.tiling_protocols import DedupFn, GatherFn


def _default_dedup_fn(xs: Any, unique_indices: Any) -> Any:
  return jax.tree.map(lambda x: x[unique_indices], xs)


def _default_gather_fn(ys_unique: Any, index_map: Any) -> Any:
  return jax.tree.map(lambda y: y[index_map], ys_unique)


@dataclass(frozen=True, eq=False)
class DedupGather:
  """Deduplicate-gather strategy: run body K times on unique elements, scatter to N.

  unique_indices: (K_bucket,) int32 host numpy — padded indices into the N-batch
      selecting unique entries. Computed once at plan-build; never a JAX dynamic value.
  index_map: (N,) int32 host numpy — inverse map; index_map[i]=k means position i
      takes result from unique slot k. NOT padded.
  k: int — raw unique count (pre-padding).
  k_bucket: int — padded static bucket size (>= k). XLA compiles on k_bucket, not k.
  dedup_fn: JIT-compatible in-trace gather. Default: _default_dedup_fn.
  gather_fn: JIT-compatible in-trace scatter. Default: _default_gather_fn.
  """

  unique_indices: np.ndarray
  index_map: np.ndarray
  k: int
  k_bucket: int
  dedup_fn: DedupFn = dataclasses.field(default_factory=lambda: _default_dedup_fn)
  gather_fn: GatherFn = dataclasses.field(default_factory=lambda: _default_gather_fn)


# Sealed union — all four are concrete and exhaustive.
AxisStrategy = Vmap | ChunkedMap | Scan | DedupGather

__all__ = ["AxisStrategy", "ChunkedMap", "DedupGather", "Scan", "Vmap"]
