"""Axis iteration strategy types for BatchPlanner.

Three strategies control how a mapped axis is iterated:
- Vmap: jax.vmap — fully parallel, materializes the full axis.
- SafeMap: jax.lax.map with tile chunking — stateless, memory-bounded.
- Scan: jax.lax.scan with carry — sequential with rolling cross-talk.
- DedupGather: in-trace dedup-gather — deduplicate K elements from N, iterate K, scatter to N.

AxisStrategy is a sealed union; use isinstance() guards, not if/elif on strings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import jax
import numpy as np

from aminx.tiling._deprecation import warn_deprecated

if TYPE_CHECKING:
  from aminx.types.tiling_protocols import DedupFn, GatherFn, ScanTransition

C = TypeVar("C")  # carry type
X = TypeVar("X")  # per-step input
Y = TypeVar("Y")  # per-step output


def _default_dedup_fn(xs: Any, unique_indices: Any) -> Any:
  return jax.tree.map(lambda x: x[unique_indices], xs)


def _default_gather_fn(ys_unique: Any, index_map: Any) -> Any:
  return jax.tree.map(lambda y: y[index_map], ys_unique)


@dataclass(frozen=True)
class Vmap:
  """Iterate axis via jax.vmap — fully parallel.

  All elements are materialized simultaneously. Use when memory budget allows
  and elements are independent (no cross-talk required).
  """


@dataclass(frozen=True)
class SafeMap:
  """Iterate axis via jax.lax.map with tile chunking — stateless.

  Elements are processed in tiles of `tile` elements at a time.
  No carry state; elements are independent. Use for memory-constrained
  axes where vmap would OOM.

  .. deprecated::
     Use ``xtrax.tiling.ChunkedMap(batch_size=...)`` -- the same strategy with the tile
     size named ``batch_size``. aminx debt #2371; aminx's own class will be removed.
  """

  tile: int

  def __post_init__(self) -> None:
    """Warn (attributed to the caller) that this class is deprecated."""
    warn_deprecated("aminx.tiling.SafeMap", "xtrax.tiling.ChunkedMap(batch_size=<tile>)")


@dataclass(frozen=True)
class Scan(Generic[C, X, Y]):
  """Iterate axis via jax.lax.scan with carry — rolling cross-talk.

  `init` is the initial carry value; must have static shape at JAX trace time.
  `transition(carry, x) -> (carry, y)` is called once per axis element.
  `ordered_sinks=True` means any Sink/Tap on this axis uses ordered=True
  in io_callback, preserving step order for downstream writers.

  CONSTRAINT: Scan is invalid on heterogeneous=True axes (variable-shape
  elements cannot be scanned — jax.lax.scan requires static carry shape).
  BatchPlanner.plan() and make_inference_plan() both enforce this.
  """

  init: Any  # C — initial carry; may contain JAX arrays (traced)
  transition: ScanTransition  # (C, X) -> (C, Y)
  ordered_sinks: bool = True


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
AxisStrategy = Vmap | SafeMap | Scan | DedupGather

__all__ = ["AxisStrategy", "DedupGather", "SafeMap", "Scan", "Vmap"]
