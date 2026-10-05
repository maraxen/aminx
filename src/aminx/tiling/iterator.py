"""Iterator protocols and concrete implementations for axis iteration.

Three iterator strategies control how a mapped axis is iterated:
- VmapIterator: jax.vmap — fully parallel, stateless.
- SafeMapIterator: safe_map with tiling — memory-bounded, stateless.
- JaxScanIterator: jax.lax.scan — carry-bearing, sequential.

MapIterator and ScanIterator are runtime_checkable Protocols defining the
two fundamental iteration patterns: stateless (MapIterator) and carry-bearing
(ScanIterator).

Pattern 5 note: Concrete iterators are eqx.Module instances, NOT marked
@runtime_checkable. The protocols (MapIterator, ScanIterator) are the types
that are @runtime_checkable; users check isinstance(concrete, Protocol).

Zero production consumers as of EPIC #1541 P3 (decode-mode classes import
MapIterator/ScanIterator/VmapIterator/SafeMapIterator/JaxScanIterator from
xtrax.tiling now). Kept because tiling/dispatch.py's make_axis_dispatch
(also production-dead, also deliberately kept) still constructs these --
see that module's docstring for why. Do not delete either without retiring
T2.GATE's standing parity gate first.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.lax

from aminx.tiling._deprecation import warn_deprecated
from aminx.utils.safe_map import safe_map


@runtime_checkable
class MapIterator(Protocol):
  """Stateless axis iteration protocol.

  Maps a function over an axis without carrying state. Signature:
      fn: Callable — function to apply per-element
      xs: Any — pytree of arrays; first axis will be iterated
      in_axes: Any — specifies which axes to iterate over (default: 0)

  Returns: ys where tree_structure(ys) == tree_structure(xs) but with
  the mapped axis consumed.
  """

  def __call__(
    self,
    fn: Any,
    xs: Any,
    *,
    in_axes: Any = 0,
  ) -> Any:
    """Apply fn over the first (or specified) axis of xs.

    Args:
        fn: Callable to apply per-element.
        xs: Input pytree; iteration happens over axis 0 (or in_axes).
        in_axes: Axis specification (default 0).

    Returns:
        Output pytree with iterated axis consumed.

    """
    ...


@runtime_checkable
class ScanIterator(Protocol):
  """Carry-bearing axis iteration protocol.

  Scans over an axis, threading a carry value through iterations. Signature:
      fn: Callable — (carry, x) -> (carry, y)
      init: Any — initial carry value
      xs: Any — pytree to scan over

  Returns: (final_carry, ys) where final_carry is the final carry value
  after all iterations, and ys contains all outputs.
  """

  def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
    """Scan a function over the first axis of xs with carry.

    Args:
        fn: Callable(carry, x) -> (carry, y).
        init: Initial carry value.
        xs: Input pytree; scan happens over axis 0.

    Returns:
        (final_carry, ys): Final carry and stacked outputs.

    """
    ...


class VmapIterator(eqx.Module):
  """Iterate via jax.vmap — fully parallel.

  All elements are materialized and computed simultaneously. Use when
  memory budget allows and elements are independent (no cross-talk).
  """

  def __call__(
    self,
    fn: Any,
    xs: Any,
    *,
    in_axes: Any = 0,
  ) -> Any:
    """Apply fn using jax.vmap.

    Args:
        fn: Callable to apply per-element.
        xs: Input pytree.
        in_axes: Axis specification for vmap (default 0).

    Returns:
        Output after vmapping over the specified axis.

    """
    return jax.vmap(fn, in_axes=in_axes)(xs)


class SafeMapIterator(eqx.Module):
  """Iterate via safe_map with tile chunking — memory-bounded, stateless.

  Elements are processed in tiles to avoid memory exhaustion and XLA
  loop construct issues. No carry state; elements are independent.

  .. deprecated::
     Get the chunked iterator from xtrax:
     ``xtrax.tiling.make_axis_dispatch(xtrax.tiling.ChunkedMap(batch_size=...))``.
     aminx debt #2371; aminx's own class will be removed.
  """

  tile: int = eqx.field(static=True)

  def __post_init__(self) -> None:
    """Warn (attributed to the caller) that this class is deprecated."""
    warn_deprecated(
      "aminx.tiling.iterator.SafeMapIterator",
      "xtrax.tiling.make_axis_dispatch(xtrax.tiling.ChunkedMap(batch_size=<tile>))",
    )

  def __call__(
    self,
    fn: Any,
    xs: Any,
    *,
    in_axes: Any = 0,
  ) -> Any:
    """Apply fn using safe_map with tiling.

    Args:
        fn: Callable to apply per-element.
        xs: Input pytree.
        in_axes: Axis specification (default 0; safe_map always uses axis 0).

    Returns:
        Output after safe_map over the first axis.

    """
    # Note: safe_map always iterates over axis 0; in_axes parameter is
    # accepted for protocol compatibility.
    if in_axes != 0:
      msg = "SafeMapIterator currently only supports in_axes=0"
      raise NotImplementedError(msg)
    return safe_map(fn, xs, batch_size=self.tile)


class JaxScanIterator(eqx.Module):
  """Iterate via jax.lax.scan — carry-bearing, sequential.

  Elements are processed sequentially with a carry value threading through.
  Use when elements have dependencies or when state must be accumulated.
  """

  def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
    """Apply fn using jax.lax.scan.

    Args:
        fn: Callable(carry, x) -> (carry, y).
        init: Initial carry value.
        xs: Input pytree to scan over.

    Returns:
        (final_carry, ys): Final carry and stacked outputs.

    """
    return jax.lax.scan(fn, init, xs, unroll=1)


__all__ = [
  "JaxScanIterator",
  "MapIterator",
  "SafeMapIterator",
  "ScanIterator",
  "VmapIterator",
]
