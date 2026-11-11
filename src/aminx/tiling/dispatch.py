"""Library-side axis dispatch factory contract (Task S6-A2).

Converts AxisStrategy instances into typed iterators, with rejection logic
for invalid axis/strategy pairs. Part of the composable_jax library surface.
Backed by xtrax.tiling.dispatch.
"""

from __future__ import annotations

from aminx.tiling.errors import TilingError
from aminx.tiling.strategy import AxisStrategy


class DispatchRejected(TilingError):
  """Raised when an AxisStrategy is rejected for a given axis.

  Example: Scan on a heterogeneous axis (state) cannot be used because
  jax.lax.scan requires static carry shape, which is incompatible with
  variable-geometry state elements.
  """


# Axis names treated as heterogeneous (variable per-element shape), and therefore
# invalid under Scan. Read by dispatch paths and forwarded to xtrax as heterogeneous_axes.
_DISPATCH_HETEROGENEOUS_AXES = frozenset({"state"})


def make_axis_dispatch(strategy: AxisStrategy, *, axis: str = "state") -> object:
  """Dispatch an AxisStrategy to a typed iterator.

  Parameters
  ----------
  strategy : AxisStrategy
      One of Vmap, ChunkedMap, Scan, or DedupGather.
  axis : str, optional
      Name of the axis being dispatched. Default "state".

  Returns
  -------
  object
      A MapIterator or ScanIterator.

  Raises
  ------
  DispatchRejected
      If strategy is Scan and axis is heterogeneous (e.g., axis="state"),
      or if strategy is DedupGather (which is handled by _dispatch_axis,
      not by make_axis_dispatch).
  """
  return make_axis_dispatch_via_xtrax(strategy, axis=axis)


def make_axis_dispatch_via_xtrax(strategy: AxisStrategy, *, axis: str = "state") -> object:
  """Axis dispatch backed directly by xtrax.tiling.make_axis_dispatch.

  Raises
  ------
  DispatchRejected
      Same conditions as make_axis_dispatch (Scan on a heterogeneous axis,
      or DedupGather). Always aminx's own DispatchRejected (a TilingError
      subclass) -- xtrax's DispatchRejected is caught and translated at this boundary.
  """
  from xtrax.tiling import DispatchRejected as _XtraxDispatchRejected
  from xtrax.tiling import make_axis_dispatch as _xtrax_make_axis_dispatch

  from aminx.tiling.strategy import DedupGather

  # Reject DedupGather up front (handled by _dispatch_axis in kernel_dispatch.py, not here).
  if isinstance(strategy, DedupGather):
    raise DispatchRejected(
      "DedupGather strategy is handled by _dispatch_axis in kernel_dispatch.py, "
      "not by make_axis_dispatch (which maps to iterator types). "
      "Use DedupGather via BatchPlanner + _dispatch_axis, not make_axis_dispatch.",
    )

  try:
    return _xtrax_make_axis_dispatch(
      strategy,
      axis=axis,
      heterogeneous_axes=set(_DISPATCH_HETEROGENEOUS_AXES),
    )
  except _XtraxDispatchRejected as exc:
    raise DispatchRejected(str(exc)) from exc


__all__ = ["DispatchRejected", "make_axis_dispatch", "make_axis_dispatch_via_xtrax"]
