"""Library-side axis dispatch factory contract (Task S6-A2).

Converts AxisStrategy instances into typed iterators, with rejection logic
for invalid axis/strategy pairs. Part of the composable_jax library surface.

make_axis_dispatch (this module's original, aminx-native implementation) has
zero production callers as of EPIC #1541 P3 (make_decode_fn / factory.py
flipped to make_axis_dispatch_via_xtrax below). It is kept deliberately, not
as forgotten legacy: it is the reference implementation T2.GATE's Gate
Measurement Protocol compares make_axis_dispatch_via_xtrax against, and that
protocol is an explicitly STANDING gate ("re-run on production shapes, not a
one-shot at flip" per .praxia/docs/specs/260611_aminx-xtrax-refactor.md) --
see tests/tiling/test_dispatch_via_xtrax_parity.py,
tests/tiling/test_t2_4_xtrax_dispatch_compat.py,
tests/tiling/test_t2_gate_bitforbit_golden.py, and
scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py. Do not delete
without retiring that gate first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aminx.tiling.errors import TilingError
from aminx.tiling.strategy import AxisStrategy, SafeMap, Scan, Vmap

if TYPE_CHECKING:
  # Type-check-only: satisfies ty without a runtime top-level xtrax import
  # (the actual imports below stay lazy/function-local, matching this file's
  # existing circular-import-avoidance convention for aminx.tiling.iterator).
  from xtrax.tiling import ChunkedMap as XtraxChunkedMap
  from xtrax.tiling import Scan as XtraxScan
  from xtrax.tiling import Vmap as XtraxVmap


class DispatchRejected(TilingError):
  """Raised when an AxisStrategy is rejected for a given axis.

  Example: Scan on a heterogeneous axis (state) cannot be used because
  jax.lax.scan requires static carry shape, which incompatible with
  variable-geometry state elements.
  """


# Axis names treated as heterogeneous (variable per-element shape), and therefore
# invalid under Scan. Read by BOTH dispatch paths -- `make_axis_dispatch` below and
# `make_axis_dispatch_via_xtrax`, which forwards it to xtrax as `heterogeneous_axes`
# -- so the two cannot disagree about which axes are heterogeneous. It previously
# sat below `make_axis_dispatch`, which hardcoded a literal `axis == "state"` check
# and never read it; adding an axis here would have started rejecting on the xtrax
# path only, which is exactly the silent divergence the T2.GATE parity suite exists
# to catch. Add new heterogeneous axes here and both paths follow.
#
# Deliberately NOT shared with host/plan.py's _HETEROGENEOUS_AXIS_NAMES
# ({"n_states", "n_structures"}), which names axes in the BatchPlanner/CarrySpec
# context and has never been shown to mean the same thing as this one. (That
# constant used to live in tiling/carry.py, which no longer exists.)
_DISPATCH_HETEROGENEOUS_AXES = frozenset({"state"})


def make_axis_dispatch(strategy: AxisStrategy, *, axis: str = "state") -> object:
  """Dispatch an AxisStrategy to a typed iterator.

  Converts a strategy (Vmap, SafeMap, Scan) into a corresponding iterator
  (VmapIterator, SafeMapIterator, JaxScanIterator). Enforces topological
  constraints: e.g., Scan on a heterogeneous axis is rejected.

  Parameters
  ----------
  strategy : AxisStrategy
      One of Vmap, SafeMap, Scan, or DedupGather.
  axis : str, optional
      Name of the axis being dispatched. Default "state". Used to detect
      heterogeneous axes (e.g., state is heterogeneous; Scan is invalid there).

  Returns
  -------
  object
      A MapIterator (VmapIterator or SafeMapIterator) or ScanIterator
      (JaxScanIterator). Concrete types are imported at call time.

  Raises
  ------
  DispatchRejected
      If strategy is Scan and axis is heterogeneous (e.g., axis="state"),
      or if strategy is DedupGather (which is handled by _dispatch_axis,
      not by make_axis_dispatch).

  """
  # Lazy import to avoid circular dependency with iterator.py.
  # iterator.py may not exist at dispatch time (parallel task in Wave A).
  from aminx.tiling.iterator import (
    JaxScanIterator,
    SafeMapIterator,
    VmapIterator,
  )
  from aminx.tiling.strategy import DedupGather

  # Reject DedupGather (handled by _dispatch_axis in kernel_dispatch.py, not here).
  if isinstance(strategy, DedupGather):
    raise DispatchRejected(
      "DedupGather strategy is handled by _dispatch_axis in kernel_dispatch.py, "
      "not by make_axis_dispatch (which maps to iterator types). "
      "Use DedupGather via BatchPlanner + _dispatch_axis, not make_axis_dispatch.",
    )

  # Reject Scan on heterogeneous axes ("state" is currently the only one; see
  # _DISPATCH_HETEROGENEOUS_AXES above, which the xtrax path reads too).
  if isinstance(strategy, Scan) and axis in _DISPATCH_HETEROGENEOUS_AXES:
    raise DispatchRejected(
      f"Cannot use Scan strategy on {axis} axis: {axis} axis contains "
      "heterogeneous (variable-shape) elements. Scan requires "
      "static carry shape across all iterations.",
    )

  # Dispatch by strategy type.
  if isinstance(strategy, Vmap):
    return VmapIterator()
  if isinstance(strategy, SafeMap):
    return SafeMapIterator(tile=strategy.tile)
  if isinstance(strategy, Scan):
    return JaxScanIterator()
  # Exhaustiveness check: should never reach here if AxisStrategy is sealed.
  raise TypeError(f"Unknown strategy type: {type(strategy)}")


# Axis names make_axis_dispatch currently treats as heterogeneous (hardcoded
# as a literal "state" check above). Kept as an explicit constant here, not
# reused from tiling/carry.py's _HETEROGENEOUS_AXIS_NAMES ({"n_states",
# "n_structures"}), since that constant serves a different naming context
# (CarrySpec) and has never been shown to mean the same thing as this one.
_DISPATCH_HETEROGENEOUS_AXES = frozenset({"state"})


def _strategy_to_xtrax(strategy: AxisStrategy) -> XtraxVmap | XtraxChunkedMap | XtraxScan:
  """Translate an aminx-native AxisStrategy into its xtrax-native equivalent.

  aminx's and xtrax's Vmap/SafeMap/Scan classes are structurally similar but
  are NOT the same classes (xtrax's `isinstance` checks are against its own
  classes) -- an aminx-native strategy instance is not a valid argument to
  xtrax.tiling.dispatch.make_axis_dispatch without this translation. See
  tests/tiling/test_t2_4_xtrax_dispatch_compat.py for the empirical finding.
  """
  from xtrax.tiling import ChunkedMap as _XtraxChunkedMap
  from xtrax.tiling import Scan as _XtraxScan
  from xtrax.tiling import Vmap as _XtraxVmap

  if isinstance(strategy, Vmap):
    return _XtraxVmap()
  if isinstance(strategy, SafeMap):
    return _XtraxChunkedMap(batch_size=strategy.tile)
  if isinstance(strategy, Scan):
    return _XtraxScan(
      transition=strategy.transition,
      init=strategy.init,
      ordered_sinks=strategy.ordered_sinks,
    )
  raise TypeError(f"Unknown strategy type: {type(strategy)}")


def make_axis_dispatch_via_xtrax(strategy: AxisStrategy, *, axis: str = "state") -> object:
  """T2.4 migration target: make_axis_dispatch, backed by xtrax.tiling.dispatch.

  EPIC #1541 (aminx.tiling -> xtrax.tiling migration,
  `.praxia/docs/specs/260611_aminx-xtrax-refactor.md`, T-FACTORY-HOME row).
  Mirrors `make_axis_dispatch`'s exact contract (same accepted/rejected
  strategy-axis pairs, same DispatchRejected/TilingError hierarchy, same
  returned-iterator call shape) while delegating iterator construction to
  xtrax. Not yet wired into any call site -- `factory.py` and friends still
  import `make_axis_dispatch` above. This is the flip target once T2.GATE
  (bit-for-bit parity + recompile-count tripwire + cluster throughput bench)
  passes; see tests/tiling/test_dispatch_via_xtrax_parity.py for the parity
  suite this function must keep matching.

  Two adaptations `make_axis_dispatch` doesn't need, both proven empirically
  in tests/tiling/test_t2_4_xtrax_dispatch_compat.py:
  1. Strategy objects are translated via `_strategy_to_xtrax` first --
     aminx-native and xtrax-native Vmap/SafeMap/Scan are distinct classes.
  2. `heterogeneous_axes` is passed explicitly to xtrax's make_axis_dispatch
     (it defaults to none rejected, unlike aminx's hardcoded axis=="state"
     check) -- forgetting this would silently stop rejecting Scan-on-state.

  Raises
  ------
  DispatchRejected
      Same conditions as make_axis_dispatch (Scan on a heterogeneous axis,
      or DedupGather). Always aminx's own DispatchRejected (a TilingError
      subclass) -- xtrax's DispatchRejected (a plain Exception, not a
      TilingError) is caught and translated at this boundary, the same
      pattern `host/plan.py:_validate_plan_topology` already uses for
      PlanTopologyError.

  """
  from xtrax.tiling import DispatchRejected as _XtraxDispatchRejected
  from xtrax.tiling import make_axis_dispatch as _xtrax_make_axis_dispatch

  from aminx.tiling.strategy import DedupGather

  # Reject DedupGather up front (same message/exception as make_axis_dispatch;
  # no need to even translate the strategy for a path that always rejects).
  if isinstance(strategy, DedupGather):
    raise DispatchRejected(
      "DedupGather strategy is handled by _dispatch_axis in kernel_dispatch.py, "
      "not by make_axis_dispatch (which maps to iterator types). "
      "Use DedupGather via BatchPlanner + _dispatch_axis, not make_axis_dispatch.",
    )

  xtrax_strategy = _strategy_to_xtrax(strategy)
  try:
    return _xtrax_make_axis_dispatch(
      xtrax_strategy,
      axis=axis,
      heterogeneous_axes=set(_DISPATCH_HETEROGENEOUS_AXES),
    )
  except _XtraxDispatchRejected as exc:
    raise DispatchRejected(str(exc)) from exc


__all__ = ["DispatchRejected", "make_axis_dispatch", "make_axis_dispatch_via_xtrax"]
