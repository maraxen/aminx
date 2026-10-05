"""EPIC #1541 / T2.4 migration-readiness probe: aminx.tiling.dispatch vs xtrax.tiling.dispatch.

Not a production test -- this verifies (empirically, not by spec-reading) whether
aminx's `inference/decode/factory.py` wrapper pattern would keep working if its
`from aminx.tiling.dispatch import ...` import were repointed to
`xtrax.tiling.dispatch`, per the migration invariants listed in
`.praxia/docs/specs/260611_aminx-xtrax-refactor.md` (T-FACTORY-HOME row).

Findings so far (see test docstrings below):
  - SafeMap field name differs (aminx: `.tile`, xtrax: `.batch_size`) -- an
    aminx-native SafeMap instance is NOT a drop-in argument to xtrax's
    make_axis_dispatch.
  - xtrax's heterogeneous-axis rejection is caller-supplied (`heterogeneous_axes`
    kwarg), not hardcoded like aminx's `axis == "state"` check -- a naive import
    swap without updating call sites to pass `heterogeneous_axes={"state"}`
    would silently stop rejecting Scan-on-state instead of raising.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.tiling.strategy import SafeMap as AminxSafeMap
from aminx.tiling.strategy import Scan as AminxScan
from aminx.tiling.strategy import Vmap as AminxVmap
from xtrax.tiling.dispatch import DispatchRejected as XtraxDispatchRejected
from xtrax.tiling.dispatch import make_axis_dispatch as xtrax_make_axis_dispatch
from xtrax.tiling.strategy import ChunkedMap as XtraxChunkedMap
from xtrax.tiling.strategy import Scan as XtraxScan
from xtrax.tiling.strategy import Vmap as XtraxVmap


def test_aminx_native_safemap_is_not_drop_in_for_xtrax_dispatch():
    """aminx's SafeMap and xtrax's SafeMap are unrelated classes (not just a
    field-name mismatch) -- xtrax's `isinstance(strategy, SafeMap)` check
    (against ITS OWN SafeMap class) is False for an aminx-native SafeMap
    instance, so it falls through every branch to the exhaustiveness TypeError.

    A naive import-swap migration (repoint `from aminx.tiling.dispatch import
    make_axis_dispatch` to xtrax's, leave strategy construction untouched) fails
    loudly at the first SafeMap call site -- not silently -- but it means the
    dispatch layer cannot migrate independently of the strategy-construction
    layer (BatchPlanner) without an adapter that translates aminx-native
    strategy instances into xtrax-native ones first.
    """
    aminx_strategy = AminxSafeMap(tile=4)
    with pytest.raises(TypeError, match="Unknown strategy type"):
        xtrax_make_axis_dispatch(aminx_strategy, axis="state")


def test_xtrax_native_safemap_works_fine_via_xtrax_dispatch():
    """Sanity check: xtrax's own SafeMap is of course a valid argument."""
    xtrax_strategy = XtraxChunkedMap(batch_size=4)
    iterator = xtrax_make_axis_dispatch(xtrax_strategy, axis="state")
    assert type(iterator).__name__ == "ChunkedMapIterator"


def test_xtrax_dispatch_does_not_reject_scan_on_state_by_default():
    """SILENT REGRESSION RISK: without passing heterogeneous_axes explicitly,
    xtrax's make_axis_dispatch does NOT reject Scan on the state axis -- unlike
    aminx's dispatch.py, which hardcodes `if axis == "state": raise
    DispatchRejected(...)` unconditionally. A call-site migration that forgets
    to pass `heterogeneous_axes={"state"}` would silently accept an invalid
    Scan-on-state strategy instead of erroring, exactly the kind of gap
    T2.GATE's parity fixtures need to catch.
    """
    strategy = XtraxScan()
    # No heterogeneous_axes passed -- default is None -> empty set.
    iterator = xtrax_make_axis_dispatch(strategy, axis="state")
    assert type(iterator).__name__ == "JaxScanIterator"  # did NOT raise


def test_xtrax_dispatch_rejects_scan_on_state_when_heterogeneous_axes_passed():
    """Confirms the fix: callers must pass heterogeneous_axes explicitly."""
    strategy = XtraxScan()
    with pytest.raises(XtraxDispatchRejected, match="heterogeneous"):
        xtrax_make_axis_dispatch(strategy, axis="state", heterogeneous_axes={"state"})


def test_xtrax_dispatch_rejected_exception_is_not_a_tilingerror():
    """aminx's DispatchRejected subclasses TilingError (so `except TilingError`
    catches it); xtrax's DispatchRejected subclasses plain Exception. A call
    site catching aminx's TilingError broadly (not just DispatchRejected) would
    NOT catch xtrax's DispatchRejected after a naive migration -- the same class
    of problem `host/plan.py:_validate_plan_topology` already solved for
    PlanTopologyError by translating xtrax's exception into aminx's own
    TilingError subclass at the boundary. The same translation pattern would be
    needed here.
    """
    from aminx.tiling.errors import TilingError

    assert not issubclass(XtraxDispatchRejected, TilingError)


def test_xtrax_vmap_iterator_call_shape_matches_aminx_expectation():
    """Confirms invariant: state_iterator(fn, inputs, in_axes=0) returns `ys`
    only (not a (final_carry, ys) tuple), matching the call pattern used at
    aminx/inference/decode/{autoregressive,conditional,unconditional}.py.
    """
    iterator = xtrax_make_axis_dispatch(XtraxVmap(), axis="state")

    def per_state_fn(x):
        return x * 2

    inputs = jnp.arange(4.0)
    result = iterator(per_state_fn, inputs, in_axes=0)
    assert result.shape == (4,)  # not a tuple -- ys only, matching MapIterator contract


def test_xtrax_iterator_is_usable_as_eqx_module_field():
    """Confirms invariant: the returned iterator can live as a field on an
    existing aminx eqx.Module (mirroring ConditionalDecode/UnconditionalDecode/
    AutoregressiveDecode, which declare `state_iterator: MapIterator` as an
    injected eqx.Module field) without breaking PyTree flatten/unflatten.
    """

    class _Probe(eqx.Module):
        state_iterator: object

    iterator = xtrax_make_axis_dispatch(XtraxVmap(), axis="state")
    probe = _Probe(state_iterator=iterator)
    leaves, treedef = jax.tree_util.tree_flatten(probe)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt.state_iterator).__name__ == "VmapIterator"
