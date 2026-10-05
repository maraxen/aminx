"""aminx's own SafeMap / SafeMapIterator are deprecated in favour of xtrax's ChunkedMap (aminx debt #2371).

Two properties matter, and each has a control:
* a caller outside aminx is warned, and the warning points at THEIR line (not at strategy.py, a dataclass
  ``__init__`` or equinox internals);
* aminx's own internal construction (planner, dispatch) stays silent, because migrating it is a later stage
  of the debt and a warning the user cannot act on is noise.
"""

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.tiling._deprecation import warn_deprecated
from aminx.tiling.axes import N_SAMPLES
from aminx.tiling.dispatch import make_axis_dispatch
from aminx.tiling.iterator import SafeMapIterator
from aminx.tiling.planner import plan_axis_strategy
from aminx.tiling.strategy import SafeMap, Vmap


def test_safemap_warns_the_caller():
    with pytest.warns(DeprecationWarning, match=r"aminx\.tiling\.SafeMap is deprecated.*xtrax\.tiling\.ChunkedMap") as rec:
        SafeMap(tile=4)
    assert len(rec) == 1
    assert rec[0].filename == __file__, "warning must be attributed to the constructing line, not aminx/dataclass internals"


def test_safemap_iterator_warns_the_caller():
    with pytest.warns(DeprecationWarning, match=r"SafeMapIterator is deprecated.*make_axis_dispatch.*ChunkedMap") as rec:
        SafeMapIterator(tile=4)
    assert len(rec) == 1
    assert rec[0].filename == __file__


def test_warning_names_the_debt_so_it_is_actionable():
    with pytest.warns(DeprecationWarning, match="debt #2371"):
        SafeMap(tile=1)


def test_behaviour_is_unchanged():
    with pytest.warns(DeprecationWarning):
        a, b = SafeMap(tile=3), SafeMap(tile=3)
    assert a == b
    assert a.tile == 3
    assert hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.tile = 5  # type: ignore[misc]
    with pytest.warns(DeprecationWarning):  # replace() re-runs __init__, so it warns at the caller too
        assert dataclasses.replace(a, tile=7).tile == 7


def test_iterator_still_maps_in_tiles():
    with pytest.warns(DeprecationWarning):
        iterator = SafeMapIterator(tile=2)
    out = iterator(lambda x: x * 2.0, jnp.arange(6.0))
    assert jnp.array_equal(out, jnp.arange(6.0) * 2.0)


def test_internal_construction_is_silent():
    """The planner builds an aminx SafeMap itself; users of plan_axis_strategy must not see a warning for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strategy = plan_axis_strategy(N_SAMPLES, 8, 4, activation_bytes_per_element=1.0)
    assert isinstance(strategy, SafeMap)
    assert strategy.tile == 4


def test_internal_dispatch_of_a_user_built_strategy_is_silent():
    """The user was warned once when they built the SafeMap; make_axis_dispatch building the iterator adds nothing."""
    with pytest.warns(DeprecationWarning):
        strategy = SafeMap(tile=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        iterator = make_axis_dispatch(strategy, axis="sample")
    assert isinstance(iterator, SafeMapIterator)


def test_pytree_round_trip_does_not_rewarn():
    """jit/tree_map rebuild modules without running __init__; that must not warn on every trace."""
    with pytest.warns(DeprecationWarning):
        iterator = SafeMapIterator(tile=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        leaves, treedef = jax.tree_util.tree_flatten(iterator)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        out = eqx.filter_jit(lambda it, x: it(lambda v: v + 1.0, x))(rebuilt, jnp.arange(4.0))
    assert jnp.array_equal(out, jnp.arange(4.0) + 1.0)


def test_unrelated_strategies_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Vmap()


def _call_helper_from_module(module_name: str) -> list[warnings.WarningMessage]:
    """Run warn_deprecated from a frame whose globals claim to be ``module_name``."""
    code = compile("def trigger():\n    warn_deprecated('old', 'new')\n", "<fake-module>", "exec")
    namespace = {"__name__": module_name, "warn_deprecated": warn_deprecated}
    exec(code, namespace)  # noqa: S102 -- controlled source, builds a function with the chosen __name__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace["trigger"]()
    return caught


def test_helper_is_silent_for_aminx_modules_and_warns_for_others():
    assert _call_helper_from_module("aminx.sampling.something") == [], "aminx-internal callers are exempt"
    assert _call_helper_from_module("aminx") == []
    warned = _call_helper_from_module("some_user_package.module")
    assert len(warned) == 1
    assert issubclass(warned[0].category, DeprecationWarning)
    assert "old is deprecated" in str(warned[0].message)
    assert "use new" in str(warned[0].message)


def test_helper_does_not_exempt_lookalike_package_names():
    """'aminxtra' starts with 'aminx' but is not aminx: only the exact package / dotted children are exempt."""
    assert len(_call_helper_from_module("aminxtra.module")) == 1
