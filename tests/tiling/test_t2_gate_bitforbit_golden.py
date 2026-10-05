"""T2.GATE bit-for-bit golden fixture: legacy vs adapter dispatch, model-level.

EPIC #1541 / T2.GATE (#1556, `.praxia/docs/specs/260611_aminx-xtrax-refactor.md`).
scripts/benchmarks/bench_xtrax_vs_aminx_tiling.py confirms decision/recompile
parity at the tiling-primitive level (synthetic toy functions); this test
confirms it at the model level -- running an actual ConditionalDecode call
through both the legacy dispatch path and T2.4's make_axis_dispatch_via_xtrax
adapter, with identical model weights/inputs/PRNG key, asserting BIT-FOR-BIT
equal output logits (not just close/approx -- any difference at all would be
a real correctness regression, since both paths are meant to run the exact
same computation, only through different tiling/dispatch plumbing).

Scope: covers the state axis (Vmap, SafeMap) via ConditionalDecode, the
simplest real call site. AutoregressiveDecode's wave-axis Scan path is a
natural follow-up (the tiling-primitive benchmark already covers Scan
recompile/decision parity in isolation) but needs its own, more involved
fixture (decoding_order_fn, wave_carry materialization) -- not built here,
flagged as a gap rather than silently assumed covered.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.tiling.dispatch import make_axis_dispatch, make_axis_dispatch_via_xtrax
from aminx.tiling.strategy import SafeMap, Vmap
from tests.inference.decode.test_conditional import _build_synthetic_fixture


@pytest.mark.parametrize("num_states", [1, 4, 8])
@pytest.mark.parametrize("strategy_cls", [Vmap, SafeMap])
def test_conditional_decode_bitforbit_identical_legacy_vs_adapter(
    num_states: int,
    strategy_cls,
) -> None:
    strategy = strategy_cls(tile=2) if strategy_cls is SafeMap else strategy_cls()

    model, _coords, _mask, _residue_index, _chain_index, _sequence_oh, bundle, config = (
        _build_synthetic_fixture(num_states=num_states, seed=42)
    )

    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    legacy_iterator = make_axis_dispatch(strategy, axis="state")
    adapter_iterator = make_axis_dispatch_via_xtrax(strategy, axis="state")

    legacy_decode = ConditionalDecode(model=model, state_iterator=legacy_iterator)
    adapter_decode = ConditionalDecode(model=model, state_iterator=adapter_iterator)

    legacy_logits = legacy_decode(
        key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set,
    )
    adapter_logits = adapter_decode(
        key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set,
    )

    max_abs_diff = float(jnp.max(jnp.abs(legacy_logits - adapter_logits)))

    if _legacy_peels_unit_chunk(strategy, num_states):
        # aminx #2391 (PR #179): the legacy path runs a chunk of one element UNBATCHED, because a vmapped
        # chunk of size 1 miscompiles on one XLA GPU stack; xtrax's ChunkedMap has no such guard and
        # vmaps it. Same maths, different XLA program, so bit equality is not guaranteed -- and on main it
        # held on some CI runs and not others (the first red run is #179's own merge commit). A real
        # adapter regression moves logits by far more than float32 reassociation, so compare closely.
        assert jnp.allclose(legacy_logits, adapter_logits, rtol=1e-5, atol=1e-5), (
            "legacy (unit chunk run unbatched) and make_axis_dispatch_via_xtrax (unit chunk vmapped) differ "
            f"by more than float32 reassociation: max |diff| = {max_abs_diff:.3e}"
        )
        return

    assert jnp.array_equal(legacy_logits, adapter_logits), (
        "Bit-for-bit mismatch between legacy make_axis_dispatch and "
        "make_axis_dispatch_via_xtrax on identical model/inputs/PRNG -- this "
        "is a real correctness regression in the T2.4 migration adapter, not "
        f"measurement noise. max |diff| = {max_abs_diff:.3e}"
    )


def _legacy_peels_unit_chunk(strategy, num_states: int) -> bool:
    """True when ``aminx.utils.safe_map`` would run a chunk of exactly one element unbatched (#179).

    That is a size-1 axis, or a trailing remainder of 1 for a chunked strategy. Vmap strategies on both
    paths go through ``jax.vmap`` and stay bit-identical; chunked strategies with no unit chunk compile the
    same program on both paths.
    """
    if num_states == 1:
        return isinstance(strategy, SafeMap)
    return isinstance(strategy, SafeMap) and num_states % strategy.tile == 1
