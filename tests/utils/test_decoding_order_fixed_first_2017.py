"""Decoding order places fixed positions before designed ones (aminx #2017).

ProteinMPNN/LigandMPNN decode in ``argsort((chain_mask + 1e-4) * |randn|)`` (``chain_mask`` 1 = designed), so
fixed / not-designed positions are decoded first and every designed position is conditioned on them. A tie group
is then placed at the step where its FIRST member (in that order) appears -- so a group containing ANY fixed
member is decoded in the fixed block.

The unified runner path (``with_decoding_order``) and ``aminx.sampling.sample`` use :func:`random_design_order`.
``STEMode`` and ``optimize_ste`` used to draw the mask-free legacy :func:`random_decoding_order` and condition their
loss on masks built from it, so a designed position could be decoded before a fixed one -- unlike the sampler. They now
draw :func:`fixed_first_decoding_order` by default (a caller's own ``decoding_order_fn`` is untouched), which is
bit-identical to the legacy draw when no position is fixed, so existing no-fixed results do not move.
``make_score_fn`` also draws the legacy order, but scoring ignores it (it uses the order-free ``full_context_ar_mask``),
so it is not a gap.

Ported from the closed-out PR #164 (``tests/utils/test_decoding_order_fixed_first_2017.py``), adapted to main's
API. One assertion is deliberately NOT ported: #164 treats a tie group with mixed members as *designed* if any
member is designed, which is the opposite of the reference rule above and of ``random_design_order``.
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
from aminx.sampling import sample
from aminx.utils.autoregression import decoding_order_from_wave
from aminx.utils.decoding_order import (
    fixed_first_decoding_order,
    random_decoding_order,
    random_design_order,
    resolve_decoding_order_fn,
)

SEEDS = (0, 1, 2, 7, 99)


def _non_designed_precede(order, designed) -> bool:
    """True when every non-designed position precedes every designed one (vacuous if either block is empty)."""
    flags = np.asarray(designed)[np.asarray(order)] > 0
    if flags.all() or not flags.any():
        return True
    first = int(np.argmax(flags))
    return bool(flags[first:].all())


def _reference_order(scores: np.ndarray, tie: np.ndarray) -> list[int]:
    """The reference's tied-order construction (model_utils.py): sort positions by score, then place each tie group
    at the step its first member appears, members in list order."""
    placed: list[int] = []
    for position in np.argsort(scores, kind="stable"):
        if int(position) in placed:
            continue
        group = [i for i in range(len(tie)) if tie[i] == tie[position]]
        placed.extend(group)
    return placed


def _groups_contiguous(order, tie) -> bool:
    order, tie = np.asarray(order), np.asarray(tie)
    steps = tie[order]
    seen: set[int] = set()
    for i, g in enumerate(steps):
        if i > 0 and steps[i - 1] != g and int(g) in seen:
            return False
        seen.add(int(g))
    return True


# --------------------------------------------------------------------------------------------------------------
# random_design_order: the fixed-first default on main
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 5, 8, 17])
def test_untied_fixed_positions_come_first(length):
    designed = np.array([(i % 3) != 0 for i in range(length)], dtype=np.float32)
    fixed = 1.0 - designed
    tie = jnp.arange(length, dtype=jnp.int32)
    for seed in SEEDS:
        order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie, jnp.asarray(fixed)))
        assert sorted(order.tolist()) == list(range(length)), "must be a permutation"
        assert _non_designed_precede(order, designed), f"designed position decoded before a fixed one (L={length}, seed={seed})"


def test_designed_block_is_still_randomised():
    """Fixed-first must not collapse the designed block to a fixed order."""
    designed = np.ones(12, dtype=np.float32)
    designed[:3] = 0.0
    tie = jnp.arange(12, dtype=jnp.int32)
    tails = {tuple(np.asarray(random_design_order(jax.random.PRNGKey(s), tie, jnp.asarray(1.0 - designed)))[3:]) for s in range(20)}
    assert len(tails) > 10


@pytest.mark.parametrize("n_groups", [1, 3, 5])
def test_homogeneous_tie_groups_fixed_first_and_contiguous(n_groups):
    length = n_groups * 2
    tie = jnp.repeat(jnp.arange(n_groups, dtype=jnp.int32), 2)
    fixed = jnp.where((tie % 2) == 0, 1.0, 0.0).astype(jnp.float32)
    designed = 1.0 - np.asarray(fixed)
    for seed in SEEDS:
        order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie, fixed))
        assert sorted(order.tolist()) == list(range(length))
        assert _groups_contiguous(order, tie), f"tie group split (G={n_groups}, seed={seed})"
        assert _non_designed_precede(order, designed), f"fixed groups not first (G={n_groups}, seed={seed})"


def test_mixed_group_counts_as_fixed_like_the_reference():
    """A group with one fixed and one designed member is decoded in the FIXED block (reference: the group is placed
    when its first member, a fixed one, is reached). #164 asserted the opposite; the reference says this."""
    tie = jnp.array([0, 0, 1], dtype=jnp.int32)  # group 0 = {0 fixed, 1 designed}; group 1 = {2 designed}
    fixed = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    for seed in SEEDS:
        order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie, fixed))
        assert order[:2].tolist() == [0, 1], f"mixed group must lead (seed={seed}): {order.tolist()}"
        assert order[2] == 2


@pytest.mark.parametrize("seed", range(12))
def test_leading_block_matches_the_reference_algorithm(seed):
    """Oracle: build the reference order from explicit scores (fixed < designed, so the 1e-4 jitter cannot flake)
    and compare the set of positions decoded in the leading fixed block against random_design_order's."""
    rng = np.random.default_rng(seed)
    length = int(rng.integers(6, 20))
    n_groups = int(rng.integers(2, length))
    tie = rng.integers(0, n_groups, size=length).astype(np.int32)
    fixed = (rng.random(length) < 0.4).astype(np.float32)
    scores = np.where(fixed > 0.5, rng.uniform(0.0, 0.1, length), rng.uniform(1.0, 2.0, length))
    ref = _reference_order(scores, tie)
    block = sum(1 for p in range(length) if any(fixed[q] > 0.5 for q in range(length) if tie[q] == tie[p]))
    got = np.asarray(random_design_order(jax.random.PRNGKey(seed), jnp.asarray(tie), jnp.asarray(fixed))).tolist()
    assert sorted(got) == list(range(length))
    assert set(got[:block]) == set(ref[:block]), (tie.tolist(), fixed.tolist(), got, ref)
    assert _groups_contiguous(got, tie)


def test_no_fixed_mask_is_a_valid_grouped_permutation():
    tie = jnp.array([0, 1, 0, 2, 1], dtype=jnp.int32)
    for seed in SEEDS:
        order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie, None))
        assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
        assert _groups_contiguous(order, tie)


def test_legacy_order_violates_fixed_first_so_the_check_can_fail():
    """Negative control: the mask-free legacy permutation does NOT respect fixed-first, so the assertions above are
    capable of failing."""
    designed = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    witnesses = [s for s in range(64) if not _non_designed_precede(np.asarray(random_decoding_order(jax.random.PRNGKey(s), 6)[0]), designed)]
    assert witnesses, "legacy permutation unexpectedly respected the design mask"


def test_non_designed_precede_helper_rejects_a_bad_order():
    designed = np.array([1.0, 0.0, 1.0])
    assert not _non_designed_precede(np.array([0, 1, 2]), designed)
    assert _non_designed_precede(np.array([1, 0, 2]), designed)


# --------------------------------------------------------------------------------------------------------------
# call sites
# --------------------------------------------------------------------------------------------------------------


def test_runner_path_with_decoding_order_decodes_fixed_first():
    length = 7
    fixed = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    bundle, _config = build_inference_bundle(
        coords=jnp.zeros((length, 4, 3), jnp.float32),
        mask=jnp.ones((length,), jnp.float32),
        residue_index=jnp.arange(length, dtype=jnp.int32),
        chain_index=jnp.zeros((length,), jnp.int32),
        fixed_mask=fixed,
        mode="sample_ar",
        inference=True,
    )
    designed = 1.0 - np.asarray(fixed)
    interleaved = 0
    for seed in range(16):
        out = with_decoding_order(bundle, jax.random.PRNGKey(seed))
        order = np.asarray(decoding_order_from_wave(out.wave, out.conditioning.tie_group_map[0]))
        assert _non_designed_precede(order, designed), f"runner order ignored fixed positions (seed={seed}): {order.tolist()}"
        interleaved += int(not _non_designed_precede(np.arange(length), designed))
    assert interleaved, "fixture must interleave fixed/designed in position order"


class _DummyModel(eqx.Module):
    sentinel: jax.Array


class _KernelOut(eqx.Module):
    sequence: jax.Array
    logits: jax.Array


def _stub_sample_kernel(model, prng_key, bundle, config, stage_set, inference_only=False):
    del model, prng_key, config, stage_set, inference_only
    length = bundle.geometry.coords.shape[1]
    return _KernelOut(sequence=jnp.zeros((length,), jnp.int8), logits=jnp.zeros((length, 21), jnp.float32))


def test_tensor_sample_threads_the_fixed_mask_into_the_order(monkeypatch):
    """aminx.sampling.sample's default order respects fixed positions (ported from #164 assertion (d))."""
    sample_module = importlib.import_module("aminx.sampling.sample")  # the function shadows the submodule name
    monkeypatch.setattr(sample_module.sample_autoregressive, "kernel", _stub_sample_kernel)
    length = 6
    fixed = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    designed = 1.0 - np.asarray(fixed)
    for seed in range(8):
        _seq, _logits, order = sample(
            jax.random.PRNGKey(seed),
            _DummyModel(sentinel=jnp.array(0, dtype=jnp.int32)),
            jnp.zeros((length, 4, 3), jnp.float32),
            jnp.ones((length,), jnp.float32),
            jnp.arange(length, dtype=jnp.int32),
            jnp.zeros((length,), jnp.int32),
            fixed_mask=fixed,
            inference_only=True,
        )
        assert _non_designed_precede(np.asarray(order), designed), f"sample() ignored fixed positions (seed={seed})"


# --------------------------------------------------------------------------------------------------------------
# fixed_first_decoding_order / resolve_decoding_order_fn: what STEMode and optimize_ste now draw by default
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tied", [False, True])
def test_no_fixed_positions_is_bit_identical_to_the_legacy_draw(tied):
    """Existing no-fixed results must not move: with nothing fixed the legacy permutation is returned unchanged."""
    length = 9
    tie = jnp.array([0, 1, 0, 2, 1, 3, 3, 4, 2], dtype=jnp.int32) if tied else None
    n_groups = 5 if tied else None
    none_fixed = jnp.zeros((length,), jnp.float32)
    for seed in range(12):
        key = jax.random.PRNGKey(seed)
        got = fixed_first_decoding_order(key, length, tie, n_groups, none_fixed)
        want, _ = random_decoding_order(key, length, tie, n_groups)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert got.dtype == want.dtype


@pytest.mark.parametrize("tied", [False, True])
def test_fixed_positions_come_first_and_match_the_design_order(tied):
    length = 9
    tie = jnp.array([0, 1, 0, 2, 1, 3, 3, 4, 2], dtype=jnp.int32) if tied else None
    fixed = jnp.array([1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=jnp.float32)
    tie_for_design = tie if tied else jnp.arange(length, dtype=jnp.int32)
    for seed in range(12):
        key = jax.random.PRNGKey(seed)
        got = np.asarray(fixed_first_decoding_order(key, length, tie, 5 if tied else None, fixed))
        np.testing.assert_array_equal(got, np.asarray(random_design_order(key, tie_for_design, fixed)))
        assert sorted(got.tolist()) == list(range(length))
    # In position order the fixed and designed positions interleave, so the check above is not vacuous.
    assert not _non_designed_precede(np.arange(length), 1.0 - np.asarray(fixed))


def test_state_batched_fixed_mask_uses_the_first_state():
    key = jax.random.PRNGKey(3)
    fixed = jnp.array([1, 0, 1, 0, 0, 1], dtype=jnp.float32)
    one = fixed_first_decoding_order(key, 6, None, None, fixed)
    two = fixed_first_decoding_order(key, 6, None, None, jnp.stack([fixed, 1.0 - fixed]))
    np.testing.assert_array_equal(np.asarray(one), np.asarray(two))


def test_helper_works_under_jit_and_vmap_over_keys():
    """STE draws a batch of orders with vmap over keys inside jit, with the mask closed over."""
    length = 8
    fixed = jnp.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=jnp.float32)
    designed = 1.0 - np.asarray(fixed)
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    batch = jax.jit(jax.vmap(lambda k: fixed_first_decoding_order(k, length, None, None, fixed)))(keys)
    assert batch.shape == (6, length)
    for row in np.asarray(batch):
        assert _non_designed_precede(row, designed)
    # And the no-fixed branch under the same transforms stays the legacy permutation.
    none_fixed = jnp.zeros((length,), jnp.float32)
    legacy_batch = jax.jit(jax.vmap(lambda k: fixed_first_decoding_order(k, length, None, None, none_fixed)))(keys)
    want = np.stack([np.asarray(random_decoding_order(k, length)[0]) for k in keys])
    np.testing.assert_array_equal(np.asarray(legacy_batch), want)


def test_resolver_upgrades_only_the_default():
    fixed = jnp.array([1, 0, 0, 1, 0], dtype=jnp.float32)

    def custom(key, num_residues, tie_group_map=None, num_groups=None):
        return jnp.arange(num_residues, dtype=jnp.int32), key

    assert resolve_decoding_order_fn(custom, fixed) is custom, "a caller's own function must be left alone"
    assert resolve_decoding_order_fn(random_decoding_order, None) is random_decoding_order, "no mask -> unchanged"
    upgraded = resolve_decoding_order_fn(random_decoding_order, fixed)
    assert upgraded is not random_decoding_order
    order, next_key = upgraded(jax.random.PRNGKey(1), 5, None, None)
    assert _non_designed_precede(np.asarray(order), 1.0 - np.asarray(fixed))
    assert next_key.shape == jax.random.PRNGKey(0).shape


def _recording_order_fn(sink: list, base):
    """Wrap an order function so every order drawn at run time (inside jit/fori_loop/vmap) lands in ``sink``."""

    def wrapped(key, num_residues, tie_group_map=None, num_groups=None):
        order, next_key = base(key, num_residues, tie_group_map, num_groups)
        jax.debug.callback(lambda o: sink.append(np.asarray(o).reshape(-1, num_residues)), order)
        return order, next_key

    return wrapped


def _run_straight_through(monkeypatch, model_inputs, rng_key, *, custom_fn):
    """Run optimize_ste's straight-through sampler with a fixed mask; return (orders drawn, fixed mask)."""
    from aminx.model.mpnn import Aminx
    from aminx.sampling import make_sample_sequences

    optimize_module = importlib.import_module("aminx.inference.optimize_ste")
    sink: list[np.ndarray] = []
    real_resolve = optimize_module.resolve_decoding_order_fn

    def spy_resolve(decoding_order_fn, fixed_mask):
        return _recording_order_fn(sink, real_resolve(decoding_order_fn, fixed_mask))

    monkeypatch.setattr(optimize_module, "resolve_decoding_order_fn", spy_resolve)
    length = int(model_inputs["mask"].shape[0])
    fixed = jnp.array([1.0 if i % 3 == 0 else 0.0 for i in range(length)], dtype=jnp.float32)
    model = Aminx(
        node_features=32, edge_features=32, hidden_features=32, num_encoder_layers=1, num_decoder_layers=1, k_neighbors=16, key=rng_key,
    )
    kwargs = {} if custom_fn is None else {"decoding_order_fn": custom_fn}
    fn = make_sample_sequences(model, sampling_strategy="straight_through", **kwargs)
    jax.block_until_ready(
        fn(
            rng_key,
            model_inputs["structure_coordinates"],
            model_inputs["mask"],
            model_inputs["residue_index"],
            model_inputs["chain_index"],
            fixed_mask=fixed,
            fixed_tokens=jnp.zeros((length,), jnp.int32),
            iterations=2,
        ),
    )
    jax.effects_barrier()
    return sink, np.asarray(fixed)


def test_optimize_ste_default_draws_fixed_first_orders(monkeypatch, model_inputs, rng_key):
    sink, fixed = _run_straight_through(monkeypatch, model_inputs, rng_key, custom_fn=None)
    assert sink, "no decoding order was drawn at run time"
    rows = np.concatenate(sink)
    assert rows.shape[0] >= 3, "expected the per-iteration batch plus the final order"
    for row in rows:
        assert _non_designed_precede(row, 1.0 - fixed), f"optimize_ste conditioned on an order that decodes designed before fixed: {row.tolist()}"


def test_optimize_ste_callers_own_order_fn_is_untouched(monkeypatch, model_inputs, rng_key):
    """Negative control: a caller-supplied order (here the identity, which interleaves fixed and designed) reaches the
    loss unchanged, so the recorder can see a non-fixed-first order and the test above can fail."""

    def identity(key, num_residues, tie_group_map=None, num_groups=None):
        return jnp.arange(num_residues, dtype=jnp.int32), key

    sink, fixed = _run_straight_through(monkeypatch, model_inputs, rng_key, custom_fn=identity)
    rows = np.concatenate(sink)
    assert rows.size
    assert all(np.array_equal(row, np.arange(rows.shape[1])) for row in rows)
    assert not _non_designed_precede(rows[0], 1.0 - fixed)


def test_ste_decode_default_draws_fixed_first_orders(monkeypatch, rng_key):
    """STEMode's decode (STEDecode) conditions its loss on fixed-first orders by default; nothing else runs it."""
    from aminx.inference.decode.factory import make_decode_fn
    from aminx.inference.decode.mode import ConditionalMode, STEMode
    from aminx.model.mpnn import Aminx
    from aminx.tiling.strategy import Vmap

    length = 24
    model = Aminx(node_features=32, edge_features=32, hidden_features=32, num_encoder_layers=1, num_decoder_layers=1, k_neighbors=8, key=rng_key)
    coords = jax.random.normal(jax.random.PRNGKey(1), (length, 4, 3)) * 3.0 + jnp.arange(length, dtype=jnp.float32)[:, None, None] * 3.8
    fixed = jnp.array([1.0 if i % 3 == 0 else 0.0 for i in range(length)], dtype=jnp.float32)
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=jnp.ones((length,)),
        residue_index=jnp.arange(length, dtype=jnp.int32),
        chain_index=jnp.zeros((length,), jnp.int32),
        fixed_mask=fixed,
        fixed_tokens=jnp.zeros((length,), jnp.int32),
        mode="score_conditional",
        inference=True,
        sequence=jnp.zeros((length,), jnp.int32),
    )
    ste_module = importlib.import_module("aminx.inference.decode.ste")
    sink: list[np.ndarray] = []
    real_resolve = ste_module.resolve_decoding_order_fn
    monkeypatch.setattr(ste_module, "resolve_decoding_order_fn", lambda fn, mask: _recording_order_fn(sink, real_resolve(fn, mask)))

    decode = make_decode_fn(model, STEMode(inner_mode=ConditionalMode(), iterations=2), Vmap(), random_decoding_order)
    out = decode(rng_key, bundle, config)
    jax.block_until_ready(out)
    jax.effects_barrier()

    assert sink, "STEDecode never drew an order through resolve_decoding_order_fn"
    rows = np.concatenate(sink)
    assert rows.shape[0] >= 9, "expected 2 iterations x batch of 4, plus the final order"
    designed = 1.0 - np.asarray(fixed)
    assert not _non_designed_precede(np.arange(length), designed), "fixture must interleave fixed/designed"
    for row in rows:
        assert _non_designed_precede(row, designed), f"STEDecode conditioned on an order that decodes designed before fixed: {row.tolist()}"
