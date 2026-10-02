"""Decoding order places fixed positions before designed ones (aminx #2017).

ProteinMPNN/LigandMPNN decode in ``argsort((chain_mask + 1e-4) * |randn|)`` (``chain_mask`` 1 = designed), so
fixed / not-designed positions are decoded first and every designed position is conditioned on them. A tie group
is then placed at the step where its FIRST member (in that order) appears -- so a group containing ANY fixed
member is decoded in the fixed block.

On main the default order that does this is :func:`random_design_order`, used by the unified runner path
(``with_decoding_order``) and by ``aminx.sampling.sample``. :func:`random_decoding_order` is the mask-free legacy
order and is still what ``make_score_fn``, ``STEMode`` and ``optimize_ste`` draw by default: those gaps are pinned
below as strict xfails, so fixing one makes its marker fail loudly and forces the marker's removal.

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
from aminx.utils.decoding_order import random_decoding_order, random_design_order

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


# Sites that still draw the mask-free legacy order by default (#164 fixed them with a design-mask argument).
_LEGACY_DEFAULT_SITES = [
    ("aminx.scoring.score", "make_score_fn / make_*score* draw decoding_order_fn(key, L, None, None)"),
    ("aminx.inference.decode.ste", "STEMode draws self.decoding_order_fn"),
    ("aminx.inference.optimize_ste", "optimize_ste draws decoding_order_fn"),
]


@pytest.mark.xfail(strict=True, reason="aminx #2017 gap: these paths still default to the mask-free random_decoding_order (fixed in #164 only)")
@pytest.mark.parametrize(("module_name", "what"), _LEGACY_DEFAULT_SITES, ids=[m for m, _ in _LEGACY_DEFAULT_SITES])
def test_legacy_default_sites_use_a_fixed_first_order(module_name, what):
    """Structural pin: the module's default decoding-order function must not be the mask-free legacy one.
    Strict xfail -- when a site is fixed this XPASSes and the marker must be removed for that site."""
    module = importlib.import_module(module_name)
    assert module._DEFAULT_DECODING_ORDER_FN is not random_decoding_order, what
