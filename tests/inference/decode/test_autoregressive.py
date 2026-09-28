"""Tests for AutoregressiveDecode mode class (Task 9).

AutoregressiveDecode wraps autoregressive sampling with two iterators:
- state_iterator: MapIterator for the S axis (state dimension)
- wave_iterator: ScanIterator for the W axis (wave schedule)

The wave_iterator carries the sequence through the scan; after the scan returns,
a post-hoc scatter scan maps per-wave logits back to per-position logits (Risk D-11).

Tests verify correct output shape and dtype over fixture sizes and
two iterator strategies. Also verify CarryShape round-trip and S-axis parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.autoregressive import AutoregressiveDecode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.inference.schedule_selector import build_wave_schedule_per_sample
from aminx.model import Aminx
from aminx.tiling.iterator import VmapIterator, SafeMapIterator, JaxScanIterator
from xtrax.tiling import CarryShape
from aminx.types.bundles import InferenceBundle, WaveScheduleBundle
from aminx.types.configs import InferenceConfig
from aminx.utils.autoregression import generate_wave_ar_mask


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 8,
    seed: int = 42,
) -> tuple[Aminx, InferenceBundle, InferenceConfig]:
    """Build deterministic AR fixture with S=num_states, L=num_residues.

    Returns: (model, bundle, config)
    """
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    # Build model
    model = Aminx(
        node_features=64,
        edge_features=64,
        hidden_features=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=5,
        dropout_rate=0.0,
        key=jax_key,
    )
    model = eqx.tree_inference(model, value=True)

    # Build geometry
    coordinates = jnp.array(
        rng.normal(size=(num_residues, 4, 3)).astype(np.float32)
    )
    mask = jnp.ones((num_residues,), dtype=jnp.float32)
    residue_index = jnp.arange(num_residues, dtype=jnp.int32)
    chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

    # Replicate geometry S times
    coordinates_stack = jnp.stack([coordinates] * num_states, axis=0)
    mask_stack = jnp.stack([mask] * num_states, axis=0)
    residue_index_stack = jnp.stack([residue_index] * num_states, axis=0)
    chain_index_stack = jnp.stack([chain_index] * num_states, axis=0)

    # Build bundle
    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        state_weights=state_weights,
        sequence=None,  # Will be sampled
        mode="sample_autoregressive",
    )

    return model, bundle, config


def _build_scheduled_fixture(
    num_residues: int,
    seed: int,
    schedule: str,
    schedule_key: jax.Array | None = None,
    schedule_k_neighbors: int = 4,
    fixed_mask: jax.Array | None = None,
    fixed_tokens: jax.Array | None = None,
) -> tuple[Aminx, InferenceBundle, InferenceConfig]:
    """Build an S=1 AR fixture with an explicit (W0.3) decoding schedule."""
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    model = Aminx(
        node_features=64,
        edge_features=64,
        hidden_features=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=5,
        dropout_rate=0.0,
        key=jax_key,
    )
    model = eqx.tree_inference(model, value=True)

    coordinates = jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32))
    mask = jnp.ones((num_residues,), dtype=jnp.float32)
    residue_index = jnp.arange(num_residues, dtype=jnp.int32)
    chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

    bundle, config = build_inference_bundle(
        coords=coordinates,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        state_weights=jnp.ones(1),
        sequence=None,
        mode="sample_autoregressive",
        schedule=schedule,
        schedule_key=schedule_key if schedule_key is not None else jax.random.PRNGKey(seed + 1),
        schedule_k_neighbors=schedule_k_neighbors,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )
    return model, bundle, config


def _run_ar_decode(
    model: Aminx,
    bundle: InferenceBundle,
    config: InferenceConfig,
    key: jax.Array,
):
    """Run a full encode + AutoregressiveDecode pass; returns SampleResult."""
    L = int(bundle.geometry.residue_index.shape[1])
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )
    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)
    return ar_decode(key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set)


def test_chromatic_schedule_full_coverage_and_parallelism():
    """W0.3+: chromatic schedule packs multiple groups per wave and decodes all positions."""
    model, bundle, config = _build_scheduled_fixture(
        num_residues=12,
        seed=50,
        schedule="chromatic",
        schedule_key=jax.random.PRNGKey(1),
        schedule_k_neighbors=4,
    )
    L = 12
    n_waves, max_groups_per_wave = bundle.wave.group_ids.shape

    # This fixture/key/k_neighbors combination must actually exercise G>1 waves,
    # or the test below isn't testing what it claims to.
    assert n_waves < L, "expected real parallelism (fewer waves than positions)"
    assert max_groups_per_wave > 1, "expected at least one wave with >1 group"

    result = _run_ar_decode(model, bundle, config, jax.random.PRNGKey(999))

    assert result.sequence.shape == (L,)
    assert result.logits.shape == (L, 21)
    # Every position must have been decoded by some (wave, group slot) -- a
    # silent drop (the original G>1 bug) would leave the corresponding logits
    # row identically zero.
    logits_row_norms = jnp.sum(jnp.abs(result.logits), axis=-1)
    assert bool(jnp.all(logits_row_norms > 0)), (
        f"some positions have all-zero logits (dropped by decode): {np.asarray(logits_row_norms)}"
    )


def _logits_at_target(
    model: Aminx,
    bundle: InferenceBundle,
    enc,  # noqa: ANN001 -- EncoderOutput, reused across runs (structure-only, sequence-independent)
    config: InferenceConfig,
    target: int,
    fix_position: int,
    fix_token: int,
    decode_key: jax.Array,
) -> jax.Array:
    """Re-run AutoregressiveDecode with `fix_position` overridden to `fix_token`, reusing `enc`."""
    L = int(bundle.geometry.residue_index.shape[1])
    fixed_mask = jnp.zeros((L,), dtype=jnp.float32).at[fix_position].set(1.0)
    fixed_tokens = jnp.zeros((L,), dtype=jnp.int32).at[fix_position].set(fix_token)
    b = eqx.tree_at(
        lambda bb: (bb.conditioning.fixed_mask, bb.conditioning.fixed_tokens),
        bundle,
        (fixed_mask, fixed_tokens),
    )
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )
    stage_set = make_stage_set(strategy="arithmetic_mean", state_weights=b.conditioning.state_weights)
    result = ar_decode(key=decode_key, enc=enc, bundle=b, config=config, stage_set=stage_set)
    return result.logits[target]


def test_fixed_n_to_c_causal_conditioning():
    """Default schedule: a later position that structurally attends to an earlier
    position must see that position's (fixed) token -- the ar_mask fix.
    """
    num_residues = 8
    model, bundle, config = _build_scheduled_fixture(
        num_residues=num_residues,
        seed=51,
        schedule="fixed_n_to_c",
    )
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, jax.random.PRNGKey(1), config)  # structure-only: sequence-independent

    # Pick a target position whose k-NN attention graph actually includes
    # position 0 -- without that, no masking scheme could show an effect.
    neighbor_indices = np.asarray(enc.neighbor_indices[0])  # (L, K)
    target = next((p for p in range(1, num_residues) if 0 in neighbor_indices[p]), None)
    assert target is not None, "fixture's k-NN graph never connects position 0 to a later position"

    logits_a = _logits_at_target(model, bundle, enc, config, target, 0, 3, jax.random.PRNGKey(123))
    logits_b = _logits_at_target(model, bundle, enc, config, target, 0, 15, jax.random.PRNGKey(123))

    diff = float(jnp.max(jnp.abs(logits_a - logits_b)))
    assert diff > 1e-5, (
        f"position {target} (a structural neighbor of position 0) did not depend on "
        f"position 0's fixed token (diff={diff}) -- ar_mask is not providing real "
        "causal conditioning"
    )


def test_chromatic_within_wave_independence():
    """Chromatic arm: positions in the SAME wave must not see each other's tokens,
    but a structurally-connected EARLIER-wave position must be visible.
    """
    num_residues = 12
    model, bundle, config = _build_scheduled_fixture(
        num_residues=num_residues,
        seed=52,
        schedule="chromatic",
        schedule_key=jax.random.PRNGKey(2),
        schedule_k_neighbors=4,
    )
    wave = bundle.wave
    n_waves, max_groups_per_wave = wave.group_ids.shape

    def _wave_of(position: int) -> int:
        for w in range(n_waves):
            for g in range(max_groups_per_wave):
                if bool(wave.group_valid[w, g]) and int(wave.group_positions[w, g, 0]) == position:
                    return w
        raise AssertionError(f"position {position} not found in any wave")

    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, jax.random.PRNGKey(1), config)
    neighbor_indices = np.asarray(enc.neighbor_indices[0])  # (L, K)

    # Find a wave (not the first) with >=2 active groups; pick two of its
    # representative positions as the same-wave pair.
    same_wave_pair = None
    for w in range(1, n_waves):
        active = np.flatnonzero(np.asarray(wave.group_valid[w]))
        if active.shape[0] >= 2:
            pos_a = int(wave.group_positions[w, active[0], 0])
            pos_b = int(wave.group_positions[w, active[1], 0])
            same_wave_pair = (pos_a, pos_b)
            break
    assert same_wave_pair is not None, "fixture did not produce a non-first wave with >1 group"
    pos_same, pos_other = same_wave_pair
    pos_other_wave = _wave_of(pos_other)

    # Among pos_other's REAL structural neighbors, find one in a strictly
    # earlier wave -- without real connectivity, "no change" would be trivial.
    earlier_position = next(
        (
            int(n)
            for n in neighbor_indices[pos_other]
            if int(n) != pos_other and _wave_of(int(n)) < pos_other_wave
        ),
        None,
    )
    assert earlier_position is not None, (
        f"position {pos_other} has no structural neighbor in an earlier wave than {pos_other_wave}"
    )

    logits_same_a = _logits_at_target(model, bundle, enc, config, pos_other, pos_same, 3, jax.random.PRNGKey(321))
    logits_same_b = _logits_at_target(model, bundle, enc, config, pos_other, pos_same, 15, jax.random.PRNGKey(321))
    same_wave_diff = float(jnp.max(jnp.abs(logits_same_a - logits_same_b)))

    logits_earlier_a = _logits_at_target(model, bundle, enc, config, pos_other, earlier_position, 3, jax.random.PRNGKey(321))
    logits_earlier_b = _logits_at_target(model, bundle, enc, config, pos_other, earlier_position, 15, jax.random.PRNGKey(321))
    earlier_diff = float(jnp.max(jnp.abs(logits_earlier_a - logits_earlier_b)))

    assert same_wave_diff < 1e-5, (
        f"position {pos_other}'s logits changed ({same_wave_diff}) when a SAME-wave "
        f"position ({pos_same}) was fixed to a different token -- same-wave positions "
        "must be conditionally independent (Jacobi-within-a-color)"
    )
    assert earlier_diff > 1e-5, (
        f"position {pos_other}'s logits did not change ({earlier_diff}) when its "
        f"structurally-connected EARLIER-wave position ({earlier_position}) was fixed "
        "to a different token -- earlier waves must be visible"
    )


def test_autoregressive_produces_valid_output_s1():
    """Test AR decode produces valid output, S=1 untied positions."""
    model, bundle, config = _build_synthetic_fixture(num_states=1, num_residues=8, seed=42)
    key = jax.random.PRNGKey(999)

    # Build AR mode
    L = int(bundle.geometry.residue_index.shape[1])
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    # Get stage set
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    # Encode
    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Call AR decode
    result = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify shape and dtype
    assert result.sequence.shape == (L,), f"Expected sequence shape ({L},), got {result.sequence.shape}"
    assert result.logits.shape == (L, 21), f"Expected logits shape ({L}, 21), got {result.logits.shape}"
    assert result.sequence.dtype == jnp.int32
    assert result.logits.dtype == jnp.float32


def test_autoregressive_produces_valid_output_s4():
    """Test AR decode produces valid output, S=4 untied positions."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=12, seed=43)
    key = jax.random.PRNGKey(999)

    # Build AR mode
    L = bundle.geometry.residue_index.shape[1]
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    result = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify shape and dtype
    assert result.sequence.shape == (L,), f"Expected sequence shape ({L},), got {result.sequence.shape}"
    assert result.logits.shape == (L, 21), f"Expected logits shape ({L}, 21), got {result.logits.shape}"
    assert result.sequence.dtype == jnp.int32
    assert result.logits.dtype == jnp.float32


def test_carry_shape_round_trip():
    """Test CarryShape materialization and round-trip."""
    L = 12
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)

    # Materialize init
    init = wave_carry.materialize()

    # Check shape and dtype
    assert init.shape == (L,), f"Expected shape ({L},), got {init.shape}"
    assert init.dtype == jnp.int32, f"Expected dtype int32, got {init.dtype}"

    # Check values are zeros
    np.testing.assert_array_equal(init, jnp.zeros((L,), dtype=jnp.int32))


def test_state_iterator_parity_vmap_vs_safemap():
    """Test S-axis parity: Vmap vs SafeMap inside wave scan."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=8, seed=44)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # AR decode with Vmap
    ar_vmap = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    result_vmap = ar_vmap(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # AR decode with SafeMap
    ar_safemap = AutoregressiveDecode(
        model=model,
        state_iterator=SafeMapIterator(tile=2),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    result_safemap = ar_safemap(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Results should be identical
    np.testing.assert_allclose(result_vmap.sequence, result_safemap.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_vmap.logits, result_safemap.logits, atol=1e-6)


def test_autoregressive_with_tied_positions():
    """Test AR decode with tied positions (S=1, tied groups)."""
    model, bundle, config = _build_synthetic_fixture(num_states=1, num_residues=8, seed=45)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]

    # Override tie_group_map: [0,0,1,1,2,2,3,3]. bundle.wave and ar_mask were
    # built (in build_inference_bundle) for the *original* untied tie_group_map,
    # so they must be rebuilt here too -- otherwise the schedule and causal mask
    # silently disagree with the tie groups actually used for averaging.
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    num_states = bundle.conditioning.tie_group_map.shape[0]
    new_wave = WaveScheduleBundle.from_tie_groups(tie_group_map, jnp.arange(L, dtype=jnp.int32))
    new_ar_mask = generate_wave_ar_mask(new_wave, tie_group_map)
    bundle = eqx.tree_at(
        lambda b: (b.conditioning.tie_group_map, b.conditioning.ar_mask, b.wave),
        bundle,
        (
            jnp.expand_dims(tie_group_map, 0),
            jnp.broadcast_to(new_ar_mask[None, ...], (num_states, L, L)),
            new_wave,
        ),
    )

    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Should not raise
    result = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify shape and dtype
    assert result.sequence.shape == (L,)
    assert result.logits.shape == (L, 21)


def test_ar_decode_state_position_map_changes_fused_logits() -> None:
    """End-to-end: a non-identity state_position_map changes AutoregressiveDecode's
    fused logits (praxia debt #572's fix) -- this is the actual live generative
    sampling path the real necklace production run uses, distinct from
    ConditionalDecode's teacher-forced path (covered separately in
    test_conditional.py). Compares SampleResult.logits (not the sampled sequence,
    which is a stochastic categorical draw) under the SAME PRNG key.
    """
    num_states, num_residues = 2, 8
    model, bundle, config = _build_synthetic_fixture(
        num_states=num_states, num_residues=num_residues, seed=11,
    )
    key = jax.random.PRNGKey(0)

    baseline_result = _run_ar_decode(model, bundle, config, key)

    permuted_row = jnp.roll(jnp.arange(num_residues), shift=1)
    custom_map = jnp.stack([jnp.arange(num_residues), permuted_row])
    permuted_bundle = eqx.tree_at(
        lambda b: b.conditioning.state_position_map,
        bundle,
        custom_map,
    )
    permuted_result = _run_ar_decode(model, permuted_bundle, config, key)

    assert not jnp.allclose(permuted_result.logits, baseline_result.logits), (
        "state_position_map must actually change AutoregressiveDecode's fused logits"
    )


def test_coloring_arm_via_per_sample_builder_decodes_every_position():
    """A coloring arm routed through the documented host-side path decodes ALL group slots.

    This pins, end to end, the composition that `build_inference_bundle`'s missing-key
    error message recommends:

        build_wave_schedule_per_sample(...)   # host-side, stacked on a sample axis
          -> slice one sample
          -> build_inference_bundle(wave=...) # bypasses `schedule` entirely
          -> AutoregressiveDecode

    It exists because two module docstrings claimed, until 2026-08-11, that
    `AutoregressiveDecode.__call__` "only decodes group slot 0 of each wave", so that
    "positions in slots 1..G-1 are silently never sampled". That was in fact fixed in
    85d8c480 (2026-06-30) and the docstrings were simply never updated -- but the stale
    claim was load-bearing enough to get a blocker filed against it. Asserting the real
    behavior here means the claim cannot be re-derived from prose alone.
    """
    num_residues = 12
    num_samples = 3
    model, _, _ = _build_scheduled_fixture(
        num_residues=num_residues,
        seed=50,
        schedule="fixed_n_to_c",
    )

    rng = np.random.default_rng(50)
    coordinates = jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32))
    mask = jnp.ones((num_residues,), dtype=jnp.float32)
    tie_group_map = jnp.arange(num_residues, dtype=jnp.int32)

    stacked = build_wave_schedule_per_sample(
        "chromatic",
        key=jax.random.PRNGKey(1),
        num_samples=num_samples,
        tie_group_map=tie_group_map,
        coords=coordinates,
        mask=mask,
        k_neighbors=4,
    )
    assert stacked.group_ids.shape[0] == num_samples, "builder must return a leading sample axis"
    max_groups_per_wave = stacked.group_ids.shape[2]
    assert max_groups_per_wave > 1, (
        "fixture must actually produce multi-group waves, or this test proves nothing "
        f"about slots 1..G-1 (got G={max_groups_per_wave})"
    )

    for sample_index in range(num_samples):
        wave = jax.tree.map(lambda leaf, i=sample_index: leaf[i], stacked)
        bundle, config = build_inference_bundle(
            coords=coordinates,
            mask=mask,
            residue_index=jnp.arange(num_residues, dtype=jnp.int32),
            chain_index=jnp.zeros((num_residues,), dtype=jnp.int32),
            state_weights=jnp.ones(1),
            sequence=None,
            mode="sample_autoregressive",
            tie_group_map=tie_group_map,
            wave=wave,
        )
        result = _run_ar_decode(model, bundle, config, jax.random.PRNGKey(999))

        assert result.sequence.shape == (num_residues,)
        # A position dropped because its group sat in slot >0 would leave an
        # identically-zero logits row -- the exact signature of the alleged bug.
        logits_row_norms = jnp.sum(jnp.abs(result.logits), axis=-1)
        assert bool(jnp.all(logits_row_norms > 0)), (
            f"sample {sample_index}: positions with all-zero logits (never decoded): "
            f"{np.flatnonzero(np.asarray(logits_row_norms) == 0).tolist()}"
        )


def test_autoregressive_with_tied_positions_s4():
    """Test AR decode with tied positions (S=4, tied groups)."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=12, seed=46)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]

    # Override tie_group_map: [0,0,1,1,2,2,3,3,4,4,5,5]. Rebuild wave + ar_mask
    # to match (see test_autoregressive_with_tied_positions for why).
    tie_group_map = (jnp.arange(L) // 2).astype(jnp.int32)
    num_states = bundle.conditioning.tie_group_map.shape[0]
    new_wave = WaveScheduleBundle.from_tie_groups(tie_group_map, jnp.arange(L, dtype=jnp.int32))
    new_ar_mask = generate_wave_ar_mask(new_wave, tie_group_map)
    bundle = eqx.tree_at(
        lambda b: (b.conditioning.tie_group_map, b.conditioning.ar_mask, b.wave),
        bundle,
        (
            jnp.expand_dims(tie_group_map, 0),
            jnp.broadcast_to(new_ar_mask[None, ...], (num_states, L, L)),
            new_wave,
        ),
    )

    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    result = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify shape and dtype
    assert result.sequence.shape == (L,)
    assert result.logits.shape == (L, 21)


# --------------------------------------------------------------------------------------
# Wave-order sort key (T4, sort-stability): the tie-free `decode_wave * L + arange(L)`
# key sorted with `stable=False` must reproduce the identical permutation
# `jnp.argsort(decode_wave, stable=True)` would (IREE does not honour stable-sort tie
# order; see `AutoregressiveDecode.__call__`, ~L616-639, and
# `xtrax.export.safety`'s `sort-stability` rule). Pure numeric property test -- no model
# or bundle needed, since the formula operates only on `decode_wave`/`L`.
# --------------------------------------------------------------------------------------


def _tie_free_wave_order(decode_wave: jnp.ndarray, length: int) -> jnp.ndarray:
    """Reproduces AutoregressiveDecode.__call__'s wave-order formula verbatim."""
    sort_key = decode_wave * length + jnp.arange(length, dtype=jnp.int32)
    return jnp.argsort(sort_key, stable=False)


@pytest.mark.parametrize("length", [1, 2, 8, 37, 128])
def test_wave_order_sort_key_matches_stable_argsort(length: int) -> None:
    """Random decode_wave arrays with many ties, including the n_waves sentinel value,
    across several sequence lengths."""
    rng = np.random.default_rng(7000 + length)
    n_waves = max(length // 3, 1)  # several real waves, plenty of room for ties
    for _ in range(25):
        # Values in [0, n_waves] inclusive: n_waves itself is the "never scheduled"
        # sentinel used by the real decode_wave construction (autoregressive.py L617-621).
        decode_wave = jnp.asarray(
            rng.integers(0, n_waves + 1, size=length).astype(np.int32)
        )
        got = _tie_free_wave_order(decode_wave, length)
        want = jnp.argsort(decode_wave, stable=True)
        chex.assert_trees_all_equal(got, want)

    # Degenerate: every position shares the sentinel (nothing scheduled) -- the densest
    # possible tie case, and the one the sentinel construction can actually produce.
    decode_wave = jnp.full((length,), n_waves, dtype=jnp.int32)
    got = _tie_free_wave_order(decode_wave, length)
    want = jnp.argsort(decode_wave, stable=True)
    chex.assert_trees_all_equal(got, want)
