"""Tests for ConditionalDecode mode class (Task 7).

ConditionalDecode wraps conditional scoring with state-axis iteration via
a MapIterator (Vmap, SafeMap). Tests verify correct output shape and dtype
over fixture sizes S ∈ {1, 4, 8} and two iterator strategies (Vmap, SafeMap).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from xtrax.tiling import ChunkedMapIterator, VmapIterator
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 8,
    seed: int = 42,
) -> tuple[Aminx, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, InferenceBundle, InferenceConfig]:
    """Build deterministic fixture with S=num_states, L=num_residues.

    Returns: (model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config)
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

    # Conditioning
    sequence_tokens = jnp.array(
        rng.integers(0, 20, size=(num_residues,), dtype=np.int32)
    )
    sequence_oh = jax.nn.one_hot(sequence_tokens, 21)

    # Use build_inference_bundle
    # Set state_weights to uniform across states
    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        sequence=sequence_oh,
        state_weights=state_weights,
        mode="score_conditional",
    )

    return model, coordinates_stack, mask_stack, residue_index_stack, chain_index_stack, sequence_oh, bundle, config


@pytest.mark.parametrize("num_states", [1, 4, 8])
@pytest.mark.parametrize("iterator_factory", [VmapIterator, ChunkedMapIterator])
def test_conditional_decode_produces_valid_logits(
    num_states: int,
    iterator_factory,
) -> None:
    """ConditionalDecode produces valid logits for various S and iterator types."""
    if iterator_factory == ChunkedMapIterator:
        iterator = iterator_factory(tile=2)
    else:
        iterator = iterator_factory()

    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = _build_synthetic_fixture(
        num_states=num_states, seed=42
    )

    # Get encoder output
    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Stage set with state weights matching the bundle's conditioning
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    # ConditionalDecode should produce logits of correct shape
    cond_decode = ConditionalDecode(model=model, state_iterator=iterator)
    logits = cond_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify shape, dtype, and finiteness
    assert logits.shape == (enc.neighbor_indices.shape[1], 21), f"Expected (L, 21), got {logits.shape}"
    assert logits.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(logits))


@pytest.mark.parametrize("num_states", [1, 3, 4, 5])
def test_conditional_decode_chunked_map_matches_vmap_numerically(num_states: int) -> None:
    """ChunkedMapIterator(tile=2) produces numerically identical logits to VmapIterator (divisible + ragged S)."""
    model, _, _, _, _, _, bundle, config = _build_synthetic_fixture(
        num_states=num_states, seed=42
    )
    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    vmap_logits = ConditionalDecode(model=model, state_iterator=VmapIterator())(
        key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set,
    )
    chunked_logits = ConditionalDecode(model=model, state_iterator=ChunkedMapIterator(tile=2))(
        key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set,
    )
    assert jnp.all(jnp.isfinite(vmap_logits))
    assert jnp.allclose(chunked_logits, vmap_logits, rtol=1e-5, atol=1e-5)


def test_conditional_decode_state_position_map_changes_fused_output() -> None:
    """End-to-end: a non-identity state_position_map actually changes fusion output.

    Confirms the wire-through from build_inference_bundle -> ConditioningBundle ->
    ConditionalDecode._apply_logit_transform (debt #572's fix) is live, not silently
    ignored. Complements the exact-math unit tests in
    tests/inference/decode/test_kernel.py::TestRealignStatesToReference.
    """
    num_states, num_residues = 2, 8
    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = (
        _build_synthetic_fixture(num_states=num_states, num_residues=num_residues, seed=7)
    )

    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )
    cond_decode = ConditionalDecode(model=model, state_iterator=VmapIterator())

    baseline_logits = cond_decode(key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set)

    # Reference state (0) stays identity; state 1's row is a real permutation
    # (no -1 gaps -- gap handling is covered separately at the kernel unit-test level).
    permuted_row = jnp.roll(jnp.arange(num_residues), shift=1)
    custom_map = jnp.stack([jnp.arange(num_residues), permuted_row])
    permuted_bundle = eqx.tree_at(
        lambda b: b.conditioning.state_position_map,
        bundle,
        custom_map,
    )

    permuted_logits = cond_decode(
        key=k_dec, enc=enc, bundle=permuted_bundle, config=config, stage_set=stage_set,
    )

    assert not jnp.allclose(permuted_logits, baseline_logits), (
        "state_position_map must actually change ConditionalDecode's fused output"
    )
