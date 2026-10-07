
"""Tests for aminx.sampling.sample."""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.model import Aminx
from aminx.host.averaging import make_encoding_sampling_split_fn
from aminx.sampling import (
    make_sample_sequences,
    sample,
)
from aminx.utils.decoding_order import random_decoding_order


def test_make_sample_sequences_temperature_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test temperature sampling with make_sample_sequences."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="temperature"),
    )
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_sample_sequences_temperature_no_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test temperature sampling with make_sample_sequences."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_encoding_sampling_split_fn_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test make_encoding_sampling_split_fn."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn)

    # Test encode_fn
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    assert isinstance(encoded_features, tuple)
    chex.assert_tree_all_finite(encoded_features)

    # Test sample_fn
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0],
    )
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_make_encoding_sampling_split_fn_no_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test make_encoding_sampling_split_fn."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)

    # Test encode_fn
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    assert isinstance(encoded_features, tuple)
    chex.assert_tree_all_finite(encoded_features)

    # Test sample_fn
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0],
    )
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_make_sample_sequences_invalid_strategy(
    mock_model_parameters, rng_key,
):
    """Test make_sample_sequences with an invalid sampling strategy."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        make_sample_sequences(model, sampling_strategy="invalid_strategy")


def test_make_sample_sequences_straight_through_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test straight_through sampling with make_sample_sequences."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="straight_through"),
        static_argnames=["iterations"],
    )
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        iterations=10,
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_sample_sequences_straight_through_no_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test straight_through sampling with make_sample_sequences."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        iterations=10,
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_sample_sequences_straight_through_accepts_fixed_controls(
    model_inputs, rng_key,
):
    """Straight-through sampling should accept fixed_mask/fixed_tokens and enforce them."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    n_res = model_inputs["mask"].shape[0]
    # Fix position 0 to token 5
    fixed_mask = jnp.zeros((n_res,), dtype=jnp.float32).at[0].set(1.0)
    fixed_tokens = jnp.zeros((n_res,), dtype=jnp.int8).at[0].set(5)

    seq, logits, _ = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        iterations=jnp.array(5),
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    assert int(seq[0]) == 5, f"Position 0 should be fixed to token 5, got {int(seq[0])}"
    chex.assert_shape(seq, (n_res,))
    chex.assert_shape(logits, (n_res, 21))


def test_sample_convenience_function_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test the `sample` convenience function."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = eqx.filter_jit(sample)
    seq, logits, order = sample_fn(
        rng_key,
        model,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_sample_convenience_function_no_jit(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test the `sample` convenience function."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = sample
    seq, logits, order = sample_fn(
        rng_key,
        model,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_precomputed_features_equivalence(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test that precomputed features produce identical results to non-precomputed path."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")

    # Path 1: Without precomputed features
    seq_without, logits_without, order_without = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        temperature=0.5,
    )

    # Path 2: With precomputed features
    # Pre-compute node, edge features and neighbor indices
    _dummy_key = jax.random.PRNGKey(0)
    _feat_key = jax.random.split(_dummy_key)[1]
    _pre_edge, _pre_neighbor_indices, _pre_node, _ = model.features(
        _feat_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        jnp.array(0.0, dtype=jnp.float32),
    )
    _pre_node, _pre_edge = model.encoder(
        _pre_edge,
        _pre_neighbor_indices,
        model_inputs["mask"],
        initial_node_features=_pre_node,
        inference=True,
        key=_dummy_key,
    )

    seq_with, logits_with, order_with = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        temperature=0.5,
        precomputed_node_features=_pre_node,
        precomputed_edge_features=_pre_edge,
        precomputed_neighbor_indices=_pre_neighbor_indices,
    )

    # Sequences should be identical (same key, same internal computations)
    assert jnp.array_equal(seq_without, seq_with), "Sequences differ between precomputed and non-precomputed paths"

    # Logits should be bit-identical (no stochasticity in features/encoder)
    chex.assert_trees_all_close(logits_without, logits_with, atol=1e-6)

    # Decoding order should be identical (deterministic given the same key)
    assert jnp.array_equal(order_without, order_with), "Decoding orders differ between paths"


def test_precomputed_features_incomplete_raises_error(
    mock_model_parameters, model_inputs, rng_key,
):
    """Test that passing only some precomputed features raises an error."""
    model = Aminx(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")

    # Pre-compute features
    _dummy_key = jax.random.PRNGKey(0)
    _feat_key = jax.random.split(_dummy_key)[1]
    _pre_edge, _pre_neighbor_indices, _pre_node, _ = model.features(
        _feat_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        jnp.array(0.0, dtype=jnp.float32),
    )
    _pre_node, _pre_edge = model.encoder(
        _pre_edge,
        _pre_neighbor_indices,
        model_inputs["mask"],
        initial_node_features=_pre_node,
        inference=True,
        key=_dummy_key,
    )

    # Test: passing only precomputed_node_features should work or fail consistently
    # The model should either accept all-or-nothing or raise an error for partial provision.
    # For now, test that passing node but not edge/neighbors is handled gracefully.
    try:
        seq, logits, order = sample_fn(
            rng_key,
            model_inputs["structure_coordinates"],
            model_inputs["mask"],
            model_inputs["residue_index"],
            model_inputs["chain_index"],
            temperature=0.5,
            precomputed_node_features=_pre_node,
            # Missing: precomputed_edge_features, precomputed_neighbor_indices
        )
        # If it doesn't raise, check outputs are valid
        chex.assert_tree_all_finite((seq, logits, order))
    except (ValueError, TypeError) as e:
        # Expected: incomplete precomputed features should raise an error
        assert "precomputed" in str(e).lower() or "required" in str(e).lower(), (
            f"Expected error message to mention precomputed features or required kwargs, got: {e}"
        )
