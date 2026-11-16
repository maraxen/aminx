"""Tests for tied-group autoregressive sampling path.

This module tests the full integration of tied positions with the autoregressive
sampling path, including:
1. Logit averaging across tied positions
2. Identical sequence generation for tied positions
3. Proper interaction with decoding order
4. JIT compilation compatibility
5. Integration with different sampling modes
"""

# ruff: noqa: S101, PLR2004, ANN001, ANN201, RUF059, F841

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.model.mpnn import Aminx
from aminx.sampling.sample import make_sample_sequences
from aminx.utils.autoregression import generate_ar_mask
from aminx.utils.decoding_order import random_decoding_order


@pytest.fixture
def simple_model(rng_key):
  """Create a small model for testing."""
  model = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=30,
    key=rng_key,
  )
  return eqx.tree_inference(model, value=True)


@pytest.fixture
def simple_structure():
  """Create a simple protein structure for testing."""
  n_residues = 10
  return {
    "structure_coordinates": jnp.ones((n_residues, 4, 3)),
    "mask": jnp.ones((n_residues,)),
    "residue_index": jnp.arange(n_residues),
    "chain_index": jnp.zeros((n_residues,), dtype=jnp.int32),
  }


class TestTiedAutoregressiveSampling:
  """Test suite for tied-group autoregressive sampling."""

  def test_tied_positions_produce_identical_sequences(
    self,
    simple_model,
    simple_structure,
    rng_key,
  ):
    """Test that tied positions receive identical amino acids."""
    n_residues = simple_structure["mask"].shape[0]

    # Create tie_group_map: positions 0,1,2 in group 0, positions 5,6 in group 5
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)
    tie_group_map = tie_group_map.at[2].set(0)
    tie_group_map = tie_group_map.at[6].set(5)
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Verify tied positions have identical amino acids
    assert seq[0] == seq[1] == seq[2], "Positions 0,1,2 should be identical (group 0)"
    assert seq[5] == seq[6], "Positions 5,6 should be identical (group 5)"

    # Verify sequence is valid
    chex.assert_shape(seq, (n_residues,))
    assert seq.dtype == jnp.int8
    assert jnp.all((seq >= 0) & (seq < 21))

  def test_tied_positions_without_ties(self, simple_model, simple_structure, rng_key):
    """Test backward compatibility: sampling works without tied positions."""
    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=None,
      num_groups=None,
    )

    # Verify sequence is valid
    chex.assert_shape(seq, (simple_structure["mask"].shape[0],))
    assert seq.dtype == jnp.int8
    assert jnp.all((seq >= 0) & (seq < 21))

  def test_fixed_tokens_are_respected_without_ties(self, simple_model, simple_structure, rng_key):
    """Fixed tokens should override stochastic sampling at selected positions."""
    n_residues = simple_structure["mask"].shape[0]
    fixed_token = 7
    fixed_position = 3
    fixed_mask = jnp.zeros((n_residues,), dtype=jnp.bool_).at[fixed_position].set(True)
    fixed_tokens = jnp.zeros((n_residues,), dtype=jnp.int32).at[fixed_position].set(fixed_token)

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")
    seq, _, _ = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.8, dtype=jnp.float32),
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
    )

    assert int(seq[fixed_position]) == fixed_token

  def test_fixed_tokens_are_respected_for_tied_group(self, simple_model, simple_structure, rng_key):
    """If any residue in a tie group is fixed, the whole group should use that token."""
    n_residues = simple_structure["mask"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32).at[1].set(0)
    num_groups = jnp.unique(tie_group_map).shape[0]
    fixed_token = 11
    fixed_mask = jnp.zeros((n_residues,), dtype=jnp.bool_).at[1].set(True)
    fixed_tokens = jnp.zeros((n_residues,), dtype=jnp.int32).at[1].set(fixed_token)

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")
    seq, _, _ = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.8, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
    )

    assert int(seq[0]) == fixed_token
    assert int(seq[1]) == fixed_token

  def test_multiple_tie_groups(self, simple_model, simple_structure, rng_key):
    """Test multiple independent tie groups."""
    n_residues = simple_structure["mask"].shape[0]

    # Create 3 tie groups: [0,1], [2,3,4], [5,6,7]
    tie_group_map = jnp.array([0, 0, 2, 2, 2, 5, 5, 5, 8, 9], dtype=jnp.int32)
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Verify each group has identical amino acids
    assert seq[0] == seq[1], "Group 0: positions 0,1"
    assert seq[2] == seq[3] == seq[4], "Group 2: positions 2,3,4"
    assert seq[5] == seq[6] == seq[7], "Group 5: positions 5,6,7"

    # Verify different groups can have different amino acids
    # (This is probabilistic but very likely with 10 positions and 21 amino acids)
    # We don't assert inequality since it could fail by chance

  def test_tied_sampling_jit_compatible(self, simple_model, simple_structure, rng_key):
    """Test that tied sampling is JIT-compatible."""
    n_residues = simple_structure["mask"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)  # Positions 0,1 tied
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    # This should not raise an error if JIT-compatible
    jitted_sample_fn = jax.jit(
      sample_fn,
      static_argnames=("num_groups",),
    )

    seq, logits, order = jitted_sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Verify tied positions are identical
    assert seq[0] == seq[1]
    chex.assert_shape(seq, (n_residues,))

  def test_tied_positions_with_different_temperatures(
    self, simple_model, simple_structure, rng_key,
  ):
    """Test tied sampling with different temperature values."""
    n_residues = simple_structure["mask"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)
    tie_group_map = tie_group_map.at[2].set(0)
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    # Low temperature (more deterministic)
    seq_low, _, _ = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.1, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # High temperature (more stochastic)
    seq_high, _, _ = sample_fn(
      jax.random.PRNGKey(1),
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(2.0, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Both should respect tied positions
    assert seq_low[0] == seq_low[1] == seq_low[2], "Low temp: tied positions"
    assert seq_high[0] == seq_high[1] == seq_high[2], "High temp: tied positions"

  def test_tied_positions_deterministic_with_same_seed(
    self, simple_model, simple_structure,
  ):
    """Test that tied sampling is deterministic with same seed."""
    tie_group_map = jnp.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    key1 = jax.random.PRNGKey(42)
    seq1, _, _ = sample_fn(
      key1,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    key2 = jax.random.PRNGKey(42)
    seq2, _, _ = sample_fn(
      key2,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Same seed should produce identical results
    chex.assert_trees_all_equal(seq1, seq2)

  def test_tied_positions_with_bias(self, simple_model, simple_structure, rng_key):
    """Test tied sampling with bias array."""
    n_residues = simple_structure["mask"].shape[0]
    tie_group_map = jnp.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Create bias favoring alanine (index 0)
    bias = jnp.zeros((n_residues, 21), dtype=jnp.float32)
    bias = bias.at[:, 0].set(5.0)  # Strong bias toward alanine

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, _, _ = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.1, dtype=jnp.float32),
      bias=bias,
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Tied positions should be identical
    assert seq[0] == seq[1], "Tied positions should match"

    # With strong bias and low temperature, most positions should be alanine
    # (This is probabilistic but very likely)
    chex.assert_shape(seq, (n_residues,))

  def _combine_logits_multistate(self, logits, mask):
    """Helper for testing logit averaging."""
    group_logits = jnp.where(mask[:, None], logits, -jnp.inf)
    n_tied = jnp.sum(mask)
    return jax.scipy.special.logsumexp(group_logits, axis=0, keepdims=True) - jnp.log(jnp.maximum(n_tied, 1))

  def test_logit_averaging_math(self):
    """Test the mathematical correctness of logit averaging."""
    # Test the helper method directly
    n_residues = 6
    logits = jax.random.normal(jax.random.PRNGKey(0), (n_residues, 21))

    # Create group mask for positions 0, 2, 4 (3 positions)
    group_mask = jnp.array([True, False, True, False, True, False])

    avg_logits = self._combine_logits_multistate(logits, group_mask)
    # Verify shape
    chex.assert_shape(avg_logits, (1, 21))

    # Verify manual calculation matches
    # Using log-sum-exp trick for numerical stability
    group_logits = logits[group_mask]  # (3, 21)
    max_logits = jnp.max(group_logits, axis=0, keepdims=True)
    shifted = group_logits - max_logits
    exp_shifted = jnp.exp(shifted)
    mean_exp = jnp.mean(exp_shifted, axis=0, keepdims=True)
    expected = jnp.log(mean_exp) + max_logits

    chex.assert_trees_all_close(avg_logits, expected, rtol=1e-5)

  def test_ar_mask_with_tied_positions(self, simple_structure, rng_key):
    """Test that AR mask correctly handles tied positions."""
    n_residues = simple_structure["mask"].shape[0]

    # Create tie groups: [0,1], [2,3], rest independent
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)
    tie_group_map = tie_group_map.at[3].set(2)
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Generate decoding order
    decoding_order, _ = random_decoding_order(
      rng_key, n_residues, tie_group_map, num_groups,
    )

    # Generate AR mask
    ar_mask = generate_ar_mask(order=decoding_order, tie_group_map=tie_group_map, num_groups=num_groups)

    # Verify mask shape
    chex.assert_shape(ar_mask, (n_residues, n_residues))

    # Positions in same group should attend to each other
    assert ar_mask[0, 1] == 1, "Tied positions 0,1 should attend"
    assert ar_mask[1, 0] == 1, "Tied positions 1,0 should attend"
    assert ar_mask[2, 3] == 1, "Tied positions 2,3 should attend"
    assert ar_mask[3, 2] == 1, "Tied positions 3,2 should attend"

    # No position attends to itself (2026-08-27): both reference implementations
    # self-exclude, and the "undrawn placeholder" rationale for the old set diagonal was
    # false -- undrawn slots held token 0, which is ALANINE. Tie visibility above is
    # unchanged and still asserted.
    assert jnp.all(jnp.diag(ar_mask) == 0)
    assert ar_mask.sum() > 0, "mask is all-zero -- that removes all context"


class TestStraightThroughWithTiedPositions:
  """Test suite for straight-through optimization with tied positions."""

  def test_ste_with_tied_positions(self, simple_model, simple_structure, rng_key):
    """Test that STE optimization respects tied positions."""
    n_residues = simple_structure["mask"].shape[0]

    # Create tie groups
    tie_group_map = jnp.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
    num_groups = jnp.unique(tie_group_map).shape[0]

    sample_fn = make_sample_sequences(
      simple_model, sampling_strategy="straight_through",
    )

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      iterations=jnp.array(10, dtype=jnp.int32),
      learning_rate=jnp.array(0.01, dtype=jnp.float32),
      temperature=jnp.array(1.0, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Verify tied positions have identical sequences
    assert seq[0] == seq[1], "STE: Tied positions 0,1 should match"

    # Verify sequence is valid
    chex.assert_shape(seq, (n_residues,))
    assert seq.dtype == jnp.int8
    assert jnp.all((seq >= 0) & (seq < 21))


class TestEdgeCases:
  """Test edge cases and error conditions."""

  def test_all_positions_tied(self, simple_model, simple_structure, rng_key):
    """Test when all positions are in the same tie group."""
    n_residues = simple_structure["mask"].shape[0]

    # All positions in group 0
    tie_group_map = jnp.zeros((n_residues,), dtype=jnp.int32)
    num_groups = 1

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # All positions should be identical
    assert jnp.all(seq == seq[0]), "All positions should be identical"
    chex.assert_shape(seq, (n_residues,))

  def test_no_tied_positions(self, simple_model, simple_structure, rng_key):
    """Test when no positions are tied (each in own group)."""
    n_residues = simple_structure["mask"].shape[0]

    # Each position in its own group
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    num_groups = n_residues

    sample_fn = make_sample_sequences(simple_model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      simple_structure["structure_coordinates"],
      simple_structure["mask"],
      simple_structure["residue_index"],
      simple_structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Should work normally (each position independent)
    chex.assert_shape(seq, (n_residues,))
    assert seq.dtype == jnp.int8

  def test_single_residue_structure(self, rng_key):
    """Test tied sampling with a single residue."""
    model = Aminx(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      num_encoder_layers=2,
      num_decoder_layers=2,
      k_neighbors=30,
      key=rng_key,
    )
    model = eqx.tree_inference(model, value=True)

    structure = {
      "structure_coordinates": jnp.ones((1, 4, 3)),
      "mask": jnp.ones((1,)),
      "residue_index": jnp.array([0]),
      "chain_index": jnp.array([0], dtype=jnp.int32),
    }

    tie_group_map = jnp.array([0], dtype=jnp.int32)
    num_groups = 1

    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      structure["structure_coordinates"],
      structure["mask"],
      structure["residue_index"],
      structure["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Should produce a valid single-residue sequence
    chex.assert_shape(seq, (1,))
    assert seq.dtype == jnp.int8
    assert 0 <= seq[0] < 21


class TestIntegrationWithRealModel:
  """Integration tests with more realistic model configurations."""

  def test_full_size_model_with_tied_positions(self, model_inputs, rng_key):
    """Test tied sampling with full-size model on real protein structure."""
    # Use actual protein structure from test fixtures
    n_residues = model_inputs["mask"].shape[0]

    # Create realistic tie groups (first 10 residues in 5 pairs)
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    if n_residues >= 10:
      for i in range(0, 10, 2):
        tie_group_map = tie_group_map.at[i + 1].set(i)
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Create full-size model
    model = Aminx(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      num_encoder_layers=3,
      num_decoder_layers=3,
      k_neighbors=48,
      key=rng_key,
    )
    model = eqx.tree_inference(model, value=True)

    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")

    seq, logits, order = sample_fn(
      rng_key,
      model_inputs["structure_coordinates"],
      model_inputs["mask"],
      model_inputs["residue_index"],
      model_inputs["chain_index"],
      temperature=jnp.array(0.5, dtype=jnp.float32),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Verify tied positions
    if n_residues >= 10:
      for i in range(0, 10, 2):
        assert seq[i] == seq[i + 1], f"Positions {i},{i+1} should be tied"

    # Verify sequence validity
    chex.assert_shape(seq, (n_residues,))
    assert seq.dtype == jnp.int8
    assert jnp.all((seq >= 0) & (seq < 21))
