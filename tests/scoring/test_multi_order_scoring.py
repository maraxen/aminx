"""Unit tests for multi-order causal permutation score averaging (Clause 4: score_only_multi_order_average)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.mpnn import Aminx
from aminx.scoring.multi_order import MultiOrderScoreResult, score_multi_order_average
from aminx.utils.autoregression import ar_mask_from_decoding_order


@pytest.fixture
def small_model() -> Aminx:
  """Create lightweight Aminx model for fast scoring tests."""
  return Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=jax.random.PRNGKey(42),
  )


def test_score_multi_order_average_shapes_and_prob_normalization(
  small_model: Aminx,
) -> None:
  """Test that multi-order score averaging produces valid normalized distributions."""
  seq_len = 10
  num_orders = 4
  key = jax.random.PRNGKey(101)

  coords = jnp.zeros((seq_len, 4, 3))
  coords = coords.at[:, 1, 0].set(jnp.arange(seq_len) * 3.8)  # Extended CA trace
  mask = jnp.ones((seq_len,))
  residue_index = jnp.arange(seq_len)
  chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
  sequence = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  result = score_multi_order_average(
    model=small_model,
    prng_key=key,
    sequence=sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    num_orders=num_orders,
  )

  assert isinstance(result, MultiOrderScoreResult)
  assert result.mean_probs.shape == (seq_len, 21)
  assert result.std_probs.shape == (seq_len, 21)
  assert result.sample_nlls.shape == (num_orders,)
  assert result.decoding_orders.shape == (num_orders, seq_len)

  # Check that mean probabilities sum to 1.0 along the vocab axis
  prob_sums = jnp.sum(result.mean_probs, axis=-1)
  assert np.asarray(prob_sums) == pytest.approx(np.ones(seq_len), abs=1e-5)

  # Check that std probabilities are non-negative
  assert jnp.all(result.std_probs >= 0.0)


def test_jensens_inequality_for_multi_order_nll(small_model: Aminx) -> None:
  """Test that ensemble NLL <= mean of individual order NLLs (Jensen's inequality).

  By the concavity of log (or convexity of -log),
      -log((1/M) * sum P^(m)) <= (1/M) * sum (-log P^(m))
  with strict inequality whenever the orders produce non-identical predictions.
  """
  seq_len = 8
  num_orders = 6
  key = jax.random.PRNGKey(202)

  # Non-trivial geometry to generate distinct attention visibility per order
  coords = jax.random.normal(jax.random.PRNGKey(303), (seq_len, 4, 3))
  mask = jnp.ones((seq_len,))
  residue_index = jnp.arange(seq_len)
  chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
  sequence = jnp.arange(seq_len) % 20

  result = score_multi_order_average(
    model=small_model,
    prng_key=key,
    sequence=sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    num_orders=num_orders,
  )

  # Jensen's inequality: ensemble NLL <= mean(sample_nlls) + numerical epsilon
  assert float(result.ensemble_nll) <= float(result.mean_sample_nll) + 1e-6


def test_multi_order_determinism(small_model: Aminx) -> None:
  """Test that identical PRNG key produces bitwise identical results."""
  seq_len = 6
  key = jax.random.PRNGKey(777)

  coords = jnp.zeros((seq_len, 4, 3))
  mask = jnp.ones((seq_len,))
  residue_index = jnp.arange(seq_len)
  chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
  sequence = jnp.array([1, 2, 3, 4, 5, 6])

  res1 = score_multi_order_average(
    model=small_model,
    prng_key=key,
    sequence=sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    num_orders=3,
  )

  res2 = score_multi_order_average(
    model=small_model,
    prng_key=key,
    sequence=sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    num_orders=3,
  )

  # Bitwise identical for an identical key: exact equality, not a tolerance.
  assert float(res1.ensemble_nll) == float(res2.ensemble_nll)
  assert np.array_equal(np.asarray(res1.decoding_orders), np.asarray(res2.decoding_orders))


def test_tied_residues_decode_as_a_unit_in_every_order(small_model: Aminx) -> None:
  """Each tie group occupies consecutive steps, and its members see each other in the mask."""
  seq_len = 6
  tie = np.array([0, 1, 0, 2, 1, 3], dtype=np.int32)  # groups {0: [0,2], 1: [1,4], 2: [3], 3: [5]}
  coords = jnp.zeros((seq_len, 4, 3))
  coords = coords.at[:, 1, 0].set(jnp.arange(seq_len) * 3.8)

  result = score_multi_order_average(
    model=small_model,
    prng_key=jax.random.PRNGKey(55),
    sequence=jnp.arange(seq_len) % 20,
    structure_coordinates=coords,
    mask=jnp.ones((seq_len,)),
    residue_index=jnp.arange(seq_len),
    chain_index=jnp.zeros((seq_len,), dtype=jnp.int32),
    num_orders=16,
    tie_group_map=jnp.asarray(tie),
  )

  for order in np.asarray(result.decoding_orders):
    assert sorted(order.tolist()) == list(range(seq_len))
    step_of = {int(pos): t for t, pos in enumerate(order)}
    for group in (0, 1):
      members = [i for i in range(seq_len) if tie[i] == group]
      steps = sorted(step_of[i] for i in members)
      assert steps[-1] - steps[0] == len(members) - 1, f"group {group} not contiguous: {steps}"

    mask = np.asarray(ar_mask_from_decoding_order(jnp.asarray(order), jnp.asarray(tie)))
    for group in (0, 1):
      members = [i for i in range(seq_len) if tie[i] == group]
      for i in members:
        for j in members:
          if i != j:
            assert mask[i, j] == 1.0 and mask[j, i] == 1.0


def test_traced_tie_map_without_num_groups_is_refused(small_model: Aminx) -> None:
  """A traced tie map needs a static num_groups; the refusal is raised before any model work."""
  seq_len = 4
  coords = jnp.zeros((seq_len, 4, 3))

  def _run(tie: jax.Array) -> jax.Array:
    return score_multi_order_average(
      model=small_model,
      prng_key=jax.random.PRNGKey(0),
      sequence=jnp.arange(seq_len) % 20,
      structure_coordinates=coords,
      mask=jnp.ones((seq_len,)),
      residue_index=jnp.arange(seq_len),
      chain_index=jnp.zeros((seq_len,), dtype=jnp.int32),
      num_orders=2,
      tie_group_map=tie,
    ).ensemble_nll

  with pytest.raises(ValueError, match="num_groups"):
    jax.jit(_run)(jnp.array([0, 0, 1, 1], dtype=jnp.int32))


def test_traced_tie_map_with_explicit_num_groups_runs_under_jit(small_model: Aminx) -> None:
  """With num_groups static, the tie map may be traced. Not executed locally (tests run on titanix)."""
  seq_len = 4
  coords = jnp.zeros((seq_len, 4, 3))

  def _run(tie: jax.Array) -> jax.Array:
    return score_multi_order_average(
      model=small_model,
      prng_key=jax.random.PRNGKey(0),
      sequence=jnp.arange(seq_len) % 20,
      structure_coordinates=coords,
      mask=jnp.ones((seq_len,)),
      residue_index=jnp.arange(seq_len),
      chain_index=jnp.zeros((seq_len,), dtype=jnp.int32),
      num_orders=2,
      tie_group_map=tie,
      num_groups=2,
    ).ensemble_nll

  eager = score_multi_order_average(
    model=small_model,
    prng_key=jax.random.PRNGKey(0),
    sequence=jnp.arange(seq_len) % 20,
    structure_coordinates=coords,
    mask=jnp.ones((seq_len,)),
    residue_index=jnp.arange(seq_len),
    chain_index=jnp.zeros((seq_len,), dtype=jnp.int32),
    num_orders=2,
    tie_group_map=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
    num_groups=2,
  ).ensemble_nll
  traced = jax.jit(_run)(jnp.array([0, 0, 1, 1], dtype=jnp.int32))
  assert float(traced) == pytest.approx(float(eager), rel=1e-5)
