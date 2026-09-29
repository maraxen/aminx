"""The sampler's ar_mask and wave schedule must describe the same decoding order (debt #1982).

``make_sample_sequences`` used to build ``ar_mask`` with ``generate_ar_mask(decoding_order)``
while the bundle kept its default N->C wave. Two things were wrong at once:

* the kernel drew positions N->C, but each position saw a context chosen by a different
  permutation -- so it read undrawn slots and missed drawn ones;
* ``generate_ar_mask``'s untied branch reads a RANK array, but ``decoding_order_fn`` returns an
  ORDER array.

For a uniformly random permutation neither is visible in distribution, which is why it
survived. These tests use a deliberate, non-involutive order so that both would show.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model import Aminx
from aminx.sampling import make_sample_sequences
from aminx.types.bundles import WaveScheduleBundle
from aminx.utils.autoregression import (
  ar_mask_from_decoding_order,
  decoding_order_from_wave,
  generate_ar_mask,
  generate_wave_ar_mask,
)

L = 7


def _rotated_order(n: int, shift: int = 3) -> jnp.ndarray:
  # A rotation by 3 of 7 is not an involution, so order != argsort(order) -- the case where
  # confusing the two conventions changes the mask.
  return jnp.roll(jnp.arange(n, dtype=jnp.int32), shift)


def test_rotated_order_is_not_its_own_rank():
  order = np.asarray(_rotated_order(L))
  assert not np.array_equal(order, np.argsort(order))


def test_mask_from_order_matches_rank_convention_untied():
  order = _rotated_order(L)
  ours = np.asarray(ar_mask_from_decoding_order(order))
  ref = np.asarray(generate_ar_mask(jnp.argsort(order).astype(jnp.int32)))
  np.testing.assert_array_equal(ours, ref)


def test_mask_from_order_matches_generate_ar_mask_tied():
  tie_group_map = jnp.array([0, 0, 1, 2, 3, 3, 4], dtype=jnp.int32)
  order = jnp.array([6, 0, 1, 4, 2, 5, 3], dtype=jnp.int32)
  ours = np.asarray(ar_mask_from_decoding_order(order, tie_group_map))
  ref = np.asarray(generate_ar_mask(order, tie_group_map=tie_group_map))
  np.testing.assert_array_equal(ours, ref)


def test_tie_map_is_not_treated_as_chain_index():
  """Pin the positional-argument trap the STE paths fell into.

  ``generate_ar_mask(order, tie_map)`` binds the tie map to ``chain_idx`` (its second
  positional parameter), which hides every position of a different tie group -- i.e. almost
  all context. ``ar_mask_from_decoding_order``'s second parameter IS the tie map, so a
  position must see earlier positions of OTHER groups.
  """
  tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3], dtype=jnp.int32)
  order = jnp.array([0, 1, 2, 3, 4, 5, 6], dtype=jnp.int32)
  ours = np.asarray(ar_mask_from_decoding_order(order, tie_group_map))
  # The last-decoded position (6, its own group) sees all six earlier positions.
  assert ours[6].sum() == 6  # noqa: PLR2004
  trap = np.asarray(generate_ar_mask(order, tie_group_map))  # tie map lands in chain_idx
  assert trap[6].sum() == 0, "premise: the positional call hides other groups entirely"


def test_from_decoding_order_matches_host_side_from_tie_groups_mask():
  tie_group_map = jnp.array([0, 0, 1, 2, 3, 3, 4], dtype=jnp.int32)
  order = jnp.array([6, 0, 1, 4, 2, 5, 3], dtype=jnp.int32)
  traced = generate_wave_ar_mask(WaveScheduleBundle.from_decoding_order(order, tie_group_map), tie_group_map)
  host = generate_wave_ar_mask(WaveScheduleBundle.from_tie_groups(tie_group_map, order), tie_group_map)
  np.testing.assert_array_equal(np.asarray(traced), np.asarray(host))


def test_from_decoding_order_is_traceable():
  """The sampler draws its order inside jit, so the constructor must accept a tracer."""

  @jax.jit
  def build(key: jax.Array) -> jax.Array:
    order = jax.random.permutation(key, L).astype(jnp.int32)
    return ar_mask_from_decoding_order(order)

  mask = np.asarray(build(jax.random.PRNGKey(0)))
  assert mask.shape == (L, L)
  assert np.all(np.diag(mask) == 0)
  # A causal mask over a total order: exactly one position sees nothing, one sees all others.
  assert sorted(mask.sum(axis=1).tolist()) == list(range(L))


@pytest.fixture
def small_model(rng_key):
  # dropout_rate=0: a freshly constructed Aminx runs encoder dropout even at inference
  # (Aminx.__call__ does not forward `inference` to the encoder), which would make the
  # logits key-dependent for reasons unrelated to the mask. Pretrained checkpoints are
  # unaffected -- verified deterministic across keys.
  return Aminx(
    dropout_rate=0.0,
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=2,
    k_neighbors=8,
    key=rng_key,
  )


def _fixed_order_fn(order: jnp.ndarray):
  def fn(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ARG001
    return order, key

  return fn


def test_sampler_first_decoded_position_sees_no_sequence_context(small_model, model_inputs):
  """The first-decoded position has no drawn neighbours, so its logits cannot depend on the draw.

  Under the old wiring the kernel decoded N->C with a mask from an unrelated permutation, so
  the position the ORDER put first still saw positions the N->C sweep had already drawn, and
  its logits moved with the sampling key.
  """
  n = model_inputs["mask"].shape[0]
  order = _rotated_order(n, shift=max(1, n // 3))
  assert not np.array_equal(np.asarray(order), np.argsort(np.asarray(order)))
  sample_fn = make_sample_sequences(small_model, decoding_order_fn=_fixed_order_fn(order))

  first = int(order[0])
  seqs, first_logits = [], []
  for seed in range(4):
    seq, logits, returned_order = sample_fn(
      jax.random.PRNGKey(100 + seed),
      model_inputs["structure_coordinates"],
      model_inputs["mask"],
      model_inputs["residue_index"],
      model_inputs["chain_index"],
      backbone_noise=0.0,
      temperature=5.0,
    )
    np.testing.assert_array_equal(np.asarray(returned_order), np.asarray(order))
    seqs.append(np.asarray(seq))
    first_logits.append(np.asarray(logits[first]))

  # The draws must actually differ, or invariance of the first position is vacuous.
  assert any(not np.array_equal(seqs[0], s) for s in seqs[1:])
  for other in first_logits[1:]:
    np.testing.assert_allclose(other, first_logits[0], rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# The order a schedule decodes (code-review finding on sample.py)
# ---------------------------------------------------------------------------


def test_decoding_order_from_wave_round_trips_untied():
  order = _rotated_order(L)
  wave = WaveScheduleBundle.from_decoding_order(order)
  got = decoding_order_from_wave(wave, jnp.arange(L, dtype=jnp.int32))
  np.testing.assert_array_equal(np.asarray(got), np.asarray(order))


def test_decoding_order_from_wave_tied_uses_step_then_position():
  tie_group_map = jnp.array([0, 0, 1, 2, 3, 3, 4], dtype=jnp.int32)
  order = jnp.array([6, 5, 1, 4, 2, 0, 3], dtype=jnp.int32)
  wave = WaveScheduleBundle.from_decoding_order(order, tie_group_map)
  got = np.asarray(decoding_order_from_wave(wave, tie_group_map))
  # Groups by first appearance: g4 {6}, g3 {4, 5}, g0 {0, 1}, g1 {2}, g2 {3};
  # a tied group decodes together, members ordered by position.
  np.testing.assert_array_equal(got, [6, 4, 5, 0, 1, 2, 3])


def test_decoding_order_from_wave_puts_unscheduled_positions_last():
  tie_group_map = jnp.arange(6, dtype=jnp.int32)
  wave = WaveScheduleBundle.from_tie_groups(tie_group_map, jnp.array([4, 1, 2], dtype=jnp.int32))
  got = np.asarray(decoding_order_from_wave(wave, tie_group_map))
  np.testing.assert_array_equal(got[:3], [4, 1, 2])
  assert sorted(got[3:].tolist()) == [0, 3, 5]


def test_decoding_order_from_wave_multi_group_waves_sorted_by_wave_then_position():
  tie_group_map = jnp.arange(4, dtype=jnp.int32)
  # colors: groups 1 and 3 in wave 0, groups 0 and 2 in wave 1 (G = 2 slots per wave).
  wave = WaveScheduleBundle.from_colors(jnp.array([1, 0, 1, 0], dtype=jnp.int32), tie_group_map)
  got = np.asarray(decoding_order_from_wave(wave, tie_group_map))
  np.testing.assert_array_equal(got, [1, 3, 0, 2])


def test_sampler_reports_the_order_of_a_caller_supplied_schedule(small_model, model_inputs):
  n = model_inputs["mask"].shape[0]
  drawn = jnp.arange(n, dtype=jnp.int32)  # what decoding_order_fn would say
  scheduled = _rotated_order(n, shift=max(1, n // 3))  # what the caller asks to decode
  sample_fn = make_sample_sequences(small_model, decoding_order_fn=_fixed_order_fn(drawn))
  _, _, returned = sample_fn(
    jax.random.PRNGKey(0),
    model_inputs["structure_coordinates"],
    model_inputs["mask"],
    model_inputs["residue_index"],
    model_inputs["chain_index"],
    backbone_noise=0.0,
    wave_schedule=WaveScheduleBundle.from_decoding_order(scheduled),
  )
  np.testing.assert_array_equal(np.asarray(returned), np.asarray(scheduled))
