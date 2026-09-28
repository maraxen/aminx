"""Tests for autoregression utilities."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.run.specs import RunSpecification
from aminx.utils.autoregression import generate_ar_mask, generate_wave_ar_mask, resolve_tie_groups
from aminx.utils.data_structures import Protein
from aminx.types.bundles import WaveScheduleBundle


def make_input(chain_id, residue_index):
  """Create a minimal Protein with batch_dim=1 for testing resolve_tie_groups."""
  n = len(chain_id)
  return Protein(
    coordinates=jnp.zeros((1, n, 4, 3)),  # batch_dim=1
    aatype=jnp.zeros((1, n), dtype=jnp.int32),  # Dummy amino acid types
    one_hot_sequence=jnp.zeros((1, n, 21), dtype=jnp.float32),  # Dummy one-hot
    mask=jnp.ones((1, n), dtype=jnp.bool_),
    chain_index=jnp.array(chain_id, dtype=jnp.int32)[None, :],  # Add batch dim
    residue_index=jnp.array(residue_index, dtype=jnp.int32)[None, :],  # Add batch dim
  )


def test_resolve_tie_groups_none():
  """Test resolve_tie_groups with tied_positions=None."""
  inp = make_input([0, 0, 1], [1, 2, 3])
  spec = RunSpecification(inputs=["dummy"], tied_positions=None)
  out = resolve_tie_groups(spec, inp)
  assert (out == jnp.arange(3)).all()


def test_resolve_tie_groups_direct():
  """Test resolve_tie_groups with tied_positions='direct'."""
  # N=6, L=3, K=2
  # Chain IDs [0,0,0, 1,1,1] indicate 2 chains
  # Mapping [0,0,0, 1,1,1] indicates 2 structures
  inp = make_input([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
  # Add structure mapping: [0,0,0, 1,1,1] for 2 structures
  inp = inp.replace(mapping=jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32))
  spec = RunSpecification(inputs=["dummy"], tied_positions="direct", pass_mode="inter")
  out = resolve_tie_groups(spec, inp)
  assert (out == jnp.array([0, 1, 2, 0, 1, 2])).all()


def test_resolve_tie_groups_auto():
  """Test resolve_tie_groups with tied_positions='auto'."""
  inp = make_input([0, 0, 1, 1], [0, 1, 0, 1])
  spec = RunSpecification(inputs=["dummy"], tied_positions="auto", pass_mode="inter")
  # structure_mappings: seq_pos 0 -> [0,2], seq_pos 1 -> [1,3]
  structure_mappings = [[0, 2], [1, 3]]
  out = resolve_tie_groups(spec, inp, structure_mappings)
  # Should assign same group to 0,2 and to 1,3
  assert out[0] == out[2]
  assert out[1] == out[3]
  assert len(jnp.unique(out)) == 2  # noqa: PLR2004


def test_resolve_tie_groups_explicit():
  """Test resolve_tie_groups with explicit tied positions."""
  inp = make_input([10000, 10001, 10000, 10001], [5, 10, 6, 11])
  spec = RunSpecification(inputs=["dummy"], tied_positions=[[(10000, 5), (10001, 10)]])
  out = resolve_tie_groups(spec, inp)
  chain_ids = inp.chain_index[0]
  residue_indices = inp.residue_index[0]
  idx0 = jnp.where((chain_ids == 10000) & (residue_indices == 5))[0][0]  # noqa: PLR2004
  idx1 = jnp.where((chain_ids == 10001) & (residue_indices == 10))[0][0]  # noqa: PLR2004
  assert out[idx0] == out[idx1]
  # The other indices should be in different groups
  assert len(jnp.unique(out)) == 3  # noqa: PLR2004


@pytest.mark.parametrize(
    "decoding_order, expected_mask",
    [
        # Diagonals are 0 as of 2026-08-27: no position reads its own slot. Each expected
        # matrix below is its previous value with the diagonal zeroed and nothing else
        # touched. Note the argument is a RANK array -- `decoding_order[i]` is the step at
        # which position i decodes -- so for [2, 0, 1] position 0 decodes last and sees
        # both others.
        (
            jnp.array([0, 1, 2]),
            jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        ),
        (
            jnp.array([2, 0, 1]),
            jnp.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]]),
        ),
        (
            jnp.array([1, 2, 0]),
            jnp.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]]),
        ),
    ],
)
def test_generate_ar_mask(decoding_order, expected_mask):
    """Test the generation of the autoregressive mask.

    Args:
        decoding_order: The order in which atoms are decoded.
        expected_mask: The expected autoregressive mask.

    Raises:
        AssertionError: If the output does not match the expected value.

    """
    mask = generate_ar_mask(decoding_order)
    chex.assert_trees_all_equal(mask, expected_mask)
    chex.assert_shape(mask, (len(decoding_order), len(decoding_order)))
    chex.assert_type(mask, int)


def test_generate_wave_ar_mask_matches_generate_ar_mask_for_g1_schedules():
  """G=1 schedules (from_tie_groups) must reduce exactly to generate_ar_mask's semantics."""
  tie_group_map = jnp.array([0, 0, 1, 2, 3, 3, 4, 5], dtype=jnp.int32)
  decoding_order = jnp.array([6, 0, 1, 4, 2, 7, 3, 5], dtype=jnp.int32)
  wave = WaveScheduleBundle.from_tie_groups(tie_group_map, decoding_order)
  new_mask = generate_wave_ar_mask(wave, tie_group_map)
  old_mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map).astype(jnp.float32)
  chex.assert_trees_all_equal(new_mask, old_mask)


def test_generate_wave_ar_mask_partial_schedule_does_not_leak_omitted_positions():
  """A *partial* decoding order (some positions never assigned a wave) must not make
  the omitted positions falsely visible to real (scheduled) positions.

  Regression test: the wave-index sentinel for "never scheduled" was originally -1,
  which is *smaller* than every real wave index, so `earlier_wave = pos_wave_index[i]
  > pos_wave_index[j]` made every omitted position `j` look earlier than everything
  real -- silently leaking its (never-decoded, all-zero) value as false context.
  Caught empirically: truncating a sequential reference schedule to only the positions
  that matter changed the sampled output relative to the full schedule.
  """
  tie_group_map = jnp.arange(6, dtype=jnp.int32)
  partial_order = jnp.array([0, 1, 2], dtype=jnp.int32)  # positions 3, 4, 5 never scheduled
  wave = WaveScheduleBundle.from_tie_groups(tie_group_map, partial_order)
  mask = generate_wave_ar_mask(wave, tie_group_map)

  # Real, scheduled position 2 (wave index 2, the last real wave) must not see any
  # omitted position as "earlier" -- omitted positions were never decoded.
  chex.assert_trees_all_equal(mask[2, 3:6], jnp.zeros(3))
  # And the omitted positions must not be treated as visible to earlier real waves either.
  chex.assert_trees_all_equal(mask[0, 3:6], jnp.zeros(3))
  chex.assert_trees_all_equal(mask[1, 3:6], jnp.zeros(3))
  # Real positions still see each other exactly as a normal sequential schedule would.
  # k=-1: STRICTLY lower-triangular, since as of 2026-08-27 no position reads its own slot.
  # The leak this test guards is unrelated to the diagonal and is unaffected by that change.
  chex.assert_trees_all_equal(mask[:3, :3], jnp.tril(jnp.ones((3, 3)), k=-1))


def test_generate_wave_ar_mask_padding_tolerant():
  """pad_bundle-style zero-padded waves (group_valid=False) must not perturb the mask."""
  tie_group_map = jnp.array([0, 0, 1, 2, 3, 3, 4, 5], dtype=jnp.int32)
  decoding_order = jnp.array([6, 0, 1, 4, 2, 7, 3, 5], dtype=jnp.int32)
  wave = WaveScheduleBundle.from_tie_groups(tie_group_map, decoding_order)
  unpadded_mask = generate_wave_ar_mask(wave, tie_group_map)

  padded_wave = WaveScheduleBundle(
    group_ids=jnp.pad(wave.group_ids, ((0, 2), (0, 0))),
    group_positions=jnp.pad(wave.group_positions, ((0, 2), (0, 0), (0, 0))),
    group_valid=jnp.pad(wave.group_valid, ((0, 2), (0, 0))),
    position_valid=jnp.pad(wave.position_valid, ((0, 2), (0, 0), (0, 0))),
  )
  padded_mask = generate_wave_ar_mask(padded_wave, tie_group_map)
  chex.assert_trees_all_equal(padded_mask, unpadded_mask)


# --------------------------------------------------------------------------------------
# generate_ar_mask's tied branch: tie-free (key, index) lex sort must reproduce the
# identical permutation a stable argsort on the key alone would (T4, sort-stability).
#
# Until this fix, `generate_ar_mask`'s tied branch computed `group_decoding_order` with
# a bare `jnp.argsort(...)` (default `stable=True`) over a key where every absent group
# shares the SAME sentinel (`N + 1`) -- the exact shape of tie IREE does not resolve the
# same way XLA does (measured 260911, see `model.features.top_k`'s docstring). This is
# the actual export blocker for `aminx.export.make_p04_unconditional` (P04): it traced,
# via `build_inference_bundle` -> `generate_ar_mask`, to `autoregression.py`'s tied
# branch, NOT to `inference/decode/autoregressive.py`'s wave-order sort (see also
# `test_wave_order_sort_key_matches_stable_argsort` below, which covers that second,
# separate sort fixed alongside this one for the same reason).
# --------------------------------------------------------------------------------------


def _stable_argsort_reference(group_present: np.ndarray, group_first_occurrence: np.ndarray, n: int):
  key = jnp.where(jnp.asarray(group_present), jnp.asarray(group_first_occurrence), n + 1)
  return jnp.argsort(key, stable=True)


def _tie_free_group_decoding_order(group_present: np.ndarray, group_first_occurrence: np.ndarray, n: int):
  """Reproduces generate_ar_mask's tied-branch formula (autoregression.py, ~L207-215)."""
  key = jnp.where(jnp.asarray(group_present), jnp.asarray(group_first_occurrence), n + 1)
  index = jax.lax.broadcasted_iota(jnp.int32, key.shape, 0)
  _, order = jax.lax.sort((key, index), dimension=0, is_stable=False, num_keys=2)
  return order


@pytest.mark.parametrize("n", [1, 2, 5, 16, 64])
def test_group_decoding_order_tie_free_matches_stable_argsort(n: int):
  """Random (group_present, group_first_occurrence) pairs, many ties (sparse presence
  forces most slots onto the shared N+1 sentinel), several group counts."""
  rng = np.random.default_rng(1234 + n)
  for _ in range(25):
    group_present = rng.random(n) < 0.3
    group_first_occurrence = rng.integers(0, max(n, 1), size=n).astype(np.int32)
    got = _tie_free_group_decoding_order(group_present, group_first_occurrence, n)
    want = _stable_argsort_reference(group_present, group_first_occurrence, n)
    chex.assert_trees_all_equal(got, want)

  # All-absent: every slot ties on the sentinel -- the densest possible tie case.
  group_present = np.zeros(n, dtype=bool)
  group_first_occurrence = rng.integers(0, max(n, 1), size=n).astype(np.int32)
  got = _tie_free_group_decoding_order(group_present, group_first_occurrence, n)
  want = _stable_argsort_reference(group_present, group_first_occurrence, n)
  chex.assert_trees_all_equal(got, want)
