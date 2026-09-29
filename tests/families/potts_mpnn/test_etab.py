# ruff: noqa: S101
"""Synthetic tests for pair merging, alphabet conversion, and Potts energy."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.families.potts_mpnn.etab import (
  ETAB_GAP,
  ETAB_X,
  N_AA,
  N_ETAB,
  etab_to_model,
  merge_pair,
  model_to_etab,
  pad_etab_energy,
  positional_potts_energy,
  potts_energy,
)

pytestmark = pytest.mark.usefixtures("_x64")


@pytest.fixture
def _x64() -> None:
  jax.config.update("jax_enable_x64", True)


def _pair_graph() -> tuple[jax.Array, jax.Array, jax.Array]:
  """Two residues, self in slot 0, each other in slot 1."""
  e_idx = jnp.asarray([[0, 1], [1, 0]])
  pad_valid = jnp.ones((2,), dtype=bool)
  etab = jnp.zeros((2, 2, N_AA, N_AA), dtype=jnp.float64)
  etab = etab.at[0, 1, 0, 1].set(4.0)
  etab = etab.at[1, 1, 1, 0].set(8.0)
  etab = etab.at[0, 0, 0, 1].set(10.0)
  etab = etab.at[0, 0, 1, 0].set(2.0)
  return etab, e_idx, pad_valid


def test_merge_d2_and_d4_hand_values() -> None:
  etab, e_idx, pad_valid = _pair_graph()
  forward = merge_pair(etab, e_idx, pad_valid, denom=2, exclude_self=False)
  energy = merge_pair(etab, e_idx, pad_valid, denom=4, exclude_self=True)

  # Reverse of 0→1 is slot 1 of residue 1: (8 + 4) / 2, and the transpose pair.
  assert float(forward[1, 1, 1, 0]) == pytest.approx(6.0)
  assert float(forward[0, 1, 0, 1]) == pytest.approx(6.0)
  assert float(energy[1, 1, 1, 0]) == pytest.approx(3.0)
  assert float(energy[0, 1, 0, 1]) == pytest.approx(3.0)
  # denom 2 symmetrizes the self slot; denom 4 leaves it undivided.
  assert float(forward[0, 0, 0, 1]) == pytest.approx(6.0)
  assert float(forward[0, 0, 1, 0]) == pytest.approx(6.0)
  assert float(energy[0, 0, 0, 1]) == pytest.approx(10.0)
  assert float(energy[0, 0, 1, 0]) == pytest.approx(2.0)
  assert not jnp.array_equal(forward, energy)


def test_missing_reverse_stays_undivided() -> None:
  etab, _, pad_valid = _pair_graph()
  # Residue 1 never points back at residue 0.
  e_idx = jnp.asarray([[0, 1], [1, 1]])
  merged = merge_pair(etab, e_idx, pad_valid, denom=2, exclude_self=False)
  assert float(merged[0, 1, 0, 1]) == pytest.approx(4.0)
  assert float(etab[0, 1, 0, 1]) == pytest.approx(4.0)


def test_pad_endpoint_is_not_merged() -> None:
  etab, e_idx, _ = _pair_graph()
  pad_valid = jnp.asarray([True, False])
  merged = merge_pair(etab, e_idx, pad_valid, denom=2, exclude_self=False)
  assert float(merged[0, 1, 0, 1]) == pytest.approx(4.0)
  assert float(merged[1, 1, 1, 0]) == pytest.approx(8.0)
  # The real residue's self edge still symmetrizes.
  assert float(merged[0, 0, 0, 1]) == pytest.approx(6.0)


def test_alphabet_round_trip_and_padded_layout() -> None:
  seq = jnp.arange(21, dtype=jnp.int64)
  assert int(model_to_etab(jnp.asarray(20, dtype=jnp.int64))) == ETAB_X
  assert ETAB_GAP == 20
  assert ETAB_X == 21
  assert model_to_etab(seq).dtype == seq.dtype
  assert jnp.array_equal(etab_to_model(model_to_etab(seq)), seq)
  assert int(etab_to_model(jnp.asarray(ETAB_GAP))) == -1

  dense = jnp.arange(2 * N_AA * N_AA, dtype=jnp.float64).reshape(1, 2, N_AA, N_AA)
  padded = pad_etab_energy(dense)
  assert padded.shape == (1, 2, N_ETAB, N_ETAB)
  assert padded.dtype == jnp.float64
  assert jnp.array_equal(padded[..., :N_AA, :N_AA], dense)
  assert float(jnp.max(jnp.abs(padded[..., N_AA:, :]))) == 0.0
  assert float(jnp.max(jnp.abs(padded[..., :, N_AA:]))) == 0.0


def test_three_residue_energy_and_padding() -> None:
  length = 3
  k = 3
  etab = jnp.zeros((length, k, N_ETAB, N_ETAB), dtype=jnp.float64)
  # seq = [0, 1, 2]. Self diagonals plus two directed pair terms.
  etab = etab.at[0, 0, 0, 0].set(1.0)
  etab = etab.at[0, 1, 0, 1].set(3.0)
  etab = etab.at[0, 2, 0, 2].set(5.0)
  etab = etab.at[1, 0, 1, 1].set(2.0)
  etab = etab.at[1, 1, 1, 0].set(7.0)
  etab = etab.at[2, 0, 2, 2].set(4.0)
  e_idx = jnp.asarray([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
  seq = jnp.asarray([0, 1, 2])
  pad_valid = jnp.ones((length,), dtype=bool)

  energy = potts_energy(etab, e_idx, pad_valid, seq)
  assert energy.dtype == jnp.float64
  assert float(energy) == pytest.approx(1.0 + 3.0 + 5.0 + 2.0 + 7.0 + 4.0)

  # Dropping the last residue removes every term that touches it: 5 and 4.
  masked = potts_energy(etab, e_idx, jnp.asarray([True, True, False]), seq)
  assert float(masked) == pytest.approx(1.0 + 3.0 + 2.0 + 7.0)

  at_zero = positional_potts_energy(etab, e_idx, pad_valid, seq, 0)
  assert float(at_zero[0]) == pytest.approx(1.0 + 3.0 + 5.0)
  assert float(jnp.max(jnp.abs(at_zero[1:]))) == 0.0
  hidden = positional_potts_energy(etab, e_idx, jnp.asarray([True, True, False]), seq, 2)
  assert float(jnp.max(jnp.abs(hidden))) == 0.0

  # Gap and X sit in the zero border, so they add nothing on top of a real pair.
  gap_seq = seq.at[2].set(ETAB_GAP)
  unknown_seq = seq.at[2].set(ETAB_X)
  assert float(potts_energy(etab, e_idx, pad_valid, gap_seq)) == pytest.approx(
    float(potts_energy(etab, e_idx, pad_valid, unknown_seq)),
  )
  assert float(potts_energy(etab, e_idx, pad_valid, gap_seq)) == pytest.approx(1.0 + 3.0 + 2.0 + 7.0)

  for pad_to in (length, 128, 512):
    wide = jnp.ones((pad_to, k, N_ETAB, N_ETAB), dtype=jnp.float64)
    wide = wide.at[:length].set(etab)
    wide_idx = jnp.zeros((pad_to, k), dtype=jnp.int32)
    wide_idx = wide_idx.at[:length].set(e_idx)
    wide_seq = jnp.zeros((pad_to,), dtype=jnp.int32).at[:length].set(seq)
    wide_pad = jnp.arange(pad_to) < length
    padded_energy = potts_energy(wide, wide_idx, wide_pad, wide_seq)
    assert padded_energy.dtype == jnp.float64
    assert float(padded_energy) == float(energy)


def test_dtype_and_jit() -> None:
  e_idx = jnp.asarray([[0, 1], [1, 0]])
  pad_valid = jnp.ones((2,), dtype=bool)
  seq = jnp.asarray([0, 1])
  for dtype in (jnp.float32, jnp.float64):
    etab = jnp.zeros((2, 2, N_AA, N_AA), dtype=dtype).at[0, 1, 0, 1].set(dtype(4))
    etab = etab.at[1, 1, 1, 0].set(dtype(8))
    merged = merge_pair(etab, e_idx, pad_valid, denom=2, exclude_self=False)
    assert merged.dtype == dtype
    scored = potts_energy(pad_etab_energy(etab[None])[0], e_idx, pad_valid, seq)
    assert scored.dtype == dtype

  etab64 = jnp.zeros((2, 2, N_AA, N_AA)).at[0, 1, 0, 1].set(4.0).at[1, 1, 1, 0].set(8.0)
  eager = merge_pair(etab64, e_idx, pad_valid, denom=4, exclude_self=True)
  jitted = jax.jit(
    lambda table, idx, valid: merge_pair(table, idx, valid, denom=4, exclude_self=True),
  )(etab64, e_idx, pad_valid)
  assert jnp.array_equal(jitted, eager)

  energy_eager = potts_energy(pad_etab_energy(etab64), e_idx, pad_valid, seq)
  energy_jit = eqx.filter_jit(potts_energy)(
    pad_etab_energy(etab64),
    e_idx,
    pad_valid,
    seq,
  )
  assert energy_jit.dtype == energy_eager.dtype
  assert float(energy_jit) == float(energy_eager)
