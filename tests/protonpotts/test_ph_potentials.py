"""Tests for the ProtonPottsMPNN pH-design conditional and block energies.

Every check is a finite-difference identity against ``potts_energy``, the reference
Hamiltonian. Graphs are random with one-way edges, so an omitted incoming term fails.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_potentials import (
  block_stability_potentials,
  candidate_energies,
  candidate_energies_at,
)

V = 30
TOL = 1e-4


def _random_graph(rng: np.random.Generator, length: int, k_slots: int, vocab: int):
  table = (0.3 * rng.normal(size=(length, k_slots, vocab, vocab))).astype(np.float32)
  table[:, 0] = table[:, 0] * np.eye(vocab, dtype=np.float32)
  e_idx = np.zeros((length, k_slots), dtype=np.int32)
  for i in range(length):
    others = np.delete(np.arange(length), i)
    e_idx[i, 0] = i
    e_idx[i, 1:] = rng.choice(others, size=k_slots - 1, replace=False)
  return table, e_idx


def _with_rows(rng, e_idx: np.ndarray, forced: dict[int, list[int]]) -> np.ndarray:
  """Overwrite the neighbour rows (slots k >= 1) of the listed positions."""
  out = e_idx.copy()
  for i, row in forced.items():
    assert len(row) == out.shape[1] - 1
    assert i not in row and len(set(row)) == len(row)
    out[i, 1:] = row
  return out


def _energy(table, e_idx, seq) -> float:
  pad_valid = jnp.ones(e_idx.shape[0], dtype=bool)
  return float(potts_energy(jnp.asarray(table), jnp.asarray(e_idx), pad_valid, jnp.asarray(seq)))


def _n_one_way(e_idx: np.ndarray) -> int:
  length = e_idx.shape[0]
  count = 0
  for i in range(length):
    for j in e_idx[i, 1:]:
      if i not in e_idx[j, 1:]:
        count += 1
  return count


@pytest.mark.parametrize(("length", "k_slots", "vocab", "seed"), [(8, 4, V, 0), (14, 6, V, 1), (10, 5, 6, 2)])
def test_candidate_energies_match_finite_differences(length, k_slots, vocab, seed):
  rng = np.random.default_rng(seed)
  table, e_idx = _random_graph(rng, length, k_slots, vocab)
  assert _n_one_way(e_idx) > 0
  seq = rng.integers(0, vocab, size=length)

  cand = np.asarray(candidate_energies(jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq)))
  assert cand.shape == (length, vocab)
  assert np.all(np.isfinite(cand))

  for i in range(length):
    for _ in range(3):
      a, b = rng.choice(vocab, size=2, replace=False)
      seq_a, seq_b = seq.copy(), seq.copy()
      seq_a[i], seq_b[i] = a, b
      dh = _energy(table, e_idx, seq_a) - _energy(table, e_idx, seq_b)
      assert abs(dh - (cand[i, a] - cand[i, b])) < TOL


def test_candidate_energies_at_matches_full_rows():
  rng = np.random.default_rng(3)
  length, k_slots = 12, 5
  table, e_idx = _random_graph(rng, length, k_slots, V)
  seq = rng.integers(0, V, size=length)
  full = np.asarray(candidate_energies(jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq)))

  positions = np.array([0, 5, length - 1, 5, 9])
  rows = np.asarray(
    candidate_energies_at(jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq), jnp.asarray(positions))
  )
  assert rows.shape == (len(positions), V)
  np.testing.assert_allclose(rows, full[positions], atol=TOL, rtol=TOL)


def _block_J(unary: np.ndarray, pair: np.ndarray, block_valid: np.ndarray, x: tuple[int, ...]) -> float:
  total = 0.0
  n = len(x)
  for b in range(n):
    if block_valid[b]:
      total += unary[b, x[b]]
  for bi in range(n):
    for bj in range(bi + 1, n):
      if block_valid[bi] and block_valid[bj]:
        total += pair[bi, bj, x[bi], x[bj]]
  return total


def _check_block_identity(table, e_idx, seq, block, block_valid, rng, sub_size=3):
  block = np.asarray(block, dtype=np.int32)
  block_valid = np.asarray(block_valid, dtype=bool)
  unary, pair = block_stability_potentials(
    jnp.asarray(table),
    jnp.asarray(e_idx),
    jnp.asarray(seq),
    jnp.asarray(block),
    jnp.asarray(block_valid),
  )
  unary = np.asarray(unary)
  pair = np.asarray(pair)
  n = len(block)
  assert unary.shape == (n, V)
  assert pair.shape == (n, n, V, V)
  # Only bi < bj is filled, and padded members carry nothing.
  for bi in range(n):
    for bj in range(bi + 1):
      assert np.all(pair[bi, bj] == 0)
    if not block_valid[bi]:
      assert np.all(unary[bi] == 0)

  valid_idx = [b for b in range(n) if block_valid[b]]
  subs = [rng.choice(V, size=sub_size, replace=False) for _ in valid_idx]
  base = None
  for combo in itertools.product(*subs):
    x = [int(rng.integers(0, V)) for _ in range(n)]  # padded entries get an arbitrary token
    for slot, b in enumerate(valid_idx):
      x[b] = int(combo[slot])
    seq_x = seq.copy()
    for b in valid_idx:
      seq_x[block[b]] = x[b]
    h = _energy(table, e_idx, seq_x)
    j = _block_J(unary, pair, block_valid, tuple(x))
    if base is None:
      base = (h, j)
    dh = h - base[0]
    dj = j - base[1]
    assert abs(dh - dj) < TOL, (block.tolist(), x, dh, dj)


def _block_case(rng, length, k_slots, forced, block):
  table, e_idx = _random_graph(rng, length, k_slots, V)
  e_idx = _with_rows(rng, e_idx, forced)
  seq = rng.integers(0, V, size=length).astype(np.int32)
  return table, e_idx, seq, block


def test_block_size_one_identity():
  rng = np.random.default_rng(10)
  table, e_idx, seq, block = _block_case(rng, 10, 5, {}, [3])
  _check_block_identity(table, e_idx, seq, block, [True], rng)


def test_block_pair_one_way_neighbours():
  rng = np.random.default_rng(11)
  forced = {2: [5, 0, 1, 7], 5: [0, 1, 8, 9]}  # 2 -> 5 only
  table, e_idx, seq, block = _block_case(rng, 10, 5, forced, [2, 5])
  assert 2 not in e_idx[5, 1:]
  _check_block_identity(table, e_idx, seq, block, [True, True], rng)


def test_block_pair_both_directions():
  rng = np.random.default_rng(12)
  forced = {1: [6, 0, 2, 3], 6: [1, 4, 7, 8]}  # 1 <-> 6
  table, e_idx, seq, block = _block_case(rng, 10, 5, forced, [1, 6])
  assert 6 in e_idx[1, 1:] and 1 in e_idx[6, 1:]
  _check_block_identity(table, e_idx, seq, block, [True, True], rng)


def test_block_pair_not_neighbours():
  rng = np.random.default_rng(13)
  forced = {0: [1, 2, 3, 4], 9: [5, 6, 7, 8]}  # no edge between 0 and 9
  table, e_idx, seq, block = _block_case(rng, 10, 5, forced, [0, 9])
  assert 9 not in e_idx[0, 1:] and 0 not in e_idx[9, 1:]
  _check_block_identity(table, e_idx, seq, block, [True, True], rng)


def test_block_triple_mixed_edges():
  rng = np.random.default_rng(14)
  # 0 -> 4 one way; 4 <-> 7 both ways; 0 and 7 not neighbours.
  forced = {0: [4, 1, 2, 3], 4: [7, 5, 6, 8], 7: [4, 2, 3, 9]}
  table, e_idx, seq, block = _block_case(rng, 10, 5, forced, [0, 4, 7])
  assert 7 not in e_idx[0, 1:] and 0 not in e_idx[7, 1:]
  _check_block_identity(table, e_idx, seq, block, [True, True, True], rng)


def test_padded_block_matches_shorter_block():
  rng = np.random.default_rng(15)
  forced = {1: [6, 0, 2, 3], 6: [1, 4, 7, 8]}
  table, e_idx, seq, _ = _block_case(rng, 10, 5, forced, [1, 6])
  block_short = jnp.asarray([1, 6], dtype=jnp.int32)
  valid_short = jnp.asarray([True, True])
  u_short, p_short = block_stability_potentials(
    jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq), block_short, valid_short
  )
  block_pad = jnp.asarray([1, 6, 4], dtype=jnp.int32)  # arbitrary, in-range padding position
  valid_pad = jnp.asarray([True, True, False])
  u_pad, p_pad = block_stability_potentials(
    jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq), block_pad, valid_pad
  )
  np.testing.assert_allclose(np.asarray(u_pad)[:2], np.asarray(u_short), atol=TOL, rtol=TOL)
  np.testing.assert_allclose(np.asarray(p_pad)[:2, :2], np.asarray(p_short), atol=TOL, rtol=TOL)
  assert np.all(np.asarray(u_pad)[2] == 0)
  assert np.all(np.asarray(p_pad)[:, 2] == 0)
  _check_block_identity(table, e_idx, seq, [1, 6, 4], [True, True, False], rng)


def test_jit_matches_eager():
  rng = np.random.default_rng(16)
  forced = {0: [4, 1, 2, 3], 4: [7, 5, 6, 8], 7: [4, 2, 3, 9]}
  table, e_idx, seq, block = _block_case(rng, 10, 5, forced, [0, 4, 7])
  t, e, s = jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq)
  b = jnp.asarray(block, dtype=jnp.int32)
  bv = jnp.ones(3, dtype=bool)

  eager_c = candidate_energies(t, e, s)
  jit_c = jax.jit(candidate_energies)(t, e, s)
  np.testing.assert_allclose(np.asarray(jit_c), np.asarray(eager_c), atol=TOL, rtol=TOL)

  eager_u, eager_p = block_stability_potentials(t, e, s, b, bv)
  jit_u, jit_p = jax.jit(block_stability_potentials)(t, e, s, b, bv)
  np.testing.assert_allclose(np.asarray(jit_u), np.asarray(eager_u), atol=TOL, rtol=TOL)
  np.testing.assert_allclose(np.asarray(jit_p), np.asarray(eager_p), atol=TOL, rtol=TOL)

  at = jax.jit(candidate_energies_at)(t, e, s, jnp.asarray([0, 7], dtype=jnp.int32))
  np.testing.assert_allclose(np.asarray(at), np.asarray(eager_c)[[0, 7]], atol=TOL, rtol=TOL)
