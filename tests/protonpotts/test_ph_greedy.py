"""Tests for the centre-free greedy energy block descent (ph_greedy).

Oracles are independent of the code under test. Energies come from ``potts_energy`` on whole sequences, the
single-mutation z-scale is enumerated from full-energy differences, and the temperature-0 trajectory is replayed in
numpy from the same full energies. Nothing here re-derives the edge formulas used by the module.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
from aminx.families.protonpotts_mpnn.ph_descent import rep_class_mask
from aminx.families.protonpotts_mpnn.ph_greedy import greedy_energy_block
from aminx.families.protonpotts_mpnn.ph_plan import valid_token_mask
from aminx.families.protonpotts_mpnn.vocab import canonical_letter, token_index

V = 30
LENGTH = 14
K_SLOTS = 5


class _Problem(NamedTuple):
  table: np.ndarray
  e_idx: np.ndarray
  seq0: np.ndarray
  designable: np.ndarray
  valid: np.ndarray
  rep_mask: np.ndarray
  residue: np.ndarray
  config: PHDesignConfig


def _edges(rng: np.random.Generator) -> np.ndarray:
  e_idx = np.zeros((LENGTH, K_SLOTS), dtype=np.int32)
  for i in range(LENGTH):
    others = np.delete(np.arange(LENGTH), i)
    e_idx[i, 0] = i
    e_idx[i, 1:] = rng.choice(others, size=K_SLOTS - 1, replace=False)  # one-way: rarely reciprocal
  return e_idx


def _problem(
  seed: int,
  *,
  block_size: int = 3,
  temperature: float = 0.0,
  cv_max: int = 50,
  n_design: int = LENGTH,
  rep_weight: float = 0.0,
) -> _Problem:
  rng = np.random.default_rng(seed)
  table = (0.3 * rng.normal(size=(LENGTH, K_SLOTS, V, V))).astype(np.float32)
  table[:, 0] = table[:, 0] * np.eye(V, dtype=np.float32)  # single-site field only on the diagonal
  e_idx = _edges(rng)
  config = PHDesignConfig(
    binder_chain="A",
    block_size=block_size,
    temperature=temperature,
    cv_max=cv_max,
    repetitive_window_weight=rep_weight,
  )
  valid = valid_token_mask(config.forbidden_tokens)
  valid_idx = np.flatnonzero(valid)
  designable = np.sort(rng.choice(LENGTH, size=n_design, replace=False)).astype(np.int64)
  seq0 = rng.integers(0, V, size=LENGTH).astype(np.int32)
  seq0[designable] = rng.choice(valid_idx, size=designable.size)  # valid starting tokens
  return _Problem(
    table=table,
    e_idx=e_idx,
    seq0=seq0,
    designable=designable,
    valid=valid,
    rep_mask=rep_class_mask(config.repetitive_window_parents),
    residue=(np.arange(LENGTH, dtype=np.int32) + 1),
    config=config,
  )


def _run(p: _Problem, *, seq=None, designable=None, config=None, uniforms=None):
  cfg = p.config if config is None else config
  des = p.designable if designable is None else np.asarray(designable)
  if uniforms is None:
    uniforms = np.zeros(max(1, cfg.cv_max * des.size), dtype=np.float32)
  return greedy_energy_block(
    jnp.asarray(p.table),
    jnp.asarray(p.e_idx),
    jnp.asarray(p.seq0 if seq is None else seq),
    jnp.asarray(des, dtype=jnp.int32),
    jnp.asarray(p.valid),
    jnp.asarray(p.rep_mask),
    jnp.asarray(p.residue),
    config=cfg,
    uniforms=jnp.asarray(uniforms, dtype=jnp.float32),
  )


def _energies(p: _Problem, seqs: np.ndarray) -> np.ndarray:
  """Full Potts energy of each row of ``seqs`` (independent oracle)."""
  out = potts_energy(
    jnp.asarray(p.table),
    jnp.asarray(p.e_idx),
    jnp.ones(LENGTH, dtype=bool),
    jnp.asarray(np.asarray(seqs, dtype=np.int32)),
  )
  return np.asarray(out, dtype=np.float64)


def _h(p: _Problem, seq: np.ndarray) -> float:
  return float(_energies(p, np.asarray(seq)[None])[0])


def _single_mutants(p: _Problem, seq: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Energies (len(positions), V): row r is the full energy with positions[r] set to each token."""
  batch = np.repeat(np.asarray(seq)[None, None], positions.size * V, axis=1)[0]
  for r, i in enumerate(positions):
    rows = slice(r * V, (r + 1) * V)
    batch[rows, i] = np.arange(V)
  return _energies(p, batch).reshape(positions.size, V)


def _canonical(seq: np.ndarray, designable, valid: np.ndarray) -> np.ndarray:
  out = np.asarray(seq).copy()
  for i in designable:
    c = token_index(canonical_letter(int(out[i])))
    if valid[c]:
      out[i] = c
  return out


def _replay_t0(p: _Problem) -> np.ndarray:
  """Numpy replay of upstream ``_greedy_energy_block`` at temperature 0, rep term off, energies by potts_energy."""
  cfg = p.config
  des = p.designable
  valid_idx = np.flatnonzero(p.valid)
  n = des.size
  seq = _canonical(p.seq0, des, p.valid)
  best, best_h = seq.copy(), _h(p, seq)
  since = step = 0
  patience, cap = cfg.cv_patience * n, cfg.cv_max * n
  while since < patience and step < cap:
    mut = _single_mutants(p, seq, des)  # (N, V)
    delta = np.array([mut[r, valid_idx].min() - mut[r, seq[i]] for r, i in enumerate(des)])
    improving = [(delta[r], int(i)) for r, i in enumerate(des) if delta[r] < -1e-4]
    if not improving:
      break
    improving.sort(key=lambda t: (t[0], t[1]))
    block = [i for _, i in improving[: cfg.block_size]]
    combos = np.array(list(itertools.product(valid_idx, repeat=len(block))))
    batch = np.repeat(seq[None], combos.shape[0], axis=0)
    batch[:, block] = combos
    seq = batch[int(np.argmin(_energies(p, batch)))].copy()  # first minimum, lexicographic over valid tokens
    step += 1
    h = _h(p, seq)
    if h < best_h - 1e-9:
      best, best_h, since = seq.copy(), h, 0
    else:
      since += 1
  return best


def test_zscale_matches_full_energy_enumeration() -> None:
  p = _problem(1)
  res = _run(p)
  valid_idx = np.flatnonzero(p.valid)
  deltas = []
  for i in p.designable:
    mut = _single_mutants(p, p.seq0, np.array([i]))[0]
    deltas.append(mut[valid_idx] - mut[p.seq0[i]])
  vals = np.concatenate(deltas)
  expected = max(float(vals.std()), 1e-6)  # population std, floored
  assert np.isfinite(vals).all()
  assert float(res.zscale) == pytest.approx(expected, rel=1e-4)


def test_t0_energy_not_above_start_and_invalid_free_with_fixed_outside() -> None:
  for seed in range(3):
    p = _problem(seed, n_design=6)
    res = _run(p)
    seq = np.asarray(res.seq)
    start = _canonical(p.seq0, p.designable, p.valid)
    assert _h(p, seq) <= _h(p, start) + 1e-5
    assert p.valid[seq[p.designable]].all()
    outside = np.setdiff1d(np.arange(LENGTH), p.designable)
    np.testing.assert_array_equal(seq[outside], p.seq0[outside])


def test_t0_rerun_from_result_does_not_get_worse() -> None:
  p = _problem(2)
  first = _run(p)
  second = _run(p, seq=np.asarray(first.seq))
  assert _h(p, np.asarray(second.seq)) <= _h(p, np.asarray(first.seq)) + 1e-5
  if int(second.steps) == 0:
    np.testing.assert_array_equal(np.asarray(second.seq), np.asarray(first.seq))


def test_t0_matches_numpy_replay_of_upstream_loop() -> None:
  for seed in (0, 1, 2):
    p = _problem(seed)
    res = _run(p)
    np.testing.assert_array_equal(np.asarray(res.seq), _replay_t0(p))


def test_canonicalisation_turns_protonation_token_into_parent_before_improving() -> None:
  # Zero table except a self field at position 5. HIS-A is invalid and its field is very low, so the bare
  # residue H (field 3.0) must be the current token for the move to A (field -1.0) to be an improvement.
  rng = np.random.default_rng(7)
  table = np.zeros((LENGTH, K_SLOTS, V, V), dtype=np.float32)
  e_idx = _edges(rng)
  pos = 5
  field = np.full(V, 1.0, dtype=np.float32)
  field[token_index("H")] = 3.0
  field[token_index("A")] = -1.0
  field[token_index("HIS-A")] = -100.0
  table[pos, 0] = np.diag(field)
  config = PHDesignConfig(binder_chain="A", block_size=3, temperature=0.0, repetitive_window_weight=0.0)
  valid = valid_token_mask(config.forbidden_tokens)
  seq0 = np.full(LENGTH, token_index("G"), dtype=np.int32)
  seq0[pos] = token_index("HIS-A")
  p = _Problem(
    table=table,
    e_idx=e_idx,
    seq0=seq0,
    designable=np.arange(LENGTH, dtype=np.int64),
    valid=valid,
    rep_mask=rep_class_mask(config.repetitive_window_parents),
    residue=np.arange(LENGTH, dtype=np.int32) + 1,
    config=config,
  )
  res = _run(p)
  assert int(res.seq[pos]) == token_index("A")
  assert int(res.steps) == 1
  assert _h(p, np.asarray(res.seq)) < _h(p, _canonical(seq0, p.designable, valid))


def test_sampling_same_stream_same_result_and_draws_count_steps() -> None:
  p = _problem(3, temperature=0.4)
  uniforms = np.random.default_rng(11).uniform(size=p.designable.size * p.config.cv_max).astype(np.float32)
  first = _run(p, uniforms=uniforms)
  second = _run(p, uniforms=uniforms)
  np.testing.assert_array_equal(np.asarray(first.seq), np.asarray(second.seq))
  assert int(first.n_draws) == int(first.steps)
  assert int(first.steps) > 0


def test_sampling_different_streams_differ() -> None:
  p = _problem(4, temperature=0.8)
  n_unif = p.designable.size * p.config.cv_max
  seqs = set()
  for trial in range(6):
    u = np.random.default_rng(100 + trial).uniform(size=n_unif).astype(np.float32)
    seqs.add(tuple(np.asarray(_run(p, uniforms=u).seq).tolist()))
  assert len(seqs) >= 2


def test_block_shorter_than_block_size_matches_bruteforce_over_reduced_tokens() -> None:
  # Only positions 2 and 9 can improve (self fields and one pair edge 2 -> 9). block_size 3 gives a block of two
  # real members and one padded slot. The valid set is reduced to four tokens, so the joint is brute-forceable.
  rng = np.random.default_rng(21)
  reduced = np.array([token_index(t) for t in ("A", "C", "D", "E")])
  table = np.zeros((LENGTH, K_SLOTS, V, V), dtype=np.float32)
  e_idx = _edges(rng)
  e_idx[2, 1] = 9
  for a in reduced:
    table[2, 0, a, a] = rng.normal()
    table[9, 0, a, a] = rng.normal()
    for b in reduced:
      table[2, 1, a, b] = rng.normal()
  config = PHDesignConfig(binder_chain="A", block_size=3, temperature=0.0, repetitive_window_weight=0.0)
  valid = np.zeros(V, dtype=bool)
  valid[reduced] = True
  seq0 = np.full(LENGTH, reduced[0], dtype=np.int32)
  seq0[2], seq0[9] = reduced[1], reduced[3]
  p = _Problem(
    table=table,
    e_idx=e_idx,
    seq0=seq0,
    designable=np.arange(LENGTH, dtype=np.int64),
    valid=valid,
    rep_mask=rep_class_mask(config.repetitive_window_parents),
    residue=np.arange(LENGTH, dtype=np.int32) + 1,
    config=config,
  )
  res = _run(p)
  combos = np.array(list(itertools.product(reduced, repeat=2)))
  batch = np.repeat(seq0[None], combos.shape[0], axis=0)
  batch[:, 2], batch[:, 9] = combos[:, 0], combos[:, 1]
  best = batch[int(np.argmin(_energies(p, batch)))]
  np.testing.assert_array_equal(np.asarray(res.seq)[[2, 9]], best[[2, 9]])
  assert int(res.steps) <= 1  # one joint move reaches the block optimum, then nothing improves


def test_refusals_and_degenerate_inputs() -> None:
  p = _problem(5)
  with pytest.raises(NotImplementedError):
    _run(p, config=PHDesignConfig(binder_chain="A", self_weight=0.5))
  sampled = _problem(5, temperature=0.5)
  with pytest.raises(ValueError):
    _run(sampled, uniforms=np.zeros(3, dtype=np.float32))
  # combined_lambda and zscale_mode are ignored by greedy, not refused.
  ignored = PHDesignConfig(binder_chain="A", zscale_mode="knn", combined_lambda=0.9)
  assert int(_run(p, config=ignored).steps) >= 0
  # No designable positions: untouched sequence, unit z-scale, no steps.
  empty = _run(p, designable=np.zeros(0, dtype=np.int64))
  np.testing.assert_array_equal(np.asarray(empty.seq), p.seq0)
  assert float(empty.zscale) == 1.0
  assert int(empty.steps) == 0


def test_patience_and_cap_bound_the_number_of_steps() -> None:
  p = _problem(6, cv_max=1)
  res = _run(p)
  assert int(res.steps) <= LENGTH


def test_jit_compiles_once_for_two_uniform_streams() -> None:
  p = _problem(8, temperature=0.5)
  n_unif = p.designable.size * p.config.cv_max
  traces: list[int] = []

  def run(uniforms: jax.Array):
    traces.append(1)
    return greedy_energy_block(
      jnp.asarray(p.table),
      jnp.asarray(p.e_idx),
      jnp.asarray(p.seq0),
      jnp.asarray(p.designable, dtype=jnp.int32),
      jnp.asarray(p.valid),
      jnp.asarray(p.rep_mask),
      jnp.asarray(p.residue),
      config=p.config,
      uniforms=uniforms,
    )

  jitted = jax.jit(run)
  u1 = jnp.asarray(np.random.default_rng(1).uniform(size=n_unif), dtype=jnp.float32)
  u2 = jnp.asarray(np.random.default_rng(2).uniform(size=n_unif), dtype=jnp.float32)
  r1 = jitted(u1)
  r2 = jitted(u2)
  assert len(traces) == 1
  assert int(r1.steps) > 0 and int(r2.steps) > 0
