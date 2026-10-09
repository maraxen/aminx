"""Tests for the ProtonPottsMPNN block descent (ph_descent).

Oracles are independent of the code under test: the full Potts energy comes from ``potts_energy``, the selective
gap is recomputed from ``candidate_energies_at`` by definition, and the repetitive term is a direct pair count.
Nothing here re-derives the edge formulas used by the module.
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
from aminx.families.protonpotts_mpnn.ph_descent import (
  block_descent,
  block_objective,
  block_zscales,
  rep_class_mask,
  select_joint,
)
from aminx.families.protonpotts_mpnn.ph_plan import block_table, valid_token_mask
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies_at
from aminx.families.protonpotts_mpnn.vocab import token_index

V = 30
LENGTH = 14
K_SLOTS = 5


class _Problem(NamedTuple):
  table: np.ndarray
  e_idx: np.ndarray
  seq0: np.ndarray
  designable: np.ndarray
  blocks: np.ndarray
  block_valid: np.ndarray
  pin_pos: np.ndarray
  pin_prot: np.ndarray
  pin_dep: np.ndarray
  pin_dep_valid: np.ndarray
  valid: np.ndarray
  rep_mask: np.ndarray
  residue: np.ndarray
  config: PHDesignConfig


def _problem(
  seed: int,
  *,
  block_size: int = 3,
  temperature: float = 0.0,
  max_rounds: int = 30,
  n_design: int = 8,
) -> _Problem:
  rng = np.random.default_rng(seed)
  table = (0.3 * rng.normal(size=(LENGTH, K_SLOTS, V, V))).astype(np.float32)
  table[:, 0] = table[:, 0] * np.eye(V, dtype=np.float32)  # single-site field only on the diagonal
  e_idx = np.zeros((LENGTH, K_SLOTS), dtype=np.int32)
  for i in range(LENGTH):
    others = np.delete(np.arange(LENGTH), i)
    e_idx[i, 0] = i
    e_idx[i, 1:] = rng.choice(others, size=K_SLOTS - 1, replace=False)

  config = PHDesignConfig(
    binder_chain="A",
    block_size=block_size,
    temperature=temperature,
    block_max_rounds=max_rounds,
  )
  valid = valid_token_mask(config.forbidden_tokens)
  valid_idx = np.flatnonzero(valid)

  perm = rng.permutation(LENGTH)
  pin_pos = perm[:2].astype(np.int32)
  designable = np.sort(perm[2 : 2 + n_design]).astype(np.int64)

  pin_prot = np.array([token_index("HIS-P"), token_index("ASP-P")], dtype=np.int32)
  # Pin 0 has two contrast tokens; pin 1 has one, padded to the same width.
  pin_dep = np.array(
    [[token_index("HIS-S"), token_index("HIS-A")], [token_index("ASP-D"), 0]],
    dtype=np.int32,
  )
  pin_dep_valid = np.array([[True, True], [True, False]])

  seq0 = rng.integers(0, V, size=LENGTH).astype(np.int32)
  seq0[designable] = rng.choice(valid_idx, size=designable.size)  # valid start: keeps J(cur) finite
  seq0[pin_pos] = pin_prot

  blocks, block_valid = block_table(e_idx, designable, block_size)
  rep_mask = rep_class_mask(config.repetitive_window_parents)
  residue = (np.arange(LENGTH, dtype=np.int32) + 1).astype(np.int32)
  return _Problem(
    table=table,
    e_idx=e_idx,
    seq0=seq0,
    designable=designable,
    blocks=blocks,
    block_valid=block_valid,
    pin_pos=pin_pos,
    pin_prot=pin_prot,
    pin_dep=pin_dep,
    pin_dep_valid=pin_dep_valid,
    valid=valid,
    rep_mask=rep_mask,
    residue=residue,
    config=config,
  )


def _run(p: _Problem, *, config: PHDesignConfig | None = None, uniforms=None, blocks=None, block_valid=None):
  cfg = p.config if config is None else config
  n_rows = p.designable.size
  if uniforms is None:
    uniforms = np.zeros(max(1, cfg.block_max_rounds * n_rows), dtype=np.float32)
  return block_descent(
    jnp.asarray(p.table),
    jnp.asarray(p.e_idx),
    jnp.asarray(p.seq0),
    jnp.asarray(p.designable),
    jnp.asarray(p.blocks if blocks is None else blocks),
    jnp.asarray(p.block_valid if block_valid is None else block_valid),
    jnp.asarray(p.pin_pos),
    jnp.asarray(p.pin_prot),
    jnp.asarray(p.pin_dep),
    jnp.asarray(p.pin_dep_valid),
    jnp.asarray(p.valid),
    jnp.asarray(p.rep_mask),
    jnp.asarray(p.residue),
    config=cfg,
    uniforms=jnp.asarray(uniforms, dtype=jnp.float32),
  )


def _energy_batch(p: _Problem, seqs: np.ndarray) -> np.ndarray:
  pad_valid = jnp.ones(LENGTH, dtype=bool)
  return np.asarray(potts_energy(jnp.asarray(p.table), jnp.asarray(p.e_idx), pad_valid, jnp.asarray(seqs)))


def _selective(p: _Problem, seq: np.ndarray) -> float:
  """sum over pinned centres of (e(protonated) - mean over valid contrast tokens e(d)), from candidate energies."""
  cand = np.asarray(
    candidate_energies_at(jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(seq), jnp.asarray(p.pin_pos)),
  )
  total = 0.0
  for c in range(p.pin_pos.size):
    deps = p.pin_dep[c][p.pin_dep_valid[c]]
    total += cand[c, p.pin_prot[c]] - cand[c, deps].mean()
  return float(total)


def _repetitive(p: _Problem, seq: np.ndarray, radius: int) -> float:
  """Unordered position pairs with 0 < |residue distance| <= radius, both in the class: the rw-free count."""
  total = 0.0
  for i in range(LENGTH):
    for j in range(i + 1, LENGTH):
      if 0 < abs(int(p.residue[i]) - int(p.residue[j])) <= radius:
        total += float(p.rep_mask[seq[i]] * p.rep_mask[seq[j]])
  return total


def _full_objective(p: _Problem, seqs: np.ndarray, wh: float, wsel: float) -> np.ndarray:
  cfg = p.config
  energy = _energy_batch(p, seqs)
  sel = np.array([_selective(p, s) for s in seqs])
  rep = np.array([_repetitive(p, s, cfg.repetitive_window_radius) for s in seqs])
  return wh * energy + wsel * sel + cfg.repetitive_window_weight * rep


def _flat_index(block_valid_row: np.ndarray, digits_of_valid: np.ndarray, vocab: int) -> np.ndarray:
  """Row-major index into the (V,)*B joint tensor, padded axes taking token 0."""
  n_block = block_valid_row.size
  axes = np.flatnonzero(block_valid_row)
  full = np.zeros((digits_of_valid.shape[0], n_block), dtype=np.int64)
  full[:, axes] = digits_of_valid
  weights = vocab ** np.arange(n_block - 1, -1, -1)
  return full @ weights


@pytest.mark.parametrize("seed", [0, 1])
def test_block_objective_differences_match_full_objective(seed):
  p = _problem(seed)
  cfg = p.config
  # Reduced alphabet so the enumeration stays small; includes a rep-class standard letter, HIS-P and ASP-D.
  tokens = np.array([token_index(t) for t in ("K", "A", "HIS-P", "ASP-D", "G")])
  sub_valid = np.zeros(V, dtype=bool)
  sub_valid[tokens] = True
  row = int(np.argmax(p.block_valid.sum(axis=1)))
  axes = np.flatnonzero(p.block_valid[row])
  members = p.blocks[row][axes]

  combos = np.array(list(itertools.product(tokens, repeat=members.size)), dtype=np.int32)
  seqs = np.repeat(p.seq0[None], combos.shape[0], axis=0)
  seqs[:, members] = combos
  wh, wsel = 0.8, 0.35
  full = _full_objective(p, seqs, wh, wsel)

  joint = np.asarray(
    block_objective(
      jnp.asarray(p.table),
      jnp.asarray(p.e_idx),
      jnp.asarray(p.seq0),
      jnp.asarray(p.blocks[row]),
      jnp.asarray(p.block_valid[row]),
      jnp.asarray(p.pin_pos),
      jnp.asarray(p.pin_prot),
      jnp.asarray(p.pin_dep),
      jnp.asarray(p.pin_dep_valid),
      jnp.asarray(sub_valid),
      jnp.asarray(p.rep_mask),
      jnp.asarray(p.residue),
      config=cfg,
      wh=wh,
      wsel=wsel,
    ),
  )
  idx = _flat_index(p.block_valid[row], combos, V)
  assert np.all(np.isfinite(joint[idx]))
  np.testing.assert_allclose(joint[idx] - joint[idx[0]], full - full[0], atol=5e-3)


def test_block_zscales_matches_enumeration():
  p = _problem(3, block_size=2)
  tokens = np.array([token_index(t) for t in ("K", "A", "HIS-P", "ASP-D", "G")])
  sub_valid = np.zeros(V, dtype=bool)
  sub_valid[tokens] = True
  pins_set = p.seq0.copy()

  var_h: list[float] = []
  var_s: list[float] = []
  for row in range(p.designable.size):
    axes = np.flatnonzero(p.block_valid[row])
    members = p.blocks[row][axes]
    combos = np.array(list(itertools.product(tokens, repeat=members.size)), dtype=np.int32)
    seqs = np.repeat(pins_set[None], combos.shape[0], axis=0)
    seqs[:, members] = combos
    energies = _energy_batch(p, seqs)
    if energies.size > 1:
      var_h.append(float(np.var(energies)))

    base = _selective(p, pins_set)
    varies = False
    for member in members:
      for tok in tokens:
        probe = pins_set.copy()
        probe[member] = tok
        if abs(_selective(p, probe) - base) > 1e-9:
          varies = True
    if varies:
      sel_vals = np.array([_selective(p, s) for s in seqs])
      if sel_vals.size > 1:
        var_s.append(float(np.var(sel_vals)))

  expected_h = max(np.sqrt(np.mean(var_h)), 1e-6) if var_h else 1.0
  expected_s = max(np.sqrt(np.mean(var_s)), 1e-6) if var_s else 1.0
  got = np.asarray(
    block_zscales(
      jnp.asarray(p.table),
      jnp.asarray(p.e_idx),
      jnp.asarray(pins_set),
      jnp.asarray(p.blocks),
      jnp.asarray(p.block_valid),
      jnp.asarray(p.pin_pos),
      jnp.asarray(p.pin_prot),
      jnp.asarray(p.pin_dep),
      jnp.asarray(p.pin_dep_valid),
      jnp.asarray(sub_valid),
    ),
  )
  np.testing.assert_allclose(got[0], expected_h, rtol=1e-3)
  np.testing.assert_allclose(got[1], expected_s, rtol=1e-3)
  assert got[2] == pytest.approx(1.0)


def test_block_zscales_is_the_same_under_every_xtrax_strategy(monkeypatch):
  """The block axis goes through xtrax, so the planner may pick one vmap or tiles of it, with padding for a ragged tail."""
  from aminx.families.protonpotts_mpnn import ph_descent  # noqa: PLC0415
  from aminx.tiling.strategy import SafeMap, Vmap  # noqa: PLC0415

  p = _problem(3, block_size=2)
  n_blocks = p.blocks.shape[0]
  assert n_blocks >= 3
  tokens = np.array([token_index(t) for t in ("K", "A", "HIS-P", "ASP-D", "G")])
  sub_valid = np.zeros(V, dtype=bool)
  sub_valid[tokens] = True
  args = (
    jnp.asarray(p.table),
    jnp.asarray(p.e_idx),
    jnp.asarray(p.seq0),
    jnp.asarray(p.blocks),
    jnp.asarray(p.block_valid),
    jnp.asarray(p.pin_pos),
    jnp.asarray(p.pin_prot),
    jnp.asarray(p.pin_dep),
    jnp.asarray(p.pin_dep_valid),
    jnp.asarray(sub_valid),
  )
  ragged_tile = n_blocks - 1  # n % (n - 1) == 1 for n > 2, so the last tile is short and gets padded
  assert n_blocks % ragged_tile != 0

  def zscales_under(strategy):
    monkeypatch.setattr(ph_descent, "plan_axis_strategy", lambda *_a, **_k: strategy)
    return np.asarray(block_zscales(*args))

  planned = np.asarray(block_zscales(*args))  # whatever the planner chooses on this host
  vmapped = zscales_under(Vmap())
  ragged = zscales_under(SafeMap(tile=ragged_tile))
  exact = zscales_under(SafeMap(tile=n_blocks))
  assert not np.allclose(vmapped[:2], 1.0), "degenerate fixture: the z-scales would not discriminate"
  np.testing.assert_allclose(ragged, vmapped, rtol=1e-5, atol=0.0)
  np.testing.assert_allclose(exact, vmapped, rtol=1e-5, atol=0.0)
  np.testing.assert_allclose(planned, vmapped, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("seed", [4, 5])
def test_zero_temperature_descent_invariants(seed):
  p = _problem(seed, temperature=0.0)
  res = _run(p)
  seq_out = np.asarray(res.seq)
  pins = p.pin_pos
  designable = p.designable
  other = np.setdiff1d(np.arange(LENGTH), np.concatenate([designable, pins]))

  assert np.array_equal(seq_out[pins], p.pin_prot)
  assert np.array_equal(seq_out[other], p.seq0[other])
  assert p.valid[seq_out[designable]].all()
  assert int(res.rounds) < p.config.block_max_rounds  # converged, so the fixed-point check below applies

  lam = p.config.combined_lambda
  zs = np.asarray(res.zscales)
  wh, wsel = (1.0 - lam) / zs[0], lam / zs[1]
  before, after = _full_objective(p, np.stack([p.seq0, seq_out]), wh, wsel)
  assert after <= before + 1e-3

  # Fixed point with the run's frozen weights: no block can improve on its current assignment.
  for row in range(designable.size):
    flat = np.asarray(
      block_objective(
        jnp.asarray(p.table),
        jnp.asarray(p.e_idx),
        jnp.asarray(seq_out),
        jnp.asarray(p.blocks[row]),
        jnp.asarray(p.block_valid[row]),
        jnp.asarray(p.pin_pos),
        jnp.asarray(p.pin_prot),
        jnp.asarray(p.pin_dep),
        jnp.asarray(p.pin_dep_valid),
        jnp.asarray(p.valid),
        jnp.asarray(p.rep_mask),
        jnp.asarray(p.residue),
        config=p.config,
        wh=wh,
        wsel=wsel,
      ),
    )
    axes = np.flatnonzero(p.block_valid[row])
    digits = seq_out[p.blocks[row][axes]][None, :]
    current = _flat_index(p.block_valid[row], digits, V)[0]
    assert flat[current] <= flat.min() + 1e-3


@pytest.mark.parametrize("seed", [6, 7])
def test_sampling_is_deterministic_and_counts_draws(seed):
  p = _problem(seed, temperature=0.6, max_rounds=5)
  n_rows = p.designable.size
  uniforms = np.random.default_rng(seed).uniform(size=5 * n_rows).astype(np.float32)
  first = _run(p, uniforms=uniforms)
  second = _run(p, uniforms=uniforms)
  assert np.array_equal(np.asarray(first.seq), np.asarray(second.seq))
  assert int(first.n_draws) == int(first.rounds) * n_rows
  assert int(first.n_draws) > 0

  outcomes = set()
  for trial in range(4):
    stream = np.random.default_rng(100 + trial).uniform(size=5 * n_rows).astype(np.float32)
    outcomes.add(tuple(np.asarray(_run(p, uniforms=stream).seq).tolist()))
  assert len(outcomes) >= 2


def test_zero_temperature_uses_no_draws():
  p = _problem(8, temperature=0.0)
  res = _run(p)
  assert int(res.n_draws) == 0


def test_select_joint_first_minimum_and_inverse_cdf():
  flat = jnp.asarray([np.inf, 0.0, 1.0, np.inf, 2.0], dtype=jnp.float32)
  assert int(select_joint(flat, jnp.float32(0.0), 0.0)) == 1  # argmin, first minimum
  assert int(select_joint(flat, jnp.float32(0.0), 0.5)) == 1  # uniform 0 -> first finite index
  assert int(select_joint(flat, jnp.float32(0.999999), 0.5)) == 4  # last finite index
  ties = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
  assert int(select_joint(ties, jnp.float32(0.0), 0.0)) == 1


def test_short_block_matches_explicit_padding():
  p = _problem(9, block_size=2)
  blocks2 = p.blocks.copy()
  valid2 = p.block_valid.copy()
  valid2[0] = [True, False]
  blocks2[0, 1] = -1
  assert valid2[0].sum() == 1

  blocks3 = np.concatenate([blocks2, -np.ones((blocks2.shape[0], 1), dtype=blocks2.dtype)], axis=1)
  valid3 = np.concatenate([valid2, np.zeros((valid2.shape[0], 1), dtype=bool)], axis=1)

  narrow = _run(p, blocks=blocks2, block_valid=valid2)
  wide = _run(p, blocks=blocks3, block_valid=valid3)
  assert np.array_equal(np.asarray(narrow.seq), np.asarray(wide.seq))
  assert int(narrow.rounds) == int(wide.rounds)
  np.testing.assert_allclose(np.asarray(narrow.zscales), np.asarray(wide.zscales), rtol=1e-6)


@pytest.mark.parametrize(
  ("knob", "value"),
  [
    ("global_weight", 0.1),
    ("adjacent_repeat_weight", 0.2),
    ("self_weight", 0.9),
    ("zscale_mode", "single_mutation"),
    ("sweep_order", "knn"),
  ],
)
def test_unsupported_knobs_are_refused(knob, value):
  p = _problem(10)
  cfg = PHDesignConfig(**{knob: value, "block_size": 3})
  with pytest.raises(NotImplementedError, match=knob):
    _run(p, config=cfg)


def test_rep_class_mask_selects_parent_residue_tokens():
  his = rep_class_mask(("HIS",))
  assert his.shape == (V,)
  assert set(np.flatnonzero(his)) == {token_index(t) for t in ("H", "HIS-P", "HIS-S", "HIS-A")}
  basic = rep_class_mask(("ARG", "LYS"))
  assert basic[token_index("K")] == 1.0
  assert basic[token_index("R")] == 1.0
  assert basic[token_index("X")] == 0.0


def test_jit_compiles_once_for_two_uniform_streams():
  p = _problem(11, temperature=0.4, max_rounds=3)
  n_rows = p.designable.size
  calls = {"n": 0}
  cfg = p.config

  def run(uniforms):
    calls["n"] += 1
    return block_descent(
      jnp.asarray(p.table),
      jnp.asarray(p.e_idx),
      jnp.asarray(p.seq0),
      jnp.asarray(p.designable),
      jnp.asarray(p.blocks),
      jnp.asarray(p.block_valid),
      jnp.asarray(p.pin_pos),
      jnp.asarray(p.pin_prot),
      jnp.asarray(p.pin_dep),
      jnp.asarray(p.pin_dep_valid),
      jnp.asarray(p.valid),
      jnp.asarray(p.rep_mask),
      jnp.asarray(p.residue),
      config=cfg,
      uniforms=uniforms,
    )

  jitted = jax.jit(run)
  u1 = jnp.asarray(np.random.default_rng(1).uniform(size=3 * n_rows), dtype=jnp.float32)
  u2 = jnp.asarray(np.random.default_rng(2).uniform(size=3 * n_rows), dtype=jnp.float32)
  r1 = jitted(u1)
  r2 = jitted(u2)
  assert calls["n"] == 1
  assert int(r1.n_draws) == int(r1.rounds) * n_rows
  assert int(r2.n_draws) == int(r2.rounds) * n_rows


def test_select_joint_cdf_order_identity_and_permutation() -> None:
  """``cdf_order`` changes which entry a uniform selects, exactly as a numpy inverse CDF in that order does."""
  vocab, n_block = 5, 2
  rng = np.random.default_rng(0)
  flat = jnp.asarray(rng.normal(size=vocab**n_block))
  temperature = 0.7
  identity = jnp.arange(vocab)
  perm = np.asarray([3, 0, 4, 1, 2])
  probs = np.exp(-(np.asarray(flat) - float(flat.min())) / temperature)
  for uniform in (0.0, 0.13, 0.5, 0.77, 0.999):
    default = int(select_joint(flat, jnp.asarray(uniform), temperature))
    assert int(select_joint(flat, jnp.asarray(uniform), temperature, n_block=n_block, cdf_order=identity)) == default
    grid = probs.reshape(vocab, vocab)[perm][:, perm]
    cdf = np.cumsum(grid.reshape(-1))
    position = min(int(np.sum(cdf <= uniform * cdf[-1])), cdf.size - 1)
    want = int(perm[position // vocab] * vocab + perm[position % vocab])
    got = int(select_joint(flat, jnp.asarray(uniform), temperature, n_block=n_block, cdf_order=jnp.asarray(perm)))
    assert got == want
