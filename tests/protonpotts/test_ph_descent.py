"""Tests for the ProtonPottsMPNN block descent (ph_descent).

Oracles are independent of the code under test: the full Potts energy comes from ``potts_energy``, the selective
gap is recomputed from ``candidate_energies_at`` by definition, and the repetitive term is a direct pair count.
Nothing here re-derives the edge formulas used by the module.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
from aminx.families.protonpotts_mpnn.ph_descent import (
  GlobalMasks,
  block_descent,
  block_objective,
  block_zscales,
  global_masks,
  rep_class_mask,
  select_joint,
  single_mutation_zscales,
  stability_table,
)
from aminx.families.protonpotts_mpnn.ph_plan import block_table, valid_token_mask
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies_at
from aminx.families.protonpotts_mpnn.vocab import canonical_letter, token_index

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
  from xtrax.tiling import ChunkedMap, Vmap  # noqa: PLC0415

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
  ragged = zscales_under(ChunkedMap(batch_size=ragged_tile))
  exact = zscales_under(ChunkedMap(batch_size=n_blocks))
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


def test_zscale_mode_single_mutation_is_accepted_and_runs():
  # global_weight, adjacent_repeat_weight, self_weight and zscale_mode="single_mutation" are all ported (#2621).
  p = _problem(10)
  cfg = dataclasses.replace(p.config, zscale_mode="single_mutation")
  res = _run(p, config=cfg)
  ref = _run(p)
  zs = np.asarray(res.zscales)
  assert zs.shape == (3,)
  assert np.all(np.isfinite(zs)) and np.all(zs > 0)
  assert np.array_equal(np.asarray(res.seq)[p.pin_pos], p.pin_prot)
  # the scales differ from block mode (different statistic), and the descent used them: wH = (1-lam)/sdH
  assert not np.allclose(zs[:2], np.asarray(ref.zscales)[:2])


def _smz_args(p: _Problem, **over):
  args = dict(
    table=jnp.asarray(p.table), e_idx=jnp.asarray(p.e_idx), seq=jnp.asarray(p.seq0), designable=jnp.asarray(p.designable),
    pin_positions=jnp.asarray(p.pin_pos), pin_prot=jnp.asarray(p.pin_prot), pin_dep=jnp.asarray(p.pin_dep),
    pin_dep_valid=jnp.asarray(p.pin_dep_valid), valid_tokens=jnp.asarray(p.valid),
  )  # fmt: skip
  args.update(over)
  return args


def _ref_std(deltas: np.ndarray, rows: np.ndarray, valid: np.ndarray) -> float:
  """Upstream _zscale: std(unbiased=False) of the finite valid-column entries of the selected rows; 1.0 if none; floor 1e-6."""
  vals = deltas[rows][:, valid]
  vals = vals[np.isfinite(vals)]
  if vals.size == 0:
    return 1.0
  return max(float(vals.std()), 1e-6)


def _ref_single_mutation(p: _Problem, masks: GlobalMasks | None, seq: np.ndarray | None = None) -> np.ndarray:
  """Brute-force mirror of upstream _block_mutation_tables + the 2243-2255 branch: one full re-evaluation per (row, token)."""
  seq = p.seq0.copy() if seq is None else seq.copy()
  seq[p.pin_pos] = p.pin_prot
  n = p.designable.size
  d_h = np.full((n, V), np.inf)
  d_sel = np.zeros((n, V))
  d_glob = np.zeros((n, V))
  sel0 = _selective(p, seq)
  g0 = _g_oracle(p, seq) if masks is not None else 0.0
  for r, i in enumerate(p.designable):
    cur = int(seq[i])
    for a in np.flatnonzero(p.valid):
      trial = seq.copy()
      trial[i] = a
      e = _energy_batch(p, np.stack([trial, seq]))
      d_h[r, a] = e[0] - e[1]
      if a == cur:
        continue
      d_sel[r, a] = _selective(p, trial) - sel0
      if masks is not None:
        d_glob[r, a] = _g_oracle(p, trial) - g0
  sd_h = _ref_std(d_h, np.ones(n, bool), p.valid)
  sel_rows = np.abs(d_sel).sum(1) > 0
  sd_s = _ref_std(d_sel, sel_rows if sel_rows.any() else np.ones(n, bool), p.valid)
  glob_rows = np.abs(d_glob).sum(1) > 0
  sd_g = _ref_std(d_glob, glob_rows, p.valid) if (masks is not None and glob_rows.any()) else 1.0
  return np.array([sd_h, sd_s, sd_g])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_mutation_zscales_match_numpy_reference_of_upstream(seed):
  p, masks = _knob_problem(seed, zscale_mode="single_mutation")
  got = np.asarray(single_mutation_zscales(**_smz_args(p), glob_masks=masks))
  want = _ref_single_mutation(p, masks)
  assert got.shape == (3,)
  assert want[2] != 1.0  # the global scale is actually exercised
  np.testing.assert_allclose(got, want, rtol=2e-4)
  without = np.asarray(single_mutation_zscales(**_smz_args(p)))
  np.testing.assert_allclose(without[:2], want[:2], rtol=2e-4)
  assert without[2] == 1.0


def test_single_mutation_zscales_set_the_pins_first():
  p = _problem(4)
  raw = p.seq0.copy()
  raw[p.pin_pos] = token_index("A")  # unpinned seed
  got = np.asarray(single_mutation_zscales(**_smz_args(p, seq=jnp.asarray(raw))))
  np.testing.assert_allclose(got, np.asarray(single_mutation_zscales(**_smz_args(p))), rtol=1e-6)


def test_single_mutation_selective_scale_floors_when_no_row_varies():
  # No pin edge reaches any designable position -> the selective deltas are all zero -> upstream's all-zero fallback: std 0 -> 1e-6.
  p = _problem(5)
  table = p.table.copy()
  for c in p.pin_pos:
    table[c, 1:] = 0.0  # edges out of a pin
    table[:, 1:][p.e_idx[:, 1:] == c] = 0.0  # edges into a pin
  q = p._replace(table=table)
  got = np.asarray(single_mutation_zscales(**_smz_args(q)))
  assert got[1] == pytest.approx(1e-6)
  assert got[0] > 1e-3
  assert got[2] == 1.0
  np.testing.assert_allclose(got, _ref_single_mutation(q, None), rtol=2e-4)


def test_single_mutation_self_weight_does_not_change_sdh():
  # Upstream uses the unweighted ctx.scorer here (block mode uses stab_scorer): self_weight must not move sdH.
  p = _problem(6)
  base = np.asarray(single_mutation_zscales(**_smz_args(p)))
  res_w = _run(p, config=dataclasses.replace(p.config, zscale_mode="single_mutation", self_weight=0.5))
  res_1 = _run(p, config=dataclasses.replace(p.config, zscale_mode="single_mutation"))
  np.testing.assert_allclose(np.asarray(res_w.zscales), np.asarray(res_1.zscales), rtol=1e-6)
  np.testing.assert_allclose(np.asarray(res_1.zscales), base, rtol=1e-6)
  # ... whereas block mode's sdH does move with self_weight
  blk_w = _run(p, config=dataclasses.replace(p.config, self_weight=0.5))
  blk_1 = _run(p)
  assert not np.isclose(np.asarray(blk_w.zscales)[0], np.asarray(blk_1.zscales)[0])


def test_single_mutation_block_descent_with_global_weight_uses_the_global_scale():
  p, masks = _knob_problem(1, zscale_mode="single_mutation", global_weight=0.5)
  res = block_descent(
    jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.seq0), jnp.asarray(p.designable), jnp.asarray(p.blocks),
    jnp.asarray(p.block_valid), jnp.asarray(p.pin_pos), jnp.asarray(p.pin_prot), jnp.asarray(p.pin_dep),
    jnp.asarray(p.pin_dep_valid), jnp.asarray(p.valid), jnp.asarray(p.rep_mask), jnp.asarray(p.residue),
    config=p.config, uniforms=jnp.zeros(p.config.block_max_rounds * p.designable.size, dtype=jnp.float32),
    binder_mask=jnp.ones(LENGTH, dtype=bool),
  )  # fmt: skip
  np.testing.assert_allclose(np.asarray(res.zscales), _ref_single_mutation(p, masks), rtol=2e-4)


# --- the three objective knobs (#2621): block_objective against independent full-objective oracles ---------------------

_CLASS_TOKENS = {
  "his": ("H", "HIS-P", "HIS-S", "HIS-A"),
  "asp": ("D", "ASP-P", "ASP-D", "ASP-A"),
  "glu": ("E", "GLU-P", "GLU-D", "GLU-A"),
}


def _with(p: _Problem, **config_kwargs) -> _Problem:
  return p._replace(config=dataclasses.replace(p.config, **config_kwargs))


def _knob_problem(seed: int, **config_kwargs) -> tuple[_Problem, GlobalMasks]:
  """Designable positions start at Ala (never a titratable class); three other positions hold native His / Asp / Glu."""
  p = _problem(seed, block_size=2)
  other = np.setdiff1d(np.arange(LENGTH), np.concatenate([p.designable, p.pin_pos]))
  seq0 = p.seq0.copy()
  seq0[p.designable] = token_index("A")
  seq0[other[:3]] = [token_index("H"), token_index("D"), token_index("E")]
  p = _with(p._replace(seq0=seq0), **config_kwargs)
  masks = global_masks(jnp.asarray(seq0), jnp.ones(LENGTH, dtype=bool))
  # Independent of the module: the native residue's parent decides the class.
  for name in ("his", "asp", "glu"):
    want = np.isin(seq0, [token_index(t) for t in _CLASS_TOKENS[name]])
    assert np.array_equal(np.asarray(getattr(masks, name)), want)
  return p, masks


def _g_oracle(p: _Problem, seq: np.ndarray) -> float:
  """H(His/Asp/Glu positions -> protonated) - H(Asp/Glu -> deprotonated, His -> HIS-S), the full energies from potts_energy."""
  native = p.seq0
  cls = {name: np.isin(native, [token_index(t) for t in _CLASS_TOKENS[name]]) for name in _CLASS_TOKENS}
  sp, sd = seq.copy(), seq.copy()
  sp[cls["his"]], sp[cls["asp"]], sp[cls["glu"]] = token_index("HIS-P"), token_index("ASP-P"), token_index("GLU-P")
  sd[cls["his"]], sd[cls["asp"]], sd[cls["glu"]] = token_index("HIS-S"), token_index("ASP-D"), token_index("GLU-D")
  h = _energy_batch(p, np.stack([sp, sd]))
  return float(h[0] - h[1])


def _adjacent_oracle(p: _Problem, seq: np.ndarray) -> float:
  """Count of sequence-adjacent (residue number differs by 1) position pairs on the same canonical amino acid."""
  total = 0.0
  for i in range(LENGTH):
    for j in range(i + 1, LENGTH):
      if abs(int(p.residue[i]) - int(p.residue[j])) == 1:
        total += float(canonical_letter(int(seq[i])) == canonical_letter(int(seq[j])))
  return total


def _block_joint(p: _Problem, row: int, sub_valid: np.ndarray, **kwargs) -> np.ndarray:
  return np.asarray(
    block_objective(
      jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.seq0), jnp.asarray(p.blocks[row]),
      jnp.asarray(p.block_valid[row]), jnp.asarray(p.pin_pos), jnp.asarray(p.pin_prot), jnp.asarray(p.pin_dep),
      jnp.asarray(p.pin_dep_valid), jnp.asarray(sub_valid), jnp.asarray(p.rep_mask), jnp.asarray(p.residue),
      config=p.config, **kwargs,
    ),
  )  # fmt: skip


def _enumerate_block(p: _Problem, row: int):
  tokens = np.array([token_index(t) for t in ("K", "A", "HIS-P", "ASP-D", "G")])
  sub_valid = np.zeros(V, dtype=bool)
  sub_valid[tokens] = True
  members = p.blocks[row][np.flatnonzero(p.block_valid[row])]
  combos = np.array(list(itertools.product(tokens, repeat=members.size)), dtype=np.int32)
  seqs = np.repeat(p.seq0[None], combos.shape[0], axis=0)
  seqs[:, members] = combos
  return sub_valid, members, seqs, _flat_index(p.block_valid[row], combos, V)


def _unary_g_sum(p: _Problem, members: np.ndarray, seqs: np.ndarray) -> np.ndarray:
  """Per assignment: the sum over members of g(seq0 with that member changed) - g(seq0), the unary form the engine uses."""
  base_g = _g_oracle(p, p.seq0)
  out = np.zeros(seqs.shape[0])
  for n, seq in enumerate(seqs):
    for m in members:
      single = p.seq0.copy()
      single[m] = seq[m]
      out[n] += _g_oracle(p, single) - base_g
  return out


@pytest.mark.parametrize("seed", [0, 1])
def test_global_weight_adds_the_weighted_unary_g_difference(seed):
  p, masks = _knob_problem(seed, global_weight=0.7)
  live = False
  for row in range(p.designable.size):
    sub_valid, members, seqs, idx = _enumerate_block(p, row)
    base = _block_joint(p, row, sub_valid, wh=0.8, wsel=0.35, wglob=0.0, glob_masks=masks)
    with_g = _block_joint(p, row, sub_valid, wh=0.8, wsel=0.35, wglob=0.7, glob_masks=masks)
    expect = _unary_g_sum(p, members, seqs)
    delta = (with_g - base)[idx]
    np.testing.assert_allclose(delta - delta[0], 0.7 * (expect - expect[0]), atol=2e-2)
    live |= bool(np.ptp(expect) > 0)
  assert live  # the term moves at least one block


def test_global_weight_sign_positive_minimises_g():
  p, masks = _knob_problem(2, global_weight=1.0)
  row = int(np.argmax(p.block_valid.sum(axis=1)))
  sub_valid, members, seqs, idx = _enumerate_block(p, row)
  unary_sum = _unary_g_sum(p, members, seqs)
  assert np.ptp(unary_sum) > 0
  rest = _block_joint(p, row, sub_valid, wh=0.0, wsel=0.0, wglob=0.0, glob_masks=masks)
  for sign in (1.0, -1.0):
    joint = _block_joint(p, row, sub_valid, wh=0.0, wsel=0.0, wglob=sign, glob_masks=masks)
    chosen = int(np.argmin((joint - rest)[idx]))
    best = unary_sum.min() if sign > 0 else unary_sum.max()  # positive weight: lowest g; negative: highest
    assert unary_sum[chosen] == pytest.approx(best, abs=1e-2)


def test_global_weight_needs_the_binder_mask():
  p, _masks = _knob_problem(0, global_weight=0.5)
  with pytest.raises(ValueError, match="binder_mask"):
    _run(p)
  with pytest.raises(ValueError, match="glob_masks"):
    _block_joint(p, 0, p.valid, wh=1.0, wsel=1.0, wglob=0.5)


def test_global_zscale_is_one_without_the_masks_and_positive_with_them():
  p, masks = _knob_problem(3)
  args = (
    jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.seq0), jnp.asarray(p.blocks), jnp.asarray(p.block_valid),
    jnp.asarray(p.pin_pos), jnp.asarray(p.pin_prot), jnp.asarray(p.pin_dep), jnp.asarray(p.pin_dep_valid), jnp.asarray(p.valid),
  )  # fmt: skip
  without = np.asarray(block_zscales(*args))
  live = np.asarray(block_zscales(*args, glob_masks=masks))
  assert without[2] == 1.0
  assert live[2] > 0.0
  assert live[2] != 1.0
  np.testing.assert_allclose(live[:2], without[:2], rtol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_adjacent_repeat_weight_charges_adjacent_same_canonical_pairs(seed):
  p, _ = _knob_problem(seed, adjacent_repeat_weight=0.9)
  off_p = _with(p, adjacent_repeat_weight=0.0)
  live = False
  for row in range(p.designable.size):
    sub_valid, _members, seqs, idx = _enumerate_block(p, row)
    delta = (_block_joint(p, row, sub_valid, wh=0.8, wsel=0.35) - _block_joint(off_p, row, sub_valid, wh=0.8, wsel=0.35))[idx]
    count = np.array([_adjacent_oracle(p, s) for s in seqs])
    np.testing.assert_allclose(delta - delta[0], 0.9 * (count - count[0]), atol=1e-4)
    live |= bool(np.ptp(count) > 0)
  assert live  # at least one block can create or break an adjacent repeat


def test_adjacent_repeat_weight_large_leaves_no_adjacent_repeat_among_designable_positions():
  p = _problem(4, block_size=2)
  seq = np.asarray(_run(p, config=dataclasses.replace(p.config, adjacent_repeat_weight=50.0)).seq)
  designable = {int(x) for x in p.designable}
  for i in range(LENGTH - 1):
    if i in designable and i + 1 in designable:
      assert canonical_letter(int(seq[i])) != canonical_letter(int(seq[i + 1]))


@pytest.mark.parametrize("weight", [0.0, 2.5])
def test_self_weight_rescales_the_slot_zero_field_of_the_stability_term_only(weight):
  p, _ = _knob_problem(1, self_weight=weight)
  p_one = _with(p, self_weight=1.0)
  wh = 0.8
  # Independent oracle: the full Potts energy on a table whose slot-0 field is scaled; the selective term stays on the true table.
  scaled = p.table.copy()
  scaled[:, 0] *= weight
  for row in range(p.designable.size):
    sub_valid, _members, seqs, idx = _enumerate_block(p, row)
    energy_diff = _energy_batch(p._replace(table=scaled), seqs) - _energy_batch(p, seqs)
    delta = (_block_joint(p, row, sub_valid, wh=wh, wsel=0.0) - _block_joint(p_one, row, sub_valid, wh=wh, wsel=0.0))[idx]
    np.testing.assert_allclose(delta - delta[0], wh * (energy_diff - energy_diff[0]), atol=5e-3)


def test_self_weight_one_is_an_exact_noop():
  p = _problem(5, block_size=2)
  explicit = _run(p, config=dataclasses.replace(p.config, self_weight=1.0))
  default = _run(p)
  assert np.array_equal(np.asarray(explicit.seq), np.asarray(default.seq))
  assert np.array_equal(np.asarray(explicit.zscales), np.asarray(default.zscales))
  table = jnp.asarray(p.table)
  assert stability_table(table, 1.0) is table


def test_self_weight_changes_the_frozen_sdh_and_not_the_selective_scale():
  p = _problem(5, block_size=2)
  default = _run(p)
  weighted = _run(p, config=dataclasses.replace(p.config, self_weight=4.0))
  assert float(weighted.zscales[0]) != float(default.zscales[0])
  assert float(weighted.zscales[1]) == pytest.approx(float(default.zscales[1]), rel=1e-6)


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
