"""Tests for the per-structure pH design orchestration (ph_design).

Oracles are independent of the code under test where it matters: the final energy comes from ``potts_energy``,
the selective gap is recomputed from ``candidate_energies_at`` by definition, and the replay test rebuilds the
``block_descent`` inputs by hand from the plan and block helpers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
from aminx.families.protonpotts_mpnn.ph_descent import block_descent, rep_class_mask
from aminx.families.protonpotts_mpnn.ph_design import design_structure
from aminx.families.protonpotts_mpnn.ph_plan import (
  block_table,
  plan_from_center_types,
  valid_token_mask,
)
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies, candidate_energies_at

V = 30
SENTINEL = 10**6


class _Problem:
  def __init__(self, seed: int) -> None:
    rng = np.random.default_rng(seed)
    self.length = 14 + seed % 5
    k_slots = 5 + seed % 2
    length = self.length
    self.table = (0.3 * rng.normal(size=(length, k_slots, V, V))).astype(np.float32)
    self.table[:, 0] = self.table[:, 0] * np.eye(V, dtype=np.float32)
    self.e_idx = np.zeros((length, k_slots), dtype=np.int32)
    for i in range(length):
      others = np.delete(np.arange(length), i)
      self.e_idx[i, 0] = i
      self.e_idx[i, 1:] = rng.choice(others, size=k_slots - 1, replace=False)  # one-way edges arise naturally
    self.valid = valid_token_mask(("HIS-A", "ASP-A", "GLU-A", "UNK"))
    self.native = rng.choice(np.flatnonzero(self.valid), size=length).astype(np.int32)
    self.binder = np.ones(length, dtype=bool)
    self.binder[rng.choice(length, size=2, replace=False)] = False
    self.res_id = np.arange(length, dtype=np.int32) + 10


def _config(**kwargs) -> PHDesignConfig:
  base = dict(
    binder_chain="A",
    center_types=("HIS-P", "ASP-P"),
    block_size=3,
    temperature=0.0,
    samples_per_site=3,
    block_max_rounds=20,
  )
  base.update(kwargs)
  return PHDesignConfig(**base)


def _designs(p: _Problem, config: PHDesignConfig, **kwargs):
  return design_structure(p.table, p.e_idx, p.native, p.binder, p.res_id, config, **kwargs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_block_descent_designs_are_pinned_scored_and_sorted(seed):
  p = _Problem(seed)
  config = _config(temperature=0.05)
  designs = _designs(p, config, key=jax.random.PRNGKey(seed))
  assert len(designs) == config.samples_per_site

  for design in designs:
    assert design.method == "block_descent"
    assert len(design.pins) == 2
    pinned = {pin.position for pin in design.pins}
    for pin in design.pins:
      assert design.sequence[pin.position] == pin.prot_idx
      assert pin.protonation_type in ("HIS-P", "ASP-P")
    assert not pinned & set(design.designable)
    untouched = [i for i in range(p.length) if i not in design.designable and i not in pinned]
    assert np.array_equal(design.sequence[untouched], p.native[untouched])

    # Independent recomputation of the energies from the returned sequence.
    energy = float(
      potts_energy(
        jnp.asarray(p.table),
        jnp.asarray(p.e_idx),
        jnp.ones(p.length, dtype=bool),
        jnp.asarray(design.sequence),
      )
    )
    assert design.final_potts_energy == pytest.approx(energy, rel=1e-5, abs=1e-5)
    rows = np.asarray(
      candidate_energies_at(
        jnp.asarray(p.table),
        jnp.asarray(p.e_idx),
        jnp.asarray(design.sequence),
        jnp.asarray([pin.position for pin in design.pins], dtype=jnp.int32),
      )
    )
    gaps = [rows[i, pin.prot_idx] - np.mean([rows[i, d] for d in pin.dep_idxs]) for i, pin in enumerate(design.pins)]
    assert design.selective_energies == pytest.approx(gaps, rel=1e-5, abs=1e-5)
    assert design.selective_energy == pytest.approx(sum(gaps), rel=1e-5, abs=1e-5)

  energies = [d.final_potts_energy for d in designs]
  assert energies == sorted(energies)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keys_and_temperature_zero_determinism(seed):
  p = _Problem(seed)
  sampled = _config(temperature=1.0, samples_per_site=2)
  first = _designs(p, sampled, key=jax.random.PRNGKey(7))
  again = _designs(p, sampled, key=jax.random.PRNGKey(7))
  assert [d.sequence.tolist() for d in first] == [d.sequence.tolist() for d in again]

  runs = [_designs(p, sampled, key=jax.random.PRNGKey(k)) for k in range(4)]
  assert len({tuple(tuple(d.sequence.tolist()) for d in run) for run in runs}) > 1

  greedy_zero = _config(temperature=0.0)
  zero_a = _designs(p, greedy_zero)
  zero_b = _designs(p, greedy_zero)
  assert [d.sequence.tolist() for d in zero_a] == [d.sequence.tolist() for d in zero_b]
  assert all(d.n_draws == 0 for d in zero_a)


@pytest.mark.parametrize("seed", [0, 1])
def test_replay_matches_direct_block_descent(seed):
  p = _Problem(seed)
  config = _config(temperature=1.0, samples_per_site=2, block_max_rounds=6)
  cdf_order = np.random.default_rng(100 + seed).permutation(V).astype(np.int32)

  plan = plan_from_center_types(
    np.asarray(candidate_energies(jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.native))),
    p.e_idx,
    p.binder,
    p.res_id,
    config.center_types,
    config.dep_map_dict(),
    infill_scope=config.infill_scope,
    neighbour_k=config.neighbour_k,
    max_mutations=config.max_mutations,
  )
  n_designable = len(plan.designable)
  need = config.block_max_rounds * n_designable
  rng = np.random.default_rng(200 + seed)
  uniforms = [rng.uniform(size=need).astype(np.float32) for _ in range(config.samples_per_site)]

  got = _designs(p, config, uniforms=uniforms, cdf_order=jnp.asarray(cdf_order))

  blocks, block_valid = block_table(p.e_idx, plan.designable, config.block_size)
  depth = max(len(pin.dep_idxs) for pin in plan.pins)
  pin_dep = np.zeros((len(plan.pins), depth), dtype=np.int32)
  pin_dep_valid = np.zeros((len(plan.pins), depth), dtype=bool)
  for i, pin in enumerate(plan.pins):
    pin_dep[i, : len(pin.dep_idxs)] = pin.dep_idxs
    pin_dep_valid[i, : len(pin.dep_idxs)] = True
  residue = np.where(p.binder, p.res_id, -SENTINEL - 100 * np.arange(p.length))
  for sample, design in enumerate(got):
    direct = block_descent(
      jnp.asarray(p.table),
      jnp.asarray(p.e_idx),
      jnp.asarray(p.native),
      jnp.asarray(plan.designable, dtype=jnp.int32),
      jnp.asarray(blocks),
      jnp.asarray(block_valid),
      jnp.asarray([pin.position for pin in plan.pins], dtype=jnp.int32),
      jnp.asarray([pin.prot_idx for pin in plan.pins], dtype=jnp.int32),
      jnp.asarray(pin_dep),
      jnp.asarray(pin_dep_valid),
      jnp.asarray(valid_token_mask(config.forbidden_tokens)),
      jnp.asarray(rep_class_mask(config.repetitive_window_parents)),
      jnp.asarray(residue, dtype=jnp.int32),
      config=config,
      uniforms=jnp.asarray(uniforms[sample]),
      cdf_order=jnp.asarray(cdf_order),
    )
    assert np.array_equal(design.sequence, np.asarray(direct.seq))
    assert design.n_draws == int(direct.n_draws)


def test_explicit_centres_pin_exactly_that_residue():
  p = _Problem(3)
  free = np.flatnonzero(p.binder)
  position = int(free[0])
  rid = int(p.res_id[position])
  config = _config(center_types=(), explicit_centers=((rid, "HIS-P"),), temperature=0.0)
  designs = _designs(p, config)
  assert designs
  for design in designs:
    assert len(design.pins) == 1
    assert design.pins[0].position == position
    assert design.pins[0].res_id == rid
    assert design.pins[0].protonation_type == "HIS-P"
    assert design.sequence[position] == design.pins[0].prot_idx


def test_plan_returning_none_gives_empty_list():
  p = _Problem(4)
  only_one = np.zeros(p.length, dtype=bool)
  only_one[int(np.flatnonzero(p.binder)[0])] = True
  config = _config(center_types=("HIS-P", "ASP-P"))
  assert _designs(p, config) != []  # control: the full binder mask plans fine, so the [] is from the mask
  assert design_structure(p.table, p.e_idx, p.native, only_one, p.res_id, config, key=jax.random.PRNGKey(0)) == []


def test_centre_free_greedy_has_no_pins_and_zero_selective_energy():
  greedy = pytest.importorskip("aminx.families.protonpotts_mpnn.ph_greedy")
  assert hasattr(greedy, "greedy_energy_block")
  p = _Problem(5)
  config = _config(
    method="greedy_energy_block",
    center_types=(),
    infill_scope="chain",
    temperature=0.05,
    samples_per_site=2,
  )
  designs = _designs(p, config, key=jax.random.PRNGKey(0))
  assert len(designs) == 2
  for design in designs:
    assert design.method == "greedy_energy_block"
    assert design.pins == ()
    assert design.selective_energy == 0.0
    assert design.selective_energies == ()
  energies = [d.final_potts_energy for d in designs]
  assert energies == sorted(energies)


def test_empty_binder_mask_raises():
  p = _Problem(0)
  with pytest.raises(ValueError, match="binder_mask"):
    design_structure(
      p.table, p.e_idx, p.native, np.zeros(p.length, dtype=bool), p.res_id, _config(), key=jax.random.PRNGKey(0)
    )


def test_positive_temperature_needs_key_or_uniforms():
  p = _Problem(0)
  with pytest.raises(ValueError, match="PRNG key or explicit uniforms"):
    _designs(p, _config(temperature=0.5))
