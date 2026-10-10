# ruff: noqa: S101
"""The loop-free pH functions (``ph_export``) driven by a HOST loop reproduce ``block_descent`` (ADR decision 11).

The host loop here is the contract the JS driver must meet: z-scales from per-block statistics pooled once, then
sweeps of ``visit`` calls in block order, one uniform consumed per visit when sampling, until a sweep changes nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.protonpotts_mpnn.ph_descent import block_descent, block_zscales
from aminx.families.protonpotts_mpnn.ph_export import field_at, make_visit, pool, zblock
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies_at
from tests.protonpotts.test_ph_descent import _problem, _run


def host_descent(p, uniforms: np.ndarray):  # noqa: ANN001, ANN201
  """Descent by repeated calls of the exported functions. Returns ``(seq, n_draws, rounds, zscales)``."""
  cfg = p.config
  visit = jax.jit(make_visit(cfg))
  zb = jax.jit(zblock)
  table, e_idx = jnp.asarray(p.table), jnp.asarray(p.e_idx)
  pins = (jnp.asarray(p.pin_pos), jnp.asarray(p.pin_prot), jnp.asarray(p.pin_dep), jnp.asarray(p.pin_dep_valid))
  valid_tokens, rep_mask, residue = jnp.asarray(p.valid), jnp.asarray(p.rep_mask), jnp.asarray(p.residue)
  seq = np.asarray(p.seq0).copy()
  seq[p.pin_pos] = p.pin_prot
  seq_j = jnp.asarray(seq)

  stats = [
    [np.asarray(x)[0] for x in zb(table, e_idx, seq_j, jnp.asarray(b), jnp.asarray(v), *pins, valid_tokens)]
    for b, v in zip(p.blocks, p.block_valid, strict=True)
  ]
  var_h, has_h, var_s, has_s = (np.asarray([row[i] for row in stats]) for i in range(4))
  lam = float(cfg.combined_lambda)
  zscales, wh, wsel = pool(
    jnp.asarray(var_h), jnp.asarray(has_h), jnp.asarray(var_s), jnp.asarray(has_s),
    jnp.asarray([1.0 - lam], jnp.float32), jnp.asarray([lam], jnp.float32),
  )

  temperature = float(cfg.temperature)
  draws, rounds = 0, 0
  changed = True
  while changed and rounds < cfg.block_max_rounds:
    changed = False
    for block, valid in zip(p.blocks, p.block_valid, strict=True):
      u = uniforms[min(draws, len(uniforms) - 1)] if temperature > 0 else 0.0
      (digits,) = visit(
        table, e_idx, jnp.asarray(seq), jnp.asarray(block), jnp.asarray(valid), *pins, valid_tokens, rep_mask,
        residue, wh, wsel, jnp.asarray([u], jnp.float32), jnp.asarray([temperature], jnp.float32),
      )
      if temperature > 0:
        draws += 1
      digits = np.asarray(digits)
      current = seq[np.where(valid, block, 0)]
      changed |= bool(np.any(valid & (digits != current)))
      for b, ok, d in zip(block, valid, digits, strict=True):
        if ok:
          seq[b] = d
    rounds += 1
  return seq, draws, rounds, np.asarray(zscales)


@pytest.mark.parametrize(("seed", "temperature"), [(4, 0.0), (5, 0.0), (6, 0.05), (7, 1.0)])
def test_host_loop_over_exported_functions_equals_block_descent(seed: int, temperature: float) -> None:
  p = _problem(seed, temperature=temperature)
  rng = np.random.default_rng(100 + seed)
  uniforms = rng.uniform(size=max(1, p.config.block_max_rounds * p.designable.size)).astype(np.float32)
  want = _run(p, uniforms=uniforms)
  seq, draws, rounds, zscales = host_descent(p, uniforms)
  np.testing.assert_array_equal(seq, np.asarray(want.seq))
  assert draws == int(want.n_draws)
  assert rounds == int(want.rounds)
  np.testing.assert_allclose(zscales, np.asarray(want.zscales), rtol=1e-6)


def test_the_loop_is_not_vacuous() -> None:
  """A wrong uniform stream must change a sampled descent, so equality above is informative."""
  p = _problem(7, temperature=1.0)
  rng = np.random.default_rng(1)
  u1 = rng.uniform(size=p.config.block_max_rounds * p.designable.size).astype(np.float32)
  u2 = np.roll(u1, 1)
  assert not np.array_equal(host_descent(p, u1)[0], host_descent(p, u2)[0])


def test_pooled_statistics_equal_block_zscales_and_flags_are_int32() -> None:
  p = _problem(4)
  table, e_idx = jnp.asarray(p.table), jnp.asarray(p.e_idx)
  pins = (jnp.asarray(p.pin_pos), jnp.asarray(p.pin_prot), jnp.asarray(p.pin_dep), jnp.asarray(p.pin_dep_valid))
  seq = np.asarray(p.seq0).copy()
  seq[p.pin_pos] = p.pin_prot
  out = zblock(table, e_idx, jnp.asarray(seq), jnp.asarray(p.blocks[0]), jnp.asarray(p.block_valid[0]), *pins,
               jnp.asarray(p.valid))
  assert [o.shape for o in out] == [(1,)] * 4
  assert out[1].dtype == jnp.int32
  assert out[3].dtype == jnp.int32
  ref = block_zscales(table, e_idx, jnp.asarray(p.seq0), jnp.asarray(p.blocks), jnp.asarray(p.block_valid), *pins,
                      jnp.asarray(p.valid))
  assert ref.shape == (3,)


def test_field_at_is_candidate_energies_at() -> None:
  p = _problem(5)
  positions = jnp.asarray([0, 3, 7, 7], dtype=jnp.int32)
  (got,) = field_at(jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.seq0), positions)
  want = candidate_energies_at(jnp.asarray(p.table), jnp.asarray(p.e_idx), jnp.asarray(p.seq0), positions)
  np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
