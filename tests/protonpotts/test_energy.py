"""ProtonPotts merge and energy against a naive numpy statement of upstream's rules (spec §42).

The reference loops below are written from upstream ``compute_potts_context`` (one reciprocal merge,
first-matching reverse slot) and ``calc_potts_eners`` (sum over every directed edge), not from aminx's
``merge_pair``/``potts_energy``, so agreement is a check and not a tautology. Whether the real head output
matches upstream is settled by the ``protonpotts_energy`` wave, not here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import merge_pair, potts_energy
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn.energy import protonpotts_energies, protonpotts_table
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

V = 30


def _graph(length: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  """A kNN-like directed graph: slot 0 is self, other slots distinct neighbours; some edges one-way."""
  rng = np.random.default_rng(seed)
  e_idx = np.zeros((length, k), dtype=np.int64)
  for i in range(length):
    others = rng.permutation([j for j in range(length) if j != i])[: k - 1]
    e_idx[i] = [i, *others]
  etab = rng.normal(size=(length, k, V, V))
  etab[:, 0] *= np.eye(V)
  return etab, e_idx


def _upstream_merge(etab: np.ndarray, e_idx: np.ndarray) -> np.ndarray:
  length, k = e_idx.shape
  out = etab.copy()
  for i in range(length):
    for slot in range(k):
      j = e_idx[i, slot]
      hits = np.nonzero(e_idx[j] == i)[0]
      if hits.size:
        out[i, slot] = 0.5 * (etab[i, slot] + etab[j, hits[0]].T)
  return out


def _upstream_energy(etab: np.ndarray, e_idx: np.ndarray, seq: np.ndarray) -> float:
  length, k = e_idx.shape
  return float(sum(etab[i, s, seq[i], seq[e_idx[i, s]]] for i in range(length) for s in range(k)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_reciprocal_merge_matches_upstream_rule(seed: int) -> None:
  etab, e_idx = _graph(14, 6, seed)
  valid = jnp.ones(14, dtype=bool)
  got = merge_pair(jnp.asarray(etab), jnp.asarray(e_idx), valid, denom=2, exclude_self=False)
  np.testing.assert_allclose(np.asarray(got), _upstream_merge(etab, e_idx), atol=1e-12)


def test_merge_is_not_idempotent_so_the_shipped_double_merge_is_a_different_table() -> None:
  etab, e_idx = _graph(14, 6, 3)
  valid = jnp.ones(14, dtype=bool)
  once = merge_pair(jnp.asarray(etab), jnp.asarray(e_idx), valid, denom=2, exclude_self=False)
  shipped = merge_pair(once, jnp.asarray(e_idx), valid, denom=4, exclude_self=True)
  assert float(jnp.abs(shipped - once).max()) > 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_energy_sums_every_directed_edge(seed: int) -> None:
  etab, e_idx = _graph(12, 5, seed)
  rng = np.random.default_rng(10 + seed)
  seqs = rng.integers(0, V, size=(4, 12))
  got = potts_energy(
    jnp.asarray(etab), jnp.asarray(e_idx), jnp.ones(12, dtype=bool), jnp.asarray(seqs)
  )
  want = [_upstream_energy(etab, e_idx, s) for s in seqs]
  np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)


def test_protonation_tokens_reach_the_table() -> None:
  """Token indices >= 21 are indexed directly: changing one residue to a protonation token changes H."""
  etab, e_idx = _graph(10, 5, 4)
  seq = np.zeros(10, dtype=np.int64)
  variant = seq.copy()
  variant[3] = PROTONPOTTS_V6.symbols.index("HIS-P")
  h = potts_energy(
    jnp.asarray(etab), jnp.asarray(e_idx), jnp.ones(10, dtype=bool), jnp.asarray(np.stack([seq, variant]))
  )
  assert float(h[0]) != float(h[1])


def test_table_from_a_v30_model_is_square_merged_and_finite() -> None:
  length = 16
  model = PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)
  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(length, 4, 3)) * 3.0 + np.arange(length)[:, None, None])
  args = (
    coords,
    jnp.ones(length),
    jnp.arange(length),
    jnp.zeros(length, dtype=jnp.int32),
    jnp.ones(length, dtype=bool),
  )
  table, e_idx = protonpotts_table(model, *args)
  assert table.shape == (length, e_idx.shape[1], V, V)
  assert bool(jnp.isfinite(table).all())
  seqs = jnp.asarray(rng.integers(0, V, size=(3, length)))
  energies = protonpotts_energies(model, *args, seqs)
  assert energies.shape == (3,)
  assert bool(jnp.isfinite(energies).all())
