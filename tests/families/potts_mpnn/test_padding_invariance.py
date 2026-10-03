# ruff: noqa: S101
"""Spec §4.2: Potts energies do not depend on how far the input was padded.

THE PROPERTY IS UNEXERCISED IN PRODUCTION, which is why it needs a test rather
than merely an invariant comment. Both ``pad`` call sites pass ``L_total``
(``driver.py:590``, ``sample_host.py:277``), so no aminx run has ever built a
graph with ``l_pad > L_total`` and the masking below has never had to work.
``RunSpecification.max_length`` (``run/specs.py:273``, default 512) does not
reach Potts padding at all.

WHY IT IS NOT TRIVIAL. ``features.py:92`` picks ``k = min(k_neighbors,
coords.shape[0])``, i.e. ``min(48, L_pad)`` and NOT ``min(48, L_total)``. Padding
an L=30 chain to 128 therefore grows the neighbour set from 30 to 48: every row
acquires 18 neighbours that do not exist, and the energy is only unchanged
because ``mask_2d`` pushes non-present pairs to the row ``D_max`` and the Potts
head and ``merge_pair`` then drop them. Different K also means a different
summation width, so the agreement is to floating-point round-off and NOT bit
exact -- the spec's "f64 exact" is unachievable here, because float addition is
not associative and the reduction order genuinely changes.

Measured 261003 at f64 on a random-init model, worst case over
``L_total in {30, 49}`` and ``l_pad in {L_total, 64, 128}``: 5.33e-15 absolute,
against energies of order 0.09 to 6.4. The band below is 1e-12, ~200x the worst
observation and twelve orders below the control separation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import absolute_energies
from aminx.families.potts_mpnn.featurize import (
  pad,
  parse_pdb_upstream,
  tied_featurize_port,
)
from aminx.families.potts_mpnn.model import PottsMPNN

_THREE = {
  "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
  "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
  "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
  "W": "TRP", "Y": "TYR",
}

# Padded rows are X in the etab alphabet (``featurize.pad`` fills S with
# ``X_INDEX``); 21 is X's etab slot, zero in the table by construction.
_ETAB_X = 21

#: Agreement band across ``l_pad``. See the module docstring for its basis.
_BAND = 1e-12


@pytest.fixture
def x64() -> Iterator[None]:
  """f64 for the duration, restored afterwards.

  Restoring matters: ``jax_enable_x64`` is process-global, and a test that
  leaves it on silently re-types every later test in the session.
  """
  previous = bool(getattr(jax.config, "jax_enable_x64", False))
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", previous)


def _write_helix(path: Path, sequence: str) -> None:
  """A crude jittered helix, so the kNN graph is not degenerate.

  A straight line or a constant-coordinate chain would make many pair distances
  equal and let ``top_k`` break ties arbitrarily, which is the §6.5b
  ``knn_boundary_tie`` case this test must stay clear of.
  """
  lines: list[str] = []
  serial = 1
  rng = np.random.default_rng(0)
  for index, amino in enumerate(sequence, start=1):
    turn = index * 0.6
    base = np.array([np.cos(turn) * 2.3, np.sin(turn) * 2.3, index * 1.5])
    for offset, atom in enumerate(("N", "CA", "C", "O")):
      point = base + rng.normal(scale=0.15, size=3) + offset * 0.4
      lines.append(
        f"{'ATOM':<6.6}{serial:5d} {atom:>4.4} {_THREE[amino]:>3.3} A"
        f"{index:4d}    {point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}",
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _features(tmp_path: Path, l_total: int):  # noqa: ANN202
  sequence = ("ACDEFGHIKLMNPQRSTVWY" * 3)[:l_total]
  pdb = tmp_path / f"helix{l_total}.pdb"
  _write_helix(pdb, sequence)
  parsed = parse_pdb_upstream(str(pdb))[0]
  return tied_featurize_port([parsed], None)[0]


def _energies(features, l_pad, model, rows, pad_valid=None):  # noqa: ANN001, ANN202
  """Absolute energies of ``rows`` with the graph padded to ``l_pad``.

  ``pad_valid`` overrides the mask ``pad`` derives, which is what lets the
  negative control below lie about which rows are real.
  """
  padded, derived = pad(features, l_pad)
  n_pad = l_pad - features.L_total
  sequences = (
    np.concatenate(
      [rows, np.full((rows.shape[0], n_pad), _ETAB_X, dtype=np.int32)], axis=1,
    )
    if n_pad
    else rows
  )
  mask = derived if pad_valid is None else pad_valid
  return np.asarray(
    absolute_energies(
      model,
      jnp.asarray(padded.x, dtype=jnp.float64),
      jnp.asarray(padded.present, dtype=jnp.float64),
      jnp.asarray(padded.residue_idx, dtype=jnp.int32),
      jnp.asarray(padded.chain_encoding, dtype=jnp.int32),
      jnp.asarray(mask),
      jnp.asarray(sequences, dtype=jnp.int32),
    ),
    dtype=np.float64,
  )


@pytest.mark.parametrize("l_total", [30, 49])
def test_energy_invariant_to_padding(
  x64: None, tmp_path: Path, l_total: int,
) -> None:
  """Energies agree across ``l_pad`` although K, and so the neighbour set, grows."""
  del x64  # the fixture's effect is the dtype, not a value
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  features = _features(tmp_path, l_total)
  rows = np.asarray(
    [[index % 20 for index in range(features.L_total)]], dtype=np.int32,
  )
  base = _energies(features, features.L_total, model, rows)

  for l_pad in (64, 128):
    if l_pad < features.L_total:
      continue
    got = _energies(features, l_pad, model, rows)
    delta = float(np.max(np.abs(got - base)))
    assert delta <= _BAND, (
      f"L_total={features.L_total} padded to {l_pad} moved the energy by "
      f"{delta:.3e}, over the {_BAND:.0e} band. K went from "
      f"{min(48, features.L_total)} to {min(48, l_pad)}, so either the pad "
      f"rows are reaching the energy or the extra neighbours are not masked."
    )


@pytest.mark.parametrize("l_total", [30, 49])
def test_padding_invariance_control_fires(
  x64: None, tmp_path: Path, l_total: int,
) -> None:
  """The comparison above is sensitive to the mask at all.

  WITHOUT THIS, ``test_energy_invariant_to_padding`` IS VACUOUS. Dropping
  ``pad_valid`` entirely would also make every ``l_pad`` agree, so "the
  energies matched" on its own is equally consistent with a correct mask and
  with no mask. Switching ``pad_valid`` off over the back half of the REAL rows
  must therefore move the energy hard -- measured 261003 at 1.38 (L=30) and
  10.06 (L=49), against a 1e-12 band.

  MEASURED AND DELIBERATELY NOT USED AS THE CONTROL: setting ``pad_valid`` to
  all-True over a 128-row graph is a NO-OP, agreeing with the honest run to the
  same 1.28e-15 / 5.33e-15. Pad rows carry ``present=0``, which already drops
  them in ``mask_2d`` (``features.py:87-90``) and in the Potts head, so the two
  masks are redundant on pad rows and only ``present`` is load-bearing there. A
  control built that way would never fire.
  """
  del x64
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  features = _features(tmp_path, l_total)
  rows = np.asarray(
    [[index % 20 for index in range(features.L_total)]], dtype=np.int32,
  )
  base = _energies(features, features.L_total, model, rows)

  half = features.L_total // 2
  lying = np.arange(features.L_total) < half
  off = _energies(features, features.L_total, model, rows, pad_valid=lying)
  delta = float(np.max(np.abs(off - base)))
  assert delta > 1.0, (
    f"disowning rows {half}.. of a real L={features.L_total} chain moved the "
    f"energy by only {delta:.3e}, so pad_valid is not reaching the energy and "
    f"test_energy_invariant_to_padding proves nothing."
  )
