# ruff: noqa: S101, SLF001
"""Spec §5.4b: a candidate residue is not scored with the native's side chain.

``_chi_for_candidate`` (``driver.py:287``) is what stops ``score:nll`` from
handing a mutated position the rotamer that belonged to the residue it
replaced. It had no test at all -- grepping the name returns its definition and
its one call site at ``:318``.

READ THE SPEC AND THE CODE AGAINST EACH OTHER AND THEY DISAGREE; see
``test_chi_candidate_spec_and_implementation_disagree`` at the bottom, which
documents the one case that separates them and is NOT asserted as correct here.
The three cases below are the ones both rules agree on, so they pin real
behaviour without taking a side in that conflict.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import _chi_for_candidate
from aminx.families.laser_mpnn.featurize import LASER_ALPHABET, featurize

_A = LASER_ALPHABET.index("A")
_R = LASER_ALPHABET.index("R")
_G = LASER_ALPHABET.index("G")
_K = LASER_ALPHABET.index("K")

# Four real chi angles, distinct so a wrongly-kept row is identifiable by value
# rather than merely by being non-NaN.
_CHI = (41.0, 52.0, 63.0, 74.0)


def _atom(serial: int, name: str, resname: str, resseq: int, x: float) -> str:
  return (
    f"ATOM  {serial:5d} {name:>4} {resname:>3} A"
    f"{resseq:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.0:6.2f}           C"
  )


@pytest.fixture
def features(tmp_path: Path):  # noqa: ANN201
  """Four backbone-only residues, with chi filled in by hand.

  Backbone-only means the featurizer finds no side chain and every chi is NaN,
  so the chi array below is set explicitly rather than depending on how prody
  reconstructs rotamers -- the test is about the candidate rule, not about
  featurization.
  """
  lines: list[str] = []
  serial = 1
  for index, resname in enumerate(("ALA", "LYS", "LYS", "LYS")):
    for offset, name in enumerate(("N", "CA", "C", "O")):
      lines.append(_atom(serial, name, resname, index + 1, float(index * 4 + offset)))
      serial += 1
  lines.append("END")
  path = tmp_path / "toy.pdb"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  base = featurize(path)
  chi = np.full((4, 4), np.nan, dtype=np.float64)
  # Row 0 is ALA: no side chain, so no angles -- that is the point of the first
  # case. Rows 1-3 are LYS with four real angles each.
  for row in (1, 2, 3):
    chi[row] = _CHI
  return dataclasses.replace(
    base,
    sequence_indices=np.asarray([_A, _K, _K, _K], dtype=base.sequence_indices.dtype),
    chi_angles=chi.astype(base.chi_angles.dtype),
  )


def test_laser_score_chi_candidate_mismatch(features) -> None:  # noqa: ANN001
  """The two cases spec :800 names, plus the control that makes them mean something."""
  # Position 0 A->K, position 1 K->G, position 2 K->K (unchanged).
  candidate = np.asarray([_K, _G, _K, _K], dtype=np.int64)
  chi = _chi_for_candidate(features, candidate)

  # A->K: the native ALA has no side chain, so a lysine candidate cannot
  # inherit four angles from it. NaN in, NaN out.
  assert np.isnan(chi[0]).all(), chi[0]

  # K->G: the native LYS HAS four real angles and glycine must not keep them.
  # This is the case that fails if the rule is "pass the native chi through".
  assert np.isnan(chi[1]).all(), chi[1]

  # CONTROL: an UNCHANGED residue keeps its angles, exactly. Without this the
  # whole file is satisfied by `return np.full_like(chi, np.nan)`, which would
  # silently discard every native rotamer and make teacher-forced chi scoring
  # meaningless rather than merely wrong at mutated positions.
  np.testing.assert_array_equal(np.asarray(chi[2]), np.asarray(_CHI))


def test_chi_candidate_mismatch_does_not_depend_on_other_rows(features) -> None:  # noqa: ANN001
  """Each row is decided by its own letter.

  Pinned because a rule written with a scalar `if` rather than a per-row mask
  reads the same on a single-mutation fixture and collapses the whole array as
  soon as two rows disagree.
  """
  # Three mutations and one match, interleaved so a row-order mistake shows.
  candidate = np.asarray([_K, _G, _K, _G], dtype=np.int64)
  chi = _chi_for_candidate(features, candidate)
  assert np.isnan(chi[0]).all()
  assert np.isnan(chi[1]).all()
  np.testing.assert_array_equal(np.asarray(chi[2]), np.asarray(_CHI))
  assert np.isnan(chi[3]).all()

  # And the all-match case leaves every row untouched.
  native = np.asarray([_A, _K, _K, _K], dtype=np.int64)
  kept = _chi_for_candidate(features, native)
  assert np.isnan(kept[0]).all()
  for row in (1, 2, 3):
    np.testing.assert_array_equal(np.asarray(kept[row]), np.asarray(_CHI))


def test_chi_candidate_spec_and_implementation_disagree(features) -> None:  # noqa: ANN001
  """DIVERGENCE MARKER -- not an endorsement of either rule. See debt.

  Spec :800 (ii) states the rule per SLOT and keyed on the CANDIDATE's mask::

      chi[i,k] = input_chi[i,k]  iff  aa_to_chi_angle_mask[cand_i (X->G)][k]
                                      AND input_chi[i,k] is not NaN
                                 else NaN

  ``_chi_for_candidate`` instead keys on whether the LETTER CHANGED, and NaNs
  the whole row when it did. Its docstring gives the reason: "a mismatched
  candidate must not be scored with the native rotamer (that is a different
  sequence's side chain)", which is a sound argument.

  The two rules agree on every case the spec names, which is why this went
  unnoticed. Native A->K: ALA's chi is NaN, so both give NaN. Native K->G:
  glycine's mask is all False, so both give NaN.

  THEY DIVERGE when the candidate's mask allows the slot and the native's chi
  is real -- native K -> candidate R. Measured: both LYS and ARG carry
  ``[True, True, True, True]``. The spec rule would KEEP lysine's four angles
  and score arginine with them; the implementation returns NaN.

  This test asserts WHAT THE CODE DOES so the conflict is visible and a silent
  change to either side breaks a test rather than a wave. It is deliberately
  not named ``test_divergence_*``: that prefix is for aminx-vs-upstream
  divergences, and this is aminx-vs-its-own-spec. Resolve it by deciding which
  rule is right, then delete this test and fix whichever side loses.
  """
  candidate = np.asarray([_A, _R, _K, _K], dtype=np.int64)
  chi = _chi_for_candidate(features, candidate)

  # Current behaviour: the row is NaN because the letter changed, even though
  # ARG has four chi slots and the native LYS angles are real.
  assert np.isnan(chi[1]).all(), (
    "implementation changed: K->R now keeps the native chi, which is the SPEC "
    "rule. If that was deliberate, update spec :800 or delete this marker."
  )
  # The spec rule would have produced this instead:
  spec_rule_would_give = np.asarray(_CHI)
  assert not np.array_equal(np.asarray(chi[1]), spec_rule_would_give)
