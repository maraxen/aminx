# ruff: noqa: S101
"""PottsMPNN knob semantics.

The LASEr half of this directory covers ``LaserOptions``; every
``test_knob_semantics_*`` test was LASEr-side until now, so all 22
``PottsMPNNOptions`` fields were uncovered and
``tests/knob_gate/test_parity_ids_passed`` could not pass for any of them. This
file starts the Potts half, taking the knobs the spec pins to an exact upstream
anchor.

Each test pairs the aminx behaviour against a local oracle transcribed from the
pinned upstream (PottsMPNN ``0cb0a58``) and a negative control that must NOT
agree, so the test cannot pass by accident. Per ``~/.claude/rules/BATHOS.md``, a
positive control that can only pass is not a check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


from aminx.families.potts_mpnn.decode import floor_temperature
from aminx.families.potts_mpnn.featurize import _AA_N_1, _BACKBONE, _parse_biounits

_GAP_CHAR = _AA_N_1[20]


def _oracle_floor_temperature(temperature: float) -> float:
  """Upstream ``sample_seqs.py:38-39``, transcribed.

  Upstream rewrites the temperature in place before the draw::

      if cfg.inference.temperature == 0:
          cfg.inference.temperature = 1e-6

  and does the same for ``optimization_temperature``. The floor is not cosmetic:
  the value goes on to divide the logits, so leaving a literal zero there yields a
  degenerate distribution rather than the argmax the user asked for. That is the
  negative control below.
  """
  if temperature == 0:
    return 1e-6
  return float(temperature)


def test_knob_semantics_t0_floor() -> None:
  """``temperature`` and ``optimization_temperature`` floor 0 to 1e-6 before the draw.

  Spec 6.5, upstream ``sample_seqs.py:38-39``. The host applies this to every
  element of the ``temperatures`` axis and to ``optimization_temperature``
  separately (``sample_host.py:506-512``), so one semantic covers both knobs.

  Deliberately not parametrized: alias_map.toml rows cite nodeids, and a
  parametrized id carries the value in brackets, so adding a case would silently
  invalidate a ledger reference.
  """
  for value in (0.0, -0.0, 1e-6, 0.1, 1.0, 2.5):
    assert floor_temperature(value) == _oracle_floor_temperature(value), value


def test_knob_semantics_t0_floor_is_load_bearing() -> None:
  """Negative control: without the floor the draw degenerates instead of sharpening.

  Spec 2.3 reads ``temperature == 0`` as argmax. Dividing by a literal zero does
  not produce argmax -- it produces non-finite logits -- so a floor that returned
  0 unchanged would be a silently wrong answer, not a near-equivalent one. This
  pins why 1e-6 is the correct port and guards against the floor being removed as
  redundant.
  """
  logits = np.asarray([1.0, 3.0, 2.0], dtype=np.float64)

  floored = logits / floor_temperature(0.0)
  assert np.all(np.isfinite(floored))
  assert int(np.argmax(floored)) == int(np.argmax(logits))

  with np.errstate(divide="ignore", invalid="ignore"):
    unfloored = logits / 0.0
  assert not np.all(np.isfinite(unfloored)), (
    "dividing by an unfloored zero must be non-finite; if this ever becomes finite "
    "the negative control has stopped discriminating and this test is vacuous"
  )


def _write_pdb(path: Path, residues: tuple[tuple[int, str], ...]) -> Path:
  """Write one chain A with the given (residue number, residue name) pairs.

  The parser slices the fixed-width record directly, so the columns have to be
  exact. An earlier draft of this helper put resName at 16:19 -- one short of the
  real layout, which reserves column 16 for altLoc -- and that pushed chainID from
  21 to 20, so ``_parse_biounits`` matched no chain and returned ``'no_chain'``
  for every input. The layout below is spelled out to keep that from recurring::

      0-5 record  6-10 serial  11 blank  12-15 atom  16 altLoc  17-19 resName
      20 blank  21 chainID  22-25 resSeq  26 iCode  30-37 x  38-45 y  46-53 z
  """
  lines: list[str] = []
  serial = 1
  for resnum, resname in residues:
    for index, atom in enumerate(_BACKBONE):
      x = float(resnum) + index * 0.5
      lines.append(
        f"ATOM  {serial:>5} {atom:<4} {resname:>3} A{resnum:>4}    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00",
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def test_knob_semantics_skip_gaps(tmp_path: Path) -> None:
  """``skip_gaps`` drops unobserved residue numbers instead of emitting a gap row.

  Spec 6.1 / 6.3, upstream ``parse_PDB_biounits``. Unset, a number absent from the
  file still becomes a residue: alphabet slot 20 and all-NaN backbone coordinates,
  so the chain keeps its numbering. Set, the position is skipped and the chain
  shortens. The fixture numbers residues 1, 2 and 4, so exactly one gap exists.
  """
  pdb = _write_pdb(tmp_path / "gapped.pdb", ((1, "ALA"), (2, "GLY"), (4, "SER")))

  kept_coords, kept_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=False)
  skipped_coords, skipped_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=True)
  assert isinstance(kept_coords, np.ndarray)
  assert isinstance(skipped_coords, np.ndarray)

  # Unset: the gap is materialised as a residue.
  assert kept_seq == f"AG{_GAP_CHAR}S", kept_seq
  assert kept_coords.shape == (4, len(_BACKBONE), 3)
  assert np.all(np.isnan(kept_coords[2])), "the gap row's coordinates must be NaN"
  assert not np.any(np.isnan(kept_coords[[0, 1, 3]])), "observed rows must be finite"

  # Set: the gap is dropped and nothing else moves.
  assert skipped_seq == "AGS", skipped_seq
  assert skipped_coords.shape == (3, len(_BACKBONE), 3)
  assert not np.any(np.isnan(skipped_coords)), "no NaN row survives skip_gaps"
  np.testing.assert_array_equal(skipped_coords, kept_coords[[0, 1, 3]])

  # The knob is not a no-op: set and unset disagree in both outputs.
  assert kept_seq != skipped_seq
  assert kept_coords.shape != skipped_coords.shape


def test_knob_semantics_skip_gaps_inert_without_a_gap(tmp_path: Path) -> None:
  """Negative control: on contiguous numbering the knob must change nothing.

  A differential test that fired on every input would be measuring the parser
  rather than the knob, so the same comparison is run on a chain with no gap and
  must come out identical.
  """
  pdb = _write_pdb(
    tmp_path / "contiguous.pdb", ((1, "ALA"), (2, "GLY"), (3, "SER")),
  )

  kept_coords, kept_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=False)
  skipped_coords, skipped_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=True)
  assert isinstance(kept_coords, np.ndarray)
  assert isinstance(skipped_coords, np.ndarray)

  assert kept_seq == skipped_seq == "AGS"
  np.testing.assert_array_equal(kept_coords, skipped_coords)
  assert _GAP_CHAR not in kept_seq
