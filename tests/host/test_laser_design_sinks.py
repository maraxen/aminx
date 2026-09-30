"""PDB and FASTA sinks honour the existing LaserOptions flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import prody as pr

from aminx.host.output_sinks import emit_laser_design, format_pdb_atom
from aminx.model.laser.rotamers import build_rotamers
from aminx.run.options import LaserOptions


def _toy_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """One alanine and one glycine, with a chi only where alanine has one."""
  backbone = np.zeros((2, 5, 3), dtype=np.float64)
  backbone[:, 0] = (0.0, 1.0, 0.0)
  backbone[:, 1] = (0.0, 0.0, 0.0)
  backbone[:, 2] = (1.0, 0.5, 0.2)
  backbone[:, 3] = (0.0, -1.0, 1.0)
  backbone[:, 4] = (1.0, -1.2, 1.4)
  chi = np.full((2, 4), np.nan, dtype=np.float64)
  sequence = np.array([0, 7], dtype=np.int64)
  return backbone, chi, sequence


def test_pdb_round_trip_keeps_coordinates(tmp_path: Path) -> None:
  """A parser must recover the coordinates the writer put in the fixed columns."""
  backbone, chi, sequence = _toy_frame()
  coords = build_rotamers(
    backbone,
    chi,
    sequence,
    dtype=np.float64,
    add_nonrotatable_hydrogens=True,
  )
  pdb_path, fasta_path = emit_laser_design(
    LaserOptions(),
    tmp_path,
    stem="design_0",
    coords=coords,
    sequence=sequence,
    fasta_header="toy",
  )
  assert pdb_path is not None and pdb_path.is_file()
  assert fasta_path is None
  parsed = pr.parsePDB(str(pdb_path))
  assert parsed is not None
  written = []
  for line in pdb_path.read_text(encoding="utf-8").splitlines():
    if not line.startswith("ATOM"):
      continue
    written.append(
      (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
      )
    )
  parsed_xyz = np.asarray(parsed.getCoords(), dtype=np.float64)
  np.testing.assert_allclose(parsed_xyz, np.asarray(written), atol=1e-3)
  finite = coords[~np.isnan(coords).any(axis=-1)]
  np.testing.assert_allclose(parsed_xyz, finite, atol=1e-3)
  # altLoc is blank and occupancy is the PDB 1.00 field, not a free-form token.
  atom = next(line for line in pdb_path.read_text(encoding="utf-8").splitlines() if line.startswith("ATOM"))
  assert atom[16] == " "
  assert atom[54:60] == "  1.00"


def test_output_fasta_only_skips_pdb(tmp_path: Path) -> None:
  backbone, chi, sequence = _toy_frame()
  coords = build_rotamers(backbone, chi, sequence, dtype=np.float32)
  pdb_path, fasta_path = emit_laser_design(
    LaserOptions(output_fasta_only=True),
    tmp_path,
    stem="design_0",
    coords=coords,
    sequence=sequence,
    fasta_header="toy_design_0_segment__chain_A score=0",
  )
  assert pdb_path is None
  assert fasta_path is not None
  text = fasta_path.read_text(encoding="utf-8")
  assert text.startswith(">toy_design_0_segment__chain_A score=0\n")
  assert "AG\n" in text
  assert not (tmp_path / "design_0.pdb").exists()


def test_output_fasta_writes_both(tmp_path: Path) -> None:
  backbone, chi, sequence = _toy_frame()
  coords = np.zeros((2, 14, 3), dtype=np.float32)
  coords[:, 4:] = np.nan
  pdb_path, fasta_path = emit_laser_design(
    LaserOptions(output_fasta=True),
    tmp_path,
    stem="design_1",
    coords=coords,
    sequence=sequence,
    fasta_header="both",
  )
  assert pdb_path is not None and pdb_path.is_file()
  assert fasta_path is not None
  assert fasta_path.read_text(encoding="utf-8").splitlines()[1] == "AG"


def test_atom_record_is_fixed_width() -> None:
  line = format_pdb_atom(1, "CA", "ALA", "A", 1, np.array([1.5, -2.25, 3.0]))
  assert len(line) == 78
  assert line.startswith("ATOM  ")
  assert line[12:16] == " CA "
  assert line[76:78] == " C"
