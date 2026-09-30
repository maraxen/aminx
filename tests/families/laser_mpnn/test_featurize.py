# ruff: noqa: S101
"""Synthetic checks for the LASEr host featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.featurize import (
  LASER_ALPHABET,
  LaserInputError,
  featurize,
  first_shell_contact_mask,
)


def _atom(
  serial: int,
  name: str,
  resname: str,
  chain: str,
  resseq: int,
  x: float,
  y: float,
  z: float,
  *,
  bfactor: float = 0.0,
  element: str = "C",
  het: bool = False,
) -> str:
  record = "HETATM" if het else "ATOM  "
  return (
    f"{record}{serial:5d} {name:>4} {resname:>3} {chain}"
    f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bfactor:6.2f}          {element:>2}"
  )


def _residue(
  serial: int,
  resname: str,
  chain: str,
  resseq: int,
  x0: float,
  *,
  atoms: tuple[tuple[str, str], ...] = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")),
  bfactor: float = 0.0,
  het: bool = False,
) -> list[str]:
  return [
    _atom(
      serial + offset,
      name,
      resname,
      chain,
      resseq,
      x0 + offset,
      0.0,
      0.0,
      bfactor=bfactor,
      element=element,
      het=het,
    )
    for offset, (name, element) in enumerate(atoms)
  ]


def _write(tmp_path: Path, lines: list[str]) -> Path:
  path = tmp_path / "struct.pdb"
  path.write_text("\n".join(lines) + "\n")
  return path


def _frame(ca: tuple[float, float, float], side: tuple[float, float, float]) -> np.ndarray:
  coords = np.full((1, 14, 3), np.nan, dtype=np.float32)
  coords[0, 1] = np.asarray(ca, dtype=np.float32)
  coords[0, 4] = np.asarray(side, dtype=np.float32)
  return coords


def test_first_shell_distance_cutoff() -> None:
  """A ligand heavy atom at 5.1 Å is outside the shell; 4.9 Å is inside."""
  sequence = np.array([LASER_ALPHABET.index("A")], dtype=np.int64)
  side = (0.0, 0.0, 0.0)
  ca = (0.0, 0.0, 0.0)
  numbers = np.array([6], dtype=np.int64)
  near = np.array([[4.9, 0.0, 0.0]], dtype=np.float32)
  far = np.array([[5.1, 0.0, 0.0]], dtype=np.float32)
  assert first_shell_contact_mask(_frame(ca, side), sequence, near, numbers)[0]
  assert not first_shell_contact_mask(_frame(ca, side), sequence, far, numbers)[0]


def test_first_shell_glycine_pad_and_empty_ligand() -> None:
  """Glycine uses Cα only, 0.3 Å past 5 Å. No ligand atoms leaves the mask false."""
  sequence = np.array([LASER_ALPHABET.index("G")], dtype=np.int64)
  coords = np.full((1, 14, 3), np.nan, dtype=np.float32)
  coords[0, 1] = np.zeros(3, dtype=np.float32)
  numbers = np.array([6], dtype=np.int64)
  marks = np.array([[5.2, 0.0, 0.0]], dtype=np.float32)
  misses = np.array([[5.4, 0.0, 0.0]], dtype=np.float32)
  assert first_shell_contact_mask(coords, sequence, marks, numbers)[0]
  assert not first_shell_contact_mask(coords, sequence, misses, numbers)[0]
  empty_xyz = np.zeros((0, 3), dtype=np.float32)
  empty_z = np.zeros((0,), dtype=np.int64)
  assert not first_shell_contact_mask(coords, sequence, empty_xyz, empty_z).any()


def test_first_shell_ignores_ligand_hydrogen() -> None:
  """Hydrogen is not a heavy atom, so a close H does not mark the shell."""
  sequence = np.array([LASER_ALPHABET.index("A")], dtype=np.int64)
  hydrogen = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
  mask = first_shell_contact_mask(
    _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    sequence,
    hydrogen,
    np.array([1], dtype=np.int64),
  )
  assert not mask[0]


def test_fix_from_bfactor_requires_zero_one(tmp_path: Path) -> None:
  """B-factors outside {0, 1} are rejected; a max of 1 marks the residue fixed."""
  ala = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))
  good = _write(
    tmp_path,
    [
      *_residue(1, "ALA", "A", 1, 0.0, atoms=ala, bfactor=1.0),
      *_residue(6, "GLY", "A", 2, 10.0, bfactor=0.0),
    ],
  )
  features = featurize(good, fix_from_bfactor=True)
  assert features.chain_mask.tolist() == [True, False]
  bad = tmp_path / "bad.pdb"
  bad.write_text(
    "\n".join(_residue(1, "ALA", "A", 1, 0.0, atoms=ala, bfactor=2.0)) + "\n",
  )
  try:
    featurize(bad, fix_from_bfactor=True)
  except LaserInputError:
    return
  msg = "B-factor 2 was accepted"
  raise AssertionError(msg)


def test_chi_nan_is_masked(tmp_path: Path) -> None:
  """Glycine has no chi angles, so every slot is NaN and the mask is false."""
  path = _write(tmp_path, _residue(1, "GLY", "A", 1, 0.0))
  features = featurize(path)
  assert features.sequence_indices.tolist() == [LASER_ALPHABET.index("G")]
  assert np.isnan(features.chi_angles).all()
  assert not features.chi_mask.any()


def test_water_and_row_to_resindex(tmp_path: Path) -> None:
  """Water is a ligand only when requested, and resindex counts the ligand row."""
  lines = [
    *_residue(1, "HOH", "A", 1, 0.0, atoms=(("O", "O"),), het=True),
    *_residue(2, "ALA", "A", 2, 20.0, atoms=(("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))),
  ]
  path = _write(tmp_path, lines)
  dry = featurize(path)
  wet = featurize(path, use_water=True)
  assert dry.ligand_coords.shape == (0, 3)
  assert not dry.first_shell_ligand_contact_mask.any()
  assert wet.ligand_atomic_numbers.tolist() == [8]
  assert dry.row_to_resindex.tolist() == [1]
  assert wet.row_to_resindex.tolist() == [1]


def test_noncanonical_residue_becomes_ligand(tmp_path: Path) -> None:
  """An X residue leaves the sequence and is parsed as ligand atoms."""
  lines = [
    *_residue(1, "ALA", "A", 1, 0.0, atoms=(("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))),
    *_residue(6, "UNK", "A", 2, 15.0),
  ]
  path = _write(tmp_path, lines)
  kept = featurize(path)
  ligand = featurize(path, noncanonical_aa_ligand=True)
  assert kept.sequence_indices.tolist() == [LASER_ALPHABET.index("A"), LASER_ALPHABET.index("X")]
  assert ligand.sequence_indices.tolist() == [LASER_ALPHABET.index("A")]
  assert ligand.ligand_coords.shape[0] > 0
  assert "N" in ligand.ligand_elements


def test_ignore_ligand_clears_the_shell(tmp_path: Path) -> None:
  """``ignore_ligand`` drops ligand atoms and the first-shell mask."""
  lines = [
    *_residue(1, "LIG", "A", 1, 0.0, atoms=(("C1", "C"),), het=True),
    *_residue(2, "ALA", "A", 2, 3.0, atoms=(("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))),
  ]
  path = _write(tmp_path, lines)
  present = featurize(path)
  ignored = featurize(path, ignore_ligand=True)
  assert present.ligand_coords.shape[0] == 1
  assert ignored.ligand_coords.shape == (0, 3)
  assert not ignored.first_shell_ligand_contact_mask.any()
