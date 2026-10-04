# ruff: noqa: S101
"""LASEr PDB writer: heavy and protonated layouts, ids, B-factors, ligand HETATM."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

import prody as pr

from aminx.families.laser_mpnn.featurize import (
  LASER_ALPHABET,
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  featurize,
  residue_identifiers,
)
from aminx.families.laser_mpnn.sample_host import format_sample_pdb
from aminx.io import laser_pdb
from aminx.io.laser_pdb import LigandAtoms, write_laser_pdb

# Upstream ``dataset_atom_order`` (heavy) and the hydrogens appended in
# ``hydrogen_extended_dataset_atom_order``. ``X`` follows alanine.
_HEAVY: dict[str, tuple[str, ...]] = {
  "G": ("N", "CA", "C", "O"),
  "X": ("N", "CA", "C", "O"),
  "A": ("N", "CA", "C", "O", "CB"),
  "S": ("N", "CA", "C", "O", "CB", "OG", "HG"),
  "C": ("N", "CA", "C", "O", "CB", "SG", "HG"),
  "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "HG1"),
  "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
  "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
  "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
  "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
  "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
  "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
  "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
  "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
  "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
  "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
  "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "HD1", "HE2"),
  "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
  "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
  "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HH"),
  "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"),
}
_EXTRA_H: dict[str, tuple[str, ...]] = {
  "G": ("HA2", "HA3"),
  "A": ("HA", "HB1", "HB2", "HB3"),
  "S": ("HA", "HB2", "HB3"),
  "C": ("HA", "HB2", "HB3"),
  "T": ("HA", "HB", "HG21", "HG22", "HG23"),
  "P": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"),
  "V": ("HA", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"),
  "M": ("HA", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"),
  "N": ("HA", "HB2", "HB3", "HD21", "HD22"),
  "I": ("HA", "HB", "HG21", "HG22", "HG23", "HD11", "HD12", "HD13", "HG12", "HG13"),
  "L": ("HA", "HB2", "HB3", "HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"),
  "D": ("HA", "HB2", "HB3"),
  "E": ("HA", "HB2", "HB3", "HG2", "HG3"),
  "K": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"),
  "Q": ("HA", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"),
  "H": ("HA", "HB2", "HB3", "HE1", "HD2"),
  "F": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"),
  "R": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH21", "HH22", "HH12", "HH11"),
  "Y": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"),
  "W": ("HA", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"),
  "X": ("HA", "HB1", "HB2", "HB3"),
}
_RESNAME = {
  "A": "ALA",
  "R": "ARG",
  "N": "ASN",
  "D": "ASP",
  "C": "CYS",
  "E": "GLU",
  "Q": "GLN",
  "G": "GLY",
  "H": "HIS",
  "I": "ILE",
  "L": "LEU",
  "K": "LYS",
  "M": "MET",
  "F": "PHE",
  "P": "PRO",
  "S": "SER",
  "T": "THR",
  "W": "TRP",
  "Y": "TYR",
  "V": "VAL",
  "X": "XAA",
}


@dataclass(frozen=True, slots=True)
class _Atom:
  record: str
  name: str
  resname: str
  chain: str
  resnum: int
  icode: str
  xyz: tuple[float, float, float]
  occupancy: float
  bfactor: float
  element: str
  segname: str


def _columns(text: str) -> list[_Atom]:
  atoms: list[_Atom] = []
  for line in text.splitlines():
    record = line[:6].strip()
    if record not in {"ATOM", "HETATM"}:
      continue
    atoms.append(
      _Atom(
        record=record,
        name=line[12:16].strip(),
        resname=line[17:20].strip(),
        chain=line[21].strip(),
        resnum=int(line[22:26]),
        icode=line[26].strip(),
        xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
        occupancy=float(line[54:60]),
        bfactor=float(line[60:66]),
        element=line[76:78].strip(),
        segname=line[72:76].strip(),
      )
    )
  return atoms


def _load(text: str) -> list[_Atom]:
  """Column parse, checked against ProDy when that parser is importable."""
  rows = _columns(text)
  pr.confProDy(verbosity="none")
  with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
    handle.write(text)
    pdb_path = handle.name
  parsed = pr.parsePDB(pdb_path)
  Path(pdb_path).unlink()
  if not isinstance(parsed, pr.AtomGroup):
    msg = "prody parsePDBStream did not return an AtomGroup"
    raise AssertionError(msg)
  hetero = np.asarray(parsed.getFlags("hetatm"), dtype=bool)
  coords = np.asarray(parsed.getCoords(), dtype=np.float64)
  betas = np.asarray(parsed.getBetas(), dtype=np.float64)
  occupancies = np.asarray(parsed.getOccupancies(), dtype=np.float64)
  names = [str(item).strip() for item in parsed.getNames()]
  resnames = [str(item).strip() for item in parsed.getResnames()]
  chains = [str(item).strip() for item in parsed.getChids()]
  icodes = [str(item).strip() for item in parsed.getIcodes()]
  elements = [str(item).strip() for item in parsed.getElements()]
  resnums = [int(item) for item in parsed.getResnums()]
  assert len(names) == len(rows)
  for index, row in enumerate(rows):
    assert names[index] == row.name
    assert resnames[index] == row.resname
    assert chains[index] == row.chain
    assert resnums[index] == row.resnum
    assert icodes[index] == row.icode
    assert elements[index] == row.element
    assert bool(hetero[index]) == (row.record == "HETATM")
    parsed_xyz = tuple(float(item) for item in coords[index])
    assert parsed_xyz == pytest.approx(row.xyz, abs=5e-4)
    assert float(betas[index]) == pytest.approx(row.bfactor, abs=5e-3)
    assert float(occupancies[index]) == pytest.approx(row.occupancy, abs=5e-3)
  return rows


def _slot_xyz(residue: int, slot: int) -> np.ndarray:
  return np.array(
    [
      (residue + 1) * 0.5 + slot * 0.125,
      slot * -0.25,
      residue * 0.25 - slot * 0.125,
    ],
    dtype=np.float64,
  )


def _pack(
  letters: list[str],
  orders: list[tuple[str, ...]],
  width: int,
  drop: set[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]]]:
  coords = np.full((len(letters), width, 3), np.nan, dtype=np.float64)
  expected: list[tuple[str, str, np.ndarray]] = []
  for residue, (letter, names) in enumerate(zip(letters, orders, strict=True)):
    for slot, name in enumerate(names):
      xyz = _slot_xyz(residue, slot)
      if (residue, slot) in drop:
        continue
      coords[residue, slot] = xyz
      expected.append((name, _RESNAME[letter], xyz))
    for slot in range(len(names), width):
      coords[residue, slot] = np.array([3.5, 3.5, 3.5], dtype=np.float64)
  return coords, expected


def test_docstring_leaves_hydrogen_cleanup_open() -> None:
  """The pending decision, not this writer, owns backbone N-H and titratable H."""
  text = laser_pdb.__doc__ or ""
  assert "DECISION_261004_laser-pdb-output-scope.md" in text
  assert "backbone" in text
  assert "titratable" in text


def test_heavy_layout_roundtrip() -> None:
  """Heavy atoms round-trip to 3 decimals and follow dataset_atom_order."""
  letters = list(LASER_ALPHABET) + ["A"]
  orders = [_HEAVY[letter] for letter in letters]
  coords, expected = _pack(letters, orders, MAX_ATOMS, {(len(letters) - 1, 4)})
  sequence = np.array([LASER_ALPHABET.index(letter) for letter in letters], dtype=np.int64)
  text = write_laser_pdb(
    coords,
    sequence,
    ["A"] * len(letters),
    list(range(1, len(letters) + 1)),
    [""] * len(letters),
  )
  atoms = _load(text)
  assert [(atom.name, atom.resname) for atom in atoms] == [
    (name, resname) for name, resname, _xyz in expected
  ]
  for atom, (_name, _resname, xyz) in zip(atoms, expected, strict=True):
    assert atom.xyz == pytest.approx(tuple(float(item) for item in xyz), abs=5e-4)
    assert atom.element == atom.name[0]
    assert atom.record == "ATOM"
    assert atom.bfactor == pytest.approx(0.0)
  by_residue: dict[int, list[str]] = {}
  for atom in atoms:
    by_residue.setdefault(atom.resnum, []).append(atom.name)
  for index, letter in enumerate(LASER_ALPHABET):
    assert tuple(by_residue[index + 1]) == _HEAVY[letter]
  assert tuple(by_residue[len(letters)]) == ("N", "CA", "C", "O")


def test_protonated_layout_writes_non_nan_hydrogens() -> None:
  """Protonated layout writes exactly the finite hydrogen-extended names."""
  letters = list(LASER_ALPHABET)
  orders = [_HEAVY[letter] + _EXTRA_H[letter] for letter in letters]
  tryptophan = LASER_ALPHABET.index("W")
  dropped = len(orders[tryptophan]) - 1
  width = MAX_PROTONATED_ATOMS + 1
  coords, expected = _pack(letters, orders, width, {(tryptophan, dropped)})
  sequence = np.arange(len(letters), dtype=np.int64)
  text = write_laser_pdb(
    coords,
    sequence,
    ["A"] * len(letters),
    list(range(1, len(letters) + 1)),
    [""] * len(letters),
  )
  atoms = _load(text)
  assert [atom.name for atom in atoms] == [name for name, _resname, _xyz in expected]
  assert "H" not in {atom.name for atom in atoms}
  written = [atom.name for atom in atoms if atom.resnum == tryptophan + 1]
  assert written == list(orders[tryptophan][:-1])
  assert len(atoms) == sum(len(names) for names in orders) - 1


def test_identifiers_and_bfactors_use_pdb_columns() -> None:
  """Chain, residue number, insertion code, and B-factor sit in the PDB columns."""
  coords = np.full((2, MAX_ATOMS, 3), np.nan, dtype=np.float64)
  coords[0, :5] = np.array(
    [[0.125, 0.25, 0.5], [1.5, 0.0, 0.0], [2.5, 0.5, 0.0], [2.25, 1.5, 0.0], [1.5, -1.25, 0.5]],
    dtype=np.float64,
  )
  coords[1, :4] = np.array(
    [[4.0, 0.0, 0.0], [5.25, 0.5, 0.0], [6.5, 0.0, 0.25], [6.75, -1.0, 0.5]],
    dtype=np.float64,
  )
  sequence = np.array([0, LASER_ALPHABET.index("G")], dtype=np.int64)
  text = write_laser_pdb(
    coords,
    sequence,
    ["B", "C"],
    [42, -3],
    ["A", ""],
    bfactors=np.array([0.5, 1.25], dtype=np.float64),
  )
  lines = [line for line in text.splitlines() if line.startswith("ATOM")]
  assert len(lines) == 9
  for line in lines[:5]:
    assert line[21] == "B"
    assert int(line[22:26]) == 42
    assert line[26] == "A"
    assert float(line[60:66]) == pytest.approx(0.5)
    assert line[17:20] == "ALA"
    assert line[76:78] == f"{line[12:16].strip()[0]:>2}"
  for line in lines[5:]:
    assert line[21] == "C"
    assert int(line[22:26]) == -3
    assert line[26] == " "
    assert float(line[60:66]) == pytest.approx(1.25)
    assert line[17:20] == "GLY"
  atoms = _load(text)
  for atom in atoms[:5]:
    assert (atom.chain, atom.resnum, atom.icode) == ("B", 42, "A")
    assert atom.bfactor == pytest.approx(0.5)
  for atom in atoms[5:]:
    assert (atom.chain, atom.resnum, atom.icode) == ("C", -3, "")
    assert atom.bfactor == pytest.approx(1.25)
  omitted = write_laser_pdb(
    coords,
    sequence,
    ["B", "C"],
    [42, -3],
    ["A", ""],
  )
  for line in omitted.splitlines():
    if line.startswith("ATOM"):
      assert float(line[60:66]) == pytest.approx(0.0)


def test_ligand_passes_through_as_hetatm() -> None:
  """Ligand atoms are appended as HETATM with the caller's names and columns."""
  coords = np.full((1, MAX_ATOMS, 3), np.nan, dtype=np.float64)
  coords[0, :4] = np.array(
    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [2.25, 2.0, 0.25]],
    dtype=np.float64,
  )
  ligand_xyz = np.array([[8.5, 1.25, -0.5], [9.0, 2.5, 0.75]], dtype=np.float64)
  ligand = LigandAtoms(
    coords=ligand_xyz,
    names=("C1", "CL"),
    elements=("C", "Cl"),
    resnames=("LIG", "LIG"),
    chain_ids=("B", "B"),
    resnums=(7, 7),
    icodes=("", "A"),
    segnames=("LG", "LG"),
    occupancies=np.array([0.8, 1.0], dtype=np.float64),
    bfactors=np.array([11.5, 2.25], dtype=np.float64),
  )
  text = write_laser_pdb(
    coords,
    np.array([LASER_ALPHABET.index("G")], dtype=np.int64),
    ["A"],
    [1],
    [""],
    ligand=ligand,
  )
  atoms = _load(text)
  assert [atom.record for atom in atoms] == ["ATOM", "ATOM", "ATOM", "ATOM", "HETATM", "HETATM"]
  het = atoms[4:]
  assert [(atom.name, atom.element, atom.resname) for atom in het] == [
    ("C1", "C", "LIG"),
    ("CL", "Cl", "LIG"),
  ]
  assert het[0].xyz == pytest.approx((8.5, 1.25, -0.5), abs=5e-4)
  assert het[1].xyz == pytest.approx((9.0, 2.5, 0.75), abs=5e-4)
  assert (het[0].chain, het[0].resnum, het[0].icode) == ("B", 7, "")
  assert (het[1].chain, het[1].resnum, het[1].icode) == ("B", 7, "A")
  assert het[0].occupancy == pytest.approx(0.8)
  assert het[0].bfactor == pytest.approx(11.5)
  assert het[1].bfactor == pytest.approx(2.25)
  assert het[0].segname == "LG"
  raw = [line for line in text.splitlines() if line.startswith("HETATM")]
  assert raw[0][12:16] == " C1 "
  assert raw[0][76:78] == " C"
  assert raw[1][12:16] == " CL "
  assert raw[1][76:78] == "Cl"
  assert raw[0][72:76] == "  LG"


def test_featurizer_identifiers_reach_the_sample_helper(tmp_path: Path) -> None:
  """Chain, residue number, and insertion code come from the input PDB."""
  icode = "A"
  lines = [
    f"ATOM  {serial:5d} {name:>4} ALA B{15:4d}{icode}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2}"
    for serial, (name, element, x) in enumerate(
      (("N", "N", 0.0), ("CA", "C", 1.5), ("C", "C", 2.5), ("O", "O", 2.2), ("CB", "C", 1.5)),
      start=1,
    )
  ]
  path = tmp_path / "struct.pdb"
  path.write_text("\n".join(lines) + "\n")
  features = featurize(path)
  assert residue_identifiers(features) == (("B", 15, "A"),)
  coords = np.full((1, MAX_ATOMS, 3), np.nan, dtype=np.float64)
  coords[0, :4] = np.array(
    [[0.25, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.5, 0.0], [2.25, 1.5, 0.0]],
    dtype=np.float64,
  )
  text = format_sample_pdb(
    features,
    np.asarray(features.sequence_indices),
    coords,
    bfactors=np.array([0.25], dtype=np.float32),
  )
  atoms = _load(text)
  assert [atom.name for atom in atoms] == ["N", "CA", "C", "O"]
  for atom in atoms:
    assert (atom.chain, atom.resnum, atom.icode) == ("B", 15, "A")
    assert atom.bfactor == pytest.approx(0.25)
