# ruff: noqa: S101
"""LASEr sample PDB sink: vectorised full-atom build and proxide round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.featurize import (  # noqa: E402
  LASER_ALPHABET,
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  featurize,
  residue_atom_names,
)

proxide = pytest.importorskip("proxide")


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
  element: str,
  icode: str = " ",
  het: bool = False,
) -> str:
  record = "HETATM" if het else "ATOM  "
  return (
    f"{record}{serial:5d} {name:>4} {resname:>3} {chain}"
    f"{resseq:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
  )


def _coords_for(letters: str, *, protonated: bool) -> np.ndarray:
  width = MAX_PROTONATED_ATOMS if protonated else MAX_ATOMS
  coords = np.full((len(letters), width, 3), np.nan, dtype=np.float32)
  for row, letter in enumerate(letters):
    count = len(residue_atom_names(letter, protonated=protonated))
    slots = np.arange(count, dtype=np.float32)
    coords[row, :count, 0] = slots
    coords[row, :count, 1] = float(row)
    coords[row, :count, 2] = 0.5
  return coords


def _columns(line: str) -> dict[str, object]:
  return {
    "record": line[0:6].strip(),
    "name": line[12:16].strip(),
    "resname": line[17:20].strip(),
    "chain": line[21],
    "resnum": int(line[22:26]),
    "icode": line[26],
    "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
    "b": float(line[60:66]),
    "element": line[76:78].strip(),
  }


def test_builder_names_drop_nan_slots(tmp_path: Path) -> None:
  """Finite slots keep their dataset names; one extra NaN drops exactly one atom."""
  from aminx.families.laser_mpnn.pdb_sink import build_full_protein

  path = tmp_path / "one.pdb"
  path.write_text(
    _atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, element="N")
    + "\n"
    + _atom(2, "CA", "ALA", "A", 1, 1.5, 0.0, 0.0, element="C")
    + "\n"
    + _atom(3, "C", "ALA", "A", 1, 2.5, 1.0, 0.0, element="C")
    + "\n"
    + _atom(4, "O", "ALA", "A", 1, 2.2, 2.2, 0.0, element="O")
    + "\n",
  )
  features = featurize(path)
  for protonated in (False, True):
    width = MAX_PROTONATED_ATOMS if protonated else MAX_ATOMS
    for letter in LASER_ALPHABET:
      coords = _coords_for(letter, protonated=protonated)
      named = len(residue_atom_names(letter, protonated=protonated))
      index = np.array([LASER_ALPHABET.index(letter)], dtype=np.int64)
      built = build_full_protein(features, index, coords, np.array([0.5], dtype=np.float32))
      atom_names = [str(name) for name in built.atom_names]
      assert len(atom_names) == named
      assert "" not in atom_names
      assert [str(element) for element in built.elements] == [name[0] for name in atom_names]
      punched = np.array(coords, copy=True)
      punched[0, named - 1] = np.nan
      shorter = build_full_protein(features, index, punched, np.array([0.5], dtype=np.float32))
      assert len(shorter.atom_names) == named - 1
      assert int(np.isfinite(coords).all(-1).sum()) == named
      assert width == coords.shape[1]


def test_builder_broadcasts_probability_hetero_and_resnum(tmp_path: Path) -> None:
  """B-factors, HETATM flags, insertion codes and residue numbers follow the residue."""
  from aminx.families.laser_mpnn.pdb_sink import build_full_protein

  lines = [
    _atom(1, "N", "ALA", "A", 7, 0.0, 0.0, 0.0, element="N", icode="A"),
    _atom(2, "CA", "ALA", "A", 7, 1.5, 0.0, 0.0, element="C", icode="A"),
    _atom(3, "C", "ALA", "A", 7, 2.5, 1.0, 0.0, element="C", icode="A"),
    _atom(4, "O", "ALA", "A", 7, 2.2, 2.2, 0.0, element="O", icode="A"),
    _atom(5, "CB", "ALA", "A", 7, 1.5, -1.4, 0.5, element="C", icode="A"),
    _atom(6, "N", "HIS", "A", 9, 4.0, 1.0, 0.0, element="N"),
    _atom(7, "CA", "HIS", "A", 9, 5.2, 1.2, 0.0, element="C"),
    _atom(8, "C", "HIS", "A", 9, 6.0, 0.2, 0.0, element="C"),
    _atom(9, "O", "HIS", "A", 9, 5.8, -1.0, 0.0, element="O"),
    _atom(10, "C1", "LIG", "B", 1, 12.0, 0.0, 0.0, element="C", het=True),
  ]
  path = tmp_path / "pair.pdb"
  path.write_text("\n".join(lines) + "\n")
  features = featurize(path)
  letters = "AH"
  coords = _coords_for(letters, protonated=False)
  prob = np.array([0.25, 0.873], dtype=np.float32)
  index = np.array([LASER_ALPHABET.index(letter) for letter in letters], dtype=np.int64)
  built = build_full_protein(features, index, coords, prob)
  counts = np.isfinite(coords).all(-1).sum(1)
  expected_b = np.repeat(prob, counts)
  ligand = int((~features.ligand_synthetic).sum())
  protein_atoms = int(counts.sum())
  assert built.b_factors.shape == (protein_atoms + ligand,)
  assert np.allclose(built.b_factors[:protein_atoms], expected_b)
  assert np.all(built.b_factors[protein_atoms:] == 0.0)
  assert built.hetero.dtype == np.bool_
  assert built.hetero.tolist() == [False] * protein_atoms + [True] * ligand
  assert all(len(code) == 1 for code in built.insertion_codes)
  assert np.asarray(built.atom_res_index).tolist() == np.repeat(features.pdb_resnums, counts).tolist() + [
    int(features.ligand_resnums[0]),
  ]


def test_round_trip_through_proxide(tmp_path: Path) -> None:
  """Written columns and a Full parse agree with the built atoms."""
  from aminx.families.laser_mpnn.pdb_sink import build_full_protein, write_sample_pdb

  lines = [
    _atom(1, "N", "HIS", "A", 4, 0.0, 0.0, 0.0, element="N", icode="B"),
    _atom(2, "CA", "HIS", "A", 4, 1.5, 0.0, 0.0, element="C", icode="B"),
    _atom(3, "C", "HIS", "A", 4, 2.5, 1.0, 0.0, element="C", icode="B"),
    _atom(4, "O", "HIS", "A", 4, 2.2, 2.2, 0.0, element="O", icode="B"),
    _atom(5, "C1", "GOL", "B", 3, 8.0, 1.0, 0.0, element="C", het=True),
  ]
  path = tmp_path / "his.pdb"
  path.write_text("\n".join(lines) + "\n")
  features = featurize(path)
  coords = _coords_for("H", protonated=True)
  prob = np.array([0.873], dtype=np.float32)
  index = np.array([LASER_ALPHABET.index("H")], dtype=np.int64)
  built = build_full_protein(features, index, coords, prob)
  out = tmp_path / "sample.pdb"
  write_sample_pdb(
    features,
    index,
    coords,
    prob,
    out,
    structure_id="his",
    provenance="none",
  )
  rows = [_columns(line) for line in out.read_text().splitlines() if line[:6].strip() in {"ATOM", "HETATM"}]
  assert [row["name"] for row in rows] == [str(name) for name in built.atom_names]
  assert [row["resname"] for row in rows if row["record"] == "ATOM"] == ["HIS"] * sum(
    row["record"] == "ATOM" for row in rows
  )
  parsed = proxide.parse_structure(
    out,
    proxide.OutputSpec(coord_format=proxide.CoordFormat.Full),
    use_jax=False,
  )
  parsed_names = [str(name) for name in parsed.atom_names]
  parsed_res = ["HIS" if str(name) == "HIP" else str(name) for name in parsed.res_names]
  assert parsed_names == [str(name) for name in built.atom_names]
  assert parsed_res == [str(name) for name in built.res_names]
  assert np.allclose(np.asarray(parsed.full_coordinates), np.asarray(built.full_coordinates), atol=5e-4)
  assert np.asarray(parsed.hetero).tolist() == np.asarray(built.hetero).tolist()
  assert [row["chain"] for row in rows] == [str(chain) for chain in built.atom_chain_ids]
  assert [row["resnum"] for row in rows] == [int(num) for num in np.asarray(built.atom_res_index)]
  assert [row["icode"] for row in rows] == list(built.insertion_codes)
  assert all(abs(float(row["b"]) - float(factor)) < 0.005 for row, factor in zip(rows, built.b_factors, strict=True))
  assert sum(row["record"] == "HETATM" for row in rows) == int(features.ligand_coords.shape[0])
