# ruff: noqa: S101
"""LASEr sample PDB sink: vectorised full-atom build and proxide round trip."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
import zarr

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


from aminx.families.laser_mpnn.driver import LASErMPNN, LaserDriver  # noqa: E402
from aminx.host.family_driver import FAMILY_DRIVERS  # noqa: E402
from aminx.host.runner import sample  # noqa: E402
from aminx.run.options import LaserOptions  # noqa: E402
from aminx.run.spec_json import (  # noqa: E402
  run_specification_from_json,
  run_specification_to_json,
)
from aminx.run.specs import SamplingSpecification  # noqa: E402


@pytest.fixture
def registered() -> Iterator[LaserDriver]:
  driver = LaserDriver()
  FAMILY_DRIVERS.register("lasermpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("lasermpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, LASErMPNN(key=jax.random.PRNGKey(0)))
  return path


def _spec(pdb: Path, model: Path, **kwargs: object) -> SamplingSpecification:
  laser = kwargs.pop("laser", LaserOptions())
  kwargs.setdefault("num_samples", 2)
  kwargs.setdefault("temperature", (0.2, 0.7))
  kwargs.setdefault("random_seed", 3)
  return SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_test",
    model_local_path=model,
    laser=laser,  # type: ignore[arg-type]
    **kwargs,  # type: ignore[arg-type]
  )


def _alanine_chain(path: Path, count: int) -> None:
  lines: list[str] = []
  serial = 1
  # Five or more residues reach the alpha-hull, which needs a non-flat point set.
  corners = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
  for index in range(count):
    origin = corners[index] if index < len(corners) else (float(index), 1.0, float(index))
    for name, element, delta in (
      ("N", "N", 0.0),
      ("CA", "C", 1.5),
      ("C", "C", 2.5),
      ("O", "O", 2.2),
    ):
      lines.append(
        _atom(
          serial,
          name,
          "ALA",
          "A",
          index + 1,
          origin[0] + delta,
          origin[1],
          origin[2],
          element=element,
        ),
      )
      serial += 1
  path.write_text("\n".join(lines) + "\n")


def _b_factors(path: Path) -> list[float]:
  return [
    float(_columns(line)["b"])
    for line in path.read_text().splitlines()
    if line.startswith("ATOM")
  ]


def _identity_columns(path: Path) -> list[tuple[object, object, object]]:
  return [
    (row["chain"], row["resnum"], row["icode"])
    for line in path.read_text().splitlines()
    if line[:6].strip() in {"ATOM", "HETATM"}
    for row in (_columns(line),)
  ]


def _backbone_xyz(path: Path) -> np.ndarray:
  rows = [
    _columns(line)["xyz"]
    for line in path.read_text().splitlines()
    if line.startswith("ATOM") and _columns(line)["name"] in {"N", "CA", "C"}
  ]
  return np.asarray(rows, dtype=np.float64)


def _zarr_arrays(path: Path) -> dict[str, bytes]:
  root = zarr.open_group(path, mode="r")
  found: dict[str, bytes] = {}

  def _walk(node: object, prefix: str) -> None:
    names = list(node.array_keys()) + list(node.group_keys())  # type: ignore[attr-defined]
    for name in names:
      child = node[name]  # type: ignore[index]
      key = f"{prefix}/{name}"
      if hasattr(child, "shape"):
        found[key] = np.ascontiguousarray(np.asarray(child)).tobytes()
      else:
        _walk(child, key)

  _walk(root, "")
  return found


def test_pdb_dir_writes_global_rows_and_default_writes_nothing(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """Two samples and two temperatures write four row-indexed PDBs; the default writes none."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _alanine_chain(pdb, 4)
  out = tmp_path / "pdbs"
  result = sample(_spec(pdb, model_path, laser=LaserOptions(pdb_output_dir=str(out))))
  names = sorted(path.name for path in out.glob("*.pdb"))
  assert names == [f"complex_n{row:05d}.pdb" for row in range(4)]
  assert len(set(_b_factors(out / names[0]))) > 1
  assert result["structures"]["0"]["arrays"]["sequence"].shape[0] == 4

  bare = tmp_path / "bare"
  bare.mkdir()
  sample(_spec(pdb, model_path, output_dir=str(bare)))
  assert list(bare.glob("**/*.pdb")) == []
  assert not (tmp_path / "pdbs_absent").exists()


def test_fasta_only_rejects_pdb_dir_and_fasta_composes(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """FASTA-only plus a PDB directory fails at construction; FASTA composes with PDBs."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _alanine_chain(pdb, 2)
  with pytest.raises(ValueError, match="output_fasta_only"):
    LaserOptions(pdb_output_dir=str(tmp_path / "pdbs"), output_fasta_only=True)
  out = tmp_path / "with_fasta"
  both = sample(
    _spec(pdb, model_path, laser=LaserOptions(pdb_output_dir=str(out), output_fasta=True)),
  )
  assert sorted(out.glob("*.pdb"))
  assert "fasta" in both["structures"]["0"]["arrays"]
  neither = sample(_spec(pdb, model_path))
  assert "fasta" not in neither["structures"]["0"]["arrays"]


def test_sink_does_not_change_zarr_arrays(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """The zarr arrays are bit-equal with and without the PDB sink."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _alanine_chain(pdb, 2)
  off_path = tmp_path / "off.zarr"
  on_path = tmp_path / "on.zarr"
  sample(_spec(pdb, model_path, output_h5_path=off_path))
  sample(
    _spec(
      pdb,
      model_path,
      output_h5_path=on_path,
      laser=LaserOptions(pdb_output_dir=str(tmp_path / "pdbs")),
    ),
  )
  off = _zarr_arrays(off_path)
  on = _zarr_arrays(on_path)
  wanted = [
    key
    for key in off
    if key.endswith(("sequence", "seq_log_prob", "chi_deg", "sidechain_coords"))
  ]
  assert wanted
  for key in wanted:
    assert off[key] == on[key]


def test_noise_axis_writes_level_suffixes(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """Two noise levels get ``_d00`` and ``_d01`` files with shared identity and different backbones."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _alanine_chain(pdb, 2)
  out = tmp_path / "pdbs"
  sample(
    _spec(
      pdb,
      model_path,
      num_samples=1,
      temperature=(0.2,),
      backbone_noise=(0.0, 0.5),
      laser=LaserOptions(pdb_output_dir=str(out)),
    ),
  )
  quiet = out / "complex_n00000_d00.pdb"
  noised = out / "complex_n00000_d01.pdb"
  assert quiet.is_file() and noised.is_file()
  assert _identity_columns(quiet) == _identity_columns(noised)
  assert not np.allclose(_backbone_xyz(quiet), _backbone_xyz(noised))


def test_provenance_sidecar_follows_the_option(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """``sidecar`` writes ``<file>.pdb.prov.toml``; ``none`` does not."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _alanine_chain(pdb, 2)

  def _run(provenance: str, folder: str) -> Path:
    out = tmp_path / folder
    sample(
      _spec(
        pdb,
        model_path,
        num_samples=1,
        temperature=(0.2,),
        laser=LaserOptions(pdb_output_dir=str(out), pdb_provenance=provenance),  # type: ignore[arg-type]
      ),
    )
    return out / "complex_n00000.pdb"

  with_side = _run("sidecar", "side")
  assert Path(str(with_side) + ".prov.toml").is_file()
  without = _run("none", "plain")
  assert not Path(str(without) + ".prov.toml").exists()


def test_over_limit_is_skipped_and_disabled_sink_ignores_the_cap(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A structure over the PDB atom cap is skipped; the cap is unread when the sink is off."""
  del registered
  pdb = tmp_path / "five.pdb"
  _alanine_chain(pdb, 5)
  monkeypatch.setattr("aminx.families.laser_mpnn.pdb_sink._MAX_PDB_ATOMS", 100)
  skipped = sample(
    _spec(
      pdb,
      model_path,
      num_samples=1,
      temperature=(0.2,),
      laser=LaserOptions(pdb_output_dir=str(tmp_path / "pdbs")),
    ),
  )
  reasons = [item["reason"] for item in skipped["skipped_inputs"]]
  assert any(str(reason).startswith("pdb_sink:") for reason in reasons)
  assert skipped["structures"] == {}

  monkeypatch.undo()
  kept = sample(
    _spec(pdb, model_path, num_samples=1, temperature=(0.2,)),
  )
  assert kept["structures"]["0"]["arrays"]["sequence"].shape[1] == 5
  assert kept["skipped_inputs"] == []

  monkeypatch.setattr("aminx.families.laser_mpnn.pdb_sink._MAX_PDB_ATOMS", 100)
  untouched = sample(_spec(pdb, model_path, num_samples=1, temperature=(0.2,)))
  assert untouched["structures"]["0"]["arrays"]["sequence"].shape[1] == 5
  assert untouched["skipped_inputs"] == []


def test_pdb_options_round_trip_through_spec_json() -> None:
  """``pdb_output_dir`` and ``pdb_provenance`` survive the spec JSON loader."""
  spec = SamplingSpecification(
    inputs="a.pdb",
    model_family="lasermpnn",
    laser=LaserOptions(pdb_output_dir="x", pdb_provenance="sidecar"),
  )
  loaded = run_specification_from_json(run_specification_to_json(spec))
  assert loaded.laser == spec.laser
