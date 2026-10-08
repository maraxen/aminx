"""Dump the B0 LASEr featurizer oracle from upstream LASErMPNN.

Run in the torch CPU oracle environment (this script does not import aminx)::

    python3 scripts/parity/dump_b0_featurize_oracle.py --laser-root <path> --out <dir>

``--laser-root`` is either the ``LASErMPNN`` package directory or a checkout
that contains it. Writes ``<out>/oracle.npz`` (every consumed ``BatchData``
field, secondary structure, ``row_to_resindex``, and ``construct_graphs``
outputs for ``num_adjacent_residues_to_drop`` in {0, 6}), ``manifest.json``,
and ``oracle.npz.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

ARRAY_FIELDS = (
  "sequence_indices",
  "chi_angles",
  "chi_mask",
  "backbone_coords",
  "phi_psi_angles",
  "heavy_atom_coords",
  "chain_indices",
  "chain_mask",
  "resnum_indices",
  "batch_indices",
  "extra_atom_contact_mask",
  "first_shell_ligand_contact_mask",
  "sidechain_contact_number",
  "residue_burial_counts",
  "sampled_chain_mask",
  "msa_depth_weight",
  "sc_mediated_hbond_counts",
  "ligand_coords",
  "ligand_atomic_numbers",
  "ligand_batch_indices",
  "ligand_subbatch_indices",
  "row_to_resindex",
  "ss_index",
  "msa_data",
)
JSON_FIELDS = (
  "pdb_code",
  "ss_code",
  "ligand_elements",
)
GRAPH_FIELDS = (
  "pr_pr_edge_index",
  "pr_pr_edge_distance",
  "lig_pr_edge_index",
  "lig_pr_edge_distance",
  "lig_lig_edge_index",
  "lig_lig_edge_distance",
  "ligand_nodes",
  "ligand_coords",
  "ligand_atomic_number_indices",
  "first_shell_ligand_contact_mask",
)
_GRAPH = {
  "pr_pr_knn_graph_k": 48,
  "lig_pr_distance_cutoff": 20.0,
  "lig_pr_knn_graph_k": 48,
  "lig_lig_knn_graph_k": 5,
}
_SS = {"-": 0, "H": 1, "E": 2}


def _package_dir(laser_root: Path) -> Path:
  root = laser_root.resolve()
  if (root / "utils" / "pdb_dataset.py").is_file():
    return root
  nested = root / "LASErMPNN"
  if (nested / "utils" / "pdb_dataset.py").is_file():
    return nested
  msg = f"{laser_root} does not contain LASErMPNN/utils/pdb_dataset.py"
  raise SystemExit(msg)


def _upstream_commit(package: Path) -> str:
  """Git HEAD of the checkout, else the ``VENDOR_PIN.toml`` commit (rsynced copies have no .git)."""
  pin_path = package / "VENDOR_PIN.toml"
  pin_commit = str(tomllib.loads(pin_path.read_text())["commit"]) if pin_path.is_file() else None
  try:
    head = subprocess.check_output(  # noqa: S603
      ["git", "-C", str(package), "rev-parse", "HEAD"],  # noqa: S607
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    head = None
  if head is None and pin_commit is None:
    msg = f"could not resolve an upstream commit under {package} (no git HEAD, no VENDOR_PIN.toml)"
    raise SystemExit(msg)
  if head is not None and pin_commit is not None and head != pin_commit:
    msg = f"LASErMPNN HEAD {head} != VENDOR_PIN commit {pin_commit}"
    raise SystemExit(msg)
  return head or str(pin_commit)


def _fixtures(package: Path, repo: Path) -> list[tuple[str, Path]]:
  example = package / "example_pdbs" / "4jnj-1_prot.pdb"
  if not example.is_file():
    msg = f"missing upstream example pdb: {example}"
    raise SystemExit(msg)
  manifest_path = repo / "tests" / "fixtures" / "laser" / "fixtures_manifest.toml"
  extras = tomllib.loads(manifest_path.read_text())["b0_extras"]
  fixtures = [("4jnj-1_prot", example)]
  for stem in extras:
    path = repo / "tests" / "fixtures" / "laser" / f"{stem}.pdb"
    if not path.is_file():
      msg = f"missing B0 fixture: {path}"
      raise SystemExit(msg)
    fixtures.append((str(stem), path))
  return fixtures


def _crystal_and_rows(hv: object, dataset_atom_order: dict[str, list[str]], aa_long_to_short: dict[str, str], is_amino_acid: object) -> tuple[np.ndarray, np.ndarray]:
  rows: list[int] = []
  crystals: list[np.ndarray] = []
  for chain in hv:  # type: ignore[attr-defined]
    for residue in chain:
      resname = str(residue.getResname())
      if resname == "HOH":
        continue
      letter = aa_long_to_short.get(resname, "X")
      if not is_amino_acid(residue):  # type: ignore[operator]
        continue
      order = dataset_atom_order[letter]
      coords = np.full((14, 3), np.nan, dtype=np.float64)
      for name, xyz in zip(residue.getNames(), residue.getCoords(), strict=False):
        if name in order:
          coords[order.index(str(name))] = xyz
      if np.isnan(coords[:3]).any():
        continue
      rows.append(int(residue.getResindex()))
      crystals.append(coords[[0, 1, 2, 3]])
  if not crystals:
    msg = "upstream featurizer kept no protein residues"
    raise SystemExit(msg)
  return np.stack(crystals), np.asarray(rows, dtype=np.int64)


def _tensor(value: object) -> np.ndarray:
  import torch  # noqa: PLC0415

  if torch.is_tensor(value):
    return value.detach().cpu().numpy()
  return np.asarray(value)


def _graphs(batch: object) -> dict[str, np.ndarray]:
  ligand = batch.ligand_data  # type: ignore[attr-defined]
  empty_long = np.zeros((0,), dtype=np.int64)
  empty_f = np.zeros((0,), dtype=np.float32)
  if ligand is None:
    return {
      "pr_pr_edge_index": _tensor(batch.pr_pr_edge_index),  # type: ignore[attr-defined]
      "pr_pr_edge_distance": _tensor(batch.pr_pr_edge_distance),  # type: ignore[attr-defined]
      "lig_pr_edge_index": _tensor(batch.lig_pr_edge_index),  # type: ignore[attr-defined]
      "lig_pr_edge_distance": _tensor(batch.lig_pr_edge_distance),  # type: ignore[attr-defined]
      "lig_lig_edge_index": np.zeros((2, 0), dtype=np.int64),
      "lig_lig_edge_distance": empty_f,
      "ligand_nodes": np.zeros((0, 0), dtype=np.float32),
      "ligand_coords": np.zeros((0, 3), dtype=np.float32),
      "ligand_atomic_number_indices": empty_long,
      "first_shell_ligand_contact_mask": _tensor(batch.first_shell_ligand_contact_mask),  # type: ignore[attr-defined]
    }
  return {
    "pr_pr_edge_index": _tensor(batch.pr_pr_edge_index),  # type: ignore[attr-defined]
    "pr_pr_edge_distance": _tensor(batch.pr_pr_edge_distance),  # type: ignore[attr-defined]
    "lig_pr_edge_index": _tensor(batch.lig_pr_edge_index),  # type: ignore[attr-defined]
    "lig_pr_edge_distance": _tensor(batch.lig_pr_edge_distance),  # type: ignore[attr-defined]
    "lig_lig_edge_index": _tensor(ligand.lig_lig_edge_index),
    "lig_lig_edge_distance": _tensor(ligand.lig_lig_edge_distance),
    "ligand_nodes": _tensor(ligand.lig_nodes),
    "ligand_coords": _tensor(ligand.lig_coords),
    "ligand_atomic_number_indices": _tensor(ligand.ligand_atomic_numbers),
    "first_shell_ligand_contact_mask": _tensor(batch.first_shell_ligand_contact_mask),  # type: ignore[attr-defined]
  }


def _featurize(path: Path, *, drop: int) -> dict[str, object]:
  import pydssp  # noqa: PLC0415
  from LASErMPNN.run_inference import (  # noqa: PLC0415
    ProteinComplexData,
    get_protein_hierview,
    is_amino_acid,
  )
  from LASErMPNN.utils.build_rotamers import RotamerBuilder  # noqa: PLC0415
  from LASErMPNN.utils.constants import aa_long_to_short, dataset_atom_order  # noqa: PLC0415
  from LASErMPNN.utils.ligand_featurization import LigandFeaturizer  # noqa: PLC0415

  view = get_protein_hierview(str(path))
  data = ProteinComplexData(view, str(path), verbose=False)
  crystal, resindex = _crystal_and_rows(view, dataset_atom_order, aa_long_to_short, is_amino_acid)
  batch = data.output_batch_data(fix_beta=False)
  batch.construct_graphs(
    RotamerBuilder(5.0),
    LigandFeaturizer(build_hydrogens=False),
    **_GRAPH,
    protein_training_noise=0.0,
    ligand_training_noise=0.0,
    subgraph_only_dropout_rate=0.0,
    num_adjacent_residues_to_drop=drop,
    build_hydrogens=False,
  )
  ligand = batch.unprocessed_ligand_input_data
  if ligand is None or ligand.lig_coords.numel() == 0:
    ligand_coords = np.zeros((0, 3), dtype=np.float32)
    ligand_z = np.zeros((0,), dtype=np.int64)
    ligand_batch = np.zeros((0,), dtype=np.int64)
    ligand_sub = np.zeros((0,), dtype=np.int64)
    elements: list[str] = []
  else:
    ligand_coords = _tensor(ligand.lig_coords).astype(np.float32)
    ligand_z = _tensor(ligand.lig_atomic_numbers).astype(np.int64)
    ligand_batch = _tensor(ligand.lig_batch_indices).astype(np.int64)
    ligand_sub = _tensor(ligand.lig_subbatch_indices).astype(np.int64)
    elements = [str(element) for element in data.ligand_info.atom_elements]
  chi = _tensor(batch.chi_angles).astype(np.float32)
  letters = [str(letter) for letter in pydssp.assign(crystal[None])[0]]
  fields: dict[str, object] = {
    "sequence_indices": _tensor(batch.sequence_indices).astype(np.int64),
    "chi_angles": chi,
    "chi_mask": ~np.isnan(chi),
    "backbone_coords": _tensor(batch.backbone_coords).astype(np.float32),
    "phi_psi_angles": _tensor(batch.phi_psi_angles).astype(np.float32),
    "heavy_atom_coords": _tensor(data.heavy_atom_coords).astype(np.float32),
    "chain_indices": _tensor(batch.chain_indices).astype(np.int64),
    "chain_mask": _tensor(batch.chain_mask).astype(bool),
    "resnum_indices": _tensor(batch.resnum_indices).astype(np.int64),
    "batch_indices": _tensor(batch.batch_indices).astype(np.int64),
    "extra_atom_contact_mask": _tensor(batch.extra_atom_contact_mask).astype(bool),
    "first_shell_ligand_contact_mask": _tensor(batch.first_shell_ligand_contact_mask).astype(bool),
    "sidechain_contact_number": _tensor(batch.sidechain_contact_number).astype(np.int64),
    "residue_burial_counts": _tensor(batch.residue_burial_counts).astype(np.int64),
    "sampled_chain_mask": _tensor(batch.sampled_chain_mask).astype(bool),
    "msa_depth_weight": _tensor(batch.msa_depth_weight).astype(np.float32),
    "sc_mediated_hbond_counts": _tensor(batch.sc_mediated_hbond_counts).astype(np.int64),
    "ligand_coords": ligand_coords,
    "ligand_atomic_numbers": ligand_z,
    "ligand_batch_indices": ligand_batch,
    "ligand_subbatch_indices": ligand_sub,
    "row_to_resindex": resindex,
    "ss_index": np.asarray([_SS[letter] for letter in letters], dtype=np.int64),
    "msa_data": _tensor(batch.msa_data).astype(np.int64),
    "pdb_code": path.name,
    "ss_code": letters,
    "ligand_elements": elements,
    "graphs": _graphs(batch),
  }
  return fields


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  package = _package_dir(args.laser_root)
  sys.path.insert(0, str(package.parent))
  repo = Path(__file__).resolve().parents[2]
  out = args.out
  out.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {}
  manifest_fixtures: list[dict[str, object]] = []
  names: list[str] = []
  for name, path in _fixtures(package, repo):
    pdb_bytes = path.read_bytes()
    digest = hashlib.sha256(pdb_bytes).hexdigest()
    fields = _featurize(path, drop=0)
    graphs6 = _featurize(path, drop=6)["graphs"]
    names.append(name)
    payload[f"{name}__pdb"] = np.asarray(pdb_bytes.decode("utf-8", "replace"))
    for key in ARRAY_FIELDS:
      payload[f"{name}__{key}"] = np.asarray(fields[key])
    for key in JSON_FIELDS:
      payload[f"{name}__{key}"] = np.asarray(json.dumps(fields[key]))
    graphs0 = fields["graphs"]
    assert isinstance(graphs0, dict)
    assert isinstance(graphs6, dict)
    for key in GRAPH_FIELDS:
      payload[f"{name}__drop0__{key}"] = np.asarray(graphs0[key])
      payload[f"{name}__drop6__{key}"] = np.asarray(graphs6[key])
    manifest_fixtures.append({"name": name, "path": str(path), "sha256": digest})
  manifest = {
    "upstream_commit": _upstream_commit(package),
    "fixtures": manifest_fixtures,
  }
  payload["fixture_names"] = np.asarray(names)
  payload["manifest"] = np.asarray(json.dumps(manifest))
  npz_path = out / "oracle.npz"
  np.savez(npz_path, **payload)
  npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
  manifest["npz_sha256"] = npz_sha
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  (out / "oracle.npz.sha256").write_text(npz_sha + "\n")


if __name__ == "__main__":
  main()
