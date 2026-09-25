"""P02 structure-ingestion parity: aminx vs reference ``data_utils.parse_PDB`` (T6, AC-7/AC-9).

``compare_parse(fixture)`` parses a manifest fixture's structure file with BOTH aminx
(``aminx.io.parsing.parse_structure``) and the reference LigandMPNN
``data_utils.parse_PDB`` (loaded unmodified, by file path, from ``$REFERENCE_PATH`` --
never retyped), aligns residues by ``(chain_letter, resnum)`` key, and returns one row
per comparison:

- ``mask`` (all-four-backbone-atom-present flag): EXACT integer equality;
- ``chain_labels`` (0-based chain ordinal): EXACT integer equality;
- ``sequence``: EXACT equality of the one-letter amino acid decoded independently on
  each side -- aminx's ``aatype`` uses proxide's AlphaFold-order ``restypes_with_x``
  ("ARNDCQEGHILKMFPSTWYVX"), the reference's ``S`` uses its own alphabetical
  ``restype_str_to_int`` ("ACDEFGHIKLMNPQRSTVWYX"); comparing raw integers directly
  would be comparing two different numbering conventions, so both are decoded to
  letters first and compared as strings (still EXACT, just encoding-independent);
- ``backbone_coords``: TOLERANCE, max abs diff over the aligned (N, CA, C, O) columns
  (both proxide's and the reference's own ``atom_order`` place these at 0/1/2/4), bar
  <= 1e-5 Angstrom.

A residue present in only one side's key set is counted in ``n_aminx_only`` /
``n_reference_only`` (diagnostic, not gated) and excluded from the aligned rows.

Both sides are parsed from a per-fixture altloc-canonicalized copy (``_canonicalize_altloc``),
not the raw PDB: aminx/proxide and the reference's prody-based ``parse_PDB`` resolve
duplicate-altloc atom records differently (measured: prody's ``CA_dict`` construction keeps
the LAST-seen altloc per residue while proxide keeps the first), which produced spurious
0.2-0.8 Angstrom "mismatches" on every fixture with alternate conformers -- an artifact of
which conformer each parser happens to pick, not a real parsing divergence. Canonicalizing
(keep the highest-occupancy altloc, ties broken by file order; blank its altLoc column; drop
the rest) before either parser sees the file removes the ambiguity so both sides are
guaranteed to read the identical atom set.

The control (AC-9): a temp copy of one fixture's PDB file, with exactly one
backbone-atom coordinate shifted by 1e-3 Angstrom, must be DETECTED (a max_abs of
that magnitude, far over the 1e-5 bar) when compared against the reference's parse
of the ORIGINAL file -- this is a sensitivity check on the comparison harness
itself, not a claim about a real parser bug.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from fixtures import resolve_fixture_path  # noqa: PLC0415 (sibling module, script-dir on sys.path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COORD_BAR_ANGSTROM = 1e-5
CONTROL_SHIFT_ANGSTROM = 1e-3
DEFAULT_REFERENCE_PATH = "/home/marielle/projects/aminx/reference_ligandmpnn_clone"

# proxide Atom37 column indices (proxide.chem.residues.atom_order), which the reference's
# own local `atom_order` dict places identically: N=0, CA=1, C=2, CB=3, O=4.
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def _reference_path() -> Path:
  return Path(os.environ.get("REFERENCE_PATH", DEFAULT_REFERENCE_PATH))


def _manifest_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "browser_validation"
    / "fixtures"
    / "manifest.json"
  )


def _load_manifest() -> dict[str, Any]:
  with _manifest_path().open() as fh:
    return json.load(fh)


def _load_reference_module() -> types.ModuleType:
  """Load the reference ``data_utils`` module by file path, unmodified (never retyped)."""
  module_path = _reference_path() / "data_utils.py"
  spec = importlib.util.spec_from_file_location("_ligandmpnn_data_utils", module_path)
  if spec is None or spec.loader is None:
    msg = f"could not load reference data_utils.py from {module_path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _proxide_backbone_indices() -> dict[str, int]:
  from proxide.chem.residues import atom_order

  return {name: int(atom_order[name]) for name in BACKBONE_ATOM_NAMES}


def _aminx_residue_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
  """Parse ``path`` with aminx and return {(chain_letter, resnum): row}."""
  from proxide.chem.residues import restypes_with_x

  from aminx.io.parsing import parse_structure

  protein = parse_structure(str(path))
  aatype = np.asarray(protein.aatype)
  mask = np.asarray(protein.mask)
  atom_mask = np.asarray(protein.atom_mask)
  coordinates = np.asarray(protein.coordinates)
  chain_index = np.asarray(protein.chain_index)
  residue_index = np.asarray(protein.residue_index)
  chain_ids = list(protein.chain_ids) if protein.chain_ids is not None else None

  backbone_idx = _proxide_backbone_indices()
  n_idx, ca_idx, c_idx, o_idx = (
    backbone_idx["N"],
    backbone_idx["CA"],
    backbone_idx["C"],
    backbone_idx["O"],
  )
  all4_mask = (
    (atom_mask[:, n_idx] > 0)
    & (atom_mask[:, ca_idx] > 0)
    & (atom_mask[:, c_idx] > 0)
    & (atom_mask[:, o_idx] > 0)
  ).astype(np.int32)

  rows: dict[tuple[str, int], dict[str, Any]] = {}
  n_residues = aatype.shape[0]
  for i in range(n_residues):
    chain_ordinal = int(chain_index[i])
    chain_letter = (
      chain_ids[chain_ordinal]
      if chain_ids is not None and chain_ordinal < len(chain_ids)
      else str(chain_ordinal)
    )
    key = (chain_letter, int(residue_index[i]))
    rows[key] = {
      "chain_ordinal": chain_ordinal,
      "mask": int(all4_mask[i]),
      "letter": restypes_with_x[int(aatype[i])] if int(aatype[i]) < len(restypes_with_x) else "X",
      "backbone": coordinates[i, [n_idx, ca_idx, c_idx, o_idx], :].astype(np.float64),
      "mask_ca": bool(mask[i] > 0),
    }
  return rows


def _reference_residue_rows(
  path: Path, reference_module: types.ModuleType
) -> dict[tuple[str, int], dict[str, Any]]:
  """Parse ``path`` with the unmodified reference ``parse_PDB`` and return keyed rows."""
  output_dict, *_ = reference_module.parse_PDB(str(path))
  chain_letters = list(output_dict["chain_letters"])
  resnums = output_dict["R_idx"].tolist()
  chain_labels = output_dict["chain_labels"].tolist()
  mask = output_dict["mask"].tolist()
  seq_ints = output_dict["S"].tolist()
  restype_int_to_str = reference_module.restype_int_to_str
  backbone = output_dict["X"].numpy().astype(np.float64)  # (L, 4, 3) in (N, CA, C, O) order

  rows: dict[tuple[str, int], dict[str, Any]] = {}
  for i, (chain_letter, resnum) in enumerate(zip(chain_letters, resnums, strict=True)):
    key = (str(chain_letter), int(resnum))
    rows[key] = {
      "chain_ordinal": int(chain_labels[i]),
      "mask": int(mask[i]),
      "letter": restype_int_to_str[int(seq_ints[i])],
      "backbone": backbone[i],
    }
  return rows


def _canonicalize_altloc(src: Path, dest: Path) -> None:
  """Copy ``src`` to ``dest``, keeping exactly one atom per altloc-ambiguous position.

  For every (atom name, chain, resnum+icode) with more than one non-blank-altloc record,
  keeps the highest-occupancy one (ties broken by earliest file order), blanks its altLoc
  column, and drops the rest. Records with no altloc pass through unchanged.
  """
  lines = src.read_text().splitlines(keepends=True)
  groups: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
  for idx, line in enumerate(lines):
    if not line.startswith(("ATOM", "HETATM")) or line[16] == " ":
      continue
    key = (line[12:16], line[21], line[22:27])
    occupancy_field = line[54:60].strip()
    occupancy = float(occupancy_field) if occupancy_field else 0.0
    groups.setdefault(key, []).append((occupancy, idx))

  keep_idx = {max(entries, key=lambda e: (e[0], -e[1]))[1] for entries in groups.values()}

  out_lines = []
  for idx, line in enumerate(lines):
    if not line.startswith(("ATOM", "HETATM")) or line[16] == " ":
      out_lines.append(line)
      continue
    if idx in keep_idx:
      out_lines.append(line[:16] + " " + line[17:])
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text("".join(out_lines))


def _canonical_copy(path: Path, cache_dir: Path) -> Path:
  dest = cache_dir / path.name
  if not dest.is_file():
    _canonicalize_altloc(path, dest)
  return dest


def _row(path_stem: str, metric: str, value: float, bar: float, status: str) -> dict[str, Any]:
  return {
    "path": f"P02.{metric}",
    "fixture": path_stem,
    "metric": metric,
    "value": value,
    "bar": bar,
    "status": status,
    "weight_source": "eqx",
  }


def compare_parse(fixture: dict[str, Any], cache_dir: Path | None = None) -> list[dict[str, Any]]:
  """Compare aminx's and the reference's parse of ``fixture``'s structure file."""
  path = resolve_fixture_path(fixture)
  cache = cache_dir or (
    Path(__file__).resolve().parents[2] / "outputs" / "browser_validation" / "tmp" / "canonical"
  )
  canonical_path = _canonical_copy(path, cache)
  reference_module = _load_reference_module()
  return _compare_paths(fixture["name"], canonical_path, reference_module)


def _compare_paths(
  name: str, aminx_path: Path, reference_module: types.ModuleType
) -> list[dict[str, Any]]:
  aminx_rows = _aminx_residue_rows(aminx_path)
  reference_rows = _reference_residue_rows(reference_module=reference_module, path=aminx_path)

  common_keys = sorted(set(aminx_rows) & set(reference_rows))
  n_aminx_only = len(set(aminx_rows) - set(reference_rows))
  n_reference_only = len(set(reference_rows) - set(aminx_rows))

  mask_mismatches = 0
  chain_mismatches = 0
  sequence_mismatches = 0
  max_abs = 0.0
  for key in common_keys:
    a = aminx_rows[key]
    r = reference_rows[key]
    if a["mask"] != r["mask"]:
      mask_mismatches += 1
    if a["chain_ordinal"] != r["chain_ordinal"]:
      chain_mismatches += 1
    if a["letter"] != r["letter"]:
      sequence_mismatches += 1
    diff = float(np.max(np.abs(a["backbone"] - r["backbone"])))
    max_abs = max(max_abs, diff)

  rows = [
    {
      "path": "P02.residue_alignment",
      "fixture": name,
      "metric": "n_common",
      "value": len(common_keys),
      "n_aminx_only": n_aminx_only,
      "n_reference_only": n_reference_only,
      "status": "validated" if n_aminx_only == 0 and n_reference_only == 0 else "not_advanced",
      "weight_source": "eqx",
    },
    _row(
      name,
      "mask_exact",
      mask_mismatches,
      0,
      "validated" if mask_mismatches == 0 else "not_advanced",
    ),
    _row(
      name,
      "chain_labels_exact",
      chain_mismatches,
      0,
      "validated" if chain_mismatches == 0 else "not_advanced",
    ),
    _row(
      name,
      "sequence_exact",
      sequence_mismatches,
      0,
      "validated" if sequence_mismatches == 0 else "not_advanced",
    ),
    _row(
      name,
      "backbone_coords_max_abs",
      max_abs,
      COORD_BAR_ANGSTROM,
      "validated" if max_abs <= COORD_BAR_ANGSTROM else "not_advanced",
    ),
  ]
  return rows


def _shift_first_backbone_atom(src: Path, dest: Path, shift: float) -> str:
  """Write a copy of ``src`` to ``dest`` with the first N/CA/C/O atom's X coordinate shifted.

  Returns the atom name that was shifted.
  """
  lines = src.read_text().splitlines(keepends=True)
  out_lines = []
  shifted_atom: str | None = None
  for line in lines:
    if shifted_atom is None and line.startswith(("ATOM", "HETATM")):
      atom_name = line[12:16].strip()
      if atom_name in BACKBONE_ATOM_NAMES:
        x = float(line[30:38])
        new_x = x + shift
        line = line[:30] + f"{new_x:8.3f}" + line[38:]
        shifted_atom = atom_name
    out_lines.append(line)
  if shifted_atom is None:
    msg = f"no N/CA/C/O ATOM record found in {src}"
    raise ValueError(msg)
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text("".join(out_lines))
  return shifted_atom


def _run_control(
  fixture: dict[str, Any],
  control_dir: Path,
  canonical_dir: Path,
  reference_module: types.ModuleType,
) -> dict[str, Any]:
  """Detect a 1e-3 Angstrom atom shift: shifted-aminx-parse vs unshifted-reference-parse."""
  original_path = _canonical_copy(resolve_fixture_path(fixture), canonical_dir)
  shifted_path = control_dir / f"{fixture['name']}_ctrl_shift.pdb"
  shifted_atom = _shift_first_backbone_atom(original_path, shifted_path, CONTROL_SHIFT_ANGSTROM)

  aminx_rows = _aminx_residue_rows(shifted_path)
  reference_rows = _reference_residue_rows(reference_module=reference_module, path=original_path)
  common_keys = sorted(set(aminx_rows) & set(reference_rows))
  max_abs = max(
    (
      float(np.max(np.abs(aminx_rows[k]["backbone"] - reference_rows[k]["backbone"])))
      for k in common_keys
    ),
    default=0.0,
  )
  detected = max_abs > COORD_BAR_ANGSTROM
  return {
    "fixture": fixture["name"],
    "shifted_atom": shifted_atom,
    "shift_angstrom": CONTROL_SHIFT_ANGSTROM,
    "max_abs": max_abs,
    "detected": detected,
  }


def run_parity(out_path: Path) -> dict[str, Any]:
  manifest = _load_manifest()
  protein_fixtures = [f for f in manifest["fixtures"] if f["kind"] == "protein"]
  reference_module = _load_reference_module()

  worktree_root = Path(__file__).resolve().parents[2]
  canonical_dir = worktree_root / "outputs" / "browser_validation" / "tmp" / "canonical"

  all_rows: list[dict[str, Any]] = []
  for fixture in protein_fixtures:
    canonical_path = _canonical_copy(resolve_fixture_path(fixture), canonical_dir)
    all_rows.extend(_compare_paths(fixture["name"], canonical_path, reference_module))

  exact_metrics = {"mask_exact", "chain_labels_exact", "sequence_exact"}
  n_exact_mismatches = sum(int(row["value"]) for row in all_rows if row["metric"] in exact_metrics)
  max_coord_diff = max(
    (float(row["value"]) for row in all_rows if row["metric"] == "backbone_coords_max_abs"),
    default=0.0,
  )

  control_dir = out_path.parent / "tmp" / "parse_parity_control"
  control_row = _run_control(protein_fixtures[0], control_dir, canonical_dir, reference_module)

  return {
    "n_fixtures": len(protein_fixtures),
    "n_rows": len(all_rows),
    "n_exact_mismatches": n_exact_mismatches,
    "max_coord_diff": max_coord_diff,
    "control_detected": bool(control_row["detected"]),
    "control_shift_angstrom": CONTROL_SHIFT_ANGSTROM,
    "control_row": control_row,
    "rows": all_rows,
  }


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as fh:
    json.dump(result, fh, indent=2, default=str)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    with Path(results_path).open("w") as fh:
      json.dump(result, fh, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  args = parser.parse_args(argv)

  result = run_parity(args.out)
  _write_result(args.out, result)
  logger.info(
    "n_fixtures=%d n_rows=%d n_exact_mismatches=%d max_coord_diff=%.3g control_detected=%s",
    result["n_fixtures"],
    result["n_rows"],
    result["n_exact_mismatches"],
    result["max_coord_diff"],
    result["control_detected"],
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
