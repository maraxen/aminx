"""Phase 1 fixture corpus, near-tie map and candidate tie-group builder (T6, AC-7).

``build_corpus(out_dir)`` assembles the browser-validation fixture corpus from the
LigandMPNN reference clone's named PDBs (``$REFERENCE_PATH``) and the pinned
ProteinMPNN clone's example PDBs (``$PROTEINMPNN_PATH``), plus a synthetic
tie-lattice fixture (``_lattice.cubic_lattice_ca``, genuine k-NN sort ties). For
every admitted protein it records:

- ``near_tie_residues`` per pre-registered ``k`` (48, 32) via
  ``aminx.parity.compare.no_tie_mask`` -- residues whose k-th/(k+1)-th CA neighbour
  distance gap is <= 1e-4 Angstrom, where an EXACT-tier comparison should not be
  evaluated;
- for multichain fixtures, candidate cross-chain homo-oligomer tie groups (one
  group per equivalent position across chains sharing an identical sequence), each
  checked for k-NN disjointness (R2-C3): a group only "qualifies"
  (``tie_groups_knn_disjoint``) if no member appears in another member's k=48 or
  k=32 neighbour set (``E_idx``, computed with the aminx featurizer's own
  primitives at noise 0); offending pairs are recorded under
  ``tie_groups_rejected``.

Fixtures are split deterministically (sort by sha256, alternate) into sets A/B,
each required to contain >= 1 multichain and >= 1 ligand-bearing fixture (swapped
in from the other set if the natural alternation misses either), and set B must
additionally clear the pre-registered floor of >= 2 qualifying multi-member tie
groups across >= 2 distinct fixtures (R2-C3/R3-C6) -- ``build_corpus`` admits
further multichain candidates or exits 2 with the shortfall rather than relaxing
the floor.

Every fixture path is recorded relative to a root token (``$REFERENCE_PATH``,
``$PROTEINMPNN_PATH``, or ``$WT``) so the manifest resolves unchanged on titanix;
``resolve_fixture_path`` is the paired loader that re-derives the path and
re-checks the recorded sha256 (reused by ``parse_parity.py``, T7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pre-registered v1 k-NN sizes (spec "fixture corpus" section).
K_VALUES: tuple[int, ...] = (48, 32)
NEAR_TIE_GAP = 1e-4
MIN_PROTEIN_L = 64
MIN_LARGE_L = 400
TIE_LATTICE_L = 96
SET_B_QUALIFYING_FIXTURE_FLOOR = 2
SET_B_QUALIFYING_GROUP_FLOOR = 2
MIN_LIGAND_BEARING = 2

REFERENCE_PDB_NAMES = ("1BC8", "2GFB", "4GYT")

DEFAULT_REFERENCE_PATH = "/home/marielle/projects/aminx/reference_ligandmpnn_clone"

# Fallback PDB IDs to fetch if the local candidate pool has < 2 ligand-bearing
# structures. Both are well-known, small, single-file, ligand-bearing entries.
FALLBACK_LIGAND_PDB_IDS = ("1HVR", "3PTB")

ROOT_TOKENS = ("$REFERENCE_PATH", "$PROTEINMPNN_PATH", "$WT")


def _reference_path() -> Path:
  return Path(os.environ.get("REFERENCE_PATH", DEFAULT_REFERENCE_PATH))


def _proteinmpnn_path() -> Path:
  default = Path(__file__).resolve().parents[2] / ".cache" / "reference" / "ProteinMPNN"
  return Path(os.environ.get("PROTEINMPNN_PATH", str(default)))


def _worktree_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _root_paths() -> dict[str, Path]:
  return {
    "$REFERENCE_PATH": _reference_path(),
    "$PROTEINMPNN_PATH": _proteinmpnn_path(),
    "$WT": _worktree_root(),
  }


def sha256_file(path: Path) -> str:
  """Return the sha256 hex digest of a file's bytes."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def root_token_path(path: Path) -> str:
  """Express ``path`` relative to $REFERENCE_PATH, $PROTEINMPNN_PATH, or $WT.

  Checked in that order (most specific first) since $PROTEINMPNN_PATH's default
  lives under $WT.
  """
  resolved = Path(path).resolve()
  for token in ROOT_TOKENS:
    root = _root_paths()[token].resolve()
    try:
      rel = resolved.relative_to(root)
    except ValueError:
      continue
    return f"{token}/{rel.as_posix()}"
  msg = f"{resolved} is not under REFERENCE_PATH, PROTEINMPNN_PATH, or the worktree"
  raise ValueError(msg)


def _resolve_root_token_path(token_path: str) -> Path:
  for token in ROOT_TOKENS:
    prefix = f"{token}/"
    if token_path.startswith(prefix):
      return _root_paths()[token] / token_path[len(prefix) :]
  msg = f"{token_path!r} does not start with a known root token {ROOT_TOKENS}"
  raise ValueError(msg)


def resolve_fixture_path(fixture: dict[str, Any]) -> Path:
  """Resolve a manifest fixture's ``path`` and re-check its recorded sha256.

  Reused by ``parse_parity.py`` (T7): every loader of a manifest fixture goes
  through this function rather than trusting the recorded path directly.
  """
  token_path = fixture["path"]
  resolved = _resolve_root_token_path(token_path)
  if not resolved.is_file():
    msg = f"fixture path does not resolve to a file: {token_path} -> {resolved}"
    raise FileNotFoundError(msg)
  actual_sha = sha256_file(resolved)
  if actual_sha != fixture["sha256"]:
    msg = (
      f"fixture {fixture.get('name', token_path)!r} sha256 mismatch: "
      f"manifest={fixture['sha256']} actual={actual_sha}"
    )
    raise ValueError(msg)
  return resolved


def discover_candidates() -> list[Path]:
  """Return the deduplicated (by sha256) candidate PDB pool, first occurrence wins."""
  reference = _reference_path()
  proteinmpnn = _proteinmpnn_path()
  ordered: list[Path] = [reference / "inputs" / f"{name}.pdb" for name in REFERENCE_PDB_NAMES]
  ordered += sorted(proteinmpnn.glob("inputs/**/*.pdb"))

  seen_sha: dict[str, Path] = {}
  deduped: list[Path] = []
  for path in ordered:
    if not path.is_file():
      logger.warning("candidate does not exist, skipping: %s", path)
      continue
    sha = sha256_file(path)
    if sha in seen_sha:
      logger.info("skipping duplicate of %s: %s", seen_sha[sha], path)
      continue
    seen_sha[sha] = path
    deduped.append(path)
  return deduped


def _ca_index() -> int:
  from proxide.chem.residues import atom_order

  return int(atom_order["CA"])


def _backbone_indices() -> tuple[int, int, int, int]:
  """Return (N, CA, C, O) proxide Atom37 column indices."""
  from proxide.chem.residues import atom_order

  return (
    int(atom_order["N"]),
    int(atom_order["CA"]),
    int(atom_order["C"]),
    int(atom_order["O"]),
  )


def _parse_fixture(path: Path) -> dict[str, Any]:
  from aminx.io.parsing import parse_structure

  protein = parse_structure(str(path))
  aatype = np.asarray(protein.aatype)
  mask = np.asarray(protein.mask, dtype=np.float64)
  chain_index = np.asarray(protein.chain_index)
  coordinates = np.asarray(protein.coordinates)
  molecule_type = None if protein.molecule_type is None else np.asarray(protein.molecule_type)
  ligand_heavy_atoms = 0 if molecule_type is None else int(np.sum(molecule_type == 1))

  return {
    "path": path,
    "sha256": sha256_file(path),
    "L": int(mask.shape[0]),
    "n_chains": int(np.unique(chain_index).shape[0]),
    "ligand_heavy_atoms": ligand_heavy_atoms,
    "aatype": aatype,
    "mask": mask,
    "chain_index": chain_index,
    "coordinates": coordinates,
  }


def _near_tie_residues(
  coordinates: np.ndarray,
  mask: np.ndarray,
  k_values: tuple[int, ...] = K_VALUES,
  gap: float = NEAR_TIE_GAP,
) -> dict[str, list[int] | None]:
  """Per-k near-tie residue indices (global, i.e. into the full fixture), CA-valid only.

  Residues with no valid CA (``mask <= 0``, zero-padded) are excluded before
  calling ``no_tie_mask`` -- their degenerate (zero) coordinates would otherwise
  register as spurious exact ties -- and results are mapped back to global indices.
  """
  from aminx.parity.compare import no_tie_mask

  ca_index = _ca_index()
  valid_idx = np.where(mask > 0)[0]
  ca_valid = coordinates[valid_idx, ca_index, :]
  n_valid = ca_valid.shape[0]

  result: dict[str, list[int] | None] = {}
  for k in k_values:
    if not (1 <= k < n_valid - 1):
      result[str(k)] = None
      continue
    ok = no_tie_mask(ca_valid, k=k, gap=gap)
    near_tie_local = np.where(~ok)[0]
    result[str(k)] = sorted(int(valid_idx[i]) for i in near_tie_local)
  return result


def _neighbor_indices(coordinates: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
  """E_idx: aminx featurizer's own k-NN selection at noise 0 (ProteinFeatures.forward_edge_stages).

  Uses the same primitives the real featurizer composes (``compute_backbone_coordinates``,
  ``compute_backbone_distance``, ``top_k`` on the mask-restricted, negated CA distance
  matrix) rather than instantiating the full ``eqx.Module`` (which would require an
  arbitrary PRNG-initialized weight set irrelevant to neighbour topology).
  """
  import jax.numpy as jnp

  from aminx.model.features import top_k as aminx_top_k
  from aminx.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance

  backbone = compute_backbone_coordinates(jnp.asarray(coordinates))
  distances = compute_backbone_distance(backbone)
  m = jnp.asarray(mask)
  valid_pair = (m[:, None] * m[None, :]) > 0
  distances_masked = jnp.where(valid_pair, distances, jnp.inf)
  k_eff = min(k, coordinates.shape[0] - 1)
  _, neighbor_indices = aminx_top_k(-distances_masked, k_eff)
  return np.asarray(neighbor_indices)


def _equivalence_classes(aatype: np.ndarray, chain_index: np.ndarray) -> list[list[int]]:
  """Chains sharing an identical (length + sequence) aatype -- candidate homo-oligomer sets."""
  chains = np.unique(chain_index)
  by_sequence: dict[tuple[int, ...], list[int]] = {}
  for chain in chains:
    sequence = tuple(int(v) for v in aatype[chain_index == chain].tolist())
    by_sequence.setdefault(sequence, []).append(int(chain))
  return [members for members in by_sequence.values() if len(members) >= 2]


def _candidate_tie_groups(chain_index: np.ndarray, equivalence_class: list[int]) -> list[list[int]]:
  """One group per equivalent local position across an equivalence class's chains."""
  chain_positions = {c: np.where(chain_index == c)[0] for c in equivalence_class}
  length = min(len(positions) for positions in chain_positions.values())
  return [[int(chain_positions[c][pos]) for c in equivalence_class] for pos in range(length)]


def _tie_group_report(
  chain_index: np.ndarray,
  aatype: np.ndarray,
  mask: np.ndarray,
  coordinates: np.ndarray,
  k_values: tuple[int, ...] = K_VALUES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Candidate cross-chain tie groups, split into qualifying (knn-disjoint) and rejected."""
  if np.unique(chain_index).shape[0] < 2:
    return [], []

  neighbor_indices_by_k = {k: _neighbor_indices(coordinates, mask, k) for k in k_values}

  qualifying: list[dict[str, Any]] = []
  rejected: list[dict[str, Any]] = []
  for equivalence_class in _equivalence_classes(aatype, chain_index):
    for group in _candidate_tie_groups(chain_index, equivalence_class):
      offending_pairs: list[list[int]] = []
      for k, neighbor_indices in neighbor_indices_by_k.items():
        for i in group:
          neighbors_i = set(neighbor_indices[i].tolist())
          for j in group:
            if i == j or j not in neighbors_i:
              continue
            pair = sorted((i, j))
            entry = {"pair": pair, "k": k}
            if entry not in offending_pairs:
              offending_pairs.append(entry)
      if offending_pairs:
        rejected.append({"members": group, "offending_pairs": offending_pairs})
      else:
        qualifying.append({"members": group, "designable_members": list(group)})
  return qualifying, rejected


def _fixture_record(path: Path) -> dict[str, Any]:
  parsed = _parse_fixture(path)
  near_tie = _near_tie_residues(parsed["coordinates"], parsed["mask"])
  qualifying, rejected = _tie_group_report(
    parsed["chain_index"],
    parsed["aatype"],
    parsed["mask"],
    parsed["coordinates"],
  )
  return {
    "name": path.stem,
    "kind": "protein",
    "path": root_token_path(path),
    "sha256": parsed["sha256"],
    "L": parsed["L"],
    "n_chains": parsed["n_chains"],
    "ligand_heavy_atoms": parsed["ligand_heavy_atoms"],
    "near_tie_residues": near_tie,
    "tie_groups_knn_disjoint": qualifying,
    "tie_groups_rejected": rejected,
  }


def _tie_lattice_fixture(n: int = TIE_LATTICE_L) -> dict[str, Any]:
  from _lattice import cubic_lattice_ca  # noqa: PLC0415

  from aminx.parity.compare import no_tie_mask  # noqa: PLC0415

  ca = np.asarray(cubic_lattice_ca(n), dtype=np.float32)
  sha = sha256_bytes(np.ascontiguousarray(ca).tobytes())

  near_tie: dict[str, list[int] | None] = {}
  for k in K_VALUES:
    if not (1 <= k < n - 1):
      near_tie[str(k)] = None
      continue
    ok = no_tie_mask(np.asarray(ca, dtype=np.float64), k=k, gap=NEAR_TIE_GAP)
    near_tie[str(k)] = sorted(int(i) for i in np.where(~ok)[0])

  return {
    "name": f"tie_lattice_L{n}",
    "kind": "tie_lattice",
    "path": f"$GENERATED/_lattice.cubic_lattice_ca(n={n})",
    "sha256": sha,
    "L": n,
    "n_chains": 1,
    "ligand_heavy_atoms": 0,
    "near_tie_residues": near_tie,
    "tie_groups_knn_disjoint": [],
    "tie_groups_rejected": [],
  }


def _fetch_ligand_bearing_fixtures(out_dir: Path, needed: int) -> list[dict[str, Any]]:
  """Fetch ``needed`` ligand-bearing PDB entries ONCE into ``out_dir/raw``, sha256-pinned.

  Only called when the local candidate pool has < 2 ligand-bearing structures.
  """
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  fetched: list[dict[str, Any]] = []
  for pdb_id in FALLBACK_LIGAND_PDB_IDS:
    if len(fetched) >= needed:
      break
    dest = raw_dir / f"{pdb_id}.pdb"
    if not dest.is_file():
      url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
      logger.info("fetching %s -> %s (ONE-TIME network fetch)", url, dest)
      try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
          data = response.read()
      except OSError as exc:
        logger.error("fetch failed for %s: %s", url, exc)
        continue
      dest.write_bytes(data)
      logger.info("pinned %s sha256=%s", dest, sha256_file(dest))
    record = _fixture_record(dest)
    if record["ligand_heavy_atoms"] > 0 and record["L"] >= MIN_PROTEIN_L:
      fetched.append(record)
  return fetched


def _swap_for(
  fixtures: list[dict[str, Any]],
  target_set: str,
  predicate: Any,
  protect: Any,
) -> bool:
  """Move the earliest-by-sha256 fixture satisfying ``predicate`` from the other set into
  ``target_set``, unless doing so would strip the other set below its own ``protect`` minimum.

  Returns True if a swap was made.
  """
  other_set = "B" if target_set == "A" else "A"
  other_members = sorted(
    (f for f in fixtures if f["set"] == other_set),
    key=lambda f: f["sha256"],
  )
  for candidate in other_members:
    if not predicate(candidate):
      continue
    remaining = [f for f in other_members if f is not candidate]
    if protect(candidate) and not any(protect(f) for f in remaining):
      # candidate is the OTHER set's sole qualifier for its own `protect` minimum; skip it.
      continue
    candidate["set"] = target_set
    return True
  return False


def _is_multichain_protein(fixture: dict[str, Any]) -> bool:
  return fixture["kind"] == "protein" and fixture["n_chains"] > 1


def _is_ligand_bearing(fixture: dict[str, Any]) -> bool:
  return fixture["ligand_heavy_atoms"] > 0


def _assign_sets(fixtures: list[dict[str, Any]]) -> None:
  """Sort by sha256, alternate A/B, then repair each set's own minimums by swapping."""
  ordered = sorted(fixtures, key=lambda f: f["sha256"])
  for i, fixture in enumerate(ordered):
    fixture["set"] = "A" if i % 2 == 0 else "B"

  for target_set in ("A", "B"):
    members = [f for f in ordered if f["set"] == target_set]
    if not any(_is_multichain_protein(f) for f in members):
      _swap_for(ordered, target_set, _is_multichain_protein, _is_multichain_protein)
    members = [f for f in ordered if f["set"] == target_set]
    if not any(_is_ligand_bearing(f) for f in members):
      _swap_for(ordered, target_set, _is_ligand_bearing, _is_ligand_bearing)

  for target_set in ("A", "B"):
    members = [f for f in ordered if f["set"] == target_set]
    if not any(_is_multichain_protein(f) for f in members):
      msg = f"set {target_set} has no multichain fixture after swap attempts"
      raise SystemExit(_shortfall(msg))
    if not any(_is_ligand_bearing(f) for f in members):
      msg = f"set {target_set} has no ligand-bearing fixture after swap attempts"
      raise SystemExit(_shortfall(msg))


def _shortfall(message: str) -> int:
  print(f"fixtures.py: shortfall -- {message}", file=sys.stderr)
  return 2


def _qualifying_group_stats(fixtures: list[dict[str, Any]], target_set: str) -> tuple[int, int]:
  members = [f for f in fixtures if f["set"] == target_set]
  qualifying_groups_per_fixture = [
    [g for g in f.get("tie_groups_knn_disjoint", []) if len(g["designable_members"]) >= 2]
    for f in members
  ]
  n_qualifying_fixtures = sum(1 for groups in qualifying_groups_per_fixture if groups)
  n_qualifying_groups = sum(len(groups) for groups in qualifying_groups_per_fixture)
  return n_qualifying_fixtures, n_qualifying_groups


def _enforce_set_b_tie_floor(fixtures: list[dict[str, Any]]) -> None:
  n_fixtures, n_groups = _qualifying_group_stats(fixtures, "B")
  if n_fixtures >= SET_B_QUALIFYING_FIXTURE_FLOOR and n_groups >= SET_B_QUALIFYING_GROUP_FLOOR:
    return

  # Try admitting further multichain candidates from set A that have qualifying groups,
  # provided set A keeps at least one multichain and one ligand-bearing fixture.
  candidates = sorted(
    (f for f in fixtures if f["set"] == "A" and _has_qualifying_group(f)),
    key=lambda f: f["sha256"],
  )
  for candidate in candidates:
    remaining_a = [f for f in fixtures if f["set"] == "A" and f is not candidate]
    if not any(_is_multichain_protein(f) for f in remaining_a):
      continue
    if not any(_is_ligand_bearing(f) for f in remaining_a):
      continue
    candidate["set"] = "B"
    n_fixtures, n_groups = _qualifying_group_stats(fixtures, "B")
    if n_fixtures >= SET_B_QUALIFYING_FIXTURE_FLOOR and n_groups >= SET_B_QUALIFYING_GROUP_FLOOR:
      return

  msg = (
    f"set B has {n_fixtures} qualifying fixtures / {n_groups} qualifying tie groups, need >= "
    f"{SET_B_QUALIFYING_FIXTURE_FLOOR} fixtures and >= {SET_B_QUALIFYING_GROUP_FLOOR} groups"
  )
  raise SystemExit(_shortfall(msg))


def _has_qualifying_group(fixture: dict[str, Any]) -> bool:
  return any(len(g["designable_members"]) >= 2 for g in fixture.get("tie_groups_knn_disjoint", []))


def build_corpus(out_dir: Path) -> dict[str, Any]:
  """Build, split, and validate the Phase-1 fixture corpus manifest."""
  candidates = discover_candidates()
  fixtures: list[dict[str, Any]] = []
  for path in candidates:
    record = _fixture_record(path)
    if record["L"] < MIN_PROTEIN_L:
      logger.info("dropping %s: L=%d < %d", path, record["L"], MIN_PROTEIN_L)
      continue
    fixtures.append(record)

  n_ligand_bearing = sum(1 for f in fixtures if _is_ligand_bearing(f))
  if n_ligand_bearing < MIN_LIGAND_BEARING:
    fixtures.extend(_fetch_ligand_bearing_fixtures(out_dir, MIN_LIGAND_BEARING - n_ligand_bearing))

  fixtures.append(_tie_lattice_fixture())

  if not any(f["kind"] == "protein" and f["L"] >= MIN_LARGE_L for f in fixtures):
    logger.warning("no candidate with L >= %d was available", MIN_LARGE_L)

  _assign_sets(fixtures)
  _enforce_set_b_tie_floor(fixtures)

  return {
    "created": datetime.now(UTC).strftime("%Y%m%d"),
    "root_tokens": {token: str(root) for token, root in _root_paths().items()},
    "k_values": list(K_VALUES),
    "near_tie_gap_angstrom": NEAR_TIE_GAP,
    "min_protein_l": MIN_PROTEIN_L,
    "fixtures": fixtures,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Output directory for manifest.json")
  args = parser.parse_args(argv)

  manifest = build_corpus(args.out)

  args.out.mkdir(parents=True, exist_ok=True)
  manifest_path = args.out / "manifest.json"
  with manifest_path.open("w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")

  n_a = sum(1 for f in manifest["fixtures"] if f["set"] == "A")
  n_b = sum(1 for f in manifest["fixtures"] if f["set"] == "B")
  logger.info(
    "wrote %s (%d fixtures: %d set A, %d set B)",
    manifest_path,
    len(manifest["fixtures"]),
    n_a,
    n_b,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
