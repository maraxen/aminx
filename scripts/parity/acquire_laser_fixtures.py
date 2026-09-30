"""Acquire the 20 LASEr laser_score_parity PDB fixtures from Zenodo.

Streams the two upstream chunks with urllib, checks the documented archive md5,
selects complexes from the LASEr split JSON by reading PDBs inside the zip, and
writes ``tests/fixtures/laser/fixtures_manifest.toml``. Graded runs write the
results JSON to ``$BTH_RESULTS_PATH``.

``--dry-run`` queries the Zenodo records API and prints the plan.
``--selftest`` selects from a synthetic zip (no network, no archive download).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO

log = logging.getLogger("acquire_laser_fixtures")

EXPECTED_MD5 = "c9418cb9368c8068a6053feebbff5fda"
N_SCORE_PARITY = 20
N_B0_EXTRAS = 4
MIN_CA = 50
MAX_CA = 300
MIN_LIGAND_HEAVY = 6
DOWNLOAD_CHUNK = 8 * 1024 * 1024
HTTP_TIMEOUT_S = 120
USER_AGENT = "aminx-acquire-laser-fixtures/b1a"

RESULTS_KEYS = (
  "md5_ok",
  "n_selected",
  "n_written",
  "all_hashes_match",
  "archive_sha256",
)

# Chunk filenames copied from databases/README.md. The records API must report
# these exact keys; a mismatch is a hard failure.
ZENODO_CHUNKS: tuple[tuple[str, str], ...] = (
  ("17990180", "reduce_filtered_pdb_bioasmb_two_letter_bug_fixed.zip.part.aa"),
  ("17990253", "reduce_filtered_pdb_bioasmb_two_letter_bug_fixed.zip.part.ab"),
)

WATER_RESNAMES = frozenset({"DOD", "H2O", "HOH", "SOL", "TIP", "TIP3", "TIP4", "WAT"})
ION_RESNAMES = frozenset(
  {
    "BA",
    "BR",
    "CA",
    "CD",
    "CL",
    "CO",
    "CS",
    "CU",
    "F",
    "FE",
    "HG",
    "IOD",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NH4",
    "NI",
    "NO3",
    "PO4",
    "RB",
    "SO4",
    "SR",
    "ZN",
  },
)

SPLIT_CHOICE_NOTE = (
  "Prefer a split JSON whose filename contains 'test'. The pinned LASEr "
  "dataset_split_info.zip has none (probe finding B2; databases/README.md: "
  "LigandMPNN reconstruction is train-only, and the streptavidin held-out "
  "split ships train/val at 30% and 70% sequence identity). Held-out fallback: "
  "a filename containing both 'val' and 'held', preferring '30pct' main "
  "clusters, else the lexicographically first such file. If no test, val, or "
  "held-out JSON exists, use the lexicographically first JSON and set "
  "split_fallback."
)

SELECTION_RULE = (
  "Deterministic selection of 20 complexes for laser_score_parity: read the "
  "LASEr split json(s) from <laser-root>/databases/dataset_split_info.zip "
  "(prefer a held-out/test split if one exists, else the lexicographically "
  "first split json and record that fallback), keep entries whose PDB in the "
  "zip has >=1 non-water, non-ion HETATM ligand with >=6 heavy atoms, and "
  "50..300 protein residues (count CA atoms), sort assembly stems lexicographically, "
  "take the first 20 satisfying entries; entries 1-4 of that list are the B0 "
  "extras. "
  f"Water resnames: {', '.join(sorted(WATER_RESNAMES))}. "
  f"Ion resnames: {', '.join(sorted(ION_RESNAMES))}. "
  "A protein residue is one unique (chain, resSeq, iCode) with an ATOM CA "
  "(altLoc blank or A) in the first MODEL only. A heavy atom is a HETATM "
  "whose element is not H or D (altLoc blank or A), grouped by "
  "(chain, resSeq, iCode, resName). Split ids have the form "
  "<pdb>_<assembly>-<segment>-<chain> (cluster keys and members); each maps "
  "to its bioassembly stem (the id up to its first '-'), stems are deduplicated, "
  "and archive members <pdb>_<assembly>.pdb are matched by filename stem, so "
  "the selection unit is one whole bioassembly file. Selection reads zip "
  "members in-archive and does not extract the full dataset."
)


@dataclass(frozen=True)
class RemoteChunk:
  """One Zenodo chunk confirmed against the records API."""

  record_id: str
  filename: str
  url: str
  api_self: str
  size: int


@dataclass(frozen=True)
class PdbAssessment:
  """Ligand and CA counts for one PDB byte stream."""

  n_ca: int
  n_qualifying_ligands: int

  @property
  def ok(self) -> bool:
    """True when the complex meets the laser_score_parity residue and ligand rule."""
    return MIN_CA <= self.n_ca <= MAX_CA and self.n_qualifying_ligands >= 1


@dataclass(frozen=True)
class FixtureManifest:
  """Fields written to fixtures_manifest.toml."""

  source_urls: tuple[str, ...]
  zenodo_record_ids: tuple[str, ...]
  archive_md5: str
  archive_sha256: str
  archive_size: int
  selection_rule: str
  split_file: str
  split_name: str
  split_fallback: bool
  split_choice_note: str
  ordered_ids: tuple[str, ...]
  file_sha256: dict[str, str]
  b0_extras: tuple[str, ...]


def records_file_url(record_id: str, filename: str) -> str:
  """Return the Zenodo records download URL for one chunk filename."""
  return f"https://zenodo.org/records/{record_id}/files/{filename}"


def records_api_url(record_id: str) -> str:
  """Return the Zenodo records API URL used to confirm filename and size."""
  return f"https://zenodo.org/api/records/{record_id}"


def _toml_string(value: str) -> str:
  return json.dumps(value, ensure_ascii=True)


def _toml_string_array(values: tuple[str, ...] | list[str]) -> str:
  if not values:
    return "[]"
  lines = ["["]
  lines.extend(f"  {_toml_string(value)}," for value in values)
  lines.append("]")
  return "\n".join(lines)


def render_manifest(manifest: FixtureManifest) -> str:
  """Render a fixture manifest as TOML text."""
  files = "\n".join(
    f"  {_toml_string(name)} = {_toml_string(digest)}"
    for name, digest in sorted(manifest.file_sha256.items())
  )
  fallback = "true" if manifest.split_fallback else "false"
  return (
    f"source_urls = {_toml_string_array(manifest.source_urls)}\n"
    f"zenodo_record_ids = {_toml_string_array(manifest.zenodo_record_ids)}\n"
    f"archive_md5 = {_toml_string(manifest.archive_md5)}\n"
    f"archive_sha256 = {_toml_string(manifest.archive_sha256)}\n"
    f"archive_size = {manifest.archive_size}\n"
    f"selection_rule = {_toml_string(manifest.selection_rule)}\n"
    f"split_file = {_toml_string(manifest.split_file)}\n"
    f"split_name = {_toml_string(manifest.split_name)}\n"
    f"split_fallback = {fallback}\n"
    f"split_choice_note = {_toml_string(manifest.split_choice_note)}\n"
    f"ordered_ids = {_toml_string_array(manifest.ordered_ids)}\n"
    f"b0_extras = {_toml_string_array(manifest.b0_extras)}\n"
    "\n"
    "[files]\n"
    f"{files}\n"
  )


def collect_entry_ids(node: object) -> set[str]:
  """Collect every string id stored in a LASEr split JSON value."""
  found: set[str] = set()

  def walk(value: object) -> None:
    if isinstance(value, str):
      if value:
        found.add(value)
      return
    if isinstance(value, dict):
      for key, child in value.items():
        if isinstance(key, str) and key:
          found.add(key)
        walk(child)
      return
    if isinstance(value, list):
      for child in value:
        walk(child)

  walk(node)
  return found


def choose_split_member(names: list[str]) -> tuple[str, bool]:
  """Pick a split JSON member. The bool is True when the held-out/test preference fell through."""
  jsons = [name for name in names if Path(name).name.lower().endswith(".json")]
  if not jsons:
    msg = "dataset_split_info.zip contains no JSON split"
    raise FileNotFoundError(msg)

  def base(name: str) -> str:
    return Path(name).name.lower()

  def pick(candidates: list[str]) -> str:
    preferred = [name for name in candidates if "30pct" in base(name)]
    pool = preferred or candidates
    return sorted(pool, key=base)[0]

  tests = [name for name in jsons if "test" in base(name)]
  if tests:
    return pick(tests), False
  heldout_vals = [name for name in jsons if "val" in base(name) and "held" in base(name)]
  if heldout_vals:
    return pick(heldout_vals), False
  generic = [name for name in jsons if "val" in base(name) or "held" in base(name)]
  if generic:
    return pick(generic), False
  return pick(jsons), True


def _element(line: bytes) -> str:
  if len(line) >= 78:
    element = line[76:78].decode("ascii", errors="replace").strip().upper()
    if element:
      return element
  raw = line[12:16].decode("ascii", errors="replace")
  if len(raw) == 4 and raw[0].isalpha() and raw[1].isalpha() and raw[0] != " ":
    return raw[:2].upper()
  letters = "".join(ch for ch in raw if ch.isalpha())
  if not letters:
    return ""
  return letters[0].upper()


def assess_pdb(handle: IO[bytes]) -> PdbAssessment:
  """Count CA residues and qualifying HETATM ligands in one PDB stream."""
  ca_keys: set[tuple[str, str, str]] = set()
  ligand_heavy: dict[tuple[str, str, str, str], int] = defaultdict(int)
  seen_atom = False
  for raw in handle:
    line = raw.rstrip(b"\r\n")
    if line.startswith(b"ENDMDL") and seen_atom:
      break
    if len(line) < 27:
      continue
    record = line[:6]
    if record not in {b"ATOM  ", b"HETATM"}:
      continue
    altloc = line[16:17]
    if altloc not in {b" ", b"A"}:
      continue
    seen_atom = True
    chain = line[21:22].decode("ascii", errors="replace")
    resseq = line[22:26].decode("ascii", errors="replace")
    icode = line[26:27].decode("ascii", errors="replace")
    resname = line[17:20].decode("ascii", errors="replace").strip().upper()
    if record == b"ATOM  ":
      atom_name = line[12:16].decode("ascii", errors="replace").strip().upper()
      if atom_name == "CA":
        ca_keys.add((chain, resseq, icode))
      continue
    if resname in WATER_RESNAMES or resname in ION_RESNAMES:
      continue
    if _element(line) in {"H", "D"}:
      continue
    ligand_heavy[(chain, resseq, icode, resname)] += 1
  n_ligands = sum(1 for count in ligand_heavy.values() if count >= MIN_LIGAND_HEAVY)
  return PdbAssessment(n_ca=len(ca_keys), n_qualifying_ligands=n_ligands)


def index_pdb_members(archive: zipfile.ZipFile) -> dict[str, str]:
  """Map PDB filename stems to one archive member. Shorter paths win ties."""
  grouped: dict[str, list[str]] = defaultdict(list)
  for name in archive.namelist():
    if name.endswith("/"):
      continue
    base = Path(name).name
    if not base.lower().endswith(".pdb"):
      continue
    grouped[base[:-4]].append(name)
  resolved: dict[str, str] = {}
  for stem, members in grouped.items():
    resolved[stem] = sorted(members, key=lambda item: (len(item), item))[0]
    if len(members) > 1:
      log.warning("duplicate PDB stem %s; using %s", stem, resolved[stem])
  return resolved


def _resolve_member(index: dict[str, str], entry_id: str) -> str | None:
  if entry_id in index:
    return index[entry_id]
  folded = entry_id.casefold()
  hits = [member for stem, member in index.items() if stem.casefold() == folded]
  if len(hits) == 1:
    return hits[0]
  return None


def select_entries(
  archive: zipfile.ZipFile,
  entry_ids: set[str] | list[str],
  *,
  limit: int,
) -> list[str]:
  """Return the first ``limit`` lex-sorted ids whose in-archive PDB qualifies."""
  index = index_pdb_members(archive)
  chosen: list[str] = []
  missing = 0
  checked = 0
  for entry_id in sorted(set(entry_ids)):
    member = _resolve_member(index, entry_id)
    if member is None:
      missing += 1
      continue
    checked += 1
    with archive.open(member, "r") as handle:
      assessment = assess_pdb(handle)
    if assessment.ok:
      chosen.append(entry_id)
      log.info(
        "selected %s ca=%d ligands=%d",
        entry_id,
        assessment.n_ca,
        assessment.n_qualifying_ligands,
      )
      if len(chosen) >= limit:
        break
    elif checked % 100 == 0:
      log.info("checked %d pdbs, selected %d, missing %d", checked, len(chosen), missing)
  log.info("selection done checked=%d selected=%d missing=%d", checked, len(chosen), missing)
  return chosen


def load_split(laser_root: Path) -> tuple[str, str, bool, list[str]]:
  """Read the preferred split and return file, stem, fallback flag, and sorted ids."""
  split_zip = laser_root / "databases" / "dataset_split_info.zip"
  if not split_zip.is_file():
    msg = f"missing LASEr split zip: {split_zip}"
    raise FileNotFoundError(msg)
  with zipfile.ZipFile(split_zip) as archive:
    member, fallback = choose_split_member(archive.namelist())
    payload: object = json.loads(archive.read(member))
  entry_ids = collect_entry_ids(payload)
  ids = split_ids_to_stems(entry_ids)
  if not ids:
    msg = f"split {member} contained no entry ids"
    raise ValueError(msg)
  split_name = Path(member).name
  if split_name.lower().endswith(".json"):
    split_name = split_name[: -len(".json")]
  log.info(
    "split %s fallback=%s n_ids=%d n_assemblies=%d",
    member,
    fallback,
    len(entry_ids),
    len(ids),
  )
  return member, split_name, fallback, ids


def split_ids_to_stems(entry_ids: set[str] | list[str]) -> list[str]:
  """Map split ids ``<pdb>_<assembly>-<segment>-<chain>`` to sorted unique assembly stems.

  Archive members are ``<pdb>_<assembly>.pdb`` (one file per bioassembly), so the stem is the
  id up to its first ``-``.
  """
  return sorted({entry_id.split("-", 1)[0] for entry_id in entry_ids if entry_id})


def _urlopen(url: str) -> urllib.response.addinfourl:
  request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
  try:
    return urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S)  # noqa: S310
  except urllib.error.HTTPError as exc:
    msg = f"HTTP {exc.code} for {url}"
    raise SystemExit(msg) from exc
  except urllib.error.URLError as exc:
    msg = f"failed to fetch {url}: {exc.reason}"
    raise SystemExit(msg) from exc


def confirm_chunk(record_id: str, filename: str) -> RemoteChunk:
  """Confirm one chunk filename against the Zenodo records API and return its size."""
  expected_url = records_file_url(record_id, filename)
  api_url = records_api_url(record_id)
  with _urlopen(api_url) as response:
    payload: object = json.load(response)
  if not isinstance(payload, dict):
    msg = f"zenodo record {record_id} API payload is not an object"
    raise SystemExit(msg)
  files = payload.get("files")
  if not isinstance(files, list):
    msg = f"zenodo record {record_id} has no files list"
    raise SystemExit(msg)
  keys: list[str] = []
  match: dict[str, object] | None = None
  for item in files:
    if not isinstance(item, dict):
      continue
    key = item.get("key")
    if isinstance(key, str):
      keys.append(key)
    if key == filename and match is None:
      match = item
  if match is None:
    msg = (
      f"zenodo record {record_id} filename differs from {filename}; API keys={keys}; "
      f"expected url {expected_url}"
    )
    raise SystemExit(msg)
  size = match.get("size")
  if not isinstance(size, int) or size <= 0:
    msg = f"zenodo record {record_id} file {filename} has no positive size"
    raise SystemExit(msg)
  links = match.get("links")
  api_self = ""
  if isinstance(links, dict):
    self_link = links.get("self")
    if isinstance(self_link, str):
      api_self = self_link
  if api_self and filename not in api_self:
    msg = (
      f"zenodo record {record_id} self URL {api_self} does not contain filename {filename}; "
      f"expected url {expected_url}"
    )
    raise SystemExit(msg)
  log.info("confirmed %s size=%d url=%s", filename, size, expected_url)
  return RemoteChunk(
    record_id=record_id,
    filename=filename,
    url=expected_url,
    api_self=api_self,
    size=size,
  )


def confirm_chunks() -> tuple[RemoteChunk, ...]:
  """Confirm both documented Zenodo chunks."""
  return tuple(confirm_chunk(record_id, filename) for record_id, filename in ZENODO_CHUNKS)


def download_chunk(remote: RemoteChunk, dest: Path) -> None:
  """Download one chunk unless dest already has the API byte size. Range requests are not used."""
  if dest.is_file() and dest.stat().st_size == remote.size:
    log.info("skip %s (%d bytes match API size)", dest, remote.size)
    return
  if dest.is_file():
    log.info(
      "re-download %s (local %d != API %d); HTTP Range is not used",
      dest.name,
      dest.stat().st_size,
      remote.size,
    )
  dest.parent.mkdir(parents=True, exist_ok=True)
  partial = dest.with_name(dest.name + ".partial")
  written = 0
  next_log = DOWNLOAD_CHUNK
  with _urlopen(remote.url) as response, partial.open("wb") as handle:
    while True:
      block = response.read(DOWNLOAD_CHUNK)
      if not block:
        break
      handle.write(block)
      written += len(block)
      if written >= next_log:
        log.info("downloaded %s %d/%d", remote.filename, written, remote.size)
        next_log += 1024 * 1024 * 1024
  if written != remote.size:
    msg = f"size mismatch for {remote.filename}: wrote {written}, API size {remote.size}"
    raise SystemExit(msg)
  partial.replace(dest)
  log.info("wrote %s (%d bytes)", dest, written)


def concat_and_hash(parts: list[Path], dest: Path) -> tuple[str, str, int]:
  """Concatenate parts into dest, hashing md5 and sha256 in the same pass."""
  md5 = hashlib.md5(usedforsecurity=False)
  sha256 = hashlib.sha256()
  total = 0
  dest.parent.mkdir(parents=True, exist_ok=True)
  with dest.open("wb") as out:
    for part in parts:
      with part.open("rb") as src:
        while True:
          block = src.read(DOWNLOAD_CHUNK)
          if not block:
            break
          md5.update(block)
          sha256.update(block)
          out.write(block)
          total += len(block)
  return md5.hexdigest(), sha256.hexdigest(), total


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while True:
      block = handle.read(DOWNLOAD_CHUNK)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def extract_selected(
  archive: zipfile.ZipFile,
  entry_ids: list[str],
  out_dir: Path,
) -> dict[str, str]:
  """Extract selected PDBs and return output filename -> sha256."""
  index = index_pdb_members(archive)
  out_dir.mkdir(parents=True, exist_ok=True)
  hashes: dict[str, str] = {}
  for entry_id in entry_ids:
    if "/" in entry_id or entry_id in {"", ".", ".."}:
      msg = f"refusing unsafe entry id {entry_id!r}"
      raise ValueError(msg)
    member = _resolve_member(index, entry_id)
    if member is None:
      msg = f"selected id {entry_id} has no PDB member"
      raise FileNotFoundError(msg)
    filename = f"{entry_id}.pdb"
    dest = out_dir / filename
    digest = hashlib.sha256()
    with archive.open(member, "r") as src, dest.open("wb") as handle:
      while True:
        block = src.read(DOWNLOAD_CHUNK)
        if not block:
          break
        digest.update(block)
        handle.write(block)
    hashes[filename] = digest.hexdigest()
    log.info("extracted %s", dest)
  return hashes


def hashes_match(out_dir: Path, file_sha256: dict[str, str]) -> bool:
  """Re-hash written PDBs and compare them to the manifest map."""
  for filename, expected in file_sha256.items():
    path = out_dir / filename
    if not path.is_file():
      log.error("missing written file %s", path)
      return False
    actual = _sha256_file(path)
    if actual != expected:
      log.error("sha256 mismatch for %s", filename)
      return False
  return True


def _results_template(**overrides: object) -> dict[str, object]:
  payload: dict[str, object] = {
    "md5_ok": False,
    "n_selected": 0,
    "n_written": 0,
    "all_hashes_match": False,
    "archive_sha256": "",
  }
  payload.update(overrides)
  unexpected = set(payload) - set(RESULTS_KEYS)
  if unexpected:
    msg = f"results JSON has keys outside the sidecar schema: {sorted(unexpected)}"
    raise RuntimeError(msg)
  return payload


def write_results(payload: dict[str, object], work_dir: Path) -> None:
  """Write the graded results JSON."""
  env = os.environ.get("BTH_RESULTS_PATH")
  path = Path(env) if env else work_dir / "acquire_laser_fixtures_results.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  log.info("wrote results %s", path)


def _fail(payload: dict[str, object], work_dir: Path, message: str) -> SystemExit:
  log.error(message)
  write_results(payload, work_dir)
  return SystemExit(message)


def _delete_archive(paths: list[Path]) -> None:
  for path in paths:
    if path.is_file():
      path.unlink()
      log.info("deleted %s", path)


def format_atom_line(
  record: str,
  serial: int,
  atom_name: str,
  resname: str,
  chain: str,
  resseq: int,
  element: str,
) -> str:
  """Build one fixed-column PDB ATOM/HETATM line (used by --selftest and unit tests)."""
  name = f"{atom_name:>4}"[:4]
  return (
    f"{record:<6}{serial:5d} {name} {resname:>3} {chain}{resseq:4d}    "
    f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
  )


def synthetic_pdb(*, n_ca: int, ligands: tuple[tuple[str, tuple[str, ...]], ...]) -> bytes:
  """Build a minimal PDB with ``n_ca`` ALA CA atoms and HETATM ligands.

  Each ligand is ``(resname, elements)``. One HETATM is written per element.
  """
  lines: list[str] = []
  serial = 1
  for index in range(1, n_ca + 1):
    lines.append(format_atom_line("ATOM", serial, "CA", "ALA", "A", index, "C"))
    serial += 1
  for resname, elements in ligands:
    for element in elements:
      lines.append(format_atom_line("HETATM", serial, element, resname, "L", 1, element))
      serial += 1
  lines.append("END")
  return ("\n".join(lines) + "\n").encode("ascii")


def run_selftest() -> None:
  """Positive and negative selection control on a 3-PDB synthetic archive."""
  qualifying = "qual_1"
  water_only = "water_1"
  too_short = "short_1"
  split_ids = [
    f"{stem}-A-{chain}" for stem in (qualifying, water_only, too_short) for chain in "AB"
  ]
  pdbs = {
    qualifying: synthetic_pdb(n_ca=80, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),)),
    water_only: synthetic_pdb(n_ca=80, ligands=(("HOH", ("O", "O", "O", "O", "O", "O")),)),
    too_short: synthetic_pdb(n_ca=10, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),)),
  }
  if not assess_pdb(io.BytesIO(pdbs[qualifying])).ok:
    msg = "selftest positive control failed: qualifying PDB was rejected"
    raise SystemExit(msg)
  if assess_pdb(io.BytesIO(pdbs[water_only])).ok or assess_pdb(io.BytesIO(pdbs[too_short])).ok:
    msg = "selftest negative control failed: water-only or too-short PDB qualified"
    raise SystemExit(msg)
  with tempfile.TemporaryDirectory(prefix="laser-fixture-selftest-") as tmp:
    archive_path = Path(tmp) / "synthetic.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
      for stem, blob in pdbs.items():
        archive.writestr(f"pdbs/{stem[:2]}/{stem}.pdb", blob)
    with zipfile.ZipFile(archive_path) as archive:
      chosen = select_entries(archive, split_ids_to_stems(split_ids), limit=N_SCORE_PARITY)
  if chosen != [qualifying]:
    msg = f"selftest selection kept {chosen}, expected only {qualifying}"
    raise SystemExit(msg)
  log.info("selftest ok")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--work-dir",
    type=Path,
    default=None,
    help="Scratch dir for chunks and reconstructed.zip.",
  )
  parser.add_argument(
    "--laser-root",
    type=Path,
    default=None,
    help="LASErMPNN checkout (split zip lives here).",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Fixture directory, e.g. tests/fixtures/laser.",
  )
  parser.add_argument(
    "--keep-archive",
    action="store_true",
    help="Keep reconstructed.zip and the Zenodo parts after a successful extract.",
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate args, query the Zenodo API, and print the plan without downloading.",
  )
  mode.add_argument(
    "--selftest",
    action="store_true",
    help="Run the synthetic selection control and exit.",
  )
  return parser.parse_args()


def _require_run_args(args: argparse.Namespace) -> None:
  missing = [
    flag
    for flag, value in (
      ("--work-dir", args.work_dir),
      ("--laser-root", args.laser_root),
      ("--out", args.out),
    )
    if value is None
  ]
  if missing:
    msg = f"missing required arguments: {' '.join(missing)}"
    raise SystemExit(msg)
  laser_root = Path(args.laser_root)
  split_zip = laser_root / "databases" / "dataset_split_info.zip"
  if not split_zip.is_file():
    msg = f"missing LASEr split zip: {split_zip}"
    raise SystemExit(msg)


def run_dry_run(args: argparse.Namespace) -> None:
  """Query Zenodo sizes and print the acquisition plan."""
  _require_run_args(args)
  work_dir = Path(args.work_dir)
  out_dir = Path(args.out)
  laser_root = Path(args.laser_root)
  chunks = confirm_chunks()
  split_file, split_name, fallback, ids = load_split(laser_root)
  log.info("dry-run plan")
  log.info("work_dir=%s", work_dir)
  log.info("out=%s", out_dir)
  log.info("laser_root=%s", laser_root)
  log.info("expected_md5=%s", EXPECTED_MD5)
  log.info(
    "split_file=%s split_name=%s fallback=%s n_ids=%d",
    split_file,
    split_name,
    fallback,
    len(ids),
  )
  log.info("selection_rule=%s", SELECTION_RULE)
  for chunk in chunks:
    dest = work_dir / chunk.filename
    local = dest.stat().st_size if dest.is_file() else 0
    log.info(
      "chunk record=%s file=%s api_size=%d local_size=%d url=%s api_self=%s",
      chunk.record_id,
      chunk.filename,
      chunk.size,
      local,
      chunk.url,
      chunk.api_self,
    )
  log.info("download skipped (--dry-run)")


def run_acquire(args: argparse.Namespace) -> int:
  """Download, select, extract, and write the fixture manifest."""
  _require_run_args(args)
  work_dir = Path(args.work_dir)
  out_dir = Path(args.out)
  laser_root = Path(args.laser_root)
  work_dir.mkdir(parents=True, exist_ok=True)
  chunks = confirm_chunks()
  part_paths = [work_dir / chunk.filename for chunk in chunks]
  for chunk, dest in zip(chunks, part_paths, strict=True):
    download_chunk(chunk, dest)
  archive_path = work_dir / "reconstructed.zip"
  digest_md5, digest_sha256, archive_size = concat_and_hash(part_paths, archive_path)
  md5_ok = digest_md5 == EXPECTED_MD5
  log.info(
    "archive md5=%s sha256=%s size=%d md5_ok=%s",
    digest_md5,
    digest_sha256,
    archive_size,
    md5_ok,
  )
  if not md5_ok:
    raise _fail(
      _results_template(md5_ok=False, archive_sha256=digest_sha256),
      work_dir,
      f"md5 {digest_md5} != {EXPECTED_MD5}",
    )
  if not args.keep_archive:
    # The verified archive supersedes the parts; free ~50 GB before selection.
    _delete_archive(part_paths)
  split_file, split_name, fallback, ids = load_split(laser_root)
  with zipfile.ZipFile(archive_path) as archive:
    chosen = select_entries(archive, ids, limit=N_SCORE_PARITY)
    if len(chosen) != N_SCORE_PARITY:
      raise _fail(
        _results_template(
          md5_ok=True,
          n_selected=len(chosen),
          archive_sha256=digest_sha256,
        ),
        work_dir,
        f"selected {len(chosen)} complexes, expected {N_SCORE_PARITY}",
      )
    file_sha256 = extract_selected(archive, chosen, out_dir)
  manifest = FixtureManifest(
    source_urls=tuple(chunk.url for chunk in chunks),
    zenodo_record_ids=tuple(chunk.record_id for chunk in chunks),
    archive_md5=digest_md5,
    archive_sha256=digest_sha256,
    archive_size=archive_size,
    selection_rule=SELECTION_RULE,
    split_file=split_file,
    split_name=split_name,
    split_fallback=fallback,
    split_choice_note=SPLIT_CHOICE_NOTE,
    ordered_ids=tuple(chosen),
    file_sha256=file_sha256,
    b0_extras=tuple(chosen[:N_B0_EXTRAS]),
  )
  text = render_manifest(manifest)
  parsed = tomllib.loads(text)
  if parsed.get("ordered_ids") != list(chosen) or parsed.get("selection_rule") != SELECTION_RULE:
    msg = "fixtures manifest TOML round-trip failed"
    raise RuntimeError(msg)
  manifest_path = out_dir / "fixtures_manifest.toml"
  manifest_path.write_text(text, encoding="utf-8")
  log.info("wrote %s", manifest_path)
  matched = hashes_match(out_dir, file_sha256)
  payload = _results_template(
    md5_ok=True,
    n_selected=len(chosen),
    n_written=len(file_sha256),
    all_hashes_match=matched,
    archive_sha256=digest_sha256,
  )
  write_results(payload, work_dir)
  if not matched:
    msg = "re-hash of written PDBs did not match the manifest"
    raise SystemExit(msg)
  if not args.keep_archive:
    _delete_archive([archive_path, *part_paths])
  return 0


def main() -> int:
  """CLI entry point."""
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()
  if args.selftest:
    run_selftest()
    return 0
  if args.dry_run:
    run_dry_run(args)
    return 0
  return run_acquire(args)


if __name__ == "__main__":
  raise SystemExit(main())
