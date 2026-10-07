"""The LASEr fixture manifest must describe the fixtures that are actually there.

WHY THIS EXISTS. Spec B1(a) states its gate as "manifest with SHA-256s resolves
and files hash-match" (``.praxia/docs/specs/260929_pottsmpnn-lasermpnn-xtrax-
composition.md`` §8). As of 261006 that gate was satisfied in FACT -- 20 files,
20 hashes, all matching -- but enforced by nothing: nine scripts READ
``tests/fixtures/laser/fixtures_manifest.toml`` and none check it. A re-saved
PDB, a line-ending normalisation, or a stray extra file would shift the inputs
underneath every LASEr parity wave while the manifest kept asserting the old
provenance, and the waves would re-baseline onto the new bytes without anyone
being told.

These 20 complexes are the ``laser_score_parity`` set and the B0 featurizer
extras. They are selected by a recorded deterministic rule from a ~50 GB Zenodo
archive that is NOT in the repo, so the per-file hashes are the only link
between what the tests read and where it came from. If they stop matching, the
link is broken and no re-download can tell you which side moved.

WHAT IT REFUSES, and each is a way the link can break quietly:
  * a file whose content no longer hashes to its recorded value;
  * a file named in the manifest that is gone;
  * a .pdb present on disk and ABSENT from the manifest -- an untracked fixture
    a test could pick up with no provenance at all;
  * ``ordered_ids`` drifting from the file table, since that list is what the
    parity scripts iterate and a mismatch silently changes which complexes run;
  * malformed archive provenance (hash widths, a record id naming no URL),
    which would make the stated re-download path unfollowable.

Cheap enough to live in the ordinary suite: 4.3 MB, ~21 ms to hash all 20.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "laser"
_MANIFEST = _FIXTURES / "fixtures_manifest.toml"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


def _load() -> dict[str, Any]:
  return tomllib.loads(_MANIFEST.read_text(encoding="utf-8"))


def content_problems(files: Mapping[str, str], directory: Path) -> list[str]:
  """Every named file must exist and hash to exactly its recorded value."""
  problems: list[str] = []
  for name, recorded in sorted(files.items()):
    if not _HEX64.match(str(recorded)):
      problems.append(f"{name}: recorded hash {recorded!r} is not 64 lowercase hex digits")
      continue
    path = directory / name
    if not path.is_file():
      problems.append(f"{name}: named in the manifest but not on disk")
      continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != recorded:
      problems.append(
        f"{name}: content hashes to {got[:16]}..., manifest records {recorded[:16]}...",
      )
  return problems


def inventory_problems(files: Mapping[str, str], directory: Path) -> list[str]:
  """A fixture on disk with no manifest entry has no provenance at all."""
  untracked = sorted(p.name for p in directory.glob("*.pdb") if p.name not in files)
  return [
    f"{name}: present in tests/fixtures/laser/ but absent from the manifest [files] table"
    for name in untracked
  ]


def ordering_problems(manifest: Mapping[str, Any]) -> list[str]:
  """``ordered_ids`` is what the parity scripts iterate; it must match."""
  files = manifest["files"]
  ordered = [str(i) for i in manifest["ordered_ids"]]
  problems: list[str] = []
  if len(set(ordered)) != len(ordered):
    duplicated = sorted({i for i in ordered if ordered.count(i) > 1})
    problems.append(f"ordered_ids repeats {duplicated}")
  as_files = [f"{i}.pdb" for i in ordered]
  missing = sorted(set(as_files) - set(files))
  surplus = sorted(set(files) - set(as_files))
  if missing:
    problems.append(f"ordered_ids names {missing} with no entry in [files]")
  if surplus:
    problems.append(f"[files] contains {surplus} that ordered_ids never lists")
  return problems


def provenance_problems(manifest: Mapping[str, Any]) -> list[str]:
  """The archive fields are the recorded route back to the source."""
  problems: list[str] = []
  if not _HEX64.match(str(manifest.get("archive_sha256", ""))):
    problems.append("archive_sha256 is not 64 lowercase hex digits")
  if not _HEX32.match(str(manifest.get("archive_md5", ""))):
    problems.append("archive_md5 is not 32 lowercase hex digits")
  size = manifest.get("archive_size")
  if not isinstance(size, int) or size <= 0:
    problems.append(f"archive_size {size!r} is not a positive integer")
  urls = [str(u) for u in manifest.get("source_urls", [])]
  if not urls:
    problems.append("source_urls is empty, so the archive has no recorded origin")
  for record in (str(r) for r in manifest.get("zenodo_record_ids", [])):
    if not any(record in url for url in urls):
      problems.append(f"zenodo_record_ids lists {record!r} but no source_url mentions it")
  if not str(manifest.get("split_file", "")).strip():
    problems.append("split_file is empty; the selection rule's input is unrecorded")
  if not isinstance(manifest.get("split_fallback"), bool):
    problems.append(
      "split_fallback must be a bool: it records whether the preferred split was found",
    )
  return problems


def test_every_fixture_matches_its_recorded_hash() -> None:
  """Spec B1(a)'s gate, stated as an assertion instead of as prose."""
  problems = content_problems(_load()["files"], _FIXTURES)
  assert not problems, (
    "the LASEr fixtures no longer match their recorded hashes, so every parity "
    "wave reading them is running on inputs the manifest does not describe:\n  "
    + "\n  ".join(problems)
  )


def test_no_untracked_fixture() -> None:
  problems = inventory_problems(_load()["files"], _FIXTURES)
  assert not problems, (
    "a fixture with no manifest entry can be read by a test with no provenance:\n  "
    + "\n  ".join(problems)
  )


def test_ordered_ids_match_the_file_table() -> None:
  problems = ordering_problems(_load())
  assert not problems, "ordered_ids and [files] disagree:\n  " + "\n  ".join(problems)


def test_archive_provenance_is_wellformed() -> None:
  problems = provenance_problems(_load())
  assert not problems, (
    "the recorded route back to the source archive is unfollowable:\n  " + "\n  ".join(problems)
  )


def test_the_checkers_actually_refuse(tmp_path: Path) -> None:
  """Negative controls on synthetic fixtures. A check that only passes is not one.

  Built on a scratch directory rather than by corrupting the real fixtures, so
  the controls are permanent and run everywhere instead of living in a
  throwaway script a reader never sees.
  """
  good = tmp_path / "a_1.pdb"
  good.write_bytes(b"ATOM  fixture a\n")
  digest = hashlib.sha256(good.read_bytes()).hexdigest()
  files = {"a_1.pdb": digest}

  # the baseline must be clean, or every control below is meaningless
  assert content_problems(files, tmp_path) == []
  assert inventory_problems(files, tmp_path) == []

  # 1. content changed under a recorded hash -- the case this file exists for
  good.write_bytes(b"ATOM  fixture a EDITED\n")
  changed = content_problems(files, tmp_path)
  assert len(changed) == 1 and "content hashes to" in changed[0]
  good.write_bytes(b"ATOM  fixture a\n")

  # 2. a manifest entry with no file
  assert content_problems({**files, "gone_1.pdb": digest}, tmp_path) == [
    "gone_1.pdb: named in the manifest but not on disk",
  ]

  # 3. a hash that is not a hash (truncated, uppercase, or prose)
  for bad in (digest[:16], digest.upper(), "not-a-hash"):
    assert content_problems({"a_1.pdb": bad}, tmp_path), f"accepted {bad!r} as a hash"

  # 4. a file on disk that the manifest never mentions
  (tmp_path / "stowaway_1.pdb").write_bytes(b"ATOM  untracked\n")
  assert inventory_problems(files, tmp_path) == [
    "stowaway_1.pdb: present in tests/fixtures/laser/ but absent from the manifest [files] table",
  ]

  # 5. ordered_ids drifting from [files], in both directions, plus duplicates
  assert ordering_problems({"files": files, "ordered_ids": ["a_1", "b_1"]})
  assert ordering_problems({"files": {**files, "b_1.pdb": digest}, "ordered_ids": ["a_1"]})
  assert ordering_problems({"files": files, "ordered_ids": ["a_1", "a_1"]})
  assert ordering_problems({"files": files, "ordered_ids": ["a_1"]}) == []

  # 6. provenance that cannot be followed back
  sound = {
    "archive_sha256": digest,
    "archive_md5": "0" * 32,
    "archive_size": 1,
    "source_urls": ["https://zenodo.org/records/123/files/x"],
    "zenodo_record_ids": ["123"],
    "split_file": "s.json",
    "split_fallback": False,
  }
  assert provenance_problems(sound) == []
  assert provenance_problems({**sound, "archive_sha256": "short"})
  assert provenance_problems({**sound, "archive_md5": "short"})
  assert provenance_problems({**sound, "archive_size": 0})
  assert provenance_problems({**sound, "source_urls": []})
  assert provenance_problems({**sound, "zenodo_record_ids": ["999"]})
  assert provenance_problems({**sound, "split_file": "  "})
  assert provenance_problems({**sound, "split_fallback": "no"})
