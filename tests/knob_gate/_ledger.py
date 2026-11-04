"""One resolution rule for the sidecar ledger shards and the family-tagged branch manifest.

Used by ``_coverage.check_branch_coverage`` (the gate), ``scripts/redsox/verify_wave.py``
and ``scripts/redsox/stale_rows.py``. Before this module each of the three carried its own
copy of the loading logic, and the copies disagreed: a family-scoped read fell back to the
legacy ``sidecar_ledger.toml`` (a symlink to ``ledgers/potts.toml``), so a laser family read
could return potts run ids.

The rule, applied identically everywhere:

* ``family=None`` (or ``"all"``) merges every shard in ``ledgers/``.
* ``family=F`` reads ``ledgers/F.toml`` and nothing else. A missing shard is EMPTY. There is no
  fallback to any other shard, legacy file or default.
* A manifest row is attributed to a family only by its own ``family`` key. A family-scoped read
  of a row that declares no family raises ``LedgerError``: it cannot be attributed, so it can
  neither be counted for a family nor silently dropped from one.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

ALL = "all"
MANIFEST_NAME = "branch_manifest.toml"
LEDGERS_DIRNAME = "ledgers"


class LedgerError(ValueError):
  """A ledger shard or manifest row that cannot be read under the resolution rule."""


def _is_all(family: str | None) -> bool:
  return family is None or family == ALL


def shard_path(ledgers_dir: Path, family: str) -> Path:
  """The one shard a family reads. No fallback."""
  return ledgers_dir / f"{family}.toml"


def load_shard(path: Path) -> dict[str, str]:
  """``slug -> bth_run_id`` for one shard. A missing file is empty; a malformed entry raises."""
  if not path.is_file():
    return {}
  data = tomllib.loads(path.read_text(encoding="utf-8"))
  table = data.get("sidecar", {})
  if not isinstance(table, dict):
    msg = f"{path}: [sidecar] must be a table"
    raise LedgerError(msg)
  ids: dict[str, str] = {}
  for slug, body in table.items():
    if not isinstance(body, dict) or "bth_run_id" not in body:
      continue
    run_id = body["bth_run_id"]
    if not isinstance(run_id, str) or not run_id:
      msg = f"{path}: [sidecar.{slug}] bth_run_id must be a non-empty string"
      raise LedgerError(msg)
    ids[str(slug)] = run_id
  return ids


def ledger_ids(ledgers_dir: Path, family: str | None) -> dict[str, str]:
  """Run ids a read of ``family`` sees: every shard for ``None``/``"all"``, else that shard only."""
  if not _is_all(family):
    return load_shard(shard_path(ledgers_dir, str(family)))
  merged: dict[str, str] = {}
  if not ledgers_dir.is_dir():
    return merged
  for shard in sorted(ledgers_dir.glob("*.toml")):
    for slug, run_id in load_shard(shard).items():
      if slug in merged and merged[slug] != run_id:
        msg = f"sidecar {slug!r} is recorded twice with different run ids (second in {shard.name})"
        raise LedgerError(msg)
      merged[slug] = run_id
  return merged


def recorded_elsewhere(ledgers_dir: Path, family: str, slug: str) -> str | None:
  """The other shard that holds ``slug``, when a family's own shard does not. For diagnostics only."""
  if not ledgers_dir.is_dir():
    return None
  for shard in sorted(ledgers_dir.glob("*.toml")):
    if shard.name == f"{family}.toml":
      continue
    if slug in load_shard(shard):
      return shard.name
  return None


def load_manifest_rows(manifest_path: Path) -> list[dict[str, object]]:
  """Every ``[[branch]]`` table in the manifest, as plain dicts."""
  if not manifest_path.is_file():
    return []
  data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
  rows = data.get("branch", [])
  if not isinstance(rows, list):
    msg = f"{manifest_path}: [[branch]] must be an array of tables"
    raise LedgerError(msg)
  return [dict(row) for row in rows if isinstance(row, dict)]


def rows_for_family(rows: Sequence[Mapping[str, object]], family: str | None) -> list[dict[str, object]]:
  """The rows a read of ``family`` covers. ``None``/``"all"`` keeps every row."""
  if _is_all(family):
    return [dict(row) for row in rows]
  scoped: list[dict[str, object]] = []
  for row in rows:
    declared = row.get("family")
    if declared is None:
      msg = (
        f"branch row {row.get('id')!r} declares no family; a family-scoped check "
        f"({family!r}) cannot attribute it. Add family = {family!r} or the row it names."
      )
      raise LedgerError(msg)
    if declared == family:
      scoped.append(dict(row))
  return scoped


def known_families(ledgers_dir: Path, rows: Sequence[Mapping[str, object]]) -> list[str]:
  """Every family name a ``--family`` value may take: one per ledger shard, plus each manifest family."""
  names: set[str] = set()
  if ledgers_dir.is_dir():
    names.update(path.stem for path in ledgers_dir.glob("*.toml"))
  names.update(str(row["family"]) for row in rows if isinstance(row.get("family"), str))
  return sorted(names)


def sidecar_slugs(rows: Sequence[Mapping[str, object]]) -> dict[str, set[str]]:
  """``slug -> {row id}`` for every sidecar-backed row."""
  by_slug: dict[str, set[str]] = {}
  for row in rows:
    vehicle = row.get("vehicle")
    if not isinstance(vehicle, dict) or vehicle.get("kind") != "sidecar":
      continue
    slug = vehicle.get("slug")
    row_id = row.get("id")
    if not isinstance(slug, str) or not slug or not isinstance(row_id, str):
      continue
    by_slug.setdefault(slug, set()).add(row_id)
  return by_slug
