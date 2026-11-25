"""Build one JSON file for the static parity + knobs dashboard.

Copies recorded values only: bathos catalog records, sidecar TOML text, knob
manifest rows. Counts, joins and groupings are done here; no statistic is
computed. Nothing is re-run and no test suite is invoked.

Reads ``run_<uuid>.parquet`` files (one row each) with pyarrow. Pass ``--catalog``
pointing at a directory of those files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The shared shard-resolution rule (tests/knob_gate/_ledger.py). Importing it has no side effects.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "knob_gate"))
from _ledger import LEDGERS_DIRNAME, ledger_ids

log = logging.getLogger("extract_parity")

SCHEMA_VERSION = "parity_dashboard_v1"
WAVE_PREFIXES = ("scripts/parity/", "scripts/protonpotts/", "scripts/browser")
WAVE_SCRIPT_RE = re.compile(r"scripts/(?:parity|protonpotts|browser)[^\s'\"]*")
SIDECAR_TITLE_CHARS = 600
RECORD_FIELDS = (
  "id",
  "git_hash",
  "git_dirty",
  "timestamp",
  "duration_s",
  "status",
  "outcome",
  "outcome_is_residual",
  "outcome_error_reason",
  "sidecar_path",
)


def _repo_root_from_file() -> Path | None:
  for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  return None


def _clean(value: object) -> object:
  """Recursively make a value JSON-safe: NaN/inf become None, datetimes become ISO strings."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, datetime):
    return value.isoformat()
  if isinstance(value, dict):
    return {str(k): _clean(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_clean(v) for v in value]
  return value


def _git_head(repo: Path) -> str | None:
  try:
    out = subprocess.run(  # noqa: S603
      ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
      check=True,
      capture_output=True,
      text=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout.strip() or None


def load_catalog(catalog: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
  """Read every run_<id>.parquet into a dict keyed by run id."""
  import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency, only needed here

  runs: dict[str, dict[str, Any]] = {}
  files = sorted(catalog.glob("run_*.parquet"))
  for path in files:
    try:
      rows = pq.read_table(path).to_pylist()
    except OSError as exc:  # pyarrow raises OSError/ArrowInvalid subclasses of it
      warnings.append(f"catalog file {path.name} unreadable: {exc}")
      continue
    if len(rows) != 1:
      warnings.append(f"catalog file {path.name} has {len(rows)} rows, expected 1")
    for row in rows:
      run_id = str(row.get("id"))
      if path.stem != f"run_{run_id}":
        warnings.append(f"catalog file {path.name} holds run id {run_id}")
      runs[run_id] = row
  log.info("loaded %d catalog runs from %d files", len(runs), len(files))
  return runs


def _record(run: dict[str, Any]) -> dict[str, Any]:
  return {k: _clean(run.get(k)) for k in RECORD_FIELDS}


def _parse_results(
  run: dict[str, Any] | None,
  warnings: list[str],
  cache: dict[str, Any],
) -> object:
  """Return the run's metadata parsed as JSON, or None. Warn once per run on bad JSON."""
  if run is None:
    return None
  run_id = str(run.get("id"))
  if run_id in cache:
    return cache[run_id]
  raw = run.get("metadata")
  parsed: Any = None
  if isinstance(raw, str) and raw.strip():
    try:
      parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
      warnings.append(f"run {run_id} metadata is not valid JSON: {exc}")
  cache[run_id] = _clean(parsed)
  return cache[run_id]


def load_ledger(repo: Path) -> dict[str, str]:
  """Run ids for every family: the dashboard is not family-scoped, so every shard is merged."""
  return ledger_ids(repo / "tests/knob_gate" / LEDGERS_DIRNAME, None)


def load_manifest(repo: Path) -> list[dict[str, Any]]:
  data = tomllib.loads((repo / "tests/knob_gate/branch_manifest.toml").read_text())
  return list(data.get("branch", []))


def load_alias_map(repo: Path) -> list[dict[str, Any]]:
  data = tomllib.loads((repo / "tests/knob_gate/alias_map.toml").read_text())
  return list(data.get("row", []))


def family_index(manifest: list[dict[str, Any]]) -> tuple[dict[str, str], tuple[str, ...]]:
  """Family lookup built from the manifest, not hardcoded: exact sidecar slug -> family, plus the
  family names as prefixes (longest first) for slugs that have no manifest row (upstream knobs)."""
  slug_table: dict[str, str] = {}
  families: set[str] = set()
  for row in manifest:
    family = row.get("family")
    if not isinstance(family, str) or not family:
      continue
    families.add(family)
    vehicle = row.get("vehicle")
    if (
      isinstance(vehicle, dict)
      and vehicle.get("kind") == "sidecar"
      and isinstance(vehicle.get("slug"), str)
    ):
      slug_table[str(vehicle["slug"])] = family
  return slug_table, tuple(sorted(families, key=len, reverse=True))


def family_of_slug(slug: str, index: tuple[dict[str, str], tuple[str, ...]]) -> str | None:
  slug_table, prefixes = index
  if slug in slug_table:
    return slug_table[slug]
  for prefix in prefixes:
    if slug.startswith(prefix):
      return prefix
  return None


def read_sidecar(repo: Path, slug: str) -> dict[str, Any] | None:
  path = repo / "scripts/parity" / f"{slug}.bth.toml"
  if not path.is_file():
    return None
  data = tomllib.loads(path.read_text())
  hypothesis = str(data.get("experiment", {}).get("hypothesis", "") or "")
  outcomes = data.get("outcomes", {})
  pass_block = outcomes.get("pass", {}) if isinstance(outcomes, dict) else {}
  return {
    "path": str(path.relative_to(repo)),
    "title": data.get("metadata", {}).get("title"),
    "hypothesis": hypothesis[:SIDECAR_TITLE_CHARS],
    "pass_condition": pass_block.get("condition") if isinstance(pass_block, dict) else None,
  }


def read_branch_controls(
  controls_root: Path | None,
  slug: str,
  run: dict[str, Any] | None,
  warnings: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """The vehicle's ``branch_controls.json`` (per-control status), as launch_wave.sh writes it.

  The run's own metadata carries only counts; the per-control map lives in the file the run
  lists in ``--output-paths``: ``<controls_root>/<slug>/<git sha8>/branch_controls.json``.
  Returns (parsed file, provenance {path, sha256}) or (None, None).
  """
  if controls_root is None or run is None or not run.get("git_hash"):
    return None, None
  path = controls_root / slug / str(run["git_hash"])[:8] / "branch_controls.json"
  if not path.is_file():
    warnings.append(f"{slug}: no branch_controls.json at {path}")
    return None, None
  raw = path.read_bytes()
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as exc:
    warnings.append(f"{slug}: {path} is not valid JSON: {exc}")
    return None, None
  provenance = {
    "path": str(path.relative_to(controls_root)),
    "sha256": hashlib.sha256(raw).hexdigest(),
  }
  return _clean(parsed), provenance


def _control_status(results: object, control_id: str) -> object:
  if not isinstance(results, dict):
    return None
  mutants = results.get("mutants")
  if not isinstance(mutants, dict) or control_id not in mutants:
    return None
  value = mutants[control_id]
  if isinstance(value, dict):
    return _clean(value.get("status"))
  return _clean(value)


def build_ledger(
  ledger: dict[str, str],
  manifest: list[dict[str, Any]],
  runs: dict[str, dict[str, Any]],
  repo: Path,
  stale: list[str] | None,
  warnings: list[str],
  cache: dict[str, Any],
  controls_root: Path | None = None,
) -> list[dict[str, Any]]:
  controls_by_slug: dict[str, list[dict[str, Any]]] = {}
  for row in manifest:
    slug = str(row.get("vehicle", {}).get("slug", ""))
    controls_by_slug.setdefault(slug, []).append(row)

  index = family_index(manifest)
  entries: list[dict[str, Any]] = []
  for slug, run_id in ledger.items():
    run = runs.get(run_id)
    entry: dict[str, Any] = {
      "slug": slug,
      "family": family_of_slug(slug, index),
      "run_id": run_id,
      "stale": (slug in stale) if stale is not None else None,
      "record": _record(run) if run is not None else None,
    }
    if run is None:
      entry["missing_record"] = True
      warnings.append(f"ledger run {run_id} for {slug} missing from catalog")
    results = _parse_results(run, warnings, cache)
    entry["results"] = results

    branch_controls, provenance = read_branch_controls(controls_root, slug, run, warnings)
    entry["branch_controls"] = provenance
    source = branch_controls if branch_controls is not None else results

    rows = controls_by_slug.get(slug, [])
    if rows and not (isinstance(source, dict) and isinstance(source.get("mutants"), dict)):
      warnings.append(f"{slug}: no 'mutants' mapping found; control status is null")
    controls: list[dict[str, Any]] = []
    for row in rows:
      control_id = str(row.get("id"))
      status = _control_status(source, control_id)
      if source is not None and status is None:
        warnings.append(f"{slug}: manifest control {control_id} has no recorded status")
      wave = (
        control_id.split(".", 1)[0] if slug == "protonpotts_parity" and "." in control_id else None
      )
      controls.append({"id": control_id, "wave": wave, "status": status})
    entry["controls"] = controls
    entry["sidecar"] = read_sidecar(repo, slug)
    entries.append(entry)
  return entries


def build_history(
  slugs: list[str],
  runs: dict[str, dict[str, Any]],
  warnings: list[str],
) -> dict[str, list[dict[str, Any]]]:
  history: dict[str, list[dict[str, Any]]] = {}
  for slug in slugs:
    suffix = f"scripts/parity/{slug}.bth.toml"
    script = f"scripts/parity/{slug}.py"
    rows = []
    for run in runs.values():
      sidecar_path = str(run.get("sidecar_path") or "")
      command = str(run.get("command") or "")
      if sidecar_path.endswith(suffix) or script in command:
        rows.append(run)
    rows.sort(key=lambda r: str(_clean(r.get("timestamp")) or ""))
    history[slug] = [
      {
        "id": str(r.get("id")),
        "git_hash": _clean(r.get("git_hash")),
        "timestamp": _clean(r.get("timestamp")),
        "status": _clean(r.get("status")),
        "outcome": _clean(r.get("outcome")),
        "duration_s": _clean(r.get("duration_s")),
      }
      for r in rows
    ]
    if not history[slug]:
      warnings.append(f"no catalog runs found for ledger slug {slug}")
  return history


def build_waves(
  runs: dict[str, dict[str, Any]],
  repo: Path,
  warnings: list[str],
  cache: dict[str, Any],
) -> list[dict[str, Any]]:
  by_script: dict[str, list[dict[str, Any]]] = {}
  for run in runs.values():
    command = str(run.get("command") or "")
    if not any(prefix in command for prefix in WAVE_PREFIXES):
      continue
    match = WAVE_SCRIPT_RE.search(command)
    if match is None:
      continue
    by_script.setdefault(match.group(0), []).append(run)

  waves: list[dict[str, Any]] = []
  for script, script_runs in by_script.items():
    script_runs.sort(key=lambda r: str(_clean(r.get("timestamp")) or ""))
    latest = script_runs[-1]
    sibling = (repo / script).with_suffix(".bth.toml")
    title: str | None = None
    if sibling.is_file():
      title = tomllib.loads(sibling.read_text()).get("metadata", {}).get("title")
    waves.append(
      {
        "script": script,
        "n_runs": len(script_runs),
        "latest": {
          "id": str(latest.get("id")),
          "git_hash": _clean(latest.get("git_hash")),
          "timestamp": _clean(latest.get("timestamp")),
          "status": _clean(latest.get("status")),
          "outcome": _clean(latest.get("outcome")),
          "outcome_is_residual": _clean(latest.get("outcome_is_residual")),
        },
        "results": _parse_results(latest, warnings, cache),
        "sidecar_title": title,
      },
    )
  waves.sort(key=lambda w: str(w["latest"]["timestamp"] or ""), reverse=True)
  return waves


def build_knobs(
  alias_rows: list[dict[str, Any]],
  index: tuple[dict[str, str], tuple[str, ...]],
) -> list[dict[str, Any]]:
  knobs = []
  for row in alias_rows:
    entry = _clean(dict(row))
    ref = str(row.get("ref", ""))
    # ref is <upstream entry point>__<knob>; "__aminx_only__<knob>" has no upstream counterpart.
    upstream_entry = ref.partition("__")[0]
    entry["upstream_entry"] = upstream_entry or None
    entry["family"] = (
      "aminx_only"
      if not upstream_entry
      else family_of_slug(upstream_entry, index) or upstream_entry
    )
    entry["covered_by_parity"] = bool(row.get("parity_test_ids"))
    knobs.append(entry)
  return knobs


def build_dashboard(
  catalog: Path,
  repo: Path,
  stale: list[str] | None,
  controls_root: Path | None = None,
) -> dict[str, Any]:
  warnings: list[str] = []
  cache: dict[str, Any] = {}
  runs = load_catalog(catalog, warnings)
  ledger = load_ledger(repo)
  manifest = load_manifest(repo)
  alias_rows = load_alias_map(repo)

  manifest_counts: dict[str, int] = {}
  for row in manifest:
    slug = str(row.get("vehicle", {}).get("slug", ""))
    manifest_counts[slug] = manifest_counts.get(slug, 0) + 1

  return {
    "schema_version": SCHEMA_VERSION,
    "generated_at": datetime.now(UTC).isoformat(),
    "repo_head": _git_head(repo),
    "ledger": build_ledger(ledger, manifest, runs, repo, stale, warnings, cache, controls_root),
    "history": build_history(list(ledger), runs, warnings),
    "waves": build_waves(runs, repo, warnings, cache),
    "knobs": build_knobs(alias_rows, family_index(manifest)),
    "manifest_counts": manifest_counts,
    "warnings": warnings,
  }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--catalog", type=Path, required=True, help="dir of run_<id>.parquet files")
  parser.add_argument(
    "--repo",
    type=Path,
    default=_repo_root_from_file(),
    help="aminx repo root (default: nearest pyproject.toml above this file)",
  )
  parser.add_argument(
    "--stale-rows-json",
    type=Path,
    default=None,
    help="optional JSON list of stale slugs; omitted means staleness is null",
  )
  parser.add_argument(
    "--controls-root",
    type=Path,
    default=None,
    help="dir holding <slug>/<sha8>/branch_controls.json (on titanix ~/.aminx/sidecars); "
    "omitted means per-control status comes only from run metadata",
  )
  parser.add_argument("--out", type=Path, required=True, help="output JSON path")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  args = _parse_args(argv)
  if args.repo is None:
    log.error("could not find repo root; pass --repo")
    return 2
  if not args.catalog.is_dir():
    log.error("catalog directory not found: %s", args.catalog)
    return 2
  stale: list[str] | None = None
  if args.stale_rows_json is not None:
    stale = [str(s) for s in json.loads(args.stale_rows_json.read_text())]

  dashboard = build_dashboard(args.catalog, args.repo, stale, args.controls_root)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(_clean(dashboard), sort_keys=False, indent=1) + "\n")
  log.info("wrote %s (%d warnings)", args.out, len(dashboard["warnings"]))
  return 0


if __name__ == "__main__":
  sys.exit(main())
