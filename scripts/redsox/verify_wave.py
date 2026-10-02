"""Explain, condition by condition, why a vehicle run will or will not grade.

``check_branch_coverage`` answers pass/invalid for the whole manifest. That is
the right contract for a gate and the wrong one for a human holding a run that
took three hours: it does not say WHICH of the nine step-1c conditions threw
the run out, and several of them fail for reasons unrelated to what the run
measured (a tracked file dirtied by pytest, a mutant string that does not
match the manifest, a weights key written as an absolute path).

So this reports every condition separately, with the observed value next to
the expected one.

Every VERDICT is the gate's own predicate, imported from ``tests/knob_gate``,
so a verdict here cannot drift from the gate it explains. Only the explanatory
detail is restated -- e.g. the gate's ``_touches_scoped`` answers yes/no, and
the list of offending paths printed beside it is computed here purely to make
the "no" actionable. An earlier revision restated the predicates themselves,
including ``_weights_required``, which is how it came to report a gradeable
run as ungradeable (see ``weights`` below).

Usage:
    uv run python3 scripts/redsox/verify_wave.py
    uv run python3 scripts/redsox/verify_wave.py --run laser_decode_e2e=9c8c5dce

With no ``--run``, each slug's id comes from the sidecar ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("verify_wave")


def _repo() -> Path:
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  msg = "no pyproject.toml above this file"
  raise RuntimeError(msg)


REPO = _repo()
sys.path.insert(0, str(REPO / "tests" / "knob_gate"))

from _coverage import (  # noqa: E402
  _SCOPED_FILES,
  _SCOPED_PREFIXES,
  _argv_mutants,
  _touches_scoped,
  _weights_required,
  default_changed_paths,
  default_is_ancestor,
  default_registry_sha256,
  default_resolve_run,
  default_sidecar_digest,
)


def _manifest_rows() -> dict[str, set[str]]:
  """``slug -> {row id}`` for every sidecar-backed row."""
  path = REPO / "tests" / "knob_gate" / "branch_manifest.toml"
  rows = tomllib.loads(path.read_text(encoding="utf-8")).get("branch", [])
  by_slug: dict[str, set[str]] = defaultdict(set)
  for row in rows:
    vehicle = row.get("vehicle", {})
    if vehicle.get("kind") == "sidecar":
      by_slug[vehicle["slug"]].add(row["id"])
  return dict(by_slug)


def _ledger_ids() -> dict[str, str]:
  path = REPO / "tests" / "knob_gate" / "sidecar_ledger.toml"
  table = tomllib.loads(path.read_text(encoding="utf-8")).get("sidecar", {})
  return {slug: body["bth_run_id"] for slug, body in table.items() if "bth_run_id" in body}


def _scoped_hits(paths: list[str]) -> list[str]:
  """The offending paths, for the message only.

  The VERDICT comes from the gate's ``_touches_scoped``; this just names which
  paths made it say yes, because "a scoped path landed since that run" is
  unactionable without knowing which commit did it.
  """
  hits = []
  for raw in paths:
    path = str(raw).replace("\\", "/").lstrip("./")
    if path in _SCOPED_FILES or any(path.startswith(p) for p in _SCOPED_PREFIXES):
      hits.append(path)
  return hits


def _report(label: str, ok: bool, detail: str = "") -> bool:
  mark = "PASS" if ok else "FAIL"
  logger.info("    [%s] %-34s %s", mark, label, detail)
  return ok


def verify(slug: str, run_id: str, row_ids: set[str]) -> bool:
  """Every step-1c condition for one slug, reported individually."""
  logger.info("=== %s  run %s", slug, run_id)
  run = default_resolve_run(run_id)
  if run is None:
    return _report("run resolves", False, "no cool-tier parquet (try: bth compact)")

  ok = True
  ok &= _report("status == completed", run["status"] == "completed", repr(run["status"]))
  ok &= _report("outcome == pass", run["outcome"] == "pass", repr(run["outcome"]))
  ok &= _report("git_dirty is False", not run["git_dirty"], repr(run["git_dirty"]))

  git_hash = run["git_hash"]
  ok &= _report("git_hash present", bool(git_hash), str(git_hash)[:12])
  if git_hash:
    ok &= _report("ancestor of HEAD", default_is_ancestor(git_hash))
    changed = default_changed_paths(git_hash)
    hits = _scoped_hits(changed)
    ok &= _report(
      "no scoped path since",
      not _touches_scoped(changed),
      f"{len(hits)} scoped: {sorted(set(hits))[:3]}" if hits else "",
    )

  try:
    expected = default_sidecar_digest(slug)
    ok &= _report("sidecar_sha256 matches", run["sidecar_sha256"] == expected, "")
  except OSError as exc:
    ok &= _report("sidecar readable", False, str(exc))

  listed = _argv_mutants(run["argv"])
  if listed is None:
    ok &= _report("--mutants in argv", False, "absent; the default list does not count")
  else:
    extra, missing = sorted(listed - row_ids), sorted(row_ids - listed)
    ok &= _report(
      "--mutants == manifest rows",
      listed == row_ids,
      "" if listed == row_ids else f"argv-only {extra} rows-only {missing}",
    )

  suffix = f"/{slug}/{str(git_hash)[:8]}/branch_controls.json"
  matches = [p for p in run["output_paths"] if str(p).replace("\\", "/").endswith(suffix)]
  ok &= _report("output path well-formed", bool(matches), suffix if not matches else "")
  if not matches:
    return False

  controls_path = Path(matches[0])
  if not controls_path.is_file():
    return _report("controls file exists", False, str(controls_path))
  controls = json.loads(controls_path.read_text(encoding="utf-8"))

  ok &= _report(
    "controls clean == pass", controls.get("clean") == "pass", repr(controls.get("clean"))
  )
  mutants = controls.get("mutants", {})
  ok &= _report(
    "controls mutants == rows",
    set(mutants) == row_ids,
    "" if set(mutants) == row_ids else f"{sorted(set(mutants))}",
  )
  not_failed = [k for k, v in mutants.items() if v != "failed"]
  ok &= _report(
    "every control failed", not not_failed, f"not failed: {not_failed}" if not_failed else ""
  )

  # The gate requires a non-empty weights map only for slugs matching
  # _WEIGHT_PREFIXES ("pottsmpnn_", "lasermpnn_") -- which NO vehicle slug in
  # branch_manifest.toml matches, so the requirement is inert here (debt: the
  # prefixes read like knob slugs, not vehicle slugs). Reporting it as a hard
  # FAIL, as this script first did, condemns a run the gate would have graded.
  # The per-key check below is NOT conditional: weights that are present must
  # resolve in checkpoint_registry.json, and that is the trap that actually
  # fires -- controls written before 6cb81d49 key by absolute path and never
  # validate.
  weights = controls.get("weights", {})
  required = _weights_required(slug)
  ok &= _report(
    "weights present",
    bool(weights) or not required,
    "" if weights else f"empty; required={required}",
  )
  for artifact, sha in weights.items():
    resolved = default_registry_sha256(artifact)
    ok &= _report(
      "weights key resolves",
      resolved == sha,
      artifact if resolved == sha else f"{artifact} -> {resolved!r} != {sha!r}",
    )
  return bool(ok)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--run",
    action="append",
    default=[],
    metavar="SLUG=RUNID",
    help="override a slug's run id; repeatable. Defaults to the ledger.",
  )
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  rows = _manifest_rows()
  ids = _ledger_ids()
  for pair in args.run:
    slug, _, rid = pair.partition("=")
    ids[slug] = rid

  if not ids:
    logger.info("no run ids: the ledger is empty and no --run was given")
    return 2

  results = {}
  for slug in sorted(ids):
    if slug not in rows:
      logger.info("=== %s  SKIPPED: no manifest rows for this slug", slug)
      results[slug] = False
      continue
    results[slug] = verify(slug, ids[slug], rows[slug])
    logger.info("")

  missing = sorted(set(rows) - set(ids))
  if missing:
    logger.info("no run id yet for: %s", ", ".join(missing))
  good = sorted(s for s, v in results.items() if v)
  bad = sorted(s for s, v in results.items() if not v)
  logger.info("would grade: %s", ", ".join(good) if good else "(none)")
  logger.info("would NOT grade: %s", ", ".join(bad) if bad else "(none)")
  return 0 if not bad and not missing else 1


if __name__ == "__main__":
  raise SystemExit(main())
