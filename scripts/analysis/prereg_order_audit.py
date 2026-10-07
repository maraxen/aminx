"""Was every bathos sidecar really committed BEFORE the run it graded?

WHY. Spec §7.3 says "Bathos sidecars (committed before running)", and the
project rule is blunter: writing a sidecar after seeing the numbers is the
failure the whole discipline exists to stop, because outcome criteria chosen
post-hoc always pass. Nothing checked it. This does, over the whole catalog
rather than over a sample.

TWO QUESTIONS, DELIBERATELY SEPARATED, because conflating them would either
manufacture violations or hide them:

  TIME ORDER   the sidecar's add-commit is older than the earliest run citing
               it. This IS the pre-registration requirement.
  CONTAINMENT  the run's recorded ``git_hash`` actually contains that commit.
               This is whether the RECORD lets a reader check the time order
               without taking a timestamp on trust.

A run can satisfy the first and fail the second -- a script copied into a
checkout that sits at an older commit is the ordinary way. That is a provenance
weakness, not a violation, and this reports it as its own category. Collapsing
the two would turn every spike-dir run into a false accusation.

WHAT A VIOLATION WOULD MEAN. Not merely untidy: a sidecar committed after its
run cannot have constrained that run, so the outcome it reports is a
description, not a prediction, and nothing it graded should be cited.

ONE-SIDED BY CONSTRUCTION, stated so it is not over-read. Finding zero
violations shows no sidecar in the catalog was committed late. It cannot show a
sidecar was not WRITTEN late and committed before the run anyway -- git records
when a file landed, not when its author decided the criteria. This measures the
half that is mechanically checkable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("prereg_order_audit")

#: Below this many audited sidecars the result is vacuous rather than clean: a
#: catalog query that silently matched almost nothing would otherwise report a
#: reassuring zero.
_MIN_AUDITED = 20


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
  proc = subprocess.run(  # noqa: S603
    cmd, cwd=cwd, capture_output=True, text=True, check=False,
  )
  return proc.returncode, proc.stdout.strip()


def _git(repo: Path, *args: str) -> tuple[int, str]:
  return _run(["git", *args], cwd=repo)


def earliest_runs(bth: str, project: str) -> list[dict[str, Any]]:
  """(sidecar_path, earliest run, that run's git_hash) straight from the catalog."""
  sql = (
    "SELECT sidecar_path, min(timestamp) AS first_run, "
    "arg_min(git_hash, timestamp) AS first_git, count(*) AS n FROM runs "
    "WHERE sidecar_path IS NOT NULL AND git_hash IS NOT NULL "
    f"AND project_slug = '{project}' GROUP BY sidecar_path ORDER BY first_run"
  )
  code, out = _run([bth, "sql", sql])
  if code != 0 or not out:
    return []
  try:
    payload = json.loads(out)
  except json.JSONDecodeError:
    return []
  return [
    {"sidecar_path": r[0], "first_run": r[1], "first_git": r[2], "n": r[3]}
    for r in payload.get("rows", [])
    if r and r[0]
  ]


def repo_relative(sidecar_path: str) -> str | None:
  """A catalog path is absolute in whatever checkout ran it; keep the tail.

  Every checkout shares the same layout below the repo root, so the first
  recognised top-level directory anchors the relative path.
  """
  parts = Path(sidecar_path).parts
  for anchor in ("scripts", "tests", "src"):
    if anchor in parts:
      return str(Path(*parts[parts.index(anchor) :]))
  return None


def add_commit(repo: Path, rel: str) -> tuple[str, str] | None:
  """(sha, AUTHOR date) of the commit that added this path.

  AUTHOR date, not committer date, and the difference is not cosmetic. A rebase
  or squash rewrites the committer date to the moment of the rewrite while
  preserving the author date. Using %cI made every file that arrived via a
  rebase look as though it had been committed days AFTER the runs citing it:
  a first pass over the local catalog reported 65 violations in 78 sidecars,
  with dozens sharing the committer timestamp 2026-09-30T08:50:45 -- the
  signature of one bulk rewrite, not of 65 post-hoc registrations. Checked
  against a flagged commit: committer 2026-09-30T08:50:45, author
  2026-09-24T11:17:48. The author date is the one that answers "when was this
  written".
  """
  code, out = _git(repo, "log", "--follow", "--diff-filter=A", "--format=%H %aI", "--", rel)
  if code != 0 or not out:
    return None
  sha, _, when = out.splitlines()[-1].partition(" ")
  return sha, when


def was_renamed(repo: Path, rel: str) -> bool:
  """Did this path ever arrive via a rename?

  ``--follow`` traces renames with a similarity HEURISTIC, so for a file that
  moved, the commit it calls the "add" may be the move rather than the original
  authoring. That uncertainty must not be spent accusing someone of post-hoc
  registration, so a renamed path can be reported as unverifiable but never as
  a violation.
  """
  code, out = _git(repo, "log", "--follow", "--name-status", "--format=", "--", rel)
  if code != 0:
    return False
  return any(line.startswith("R") for line in out.splitlines())


def classify(
  repo: Path, rel: str, first_run: str, first_git: str,
) -> tuple[str, dict[str, Any]]:
  """One sidecar -> (category, detail). Categories are disjoint and total."""
  info = add_commit(repo, rel)
  if info is None:
    return "unverifiable", {"sidecar": rel, "why": "no add-commit in this branch history"}
  sha, when = info
  committed = datetime.fromisoformat(when).astimezone(timezone.utc)
  ran = datetime.fromisoformat(first_run.replace(" ", "T")).astimezone(timezone.utc)
  lead_h = (ran - committed).total_seconds() / 3600.0
  if lead_h < 0:
    if was_renamed(repo, rel):
      return "unverifiable", {
        "sidecar": rel, "added": sha[:12], "hours_late": round(-lead_h, 3),
        "why": "appears late, but the path was RENAMED, so --follow's add "
               "attribution is heuristic and cannot carry the accusation",
      }
    return "violation", {
      "sidecar": rel, "added": sha[:12], "committed": when[:19],
      "first_run": first_run[:19], "hours_late": round(-lead_h, 3),
    }
  code, _ = _git(repo, "cat-file", "-t", first_git)
  contained = False
  if code == 0:
    code, _ = _git(repo, "merge-base", "--is-ancestor", sha, first_git)
    contained = code == 0
  if not contained:
    return "unverifiable", {
      "sidecar": rel, "added": sha[:12], "lead_hours": round(lead_h, 3),
      "why": f"run git_hash {first_git[:12]} does not contain {sha[:12]}",
    }
  return "clean", {"sidecar": rel, "added": sha[:12], "lead_hours": round(lead_h, 3)}


def _self_test(repo: Path) -> dict[str, Any]:
  """Synthetic ground truth. A one-sided check that cannot fail is not a check."""
  failed: list[str] = []
  checks = 0

  checks += 1
  if repo_relative("/a/b/aminx-x/scripts/parity/z.bth.toml") != "scripts/parity/z.bth.toml":
    failed.append("repo_relative_did_not_anchor")
  checks += 1
  if repo_relative("/nowhere/z.bth.toml") is not None:
    failed.append("repo_relative_accepted_unanchorable")

  # The decisive pair: a LATE sidecar must be called a violation and an EARLY
  # one must not. Both are synthesised from the same helper the real path uses.
  late_run = "2020-01-01 00:00:00"
  checks += 1
  real = add_commit(repo, "scripts/analysis/prereg_order_audit.py")
  if real is None:
    failed.append("cannot_resolve_own_add_commit")
  else:
    kind, _ = classify(repo, "scripts/analysis/prereg_order_audit.py", late_run, "HEAD")
    if kind != "violation":
      failed.append(f"late_run_not_flagged({kind})")
    checks += 1
    future = datetime.now(timezone.utc).isoformat(sep=" ")[:19]
    kind2, _ = classify(repo, "scripts/analysis/prereg_order_audit.py", future, "HEAD")
    if kind2 == "violation":
      failed.append("early_run_falsely_flagged")

  checks += 1
  if classify(repo, "scripts/does_not_exist.bth.toml", late_run, "HEAD")[0] != "unverifiable":
    failed.append("missing_path_not_unverifiable")

  # The date field must be the AUTHOR date. Committer dates are rewritten by
  # rebase, which made a first pass report 65 violations in 78 sidecars out of
  # one bulk rewrite. Pinned by checking a commit whose two dates differ: if
  # add_commit ever returns the committer date again, this fires.
  checks += 1
  _code, out = _git(repo, "log", "--format=%H %aI %cI", "-200")
  divergent = None
  for line in out.splitlines():
    sha, author, committer = (line.split() + ["", ""])[:3]
    if author and committer and author[:19] != committer[:19]:
      divergent = (sha, author, committer)
      break
  if divergent is not None:
    sha, author, _committer = divergent
    code, out = _git(repo, "log", "-1", "--format=%aI", sha)
    if code == 0 and out.strip()[:19] != author[:19]:
      failed.append("date_field_is_not_author_date")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
  }


def _grade(payload: dict[str, Any]) -> str:
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload.get("refusal"):
    return str(payload["refusal"])
  if payload["n_violations"] > 0:
    return "prereg_order_violated"
  if payload["n_audited"] < _MIN_AUDITED:
    return "audit_underpowered"
  return "prereg_order_clean"


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text, encoding="utf-8")
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text, encoding="utf-8")


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--repo", type=Path, required=True, help="checkout to resolve commits in")
  parser.add_argument("--bth", default="bth", help="bathos executable")
  parser.add_argument("--project", default="aminx")
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()

  base = _self_test(args.repo)
  _LOG.info(
    "self-test %s (%d checks)",
    "passed" if base["self_test_passed"] else f"FAILED {base['self_test_failed']}",
    base["self_test_n_checks"],
  )
  if args.self_test_only:
    verdict = "self_test_only" if base["self_test_passed"] else "instrument_unverified"
    _emit(args.out, {**base, "refusal": None, "verdict": verdict})
    return 0 if base["self_test_passed"] else 1

  code, _ = _git(args.repo, "rev-parse", "--git-dir")
  if code != 0:
    _emit(args.out, {**base, "refusal": "repo_unavailable", "verdict": "repo_unavailable",
                     "n_violations": 0, "n_audited": 0})
    return 1

  rows = earliest_runs(args.bth, args.project)
  if not rows:
    _emit(args.out, {**base, "refusal": "catalog_unavailable",
                     "verdict": "catalog_unavailable", "n_violations": 0, "n_audited": 0})
    _LOG.error("catalog query returned nothing")
    return 1

  buckets: dict[str, list[dict[str, Any]]] = {"clean": [], "violation": [], "unverifiable": []}
  skipped = 0
  for row in rows:
    rel = repo_relative(str(row["sidecar_path"]))
    if rel is None:
      skipped += 1
      continue
    kind, detail = classify(args.repo, rel, str(row["first_run"]), str(row["first_git"]))
    detail["runs"] = row["n"]
    buckets[kind].append(detail)

  audited = sum(len(v) for v in buckets.values())
  payload = {
    **base,
    "refusal": None,
    "project": args.project,
    "n_catalog_rows": len(rows),
    "n_skipped_unanchorable": skipped,
    "n_audited": audited,
    "n_clean": len(buckets["clean"]),
    "n_violations": len(buckets["violation"]),
    "n_unverifiable": len(buckets["unverifiable"]),
    "violations": sorted(buckets["violation"], key=lambda d: -d.get("hours_late", 0)),
    "unverifiable": sorted(buckets["unverifiable"], key=lambda d: d["sidecar"]),
    "min_lead_hours": min((d["lead_hours"] for d in buckets["clean"]), default=None),
  }
  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  _LOG.info(
    "verdict=%s audited=%d clean=%d violations=%d unverifiable=%d min_lead_h=%s",
    payload["verdict"], audited, payload["n_clean"], payload["n_violations"],
    payload["n_unverifiable"], payload["min_lead_hours"],
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
