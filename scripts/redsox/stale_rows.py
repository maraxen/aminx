"""Which ledger rows must be re-run at HEAD, and which changed paths say so.

The gate (``tests/knob_gate/_coverage.py``) decides a row is stale with
``row_is_stale``: a changed path matters only if it is in that row's import closure
(see ``tests/knob_gate/_closure.py``). This reports the same verdict the gate will
reach, next to the old global-prefix verdict, so a wave can be limited to the rows
that need it.

It reuses the gate's own predicates, so it cannot drift from the gate. It checks
ONLY the staleness condition. A row can be fresh here and still fail another step-1c
condition (dirty tree, wrong argv, unresolvable weights key); ``verify_wave.py``
reports all fourteen.

Usage:
    uv run python3 scripts/redsox/stale_rows.py            # table
    uv run python3 scripts/redsox/stale_rows.py --slugs    # stale slugs, one per line

What-if mode, for a machine that does not hold the run records (they live on the
host that ran the wave): ``--run-commit SHA`` assumes every row was recorded at SHA
and ``--head REF`` diffs against REF instead of HEAD. Nothing is resolved from the
catalog in that mode, so it answers only the staleness question.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tomllib
from pathlib import Path

logger = logging.getLogger("stale_rows")


def _repo() -> Path:
  for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  msg = "no pyproject.toml above this file"
  raise RuntimeError(msg)


REPO = _repo()
sys.path.insert(0, str(REPO / "tests" / "knob_gate"))

from _closure import row_is_stale  # noqa: E402
from _coverage import (  # noqa: E402
  _touches_scoped,
  default_changed_paths,
  default_closure_for,
  default_is_ancestor,
  default_resolve_run,
)


def _manifest_slugs() -> list[str]:
  path = REPO / "tests" / "knob_gate" / "branch_manifest.toml"
  rows = tomllib.loads(path.read_text(encoding="utf-8")).get("branch", [])
  slugs = {r["vehicle"]["slug"] for r in rows if r.get("vehicle", {}).get("kind") == "sidecar"}
  return sorted(slugs)


def _ledger_ids() -> dict[str, str]:
  path = REPO / "tests" / "knob_gate" / "sidecar_ledger.toml"
  table = tomllib.loads(path.read_text(encoding="utf-8")).get("sidecar", {})
  return {slug: body["bth_run_id"] for slug, body in table.items() if "bth_run_id" in body}


def _diff_names(base: str, head: str) -> list[str]:
  done = subprocess.run(
    ["git", "diff", "--name-only", f"{base}..{head}"],
    check=True,
    capture_output=True,
    text=True,
    cwd=REPO,
  )
  return [line for line in done.stdout.splitlines() if line]


def assess(
  slug: str,
  run_id: str | None,
  *,
  run_commit: str | None = None,
  head: str = "HEAD",
) -> dict[str, object]:
  """Verdicts for one slug. ``stale`` is the NEW rule; ``stale_old`` the global prefix test."""
  if run_commit:
    changed = _diff_names(run_commit, head)
    closure = default_closure_for(slug)
    hits = [p for p in changed if row_is_stale([p], closure)]
    why = "closure unsound or unavailable: global rule" if closure is None or not closure.sound else ""
    return {
      "slug": slug,
      "stale": bool(hits),
      "stale_old": _touches_scoped(changed),
      "why": why,
      "hits": hits,
      "run_commit": run_commit[:8],
    }
  if not run_id:
    return {"slug": slug, "stale": True, "stale_old": True, "why": "no ledger run id", "hits": []}
  run = default_resolve_run(run_id)
  if run is None:
    return {"slug": slug, "stale": True, "stale_old": True, "why": "run does not resolve", "hits": []}
  git_hash = run["git_hash"]
  if not git_hash or not default_is_ancestor(git_hash):
    return {
      "slug": slug,
      "stale": True,
      "stale_old": True,
      "why": f"run commit {str(git_hash)[:8]} is not an ancestor of HEAD",
      "hits": [],
    }
  changed = default_changed_paths(git_hash)
  closure = default_closure_for(slug)
  hits = [p for p in changed if row_is_stale([p], closure)]
  why = "closure unsound or unavailable: global rule" if closure is None or not closure.sound else ""
  return {
    "slug": slug,
    "stale": bool(hits),
    "stale_old": _touches_scoped(changed),
    "why": why,
    "hits": hits,
    "run_commit": str(git_hash)[:8],
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--slugs", action="store_true", help="print only the stale slugs")
  parser.add_argument("--run-commit", help="what-if: assume every row was recorded at this commit")
  parser.add_argument("--head", default="HEAD", help="what-if: diff against this ref")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  ids = _ledger_ids()
  results = [
    assess(slug, ids.get(slug), run_commit=args.run_commit, head=args.head)
    for slug in _manifest_slugs()
  ]

  if args.slugs:
    for row in results:
      if row["stale"]:
        sys.stdout.write(f"{row['slug']}\n")
    return 0

  for row in results:
    verdict = "STALE" if row["stale"] else "fresh"
    old = "stale" if row["stale_old"] else "fresh"
    hits = row["hits"]
    assert isinstance(hits, list)
    detail = f"  {row['why']}" if row["why"] else ""
    logger.info("%-38s %-5s (old rule: %s) %d path(s)%s", row["slug"], verdict, old, len(hits), detail)
    for path in hits[:6]:
      logger.info("    %s", path)
    if len(hits) > 6:
      logger.info("    ... and %d more", len(hits) - 6)
  stale = [str(r["slug"]) for r in results if r["stale"]]
  old_stale = [str(r["slug"]) for r in results if r["stale_old"]]
  logger.info("")
  logger.info("must re-run: %d of %d (old rule: %d)", len(stale), len(results), len(old_stale))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
