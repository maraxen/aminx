# ruff: noqa: S101
"""The manifest must be *capable* of grading to "pass".

``check_branch_coverage`` returns ``instrument_invalid`` both when the runs are
stale and when the manifest itself is malformed -- a slug that no vehicle
writes, a row whose id is not the mutant name, a sidecar row with no ledger
entry. The single verdict word hides which, so a structural defect looks
exactly like "the runs are old" and survives until someone spends hours on a
vehicle wave and lands red anyway.

This is the positive half of that control: stub the five resolver seams
``check_branch_coverage`` already exposes so that every slug reports a flawless
run, and require the verdict to be "pass". The negative half is
``test_branch_coverage``, which runs against the real ledger -- so between them
the instrument is shown to discriminate rather than to always agree.

Everything structural is real here: the manifest on disk, the grouping by slug,
and the controls files. Only run state is stubbed.
"""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from knob_gate._coverage import check_branch_coverage

_HERE = Path(__file__).resolve().parent
_SHA = "a" * 40
_WEIGHTS_SHA = "f" * 64


def _rows_by_slug() -> dict[str, set[str]]:
  manifest = tomllib.loads((_HERE / "branch_manifest.toml").read_text(encoding="utf-8"))
  by_slug: dict[str, set[str]] = defaultdict(set)
  for row in manifest["branch"]:
    vehicle = row.get("vehicle", {})
    if vehicle.get("kind") == "sidecar":
      by_slug[str(vehicle["slug"])].add(str(row["id"]))
  return dict(by_slug)


def test_a_flawless_wave_would_grade_pass(tmp_path: Path) -> None:
  """Every sidecar slug reporting a perfect run must grade to "pass"."""
  by_slug = _rows_by_slug()
  assert by_slug, "no sidecar rows; the manifest would read as covered by doing nothing"

  controls_of: dict[str, Path] = {}
  for slug, ids in by_slug.items():
    path = tmp_path / slug / _SHA[:8] / "branch_controls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps(
        {
          "clean": "pass",
          "mutants": dict.fromkeys(sorted(ids), "failed"),
          "weights": {f"{slug}/weights.pt": _WEIGHTS_SHA},
        },
      ),
      encoding="utf-8",
    )
    controls_of[slug] = path

  ledger = tmp_path / "ledger.toml"
  body = "[sidecar]\n"
  for slug in sorted(by_slug):
    body += f'[sidecar.{slug}]\nbth_run_id = "run-{slug}"\n'
  ledger.write_text(body, encoding="utf-8")

  slug_of_run = {f"run-{slug}": slug for slug in by_slug}

  def resolve_run(run_id: str) -> dict[str, Any] | None:
    slug = slug_of_run.get(run_id)
    if slug is None:
      return None
    return {
      "status": "completed",
      "outcome": "pass",
      "git_dirty": False,
      "git_hash": _SHA,
      "sidecar_sha256": f"digest-{slug}",
      # The raw --mutants string is what step 1c compares to the row ids.
      "argv": ["python3", f"{slug}.py", "--mutants", ",".join(sorted(by_slug[slug]))],
      "output_paths": [str(controls_of[slug])],
    }

  outcomes = tmp_path / "outcomes.jsonl"
  outcomes.write_text("", encoding="utf-8")

  verdict = check_branch_coverage(
    _HERE / "branch_manifest.toml",
    outcomes,
    ledger,
    resolve_run=resolve_run,
    changed_paths=lambda _hash: [],
    is_ancestor=lambda _hash: True,
    registry_sha256=lambda _artifact: _WEIGHTS_SHA,
    sidecar_digest=lambda slug: f"digest-{slug}",
  )
  assert verdict == "pass", (
    f"the manifest cannot grade to pass even with flawless runs ({verdict}); "
    "a vehicle wave would not fix this"
  )


def test_a_mutant_set_mismatch_is_caught(tmp_path: Path) -> None:
  """The same harness must FAIL when one slug's argv does not match its rows.

  Without this, the test above would pass just as happily against a stub that
  ignored the mutant comparison entirely.
  """
  by_slug = _rows_by_slug()
  victim = sorted(by_slug)[0]

  controls_of: dict[str, Path] = {}
  for slug, ids in by_slug.items():
    path = tmp_path / slug / _SHA[:8] / "branch_controls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps(
        {
          "clean": "pass",
          "mutants": dict.fromkeys(sorted(ids), "failed"),
          "weights": {f"{slug}/weights.pt": _WEIGHTS_SHA},
        },
      ),
      encoding="utf-8",
    )
    controls_of[slug] = path

  ledger = tmp_path / "ledger.toml"
  body = "[sidecar]\n"
  for slug in sorted(by_slug):
    body += f'[sidecar.{slug}]\nbth_run_id = "run-{slug}"\n'
  ledger.write_text(body, encoding="utf-8")

  slug_of_run = {f"run-{slug}": slug for slug in by_slug}

  def resolve_run(run_id: str) -> dict[str, Any] | None:
    slug = slug_of_run.get(run_id)
    if slug is None:
      return None
    listed = sorted(by_slug[slug])
    if slug == victim:
      # Exactly the trap-4 shape: an extra id in argv that has no manifest row.
      listed = [*listed, "a_positive_control_riding_along"]
    return {
      "status": "completed",
      "outcome": "pass",
      "git_dirty": False,
      "git_hash": _SHA,
      "sidecar_sha256": f"digest-{slug}",
      "argv": ["python3", f"{slug}.py", "--mutants", ",".join(listed)],
      "output_paths": [str(controls_of[slug])],
    }

  outcomes = tmp_path / "outcomes.jsonl"
  outcomes.write_text("", encoding="utf-8")

  verdict = check_branch_coverage(
    _HERE / "branch_manifest.toml",
    outcomes,
    ledger,
    resolve_run=resolve_run,
    changed_paths=lambda _hash: [],
    is_ancestor=lambda _hash: True,
    registry_sha256=lambda _artifact: _WEIGHTS_SHA,
    sidecar_digest=lambda slug: f"digest-{slug}",
  )
  assert verdict != "pass", (
    "an argv mutant set that does not match the manifest graded as pass; "
    "step 1c's exact-set comparison is not being exercised"
  )
