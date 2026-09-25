#!/usr/bin/env python3
"""Re-verify and self-attest the ``mpnn_metric_synthetic_truth`` BP-2 gate (T5).

bathos does not run the test itself (see ``bathos.gate``): a project's own harness must prove
the invariant test currently passes, then call ``bth gate stamp``. This script IS that proof
for T5 -- it refuses to stamp unless both guarded files are committed and clean, then re-runs
the synthetic-truth test file and the no-hollow-tests guard before stamping.

Run AFTER the T5 commit (see the task Gate).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_NAME = "mpnn_metric_synthetic_truth"
GUARDED_FILES = (
  "src/aminx/parity/compare.py",
  "tests/parity/test_compare_metrics.py",
)
JUNIT_PATH = REPO_ROOT / "outputs" / "browser_validation" / "junit_stamp_metric_gate.xml"


def _run(cmd: list[str]) -> None:
  print("+ " + " ".join(cmd), file=sys.stderr)
  subprocess.run(cmd, cwd=REPO_ROOT, check=True)  # noqa: S603


def _assert_committed_and_clean(relative_path: str) -> None:
  """Refuse unless ``relative_path`` is tracked by git with no working-tree diff.

  Scoped to this single guarded path, never the whole tree: the worktree legitimately
  carries many untracked/harness files unrelated to this gate.
  """
  tracked = subprocess.run(  # noqa: S603
    ["git", "ls-files", "--error-unmatch", relative_path],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  if tracked.returncode != 0:
    msg = f"{relative_path} is not committed (git ls-files --error-unmatch failed)"
    raise SystemExit(msg)

  status = subprocess.run(  # noqa: S603
    ["git", "status", "--porcelain", "--", relative_path],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  if status.stdout.strip():
    msg = f"{relative_path} has uncommitted changes:\n{status.stdout}"
    raise SystemExit(msg)


def main() -> int:
  """Verify the guarded files are clean, re-run the gate's test, then stamp it."""
  for guarded in GUARDED_FILES:
    _assert_committed_and_clean(guarded)

  JUNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
  _run(
    [
      "uv",
      "run",
      "--frozen",
      "--extra",
      "dev",
      "pytest",
      "tests/parity/test_compare_metrics.py",
      "-q",
      "-rs",
      f"--junitxml={JUNIT_PATH}",
    ]
  )
  _run(
    [
      "uv",
      "run",
      "--frozen",
      "--extra",
      "dev",
      "python",
      "scripts/browser_validation/assert_no_skips.py",
      str(JUNIT_PATH),
    ]
  )

  _run(["bth", "gate", "stamp", GATE_NAME, "--result", "pass"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
