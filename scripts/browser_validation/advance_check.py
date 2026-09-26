"""Phase-2 advance gate for the literature-parity protocol (T11 step 3).

Builds `outputs/browser_validation/layer_a/advance_table.json`: for every in-scope
layer-(a) P-ID (`litparity_schema.IN_SCOPE_LAYER_A`), decides whether that path
"advances" into Phase 2, per the spec's advance rule:

    advances = true iff
      (1) its exact-tier rows (AC-8, from the rsynced `layer_a_exact_validate`
          result at the run ledger's `H_v`) exist and are ALL `status ==
          "validated"` (within bar);
      (2) if the P-ID has sampling lanes (P07, P08, P09, P11 -- AC-10/AC-11),
          those lanes exist and are ALL declared `equiv`; a missing
          `layer_a_sampling_validate` ledger entry blocks every sampling-lane
          P-ID with a reason naming the last `layer_a_sampling_calibrate`
          outcome (this branch, 260925: no sampling-validate entry exists at
          all -- the sampling calibrate ended `budget_exceeded` -- so every
          sampling-lane P-ID is blocked);
      (3) no confirmed defect in `adjudication.json` names the P-ID in its
          `paths` (any severity -- this is a stricter block than the
          core-severity-only rule `adversarial_survived` uses for the global
          grade, per this task's step 3 wording: "no confirmed defect ...
          names it").

A P-ID with zero exact-tier rows at all is blocked with reason "no layer-(a)
rows" regardless of (2)/(3).

Every helper here is pure (accepts already-loaded JSON, never reads a path
itself) so a test can inject two small fixtures -- one advancing path, one
blocked path -- without touching the real run ledger, titanix rsync tree, or
`outputs/browser_validation/litparity/`. `_gather_context`/`main` are the only
filesystem-touching functions, and `main` is the only one that resolves the
real worktree paths.

CLI: `advance_check.py <P-ID>` prints that path's table entry and exits 0 iff
it advances, 1 otherwise (the Phase-2 harness calls this per-path).
`advance_check.py --build` writes the full table to
`outputs/browser_validation/layer_a/advance_table.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import litparity_schema as ls  # noqa: E402

_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]

_LEDGER_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "run_ledger.json"
_ADJUDICATION_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "litparity" / "adjudication.json"
)
_TABLE_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "advance_table.json"

# Lane names as written by `layer_a_sampling_validate.py` (spec "Sampling
# statistics") mapped to the P-ID each lane's equivalence result covers.
LANE_TO_PID: dict[str, str] = {
  "P07@T0.1": "P07",
  "P07@T1.0": "P07",
  "P08": "P08",
  "P09-s": "P09",
  "P11-s": "P11",
}
SAMPLING_LANE_PIDS: frozenset[str] = frozenset(LANE_TO_PID.values())

NO_ROWS_REASON = "no layer-(a) rows"


# --------------------------------------------------------------------------------------
# Pure grouping / lookup helpers
# --------------------------------------------------------------------------------------


def group_rows_by_pid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  """Group AC-8 exact-tier rows by their leading P-ID (`"P05.log_probs"` -> `"P05"`)."""
  by_pid: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    path = row.get("path")
    if not isinstance(path, str) or "." not in path:
      continue
    pid = path.split(".", 1)[0]
    by_pid.setdefault(pid, []).append(row)
  return by_pid


def group_lanes_by_pid(lanes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  """Group AC-10/AC-11 sampling lane results by the P-ID each lane covers."""
  by_pid: dict[str, list[dict[str, Any]]] = {}
  for lane in lanes:
    lane_name = lane.get("lane")
    pid = LANE_TO_PID.get(lane_name) if isinstance(lane_name, str) else None
    if pid is None:
      continue
    by_pid.setdefault(pid, []).append(lane)
  return by_pid


def group_defects_by_pid(defects_confirmed: list[dict[str, Any]]) -> dict[str, list[str]]:
  """Group confirmed-defect ids by every P-ID each defect's `paths` names."""
  by_pid: dict[str, list[str]] = {}
  for defect in defects_confirmed:
    defect_id = defect.get("id")
    if not isinstance(defect_id, str):
      continue
    for pid in defect.get("paths", []):
      if isinstance(pid, str):
        by_pid.setdefault(pid, []).append(defect_id)
  return by_pid


# --------------------------------------------------------------------------------------
# Per-path verdict + table construction (pure)
# --------------------------------------------------------------------------------------


def compute_path_verdict(
  pid: str,
  *,
  exact_rows_by_pid: dict[str, list[dict[str, Any]]],
  sampling_status: str,
  sampling_missing_reason: str | None,
  sampling_lanes_by_pid: dict[str, list[dict[str, Any]]],
  defects_by_pid: dict[str, list[str]],
) -> tuple[bool, list[str]]:
  """Return `(advances, reasons)` for one P-ID (empty `reasons` iff `advances`)."""
  reasons: list[str] = []
  exact_rows = exact_rows_by_pid.get(pid, [])

  if not exact_rows:
    reasons.append(NO_ROWS_REASON)
  else:
    bad_statuses = sorted(
      {str(r.get("status")) for r in exact_rows if r.get("status") != "validated"}
    )
    if bad_statuses:
      reasons.append(f"exact rows not all validated (statuses={bad_statuses})")
    over_bar = sorted(
      r["path"] for r in exact_rows if r.get("ratio") is not None and r["ratio"] > 1.0
    )
    if over_bar:
      reasons.append(f"exact rows over bar: {over_bar}")

  if pid in SAMPLING_LANE_PIDS:
    if sampling_status != "ok":
      reasons.append(sampling_missing_reason or "sampling tier not validated")
    else:
      lanes = sampling_lanes_by_pid.get(pid, [])
      if not lanes:
        reasons.append(f"no sampling lanes found for {pid}")
      else:
        not_equiv = sorted(str(lane.get("lane")) for lane in lanes if not lane.get("equiv"))
        if not_equiv:
          reasons.append(f"sampling lane(s) not equivalent: {not_equiv}")

  defect_ids = defects_by_pid.get(pid)
  if defect_ids:
    reasons.append(f"confirmed defect(s) name this path: {sorted(defect_ids)}")

  return not reasons, reasons


def build_advance_table(
  *,
  exact_rows_by_pid: dict[str, list[dict[str, Any]]],
  exact_result_sha256: str | None,
  sampling_status: str,
  sampling_missing_reason: str | None,
  sampling_lanes_by_pid: dict[str, list[dict[str, Any]]],
  sampling_result_sha256: str | None,
  defects_by_pid: dict[str, list[str]],
  in_scope_pids: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
  """Build the full `{paths: {P-ID: {...}}}` advance table (pure; JSON-serializable)."""
  pids = sorted(in_scope_pids if in_scope_pids is not None else ls.IN_SCOPE_LAYER_A)
  paths: dict[str, Any] = {}
  for pid in pids:
    advances, reasons = compute_path_verdict(
      pid,
      exact_rows_by_pid=exact_rows_by_pid,
      sampling_status=sampling_status,
      sampling_missing_reason=sampling_missing_reason,
      sampling_lanes_by_pid=sampling_lanes_by_pid,
      defects_by_pid=defects_by_pid,
    )
    paths[pid] = {
      "advances": advances,
      "reasons": reasons,
      "exact_rows": exact_rows_by_pid.get(pid, []),
      "sampling_lanes": sampling_lanes_by_pid.get(pid, []) if pid in SAMPLING_LANE_PIDS else [],
    }
  return {
    "paths": paths,
    "exact_result_sha256": exact_result_sha256,
    "sampling_status": sampling_status,
    "sampling_result_sha256": sampling_result_sha256,
  }


# --------------------------------------------------------------------------------------
# Filesystem-touching context gathering (main() only)
# --------------------------------------------------------------------------------------


def _last_ledger_entry(ledger: list[dict[str, Any]], stem: str) -> dict[str, Any] | None:
  matches = [entry for entry in ledger if entry.get("stem") == stem]
  return matches[-1] if matches else None


def _titanix_result_path(worktree_root: Path, stem: str, h_v: str) -> Path:
  return (
    worktree_root
    / "outputs"
    / "browser_validation"
    / "titanix"
    / f"{stem}-{h_v[:12]}"
    / "layer_a"
    / f"{stem}.json"
  )


def _sha256_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_hash(worktree_root: Path) -> str | None:
  try:
    proc = subprocess.run(  # noqa: S603, S607
      ["git", "rev-parse", "HEAD"], cwd=worktree_root, capture_output=True, text=True, check=True
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return proc.stdout.strip()


def gather_context(worktree_root: Path) -> dict[str, Any]:
  """Read the real run ledger + titanix rsync tree + adjudication.json into a context dict.

  The only function besides `main` that touches the filesystem; `build_advance_table`
  and `compute_path_verdict` take the resulting plain-data context (or an
  equivalent hand-built one in tests).
  """
  ledger_path = worktree_root / "outputs" / "browser_validation" / "layer_a" / "run_ledger.json"
  ledger = json.loads(ledger_path.read_text())

  exact_entry = _last_ledger_entry(ledger, "layer_a_exact_validate")
  exact_rows_by_pid: dict[str, list[dict[str, Any]]] = {}
  exact_result_sha256: str | None = None
  if exact_entry is not None:
    result_path = _titanix_result_path(worktree_root, "layer_a_exact_validate", exact_entry["H_v"])
    if result_path.is_file():
      data = json.loads(result_path.read_text())
      exact_rows_by_pid = group_rows_by_pid(data.get("rows", []))
      exact_result_sha256 = _sha256_file(result_path)

  sampling_entry = _last_ledger_entry(ledger, "layer_a_sampling_validate")
  sampling_lanes_by_pid: dict[str, list[dict[str, Any]]] = {}
  sampling_result_sha256: str | None = None
  if sampling_entry is not None and sampling_entry.get("outcome") == "pass":
    sampling_status = "ok"
    sampling_missing_reason = None
    result_path = _titanix_result_path(
      worktree_root, "layer_a_sampling_validate", sampling_entry["H_v"]
    )
    if result_path.is_file():
      data = json.loads(result_path.read_text())
      sampling_lanes_by_pid = group_lanes_by_pid(data.get("lanes", []))
      sampling_result_sha256 = _sha256_file(result_path)
  else:
    sampling_status = "missing"
    calibrate_entry = _last_ledger_entry(ledger, "layer_a_sampling_calibrate")
    outcome = calibrate_entry.get("outcome") if calibrate_entry else "no ledger entry"
    sampling_missing_reason = f"sampling tier not validated ({outcome})"

  defects_by_pid: dict[str, list[str]] = {}
  adjudication_path = (
    worktree_root / "outputs" / "browser_validation" / "litparity" / "adjudication.json"
  )
  if adjudication_path.is_file():
    adjudication = json.loads(adjudication_path.read_text())
    defects_by_pid = group_defects_by_pid(adjudication.get("defects_confirmed", []))

  return {
    "exact_rows_by_pid": exact_rows_by_pid,
    "exact_result_sha256": exact_result_sha256,
    "sampling_status": sampling_status,
    "sampling_missing_reason": sampling_missing_reason,
    "sampling_lanes_by_pid": sampling_lanes_by_pid,
    "sampling_result_sha256": sampling_result_sha256,
    "defects_by_pid": defects_by_pid,
    "git_hash": _git_hash(worktree_root),
  }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("path_id", nargs="?", help="P-ID to check advance status for, e.g. P05.")
  parser.add_argument(
    "--build", action="store_true", help="Write the full advance table to disk and exit 0."
  )
  parser.add_argument(
    "--worktree-root",
    type=Path,
    default=_WORKTREE_ROOT,
    help="Worktree root to resolve the run ledger / titanix tree / litparity dir against.",
  )
  parser.add_argument(
    "--out", type=Path, default=None, help="Override the advance-table output path (--build only)."
  )
  args = parser.parse_args(argv)

  if not args.build and not args.path_id:
    parser.error("either a P-ID or --build is required")

  ctx = gather_context(args.worktree_root)
  table = build_advance_table(
    exact_rows_by_pid=ctx["exact_rows_by_pid"],
    exact_result_sha256=ctx["exact_result_sha256"],
    sampling_status=ctx["sampling_status"],
    sampling_missing_reason=ctx["sampling_missing_reason"],
    sampling_lanes_by_pid=ctx["sampling_lanes_by_pid"],
    sampling_result_sha256=ctx["sampling_result_sha256"],
    defects_by_pid=ctx["defects_by_pid"],
  )
  table["git_hash"] = ctx["git_hash"]

  if args.build:
    out_path = args.out or (
      args.worktree_root / "outputs" / "browser_validation" / "layer_a" / "advance_table.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
      json.dump(table, fh, indent=2)
    print(f"advance_check --build: wrote {out_path}")
    return 0

  entry = table["paths"].get(args.path_id)
  if entry is None:
    print(f"advance_check: unknown P-ID {args.path_id!r}", file=sys.stderr)
    return 2
  print(json.dumps({args.path_id: entry}, indent=2))
  return 0 if entry["advances"] else 1


if __name__ == "__main__":
  sys.exit(main())
