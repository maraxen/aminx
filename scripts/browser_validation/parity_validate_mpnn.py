"""Literature-parity graded verdict (T11 step 2, `parity_validate_mpnn.bth.toml`).

Grades the global literature-parity claim (spec "Literature-parity run" /
"Grade inputs", R2-C8/O8) from three already-run, already-committed sources --
this script never re-runs the layer-(a) exact/sampling engines itself:

1. **The run ledger identity check (R3-C3).** The graded HEAD legitimately
   differs from each validate script's own commit (T8/T10/T11 all commit
   afterwards), so a validate's identity is its ledger `H_v`, never `HEAD`.
   For `layer_a_exact_validate` and `layer_a_sampling_validate` this takes the
   LAST ledger entry, then requires (a) `H_v` is an ancestor of `HEAD`
   (`git merge-base --is-ancestor`), (b) every `FROZEN` path is byte-identical
   between `H_v` and `HEAD` (`git diff --quiet H_v HEAD -- <FROZEN>` --
   `pyproject.toml` added per this task's carry-forward note F-C8), and
   (c) an identity-form `bth sql` query against NATIVE columns only
   (`git_hash`, `git_dirty`, `dependency_lock_sha256`, `sidecar_sha256`,
   `differential_status`, `exit_code`, `status`) returns exactly one matching
   row (`SELECT count(*) ... = [[1]]`), whose `outcome` is then read by a
   second query. bathos 84be544e drops `runs.metadata` entirely on write, so
   this NEVER queries `json_extract(metadata, ...)` -- the params/lock/sidecar
   binding is native-column + `git show H_v:<path>` sha256, not metadata.
2. **The clauses anti-HARKing gate (O8).** `sha256(litparity/clauses.json)`
   must equal the value the orchestrator committed to
   `preregistered_params.json:litparity.clauses_sha256` BEFORE any grading run
   -- a mismatch means `core`/reconciliation-note assignment happened (or
   could have happened) after seeing adjudication results, so grading is
   refused outright (exit 7).
3. **The re-derived invariant suite and the litparity directory.** The
   orchestrator-owned `tests/parity/test_mpnn_reference_invariants.py` is run
   fresh (`-m parity_heavy`, junit-parsed) so `invariant_pass` is never a
   stale claim; `litparity_schema.py` validates `clauses.json` (always) and
   `adjudication.json` (required in a real run -- exit 7 if absent, spec's
   anti-HARKing note that adjudication happens in a LATER phase than
   `clauses.json` but must exist before a grading run completes).

**The O1 mapping is a single, hard-coded table**, never re-derived: both
validates' identity-verified outcome `pass` -> `prereq_status = "ok"`, rung
`"R1"`; exact `partial_headroom` with sampling `pass` -> `"headroom"`, rung
`"R2"`; every other combination (including either ledger entry, ancestry
check, FROZEN check, or identity query being absent/failed) -> `"missing"` or
`"failed"`, `parity_grade` forced to `"FAIL"` WITHOUT calling
`bathos.parity.compute_grade` (only `prereq_status in ("ok", "headroom")`
calls it) -- and the FULL result (every `result_schema` key, `ceilings`,
`metadata`) is still written before this script exits 6. This branch is the
one live on `dogfood/xtrax-probing-stage2` right now: the ledger has a
`layer_a_exact_validate` entry (`partial_headroom`) but no
`layer_a_sampling_validate` entry at all (the sampling calibrate ended
`budget_exceeded`, so no sampling params exist to validate against) ->
`prereq_status = "missing"`.

Exit codes: 5 (`bth sync --pull`/`bth compact` failed; skippable only via
`--no-sync`, tests only -- a real `bth run` always syncs first, R2-C1); 6
(`prereq_status` not in `{"ok", "headroom"}`, result written); 7 (clauses
sha256 mismatch, `litparity_schema.py` validation failure, or a real run's
`adjudication.json` missing). Exit 0 otherwise -- the sidecar's own
`[outcomes]` conditions, not this exit code, are what bathos evaluates into
`pass`/`partial`/`fail`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import litparity_schema as lps  # noqa: E402

_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]

PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "preregistered_params.json"
)
LEDGER_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "run_ledger.json"
LITPARITY_DIR = _WORKTREE_ROOT / "outputs" / "browser_validation" / "litparity"
INVARIANTS_TEST_PATH = _WORKTREE_ROOT / "tests" / "parity" / "test_mpnn_reference_invariants.py"

EXACT_STEM = "layer_a_exact_validate"
SAMPLING_STEM = "layer_a_sampling_validate"

# Spec T11 step 2 FROZEN list, `pyproject.toml` added per this task's carry-forward
# note F-C8 (optional hardening). Byte-identical between a validate's `H_v` and
# `HEAD` is required -- the engine the graded HEAD imports must be the SAME engine
# that stem's validate run actually measured against.
FROZEN_PATHS: tuple[str, ...] = (
  "src/",
  "uv.lock",
  "pyproject.toml",
  "tests/parity/test_full_model_parity.py",
  "tests/parity/test_soluble_membrane_parity.py",
  "tests/parity/test_sidechain_context_parity.py",
  "tests/parity/test_packer_parity.py",
  "scripts/browser_validation/layer_a_common.py",
  "scripts/browser_validation/layer_a_exact.py",
  "scripts/browser_validation/layer_a_exact_validate.py",
  "scripts/browser_validation/layer_a_sampling.py",
  "scripts/browser_validation/layer_a_sampling_validate.py",
  "scripts/browser_validation/fixtures.py",
  "scripts/browser_validation/parse_parity.py",
  "scripts/browser_validation/pins.py",
  "scripts/browser_validation/_lattice.py",
  "scripts/browser_validation/reference_pins.json",
  "scripts/browser_validation/titanix_run.sh",
  "outputs/browser_validation/fixtures/manifest.json",
  "outputs/browser_validation/fixtures/raw/",
)

EXIT_SYNC_FAILED = 5
EXIT_PREREQ_BAD = 6
EXIT_ANTI_HARKING_OR_SCHEMA = 7

RESULT_SCHEMA_KEYS = (
  "parity_grade",
  "prereq_status",
  "exact_validate_outcome",
  "sampling_validate_outcome",
  "clause_parity_pct",
  "invariant_pass",
  "n_invariant_tests",
  "n_invariant_skipped",
  "adversarial_survived",
  "reproduction_rung",
  "ambiguity_load",
  "n_ambiguous",
)


# --------------------------------------------------------------------------------------
# Pure helpers: identity-form SQL construction
# --------------------------------------------------------------------------------------


def _identity_where_clause(stem: str, h_v: str, sidecar_sha256: str, lock_sha256: str) -> str:
  """The shared WHERE clause both identity-form queries filter on (native columns only)."""
  return (
    f"command LIKE '%{stem}.py%' AND git_hash = '{h_v}' AND git_dirty = false "
    f"AND dependency_lock_sha256 = '{lock_sha256}' AND sidecar_sha256 = '{sidecar_sha256}' "
    "AND differential_status = 'passed' AND exit_code = 0 AND status = 'completed'"
  )


def identity_query_sql(stem: str, h_v: str, sidecar_sha256: str, lock_sha256: str) -> str:
  """`SELECT count(*) ...` -- must return exactly `[[1]]` (T7 step-3 identity form)."""
  where = _identity_where_clause(stem, h_v, sidecar_sha256, lock_sha256)
  return f"SELECT count(*) FROM runs WHERE {where}"


def outcome_query_sql(stem: str, h_v: str, sidecar_sha256: str, lock_sha256: str) -> str:
  """`SELECT outcome ...` against the SAME identity filter, for the row the count found."""
  where = _identity_where_clause(stem, h_v, sidecar_sha256, lock_sha256)
  return f"SELECT outcome FROM runs WHERE {where} ORDER BY timestamp DESC LIMIT 1"


# --------------------------------------------------------------------------------------
# Pure helpers: the O1 mapping
# --------------------------------------------------------------------------------------


def determine_prereq_status(
  *,
  exact_ready: bool,
  exact_outcome: str | None,
  sampling_ready: bool,
  sampling_outcome: str | None,
) -> tuple[str, str]:
  """The SINGLE O1 mapping table. Returns `(prereq_status, reproduction_rung)`.

  `exact_ready`/`sampling_ready` are true iff that stem's ledger entry exists AND
  passed ancestry + FROZEN + identity-query verification (so `*_outcome` is a
  DB-read, identity-verified outcome, never a raw ledger-JSON field).
  """
  if not (exact_ready and sampling_ready):
    return "missing", "R4"
  if exact_outcome == "pass" and sampling_outcome == "pass":
    return "ok", "R1"
  if exact_outcome == "partial_headroom" and sampling_outcome == "pass":
    return "headroom", "R2"
  return "failed", "R4"


def display_outcome(entry: dict[str, Any] | None, evaluated_outcome: str | None) -> str:
  """The result JSON's `*_validate_outcome` field: identity-verified outcome if we have
  one, else the raw ledger outcome for diagnostics, else the literal `"missing"`.
  """
  if evaluated_outcome is not None:
    return evaluated_outcome
  if entry is not None:
    return str(entry.get("outcome", "unknown"))
  return "missing"


# --------------------------------------------------------------------------------------
# Pure helpers: clause parity / adversarial survival (spec "Grade inputs")
# --------------------------------------------------------------------------------------


def compute_clause_parity(clauses: list[dict[str, Any]]) -> tuple[float, int, int]:
  """Return `(clause_parity_pct, numerator, denominator)`.

  Core-restricted in BOTH terms: `denominator = |{c: core}|`,
  `numerator = |{c: core AND verdict == MATCH}|`. AMBIGUOUS clauses are excluded
  from both terms regardless of their `core` flag.
  """
  core_clauses = [c for c in clauses if c.get("core") is True and c.get("verdict") != "AMBIGUOUS"]
  denominator = len(core_clauses)
  numerator = sum(1 for c in core_clauses if c.get("verdict") == "MATCH")
  pct = (numerator / denominator) if denominator else 0.0
  return pct, numerator, denominator


def compute_adversarial_survived(defects_confirmed: list[dict[str, Any]]) -> bool:
  """`not any(d.severity == 'core' for d in defects_confirmed)` (spec "Grade inputs")."""
  return not any(d.get("severity") == "core" for d in defects_confirmed)


# --------------------------------------------------------------------------------------
# Pure helper: invariant junit parsing
# --------------------------------------------------------------------------------------


def parse_invariant_junit(path: Path) -> dict[str, Any]:
  """Parse the invariant suite's `--junitxml` report into the three schema fields.

  `invariant_pass` iff zero failures, zero errors, zero skips, AND at least 8
  tests ran (spec: "≥ 8 invariants ... zero skips").
  """
  root = ET.parse(path).getroot()
  suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
  tests = sum(int(s.get("tests", 0)) for s in suites)
  failures = sum(int(s.get("failures", 0)) for s in suites)
  errors = sum(int(s.get("errors", 0)) for s in suites)
  skipped = sum(int(s.get("skipped", 0)) for s in suites)
  invariant_pass = failures == 0 and errors == 0 and skipped == 0 and tests >= 8
  return {
    "n_invariant_tests": tests,
    "n_invariant_skipped": skipped,
    "invariant_pass": invariant_pass,
  }


# --------------------------------------------------------------------------------------
# I/O: git / bth subprocess helpers
# --------------------------------------------------------------------------------------


def git_show_bytes(worktree_root: Path, ref: str, rel_path: str) -> bytes:
  proc = subprocess.run(  # noqa: S603, S607
    ["git", "show", f"{ref}:{rel_path}"], cwd=worktree_root, capture_output=True, check=True
  )
  return proc.stdout


def check_is_ancestor(h_v: str, worktree_root: Path) -> bool:
  proc = subprocess.run(  # noqa: S603, S607
    ["git", "merge-base", "--is-ancestor", h_v, "HEAD"], cwd=worktree_root, check=False
  )
  return proc.returncode == 0


def check_frozen_unchanged(h_v: str, worktree_root: Path, frozen_paths: tuple[str, ...]) -> bool:
  proc = subprocess.run(  # noqa: S603, S607
    ["git", "diff", "--quiet", h_v, "HEAD", "--", *frozen_paths], cwd=worktree_root, check=False
  )
  return proc.returncode == 0


def run_bth_sql(sql: str, bth_bin: str, cwd: Path) -> dict[str, Any]:
  proc = subprocess.run(  # noqa: S603
    [bth_bin, "sql", "--sql", sql], cwd=cwd, capture_output=True, text=True, check=False
  )
  if proc.returncode != 0:
    return {"rows": [], "count": 0, "error": proc.stderr}
  try:
    payload: dict[str, Any] = json.loads(proc.stdout)
  except json.JSONDecodeError:
    return {"rows": [], "count": 0, "error": f"could not parse `{bth_bin} sql` output"}
  return payload


def identity_check(stem: str, h_v: str, worktree_root: Path, bth_bin: str) -> dict[str, Any]:
  """Run both identity-form queries for one stem; return `{ok, outcome, reason}`."""
  sidecar_rel = f"scripts/browser_validation/{stem}.bth.toml"
  try:
    sidecar_sha256 = hashlib.sha256(git_show_bytes(worktree_root, h_v, sidecar_rel)).hexdigest()
    lock_sha256 = hashlib.sha256(git_show_bytes(worktree_root, h_v, "uv.lock")).hexdigest()
  except subprocess.CalledProcessError as exc:
    return {"ok": False, "outcome": None, "reason": f"`git show {h_v}:...` failed: {exc}"}

  count_result = run_bth_sql(
    identity_query_sql(stem, h_v, sidecar_sha256, lock_sha256), bth_bin, worktree_root
  )
  if count_result.get("rows") != [[1]]:
    return {
      "ok": False,
      "outcome": None,
      "reason": f"identity-form count query did not return [[1]]: {count_result}",
    }

  outcome_result = run_bth_sql(
    outcome_query_sql(stem, h_v, sidecar_sha256, lock_sha256), bth_bin, worktree_root
  )
  rows = outcome_result.get("rows") or []
  if len(rows) != 1:
    return {"ok": False, "outcome": None, "reason": "outcome query did not return exactly one row"}
  return {"ok": True, "outcome": rows[0][0], "reason": None}


def evaluate_stem_prereq(
  stem: str, entry: dict[str, Any] | None, worktree_root: Path, bth_bin: str
) -> dict[str, Any]:
  """Ancestry + FROZEN + identity-query verification for one ledger stem.

  Returns `{ready, outcome, reasons}`; `outcome` is `None` unless `ready`.
  """
  if entry is None:
    return {"ready": False, "outcome": None, "reasons": [f"no ledger entry for {stem}"]}

  h_v = entry["H_v"]
  reasons: list[str] = []
  if not check_is_ancestor(h_v, worktree_root):
    reasons.append(f"H_v {h_v} is not an ancestor of HEAD")
  if not check_frozen_unchanged(h_v, worktree_root, FROZEN_PATHS):
    reasons.append(f"FROZEN paths differ between H_v {h_v} and HEAD")

  identity = identity_check(stem, h_v, worktree_root, bth_bin)
  if not identity["ok"]:
    reasons.append(identity["reason"] or "identity-form query failed")

  ready = not reasons
  return {"ready": ready, "outcome": identity.get("outcome") if ready else None, "reasons": reasons}


def _last_ledger_entry(ledger: list[dict[str, Any]], stem: str) -> dict[str, Any] | None:
  matches = [entry for entry in ledger if entry.get("stem") == stem]
  return matches[-1] if matches else None


def run_sync_and_compact(bth_bin: str, worktree_root: Path) -> bool:
  """`bth sync --pull --remote-name titanix` then `bth compact`; True iff both exit 0."""
  sync = subprocess.run(  # noqa: S603
    [bth_bin, "sync", "--pull", "--remote-name", "titanix"], cwd=worktree_root, check=False
  )
  if sync.returncode != 0:
    logger.error("parity_validate_mpnn: `%s sync --pull --remote-name titanix` failed", bth_bin)
    return False
  compact = subprocess.run([bth_bin, "compact"], cwd=worktree_root, check=False)  # noqa: S603
  if compact.returncode != 0:
    logger.error("parity_validate_mpnn: `%s compact` failed", bth_bin)
    return False
  return True


def run_invariant_tests(worktree_root: Path, junit_path: Path) -> int:
  """Run the orchestrator-owned invariant suite fresh; return the pytest exit code."""
  junit_path.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    sys.executable,
    "-m",
    "pytest",
    str(INVARIANTS_TEST_PATH),
    "-m",
    "parity_heavy",
    "-q",
    f"--junitxml={junit_path}",
  ]
  proc = subprocess.run(cmd, cwd=worktree_root, check=False)  # noqa: S603
  return proc.returncode


def call_compute_grade(evidence: dict[str, Any]) -> dict[str, Any]:
  """`bathos.parity.compute_grade`, imported lazily (only called when prereq ok/headroom)."""
  import bathos.parity as bp  # noqa: PLC0415

  grade_result = bp.compute_grade(bp.ParityEvidence(**evidence))
  return {"grade": grade_result.grade, "ceilings": grade_result.ceilings}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--no-sync",
    action="store_true",
    help="Skip `bth sync --pull`/`bth compact` (tests only -- a real run always syncs).",
  )
  parser.add_argument(
    "--bth-bin",
    default=None,
    help="bth binary to invoke (default: $BTH_BIN, else `bth` on PATH).",
  )
  parser.add_argument("--worktree-root", type=Path, default=_WORKTREE_ROOT)
  args = parser.parse_args(argv)

  bth_bin = args.bth_bin or os.environ.get("BTH_BIN", "bth")
  worktree_root = args.worktree_root

  if not args.no_sync and not run_sync_and_compact(bth_bin, worktree_root):
    return EXIT_SYNC_FAILED

  with LEDGER_PATH.open() as fh:
    ledger = json.load(fh)
  exact_entry = _last_ledger_entry(ledger, EXACT_STEM)
  sampling_entry = _last_ledger_entry(ledger, SAMPLING_STEM)

  exact_eval = evaluate_stem_prereq(EXACT_STEM, exact_entry, worktree_root, bth_bin)
  sampling_eval = evaluate_stem_prereq(SAMPLING_STEM, sampling_entry, worktree_root, bth_bin)

  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  clauses_path = LITPARITY_DIR / "clauses.json"
  actual_clauses_sha256 = hashlib.sha256(clauses_path.read_bytes()).hexdigest()
  expected_clauses_sha256 = params.get("litparity", {}).get("clauses_sha256")
  if actual_clauses_sha256 != expected_clauses_sha256:
    print(
      "parity_validate_mpnn: sha256(litparity/clauses.json) "
      f"{actual_clauses_sha256!r} != preregistered_params.json:litparity.clauses_sha256 "
      f"{expected_clauses_sha256!r} (O8 anti-HARKing gate) -- refusing to grade",
      file=sys.stderr,
    )
    return EXIT_ANTI_HARKING_OR_SCHEMA

  prereq_status, reproduction_rung = determine_prereq_status(
    exact_ready=exact_eval["ready"],
    exact_outcome=exact_eval["outcome"],
    sampling_ready=sampling_eval["ready"],
    sampling_outcome=sampling_eval["outcome"],
  )

  junit_path = worktree_root / "outputs" / "browser_validation" / "junit_parity_invariants.xml"
  run_invariant_tests(worktree_root, junit_path)
  invariant_info = parse_invariant_junit(junit_path)

  schema_errors = lps.validate(LITPARITY_DIR)
  if schema_errors:
    print(
      f"parity_validate_mpnn: litparity_schema validation failed ({len(schema_errors)} "
      "violation(s)):",
      file=sys.stderr,
    )
    for err in schema_errors:
      print(f"  - {err}", file=sys.stderr)
    return EXIT_ANTI_HARKING_OR_SCHEMA

  adjudication_path = LITPARITY_DIR / "adjudication.json"
  if not adjudication_path.is_file():
    print(
      f"parity_validate_mpnn: {adjudication_path} does not exist -- a real grading run "
      "requires it (exit 7)",
      file=sys.stderr,
    )
    return EXIT_ANTI_HARKING_OR_SCHEMA

  clauses_obj = json.loads(clauses_path.read_text())
  clauses = [c for c in clauses_obj.get("clauses", []) if isinstance(c, dict)]
  clause_parity_pct, _numerator, _denominator = compute_clause_parity(clauses)
  n_ambiguous, ambiguity_load = lps.compute_ambiguity(clauses)

  adjudication_obj = json.loads(adjudication_path.read_text())
  defects_confirmed = adjudication_obj.get("defects_confirmed", [])
  adversarial_survived = compute_adversarial_survived(defects_confirmed)

  if prereq_status in ("ok", "headroom"):
    grade_info = call_compute_grade(
      {
        "clause_parity_pct": clause_parity_pct,
        "adversarial_survived": adversarial_survived,
        "invariant_pass": invariant_info["invariant_pass"],
        "reproduction_rung": reproduction_rung,
        "ambiguity_load": ambiguity_load,
      }
    )
    parity_grade = grade_info["grade"]
    ceilings = grade_info["ceilings"]
  else:
    parity_grade = "FAIL"
    ceilings = {"prereq_status": "FAIL"}

  result: dict[str, Any] = {
    "parity_grade": parity_grade,
    "prereq_status": prereq_status,
    "exact_validate_outcome": display_outcome(exact_entry, exact_eval["outcome"]),
    "sampling_validate_outcome": display_outcome(sampling_entry, sampling_eval["outcome"]),
    "clause_parity_pct": clause_parity_pct,
    "invariant_pass": invariant_info["invariant_pass"],
    "n_invariant_tests": invariant_info["n_invariant_tests"],
    "n_invariant_skipped": invariant_info["n_invariant_skipped"],
    "adversarial_survived": adversarial_survived,
    "reproduction_rung": reproduction_rung,
    "ambiguity_load": ambiguity_load,
    "n_ambiguous": n_ambiguous,
    "ceilings": ceilings,
    "metadata": {"parity_run_type": "literature_parity"},
    "exact_reasons": exact_eval["reasons"],
    "sampling_reasons": sampling_eval["reasons"],
    **lac.provenance(PARAMS_PATH),
  }
  lac.emit(result, args.out)
  logger.info(
    "parity_validate_mpnn: prereq_status=%s parity_grade=%s reproduction_rung=%s",
    prereq_status,
    parity_grade,
    reproduction_rung,
  )

  if prereq_status not in ("ok", "headroom"):
    return EXIT_PREREQ_BAD
  return 0


if __name__ == "__main__":
  sys.exit(main())
