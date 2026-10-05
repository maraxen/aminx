"""Layer (a) exact-tier validation (T7b, step 4, held-out fixture set B).

Bathos staging's "validation" twin for `layer_a_exact.py`'s engine, run against
`layer_a_exact_calibrate.py`'s committed `preregistered_params.json`. Every
invocation -- `--help` excepted (argparse intercepts it before `main` runs) --
starts with the anti-HARKing gate (spec "Bathos staging": "every validation script
refuses to start unless `outputs/browser_validation/layer_a/preregistered_params.json`
is tracked and clean"): `git ls-files --error-unmatch` and `git diff --quiet HEAD --`
on the params file. Before T7b/T7c's params commit lands, this ALWAYS fails --
`--dry-run` and `--smoke` included -- by design (T7 fixer scope note: "make sure
--dry-run/--smoke fail with a clear message now and will work once params are
committed").

**Differential pre-flight (spec "Differential pre-flight rules").** `bth run`
re-executes this script's exact argv up to two more times with
`BTH_DIFFERENTIAL_{KNOB,VALUE,PHASE}` set (`layer_a_common.differential_mode`);
whenever `BTH_DIFFERENTIAL_PHASE` is set this script:

- exits 2 unless `BTH_DIFFERENTIAL_KNOB == "AMINX_BV_PERTURB"` (the sidecar's own
  `[differential].knob`);
- computes ONLY `layer_a_exact.sentinel_ratio_to_bar` on the params' pre-registered
  `exact.reduced_subset`, with the calibrated `w_out_bias_perturb_magnitude`, the
  perturbation applied iff `BTH_DIFFERENTIAL_VALUE == "1"`;
- writes ONLY to `$BTH_RESULTS_PATH` (`layer_a_common.emit` already skips `--out`
  whenever a differential phase is active -- never reimplemented here).

This branch runs BEFORE `--dry-run`/`--smoke` are consulted: bathos passes the
identical argv for the main arm and both differential arms, so the flag the user
actually typed is irrelevant once `BTH_DIFFERENTIAL_PHASE` is set.

**`--dry-run`** checks imports (this module's own top-level imports already ran by
the time `main` executes), that the fixture manifest loads and has at least one
fixture, and that this script's own sidecar (`layer_a_exact_validate.bth.toml`)
`[differential].min_effect` is `> 0` and equals the committed `exact.min_effect`
(R2-C6 equality, not just both being positive).

**`--smoke`** runs `layer_a_exact.row_p05` on ONE fixture from `--fixture-set`
(default the held-out set `B`) -- never the full engine -- in well under a minute,
exactly like the calibrate script's own `--smoke`, plus `layer_a_common.prereq_check`
(spec step 4: "runs `prereq_check('layer_a_exact', 'exact')`") so `prereq_ok` is
always present in the written result, even for a smoke run.

Every non-differential-arm invocation writes to `$BTH_RESULTS_PATH` AND `--out`
(`layer_a_common.emit`), and, when `n_skipped > 0` after writing, exits 4 per the
common context's `reference_call` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_a_exact as lae  # noqa: E402

_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "preregistered_params.json"
)
_MANIFEST_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
_SIDECAR_PATH = _SCRIPT_DIR / "layer_a_exact_validate.bth.toml"
_REFERENCE_PINS_PATH = _SCRIPT_DIR / "reference_pins.json"

DIFFERENTIAL_KNOB = "AMINX_BV_PERTURB"
SMOKE_FIXTURE_NAME = "6MRR"

EXIT_ANTI_HARKING = 5
EXIT_DIFFERENTIAL_KNOB_MISMATCH = 2
EXIT_SKIPPED = 4


def _relative_to_worktree(path: Path) -> str:
  return str(path.resolve().relative_to(_WORKTREE_ROOT))


def _params_committed_and_clean() -> tuple[bool, str]:
  """Anti-HARKing gate: PARAMS_PATH must be `git`-tracked AND clean at HEAD.

  Runs `git ls-files --error-unmatch` (tracked) and `git diff --quiet HEAD --`
  (no uncommitted changes) against the worktree root, matching the spec's
  "Bathos staging" recipe exactly.
  """
  if not PARAMS_PATH.is_file():
    return False, f"{PARAMS_PATH} does not exist (calibrate has not been committed yet)"
  rel = _relative_to_worktree(PARAMS_PATH)
  tracked = subprocess.run(  # noqa: S603, S607
    ["git", "ls-files", "--error-unmatch", rel],
    cwd=_WORKTREE_ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  if tracked.returncode != 0:
    return False, f"{rel} is not tracked by git (`git ls-files --error-unmatch` failed)"
  clean = subprocess.run(  # noqa: S603, S607
    ["git", "diff", "--quiet", "HEAD", "--", rel],
    cwd=_WORKTREE_ROOT,
    check=False,
  )
  if clean.returncode != 0:
    return False, f"{rel} has uncommitted changes (`git diff --quiet HEAD --` failed)"
  return True, ""


def _load_manifest() -> dict[str, Any]:
  with _MANIFEST_PATH.open() as fh:
    return json.load(fh)


def _fixtures_for_set(manifest: dict[str, Any], fixture_set: str) -> list[dict[str, Any]]:
  return [f for f in manifest["fixtures"] if f.get("set") == fixture_set]


def _protein_fixtures(fixtures_for_set: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [f for f in fixtures_for_set if f.get("kind") == "protein"]


def _pick_smoke_fixture(protein_fixtures: list[dict[str, Any]]) -> dict[str, Any]:
  for fixture in protein_fixtures:
    if fixture["name"] == SMOKE_FIXTURE_NAME:
      return fixture
  if not protein_fixtures:
    msg = "no protein fixtures in the requested --fixture-set to run a P05 smoke row on"
    raise SystemExit(msg)
  return min(protein_fixtures, key=lambda f: f.get("L", sys.maxsize))


def _reduced_subset_fixtures(manifest: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
  by_name = {f["name"]: f for f in manifest["fixtures"]}
  missing = [name for name in names if name not in by_name]
  if missing:
    msg = f"exact.reduced_subset names {missing!r} not found in the fixture manifest"
    raise SystemExit(msg)
  return [by_name[name] for name in names]


def _reference_commit() -> str:
  with _REFERENCE_PINS_PATH.open() as fh:
    pins_data = json.load(fh)
  return str(pins_data.get("ligandmpnn_commit", ""))


def _worst_row(rows: list[dict[str, Any]]) -> tuple[str | None, float]:
  worst_path: str | None = None
  worst_ratio = 0.0
  for row in rows:
    ratio = row.get("ratio")
    if ratio is None:
      continue
    if worst_path is None or ratio > worst_ratio:
      worst_path = row["path"]
      worst_ratio = ratio
  return worst_path, worst_ratio


def _sidecar_min_effect() -> float:
  with _SIDECAR_PATH.open("rb") as fh:
    doc = tomllib.load(fh)
  return float(doc["differential"]["min_effect"])


def run_differential_arm(
  mode: dict[str, Any], params: dict[str, Any], args: argparse.Namespace
) -> int:
  """One `bth run` differential arm: only `sentinel_ratio_to_bar`, only `$BTH_RESULTS_PATH`."""
  if mode["knob"] != DIFFERENTIAL_KNOB:
    print(
      f"layer_a_exact_validate: BTH_DIFFERENTIAL_KNOB={mode['knob']!r}, expected "
      f"{DIFFERENTIAL_KNOB!r} (this script's sidecar knob)",
      file=sys.stderr,
    )
    return EXIT_DIFFERENTIAL_KNOB_MISMATCH

  exact = params["exact"]
  manifest = _load_manifest()
  subset = _reduced_subset_fixtures(manifest, exact["reduced_subset"])
  magnitude = exact["controls"]["w_out_bias_perturb_magnitude"]
  perturb = mode["value"] == "1"

  metric = lae.sentinel_ratio_to_bar(
    subset, weight_source="eqx", perturb=perturb, magnitude=magnitude
  )
  result = {
    "sentinel_ratio_to_bar": metric,
    "differential_phase": mode["phase"],
    "differential_value": mode["value"],
    "fixture_set": args.fixture_set,
  }
  lac.emit(result, args.out)
  logger.info(
    "layer_a_exact_validate: differential arm value=%s sentinel_ratio_to_bar=%.6g",
    mode["value"],
    metric,
  )
  return 0


def run_dry_run(params: dict[str, Any], args: argparse.Namespace) -> int:
  manifest = _load_manifest()
  fixtures_for_set = _fixtures_for_set(manifest, args.fixture_set)
  sidecar_min_effect = _sidecar_min_effect()
  params_min_effect = float(params.get("exact", {}).get("min_effect", 0.0))

  checks = {
    "manifest_loads": bool(manifest.get("fixtures")),
    "fixture_set_nonempty": bool(fixtures_for_set),
    "sidecar_min_effect_positive": sidecar_min_effect > 0.0,
    "sidecar_min_effect_matches_params": sidecar_min_effect == params_min_effect,
  }
  ok = all(checks.values())
  result = {
    "dry_run": True,
    "checks": checks,
    "sidecar_min_effect": sidecar_min_effect,
    "params_min_effect": params_min_effect,
    "fixture_set": args.fixture_set,
  }
  lac.emit(result, args.out)
  if not ok:
    failed = [name for name, passed in checks.items() if not passed]
    print(f"layer_a_exact_validate --dry-run: failed checks: {failed!r}", file=sys.stderr)
    return 1
  logger.info("layer_a_exact_validate --dry-run: all checks OK")
  return 0


def run_measurement(
  params: dict[str, Any], args: argparse.Namespace, *, smoke: bool
) -> tuple[dict[str, Any], int]:
  """Shared body for `--smoke` and the full run: measure, `prereq_check`, provenance."""
  prov = lac.provenance(PARAMS_PATH)
  prereq = lac.prereq_check("layer_a_exact", "exact", params)
  manifest = _load_manifest()
  fixtures_for_set = _fixtures_for_set(manifest, args.fixture_set)
  manifest_sha256 = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()

  exact = params.get("exact", {})
  magnitude = exact.get("controls", {}).get("w_out_bias_perturb_magnitude")
  reduced_subset_names = exact.get("reduced_subset", [])
  sentinel_off = None
  if magnitude is not None and reduced_subset_names:
    reduced_subset = _reduced_subset_fixtures(manifest, reduced_subset_names)
    sentinel_off = lae.sentinel_ratio_to_bar(
      reduced_subset, weight_source="eqx", perturb=False, magnitude=magnitude
    )

  if smoke:
    protein_fixtures = _protein_fixtures(fixtures_for_set)
    fixture = _pick_smoke_fixture(protein_fixtures)
    data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
    full_model_bundle = lac.load_full_model("eqx")
    batch = lae.build_exact_batch(fixture, data_utils_module)
    rows = lae.row_p05(batch, full_model_bundle, data_utils_module, "eqx")
    engine_summary: dict[str, Any] = {
      "rows": rows,
      "n_comparisons": len(rows),
      "n_over_bar": sum(1 for r in rows if r.get("ratio") is not None and r["ratio"] > 1.0),
      "n_not_advanced": 0,
      "n_near_tie_excluded": 0,
      "n_skipped": 0,
      "skip_reasons": [],
      "controls": [],
      "controls_total": 0,
      "controls_detected": 0,
      "not_implemented": [],
    }
  else:
    engine_summary = lae.run_rows(fixtures_for_set, params, "eqx", controls=True)

  worst_path, worst_ratio = _worst_row(engine_summary["rows"])
  result = {
    "rows": engine_summary["rows"],
    "controls": engine_summary["controls"],
    "n_comparisons": engine_summary["n_comparisons"],
    "n_over_bar": engine_summary["n_over_bar"],
    "n_not_advanced_over_bar": engine_summary.get("n_not_advanced_over_bar", 0),
    "n_not_advanced": engine_summary["n_not_advanced"],
    "n_near_tie_excluded": engine_summary["n_near_tie_excluded"],
    "n_skipped": engine_summary["n_skipped"],
    "skip_reasons": engine_summary["skip_reasons"],
    "not_implemented": engine_summary["not_implemented"],
    "prereq_ok": prereq["prereq_ok"],
    "prereq_reason": prereq["reason"],
    "controls_total": engine_summary["controls_total"],
    "controls_detected": engine_summary["controls_detected"],
    "worst_path": worst_path,
    "worst_ratio_to_bar": worst_ratio,
    "sentinel_ratio_to_bar": sentinel_off,
    "reference_commit": _reference_commit(),
    "weight_source": "eqx",
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "smoke": smoke,
    **prov,
  }

  exit_code = 0
  if not prereq["prereq_ok"]:
    logger.error("prereq_check failed: %s", prereq["reason"])
  if engine_summary["n_skipped"] > 0:
    exit_code = EXIT_SKIPPED
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dry-run", action="store_true", help="Check imports/manifest/sidecar only.")
  parser.add_argument("--smoke", action="store_true", help="One fixture, P05 only, < 60 s.")
  parser.add_argument(
    "--fixture-set",
    choices=("A", "B"),
    default="B",
    help="Manifest fixture set to validate on (held-out set B by default).",
  )
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  args = parser.parse_args(argv)

  ok, reason = _params_committed_and_clean()
  if not ok:
    print(
      f"layer_a_exact_validate: anti-HARKing gate failed: {reason} -- "
      f"refusing to start (spec 'Bathos staging': validation requires "
      f"{_relative_to_worktree(PARAMS_PATH) if PARAMS_PATH.is_file() else PARAMS_PATH} "
      "tracked and clean)",
      file=sys.stderr,
    )
    return EXIT_ANTI_HARKING

  with PARAMS_PATH.open() as fh:
    params = json.load(fh)

  mode = lac.differential_mode()
  if mode["active"]:
    return run_differential_arm(mode, params, args)

  if args.dry_run:
    return run_dry_run(params, args)

  result, exit_code = run_measurement(params, args, smoke=args.smoke)
  lac.emit(result, args.out)
  logger.info(
    "layer_a_exact_validate: fixture_set=%s smoke=%s n_comparisons=%s prereq_ok=%s exit_code=%d",
    args.fixture_set,
    args.smoke,
    result.get("n_comparisons"),
    result.get("prereq_ok"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
