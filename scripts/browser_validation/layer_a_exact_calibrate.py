"""Layer (a) exact-tier calibration (T7b, step 3, fixture set A).

Bathos staging's "calibration" twin for `layer_a_exact.py`'s engine: this script owns
the ONE place that decides, ex ante, everything `layer_a_exact_validate.py` (set B,
held out) is only allowed to READ back:

- **Sized-control magnitude search.** `layer_a_exact._control_w_out_bias_perturb`
  (reused, never reimplemented) is tried at each of `MAGNITUDE_CANDIDATES` until one
  lands its effect in `SIZING_RATIO_RANGE` (`[2x, 10x]` the P05 bar, spec
  "Positive-control sizing"); the winning magnitude is committed to
  `exact.controls.w_out_bias_perturb_magnitude`.
- **Headroom rule.** Every row from a full `layer_a_exact.run_rows(..., params=None,
  weight_source="eqx")` pass over fixture set A whose OWN measurement exceeds
  `bar / 2` is recorded in `exact.not_advanced_paths` (spec "Pre-registered layer-(a)
  bars": "Headroom rule: if a path's calibration measurement exceeds bar/2 it is
  marked not_advanced"). This is a RESULT-FIELD verdict written into the params file
  for validate to apply via `layer_a_exact._apply_headroom` -- calibration never
  tightens a row's *bar*, only its *advancement*.
- **Differential sensitivity, pre-registered reduced subset.** A small, fixed subset
  of set-A fixtures (`REDUCED_SUBSET_SIZE` of them, smallest-`L` first, `2GFB`
  excluded by construction since it is never sized this small) is measured with
  `layer_a_exact.sentinel_ratio_to_bar` off vs on (the just-sized magnitude);
  `exact.measured_half_effect` is half the absolute on-off difference and must be
  `>= 1.0` (asserted here, failing the run otherwise) -- `exact.min_effect` itself is
  PINNED to `1.0` exactly (R2-C6: not derived from the measurement), matching the
  validate sidecar's hard-coded `min_effect = 1.0`.
- **Budget.** `exact.budget_wall_hours` (this run's own measured wall time) and
  `exact.projected_peak_rss_gib` (`resource.getrusage(RUSAGE_SELF).ru_maxrss`,
  Linux KiB) are recorded; a FULL (non-`--smoke`) run fails (after still writing a
  diagnostic result to `--out`/`$BTH_RESULTS_PATH`, but WITHOUT writing
  `--params-out`) if either exceeds the caps (16 h / 48 GiB) -- the fix is fewer
  set-B fixtures at validation, never raising the cap (common context / spec
  "Titanix budget").
- **AC-15 cross-reference (`pt_convert` diagnostic).** A second, `controls=False`
  `run_rows` pass at `weight_source="pt_convert"` is recorded under
  `exact.pt_convert_diagnostic` for comparison against the `eqx` numbers that
  actually gate headroom/advancement -- diagnostic only, never gating (spec line
  "pt_convert rows reported as diagnostic"). Failures here (e.g. a prerequisite
  genuinely missing) are caught and recorded as `{"error": ...}`, never allowed to
  abort the `eqx` measurement the rest of this script depends on.

**`--smoke`** runs `layer_a_exact.row_p05` on ONE small fixture from `--fixture-set`
(named fixture `SMOKE_FIXTURE_NAME` if present, else the smallest-`L` protein
fixture in that set) -- never the full engine, never `2GFB` (`L=3464`, set A) --
so it finishes in well under a minute. It REQUIRES a non-default `--params-out`: the
smoke run's params write is scratch-only and MUST NOT be mistaken for (or read by)
`layer_a_exact_validate.py`, which only ever reads the default path.

Writes results to `$BTH_RESULTS_PATH` AND `--out` via `layer_a_common.emit` (a
calibration run is never itself a differential-phase re-execution, but `emit` is
reused unconditionally for consistency with the validate script and the common
context's "Results go to `$BTH_RESULTS_PATH` AND `--out`" rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import sys
import time
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
DEFAULT_PARAMS_OUT = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "preregistered_params.json"
)
_MANIFEST_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"

#: Spec "T7 step 3": "searches each sized control over {1e-4, 2e-4, 5e-4, 1e-3, 2e-3}".
MAGNITUDE_CANDIDATES: tuple[float, ...] = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3)
#: Spec "Positive-control sizing": "must move its metric into [2x, 10x] the bar".
SIZING_RATIO_RANGE: tuple[float, float] = (2.0, 10.0)
#: Spec "Titanix budget": "fails if budget_wall_hours > 16 or projected_peak_rss_gib > 48".
BUDGET_WALL_HOURS_CAP = 16.0
PROJECTED_PEAK_RSS_GIB_CAP = 48.0
#: R2-C6: min_effect is PINNED at 1.0, never derived.
PINNED_MIN_EFFECT = 1.0
#: Spec T7 step 4/step 3: "min_effect ... asserts half the measured on-off effect ... >= 1.0".
MIN_MEASURED_HALF_EFFECT = 1.0

SMOKE_FIXTURE_NAME = "5L33"
REDUCED_SUBSET_SIZE = 2

EXIT_SKIPPED = 4


def _load_manifest() -> dict[str, Any]:
  with _MANIFEST_PATH.open() as fh:
    return json.load(fh)


def _fixtures_for_set(manifest: dict[str, Any], fixture_set: str) -> list[dict[str, Any]]:
  return [f for f in manifest["fixtures"] if f.get("set") == fixture_set]


def _protein_fixtures(fixtures_for_set: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [f for f in fixtures_for_set if f.get("kind") == "protein"]


def _peak_rss_gib() -> float:
  """Peak RSS of this process (`ru_maxrss` is KiB on Linux, the only platform in scope)."""
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def _pick_smoke_fixture(protein_fixtures: list[dict[str, Any]]) -> dict[str, Any]:
  for fixture in protein_fixtures:
    if fixture["name"] == SMOKE_FIXTURE_NAME:
      return fixture
  if not protein_fixtures:
    msg = "no protein fixtures in the requested --fixture-set to run a P05 smoke row on"
    raise SystemExit(msg)
  return min(protein_fixtures, key=lambda f: f.get("L", sys.maxsize))


def _pick_reduced_subset(protein_fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Pre-registered reduced subset for the differential sentinel: smallest-`L` first.

  Deliberately excludes the largest fixtures (`2GFB`, `L=3464`, is never small
  enough to be picked here) -- the reduced subset exists so `sentinel_ratio_to_bar`
  is cheap enough to run twice (off/on) inside a normal validate invocation, not to
  exercise the harness's largest structures.
  """
  ordered = sorted(protein_fixtures, key=lambda f: f.get("L", sys.maxsize))
  subset = ordered[:REDUCED_SUBSET_SIZE]
  if not subset:
    msg = "no protein fixtures in the requested --fixture-set to build a reduced subset from"
    raise SystemExit(msg)
  return subset


def _search_sized_magnitude(
  batch: lae.ExactBatch, full_model_bundle: tuple[Any, Any, Any, Any]
) -> tuple[float, dict[str, Any]]:
  """Return `(magnitude, control)` for the first candidate landing in `SIZING_RATIO_RANGE`."""
  lo, hi = SIZING_RATIO_RANGE
  tried: list[dict[str, Any]] = []
  for magnitude in MAGNITUDE_CANDIDATES:
    control = lae._control_w_out_bias_perturb(batch, full_model_bundle, magnitude)  # noqa: SLF001
    tried.append({"magnitude": magnitude, "ratio_to_bar": control["ratio_to_bar"]})
    if lo <= control["ratio_to_bar"] <= hi:
      return magnitude, control
  msg = (
    f"no magnitude in {MAGNITUDE_CANDIDATES!r} sized the w_out_bias_perturb control into "
    f"[{lo}x, {hi}x] the bar (tried: {tried!r})"
  )
  raise SystemExit(msg)


def _compute_not_advanced_paths(rows: list[dict[str, Any]]) -> list[str]:
  """Headroom rule: `value > bar / 2` -> `not_advanced` (bars-table "Headroom rule")."""
  paths: set[str] = set()
  for row in rows:
    if row.get("status") == "not_implemented":
      continue
    bar = row.get("bar")
    value = row.get("value")
    if bar is None or value is None:
      continue
    if float(value) > float(bar) / 2.0:
      paths.add(row["path"])
  return sorted(paths)


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


def _pt_convert_diagnostic(fixtures_for_set: list[dict[str, Any]]) -> dict[str, Any]:
  """AC-15 cross-reference: `pt_convert` numbers, diagnostic only, never gating."""
  try:
    pt_result = lae.run_rows(fixtures_for_set, None, "pt_convert", controls=False)
  except Exception as exc:  # noqa: BLE001 -- diagnostic-only side channel, must never abort eqx
    logger.warning("pt_convert diagnostic run failed (recorded, not gating): %s", exc)
    return {"error": str(exc)}
  worst_path, worst_ratio = _worst_row(pt_result["rows"])
  return {
    "n_comparisons": pt_result["n_comparisons"],
    "n_over_bar": pt_result["n_over_bar"],
    "n_skipped": pt_result["n_skipped"],
    "worst_path": worst_path,
    "worst_ratio_to_bar": worst_ratio,
  }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures = _protein_fixtures(_fixtures_for_set(manifest, args.fixture_set))
  fixture = _pick_smoke_fixture(protein_fixtures)

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  full_model_bundle = lac.load_full_model("eqx")
  batch = lae.build_exact_batch(fixture, data_utils_module)
  rows = lae.row_p05(batch, full_model_bundle, data_utils_module, "eqx")
  elapsed_seconds = time.monotonic() - start

  exact_section = {
    "git_hash": prov["git_hash"],
    "min_effect": PINNED_MIN_EFFECT,
    "measured_half_effect": None,
    "subset_size": 1,
    "reduced_subset": [fixture["name"]],
    "budget_wall_hours": max(elapsed_seconds / 3600.0, sys.float_info.min),
    "projected_peak_rss_gib": _peak_rss_gib(),
    "not_advanced_paths": [],
    "controls": {"w_out_bias_perturb_magnitude": lae.DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE},
    "fixture_set": args.fixture_set,
    "smoke": True,
  }
  params = _merge_params(args.params_out, exact_section)
  args.params_out.parent.mkdir(parents=True, exist_ok=True)
  with args.params_out.open("w") as fh:
    json.dump(params, fh, indent=2, default=str)
  params_section_sha256 = lac.section_sha(params, "exact")

  worst_path, worst_ratio = _worst_row(rows)
  return {
    "rows": rows,
    "n_comparisons": len(rows),
    "n_over_bar": sum(1 for r in rows if r.get("ratio") is not None and r["ratio"] > 1.0),
    "n_not_advanced": 0,
    "n_skipped": 0,
    "worst_path": worst_path,
    "worst_ratio_to_bar": worst_ratio,
    "controls_sized": 0,
    "controls_total": 0,
    "params_written": True,
    "params_section_sha256": params_section_sha256,
    "fixture_set": args.fixture_set,
    "elapsed_seconds": elapsed_seconds,
    "smoke": True,
    **prov,
  }


def _merge_params(params_out: Path, exact_section: dict[str, Any]) -> dict[str, Any]:
  """Merge `exact_section` into any already-committed sections (e.g. T8's `sampling`)."""
  params: dict[str, Any] = {}
  if params_out.is_file():
    with params_out.open() as fh:
      params = json.load(fh)
  params["exact"] = exact_section
  return params


def run_full(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  """Full (non-`--smoke`) calibration over `--fixture-set`. Returns `(result, exit_code)`."""
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  fixtures_for_set = _fixtures_for_set(manifest, args.fixture_set)
  protein_fixtures = _protein_fixtures(fixtures_for_set)
  manifest_sha256 = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()

  eqx_result = lae.run_rows(fixtures_for_set, None, "eqx", controls=True)
  not_advanced_paths = _compute_not_advanced_paths(eqx_result["rows"])
  worst_path, worst_ratio = _worst_row(eqx_result["rows"])

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  full_model_bundle = lac.load_full_model("eqx")
  reduced_subset = _pick_reduced_subset(protein_fixtures)
  sizing_batch = lae.build_exact_batch(reduced_subset[0], data_utils_module)
  magnitude, _sizing_control = _search_sized_magnitude(sizing_batch, full_model_bundle)

  off_ratio = lae.sentinel_ratio_to_bar(
    reduced_subset, weight_source="eqx", perturb=False, magnitude=magnitude
  )
  on_ratio = lae.sentinel_ratio_to_bar(
    reduced_subset, weight_source="eqx", perturb=True, magnitude=magnitude
  )
  measured_half_effect = abs(on_ratio - off_ratio) / 2.0
  if measured_half_effect < MIN_MEASURED_HALF_EFFECT:
    msg = (
      f"measured_half_effect={measured_half_effect!r} < {MIN_MEASURED_HALF_EFFECT} "
      f"(off={off_ratio!r}, on={on_ratio!r}, magnitude={magnitude!r}) -- the sentinel is not "
      "sensitive enough on the reduced subset; widen the subset or re-search the magnitude"
    )
    raise SystemExit(msg)

  pt_convert_diagnostic = _pt_convert_diagnostic(fixtures_for_set)

  elapsed_seconds = time.monotonic() - start
  budget_wall_hours = elapsed_seconds / 3600.0
  projected_peak_rss_gib = _peak_rss_gib()
  within_budget = (
    budget_wall_hours <= BUDGET_WALL_HOURS_CAP
    and projected_peak_rss_gib <= PROJECTED_PEAK_RSS_GIB_CAP
  )

  exact_section = {
    "git_hash": prov["git_hash"],
    "min_effect": PINNED_MIN_EFFECT,
    "measured_half_effect": measured_half_effect,
    "subset_size": len(reduced_subset),
    "reduced_subset": [f["name"] for f in reduced_subset],
    "budget_wall_hours": budget_wall_hours,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "not_advanced_paths": not_advanced_paths,
    "controls": {"w_out_bias_perturb_magnitude": magnitude},
    "pt_convert_diagnostic": pt_convert_diagnostic,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
  }

  params_written = False
  params_section_sha256 = lac.section_sha({"exact": exact_section}, "exact")
  if within_budget:
    params = _merge_params(args.params_out, exact_section)
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    with args.params_out.open("w") as fh:
      json.dump(params, fh, indent=2, default=str)
    params_written = True
  else:
    logger.error(
      "budget exceeded (budget_wall_hours=%.3f cap=%.1f, projected_peak_rss_gib=%.3f cap=%.1f) "
      "-- NOT writing --params-out; reduce set-B fixtures, never raise the cap",
      budget_wall_hours,
      BUDGET_WALL_HOURS_CAP,
      projected_peak_rss_gib,
      PROJECTED_PEAK_RSS_GIB_CAP,
    )

  result = {
    "rows": eqx_result["rows"],
    "controls": eqx_result["controls"],
    "n_comparisons": eqx_result["n_comparisons"],
    "n_over_bar": eqx_result["n_over_bar"],
    "n_not_advanced": len(not_advanced_paths),
    "n_near_tie_excluded": eqx_result["n_near_tie_excluded"],
    "n_skipped": eqx_result["n_skipped"],
    "skip_reasons": eqx_result["skip_reasons"],
    "not_implemented": eqx_result["not_implemented"],
    "worst_path": worst_path,
    "worst_ratio_to_bar": worst_ratio,
    "controls_total": eqx_result["controls_total"],
    "controls_sized": eqx_result["controls_detected"],
    "params_written": params_written,
    "params_section_sha256": params_section_sha256,
    "sentinel_ratio_to_bar_off": off_ratio,
    "sentinel_ratio_to_bar_on": on_ratio,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "weight_source": "eqx",
    "budget_wall_hours": budget_wall_hours,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "elapsed_seconds": elapsed_seconds,
    **prov,
  }
  exit_code = 0
  if not within_budget:
    exit_code = 1
  elif eqx_result["n_skipped"] > 0 or not prov["git_clean"]:
    exit_code = EXIT_SKIPPED
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--smoke", action="store_true", help="One fixture, P05 only, < 60 s.")
  parser.add_argument(
    "--fixture-set", choices=("A", "B"), default="A", help="Manifest fixture set to calibrate on."
  )
  parser.add_argument(
    "--params-out",
    type=Path,
    default=DEFAULT_PARAMS_OUT,
    help="Where to write preregistered_params.json's exact section.",
  )
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  args = parser.parse_args(argv)

  if args.smoke and args.params_out == DEFAULT_PARAMS_OUT:
    parser.error(
      "--smoke requires a non-default --params-out (a scratch file "
      "layer_a_exact_validate.py never reads)"
    )

  if args.smoke:
    result = run_smoke(args)
    exit_code = EXIT_SKIPPED if result.get("n_skipped", 0) > 0 else 0
  else:
    result, exit_code = run_full(args)

  lac.emit(result, args.out)
  logger.info(
    "layer_a_exact_calibrate: fixture_set=%s smoke=%s n_comparisons=%s params_written=%s "
    "exit_code=%d",
    args.fixture_set,
    args.smoke,
    result.get("n_comparisons"),
    result.get("params_written"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
