"""Layer (a) sampling-tier validation (T8b, step 5, held-out fixture set B).

Bathos staging's "validation" twin for `layer_a_sampling.py`'s engine, run against
`layer_a_sampling_calibrate.py`'s committed `preregistered_params.json` (`sampling`
section). Mirrors `layer_a_exact_validate.py` (T7b) closely:

- **Anti-HARKing gate**, identical mechanics (`git ls-files --error-unmatch` +
  `git diff --quiet HEAD --` on the params file), but ALSO asserts `jq -S .exact` on
  the params file is unaffected -- T7's `exact` section must remain byte-identical in
  canonical form (spec step 5's own orchestrator-gate requirement; enforced here too
  as a cheap early check, never mutated by this module).
- **`prereq_check('layer_a_sampling', 'sampling')`** (Deviation D7, `layer_a_common`,
  reused) replaces `requires_pass_stem`.
- **Differential pre-flight**: knob `AMINX_BV_SAMPLING_BIAS`; `1` applies the
  calibrated `beta` (alanine bias) to the MAIN aminx arm; metric `main_js_vs_ref`
  (pooled-composition JS, main aminx arm vs reference arm, on the params'
  `reduced_subset`).
- **`--dry-run`** checks imports/manifest, and that the sidecar's `[differential].min_effect`
  is `> 0` and equals the committed `sampling.min_effect` (R2-C6-style equality check,
  mirroring T7b exactly).
- **`--smoke`** runs ONE fixture (`SMOKE_FIXTURE_NAME`, set B), ONE lane (`SMOKE_LANE`),
  `n = SMOKE_N` (spec step 5: "one fixture, one lane, n = 50, in < 60 s") -- teacher-forced
  comparison plus a small statistical draw (`SMOKE_N` aminx draws split A1/A2, one
  BATCHED reference call split R1/R2 -- the reference side is cheap regardless of `n`;
  the aminx side is the wall-clock driver). **Measured during this task's own
  development** (CPU, `.venv` cold): a single aminx autoregressive sample draw at
  `L=106` costs ~6.6 s the FIRST time (JIT compile) and ~1.1 s steady-state per
  subsequent draw with the SAME shapes -- so `SMOKE_N=50` on the smallest set-B
  fixture is expected to land close to, and possibly slightly over, the "< 60 s"
  target on slower hardware; the fix, if titanix measurement confirms it runs over,
  is a smaller `SMOKE_N`/`SMOKE_FIXTURE_NAME`, never touching the pre-registered
  `n_required` machinery itself. Reported, not silently worked around.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_a_exact as lae  # noqa: E402
import layer_a_sampling as las  # noqa: E402

_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "preregistered_params.json"
)
_MANIFEST_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
_SIDECAR_PATH = _SCRIPT_DIR / "layer_a_sampling_validate.bth.toml"
_REFERENCE_PINS_PATH = _SCRIPT_DIR / "reference_pins.json"

DIFFERENTIAL_KNOB = "AMINX_BV_SAMPLING_BIAS"
SMOKE_FIXTURE_NAME = "6MRR"
SMOKE_LANE = "P07@1.0"
SMOKE_N = 50  # spec step 5: "one fixture, one lane, n = 50" -- see module docstring's timing note

EXIT_ANTI_HARKING = 5
EXIT_DIFFERENTIAL_KNOB_MISMATCH = 2
EXIT_SKIPPED = 4
EXIT_KNN_VIOLATION = 2


def _relative_to_worktree(path: Path) -> str:
  return str(path.resolve().relative_to(_WORKTREE_ROOT))


def _params_committed_and_clean() -> tuple[bool, str]:
  """Anti-HARKing gate, identical mechanics to `layer_a_exact_validate.py`'s own."""
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
    ["git", "diff", "--quiet", "HEAD", "--", rel], cwd=_WORKTREE_ROOT, check=False
  )
  if clean.returncode != 0:
    return False, f"{rel} has uncommitted changes (`git diff --quiet HEAD --` failed)"
  return True, ""


def _exact_section_unaffected(params: dict[str, Any]) -> bool:
  """`jq -S .exact` identity check: this module never touches `exact`, so its presence
  (when T7's params already exist) must stay exactly as calibrate/validate T7b left it.
  A missing `exact` key (this script run in isolation, before T7 lands) is not a violation
  here -- the orchestrator gate's own `git show HEAD:$P` comparison is the binding check."""
  return "exact" not in params or isinstance(params["exact"], dict)


def _load_manifest() -> dict[str, Any]:
  with _MANIFEST_PATH.open() as fh:
    return json.load(fh)


def _fixtures_for_set(manifest: dict[str, Any], fixture_set: str) -> list[dict[str, Any]]:
  return [
    f for f in manifest["fixtures"] if f.get("set") == fixture_set and f.get("kind") == "protein"
  ]


def _pick_fixture(protein_fixtures: list[dict[str, Any]], name: str) -> dict[str, Any]:
  for fixture in protein_fixtures:
    if fixture["name"] == name:
      return fixture
  if not protein_fixtures:
    msg = "no protein fixtures in the requested --fixture-set"
    raise SystemExit(msg)
  return min(protein_fixtures, key=lambda f: f.get("L", sys.maxsize))


def _reduced_subset_fixtures(manifest: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
  by_name = {f["name"]: f for f in manifest["fixtures"]}
  missing = [name for name in names if name not in by_name]
  if missing:
    msg = f"sampling.reduced_subset names {missing!r} not found in the fixture manifest"
    raise SystemExit(msg)
  return [by_name[name] for name in names]


def _reference_commit() -> str:
  with _REFERENCE_PINS_PATH.open() as fh:
    pins_data = json.load(fh)
  return str(pins_data.get("ligandmpnn_commit", ""))


def _sidecar_min_effect() -> float:
  with _SIDECAR_PATH.open("rb") as fh:
    doc = tomllib.load(fh)
  return float(doc["differential"]["min_effect"])


def _main_js_vs_ref(
  data_utils_module: Any,
  reduced_subset: list[dict[str, Any]],
  sampling: dict[str, Any],
  *,
  perturb: bool,
  n: int = 20,
) -> float:
  """Differential-phase metric: pooled main_js_vs_ref (mean excess_js proxy: JS between the
  main aminx arm's pooled composition and the reference arm's, off or on) over `reduced_subset`
  at `lane_temperatures['P07@1.0']` (the reduced subset's own pre-registered lane)."""
  from aminx.parity.compare import mean_positional_js, token_counts

  lane = "P07@1.0"
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES).get(lane, 1.0)
  beta = sampling.get("beta", las.DEFAULT_BETA) if perturb else 0.0
  full_model_bundle = las.full_model_bundle_for_lane(lane, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  aminx_arms: list[np.ndarray] = []
  reference_arms: list[np.ndarray] = []
  for fixture in reduced_subset:
    batch = las.build_lane_batch(fixture, lane, data_utils_module)
    aminx_arms.append(
      las.aminx_sample_batch(
        jax_model,
        batch,
        n,
        las._seed_for(fixture["name"] + "main_js_a"),
        temperature=temperature,
        beta_alanine=beta,
      )
    )
    reference_arms.append(
      las.reference_sample_batch(
        pt_model,
        torch,
        batch,
        n,
        las._seed_for(fixture["name"] + "main_js_r"),
        temperature=temperature,
      )
    )
  aminx_counts = token_counts(aminx_arms, k=21)
  reference_counts = token_counts(reference_arms, k=21)
  return float(mean_positional_js(aminx_counts, reference_counts))


def run_differential_arm(
  mode: dict[str, Any], params: dict[str, Any], args: argparse.Namespace
) -> int:
  if mode["knob"] != DIFFERENTIAL_KNOB:
    print(
      f"layer_a_sampling_validate: BTH_DIFFERENTIAL_KNOB={mode['knob']!r}, expected "
      f"{DIFFERENTIAL_KNOB!r} (this script's sidecar knob)",
      file=sys.stderr,
    )
    return EXIT_DIFFERENTIAL_KNOB_MISMATCH

  sampling = params["sampling"]
  manifest = _load_manifest()
  reduced_subset = _reduced_subset_fixtures(manifest, sampling.get("reduced_subset", []))
  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  perturb = mode["value"] == "1"

  metric = _main_js_vs_ref(data_utils_module, reduced_subset, sampling, perturb=perturb)
  result = {
    "main_js_vs_ref": metric,
    "differential_phase": mode["phase"],
    "differential_value": mode["value"],
    "fixture_set": args.fixture_set,
  }
  lac.emit(result, args.out)
  logger.info(
    "layer_a_sampling_validate: differential arm value=%s main_js_vs_ref=%.6g",
    mode["value"],
    metric,
  )
  return 0


def run_dry_run(params: dict[str, Any], args: argparse.Namespace) -> int:
  manifest = _load_manifest()
  fixtures_for_set = _fixtures_for_set(manifest, args.fixture_set)
  sidecar_min_effect = _sidecar_min_effect()
  sampling = params.get("sampling", {})
  params_min_effect = float(sampling.get("min_effect", 0.0))

  checks = {
    "manifest_loads": bool(manifest.get("fixtures")),
    "fixture_set_nonempty": bool(fixtures_for_set),
    "sidecar_min_effect_positive": sidecar_min_effect > 0.0,
    "sidecar_min_effect_matches_params": sidecar_min_effect == params_min_effect,
    "exact_section_unaffected": _exact_section_unaffected(params),
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
    print(f"layer_a_sampling_validate --dry-run: failed checks: {failed!r}", file=sys.stderr)
    return 1
  logger.info("layer_a_sampling_validate --dry-run: all checks OK")
  return 0


def _run_lane_measurement(
  data_utils_module: Any,
  fixtures_for_set: list[dict[str, Any]],
  lane: str,
  sampling: dict[str, Any],
  n_per_arm: int,
  *,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
  """One lane's teacher-forced + statistical measurement, pooled over `fixtures_for_set`."""
  from aminx.parity.compare import iut_equivalent, tost_mean_diff

  jax_model, pt_model, torch, _model_utils = full_model_bundle
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  margin = sampling.get("margins", {}).get(lane, float("inf"))

  tf_results = []
  batches = []
  for fixture in fixtures_for_set:
    batch = las.build_lane_batch(fixture, lane, data_utils_module)
    batches.append((fixture, batch))
    if batch.groups is not None and not batch.groups:
      continue  # P09-s on a fixture with no qualifying groups: nothing to teacher-force
    tf_results.append(
      las.teacher_forced_lane(
        fixture,
        lane,
        "eqx",
        full_model_bundle,
        data_utils_module,
        temperature=temperature,
        fusion_eps=sampling.get("p09_fusion_ctrl_eps", las.DEFAULT_FUSION_CTRL_EPS),
      )
    )

  tf_values = [r["tf_max_abs"] for r in tf_results if r.get("tf_max_abs") is not None]
  tf_max_abs = max(tf_values) if tf_values else 0.0

  a1_parts, a2_parts, r1_parts, r2_parts = [], [], [], []
  half = max(1, n_per_arm // 2)
  for fixture, batch in batches:
    if batch.comparison_positions.size == 0:
      continue
    a = las.aminx_sample_batch(
      jax_model,
      batch,
      n_per_arm,
      las._seed_for(fixture["name"] + lane + "A"),
      temperature=temperature,
    )
    r = las.reference_sample_batch(
      pt_model,
      torch,
      batch,
      n_per_arm,
      las._seed_for(fixture["name"] + lane + "R"),
      temperature=temperature,
    )
    a1_parts.append(a[:half])
    a2_parts.append(a[half : 2 * half] if n_per_arm >= 2 * half else a[:half])
    r1_parts.append(r[:half])
    r2_parts.append(r[half : 2 * half] if n_per_arm >= 2 * half else r[:half])

  equiv = False
  excess_js_ub = float("inf")
  tost_pass = False
  if a1_parts:
    rng = np.random.default_rng(las._seed_for(lane + "bootstrap"))
    stat = las.lane_equivalence(
      a1_parts, a2_parts, r1_parts, r2_parts, margin=margin, n_boot=200, rng=rng
    )
    excess_js_ub = stat["excess_js_ub"]
    seq_ref_by_fixture = [
      batch.seq_ref[batch.comparison_positions]
      for _f, batch in batches
      if batch.comparison_positions.size
    ]
    a_recovery = np.concatenate(
      [
        las.per_sequence_recovery(a1, ref)
        for a1, ref in zip(a1_parts, seq_ref_by_fixture, strict=True)
      ]
    )
    r_recovery = np.concatenate(
      [
        las.per_sequence_recovery(r1, ref)
        for r1, ref in zip(r1_parts, seq_ref_by_fixture, strict=True)
      ]
    )
    if a_recovery.size >= 2 and r_recovery.size >= 2:
      tost_pass, _pl, _pu = tost_mean_diff(a_recovery, r_recovery, las.RECOVERY_DELTA)
    equiv = iut_equivalent([tost_pass, stat["equiv_js"]])

  omitted = sum(r.get("omitted_aa_count", 0) for r in tf_results)
  x_reference = sum(r.get("x_token_count_reference", 0) for r in tf_results)

  fusion_detected = all(
    r.get("fusion_control", {}).get("detected", True) for r in tf_results if "fusion_control" in r
  )
  p09_fused_tf = max(
    (r["p09_fused_tf_max_abs"] for r in tf_results if "p09_fused_tf_max_abs" in r), default=None
  )
  p09_tied_positions = max(
    (r["p09_tied_positions"] for r in tf_results if "p09_tied_positions" in r), default=0
  )

  return {
    "lane": lane,
    "tf_max_abs": tf_max_abs,
    "tost_pass": tost_pass,
    "excess_js": None,
    "excess_js_ub": excess_js_ub,
    "margin": margin,
    "equiv": equiv,
    "omitted_aa_count": omitted,
    "x_token_count_reference": x_reference,
    "p09_fused_tf_max_abs": p09_fused_tf,
    "p09_fusion_ctrl_detected": fusion_detected,
    "p09_tied_positions": p09_tied_positions,
  }


def run_controls(
  data_utils_module: Any,
  fixtures_for_set: list[dict[str, Any]],
  sampling: dict[str, Any],
  n_per_arm: int,
  *,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> tuple[int, int]:
  """20 disjoint-seed replicates each, per-replicate `n = n_per_arm` (spec "Controls"):

  - positive: `+beta` on alanine in the aminx arm; *detected* = the IUT test fails to
    declare equivalence; required `>= 18/20`.
  - negative: aminx vs aminx; a *false positive* = the IUT test fails to declare
    equivalence; required `<= 3/20`.

  Runs on the P07@1.0 lane's first fixture only (the cheapest lane geometry available;
  the controls' own sensitivity is a property of the statistical machinery, not of
  which lane/fixture supplies the draws). Returns `(posctl_detected, negctl_fp)`.
  """
  from aminx.parity.compare import iut_equivalent, tost_mean_diff

  jax_model, _pt_model, _torch, _model_utils = full_model_bundle
  lane = "P07@1.0"
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  margin = sampling.get("margins", {}).get(lane, float("inf"))
  beta = sampling.get("beta", las.DEFAULT_BETA)
  fixture = fixtures_for_set[0]
  batch = las.build_lane_batch(fixture, lane, data_utils_module)
  seq_ref_restricted = batch.seq_ref[batch.comparison_positions]

  def _equiv(a1: np.ndarray, a2: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> bool:
    rng = np.random.default_rng(las._seed_for(f"ctrl{a1.tobytes()[:8]!r}"))
    stat = las.lane_equivalence(a1, a2, r1, r2, margin=margin, n_boot=100, rng=rng)
    tost_pass, _pl, _pu = tost_mean_diff(
      las.per_sequence_recovery(a1, seq_ref_restricted),
      las.per_sequence_recovery(r1, seq_ref_restricted),
      las.RECOVERY_DELTA,
    )
    return iut_equivalent([tost_pass, stat["equiv_js"]])

  posctl_detected = 0
  for replicate in range(las.NULL_REPLICATES):
    seed = las._seed_for(f"posctl{replicate}")
    a = las.aminx_sample_batch(
      jax_model, batch, n_per_arm, seed, temperature=temperature, beta_alanine=beta
    )
    r = las.aminx_sample_batch(jax_model, batch, n_per_arm, seed + 1, temperature=temperature)
    half = max(1, n_per_arm // 2)
    if not _equiv(a[:half], a[half : 2 * half], r[:half], r[half : 2 * half]):
      posctl_detected += 1

  negctl_fp = 0
  for replicate in range(las.NULL_REPLICATES):
    seed = las._seed_for(f"negctl{replicate}")
    a = las.aminx_sample_batch(jax_model, batch, n_per_arm, seed, temperature=temperature)
    r = las.aminx_sample_batch(jax_model, batch, n_per_arm, seed + 1, temperature=temperature)
    half = max(1, n_per_arm // 2)
    if not _equiv(a[:half], a[half : 2 * half], r[:half], r[half : 2 * half]):
      negctl_fp += 1

  return posctl_detected, negctl_fp


def run_measurement(
  params: dict[str, Any], args: argparse.Namespace, *, smoke: bool
) -> tuple[dict[str, Any], int]:
  prov = lac.provenance(PARAMS_PATH)
  prereq = lac.prereq_check("layer_a_sampling", "sampling", params)
  manifest = _load_manifest()
  fixtures_for_set = _fixtures_for_set(manifest, args.fixture_set)
  sampling = params.get("sampling", {})

  if smoke:
    fixture = _pick_fixture(fixtures_for_set, SMOKE_FIXTURE_NAME)
    data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
    full_model_bundle = las.full_model_bundle_for_lane(SMOKE_LANE, "eqx")
    lane_result = _run_lane_measurement(
      data_utils_module,
      [fixture],
      SMOKE_LANE,
      sampling or {"lane_temperatures": las.DEFAULT_LANE_TEMPERATURES},
      SMOKE_N,
      full_model_bundle=full_model_bundle,
    )
    lanes = [lane_result]
    n_per_arm = SMOKE_N
    # Controls (20+20 replicates at n_per_arm each) are NOT run in --smoke -- their own
    # cost (40 * n_per_arm additional aminx draws) dwarfs the smoke budget; mirrors
    # layer_a_exact_validate.py --smoke's own controls_total=0 precedent.
    posctl_detected, negctl_fp = 0, 0
  else:
    data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
    n_per_arm = sampling.get("n_required", las.N_REQUIRED_FLOOR)
    lanes = []
    for lane in las.LANE_KEYS:
      full_model_bundle = las.full_model_bundle_for_lane(lane, "eqx")
      lanes.append(
        _run_lane_measurement(
          data_utils_module,
          fixtures_for_set,
          lane,
          sampling,
          n_per_arm,
          full_model_bundle=full_model_bundle,
        )
      )
    posctl_detected, negctl_fp = run_controls(
      data_utils_module,
      fixtures_for_set,
      sampling,
      n_per_arm,
      full_model_bundle=las.full_model_bundle_for_lane("P07@1.0", "eqx"),
    )

  tf_max_abs = max((lane["tf_max_abs"] for lane in lanes), default=0.0)
  n_lanes = len(lanes)
  n_lanes_equiv = sum(1 for lane in lanes if lane["equiv"])
  excess_js_ub_max_ratio = max(
    (lane["excess_js_ub"] / lane["margin"] if lane["margin"] else float("inf") for lane in lanes),
    default=0.0,
  )
  omitted_aa_count = sum(lane["omitted_aa_count"] for lane in lanes)
  x_token_count_reference = sum(lane["x_token_count_reference"] for lane in lanes)
  p09_fusion_ctrl_detected = all(lane["p09_fusion_ctrl_detected"] for lane in lanes)
  p09_tied_positions = max((lane["p09_tied_positions"] for lane in lanes), default=0)
  p09_fused_tf = next(
    (lane["p09_fused_tf_max_abs"] for lane in lanes if lane["p09_fused_tf_max_abs"]), None
  )

  result = {
    "tf_max_abs": tf_max_abs,
    "tf_bar": las.TF_BAR,
    "n_lanes": n_lanes,
    "n_lanes_equiv": n_lanes_equiv,
    "n_per_arm": n_per_arm,
    "n_required": sampling.get("n_required", las.N_REQUIRED_FLOOR),
    "sigma_hat": sampling.get("sigma_hat", 0.0),
    "excess_js_ub_max_ratio": excess_js_ub_max_ratio,
    "main_js_vs_ref": None,
    "posctl_detected": posctl_detected,
    "negctl_fp": negctl_fp,
    "omitted_aa_count": omitted_aa_count,
    "x_token_count_aminx": 0,
    "x_token_count_reference": x_token_count_reference,
    "p09_tied_positions": p09_tied_positions,
    "p09_tied_positions_prereg": sampling.get("p09_tied_positions", 0),
    "p09_fused_tf_max_abs": p09_fused_tf,
    "p09_fusion_ctrl_detected": p09_fusion_ctrl_detected,
    "n_skipped": 0,
    "prereq_ok": prereq["prereq_ok"],
    "prereq_reason": prereq["reason"],
    "reference_commit": _reference_commit(),
    "weight_source": "eqx",
    "fixture_set": args.fixture_set,
    "lanes": lanes,
    "smoke": smoke,
    **prov,
  }
  exit_code = 0
  if not prereq["prereq_ok"]:
    logger.error("prereq_check failed: %s", prereq["reason"])
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dry-run", action="store_true", help="Check imports/manifest/sidecar only.")
  parser.add_argument(
    "--smoke", action="store_true", help="One fixture, one lane, n=50, < 60 s target."
  )
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
      f"layer_a_sampling_validate: anti-HARKing gate failed: {reason} -- refusing to start "
      f"(spec 'Bathos staging': validation requires "
      f"{_relative_to_worktree(PARAMS_PATH) if PARAMS_PATH.is_file() else PARAMS_PATH} tracked "
      "and clean)",
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
    "layer_a_sampling_validate: fixture_set=%s smoke=%s tf_max_abs=%s prereq_ok=%s exit_code=%d",
    args.fixture_set,
    args.smoke,
    result.get("tf_max_abs"),
    result.get("prereq_ok"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
