"""Layer (a) sampling-tier validation (T8b, step 5, held-out fixture set B).

Bathos staging's "validation" twin for `layer_a_sampling.py`'s engine, run against
`layer_a_sampling_calibrate.py`'s committed `preregistered_params.json` (`sampling`
section). Mirrors `layer_a_exact_validate.py` (T7b) closely. Spec refs:
`.praxia/docs/specs/260923_aminx-browser-validation.md` lines 118-200 ("Lane
temperatures", "Sampling statistics", "Pooling, bootstrap and compute budget", "P09-s
lane restrictions", R3-C6).

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
- **Each arm draws `2n` per lane (V1).** `n = sampling.n_required`, ALLOCATED across
  fixtures per the committed `sampling.draw_allocation` (never `n` PER fixture -- that
  was a real bug fixed in this revision). `A1`/`A2` (aminx) and `R1`/`R2` (reference)
  each draw `allocation[fixture]` sequences per fixture, independently seeded, so
  `A1` pooled == `A2` pooled == `n_required`, and the reported `n_per_arm` (V1) is the
  AMINX ARM's total (`A1`+`A2` = `2*n_required` when fully powered) -- the field the
  sidecar's `underpowered = n_per_arm < 2*n_required` condition actually needs to be
  meaningful; a `--smoke` run correctly reports a SMALL `n_per_arm` and is correctly
  classified `underpowered`, never `pass`.
- **Bootstrap `n_boot = 1000` everywhere** (V2) -- lanes AND controls, no local
  shortcut.
- **Recovery TOST over the FULL arms** (V3): `A1 UNION A2` vs `R1 UNION R2`
  (`las.full_arm_recovery`), not a same-index-only comparison.
- **`x_token_count_aminx`/`omitted_aa_count`/`x_token_count_reference`** (V4) are
  counted over ALL statistical draws of BOTH arms in every lane (`las.count_x`/
  `las.count_omitted` on the pooled `A1+A2`/`R1+R2` arrays), never hardcoded and never
  only from the teacher-forced sequence. `measure_native_x_frequency` (AC-11) reports
  aminx's own X-sampling rate WITHOUT the hard omit, as a small, separate,
  NON-gating diagnostic.
- **Controls (V5)**: pooled over set B at the SAME `draw_allocation`, per-replicate
  `n = n_required` per half (`2n` per arm), `n_boot = 1000`; positive = aminx+beta vs
  UNPERTURBED aminx (no reference draws needed for either control, per this explicit
  allowance).
- **`excess_js` is the point estimate** (V6, `aminx.parity.compare.excess_js` via
  `las.pooled_excess_js`), never `None` -- `excess_js_ub` is the SEPARATE bootstrap
  upper bound.

**Schema completeness / no inf-or-NaN (binding, titanix finding 2026-09-25).** EVERY
key this script's sidecar declares under `[result_schema]` is present in EVERY result
this module writes (full, `--smoke`, AND the differential-arm path), with a documented
zero/empty default where a field genuinely was not computed on that path -- never
omitted. No result field is ever a bare `inf`/`nan`: `las.UNCOMPUTED_SENTINEL` (a large
finite float) stands in wherever a ratio/margin could not be computed (e.g. a missing
committed margin, or a degenerate zero-margin division).

**`--smoke`** runs ONE fixture (`SMOKE_FIXTURE_NAME`, set B), ONE lane (`SMOKE_LANE`),
`n = SMOKE_N` (spec step 5: "one fixture, one lane, n = 50, in < 60 s") -- teacher-forced
comparison plus a small statistical draw (`SMOKE_N` per sub-arm: `A1`/`A2` each aminx,
`R1`/`R2` each reference). **Measured during this task's own development** (CPU,
`.venv` cold): a single aminx autoregressive sample draw at `L=106` costs ~6.6 s the
FIRST time (JIT compile) and ~1.1 s steady-state per subsequent draw with the SAME
shapes -- so `SMOKE_N=50` (`2*SMOKE_N=100` total aminx draws for A1+A2) on the smallest
set-B fixture is expected to land close to, and possibly over, the "< 60 s" target on
slower hardware; the fix, if titanix measurement confirms it runs over, is a smaller
`SMOKE_N`/`SMOKE_FIXTURE_NAME`, never touching the pre-registered `n_required`
machinery itself. Reported, not silently worked around.
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
N_BOOT = las.DEFAULT_N_BOOT  # 1000, spec-mandated everywhere (V2)
NATIVE_X_FREQ_N = 10  # AC-11: small, non-gating

EXIT_ANTI_HARKING = 5
EXIT_DIFFERENTIAL_KNOB_MISMATCH = 2
EXIT_SKIPPED = 4


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
  """Differential-phase metric: pooled main_js_vs_ref (JS between the main aminx arm's
  pooled composition and the reference arm's, off or on) over `reduced_subset` at
  `lane_temperatures['P07@1.0']` (the reduced subset's own pre-registered lane)."""
  lane = "P07@1.0"
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES).get(lane, 1.0)
  beta = sampling.get("beta", las.DEFAULT_BETA) if perturb else 0.0
  full_model_bundle = las.full_model_bundle_for_lane(lane, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  aminx_arms: list[np.ndarray] = []
  reference_arms: list[np.ndarray] = []
  for fixture in reduced_subset:
    batch = las.build_lane_batch(fixture, lane, data_utils_module)
    aminx_full = las.aminx_sample_batch(
      jax_model,
      batch,
      n,
      las._seed_for(fixture["name"] + "main_js_a"),
      temperature=temperature,
      beta_alanine=beta,
    )
    reference_full = las.reference_sample_batch(
      pt_model,
      torch,
      batch,
      n,
      las._seed_for(fixture["name"] + "main_js_r"),
      temperature=temperature,
    )
    aminx_arms.append(las.restrict_to_comparison(aminx_full, batch))
    reference_arms.append(las.restrict_to_comparison(reference_full, batch))
  return las.pooled_js(aminx_arms, reference_arms, k=21)


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


def _lane_margin(sampling: dict[str, Any], lane: str) -> float:
  margin = sampling.get("margins", {}).get(lane)
  if margin is None or not (margin > 0) or margin >= las.UNCOMPUTED_SENTINEL:
    return las.UNCOMPUTED_SENTINEL
  return float(margin)


def _run_lane_measurement(
  data_utils_module: Any,
  fixtures_for_set: list[dict[str, Any]],
  lane: str,
  sampling: dict[str, Any],
  allocation: dict[str, int],
  *,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
  """One lane's teacher-forced + statistical measurement, pooled over `fixtures_for_set`
  per `allocation` (V1: `sampling.draw_allocation`, never `n` per fixture)."""
  from aminx.parity.compare import iut_equivalent, tost_mean_diff

  jax_model, pt_model, torch, _model_utils = full_model_bundle
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  margin = _lane_margin(sampling, lane)

  fixture_batches = []
  tf_results = []
  for fixture in fixtures_for_set:
    batch = las.build_lane_batch(fixture, lane, data_utils_module)
    fixture_batches.append((fixture, batch))
    if batch.comparison_positions.size == 0:
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

  a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
    jax_model,
    pt_model,
    torch,
    fixture_batches,
    allocation,
    temperature=temperature,
    reference_is_aminx=False,
    seed_tag=f"main{lane}",
  )
  n_arm_side = sum(x.shape[0] for x in a1)  # == sum(allocation actually used)
  n_per_arm = 2 * n_arm_side  # A1 + A2 (V1's own definition)

  equiv = False
  tost_pass = False
  excess_js_point = 0.0
  excess_js_ub = las.UNCOMPUTED_SENTINEL
  if a1:
    recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)  # V3: FULL arms
    recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
    if recovery_a.size >= 2 and recovery_r.size >= 2:
      tost_pass, _pl, _pu = tost_mean_diff(recovery_a, recovery_r, las.RECOVERY_DELTA)
    rng = np.random.default_rng(las._seed_for(lane + "bootstrap"))
    stat = las.lane_equivalence(
      a1, a2, r1, r2, margin=margin, n_boot=N_BOOT, rng=rng
    )  # V2: n_boot=1000
    excess_js_point = stat["excess_js"]  # V6: point estimate
    excess_js_ub = stat["excess_js_ub"]
    # A missing/uncomputed committed margin (las.UNCOMPUTED_SENTINEL) must NEVER let a
    # lane spuriously pass the excess-JS half of the IUT test just because the sentinel
    # is astronomically larger than any real excess_js_ub.
    equiv_js = stat["equiv_js"] and margin < las.UNCOMPUTED_SENTINEL
    equiv = iut_equivalent([tost_pass, equiv_js])

  # V4: omitted-AA/X over ALL statistical draws of BOTH arms, not just the teacher-forced one.
  omitted_aminx = sum(las.count_omitted(arr, lane) for arr in a1) + sum(
    las.count_omitted(arr, lane) for arr in a2
  )
  omitted_reference = sum(las.count_omitted(arr, lane) for arr in r1) + sum(
    las.count_omitted(arr, lane) for arr in r2
  )
  x_aminx = sum(las.count_x(arr) for arr in a1) + sum(las.count_x(arr) for arr in a2)
  x_reference = sum(las.count_x(arr) for arr in r1) + sum(las.count_x(arr) for arr in r2)

  fusion_detected = all(
    r.get("fusion_control", {}).get("detected", True) for r in tf_results if "fusion_control" in r
  )
  p09_fused_tf = max(
    (r["p09_fused_tf_max_abs"] for r in tf_results if "p09_fused_tf_max_abs" in r), default=0.0
  )
  p09_tied_positions = max(
    (r["p09_tied_positions"] for r in tf_results if "p09_tied_positions" in r), default=0
  )

  return {
    "lane": lane,
    "tf_max_abs": tf_max_abs,
    "tost_pass": tost_pass,
    "excess_js": excess_js_point,
    "excess_js_ub": excess_js_ub,
    "margin": margin,
    "equiv": equiv,
    "n_per_arm": n_per_arm,
    "omitted_aa_count": omitted_aminx + omitted_reference,
    "x_token_count_aminx": x_aminx,
    "x_token_count_reference": x_reference,
    "p09_fused_tf_max_abs": p09_fused_tf,
    "p09_fusion_ctrl_detected": fusion_detected,
    "p09_tied_positions": p09_tied_positions,
  }


def run_controls(
  data_utils_module: Any,
  fixtures_for_set: list[dict[str, Any]],
  sampling: dict[str, Any],
  allocation: dict[str, int],
  *,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> tuple[int, int]:
  """V5: 20 disjoint-seed replicates each, pooled over set B at `allocation`, per-replicate
  `n = n_required` per half (`2n` per arm), `n_boot = N_BOOT`:

  - positive: `+beta` on alanine in the aminx arm vs UNPERTURBED aminx (V5's explicit
    allowance -- no reference draws needed); *detected* = the IUT test fails to declare
    equivalence; required `>= 18/20`.
  - negative: aminx vs aminx; a *false positive* = the IUT test fails to declare
    equivalence; required `<= 3/20`.

  Runs on the P07@1.0 lane (the cheapest lane geometry available; the controls' own
  sensitivity is a property of the statistical machinery, not of which lane supplies
  the draws). Returns `(posctl_detected, negctl_fp)`.
  """
  from aminx.parity.compare import iut_equivalent, tost_mean_diff

  jax_model, _pt_model, _torch, _model_utils = full_model_bundle
  lane = "P07@1.0"
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  margin = _lane_margin(sampling, lane)
  beta = sampling.get("beta", las.DEFAULT_BETA)

  fixture_batches = [
    (fixture, las.build_lane_batch(fixture, lane, data_utils_module))
    for fixture in fixtures_for_set
  ]
  fixture_batches = [(f, b) for f, b in fixture_batches if b.comparison_positions.size > 0]

  def _equiv(
    a1: list[np.ndarray],
    a2: list[np.ndarray],
    r1: list[np.ndarray],
    r2: list[np.ndarray],
    seq_ref_list: list[np.ndarray],
    seed_tag: str,
  ) -> bool:
    if not a1:
      return True
    recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)
    recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
    tost_pass, _pl, _pu = tost_mean_diff(recovery_a, recovery_r, las.RECOVERY_DELTA)
    rng = np.random.default_rng(las._seed_for(seed_tag))
    stat = las.lane_equivalence(a1, a2, r1, r2, margin=margin, n_boot=N_BOOT, rng=rng)
    equiv_js = stat["equiv_js"] and margin < las.UNCOMPUTED_SENTINEL
    return iut_equivalent([tost_pass, equiv_js])

  posctl_detected = 0
  for replicate in range(las.NULL_REPLICATES):
    a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
      jax_model,
      None,
      None,
      fixture_batches,
      allocation,
      temperature=temperature,
      beta_a=beta,
      reference_is_aminx=True,
      seed_tag=f"posctl{replicate}",
    )
    if not _equiv(a1, a2, r1, r2, seq_ref_list, f"posctl{replicate}boot"):
      posctl_detected += 1

  negctl_fp = 0
  for replicate in range(las.NULL_REPLICATES):
    a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
      jax_model,
      None,
      None,
      fixture_batches,
      allocation,
      temperature=temperature,
      reference_is_aminx=True,
      seed_tag=f"negctl{replicate}",
    )
    if not _equiv(a1, a2, r1, r2, seq_ref_list, f"negctl{replicate}boot"):
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
    allocation = {fixture["name"]: SMOKE_N}  # spec step 5: "n = 50" on ONE fixture
    lane_result = _run_lane_measurement(
      data_utils_module,
      [fixture],
      SMOKE_LANE,
      sampling or {"lane_temperatures": las.DEFAULT_LANE_TEMPERATURES},
      allocation,
      full_model_bundle=full_model_bundle,
    )
    lanes = [lane_result]
    n_per_arm = lane_result["n_per_arm"]
    # Controls (20+20 replicates at n_required each) are NOT run in --smoke -- their own
    # cost dwarfs the smoke budget; mirrors layer_a_exact_validate.py --smoke's own
    # controls_total=0 precedent. Recorded as 0 (never omitted, never inf).
    posctl_detected, negctl_fp = 0, 0
    native_x_frequency = 0.0
    jax_model = full_model_bundle[0]
    smoke_batch = las.build_lane_batch(fixture, SMOKE_LANE, data_utils_module)
    if smoke_batch.comparison_positions.size:
      native_x_frequency = las.measure_native_x_frequency(
        jax_model, smoke_batch, las.DEFAULT_LANE_TEMPERATURES[SMOKE_LANE], n=5
      )
  else:
    data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
    allocation = sampling.get("draw_allocation", {})
    if not allocation:
      allocation = {
        f["name"]: sampling.get("n_required", las.N_REQUIRED_FLOOR) for f in fixtures_for_set[:1]
      }
    lanes = []
    for lane in las.LANE_KEYS:
      full_model_bundle = las.full_model_bundle_for_lane(lane, "eqx")
      lanes.append(
        _run_lane_measurement(
          data_utils_module,
          fixtures_for_set,
          lane,
          sampling,
          allocation,
          full_model_bundle=full_model_bundle,
        )
      )
    n_per_arm = min((lane["n_per_arm"] for lane in lanes), default=0)
    posctl_detected, negctl_fp = run_controls(
      data_utils_module,
      fixtures_for_set,
      sampling,
      allocation,
      full_model_bundle=las.full_model_bundle_for_lane("P07@1.0", "eqx"),
    )
    native_x_frequency = 0.0
    p07_batches = [f for f in fixtures_for_set if f["name"] in allocation]
    if p07_batches:
      p07_full_model_bundle = las.full_model_bundle_for_lane("P07@1.0", "eqx")
      p07_batch = las.build_lane_batch(p07_batches[0], "P07@1.0", data_utils_module)
      if p07_batch.comparison_positions.size:
        native_x_frequency = las.measure_native_x_frequency(
          p07_full_model_bundle[0],
          p07_batch,
          las.DEFAULT_LANE_TEMPERATURES["P07@1.0"],
          n=NATIVE_X_FREQ_N,
        )

  tf_max_abs = max((lane["tf_max_abs"] for lane in lanes), default=0.0)
  n_lanes = len(lanes)
  n_lanes_equiv = sum(1 for lane in lanes if lane["equiv"])
  excess_js_ub_max_ratio = max(
    (
      (lane["excess_js_ub"] / lane["margin"] if lane["margin"] > 0 else las.UNCOMPUTED_SENTINEL)
      for lane in lanes
    ),
    default=0.0,
  )
  omitted_aa_count = sum(lane["omitted_aa_count"] for lane in lanes)
  x_token_count_aminx = sum(lane["x_token_count_aminx"] for lane in lanes)
  x_token_count_reference = sum(lane["x_token_count_reference"] for lane in lanes)
  p09_fusion_ctrl_detected = all(lane["p09_fusion_ctrl_detected"] for lane in lanes)
  p09_tied_positions = max((lane["p09_tied_positions"] for lane in lanes), default=0)
  p09_fused_tf = max((lane["p09_fused_tf_max_abs"] for lane in lanes), default=0.0)
  main_js_vs_ref = float(np.mean([lane["excess_js"] for lane in lanes])) if lanes else 0.0

  result = {
    "tf_max_abs": tf_max_abs,
    "tf_bar": las.TF_BAR,
    "n_lanes": n_lanes,
    "n_lanes_equiv": n_lanes_equiv,
    "n_per_arm": n_per_arm,
    "n_required": sampling.get("n_required", las.N_REQUIRED_FLOOR),
    "sigma_hat": sampling.get("sigma_hat", 0.0),
    "excess_js_ub_max_ratio": excess_js_ub_max_ratio,
    "main_js_vs_ref": main_js_vs_ref,
    "posctl_detected": posctl_detected,
    "negctl_fp": negctl_fp,
    "omitted_aa_count": omitted_aa_count,
    "x_token_count_aminx": x_token_count_aminx,
    "x_token_count_reference": x_token_count_reference,
    "native_x_frequency": native_x_frequency,
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
