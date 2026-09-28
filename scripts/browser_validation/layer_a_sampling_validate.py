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
this module writes (full, `--smoke`, the differential-arm path, AND a sharded partial),
with a documented zero/empty default where a field genuinely was not computed on that
path -- never omitted. A sharded partial (`n_shards > 1`) writes draw records to a
sibling `*.records.json` and emits only the flat shard schema; a graded shortfall still
exits 0. No result field is ever a bare `inf`/`nan`: `las.UNCOMPUTED_SENTINEL` (a large
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
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from aminx.parity.compare import iut_equivalent, tost_mean_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_a_exact as lae  # noqa: E402
import layer_a_sampling as las  # noqa: E402
import layer_a_sampling_shard as lass

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
    ["git", "diff", "--quiet", "HEAD", "--", rel],
    cwd=_WORKTREE_ROOT,
    check=False,
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


def shard_schema_fields(
  *,
  units_assigned: int = 0,
  units_drawn: int = 0,
  prereq_ok: bool = False,
  git_clean: bool = False,
  n_shards: int = 1,
  shard_index: int = 0,
  records_path: str = "",
  git_hash: str = "",
) -> dict[str, object]:
  """Flat shard-contract fields. ``units_assigned <= 0`` marks a path that did not draw."""
  return {
    "units_assigned": int(units_assigned),
    "units_drawn": int(units_drawn),
    "prereq_ok": bool(prereq_ok),
    "git_clean": bool(git_clean),
    "n_shards": int(n_shards),
    "shard_index": int(shard_index),
    "records_path": str(records_path),
    "git_hash": str(git_hash),
  }


def _write_shard_records(partial: dict[str, Any], out: Path) -> Path:
  """Persist the merge-ready partial (draw records included) beside the bathos result."""
  records_path = out.with_name(f"{out.stem}.records.json")
  records_path.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.loads(json.dumps(partial, default=str))
  with records_path.open("w") as handle:
    json.dump(encoded, handle, indent=2, allow_nan=False)
    handle.write("\n")
  return records_path.resolve()


def finalize_shard_partial(
  partial: dict[str, Any],
  out: Path,
  *,
  units_assigned: int,
  units_drawn: int,
  prereq_ok: bool,
  git_clean: bool,
  n_shards: int,
  shard_index: int,
  git_hash: str,
) -> tuple[dict[str, Any], int]:
  """Emit the flat shard schema and exit 0 even when the shard is graded incomplete."""
  records_path = _write_shard_records(partial, out)
  flat = shard_schema_fields(
    units_assigned=units_assigned,
    units_drawn=units_drawn,
    prereq_ok=prereq_ok,
    git_clean=git_clean,
    n_shards=n_shards,
    shard_index=shard_index,
    records_path=str(records_path),
    git_hash=git_hash,
  )
  return flat, 0


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
  mode: dict[str, Any],
  params: dict[str, Any],
  args: argparse.Namespace,
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
    **shard_schema_fields(
      n_shards=int(getattr(args, "n_shards", 1)),
      shard_index=int(getattr(args, "shard_index", 0)),
    ),
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
    **shard_schema_fields(
      n_shards=int(getattr(args, "n_shards", 1)),
      shard_index=int(getattr(args, "shard_index", 0)),
    ),
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


def _score_drawn_arms(
  lane: str,
  margin: float,
  a1: list[np.ndarray],
  a2: list[np.ndarray],
  r1: list[np.ndarray],
  r2: list[np.ndarray],
  seq_ref_list: list[np.ndarray],
) -> dict[str, object]:
  """TOST, excess-JS, and omitted-AA counts for one lane's drawn arms."""
  equiv = False
  tost_pass = False
  excess_js_point = 0.0
  excess_js_ub = las.UNCOMPUTED_SENTINEL
  if a1:
    recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)
    recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
    if recovery_a.size >= 2 and recovery_r.size >= 2:
      tost_pass, _pl, _pu = tost_mean_diff(recovery_a, recovery_r, las.RECOVERY_DELTA)
    rng = np.random.default_rng(las._seed_for(lane + "bootstrap"))
    stat = las.lane_equivalence(
      a1,
      a2,
      r1,
      r2,
      margin=margin,
      n_boot=N_BOOT,
      rng=rng,
    )
    excess_js_point = stat["excess_js"]
    excess_js_ub = stat["excess_js_ub"]
    equiv_js = stat["equiv_js"] and margin < las.UNCOMPUTED_SENTINEL
    equiv = iut_equivalent([tost_pass, equiv_js])
  omitted_aminx = sum(las.count_omitted(arr, lane) for arr in a1) + sum(
    las.count_omitted(arr, lane) for arr in a2
  )
  omitted_reference = sum(las.count_omitted(arr, lane) for arr in r1) + sum(
    las.count_omitted(arr, lane) for arr in r2
  )
  x_aminx = sum(las.count_x(arr) for arr in a1) + sum(las.count_x(arr) for arr in a2)
  x_reference = sum(las.count_x(arr) for arr in r1) + sum(las.count_x(arr) for arr in r2)
  return {
    "equiv": equiv,
    "tost_pass": tost_pass,
    "excess_js": excess_js_point,
    "excess_js_ub": excess_js_ub,
    "omitted_aa_count": omitted_aminx + omitted_reference,
    "x_token_count_aminx": x_aminx,
    "x_token_count_reference": x_reference,
  }


def _run_lane_measurement(
  data_utils_module: Any,
  fixtures_for_set: list[dict[str, Any]],
  lane: str,
  sampling: dict[str, Any],
  allocation: dict[str, int],
  *,
  full_model_bundle: tuple[Any, Any, Any, Any] | None = None,
  drawn: tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
  ]
  | None = None,
  tf_override: dict[str, object] | None = None,
) -> dict[str, Any]:
  """One lane's teacher-forced + statistical measurement, pooled over `fixtures_for_set`
  per `allocation` (V1: `sampling.draw_allocation`, never `n` per fixture)."""
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  margin = _lane_margin(sampling, lane)

  fixture_batches = []
  tf_results: list[dict[str, Any]] = []
  if tf_override is None:
    if full_model_bundle is None:
      msg = "full_model_bundle is required to teacher-force a lane"
      raise ValueError(msg)
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
        ),
      )

  if tf_override is None:
    tf_values = [r["tf_max_abs"] for r in tf_results if r.get("tf_max_abs") is not None]
    tf_max_abs = max(tf_values) if tf_values else 0.0
  else:
    tf_raw = tf_override.get("tf_max_abs", 0.0)
    tf_max_abs = float(tf_raw) if isinstance(tf_raw, int | float) else 0.0

  if drawn is None:
    if full_model_bundle is None:
      msg = "full_model_bundle is required to draw a lane"
      raise ValueError(msg)
    jax_model, pt_model, torch, _model_utils = full_model_bundle
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
  else:
    a1, a2, r1, r2, seq_ref_list = drawn
  n_arm_side = sum(x.shape[0] for x in a1)  # == sum(allocation actually used)
  n_per_arm = 2 * n_arm_side  # A1 + A2 (V1's own definition)
  scored = _score_drawn_arms(lane, margin, a1, a2, r1, r2, seq_ref_list)

  if tf_override is None:
    fusion_detected = all(
      r.get("fusion_control", {}).get("detected", True) for r in tf_results if "fusion_control" in r
    )
    p09_fused_tf = max(
      (r["p09_fused_tf_max_abs"] for r in tf_results if "p09_fused_tf_max_abs" in r),
      default=0.0,
    )
    p09_tied_positions = max(
      (r["p09_tied_positions"] for r in tf_results if "p09_tied_positions" in r),
      default=0,
    )
  else:
    fusion_raw = tf_override.get("p09_fusion_ctrl_detected", True)
    fusion_detected = bool(fusion_raw)
    fused_raw = tf_override.get("p09_fused_tf_max_abs", 0.0)
    p09_fused_tf = float(fused_raw) if isinstance(fused_raw, int | float) else 0.0
    tied_raw = tf_override.get("p09_tied_positions", 0)
    p09_tied_positions = int(tied_raw) if isinstance(tied_raw, int | float) else 0

  return {
    "lane": lane,
    "tf_max_abs": tf_max_abs,
    "tost_pass": scored["tost_pass"],
    "excess_js": scored["excess_js"],
    "excess_js_ub": scored["excess_js_ub"],
    "margin": margin,
    "equiv": scored["equiv"],
    "n_per_arm": n_per_arm,
    "omitted_aa_count": scored["omitted_aa_count"],
    "x_token_count_aminx": scored["x_token_count_aminx"],
    "x_token_count_reference": scored["x_token_count_reference"],
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
  full_model_bundle: tuple[Any, Any, Any, Any] | None = None,
  records: list[dict[str, object]] | None = None,
) -> tuple[int, int]:
  """V5: 20 disjoint-seed replicates each, pooled over set B at `allocation`, per-replicate
  `n = n_required` per half (`2n` per arm), `n_boot = N_BOOT`:

  - positive: `+beta` on alanine in the aminx arm vs UNPERTURBED aminx (V5's explicit
    allowance -- no reference draws needed); *detected* = the IUT test fails to declare
    equivalence; required `>= POSCTL_PASS_FLOOR` (18/20).
  - negative: aminx vs aminx; a *false positive* = the IUT test fails to declare
    equivalence; required `<= NEGCTL_FP_CEIL` (3/20).

  Runs on ``CONTROL_LANE`` (P07@1.0; the controls' own
  sensitivity is a property of the statistical machinery, not of which lane supplies
  the draws). Returns `(posctl_detected, negctl_fp)`.
  """
  from aminx.parity.compare import iut_equivalent, tost_mean_diff

  lane = las.CONTROL_LANE
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

  if records is not None:
    posctl_detected = 0
    negctl_fp = 0
    for replicate in range(las.N_CONTROL_REPLICATES):
      pos = _drawn_for(records, lane, "pos", replicate)
      if pos[0] and not _equiv(*pos, f"posctl{replicate}boot"):
        posctl_detected += 1
      neg = _drawn_for(records, lane, "neg", replicate)
      if neg[0] and not _equiv(*neg, f"negctl{replicate}boot"):
        negctl_fp += 1
    return posctl_detected, negctl_fp

  if full_model_bundle is None:
    msg = "full_model_bundle is required to draw controls"
    raise ValueError(msg)
  jax_model, _pt_model, _torch, _model_utils = full_model_bundle
  temperature = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)[lane]
  posctl_detected = 0
  for replicate in range(las.N_CONTROL_REPLICATES):
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
  for replicate in range(las.N_CONTROL_REPLICATES):
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


def _as_float(value: object, default: float = 0.0) -> float:
  if isinstance(value, int | float):
    return float(value)
  return default


def _stack_arm(
  records: list[dict[str, object]],
  lane: str,
  arm: str,
  replicate: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  grouped: dict[str, list[dict[str, object]]] = {}
  for record in records:
    if (
      record.get("lane") == lane
      and record.get("arm") == arm
      and int(record.get("replicate", -1)) == replicate
      and "tokens" in record
    ):
      grouped.setdefault(str(record["fixture"]), []).append(record)
  arrays: list[np.ndarray] = []
  seqs: list[np.ndarray] = []
  for fixture in sorted(grouped):
    rows = sorted(grouped[fixture], key=lambda row: int(row["draw_index"]))
    arrays.append(np.asarray([row["tokens"] for row in rows], dtype=np.int32))
    seqs.append(np.asarray(rows[0]["seq_ref"], dtype=np.int32))
  return arrays, seqs


def _drawn_for(
  records: list[dict[str, object]],
  lane: str,
  kind: str,
  replicate: int,
) -> tuple[
  list[np.ndarray],
  list[np.ndarray],
  list[np.ndarray],
  list[np.ndarray],
  list[np.ndarray],
]:
  a1, seq = _stack_arm(records, lane, f"{kind}/A1", replicate)
  a2, _seq_a2 = _stack_arm(records, lane, f"{kind}/A2", replicate)
  r1, _seq_r1 = _stack_arm(records, lane, f"{kind}/R1", replicate)
  r2, _seq_r2 = _stack_arm(records, lane, f"{kind}/R2", replicate)
  return a1, a2, r1, r2, seq


def _score_controls_from_records(
  records: list[dict[str, object]],
  sampling: dict[str, Any],
) -> tuple[int, int]:
  return run_controls(None, [], sampling, {}, records=records)


def _sample_units(
  units: list[lass.WorkUnit],
  fixtures_for_set: list[dict[str, Any]],
  sampling: dict[str, Any],
  data_utils_module: object,
  *,
  shard_index: int,
) -> list[dict[str, object]]:
  by_name = {fixture["name"]: fixture for fixture in fixtures_for_set}
  bundles: dict[str, tuple[Any, Any, Any, Any]] = {}
  batches: dict[tuple[str, str], Any] = {}
  temps = sampling.get("lane_temperatures", las.DEFAULT_LANE_TEMPERATURES)
  beta_default = _as_float(sampling.get("beta", las.DEFAULT_BETA), las.DEFAULT_BETA)
  records: list[dict[str, object]] = []
  for unit in units:
    fixture = by_name[unit.fixture]
    batch_key = (unit.lane, unit.fixture)
    if batch_key not in batches:
      batches[batch_key] = las.build_lane_batch(fixture, unit.lane, data_utils_module)
    batch = batches[batch_key]
    if unit.lane not in bundles:
      bundles[unit.lane] = las.full_model_bundle_for_lane(unit.lane, "eqx")
    jax_model, pt_model, torch, _model_utils = bundles[unit.lane]
    kind, half = unit.arm.split("/", 1)
    temperature = _as_float(temps[unit.lane], las.DEFAULT_LANE_TEMPERATURES[unit.lane])
    beta = beta_default if kind == "pos" and half in {"A1", "A2"} else 0.0
    for draw_index in range(unit.draw_start, unit.draw_end):
      seed = lass.folded_seed(
        unit.lane,
        unit.fixture,
        unit.arm,
        unit.replicate,
        draw_index,
        shard_index=shard_index,
      )
      if kind == "main" and half in {"R1", "R2"}:
        seqs = las.reference_sample_batch(pt_model, torch, batch, 1, seed, temperature=temperature)
      else:
        seqs = las.aminx_sample_batch(
          jax_model,
          batch,
          1,
          seed,
          temperature=temperature,
          beta_alanine=beta,
        )
      restricted = las.restrict_to_comparison(seqs, batch)
      seq_ref = np.asarray(batch.seq_ref[batch.comparison_positions])
      records.append(
        {
          "lane": unit.lane,
          "fixture": unit.fixture,
          "arm": unit.arm,
          "replicate": unit.replicate,
          "draw_index": draw_index,
          "tokens": np.asarray(restricted[0]).astype(int).tolist(),
          "seq_ref": seq_ref.astype(int).tolist(),
        },
      )
  return records


def statistics_from_draw_records(
  records: list[dict[str, object]],
  *,
  sampling: dict[str, Any],
  tf_by_lane: dict[str, object],
  native_x_frequency: float = 0.0,
) -> dict[str, Any]:
  """Score merged draws with the same lane and control tests as an unsharded validate."""
  lane_names = list(
    dict.fromkeys(
      str(record["lane"]) for record in records if str(record.get("arm", "")).startswith("main/")
    ),
  )
  if not lane_names:
    lane_names = [str(lane) for lane in tf_by_lane]
  lanes: list[dict[str, Any]] = []
  for lane in lane_names:
    override = tf_by_lane.get(lane, {})
    tf_override = override if isinstance(override, dict) else {}
    lanes.append(
      _run_lane_measurement(
        None,
        [],
        lane,
        sampling,
        {},
        drawn=_drawn_for(records, lane, "main", 0),
        tf_override=tf_override,
      ),
    )
  posctl_detected, negctl_fp = _score_controls_from_records(records, sampling)
  return _pack_lane_results(
    lanes,
    posctl_detected,
    negctl_fp,
    sampling,
    native_x_frequency=native_x_frequency,
  )


def _pack_lane_results(
  lanes: list[dict[str, Any]],
  posctl_detected: int,
  negctl_fp: int,
  sampling: dict[str, Any],
  *,
  native_x_frequency: float,
) -> dict[str, Any]:
  tf_max_abs = max((lane["tf_max_abs"] for lane in lanes), default=0.0)
  n_per_arm = min((lane["n_per_arm"] for lane in lanes), default=0)
  excess_js_ub_max_ratio = max(
    (
      (lane["excess_js_ub"] / lane["margin"] if lane["margin"] > 0 else las.UNCOMPUTED_SENTINEL)
      for lane in lanes
    ),
    default=0.0,
  )
  main_js_vs_ref = float(np.mean([lane["excess_js"] for lane in lanes])) if lanes else 0.0
  return {
    "tf_max_abs": tf_max_abs,
    "tf_bar": las.TF_BAR,
    "n_lanes": len(lanes),
    "n_lanes_equiv": sum(1 for lane in lanes if lane["equiv"]),
    "n_per_arm": n_per_arm,
    "n_required": sampling.get("n_required", las.N_REQUIRED_FLOOR),
    "sigma_hat": sampling.get("sigma_hat", 0.0),
    "excess_js_ub_max_ratio": excess_js_ub_max_ratio,
    "main_js_vs_ref": main_js_vs_ref,
    "posctl_detected": posctl_detected,
    "negctl_fp": negctl_fp,
    **las.grade_controls(posctl_detected, negctl_fp),
    "omitted_aa_count": sum(lane["omitted_aa_count"] for lane in lanes),
    "x_token_count_aminx": sum(lane["x_token_count_aminx"] for lane in lanes),
    "x_token_count_reference": sum(lane["x_token_count_reference"] for lane in lanes),
    "native_x_frequency": native_x_frequency,
    "p09_tied_positions": max((lane["p09_tied_positions"] for lane in lanes), default=0),
    "p09_tied_positions_prereg": sampling.get("p09_tied_positions", 0),
    "p09_fused_tf_max_abs": max((lane["p09_fused_tf_max_abs"] for lane in lanes), default=0.0),
    "p09_fusion_ctrl_detected": all(lane["p09_fusion_ctrl_detected"] for lane in lanes),
    "lanes": lanes,
  }


def _tf_map(
  selected_lanes: tuple[str, ...],
  fixtures_for_set: list[dict[str, Any]],
  sampling: dict[str, Any],
  data_utils_module: object,
) -> dict[str, object]:
  zero_alloc = {fixture["name"]: 0 for fixture in fixtures_for_set}
  out: dict[str, object] = {}
  for lane in selected_lanes:
    measured = _run_lane_measurement(
      data_utils_module,
      fixtures_for_set,
      lane,
      sampling,
      zero_alloc,
      full_model_bundle=las.full_model_bundle_for_lane(lane, "eqx"),
    )
    out[lane] = {
      "tf_max_abs": measured["tf_max_abs"],
      "p09_fusion_ctrl_detected": measured["p09_fusion_ctrl_detected"],
      "p09_fused_tf_max_abs": measured["p09_fused_tf_max_abs"],
      "p09_tied_positions": measured["p09_tied_positions"],
    }
  return out


def _v2_measurement(
  args: argparse.Namespace,
  sampling: dict[str, Any],
  fixtures_for_set: list[dict[str, Any]],
  prov: dict[str, Any],
  prereq: dict[str, Any],
) -> tuple[dict[str, Any], int]:
  selected_lanes = lass.require_control_lane(lass.parse_lanes(getattr(args, "lanes", None)))
  n_shards = int(getattr(args, "n_shards", 1))
  shard_index = int(getattr(args, "shard_index", 0))
  if n_shards < 1 or not 0 <= shard_index < n_shards:
    msg = f"--shard-index {shard_index} is outside 0..{n_shards - 1}"
    raise SystemExit(msg)
  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  allocation = sampling.get("draw_allocation", {})
  if not allocation:
    allocation = {
      fixture["name"]: sampling.get("n_required", las.N_REQUIRED_FLOOR)
      for fixture in fixtures_for_set[:1]
    }
  raw_costs = sampling.get("per_draw_costs_s")
  costs = raw_costs if isinstance(raw_costs, dict) else None
  units = lass.enumerate_work_units(
    lanes=selected_lanes,
    allocation=allocation,
    n_control_replicates=las.N_CONTROL_REPLICATES,
    costs=costs,
    control_scope=lass.PROTOCOL_CONTROL_SCOPE,
    control_lane=las.CONTROL_LANE,
  )
  plan = lass.plan_shards(units, n_shards)
  records = _sample_units(
    plan[shard_index],
    fixtures_for_set,
    sampling,
    data_utils_module,
    shard_index=shard_index,
  )
  tf_by_lane: dict[str, object] = {}
  native_x_frequency = 0.0
  if shard_index == 0:
    tf_by_lane = _tf_map(selected_lanes, fixtures_for_set, sampling, data_utils_module)
    named = [fixture for fixture in fixtures_for_set if fixture["name"] in allocation]
    if named and las.CONTROL_LANE in selected_lanes:
      bundle = las.full_model_bundle_for_lane(las.CONTROL_LANE, "eqx")
      batch = las.build_lane_batch(named[0], las.CONTROL_LANE, data_utils_module)
      if batch.comparison_positions.size:
        native_x_frequency = las.measure_native_x_frequency(
          bundle[0],
          batch,
          las.DEFAULT_LANE_TEMPERATURES[las.CONTROL_LANE],
          n=NATIVE_X_FREQ_N,
        )
  protocol: dict[str, object] = {
    "lanes": list(selected_lanes),
    "allocation": allocation,
    "n_control_replicates": las.N_CONTROL_REPLICATES,
    "control_scope": lass.PROTOCOL_CONTROL_SCOPE,
    "control_lane": las.CONTROL_LANE,
  }
  partial = lass.make_partial(
    git_hash=str(prov.get("git_hash", "")),
    params={"sampling": sampling},
    shard_index=shard_index,
    n_shards=n_shards,
    records=records,
    lanes=selected_lanes,
    n_control_replicates=las.N_CONTROL_REPLICATES,
    protocol=protocol,
  )
  partial["tf_by_lane"] = tf_by_lane
  partial["native_x_frequency"] = native_x_frequency
  partial["fixture_set"] = args.fixture_set
  partial["prereq_ok"] = prereq["prereq_ok"]
  partial["prereq_reason"] = prereq["reason"]
  partial["git_clean"] = bool(prov.get("git_clean", False))
  partial.update(lass.device_record())
  assigned = plan[shard_index]
  units_assigned = sum(unit.draw_end - unit.draw_start for unit in assigned)
  if n_shards > 1:
    return finalize_shard_partial(
      partial,
      Path(args.out),
      units_assigned=units_assigned,
      units_drawn=len(records),
      prereq_ok=bool(prereq["prereq_ok"]),
      git_clean=bool(prov.get("git_clean", False)),
      n_shards=n_shards,
      shard_index=shard_index,
      git_hash=str(prov.get("git_hash", "")),
    )
  scored = statistics_from_draw_records(
    records,
    sampling=sampling,
    tf_by_lane=tf_by_lane,
    native_x_frequency=native_x_frequency,
  )
  scored.update(
    {
      "n_skipped": 0,
      "prereq_ok": prereq["prereq_ok"],
      "prereq_reason": prereq["reason"],
      "reference_commit": _reference_commit(),
      "weight_source": "eqx",
      "fixture_set": args.fixture_set,
      "smoke": False,
      "n_shards": n_shards,
      "shard_index": shard_index,
      "n_control_replicates": las.N_CONTROL_REPLICATES,
      "selected_lanes": list(selected_lanes),
      **lass.device_record(),
      **prov,
      **shard_schema_fields(
        units_assigned=units_assigned,
        units_drawn=len(records),
        prereq_ok=bool(prereq["prereq_ok"]),
        git_clean=bool(prov.get("git_clean", False)),
        n_shards=n_shards,
        shard_index=shard_index,
        git_hash=str(prov.get("git_hash", "")),
      ),
    },
  )
  exit_code = 0
  if not prereq["prereq_ok"]:
    logger.error("prereq_check failed: %s", prereq["reason"])
  return scored, exit_code


def run_measurement(
  params: dict[str, Any],
  args: argparse.Namespace,
  *,
  smoke: bool,
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
        jax_model,
        smoke_batch,
        las.DEFAULT_LANE_TEMPERATURES[SMOKE_LANE],
        n=5,
      )
  else:
    return _v2_measurement(args, sampling, fixtures_for_set, prov, prereq)

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
    **las.grade_controls(posctl_detected, negctl_fp),
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
    "n_shards": int(getattr(args, "n_shards", 1)),
    "shard_index": int(getattr(args, "shard_index", 0)),
    "n_control_replicates": las.N_CONTROL_REPLICATES,
    **lass.device_record(),
    **prov,
    **shard_schema_fields(
      prereq_ok=bool(prereq["prereq_ok"]),
      git_clean=bool(prov.get("git_clean", False)),
      n_shards=int(getattr(args, "n_shards", 1)),
      shard_index=int(getattr(args, "shard_index", 0)),
      git_hash=str(prov.get("git_hash", "")),
    ),
  }
  exit_code = 0
  if not prereq["prereq_ok"]:
    logger.error("prereq_check failed: %s", prereq["reason"])
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dry-run", action="store_true", help="Check imports/manifest/sidecar only.")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="One fixture, one lane, n=50, < 60 s target.",
  )
  parser.add_argument(
    "--fixture-set",
    choices=("A", "B"),
    default="B",
    help="Manifest fixture set to validate on (held-out set B by default).",
  )
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--lanes",
    default=None,
    help="Comma-separated lane keys (default: the four ProteinMPNN V2_LANES).",
  )
  parser.add_argument("--n-shards", type=int, default=1, help="How many shards the plan uses.")
  parser.add_argument("--shard-index", type=int, default=0, help="Which shard this process runs.")
  args = parser.parse_args(argv)
  lass.ensure_xla_preallocate_false()

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
