"""Layer (a) sampling-tier calibration (T8b, step 4, fixture set A).

Bathos staging's "calibration" twin for `layer_a_sampling.py`'s engine -- mirrors
`layer_a_exact_calibrate.py` (T7b)'s structure and Calibration-twin outcomes (R2-C10)
exactly, but computes the `sampling` section of `preregistered_params.json` instead of
`exact`. Spec refs: `.praxia/docs/specs/260923_aminx-browser-validation.md` lines
118-200 ("Lane temperatures", "Sampling statistics", "Pooling, bootstrap and compute
budget", "P09-s lane restrictions", R3-C6).

**Every quantity below is computed at PRE-REGISTERED scale (`n_required`, 20+20
replicates, per-lane) -- nothing is shrunk to fit a local dev box.** `run_full` is
correspondingly expensive by design and is a titanix-only path; the budget cap
(`BUDGET_WALL_HOURS_CAP`/`PROJECTED_PEAK_RSS_GIB_CAP`) is the enforced safety net that
FAILS the run (no params written) rather than silently shrinking any threshold.

- **sigma_hat / n_required.** A pilot (set A, `n = PILOT_N`) of aminx draws at the
  pilot lane's temperature `T` vs `1.05*T` gives per-sequence recovery; `sigma_hat` is
  its SD. `n_required = max(1500, required_n(sigma_hat, 0.01))`
  (`aminx.parity.compare.required_n`, reused).
- **Null-replicate criterion (C2).** 20 aminx-vs-aminx replicates, each drawing `2n`
  per arm (A1/A2 vs R1/R2, exactly as validate will) at the CURRENT `n_required`,
  scored with the SAME IUT test validate uses (recovery TOST delta=0.01 over the FULL
  arms, AND `excess_js_upper` -- 1000-resample fixture-stratified bootstrap -- against
  the pilot lane's own margin). `n_required` doubles (bounded at `MAX_N_DOUBLINGS`)
  until >= 18/20 declare equivalence. If the criterion is STILL not met after
  `MAX_N_DOUBLINGS`, this run does NOT write params -- `null_criterion_met=false` is
  recorded and the run exits 1, exactly like a budget failure (never proceeds on an
  unmet criterion).
- **Margin m_l, PER LANE (C1).** For EACH of `las.LANE_KEYS`, `mean(excess_js(...))`
  between aminx@T and aminx@1.05*T (`reference_is_aminx=True` in
  `las.draw_iut_arms` -- both "arms" are aminx, no reference draws needed), pooled
  over set-A protein fixtures (`MARGIN_EXCLUDE_FIXTURES` -- `2GFB` -- excluded, with
  the reason recorded in `sampling.margin_fixtures_excluded_reason`: its cost at
  `n_required` draws would dominate the pooled allocation disproportionately to its
  designable-position weight), allocated by designable-position count, at
  `n = n_required`. For P09-s, `comparison_positions` is ALREADY restricted to
  tie-group member positions (`LaneBatch`), so this falls out of the SAME machinery
  automatically. Recorded as all 5 lane values in `sampling.margins` -- never one
  value copied across lanes.
- **Beta, PER LANE (C3).** For EACH lane, the smallest `BETA_CANDIDATES` value with
  `mean(excess_js(aminx+beta, aminx))` landing `>= 2*m_l` for THAT lane, evaluated at
  `SIZING_N` (>= 100) draws per arm. `sampling.beta` records the MAXIMUM of the 5
  per-lane values (a single scalar `beta` is what the positive control and the
  differential sentinel apply -- using the largest ensures every lane's control is at
  least as sized as its own search found); `sampling.beta_by_lane` records all 5.
  A lane with no qualifying candidate is recorded as unsized (`controls_sized` docked
  accordingly), never silently dropped or raised as a hard crash -- see
  `_search_beta_for_lane`.
- **Fusion eps (P09-s).** Smallest `FUSION_EPS_CANDIDATES` value sizing the fusion
  control into `SIZING_RATIO_RANGE` of the `1e-4` bar, on a set-A fixture with
  qualifying tie groups.
- **Budget (C4).** `sampling.budget_wall_hours` is a PROJECTION of VALIDATE's cost
  (never calibrate's own elapsed time), per `sampling.budget_formula` (also recorded
  verbatim): `(aminx_draws_total * per_draw_cost_aminx_s + reference_draws_total *
  per_draw_cost_reference_s) / 3600`, where `per_draw_cost_aminx_s` is measured
  EMPIRICALLY from the real per-lane margin draws above (`4*n_required` real aminx
  draws per lane -- the timing IS the measurement, no separate synthetic probe), and
  `per_draw_cost_reference_s` from a small dedicated batched reference call
  (`REFERENCE_COST_SAMPLE_N`). `aminx_draws_total`/`reference_draws_total` are the
  EXACT draw counts this module's own functions would issue for a full validate run
  (main lane measurement + both controls, both of which are aminx-vs-aminx per V5's
  explicit allowance -- see `_project_validate_draw_counts`'s docstring for the
  derivation). A FULL run fails (writes a diagnostic result but NOT `--params-out`)
  above the 16h/48GiB caps -- the fix is fewer set-B fixtures/lower `n_required`,
  recorded, never a raised cap.
- **`controls_total`/`controls_sized` (C5).** Computed from REAL outcomes: 5 per-lane
  beta searches + 1 fusion-eps search + 1 null-replicate criterion = 7 controls;
  `controls_sized` counts how many actually succeeded.
- **Differential sentinel (C6).** `min_effect` = half the measured
  `main_js_vs_ref`-style on/off effect on the reduced subset, at `>= SENTINEL_HALF_N`
  (>= 100) draws per arm (real reference draws for the "off"/reference side, real
  aminx draws for on/off aminx sides) -- never a 10/10 split.
- **P09-s non-vacuity floor.** `sampling.p09_qualifying_groups` /
  `sampling.p09_tied_positions` describe fixture SET B (the held-out set P09-s and
  validate actually run on), counted directly from the fixture manifest, always,
  smoke included.

**`--smoke`** runs a small pilot on ONE small fixture (`SMOKE_FIXTURE_NAME`, set A) at
ONE lane (`SMOKE_LANE`) -- never the full 5-lane x n_required x controls protocol.
Per-lane margins ARE still computed separately per lane at a SMALL smoke n (never
copied from one lane to the rest); every `sampling` field this run writes is marked
`"smoke": true` and MUST NOT be read by validate as a real params commit (requires a
non-default scratch `--params-out`, exactly like T7).
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
DEFAULT_PARAMS_OUT = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_a" / "preregistered_params.json"
)
_MANIFEST_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"

PILOT_N = 300  # spec "set A, n = 300 pilot"
SMOKE_PILOT_N = 3  # each pilot draw costs several real seconds (see _pilot_sigma_and_margin);
# 4*SMOKE_PILOT_N total aminx draws for the one combined sigma/margin computation
SMOKE_FIXTURE_NAME = "5L33"
SMOKE_LANE = "P07@1.0"
BUDGET_WALL_HOURS_CAP = 16.0
PROJECTED_PEAK_RSS_GIB_CAP = 48.0
SIZING_N = 100  # C3: "evaluate per lane at a sizing n you document, >= 100 per arm"
SENTINEL_HALF_N = 100  # C6: ">= 100 draws/arm"
REFERENCE_COST_SAMPLE_N = 10
N_BOOT = las.DEFAULT_N_BOOT  # 1000, spec-mandated everywhere (V2)
UNCOMPUTED_SENTINEL = las.UNCOMPUTED_SENTINEL  # shared with layer_a_sampling_validate.py

# C1: 2GFB's L=3464 would dominate any pooled allocation's wall-clock cost far beyond
# its designable-position weight; excluded from margin pooling (recorded, never silent).
MARGIN_EXCLUDE_FIXTURES: tuple[str, ...] = ("2GFB",)
MARGIN_EXCLUDE_REASON = (
  "2GFB (L=3464) excluded from per-lane margin pooling: at n_required draws its "
  "per-sequence AR decode cost would dominate wall time far out of proportion to its "
  "designable-position allocation weight (a handful of extra positions costing as much "
  "as the rest of set A combined). Included normally in set-B draw_allocation (a "
  "wholly separate fixture set) and everywhere else calibration touches set A."
)

EXIT_SKIPPED = 4
EXIT_NOT_WRITTEN = 1


def _load_manifest() -> dict[str, Any]:
  with _MANIFEST_PATH.open() as fh:
    return json.load(fh)


def _fixtures_for_set(manifest: dict[str, Any], fixture_set: str) -> list[dict[str, Any]]:
  return [
    f for f in manifest["fixtures"] if f.get("set") == fixture_set and f.get("kind") == "protein"
  ]


def _peak_rss_gib() -> float:
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def _pick_fixture(protein_fixtures: list[dict[str, Any]], name: str) -> dict[str, Any]:
  for fixture in protein_fixtures:
    if fixture["name"] == name:
      return fixture
  if not protein_fixtures:
    msg = "no protein fixtures available to pilot on"
    raise SystemExit(msg)
  return min(protein_fixtures, key=lambda f: f.get("L", sys.maxsize))


def _p09_set_b_counts(manifest: dict[str, Any]) -> tuple[int, int]:
  """`(p09_qualifying_groups, p09_tied_positions)` counted over fixture SET B (the held-out
  set P09-s/validate actually run on), regardless of which set THIS calibrate run pilots on."""
  set_b = [f for f in manifest["fixtures"] if f.get("set") == "B"]
  n_groups = 0
  member_positions: set[tuple[str, int]] = set()
  for fixture in set_b:
    groups = las.lae._qualifying_groups(fixture)  # noqa: SLF001 (reuse, not reimplement)
    n_groups += len(groups)
    for group in groups:
      member_positions.update((fixture["name"], m) for m in group)
  return n_groups, len(member_positions)


def _merge_params(params_out: Path, sampling_section: dict[str, Any]) -> dict[str, Any]:
  """Merge `sampling_section` into any already-committed sections (T7's `exact`, kept
  byte-identical in canonical form -- only the `sampling` key is replaced)."""
  params: dict[str, Any] = {}
  if params_out.is_file():
    with params_out.open() as fh:
      params = json.load(fh)
  params["sampling"] = sampling_section
  return params


def _lane_fixture_batches(
  protein_fixtures: list[dict[str, Any]],
  lane: str,
  data_utils_module: Any,
  *,
  exclude: tuple[str, ...] = (),
) -> list[tuple[dict[str, Any], Any]]:
  """`(fixture, LaneBatch)` pairs for `lane`, skipping `exclude`-named fixtures and any
  fixture whose `comparison_positions` is empty for this lane (e.g. P09-s with no
  qualifying tie groups on that fixture)."""
  out = []
  for fixture in protein_fixtures:
    if fixture["name"] in exclude:
      continue
    batch = las.build_lane_batch(fixture, lane, data_utils_module)
    if batch.comparison_positions.size == 0:
      continue
    out.append((fixture, batch))
  return out


def _pilot_sigma(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  n: int,
  seed_tag: str,
) -> float:
  """Per-sequence recovery SD, pooled over `fixture_batches`' aminx@T vs aminx@1.05*T draws."""
  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, n)
  a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
    jax_model,
    None,
    None,
    fixture_batches,
    allocation,
    temperature=temp,
    temperature_r=temp * las.MARGIN_TEMPERATURE_RATIO,
    reference_is_aminx=True,
    seed_tag=seed_tag,
  )
  recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)
  recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
  return float(np.std(np.concatenate([recovery_a, recovery_r]), ddof=1))


def _lane_margin_and_cost(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  n_required: int,
  seed_tag: str,
) -> tuple[float, float, int, dict[str, int]]:
  """`(margin, per_draw_cost_aminx_s, n_aminx_draws, allocation)` for one lane, at `n_required`.

  The margin computation itself draws `4*n_required` REAL aminx sequences (a1,a2 at `T`;
  r1,r2 at `1.05*T`, both "arms" aminx) -- its wall-clock time IS this lane's own
  `per_draw_cost_aminx_s` measurement (C4): no separate synthetic timing probe, the
  measurement and the margin computation are the SAME draws.
  """
  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, n_required)
  start = time.monotonic()
  a1, a2, r1, r2, _seq_ref = las.draw_iut_arms(
    jax_model,
    None,
    None,
    fixture_batches,
    allocation,
    temperature=temp,
    temperature_r=temp * las.MARGIN_TEMPERATURE_RATIO,
    reference_is_aminx=True,
    seed_tag=seed_tag,
  )
  elapsed = time.monotonic() - start
  margin = las.pooled_excess_js(a1, a2, r1, r2)
  n_draws = 4 * sum(allocation.values())
  per_draw_cost = elapsed / n_draws if n_draws else 0.0
  return margin, per_draw_cost, n_draws, allocation


def _pilot_sigma_and_margin(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  n: int,
  seed_tag: str,
) -> tuple[float, float, float]:
  """`(sigma_hat, margin, elapsed_s)` from a SINGLE `aminx@T vs aminx@1.05*T` draw set --
  both quantities are derived from the SAME draws (sigma from per-sequence recovery,
  margin from `pooled_excess_js`) rather than drawing twice. Real aminx AR sampling is
  wall-clock dominated by `WaveScheduleBundle.from_tie_groups`'s host-side per-position
  Python construction (observed empirically during this task's own development: several
  seconds per draw, not the sub-second steady state a single reused-bundle micro-benchmark
  would suggest) -- halving the draw count this way materially matters for `--smoke`."""
  import time as _time

  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, n)
  start = _time.monotonic()
  a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
    jax_model,
    None,
    None,
    fixture_batches,
    allocation,
    temperature=temp,
    temperature_r=temp * las.MARGIN_TEMPERATURE_RATIO,
    reference_is_aminx=True,
    seed_tag=seed_tag,
  )
  elapsed = _time.monotonic() - start
  recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)
  recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
  sigma_hat = float(np.std(np.concatenate([recovery_a, recovery_r]), ddof=1))
  margin = las.pooled_excess_js(a1, a2, r1, r2)
  return sigma_hat, margin, elapsed


def _null_replicate_check(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  n: int,
  margin: float,
  seed_prefix: str,
) -> int:
  """C2: 20 aminx-vs-aminx replicates, EACH drawing `2n` per arm (A1/A2 vs R1/R2, exactly
  as validate will), scored with the SAME IUT validate uses (recovery TOST over the FULL
  arms AND `excess_js_upper` (`N_BOOT`-resample bootstrap) < `margin`). Returns the count
  declaring equivalence (target `>= NULL_REPLICATES_PASS_FLOOR`)."""
  from aminx.parity.compare import tost_mean_diff

  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, n)
  null_pass = 0
  for replicate in range(las.NULL_REPLICATES):
    a1, a2, r1, r2, seq_ref_list = las.draw_iut_arms(
      jax_model,
      None,
      None,
      fixture_batches,
      allocation,
      temperature=temp,
      reference_is_aminx=True,
      seed_tag=f"{seed_prefix}null{replicate}",
    )
    recovery_a = las.full_arm_recovery(a1, a2, seq_ref_list)
    recovery_r = las.full_arm_recovery(r1, r2, seq_ref_list)
    tost_pass, _pl, _pu = tost_mean_diff(recovery_a, recovery_r, las.RECOVERY_DELTA)
    rng = np.random.default_rng(las._seed_for(f"{seed_prefix}null{replicate}boot"))
    excess_js_ub = __import__("aminx.parity.compare", fromlist=["excess_js_upper"]).excess_js_upper(
      a1, a2, r1, r2, n_boot=N_BOOT, rng=rng, k=21
    )
    if tost_pass and excess_js_ub < margin:
      null_pass += 1
  return null_pass


def _search_beta_for_lane(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  margin: float,
  seed_tag: str,
) -> tuple[float | None, bool, list[dict[str, Any]]]:
  """C3: smallest `BETA_CANDIDATES` value with `mean(excess_js(aminx+beta, aminx)) >= 2*margin`
  on THIS lane, at `SIZING_N` (>= 100) draws per arm. Returns `(beta_or_None, sized, tried)` --
  NEVER raises: an unsized lane is recorded, not silently dropped or crashed on (C5)."""
  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, SIZING_N)
  lo = las.SIZING_RATIO_RANGE[0]
  tried: list[dict[str, Any]] = []
  for beta in las.BETA_CANDIDATES:
    a1, a2, r1, r2, _seq_ref = las.draw_iut_arms(
      jax_model,
      None,
      None,
      fixture_batches,
      allocation,
      temperature=temp,
      beta_a=beta,
      reference_is_aminx=True,
      seed_tag=f"{seed_tag}beta{beta}",
    )
    effect = las.pooled_excess_js(a1, a2, r1, r2)
    ratio = effect / margin if margin > 0 else UNCOMPUTED_SENTINEL
    tried.append({"beta": beta, "ratio_to_margin": ratio, "effect": effect})
    if ratio >= lo:
      return beta, True, tried
  logger.warning(
    "lane %s: no beta in %r sized the positive control to >= %sx the margin (tried: %r)",
    lane,
    las.BETA_CANDIDATES,
    lo,
    tried,
  )
  return None, False, tried


def _search_fusion_eps(
  jax_model: Any, batch: Any, decoding_order: np.ndarray
) -> tuple[float, bool]:
  lo, hi = las.SIZING_RATIO_RANGE
  tried = []
  for eps in las.FUSION_EPS_CANDIDATES:
    control = las._fusion_sized_control(jax_model, batch, decoding_order, eps=eps)  # noqa: SLF001
    tried.append({"eps": eps, "ratio_to_bar": control["ratio_to_bar"]})
    if lo <= control["ratio_to_bar"] <= hi:
      return eps, True
  logger.warning(
    "no eps in %r sized the fusion control into [%sx, %sx] the bar (tried: %r)",
    las.FUSION_EPS_CANDIDATES,
    lo,
    hi,
    tried,
  )
  return las.FUSION_EPS_CANDIDATES[-1], False


def _project_validate_draw_counts(n_required: int, n_lanes: int) -> tuple[int, int]:
  """`(aminx_draws_total, reference_draws_total)` -- the EXACT draw counts a full validate
  run (`layer_a_sampling_validate.py`, non-smoke) would issue, per THIS module's own
  implementation (C4):

  - Main lane measurement: `2*n_required` aminx (A1+A2) AND `2*n_required` reference
    (R1+R2) per lane, summed over `n_lanes`.
  - Positive control (`run_controls`): aminx+beta (A1,A2) vs UNPERTURBED aminx (R1,R2)
    per V5's explicit allowance ("positive = aminx+beta vs aminx is acceptable") --
    `4*n_required` aminx draws, ZERO reference draws, per replicate, x20 replicates.
  - Negative control: aminx vs aminx, likewise `4*n_required` aminx draws, ZERO
    reference draws, per replicate, x20 replicates.

  So `aminx_draws_total = 2*n_required*n_lanes + 4*n_required*NULL_REPLICATES*2` and
  `reference_draws_total = 2*n_required*n_lanes` (reference draws come ONLY from the
  main lane measurement -- both controls are aminx-only by design).
  """
  aminx_draws_total = 2 * n_required * n_lanes + 4 * n_required * las.NULL_REPLICATES * 2
  reference_draws_total = 2 * n_required * n_lanes
  return aminx_draws_total, reference_draws_total


FLOOR_BUDGET_FORMULA = (
  "budget_wall_hours = sum over lanes l and set-B fixtures f of "
  "a_lf * (162 * c_aminx(f,l) + 2 * c_ref(f,l)) / 3600, with a_lf = draw_allocation at "
  "n = N_REQUIRED_FLOOR (per sub-arm, sums to n per lane). 162 = main arm A1+A2 (2) + "
  "positive control 20 replicates x (A1+A2 aminx+beta, R1+R2 aminx) (80) + negative "
  "control 20 x 4 (80) aminx draws per allocated slot; 2 = main arm R1+R2 reference draws. "
  "c_* are measured wall seconds per draw on this run's own hardware (batched aminx path "
  "incl. host-side wave construction; batched reference .sample()), so no core-count "
  "division is applied. The projection is a LOWER bound on validate's cost: n_required >= "
  "N_REQUIRED_FLOOR and cost is increasing in n."
)


def _budget_at_floor(
  jax_model: Any,
  pt_model: Any,
  torch: Any,
  protein_fixtures_b: list[dict[str, Any]],
  data_utils_module: Any,
) -> dict[str, Any]:
  """Project validate's wall time at `n = N_REQUIRED_FLOOR` from per-draw costs measured
  per (lane, set-B fixture) (D10). Validate cost only grows with `n`, so a floor projection
  above `BUDGET_WALL_HOURS_CAP` proves the pre-registered protocol cannot fit, before any
  pilot, margin or null-replicate draw is spent."""
  n_floor = las.N_REQUIRED_FLOOR
  per_lane_hours: dict[str, float] = {}
  costs: dict[str, dict[str, dict[str, float]]] = {}
  allocations: dict[str, dict[str, int]] = {}
  for lane in las.LANE_KEYS:
    lane_model = las.full_model_bundle_for_lane(lane, "eqx")
    lane_jax, lane_pt = lane_model[0], lane_model[1]
    batches = _lane_fixture_batches(protein_fixtures_b, lane, data_utils_module)
    alloc = las.allocate_draws(batches, n_floor)
    temp = las.DEFAULT_LANE_TEMPERATURES[lane]
    hours = 0.0
    costs[lane] = {}
    for fixture, batch in batches:
      c_a = las.measure_aminx_draw_cost_s(lane_jax, batch, temp)
      c_r = las.measure_reference_draw_cost_s(lane_pt, torch, batch, temp, REFERENCE_COST_SAMPLE_N)
      costs[lane][fixture["name"]] = {"aminx_s": c_a, "reference_s": c_r}
      hours += alloc[fixture["name"]] * (162 * c_a + 2 * c_r) / 3600.0
      logger.info(
        "budget floor: lane %s fixture %s L=%d alloc=%d aminx %.3fs/draw ref %.3fs/draw",
        lane,
        fixture["name"],
        batch.length,
        alloc[fixture["name"]],
        c_a,
        c_r,
      )
    per_lane_hours[lane] = hours
    allocations[lane] = alloc
  del jax_model, pt_model  # the pilot lane's models; each lane loads its own above
  total = sum(per_lane_hours.values())
  return {
    "n_floor": n_floor,
    "budget_wall_hours": total,
    "per_lane_hours": per_lane_hours,
    "per_draw_costs_s": costs,
    "draw_allocation_by_lane": allocations,
    "budget_formula": FLOOR_BUDGET_FORMULA,
  }


BUDGET_FORMULA = (
  "budget_wall_hours = (aminx_draws_total * per_draw_cost_aminx_s "
  "+ reference_draws_total * per_draw_cost_reference_s) / 3600, where "
  "aminx_draws_total = 2*n_required*n_lanes + 4*n_required*NULL_REPLICATES*2 "
  "(main-lane A1+A2 per lane, plus BOTH controls' aminx-only 4*n_required-per-replicate "
  "draws x20 replicates each -- V5: positive control is aminx+beta vs aminx, no reference "
  "draws), reference_draws_total = 2*n_required*n_lanes (main-lane R1+R2 per lane only). "
  "per_draw_cost_{aminx,reference}_s are measured empirically on THIS run's own real "
  "draws (aminx: timed directly from the per-lane margin computation's 4*n_required "
  "draws; reference: a small dedicated REFERENCE_COST_SAMPLE_N-draw batched call), "
  "already reflecting whatever thread/core pinning this process ran under -- no separate "
  "division by core count is applied (per-draw cost already measures wall time under "
  "that pinning, e.g. titanix's taskset -c 0-15 per ODQ-13)."
)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures = _fixtures_for_set(manifest, args.fixture_set)
  fixture = _pick_fixture(protein_fixtures, SMOKE_FIXTURE_NAME)

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  full_model_bundle = las.full_model_bundle_for_lane(SMOKE_LANE, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  fixture_batches = [(fixture, las.build_lane_batch(fixture, SMOKE_LANE, data_utils_module))]
  # sigma_hat AND the SMOKE_LANE's margin come from the SAME draw set (`_pilot_sigma_and_margin`)
  # -- drawing them separately would double smoke's own (empirically expensive, see that
  # function's docstring) wall time for no benefit.
  sigma_hat, smoke_lane_margin, _elapsed = _pilot_sigma_and_margin(
    jax_model, fixture_batches, SMOKE_LANE, SMOKE_PILOT_N, "smokesigmamargin"
  )
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))

  # C1 (smoke variant): the reviewer explicitly sanctions marking the OTHER four lanes'
  # margins as an explicit placeholder (never a copy of SMOKE_LANE's real value) when
  # computing all 5 at even a small n is genuinely too expensive for a smoke gate --
  # measured during this revision: `WaveScheduleBundle.from_tie_groups`'s per-draw
  # host-side construction cost made 5-lane-at-n=6 smoke run past 9m50s before being
  # killed. `margins_smoke_placeholder` records which entries are real vs placeholder.
  margins: dict[str, float] = dict.fromkeys(las.LANE_KEYS, 0.0)
  margins[SMOKE_LANE] = smoke_lane_margin
  margins_smoke_placeholder: dict[str, bool] = {lane: lane != SMOKE_LANE for lane in las.LANE_KEYS}

  p09_qualifying_groups, p09_tied_positions = _p09_set_b_counts(manifest)

  elapsed_seconds = time.monotonic() - start
  representative_margin = smoke_lane_margin

  sampling_section = {
    "git_hash": prov["git_hash"],
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "margins": margins,
    "margins_smoke_placeholder": margins_smoke_placeholder,
    "margin_fixtures_excluded_reason": MARGIN_EXCLUDE_REASON,
    "beta": las.DEFAULT_BETA,
    "beta_by_lane": dict.fromkeys(las.LANE_KEYS, las.DEFAULT_BETA),
    "reduced_subset": [fixture["name"]],
    "reduced_subset_size": 1,
    "min_effect": max(representative_margin / 2.0, sys.float_info.min),
    "measured_half_effect": representative_margin / 2.0,
    "lane_temperatures": dict(las.DEFAULT_LANE_TEMPERATURES),
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": las.DEFAULT_FUSION_CTRL_EPS,
    "draw_allocation": {fixture["name"]: SMOKE_PILOT_N},
    "budget_wall_hours": max(elapsed_seconds / 3600.0, sys.float_info.min),
    "budget_formula": BUDGET_FORMULA,
    "projected_peak_rss_gib": _peak_rss_gib(),
    "null_criterion_met": False,  # smoke never runs the doubling protocol; not a real verdict
    "fixture_set": args.fixture_set,
    "smoke": True,
  }
  params = _merge_params(args.params_out, sampling_section)
  args.params_out.parent.mkdir(parents=True, exist_ok=True)
  with args.params_out.open("w") as fh:
    json.dump(params, fh, indent=2, default=str)
  params_section_sha256 = lac.section_sha(params, "sampling")

  return {
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "margins": margins,
    "margins_smoke_placeholder": margins_smoke_placeholder,
    "beta": las.DEFAULT_BETA,
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": las.DEFAULT_FUSION_CTRL_EPS,
    "min_effect": sampling_section["min_effect"],
    "measured_half_effect": sampling_section["measured_half_effect"],
    "n_not_advanced": 0,
    "n_skipped": 0,
    "controls_total": 0,
    "controls_sized": 0,
    "budget_wall_hours": sampling_section["budget_wall_hours"],
    "projected_peak_rss_gib": sampling_section["projected_peak_rss_gib"],
    "params_written": True,
    "params_section_sha256": params_section_sha256,
    "fixture_set": args.fixture_set,
    "pilot_n": SMOKE_PILOT_N,
    "lane": SMOKE_LANE,
    "fixture": fixture["name"],
    "elapsed_seconds": elapsed_seconds,
    "smoke": True,
    **prov,
  }


def run_full(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  """Full (non-`--smoke`) calibration over `--fixture-set`. Titanix-only in practice --
  every threshold below is the PRE-REGISTERED one, never shrunk; the budget cap is the
  enforced safety net (FAILS loudly, writes no params, rather than cutting a corner)."""
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures_a = _fixtures_for_set(manifest, args.fixture_set)
  protein_fixtures_b = _fixtures_for_set(manifest, "B")
  manifest_sha256 = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  pilot_lane = args.pilot_lane
  full_model_bundle = las.full_model_bundle_for_lane(pilot_lane, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  pilot_batches = _lane_fixture_batches(
    protein_fixtures_a, pilot_lane, data_utils_module, exclude=MARGIN_EXCLUDE_FIXTURES
  )
  if not pilot_batches:
    msg = (
      f"no set-A fixtures usable for pilot lane {pilot_lane!r} "
      f"(after excluding {MARGIN_EXCLUDE_FIXTURES!r})"
    )
    raise SystemExit(msg)

  # D10: fail fast when even the n = N_REQUIRED_FLOOR projection is over budget.
  floor = _budget_at_floor(jax_model, pt_model, torch, protein_fixtures_b, data_utils_module)
  floor_rss = _peak_rss_gib()
  if floor["budget_wall_hours"] > BUDGET_WALL_HOURS_CAP or floor_rss > PROJECTED_PEAK_RSS_GIB_CAP:
    logger.error(
      "validate budget at the n=%d floor is %.1f h (cap %.1f h), peak RSS %.2f GiB (cap %.1f) "
      "-- the pre-registered protocol cannot fit; NOT running the pilot and NOT writing "
      "--params-out",
      floor["n_floor"],
      floor["budget_wall_hours"],
      BUDGET_WALL_HOURS_CAP,
      floor_rss,
      PROJECTED_PEAK_RSS_GIB_CAP,
    )
    p09_groups, p09_positions = _p09_set_b_counts(manifest)
    result = {
      "sigma_hat": 0.0,
      "sigma_hat_computed": False,
      "n_required": floor["n_floor"],
      "n_required_is_floor": True,
      "null_criterion_met": False,
      "beta": 0.0,
      "p09_qualifying_groups": p09_groups,
      "p09_tied_positions": p09_positions,
      "p09_fusion_ctrl_eps": 0.0,
      "min_effect": 0.0,
      "measured_half_effect": 0.0,
      "budget_wall_hours": floor["budget_wall_hours"],
      "budget_exceeded": True,
      "budget_floor": floor,
      "projected_peak_rss_gib": floor_rss,
      "params_written": False,
      "params_section_sha256": "",
      "n_not_advanced": 0,
      "n_skipped": 0,
      "controls_total": len(las.LANE_KEYS) + 1 + 1,
      "controls_sized": 0,
      "fixture_set": args.fixture_set,
      "fixture_manifest_sha256": manifest_sha256,
      "elapsed_seconds": time.monotonic() - start,
      **prov,
    }
    # Exit 0, not EXIT_NOT_WRITTEN: the result is complete and the sidecar's
    # budget_exceeded branch classifies it. bathos 84be544e overrides the outcome to
    # "error" on any non-zero exit (runner.py ~703), which would hide the pre-registered
    # branch from the record (observed on titanix run 1a45a859).
    return result, 0

  sigma_hat = _pilot_sigma(jax_model, pilot_batches, pilot_lane, PILOT_N, "pilotsigma")
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))

  # C2: null-replicate criterion at the CURRENT n_required, using the pilot lane's own
  # margin (recomputed at each doubling, since it must be evaluated "at n = n_required").
  n_doublings = 0
  null_criterion_met = False
  pilot_margin = 0.0
  while True:
    pilot_margin, _cost, _n, _alloc = _lane_margin_and_cost(
      jax_model, pilot_batches, pilot_lane, n_required, f"pilotmargin{n_doublings}"
    )
    null_pass = _null_replicate_check(
      jax_model, pilot_batches, pilot_lane, n_required, pilot_margin, f"doubling{n_doublings}"
    )
    if null_pass >= las.NULL_REPLICATES_PASS_FLOOR:
      null_criterion_met = True
      break
    if n_doublings >= las.MAX_N_DOUBLINGS:
      break
    n_required *= 2
    n_doublings += 1

  if not null_criterion_met:
    elapsed_seconds = time.monotonic() - start
    # Every key the sidecar's [result_schema] declares MUST be present even on this
    # early-abort path (bathos evaluate_outcome needs the full schema; a run that
    # legitimately stops early is still a real result row, not a partial one) --
    # fields not yet computed at this point get their documented zero/empty default,
    # never omitted and never inf/NaN.
    result = {
      "sigma_hat": sigma_hat,
      "n_required": n_required,
      "null_criterion_met": False,
      "n_doublings": n_doublings,
      "margins": {pilot_lane: pilot_margin},
      "beta": 0.0,
      "p09_qualifying_groups": 0,
      "p09_tied_positions": 0,
      "p09_fusion_ctrl_eps": 0.0,
      "min_effect": 0.0,
      "measured_half_effect": 0.0,
      "budget_wall_hours": 0.0,
      "projected_peak_rss_gib": _peak_rss_gib(),
      "params_written": False,
      "params_section_sha256": "",
      "n_not_advanced": 0,
      "n_skipped": 0,
      "controls_total": len(las.LANE_KEYS) + 1 + 1,
      "controls_sized": 0,
      "fixture_set": args.fixture_set,
      "fixture_manifest_sha256": manifest_sha256,
      "elapsed_seconds": elapsed_seconds,
      **prov,
    }
    logger.error(
      "null-replicate criterion NOT met after %d doublings (n_required=%d) -- NOT writing "
      "--params-out; this run's own doubled n was still insufficient, never proceeding on "
      "an unmet criterion (C2)",
      n_doublings,
      n_required,
    )
    return result, EXIT_NOT_WRITTEN

  # C1: per-lane margins, at the FINAL n_required, for the remaining 4 lanes (the pilot
  # lane's margin at this n was already computed as part of the doubling loop above).
  margins: dict[str, float] = {pilot_lane: pilot_margin}
  lane_costs: list[tuple[int, float]] = []  # (n_draws, per_draw_cost_s), for the budget projection
  pilot_n_draws = 4 * sum(las.allocate_draws(pilot_batches, n_required).values())
  # (re-time the pilot lane's own margin cost at the FINAL n once more, cleanly, rather
  # than reusing a doubling-loop timing that may have run at a smaller n)
  _pilot_margin_final, pilot_cost, pilot_n_draws, _alloc = _lane_margin_and_cost(
    jax_model, pilot_batches, pilot_lane, n_required, "pilotmarginfinal"
  )
  margins[pilot_lane] = _pilot_margin_final
  lane_costs.append((pilot_n_draws, pilot_cost))

  for lane in las.LANE_KEYS:
    if lane == pilot_lane:
      continue
    lane_batches = _lane_fixture_batches(
      protein_fixtures_a, lane, data_utils_module, exclude=MARGIN_EXCLUDE_FIXTURES
    )
    if not lane_batches:
      logger.error(
        "lane %s: no set-A fixtures usable (after excluding %r) -- margin cannot be "
        "computed; this lane will be reported as incomplete",
        lane,
        MARGIN_EXCLUDE_FIXTURES,
      )
      margins[lane] = UNCOMPUTED_SENTINEL
      continue
    margin, cost, n_draws, _alloc = _lane_margin_and_cost(
      jax_model, lane_batches, lane, n_required, f"margin{lane}"
    )
    margins[lane] = margin
    lane_costs.append((n_draws, cost))

  total_n_draws = sum(n for n, _c in lane_costs)
  per_draw_cost_aminx_s = (
    sum(n * c for n, c in lane_costs) / total_n_draws if total_n_draws else 0.0
  )

  # C3: beta, per lane, EVERY lane must be attempted (sized or recorded unsized).
  beta_by_lane: dict[str, float | None] = {}
  beta_sized_count = 0
  for lane in las.LANE_KEYS:
    lane_batches = _lane_fixture_batches(
      protein_fixtures_a, lane, data_utils_module, exclude=MARGIN_EXCLUDE_FIXTURES
    )
    if not lane_batches or margins.get(lane, UNCOMPUTED_SENTINEL) == UNCOMPUTED_SENTINEL:
      beta_by_lane[lane] = None
      continue
    beta_lane, sized, _tried = _search_beta_for_lane(
      jax_model, lane_batches, lane, margins[lane], f"betasearch{lane}"
    )
    beta_by_lane[lane] = beta_lane
    beta_sized_count += int(sized)
  beta = max((b for b in beta_by_lane.values() if b is not None), default=las.DEFAULT_BETA)

  # Fusion eps (P09-s), on a set-A fixture with qualifying groups.
  p09_qualifying_groups, p09_tied_positions = _p09_set_b_counts(manifest)
  p09_fusion_ctrl_eps = las.DEFAULT_FUSION_CTRL_EPS
  fusion_sized = False
  set_a_with_groups = next(
    (
      f
      for f in protein_fixtures_a
      if f["name"] not in MARGIN_EXCLUDE_FIXTURES and las.lae._qualifying_groups(f)
    ),  # noqa: SLF001
    None,
  )
  if set_a_with_groups is not None:
    p09s_batch = las.build_lane_batch(set_a_with_groups, "P09-s@1.0", data_utils_module)
    if p09s_batch.groups:
      _seq, _lp, decoding_order, _randn = las.reference_sample_one(
        pt_model,
        torch,
        p09s_batch,
        las._seed_for(set_a_with_groups["name"] + "p09s"),
        temperature=1.0,
      )
      p09_fusion_ctrl_eps, fusion_sized = _search_fusion_eps(jax_model, p09s_batch, decoding_order)

  # C6: differential sentinel, reduced subset (pilot fixture only), SENTINEL_HALF_N per arm.
  reduced_subset = [pilot_batches[0][0]]
  sentinel_batches = [(pilot_batches[0][0], pilot_batches[0][1])]
  sentinel_allocation = {pilot_batches[0][0]["name"]: SENTINEL_HALF_N}
  a1_off, a2_off, r1_off, r2_off, _seq_ref = las.draw_iut_arms(
    jax_model,
    pt_model,
    torch,
    sentinel_batches,
    sentinel_allocation,
    temperature=las.DEFAULT_LANE_TEMPERATURES[pilot_lane],
    reference_is_aminx=False,
    seed_tag="sentineloff",
  )
  off_metric = las.pooled_js(a1_off + a2_off, r1_off + r2_off)
  a1_on, a2_on, r1_on, r2_on, _seq_ref = las.draw_iut_arms(
    jax_model,
    pt_model,
    torch,
    sentinel_batches,
    sentinel_allocation,
    temperature=las.DEFAULT_LANE_TEMPERATURES[pilot_lane],
    beta_a=beta,
    reference_is_aminx=False,
    seed_tag="sentinelon",
  )
  on_metric = las.pooled_js(a1_on + a2_on, r1_on + r2_on)
  measured_half_effect = abs(on_metric - off_metric) / 2.0
  min_effect = measured_half_effect
  if min_effect <= 0:
    msg = (
      f"measured_half_effect={measured_half_effect!r} <= 0 (off={off_metric!r}, on={on_metric!r}) "
      "-- the sampling sentinel is not sensitive on the reduced subset"
    )
    raise SystemExit(msg)

  # C4: budget projection (VALIDATE's projected cost, never calibrate's own elapsed time).
  reference_cost_batch = _lane_fixture_batches([pilot_batches[0][0]], pilot_lane, data_utils_module)
  per_draw_cost_reference_s = las.measure_reference_draw_cost_s(
    pt_model,
    torch,
    reference_cost_batch[0][1],
    las.DEFAULT_LANE_TEMPERATURES[pilot_lane],
    REFERENCE_COST_SAMPLE_N,
  )
  aminx_draws_total, reference_draws_total = _project_validate_draw_counts(
    n_required, len(las.LANE_KEYS)
  )
  budget_wall_hours = (
    aminx_draws_total * per_draw_cost_aminx_s + reference_draws_total * per_draw_cost_reference_s
  ) / 3600.0
  projected_peak_rss_gib = _peak_rss_gib()
  within_budget = (
    budget_wall_hours <= BUDGET_WALL_HOURS_CAP
    and projected_peak_rss_gib <= PROJECTED_PEAK_RSS_GIB_CAP
  )

  allocation = las.allocate_draws(
    _lane_fixture_batches(protein_fixtures_b, "P07@1.0", data_utils_module), n_required
  )

  controls_total = len(las.LANE_KEYS) + 1 + 1  # 5 per-lane betas + fusion eps + null-replicate
  controls_sized = beta_sized_count + int(fusion_sized) + int(null_criterion_met)

  sampling_section = {
    "git_hash": prov["git_hash"],
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "null_criterion_met": null_criterion_met,
    "n_doublings": n_doublings,
    "margins": margins,
    "margin_fixtures_excluded_reason": MARGIN_EXCLUDE_REASON,
    "beta": beta,
    "beta_by_lane": beta_by_lane,
    "reduced_subset": [f["name"] for f in reduced_subset],
    "reduced_subset_size": len(reduced_subset),
    "min_effect": min_effect,
    "measured_half_effect": measured_half_effect,
    "lane_temperatures": dict(las.DEFAULT_LANE_TEMPERATURES),
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": p09_fusion_ctrl_eps,
    "draw_allocation": allocation,
    "budget_wall_hours": budget_wall_hours,
    "budget_formula": BUDGET_FORMULA,
    "per_draw_cost_aminx_s": per_draw_cost_aminx_s,
    "per_draw_cost_reference_s": per_draw_cost_reference_s,
    "aminx_draws_total_projected": aminx_draws_total,
    "reference_draws_total_projected": reference_draws_total,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
  }

  params_written = False
  params_section_sha256 = lac.section_sha({"sampling": sampling_section}, "sampling")
  if within_budget:
    params = _merge_params(args.params_out, sampling_section)
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    with args.params_out.open("w") as fh:
      json.dump(params, fh, indent=2, default=str)
    params_written = True
  else:
    logger.error(
      "budget exceeded (budget_wall_hours=%.3f cap=%.1f, projected_peak_rss_gib=%.3f cap=%.1f) "
      "-- NOT writing --params-out; reduce set-B fixtures/n_required, never raise the cap",
      budget_wall_hours,
      BUDGET_WALL_HOURS_CAP,
      projected_peak_rss_gib,
      PROJECTED_PEAK_RSS_GIB_CAP,
    )

  result = {
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "null_criterion_met": null_criterion_met,
    "margins": margins,
    "beta": beta,
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": p09_fusion_ctrl_eps,
    "min_effect": min_effect,
    "measured_half_effect": measured_half_effect,
    "params_written": params_written,
    "params_section_sha256": params_section_sha256,
    "controls_total": controls_total,
    "controls_sized": controls_sized,
    "budget_wall_hours": budget_wall_hours,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "n_not_advanced": 0,
    "n_skipped": 0,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "elapsed_seconds": time.monotonic() - start,
    **prov,
  }
  exit_code = 0
  if not within_budget:
    exit_code = 0  # sidecar budget_exceeded classifies it; see the floor branch above
  elif not prov["git_clean"]:
    exit_code = EXIT_SKIPPED
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--smoke", action="store_true", help="One fixture, one lane, small pilot, per-lane margins."
  )
  parser.add_argument(
    "--fixture-set", choices=("A", "B"), default="A", help="Manifest fixture set to calibrate on."
  )
  parser.add_argument(
    "--pilot-lane",
    choices=las.LANE_KEYS,
    default="P07@1.0",
    help="Lane used for the sigma/null-replicate pilot.",
  )
  parser.add_argument(
    "--params-out",
    type=Path,
    default=DEFAULT_PARAMS_OUT,
    help="Where to write preregistered_params.json's sampling section.",
  )
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  args = parser.parse_args(argv)

  if args.smoke and args.params_out == DEFAULT_PARAMS_OUT:
    parser.error(
      "--smoke requires a non-default --params-out (a scratch file "
      "layer_a_sampling_validate.py never reads)"
    )

  if args.smoke:
    result = run_smoke(args)
    exit_code = 0
  else:
    result, exit_code = run_full(args)

  lac.emit(result, args.out)
  logger.info(
    "layer_a_sampling_calibrate: fixture_set=%s smoke=%s n_required=%s params_written=%s "
    "exit_code=%d",
    args.fixture_set,
    args.smoke,
    result.get("n_required"),
    result.get("params_written"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
