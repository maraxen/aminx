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

**T10e pre-registration AMENDMENT (this revision), made BEFORE any run has used it, in
response to run e091a33e's `ctrl_unsized` grade (5/6 controls):** the P07@0.1 lane is
DROPPED from the selected lanes -- `las.V2_LANES` is now `(P07@1.0, P08@1.0, P09-s@1.0)`,
three lanes, not four -- per user decision, because (a) its own margin (aminx@T vs
aminx@1.05*T excess-JS) measured -8.5e-5, below estimator noise at this low a
temperature, which the old `_search_beta_for_lane` silently mishandled by comparing a
"sized" ratio against `UNCOMPUTED_SENTINEL` instead of declaring the lane unsized, and
(b) T=0.1 stays covered bitwise by the P07 knobs gate (`layer_a_exact.py`) regardless, so
nothing is lost by removing it from this distributional tier. Every "5 lanes"/"4 lanes"
count below is now "3 lanes"/"2 lanes" and `controls_total` follows mechanically from
`len(selected_lanes)` (3 betas + 1 fusion eps + 1 null-replicate = 5, not 7). The P09-s
fusion-eps grid (`las.FUSION_EPS_CANDIDATES`) is separately extended downward (a second,
independent defect: the old grid floored at 0.01 and never reached `SIZING_RATIO_RANGE`)
-- see that constant's own docstring in `layer_a_sampling.py`.

- **sigma_hat / n_required.** A pilot (set A, `n = PILOT_N`) of aminx draws at the
  pilot lane's temperature `T` vs `1.05*T` gives per-sequence recovery; `sigma_hat` is
  its SD. `n_required = max(1500, required_n(sigma_hat, 0.01))`
  (`aminx.parity.compare.required_n`, reused).
- **Null-replicate criterion (C2).** 20 aminx-vs-aminx replicates, each drawing `2n`
  per arm (A1/A2 vs R1/R2, exactly as validate will) at the CURRENT `n_required`,
  scored with the SAME IUT test validate uses (recovery TOST delta=0.01 over the FULL
  arms, AND `excess_js_upper` -- 1000-resample fixture-stratified bootstrap -- against
  the pilot lane's own margin). `n_required` doubles (bounded at `MAX_N_DOUBLINGS`)
  until >= POSCTL_PASS_FLOOR of N_CONTROL_REPLICATES (18/20) declare equivalence.
  If the criterion is STILL not met after `MAX_N_DOUBLINGS`, this run does NOT
  write params -- `null_criterion_met=false` is
  recorded and the run exits 1, exactly like a budget failure (never proceeds on an
  unmet criterion).
- **Margin m_l, PER LANE (C1).** For EACH of `selected_lanes` (default `las.V2_LANES`,
  three lanes post-T10e), `mean(excess_js(...))`
  between aminx@T and aminx@1.05*T (`reference_is_aminx=True` in
  `las.draw_iut_arms` -- both "arms" are aminx, no reference draws needed), pooled
  over set-A protein fixtures (`MARGIN_EXCLUDE_FIXTURES` -- `2GFB` -- excluded, with
  the reason recorded in `sampling.margin_fixtures_excluded_reason`: its cost at
  `n_required` draws would dominate the pooled allocation disproportionately to its
  designable-position weight), allocated by designable-position count, at
  `n = n_required`. For P09-s, `comparison_positions` is ALREADY restricted to
  tie-group member positions (`LaneBatch`), so this falls out of the SAME machinery
  automatically. Recorded as all `len(selected_lanes)` lane values in `sampling.margins`
  -- never one value copied across lanes.
- **Beta, PER LANE (C3).** For EACH lane, the smallest `BETA_CANDIDATES` value with
  `mean(excess_js(aminx+beta, aminx))` landing `>= 2*m_l` for THAT lane, evaluated at
  `SIZING_N` (>= 100) draws per arm. `sampling.beta` records the MAXIMUM of the
  per-lane values (a single scalar `beta` is what the positive control and the
  differential sentinel apply -- using the largest ensures every lane's control is at
  least as sized as its own search found); `sampling.beta_by_lane` records all of them.
  A lane with no qualifying candidate is recorded as unsized (`controls_sized` docked
  accordingly), never silently dropped or raised as a hard crash -- see
  `_search_beta_for_lane`. A margin that is `<= 0` or non-finite (T10e: P07@0.1's own
  margin was -8.5e-5) is ALSO unsized immediately, never compared against
  `UNCOMPUTED_SENTINEL` as if it were a real ratio.
- **Fusion eps (P09-s).** Smallest `FUSION_EPS_CANDIDATES` value sizing the fusion
  control into `SIZING_RATIO_RANGE` of the `1e-4` bar, on a set-A fixture with
  qualifying tie groups. T10e: the grid now extends down to `1e-5` (was `0.01`); when
  nothing sizes, `_search_fusion_eps` returns `(0.0, False)` -- never the grid's last
  candidate, which would be indistinguishable from a genuinely sized value.
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
- **`controls_total`/`controls_sized` (C5).** Computed from REAL outcomes:
  `len(selected_lanes)` per-lane beta searches + 1 fusion-eps search + 1 null-replicate
  criterion (post-T10e, with the default 3-lane `V2_LANES`: 3 + 1 + 1 = 5 controls, not
  the pre-amendment 7); `controls_sized` counts how many actually succeeded.
- **Differential sentinel (C6).** `min_effect` = half the measured
  `main_js_vs_ref`-style on/off effect on the reduced subset, at `>= SENTINEL_HALF_N`
  (>= 100) draws per arm (real reference draws for the "off"/reference side, real
  aminx draws for on/off aminx sides) -- never a 10/10 split.
- **P09-s non-vacuity floor.** `sampling.p09_qualifying_groups` /
  `sampling.p09_tied_positions` describe fixture SET B (the held-out set P09-s and
  validate actually run on), counted directly from the fixture manifest, always,
  smoke included.

**`--smoke`** runs a small pilot on ONE small fixture (`SMOKE_FIXTURE_NAME`, set A) at
ONE lane (`SMOKE_LANE`) -- never the full 3-lane x n_required x controls protocol
(post-T10e; was 5-lane pre-amendment).
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
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac
import layer_a_exact as lae
import layer_a_sampling as las
import layer_a_sampling_shard as lass

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


# --------------------------------------------------------------------------------------
# T10g: checkpointing + progress logging for run_full's 6-7+ hour, previously-silent,
# previously-unresumable middle stretch (pilot -> null-replicate doubling -> per-lane
# margins -> per-lane beta search -> fusion-eps search -> differential sentinel ->
# budget projection -> params write).
# --------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, obj: Any) -> None:
  """tmp-file + `os.replace` -- never a partially-written checkpoint file, even if the
  process is killed mid-write (the failure mode this whole feature exists to survive)."""
  tmp = path.with_name(path.name + f".tmp{os.getpid()}")
  with tmp.open("w") as fh:
    json.dump(obj, fh, indent=2, sort_keys=True)
  os.replace(tmp, path)


class CheckpointStore:
  """One JSON file per completed checkpoint UNIT under `checkpoint_dir`
  (`<stage>__<key>.json`), each carrying a `fingerprint` (sha256 over this run's
  identity -- git hash, fixture_set, smoke flag, selected lanes, every constant the
  unit's own computation depends on -- plus the unit's own JSON-round-trippable
  `value`). `checkpoint_dir=None` makes every method a pure pass-through/no-op --
  callers do not need an `if store is not None` at every call site (they still may,
  for clarity, but it is never required for correctness).

  A load with a MATCHING fingerprint short-circuits recomputation (`resumed <unit>`,
  INFO) -- this is what makes `run_full` resumable at the exact seed/data a
  never-interrupted run would have used (T10g's `_seed_for` fix, see
  `layer_a_sampling.py`, is what makes that recomputation-free load bit-identical to
  what an uninterrupted run would have computed in the first place). A MISMATCHING
  fingerprint is logged WARNING and recomputed -- a stale checkpoint from a different
  commit, fixture set, lane selection, or constant must NEVER be silently reused.
  """

  def __init__(self, checkpoint_dir: Path | None, base_fingerprint: dict[str, Any]) -> None:
    self.checkpoint_dir = checkpoint_dir
    self.base_fingerprint = dict(base_fingerprint)
    self.units_resumed: list[str] = []
    if self.checkpoint_dir is not None:
      self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

  @staticmethod
  def _safe_unit_name(unit: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in unit)

  def _path(self, unit: str) -> Path:
    if self.checkpoint_dir is None:
      msg = "CheckpointStore._path called with checkpoint_dir=None"
      raise RuntimeError(msg)
    return self.checkpoint_dir / f"{self._safe_unit_name(unit)}.json"

  def fingerprint(self, unit: str, extra: dict[str, Any]) -> str:
    payload = {"unit": unit, **self.base_fingerprint, **extra}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

  def load(self, unit: str, expected_fingerprint: str) -> tuple[bool, Any]:
    """`(found, value)`. `found=False` means "recompute" -- missing, unreadable, OR a
    fingerprint mismatch (the last is logged WARNING; the other two are silent since
    they are the ordinary first-run/no-checkpoint-yet case, not an anomaly)."""
    if self.checkpoint_dir is None:
      return False, None
    path = self._path(unit)
    if not path.is_file():
      return False, None
    try:
      with path.open() as fh:
        data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
      logger.warning("checkpoint %s: unreadable (%s) -- recomputing", unit, exc)
      return False, None
    stored_fp = data.get("fingerprint")
    if stored_fp != expected_fingerprint:
      logger.warning(
        "checkpoint %s: fingerprint mismatch (stored=%s expected=%s) -- recomputing, "
        "never silently reusing a stale checkpoint",
        unit,
        stored_fp,
        expected_fingerprint,
      )
      return False, None
    logger.info("resumed %s", unit)
    if unit not in self.units_resumed:
      self.units_resumed.append(unit)
    return True, data["value"]

  def save(self, unit: str, fingerprint_value: str, value: Any) -> None:
    if self.checkpoint_dir is None:
      return
    _atomic_write_json(self._path(unit), {"unit": unit, "fingerprint": fingerprint_value, "value": value})

  def get_or_compute(
    self,
    unit: str,
    extra_fingerprint: dict[str, Any],
    compute: Callable[[], Any],
  ) -> Any:
    """Load-or-compute for one unit. `compute` runs iff no valid checkpoint is found
    (missing, unreadable, or fingerprint-mismatched) -- or always, when checkpointing
    is disabled (`checkpoint_dir is None`), which is the default and leaves behavior
    completely unchanged from before T10g except for the logging callers add around it."""
    if self.checkpoint_dir is None:
      return compute()
    fp = self.fingerprint(unit, extra_fingerprint)
    found, value = self.load(unit, fp)
    if found:
      return value
    value = compute()
    self.save(unit, fp, value)
    return value


class _StageLog:
  """Context manager: INFO stage-start/stage-end lines with per-stage AND
  cumulative-since-run-start elapsed seconds, via the module logger (flush-friendly
  when the caller sets `PYTHONUNBUFFERED=1` -- see `titanix_run.sh`/`engaging_run.sbatch`)."""

  def __init__(self, run_start: float, name: str) -> None:
    self._run_start = run_start
    self._name = name
    self._stage_start = 0.0

  def __enter__(self) -> "_StageLog":
    self._stage_start = time.monotonic()
    logger.info(
      "stage start: %s (cumulative %.1fs)",
      self._name,
      self._stage_start - self._run_start,
    )
    return self

  def __exit__(self, *exc: object) -> None:
    now = time.monotonic()
    logger.info(
      "stage end: %s (stage %.1fs, cumulative %.1fs)",
      self._name,
      now - self._stage_start,
      now - self._run_start,
    )


def _checkpointed_lane_margin(
  store: CheckpointStore | None,
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  n_required: int,
  seed_tag: str,
) -> tuple[float, float, int, dict[str, int]]:
  """`_lane_margin_and_cost`, checkpointed as unit `lanemargin_{lane}_n{n_required}_{seed_tag}`.

  The checkpointed VALUE includes the measured `per_draw_cost_s` -- a wall-clock
  timing, not a pure function of identity -- so a resumed run reuses the ORIGINAL
  measurement rather than re-timing (which would legitimately differ run to run);
  this is what keeps a resumed run's budget projection bit-identical to an
  uninterrupted one, not just its margins.
  """
  unit = f"lanemargin_{lane}_n{n_required}_{seed_tag}"

  def _compute() -> dict[str, Any]:
    margin, cost, n_draws, allocation = _lane_margin_and_cost(
      jax_model,
      fixture_batches,
      lane,
      n_required,
      seed_tag,
    )
    return {
      "margin": margin,
      "per_draw_cost_s": cost,
      "n_draws": n_draws,
      "allocation": allocation,
    }

  if store is not None:
    value = store.get_or_compute(
      unit,
      {"lane": lane, "n_required": n_required, "seed_tag": seed_tag},
      _compute,
    )
  else:
    value = _compute()
  return value["margin"], value["per_draw_cost_s"], value["n_draws"], value["allocation"]


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
  *,
  store: CheckpointStore | None = None,
  n_doublings: int = 0,
  run_start: float | None = None,
) -> int:
  """C2: 20 aminx-vs-aminx replicates, EACH drawing `2n` per arm (A1/A2 vs R1/R2, exactly
  as validate will), scored with the SAME IUT validate uses (recovery TOST over the FULL
  arms AND `excess_js_upper` (`N_BOOT`-resample bootstrap) < `margin`). Returns the count
  declaring equivalence (target `>= POSCTL_PASS_FLOOR`, the same 18/20 bar).

  T10g: each replicate is its own checkpoint unit (`nullreplicate_d{n_doublings}_r{i}_n{n}`)
  -- this is the single most expensive loop in `run_full` (up to `MAX_N_DOUBLINGS + 1`
  doubling levels x `NULL_REPLICATES` replicates x `4*n` real draws each), so per-replicate
  resumability is what actually prevents a crash from losing hours of work. `store=None`
  (the default) reproduces the exact pre-T10g behavior with no checkpoint I/O at all.
  """
  from aminx.parity.compare import tost_mean_diff

  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, n)
  null_pass = 0
  for replicate in range(las.NULL_REPLICATES):
    unit = f"nullreplicate_d{n_doublings}_r{replicate}_n{n}"

    def _compute(replicate: int = replicate) -> dict[str, Any]:
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
      excess_js_ub = __import__(
        "aminx.parity.compare",
        fromlist=["excess_js_upper"],
      ).excess_js_upper(a1, a2, r1, r2, n_boot=N_BOOT, rng=rng, k=21)
      return {"tost_pass": bool(tost_pass), "excess_js_ub": float(excess_js_ub)}

    if store is not None:
      value = store.get_or_compute(
        unit,
        {
          "lane": lane,
          "n": n,
          "n_doublings": n_doublings,
          "replicate": replicate,
          "null_replicates": las.NULL_REPLICATES,
          "n_boot": N_BOOT,
          "recovery_delta": las.RECOVERY_DELTA,
          "seed_prefix": seed_prefix,
        },
        _compute,
      )
    else:
      value = _compute()

    tost_pass = value["tost_pass"]
    excess_js_ub = value["excess_js_ub"]
    if tost_pass and excess_js_ub < margin:
      null_pass += 1
    elapsed = (time.monotonic() - run_start) if run_start is not None else -1.0
    logger.info(
      "null replicate %d/%d doubling=%d n=%d tost_pass=%s excess_js_ub=%.6g margin=%.6g "
      "running_pass=%d/%d (cumulative %.1fs)",
      replicate + 1,
      las.NULL_REPLICATES,
      n_doublings,
      n,
      tost_pass,
      excess_js_ub,
      margin,
      null_pass,
      replicate + 1,
      elapsed,
    )
  return null_pass


def _search_beta_for_lane(
  jax_model: Any,
  fixture_batches: list[tuple[dict[str, Any], Any]],
  lane: str,
  margin: float,
  seed_tag: str,
  *,
  store: CheckpointStore | None = None,
  run_start: float | None = None,
) -> tuple[float | None, bool, list[dict[str, Any]]]:
  """C3: smallest `BETA_CANDIDATES` value with `mean(excess_js(aminx+beta, aminx)) >= 2*margin`
  on THIS lane, at `SIZING_N` (>= 100) draws per arm. Returns `(beta_or_None, sized, tried)` --
  NEVER raises: an unsized lane is recorded, not silently dropped or crashed on (C5).

  T10e fix (run e091a33e): a `margin` that is `<= 0` or non-finite makes `effect / margin`
  meaningless -- P07@0.1's own margin measured -8.5e-5 (below estimator noise at low T),
  and the old code's `margin > 0 else UNCOMPUTED_SENTINEL` fallback compared `ratio` (the
  1e18 sentinel) against `lo`, which ALWAYS passes on the first candidate -- a silent false
  "sized" that spent zero real signal. Such a lane is declared UNSIZED immediately, with
  `tried == []` (no draws are wasted searching against a margin that cannot be sized
  against), and the sentinel is never compared as if it were a real ratio.
  """
  if margin is None or not np.isfinite(margin) or margin <= 0:
    logger.warning(
      "lane %s: margin %r is <= 0 or non-finite -- cannot size a beta against it; "
      "declaring UNSIZED without running the beta search",
      lane,
      margin,
    )
    return None, False, []
  temp = las.DEFAULT_LANE_TEMPERATURES[lane]
  allocation = las.allocate_draws(fixture_batches, SIZING_N)
  lo = las.SIZING_RATIO_RANGE[0]
  tried: list[dict[str, Any]] = []
  for beta in las.BETA_CANDIDATES:
    candidate_seed_tag = f"{seed_tag}beta{beta}"
    unit = f"betaeval_{lane}_beta{beta}"

    def _compute(beta: float = beta, candidate_seed_tag: str = candidate_seed_tag) -> dict[str, Any]:
      a1, a2, r1, r2, _seq_ref = las.draw_iut_arms(
        jax_model,
        None,
        None,
        fixture_batches,
        allocation,
        temperature=temp,
        beta_a=beta,
        reference_is_aminx=True,
        seed_tag=candidate_seed_tag,
      )
      return {"effect": las.pooled_excess_js(a1, a2, r1, r2)}

    if store is not None:
      value = store.get_or_compute(
        unit,
        {"lane": lane, "beta": beta, "sizing_n": SIZING_N, "seed_tag": candidate_seed_tag},
        _compute,
      )
    else:
      value = _compute()
    effect = value["effect"]
    ratio = effect / margin
    tried.append({"beta": beta, "ratio_to_margin": ratio, "effect": effect})
    elapsed = (time.monotonic() - run_start) if run_start is not None else -1.0
    logger.info(
      "beta candidate lane=%s beta=%s effect=%.6g ratio_to_margin=%.4f (cumulative %.1fs)",
      lane,
      beta,
      effect,
      ratio,
      elapsed,
    )
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
  jax_model: Any,
  batch: Any,
  decoding_order: np.ndarray,
  *,
  store: CheckpointStore | None = None,
  fixture_name: str = "",
  run_start: float | None = None,
) -> tuple[float, bool]:
  """Smallest `FUSION_EPS_CANDIDATES` value sizing the fusion control's `ratio_to_bar` into
  `SIZING_RATIO_RANGE`. Returns `(eps, sized)`.

  T10e fix (run e091a33e): when NOTHING in the grid sizes, this must return a value that
  cannot be mistaken for a real sized eps -- the old code returned
  `FUSION_EPS_CANDIDATES[-1]` (1.0) with `sized=False`, but callers/readers checking only
  `p09_fusion_ctrl_eps` (not the paired `sized` flag) could not tell that from a genuinely
  sized 1.0. Returns `0.0` instead (never a valid eps -- `eps=0.0` is a no-op perturbation
  by construction) so an unsized search is unambiguous even from the eps value alone.
  """
  lo, hi = las.SIZING_RATIO_RANGE
  tried = []
  for eps in las.FUSION_EPS_CANDIDATES:
    unit = f"fusioneval_eps{eps}"

    def _compute(eps: float = eps) -> dict[str, Any]:
      control = las._fusion_sized_control(jax_model, batch, decoding_order, eps=eps)  # noqa: SLF001
      return {"ratio_to_bar": control["ratio_to_bar"]}

    if store is not None:
      value = store.get_or_compute(
        unit,
        {"eps": eps, "fixture": fixture_name, "sizing_ratio_range": list(las.SIZING_RATIO_RANGE)},
        _compute,
      )
    else:
      value = _compute()
    ratio_to_bar = value["ratio_to_bar"]
    tried.append({"eps": eps, "ratio_to_bar": ratio_to_bar})
    elapsed = (time.monotonic() - run_start) if run_start is not None else -1.0
    logger.info(
      "fusion eps candidate eps=%s ratio_to_bar=%.6g (cumulative %.1fs)",
      eps,
      ratio_to_bar,
      elapsed,
    )
    if lo <= ratio_to_bar <= hi:
      return eps, True
  logger.warning(
    "no eps in %r sized the fusion control into [%sx, %sx] the bar (tried: %r)",
    las.FUSION_EPS_CANDIDATES,
    lo,
    hi,
    tried,
  )
  return 0.0, False


def _project_validate_draw_counts(
  n_required: int,
  n_lanes: int,
  n_control_replicates: int = las.N_CONTROL_REPLICATES,
) -> tuple[int, int]:
  """Draw totals at ``n_required`` across ``n_lanes``, matching validate.

  Every lane draws main A1+A2 and main R1+R2. Positive and negative controls
  (``8*R`` aminx draws, allocation summing to ``n_required``) are charged once
  on the control lane. ``R`` is ``N_CONTROL_REPLICATES``. The v1 per-slot factor
  ``2+8*R`` (R=20 -> 162) is not this projection.
  """
  main_aminx = len(lass.MAIN_AMINX_ARMS) * n_required * n_lanes
  control_aminx = 8 * n_control_replicates * n_required
  if lass.PROTOCOL_CONTROL_SCOPE == "per_slot":
    control_aminx *= n_lanes
  reference_draws_total = las.REFERENCE_DRAWS_PER_SLOT * n_required * n_lanes
  return main_aminx + control_aminx, reference_draws_total


FLOOR_BUDGET_FORMULA = (
  "budget_wall_hours = max over shards of the LPT-packed per-shard sums; "
  "budget_gpu_hours = sum over shards. Every selected lane prices main A1+A2 "
  "(2 aminx draws) and main R1+R2 (2 reference draws) per allocated slot. "
  "Positive and negative controls (8*N_CONTROL_REPLICATES aminx draws per slot) "
  "are priced once, on the control lane P07@1.0, matching validate.run_controls. "
  "The per-slot factor (2+8*R on every lane; R=20 -> 162) is the v1 formula and "
  "is not this budget. c_* are measured wall seconds per draw. The projection is "
  "a LOWER bound on validate's cost: n_required >= N_REQUIRED_FLOOR and cost is "
  "increasing in n. Only the selected lanes (default V2_LANES) are priced."
)


def _budget_at_floor(
  jax_model: Any,
  pt_model: Any,
  torch: Any,
  protein_fixtures_b: list[dict[str, Any]],
  data_utils_module: Any,
  *,
  lanes: tuple[str, ...] | None = None,
  n_shards: int = las.N_SHARDS,
  n_control_replicates: int = las.N_CONTROL_REPLICATES,
) -> dict[str, Any]:
  """Project validate's wall time at `n = N_REQUIRED_FLOOR` from per-draw costs measured
  per (lane, set-B fixture) (D10). Validate cost only grows with `n`, so a floor projection
  above `BUDGET_WALL_HOURS_CAP` proves the pre-registered protocol cannot fit, before any
  pilot, margin or null-replicate draw is spent.

  ``budget_wall_hours`` is the longest shard after LPT packing. ``budget_gpu_hours`` is
  the sum across shards. Only ``lanes`` (default ``V2_LANES``) are priced.
  """
  selected = las.V2_LANES if lanes is None else lanes
  if lass.PROTOCOL_CONTROL_SCOPE == "once":
    lass.require_control_lane(selected)
  n_floor = las.N_REQUIRED_FLOOR
  per_lane_hours: dict[str, float] = {}
  costs: dict[str, dict[str, dict[str, float]]] = {}
  allocations: dict[str, dict[str, int]] = {}
  for lane in selected:
    lane_model = las.full_model_bundle_for_lane(lane, "eqx")
    lane_jax, lane_pt = lane_model[0], lane_model[1]
    batches = _lane_fixture_batches(protein_fixtures_b, lane, data_utils_module)
    alloc = las.allocate_draws(batches, n_floor)
    temp = las.DEFAULT_LANE_TEMPERATURES[lane]
    hours = 0.0
    costs[lane] = {}
    include_controls = lass.PROTOCOL_CONTROL_SCOPE == "per_slot" or lane == las.CONTROL_LANE
    factor = (
      las.aminx_draws_per_slot(n_control_replicates)
      if include_controls
      else len(lass.MAIN_AMINX_ARMS)
    )
    for fixture, batch in batches:
      c_a = las.measure_aminx_draw_cost_s(lane_jax, batch, temp)
      c_r = las.measure_reference_draw_cost_s(lane_pt, torch, batch, temp, REFERENCE_COST_SAMPLE_N)
      costs[lane][fixture["name"]] = {"aminx_s": c_a, "reference_s": c_r}
      hours += alloc[fixture["name"]] * (factor * c_a + las.REFERENCE_DRAWS_PER_SLOT * c_r) / 3600.0
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
  units = lass.enumerate_work_units(
    lanes=selected,
    allocation=allocations,
    n_control_replicates=n_control_replicates,
    costs=costs,
    control_scope=lass.PROTOCOL_CONTROL_SCOPE,
    control_lane=las.CONTROL_LANE,
  )
  report = lass.shard_budget_report(units, n_shards)
  return {
    "n_floor": n_floor,
    "budget_wall_hours": report["budget_wall_hours"],
    "budget_gpu_hours": report["budget_gpu_hours"],
    "per_shard_hours": report["per_shard_hours"],
    "n_shards": n_shards,
    "n_control_replicates": n_control_replicates,
    "lanes": list(selected),
    "control_scope": lass.PROTOCOL_CONTROL_SCOPE,
    "control_lane": las.CONTROL_LANE,
    "aminx_draws_per_slot": las.aminx_draws_per_slot(n_control_replicates),
    "per_lane_hours": per_lane_hours,
    "per_draw_costs_s": costs,
    "draw_allocation_by_lane": allocations,
    "budget_formula": FLOOR_BUDGET_FORMULA,
  }


BUDGET_FORMULA = (
  "budget_wall_hours = max over shards of an LPT packing. Every selected lane "
  "(default V2_LANES) prices 2*c_aminx + 2*c_ref per allocated slot (main A1+A2 "
  "and R1+R2). Controls add 8*N_CONTROL_REPLICATES*c_aminx once, on P07@1.0, "
  "the same lane and replicate count validate's control loops use (R=20). "
  "budget_gpu_hours is the sum across shards. The v1 per-slot factor is "
  "2+8*R on every lane (R=20 -> 162) and is not this budget. "
  "per-draw costs are measured on this run's hardware; no core-count division is applied."
)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
  start = time.monotonic()
  prov = lac.provenance(None)
  selected_lanes = lass.require_control_lane(lass.parse_lanes(getattr(args, "lanes", None)))
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
    jax_model,
    fixture_batches,
    SMOKE_LANE,
    SMOKE_PILOT_N,
    "smokesigmamargin",
  )
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))

  # C1 (smoke variant): the reviewer explicitly sanctions marking the OTHER two lanes'
  # margins as an explicit placeholder (never a copy of SMOKE_LANE's real value) when
  # computing all 5 at even a small n is genuinely too expensive for a smoke gate --
  # measured during this revision: `WaveScheduleBundle.from_tie_groups`'s per-draw
  # host-side construction cost made 5-lane-at-n=6 smoke run past 9m50s before being
  # killed. `margins_smoke_placeholder` records which entries are real vs placeholder.
  margins: dict[str, float] = dict.fromkeys(selected_lanes, 0.0)
  margins[SMOKE_LANE] = smoke_lane_margin
  margins_smoke_placeholder: dict[str, bool] = {lane: lane != SMOKE_LANE for lane in selected_lanes}

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
    "beta_by_lane": dict.fromkeys(selected_lanes, las.DEFAULT_BETA),
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
    # T10g: schema fields shared with run_full's result -- smoke never checkpoints
    # (it is already a small, fast, single-pass path), so these are always the
    # unset/empty default.
    "checkpoint_dir": "",
    "checkpoint_units_resumed": [],
    **prov,
  }


def run_full(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  """Full (non-`--smoke`) calibration over `--fixture-set`. Titanix-only in practice --
  every threshold below is the PRE-REGISTERED one, never shrunk; the budget cap is the
  enforced safety net (FAILS loudly, writes no params, rather than cutting a corner)."""
  start = time.monotonic()
  prov = lac.provenance(None)
  selected_lanes = lass.require_control_lane(lass.parse_lanes(getattr(args, "lanes", None)))
  n_shards = int(getattr(args, "n_shards", las.N_SHARDS))
  if args.pilot_lane not in selected_lanes:
    msg = f"--pilot-lane {args.pilot_lane} is not in the selected lanes {selected_lanes}"
    raise SystemExit(msg)
  manifest = _load_manifest()
  protein_fixtures_a = _fixtures_for_set(manifest, args.fixture_set)
  protein_fixtures_b = _fixtures_for_set(manifest, "B")
  manifest_sha256 = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  pilot_lane = args.pilot_lane
  full_model_bundle = las.full_model_bundle_for_lane(pilot_lane, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  pilot_batches = _lane_fixture_batches(
    protein_fixtures_a,
    pilot_lane,
    data_utils_module,
    exclude=MARGIN_EXCLUDE_FIXTURES,
  )
  if not pilot_batches:
    msg = (
      f"no set-A fixtures usable for pilot lane {pilot_lane!r} "
      f"(after excluding {MARGIN_EXCLUDE_FIXTURES!r})"
    )
    raise SystemExit(msg)

  # T10g: checkpoint store, keyed on this run's identity + every constant the
  # checkpointed stages below depend on. `--checkpoint-dir` absent (checkpoint_dir=None)
  # makes every store call a pure pass-through -- behavior is unchanged from pre-T10g,
  # only the logging below is new.
  checkpoint_dir = getattr(args, "checkpoint_dir", None)
  base_fingerprint = {
    "git_hash": prov["git_hash"],
    "fixture_set": args.fixture_set,
    "smoke": False,
    "selected_lanes": list(selected_lanes),
    "pilot_lane": pilot_lane,
    "n_shards": n_shards,
    "pilot_n": PILOT_N,
    "sizing_n": SIZING_N,
    "sentinel_half_n": SENTINEL_HALF_N,
    "null_replicates": las.NULL_REPLICATES,
    "max_n_doublings": las.MAX_N_DOUBLINGS,
    "posctl_pass_floor": las.POSCTL_PASS_FLOOR,
    "beta_candidates": list(las.BETA_CANDIDATES),
    "fusion_eps_candidates": list(las.FUSION_EPS_CANDIDATES),
    "sizing_ratio_range": list(las.SIZING_RATIO_RANGE),
    "recovery_delta": las.RECOVERY_DELTA,
    "n_boot": N_BOOT,
    "margin_temperature_ratio": las.MARGIN_TEMPERATURE_RATIO,
    "margin_exclude_fixtures": list(MARGIN_EXCLUDE_FIXTURES),
  }
  store = CheckpointStore(checkpoint_dir, base_fingerprint)
  logger.info(
    "run_full starting: fixture_set=%s pilot_lane=%s lanes=%r checkpoint_dir=%s",
    args.fixture_set,
    pilot_lane,
    selected_lanes,
    checkpoint_dir,
  )

  # D10: fail fast when even the n = N_REQUIRED_FLOOR projection is over budget.
  # (Not itself checkpointed -- it already logs per (lane, fixture) as it goes, so it
  # is not the silent stretch this feature targets; see module docstring / T10g brief.)
  with _StageLog(start, "budget_floor_projection"):
    floor = _budget_at_floor(
      jax_model,
      pt_model,
      torch,
      protein_fixtures_b,
      data_utils_module,
      lanes=selected_lanes,
      n_shards=n_shards,
      n_control_replicates=las.N_CONTROL_REPLICATES,
    )
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
      "controls_total": len(selected_lanes) + 1 + 1,
      "controls_sized": 0,
      "fixture_set": args.fixture_set,
      "fixture_manifest_sha256": manifest_sha256,
      "elapsed_seconds": time.monotonic() - start,
      "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
      "checkpoint_units_resumed": list(store.units_resumed),
      **prov,
    }
    # Exit 0, not EXIT_NOT_WRITTEN: the result is complete and the sidecar's
    # budget_exceeded branch classifies it. bathos 84be544e overrides the outcome to
    # "error" on any non-zero exit (runner.py ~703), which would hide the pre-registered
    # branch from the record (observed on titanix run 1a45a859).
    return result, 0

  with _StageLog(start, "pilot_sigma"):
    sigma_hat = store.get_or_compute(
      f"pilot_{pilot_lane}",
      {
        "lane": pilot_lane,
        "pilot_n": PILOT_N,
        "seed_tag": "pilotsigma",
      },
      lambda: {"sigma_hat": _pilot_sigma(jax_model, pilot_batches, pilot_lane, PILOT_N, "pilotsigma")},
    )["sigma_hat"]
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))
  logger.info(
    "pilot done: sigma_hat=%.6g n_required=%d (cumulative %.1fs)",
    sigma_hat,
    n_required,
    time.monotonic() - start,
  )

  # C2: null-replicate criterion at the CURRENT n_required, using the pilot lane's own
  # margin (recomputed at each doubling, since it must be evaluated "at n = n_required").
  #
  # T10g: each doubling level is its own checkpoint unit (`nulldecision_d{n_doublings}`),
  # whose `compute()` calls the (also checkpointed) lane-margin + null-replicate-check
  # helpers -- so a doubling level already fully decided in a prior run is skipped
  # entirely (`resumed nulldecision_d{n}`), while a doubling level interrupted PARTWAY
  # through its 20 replicates resumes each already-computed replicate individually and
  # only spends new draws on the remainder (see `_null_replicate_check`).
  n_doublings = 0
  null_criterion_met = False
  pilot_margin = 0.0
  while True:

    def _doubling_decision(n_doublings: int = n_doublings, n_required: int = n_required) -> dict[str, Any]:
      margin, _cost, _n, _alloc = _checkpointed_lane_margin(
        store,
        jax_model,
        pilot_batches,
        pilot_lane,
        n_required,
        f"pilotmargin{n_doublings}",
      )
      null_pass = _null_replicate_check(
        jax_model,
        pilot_batches,
        pilot_lane,
        n_required,
        margin,
        f"doubling{n_doublings}",
        store=store,
        n_doublings=n_doublings,
        run_start=start,
      )
      return {
        "pilot_margin": margin,
        "null_pass": null_pass,
        "criterion_met": null_pass >= las.POSCTL_PASS_FLOOR,
      }

    with _StageLog(start, f"null_criterion_doubling_{n_doublings}"):
      decision = store.get_or_compute(
        f"nulldecision_d{n_doublings}",
        {
          "lane": pilot_lane,
          "n_doublings": n_doublings,
          "n_required": n_required,
          "posctl_pass_floor": las.POSCTL_PASS_FLOOR,
        },
        _doubling_decision,
      )
    pilot_margin = decision["pilot_margin"]
    null_pass = decision["null_pass"]
    logger.info(
      "doubling %d done: n_required=%d pilot_margin=%.6g null_pass=%d/%d (floor %d) "
      "(cumulative %.1fs)",
      n_doublings,
      n_required,
      pilot_margin,
      null_pass,
      las.NULL_REPLICATES,
      las.POSCTL_PASS_FLOOR,
      time.monotonic() - start,
    )
    if decision["criterion_met"]:
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
      "controls_total": len(selected_lanes) + 1 + 1,
      "controls_sized": 0,
      "fixture_set": args.fixture_set,
      "fixture_manifest_sha256": manifest_sha256,
      "elapsed_seconds": elapsed_seconds,
      "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
      "checkpoint_units_resumed": list(store.units_resumed),
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

  # C1: per-lane margins, at the FINAL n_required, for the remaining lanes (the pilot
  # lane's margin at this n was already computed as part of the doubling loop above).
  margins: dict[str, float] = {pilot_lane: pilot_margin}
  lane_costs: list[tuple[int, float]] = []  # (n_draws, per_draw_cost_s), for the budget projection
  pilot_n_draws = 4 * sum(las.allocate_draws(pilot_batches, n_required).values())
  # (re-time the pilot lane's own margin cost at the FINAL n once more, cleanly, rather
  # than reusing a doubling-loop timing that may have run at a smaller n)
  with _StageLog(start, f"lane_margin_final_{pilot_lane}"):
    _pilot_margin_final, pilot_cost, pilot_n_draws, _alloc = _checkpointed_lane_margin(
      store,
      jax_model,
      pilot_batches,
      pilot_lane,
      n_required,
      "pilotmarginfinal",
    )
  margins[pilot_lane] = _pilot_margin_final
  lane_costs.append((pilot_n_draws, pilot_cost))
  logger.info(
    "lane margin lane=%s (final) margin=%.6g per_draw_cost_s=%.6g (cumulative %.1fs)",
    pilot_lane,
    _pilot_margin_final,
    pilot_cost,
    time.monotonic() - start,
  )

  for lane in selected_lanes:
    if lane == pilot_lane:
      continue
    lane_batches = _lane_fixture_batches(
      protein_fixtures_a,
      lane,
      data_utils_module,
      exclude=MARGIN_EXCLUDE_FIXTURES,
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
    with _StageLog(start, f"lane_margin_{lane}"):
      margin, cost, n_draws, _alloc = _checkpointed_lane_margin(
        store,
        jax_model,
        lane_batches,
        lane,
        n_required,
        f"margin{lane}",
      )
    margins[lane] = margin
    lane_costs.append((n_draws, cost))
    logger.info(
      "lane margin lane=%s margin=%.6g per_draw_cost_s=%.6g (cumulative %.1fs)",
      lane,
      margin,
      cost,
      time.monotonic() - start,
    )

  total_n_draws = sum(n for n, _c in lane_costs)
  per_draw_cost_aminx_s = (
    sum(n * c for n, c in lane_costs) / total_n_draws if total_n_draws else 0.0
  )

  # C3: beta, per lane, EVERY lane must be attempted (sized or recorded unsized).
  beta_by_lane: dict[str, float | None] = {}
  beta_sized_count = 0
  for lane in selected_lanes:
    lane_batches = _lane_fixture_batches(
      protein_fixtures_a,
      lane,
      data_utils_module,
      exclude=MARGIN_EXCLUDE_FIXTURES,
    )
    if not lane_batches or margins.get(lane, UNCOMPUTED_SENTINEL) == UNCOMPUTED_SENTINEL:
      beta_by_lane[lane] = None
      continue
    with _StageLog(start, f"beta_search_{lane}"):
      beta_lane, sized, _tried = _search_beta_for_lane(
        jax_model,
        lane_batches,
        lane,
        margins[lane],
        f"betasearch{lane}",
        store=store,
        run_start=start,
      )
    beta_by_lane[lane] = beta_lane
    beta_sized_count += int(sized)
    logger.info(
      "lane %s: beta search done beta=%s sized=%s (cumulative %.1fs)",
      lane,
      beta_lane,
      sized,
      time.monotonic() - start,
    )
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
    ),
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
      with _StageLog(start, "fusion_eps_search"):
        p09_fusion_ctrl_eps, fusion_sized = _search_fusion_eps(
          jax_model,
          p09s_batch,
          decoding_order,
          store=store,
          fixture_name=set_a_with_groups["name"],
          run_start=start,
        )
      logger.info(
        "fusion eps search done eps=%s sized=%s (cumulative %.1fs)",
        p09_fusion_ctrl_eps,
        fusion_sized,
        time.monotonic() - start,
      )

  # C6: differential sentinel, reduced subset (pilot fixture only), SENTINEL_HALF_N per arm.
  with _StageLog(start, "differential_sentinel"):
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
  logger.info(
    "differential sentinel: off=%.6g on=%.6g measured_half_effect=%.6g (cumulative %.1fs)",
    off_metric,
    on_metric,
    measured_half_effect,
    time.monotonic() - start,
  )
  if min_effect <= 0:
    msg = (
      f"measured_half_effect={measured_half_effect!r} <= 0 (off={off_metric!r}, on={on_metric!r}) "
      "-- the sampling sentinel is not sensitive on the reduced subset"
    )
    raise SystemExit(msg)

  # C4: budget projection (VALIDATE's projected cost, never calibrate's own elapsed time).
  with _StageLog(start, "budget_projection"):
    reference_cost_batch = _lane_fixture_batches([pilot_batches[0][0]], pilot_lane, data_utils_module)
    per_draw_cost_reference_s = las.measure_reference_draw_cost_s(
      pt_model,
      torch,
      reference_cost_batch[0][1],
      las.DEFAULT_LANE_TEMPERATURES[pilot_lane],
      REFERENCE_COST_SAMPLE_N,
    )
    aminx_draws_total, reference_draws_total = _project_validate_draw_counts(
      n_required,
      len(selected_lanes),
    )
    alloc_by_lane = {
      lane: las.allocate_draws(
        _lane_fixture_batches(protein_fixtures_b, lane, data_utils_module),
        n_required,
      )
      for lane in selected_lanes
    }
    shard_report = lass.shard_budget_report(
      lass.enumerate_work_units(
        lanes=selected_lanes,
        allocation=alloc_by_lane,
        n_control_replicates=las.N_CONTROL_REPLICATES,
        costs=floor["per_draw_costs_s"],
        control_scope=lass.PROTOCOL_CONTROL_SCOPE,
        control_lane=las.CONTROL_LANE,
      ),
      n_shards,
    )
    budget_wall_hours = float(shard_report["budget_wall_hours"])
    budget_gpu_hours = float(shard_report["budget_gpu_hours"])
    per_shard_hours = shard_report["per_shard_hours"]
    projected_peak_rss_gib = _peak_rss_gib()
    within_budget = (
      budget_wall_hours <= BUDGET_WALL_HOURS_CAP
      and projected_peak_rss_gib <= PROJECTED_PEAK_RSS_GIB_CAP
    )

    allocation = las.allocate_draws(
      _lane_fixture_batches(protein_fixtures_b, "P07@1.0", data_utils_module),
      n_required,
    )
  logger.info(
    "budget projection: budget_wall_hours=%.3f budget_gpu_hours=%.3f within_budget=%s "
    "(cumulative %.1fs)",
    budget_wall_hours,
    budget_gpu_hours,
    within_budget,
    time.monotonic() - start,
  )

  controls_total = len(selected_lanes) + 1 + 1  # per-lane betas + fusion eps + null-replicate
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
    "budget_gpu_hours": budget_gpu_hours,
    "per_shard_hours": per_shard_hours,
    "n_shards": n_shards,
    "n_control_replicates": las.N_CONTROL_REPLICATES,
    "lanes": list(selected_lanes),
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
  with _StageLog(start, "params_write"):
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
    "budget_gpu_hours": budget_gpu_hours,
    "per_shard_hours": per_shard_hours,
    "n_shards": n_shards,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "n_not_advanced": 0,
    "n_skipped": 0,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "elapsed_seconds": time.monotonic() - start,
    "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
    "checkpoint_units_resumed": list(store.units_resumed),
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
    "--smoke",
    action="store_true",
    help="One fixture, one lane, small pilot, per-lane margins.",
  )
  parser.add_argument(
    "--fixture-set",
    choices=("A", "B"),
    default="A",
    help="Manifest fixture set to calibrate on.",
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
  parser.add_argument(
    "--lanes",
    default=None,
    help="Comma-separated lane keys (default: the three ProteinMPNN V2_LANES).",
  )
  parser.add_argument(
    "--n-shards",
    type=int,
    default=las.N_SHARDS,
    help="Shard count for the budget wall-clock projection (default N_SHARDS).",
  )
  parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=None,
    help=(
      "T10g: optional dir for run_full's per-unit checkpoints (pilot, null replicates, "
      "lane margins, beta/fusion-eps searches). Absent (default): behavior is unchanged "
      "except for the new progress logging. A relaunch at the SAME commit/config with the "
      "SAME --checkpoint-dir resumes already-completed units instead of recomputing them, "
      "bit-identically (see layer_a_sampling._seed_for's T10g fix). Ignored by --smoke."
    ),
  )
  args = parser.parse_args(argv)
  lass.ensure_xla_preallocate_false()

  if args.smoke and args.params_out == DEFAULT_PARAMS_OUT:
    parser.error(
      "--smoke requires a non-default --params-out (a scratch file "
      "layer_a_sampling_validate.py never reads)",
    )

  if args.smoke:
    result = run_smoke(args)
    exit_code = 0
  else:
    result, exit_code = run_full(args)
  result.update(lass.device_record())

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
