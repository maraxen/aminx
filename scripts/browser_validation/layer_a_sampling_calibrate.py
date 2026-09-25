"""Layer (a) sampling-tier calibration (T8b, step 4, fixture set A).

Bathos staging's "calibration" twin for `layer_a_sampling.py`'s engine -- mirrors
`layer_a_exact_calibrate.py` (T7b)'s structure and Calibration-twin outcomes (R2-C10)
exactly, but computes the `sampling` section of `preregistered_params.json` instead of
`exact`:

- **sigma_hat / n_required.** A pilot (set A, `n = PILOT_N`) of aminx draws at one lane's
  temperature `T` vs `1.05*T` gives per-sequence recovery; `sigma_hat` is its SD.
  `n_required = max(1500, required_n(sigma_hat, 0.01))` (`aminx.parity.compare.required_n`,
  reused). The spec's "doubles n until 20 aminx-vs-aminx null replicates declare
  equivalence in >= 18/20" loop is genuinely expensive (each doubling re-runs 20 null
  replicates at the NEW `n`) -- bounded here at `MAX_N_DOUBLINGS` iterations so a
  pathological sigma cannot loop indefinitely; the full, unbounded protocol is a
  titanix-only cost (see the "Titanix budget" cap below).
- **Margin m_l per lane.** `mean(excess_js(...))` between aminx@T and aminx@1.05*T (both
  arms aminx -- no reference draws needed for this quantity), computed for the SAME
  lane pilot draws that produced `sigma_hat`. Recorded under `sampling.margins`, keyed
  by `LANE_KEYS` (length 5).
- **Beta / fusion-eps sizing.** `run_full` searches `BETA_CANDIDATES` /
  `FUSION_EPS_CANDIDATES` for the smallest candidate landing the respective sized
  control in `SIZING_RATIO_RANGE`; `--smoke` records `DEFAULT_BETA` /
  `DEFAULT_FUSION_CTRL_EPS` directly (no search), mirroring
  `layer_a_exact_calibrate.py`'s own smoke precedent (`DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE`,
  never searched in its smoke path either).
- **P09-s non-vacuity floor.** `sampling.p09_qualifying_groups` / `sampling.p09_tied_positions`
  describe fixture SET B (the held-out set P09-s and validate will actually run on), NOT
  set A -- counted directly from the fixture manifest (no model call needed), always,
  smoke included.
- **Budget.** `sampling.budget_wall_hours` / `sampling.projected_peak_rss_gib`, exactly
  like T7b; a FULL run fails (writing a diagnostic result but no `--params-out`) above
  the 16h/48GiB caps.

**`--smoke`** runs a small pilot (`SMOKE_PILOT_N` draws at `T` and `1.05*T`) on ONE
small fixture (`SMOKE_FIXTURE_NAME`, set A) at ONE lane (`SMOKE_LANE`) -- never the
full 5-lane x 2n x controls protocol -- and applies the SAME pilot-derived
sigma_hat/margin to every lane key (documented simplification: a representative,
schema-valid smoke section, not a per-lane measurement; mirrors
`layer_a_exact_calibrate.py --smoke`'s own single-row-only scope). Requires a
non-default scratch `--params-out`, exactly like T7.
"""

from __future__ import annotations

import argparse
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
SMOKE_PILOT_N = 6
SMOKE_FIXTURE_NAME = "5L33"
SMOKE_LANE = "P07@1.0"
BUDGET_WALL_HOURS_CAP = 16.0
PROJECTED_PEAK_RSS_GIB_CAP = 48.0
MIN_MEASURED_HALF_EFFECT = 0.0  # main_js_vs_ref's own half-effect must be > 0 (asserted below)

EXIT_SKIPPED = 4


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


def _pilot_recovery_and_margin(
  jax_model: Any, batch_t: Any, batch_t105: Any, n: int, seed_base: int
) -> tuple[np.ndarray, np.ndarray, float]:
  """Draw `n` aminx sequences at `T` and `1.05*T`, return `(recovery_t, recovery_t105, margin)`.

  `margin` is `mean(excess_js(a1, a2, r1, r2))` over a single split of the `n` draws into
  two halves per temperature (a1/a2 from `T`, r1/r2 from `1.05*T`) -- the spec's "mean E
  between aminx at T and aminx at 1.05*T", reusing `aminx.parity.compare.excess_js`
  directly rather than re-deriving the JS math.
  """
  from aminx.parity.compare import excess_js

  seqs_t = las.aminx_sample_batch(jax_model, batch_t, n, seed_base, temperature=1.0)
  seqs_t105 = las.aminx_sample_batch(jax_model, batch_t105, n, seed_base + 100_000, temperature=1.0)

  seq_ref_restricted = batch_t.seq_ref[batch_t.comparison_positions]
  recovery_t = las.per_sequence_recovery(seqs_t, seq_ref_restricted)
  recovery_t105 = las.per_sequence_recovery(seqs_t105, seq_ref_restricted)

  half = max(1, n // 2)
  a1, a2 = seqs_t[:half], seqs_t[half : 2 * half] if n >= 2 * half else seqs_t[:half]
  r1, r2 = seqs_t105[:half], seqs_t105[half : 2 * half] if n >= 2 * half else seqs_t105[:half]
  margin = float(np.atleast_1d(excess_js(a1, a2, r1, r2)).mean())
  return recovery_t, recovery_t105, margin


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures = _fixtures_for_set(manifest, args.fixture_set)
  fixture = _pick_fixture(protein_fixtures, SMOKE_FIXTURE_NAME)

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  full_model_bundle = las.full_model_bundle_for_lane(SMOKE_LANE, "eqx")
  jax_model = full_model_bundle[0]

  batch_t = las.build_lane_batch(fixture, SMOKE_LANE, data_utils_module)
  # 1.05*T pilot batch shares the same geometry/bias/chain_mask; only the sampling
  # temperature passed to `aminx_sample_batch` differs (T vs 1.05*T are both driven
  # through the SAME LaneBatch below, matching the margin rule's "same lane" contract).
  recovery_t, recovery_t105, margin = _pilot_recovery_and_margin(
    jax_model, batch_t, batch_t, SMOKE_PILOT_N, las._seed_for(fixture["name"] + SMOKE_LANE)
  )
  sigma_hat = float(np.std(np.concatenate([recovery_t, recovery_t105]), ddof=1))
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))

  p09_qualifying_groups, p09_tied_positions = _p09_set_b_counts(manifest)

  elapsed_seconds = time.monotonic() - start
  margins = {lane: margin for lane in las.LANE_KEYS}

  sampling_section = {
    "git_hash": prov["git_hash"],
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "margins": margins,
    "beta": las.DEFAULT_BETA,
    "reduced_subset": [fixture["name"]],
    "reduced_subset_size": 1,
    "min_effect": max(margin / 2.0, sys.float_info.min),
    "measured_half_effect": margin / 2.0,
    "lane_temperatures": dict(las.DEFAULT_LANE_TEMPERATURES),
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": las.DEFAULT_FUSION_CTRL_EPS,
    "draw_allocation": {fixture["name"]: SMOKE_PILOT_N},
    "budget_wall_hours": max(elapsed_seconds / 3600.0, sys.float_info.min),
    "projected_peak_rss_gib": _peak_rss_gib(),
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


def _merge_params(params_out: Path, sampling_section: dict[str, Any]) -> dict[str, Any]:
  """Merge `sampling_section` into any already-committed sections (T7's `exact`, kept
  byte-identical in canonical form -- only the `sampling` key is replaced)."""
  params: dict[str, Any] = {}
  if params_out.is_file():
    with params_out.open() as fh:
      params = json.load(fh)
  params["sampling"] = sampling_section
  return params


def _search_beta(
  jax_model: Any, pt_model: Any, torch: Any, batch: Any, margin: float, seed_base: int
) -> tuple[float, float]:
  """Smallest `BETA_CANDIDATES` value landing `mean(excess_js(aminx+beta, aminx))` into
  `SIZING_RATIO_RANGE * margin`, using a small (n=20) sizing sample per candidate (the full
  18/20-replicate protocol runs later, at `n_required`, as the positive control itself)."""
  from aminx.parity.compare import excess_js

  del pt_model, torch
  lo, hi = las.SIZING_RATIO_RANGE
  sizing_n = 20
  off = las.aminx_sample_batch(jax_model, batch, sizing_n, seed_base, temperature=1.0)
  tried = []
  for beta in las.BETA_CANDIDATES:
    on = las.aminx_sample_batch(
      jax_model, batch, sizing_n, seed_base + 500_000, temperature=1.0, beta_alanine=beta
    )
    half = sizing_n // 2
    effect = float(np.atleast_1d(excess_js(on[:half], on[half:], off[:half], off[half:])).mean())
    ratio = effect / margin if margin > 0 else float("inf")
    tried.append({"beta": beta, "ratio_to_margin": ratio})
    if lo <= ratio <= hi:
      return beta, effect
  msg = (
    f"no beta in {las.BETA_CANDIDATES!r} sized the positive control into [{lo}x, {hi}x] "
    f"the margin (tried: {tried!r})"
  )
  raise SystemExit(msg)


def _search_fusion_eps(jax_model: Any, batch: Any, decoding_order: np.ndarray) -> float:
  lo, hi = las.SIZING_RATIO_RANGE
  tried = []
  for eps in las.FUSION_EPS_CANDIDATES:
    control = las._fusion_sized_control(jax_model, batch, decoding_order, eps=eps)  # noqa: SLF001
    tried.append({"eps": eps, "ratio_to_bar": control["ratio_to_bar"]})
    if lo <= control["ratio_to_bar"] <= hi:
      return eps
  msg = (
    f"no eps in {las.FUSION_EPS_CANDIDATES!r} sized the fusion control into [{lo}x, {hi}x] "
    f"the bar (tried: {tried!r})"
  )
  raise SystemExit(msg)


def run_full(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  """Full (non-`--smoke`) calibration over `--fixture-set`. Titanix-only in practice (the
  full protocol's cost -- pilot + per-lane margins + beta/eps search + 20+20 null-replicate
  doubling loop -- comfortably exceeds what this task's own Gate may run locally); the
  budget cap below is the enforced safety net, matching T7b's own pattern exactly."""
  start = time.monotonic()
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures = _fixtures_for_set(manifest, args.fixture_set)
  manifest_sha256 = __import__("hashlib").sha256(_MANIFEST_PATH.read_bytes()).hexdigest()

  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  pilot_lane = args.pilot_lane
  full_model_bundle = las.full_model_bundle_for_lane(pilot_lane, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle
  pilot_fixture = _pick_fixture(protein_fixtures, args.pilot_fixture or SMOKE_FIXTURE_NAME)
  pilot_batch = las.build_lane_batch(pilot_fixture, pilot_lane, data_utils_module)

  recovery_t, recovery_t105, margin = _pilot_recovery_and_margin(
    jax_model, pilot_batch, pilot_batch, PILOT_N, las._seed_for(pilot_fixture["name"] + pilot_lane)
  )
  sigma_hat = float(np.std(np.concatenate([recovery_t, recovery_t105]), ddof=1))
  from aminx.parity.compare import required_n

  n_required = max(las.N_REQUIRED_FLOOR, required_n(max(sigma_hat, 1e-6), las.RECOVERY_DELTA))

  # Bounded doubling: the null-replicate criterion (>= 18/20 aminx-vs-aminx equivalence at
  # the CURRENT n) is checked with a cheap n-scaled proxy (recovery-only TOST, not the full
  # excess-JS bootstrap) up to MAX_N_DOUBLINGS times; see module docstring.
  from aminx.parity.compare import tost_mean_diff

  n_doublings = 0
  while n_doublings < las.MAX_N_DOUBLINGS:
    null_pass = 0
    for replicate in range(las.NULL_REPLICATES):
      seeds = las._seed_for(f"{pilot_fixture['name']}{pilot_lane}null{replicate}")
      a = las.aminx_sample_batch(
        jax_model, pilot_batch, min(n_required, PILOT_N), seeds, temperature=1.0
      )
      b = las.aminx_sample_batch(
        jax_model, pilot_batch, min(n_required, PILOT_N), seeds + 1, temperature=1.0
      )
      seq_ref_restricted = pilot_batch.seq_ref[pilot_batch.comparison_positions]
      rec_a = las.per_sequence_recovery(a, seq_ref_restricted)
      rec_b = las.per_sequence_recovery(b, seq_ref_restricted)
      passed, _pl, _pu = tost_mean_diff(rec_a, rec_b, las.RECOVERY_DELTA)
      null_pass += int(passed)
    if null_pass >= las.NULL_REPLICATES_PASS_FLOOR:
      break
    n_required *= 2
    n_doublings += 1

  margins = {lane: margin for lane in las.LANE_KEYS}

  beta, _beta_effect = _search_beta(
    jax_model, pt_model, torch, pilot_batch, margin, las._seed_for(pilot_fixture["name"] + "beta")
  )

  p09_qualifying_groups, p09_tied_positions = _p09_set_b_counts(manifest)
  p09_fusion_ctrl_eps = las.DEFAULT_FUSION_CTRL_EPS
  set_a_with_groups = next(
    (f for f in protein_fixtures if las.lae._qualifying_groups(f)),
    None,  # noqa: SLF001
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
      p09_fusion_ctrl_eps = _search_fusion_eps(jax_model, p09s_batch, decoding_order)

  # Differential sentinel: reduced subset (smallest-L, single fixture), main_js_vs_ref off/on.
  reduced_subset = [pilot_fixture]
  from aminx.parity.compare import excess_js

  off_a = las.aminx_sample_batch(
    jax_model, pilot_batch, 20, las._seed_for("sentinel_off_a"), temperature=1.0
  )
  off_r = las.reference_sample_batch(
    pt_model, torch, pilot_batch, 20, las._seed_for("sentinel_off_r"), temperature=1.0
  )
  on_a = las.aminx_sample_batch(
    jax_model, pilot_batch, 20, las._seed_for("sentinel_on_a"), temperature=1.0, beta_alanine=beta
  )
  half = 10
  off_metric = float(
    np.atleast_1d(excess_js(off_a[:half], off_a[half:], off_r[:half], off_r[half:])).mean()
  )
  on_metric = float(
    np.atleast_1d(excess_js(on_a[:half], on_a[half:], off_r[:half], off_r[half:])).mean()
  )
  measured_half_effect = abs(on_metric - off_metric) / 2.0
  min_effect = measured_half_effect
  if min_effect <= 0:
    msg = (
      f"measured_half_effect={measured_half_effect!r} <= 0 (off={off_metric!r}, on={on_metric!r}) "
      "-- the sampling sentinel is not sensitive on the reduced subset"
    )
    raise SystemExit(msg)

  elapsed_seconds = time.monotonic() - start
  budget_wall_hours = elapsed_seconds / 3600.0
  projected_peak_rss_gib = _peak_rss_gib()
  within_budget = (
    budget_wall_hours <= BUDGET_WALL_HOURS_CAP
    and projected_peak_rss_gib <= PROJECTED_PEAK_RSS_GIB_CAP
  )

  allocation = las.allocate_draws(
    [
      (f, las.build_lane_batch(f, "P07@1.0", data_utils_module))
      for f in _fixtures_for_set(manifest, "B")
    ],
    n_required,
  )

  sampling_section = {
    "git_hash": prov["git_hash"],
    "sigma_hat": sigma_hat,
    "n_required": n_required,
    "margins": margins,
    "beta": beta,
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
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "n_doublings": n_doublings,
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
    "margins": margins,
    "beta": beta,
    "p09_qualifying_groups": p09_qualifying_groups,
    "p09_tied_positions": p09_tied_positions,
    "p09_fusion_ctrl_eps": p09_fusion_ctrl_eps,
    "min_effect": min_effect,
    "measured_half_effect": measured_half_effect,
    "params_written": params_written,
    "params_section_sha256": params_section_sha256,
    "controls_total": 2,  # positive + fusion sizing, both sized above
    "controls_sized": 2,
    "n_not_advanced": 0,
    "n_skipped": 0,
    "fixture_set": args.fixture_set,
    "fixture_manifest_sha256": manifest_sha256,
    "elapsed_seconds": elapsed_seconds,
    **prov,
  }
  exit_code = 0
  if not within_budget:
    exit_code = 1
  elif not prov["git_clean"]:
    exit_code = EXIT_SKIPPED
  return result, exit_code


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--smoke", action="store_true", help="One fixture, one lane, small pilot, < ~60 s."
  )
  parser.add_argument(
    "--fixture-set", choices=("A", "B"), default="A", help="Manifest fixture set to calibrate on."
  )
  parser.add_argument(
    "--pilot-lane",
    choices=las.LANE_KEYS,
    default="P07@1.0",
    help="Lane used for the sigma/margin pilot.",
  )
  parser.add_argument(
    "--pilot-fixture", default=None, help="Fixture name for the pilot (default: smallest)."
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
