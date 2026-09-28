r"""Layer (a) sampling-tier budget-floor probe with the FIXED batched harness (T9, step 3).

Companion to `layer_a_sampling_calibrate.py`: reuses that module's `_budget_at_floor`
(the validate-cost projection at `n = N_REQUIRED_FLOOR`, V21 caps) UNMODIFIED -- no
copy, no hard-coded caps -- and adds the controls this task's F-S1 fix + `incremental`
plumbing (`sample_autoregressive.kernel`, D-F) exist to make measurable:

- **`fastpath_ratio = c_force / c_off`** (guarded): per-draw aminx cost with the
  incremental fastpath forced ON vs forced OFF, at the largest set-B fixture
  (dynamically resolved from the manifest; V9 -- 4YOW, L=693) on lane `P07@1.0`. Under
  plain `vmap`, `"auto"` alone cannot show this (V16: the device predicate becomes
  batched and `lax.cond` runs both branches) -- this is why `incremental` had to reach
  the kernel at all, and `aminx_sample_batch_at_incremental`/`measure_aminx_draw_cost_
  at_incremental_s` (`layer_a_sampling.py`) route every draw in a chunk through the
  SAME, caller-chosen mode so the two costs are a real, homogeneous A/B.
- **`aa_ratio`** (guarded): force-vs-force, two independently-seeded batches at the
  same fixture/lane -- a sanity control (BATHOS.md) that the cost MEASUREMENT itself
  is stable, not a finding. If this drifts outside [0.8, 1.25] (the sidecar's
  `ctrl_blind` outcome), `fastpath_ratio` cannot be trusted either.
- **`n_draws_forced_off` per lane**: over the SAME `draw_allocation_by_lane` `_budget_
  at_floor` already computed (not re-derived), for each (lane, fixture) the number of
  its allocated draws whose own wave fails `layer_a_sampling.incremental_predicates_
  host`'s three-way AND (D-F) -- i.e. how many draws the FIXED harness's own
  auto-dispatch would have to fall back to `"off"` for, purely diagnostic (never
  gates the outcome).

Never run without `$BATHOSW` locally (needs `bathos.git`/`bathos.sidecar` for
`layer_a_common.provenance`) -- `titanix_launch.sh`/`titanix_run.sh` already carry that
overlay; this script never invokes bathos itself beyond what `layer_a_common` does.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_a_exact as lae  # noqa: E402
import layer_a_sampling as las  # noqa: E402
import layer_a_sampling_calibrate as lasc  # noqa: E402
import layer_a_sampling_shard as lass

_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
_MANIFEST_PATH = _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"

CONTROL_LANE = las.CONTROL_LANE  # spec T9 step 3: controls at lane P07@1.0
AA_CONTROL_SEED_BASE_1 = 5_101_000
AA_CONTROL_SEED_BASE_2 = 5_202_000
FASTPATH_CONTROL_SEED_BASE = 5_303_000
REFERENCE_COST_SAMPLE_N = lasc.REFERENCE_COST_SAMPLE_N

# T9 remediation (step 3): local CPU smoke support. `--smoke` restricts every reused
# helper below (`_budget_at_floor`, `_lane_fixture_batches`, `_largest_fixture`) to ONE
# small set-B fixture by shrinking the `protein_fixtures_b` list THEY are called with --
# never by editing those functions themselves (both stay reused unmodified, per this
# module's own docstring). "1BC8" is the only set-B protein fixture resolvable with just
# $REFERENCE_PATH (no $PROTEINMPNN_PATH clone needed), so it is the one every dev box
# with the LigandMPNN reference clone can smoke against.
SMOKE_FIXTURE_NAME = "1BC8"
# n_floor's real per-fixture allocation is computed at N_REQUIRED_FLOOR=1500 (unmodified,
# per the module docstring); with a single smoke fixture it would receive the FULL 1500,
# and _n_draws_forced_off would host-loop that many wave/predicate computations per lane
# purely to prove "does not raise". SMOKE_MAX_DRAWS_PER_FIXTURE caps that loop under
# --smoke ONLY (never a real run) -- it does not touch `floor`/`draw_allocation_by_lane`
# itself, so the recorded allocation and budget projection are unaffected; only the
# smoke-only diagnostic draw count they feed is capped.
SMOKE_MAX_DRAWS_PER_FIXTURE = 5


def _load_manifest() -> dict[str, Any]:
  with _MANIFEST_PATH.open() as fh:
    return json.load(fh)


def _largest_fixture(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
  return max(fixtures, key=lambda f: f.get("L", 0))


def _n_draws_forced_off(
  jax_model: Any,
  batches: list[tuple[dict[str, Any], las.LaneBatch]],
  lane: str,
  allocation: dict[str, int],
  *,
  max_draws: int | None = None,
) -> int:
  """Count how many of `allocation`'s allocated draws, over `batches`, fail the
  three-way host predicate AND (D-F) -- purely diagnostic (spec "n_draws_forced_off
  per lane"). One deterministic encode per fixture (noise 0, `mode="sample"`'s own
  bundle construction re-derives the SAME neighbour geometry the kernel itself would
  use for a real draw -- V4/V19: encode does not depend on the sampled sequence).

  `max_draws`, when given, caps how many of `allocation`'s draws are actually
  inspected per fixture (smoke acceleration ONLY -- `None`, the default, inspects every
  allocated draw exactly as a real run does). It never changes `allocation` itself.
  """
  import jax

  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.encode import make_encode_fn
  from aminx.utils.autoregression import generate_wave_ar_mask

  n_off = 0
  for fixture, batch in batches:
    k = allocation.get(fixture["name"], 0)
    if k <= 0:
      continue
    if max_draws is not None:
      k = min(k, max_draws)
    kw: dict[str, Any] = {
      "coords": batch.x4,
      "mask": batch.mask,
      "residue_index": batch.residue_index,
      "chain_index": batch.chain_index,
      "chain_mask": batch.chain_mask,
      "bias": batch.bias,
      "fixed_mask": batch.fixed_mask,
      "fixed_tokens": batch.seq_ref,
      "tie_group_map": batch.tie_group_map,
      "temperature": las.DEFAULT_LANE_TEMPERATURES[lane],
      "mode": "sample",
    }
    # T9 remediation: reuse the SAME per-lane bundle-kwargs helper `aminx_sample_one`/
    # `aminx_sample_batch` use, rather than re-deriving lane inputs here -- this is what
    # was missing (P11-s's atom_37/atom_37_mask/ligand_* kwargs), causing the ValueError.
    kw.update(las.side_chain_context_kwargs(batch))
    seed_base = las._seed_for(f"{fixture['name']}{lane}forcedoff")  # noqa: SLF001
    waves = [
      las.wave_from_tie_groups_np(
        batch.tie_group_map,
        las._draw_order_for(batch, seed_base + i)[1],  # noqa: SLF001
      )
      for i in range(k)
    ]
    bundle, config = build_inference_bundle(wave=waves[0], **kw)
    enc = make_encode_fn(jax_model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)
    neighbor_indices = np.asarray(enc.neighbor_indices[0])
    valid_mask = np.asarray(enc.mask[0])
    state_position_map = np.asarray(bundle.conditioning.state_position_map[0])
    tie_group_map_row = np.asarray(bundle.conditioning.tie_group_map[0])
    for wave in waves:
      ar_mask_2d = np.asarray(generate_wave_ar_mask(wave, bundle.conditioning.tie_group_map[0]))
      verdict = las.incremental_predicates_host(
        wave=wave,
        tie_group_map=tie_group_map_row,
        ar_mask=ar_mask_2d,
        neighbor_indices=neighbor_indices,
        valid_mask=valid_mask,
        state_position_map=state_position_map,
      )
      if not (verdict["consistent"] and verdict["identity_frame"] and verdict["fits_slab"]):
        n_off += 1
  return n_off


def _device_kind() -> str:
  import jax

  return jax.devices()[0].platform


def run(args: argparse.Namespace) -> dict[str, Any]:
  prov = lac.provenance(None)
  manifest = _load_manifest()
  protein_fixtures_b = [
    f for f in manifest["fixtures"] if f.get("set") == "B" and f.get("kind") == "protein"
  ]
  smoke = bool(getattr(args, "smoke", False))
  if smoke:
    # T9 remediation (step 1): restrict to the one smoke fixture by shrinking the LIST
    # every reused helper below is called with -- never by editing those functions
    # (`_budget_at_floor`/`_lane_fixture_batches`/`_largest_fixture` stay reused
    # unmodified, per this module's own docstring).
    protein_fixtures_b = [f for f in protein_fixtures_b if f.get("name") == SMOKE_FIXTURE_NAME]
    if not protein_fixtures_b:
      msg = f"--smoke fixture {SMOKE_FIXTURE_NAME!r} not found in set-B protein fixtures"
      raise ValueError(msg)
  max_draws = SMOKE_MAX_DRAWS_PER_FIXTURE if smoke else None
  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
  full_model_bundle = las.full_model_bundle_for_lane(CONTROL_LANE, "eqx")
  jax_model, pt_model, torch, _model_utils = full_model_bundle

  # --- floor budget projection: layer_a_sampling_calibrate._budget_at_floor, reused ---
  selected_lanes = lass.require_control_lane(lass.parse_lanes(getattr(args, "lanes", None)))
  n_shards = int(getattr(args, "n_shards", las.N_SHARDS))
  floor = lasc._budget_at_floor(  # noqa: SLF001 -- reuse, not reimplement (spec T9 step 3)
    jax_model,
    pt_model,
    torch,
    protein_fixtures_b,
    data_utils_module,
    lanes=selected_lanes,
    n_shards=n_shards,
    n_control_replicates=las.N_CONTROL_REPLICATES,
  )
  projected_peak_rss_gib = lasc._peak_rss_gib()  # noqa: SLF001

  # --- n_draws_forced_off per lane, over the SAME floor allocation ---
  n_draws_forced_off: dict[str, int] = {}
  for lane in selected_lanes:
    lane_model = las.full_model_bundle_for_lane(lane, "eqx")
    lane_jax = lane_model[0]
    batches = lasc._lane_fixture_batches(  # noqa: SLF001 -- reuse, not reimplement
      protein_fixtures_b,
      lane,
      data_utils_module,
    )
    n_draws_forced_off[lane] = _n_draws_forced_off(
      lane_jax,
      batches,
      lane,
      floor["draw_allocation_by_lane"].get(lane, {}),
      max_draws=max_draws,
    )

  # --- controls at the largest set-B fixture, CONTROL_LANE ---
  largest = _largest_fixture(protein_fixtures_b)
  control_batch = las.build_lane_batch(largest, CONTROL_LANE, data_utils_module)
  temp = las.DEFAULT_LANE_TEMPERATURES[CONTROL_LANE]

  c_force_1 = las.measure_aminx_draw_cost_at_incremental_s(
    jax_model,
    control_batch,
    temp,
    incremental="force",
    seed_base=AA_CONTROL_SEED_BASE_1,
  )
  c_force_2 = las.measure_aminx_draw_cost_at_incremental_s(
    jax_model,
    control_batch,
    temp,
    incremental="force",
    seed_base=AA_CONTROL_SEED_BASE_2,
  )
  c_off = las.measure_aminx_draw_cost_at_incremental_s(
    jax_model,
    control_batch,
    temp,
    incremental="off",
    seed_base=FASTPATH_CONTROL_SEED_BASE,
  )
  c_ref = las.measure_reference_draw_cost_s(
    pt_model,
    torch,
    control_batch,
    temp,
    REFERENCE_COST_SAMPLE_N,
  )

  aa_ratio = c_force_1 / c_force_2 if c_force_2 > 0 else las.UNCOMPUTED_SENTINEL
  fastpath_ratio = c_force_1 / c_off if c_off > 0 else las.UNCOMPUTED_SENTINEL
  ref_arm_cost = 2.0 * c_ref  # spec: "ref_arm_cost = 2*c_ref per lane"

  result: dict[str, Any] = {
    "git_hash": prov["git_hash"],
    "git_clean": prov["git_clean"],
    "n_floor": floor["n_floor"],
    "budget_wall_hours": floor["budget_wall_hours"],
    "budget_gpu_hours": floor["budget_gpu_hours"],
    "per_shard_hours": floor["per_shard_hours"],
    "n_shards": floor["n_shards"],
    "n_control_replicates": floor["n_control_replicates"],
    "lanes": floor["lanes"],
    "budget_formula": floor["budget_formula"],
    "per_lane_hours": floor["per_lane_hours"],
    "per_draw_costs_s": floor["per_draw_costs_s"],
    "draw_allocation_by_lane": floor["draw_allocation_by_lane"],
    "n_draws_forced_off": n_draws_forced_off,
    "budget_cap_hours": lasc.BUDGET_WALL_HOURS_CAP,
    "rss_cap_gib": lasc.PROJECTED_PEAK_RSS_GIB_CAP,
    "projected_peak_rss_gib": projected_peak_rss_gib,
    "control_fixture": largest["name"],
    "control_lane": CONTROL_LANE,
    "c_force": c_force_1,
    "c_force_replicate": c_force_2,
    "c_off": c_off,
    "c_ref": c_ref,
    "ref_arm_cost": ref_arm_cost,
    "budget_includes_ref": False,
    "aa_ratio": aa_ratio,
    "fastpath_ratio": fastpath_ratio,
    "device_kind": lass.device_record()["device_kind"],
    "visible_devices": lass.device_record()["visible_devices"],
    "smoke": smoke,
    **prov,
  }
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--lanes",
    default=None,
    help="Comma-separated lane keys (default: the four ProteinMPNN V2_LANES).",
  )
  parser.add_argument(
    "--n-shards",
    type=int,
    default=las.N_SHARDS,
    help="Shard count for the budget wall-clock projection (default N_SHARDS).",
  )
  parser.add_argument(
    "--smoke",
    action="store_true",
    help=(
      "Local CPU smoke mode: restrict to the single SMOKE_FIXTURE_NAME set-B fixture and "
      "cap the n_draws_forced_off diagnostic loop at SMOKE_MAX_DRAWS_PER_FIXTURE draws per "
      "lane. Never bathos-tracked; a real run (no --smoke) is unchanged."
    ),
  )
  args = parser.parse_args(argv)
  lass.ensure_xla_preallocate_false()

  result = run(args)
  lac.emit(result, args.out)
  logger.info(
    "layer_a_sampling_budget_floor: n_floor=%s budget_wall_hours=%.3f (cap %.1f) "
    "projected_peak_rss_gib=%.3f (cap %.1f) fastpath_ratio=%.4f aa_ratio=%.4f device=%s",
    result.get("n_floor"),
    result.get("budget_wall_hours", -1.0),
    lasc.BUDGET_WALL_HOURS_CAP,
    result.get("projected_peak_rss_gib", -1.0),
    lasc.PROJECTED_PEAK_RSS_GIB_CAP,
    result.get("fastpath_ratio", -1.0),
    result.get("aa_ratio", -1.0),
    result.get("device_kind"),
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
