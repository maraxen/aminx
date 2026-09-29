"""v2 sampling protocol: replicate factor, lane scope, LPT shards, merge equivalence.

CPU only. Draw values come from a tiny key-addressed sampler, not a structure model.
"""

# ruff: noqa: S101

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scripts.browser_validation.layer_a_sampling as las
import scripts.browser_validation.layer_a_sampling_merge as merge
import scripts.browser_validation.layer_a_sampling_shard as shard
import scripts.browser_validation.layer_a_sampling_validate as lasv

_PARAMS = {"sampling": {"n_required": 4}}
_GIT = "abc123"


def test_control_replicate_factor_v1_and_v2() -> None:
  assert las.N_CONTROL_REPLICATES == 20
  assert las.POSCTL_PASS_FLOOR == 18
  assert las.NEGCTL_FP_CEIL == 3
  assert las.NULL_REPLICATES_PASS_FLOOR == las.POSCTL_PASS_FLOOR
  assert las.aminx_draws_per_slot(20) == 162
  assert las.aminx_draws_per_slot(5) == 42
  assert las.aminx_draws_per_slot() == 162
  assert las.REFERENCE_DRAWS_PER_SLOT == 2


def test_v2_lanes_budget_excludes_p11() -> None:
  # T10e amendment (run e091a33e, ctrl_unsized): P07@0.1 DROPPED, not merely deferred
  # like P11-s -- see layer_a_sampling.V2_LANES's own docstring for the two defects.
  assert las.V2_LANES == ("P07@1.0", "P08@1.0", "P09-s@1.0")
  assert "P11-s@1.0" not in las.V2_LANES
  assert "P07@0.1" not in las.V2_LANES
  assert "P07@0.1" not in las.LANE_KEYS
  costs = {lane: {"fx": {"aminx_s": 3600.0, "reference_s": 0.0}} for lane in las.LANE_KEYS}
  allocation = {lane: {"fx": 1} for lane in las.LANE_KEYS}
  units = shard.enumerate_work_units(
    lanes=las.V2_LANES,
    allocation=allocation,
    costs=costs,
  )
  assert units
  assert all(unit.lane != "P11-s@1.0" for unit in units)
  report = shard.shard_budget_report(units, n_shards=2)
  per_shard = report["per_shard_hours"]
  assert isinstance(per_shard, list)
  # Once-pricing: 162 aminx hours on the control lane, 2 on each other lane.
  other_lanes = len(las.V2_LANES) - 1
  assert report["budget_gpu_hours"] == pytest.approx(162.0 + 2.0 * other_lanes)
  assert report["budget_wall_hours"] == pytest.approx(max(per_shard))
  assert report["budget_gpu_hours"] == pytest.approx(sum(per_shard))
  one = shard.enumerate_work_units(
    lanes=("P07@1.0",),
    allocation={"P07@1.0": {"fx": 1}},
    n_control_replicates=20,
    costs={"P07@1.0": {"fx": {"aminx_s": 3600.0, "reference_s": 0.0}}},
    control_scope="per_slot",
  )
  v1 = shard.shard_budget_report(one, n_shards=1)
  assert v1["budget_wall_hours"] == pytest.approx(162.0)
  assert v1["budget_gpu_hours"] == pytest.approx(162.0)


# Bathos run 27f9f25e per-draw costs (aminx_s, reference_s) and floor allocation.
# The P07@0.1 entries are RETAINED (historical record of that run's measurement) but no
# longer PRICED: `_t9_gpu_hours` passes `lanes=las.V2_LANES`, and P07@0.1 is dropped from
# V2_LANES as of T10e (run e091a33e, ctrl_unsized) -- `enumerate_work_units` only reads
# entries for lanes in the `lanes=` it is given, so this dict's extra P07@0.1 key is
# inert dead data here, not a live lane list.
_T9_ALLOCATION: dict[str, dict[str, int]] = {
  "P07@0.1": {"1BC8": 106, "3HTN": 498, "4YOW": 815, "6MRR": 81},
  "P07@1.0": {"1BC8": 112, "3HTN": 495, "4YOW": 813, "6MRR": 80},
  "P08@1.0": {"1BC8": 111, "3HTN": 496, "4YOW": 812, "6MRR": 81},
  "P09-s@1.0": {"3HTN": 410, "4YOW": 1090},
}
_T9_COSTS: dict[str, dict[str, dict[str, float]]] = {
  "P07@0.1": {
    "1BC8": {"aminx_s": 0.04624, "reference_s": 0.05276},
    "3HTN": {"aminx_s": 0.12829, "reference_s": 0.19492},
    "4YOW": {"aminx_s": 0.34249, "reference_s": 0.36918},
    "6MRR": {"aminx_s": 0.04315, "reference_s": 0.03233},
  },
  "P07@1.0": {
    "1BC8": {"aminx_s": 0.04628, "reference_s": 0.05957},
    "3HTN": {"aminx_s": 0.12878, "reference_s": 0.19837},
    "4YOW": {"aminx_s": 0.34620, "reference_s": 0.33511},
    "6MRR": {"aminx_s": 0.04314, "reference_s": 0.03356},
  },
  "P08@1.0": {
    "1BC8": {"aminx_s": 0.04514, "reference_s": 0.05384},
    "3HTN": {"aminx_s": 0.12837, "reference_s": 0.18695},
    "4YOW": {"aminx_s": 0.33690, "reference_s": 0.32389},
    "6MRR": {"aminx_s": 0.04294, "reference_s": 0.03235},
  },
  "P09-s@1.0": {
    "3HTN": {"aminx_s": 0.12773, "reference_s": 0.90765},
    "4YOW": {"aminx_s": 0.11585, "reference_s": 1.56018},
  },
}


def _t9_gpu_hours(control_scope: str) -> float:
  units = shard.enumerate_work_units(
    lanes=las.V2_LANES,
    allocation=_T9_ALLOCATION,
    n_control_replicates=las.N_CONTROL_REPLICATES,
    costs=_T9_COSTS,
    control_scope=control_scope,
  )
  report = shard.shard_budget_report(units, n_shards=1)
  hours = report["budget_gpu_hours"]
  assert isinstance(hours, float)
  return hours


def test_t9_once_pricing_near_17_8_and_per_slot_near_41_1() -> None:
  # T10e: recomputed for the 3-lane V2_LANES (P07@0.1 dropped) -- was 18.2/57.1 over
  # the pre-amendment 4 lanes; recomputed directly via `_t9_gpu_hours`, same recorded
  # run-27f9f25e per-draw costs, just priced over 3 lanes instead of 4.
  assert las.N_CONTROL_REPLICATES == 20
  once = _t9_gpu_hours(shard.PROTOCOL_CONTROL_SCOPE)
  assert once == pytest.approx(17.78, rel=0.01)
  per_slot = _t9_gpu_hours("per_slot")
  assert per_slot == pytest.approx(41.08, rel=0.01)


def test_planner_controls_only_on_p07() -> None:
  units = shard.enumerate_work_units(
    lanes=las.V2_LANES,
    allocation=_T9_ALLOCATION,
    n_control_replicates=las.N_CONTROL_REPLICATES,
    costs=_T9_COSTS,
  )
  control_units = [
    unit for unit in units if unit.arm.startswith("pos/") or unit.arm.startswith("neg/")
  ]
  assert control_units
  assert {unit.lane for unit in control_units} == {las.CONTROL_LANE}
  assert shard.PROTOCOL_CONTROL_SCOPE == "once"
  with pytest.raises(ValueError, match="control lane"):
    shard.enumerate_work_units(
      lanes=("P08@1.0",),
      allocation={"P08@1.0": {"fx": 1}},
    )
  with pytest.raises(ValueError, match="control lane"):
    shard.require_control_lane(("P08@1.0", "P09-s@1.0"))


def test_merged_control_grading_uses_18_of_20() -> None:
  packed = lasv._pack_lane_results(  # noqa: SLF001
    [],
    las.POSCTL_PASS_FLOOR,
    las.NEGCTL_FP_CEIL,
    {},
    native_x_frequency=0.0,
  )
  assert packed["posctl_pass_floor"] == las.POSCTL_PASS_FLOOR == 18
  assert packed["negctl_fp_ceil"] == las.NEGCTL_FP_CEIL == 3
  assert packed["controls_pass"] is True
  low = lasv._pack_lane_results(  # noqa: SLF001
    [],
    las.POSCTL_PASS_FLOOR - 1,
    0,
    {},
    native_x_frequency=0.0,
  )
  assert low["controls_pass"] is False
  merged = merge.grade_scored({"posctl_detected": 18, "negctl_fp": 3})
  assert merged["controls_pass"] is True
  assert merged["posctl_pass_floor"] == las.POSCTL_PASS_FLOOR
  assert merged["negctl_fp_ceil"] == las.NEGCTL_FP_CEIL
  blind = merge.grade_scored({"posctl_detected": 18, "negctl_fp": 4})
  assert blind["controls_pass"] is False


def _mixed_units() -> list[shard.WorkUnit]:
  return [
    shard.WorkUnit("P07@1.0", "fx", "main/A1", 0, 0, 4, 8.0),
    shard.WorkUnit("P07@1.0", "fx", "main/R1", 0, 0, 3, 9.0),
    shard.WorkUnit("P08@1.0", "fy", "pos/A1", 1, 0, 5, 5.0),
    shard.WorkUnit("P09-s@1.0", "fz", "neg/R2", 2, 0, 2, 6.0),
  ]


def test_planner_covers_every_draw_within_lpt_bound() -> None:
  units = _mixed_units()
  expected = sorted(shard.draw_identities(units))
  assert len(expected) == len(set(expected))
  atomic = shard.atomic_costs(units)
  for n_shards in (1, 2, 3):
    plan = shard.plan_shards(units, n_shards)
    again = shard.plan_shards(units, n_shards)
    assert plan == again
    assert len(plan) == n_shards
    flat = [unit for shard_units in plan for unit in shard_units]
    assert sorted(shard.draw_identities(flat)) == expected
    loads = shard.shard_loads(plan)
    assert max(loads) <= shard.lpt_load_bound(atomic, n_shards) + 1e-9
  lone = [shard.WorkUnit("P07@1.0", "fx", "main/A1", 0, 0, 4, 4.0)]
  split = shard.plan_shards(lone, 2)
  assert all(shard_units for shard_units in split)


def _sampler(key: jax.Array) -> np.ndarray:
  return np.asarray(jax.random.uniform(key, (4,), dtype=jnp.float32))


def _partial_for(
  units: list[shard.WorkUnit],
  n_shards: int,
  shard_index: int,
  *,
  git_hash: str = _GIT,
) -> dict[str, object]:
  plan = shard.plan_shards(units, n_shards)
  records = shard.run_assigned_units(plan[shard_index], _sampler, shard_index=shard_index)
  return shard.make_partial(
    git_hash=git_hash,
    params=_PARAMS,
    shard_index=shard_index,
    n_shards=n_shards,
    records=records,
    lanes=["P07@1.0", "P08@1.0", "P09-s@1.0"],
    n_control_replicates=las.N_CONTROL_REPLICATES,
  )


def test_sharded_merge_matches_unsharded_bitwise() -> None:
  units = _mixed_units()
  unsharded = shard.merge_partials(
    [_partial_for(units, 1, 0)],
    expected_units=units,
  )
  sharded = shard.merge_partials(
    [_partial_for(units, 2, 0), _partial_for(units, 2, 1)],
    expected_units=units,
  )
  assert unsharded["value_sha256"] == sharded["value_sha256"]
  assert unsharded["value_sum"] == sharded["value_sum"]
  np.testing.assert_array_equal(
    np.asarray(unsharded["values"]),
    np.asarray(sharded["values"]),
  )


def test_merge_refuses_missing_duplicate_and_git_hash() -> None:
  units = _mixed_units()
  good = [_partial_for(units, 2, 0), _partial_for(units, 2, 1)]
  missing = copy.deepcopy(good)
  records = missing[0]["records"]
  assert isinstance(records, list)
  assert records
  records.pop()
  with pytest.raises(shard.ShardMergeError, match="missing unit"):
    shard.merge_partials(missing, expected_units=units)

  duplicated = copy.deepcopy(good)
  left = duplicated[0]["records"]
  right = duplicated[1]["records"]
  assert isinstance(left, list)
  assert isinstance(right, list)
  assert left
  right.append(copy.deepcopy(left[0]))
  with pytest.raises(shard.ShardMergeError, match="duplicated unit"):
    shard.merge_partials(duplicated, expected_units=units)

  mismatch = copy.deepcopy(good)
  mismatch[1]["git_hash"] = "deadbeef"
  with pytest.raises(shard.ShardMergeError, match="git_hash mismatch"):
    shard.merge_partials(mismatch, expected_units=units)
