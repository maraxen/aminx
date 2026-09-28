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
import scripts.browser_validation.layer_a_sampling_shard as shard

_PARAMS = {"sampling": {"n_required": 4}}
_GIT = "abc123"


def test_control_replicate_factor_v1_and_v2() -> None:
  assert las.N_CONTROL_REPLICATES == 5
  assert las.aminx_draws_per_slot(20) == 162
  assert las.aminx_draws_per_slot(5) == 42
  assert las.aminx_draws_per_slot() == 42
  assert las.REFERENCE_DRAWS_PER_SLOT == 2


def test_v2_lanes_budget_excludes_p11() -> None:
  assert las.V2_LANES == ("P07@0.1", "P07@1.0", "P08@1.0", "P09-s@1.0")
  assert "P11-s@1.0" not in las.V2_LANES
  costs = {lane: {"fx": {"aminx_s": 3600.0, "reference_s": 0.0}} for lane in las.LANE_KEYS}
  allocation = {lane: {"fx": 1} for lane in las.LANE_KEYS}
  units = shard.enumerate_work_units(
    lanes=las.V2_LANES,
    allocation=allocation,
    n_control_replicates=5,
    costs=costs,
    control_scope="per_slot",
  )
  assert units
  assert all(unit.lane != "P11-s@1.0" for unit in units)
  report = shard.shard_budget_report(units, n_shards=2)
  per_shard = report["per_shard_hours"]
  assert isinstance(per_shard, list)
  assert report["budget_gpu_hours"] == pytest.approx(42.0 * len(las.V2_LANES))
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
