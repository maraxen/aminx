"""v2 shard planner for layer-(a) sampling.

Work units are ``(lane, fixture, arm, replicate, draw-index range)``. They are packed
onto ``n_shards`` shards by greedy longest-processing-time (LPT) list scheduling on
the measured per-draw cost, so one ``(lane, fixture)`` cell can land on more than one
shard. PRNG keys come from the unit identity via ``jax.random.fold_in`` and do not
depend on which shard runs the draw.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_sampling as las

MAIN_AMINX_ARMS: tuple[str, ...] = ("main/A1", "main/A2")
MAIN_REFERENCE_ARMS: tuple[str, ...] = ("main/R1", "main/R2")
_CONTROL_HALVES: tuple[str, ...] = ("A1", "A2", "R1", "R2")
_CONTROL_KINDS: tuple[str, ...] = ("pos", "neg")
# validate.run_controls draws the control replicates once. "per_slot" is the v1
# formula (factor 2+8*R on every lane; R=20 -> 162), kept for that test only.
PROTOCOL_CONTROL_SCOPE = "once"


class WorkUnit(NamedTuple):
  lane: str
  fixture: str
  arm: str
  replicate: int
  draw_start: int
  draw_end: int
  cost: float


class ShardMergeError(ValueError):
  """The shard set is not a partition of the protocol units."""


def ensure_xla_preallocate_false() -> None:
  """GPU 3 is shared; do not let XLA grab the whole device unless the caller already did."""
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def device_record() -> dict[str, str]:
  """``device_kind`` plus the visible-device string the launcher set."""
  devices = jax.devices()
  kind = str(getattr(devices[0], "device_kind", devices[0].platform)) if devices else "none"
  return {
    "device_kind": kind,
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  }


def parse_lanes(spec: str | None, *, valid: Sequence[str] = las.LANE_KEYS) -> tuple[str, ...]:
  """Comma list of lane keys. ``None`` or blank selects ``las.V2_LANES``."""
  if spec is None or not spec.strip():
    return tuple(las.V2_LANES)
  lanes = tuple(part.strip() for part in spec.split(",") if part.strip())
  if not lanes:
    msg = "--lanes is empty"
    raise ValueError(msg)
  unknown = [lane for lane in lanes if lane not in valid]
  if unknown:
    msg = f"unknown lanes {unknown}; valid: {tuple(valid)}"
    raise ValueError(msg)
  return lanes


def require_control_lane(
  lanes: Sequence[str],
  *,
  control_lane: str = las.CONTROL_LANE,
) -> tuple[str, ...]:
  """Refuse ``--lanes`` that omits the lane controls are scheduled on."""
  selected = tuple(lanes)
  if control_lane not in selected:
    msg = (
      f"--lanes omits the control lane {control_lane}; "
      "positive and negative controls run once on that lane"
    )
    raise ValueError(msg)
  return selected


def _stable_id(text: str) -> int:
  acc = 0
  for char in text:
    acc = (acc * 131 + ord(char)) & 0x7FFFFFFF
  return acc


def unit_prng_key(
  lane: str,
  fixture: str,
  arm: str,
  replicate: int,
  draw_index: int,
  *,
  shard_index: int | None = None,
) -> jax.Array:
  """Key for one draw. Identity fields only; ``shard_index`` must not enter the key."""
  del shard_index
  key = jax.random.PRNGKey(0)
  key = jax.random.fold_in(key, _stable_id(lane))
  key = jax.random.fold_in(key, _stable_id(fixture))
  key = jax.random.fold_in(key, _stable_id(arm))
  key = jax.random.fold_in(key, int(replicate))
  return jax.random.fold_in(key, int(draw_index))


def folded_seed(
  lane: str,
  fixture: str,
  arm: str,
  replicate: int,
  draw_index: int,
  *,
  shard_index: int | None = None,
) -> int:
  """Integer seed derived from :func:`unit_prng_key` for the existing sample entry points."""
  key = unit_prng_key(
    lane,
    fixture,
    arm,
    replicate,
    draw_index,
    shard_index=shard_index,
  )
  bits = np.asarray(jax.random.bits(key)).reshape(-1)
  return int(bits[0] & 0x7FFFFFFF)


def _allocation_n(
  allocation: Mapping[str, Mapping[str, int] | int],
  lane: str,
  fixture: str,
) -> int:
  if not allocation:
    return 0
  sample = next(iter(allocation.values()))
  if isinstance(sample, dict):
    lane_alloc = allocation.get(lane, {})
    if isinstance(lane_alloc, dict):
      return int(lane_alloc.get(fixture, 0))
    return 0
  return int(allocation.get(fixture, 0))


def _fixtures_for_lane(
  allocation: Mapping[str, Mapping[str, int] | int],
  lane: str,
) -> tuple[str, ...]:
  if not allocation:
    return ()
  sample = next(iter(allocation.values()))
  if isinstance(sample, dict):
    lane_alloc = allocation.get(lane, {})
    if isinstance(lane_alloc, dict):
      return tuple(sorted(str(name) for name in lane_alloc))
    return ()
  return tuple(sorted(str(name) for name in allocation))


def _arm_rows(
  n_control_replicates: int,
  *,
  include_controls: bool,
) -> tuple[tuple[str, int, str], ...]:
  rows: list[tuple[str, int, str]] = [(arm, 0, "aminx") for arm in MAIN_AMINX_ARMS]
  rows.extend((arm, 0, "reference") for arm in MAIN_REFERENCE_ARMS)
  if include_controls:
    rows.extend(
      (f"{kind}/{half}", replicate, "aminx")
      for replicate in range(n_control_replicates)
      for kind in _CONTROL_KINDS
      for half in _CONTROL_HALVES
    )
  return tuple(rows)


def enumerate_work_units(
  *,
  lanes: Sequence[str],
  allocation: Mapping[str, Mapping[str, int] | int],
  n_control_replicates: int = las.N_CONTROL_REPLICATES,
  costs: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
  control_scope: str = PROTOCOL_CONTROL_SCOPE,
  control_lane: str = las.CONTROL_LANE,
) -> list[WorkUnit]:
  """Expand the protocol into priced draw-ranges.

  ``control_scope="once"`` (the v2 budget and what validate runs) emits positive
  and negative controls only on ``control_lane``, which must be selected.
  ``control_scope="per_slot"`` is the v1 formula: those draws on every lane.
  """
  if control_scope not in ("per_slot", "once"):
    msg = f"control_scope must be 'per_slot' or 'once', got {control_scope!r}"
    raise ValueError(msg)
  if n_control_replicates < 0:
    msg = f"n_control_replicates must be >= 0, got {n_control_replicates}"
    raise ValueError(msg)
  if control_scope == "once":
    require_control_lane(lanes, control_lane=control_lane)
  units: list[WorkUnit] = []
  for lane in lanes:
    include_controls = control_scope == "per_slot" or lane == control_lane
    for fixture in _fixtures_for_lane(allocation, lane):
      n_draws = _allocation_n(allocation, lane, fixture)
      if n_draws <= 0:
        continue
      per_draw = _per_draw_costs(costs, lane, fixture)
      for arm, replicate, kind in _arm_rows(
        n_control_replicates,
        include_controls=include_controls,
      ):
        seconds = per_draw[kind] * n_draws
        units.append(
          WorkUnit(
            lane=lane,
            fixture=fixture,
            arm=arm,
            replicate=replicate,
            draw_start=0,
            draw_end=n_draws,
            cost=float(seconds),
          ),
        )
  return units


def _per_draw_costs(
  costs: Mapping[str, Mapping[str, Mapping[str, float]]] | None,
  lane: str,
  fixture: str,
) -> dict[str, float]:
  if costs is None:
    return {"aminx": 1.0, "reference": 1.0}
  lane_costs = costs[lane][fixture]
  return {"aminx": float(lane_costs["aminx_s"]), "reference": float(lane_costs["reference_s"])}


def _lpt_counts(n_items: int, cost: float, loads: list[float]) -> list[int]:
  """How many identical items sequential LPT gives each shard. Updates ``loads``."""
  counts = [0] * len(loads)
  heap = [(loads[index], index) for index in range(len(loads))]
  heapq.heapify(heap)
  for _ in range(n_items):
    load, index = heapq.heappop(heap)
    counts[index] += 1
    new_load = load + cost
    loads[index] = new_load
    heapq.heappush(heap, (new_load, index))
  return counts


def plan_shards(units: Sequence[WorkUnit], n_shards: int) -> list[list[WorkUnit]]:
  """Greedy LPT. Draw ranges are split so a cell need not stay on one shard.

  Within an equal-cost draw range the per-shard *counts* match one-by-one LPT
  (ties go to the lower shard index). Those counts are laid out as contiguous
  ranges so the plan stays compact; the load vector is the LPT load vector.
  """
  if n_shards < 1:
    msg = f"n_shards must be >= 1, got {n_shards}"
    raise ValueError(msg)
  groups: list[tuple[float, WorkUnit]] = []
  for unit in units:
    n_draws = unit.draw_end - unit.draw_start
    if n_draws <= 0:
      continue
    groups.append((unit.cost / n_draws, unit))
  groups.sort(
    key=lambda item: (
      -item[0],
      item[1].lane,
      item[1].fixture,
      item[1].arm,
      item[1].replicate,
      item[1].draw_start,
    ),
  )
  loads = [0.0] * n_shards
  buckets: list[list[WorkUnit]] = [[] for _ in range(n_shards)]
  for per_draw, unit in groups:
    n_draws = unit.draw_end - unit.draw_start
    counts = _lpt_counts(n_draws, per_draw, loads)
    offset = unit.draw_start
    for shard_index, count in enumerate(counts):
      if count <= 0:
        continue
      buckets[shard_index].append(
        WorkUnit(
          lane=unit.lane,
          fixture=unit.fixture,
          arm=unit.arm,
          replicate=unit.replicate,
          draw_start=offset,
          draw_end=offset + count,
          cost=per_draw * count,
        ),
      )
      offset += count
  return buckets


def shard_loads(plan: Sequence[Sequence[WorkUnit]]) -> list[float]:
  return [float(sum(unit.cost for unit in shard)) for shard in plan]


def lpt_load_bound(costs: Sequence[float], n_shards: int) -> float:
  """List-scheduling makespan bound ``(sum + (m-1)*max) / m`` (Graham)."""
  if n_shards < 1:
    msg = f"n_shards must be >= 1, got {n_shards}"
    raise ValueError(msg)
  if not costs:
    return 0.0
  total = float(sum(costs))
  largest = float(max(costs))
  return (total + (n_shards - 1) * largest) / n_shards


def atomic_costs(units: Sequence[WorkUnit]) -> list[float]:
  costs: list[float] = []
  for unit in units:
    n_draws = unit.draw_end - unit.draw_start
    if n_draws <= 0:
      continue
    per_draw = unit.cost / n_draws
    costs.extend([per_draw] * n_draws)
  return costs


def draw_identities(units: Sequence[WorkUnit]) -> list[tuple[str, str, str, int, int]]:
  identities: list[tuple[str, str, str, int, int]] = []
  for unit in units:
    identities.extend(
      (unit.lane, unit.fixture, unit.arm, unit.replicate, draw_index)
      for draw_index in range(unit.draw_start, unit.draw_end)
    )
  return identities


def shard_budget_report(units: Sequence[WorkUnit], n_shards: int) -> dict[str, object]:
  """Per-shard hours, total GPU-hours, and wall hours = the longest shard."""
  plan = plan_shards(units, n_shards)
  seconds = shard_loads(plan)
  hours = [value / 3600.0 for value in seconds]
  wall = max(hours) if hours else 0.0
  return {
    "per_shard_hours": hours,
    "budget_gpu_hours": float(sum(hours)),
    "budget_wall_hours": float(wall),
    "n_shards": n_shards,
    "plan": plan,
  }


Sampler = Callable[[jax.Array], np.ndarray]


def run_assigned_units(
  units: Sequence[WorkUnit],
  sampler: Sampler,
  *,
  shard_index: int,
) -> list[dict[str, object]]:
  """Draw every index in ``units``. The key ignores ``shard_index`` (see ``unit_prng_key``)."""
  records: list[dict[str, object]] = []
  for unit in units:
    for draw_index in range(unit.draw_start, unit.draw_end):
      key = unit_prng_key(
        unit.lane,
        unit.fixture,
        unit.arm,
        unit.replicate,
        draw_index,
        shard_index=shard_index,
      )
      value = np.asarray(sampler(key))
      records.append(
        {
          "lane": unit.lane,
          "fixture": unit.fixture,
          "arm": unit.arm,
          "replicate": unit.replicate,
          "draw_index": draw_index,
          "value": value,
        },
      )
  return records


def make_partial(
  *,
  git_hash: str,
  params: Mapping[str, object],
  shard_index: int,
  n_shards: int,
  records: Sequence[Mapping[str, object]],
  lanes: Sequence[str],
  n_control_replicates: int,
  protocol: Mapping[str, object] | None = None,
) -> dict[str, object]:
  partial: dict[str, object] = {
    "partial": True,
    "git_hash": git_hash,
    "params": dict(params),
    "shard_index": shard_index,
    "n_shards": n_shards,
    "lanes": list(lanes),
    "n_control_replicates": n_control_replicates,
    "records": [dict(record) for record in records],
  }
  if protocol is not None:
    partial["protocol"] = dict(protocol)
  return partial


def _canonical_params(params: Mapping[str, object]) -> str:
  return json.dumps(params, sort_keys=True, default=str)


def _record_identity(record: Mapping[str, object]) -> tuple[str, str, str, int, int]:
  return (
    str(record["lane"]),
    str(record["fixture"]),
    str(record["arm"]),
    int(record["replicate"]),
    int(record["draw_index"]),
  )


def statistics_from_values(ordered_records: Sequence[Mapping[str, object]]) -> dict[str, object]:
  """Reduction shared by an unsharded run and by merge. Order is the identity order."""
  if not ordered_records:
    empty = np.zeros((0, 0), dtype=np.float64)
    return {
      "n_values": 0,
      "value_sum": 0.0,
      "value_sha256": hashlib.sha256(empty.tobytes()).hexdigest(),
      "values": empty,
    }
  rows = [np.asarray(record["value"], dtype=np.float64) for record in ordered_records]
  stacked = np.stack(rows)
  contiguous = np.ascontiguousarray(stacked)
  return {
    "n_values": int(contiguous.size),
    "value_sum": float(contiguous.sum()),
    "value_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    "values": contiguous,
  }


def _ingest_partials(
  partials: Sequence[Mapping[str, object]],
) -> tuple[str, int, list[object], int, dict[tuple[str, str, str, int, int], Mapping[str, object]]]:
  if not partials:
    msg = "no shard partials"
    raise ShardMergeError(msg)
  first = partials[0]
  git_hash = str(first["git_hash"])
  params_text = _canonical_params(_as_params(first.get("params", {})))
  n_shards = int(first["n_shards"])
  lanes = list(first.get("lanes", []))
  n_control_replicates = int(first["n_control_replicates"])
  indices: list[int] = []
  seen: dict[tuple[str, str, str, int, int], Mapping[str, object]] = {}
  for partial in partials:
    _require_same_header(partial, git_hash, params_text, n_shards, lanes, n_control_replicates)
    indices.append(int(partial["shard_index"]))
    _take_records(partial, seen)
  if sorted(indices) != list(range(n_shards)):
    msg = f"shard index set {sorted(indices)} does not cover 0..{n_shards - 1}"
    raise ShardMergeError(msg)
  return git_hash, n_shards, lanes, n_control_replicates, seen


def _require_same_header(
  partial: Mapping[str, object],
  git_hash: str,
  params_text: str,
  n_shards: int,
  lanes: list[object],
  n_control_replicates: int,
) -> None:
  if str(partial["git_hash"]) != git_hash:
    msg = f"git_hash mismatch: {partial['git_hash']} != {git_hash}"
    raise ShardMergeError(msg)
  if _canonical_params(_as_params(partial.get("params", {}))) != params_text:
    msg = "params mismatch across shard partials"
    raise ShardMergeError(msg)
  if int(partial["n_shards"]) != n_shards:
    msg = "n_shards mismatch across shard partials"
    raise ShardMergeError(msg)
  if list(partial.get("lanes", [])) != lanes:
    msg = "lanes mismatch across shard partials"
    raise ShardMergeError(msg)
  if int(partial["n_control_replicates"]) != n_control_replicates:
    msg = "n_control_replicates mismatch across shard partials"
    raise ShardMergeError(msg)


def _take_records(
  partial: Mapping[str, object],
  seen: dict[tuple[str, str, str, int, int], Mapping[str, object]],
) -> None:
  records = partial.get("records", [])
  if not isinstance(records, list):
    msg = "shard records must be a list"
    raise ShardMergeError(msg)
  for record in records:
    if not isinstance(record, Mapping):
      msg = "shard record must be a mapping"
      raise ShardMergeError(msg)
    identity = _record_identity(record)
    if identity in seen:
      msg = f"duplicated unit {identity}"
      raise ShardMergeError(msg)
    seen[identity] = record


def merge_partials(
  partials: Sequence[Mapping[str, object]],
  *,
  expected_units: Sequence[WorkUnit],
) -> dict[str, object]:
  """Refuse a hole, a double-covered draw, or a git/params mismatch. Then reduce."""
  git_hash, n_shards, lanes, n_control_replicates, seen = _ingest_partials(partials)
  first = partials[0]
  expected = set(draw_identities(expected_units))
  missing = expected - set(seen)
  if missing:
    missing_one = min(missing)
    msg = f"missing unit {missing_one}"
    raise ShardMergeError(msg)
  extra = set(seen) - expected
  if extra:
    msg = f"unexpected unit {min(extra)}"
    raise ShardMergeError(msg)
  ordered_keys = sorted(expected)
  ordered_records = [seen[key] for key in ordered_keys]
  if ordered_records and "tokens" in ordered_records[0]:
    return {
      "record_kind": "tokens",
      "records": ordered_records,
      "git_hash": git_hash,
      "params": dict(_as_params(first.get("params", {}))),
      "lanes": lanes,
      "n_control_replicates": n_control_replicates,
      "n_shards": n_shards,
    }
  stats = statistics_from_values(ordered_records)
  stats["record_kind"] = "values"
  stats["git_hash"] = git_hash
  stats["params"] = dict(_as_params(first.get("params", {})))
  stats["lanes"] = lanes
  stats["n_control_replicates"] = n_control_replicates
  stats["n_shards"] = n_shards
  return stats


def _as_params(value: object) -> Mapping[str, object]:
  if isinstance(value, Mapping):
    return value
  msg = "params must be a mapping"
  raise ShardMergeError(msg)
