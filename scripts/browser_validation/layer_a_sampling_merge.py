"""Merge layer-(a) sampling shard partials into one validate result.

Refuses a shard set that misses a draw, covers a draw twice, or disagrees on
``git_hash`` / params / lane scope / control-replicate count. Value-draws (the
synthetic sampler) are reduced with the same function an unsharded run uses.
Token-draws are scored with ``layer_a_sampling_validate.statistics_from_draw_records``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_sampling as las
import layer_a_sampling_shard as lass
import layer_a_sampling_validate as lasv


def _load(path: Path) -> dict[str, object]:
  with path.open() as handle:
    payload: object = json.load(handle)
  if not isinstance(payload, dict):
    msg = f"{path} is not a JSON object"
    raise lass.ShardMergeError(msg)
  return payload


def _expected_units(partials: list[dict[str, object]]) -> list[lass.WorkUnit]:
  protocol = partials[0].get("protocol")
  if not isinstance(protocol, dict):
    msg = "shard partial is missing protocol (lanes, allocation, n_control_replicates)"
    raise lass.ShardMergeError(msg)
  lanes = protocol.get("lanes")
  allocation = protocol.get("allocation")
  if not isinstance(lanes, list) or not isinstance(allocation, dict):
    msg = "protocol.lanes and protocol.allocation are required"
    raise lass.ShardMergeError(msg)
  return lass.enumerate_work_units(
    lanes=[str(lane) for lane in lanes],
    allocation=allocation,
    n_control_replicates=int(protocol.get("n_control_replicates", 0)),
    costs=None,
    control_scope=str(protocol.get("control_scope", lass.PROTOCOL_CONTROL_SCOPE)),
    control_lane=str(protocol.get("control_lane", las.CONTROL_LANE)),
  )


def grade_scored(scored: dict[str, object]) -> dict[str, object]:
  """Apply the pre-registered 18/20 and 3/20 control bar to a merged result."""
  detected = scored["posctl_detected"]
  fp = scored["negctl_fp"]
  if not isinstance(detected, int) or not isinstance(fp, int):
    msg = "posctl_detected and negctl_fp must be ints"
    raise lass.ShardMergeError(msg)
  scored.update(las.grade_controls(detected, fp))
  return scored


def merge_files(paths: list[Path]) -> dict[str, object]:
  partials = [_load(path) for path in paths]
  expected = _expected_units(partials)
  merged = lass.merge_partials(partials, expected_units=expected)
  if merged.get("record_kind") == "tokens":
    records = merged["records"]
    if not isinstance(records, list):
      msg = "merged token records must be a list"
      raise lass.ShardMergeError(msg)
    scored = lasv.statistics_from_draw_records(
      records,
      sampling=_sampling(merged.get("params", {})),
      tf_by_lane=_tf_by_lane(partials),
      native_x_frequency=_native_x(partials),
    )
    scored["git_hash"] = merged["git_hash"]
    scored["lanes"] = merged["lanes"]
    scored["n_control_replicates"] = merged["n_control_replicates"]
    scored["n_shards"] = merged["n_shards"]
    return grade_scored(scored)
  values = np.asarray(merged["values"])
  out = {key: value for key, value in merged.items() if key != "values"}
  out["values"] = values.tolist()
  return out


def _sampling(params: object) -> dict[str, object]:
  if isinstance(params, dict):
    sampling = params.get("sampling", params)
    if isinstance(sampling, dict):
      return sampling
  return {}


def _tf_by_lane(partials: list[dict[str, object]]) -> dict[str, object]:
  for partial in partials:
    if int(partial.get("shard_index", -1)) == 0:
      tf = partial.get("tf_by_lane", {})
      if isinstance(tf, dict):
        return tf
  return {}


def _native_x(partials: list[dict[str, object]]) -> float:
  for partial in partials:
    if int(partial.get("shard_index", -1)) == 0:
      value = partial.get("native_x_frequency", 0.0)
      if isinstance(value, int | float):
        return float(value)
  return 0.0


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("partials", nargs="+", type=Path, help="Shard partial JSON files.")
  parser.add_argument("--out", required=True, type=Path, help="Merged result JSON.")
  args = parser.parse_args(argv)
  try:
    result = merge_files(args.partials)
  except lass.ShardMergeError as exc:
    print(f"layer_a_sampling_merge: {exc}", file=sys.stderr)
    return 1
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("w") as handle:
    json.dump(result, handle, indent=2, default=str)
    handle.write("\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
