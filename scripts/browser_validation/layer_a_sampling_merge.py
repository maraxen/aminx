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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac
import layer_a_sampling as las
import layer_a_sampling_shard as lass
import layer_a_sampling_validate as lasv

# Flat finding schema. Keys match layer_a_sampling_merge.bth.toml [result_schema].
FINDING_DEFAULTS: dict[str, object] = {
  "merge_refused": False,
  "tf_max_abs": 0.0,
  "tf_bar": float(las.TF_BAR),
  "n_lanes": 0,
  "n_lanes_equiv": 0,
  "n_per_arm": 0,
  "n_required": 0,
  "sigma_hat": 0.0,
  "excess_js_ub_max_ratio": 0.0,
  "main_js_vs_ref": 0.0,
  "posctl_detected": 0,
  "negctl_fp": 0,
  "omitted_aa_count": 0,
  "x_token_count_aminx": 0,
  "x_token_count_reference": 0,
  "p09_tied_positions": 0,
  "p09_tied_positions_prereg": 0,
  "p09_fused_tf_max_abs": 0.0,
  "p09_fusion_ctrl_detected": False,
  "n_skipped": 0,
  "prereq_ok": True,
  "reference_commit": "",
  "git_hash": "",
  "git_clean": True,
  "prereg_sha256": "",
}


def _load(path: Path) -> dict[str, object]:
  with path.open() as handle:
    payload: object = json.load(handle)
  if not isinstance(payload, dict):
    msg = f"{path} is not a JSON object"
    raise lass.ShardMergeError(msg)
  return payload


def _resolve_partial(path: Path) -> dict[str, object]:
  """Load a partial. A flat shard result points at its records file via ``records_path``."""
  payload = _load(path)
  if "records" in payload:
    return payload
  raw = payload.get("records_path")
  if not isinstance(raw, str) or not raw:
    msg = f"{path} has no records and no records_path"
    raise lass.ShardMergeError(msg)
  records_file = Path(raw)
  if not records_file.is_absolute():
    records_file = path.parent / records_file
  return _load(records_file)


def _as_int(value: object, default: int = 0) -> int:
  if isinstance(value, bool):
    return int(value)
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(value)
  return default


def _as_float(value: object, default: float = 0.0) -> float:
  if isinstance(value, bool):
    return float(value)
  if isinstance(value, int | float):
    return float(value)
  return default


def finding_result(
  scored: dict[str, object] | None = None,
  partials: list[dict[str, object]] | None = None,
  *,
  merge_refused: bool = False,
) -> dict[str, object]:
  """Flat, finite finding. ``merge_refused`` is a graded outcome, not a crash."""
  out = dict(FINDING_DEFAULTS)
  if scored:
    for key, value in scored.items():
      if key not in out or isinstance(value, dict | list):
        continue
      if isinstance(out[key], bool):
        out[key] = bool(value)
      elif isinstance(out[key], int) and not isinstance(out[key], bool):
        out[key] = _as_int(value)
      elif isinstance(out[key], float):
        out[key] = _as_float(value)
      else:
        out[key] = "" if value is None else str(value)
  out["merge_refused"] = bool(merge_refused)
  rows = partials or []
  if rows:
    out["git_hash"] = str(rows[0].get("git_hash", out["git_hash"]))
    sampling = _sampling(rows[0].get("params", {}))
    if "n_required" not in (scored or {}):
      out["n_required"] = _as_int(sampling.get("n_required"), _as_int(out["n_required"]))
    if "sigma_hat" not in (scored or {}):
      out["sigma_hat"] = _as_float(sampling.get("sigma_hat"), _as_float(out["sigma_hat"]))
    tied = sampling.get("p09_tied_positions")
    if isinstance(tied, int) and "p09_tied_positions_prereg" not in (scored or {}):
      out["p09_tied_positions_prereg"] = tied
    prereq_flags = [row["prereq_ok"] for row in rows if "prereq_ok" in row]
    if prereq_flags:
      out["prereq_ok"] = all(bool(flag) for flag in prereq_flags)
    clean_flags = [row["git_clean"] for row in rows if "git_clean" in row]
    if clean_flags:
      out["git_clean"] = all(bool(flag) for flag in clean_flags)
    if any("n_skipped" in row for row in rows):
      out["n_skipped"] = sum(_as_int(row.get("n_skipped")) for row in rows)
    for row in rows:
      commit = row.get("reference_commit")
      if isinstance(commit, str) and commit:
        out["reference_commit"] = commit
        break
    for row in rows:
      digest = row.get("prereg_sha256")
      if isinstance(digest, str) and digest:
        out["prereg_sha256"] = digest
        break
  return out


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
  partials = [_resolve_partial(path) for path in paths]
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
    graded = grade_scored(scored)
    return finding_result(graded, partials, merge_refused=False)
  projected = finding_result({}, partials, merge_refused=False)
  projected["git_hash"] = str(merged.get("git_hash", projected["git_hash"]))
  return projected


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
    result = finding_result(merge_refused=True)
    lac.emit(result, args.out)
    return 0
  lac.emit(result, args.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
