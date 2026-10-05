"""Aggregate an ONNX Runtime chrome-trace profile into per-op stats + coverage (T6 step 1).

`enable_profiling=True` (Python) writes a JSON list of chrome-trace-format events. Two
shapes matter here (empirically confirmed against onnxruntime 1.30.0, `/tmp/ort_profile_test`,
260928):

- **Node events**: `cat == "Node"`, `dur` in MICROSECONDS, `args.op_name`, `args.provider`.
  One pair (or more) of these precedes each `session.run()` call in the event stream.
- **`model_run` boundary events**: `cat == "Session"`, `name == "model_run"`, `dur` in
  microseconds -- one per `session.run()` call, and it appears in the stream immediately
  AFTER the Node events belonging to that run (and after a `SequentialExecutor::Execute`
  event, which this module ignores). Grouping Node events into buckets-closed-by-the-next-
  `model_run` event therefore recovers exactly one bucket per `session.run()` call, in order,
  including the very first (whose Node events precede any Session event at all).

`aggregate_profile` drops the first `n_warmup` buckets (default 5, per the common context's
result-emission rule and this task's own step 2: "5 warm-ups + 30 runs"), then computes, over
the REMAINING buckets: per-op-type `{mean_us_per_run, share, count}` and
`coverage = sum(node dur) / sum(model_run dur)`.

`synthetic_self_check` is the in-run instrument self-test `layer_b_profile.py` calls before
trusting any real aggregation (this task's sidecar `ctrl_blind` branch: "synthetic aggregator
check fails in-run"); it is also exercised directly by
`tests/parity/test_ort_profile_agg.py`, so both the pytest gate and the tracked run call
the identical function.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_N_WARMUP = 5


def group_by_model_run(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Group `cat == "Node"` events into buckets closed by the next `model_run` event.

  Returns one `{"node_events": [...], "model_run_dur": float}` dict per `model_run`
  boundary encountered, in stream order. Node events accumulate into the current
  (not-yet-closed) bucket; any Node events left over after the LAST `model_run` event (an
  incomplete trailing bucket with no boundary duration to normalize against) are dropped.
  """
  if not isinstance(events, list):
    msg = f"ort profile JSON must be a list of trace events, got {type(events).__name__}"
    raise ValueError(msg)

  buckets: list[dict[str, Any]] = []
  current: list[dict[str, Any]] = []
  for event in events:
    if not isinstance(event, dict):
      msg = f"ort profile event is not an object: {event!r}"
      raise ValueError(msg)
    if event.get("cat") == "Node":
      current.append(event)
    elif event.get("cat") == "Session" and event.get("name") == "model_run":
      buckets.append({"node_events": current, "model_run_dur": float(event.get("dur", 0.0))})
      current = []
  return buckets


def aggregate_profile(
  events: list[dict[str, Any]], *, n_warmup: int = DEFAULT_N_WARMUP
) -> dict[str, Any]:
  """Aggregate `events` (already-parsed ORT profile JSON) into per-op stats + coverage.

  Raises `ValueError` if the input is malformed (not a list, an event missing
  `args.op_name`), or if there are not more than `n_warmup` `model_run` buckets to begin
  with (nothing would be left to measure).
  """
  buckets = group_by_model_run(events)
  if len(buckets) <= n_warmup:
    msg = (
      f"ort profile has {len(buckets)} model_run boundaries, need > n_warmup={n_warmup} "
      "to leave at least one measured run"
    )
    raise ValueError(msg)
  measured = buckets[n_warmup:]

  per_op: dict[str, dict[str, float]] = {}
  total_node_dur_us = 0.0
  total_model_run_dur_us = 0.0
  for bucket in measured:
    total_model_run_dur_us += bucket["model_run_dur"]
    for event in bucket["node_events"]:
      args = event.get("args") or {}
      op_name = args.get("op_name")
      if op_name is None:
        msg = f"Node event missing args.op_name: {event!r}"
        raise ValueError(msg)
      dur_us = float(event.get("dur", 0.0))
      total_node_dur_us += dur_us
      entry = per_op.setdefault(op_name, {"total_dur_us": 0.0, "count": 0.0})
      entry["total_dur_us"] += dur_us
      entry["count"] += 1.0

  n_runs = len(measured)
  ops: dict[str, dict[str, float]] = {}
  for op_name, entry in per_op.items():
    total_dur_us = entry["total_dur_us"]
    ops[op_name] = {
      "mean_us_per_run": total_dur_us / n_runs,
      "share": (total_dur_us / total_node_dur_us) if total_node_dur_us > 0 else 0.0,
      "count": int(entry["count"]),
    }
  coverage = (total_node_dur_us / total_model_run_dur_us) if total_model_run_dur_us > 0 else 0.0

  return {
    "n_runs_total": len(buckets),
    "n_runs_measured": n_runs,
    "n_runs_dropped_warmup": n_warmup,
    "ops": ops,
    "total_node_dur_us": total_node_dur_us,
    "total_model_run_dur_us": total_model_run_dur_us,
    "coverage": coverage,
  }


def load_and_aggregate(path: Path, *, n_warmup: int = DEFAULT_N_WARMUP) -> dict[str, Any]:
  """Read `path` (an ORT profile JSON file) and aggregate it. Propagates JSONDecodeError
  on a truncated/malformed file (the caller's `incomplete`/`ctrl_blind` branch)."""
  with Path(path).open() as fh:
    events = json.load(fh)
  return aggregate_profile(events, n_warmup=n_warmup)


# ----------------------------------------------------------------------------------
# Synthetic self-check (T6 step 3, ctrl_blind: "synthetic aggregator check fails in-run")
# ----------------------------------------------------------------------------------

_SYNTH_NODE_DURS_US: dict[str, float] = {"Add": 10.0, "Relu": 5.0}
_SYNTH_MODEL_RUN_DUR_US = 20.0
_SYNTH_N_WARMUP = 5
_SYNTH_N_MEASURED = 3


def build_synthetic_profile(
  *,
  n_warmup: int = _SYNTH_N_WARMUP,
  n_measured: int = _SYNTH_N_MEASURED,
  node_durs_us: dict[str, float] | None = None,
  model_run_dur_us: float = _SYNTH_MODEL_RUN_DUR_US,
) -> list[dict[str, Any]]:
  """Build a synthetic ORT-profile-shaped event list with EXACTLY known durations.

  `n_warmup + n_measured` identical `model_run` buckets, each containing one `Node` event
  per `node_durs_us` entry (fixed `dur`) followed by the bucket's `model_run` boundary
  event. Used by both `synthetic_self_check` (the in-run instrument self-test) and
  `tests/parity/test_ort_profile_agg.py` (the tracked unit test), so both paths exercise
  the identical known-ground-truth construction.
  """
  durs = node_durs_us if node_durs_us is not None else _SYNTH_NODE_DURS_US
  events: list[dict[str, Any]] = []
  for _ in range(n_warmup + n_measured):
    for op_name, dur_us in durs.items():
      events.append(
        {
          "cat": "Node",
          "name": f"{op_name}_kernel_time",
          "dur": dur_us,
          "args": {"op_name": op_name},
        }
      )
    events.append({"cat": "Session", "name": "model_run", "dur": model_run_dur_us})
  return events


def synthetic_self_check() -> dict[str, Any]:
  """Aggregate a synthetic profile with known durations; assert recovery within 1e-9.

  Returns `{"ok": bool, "detail": {...}}` -- never raises (a raising self-check would be
  indistinguishable, at the caller, from "the instrument itself crashed", which is a
  DIFFERENT finding than "the instrument ran and disagreed with ground truth").
  """
  events = build_synthetic_profile()
  expected_coverage = sum(_SYNTH_NODE_DURS_US.values()) / _SYNTH_MODEL_RUN_DUR_US
  try:
    result = aggregate_profile(events, n_warmup=_SYNTH_N_WARMUP)
    checks = {
      "n_runs_measured": result["n_runs_measured"] == _SYNTH_N_MEASURED,
      "coverage": abs(result["coverage"] - expected_coverage) < 1e-9,
    }
    for op_name, dur_us in _SYNTH_NODE_DURS_US.items():
      row = result["ops"].get(op_name)
      checks[f"{op_name}_mean"] = row is not None and abs(row["mean_us_per_run"] - dur_us) < 1e-9
      checks[f"{op_name}_count"] = row is not None and row["count"] == _SYNTH_N_MEASURED
    ok = all(checks.values())
    return {"ok": ok, "detail": {"checks": checks, "result": result}}
  except Exception as exc:  # noqa: BLE001 -- the exception text itself is the finding
    return {"ok": False, "detail": {"error": f"{type(exc).__name__}: {exc}"}}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("profile_json", type=Path, help="Path to an ORT profile JSON file.")
  parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
  parser.add_argument(
    "--out", type=Path, default=None, help="Optional path to write the aggregate JSON."
  )
  args = parser.parse_args(argv)

  aggregate = load_and_aggregate(args.profile_json, n_warmup=args.n_warmup)
  text = json.dumps(aggregate, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.write_text(text)
  else:
    print(text)
  return 0


if __name__ == "__main__":
  sys.exit(main())
