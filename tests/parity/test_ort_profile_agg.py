"""Tests for `scripts/browser_validation/ort_profile_agg.py` (T6 step 1).

Covers: exact recovery of known synthetic durations within 1e-9 (both directly and via
the shared `synthetic_self_check` the tracked run also calls), a truncated/malformed
JSON file raising, a Node event missing `args.op_name` raising, and the warm-up-dropping
red-check mutation (an off-by-one in `buckets[n_warmup:]` must break the exact-recovery
assertion).
"""

from __future__ import annotations

import json

import pytest

from scripts.browser_validation.ort_profile_agg import (
  aggregate_profile,
  build_synthetic_profile,
  group_by_model_run,
  load_and_aggregate,
  synthetic_self_check,
)


def test_synthetic_durations_recovered_within_1e9() -> None:
  """Positive control: a synthetic profile with KNOWN durations (Add=10us, Relu=5us,
  model_run=20us, 5 warm-ups + 3 measured) must recover mean_us_per_run, count and
  coverage exactly (within 1e-9), matching this task's Gate requirement verbatim."""
  events = build_synthetic_profile(
    n_warmup=5, n_measured=3, node_durs_us={"Add": 10.0, "Relu": 5.0}, model_run_dur_us=20.0
  )
  result = aggregate_profile(events, n_warmup=5)

  assert result["n_runs_total"] == 8
  assert result["n_runs_measured"] == 3
  assert result["n_runs_dropped_warmup"] == 5
  assert result["ops"]["Add"]["count"] == 3
  assert abs(result["ops"]["Add"]["mean_us_per_run"] - 10.0) < 1e-9
  assert abs(result["ops"]["Relu"]["mean_us_per_run"] - 5.0) < 1e-9
  assert abs(result["ops"]["Add"]["share"] - (10.0 / 15.0)) < 1e-9
  assert abs(result["ops"]["Relu"]["share"] - (5.0 / 15.0)) < 1e-9
  expected_coverage = (10.0 + 5.0) / 20.0
  assert abs(result["coverage"] - expected_coverage) < 1e-9


def test_synthetic_self_check_passes_on_the_real_instrument() -> None:
  """The exact self-check `layer_b_profile.py` calls in-run (ctrl_blind gate) must itself
  report ok=True against the real (unmutated) `aggregate_profile`."""
  outcome = synthetic_self_check()
  assert outcome["ok"] is True, outcome["detail"]


def test_group_by_model_run_drops_trailing_incomplete_bucket() -> None:
  """Node events with no following model_run boundary are dropped (no dur to normalize
  against), not silently folded into the previous or a phantom next bucket."""
  events = [
    {"cat": "Node", "name": "a", "dur": 1.0, "args": {"op_name": "A"}},
    {"cat": "Session", "name": "model_run", "dur": 5.0},
    {"cat": "Node", "name": "b", "dur": 2.0, "args": {"op_name": "B"}},  # trailing, no boundary
  ]
  buckets = group_by_model_run(events)
  assert len(buckets) == 1
  assert buckets[0]["node_events"][0]["args"]["op_name"] == "A"


def test_malformed_missing_op_name_raises() -> None:
  """A Node event missing args.op_name must raise, not silently drop the row."""
  events = build_synthetic_profile(n_warmup=0, n_measured=1, node_durs_us={"Add": 10.0})
  # Corrupt the sole Node event's args.
  for event in events:
    if event.get("cat") == "Node":
      event["args"] = {}
  with pytest.raises(ValueError, match="op_name"):
    aggregate_profile(events, n_warmup=0)


def test_not_a_list_raises() -> None:
  with pytest.raises(ValueError, match="list"):
    aggregate_profile({"not": "a list"}, n_warmup=0)  # type: ignore[arg-type]


def test_truncated_json_file_raises(tmp_path) -> None:
  """A truncated/malformed profile JSON file raises when loaded (json.JSONDecodeError),
  never silently degrading to an empty/zero aggregate."""
  path = tmp_path / "truncated.json"
  path.write_text('[{"cat": "Node", "dur": 1.0, "args": {"op_name": "A"}')  # missing closes
  with pytest.raises(json.JSONDecodeError):
    load_and_aggregate(path)


def test_too_few_buckets_for_n_warmup_raises() -> None:
  """Fewer model_run buckets than n_warmup leaves nothing to measure -- must raise, not
  silently report an empty/zero aggregate."""
  events = build_synthetic_profile(n_warmup=0, n_measured=2)
  with pytest.raises(ValueError, match="n_warmup"):
    aggregate_profile(events, n_warmup=5)


# --------------------------------------------------------------------------------------
# Red-check mutation (Gate): an off-by-one in warm-up dropping must fail
# `test_synthetic_durations_recovered_within_1e9` above. This test pins the SPECIFIC
# failure mode (wrong n_runs_measured / wrong per-op stats) an off-by-one produces, so the
# fixer's manual mutate-observe-revert cycle (this task's own red-check requirement) has a
# concrete target: `aggregate_profile`'s `measured = buckets[n_warmup:]` line.
# --------------------------------------------------------------------------------------


def test_off_by_one_warmup_boundary_would_be_caught() -> None:
  """Demonstrates the sensitivity the red-check mutation trips: dropping one FEWER (or
  one MORE) warm-up bucket than declared changes n_runs_measured and the recovered
  per-op mean, so a mutated `buckets[n_warmup:]` (e.g. `buckets[n_warmup - 1:]`) is
  caught by comparing against the same known-ground-truth synthetic profile."""
  events = build_synthetic_profile(
    n_warmup=5, n_measured=3, node_durs_us={"Add": 10.0}, model_run_dur_us=20.0
  )
  correct = aggregate_profile(events, n_warmup=5)
  off_by_one = aggregate_profile(events, n_warmup=4)  # simulates the off-by-one mutation
  assert correct["n_runs_measured"] == 3
  assert off_by_one["n_runs_measured"] == 4
  assert correct["n_runs_measured"] != off_by_one["n_runs_measured"]
