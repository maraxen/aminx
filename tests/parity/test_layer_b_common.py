"""Tests for `scripts/browser_validation/layer_b_common.py`'s T3a additions.

Covers the near-tie rule (`near_tie_rows`, `compare_neighbor_indices` -- both a
positive case that must detect a mismatch and a negative case that must NOT flag a
near-tie row as a genuine failure), the headroom three-way classifier
(`classify_headroom`), the geometric-bisection control sizer (`size_control_delta` --
both a case that must land in its window and one that must NOT, exhausting the
bracket), per-path build availability (`converted_by_path`), artifact-subdir discovery
by sha256 verification (`resolve_artifact_subdir`), and the ONNX Runtime profile
parser (`ep_provider_histogram`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.browser_validation.layer_b_common import (
  classify_headroom,
  compare_neighbor_indices,
  converted_by_path,
  ep_provider_histogram,
  near_tie_rows,
  resolve_artifact_subdir,
  size_control_delta,
)


def test_near_tie_rows_flags_a_genuine_near_tie() -> None:
  """Positive control: row 0's two nearest neighbours (rows 1, 2) sit 1.0 and 1.00005 A
  away -- a 5e-5 A gap, below epsilon=1e-4 -- so row 0's own first-k+1 sorted distances
  carry a near-tie and must flag. Row 2 (whose own nearest neighbour, row 1, is far
  closer than its second-nearest) must NOT flag: this is the asymmetry the near-tie rule
  is precise about (it is a property of each row's OWN sorted distances, not a symmetric
  relation between the two points that happen to be close)."""
  ca = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.00005, 0, 0], [10.0, 0, 0]])
  flagged = near_tie_rows(ca, k=1, epsilon=1e-4)
  assert bool(flagged[0]) is True
  assert bool(flagged[2]) is False


def test_near_tie_rows_does_not_flag_well_separated_points() -> None:
  """Negative control: points >> epsilon apart at every rank must NOT flag."""
  ca = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [20.0, 0, 0]])
  flagged = near_tie_rows(ca, k=1, epsilon=1e-4)
  assert not flagged.any()


def test_compare_neighbor_indices_order_independent_match() -> None:
  """Same SET, different order -> not a mismatch (set comparison, not positional)."""
  idx_a = np.array([[1, 2], [0, 2]])
  idx_b = np.array([[2, 1], [0, 2]])
  cmp = compare_neighbor_indices(
    idx_a, idx_b, real_mask=np.array([True, True]), near_tie=np.array([False, False])
  )
  assert cmp["n_mismatch_total"] == 0


def test_compare_neighbor_indices_flags_genuine_mismatch_on_non_near_tie_row() -> None:
  """Positive control: a real index disagreement on a NON-near-tie row must count."""
  idx_a = np.array([[1, 2]])
  idx_b = np.array([[1, 3]])
  cmp = compare_neighbor_indices(
    idx_a, idx_b, real_mask=np.array([True]), near_tie=np.array([False])
  )
  assert cmp["n_mismatch_total"] == 1
  assert cmp["n_mismatch_non_near_tie"] == 1
  assert cmp["all_mismatches_near_tie"] is False


def test_compare_neighbor_indices_excuses_mismatch_on_near_tie_row() -> None:
  """Negative control: the SAME disagreement, but on a row flagged near-tie, must NOT
  count toward `n_mismatch_non_near_tie` (it is a `tie_unstable` candidate instead)."""
  idx_a = np.array([[1, 2]])
  idx_b = np.array([[1, 3]])
  cmp = compare_neighbor_indices(
    idx_a, idx_b, real_mask=np.array([True]), near_tie=np.array([True])
  )
  assert cmp["n_mismatch_total"] == 1
  assert cmp["n_mismatch_non_near_tie"] == 0
  assert cmp["all_mismatches_near_tie"] is True
  assert cmp["n_tie_swapped_slots"] == 2  # symmetric difference {2, 3}


def test_classify_headroom_three_states() -> None:
  assert classify_headroom(0.0, bar=1e-4) == "advanced"
  assert classify_headroom(4e-5, bar=1e-4) == "advanced"
  assert classify_headroom(6e-5, bar=1e-4) == "low_headroom"
  assert classify_headroom(2e-4, bar=1e-4) == "not_advanced"


def test_size_control_delta_lands_in_window() -> None:
  """Positive control: a metric linear in delta must find a delta in [2x, 10x] of bar."""
  delta, tried = size_control_delta(
    lambda d: d * 1000.0, bar=1.0, lo=1e-7, hi=1e-1, target=(2.0, 10.0), max_steps=20
  )
  assert delta is not None
  ratio = (delta * 1000.0) / 1.0
  assert 2.0 <= ratio <= 10.0
  assert len(tried) >= 1


def test_size_control_delta_returns_none_when_bracket_cannot_reach_window() -> None:
  """Negative control: a metric that is always far below the window even at `hi` must
  return `None` (the `ctrl_unsized` case), not a false-positive delta."""
  delta, tried = size_control_delta(
    lambda d: d * 1e-3, bar=1.0, lo=1e-7, hi=1e-1, target=(2.0, 10.0), max_steps=20
  )
  assert delta is None
  assert len(tried) == 20


def test_converted_by_path_requires_every_bucket_clean() -> None:
  manifest = {
    "artifacts": {
      "p03_L128.onnx": {"role": "clean"},
      "p03_L256.onnx": {"role": "clean"},
      "p04_L128.onnx": {"role": "clean"},
      # p04_L256.onnx missing -> p04 not available
    }
  }
  result = converted_by_path(manifest, ("p03", "p04"), (128, 256))
  assert result == {"p03": True, "p04": False}


def test_resolve_artifact_subdir_discovers_by_sha256(tmp_path: Path) -> None:
  artifacts_base = tmp_path / "artifacts"
  good_dir = artifacts_base / "good"
  bad_dir = artifacts_base / "bad"
  good_dir.mkdir(parents=True)
  bad_dir.mkdir(parents=True)
  (good_dir / "p03_L128.onnx").write_bytes(b"real-bytes")
  (bad_dir / "p03_L128.onnx").write_bytes(b"stale-bytes")

  import hashlib

  sha = hashlib.sha256(b"real-bytes").hexdigest()
  manifest_path = tmp_path / "artifact_manifest.json"
  manifest_path.write_text(json.dumps({"artifacts": {"p03_L128.onnx": {"sha256": sha}}}))

  subdir = resolve_artifact_subdir(artifacts_base, manifest_path)
  assert subdir == "good"


def test_ep_provider_histogram_counts_node_provider_events(tmp_path: Path) -> None:
  profile = [
    {"cat": "Node", "name": "n1", "args": {"provider": "CPUExecutionProvider"}},
    {"cat": "Node", "name": "n2", "args": {"provider": "CPUExecutionProvider"}},
    {"cat": "Session", "name": "init"},  # no provider -> not counted
  ]
  profile_path = tmp_path / "profile.json"
  profile_path.write_text(json.dumps(profile))
  histogram = ep_provider_histogram(profile_path)
  assert histogram == {"CPUExecutionProvider": 2}
