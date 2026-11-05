"""Tests for aminx host/plan_eda.py integration with xtrax.eda."""

from __future__ import annotations

from pathlib import Path

import pytest
from xtrax.tiling import AxisDecision, AxisSpec, BatchPlan, Bucket, ChunkedMap

from aminx.host.plan_eda import (
  analyze_aminx_bucket,
  explain_aminx_plan,
  export_and_verify_plan_artifacts,
  generate_aminx_plan_spec,
  render_aminx_plan_panels,
  verify_checkpoint_and_explain_plan,
  verify_export_manifest_sha256,
)


def _sample_plan() -> tuple[BatchPlan, AxisDecision]:
  spec_l = AxisSpec(name="L", cardinality=120, default_batch_size=10, bucket_boundaries=(64, 128, 256))
  dec_chunk = AxisDecision(
    spec=spec_l,
    batch_size=10,
    reasoning="chunk residues",
    strategy=ChunkedMap(batch_size=10),
  )
  spec_b = AxisSpec(name="B", cardinality=64, default_batch_size=64, bucket_boundaries=(32, 64, 128))
  dec_bucket = AxisDecision(
    spec=spec_b,
    batch_size=64,
    reasoning="bucket batch size",
    strategy=Bucket(boundaries=(32, 64, 128)),
  )
  plan = BatchPlan(decisions=[dec_chunk, dec_bucket])
  return plan, dec_bucket


def test_explain_aminx_plan() -> None:
  """Tests plan explanation including summary and bucket analyses."""
  plan, _ = _sample_plan()
  explained = explain_aminx_plan(plan)

  assert "plan_summary" in explained
  assert "bucket_analyses" in explained

  summary = explained["plan_summary"]
  assert summary["total_axes"] == 2
  assert summary["strategy_counts"].get("ChunkedMap") == 1
  assert summary["strategy_counts"].get("Bucket") == 1

  bucket_analyses = explained["bucket_analyses"]
  assert len(bucket_analyses) == 1
  b_stat = bucket_analyses[0]
  assert b_stat["axis_name"] == "B"
  assert b_stat["bucket_count"] == 3
  assert b_stat["bucket_boundaries"] == [32, 64, 128]


def test_analyze_aminx_bucket() -> None:
  """Tests standalone analyze_aminx_bucket."""
  _, dec_bucket = _sample_plan()
  stat = analyze_aminx_bucket(dec_bucket)

  assert stat["axis_name"] == "B"
  assert stat["bucket_count"] == 3
  assert stat["bucket_boundaries"] == [32, 64, 128]


def test_verify_checkpoint_and_explain_plan() -> None:
  """Tests verify_checkpoint_and_explain_plan with known pinned family checkpoint."""
  plan, _ = _sample_plan()
  result = verify_checkpoint_and_explain_plan(
    plan,
    checkpoint_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
  )
  assert "plan_summary" in result
  assert "bucket_analyses" in result
  assert result["plan_summary"]["total_axes"] == 2


def test_verify_checkpoint_and_explain_plan_rejects_unknown_checkpoint() -> None:
  """Tests that nonexistent checkpoint family raises during verification."""
  plan, _ = _sample_plan()
  with pytest.raises(Exception):
    verify_checkpoint_and_explain_plan(
      plan,
      checkpoint_family="unknown_family_nonexistent",
      checkpoint_id="invalid_id_999",
    )


def test_render_aminx_plan_panels() -> None:
  """Tests rendering plan panels to SVG format via xtrax.eda."""
  plan, _ = _sample_plan()
  svg_bytes = render_aminx_plan_panels(plan, fmt="svg")
  assert isinstance(svg_bytes, bytes)
  assert len(svg_bytes) > 0
  assert b"<svg" in svg_bytes.lower()


def test_generate_aminx_plan_spec() -> None:
  """Tests generating declarative Vega-Lite specification."""
  plan, _ = _sample_plan()
  spec = generate_aminx_plan_spec(plan, panels={"strategy", "cardinality"})
  assert isinstance(spec, dict)
  assert "$schema" in spec
  assert "description" in spec


def test_export_and_verify_plan_artifacts(tmp_path: Path) -> None:
  """Positive test: export plan artifacts and verify SHA-256 digests."""
  plan, _ = _sample_plan()
  manifest = export_and_verify_plan_artifacts(plan, tmp_path, prefix="test_plan")
  assert manifest["schema_version"] == "2.0"
  assert manifest["total_axes"] == 2
  assert "test_plan.canonical.json" in manifest["artifacts"]
  assert "test_plan_axes.csv" in manifest["artifacts"]
  assert "test_plan.vl.json" in manifest["artifacts"]


def test_negative_control_corrupted_export_artifact_raises(tmp_path: Path) -> None:
  """Negative control: tampering with an exported artifact is detected and raises ValueError."""
  plan, _ = _sample_plan()
  export_and_verify_plan_artifacts(plan, tmp_path, prefix="test_plan")

  # Tamper with test_plan_axes.csv
  target_file = tmp_path / "test_plan_axes.csv"
  target_file.write_text("corrupted,axes,data\n1,2,3\n", encoding="utf-8")

  with pytest.raises(ValueError, match="Byte length mismatch|SHA-256 digest mismatch"):
    verify_export_manifest_sha256(tmp_path)
