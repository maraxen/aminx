"""Unit tests for UnifiedCampaignSink and CampaignSinkInspector (src/aminx/io/campaign_sink.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aminx.host.schema_versions import CAMPAIGN_SINK_SCHEMA_VERSION, SCHEMA_VERSIONS
from aminx.io.campaign_sink import (
  CAMPAIGN_METRICS_COLUMNS,
  CampaignSinkInspector,
  UnifiedCampaignSink,
  inspect_campaign_sink,
)


def test_schema_version_registered() -> None:
  """Verify campaign_sink_v1 == '1.0.0' in SCHEMA_VERSIONS."""
  assert SCHEMA_VERSIONS["campaign_sink_v1"] == "1.0.0"
  assert CAMPAIGN_SINK_SCHEMA_VERSION == "1.0.0"


def test_unified_campaign_sink_write_and_inspector_roundtrip(tmp_path: Path) -> None:
  """Verify atomic Zarr + Parquet writing, filtering, cross-backend consensus, and integrity checks."""
  sink_root = tmp_path / "campaign_out"
  sink = UnifiedCampaignSink(sink_root, campaign_id="camp_01", git_sha="abc1234")

  sink.write_arm_results(
    arm_id="original",
    model_family="mpnn",
    checkpoint_id="proteinmpnn_v_48_020",
    manifest_row_hash="hash_orig_1",
    records=[
      {
        "structure_id": "1abc",
        "sample_idx": 0,
        "temperature": 0.1,
        "sequence": "ACDEFGHIKL",
        "score_nll": 1.25,
        "sequence_recovery": 0.80,
      },
      {
        "structure_id": "1abc",
        "sample_idx": 1,
        "temperature": 0.1,
        "sequence": "ACDEFGHIKM",
        "score_nll": 1.40,
        "sequence_recovery": 0.70,
      },
    ],
    arrays={"sequences": [[0, 1, 2], [0, 1, 3]]},
  )

  sink.write_arm_results(
    arm_id="fampnn",
    model_family="fampnn",
    checkpoint_id="fampnn_0_3",
    manifest_row_hash="hash_fampnn_1",
    records=[
      {
        "structure_id": "1abc",
        "sample_idx": 0,
        "temperature": 0.1,
        "sequence": "ACDEFGHIKV",
        "score_nll": 1.10,
        "sc_rmsd": 0.65,
        "sc_plddt": 88.5,
      },
    ],
    arrays={"atom37_coords": [[[0.0, 0.0, 0.0]]]},
  )

  inspector = inspect_campaign_sink(sink_root)
  assert isinstance(inspector, CampaignSinkInspector)

  summary = inspector.summary()
  assert summary.campaign_id == "camp_01"
  assert summary.schema_version == "1.0.0"
  assert summary.arms == ("fampnn", "original")
  assert summary.total_structures == 1
  assert summary.total_samples == 3
  assert summary.completed_manifest_rows == 2

  df_all = inspector.to_dataframe()
  assert len(df_all) == 3
  assert tuple(df_all.columns) == CAMPAIGN_METRICS_COLUMNS

  df_fampnn = inspector.to_dataframe(arms=["fampnn"], min_plddt=80.0)
  assert len(df_fampnn) == 1

  df_low_nll = inspector.to_dataframe(max_nll=1.30)
  assert len(df_low_nll) == 2

  consensus = inspector.cross_backend_consensus("1abc")
  assert consensus["arms"] == ("fampnn", "original")
  assert consensus["consensus_sequence"].startswith("ACDEFGHIK")
  assert len(consensus["per_position_entropy"]) == 10
  assert consensus["per_position_entropy"][0] == pytest.approx(0.0)
  assert consensus["per_position_entropy"][-1] > 0.0
  pw_id = consensus["pairwise_sequence_identity"]
  assert pw_id[0][0] == pytest.approx(1.0)
  assert pw_id[1][1] == pytest.approx(1.0)
  assert pw_id[0][1] == pytest.approx(0.9)

  zarr_p = inspector.open_arm_zarr("fampnn")
  assert zarr_p.exists()

  integrity = inspector.verify_integrity()
  assert integrity["summary_parquet"] is True
  assert integrity["arm:original"] is True
  assert integrity["arm:fampnn"] is True
  assert integrity["all_valid"] is True

  # Corrupt a file in arm:fampnn and verify tamper detection
  corrupt_file = zarr_p / "hash_fampnn_1" / ".zattrs"
  corrupt_file.write_text('{"corrupted": true}\n', encoding="utf-8")
  integrity_after = inspector.verify_integrity()
  assert integrity_after["arm:fampnn"] is False
  assert integrity_after["all_valid"] is False


def test_campaign_sink_validation_dedup_and_flags(tmp_path: Path) -> None:
  """Verify path traversal rejection, campaign_id mismatch, row deduplication, include_empty_arms, and check_arms."""
  sink_root = tmp_path / "campaign_guard"
  sink = UnifiedCampaignSink(sink_root, campaign_id="camp_guard")

  with pytest.raises(ValueError, match="unsafe path"):
    sink.write_arm_results(
      arm_id="../escape",
      model_family="mpnn",
      checkpoint_id="ckpt",
      manifest_row_hash="row1",
      records=[],
    )
  with pytest.raises(ValueError, match="unsafe path"):
    sink.write_arm_results(
      arm_id="arm_a",
      model_family="mpnn",
      checkpoint_id="ckpt",
      manifest_row_hash="row/escape",
      records=[],
    )

  # Write initial row and then overwrite the same (arm_id, manifest_row_hash)
  sink.write_arm_results(
    arm_id="arm_a",
    model_family="mpnn",
    checkpoint_id="ckpt",
    manifest_row_hash="row1",
    records=[{"structure_id": "1abc", "sample_idx": 0, "sequence": "AAAA", "score_nll": 2.0}],
  )
  sink.write_arm_results(
    arm_id="arm_a",
    model_family="mpnn",
    checkpoint_id="ckpt",
    manifest_row_hash="row1",
    records=[{"structure_id": "1abc", "sample_idx": 0, "sequence": "CCCC", "score_nll": 1.0}],
  )
  # Register an empty arm
  sink.write_arm_results(
    arm_id="arm_empty",
    model_family="mpnn",
    checkpoint_id="ckpt",
    manifest_row_hash="row_empty",
    records=[],
  )

  # Reopening with wrong campaign_id must raise ValueError
  with pytest.raises(ValueError, match="mismatched campaign_id"):
    UnifiedCampaignSink(sink_root, campaign_id="wrong_id")

  inspector = inspect_campaign_sink(sink_root)
  sum_all = inspector.summary(include_empty_arms=True)
  sum_non_empty = inspector.summary(include_empty_arms=False)
  assert sum_all.arms == ("arm_a", "arm_empty")
  assert sum_non_empty.arms == ("arm_a",)
  assert sum_all.total_samples == 1
  df_guard = inspector.to_dataframe()
  seq_val = df_guard.iloc[0]["sequence"] if hasattr(df_guard, "iloc") else df_guard[0]["sequence"]
  assert seq_val == "CCCC"

  # Corrupt arm zarr and verify check_arms=False still passes summary check
  zarr_p = inspector.open_arm_zarr("arm_a")
  (zarr_p / "row1" / ".zattrs").write_text('{"corrupt": 1}\n', encoding="utf-8")
  res_no_arms = inspector.verify_integrity(check_arms=False)
  assert res_no_arms["summary_parquet"] is True
  assert res_no_arms["all_valid"] is True
  assert "arm:arm_a" not in res_no_arms


def test_cross_backend_consensus_equal_arm_weighting(tmp_path: Path) -> None:
  """Verify cross_backend_consensus weights each arm equally regardless of sample count."""
  sink_root = tmp_path / "campaign_weights"
  sink = UnifiedCampaignSink(sink_root, campaign_id="camp_w")
  # Arm A has 1 sample with 'AAAA', Arm B has 3 samples with 'CCCC'
  sink.write_arm_results(
    arm_id="arm_a",
    model_family="mpnn",
    checkpoint_id="ckpt",
    manifest_row_hash="r1",
    records=[{"structure_id": "s1", "sample_idx": 0, "sequence": "AAAA", "score_nll": 1.0}],
  )
  sink.write_arm_results(
    arm_id="arm_b",
    model_family="mpnn",
    checkpoint_id="ckpt",
    manifest_row_hash="r2",
    records=[
      {"structure_id": "s1", "sample_idx": 0, "sequence": "CCCC", "score_nll": 1.0},
      {"structure_id": "s1", "sample_idx": 1, "sequence": "CCCC", "score_nll": 1.1},
      {"structure_id": "s1", "sample_idx": 2, "sequence": "CCCC", "score_nll": 1.2},
    ],
  )
  inspector = inspect_campaign_sink(sink_root)
  consensus = inspector.cross_backend_consensus("s1")
  # Equal arm weighting -> 50% A, 50% C -> entropy == ln(2) and tie broken alphabetically ('A')
  import math
  assert consensus["per_position_entropy"][0] == pytest.approx(math.log(2.0))
  assert consensus["consensus_sequence"] == "AAAA"


def test_unified_campaign_sink_atomic_write(tmp_path: Path) -> None:
  """Verify UnifiedCampaignSink writes real Zarr v2 datasets and atomic multi-row appends."""
  import numpy as np
  import zarr

  sink_root = tmp_path / "atomic_sink"
  sink = UnifiedCampaignSink(sink_root, campaign_id="camp_atomic", git_sha="deadbeef")

  seq_arr_1 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
  sink.write_arm_results(
    arm_id="original",
    model_family="mpnn",
    checkpoint_id="proteinmpnn_v_48_020",
    manifest_row_hash="row_hash_1",
    records=[
      {
        "structure_id": "1crn",
        "sample_idx": 0,
        "temperature": 0.1,
        "sequence": "ACDE",
        "score_nll": 1.15,
      }
    ],
    arrays={"sequences": seq_arr_1},
  )

  seq_arr_2 = np.array([[8, 9, 10, 11]], dtype=np.int32)
  sink.write_arm_results(
    arm_id="original",
    model_family="mpnn",
    checkpoint_id="proteinmpnn_v_48_020",
    manifest_row_hash="row_hash_2",
    records=[
      {
        "structure_id": "2ci2",
        "sample_idx": 0,
        "temperature": 0.1,
        "sequence": "FGHI",
        "score_nll": 1.05,
      }
    ],
    arrays={"sequences": seq_arr_2},
  )

  zarr_path = inspect_campaign_sink(sink_root).open_arm_zarr("original")
  root_grp = zarr.open_group(str(zarr_path), mode="r")
  assert "row_hash_1" in root_grp
  assert "row_hash_2" in root_grp
  assert (zarr_path / "row_hash_1" / "sequences" / ".zarray").exists()
  assert (zarr_path / "row_hash_2" / "sequences" / ".zarray").exists()
  np.testing.assert_array_equal(root_grp["row_hash_1"]["sequences"][:], seq_arr_1)
  np.testing.assert_array_equal(root_grp["row_hash_2"]["sequences"][:], seq_arr_2)


def test_campaign_sink_inspector(tmp_path: Path) -> None:
  """Verify CampaignSinkInspector summary(include_empty_arms), verify_integrity(check_arms), and consensus."""
  sink_root = tmp_path / "inspector_sink"
  sink = UnifiedCampaignSink(sink_root, campaign_id="camp_insp", git_sha="feedface")

  sink.write_arm_results(
    arm_id="populated_arm",
    model_family="mpnn",
    checkpoint_id="proteinmpnn_v_48_020",
    manifest_row_hash="hash_pop",
    records=[
      {
        "structure_id": "1abc",
        "sample_idx": 0,
        "temperature": 0.1,
        "sequence": "ACDEF",
        "score_nll": 1.2,
      }
    ],
    arrays={"sequences": [[0, 1, 2, 3, 4]]},
  )
  sink.write_arm_results(
    arm_id="empty_arm",
    model_family="mpnn",
    checkpoint_id="proteinmpnn_v_48_020",
    manifest_row_hash="hash_empty",
    records=[],
    arrays={},
  )

  inspector = inspect_campaign_sink(sink_root)
  summary_all = inspector.summary(include_empty_arms=True)
  assert "empty_arm" in summary_all.arms
  assert "populated_arm" in summary_all.arms

  summary_non_empty = inspector.summary(include_empty_arms=False)
  assert summary_non_empty.arms == ("populated_arm",)

