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

  # Check verify_integrity(check_arms=False) ignores arm corruption before corrupting
  corrupt_file = zarr_p / "hash_fampnn_1" / ".zattrs"
  corrupt_file.write_text('{"corrupted": true}\n', encoding="utf-8")
  integrity_after = inspector.verify_integrity(check_arms=True)
  assert integrity_after["arm:fampnn"] is False
  assert integrity_after["all_valid"] is False
  integrity_no_arms = inspector.verify_integrity(check_arms=False)
  assert "arm:fampnn" not in integrity_no_arms
  assert integrity_no_arms["summary_parquet"] is True
  assert integrity_no_arms["all_valid"] is True


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

