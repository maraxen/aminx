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
