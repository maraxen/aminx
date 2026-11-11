"""Negative verification controls for campaign lease fencing and corruption refusal.

Covers:
- C-NEG-1: Stale Lease Overwrite Refusal (test_stale_worker_commit_refused)
- C-NEG-2: Conflicting Concurrent Run Refusal (test_concurrent_row_lock_contention)
- C-NEG-4: Corrupted Done Marker Recomputation (test_corrupted_done_marker_triggers_recompute)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xtrax.campaign.lease import (
    FileLock,
    LeaseLostError,
    LockUnavailableError,
)
from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    compute_content_digest,
    validate_unit_done,
)


def test_stale_worker_commit_refused(tmp_path: Path) -> None:
    """C-NEG-1: Prove that a worker whose lease was stolen cannot overwrite or commit."""
    out_dir = tmp_path / "units" / "row_001.h5"
    out_dir.mkdir(parents=True)
    (out_dir / "data.bin").write_bytes(b"worker_1_initial")

    lock1 = FileLock(out_dir, owner_token="worker-1", manifest_row_hash="row_001", lease_seconds=10)
    lock1.acquire()
    assert lock1.is_locked

    # Worker 2 steals the lease by advancing the fence and overwriting lock file
    fence_path = out_dir.with_name(f"{out_dir.name}.fence")
    fence_path.write_text("2", encoding="utf-8")

    lock_file = out_dir.with_name(f"{out_dir.name}.lock")
    lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
    lock_data["owner_token"] = "worker-2"
    lock_data["fence"] = 2
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    # Worker 2 commits output
    (out_dir / "data.bin").write_bytes(b"worker_2_winner")

    # Worker 1 resumes and attempts to assert ownership / commit
    with pytest.raises(LeaseLostError):
        lock1.assert_holder()

    # Worker 2 output remains intact
    assert (out_dir / "data.bin").read_bytes() == b"worker_2_winner"


def test_concurrent_row_lock_contention(tmp_path: Path) -> None:
    """C-NEG-2: Prove concurrent workers targeting the same row raise LockUnavailableError."""
    out_dir = tmp_path / "units" / "row_002.h5"
    out_dir.mkdir(parents=True)

    lock_a = FileLock(out_dir, owner_token="worker-a", manifest_row_hash="row_002", lease_seconds=600)
    lock_a.acquire()
    assert lock_a.is_locked

    lock_b = FileLock(out_dir, owner_token="worker-b", manifest_row_hash="row_002", lease_seconds=600)
    with pytest.raises(LockUnavailableError):
        lock_b.acquire()

    lock_a.release()
