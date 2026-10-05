"""Campaign done markers are v3 stamps that carry an input hash, and reuse is refused when the inputs changed (aminx #2421).

Before this, the manifest row hash omitted the input structures, ``checkpoint_id``, ``random_seed``, the fixed mask and the
bias, and the output path is derived from it. A re-planned or patched manifest therefore resolved to the SAME path and was
reported ``already_done`` even though its output was for a different computation. Each control below fails if that comes back.

Valid v2 markers (digest-verified Zarr, but no inputs recorded) are quarantined and recomputed by default, never deleted,
and ``adopt-legacy`` is the explicit way to keep them. Corruption handling is deliberately unchanged: a digest mismatch
under a valid stamp is still a hard error (tests/host/test_campaign_done_marker.py pins that).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from xtrax.run import zarr_content_digest

from aminx.host import campaign
from aminx.host.campaign import (
    DONE_MARKER_SCHEMA_VERSION,
    LEGACY_DONE_MARKER_SCHEMA_VERSION,
    InputHashUnavailableError,
    _done_marker_path,
    adopt_legacy_done_markers,
    compute_unit_input_hash,
    execute_manifest,
    run_manifest_row,
)

ROW_HASH = "row_hash_for_stamp_v3_tests"


def _write_manifest(tmp_path: Path, **spec_overrides: Any) -> tuple[Path, Path]:
    """One-row manifest. Returns (manifest_path, output_store_path). Mutating the spec re-writes the same row hash."""
    pdb = tmp_path / "structure.pdb"
    if not pdb.exists():
        pdb.write_text("ATOM      1  N   MET A   1       0.000   0.000   0.000\n")
    output = tmp_path / "row_output.zarr"
    spec: dict[str, Any] = {
        "inputs": [str(pdb)],
        "output_h5_path": str(output),
        "random_seed": 1,
        "temperature": [0.5],
    }
    spec.update(spec_overrides)
    manifest = {
        "schema_version": campaign.MANIFEST_SCHEMA_VERSION,
        "rows": [{"manifest_row_hash": ROW_HASH, "job_id": "job0", "job_index": 0, "sampling_spec": spec}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, output


def _fake_sample(spec: Any) -> dict[str, Any]:
    """Stand-in for the real sampler: writes a store whose content depends on the seed, so a recompute is visible."""
    root = zarr.open_group(str(spec.output_h5_path), mode="a")
    arr = root.create_array(name="sequences", shape=(2,), dtype="int32")
    arr[...] = np.array([int(spec.random_seed or 0), 7], dtype=np.int32)
    return {"status": "completed"}


def _run(manifest_path: Path) -> tuple[dict[str, Any], Any]:
    with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
        result = run_manifest_row(str(manifest_path), row_hash=ROW_HASH, lock_backend="local_fs")
    return result, mock_sample


def _marker(output: Path) -> dict[str, Any]:
    return json.loads(_done_marker_path(output).read_text())


def _downgrade_marker_to_v2(output: Path) -> dict[str, Any]:
    """Rewrite a v3 marker the way a v2 build wrote it: same digest, no input hash, v2 schema."""
    marker = _marker(output)
    v2 = {k: marker[k] for k in ("manifest_row_hash", "attempt_id", "output_h5_path", "content_digest_sha256", "lock_backend", "completed_at_unix_s")}
    v2["schema_version"] = LEGACY_DONE_MARKER_SCHEMA_VERSION
    _done_marker_path(output).write_text(json.dumps(v2))
    return v2


def _superseded(output: Path) -> list[Path]:
    return sorted(p for p in output.parent.iterdir() if ".superseded." in p.name and not p.name.endswith(".done.json") and p.is_dir())


# --- the stamp ----------------------------------------------------------------------------------------------------------------------


def test_first_run_writes_a_v3_stamp_with_the_input_hash(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    result, mock_sample = _run(manifest)
    mock_sample.assert_called_once()
    assert result["status"] == "completed"
    assert result["reuse"]["decision"] == "computed"

    marker = _marker(output)
    assert marker["schema_version"] == DONE_MARKER_SCHEMA_VERSION == "campaign_done_marker_v3"
    assert marker["input_hash"] == result["input_hash"]
    assert marker["content_digest_sha256"] == zarr_content_digest(output)
    assert set(marker["input_hash_components"]) >= {"spec_sha256", "inputs", "checkpoint", "versions"}
    assert marker["producer"]["aminx"] and marker["producer"]["xtrax"]
    assert marker["digest_algo"].startswith("xtrax.run.zarr_content_digest@")
    assert "adopted_from_v2" not in marker


def test_untouched_unit_is_reused_and_reports_its_hashes(tmp_path):
    """Positive control: the same inputs are reused, so the mismatch controls below can genuinely fail."""
    manifest, output = _write_manifest(tmp_path)
    first, _ = _run(manifest)
    second, mock_sample = _run(manifest)
    mock_sample.assert_not_called()
    assert second["status"] == "already_done"
    reuse = second["reuse"]
    assert reuse["decision"] == "reused"
    assert reuse["input_hash"] == first["input_hash"]
    assert reuse["artifact_digest"] == zarr_content_digest(output)
    assert reuse["stamp_attempt_id"] == first["attempt_id"]
    assert reuse["adopted_from_v2"] is False
    assert _superseded(output) == []


# --- must-fail controls: a changed computation must NOT be reused --------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"random_seed": 2}, id="seed"),
        pytest.param({"temperature": [0.1]}, id="temperature"),
        pytest.param({"bias": [[0.0] * 21]}, id="bias"),
        pytest.param({"fixed_mask": [1, 0, 1]}, id="fixed_mask"),
        pytest.param({"checkpoint_id": "some_other_checkpoint"}, id="checkpoint_id"),
    ],
)
def test_changed_spec_field_under_the_same_row_hash_is_recomputed(tmp_path, change):
    manifest, output = _write_manifest(tmp_path)
    first, _ = _run(manifest)
    digest_before = zarr_content_digest(output)

    # Same manifest_row_hash, same output path, different computation (the exact hole the row hash leaves open).
    _write_manifest(tmp_path, **change)
    with patch("aminx.host.campaign._checkpoint_identity", side_effect=lambda p: {"route": "stub", "id": p.get("checkpoint_id")}):
        with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
            second = run_manifest_row(str(manifest), row_hash=ROW_HASH, lock_backend="local_fs")

    assert second["status"] == "completed", "a changed spec must not be reported already_done"
    mock_sample.assert_called_once()
    assert second["reuse"]["decision"] == "recomputed"
    assert second["reuse"]["reason"] == "input_hash_mismatch"
    assert second["reuse"]["stamped_input_hash"] == first["input_hash"]
    assert _marker(output)["input_hash"] == second["input_hash"] != first["input_hash"]
    # the old output was kept aside, not deleted, and is byte-for-byte what was computed before
    kept = _superseded(output)
    assert len(kept) == 1
    assert zarr_content_digest(kept[0]) == digest_before


def test_changed_input_file_content_is_recomputed(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    first, _ = _run(manifest)
    (tmp_path / "structure.pdb").write_text("ATOM      1  N   MET A   1       9.999   9.999   9.999\n")  # same path, new content
    second, mock_sample = _run(manifest)
    mock_sample.assert_called_once()
    assert second["reuse"]["reason"] == "input_hash_mismatch"
    assert second["input_hash"] != first["input_hash"]


def test_changed_weights_file_is_recomputed(tmp_path):
    weights = tmp_path / "weights.eqx.zst"
    weights.write_bytes(b"weights version one")
    manifest, output = _write_manifest(tmp_path, model_local_path=str(weights))
    first, _ = _run(manifest)
    assert _run(manifest)[1].call_count == 0, "control: identical weights are reused"

    weights.write_bytes(b"weights version two, different bytes")
    second, mock_sample = _run(manifest)
    mock_sample.assert_called_once()
    assert second["reuse"]["reason"] == "input_hash_mismatch"
    assert second["input_hash"] != first["input_hash"]


def test_corrupted_store_under_a_valid_stamp_is_still_a_hard_error(tmp_path):
    """Unchanged policy: real corruption is not silently recomputed (the existing tests pin this for the primitive)."""
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    zarr.open_group(str(output), mode="a")["sequences"][...] = np.array([999, 999], dtype=np.int32)
    with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
        with pytest.raises(ValueError, match="content digest mismatch"):
            run_manifest_row(str(manifest), row_hash=ROW_HASH, lock_backend="local_fs")
    mock_sample.assert_not_called()
    assert _superseded(output) == []


def test_flipped_digest_byte_in_the_stamp_is_still_a_hard_error(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    marker = _marker(output)
    d = marker["content_digest_sha256"]
    marker["content_digest_sha256"] = ("0" if d[0] != "0" else "1") + d[1:]
    _done_marker_path(output).write_text(json.dumps(marker))
    with patch("aminx.host.campaign.sample", side_effect=_fake_sample):
        with pytest.raises(ValueError, match="content digest mismatch"):
            run_manifest_row(str(manifest), row_hash=ROW_HASH, lock_backend="local_fs")


def test_unresolvable_inputs_stop_the_run_and_touch_nothing(tmp_path):
    """If the inputs cannot be hashed (e.g. no weights on this node) reuse cannot be judged: stop, do not discard."""
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    digest = zarr_content_digest(output)
    with patch("aminx.host.campaign._checkpoint_identity", side_effect=InputHashUnavailableError("no weights here")):
        with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
            with pytest.raises(InputHashUnavailableError, match="no weights here"):
                run_manifest_row(str(manifest), row_hash=ROW_HASH, lock_backend="local_fs")
    mock_sample.assert_not_called()
    assert zarr_content_digest(output) == digest
    assert _superseded(output) == []
    assert _done_marker_path(output).exists()


# --- legacy v2 markers ----------------------------------------------------------------------------------------------------------------


def test_valid_v2_marker_is_quarantined_not_deleted_and_recomputed(tmp_path, caplog):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    digest_before = zarr_content_digest(output)
    _downgrade_marker_to_v2(output)

    with caplog.at_level(logging.WARNING, logger="aminx.host.campaign"):
        second, mock_sample = _run(manifest)

    mock_sample.assert_called_once()
    assert second["reuse"]["decision"] == "recomputed"
    assert second["reuse"]["reason"] == "legacy_v2_marker"
    assert any("adopt-legacy" in r.getMessage() for r in caplog.records), "the warning must point at the way to keep the output"
    kept = _superseded(output)
    assert len(kept) == 1, "a valid v2 output must be kept aside, not rmtree'd"
    assert zarr_content_digest(kept[0]) == digest_before
    assert _marker(output)["schema_version"] == DONE_MARKER_SCHEMA_VERSION


def test_adopt_legacy_keeps_a_verified_v2_output_and_flags_it(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    v2 = _downgrade_marker_to_v2(output)
    digest = zarr_content_digest(output)

    dry = adopt_legacy_done_markers(manifest, dry_run=True)
    assert dry["adopted_rows"] == 1 and dry["dry_run"] is True
    assert json.loads(_done_marker_path(output).read_text())["schema_version"] == LEGACY_DONE_MARKER_SCHEMA_VERSION, "dry run changes nothing"

    report = adopt_legacy_done_markers(manifest)
    assert report["schema_version"] == "campaign_adopt_legacy_report_v1"
    assert report["adopted_rows"] == 1 and report["skipped_rows"] == 0
    marker = _marker(output)
    assert marker["schema_version"] == DONE_MARKER_SCHEMA_VERSION
    assert marker["adopted_from_v2"] is True
    assert marker["adopted_from"] == v2
    assert marker["content_digest_sha256"] == digest
    assert marker["completed_at_unix_s"] == v2["completed_at_unix_s"], "the original completion time is preserved"
    assert Path(report["rows"][0]["backup"]).exists(), "the v2 marker is kept as a .v2.bak"

    second, mock_sample = _run(manifest)
    mock_sample.assert_not_called()
    assert second["reuse"]["decision"] == "reused"
    assert second["reuse"]["adopted_from_v2"] is True
    assert _superseded(output) == []


def test_adopt_legacy_refuses_a_v2_output_whose_digest_no_longer_matches(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    _downgrade_marker_to_v2(output)
    zarr.open_group(str(output), mode="a")["sequences"][...] = np.array([5, 5], dtype=np.int32)
    report = adopt_legacy_done_markers(manifest)
    assert report["adopted_rows"] == 0
    assert report["rows"][0]["skipped"] == "artifact_digest_mismatch"
    assert json.loads(_done_marker_path(output).read_text())["schema_version"] == LEGACY_DONE_MARKER_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param(lambda out: None, "already_v3", id="already_v3"),
        pytest.param(lambda out: _done_marker_path(out).unlink(), "no_marker", id="no_marker"),
        pytest.param(
            lambda out: _done_marker_path(out).write_text(json.dumps({**_marker(out), "schema_version": "campaign_done_marker_v1"})),
            "not_v2:campaign_done_marker_v1",
            id="v1",
        ),
    ],
)
def test_adopt_legacy_only_touches_v2_markers(tmp_path, setup, expected):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    setup(output)
    before = _done_marker_path(output).read_text() if _done_marker_path(output).exists() else None
    report = adopt_legacy_done_markers(manifest)
    assert report["rows"][0]["skipped"] == expected
    after = _done_marker_path(output).read_text() if _done_marker_path(output).exists() else None
    assert after == before


def test_adopt_legacy_skips_a_marker_for_a_different_row(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    v2 = _downgrade_marker_to_v2(output)
    v2["manifest_row_hash"] = "some_other_row"
    _done_marker_path(output).write_text(json.dumps(v2))
    assert adopt_legacy_done_markers(manifest)["rows"][0]["skipped"] == "manifest_row_hash_mismatch"


def test_adopt_legacy_cli_runs_and_honours_dry_run(tmp_path, capsys):
    manifest, output = _write_manifest(tmp_path)
    _run(manifest)
    _downgrade_marker_to_v2(output)
    summary = tmp_path / "adopt.json"
    assert campaign.main(["adopt-legacy", "--manifest-path", str(manifest), "--dry-run", "--summary-path", str(summary)]) == 0
    assert json.loads(summary.read_text())["adopted_rows"] == 1
    assert json.loads(_done_marker_path(output).read_text())["schema_version"] == LEGACY_DONE_MARKER_SCHEMA_VERSION


# --- the report ---------------------------------------------------------------------------------------------------------------------------


def test_execute_manifest_reports_reuse_and_recompute_counts(tmp_path):
    manifest, output = _write_manifest(tmp_path)
    with patch("aminx.host.campaign.sample", side_effect=_fake_sample):
        first = execute_manifest(manifest)
        digest_after_first = zarr_content_digest(output)
        second = execute_manifest(manifest)
        _write_manifest(tmp_path, random_seed=99)
        third = execute_manifest(manifest)
    assert (first["reused_rows"], first["recomputed_rows"]) == (0, 0)
    assert (second["reused_rows"], second["recomputed_rows"]) == (1, 0)
    assert (third["reused_rows"], third["recomputed_rows"]) == (0, 1)
    assert second["row_results"][0]["reuse"]["artifact_digest"] == digest_after_first
    assert third["row_results"][0]["reuse"]["reason"] == "input_hash_mismatch"
    assert third["row_results"][0]["reuse"]["quarantined"]["output"] is not None
    assert third["schema_version"] == "campaign_execute_report_v1", "additive change: existing report consumers keep working"


# --- the hash itself ----------------------------------------------------------------------------------------------------------------------


def test_input_hash_ignores_the_output_path_and_dict_order(tmp_path):
    pdb = tmp_path / "s.pdb"
    pdb.write_text("x")
    a = {"inputs": [str(pdb)], "output_h5_path": "/a/b.zarr", "random_seed": 1, "temperature": [0.5]}
    b = {"temperature": [0.5], "random_seed": 1, "output_h5_path": "/completely/elsewhere.zarr", "inputs": [str(pdb)]}
    assert compute_unit_input_hash(a)[0] == compute_unit_input_hash(b)[0]


def test_input_hash_changes_with_every_component(tmp_path, monkeypatch):
    pdb = tmp_path / "s.pdb"
    pdb.write_text("one")
    base = {"inputs": [str(pdb)], "output_h5_path": "x", "random_seed": 1}
    h0, comps = compute_unit_input_hash(base)
    assert comps["inputs"][0]["kind"] == "file" and len(comps["inputs"][0]["sha256"]) == 64
    assert compute_unit_input_hash({**base, "random_seed": 2})[0] != h0
    pdb.write_text("two")
    assert compute_unit_input_hash(base)[0] != h0, "input file content"
    pdb.write_text("one")
    assert compute_unit_input_hash(base)[0] == h0, "control: restoring the content restores the hash"
    monkeypatch.setattr(campaign, "SAMPLING_NUMERICS_EPOCH", campaign.SAMPLING_NUMERICS_EPOCH + 1)
    assert compute_unit_input_hash(base)[0] != h0, "numerics epoch"


def test_input_hash_handles_directories_and_unresolvable_entries(tmp_path):
    d = tmp_path / "structs"
    d.mkdir()
    (d / "a.pdb").write_text("a")
    (d / "b.pdb").write_text("b")
    h1, comps = compute_unit_input_hash({"inputs": [str(d)], "output_h5_path": "x"})
    assert comps["inputs"][0]["kind"] == "dir" and comps["inputs"][0]["files"] == 2
    (d / "b.pdb").write_text("changed")
    assert compute_unit_input_hash({"inputs": [str(d)], "output_h5_path": "x"})[0] != h1

    _, comps2 = compute_unit_input_hash({"inputs": ["1ubq", "/no/such/file.pdb"], "output_h5_path": "x"})
    assert [e["kind"] for e in comps2["inputs"]] == ["unresolved", "unresolved"]
