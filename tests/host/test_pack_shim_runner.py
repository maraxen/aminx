"""Runner-level tests for aminx.host.pack_shim.run_packed_manifest_row.

The model call is replaced by a fake worker that writes a small zarr store, so these tests cover
what the `--pack` runner does around the computation: locking, resume, quarantine, failure
stamps and `campaign gates` visibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner
from xtrax.pack import CodecUnavailable, is_packed
from xtrax.run import zarr_content_digest

import aminx.host.pack_shim as shim
from aminx.host.campaign import (
  MANIFEST_SCHEMA_VERSION,
  _acquire_local_fs_lock,
  _collect_manifest_gate_state,
  _done_marker_path,
  _lock_path,
  _write_done_marker,
  compute_unit_input_hash,
)
from aminx.host.pack_shim import (
  PACKED_GATE_DIGEST_PREFIX,
  failure_stamp_path,
  run_packed_manifest_row,
  validate_aminx_unit,
)

ROW_HASH = "packed_runner_row_001"
PDB_PATH = Path("tests/data/1ubq.pdb").resolve()


def _write_manifest(tmp_path: Path, out: Path, *, seed: int = 42, name: str = "manifest.json") -> Path:
  spec = {
    "inputs": [str(PDB_PATH)],
    "random_seed": seed,
    "temperature": [0.1],
    "num_samples": 2,
    "output_h5_path": str(out),
  }
  row = {
    "manifest_row_hash": ROW_HASH,
    "sampling_spec": spec,
    # Fields campaign gates reads to group rows into cells.
    "fixed_policy": "none",
    "state_weight_profile": "equal",
    "multi_state_strategy": "none",
    "temperature": [0.1],
    "backbone_noise": [0.0],
    "ligand_conditioning": False,
    "sidechain_conditioning": False,
    "checkpoint_id": "proteinmpnn_v_48_020",
  }
  path = tmp_path / name
  path.write_text(
    json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, "rows": [row]}),
    encoding="utf-8",
  )
  return path


def _spec(manifest_path: Path) -> dict[str, Any]:
  return json.loads(manifest_path.read_text(encoding="utf-8"))["rows"][0]["sampling_spec"]


def _write_store(path: Path) -> None:
  root = zarr.open_group(str(path), mode="w")
  root.create_array("sequences", data=np.arange(8, dtype=np.int32))


@pytest.fixture
def worker_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
  calls: list[Path] = []

  def _fake_worker(_spec: object, partial_path: Path, *, is_multistate: bool) -> dict[str, Any]:
    del is_multistate
    calls.append(partial_path)
    _write_store(partial_path)
    return {"status": "completed"}

  monkeypatch.setattr(shim, "_run_worker", _fake_worker)
  return calls


def test_packed_row_completes_and_reruns_as_already_done(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)

  first = run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  assert first["status"] == "completed"
  assert first["output_h5_path"] == str(out)
  assert first["input_hash"] == compute_unit_input_hash(_spec(manifest))[0]
  assert is_packed(out)
  assert not _lock_path(out).exists(), "lock released after the row"

  second = run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  assert second["status"] == "already_done"
  assert second["packed"] is True
  assert len(worker_calls) == 1


def test_gates_count_packed_rows(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  run_packed_manifest_row(manifest, row_hash=ROW_HASH)

  state = _collect_manifest_gate_state(manifest)
  assert state.completed_rows == 1
  assert state.metadata_issues == []
  digest = state.digest_by_hash[ROW_HASH]
  assert digest.startswith(PACKED_GATE_DIGEST_PREFIX)
  assert len(worker_calls) == 1


def test_gates_report_a_corrupted_packed_row(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  pack_file = out / "members.tar.zst"
  data = bytearray(pack_file.read_bytes())
  data[len(data) // 2] ^= 0xFF
  pack_file.write_bytes(bytes(data))

  state = _collect_manifest_gate_state(manifest)
  assert state.completed_rows == 0
  assert any(ROW_HASH in issue for issue in state.metadata_issues)

  # Resume treats corruption as a hard error (as the reference runner does) and keeps the unit.
  with pytest.raises((ValueError, OSError)):
    run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  assert out.exists()
  assert len(worker_calls) == 1


def test_changed_inputs_quarantine_the_old_unit(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  run_packed_manifest_row(_write_manifest(tmp_path, out, seed=1), row_hash=ROW_HASH)

  changed = _write_manifest(tmp_path, out, seed=2, name="manifest_seed2.json")
  result = run_packed_manifest_row(changed, row_hash=ROW_HASH)

  assert result["status"] == "completed"
  assert result["reuse"]["reason"] == "input_hash_mismatch"
  moved = Path(result["reuse"]["quarantined"]["output"])
  assert moved.exists(), "old unit moved aside, not deleted"
  assert is_packed(moved)
  validate_aminx_unit(out, expected_input_digest=compute_unit_input_hash(_spec(changed))[0])
  assert len(worker_calls) == 2


def test_existing_output_without_stamp_is_refused_not_deleted(
  tmp_path: Path, worker_calls: list[Path]
) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  _write_store(out)
  before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))

  with pytest.raises(RuntimeError, match="without a unit record or done marker"):
    run_packed_manifest_row(manifest, row_hash=ROW_HASH)

  assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before
  assert worker_calls == []


def test_row_finished_by_reference_runner_is_reused(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  _write_store(out)
  input_hash, components = compute_unit_input_hash(_spec(manifest))
  _write_done_marker(
    marker_path=_done_marker_path(out),
    output_h5_path=out,
    manifest_row_hash=ROW_HASH,
    attempt_id="legacy",
    content_digest_sha256=zarr_content_digest(out),
    lock_backend="local_fs",
    input_hash=input_hash,
    input_hash_components=components,
  )

  result = run_packed_manifest_row(manifest, row_hash=ROW_HASH)

  assert result["status"] == "already_done"
  assert result["reuse"]["reason"] == "stamp_verified"
  assert _done_marker_path(out).exists()
  assert worker_calls == []


def test_active_lock_blocks_a_second_worker(tmp_path: Path, worker_calls: list[Path]) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  out.parent.mkdir(parents=True)
  _acquire_local_fs_lock(
    lock_path=_lock_path(out),
    owner_token="other-worker",
    manifest_row_hash=ROW_HASH,
    attempt_id="other",
    lease_seconds=600,
  )

  with pytest.raises(RuntimeError, match="Active lock already held"):
    run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  assert worker_calls == []
  assert not out.exists()


def test_pack_failure_after_commit_stamps_beside_and_resume_packs(
  tmp_path: Path,
  worker_calls: list[Path],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  real_pack = shim.pack_aminx_unit

  def _no_codec(*_args: object, **_kwargs: object) -> None:
    msg = "no zstd"
    raise CodecUnavailable(msg)

  monkeypatch.setattr(shim, "pack_aminx_unit", _no_codec)
  with pytest.raises(CodecUnavailable):
    run_packed_manifest_row(manifest, row_hash=ROW_HASH)

  stamp = failure_stamp_path(out)
  assert stamp.exists()
  assert json.loads(stamp.read_text(encoding="utf-8"))["meta"]["record_committed"] is True
  assert not (out / stamp.name).exists()
  # The committed loose unit is intact: the stamp is not one of its members.
  validate_aminx_unit(out, expected_input_digest=compute_unit_input_hash(_spec(manifest))[0])

  monkeypatch.setattr(shim, "pack_aminx_unit", real_pack)
  resumed = run_packed_manifest_row(manifest, row_hash=ROW_HASH)
  assert resumed["status"] == "already_done"
  assert resumed["packed"] is True
  assert is_packed(out)
  assert len(worker_calls) == 1


def test_cli_pack_refuses_distributed_lock_backend(tmp_path: Path, worker_calls: list[Path]) -> None:
  from aminx.cli import app

  out = tmp_path / "campaign" / f"{ROW_HASH}.h5"
  manifest = _write_manifest(tmp_path, out)
  result = CliRunner().invoke(
    app,
    [
      "campaign",
      "worker",
      "--manifest-path",
      str(manifest),
      "--row-index",
      "0",
      "--pack",
      "--lock-backend",
      "distributed",
    ],
  )
  assert result.exit_code != 0
  assert worker_calls == []
  assert not out.exists()
