"""Tests for aminx host/pack_shim.py integration with xtrax.pack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.pack import (
  CodecUnavailable,
  UnitPackRefused,
  is_packed,
  list_unit_members,
  read_unit_member,
  record_problem,
  unit_content_digest,
)
from xtrax.run import zarr_content_digest
from xtrax.run.unit_index_adapters import AminxDoneMarkerAdapter, UnitJsonAdapter

from aminx.host.pack_shim import (
  UNIT_FAILED_STAMP,
  aminx_unit_check,
  failure_stamp_path,
  pack_aminx_unit,
  packed_unit_gate_digest,
  resolve_unit_adapter,
  validate_aminx_unit,
  write_aminx_unit,
  write_aminx_unit_failure,
)


# Identity for units built outside a manifest row; write_aminx_unit refuses to invent one.
_DIGEST = "a" * 64


def _make_store(unit_dir: Path) -> Path:
  unit_dir.mkdir(parents=True, exist_ok=True)
  root = zarr.open_group(str(unit_dir), mode="a")
  arr = root.create_array(name="data", shape=(4,), dtype="int32")
  arr[...] = np.array([10, 20, 30, 40], dtype=np.int32)
  return unit_dir


def test_unit_creation_packing_and_member_readback(tmp_path: Path) -> None:
  """Test 1: Unit creation, packing, and byte-identical member readback."""
  unit_dir = tmp_path / "unit_1"
  _make_store(unit_dir)

  record = write_aminx_unit(
    unit_dir,
    input_digest=_DIGEST,
    run_id="run-test-1",
    meta={"experiment": "test_pack"},
  )
  assert record["schema"] == "xtrax.unit/1"
  assert record["run_id"] == "run-test-1"
  pre_pack_digest = unit_content_digest(unit_dir)
  assert record["content_digest"] == pre_pack_digest

  # Find an array chunk file to compare before and after packing
  chunk_files = [p for p in unit_dir.rglob("*") if p.is_file() and p.name not in ("unit.json", ".zgroup", "zarr.json")]
  assert chunk_files, "Expected chunk files in zarr store"
  target_chunk = chunk_files[0]
  rel_chunk_path = target_chunk.relative_to(unit_dir).as_posix()
  raw_bytes_before = target_chunk.read_bytes()

  # Pack unit
  pack_info = pack_aminx_unit(unit_dir)
  assert is_packed(unit_dir)
  assert (unit_dir / "members.tar.zst").exists()
  assert (unit_dir / "members.idx.json").exists()

  # Loose chunk file should now be packed
  assert not target_chunk.exists()

  # Content digest remains invariant across pack
  post_pack_digest = unit_content_digest(unit_dir)
  assert post_pack_digest == pre_pack_digest

  # Member list includes the relative chunk path
  members = list_unit_members(unit_dir)
  assert rel_chunk_path in members

  # Member readback is byte-identical
  raw_bytes_after = read_unit_member(unit_dir, rel_chunk_path)
  assert raw_bytes_after == raw_bytes_before

  # validate_aminx_unit succeeds on packed unit
  resumed_record = validate_aminx_unit(unit_dir)
  assert resumed_record["content_digest"] == post_pack_digest


def test_reader_backward_compatibility_aminx_done_marker_adapter(tmp_path: Path) -> None:
  """Test 2: Reader backward compatibility with campaign_done_marker_v3."""
  store_dir = tmp_path / "hash_row_123.h5"
  _make_store(store_dir)

  marker_path = store_dir.with_name(f"{store_dir.name}.done.json")
  digest = zarr_content_digest(store_dir)
  input_hash = "f" * 64
  marker_data = {
    "schema_version": "campaign_done_marker_v3",
    "manifest_row_hash": "hash_row_123",
    "attempt_id": "attempt_01",
    "content_digest_sha256": digest,
    "input_hash": input_hash,
  }
  marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

  # Adapter resolution for loose .done.json returns AminxDoneMarkerAdapter
  adapter = resolve_unit_adapter(store_dir)
  assert isinstance(adapter, AminxDoneMarkerAdapter)

  # Find records
  found = list(adapter.find(tmp_path))
  assert len(found) == 1
  locator, core = found[0]
  assert locator == "hash_row_123.h5"
  assert core.unit_key == "hash_row_123"
  assert core.input_digest == input_hash
  assert core.content_digest == digest

  # Classify returns IndexReuse when digests match
  verdict = adapter.classify(tmp_path, locator, input_hash, verify=True)
  assert verdict.__class__.__name__ == "IndexReuse"


def test_negative_control_bit_corruption_raises_value_error(tmp_path: Path) -> None:
  """Negative Control 1: Bit corruption in store rejects resume and raises ValueError."""
  unit_dir = tmp_path / "unit_corrupted"
  _make_store(unit_dir)

  record = write_aminx_unit(unit_dir, input_digest=_DIGEST, run_id="run-corrupt")
  # Pre-corruption validation passes
  validate_aminx_unit(unit_dir)

  # Mutate data chunk (single bit/value corruption)
  root = zarr.open_group(str(unit_dir), mode="a")
  root["data"][...] = np.array([999, 999, 999, 999], dtype=np.int32)

  with pytest.raises(ValueError, match="content digest mismatch"):
    validate_aminx_unit(unit_dir)


def test_negative_control_loose_file_collision_refuses_pack(tmp_path: Path) -> None:
  """Negative Control 2: Loose file collision (R-U3) refuses packing."""
  # Part A: Pack artifact collision before record creation
  artifact_dir = tmp_path / "unit_pre_artifact_collision"
  _make_store(artifact_dir)
  (artifact_dir / "members.tar.zst").write_bytes(b"pseudo archive")
  with pytest.raises(UnitPackRefused, match="R-U3"):
    write_aminx_unit(artifact_dir, input_digest=_DIGEST)

  # Part B: Regenerating loose files in an already packed unit
  unit_dir = tmp_path / "unit_collision"
  _make_store(unit_dir)

  write_aminx_unit(unit_dir, input_digest=_DIGEST)
  pack_aminx_unit(unit_dir)
  assert is_packed(unit_dir)

  # Simulate invalid rerun regenerating loose array files beside archive
  loose_file = unit_dir / "rogue_loose_chunk.bin"
  loose_file.write_bytes(b"unexpected loose file")

  with pytest.raises(UnitPackRefused, match="R-U3"):
    pack_aminx_unit(unit_dir)


def test_negative_control_empty_unit_refusal(tmp_path: Path) -> None:
  """Negative Control 3: Empty unit (R-U1) refuses packing when no array chunks exist."""
  empty_dir = tmp_path / "empty_unit"
  empty_dir.mkdir(parents=True, exist_ok=True)

  # Direct callback refusal
  with pytest.raises(UnitPackRefused, match="R-U1"):
    aminx_unit_check(empty_dir)

  # write_aminx_unit refuses empty unit before committing
  with pytest.raises(UnitPackRefused, match="R-U1"):
    write_aminx_unit(empty_dir, input_digest=_DIGEST)


def test_quarantining_legacy_done_marker_upon_packing(tmp_path: Path) -> None:
  """Test 4: legacy .done.json is archived at commit, renamed after a pack, record still verifies."""
  unit_dir = tmp_path / "row_456.h5"
  _make_store(unit_dir)

  legacy_marker = unit_dir.with_name(f"{unit_dir.name}.done.json")
  legacy_payload = {
    "schema_version": "campaign_done_marker_v3",
    "manifest_row_hash": "row_456",
    "attempt_id": "legacy_att",
  }
  legacy_marker.write_text(json.dumps(legacy_payload), encoding="utf-8")

  committed = write_aminx_unit(unit_dir, input_digest=_DIGEST, run_id="run-quarantine")
  assert committed["meta"]["legacy_done_marker"] == legacy_payload
  assert legacy_marker.exists()

  pack_aminx_unit(unit_dir)

  assert not legacy_marker.exists()
  packed_marker = unit_dir.with_name(f"{unit_dir.name}.done.json.packed")
  assert packed_marker.exists()

  record = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
  assert record["meta"]["legacy_done_marker"] == legacy_payload
  # The record was never edited after commit, so its digest still verifies and readers accept it.
  assert record_problem(record) is None
  validate_aminx_unit(unit_dir, expected_input_digest=_DIGEST)

  adapter = resolve_unit_adapter(unit_dir)
  assert isinstance(adapter, UnitJsonAdapter)
  verdict = adapter.classify(tmp_path, unit_dir.name, _DIGEST, verify=True)
  assert verdict.__class__.__name__ == "IndexReuse"
  assert [locator for locator, _ in adapter.find(tmp_path)] == [unit_dir.name]


def test_failed_pack_leaves_legacy_marker_and_record_intact(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A pack that fails must not rename the legacy marker or leave a record that fails to verify."""
  import aminx.host.pack_shim as shim

  unit_dir = tmp_path / "row_789.h5"
  _make_store(unit_dir)
  legacy_marker = unit_dir.with_name(f"{unit_dir.name}.done.json")
  legacy_marker.write_text(json.dumps({"schema_version": "campaign_done_marker_v3"}), encoding="utf-8")
  write_aminx_unit(unit_dir, input_digest=_DIGEST)

  def _no_codec(*_args: object, **_kwargs: object) -> None:
    msg = "no zstd"
    raise CodecUnavailable(msg)

  monkeypatch.setattr(shim, "pack_unit", _no_codec)
  with pytest.raises(CodecUnavailable):
    pack_aminx_unit(unit_dir)

  assert legacy_marker.exists()
  assert not is_packed(unit_dir)
  validate_aminx_unit(unit_dir, expected_input_digest=_DIGEST)


def test_negative_control_member_changed_after_commit_refuses_pack(tmp_path: Path) -> None:
  """The pack check keeps xtrax's content-digest check: a member edited after commit is refused."""
  unit_dir = tmp_path / "unit_tampered"
  _make_store(unit_dir)
  write_aminx_unit(unit_dir, input_digest=_DIGEST)

  chunk = next(
    p for p in unit_dir.rglob("*") if p.is_file() and p.name not in ("unit.json", "zarr.json")
  )
  chunk.write_bytes(b"tampered")

  with pytest.raises(UnitPackRefused, match="content_digest"):
    pack_aminx_unit(unit_dir)
  assert not is_packed(unit_dir)


def test_negative_control_edited_record_refuses_pack(tmp_path: Path) -> None:
  """A record whose core was edited after commit no longer verifies and is refused at pack."""
  unit_dir = tmp_path / "unit_edited_record"
  _make_store(unit_dir)
  write_aminx_unit(unit_dir, input_digest=_DIGEST)
  record_path = unit_dir / "unit.json"
  record = json.loads(record_path.read_text(encoding="utf-8"))
  record["meta"]["late_edit"] = True
  record_path.write_text(json.dumps(record), encoding="utf-8")

  with pytest.raises(UnitPackRefused, match="record_digest"):
    pack_aminx_unit(unit_dir)


def test_write_aminx_unit_requires_identity(tmp_path: Path) -> None:
  """Without a spec payload or an explicit input_digest there is no identity to record."""
  unit_dir = tmp_path / "unit_no_identity"
  _make_store(unit_dir)
  with pytest.raises(ValueError, match="input_digest"):
    write_aminx_unit(unit_dir)
  assert not (unit_dir / "unit.json").exists()


def test_write_aminx_unit_failure_stamp(tmp_path: Path) -> None:
  """Test writing structured unit failure stamp."""
  unit_dir = tmp_path / "unit_failure"
  error = RuntimeError("Simulated failure in worker execution")
  stamp_path = write_aminx_unit_failure(
    unit_dir,
    exc=error,
    attempt_id="att_fail_99",
    meta={"stage": "sampling"},
  )

  assert stamp_path.exists()
  # The stamp sits beside the unit: inside it, it would become a member of a committed unit.
  assert stamp_path == failure_stamp_path(unit_dir)
  assert stamp_path.name == f"{unit_dir.name}.{UNIT_FAILED_STAMP}"
  assert not unit_dir.exists()

  data = json.loads(stamp_path.read_text(encoding="utf-8"))
  assert data["schema"] == "xtrax.unit_failure/1"
  assert data["attempt_id"] == "att_fail_99"
  assert data["error_type"] == "RuntimeError"
  assert "Simulated failure" in data["error_message"]
  assert data["meta"]["stage"] == "sampling"


def test_ac10_manifest_slice_parity(tmp_path: Path) -> None:
  """AC-10: Parity verification between campaign.py:run_manifest_row and pack_shim.py:run_packed_manifest_row."""
  from unittest.mock import patch

  import equinox as eqx
  import jax

  from aminx.host.campaign import (
    MANIFEST_SCHEMA_VERSION,
    compute_unit_input_hash,
    run_manifest_row,
  )
  from aminx.host.pack_shim import run_packed_manifest_row
  from aminx.model import Aminx

  model = Aminx(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(42),
  )
  synth_model = eqx.tree_inference(model, value=True)

  pdb_path = Path("tests/data/1ubq.pdb").resolve()
  assert pdb_path.exists(), "Required fixture 1ubq.pdb missing"

  # Build reference manifest row
  ref_out = tmp_path / "ref_unit.h5"
  ref_manifest_path = tmp_path / "ref_manifest.json"
  row_hash = "ac10_parity_hash_001"
  sampling_spec = {
    "inputs": [str(pdb_path)],
    "random_seed": 42,
    "temperature": [0.1],
    "num_samples": 2,
    "return_logits": True,
    "output_h5_path": str(ref_out),
  }
  ref_manifest_payload = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "rows": [
      {
        "manifest_row_hash": row_hash,
        "sampling_spec": sampling_spec,
      }
    ],
  }
  ref_manifest_path.write_text(json.dumps(ref_manifest_payload), encoding="utf-8")

  # Build packed manifest row (same spec, separate output path)
  packed_out = tmp_path / "packed_unit.h5"
  packed_manifest_path = tmp_path / "packed_manifest.json"
  packed_spec = dict(sampling_spec)
  packed_spec["output_h5_path"] = str(packed_out)
  packed_manifest_payload = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "rows": [
      {
        "manifest_row_hash": row_hash,
        "sampling_spec": packed_spec,
      }
    ],
  }
  packed_manifest_path.write_text(json.dumps(packed_manifest_payload), encoding="utf-8")

  # 1. Execute reference runner (campaign.py:run_manifest_row)
  with patch("aminx.host.prep.load_model", return_value=synth_model):
    res_ref = run_manifest_row(
      manifest_path=str(ref_manifest_path),
      row_hash=row_hash,
      lock_backend="local_fs",
    )

  # 2. Execute packed runner (pack_shim.py:run_packed_manifest_row)
  with patch("aminx.host.prep.load_model", return_value=synth_model):
    res_pack = run_packed_manifest_row(
      manifest_path=str(packed_manifest_path),
      row_hash=row_hash,
      pack=True,
    )

  assert res_ref["status"] == "completed"
  assert res_pack["status"] == "completed"
  assert ref_out.exists()
  assert packed_out.exists()
  assert is_packed(packed_out)

  # 3. Verify numeric parity across outputs
  members = list_unit_members(packed_out)
  unpacked_dir = tmp_path / "unpacked_packed_unit"
  unpacked_dir.mkdir()
  for m in members:
    raw_bytes = read_unit_member(packed_out, m)
    dest = unpacked_dir / m
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw_bytes)

  ref_store = zarr.open_group(str(ref_out), mode="r")
  pack_store = zarr.open_group(str(unpacked_dir), mode="r")
  assert list(ref_store.keys()) == list(pack_store.keys())

  for k in ref_store.keys():
    ref_sub = ref_store[k]
    pack_sub = pack_store[k]
    if hasattr(ref_sub, "keys"):
      assert set(ref_sub.keys()) == set(pack_sub.keys())
      for arr_name in ref_sub.keys():
        arr_ref = np.asarray(ref_sub[arr_name])
        arr_pack = np.asarray(pack_sub[arr_name])
        np.testing.assert_array_equal(arr_ref, arr_pack)
    else:
      arr_ref = np.asarray(ref_sub)
      arr_pack = np.asarray(pack_sub)
      np.testing.assert_array_equal(arr_ref, arr_pack)

  # 4. Verify dual-mode reader index adapters
  ref_adapter = resolve_unit_adapter(ref_out)
  assert isinstance(ref_adapter, AminxDoneMarkerAdapter)
  packed_adapter = resolve_unit_adapter(packed_out)
  assert isinstance(packed_adapter, UnitJsonAdapter)

  input_hash, _ = compute_unit_input_hash(sampling_spec)
  ref_classification = ref_adapter.classify(tmp_path, ref_out.name, input_hash, verify=True)
  assert ref_classification.__class__.__name__ == "IndexReuse"

  packed_classification = packed_adapter.classify(tmp_path, packed_out.name, input_hash, verify=True)
  assert packed_classification.__class__.__name__ == "IndexReuse"

  # 5. Verify resume parity short-circuiting
  with patch("aminx.host.prep.load_model", return_value=synth_model):
    res_ref_2 = run_manifest_row(
      manifest_path=str(ref_manifest_path),
      row_hash=row_hash,
      lock_backend="local_fs",
    )
    res_pack_2 = run_packed_manifest_row(
      manifest_path=str(packed_manifest_path),
      row_hash=row_hash,
      pack=True,
    )

  assert res_ref_2["status"] == "already_done"
  assert res_pack_2["status"] == "already_done"


def test_custom_record_name_roundtrip(tmp_path: Path) -> None:
  """Test custom record_name roundtrip (write, pack, read member, validate, negative control)."""
  unit_dir = tmp_path / "custom_unit"
  _make_store(unit_dir)
  custom_record = "custom_record.json"

  # Write unit record under custom name
  record = write_aminx_unit(
    unit_dir,
    input_digest=_DIGEST,
    run_id="run-custom-1",
    record_name=custom_record,
    meta={"custom": "field"},
  )
  assert (unit_dir / custom_record).exists()
  assert not (unit_dir / "unit.json").exists(), "unit.json must not leak when custom record_name is specified"
  assert record["schema"] == "xtrax.unit/1"
  assert record["run_id"] == "run-custom-1"

  digest_before = unit_content_digest(unit_dir, record_name=custom_record)
  assert record["content_digest"] == digest_before

  # Negative control: duplicate record write refusal (R-U1)
  with pytest.raises(UnitPackRefused, match="already has a record"):
    write_aminx_unit(
      unit_dir,
      input_digest=_DIGEST,
      run_id="run-custom-2",
      record_name=custom_record,
    )

  # Identify chunk file before packing
  chunk_files = [
    p
    for p in unit_dir.rglob("*")
    if p.is_file() and p.name not in (custom_record, ".zgroup", "zarr.json")
  ]
  assert chunk_files, "Expected chunk files in store"
  target_chunk = chunk_files[0]
  rel_chunk_path = target_chunk.relative_to(unit_dir).as_posix()
  raw_bytes_before = target_chunk.read_bytes()

  # Pack unit with custom record_name
  pack_info = pack_aminx_unit(unit_dir, record_name=custom_record)
  assert pack_info["storage"]["n_members"] >= 1
  assert is_packed(unit_dir, record_name=custom_record)
  assert not is_packed(unit_dir, record_name="unit.json"), "Default record_name must not see custom pack"
  assert (unit_dir / "members.tar.zst").exists()
  assert (unit_dir / "members.idx.json").exists()
  assert (unit_dir / custom_record).exists()
  assert not (unit_dir / "unit.json").exists()

  # Member readback byte-identical
  raw_bytes_after = read_unit_member(unit_dir, rel_chunk_path, record_name=custom_record)
  assert raw_bytes_after == raw_bytes_before

  # validate_aminx_unit succeeds with custom record_name
  resumed_record = validate_aminx_unit(unit_dir, record_name=custom_record)
  assert resumed_record["content_digest"] == digest_before

  # Negative control: validating with default record_name raises ValueError
  with pytest.raises(ValueError, match="Unit record missing"):
    validate_aminx_unit(unit_dir, record_name="unit.json")

  # No xtrax adapter reads a custom record name (UnitJsonAdapter reads only unit.json), so
  # resolution refuses instead of returning an adapter that would report the unit missing.
  with pytest.raises(ValueError, match="custom record name"):
    resolve_unit_adapter(unit_dir, record_name=custom_record)




def test_all_fill_value_array_is_not_an_empty_unit(tmp_path: Path) -> None:
  """zarr writes no chunk for an all-fill-value array; such a store is still a real unit."""
  unit_dir = tmp_path / "unit_zeros"
  root = zarr.open_group(str(unit_dir), mode="a")
  root.create_array(name="ddg", shape=(4,), dtype="float32", fill_value=0.0)[...] = 0.0
  assert not [p for p in unit_dir.rglob("*") if p.is_file() and p.name != "zarr.json"]

  write_aminx_unit(unit_dir, input_digest=_DIGEST)
  validate_aminx_unit(unit_dir, expected_input_digest=_DIGEST)


def test_group_without_arrays_is_an_empty_unit(tmp_path: Path) -> None:
  unit_dir = tmp_path / "unit_group_only"
  zarr.open_group(str(unit_dir), mode="a").attrs["created_at"] = "x"
  with pytest.raises(UnitPackRefused, match="R-U1"):
    write_aminx_unit(unit_dir, input_digest=_DIGEST)
