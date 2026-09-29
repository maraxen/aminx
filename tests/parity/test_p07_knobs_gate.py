"""CPU checks for the P07 sampling-knobs gate. No Chromium and no reference weights."""

# ruff: noqa: S101

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scripts.browser_validation.p07_knobs_gate import (
  N_AA,
  OUTCOMES,
  P07_KEYS,
  SplitMix64,
  _json_default,
  _node_bin,
  _raise_on_failed_cells,
  bad_gumbel_from_uniform,
  build_p07_inputs,
  cell_reuse_eligible,
  check_outputs_complete,
  evaluate_outcome,
  expected_output_bytes,
  files_identical,
  find_old_site_file,
  gumbel_from_uniform,
  js_build,
  old_bucket_onnx_files,
  old_output_paths,
  order_counts,
  passing_result,
  plan_chunks,
  result_template,
  sha256_file,
  sha256_files,
  stamp_reusable,
  summarize_gumbel,
  summarize_orders,
  tie_group_map,
  uniform_from_u32,
  write_cell_stamp,
)

_ROOT = Path(__file__).resolve().parents[2]
_SIDECAR = _ROOT / "scripts" / "browser_validation" / "p07_knobs_gate.bth.toml"
_CASES = _ROOT / "scripts" / "browser_validation" / "p07_knobs_gate.cases.json"


def _noise(transform: str) -> Callable[[int, int, int], np.ndarray]:
  fn = gumbel_from_uniform if transform == "gumbel" else bad_gumbel_from_uniform

  def noise(seed: int, rows: int, cols: int) -> np.ndarray:
    rng = SplitMix64(seed)
    out = np.empty((rows, cols), dtype=np.float32)
    flat = out.reshape(-1)
    for index in range(flat.shape[0]):
      flat[index] = fn(uniform_from_u32(rng.next_u32()))
    return out

  return noise


def test_tie_ids_stay_in_range() -> None:
  length = 12
  ids = tie_group_map(length, [[4, 1, 9], [2, 7]])
  assert int(ids.min()) >= 0
  assert int(ids.max()) < length
  assert int(ids[4]) == 1
  assert int(ids[1]) == 1
  assert int(ids[9]) == 1
  assert int(ids[2]) == 2
  assert int(ids[7]) == 2
  assert int(ids[0]) == 0


def test_runspec_matches_python_for_each_field() -> None:
  length = 6
  coords = np.arange(length * 12, dtype=np.float32).reshape(length, 4, 3)
  structure = {
    "coords": coords,
    "coords_shape": [length, 4, 3],
    "mask": np.ones((length,), dtype=np.float32),
    "residue_index": np.arange(length, dtype=np.int32),
    "chain_index": np.zeros((length,), dtype=np.int32),
    "chain_ids": ["A", "A", "A", "B", "B", "B"],
    "native_tokens": (np.arange(length, dtype=np.int32) % 20),
  }
  specs: dict[str, dict[str, object]] = {
    "temperature": {"seed": 3, "decoding_order": [5, 4, 3, 2, 1, 0]},
    "bias_AA": {"seed": 4, "bias_AA": {"A": 1.5}, "decoding_order": "random"},
    "omit": {
      "seed": 5,
      "omit_AA": "CX",
      "omit_AA_per_residue": {"1": "W"},
      "decoding_order": [0, 1, 2, 3, 4, 5],
    },
    "fixed_positions": {"seed": 6, "fixed_positions": {"0": "A"}, "decoding_order": "random"},
    "chains_to_design": {"seed": 7, "chains_to_design": ["A"], "decoding_order": "random"},
    "tied_positions": {"seed": 8, "tied_positions": [[4, 1, 3]], "decoding_order": "random"},
    "bias_AA_per_residue": {
      "seed": 9,
      "bias_AA_per_residue": {"2": {"G": 0.25}},
      "decoding_order": [0, 1, 2, 3, 4, 5],
    },
  }
  node = _node_bin(None)
  for name, spec in specs.items():
    blob = json.loads(json.dumps({"structure": structure, "runspec": spec}, default=_json_default))
    py = build_p07_inputs(blob["structure"], blob["runspec"])
    js = js_build(blob["structure"], blob["runspec"], node)
    for key in P07_KEYS:
      assert np.array_equal(py[key], js[key]), (name, key)


def test_b2_statistic_passes_on_the_real_transform_and_fails_on_controls() -> None:
  good = summarize_gumbel(_noise("gumbel"), 20_000, 50_000)
  bad = summarize_gumbel(_noise("bad"), 20_000, 50_000)
  assert good["ok"], good
  assert not bad["ok"], bad
  fair = summarize_orders(order_counts(20_000, 4, 7, kind="fisher"))
  biased = summarize_orders(order_counts(20_000, 4, 7, kind="biased"))
  assert fair["ok"], fair
  assert not biased["ok"], biased


def test_outcome_pass_and_each_failure_and_sidecar_schema() -> None:
  doc = tomllib.loads(_SIDECAR.read_text())
  for name, expr in OUTCOMES:
    assert doc["outcomes"][name]["condition"] == expr
  assert set(doc["result_schema"]) == set(result_template())
  assert evaluate_outcome(passing_result()) == "pass"
  for row in json.loads(_CASES.read_text()):
    assert evaluate_outcome(row["result"]) == row["expect"], row["name"]


def test_raise_on_failed_cells_reports_the_failed_cell_name() -> None:
  harness = {
    "harnessOk": True,
    "result": {
      "cells": {
        "good": {"ok": True},
        "bad": {"ok": False, "error": "ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc"},
      },
    },
  }
  with pytest.raises(RuntimeError, match="bad"):
    _raise_on_failed_cells(harness, ["good", "bad"])


def test_raise_on_failed_cells_does_not_raise_when_all_ok() -> None:
  harness = {
    "harnessOk": True,
    "result": {"cells": {"good": {"ok": True}, "also_good": {"ok": True}}},
  }
  _raise_on_failed_cells(harness, ["good", "also_good"])


def test_raise_on_failed_cells_reports_a_missing_cell() -> None:
  harness = {"harnessOk": True, "result": {"cells": {"good": {"ok": True}}}}
  with pytest.raises(RuntimeError, match="missing"):
    _raise_on_failed_cells(harness, ["good", "never_ran"])


def test_cases_grade_with_the_pinned_bathos_evaluator() -> None:
  """The local evaluator above mirrors the sidecar; this grades it with bathos itself."""
  sidecar_mod = pytest.importorskip("bathos.sidecar")
  sidecar = sidecar_mod.parse_sidecar(_SIDECAR)
  for row in json.loads(_CASES.read_text()):
    got = sidecar_mod.evaluate_outcome(sidecar, row["result"])
    got = getattr(got, "outcome", got)
    assert got == row["expect"], row["name"]
  passing = sidecar_mod.evaluate_outcome(sidecar, passing_result())
  assert getattr(passing, "outcome", passing) == "pass"


# --- T11f: browser chunking + resume (pure, no browser, no JAX export) ---------------------


def _write_cell_inputs(
  base: Path,
  *,
  bucket: int,
  name: str,
  n_inputs: int,
  payloads: list[bytes],
) -> list[Path]:
  """Write ``<name>_<i>.bin`` under ``base/site_L{bucket}/data/``; return their paths."""
  data_dir = base / f"site_L{bucket}" / "data"
  data_dir.mkdir(parents=True, exist_ok=True)
  paths = []
  for i in range(n_inputs):
    path = data_dir / f"{name}_{i}.bin"
    path.write_bytes(payloads[i])
    paths.append(path)
  return paths


def _write_cell_outputs(
  base: Path, *, bucket: int, name: str, length: int, ok: bool = True
) -> None:
  """Write ``<name>__tokens.bin`` / ``<name>__log_probs.bin`` under ``base/browser_L{bucket}/``."""
  out_dir = base / f"browser_L{bucket}"
  out_dir.mkdir(parents=True, exist_ok=True)
  tokens_size = expected_output_bytes(length, "tokens")
  log_probs_size = expected_output_bytes(length, "log_probs")
  if not ok:
    tokens_size -= 4  # wrong size: simulates a truncated/partial write
  (out_dir / f"{name}__tokens.bin").write_bytes(b"\x01" * tokens_size)
  (out_dir / f"{name}__log_probs.bin").write_bytes(b"\x02" * log_probs_size)


def test_plan_chunks_splits_with_a_remainder() -> None:
  names = [f"c{i}" for i in range(37)]
  chunks = plan_chunks(names, 16)
  assert [len(chunk) for chunk in chunks] == [16, 16, 5]
  assert [name for chunk in chunks for name in chunk] == names


def test_plan_chunks_empty_names_gives_no_chunks() -> None:
  assert plan_chunks([], 16) == []


def test_plan_chunks_rejects_non_positive_chunk_size() -> None:
  with pytest.raises(ValueError, match="positive"):
    plan_chunks(["a"], 0)


def test_expected_output_bytes_matches_dtype_sizes() -> None:
  assert expected_output_bytes(10, "tokens") == 10 * 4
  assert expected_output_bytes(10, "log_probs") == 10 * N_AA * 4
  with pytest.raises(ValueError, match="unknown"):
    expected_output_bytes(10, "bogus")


def test_files_identical(tmp_path: Path) -> None:
  a = tmp_path / "a.bin"
  b = tmp_path / "b.bin"
  a.write_bytes(b"same-bytes")
  b.write_bytes(b"same-bytes")
  assert files_identical(a, b)
  b.write_bytes(b"different")
  assert not files_identical(a, b)
  assert not files_identical(a, tmp_path / "missing.bin")


def test_find_old_site_file_direct_and_missing(tmp_path: Path) -> None:
  old_dir = tmp_path / "old"
  (old_dir / "site_L128" / "data").mkdir(parents=True)
  (old_dir / "site_L128" / "data" / "foo_0.bin").write_bytes(b"x")
  assert find_old_site_file(old_dir, 128, "data/foo_0.bin") is not None
  assert find_old_site_file(old_dir, 256, "data/foo_0.bin") is None


def test_find_old_site_file_supports_chunked_layout(tmp_path: Path) -> None:
  old_dir = tmp_path / "old"
  (old_dir / "site_L256_c003" / "data").mkdir(parents=True)
  (old_dir / "site_L256_c003" / "data" / "bar_1.bin").write_bytes(b"y")
  found = find_old_site_file(old_dir, 256, "data/bar_1.bin")
  assert found is not None
  assert found.read_bytes() == b"y"


def test_old_bucket_onnx_files_includes_sidecar_data_when_present(tmp_path: Path) -> None:
  old_dir = tmp_path / "old"
  models_dir = old_dir / "site_L128" / "models"
  models_dir.mkdir(parents=True)
  onnx = models_dir / "p07_L128.onnx"
  onnx.write_bytes(b"onnx-bytes")
  assert old_bucket_onnx_files(old_dir, 128, "p07_L128.onnx") == [onnx]

  Path(f"{onnx}.data").write_bytes(b"external")
  assert old_bucket_onnx_files(old_dir, 128, "p07_L128.onnx") == [onnx, Path(f"{onnx}.data")]

  assert old_bucket_onnx_files(old_dir, 256, "p07_L256.onnx") is None


def test_check_outputs_complete(tmp_path: Path) -> None:
  bucket, name, length = 128, "cellZ", 4
  _write_cell_outputs(tmp_path, bucket=bucket, name=name, length=length)
  assert check_outputs_complete(old_output_paths(tmp_path, bucket, name), length)
  assert not check_outputs_complete(old_output_paths(tmp_path, bucket, "missing_cell"), length)


def test_write_cell_stamp_round_trip(tmp_path: Path) -> None:
  out_dir = tmp_path / "browser_L128"
  out_dir.mkdir()
  write_cell_stamp(
    out_dir=out_dir,
    name="cellX",
    onnx_sha256="sha-onnx",
    input_sha256s=["sha-in-0", "sha-in-1"],
    output_sha256s={"tokens": "sha-tok", "log_probs": "sha-lp"},
    git_hash="deadbeef",
    reused_from="/tmp/old",
  )
  payload = json.loads((out_dir / "cellX__stamp.json").read_text())
  assert payload == {
    "onnx_sha256": "sha-onnx",
    "input_sha256s": ["sha-in-0", "sha-in-1"],
    "output_sha256s": {"tokens": "sha-tok", "log_probs": "sha-lp"},
    "git_hash": "deadbeef",
    "reused_from": "/tmp/old",
  }


def test_write_cell_stamp_omits_reused_from_when_fresh(tmp_path: Path) -> None:
  out_dir = tmp_path / "browser_L128"
  out_dir.mkdir()
  write_cell_stamp(
    out_dir=out_dir,
    name="cellX",
    onnx_sha256="s",
    input_sha256s=[],
    output_sha256s={},
    git_hash="h",
  )
  payload = json.loads((out_dir / "cellX__stamp.json").read_text())
  assert "reused_from" not in payload


def test_stamp_reusable_true_when_absent(tmp_path: Path) -> None:
  assert stamp_reusable(tmp_path / "nope__stamp.json", "sha", [], {})


def test_stamp_reusable_true_when_matching(tmp_path: Path) -> None:
  write_cell_stamp(
    out_dir=tmp_path,
    name="s",
    onnx_sha256="sha-onnx",
    input_sha256s=["a", "b"],
    output_sha256s={"tokens": "t", "log_probs": "l"},
    git_hash="h",
  )
  stamp = tmp_path / "s__stamp.json"
  assert stamp_reusable(stamp, "sha-onnx", ["a", "b"], {"tokens": "t", "log_probs": "l"})


def test_stamp_reusable_false_when_onnx_sha_differs(tmp_path: Path) -> None:
  write_cell_stamp(
    out_dir=tmp_path,
    name="s",
    onnx_sha256="sha-onnx",
    input_sha256s=["a", "b"],
    output_sha256s={"tokens": "t", "log_probs": "l"},
    git_hash="h",
  )
  stamp = tmp_path / "s__stamp.json"
  assert not stamp_reusable(stamp, "different-sha", ["a", "b"], {"tokens": "t", "log_probs": "l"})


def test_cell_reuse_eligible_when_everything_matches(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  _write_cell_inputs(old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads)
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length)

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir,
    bucket=bucket,
    name=name,
    n_inputs=n_inputs,
    payloads=payloads,
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="abc123",
  )
  assert result["eligible"] is True
  assert result["reason"] is None
  assert len(result["input_sha256s"]) == n_inputs
  assert set(result["output_sha256s"]) == {"tokens", "log_probs"}


def test_cell_reuse_eligible_input_byte_differs_is_not_eligible(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  old_dir = tmp_path / "old"
  _write_cell_inputs(
    old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=[b"in0", b"in1"]
  )
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length)

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir,
    bucket=bucket,
    name=name,
    n_inputs=n_inputs,
    payloads=[b"in0", b"DIFFERENT"],
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="abc123",
  )
  assert result["eligible"] is False
  assert result["reason"] == "input_mismatch"


def test_cell_reuse_eligible_onnx_mismatch_reuses_nothing(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  _write_cell_inputs(old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads)
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length)

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=False,
    onnx_sha256="abc123",
  )
  assert result == {"eligible": False, "reason": "onnx_mismatch"}


def test_cell_reuse_eligible_output_missing_is_not_eligible(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  _write_cell_inputs(old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads)
  # No outputs written for this cell.

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="abc123",
  )
  assert result["eligible"] is False
  assert result["reason"] == "output_incomplete"


def test_cell_reuse_eligible_output_wrong_size_is_not_eligible(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  _write_cell_inputs(old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads)
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length, ok=False)

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="abc123",
  )
  assert result["eligible"] is False
  assert result["reason"] == "output_incomplete"


def test_cell_reuse_eligible_stamp_mismatch_is_not_eligible(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 128, "cellA", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  _write_cell_inputs(old_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads)
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length)
  write_cell_stamp(
    out_dir=old_dir / f"browser_L{bucket}",
    name=name,
    onnx_sha256="stale-onnx-sha",
    input_sha256s=["whatever"],
    output_sha256s={"tokens": "x", "log_probs": "y"},
    git_hash="deadbeef",
  )

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="current-onnx-sha",
  )
  assert result["eligible"] is False
  assert result["reason"] == "stamp_mismatch"


def test_cell_reuse_eligible_supports_chunked_old_layout(tmp_path: Path) -> None:
  bucket, name, length, n_inputs = 256, "cellB", 6, 2
  payloads = [b"in0", b"in1"]
  old_dir = tmp_path / "old"
  data_dir = old_dir / f"site_L{bucket}_c002" / "data"
  data_dir.mkdir(parents=True)
  for i, payload in enumerate(payloads):
    (data_dir / f"{name}_{i}.bin").write_bytes(payload)
  _write_cell_outputs(old_dir, bucket=bucket, name=name, length=length)

  new_dir = tmp_path / "new"
  new_paths = _write_cell_inputs(
    new_dir, bucket=bucket, name=name, n_inputs=n_inputs, payloads=payloads
  )

  result = cell_reuse_eligible(
    name=name,
    bucket=bucket,
    length=length,
    new_input_paths=new_paths,
    old_dir=old_dir,
    onnx_ok=True,
    onnx_sha256="abc123",
  )
  assert result["eligible"] is True


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
  import hashlib

  path = tmp_path / "f.bin"
  path.write_bytes(b"hello world")
  assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_files_concatenates_in_order(tmp_path: Path) -> None:
  import hashlib

  a = tmp_path / "a.bin"
  b = tmp_path / "b.bin"
  a.write_bytes(b"AAA")
  b.write_bytes(b"BBB")
  assert sha256_files([a, b]) == hashlib.sha256(b"AAABBB").hexdigest()
  assert sha256_files([b, a]) == hashlib.sha256(b"BBBAAA").hexdigest()
