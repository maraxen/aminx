"""Unit tests for host length-bucketing helpers and the spec JSON field."""

from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pytest

from aminx.host.bucketing import (
  batch_span,
  bucket_ladder,
  repad_residue_axis,
  rung_for,
  sample_rung,
  trim_residue_axis,
)
from aminx.run.spec_json import (
  run_specification_from_json_dict,
  run_specification_to_json_dict,
)
from aminx.run.specs import RunSpecification


def test_batch_span_one_dimensional_and_batched() -> None:
  assert batch_span(np.array([0, 1, 0, 0])) == 2
  assert batch_span(np.array([[1, 0, 0, 0], [0, 0, 1, 0]])) == 3


def test_batch_span_gapped_mask_uses_last_valid_index() -> None:
  mask = np.array([1, 0, 0, 1, 0, 0])
  assert batch_span(mask) == 4


def test_batch_span_all_masked_is_zero() -> None:
  assert batch_span(np.zeros(5)) == 0
  assert batch_span(np.zeros((2, 4))) == 0


def test_rung_for_ladder_and_clamp() -> None:
  assert rung_for(76, 512, enabled=True) == 128
  assert rung_for(64, 512, enabled=True) == 64
  assert rung_for(65, 512, enabled=True) == 128
  assert rung_for(600, 512, enabled=True) == 512
  assert rung_for(3000, 512, enabled=True) is None
  assert rung_for(76, 512, enabled=False) is None
  assert rung_for(76, None, enabled=True) is None
  assert rung_for(76, 512, enabled=True, pass_mode="inter") is None


def test_bucket_ladder_matches_xtrax() -> None:
  from xtrax.export.rings import BUCKET_LADDER

  assert bucket_ladder() == tuple(BUCKET_LADDER)


def test_rung_for_does_not_import_onnx_stack() -> None:
  code = (
    "import sys\n"
    "import aminx.host.bucketing as bucketing\n"
    "bucketing.rung_for(76, 512, enabled=True)\n"
    "banned = {'jax2onnx', 'onnx', 'onnxruntime'}\n"
    "loaded = sorted(banned & set(sys.modules))\n"
    "raise SystemExit(0 if not loaded else ','.join(loaded))\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", code],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def test_length_bucketing_spec_json_round_trip() -> None:
  spec = RunSpecification(inputs="a.pdb", length_bucketing=False)
  payload = run_specification_to_json_dict(spec)
  assert payload["length_bucketing"] is False
  restored = run_specification_from_json_dict(payload)
  assert restored.length_bucketing is False

  missing = dict(payload)
  del missing["length_bucketing"]
  decoded = run_specification_from_json_dict(missing)
  assert decoded.length_bucketing is True


def _mask(span: int, padded: int = 512) -> np.ndarray:
  mask = np.zeros(padded, dtype=np.float32)
  mask[:span] = 1.0
  return mask


def test_sample_rung_selects_the_ladder_rung() -> None:
  assert sample_rung(_mask(76), 512, max_length=512, enabled=True) == 128


def test_sample_rung_returns_none_when_nothing_to_trim() -> None:
  assert sample_rung(_mask(76), 76, max_length=76, enabled=True) is None
  assert sample_rung(_mask(76), 512, max_length=None, enabled=True) is None
  assert sample_rung(_mask(76), 512, max_length=512, enabled=False) is None
  assert sample_rung(_mask(76), 512, max_length=512, enabled=True, pass_mode="inter") is None
  assert sample_rung(None, 512, max_length=512, enabled=True) is None


def test_sample_rung_tracer_is_not_trimmed() -> None:
  import jax
  import jax.numpy as jnp

  seen: dict[str, object] = {}

  def capture(mask: jax.Array) -> jax.Array:
    seen["tracer"] = type(mask).__name__
    seen["rung"] = sample_rung(mask, int(mask.shape[-1]), max_length=512, enabled=True)
    return mask

  jax.jit(capture)(jnp.ones((512,), dtype=jnp.float32))
  assert seen["rung"] is None


def test_sample_rung_skip_guards_log_and_return_none(caplog: pytest.LogCaptureFixture) -> None:
  mask = _mask(76)
  cases = (
    {"bias": np.zeros(10), "needle": "bias"},
    {"tie_group_map": np.zeros(10, dtype=np.int32), "needle": "tie_group_map length"},
    {"tie_group_map": np.arange(512, dtype=np.int32), "needle": "tie_group_map id"},
    {"state_position_map": np.zeros((2, 10), dtype=np.int32), "needle": "state_position_map last axis"},
    {"state_position_map": np.full((2, 512), 200, dtype=np.int32), "needle": "state_position_map value"},
    {"structure_mapping": np.zeros(10, dtype=np.int32), "needle": "structure_mapping"},
  )
  for case in cases:
    needle = str(case.pop("needle"))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="aminx.host.bucketing"):
      rung = sample_rung(mask, 512, max_length=512, enabled=True, **case)
    assert rung is None, needle
    assert any(needle in rec.message for rec in caplog.records), needle


def test_sample_rung_accepts_aligned_controls() -> None:
  rung = sample_rung(
    _mask(76),
    512,
    max_length=512,
    enabled=True,
    bias=np.zeros((512, 21), dtype=np.float32),
    tie_group_map=np.zeros(512, dtype=np.int32),
    state_position_map=np.zeros((1, 512), dtype=np.int32),
    structure_mapping=np.zeros((1, 512), dtype=np.int32),
  )
  assert rung == 128


def test_trim_and_repad_round_trip_and_none() -> None:
  arr = np.arange(24, dtype=np.int32).reshape(2, 6, 2)
  trimmed = trim_residue_axis(arr, 4, 1)
  assert trimmed.shape == (2, 4, 2)
  padded = repad_residue_axis(trimmed, 6, 1)
  assert padded.shape == arr.shape
  assert np.array_equal(padded[:, :4], arr[:, :4])
  assert np.all(padded[:, 4:] == 0)
  assert trim_residue_axis(None, 4, 1) is None
  assert repad_residue_axis(None, 6, 1) is None


def test_state_position_map_trims_the_last_axis() -> None:
  spm = np.arange(30, dtype=np.int32).reshape(2, 3, 5)
  trimmed = trim_residue_axis(spm, 3, -1)
  assert trimmed.shape == (2, 3, 3)
  assert np.array_equal(trimmed, spm[:, :, :3])
  padded = repad_residue_axis(trimmed, 5, -1)
  assert padded.shape == spm.shape
  assert np.all(padded[:, :, 3:] == 0)
