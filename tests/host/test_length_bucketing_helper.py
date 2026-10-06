"""Unit tests for host length-bucketing helpers and the spec JSON field."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from aminx.host.bucketing import batch_span, bucket_ladder, rung_for
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
