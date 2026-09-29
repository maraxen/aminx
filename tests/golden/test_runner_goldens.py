"""Byte-exact check of the T0.5a in-memory MPNN runner goldens.

Skips when ``tests/golden/runner_v0/manifest.json`` is absent (capture runs on
titanix under bathos) or when the live JAX device kind differs from the
manifest. Case definitions live in ``scripts/parity/runner_golden_cases.py``.
"""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.parity.runner_golden_cases import (
  GOLDEN_DIR,
  cases_by_id,
  compare_case,
  current_device_kind,
  load_case_npz,
  run_case,
)

pytestmark = pytest.mark.parity_targeted


def test_runner_v0_goldens_byte_exact() -> None:
  """Re-run each captured case and require dtype, shape, and values to match."""
  manifest_path = GOLDEN_DIR / "manifest.json"
  if not manifest_path.is_file():
    pytest.skip(
      "tests/golden/runner_v0/manifest.json is absent; "
      "capture runs on titanix under bathos and is not generated in-tree",
    )
  manifest = json.loads(manifest_path.read_text())
  device = current_device_kind()
  captured_device = str(manifest["device_kind"])
  if device != captured_device:
    pytest.skip(
      f"device kind {device} differs from capture device {captured_device}",
    )

  defined = cases_by_id()
  for entry in manifest["cases"]:
    case_id = str(entry["case_id"])
    case = defined[case_id]
    npz_path = Path(GOLDEN_DIR / str(entry["npz"]))
    assert npz_path.is_file(), f"golden npz missing for {case_id}: {npz_path}"
    captured_arrays, captured_payload = load_case_npz(npz_path)
    messages = compare_case(captured_arrays, captured_payload, run_case(case))
    assert messages == [], f"{case_id}: {messages}"
