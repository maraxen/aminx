"""Browser-vs-native timing parse and per-cell graded failure (T8)."""

# ruff: noqa: S101

from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.browser_validation.layer_c_bench import (
  _empty_result,
  collect_cell_timing,
  graded_exit_code,
  parse_route_timing,
)

_SIDECAR = (
  Path(__file__).resolve().parents[2] / "scripts" / "browser_validation" / "layer_c_bench.bth.toml"
)
_CELL = "p04_L256_ort_wasm_t1#r0"


def _result_schema_keys() -> set[str]:
  with _SIDECAR.open("rb") as handle:
    schema = tomllib.load(handle)["result_schema"]
  return set(schema)


def test_browser_and_native_payloads_parse_to_the_same_samples() -> None:
  """`steady_ms` under `phases` and a top-level native payload yield the same samples."""
  steady = [1.25, 2.5, 3.75]
  browser = {
    "ok": True,
    "crossOriginIsolated": True,
    "threadsRequested": 2,
    "threadsObserved": 2,
    "threadsMismatch": False,
    "phases": {
      "fetch_load_ms": 4.0,
      "session_create_ms": 5.0,
      "first_inference_ms": 6.0,
      "steady_ms": list(steady),
      "n_warmup": 5,
      "n_iter": 3,
    },
    "peakMemory": {"available": True, "bytes": 128},
    "wasmHeapBytes": 4096,
  }
  native = {
    "ok": True,
    "steady_ms": list(steady),
    "n_warmup": 5,
    "n_iter": 3,
    "phases": {
      "weight_load_s": 0.1,
      "lower_compile_s": 0.2,
      "first_call_s": 0.3,
    },
    "jax_memory_available": True,
  }
  parsed_browser = parse_route_timing(browser, browser=True)
  parsed_native = parse_route_timing(native, browser=False)
  assert parsed_browser.steady_ms == steady
  assert parsed_native.steady_ms == steady
  assert parsed_browser.n_warmup == parsed_native.n_warmup == 5
  assert parsed_browser.n_iter == parsed_native.n_iter == 3
  assert parsed_browser.phases is not None
  assert parsed_browser.phases["fetch_load_ms"] == 4.0
  assert parsed_browser.phases["session_create_ms"] == 5.0
  assert parsed_browser.phases["first_inference_ms"] == 6.0
  assert parsed_browser.threads_observed == 2
  assert parsed_browser.threads_mismatch is False
  assert parsed_browser.peak_memory_available is True
  assert parsed_browser.wasm_heap_bytes == 4096
  assert parsed_native.peak_memory_available is True
  assert parsed_native.phases is not None
  assert "steady_ms" not in parsed_native.phases
  assert parsed_native.wasm_heap_bytes is None


def test_missing_timings_count_as_cell_error_and_keep_schema() -> None:
  """A cell with no steady-state samples is an error; the graded exit stays 0."""
  result = _empty_result("abc123", True)  # noqa: FBT003
  timed = {
    "ok": True,
    "phases": {
      "fetch_load_ms": 1.0,
      "session_create_ms": 2.0,
      "first_inference_ms": 3.0,
    },
  }
  parsed = collect_cell_timing(result, timed, browser=True, cell_key=_CELL)
  assert parsed is None
  assert result["n_cells_error"] == 1
  message = result["_errors"][_CELL]
  assert message.startswith("KeyError:")
  assert "steady_ms" in message
  assert _result_schema_keys() <= set(result)
  assert graded_exit_code(integrity_failure=False) == 0
