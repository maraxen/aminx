"""Layer (b)/(c) profiling: ORT per-op (CPU + Web) and JAX-native cost analysis (T6).

For each already-converted P03/P04 artifact (T2's tracked manifest, D-G) at every
export bucket, this script measures:

- **ORT-CPU**: `enable_profiling=True`, 5 warm-ups + 30 measured runs, aggregated by
  `ort_profile_agg.aggregate_profile`. Two thread settings (D-I): `intra_op_num_threads
  = inter_op_num_threads = 1` (the "t1" table, the one paired against ORT Web t1 in a
  cross-route per-op statement) and `intra_op_num_threads = 4` (the "t4" table,
  reported only on its own -- it makes no cross-route claim).
- **ORT Web**: a `profile` mode of `browser/layer_c/parity.mjs` (opted into per-cell via
  the `cells.json` manifest, NOT a URL parameter -- so every EXISTING caller, which never
  sets that key, takes the byte-identical code path). onnxruntime-web's wasm EP exposes
  no JS-readable per-node profiling event array (`endProfiling()` only frees an internal
  Emscripten MEMFS handle; confirmed against the bundled 1.30.0 dist, 260928), so this
  arm's `web_per_op_available` is expected to be `False` -- a first-class `partial`
  outcome, not a fabricated positive.
- **JAX native**: `jax.jit(wrapper).lower(*specs).compile()`; `cost_analysis()`, the HLO
  op histogram from `compiled.as_text()`, and `jax.profiler.trace` over 30 steady-state
  iterations (after 5 warm-ups) parsed via `xtrax.profiling.trace.parse_hlo_op_times` /
  `parse_dispatch_counts`. A `DISPATCH_COUNT` claim (`xtrax.profiling.claims`) is built
  from a `ProbeRecord` per (path, bucket) cell, `n_executions` cross-checked against the
  trace's own executable-launch count (mismatch -> `ctrl_blind`), `n_compilations`/
  `n_jit_traces` measured via `jax.monitoring.register_event_duration_secs_listener` on
  the exact event names resolved from `jax/_src/dispatch.py` on the installed jax 0.10.0
  (`/jax/core/compile/backend_compile_duration`, `/jax/core/compile/jaxpr_trace_duration`)
  -- an in-run self-check (a deliberately fresh `jax.jit(lambda x: x + 1)` call must
  increment both by >= 1; a repeated call must increment neither) verifies the listener
  actually fires before any real measurement is trusted.

Two more in-run self-tests gate the whole run into `ctrl_blind` if either fails: the
synthetic aggregator check (`ort_profile_agg.synthetic_self_check`) and the two malformed-
`ProbeRecord` controls (a stage-0 record missing `n_jit_traces` must be REFUSED by
`assert_claim_supported`; `metrics={"n_executions": True}` must raise `ClaimValidityError`
at construction).

Uses synthetic inputs (`layer_b_build._synthetic_inputs`) at each export bucket -- this is
a timing/cost-analysis pass, not a correctness comparison, so no real fixture parsing is
needed (mirrors T4's own safety/compile cells' use of the same synthetic-input convention).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_b_common as lbc  # noqa: E402
import layer_c_common as lcc  # noqa: E402
from layer_b_build import _synthetic_inputs  # noqa: E402
from ort_profile_agg import load_and_aggregate, synthetic_self_check  # noqa: E402

CHECKPOINT_ID = "proteinmpnn_v_48_020"
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
DEFAULT_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024)  # aminx.export.EXPORT_BUCKETS

N_WARMUP = 5
N_MEASURED = 30
LOW_COVERAGE_BAR = 0.8

# jax 0.10.0 monitoring event names, resolved from jax/_src/dispatch.py + compiler.py (V-style
# "to be verified" note in this task's spec; the in-run self-check below re-verifies presence,
# not just spelling, every time this script runs -- see `_monitor_selfcheck`).
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"

_HLO_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*\S+\s+([a-zA-Z][\w.\-]*)\(")


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


def _hlo_op_histogram(hlo_text: str) -> dict[str, int]:
  """Count HLO leaf-instruction primitive names (e.g. "add", "dot", "fusion") in `hlo_text`."""
  histogram: dict[str, int] = {}
  for line in hlo_text.splitlines():
    m = _HLO_INSTR_RE.match(line)
    if m:
      op = m.group(1)
      histogram[op] = histogram.get(op, 0) + 1
  return histogram


def _load_trace_events(trace_dir: Path) -> list[dict[str, Any]]:
  """Read the `*.trace.json.gz` a `jax.profiler.trace(trace_dir)` context wrote."""
  profile_dir = trace_dir / "plugins" / "profile"
  if not profile_dir.is_dir():
    msg = f"no plugins/profile directory under {trace_dir}"
    raise RuntimeError(msg)
  subdirs = [p for p in profile_dir.iterdir() if p.is_dir()]
  if not subdirs:
    msg = f"no trace run directory under {profile_dir}"
    raise RuntimeError(msg)
  latest = max(subdirs, key=lambda p: p.stat().st_mtime)
  candidates = list(latest.glob("*.trace.json.gz"))
  if len(candidates) != 1:
    msg = f"expected exactly one *.trace.json.gz under {latest}, found {len(candidates)}"
    raise RuntimeError(msg)
  with gzip.open(candidates[0], "rb") as fh:
    data = json.load(fh)
  events = data.get("traceEvents")
  if not isinstance(events, list):
    msg = f"trace JSON at {candidates[0]} has no traceEvents list"
    raise RuntimeError(msg)
  return events


# --------------------------------------------------------------------------------------
# jax.monitoring duration-event counter (n_compilations/n_jit_traces, per T6 step 2)
# --------------------------------------------------------------------------------------


class MonitorCounter:
  """Appends every `jax.monitoring.record_event_duration_secs` event name it sees.

  Registered ONCE per process (jax.monitoring has no unregister API); callers take
  index-range deltas (`delta(start, end)`) against the cumulative `events` list rather
  than resetting counters, so overlapping/serial measurement windows never race.
  """

  def __init__(self) -> None:
    self.events: list[str] = []
    self._registered = False

  def register(self) -> None:
    if self._registered:
      return
    import jax.monitoring as jax_monitoring

    jax_monitoring.register_event_duration_secs_listener(self._on_event)
    self._registered = True

  def _on_event(self, event: str, duration_secs: float, **_kwargs: Any) -> None:
    del duration_secs
    self.events.append(event)

  def mark(self) -> int:
    return len(self.events)

  def delta(self, start: int, end: int) -> tuple[int, int]:
    window = self.events[start:end]
    n_compilations = sum(1 for e in window if e == BACKEND_COMPILE_EVENT)
    n_jit_traces = sum(1 for e in window if e == JAXPR_TRACE_EVENT)
    return n_compilations, n_jit_traces


def monitor_selfcheck(monitor: MonitorCounter) -> dict[str, Any]:
  """In-run verification (T6 step 2): a fresh `jax.jit` call must fire both duration
  events >= 1; a repeated call (same shapes, cached) must fire neither."""
  fresh_fn = jax.jit(lambda x: x + 1)
  idx0 = monitor.mark()
  jax.block_until_ready(fresh_fn(jnp.asarray(1.0)))
  idx1 = monitor.mark()
  jax.block_until_ready(fresh_fn(jnp.asarray(1.0)))
  idx2 = monitor.mark()

  fresh_compile, fresh_trace = monitor.delta(idx0, idx1)
  repeat_compile, repeat_trace = monitor.delta(idx1, idx2)
  ok = fresh_compile >= 1 and fresh_trace >= 1 and repeat_compile == 0 and repeat_trace == 0
  return {
    "ok": ok,
    "fresh_compile": fresh_compile,
    "fresh_trace": fresh_trace,
    "repeat_compile": repeat_compile,
    "repeat_trace": repeat_trace,
  }


def malformed_record_controls() -> dict[str, Any]:
  """Both malformed-`ProbeRecord` controls (T6 step 2, "Malformed-record controls")."""
  from xtrax.profiling.claims import ClaimClass, ClaimValidityError, assert_claim_supported
  from xtrax.profiling.record import ProbeRecord

  ctrl_i_detected = False
  try:
    bad_stage0 = ProbeRecord(
      probe_id="ctrl_missing_n_jit_traces",
      stage=0,
      n_atoms=64,
      platform="cpu",
      metrics={"n_executions": 1.0, "n_compilations": 1.0},
    )
    assert_claim_supported([bad_stage0], ClaimClass.DISPATCH_COUNT)
  except ClaimValidityError:
    ctrl_i_detected = True

  ctrl_ii_detected = False
  try:
    ProbeRecord(
      probe_id="ctrl_bool_metric",
      stage=0,
      n_atoms=64,
      platform="cpu",
      metrics={"n_executions": True},
    )
  except ClaimValidityError:
    ctrl_ii_detected = True

  return {
    "ok": ctrl_i_detected and ctrl_ii_detected,
    "ctrl_i_missing_metric_refused": ctrl_i_detected,
    "ctrl_ii_boolean_metric_refused": ctrl_ii_detected,
  }


# --------------------------------------------------------------------------------------
# ORT-CPU profiling (5 warm-ups + 30 measured, per thread setting)
# --------------------------------------------------------------------------------------


def _ort_cpu_profile(
  onnx_path: Path, inputs: tuple[Any, Any, Any, Any], *, intra_op_num_threads: int
) -> dict[str, Any]:
  import onnxruntime as ort

  so = ort.SessionOptions()
  so.intra_op_num_threads = intra_op_num_threads
  so.inter_op_num_threads = intra_op_num_threads
  so.enable_profiling = True
  sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
  feed = {inp.name: arr for inp, arr in zip(sess.get_inputs(), inputs, strict=True)}
  for _ in range(N_WARMUP + N_MEASURED):
    sess.run(None, feed)
  profile_path = Path(sess.end_profiling())
  aggregate = load_and_aggregate(profile_path, n_warmup=N_WARMUP)
  profile_path.unlink(missing_ok=True)
  return {
    "aggregate": aggregate,
    "threads_requested": intra_op_num_threads,
    "threads_observed": intra_op_num_threads,
    "source": "SessionOptions.intra_op_num_threads",
  }


# --------------------------------------------------------------------------------------
# JAX-native profiling (cost_analysis + HLO histogram + trace-derived dispatch counts)
# --------------------------------------------------------------------------------------


def _jax_native_profile(
  path: str,
  bucket: int,
  model: Any,
  stage_set: Any,
  inputs_np: tuple[Any, Any, Any, Any],
  monitor: MonitorCounter,
) -> dict[str, Any]:
  from aminx.export import make_p03_featurize, make_p04_unconditional

  fn = make_p03_featurize(model) if path == "p03" else make_p04_unconditional(model, stage_set)
  args = tuple(jnp.asarray(a) for a in inputs_np)

  jitted = jax.jit(fn)
  compiled = jitted.lower(*args).compile()
  cost_analysis_raw = compiled.cost_analysis()
  cost_analysis = dict(cost_analysis_raw) if isinstance(cost_analysis_raw, dict) else {}
  hlo_text = compiled.as_text() or ""
  hlo_op_histogram = _hlo_op_histogram(hlo_text)

  for _ in range(N_WARMUP):
    jax.block_until_ready(jitted(*args))

  idx_before = monitor.mark()
  with tempfile.TemporaryDirectory(prefix=f"jax_trace_{path}_L{bucket}_") as trace_dir_str:
    trace_dir = Path(trace_dir_str)
    t0 = time.perf_counter()
    with jax.profiler.trace(trace_dir):
      for _ in range(N_MEASURED):
        jax.block_until_ready(jitted(*args))
    total_step_seconds = (time.perf_counter() - t0) / N_MEASURED
    trace_events = _load_trace_events(trace_dir)
  idx_after = monitor.mark()

  from xtrax.profiling.trace import parse_dispatch_counts, parse_hlo_op_times

  dispatch_counts = parse_dispatch_counts(trace_events)
  hlo_op_times = parse_hlo_op_times(trace_events)
  n_compilations, n_jit_traces = monitor.delta(idx_before, idx_after)
  dispatch_count_mismatch = dispatch_counts["n_executions"] != N_MEASURED

  return {
    "cost_analysis": cost_analysis,
    "hlo_op_histogram": hlo_op_histogram,
    "hlo_op_times_us": {op: {"seconds": s, "count": c} for op, (s, c) in hlo_op_times.items()},
    "total_step_seconds": total_step_seconds,
    "dispatch_counts": dispatch_counts,
    "dispatch_count_mismatch": dispatch_count_mismatch,
    "n_compilations": n_compilations,
    "n_jit_traces": n_jit_traces,
    "n_measured": N_MEASURED,
  }


def _build_probe_record(path: str, bucket: int, jax_native: dict[str, Any]) -> Any:
  from xtrax.profiling.record import ProbeRecord

  return ProbeRecord(
    probe_id=f"{path}_L{bucket}",
    stage=1,
    n_atoms=bucket,
    platform="cpu",
    metrics={
      "total_step_seconds": jax_native["total_step_seconds"],
      "n_executions": float(N_MEASURED),
      "n_compilations": float(jax_native["n_compilations"]),
      "n_jit_traces": float(jax_native["n_jit_traces"]),
    },
    config={"path": path, "bucket": str(bucket), "route": "jax_native"},
  )


# --------------------------------------------------------------------------------------
# ORT Web profiling (single batched browser session, opt-in per cell via cells.json)
# --------------------------------------------------------------------------------------


def _run_web_profile(
  cells_meta: list[dict[str, Any]], *, node_bin: str | None, timeout_s: float
) -> dict[str, dict[str, Any]]:
  """Batch every (path, bucket) cell into ONE site + ONE browser session with `profile:
  true`. Returns `{cell_name: browser_cell_result}`."""
  resolved_node = lcc.find_node(node_bin)
  if resolved_node is None:
    return {}

  with tempfile.TemporaryDirectory(prefix="layer_b_profile_web_site_") as tmp:
    site_dir = Path(tmp) / "site"
    out_dir = Path(tmp) / "results"
    data_dir = site_dir / "data"
    model_files: dict[str, Path] = {}
    site_cells = []
    for meta in cells_meta:
      name = meta["name"]
      onnx_path = meta["onnx_path"]
      model_files[onnx_path.name] = onnx_path
      coords, mask, residue_index, chain_index = meta["inputs_np"]
      inputs = [
        lcc.write_raw_input(coords, data_dir / f"{name}_in0.bin", "float32"),
        lcc.write_raw_input(mask, data_dir / f"{name}_in1.bin", "float32"),
        lcc.write_raw_input(residue_index, data_dir / f"{name}_in2.bin", "int32"),
        lcc.write_raw_input(chain_index, data_dir / f"{name}_in3.bin", "int32"),
      ]
      outputs = (
        [
          {"label": "neighbor_indices", "dtype": "int32"},
          {"label": "edge_features", "dtype": "float32"},
        ]
        if meta["path"] == "p03"
        else [
          {"label": "logits", "dtype": "float32"},
          {"label": "neighbor_indices", "dtype": "int32"},
        ]
      )
      site_cells.append(
        {
          "name": name,
          "onnx": f"models/{onnx_path.name}",
          "inputs": inputs,
          "outputs": outputs,
          "profile": True,
        }
      )
    lcc.assemble_site(site_cells, model_files, site_dir)
    try:
      harness = lcc.run_browser(
        site_dir=site_dir,
        out_dir=out_dir,
        node_bin=resolved_node,
        num_threads=1,
        isolate=True,
        timeout_s=timeout_s,
      )
    except RuntimeError as exc:
      logger.warning("layer_b_profile: web profiling harness failed: %s", exc)
      return {}
    if not harness.get("harnessOk"):
      return {}
    return dict((harness.get("result") or {}).get("cells", {}))


# --------------------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------------------


def _empty_result(git_hash: str | None, git_clean: bool) -> dict[str, Any]:
  return {
    "git_hash": git_hash,
    "git_clean": git_clean,
    "n_paths_available": 0,
    "converted_by_path": {},
    "cells_total": 0,
    "cells_measured": 0,
    "n_skipped": 0,
    "min_ort_cpu_coverage": None,
    "min_ort_cpu_coverage_available": False,
    "web_per_op_available": False,
    "agg_self_check_pass": False,
    "monitor_selfcheck_pass": False,
    "malformed_record_ctrl_pass": False,
    "dispatch_count_mismatch": False,
    "dispatch_claim_ok": False,
    "artifact_subdir": None,
    "versions": {},
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  from xtrax.profiling.claims import ClaimClass, ClaimValidityError, assert_claim_supported

  from aminx.export import EXPORT_BUCKETS, PINNED_CHECKPOINT_ID
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  buckets = tuple(int(b) for b in args.buckets.split(",")) if args.buckets else DEFAULT_BUCKETS
  artifacts_base = Path(args.artifacts_dir)

  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)  # raises -> exit 3
  resolved_dir = artifacts_base / subdir
  manifest = lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)  # raises -> exit 3

  provenance = lac.provenance()  # exits 3 on a non-git provenance channel
  import os

  os.environ["XTRAX_GIT_SHA"] = provenance["git_hash"]

  result = _empty_result(provenance["git_hash"], provenance["git_clean"])
  result["artifact_subdir"] = subdir

  paths_by_path = lbc.converted_by_path(manifest, ("p03", "p04"), EXPORT_BUCKETS)
  available_paths = [p for p, ok in paths_by_path.items() if ok]
  result["converted_by_path"] = paths_by_path
  result["n_paths_available"] = len(available_paths)

  # Two in-run self-checks that gate EVERYTHING downstream (ctrl_blind, T6 step 3).
  agg_check = synthetic_self_check()
  result["agg_self_check_pass"] = agg_check["ok"]
  result["_agg_self_check_detail"] = agg_check["detail"]

  monitor = MonitorCounter()
  monitor.register()
  monitor_check = monitor_selfcheck(monitor)
  result["monitor_selfcheck_pass"] = monitor_check["ok"]
  result["_monitor_selfcheck_detail"] = monitor_check

  malformed_ctrl = malformed_record_controls()
  result["malformed_record_ctrl_pass"] = malformed_ctrl["ok"]
  result["_malformed_record_ctrl_detail"] = malformed_ctrl

  if not (agg_check["ok"] and monitor_check["ok"] and malformed_ctrl["ok"]):
    result["_note"] = "ctrl_blind: one or more in-run instrument self-checks failed."
    return result

  if not available_paths:
    result["_note"] = "T2 BLOCKED table: neither path fully converted; nothing measured."
    return result

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = (
      f"CHECKPOINT_ID={CHECKPOINT_ID!r} != "
      f"aminx.export.PINNED_CHECKPOINT_ID={PINNED_CHECKPOINT_ID!r}"
    )
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises -> exit 3
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()

  cells_total = len(available_paths) * len(buckets)
  cells_measured = 0
  n_skipped = 0
  ort_t1_coverages: list[float] = []
  probe_records: list[Any] = []
  any_dispatch_mismatch = False
  cell_details: dict[str, Any] = {}
  errors: dict[str, str] = {}
  web_cells_meta: list[dict[str, Any]] = []
  flops_by_path: dict[str, list[tuple[int, float]]] = {p: [] for p in available_paths}

  for path in available_paths:
    for bucket in buckets:
      label = f"{path}_L{bucket}"
      onnx_path = resolved_dir / f"{path}_L{bucket}.onnx"
      inputs_np = _synthetic_inputs(bucket)
      try:
        ort_t1 = _ort_cpu_profile(onnx_path, inputs_np, intra_op_num_threads=1)
        ort_t4 = _ort_cpu_profile(onnx_path, inputs_np, intra_op_num_threads=4)
        jax_native = _jax_native_profile(path, bucket, model, stage_set, inputs_np, monitor)
        probe = _build_probe_record(path, bucket, jax_native)
        probe_records.append(probe)

        coverage_t1 = ort_t1["aggregate"]["coverage"]
        ort_t1_coverages.append(coverage_t1)
        if jax_native["dispatch_count_mismatch"]:
          any_dispatch_mismatch = True
        flops = jax_native["cost_analysis"].get("flops")
        if isinstance(flops, (int, float)):
          flops_by_path[path].append((bucket, float(flops)))

        cell_details[label] = {
          "ort_cpu_t1": ort_t1,
          "ort_cpu_t4": ort_t4,
          "jax_native": jax_native,
        }
        cells_measured += 1
        web_cells_meta.append(
          {"name": label, "path": path, "onnx_path": onnx_path, "inputs_np": inputs_np}
        )
      except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
        n_skipped += 1
        errors[label] = f"{type(exc).__name__}: {exc}"

  dispatch_claim_ok = False
  if probe_records:
    try:
      assert_claim_supported(probe_records, ClaimClass.DISPATCH_COUNT)
      dispatch_claim_ok = True
    except ClaimValidityError as exc:
      errors["dispatch_claim"] = f"{type(exc).__name__}: {exc}"

  web_cells: dict[str, Any] = {}
  if web_cells_meta:
    web_cells = _run_web_profile(web_cells_meta, node_bin=args.node_bin, timeout_s=args.timeout_s)
  web_per_op_available = any(
    bool(cell.get("profileEventsAvailable")) for cell in web_cells.values() if cell.get("ok")
  )
  for label, cell in web_cells.items():
    if label in cell_details:
      cell_details[label]["ort_web"] = cell

  flops_scaling_exponent: dict[str, float | None] = {}
  for path, points in flops_by_path.items():
    if len(points) >= 2:
      import math

      x0, y0 = points[0]
      x1, y1 = points[-1]
      if x0 > 0 and x1 > 0 and y0 > 0 and y1 > 0 and x0 != x1:
        flops_scaling_exponent[path] = (math.log(y1 / y0)) / (math.log(x1 / x0))
      else:
        flops_scaling_exponent[path] = None
    else:
      flops_scaling_exponent[path] = None

  min_coverage = min(ort_t1_coverages) if ort_t1_coverages else None

  versions = {
    "jax": jax.__version__,
    "onnxruntime": _pkg_version("onnxruntime"),
    "xtrax": _pkg_version("xtrax"),
    "bathos_overlay": lbc.bathos_overlay_provenance(),
  }

  result.update(
    {
      "cells_total": cells_total,
      "cells_measured": cells_measured,
      "n_skipped": n_skipped,
      "min_ort_cpu_coverage": min_coverage,
      "min_ort_cpu_coverage_available": min_coverage is not None,
      "web_per_op_available": web_per_op_available,
      "dispatch_count_mismatch": any_dispatch_mismatch,
      "dispatch_claim_ok": dispatch_claim_ok,
      "versions": versions,
      "_cells": cell_details,
      "_flops_scaling_exponent": flops_scaling_exponent,
      "_errors": errors,
    }
  )
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  parser.add_argument(
    "--buckets",
    default=",".join(str(b) for b in DEFAULT_BUCKETS),
    help="Comma-separated bucket ladder (default: aminx.export.EXPORT_BUCKETS).",
  )
  parser.add_argument("--node-bin", default=None, help="Explicit node binary (else discovered).")
  parser.add_argument(
    "--timeout-s", type=float, default=300.0, help="Per-browser-invocation timeout (seconds)."
  )
  args = parser.parse_args(argv)

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_b_profile: integrity refusal: %s", exc)
    result = _empty_result(None, False)
    result["_integrity_error"] = str(exc)
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_b_profile: n_paths_available=%s cells=%s/%s agg_self_check=%s monitor_selfcheck=%s "
    "malformed_ctrl=%s dispatch_mismatch=%s web_per_op_available=%s min_coverage=%s exit_code=%d",
    result.get("n_paths_available"),
    result.get("cells_measured"),
    result.get("cells_total"),
    result.get("agg_self_check_pass"),
    result.get("monitor_selfcheck_pass"),
    result.get("malformed_record_ctrl_pass"),
    result.get("dispatch_count_mismatch"),
    result.get("web_per_op_available"),
    result.get("min_ort_cpu_coverage"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
