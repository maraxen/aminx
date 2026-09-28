"""Layer (c) benchmark driver: one Python process, one Chromium session (T8).

Cells are paths {p03, p04} x buckets {128, 256, 512, 1024} x routes
{ort_wasm_t1, ort_wasm_tN, native_jax_cpu}, plus p04@256 controls `aa_dup` and
`planted_5ms`. Order is shuffled per repeat. Browser cells go through
`browser/layer_c/run_bench.mjs` (JSON-lines, two contexts created at startup).
The native arm is in-process. Sentences are emitted only after
`assert_claim_supported(..., END_TO_END)` and route ratios only for pairs from
`paired_configs` that hold threads fixed.

Graded outcomes (blocked, incomplete, not_isolated, ctrl_blind) write a result
and exit 0. Integrity failures (missing manifest) exit 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import bench_stats

LAYER_C_DIR = _WORKTREE_ROOT / "browser" / "layer_c"
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
PATHS: tuple[str, ...] = ("p03", "p04")
BUCKETS: tuple[int, ...] = (128, 256, 512, 1024)
CLAIM_ROUTES: tuple[str, ...] = ("ort_wasm_t1", "ort_wasm_tN", "native_jax_cpu")
N_REPEATS = 3
N_WARMUP = 5
N_ITER = 30
BUDGET_S = 90.0 * 60.0
PIN_XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CHECKPOINT_ID = "proteinmpnn_v_48_020"

_INDEX_HTML = (
  '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
  '<script type="module" src="./bench.mjs"></script></body></html>\n'
)

_MATMUL_WORKER = """
import time
import jax
import jax.numpy as jnp
n = 2048
x = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.float32)
y = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=jnp.float32)
fn = jax.jit(lambda a, b: a @ b)
jax.block_until_ready(fn(x, y))
t0 = time.perf_counter()
c0 = time.process_time()
for _ in range(3):
    jax.block_until_ready(fn(x, y))
wall = time.perf_counter() - t0
cpu = time.process_time() - c0
print(cpu / wall)
"""


def measure_cpu_wall_ratio(xla_flags: str) -> float:
  """CPU-time/wall of a 2048² f32 matmul in a fresh process (XLA reads flags at init)."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = xla_flags
  proc = subprocess.run(  # noqa: S603 -- fixed argv, worker is a constant
    [sys.executable, "-c", _MATMUL_WORKER],
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    msg = f"thread-pin matmul failed (exit {proc.returncode}): {proc.stderr[-2000:]}"
    raise RuntimeError(msg)
  lines = [line for line in proc.stdout.splitlines() if line.strip()]
  if not lines:
    msg = f"thread-pin matmul produced no stdout: {proc.stderr[-2000:]}"
    raise RuntimeError(msg)
  return float(lines[-1])


def thread_pin_status() -> dict[str, Any]:
  """Step 0. `native_threads` is `pinned` only when the documented flag meets the bar."""
  pinned_ratio = measure_cpu_wall_ratio(PIN_XLA_FLAGS)
  unpinned_ratio = measure_cpu_wall_ratio("")
  ok = pinned_ratio <= 1.2 and unpinned_ratio > 1.5
  reason = (
    f"jax CPU matmul CPU/wall was {pinned_ratio:.3f} with XLA_FLAGS={PIN_XLA_FLAGS!r} "
    f"and {unpinned_ratio:.3f} without it; a pin requires <= 1.2 and an unpin > 1.5"
  )
  return {
    "pinned_ratio": pinned_ratio,
    "unpinned_ratio": unpinned_ratio,
    "ok": ok,
    "native_threads": "pinned" if ok else "unpinned",
    "reason": reason,
  }


def _empty_result(git_hash: str, git_clean: bool) -> dict[str, Any]:
  return {
    "n_paths_available": 0,
    "converted_by_path": {},
    "git_hash": git_hash,
    "git_clean": git_clean,
    "cells_total": 0,
    "cells_measured": 0,
    "n_cells_error": 0,
    "n_cells_below_minima": 0,
    "n_repeats": 0,
    "min_warmup": 0,
    "min_iterations": 0,
    "harness_ok": False,
    "budget_expired": False,
    "cross_origin_isolated": False,
    "isolation_ok": False,
    "planted_diff_ci_excludes_zero": False,
    "planted_point_estimate_ms": None,
    "planted_point_estimate_ms_available": False,
    "aa_ratio_point": None,
    "aa_ratio_point_available": False,
    "aa_ratio_ci_contains_one": False,
    "native_threads": "unpinned",
    "session_id": "",
    "wall_seconds": 0.0,
    "peak_memory_available": False,
    "artifact_subdir": "",
    "versions": {},
    "hardware_concurrency": 0,
    "n_sentences": 0,
    "thread_caps": {},
  }


def make_probe_record(
  *,
  probe_id: str,
  path: str,
  bucket: int,
  route: str,
  session_id: str,
  threads: str,
  total_step_seconds: float | None,
  threads_mismatch: bool = False,
  git_sha: str | None = None,
) -> Any:
  """Stage-1 END_TO_END record. `metrics` is exactly `{total_step_seconds}` when timed."""
  from xtrax.profiling.record import ProbeRecord

  metrics: dict[str, float] = {}
  if total_step_seconds is not None:
    metrics["total_step_seconds"] = float(total_step_seconds)
  kwargs: dict[str, Any] = {
    "probe_id": probe_id,
    "stage": 1,
    "n_atoms": bucket,
    "platform": "cpu",
    "metrics": metrics,
    "config": {
      "path": path,
      "bucket": str(bucket),
      "route": route,
      "session_id": session_id,
      "threads": threads,
      "threads_mismatch": "true" if threads_mismatch else "false",
    },
  }
  if git_sha is not None:
    kwargs["git_sha"] = git_sha
  return ProbeRecord(**kwargs)


def cite_bucket(records: list[Any], bucket: int) -> None:
  """Refuse (ClaimValidityError) unless END_TO_END is supported at this bucket."""
  from xtrax.profiling.claims import ClaimClass, assert_claim_supported

  group = [record for record in records if record.n_atoms == bucket]
  assert_claim_supported(group, ClaimClass.END_TO_END, target_n_atoms=bucket)


def _citable(records: list[Any], *, native_pinned: bool, for_ratio: bool) -> list[Any]:
  kept: list[Any] = []
  for record in records:
    if record.config.get("threads_mismatch") == "true":
      continue
    if record.config.get("route") not in CLAIM_ROUTES:
      continue
    if for_ratio and not native_pinned and record.config.get("route") == "native_jax_cpu":
      continue
    kept.append(record)
  return kept


def claim_ratio_pairs(records: list[Any], *, native_pinned: bool) -> list[tuple[Any, Any]]:
  """Pairs from `paired_configs` after a per-bucket END_TO_END cite.

  `threads` is held fixed, so a tN record cannot pair with a 1-thread native or
  t1 record. When native is unpinned the native route is omitted entirely.
  """
  from xtrax.profiling.claims import ClaimClass, paired_configs

  pool = _citable(records, native_pinned=native_pinned, for_ratio=True)
  buckets = sorted({record.n_atoms for record in pool})
  for bucket in buckets:
    cite_bucket(pool, bucket)
  if not pool:
    return []
  return paired_configs(
    pool,
    ClaimClass.END_TO_END,
    axis="route",
    hold_fixed=("path", "bucket", "session_id", "threads"),
  )


def timing_sentences(
  records: list[Any],
  summaries: dict[str, bench_stats.WithinSessionCI],
) -> list[str]:
  """Timing sentences for citable records, only after END_TO_END accepts the bucket."""
  pool = _citable(records, native_pinned=True, for_ratio=False)
  sentences: list[str] = []
  for bucket in sorted({record.n_atoms for record in pool}):
    group = [record for record in pool if record.n_atoms == bucket]
    cite_bucket(group, bucket)
    for record in group:
      summary = summaries.get(record.probe_id)
      if summary is None:
        continue
      sentences.append(
        f"Within-session {record.config['route']} path {record.config['path']} "
        f"bucket {record.n_atoms} steady-state p50 is {summary.point:.6g} ms "
        f"(95% CI [{summary.low:.6g}, {summary.high:.6g}], {summary.scope}, "
        f"seed {summary.seed}).",
      )
  return sentences


def ratio_sentences(
  records: list[Any],
  samples: dict[str, list[np.ndarray]],
  *,
  native_pinned: bool,
  seed: int,
) -> list[str]:
  """Ratio sentences only for `claim_ratio_pairs` results. Each CI is within-session."""
  sentences: list[str] = []
  for left, right in claim_ratio_pairs(records, native_pinned=native_pinned):
    left_samples = samples.get(left.probe_id)
    right_samples = samples.get(right.probe_id)
    if not left_samples or not right_samples:
      continue
    interval = bench_stats.ratio_p50_ci(left_samples, right_samples, seed=seed)
    sentences.append(
      f"Within-session p50 ratio {left.config['route']}/{right.config['route']} "
      f"path {left.config['path']} bucket {left.n_atoms} is {interval.point:.6g} "
      f"(95% CI [{interval.low:.6g}, {interval.high:.6g}], {interval.scope}, "
      f"seed {interval.seed}).",
    )
  return sentences


class _BenchProcess:
  """JSON-lines client for one `run_bench.mjs` process (one Chromium session)."""

  def __init__(self, node_bin: str, site_dir: Path) -> None:
    self.stderr_tail: list[str] = []
    self.proc = subprocess.Popen(  # noqa: S603
      [node_bin, str(LAYER_C_DIR / "run_bench.mjs"), "--site", str(site_dir)],
      cwd=LAYER_C_DIR,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
    )
    threading.Thread(target=self._drain_stderr, daemon=True).start()

  def _drain_stderr(self) -> None:
    stderr = self.proc.stderr
    if stderr is None:
      return
    for line in stderr:
      self.stderr_tail.append(line.rstrip())
      del self.stderr_tail[:-40]

  def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    stdin = self.proc.stdin
    stdout = self.proc.stdout
    if stdin is None or stdout is None:
      msg = "bench process pipes are closed"
      raise RuntimeError(msg)
    stdin.write(json.dumps(payload) + "\n")
    stdin.flush()
    ready, _, _ = select.select([stdout], [], [], timeout_s)
    if not ready:
      msg = f"bench process timed out after {timeout_s}s; stderr={self.stderr_tail!r}"
      raise TimeoutError(msg)
    line = stdout.readline()
    if not line:
      msg = f"bench process closed stdout; stderr={self.stderr_tail!r}"
      raise RuntimeError(msg)
    return json.loads(line)

  def close(self) -> None:
    try:
      if self.proc.poll() is None:
        self.request({"cmd": "close"}, timeout_s=30.0)
    finally:
      if self.proc.poll() is None:
        self.proc.kill()
      self.proc.wait(timeout=30)


def _assemble_site(site_dir: Path, model_files: dict[str, Path]) -> None:
  site_dir.mkdir(parents=True, exist_ok=True)
  (site_dir / "index.html").write_text(_INDEX_HTML)
  shutil.copyfile(LAYER_C_DIR / "bench.mjs", site_dir / "bench.mjs")
  ort_src = LAYER_C_DIR / "node_modules" / "onnxruntime-web" / "dist"
  if not ort_src.is_dir():
    msg = f"onnxruntime-web dist missing at {ort_src} (npm ci in browser/layer_c)"
    raise FileNotFoundError(msg)
  shutil.copytree(ort_src, site_dir / "ort")
  models = site_dir / "models"
  models.mkdir(parents=True, exist_ok=True)
  for name, src in model_files.items():
    shutil.copyfile(src, models / name)


def _rss_kb() -> int:
  import resource

  return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _thread_caps() -> dict[str, Any]:
  affinity: list[int] = []
  if hasattr(os, "sched_getaffinity"):
    affinity = sorted(os.sched_getaffinity(0))
  return {
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "sched_affinity": affinity,
  }


def _cell_plans(paths: list[str], buckets: tuple[int, ...], threads_n: int) -> list[dict[str, Any]]:
  plans: list[dict[str, Any]] = []
  for path in paths:
    for bucket in buckets:
      plans.append(
        {
          "cell_id": f"{path}_L{bucket}_ort_wasm_t1",
          "path": path,
          "bucket": bucket,
          "route": "ort_wasm_t1",
          "context": "ctx_t1",
          "planted_ms": 0.0,
          "claim": True,
        },
      )
      plans.append(
        {
          "cell_id": f"{path}_L{bucket}_ort_wasm_tN",
          "path": path,
          "bucket": bucket,
          "route": "ort_wasm_tN",
          "context": "ctx_tN",
          "planted_ms": 0.0,
          "claim": True,
        },
      )
      plans.append(
        {
          "cell_id": f"{path}_L{bucket}_native_jax_cpu",
          "path": path,
          "bucket": bucket,
          "route": "native_jax_cpu",
          "context": None,
          "planted_ms": 0.0,
          "claim": True,
        },
      )
  if "p04" in paths and 256 in buckets:
    plans.append(
      {
        "cell_id": "p04_L256_aa_dup",
        "path": "p04",
        "bucket": 256,
        "route": "aa_dup",
        "context": "ctx_t1",
        "planted_ms": 0.0,
        "claim": False,
      },
    )
    plans.append(
      {
        "cell_id": "p04_L256_planted_5ms",
        "path": "p04",
        "bucket": 256,
        "route": "planted_5ms",
        "context": "ctx_t1",
        "planted_ms": 5.0,
        "claim": False,
      },
    )
  for plan in plans:
    plan["threads_n"] = threads_n
  return plans


def _time_native(
  path: str,
  bucket: int,
  model: Any,
  stage_set: Any,
  inputs: tuple[Any, ...],
  *,
  n_warmup: int,
  n_iter: int,
  weight_load_s: float,
) -> dict[str, Any]:
  import jax
  import jax.numpy as jnp

  from aminx.export import make_p03_featurize, make_p04_unconditional

  fn = make_p03_featurize(model) if path == "p03" else make_p04_unconditional(model, stage_set)
  args = tuple(jnp.asarray(array) for array in inputs)
  rss_before = _rss_kb()
  compile_start = time.perf_counter()
  compiled = jax.jit(fn).lower(*args).compile()
  compile_s = time.perf_counter() - compile_start
  first_start = time.perf_counter()
  jax.block_until_ready(compiled(*args))
  first_s = time.perf_counter() - first_start
  for _ in range(n_warmup):
    jax.block_until_ready(compiled(*args))
  steady_ms: list[float] = []
  for _ in range(n_iter):
    step_start = time.perf_counter()
    jax.block_until_ready(compiled(*args))
    steady_ms.append((time.perf_counter() - step_start) * 1000.0)
  rss_after = _rss_kb()
  memory_stats = jax.devices()[0].memory_stats()
  return {
    "ok": True,
    "steady_ms": steady_ms,
    "n_warmup": n_warmup,
    "n_iter": len(steady_ms),
    "phases": {
      "weight_load_s": weight_load_s,
      "lower_compile_s": compile_s,
      "first_call_s": first_s,
    },
    "rss_kb_before": rss_before,
    "rss_kb_after": rss_after,
    "jax_memory_stats": memory_stats,
    "jax_memory_available": memory_stats is not None,
  }


class RouteTiming(NamedTuple):
  """Steady-state samples plus the fields both arms actually return."""

  steady_ms: list[float]
  n_warmup: int
  n_iter: int
  phases: dict[str, Any] | None
  threads_observed: Any
  threads_mismatch: bool
  peak_memory_available: bool
  wasm_heap_bytes: Any


def parse_route_timing(timed: dict[str, Any], *, browser: bool) -> RouteTiming:
  """Read one cell's timings.

  `browser/layer_c/bench.mjs` `timeCell` nests `steady_ms`, `n_warmup`, and
  `n_iter` under `phases`. The native arm (`_time_native`) returns those three
  at the top level. `threadsObserved`, `threadsMismatch`, `peakMemory`, and
  `wasmHeapBytes` stay top-level on the browser response.
  """
  if browser:
    phases = timed.get("phases")
    if not isinstance(phases, dict):
      raise KeyError("phases")
    source: dict[str, Any] = phases
  else:
    source = timed
  steady_ms = [float(value) for value in source["steady_ms"]]
  peak = timed.get("peakMemory")
  peak_memory_available = (isinstance(peak, dict) and bool(peak.get("available"))) or bool(
    timed.get("jax_memory_available"),
  )
  phase_map = timed.get("phases")
  return RouteTiming(
    steady_ms=steady_ms,
    n_warmup=int(source["n_warmup"]),
    n_iter=int(source["n_iter"]),
    phases=phase_map if isinstance(phase_map, dict) else None,
    threads_observed=timed.get("threadsObserved"),
    threads_mismatch=bool(timed.get("threadsMismatch")),
    peak_memory_available=peak_memory_available,
    wasm_heap_bytes=timed.get("wasmHeapBytes"),
  )


def record_cell_exception(result: dict[str, Any], cell_key: str, exc: BaseException) -> None:
  """Count one per-cell failure and keep its message. The run itself continues."""
  result["n_cells_error"] = int(result["n_cells_error"]) + 1
  errors = result.setdefault("_errors", {})
  errors[cell_key] = f"{type(exc).__name__}: {exc}"


def collect_cell_timing(
  result: dict[str, Any],
  timed: dict[str, Any],
  *,
  browser: bool,
  cell_key: str,
) -> RouteTiming | None:
  """Parse one fetched cell. A missing timing is recorded, not raised."""
  try:
    return parse_route_timing(timed, browser=browser)
  except Exception as exc:  # noqa: BLE001 -- recorded on the cell; the run still exits 0
    record_cell_exception(result, cell_key, exc)
    return None


def graded_exit_code(*, integrity_failure: bool) -> int:
  """Graded outcomes (per-cell, harness, budget) exit 0. Integrity refusals exit 3."""
  if integrity_failure:
    return 3
  return 0


def _accepted_browser_timing(response: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
  """Unwrap a `run_bench.mjs` time reply. Isolation failure is data on `result`."""
  if not response.get("ok"):
    msg = str(response.get("error"))
    raise RuntimeError(msg)
  timed = response.get("result") or {}
  if not timed.get("ok"):
    if timed.get("crossOriginIsolated") is False:
      result["isolation_ok"] = False
      result["cross_origin_isolated"] = False
    msg = str(timed.get("error") or "cell failed")
    raise RuntimeError(msg)
  return timed


def _browser_spec(
  plan: dict[str, Any],
  inputs: list[dict[str, Any]],
  *,
  n_warmup: int,
  n_iter: int,
) -> dict[str, Any]:
  return {
    "onnx": f"models/{plan['path']}_L{plan['bucket']}.onnx",
    "inputs": inputs,
    "nWarmup": n_warmup,
    "nIter": n_iter,
    "plantedBusyWaitMs": plan["planted_ms"],
  }


def _control_fields(
  samples: dict[str, list[np.ndarray]],
  *,
  seed: int,
) -> dict[str, Any]:
  planted = samples.get("p04_L256_planted_5ms")
  base = samples.get("p04_L256_ort_wasm_t1")
  aa = samples.get("p04_L256_aa_dup")
  fields: dict[str, Any] = {
    "planted_diff_ci_excludes_zero": False,
    "planted_point_estimate_ms": None,
    "planted_point_estimate_ms_available": False,
    "aa_ratio_point": None,
    "aa_ratio_point_available": False,
    "aa_ratio_ci_contains_one": False,
  }
  if planted and base and len(planted) == len(base):
    diff = bench_stats.diff_p50_ci(planted, base, seed=seed)
    fields["planted_point_estimate_ms"] = diff.point
    fields["planted_point_estimate_ms_available"] = True
    fields["planted_diff_ci_excludes_zero"] = bench_stats.ci_excludes(diff, 0.0)
  if aa and base and len(aa) == len(base):
    ratio = bench_stats.ratio_p50_ci(aa, base, seed=seed + 1)
    fields["aa_ratio_point"] = ratio.point
    fields["aa_ratio_point_available"] = True
    fields["aa_ratio_ci_contains_one"] = bench_stats.ci_contains(ratio, 1.0)
  return fields


def run(args: argparse.Namespace) -> dict[str, Any]:
  import layer_a_common as lac
  import layer_b_common as lbc
  import layer_c_common as lcc

  started = time.monotonic()
  deadline = started + float(args.budget_s)
  provenance = lac.provenance()
  git_hash = str(provenance["git_hash"])
  git_clean = bool(provenance["git_clean"])
  os.environ["XTRAX_GIT_SHA"] = git_hash if git_clean else f"{git_hash}-dirty"

  result = _empty_result(git_hash, git_clean)
  result["thread_caps"] = _thread_caps()
  buckets = tuple(int(part) for part in args.buckets.split(",")) if args.buckets else BUCKETS

  artifacts_base = Path(args.artifacts_dir)
  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)
  resolved_dir = artifacts_base / subdir
  manifest = lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)
  result["artifact_subdir"] = subdir

  from aminx.export import EXPORT_BUCKETS

  paths_by_path = lbc.converted_by_path(manifest, PATHS, EXPORT_BUCKETS)
  available = [path for path, ok in paths_by_path.items() if ok]
  result["converted_by_path"] = paths_by_path
  result["n_paths_available"] = len(available)
  if not available:
    result["wall_seconds"] = time.monotonic() - started
    result["_note"] = "T2 BLOCKED table: neither path fully converted."
    return result

  pin = thread_pin_status()
  result["native_threads"] = pin["native_threads"]
  result["_thread_pin"] = pin
  native_pinned = pin["ok"]
  os.environ["XLA_FLAGS"] = PIN_XLA_FLAGS

  from layer_b_build import _synthetic_inputs

  from aminx.export import PINNED_CHECKPOINT_ID
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = f"CHECKPOINT_ID={CHECKPOINT_ID!r} != PINNED_CHECKPOINT_ID={PINNED_CHECKPOINT_ID!r}"
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)
  weight_start = time.perf_counter()
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()
  weight_load_s = time.perf_counter() - weight_start

  node_bin = lcc.find_node(args.node_bin)
  import tempfile

  with tempfile.TemporaryDirectory(prefix="layer_c_bench_") as tmp:
    site_dir = Path(tmp) / "site"
    model_files: dict[str, Path] = {}
    input_specs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    data_dir = site_dir / "data"
    for path in available:
      for bucket in buckets:
        onnx_name = f"{path}_L{bucket}.onnx"
        model_files[onnx_name] = resolved_dir / onnx_name
        arrays = _synthetic_inputs(bucket)
        specs = []
        for index, (array, dtype) in enumerate(
          zip(arrays, ("float32", "float32", "int32", "int32"), strict=True),
        ):
          specs.append(
            lcc.write_raw_input(array, data_dir / f"{path}_L{bucket}_in{index}.bin", dtype),
          )
        input_specs[(path, bucket)] = specs
    _assemble_site(site_dir, model_files)

    if node_bin is None:
      result["harness_ok"] = False
      result["n_cells_error"] = 1
      result["wall_seconds"] = time.monotonic() - started
      result["_note"] = "no node binary found"
      return result

    session = _BenchProcess(node_bin, site_dir)
    try:
      hello = session.request({"cmd": "init"}, timeout_s=float(args.timeout_s))
      if not hello.get("ok"):
        result["harness_ok"] = False
        result["n_cells_error"] = 1
        result["_note"] = str(hello.get("error"))
        result["wall_seconds"] = time.monotonic() - started
        return result
      result["harness_ok"] = True
      result["session_id"] = str(hello.get("sessionId") or uuid.uuid4())
      result["hardware_concurrency"] = int(hello.get("hardwareConcurrency") or 0)
      result["cross_origin_isolated"] = bool(hello.get("crossOriginIsolated"))
      result["isolation_ok"] = bool(hello.get("crossOriginIsolated"))
      threads_n = int(hello.get("threadsN") or 1)
      result["_contexts"] = hello.get("contexts")
      plans = _cell_plans(available, buckets, threads_n)
      result["cells_total"] = len(plans) * N_REPEATS
      samples: dict[str, list[np.ndarray]] = {plan["cell_id"]: [] for plan in plans}
      details: dict[str, Any] = {}
      errors: dict[str, str] = {}
      result["_errors"] = errors
      warmups: list[int] = []
      iterations: list[int] = []
      measured = 0
      below = 0
      peak_available = False
      repeats_done = 0
      budget_expired = False
      rng = np.random.default_rng(int(args.seed))
      native_inputs = {
        (path, bucket): _synthetic_inputs(bucket) for path in available for bucket in buckets
      }

      for repeat_index in range(N_REPEATS):
        if time.monotonic() >= deadline:
          budget_expired = True
          break
        order = list(plans)
        rng.shuffle(order)
        for plan in order:
          if time.monotonic() >= deadline:
            budget_expired = True
            break
          cell_key = f"{plan['cell_id']}#r{repeat_index}"
          try:
            if plan["route"] == "native_jax_cpu":
              timed = _time_native(
                plan["path"],
                plan["bucket"],
                model,
                stage_set,
                native_inputs[(plan["path"], plan["bucket"])],
                n_warmup=N_WARMUP,
                n_iter=N_ITER,
                weight_load_s=weight_load_s,
              )
              browser = False
            else:
              response = session.request(
                {
                  "cmd": "time",
                  "context": plan["context"],
                  "spec": _browser_spec(
                    plan,
                    input_specs[(plan["path"], plan["bucket"])],
                    n_warmup=N_WARMUP,
                    n_iter=N_ITER,
                  ),
                },
                timeout_s=float(args.timeout_s),
              )
              timed = _accepted_browser_timing(response, result)
              browser = True
          except Exception as exc:  # noqa: BLE001 -- cell failures are stored, then the run continues
            record_cell_exception(result, cell_key, exc)
            continue
          parsed = collect_cell_timing(result, timed, browser=browser, cell_key=cell_key)
          if parsed is None:
            continue
          steady = parsed.steady_ms
          if len(steady) < N_ITER or parsed.n_warmup < N_WARMUP:
            below += 1
          if not np.isfinite(steady).all():
            record_cell_exception(
              result,
              cell_key,
              ValueError("non-finite steady-state sample"),
            )
            continue
          samples[plan["cell_id"]].append(np.asarray(steady, dtype=np.float64))
          warmups.append(parsed.n_warmup)
          iterations.append(parsed.n_iter)
          if parsed.peak_memory_available:
            peak_available = True
          details[cell_key] = {
            "route": plan["route"],
            "threads_requested": (
              1
              if plan["route"] in {"ort_wasm_t1", "aa_dup", "planted_5ms"}
              else threads_n
              if plan["route"] == "ort_wasm_tN"
              else 1
            ),
            "threads_observed": parsed.threads_observed,
            "threads_mismatch": parsed.threads_mismatch,
            "source": "ort.env.wasm.numThreads" if plan["context"] else "XLA_FLAGS",
            "phases": parsed.phases,
            "n_warmup": parsed.n_warmup,
            "n_iter": parsed.n_iter,
            "wasm_heap_bytes": parsed.wasm_heap_bytes,
          }
          if plan["route"] == "native_jax_cpu":
            details[cell_key]["threads_observed"] = 1 if native_pinned else None
            details[cell_key]["source"] = "XLA_FLAGS"
          measured += 1
        if budget_expired:
          break
        repeats_done += 1

      result["cells_measured"] = measured
      result["n_repeats"] = repeats_done
      result["min_warmup"] = min(warmups) if warmups else 0
      result["min_iterations"] = min(iterations) if iterations else 0
      result["n_cells_below_minima"] = below + max(0, N_REPEATS - repeats_done) * len(plans)
      result["budget_expired"] = budget_expired
      result["peak_memory_available"] = peak_available
      result.update(_control_fields(samples, seed=int(args.seed)))

      summaries: dict[str, bench_stats.WithinSessionCI] = {}
      records: list[Any] = []
      for plan in plans:
        repeats = samples[plan["cell_id"]]
        if len(repeats) < N_REPEATS:
          continue
        summary = bench_stats.p50_ci(repeats, seed=int(args.seed))
        summaries[plan["cell_id"]] = summary
        if not plan["claim"]:
          continue
        mismatch = any(
          bool(details.get(f"{plan['cell_id']}#r{repeat}", {}).get("threads_mismatch"))
          for repeat in range(N_REPEATS)
        )
        if plan["route"] == "native_jax_cpu":
          threads = "1" if native_pinned else "unpinned"
        elif plan["route"] == "ort_wasm_t1":
          threads = "1"
        else:
          threads = str(threads_n)
        records.append(
          make_probe_record(
            probe_id=plan["cell_id"],
            path=plan["path"],
            bucket=plan["bucket"],
            route=plan["route"],
            session_id=result["session_id"],
            threads=threads,
            total_step_seconds=summary.point / 1000.0,
            threads_mismatch=mismatch,
          ),
        )
      sentences: list[str] = []
      claim_errors: dict[str, str] = {}
      from xtrax.profiling.claims import ClaimValidityError

      try:
        sentences.extend(timing_sentences(records, summaries))
        sentences.extend(
          ratio_sentences(records, samples, native_pinned=native_pinned, seed=int(args.seed)),
        )
      except ClaimValidityError as exc:
        claim_errors["cite"] = f"{type(exc).__name__}: {exc}"
        sentences = []
      result["n_sentences"] = len(sentences)
      result["_sentences"] = sentences
      result["_cells"] = details
      result["_errors"] = errors
      result["_claim_errors"] = claim_errors
      result["_arms"] = {
        "ort_wasm_t1": {
          "threads_requested": 1,
          "threads_observed": 1,
          "source": "ort.env.wasm.numThreads",
        },
        "ort_wasm_tN": {
          "threads_requested": threads_n,
          "threads_observed": next(
            (
              details[key].get("threads_observed")
              for key in details
              if details[key].get("route") == "ort_wasm_tN"
            ),
            None,
          ),
          "source": "ort.env.wasm.numThreads",
        },
        "native_jax_cpu": {
          "threads_requested": 1,
          "threads_observed": 1 if native_pinned else None,
          "source": "XLA_FLAGS",
          "native_threads": result["native_threads"],
        },
      }
    finally:
      session.close()

  from importlib.metadata import PackageNotFoundError
  from importlib.metadata import version as pkg_version

  import jax

  def _ver(name: str) -> str:
    try:
      return pkg_version(name)
    except PackageNotFoundError:
      return "not-installed"

  result["versions"] = {
    "jax": jax.__version__,
    "onnxruntime": _ver("onnxruntime"),
    "xtrax": _ver("xtrax"),
    "node": lcc.node_env_report(node_bin) if node_bin else {},
  }
  result["wall_seconds"] = time.monotonic() - started
  return result


def _write_report(path: Path, sentences: list[str]) -> None:
  body = (
    "\n".join(sentences)
    if sentences
    else "No within-session sentence was emitted for this session."
  )
  path.write_text(body + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  import layer_a_common as lac

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--artifacts-dir", required=True, type=Path)
  parser.add_argument("--buckets", default=",".join(str(bucket) for bucket in BUCKETS))
  parser.add_argument("--node-bin", default=None)
  parser.add_argument("--timeout-s", type=float, default=180.0)
  parser.add_argument("--seed", type=int, default=260928)
  parser.add_argument("--budget-s", type=float, default=BUDGET_S)
  args = parser.parse_args(argv)

  integrity_failure = False
  try:
    result = run(args)
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_c_bench: integrity refusal: %s", exc)
    result = _empty_result("", False)
    result["_integrity_error"] = str(exc)
    integrity_failure = True
  exit_code = graded_exit_code(integrity_failure=integrity_failure)

  lac.emit(result, args.out)
  if not lac.differential_mode()["active"]:
    _write_report(Path(args.out).with_name("report.md"), list(result.get("_sentences") or []))
  logger.info(
    "layer_c_bench: native_threads=%s n_paths=%s cells=%s/%s errors=%s sentences=%s exit=%s",
    result.get("native_threads"),
    result.get("n_paths_available"),
    result.get("cells_measured"),
    result.get("cells_total"),
    result.get("n_cells_error"),
    result.get("n_sentences"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
