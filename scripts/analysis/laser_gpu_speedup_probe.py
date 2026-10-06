#!/usr/bin/env python3
"""Measure LASEr sampling throughput on a TITAN RTX, against the CPU reference.

WHY THIS EXISTS. The ~223 h LASEr distributional confirm has no software lever
left against it. Measured 261006: the planner hypothesis is falsified (chunk size
never reaches the path), max_length padding is a null four independent ways, and
the chunk-width lever FAILED ITS OWN CONTROL (width 1 vs width 8 at equal n came
in at 1.014 against a pre-registered >= 1.20, so per-chunk dispatch is
negligible, and the apparent width effect was the size of the within-
configuration replicate spread). The surviving explanation is that LASEr sampling
is simply heavy: ~44-60 s/sample at L=154 across fourteen units. That leaves
HARDWARE as the only lever, and titanix has two completely idle 24 GB cards.

WHAT IT MEASURES. One LASEr sampling unit at n=8, width 8, max_length 512 -- the
same configuration, structure and checkpoint as the CPU units already measured --
run through THE SAME WORKER CODE so the number is directly comparable rather than
produced by a parallel implementation.

THE CONTROL THAT MATTERS. A CUDA venv that silently falls back to CPU reports "no
speedup", which reads exactly like a real negative finding. jax prints
"...a CUDA-enabled jaxlib is not installed. Falling back to cpu." and carries on.
So this samples nvidia-smi THROUGHOUT the timed run and requires the pinned cards
to show real memory and nonzero utilisation. A backend string alone is not
enough: it is read in a different process from the one doing the work.

FOUR CONSTRAINTS, each of which would invalidate the probe if ignored.
  1. PINNED TO GPUs 2 AND 3. GPUs 0 and 1 hold a vLLM service at 21.5 GB each
     that is not mine to disturb.
  2. A SEPARATE VENV, never a vehicle's. A CUDA jaxlib changes the numerical
     backend and this is a parity project under a freeze.
  3. TITAN RTX IS TURING (sm_7.5) AND ITS f64 THROUGHPUT IS 1/32 OF f32. This
     probe is therefore informative for f32 production sampling -- which is what
     the distributional confirm does -- and a POOR guide for the f64 parity
     tiers, which could well run slower here than on CPU. DO NOT generalise this
     speedup to the f64 waves. This is the detail most likely to be missed,
     because "we have GPUs now" invites exactly that generalisation.
  4. IT MEASURES A CEILING, NOT A PLAN. Two cards against Engaging's 40-way
     array: even a large per-unit speedup may leave the array the better shape
     for the full 40-unit confirm. The job is to size the speedup so the
     provisioning decision rests on a number instead of a hope.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The CPU reference. Median of the two graded n=8/width=8/max_length=512 units
# from run cw-work-261006c (ref_n8_w8_p512: 52.68 and 44.43 s/sample). Fixed
# here BEFORE the GPU number exists so the comparison cannot drift to flatter it.
_CPU_REF_S_PER_SAMPLE = 48.555
_CPU_REF_UNITS = (52.68, 44.43)
_CPU_RANGE = (43.65, 60.25)  # full spread across all 14 CPU units, any config

_PINNED_DEVICES = "2,3"
_N = 8
_WIDTH = 8
_MAX_LENGTH = 512
_UNIT_TIMEOUT_S = 1800
_SAMPLE_EVERY_S = 3.0
_MIN_GPU_MEM_MIB = 256  # a real jax process reserves far more than an idle card
_MIN_GPU_UTIL_PCT = 5


class _Refused(RuntimeError):
  """The instrument refuses to grade."""


def _smi(fields: str) -> list[list[str]]:
  proc = subprocess.run(  # noqa: S603
    ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],  # noqa: S607
    capture_output=True,
    text=True,
    timeout=60,
    check=False,
  )
  if proc.returncode != 0:
    msg = f"nvidia-smi failed: {proc.stderr[-400:]}"
    raise _Refused(msg)
  return [[c.strip() for c in line.split(",")] for line in proc.stdout.strip().splitlines()]


class _GpuSampler:
  """Samples the pinned cards while the unit runs. This is the control."""

  def __init__(self, indices: tuple[int, ...]) -> None:
    self._indices = indices
    self._samples: list[dict[str, Any]] = []
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._loop, daemon=True)

  def _loop(self) -> None:
    while not self._stop.is_set():
      try:
        rows = _smi("index,memory.used,utilization.gpu")
      except Exception:  # noqa: BLE001 - a sampling hiccup must not kill the run
        self._stop.wait(_SAMPLE_EVERY_S)
        continue
      for row in rows:
        if int(row[0]) in self._indices:
          self._samples.append(
            {"index": int(row[0]), "mem_mib": int(row[1]), "util_pct": int(row[2])}
          )
      self._stop.wait(_SAMPLE_EVERY_S)

  def __enter__(self) -> _GpuSampler:
    self._thread.start()
    return self

  def __exit__(self, *_: object) -> None:
    self._stop.set()
    self._thread.join(timeout=30)

  def summary(self) -> dict[str, Any]:
    if not self._samples:
      return {"n_samples": 0, "peak_mem_mib": 0, "peak_util_pct": 0, "per_card": {}}
    per_card: dict[str, Any] = {}
    for index in self._indices:
      mine = [s for s in self._samples if s["index"] == index]
      if mine:
        per_card[str(index)] = {
          "peak_mem_mib": max(s["mem_mib"] for s in mine),
          "peak_util_pct": max(s["util_pct"] for s in mine),
        }
    return {
      "n_samples": len(self._samples),
      "peak_mem_mib": max(s["mem_mib"] for s in self._samples),
      "peak_util_pct": max(s["util_pct"] for s in self._samples),
      "per_card": per_card,
    }


def _backend_report(python: Path, env: dict[str, str]) -> dict[str, Any]:
  code = (
    "import json, jax, jaxlib\n"
    "print(json.dumps({'backend': jax.default_backend(),"
    " 'devices': [str(d) for d in jax.devices()],"
    " 'jax': jax.__version__, 'jaxlib': jaxlib.__version__}))\n"
  )
  proc = subprocess.run(  # noqa: S603
    [str(python), "-c", code], capture_output=True, text=True, timeout=600, env=env, check=False
  )
  if proc.returncode != 0:
    msg = f"backend probe failed: {proc.stderr[-600:]}"
    raise _Refused(msg)
  line = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
  if not line:
    msg = f"backend probe printed no json: {proc.stdout[-400:]}"
    raise _Refused(msg)
  return json.loads(line[-1])


def _self_test() -> None:
  """Synthetic checks on the grading arithmetic, two of which must refuse."""
  assert _speedup(48.555, 48.555) == 1.0
  assert abs(_speedup(48.555, 4.8555) - 10.0) < 1e-9
  assert abs(_speedup(48.555, 97.11) - 0.5) < 1e-9
  for bad in (0.0, -1.0):
    try:
      _speedup(48.555, bad)
    except _Refused:
      pass
    else:  # pragma: no cover
      msg = f"accepted gpu seconds {bad}"
      raise AssertionError(msg)
  # A sampler that saw nothing must not read as a healthy GPU run.
  empty = _GpuSampler((2, 3)).summary()
  assert empty["n_samples"] == 0, empty
  assert empty["peak_util_pct"] == 0, empty
  assert not _gpu_did_work(empty), empty
  # An idle card (a few MiB, 0%) must also fail the control.
  idle = {"n_samples": 10, "peak_mem_mib": 38, "peak_util_pct": 0, "per_card": {}}
  assert not _gpu_did_work(idle), idle
  # A card doing real work must pass it.
  busy = {"n_samples": 10, "peak_mem_mib": 4096, "peak_util_pct": 71, "per_card": {}}
  assert _gpu_did_work(busy), busy
  logger.info("self-test passed (9 checks, 2 refusals)")


def _speedup(cpu_s: float, gpu_s: float) -> float:
  if gpu_s <= 0:
    msg = f"non-positive gpu seconds/sample: {gpu_s}"
    raise _Refused(msg)
  return cpu_s / gpu_s


def _gpu_did_work(summary: dict[str, Any]) -> bool:
  return (
    int(summary.get("n_samples", 0)) > 0
    and int(summary.get("peak_mem_mib", 0)) >= _MIN_GPU_MEM_MIB
    and int(summary.get("peak_util_pct", 0)) >= _MIN_GPU_UTIL_PCT
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--venv-python", type=Path, required=False)
  parser.add_argument("--worker-script", type=Path, required=False)
  parser.add_argument("--pdb", type=Path, required=False)
  parser.add_argument("--checkpoint", type=Path, required=False)
  parser.add_argument("--work-dir", type=Path, required=False)
  parser.add_argument("--out", type=Path, required=False)
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  _self_test()
  if args.self_test_only:
    return 0
  for name in ("venv_python", "worker_script", "pdb", "checkpoint", "work_dir"):
    if getattr(args, name) is None:
      parser.error(f"--{name.replace('_', '-')} is required unless --self-test-only")

  work: Path = args.work_dir
  work.mkdir(parents=True, exist_ok=True)
  env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": _PINNED_DEVICES,
    "JAX_PLATFORMS": "",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
  }

  backend = _backend_report(args.venv_python, env)
  logger.info("backend report: %s", backend)

  unit_out = work / "gpu_unit.json"
  argv = [
    str(args.venv_python),
    str(args.worker_script),
    "--worker",
    "--worker-label",
    "gpu_n8_w8_p512",
    "--worker-n",
    str(_N),
    "--worker-width",
    str(_WIDTH),
    "--worker-max-length",
    str(_MAX_LENGTH),
    "--worker-pair",
    "gpu_probe",
    "--worker-repeat",
    "0",
    "--worker-out",
    str(unit_out),
    "--pdb",
    str(args.pdb),
    "--checkpoint",
    str(args.checkpoint),
    "--cache-dir",
    str(work / "xla_cache"),
  ]

  started = time.monotonic()
  with _GpuSampler((2, 3)) as sampler:
    proc = subprocess.run(  # noqa: S603
      argv, capture_output=True, text=True, timeout=_UNIT_TIMEOUT_S, env=env, check=False
    )
    gpu_state = sampler.summary()
  wall_s = time.monotonic() - started

  unit: dict[str, Any] = {}
  if unit_out.is_file():
    unit = json.loads(unit_out.read_text(encoding="utf-8"))
  gpu_s = float(unit.get("seconds_per_sample", 0.0) or 0.0)

  backend_is_gpu = backend.get("backend") == "gpu"
  gpu_worked = _gpu_did_work(gpu_state)
  unit_ok = proc.returncode == 0 and gpu_s > 0
  controls_detected = backend_is_gpu and gpu_worked and unit_ok

  speedup = _speedup(_CPU_REF_S_PER_SAMPLE, gpu_s) if gpu_s > 0 else 0.0
  result = {
    "controls_detected": controls_detected,
    "backend_is_gpu": backend_is_gpu,
    "gpu_did_work": gpu_worked,
    "unit_ok": unit_ok,
    "gpu_seconds_per_sample": gpu_s,
    "cpu_ref_seconds_per_sample": _CPU_REF_S_PER_SAMPLE,
    "cpu_ref_units": list(_CPU_REF_UNITS),
    "cpu_full_range": list(_CPU_RANGE),
    "speedup": speedup,
    "wall_s": wall_s,
    "gpu_state": gpu_state,
    "backend_report": backend,
    "worker_returncode": proc.returncode,
    "worker_stderr_tail": proc.stderr[-2000:],
    "pinned_devices": _PINNED_DEVICES,
    "turing_f64_caveat": (
      "TITAN RTX is sm_7.5; f64 throughput is 1/32 of f32. This speedup is for f32 "
      "sampling only and must NOT be generalised to the f64 parity tiers."
    ),
  }

  payload = json.dumps(result, indent=2) + "\n"
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).write_text(payload, encoding="utf-8")
    logger.info("emitted result to BTH_RESULTS_PATH=%s", bth_results)
  else:
    logger.warning("BTH_RESULTS_PATH unset; this run will NOT be graded by its sidecar")
  print(payload)  # noqa: T201
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
