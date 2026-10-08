"""X5b browser bench: per-design wall time of the PottsMPNN browser page in REAL headless Chromium.

task_id 261007_potts-onnx-export, backlog #5815 (X5 part B). Sidecar: potts_browser_bench.bth.toml
(pre-registered).

Mirrors p07_split_browser_bench.py: a planted delay is timed in the page through the same clock and
the same await wrapper as the real cells, and nothing is published unless the page sees it. NO
performance DIRECTION is pre-registered; the graded pass/fail is measurement validity, and the
timings are result fields.

Measured per design: the full browser path, PDB text -> buildPottsInputs -> applyPottsControls ->
sampler.sample (encode, decode, energy, refine), with the cheap split into `prep` (build + controls)
and `sample` (the four graphs). Cells: 3dkm L128 and 3gg7 L256, default controls, seed 21. Each cell
runs one warm-up and `reps` measured repetitions, at 1 and at 4 ORT threads. Session-creation cost
is recorded per bucket. The effective thread count and crossOriginIsolated are recorded as the page
reports them, not as requested.

Site assembly, the harness runner and the noise/cell construction come from potts_browser_parity,
so the bench serves byte-identical sites to the parity gate.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit
from scripts.browser_validation.layer_c_common import (
  LAYER_C_DIR,
  find_node,
  node_env_report,
)
from scripts.browser_validation.potts_browser_parity import (
  CELLS,
  assemble_potts_site,
  git_state,
  noise_cell,
  run_harness,
)
from scripts.browser_validation.potts_knobs_gate import DEFAULT, _require
from scripts.browser_validation.potts_loop_gate import _verify_artifacts

logger = logging.getLogger("potts_browser_bench")

THREADS = (1, 4)
BENCH_SEED = 21
CTRL_LO = 0.9
CTRL_SLACK_MS = 2000.0


def _summarize(run: dict[str, Any], payload: dict[str, Any], n_cells: int, reps: int) -> dict[str, Any]:
  ok_cells = [c for c in payload.get("cells", []) if c.get("ok")]
  walls = [w for c in ok_cells for w in c["wall_ms_all"]]
  preps = [w for c in ok_cells for w in c["prep_ms_all"]]
  samples = [w for c in ok_cells for w in c["sample_ms_all"]]
  planted = float(payload.get("planted_ms", 0.0))
  measured = float(payload.get("ctrl_measured_ms", -1.0))
  run.update(
    cross_origin_isolated=bool(payload.get("cross_origin_isolated")),
    num_threads_after_init=int(payload.get("num_threads_after_init", -1)),
    num_threads_effective=int(payload.get("num_threads_effective", -1)),
    ctrl_measured_ms=measured,
    ctrl_ratio=(measured / planted) if planted > 0 else -1.0,
    ctrl_in_window=bool(
      planted > 0 and CTRL_LO * planted <= measured <= planted + CTRL_SLACK_MS,
    ),
    cells_ok=len(ok_cells),
    cells_total=n_cells,
    n_walls=len(walls),
    reps_ok=all(len(c["wall_ms_all"]) == reps for c in ok_cells) and len(ok_cells) == n_cells,
    median_ms=float(statistics.median(walls)) if walls else -1.0,
    min_ms=float(min(walls)) if walls else -1.0,
    max_ms=float(max(walls)) if walls else -1.0,
    median_prep_ms=float(statistics.median(preps)) if preps else -1.0,
    median_sample_ms=float(statistics.median(samples)) if samples else -1.0,
    session_create_total_ms=float(sum(float(v) for v in payload.get("session_create_ms", {}).values())),
    per_cell_median_ms={
      c["id"]: float(statistics.median(c["wall_ms_all"])) for c in ok_cells if c["wall_ms_all"]
    },
  )
  return run


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True)
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))),
                      help="directory whose node_modules/onnxruntime-web/dist supplies the ORT files")
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--planted-ms", type=float, default=250.0)
  parser.add_argument("--timeout-ms", type=int, default=3_600_000)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean,
    "manifest_sha256": "", "artifacts_verified": False,
    "onnxruntime_web_version": "", "playwright_version": "", "node_version": "",
    "reps": args.reps, "warmup": args.warmup, "planted_ms": args.planted_ms,
    "cells_total": 0, "bench_ok": False, "ctrl_measured_ms": -1.0, "ctrl_in_window": False,
    "threads_requested": list(THREADS), "runs": [], "failure": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True
    result["manifest_sha256"] = args.manifest_sha256

    cells = [
      noise_cell(
        f"default__{pdb.removesuffix('.pdb')}_L{bucket}__s{BENCH_SEED}",
        args.potts_root / "inputs" / "example_pdbs" / pdb,
        bucket, {}, dict(DEFAULT), BENCH_SEED, args.work_dir / "cells",
      )
      for pdb, bucket in CELLS
    ]
    result["cells_total"] = len(cells)

    node = find_node(args.node_bin)
    _require(node is not None, "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)")
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    site = args.work_dir / "site"
    assemble_potts_site(
      site, args.models, args.ort_dir, cells,
      reps=args.reps, warmup=args.warmup, planted_ms=args.planted_ms,
    )
    logger.info("assembled site: %d cells x (%d warm-up + %d reps)", len(cells), args.warmup, args.reps)

    for threads in THREADS:
      run: dict[str, Any] = {"threads_requested": threads, "harness_ok": False}
      result["runs"].append(run)
      try:
        harness = run_harness(node, site, args.work_dir / f"browser_t{threads}", threads, args.timeout_ms)
        run["chromium_version"] = str(harness.get("chromiumVersion") or "")
        if not harness.get("harnessOk"):
          run["harness_error"] = str(harness.get("harnessError"))[:800]
          continue
        run["harness_ok"] = True
        _summarize(run, harness["result"], len(cells), args.reps)
        logger.info(
          "threads %d (effective %d, isolated %s): median %.0f ms | prep %.1f | sample %.0f | "
          "session %.0f ms | control %.1f ms",
          threads, run["num_threads_effective"], run["cross_origin_isolated"], run["median_ms"],
          run["median_prep_ms"], run["median_sample_ms"], run["session_create_total_ms"],
          run["ctrl_measured_ms"],
        )
      except Exception as exc:
        run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
        logger.exception("threads %d bench run failed", threads)

    runs = result["runs"]
    result["bench_ok"] = len(runs) == len(THREADS) and all(
      r["harness_ok"] and r.get("cells_ok") == len(cells) and r.get("reps_ok") for r in runs
    )
    result["ctrl_in_window"] = bool(runs) and all(r.get("ctrl_in_window") for r in runs)
    measured = [r["ctrl_measured_ms"] for r in runs if "ctrl_measured_ms" in r]
    result["ctrl_measured_ms"] = float(min(measured)) if measured else -1.0
  except Exception as exc:
    logger.exception("browser bench failed")
    result["failure"] = f"{type(exc).__name__}: {exc}"[:4000]

  result["outcome"] = (
    "incomplete"
    if not result["bench_ok"]
    or result["cells_total"] < 2
    or result["reps"] < 3
    or not result["git_clean"]
    or result["failure"]
    else "timer_blind"
    if not result["ctrl_in_window"]
    else "pass"
  )
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("outcome=%s", result["outcome"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
