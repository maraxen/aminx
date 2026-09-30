"""G4b: per-design wall time in a REAL browser, with a browser timer control.

G4 (run be45e729) measured timings under Node and validated `process.hrtime` against a
planted delay first. The browser uses `performance.now` in a different runtime, and that
clock has never been checked here, so the ~2.4x threading speedup visible in the parity
gates (`d6be02ea` vs `da02516e`) is currently INDICATIVE only -- those gates grade
correctness and carry no timer control. Quoting it as measured would repeat the mistake
that got this project's first benchmark discarded.

This gate plants a delay in the page, through the same clock and the same await machinery
the per-design timings use, and publishes nothing unless the page sees it. As in G4, no
performance DIRECTION is pre-registered: the graded pass/fail is measurement validity and
the timings are result fields.

Site assembly and cell construction are imported from the browser parity gate rather than
duplicated, so both gates serve byte-identical sites and a timing difference cannot be an
assembly difference.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
_BV = _REPO_ROOT / "scripts" / "browser_validation"
if str(_BV) not in sys.path:
  sys.path.insert(0, str(_BV))

logger = logging.getLogger("p07_split_browser_bench")

_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")


def _git_state(repo: Path) -> tuple[str, bool]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  )


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--ort-dir", type=Path, required=True)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--cells", type=int, default=4)
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--threads", type=int, default=1)
  parser.add_argument("--planted-ms", type=float, default=250.0)
  parser.add_argument("--timeout-ms", type=int, default=3_600_000)
  parser.add_argument("--node-bin", type=str, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
  )

  git_hash, git_clean = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "reps": args.reps,
    "planted_ms": args.planted_ms,
    "ctrl_measured_ms": -1.0,
    "ctrl_ratio": -1.0,
    "cells_total": 0,
    "num_threads_requested": args.threads,
    "num_threads_effective": -1,
    "cross_origin_isolated": False,
    "chromium_version": "",
    "median_ms": -1.0,
    "min_ms": -1.0,
    "max_ms": -1.0,
    "session_create_total_ms": -1.0,
    "bench_ok": False,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "failure": "",
  }

  try:
    from p07_knobs_gate import (  # noqa: PLC0415
      FULL_KNOBS,
      FULL_SEEDS,
      P07_KEYS,
      _load_manifest,
      _node_bin,
      _pad_geometry,
      _parse_geometry,
      _select_fixtures,
      _structure_payload,
      _with_chain_split,
      js_build,
      runspec_for,
    )
    from p07_split_browser_parity import _assemble_site, _spec  # noqa: PLC0415

    node = _node_bin(args.node_bin)
    work = Path(tempfile.mkdtemp(prefix="p07_split_bbench_"))
    geometries = [
      _parse_geometry(row, work / "canonical")
      for row in _select_fixtures(_load_manifest(), [args.bucket])
    ]

    cells: list[dict[str, Any]] = []
    for geom in geometries:
      if int(geom["n_real"]) > args.bucket:
        continue
      padded = _pad_geometry(geom, args.bucket)
      chain_split = _pad_geometry(_with_chain_split(geom), args.bucket)
      for knob in FULL_KNOBS:
        host = chain_split if knob in ("chains_to_design", "all_combined") else padded
        for seed in FULL_SEEDS[:1]:
          if len(cells) >= args.cells:
            break
          spec = runspec_for(knob, seed, host)
          arrays = js_build(_structure_payload(host), spec, node)
          cells.append(
            {
              "name": f"{geom['name']}_{knob}_s{seed}",
              "arrays": {k: _spec(np.asarray(arrays[k])) for k in P07_KEYS},
            }
          )
        if len(cells) >= args.cells:
          break
      if len(cells) >= args.cells:
        break

    site = work / "site"
    out_dir = work / "results"
    _assemble_site(site, args.release_dir, args.ort_dir, args.bucket)
    (site / "cells.json").write_text(
      json.dumps(
        {
          "bucket": args.bucket,
          "reps": args.reps,
          "planted_ms": args.planted_ms,
          "cells": cells,
        }
      )
    )
    logger.info("assembled site: %d cells x %d reps", len(cells), args.reps)

    runner = _REPO_ROOT / "browser" / "layer_c" / "run_p07.mjs"
    cmd = [
      node, str(runner),
      "--site", str(site),
      "--out-dir", str(out_dir),
      "--num-threads", str(args.threads),
      "--timeout-ms", str(args.timeout_ms),
    ]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, timeout=args.timeout_ms // 1000 + 600)  # noqa: S603
    harness_path = out_dir / "harness.json"
    if not harness_path.is_file():
      msg = f"browser harness wrote no harness.json (rc={proc.returncode})"
      raise RuntimeError(msg)
    harness = json.loads(harness_path.read_text())
    result["chromium_version"] = str(harness.get("chromiumVersion") or "")
    if not harness.get("harnessOk"):
      msg = f"browser harness failed: {str(harness.get('harnessError'))[:800]}"
      raise RuntimeError(msg)

    payload = harness["result"]
    result["bench_ok"] = True
    result["ctrl_measured_ms"] = float(payload.get("ctrl_measured_ms", -1.0))
    if args.planted_ms > 0:
      result["ctrl_ratio"] = result["ctrl_measured_ms"] / args.planted_ms
    result["num_threads_effective"] = int(payload.get("num_threads_effective", -1))
    result["cross_origin_isolated"] = bool(payload.get("cross_origin_isolated"))
    result["cells_total"] = int(payload.get("n_cells", 0))
    sc = payload.get("session_create_ms", {})
    result["session_create_total_ms"] = float(sum(float(v) for v in sc.values()))

    walls = [w for c in payload["cells"] for w in c.get("wall_ms_all", [])]
    if walls:
      result["median_ms"] = float(statistics.median(walls))
      result["min_ms"] = float(min(walls))
      result["max_ms"] = float(max(walls))

    logger.info(
      "control: planted %.0f ms measured %.1f ms (ratio %.3f)",
      args.planted_ms, result["ctrl_measured_ms"], result["ctrl_ratio"],
    )
    logger.info(
      "threads %d | median %.0f ms | min %.0f | max %.0f",
      result["num_threads_effective"], result["median_ms"],
      result["min_ms"], result["max_ms"],
    )

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G4b browser bench failed")

  lo = 0.9 * args.planted_ms
  hi = args.planted_ms + 2000.0
  result["outcome"] = (
    "incomplete"
    if not result["bench_ok"]
    or result["cells_total"] < 2
    or result["reps"] < 3
    or not result["git_clean"]
    or result["failure"]
    else "timer_blind"
    if result["ctrl_measured_ms"] < lo or result["ctrl_measured_ms"] > hi
    else "pass"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
