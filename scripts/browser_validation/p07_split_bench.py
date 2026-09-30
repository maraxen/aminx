"""G4: wall-clock cost of the split vs the monolith under ORT-Web wasm.

Reports timings only if the clock first proves it can see a planted delay. An earlier
benchmark in this project produced numbers whose timer control was never established and
they had to be discarded; this gate refuses to publish anything under that condition.

No performance direction is pre-registered. The split trades one in-graph loop for L
JavaScript-to-wasm round trips per design, and which dominates at these sizes is the
unknown being measured. The graded pass/fail is measurement validity; the timings are
result fields.

This gate needs only the INPUT arrays, not reference outputs, so unlike the parity gates
it skips JAX and the correctness arms entirely -- setup is seconds rather than the
~15 minutes those spend building references. Correctness is G1/G1n's job, already passed;
re-deriving it here would cost a quarter hour to prove nothing new.
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

logger = logging.getLogger("p07_split_bench")

_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")


def _git_state(repo: Path) -> tuple[str, bool]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo,
      capture_output=True,
      text=True,
      check=True,
      timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  )


def _spec(a: np.ndarray) -> dict[str, Any]:
  arr = np.asarray(a)
  dtype = {"float32": "float32", "int32": "int32", "bool": "bool"}.get(str(arr.dtype))
  if dtype is None:
    msg = f"unsupported dtype {arr.dtype}"
    raise ValueError(msg)
  return {"dtype": dtype, "shape": list(arr.shape), "data": arr.reshape(-1).tolist()}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--monolith", type=Path, required=True)
  parser.add_argument("--ort-dir", type=Path, required=True)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--cells", type=int, default=8)
  parser.add_argument("--reps", type=int, default=5)
  parser.add_argument("--planted-ms", type=float, default=250.0)
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
    "split_median_ms": -1.0,
    "monolith_median_ms": -1.0,
    "split_over_monolith": -1.0,
    "split_min_ms": -1.0,
    "split_max_ms": -1.0,
    "monolith_min_ms": -1.0,
    "monolith_max_ms": -1.0,
    "session_create_split_total_ms": -1.0,
    "session_create_monolith_ms": -1.0,
    "decoder_calls_per_design": -1,
    "peak_rss_bytes": -1,
    "num_threads": -1,
    "ort_version": "",
    "bench_ok": False,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "failure": "",
  }

  try:
    from scripts.browser_validation.p07_knobs_gate import (  # noqa: PLC0415
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

    node = _node_bin(args.node_bin)
    work = Path(tempfile.mkdtemp(prefix="p07_split_bench_"))
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

    cells_path = work / "cells.json"
    bench_out = work / "bench_out.json"
    cells_path.write_text(json.dumps(cells))
    logger.info("dumped %d cells for benchmarking", len(cells))

    runner = _REPO_ROOT / "browser" / "aminx-sampler" / "bench_split_node.mjs"
    cmd = [
      node, str(runner),
      "--cells", str(cells_path),
      "--models", str(args.release_dir),
      "--monolith", str(args.monolith),
      "--bucket", str(args.bucket),
      "--reps", str(args.reps),
      "--planted-ms", str(args.planted_ms),
      "--out", str(bench_out),
      "--ort-dir", str(args.ort_dir),
    ]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, timeout=14400)  # noqa: S603
    if proc.returncode != 0 or not bench_out.is_file():
      msg = f"bench runner failed rc={proc.returncode}; stderr streamed above"
      raise RuntimeError(msg)

    b = json.loads(bench_out.read_text())
    result["bench_ok"] = True
    result["ctrl_measured_ms"] = float(b["ctrl_measured_ms"])
    result["ctrl_ratio"] = float(b["ctrl_measured_ms"]) / float(b["planted_ms"])
    result["num_threads"] = int(b["num_threads"])
    result["ort_version"] = str(b["ort_version"])
    result["peak_rss_bytes"] = int(b["peak_rss_bytes"])
    sc = b["session_create_ms"]
    result["session_create_split_total_ms"] = float(
      sum(sc[k] for k in ("encoder", "wave", "decoder", "fuse"))
    )
    result["session_create_monolith_ms"] = float(sc["monolith"])

    # Median over the flattened (cell, rep) population: every design is a sample, and no
    # cell is weighted more heavily than another because all run the same rep count.
    split_all = [x for c in b["cells"] for x in c["split_ms"]]
    mono_all = [x for c in b["cells"] for x in c["monolith_ms"]]
    result["cells_total"] = len(b["cells"])
    if split_all and mono_all:
      result["split_median_ms"] = float(statistics.median(split_all))
      result["monolith_median_ms"] = float(statistics.median(mono_all))
      result["split_over_monolith"] = result["split_median_ms"] / result["monolith_median_ms"]
      result["split_min_ms"] = float(min(split_all))
      result["split_max_ms"] = float(max(split_all))
      result["monolith_min_ms"] = float(min(mono_all))
      result["monolith_max_ms"] = float(max(mono_all))
    result["decoder_calls_per_design"] = args.bucket

    logger.info(
      "control: planted %.0f ms measured %.1f ms (ratio %.3f)",
      result["planted_ms"],
      result["ctrl_measured_ms"],
      result["ctrl_ratio"],
    )
    logger.info(
      "split median %.1f ms | monolith median %.1f ms | ratio %.2f",
      result["split_median_ms"],
      result["monolith_median_ms"],
      result["split_over_monolith"],
    )

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G4 bench failed")

  lo = 0.9 * result["planted_ms"]
  hi = result["planted_ms"] + 2000.0
  result["outcome"] = (
    "incomplete"
    if not result["bench_ok"]
    or result["cells_total"] < 4
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
