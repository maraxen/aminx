"""S8-06: wall-clock of runner ``sample`` with length bucketing on vs off (spec S8, gate G-SPEED).

Cells are (structure, mode) pairs; each runs in its own subprocess with its own timeout and writes
``<out>/cells/<cell>.json`` on completion, so a crash or timeout loses at most one cell. A rerun reuses a
cell only when its key (sha256 of the structure file, the call parameters and the aminx source tree) matches,
and the summary records which cells were reused.

Per cell: one warm-up call (pays compilation), then ``--reps`` timed calls in the same process; the cell's
statistic is the median of the timed calls. 1ubq (span 76) buckets 512 -> 128; 3pgk (span 415) stays at 512,
so it is the control whose ratio should sit near 1.

Usage (titanix, CPU):
  bth run --project-slug aminx -- uv run --no-sync python3 scripts/parity/bench_length_bucketing.py \
    --out outputs/s8_bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("bench_length_bucketing")

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "tests" / "data"
STRUCTURES = {"1ubq": "1ubq.pdb", "3pgk": "3pgk.pdb"}
MODES = {"bucketed": True, "optout": False}
PARAMS = {"num_samples": 4, "temperature": 0.1, "random_seed": 7, "max_length": 512}
SPEEDUP_PASS = 3.0
SPEEDUP_MARGINAL = 1.5
CONTROL_BAND = (0.8, 1.25)


def _sha256_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash() -> str:
  digest = hashlib.sha256()
  for path in sorted((_ROOT / "src" / "aminx").rglob("*.py")):
    digest.update(str(path.relative_to(_ROOT)).encode())
    digest.update(path.read_bytes())
  return digest.hexdigest()


def _cell_key(structure: str, mode: str, reps: int, source_hash: str) -> str:
  payload = {
    "structure_sha256": _sha256_file(_DATA / STRUCTURES[structure]),
    "mode": mode,
    "params": PARAMS,
    "reps": reps,
    "source_sha256": source_hash,
  }
  return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_cell(structure: str, mode: str, reps: int, out: Path, key: str) -> None:
  """Time one cell in this process and write its JSON atomically."""
  import jax

  from aminx.host.runner import sample

  pdb = str(_DATA / STRUCTURES[structure])
  kwargs = {**PARAMS, "length_bucketing": MODES[mode]}

  def call() -> float:
    start = time.perf_counter()
    result = sample(inputs=[pdb], **kwargs)
    jax.block_until_ready(result["sequences"])
    return time.perf_counter() - start

  warmup = call()
  timed = [call() for _ in range(reps)]
  record = {
    "cell": f"{structure}__{mode}",
    "key": key,
    "structure": structure,
    "mode": mode,
    "length_bucketing": MODES[mode],
    "warmup_s": warmup,
    "timed_s": timed,
    "median_s": statistics.median(timed),
    "backend": jax.default_backend(),
    "jax_version": jax.__version__,
    "complete": True,
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_suffix(".tmp")
  tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
  tmp.replace(out)


def _load_complete(path: Path, key: str) -> dict | None:
  if not path.is_file():
    return None
  try:
    record = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return None
  if record.get("complete") is True and record.get("key") == key:
    return record
  return None


def main() -> int:
  parser = argparse.ArgumentParser(description="S8-06 length-bucketing timing.")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--cell-timeout", type=int, default=1800)
  parser.add_argument("--_cell", nargs=2, metavar=("STRUCTURE", "MODE"), help=argparse.SUPPRESS)
  parser.add_argument("--_key", help=argparse.SUPPRESS)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  cells_dir = args.out / "cells"
  if args._cell:
    structure, mode = args._cell
    run_cell(structure, mode, args.reps, cells_dir / f"{structure}__{mode}.json", args._key)
    return 0

  source_hash = _source_hash()
  records: dict[str, dict] = {}
  reused: list[str] = []
  failed: dict[str, str] = {}
  for structure in STRUCTURES:
    for mode in MODES:
      name = f"{structure}__{mode}"
      key = _cell_key(structure, mode, args.reps, source_hash)
      path = cells_dir / f"{name}.json"
      prior = _load_complete(path, key)
      if prior is not None:
        logger.info("reuse %s (key %s)", name, key[:12])
        records[name] = prior
        reused.append(name)
        continue
      logger.info("run %s", name)
      cmd = [sys.executable, __file__, "--out", str(args.out), "--reps", str(args.reps),
             "--_cell", structure, mode, "--_key", key]
      try:
        proc = subprocess.run(cmd, timeout=args.cell_timeout, check=False)
      except subprocess.TimeoutExpired:
        failed[name] = f"timeout after {args.cell_timeout}s"
        continue
      record = _load_complete(path, key)
      if proc.returncode != 0 or record is None:
        failed[name] = f"exit {proc.returncode}, record {'present' if record else 'missing'}"
        continue
      records[name] = record

  all_cells_ok = not failed and len(records) == len(STRUCTURES) * len(MODES)
  speedup = control_ratio = None
  if all_cells_ok:
    speedup = records["1ubq__optout"]["median_s"] / records["1ubq__bucketed"]["median_s"]
    control_ratio = records["3pgk__optout"]["median_s"] / records["3pgk__bucketed"]["median_s"]
  control_ok = control_ratio is not None and CONTROL_BAND[0] <= control_ratio <= CONTROL_BAND[1]
  result = {
    "all_cells_ok": all_cells_ok,
    "speedup_1ubq": speedup,
    "control_ratio_3pgk": control_ratio,
    "control_ok": control_ok,
    "cells": records,
    "reused_cells": reused,
    "failed_cells": failed,
    "source_sha256": source_hash,
    "params": PARAMS,
    "reps": args.reps,
  }
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
  logger.info("speedup_1ubq=%s control_ratio_3pgk=%s failed=%s", speedup, control_ratio, failed)
  return 0 if all_cells_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
