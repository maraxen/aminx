"""LASErMPNN sampling cost against ``samples_chunk_size`` at L=154 (task 261027_aminx-laser-chunk-sweep).

``samples_chunk_size`` is the number of samples per device call of the sample step
(``host/streaming.py``); the confirm script ``scripts/parity/laser_sample_dist_confirm.py``
fixes it at 8. This sweep times production sampling of ``tests/fixtures/laser/103m_1.pdb``
(myoglobin, 154 residues) with the Hub-pinned ``lasermpnn_nothing_heldout`` checkpoint at each
chunk size, so the chunk choice for long sampling runs rests on a measurement.

Each chunk size is one unit, run in its own subprocess with its own timeout:

1. a warm call with ``num_samples = --n`` (weight load + compile + one full run), timed as
   ``first_s``;
2. the same call again, timed as ``steady_s``, with XLA compiles counted
   through ``jax_log_compiles`` (``steady_compiles``; 0 means no recompile at steady state);
3. a validity check: ``--n`` sequences of length 154 came back.

A unit writes ``unit_<chunk>.json`` atomically with ``complete: true`` and a ``config_sha256``
over its inputs (chunk, n, seed, temperature, structure sha256, checkpoint id, weights revision,
git HEAD). ``--resume`` reuses a unit only when that digest matches, and the result lists every
reused unit. The parent writes one result JSON to ``--out`` and to ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("bench_laser_chunk_sweep")

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = ROOT / "tests/fixtures/laser/103m_1.pdb"
CHECKPOINT_ID = "lasermpnn_nothing_heldout"
EXPECTED_LENGTH = 154


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
  out = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
  )
  return out.stdout.strip() or "unknown"


def _config(chunk: int, args: argparse.Namespace) -> dict[str, Any]:
  from aminx.io.weights import HF_REVISION

  return {
    "chunk": chunk,
    "n": args.n,
    "seed": args.seed,
    "temperature": args.temperature,
    "structure_sha256": _sha256(STRUCTURE),
    "checkpoint_id": CHECKPOINT_ID,
    "weights_revision": os.environ.get("AMINX_WEIGHTS_REVISION") or HF_REVISION,
    "git_head": _git_head(),
  }


def _config_sha(config: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class _CompileCounter(logging.Handler):
  """Counts the ``Compiling ...`` records JAX logs when ``jax_log_compiles`` is on."""

  def __init__(self) -> None:
    super().__init__(level=logging.DEBUG)
    self.count = 0

  def emit(self, record: logging.LogRecord) -> None:
    if record.getMessage().startswith("Compiling"):
      self.count += 1


def _sample(chunk: int, num_samples: int, args: argparse.Namespace) -> dict[str, Any]:
  from aminx.host.runner import sample as runner_sample
  from aminx.run.options import LaserOptions
  from aminx.run.specs import SamplingSpecification

  spec = SamplingSpecification(
    inputs=str(STRUCTURE),
    model_family="lasermpnn",
    checkpoint_id=CHECKPOINT_ID,
    num_samples=num_samples,
    samples_chunk_size=chunk,
    return_logits=False,
    random_seed=args.seed,
    temperature=args.temperature,
    backbone_noise=0.0,
    laser=LaserOptions(disabled_residues=("X",)),
  )
  return runner_sample(spec)


def _peak_bytes() -> int | None:
  import jax

  stats = jax.devices()[0].memory_stats()
  return None if stats is None else stats.get("peak_bytes_in_use")


def run_unit(chunk: int, args: argparse.Namespace) -> None:
  """Measure one chunk size and write its stamped unit file."""
  import jax
  import numpy as np

  config = _config(chunk, args)
  # The warm call's count is the counter's control: it must see the first compile, or a
  # steady count of 0 means nothing.
  jax.config.update("jax_log_compiles", True)
  warm_counter = _CompileCounter()
  logging.getLogger("jax").addHandler(warm_counter)
  t0 = time.perf_counter()
  _sample(chunk, args.n, args)
  first_s = time.perf_counter() - t0
  logging.getLogger("jax").removeHandler(warm_counter)

  counter = _CompileCounter()
  logging.getLogger("jax").addHandler(counter)
  t0 = time.perf_counter()
  result = _sample(chunk, args.n, args)
  steady_s = time.perf_counter() - t0
  jax.config.update("jax_log_compiles", False)
  logging.getLogger("jax").removeHandler(counter)

  sequences = np.asarray(result["structures"]["0"]["arrays"]["sequence"])
  valid = sequences.shape[0] == args.n and sequences.shape[-1] == EXPECTED_LENGTH
  row = {
    "chunk": chunk,
    "first_s": first_s,
    "steady_s": steady_s,
    "per_sample_s": steady_s / args.n,
    "first_compiles": warm_counter.count,
    "steady_compiles": counter.count,
    "peak_bytes": _peak_bytes(),
    "sequence_shape": list(sequences.shape),
    "valid": bool(valid),
    "config": config,
    "config_sha256": _config_sha(config),
    "complete": True,
  }
  log.info("unit %s", json.dumps(row))
  path = args.work / f"unit_{chunk}.json"
  tmp = path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(row, indent=2))
  tmp.replace(path)


def _reusable(chunk: int, args: argparse.Namespace) -> dict[str, Any] | None:
  path = args.work / f"unit_{chunk}.json"
  if not (args.resume and path.is_file()):
    return None
  row = json.loads(path.read_text())
  if row.get("complete") and row.get("config_sha256") == _config_sha(_config(chunk, args)):
    return row
  log.info("unit %s on disk does not match this config; re-running it", chunk)
  return None


def main() -> None:
  """Run or reuse every unit, then write the result JSON."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--chunks", default="8,32,128")
  parser.add_argument("--n", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--temperature", type=float, default=0.1)
  parser.add_argument("--unit-timeout", type=int, default=1800)
  parser.add_argument("--work", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--unit", type=int, default=None, help=argparse.SUPPRESS)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  args.work.mkdir(parents=True, exist_ok=True)

  if args.unit is not None:
    run_unit(args.unit, args)
    return

  import jax

  chunks = [int(c) for c in args.chunks.split(",")]
  rows: dict[int, dict[str, Any]] = {}
  reused: list[int] = []
  failed: dict[int, str] = {}
  for chunk in chunks:
    row = _reusable(chunk, args)
    if row is not None:
      rows[chunk] = row
      reused.append(chunk)
      continue
    cmd = [sys.executable, __file__, "--unit", str(chunk), "--n", str(args.n)]
    cmd += ["--seed", str(args.seed), "--temperature", str(args.temperature)]
    cmd += ["--work", str(args.work)]
    try:
      proc = subprocess.run(cmd, timeout=args.unit_timeout, check=False)
    except subprocess.TimeoutExpired:
      failed[chunk] = f"timeout after {args.unit_timeout}s"
      continue
    path = args.work / f"unit_{chunk}.json"
    if proc.returncode != 0 or not path.is_file():
      failed[chunk] = f"exit {proc.returncode}"
      continue
    rows[chunk] = json.loads(path.read_text())

  smallest, largest = min(chunks), max(chunks)
  done = smallest in rows and largest in rows
  speedup = rows[smallest]["per_sample_s"] / rows[largest]["per_sample_s"] if done else None
  measured = [rows[c] for c in chunks if c in rows]
  result = {
    "backend": jax.default_backend(),
    "device_kind": jax.devices()[0].device_kind,
    "jax_version": jax.__version__,
    "checkpoint_id": CHECKPOINT_ID,
    "length": EXPECTED_LENGTH,
    "n": args.n,
    "chunks": chunks,
    "all_valid": not failed and all(r["valid"] for r in measured),
    "compile_counter_live": bool(measured) and all(r["first_compiles"] > 0 for r in measured),
    "max_steady_compiles": max((r["steady_compiles"] for r in measured), default=None),
    "speedup_largest_vs_smallest": speedup,
    "best_chunk": min(measured, key=lambda r: r["per_sample_s"])["chunk"] if measured else None,
    "failed_units": {str(k): v for k, v in failed.items()},
    "reused_units": reused,
    "per_chunk": measured,
  }
  text = json.dumps(result, indent=2)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(text)
  log.info("result %s", json.dumps({k: v for k, v in result.items() if k != "per_chunk"}))


if __name__ == "__main__":
  main()
