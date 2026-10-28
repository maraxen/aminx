"""LASErMPNN batched decode against the per-sample loop on GPU (task 261028_aminx-laser-batched-decode-6273).

Backlog #6273 replaced the per-(sample, temperature) ``decode_order`` loop in
``families/laser_mpnn/sample_host.py`` with one encode plus a ``jax.vmap``'d scan per chunk
window (``decode_order_batched``). This benchmark measures the change on production sampling of
``tests/fixtures/laser/103m_1.pdb`` (myoglobin, 154 residues) with the Hub-pinned
``lasermpnn_nothing_heldout`` checkpoint, seed 0, one temperature.

Units, each in its own subprocess with its own timeout:

- ``loop``: the per-sample reference path (``SampleStages._decode``, which encodes and scans once
  per sample, exactly as before #6273) over ``--n-loop`` samples with the same fold_in keys
  ``runner.sample`` uses. Warm pass, then a timed pass. Saves its sequences.
- ``chunk_<c>`` for each ``--chunks`` value: ``runner.sample`` with ``num_samples = --n`` and
  ``samples_chunk_size = c`` (now the batch size of one device call). Warm call, then a timed
  call with XLA compiles counted through ``jax_log_compiles``. Saves its sequences.

Each unit writes ``unit_<name>.json`` atomically with ``complete: true`` and a ``config_sha256``
over its inputs (unit, n, seed, temperature, structure sha256, checkpoint id, weights revision,
git HEAD, device kind, jax and jaxlib versions). ``--resume`` reuses a unit only when that digest
matches, and the result lists every reused unit.

The parent reports:

- ``speedup_best``: looped per-sample time over the best chunk's per-sample time;
- ``seq_match_frac``: the fraction of the first ``--n-loop`` samples whose batched sequence (from
  every chunk size) equals the looped sequence exactly;
- ``knob_live``: peak device memory grows from the smallest to the largest chunk, so the chunk
  size reached the device call;
- ``same_device``: every unit ran on one device kind.
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

log = logging.getLogger("bench_laser_batched_decode")

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


def _environment() -> dict[str, str]:
  import jax
  import jaxlib

  return {
    "device_kind": jax.devices()[0].device_kind,
    "jax_version": jax.__version__,
    "jaxlib_version": jaxlib.__version__,
  }


def _config(unit: str, args: argparse.Namespace, environment: dict[str, str]) -> dict[str, Any]:
  from aminx.io.weights import HF_REVISION

  return {
    "unit": unit,
    "n": args.n_loop if unit == "loop" else args.n,
    "seed": args.seed,
    "temperature": args.temperature,
    "structure_sha256": _sha256(STRUCTURE),
    "checkpoint_id": CHECKPOINT_ID,
    "weights_revision": os.environ.get("AMINX_WEIGHTS_REVISION") or HF_REVISION,
    "git_head": _git_head(),
    **environment,
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


def _spec(num_samples: int, chunk: int, args: argparse.Namespace) -> Any:  # noqa: ANN401
  from aminx.run.options import LaserOptions
  from aminx.run.specs import SamplingSpecification

  return SamplingSpecification(
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


def _peak_bytes() -> int | None:
  import jax

  stats = jax.devices()[0].memory_stats()
  return None if stats is None else stats.get("peak_bytes_in_use")


def _timed(fn: Any) -> tuple[Any, float, int]:  # noqa: ANN401
  """Run ``fn`` with compiles counted. Returns ``(result, seconds, compiles)``."""
  import jax

  jax.config.update("jax_log_compiles", True)
  counter = _CompileCounter()
  logging.getLogger("jax").addHandler(counter)
  t0 = time.perf_counter()
  result = fn()
  seconds = time.perf_counter() - t0
  logging.getLogger("jax").removeHandler(counter)
  jax.config.update("jax_log_compiles", False)
  return result, seconds, counter.count


def _loop_sequences(args: argparse.Namespace) -> Any:  # noqa: ANN401
  """The per-sample reference path over ``--n-loop`` samples, keys as ``runner.sample`` makes them."""
  import jax
  import numpy as np

  from aminx.families.laser_mpnn.driver import LaserDriver
  from aminx.families.laser_mpnn.sample_host import SampleStages, sample_batches

  spec = _spec(args.n_loop, args.n_loop, args)
  model = LaserDriver().load(spec)
  stages = SampleStages(model, spec)
  item = next(sample_batches(spec)).arrays["sampled"][0]
  dropout_key = jax.random.PRNGKey(int(spec.random_seed))
  rows = [
    stages._decode(item, args.temperature, jax.random.fold_in(dropout_key, index))["sequence"]  # noqa: SLF001
    for index in range(args.n_loop)
  ]
  return np.stack(rows)


def run_unit(unit: str, args: argparse.Namespace) -> None:
  """Measure one unit and write its stamped unit file and sequences."""
  import numpy as np

  config = _config(unit, args, _environment())
  if unit == "loop":
    n = args.n_loop

    def call() -> Any:  # noqa: ANN401
      return _loop_sequences(args)

  else:
    from aminx.host.runner import sample as runner_sample

    n = args.n
    chunk = int(unit.removeprefix("chunk_"))

    def call() -> Any:  # noqa: ANN401
      out = runner_sample(_spec(args.n, chunk, args))
      return np.asarray(out["structures"]["0"]["arrays"]["sequence"])

  _, first_s, first_compiles = _timed(call)
  sequences, steady_s, steady_compiles = _timed(call)
  sequences = np.asarray(sequences)
  valid = sequences.shape[0] == n and sequences.shape[-1] == EXPECTED_LENGTH
  seq_path = args.work / f"seq_{unit}.npy"
  np.save(seq_path, sequences)
  row = {
    "unit": unit,
    "n": n,
    "first_s": first_s,
    "steady_s": steady_s,
    "per_sample_s": steady_s / n,
    "first_compiles": first_compiles,
    "steady_compiles": steady_compiles,
    "peak_bytes": _peak_bytes(),
    "sequence_shape": list(sequences.shape),
    "sequences_sha256": _sha256(seq_path),
    "valid": bool(valid),
    "config": config,
    "config_sha256": _config_sha(config),
    "complete": True,
  }
  log.info("unit %s", json.dumps(row))
  path = args.work / f"unit_{unit}.json"
  tmp = path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(row, indent=2))
  tmp.replace(path)


def _reusable(unit: str, args: argparse.Namespace, environment: dict[str, str]) -> dict[str, Any] | None:
  path = args.work / f"unit_{unit}.json"
  seq_path = args.work / f"seq_{unit}.npy"
  if not (args.resume and path.is_file() and seq_path.is_file()):
    return None
  try:
    row = json.loads(path.read_text())
  except json.JSONDecodeError:
    log.warning("unit %s file is not valid JSON; re-running it", unit)
    return None
  expected = _config_sha(_config(unit, args, environment))
  if row.get("complete") and row.get("config_sha256") == expected and row.get("sequences_sha256") == _sha256(seq_path):
    return row
  log.info("unit %s on disk does not match this config or its sequences; re-running it", unit)
  return None


def _arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--chunks", default="8,32,128")
  parser.add_argument("--n", type=int, default=256)
  parser.add_argument("--n-loop", type=int, default=64)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--temperature", type=float, default=0.1)
  parser.add_argument("--unit-timeout", type=int, default=1800)
  parser.add_argument("--work", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--unit", default=None, help=argparse.SUPPRESS)
  return parser.parse_args()


def main() -> int:
  """Run or reuse every unit, then write the result JSON. Exits 1 when a unit failed."""
  args = _arguments()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  args.work.mkdir(parents=True, exist_ok=True)

  if args.unit is not None:
    run_unit(args.unit, args)
    return 0

  import jax
  import numpy as np

  environment = _environment()
  chunks = [int(c) for c in args.chunks.split(",")]
  units = ["loop", *(f"chunk_{c}" for c in chunks)]
  rows: dict[str, dict[str, Any]] = {}
  reused: list[str] = []
  failed: dict[str, str] = {}
  for unit in units:
    row = _reusable(unit, args, environment)
    if row is not None:
      rows[unit] = row
      reused.append(unit)
      continue
    cmd = [sys.executable, __file__, "--unit", unit, "--n", str(args.n), "--n-loop", str(args.n_loop)]
    cmd += ["--seed", str(args.seed), "--temperature", str(args.temperature), "--work", str(args.work)]
    try:
      proc = subprocess.run(cmd, timeout=args.unit_timeout, check=False)
    except subprocess.TimeoutExpired:
      failed[unit] = f"timeout after {args.unit_timeout}s"
      continue
    path = args.work / f"unit_{unit}.json"
    if proc.returncode != 0 or not path.is_file():
      failed[unit] = f"exit {proc.returncode}"
      continue
    rows[unit] = json.loads(path.read_text())

  batched = [rows[f"chunk_{c}"] for c in chunks if f"chunk_{c}" in rows]
  loop = rows.get("loop")
  speedup_best = None
  best_chunk = None
  seq_match_frac = None
  if loop is not None and batched:
    best = min(batched, key=lambda r: r["per_sample_s"])
    best_chunk = int(best["unit"].removeprefix("chunk_"))
    speedup_best = loop["per_sample_s"] / best["per_sample_s"]
    reference = np.load(args.work / "seq_loop.npy")
    matches = []
    for row in batched:
      got = np.load(args.work / f"seq_{row['unit']}.npy")[: args.n_loop]
      matches.extend(bool(np.array_equal(got[i], reference[i])) for i in range(reference.shape[0]))
    seq_match_frac = sum(matches) / len(matches)
  small, large = rows.get(f"chunk_{min(chunks)}"), rows.get(f"chunk_{max(chunks)}")
  knob_live = bool(
    small and large and small["peak_bytes"] is not None and large["peak_bytes"] is not None
    and large["peak_bytes"] > small["peak_bytes"]
  )
  measured = [rows[u] for u in units if u in rows]
  result = {
    "backend": jax.default_backend(),
    "device_kind": environment["device_kind"],
    "jax_version": environment["jax_version"],
    "checkpoint_id": CHECKPOINT_ID,
    "length": EXPECTED_LENGTH,
    "n": args.n,
    "n_loop": args.n_loop,
    "chunks": chunks,
    "all_valid": not failed and len(measured) == len(units) and all(r["valid"] for r in measured),
    "compile_counter_live": bool(measured) and all(r["first_compiles"] > 0 for r in measured),
    "max_steady_compiles": max((r["steady_compiles"] for r in batched), default=None),
    "same_device": all(r["config"]["device_kind"] == environment["device_kind"] for r in measured),
    "knob_live": knob_live,
    "seq_match_frac": seq_match_frac,
    "loop_per_sample_s": loop["per_sample_s"] if loop else None,
    "speedup_best": speedup_best,
    "best_chunk": best_chunk,
    "failed_units": failed,
    "reused_units": reused,
    "units": measured,
  }
  text = json.dumps(result, indent=2)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(text)
  log.info("result %s", json.dumps({k: v for k, v in result.items() if k != "units"}))
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
