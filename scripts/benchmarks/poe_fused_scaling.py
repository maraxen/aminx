"""Does ``sample_states_fused`` survive production sample counts on a production-shaped bead?

Debt #2158: a fused multi-state PoE bead crashed at sample_count>=128 (an XLA autotuning failure
after an 809 s compile; 512 failed differently). PR #127 changed the state axis to resolve through
BatchPlanner and the samples-axis memory estimate to ``lowered_memory_estimate``, but that commit
only verified on CPU unit tests, and its own regression test states it is "far too small to
reproduce the crash itself". Nothing has shown the crash is gone at production scale.

Shape: S=4 states, L=512, K=48 (``proteinmpnn_v_48_020``). The debt's failing tensor,
``f32[128,12582912]``, is 4 * 512 * 48 * 128, so this is the crash's actual geometry. Backbones
are synthetic (random-walk CA trace) at real size: compile and memory behaviour depend on shapes,
not on coordinates, the same convention as ``bench_xtrax_vs_aminx_dispatch_gpu.py``.

Cells (each is its OWN process, so a crash or timeout loses at most one cell):

  planned_n128 / planned_n512 / planned_n2048
      The shipped path, ``sample_states_fused``. Must complete with valid output.
  control_forced_vmap_n128
      NEGATIVE CONTROL. The pre-#127 configuration: state axis AND sample axis as unbounded
      ``vmap`` with no planning. If this does not fail on the hardware used, the experiment
      cannot discriminate "fixed" from "never broken here", and the verdict says so.

Persistence / resume (preemption-safe by design): every cell writes ``<cell>.started`` when it
begins and ``<cell>.done`` (sha256 of its result and of its inputs) when it finishes. A rerun skips
a cell only when its ``.done`` matches the CURRENT inputs (args, code commit, weights sha256) and
the result file's hash; otherwise it recomputes. ``--aggregate`` reads the files and reports which
cells ran, which were absent, and the verdict fields -- judged from records, never exit codes.

L1 ``--dry-run``: imports/paths only.  L2 ``--smoke``: tiny random model, CPU, well under 60 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

N_STATES = 4
SEQ_LEN = 512
CHECKPOINT = "proteinmpnn_v_48_020"
SEED = 2158

#: cell name -> (mode, n_samples). ``planned`` = shipped path; ``forced_vmap`` = negative control.
CELLS: dict[str, tuple[str, int]] = {
  "planned_n128": ("planned", 128),
  "planned_n512": ("planned", 512),
  "planned_n2048": ("planned", 2048),
  "control_forced_vmap_n128": ("forced_vmap", 128),
}
PLANNED_CELLS = tuple(name for name, (mode, _) in CELLS.items() if mode == "planned")
CONTROL_CELL = "control_forced_vmap_n128"


def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
  return _sha256_bytes(path.read_bytes())


def _synthetic_bead(num_states: int, length: int, seed: int):
  """Random-walk CA trace at real size; N/C/O placed as fixed offsets from CA."""
  import jax.numpy as jnp
  import numpy as np

  rng = np.random.default_rng(seed)
  steps = rng.normal(size=(length, 3))
  steps = 3.8 * steps / np.linalg.norm(steps, axis=-1, keepdims=True)
  ca = np.cumsum(steps, axis=0)
  offsets = np.array([[-1.2, 0.5, 0.0], [0.0, 0.0, 0.0], [1.3, 0.4, 0.0], [1.6, 1.5, 0.0]])
  one_state = (ca[:, None, :] + offsets[None, :, :]).astype(np.float32)  # (L, 4, 3)
  coords = jnp.asarray(np.stack([one_state] * num_states, axis=0))
  mask = jnp.ones((num_states, length), dtype=jnp.float32)
  residue_index = jnp.tile(jnp.arange(length, dtype=jnp.int32)[None], (num_states, 1))
  chain_index = jnp.zeros((num_states, length), dtype=jnp.int32)
  return coords, mask, residue_index, chain_index


def _build(smoke: bool):
  """Return (model, bundle, config, stage_set) for the bead; ``smoke`` = tiny random model."""
  import equinox as eqx
  import jax
  import jax.numpy as jnp

  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  if smoke:
    from aminx.model import Aminx

    num_states, length = 2, 24
    model = Aminx(
      node_features=32, edge_features=32, hidden_features=32, num_encoder_layers=1,
      num_decoder_layers=1, k_neighbors=6, dropout_rate=0.0, key=jax.random.PRNGKey(0),
    )
  else:
    from aminx.io.weights import load_model

    num_states, length = N_STATES, SEQ_LEN
    model = load_model(CHECKPOINT)
  model = eqx.tree_inference(model, value=True)
  coords, mask, residue_index, chain_index = _synthetic_bead(num_states, length, SEED)
  bundle, config = build_inference_bundle(
    coords=coords, mask=mask, residue_index=residue_index, chain_index=chain_index,
    state_weights=jnp.ones(num_states) / num_states, sequence=None, mode="sample_ar",
  )
  stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
  return model, bundle, config, stage_set


def _run_planned(model, bundle, config, stage_set, key, n_samples, recorder):
  """The shipped path, with the planner's decisions recorded (trace-time side effects)."""
  from aminx.sampling import multistate_poe as poe

  real_plan, real_est = poe._plan_axis_strategy, poe.lowered_memory_estimate

  def plan(axis, n, batch_size, *, activation_bytes_per_element):
    strategy = real_plan(axis, n, batch_size, activation_bytes_per_element=activation_bytes_per_element)
    recorder["samples_axis"] = {
      "n": int(n), "activation_bytes_per_element": int(activation_bytes_per_element),
      "strategy": repr(strategy),
    }
    return strategy

  def est(fn, *args, **kwargs):
    value = real_est(fn, *args, **kwargs)
    recorder["lowered_memory_estimate_bytes"] = int(value)
    return value

  poe._plan_axis_strategy, poe.lowered_memory_estimate = plan, est
  try:
    return poe.sample_states_fused(model, bundle, config, stage_set, key, n_samples)
  finally:
    poe._plan_axis_strategy, poe.lowered_memory_estimate = real_plan, real_est


def _run_forced_vmap(model, bundle, config, stage_set, key, n_samples):
  """NEGATIVE CONTROL: the pre-#127 configuration, no planning anywhere.

  The kernel's default state strategy is an unconditional ``Vmap`` and the sample axis is a bare
  ``vmap`` over all ``n_samples`` at once -- exactly what ``sample_states_fused`` did before it
  resolved either axis through BatchPlanner.
  """
  import equinox as eqx
  import jax

  from aminx.inference.bundle_builder import decoding_order_key, with_decoding_order
  from aminx.inference.sample_autoregressive import kernel

  @eqx.filter_jit
  def forced(model, bundle, config, stage_set, keys):
    def one(k):
      sample_bundle = with_decoding_order(bundle, decoding_order_key(k))
      result = kernel(model, k, sample_bundle, config, stage_set)  # default state Vmap
      return result.sequence, result.logits

    return jax.vmap(one)(keys)

  return forced(model, bundle, config, stage_set, jax.random.split(key, n_samples))


def _validate_output(sequences, logits, n_samples: int, length: int) -> dict[str, Any]:
  import numpy as np

  seq = np.asarray(sequences)
  lg = np.asarray(logits)
  return {
    "sequences_shape": list(seq.shape),
    "logits_shape": list(lg.shape),
    "shape_ok": seq.shape == (n_samples, length) and lg.shape == (n_samples, length, 21),
    "tokens_in_vocab": bool(((seq >= 0) & (seq <= 20)).all()),
    "logits_finite": bool(np.isfinite(lg).all()),
    "n_unique_sequences": int(len({tuple(row) for row in seq.tolist()})),
  }


def _inputs_hash(cell: str, code_commit: str, weights_sha: str, smoke: bool) -> str:
  payload = {
    "cell": cell, "mode_n": CELLS[cell], "states": N_STATES, "length": SEQ_LEN,
    "checkpoint": CHECKPOINT, "seed": SEED, "code_commit": code_commit,
    "weights_sha256": weights_sha, "smoke": smoke,
  }
  return _sha256_bytes(json.dumps(payload, sort_keys=True).encode())


def _weights_sha() -> str:
  from aminx.io.weights import weight_provenance

  return weight_provenance(CHECKPOINT).sha256


def run_cell(cell: str, out_dir: Path, *, code_commit: str, smoke: bool) -> int:
  mode, n_samples = CELLS[cell]
  if smoke:
    n_samples = 2
  out_dir.mkdir(parents=True, exist_ok=True)
  result_path = out_dir / f"{cell}.json"
  started, done = out_dir / f"{cell}.started", out_dir / f"{cell}.done"

  weights_sha = "smoke-random-weights" if smoke else _weights_sha()
  inputs_hash = _inputs_hash(cell, code_commit, weights_sha, smoke)

  if done.exists() and result_path.exists():
    stamp = json.loads(done.read_text())
    if stamp.get("inputs_hash") == inputs_hash and stamp.get("result_sha256") == _sha256_file(result_path):
      print(json.dumps({"reused": cell, "from": str(result_path), **stamp}))
      return 0
    print(f"[poe_scaling] stale or mismatched stamp for {cell}; recomputing")
  for stale in (done, result_path):
    stale.unlink(missing_ok=True)
  started.write_text(json.dumps({"cell": cell, "pid": os.getpid(), "t": time.time()}))

  import jax

  import aminx

  device = jax.devices()[0]
  record: dict[str, Any] = {
    "cell": cell, "mode": mode, "n_samples": n_samples, "smoke": smoke,
    "states": 2 if smoke else N_STATES, "seq_len": 24 if smoke else SEQ_LEN,
    "code_commit": code_commit, "aminx_file": aminx.__file__, "weights_sha256": weights_sha,
    "device_platform": device.platform, "device_kind": getattr(device, "device_kind", "?"),
    "jax_version": jax.__version__, "planner": {},
  }
  try:
    import xtrax

    record["xtrax_version"] = getattr(xtrax, "__version__", "?")
  except ImportError:
    record["xtrax_version"] = "missing"
  # The planner budgets against memory_stats()["bytes_limit"], i.e. what this process was GRANTED,
  # so on a shared GPU the cap in force (XLA_PYTHON_CLIENT_MEM_FRACTION) is part of the experiment.
  try:
    record["device_bytes_limit"] = (device.memory_stats() or {}).get("bytes_limit")
  except Exception:  # noqa: BLE001 - not all backends expose it
    record["device_bytes_limit"] = None
  record["xla_mem_fraction_env"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
  record["cuda_visible_devices_env"] = os.environ.get("CUDA_VISIBLE_DEVICES")

  t0 = time.perf_counter()
  try:
    model, bundle, config, stage_set = _build(smoke)
    key = jax.random.PRNGKey(SEED)
    if mode == "planned":
      sequences, logits = _run_planned(model, bundle, config, stage_set, key, n_samples, record["planner"])
    else:
      sequences, logits = _run_forced_vmap(model, bundle, config, stage_set, key, n_samples)
    jax.block_until_ready((sequences, logits))
    record["first_call_seconds"] = round(time.perf_counter() - t0, 3)
    record.update(_validate_output(sequences, logits, n_samples, record["seq_len"]))
    record["status"] = "ok"
  except BaseException as exc:  # noqa: BLE001 - a crash IS the measurement; record it
    record["first_call_seconds"] = round(time.perf_counter() - t0, 3)
    record["status"] = "error"
    record["error_type"] = type(exc).__name__
    record["error_head"] = str(exc)[:600]
    record["traceback_tail"] = traceback.format_exc()[-1200:]
  try:
    stats = device.memory_stats() or {}
    record["peak_bytes_in_use"] = stats.get("peak_bytes_in_use")
  except Exception:  # noqa: BLE001 - not all backends expose it
    record["peak_bytes_in_use"] = None

  result_path.write_text(json.dumps(record, indent=2, sort_keys=True))
  done.write_text(json.dumps({
    "cell": cell, "inputs_hash": inputs_hash, "result_sha256": _sha256_file(result_path),
    "status": record["status"],
  }))
  print(json.dumps({"cell": cell, "status": record["status"], "first_call_seconds": record["first_call_seconds"]}))
  return 0  # the record, not the exit code, carries the outcome


def _cell_state(out_dir: Path, cell: str) -> dict[str, Any]:
  """What the FILES say about a cell: absent / crashed (started, no record) / ok / error."""
  result_path = out_dir / f"{cell}.json"
  started = (out_dir / f"{cell}.started").exists()
  if not result_path.exists():
    return {"cell": cell, "state": "crashed_or_killed" if started else "absent"}
  record = json.loads(result_path.read_text())
  valid = (
    record.get("status") == "ok" and record.get("shape_ok") is True
    and record.get("tokens_in_vocab") is True and record.get("logits_finite") is True
    and (record.get("n_samples", 0) < 2 or record.get("n_unique_sequences", 0) > 1)
  )
  return {
    "cell": cell, "state": "ok" if valid else ("error" if record.get("status") == "error" else "invalid_output"),
    "first_call_seconds": record.get("first_call_seconds"), "device_kind": record.get("device_kind"),
    "error_type": record.get("error_type"), "error_head": record.get("error_head"),
    "peak_bytes_in_use": record.get("peak_bytes_in_use"), "planner": record.get("planner"),
    "code_commit": record.get("code_commit"), "device_bytes_limit": record.get("device_bytes_limit"),
  }


def aggregate(out_dir: Path) -> dict[str, Any]:
  states = {cell: _cell_state(out_dir, cell) for cell in CELLS}
  planned_ok = [c for c in PLANNED_CELLS if states[c]["state"] == "ok"]
  control_state = states[CONTROL_CELL]["state"]
  # The control "fires" only on evidence of a failure: a recorded error, or a process that
  # started and left no record. A cell that never started (queued, cancelled) says nothing.
  control_failed = control_state in ("error", "crashed_or_killed", "invalid_output")
  commits = {s.get("code_commit") for s in states.values() if s.get("code_commit")}
  return {
    "planned_cells_ok": len(planned_ok) == len(PLANNED_CELLS),
    "n_planned_ok": len(planned_ok),
    "negative_control_failed": control_failed,
    "negative_control_state": control_state,
    # Reported, not judged: OOM vs an XLA autotuning error are different failures, and only the
    # latter is the original crash. The verdict text says what each does and does not show.
    "negative_control_error_type": states[CONTROL_CELL].get("error_type"),
    "single_code_commit": len(commits) <= 1,
    "cells": states,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--cell", choices=sorted(CELLS), help="run one cell (one process per cell)")
  parser.add_argument("--out-dir", type=Path, default=Path("outputs/results/poe_fused_scaling"))
  parser.add_argument("--code-commit", default=os.environ.get("POE_CODE_COMMIT", "unknown"))
  parser.add_argument("--dry-run", action="store_true", help="L1: imports and paths only")
  parser.add_argument("--smoke", action="store_true", help="L2: tiny random model, n=2, CPU")
  parser.add_argument("--aggregate", type=Path, metavar="DIR", help="summarise a results dir; prints JSON")
  args = parser.parse_args()

  if args.aggregate is not None:
    print(json.dumps(aggregate(args.aggregate), indent=2, sort_keys=True))
    return 0
  if args.dry_run:
    import jax

    import aminx
    from aminx.sampling import multistate_poe  # noqa: F401

    print(json.dumps({"dry_run": "ok", "aminx": aminx.__file__, "devices": [str(d) for d in jax.devices()],
                      "cells": list(CELLS)}))
    return 0
  if args.cell is None:
    parser.error("--cell is required unless --aggregate / --dry-run")
  return run_cell(args.cell, args.out_dir, code_commit=args.code_commit, smoke=args.smoke)


if __name__ == "__main__":
  sys.exit(main())
