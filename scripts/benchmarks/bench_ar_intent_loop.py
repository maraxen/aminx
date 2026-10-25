"""AR decode on xtrax.intent.loop vs hand-picked while_loop and lax.scan (xtrax criterion 8).

Pre-registered in ``bench_ar_intent_loop.bth.toml``. One production-shaped model (128-dim,
3+3 layers, k = 48, random weights; decode cost depends on shapes only) decodes one L-residue
structure under ``incremental="auto"`` (the production default). Three wave-loop forms:

- ``scan``: ``use_while_loop=False`` (today's default, differentiable),
- ``while``: ``use_while_loop=True`` (today's hand-set inference-only flag),
- ``intent``: ``intent_loop=True`` (xtrax.intent.loop: while undifferentiated, scan under AD).

Cells (form x kind) each run in their own process with their own timeout, so compile caches
and the device allocator's peak are per cell. ``primal`` cells time compile and the median of
``--repeats`` post-warm-up decodes and count steady-state recompiles; ``grad`` cells (scan and
intent only; while_loop has no reverse mode) do the same for ``grad(sum(logits))`` with respect
to the model's inexact parameters at ``--grad-length`` (default 128: at L=512 the full-recompute
branch keeps ~19 GB of residuals) and record the device's peak bytes in use. Each finished
cell is written atomically with a stamp (config sha256 + git hash) and reused on resume only
when both match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("bench_ar_intent_loop")

FORMS = {"scan": (False, False), "while": (True, False), "intent": (False, True)}
CELLS = [
  ("scan", "primal"),
  ("while", "primal"),
  ("intent", "primal"),
  ("scan", "grad"),
  ("intent", "grad"),
]


def _git_hash() -> str:
  return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _config_sha(args: argparse.Namespace) -> str:
  blob = json.dumps(
    {"length": args.length, "grad_length": args.grad_length, "repeats": args.repeats, "seed": args.seed},
    sort_keys=True,
  )
  return hashlib.sha256(blob.encode()).hexdigest()


class _CompileCounter(logging.Handler):
  """Counts JAX "Compiling" log records (``jax_log_compiles``)."""

  def __init__(self) -> None:
    super().__init__(level=logging.WARNING)
    self.count = 0

  def emit(self, record: logging.LogRecord) -> None:
    if "Compiling" in record.getMessage():
      self.count += 1


def _run_cell(form: str, kind: str, args: argparse.Namespace, cell_dir: Path) -> None:
  import equinox as eqx
  import jax
  import jax.numpy as jnp
  import numpy as np

  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.decode.factory import make_decode_fn
  from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
  from aminx.inference.encode import make_encode_fn
  from aminx.inference.logits import make_stage_set
  from aminx.model.mpnn import Aminx
  from aminx.tiling.strategy import Vmap

  jax.config.update("jax_log_compiles", True)
  counter = _CompileCounter()
  logging.getLogger("jax").addHandler(counter)
  # jax_log_compiles emits "Compiling ..." at WARNING; DEBUG would log during the timed compile.
  logging.getLogger("jax").setLevel(logging.WARNING)

  length = args.length if kind == "primal" else args.grad_length
  key = jax.random.PRNGKey(args.seed)
  ca = jnp.cumsum(jax.random.normal(key, (length, 1, 3)) * 2.2, axis=0)
  coords = ca + jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 3)) * 0.8
  bundle, config = build_inference_bundle(
    coords,
    jnp.ones((length,)),
    jnp.arange(length),
    jnp.zeros((length,), dtype=jnp.int32),
    mode="sample_autoregressive",
    temperature=1.0,
  )
  model = eqx.tree_inference(
    Aminx(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      num_encoder_layers=3,
      num_decoder_layers=3,
      k_neighbors=48,
      key=jax.random.PRNGKey(0),
    ),
    value=True,
  )
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)
  stage_set = make_stage_set()
  inference_only, intent_loop = FORMS[form]
  ar_config = AutoregressiveConfig(inference_only=inference_only, intent_loop=intent_loop)

  def decoder(m: Any) -> Any:
    return make_decode_fn(model=m, mode=AutoregressiveMode(), strategy=Vmap(), autoregressive_config=ar_config)

  record: dict[str, Any] = {"form": form, "kind": kind, "length": length, "backend": jax.default_backend()}
  if kind == "primal":
    decode = eqx.filter_jit(decoder(model))

    def call(r: int) -> Any:
      return decode(jax.random.PRNGKey(7 + r), enc, bundle, config, stage_set)

  else:
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss(p: Any) -> Any:
      return jnp.sum(decoder(eqx.combine(p, static))(jax.random.PRNGKey(7), enc, bundle, config, stage_set).logits)

    grad_fn = eqx.filter_jit(jax.grad(loss))

    def call(r: int) -> Any:
      del r
      return grad_fn(params)

  t0 = time.perf_counter()
  first = jax.block_until_ready(call(0))
  record["compile_s"] = time.perf_counter() - t0
  compiles_before = counter.count
  times = []
  for r in range(args.repeats):
    t0 = time.perf_counter()
    jax.block_until_ready(call(r))
    times.append(time.perf_counter() - t0)
  record["step_s"] = float(np.median(times))
  record["step_cv"] = float(np.std(times) / np.mean(times))
  record["steady_recompiles"] = counter.count - compiles_before
  stats = jax.devices()[0].memory_stats() or {}
  record["peak_bytes"] = stats.get("peak_bytes_in_use")

  arrays_path = cell_dir / f"{form}_{kind}.npz"
  if kind == "primal":
    np.savez(arrays_path, sequence=np.asarray(first.sequence), logits=np.asarray(first.logits))
  else:
    leaves = jax.tree_util.tree_leaves(first)
    np.savez(arrays_path, *[np.asarray(g) for g in leaves])
  record["arrays"] = arrays_path.name
  out = cell_dir / f"{form}_{kind}.json"
  tmp = out.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(record, indent=2))
  os.replace(tmp, out)


def _stamped(cell_dir: Path, form: str, kind: str, stamp: dict[str, str]) -> dict[str, Any] | None:
  path = cell_dir / f"{form}_{kind}.json"
  if not path.exists():
    return None
  rec = json.loads(path.read_text())
  if rec.get("stamp") != stamp:
    return None
  return rec


def _aggregate(cell_dir: Path, recs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
  import numpy as np

  def arrays(form: str, kind: str) -> Any:
    return np.load(cell_dir / recs[(form, kind)]["arrays"])

  seqs = {f: arrays(f, "primal")["sequence"] for f in ("scan", "while", "intent")}
  logits = {f: arrays(f, "primal")["logits"] for f in ("scan", "while", "intent")}
  g_scan, g_intent = arrays("scan", "grad"), arrays("intent", "grad")
  rel = []
  for name in g_scan.files:
    a, b = g_scan[name], g_intent[name]
    denom = max(float(np.linalg.norm(a)), 1e-30)
    rel.append(float(np.linalg.norm(a - b)) / denom)
  grad_nonzero = any(float(np.max(np.abs(g_scan[n]))) > 0 for n in g_scan.files)

  def r(form: str, kind: str, field: str) -> Any:
    return recs[(form, kind)][field]

  scan_peak = r("scan", "grad", "peak_bytes")
  intent_peak = r("intent", "grad", "peak_bytes")
  return {
    "tokens_equal_all": bool(
      np.array_equal(seqs["scan"], seqs["intent"]) and np.array_equal(seqs["while"], seqs["intent"])
    ),
    "logits_max_abs_diff": float(
      max(np.max(np.abs(logits["scan"] - logits["intent"])), np.max(np.abs(logits["while"] - logits["intent"])))
    ),
    "grad_rel_l2_max": float(max(rel)),
    "grad_nonzero": grad_nonzero,
    "scan_primal_compile_s": r("scan", "primal", "compile_s"),
    "while_primal_compile_s": r("while", "primal", "compile_s"),
    "intent_primal_compile_s": r("intent", "primal", "compile_s"),
    "scan_primal_step_s": r("scan", "primal", "step_s"),
    "while_primal_step_s": r("while", "primal", "step_s"),
    "intent_primal_step_s": r("intent", "primal", "step_s"),
    "scan_grad_compile_s": r("scan", "grad", "compile_s"),
    "intent_grad_compile_s": r("intent", "grad", "compile_s"),
    "scan_grad_step_s": r("scan", "grad", "step_s"),
    "intent_grad_step_s": r("intent", "grad", "step_s"),
    "scan_grad_peak_bytes": scan_peak,
    "intent_grad_peak_bytes": intent_peak,
    "compile_contrast": r("scan", "primal", "compile_s") / r("while", "primal", "compile_s"),
    "intent_vs_while_compile": r("intent", "primal", "compile_s") / r("while", "primal", "compile_s"),
    "intent_vs_while_step": r("intent", "primal", "step_s") / r("while", "primal", "step_s"),
    "intent_vs_scan_grad_peak": (intent_peak / scan_peak) if scan_peak and intent_peak else -1.0,
    "intent_steady_recompiles": r("intent", "primal", "steady_recompiles") + r("intent", "grad", "steady_recompiles"),
    "backend": r("intent", "primal", "backend"),
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--length", type=int, default=512)
  parser.add_argument("--grad-length", type=int, default=128)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--seed", type=int, default=512)
  parser.add_argument("--cell-timeout-s", type=int, default=7200)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--cell", nargs=2, metavar=("FORM", "KIND"), help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  cell_dir = args.out.parent / f"{args.out.stem}_cells"
  cell_dir.mkdir(parents=True, exist_ok=True)

  if args.cell:
    _run_cell(args.cell[0], args.cell[1], args, cell_dir)
    return 0

  stamp = {"config_sha256": _config_sha(args), "git_hash": _git_hash()}
  recs: dict[tuple[str, str], dict[str, Any]] = {}
  for form, kind in CELLS:
    rec = _stamped(cell_dir, form, kind, stamp)
    if rec is not None:
      logger.info("reusing %s/%s from %s", form, kind, cell_dir)
      recs[(form, kind)] = rec
      continue
    logger.info("running %s/%s (timeout %ds)", form, kind, args.cell_timeout_s)
    t0 = time.perf_counter()
    subprocess.run(
      [sys.executable, __file__, "--length", str(args.length), "--grad-length", str(args.grad_length),
       "--repeats", str(args.repeats),
       "--seed", str(args.seed), "--out", str(args.out), "--cell", form, kind],
      check=True,
      timeout=args.cell_timeout_s,
    )
    path = cell_dir / f"{form}_{kind}.json"
    rec = json.loads(path.read_text())
    rec["stamp"] = stamp
    rec["cell_wall_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec, indent=2))
    recs[(form, kind)] = rec
    logger.info("%s/%s compile %.1fs step %.4fs peak %s", form, kind, rec["compile_s"], rec["step_s"], rec["peak_bytes"])

  result = _aggregate(cell_dir, recs)
  result["length"] = args.length
  result["grad_length"] = args.grad_length
  result["git_hash"] = stamp["git_hash"]
  for k, v in result.items():
    if isinstance(v, float) and not math.isfinite(v):
      msg = f"{k} is not finite: {v}"
      raise ValueError(msg)
  args.out.write_text(json.dumps(result, indent=2))
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(result, indent=2))
  logger.info("%s", json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
