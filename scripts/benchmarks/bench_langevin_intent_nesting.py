"""Langevin trajectories under an outer map: while-under-scan hazard vs xtrax.intent (xtrax criterion 8b).

Pre-registered in ``bench_langevin_intent_nesting.bth.toml``. One production-shaped
``ProteinEBMModel`` (token_s 256, token_z 128, depth 16, heads 8, random weights; compile and
step cost depend on shapes only) runs ``--n-steps`` Langevin steps for each of ``--batch``
trajectories of one ``--length``-residue structure. The per-trajectory loop is the one in
``aminx.ebm.langevin.run_langevin_equilibration``. Forms of the outer trajectory map:

- ``safemap_while``: ``jax.lax.map(batch_size=--chunk)`` around the ``while_loop`` (what
  ``SafeMap`` lowers to; the nesting that hung SLURM job 18182309 on SM120),
- ``vmap_while``: ``jax.vmap`` around the ``while_loop`` (the hand fix, commit 0faa1d0),
- ``intent``: ``xtrax.intent.compile`` of ``intent.map`` around ``intent.loop``, with
  ``Context(chunk_size=--chunk)`` and no memory budget,
- ``intent_tight`` (plan only): the same function planned with a memory budget of half the
  vmap form's compiled estimate, so the planner has to consider chunking.

``grad`` cells (``scan_ref``: ``jax.vmap`` around ``lax.scan``, written by hand; ``intent``)
take ``grad`` of the mean squared final coordinates with respect to the model's inexact
parameters and the initial coordinates, at ``--grad-length``/``--grad-batch``/
``--grad-steps``, and record the device's peak bytes in use.

Each cell runs in its own process with its own timeout. ``safemap_while`` and ``intent_tight``
use ``--hazard-timeout-s``; on timeout they are recorded as censored (``timed_out``) instead of
failing the run. Before compiling, every primal cell writes whether its jaxpr has a ``while``
inside a ``scan`` (``<cell>.pre.json``), so a censored cell still records what it compiled.
Finished cells are written atomically with a stamp (config sha256 + git hash) and reused on
resume only when both match.
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

logger = logging.getLogger("bench_langevin_intent_nesting")

CELLS = [
  ("vmap_while", "primal"),
  ("safemap_while", "primal"),
  ("intent", "primal"),
  ("intent_tight", "plan"),
  ("scan_ref", "grad"),
  ("intent", "grad"),
]
HAZARD_CELLS = {("safemap_while", "primal"), ("intent_tight", "plan")}
DIFFUSION_TIME = 0.05
DT = 1e-3
# --model tiny: the langevin benchmark's smoke dims, for the untracked CPU smoke only.
TINY_DIMS = {
  "token_s": 32,
  "token_z": 16,
  "dim_fourier": 12,
  "conditioning_transition_layers": 1,
  "transformer_depth": 2,
  "transformer_heads": 2,
}


def _git_hash() -> str:
  return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _config_sha(args: argparse.Namespace) -> str:
  blob = json.dumps(
    {
      k: getattr(args, k)
      for k in (
        "length", "batch", "n_steps", "chunk", "grad_length", "grad_batch", "grad_steps", "repeats", "seed", "model",
      )
    },
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


def _gpu_others() -> str | None:
  """Utilization and memory of the visible GPU, from nvidia-smi (contention record), or None."""
  visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
  try:
    out = subprocess.run(
      ["nvidia-smi", "-i", visible, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
      capture_output=True,
      text=True,
      timeout=30,
      check=True,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return out.stdout.strip()


def _primitive_counts(closed: Any) -> dict[str, Any]:
  """``while`` and ``scan`` counts in a jaxpr, and whether a ``while`` sits inside a ``scan`` body."""
  from jax.extend.core import jaxprs_in_params

  counts = {"while": 0, "scan": 0}
  nested = False

  def walk(jaxpr: Any, under_scan: bool) -> None:
    nonlocal nested
    for eqn in jaxpr.eqns:
      name = eqn.primitive.name
      if name in counts:
        counts[name] += 1
      if name == "while" and under_scan:
        nested = True
      for sub in jaxprs_in_params(eqn.params):
        walk(sub, under_scan or name == "scan")

  walk(getattr(closed, "jaxpr", closed), False)
  return {"while_count": counts["while"], "scan_count": counts["scan"], "scan_encloses_while": nested}


def _write_json(path: Path, record: dict[str, Any]) -> None:
  tmp = path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(record, indent=2))
  os.replace(tmp, path)


def _run_cell(form: str, kind: str, args: argparse.Namespace, cell_dir: Path) -> None:
  # Allocate on demand: peak_bytes_in_use is the allocator's in-use peak either way.
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  load_start = os.getloadavg()[0]

  import equinox as eqx
  import jax
  import jax.numpy as jnp
  import numpy as np
  from xtrax import intent
  from xtrax.tiling.estimators import lowered_memory_estimate

  from aminx.ebm.langevin import langevin_step, run_langevin_equilibration
  from aminx.ebm.model import ProteinEBMModel

  jax.config.update("jax_log_compiles", True)
  counter = _CompileCounter()
  logging.getLogger("jax").addHandler(counter)
  # jax_log_compiles emits "Compiling ..." at WARNING; DEBUG would log during the timed compile.
  logging.getLogger("jax").setLevel(logging.WARNING)

  grad = kind == "grad"
  length = args.grad_length if grad else args.length
  batch = args.grad_batch if grad else args.batch
  n_steps = args.grad_steps if grad else args.n_steps
  key = jax.random.PRNGKey(args.seed)
  k_coords, k_aa, k_traj = jax.random.split(key, 3)
  coords = jax.random.normal(k_coords, (batch, length, 3))
  aatype = jax.random.randint(k_aa, (length,), 0, 20)
  mask = jnp.ones((length,))
  t = jnp.asarray(DIFFUSION_TIME)
  keys = jax.random.split(k_traj, batch)
  dims = TINY_DIMS if args.model == "tiny" else {}
  model = ProteinEBMModel(key=jax.random.PRNGKey(0), **dims)
  params, static = eqx.partition(model, eqx.is_inexact_array)

  def while_one(p: Any, c: Any, k: Any) -> Any:
    return run_langevin_equilibration(eqx.combine(p, static), c, aatype, t, mask, n_steps, DT, k)

  def step_body(p: Any) -> Any:
    m = eqx.combine(p, static)

    # Same key order as run_langevin_equilibration's body_fun, so outputs match it.
    def body(carry: Any, _: Any) -> Any:
      coords_i, key_i = carry
      key_i, step_key = jax.random.split(key_i)
      return (langevin_step(m, coords_i, aatype, t, DT, mask, step_key), key_i), None

    return body

  def scan_one(p: Any, c: Any, k: Any) -> Any:
    (final, _), _ = jax.lax.scan(step_body(p), (c, k), None, length=n_steps)
    return final

  def intent_one(p: Any, c: Any, k: Any) -> Any:
    (final, _), _ = intent.loop(step_body(p), (c, k), length=n_steps, name="langevin")
    return final

  def intent_fn(p: Any, c: Any, k: Any) -> Any:
    return intent.map(lambda ck: intent_one(p, *ck), (c, k), name="trajectories")

  ctx = intent.Context(chunk_size=args.chunk)
  record: dict[str, Any] = {
    "form": form,
    "kind": kind,
    "length": length,
    "batch": batch,
    "n_steps": n_steps,
    "backend": jax.default_backend(),
  }
  stem = f"{form}_{kind}"

  if kind == "plan":
    vmapped = jax.vmap(while_one, in_axes=(None, 0, 0))
    t0 = time.perf_counter()
    vmap_est = lowered_memory_estimate(vmapped, params, coords, keys)
    record["vmap_estimate_bytes"] = vmap_est
    record["vmap_estimate_s"] = time.perf_counter() - t0
    tight = intent.Context(chunk_size=args.chunk, memory_budget=vmap_est // 2)
    record["budget_bytes"] = vmap_est // 2
    _write_json(cell_dir / f"{stem}.pre.json", dict(record))
    t0 = time.perf_counter()
    try:
      plan = intent.compile(intent_fn, tight).plan_for(params, coords, keys)
      record["tight_result"] = "plan:" + ",".join(
        f"{d.key}={type(d.primal).__name__}" for d in plan.decisions
      )
    except intent.UnsatisfiableContextError as exc:
      record["tight_result"] = f"unsat:{exc.constraint}"
    record["plan_s"] = time.perf_counter() - t0
    record["load_1m"] = [load_start, os.getloadavg()[0]]
    record["gpu_others"] = _gpu_others()
    _write_json(cell_dir / f"{stem}.json", record)
    return

  if grad:

    def loss_of(one: Any) -> Any:
      def loss(p: Any, c: Any) -> Any:
        if one is None:
          final = intent.compile(intent_fn, ctx)(p, c, keys)
        else:
          final = jax.vmap(one, in_axes=(None, 0, 0))(p, c, keys)
        return jnp.mean(final**2)

      return loss

    fn = eqx.filter_jit(jax.grad(loss_of(None if form == "intent" else scan_one), argnums=(0, 1)))
    call_args = (params, coords)
  elif form == "intent":
    compiled = intent.compile(intent_fn, ctx)
    t0 = time.perf_counter()
    plan = compiled.plan_for(params, coords, keys)
    record["plan_s"] = time.perf_counter() - t0
    record["plan"] = [
      {"key": str(d.key), "primal": type(d.primal).__name__, "diff": type(d.diff).__name__ if d.diff else None}
      for d in plan.decisions
    ]
    fn = compiled
    call_args = (params, coords, keys)
  else:
    one = jax.vmap(while_one, in_axes=(None, 0, 0))
    if form == "safemap_while":

      def mapped(p: Any, c: Any, k: Any) -> Any:
        return jax.lax.map(lambda ck: while_one(p, *ck), (c, k), batch_size=args.chunk)

      fn = eqx.filter_jit(mapped)
    else:
      fn = eqx.filter_jit(one)
    call_args = (params, coords, keys)

  try:
    record.update(_primitive_counts(jax.make_jaxpr(lambda *a: fn(*a))(*call_args)))
  except Exception as exc:  # noqa: BLE001 -- recorded, not gated: the cell still times the real call
    record["jaxpr_error"] = repr(exc)
  _write_json(cell_dir / f"{stem}.pre.json", dict(record))

  t0 = time.perf_counter()
  first = jax.block_until_ready(fn(*call_args))
  record["first_call_s"] = time.perf_counter() - t0
  record["compile_s"] = record["first_call_s"] + record.get("plan_s", 0.0)
  compiles_before = counter.count
  times = []
  for _ in range(args.repeats):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*call_args))
    times.append(time.perf_counter() - t0)
  record["step_s"] = float(np.median(times))
  record["step_cv"] = float(np.std(times) / np.mean(times))
  record["steady_recompiles"] = counter.count - compiles_before
  stats = jax.devices()[0].memory_stats() or {}
  record["peak_bytes"] = stats.get("peak_bytes_in_use")
  record["load_1m"] = [load_start, os.getloadavg()[0]]
  record["gpu_others"] = _gpu_others()

  arrays_path = cell_dir / f"{stem}.npz"
  np.savez(arrays_path, *[np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(first)])
  record["arrays"] = arrays_path.name
  _write_json(cell_dir / f"{stem}.json", record)


def _stamped(cell_dir: Path, form: str, kind: str, stamp: dict[str, str]) -> dict[str, Any] | None:
  path = cell_dir / f"{form}_{kind}.json"
  if not path.exists():
    return None
  rec = json.loads(path.read_text())
  if rec.get("stamp") != stamp:
    return None
  return rec


def _max_abs(a: Any, b: Any) -> float:
  import numpy as np

  return float(max(np.max(np.abs(a[n] - b[n])) for n in a.files))


def _aggregate(cell_dir: Path, recs: dict[tuple[str, str], dict[str, Any]], hazard_timeout_s: int) -> dict[str, Any]:
  import numpy as np

  def r(form: str, kind: str, field: str, default: Any = None) -> Any:
    return recs[(form, kind)].get(field, default)

  def arrays(form: str, kind: str) -> Any:
    return np.load(cell_dir / recs[(form, kind)]["arrays"])

  hazard = recs[("safemap_while", "primal")]
  hazard_timed_out = bool(hazard.get("timed_out", False))
  hazard_compile = float(hazard_timeout_s) if hazard_timed_out else hazard["compile_s"]
  vmap_compile = r("vmap_while", "primal", "compile_s")
  intent_vs_vmap = _max_abs(arrays("intent", "primal"), arrays("vmap_while", "primal"))
  safemap_vs_vmap = -1.0 if hazard_timed_out else _max_abs(arrays("safemap_while", "primal"), arrays("vmap_while", "primal"))

  g_ref, g_intent = arrays("scan_ref", "grad"), arrays("intent", "grad")
  # Gated: relative L2 over all leaves at once. Reported only: the per-leaf maximum, with the
  # denominator floored at 1e-6 of the global norm. A leaf whose true gradient is rounding noise
  # (a LayerNorm bias the loss is invariant to reads ~1e-10 of the global norm) compares noise
  # to noise per leaf.
  sq = lambda x: float(np.sum(np.asarray(x, dtype=np.float64) ** 2))  # noqa: E731
  global_norm = math.sqrt(sum(sq(g_ref[n]) for n in g_ref.files))
  global_diff = math.sqrt(sum(sq(g_ref[n].astype(np.float64) - g_intent[n]) for n in g_ref.files))
  rel = []
  for name in g_ref.files:
    a, b = g_ref[name], g_intent[name]
    denom = max(float(np.linalg.norm(a)), 1e-6 * global_norm, 1e-30)
    rel.append(float(np.linalg.norm(a - b)) / denom)
  grad_nonzero = any(float(np.max(np.abs(g_ref[n]))) > 0 for n in g_ref.files)

  tight = recs[("intent_tight", "plan")]
  tight_timed_out = bool(tight.get("timed_out", False))
  ref_peak = r("scan_ref", "grad", "peak_bytes")
  intent_peak = r("intent", "grad", "peak_bytes")
  intent_plan = r("intent", "primal", "plan") or []
  outer = [d["primal"] for d in intent_plan if d["key"].count("/") == 1]
  return {
    "compile_contrast": hazard_compile / vmap_compile,
    "hazard_timed_out": hazard_timed_out,
    "hazard_scan_encloses_while": bool(hazard.get("scan_encloses_while", False)),
    "intent_scan_encloses_while": bool(r("intent", "primal", "scan_encloses_while", True)),
    "intent_outer_strategy": outer[0] if len(outer) == 1 else "|".join(outer) or "missing",
    "intent_vs_vmap_compile": r("intent", "primal", "compile_s") / vmap_compile,
    "intent_first_call_vs_vmap_compile": r("intent", "primal", "first_call_s") / vmap_compile,
    "intent_vs_vmap_step": r("intent", "primal", "step_s") / r("vmap_while", "primal", "step_s"),
    "intent_vs_vmap_max_abs": intent_vs_vmap,
    "safemap_vs_vmap_max_abs": safemap_vs_vmap,
    "grad_rel_l2_global": global_diff / max(global_norm, 1e-30),
    "grad_rel_l2_leaf_max": float(max(rel)),
    "grad_nonzero": grad_nonzero,
    "intent_grad_has_while": int(r("intent", "grad", "while_count", -1)) != 0,
    "scan_ref_grad_has_while": int(r("scan_ref", "grad", "while_count", -1)) != 0,
    "intent_vs_ref_grad_peak": (intent_peak / ref_peak) if ref_peak and intent_peak else -1.0,
    "intent_steady_recompiles": r("intent", "primal", "steady_recompiles") + r("intent", "grad", "steady_recompiles"),
    "tight_timed_out": tight_timed_out,
    "tight_plan_s": float(hazard_timeout_s) if tight_timed_out else tight["plan_s"],
    "tight_result": "timeout" if tight_timed_out else tight["tight_result"],
    "vmap_primal_compile_s": vmap_compile,
    "safemap_primal_compile_s": hazard_compile,
    "intent_primal_compile_s": r("intent", "primal", "compile_s"),
    "intent_plan_s": r("intent", "primal", "plan_s"),
    "vmap_primal_step_s": r("vmap_while", "primal", "step_s"),
    "intent_primal_step_s": r("intent", "primal", "step_s"),
    "scan_ref_grad_compile_s": r("scan_ref", "grad", "compile_s"),
    "intent_grad_compile_s": r("intent", "grad", "compile_s"),
    "scan_ref_grad_step_s": r("scan_ref", "grad", "step_s"),
    "intent_grad_step_s": r("intent", "grad", "step_s"),
    "max_load_1m": max(max(rec.get("load_1m") or [0.0]) for rec in recs.values()),
    "backend": r("vmap_while", "primal", "backend"),
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--length", type=int, default=128)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--n-steps", type=int, default=20)
  parser.add_argument("--chunk", type=int, default=16)
  parser.add_argument("--grad-length", type=int, default=64)
  parser.add_argument("--grad-batch", type=int, default=4)
  parser.add_argument("--grad-steps", type=int, default=4)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--seed", type=int, default=128)
  parser.add_argument("--model", choices=("production", "tiny"), default="production")
  parser.add_argument("--cell-timeout-s", type=int, default=3600)
  parser.add_argument("--hazard-timeout-s", type=int, default=2400)
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
  passthrough = [
    "--length", str(args.length), "--batch", str(args.batch), "--n-steps", str(args.n_steps),
    "--chunk", str(args.chunk), "--grad-length", str(args.grad_length), "--grad-batch", str(args.grad_batch),
    "--grad-steps", str(args.grad_steps), "--repeats", str(args.repeats), "--seed", str(args.seed),
    "--model", args.model, "--out", str(args.out),
  ]
  recs: dict[tuple[str, str], dict[str, Any]] = {}
  for form, kind in CELLS:
    rec = _stamped(cell_dir, form, kind, stamp)
    if rec is not None:
      logger.info("reusing %s/%s from %s", form, kind, cell_dir)
      recs[(form, kind)] = rec
      continue
    hazard = (form, kind) in HAZARD_CELLS
    timeout = args.hazard_timeout_s if hazard else args.cell_timeout_s
    logger.info("running %s/%s (timeout %ds)", form, kind, timeout)
    path = cell_dir / f"{form}_{kind}.json"
    t0 = time.perf_counter()
    try:
      subprocess.run([sys.executable, __file__, *passthrough, "--cell", form, kind], check=True, timeout=timeout)
      rec = json.loads(path.read_text())
    except subprocess.TimeoutExpired:
      if not hazard:
        raise
      pre = cell_dir / f"{form}_{kind}.pre.json"
      rec = json.loads(pre.read_text()) if pre.exists() else {"form": form, "kind": kind}
      rec["timed_out"] = True
      logger.warning("%s/%s timed out after %ds; recorded as censored", form, kind, timeout)
    rec["stamp"] = stamp
    rec["cell_wall_s"] = time.perf_counter() - t0
    _write_json(path, rec)
    recs[(form, kind)] = rec
    logger.info("%s/%s done in %.1fs", form, kind, rec["cell_wall_s"])

  result = _aggregate(cell_dir, recs, args.hazard_timeout_s)
  for k in ("length", "batch", "n_steps", "chunk", "grad_length", "grad_batch", "grad_steps", "model"):
    result[k] = getattr(args, k)
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
