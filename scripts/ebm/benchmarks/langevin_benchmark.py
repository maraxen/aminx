"""ProteinEBM Langevin-sampler throughput/latency benchmark: JAX (ported) vs PyTorch (reference).

Backlog node **E11d** (the fourth and final benchmark harness in the
ProteinEBM decomposition follow-on epic, after E11a/decoy_benchmark.py,
E11b/ddg_benchmark.py, E11c/biasing_benchmark.py -- see
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` and design spec
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §8.3). This
script benchmarks the *sampler itself* (backlog nodes E9-core/E9-outer, now
committed): ``aminx.ebm.langevin.langevin_step``/``run_langevin_equilibration``
and, one level up, ``aminx.ebm.langevin_schedule.run_annealing_schedule``'s
own inner primitive. It measures *speed*, not correctness -- E9's own test
suite (``tests/ebm/test_langevin.py``, ``tests/ebm/test_langevin_schedule.py``)
already validated convergence/dispatch behavior; this script only times it.

**Apples-to-apples methodology (mirrors decoy_benchmark.py/E11a's §8.3
conventions -- read before trusting any number this script prints):**

1. **Bucket-aligned lengths.** Benchmarked at exactly the residue-axis bucket
   boundaries the EPIC locked at E4.5: ``{64, 128, 256, 512}``
   (:data:`DEFAULT_LENGTHS`). Coordinates are synthetic (i.i.d. Gaussian) --
   this is a throughput benchmark, not a correctness benchmark.
2. **Fixed single noise level ``t``, single pre-loaded model.** Per the task
   brief and ``aminx.ebm.langevin``'s own documented scope limit, this script
   benchmarks *only* the inner sampler at one fixed ``t``
   (:data:`DEFAULT_DIFFUSION_TIME`, matching the other E11x scripts' MVP
   target), against a single model. The **outer** noise-schedule loop and its
   ``jax.lax.cond``/``jax.lax.switch`` model-swap dispatcher
   (``aminx.ebm.langevin_schedule.run_annealing_schedule``/
   ``select_model_for_t``) are **out of scope** here -- benchmarking the
   model-swap dispatch's overhead, or the full multi-round E10 pipeline, is a
   separate, not-yet-built node. This script only exercises
   ``langevin_step``/``run_langevin_equilibration`` directly.
3. **PyTorch baseline must not pay for an unnecessary retained autograd
   graph.** Per-step energy/aux-score evaluation uses ``torch.no_grad()`` --
   the Langevin sampler's ``aux_score``/``r_update_aux`` path is
   non-conservative by construction (no ``jax.grad``/``torch.autograd.grad``
   anywhere in ``langevin_step`` either), so unlike E11a/E11b's
   ``score_grad_ms`` metric there is no gradient-graph-retention footgun to
   route around here -- this point is included for consistency with the
   sibling scripts' documented methodology, not because it changes anything
   in this script.
4. **Exclude JIT/compile warmup.** For JAX, every timed metric is called once
   (untimed) before the timed-repeats loop, forcing ``eqx.filter_jit``
   compilation for that shape; only steady-state (post-compile) calls are
   timed. For PyTorch there is no JIT step, but a matching untimed warmup
   call is made anyway for cache-warming parity.
5. **Metrics** (matches the ``[result_schema]`` in the sidecar
   ``langevin_benchmark.bth.toml``):
   - ``langevin_steps_per_sec``: batched multi-trajectory equilibration
     throughput. ``n_trajectories`` independent chains (distinct synthetic
     starting coordinates, distinct PRNG keys, shared sequence/mask -- the
     same "shared sequence, varying per-row state" convention
     ``decoy_benchmark.py``'s ``n_decoys`` axis uses) each run
     :data:`DEFAULT_N_STEPS` (or ``--n-steps``) full Euler-Maruyama update
     steps via ``run_langevin_equilibration``/a PyTorch step-loop of the
     matching per-step math. JAX batches the ``n_trajectories`` axis via
     ``jax.vmap`` around ``run_langevin_equilibration`` (this is a genuine
     batched ``while_loop`` under vmap, not per-trajectory-varying trip
     counts -- every lane runs the same fixed ``n_steps``, so this does not
     exercise JAX's masked/select-based batched-while-loop semantics, only
     its "same trip count, different data" case). PyTorch times a Python
     ``for`` loop of ``n_steps`` calls to the equivalent per-step update, with
     the trajectory axis as a plain batch dimension on every tensor (no
     per-trajectory Python loop) -- mirroring how ``run_dynamics.py`` batches
     its own ``bsize`` axis. Reported as ``(n_trajectories * n_steps) /
     elapsed_seconds``.
   - ``langevin_step_ms``: single-trajectory, single-step latency -- exactly
     one call to ``langevin_step`` (JAX) or its replicated PyTorch
     per-step update, timed directly (no ``run_langevin_equilibration``
     loop). This is E11d's analog of E11a/E11b's ``score_grad_ms``: a
     single-call latency number, not a batched-throughput number.
6. Every per-(length, impl) row is written to a structured JSON file
   (``--out``), not just logged -- see :func:`main`.

**PyTorch per-step replication (design choice, per the task brief's option
(a)).** ``~/repos/ProteinEBM/protein_ebm/scripts/run_dynamics.py`` is a
monolithic CLI script, not an importable function -- there is no smaller
importable helper for its inner per-``t`` update (verified by reading the
file; the update math lives inline in ``main()``, lines ~830-845). Per the
task brief and matching ``decoy_benchmark.py``'s own precedent of importing
``protein_ebm.model.ebm.ProteinEBM``/``protein_ebm.model.r3_diffuser.R3Diffuser``
directly rather than shelling out to the CLI script, this script imports
``ProteinEBM`` directly and replicates *only* the non-Metropolis branch's
per-step update (``run_dynamics.py`` lines ~830-843, the same lines
``aminx.ebm.langevin.langevin_step``'s own docstring says it ported) --
:func:`_pytorch_langevin_step` below. It does **not** reproduce the
Metropolis-Hastings branch (E9-core's own ``metropolis_hastings_step`` is
similarly out of scope here -- this script only benchmarks the
``use_metropolis=False`` plain-Euler-Maruyama path, :data:`DEFAULT_LENGTHS`'s
sampler default), nor the "forward Ito step"/``dt_rev``-transport
discretization details ``aminx.ebm.langevin_schedule``'s module docstring
documents as deliberately not reproduced by the *outer* schedule loop (this
script never touches the outer schedule at all -- see methodology point 2).
``ref_model.diffuser`` (a real, already-constructed ``R3Diffuser`` instance
stored as ``ProteinEBM.__init__``'s ``self.diffuser``, confirmed by reading
``ebm.py``) supplies ``drift_coef``/``diffusion_coef``/
``config.coordinate_scaling`` directly -- no separate ``R3Diffuser`` import
or construction is needed in this script.

**What this script does NOT do (read before citing any number):**

- It does not apply the mandatory SM120 ``XLA_FLAGS=--xla_gpu_shard_
  autotuning=false`` cluster workaround (``~/.claude/rules/CLUSTER.md``) --
  that flag only matters on the actual ``pi_so3`` GPU nodes this script never
  touches locally. A future cluster run of this exact script MUST set that
  flag or its JAX throughput numbers will be ~1000x wrong.
- It does not run on GPU. This environment has no GPU (JAX and PyTorch both
  fall back to CPU here). Every number this script has actually produced
  (see the L1/L2 gate report) is a CPU number, reported honestly via the
  ``device`` column -- it is NOT the real JAX-vs-PyTorch hardware comparison
  the design spec's target is about. That comparison is a follow-on cluster
  (L3) run, explicitly out of scope for this dispatch.
- ``--smoke`` mode does NOT exercise the real E3.5-ported checkpoint weights
  (430MB) -- it builds tiny, randomly-initialized models of the *same*
  architecture (both ``protein_ebm.model.ebm.ProteinEBM`` and
  ``aminx.ebm.model.ProteinEBMModel``), matching ``decoy_benchmark.py``'s own
  ``--smoke`` precedent, so the harness's own dispatch/timing/JSON-schema
  logic is exercised end-to-end fast (<60s on CPU).
- It does NOT benchmark the full multi-round E10 pipeline
  (``aminx.ebm.structure_prediction``) or the E9-outer model-swap dispatcher
  (``aminx.ebm.langevin_schedule.run_annealing_schedule``/
  ``select_model_for_t``) -- both are explicitly out of scope (methodology
  point 2). A future node could extend this script (or add a sibling) to time
  ``run_annealing_schedule``'s own per-level dispatch overhead specifically.
- It does NOT benchmark ``metropolis_hastings_step`` (the accept/reject-
  corrected variant) -- only the default, plain Euler-Maruyama
  ``langevin_step`` path, matching ``run_langevin_equilibration``'s
  ``use_metropolis=False`` default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
from xtrax.tiling import AxisSpec, BatchPlanner, ChunkedMap

from aminx.ebm.checkpoint import load_pytorch_checkpoint
from aminx.ebm.langevin import DEFAULT_EFFECTIVE_TEMP_SCALING, langevin_step, run_langevin_equilibration
from aminx.ebm.model import ProteinEBMModel

if TYPE_CHECKING:
  from collections.abc import Mapping

  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

log = logging.getLogger("langevin_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()

# Residue-axis bucket boundaries locked at the E4.5 xtrax HiTL gate -- the
# same points every E11x sibling sweeps.
DEFAULT_LENGTHS: tuple[int, ...] = (64, 128, 256, 512)
SMOKE_LENGTHS: tuple[int, ...] = (64, 128)

DEFAULT_DIFFUSION_TIME = 0.05  # ProteinEBM-x MVP target t (matches E11a/E11b).
DEFAULT_DT = 1e-3  # Matches run_dynamics.py's own `--dt` default (0.001).

# --- --smoke-mode architecture (tiny, randomly-initialized; NOT the real
# checkpoint -- duplicated from decoy_benchmark.py's smoke dims rather than
# imported, per the E11a-c precedent of each E11x script being a standalone
# entry point). ---
SMOKE_TOKEN_S = 32
SMOKE_TOKEN_Z = 16
SMOKE_DIM_FOURIER = 12
SMOKE_TRANSITION_LAYERS = 1
SMOKE_DEPTH = 2
SMOKE_HEADS = 2
SMOKE_NUM_CONTACT_EMBEDDINGS = 2


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--lengths",
    type=str,
    default=None,
    help="Comma-separated residue counts, e.g. '64,128,256,512'. Defaults to "
    "the bucket points (DEFAULT_LENGTHS), or SMOKE_LENGTHS under --smoke.",
  )
  parser.add_argument(
    "--n-trajectories",
    type=int,
    default=None,
    help="Single independent-trajectory batch size (shorthand for --batch-sizes N). "
    "Mutually exclusive with --batch-sizes.",
  )
  parser.add_argument(
    "--batch-sizes",
    type=int,
    nargs="+",
    default=None,
    help="Trajectory batch sizes to sweep, e.g. '--batch-sizes 4 16 64 400'. "
    "400 is run_dynamics.py's own --batch_size default. Default: (4, 16, 64, 400), "
    "or (2,) under --smoke.",
  )
  parser.add_argument(
    "--n-steps",
    type=int,
    default=None,
    help="Euler-Maruyama steps per trajectory for the langevin_steps_per_sec "
    "metric (default 20, or 3 under --smoke).",
  )
  parser.add_argument(
    "--n-repeats",
    type=int,
    default=None,
    help="Timed repeats per (length, metric), after the untimed warmup call "
    "(default 5, or 2 under --smoke).",
  )
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--diffusion-time",
    type=float,
    default=DEFAULT_DIFFUSION_TIME,
    help="Fixed VP-SDE diffusion time t (ProteinEBM-x MVP target, design spec §9).",
  )
  parser.add_argument(
    "--dt",
    type=float,
    default=DEFAULT_DT,
    help="Euler-Maruyama step size (matches run_dynamics.py's --dt default).",
  )
  parser.add_argument("--out", type=Path, required=True, help="JSON output path.")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Reduced-scale, real-architecture-but-untrained local smoke run (L2 "
    "gate): tiny synthetic dims, no checkpoint load, <60s on CPU. See module docstring.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="L1 gate: import + path + one-shot construct/execute check on a "
    "trivial tiny model. No timed loop, no dependence on the real checkpoint "
    "being present.",
  )
  args = parser.parse_args()
  if args.n_repeats is not None and args.n_repeats < 1:
    parser.error("--n-repeats must be >= 1")
  if args.n_trajectories is not None and args.n_trajectories < 1:
    parser.error("--n-trajectories must be >= 1")
  if args.n_trajectories is not None and args.batch_sizes is not None:
    parser.error("--n-trajectories and --batch-sizes are mutually exclusive")
  if args.batch_sizes is not None and any(b < 1 for b in args.batch_sizes):
    parser.error("--batch-sizes entries must all be >= 1")
  if args.n_steps is not None and args.n_steps < 1:
    parser.error("--n-steps must be >= 1")
  return args


DEFAULT_BATCH_SIZES: tuple[int, ...] = (4, 16, 64, 400)
SMOKE_BATCH_SIZES: tuple[int, ...] = (2,)

# Per-length safe batch size for the n_trajectories axis, derived from the
# empirically confirmed Blackwell/SM120 XLA-autotuning crash thresholds
# (.praxia/docs/audits/260716_proteinebm-parity-report.md §7): every batch
# size at or below these values has succeeded cleanly across three separate
# cluster jobs (18059808, 18069513, 18161115) and three jax/jaxlib versions
# (0.9.2, 0.10.2, 0.11.0); above them, XLA's kernel-autotuning search hits
# CUDA_ERROR_ILLEGAL_ADDRESS. This feeds an xtrax AxisSpec/BatchPlanner
# decision (see _dispatch_trajectories) rather than a crash-and-retry loop:
# the n_trajectories axis is proactively dispatched via Vmap at or below the
# threshold, or via SafeMap (jax.lax.map in chunks of this size) above it --
# the REQUESTED batch size is always what gets measured and reported; only
# the internal execution strategy changes, so a recovered cell can never be
# confused with data at some other, smaller batch size, and there is no
# wasted/crashed attempt to worry about confounding the timing.
SAFE_TRAJECTORY_BATCH_BY_LENGTH: dict[int, int] = {
  64: 400,  # no crash observed at any tested batch size up to 400
  128: 64,  # crashes at 400; 64 is the largest confirmed-safe size
  256: 16,  # crashes at 64; 16 is the largest confirmed-safe size
  512: 4,  # crashes at 16; 4 is the largest confirmed-safe size
}
# Conservative fallback for a length outside the table above (e.g. a custom
# --lengths value) -- the smallest safe size confirmed at any length so far.
DEFAULT_SAFE_TRAJECTORY_BATCH = 4


def _resolve_run_params(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
  """Resolve (lengths, batch_sizes, n_steps, n_repeats), applying --smoke's reduced defaults."""
  if args.lengths is not None:
    lengths = tuple(int(x) for x in args.lengths.split(","))
  else:
    lengths = SMOKE_LENGTHS if args.smoke else DEFAULT_LENGTHS

  if args.n_trajectories is not None:
    batch_sizes = (args.n_trajectories,)
  elif args.batch_sizes is not None:
    batch_sizes = tuple(args.batch_sizes)
  else:
    batch_sizes = SMOKE_BATCH_SIZES if args.smoke else DEFAULT_BATCH_SIZES

  n_steps = args.n_steps if args.n_steps is not None else (3 if args.smoke else 20)
  n_repeats = args.n_repeats if args.n_repeats is not None else (2 if args.smoke else 5)
  return lengths, batch_sizes, n_steps, n_repeats


# ---------------------------------------------------------------------------
# Model construction -- both the real PyTorch reference class and the real
# aminx.ebm.model.ProteinEBMModel class, either from the real E3.5 checkpoint
# (full run) or from tiny random init (--smoke / --dry-run). Duplicated from
# decoy_benchmark.py rather than imported (each E11x script is a standalone
# entry point -- see module docstring).
# ---------------------------------------------------------------------------


def _smoke_model_configs() -> tuple[SimpleNamespace, SimpleNamespace]:
  """Minimal (model_cfg, diffuser_cfg) namespaces for the tiny --smoke architecture.

  Identical to ``decoy_benchmark.py``'s ``_smoke_model_configs`` -- see that
  function's docstring for why ``aux_score`` must stay ``True`` (a reference
  code wart in ``ProteinEBM.forward``, not a port bug).
  """
  model_cfg = SimpleNamespace(
    token_s=SMOKE_TOKEN_S,
    token_z=SMOKE_TOKEN_Z,
    dim_fourier=SMOKE_DIM_FOURIER,
    conditioning_transition_layers=SMOKE_TRANSITION_LAYERS,
    token_transformer_depth=SMOKE_DEPTH,
    token_transformer_heads=SMOKE_HEADS,
    num_contact_embeddings=SMOKE_NUM_CONTACT_EMBEDDINGS,
    aux_score=True,
    predict_sidechain=False,
    diffuse_sidechain=False,
    use_self_conditioning=True,
    use_present_embedding=False,
    use_attention_mask=False,
    direct_score=False,
  )
  diffuser_cfg = SimpleNamespace(min_b=0.1, max_b=20.0, coordinate_scaling=0.1)
  return model_cfg, diffuser_cfg


def _build_reference_model(
  reference_repo: Path,
  model_cfg: Any,  # noqa: ANN401 -- duck-typed ml_collections.ConfigDict | SimpleNamespace
  diffuser_cfg: Any,  # noqa: ANN401
  state_dict: "Mapping[str, torch.Tensor] | None",
  device: "str | torch.device" = "cpu",
) -> "ProteinEBM":
  """Construct the real reference ``ProteinEBM``, optionally strict-loading a checkpoint.

  Mirrors ``decoy_benchmark.py::_build_reference_model`` exactly (same
  construction + ``load_state_dict(strict=False)`` + missing/unexpected
  check) -- reimplemented here (not imported), per the E11x standalone-entry-
  point convention.

  ``device`` defaults to ``"cpu"`` (matches the checkpoint's own
  ``map_location="cpu"`` load) but callers doing real throughput measurement
  MUST pass the actual compute device -- the model is moved there via
  ``.to(device)`` before returning. Without this, every PyTorch number this
  script reports would silently be a CPU number, no matter what the JSON's
  ``device`` field claims (this was a real, previously-undiscovered bug: see
  ``.praxia/docs/audits/260716_proteinebm-parity-report.md`` §7 -- confirmed
  live on cluster hardware via ``next(ref_model.parameters()).device``).
  """
  sys.path.insert(0, str(reference_repo))
  from protein_ebm.model.ebm import ProteinEBM  # noqa: PLC0415
  from protein_ebm.model.r3_diffuser import R3Diffuser  # noqa: PLC0415

  diffuser = R3Diffuser(diffuser_cfg)
  model = ProteinEBM(model_cfg, diffuser)
  if state_dict is not None:
    stripped = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
      msg = f"reference model.load_state_dict was not exact: missing={missing}, unexpected={unexpected}"
      raise RuntimeError(msg)
  model.eval()
  return model.to(device)


def _build_jax_model(
  model_cfg: Any,  # noqa: ANN401
  seed: int,
  state_dict: "Mapping[str, Any] | None",
) -> ProteinEBMModel:
  """Construct the real ``aminx.ebm.model.ProteinEBMModel``, optionally porting a checkpoint."""
  key = jax.random.PRNGKey(seed)
  model = ProteinEBMModel(
    token_s=model_cfg.token_s,
    token_z=model_cfg.token_z,
    dim_fourier=model_cfg.dim_fourier,
    conditioning_transition_layers=model_cfg.conditioning_transition_layers,
    transformer_depth=model_cfg.token_transformer_depth,
    transformer_heads=model_cfg.token_transformer_heads,
    num_contact_embeddings=model_cfg.num_contact_embeddings,
    key=key,
  )
  if state_dict is not None:
    model, _report = load_pytorch_checkpoint(model, state_dict)
  return model


def build_models(
  args: argparse.Namespace,
) -> tuple[ProteinEBMModel, "ProteinEBM", Any, Any]:
  """Build (jax_model, reference_model, model_cfg, diffuser_cfg) per ``--smoke``/full-run mode.

  The reference PyTorch model is moved to ``_torch_device()``'s actual device
  (``ref_model.to(device)`` inside ``_build_reference_model``) -- previously
  it was left on CPU (the checkpoint's own ``map_location="cpu"`` load
  target) regardless of GPU availability, so every reported PyTorch number
  was silently a CPU number despite the JSON claiming ``"device": "cuda"``.
  """
  import torch  # noqa: PLC0415

  device = _torch_device()

  if args.smoke:
    log.info("Building SMOKE models: tiny synthetic dims, randomly initialized, no checkpoint load.")
    model_cfg, diffuser_cfg = _smoke_model_configs()
    jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
    ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None, device=device)
    return jax_model, ref_model, model_cfg, diffuser_cfg

  if not args.checkpoint.exists():
    msg = f"checkpoint not found: {args.checkpoint} (see E3.5 task brief for the authorized source)"
    raise FileNotFoundError(msg)
  if not args.reference_repo.exists():
    msg = f"reference repo not found: {args.reference_repo}"
    raise FileNotFoundError(msg)

  log.info("Loading checkpoint: %s", args.checkpoint)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model_cfg = ckpt["hyper_parameters"]["config"].model
  diffuser_cfg = ckpt["hyper_parameters"]["config"].diffuser

  log.info("Building + strict-loading reference PyTorch model...")
  ref_model = _build_reference_model(
    args.reference_repo, model_cfg, diffuser_cfg, ckpt["state_dict"], device=device,
  )

  log.info("Building + porting JAX ProteinEBMModel...")
  jax_model = _build_jax_model(model_cfg, args.seed, ckpt["state_dict"])

  return jax_model, ref_model, model_cfg, diffuser_cfg


# ---------------------------------------------------------------------------
# Synthetic trajectory inputs
# ---------------------------------------------------------------------------


def _synthetic_trajectories(n_trajectories: int, length: int, seed: int) -> dict[str, np.ndarray]:
  """Build one fixed synthetic trajectory batch at a given residue length.

  Coordinates are i.i.d. Gaussian in already-scaled ("nm") units -- this is a
  throughput benchmark, not a correctness benchmark (mirrors
  ``decoy_benchmark.py``'s ``_synthetic_decoys`` exactly, with ``n_decoys``
  renamed ``n_trajectories``: each independent Langevin chain gets its own
  starting coordinates and, downstream, its own PRNG key, while
  ``aatype``/``residue_index``/``chain_id``/``contacts``/``mask`` stay shared
  across the batch axis).
  """
  rng = np.random.default_rng(seed)
  coords = rng.normal(scale=1.0, size=(n_trajectories, length, 3)).astype(np.float32)
  aatype = rng.integers(0, 21, size=(length,)).astype(np.int64)
  residue_index = np.arange(length, dtype=np.int64)
  chain_id = np.zeros((length,), dtype=np.int64)
  contacts = np.zeros((length,), dtype=np.int64)
  mask = np.ones((length,), dtype=bool)
  return {
    "coords": coords,
    "aatype": aatype,
    "residue_index": residue_index,
    "chain_id": chain_id,
    "contacts": contacts,
    "mask": mask,
  }


def _timed_calls(fn: Any, n_repeats: int) -> list[float]:  # noqa: ANN401 -- Callable[[], object]
  """Run ``fn`` (already warmed up) ``n_repeats`` times, returning per-call wall-clock seconds."""
  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    fn()
    times.append(time.perf_counter() - start)
  return times


def _wall_clock_ms_stats(times_seconds: list[float]) -> tuple[float, float]:
  arr_ms = np.asarray(times_seconds) * 1000.0
  return float(np.mean(arr_ms)), float(np.std(arr_ms))


# ---------------------------------------------------------------------------
# JAX timing (langevin_steps_per_sec via a vmapped run_langevin_equilibration;
# langevin_step_ms via a single langevin_step call). Both wrapped in
# eqx.filter_jit -- the untimed warmup call forces XLA compilation for that
# shape; timed calls hit the compiled cache (methodology point 4).
# ---------------------------------------------------------------------------


def _chunked_vmap(
  fn: Any,  # noqa: ANN401 -- Callable[[tuple[jax.Array, jax.Array]], jax.Array]
  xs: tuple[jax.Array, jax.Array],
  chunk_size: int,
) -> jax.Array:
  """Apply ``fn`` over the leading axis of ``xs`` in Python-level (trace-time-unrolled) chunks.

  Deliberately NOT ``xtrax.tiling.SafeMap``/``aminx.utils.safe_map.safe_map``:
  those lower to ``jax.lax.map``, which JAX's own docs describe as "like
  scan, implemented in terms of JAX primitives... compiled once" -- i.e. a
  ``lax.scan``. ``run_langevin_equilibration`` already contains an internal
  ``jax.lax.while_loop``, deliberately chosen (see the ``aminx.ebm.langevin``
  module docstring) because ``while_loop`` compiles ~300-400x faster than the
  ``lax.scan`` equivalent on SM120/Blackwell for a fixed-but-not-compile-time-
  constant trip count. Wrapping that in an outer ``lax.scan`` (what ``SafeMap``
  does) reintroduces exactly the compile-time pathology that choice exists to
  avoid. Confirmed two ways before writing this: SLURM job `18182309` hung for
  a full hour without finishing the compile ("warmup") call for the first
  ``SafeMap``-dispatched cell (L=128/batch=400) on node4008/SM120, and a local
  ``jax.make_jaxpr`` inspection showed the ``SafeMap`` path's jaxpr contains
  both ``scan`` and ``while`` primitives (10 eqns) versus ``Vmap``'s single
  ``while``-only equation (1 eqn) -- see
  ``.praxia/docs/audits/260716_proteinebm-parity-report.md`` §7.

  A plain Python ``for`` loop over static-size slices, each independently
  ``jax.vmap``'d, avoids ``lax.scan`` entirely -- every chunk becomes its own
  traced sub-graph (verified: zero ``scan`` nodes in the resulting jaxpr,
  output bit-identical to a single ``jax.vmap`` call). The tradeoff is a
  larger compiled program (one sub-graph per chunk instead of one reused
  scan body) in exchange for avoiding a documented, order-of-magnitude
  SM120 compile-time trap -- worth it for the handful of chunks
  (``SAFE_TRAJECTORY_BATCH_BY_LENGTH``'s ratios are all single digits) this
  script ever needs.
  """
  n = xs[0].shape[0]
  parts = [
    jax.vmap(fn)(jax.tree_util.tree_map(lambda x, s=start, e=min(start + chunk_size, n): x[s:e], xs))
    for start in range(0, n, chunk_size)
  ]
  return jax.tree_util.tree_map(lambda *p: jnp.concatenate(p, axis=0), *parts)


@eqx.filter_jit
def _jax_equilibration_batch(
  model: ProteinEBMModel,
  coords_batch: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_steps: int,
  dt: float,
  keys: jax.Array,
  safe_batch_size: int,
) -> jax.Array:
  """Dispatch ``run_langevin_equilibration`` over the ``n_trajectories`` axis via an xtrax BatchPlanner decision.

  Vmap in one shot when ``n_trajectories <= safe_batch_size``; a Python-level
  chunked vmap (:func:`_chunked_vmap`, chunks of ``safe_batch_size``) above
  it -- see :data:`SAFE_TRAJECTORY_BATCH_BY_LENGTH` for why the threshold,
  and :func:`_chunked_vmap`'s docstring for why chunking is NOT done via
  ``xtrax.tiling.SafeMap``. Either way this computes the full
  ``n_trajectories`` batch -- the strategy only changes how XLA schedules
  the work, never what gets measured, so the reported batch size always
  matches what was requested.

  ``n_steps`` is the *same* fixed value for every trajectory (a Python
  ``int``, not batched) -- so this only exercises JAX's "identical trip
  count, different data" batched-``while_loop`` case, not its per-lane-
  varying-trip-count/masking semantics (module docstring point 5).

  The planning half (``AxisSpec`` + ``BatchPlanner`` deciding ``Vmap`` vs.
  ``SafeMap``) mirrors ``aminx.ebm.plan.plan_axis``'s pattern; only the
  *execution* of a ``SafeMap`` decision is written locally, via
  :func:`_chunked_vmap` rather than ``aminx.ebm.plan.dispatch_axis`` -- that
  helper's generic signature assumes ``body`` returns the same pytree type
  it's fed one slice of at a time (fits E4's Coords-in/Energy-out calls,
  both bare ``Array``\\ s under jaxtyping) but not this ``(coords, key)``
  pair-in/``Array``-out shape, and it would dispatch ``SafeMap`` via
  ``aminx.utils.safe_map.safe_map`` (``lax.map``/``scan``-based) regardless.
  """

  def _single(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
    coords, key = pair
    return run_langevin_equilibration(model, coords, aatype, t, mask, n_steps, dt, key)

  n_trajectories = coords_batch.shape[0]
  spec = AxisSpec(name="n_trajectories", cardinality=n_trajectories, default_batch_size=safe_batch_size)
  with warnings.catch_warnings():
    # BatchPlanner warns "will raise ValueError at make_axis_dispatch time" for a
    # non-divisible (cardinality, batch_size) pair (e.g. 400 trajectories chunked
    # at 64) -- that failure mode belongs to xtrax's make_axis_dispatch iterators,
    # which this function never calls. _chunked_vmap handles a non-divisible
    # remainder chunk correctly (verified: bit-identical to a plain jax.vmap for
    # the same inputs), so the warning is a false alarm for this dispatch path
    # specifically and is suppressed rather than left as confusing log noise on
    # every non-divisible cell.
    warnings.filterwarnings("ignore", message=r".*is not divisible by batch_size.*", category=RuntimeWarning)
    decision = BatchPlanner().plan([spec]).decisions[0]
  if isinstance(decision.strategy, ChunkedMap):
    return _chunked_vmap(_single, (coords_batch, keys), decision.strategy.batch_size)
  return jax.vmap(_single)((coords_batch, keys))


@eqx.filter_jit
def _jax_single_step(
  model: ProteinEBMModel,
  coords: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  dt: float,
  mask: jax.Array,
  key: jax.Array,
) -> jax.Array:
  return langevin_step(model, coords, aatype, t, dt, mask, key)


def _run_jax_equilibration(
  model: ProteinEBMModel,
  batch: dict[str, np.ndarray],
  t: float,
  n_steps: int,
  dt: float,
  seed: int,
  n_repeats: int,
  safe_batch_size: int,
) -> list[float]:
  coords_batch = jnp.asarray(batch["coords"])
  aatype = jnp.asarray(batch["aatype"], dtype=jnp.int32)
  mask = jnp.asarray(batch["mask"])
  t_arr = jnp.asarray(t)
  n_trajectories = coords_batch.shape[0]
  keys = jax.random.split(jax.random.PRNGKey(seed), n_trajectories)

  def call() -> None:
    jax.block_until_ready(
      _jax_equilibration_batch(model, coords_batch, aatype, t_arr, mask, n_steps, dt, keys, safe_batch_size),
    )

  call()  # untimed: forces XLA compilation for this (n_trajectories, length, n_steps) shape.
  return _timed_calls(call, n_repeats)


def _run_jax_single_step(
  model: ProteinEBMModel,
  batch: dict[str, np.ndarray],
  t: float,
  dt: float,
  seed: int,
  n_repeats: int,
) -> list[float]:
  coords0 = jnp.asarray(batch["coords"][0])
  aatype = jnp.asarray(batch["aatype"], dtype=jnp.int32)
  mask = jnp.asarray(batch["mask"])
  t_arr = jnp.asarray(t)
  key = jax.random.PRNGKey(seed)

  def call() -> None:
    jax.block_until_ready(_jax_single_step(model, coords0, aatype, t_arr, dt, mask, key))

  call()  # untimed: forces XLA compilation for this length's shape.
  return _timed_calls(call, n_repeats)


# ---------------------------------------------------------------------------
# PyTorch timing -- replicates langevin_step's non-Metropolis per-step update
# (run_dynamics.py lines ~830-843: aux-score forward via compute_energy,
# Euler-Maruyama update via ref_model.diffuser's drift_coef/diffusion_coef,
# then center_random_augmentation(rotate=False)). See module docstring's
# "PyTorch per-step replication" section for exactly what is and is not
# reproduced.
# ---------------------------------------------------------------------------


def _pytorch_langevin_step(
  ref_model: "ProteinEBM",
  r_noisy: "torch.Tensor",
  aatype: "torch.Tensor",
  residue_idx: "torch.Tensor",
  mask: "torch.Tensor",
  chain_id: "torch.Tensor",
  contacts: "torch.Tensor",
  t: float,
  dt: float,
) -> "torch.Tensor":
  """One non-Metropolis Euler-Maruyama step, PyTorch reference math.

  ``effective_temp_scaling`` is fixed at ``aminx.ebm.langevin.
  DEFAULT_EFFECTIVE_TEMP_SCALING`` (1.0) on both the JAX and PyTorch sides of
  this benchmark -- neither side exposes a CLI flag for it (out of scope; see
  module docstring point 2 for why the outer, ``t``-dependent temperature
  variant is not reproduced either).
  """
  import torch  # noqa: PLC0415

  bsize = r_noisy.shape[0]
  input_feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": mask,
    "t": torch.full((bsize,), t, dtype=torch.float32, device=r_noisy.device),
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }
  with torch.no_grad():
    out = ref_model.compute_energy(input_feats, rescale_input_coords=False)
    score = out["r_update_aux"]
    diffuser = ref_model.diffuser
    coordinate_scaling = diffuser.config.coordinate_scaling
    drift = diffuser.drift_coef(r_noisy, t)
    g = diffuser.diffusion_coef(t)
    noise = torch.randn_like(r_noisy)
    r_new = (
      r_noisy
      - (drift - g**2 * score * DEFAULT_EFFECTIVE_TEMP_SCALING / coordinate_scaling) * dt
      + g * (dt**0.5) * noise / coordinate_scaling
    )
    r_new, _second = _center_random_augmentation_ref(
      ref_model, r_new, mask.float(), second_coords=out["pred_coords_aux"].reshape(r_new.shape),
    )
  return r_new


def _center_random_augmentation_ref(
  ref_model: "ProteinEBM",
  coords: "torch.Tensor",
  mask: "torch.Tensor",
  second_coords: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
  """Thin wrapper around ``protein_ebm.model.boltz_utils.center_random_augmentation``.

  Imported lazily (module-scope import would require ``reference_repo`` to
  already be on ``sys.path``, which only happens inside
  ``_build_reference_model``) -- mirrors this script's other ``from
  protein_ebm...`` imports. ``rotate=False`` matches
  ``aminx.ebm.langevin.langevin_step``'s own re-centering call
  (``center_random_augmentation(..., rotate=False)``) exactly; ``ref_model``
  is unused except to document that this helper is only ever called after a
  reference model has been built (so the import is guaranteed to succeed).
  """
  del ref_model
  from protein_ebm.model.boltz_utils import center_random_augmentation  # noqa: PLC0415

  return center_random_augmentation(
    coords, mask, rotate=False, second_coords=second_coords, return_second_coords=True,
  )


def _run_pytorch_equilibration(
  ref_model: "ProteinEBM",
  batch: dict[str, np.ndarray],
  t: float,
  n_steps: int,
  dt: float,
  n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> list[float]:
  import torch  # noqa: PLC0415

  n_trajectories, length = batch["coords"].shape[0], batch["coords"].shape[1]
  del length  # unused; kept for readability of the tile shapes below.
  r_noisy0 = torch.tensor(batch["coords"], dtype=torch.float32, device=device)
  aatype = torch.tensor(np.tile(batch["aatype"], (n_trajectories, 1)), dtype=torch.long, device=device)
  residue_idx = torch.tensor(np.tile(batch["residue_index"], (n_trajectories, 1)), dtype=torch.long, device=device)
  chain_id = torch.tensor(np.tile(batch["chain_id"], (n_trajectories, 1)), dtype=torch.long, device=device)
  contacts = torch.tensor(np.tile(batch["contacts"], (n_trajectories, 1)), dtype=torch.long, device=device)
  mask = torch.tensor(np.tile(batch["mask"], (n_trajectories, 1)), dtype=torch.bool, device=device)

  def call() -> None:
    r = r_noisy0.clone()
    for _ in range(n_steps):
      r = _pytorch_langevin_step(ref_model, r, aatype, residue_idx, mask, chain_id, contacts, t, dt)

  call()  # untimed: no JIT step in PyTorch, but a first call still primes allocator/cache state.
  return _timed_calls(call, n_repeats)


def _run_pytorch_single_step(
  ref_model: "ProteinEBM",
  batch: dict[str, np.ndarray],
  t: float,
  dt: float,
  n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> list[float]:
  import torch  # noqa: PLC0415

  coords0 = batch["coords"][0]
  r_noisy = torch.tensor(coords0, dtype=torch.float32, device=device).unsqueeze(0)
  aatype = torch.tensor(batch["aatype"], dtype=torch.long, device=device).unsqueeze(0)
  residue_idx = torch.tensor(batch["residue_index"], dtype=torch.long, device=device).unsqueeze(0)
  chain_id = torch.tensor(batch["chain_id"], dtype=torch.long, device=device).unsqueeze(0)
  contacts = torch.tensor(batch["contacts"], dtype=torch.long, device=device).unsqueeze(0)
  mask = torch.tensor(batch["mask"], dtype=torch.bool, device=device).unsqueeze(0)

  def call() -> None:
    _pytorch_langevin_step(ref_model, r_noisy, aatype, residue_idx, mask, chain_id, contacts, t, dt)

  call()  # untimed cache-warming call (no JIT step in PyTorch).
  return _timed_calls(call, n_repeats)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _jax_device() -> str:
  return jax.devices()[0].platform


def _torch_device() -> str:
  import torch  # noqa: PLC0415

  return "cuda" if torch.cuda.is_available() else "cpu"


def _run_dry_run(args: argparse.Namespace) -> int:
  """L1 gate: imports + paths + one real (but trivial) end-to-end call, no timed loop.

  Per ``~/.claude/rules/CLUSTER.md``'s L1 definition: checks the checkpoint/
  reference-repo paths exist (skipped under --smoke), then builds the tiny
  --smoke-dim models (regardless of whether --smoke was also passed) and runs
  each of the four timing functions once with ``n_repeats=1`` on a trivially
  small (2 trajectories, 8 residues, 2 steps) input -- enough to prove every
  import/construction/dispatch path actually executes.
  """
  log.info("=== L1 dry-run: import + construction + one-shot execution check ===")
  import torch  # noqa: PLC0415, F401 -- import-availability check

  if not args.reference_repo.exists():
    log.error("[FAIL] reference repo not found: %s", args.reference_repo)
    return 1
  log.info("[PASS] reference repo path exists: %s", args.reference_repo)

  if not args.smoke:
    if not args.checkpoint.exists():
      log.error("[FAIL] checkpoint not found: %s", args.checkpoint)
      return 1
    log.info("[PASS] checkpoint path exists: %s", args.checkpoint)

  log.info("Constructing tiny synthetic-dim models (real classes, random init) to prove the wiring...")
  model_cfg, diffuser_cfg = _smoke_model_configs()
  jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None)

  batch = _synthetic_trajectories(n_trajectories=2, length=8, seed=args.seed)
  _run_jax_equilibration(
    jax_model, batch, args.diffusion_time, n_steps=2, dt=args.dt, seed=args.seed, n_repeats=1,
    safe_batch_size=DEFAULT_SAFE_TRAJECTORY_BATCH,
  )
  _run_jax_single_step(jax_model, batch, args.diffusion_time, args.dt, args.seed, n_repeats=1)
  _run_pytorch_equilibration(ref_model, batch, args.diffusion_time, n_steps=2, dt=args.dt, n_repeats=1)
  _run_pytorch_single_step(ref_model, batch, args.diffusion_time, args.dt, n_repeats=1)
  log.info("[PASS] JAX equilibration/single-step + PyTorch equilibration/single-step all executed once without error.")

  payload = {
    "dry_run": True,
    "note": "L1 gate only -- tiny synthetic dims, no real checkpoint/GPU timing performed.",
    "smoke_dims": {
      "token_s": SMOKE_TOKEN_S,
      "token_z": SMOKE_TOKEN_Z,
      "transformer_depth": SMOKE_DEPTH,
      "transformer_heads": SMOKE_HEADS,
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  log.info("[PASS] L1 dry-run OK. Wrote stub report to %s", args.out)
  return 0


def _run_cell(
  jax_model: ProteinEBMModel,
  ref_model: "ProteinEBM",
  length: int,
  n_trajectories: int,
  args: argparse.Namespace,
  n_steps: int,
  n_repeats: int,
  jax_device: str,
  torch_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Run one (length, n_trajectories) cell. Returns (jax_row, pytorch_row). Raises on failure.

  The JAX side's ``n_trajectories`` axis is dispatched via an xtrax
  ``AxisSpec``/``BatchPlanner`` decision (:func:`_jax_equilibration_batch`),
  not a plain ``jax.vmap`` -- see :data:`SAFE_TRAJECTORY_BATCH_BY_LENGTH` for
  why. ``jax_row`` records which strategy actually ran (``dispatch_strategy``,
  plus ``safe_map_chunk_size`` when chunked) so a reader can see when a number
  came from chunked ``SafeMap`` execution instead of a single ``Vmap`` call.
  """
  batch = _synthetic_trajectories(n_trajectories, length, args.seed)
  safe_batch_size = SAFE_TRAJECTORY_BATCH_BY_LENGTH.get(length, DEFAULT_SAFE_TRAJECTORY_BATCH)

  log.info(
    "[length=%d batch=%d] JAX warmup + timed batched equilibration (n_steps=%d)...",
    length, n_trajectories, n_steps,
  )
  jax_equil_times = _run_jax_equilibration(
    jax_model, batch, args.diffusion_time, n_steps, args.dt, args.seed, n_repeats, safe_batch_size,
  )
  jax_steps_per_sec = (n_trajectories * n_steps) / float(np.mean(jax_equil_times))
  jax_equil_ms_mean, jax_equil_ms_std = _wall_clock_ms_stats(jax_equil_times)

  jax_step_times = _run_jax_single_step(jax_model, batch, args.diffusion_time, args.dt, args.seed, n_repeats)
  jax_step_ms = float(np.mean(jax_step_times)) * 1000.0
  jax_step_ms_std = float(np.std(jax_step_times) * 1000.0)

  log.info("[length=%d batch=%d] PyTorch warmup + timed batched equilibration...", length, n_trajectories)
  pt_equil_times = _run_pytorch_equilibration(
    ref_model, batch, args.diffusion_time, n_steps, args.dt, n_repeats, device=torch_device,
  )
  pt_steps_per_sec = (n_trajectories * n_steps) / float(np.mean(pt_equil_times))
  pt_equil_ms_mean, pt_equil_ms_std = _wall_clock_ms_stats(pt_equil_times)

  pt_step_times = _run_pytorch_single_step(ref_model, batch, args.diffusion_time, args.dt, n_repeats, device=torch_device)
  pt_step_ms = float(np.mean(pt_step_times)) * 1000.0
  pt_step_ms_std = float(np.std(pt_step_times) * 1000.0)

  jax_row = {
    "protein_length": length,
    "batch_size": n_trajectories,
    "impl": "jax",
    "device": jax_device,
    "langevin_steps_per_sec": jax_steps_per_sec,
    "langevin_step_ms": jax_step_ms,
    "langevin_step_ms_std": jax_step_ms_std,
    "equilibration_wall_clock_mean_ms": jax_equil_ms_mean,
    "equilibration_wall_clock_std_ms": jax_equil_ms_std,
    "dispatch_strategy": "vmap" if n_trajectories <= safe_batch_size else "safe_map",
  }
  if n_trajectories > safe_batch_size:
    jax_row["safe_map_chunk_size"] = safe_batch_size
  pt_row = {
    "protein_length": length,
    "batch_size": n_trajectories,
    "impl": "pytorch",
    "device": torch_device,
    "langevin_steps_per_sec": pt_steps_per_sec,
    "langevin_step_ms": pt_step_ms,
    "langevin_step_ms_std": pt_step_ms_std,
    "equilibration_wall_clock_mean_ms": pt_equil_ms_mean,
    "equilibration_wall_clock_std_ms": pt_equil_ms_std,
  }
  return jax_row, pt_row


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if args.dry_run:
    return _run_dry_run(args)

  lengths, batch_sizes, n_steps, n_repeats = _resolve_run_params(args)
  log.info(
    "=== langevin_benchmark: lengths=%s batch_sizes=%s n_steps=%d n_repeats=%d smoke=%s ===",
    lengths, batch_sizes, n_steps, n_repeats, args.smoke,
  )

  jax_model, ref_model, _model_cfg, diffuser_cfg = build_models(args)
  jax_device = _jax_device()
  torch_device = _torch_device()
  log.info("JAX device: %s | PyTorch device: %s", jax_device, torch_device)

  results: list[dict[str, Any]] = []
  wall_start = time.perf_counter()
  for length in lengths:
    for n_trajectories in batch_sizes:
      try:
        jax_row, pt_row = _run_cell(
          jax_model, ref_model, length, n_trajectories, args, n_steps, n_repeats, jax_device, torch_device,
        )
      except Exception as e:  # noqa: BLE001 -- a single (length, batch) cell must not lose already-collected rows
        log.error("[length=%d batch=%d] FAILED: %s: %s", length, n_trajectories, type(e).__name__, e)
        results.append({
          "protein_length": length,
          "batch_size": n_trajectories,
          "impl": "error",
          "error": f"{type(e).__name__}: {e}",
        })
        _write_payload(args, batch_sizes, n_steps, n_repeats, diffuser_cfg, results, time.perf_counter() - wall_start)
        continue

      results.append(jax_row)
      results.append(pt_row)

      speedup = (
        jax_row["langevin_steps_per_sec"] / pt_row["langevin_steps_per_sec"] if pt_row["langevin_steps_per_sec"] else float("nan")
      )
      log.info(
        "[length=%d batch=%d strategy=%s] jax: %.1f steps/s, %.3f ms/step | pytorch: %.1f steps/s, %.3f ms/step | "
        "speedup=%.3fx",
        length, n_trajectories, jax_row["dispatch_strategy"],
        jax_row["langevin_steps_per_sec"], jax_row["langevin_step_ms"],
        pt_row["langevin_steps_per_sec"], pt_row["langevin_step_ms"], speedup,
      )
      _write_payload(args, batch_sizes, n_steps, n_repeats, diffuser_cfg, results, time.perf_counter() - wall_start)

  wall_elapsed = time.perf_counter() - wall_start
  _write_payload(args, batch_sizes, n_steps, n_repeats, diffuser_cfg, results, wall_elapsed)
  log.info("Wrote %d result rows to %s (wall clock %.1fs)", len(results), args.out, wall_elapsed)
  return 0


def _write_payload(
  args: argparse.Namespace,
  batch_sizes: tuple[int, ...],
  n_steps: int,
  n_repeats: int,
  diffuser_cfg: Any,
  results: list[dict[str, Any]],
  wall_elapsed: float,
) -> None:
  """Write accumulated results so far -- called after every (length, batch) cell so a crash mid-sweep loses only the in-flight cell, not everything already timed."""
  payload = {
    "meta": {
      "smoke": args.smoke,
      "batch_sizes": list(batch_sizes),
      "n_steps": n_steps,
      "n_repeats": n_repeats,
      "diffusion_time": args.diffusion_time,
      "dt": args.dt,
      "seed": args.seed,
      "checkpoint": None if args.smoke else str(args.checkpoint),
      "reference_repo": str(args.reference_repo),
      "coordinate_scaling": diffuser_cfg.coordinate_scaling,
      "effective_temp_scaling": DEFAULT_EFFECTIVE_TEMP_SCALING,
      "use_metropolis": False,
      "wall_clock_seconds": wall_elapsed,
      "methodology_notes": [
        "Fixed single noise level t, single model -- the outer noise-schedule "
        "loop and model-swap dispatcher (aminx.ebm.langevin_schedule) are out "
        "of scope for this script; see module docstring.",
        "langevin_steps_per_sec batches n_trajectories independent chains via "
        "an xtrax AxisSpec/BatchPlanner-dispatched Vmap or SafeMap (JAX; see "
        "dispatch_strategy) / a plain batch dimension (PyTorch), each run for "
        "a fixed n_steps -- not a per-trajectory-varying trip count.",
        "langevin_step_ms is a single-trajectory, single-step latency (one "
        "langevin_step call), not a batched-throughput number -- E11d's "
        "analog of E11a/E11b's score_grad_ms.",
        "Both impls get one untimed warmup call per (length, metric) before "
        "timing (JAX: forces jit compilation; PyTorch: cache/allocator warm-up).",
        "No SM120 XLA autotuning workaround applied here -- CPU-only local "
        "run. A cluster (GPU) run of this script MUST set "
        "XLA_FLAGS=--xla_gpu_shard_autotuning=false per ~/.claude/rules/CLUSTER.md "
        "or JAX throughput numbers will be wrong by orders of magnitude.",
        "Only the plain Euler-Maruyama langevin_step path is benchmarked "
        "(use_metropolis=False) -- metropolis_hastings_step is not exercised here.",
        "A row with impl='error' means that (protein_length, batch_size) cell "
        "raised during timing (see its 'error' field) -- all other cells in "
        "this file completed normally and are unaffected.",
        "jax rows' dispatch_strategy is 'vmap' (one shot) or 'safe_map' "
        "(jax.lax.map in chunks of safe_map_chunk_size, an xtrax BatchPlanner "
        "decision keyed on protein_length -- see SAFE_TRAJECTORY_BATCH_BY_LENGTH "
        "in the script) -- either way the full requested batch_size is what "
        "gets measured and reported; only the internal execution strategy "
        "changes, proactively avoiding the Blackwell/SM120 XLA-autotuning "
        "crash (.praxia/docs/audits/260716_proteinebm-parity-report.md §7) "
        "rather than catching and retrying it after the fact.",
      ],
    },
    "results": results,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
  raise SystemExit(main())
