"""Composed run configuration (`RunSpec`) and sub-config modules.

See `.agents/REFACTOR_ROADMAP.md` §3.5. This module is the Equinox/pytree representation;
`run.specs.RunSpecification` remains the public dataclass façade for one minor version.

PlannerTopology (added in RS-2) configures the amino-acid kernel dispatch topology for the
planner/sampler. Use `topology_hash()` to derive cache keys from PlannerTopology fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import field, fields
from os import PathLike, fspath
from pathlib import Path
from typing import Any, Literal, cast

import equinox as eqx
import jax
from xtrax.run import RunSpec as _XtraxRunSpec

from aminx.types.stages import DecodingFusionFn, EncodingFusionFn


class IOConfig(eqx.Module):
  """Output / manifest / sink selection (host-side)."""

  output_dir: Path | None = eqx.field(static=True)
  sink_kind: str = eqx.field(static=True)
  manifest_path: Path | None = eqx.field(static=True)
  output_h5_path: Path | None = eqx.field(static=True)
  cache_path: Path | None = eqx.field(static=True)


class ResourceConfig(eqx.Module):
  """Device and batching caps (structure vs sample streams)."""

  n_devices: int = eqx.field(static=True)
  sample_batch_size: int = eqx.field(static=True)
  structure_batch_size: int = eqx.field(static=True)
  max_buffer_size: int | None = eqx.field(static=True)


class MultistateConfig(eqx.Module):
  """Multistate / combine-strategy knobs (registry keys are strings, not int indices)."""

  mode: str = eqx.field(static=True)
  n_states: int = eqx.field(static=True)
  combine_strategy: str = eqx.field(static=True)


class LigandConfig(eqx.Module):
  """LigandMPNN-specific options layered on `RunSpecification`."""

  model_family: str = eqx.field(static=True)
  use_side_chain_context: bool | None = eqx.field(static=True)
  ligand_conditioning: bool | None = eqx.field(static=True)
  sidechain_conditioning: bool = eqx.field(static=True)
  context_path: Path | None = eqx.field(static=True)


class GridLineageConfig(eqx.Module):
  """Grid / campaign lineage metadata (mostly static host fields)."""

  grid_mode: bool = eqx.field(static=True)
  campaign_mode: bool = eqx.field(static=True)
  job_id: str | None = eqx.field(static=True)
  chunk_id: int | None = eqx.field(static=True)
  sample_start: int | None = eqx.field(static=True)
  sample_count: int | None = eqx.field(static=True)


class PrecisionConfig(eqx.Module):
  """JAX compute dtype policy for training and inference (mirrors dataclass ``precision`` when set)."""

  compute: Literal["fp32", "fp16", "bf16"] = eqx.field(static=True)


class PlannerTopology(eqx.Module):
  """aminx kernel dispatch topology config.

  Configures planner and sampler dispatch topology knobs for RunSpec. Wraps
  aminx.tiling.BatchPlanner config fields. Will gain xtrax.ExecutionProfile once
  xtrax BatchPlanner reaches multi-phase parity (T2.5).

  The `use_unified_driver` field selects between the unified plan-driven sampling
  dispatch (True) and the legacy fallback sampling dispatch (False). Default is True.

  See `topology_hash()` below for deterministic cache-key derivation from these fields.
  """

  use_unified_driver: bool = eqx.field(static=True)
  """Select unified plan-driven dispatch (True) or legacy fallback (False). Default: True."""


class SamplingConfig(eqx.Module):
  """Generic sampling knobs (RS-1 migration map section 4).

  ``sampling_strategy`` through ``structure_mapping`` below were added 260716 (EPIC #1541 P4,
  the decode-plan-unification pass) so that decode-class resolution (``ConditionalMode`` vs
  ``AutoregressiveMode`` vs ``STEMode``) and multistate/tie fusion can be driven from ONE place
  (``spec.run_spec.sampling``) instead of scattered flat-spec reads across ``host/plan.py`` and
  ``sampling/multistate_poe.py`` independently. See aminx#110/#113 and
  ``.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md`` for why the
  prior attempt at this (``TiedPositionsConfig``/``AveragingConfig``/``BatchingConfig``) went
  100% dead and was deleted (`dd0e952`) rather than fixed -- fields live directly on
  ``SamplingConfig`` this time, not a separate sub-config, since that's the one that stuck.
  """

  num_samples: int = eqx.field(static=True)
  random_seed: int = eqx.field(static=True)
  return_logits: bool = eqx.field(static=True)
  compute_pseudo_perplexity: bool = eqx.field(static=True)
  return_decoding_orders: bool = eqx.field(static=True)
  return_logit_fingerprint: bool = eqx.field(static=True)
  backbone_noise: tuple[float, ...] = eqx.field(static=True)
  temperature: tuple[float, ...] = eqx.field(static=True)
  bias: Any = None
  fixed_mask: Any = None
  fixed_positions: Any = None
  fixed_tokens: Any = None
  sampling_strategy: str = eqx.field(static=True, default="temperature")
  multi_state_strategy: str = eqx.field(static=True, default="arithmetic_mean")
  multi_state_temperature: float = eqx.field(static=True, default=1.0)
  # Logit scale for the "product" strategy. Separate from state_weights on
  # purpose: weights are mixing proportions, sharpness is concentration. None
  # means "match a plain product of S experts". See ProductOfProbabilities.
  multi_state_sharpness: float | None = eqx.field(static=True, default=1.0)
  use_rolling_state: bool = eqx.field(static=True, default=False)
  decoding_order_fn: Any = eqx.field(static=True, default=None)
  tie_group_map: Any = None
  state_position_map: Any = None
  state_weights: Any = None
  structure_mapping: Any = None


class RunSpec(_XtraxRunSpec):
  """Composed configuration for run/prep pipelines."""

  io: IOConfig = field(default_factory=lambda: None)  # type: ignore
  resource: ResourceConfig = field(default_factory=lambda: None)  # type: ignore
  multistate: MultistateConfig = field(default_factory=lambda: None)  # type: ignore
  ligand: LigandConfig = field(default_factory=lambda: None)  # type: ignore
  grid: GridLineageConfig = field(default_factory=lambda: None)  # type: ignore
  precision: PrecisionConfig = field(default_factory=lambda: None)  # type: ignore
  plan: PlannerTopology = field(default_factory=lambda: None)  # type: ignore
  sampling: SamplingConfig = field(default_factory=lambda: None)  # type: ignore
  encoding_fusion: EncodingFusionFn | None = eqx.field(static=True, default=None)
  decoding_fusion: DecodingFusionFn | None = eqx.field(static=True, default=None)


def _as_float_tuple(v: object | None) -> tuple[float, ...]:
  """Convert scalar or iterable (or None) to tuple of floats."""
  if v is None:
    return ()
  # Handle scalar float/int case
  if isinstance(v, (int, float)):
    return (float(v),)
  # Handle iterable case (guaranteed by elimination)
  return tuple(float(x) for x in cast("Any", v))


def _optional_bool(value: object | None) -> bool | None:
  """``bool(value)`` that keeps ``None`` as ``None`` (a tri-state flag's "unset")."""
  return None if value is None else bool(value)


def _optional_path(value: object | None) -> Path | None:
  if value is None:
    return None
  if isinstance(value, Path):
    return value
  if isinstance(value, str):
    return Path(value)
  return Path(fspath(cast("PathLike[str]", value)))


def _output_h5_path(spec: object) -> Path | None:
  raw = getattr(spec, "output_h5_path", None)
  return _optional_path(raw)


def _infer_output_dir(spec: object) -> Path | None:
  explicit = getattr(spec, "output_dir", None)
  if explicit is not None:
    return _optional_path(explicit)
  h5 = _output_h5_path(spec)
  if h5 is not None:
    return h5.parent
  cache = getattr(spec, "cache_path", None)
  if cache is not None:
    p = _optional_path(cache)
    return p.parent if p is not None and p.suffix else p
  # TODO(REFACTOR_ROADMAP §3.5): unify with explicit `output_dir` once call-sites migrate.
  return None


def _run_spec_precision_compute(spec: object) -> Literal["fp32", "fp16", "bf16"]:
  """Resolve compute precision: training specs use ``precision``; otherwise default fp32."""
  raw = getattr(spec, "precision", "fp32")
  if raw in ("fp32", "fp16", "bf16"):
    return cast("Literal['fp32', 'fp16', 'bf16']", raw)
  return "fp32"


def _coerce_max_buffer_size(spec: object) -> int | None:
  """Optional positive buffer cap (bytes); ``None`` when unset or invalid."""
  raw = getattr(spec, "max_buffer_size", None)
  if raw is None:
    return None
  try:
    v = int(raw.item()) if hasattr(raw, "item") else int(raw)
  except (TypeError, ValueError):
    return None
  return v if v > 0 else None


def _infer_sink_kind(spec: object) -> str:
  if _output_h5_path(spec) is not None:
    return "zarr"
  # IOConfig.sink_kind uses "zarr" | "none"; host tensor sinks register under OUTPUT_SINKS
  # (e.g. "noop", "streaming_tensor_staging" — see aminx.run.output_sinks).
  return "none"


def _coerce_n_states(spec: object) -> int:
  cs = getattr(spec, "conformational_states", None)
  if cs is None:
    return 1
  ns = getattr(cs, "n_states", None)
  if ns is None:
    return 1
  try:
    if hasattr(ns, "item"):
      return int(ns.item())
    return int(ns)
  except (TypeError, ValueError):
    # TODO(REFACTOR_ROADMAP §3.5): stable int from `ProteinBundle` / stack shape.
    return 1


def _infer_multistate_mode(spec: object) -> str:
  # TODO(REFACTOR_ROADMAP §3.3 `MULTISTATE_MODES`): replace placeholder with registry key
  # derived from prep/model capabilities once `multistate_mode` leaves ad-hoc if-ladders.
  if getattr(spec, "conformational_states", None) is not None:
    return "conformational_context"
  if _coerce_n_states(spec) > 1:
    return "multi_state"
  return "single"


def _infer_n_devices(spec: object) -> int:
  """Prefer explicit ``n_devices`` on the spec; else ``jax.local_device_count()``; else ``1``."""
  raw = getattr(spec, "n_devices", None)
  if raw is not None:
    try:
      if hasattr(raw, "item"):
        return int(raw.item())
      return int(raw)
    except (TypeError, ValueError):
      pass
  try:
    return int(jax.local_device_count())
  except (RuntimeError, OSError, AttributeError, ImportError, ValueError):
    return 1


def topology_hash(plan: PlannerTopology) -> str:
  """Deterministic 16-char hex hash of PlannerTopology for cache-key derivation.

  Hashes PlannerTopology fields to produce a stable cache key for static-field identity.
  Used to detect recompile requirements when topology changes between runs.

  The payload dict is derived reflectively from PlannerTopology fields via dataclasses.fields(),
  so new/removed fields are picked up automatically without manual updates.
  """
  payload = {f.name: getattr(plan, f.name) for f in fields(plan)}
  hash_input = json.dumps(payload, sort_keys=True, default=str).encode()
  return hashlib.sha256(hash_input).hexdigest()[:16]


def _coerce_sharpness(value: object) -> float | None:
  """Normalise a user-supplied ``multi_state_sharpness`` to ``float | None``.

  ``None`` is a meaningful value here -- "scale by S, matching a plain product of
  S experts" -- so it must survive rather than being coalesced to the 1.0 default.
  Anything else is coerced to float, which surfaces a bad type as a TypeError at
  spec-build time instead of silently reverting to unscaled fusion deep inside a
  jitted decode.
  """
  if value is None:
    return None
  return float(value)


def build_run_spec(spec: object) -> RunSpec:
  """Build a :class:`RunSpec` view from a :class:`RunSpecification` (or subclass) instance."""
  pre_idx = getattr(spec, "preprocessed_index_path", None)
  io = IOConfig(
    output_dir=_infer_output_dir(spec),
    sink_kind=_infer_sink_kind(spec),
    manifest_path=_optional_path(pre_idx),
    output_h5_path=_output_h5_path(spec),
    cache_path=_optional_path(getattr(spec, "cache_path", None)),
  )

  batch_size = int(getattr(spec, "batch_size", 32))
  samples_batch = getattr(spec, "samples_batch_size", None)
  resource = ResourceConfig(
    n_devices=_infer_n_devices(spec),
    sample_batch_size=int(samples_batch) if samples_batch is not None else batch_size,
    structure_batch_size=batch_size,
    max_buffer_size=_coerce_max_buffer_size(spec),
  )

  combine = getattr(spec, "multi_state_strategy", "arithmetic_mean")
  multistate = MultistateConfig(
    mode=_infer_multistate_mode(spec),
    n_states=_coerce_n_states(spec),
    combine_strategy=str(combine),
  )

  ctx_path = getattr(spec, "ligand_context_path", None)
  ligand = LigandConfig(
    model_family=str(getattr(spec, "model_family", "proteinmpnn")),
    use_side_chain_context=getattr(spec, "ligand_mpnn_use_side_chain_context", None),
    # Tri-state passes through unchanged: None ("use whatever is present") must not be
    # collapsed to False ("ablate"). ScoringSpecification's field is a plain bool.
    ligand_conditioning=_optional_bool(getattr(spec, "ligand_conditioning", None)),
    sidechain_conditioning=bool(getattr(spec, "sidechain_conditioning", False)),
    context_path=_optional_path(ctx_path),
  )

  grid = GridLineageConfig(
    grid_mode=bool(getattr(spec, "grid_mode", False)),
    campaign_mode=bool(getattr(spec, "campaign_mode", False)),
    job_id=getattr(spec, "job_id", None),
    chunk_id=getattr(spec, "chunk_id", None),
    sample_start=getattr(spec, "sample_start", None),
    sample_count=getattr(spec, "sample_count", None),
  )

  precision = PrecisionConfig(compute=_run_spec_precision_compute(spec))

  plan = PlannerTopology(
    use_unified_driver=bool(getattr(spec, "use_unified_driver", True)),
  )

  sampling = SamplingConfig(
    num_samples=int(getattr(spec, "num_samples", 1) or 1),
    random_seed=int(getattr(spec, "random_seed", 42)),
    return_logits=bool(getattr(spec, "return_logits", False)),
    compute_pseudo_perplexity=bool(getattr(spec, "compute_pseudo_perplexity", False)),
    return_decoding_orders=bool(getattr(spec, "return_decoding_orders", False)),
    return_logit_fingerprint=bool(getattr(spec, "return_logit_fingerprint", False)),
    backbone_noise=_as_float_tuple(getattr(spec, "backbone_noise", None)),
    temperature=_as_float_tuple(getattr(spec, "temperature", None)),
    bias=getattr(spec, "bias", None),
    fixed_mask=getattr(spec, "fixed_mask", None),
    fixed_positions=getattr(spec, "fixed_positions", None),
    fixed_tokens=getattr(spec, "fixed_tokens", None),
    sampling_strategy=str(getattr(spec, "sampling_strategy", "temperature") or "temperature"),
    multi_state_strategy=str(
      getattr(spec, "multi_state_strategy", "arithmetic_mean") or "arithmetic_mean",
    ),
    multi_state_temperature=float(getattr(spec, "multi_state_temperature", 1.0) or 1.0),
    # `or`-coalescing is WRONG here: sharpness=None is a meaningful value ("use S"),
    # not a missing one, and 0.0 would be a caller error worth surfacing rather than
    # silently rewriting to 1.0. Use the sentinel-free getattr default instead.
    multi_state_sharpness=_coerce_sharpness(getattr(spec, "multi_state_sharpness", 1.0)),
    use_rolling_state=bool(getattr(spec, "use_rolling_state", False)),
    decoding_order_fn=getattr(spec, "decoding_order_fn", None),
    tie_group_map=getattr(spec, "tie_group_map", None),
    state_position_map=getattr(spec, "state_position_map", None),
    state_weights=getattr(spec, "state_weights", None),
    structure_mapping=getattr(spec, "structure_mapping", None),
  )

  return RunSpec(
    seed=getattr(spec, "seed", 0),
    axes=getattr(spec, "axes", []),
    carry_specs=getattr(spec, "carry_specs", []),
    boundaries=getattr(spec, "boundaries", None),
    io=io,
    resource=resource,
    multistate=multistate,
    ligand=ligand,
    grid=grid,
    precision=precision,
    plan=plan,
    sampling=sampling,
  )
