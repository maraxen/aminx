"""Core user interface for the Aminx package."""

from __future__ import annotations

import logging
import warnings
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TextIO, cast

from aminx.model.versions import MODEL_VERSION, MODEL_WEIGHTS

from .spec import RunSpec, build_run_spec

_DEPRECATED_SPEC_KWARGS = frozenset(
  {
    "output_path",
    "score_batch_size",
    "average_logits",
    "combine_noise_batch_size",
    "gmm_min_iters",
    "average_encoding_mode",
  },
)


def pop_deprecated_spec_kwargs(kwargs: MutableMapping[str, Any]) -> None:
  """Remove legacy serialized keys dropped from specification dataclasses.

  Mutates ``kwargs`` in place. Emits :class:`DeprecationWarning` for each removed key.
  """
  for key in _DEPRECATED_SPEC_KWARGS:
    if key in kwargs:
      kwargs.pop(key)
      warnings.warn(
        f"Specification kwarg {key!r} is deprecated and ignored.",
        DeprecationWarning,
        stacklevel=3,
      )


if TYPE_CHECKING:
  from typing import Protocol

  from jaxtyping import ArrayLike
  from proxide.io.parsing.foldcomp import FoldCompDatabase

  from aminx.tiling.carry import CarrySpec
  from aminx.tiling.dedup import DedupSpec
  from aminx.types.stages import DecodingFusionFn, EncodingFusionFn
  from aminx.utils.catjac import CombineCatJacPairFn
  from aminx.utils.decoding_order import DecodingOrderFn

  class ConformationalStates(Protocol):
    """Protocol for conformational state containers (see ensemble_tools.dbscan)."""

    n_states: ArrayLike


# Type aliases for convenience
ModelWeights = MODEL_WEIGHTS
ModelVersion = MODEL_VERSION


AlignmentStrategy = Literal["sequence", "structure"]

# Type aliases for tied-position logit fusion configuration
TiedPositionMode = Literal["auto", "direct"] | None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureNoiseBundle:
  """Encapsulates noise schedule and mode for a single feature type (backbone/electrostatic/vdw)."""

  feature_type: Literal["backbone", "electrostatic", "vdw"]
  noise_levels: tuple[float, ...]
  mode: Literal["direct", "thermal"] = "direct"
  enabled: bool = True


def _loader_inputs(inputs: Sequence[str | TextIO] | str | TextIO) -> Sequence[str | TextIO]:
  out = (inputs,) if not isinstance(inputs, Sequence) else inputs
  return cast("Sequence[str | TextIO]", out)


def _extract_noise_levels(bundles: list[FeatureNoiseBundle], ftype: str) -> tuple[float, ...]:
  """Extract noise_levels from a bundle matching the given feature_type."""
  for b in bundles:
    if b.feature_type == ftype and b.enabled:
      return b.noise_levels
  # Return default (no noise) if bundle not found or not enabled
  return (0.0,) if ftype == "backbone" else ()


def _has_enabled(bundles: list[FeatureNoiseBundle], ftype: str) -> bool:
  """Check if any enabled bundle exists for the given feature_type."""
  return any(b.feature_type == ftype and b.enabled for b in bundles)


_WRAPPED_DEPRECATED_INIT: set[type] = set()


def register_spec(cls: type) -> type:
  """Decorator to wrap spec __init__ with deprecated kwarg warnings.

  Strips deprecated kwargs and emits DeprecationWarning for each removed key.
  Handles special migration: average_encoding_mode → encoding_aggregation_fn.
  Safe to apply multiple times; idempotent via _WRAPPED_DEPRECATED_INIT tracking.
  """
  if cls in _WRAPPED_DEPRECATED_INIT:
    return cls
  _WRAPPED_DEPRECATED_INIT.add(cls)
  original_init = cls.__init__

  def patched_init(self: object, *args: object, **kwargs: object) -> None:
    kwargs.pop("run_spec", None)

    # Special handling for average_encoding_mode → encoding_fusion migration
    if "average_encoding_mode" in kwargs:
      kwargs.pop("average_encoding_mode")
      warnings.warn(
        "Specification kwarg 'average_encoding_mode' is deprecated. "
        "Use 'encoding_fusion' (EncodingFusionFn) on RunSpecification instead.",
        DeprecationWarning,
        stacklevel=3,
      )

    for key in list(kwargs):
      if key in _DEPRECATED_SPEC_KWARGS:
        kwargs.pop(key)
        warnings.warn(
          f"Specification kwarg {key!r} is deprecated and ignored.",
          DeprecationWarning,
          stacklevel=3,
        )
    original_init(self, *args, **kwargs)

  setattr(cls, "__init__", patched_init)  # noqa: B010
  return cls


@register_spec
@dataclass
class RunSpecification:
  """Configuration for running the model.

  Attributes:
      inputs: A sequence of input file paths or text streams, or a single input.
      model_weights: The model weights to use (default is "original").
      model_version: The model version to use (default is "v_48_020.pkl").
      batch_size: The batch size to use (default is 32).
      backbone_noise: The backbone noise levels to use (default is (0.0,)).
                      Can be a single float or a sequence of floats.
      foldcomp_database: An optional path to a FoldComp database (default is None).
      ar_mask: An optional array-like mask for autoregressive positions (default is None).
      random_seed: The random seed to use (default is 42).
      chain_id: An optional chain ID to use (default is None).
      model: An optional model ID to use (default is None).
      altloc: The alternate location to use (default is "first").
      decoding_order_fn: An optional function to generate the decoding order (default is None).
      conformational_states: ConformationalStates (from ensemble_tools.dbscan) to use for coarse graining the inference.
      max_length: Maximum sequence length for padding/truncation (default is 512).
                  Set to None to disable padding. Controls memory usage - smaller values
                  reduce memory but may cause recompilation for different sequence lengths.
      truncation_strategy: Strategy for handling sequences longer than max_length.
                          Options: "none" (default, no truncation), "random_crop", "center_crop".
      host_resource_allocation_strategy: Strategy for host resource allocation (default is "auto").
      ram_budget_mb: Optional RAM budget for data loading in megabytes.
      max_workers: Optional maximum number of data loading workers.

  Note:
      Use ``output_h5_path`` on task-specific subclasses for HDF5 output.
      ``output_dir`` (when set) overrides inferred output roots from ``cache_path`` / HDF5 parents
      in :func:`~aminx.run.spec.build_run_spec`.
      Legacy serialized ``output_path`` is ignored with a :class:`DeprecationWarning`.

  """

  inputs: Sequence[str | TextIO] | str | TextIO
  topology: str | Path | None = None
  model_weights: ModelWeights = "original"
  model_version: ModelVersion = "v_48_020"
  model_family: Literal["proteinmpnn", "ligandmpnn"] = "proteinmpnn"
  checkpoint_id: str | None = None
  model_local_path: str | Path | None = None
  checkpoint_registry_path: str | Path | None = None
  ligand_mpnn_use_side_chain_context: bool | None = None
  batch_size: int = 32
  noise: list[FeatureNoiseBundle] = field(default_factory=list)
  encoding_fusion: EncodingFusionFn | None = None
  decoding_fusion: DecodingFusionFn | None = None
  # Deprecated old noise fields (will be removed in next major version)
  # These are kept for backward compatibility; prefer using 'noise' list
  backbone_noise: Sequence[float] | float | None = None
  backbone_noise_mode: Literal["direct", "thermal"] | None = None
  estat_noise: Sequence[float] | float | None = None
  estat_noise_mode: Literal["direct", "thermal"] | None = None
  vdw_noise: Sequence[float] | float | None = None
  vdw_noise_mode: Literal["direct", "thermal"] | None = None
  use_electrostatics: bool | None = None
  use_vdw: bool | None = None
  foldcomp_database: FoldCompDatabase | None = None
  ar_mask: None | ArrayLike = None
  random_seed: int = 42
  chain_id: Sequence[str] | str | None = None
  model: int | None = None
  altloc: Literal["first", "all"] = "first"
  decoding_order_fn: DecodingOrderFn | None = None
  conformational_states: ConformationalStates | None = None
  cache_path: str | Path | None = None
  output_dir: str | Path | None = None
  """Explicit host output root; preferred over ``cache_path`` / ``output_h5_path`` inference."""
  max_buffer_size: int | None = None
  """Optional dataset buffer cap (bytes); forwarded to proxide-style loaders when set."""
  overwrite_cache: bool = False
  max_length: int | None = 512
  truncation_strategy: Literal["none", "random_crop", "center_crop"] = "random_crop"

  # Host Resource Allocation
  host_resource_allocation_strategy: Literal["auto", "full"] = "auto"
  """Strategy for host resource allocation.

  - "auto": Use defaults based on context (90% RAM for training, 50% for inference)
  - "full": Use maximum available resources as detected by automatic inspection

  When manually specified values are provided (ram_budget_mb, max_workers),
  they take precedence over inferred defaults.
  """

  ram_budget_mb: int | None = None
  """RAM budget for data loading in megabytes.

  Priority order:
  1. If "full" strategy: Use maximum available RAM
  2. If explicitly set: Use this value
  3. If None with "auto": 90% of host RAM for training, 50% for inference
  """

  max_workers: int | None = None
  """Maximum number of data loading workers.

  Priority order:
  1. If "full" strategy: Use all CPU cores
  2. If explicitly set: Use this value
  3. If None with "auto": All cores for training, 50% for inference
  """

  n_devices: int | None = None
  """Optional JAX device count override for :class:`~aminx.run.spec.ResourceConfig`.

  When ``None``, :func:`build_run_spec` uses ``jax.local_device_count()`` when JAX is importable.
  """

  # Data/Sharding
  use_preprocessed: bool = False
  preprocessed_index_path: str | Path | None = None
  split: str = "inference"

  # Tied-position logit averaging fields
  tied_positions: Sequence[tuple[int, int]] | TiedPositionMode = None
  pass_mode: Literal["inter", "intra"] = "intra"  # noqa: S105
  tie_group_map: ArrayLike | None = None
  structure_mapping: ArrayLike | None = None
  multi_state_temperature: float = 1.0
  fixed_mask: ArrayLike | None = None
  sidechain_conditioning: bool = False

  run_spec: RunSpec = field(init=False, repr=False)
  _run_spec_synced: bool = field(init=False, default=False)

  def _sync_run_spec(self) -> None:
    if self._run_spec_synced:
      return
    object.__setattr__(self, "_run_spec_synced", True)
    object.__setattr__(self, "run_spec", build_run_spec(self))

  def __post_init__(self) -> None:
    """Post-initialization processing and validation for tied-position logit averaging."""
    # Ensure guard flag is initialized for first-time use
    if not hasattr(self, "_run_spec_synced"):
      object.__setattr__(self, "_run_spec_synced", False)

    # Handle backward compatibility: convert old-style noise params to noise list
    bundles = list(self.noise)  # Start with provided bundles

    # Backbone noise (always convert to bundle for consistency)
    if self.backbone_noise is not None or not bundles or all(b.feature_type != "backbone" for b in bundles):
      bb_noise: tuple[float, ...] = (0.0,)
      if self.backbone_noise is not None:
        if isinstance(self.backbone_noise, float):
          bb_noise = (self.backbone_noise,)
        elif isinstance(self.backbone_noise, Sequence):
          bb_noise = tuple(self.backbone_noise)  # type: ignore

      bb_mode: Literal["direct", "thermal"] = self.backbone_noise_mode or "direct"
      bundles = [b for b in bundles if b.feature_type != "backbone"]
      if bb_noise != (0.0,) or bb_mode != "direct":
        bundles.append(FeatureNoiseBundle(
          feature_type="backbone",
          noise_levels=bb_noise,
          mode=bb_mode,
          enabled=True,
        ))

    # Electrostatic noise
    if self.estat_noise is not None or self.use_electrostatics:
      bundles = [b for b in bundles if b.feature_type != "electrostatic"]
      if self.estat_noise is not None:
        estat_levels: tuple[float, ...] = (0.0,)
        if isinstance(self.estat_noise, float):
          estat_levels = (self.estat_noise,)
        elif isinstance(self.estat_noise, Sequence):
          estat_levels = tuple(self.estat_noise)  # type: ignore
        bundles.append(FeatureNoiseBundle(
          feature_type="electrostatic",
          noise_levels=estat_levels,
          mode=self.estat_noise_mode or "direct",
          enabled=True,
        ))
      elif self.use_electrostatics:
        bundles.append(FeatureNoiseBundle(
          feature_type="electrostatic",
          noise_levels=(0.0,),
          mode=self.estat_noise_mode or "direct",
          enabled=True,
        ))

    # VDW noise
    if self.vdw_noise is not None or self.use_vdw:
      bundles = [b for b in bundles if b.feature_type != "vdw"]
      if self.vdw_noise is not None:
        vdw_levels: tuple[float, ...] = (0.0,)
        if isinstance(self.vdw_noise, float):
          vdw_levels = (self.vdw_noise,)
        elif isinstance(self.vdw_noise, Sequence):
          vdw_levels = tuple(self.vdw_noise)  # type: ignore
        bundles.append(FeatureNoiseBundle(
          feature_type="vdw",
          noise_levels=vdw_levels,
          mode=self.vdw_noise_mode or "direct",
          enabled=True,
        ))
      elif self.use_vdw:
        bundles.append(FeatureNoiseBundle(
          feature_type="vdw",
          noise_levels=(0.0,),
          mode=self.vdw_noise_mode or "direct",
          enabled=True,
        ))

    # Update self.noise with merged bundles
    object.__setattr__(self, "noise", bundles)

    # Set backward-compat attributes from noise list
    object.__setattr__(self, "backbone_noise", _extract_noise_levels(bundles, "backbone"))
    object.__setattr__(self, "use_electrostatics", _has_enabled(bundles, "electrostatic"))
    object.__setattr__(self, "use_vdw", _has_enabled(bundles, "vdw"))

    # Also convert deprecated field attributes to tuples for backward compat
    if isinstance(self.backbone_noise, float):
      object.__setattr__(self, "backbone_noise", (self.backbone_noise,))
    if self.estat_noise is not None:
      if isinstance(self.estat_noise, float):
        object.__setattr__(self, "estat_noise", (self.estat_noise,))
      elif not isinstance(self.estat_noise, tuple) and isinstance(self.estat_noise, Sequence):
        object.__setattr__(self, "estat_noise", tuple(self.estat_noise))
    if self.vdw_noise is not None:
      if isinstance(self.vdw_noise, float):
        object.__setattr__(self, "vdw_noise", (self.vdw_noise,))
      elif not isinstance(self.vdw_noise, tuple) and isinstance(self.vdw_noise, Sequence):
        object.__setattr__(self, "vdw_noise", tuple(self.vdw_noise))

    if self.cache_path and isinstance(self.cache_path, str):
      object.__setattr__(self, "cache_path", Path(self.cache_path))
    if self.output_dir and isinstance(self.output_dir, str):
      object.__setattr__(self, "output_dir", Path(self.output_dir))
    if self.model_local_path and isinstance(self.model_local_path, str):
      object.__setattr__(self, "model_local_path", Path(self.model_local_path))
    if self.checkpoint_registry_path and isinstance(self.checkpoint_registry_path, str):
      object.__setattr__(self, "checkpoint_registry_path", Path(self.checkpoint_registry_path))
    # Validation for tied-position logit averaging
    if self.tied_positions in ("auto", "direct") and self.pass_mode != "inter":  # noqa: S105
      msg = (
        f"If tied_positions is '{self.tied_positions}', pass_mode must be 'inter'. "
        f"Got pass_mode='{self.pass_mode}'."
      )
      raise ValueError(msg)
    self._sync_run_spec()


@register_spec
@dataclass
class ScoringSpecification(RunSpecification):
  """Configuration for scoring sequences.

  Attributes:
      sequences_to_score: A sequence of amino acid sequences to score.
      temperature: The temperature for scoring (default is 1.0).
      return_logits: Whether to return the raw logits (default is False).
      return_decoding_orders: Whether to return decoding orders (default is False).
      return_all_scores: Whether to return scores for all sequences (default is False).
      output_h5_path: Optional path to an HDF5 file for streaming output.
      average_node_features: Whether to average node features (default is False).
      average_encoding_mode: Mode for averaging encodings (default is "inputs_and_noise").
      noise_batch_size: The batch size for noise levels (default is 4).

  """

  sequences_to_score: Sequence[str] = ()
  temperature: float = 1.0
  return_logits: bool = False
  return_decoding_orders: bool = False
  return_all_scores: bool = False
  output_h5_path: str | Path | None = None
  average_node_features: bool = False
  noise_batch_size: int = 4
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean"

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    object.__setattr__(self, "_run_spec_synced", False)
    super().__post_init__()
    if not self.sequences_to_score:
      msg = (
        "No sequences provided for scoring."
        "`sequences_to_score` must be a non-empty list of strings."
      )
      raise ValueError(msg)
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))
    self._sync_run_spec()


@register_spec
@dataclass
class SamplingSpecification(RunSpecification):
  """Configuration for sampling sequences."""

  num_samples: int = 1
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature"
  temperature: Sequence[float] | float = 0.1
  use_unified_driver: bool = True
  bias: ArrayLike | None = None
  fixed_positions: ArrayLike | None = None
  fixed_tokens: ArrayLike | None = None
  iterations: int | None = None
  learning_rate: float | None = None
  use_concrete: bool = False
  concrete_tau_start: float = 1.0
  concrete_tau_end: float = 0.1
  output_h5_path: str | Path | None = None
  use_arrayrecord: bool = False
  return_logits: bool = True
  samples_batch_size: int = 16
  samples_chunk_size: int | None = None
  noise_batch_size: int = 1
  temperature_batch_size: int = 1
  average_node_features: bool = False
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean"
  compute_pseudo_perplexity: bool = False
  state_weights: ArrayLike | None = None
  ligand_conditioning: bool = False
  campaign_mode: bool = False
  allow_logits_in_campaign: bool = False
  logits_memory_budget_mb: int | None = None
  ligand_context_path: str | Path | None = None
  grid_mode: bool = False
  job_id: str | None = None
  chunk_id: int | None = None
  sample_start: int | None = None
  sample_count: int | None = None
  carry_specs: list[CarrySpec] | None = None
  dedup_specs: list[DedupSpec] | None = None
  chain_mask_fixed: ArrayLike | None = None

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    object.__setattr__(self, "_run_spec_synced", False)
    super().__post_init__()
    if isinstance(self.temperature, float):
      object.__setattr__(self, "temperature", (self.temperature,))
    if self.sampling_strategy == "straight_through" and (
      self.iterations is None or self.learning_rate is None
    ):
      msg = "For 'straight_through' sampling, 'iterations' and 'learning_rate' must be provided."
      raise ValueError(msg)
    if self.grid_mode and self.sampling_strategy != "temperature":
      msg = "Grid mode only supports 'temperature' sampling."
      raise ValueError(msg)
    if self.grid_mode and self.average_node_features:
      msg = "Grid mode does not support average_node_features=True."
      raise ValueError(msg)
    if not self.return_logits and self.compute_pseudo_perplexity:
      msg = "compute_pseudo_perplexity requires return_logits=True."
      raise ValueError(msg)
    if self.logits_memory_budget_mb is not None:
      logits_budget = int(self.logits_memory_budget_mb)
      if logits_budget <= 0:
        msg = "logits_memory_budget_mb must be positive when provided."
        raise ValueError(msg)
      object.__setattr__(self, "logits_memory_budget_mb", logits_budget)
    if self.campaign_mode and self.return_logits:
      if not self.allow_logits_in_campaign:
        msg = "campaign_mode requires return_logits=False unless allow_logits_in_campaign=True."
        raise ValueError(msg)
      if self.logits_memory_budget_mb is None:
        msg = (
          "campaign_mode with return_logits=True requires logits_memory_budget_mb "
          "to be explicitly set."
        )
        raise ValueError(msg)
    if self.job_id is not None:
      normalized_job_id = str(self.job_id).strip()
      if self.grid_mode and not normalized_job_id:
        msg = "job_id cannot be empty when grid_mode=True."
        raise ValueError(msg)
      object.__setattr__(self, "job_id", normalized_job_id or None)
    if self.chunk_id is not None:
      chunk_id = int(self.chunk_id)
      if self.grid_mode and chunk_id < 0:
        msg = "chunk_id must be non-negative when grid_mode=True."
        raise ValueError(msg)
      object.__setattr__(self, "chunk_id", chunk_id)
    if self.sample_start is not None:
      sample_start = int(self.sample_start)
      if self.grid_mode and sample_start < 0:
        msg = "sample_start must be non-negative when grid_mode=True."
        raise ValueError(msg)
      object.__setattr__(self, "sample_start", sample_start)
    if self.sample_count is not None:
      sample_count = int(self.sample_count)
      if self.grid_mode and sample_count <= 0:
        msg = "sample_count must be positive when grid_mode=True."
        raise ValueError(msg)
      object.__setattr__(self, "sample_count", sample_count)
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))
    if self.ligand_context_path and isinstance(self.ligand_context_path, str):
      object.__setattr__(self, "ligand_context_path", Path(self.ligand_context_path))
    self._sync_run_spec()


@register_spec
@dataclass
class JacobianSpecification(RunSpecification):
  """Configuration for computing categorical Jacobians."""

  noise_batch_size: int = 1
  jacobian_batch_size: int = 16
  average_encodings: bool = True
  combine: bool = False
  combine_batch_size: int = 8
  combine_weights: ArrayLike | None = None
  combine_fn: CombineCatJacPairFn | None = None
  combine_fn_kwargs: dict[str, Any] | None = None
  output_h5_path: str | Path | None = None
  compute_apc: bool = True
  apc_batch_size: int = 8
  apc_residue_batch_size: int = 1000
  jacobian_mode: Literal["categorical", "reverse"] = "categorical"

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    object.__setattr__(self, "_run_spec_synced", False)
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))
    self._sync_run_spec()




MIN_PAIR = 2


@register_spec
@dataclass
class InspectionSpecification(RunSpecification):
  """Configuration for inspecting model encodings and features."""

  output_h5_path: str | Path | None = None
  inspection_features: Sequence[
    Literal[
      "unconditional_logits",
      "encoded_node_features",
      "edge_features",
      "decoded_node_features",
      "conditional_logits",
    ]
  ] = ("unconditional_logits",)
  distance_matrix: bool = False
  distance_matrix_method: Literal["ca", "cb", "backbone_average", "closest_atom"] = "ca"
  cross_input_similarity: bool = False
  similarity_metric: Literal[
    "rmsd",
    "tm-score",
    "gdt_ts",
    "gdt_ha",
    "cosine",
  ] = "rmsd"

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    object.__setattr__(self, "_run_spec_synced", False)
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))
    if self.cross_input_similarity and len(_loader_inputs(self.inputs)) < MIN_PAIR:
      msg = f"Cross-input similarity requires at least {MIN_PAIR} input structures."
      raise ValueError(msg)
    self._sync_run_spec()


Specs = (
  RunSpecification
  | ScoringSpecification
  | SamplingSpecification
  | JacobianSpecification
  | InspectionSpecification
)


def apply_deprecated_spec_init_warnings(cls: type) -> None:
  """Wrap ``cls.__init__`` like task specs: strip deprecated kwargs and warn.

  Call once per extra subclass (e.g. :class:`~aminx.training.specs.TrainingSpecification`)
  defined outside this module.
  """
  register_spec(cls)
