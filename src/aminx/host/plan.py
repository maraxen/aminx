"""Batch planning and scheduling logic for sampling operations.

Also includes InferencePlan and related components for unified inference dispatch.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisDecision, BatchPlan, BatchPlanner, ChunkedMap, MemoryBudget
from xtrax.tiling import BudgetInfeasibleError as _XtraxBudgetInfeasibleError

from aminx.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from aminx.tiling.errors import TilingError
from aminx.tiling.planner import estimate_memory_theoretical

if TYPE_CHECKING:
  from collections.abc import Sequence

  from jaxtyping import PRNGKeyArray
  from xtrax.tiling import AxisSpec, CarrySpec, DedupSpec

  from aminx.inference.decode.protocols import ARDecodeFn, DecodeScoreFn, STEDecodeFn
  from aminx.run.specs import SamplingSpecification
  from aminx.tiling.bucketing import BucketAssignment
  from aminx.types.arrays import Logits
  from aminx.types.bundles import InferenceBundle
  from aminx.types.configs import InferenceConfig
  from aminx.types.encodings import EncoderOutput
  from aminx.types.protocols import ModelProtocol

logger = logging.getLogger(__name__)
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


class PlanTopologyError(TilingError):
  """Raised at make_inference_plan() time when plan topology is invalid.

  This fires before any JAX compilation — topology errors are caught at
  plan construction time, not at trace time or runtime.
  """


class PlanBudgetInfeasibleError(TilingError):
  """Raised when no demotion sequence brings a sampling plan under its memory budget.

  Translates xtrax.tiling.BudgetInfeasibleError (a plain Exception) into
  aminx's own TilingError hierarchy at the make_sampling_planner boundary,
  the same pattern host/plan.py already uses for PlanTopologyError /
  xtrax.stages.PlanTopologyError and tiling/dispatch.py uses for
  DispatchRejected / xtrax's DispatchRejected.
  """


# Axes with variable element shapes across the batch: jax.vmap is structurally
# invalid on these (ragged/variable shapes), so they must always use SafeMap.
# Passed to xtrax.tiling.BatchPlanner as heterogeneous_axes (rejects Scan on
# these in Phase 0) AND used by _plan_with_joint_budget below to pre-fix these
# axes to SafeMap before calling xtrax's engine -- xtrax's joint-budget mode
# has no equivalent of aminx's old Phase 1 (force non-carry/dedup heterogeneous
# axes to SafeMap unconditionally, before the budget loop runs); confirmed
# empirically (2026-07-06) that without this fix, xtrax's engine assigns Vmap
# to a heterogeneous axis whenever the joint estimate already fits budget.
_HETEROGENEOUS_AXIS_NAMES = frozenset({"n_states", "n_structures"})


def _plan_with_joint_budget(
  axes: Sequence[AxisSpec],
  *,
  budget_bytes: int,
  estimate_fn: Callable[[Sequence[AxisDecision]], float],
  carry_specs: list[CarrySpec] | None = None,
  dedup_specs: list[DedupSpec] | None = None,
) -> BatchPlan:
  """Plan axes under a joint memory budget via xtrax.tiling.BatchPlanner.

  Thin wrapper around xtrax's joint-budget mode (EPIC #1541 T-PLANNER.2) that
  additionally pre-fixes heterogeneous axes to SafeMap before delegating —
  see _HETEROGENEOUS_AXIS_NAMES above for why this is necessary. The fixed
  decisions are folded into the estimator closure (so xtrax's greedy
  demotion loop sees the correct joint memory estimate at every step) and
  merged back into the final plan, in the original axes order, after
  xtrax's engine resolves the remaining (non-heterogeneous, non-fixed) axes.

  Raises
  ------
  PlanBudgetInfeasibleError
      If no demotion sequence for the remaining axes fits budget_bytes.

  """
  carry_specs = carry_specs or []
  dedup_specs = dedup_specs or []
  carry_names = {cs.axis_name for cs in carry_specs}
  dedup_names = {ds.axis_name for ds in dedup_specs}

  fixed_heterogeneous: dict[str, AxisDecision] = {}
  remaining_axes: list[AxisSpec] = []
  for ax in axes:
    if ax.heterogeneous and ax.name not in carry_names and ax.name not in dedup_names:
      fixed_heterogeneous[ax.name] = AxisDecision(
        spec=ax,
        batch_size=ax.default_batch_size,
        reasoning=(
          "heterogeneous axis: element shapes vary; safe_map required "
          "(aminx wrapper, EPIC #1541 T-PLANNER.2 -- xtrax's joint-budget "
          "mode has no automatic heterogeneous-axis guard)"
        ),
        strategy=ChunkedMap(batch_size=ax.default_batch_size),
      )
    else:
      remaining_axes.append(ax)

  def _estimate(pending_decisions: Sequence[AxisDecision]) -> int:
    full = list(fixed_heterogeneous.values()) + list(pending_decisions)
    return int(estimate_fn(full))

  budget = MemoryBudget(bytes=budget_bytes, estimate=_estimate)
  planner = BatchPlanner(
    budget=budget,
    carry_specs=carry_specs,
    dedup_specs=dedup_specs,
    heterogeneous_axes=_HETEROGENEOUS_AXIS_NAMES,
  )
  try:
    sub_plan = planner.plan(remaining_axes)
  except _XtraxBudgetInfeasibleError as exc:
    raise PlanBudgetInfeasibleError(str(exc)) from exc

  decision_by_name = {d.spec.name: d for d in sub_plan.decisions}
  decision_by_name.update(fixed_heterogeneous)
  ordered_decisions = tuple(decision_by_name[ax.name] for ax in axes)
  return BatchPlan(decisions=ordered_decisions)


def decision_for(plan: BatchPlan, name: str) -> AxisDecision:
  """Look up the AxisDecision for a named axis in a BatchPlan.

  xtrax.tiling.BatchPlan (unlike aminx's retired local BatchPlan) has no
  .decision_for() convenience method -- this is the direct replacement,
  used everywhere host/kernel_dispatch.py used to call plan.decision_for(name).

  Raises
  ------
  KeyError
      If no decision matches ``name``.

  """
  for d in plan.decisions:
    if d.spec.name == name:
      return d
  raise KeyError(name)


# Axis name constants for batch planning
class AxisNames:
  """Named access to batch axes to avoid hardcoded strings."""

  N_STRUCTURES = "n_structures"
  N_SAMPLES = "n_samples"
  N_TEMPERATURES = "n_temperatures"
  N_NOISES = "n_noises"


# Conservative per-residue activation footprint (float32 bytes) for one innermost
# (structure, noise, temperature, sample) combination in the sampling dispatch built
# by host/kernel_dispatch.py's _sample_batch. Two live tensors dominate:
#   - the EncoderOutput held for the noise/temp loop (aminx.types.encodings.EncoderOutput:
#     node_features (L, D), edge_features (L, K, D)) -- same shape family as, and reusing
#     the same conservative D=128/K=32/D_edge=48 constants already established by
#     aminx.sampling.conditional_logits._plan_axis_strategy's callers for this exact type
#     (see batched_encode_fn's `activation_bytes = seq_len * (128 + 32 * 48) * 4`), not a
#     fresh guess;
#   - the per-sample decode output (res.logits, shape (L, 21)) -- same convention as
#     conditional_logits.py's batched_decode_fn (`seq_len * 21 * 4`).
# estimate_memory_theoretical's single flat product (see aminx.tiling.planner) does not
# distinguish "computed once per noise/temp" from "computed once per sample" -- summing
# both into one per-element constant is the conservative choice given that architecture,
# not an attempt to model exact live-tensor accounting.
_SAMPLING_NODE_FEATURE_DIM = 128
_SAMPLING_EDGE_NEIGHBORS = 32
_SAMPLING_EDGE_FEATURE_DIM = 48
_SAMPLING_DECODE_VOCAB = 21
_SAMPLING_BYTES_PER_ELEMENT = 4  # float32


def _sampling_activation_bytes_per_element(seq_len: int) -> float:
  """Conservative per-residue activation-byte estimate for one sampling dispatch unit.

  See the module-level constants above for the shape/constant provenance. Used as
  ``estimate_memory_theoretical``'s ``base_shape_bytes`` argument in
  ``make_sampling_planner`` -- previously hardcoded to ``1.0`` (a dimensionless
  placeholder, off by ~7-8 orders of magnitude from a real device memory budget for
  a realistic samples/temperatures/noises composition), which made the joint-budget
  Vmap-vs-SafeMap demotion check for those three axes practically inert regardless of
  real activation size (found investigating the E11e heterogeneous-batch benchmark
  session, 2026-07-23; git-blame traces the ``1.0`` to the 2026-07-06 joint-budget
  migration, commit 35dc18b8, inherited unreviewed from the pre-migration local
  planner -- see .praxia/docs/specs/260706_epic1541-planner-joint-budget-migration.md,
  which never discusses base_shape_bytes/activation_multiplier calibration).
  """
  per_residue_elements = (
    _SAMPLING_NODE_FEATURE_DIM
    + _SAMPLING_EDGE_NEIGHBORS * _SAMPLING_EDGE_FEATURE_DIM
    + _SAMPLING_DECODE_VOCAB
  )
  return seq_len * per_residue_elements * _SAMPLING_BYTES_PER_ELEMENT


def make_sampling_planner(
  spec: SamplingSpecification,
  param_bytes: float = 0.0,
  headroom: float = 0.80,
  activation_multiplier: float = 2.5,
  n_samples_override: int | None = None,
  seq_len: int | None = None,
) -> BatchPlan:
  """Create a BatchPlan for _sample_batch dispatch with advisory logging.

  Parameters
  ----------
  spec : SamplingSpecification
      Sampling specification containing batch size and temperature/noise parameters.
  param_bytes : float, optional
      Estimated model parameter size in bytes. Default 0.0.
  headroom : float, optional
      Fraction of device memory to use. Default 0.80 (80% headroom).
  activation_multiplier : float, optional
      Multiplier for activation memory estimation. Default 2.5.
  n_samples_override : int | None, optional
      Actual n_samples cardinality for this call, e.g. from
      ``resolve_target_samples``. When given, this is what the N_SAMPLES
      axis is planned against instead of ``spec.samples_batch_size``.

      ``spec.samples_batch_size`` and the real per-call sample count
      (``spec.samples_chunk_size``/``spec.num_samples``, resolved via
      ``resolve_target_samples``) are two independently-settable fields with
      no cross-validation -- passing this override closes that gap at the
      source: the planner's Vmap/SafeMap decision is verified against the
      array size that's actually dispatched, not a stale, disconnected
      proxy for it. Without this override, a plan that decides Vmap because
      the small default ``samples_batch_size`` (16) fit the memory budget
      gets applied to a possibly much larger real sample count with no
      re-check -- see
      ``.praxia/docs/specs/260706_samples-axis-planner-cardinality-mismatch.md``
      and Finding D of
      ``.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md``.
      Defaults to None (falls back to the pre-existing, disconnected
      ``spec.samples_batch_size`` behavior) only for backward compatibility
      with callers/tests that construct a plan without a real per-call
      sample count in hand; the one production call site
      (``host/kernel_dispatch.py``'s ``_sample_batch``) always passes it.
  seq_len : int | None, optional
      Real per-structure residue count, e.g. from
      ``batched_ensemble.coordinates.shape[1]``. Used to compute a real
      per-element activation-byte estimate for the joint memory-budget check
      (see ``_sampling_activation_bytes_per_element``) instead of the
      previous hardcoded ``base_shape_bytes=1.0`` placeholder, which made the
      budget check practically inert for the samples/temperatures/noises
      axes regardless of real activation size. Defaults to ``None`` (falls
      back to the old ``1.0`` placeholder) only for backward compatibility
      with callers/tests that construct a plan without a real structure in
      hand; the one production call site (``host/kernel_dispatch.py``'s
      ``_sample_batch``) always passes it.

  Returns
  -------
  BatchPlan
      Batch size decisions for each sampling axis (structures, samples, temps, noises).

  """
  try:
    limit = jax.devices()[0].memory_stats()["bytes_limit"]
  except Exception:
    limit = 4 * 1024**3
  budget_bytes = int(limit * headroom - param_bytes)
  if n_samples_override is not None:
    samples_cardinality = max(1, n_samples_override)
  else:
    samples_cardinality = max(1, getattr(spec, "samples_batch_size", 128) or 128)
  axes = [
    dataclasses.replace(N_STRUCTURES, cardinality=max(1, getattr(spec, "batch_size", 1) or 1)),
    dataclasses.replace(N_SAMPLES, cardinality=samples_cardinality),
    dataclasses.replace(
      N_TEMPERATURES,
      cardinality=max(1, len(getattr(spec, "temperature", [1.0]))),
    ),
    dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
  ]
  base_shape_bytes = 1.0 if seq_len is None else _sampling_activation_bytes_per_element(seq_len)
  return _plan_with_joint_budget(
    axes,
    budget_bytes=budget_bytes,
    estimate_fn=lambda ds: estimate_memory_theoretical(ds, base_shape_bytes, activation_multiplier),
    carry_specs=getattr(spec, "carry_specs", None) or [],
    dedup_specs=getattr(spec, "dedup_specs", None) or [],
  )


def _legacy_batch_size(decision: AxisDecision) -> int:
  """Translate an AxisDecision to aminx.utils.safe_map's calling convention.

  safe_map treats batch_size=0 (or None) as "no chunking, run everything at
  once" -- the Vmap-equivalent -- and any positive value as "chunk into
  groups of that size." aminx's retired local BatchPlanner always set
  batch_size=0 for Vmap decisions to match; xtrax's BatchPlanner never does
  (Vmap decisions carry batch_size=spec.default_batch_size, e.g. 1). Feeding
  that straight to safe_map would silently turn a parallel Vmap axis into a
  fully serial per-element loop for aminx's legacy (use_unified_driver=False)
  dispatch path in host/kernel_dispatch.py, which is not dead code (a real,
  still-tested CLI flag) -- EPIC #1541 T-PLANNER.2 finding, 2026-07-06.
  """
  if type(decision.strategy).__name__ == "Vmap":
    return 0
  return decision.batch_size


def extract_batch_sizes(plan: BatchPlan) -> tuple[int, int, int, int]:
  """Extract batch sizes for all sampling axes from a BatchPlan.

  Parameters
  ----------
  plan : BatchPlan
      Batch plan from make_sampling_planner.

  Returns
  -------
  tuple[int, int, int, int]
      Tuple of (structures_bs, samples_bs, temps_bs, noises_bs), in
      aminx.utils.safe_map's convention (0 means Vmap/no-chunking) --
      see _legacy_batch_size.

  """
  structures_bs = _legacy_batch_size(decision_for(plan, AxisNames.N_STRUCTURES))
  samples_bs = _legacy_batch_size(decision_for(plan, AxisNames.N_SAMPLES))
  temps_bs = _legacy_batch_size(decision_for(plan, AxisNames.N_TEMPERATURES))
  noises_bs = _legacy_batch_size(decision_for(plan, AxisNames.N_NOISES))
  return structures_bs, samples_bs, temps_bs, noises_bs


def compute_sample_keys(
  base_key: PRNGKeyArray,
  target_num_samples: int,
  chunk_sample_start: int | None = None,
  grid_lineage_sample_start: int | None = None,
) -> jax.Array:
  """Compute deterministic PRNG keys for all samples via fold_in.

  Parameters
  ----------
  base_key : PRNGKeyArray
      Base PRNG key for folding.
  target_num_samples : int
      Number of samples to generate keys for.
  chunk_sample_start : int | None, optional
      Explicit chunk start index. Default None.
  grid_lineage_sample_start : int | None, optional
      Grid lineage sample start index. Default None.

  Returns
  -------
  jax.Array
      Array of shape ``(target_num_samples,)`` containing folded keys.

  """
  sample_indices = np.arange(target_num_samples, dtype=np.int32)
  if chunk_sample_start is not None:
    sample_indices += int(chunk_sample_start)
  elif grid_lineage_sample_start is not None:
    sample_indices += int(grid_lineage_sample_start)

  # Generate keys for each sample via fold_in
  return jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(sample_indices)


def resolve_target_samples(
  spec: SamplingSpecification,
  chunk_sample_count: int | None = None,
  grid_lineage: dict[str, int | str] | None = None,
) -> int:
  """Resolve the target number of samples for this batch.

  Prioritizes explicit chunk_sample_count, then grid lineage sample_count,
  then spec.num_samples as fallback.

  Parameters
  ----------
  spec : SamplingSpecification
      Sampling specification.
  chunk_sample_count : int | None, optional
      Explicit sample count for this chunk. Default None.
  grid_lineage : dict[str, int | str] | None, optional
      Grid lineage dict with 'sample_count' key. Default None.

  Returns
  -------
  int
      Resolved target sample count (positive).

  Raises
  ------
  ValueError
      If resolved sample count is not positive.

  """
  if chunk_sample_count is not None:
    target = int(chunk_sample_count)
  elif grid_lineage is not None:
    target = int(grid_lineage["sample_count"])
  else:
    target = int(spec.run_spec.sampling.num_samples)

  if target <= 0:
    msg = "num_samples must be positive."
    raise ValueError(msg)
  return target


def resolve_chunk_size(
  spec: SamplingSpecification,
  total_num_samples: int,
  grid_lineage: dict[str, int | str] | None = None,
) -> int:
  """Resolve the chunk size for streaming sample output.

  Uses spec.samples_chunk_size if set, otherwise grid_lineage['sample_count']
  if grid_lineage exists, otherwise total_num_samples.

  Parameters
  ----------
  spec : SamplingSpecification
      Sampling specification.
  total_num_samples : int
      Total number of samples to be generated.
  grid_lineage : dict[str, int | str] | None, optional
      Grid lineage dict. Default None.

  Returns
  -------
  int
      Chunk size (positive).

  """
  if hasattr(spec, "samples_chunk_size") and spec.samples_chunk_size:
    return int(spec.samples_chunk_size)
  if grid_lineage is not None:
    return int(grid_lineage["sample_count"])
  return total_num_samples


def resolve_sample_start(
  grid_lineage: dict[str, int | str] | None = None,
) -> int:
  """Resolve the sample start index from grid lineage or default to 0.

  Parameters
  ----------
  grid_lineage : dict[str, int | str] | None, optional
      Grid lineage dict with 'sample_start' key. Default None.

  Returns
  -------
  int
      Sample start index (0-based).

  """
  return int(grid_lineage["sample_start"]) if grid_lineage is not None else 0


def _validate_plan_topology(
  plan: BatchPlan,
  stage_set: StageSet,
) -> None:
  """Validate plan topology at plan construction time.

  Checks:
  1. No Scan strategy on heterogeneous axes (jax.lax.scan requires static carry shape).
  2. No ordered=True boundary op (Tap/Sink) on Vmap axes (vmap has no step ordering).
  3. No STEDecode paired with UnconditionalDecodeStep (STE requires conditional scoring).

  Rules 1-2 are generic tiling/boundary concerns, not MPNN-specific -- they're
  delegated to xtrax.stages.validate_plan_topology (upstreamed; aminx had
  independently built the same checks that xtrax's own AxisBoundary docstring
  promised but never implemented). Rule 3 is aminx/MPNN-domain-specific and
  stays local. xtrax's PlanTopologyError is translated to aminx's own (a
  TilingError subclass) so the existing `except TilingError` / `isinstance`
  contract is preserved for callers.

  Raises:
      PlanTopologyError: on first violation found.

  """
  from xtrax.stages import PlanTopologyError as _XtraxPlanTopologyError
  from xtrax.stages import validate_plan_topology as _xtrax_validate_plan_topology

  try:
    _xtrax_validate_plan_topology(plan.decisions, stage_set.axis_boundaries)
  except _XtraxPlanTopologyError as exc:
    raise PlanTopologyError(str(exc)) from exc

  # Rule 3: STEDecode requires ConditionalDecodeStep, not UnconditionalDecodeStep (Sprint 6)
  decode_fn = getattr(plan, "decode_fn", None)
  if decode_fn is not None:
    # Lazy import to avoid circular dependencies
    from aminx.inference.decode.ste import STEDecode
    from aminx.types.stages import UnconditionalDecodeStep

    if isinstance(decode_fn, STEDecode):
      decode_step = getattr(stage_set, "decode_step", None)
      if isinstance(decode_step, UnconditionalDecodeStep):
        msg = (
          "PlanTopologyError: STEDecode requires ConditionalDecodeStep on "
          "stage_set.decode_step; got UnconditionalDecodeStep. "
          "STE (straight-through estimator) requires conditional scoring via "
          "sequence context; unconditional decoding is incompatible."
        )
        raise PlanTopologyError(msg)


# ---------------------------------------------------------------------------
# COMP-8: InferencePlan and related components for unified inference dispatch
# ---------------------------------------------------------------------------


class InferenceComponents(NamedTuple):
  """Resolved inference components for encode-once/decode-many pipeline.

  Parameters
  ----------
  encode_fn : Callable
      Encoder forward pass. Signature:
      ``(bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) → EncoderOutput``
  stage_set : Any
      StageSet instance with all slots wired (logit_transform, ar_logit_transform,
      decode_step, sample_step, tie_group_fuse). Ready for JIT.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """

  encode_fn: Callable
  stage_set: Any  # StageSet


class InferencePlan(eqx.Module):
  """Resolved inference plan with encode-once/decode-many pattern.

  Encodes geometry and ligand context once, then reuses encoder output for
  multiple decode passes (sampling or scoring) with different stage_set instances.

  Parameters
  ----------
  model : Any
      Parameterized protein/ligand model (carries JAX arrays).
  components : InferenceComponents
      Resolved components: encode_fn, driver, stage_set.
  decode_fn : DecodeScoreFn | ARDecodeFn | STEDecodeFn
      Resolved decode mode class instance (ConditionalDecode, UnconditionalDecode,
      AutoregressiveDecode, or STEDecode). Wired once at plan construction time.

  Notes
  -----
  `.sample()` and `.score()` invoke the same encode → decode pipeline but with
  different stage_set configurations. The decode_step and sample_step fields in
  stage_set determine the output (sampled sequence or logits).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """

  model: Any
  components: InferenceComponents
  decode_fn: DecodeScoreFn | ARDecodeFn | STEDecodeFn
  packer: Any = None
  # Concrete/Gumbel annealing for the STE decode. STEDecode.__call__ accepts these as
  # CALL-TIME arguments with defaults, so a plan that does not forward them silently pins the
  # schedule at (False, 1.0, 0.1) regardless of what the SamplingSpecification asked for.
  # Static: they are Python scalars consumed as trace-time constants, not traced arrays.
  use_concrete: bool = eqx.field(static=True, default=False)
  concrete_tau_start: float = eqx.field(static=True, default=1.0)
  concrete_tau_end: float = eqx.field(static=True, default=0.1)

  @property
  def stage_set(self) -> Any:
    """Access the wired StageSet directly."""
    return self.components.stage_set

  def with_decode_fn(self, fn: DecodeScoreFn | ARDecodeFn | STEDecodeFn) -> InferencePlan:
    """Return a new InferencePlan with the given decode function.

    Parameters
    ----------
    fn : DecodeScoreFn | ARDecodeFn | STEDecodeFn
        New decode function to use.

    Returns
    -------
    InferencePlan
        New immutable InferencePlan with updated decode_fn.

    """
    return eqx.tree_at(lambda p: p.decode_fn, self, fn)

  @eqx.filter_jit
  def encode(
    self,
    bundle: InferenceBundle,
    key: PRNGKeyArray,
    config: InferenceConfig,
  ) -> EncoderOutput:
    """Encode geometry and context into reusable encoder output.

    Parameters
    ----------
    bundle : InferenceBundle
        Input geometry, conditioning, ligand, and wave schedule.
    key : PRNGKeyArray
        PRNG key (passed to encoder; may be used for noise injection).
    config : InferenceConfig
        Inference configuration.

    Returns
    -------
    EncoderOutput
        Encoded node features, edge features, neighbor indices, and mask.
        Reuse this output across multiple decode() calls for encode-once/decode-many.

    """
    return self.components.encode_fn(bundle, key, config)

  @eqx.filter_jit
  def decode(
    self,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    key: PRNGKeyArray,
    config: InferenceConfig,
  ) -> Any:
    """Decode pre-encoded features into logits or sampled sequences.

    Parameters
    ----------
    enc : EncoderOutput
        Pre-computed encoder output from encode(). May be reused across calls.
    bundle : InferenceBundle
        Input bundle providing conditioning and wave schedule for decoder.
    key : PRNGKeyArray
        PRNG key for any stochastic decoding.
    config : InferenceConfig
        Inference configuration.

    Returns
    -------
    Any
        SampleResult containing logits (shape (L, 21)) and argmax sequence.
        Dispatches via self.decode_fn (ConditionalDecode by default; see
        make_inference_plan for mode configuration).

    """
    from aminx.inference.sample_autoregressive import SampleResult

    if self.packer is not None and getattr(bundle, "packer", None) is None:
      raise ValueError(
        "Packer model is configured on the InferencePlan, but the "
        "InferenceBundle contains no packer bundle data.",
      )

    # Forward the annealing schedule ONLY to STEDecode: the other mode classes
    # (Conditional/Unconditional/Autoregressive) do not accept these kwargs, and passing
    # them unconditionally would be a TypeError on every non-STE decode.
    # Imported locally, matching this module's existing convention (see the topology check
    # around line 511) which keeps aminx.inference.decode out of plan.py's import cycle.
    from aminx.inference.decode.ste import STEDecode

    if isinstance(self.decode_fn, STEDecode):
      result = self.decode_fn(
        key,
        enc,
        bundle,
        config,
        self.components.stage_set,
        use_concrete=self.use_concrete,
        tau_start=self.concrete_tau_start,
        tau_end=self.concrete_tau_end,
      )
    else:
      result = self.decode_fn(
        key,
        enc,
        bundle,
        config,
        self.components.stage_set,
      )
    # Normalize: if driver returned raw logits, wrap as SampleResult
    if not isinstance(result, SampleResult):
      result = SampleResult(
        sequence=jnp.argmax(result, axis=-1).astype(jnp.int32),
        logits=result,
      )

    if self.packer is not None:
      n_states = enc.node_features.shape[0]
      seq = result.sequence
      if seq.ndim == 1:
        seq_broadcast = jnp.broadcast_to(seq, (n_states, seq.shape[0]))
      else:
        seq_broadcast = jnp.broadcast_to(seq, (n_states, seq.shape[-1]))

      bundle = eqx.tree_at(
        lambda b: b.packer.sequence,
        bundle,
        seq_broadcast,
      )

      keys_for_states = jax.random.split(key, n_states)
      packer_result_stack = _run_packer_vmap(
        self.packer,
        keys_for_states,
        bundle.packer,
        config,
      )

      result = dataclasses.replace(result, packer_result=packer_result_stack)

    return result

  @eqx.filter_jit
  def sample(self, bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) -> Any:
    """Encode and sample from the pipeline.

    Parameters
    ----------
    bundle : InferenceBundle
        Input geometry, conditioning, ligand, and wave schedule.
    key : PRNGKeyArray
        PRNG key for sampling.
    config : InferenceConfig
        Inference configuration (batch, device, etc.).

    Returns
    -------
    Any
        Sampled sequence (or auxiliary output from driver).

    """
    enc = self.encode(bundle, key, config)
    return self.decode(enc, bundle, key, config)

  @eqx.filter_jit
  def score(self, bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) -> Logits:
    """Encode and score the pipeline.

    Parameters
    ----------
    bundle : InferenceBundle
        Input geometry, conditioning, ligand, and wave schedule.
    key : PRNGKeyArray
        PRNG key for any stochastic operations.
    config : InferenceConfig
        Inference configuration (batch, device, etc.).

    Returns
    -------
    Logits
        Logit scores per position per amino acid.

    """
    enc = self.encode(bundle, key, config)
    return self.decode(enc, bundle, key, config)


@eqx.filter_jit
def _run_packer_vmap(packer_model, keys, packer_bundle, config_params):
  from aminx.types.bundles import PackerBundle

  packer_in_axes = PackerBundle(
    sequence=0,
    backbone_coords=0,
    backbone_mask=0,
    ligand_coords=0,
    ligand_mask=0,
    ligand_atom_types=0,
    mask=0,
    residue_index=0,
    chain_labels=0,
    backbone_noise=None,
  )

  def run_single(k, b):
    return packer_model(k, b, config_params)

  return jax.vmap(run_single, in_axes=(0, packer_in_axes))(keys, packer_bundle)


def resolve_decode_mode(
  run_spec: Any,
  *,
  purpose: Literal["sample", "score"],
) -> Any:
  """Resolve which DecodeMode to build from a RunSpec -- the ONE place this decision is made.

  Both `host.plan.make_inference_plan` (the single-structure/scoring path) and
  `sampling.multistate_poe.sample_multistate_poe_bead` (the genuine multi-state PoE path) call
  this instead of each independently deciding, which is how aminx#110 happened: the two paths
  disagreed about which decode class to build (and therefore which conditioning fields --
  fixed_mask, temperature, num_samples -- had any effect at all) with nothing forcing agreement.

  `purpose` distinguishes "produce new sequences" from "evaluate a given sequence" --
  `run_spec.sampling.sampling_strategy` alone cannot do this: a sampling call at its default
  "temperature" strategy and a scoring call share the identical (non-"straight_through") value,
  and were structurally indistinguishable before this parameter existed. That indistinguishability
  is the literal root cause of aminx#110 -- `make_inference_plan` always built `ConditionalDecode`
  (a single teacher-forced pass, correct for scoring, silently wrong for sampling) because nothing
  told it which of the two callers it was serving.

  Parameters
  ----------
  run_spec : RunSpec
      The resolved `spec.run_spec` (not the flat spec) -- `sampling_strategy` lives at
      `run_spec.sampling.sampling_strategy` (added 260716, EPIC #1541 P4).
  purpose : {"sample", "score"}
      "sample" -- caller wants new sequences (host.runner.sample(), the PoE campaign path).
      "score" -- caller wants to evaluate/inspect a given sequence (host.runner.score(),
      inspect()); always resolves to ConditionalMode unless straight_through is requested.

  Returns
  -------
  ConditionalMode | AutoregressiveMode | STEMode

  """
  from aminx.inference.decode.mode import AutoregressiveMode, ConditionalMode, STEMode

  if run_spec.sampling.sampling_strategy == "straight_through":
    return STEMode()
  if purpose == "sample":
    return AutoregressiveMode()
  return ConditionalMode()


def make_inference_plan(
  model: ModelProtocol,
  spec: Any,
  packer: Any = None,
  *,
  purpose: Literal["sample", "score"] = "score",
) -> InferencePlan:
  """Factory: resolve and create an InferencePlan from model and spec.

  Assembles the inference pipeline by resolving encode_fn (from use_rolling_state),
  wiring logit_transform (from multi_state_strategy), and instantiating stage_set
  with ARLogitFuse and TieGroupProductOfExperts.

  Parameters
  ----------
  model : ModelProtocol
      Parameterized model with decoder, encoder, and embeddings.
  spec : Any
      Specification; decode-relevant fields are read from ``spec.run_spec.sampling``
      (use_rolling_state, multi_state_strategy, multi_state_temperature, state_weights,
      sampling_strategy, decoding_order_fn -- see ``run/spec.py::SamplingConfig``).
  purpose : {"sample", "score"}, keyword-only, default "score"
      Passed to :func:`resolve_decode_mode` -- disambiguates "produce new sequences"
      (``host.runner.sample()``) from "evaluate a given sequence" (``score()``/``inspect()``),
      since ``sampling_strategy`` alone cannot (see ``resolve_decode_mode``'s docstring).

  Returns
  -------
  InferencePlan
      Ready-to-use inference plan for sampling/scoring.

  Notes
  -----
  Resolution order:

  1. ``encode_fn`` — uses ``make_encode_fn(model, use_rolling_state=use_rolling_state)``.
     ``use_rolling_state=True`` selects scan-based multi-state encoding; False uses vmap.
  2. ``logit_transform`` — instantiated from ``LOGIT_STRATEGIES[multi_state_strategy]``
     with ``state_weights`` and ``multi_state_temperature``.
  3. ``ar_logit_transform`` — the SAME instance as ``logit_transform`` (step 2), so the
     autoregressive path honours ``multi_state_strategy``, ``state_weights`` and
     ``multi_state_temperature`` identically to the non-AR path; identity when S=1.
     Until 2026-07-28 this was hardcoded to ``ARLogitFuse()`` (unweighted arithmetic
     mean), which silently discarded all three on the AR path.
  4. ``tie_group_fuse`` — always wired as ``TieGroupProductOfExperts()`` (log-softmax sum
     across tied positions).
  5. ``decode_step`` and ``sample_step`` — for ``sampling_strategy="straight_through"``,
     ``decode_step`` is wired to ``ConditionalDecodeStep`` and ``sample_step`` remains
     ``None`` (teacher-forced path). All other strategies leave both slots as ``None``.
     Note: the pre-260716 note here about "driver selects topology at call time based on
     stage_set slot occupancy" described the now-deprecated ``inference/driver.py`` dispatch
     mechanism (``driver.decode()`` raises ``NotImplementedError`` today) -- decode dispatch is
     resolved once below, at plan-construction time, via ``resolve_decode_mode``, not at call
     time from stage_set slot occupancy.
  6. ``encoding_fusion`` — wired as ``ArithmeticMeanEncodingFusion`` when
     ``spec.average_node_features=True``; otherwise left ``None``.
  7. ``decode_fn`` — resolved via ``resolve_decode_mode(spec.run_spec, purpose=purpose)`` (see
     that function's docstring) then ``make_decode_fn(model, mode, strategy, decoding_order_fn)``,
     and wired as a top-level field on the plan. ``purpose`` is what lets this same factory serve
     both ``sample()`` (real autoregressive sampling) and ``score()``/``inspect()`` (teacher-forced
     evaluation) correctly -- see ``resolve_decode_mode`` for why ``sampling_strategy`` alone
     can't make that distinction (aminx#110).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """
  from aminx.inference.decode.factory import make_decode_fn
  from aminx.inference.encode import make_encode_fn
  from aminx.inference.logits import make_stage_set
  from aminx.tiling.strategy import Vmap

  sampling_config = spec.run_spec.sampling

  use_rolling_state = sampling_config.use_rolling_state
  encode_fn = make_encode_fn(model, use_rolling_state=use_rolling_state)

  strategy_name = sampling_config.multi_state_strategy or "arithmetic_mean"
  strategy_temp = sampling_config.multi_state_temperature or 1.0
  state_weights = sampling_config.state_weights
  # No `or` fallback: sharpness=None means "use S", a real value, not a missing one.
  sharpness = getattr(sampling_config, "multi_state_sharpness", 1.0)

  stage_set = make_stage_set(
    strategy_name,
    strategy_temp,
    state_weights,
    sharpness=sharpness,
  )

  # Wire encoding fusion for averaged mode
  if getattr(spec, "average_node_features", False):
    import equinox as eqx

    from aminx.host.averaging import ArithmeticMeanEncodingFusion

    stage_set = eqx.tree_at(
      lambda s: s.encoding_fusion,
      stage_set,
      ArithmeticMeanEncodingFusion(),
      is_leaf=lambda x: x is None,
    )

  # Wire decoding fusion if specified in the spec
  decoding_fusion = getattr(spec, "decoding_fusion", None)
  if decoding_fusion is not None:
    import equinox as eqx

    stage_set = eqx.tree_at(
      lambda s: s.decoding_fusion,
      stage_set,
      decoding_fusion,
      is_leaf=lambda x: x is None,
    )

  # Wire STE (straight-through estimator) decode topology
  if sampling_config.sampling_strategy == "straight_through":
    import equinox as eqx

    from aminx.types.stages import ConditionalDecodeStep

    conditional_decode_step = ConditionalDecodeStep(
      decoder=model.decoder,
      w_s_embed=model.w_s_embed.weight,
    )
    stage_set = eqx.tree_at(
      lambda s: s.decode_step,
      stage_set,
      conditional_decode_step,
      is_leaf=lambda x: x is None,
    )
    # sample_step stays None (already None from make_stage_set) — teacher-forced path

  # Resolve decode_fn via the shared resolver (aminx#110: this is the ONE place both this
  # function and sample_multistate_poe_bead decide which DecodeMode to build -- see
  # resolve_decode_mode's docstring for why sampling_strategy alone can't disambiguate
  # "sample" from "score" callers).
  decode_mode = resolve_decode_mode(spec.run_spec, purpose=purpose)
  decode_strategy = Vmap()
  decode_fn = make_decode_fn(
    model,
    mode=decode_mode,
    strategy=decode_strategy,
    decoding_order_fn=sampling_config.decoding_order_fn,
  )

  components = InferenceComponents(
    encode_fn=encode_fn,
    stage_set=stage_set,
  )

  # The annealing knobs live on the FLAT SamplingSpecification (specs.py), not on
  # run_spec.sampling, so read them off the spec with getattr -- ScoringSpecification and the
  # other spec classes reach this function too and do not declare them.
  return InferencePlan(
    model=model,
    components=components,
    decode_fn=decode_fn,
    packer=packer,
    use_concrete=bool(getattr(spec, "use_concrete", False)),
    concrete_tau_start=float(getattr(spec, "concrete_tau_start", 1.0)),
    concrete_tau_end=float(getattr(spec, "concrete_tau_end", 0.1)),
  )


def plan_bucketed(
  spec: SamplingSpecification,
  sequence_lengths: list[int],
  axes: list[AxisSpec],
  *,
  budget_bytes: int,
  estimate_fn: Callable[[Sequence[AxisDecision]], float],
  carry_specs: list[CarrySpec] | None = None,
  dedup_specs: list[DedupSpec] | None = None,
  bucketing_config: BucketingConfig | None = None,
) -> BucketAssignment:
  """Plan inference for a batch grouped by sequence-length buckets.

  For each bucket, override the "n_structures" axis cardinality to the bucket
  ceiling (number of sequences in that bucket) and plan once via
  _plan_with_joint_budget. Returns a BucketAssignment with per-bucket BatchPlans.

  NOTE: The implementation overrides n_structures cardinality (a structure count)
  to the bucket ceiling (a sequence length). This may be a semantic mismatch.
  Implementing as specified, but this should be reviewed.

  Takes axes/budget_bytes/estimate_fn/carry_specs/dedup_specs directly (EPIC
  #1541 T-PLANNER.3) rather than a pre-built BatchPlanner: xtrax.tiling.
  BatchPlanner isn't a dataclass and doesn't hold axes as an attribute (axes
  are a per-call .plan(specs) argument), so the old "mutate an existing
  planner's .axes field" design has no equivalent -- there's nothing to
  dataclasses.replace(). This function has no production callers today
  (only its own test), so the signature change is contained to this file.

  Parameters
  ----------
  spec : SamplingSpecification
      Sampling specification (unused in function, required for interface).
  sequence_lengths : list[int]
      Sequence lengths for each position in the batch.
  axes : list[AxisSpec]
      Axes to plan (must include an "n_structures" axis).
  budget_bytes : int
      Joint memory budget in bytes, passed to _plan_with_joint_budget.
  estimate_fn : Callable[[Sequence[AxisDecision]], float]
      Joint memory estimator, passed to _plan_with_joint_budget.
  carry_specs : list[CarrySpec] | None, optional
      CarrySpec declarations, passed to _plan_with_joint_budget.
  dedup_specs : list[DedupSpec] | None, optional
      DedupSpec declarations, passed to _plan_with_joint_budget.
  bucketing_config : BucketingConfig | None, optional
      Bucketing configuration. Default is BucketingConfig().

  Returns
  -------
  BucketAssignment
      Assignment with bucket grouping, boundaries, and per-bucket plans.

  Raises
  ------
  ValueError
      If sequence_lengths is empty or any length exceeds all buckets.
  KeyError
      If no "n_structures" axis found in axes.

  """
  from aminx.tiling.bucketing import (
    BucketAssignment,
    BucketingConfig,
    group_by_bucket,
  )

  if not sequence_lengths:
    raise ValueError("sequence_lengths cannot be empty")

  if bucketing_config is None:
    bucketing_config = BucketingConfig()

  # Group sequences by bucket
  bucket_groups = group_by_bucket(sequence_lengths, bucketing_config)

  # Confirm an n_structures axis is present to override
  if not any(axis.name == "n_structures" for axis in axes):
    raise KeyError('No "n_structures" axis found in axes')

  # Plan for each bucket
  per_bucket_plans: dict[int, BatchPlan] = {}
  for bucket_ceil, _indices in bucket_groups.items():
    # Override n_structures cardinality to bucket ceiling
    modified_axes = []
    for axis in axes:
      if axis.name == "n_structures":
        # NOTE: This overrides cardinality (structure count) to bucket ceiling (seq length).
        # May be semantic mismatch; implementing as specified.
        modified_axis = dataclasses.replace(axis, cardinality=bucket_ceil)
        modified_axes.append(modified_axis)
      else:
        modified_axes.append(axis)

    per_bucket_plans[bucket_ceil] = _plan_with_joint_budget(
      modified_axes,
      budget_bytes=budget_bytes,
      estimate_fn=estimate_fn,
      carry_specs=carry_specs,
      dedup_specs=dedup_specs,
    )

  # Create sorted bucket boundaries
  bucket_boundaries = tuple(sorted(bucket_groups.keys()))

  return BucketAssignment(
    bucket_boundaries=bucket_boundaries,
    bucket_groups=bucket_groups,
    per_bucket_plans=per_bucket_plans,
  )
