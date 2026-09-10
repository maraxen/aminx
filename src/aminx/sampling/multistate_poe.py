"""Genuine multi-state PoE-fused autoregressive sampling for a bead of k reference states.

Fixes a real gap found 2026-07-13 (tev_design task 260709_multistate-fusion-strategy-comparison,
Phase 3; see ../../.praxia/docs/decisions/260713_no-real-multistate-sampling-path-exists.md for
the full trace): `aminx campaign plan`/`run`'s real sampling dispatcher
(`host/kernel_dispatch.py::_sample_batch`) treats every `--inputs` path as an independent
`N_STRUCTURES` batch item and builds a single-state (`num_states=1`) bundle per call -- no
CLI-reachable path ever constructs a genuinely stacked `num_states>1` bundle for autoregressive
sampling, so `multi_state_strategy="product"` fusion never actually combines states in any real
campaign row.

This module is a new, separate, minimal-risk entry point (chosen over touching
`_sample_batch`/`kernel_dispatch.py` itself -- see the decision doc's option 1): it builds ONE
combined multi-state bundle from all of a bead's reference states (reusing the same structure
loading/padding/chain-filtering (`prep_protein_stream_and_model`) and bundle-building
(`build_inference_bundle`) machinery `_sample_batch` already uses per-structure, just without the
per-structure_idx slicing loop) and samples from it via `aminx.inference.sample_autoregressive`,
which already correctly wires `AutoregressiveMode` and genuinely fuses an `S>1` state axis
(confirmed by the existing `test_ar_decode_state_position_map_changes_fused_logits` test).

Does NOT touch `_sample_batch`/`kernel_dispatch.py`, or `bundle_builder.py` -- every existing
campaign row (single-structure spike-in beads) is unaffected.

`sample_multistate_poe_campaign_row` (added 2026-07-14) wires this into `aminx campaign
plan/run`'s execution path: `host/campaign.py::run_manifest_row` detects a genuine multi-state
PoE row (`len(sampling_spec.inputs) > 1`) and calls it instead of `sample()`, so real campaign
manifests actually produce fused output through the normal `aminx campaign run` CLI, retaining
the surrounding locking/done-marker/retry/resumability infrastructure unchanged. It writes via
`xtrax.run.ZarrStagingSink`/`SinkSpec` -- the same Sink primitive `host/streaming.py::
_sample_streaming` uses -- matching that function's campaign-mode attrs/schema convention as
closely as the fused (not per-structure) output shape allows. The more general,
xtrax-composable arbitrary-fusion-axis version of the *underlying dispatch itself* (option 2
from the decision doc, a real `N_POE_STATES`-style `AxisSpec`) remains real, separate follow-up
work, tracked as praxia debt #589 -- this campaign wiring does not close that debt, it just makes
option 1's MVP actually reachable from a real campaign run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.run import SinkSpec, ZarrStagingSink
from xtrax.tiling import BatchPlanner
from xtrax.tiling import SafeMap as _XtraxSafeMap
from xtrax.tiling import Vmap as _XtraxVmap
from xtrax.tiling.estimators import lowered_memory_estimate

from aminx.host._sampling_grid_lineage import (
  _base_sampling_key,
  _grid_iteration_arrays,
  _grid_job_seed_hash,
  _grid_manifest_row_hash,
  _grid_sample_indices,
  _resolve_grid_lineage,
)
from aminx.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _prepare_fixed_controls,
  _prepare_ligand_context,
  fixed_provenance_outputs,
)
from aminx.host.plan import (
  resolve_chunk_size,
  resolve_decode_mode,
  resolve_sample_start,
  resolve_target_samples,
)
from aminx.host.prep import prep_protein_stream_and_model
from aminx.host.streaming import GRID_SCHEMA_VERSION, SAMPLING_SCHEMA_VERSION, _grid_lineage_attrs
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference.sample_autoregressive import kernel as _sample_autoregressive_kernel
from aminx.io.sink_provenance import (
  SINK_PROVENANCE_VERSION,
  assert_uniform_group_attr,
  logits_bias_semantics_outputs,
  prng_seed_attrs,
  resolve_aminx_version,
)
from aminx.sampling.conditional_logits import _plan_axis_strategy
from aminx.tiling.axes import N_SAMPLES, N_STATES
from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax
from aminx.tiling.strategy import SafeMap, Vmap

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from aminx.model.mpnn import Aminx
  from aminx.run.specs import SamplingSpecification
  from aminx.types.bundles import InferenceBundle
  from aminx.types.configs import InferenceConfig
  from aminx.types.stages import StageSet


@eqx.filter_jit
def sample_states_fused(
  model: Aminx,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
  prng_key: PRNGKeyArray,
  n_samples: int,
  sample_batch_size: int | None = None,
) -> tuple[Int[Array, "n_samples L"], Float[Array, "n_samples L 21"]]:
  """Draw n_samples independent samples from an already-built, already-fused bundle.

  Dispatches the sample-key axis via the same composable_jax idiom every other axis in this
  package uses -- `_plan_axis_strategy` (`aminx.sampling.conditional_logits`) resolves a real
  `xtrax.tiling.BatchPlanner` decision (Vmap when `n_samples` fits the device memory budget,
  SafeMap-chunked otherwise) against the existing `N_SAMPLES` `AxisSpec`
  (`aminx.tiling.axes`), then `make_axis_dispatch_via_xtrax` turns that decision into a typed
  iterator -- exactly how `make_batched_conditional_logits_split_fn` dispatches
  `N_REPLICATES`/`N_CANDIDATES` (`conditional_logits.py:390-397,421-427`). Deliberately not a
  bare `jax.vmap`: a fixed `jax.vmap` over an unbounded `n_samples` would attempt every sample
  in one shot regardless of the device's actual memory, exactly the failure mode
  `_plan_axis_strategy`'s `BatchPlanner`/`MemoryBudget` exists to prevent for large campaign-
  scale sample counts (thousands of samples per bead).

  The per-sample memory estimate feeding that decision (`activation_bytes_per_element`) is
  measured, not guessed: `xtrax.tiling.estimators.lowered_memory_estimate` lowers+compiles a
  ONE-sample representative tile of the real computation (state axis already resolved, see
  below) and reads XLA's own buffer-assignment numbers, rather than a hand-typed byte formula.
  The prior formula (`num_states * seq_len * a hidden-dim-sized magic constant`) was linear and
  missed the AR mask's quadratic `(L, L)` term and the `(L, K, D)` edge-feature term entirely --
  exactly the kind of drift a graph-derived measurement can't have (aminx debt #945).

  Calls `aminx.inference.sample_autoregressive.kernel` -- the genuine, already-tested
  `AutoregressiveMode` sampling path, which fuses `bundle`'s own state axis
  (`bundle.geometry.n_states`) via `stage_set.logit_transform`/`_realign_states_to_reference`
  exactly as `ConditionalDecode`'s teacher-forced scoring path does. Each sample gets its own
  PRNG key (`jax.random.split`); `bundle`/`config`/`stage_set` are shared and re-encoded once
  per sample by the kernel (matching `_sample_batch`'s own per-sample-key semantics).

  The state axis (`bundle.geometry.n_states`) is ALSO resolved through `BatchPlanner`, against
  the canonical `N_STATES` `AxisSpec` (`aminx.tiling.axes` -- `default_batch_size=1`, i.e.
  SafeMap-one-state-at-a-time whenever `num_states>1`), and passed to `kernel` as
  `state_strategy`. Previously `kernel` hardcoded `strategy=Vmap()` for the state axis
  unconditionally, invisible to any budget accounting -- fine at num_states=1 (the only
  case `sample.py`'s single-structure campaign path ever hits), but for a genuinely fused
  multi-state bundle it batches every decoder layer's per-state MLPs simultaneously. At
  production sample counts this produces a single fused GEMM whose shape XLA's autotuner
  cannot find a valid kernel config for: sample_count=128 crashed after an 809s compile
  ("Autotuning failed for HLO: f32[128,12582912]{1,0} fusion(...)"), sample_count=512 failed
  differently ("9 out of 89 instructions"). Resolving N_STATES through BatchPlanner instead
  removes the state axis from that multiplicative blowup entirely.

  Parameters
  ----------
  model : Aminx
      Parameterized model.
  bundle : InferenceBundle
      Bundle to sample from -- may be single-state (num_states=1) or genuinely multi-state
      (num_states>1, pre-fused via a real `state_position_map`); this function does not care
      which, it just samples from whatever bundle it's given.
  config : InferenceConfig
      Paired with `bundle` (from the same `build_inference_bundle` call).
  stage_set : StageSet
      Fusion/decode configuration (e.g. from `make_stage_set(strategy="product", ...)`).
  prng_key : PRNGKeyArray
      Base key; split into `n_samples` independent per-sample keys.
  n_samples : int
      Number of independent samples to draw.
  sample_batch_size : int | None, default None
      Fixed SafeMap tile size for the sample axis. None defers to the BatchPlanner's
      memory-budget-driven Vmap/SafeMap choice, matching
      `make_batched_conditional_logits_split_fn`'s `replicate_batch_size`/
      `candidate_batch_size` default behavior.

  Returns
  -------
  tuple
      (sequences, logits): sequences shape (n_samples, L) int32, logits shape
      (n_samples, L, 21) float32.

  """
  sample_keys = jax.random.split(prng_key, n_samples)
  num_states = bundle.geometry.coords.shape[0]

  # Resolve the state axis through BatchPlanner FIRST (see docstring): N_STATES'
  # default_batch_size=1 already encodes "SafeMap one state at a time whenever
  # num_states>1" via the plain cardinality-vs-batch-size rule -- deliberately
  # NOT a MemoryBudget/estimator call: a wrong or optimistic byte estimate could
  # keep this at Vmap, reproducing a variant of the exact bug this fixes, since
  # a hand-typed estimate is precisely what let it go unnoticed (aminx debt
  # #942). This is what previously never happened at all (kernel() hardcoded
  # Vmap unconditionally, bypassing BatchPlanner for this axis entirely).
  state_spec = dataclasses.replace(N_STATES, cardinality=num_states)
  xtrax_state_strategy = BatchPlanner().plan([state_spec]).decisions[0].strategy
  # BatchPlanner is xtrax-native; translate back to aminx's own AxisStrategy
  # union before handing it to aminx call sites (make_axis_dispatch_via_xtrax
  # expects aminx-native instances -- mirrors _plan_axis_strategy's identical
  # translation for the samples axis below).
  if isinstance(xtrax_state_strategy, _XtraxSafeMap):
    state_strategy = SafeMap(tile=xtrax_state_strategy.batch_size)
  elif isinstance(xtrax_state_strategy, _XtraxVmap):
    state_strategy = Vmap()
  else:
    msg = (
      "sample_states_fused: unexpected BatchPlanner state-axis decision "
      f"{type(xtrax_state_strategy)}"
    )
    raise TypeError(msg)

  def _one_sample(key: PRNGKeyArray) -> tuple[Any, Any]:
    result = _sample_autoregressive_kernel(
      model, key, bundle, config, stage_set, state_strategy=state_strategy,
    )
    return result.sequence, result.logits

  # Measure the REAL per-sample peak memory (state axis already correctly
  # resolved above) via xtrax's own compiled-graph-based estimator
  # (xtrax.tiling.estimators.lowered_memory_estimate -- XLA's own buffer-
  # assignment numbers from lowering+compiling a ONE-sample representative
  # tile, not the full n_samples shape), instead of a hand-typed byte formula.
  # The previous formula (num_states * seq_len * a hidden-dim-sized magic
  # constant) was linear and missed the AR mask's quadratic (L, L) term and
  # the (L, K, D) edge-feature term entirely -- exactly the kind of drift a
  # graph-derived measurement can't have, since it reads the real computation
  # instead of approximating it (aminx debt #945; this is a systemic gap
  # xtrax already provides the primitive to close -- lowered_memory_estimate
  # had zero call sites anywhere in aminx before this).
  activation_bytes = lowered_memory_estimate(_one_sample, sample_keys[0])
  strategy = _plan_axis_strategy(
    N_SAMPLES,
    n_samples,
    sample_batch_size,
    activation_bytes_per_element=activation_bytes,
  )
  iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_SAMPLES.name)

  sequences, logits = iterator(_one_sample, sample_keys)
  return sequences, logits


def sample_multistate_poe_bead(
  spec: SamplingSpecification,
  prng_key: PRNGKeyArray,
  n_samples: int,
) -> tuple[Int[Array, "n_samples L"], Float[Array, "n_samples L 21"]]:
  """Sample n_samples designs with genuine cross-state PoE fusion over spec.inputs' k states.

  Builds ONE combined multi-state bundle from all of `spec.inputs` (reusing the same
  structure-loading/padding pipeline `_sample_batch` uses per-structure) instead of `_sample_batch`'s
  real behavior of building `len(spec.inputs)` independent single-state bundles. `spec.state_position_map`
  (if set) is applied against this bundle's real `num_states=len(spec.inputs)` state axis, so
  `spec.multi_state_strategy` genuinely fuses across states -- unlike every existing campaign row.

  PRECONDITION: `spec.batch_size` must equal `len(spec.inputs)`, so the underlying protein
  dataset iterator yields exactly one batch covering every state (this function does not
  itself chunk or re-batch inputs; the caller is responsible for choosing a bead's k small
  enough that all states fit in one host batch, which the necklace campaign's k=4 always does).

  Only a single temperature and a single backbone-noise value are supported per call (the
  first entry of `spec.run_spec.sampling.temperature`/`backbone_noise` if either is a
  sequence) -- callers wanting a temperature/noise grid should call this once per value,
  mirroring how `build_necklace_p2_manifest.py` already treats those as manifest-row-level
  axes, not something this function's single call needs to dispatch internally.

  Parameters
  ----------
  spec : SamplingSpecification
      Real campaign-style spec -- same fields `_run_campaign_plan`/`run_manifest_row` already
      construct (`inputs`, `chain_id`, `checkpoint_id`, `multi_state_strategy`,
      `state_position_map`, `state_weights`, `tie_group_map`, etc).
  prng_key : PRNGKeyArray
      Base key for sampling (split internally into `n_samples` per-sample keys).
  n_samples : int
      Number of independent fused-PoE samples to draw for this bead.

  Returns
  -------
  tuple
      (sequences, logits) -- see `sample_states_fused`.

  Raises
  ------
  ValueError
      If `spec.inputs` has fewer than 2 entries, `spec.batch_size` doesn't match
      `len(spec.inputs)`, or the protein dataset iterator doesn't yield exactly one batch
      covering every input state (i.e. the precondition above was violated).

  """
  if not isinstance(spec.inputs, (list, tuple)) or len(spec.inputs) < 2:
    msg = (
      f"sample_multistate_poe_bead needs >=2 states in spec.inputs to fuse across "
      f"(got {spec.inputs!r}) -- for a single structure, use the existing campaign path."
    )
    raise ValueError(msg)
  n_states = len(spec.inputs)

  if spec.batch_size != n_states:
    msg = (
      f"spec.batch_size ({spec.batch_size}) must equal len(spec.inputs) ({n_states}) so the "
      "protein dataset iterator yields all states in one combined batch -- see this "
      "function's docstring precondition."
    )
    raise ValueError(msg)

  protein_iterator, model = prep_protein_stream_and_model(spec)
  batches = list(protein_iterator)
  if len(batches) != 1:
    msg = (
      f"expected exactly one combined batch covering all {n_states} states, got "
      f"{len(batches)} -- check spec.batch_size and the dataset iterator's actual "
      "batching behavior before trusting this function's output."
    )
    raise ValueError(msg)
  batched_ensemble = batches[0]
  if batched_ensemble.coordinates.shape[0] != n_states:
    msg = (
      f"batched_ensemble covers {batched_ensemble.coordinates.shape[0]} structures, "
      f"expected {n_states} (len(spec.inputs)) -- the dataset iterator split/dropped "
      "inputs unexpectedly."
    )
    raise ValueError(msg)

  seq_len = batched_ensemble.coordinates.shape[1]
  # spec.batch_size == n_states and the check above confirms exactly one combined batch, so
  # canonical_structure_ids IS the per-row identity for this batch with no offset needed --
  # unlike kernel_dispatch.py's chunked case, there's only ever one chunk here.
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  # _prepare_fixed_controls returns a (num_states, L) array -- shaped for
  # kernel_dispatch.py's per-structure_idx SLICING (each independent call gets its own
  # 1D (L,) row). build_inference_bundle's fixed_mask/fixed_tokens are design-level, not
  # per-state (its own default is 1D: jnp.zeros(seq_len)), and every row here is already
  # an identical broadcast of the same spec-level fixed positions (confirmed in
  # _prepare_fixed_controls's own body) -- take row 0 rather than passing the full 2D
  # array, which silently produces a wrong-shaped ConditioningBundle.fixed_mask/tokens
  # further downstream (confirmed 2026-07-13: this exact mistake broke AutoregressiveDecode's
  # wave-scan with a `Cannot broadcast to shape with fewer dimensions` error at do_sample's
  # seq_oh_stack construction -- not an aminx bug, a caller contract violation).
  #
  # The row-identity assertion below assumes fixed positions are the same across all states
  # for this call -- true for every real caller checked so far (tev_design's necklace P2
  # manifest builder never sets fixed_positions/fixed_mask at all; the prereg locks junction
  # placement identical across strata by design, see scripts/analysis/necklace_junction_placement.py).
  # _broadcast_per_structure/_prepare_fixed_controls DO support genuinely different fixed
  # positions per structure in general (that's their whole purpose for kernel_dispatch.py's
  # independent-structures case) -- a future caller with real per-state heterogeneous fixed
  # positions would hit the ValueError below, correctly, rather than this function silently
  # picking an arbitrary one of several different real fixed-position sets.
  fixed_mask_per_structure, fixed_tokens_per_structure = _prepare_fixed_controls(
    spec, batched_ensemble=batched_ensemble, batch_structure_ids=canonical_structure_ids,
  )
  if not bool(jnp.all(fixed_mask_per_structure == fixed_mask_per_structure[0])) or not bool(
    jnp.all(fixed_tokens_per_structure == fixed_tokens_per_structure[0]),
  ):
    msg = (
      "_prepare_fixed_controls returned different fixed_mask/fixed_tokens rows per "
      "structure -- this function assumes fixed positions are design-level (identical "
      "across all states), per _prepare_fixed_controls's own broadcast contract; a "
      "genuinely per-state fixed_mask is not supported by this function."
    )
    raise ValueError(msg)
  fixed_mask = fixed_mask_per_structure[0]
  fixed_tokens = fixed_tokens_per_structure[0]
  ligand_context = _prepare_ligand_context(
    spec,
    batched_ensemble=batched_ensemble,
    batch_size=n_states,
    seq_len=seq_len,
    canonical_structure_ids=canonical_structure_ids,
    batch_structure_ids=canonical_structure_ids,
  )

  state_weights = (
    jnp.asarray(spec.state_weights, dtype=jnp.float32) if spec.state_weights is not None else None
  )
  state_position_map = (
    jnp.asarray(spec.state_position_map) if spec.state_position_map is not None else None
  )
  if (
    state_position_map is not None
    and state_position_map.ndim == 2
    and state_position_map.shape[0] != n_states
  ):
    msg = (
      f"spec.state_position_map declares {state_position_map.shape[0]} states but this bead "
      f"has {n_states} (len(spec.inputs)) -- recompute state_position_map for this exact "
      "input set before calling this function (this is exactly the cardinality mismatch "
      "_realign_states_to_reference now rejects at fusion time, per the corruption fix)."
    )
    raise ValueError(msg)
  tie_group_map = jnp.asarray(spec.tie_group_map) if spec.tie_group_map is not None else None

  temperature_val = spec.run_spec.sampling.temperature
  temperature = float(temperature_val[0]) if isinstance(temperature_val, (list, tuple)) else float(
    temperature_val,
  )
  noise_val = spec.run_spec.sampling.backbone_noise
  backbone_noise = float(noise_val[0]) if isinstance(noise_val, (list, tuple)) else float(noise_val)
  bias = (
    jnp.asarray(spec.run_spec.sampling.bias, dtype=jnp.float32)
    if spec.run_spec.sampling.bias is not None
    else None
  )

  bundle, config = build_inference_bundle(
    coords=batched_ensemble.coordinates,
    mask=batched_ensemble.mask,
    residue_index=batched_ensemble.residue_index,
    chain_index=batched_ensemble.chain_index,
    backbone_noise=backbone_noise,
    fixed_mask=fixed_mask,
    fixed_tokens=fixed_tokens,
    bias=bias,
    tie_group_map=tie_group_map,
    state_weights=state_weights,
    state_position_map=state_position_map,
    ligand_coords=ligand_context["Y"],
    ligand_atom_types=ligand_context["Y_t"],
    ligand_mask=ligand_context["Y_m"],
    atom_37=ligand_context["atom_37"],
    atom_37_mask=ligand_context["atom_37_mask"],
    chain_mask=ligand_context["chain_mask"],
    structure_mapping=batched_ensemble.mapping,
    temperature=temperature,
    mode="sample_ar",
    inference=True,
  )

  stage_set = make_stage_set(
    strategy=spec.multi_state_strategy,
    strategy_temperature=getattr(spec, "multi_state_temperature", 1.0) or 1.0,
    state_weights=state_weights,
  )

  # Consult the SAME decode-mode resolver host/plan.py::make_inference_plan uses (aminx#110):
  # this function is hardwired to sample_states_fused's AutoregressiveMode-only kernel, so a
  # caller asking for anything else (today, only sampling_strategy="straight_through") must be
  # told loudly that this path can't honor it, rather than have the request silently ignored --
  # which is exactly how #110 went unnoticed for as long as it did.
  from aminx.inference.decode.mode import AutoregressiveMode  # noqa: PLC0415

  decode_mode = resolve_decode_mode(spec.run_spec, purpose="sample")
  if not isinstance(decode_mode, AutoregressiveMode):
    msg = (
      f"sample_multistate_poe_bead only implements AutoregressiveMode sampling, but "
      f"spec.sampling_strategy={spec.sampling_strategy!r} resolves to "
      f"{type(decode_mode).__name__} via the shared resolver. straight_through (STE) "
      f"sampling is not wired for genuine multi-state PoE beads -- only the default "
      f"autoregressive path is supported here."
    )
    raise NotImplementedError(msg)

  return sample_states_fused(model, bundle, config, stage_set, prng_key, n_samples)


def sample_multistate_poe_campaign_row(spec: SamplingSpecification) -> dict[str, Any]:
  """Execute one campaign manifest row as a genuine multi-state PoE bead.

  Real integration point for `host/campaign.py::run_manifest_row`: called INSTEAD of
  `host/runner.py::sample()` for rows where `len(spec.inputs) > 1` (a genuine multi-state PoE
  bead), so `aminx campaign run` produces real fused output for these rows rather than
  `_sample_batch`'s per-structure-independent output. Single-structure ("spike-in") rows are
  untouched -- they still go through `sample()`/`_sample_batch` exactly as before; this function
  is never called for them.

  Writes to `spec.run_spec.io.output_h5_path` via `xtrax.run.ZarrStagingSink`/`SinkSpec` -- the
  same Sink primitive `host/streaming.py::_sample_streaming` uses -- so
  `run_manifest_row`'s surrounding done-marker/lock/retry machinery (which only cares that
  `output_h5_path` gets populated and content-digested, not which function populated it) works
  completely unchanged. The written schema mirrors `_sample_streaming`'s campaign-mode attrs
  as closely as a genuinely fused (not per-structure) result allows: ONE Zarr group
  (`"poe_fused"`) replaces the `structure_0..structure_{k-1}` groups `_sample_streaming` would
  otherwise write for this many `--inputs`, since there is no longer "k independent structures"
  to store -- states are fused into one design series per sample.

  Resolves the real campaign chunk/sample-count/lineage axes via the SAME helpers
  `_sample_streaming` uses (`resolve_target_samples`/`resolve_chunk_size`/`resolve_sample_start`/
  `_resolve_grid_lineage`), so chunked resubmission (`samples_chunk_size`, necklace's real
  SAMPLES_CHUNK_SIZE=20 convention) and grid-lineage manifest-row-hash bookkeeping behave
  identically to every other row. Derives its base PRNG key via the same
  `_base_sampling_key(spec, grid_lineage=...)` helper (keyed off `spec.run_spec.sampling.
  random_seed` + grid lineage), matching this campaign's real determinism/reproducibility
  convention -- NOT bit-identical to `_sample_batch`'s own per-sample `compute_sample_keys`
  fold-in scheme (a deliberately separate code path, not meant to reproduce that scheme
  key-for-key; only the SEED derivation matches, not every downstream fold_in step).

  CRITICAL (found by independent PR audit, 2026-07-14): `_base_sampling_key`'s own hash
  (`_grid_job_seed_hash`) is keyed off `job_id` + strategy/conditioning fields -- it does NOT
  incorporate `chunk_id`/`sample_start`. A bead whose real design count exceeds
  `samples_chunk_size` (necklace's real SAMPLES_CHUNK_SIZE=20 -- i.e. any bead with more than
  20 designs, which is the real production target, not the current placeholder default) is
  split into MULTIPLE manifest rows sharing one `job_id` but different `chunk_id`/
  `sample_start` (`build_necklace_p2_manifest.py`'s real chunking). Without folding
  `sample_start` into the key, every chunk of the same job would derive the IDENTICAL base key
  and produce byte-identical duplicated samples -- this was caught before it could ship: this
  function explicitly folds `resolve_sample_start(grid_lineage)` into `base_key` below before
  any per-cell derivation, specifically so different chunks of the same job produce genuinely
  different (not duplicated) samples. Do not remove this fold_in as "redundant" without adding
  an equivalent safeguard -- see the regression test asserting two rows with the same job_id
  but different chunk_id/sample_start produce distinct sequences.

  Handles `spec.run_spec.sampling.temperature`/`backbone_noise` as real grid axes (each real
  necklace PoE row uses 5 temperatures x 1 noise level): loops over every (noise, temperature)
  combination, building a per-cell `SamplingSpecification` via `dataclasses.replace` (confirmed
  to correctly re-sync the nested `run_spec.sampling.{temperature,backbone_noise}` fields, not
  just the flat ones) and calling `sample_multistate_poe_bead` once per cell -- this means
  `prep_protein_stream_and_model` (the real PDB parsing/padding/chain-filtering step) reruns
  once per (noise, temperature) cell rather than once total. This is a known, deliberate
  correctness-first tradeoff, not an oversight: real necklace rows have 5 temperatures x 1
  noise = 5 redundant structure reloads (real PDB I/O + padding + chain filtering) per row --
  cheap relative to the AR sampling compute itself (an independent PR audit confirmed
  `sample_states_fused`'s `@eqx.filter_jit` sits on the OUTER function specifically so the
  compiled AR-sampling executable is traced ONCE and reused across all 5 cells' worth of
  structure-reload-then-sample calls, not retraced per cell -- only the cheap host-side
  reload repeats, not the expensive JIT compile). Refactoring to load structures once and
  reuse across cells is a real, tracked follow-up optimization if profiling ever shows this
  reload cost matters at production scale -- not applied here to keep this integration's
  first version small and easy to verify correct.

  Parameters
  ----------
  spec : SamplingSpecification
      Real campaign row spec (as `run_manifest_row` constructs via
      `SamplingSpecification(**worker_payload)`) -- must have `len(spec.inputs) > 1` and
      `spec.batch_size == len(spec.inputs)` (same precondition as `sample_multistate_poe_bead`).

  Returns
  -------
  dict
      Matches `sample()`'s streaming-mode return contract: `output_zarr_path`,
      `schema_version`, `metadata` (with `specification`, `skipped_inputs`, `structure_ids`,
      and `lineage` if grid_mode). `run_manifest_row` merges this into its own result payload
      exactly as it does for `sample()`'s return value.

  Raises
  ------
  ValueError
      If `spec.inputs` has fewer than 2 entries (same precondition as
      `sample_multistate_poe_bead`; this function should only ever be called for genuine
      multi-state rows, but re-asserts the precondition rather than trusting the caller).

  """
  if not isinstance(spec.inputs, (list, tuple)) or len(spec.inputs) < 2:
    msg = (
      f"sample_multistate_poe_campaign_row needs >=2 states in spec.inputs to fuse across "
      f"(got {spec.inputs!r}) -- single-structure rows must go through sample() instead."
    )
    raise ValueError(msg)

  # Resolved at RUN ENTRY, before the (expensive) AR sampling loop below -- not at sink
  # construction further down, which in this function happens AFTER that loop runs. A
  # PackageNotFoundError here fails before any compute happens, so it can never discard
  # already-completed sampling work the way resolving it post-loop would.
  resolved_aminx_version = resolve_aminx_version()

  grid_lineage = _resolve_grid_lineage(spec)
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  total_num_samples = resolve_target_samples(spec, grid_lineage=grid_lineage)
  chunk_size = resolve_chunk_size(spec, total_num_samples, grid_lineage)
  if total_num_samples != chunk_size:
    msg = (
      f"sample_multistate_poe_campaign_row assumes one manifest row produces exactly one "
      f"chunk (total_num_samples == chunk_size); got total_num_samples={total_num_samples}, "
      f"chunk_size={chunk_size}. Every real necklace manifest row sets samples_chunk_size == "
      "sample_count per row (build_necklace_p2_manifest.py's chunking splits a bead across "
      "MULTIPLE rows, not multiple chunks within one row) -- this row's manifest builder "
      "violated that convention. Multi-chunk-per-row accumulation is not implemented here; "
      "fix the manifest builder rather than silently truncating output to chunk_size."
    )
    raise ValueError(msg)

  # CRITICAL: fold sample_start into the base key -- _base_sampling_key's own hash does not
  # incorporate chunk_id/sample_start (it's keyed off job_id + strategy/conditioning fields
  # only), so without this, two manifest rows sharing one job_id but different chunk_id/
  # sample_start (exactly how a >samples_chunk_size bead gets split into multiple rows) would
  # derive the SAME base key and produce byte-identical duplicated samples. See this
  # function's docstring "CRITICAL" note and the two-rows-same-job regression test.
  sample_start = resolve_sample_start(grid_lineage)
  base_key = jax.random.fold_in(_base_sampling_key(spec, grid_lineage=grid_lineage), sample_start)

  temperature_val = spec.run_spec.sampling.temperature
  temperatures = (
    list(temperature_val) if isinstance(temperature_val, (list, tuple)) else [temperature_val]
  )
  noise_val = spec.run_spec.sampling.backbone_noise
  noises = list(noise_val) if isinstance(noise_val, (list, tuple)) else [noise_val]
  return_logits = spec.run_spec.sampling.return_logits

  sequence_rows: list[list[np.ndarray]] = []
  logits_rows: list[list[np.ndarray]] = []
  # AC1b: this loop is exactly the "sample/noise/temperature axes fused into one group"
  # case -- every cell below contributes to the SAME staged key ("poe_fused") below, so a
  # per-cell bias_attrs value that diverged across cells would silently resolve via
  # last-write-wins if not caught here. `cell_spec` never touches `bias` (only
  # `backbone_noise`/`temperature` are replaced per cell), so this is expected to always
  # be uniform -- the assertion after the loop is the drift detector for that invariant.
  cell_bias_attrs: list[dict[str, Any]] = []
  seq_len: int | None = None
  for noise_idx, noise in enumerate(noises):
    sequence_row: list[np.ndarray] = []
    logits_row: list[np.ndarray] = []
    for temp_idx, temperature in enumerate(temperatures):
      cell_key = jax.random.fold_in(jax.random.fold_in(base_key, noise_idx), temp_idx)
      cell_spec = dataclasses.replace(
        spec, backbone_noise=[noise], temperature=[temperature],
      )
      cell_sequences, cell_logits = sample_multistate_poe_bead(
        cell_spec, cell_key, n_samples=chunk_size,
      )
      cell_sequences_np = np.asarray(jax.device_get(cell_sequences), dtype=np.int32)
      cell_logits_np = np.asarray(jax.device_get(cell_logits), dtype=np.float32)
      if seq_len is None:
        seq_len = cell_sequences_np.shape[-1]
      sequence_row.append(cell_sequences_np)
      logits_row.append(cell_logits_np)
      if return_logits:
        _, cell_attrs = logits_bias_semantics_outputs(cell_spec.run_spec.sampling.bias)
        cell_bias_attrs.append(cell_attrs["logits_bias_semantics"])
    sequence_rows.append(sequence_row)
    logits_rows.append(logits_row)

  if cell_bias_attrs:
    assert_uniform_group_attr(
      cell_bias_attrs,
      attr_name="logits_bias_semantics",
      group_key=("poe_fused",),
    )

  # sequence_rows[noise_idx][temp_idx] has shape (chunk_size, L) -- stack into the
  # (chunk_size, num_noise, num_temperatures, L) convention _sample_streaming's campaign-mode
  # schema uses, so a reader (e.g. a future necklace_zarr_reader.py) sees the same axis order
  # regardless of whether a row is fused or per-structure.
  sequences_arr = np.stack(
    [np.stack(row, axis=1) for row in sequence_rows], axis=1,
  )  # (chunk_size, num_noise, num_temperatures, L)
  logits_arr = (
    np.stack([np.stack(row, axis=1) for row in logits_rows], axis=1)
    if return_logits
    else None
  )  # (chunk_size, num_noise, num_temperatures, L, 21)

  output_dir = Path(spec.run_spec.io.output_h5_path)
  sink = ZarrStagingSink(SinkSpec(output_dir=output_dir, format="zarr", flush_every=1))

  root_attrs: dict[str, Any] = {
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    # Hash-free marker (audit finding A) -- participates in NO hash, so unlike
    # `schema_version` it can be added without moving any manifest row hash or output path.
    "sink_provenance_version": SINK_PROVENANCE_VERSION,
    "model_family": spec.model_family,
    "ligand_conditioning": int(spec.ligand_conditioning),
    "sidechain_conditioning": int(spec.sidechain_conditioning),
    "samples_chunk_size": chunk_size,
    "multistate_poe_fused": 1,
    "aminx_version": resolved_aminx_version,
    **prng_seed_attrs(spec.run_spec.sampling.random_seed),
  }
  root_arrays: dict[str, np.ndarray] = {}
  bias_attrs: dict[str, Any] = {}
  if return_logits:
    bias_arrays, bias_attrs = logits_bias_semantics_outputs(spec.run_spec.sampling.bias)
    root_arrays.update(bias_arrays)
    root_attrs.update(bias_attrs)
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    root_attrs.update(_grid_lineage_attrs(grid_lineage))
    root_attrs["manifest_row_hash"] = manifest_row_hash
    # AUDIT FINDING C -- see the matching comment in `host/streaming.py`. This writer calls
    # `_base_sampling_key(spec, grid_lineage=grid_lineage)` directly (see its use above), so
    # it folds the SAME `_grid_job_seed_hash` into the key and has the identical gap:
    # `prng_seed` alone under-determines the key that actually drove sampling.
    root_attrs["grid_job_seed_hash"] = _grid_job_seed_hash(spec, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage, chunk_size=chunk_size,
    )
    # `.update(...)`, NOT reassignment -- a bare `root_arrays = {...}` here silently
    # discarded whatever `bias_arrays` staged above (audit finding, task_id
    # `260910_aminx-sink-provenance-schema`): `bias_persisted: true` would then be a lying
    # attr, since the `bias` array it claims exists never reached the store in grid mode.
    root_arrays.update({
      "sample_indices": _grid_sample_indices(grid_lineage),
      "grid_iteration_ids": iteration_ids,
      "grid_iteration_sample_start": iteration_starts,
      "grid_iteration_sample_count": iteration_counts,
    })
  sink.stage((), attrs=root_attrs, **root_arrays)

  key = ("poe_fused",)
  arrays: dict[str, np.ndarray] = {"sequences": sequences_arr}
  if logits_arr is not None:
    arrays["logits"] = logits_arr
  attrs: dict[str, Any] = {
    "structure_index": 0,
    "structure_id": "poe_fused",
    "fused_structure_ids": list(canonical_structure_ids),
    "num_samples": chunk_size,
    "num_noise_levels": len(noises),
    "num_temperatures": len(temperatures),
    "sequence_length": int(seq_len),
  }
  if grid_lineage is not None:
    attrs.update(_grid_lineage_attrs(grid_lineage))
  if logits_arr is not None:
    attrs.update(bias_attrs)
  # Same provenance as the single-structure branch. These two writers share no write function
  # -- they only share the sink primitive -- so anything added to one and not the other leaves
  # exactly the arm that matters here (the necklace's PoE beads) with no evidence at all.
  fixed_arrays, fixed_attrs = fixed_provenance_outputs(spec, seq_len=int(seq_len))
  arrays.update(fixed_arrays)
  attrs.update(fixed_attrs)
  sink.stage(key, attrs=attrs, **arrays)
  sink.drain()

  results: dict[str, Any] = {
    "output_zarr_path": str(output_dir),
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": [],
      "structure_ids": ["poe_fused"],
      "fused_structure_ids": list(canonical_structure_ids),
    },
  }
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage, chunk_size=chunk_size,
    )
    results["metadata"]["lineage"] = {
      **grid_lineage,
      "manifest_row_hash": manifest_row_hash,
      "sample_indices": _grid_sample_indices(grid_lineage).tolist(),
      "grid_iteration_ids": iteration_ids.tolist(),
      "grid_iteration_sample_start": iteration_starts.tolist(),
      "grid_iteration_sample_count": iteration_counts.tolist(),
    }
  return results
