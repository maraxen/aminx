"""Core user interface for the Aminx package."""

# COMP-NEW (2026-05-25): non-streaming path now drains via streaming_tensor_sink_session (§14 resolved).

from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp

from aminx.host._sampling_grid_lineage import (
  _grid_iteration_arrays,
  _grid_manifest_row_hash,
  _grid_sample_indices,
  _resolve_grid_lineage,
)
from aminx.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _structure_ids_for_batch,
)
from aminx.host.kernel_dispatch import _sample_batch
from aminx.host.logit_aggregation import (
  aggregate_logits,
  aggregate_pseudo_perplexities,
  compute_logit_fingerprint,
  pad_to_max,
)
from aminx.host.output_sinks import (
  streaming_tensor_sink_session,
  take_staging_sequences_logits,
)
from aminx.host.plan import (
  InferencePlan,
  make_inference_plan,
  resolve_chunk_size,
  resolve_target_samples,
)
from aminx.host.streaming import (
  GRID_SCHEMA_VERSION,
  SAMPLING_SCHEMA_VERSION,
  _sample_streaming,
)
from aminx.host.streaming_host import StreamingBatchHost
from aminx.io.sink_provenance import prng_seed_attrs, resolve_aminx_version
from aminx.run.batch_mapping import MappedBy
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  SamplingSpecification,
  ScoringSpecification,
  pop_deprecated_spec_kwargs,
)

from .prep import prep_protein_stream_and_model


def _fixed_mask_has_fixed_positions(fixed_mask: object) -> bool:
  """True if ``fixed_mask`` declares any fixed position -- MappedBy is conservatively True.

  A ``MappedBy`` can't be cheaply checked for "all zero" without the per-batch structure
  identity resolution machinery this guard runs before (score/jacobian don't support
  fixed_mask at all, so there is nothing to resolve against) -- treat any MappedBy as
  "has fixed positions set" rather than trying to peek inside its mapping.
  """
  if isinstance(fixed_mask, MappedBy):
    return True
  return bool(jnp.any(jnp.asarray(fixed_mask)))


logger = logging.getLogger(__name__)
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Sample new sequences for given input structures using high-performance Grain pipeline.

  Routes to streaming or in-memory sampling based on output configuration and averaging mode.
  Loads structures via Grain iterator, prepares model, processes in batches, and aggregates
  results with optional logits and pseudo-perplexity calculations.

  Parameters
  ----------
  spec : SamplingSpecification, optional
      SamplingSpecification object configuring inputs, model, and sampling strategy.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for SamplingSpecification if spec is None. Options include:
        inputs : str or list[str]
            Input structures (file paths, PDB IDs, etc.).
        chain_id : str, optional
            Specific chain(s) to parse.
        model : int, optional
            Model number to load. If None, all models used.
        altloc : str, optional
            Alternate location identifier.
        model_version : str, optional
            Model version (e.g., 'proteinmpnn_v1').
        model_weights : str, optional
            Path to model weights.
        foldcomp_database : str, optional
            FoldComp database for compressed structures.
        random_seed : int, optional
            Random seed for sampling.
        backbone_noise : float, optional
            Noise added to backbone coordinates.
        num_samples : int, optional
            Number of sequences per structure/noise level.
        sampling_strategy : str, optional
            Sampling strategy (e.g., 'autoregressive', 'straight_through').
        temperature : float, optional
            Sampling temperature.
        bias : jax.Array, optional
            Per-position or per-position-per-token logit bias. Shape: (L,) or (L, 21).
        fixed_positions : list[int], optional
            Residue indices to keep fixed.
        iterations : int, optional
            Optimization iterations for 'straight_through' strategy.
        learning_rate : float, optional
            Learning rate for 'straight_through' strategy.
        batch_size : int, optional
            Number of structures per batch.
        return_logits : bool, optional
            Whether to return logits for each position.
        output_h5_path : str, optional
            Path to streaming H5 output file.
        grid_mode : bool, optional
            Whether in grid sampling mode (affects metadata lineage).

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        sequences : jax.Array
            Sampled sequences, shape (B*N, L) where B is batch count,
            N is num_samples per structure, L is sequence length.
            Padded to max length across all sequences, masked with 'mask' key.
        mask : jax.Array
            Sequence validity mask (1 for valid, 0 for padding). Shape: (B*N, L).
        schema_version : str
            Schema version for results ('grid_v1' or 'sampling_v1').
        metadata : dict
            Metadata including specification, skipped_inputs, structure_ids,
            and optional lineage info (grid mode).
        logits : jax.Array, optional
            Per-position logits if return_logits=True. Shape: (B*N, L, 21).
        pseudo_perplexity : jax.Array, optional
            Per-sequence pseudo-perplexity scores if computed.

  Notes
  -----
  Streaming vs. in-memory: If output_h5_path is set, uses streaming I/O.
  Otherwise, concatenates per-batch results in memory. See _sample_streaming
  in streaming.py for streaming mode details.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = SamplingSpecification(**kw)

  # F002/F003 guard [260826_aminx-invariant-audit]: runner.sample cannot honour
  # multistate spec fields.  score() routes through _score_fused_multistate which
  # performs real cross-state fusion; sample() has no such path (the internal
  # _sample_batch kernel silently discards multi_state_strategy via `del` and never
  # reads state_position_map).  Silently returning per-structure single-state output
  # when the caller supplied multistate fields is a lie; raise loudly instead and
  # direct callers to the campaign verbs that do implement multistate sampling.
  _multistate_default = "arithmetic_mean"
  if spec.state_position_map is not None or spec.multi_state_strategy != _multistate_default:
    _bad_fields = []
    if spec.state_position_map is not None:
      _bad_fields.append(f"state_position_map (shape {getattr(spec.state_position_map, 'shape', type(spec.state_position_map))})")
    if spec.multi_state_strategy != _multistate_default:
      _bad_fields.append(f"multi_state_strategy={spec.multi_state_strategy!r}")
    msg = (
      f"runner.sample() does not implement multistate sampling; the following "
      f"spec fields cannot be honoured: {', '.join(_bad_fields)}. "
      f"Use `aminx campaign run` or the campaign worker verb to reach the "
      f"multistate sampling path (aminx.sampling.multistate_poe."
      f"sample_multistate_poe_campaign_row). "
      f"runner.score() does honour these fields via _score_fused_multistate."
    )
    raise NotImplementedError(msg)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  # Construct inference plan once before routing to streaming or non-streaming path.
  # purpose="sample": this is the one caller that wants new sequences, not a score of a given
  # one -- aminx#110 was this exact resolution defaulting to ConditionalMode (score-shaped)
  # here with no way to ask for real autoregressive sampling.
  plan: InferencePlan = make_inference_plan(model, spec, purpose="sample")

  if spec.run_spec.io.output_h5_path:
    return _sample_streaming(spec, protein_iterator, plan, _sample_batch)

  # Non-streaming path uses io_callback staging via streaming_tensor_sink_session;
  # drains per-batch via take_staging_sequences_logits.
  all_sequences, all_pseudo_perplexities = [], []
  needs_logits = spec.run_spec.sampling.return_logits or spec.run_spec.sampling.return_logit_fingerprint
  all_logits = [] if needs_logits else None
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)
  grid_lineage = _resolve_grid_lineage(spec)

  with streaming_tensor_sink_session():
    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_offset,
        batch_size=batch_size,
      )
      target_for_batch = resolve_target_samples(spec, None, grid_lineage)
      _, _, pseudo_perplexity = _sample_batch(
        spec,
        batched_ensemble,
        plan,
        canonical_structure_ids=canonical_structure_ids,
        batch_structure_ids=batch_structure_ids,
        batch_idx=batch_idx,
        structure_batch_count=structure_batch_count,
      )
      StreamingBatchHost.sink_barrier()
      sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
        batch_idx,
        0,
        target_for_batch,
      )
      all_sequences.append(jnp.asarray(sampled_sequences_np))
      if needs_logits and all_logits is not None:
        all_logits.append(jnp.asarray(sampled_logits_np))
      if pseudo_perplexity is not None:
        all_pseudo_perplexities.append(pseudo_perplexity)
      resolved_structure_ids.extend(batch_structure_ids)
      structure_offset += batch_size
  max_len = max(arr.shape[-1] for arr in all_sequences)

  all_sequences_padded = [pad_to_max(seq, max_len, axis=-1, pad_value=0) for seq in all_sequences]

  all_masks = [
    pad_to_max(
      jnp.ones(seq.shape, dtype=jnp.int32),
      max_len,
      axis=-1,
      pad_value=0,
    )
    for seq in all_sequences
  ]

  results: dict[str, Any] = {
    "sequences": jnp.concatenate(all_sequences_padded, axis=0),
    "mask": jnp.concatenate(all_masks, axis=0),
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }
  aggregated_logits = aggregate_logits(all_logits, max_len) if needs_logits and all_logits is not None else None
  if spec.run_spec.sampling.return_logits and aggregated_logits is not None:
    results["logits"] = aggregated_logits
  if spec.run_spec.sampling.return_logit_fingerprint and aggregated_logits is not None:
    fingerprint_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed)
    results["logit_fingerprint"] = compute_logit_fingerprint(aggregated_logits, fingerprint_key)
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = aggregate_pseudo_perplexities(all_pseudo_perplexities)

  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    total_num_samples_for_grid = resolve_target_samples(spec, grid_lineage=grid_lineage)
    chunk_size_for_grid = resolve_chunk_size(spec, total_num_samples_for_grid, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size_for_grid,
    )
    sample_indices = _grid_sample_indices(grid_lineage)
    results["sample_indices"] = jnp.asarray(sample_indices, dtype=jnp.int32)
    results["metadata"]["lineage"] = {
      **grid_lineage,
      "manifest_row_hash": manifest_row_hash,
      "sample_indices": sample_indices.tolist(),
      "grid_iteration_ids": iteration_ids.tolist(),
      "grid_iteration_sample_start": iteration_starts.tolist(),
      "grid_iteration_sample_count": iteration_counts.tolist(),
    }

  return results


SCORING_SCHEMA_VERSION = "scoring_v1"
INSPECTION_SCHEMA_VERSION = "inspection_v1"
JACOBIAN_SCHEMA_VERSION = "jacobian_v1"


def _make_averaged_score_fn(
  plan: Any,  # noqa: ANN401
  spec: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
  """Create a scoring function that averages over backbone noise levels.

  Builds D InferenceBundle objects from spec.backbone_noise, encodes each,
  fuses node/edge features via plan.stage_set.encoding_fusion, then decodes
  once to produce logits. Uses _nll_from_logits for the NLL calculation.

  Args:
    plan: InferencePlan with encode, decode, stage_set, and encoding_fusion.
    spec: ScoringSpecification with backbone_noise field.

  Returns:
    A scoring function with signature matching make_score_fn output.

  Raises:
    AssertionError: If bundles' conditioning fields are not noise-invariant.
  """
  from functools import partial  # noqa: PLC0415

  import equinox as eqx  # noqa: PLC0415

  from aminx.inference.bundle_builder import build_inference_bundle  # noqa: PLC0415
  from aminx.inference.score_conditional import score_averaged  # noqa: PLC0415
  from aminx.scoring.score import _nll_from_logits  # noqa: PLC0415
  from aminx.utils.decoding_order import random_decoding_order  # noqa: PLC0415

  def _check_r3_invariance(bundles_per_noise: list) -> None:
    """Check R3: all D bundles must share conditioning fields; only backbone_noise may vary.

    Must be called on concrete (non-JIT-traced) bundles so eqx.tree_equal returns a
    Python bool rather than an abstract tracer. The R3 check is moved here from the
    JIT closure to guarantee it fires correctly on real values.
    """
    ref = bundles_per_noise[0]
    for _d, _bnd in enumerate(bundles_per_noise[1:], start=1):
      for _attr in ("conditioning", "geometry", "ligand", "wave"):
        result = eqx.tree_equal(getattr(ref, _attr), getattr(_bnd, _attr))
        # eqx.tree_equal returns True/False/None (or a concrete array) on real values.
        # Cast to bool — if it's a JAX scalar, bool() concretizes it safely here (outside JIT).
        if not bool(result):
          msg = (
            f"_make_averaged_score_fn: bundle[{_d}].{_attr} differs from bundle[0].{_attr}. "
            f"D bundles must share all conditioning fields; only backbone_noise may vary. "
            f"(R3 invariant)"
          )
          raise AssertionError(msg)

  @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state"))
  def _score_averaged_jit(
    prng_key: jax.Array,
    bundles_per_noise: list,
    config_out: Any,  # noqa: ANN401
    sequence: jax.Array,
    mask: jax.Array,
    multi_state_strategy: str = "arithmetic_mean",
    use_rolling_state: bool = False,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled core: encode, fuse, decode. Receives pre-built concrete bundles."""
    del use_rolling_state, multi_state_strategy

    L = sequence.shape[0]
    decoding_order, _key = random_decoding_order(prng_key, L, None, None)

    # Encode at each noise level, fuse, decode
    logits = score_averaged(
      plan.model,
      prng_key,
      bundles_per_noise,
      config_out,
      plan.stage_set,
      plan.stage_set.encoding_fusion,
    )

    # Compute NLL using the extracted helper
    nll = _nll_from_logits(logits, sequence, mask)

    return nll, logits, decoding_order

  def score_sequence_averaged(
    prng_key: jax.Array,
    sequence: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    backbone_noise: float | None = None,
    ar_mask_override: jax.Array | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: str = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
    state_weights: jax.Array | None = None,
    state_position_map: jax.Array | None = None,
    bias: jax.Array | None = None,
    use_rolling_state: bool = False,
    ligand_coords: jax.Array | None = None,
    ligand_atom_types: jax.Array | None = None,
    ligand_mask: jax.Array | None = None,
    **kwargs: Any,  # noqa: ANN401
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Python-level wrapper: build bundles, check R3 on concrete arrays, call JIT core."""
    del (
      multi_state_temperature,
      backbone_noise,
    )  # backbone_noise is overridden by spec.backbone_noise
    del kwargs

    L = int(sequence.shape[0])

    # Build decoding AR mask: use provided override or default full-context mask. This is a
    # direct caller-supplied override, not sourced from any RunSpecification field -- aminx#113
    # confirmed the spec-level `ar_mask` field (now deleted) never fed anything; this parameter
    # is unrelated and always was.
    if ar_mask_override is not None:
      ar_mask_single = ar_mask_override[0] if ar_mask_override.ndim == 3 else ar_mask_override
    else:
      ar_mask_single = None  # bundle_builder will create the default

    # Build D bundles on concrete (Python-level) arrays — one per backbone_noise level.
    # This runs OUTSIDE the JIT boundary so all arrays are concrete.
    backbone_noises = spec.run_spec.sampling.backbone_noise or (0.0,)
    bundles_per_noise = []
    config_out = None
    for noise_val in backbone_noises:
      bundle, config_out = build_inference_bundle(
        coords=structure_coordinates,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        backbone_noise=float(noise_val),
        ar_mask=ar_mask_single,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        state_weights=state_weights,
        state_position_map=state_position_map,
        bias=bias,
        ligand_coords=ligand_coords,
        ligand_atom_types=ligand_atom_types,
        ligand_mask=ligand_mask,
        mode="score_conditional",
        inference=True,
      )
      bundles_per_noise.append(bundle)

    # R3 assertion: runs on concrete arrays (outside JIT) so eqx.tree_equal returns
    # a Python bool, not an abstract tracer. This is the safe location for this check.
    if len(bundles_per_noise) > 1:
      _check_r3_invariance(bundles_per_noise)

    # Delegate to JIT-compiled core
    del L  # unused after bundle build
    return _score_averaged_jit(
      prng_key,
      bundles_per_noise,
      cast("Any", config_out),
      sequence,
      mask,
      multi_state_strategy=multi_state_strategy,
      use_rolling_state=use_rolling_state,
    )

  return score_sequence_averaged


def score(  # noqa: PLR0915
  spec: ScoringSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Score sequences against input structures.

  Loads structures via Grain iterator, prepares model, and computes negative
  log-likelihood (NLL) scores for provided sequences. Optionally returns logits
  and decoding orders.

  Parameters
  ----------
  spec : ScoringSpecification, optional
      ScoringSpecification object configuring inputs, sequences, and scoring options.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for ScoringSpecification if spec is None.

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        scores : jax.Array
            Negative log-likelihood scores. Shape: (num_structures, num_sequences).
        mask : jax.Array
            Structure validity mask (1 for valid, 0 for padding). Shape: (num_structures,).
        schema_version : str
            Schema version for results ('scoring_v1').
        metadata : dict
            Metadata including specification, structure_ids, and skipped_inputs.
        logits : jax.Array, optional
            Per-position logits if return_logits=True. Shape: (num_structures, num_sequences, L, 21).
        decoding_orders : jax.Array, optional
            Decoding orders if return_decoding_orders=True. Shape: (num_structures, num_sequences, L).

  Raises
  ------
  NotImplementedError
      If output_h5_path is set (HDF5 streaming not yet implemented for scoring).
  ValueError
      If sequence length doesn't match structure length.

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = ScoringSpecification(**kw)

  if spec.output_h5_path:
    msg = "score runner: HDF5 streaming output not yet implemented; omit --output-h5-path for in-memory results"
    raise NotImplementedError(msg)

  # spec.fixed_mask -- audit 260826_chain-selection-vendor-superset-audit finding FA2.
  # aminx.scoring.score.score_sequence has no fixed_mask/chain_mask parameter and never
  # reads it; confirmed via a differential probe (fixed_mask=all-ones vs all-zeros on an
  # identical sequences_to_score input produced bit-identical scores). Silently accepting
  # and ignoring it would let a caller believe the returned NLL excludes or otherwise
  # accounts for the positions they marked fixed, when it does not -- score() already
  # deliberately scores every position under full context (see score_sequence's ar_mask
  # comment), so there is no established semantics here to guess at silently.
  if spec.fixed_mask is not None and _fixed_mask_has_fixed_positions(spec.fixed_mask):
    msg = (
      "score runner: spec.fixed_mask has one or more fixed positions set, but "
      "aminx.scoring.score.score_sequence has no parameter for it and never reads it. "
      "runner.sample honours fixed_mask (via host/_sampling_helper.py's "
      "_prepare_fixed_controls); this surface does not yet. Omit fixed_mask when calling "
      "score(), or use runner.sample if you need fixed-position-aware behavior."
    )
    raise NotImplementedError(msg)

  from aminx.scoring.score import make_score_fn  # noqa: PLC0415
  from aminx.utils.aa_convert import string_to_protein_sequence  # noqa: PLC0415

  protein_iterator, model = prep_protein_stream_and_model(spec)

  # Build score function: standard or averaged-feature
  if spec.average_node_features:
    from aminx.host.plan import make_inference_plan  # noqa: PLC0415

    # purpose="score" (the default): this call evaluates a given sequence, it never samples one.
    plan = make_inference_plan(model, spec, purpose="score")
    score_fn = _make_averaged_score_fn(plan, spec)  # type: ignore[arg-type]
  else:
    score_fn = make_score_fn(model)  # type: ignore[arg-type]

  # Convert string sequences to integer indices
  sequence_indices_list = []
  for seq_str in spec.sequences_to_score:
    seq_idx = string_to_protein_sequence(seq_str)
    sequence_indices_list.append(seq_idx)

  # spec.state_position_map's ONLY sensible purpose is cross-state residue alignment
  # for genuine multi-state PoE fusion -- its presence is an unambiguous signal the
  # caller wants ONE fused score across every spec.inputs structure, not N independent
  # per-structure scores (the loop below, unchanged, for the state_position_map=None
  # default -- preserves exact prior behavior for every existing caller that scores a
  # sequence against several unrelated candidate structures). Previously
  # state_position_map (and multi_state_strategy, forwarded per-structure into a
  # call site where S was always 1) were silently inert here -- confirmed via a live
  # differential (arithmetic_mean vs product produced byte-identical logits) before
  # this fix, not assumed from a source read alone.
  if spec.state_position_map is not None:
    return _score_fused_multistate(
      spec, protein_iterator, score_fn, sequence_indices_list,
    )

  from aminx.sampling.conditional_logits import _plan_axis_strategy  # noqa: PLC0415
  from aminx.tiling.axes import N_CANDIDATES  # noqa: PLC0415
  from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax  # noqa: PLC0415

  all_scores, all_logits, all_decoding_orders = [], None, None
  if spec.return_logits:
    all_logits = []
  if spec.return_decoding_orders:
    all_decoding_orders = []

  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0

  # Prepare random key
  prng_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed or 42)

  # Structures within one Grain batch are already stacked to a common length by the
  # loader (batched_ensemble.coordinates is a genuine (batch_size, L, 4, 3) array) --
  # both the structure axis and the candidate (sequences-to-score) axis are therefore
  # homogeneous per batch and dispatched via jax.vmap / aminx's own N_CANDIDATES
  # BatchPlanner composition, never a hand-rolled Python loop over the jitted
  # score_fn. Only the OUTER loop over Grain batches remains a plain Python loop --
  # that is host-side streaming-iterator drainage, not JAX computation, the same
  # exemption generate_multistate_conditional_logits.py's own docstring gives its
  # result-writing loop.
  for _batch_idx, batched_ensemble in enumerate(protein_iterator):
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )
    struct_len = batched_ensemble.coordinates.shape[1]

    padded_seqs = []
    for seq_idx in sequence_indices_list:
      # The structure may be padded to max_length by the loader; the user
      # sequence must fit within the (padded) structure length.
      if seq_idx.shape[0] > struct_len:
        msg = (
          f"Sequence too long for batch starting at structure "
          f"{batch_structure_ids[0]}: structure has {struct_len} residues, but "
          f"sequence has {seq_idx.shape[0]} residues"
        )
        raise ValueError(msg)
      # Pad the sequence to the structure length with X (index 20). Padded positions
      # contribute 0 to the NLL because the structure MASK is 0 there, which drops them
      # from both the numerator and the denominator of the masked mean.
      #
      # This comment used to credit the ``[..., :20]`` vocabulary slice in
      # ``_nll_from_logits`` instead. That was wrong, and the wrong attribution kept a real
      # defect alive: the mask alone already handled padding (verified -- the denominator
      # is 214 for a 214-residue chain at max_length=512), so the slice's only live effect
      # was to charge 0 nats for a *genuine* X residue while still counting it in the
      # denominator. The slice is gone; do not reintroduce it to "handle padding".
      if seq_idx.shape[0] < struct_len:
        pad = jnp.full((struct_len - seq_idx.shape[0],), 20, dtype=seq_idx.dtype)
        seq_idx = jnp.concatenate([seq_idx, pad])  # noqa: PLW2901
      padded_seqs.append(seq_idx)
    stacked_sequences = jnp.stack(padded_seqs, axis=0)  # (C, struct_len)
    n_candidates = stacked_sequences.shape[0]

    # Pre-derive one key per (structure, candidate) pair via the SAME sequential
    # jax.random.split chain the pre-refactor loop used (struct outer, candidate
    # inner) -- a single `split(prng_key, n)` in one shot is a different derivation
    # tree than n chained `prng_key, subkey = split(prng_key)` calls, so reusing one
    # batch_key across every structure in a batch (the first version of this fix)
    # silently changed the sampled decoding order -- and hence the NLL -- for any
    # batch with more than one structure. Splitting here is cheap (host-side PRNG
    # ops, no model call) so this loop isn't the composability violation the
    # candidate/structure axes below are dispatched via vmap/xtrax to avoid.
    n_keys_needed = batch_size * n_candidates
    flat_keys = []
    for _ in range(n_keys_needed):
      prng_key, subkey = jax.random.split(prng_key)
      flat_keys.append(subkey)
    batch_keys = jnp.stack(flat_keys, axis=0).reshape(batch_size, n_candidates, -1)

    activation_bytes = struct_len * 21 * 4  # (L, 21) float32 logits per candidate
    strategy = _plan_axis_strategy(
      N_CANDIDATES, n_candidates, None, activation_bytes_per_element=activation_bytes,
    )
    candidate_iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_CANDIDATES.name)

    def _score_structure(
      struct_coords: jax.Array,
      struct_mask: jax.Array,
      struct_residue_index: jax.Array,
      struct_chain_index: jax.Array,
      struct_keys: jax.Array,
      _candidate_iterator: Any = candidate_iterator,  # noqa: ANN401
      _stacked_sequences: jax.Array = stacked_sequences,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      def _score_one_candidate(
        item: dict[str, jax.Array],
      ) -> tuple[jax.Array, jax.Array, jax.Array]:
        seq_one_hot = jax.nn.one_hot(item["seq"], 21)
        return score_fn(  # type: ignore[misc]
          item["key"],
          seq_one_hot,
          struct_coords,
          struct_mask,
          struct_residue_index,
          struct_chain_index,
          multi_state_strategy=spec.multi_state_strategy,
        )

      return _candidate_iterator(
        _score_one_candidate, {"key": struct_keys, "seq": _stacked_sequences},
      )

    batch_scores, batch_logits, batch_decoding_orders = jax.vmap(_score_structure)(
      batched_ensemble.coordinates,
      batched_ensemble.mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      batch_keys,
    )

    all_scores.append(batch_scores)
    if spec.run_spec.sampling.return_logits and all_logits is not None:
      all_logits.append(batch_logits)
    if spec.run_spec.sampling.return_decoding_orders and all_decoding_orders is not None:
      all_decoding_orders.append(batch_decoding_orders)

    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  # Concatenate results: shape (num_structures, num_sequences)
  scores_array = jnp.concatenate(all_scores, axis=0)

  results = {
    "scores": scores_array,
    "mask": jnp.ones(scores_array.shape[0], dtype=jnp.int32),
    "schema_version": SCORING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }

  if spec.run_spec.sampling.return_logits and all_logits is not None:
    # Shape: (num_structures, num_sequences, L, 21)
    results["logits"] = jnp.concatenate(all_logits, axis=0)

  if spec.run_spec.sampling.return_decoding_orders and all_decoding_orders is not None:
    # Shape: (num_structures, num_sequences, L)
    results["decoding_orders"] = jnp.concatenate(all_decoding_orders, axis=0)

  return results


def _score_fused_multistate(
  spec: ScoringSpecification,
  protein_iterator: Any,  # noqa: ANN401
  score_fn: Any,  # noqa: ANN401
  sequence_indices_list: list[jax.Array],
) -> dict[str, Any]:
  """Score sequences against ONE genuinely fused multi-state ensemble.

  Every structure loaded from `spec.inputs` is treated as a single state of one PoE
  bead (matching the sampling path's `sample_multistate_poe_bead` convention) and
  stacked into a real leading-S bundle, fused via `spec.multi_state_strategy` and
  aligned via `spec.state_position_map` -- this is what `score()`'s default
  independent-per-structure loop cannot do (S is always 1 at that call site, so
  `multi_state_strategy` has nothing to fuse and `state_position_map` is never even
  read). Drains the ENTIRE iterator first (not just one batch) so this is correct
  even if a caller sets a small `--batch-size` that would otherwise split the
  ensemble's states across multiple iterator batches.

  Returns a `scores`/`logits` shaped with a leading dimension of 1 (one fused
  "structure"), not `len(spec.inputs)` independent ones -- the semantics are
  genuinely different from the default branch's per-structure output, by design.
  """
  # Draining the Grain protein_iterator into host-side lists is not the "hand-rolled
  # loop over JAX computation" this project's composability convention forbids -- no
  # JAX op runs until the single jnp.stack below; this is host-side data collection
  # before entering JAX, the same exemption generate_multistate_conditional_logits.py's
  # own docstring gives its result-writing loop.
  all_coords, all_mask, all_residue_index, all_chain_index = [], [], [], []
  structure_ids: list[str] = []
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  structure_offset = 0

  for batched_ensemble in protein_iterator:
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids, structure_offset=structure_offset, batch_size=batch_size,
    )
    for struct_idx in range(batch_size):
      all_coords.append(batched_ensemble.coordinates[struct_idx])
      all_mask.append(batched_ensemble.mask[struct_idx])
      all_residue_index.append(batched_ensemble.residue_index[struct_idx])
      all_chain_index.append(batched_ensemble.chain_index[struct_idx])
    structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  n_states = len(all_coords)
  state_position_map = jnp.asarray(spec.state_position_map)
  if state_position_map.shape[0] != n_states:
    msg = (
      f"spec.state_position_map has {state_position_map.shape[0]} states but "
      f"{n_states} structures were loaded from spec.inputs -- these must match for "
      "genuine multi-state fusion. If you did not intend fusion, leave "
      "state_position_map unset (the default per-structure independent-scoring path "
      "is unaffected by this branch)."
    )
    raise ValueError(msg)

  stacked_coords = jnp.stack(all_coords, axis=0)  # (S, L, 4, 3)
  stacked_mask = jnp.stack(all_mask, axis=0)  # (S, L)
  stacked_residue_index = jnp.stack(all_residue_index, axis=0)
  stacked_chain_index = jnp.stack(all_chain_index, axis=0)
  struct_len = stacked_coords.shape[1]

  # Pad every sequence to struct_len (host-side, static shapes, no JAX computation
  # yet) so the candidate axis below is genuinely homogeneous -- a precondition for
  # N_CANDIDATES.heterogeneous=False and for stacking into one array to dispatch over.
  padded_seqs = []
  for seq_idx in sequence_indices_list:
    if seq_idx.shape[0] > struct_len:
      msg = (
        f"Sequence too long for the fused ensemble: structures have {struct_len} "
        f"residues, but sequence has {seq_idx.shape[0]} residues"
      )
      raise ValueError(msg)
    if seq_idx.shape[0] < struct_len:
      pad = jnp.full((struct_len - seq_idx.shape[0],), 20, dtype=seq_idx.dtype)
      seq_idx = jnp.concatenate([seq_idx, pad])  # noqa: PLW2901
    padded_seqs.append(seq_idx)
  stacked_sequences = jnp.stack(padded_seqs, axis=0)  # (C, struct_len)
  n_candidates = stacked_sequences.shape[0]
  candidate_keys = jax.random.split(
    jax.random.PRNGKey(spec.run_spec.sampling.random_seed or 42), n_candidates,
  )

  # Candidate (sequences-to-score) axis dispatched via aminx's own BatchPlanner ->
  # Vmap/SafeMap composition (never a hand-rolled Python loop over a jitted call) --
  # the same N_CANDIDATES primitive make_batched_conditional_logits_split_fn's
  # batched_decode_fn already uses for an analogous "many candidates, one structure
  # ensemble" dispatch.
  from aminx.sampling.conditional_logits import _plan_axis_strategy  # noqa: PLC0415
  from aminx.tiling.axes import N_CANDIDATES  # noqa: PLC0415
  from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax  # noqa: PLC0415

  activation_bytes = struct_len * 21 * 4  # (L, 21) float32 logits per candidate
  strategy = _plan_axis_strategy(
    N_CANDIDATES, n_candidates, None, activation_bytes_per_element=activation_bytes,
  )
  candidate_iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_CANDIDATES.name)

  def _score_one_candidate(item: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
    seq_one_hot = jax.nn.one_hot(item["seq"], 21)
    return score_fn(
      item["key"],
      seq_one_hot,
      stacked_coords,
      stacked_mask,
      stacked_residue_index,
      stacked_chain_index,
      state_position_map=state_position_map,
      multi_state_strategy=spec.multi_state_strategy,
      multi_state_temperature=spec.multi_state_temperature,
    )

  all_scores, all_logits, all_decoding_orders = candidate_iterator(
    _score_one_candidate, {"key": candidate_keys, "seq": stacked_sequences},
  )

  # Leading dim of 1: ONE fused "structure", not len(spec.inputs) independent ones.
  results: dict[str, Any] = {
    "scores": all_scores[None, :],
    "mask": jnp.ones(1, dtype=jnp.int32),
    "schema_version": SCORING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": ["fused_multistate"],
      "fused_structure_ids": structure_ids,
    },
  }
  if spec.return_logits:
    results["logits"] = all_logits[None, :]
  if spec.return_decoding_orders:
    results["decoding_orders"] = all_decoding_orders[None, :]

  return results


def inspect(  # noqa: PLR0915
  spec: InspectionSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Inspect model encodings and features for input structures.

  Computes requested features (unconditional logits, encoded node features, etc.)
  for each input structure. Features are collected per-structure in the result dict.

  Parameters
  ----------
  spec : InspectionSpecification, optional
      InspectionSpecification object configuring inputs and inspection options.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for InspectionSpecification if spec is None.

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        {feature_name}: list or jax.Array
            Per-feature results. Each feature is a list of arrays, one per structure.
        mask : jax.Array
            Structure validity mask. Shape: (num_structures,).
        schema_version : str
            Schema version for results ('inspection_v1').
        metadata : dict
            Metadata including specification, structure_ids, and skipped_inputs.

  Raises
  ------
  NotImplementedError
      If output_h5_path is set (HDF5 streaming not yet implemented for inspect).
  ValueError
      If conditional_logits or decoded_node_features are requested without sequence data.

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = InspectionSpecification(**kw)

  # F005 guard [260826_aminx-invariant-audit]: runner.inspect cannot honour
  # spec.state_position_map -- no code path in this body reads it (AST hit
  # count 0; see findings.jsonl F005 evidence), so a caller-supplied map would
  # be silently discarded and the output would be single-state while stamped
  # as multistate. Raise loudly instead, mirroring the F002 guard at
  # runner.sample (7460516a); score() honours the field via
  # _score_fused_multistate and campaign verbs implement real state fusion.
  if getattr(spec, "state_position_map", None) is not None:
    msg = (
      "inspect runner: spec.state_position_map is set but runner.inspect has no "
      "cross-state fusion path -- the field would be silently discarded. Use "
      "runner.score (which honours state_position_map via _score_fused_multistate) "
      "or the campaign verbs for genuine multistate output."
    )
    raise NotImplementedError(msg)

  if spec.output_h5_path:
    msg = "inspect runner: HDF5 streaming output not yet implemented; omit --output-h5-path for in-memory results"
    raise NotImplementedError(msg)

  from aminx.inference.bundle_builder import build_inference_bundle  # noqa: PLC0415
  from aminx.inference.score_unconditional import kernel as score_unconditional  # noqa: PLC0415
  from aminx.sampling.conditional_logits import (  # noqa: PLC0415
    make_batched_conditional_logits_split_fn,
    make_conditional_logits_fn,
    make_encoding_conditional_logits_split_fn,
  )
  from aminx.utils.autoregression import full_context_ar_mask  # noqa: PLC0415
  from aminx.utils.structure_metrics import (  # noqa: PLC0415
    _extract_ca_coordinates,
    calculate_ca_distance_matrix,
    calculate_cb_distance_matrix,
    calculate_closest_atom_distance_matrix,
    calculate_cosine_similarity,
    calculate_rmsd,
    calculate_tm_score,
  )

  protein_iterator, model = prep_protein_stream_and_model(spec)

  # Stage set flows from the inference plan (single construction) rather than a
  # direct make_stage_set import — keeps runner.py free of make_stage_set (COMP-534).
  # purpose="score": only .stage_set is used below, decode_fn is never invoked from this
  # plan, but "score" is still the semantically correct purpose for an inspection call.
  unconditional_stage_set = make_inference_plan(model, spec, purpose="score").stage_set

  results_per_feature = {feat: [] for feat in spec.inspection_features}
  distance_matrices: list[jax.Array] = []
  ca_coords_for_similarity: list[jax.Array] = []

  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0

  # Prepare random key
  prng_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed or 42)

  for _batch_idx, batched_ensemble in enumerate(protein_iterator):
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )

    for struct_idx in range(batch_size):
      struct_coords = batched_ensemble.coordinates[struct_idx]
      struct_mask = batched_ensemble.mask[struct_idx]
      struct_residue_index = batched_ensemble.residue_index[struct_idx]
      struct_chain_index = batched_ensemble.chain_index[struct_idx]

      prng_key, subkey = jax.random.split(prng_key)

      for feature_name in spec.inspection_features:
        if feature_name == "unconditional_logits":
          # Build inference bundle and compute unconditional logits
          # Prepare atom_37 data if sidechain conditioning is enabled
          if spec.sidechain_conditioning and hasattr(batched_ensemble, "coordinates"):
            atom_37 = batched_ensemble.coordinates[struct_idx]
            atom_37_mask = (
              batched_ensemble.atom_mask[struct_idx]
              if hasattr(batched_ensemble, "atom_mask") and batched_ensemble.atom_mask is not None
              else (
                batched_ensemble.full_atom_mask[struct_idx]
                if hasattr(batched_ensemble, "full_atom_mask")
                and batched_ensemble.full_atom_mask is not None
                else None
              )
            )
          else:
            atom_37 = None
            atom_37_mask = None

          bundle, config = build_inference_bundle(
            coords=struct_coords,
            mask=struct_mask,
            residue_index=struct_residue_index,
            chain_index=struct_chain_index,
            sequence=None,  # Not needed for unconditional
            backbone_noise=0.0,
            ar_mask=None,
            structure_mapping=None,
            fixed_mask=spec.fixed_mask,
            atom_37=atom_37,
            atom_37_mask=atom_37_mask,
            mode="score_unconditional",
            inference=True,
          )
          logits = score_unconditional(model, subkey, bundle, config, unconditional_stage_set)  # type: ignore[arg-type]
          results_per_feature[feature_name].append(logits)

        elif feature_name == "conditional_logits":
          # Use native sequence from parsed structure, converted to the model's alphabet.
          # aatype is AF; the model's token space is MPNN. Note :802 below already converts
          # candidate *strings* via string_to_protein_sequence -- the two alphabets were
          # ~30 lines apart in this same loop, compared against each other.
          from aminx.utils.aa_convert import af_to_mpnn  # noqa: PLC0415

          if hasattr(batched_ensemble, "aatype") and batched_ensemble.aatype is not None:
            native_seq = af_to_mpnn(batched_ensemble.aatype[struct_idx])
          else:
            msg = (
              "Structure does not contain sequence information; cannot compute conditional_logits"
            )
            raise ValueError(msg)

          cond_logits_fn = make_conditional_logits_fn(model)
          logits = cond_logits_fn(
            subkey,
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            native_seq,
          )
          results_per_feature[feature_name].append(logits)

        elif feature_name == "batched_conditional_logits":
          # Composable_jax batched path (Part 2, Option B): R replicate keys (bb-noise
          # draws) x C externally provided candidate sequences, teacher-forced. See
          # sampling/conditional_logits.py:make_batched_conditional_logits_split_fn and
          # the using-xtrax skill — axis dispatch (Vmap/SafeMap) is BatchPlanner-resolved,
          # not hand-rolled. candidate_sequences/n_replicates validated in specs.py
          # __post_init__ (non-empty candidates; backbone_noise > 0 when n_replicates > 1).
          from aminx.utils.aa_convert import string_to_protein_sequence  # noqa: PLC0415

          candidate_array = jnp.stack(
            [string_to_protein_sequence(seq) for seq in spec.candidate_sequences],
          )
          bb_noise = spec.backbone_noise
          if bb_noise is None:
            bb_noise_scalar = 0.0
          elif isinstance(bb_noise, (int, float)):
            bb_noise_scalar = float(bb_noise)
          else:
            bb_noise_scalar = float(next(iter(bb_noise)))

          replicate_keys = jax.random.split(subkey, spec.n_replicates)

          batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(
            model,
            replicate_batch_size=spec.replicate_batch_size,
            candidate_batch_size=spec.candidate_batch_size,
          )
          encodings = batched_encode_fn(
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            replicate_keys,
            backbone_noise=bb_noise_scalar,
            structure_mapping=None,
          )
          batched_logits = batched_decode_fn(encodings, candidate_array)  # (R, C, L, 21)
          results_per_feature[feature_name].append(batched_logits)

        elif feature_name == "decoded_node_features":
          from aminx.utils.aa_convert import af_to_mpnn  # noqa: PLC0415

          if hasattr(batched_ensemble, "aatype") and batched_ensemble.aatype is not None:
            native_seq = af_to_mpnn(batched_ensemble.aatype[struct_idx])  # AF -> model space
          else:
            msg = "Structure does not contain sequence information; cannot compute decoded_node_features"
            raise ValueError(msg)

          encode_fn, _ = make_encoding_conditional_logits_split_fn(model)
          encoding = encode_fn(
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            backbone_noise=0.0,
            prng_key=subkey,
            structure_mapping=None,
          )
          one_hot = jax.nn.one_hot(native_seq, 21)
          length = one_hot.shape[0]
          # Full context minus self, NOT zeros (#4204). ar_mask[i, j] == 1 means i SEES j,
          # so the previous zeros mask meant these "decoded node features" were decoded
          # from structure alone -- the native one_hot gathered just above reached
          # call_conditional but could not influence it.
          ar_mask = full_context_ar_mask(length)
          decoded = model.decoder.call_conditional(  # type: ignore[attr-defined]
            encoding.node_features,
            encoding.edge_features,
            encoding.neighbor_indices,
            encoding.mask,
            ar_mask,
            one_hot,
            model.w_s_embed.weight,
          )
          results_per_feature[feature_name].append(decoded)

        elif feature_name in ("encoded_node_features", "edge_features"):
          # Use the split function to compute encoder output
          encode_fn, _ = make_encoding_conditional_logits_split_fn(model)
          enc_output = encode_fn(
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            backbone_noise=0.0,
            prng_key=subkey,
            structure_mapping=None,
          )

          if feature_name == "encoded_node_features":
            results_per_feature[feature_name].append(enc_output.node_features)
          else:  # edge_features
            results_per_feature[feature_name].append(enc_output.edge_features)

      if spec.distance_matrix:
        if spec.distance_matrix_method == "ca":
          distance_matrices.append(calculate_ca_distance_matrix(struct_coords))
        elif spec.distance_matrix_method == "cb":
          distance_matrices.append(calculate_cb_distance_matrix(struct_coords))
        elif spec.distance_matrix_method == "closest_atom":
          atom_mask = (
            batched_ensemble.atom_mask[struct_idx]
            if hasattr(batched_ensemble, "atom_mask") and batched_ensemble.atom_mask is not None
            else jnp.ones(struct_coords.shape[:2])
          )
          distance_matrices.append(
            calculate_closest_atom_distance_matrix(struct_coords, atom_mask),
          )
        else:
          msg = f"Unsupported distance_matrix_method: {spec.distance_matrix_method!r}"
          raise NotImplementedError(msg)

      if spec.cross_input_similarity:
        ca_coords_for_similarity.append(_extract_ca_coordinates(struct_coords))

    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  results = {
    "mask": jnp.ones(len(resolved_structure_ids), dtype=jnp.int32),
    "schema_version": INSPECTION_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }

  # Add feature results
  results.update(results_per_feature)

  if spec.distance_matrix:
    results["distance_matrix"] = distance_matrices

  if spec.cross_input_similarity and len(ca_coords_for_similarity) >= 2:
    n_structures = len(ca_coords_for_similarity)
    similarity = jnp.zeros((n_structures, n_structures))
    for i in range(n_structures):
      for j in range(n_structures):
        ca_i = ca_coords_for_similarity[i]
        ca_j = ca_coords_for_similarity[j]
        min_len = min(ca_i.shape[0], ca_j.shape[0])
        ca_i = ca_i[:min_len]
        ca_j = ca_j[:min_len]
        if spec.similarity_metric == "rmsd":
          value = calculate_rmsd(ca_i, ca_j, align=True)
        elif spec.similarity_metric == "tm-score":
          value = calculate_tm_score(ca_i, ca_j, sequence_length=min_len)
        elif spec.similarity_metric == "cosine":
          value = calculate_cosine_similarity(ca_i.reshape(-1), ca_j.reshape(-1))
        else:
          msg = f"Unsupported similarity_metric: {spec.similarity_metric!r}"
          raise NotImplementedError(msg)
        similarity = similarity.at[i, j].set(value)
    results["cross_input_similarity"] = similarity

  return results


def jacobian(
  spec: JacobianSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Compute Jacobians for input structures.

  ``jacobian_mode='categorical'`` (default): full forward-mode categorical Jacobian
  (L, 21, L, 21) via ``jax.jacfwd``.

  ``jacobian_mode='reverse'``: reverse-mode score gradient (L, 21) — one backward
  pass; mutation effect approximated as ``grad[i, m] - grad[i, w]`` (see
  ``aminx.utils.reverse_jac``).
  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = JacobianSpecification(**kw)

  # F005 guard [260826_aminx-invariant-audit]: runner.jacobian cannot honour
  # spec.state_position_map -- no code path in this body reads it (AST hit
  # count 0; see findings.jsonl F005 evidence), so a caller-supplied map would
  # be silently discarded and the output would be single-state while stamped
  # as multistate. Raise loudly instead, mirroring the F002 guard at
  # runner.sample (7460516a); score() honours the field via
  # _score_fused_multistate and campaign verbs implement real state fusion.
  if getattr(spec, "state_position_map", None) is not None:
    msg = (
      "jacobian runner: spec.state_position_map is set but runner.jacobian has "
      "no cross-state fusion path -- the field would be silently discarded. Use "
      "runner.score (which honours state_position_map via _score_fused_multistate) "
      "or the campaign verbs for genuine multistate output."
    )
    raise NotImplementedError(msg)

  if spec.combine:
    msg = "jacobian runner: combine=True not yet implemented; use in-memory results"
    raise NotImplementedError(msg)

  if spec.jacobian_mode == "reverse" and spec.compute_apc:
    msg = "jacobian runner: compute_apc applies to categorical Jacobians only; use --no-compute-apc with reverse mode"
    raise ValueError(msg)

  # spec.fixed_mask -- audit 260826_chain-selection-vendor-superset-audit finding FA3.
  # Neither make_categorical_jacobian_fn nor make_reverse_jacobian_score_fn takes a
  # fixed_mask/chain_mask parameter; confirmed via a differential probe (fixed_mask=
  # all-ones vs all-zeros on an identical structure produced bit-identical
  # categorical_jacobians). Unlike score() this is not obviously N/A-by-definition --
  # excluding fixed positions from the sensitivity computation is a plausible real
  # need -- so this is flagged loud rather than silently guessed at.
  if spec.fixed_mask is not None and _fixed_mask_has_fixed_positions(spec.fixed_mask):
    msg = (
      "jacobian runner: spec.fixed_mask has one or more fixed positions set, but "
      "neither the categorical nor reverse Jacobian kernel has a parameter for it and "
      "neither reads it. runner.sample honours fixed_mask (via "
      "host/_sampling_helper.py's _prepare_fixed_controls); this surface does not yet. "
      "Omit fixed_mask when calling jacobian()."
    )
    raise NotImplementedError(msg)

  import numpy as np  # noqa: PLC0415
  from xtrax.run import SinkSpec, ZarrStagingSink  # noqa: PLC0415

  from aminx.utils.apc import apc_corrected_frobenius_norm  # noqa: PLC0415
  from aminx.utils.autoregression import full_context_ar_mask  # noqa: PLC0415
  from aminx.utils.forward_jac import make_categorical_jacobian_fn  # noqa: PLC0415
  from aminx.utils.reverse_jac import make_reverse_jacobian_score_fn  # noqa: PLC0415

  protein_iterator, model = prep_protein_stream_and_model(spec)
  # jacobian_batch_size now reaches the tangent-axis SafeMap tile in forward_jac. Until
  # 2026-08-12 it was declared on the CLI and the spec and referenced NOWHERE, so a user
  # hitting an OOM could set it and observe no change whatsoever.
  cat_jac_fn = make_categorical_jacobian_fn(
    model,  # type: ignore[arg-type]
    tangent_batch_size=spec.jacobian_batch_size or None,
  )
  # `model`'s runtime union (Aminx | PrxteinLigandMPNN) doesn't structurally satisfy
  # ModelProtocol -- __call__'s first positional param is named `coords` there vs `key`
  # in the protocol. Same pre-existing mismatch as the `make_encoding_conditional_logits_
  # split_fn` calls above. The codebase-wide suppression convention seen nearby (a
  # mypy-style comment naming the ty rule) predates ty and ty does not act on it -- ty's
  # own directive uses a different tool prefix, spelled out below -- so this is the first
  # suppression on this call written in the syntax ty actually reads.
  encode_fn, reverse_grad_fn = make_reverse_jacobian_score_fn(
    model,  # ty: ignore[invalid-argument-type]
  )

  all_jacobians: list[jax.Array] = []
  apc_matrices: list[jax.Array] | None = [] if spec.compute_apc else None

  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0
  prng_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed or 42)
  use_io_sink = spec.output_h5_path is not None
  n_staged = 0

  # Output goes straight through xtrax's ZarrStagingSink, one structure at a time.
  #
  # This replaced aminx's own JacobianAccumulationSink + io_callback round-trip
  # (host/output_sinks.py), which was a plain Python list: every Jacobian was held in
  # memory until the loop finished and only then written. At L=242 one categorical
  # Jacobian is ~103 MB, so an N-structure run peaked at N x 103 MB for no reason -- the
  # tensor is already materialized on the host inside this Python loop, so the io_callback
  # bought nothing here. Staging incrementally means peak memory is one structure.
  #
  # The APC matrix is staged alongside it. It was previously computed and then discarded
  # by every CLI invocation: `results["apc_frobenius_norm"]` only ever existed in the
  # returned dict, which `cli.py` throws away, so the (L, L) map -- the thing an actual
  # coupling analysis wants -- could not be obtained from the CLI at all.
  result_key = "score_gradients" if spec.jacobian_mode == "reverse" else "categorical_jacobians"

  zarr_sink: ZarrStagingSink | None = None
  resolved_aminx_version: str | None = None
  if use_io_sink:
    assert spec.output_h5_path is not None
    zarr_sink = ZarrStagingSink(
      SinkSpec(
        output_dir=spec.output_h5_path,
        format="zarr",
        flush_every=max(1, spec.combine_batch_size or 1),
      ),
    )
    # Resolved HERE, at sink construction -- before `_run_structure_loop` runs any of the
    # actual (expensive) Jacobian/gradient compute below -- so a PackageNotFoundError
    # fails this run before it starts, never after a structure's Jacobian has already
    # been computed and is only now being staged.
    resolved_aminx_version = resolve_aminx_version()

  def _run_structure_loop(*, stage_to_sink: bool) -> None:
    nonlocal prng_key, structure_offset, n_staged

    global_idx = 0
    for _batch_idx, batched_ensemble in enumerate(protein_iterator):
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_offset,
        batch_size=batch_size,
      )

      for struct_idx in range(batch_size):
        struct_coords = batched_ensemble.coordinates[struct_idx]
        struct_mask = batched_ensemble.mask[struct_idx]
        struct_residue_index = batched_ensemble.residue_index[struct_idx]
        struct_chain_index = batched_ensemble.chain_index[struct_idx]

        from aminx.utils.aa_convert import af_to_mpnn  # noqa: PLC0415

        if hasattr(batched_ensemble, "aatype") and batched_ensemble.aatype is not None:
          native_seq = af_to_mpnn(batched_ensemble.aatype[struct_idx])  # AF -> model space
        else:
          msg = "Structure does not contain sequence information; cannot compute Jacobian"
          raise ValueError(msg)

        prng_key, subkey = jax.random.split(prng_key)
        if spec.jacobian_mode == "reverse":
          encoding = encode_fn(
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            backbone_noise=0.0,
            prng_key=subkey,
            structure_mapping=None,
          )
          one_hot = jax.nn.one_hot(native_seq, 21)
          length = one_hot.shape[0]
          # Full context minus self, NOT zeros. An all-zero ar_mask admits no sequence
          # information into the decoder at all (model/decoder.py:144-147 uses it to gate
          # the sequence edge features). That annihilated the forward categorical Jacobian
          # outright; measured here it shifts the reverse score gradient by ~0.16% of its
          # magnitude rather than zeroing it, because the score reads one_hot directly --
          # affected, not destroyed, but wrong either way for a mutation-effect estimate.
          ar_mask = full_context_ar_mask(length)
          jac = reverse_grad_fn(encoding, one_hot, ar_mask)
        else:
          jac = cat_jac_fn(
            subkey,
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            native_seq,
          )

        apc = None
        if apc_matrices is not None:
          apc = apc_corrected_frobenius_norm(
            jac,
            residue_batch_size=spec.apc_residue_batch_size,
          )

        if stage_to_sink:
          assert zarr_sink is not None
          payload: dict[str, Any] = {result_key: np.asarray(jac)}
          if apc is not None:
            payload["apc_frobenius_norm"] = np.asarray(apc)
          # No root stage on this path (see task_id `260910_aminx-sink-provenance-schema`
          # AC3) -- the wheel version and PRNG seed are stamped per-record instead. Uses
          # the SAME `spec.run_spec.sampling.random_seed or 42` expression that built
          # `prng_key` above, so the recorded seed matches what was actually used even
          # when the raw field is falsy (0). This record never stages a "logits" array
          # (result_key is "score_gradients" or "categorical_jacobians"), so
          # `logits_bias_semantics` does not apply here.
          zarr_sink.stage(
            (str(global_idx),),
            attrs={
              "aminx_version": resolved_aminx_version,
              **prng_seed_attrs(spec.run_spec.sampling.random_seed or 42),
            },
            **payload,
          )
          n_staged += 1
          # Deliberately NOT retained: holding `jac` here would reinstate the
          # accumulate-everything behavior this streaming path exists to remove.
        else:
          all_jacobians.append(jac)
          if apc_matrices is not None and apc is not None:
            apc_matrices.append(apc)
        global_idx += 1

      resolved_structure_ids.extend(batch_structure_ids)
      structure_offset += batch_size

  _run_structure_loop(stage_to_sink=use_io_sink)

  n_results = n_staged if use_io_sink else len(all_jacobians)
  results: dict[str, Any] = {
    # Empty when streaming: the tensors are on disk, not in this dict. Read them back from
    # ``results["output_path"]`` rather than expecting them here -- materializing an
    # N x (L, 21, L, 21) list is precisely what the streaming path avoids.
    result_key: [] if use_io_sink else all_jacobians,
    "jacobian_mode": spec.jacobian_mode,
    "mask": jnp.ones(n_results, dtype=jnp.int32),
    "n_structures": n_results,
    "schema_version": JACOBIAN_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }

  if apc_matrices is not None and not use_io_sink:
    results["apc_frobenius_norm"] = apc_matrices

  if use_io_sink and zarr_sink is not None:
    from xtrax.run import fsync_tree, zarr_content_digest  # noqa: PLC0415

    out_path = spec.output_h5_path
    assert out_path is not None
    zarr_sink.drain()
    # fsync BEFORE digesting: the digest is only a claim about durable bytes if the tree
    # has actually reached disk. Both primitives are xtrax's -- they were hoisted out of
    # aminx into xtrax.run.zarr_integrity, so this uses the promoted version rather than
    # a local copy.
    fsync_tree(out_path)
    results["output_path"] = str(out_path)
    results["output_digest"] = zarr_content_digest(out_path)

  return results
