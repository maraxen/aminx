"""Factory for creating conditional logits functions.

Conditional logits are computed given a specific sequence input,
allowing the model to evaluate how well a sequence fits a structure.

Stacked multistate (``state_vmap_exact``): :func:`make_conditional_logits_state_vmap_fn`,
or ``model(..., decoding_approach=\"conditional\", multistate_mode=\"state_vmap_exact\", ...)``
with stacked geometry tensors and a flat ``one_hot_sequence`` / aa indices.

For a single carrier object, use
(geometry in :class:`~aminx.types.bundles.GeometryBundle`; ligand tensors in :class:`~aminx.types.bundles.LigandBundle`).

This is used for:
- Jacobian computation (sensitivity analysis)
- Sequence scoring and validation
- Conformational inference
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference.score_conditional import kernel as score_conditional
from aminx.tiling.axes import N_CANDIDATES, N_REPLICATES
from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax
from aminx.tiling.planner import plan_axis_strategy
from aminx.types.arrays import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  Logits,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)
from aminx.types.encodings import EncoderOutput
from aminx.types.protocols import ConditionalLogitsFn, ModelProtocol
from aminx.utils.autoregression import full_context_ar_mask


def make_conditional_logits_fn(
  model: ModelProtocol,
) -> ConditionalLogitsFn:
  """Create a function to compute conditional logits for a given sequence.

  Conditional logits evaluate how well a sequence fits a structure by
  running the model with the sequence as input.

  Args:
    model: A Aminx Equinox model instance.

  Returns:
    A function that computes conditional logits for sequence-structure pairs.

  Example:
    >>> from aminx.io.weights import load_model
    >>> model = load_model()
    >>> logits_fn = make_conditional_logits_fn(model)
    >>> logits = logits_fn(key, coords, mask, res_idx, chain_idx, sequence)

  """

  @partial(jax.jit)
  def conditional_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> Logits:
    """Compute conditional logits for a sequence-structure pair.

    Args:
      prng_key: JAX random key. Drives backbone-noise injection when ``backbone_noise``
        is nonzero; key-invariant only at ``backbone_noise=0`` (the default).
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      sequence: Protein sequence as integer array (N,) or one-hot (N, 21).
      ar_mask: Optional autoregressive mask (N, N).
      backbone_noise: Optional noise for backbone coordinates.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.

    Returns:
      Logits of shape (N, 21) for each residue position.

    Example:
      >>> logits = conditional_logits(
      ...     key, coords, mask, res_idx, chain_idx, sequence
      ... )

    """
    bundle, config = build_inference_bundle(
      coords=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=sequence,
      backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
      ar_mask=ar_mask,
      structure_mapping=structure_mapping,
      mode="score_conditional",
      inference=True,
    )
    stage_set = make_stage_set()

    return score_conditional(
      model=model,
      prng_key=prng_key,
      bundle=bundle,
      config=config,
      stage_set=stage_set,
    )

  return cast("ConditionalLogitsFn", conditional_logits)


def make_encoding_conditional_logits_split_fn(
  model: ModelProtocol,
) -> tuple[Callable, Callable]:
  """Create separate encoding and decoding functions for averaged encodings.

  This splits the model into two parts:
  1. Encoding: Structure -> Encoder features (node_features, edge_features, neighbor_indices)
  2. Decoding: (Encoder features, Sequence) -> Logits

  This separation allows:
  - Averaging encoder features across multiple noise levels
  - Efficient jacobian computation by caching encoder output
  - Reusing encoder output for multiple sequence evaluations

  Args:
    model: A Aminx Equinox model instance.

  Returns:
    Tuple of (encode_fn, decode_fn) where:
      - encode_fn: Computes encoder features from structure
      - decode_fn: Computes logits from cached features and sequence

  Example:
    >>> encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)
    >>> # Encode once
    >>> key = jax.random.key(0)
    >>> encoding = encode_fn(key, coords, mask, res_idx, chain_idx, noise=0.1)
    >>> # Decode multiple sequences using same encoding
    >>> logits1 = decode_fn(encoding, sequence1)
    >>> logits2 = decode_fn(encoding, sequence2)

  """
  # Use inference mode for decoding to skip dropout (allows running without PRNG key)
  inference_model = eqx.nn.inference_mode(model, value=True)

  def encode_fn(
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    prng_key: PRNGKeyArray | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> tuple:
    """Encode structure to get encoder features.

    Args:
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Optional noise for backbone coordinates.
      prng_key: JAX random key for feature extraction.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.

    Returns:
      Tuple of (node_features, edge_features, neighbor_indices, mask, ar_mask_placeholder)
      where ar_mask_placeholder is zeros to maintain consistent shape.

    """
    if backbone_noise is None:
      backbone_noise = jax.numpy.array(0.0, dtype=jax.numpy.float32)

    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    key_features, key_encoder = jax.random.split(prng_key, 2)

    edge_features, neighbor_indices, initial_node_features, _ = model.features(
      key_features,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      structure_mapping=structure_mapping,
    )

    node_features, processed_edge_features = model.encoder(
      edge_features,
      neighbor_indices,
      mask,
      initial_node_features,
      key=key_encoder,
    )

    return EncoderOutput(
      node_features=node_features,
      edge_features=processed_edge_features,
      neighbor_indices=neighbor_indices,
      mask=mask,
    )

  def decode_fn(
    encoding: EncoderOutput,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
  ) -> Logits:
    """Decode encoder features to conditional logits for a given sequence.

    Args:
      encoding: EncoderOutput from encode_fn containing node_features, edge_features, neighbor_indices, and mask.
      sequence: Protein sequence as integer array (N,) or one-hot (N, 21).
      ar_mask: Optional autoregressive mask (N, N). ``ar_mask[i, j] == 1`` means position
        ``i`` SEES position ``j``'s sequence. Defaults to full context minus self
        (``1 - I``), matching ``build_inference_bundle(mode="score_conditional")``, so the
        default here and the default of :func:`make_conditional_logits_fn` agree.

        Pass an explicit all-zero mask for UNCONDITIONAL (structure-only) logits. That is a
        legitimate request, but it must be made explicitly -- see below.

    Returns:
      Logits of shape (N, 21) for each residue position.

    Note:
      This default was ZEROS until #4222, and the docstring described that as "If None,
      uses zeros" without saying what it meant. Since an all-zero mask admits no sequence
      information, the function returned the SAME logits for every sequence: measured on
      1ubq, max|logits(native) - logits(all-Ala)| was 0.000000 through this path and 8.249
      through :func:`make_conditional_logits_fn`. Callers that omitted ``ar_mask`` -- which
      included the runner's own batched ``conditional_logits`` feature -- were silently
      getting unconditional logits from a function named for conditional ones.

      The failure mode is worse than a crash: downstream, identical conditionals for every
      sequence look like the finding "these positions carry no sequence-context
      information" rather than like a bug.
    """
    node_features = encoding.node_features
    processed_edge_features = encoding.edge_features
    neighbor_indices = encoding.neighbor_indices
    mask = encoding.mask

    if ar_mask is None:
      ar_mask = full_context_ar_mask(node_features.shape[0])

    if sequence.ndim == 1:
      one_hot_sequence = jax.nn.one_hot(sequence, inference_model.w_s_embed.num_embeddings)
    else:
      one_hot_sequence = sequence

    decoded_node_features = inference_model.decoder.call_conditional(
      node_features,
      processed_edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      inference_model.w_s_embed.weight,
    )

    return jax.vmap(inference_model.w_out, in_axes=0)(decoded_node_features)

  return encode_fn, decode_fn


# Promoted to aminx.tiling.planner.plan_axis_strategy (2026-08-12) once it had a third
# consumer -- this module, mbr_consensus (which imported the private name across modules),
# and the categorical Jacobian. Kept as an alias so those imports and their tests keep
# working unchanged; the implementation is byte-identical, only relocated.
_plan_axis_strategy = plan_axis_strategy


def make_batched_conditional_logits_split_fn(
  model: ModelProtocol,
  *,
  replicate_batch_size: int | None = None,
  candidate_batch_size: int | None = None,
) -> tuple[Callable, Callable]:
  """Batched encode-over-replicates / decode-over-candidates split fn.

  Builds on :func:`make_encoding_conditional_logits_split_fn`: the replicate axis
  (R distinct backbone-noise PRNG keys) drives ``encode_fn``; the candidate axis
  (C externally provided sequences) drives the keyless, inference-mode ``decode_fn``.
  This is the clean CRN (common random numbers) factoring for paired-key JSD
  analyses: replicate = encode, candidate = decode. No kernel or fusion change —
  this composes the existing encode/decode split via BatchPlanner-selected axis
  strategies (Vmap when the axis fits the memory budget, SafeMap tiling otherwise).

  Args:
    model: A Aminx Equinox model instance.
    replicate_batch_size: Fixed SafeMap tile size for the replicate (R) axis.
      None defers to the BatchPlanner's memory-budget-driven choice.
    candidate_batch_size: Fixed SafeMap tile size for the candidate (C) axis.
      Same default behavior as ``replicate_batch_size``.

  Returns:
    Tuple of (batched_encode_fn, batched_decode_fn):
      - ``batched_encode_fn(coords, mask, res_idx, chain_idx, replicate_keys,
        backbone_noise=None, structure_mapping=None) -> EncoderOutput`` batched
        over R (leading axis).
      - ``batched_decode_fn(encodings, candidate_sequences, ar_mask=None) ->
        Logits`` of shape ``(R, C, L, 21)``.

  Example:
    >>> batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)
    >>> keys = jax.random.split(jax.random.key(0), 256)  # R=256 replicates
    >>> encodings = batched_encode_fn(coords, mask, res_idx, chain_idx, keys, backbone_noise=0.1)
    >>> logits = batched_decode_fn(encodings, candidate_sequences)  # (256, C, L, 21)

  """
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  def batched_encode_fn(
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    replicate_keys: PRNGKeyArray,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> EncoderOutput:
    """Encode a structure once per replicate key; returns EncoderOutput batched over R."""
    n_replicates = replicate_keys.shape[0]
    seq_len = structure_coordinates.shape[0]
    # Conservative per-replicate activation estimate (node + edge features, float32).
    activation_bytes = seq_len * (128 + 32 * 48) * 4
    strategy = _plan_axis_strategy(
      N_REPLICATES,
      n_replicates,
      replicate_batch_size,
      activation_bytes_per_element=activation_bytes,
    )
    iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_REPLICATES.name)

    def _encode_one(key: PRNGKeyArray) -> EncoderOutput:
      return encode_fn(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        backbone_noise=backbone_noise,
        prng_key=key,
        structure_mapping=structure_mapping,
      )

    return iterator(_encode_one, replicate_keys)

  def batched_decode_fn(
    encodings: EncoderOutput,
    candidate_sequences: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
  ) -> Logits:
    """Decode R encodings x C candidate sequences; returns (R, C, L, 21) logits.

    ``ar_mask`` is forwarded to the per-candidate ``decode_fn``, so ``None`` means full
    context minus self (``1 - I``) -- NOT zeros. Until #4222 it meant zeros, which made
    every candidate in the batch score identically because none of them could see any
    sequence at all.
    """
    n_candidates = candidate_sequences.shape[0]
    seq_len = candidate_sequences.shape[1]
    activation_bytes = seq_len * 21 * 4  # (L, 21) float32 logits per candidate
    strategy = _plan_axis_strategy(
      N_CANDIDATES,
      n_candidates,
      candidate_batch_size,
      activation_bytes_per_element=activation_bytes,
    )
    iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_CANDIDATES.name)

    def _decode_over_candidates(enc: EncoderOutput) -> Logits:
      def _decode_one(seq: ProteinSequence) -> Logits:
        return decode_fn(enc, seq, ar_mask)

      return iterator(_decode_one, candidate_sequences)

    # R is already materialized by batched_encode_fn (its own Vmap/SafeMap choice
    # fixed this shape); a plain vmap composes the R axis without re-planning it.
    return jax.vmap(_decode_over_candidates)(encodings)

  return batched_encode_fn, batched_decode_fn


def make_batched_conditional_logits_fn(
  model: ModelProtocol,
  *,
  replicate_batch_size: int | None = None,
  candidate_batch_size: int | None = None,
) -> Callable:
  """Monolithic batched conditional-logits fn (re-encodes per candidate).

  Secondary convenience wrapper around :func:`make_batched_conditional_logits_split_fn`
  for small-C call sites that do not need to reuse an encoding across multiple decode
  calls. Prefer the split fn for the paired-CRN JSD / MPNN-filter workloads.
  """
  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(
    model,
    replicate_batch_size=replicate_batch_size,
    candidate_batch_size=candidate_batch_size,
  )

  def batched_conditional_logits(
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    replicate_keys: PRNGKeyArray,
    candidate_sequences: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> Logits:
    encodings = batched_encode_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      replicate_keys,
      backbone_noise=backbone_noise,
      structure_mapping=structure_mapping,
    )
    return batched_decode_fn(encodings, candidate_sequences, ar_mask)

  return batched_conditional_logits
