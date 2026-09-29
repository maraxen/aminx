"""Contains the logic for averaging node features over multiple structures and/or noise levels."""

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, Int, PRNGKeyArray

from aminx.inference.logits import LOGIT_STRATEGIES
from aminx.model.capabilities import PRXTEIN_MPNN_CAPABILITIES
from aminx.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
from aminx.types.arrays import (
  AlphaCarbonMask,
  BackboneCoordinates,
  BackboneNoise,
  ChainIndex,
  InputBias,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)
from aminx.types.bundles import EncoderOutput
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import (
  EncodingFusionFn as _EncodingFusionFn,  # noqa: F401 (type reference)
)
from aminx.utils.autoregression import ar_mask_from_decoding_order
from aminx.utils.data_structures import Protein
from aminx.utils.decoding_order import DecodingOrder, DecodingOrderFn
from aminx.utils.ste import straight_through_estimator


class ArithmeticMeanEncodingFusion(eqx.Module):
  """Average D encoded outputs element-wise over the leading noise axis.

  Receives a stacked EncoderOutput with leading D axis (one entry per noise
  level) and returns a single EncoderOutput with the D axis reduced. K=1 output.

  neighbor_indices and mask are taken from the first noise-level entry (they
  are structural constants, not noise-varying).
  """

  def __call__(self, stacked: EncoderOutput) -> EncoderOutput:
    return cast(
      "EncoderOutput",
      EncoderOutput(
        node_features=jnp.mean(stacked.node_features, axis=0),
        edge_features=jnp.mean(stacked.edge_features, axis=0),
        neighbor_indices=stacked.neighbor_indices[0],
        mask=stacked.mask[0],
      ),
    )


class IdentityEncodingFusion(eqx.Module):
  """Pass-through fusion: return all D encoded outputs unchanged (K=D).

  No reduction is applied. The output has the same leading D axis as the
  input. Used for testing arbitrary-K dispatch and as a no-op baseline.
  """

  def __call__(self, stacked: EncoderOutput) -> EncoderOutput:
    return cast("EncoderOutput", stacked)


def _get_averaged_encodings_legacy(
  batched_ensemble: Protein,
  model: ModelProtocol,
  backbone_noise: Sequence[float] | float,
  noise_batch_size: int,
  random_seed: int,
  average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"],
  structure_mapping: jax.Array | None = None,
) -> tuple:
  """Compute averaged node and edge features from an ensemble of protein structures.

  This function encodes a batch of protein structures at multiple noise levels,
  then averages the resulting node and edge features based on the specified
  averaging mode.

  Args:
      batched_ensemble: A batch of protein structures.
      model: The Aminx model.
      backbone_noise: The backbone noise levels to use.
      noise_batch_size: The batch size for noise levels.
      random_seed: The random seed to use.
      average_encoding_mode: The mode for averaging encodings.

  Returns:
      A tuple containing the averaged encodings and other necessary features for
      downstream tasks (sampling, scoring, etc.).

  """
  encode_fn, _, _ = make_encoding_sampling_split_fn(model)
  caps = getattr(model, "capabilities", None)
  supports_structure_mapping = (
    caps.encode_fn_supports_structure_mapping
    if caps is not None
    else PRXTEIN_MPNN_CAPABILITIES.encode_fn_supports_structure_mapping
  )

  noise_array = (
    jnp.asarray(backbone_noise, dtype=jnp.float32) if backbone_noise is not None else jnp.zeros(1)
  )

  def encode_single_noise(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise: BackboneNoise,
    mapping: jax.Array | None = None,
    _encoder: Callable = encode_fn,
  ) -> tuple:
    """Encode one structure at one noise level."""
    if supports_structure_mapping:
      return _encoder(
        key,
        coords,
        mask,
        residue_ix,
        chain_ix,
        backbone_noise=noise,
        structure_mapping=mapping,
      )
    return _encoder(
      key,
      coords,
      mask,
      residue_ix,
      chain_ix,
      backbone_noise=noise,
    )

  def mapped_encode_noise(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise_arr: BackboneNoise,
    mapping: jax.Array | None = None,
  ) -> tuple:
    """Compute encodings across all noise levels for a single structure."""
    return jax.lax.map(
      partial(
        encode_single_noise,
        key,
        coords,
        mask,
        residue_ix,
        chain_ix,
        mapping=mapping,
      ),
      noise_arr,
      batch_size=noise_batch_size,
    )

  mapping_for_vmap = structure_mapping
  if mapping_for_vmap is not None and mapping_for_vmap.ndim == 1:
    batch_size = batched_ensemble.coordinates.shape[0]
    mapping_for_vmap = jnp.broadcast_to(
      jnp.atleast_2d(mapping_for_vmap),
      (batch_size, mapping_for_vmap.shape[0]),
    )
  mapping_in_axis = 0 if mapping_for_vmap is not None else None

  vmap_encode_structures = jax.vmap(
    mapped_encode_noise,
    in_axes=(None, 0, 0, 0, 0, None, mapping_in_axis),
  )

  encoded_features_per_noise = vmap_encode_structures(
    jax.random.key(random_seed),
    batched_ensemble.coordinates,
    batched_ensemble.mask,
    batched_ensemble.residue_index,
    batched_ensemble.chain_index,
    noise_array,
    mapping_for_vmap,
  )

  (
    node_features,
    processed_edge_features,
    neighbor_indices,
    mask,
    ar_mask,
  ) = encoded_features_per_noise

  if average_encoding_mode == "inputs":
    averaging_axis = 0
  elif average_encoding_mode == "noise_levels":
    averaging_axis = 1
  else:  # "inputs_and_noise"
    averaging_axis = (0, 1)

  avg_node_features = jnp.mean(node_features, axis=averaging_axis)
  avg_processed_edge_features = jnp.mean(processed_edge_features, axis=averaging_axis)

  return (
    avg_node_features,
    avg_processed_edge_features,
    neighbor_indices,
    mask,
    ar_mask,
  )


def get_averaged_encodings(*args, **kwargs):
  """Deprecated. Use ArithmeticMeanEncodingFusion via StageSet.encoding_fusion instead."""
  warnings.warn(
    "get_averaged_encodings is deprecated. Use ArithmeticMeanEncodingFusion "
    "via StageSet.encoding_fusion instead.",
    DeprecationWarning,
    stacklevel=2,
  )
  return _get_averaged_encodings_legacy(*args, **kwargs)


def _ste_optimize_sequence_legacy(
  prng_key: PRNGKeyArray,
  encoded_features: tuple,
  decode_logits_fn: Callable,
  autoregressive_mask: jnp.ndarray,
  seq_length: int,
  iterations: Int,
  learning_rate: Float,
  temperature: Float,
  bias: InputBias | None = None,
) -> ProteinSequence:
  """Straight-through estimator (STE) optimization for legacy sampling path.

  Optimizes sequence logits using gradient descent with the straight-through
  estimator to allow gradients through discrete sampling. Uses cached encoder
  features and the legacy decode_logits_fn interface.

  Args:
    prng_key: JAX random key for initialization.
    encoded_features: Cached encoder features tuple from encode_fn.
    decode_logits_fn: Function (encoded_features, sequence, ar_mask) -> logits.
    autoregressive_mask: Autoregressive masking array (L, L).
    seq_length: Length of sequence.
    iterations: Number of STE optimization iterations.
    learning_rate: Optimizer learning rate.
    temperature: Temperature for softmax in STE.
    bias: Optional bias to add to logits (N, 21).

  Returns:
    Optimized sequence as integer array (N,).

  """
  # Initialize sequence logits
  sequence_logits = jnp.zeros((seq_length, 21), dtype=jnp.float32)

  # Create optimizer
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(sequence_logits)

  def loss_fn(logits: jnp.ndarray) -> jnp.ndarray:
    """Loss function for STE optimization."""
    # Apply STE to get differentiable one-hot
    one_hot_seq = straight_through_estimator(logits / temperature)

    # Convert one-hot to discrete sequence for model evaluation
    discrete_seq = one_hot_seq.argmax(axis=-1).astype(jnp.int8)

    # Get model logits for this sequence
    model_logits = decode_logits_fn(encoded_features, discrete_seq, autoregressive_mask)

    # Apply bias if provided
    if bias is not None:
      model_logits = model_logits + bias

    # Cross-entropy loss between STE and model logits
    loss = optax.softmax_cross_entropy(logits=model_logits, labels=one_hot_seq)
    return loss.mean()

  def update_step(
    iteration: int,
    carry: tuple[jnp.ndarray, optax.OptState],
  ) -> tuple[jnp.ndarray, optax.OptState]:
    """Single STE optimization step."""
    current_logits, current_opt_state = carry

    # Compute loss and gradients
    _, grads = jax.value_and_grad(loss_fn)(current_logits)

    # Update via optimizer
    updates, next_opt_state = optimizer.update(grads, current_opt_state)
    next_logits = optax.apply_updates(current_logits, updates)

    return next_logits, next_opt_state

  # Run optimization loop
  final_logits, _ = jax.lax.fori_loop(
    0,
    iterations,
    update_step,
    (sequence_logits, opt_state),
  )

  # Convert final logits to sequence via argmax
  final_sequence = final_logits.argmax(axis=-1).astype(jnp.int8)

  return final_sequence


def _make_encoding_sampling_split_fn_legacy(
  model_parameters: ModelProtocol,
  decoding_order_fn: DecodingOrderFn | None = None,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  decode_fn_wrapper: Callable[[Callable], Callable] | None = None,
) -> tuple[Callable, Callable, Callable]:
  """Create separate encoding and sampling functions for averaged encodings.

  This splits the sampling process into two parts:
  1. Encoding: Structure -> Encoder features (can be averaged across noise levels)
  2. Sampling: (Encoder features, PRNGKey) -> Sequences

  This separation allows:
  - Averaging encoder features across multiple noise levels
  - Efficient reuse of encoder output for multiple samples
  - Lower memory usage when sampling many sequences

  Supports tied positions: when tie_group_map is provided, positions in the same
  group will be sampled together and receive identical amino acids.

  Args:
    model_parameters: A Aminx Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".
    decode_fn_wrapper: Optional wrapper for the decode function.

  Returns:
    Tuple of (encode_fn, sample_fn) where:
      - encode_fn: Computes encoder features from structure
      - sample_fn: Samples sequences from cached encoder features

  Example:
    >>> from aminx.io.weights import load_model
    >>> model = load_model()
    >>> encode_fn, sample_fn = make_encoding_sampling_split_fn(
    ...     model, sampling_strategy="temperature"
    ... )
    >>> # Encode once
    >>> encoding = encode_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     k_neighbors=48, backbone_noise=0.1
    ... )
    >>> # Sample multiple sequences using same encoding
    >>> seq1 = sample_fn(key1, encoding, order1, temperature=0.1)
    >>> seq2 = sample_fn(key2, encoding, order2, temperature=0.5)

  """
  del sampling_strategy, decoding_order_fn  # Currently only temperature sampling supported

  # Get the encoder/decoder split functions
  encode_logits_fn, decode_logits_fn = make_encoding_conditional_logits_split_fn(
    model_parameters,
  )

  if decode_fn_wrapper is not None:
    decode_logits_fn = decode_fn_wrapper(decode_logits_fn)

  @partial(jax.jit)
  def encode_fn(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> tuple:
    """Encode structure to get encoder features.

    Args:
      prng_key: JAX random key for feature extraction.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      k_neighbors: Number of nearest neighbors (currently not used, for API compat).
      backbone_noise: Optional noise for backbone coordinates.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.

    Returns:
      Encoder features tuple that can be passed to sample_fn.

    """
    return encode_logits_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
      structure_mapping=structure_mapping,
    )

  @partial(jax.jit, static_argnames=("num_groups", "multi_state_strategy", "sampling_strategy"))
  def sample_fn(
    prng_key: PRNGKeyArray,
    encoded_features: tuple,
    decoding_order: DecodingOrder,
    bias: InputBias | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
  ) -> ProteinSequence:
    """Sample a sequence from cached encoder features.

    Args:
      prng_key: JAX random key for sampling.
      encoded_features: Encoder features from encode_fn.
      decoding_order: Decoding order for autoregressive sampling.
      bias: Optional bias to add to logits (N, 21).
      iterations: For STE optimization (default: 100 for STE, unused for temperature).
      learning_rate: For STE optimization (default: 0.01 for STE, unused for temperature).
      temperature: Temperature for sampling (default: 1.0).
      sampling_strategy: "temperature" or "straight_through".
      tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
      num_groups: Number of unique groups when using tied positions.
      multi_state_strategy: Strategy for combining logits across tied positions.
      multi_state_temperature: Temperature for geometric_mean strategy.

    Returns:
      Sampled sequence as integer array (N,).

    """
    if temperature is None:
      temperature = jnp.array(1.0, dtype=jnp.float32)

    # `decoding_order` is an ORDER array. generate_ar_mask would read it as a RANK array, and
    # its second positional parameter is chain_idx, not the tie map (debt #1982).
    autoregressive_mask = jnp.asarray(ar_mask_from_decoding_order(decoding_order, tie_group_map))

    # Dispatch based on sampling strategy
    if sampling_strategy == "straight_through":
      return _ste_optimize_sequence_legacy(
        prng_key,
        encoded_features,
        decode_logits_fn,
        autoregressive_mask,
        seq_length=autoregressive_mask.shape[0],
        iterations=100 if iterations is None else iterations,
        learning_rate=0.01 if learning_rate is None else learning_rate,
        temperature=temperature,
        bias=bias,
      )

    # Temperature-based sampling path
    del iterations, learning_rate  # Not used in temperature sampling

    seq_length = autoregressive_mask.shape[0]
    _, prng_key = jax.random.split(prng_key)
    initial_seq = jax.random.randint(
      prng_key,
      shape=(seq_length,),
      minval=0,
      maxval=21,
      dtype=jnp.int8,
    )

    if tie_group_map is not None and num_groups is not None:
      # Visit groups in the order they first appear in `decoding_order` -- the order the mask
      # above encodes -- not in group-id order, which the mask knows nothing about.
      seq_len_groups = tie_group_map.shape[0]
      first_step_of_group = (
        jnp.full((seq_len_groups,), seq_len_groups, dtype=jnp.int32)
        .at[tie_group_map[decoding_order]]
        .min(jnp.arange(seq_len_groups, dtype=jnp.int32))
      )
      group_sequence = jnp.argsort(first_step_of_group, stable=True)

      def sample_group_step(
        group_idx: int,
        state: tuple[ProteinSequence, PRNGKeyArray],
      ) -> tuple[ProteinSequence, PRNGKeyArray]:
        """Sample one group of tied positions together."""
        sequence, key = state

        logits = decode_logits_fn(encoded_features, sequence, autoregressive_mask)
        group_member_mask = tie_group_map == group_sequence[group_idx]  # (N,) boolean
        # Stubbed for Phase 5 refactor compatibility
        if multi_state_strategy == "arithmetic_mean":
          strategy_cls = LOGIT_STRATEGIES.get("arithmetic_mean")
          avg_logits = strategy_cls(jnp.ones(1))(logits[None, ...])
        elif multi_state_strategy == "geometric_mean":
          strategy_cls = LOGIT_STRATEGIES.get("geometric_mean")
          avg_logits = strategy_cls(jnp.ones(1), temperature=multi_state_temperature)(
            logits[None, ...],
          )
        else:
          strategy_cls = LOGIT_STRATEGIES.get("product")
          avg_logits = strategy_cls(jnp.ones(1))(logits[None, ...])
        if bias is not None:
          group_mask = group_member_mask.astype(jnp.float32)[:, None]  # (N, 1)
          group_count = group_mask.sum(axis=0, keepdims=True)  # (1, 1)
          group_bias = (bias * group_mask).sum(axis=0) / (group_count.squeeze() + 1e-8)
          avg_logits = avg_logits + group_bias
        scaled_logits = avg_logits / temperature
        key, subkey = jax.random.split(key)
        sampled_aa = jax.random.categorical(subkey, scaled_logits).astype(jnp.int8)
        updated_seq = jnp.where(group_member_mask, sampled_aa, sequence)
        return updated_seq, key

      final_seq, _ = jax.lax.fori_loop(
        0,
        num_groups,
        sample_group_step,
        (initial_seq, prng_key),
      )

    else:

      def sample_step(
        i: int,
        state: tuple[ProteinSequence, PRNGKeyArray],
      ) -> tuple[ProteinSequence, PRNGKeyArray]:
        """Sample one position in the decoding order."""
        sequence, key = state
        logits = decode_logits_fn(encoded_features, sequence, autoregressive_mask)
        pos = decoding_order[i]
        if bias is not None:
          logits = logits.at[pos].add(bias[pos])
        scaled_logits = logits[pos] / temperature
        key, subkey = jax.random.split(key)
        sampled_aa = jax.random.categorical(subkey, scaled_logits).astype(jnp.int8)
        updated_seq = cast("jax.Array", sequence).at[pos].set(sampled_aa)
        return updated_seq, key

      final_seq, _ = jax.lax.fori_loop(
        0,
        seq_length,
        sample_step,
        (initial_seq, prng_key),
      )

    return final_seq

  return encode_fn, sample_fn, decode_logits_fn


def make_encoding_sampling_split_fn(*args, **kwargs):
  """Deprecated. Use InferencePlan.encode and InferencePlan.decode instead."""
  warnings.warn(
    "make_encoding_sampling_split_fn is deprecated. Use InferencePlan.encode "
    "and InferencePlan.decode instead.",
    DeprecationWarning,
    stacklevel=2,
  )
  return _make_encoding_sampling_split_fn_legacy(*args, **kwargs)
