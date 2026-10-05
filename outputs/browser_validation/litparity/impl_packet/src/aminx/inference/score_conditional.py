"""Conditional scoring kernel using InferenceBundle."""

from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from xtrax.tiling import VmapIterator

from aminx.inference.decode.conditional import ConditionalDecode  # noqa: TID251
from aminx.inference.encode import make_encode_fn
from aminx.types.arrays import Logits
from aminx.types.bundles import EncoderOutput as BundleEncoderOutput
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.encodings import EncoderOutput
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import StageSet  # noqa: TID251


def encode(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
) -> EncoderOutput:
  """Encode structure using vmap strategy (parallel over S conformational states).

  Args:
    model: Aminx model with encoder.
    prng_key: PRNG key.
    bundle: Inference bundle containing geometry and conditioning.
    config: Inference configuration.

  Returns:
    EncoderOutput with node_features, edge_features, neighbor_indices, mask
    stacked over S states.
  """
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  return encode_fn(bundle, prng_key, config)


def score_from_encoding(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  enc: Any,  # noqa: ANN401
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Decode from a pre-computed encoding to produce logits.

  Args:
    model: Aminx model with decoder.
    prng_key: PRNG key.
    enc: Pre-computed encoder output (EncoderOutput from encode.py).
    bundle: Inference bundle containing conditioning.
    config: Inference configuration.
    stage_set: Stage set for logit fusion strategy.

  Returns:
    Logits (teacher-forced conditional logits).
  """
  # Use ConditionalDecode mode class for conditional decoding
  # (hardcoded Vmap iterator; STE uses this internally)
  decode_fn = ConditionalDecode(model=model, state_iterator=VmapIterator())
  return decode_fn(
    key=prng_key,
    enc=cast("BundleEncoderOutput", enc),
    bundle=bundle,
    config=config,
    stage_set=stage_set,
  )


def _stack_encoder_outputs(encs: Sequence[Any]) -> Any:  # noqa: ANN401
  """Stack D encoder outputs along a new leading D axis.

  All fields (node_features, edge_features, neighbor_indices, mask) are stacked
  with a new leading D dimension, matching the sampling path convention in
  kernel_dispatch.py:321-325. ArithmeticMeanEncodingFusion then reduces D to 1:
  features via mean(axis=0), geometry via [0] indexing -- which correctly
  recovers the noise-invariant geometry from the first stack entry.

  Args:
    encs: Sequence of D EncoderOutput objects from encode() calls.

  Returns:
    A stacked EncoderOutput with leading D axis on all fields.
  """
  # Stack all fields along new leading D axis.
  # node_features: (S,L,H) -> (D,S,L,H); same for edge_features.
  # neighbor_indices: (S,L,K) -> (D,S,L,K); mask: (S,L) -> (D,S,L).
  # This matches the convention in kernel_dispatch.py:321-325 (sampling path).
  # ArithmeticMeanEncodingFusion then reduces D->1 via mean(axis=0) for features
  # and [0] indexing for geometry, correctly recovering the noise-invariant geometry.
  node_features_stacked = jnp.stack([enc.node_features for enc in encs], axis=0)
  edge_features_stacked = jnp.stack([enc.edge_features for enc in encs], axis=0)
  neighbor_indices_stacked = jnp.stack([enc.neighbor_indices for enc in encs], axis=0)
  mask_stacked = jnp.stack([enc.mask for enc in encs], axis=0) if encs[0].mask is not None else None

  return EncoderOutput(
    node_features=node_features_stacked,
    edge_features=edge_features_stacked,
    neighbor_indices=neighbor_indices_stacked,
    mask=mask_stacked,
  )


def score_averaged(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundles_per_noise: Sequence[InferenceBundle],
  config: InferenceConfig,
  stage_set: StageSet,
  encoding_fusion: Any,  # noqa: ANN401
) -> Logits:
  """Score by averaging node features over D backbone noise levels.

  Interprets `average_node_features=True` as: encode at D noise levels, average
  the node/edge features, then decode once from the fused features. This is
  Interpretation (A) in the AC-RS-7 spec.

  Args:
    model: Aminx model with encoder and decoder.
    prng_key: PRNG key for encoding steps.
    bundles_per_noise: List of D InferenceBundle objects, one per backbone_noise.
      All bundles must have identical conditioning fields (only noise varies).
    config: Inference configuration (shared across all D bundles).
    stage_set: Stage set for logit fusion.
    encoding_fusion: Fusion function (e.g., ArithmeticMeanEncodingFusion) that
      reduces D encodings to 1.

  Returns:
    Logits (teacher-forced conditional logits from fused encoding).
  """
  # Encode each bundle and collect encodings
  num_noise = len(bundles_per_noise)
  encs = []
  for d_idx in range(num_noise):
    # Split key for each bundle to ensure independence
    k_enc = jax.random.fold_in(prng_key, d_idx)
    enc = encode(model, k_enc, bundles_per_noise[d_idx], config)
    encs.append(enc)

  # Stack encodings along leading D axis
  stacked_enc = _stack_encoder_outputs(encs)

  # Fuse D→1 via encoding_fusion (e.g., ArithmeticMeanEncodingFusion)
  fused_enc = encoding_fusion(stacked_enc)

  # Decode once from fused encoding (use first bundle for conditioning, since all are identical)
  return score_from_encoding(
    model,
    prng_key,
    fused_enc,
    bundles_per_noise[0],
    config,
    stage_set,
  )


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Compute teacher-forced conditional logits.

  Side-chain context (atom_37/atom_37_mask) is packaged onto the GeometryBundle by
  build_inference_bundle; it is not accepted as loose kernel kwargs (see #105).
  """
  k_enc, k_dec = jax.random.split(prng_key)

  # Encode and decode, byte-identical to original implementation
  enc = encode(model, k_enc, bundle, config)
  return score_from_encoding(model, k_dec, enc, bundle, config, stage_set)
