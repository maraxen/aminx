"""Unfused conditional decode: per-state logits without fusion (MBR consensus reranking).

Mirrors ConditionalDecode.__call__ (conditional.py) exactly through its pre-fusion
logits_stack computation, then returns that directly instead of calling the two
fusion helpers. See ../../.praxia/docs/specs/260709_mbr-consensus-reranking-composition.md
for the full design rationale (§1, §3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, Float32, Int8, PRNGKeyArray
from xtrax.tiling import MapIterator

from aminx.inference.decode._kernel import _decode_one_step, _project_logits
from aminx.types.bundles import EncoderOutput
from aminx.types.configs import InferenceConfig


def decode_states_unfused(
  model: Any,
  encodings: EncoderOutput,
  sequence_oh: Float[Array, "L V"],
  ar_mask: Int8[Array, "S L L"] | Float32[Array, "S L L"],
  key: PRNGKeyArray,
  config: InferenceConfig,
  state_iterator: MapIterator,
  decode_step: Callable | None = None,
) -> Float[Array, "S L V"]:
  """Genuine per-state, unfused conditional logits via jax.vmap over states.

  The exact intermediate value ConditionalDecode.__call__ computes immediately before
  its two fusion calls (_apply_logit_transform, _apply_tie_group_fuse) -- this function
  stops there instead of fusing, giving independent per-state scores (what MBR consensus
  reranking needs, as opposed to a score against an already-fused quantity).

  PRECONDITION (not enforced here -- see
  ../../.praxia/docs/decisions/260709_n-states-heterogeneous-flag-unenforced.md):
  ``encodings`` must come from a bundle whose state axis is already uniform-shape (e.g.
  tev_design's build_canonical_bundle.py, N_CANONICAL=214-padded). aminx's own
  N_STATES.heterogeneous=True registry flag is a conservative declaration for the axis
  in the abstract, not an enforced runtime check -- this function does not detect or
  reject genuinely ragged per-state input; calling it with unpadded, differently-shaped
  states will fail (or silently misbehave) inside the state_iterator's jax.vmap, not here.

  Parameters
  ----------
  model : Any
      Model instance with decoder and w_out attributes.
  encodings : EncoderOutput
      Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
  sequence_oh : ndarray
      Single one-hot sequence (L, V), broadcast to all S states internally --
      the same sequence is scored against every reference state independently.
  ar_mask : ndarray
      Autoregressive attention mask, already per-state (S, L, L) -- matches
      ConditioningBundle.ar_mask's real shape (all-ones for purely conditional scoring).
  key : PRNGKeyArray
      PRNG key for dropout/stochasticity.
  config : InferenceConfig
      Inference configuration (only config.inference is consulted).
  state_iterator : MapIterator
      Iterator for the S axis. Callers should pass VmapIterator() given the
      pre-padded-uniform-shape precondition above -- there is no reason to SafeMap
      an axis that's guaranteed uniform.
  decode_step : Callable | None, default None
      Optional override decode function; None uses model.decoder.call_conditional.

  Returns
  -------
  ndarray
      Per-state, unfused logits. Shape (S, L, V). V = 21 (vocabulary size).

  """
  S = encodings.node_features.shape[0]

  # Broadcast sequence one-hot to all states: (L, V) -> (S, L, V). Mirrors
  # ConditionalDecode.__call__ lines 112-116 exactly.
  seq_oh_stack = jnp.broadcast_to(sequence_oh[None, ...], (S, *sequence_oh.shape))

  per_state_inputs = (
    encodings.node_features,
    encodings.edge_features,
    encodings.neighbor_indices,
    encodings.mask,
    ar_mask,
    seq_oh_stack,
  )

  def per_state_fn(inputs: tuple) -> Any:
    """Decode one state: unpacks the per-state pytree tuple and calls _decode_one_step.

    Parameters
    ----------
    inputs : tuple
        Per-state slice of the (node_features, edge_features, neighbor_indices, mask,
        ar_mask, sequence_oh) pytree, sliced along axis 0 by state_iterator.

    Returns
    -------
    ndarray
        Decoded hidden features for this state. Shape (L, H).

    """
    node_features, edge_features, neighbor_indices, mask, per_state_ar_mask, seq_oh = inputs
    return _decode_one_step(
      model=model,
      node_features=node_features,
      edge_features=edge_features,
      neighbor_indices=neighbor_indices,
      mask=mask,
      ar_mask=per_state_ar_mask,
      sequence_oh=seq_oh,
      key=key,
      inference=config.inference,
      decode_step=decode_step,
    )

  decoded = state_iterator(per_state_fn, per_state_inputs, in_axes=0)

  # Project to logits and STOP -- no _apply_logit_transform, no _apply_tie_group_fuse.
  return _project_logits(model, decoded)
