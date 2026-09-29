"""STEDecode mode class (Task 10).

Straight-Through Estimator (STE) decode: iterative sequence refinement via
gradient-based optimization. Wraps a _ConditionalDecodeBase inner mode
(typically ConditionalDecode) and applies update_step iterations with
optax optimizer, reproducing the tied-group einsum averaging (Risk D-2)
explicitly via _tied_group_einsum_average from _kernel.py.

The inner stage_set is projected to logit_transform only for the scoring call.
Tied-group averaging is handled by STEDecode itself, not by the projection.
"""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from aminx.inference.decode._base import _ConditionalDecodeBase
from aminx.inference.decode._kernel import _tied_group_einsum_average
from aminx.inference.encode import make_encode_fn
from aminx.types.arrays import Logits, ProteinSequence
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.stages import StageSet
from aminx.utils.autoregression import ar_mask_from_decoding_order
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order
from aminx.utils.ste import gumbel_softmax, straight_through_estimator

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def _project_stage_set_for_ste(stage_set: StageSet) -> StageSet:
  """Project stage_set to logit_transform only (module-local helper).

  STE scoring call needs only the logit_transform; all other transforms
  are inapplicable during the optimization loop.

  Parameters
  ----------
  stage_set : StageSet
      Input stage_set with potentially multiple transforms.

  Returns
  -------
  StageSet
      Projected stage_set with only logit_transform; all other slots are None.

  """
  # Use dataclasses.replace to create a new StageSet with nulled-out transforms
  from dataclasses import replace as dataclass_replace

  return dataclass_replace(
    stage_set,
    ar_logit_transform=None,
    tie_group_fuse=None,
    decoder_sink=(),
    decode_step=None,
  )


class STEDecode(eqx.Module):
  """Straight-Through Estimator decode mode: iterative refinement with gradients.

  Composes a _ConditionalDecodeBase inner mode and applies gradient-based
  sequence optimization via optax optimizer. Reproduces the tied-group einsum
  averaging (Risk D-2) explicitly using _tied_group_einsum_average.

  Parameters
  ----------
  inner : _ConditionalDecodeBase
      Inner conditional decode instance (e.g., ConditionalDecode).
      Used to score sequences during optimization.
  iterations : int, default=100
      Number of STE refinement iterations (static).
  optimizer : Any, default=None
      optax optimizer instance (static).
  decoding_order_fn : DecodingOrderFn, optional
      Function to generate decoding orders. Defaults to random_decoding_order.

  Attributes
  ----------
  inner : _ConditionalDecodeBase
      Inner decode mode (traced, not static).
  iterations : int = eqx.field(static=True)
      Number of iterations (static, determines JIT recompilation boundaries).
  optimizer : Any = eqx.field(static=True)
      Optimizer instance (static).

  Notes
  -----
  The inner field is NOT static so that switching between different inner
  implementations (e.g., different state_iterator strategies) only triggers
  a JIT recompile on treedef change, not every call. The iterations and
  optimizer are static to embed them in the JIT.

  Tied-group einsum (Risk D-2) is reproduced character-for-character inside
  update_step via _tied_group_einsum_average(logits, tie_group_map, num_groups).

  """

  inner: _ConditionalDecodeBase
  iterations: int = eqx.field(static=True)
  optimizer: Any = eqx.field(static=True)
  decoding_order_fn: DecodingOrderFn = eqx.field(static=True)

  def __init__(
    self,
    inner: _ConditionalDecodeBase,
    iterations: int = 100,
    optimizer: Any = None,
    decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  ):
    """Initialize STEDecode.

    Parameters
    ----------
    inner : _ConditionalDecodeBase
        Inner conditional decode instance.
    iterations : int, default=100
        Number of STE iterations.
    optimizer : Any, default=None
        optax optimizer. If None, defaults to optax.adam(1e-3).
    decoding_order_fn : DecodingOrderFn, optional
        Decoding order function; defaults to random_decoding_order.

    """
    self.inner = inner
    self.iterations = iterations
    self.optimizer = optimizer if optimizer is not None else optax.adam(1e-3)
    self.decoding_order_fn = decoding_order_fn

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    temperature: float = 1.0,
    stage_set: StageSet | None = None,
    use_concrete: bool = False,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
  ) -> tuple[ProteinSequence, Logits, Logits]:
    """Optimize sequence via STE.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        PRNG key for stochasticity.
    bundle : InferenceBundle
        Inference bundle with geometry and conditioning.
    config : InferenceConfig
        Inference configuration.
    temperature : float, default=1.0
        Temperature for softmax in STE.
    stage_set : StageSet | None, default=None
        Pipeline stages. If None, constructs default. Will be projected
        to logit_transform only for inner scoring.
    use_concrete : bool, default=False
        Use Gumbel-softmax (True) or STE (False).
    tau_start : float, default=1.0
        Initial temperature for Gumbel-softmax.
    tau_end : float, default=0.1
        Final temperature for Gumbel-softmax.

    Returns
    -------
    tuple[ProteinSequence, Logits, Logits]
        (final_sequence, final_output_logits, final_ste_logits)
        - final_sequence: Shape (L,), argmax of final logits.
        - final_output_logits: Shape (L, 21), final score output.
        - final_ste_logits: Shape (L, 21), final STE optimization logits.

    """
    num_residues = bundle.geometry.coords.shape[1]
    num_classes = 21
    num_groups = bundle.geometry.n_canonical

    fixed_mask = bundle.conditioning.fixed_mask
    fixed_tokens = bundle.conditioning.fixed_tokens
    tie_group_map = bundle.conditioning.tie_group_map

    # Handle fixed positions
    CLAMP_LOGIT = 1e4
    if fixed_mask is not None and fixed_tokens is not None:
      fixed_mask_bool = fixed_mask.astype(jnp.bool_)
      fixed_one_hot = jax.nn.one_hot(
        fixed_tokens,
        num_classes,
        dtype=jnp.float32,
      )
      fixed_bias = fixed_mask_bool[:, None] * (
        CLAMP_LOGIT * fixed_one_hot - CLAMP_LOGIT * (1.0 - fixed_one_hot)
      )
    else:
      fixed_bias = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)
      fixed_mask_bool = jnp.zeros(num_residues, dtype=jnp.bool_)

    # Initialize sequence logits
    sequence_logits = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)

    # Initialize optimizer state
    opt_state = self.optimizer.init(sequence_logits)

    # Project stage_set for inner scoring
    projected_stage_set = _project_stage_set_for_ste(
      stage_set if stage_set is not None else StageSet(),
    )

    # Build encoder function (reuse across iterations)
    # Model is in the inner ConditionalDecode
    model = self.inner.model if hasattr(self.inner, "model") else None
    if model is None:
      raise ValueError("STEDecode requires inner mode with 'model' attribute")
    encode_fn = make_encode_fn(model, use_rolling_state=False)

    def loss_fn(
      logits: Logits,
      iteration: int,
      key_decoding_orders: PRNGKeyArray,
      gumbel_key: PRNGKeyArray,
      next_key: PRNGKeyArray,
    ) -> jnp.ndarray:
      """Compute STE loss for a given logit initialization."""
      if use_concrete:
        # Gumbel-softmax
        progress = jnp.float32(iteration) / jnp.maximum(
          jnp.float32(self.iterations) - 1.0,
          1.0,
        )
        tau = jnp.float32(tau_start) * (jnp.float32(tau_end / tau_start) ** progress)
        seq_repr = gumbel_softmax(
          logits / temperature,
          tau,
          gumbel_key,
          hard=True,
        )
        one_hot_sequence = seq_repr
      else:
        # STE
        one_hot_sequence = straight_through_estimator(logits / temperature)
        seq_repr = one_hot_sequence

      # Split for batch of decoding orders (batch_size=4)
      batch_size = 4
      keys_for_decoding = jax.random.split(key_decoding_orders, batch_size)

      # Generate batch_size decoding orders
      decoding_orders, _ = jax.vmap(
        self.decoding_order_fn, in_axes=(0, None, None, None),
      )(
        keys_for_decoding,
        num_residues,
        tie_group_map[0] if tie_group_map is not None else None,
        num_groups,
      )

      # Generate AR masks for each decoding order (batch_size, L, L). Not generate_ar_mask:
      # it reads a RANK array, and its second positional parameter is `chain_idx`, so the
      # tie map passed here used to land there (debt #1982).
      ar_masks = jax.vmap(ar_mask_from_decoding_order, in_axes=(0, None))(
        decoding_orders,
        tie_group_map[0] if tie_group_map is not None else None,
      )

      def eval_with_mask(ar_mask: jnp.ndarray) -> Logits:
        """Evaluate with a specific AR mask, return output logits."""
        # Broadcast to all states
        ar_mask_stack = jnp.broadcast_to(
          ar_mask[None, ...],
          (bundle.geometry.n_states, *ar_mask.shape),
        )

        # Update bundle with current sequence and AR mask
        cond_new = eqx.tree_at(
          lambda c: (c.sequence_oh, c.ar_mask),
          bundle.conditioning,
          (one_hot_sequence, ar_mask_stack),
        )
        bundle_new = eqx.tree_at(
          lambda b: b.conditioning,
          bundle,
          cond_new,
        )

        # Split key for encode and decode (matching reference pattern)
        key_enc, key_dec = jax.random.split(next_key)

        # Encode the bundle
        enc = encode_fn(bundle_new, key_enc, config)

        # Score with inner decode mode (use full stage_set, matching reference)
        output_logits = self.inner(
          key=key_dec,
          enc=enc,
          bundle=bundle_new,
          config=config,
          stage_set=stage_set if stage_set is not None else StageSet(),
        )
        return output_logits

      # Vmap eval_with_mask over ar_masks (batch_size, L, L) → (batch_size, L, V)
      pred_logits_batch = jax.vmap(eval_with_mask, in_axes=0)(ar_masks)

      # Average logits across batch: (batch_size, L, V) → (L, V)
      output_logits = jnp.mean(pred_logits_batch, axis=0)

      # Compute cross-entropy loss
      labels = seq_repr if use_concrete else straight_through_estimator(logits)
      loss = optax.softmax_cross_entropy(
        logits=output_logits,
        labels=labels,
      )

      mask = bundle.geometry.mask[0]  # Assuming identical masks
      return (loss * mask).sum() / (mask.sum() + 1e-8)

    def update_step(
      iteration: int,
      carry: tuple[Logits, optax.OptState, PRNGKeyArray],
    ) -> tuple[Logits, optax.OptState, PRNGKeyArray]:
      """Single STE optimization step with tied-group averaging."""
      current_logits, current_opt_state, current_key = carry

      # Split the key matching reference implementation pattern
      if use_concrete:
        key_decoding_orders, gumbel_key, next_key = jax.random.split(current_key, 3)
      else:
        key_decoding_orders, next_key = jax.random.split(current_key)
        gumbel_key = None  # type: ignore

      # Create a closure that captures the iteration and split keys
      def loss_fn_with_keys(logits: Logits) -> jnp.ndarray:
        return loss_fn(logits, iteration, key_decoding_orders, gumbel_key, next_key)

      # Compute loss and gradients
      _, grads = jax.value_and_grad(loss_fn_with_keys)(current_logits)

      # Update via optimizer
      updates, next_opt_state = self.optimizer.update(grads, current_opt_state)
      next_logits: Logits = optax.apply_updates(current_logits, updates)

      # Apply fixed positions
      next_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, next_logits)

      # Apply tied-group einsum averaging (Risk D-2)
      if tie_group_map is not None and num_groups is not None:
        # Expand to (1, L, V) for the einsum helper
        next_logits_expanded = next_logits[None, :, :]
        next_logits_averaged = _tied_group_einsum_average(
          next_logits_expanded,
          tie_group_map[0],
          num_groups,
        )
        next_logits = next_logits_averaged[0, :, :]

      return next_logits, next_opt_state, next_key

    # Run optimization loop
    final_logits, _, final_key = jax.lax.fori_loop(
      0,
      self.iterations,
      update_step,
      (sequence_logits, opt_state, prng_key),
    )

    # Apply fixed positions one more time
    final_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, final_logits)

    # Final STE pass
    final_one_hot = straight_through_estimator(final_logits / temperature)
    final_sequence = final_one_hot.argmax(axis=-1).astype(jnp.int8)

    # Generate final decoding order
    final_decoding_order, _ = self.decoding_order_fn(
      final_key,
      num_residues,
      tie_group_map[0] if tie_group_map is not None else None,
      num_groups,
    )
    final_ar_mask = ar_mask_from_decoding_order(
      final_decoding_order,
      tie_group_map[0] if tie_group_map is not None else None,
    )

    # Broadcast to all states
    ar_mask_stack = jnp.broadcast_to(
      final_ar_mask[None, ...],
      (bundle.geometry.n_states, *final_ar_mask.shape),
    )

    # Update bundle with final sequence
    cond_new = eqx.tree_at(
      lambda c: (c.sequence_oh, c.ar_mask),
      bundle.conditioning,
      (final_one_hot, ar_mask_stack),
    )
    bundle_new = eqx.tree_at(
      lambda b: b.conditioning,
      bundle,
      cond_new,
    )

    # Split the final key for encoding and decoding (matching reference pattern)
    key_enc_final, key_dec_final = jax.random.split(final_key)

    # Encode the final bundle
    enc_final = encode_fn(bundle_new, key_enc_final, config)

    # Score final sequence using unprojected stage_set (matching reference)
    final_output_logits = self.inner(
      key=key_dec_final,
      enc=enc_final,
      bundle=bundle_new,
      config=config,
      stage_set=stage_set if stage_set is not None else StageSet(),
    )

    return final_sequence, final_output_logits, final_logits


__all__ = ["STEDecode"]
