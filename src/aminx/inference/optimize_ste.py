"""Straight-through estimator optimization for sequence design.

This module implements iterative sequence optimization using the straight-through
estimator (STE) to allow gradients through discrete sampling operations.

Rematerialization (jax.checkpoint) is used to reduce memory during backward passes
by recomputing activations on demand instead of storing them. Activate via the
JAX_ENABLE_REMAT environment variable to trade CPU speed for memory efficiency.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from aminx.inference.logits import LOGIT_STRATEGIES
from aminx.inference.score_conditional import kernel as score_conditional
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.stages import StageSet

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
  from jaxtyping import Float, PRNGKeyArray

  from aminx.io.designs import DesignMetadata, DesignZarrWriter
  from aminx.model import Aminx
  from aminx.types.arrays import (
    AutoRegressiveMask,
    Logits,
    ProteinSequence,
  )


from aminx.utils.autoregression import ar_mask_from_decoding_order
from aminx.utils.decoding_order import (
  DecodingOrderFn,
  random_decoding_order,
  resolve_decoding_order_fn,
)
from aminx.utils.ste import gumbel_softmax, straight_through_estimator

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def make_optimize_sequence_fn(
  model: Aminx,
  stage_set: StageSet,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  batch_size: int = 4,
  use_concrete: bool = False,
  tau_start: float = 1.0,
  tau_end: float = 0.1,
) -> Callable[..., tuple[ProteinSequence, Logits, Logits]]:
  """Create an STE (straight-through estimator) sequence optimization function.

  Parameters
  ----------
  model : Aminx
      Protein/ligand model instance.
  stage_set : StageSet
      Resolved stage set with logit_transform, decode_step, and sample_step wired.
      REQUIRED (no default). TypeError raised if omitted.
  decoding_order_fn : DecodingOrderFn, optional
      Function to generate decoding order (default: random).
  batch_size : int, optional
      Batch size for optimization (default: 4).
  use_concrete : bool, optional
      Use Gumbel-Softmax for STE (default: False).
  tau_start : float, optional
      Start temperature for Gumbel-Softmax (default: 1.0).
  tau_end : float, optional
      End temperature for Gumbel-Softmax (default: 0.1).

  Returns
  -------
  Callable
      Optimization function with signature:
      (prng_key, bundle, config, iterations, learning_rate, temperature,
       use_rolling_state=False, logit_combine_strategy=0, writer=None)
      → (sequence, logits_output, logits_ste)

  Notes
  -----
  Hard-cut (Task 13 / Risk REC-5): stage_set is now mandatory. Calling without
  it raises TypeError at function-call time (Python signature enforcement).
  No DeprecationWarning. Migration required for all callers: pass stage_set=
  at call time or construct via make_stage_set(...).

  """

  @partial(jax.jit, static_argnames=("use_rolling_state", "logit_combine_strategy"))
  def optimize_sequence(
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    iterations: int,
    learning_rate: float,
    temperature: float,
    use_rolling_state: bool = False,
    logit_combine_strategy: int = 0,
    writer: DesignZarrWriter | None = None,
  ) -> tuple[ProteinSequence, Logits, Logits]:
    num_residues = bundle.geometry.coords.shape[1]
    num_classes = 21
    num_groups = bundle.geometry.n_canonical

    fixed_mask = bundle.conditioning.fixed_mask
    # Fixed positions first by default (aminx #2017), like the sampler; a caller's own function is untouched.
    order_fn = resolve_decoding_order_fn(decoding_order_fn, fixed_mask)
    fixed_tokens = bundle.conditioning.fixed_tokens
    tie_group_map = bundle.conditioning.tie_group_map

    CLAMP_LOGIT = 1e4
    if fixed_mask is not None and fixed_tokens is not None:
      fixed_mask_bool = fixed_mask.astype(jnp.bool_)
      fixed_one_hot = jax.nn.one_hot(fixed_tokens, num_classes, dtype=jnp.float32)
      fixed_bias = fixed_mask_bool[:, None] * (
        CLAMP_LOGIT * fixed_one_hot - CLAMP_LOGIT * (1.0 - fixed_one_hot)
      )
    else:
      fixed_bias = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)
      fixed_mask_bool = jnp.zeros(num_residues, dtype=jnp.bool_)

    sequence_logits = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(sequence_logits)

    def update_step(
      _iteration: int,
      carry: tuple[Logits, optax.OptState, PRNGKeyArray],
    ) -> tuple[Logits, optax.OptState, PRNGKeyArray]:
      current_logits, current_opt_state, current_key = carry

      if use_concrete:
        key_decoding_orders, gumbel_key, next_key = jax.random.split(current_key, 3)
        progress = jnp.float32(_iteration) / jnp.maximum(
          jnp.float32(iterations) - 1.0,
          1.0,
        )
        tau = jnp.float32(tau_start) * (jnp.float32(tau_end / tau_start) ** progress)
      else:
        key_decoding_orders, next_key = jax.random.split(current_key)

      keys_for_decoding = jax.random.split(key_decoding_orders, batch_size)
      decoding_orders, _ = jax.vmap(order_fn, in_axes=(0, None, None, None))(
        keys_for_decoding,
        num_residues,
        tie_group_map[0]
        if tie_group_map is not None
        else None,  # assume identical for all states for the tie group
        num_groups,
      )

      # ar_masks will have shape (batch_size, L, L). This used to be
      # `vmap(generate_ar_mask, in_axes=(0, None))(orders, tie_map)`, which was wrong twice:
      # generate_ar_mask's untied branch reads a RANK array, and its SECOND positional
      # parameter is `chain_idx`, not `tie_group_map` -- so a tie map restricted every
      # position to seeing only earlier members of its own tie group (debt #1982).
      ar_masks = jax.vmap(ar_mask_from_decoding_order, in_axes=(0, None))(
        decoding_orders, tie_group_map[0] if tie_group_map is not None else None,
      )

      def loss_fn(logits: Logits) -> Float:
        if use_concrete:
          seq_repr = gumbel_softmax(logits / temperature, tau, gumbel_key, hard=True)
          one_hot_sequence = seq_repr
        else:
          one_hot_sequence = straight_through_estimator(logits / temperature)
          seq_repr = one_hot_sequence  # to prevent undefined variable in labels later

        def eval_with_mask(ar_mask: AutoRegressiveMask) -> Logits:
          # Broadcast ar_mask to S
          ar_mask_stack = jnp.broadcast_to(
            ar_mask[None, ...], (bundle.geometry.n_states, *ar_mask.shape),
          )

          # Replace conditioning with our local one_hot_sequence and ar_mask
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

          # stage_set is required (hard-cut in Task 13); use it directly
          # Call the score_conditional kernel
          output_logits = score_conditional(
            model=model,
            prng_key=next_key,
            bundle=bundle_new,
            config=config,
            stage_set=stage_set,
          )
          return output_logits

        remat_value = os.getenv("JAX_ENABLE_REMAT", "0").lower()
        remat_truthy = {"1", "true", "yes", "on", "enabled"}
        enable_remat = remat_value in remat_truthy

        if enable_remat:
          eval_with_mask_remat = jax.checkpoint(eval_with_mask)
          pred_logits_batch = jax.vmap(eval_with_mask_remat, in_axes=0)(ar_masks)
        else:
          pred_logits_batch = jax.vmap(eval_with_mask, in_axes=0)(ar_masks)

        output_logits = jnp.mean(pred_logits_batch, axis=0)

        labels = seq_repr if use_concrete else straight_through_estimator(logits)
        loss = optax.softmax_cross_entropy(
          logits=output_logits,
          labels=labels,
        )

        mask = bundle.geometry.mask[0]  # assuming identical masks across states for the loss
        return (loss * mask).sum() / (mask.sum() + 1e-8)

      _, grads = jax.value_and_grad(loss_fn)(current_logits)

      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_logits: Logits = optax.apply_updates(current_logits, updates)
      next_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, next_logits)

      if tie_group_map is not None and num_groups is not None:
        group_one_hot = jax.nn.one_hot(
          tie_group_map[0],
          num_groups,
          dtype=jnp.float32,
        )

        group_logit_sums = jnp.einsum("ng,na->ga", group_one_hot, next_logits)
        group_counts = group_one_hot.sum(axis=0)
        group_avg_logits = group_logit_sums / (group_counts[:, None] + 1e-8)
        next_logits = jnp.einsum("ng,ga->na", group_one_hot, group_avg_logits)

      return next_logits, next_opt_state, next_key

    final_logits, _, final_key = jax.lax.fori_loop(
      0,
      iterations,
      update_step,
      (sequence_logits, opt_state, prng_key),
    )

    final_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, final_logits)
    final_one_hot = straight_through_estimator(final_logits / temperature)
    final_sequence = final_one_hot.argmax(axis=-1).astype(jnp.int8)

    if writer is not None:

      def _save_logits(logits: jnp.ndarray) -> None:
        scores = jnp.array([0.0], dtype=jnp.float32)
        n_states = bundle.geometry.n_states
        state_weights = jnp.ones(n_states, dtype=jnp.float32)
        metadata: DesignMetadata = {
          "pool_type": "BackboneOnly",
          "state_mapping": [],
          "weight_strategy": "uniform",
          "combination_algorithm": "ste_optimization",
          "structure_ids": [],
          "parent_structure_idx": 0,
        }

        payload = {
          "sequence": final_sequence,
          "logits": logits,
          "scores": scores,
          "state_weights": state_weights,
          "metadata": metadata,
        }
        writer.write((0,), payload)

      jax.experimental.io_callback(_save_logits, None, final_logits, ordered=False)
      jax.effects_barrier()

    final_decoding_order, _ = order_fn(
      final_key, num_residues, tie_group_map[0] if tie_group_map is not None else None, num_groups,
    )
    final_ar_mask = ar_mask_from_decoding_order(
      final_decoding_order, tie_group_map[0] if tie_group_map is not None else None,
    )

    ar_mask_stack = jnp.broadcast_to(
      final_ar_mask[None, ...], (bundle.geometry.n_states, *final_ar_mask.shape),
    )
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

    if stage_set is None:
      strategy_cls = LOGIT_STRATEGIES.get(
        "arithmetic_mean"
        if logit_combine_strategy == 0
        else "geometric_mean"
        if logit_combine_strategy == 1
        else "product",
      )
      fuse = strategy_cls(bundle.conditioning.state_weights)
      final_stage_set = StageSet(logit_transform=fuse)
    else:
      final_stage_set = stage_set

    final_output_logits = score_conditional(
      model=model,
      prng_key=final_key,
      bundle=bundle_new,
      config=config,
      stage_set=final_stage_set,
    )

    return final_sequence, final_output_logits, final_logits

  return cast("Callable", optimize_sequence)
