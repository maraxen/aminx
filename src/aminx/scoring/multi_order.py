"""Multi-order causal permutation averaging for sequence scoring (Clause 4).

Implements the reference LigandMPNN multi-order score averaging protocol (clause
`score_only_multi_order_average`, score.py:280-327):
- Evaluates teacher-forced causal decoding passes over M independent random permutations.
- For each order draw m, constructs a strictly causal autoregressive DAG mask
  (ar_mask_from_decoding_order).
- Transforms conditional logits to probability space: P^(m) = exp(log_probs^(m)).
- Averages across draws in probability space:
    P_mean = (1 / M) * sum_m P^(m)
    P_std = sqrt((1 / M) * sum_m (P^(m) - P_mean)^2)
- Reports ensemble negative log-likelihood:
    NLL = -sum(log(P_mean[i, s_i]) * mask[i]) / (sum(mask) + 1e-8)
  as well as individual sample NLLs and order arrays.

Formally tracked under Praxia debt `debt-multi-order-scoring-average` (debt #3166).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.inference import score_conditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set  # noqa: TID251
from aminx.scoring.score import _nll_from_logits
from aminx.utils.autoregression import ar_mask_from_decoding_order
from aminx.utils.decoding_order import random_decoding_order

if TYPE_CHECKING:
  from aminx.types.protocols import ModelProtocol

SCORE_EPS: float = 1e-8


class MultiOrderScoreResult(NamedTuple):
  """Container for multi-order causal score averaging outputs."""

  ensemble_nll: jax.Array
  mean_probs: jax.Array
  std_probs: jax.Array
  sample_nlls: jax.Array
  decoding_orders: jax.Array
  mean_sample_nll: jax.Array


def score_multi_order_average(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  sequence: jax.Array,
  structure_coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  num_orders: int = 16,
  *,
  backbone_noise: float = 0.0,
  tie_group_map: jax.Array | None = None,
  ligand_coords: jax.Array | None = None,
  ligand_atom_types: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
  multi_state_strategy: str = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
  inference: bool = True,
  **kwargs: Any,  # noqa: ANN401, ARG001
) -> MultiOrderScoreResult:
  """Score sequence by averaging predicted probabilities over M causal orders.

  Matches reference score.py:280-327 behavior:
  For each order:
    1. Fresh random decoding order over positions. Tied positions (``tie_group_map``) are drawn
       as units and decode at one step, the same draw the sampler uses, so the causal mask built
       from the order is tie-aware.
    2. Causal order mask (no future positions visible, zero diagonal).
    3. Forward decoder pass yielding logits and log_probs.
    4. Probabilities P^(m) = exp(log_probs).
  Averaging:
    P_mean = mean_m(P^(m))
    P_std = std_m(P^(m))
    Ensemble NLL is computed from P_mean evaluated on the true sequence tokens.

  Args:
    model: Equinox MPNN model.
    prng_key: JAX PRNG key.
    sequence: Integer sequence tokens (L,) or one-hot (L, 21).
    structure_coordinates: Backbone coordinates.
    mask: Residue validity mask (L,).
    residue_index: Residue sequence indices (L,).
    chain_index: Chain indices (L,).
    num_orders: Number of random causal permutation draws M (default: 16).
    backbone_noise: Backbone coordinate perturbation standard deviation.
    tie_group_map: Optional (L,) tie group id per position, ids in ``[0, L)``. Tied positions
      decode together and are mutually visible in each order's mask.
    ligand_coords: Optional ligand coordinates.
    ligand_atom_types: Optional ligand atom types.
    ligand_mask: Optional ligand atom validity mask.
    multi_state_strategy: Multi-state aggregation strategy.
    multi_state_temperature: Multi-state temperature.
    inference: Set inference mode on model.
    **kwargs: Additional kwargs ignored for backward compatibility.

  Returns:
    MultiOrderScoreResult containing ensemble_nll, mean_probs, std_probs,
    sample_nlls, decoding_orders, and mean_sample_nll.
  """
  if inference and isinstance(model, eqx.Module):
    model = eqx.nn.inference_mode(model, value=True)

  l_len = sequence.shape[0]
  vocab_size = sequence.shape[-1] if sequence.ndim == 2 else 21

  if sequence.ndim == 1:
    seq_tokens = sequence.astype(jnp.int32)
    seq_one_hot = jax.nn.one_hot(seq_tokens, vocab_size)
  else:
    seq_one_hot = sequence
    seq_tokens = jnp.argmax(sequence, axis=-1).astype(jnp.int32)

  res_mask = mask.astype(jnp.float32)

  # Generate M random causal decoding orders with the same draw the sampler uses, so the
  # order and the mask built from it agree on ties. num_groups is static and must bound the ids.
  order_keys = jax.random.split(prng_key, num_orders)
  num_groups = None if tie_group_map is None else int(jnp.max(tie_group_map)) + 1

  def _draw_order(k: PRNGKeyArray) -> jax.Array:
    order, _ = random_decoding_order(k, l_len, tie_group_map, num_groups)
    return order

  decoding_orders = jax.vmap(_draw_order)(order_keys)  # (M, L)

  stage_set = make_stage_set(
    strategy=multi_state_strategy,
    strategy_temperature=multi_state_temperature,
  )

  # Pre-encode structure once (structure coordinates and ligand condition the encoder)
  enc_key, decode_base_key = jax.random.split(order_keys[0])
  init_bundle, config = build_inference_bundle(
    coords=structure_coordinates,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    sequence=seq_tokens,
    backbone_noise=backbone_noise,
    ar_mask=ar_mask_from_decoding_order(decoding_orders[0], tie_group_map=tie_group_map),
    tie_group_map=tie_group_map,
    ligand_coords=ligand_coords,
    ligand_atom_types=ligand_atom_types,
    ligand_mask=ligand_mask,
    mode="score_conditional",
    inference=True,
  )
  encoding = score_conditional.encode(model, enc_key, init_bundle, config)

  # Evaluate each order m
  def _eval_single_order(
    order: jax.Array,
    k: PRNGKeyArray,
  ) -> tuple[jax.Array, jax.Array]:
    ar_mask = ar_mask_from_decoding_order(order, tie_group_map=tie_group_map)
    order_bundle, _ = build_inference_bundle(
      coords=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=seq_tokens,
      backbone_noise=backbone_noise,
      ar_mask=ar_mask,
      tie_group_map=tie_group_map,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
      mode="score_conditional",
      inference=True,
    )
    logits = score_conditional.score_from_encoding(
      model, k, encoding, order_bundle, config, stage_set,
    )
    order_nll = _nll_from_logits(logits, seq_one_hot, res_mask)
    probs = jax.nn.softmax(logits, axis=-1)
    return order_nll, probs

  decode_keys = jax.random.split(decode_base_key, num_orders)
  sample_nlls, probs_stack = jax.vmap(_eval_single_order)(decoding_orders, decode_keys)

  # Probability-space mean and std over the M draws
  mean_probs = jnp.mean(probs_stack, axis=0)  # (L, 21)
  std_probs = jnp.std(probs_stack, axis=0)  # (L, 21)

  # True-token probabilities under the ensemble distribution
  p_true = jnp.sum(seq_one_hot * mean_probs, axis=-1)  # (L,)
  log_p_true = jnp.log(jnp.maximum(p_true, 1e-12))
  ensemble_nll = -jnp.sum(log_p_true * res_mask) / (jnp.sum(res_mask) + SCORE_EPS)
  mean_sample_nll = jnp.mean(sample_nlls)

  return MultiOrderScoreResult(
    ensemble_nll=ensemble_nll,
    mean_probs=mean_probs,
    std_probs=std_probs,
    sample_nlls=sample_nlls,
    decoding_orders=decoding_orders,
    mean_sample_nll=mean_sample_nll,
  )
