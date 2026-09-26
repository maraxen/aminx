"""Factory for creating sequence sampling functions for Aminx."""

from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.inference import optimize_ste, sample_autoregressive
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.registry import SAMPLERS
from aminx.types.bundles import WaveScheduleBundle
from aminx.types.protocols import ModelProtocol, SamplerFn
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order

_AMINO_ACID_VOCAB = 21


_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def make_sample_sequences(
  model: ModelProtocol,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  use_concrete: bool = False,
  tau_start: float = 1.0,
  tau_end: float = 0.1,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using Aminx.

  Args:
    model: A Aminx Equinox model instance.
    decoding_order_fn: Function to generate decoding order.
    sampling_strategy: "temperature" (autoregressive) or "straight_through" (iterative).
    use_concrete: Use Gumbel-Softmax for straight_through.
    tau_start: Start temperature for Gumbel-Softmax.
    tau_end: End temperature for Gumbel-Softmax.

  Returns:
    A function that samples sequences from structures.

  """
  del _num_encoder_layers, _num_decoder_layers

  if sampling_strategy == "straight_through":
    # Construct a default stage_set with arithmetic_mean strategy (default for STE).
    # Callers can pass state_weights at sample time to override the logit_transform.
    default_stage_set = make_stage_set(strategy="arithmetic_mean")

    optimize_fn = optimize_ste.make_optimize_sequence_fn(
      model,
      default_stage_set,
      decoding_order_fn,
      use_concrete=use_concrete,
      tau_start=tau_start,
      tau_end=tau_end,
    )

    @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state", "num_groups"))
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      bias: jax.Array | None = None,
      fixed_mask: jax.Array | None = None,
      fixed_tokens: jax.Array | None = None,
      backbone_noise: float | None = None,
      iterations: int = 100,
      learning_rate: float = 0.01,
      temperature: float = 1.0,
      tie_group_map: jax.Array | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal[
        "arithmetic_mean", "geometric_mean", "product",
      ] = "arithmetic_mean",
      multi_state_temperature: float = 1.0,
      state_weights: jax.Array | None = None,
      use_rolling_state: bool = False,
      ligand_coords: jax.Array | None = None,
      ligand_atom_types: jax.Array | None = None,
      ligand_mask: jax.Array | None = None,
      wave_schedule: WaveScheduleBundle | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:

      L = (
        structure_coordinates.shape[1]
        if structure_coordinates.ndim == 4
        else structure_coordinates.shape[0]
      )
      S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

      bundle, config = build_inference_bundle(
        coords=structure_coordinates,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
        bias=bias,
        tie_group_map=tie_group_map,
        state_weights=state_weights,
        ligand_coords=ligand_coords,
        ligand_atom_types=ligand_atom_types,
        ligand_mask=ligand_mask,
        temperature=temperature,
        mode="score_conditional",
        inference=True,
      )
      if wave_schedule is not None:
        bundle = eqx.tree_at(lambda b: b.wave, bundle, wave_schedule)

      final_seq, final_logits, _ = optimize_fn(
        prng_key,
        bundle,
        config,
        iterations,
        learning_rate,
        temperature,
        use_rolling_state=use_rolling_state,
      )
      return final_seq, final_logits, jnp.arange(L)

    return cast("SamplerFn", sample_sequences)

  if sampling_strategy == "temperature":

    @partial(
      jax.jit,
      static_argnames=("multi_state_strategy", "use_rolling_state", "num_groups", "inference_only"),
    )
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      bias: jax.Array | None = None,
      fixed_mask: jax.Array | None = None,
      fixed_tokens: jax.Array | None = None,
      backbone_noise: float | None = None,
      temperature: float = 1.0,
      tie_group_map: jax.Array | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal[
        "arithmetic_mean", "geometric_mean", "product",
      ] = "arithmetic_mean",
      multi_state_temperature: float = 1.0,
      state_weights: jax.Array | None = None,
      use_rolling_state: bool = False,
      ligand_coords: jax.Array | None = None,
      ligand_atom_types: jax.Array | None = None,
      ligand_mask: jax.Array | None = None,
      precomputed_node_features: jax.Array | None = None,
      precomputed_edge_features: jax.Array | None = None,
      precomputed_neighbor_indices: jax.Array | None = None,
      wave_schedule: WaveScheduleBundle | None = None,
      inference_only: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:

      L = (
        structure_coordinates.shape[1]
        if structure_coordinates.ndim == 4
        else structure_coordinates.shape[0]
      )
      S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

      k_order, prng_key = jax.random.split(prng_key)
      decoding_order, _ = decoding_order_fn(k_order, L, None, None)

      from aminx.utils.autoregression import generate_ar_mask

      ar_mask_single = generate_ar_mask(
        decoding_order if decoding_order is not None else jnp.arange(L),
        tie_group_map=tie_group_map,
        num_groups=num_groups,
      )

      bundle, config = build_inference_bundle(
        coords=structure_coordinates,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
        bias=bias,
        tie_group_map=tie_group_map,
        state_weights=state_weights,
        ar_mask=ar_mask_single,
        ligand_coords=ligand_coords,
        ligand_atom_types=ligand_atom_types,
        ligand_mask=ligand_mask,
        temperature=temperature,
        mode="sample_ar",
        inference=True,
      )
      stage_set = make_stage_set(
        strategy=multi_state_strategy,
        strategy_temperature=multi_state_temperature,
        state_weights=state_weights,
      )
      if wave_schedule is not None:
        bundle = eqx.tree_at(lambda b: b.wave, bundle, wave_schedule)

      result = sample_autoregressive.kernel(
        model, prng_key, bundle, config, stage_set, inference_only=inference_only,
      )

      return result.sequence.astype(jnp.int8), result.logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)


SAMPLERS.register("make_sample_sequences")(make_sample_sequences)
