"""Encode-once temperature sampler: one structure encode per call, decode per PRNG key.

``make_sample_sequences(...)`` under ``jax.vmap`` over PRNG keys re-runs the structure
encoder once per key. At ``backbone_noise == 0`` the encoder output does not depend on the
key, but the jaxpr still does (the noise key is split unconditionally), so neither vmap nor
XLA hoists the encoder. ``make_hoisted_sampler`` runs the encoder once per call through
``xtrax.inference.make_hoisted`` and maps only the decode over the keys.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from xtrax.inference import make_hoisted

from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
from aminx.inference.logits import make_stage_set
from aminx.inference.sample_autoregressive import decode_sampling, encode_for_sampling
from aminx.sampling.sample import _DEFAULT_DECODING_ORDER_FN, make_sample_sequences
from aminx.tiling.axes import N_SAMPLES
from aminx.utils.autoregression import decoding_order_from_wave

if TYPE_CHECKING:
  from aminx.types import StageSet
  from aminx.types.bundles import InferenceBundle
  from aminx.types.configs import InferenceConfig
  from aminx.types.encodings import EncoderOutput
  from aminx.types.protocols import ModelProtocol
  from aminx.utils.decoding_order import DecodingOrderFn

_Shared = tuple["InferenceBundle", "InferenceConfig", "StageSet", "EncoderOutput"]
_ArgValue = jax.Array | float | None

__all__ = ["SAMPLER_AXIS", "HoistedSampler", "make_hoisted_sampler"]

# The mapped axis is the sample (PRNG key) axis. ``keys`` is the only input that varies along
# it; xtrax refuses a shared argument with this name, so a key cannot leak into the encoder.
SAMPLER_AXIS = replace(N_SAMPLES, varying_inputs=("keys",))

# Fixed key for the encoder. Valid only because the encoder output does not depend on it at
# ``backbone_noise == 0`` (the hoisted sampler never accepts a noise level).
_ENCODE_KEY_SEED = 0


class HoistedSampler:
  """Callable returned by ``make_hoisted_sampler``; hold one instance across calls.

  Validity. The hoist is valid only when the encoder output does not depend on the PRNG key at
  ``backbone_noise == 0``. Tests cover this for Aminx protein models: A1 (parity with the
  monolithic vmap, including bias, fixed positions, multi-state and tie groups) and A2 (one
  encoder execution per call). Other model families are not verified. For a first use with any
  other family, call with ``check_elements > 0`` and compare against the monolithic sampler.

  Scope. Only the protein encode path is covered by tests. Ligand arguments (``ligand_coords``,
  ``ligand_atom_types``, ``ligand_mask``) are forwarded but LigandMPNN is not verified.

  Not supported, each with a reason:

  * ``backbone_noise``: the hoist is valid only at 0. A per-sample noise draw is per-element
    work, and the encoder would then depend on the key.
  * ``wave_schedule``: a caller with a fixed schedule has no per-key decoding order; use
    ``make_sample_sequences``.
  * ``use_rolling_state``: the autoregressive kernel does not use it.
  """

  def __init__(
    self,
    model: ModelProtocol,
    *,
    decoding_order_fn: DecodingOrderFn,
    num_groups: int | None,
    multi_state_strategy: str,
    multi_state_temperature: float,
    inference_only: bool,
    batch_size: int | None,
  ) -> None:
    """Store the sampler options and build the hoisted map and the parity reference.

    Args:
      model: Aminx model instance.
      decoding_order_fn: Decoding-order drawer, as in ``make_sample_sequences``.
      num_groups: Number of tie groups, static as in ``make_sample_sequences``.
      multi_state_strategy: How multi-state logits are combined.
      multi_state_temperature: Temperature for the multi-state combination.
      inference_only: Use a while loop for the AR wave axis.
      batch_size: Chunk size for the mapped decode; None uses vmap.

    """
    self._model = model
    self._decoding_order_fn = decoding_order_fn
    self._num_groups = num_groups
    self._multi_state_strategy = multi_state_strategy
    self._multi_state_temperature = multi_state_temperature
    self._inference_only = inference_only
    self._custom_fn = None if decoding_order_fn is _DEFAULT_DECODING_ORDER_FN else decoding_order_fn

    self._hoisted = make_hoisted(
      self._shared,
      self._per_element,
      batch_size=batch_size,
      axis=SAMPLER_AXIS,
    )
    # Monolithic reference for the parity probe (``check_elements``): one key at a time.
    monolithic = make_sample_sequences(model, decoding_order_fn=decoding_order_fn)

    def _reference(
      shared_args: dict[str, _ArgValue],
      key: PRNGKeyArray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      return monolithic(
        key,
        shared_args["structure_coordinates"],
        shared_args["mask"],
        shared_args["residue_index"],
        shared_args["chain_index"],
        bias=shared_args["bias"],
        fixed_mask=shared_args["fixed_mask"],
        fixed_tokens=shared_args["fixed_tokens"],
        temperature=shared_args["temperature"],
        tie_group_map=shared_args["tie_group_map"],
        num_groups=self._num_groups,
        multi_state_strategy=self._multi_state_strategy,
        multi_state_temperature=self._multi_state_temperature,
        state_weights=shared_args["state_weights"],
        ligand_coords=shared_args["ligand_coords"],
        ligand_atom_types=shared_args["ligand_atom_types"],
        ligand_mask=shared_args["ligand_mask"],
        inference_only=self._inference_only,
      )

    self._reference = _reference

  def _shared(self, **args: _ArgValue) -> _Shared:
    """Structure-only work, run once per call: bundle, config, stage set and encoder output."""
    bundle, config = build_inference_bundle(
      coords=args["structure_coordinates"],
      mask=args["mask"],
      residue_index=args["residue_index"],
      chain_index=args["chain_index"],
      backbone_noise=0.0,
      fixed_mask=args["fixed_mask"],
      fixed_tokens=args["fixed_tokens"],
      bias=args["bias"],
      tie_group_map=args["tie_group_map"],
      state_weights=args["state_weights"],
      ligand_coords=args["ligand_coords"],
      ligand_atom_types=args["ligand_atom_types"],
      ligand_mask=args["ligand_mask"],
      temperature=args["temperature"],
      mode="sample_ar",
      inference=True,
    )
    stage_set = make_stage_set(
      strategy=self._multi_state_strategy,
      strategy_temperature=self._multi_state_temperature,
      state_weights=args["state_weights"],
    )
    enc = encode_for_sampling(self._model, bundle, jax.random.key(_ENCODE_KEY_SEED), config)
    return bundle, config, stage_set, enc

  def _per_element(
    self,
    shared: _Shared,
    key: PRNGKeyArray,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Decode one key. Mirrors ``make_sample_sequences``' key stream exactly."""
    bundle, config, stage_set, enc = shared
    # Same split order as sample.py (k_order, prng_key) and kernel (k_enc, k_dec).
    k_order, k = jax.random.split(key)
    _, k_dec = jax.random.split(k)
    bundle = with_decoding_order(bundle, k_order, self._custom_fn, self._num_groups)
    decoding_order = decoding_order_from_wave(bundle.wave, bundle.conditioning.tie_group_map[0])
    result = decode_sampling(
      self._model,
      k_dec,
      enc,
      bundle,
      config,
      stage_set,
      inference_only=self._inference_only,
    )
    return result.sequence.astype(jnp.int8), result.logits, decoding_order

  def __call__(
    self,
    keys: jax.Array,
    *,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    bias: jax.Array | None = None,
    fixed_mask: jax.Array | None = None,
    fixed_tokens: jax.Array | None = None,
    temperature: float = 1.0,
    tie_group_map: jax.Array | None = None,
    state_weights: jax.Array | None = None,
    ligand_coords: jax.Array | None = None,
    ligand_atom_types: jax.Array | None = None,
    ligand_mask: jax.Array | None = None,
    check_elements: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-6,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Same ``(seqs, logits, decoding_order)`` as ``vmap(sample_sequences)(keys, ...)``.

    Leading axis N over ``keys`` (typed keys ``(N,)`` or raw ``(N, 2)``). The encoder runs
    once per call. ``temperature`` is converted to float32 so distinct Python floats do not
    recompile the shared stage.

    Args:
      keys: PRNG keys, one per sample.
      structure_coordinates: Backbone coordinates ``(L, 4, 3)`` or ``(S, L, 4, 3)``.
      mask: Residue mask, ``(L,)`` or ``(S, L)``.
      residue_index: Residue indices, same leading shape as ``mask``.
      chain_index: Chain indices, same leading shape as ``mask``.
      bias: Optional sequence bias.
      fixed_mask: Optional mask of fixed positions.
      fixed_tokens: Optional fixed token values.
      temperature: Sampling temperature.
      tie_group_map: Optional tie-group map.
      state_weights: Optional per-state weights (multi-state).
      ligand_coords: Optional ligand coordinates.
      ligand_atom_types: Optional ligand atom types.
      ligand_mask: Optional ligand atom mask.
      check_elements: Number of leading elements to compare with the monolithic sampler
        (0 disables the probe).
      rtol: Relative tolerance for the probe.
      atol: Absolute tolerance for the probe.

    Returns:
      Sequences ``(N, L)`` int8, logits ``(N, L, 21)``, decoding orders ``(N, L)``.

    """
    shared_args = {
      "structure_coordinates": structure_coordinates,
      "mask": mask,
      "residue_index": residue_index,
      "chain_index": chain_index,
      "bias": bias,
      "fixed_mask": fixed_mask,
      "fixed_tokens": fixed_tokens,
      "temperature": jnp.asarray(temperature, dtype=jnp.float32),
      "tie_group_map": tie_group_map,
      "state_weights": state_weights,
      "ligand_coords": ligand_coords,
      "ligand_atom_types": ligand_atom_types,
      "ligand_mask": ligand_mask,
    }
    return cast(
      "tuple[jax.Array, jax.Array, jax.Array]",
      self._hoisted(
        shared_args,
        keys,
        reference=self._reference,
        check_elements=check_elements,
        rtol=rtol,
        atol=atol,
      ),
    )

  def stats(self) -> dict[str, int]:
    """Call counters from the underlying ``HoistedMap``.

    Returns:
      Dict with ``shared_calls`` and ``traced_bypass``.

    """
    return cast("dict[str, int]", self._hoisted.stats())


def make_hoisted_sampler(
  model: ModelProtocol,
  *,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  num_groups: int | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
  inference_only: bool = False,
  batch_size: int | None = None,
) -> HoistedSampler:
  """Build an encode-once temperature sampler over PRNG keys (see ``HoistedSampler``).

  The hoist is valid only when the encoder output is key-independent at zero noise. This holds
  for Aminx protein models (covered by tests A1 and A2). For other model families, run the
  first call with ``check_elements > 0`` and compare with ``make_sample_sequences``.

  Args:
    model: Aminx model instance.
    decoding_order_fn: Decoding-order drawer, as in ``make_sample_sequences``.
    num_groups: Number of tie groups, static.
    multi_state_strategy: How multi-state logits are combined.
    multi_state_temperature: Temperature for the multi-state combination.
    inference_only: Use a while loop for the AR wave axis.
    batch_size: Chunk size for the mapped decode; None uses vmap.

  Returns:
    A ``HoistedSampler`` to call with ``keys`` and the structure keyword arguments.

  """
  return HoistedSampler(
    model,
    decoding_order_fn=decoding_order_fn,
    num_groups=num_groups,
    multi_state_strategy=multi_state_strategy,
    multi_state_temperature=multi_state_temperature,
    inference_only=inference_only,
    batch_size=batch_size,
  )
