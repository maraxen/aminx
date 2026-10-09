"""Fixed-shape functions of the 30-token decoder, for export to ONNX and a JS sampler (ADR decision 12, spec §53-54).

Two graphs, weights baked in, every runtime value an input:

- ``make_encode`` -> ``encode``: coordinates and geometry to the encoder's final states ``(h_V, h_E, E_idx)``, once per structure.
- ``make_decode`` -> ``decode``: the autoregressive sampler (:class:`decode.ProtonPottsARDecode`) over those states, with the noise
  that fixes the decoding order and the uniforms that fix the draws as INPUTS (the JS side generates them), per-residue
  temperature and bias as inputs, and the designed mask as an input. It returns ``(sequence, decoding_order, log_probs)``.

The inverse CDF accumulates in UPSTREAM's token order (``decode``'s ``cdf_order``, baked in as a constant): any order is a valid
sampler, but with upstream's, the same uniforms and the same decoding order give the same tokens as upstream's own sampler (graded by
the ``protonpotts_decoder`` wave). Output flags and the designed mask are int32/bool arrays; there are no boolean outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states, cast_floating
from aminx.families.protonpotts_mpnn import convert
from aminx.families.protonpotts_mpnn.decode import ProtonPottsARDecode

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Bool, Float, Int


def upstream_cdf_order() -> np.ndarray:
  """``cdf_order[j]`` = the aminx index of upstream's ``j``-th token (the inverse of the converter's permutation)."""
  return np.argsort(convert.token_permutation()).astype(np.int32)


def make_encode(model: PottsMPNN) -> Callable[..., tuple[Array, Array, Array]]:
  """The encoder graph function: float32 geometry in, ``(h_V, h_E, E_idx)`` out."""
  scored = cast_floating(model, jnp.float32)

  def encode(
    coords: Float[Array, "L 4 3"],
    present: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_index: Int[Array, " L"],
  ) -> tuple[Array, Array, Array]:
    nodes, edges, e_idx = _encoder_states(
      scored.mpnn, coords, present, residue_idx.astype(jnp.int32), chain_index.astype(jnp.int32), inference=True,
    )
    return nodes[-1], edges[-1], e_idx.astype(jnp.int32)

  return encode


def make_decode(model: PottsMPNN) -> Callable[..., tuple[Array, Array, Array]]:
  """The decoder graph function over the encoder states."""
  scored = cast_floating(model, jnp.float32)
  decoder = ProtonPottsARDecode(
    layers=scored.mpnn.decoder.layers, w_s_embed=scored.mpnn.w_s_embed, w_out=scored.mpnn.w_out,
  )
  order = jnp.asarray(upstream_cdf_order())

  def decode(
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K H"],
    e_idx: Int[Array, "L K"],
    present: Float[Array, " L"],
    pad_valid: Bool[Array, " L"],
    s_true: Int[Array, " L"],
    designed: Bool[Array, " L"],
    temperature: Float[Array, " L"],
    bias: Float[Array, "L V"],
    uniforms: Float[Array, " L"],
    noise: Float[Array, " L"],
  ) -> tuple[Array, Array, Array]:
    out = decoder(
      h_v, h_e, e_idx, present, pad_valid, s_true, designed, temperature, bias, uniforms, noise=noise, cdf_order=order,
    )
    return out.sequence, out.decoding_order, out.log_probs

  return decode
