"""The ``mpnn_sample`` pH-design method: N whole-chain decoder samples of the binder chain (debt #2617).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §55. Upstream is ``potts_mpnn_ph.py`` ``_run_whole_chain``
(lines 1226-1231): the network input is prepared with ``designed_chains=[binder_chain]``, ``repeat_sample_num = N``, the model runs
once and ``decoder_features["S_sampled"]`` ([N, L]) are the designs; each is then scored (``_score_design``). The decode itself is
graded against upstream by the ``protonpotts_sample`` wave (:func:`decode.sample_rows` against the P4i dump).

What this module adds on top of the graded sampler, and what it mirrors on purpose:

- N is ``config.samples_per_site`` (upstream ``num_designs``).
- The temperature is NOT ``config.temperature``. Upstream's ``mpnn_sample`` samples at the temperature the prepared network input
  carries (``prepare_potts_input``'s 0.1, stored float32), whatever the criteria say; this module does the same and logs a warning when
  ``config.temperature`` differs, because a knob that is silently ignored is a trap. No bias, no forbidden-token mask: upstream
  applies the criteria's ``forbidden_tokens`` only when SCORING, never when sampling.
- Positions outside the binder chain keep their native token; the binder chain is decoded after them (``decoding_order_from_noise``).
- A design has no centres: ``pins == ()``, ``selective_energy == 0.0``, ``label == ""``. Its energy is the whole-sequence Potts
  energy on the merged table (:func:`ph_design._score`'s first value), the same score block descent reports.
- The draws come from a JAX key (no upstream anchor, as for the other methods) or are injected (``uniforms`` and ``noise``, each
  ``(N, L)``) for replay; ``uniforms[r, k]`` is row ``r``'s draw at decoding step ``k`` and ``cdf_order`` is upstream's token order.
- Not ported from ``_score_design``: ``sequence_decoded_prob_score`` and ``sequence_entropy`` (aminx's :class:`PHDesign` has no such
  fields), so a downstream that wants them recomputes them from the sequence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states, cast_floating
from aminx.families.protonpotts_mpnn.decode import ProtonPottsARDecode, sample_rows
from aminx.families.protonpotts_mpnn.ph_design import PHDesign
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

if TYPE_CHECKING:
  from collections.abc import Sequence

  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig

logger = logging.getLogger(__name__)

# prepare_potts_input's temperature: 0.1 stored float32 in the features, so float32(0.1) widened (the P4h finding).
UPSTREAM_SAMPLE_TEMPERATURE = float(np.float32(0.1))


def _states(
  model: PottsMPNN,
  coords: jax.Array,
  present: jax.Array,
  residue_idx: jax.Array,
  chain_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """The encoder's final node and edge states and the neighbour index (the decoder's inputs)."""
  dtype = coords.dtype
  scored = cast_floating(model, dtype)
  nodes, edges, e_idx = _encoder_states(
    scored.mpnn,
    coords,
    present.astype(dtype),
    residue_idx.astype(jnp.int32),
    chain_index.astype(jnp.int32),
    inference=True,
  )
  return nodes[-1], edges[-1], e_idx


_JIT_STATES = eqx.filter_jit(_states)


def _cdf_order() -> jax.Array:
  from aminx.families.protonpotts_mpnn import convert  # noqa: PLC0415

  return jnp.asarray(np.argsort(convert.token_permutation()).astype(np.int32))


def mpnn_sample_designs(
  model: PottsMPNN,
  graph_args: Sequence[jax.Array],
  table: jax.Array,
  e_idx: jax.Array,
  native: np.ndarray,
  binder_mask: np.ndarray,
  config: PHDesignConfig,
  *,
  key: jax.Array | None = None,
  uniforms: np.ndarray | None = None,
  noise: np.ndarray | None = None,
) -> list[PHDesign]:
  """``config.samples_per_site`` whole-chain decoder samples of one structure, each scored by its Potts energy.

  ``graph_args`` is ``(coords, present, residue_idx, chain_index, pad_valid)`` as the driver builds it; ``table`` and ``e_idx`` are the
  merged Potts table and neighbour index of the same structure. ``native`` and ``binder_mask`` are ``(L,)``. Give ``key`` (draws) or
  both ``uniforms`` and ``noise`` (replay), each ``(N, L)``. Designs are returned in sample order, as upstream's ``_run_whole_chain``
  appends them.
  """
  binder_mask = np.asarray(binder_mask, dtype=bool)
  if not binder_mask.any():
    msg = "binder_mask has no True entries: no designable binder position"
    raise ValueError(msg)
  if (uniforms is None) != (noise is None):
    msg = "give uniforms and noise together (replay) or neither (draw from key)"
    raise ValueError(msg)
  if uniforms is None and key is None:
    msg = "mpnn_sample needs a jax PRNG key or injected uniforms and noise"
    raise ValueError(msg)
  if abs(config.temperature - UPSTREAM_SAMPLE_TEMPERATURE) > 1e-12:
    logger.warning(
      "mpnn_sample ignores temperature=%s: upstream samples at the prepared input's 0.1 (debt #2617, spec 55)",
      config.temperature,
    )
  coords, present, residue_idx, chain_index, pad_valid = graph_args
  h_v, h_e, nbr = _JIT_STATES(model, coords, present, residue_idx, chain_index)
  dtype = h_v.dtype
  length = int(native.shape[0])
  n_rows = int(config.samples_per_site)
  if uniforms is None:
    if key is None:  # unreachable (checked above); narrows the type for the checker
      raise ValueError("mpnn_sample needs a jax PRNG key or injected uniforms and noise")
    k_u, k_n = jax.random.split(key)
    draws_u = jax.random.uniform(k_u, (n_rows, length), dtype=dtype)
    draws_n = jax.random.normal(k_n, (n_rows, length), dtype=dtype)
  else:
    draws_u, draws_n = jnp.asarray(uniforms, dtype=dtype), jnp.asarray(noise, dtype=dtype)
    if draws_u.shape != (n_rows, length) or draws_n.shape != (n_rows, length):
      msg = f"uniforms and noise must be ({n_rows}, {length}); got {draws_u.shape} and {draws_n.shape}"
      raise ValueError(msg)

  scored = cast_floating(model, dtype)
  decoder = ProtonPottsARDecode(layers=scored.mpnn.decoder.layers, w_s_embed=scored.mpnn.w_s_embed, w_out=scored.mpnn.w_out)
  out = sample_rows(
    decoder,
    h_v,
    h_e,
    nbr,
    present.astype(dtype),
    pad_valid.astype(bool),
    jnp.asarray(native, dtype=jnp.int32),
    jnp.asarray(binder_mask),
    jnp.full((length,), UPSTREAM_SAMPLE_TEMPERATURE, dtype=dtype),
    jnp.zeros((length, PROTONPOTTS_V6.size), dtype=dtype),
    draws_u,
    noise=draws_n,
    cdf_order=_cdf_order(),
  )
  sequences = np.asarray(out.sequence, dtype=np.int32)
  valid = jnp.ones(length, dtype=bool)
  designable = tuple(int(i) for i in np.flatnonzero(binder_mask))
  designs: list[PHDesign] = []
  for sample in range(n_rows):
    seq = sequences[sample]
    energy = float(potts_energy(table, e_idx, valid, jnp.asarray(seq, dtype=jnp.int32)))
    designs.append(
      PHDesign(
        sequence=seq,
        method="mpnn_sample",
        sample=sample,
        pins=(),
        designable=designable,
        label="",
        final_potts_energy=energy,
        selective_energy=0.0,
        selective_energies=(),
        n_draws=len(designable),
      ),
    )
  return designs
