"""ProtonPottsMPNN decoder over the 30-token v6 alphabet (debt #2617).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §53. Upstream is foundry's
``ProteinMPNN.decode_auto_regressive`` / ``decode_teacher_forcing`` (``mpnn/model/mpnn.py``), inherited by ``PottsMPNN``.
Both are graded against the P4h dump (``scripts/protonpotts/dump_protonpotts_decoder.py``) by the wave
``protonpotts_decoder``.

What this module is, and is not:

- It REUSES the graded Potts building blocks by import and edits none of them (``forward_context``, ``update_row``,
  ``readout``, ``embed_at``, ``neighbor_valid``, ``categorical_draw`` from ``potts_mpnn.decode``; the stock
  ``Decoder.call_conditional`` for the teacher-forced modes), so no PottsMPNN ledger row is staled.
- It differs from ``PottsARDecode`` where foundry differs: the temperature is PER RESIDUE, the bias is a per-residue
  ``(L, V)`` array added to the logits before the division, and the sampling distribution is the softmax with the
  unknown token (``X``) zeroed and the rest renormalised (``logits_to_sample``), not the Potts PSSM mix.
- It is UNTIED. foundry's ``symmetry_equivalence_group`` is not ported: the pH engine does not use it, and a tied
  decode is refused rather than guessed.
- The decoding order is ``argsort((designed + 1e-4) * |noise|)``, foundry's formula with ``decode_last_mask`` the
  designed mask: fixed residues are decoded first, in noise order. Pad rows are decoded last (aminx's convention);
  foundry has no padding, so this is not graded against it and is covered by an internal padding-invariance test.

``sample_rows`` is N such decodes over shared encoder states (the engine's ``mpnn_sample``).

Token sampling is the inverse CDF on an injected uniform (``categorical_draw``); the oracle replaces
``torch.multinomial`` with the same convention. The draw for decoding step ``k`` is ``uniforms[k]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.potts_mpnn.decode import (
  categorical_draw,
  embed_at,
  forward_context,
  neighbor_valid,
  readout,
  update_row,
)
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

if TYPE_CHECKING:
  from aminx.model.decoder import Decoder, DecoderLayer

TEACHER_FORCING_PATTERNS = ("auto_regressive", "conditional", "conditional_minus_self")
ORDER_EPS = 1.0e-4  # foundry ``decoding_eps``


class DecodeResult(NamedTuple):
  sequence: Int[Array, " L"]  # sampled tokens; fixed positions hold their native token
  decoding_order: Int[Array, " L"]  # decoding_order[k] = the position decoded at step k
  logits: Float[Array, "L V"]  # raw W_out output per position
  log_probs: Float[Array, "L V"]  # log_softmax((logits + bias) / temperature) per position
  probs_sample: Float[Array, "L V"]  # softmax with X zeroed and renormalised, the distribution drawn from
  drawn_token: Int[Array, " L"]  # the draw at each position, BEFORE a fixed position is reset to native
  h_v_stack: Float[Array, "layers L H"]  # decoder layer outputs per position (row 0 is the encoder's h_V)


def decoding_order_from_noise(
  designed: Bool[Array, " L"],
  pad_valid: Bool[Array, " L"],
  noise: Float[Array, " L"],
) -> Int[Array, " L"]:
  """``argsort((designed + eps) * |noise|)`` (foundry ``setup_causality_masks``), pad rows last."""
  key = (designed.astype(noise.dtype) + jnp.asarray(ORDER_EPS, dtype=noise.dtype)) * jnp.abs(noise)
  key = jnp.where(pad_valid, key, jnp.asarray(jnp.inf, dtype=key.dtype))
  return jnp.argsort(key).astype(jnp.int32)


def _unknown_mask(vocab: int, dtype: jnp.dtype) -> Float[Array, " V"]:
  """1.0 at the token upstream never samples (``X``; ``unknown_token_indices`` is ``[X]``)."""
  return jnp.zeros((vocab,), dtype=dtype).at[PROTONPOTTS_V6.x_index].set(1.0)


class ProtonPottsARDecode(eqx.Module):
  """Incremental autoregressive decode of one structure over the v6 vocabulary. Fields are the model's own modules."""

  layers: tuple[DecoderLayer, ...]
  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  def __call__(
    self,
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
    *,
    decoding_order: Int[Array, " L"] | None = None,
    noise: Float[Array, " L"] | None = None,
    forced_tokens: Int[Array, " L"] | None = None,
    cdf_order: Int[Array, " V"] | None = None,
  ) -> DecodeResult:
    """Decode. Give ``decoding_order`` (replay) or ``noise`` (the order is drawn from it). ``uniforms[k]`` is step ``k``.

    ``forced_tokens`` (``-1`` = none) replaces the token a position CONTRIBUTES to later steps with a given one, while
    ``drawn_token`` still records this decode's own draw. It exists for the oracle wave: it checks every step's draw and
    distribution on the oracle's own prefix, so one numerically fragile draw cannot cascade into the rest of the sequence.

    ``cdf_order`` (default None: aminx token order) is the token order in which the inverse CDF accumulates; entry ``j`` is
    the aminx index of the ``j``-th token. Any order is a valid sampler, but a given uniform selects a different token in each,
    so parity with an upstream dump that recorded the draws needs upstream's order (``convert.token_permutation()``). Same
    contract as ``ph_descent.select_joint``.
    """
    dtype = h_v.dtype
    length = h_v.shape[0]
    vocab = bias.shape[-1]
    if decoding_order is None:
      if noise is None:
        msg = "give decoding_order or noise"
        raise ValueError(msg)
      order = decoding_order_from_noise(designed & pad_valid, pad_valid, noise.astype(dtype))
    else:
      order = decoding_order.astype(jnp.int32)
    rank = jnp.argsort(order).astype(jnp.int32)
    present = present.astype(dtype)
    mask_bw, h_exv_fw = forward_context(h_v, h_e, e_idx, rank, present)
    nbr = neighbor_valid(e_idx, pad_valid)
    unknown = _unknown_mask(vocab, dtype)
    temperature = temperature.astype(dtype)
    bias = bias.astype(dtype)
    uniforms = uniforms.astype(dtype)
    n_layers = len(self.layers)
    h_v_stack = jnp.concatenate([h_v[None], jnp.zeros((n_layers, *h_v.shape), dtype=dtype)], axis=0)

    def step(
      carry: tuple[Array, Array, Array, Array, Array, Array, Array],
      k: Array,
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
      h_v_c, h_s, seq, logits_all, log_probs_all, probs_all, drawn_all = carry
      pos = order[k]
      h_v_c = update_row(self.layers, h_v_c, h_s, h_e, e_idx, mask_bw, h_exv_fw, present, nbr, pos)
      logits = readout(self.w_out, h_v_c[-1, pos])
      modified = (logits + bias[pos]) / temperature[pos]
      log_probs = jax.nn.log_softmax(modified)
      probs = jax.nn.softmax(modified)
      kept = probs * (1.0 - unknown)
      probs_sample = kept / jnp.sum(kept)
      if cdf_order is None:
        drawn = categorical_draw(probs_sample, uniforms[k])
      else:
        # Accumulate the CDF in another token order (upstream's), then map the draw back to an aminx token.
        drawn = cdf_order[categorical_draw(probs_sample[cdf_order], uniforms[k])]
      token = jnp.where(designed[pos], drawn, s_true[pos]).astype(jnp.int32)
      if forced_tokens is not None:
        token = jnp.where(forced_tokens[pos] >= 0, forced_tokens[pos], token).astype(jnp.int32)
      seq = seq.at[pos].set(token)
      h_s = h_s.at[pos].set(embed_at(self.w_s_embed, token))
      logits_all = logits_all.at[pos].set(logits)
      log_probs_all = log_probs_all.at[pos].set(log_probs)
      probs_all = probs_all.at[pos].set(probs_sample)
      drawn_all = drawn_all.at[pos].set(drawn.astype(jnp.int32))
      return (h_v_c, h_s, seq, logits_all, log_probs_all, probs_all, drawn_all), None

    zeros_lv = jnp.zeros((length, vocab), dtype=dtype)
    init = (
      h_v_stack,
      jnp.zeros_like(h_v),
      jnp.zeros((length,), dtype=jnp.int32),
      zeros_lv,
      zeros_lv,
      zeros_lv,
      jnp.zeros((length,), dtype=jnp.int32),
    )
    (h_v_stack, _h_s, seq, logits_all, log_probs_all, probs_all, drawn_all), _unused = jax.lax.scan(
      step,
      init,
      jnp.arange(length, dtype=jnp.int32),
    )
    return DecodeResult(
      sequence=seq,
      decoding_order=order,
      logits=logits_all,
      log_probs=log_probs_all,
      probs_sample=probs_all,
      drawn_token=drawn_all,
      h_v_stack=h_v_stack,
    )


def sample_rows(
  decoder: ProtonPottsARDecode,
  h_v: Float[Array, "L H"],
  h_e: Float[Array, "L K H"],
  e_idx: Int[Array, "L K"],
  present: Float[Array, " L"],
  pad_valid: Bool[Array, " L"],
  s_true: Int[Array, " L"],
  designed: Bool[Array, " L"],
  temperature: Float[Array, " L"],
  bias: Float[Array, "L V"],
  uniforms: Float[Array, "N L"],
  *,
  noise: Float[Array, "N L"] | None = None,
  decoding_order: Int[Array, "N L"] | None = None,
  cdf_order: Int[Array, " V"] | None = None,
) -> DecodeResult:
  """``N`` independent decodes of ONE structure sharing its encoder states: the engine's ``mpnn_sample`` (debt #2617).

  Upstream runs the encoder once (``B = 1``), repeats the sample along the batch dimension (``repeat_sample_num = N``) and lets each
  row draw its own decoding-order noise and its own tokens. Here row ``r`` is one ``ProtonPottsARDecode`` call with
  ``uniforms[r]`` and ``noise[r]`` (or ``decoding_order[r]`` for replay); every other argument is shared. Every field of the
  result gains a leading axis of length ``N``. Row ``r`` equals the single-row call exactly; there is no coupling between rows.
  """
  if (noise is None) == (decoding_order is None):
    msg = "give exactly one of noise or decoding_order, each of shape (N, L)"
    raise ValueError(msg)

  def one(u: Array, per_row: Array) -> DecodeResult:
    return decoder(
      h_v, h_e, e_idx, present, pad_valid, s_true, designed, temperature, bias, u,
      noise=per_row if noise is not None else None,
      decoding_order=per_row if decoding_order is not None else None,
      cdf_order=cdf_order,
    )

  return jax.vmap(one)(uniforms, noise if noise is not None else decoding_order)


def teacher_forcing_mask(
  pattern: str,
  decoding_order: Int[Array, " L"],
  dtype: jnp.dtype,
) -> Float[Array, "L L"]:
  """``ar_mask[i, j] = 1`` when position ``j`` is visible to position ``i`` (foundry ``causality_pattern``)."""
  length = decoding_order.shape[0]
  if pattern == "auto_regressive":
    rank = jnp.argsort(decoding_order)
    return (rank[None, :] < rank[:, None]).astype(dtype)
  if pattern == "conditional":
    return jnp.ones((length, length), dtype=dtype)
  if pattern == "conditional_minus_self":
    return jnp.ones((length, length), dtype=dtype) - jnp.eye(length, dtype=dtype)
  msg = f"unknown causality pattern {pattern!r}; expected one of {TEACHER_FORCING_PATTERNS}"
  raise ValueError(msg)


def teacher_forced(
  decoder: Decoder,
  w_s_embed: eqx.nn.Embedding,
  w_out: eqx.nn.Linear,
  h_v: Float[Array, "L H"],
  h_e: Float[Array, "L K H"],
  e_idx: Int[Array, "L K"],
  present: Float[Array, " L"],
  seq: Int[Array, " L"],
  temperature: Float[Array, " L"],
  bias: Float[Array, "L V"],
  *,
  pattern: str,
  decoding_order: Int[Array, " L"],
) -> tuple[Float[Array, "L V"], Float[Array, "L V"]]:
  """``(logits, log_probs)`` of ``seq`` under one causality pattern, sequence embedding initialised from ``seq``.

  The stock conditional decoder does the work; only the visibility mask differs between patterns.
  """
  dtype = h_v.dtype
  vocab = w_s_embed.weight.shape[0]
  ar_mask = teacher_forcing_mask(pattern, decoding_order, dtype)
  one_hot = jax.nn.one_hot(seq, vocab, dtype=dtype)
  hidden = decoder.call_conditional(
    h_v,
    h_e,
    e_idx.astype(jnp.int32),
    present.astype(dtype),
    ar_mask,
    one_hot,
    w_s_embed.weight.astype(dtype),
    inference=True,
  )
  logits = hidden @ w_out.weight.astype(dtype).T
  if w_out.bias is not None:
    logits = logits + w_out.bias.astype(dtype)
  modified = (logits + bias.astype(dtype)) / temperature.astype(dtype)[:, None]
  return logits, jax.nn.log_softmax(modified, axis=-1)
