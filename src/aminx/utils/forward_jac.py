"""Forward-mode categorical Jacobian, tiled over the tangent axis via xtrax.

The categorical Jacobian is ``J[i, a, j, b] = d logits[i, a] / d one_hot[j, b]`` --
shape ``(L, 21, L, 21)``.

Why this is tiled at all
------------------------
``jax.jacfwd`` is a ``vmap`` of JVPs over the *entire* tangent basis at once, and the
basis here has ``L * 21`` elements. Every one of those tangents carries a full set of
decoder activations, so peak memory scales with ``L * 21 * (per-tangent activation)``
rather than with the ``(L, 21, L, 21)`` output. At L=242 the output is a manageable
~103 MB while the unchunked activation footprint is on the order of 10^2 GB -- the
output was never the problem.

So the tangent axis is planned and dispatched through xtrax
(:func:`aminx.tiling.planner.plan_axis_strategy` +
:func:`aminx.tiling.dispatch.make_axis_dispatch_via_xtrax`), exactly as the candidate
axis is in ``sampling/mbr_consensus.py`` and the replicate/candidate axes are in
``sampling/conditional_logits.py``. Small proteins keep the full-parallel ``Vmap`` path;
larger ones are auto-demoted to ``SafeMap`` against the device memory budget. Per the
standing project rule (``using-xtrax`` skill, ``feedback_xtrax_composable_primitives``
memory), the chunking is **never** a hand-rolled ``jax.vmap``/``lax.map`` here.

This wires the axis template ``aminx.tiling.axes.N_JACOBIAN_PAIRS``, which was scaffolded
for exactly this purpose and left marked DEFERRED, and gives
``JacobianSpecification.jacobian_batch_size`` a real effect for the first time -- it was
previously declared on the CLI and the spec but referenced nowhere, so a user hitting an
OOM could set it and observe no change.

Numerical contract: the tiled result is the same computation as ``jax.jacfwd``, differing
only in the order JVPs are batched. ``tests/utils/test_forward_jac_tiling.py`` pins that
against the reference implementation kept below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from xtrax.tiling import ChunkedMap

from aminx.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
from aminx.tiling.axes import N_JACOBIAN_PAIRS
from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax
from aminx.tiling.planner import plan_axis_strategy
from aminx.types.arrays import CategoricalJacobian
from aminx.utils.autoregression import full_context_ar_mask

NUM_TOKENS = 21

# Fallbacks for models that do not expose these; only used to size the memory estimate,
# never to shape an array.
_DEFAULT_K_NEIGHBORS = 48
_DEFAULT_HIDDEN = 128
_DEFAULT_DECODER_LAYERS = 3
# A JVP holds primal and tangent simultaneously.
_JVP_PRIMAL_AND_TANGENT = 2
_BYTES_PER_FLOAT32 = 4


def _full_context_ar_mask(seq_len: int) -> jax.Array:
  """Deprecated alias for :func:`aminx.utils.autoregression.full_context_ar_mask`.

  Promoted out of this module in the #4222 pass: it had four consumers (this module, the
  runner's reverse-mode path, the runner's ``decoded_node_features`` feature, and the
  split ``decode_fn`` default), which is three too many for a leading underscore in the
  Jacobian module. Behavior is byte-identical, only relocated. The full rationale for
  ``1 - I`` now lives on the public function.

  The original docstring is kept below because it is the record of how the defect was
  found.

  ---

  Every position sees every other position's sequence, but not its own.

  ``ar_mask[i, j] == 1`` means position ``i`` SEES position ``j``'s sequence: the decoder
  gathers it into ``attention_mask`` and uses it as ``mask_bw`` to gate the *sequence*
  edge features, while ``1 - attention_mask`` gates the structure-only path
  (``model/decoder.py:144-147``). So an all-zero ``ar_mask`` admits no sequence
  information at all.

  This function exists because that is exactly what this module used to pass. With
  ``ar_mask = zeros``, the logits were a function of structure alone, their derivative
  with respect to ``one_hot`` was **identically zero**, and the categorical Jacobian came
  back as an all-zero ``(L, 21, L, 21)`` tensor -- correct shape, correct dtype, no error,
  no warning. The accompanying comment read "fully conditional / no autoregressive
  masking, every position sees every other", which is precisely backwards.

  ``1 - I`` matches the "full context minus self" default that
  ``inference/bundle_builder.py:227-228`` already documents and uses for
  ``mode="score_conditional"``. Self-exclusion is wanted here independently: ``J[i,:,i,:]``
  is self-dependence, not a coupling, and is dropped by every downstream reduction anyway.
  """
  return full_context_ar_mask(seq_len)


def _activation_bytes_per_tangent(model, seq_len: int) -> float:
  """Live bytes one tangent holds while its JVP is in flight.

  Dominated by the decoder's per-layer edge features, ``(L, K, D)`` float32, held for
  both primal and tangent. This only has to be right to an order of magnitude -- it
  drives the Vmap/SafeMap demotion decision, not any array shape.
  """
  k_neighbors = int(getattr(model, "k_neighbors", _DEFAULT_K_NEIGHBORS) or _DEFAULT_K_NEIGHBORS)
  hidden = int(getattr(model, "hidden_features", _DEFAULT_HIDDEN) or _DEFAULT_HIDDEN)
  layers = int(
    getattr(model, "num_decoder_layers", _DEFAULT_DECODER_LAYERS) or _DEFAULT_DECODER_LAYERS,
  )
  return float(
    _JVP_PRIMAL_AND_TANGENT * layers * seq_len * k_neighbors * hidden * _BYTES_PER_FLOAT32,
  )


def make_categorical_jacobian_fn(
  model,
  *,
  tangent_batch_size: int | None = None,
):
  """Build a function computing the ``(L, 21, L, 21)`` categorical Jacobian.

  Args:
    model: Model exposing encoder/decoder, as accepted by
      :func:`make_encoding_conditional_logits_split_fn`.
    tangent_batch_size: Fixed SafeMap tile over the ``L * 21`` tangent axis. ``None``
      defers to xtrax's ``BatchPlanner``, which picks Vmap or SafeMap against the device
      memory budget. Set this when you already know your ceiling.

  Returns:
    ``compute_categorical_jacobian(prng_key, coords, mask, residue_index, chain_index,
    sequence, *, backbone_noise=0.0) -> (L, 21, L, 21)``.
  """
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  def compute_categorical_jacobian(
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    sequence: jax.Array,
    *,
    backbone_noise: float = 0.0,
  ) -> CategoricalJacobian:
    encoding = encode_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
      structure_mapping=None,
    )
    one_hot = jax.nn.one_hot(sequence, NUM_TOKENS) if sequence.ndim == 1 else sequence

    seq_len = one_hot.shape[0]
    ar_mask = _full_context_ar_mask(seq_len)

    def logits_from_one_hot(oh: jax.Array) -> jax.Array:
      return decode_fn(encoding, oh, ar_mask=ar_mask)

    def jvp_for_tangent(tangent_index: jax.Array) -> jax.Array:
      """One basis tangent e_(j,b) -> d logits / d one_hot[j, b], shape (L, 21)."""
      residue = tangent_index // NUM_TOKENS
      token = tangent_index % NUM_TOKENS
      tangent = jnp.zeros_like(one_hot).at[residue, token].set(1.0)
      _, out_tangent = jax.jvp(logits_from_one_hot, (one_hot,), (tangent,))
      return out_tangent

    n_tangents = seq_len * NUM_TOKENS
    strategy = plan_axis_strategy(
      N_JACOBIAN_PAIRS,
      n_tangents,
      tangent_batch_size,
      activation_bytes_per_element=_activation_bytes_per_tangent(model, seq_len),
    )
    iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_JACOBIAN_PAIRS.name)

    # xtrax's SafeMap requires the iterated cardinality to be an exact multiple of the
    # tile -- a short final chunk RAISES rather than being handled (backlog #4159: the
    # using-xtrax skill documents this as tolerable, and it is not). Real tangent counts
    # are arbitrary (L=242 -> 5082 = 2*3*7*11*11), so pad the iterated indices up and slice
    # the extras off. Padded entries repeat tangent 0: wasted work on at most tile-1
    # tangents, never a wrong value.
    tangent_indices = jnp.arange(n_tangents, dtype=jnp.int32)
    if isinstance(strategy, ChunkedMap) and n_tangents % strategy.batch_size:
      pad = strategy.batch_size - (n_tangents % strategy.batch_size)
      tangent_indices = jnp.concatenate(
        [tangent_indices, jnp.zeros((pad,), dtype=jnp.int32)],
      )

    stacked = iterator(jvp_for_tangent, tangent_indices)[:n_tangents]

    # stacked[t, i, a] with t = j * 21 + b, i.e. axes (j, b, i, a) after the reshape.
    # The Jacobian contract is (i, a, j, b), so move the output pair in front.
    return jnp.transpose(
      stacked.reshape(seq_len, NUM_TOKENS, seq_len, NUM_TOKENS),
      (2, 3, 0, 1),
    )

  return compute_categorical_jacobian


def make_reference_categorical_jacobian_fn(model):
  """Unchunked ``jax.jacfwd`` reference -- the parity target, not for production use.

  Retained so :func:`make_categorical_jacobian_fn`'s tiling can be checked against the
  computation it replaced. It allocates the whole ``L * 21`` tangent basis at once and
  will exhaust memory on anything but small proteins; that is the entire reason the tiled
  path exists.
  """
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  @jax.jit
  def compute_categorical_jacobian(
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    sequence: jax.Array,
    *,
    backbone_noise: float = 0.0,
  ) -> CategoricalJacobian:
    encoding = encode_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
      structure_mapping=None,
    )
    one_hot = jax.nn.one_hot(sequence, NUM_TOKENS) if sequence.ndim == 1 else sequence

    seq_len = one_hot.shape[0]
    ar_mask = _full_context_ar_mask(seq_len)

    def logits_from_one_hot(oh: jax.Array) -> jax.Array:
      return decode_fn(encoding, oh, ar_mask=ar_mask)

    return jax.jacfwd(logits_from_one_hot)(one_hot)

  return compute_categorical_jacobian
