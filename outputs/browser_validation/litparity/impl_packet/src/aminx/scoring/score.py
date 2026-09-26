"""Score a given sequence on a structure using the ProteinMPNN model."""

from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from aminx.inference import score_conditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.types.protocols import ModelProtocol, ScoreFn
from aminx.utils.autoregression import full_context_ar_mask
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)

SCORE_EPS = 1e-8


def _residue_mask_for_scoring(mask: jax.Array) -> jax.Array:
  """Return per-residue mask (L,) for masked-average NLL scoring."""
  if mask.ndim == 1:
    return mask
  return mask[0]


def _nll_from_logits(
  logits: jax.Array,
  seq_one_hot: jax.Array,
  mask: jax.Array,
) -> jax.Array:
  """Compute negative log-likelihood from logits and one-hot sequence.

  Computes masked-average NLL over the FULL 21-token vocabulary, matching the reference
  aggregator (``data_utils.get_score``, which one-hots over all 21).

  This used to slice ``[..., :20]`` on both factors, on the stated grounds that it made
  padded positions contribute 0. **Padding is not what that slice was doing.** ``mask`` is
  already 0 at padded positions, which removes them from the numerator (``score *
  mask_flat``) AND the denominator -- verified directly: the scoring denominator is 214 for
  a 214-residue chain at both ``max_length=214`` and ``max_length=512``. The slice was
  therefore redundant for padding and had exactly one live effect: it charged 0 nats for a
  *real* ``X`` residue while still counting that position in the denominator, silently
  diluting the mean. 1LVB carries six genuine ``X`` (selenomethionines at canonical 74, 79,
  113, 116, 179, 210), so this was not hypothetical.

  Args:
    logits: Model output logits, shape (..., 21) where last axis is vocab.
    seq_one_hot: One-hot encoded sequence, shape (..., 21).
    mask: Per-residue mask for scoring. If 2-D (S, L), uses first state's mask.
      Must be 0 at padded positions -- that, not the vocabulary, is what excludes padding.

  Returns:
    Scalar NLL (negative log-likelihood), masked and averaged.
  """
  # Accept either one-hot ``(..., 21)`` or raw token indices ``(...,)``. Both callers
  # forward whatever they were handed (``host/runner.py`` one-hots first, so production was
  # always one-hot; ``make_score_fn``'s own low-level callers frequently pass indices).
  #
  # This normalization is not cosmetic. Under the previous ``[..., :20]`` slice a token-index
  # array was sliced on its ONLY axis, yielding the first 20 *positions* rather than 20 vocab
  # entries -- which then broadcast against ``(L, 20)`` log-probs whenever ``L > 20`` and
  # produced a silently meaningless number (measured on a synthetic L=76 case: 530.7 against a
  # true 3.40), while raising a shape error for ``L <= 20``. ``tests/scoring/test_score.py``
  # exercised exactly that path and passed, because it only asserted shape and dtype.
  # Discriminate on the VOCAB axis alone. Comparing ``ndim`` instead is wrong: multi-state
  # logits are ``(S, L, 21)`` while the one-hot stays ``(L, 21)``, and that legitimate
  # rank mismatch would be misread as indices. (Ambiguous only for a token array whose
  # trailing axis happens to equal the vocab size; pass one-hot explicitly in that case.)
  vocab_size = logits.shape[-1]
  if seq_one_hot.shape[-1] != vocab_size:
    seq_one_hot = jax.nn.one_hot(seq_one_hot.astype("int32"), vocab_size)

  log_probability = jax.nn.log_softmax(logits, axis=-1)
  score = -(seq_one_hot * log_probability).sum(-1)

  # Use the first state's mask when batched (S, L); use full vector when 1-D (L,).
  mask_flat = _residue_mask_for_scoring(mask)
  masked_score_sum = (score * mask_flat).sum(-1)
  mask_sum = mask_flat.sum() + SCORE_EPS

  return masked_score_sum / mask_sum


def make_score_fn(
  model: ModelProtocol,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  inference: bool = True,  # noqa: FBT001, FBT002
) -> ScoreFn:
  """Create a function to score a sequence on a structure using Aminx.

  Args:
    model: Protein or Ligand Equinox checkpoint.
    decoding_order_fn: Decoding order.
    inference: Use ``eqx.nn.inference_mode`` when True.

  Returns:
    JIT scoring function.

  """
  del _num_encoder_layers, _num_decoder_layers

  if inference and isinstance(model, eqx.Module):
    model = eqx.nn.inference_mode(model, value=True)

  n_aa = (
    int(getattr(model, "w_s_embed", None).num_embeddings) if hasattr(model, "w_s_embed") else 21
  )

  @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state"))
  def score_sequence(
    prng_key: jax.Array,
    sequence: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    backbone_noise: float | None = None,
    ar_mask: jax.Array | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    state_position_map: jax.Array | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
    state_weights: jax.Array | None = None,
    bias: jax.Array | None = None,
    use_rolling_state: bool = False,
    ligand_coords: jax.Array | None = None,
    ligand_atom_types: jax.Array | None = None,
    ligand_mask: jax.Array | None = None,
    **kwargs,  # Accept but ignore extra kwargs (e.g., _k_neighbors for backward compat)
  ) -> tuple[jax.Array, jax.Array, jax.Array]:

    L = sequence.shape[0]
    S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

    # The decoding order is still drawn (and returned) so callers that want an
    # autoregressive factorization can pass the matching ``ar_mask`` back in, but it no
    # longer selects the default mask -- see the ``ar_mask is None`` branch below.
    decoding_order, prng_key = decoding_order_fn(prng_key, L, None, None)
    if ar_mask is None:
      # Full context minus self (``1 - I``): every position is scored given every OTHER
      # position's sequence, never its own. This is the conditional estimand
      # p(s_i | s_{-i}, X) -- the same quantity the reference exposes as
      # ``--conditional_probs_only`` -- and it is what ``mode="score_conditional"`` means.
      #
      # It replaces ``generate_ar_mask(decoding_order)``, which was wrong twice over:
      #
      # 1. Its untied branch is ``(row_indices >= col_indices)`` -- NON-strict -- so
      #    ``ar_mask[i, i] == 1`` under every permutation. Because a residue is always
      #    among its own KNN neighbours, ``model/decoder.py`` gated that residue's TRUE
      #    one-hot into its own node update. Measured leak on 1LVB chain A: +0.036243
      #    nats (paired over 8 seeds, sd 0.002482, t = 41.3), i.e. the score was
      #    systematically over-confident by ~2.3x the effect sizes this is used to
      #    resolve. Every reference construction is ``1 - triu(ones)``, self-excluding.
      # 2. It made the score depend on the PRNG key even at ``backbone_noise=0``, since a
      #    fresh order means a fresh mask. That was 100% of the observed seed-to-seed
      #    variance (sd 0.0177 at L=214), and because the runner derives one key per
      #    CANDIDATE, two candidates in one call were scored under different orders --
      #    making every WT/MUT contrast unpaired for no reason.
      #
      # ``full_context_ar_mask`` is order-free, so scoring is now deterministic in the key
      # at ``backbone_noise=0``. ``sampling/conditional_logits.py`` already defaulted this
      # way; this brings the score path in line with it.
      ar_mask_single = full_context_ar_mask(L)
    else:
      ar_mask_single = ar_mask[0] if ar_mask.ndim == 3 else ar_mask

    bundle, config = build_inference_bundle(
      coords=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=sequence,
      backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
      ar_mask=ar_mask_single,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      state_position_map=state_position_map,
      state_weights=state_weights,
      bias=bias,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
      mode="score_conditional",
      inference=True,
    )
    stage_set = make_stage_set(
      strategy=multi_state_strategy,
      strategy_temperature=multi_state_temperature,
      state_weights=state_weights,
    )

    logits = score_conditional.kernel(model, prng_key, bundle, config, stage_set)

    # Compute score using single-source NLL formula
    nll = _nll_from_logits(logits, sequence, mask)

    return nll, logits, decoding_order

  return cast("ScoreFn", score_sequence)


make_score_sequence = make_score_fn


def score(
  prng_key: PRNGKeyArray,
  model: ModelProtocol,
  structure_coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array | None = None,
  backbone_noise: float | None = None,
  ar_mask: jax.Array | None = None,
  structure_mapping: jax.Array | None = None,
  tie_group_map: jax.Array | None = None,
  state_position_map: jax.Array | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
  state_weights: jax.Array | None = None,
  bias: jax.Array | None = None,
  use_rolling_state: bool = False,
  ligand_coords: jax.Array | None = None,
  ligand_atom_types: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Score a sequence on a structure using the default scoring function.

  This is a convenience wrapper around `make_score_fn`.

  Args:
      prng_key: JAX random key.
      model: A Aminx Equinox model instance.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      sequence: Protein sequence to score.
      backbone_noise: Noise level for backbone coordinates.
      ar_mask: Autoregressive mask for scoring.
      structure_mapping: Mapping between structures.
      tie_group_map: Groups of tied positions.
      state_position_map: Cross-state residue alignment for multi-state PoE fusion,
          shape (S, L); -1 marks an indel. Only meaningful when structure_coordinates
          carries a genuine leading S axis (ndim == 4).
      multi_state_strategy: How to combine multi-state logits.
      multi_state_temperature: Temperature for multi-state combination.
      state_weights: Weights for each state.
      bias: Sequence bias.
      use_rolling_state: Use rolling state scan vs vmap.
      ligand_coords: Ligand coordinates.
      ligand_atom_types: Ligand atom types.
      ligand_mask: Ligand atom mask.

  Returns:
      Tuple of (masked average score, logits, decoding order).

  """
  score_fn = make_score_fn(model)
  return cast(
    "tuple[jax.Array, jax.Array, jax.Array]",
    score_fn(
      prng_key=prng_key,
      sequence=sequence,
      structure_coordinates=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      backbone_noise=backbone_noise,
      ar_mask=ar_mask,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      state_position_map=state_position_map,
      multi_state_strategy=multi_state_strategy,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      bias=bias,
      use_rolling_state=use_rolling_state,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
    ),
  )
