"""MBR post-hoc consensus reranking: independent per-state scoring + rerank.

Post-hoc batch utility for already-sampled sequences (e.g. a completed production run's
output), NOT a live-decode StageSet stage -- see
../../.praxia/docs/specs/260709_mbr-consensus-reranking-composition.md §4 for why this
does not populate StageSet.axis_boundaries.

States go through a genuine jax.vmap (reusing score_conditional.encode()'s existing
_VmapEncode and decode_states_unfused's state_iterator, legitimate because the caller's
bundle is pre-padded to a uniform state shape -- see mbr_rerank's docstring). Candidates go
through xtrax's BatchPlanner-driven Vmap/SafeMap dispatch on N_CANDIDATES, the exact same
composable_jax idiom make_batched_conditional_logits_split_fn's batched_decode_fn already
uses (sampling/conditional_logits.py:412-437) -- no hand-rolled jax.vmap/lax.map chunking
here, per the using-xtrax skill's "never hand-roll" rule.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from xtrax.tiling import VmapIterator

from aminx.inference.decode.unfused import decode_states_unfused
from aminx.inference.score_conditional import encode as score_conditional_encode
from aminx.tiling.axes import N_CANDIDATES
from aminx.tiling.dispatch import make_axis_dispatch
from aminx.tiling.planner import plan_axis_strategy
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig


def _nll_from_logits(
  logits: Float[Array, "... L V"],
  sequences_idx: Int[Array, "... L"],
) -> Float[Array, "..."]:
  """Mean per-position negative log-likelihood, axis-generic via broadcasting.

  Deliberately not compute_pseudo_perplexity (host/logit_aggregation.py:14-53) -- that
  function hardcodes a rigid 4-leading-dim shape convention ([batch, samples, noise, temp,
  seq_len, 21]) inherited from a different pipeline; this design's logits have 2 leading
  dims (S, C), not 4. Caller is responsible for aligning ``sequences_idx``'s leading dims
  against ``logits``'s leading dims (minus the vocab axis) via standard broadcasting --
  e.g. ``sequences_idx[:, None, :]`` to score one sequence per candidate against every
  state.

  Parameters
  ----------
  logits : ndarray
      Shape (..., L, V).
  sequences_idx : ndarray
      Integer amino-acid indices, shape (..., L), broadcastable against logits' leading
      dims.

  Returns
  -------
  ndarray
      Mean NLL over L. Shape matches the broadcast of logits' and sequences_idx's leading
      dims (logits' shape minus the last two axes).

  """
  # An out-of-range index (notably UNDRAWN_TOKEN = -1, which `SampleResult.sequence` may
  # carry at positions a partial wave schedule never covered) one-hots to the ZERO vector,
  # so its `-sum(one_hot * log_probs)` is exactly 0.0. That is not a neutral value here: it
  # is the best possible per-position NLL, so an undecided position would score as a free
  # perfect match and bias `average_cross_state_scores` TOWARD the candidate carrying it.
  # Silent, and in the dangerous direction -- hence a hard failure rather than a clamp.
  #
  # Not reachable from any in-library schedule constructor today (all produce full-length
  # orders), so this guards a future caller rather than a live defect.
  n_classes = logits.shape[-1]
  if not isinstance(sequences_idx, jax.core.Tracer):
    idx = jnp.asarray(sequences_idx)
    if bool(jnp.any((idx < 0) | (idx >= n_classes))):
      msg = (
        f"sequences_idx contains out-of-range token indices (valid range [0, {n_classes})). "
        "A -1 here is almost certainly UNDRAWN_TOKEN from a position the wave schedule "
        "never covered; one-hot would silently score it 0.0 NLL -- a free perfect match. "
        "Trim or fill undecided positions before scoring."
      )
      raise ValueError(msg)

  one_hot = jax.nn.one_hot(sequences_idx, n_classes)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  nll_per_position = -jnp.sum(one_hot * log_probs, axis=-1)
  return jnp.mean(nll_per_position, axis=-1)


def average_cross_state_scores(
  per_state_scores: Float[Array, "S C"],
) -> Float[Array, "C"]:
  """Average per-state scores across the k reference states. Stacked[S] -> Out[C].

  Elementwise mean over the state axis (axis 0). At k=1 this is a true no-op: the
  single state's own per-candidate scores pass through unchanged.

  Parameters
  ----------
  per_state_scores : ndarray
      Per-state, per-candidate NLL scores. Shape (S, C).

  Returns
  -------
  ndarray
      Mean NLL per candidate, across states. Shape (C,).

  """
  return jnp.mean(per_state_scores, axis=0)


def select_mbr_candidates(
  mean_scores: Float[Array, "C"],
  candidate_sequences: Int[Array, "C L"],
  top_k: int = 1,
) -> tuple[Int[Array, " top_k"], Int[Array, "top_k L"]]:
  """Select the top_k LOWEST-scoring (lowest-NLL) candidates.

  Lower NLL = better fit, per score_conditional's scoring convention (aminx.score() is
  documented as lower-is-better; see spec §1/§2). jnp.argsort is ascending by default --
  do not negate or reverse this: an accidental argmax/descending-sort here would silently
  select the WORST candidates while still returning a runnable, plausible-looking result.

  Parameters
  ----------
  mean_scores : ndarray
      Mean cross-state NLL per candidate. Shape (C,).
  candidate_sequences : ndarray
      Candidate sequences to select from. Shape (C, L).
  top_k : int, default 1
      Number of lowest-NLL candidates to return.

  Returns
  -------
  tuple
      (selected_indices, selected_sequences), both length top_k, best-first.

  """
  selected_indices = jnp.argsort(mean_scores)[:top_k]
  return selected_indices, candidate_sequences[selected_indices]


def mbr_rerank(
  model: Any,
  canonical_bundle: InferenceBundle,
  candidate_sequences_idx: Int[Array, "C L"],
  prng_key: PRNGKeyArray,
  config: InferenceConfig | None = None,
  top_k: int = 1,
  candidate_batch_size: int | None = None,
) -> tuple[Int[Array, " top_k"], Int[Array, "top_k L"]]:
  """MBR post-hoc consensus reranking of already-sampled candidate sequences.

  Scores each candidate independently against each of the k reference states (via
  decode_states_unfused -- genuine per-state logits, no in-loop fusion), averages the
  per-state NLL across states, and returns the top_k lowest-mean-NLL candidates.

  PRECONDITION (stated contract, not a runtime check -- see spec §6's first acceptance
  criterion and ../../.praxia/docs/decisions/260709_n-states-heterogeneous-flag-unenforced.md):
  ``canonical_bundle`` must already be padded to a uniform per-state shape (e.g.
  tev_design's build_canonical_bundle.py, N_CANONICAL=214) before this call.
  aminx's own N_STATES.heterogeneous=True registry flag is a conservative declaration for
  the axis in the abstract and is NOT enforced anywhere in aminx's production encode path
  (InferenceBundle/GeometryBundle cannot even represent genuinely ragged per-state data --
  coords/atom_37/masks are single stacked arrays with a uniform leading S axis). This
  function will not detect or reject a caller who somehow bypasses that padding step; it
  will fail (or silently misbehave) inside the internal jax.vmap over states instead.

  Parameters
  ----------
  model : Any
      Aminx model instance.
  canonical_bundle : InferenceBundle
      Pre-padded, uniform-shape multi-state bundle (see precondition above).
  candidate_sequences_idx : ndarray
      Integer amino-acid indices for C already-sampled candidate sequences, shape (C, L).
  prng_key : PRNGKeyArray
      PRNG key (encode + decode are both deterministic/inference-mode here, but a key is
      still threaded through to match score_conditional.encode()'s signature).
  config : InferenceConfig | None
      Inference configuration. Defaults to InferenceConfig() (inference=True) when None.
  top_k : int, default 1
      Number of lowest-NLL candidates to return.
  candidate_batch_size : int | None, default None
      Fixed SafeMap tile size for the candidate axis. None defers to the BatchPlanner's
      memory-budget-driven Vmap/SafeMap choice (same default behavior as
      make_batched_conditional_logits_split_fn's candidate_batch_size).

  Returns
  -------
  tuple
      (selected_indices, selected_sequences) from select_mbr_candidates, best-first.

  """
  if config is None:
    config = InferenceConfig()

  key_encode, key_decode = jax.random.split(prng_key)

  # Encode once -- real jax.vmap over S, legitimate given the precondition above.
  # Reused as-is (no new encode code): _VmapEncode via score_conditional.encode().
  encodings = score_conditional_encode(model, key_encode, canonical_bundle, config)

  n_candidates = candidate_sequences_idx.shape[0]
  seq_len = candidate_sequences_idx.shape[1]
  vocab_size = canonical_bundle.conditioning.sequence_oh.shape[-1]
  candidate_sequences_oh = jax.nn.one_hot(candidate_sequences_idx, vocab_size)

  # Candidate axis: xtrax BatchPlanner-driven Vmap/SafeMap dispatch, exactly the pattern
  # batched_decode_fn already uses for N_CANDIDATES (conditional_logits.py:412-437).
  activation_bytes = seq_len * 21 * 4  # (L, 21) float32 logits per candidate, per state
  strategy = plan_axis_strategy(
    N_CANDIDATES,
    n_candidates,
    candidate_batch_size,
    activation_bytes_per_element=activation_bytes,
  )
  candidate_iterator = make_axis_dispatch(strategy, axis=N_CANDIDATES.name)
  state_iterator = VmapIterator()  # states are pre-padded uniform-shape; no reason to SafeMap

  def _decode_one_candidate(seq_oh: Float[Array, "L V"]) -> Float[Array, "S L V"]:
    return decode_states_unfused(
      model=model,
      encodings=encodings,
      sequence_oh=seq_oh,
      ar_mask=canonical_bundle.conditioning.ar_mask,
      key=key_decode,
      config=config,
      state_iterator=state_iterator,
    )

  logits = candidate_iterator(_decode_one_candidate, candidate_sequences_oh)  # (C, S, L, V)

  # Move to (S, C, L, V) to match average_cross_state_scores's stated (S, C) contract.
  logits_state_major = jnp.moveaxis(logits, 0, 1)
  per_state_scores = _nll_from_logits(
    logits_state_major,
    candidate_sequences_idx[None, :, :],  # (1, C, L) broadcasts against (S, C, L, V)
  )  # (S, C)

  mean_scores = average_cross_state_scores(per_state_scores)  # (C,)
  return select_mbr_candidates(mean_scores, candidate_sequences_idx, top_k=top_k)
