"""Reverse-mode Jacobian implementation for efficient mutation scoring.

While forward-mode Jacobian (CatJac) computes sensitivity of *all* outputs
to perturbations, the reverse-mode gradient is much more efficient when we
only care about the gradient of a single scalar (e.g., the pooled log-likelihood
score). This allows computing the sensitivity of the score to all possible mutations
in a single backward pass.
"""

from collections.abc import Callable

import jax
from jaxtyping import Array, Float

from aminx.sampling.conditional_logits import (
  make_encoding_conditional_logits_split_fn,
)
from aminx.scoring.score import _nll_from_logits
from aminx.types.arrays import AutoRegressiveMask
from aminx.types.protocols import ModelProtocol


def make_reverse_jacobian_score_fn(model: ModelProtocol) -> tuple[Callable, Callable]:
  """Create a function to compute reverse-mode gradients of the score.

  The returned function `grad_fn` takes the pre-computed `encoding` and
  a one-hot `sequence` array, and returns a gradient of identical shape
  (L, 21), representing `∂score / ∂one_hot[i, a]`.

  For a given mutation from WT amino acid `w` to mutant `m` at position `i`,
  the first-order approximation of the change in score is given by:
      Δscore ≈ grad[i, m] - grad[i, w]
  """
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  @jax.jit
  def compute_grad_score(
    encoding: tuple,
    one_hot_sequence: Float[Array, "L 21"],
    ar_mask: AutoRegressiveMask,
  ) -> Float[Array, "L 21"]:
    """Compute ∂score/∂one_hot for a given sequence encoding."""
    # Extract the mask from the encoding tuple (4th element)
    mask = encoding[3]

    def score_fn(oh_2d: Float[Array, "L 21"]) -> jax.Array:
      logits = decode_fn(encoding, oh_2d, ar_mask=ar_mask)
      # Delegate to the single NLL definition rather than repeating the formula. This used
      # to inline its own copy that sliced `[..., :20]` on both factors, with the comment
      # "Only typical 20 amino acids contribute to the actual score probability". That is
      # the same defect that was live in `scoring/score.py`, and it is worse here: slicing
      # the one-hot away from the X column makes the GRADIENT with respect to that column
      # identically zero, so a mutation-effect estimate is structurally blind to X rather
      # than merely mis-scaled. Two copies is also how the two definitions drift.
      return _nll_from_logits(logits, oh_2d, mask)

    # Single reverse-mode backward pass
    return jax.grad(score_fn)(one_hot_sequence)

  return encode_fn, compute_grad_score
