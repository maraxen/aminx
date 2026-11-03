"""Regression tests for the two defects that made ``runner.score`` score the wrong thing.

Both defects were live on the default production path, both were silent (right shape, right
dtype, no error, plausible numbers), and neither was covered by any existing test. The
existing suite passed throughout because it exercises the *averaged* branch, which passes
``ar_mask=None`` and therefore already received ``build_inference_bundle``'s
``score_conditional`` default.

DEFECT 1 -- the AR mask included the diagonal. ``scoring/score.py`` built its mask with
``generate_ar_mask(decoding_order)``, whose untied branch was ``(row >= col)`` -- non-strict --
so ``ar_mask[i, i] == 1`` under every permutation. Since a residue is always among its own
KNN neighbours, the decoder gated that residue's TRUE one-hot into its own node update.
Measured on 1LVB chain A: +0.036243 nats (paired over 8 seeds, sd 0.002482, t = 41.3),
always in the over-confident direction, i.e. roughly 2.3x the effect sizes the score is used
to resolve. It also made the score key-dependent at ``backbone_noise=0``, because a fresh
decoding order meant a fresh mask.

DEFECT 2 -- the vocabulary was 20, not 21. ``_nll_from_logits`` sliced ``[..., :20]`` on both
factors, justified in-comment as excluding padding. Padding was never what it excluded: the
residue ``mask`` is already 0 at padded positions, which removes them from numerator and
denominator alike. The slice's one live effect was to charge exactly 0 nats for a *real* ``X``
residue while still counting it in the denominator. 1LVB carries six genuine ``X``
(selenomethionines), so this was not hypothetical.

The slice had a second, unrelated consequence: applied to a token-INDEX array it sliced the
only axis, yielding the first 20 *positions*, which then broadcast against ``(L, 20)``
log-probs whenever ``L > 20`` and produced a silently meaningless number.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.scoring.score import _nll_from_logits
from aminx.utils.autoregression import full_context_ar_mask, generate_ar_mask

VOCAB = 21
X_TOKEN = 20


# --------------------------------------------------------------------------------------
# DEFECT 1 -- mask self-exclusion
# --------------------------------------------------------------------------------------


def test_full_context_ar_mask_has_zero_diagonal() -> None:
  """The default the score path now uses must never let a position see itself."""
  mask = np.asarray(full_context_ar_mask(9))
  assert np.all(np.diag(mask) == 0), "a position must not see its own token"
  off_diagonal = mask[~np.eye(9, dtype=bool)]
  assert np.all(off_diagonal == 1), "every position must see every OTHER position"


def test_generate_ar_mask_is_self_excluding_under_every_permutation() -> None:
  """INVERTED 2026-08-27. This test previously pinned the opposite, and it did its job.

  It used to assert ``generate_ar_mask`` was self-INCLUSIVE, to keep visible the reason
  score.py stopped using it, and instructed that if the behaviour ever changed, every
  teacher-forced caller should be revisited. That change has now happened and the callers
  were revisited, so the assertion is inverted rather than deleted.

  What changed: the "undrawn placeholder" rationale for the set diagonal was false --
  undrawn slots held token 0, which is ALANINE, embedded as a real nonzero vector -- and
  both LigandMPNN (``1 - torch.triu(ones)``) and ColabDesign (``jnp.tri(L, k=-1)``)
  self-exclude. ``generate_ar_mask`` now matches them.

  WHAT DID NOT CHANGE, and is the point of keeping this test here: score.py must still use
  :func:`full_context_ar_mask`. Its reason is now solely NON-CAUSALITY -- scoring a known
  sequence wants full context minus self, not a causal prefix -- and no longer has anything
  to do with the diagonal. Do not "simplify" score.py onto ``generate_ar_mask`` on the
  grounds that the diagonal is fixed; that would silently make scoring order-dependent
  again, which is the defect DEFECT 1 in this file exists to prevent.
  """
  key = jax.random.PRNGKey(0)
  for i in range(5):
    order = jax.random.permutation(jax.random.fold_in(key, i), 8)
    mask = np.asarray(generate_ar_mask(order))
    assert np.all(np.diag(mask) == 0), (
      "generate_ar_mask must be self-excluding; if this ever changes back, revisit every "
      "teacher-forced caller"
    )
    # Non-degeneracy: an all-zero mask satisfies the diagonal check while removing all
    # context, a separate and worse defect class.
    assert mask.sum() > 0


def test_score_path_default_mask_is_order_free_and_self_excluding() -> None:
  """The score path's default must not depend on the decoding order at all.

  Order-dependence is what made the score key-dependent at ``backbone_noise=0``: 100% of the
  observed seed-to-seed sd (0.0177 nats at L=214) came from redrawing the order. Two
  different orders must now yield the same mask.
  """
  length = 12
  key = jax.random.PRNGKey(3)
  order_a = jax.random.permutation(key, length)
  order_b = jax.random.permutation(jax.random.fold_in(key, 1), length)
  assert not np.array_equal(np.asarray(order_a), np.asarray(order_b)), "orders must differ"

  # The default no longer consults the order, so it is identical for both.
  mask_a = np.asarray(full_context_ar_mask(length))
  mask_b = np.asarray(full_context_ar_mask(length))
  assert np.array_equal(mask_a, mask_b)

  # And it still differs from what the old default would have produced -- but as of
  # 2026-08-27 the difference is CAUSALITY, not the diagonal. Both masks are now
  # self-excluding; only `full_context_ar_mask` is order-free and non-causal, which is
  # exactly why the score path must keep using it.
  causal = np.asarray(generate_ar_mask(order_a))
  assert not np.array_equal(causal, mask_a)
  assert np.all(np.diag(causal) == 0), "the causal mask is self-excluding too now"
  assert np.all(np.diag(mask_a) == 0)
  # The causal mask IS order-dependent; the score path's default is not. This is the
  # property that made the score key-dependent at backbone_noise=0.
  causal_b = np.asarray(generate_ar_mask(order_b))
  assert not np.array_equal(causal, causal_b), (
    "generate_ar_mask must remain order-dependent -- if it stopped being so, this test no "
    "longer demonstrates why the score path needs an order-free default"
  )


# --------------------------------------------------------------------------------------
# DEFECT 2 -- 21-token vocabulary
# --------------------------------------------------------------------------------------


def _uniform_logits(length: int) -> jnp.ndarray:
  return jnp.zeros((length, VOCAB))


def test_real_x_residue_is_charged_not_silently_free() -> None:
  """An X at a masked-IN position must cost real NLL.

  Under uniform logits every token costs log(21). The old ``[..., :20]`` slice charged 0 for
  an X, so a sequence of all-X scored 0.0 -- a perfect score for a sequence carrying no
  information at all.
  """
  length = 8
  logits = _uniform_logits(length)
  mask = jnp.ones((length,))

  all_x = jax.nn.one_hot(jnp.full((length,), X_TOKEN), VOCAB)
  nll_all_x = float(_nll_from_logits(logits, all_x, mask))

  expected = float(np.log(VOCAB))
  assert nll_all_x == pytest.approx(expected, abs=1e-5), (
    "an X residue must be charged like any other token; 0.0 here is the old defect"
  )
  assert nll_all_x > 0.5, "a sequence of X must not score as free"


def test_x_residue_shifts_the_mean_relative_to_a_real_residue() -> None:
  """Swapping one position to X must move the score when the model disfavours X.

  Under the old slice this was a no-op by construction: the X column was never read, so an X
  contributed 0 to the numerator while still counting 1 in the denominator -- silently
  diluting the mean toward zero rather than reflecting the model's actual opinion.
  """
  length = 6
  logits = jnp.array(np.tile(np.concatenate([np.zeros(20), [-5.0]]), (length, 1)))
  mask = jnp.ones((length,))

  base = jnp.full((length,), 3)
  with_x = base.at[2].set(X_TOKEN)

  nll_base = float(_nll_from_logits(logits, jax.nn.one_hot(base, VOCAB), mask))
  nll_with_x = float(_nll_from_logits(logits, jax.nn.one_hot(with_x, VOCAB), mask))

  assert nll_with_x > nll_base, "a strongly disfavoured X must raise the NLL, not lower it"


def test_padding_is_excluded_by_the_mask_not_by_the_vocabulary() -> None:
  """Masked-out positions must not reach the mean, whatever token sits there.

  This is the property the deleted slice was *credited* with providing. It is the mask that
  provides it, which is why removing the slice is safe: padded positions are dropped from
  numerator and denominator alike, so their token value is irrelevant.
  """
  real_length, padded_length = 5, 11
  logits = _uniform_logits(padded_length)

  mask = jnp.concatenate([jnp.ones((real_length,)), jnp.zeros((padded_length - real_length,))])
  tokens = jnp.concatenate(
    [jnp.full((real_length,), 3), jnp.full((padded_length - real_length,), X_TOKEN)],
  )
  padded = float(_nll_from_logits(logits, jax.nn.one_hot(tokens, VOCAB), mask))

  unpadded = float(
    _nll_from_logits(
      _uniform_logits(real_length),
      jax.nn.one_hot(jnp.full((real_length,), 3), VOCAB),
      jnp.ones((real_length,)),
    ),
  )
  assert padded == pytest.approx(unpadded, abs=1e-6), (
    "padding must not shift the masked mean"
  )

  # And the padded token's identity must not matter at all.
  other_pad = tokens.at[real_length:].set(7)
  assert float(_nll_from_logits(logits, jax.nn.one_hot(other_pad, VOCAB), mask)) == pytest.approx(
    padded, abs=1e-6,
  )


def test_token_indices_and_one_hot_agree() -> None:
  """Index input must give the same answer as one-hot input.

  Regression for the silent broadcast: under ``[..., :20]`` a token array of length L > 20
  was sliced to its first 20 POSITIONS and broadcast against ``(L, 20)`` log-probs, giving a
  meaningless number with no error (measured on a synthetic L=76 case: 530.7 against a true
  3.40). L=76 is used here because the failure needed L > 20 -- at L <= 20 it raised instead,
  which is why a short-sequence test would not have caught it.
  """
  length = 76
  rng = np.random.default_rng(0)
  logits = jnp.array(rng.normal(size=(length, VOCAB)))
  tokens = jnp.array(rng.integers(0, 20, size=(length,)), dtype=jnp.int32)
  mask = jnp.ones((length,))

  from_indices = float(_nll_from_logits(logits, tokens, mask))
  from_one_hot = float(_nll_from_logits(logits, jax.nn.one_hot(tokens, VOCAB), mask))

  assert from_indices == pytest.approx(from_one_hot, rel=1e-6)
  assert 0.0 < from_indices < 20.0, "a sane per-residue NLL, not a broadcast artefact"


def test_gradient_with_respect_to_the_x_column_is_not_identically_zero() -> None:
  """The X column must carry gradient, or mutation-effect estimates are blind to it.

  This pins the ``utils/reverse_jac.py`` half of DEFECT 2, which was strictly worse than the
  scoring half. ``make_reverse_jacobian_score_fn`` inlined its own copy of the NLL that
  sliced ``oh_2d[..., :20]``; differentiating a quantity the X column was sliced out of makes
  ``d(score)/d(one_hot[i, 20])`` **identically zero**, so a single-backward-pass mutation
  scan could not see X as a source or a destination at all -- not mis-scaled, structurally
  absent.

  That path is live (``host/runner.py`` builds its reverse-mode Jacobian feature from it) and
  had NO test coverage whatsoever, which is how the duplicated slice survived the fix to its
  twin. ``reverse_jac`` now delegates to ``_nll_from_logits``, so exercising the shared
  function here covers it.
  """
  length = 12
  rng = np.random.default_rng(0)
  logits = jnp.array(rng.normal(size=(length, VOCAB)))
  mask = jnp.ones((length,))
  one_hot = jax.nn.one_hot(jnp.array(rng.integers(0, 20, size=(length,)), dtype=jnp.int32), VOCAB)

  gradient = np.asarray(jax.grad(lambda oh: _nll_from_logits(logits, oh, mask))(one_hot))

  x_column = float(np.abs(gradient[:, X_TOKEN]).max())
  standard_columns = float(np.abs(gradient[:, :X_TOKEN]).max())
  assert x_column > 0.0, "d(score)/d(X) is identically zero -- the [..., :20] slice is back"
  assert x_column == pytest.approx(standard_columns, rel=0.9), (
    "the X column should carry gradient of a comparable order to the other tokens, "
    f"got {x_column:.4f} against {standard_columns:.4f}"
  )


def test_correlation_cannot_see_a_scale_error_but_the_nll_bound_can() -> None:
  """Justify the parity gate's tolerances by showing the failure correlation admits.

  ``tests/parity/test_full_model_parity.py`` gated every post-featurization stage on
  ``pearson_correlation >= 0.95`` alone. Correlation is affine-invariant, so the realistic
  failure mode of a reimplementation -- a missing scale factor, a dropped normalization, a
  temperature applied at the wrong point -- is exactly what it cannot detect.

  This is a self-check on the gate, not on the model: it asserts that a perturbation which
  SAILS through the correlation bound is caught by the NLL bound. Without it, the new
  tolerances would be numbers nobody had shown could bite.

  Needs no weights and no reference checkout, so it runs in the default suite -- unlike the
  parity tests it is reasoning about, which ``addopts`` deselects.
  """
  # Local: importing the parity module at collection time would drag in the heavy
  # reference/torch machinery for a test that needs neither.
  from scipy.stats import pearsonr  # noqa: PLC0415

  from tests.parity.test_full_model_parity import (  # noqa: PLC0415
    LOG_PROB_MAX_ABS_DEVIATION,
    NLL_ABS_TOLERANCE_NATS,
  )

  length = 64
  rng = np.random.default_rng(1)
  logits = rng.normal(size=(length, VOCAB)) * 2.0
  tokens = rng.integers(0, 20, size=(length,))

  def log_softmax(values: np.ndarray) -> np.ndarray:
    peak = values.max(-1, keepdims=True)
    return values - (peak + np.log(np.exp(values - peak).sum(-1, keepdims=True)))

  def nll(values: np.ndarray) -> float:
    return float(-log_softmax(values)[np.arange(length), tokens].mean())

  baseline = log_softmax(logits)
  scaled = log_softmax(logits * 2.0)

  correlation = float(pearsonr(baseline.ravel(), scaled.ravel())[0])
  assert correlation >= 0.95, (
    "the whole point: a doubled-logit implementation still passes a correlation gate"
  )

  assert float(np.abs(baseline - scaled).max()) > LOG_PROB_MAX_ABS_DEVIATION
  assert abs(nll(logits * 2.0) - nll(logits)) > NLL_ABS_TOLERANCE_NATS, (
    "the NLL bound must reject what correlation waved through"
  )


def test_multi_state_logits_keep_one_hot_input_intact() -> None:
  """A rank mismatch between (S, L, 21) logits and (L, 21) one-hot is legitimate.

  Discriminating on ``ndim`` rather than the vocab axis misreads that as index input and
  one-hots it a second time into ``(L, 21, 21)``. That broke five tests in
  ``test_averaged_parity.py`` when first written, so it is pinned here.
  """
  states, length = 3, 9
  logits = jnp.zeros((states, length, VOCAB))
  one_hot = jax.nn.one_hot(jnp.full((length,), 4), VOCAB)
  mask = jnp.ones((length,))

  value = _nll_from_logits(logits, one_hot, mask)
  assert np.isfinite(np.asarray(value)).all()
  assert float(np.asarray(value).ravel()[0]) == pytest.approx(float(np.log(VOCAB)), abs=1e-5)
