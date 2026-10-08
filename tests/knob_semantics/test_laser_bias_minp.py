# ruff: noqa: S101
"""Spec §5.3.1 steps 1 and 6: bias, then min-p, then ``/T`` -- in that order.

``joint_decode.py:309`` adds the bias row into ``logits``; ``:330`` warps the
BIASED logits with min-p; ``:335`` divides the WARPED tensor by the
temperature. Each of the two reorderings the spec calls out as negative
controls changes the sampled distribution, and neither raises:

  - bias AFTER min-p. The threshold is then computed on unbiased
    probabilities, so a token the bias was meant to rescue is already ``-inf``
    when the bias arrives -- and ``-inf + bias`` is still ``-inf``. The bias is
    silently discarded for exactly the tokens it was set to promote, which is
    the only case anyone sets a bias for.
  - bias AFTER ``/T``. The bias then escapes the temperature scaling, so its
    effective strength is multiplied by T relative to the specified pipeline.

Both are tested against an ANALYTIC distribution computed from the surviving
token set by hand, not by calling the function under test a second time.

These two names (``test_knob_semantics_laser_bias_minp``,
``test_knob_semantics_laser_stored_logits_minp``) are the §5.3.1 rows that did
not exist. The halves of those rows that compare against oracle
``sequence_logits`` need an upstream fixture and live in ``tests/port/``; the
ordering and the stored-tensor identity are analytic and are what this file
pins.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aminx.model.laser.joint_decode import minp_warp_logits

_V = 21
_MIN_P = 0.05
_T = 0.3

# Token 0 dominates; token 1 sits just BELOW the min-p line without the bias and
# just ABOVE it with. Everything else is far enough down to be removed either
# way, so the fixture isolates the one token whose fate the ordering decides.
_RAW = np.full((_V,), -20.0)
_RAW[0] = 0.0
_RAW[1] = -4.0

_BIAS = np.zeros((_V,))
_BIAS[1] = 2.0


def _logits() -> jnp.ndarray:
  return jnp.asarray(_RAW + _BIAS)


def _analytic_probs(survivors: tuple[int, ...], logits: np.ndarray, temperature: float) -> np.ndarray:
  """Softmax over a KNOWN survivor set, computed without min-p.

  This is the independent half of the test: it never calls
  ``minp_warp_logits``, so agreeing with it is evidence rather than tautology.
  """
  out = np.zeros((_V,))
  kept = np.asarray([logits[index] for index in survivors]) / temperature
  kept = kept - kept.max()
  weights = np.exp(kept)
  weights = weights / weights.sum()
  for slot, index in enumerate(survivors):
    out[index] = weights[slot]
  return out


def test_knob_semantics_laser_bias_minp() -> None:
  """The bias is inside the min-p threshold, and both reorderings differ."""
  # Without the bias, token 1 is below 0.05 x the top probability.
  unbiased = np.asarray(minp_warp_logits(jnp.asarray(_RAW), _MIN_P))
  assert np.isneginf(unbiased[1]), (
    f"fixture broken: token 1 must be removed WITHOUT the bias, got {unbiased[1]}"
  )

  # With it, token 1 survives. That is the whole knob.
  biased = np.asarray(minp_warp_logits(_logits(), _MIN_P))
  assert np.isfinite(biased[1]), f"the bias must carry token 1 over the line: {biased[1]}"
  assert np.isfinite(biased[0])
  assert np.isneginf(biased[2:]).all()

  # Against an analytic distribution over the survivor set {0, 1}.
  probs = np.asarray(jnp.exp(jnp.asarray(biased) / _T - jnp.max(jnp.asarray(biased) / _T)))
  probs = probs / probs.sum()
  # The decode path runs in float32, so the band is float32 round-off and not a
  # judgement call: MEASURED worst relative difference 1.31e-07 against an f64
  # analytic reference, which is ~1.1 ulp (float32 eps = 1.19e-07). A tighter
  # band is unreachable in the production dtype; a looser one would stop
  # distinguishing round-off from a reordered pipeline, whose controls below
  # move the distribution by whole orders of magnitude.
  np.testing.assert_allclose(
    probs, _analytic_probs((0, 1), _RAW + _BIAS, _T), rtol=1e-6, atol=0,
  )

  # NEGATIVE CONTROL 1 -- bias after min-p. -inf + 2.0 is still -inf, so the
  # bias is discarded for the one token it was set to promote.
  after_minp = unbiased + _BIAS
  assert np.isneginf(after_minp[1]), after_minp[1]
  assert not np.array_equal(np.isneginf(after_minp), np.isneginf(biased))

  # NEGATIVE CONTROL 2 -- bias after /T. The survivor set is the unbiased one
  # AND the bias is no longer divided by T, so it lands at a different size.
  after_temp = unbiased / _T + _BIAS
  assert np.isneginf(after_temp[1])
  # The specified pipeline scales the bias by 1/T; this one does not.
  np.testing.assert_allclose(np.asarray(biased)[1] / _T, (_RAW[1] + _BIAS[1]) / _T)
  assert (_RAW[1] + _BIAS[1]) / _T != _RAW[1] / _T + _BIAS[1]


def test_knob_semantics_laser_stored_logits_minp() -> None:
  """``stored`` is post-min-p and UNTEMPERED, and ``min_p=0`` adds no ``-inf``."""
  logits = _logits()
  stored = np.asarray(minp_warp_logits(logits, _MIN_P))

  # Post-min-p: the -inf pattern is present in the STORED tensor, not only in
  # the tempered copy used for the draw.
  assert np.isneginf(stored[2:]).all()

  # Untempered: surviving entries are the biased logits themselves, NOT divided
  # by T. Storing the tempered tensor would make seq_log_prob a function of the
  # temperature, which is the defect this pins.
  np.testing.assert_allclose(stored[0], _RAW[0] + _BIAS[0], rtol=0, atol=0)
  np.testing.assert_allclose(stored[1], _RAW[1] + _BIAS[1], rtol=0, atol=0)
  assert stored[1] != (_RAW[1] + _BIAS[1]) / _T, "stored must not be tempered"

  # CONTROL: min_p = 0 returns the logits unchanged, with NO -inf fill. The
  # function short-circuits (joint_decode.py:159), so a rewrite that always
  # thresholded would silently truncate the vocabulary at the default setting.
  unwarped = np.asarray(minp_warp_logits(logits, 0.0))
  np.testing.assert_array_equal(unwarped, np.asarray(logits))
  assert np.isfinite(unwarped).all()


def test_minp_top_token_guard_only_matters_above_one() -> None:
  """The ``at[0].set(False)`` guard is dead for ``min_p < 1`` and load-bearing above it.

  Written after measuring, because the obvious reading of ``:165`` is wrong.
  The threshold is ``min_p * top``, so for any ``min_p < 1`` it is strictly
  BELOW the top probability and the best token can never be removed -- the
  guard is a no-op on every in-range setting.

  MEASURED: a flat 21-token row at ``min_p=0.99`` keeps all 21, not one. Every
  token ties with the top, and ``p < 0.99*p`` is false for all of them.

  The guard earns its place only at ``min_p >= 1``, where the threshold reaches
  or passes the top probability. Without it that returns an all-``-inf`` row and
  the softmax downstream is NaN -- the same class of failure as the
  temperature-zero divergence, and the reason this is tested rather than
  deleted as unreachable.
  """
  flat = jnp.zeros((_V,))
  # Below 1: nothing is removed at all, flat or not.
  kept = np.asarray(minp_warp_logits(flat, 0.99))
  assert int(np.isfinite(kept).sum()) == _V, kept

  # At and above 1 the guard is the only thing standing between the caller and
  # a NaN softmax. A peaked row makes the threshold exceed every probability.
  peaked = jnp.asarray(np.concatenate([[10.0], np.zeros(_V - 1)]))
  for min_p in (1.0, 1.5):
    warped = np.asarray(minp_warp_logits(peaked, min_p))
    assert np.isfinite(warped).any(), f"min-p {min_p} removed every token"
    assert int(np.argmax(warped)) == 0, warped

  # And the survivor at 1.5 is the argmax ALONE -- the guard let exactly one
  # through, rather than the threshold quietly failing to apply.
  assert int(np.isfinite(np.asarray(minp_warp_logits(peaked, 1.5))).sum()) == 1
