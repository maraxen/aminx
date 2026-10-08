# ruff: noqa: S101
"""Spec §6.5b divergence register: LASEr ``temperature == 0`` is floored, not NaN.

UPSTREAM DISAGREES WITH ITSELF HERE, which is why aminx has to pick one. The
batch CLI declares ``--sequence_temp`` as ``type=float``, and
``run_batch_inference.py:245,370`` maps ``0.0`` to ``None``, i.e. argmax. The
single-input and tied CLIs take ``--temp`` as a STRING, and ``'0'`` is truthy,
so ``run_inference.py:777,796`` / ``run_inference_tied.py:880`` pass a real
``0.0`` straight through to ``softmax(logits / 0.0)`` -- a division by zero
that yields NaN and silently poisons the whole distribution.

aminx follows the batch rule for ``sample`` (``0.0`` -> argmax, §2.3) and the
``utils/model.py:617-618`` rule for tied (``None`` and ``0.0`` alike -> 1e-6).
**The NaN path is never reproduced**, and that is a deliberate divergence
rather than an oversight, so it gets a test that says so.

This file covers the tied half, ``model/laser/tied._tied_temperature``, which
had no test at all despite being the thing standing between a user typing
``--temp 0`` and a tensor of NaN.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aminx.model.laser.tied import _tied_temperature, mix_sequence_probabilities

#: ``utils/model.py:617-618``. Not a tolerance -- the exact constant upstream uses.
_FLOOR = 1e-6


def test_divergence_laser_temperature_zero() -> None:
  """``None`` and ``0.0`` both floor to 1e-6; everything else passes through."""
  assert _tied_temperature(None) == _FLOOR
  assert _tied_temperature(0.0) == _FLOOR
  # -0.0 == 0.0 in IEEE 754, so it takes the same branch. Worth pinning: a
  # rewrite to `if not value` would keep this behaviour, but one to
  # `if value is None or value < 1e-6` would NOT floor a negative temperature,
  # and negative is strictly worse than zero (it inverts the distribution).
  assert _tied_temperature(-0.0) == _FLOOR

  # Pass-through, so the function is not simply a constant. Without this the
  # three assertions above are satisfied by `return 1e-6`.
  assert _tied_temperature(0.5) == 0.5
  assert _tied_temperature(1.0) == 1.0


def test_tied_temperature_floor_is_what_prevents_the_nan() -> None:
  """CONTROL: the raw 0.0 upstream passes really does produce NaN here.

  Without this, ``test_divergence_laser_temperature_zero`` only says the
  function returns a particular number -- it does not establish that the number
  matters. Feeding the unfloored value through the same mixer upstream uses
  must poison the distribution, and the floored one must not.
  """
  logits_1 = jnp.asarray([2.0, 1.0, 0.0, -1.0])
  logits_2 = jnp.asarray([0.0, 1.0, 2.0, -1.0])

  poisoned = mix_sequence_probabilities(
    logits_1, logits_2, lambda_=0.5, temperature=0.0, on_logits=False,
  )
  assert not bool(jnp.all(jnp.isfinite(poisoned))), (
    "softmax(logits / 0.0) is finite, so this control no longer demonstrates "
    "what the floor prevents -- re-read the divergence row before trusting it."
  )

  floored = mix_sequence_probabilities(
    logits_1, logits_2, lambda_=0.5,
    temperature=_tied_temperature(0.0), on_logits=False,
  )
  assert bool(jnp.all(jnp.isfinite(floored)))
  # At 1e-6 the softmax is effectively argmax, which is the behaviour the
  # floor is standing in for: all mass on each input's own maximum, mixed 50/50.
  np.testing.assert_allclose(
    np.asarray(floored, dtype=np.float64),
    np.asarray([0.5, 0.0, 0.5, 0.0], dtype=np.float64),
    rtol=0, atol=1e-9,
  )
