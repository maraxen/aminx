"""A caller's ``decoding_order_fn`` must change what is decoded, or be refused (#166).

``AutoregressiveDecode.decoding_order_fn`` was declared, documented as choosing the order, and
never read; a caller who supplied one to ``runner.sample`` got plausible output, no error and
no effect. The order an AR decode follows is whatever ``bundle.wave`` /
``bundle.conditioning.ar_mask`` encode, so the function has to be applied where the bundle is
built (``with_decoding_order``). These tests pin both halves: the dead field stays gone, and a
supplied function measurably changes the end-to-end result.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.decode.autoregressive import AutoregressiveDecode
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import ConditionalMode, STEMode
from aminx.tiling.strategy import Vmap

_MAX_LENGTH = 128
_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"


def test_autoregressive_decode_has_no_dead_order_field() -> None:
  """The order lives in the bundle; a field here would be a knob wired to nothing."""
  names = {f.name for f in dataclasses.fields(AutoregressiveDecode)}
  assert "decoding_order_fn" not in names


def test_ste_decode_receives_the_callers_order_function() -> None:
  """STEDecode consumes the function; the factory used to hand it only to the inner mode."""

  def custom(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ANN001, ANN202
    return jnp.arange(num_residues, dtype=jnp.int32), key

  decode = make_decode_fn(
    MagicMock(), STEMode(inner_mode=ConditionalMode(), iterations=1), Vmap(), custom,
  )
  assert decode.decoding_order_fn is custom


# --- end-to-end, real checkpoint ---------------------------------------------------------------


def _identity_order(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ANN001, ANN202
  return jnp.arange(num_residues, dtype=jnp.int32), key


def _reversed_order(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ANN001, ANN202
  return jnp.arange(num_residues, dtype=jnp.int32)[::-1], key


def _evens_then_odds_order(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ANN001, ANN202
  # Interleaved, so it reorders the REAL residues. A plain rotation is a trap here: the
  # structure (76 residues) is padded to max_length=128, and rotating by k leaves residues
  # 0..75 in the same relative order, i.e. the same decode as the identity order.
  order = jnp.concatenate(
    [jnp.arange(0, num_residues, 2, dtype=jnp.int32), jnp.arange(1, num_residues, 2, dtype=jnp.int32)],
  )
  return order, key


def _sample(decoding_order_fn=None) -> np.ndarray:  # noqa: ANN001
  from aminx.host.runner import sample

  kwargs = {} if decoding_order_fn is None else {"decoding_order_fn": decoding_order_fn}
  result = sample(
    inputs=[_STRUCTURE],
    checkpoint_id=_CHECKPOINT,
    num_samples=1,
    temperature=0.5,
    random_seed=0,
    max_length=_MAX_LENGTH,
    **kwargs,
  )
  return np.asarray(result["sequences"]).reshape(1, -1)[0, :_MAX_LENGTH]


@pytest.mark.slow
@pytest.mark.requires_weights
def test_supplied_order_function_changes_the_sampled_sequence() -> None:
  """Differential: identical seed, three deliberate orders, three different sequences.

  All three arms were bit-identical while the knob was inert, because the spec was otherwise
  the same and the PRNG stream seeded. The control arm (same function twice) must agree, so a
  difference cannot be blamed on run-to-run nondeterminism.
  """
  calls: list[int] = []

  def counting_reversed(key, num_residues, tie_group_map=None, num_groups=None):  # noqa: ANN001, ANN202
    calls.append(num_residues)
    return _reversed_order(key, num_residues, tie_group_map, num_groups)

  identity = _sample(_identity_order)
  reversed_ = _sample(counting_reversed)
  interleaved = _sample(_evens_then_odds_order)
  reversed_again = _sample(_reversed_order)

  assert calls, "the supplied decoding_order_fn was never called -- the knob is inert"
  assert np.array_equal(reversed_, reversed_again), "control: same function + seed must agree"
  assert not np.array_equal(identity, reversed_)
  assert not np.array_equal(identity, interleaved)
  assert not np.array_equal(reversed_, interleaved)
