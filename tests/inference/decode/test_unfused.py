"""Tests for decode_states_unfused (spec §6 acceptance criteria, unfused.py).

decode_states_unfused mirrors ConditionalDecode.__call__ up to (not including) its two
fusion calls. The core claim under test: stopping before fusion doesn't change anything
else -- verified by comparing against ConditionalDecode's own fused output at S=1, where
fusion over one element is a mathematical no-op (§6, second acceptance criterion).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.decode.unfused import decode_states_unfused
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from xtrax.tiling import ChunkedMapIterator, VmapIterator
from tests.inference.decode.test_conditional import _build_synthetic_fixture


@pytest.mark.parametrize("iterator_factory", [VmapIterator, ChunkedMapIterator])
def test_decode_states_unfused_matches_conditional_decode_at_s1(iterator_factory) -> None:
  """S=1: fusion is a no-op, so decode_states_unfused must equal ConditionalDecode's output.

  This is the direct, isolating check the spec calls for (§6): it ties the new pre-fusion
  path back to the existing, trusted ConditionalDecode path rather than trusting a
  from-scratch reimplementation on its own.
  """
  iterator = iterator_factory(tile=1) if iterator_factory is ChunkedMapIterator else iterator_factory()

  model, _, _, _, _, sequence_oh, bundle, config = _build_synthetic_fixture(
    num_states=1, seed=42,
  )

  k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  stage_set = make_stage_set(
    strategy="arithmetic_mean",
    state_weights=bundle.conditioning.state_weights,
  )

  fused = ConditionalDecode(model=model, state_iterator=iterator)(
    key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set,
  )  # (L, 21)

  unfused = decode_states_unfused(
    model=model,
    encodings=enc,
    sequence_oh=sequence_oh,
    ar_mask=bundle.conditioning.ar_mask,
    key=k_dec,
    config=config,
    state_iterator=iterator,
  )  # (S=1, L, 21)

  assert unfused.shape == (1, *fused.shape)
  np.testing.assert_allclose(np.asarray(unfused[0]), np.asarray(fused), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("num_states", [1, 4, 8])
@pytest.mark.parametrize("iterator_factory", [VmapIterator, ChunkedMapIterator])
def test_decode_states_unfused_shape_dtype(num_states: int, iterator_factory) -> None:
  """Sanity: valid (S, L, 21) float32 output across state counts and iterator strategies."""
  iterator = iterator_factory(tile=2) if iterator_factory is ChunkedMapIterator else iterator_factory()

  model, _, _, _, _, sequence_oh, bundle, config = _build_synthetic_fixture(
    num_states=num_states, seed=42,
  )
  k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  unfused = decode_states_unfused(
    model=model,
    encodings=enc,
    sequence_oh=sequence_oh,
    ar_mask=bundle.conditioning.ar_mask,
    key=k_dec,
    config=config,
    state_iterator=iterator,
  )

  assert unfused.shape == (num_states, enc.neighbor_indices.shape[1], 21)
  assert unfused.dtype == jnp.float32
