"""Parity + tiling tests for the xtrax-tiled categorical Jacobian.

The tiled implementation replaced an unchunked `jax.jacfwd`. The only thing that makes
that swap safe is that it computes the SAME tensor, so the load-bearing test here is
elementwise parity against the reference, at more than one tile size, including tile
sizes that do NOT divide the tangent count (`L * 21`), which is the case xtrax's SafeMap
raises on if the caller does not pad (backlog #4159).

Everything else -- axis ordering, the (i,a,j,b) contract -- is downstream of that parity.
An axis-ordering mistake here would produce a complete, symmetric, plausible tensor that
nothing downstream would flag, which is the same silent-corruption class as the coupling
orientation bug in asr's params loader.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model import Aminx
from aminx.utils.forward_jac import (
  NUM_TOKENS,
  make_categorical_jacobian_fn,
  make_reference_categorical_jacobian_fn,
)

SEQ_LEN = 9


def _fixture(seed: int = 0):
  """Small deterministic model + structure. Kept tiny so the reference path is affordable."""
  key = jax.random.PRNGKey(seed)
  model = Aminx(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=key,
  )
  model = eqx.tree_inference(model, value=True)

  rng = np.random.default_rng(seed)
  coords = jnp.asarray(rng.normal(size=(SEQ_LEN, 4, 3)).astype(np.float32))
  mask = jnp.ones((SEQ_LEN,), dtype=jnp.float32)
  residue_index = jnp.arange(SEQ_LEN, dtype=jnp.int32)
  chain_index = jnp.zeros((SEQ_LEN,), dtype=jnp.int32)
  sequence = jnp.asarray(rng.integers(0, 20, size=(SEQ_LEN,)), dtype=jnp.int32)
  return model, (jax.random.PRNGKey(seed + 1), coords, mask, residue_index, chain_index, sequence)


def test_jacobian_is_not_identically_zero():
  """REGRESSION GUARD for the bug this module shipped with: an all-zero Jacobian.

  `forward_jac` used to pass `ar_mask = zeros((L, L))` while its comment claimed "fully
  conditional / no autoregressive masking, every position sees every other". The
  convention is the opposite: `ar_mask[i, j] == 1` means i SEES j, gating the sequence
  edge features via `mask_bw` (`model/decoder.py:144-147`). With an all-zero mask no
  sequence information entered the decoder at all, so the logits were a function of
  structure alone and d logits / d one_hot was **exactly zero everywhere**.

  The failure was completely silent: correct shape, correct dtype, no error. Every
  structural assertion in this file -- shape, parity against jacfwd, symmetry -- passes
  trivially when both sides are zero. This test is what makes the others mean anything.
  """
  model, args = _fixture()
  jacobian = np.asarray(make_categorical_jacobian_fn(model)(*args))

  assert np.all(np.isfinite(jacobian))
  assert float(np.max(np.abs(jacobian))) > 0.0, (
    "categorical Jacobian is identically zero -- the decoder is receiving no sequence "
    "information (check ar_mask: it must be 1 - I, not zeros)"
  )
  nonzero_fraction = float((jacobian != 0).mean())
  assert nonzero_fraction > 0.1, (
    f"only {nonzero_fraction:.3f} of the Jacobian is non-zero; a nearly-empty tensor "
    "suggests the sequence path is still mostly masked off"
  )


def test_full_context_ar_mask_excludes_self_only():
  """The mask must be `1 - I`: all context except the position's own token."""
  from aminx.utils.forward_jac import _full_context_ar_mask

  mask = np.asarray(_full_context_ar_mask(6))
  assert mask.shape == (6, 6)
  np.testing.assert_array_equal(np.diagonal(mask), np.zeros(6))
  off_diagonal = mask[~np.eye(6, dtype=bool)]
  np.testing.assert_array_equal(off_diagonal, np.ones(30))


@pytest.mark.parametrize("tangent_batch_size", [None, 1, 7, 32, 10_000])
def test_tiled_matches_unchunked_jacfwd(tangent_batch_size):
  """The tiled Jacobian must equal `jax.jacfwd` elementwise, at every tile size.

  `7` and `32` are deliberately chosen NOT to divide `SEQ_LEN * 21 = 189` evenly (189 =
  27*7 does divide by 7 -- so 32 is the indivisible case, and 7 the divisible one), and
  `10_000` exceeds the cardinality so the planner picks Vmap. `1` is the degenerate
  maximally-chunked case.
  """
  model, args = _fixture()
  tiled = make_categorical_jacobian_fn(model, tangent_batch_size=tangent_batch_size)(*args)
  reference = make_reference_categorical_jacobian_fn(model)(*args)

  assert tiled.shape == (SEQ_LEN, NUM_TOKENS, SEQ_LEN, NUM_TOKENS)
  assert tiled.shape == reference.shape
  np.testing.assert_allclose(np.asarray(tiled), np.asarray(reference), rtol=1e-5, atol=1e-5)


def test_indivisible_tile_does_not_silently_truncate():
  """A tile that does not divide `L * 21` must still return every tangent.

  Guards the padding path specifically: a naive implementation that dropped the short
  final chunk would return a correctly-SHAPED tensor whose last rows are zero, which the
  parity test above would catch only because it compares elementwise. This asserts the
  tail directly so the failure mode is named.
  """
  model, args = _fixture()
  n_tangents = SEQ_LEN * NUM_TOKENS
  tile = 32
  assert n_tangents % tile != 0, "fixture no longer exercises the indivisible case"

  tiled = np.asarray(make_categorical_jacobian_fn(model, tangent_batch_size=tile)(*args))
  # The final tangent is (j, b) = (L-1, 20); it maps to J[:, :, L-1, 20].
  tail = tiled[:, :, SEQ_LEN - 1, NUM_TOKENS - 1]
  assert np.any(tail != 0.0), "last tangent is all zeros -- the short final chunk was dropped"


def test_axis_order_is_output_pair_first():
  """`J[i, a, j, b] = d logits[i,a] / d one_hot[j,b]` -- verified against a direct JVP.

  NEGATIVE CONTROL for the reshape/transpose. Computes one basis JVP by hand and checks it
  lands at the right slice; a transposed assembly would put it at `J[j, b, i, a]` instead,
  which has identical shape here (L and 21 both appear twice) and so cannot be caught by
  any shape assertion.
  """
  model, args = _fixture(seed=3)
  prng_key, coords, mask, residue_index, chain_index, sequence = args

  tiled = np.asarray(make_categorical_jacobian_fn(model, tangent_batch_size=16)(*args))

  # Recompute a single column directly: perturb one_hot at (j=2, b=5).
  from aminx.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
  from aminx.utils.forward_jac import _full_context_ar_mask

  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)
  encoding = encode_fn(
    coords, mask, residue_index, chain_index,
    backbone_noise=0.0, prng_key=prng_key, structure_mapping=None,
  )
  one_hot = jax.nn.one_hot(sequence, NUM_TOKENS)
  # Must be the SAME mask the implementation uses. Building `zeros` here (as this test
  # first did) silently makes `direct` all-zero, and the comparison then "fails" for a
  # reason that has nothing to do with axis order.
  ar_mask = _full_context_ar_mask(SEQ_LEN)
  tangent = jnp.zeros_like(one_hot).at[2, 5].set(1.0)
  _, direct = jax.jvp(
    lambda oh: decode_fn(encoding, oh, ar_mask=ar_mask), (one_hot,), (tangent,),
  )

  # direct is d logits[:, :] / d one_hot[2, 5] -> must equal J[:, :, 2, 5].
  np.testing.assert_allclose(tiled[:, :, 2, 5], np.asarray(direct), rtol=1e-5, atol=1e-5)

  # And must NOT equal the transposed slot, or the assertion above proves nothing.
  transposed_slot = tiled[2, 5, :, :]
  assert not np.allclose(transposed_slot, np.asarray(direct), rtol=1e-5, atol=1e-5), (
    "J[i,a,j,b] and J[j,b,i,a] are indistinguishable on this fixture -- the axis-order "
    "check is vacuous"
  )


def test_planner_demotes_to_chunked_map_when_the_axis_is_large():
  """A big tangent axis must not stay on Vmap: that is the whole point of the tiling.

  Exercises `plan_axis_strategy` directly rather than through a real Jacobian, so the
  demotion is asserted without allocating anything.
  """
  from aminx.tiling.axes import N_JACOBIAN_PAIRS
  from aminx.tiling.planner import plan_axis_strategy
  from xtrax.tiling import ChunkedMap, Vmap

  small = plan_axis_strategy(
    N_JACOBIAN_PAIRS, 16, None, activation_bytes_per_element=1024.0,
  )
  assert isinstance(small, Vmap)

  # 242 * 21 tangents at a realistic per-tangent activation footprint (~35 MB) is ~181 GB.
  huge = plan_axis_strategy(
    N_JACOBIAN_PAIRS, 242 * NUM_TOKENS, None, activation_bytes_per_element=35e6,
  )
  assert isinstance(huge, ChunkedMap), (
    "a 242-residue tangent axis at ~35 MB/tangent stayed on Vmap -- the memory budget is "
    "not being consulted, and this is exactly the configuration predicted to OOM"
  )


def test_explicit_tile_overrides_the_planner():
  """`tangent_batch_size` must actually take effect -- it was a dead knob before this."""
  from aminx.tiling.axes import N_JACOBIAN_PAIRS
  from aminx.tiling.planner import plan_axis_strategy
  from xtrax.tiling import ChunkedMap

  strategy = plan_axis_strategy(
    N_JACOBIAN_PAIRS, 16, 4, activation_bytes_per_element=1.0,
  )
  assert isinstance(strategy, ChunkedMap)
  assert strategy.batch_size == 4
