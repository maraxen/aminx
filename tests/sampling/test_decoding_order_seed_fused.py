"""End to end: ``decoding_order_seed`` removes the decoding-order variable from a one-free-probe comparison.

Only one position is free; every other position is fixed (teacher-forced). One might expect the free position's
logits to be the same whatever order the fixed positions are decoded in. They are not: in an autoregressive MPNN each
neighbour's hidden state is conditioned on its own predecessors only, so the logits at the probe depend on the order of
the other positions even though all of them are known. Two rows with different sample keys (the key folds in the row
hash, so every distinct row has its own) therefore decode in different orders and their probe logits differ for a reason
that has nothing to do with the operator under test.

With ``decoding_order_seed`` the order is drawn from the seed alone. These tests use a synthetic random-weight model
(no checkpoint) and pin, in order: the order dependence exists without a seed (a control, so the other tests can fail),
the probe logits are bit-identical across samples under a seed, and a fused row of ``k`` identical states
(``weights = 1``, product of experts) has exactly ``k`` times the single-state probe logits under a shared seed, where
under unshared keys it does not.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.sampling.multistate_poe import sample_states_fused

LENGTH = 16
PROBE = 7
WEIGHT_SCALE = 4.0


def _model():
  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(3),
  )
  # Random-init weights are too small for the decoder's neighbour coupling to show; scale them so the
  # synthetic model has trained-like sensitivity to what its neighbours decoded.
  scaled = jax.tree_util.tree_map(
    lambda x: x * WEIGHT_SCALE if eqx.is_inexact_array(x) else x, model,
  )
  return eqx.tree_inference(scaled, value=True)


def _bundle(num_states: int):
  """``num_states`` byte-identical copies of one random backbone, every position fixed but PROBE."""
  rng = np.random.default_rng(5)
  coordinates = jnp.array(rng.normal(size=(LENGTH, 4, 3)).astype(np.float32))
  stack = lambda x: jnp.stack([x] * num_states, axis=0)  # noqa: E731
  fixed_mask = jnp.ones((LENGTH,), dtype=jnp.float32).at[PROBE].set(0.0)
  fixed_tokens = jnp.asarray(rng.integers(0, 20, size=(LENGTH,)), dtype=jnp.int32)
  bundle, config = build_inference_bundle(
    coords=stack(coordinates),
    mask=stack(jnp.ones((LENGTH,), dtype=jnp.float32)),
    residue_index=stack(jnp.arange(LENGTH, dtype=jnp.int32)),
    chain_index=stack(jnp.zeros((LENGTH,), dtype=jnp.int32)),
    state_weights=jnp.ones(num_states),
    fixed_mask=fixed_mask,
    fixed_tokens=fixed_tokens,
    sequence=None,
    mode="sample_ar",
  )
  return bundle, config


def _probe_logits(num_states: int, key, n_samples: int, order_seed: int | None):
  bundle, config = _bundle(num_states)
  stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(num_states))
  _, logits = sample_states_fused(
    _model(), bundle, config, stage_set, key, n_samples, decoding_order_seed=order_seed,
  )
  centered = np.asarray(logits[:, PROBE, :], dtype=np.float64)
  return centered - centered.mean(axis=-1, keepdims=True)


def _rel(a, b) -> float:
  return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def test_probe_logits_depend_on_decoding_order_without_a_seed():
  """Control: with every other position fixed the probe still varies with the (per-sample) order."""
  v = _probe_logits(1, jax.random.PRNGKey(0), 8, None)
  assert max(_rel(v[s], v[0]) for s in range(1, 8)) > 1e-4


def test_probe_logits_are_identical_across_samples_under_a_seed():
  v = _probe_logits(1, jax.random.PRNGKey(0), 8, 3)
  assert np.array_equal(v, np.broadcast_to(v[0], v.shape))


def test_fused_k_copies_equal_k_times_single_under_a_shared_seed():
  k = 3
  # Different sample keys on purpose: with the SAME key the single and fused rows draw the same order
  # whether or not a seed is set, so only the seed can pair them here.
  single = _probe_logits(1, jax.random.PRNGKey(0), 1, 3)[0]
  fused = _probe_logits(k, jax.random.PRNGKey(1), 1, 3)[0]
  assert _rel(fused / k, single) < 1e-4


def test_unshared_keys_do_not_pair_the_rows():
  """Control: the same comparison with different keys (as two distinct rows get) is off by an order effect."""
  k = 3
  single = _probe_logits(1, jax.random.PRNGKey(0), 1, None)[0]
  fused = _probe_logits(k, jax.random.PRNGKey(1), 1, None)[0]
  assert _rel(fused / k, single) > 1e-3
