"""``fixed_position_context``: how fixed positions are conditioned on each other when sampling.

In an autoregressive MPNN the logits at a position depend on the order in which the OTHER positions were decoded,
even when all of them are fixed, because each neighbour's hidden state is conditioned on its own predecessors only.
The reference implementations decode the fixed block in a random order, so that order is a nuisance variable. These
tests pin three behaviours: ``random_order`` (default, reference behaviour, unchanged), ``canonical_order`` (fixed
block in position order) and ``mutual_context`` (every fixed position sees every other fixed position).

The end-to-end tests use a synthetic random-weight model (no checkpoint) in a one-free-probe design with NO
``decoding_order_seed`` and different sample keys, as two distinct campaign rows would have, and ask whether a fused
row of ``k`` identical states equals ``k`` times the single-state row. Only the mode differs between the control and
the treatments.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.sampling.multistate_poe import sample_states_fused
from aminx.utils.decoding_order import random_decoding_order, random_design_order

LENGTH = 16
PROBE = 7
WEIGHT_SCALE = 4.0
DESIGNED = (2, 9, 11)


# ---------------------------------------------------------------- order (cheap, no model)
def _fixed_mask():
  m = np.ones(12, dtype=np.float32)
  m[list(DESIGNED)] = 0.0
  return jnp.asarray(m)


def test_random_mode_fixed_block_order_varies_with_the_key():
  """Control: the reference behaviour randomises the fixed block, so the index-mode test can fail."""
  tie = jnp.arange(12, dtype=jnp.int32)
  blocks = {
    tuple(np.asarray(random_design_order(jax.random.PRNGKey(k), tie, _fixed_mask(), "random"))[:9]) for k in range(8)
  }
  assert len(blocks) > 1


def test_index_mode_fixed_block_is_position_order_and_key_independent():
  tie = jnp.arange(12, dtype=jnp.int32)
  fixed_positions = [i for i in range(12) if i not in DESIGNED]
  designed_orders = set()
  for k in range(8):
    order = np.asarray(random_design_order(jax.random.PRNGKey(k), tie, _fixed_mask(), "index"))
    assert list(order[:9]) == fixed_positions
    designed_orders.add(tuple(order[9:]))
  assert len(designed_orders) > 1  # designed positions are still randomised


def test_default_mode_is_the_reference_order_bit_for_bit():
  tie = jnp.arange(12, dtype=jnp.int32)
  key = jax.random.PRNGKey(4)
  assert np.array_equal(
    np.asarray(random_design_order(key, tie, _fixed_mask())),
    np.asarray(random_design_order(key, tie, _fixed_mask(), "random")),
  )


def test_invalid_fixed_order_raises():
  with pytest.raises(ValueError, match="fixed_order"):
    random_design_order(jax.random.PRNGKey(0), jnp.arange(12, dtype=jnp.int32), _fixed_mask(), "bogus")


# ---------------------------------------------------------------- model and bundle
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
  scaled = jax.tree_util.tree_map(lambda x: x * WEIGHT_SCALE if eqx.is_inexact_array(x) else x, model)
  return eqx.tree_inference(scaled, value=True)


def _bundle(num_states: int):
  rng = np.random.default_rng(5)
  coordinates = jnp.array(rng.normal(size=(LENGTH, 4, 3)).astype(np.float32))
  stack = lambda x: jnp.stack([x] * num_states, axis=0)  # noqa: E731
  fixed_mask = jnp.ones((LENGTH,), dtype=jnp.float32).at[PROBE].set(0.0)
  fixed_tokens = jnp.asarray(rng.integers(0, 20, size=(LENGTH,)), dtype=jnp.int32)
  return build_inference_bundle(
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


def _mask_for(mode: str, key: int = 0):
  bundle, _ = _bundle(1)
  out = with_decoding_order(bundle, jax.random.PRNGKey(key), fixed_context=mode)
  return np.asarray(out.conditioning.ar_mask)[0]


def test_mutual_context_mask_structure():
  mask = _mask_for("mutual_context")
  fixed = [i for i in range(LENGTH) if i != PROBE]
  for i in fixed:
    for j in fixed:
      assert mask[i, j] == (0.0 if i == j else 1.0)
    assert mask[i, PROBE] == 0.0  # fixed positions never see the free probe
  assert all(mask[PROBE, j] == 1.0 for j in fixed)  # the probe decodes last and sees all of them


def test_mutual_context_mask_is_key_independent_and_default_varies():
  assert np.array_equal(_mask_for("mutual_context", 0), _mask_for("mutual_context", 1))
  assert np.array_equal(_mask_for("random_order", 0), _mask_for("random_order", 0))
  assert not np.array_equal(_mask_for("random_order", 0), _mask_for("random_order", 1))  # control


def test_invalid_mode_and_unsupported_combination_raise():
  bundle, _ = _bundle(1)
  with pytest.raises(ValueError, match="fixed_context"):
    with_decoding_order(bundle, jax.random.PRNGKey(0), fixed_context="bogus")
  with pytest.raises(ValueError, match="decoding_order_fn"):
    with_decoding_order(bundle, jax.random.PRNGKey(0), random_decoding_order, None, fixed_context="canonical_order")


# ---------------------------------------------------------------- end to end, no seed, different keys
def _probe_logits(num_states: int, key, mode: str):
  bundle, config = _bundle(num_states)
  stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(num_states))
  _, logits = sample_states_fused(_model(), bundle, config, stage_set, key, 1, fixed_position_context=mode)
  v = np.asarray(logits[0, PROBE, :], dtype=np.float64)
  return v - v.mean()


def _residual(mode: str, k: int = 3) -> float:
  single = _probe_logits(1, jax.random.PRNGKey(0), mode)
  fused = _probe_logits(k, jax.random.PRNGKey(1), mode)
  return float(np.linalg.norm(fused / k - single) / max(float(np.linalg.norm(single)), 1e-12))


def test_random_order_leaves_a_fused_vs_single_residual_control():
  assert _residual("random_order") > 1e-3


@pytest.mark.parametrize("mode", ["canonical_order", "mutual_context"])
def test_other_modes_remove_the_residual_without_any_seed(mode):
  assert _residual(mode) < 1e-4


def test_spec_refuses_a_non_default_mode_on_paths_that_cannot_honour_it():
  """A mode that would be recorded in the store but ignored by the sampler is refused at the spec."""
  from aminx.run.specs import SamplingSpecification

  base = {"inputs": ["/tmp/x.pdb"], "checkpoint_id": "ckpt_001"}
  for bypass in ({"decoding_order_fn": random_decoding_order}, {"wave": object()}, {"sampling_strategy": "straight_through"}):
    with pytest.raises(ValueError, match="no effect"):
      SamplingSpecification(**base, fixed_position_context="canonical_order", **bypass)
  # control: the default mode is accepted alongside the same bypasses, and the mode alone is accepted
  SamplingSpecification(**base, decoding_order_fn=random_decoding_order)
  SamplingSpecification(**base, fixed_position_context="mutual_context")


def test_fused_sampler_refuses_modes_when_the_order_is_not_randomized():
  """``randomize_decoding_order=False`` skips with_decoding_order, so a mode or seed would be silently dropped."""
  bundle, config = _bundle(1)
  stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(1))
  for kwargs in ({"fixed_position_context": "canonical_order"}, {"decoding_order_seed": 3}):
    with pytest.raises(ValueError, match="randomize_decoding_order"):
      sample_states_fused(_model(), bundle, config, stage_set, jax.random.PRNGKey(0), 1, randomize_decoding_order=False, **kwargs)


def test_mutual_context_and_canonical_order_give_distinct_probe_logits():
  """The two non-default modes are different conditionings, not aliases of each other."""
  canonical = _probe_logits(1, jax.random.PRNGKey(0), "canonical_order")
  mutual = _probe_logits(1, jax.random.PRNGKey(0), "mutual_context")
  assert np.linalg.norm(canonical - mutual) / np.linalg.norm(canonical) > 1e-3


def test_index_mode_orders_tied_fixed_groups_by_first_position():
  """Tie groups sort by their first member, not by their id; designed groups still follow."""
  tie = jnp.asarray([2, 0, 2, 0, 1, 1, 3, 3], dtype=jnp.int32)
  fixed = jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.float32)
  for k in range(6):
    order = list(np.asarray(random_design_order(jax.random.PRNGKey(k), tie, fixed, "index")))
    # group {0,2} (first position 0) precedes group {1,3} (first position 1); members adjacent by position
    assert order[:4] == [0, 2, 1, 3]
    assert set(order[4:]) == {4, 5, 6, 7}


def test_mutual_context_ignores_a_non_fixed_member_of_a_partly_fixed_tie_group():
  """A non-fixed tie member is undrawn until its wave, so no fixed position may be told to read it."""
  bundle, _ = _bundle(1)
  tie = jnp.arange(LENGTH, dtype=jnp.int32).at[3].set(PROBE)  # positions 3 and PROBE tied; 3 is fixed, PROBE is not
  cond = eqx.tree_at(lambda c: c.tie_group_map, bundle.conditioning, jnp.broadcast_to(tie, bundle.conditioning.tie_group_map.shape))
  bundle = eqx.tree_at(lambda b: b.conditioning, bundle, cond)
  key = jax.random.PRNGKey(0)
  base = np.asarray(with_decoding_order(bundle, key, fixed_context="random_order").conditioning.ar_mask)[0]
  mutual = np.asarray(with_decoding_order(bundle, key, fixed_context="mutual_context").conditioning.ar_mask)[0]
  added = mutual - base
  assert (added >= 0).all()  # mutual_context only ever adds visibility
  # PROBE is not itself fixed (only its tie partner 3 is), so mutual_context adds nothing that reads or is read by it
  assert added[PROBE].sum() == 0 and added[:, PROBE].sum() == 0
  fixed = [i for i in range(LENGTH) if i != PROBE]
  # and among the genuinely fixed positions every pair ends up mutually visible
  assert all(mutual[i, j] == 1.0 for i in fixed for j in fixed if i != j)
