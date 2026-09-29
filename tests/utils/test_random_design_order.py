"""The default sampling order: fixed positions first, then a uniform random order of tie groups.

Every sampling path draws its default decoding order from ``random_design_order`` via
``with_decoding_order``. These pin the three properties that make it the reference order
(ProteinMPNN's ``argsort((chain_mask + 1e-4) * |randn|)``) plus tie groups drawn as units.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
from aminx.types.bundles import WaveScheduleBundle
from aminx.utils.autoregression import decoding_order_from_wave, generate_wave_ar_mask
from aminx.utils.decoding_order import random_design_order


def test_is_a_permutation_and_varies_with_the_key():
  tie = jnp.arange(12, dtype=jnp.int32)
  a = np.asarray(random_design_order(jax.random.PRNGKey(0), tie))
  b = np.asarray(random_design_order(jax.random.PRNGKey(1), tie))
  assert sorted(a.tolist()) == list(range(12))
  assert not np.array_equal(a, b)


def test_fixed_positions_come_first():
  tie = jnp.arange(10, dtype=jnp.int32)
  fixed = jnp.zeros(10).at[jnp.array([2, 7, 9])].set(1.0)
  for seed in range(20):
    order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie, fixed))
    assert set(order[:3].tolist()) == {2, 7, 9}


def test_tied_members_are_adjacent_in_position_order():
  tie = jnp.array([0, 1, 0, 2, 1, 0], dtype=jnp.int32)
  for seed in range(20):
    order = np.asarray(random_design_order(jax.random.PRNGKey(seed), tie)).tolist()
    for group in (0, 1, 2):
      members = [p for p in order if int(tie[p]) == group]
      start = order.index(members[0])
      assert order[start : start + len(members)] == sorted(members)


def test_group_order_is_uniform_not_size_weighted():
  """One 10-member group + 10 singletons: 11 groups, so the big one leads 1/11 of the time.

  A per-position draw would let the big group lead whenever ANY of its 10 members drew
  first -- about 10/20 of the time. 2000 draws put 1/11 = 0.091 well inside [0.06, 0.125]
  (about 5 standard errors either side) and 0.5 far outside.
  """
  tie = jnp.concatenate([jnp.zeros(10, dtype=jnp.int32), jnp.arange(1, 11, dtype=jnp.int32)])
  keys = jax.random.split(jax.random.PRNGKey(123), 2000)
  firsts = np.asarray(jax.vmap(lambda k: random_design_order(k, tie)[0])(keys))
  frac_big_first = float(np.mean(firsts < 10))  # noqa: PLR2004
  assert 0.06 < frac_big_first < 0.125, frac_big_first


def test_with_decoding_order_sets_wave_and_mask_from_one_order():
  seq_len = 8
  tie = jnp.array([0, 0, 1, 2, 3, 3, 4, 5], dtype=jnp.int32)
  fixed = jnp.zeros(seq_len).at[5].set(1.0)
  bundle, _ = build_inference_bundle(
    coords=jnp.zeros((seq_len, 4, 3)),
    mask=jnp.ones(seq_len),
    residue_index=jnp.arange(seq_len),
    chain_index=jnp.zeros(seq_len, dtype=jnp.int32),
    fixed_mask=fixed,
    fixed_tokens=jnp.zeros(seq_len, dtype=jnp.int32),
    tie_group_map=tie,
    mode="sample_ar",
    inference=True,
  )
  key = jax.random.PRNGKey(7)
  out = jax.jit(with_decoding_order)(bundle, key)

  order = random_design_order(key, tie, fixed)
  expected_mask = generate_wave_ar_mask(WaveScheduleBundle.from_decoding_order(order, tie), tie)
  np.testing.assert_array_equal(np.asarray(out.conditioning.ar_mask[0]), np.asarray(expected_mask))
  reported = np.asarray(decoding_order_from_wave(out.wave, tie))
  np.testing.assert_array_equal(reported, np.asarray(order))
  assert set(reported[:2].tolist()) == {4, 5}, "the fixed tie group {4, 5} decodes first"
