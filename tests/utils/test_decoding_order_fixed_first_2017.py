"""Decoding order places fixed positions before designed ones (#2017).

ProteinMPNN decodes fixed, not-designed, and padded positions first
(``chain_mask`` 1 = designed). ``random_decoding_order`` keeps the historical
permutation when ``chain_mask`` is omitted.
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.sampling import sample
from aminx.utils.autoregression import get_decoding_step_map
from aminx.utils.decoding_order import random_decoding_order


def _legacy_random_decoding_order(prng_key, num_residues, tie_group_map=None, num_groups=None):
  """Historical ``random_decoding_order`` (no design mask). Reference for (b) and (c)."""
  current_key, next_key = jax.random.split(prng_key)

  if num_residues < 0:
    msg = f"num_residues must be non-negative, but got {num_residues}"
    raise TypeError(msg)

  if tie_group_map is None:
    decoding_order = jax.random.permutation(current_key, jnp.arange(0, num_residues))
    return jnp.asarray(decoding_order, dtype=jnp.int32), next_key

  if num_groups is None:
    msg = "num_groups must be provided when tie_group_map is not None"
    raise ValueError(msg)

  group_order = jax.random.permutation(current_key, jnp.arange(num_groups))
  decoding_step_map = get_decoding_step_map(tie_group_map, group_order, num_groups)
  decoding_order = jnp.argsort(decoding_step_map)
  return jnp.asarray(decoding_order, dtype=jnp.int32), next_key


def _non_designed_precede(order: jnp.ndarray, chain_mask: jnp.ndarray) -> bool:
  """True when every non-designed position precedes every designed one.

  ``chain_mask`` uses 1 = designed. Vacuously true when one of the two blocks
  is empty.
  """
  designed = chain_mask[order] > 0
  n_designed = int(jnp.sum(designed))
  if n_designed == 0 or n_designed == int(order.shape[0]):
    return True
  first_designed = int(jnp.argmax(designed))
  before_clear = not bool(jnp.any(designed[:first_designed]))
  after_all_designed = bool(jnp.all(designed[first_designed:]))
  return before_clear and after_all_designed


def test_mask_places_non_designed_positions_first():
  """(a) With a mask, non-designed positions precede designed ones."""
  lengths = (1, 5, 8, 17)
  seeds = (0, 1, 2, 7, 99)
  for length in lengths:
    for seed in seeds:
      chain_mask = jnp.array([(i % 3) != 0 for i in range(length)], dtype=jnp.float32)
      order, _ = random_decoding_order(
        jax.random.PRNGKey(seed),
        length,
        chain_mask=chain_mask,
      )
      assert _non_designed_precede(order, chain_mask), (
        f"designed position decoded before a fixed one (L={length}, seed={seed})"
      )
      assert int(jnp.unique(order).shape[0]) == length

  # Homogeneous tie groups: a group is designed only when its members are.
  for n_groups in (1, 3, 5):
    length = n_groups * 2
    tie_group_map = jnp.repeat(jnp.arange(n_groups, dtype=jnp.int32), 2)
    chain_mask = jnp.where((tie_group_map % 2) == 0, 0.0, 1.0).astype(jnp.float32)
    for seed in seeds:
      order, _ = random_decoding_order(
        jax.random.PRNGKey(seed),
        length,
        tie_group_map,
        n_groups,
        chain_mask=chain_mask,
      )
      assert _non_designed_precede(order, chain_mask), (
        f"tie-group order mixed designed and fixed (G={n_groups}, seed={seed})"
      )

  # Mixed group: designed if ANY member is designed, so the fixed partner in
  # that group is decoded with the designed block, after fully fixed groups.
  tie_group_map = jnp.array([0, 0, 1], dtype=jnp.int32)
  chain_mask = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
  for seed in seeds:
    order, _ = random_decoding_order(
      jax.random.PRNGKey(seed),
      3,
      tie_group_map,
      2,
      chain_mask=chain_mask,
    )
    rank = jnp.argsort(order)
    assert int(rank[2]) < int(rank[0])
    assert int(rank[2]) < int(rank[1])


def test_omitted_mask_matches_legacy_permutation():
  """(b) No mask is bit-identical to the historical implementation."""
  seeds = (0, 1, 2, 7, 99)
  for length in (0, 1, 5, 8, 17):
    for seed in seeds:
      key = jax.random.PRNGKey(seed)
      got, got_key = random_decoding_order(key, length)
      ref, ref_key = _legacy_random_decoding_order(key, length)
      assert jnp.array_equal(got, ref)
      assert jnp.array_equal(got_key, ref_key)
      assert got.dtype == ref.dtype

  for seed in seeds:
    key = jax.random.PRNGKey(seed)
    tie_group_map = jnp.array([0, 1, 0, 2, 1], dtype=jnp.int32)
    got, got_key = random_decoding_order(key, 5, tie_group_map, 3)
    ref, ref_key = _legacy_random_decoding_order(key, 5, tie_group_map, 3)
    assert jnp.array_equal(got, ref)
    assert jnp.array_equal(got_key, ref_key)

  # All-designed mask, including unused group ids, matches the omitted-mask order.
  tie_group_map = jnp.array([0, 0, 1], dtype=jnp.int32)
  all_designed = jnp.ones((3,), dtype=jnp.float32)
  for seed in seeds:
    key = jax.random.PRNGKey(seed)
    omitted, omitted_key = random_decoding_order(key, 3, tie_group_map, 4)
    explicit, explicit_key = random_decoding_order(
      key,
      3,
      tie_group_map,
      4,
      chain_mask=all_designed,
    )
    assert jnp.array_equal(explicit, omitted)
    assert jnp.array_equal(explicit_key, omitted_key)
    ref, _ = _legacy_random_decoding_order(key, 3, tie_group_map, 4)
    assert jnp.array_equal(omitted, ref)


def test_legacy_order_violates_fixed_first_on_fixture():
  """(c) The historical permutation does not put fixed positions first."""
  chain_mask = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
  witness = None
  for seed in range(64):
    order, _ = _legacy_random_decoding_order(jax.random.PRNGKey(seed), 6)
    if not _non_designed_precede(order, chain_mask):
      witness = seed
      break
  assert witness is not None, "legacy permutation unexpectedly respected the design mask"
  order, _ = _legacy_random_decoding_order(jax.random.PRNGKey(witness), 6)
  assert not _non_designed_precede(order, chain_mask)


class _DummyModel(eqx.Module):
  """Pytree stand-in. ``sample`` closes over the model inside ``jax.jit``."""

  sentinel: jax.Array


class _KernelOut(eqx.Module):
  sequence: jax.Array
  logits: jax.Array


def _stub_sample_kernel(model, prng_key, bundle, config, stage_set, inference_only=False):
  """Stand-in for the AR kernel so the order thread can be checked without a model."""
  del model, prng_key, config, stage_set, inference_only
  length = bundle.geometry.coords.shape[1]
  return _KernelOut(
    sequence=jnp.zeros((length,), dtype=jnp.int8),
    logits=jnp.zeros((length, 21), dtype=jnp.float32),
  )


def test_sample_threads_fixed_positions_mask(monkeypatch):
  """(d) ``sample()`` passes the design mask built from fixed positions.

  ``fixed_mask`` is sample()'s fixed-position argument (1 = fixed). The design
  mask is its complement on valid residues (1 = designed). This assertion fails
  when that mask is dropped at the ``decoding_order_fn`` call in
  ``aminx.sampling.sample``.
  """
  # ``aminx.sampling`` re-exports the ``sample`` function, which shadows the
  # ``aminx.sampling.sample`` submodule in dotted-path resolution; patch the
  # module object ``aminx.sampling.sample`` actually imported.
  sample_module = importlib.import_module("aminx.sampling.sample")
  monkeypatch.setattr(sample_module.sample_autoregressive, "kernel", _stub_sample_kernel)
  length = 6
  # 1 = fixed (not designed). Chosen so the unmasked permutation interleaves.
  fixed_mask = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
  chain_mask = 1.0 - fixed_mask
  key = None
  for seed in range(64):
    candidate = jax.random.PRNGKey(seed)
    k_order, _ = jax.random.split(candidate)
    legacy_order, _ = _legacy_random_decoding_order(k_order, length)
    if not _non_designed_precede(legacy_order, chain_mask):
      key = candidate
      break
  assert key is not None

  coords = jnp.zeros((length, 4, 3), dtype=jnp.float32)
  residue_mask = jnp.ones((length,), dtype=jnp.float32)
  residue_index = jnp.arange(length, dtype=jnp.int32)
  chain_index = jnp.zeros((length,), dtype=jnp.int32)
  _seq, _logits, order = sample(
    key,
    _DummyModel(sentinel=jnp.array(0, dtype=jnp.int32)),
    coords,
    residue_mask,
    residue_index,
    chain_index,
    fixed_mask=fixed_mask,
    inference_only=True,
  )
  assert _non_designed_precede(order, chain_mask), (
    "sample() decoding order ignored fixed positions; the design mask was not threaded"
  )
