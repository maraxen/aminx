"""aminx #282: the chunked sample axis must reuse ONE compiled executable across rows.

Before the fix, ``_dispatch_axis`` ran a ChunkedMap sample axis as an eager ``lax.map`` over a closure
rebuilt for every row, so jax traced and compiled a fresh ``jit(scan)`` per row (~6 s at 128 samples on
a TITAN RTX). ``_chunked_sample_decode`` takes the per-row data as arguments, so a row of the same shape
hits the cache. Tracing is counted with a Python-side counter that only runs while jax traces.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.host.kernel_dispatch import (
  _chunk_batch_size,
  _chunked_sample_decode,
  _dispatch_axis,
)
from aminx.tiling.strategy import ChunkedMap, Vmap

TRACES = {"count": 0}
# xtrax.transforms.chunked_map traces twice per compilation: once during
# jax.eval_shape (zero-size output check) and once during jax.lax.map.
TRACES_PER_COMPILE = 2
ORDER_CFG = (True, 0, None, None, "none")  # skip_order=True: the bundle is used as is


@dataclass
class _Result:
  """Stand-in for a decode result: a sequence and its logits."""

  sequence: jax.Array
  logits: jax.Array


class _StubDecode(eqx.Module):
  """A decode module with one array leaf; calling it bumps TRACES (a trace-time side effect)."""

  scale: jax.Array

  def __call__(self, enc, bundle, key, config):  # noqa: ANN001, ARG002
    """Decode one sample key.

    Args:
      enc: The encoding stand-in.
      bundle: The bundle stand-in.
      key: The sample key.
      config: Unused.

    Returns:
      A result whose sequence depends on ``enc``, ``bundle`` and ``scale``.
    """
    TRACES["count"] += 1
    seq = (enc * self.scale + bundle + jax.random.randint(key, enc.shape, 0, 1)).astype(jnp.int32)
    return _Result(sequence=seq, logits=jnp.stack([seq.astype(jnp.float32)] * 3, axis=-1))


def _inputs(n_samples: int, value: float):
  """Encoding, bundle and sample keys for one row of ``n_samples`` samples."""
  keys = jax.random.split(jax.random.key(0), n_samples)
  return jnp.full((4,), value, dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32), keys


def test_chunked_decode_traces_once_across_fresh_modules_and_rows():
  """Three rows, each with a NEW decode module, trace once and decode correctly."""
  TRACES["count"] = 0
  outs = []
  for row in range(3):
    enc, bundle, keys = _inputs(70, float(row + 1))
    decode = _StubDecode(scale=jnp.float32(2.0))  # a NEW module object every row
    outs.append(_chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, batch_size=35))
  assert TRACES["count"] == TRACES_PER_COMPILE, f"retraced across rows: {TRACES['count']}"
  for row, (seq, _logits) in enumerate(outs):
    assert seq.shape == (70, 4)
    np.testing.assert_array_equal(np.asarray(seq), np.full((70, 4), (row + 1) * 2 + 1, dtype=np.int32))


def test_negative_control_eager_closure_map_retraces_every_row():
  """The pre-fix shape (a fresh closure per row through an eager lax.map) traces every row, so the counter can fire."""
  TRACES["count"] = 0
  for row in range(3):
    enc, bundle, keys = _inputs(70, float(row + 1))
    decode = _StubDecode(scale=jnp.float32(2.0))

    def body(k, decode=decode, enc=enc, bundle=bundle):
      """Decode one key through a closure over this row's data."""
      return decode(enc, bundle, k, None).sequence

    jax.lax.map(body, keys, batch_size=35)
  assert TRACES["count"] >= 3, f"the counter did not fire per row: {TRACES['count']}"


def test_different_batch_size_is_a_different_executable_and_still_correct():
  """A different batch size compiles separately but gives the same tokens."""
  TRACES["count"] = 0
  enc, bundle, keys = _inputs(60, 1.0)
  decode = _StubDecode(scale=jnp.float32(2.0))
  a, _ = _chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, batch_size=20)
  b, _ = _chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, batch_size=30)
  assert TRACES["count"] == 2 * TRACES_PER_COMPILE
  np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_chunk_batch_size_reads_strategy_and_returns_none_for_vmap():
  """``_chunk_batch_size`` reads ChunkedMap.batch_size, normalizing 0 and Vmap to None."""
  assert _chunk_batch_size(ChunkedMap(batch_size=35)) == 35
  assert _chunk_batch_size(ChunkedMap(batch_size=0)) is None
  assert _chunk_batch_size(Vmap()) is None


def test_dispatch_axis_still_serves_a_chunked_strategy():
  """The generic dispatcher still handles a chunked strategy for the paths that keep using it."""
  out = _dispatch_axis(ChunkedMap(batch_size=35), lambda x: x + 1, jnp.arange(70))
  np.testing.assert_array_equal(np.asarray(out), np.arange(70) + 1)


def _patch_order_helpers(monkeypatch):
  """Replace the order helpers with stand-ins that make the seed visible in the decoded tokens."""
  import aminx.host.kernel_dispatch as kd

  def fake_key(k, seed):  # noqa: ARG001
    """Stand-in order key: the seed itself as an int32 scalar (0 when there is none)."""
    return jnp.int32(0) if seed is None else jnp.asarray(seed, dtype=jnp.int32)

  def fake_with_order(bundle, order_key, order_fn, num_groups, fixed_context="random_order"):  # noqa: ARG001
    """Stand-in order application: add the order key to the bundle."""
    return bundle + order_key.astype(bundle.dtype)

  monkeypatch.setattr(kd, "decoding_order_key", fake_key)
  monkeypatch.setattr(kd, "with_decoding_order", fake_with_order)


def test_order_branch_applies_the_seed(monkeypatch):
  """skip_order=False runs the order helpers: the seed reaches the decoded tokens (the default path)."""
  _patch_order_helpers(monkeypatch)
  enc, bundle, keys = _inputs(70, 1.0)
  decode = _StubDecode(scale=jnp.float32(2.0))
  seq, _ = _chunked_sample_decode(
    decode, enc, bundle, None, keys, (False, jnp.int32(5), None, None, "none"), batch_size=35,
  )
  np.testing.assert_array_equal(np.asarray(seq), np.full((70, 4), 1 * 2 + 1 + 5, dtype=np.int32))


def test_different_order_seeds_share_one_executable(monkeypatch):
  """Rows that differ only in decoding-order seed reuse one compiled executable (agy review F1)."""
  from aminx.host.kernel_dispatch import _order_seed_leaf

  _patch_order_helpers(monkeypatch)
  TRACES["count"] = 0
  enc, bundle, keys = _inputs(66, 1.0)  # 66 = 2 x 33: no remainder, so one compile traces once
  decode = _StubDecode(scale=jnp.float32(2.0))
  outs = []
  for seed in (0, 1, 2, 3):
    cfg = (False, _order_seed_leaf(seed), None, None, "none")
    # batch_size=33 is used by no other test, so the first call here must compile (count 1, not a hit).
    outs.append(np.asarray(_chunked_sample_decode(decode, enc, bundle, None, keys, cfg, batch_size=33)[0]))
  assert TRACES["count"] == TRACES_PER_COMPILE, f"retraced per seed: {TRACES['count']}"
  for seed, seq in zip((0, 1, 2, 3), outs, strict=True):
    np.testing.assert_array_equal(seq, np.full((66, 4), 3 + seed, dtype=np.int32))


def test_negative_control_static_python_seed_retraces_per_seed(monkeypatch):
  """A raw Python int seed is a static argument and traces once per seed, so the counter can fire."""
  _patch_order_helpers(monkeypatch)
  TRACES["count"] = 0
  enc, bundle, keys = _inputs(70, 1.0)
  decode = _StubDecode(scale=jnp.float32(2.0))
  for seed in (0, 1, 2):
    _chunked_sample_decode(decode, enc, bundle, None, keys, (False, seed, None, None, "none"), batch_size=35)
  assert TRACES["count"] == 3 * TRACES_PER_COMPILE, f"expected a trace per static seed: {TRACES['count']}"


def test_traced_order_seed_gives_the_same_key_as_the_int():
  """The real PRNGKey of a traced int32 seed equals that of the Python int, so tokens are unchanged."""
  from aminx.host.kernel_dispatch import _order_seed_leaf
  from aminx.inference.bundle_builder import decoding_order_key

  sample_key = jax.random.key(7)
  traced = jax.jit(lambda s: decoding_order_key(sample_key, s))
  for seed in (0, 1, 4679, 2**31 - 1):
    want = jax.random.key_data(decoding_order_key(sample_key, seed))
    got = jax.random.key_data(traced(_order_seed_leaf(seed)))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_order_seed_leaf_passes_through_none_and_out_of_range():
  """None and out-of-int32-range seeds are left as they are (static), everything else becomes an array."""
  from aminx.host.kernel_dispatch import _order_seed_leaf

  assert _order_seed_leaf(None) is None
  assert _order_seed_leaf(2**31) == 2**31
  assert _order_seed_leaf(-1) == -1
  assert isinstance(_order_seed_leaf(3), jax.Array)
