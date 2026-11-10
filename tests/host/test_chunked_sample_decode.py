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

from aminx.host.kernel_dispatch import _chunk_tile, _chunked_sample_decode, _dispatch_axis

TRACES = {"count": 0}
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
    outs.append(_chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, tile=35))
  assert TRACES["count"] == 1, f"retraced across rows: {TRACES['count']}"
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


def test_different_tile_is_a_different_executable_and_still_correct():
  """A different tile compiles separately but gives the same tokens."""
  TRACES["count"] = 0
  enc, bundle, keys = _inputs(60, 1.0)
  decode = _StubDecode(scale=jnp.float32(2.0))
  a, _ = _chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, tile=20)
  b, _ = _chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, tile=30)
  assert TRACES["count"] == 2
  np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class _XtraxChunked:
  """xtrax spelling of a chunked strategy: the tile is ``batch_size``."""

  batch_size = 35


_XtraxChunked.__name__ = "ChunkedMap"


class _AminxSafeMap:
  """aminx spelling of a chunked strategy: the tile is ``tile``."""

  tile = 7


_AminxSafeMap.__name__ = "SafeMap"


class _Vmap:
  """A non-chunked strategy."""


_Vmap.__name__ = "Vmap"


def test_chunk_tile_reads_both_strategy_spellings():
  """``_chunk_tile`` reads xtrax's batch_size and aminx's tile, and returns None for Vmap."""
  assert _chunk_tile(_XtraxChunked()) == 35
  assert _chunk_tile(_AminxSafeMap()) == 7
  assert _chunk_tile(_Vmap()) is None


def test_dispatch_axis_still_serves_a_chunked_strategy():
  """The generic dispatcher still handles a chunked strategy for the paths that keep using it."""
  out = _dispatch_axis(_XtraxChunked(), lambda x: x + 1, jnp.arange(70))
  np.testing.assert_array_equal(np.asarray(out), np.arange(70) + 1)
