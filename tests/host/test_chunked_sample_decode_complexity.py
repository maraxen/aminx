"""Chunked sample-decode seam and Scan dispatch complexity (debt #1981).

Both loops iterate once per sample. Per-iteration work must not grow with N
(criterion A). The seam is traced with a stub decode so the guard covers the
chunk plumbing only; the real decoder is guarded by the AR, STE and score
tests. Work is a jaxpr proxy, not wall time.

The seam mutant wraps ``chunked_map`` so each chunk also runs ``f`` over every
sample (three passes: one pass sat under the 1.7 margin on the graded probe).
The Scan mutant closes over the whole ``xs``. Fresh extents per case keep a
``filter_jit`` cache from hiding the patched body. The patch uses
``monkeypatch``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.host.kernel_dispatch import _chunked_sample_decode, _dispatch_axis
from aminx.tiling.strategy import Scan
from tests.helpers.loop_scaling import (
  A_MUTANT_MIN,
  A_THRESHOLD,
  LoopRow,
  assert_work_independent,
  measure_rows,
  pick_trips_scaling,
  work_ratios,
)

ORDER_CFG = (True, 0, None, None, "none")
_SENTINEL = {"seam": 0, "scan": 0}


@dataclass
class _Result:
  """Stand-in decode result (a pytree of the two arrays the seam returns)."""

  sequence: jax.Array
  logits: jax.Array


class _StubDecode(eqx.Module):
  """Cheap decode stand-in so the seam guard traces the chunk loop only."""

  scale: jax.Array

  def __call__(self, enc: jax.Array, bundle: jax.Array, key: jax.Array, config: Any) -> _Result:
    del config
    seq = (enc * self.scale + bundle + jax.random.randint(key, enc.shape, 0, 1)).astype(jnp.int32)
    return _Result(sequence=seq, logits=jnp.stack([seq.astype(jnp.float32)] * 3, axis=-1))


def _trace_seam(extent: int) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  enc = jnp.ones((4,), dtype=jnp.float32)
  bundle = jnp.ones((4,), dtype=jnp.float32)
  keys = jax.random.split(jax.random.PRNGKey(0), extent)
  decode = _StubDecode(scale=jnp.float32(2.0))

  def fn(enc_arr: jax.Array, bundle_arr: jax.Array, key_arr: jax.Array) -> Any:
    return _chunked_sample_decode(decode, enc_arr, bundle_arr, None, key_arr, ORDER_CFG, batch_size=8)

  return fn, (enc, bundle, keys)


def _seam_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  """Per-chunk body also runs ``f`` over all ``xs``.

  One pass leaves the N=16→32 ratio under the 1.7 margin, because the chunk's
  own work is still a large share of a 16-wide vmap. Three passes keep the same
  whole-body shape and put both ratios above 1.7.
  """

  def bad(fn: Callable[..., Any], xs: Any, batch_size: int | None = None) -> Any:
    def wrapped(x: Any) -> Any:
      _SENTINEL["seam"] += 1
      y = fn(x)
      extra = jax.vmap(fn)(xs)
      for _ in range(2):
        extra = jax.tree.map(lambda left, right: left + right, extra, jax.vmap(fn)(xs))

      def mix(left: jax.Array, right: jax.Array) -> jax.Array:
        return left + jnp.float32(0) * jnp.sum(right)

      return jax.tree.map(mix, y, extra)

    return real(wrapped, xs, batch_size=batch_size)

  return bad


def _trace_scan(extent: int, *, mutant: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  xs = jnp.ones((extent, 4), dtype=jnp.float32)
  weight = jnp.ones((4, 4), dtype=jnp.float32)

  def clean(x: jax.Array) -> jax.Array:
    return x @ weight

  def bad(x: jax.Array, full: jax.Array = xs, weight_arr: jax.Array = weight) -> jax.Array:
    _SENTINEL["scan"] += 1
    return x @ weight_arr + jnp.float32(0) * jnp.sum(full @ weight_arr)

  body = bad if mutant else clean

  def fn(xs_arr: jax.Array) -> jax.Array:
    return _dispatch_axis(Scan(), body, xs_arr)

  return fn, (xs,)


def _assert_clean(rows: list[LoopRow]) -> LoopRow:
  row = pick_trips_scaling(rows)
  assert row.primitive == "scan"
  assert_work_independent([row])
  assert max(work_ratios(row)) < A_THRESHOLD
  return row


def _assert_mutant(clean: LoopRow, mutant: LoopRow) -> None:
  clean_ratios = work_ratios(clean)
  mutant_ratios = work_ratios(mutant)
  assert max(clean_ratios) < A_THRESHOLD
  assert min(mutant_ratios) >= A_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_independent([mutant])


def test_chunked_sample_decode_work_is_independent_of_n() -> None:
  _assert_clean(measure_rows(_trace_seam, (16, 32, 64)))


def test_chunked_sample_decode_full_xs_fold_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  import aminx.host.kernel_dispatch as dispatch_mod

  clean = measure_rows(_trace_seam, (16, 32, 64))
  _SENTINEL["seam"] = 0
  monkeypatch.setattr(dispatch_mod, "chunked_map", _seam_fold(dispatch_mod.chunked_map))
  # Fresh N: a same-shape filter_jit hit must not hide the patched body.
  mutant = measure_rows(_trace_seam, (24, 48, 96))
  assert _SENTINEL["seam"] > 0
  _assert_mutant(pick_trips_scaling(clean), pick_trips_scaling(mutant))


def test_scan_dispatch_work_is_independent_of_n() -> None:
  _assert_clean(measure_rows(lambda n: _trace_scan(n, mutant=False), (32, 64, 128)))


def test_scan_dispatch_full_xs_closure_must_fire() -> None:
  _SENTINEL["scan"] = 0
  clean = measure_rows(lambda n: _trace_scan(n, mutant=False), (32, 64, 128))
  mutant = measure_rows(lambda n: _trace_scan(n, mutant=True), (48, 96, 192))
  assert _SENTINEL["scan"] > 0
  _assert_mutant(pick_trips_scaling(clean), pick_trips_scaling(mutant))
