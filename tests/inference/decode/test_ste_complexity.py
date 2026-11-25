"""STE iteration complexity (debt #1981).

``STEDecode`` and ``make_optimize_sequence_fn`` run a full decode a fixed number
of times. Per-iteration work may grow linearly with L and must stay under 3
(criterion B). Trip counts must not start scaling with L. Work is a jaxpr
proxy, not wall time.

The must-fire mutant re-runs the decode on L perturbed copies and keeps element
0, the fold the graded probe measured (STE ratios about 4.0, optimize_ste about
4.0). ``optimize_ste`` is patched with ``monkeypatch``, not ``try``/``finally``.

``make_optimize_sequence_fn`` lowers ``fori_loop`` to a ``while`` because
``iterations`` is a traced jit argument. The helper recovers that static trip
count. A second mutant multiplies the upper bound by L; the recovered trips
then scale with L and criterion B rejects them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from xtrax.tiling import VmapIterator

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.decode.ste import STEDecode
from aminx.inference.logits import make_stage_set
from aminx.inference.optimize_ste import make_optimize_sequence_fn
from tests.helpers.aminx_small import make_guard_model
from tests.helpers.loop_scaling import (
  B_MUTANT_MIN,
  B_THRESHOLD,
  LoopRow,
  assert_work_at_most_linear,
  measure_rows,
  pick_fixed_scan_largest,
  pick_outermost_while,
  work_ratios,
)

_MODEL = make_guard_model()
_SENTINEL = {"ste_untied": 0, "ste_tied": 0, "optimize_ste": 0, "optimize_trips": 0}
_EXTENTS = (32, 64, 128)


def _ste_bundle(length: int, *, tied: bool) -> tuple[Any, Any]:
  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(1, length, 4, 3)), dtype=jnp.float32)
  mask = jnp.ones((1, length), dtype=jnp.float32)
  residue_index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (1, length))
  chain_index = jnp.zeros((1, length), dtype=jnp.int32)
  tokens = jnp.asarray(rng.integers(0, 20, size=(length,)), dtype=jnp.int32)
  tie = jnp.arange(length, dtype=jnp.int32)
  if tied:
    tie = tie.at[5].set(0).at[9].set(0).at[17].set(2)
  return build_inference_bundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    sequence=jax.nn.one_hot(tokens, 21),
    state_weights=jnp.ones((1,), dtype=jnp.float32),
    mode="score_conditional",
    tie_group_map=tie,
  )


class FoldConditional(eqx.Module):
  """Call the inner decode on ``n_fold`` copies and keep element 0."""

  inner: ConditionalDecode
  n_fold: int = eqx.field(static=True)
  which: str = eqx.field(static=True)

  @property
  def model(self) -> Any:
    return self.inner.model

  def __call__(
    self, key: jax.Array, enc: Any, bundle: Any, config: Any, stage_set: Any
  ) -> jax.Array:
    _SENTINEL[self.which] += 1

    def one(index: jax.Array) -> jax.Array:
      # The index has to reach the body. A vmap that ignores it is deleted by
      # make_jaxpr when only element 0 is kept, and the fold then costs nothing.
      shifted = eqx.tree_at(
        lambda item: item.node_features,
        enc,
        enc.node_features + index.astype(enc.node_features.dtype),
      )
      return self.inner(key=key, enc=shifted, bundle=bundle, config=config, stage_set=stage_set)

    logits = jax.vmap(one)(jnp.arange(self.n_fold))
    return logits[0] + jnp.float32(0) * jnp.sum(logits)


def _trace_ste(
  extent: int, *, tied: bool, mutant: bool
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  built, config = _ste_bundle(extent, tied=tied)
  inner: Any = ConditionalDecode(model=_MODEL, state_iterator=VmapIterator())
  if mutant:
    which = "ste_tied" if tied else "ste_untied"
    inner = FoldConditional(inner=inner, n_fold=extent, which=which)
  ste = STEDecode(inner=inner, iterations=3, optimizer=optax.adam(1e-3))
  stage = make_stage_set(state_weights=built.conditioning.state_weights)
  key = jax.random.PRNGKey(0)

  def fn(key_arr: jax.Array, bundle_arr: Any, config_arr: Any) -> Any:
    return ste(key_arr, bundle_arr, config_arr, 1.0, stage)

  return fn, (key, built, config)


def _trace_optimize(extent: int, *, tied: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  built, config = _ste_bundle(extent, tied=tied)
  stage = make_stage_set(state_weights=built.conditioning.state_weights)
  opt_fn = make_optimize_sequence_fn(_MODEL, stage)
  key = jax.random.PRNGKey(0)

  def fn(key_arr: jax.Array, bundle_arr: Any, config_arr: Any) -> Any:
    return opt_fn(key_arr, bundle_arr, config_arr, 3, 1e-3, 1.0)

  return fn, (key, built, config)


def _optimize_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(model: Any, prng_key: jax.Array, bundle: Any, config: Any, stage_set: Any) -> jax.Array:
    _SENTINEL["optimize_ste"] += 1
    length = bundle.geometry.coords.shape[1]

    def one(index: jax.Array) -> jax.Array:
      coords = bundle.geometry.coords + index.astype(bundle.geometry.coords.dtype)
      shifted = eqx.tree_at(lambda item: item.geometry.coords, bundle, coords)
      return real(model, prng_key, shifted, config, stage_set)

    logits = jax.vmap(one)(jnp.arange(length))
    return logits[0] + jnp.float32(0) * jnp.sum(logits)

  return bad


def _assert_clean(row: LoopRow) -> None:
  assert_work_at_most_linear([row])
  assert max(work_ratios(row)) < B_THRESHOLD


def _assert_mutant(clean: LoopRow, mutant: LoopRow) -> None:
  clean_ratios = work_ratios(clean)
  mutant_ratios = work_ratios(mutant)
  assert max(clean_ratios) < B_THRESHOLD
  assert min(mutant_ratios) > B_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_at_most_linear([mutant])


@pytest.mark.parametrize("tied", [False, True])
def test_ste_decode_work_is_at_most_linear(tied: bool) -> None:
  rows = measure_rows(lambda n: _trace_ste(n, tied=tied, mutant=False), _EXTENTS)
  # STEDecode(iterations=3): the graded probe records trips=(3, 3, 3).
  row = pick_fixed_scan_largest(rows, expected_trips=(3, 3, 3))
  assert row.primitive == "scan"
  assert row.trips == [3, 3, 3]
  _assert_clean(row)


@pytest.mark.parametrize("tied", [False, True])
def test_ste_decode_length_fold_must_fire(tied: bool) -> None:
  which = "ste_tied" if tied else "ste_untied"
  _SENTINEL[which] = 0
  clean = measure_rows(lambda n: _trace_ste(n, tied=tied, mutant=False), _EXTENTS)
  mutant = measure_rows(lambda n: _trace_ste(n, tied=tied, mutant=True), _EXTENTS)
  assert _SENTINEL[which] > 0
  _assert_mutant(
    pick_fixed_scan_largest(clean, expected_trips=(3, 3, 3)),
    pick_fixed_scan_largest(mutant, expected_trips=(3, 3, 3)),
  )


def test_optimize_ste_work_is_at_most_linear() -> None:
  rows = measure_rows(lambda n: _trace_optimize(n, tied=False), _EXTENTS)
  row = pick_outermost_while(rows)
  assert row.primitive == "while"
  # iterations=3 is a static bound even though fori_loop lowered to while.
  assert row.trips == [3, 3, 3]
  _assert_clean(row)


def test_optimize_ste_trip_count_must_not_scale_with_length(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Trips must stay at ``iterations``. Scaling the fori upper bound by L has to fail."""
  import jax.lax as lax

  scale = {"L": 0}
  real = lax.fori_loop

  def scaled(
    lower: Any, upper: Any, body_fun: Any, init_val: Any, *args: Any, **kwargs: Any
  ) -> Any:
    if isinstance(upper, (int, np.integer)):
      return real(lower, upper, body_fun, init_val, *args, **kwargs)
    _SENTINEL["optimize_trips"] += 1
    return real(lower, upper * scale["L"], body_fun, init_val, *args, **kwargs)

  def trace(extent: int) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    scale["L"] = extent
    return _trace_optimize(extent, tied=False)

  monkeypatch.setattr(lax, "fori_loop", scaled)
  _SENTINEL["optimize_trips"] = 0
  mutant = measure_rows(trace, _EXTENTS)
  assert _SENTINEL["optimize_trips"] > 0
  row = pick_outermost_while(mutant)
  # iterations=3, extents 32/64/128: the recovered upper bound is iterations * L.
  assert row.trips == [3 * extent for extent in _EXTENTS]
  with pytest.raises(AssertionError, match="trips must not depend on L"):
    assert_work_at_most_linear([row])


def test_optimize_ste_length_fold_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  import aminx.inference.optimize_ste as opt_mod

  clean = measure_rows(lambda n: _trace_optimize(n, tied=False), _EXTENTS)
  _SENTINEL["optimize_ste"] = 0
  monkeypatch.setattr(opt_mod, "score_conditional", _optimize_fold(opt_mod.score_conditional))
  mutant = measure_rows(lambda n: _trace_optimize(n, tied=False), _EXTENTS)
  assert _SENTINEL["optimize_ste"] > 0
  _assert_mutant(pick_outermost_while(clean), pick_outermost_while(mutant))
