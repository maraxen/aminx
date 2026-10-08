# ruff: noqa: S101
"""Synthetic tests for the Potts pair-energy head."""

from __future__ import annotations

from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.families.potts_mpnn.potts_head import N_AA, PAIR_DIM, PottsHead

pytestmark = pytest.mark.usefixtures("_x64")


@pytest.fixture
def _x64() -> Iterator[None]:
  """Enable ``jax_enable_x64`` for one test, then put it back.

  This used to set the flag and never restore it. ``jax.config`` is process
  global, so every test that ran after this module in the same session built
  its arrays in float64. Restoring matches tests/ebm/conftest.py and
  tests/port/conftest.py, which both already scope the flag.

  NOT a fix for debt #2419, despite the resemblance. An earlier version of
  this docstring claimed the leak caused the 18 tests/host failures that look
  exactly like it ("Deserialised leaf ... has changed dtype from float64 in
  `like` to float32 on disk"). Measured both ways on the same selection: 18
  failed / 799 passed before the fix, 18 failed / 799 passed after, the same
  18 node ids. The leak is real and worth closing on its own; it is not that
  cause. See #2426.
  """
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", previous)


def _head(weight: jax.Array, bias: jax.Array) -> PottsHead:
  module = PottsHead(int(weight.shape[1]), key=jax.random.key(0))
  return module.with_weights(weight, bias)


def test_row_mask_pad_mask_and_self_eye() -> None:
  dtype = jnp.float64
  bias = jnp.arange(PAIR_DIM, dtype=dtype)
  weight = jnp.zeros((PAIR_DIM, 4), dtype=dtype)
  head = _head(weight, bias)
  edge = jnp.ones((3, 2, 4), dtype=dtype)
  # Row 0 is present. Its self slot stays; its neighbour is a pad row.
  # Row 1 is a gap (present 0, still pad_valid). Row 2 is padding.
  e_idx = jnp.asarray([[0, 2], [1, 0], [2, 0]])
  present = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)
  pad_valid = jnp.asarray([True, True, False])

  got = head(edge, e_idx, present, pad_valid)

  table = bias.reshape(N_AA, N_AA)
  expected_self = table * jnp.eye(N_AA, dtype=dtype)
  assert jnp.array_equal(got[0, 0], expected_self)
  assert jnp.array_equal(got[0, 1], jnp.zeros_like(table))
  assert jnp.array_equal(got[1], jnp.zeros_like(got[1]))
  assert jnp.array_equal(got[2], jnp.zeros_like(got[2]))
  # Off-diagonal self entries are cleared even though the linear output is not.
  assert float(table[0, 1]) != 0.0
  assert float(got[0, 0, 0, 1]) == 0.0
  assert float(got[0, 0, 0, 0]) == float(table[0, 0])


def test_linear_reads_edge_features() -> None:
  dtype = jnp.float32
  weight = jnp.zeros((PAIR_DIM, 4), dtype=dtype).at[0, 0].set(1.0)
  bias = jnp.zeros((PAIR_DIM,), dtype=dtype)
  head = _head(weight, bias)
  edge = jnp.zeros((2, 2, 4), dtype=dtype).at[:, :, 0].set(3.0)
  e_idx = jnp.asarray([[0, 1], [1, 0]])
  present = jnp.ones((2,), dtype=dtype)
  pad_valid = jnp.ones((2,), dtype=bool)

  got = head(edge, e_idx, present, pad_valid)

  assert got.dtype == jnp.float32
  assert float(got[0, 1, 0, 0]) == pytest.approx(3.0)
  assert float(got[0, 0, 0, 0]) == pytest.approx(3.0)
  assert float(got[0, 0, 0, 1]) == 0.0
  assert float(jnp.max(jnp.abs(got[0, 1, 0, 1:]))) == 0.0


def test_dtype_follows_inputs() -> None:
  e_idx = jnp.asarray([[0, 1], [1, 0]])
  pad_valid = jnp.ones((2,), dtype=bool)
  for dtype in (jnp.float32, jnp.float64):
    weight = jnp.zeros((PAIR_DIM, 2), dtype=dtype).at[3, 1].set(0.5)
    bias = jnp.zeros((PAIR_DIM,), dtype=dtype)
    head = _head(weight, bias)
    edge = jnp.ones((2, 2, 2), dtype=dtype)
    present = jnp.ones((2,), dtype=dtype)
    got = head(edge, e_idx, present, pad_valid)
    assert got.dtype == dtype


def test_jit_matches_eager() -> None:
  dtype = jnp.float64
  weight = jnp.arange(PAIR_DIM * 3, dtype=dtype).reshape(PAIR_DIM, 3) / 1000.0
  bias = jnp.linspace(-1.0, 1.0, PAIR_DIM, dtype=dtype)
  head = _head(weight, bias)
  edge = jnp.arange(2 * 3 * 3, dtype=dtype).reshape(2, 3, 3) / 10.0
  e_idx = jnp.asarray([[0, 1, 0], [1, 0, 1]])
  present = jnp.asarray([1.0, 0.5], dtype=dtype)
  pad_valid = jnp.asarray([True, True])

  eager = head(edge, e_idx, present, pad_valid)
  jitted = eqx.filter_jit(head)(edge, e_idx, present, pad_valid)
  assert jitted.dtype == eager.dtype
  assert jnp.array_equal(jitted, eager)
