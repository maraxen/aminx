"""potts_merge_pair_d4 wave: merge_pair(denom=4, exclude_self) against etab_energy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_close, assert_shape_dtype, iter_cases, open_dump, x64_context

from aminx.families.potts_mpnn.etab import merge_pair

pytestmark = [pytest.mark.port_wave("potts_merge_pair_d4"), pytest.mark.parity_heavy]

_WAVE = "potts_merge_pair_d4"
_SOURCE = "etab_forward"
_TARGET = "etab_energy"
_DENOM = 4
_EXCLUDE_SELF = True
_TOL = {"f64": {"rtol": 0.0, "atol": 0.0}, "f32": {"rtol": 0.0, "atol": 1e-7}}


def _run(data: np.lib.npyio.NpzFile, precision: str, *, numeric: bool) -> int:
  compared = 0
  with x64_context(precision):
    for checkpoint, fixture, prefix, _mask, e_idx, keep in iter_cases(data, require_edges=True):
      table = jnp.asarray(data[prefix + _SOURCE][0])
      pred = np.asarray(
        merge_pair(
          table,
          jnp.asarray(e_idx[0]),
          jnp.ones((table.shape[0],), dtype=bool),
          denom=_DENOM,
          exclude_self=_EXCLUDE_SELF,
        ),
      )
      ref = np.asarray(data[prefix + _TARGET][0])
      label = f"{_WAVE} {precision} {checkpoint} {fixture}"
      if numeric:
        assert_close(pred[keep], ref[keep], label=label, **_TOL[precision])
      else:
        assert_shape_dtype(pred, ref, label)
      compared += 1
  return compared


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle: object) -> None:
  """Merged table dtype and shape match the sealed energy etab."""
  for precision in ("f64", "f32"):
    data = open_dump(oracle, precision)
    try:
      assert _run(data, precision, numeric=False) > 0
    finally:
      data.close()


@pytest.mark.tier_2
def test_tier_2_f64_exact(oracle: object) -> None:
  """f64 merge_pair(denom=4, exclude_self) is exact under x64."""
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
def test_tier_3_f32(oracle: object) -> None:
  """f32 merge_pair(denom=4, exclude_self) matches at atol=1e-7."""
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
def test_tier_5_trace_budget(oracle: object, max_traces: int) -> None:
  """One static shape traces at most ``max_traces`` times."""
  import chex

  data = open_dump(oracle, "f32")
  try:
    _checkpoint, _fixture, prefix, _mask, e_idx, _keep = next(iter_cases(data, require_edges=True))
    table = jnp.asarray(data[prefix + _SOURCE][0])
    neighbours = jnp.asarray(e_idx[0])
  finally:
    data.close()
  pad_valid = jnp.ones((table.shape[0],), dtype=bool)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(etab: jax.Array, idx: jax.Array, valid: jax.Array) -> jax.Array:
    return merge_pair(etab, idx, valid, denom=_DENOM, exclude_self=_EXCLUDE_SELF)

  compiled = jax.jit(kernel)
  compiled(table, neighbours, pad_valid)
  compiled(table, neighbours, pad_valid)
