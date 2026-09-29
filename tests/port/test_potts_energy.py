"""potts_energy wave.

f64 uses rtol=1e-10. f32 uses the r15 condition bound
``|err| <= 1e-5 * sum|terms|``, where ``sum|terms| = potts_energy(|etab|, seq)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import (
  assert_close,
  assert_energy_f32,
  assert_shape_dtype,
  iter_cases,
  open_dump,
  x64_context,
)

from aminx.families.potts_mpnn.etab import model_to_etab, potts_energy

pytestmark = [pytest.mark.port_wave("potts_energy"), pytest.mark.parity_heavy]


def _run(data: np.lib.npyio.NpzFile, precision: str, *, numeric: bool) -> int:
  compared = 0
  with x64_context(precision):
    for checkpoint, fixture, prefix, _mask, e_idx, _keep in iter_cases(data, require_edges=False):
      neighbours = jnp.asarray(e_idx[0])
      etab = jnp.asarray(data[prefix + "etab_energy_padded"][0])
      pad_valid = jnp.ones((etab.shape[0],), dtype=bool)
      native = model_to_etab(jnp.asarray(data[prefix + "seq_native"]))
      random = model_to_etab(jnp.asarray(data[prefix + "seq_random"]))
      got_native = np.asarray(potts_energy(etab, neighbours, pad_valid, native))
      got_random = np.asarray(potts_energy(etab, neighbours, pad_valid, random))
      ref_native = np.asarray(data[prefix + "energy_native"])
      ref_random = np.asarray(data[prefix + "energy_random"])
      label = f"potts_energy {precision} {checkpoint} {fixture}"
      got_random = got_random.reshape(ref_random.shape)
      if not numeric:
        assert_shape_dtype(got_native, ref_native, label + " native")
        assert_shape_dtype(got_random, ref_random, label + " random")
      elif precision == "f32":
        abs_etab = jnp.abs(etab)
        scale_native = np.asarray(potts_energy(abs_etab, neighbours, pad_valid, native))
        scale_random = np.asarray(potts_energy(abs_etab, neighbours, pad_valid, random))
        assert_energy_f32(got_native, ref_native, scale_native, label + " native")
        assert_energy_f32(
          got_random,
          ref_random,
          scale_random.reshape(ref_random.shape),
          label + " random",
        )
      else:
        assert_close(got_native, ref_native, rtol=1e-10, atol=0.0, label=label + " native")
        assert_close(got_random, ref_random, rtol=1e-10, atol=0.0, label=label + " random")
      compared += 1
  return compared


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle: object) -> None:
  """Native and random energies match the sealed dtype and shape."""
  for precision in ("f64", "f32"):
    data = open_dump(oracle, precision)
    try:
      assert _run(data, precision, numeric=False) > 0
    finally:
      data.close()


@pytest.mark.tier_2
def test_tier_2_f64(oracle: object) -> None:
  """f64 energies match at rtol=1e-10 under x64."""
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
def test_tier_3_f32_conditioned(oracle: object) -> None:
  """f32 energies satisfy ``|err| <= 1e-5 * sum|terms|``."""
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
def test_tier_5_trace_budget(oracle: object, max_traces: int) -> None:
  """One static sequence shape traces at most ``max_traces`` times."""
  import chex

  data = open_dump(oracle, "f32")
  try:
    _checkpoint, _fixture, prefix, _mask, e_idx, _keep = next(iter_cases(data, require_edges=False))
    etab = jnp.asarray(data[prefix + "etab_energy_padded"][0])
    neighbours = jnp.asarray(e_idx[0])
    seq = model_to_etab(jnp.asarray(data[prefix + "seq_native"]))
  finally:
    data.close()
  pad_valid = jnp.ones((etab.shape[0],), dtype=bool)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    table: jax.Array,
    idx: jax.Array,
    valid: jax.Array,
    sequence: jax.Array,
  ) -> jax.Array:
    return potts_energy(table, idx, valid, sequence)

  compiled = jax.jit(kernel)
  compiled(etab, neighbours, pad_valid, seq)
  compiled(etab, neighbours, pad_valid, seq)
