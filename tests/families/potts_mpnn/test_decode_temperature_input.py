# ruff: noqa: S101
"""PottsARDecode takes temperature as a Python float OR a scalar array (export, #5815).

The browser graph needs temperature as a runtime input. The array path must give the
exact float-path result, including upstream's ``0 -> 1e-6`` floor, and under ``jit`` with
the temperature traced. A control decodes at a different temperature and must differ, so
"identical" is not what any two decodes would show.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature, floor_temperature_array
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.potts_mpnn.sample_host import _encode, prepare_sample
from aminx.run.options import PottsMPNNOptions
from tests.families.potts_mpnn.test_decode_padding_invariance import _features

_L = 24


def _setup(tmp_path: Path) -> tuple[PottsARDecode, tuple, dict]:
  features = _features(tmp_path, _L)
  ready = prepare_sample(features, (("A", "A" * _L),), PottsMPNNOptions(), SimpleNamespace(), l_pad=_L)
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  h_v, h_e, e_idx, _forward, _table = _encode(
    model,
    jnp.asarray(ready.coords),
    jnp.asarray(ready.present),
    jnp.asarray(ready.residue_idx, dtype=jnp.int32),
    jnp.asarray(ready.chain_index, dtype=jnp.int32),
    jnp.asarray(ready.pad_valid),
  )
  rng = np.random.default_rng(3)
  args = (
    h_v, h_e, e_idx, jnp.asarray(ready.present), jnp.asarray(ready.pad_valid), jnp.asarray(ready.s_true),
    jnp.asarray(ready.chain_mask), jnp.asarray(ready.chain_m_pos), jnp.asarray(ready.tie_groups),
    jnp.asarray(ready.tied_beta), jnp.asarray(rng.standard_normal(_L), jnp.float32),
    jnp.asarray(rng.uniform(size=_L), jnp.float32), jnp.asarray(ready.omit), jnp.asarray(ready.bias),
    jnp.asarray(ready.bias_by_res), jnp.asarray(ready.pssm_coef), jnp.asarray(ready.pssm_bias),
    jnp.asarray(ready.pssm_log_odds_mask), jnp.asarray(ready.omit_aa_mask),
  )
  decoder = PottsARDecode(layers=model.mpnn.decoder.layers, w_s_embed=model.mpnn.w_s_embed, w_out=model.mpnn.w_out)
  static = {"pssm_multi": 0.0, "pssm_bias_flag": False, "pssm_log_odds_flag": False}
  return decoder, args, static


def test_floor_temperature_array_matches_float_floor() -> None:
  for value in (0.0, 1e-6, 0.1, 1.0, 2.5):
    got = float(floor_temperature_array(jnp.asarray(value), jnp.float32))
    assert got == float(np.float32(floor_temperature(value)))


@pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0])
def test_array_temperature_matches_float(tmp_path: Path, temperature: float) -> None:
  decoder, args, static = _setup(tmp_path)
  # Both arms jitted: jit-vs-eager f32 differences (debt #2432) are not what this tests.
  ref = jax.jit(lambda: decoder(*args, temperature=temperature, **static))()

  @jax.jit
  def traced(t: jax.Array):  # noqa: ANN202
    return decoder(*args, temperature=t, **static)

  got = traced(jnp.asarray(temperature, jnp.float32))
  np.testing.assert_array_equal(np.asarray(got.sequence), np.asarray(ref.sequence))
  np.testing.assert_array_equal(np.asarray(got.decoding_order), np.asarray(ref.decoding_order))
  np.testing.assert_array_equal(np.asarray(got.h_v_stack), np.asarray(ref.h_v_stack))


def test_temperature_control_fires(tmp_path: Path) -> None:
  """A different temperature changes the draw, so the equality above is informative."""
  decoder, args, static = _setup(tmp_path)
  cold = decoder(*args, temperature=0.1, **static)
  hot = decoder(*args, temperature=jnp.asarray(5.0, jnp.float32), **static)
  assert not np.array_equal(np.asarray(cold.sequence), np.asarray(hot.sequence))
