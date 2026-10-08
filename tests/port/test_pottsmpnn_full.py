"""pottsmpnn_full wave: etab_forward and teacher-forced log-probs.

Skipped until A2 lands ``aminx.families.potts_mpnn.model.PottsMPNN`` and
``teacher_forced_forward``. That callable takes one fixture's arrays and
returns ``(etab_forward, log_probs)`` with the batch axis removed:

  teacher_forced_forward(
    precision, checkpoint, X, S, mask, residue_idx, chain_encoding,
    E_idx, chain_M, decoding_order,
  ) -> (etab_forward (L,K,20,20), log_probs (L,21))

f64 tolerance is atol=1e-8. f32 uses rtol=1e-4, atol=1e-4 (spec r16). Trace budget is one
trace per static bucket.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_close, assert_shape_dtype, iter_cases, open_dump, x64_context

pytestmark = [pytest.mark.port_wave("pottsmpnn_full"), pytest.mark.parity_heavy]

_INPUTS = (
  "X",
  "S",
  "mask",
  "residue_idx",
  "chain_encoding",
  "E_idx",
  "chain_M",
  "decoding_order",
)


def _forward() -> Callable[..., tuple[Any, Any]]:
  module = pytest.importorskip("aminx.families.potts_mpnn.model")
  if getattr(module, "PottsMPNN", None) is None:
    pytest.skip("PottsMPNN pending A2")
  runner = getattr(module, "teacher_forced_forward", None)
  if runner is None:
    runner = getattr(module.PottsMPNN, "teacher_forced_forward", None)
  if runner is None:
    pytest.skip("PottsMPNN.teacher_forced_forward pending A2")
  return runner


def _call(
  runner: Callable[..., tuple[Any, Any]],
  precision: str,
  checkpoint: str,
  prefix: str,
  data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
  kwargs = {name: jnp.asarray(data[prefix + name][0]) for name in _INPUTS}
  etab, log_probs = runner(precision, checkpoint, **kwargs)
  return np.asarray(etab), np.asarray(log_probs)


def _run(data: np.lib.npyio.NpzFile, precision: str, *, numeric: bool) -> int:
  runner = _forward()
  atol = 1e-8 if precision == "f64" else 1e-4
  rtol = 0.0 if precision == "f64" else 1e-4
  compared = 0
  with x64_context(precision):
    for checkpoint, fixture, prefix, _mask, _e_idx, _keep in iter_cases(data, require_edges=False):
      etab, log_probs = _call(runner, precision, checkpoint, prefix, data)
      ref_etab = np.asarray(data[prefix + "etab_forward"][0])
      ref_logs = np.asarray(data[prefix + "log_probs"][0])
      label = f"pottsmpnn_full {precision} {checkpoint} {fixture}"
      if numeric:
        assert_close(etab, ref_etab, rtol=rtol, atol=atol, label=label + " etab")
        assert_close(log_probs, ref_logs, rtol=rtol, atol=atol, label=label + " log_probs")
      else:
        assert_shape_dtype(etab, ref_etab, label + " etab")
        assert_shape_dtype(log_probs, ref_logs, label + " log_probs")
      compared += 1
  return compared


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle: object) -> None:
  """etab_forward and log_probs match the sealed dtype and shape."""
  _forward()
  for precision in ("f64", "f32"):
    data = open_dump(oracle, precision)
    try:
      assert _run(data, precision, numeric=False) > 0
    finally:
      data.close()


@pytest.mark.tier_2
def test_tier_2_f64(oracle: object) -> None:
  """f64 etab_forward and log-probs match at atol=1e-8 under x64."""
  _forward()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
def test_tier_3_f32(oracle: object) -> None:
  """f32 etab_forward and log-probs match at rtol=1e-4, atol=1e-4 (r16)."""
  _forward()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
def test_tier_5_trace_budget(oracle: object, max_traces: int) -> None:
  """One static bucket traces at most ``max_traces`` times."""
  import chex

  runner = _forward()
  data = open_dump(oracle, "f32")
  try:
    checkpoint, _fixture, prefix, _mask, _e_idx, _keep = next(iter_cases(data, require_edges=False))
    kwargs = {name: jnp.asarray(data[prefix + name][0]) for name in _INPUTS}
  finally:
    data.close()
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(**arrays: jax.Array) -> tuple[jax.Array, jax.Array]:
    return runner("f32", checkpoint, **arrays)

  compiled = jax.jit(kernel)
  compiled(**kwargs)
  compiled(**kwargs)
