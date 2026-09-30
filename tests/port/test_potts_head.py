"""potts_head wave: dtype/shape, f64, f32, and one JIT trace.

Weights come from the torch checkpoint under ``AMINX_POTTS_ROOT``. A missing
oracle directory or a missing torch/checkpoint skips. A sha256 mismatch fails.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_close, assert_shape_dtype, iter_cases, open_dump, x64_context

from aminx.families.potts_mpnn.potts_head import PottsHead

pytestmark = [pytest.mark.port_wave("potts_head"), pytest.mark.parity_heavy]

_REF = Path(__file__).resolve().parent / "reference" / "a1_potts"
_TOL = {
  "f64": {"rtol": 1e-10, "atol": 1e-12},
  "f32": {"rtol": 1e-5, "atol": 1e-6},
}


def _checkpoint_paths() -> dict[str, str]:
  manifest = tomllib.loads((_REF / "oracle_manifest.toml").read_text(encoding="utf-8"))
  return {str(row["id"]): str(row["path"]) for row in manifest["checkpoint"]}


def _etab_out(checkpoint_id: str, precision: str) -> tuple[jax.Array, jax.Array]:
  torch = pytest.importorskip("torch")
  root = Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser()
  if not root.is_dir():
    pytest.skip(f"PottsMPNN checkout absent: {root}")
  relative = _checkpoint_paths()[f"{checkpoint_id}_{precision}"]
  path = root / relative
  if not path.is_file():
    pytest.skip(f"checkpoint absent: {path}")
  blob = torch.load(path, map_location="cpu", weights_only=False)
  state = blob["model_state_dict"]
  np_dtype = np.float64 if precision == "f64" else np.float32
  weight = np.asarray(state["etab_out.weight"].detach().cpu().numpy(), dtype=np_dtype)
  bias = np.asarray(state["etab_out.bias"].detach().cpu().numpy(), dtype=np_dtype)
  return jnp.asarray(weight), jnp.asarray(bias)


def _predict(data: np.lib.npyio.NpzFile, precision: str, *, numeric: bool) -> int:
  compared = 0
  weights: dict[str, tuple[jax.Array, jax.Array]] = {}
  with x64_context(precision), jax.default_matmul_precision("highest"):
    for checkpoint, fixture, prefix, mask, e_idx, keep in iter_cases(data, require_edges=True):
      if checkpoint not in weights:
        weights[checkpoint] = _etab_out(checkpoint, precision)
      weight, bias = weights[checkpoint]
      h_e = jnp.asarray(data[prefix + "h_E"][0])
      weight = weight.astype(h_e.dtype)
      bias = bias.astype(h_e.dtype)
      head = PottsHead(int(h_e.shape[-1]), key=jax.random.key(0)).with_weights(weight, bias)
      present = jnp.asarray(mask[0])
      pad_valid = jnp.ones((h_e.shape[0],), dtype=bool)
      pred = np.asarray(head(h_e, jnp.asarray(e_idx[0]), present, pad_valid))
      ref = np.asarray(data[prefix + "etab_raw"][0])
      label = f"potts_head {precision} {checkpoint} {fixture}"
      if numeric:
        assert_close(pred[keep], ref[keep], label=label, **_TOL[precision])
      else:
        assert_shape_dtype(pred, ref, label)
        assert_shape_dtype(pred[keep], ref[keep], label + " kept")
      compared += 1
  return compared


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle: object) -> None:
  """Output dtype and shape match the sealed etab_raw on kept edges."""
  for precision in ("f64", "f32"):
    data = open_dump(oracle, precision)
    try:
      assert _predict(data, precision, numeric=False) > 0
    finally:
      data.close()


@pytest.mark.tier_2
def test_tier_2_f64(oracle: object) -> None:
  """f64 etab_raw matches at rtol=1e-10, atol=1e-12 under x64."""
  data = open_dump(oracle, "f64")
  try:
    assert _predict(data, "f64", numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
def test_tier_3_f32(oracle: object) -> None:
  """f32 etab_raw matches at rtol=1e-5, atol=1e-6."""
  data = open_dump(oracle, "f32")
  try:
    assert _predict(data, "f32", numeric=True) > 0
  finally:
    data.close()


def _first_head_inputs(data: np.lib.npyio.NpzFile) -> tuple[Any, ...]:
  for _checkpoint, _fixture, prefix, mask, e_idx, _keep in iter_cases(data, require_edges=True):
    h_e = jnp.asarray(data[prefix + "h_E"][0])
    head = PottsHead(int(h_e.shape[-1]), key=jax.random.key(0))
    present = jnp.asarray(mask[0])
    pad_valid = jnp.ones((h_e.shape[0],), dtype=bool)
    return head, h_e, jnp.asarray(e_idx[0]), present, pad_valid
  pytest.fail("potts_head dump has no comparable fixture")


@pytest.mark.tier_5
def test_tier_5_trace_budget(oracle: object, max_traces: int) -> None:
  """One static shape traces at most ``max_traces`` times."""
  import chex

  data = open_dump(oracle, "f32")
  try:
    head, h_e, e_idx, present, pad_valid = _first_head_inputs(data)
  finally:
    data.close()
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    module: PottsHead,
    edge: jax.Array,
    neighbours: jax.Array,
    row_present: jax.Array,
    valid: jax.Array,
  ) -> jax.Array:
    return module(edge, neighbours, row_present, valid)

  compiled = eqx.filter_jit(kernel)
  compiled(head, h_e, e_idx, present, pad_valid)
  compiled(head, h_e, e_idx, present, pad_valid)
