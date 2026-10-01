"""A fabricated Potts arm returns a constant. These checks fail if it still does."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.decode import PottsARDecode
from aminx.families.potts_mpnn.refine import PottsRefine
from aminx.model.decoder import DecoderLayer

_ROOT = Path(__file__).resolve().parents[2]
_H = 4
_V = 21


def _load(name: str):
  path = _ROOT / "scripts" / "parity" / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    msg = f"cannot load {path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _zero_floats(module: eqx.Module) -> eqx.Module:
  def _leaf(leaf: Any) -> Any:
    if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating):
      return jnp.zeros(leaf.shape, leaf.dtype)
    return leaf

  return jax.tree.map(_leaf, module)


def _decoder(bias_index: int | None) -> PottsARDecode:
  layer = DecoderLayer(_H, 3 * _H, _H, dropout_rate=0.0, key=jax.random.PRNGKey(0))
  layer = eqx.tree_at(
    lambda item: (item.message_mlp, item.dense),
    layer,
    (_zero_floats(layer.message_mlp), _zero_floats(layer.dense)),
  )
  embed = eqx.nn.Embedding(_V, _H, key=jax.random.PRNGKey(1))
  readout = eqx.nn.Linear(_H, _V, key=jax.random.PRNGKey(2))
  weight = jnp.zeros_like(readout.weight)
  bias = jnp.zeros((_V,))
  if bias_index is not None:
    bias = jnp.full((_V,), -50.0).at[bias_index].set(50.0)
  readout = eqx.tree_at(lambda item: (item.weight, item.bias), readout, (weight, bias))
  return PottsARDecode(layers=(layer,), w_s_embed=embed, w_out=readout)


def _arrays() -> dict[str, np.ndarray | int]:
  length = 1
  return {
    "h_v": np.zeros((length, _H), dtype=np.float32),
    "h_e": np.zeros((length, 1, _H), dtype=np.float32),
    "e_idx": np.zeros((length, 1), dtype=np.int32),
    "forward": np.zeros((length, 1, 20, 20), dtype=np.float32),
    "present": np.ones((length,), dtype=np.float32),
    "pad_valid": np.ones((length,), dtype=np.bool_),
    "s_true": np.zeros((length,), dtype=np.int32),
    "chain_mask": np.ones((length,), dtype=np.float32),
    "chain_m_pos": np.ones((length,), dtype=np.float32),
    "tie_groups": np.asarray([[0]], dtype=np.int32),
    "tied_beta": np.ones((length,), dtype=np.float32),
    "omit": np.zeros((_V,), dtype=np.float32),
    "bias": np.zeros((_V,), dtype=np.float32),
    "bias_by_res": np.zeros((length, _V), dtype=np.float32),
    "pssm_coef": np.zeros((length,), dtype=np.float32),
    "pssm_bias": np.zeros((length, _V), dtype=np.float32),
    "pssm_log_odds_mask": np.zeros((length, _V), dtype=np.float32),
    "omit_aa_mask": np.zeros((length, _V), dtype=np.float32),
    "length": length,
  }


def test_ar_tokens_depend_on_draws() -> None:
  module = _load("potts_ar_decode")
  decode = _decoder(None)
  arrays = _arrays()
  low = module.decode_tokens(decode, arrays, np.zeros(1), np.asarray([0.01]), temperature=1.0)
  high = module.decode_tokens(decode, arrays, np.zeros(1), np.asarray([0.99]), temperature=1.0)
  assert low != high


def test_ar_tokens_depend_on_checkpoint() -> None:
  module = _load("potts_ar_decode")
  arrays = _arrays()
  draws = np.asarray([0.5])
  first = module.decode_tokens(_decoder(1), arrays, np.zeros(1), draws, temperature=1.0)
  second = module.decode_tokens(_decoder(8), arrays, np.zeros(1), draws, temperature=1.0)
  assert first != second


def test_refine_tokens_depend_on_draws() -> None:
  module = _load("potts_refine")
  decode = _decoder(None)
  refiner = PottsRefine(layers=decode.layers, w_s_embed=decode.w_s_embed, w_out=decode.w_out)
  arrays = _arrays()
  order = np.asarray([0], dtype=np.int32)
  low = module.refine_tokens(
    refiner,
    arrays,
    order,
    np.asarray([[0.01]]),
    mode="potts",
    temperature=1.0,
  )
  high = module.refine_tokens(
    refiner,
    arrays,
    order,
    np.asarray([[0.99]]),
    mode="potts",
    temperature=1.0,
  )
  assert low != high


def test_refine_tokens_depend_on_checkpoint() -> None:
  module = _load("potts_refine")
  arrays = _arrays()
  order = np.asarray([0], dtype=np.int32)
  draws = np.asarray([[0.5]])
  first_dec = _decoder(1)
  second_dec = _decoder(8)
  first = module.refine_tokens(
    PottsRefine(layers=first_dec.layers, w_s_embed=first_dec.w_s_embed, w_out=first_dec.w_out),
    arrays,
    order,
    draws,
    mode="nodes",
    temperature=1.0,
  )
  second = module.refine_tokens(
    PottsRefine(layers=second_dec.layers, w_s_embed=second_dec.w_s_embed, w_out=second_dec.w_out),
    arrays,
    order,
    draws,
    mode="nodes",
    temperature=1.0,
  )
  assert first != second


def test_exact_tokens_depend_on_draws() -> None:
  module = _load("potts_ar_refine_exact")
  decode = _decoder(3)
  refiner = PottsRefine(layers=decode.layers, w_s_embed=decode.w_s_embed, w_out=decode.w_out)
  arrays = _arrays()
  low = module.exact_tokens(
    decode,
    refiner,
    arrays,
    {"randn": np.zeros(1), "uniforms": np.asarray([0.2]), "refine_uniforms": np.asarray([[0.01]])},
  )
  high = module.exact_tokens(
    decode,
    refiner,
    arrays,
    {"randn": np.zeros(1), "uniforms": np.asarray([0.2]), "refine_uniforms": np.asarray([[0.99]])},
  )
  assert low != high


def test_arms_call_the_port_and_do_not_echo() -> None:
  ar = _load("potts_ar_decode")
  refine = _load("potts_refine")
  exact = _load("potts_ar_refine_exact")
  common = _load("potts_graded_common")
  assert "PottsARDecode(" in inspect.getsource(ar._run_arm)
  assert "decode_tokens(" in inspect.getsource(ar._run_arm)
  assert "PottsRefine(" in inspect.getsource(refine._run_arm)
  assert "refine_tokens(" in inspect.getsource(refine._run_arm)
  assert "PottsARDecode(" in inspect.getsource(exact._run_arm)
  assert "PottsRefine(" in inspect.getsource(exact._run_arm)
  assert "exact_tokens(" in inspect.getsource(exact._run_arm)
  parent = inspect.getsource(common.parent)
  assert "oracle_python(args)" in parent
  oracle_list = parent.split("oracle = _launch", 1)[1].split("if oracle.returncode", 1)[0]
  assert "sys.executable" not in oracle_list
  for name in ("potts_ar_decode", "potts_refine", "potts_ar_refine_exact"):
    text = (_ROOT / "scripts" / "parity" / f"{name}.py").read_text(encoding="utf-8")
    assert "aminx_tokens" not in text
    assert "control_match" not in text
    assert "min(match" not in text
    assert "injected_uniform_draws(" in text
    assert "_load_oracle(" in inspect.getsource(_load(name)._oracle_worker)


@pytest.mark.parametrize(
  "name",
  ["potts_ar_decode", "potts_refine", "potts_ar_refine_exact"],
)
def test_job_builder_has_no_fabricated_measurement(name: str) -> None:
  module = _load(name)
  source = inspect.getsource(module._build_job)
  assert "aminx_tokens" not in source
  assert "control_match" not in source
