"""Tests for AutoregressiveConfig.intent_loop (the wave axis on xtrax.intent.loop).

xtrax.intent.loop runs one ``while_loop`` when the decode is not differentiated and
``lax.scan`` when JAX differentiates it. These tests pin that contract against the scan
reference (``use_while_loop=False``): identical tokens, matching logits and gradients, a
``while`` and no ``scan`` in the undifferentiated jaxpr, and the reverse under ``grad``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.tiling import CarryShape

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode import autoregressive as ar_module
from aminx.inference.decode.autoregressive import AutoregressiveDecode
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.tiling.iterator import JaxScanIterator, VmapIterator
from aminx.tiling.strategy import Vmap

pytest.importorskip("xtrax.intent")


def _fixture(num_residues: int = 8, seed: int = 7) -> dict[str, Any]:
  rng = np.random.default_rng(seed)
  model = eqx.tree_inference(
    Aminx(
      node_features=64,
      edge_features=64,
      hidden_features=64,
      num_encoder_layers=2,
      num_decoder_layers=2,
      k_neighbors=5,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(seed),
    ),
    value=True,
  )
  bundle, config = build_inference_bundle(
    coords=jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32)),
    mask=jnp.ones((num_residues,), dtype=jnp.float32),
    residue_index=jnp.arange(num_residues, dtype=jnp.int32),
    chain_index=jnp.zeros((num_residues,), dtype=jnp.int32),
    state_weights=jnp.ones(1),
    sequence=None,
    mode="sample_autoregressive",
  )
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(1), config)
  stage_set = make_stage_set(
    strategy="arithmetic_mean",
    state_weights=bundle.conditioning.state_weights,
  )
  return {"model": model, "bundle": bundle, "config": config, "enc": enc, "stage_set": stage_set}


def _decoder(model: Aminx, num_residues: int, *, intent_loop: bool, incremental: str) -> AutoregressiveDecode:
  return AutoregressiveDecode(
    model=model,
    state_iterator=VmapIterator(),
    wave_iterator=JaxScanIterator(),
    wave_carry=CarryShape(name="sequence", shape=(num_residues,), dtype=jnp.int32),
    use_while_loop=False,
    incremental=incremental,
    intent_loop=intent_loop,
  )


def _primitive_names(jaxpr: Any) -> set[str]:
  names: set[str] = set()
  for eqn in jaxpr.eqns:
    names.add(eqn.primitive.name)
    for sub in jax.extend.core.jaxprs_in_params(eqn.params):
      names |= _primitive_names(sub)
  return names


@pytest.mark.parametrize("incremental", ["off", "force", "auto"])
def test_intent_loop_matches_scan(incremental: str) -> None:
  """intent_loop=True gives the scan path's tokens and logits on every incremental mode."""
  fx = _fixture()
  num_residues = int(fx["bundle"].geometry.residue_index.shape[0])
  key = jax.random.PRNGKey(123)
  outs = {}
  for intent_loop in (False, True):
    decode = _decoder(fx["model"], num_residues, intent_loop=intent_loop, incremental=incremental)
    outs[intent_loop] = decode(
      key=key, enc=fx["enc"], bundle=fx["bundle"], config=fx["config"], stage_set=fx["stage_set"]
    )
  np.testing.assert_array_equal(np.asarray(outs[False].sequence), np.asarray(outs[True].sequence))
  np.testing.assert_allclose(
    np.asarray(outs[False].logits), np.asarray(outs[True].logits), rtol=1e-5, atol=1e-5
  )


def _loss_fn(fx: dict[str, Any], *, intent_loop: bool, incremental: str = "off"):
  num_residues = int(fx["bundle"].geometry.residue_index.shape[0])
  params, static = eqx.partition(fx["model"], eqx.is_inexact_array)

  def loss(p: Any) -> jax.Array:
    decode = _decoder(eqx.combine(p, static), num_residues, intent_loop=intent_loop, incremental=incremental)
    out = decode(
      key=jax.random.PRNGKey(5),
      enc=fx["enc"],
      bundle=fx["bundle"],
      config=fx["config"],
      stage_set=fx["stage_set"],
    )
    return jnp.sum(out.logits)

  return loss, params


@pytest.mark.parametrize(
  ("incremental", "jit"),
  [("off", False), ("auto", True)],
)
def test_intent_loop_grad_matches_scan(incremental: str, jit: bool) -> None:
  """Gradients through the intent loop equal the scan reference's.

  ``auto`` under ``filter_jit`` puts the wave loop in both branches of a traced cond whose
  incremental branch reads in-trace int index arrays: the case xtrax PR #229 fixed.
  """
  fx = _fixture()
  wrap = eqx.filter_jit if jit else (lambda f: f)
  loss_scan, params = _loss_fn(fx, intent_loop=False, incremental=incremental)
  loss_intent, _ = _loss_fn(fx, intent_loop=True, incremental=incremental)
  g_scan = wrap(jax.grad(loss_scan))(params)
  g_intent = wrap(jax.grad(loss_intent))(params)
  leaves_scan = jax.tree_util.tree_leaves(g_scan)
  leaves_intent = jax.tree_util.tree_leaves(g_intent)
  assert len(leaves_scan) == len(leaves_intent) > 0
  assert any(float(jnp.max(jnp.abs(g))) > 0 for g in leaves_scan), "control: scan gradient is all zero"
  for a, b in zip(leaves_scan, leaves_intent, strict=True):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_intent_loop_jaxprs_while_primal_scan_grad() -> None:
  """Undifferentiated: while and no scan. Under grad: scan and no while."""
  fx = _fixture()
  num_residues = int(fx["bundle"].geometry.residue_index.shape[0])

  def primal_names_for(intent_loop: bool) -> set[str]:
    decode = _decoder(fx["model"], num_residues, intent_loop=intent_loop, incremental="off")
    primal = jax.make_jaxpr(
      lambda k: decode(key=k, enc=fx["enc"], bundle=fx["bundle"], config=fx["config"], stage_set=fx["stage_set"]).logits
    )(jax.random.PRNGKey(0))
    return _primitive_names(primal.jaxpr)

  # Control: the walker must see the scan reference's nested scan, or "no scan" proves nothing.
  assert "scan" in primal_names_for(intent_loop=False)
  primal_names = primal_names_for(intent_loop=True)
  assert "while" in primal_names
  assert "scan" not in primal_names

  loss, params = _loss_fn(fx, intent_loop=True)
  grad_names = _primitive_names(jax.make_jaxpr(jax.grad(loss))(params).jaxpr)
  assert "scan" in grad_names
  assert "while" not in grad_names


def test_factory_wires_intent_loop() -> None:
  """make_decode_fn passes AutoregressiveConfig.intent_loop through; the default is off."""
  fx = _fixture()
  default = make_decode_fn(model=fx["model"], mode=AutoregressiveMode(), strategy=Vmap())
  assert default.intent_loop is False
  wired = make_decode_fn(
    model=fx["model"],
    mode=AutoregressiveMode(),
    strategy=Vmap(),
    autoregressive_config=AutoregressiveConfig(intent_loop=True),
  )
  assert wired.intent_loop is True


def test_intent_loop_without_xtrax_intent_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  """Requesting intent_loop when xtrax.intent is unavailable raises ImportError."""
  fx = _fixture()
  monkeypatch.setattr(ar_module, "_intent_loop", None)
  with pytest.raises(ImportError, match="xtrax.intent"):
    _decoder(fx["model"], 8, intent_loop=True, incremental="off")
