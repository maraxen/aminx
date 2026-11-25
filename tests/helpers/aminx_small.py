"""Small Aminx fixtures shared by the loop-complexity guards.

``make_ar_model``, ``structure`` and ``bundle`` are the AR complexity test's
``model`` / ``_structure`` / ``_bundle`` definitions. That test keeps its own
copies; this module does not replace them.

``make_guard_model`` is the 64-d, 2-layer, k=5 network Task 0 traced. The
pre-registered A/B thresholds were measured on it, so the STE guard uses it
(the same shape as ``tests/inference/decode/test_ste.py``).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.model.mpnn import Aminx

L_DEFAULT = 24


def make_ar_model() -> Aminx:
  """The AR test's module-scoped model, including the 30x sequence-embedding scale."""
  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=3,
    k_neighbors=8,
    key=jax.random.PRNGKey(0),
  )
  model = eqx.tree_at(lambda mm: mm.w_s_embed.weight, model, model.w_s_embed.weight * 30.0)
  return eqx.tree_inference(model, value=True)


def make_guard_model() -> Aminx:
  """Task 0's traced model: 2 decoder layers, 5 neighbours, dropout off, inference mode."""
  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(0),
  )
  return eqx.tree_inference(model, value=True)


def structure(length: int, n_states: int = 1, seed: int = 1) -> dict[str, jax.Array]:
  """A random-walk backbone (distinct neighbours, no degenerate ties)."""
  key = jax.random.PRNGKey(seed)
  steps = jax.random.normal(key, (n_states, length, 1, 3)) * 2.0
  ca = jnp.cumsum(steps, axis=1)
  offsets = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 4, 3)) * 0.8
  coords = ca + offsets
  return {
    "coords": coords if n_states > 1 else coords[0],
    "mask": jnp.ones((n_states, length)) if n_states > 1 else jnp.ones((length,)),
    "residue_index": jnp.broadcast_to(jnp.arange(length), (n_states, length))
    if n_states > 1
    else jnp.arange(length),
    "chain_index": jnp.zeros((n_states, length) if n_states > 1 else (length,), dtype=jnp.int32),
  }


def bundle(length: int = L_DEFAULT, **kw: Any) -> tuple[Any, Any]:
  """``build_inference_bundle`` around :func:`structure`, mode ``sample_autoregressive``."""
  n_states = kw.pop("n_states", 1)
  coords = structure(length, n_states=n_states)
  return build_inference_bundle(
    coords["coords"],
    coords["mask"],
    coords["residue_index"],
    coords["chain_index"],
    mode="sample_autoregressive",
    **kw,
  )
