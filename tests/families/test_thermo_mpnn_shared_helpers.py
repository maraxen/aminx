"""Bitwise equivalence tests for ThermoMPNN shared ``_linear`` and ``_layer_norm`` helpers.

PR #236 replaced the local ``_linear`` and ``_layer_norm`` helpers in
``aminx.families.thermo_mpnn.model`` with imports from
``aminx.families.potts_mpnn.features``. This test embeds the removed local
implementations verbatim and verifies bitwise equality (``np.testing.assert_array_equal``)
across all input shapes and parameter configurations passed by ``ThermoMPNN``,
``ThermoMPNNDouble``, ``thermo_mpnn_trunk``, and ``potts_edge_features``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from aminx.families.potts_mpnn.features import _layer_norm as _potts_layer_norm
from aminx.families.potts_mpnn.features import _linear as _potts_linear
from aminx.families.thermo_mpnn.model import _layer_norm, _linear


def _removed_linear(layer: eqx.nn.Linear, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
  """Apply ``eqx.nn.Linear`` over the trailing axis of ``x`` without ``vmap``."""
  out = x @ layer.weight.T
  if layer.bias is not None:
    out = out + layer.bias
  return out


def _removed_layer_norm(
  norm: eqx.nn.LayerNorm,
  x: Float[Array, "... H"],
) -> Float[Array, "... H"]:
  """Apply ``eqx.nn.LayerNorm`` over the trailing axis of ``x`` without ``vmap``."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.var(x, axis=-1, keepdims=True)
  out = (x - mean) * jax.lax.rsqrt(variance + norm.eps)
  if norm.weight is not None:
    out = out * norm.weight
  if norm.bias is not None:
    out = out + norm.bias
  return out


def test_thermo_mpnn_reexports_potts_shared_helpers() -> None:
  """Verify ``thermo_mpnn.model`` binds the exact helper objects from ``potts_mpnn.features``."""
  assert _linear is _potts_linear  # noqa: S101
  assert _layer_norm is _potts_layer_norm  # noqa: S101


def test_linear_equivalence_across_thermo_mpnn_call_sites() -> None:
  """Assert bitwise equality between shared and removed ``_linear`` across ThermoMPNN call sites."""
  num_residues = 6
  k_neighbors = 4
  num_mutations = 5

  # (in_features, out_features, use_bias, input_shape)
  call_site_specs: tuple[tuple[int, int, bool, tuple[int, ...]], ...] = (
    # thermo_mpnn_trunk w_out (model.py:206): (L, 128) -> (L, 21)
    (128, 21, True, (num_residues, 128)),
    # ThermoMPNN.__call__ ddg_out (model.py:452): (L, 21, 1) -> (L, 21, 1)
    (1, 1, True, (num_residues, 21, 1)),
    # _siamese_mlp layers (model.py:466-467): (M, 256) -> (M, 128) -> (M, 128) -> (M, 1)
    (256, 128, True, (num_mutations, 256)),
    (128, 128, True, (num_mutations, 128)),
    (128, 1, True, (num_mutations, 128)),
    # ThermoMPNNDouble.__call__ proj_linear (model.py:522, 605): (M, 512) -> (M, 128)
    (512, 128, True, (num_mutations, 512)),
    # potts_edge_features w_pos, w_e (no bias), w_e_proj (features.py:112-116)
    (66, 16, True, (num_residues, k_neighbors, 66)),
    (416, 128, False, (num_residues, k_neighbors, 416)),
    (128, 128, True, (num_residues, k_neighbors, 128)),
  )

  base_key = jax.random.PRNGKey(20261027)
  case_keys = jax.random.split(base_key, len(call_site_specs))

  for case_key, (in_dim, out_dim, use_bias, x_shape) in zip(
    case_keys,
    call_site_specs,
    strict=True,
  ):
    k_layer, k_input = jax.random.split(case_key, 2)
    layer = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=k_layer)
    x = jax.random.normal(k_input, x_shape, dtype=jnp.float32)

    got = _linear(layer, x)
    expected = _removed_linear(layer, x)

    assert got.dtype == jnp.float32  # noqa: S101
    assert got.shape == (*x_shape[:-1], out_dim)  # noqa: S101
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_layer_norm_equivalence_across_thermo_mpnn_call_sites() -> None:
  """Assert bitwise equality between shared and removed ``_layer_norm`` across call sites."""
  num_residues = 6
  k_neighbors = 4
  num_mutations = 5

  k_proj_w, k_proj_b, k_proj_x, k_edge_w, k_edge_b, k_edge_x = jax.random.split(
    jax.random.PRNGKey(20261028),
    6,
  )

  # 1. ThermoMPNNDouble.proj_norm (model.py:521, 605): fresh init (weight=1, bias=0), shape (M, 512)
  proj_norm_fresh = eqx.nn.LayerNorm(512)
  g_embed = jax.random.normal(k_proj_x, (num_mutations, 512), dtype=jnp.float32)
  np.testing.assert_array_equal(
    np.asarray(_layer_norm(proj_norm_fresh, g_embed)),
    np.asarray(_removed_layer_norm(proj_norm_fresh, g_embed)),
  )

  # 2. ThermoMPNNDouble.proj_norm with non-trivial checkpoint weight & bias (convert.py:757-762)
  proj_weight = jax.random.normal(k_proj_w, (512,), dtype=jnp.float32)
  proj_bias = jax.random.normal(k_proj_b, (512,), dtype=jnp.float32)
  proj_norm_loaded = eqx.tree_at(
    lambda n: (n.weight, n.bias),
    proj_norm_fresh,
    (proj_weight, proj_bias),
  )
  np.testing.assert_array_equal(
    np.asarray(_layer_norm(proj_norm_loaded, g_embed)),
    np.asarray(_removed_layer_norm(proj_norm_loaded, g_embed)),
  )

  # 3. potts_edge_features norm_edges (features.py:115): 3D input (L, K, 128) with learned affine
  edge_norm_fresh = eqx.nn.LayerNorm(128)
  edge_weight = jax.random.normal(k_edge_w, (128,), dtype=jnp.float32)
  edge_bias = jax.random.normal(k_edge_b, (128,), dtype=jnp.float32)
  edge_norm_loaded = eqx.tree_at(
    lambda n: (n.weight, n.bias),
    edge_norm_fresh,
    (edge_weight, edge_bias),
  )
  embedded_edges = jax.random.normal(
    k_edge_x,
    (num_residues, k_neighbors, 128),
    dtype=jnp.float32,
  )
  np.testing.assert_array_equal(
    np.asarray(_layer_norm(edge_norm_loaded, embedded_edges)),
    np.asarray(_removed_layer_norm(edge_norm_loaded, embedded_edges)),
  )

  # 4. Optional weight/bias = None branches and non-default eps
  for use_weight, use_bias, eps in (
    (False, True, 1e-5),
    (True, False, 1e-5),
    (False, False, 1e-5),
    (True, True, 1e-6),
  ):
    norm_variant = eqx.nn.LayerNorm(512, eps=eps, use_weight=use_weight, use_bias=use_bias)
    np.testing.assert_array_equal(
      np.asarray(_layer_norm(norm_variant, g_embed)),
      np.asarray(_removed_layer_norm(norm_variant, g_embed)),
    )


def test_thermo_mpnn_double_slot_projection_composition() -> None:
  """Assert bitwise equality on the composed slot projection at ``model.py:605``."""
  num_mutations = 5
  k_w, k_b, k_lin, k_x = jax.random.split(jax.random.PRNGKey(20261029), 4)

  proj_norm = eqx.tree_at(
    lambda n: (n.weight, n.bias),
    eqx.nn.LayerNorm(512),
    (
      jax.random.normal(k_w, (512,), dtype=jnp.float32),
      jax.random.normal(k_b, (512,), dtype=jnp.float32),
    ),
  )
  proj_linear = eqx.nn.Linear(512, 128, key=k_lin)
  g_embed = jax.random.normal(k_x, (num_mutations, 512), dtype=jnp.float32)

  got = jax.nn.relu(_linear(proj_linear, _layer_norm(proj_norm, g_embed)))
  expected = jax.nn.relu(_removed_linear(proj_linear, _removed_layer_norm(proj_norm, g_embed)))

  assert got.shape == (num_mutations, 128)  # noqa: S101
  assert got.dtype == jnp.float32  # noqa: S101
  np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
