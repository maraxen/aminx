"""Compiled-graph (HLO) smell checks for JAX kernels and models.

Addresses backlog-5596:
Inspects XLA HLO compiler IR via `export_hlo` to verify:
1. Absence of host Python callback leaks (e.g. `xla_ffi_python_cpu_callback`).
2. Absence of single-device distributed collectives (`all-reduce`, `all-to-all`).
3. Absence of dynamic allocations and unvetted dynamic shape operations.
4. Clean static compilation of core Potts and model kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.model.mpnn import Aminx
from aminx.potts.trw import DifferentiableTRW, project_to_zero_sum_gauge, tree_marginals_to_potts
from aminx.profiling.hlo_tools import (
  assert_no_hlo_smells,
  check_hlo_smells,
  export_hlo,
  parse_hlo_op_histogram,
)


def _make_minimal_mpnn() -> Aminx:
  """Construct a lightweight Aminx instance for fast lowering tests."""
  return Aminx(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    key=jax.random.PRNGKey(42),
  )


def test_hlo_op_histogram_basic() -> None:
  """Opcode histogram counts elementary XLA operations."""

  def simple_kernel(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.dot(x, y) + 1.0

  x = jnp.ones((4, 4), dtype=jnp.float32)
  y = jnp.ones((4, 4), dtype=jnp.float32)
  hlo = export_hlo(simple_kernel, x, y)
  hist = parse_hlo_op_histogram(hlo)

  assert isinstance(hist, dict)
  assert len(hist) > 0
  # Basic math involves dot/add/multiply or fusion containing them
  assert any(op in hist for op in ("dot", "add", "fusion"))


def test_potts_gauge_projection_no_smells() -> None:
  """Potts zero-sum gauge projection lowers with zero HLO smells."""
  h = jnp.zeros((6, 21), dtype=jnp.float32)
  J = jnp.zeros((6, 6, 21, 21), dtype=jnp.float32)
  hlo = export_hlo(project_to_zero_sum_gauge, h, J)

  assert_no_hlo_smells(hlo, disallow_unvetted_custom_calls=True)
  hist = parse_hlo_op_histogram(hlo)
  assert "custom-call" not in hist


def test_tree_marginals_to_potts_no_smells() -> None:
  """Closed-form tree marginals inverse lowers to pure static arithmetic."""
  n_nodes = 5
  n_states = 4
  node_m = jnp.ones((n_nodes, n_states), dtype=jnp.float32) / float(n_states)
  edge_m = jnp.ones((n_nodes - 1, n_states, n_states), dtype=jnp.float32) / float(n_states**2)
  edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=jnp.int32)

  hlo = export_hlo(tree_marginals_to_potts, node_m, edge_m, edges)
  assert_no_hlo_smells(hlo, disallow_unvetted_custom_calls=True)

  hist = parse_hlo_op_histogram(hlo)
  assert "custom-call" not in hist
  assert "all-reduce" not in hist


def test_aminx_forward_pass_no_smells() -> None:
  """Aminx model forward pass lowers without custom calls or collectives."""
  model = _make_minimal_mpnn()
  L = 8
  X = jnp.zeros((L, 4, 3), dtype=jnp.float32)
  mask = jnp.ones((L,), dtype=jnp.float32)
  residue_idx = jnp.arange(L, dtype=jnp.int32)
  chain_idx = jnp.zeros((L,), dtype=jnp.int32)

  def forward(X, mask, residue_idx, chain_idx):
    return model(X, mask, residue_idx, chain_idx)

  hlo = export_hlo(forward, X, mask, residue_idx, chain_idx)
  assert_no_hlo_smells(hlo, disallow_unvetted_custom_calls=True)

  hist = parse_hlo_op_histogram(hlo)
  assert "custom-call" not in hist
  assert "all-reduce" not in hist


def test_trw_lapack_custom_call_permitted() -> None:
  """DifferentiableTRW allows LAPACK SVD FFI while prohibiting Python callbacks."""
  trw = DifferentiableTRW(q=4, trw_iters=3)
  W = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
  J = jnp.zeros((3, 3, 4, 4), dtype=jnp.float32)
  h = jnp.zeros((3, 4), dtype=jnp.float32)

  def forward(W, J, h):
    return trw(W, J, h)

  hlo = export_hlo(forward, W, J, h)

  # Disallowing python callbacks and collectives should pass
  assert_no_hlo_smells(
    hlo,
    allow_lapack_ffi=True,
    disallow_python_callbacks=True,
    disallow_collectives=True,
    disallow_dynamic_shapes=True,
  )

  # Verify no Python callbacks exist
  smells = check_hlo_smells(hlo, disallow_python_callbacks=True)
  assert not any("callback" in s.lower() for s in smells)


def test_smell_check_catches_python_callbacks() -> None:
  """Smell checker flags pure_callback/io_callback host leaks."""

  def bad_kernel(x: jax.Array) -> jax.Array:
    def host_fn(val):
      return val * 2.0

    return jax.pure_callback(host_fn, x, x)

  x = jnp.ones((4,), dtype=jnp.float32)
  hlo = export_hlo(bad_kernel, x)

  smells = check_hlo_smells(hlo, disallow_python_callbacks=True)
  assert len(smells) > 0
  assert any("Host Python callback detected" in s for s in smells)

  with pytest.raises(AssertionError, match="Host Python callback detected"):
    assert_no_hlo_smells(hlo, disallow_python_callbacks=True)


def test_smell_check_catches_synthetic_collectives() -> None:
  """Smell checker flags distributed collective instructions in HLO text."""
  synthetic_hlo = """
  HloModule collective_module
  ENTRY main {
    %arg0 = f32[4] parameter(0)
    %all_red = f32[4] all-reduce(%arg0), replica_groups={}
    ROOT %result = f32[4] add(%arg0, %all_red)
  }
  """
  smells = check_hlo_smells(synthetic_hlo, disallow_collectives=True)
  assert len(smells) == 1
  assert "Distributed collective instruction detected: 'all-reduce'" in smells[0]

  with pytest.raises(AssertionError, match="Distributed collective instruction"):
    assert_no_hlo_smells(synthetic_hlo, disallow_collectives=True)


def test_smell_check_catches_synthetic_dynamic_shapes() -> None:
  """Smell checker flags dynamic shape manipulation in HLO text."""
  synthetic_hlo = """
  HloModule dynamic_module
  ENTRY main {
    %arg0 = f32[10] parameter(0)
    %dim = s32[] parameter(1)
    %dyn_shape = f32[<=10] set-dimension-size(%arg0, %dim), dimensions={0}
    ROOT %result = f32[<=10] dynamic-reshape(%dyn_shape)
  }
  """
  smells = check_hlo_smells(synthetic_hlo, disallow_dynamic_shapes=True)
  assert len(smells) >= 2
  assert any("set-dimension-size" in s for s in smells)
  assert any("dynamic-reshape" in s for s in smells)
  assert any("Bounded dynamic shape" in s for s in smells)

  with pytest.raises(AssertionError, match="Dynamic shape instruction"):
    assert_no_hlo_smells(synthetic_hlo, disallow_dynamic_shapes=True)


def test_unvetted_custom_calls_flagged() -> None:
  """When disallow_unvetted_custom_calls is active, unvetted custom calls are flagged."""
  synthetic_hlo = """
  HloModule custom_module
  ENTRY main {
    %arg0 = f32[4] parameter(0)
    ROOT %out = f32[4] custom-call(%arg0), custom_call_target="unknown_foreign_kernel"
  }
  """
  smells = check_hlo_smells(synthetic_hlo, disallow_unvetted_custom_calls=True)
  assert len(smells) == 1
  assert "Unvetted custom call target: 'unknown_foreign_kernel'" in smells[0]


def test_tuple_return_ops_and_non_f32_dynamic_shapes() -> None:
  """Verify tuple-typed HLO instructions, non-f32 bounded dynamic shapes, and static kwargs in export_hlo."""
  synthetic_hlo = """
  HloModule tuple_and_bf16_module
  ENTRY main {
    %arg0 = bf16[<=8, 4] parameter(0)
    %arg1 = f64[<=4] parameter(1)
    ROOT %tup = (bf16[<=8, 4], f64[<=4]) all-reduce(%arg0, %arg1), replica_groups={}
  }
  """
  hist = parse_hlo_op_histogram(synthetic_hlo)
  assert hist.get("all-reduce") == 1

  smells = check_hlo_smells(synthetic_hlo, disallow_collectives=True, disallow_dynamic_shapes=True)
  assert any("all-reduce" in s for s in smells)
  bounded = [s for s in smells if "Bounded dynamic shape" in s]
  assert len(bounded) == 3

  # Verify export_hlo handles static boolean kwargs via functools.partial
  n_nodes, n_states = 4, 3
  node_m = jnp.ones((n_nodes, n_states), dtype=jnp.float32) / float(n_states)
  edge_m = jnp.ones((n_nodes - 1, n_states, n_states), dtype=jnp.float32) / float(n_states**2)
  edges = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
  hlo_raw = export_hlo(
    tree_marginals_to_potts, node_m, edge_m, edges, zero_sum_gauge=False, dense_output=False
  )
  assert_no_hlo_smells(hlo_raw, disallow_unvetted_custom_calls=True)
