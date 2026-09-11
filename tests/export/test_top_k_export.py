"""Guards that ``aminx.model.features.top_k`` stays IREE-compilable.

``jax.lax.top_k`` lowers to a ``stablehlo.composite`` wrapping ``chlo.top_k``,
which IREE's StableHLO importer marks explicitly illegal. Measured 260911 against
``iree-base-compiler 3.11.0rc20260316``, a bare ``jax.lax.top_k`` is rejected on
every ``input_type`` IREE offers (``stablehlo``, ``stablehlo_xla``, ``auto``),
while a control compiles on all three -- so it is a gap in the importer, not a
flag. Because this is the package's only kNN selection site, reverting it to
``jax.lax.top_k`` silently makes the whole model uncompilable again, and the
failure surfaces far away, deep inside ``iree-compile``.

These tests need no IREE toolchain: the composite is visible in the StableHLO
that ``jax.export`` produces, so the guard is a string check on the IR.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.features import top_k

K = 8
SHAPE = (16, 32)


def _ir(fn, aval) -> str:
    return jax.export.export(jax.jit(fn))(aval).mlir_module()


class TestTopKIsExportable:
  """The IR must not carry the composite IREE refuses to legalize."""

  def test_emits_no_stablehlo_composite(self) -> None:
    aval = jax.ShapeDtypeStruct(SHAPE, jnp.float32)
    ir = _ir(lambda x: top_k(x, K), aval)
    assert "stablehlo.composite" not in ir, (
      "aminx.model.features.top_k emitted a stablehlo.composite. IREE marks that "
      "op illegal, so the model no longer compiles. Did this revert to "
      "jax.lax.top_k? See the function's docstring."
    )
    assert "chlo.top_k" not in ir

  def test_lax_top_k_would_emit_one(self) -> None:
    """Control: proves the guard above detects the thing it claims to."""
    aval = jax.ShapeDtypeStruct(SHAPE, jnp.float32)
    ir = _ir(lambda x: jax.lax.top_k(x, K), aval)
    assert "stablehlo.composite" in ir


class TestTopKMatchesLax:
  """Equivalence with the op it replaces, including tie-breaking."""

  @pytest.mark.parametrize("seed", [0, 1, 2])
  def test_matches_on_random_input(self, seed: int) -> None:
    x = jax.random.normal(jax.random.PRNGKey(seed), SHAPE, dtype=jnp.float32)
    vals, idx = top_k(x, K)
    ref_vals, ref_idx = jax.lax.top_k(x, K)
    assert np.array_equal(np.asarray(vals), np.asarray(ref_vals))
    assert np.array_equal(np.asarray(idx), np.asarray(ref_idx))

  def test_breaks_ties_toward_the_lower_index(self) -> None:
    """The reason ``stable=True`` is not cosmetic.

    An all-equal row makes every comparison a tie, so an unstable sort is free to
    return any permutation while ``jax.lax.top_k`` must return ``0..K-1``.
    """
    x = jnp.zeros(SHAPE, dtype=jnp.float32)
    _, idx = top_k(x, K)
    expected = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), (SHAPE[0], K))
    assert np.array_equal(np.asarray(idx), np.asarray(expected))

  def test_matches_lax_on_a_tie_heavy_input(self) -> None:
    """Coarse quantisation forces many exact ties without being degenerate."""
    x = jax.random.normal(jax.random.PRNGKey(7), SHAPE, dtype=jnp.float32)
    x = jnp.round(x * 2.0) / 2.0
    vals, idx = top_k(x, K)
    ref_vals, ref_idx = jax.lax.top_k(x, K)
    assert np.array_equal(np.asarray(vals), np.asarray(ref_vals))
    assert np.array_equal(np.asarray(idx), np.asarray(ref_idx))

  def test_returns_int32_indices(self) -> None:
    x = jax.random.normal(jax.random.PRNGKey(3), SHAPE, dtype=jnp.float32)
    _, idx = top_k(x, K)
    assert idx.dtype == jnp.int32
