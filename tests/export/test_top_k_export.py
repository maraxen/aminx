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

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aminx
from aminx.model.features import top_k

K = 8
SHAPE = (16, 32)


def _ir(fn, aval) -> str:
    return jax.export.export(jax.jit(fn))(aval).mlir_module()


def _sort_params(fn) -> list[dict]:
  """Params of every ``sort`` primitive in ``fn``'s jaxpr, sub-jaxprs included.

  Recursing is mandatory rather than defensive: ``jnp.argsort`` emits its
  ``sort`` inside a nested ``jit`` sub-jaxpr, never at the top level, so a
  top-level-only walk finds nothing at all and reports a confident all-clear.
  """
  aval = jax.ShapeDtypeStruct(SHAPE, jnp.float32)
  found: list[dict] = []

  def walk(jaxpr) -> None:
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "sort":
        found.append(dict(eqn.params))
      for value in eqn.params.values():
        for candidate in value if isinstance(value, (list, tuple)) else (value,):
          inner = getattr(candidate, "jaxpr", candidate)
          if hasattr(inner, "eqns"):
            walk(inner)

  walk(jax.make_jaxpr(fn)(aval).jaxpr)
  return found


def _lexicographic_oracle(x, k: int):
  """Intended semantics, computed independently in numpy.

  ``np.argsort(-x, kind="stable")`` orders ties by original index, which is the
  same total order as sorting on the pair ``(-x, index)``. Deliberately not
  written in JAX: an oracle sharing the implementation's sort would agree with
  it for the wrong reason.
  """
  arr = np.asarray(x)
  order = np.argsort(-arr, axis=-1, kind="stable")[..., :k]
  return np.take_along_axis(arr, order, axis=-1), order.astype(np.int32)


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
    """An all-equal row is nothing but ties, and the answer is still forced.

    Every comparison in this row is a tie, so a sort that delegated tie order to
    the backend would be free to return any permutation, while ``jax.lax.top_k``
    must return ``0..K-1``. Folding the index into the sort key is what makes
    ``0..K-1`` the only admissible answer here rather than the likely one.
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


class TestTieBreakIsNotDelegatedToTheBackend:
  """Guards the property that has to survive lowering to a foreign sort.

  Nothing in ``TestTopKMatchesLax`` can establish this. Those tests compare
  eager JAX against eager JAX, so a tie-break difference that appears only once
  IREE supplies the sort is invisible to every one of them -- and invisible to
  float parity checking too, since the divergence lands in the integer indices
  while the gathered values stay bit-identical. The claim therefore has to be
  made structurally, about the ``sort`` that is actually emitted.
  """

  def test_index_is_part_of_the_sort_key(self) -> None:
    """``num_keys >= 2`` means no two entries can compare equal.

    With the index as a sort key the ordering is a strict total order, so every
    correct sort must return the same permutation and stability stops mattering.
    """
    params = _sort_params(lambda x: top_k(x, K))
    assert params, (
      "no sort primitive found in top_k's jaxpr -- either the implementation "
      "stopped sorting, or _sort_params stopped recursing into sub-jaxprs."
    )
    for p in params:
      assert p["num_keys"] >= 2, (
        f"top_k emitted a sort with num_keys={p['num_keys']}, so its tie order "
        "is whatever the backend's stability happens to give. IREE does not "
        "honour stable-sort tie order (measured 260911: 45 of 64 positions "
        "disagreed with XLA), and float parity cannot see the difference. Fold "
        "the index into the sort key; see top_k's docstring."
      )

  def test_the_argsort_spelling_would_delegate(self) -> None:
    """Control: proves the assertion above discriminates.

    This is the implementation this test exists to rule out -- the one that
    shipped in PR #155 -- and it must fail the ``num_keys >= 2`` check.
    """
    params = _sort_params(lambda x: jnp.argsort(-x, axis=-1, stable=True)[..., :K])
    assert [p["num_keys"] for p in params] == [1]

  @pytest.mark.parametrize("seed", [0, 7, 11])
  def test_matches_an_independent_numpy_oracle(self, seed: int) -> None:
    """Pins the intended answer without going through JAX's sort at all."""
    x = jax.random.normal(jax.random.PRNGKey(seed), SHAPE, dtype=jnp.float32)
    x = jnp.round(x * 2.0) / 2.0
    vals, idx = top_k(x, K)
    ref_vals, ref_idx = _lexicographic_oracle(x, K)
    assert np.array_equal(np.asarray(idx), ref_idx)
    assert np.array_equal(np.asarray(vals), ref_vals)

  def test_every_row_of_a_single_valued_input_is_forced(self) -> None:
    """The degenerate case an unstable backend is most free to permute."""
    for fill in (0.0, 1.0, -3.5):
      x = jnp.full(SHAPE, fill, dtype=jnp.float32)
      _, idx = top_k(x, K)
      expected = np.broadcast_to(np.arange(K, dtype=np.int32), (SHAPE[0], K))
      assert np.array_equal(np.asarray(idx), expected), f"fill={fill}"


def _lax_top_k_call_lines(path: Path) -> list[int]:
  """Lines in ``path`` calling ``jax.lax.top_k`` / ``lax.top_k``.

  Parsed rather than grepped: ``features.top_k``'s docstring discusses
  ``jax.lax.top_k`` at length, and a text search flags its own explanation.
  """
  lines: list[int] = []
  for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
    if not isinstance(node, ast.Call):
      continue
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "top_k"):
      continue
    owner = func.value
    is_lax = (isinstance(owner, ast.Attribute) and owner.attr == "lax") or (
      isinstance(owner, ast.Name) and owner.id == "lax"
    )
    if is_lax:
      lines.append(node.lineno)
  return lines


class TestNoRemainingLaxTopKCallSites:
  """``jax.lax.top_k`` must not be called anywhere in the package.

  This guard exists because the narrower one above was not enough. PR #155
  routed around ``jax.lax.top_k`` at a single call site and described that site
  as "the package's only kNN selection site" -- but two more were live in
  ``model/ligand_features.py`` the whole time, so the ligand path stayed
  uncompilable while every export test reported green. A whole-tree check is
  blunt, but it is the only one that covers call sites nobody thought to test.
  """

  def test_model_package_calls_only_aminx_top_k(self) -> None:
    """``aminx.model`` is what gets traced, so it is what must stay clean."""
    root = Path(aminx.__file__).parent
    offenders = [
      f"{path.relative_to(root)}:{line}"
      for path in sorted((root / "model").rglob("*.py"))
      for line in _lax_top_k_call_lines(path)
    ]
    assert not offenders, (
      "jax.lax.top_k is called at: "
      + ", ".join(offenders)
      + ". It lowers to a stablehlo.composite IREE marks illegal, so any model "
      "reaching these lines cannot be exported, and its tie order would come "
      "from backend sort stability. Use aminx.model.features.top_k instead."
    )

  def test_host_side_call_sites_are_the_known_set(self) -> None:
    """Pins the call sites deliberately left alone, so new ones surface.

    These two are eager host code, not traced into an exported artifact --
    ``schedule_selector`` converts straight to numpy and loops in Python, and
    ``logit_aggregation`` aggregates probabilities after inference. Neither is
    reachable by ``iree-compile``, so neither is fixed here. They are listed
    rather than ignored: if the set changes, something moved onto or off the
    export path and this test should be re-read, not just re-baselined.
    """
    root = Path(aminx.__file__).parent
    found = {
      f"{path.relative_to(root)}"
      for path in sorted(root.rglob("*.py"))
      if _lax_top_k_call_lines(path)
    }
    assert found == {
      "host/logit_aggregation.py",
      "inference/schedule_selector.py",
    }, f"host-side jax.lax.top_k call sites changed: {sorted(found)}"

  def test_the_guard_detects_a_call(self, tmp_path: Path) -> None:
    """Control: proves the AST walk finds what it claims to."""
    sample = tmp_path / "sample.py"
    sample.write_text(
      '"""A docstring mentioning jax.lax.top_k must not count."""\n'
      "import jax\n"
      "def f(x):\n"
      "  return jax.lax.top_k(x, 4)\n",
      encoding="utf-8",
    )
    assert _lax_top_k_call_lines(sample) == [4]
