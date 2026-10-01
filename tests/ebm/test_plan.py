"""Tests for ``aminx.ebm.plan``: xtrax tiling composition wiring (backlog node E4).

Covers the axis-dispatch wiring (``plan_axis``/``dispatch_axis``, one case each
for the Vmap and SafeMap branches of ``BatchPlanner``'s own
cardinality-vs-batch-size rule) and the two ``EnergyFusionFn`` primitives
(``difference_fuse``, ``mean_fuse``) on toy energy arrays. End-to-end coverage
of the three dispatch entry points built on top of this module lives in
``tests/ebm/test_dispatch.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from xtrax.tiling.strategy import Bucket, ChunkedMap, Vmap

from aminx.ebm.plan import (
  EBMAxisNames,
  difference_fuse,
  dispatch_axis,
  mean_fuse,
  plan_axis,
)


class TestPlanAxis:
  def test_small_cardinality_selects_vmap(self) -> None:
    """cardinality <= default_batch_size -> Vmap (BatchPlanner Rule 3)."""
    decision = plan_axis(EBMAxisNames.N_DECOYS, cardinality=4, default_batch_size=8)
    assert isinstance(decision.strategy, Vmap)

  def test_large_divisible_cardinality_selects_safemap(self) -> None:
    """cardinality > default_batch_size and divisible -> SafeMap (BatchPlanner Rule 4)."""
    decision = plan_axis(EBMAxisNames.N_DECOYS, cardinality=8, default_batch_size=4)
    assert isinstance(decision.strategy, ChunkedMap)
    assert decision.strategy.batch_size == 4

  def test_unknown_axis_name_raises(self) -> None:
    with pytest.raises(KeyError):
      plan_axis("not_a_real_axis", cardinality=1)

  def test_default_batch_size_override_changes_decision(self) -> None:
    """Overriding default_batch_size actually changes Vmap vs SafeMap selection."""
    vmap_decision = plan_axis(EBMAxisNames.N_MUTANTS, cardinality=4, default_batch_size=8)
    safemap_decision = plan_axis(EBMAxisNames.N_MUTANTS, cardinality=4, default_batch_size=2)
    assert isinstance(vmap_decision.strategy, Vmap)
    assert isinstance(safemap_decision.strategy, ChunkedMap)


class TestDispatchAxis:
  def test_vmap_dispatch_matches_direct_computation(self) -> None:
    xs = jnp.arange(4.0)
    decision = plan_axis(EBMAxisNames.N_STATES, cardinality=4, default_batch_size=8)
    assert isinstance(decision.strategy, Vmap)

    result = dispatch_axis(decision.strategy, lambda x: x * x, xs)
    assert jnp.allclose(result, xs * xs)

  def test_safemap_dispatch_matches_direct_computation(self) -> None:
    xs = jnp.arange(8.0)
    decision = plan_axis(EBMAxisNames.N_MUTANTS, cardinality=8, default_batch_size=4)
    assert isinstance(decision.strategy, ChunkedMap)

    result = dispatch_axis(decision.strategy, lambda x: x * x, xs)
    assert jnp.allclose(result, xs * xs)

  def test_unsupported_strategy_raises_type_error(self) -> None:
    with pytest.raises(TypeError):
      dispatch_axis(Bucket(boundaries=(4, 8)), lambda x: x, jnp.arange(4.0))


class TestFusionPrimitives:
  def test_difference_fuse_two_state(self) -> None:
    energies = jnp.array([3.0, 1.0])
    assert jnp.allclose(difference_fuse(energies), 2.0)

  def test_difference_fuse_with_bias(self) -> None:
    energies = jnp.array([3.0, 1.0])
    assert jnp.allclose(difference_fuse(energies, bias=jnp.array(0.5)), 2.5)

  def test_mean_fuse(self) -> None:
    energies = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(mean_fuse(energies), 2.5)

  def test_mean_fuse_with_bias(self) -> None:
    energies = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(mean_fuse(energies, bias=jnp.array(1.0)), 3.5)
