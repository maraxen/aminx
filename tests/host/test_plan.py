"""Tests for aminx.host.plan pure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.plan import (
    AxisNames,
    compute_sample_keys,
    extract_batch_sizes,
    resolve_chunk_size,
    resolve_sample_start,
    resolve_target_samples,
)


# Minimal stub for RunSpec subconfigs
@dataclass
class MinimalSamplingConfig:
    """Minimal stub for sampling sub-config."""
    num_samples: int = 100
    temperature: list[float] | tuple[float, ...] = None
    backbone_noise: list[float] | tuple[float, ...] = None

    def __post_init__(self):
        if self.temperature is None:
            object.__setattr__(self, "temperature", [0.1, 1.0])
        if self.backbone_noise is None:
            object.__setattr__(self, "backbone_noise", [0.0])


@dataclass
class MinimalRunSpec:
    """Minimal stub for RunSpec with sub-configs."""
    sampling: MinimalSamplingConfig = None

    def __post_init__(self):
        if self.sampling is None:
            object.__setattr__(self, "sampling", MinimalSamplingConfig())


# Minimal stub for SamplingSpecification for testing
@dataclass
class MinimalSamplingSpec:
    """Minimal stub for testing plan functions without full spec dependencies."""
    batch_size: int = 1
    samples_batch_size: int = 16
    temperature: list[float] | tuple[float, ...] = None
    backbone_noise: list[float] | tuple[float, ...] = None
    num_samples: int = 100
    samples_chunk_size: int | None = None
    run_spec: MinimalRunSpec = None

    def __post_init__(self):
        if self.temperature is None:
            object.__setattr__(self, "temperature", [0.1, 1.0])
        if self.backbone_noise is None:
            object.__setattr__(self, "backbone_noise", [0.0])
        if self.run_spec is None:
            object.__setattr__(
                self,
                "run_spec",
                MinimalRunSpec(
                    sampling=MinimalSamplingConfig(
                        num_samples=self.num_samples,
                        temperature=self.temperature,
                        backbone_noise=self.backbone_noise,
                    )
                ),
            )


class TestResolveTargetSamples:
    """Test suite for resolve_target_samples function."""

    def test_resolve_target_samples_explicit_chunk_count(self):
        """Explicit chunk_sample_count should take priority."""
        spec = MinimalSamplingSpec(num_samples=100)
        result = resolve_target_samples(spec, chunk_sample_count=50)
        assert result == 50

    def test_resolve_target_samples_grid_lineage(self):
        """Grid lineage sample_count should be used when chunk is None."""
        spec = MinimalSamplingSpec(num_samples=100)
        grid_lineage = {"sample_count": 75}
        result = resolve_target_samples(spec, chunk_sample_count=None, grid_lineage=grid_lineage)
        assert result == 75

    def test_resolve_target_samples_spec_fallback(self):
        """spec.num_samples should be used as fallback."""
        spec = MinimalSamplingSpec(num_samples=100)
        result = resolve_target_samples(spec, chunk_sample_count=None, grid_lineage=None)
        assert result == 100

    def test_resolve_target_samples_priority_order(self):
        """Chunk takes priority over grid, grid takes priority over spec."""
        spec = MinimalSamplingSpec(num_samples=100)
        grid_lineage = {"sample_count": 75}
        result1 = resolve_target_samples(spec, chunk_sample_count=50, grid_lineage=grid_lineage)
        assert result1 == 50  # chunk wins

        result2 = resolve_target_samples(spec, chunk_sample_count=None, grid_lineage=grid_lineage)
        assert result2 == 75  # grid wins

        result3 = resolve_target_samples(spec, chunk_sample_count=None, grid_lineage=None)
        assert result3 == 100  # spec wins

    def test_resolve_target_samples_zero_raises(self):
        """Zero or negative sample count should raise ValueError."""
        spec = MinimalSamplingSpec(num_samples=0)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            resolve_target_samples(spec)

    def test_resolve_target_samples_negative_raises(self):
        """Negative sample count should raise ValueError."""
        spec = MinimalSamplingSpec(num_samples=-5)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            resolve_target_samples(spec)

    def test_resolve_target_samples_type_conversion(self):
        """Should convert numpy types to int."""
        spec = MinimalSamplingSpec(num_samples=100)
        result = resolve_target_samples(spec, chunk_sample_count=np.int32(50))
        assert isinstance(result, int)
        assert result == 50


class TestResolveChunkSize:
    """Test suite for resolve_chunk_size function."""

    def test_resolve_chunk_size_explicit_samples_chunk_size(self):
        """Explicit samples_chunk_size should take priority."""
        spec = MinimalSamplingSpec(samples_chunk_size=32)
        result = resolve_chunk_size(spec, total_num_samples=100)
        assert result == 32

    def test_resolve_chunk_size_grid_lineage(self):
        """Grid lineage sample_count should be used when samples_chunk_size is None."""
        spec = MinimalSamplingSpec(samples_chunk_size=None)
        grid_lineage = {"sample_count": 50}
        result = resolve_chunk_size(spec, total_num_samples=100, grid_lineage=grid_lineage)
        assert result == 50

    def test_resolve_chunk_size_fallback_to_total(self):
        """total_num_samples should be default fallback."""
        spec = MinimalSamplingSpec(samples_chunk_size=None)
        result = resolve_chunk_size(spec, total_num_samples=100, grid_lineage=None)
        assert result == 100

    def test_resolve_chunk_size_priority_order(self):
        """samples_chunk_size > grid lineage > total_num_samples."""
        spec = MinimalSamplingSpec(samples_chunk_size=32)
        grid_lineage = {"sample_count": 50}
        result1 = resolve_chunk_size(spec, total_num_samples=100, grid_lineage=grid_lineage)
        assert result1 == 32  # samples_chunk_size wins

        spec2 = MinimalSamplingSpec(samples_chunk_size=None)
        result2 = resolve_chunk_size(spec2, total_num_samples=100, grid_lineage=grid_lineage)
        assert result2 == 50  # grid lineage wins

        spec3 = MinimalSamplingSpec(samples_chunk_size=None)
        result3 = resolve_chunk_size(spec3, total_num_samples=100, grid_lineage=None)
        assert result3 == 100  # total_num_samples wins

    def test_resolve_chunk_size_type_conversion(self):
        """Should convert numpy types to int."""
        spec = MinimalSamplingSpec(samples_chunk_size=np.int32(32))
        result = resolve_chunk_size(spec, total_num_samples=100)
        assert isinstance(result, int)
        assert result == 32


class TestResolveSampleStart:
    """Test suite for resolve_sample_start function."""

    def test_resolve_sample_start_from_grid_lineage(self):
        """Should extract sample_start from grid_lineage."""
        grid_lineage = {"sample_start": 42}
        result = resolve_sample_start(grid_lineage)
        assert result == 42

    def test_resolve_sample_start_default_zero(self):
        """Should default to 0 when grid_lineage is None."""
        result = resolve_sample_start(None)
        assert result == 0

    def test_resolve_sample_start_type_conversion(self):
        """Should convert numpy types to int."""
        grid_lineage = {"sample_start": np.int32(100)}
        result = resolve_sample_start(grid_lineage)
        assert isinstance(result, int)
        assert result == 100

    def test_resolve_sample_start_large_values(self):
        """Should handle large sample start indices."""
        grid_lineage = {"sample_start": 1000000}
        result = resolve_sample_start(grid_lineage)
        assert result == 1000000


class TestComputeSampleKeys:
    """Test suite for compute_sample_keys function."""

    def test_compute_sample_keys_output_shape(self):
        """Output should have shape (target_num_samples,)."""
        base_key = jax.random.PRNGKey(0)
        result = compute_sample_keys(base_key, target_num_samples=10)
        assert result.shape == (10, 2)

    def test_compute_sample_keys_deterministic(self):
        """Same inputs should produce same keys."""
        base_key = jax.random.PRNGKey(42)
        result1 = compute_sample_keys(base_key, target_num_samples=5)
        result2 = compute_sample_keys(base_key, target_num_samples=5)
        np.testing.assert_array_equal(result1, result2)

    def test_compute_sample_keys_different_base_keys_differ(self):
        """Different base keys should produce different results."""
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        result1 = compute_sample_keys(key1, target_num_samples=5)
        result2 = compute_sample_keys(key2, target_num_samples=5)
        assert not np.allclose(result1, result2)

    def test_compute_sample_keys_with_chunk_sample_start(self):
        """Keys should differ based on chunk_sample_start offset."""
        base_key = jax.random.PRNGKey(0)
        result1 = compute_sample_keys(base_key, target_num_samples=5, chunk_sample_start=0)
        result2 = compute_sample_keys(base_key, target_num_samples=5, chunk_sample_start=100)
        # Different offsets should produce different keys
        assert not np.allclose(result1, result2)

    def test_compute_sample_keys_chunk_sample_start_offset(self):
        """Same samples with different offsets should be sequential."""
        base_key = jax.random.PRNGKey(0)
        result_offset_0 = compute_sample_keys(base_key, target_num_samples=2, chunk_sample_start=0)
        result_offset_1 = compute_sample_keys(base_key, target_num_samples=1, chunk_sample_start=1)
        np.testing.assert_array_equal(result_offset_0[1], result_offset_1[0])

    def test_compute_sample_keys_grid_lineage_sample_start(self):
        """grid_lineage_sample_start should provide offset like chunk_sample_start."""
        base_key = jax.random.PRNGKey(0)
        result1 = compute_sample_keys(base_key, target_num_samples=5, grid_lineage_sample_start=0)
        result2 = compute_sample_keys(base_key, target_num_samples=5, grid_lineage_sample_start=10)
        # Different offsets should produce different keys
        assert not np.allclose(result1, result2)

    def test_compute_sample_keys_chunk_sample_start_priority(self):
        """chunk_sample_start should take priority over grid_lineage_sample_start."""
        base_key = jax.random.PRNGKey(0)
        result1 = compute_sample_keys(
            base_key,
            target_num_samples=5,
            chunk_sample_start=10,
            grid_lineage_sample_start=20
        )
        result2 = compute_sample_keys(
            base_key,
            target_num_samples=5,
            chunk_sample_start=10,
            grid_lineage_sample_start=30
        )
        np.testing.assert_array_equal(result1, result2)

    def test_compute_sample_keys_single_sample(self):
        """Should handle single sample correctly."""
        base_key = jax.random.PRNGKey(0)
        result = compute_sample_keys(base_key, target_num_samples=1)
        assert result.shape == (1, 2)

    def test_compute_sample_keys_large_count(self):
        """Should handle large sample counts."""
        base_key = jax.random.PRNGKey(0)
        result = compute_sample_keys(base_key, target_num_samples=1000)
        assert result.shape == (1000, 2)

    def test_compute_sample_keys_jax_key_type(self):
        """Result should be JAX array."""
        base_key = jax.random.PRNGKey(0)
        result = compute_sample_keys(base_key, target_num_samples=5)
        assert isinstance(result, jax.Array)


class TestExtractBatchSizes:
    """Test suite for extract_batch_sizes function."""

    def test_extract_batch_sizes_returns_tuple(self):
        """Should return 4-tuple."""
        plan = MockBatchPlan()
        result = extract_batch_sizes(plan)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_extract_batch_sizes_correct_values(self):
        """Should extract correct batch sizes from plan."""
        plan = MockBatchPlan(
            structures_bs=2,
            samples_bs=16,
            temps_bs=3,
            noises_bs=2
        )
        structures_bs, samples_bs, temps_bs, noises_bs = extract_batch_sizes(plan)
        assert structures_bs == 2
        assert samples_bs == 16
        assert temps_bs == 3
        assert noises_bs == 2

    def test_extract_batch_sizes_all_positive(self):
        """All batch sizes should be positive."""
        plan = MockBatchPlan(
            structures_bs=1,
            samples_bs=1,
            temps_bs=1,
            noises_bs=1
        )
        structures_bs, samples_bs, temps_bs, noises_bs = extract_batch_sizes(plan)
        assert structures_bs > 0
        assert samples_bs > 0
        assert temps_bs > 0
        assert noises_bs > 0

    def test_extract_batch_sizes_type_conversion(self):
        """Should return integers."""
        plan = MockBatchPlan(
            structures_bs=2,
            samples_bs=16,
            temps_bs=3,
            noises_bs=2
        )
        result = extract_batch_sizes(plan)
        for bs in result:
            assert isinstance(bs, int)

    def test_extract_batch_sizes_vmap_and_zero_translate_to_none(self):
        """Vmap strategies and batch_size=0 translate to None for chunked_map."""
        from aminx.tiling.strategy import ChunkedMap, Vmap
        from xtrax.tiling import AxisDecision, AxisSpec, BatchPlan

        decisions = (
            AxisDecision(spec=AxisSpec(name=AxisNames.N_STRUCTURES, cardinality=2, default_batch_size=1), batch_size=2, reasoning="vmap", strategy=Vmap()),
            AxisDecision(spec=AxisSpec(name=AxisNames.N_SAMPLES, cardinality=16, default_batch_size=1), batch_size=0, reasoning="zero", strategy=ChunkedMap(batch_size=0)),
            AxisDecision(spec=AxisSpec(name=AxisNames.N_TEMPERATURES, cardinality=4, default_batch_size=1), batch_size=2, reasoning="chunk", strategy=ChunkedMap(batch_size=2)),
            AxisDecision(spec=AxisSpec(name=AxisNames.N_NOISES, cardinality=1, default_batch_size=1), batch_size=1, reasoning="vmap1", strategy=Vmap()),
        )
        plan = BatchPlan(decisions=decisions)
        assert extract_batch_sizes(plan) == (None, None, 2, None)


class TestAxisNames:
    """Test suite for AxisNames class."""

    def test_axis_names_constants_exist(self):
        """All required axis name constants should exist."""
        assert hasattr(AxisNames, "N_STRUCTURES")
        assert hasattr(AxisNames, "N_SAMPLES")
        assert hasattr(AxisNames, "N_TEMPERATURES")
        assert hasattr(AxisNames, "N_NOISES")

    def test_axis_names_values(self):
        """Axis name constants should have expected values."""
        assert AxisNames.N_STRUCTURES == "n_structures"
        assert AxisNames.N_SAMPLES == "n_samples"
        assert AxisNames.N_TEMPERATURES == "n_temperatures"
        assert AxisNames.N_NOISES == "n_noises"


class TestIntegration:
    """Integration tests combining multiple plan functions."""

    def test_resolve_functions_consistency(self):
        """Resolve functions should work together consistently."""
        spec = MinimalSamplingSpec(
            num_samples=200,
            samples_chunk_size=50
        )
        grid_lineage = {
            "sample_count": 100,
            "sample_start": 50
        }

        target = resolve_target_samples(spec, chunk_sample_count=None, grid_lineage=grid_lineage)
        assert target == 100

        chunk = resolve_chunk_size(spec, total_num_samples=target, grid_lineage=grid_lineage)
        assert chunk == 50

        start = resolve_sample_start(grid_lineage)
        assert start == 50

    def test_sample_keys_with_resolved_params(self):
        """Sample keys should use resolved parameters correctly."""
        base_key = jax.random.PRNGKey(42)
        spec = MinimalSamplingSpec(num_samples=100)
        grid_lineage = {
            "sample_count": 50,
            "sample_start": 25
        }

        target = resolve_target_samples(spec, grid_lineage=grid_lineage)
        start = resolve_sample_start(grid_lineage)

        keys = compute_sample_keys(
            base_key,
            target_num_samples=target,
            grid_lineage_sample_start=start
        )

        assert keys.shape == (50, 2)


class MockBatchPlan:
    """Mock BatchPlan for testing extract_batch_sizes.

    .decisions is a plain list (not .decision_for()): decision_for()/
    extract_batch_sizes() became free functions in host/plan.py that iterate
    plan.decisions (EPIC #1541 T-PLANNER.2 -- xtrax.tiling.BatchPlan has no
    .decision_for() method).
    """

    def __init__(
        self,
        structures_bs: int = 1,
        samples_bs: int = 16,
        temps_bs: int = 2,
        noises_bs: int = 1
    ):
        self.decisions = [
            MockBatchDecision(AxisNames.N_STRUCTURES, structures_bs),
            MockBatchDecision(AxisNames.N_SAMPLES, samples_bs),
            MockBatchDecision(AxisNames.N_TEMPERATURES, temps_bs),
            MockBatchDecision(AxisNames.N_NOISES, noises_bs),
        ]


class MockBatchDecision:
    """Mock AxisDecision for testing.

    .spec.name + .strategy match the real xtrax.tiling.AxisDecision shape
    that decision_for()/_legacy_batch_size() duck-type against. .strategy is
    a plain SimpleNamespace (type name isn't "Vmap") so _legacy_batch_size
    passes .batch_size through unchanged -- these tests assert literal
    batch-size values, not the Vmap-means-0 translation.
    """

    def __init__(self, name: str, batch_size: int):
        self.spec = SimpleNamespace(name=name)
        self.batch_size = batch_size
        self.strategy = SimpleNamespace()
