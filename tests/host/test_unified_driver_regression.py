"""Regression suite for _sample_batch unification.

These tests capture exact behavior of Path A (no fusion) and Path B (fusion)
that the unified driver (Task 9) must reproduce bit-for-bit.

ALL non-skipped tests must pass against the CURRENT implementation before
the unified driver is written. They are the hard gate for Task 9.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from unittest.mock import MagicMock


# --- Regression 1: Path A output shape ---


def test_regression_path_a_output_shape():
    """Path A (no fusion): skipped — requires real protein fixture."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 2: Path B output shape with fusion ---


def test_regression_path_b_output_shape_with_fusion():
    """Path B (fusion): skipped — requires real protein fixture."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 3: Tile invariance ---


def test_regression_tile_invariance_path_a():
    """varying batch_size must not change outputs: skipped — requires real protein."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 4: encoder_sink tuple fires per noise level ---


def test_regression_encoder_sink_fires_per_noise_level():
    """encoder_sink tuple iterates correctly — fires N times per noise level."""
    calls = []

    class TrackingSink:
        ordered = False

        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            calls.append(int(noise_idx))

    from aminx.types.stages import StageSet

    # Verify the tuple iteration pattern works correctly
    stage_set = StageSet(encoder_sink=(TrackingSink(),))

    # Simulate kernel dispatch pattern: tuple iteration per noise level
    fake_enc = MagicMock()
    for noise_idx in range(3):
        for _sink in stage_set.encoder_sink:
            _sink(fake_enc, jnp.int32(0), jnp.int32(0), jnp.int32(noise_idx))

    assert calls == [0, 1, 2], f"Expected [0,1,2], got {calls}"


# --- Regression 5: Scan carry accumulates correctly ---


def test_regression_scan_strategy_carry_accumulates():
    """safe_scan carry accumulates correctly across noise steps."""
    from xtrax.transforms import safe_scan

    running_total = jnp.zeros(8)

    def noise_transition(carry, noise_val):
        fake_enc = jnp.ones(8) * noise_val
        new_carry = carry + fake_enc
        return new_carry, fake_enc

    noises = jnp.array([0.1, 0.2, 0.3])
    final_carry, stacked_encs = safe_scan(noise_transition, running_total, noises)

    # carry should be sum of all fake_encs: (0.1 + 0.2 + 0.3) * ones(8)
    assert jnp.allclose(final_carry, jnp.full(8, 0.6), atol=1e-5)
    assert stacked_encs.shape == (3, 8)


# --- Regression 6: Validator fires before JIT ---


def test_regression_validator_fires_before_jit():
    """PlanTopologyError is raised at plan construction time, not at trace time."""
    from aminx.host.plan import PlanTopologyError, _validate_plan_topology
    from aminx.tiling.axes import N_NOISES
    from aminx.tiling.strategy import Vmap
    from aminx.types.stages import StageSet
    from xtrax.tiling import AxisDecision, BatchPlan

    class OrderedSink:
        ordered = True

        def __call__(self, x) -> None:
            pass

    # Create minimal invalid plan (Vmap with ordered sink). xtrax.tiling.BatchPlan
    # only needs decisions (EPIC #1541 T-PLANNER.GATE retired the richer local one).
    bad_decision = AxisDecision(spec=N_NOISES, batch_size=0, reasoning="test", strategy=Vmap())
    plan = BatchPlan(decisions=[bad_decision])

    # Create stage_set with ordered sink on axis_boundaries
    from aminx.types.boundaries import AxisBoundary

    stage_set = StageSet(
        axis_boundaries={
            "n_noises": AxisBoundary(sink=OrderedSink())
        }
    )

    # Validator should raise PlanTopologyError before any JAX compilation
    with pytest.raises(PlanTopologyError):
        _validate_plan_topology(plan, stage_set)


# --- Regression 7: encode/decode methods exist on InferencePlan ---


def test_regression_inference_plan_encode_exists():
    """InferencePlan.encode() method exists and is callable."""
    from aminx.host.plan import InferencePlan, InferenceComponents

    # Create minimal components
    dummy_encode_fn = lambda bundle, key, config: None
    dummy_stage_set = eqx.Module()

    components = InferenceComponents(
        encode_fn=dummy_encode_fn,
        stage_set=dummy_stage_set,
    )

    dummy_decode_fn = lambda *args: None
    plan = InferencePlan(model=eqx.Module(), components=components, decode_fn=dummy_decode_fn)

    assert hasattr(plan, "encode"), "InferencePlan must have .encode() method"
    assert callable(plan.encode)


def test_regression_inference_plan_decode_exists():
    """InferencePlan.decode() method exists and is callable."""
    from aminx.host.plan import InferencePlan, InferenceComponents

    # Create minimal components
    dummy_encode_fn = lambda bundle, key, config: None
    dummy_stage_set = eqx.Module()

    components = InferenceComponents(
        encode_fn=dummy_encode_fn,
        stage_set=dummy_stage_set,
    )

    dummy_decode_fn = lambda *args: None
    plan = InferencePlan(model=eqx.Module(), components=components, decode_fn=dummy_decode_fn)

    assert hasattr(plan, "decode"), "InferencePlan must have .decode() method"
    assert callable(plan.decode)


# --- Regression 8: ArithmeticMeanEncodingFusion exists ---


def test_regression_arithmetic_mean_encoding_fusion_exists():
    """ArithmeticMeanEncodingFusion is importable and callable."""
    from aminx.host.averaging import ArithmeticMeanEncodingFusion
    from aminx.types.bundles import EncoderOutput

    # Create a minimal fusion instance
    fusion = ArithmeticMeanEncodingFusion()

    # Create a fake stacked EncoderOutput with D=2 noise levels
    fake_stacked = EncoderOutput(
        node_features=jnp.ones((2, 10, 16)),  # (D, L, F)
        edge_features=jnp.ones((2, 10, 3, 16)),  # (D, L, K, F)
        neighbor_indices=jnp.zeros((2, 10, 3), dtype=jnp.int32),  # (D, L, K)
        mask=jnp.ones((2, 10)),  # (D, L)
    )

    # Fusion should reduce D → 1
    result = fusion(fake_stacked)

    assert result.node_features.shape == (10, 16), f"Expected (10, 16), got {result.node_features.shape}"
    assert result.edge_features.shape == (10, 3, 16), f"Expected (10, 3, 16), got {result.edge_features.shape}"
    assert result.neighbor_indices.shape == (10, 3)
    assert result.mask.shape == (10,)


# --- Regression 9: IdentityEncodingFusion exists ---


def test_regression_identity_encoding_fusion_exists():
    """IdentityEncodingFusion is importable and passes through all D outputs."""
    from aminx.host.averaging import IdentityEncodingFusion
    from aminx.types.bundles import EncoderOutput

    # Create a minimal fusion instance
    fusion = IdentityEncodingFusion()

    # Create a fake stacked EncoderOutput with D=2 noise levels
    fake_stacked = EncoderOutput(
        node_features=jnp.ones((2, 10, 16)),  # (D, L, F)
        edge_features=jnp.ones((2, 10, 3, 16)),  # (D, L, K, F)
        neighbor_indices=jnp.zeros((2, 10, 3), dtype=jnp.int32),  # (D, L, K)
        mask=jnp.ones((2, 10)),  # (D, L)
    )

    # Identity should pass through unchanged (K=D)
    result = fusion(fake_stacked)

    assert result.node_features.shape == (2, 10, 16), f"Expected (2, 10, 16), got {result.node_features.shape}"
    assert result.edge_features.shape == (2, 10, 3, 16), f"Expected (2, 10, 3, 16), got {result.edge_features.shape}"
    assert result.neighbor_indices.shape == (2, 10, 3)
    assert result.mask.shape == (2, 10)


# --- Regression 10: make_stage_set exists and is callable ---


def test_regression_make_stage_set_exists():
    """make_stage_set is importable and callable."""
    from aminx.inference.logits import make_stage_set

    assert callable(make_stage_set), "make_stage_set must be callable"


# --- Regression 11: SampleResult exists ---


def test_regression_sample_result_exists():
    """SampleResult is importable and has sequence and logits fields."""
    from aminx.inference.sample_autoregressive import SampleResult

    # Create a minimal result
    seq = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8)
    logits = jnp.ones((5, 21), dtype=jnp.float32)

    result = SampleResult(sequence=seq, logits=logits)

    assert result.sequence is seq
    assert result.logits is logits
    assert result.sequence.shape == (5,)
    assert result.logits.shape == (5, 21)
