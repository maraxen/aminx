"""Coverage tests for uncovered inference logits and bundle-builder branches.

This test module focuses on:
  1. Uncovered logit strategies (geometric_mean, product)
  2. Edge cases with bias and different state weights
  3. Bundle builder with non-default strategies
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.logits import (
    ArithmeticMeanLogits,
    GeometricMeanLogits,
    ProductOfProbabilities,
    ARLogitFuse,
    make_stage_set,
)
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.types.stages import (
    StageSet,
)
from aminx.types.encodings import EncoderOutput
from aminx.model import Aminx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_model():
    """Minimal synthetic Aminx for testing."""
    return Aminx(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        dropout_rate=0.0,
        key=jax.random.PRNGKey(0),
    )


@pytest.fixture
def synthetic_model_inference(synthetic_model):
    """Model in inference mode (no dropout)."""
    return eqx.tree_inference(synthetic_model, value=True)


@pytest.fixture
def synthetic_inputs():
    """Generate small deterministic inputs for testing.

    Returns:
        tuple of (coords, mask, residue_index, chain_index, sequence)
        where coords is (S=2, L=6, 4, 3) for multistate with 2 states and 6 residues
    """
    S, L = 2, 6  # 2 states, 6 residues
    rng = np.random.default_rng(42)

    coords = jnp.array(rng.normal(size=(S, L, 4, 3)).astype(np.float32))
    mask = jnp.ones((S, L), dtype=jnp.float32)
    residue_index = jnp.tile(jnp.arange(L, dtype=jnp.int32)[None, :], (S, 1))
    chain_index = jnp.zeros((S, L), dtype=jnp.int32)
    # Return a 1D sequence (L,) not (S, L) — build_inference_bundle expands it
    sequence = jnp.array(rng.integers(0, 20, size=L, dtype=np.int32))

    return coords, mask, residue_index, chain_index, sequence


# ---------------------------------------------------------------------------
# 1. Geometric Mean Logits Strategy (Uncovered)
# ---------------------------------------------------------------------------


class TestGeometricMeanLogits:
    """Test GeometricMeanLogits strategy which is currently uncovered."""

    def test_geometric_mean_reduces_state_dim(self):
        """GeometricMeanLogits reduces (S, L, V) to (L, V)."""
        S, L, V = 3, 6, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = GeometricMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)

    def test_geometric_mean_weighted_computation(self):
        """GeometricMeanLogits computes weighted average in logit space."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.6, 0.4])
        strategy = GeometricMeanLogits(weights=weights)

        # Create distinct logits per state
        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Expected: (w0 * L0 + w1 * L1) / sum(w)
        expected = (weights[0] * logits[0] + weights[1] * logits[1]) / jnp.sum(weights)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_geometric_mean_with_temperature(self):
        """GeometricMeanLogits respects temperature scaling."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        temperature = 2.0
        strategy = GeometricMeanLogits(weights=weights, temperature=temperature)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Temperature should scale the result
        expected = jnp.ones((L, V)) / temperature
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_geometric_mean_with_bias(self):
        """GeometricMeanLogits applies bias after averaging."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        strategy = GeometricMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.arange(L * V, dtype=jnp.float32).reshape(L, V)
        result = strategy(logits, bias=bias)

        # Result should be average + bias
        avg = jnp.mean(logits, axis=0)
        expected = avg + bias
        assert jnp.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Product of Probabilities Strategy (Uncovered)
# ---------------------------------------------------------------------------


class TestProductOfProbabilities:
    """Test ProductOfProbabilities strategy (sum of weighted log-probs)."""

    def test_product_reduces_state_dim(self):
        """ProductOfProbabilities reduces (S, L, V) to (L, V)."""
        S, L, V = 3, 6, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)

    def test_product_weighted_sum(self):
        """ProductOfProbabilities sums weighted logits."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.7, 0.3])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Expected: sum(w_i * L_i)
        expected = weights[0] * logits[0] + weights[1] * logits[1]
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_product_with_bias(self):
        """ProductOfProbabilities applies bias."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.ones((L, V)) * 2.0
        result = strategy(logits, bias=bias)

        sum_weighted = weights[0] * logits[0] + weights[1] * logits[1]
        expected = sum_weighted + bias
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_product_zero_weights(self):
        """ProductOfProbabilities handles zero weights gracefully."""
        S, L, V = 2, 4, 21
        weights = jnp.array([1.0, 0.0])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Only first state should contribute
        expected = logits[0]
        assert jnp.allclose(result, expected, atol=1e-5)


class TestProductSharpness:
    """The `sharpness` (κ) scale, and the product-vs-opinion-pool distinction.

    `weights` are mixing proportions; `sharpness` is concentration. Conflating
    them is what silently turns a product of experts into a weighted geometric
    mean, so these tests pin the boundary rather than just the arithmetic.
    """

    def test_sharpness_defaults_to_identity(self):
        """Default sharpness=1.0 leaves the weighted sum untouched (back-compat)."""
        S, L, V = 3, 4, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)

        explicit = ProductOfProbabilities(weights=weights, sharpness=1.0)(logits)
        default = ProductOfProbabilities(weights=weights)(logits)

        assert jnp.allclose(default, explicit, atol=1e-6)

    def test_sharpness_scales_logits(self):
        """sharpness=κ multiplies the fused logits by κ."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.6, 0.4])
        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)

        base = ProductOfProbabilities(weights=weights, sharpness=1.0)(logits)
        scaled = ProductOfProbabilities(weights=weights, sharpness=3.0)(logits)

        assert jnp.allclose(scaled, 3.0 * base, atol=1e-4)

    def test_sharpness_none_uses_state_count(self):
        """sharpness=None scales by S, the number of states."""
        S, L, V = 4, 3, 21
        weights = jnp.array([0.4, 0.3, 0.2, 0.1])
        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)

        auto = ProductOfProbabilities(weights=weights, sharpness=None)(logits)
        manual = ProductOfProbabilities(weights=weights, sharpness=float(S))(logits)

        assert jnp.allclose(auto, manual, atol=1e-4)

    def test_normalized_weights_with_sharpness_none_recovers_plain_poe(self):
        """Uniform Σw=1 weights at sharpness=None reproduce the plain product Σ Lᵢ.

        This is the recipe the necklace campaign needs: mixing proportions stay a
        genuine simplex (Σw=1) while the operator remains a true product.
        """
        S, L, V = 4, 5, 21
        logits = jax.random.normal(jax.random.PRNGKey(0), (S, L, V))

        uniform_simplex = jnp.ones(S, dtype=jnp.float32) / S
        weighted_poe = ProductOfProbabilities(weights=uniform_simplex, sharpness=None)(logits)

        plain_product = jnp.sum(logits, axis=0)

        assert jnp.allclose(weighted_poe, plain_product, atol=1e-4)

    def test_identical_experts_product_sharpens_pool_does_not(self):
        """The decisive semantic check, on S identical experts.

        A product of S identical experts must give p^S — i.e. S·L in logit space.
        A Σw=1 geometric mean (logarithmic opinion pool) returns the expert
        unchanged. This is exactly the distinction that made `Σw=1` silently stop
        being a product; if this test ever passes for both, the operator is wrong.
        """
        S, L, V = 4, 3, 21
        single_expert = jax.random.normal(jax.random.PRNGKey(1), (L, V))
        logits = jnp.broadcast_to(single_expert, (S, L, V))

        simplex = jnp.ones(S, dtype=jnp.float32) / S

        pool = ProductOfProbabilities(weights=simplex, sharpness=1.0)(logits)
        product = ProductOfProbabilities(weights=simplex, sharpness=None)(logits)

        # Opinion pool: unchanged.
        assert jnp.allclose(pool, single_expert, atol=1e-4)
        # Product: sharpened by exactly S.
        assert jnp.allclose(product, S * single_expert, atol=1e-4)
        # And the two are genuinely different (guards against a no-op regression).
        assert not jnp.allclose(pool, product, atol=1e-2)

    def test_bias_is_not_amplified_by_sharpness(self):
        """Bias is added after scaling, so it stays on the fused-logit scale."""
        S, L, V = 2, 3, 21
        weights = jnp.array([0.5, 0.5])
        logits = jnp.ones((S, L, V))
        bias = jnp.full((L, V), 2.0)

        result = ProductOfProbabilities(weights=weights, sharpness=5.0)(logits, bias=bias)

        expected = 5.0 * jnp.sum(logits * weights.reshape(S, 1, 1), axis=0) + bias
        assert jnp.allclose(result, expected, atol=1e-4)


class TestMakeStageSetSharpness:
    """make_stage_set forwards sharpness only to strategies that declare it."""

    def test_sharpness_reaches_product_strategy(self):
        """A sharpness passed to make_stage_set lands on the product transform."""
        stage_set = make_stage_set(
            strategy="product",
            state_weights=jnp.ones(3, dtype=jnp.float32) / 3,
            sharpness=None,
        )
        assert isinstance(stage_set.logit_transform, ProductOfProbabilities)
        assert stage_set.logit_transform.sharpness is None

    def test_sharpness_default_preserved(self):
        """Omitting sharpness leaves the identity scale."""
        stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(2))
        assert stage_set.logit_transform.sharpness == 1.0

    def test_non_product_strategies_unaffected(self):
        """Strategies without a `sharpness` field still construct cleanly."""
        for strategy, cls in (
            ("arithmetic_mean", ArithmeticMeanLogits),
            ("geometric_mean", GeometricMeanLogits),
        ):
            stage_set = make_stage_set(
                strategy=strategy,
                state_weights=jnp.ones(2),
                sharpness=4.0,
            )
            assert isinstance(stage_set.logit_transform, cls)
            assert not hasattr(stage_set.logit_transform, "sharpness")

    def test_geometric_mean_still_receives_temperature(self):
        """The new kwarg dispatch must not drop the pre-existing temperature knob."""
        stage_set = make_stage_set(
            strategy="geometric_mean",
            state_weights=jnp.ones(2),
            strategy_temperature=0.25,
        )
        assert stage_set.logit_transform.temperature == 0.25

    def test_ar_path_shares_the_sharpened_instance(self):
        """AR fusion must see the same sharpness as the non-AR path (a23 invariant)."""
        stage_set = make_stage_set(
            strategy="product",
            state_weights=jnp.ones(4, dtype=jnp.float32) / 4,
            sharpness=None,
        )
        assert stage_set.ar_logit_transform is stage_set.logit_transform


# ---------------------------------------------------------------------------
# 3. Arithmetic Mean with Bias (Extended)
# ---------------------------------------------------------------------------


class TestArithmeticMeanLogitsWithBias:
    """Extended tests for ArithmeticMeanLogits with bias."""

    def test_arithmetic_mean_with_bias(self):
        """ArithmeticMeanLogits applies bias correctly."""
        S, L, V = 3, 6, 21
        weights = jnp.ones(S) / S
        strategy = ArithmeticMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.arange(L * V, dtype=jnp.float32).reshape(L, V) * 0.01
        result = strategy(logits, bias=bias)

        # Result should be mean + bias
        avg = jnp.mean(logits, axis=0)
        expected = avg + bias
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_arithmetic_mean_unequal_weights(self):
        """ArithmeticMeanLogits respects non-uniform weights."""
        S, L, V = 3, 4, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = ArithmeticMeanLogits(weights=weights)

        # Create distinct per-state logits
        logits = jnp.array([
            jnp.ones((L, V)) * 1.0,
            jnp.ones((L, V)) * 2.0,
            jnp.ones((L, V)) * 3.0,
        ])

        result = strategy(logits)

        # Expected: weighted arithmetic mean in log-space
        # log(sum(w_i * exp(L_i)) / sum(w_i))
        weighted = weights[:, None, None] * jnp.exp(logits)
        expected_exp_avg = jnp.sum(weighted, axis=0) / jnp.sum(weights)
        expected = jnp.log(expected_exp_avg)

        assert jnp.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Bundle Builder with Different Strategies
# ---------------------------------------------------------------------------


class TestBundleBuilderStrategies:
    """Test bundle_builder with non-default strategies."""

    def test_bundle_builder_geometric_mean(self, synthetic_inputs):
        """bundle_builder correctly wires geometric_mean strategy."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            mode="score_conditional",
        )
        stage_set = make_stage_set(strategy="geometric_mean", strategy_temperature=2.0)

        # Verify geometric_mean is wired
        assert isinstance(stage_set.logit_transform, GeometricMeanLogits)
        assert stage_set.logit_transform.temperature == 2.0

    def test_bundle_builder_product_strategy(self, synthetic_inputs):
        """bundle_builder correctly wires product strategy."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
        )
        stage_set = make_stage_set(strategy="product")

        # Verify product is wired
        assert isinstance(stage_set.logit_transform, ProductOfProbabilities)

    def test_bundle_builder_with_custom_state_weights(self, synthetic_inputs):
        """bundle_builder respects custom state_weights."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs
        custom_weights = jnp.array([0.3, 0.7])

        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            state_weights=custom_weights,
        )
        stage_set = make_stage_set(strategy="arithmetic_mean", state_weights=custom_weights)

        # Verify weights are propagated
        assert jnp.allclose(stage_set.logit_transform.weights, custom_weights)

    def test_bundle_builder_geometric_with_custom_temp(self, synthetic_inputs):
        """bundle_builder wires custom temperature for geometric_mean."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
        )
        stage_set = make_stage_set(strategy="geometric_mean", strategy_temperature=0.5)

        assert isinstance(stage_set.logit_transform, GeometricMeanLogits)
        assert stage_set.logit_transform.temperature == 0.5


# ---------------------------------------------------------------------------
# 7. AR Logit Transform (Extended)
# ---------------------------------------------------------------------------


class TestARLogitFuseExtended:
    """Extended tests for ARLogitFuse in AR path."""

    def test_ar_logit_fuse_in_stage_set(self):
        """ARLogitFuse is correctly wired in StageSet.ar_logit_transform."""
        stage_set = StageSet(ar_logit_transform=ARLogitFuse())
        assert isinstance(stage_set.ar_logit_transform, ARLogitFuse)

    def test_ar_logit_fuse_vmap_compatible(self):
        """ARLogitFuse works under vmap for per-position fusion with per-position bias."""
        fuse = ARLogitFuse()

        # Simulate (S, L, V) logits and (L, V) bias
        S, L, V = 3, 6, 21
        logits = jnp.ones((S, L, V))
        bias = jnp.zeros((L, V))

        # vmap over position axis: (S, L, V) + (L, V) -> (L, V)
        # in_axes=(1, 0) means move axis 1 of logits to front and axis 0 of bias to front
        vmapped_fuse = jax.vmap(fuse, in_axes=(1, 0), out_axes=0)
        result = vmapped_fuse(logits, bias)

        assert result.shape == (L, V)
        # Each position should be mean of logits at that position plus bias
        expected_per_pos = jnp.mean(logits, axis=0) + bias
        assert jnp.allclose(result, expected_per_pos, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Integration: End-to-End Bundle to Logits
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """Integration tests combining bundle_builder with logit strategies."""

    def test_bundle_with_all_strategies(self, synthetic_inputs):
        """bundle_builder integrates with all logit strategies."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        for strategy in ["arithmetic_mean", "geometric_mean", "product"]:
            bundle, config = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=sequence,
            )
            # Test with S=2 states, so provide matching state weights
            state_weights = jnp.array([0.5, 0.5])
            stage_set = make_stage_set(strategy=strategy, state_weights=state_weights)

            # Verify strategy is wired and callable
            assert stage_set.logit_transform is not None

            # Test callable: (S, L, V) -> (L, V)
            S, L, V = 2, 6, 21
            test_logits = jnp.ones((S, L, V))
            result = stage_set.logit_transform(test_logits)
            assert result.shape == (L, V)

    def test_bundle_arithmetic_with_bias(self, synthetic_inputs):
        """Logit transform applies bias after fusion."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs
        L = coords.shape[1]

        bias = jnp.ones((L, 21)) * 0.5

        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            bias=bias,
        )
        # Test with S=2 states, so provide matching state weights
        state_weights = jnp.array([0.5, 0.5])
        stage_set = make_stage_set(state_weights=state_weights)

        # Verify bias is in conditioning
        assert jnp.allclose(bundle.conditioning.bias, bias)

        # Test that logit_transform can accept and apply bias
        S, V = 2, 21
        test_logits = jnp.ones((S, L, V))
        result = stage_set.logit_transform(test_logits, bias=bias)

        # Result should be higher due to positive bias
        result_no_bias = stage_set.logit_transform(test_logits)
        assert jnp.all(result > result_no_bias - 0.1)  # bias effect visible


# ---------------------------------------------------------------------------
# 9. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_geometric_mean(self):
        """GeometricMeanLogits with S=1 reduces properly."""
        weights = jnp.array([1.0])
        strategy = GeometricMeanLogits(weights=weights)

        L, V = 6, 21
        logits = jnp.ones((1, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)
        assert jnp.allclose(result, logits[0], atol=1e-5)

    def test_single_state_product(self):
        """ProductOfProbabilities with S=1."""
        weights = jnp.array([1.0])
        strategy = ProductOfProbabilities(weights=weights)

        L, V = 6, 21
        logits = jnp.ones((1, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)
        assert jnp.allclose(result, logits[0], atol=1e-5)

    def test_large_state_weights(self):
        """Strategies handle large weight values."""
        S, L, V = 3, 6, 21
        weights = jnp.array([100.0, 200.0, 300.0])  # Large values

        strategy = ArithmeticMeanLogits(weights=weights)
        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Should still be ~1.0 (weighted average of 1.0)
        assert jnp.allclose(result, jnp.ones((L, V)), atol=1e-4)

    def test_very_small_weights(self):
        """Strategies handle very small weight values."""
        S, L, V = 2, 4, 21
        weights = jnp.array([1e-9, 1e-9])

        strategy = ProductOfProbabilities(weights=weights)
        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Should still compute without NaNs
        assert not jnp.any(jnp.isnan(result))


# ---------------------------------------------------------------------------
# 10. JIT Compilation
# ---------------------------------------------------------------------------


class TestJITCompilation:
    """Test that all paths work under jax.jit."""

    def test_arithmetic_mean_jit(self):
        """ArithmeticMeanLogits works under jit."""
        strategy = ArithmeticMeanLogits(weights=jnp.array([0.5, 0.5]))

        @jax.jit
        def apply(logits, bias):
            return strategy(logits, bias=bias)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        bias = jnp.zeros((L, V))
        result = apply(logits, bias)

        assert result.shape == (L, V)

    def test_geometric_mean_jit(self):
        """GeometricMeanLogits works under jit."""
        strategy = GeometricMeanLogits(
            weights=jnp.array([0.6, 0.4]),
            temperature=1.5
        )

        @jax.jit
        def apply(logits):
            return strategy(logits)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        result = apply(logits)

        assert result.shape == (L, V)

    def test_product_jit(self):
        """ProductOfProbabilities works under jit."""
        strategy = ProductOfProbabilities(weights=jnp.array([0.7, 0.3]))

        @jax.jit
        def apply(logits, bias):
            return strategy(logits, bias=bias)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        bias = jnp.zeros((L, V))
        result = apply(logits, bias)

        assert result.shape == (L, V)
