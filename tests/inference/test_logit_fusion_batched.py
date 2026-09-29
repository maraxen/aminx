"""RED tests for DEFECT 1: fusion-weight reshape crash at batch>1.

These tests assert correct behavior for ArithmeticMeanLogits, GeometricMeanLogits,
and ProductOfProbabilities when weights.shape=(1,) but per_state.shape=(S>1,...,V).

All three tests in group 1 CURRENTLY FAIL with a reshape error:
    ValueError: cannot reshape array of shape (1,) into shape (S, 1, 1)

This happens because make_stage_set defaults to jnp.ones(1) when state_weights=None,
but these modules call self.weights.reshape((per_state.shape[0],) + (1,)*dims_to_add).
When S=16 but weights has shape (1,), the reshape fails.

The fix must:
- Broadcast weights of shape (1,) to shape (S,) at call time, OR
- Only allow weights of shape (1,) OR (S,) (reject other lengths with a clear error).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# GROUP 1: Crash reproduction — shape (1,) weights with S>1 per_state
# ---------------------------------------------------------------------------

def test_arithmetic_mean_logits_shape1_weights_with_S16():
    """ArithmeticMeanLogits(weights=ones(1)) must not crash with per_state (16,5,21).

    CURRENTLY FAILS with:
        ValueError: cannot reshape array of shape (1,) into shape (16,1,1)
    After fix: must return shape (5, 21) without error.
    """
    from aminx.inference.logits import ArithmeticMeanLogits

    S, L, V = 16, 5, 21
    fuser = ArithmeticMeanLogits(weights=jnp.ones(1))
    per_state = jnp.zeros((S, L, V))

    result = fuser(per_state)

    assert result.shape == (L, V), (
        f"Expected shape ({L}, {V}), got {result.shape}"
    )


def test_geometric_mean_logits_shape1_weights_with_S16():
    """GeometricMeanLogits(weights=ones(1)) must not crash with per_state (16,5,21).

    CURRENTLY FAILS with:
        ValueError: cannot reshape array of shape (1,) into shape (16,1,1)
    After fix: must return shape (5, 21) without error.
    """
    from aminx.inference.logits import GeometricMeanLogits

    S, L, V = 16, 5, 21
    fuser = GeometricMeanLogits(weights=jnp.ones(1))
    per_state = jnp.zeros((S, L, V))

    result = fuser(per_state)

    assert result.shape == (L, V), (
        f"Expected shape ({L}, {V}), got {result.shape}"
    )


def test_product_of_probabilities_shape1_weights_with_S16():
    """ProductOfProbabilities(weights=ones(1)) must not crash with per_state (16,5,21).

    CURRENTLY FAILS with:
        ValueError: cannot reshape array of shape (1,) into shape (16,1,1)
    After fix: must return shape (5, 21) without error.
    """
    from aminx.inference.logits import ProductOfProbabilities

    S, L, V = 16, 5, 21
    fuser = ProductOfProbabilities(weights=jnp.ones(1))
    per_state = jnp.zeros((S, L, V))

    result = fuser(per_state)

    assert result.shape == (L, V), (
        f"Expected shape ({L}, {V}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# GROUP 2: Synthetic ground-truth invariant (BATHOS)
#
# For ArithmeticMeanLogits with uniform weights, fusing S identical states
# MUST equal the single state's logits. This tests both shape correctness and
# the broadcasting correctness: if weights are incorrectly broadcast, the
# normalization will be off and the result will differ.
# ---------------------------------------------------------------------------

def test_arithmetic_mean_identical_states_S1_equals_state():
    """ArithmeticMeanLogits: fusing 1 identical state returns that state's logits.

    This is a BATHOS invariant: identical states -> fused == single state.
    For S=1 this must always pass (regression baseline).
    """
    from aminx.inference.logits import ArithmeticMeanLogits

    S, L, V = 1, 5, 21
    # Analytic case: use arange for easy manual verification
    logits_1d = jnp.broadcast_to(jnp.arange(V, dtype=jnp.float32), (L, V))
    per_state = logits_1d[None, :, :]  # (1, L, V)

    fuser = ArithmeticMeanLogits(weights=jnp.ones(S))
    result = fuser(per_state)

    assert result.shape == (L, V), f"Expected ({L}, {V}), got {result.shape}"
    # Arithmetic mean of a single state must equal the state
    assert jnp.allclose(result, logits_1d, atol=1e-5), (
        f"S=1 arithmetic mean diverges from input logits: max diff "
        f"{jnp.max(jnp.abs(result - logits_1d))}"
    )


def test_arithmetic_mean_identical_states_S16_equals_state():
    """ArithmeticMeanLogits: fusing 16 identical states returns that state's logits.

    BATHOS invariant: identical states, uniform weights -> result == single state.
    This test will FAIL until both the reshape bug AND broadcasting are fixed:
    - If shape bug is not fixed: crash
    - If shape bug is fixed but broadcasting wrong: assertion fails
    """
    from aminx.inference.logits import ArithmeticMeanLogits

    S, L, V = 16, 5, 21
    logits_1d = jnp.broadcast_to(jnp.arange(V, dtype=jnp.float32), (L, V))
    per_state = jnp.broadcast_to(logits_1d[None, :, :], (S, L, V))

    fuser = ArithmeticMeanLogits(weights=jnp.ones(S))
    result = fuser(per_state)

    assert result.shape == (L, V), f"Expected ({L}, {V}), got {result.shape}"
    assert jnp.allclose(result, logits_1d, atol=1e-4), (
        f"S=16 identical states: arithmetic mean diverges from input logits: "
        f"max diff {jnp.max(jnp.abs(result - logits_1d))}"
    )


def test_arithmetic_mean_shape1_weights_identical_states_S16_equals_state():
    """ArithmeticMeanLogits(weights=ones(1)): fusing 16 identical states equals state.

    This covers the make_stage_set default code path (state_weights=None produces
    jnp.ones(1)). After fix, weights=(1,) with S=16 must broadcast correctly and
    produce the same result as weights=(16,) with identical states.
    """
    from aminx.inference.logits import ArithmeticMeanLogits

    S, L, V = 16, 5, 21
    logits_1d = jnp.broadcast_to(jnp.arange(V, dtype=jnp.float32), (L, V))
    per_state = jnp.broadcast_to(logits_1d[None, :, :], (S, L, V))

    fuser = ArithmeticMeanLogits(weights=jnp.ones(1))
    result = fuser(per_state)

    assert result.shape == (L, V), f"Expected ({L}, {V}), got {result.shape}"
    assert jnp.allclose(result, logits_1d, atol=1e-4), (
        f"weights=(1,) S=16 identical: arithmetic mean diverges. "
        f"max diff {jnp.max(jnp.abs(result - logits_1d))}"
    )


# ---------------------------------------------------------------------------
# GROUP 3: Shape guard for genuinely wrong weight count
#
# weights of length 3 with S=16 is a genuine misconfiguration.
# The current code raises a raw reshape error; the fix should raise a clear error.
# Marked xfail(strict=False): this test may pass (clean error) or fail (raw error).
# After fix: fixer should raise ValueError with a clear message when
# len(weights) not in {1, S}. Document that invariant here.
# ---------------------------------------------------------------------------

def test_arithmetic_mean_mismatched_weights_raises_clear_error():
    """ArithmeticMeanLogits(weights=ones(3)) with S=16 should raise a clear ValueError.

    weights of length 3 for 16 states is genuinely misconfigured (not a broadcast
    case). The code should detect this and raise:
        ValueError: 'weights length must be 1 (broadcast) or S=16, got 3'
    or similar. Currently raises raw reshape error which is confusing.
    """
    from aminx.inference.logits import ArithmeticMeanLogits

    S, L, V = 16, 5, 21
    fuser = ArithmeticMeanLogits(weights=jnp.ones(3))
    per_state = jnp.zeros((S, L, V))

    with pytest.raises(ValueError, match=r"state_weights length 3 incompatible with 16 states; expected length 1 or 16"):
        fuser(per_state)
