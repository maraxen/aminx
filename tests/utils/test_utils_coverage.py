"""Comprehensive unit tests for aminx.utils modules with 0% or low coverage.

This file provides tests for:
- aa_convert.py (0%)
- wave_parallel.py (0%)
- autoregression.py (38%, fills gaps in lines 62-141)
- gelu.py (0%)
- normalize.py (0%)
- safe_map.py (0%)
- decoding_order.py (77%, fills gaps)
- coordinates.py (82%, fills gaps)
- entropy.py (0%)
- ste.py (52%, fills gaps)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.utils import gelu, normalize, safe_map, ste
from aminx.utils.autoregression import (
    generate_ar_mask,
    get_decoding_step_map,
    make_autoregressive_mask,
    resolve_tie_groups,
)
from aminx.utils.coordinates import (
    apply_noise_to_coordinates,
    compute_backbone_coordinates,
    compute_c_beta,
)
from aminx.utils.decoding_order import (
    random_decoding_order,
    single_decoding_order,
)
from aminx.utils.entropy import (
    mle_entropy,
    posterior_entropy_mean,
    posterior_entropy_moments,
    posterior_entropy_squared_mean,
)

KEY = jax.random.PRNGKey(42)


def _check_proxide_available() -> bool:
    """Check if proxide is available."""
    try:
        import proxide  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# Tests for gelu.py (0% coverage)
# ============================================================================


class TestGeLU(chex.TestCase):
    """Test GeLU activation function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_gelu_basic_values(self):
        """Test GeLU against known JAX implementation."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        gelu_fn = self.variant(gelu.GeLU)
        result = gelu_fn(x)

        expected = jax.nn.gelu(x, approximate=False)
        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gelu_zero(self):
        """Test GeLU at zero."""
        x = jnp.array([0.0])
        gelu_fn = self.variant(gelu.GeLU)
        result = gelu_fn(x)

        chex.assert_trees_all_close(result, jnp.array([0.0]))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gelu_shape_preservation(self):
        """Test that GeLU preserves input shape."""
        x = jnp.ones((3, 4, 5))
        gelu_fn = self.variant(gelu.GeLU)
        result = gelu_fn(x)

        chex.assert_shape(result, x.shape)
        chex.assert_tree_all_finite(result)


# ============================================================================
# Tests for normalize.py (0% coverage)
# ============================================================================


class TestNormalization(chex.TestCase):
    """Test normalization utilities."""

    def test_normalize_standard_case(self):
        """Test normalize function with standard input."""
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale = jnp.array([2.0, 2.0, 2.0])
        offset = jnp.array([0.1, 0.1, 0.1])

        result = normalize.normalize(x, scale, offset, axis=-1)

        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        expected = (x - mean) / jnp.sqrt(variance + normalize.STANDARD_EPSILON)
        expected = expected * scale + offset

        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    def test_normalize_3d_input(self):
        """Test normalize with 3D input."""
        x = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))
        scale = jnp.ones(4)
        offset = jnp.zeros(4)

        result = normalize.normalize(x, scale, offset, axis=-1)

        chex.assert_shape(result, x.shape)
        chex.assert_tree_all_finite(result)

    def test_normalize_custom_epsilon(self):
        """Test normalize with custom epsilon."""
        x = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        scale = jnp.ones(2)
        offset = jnp.zeros(2)

        result = normalize.normalize(x, scale, offset, eps=1e-3)

        chex.assert_shape(result, x.shape)
        chex.assert_tree_all_finite(result)

    def test_layer_normalization_dict_params(self):
        """Test layer_normalization with dictionary parameters."""
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer_params = {
            "scale": jnp.array([1.5, 1.5, 1.5]),
            "offset": jnp.array([0.5, 0.5, 0.5]),
        }

        result = normalize.layer_normalization(x, layer_params)

        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        expected_norm = (x - mean) / jnp.sqrt(variance + normalize.STANDARD_EPSILON)
        expected = expected_norm * layer_params["scale"] + layer_params["offset"]

        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)


# ============================================================================
# Tests for safe_map.py (0% coverage)
# ============================================================================


class TestSafeMap(chex.TestCase):
    """Test safe_map dispatch logic."""

    def test_safe_map_vmap_path(self):
        """Test safe_map dispatches to vmap when batch_size >= input_size."""
        xs = jnp.arange(5)

        def f(x):
            return x * 3

        result = safe_map.safe_map(f, xs, batch_size=10)
        expected = xs * 3
        chex.assert_trees_all_close(result, expected)

    def test_safe_map_lax_map_path(self):
        """Test safe_map dispatches to lax.map when batch_size < input_size."""
        xs = jnp.arange(10)

        def f(x):
            return x + 2

        result = safe_map.safe_map(f, xs, batch_size=3)
        expected = xs + 2
        chex.assert_trees_all_close(result, expected)

    def test_safe_map_batch_size_none(self):
        """Test safe_map with batch_size=None always uses vmap."""
        xs = jnp.arange(8)

        def f(x):
            return x ** 2

        result = safe_map.safe_map(f, xs, batch_size=None)
        expected = xs ** 2
        chex.assert_trees_all_close(result, expected)

    def test_safe_map_batch_size_zero(self):
        """Test safe_map with batch_size=0 dispatches to vmap."""
        xs = jnp.arange(6)

        def f(x):
            return x * 2

        result = safe_map.safe_map(f, xs, batch_size=0)
        expected = xs * 2
        chex.assert_trees_all_close(result, expected)

    def test_safe_map_pytree_input(self):
        """Test safe_map handles PyTree inputs."""
        xs = {"a": jnp.arange(4), "b": jnp.arange(4) + 10}

        def f(x):
            return x["a"] + x["b"]

        result = safe_map.safe_map(f, xs, batch_size=10)
        expected = xs["a"] + xs["b"]
        chex.assert_trees_all_close(result, expected)

    def test_safe_map_empty_pytree_raises(self):
        """Test safe_map raises on empty PyTree."""
        with pytest.raises(ValueError, match="must not be an empty PyTree"):
            safe_map.safe_map(lambda x: x, {}, batch_size=10)

    def test_safe_map_nested_pytree(self):
        """Test safe_map with nested PyTree structures."""
        xs = {
            "x": jnp.arange(5),
            "nested": {"y": jnp.arange(5) * 2},
        }

        def f(x):
            return x["x"] + x["nested"]["y"]

        result = safe_map.safe_map(f, xs, batch_size=10)
        expected = xs["x"] + xs["nested"]["y"]
        chex.assert_trees_all_close(result, expected)


# ============================================================================
# Tests for decoding_order.py (77% coverage, fill gaps)
# ============================================================================


class TestDecodingOrder(chex.TestCase):
    """Test decoding order generation."""

    def test_random_decoding_order_no_ties(self):
        """Test random decoding order without tie groups."""
        order, _ = random_decoding_order(KEY, 5)

        chex.assert_shape(order, (5,))
        # All indices should be present
        assert jnp.all(jnp.isin(order, jnp.arange(5)))
        # All indices should be unique
        assert len(jnp.unique(order)) == 5

    def test_random_decoding_order_with_ties(self):
        """Test random decoding order respects tie groups."""
        tie_map = jnp.array([0, 1, 0, 2, 1])  # Groups: {0: [0,2], 1: [1,4], 2: [3]}
        order, _ = random_decoding_order(KEY, 5, tie_group_map=tie_map, num_groups=3)

        chex.assert_shape(order, (5,))
        assert len(jnp.unique(order)) == 5

    def test_random_decoding_order_negative_residues_raises(self):
        """Test random_decoding_order raises on negative num_residues."""
        with pytest.raises(TypeError, match="must be non-negative"):
            random_decoding_order(KEY, -1)

    def test_random_decoding_order_no_num_groups_raises(self):
        """Test random_decoding_order raises when tie_group_map provided but num_groups missing."""
        tie_map = jnp.array([0, 1, 0])
        with pytest.raises(ValueError, match="num_groups must be provided"):
            random_decoding_order(KEY, 3, tie_group_map=tie_map)

    def test_single_decoding_order_identity(self):
        """Test single_decoding_order returns identity permutation."""
        order, _ = single_decoding_order(KEY, 5)

        expected = jnp.arange(5, dtype=jnp.int32)
        chex.assert_trees_all_close(order, expected)

    def test_single_decoding_order_with_ties(self):
        """Test single_decoding_order ignores tie_group_map."""
        tie_map = jnp.array([0, 1, 0, 2])
        order, _ = single_decoding_order(KEY, 4, tie_group_map=tie_map, num_groups=3)

        expected = jnp.arange(4, dtype=jnp.int32)
        chex.assert_trees_all_close(order, expected)


# ============================================================================
# Tests for autoregression.py (38% coverage, fill gaps)
# ============================================================================


class TestAutoregression(chex.TestCase):
    """Test autoregression utilities."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_get_decoding_step_map(self):
        """Test get_decoding_step_map computation."""
        tie_group_map = jnp.array([0, 1, 0, 2, 1])
        group_decoding_order = jnp.array([2, 0, 1])  # Groups in order: 2, 0, 1

        step_map_fn = self.variant(get_decoding_step_map)
        step_map = step_map_fn(tie_group_map, group_decoding_order)

        # Verify shape and types
        chex.assert_shape(step_map, (5,))
        assert step_map.dtype == jnp.int32

    @chex.variants(with_jit=True, without_jit=True)
    def test_make_autoregressive_mask_basic(self):
        """Test make_autoregressive_mask with simple decoding order."""
        decoding_step_map = jnp.array([0, 1, 0, 2, 1])

        mask_fn = self.variant(make_autoregressive_mask)
        mask = mask_fn(decoding_step_map)

        # Verify shape
        chex.assert_shape(mask, (5, 5))
        # Diagonal should be all True
        assert jnp.all(jnp.diag(mask))
        # Mask should be upper triangular (or causal)
        chex.assert_tree_all_finite(mask)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_ar_mask_no_ties(self):
        """Test generate_ar_mask without tie groups."""
        decoding_order = jnp.array([0, 2, 1, 3, 4])

        ar_mask_fn = self.variant(generate_ar_mask)
        mask = ar_mask_fn(order=decoding_order)

        # Verify shape
        chex.assert_shape(mask, (5, 5))
        # Diagonal must be all 0: no position reads its own slot. Changed 2026-08-27 --
        # this previously asserted a set diagonal, matching the old non-strict
        # `row_indices >= col_indices`. Both reference implementations self-exclude
        # (LigandMPNN `1 - torch.triu(ones)`, ColabDesign `jnp.tri(L, k=-1)`), and the
        # "harmless placeholder" rationale for the old behaviour was false: undrawn slots
        # held token 0, which is ALANINE, embedded as a real nonzero vector.
        assert not jnp.any(jnp.diag(mask))
        # Non-degeneracy: an all-zero mask would also satisfy the line above while
        # removing every position's context.
        assert mask.sum() > 0
        chex.assert_tree_all_finite(mask)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_ar_mask_with_ties(self):
        """Test generate_ar_mask with tie groups."""
        decoding_order = jnp.array([0, 2, 1, 3])
        tie_group_map = jnp.array([0, 1, 0, 1])

        ar_mask_fn = self.variant(generate_ar_mask)
        mask = ar_mask_fn(order=decoding_order, tie_group_map=tie_group_map)

        chex.assert_shape(mask, (4, 4))
        # Diagonal must be all 0 on the tied branch too (see the untied case above for
        # why). Tie semantics themselves are unchanged: same-group positions remain
        # mutually visible, which the dedicated tie tests still assert.
        assert not jnp.any(jnp.diag(mask))
        assert mask.sum() > 0
        chex.assert_tree_all_finite(mask)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_ar_mask_with_chain_idx(self):
        """Test generate_ar_mask applies chain masking."""
        decoding_order = jnp.array([0, 1, 2, 3, 4])
        chain_idx = jnp.array([0, 0, 1, 1, 1])

        ar_mask_fn = self.variant(generate_ar_mask)
        mask = ar_mask_fn(order=decoding_order, chain_idx=chain_idx)

        chex.assert_shape(mask, (5, 5))
        # Cross-chain entries should be 0
        assert mask[0, 2] == 0
        assert mask[1, 3] == 0
        chex.assert_tree_all_finite(mask)


# ============================================================================
# Tests for coordinates.py (82% coverage, fill gaps)
# ============================================================================


class TestCoordinatesFilled(chex.TestCase):
    """Additional coordinate tests for gap coverage."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_noise_scalar_backbone_noise(self):
        """Test apply_noise_to_coordinates with scalar backbone_noise."""
        coords = jnp.ones((5, 5, 3))

        noise_fn = self.variant(apply_noise_to_coordinates)
        noisy, _ = noise_fn(KEY, coords, backbone_noise=0.5)

        chex.assert_shape(noisy, coords.shape)
        # With nonzero noise, should differ
        assert not jnp.allclose(coords, noisy)
        chex.assert_tree_all_finite(noisy)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_c_beta_orthogonal_vectors(self):
        """Test compute_c_beta with orthogonal bond vectors."""
        # Unit orthogonal vectors
        alpha_to_nitrogen = jnp.array([1.0, 0.0, 0.0])
        carbon_to_alpha = jnp.array([0.0, 1.0, 0.0])
        alpha_carbon = jnp.array([0.0, 0.0, 0.0])

        cb_fn = self.variant(compute_c_beta)
        cb = cb_fn(alpha_to_nitrogen, carbon_to_alpha, alpha_carbon)

        chex.assert_shape(cb, (3,))
        # CB should not be at origin
        assert not jnp.allclose(cb, alpha_carbon)
        chex.assert_tree_all_finite(cb)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_backbone_coordinates_large_input(self):
        """Test compute_backbone_coordinates with larger structures."""
        coords = jnp.zeros((50, 37, 3))
        coords = coords.at[:, 1, 0].set(jnp.arange(50))  # Vary CA x-coordinate

        bb_fn = self.variant(compute_backbone_coordinates)
        backbone = bb_fn(coords)

        chex.assert_shape(backbone, (50, 5, 3))
        chex.assert_tree_all_finite(backbone)


# ============================================================================
# Tests for entropy.py (0% coverage)
# ============================================================================


class TestEntropyFunctions(chex.TestCase):
    """Test entropy utility functions."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_zeros(self):
        """Test MLE entropy with zero counts (edge case)."""
        # When one state has all mass
        states = jnp.array([100.0, 0.0, 0.0])
        mle_fn = self.variant(mle_entropy)
        result = mle_fn(states)

        # Entropy should be 0 (no uncertainty)
        chex.assert_trees_all_close(result, 0.0, atol=1e-5)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_uniform(self):
        """Test MLE entropy with uniform distribution."""
        states = jnp.array([1.0, 1.0, 1.0, 1.0])
        mle_fn = self.variant(mle_entropy)
        result = mle_fn(states)

        # Entropy should be log(4) for uniform
        expected = jnp.log(4.0)
        chex.assert_trees_all_close(result, expected, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_mean_single_alpha(self):
        """Test posterior_entropy_mean with single element."""
        alpha = jnp.array([1.0])
        pem_fn = self.variant(posterior_entropy_mean)
        result = pem_fn(alpha)

        # Single element should have zero entropy
        chex.assert_trees_all_close(result, 0.0)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_mean_high_alpha(self):
        """Test posterior_entropy_mean with high concentration."""
        alpha = jnp.array([100.0, 1.0, 1.0])
        pem_fn = self.variant(posterior_entropy_mean)
        result = pem_fn(alpha)

        # Should be low entropy (concentrated distribution)
        assert result < 0.5
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_squared_mean_positive(self):
        """Test posterior_entropy_squared_mean is non-negative."""
        alpha = jnp.array([1.0, 2.0, 3.0])
        pesm_fn = self.variant(posterior_entropy_squared_mean)
        result = pesm_fn(alpha)

        assert result >= 0.0
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_moments_array(self):
        """Test posterior_entropy_moments returns 2-element array."""
        alpha = jnp.array([2.0, 3.0, 1.0])
        pem_fn = self.variant(posterior_entropy_moments)
        result = pem_fn(alpha)

        chex.assert_shape(result, (2,))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_moments_consistency(self):
        """Test moments are consistent with individual functions."""
        alpha = jnp.array([1.5, 2.0, 2.5])
        pem_fn = self.variant(posterior_entropy_moments)
        moments = pem_fn(alpha)

        mean = posterior_entropy_mean(alpha)
        squared_mean = posterior_entropy_squared_mean(alpha)

        chex.assert_trees_all_close(moments[0], mean, atol=1e-6)
        chex.assert_trees_all_close(moments[1], squared_mean, atol=1e-6)


# ============================================================================
# Tests for ste.py (52% coverage, fill gaps)
# ============================================================================


class TestSTEGradients(chex.TestCase):
    """Additional STE tests for gradient flow coverage."""

    def test_gumbel_softmax_hard(self):
        """Test Gumbel-softmax with hard=True."""
        logits = jnp.array([1.0, 2.0, 0.5])
        tau = 0.1

        result = ste.gumbel_softmax(logits, jnp.array(tau), KEY, hard=True)

        # Result should be soft in backward but one-hot in forward
        chex.assert_shape(result, logits.shape)
        chex.assert_tree_all_finite(result)

    def test_gumbel_softmax_soft(self):
        """Test Gumbel-softmax with hard=False."""
        logits = jnp.array([1.0, 2.0, 0.5])
        tau = 1.0

        result = ste.gumbel_softmax(logits, jnp.array(tau), KEY, hard=False)

        chex.assert_shape(result, logits.shape)
        # Result should be probability-like (sum to ~1)
        sum_result = jnp.sum(result)
        chex.assert_trees_all_close(sum_result, 1.0, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_ste_loss_no_mask(self):
        """Test ste_loss when all positions are valid."""
        logits_opt = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
        target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
        mask = jnp.ones(3, dtype=bool)

        ste_loss_fn = self.variant(ste.ste_loss)
        loss = ste_loss_fn(logits_opt, target_logits, mask)

        chex.assert_shape(loss, ())
        chex.assert_tree_all_finite(loss)
        assert loss > 0.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_ste_loss_partial_mask(self):
        """Test ste_loss with partial masking."""
        logits_opt = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
        target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
        mask = jnp.array([True, False, True], dtype=bool)

        ste_loss_fn = self.variant(ste.ste_loss)
        loss = ste_loss_fn(logits_opt, target_logits, mask)

        chex.assert_shape(loss, ())
        chex.assert_tree_all_finite(loss)


# ============================================================================
# Tests for aa_convert.py (0% coverage)
# Marked as skip if proxide import fails
# ============================================================================


@pytest.mark.skipif(
    not _check_proxide_available(),
    reason="proxide not available",
)
def test_af_to_mpnn_conversion():
    """Test AlphaFold to ProteinMPNN alphabet conversion."""
    from aminx.utils.aa_convert import af_to_mpnn

    # AF sequence (indices in AF alphabet)
    af_seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    mpnn_seq = af_to_mpnn(af_seq)

    chex.assert_shape(mpnn_seq, af_seq.shape)
    assert mpnn_seq.dtype == jnp.int8


@pytest.mark.skipif(
    not _check_proxide_available(),
    reason="proxide not available",
)
def test_mpnn_to_af_conversion():
    """Test ProteinMPNN to AlphaFold alphabet conversion."""
    from aminx.utils.aa_convert import mpnn_to_af

    # MPNN sequence (indices in MPNN alphabet)
    mpnn_seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    af_seq = mpnn_to_af(mpnn_seq)

    chex.assert_shape(af_seq, mpnn_seq.shape)
    assert af_seq.dtype == jnp.int8


@pytest.mark.skipif(
    not _check_proxide_available(),
    reason="proxide not available",
)
def test_round_trip_conversion():
    """Test round-trip conversion preserves sequence."""
    from aminx.utils.aa_convert import af_to_mpnn, mpnn_to_af

    original = jnp.array([0, 5, 10, 15, 20], dtype=jnp.int32)

    # Clamp to valid range for both alphabets
    af_alphabet_size = 21
    valid_seq = original % af_alphabet_size

    converted = af_to_mpnn(valid_seq)
    back = mpnn_to_af(converted)

    chex.assert_trees_all_close(back, valid_seq)


@pytest.mark.skipif(
    not _check_proxide_available(),
    reason="proxide not available",
)
def test_string_key_to_index_basic():
    """Test string_key_to_index conversion."""
    from aminx.utils.aa_convert import string_key_to_index

    string_keys = np.array(["A", "C", "G", "X"])
    key_map = {"A": 0, "C": 1, "G": 2}
    unk_index = 3

    indices = string_key_to_index(string_keys, key_map, unk_index)

    # A, C, G should map to 0, 1, 2; X should map to unknown index
    assert indices[3] == unk_index
    chex.assert_tree_all_finite(indices)


@pytest.mark.skipif(
    not _check_proxide_available(),
    reason="proxide not available",
)
def test_string_key_to_index_all_known():
    """Test string_key_to_index with all known keys."""
    from aminx.utils.aa_convert import string_key_to_index

    string_keys = np.array(["A", "B", "C"])
    key_map = {"A": 10, "B": 20, "C": 30}

    indices = string_key_to_index(string_keys, key_map)

    expected = jnp.array([10, 20, 30])
    chex.assert_trees_all_close(indices, expected)


# ============================================================================
# Tests for wave_parallel.py (0% coverage)
# ============================================================================


def test_compute_wave_assignments_basic():
    """Test compute_wave_assignments with small input."""
    from aminx.utils.wave_parallel import compute_wave_assignments

    # Small synthetic test case: 4 canonical positions
    n_canonical = 4
    ca_coords = jnp.ones((n_canonical, 4, 3))
    # Set different positions for CA
    ca_coords = ca_coords.at[0, 1, :].set(jnp.array([0.0, 0.0, 0.0]))
    ca_coords = ca_coords.at[1, 1, :].set(jnp.array([1.0, 0.0, 0.0]))
    ca_coords = ca_coords.at[2, 1, :].set(jnp.array([0.0, 1.0, 0.0]))
    ca_coords = ca_coords.at[3, 1, :].set(jnp.array([1.0, 1.0, 0.0]))

    # Tie groups and indices
    tie_group_flat = jnp.array([0, 0, 1, 1, 2, 2])
    max_group_size = 2
    group_indices_table = jnp.array([
        [0, 1, 0],
        [2, 3, 0],
        [4, 5, 0],
        [0, 0, 0],
    ])
    group_valid_table = jnp.array([
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, False, False],
    ])

    wave_ids, wave_positions, wave_valid, wave_pos_valid = compute_wave_assignments(
        ca_coords,
        tie_group_flat,
        group_indices_table,
        group_valid_table,
        k_neighbors=2,
        n_canonical=n_canonical,
    )

    # Check shapes
    assert wave_ids.shape[0] > 0  # Should have at least one wave
    assert wave_positions.shape[0] == wave_ids.shape[0]
    assert wave_valid.shape == wave_ids.shape
    assert wave_pos_valid.shape == wave_positions.shape


def test_compute_wave_assignments_outputs_valid():
    """Test compute_wave_assignments output validity."""
    from aminx.utils.wave_parallel import compute_wave_assignments

    n_canonical = 6
    ca_coords = jnp.zeros((n_canonical, 4, 3))
    for i in range(n_canonical):
        ca_coords = ca_coords.at[i, 1, :].set(jnp.array([float(i), 0.0, 0.0]))

    tie_group_flat = jnp.arange(6)
    max_group_size = 1
    group_indices_table = jnp.arange(6).reshape(6, 1)
    group_valid_table = jnp.ones((6, 1), dtype=bool)

    wave_ids, wave_positions, wave_valid, wave_pos_valid = compute_wave_assignments(
        ca_coords,
        tie_group_flat,
        group_indices_table,
        group_valid_table,
        k_neighbors=3,
        n_canonical=n_canonical,
    )

    # wave_valid entries should sum to n_canonical
    num_valid_groups = jnp.sum(wave_valid)
    assert num_valid_groups == n_canonical
