"""Tests for coordinate utilities."""

import chex
import jax
import jax.numpy as jnp

from aminx.utils.coordinates import (
    apply_noise_to_coordinates,
    compute_backbone_coordinates,
    compute_backbone_distance,
    compute_c_beta,
    compute_cb_precise,
    extend_coordinate,
)

KEY = jax.random.PRNGKey(42)


class TestCoordinates(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_noise_to_coordinates(self):
        """Test adding Gaussian noise to atomic coordinates."""
        coords = jnp.ones((10, 5, 3))

        # Test with no noise
        apply_noise_fn = self.variant(apply_noise_to_coordinates)
        coords_no_noise, _ = apply_noise_fn(
            KEY, coords, backbone_noise=jnp.array(0.0),
        )
        chex.assert_trees_all_close(coords, coords_no_noise)

        # Test with noise
        coords_with_noise, _ = apply_noise_fn(
            KEY, coords, backbone_noise=jnp.array(0.1),
        )
        chex.assert_shape(coords, coords_with_noise.shape)
        assert not jnp.allclose(coords, coords_with_noise)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_c_beta(self):
        """Test the computation of C-beta coordinates."""
        alpha_carbon = jnp.array([0.0, 0.0, 0.0])
        alpha_to_nitrogen = jnp.array([1.0, 0.0, 0.0])
        carbon_to_alpha = jnp.array([0.0, 1.0, 0.0])

        # Manually computed expected result based on the function's formula
        f1, f2, f3 = -0.58273431, 0.56802827, -0.54067466
        term1 = f1 * jnp.cross(alpha_to_nitrogen, carbon_to_alpha)
        term2 = f2 * alpha_to_nitrogen
        term3 = f3 * carbon_to_alpha
        expected_cb = term1 + term2 + term3 + alpha_carbon

        compute_c_beta_fn = self.variant(compute_c_beta)
        computed_cb = compute_c_beta_fn(
            alpha_to_nitrogen, carbon_to_alpha, alpha_carbon,
        )
        chex.assert_trees_all_close(computed_cb, expected_cb)
        chex.assert_tree_all_finite(computed_cb)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_backbone_coordinates(self):
        """Test the computation of backbone coordinates."""
        coords = jnp.arange(10 * 5 * 3).reshape((10, 5, 3)).astype(jnp.float32)
        compute_backbone_coordinates_fn = self.variant(compute_backbone_coordinates)
        backbone_coords = compute_backbone_coordinates_fn(coords)

        chex.assert_shape(backbone_coords, (10, 5, 3))
        # N, CA, C, O should be passed through
        chex.assert_trees_all_equal(coords[:, :3, :], backbone_coords[:, :3, :])
        assert not jnp.allclose(coords[:, 4, :], backbone_coords[:, 4, :])
        chex.assert_tree_all_finite(backbone_coords)

    def test_compute_backbone_coordinates_compact_backbone_layout(self):
        """Regression (260926_browser-export-loop, T2b).

        The export wrappers (``aminx.export.wrappers.make_p03_featurize`` /
        ``make_p04_unconditional``) pass a compact, backbone-only ``(L, 4, 3)`` array in
        the literal column order (N, CA, C, O) -- there is no CB column at all, unlike
        the >= 5-atom Atom37 layout the other branch handles. ``proxide``'s
        ``atom_order["O"] == 4`` is out of range on a 4-wide axis; the previous
        implementation relied on JAX's basic-indexing fallback to
        ``lax.dynamic_slice``, whose start index XLA silently clamps to the last valid
        column (3) at runtime -- numerically correct here only by coincidence (column 3
        IS oxygen in this layout), but it traces a ``dynamic_slice`` this test asserts
        against directly. ``jax2onnx`` exports that same computation as an unclamped
        ONNX ``Slice(start=4)`` on a dim of size 4, which onnxruntime rejects at
        ``InferenceSession`` construction (``ShapeInferenceError`` on a genuinely empty
        ``[L, 0, 3]`` slice feeding ``Squeeze``, node ``node_Squeeze_24``).
        """
        coords = jnp.arange(6 * 4 * 3).reshape((6, 4, 3)).astype(jnp.float32)

        backbone_coords = compute_backbone_coordinates(coords)
        chex.assert_shape(backbone_coords, (6, 5, 3))
        # N, CA, C, O pass through unchanged from the compact layout's own columns.
        chex.assert_trees_all_equal(coords, backbone_coords[:, :4, :])
        chex.assert_tree_all_finite(backbone_coords)

        # The traced jaxpr for a 4-wide atom axis must contain no `dynamic_slice`: every
        # backbone-atom read is a static, in-range `slice`. Red-check (revert the
        # `coordinates.shape[-2] == 4` branch in `compute_backbone_coordinates`): this
        # assertion fails because the O read falls back to `dynamic_slice`, exactly the
        # primitive `jax2onnx` lowers to the unclamped ONNX `Slice` that breaks ORT.
        jaxpr = jax.make_jaxpr(compute_backbone_coordinates.__wrapped__)(coords)
        primitive_names = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
        assert "dynamic_slice" not in primitive_names, (
            f"expected no dynamic_slice for a 4-wide atom axis, got eqns: {primitive_names}"
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_backbone_distance(self):
        """Test computation of pairwise distances between backbone alpha carbons."""
        coords = jnp.zeros((2, 5, 3), dtype=jnp.float32)
        coords = coords.at[1, 1, 0].set(3.0)  # Distance of 3 between the two CA atoms

        compute_backbone_distance_fn = self.variant(compute_backbone_distance)
        distances = compute_backbone_distance_fn(coords)
        # The shape should be (N, N) as it's the distance between corresponding atoms
        chex.assert_shape(distances, (2, 2))
        chex.assert_trees_all_close(distances[0, 0], jnp.sqrt(1e-6))
        chex.assert_trees_all_close(distances[0, 1], 3.0, atol=1e-3)
        chex.assert_tree_all_finite(distances)

    @chex.variants(with_jit=True, without_jit=True)
    def test_extend_coordinate(self):
        """Test the extension of coordinates to a fourth atom."""
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 0.0, 0.0])
        c = jnp.array([0.0, 1.0, 0.0])

        extend_coordinate_fn = self.variant(extend_coordinate)

        # Place D in the xy-plane
        d = extend_coordinate_fn(
            a, b, c, bond_length=1.0, bond_angle=jnp.pi / 2, dihedral_angle=0.0,
        )
        # Corrected expected value
        chex.assert_trees_all_close(d, jnp.array([1.0, 1.0, 0.0]), atol=1e-6)
        chex.assert_tree_all_finite(d)

        # Place D out of the xy-plane
        d_dihedral = extend_coordinate_fn(
            a, b, c, bond_length=1.0, bond_angle=jnp.pi / 2, dihedral_angle=jnp.pi / 2,
        )
        chex.assert_trees_all_close(d_dihedral, jnp.array([0.0, 1.0, -1.0]), atol=1e-6)
        chex.assert_tree_all_finite(d_dihedral)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_cb_precise(self):
        """Test the precise computation of C-beta coordinates."""
        n = jnp.array([1.45, 0.0, 0.0])
        ca = jnp.array([0.0, 0.0, 0.0])
        c = jnp.array([0.0, 1.53, 0.0])

        compute_cb_precise_fn = self.variant(compute_cb_precise)
        cb = compute_cb_precise_fn(n, ca, c)
        chex.assert_shape(cb, (3,))

        # Check if bond length from CA to CB is roughly correct
        bond_length = jnp.linalg.norm(cb - ca)
        chex.assert_trees_all_close(bond_length, 1.522, atol=1e-3)
        chex.assert_tree_all_finite(cb)
