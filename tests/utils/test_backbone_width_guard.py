"""Coordinate arrays with too few atoms per residue must raise, not be silently clamped (aminx #2152).

``compute_backbone_coordinates`` reads O at ``atom_order["O"] == 4``. On an array whose atom axis is narrower than
that, JAX does not raise for the out-of-range static index: it clamps to the last column, so O silently becomes the
last real atom (for a 3-wide N, CA, C array, O = C) and the output still looks like a valid ``(L, 5, 3)`` backbone.
The CB helper has the same hazard at column 3. These tests pin the guards, and that every valid layout is unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.utils.coordinates import compute_backbone_coordinates
from aminx.utils.structure_metrics import calculate_ca_distance_matrix, calculate_cb_distance_matrix

_L = 6


def _coords(width: int, seed: int = 0) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).standard_normal((_L, width, 3)).astype(np.float32))


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_compute_backbone_coordinates_rejects_fewer_than_four_atoms(width, jitted):
    fn = jax.jit(compute_backbone_coordinates) if jitted else compute_backbone_coordinates
    with pytest.raises(ValueError, match="at least 4 atoms"):
        fn(_coords(width))


@pytest.mark.parametrize("width", [4, 5, 14, 37])
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_valid_layouts_are_unchanged(width, jitted):
    """4 = compact (N, CA, C, O); >= 5 = atom37-style (N, CA, C, CB, O, ...). Output is (N, CA, C, O, CB)."""
    x = _coords(width)
    fn = jax.jit(compute_backbone_coordinates) if jitted else compute_backbone_coordinates
    out = np.asarray(fn(x))
    xn = np.asarray(x)
    o_col = 3 if width == 4 else 4
    assert out.shape == (_L, 5, 3)
    np.testing.assert_array_equal(out[:, 0], xn[:, 0])  # N
    np.testing.assert_array_equal(out[:, 1], xn[:, 1])  # CA
    np.testing.assert_array_equal(out[:, 2], xn[:, 2])  # C
    np.testing.assert_array_equal(out[:, 3], xn[:, o_col])  # O from its own column, never a clamped one


def test_the_hazard_is_real_without_the_guard():
    """Control: JAX really does clamp this read (so the guard is doing work, not guarding a non-problem)."""
    x = _coords(3)
    clamped = np.asarray(x[:, 4, :])  # the unguarded read compute_backbone_coordinates would have made
    np.testing.assert_array_equal(clamped, np.asarray(x[:, 2, :]), err_msg="expected O to clamp to column 2 (C)")


@pytest.mark.parametrize("width", [1, 2, 3])
def test_cb_distance_rejects_arrays_without_a_cb_column(width):
    with pytest.raises(ValueError, match="CB column"):
        calculate_cb_distance_matrix(_coords(width))


@pytest.mark.parametrize("width", [4, 5, 37])
def test_cb_distance_uses_column_three_for_valid_widths(width):
    x = _coords(width)
    got = np.asarray(calculate_cb_distance_matrix(x))
    cb = np.asarray(x)[:, 3, :]
    want = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_ca_distance_rejects_arrays_without_a_ca_column():
    with pytest.raises(ValueError, match="CA column"):
        calculate_ca_distance_matrix(_coords(1))
    calculate_ca_distance_matrix(_coords(2))  # CA is column 1: two atoms are enough


def _bundle(coords):
    length = coords.shape[-3]
    lead = coords.shape[:-3]  # () for one state, (S,) for a state-batched array: the builder wants matching leading dims
    return build_inference_bundle(
        coords=coords,
        mask=jnp.ones((*lead, length), jnp.float32),
        residue_index=jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (*lead, length)),
        chain_index=jnp.zeros((*lead, length), jnp.int32),
        mode="sample_ar",
        inference=True,
    )


@pytest.mark.parametrize("shape", [(_L, 3, 3), (2, _L, 3, 3)], ids=["single-state", "multi-state"])
def test_bundle_boundary_refuses_undersized_coords(shape):
    with pytest.raises(ValueError, match="at least 4 atoms"):
        _bundle(jnp.zeros(shape, jnp.float32))


def test_bundle_boundary_still_accepts_compact_and_atom37_coords():
    for width in (4, 37):
        bundle, _config = _bundle(_coords(width))
        assert bundle.geometry.coords.shape[2] == width
