"""Tests for vectorized WaveScheduleBundle.from_colors and wave wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.kernel_dispatch import _structure_bundle_kwargs
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.types.bundles import WaveScheduleBundle


def test_from_colors_empty_tie_groups():
    schedule = WaveScheduleBundle.from_colors(
        group_colors=jnp.array([], dtype=jnp.int32),
        tie_group_map=jnp.array([], dtype=jnp.int32),
    )
    assert schedule.group_ids.shape == (0, 0)
    assert schedule.group_positions.shape == (0, 0, 0)


def test_from_colors_mismatched_length_raises():
    with pytest.raises(ValueError, match="entries but tie_group_map has"):
        WaveScheduleBundle.from_colors(
            group_colors=jnp.array([0, 1]),
            tie_group_map=jnp.array([0, 1, 2]),
        )


def test_from_colors_vectorized_matches_expected():
    # 3 groups across 6 positions:
    # group 0: positions [0, 3]
    # group 1: positions [1, 4]
    # group 2: positions [2, 5]
    tie_group_map = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
    # group 0 and 2 share color 0 (wave 0); group 1 has color 1 (wave 1)
    group_colors = jnp.array([0, 1, 0], dtype=jnp.int32)

    bundle = WaveScheduleBundle.from_colors(group_colors, tie_group_map)

    # num_waves = 2 (colors 0 and 1)
    # wave 0 has groups 0 and 2 (slot 0: group 0, slot 1: group 2)
    # wave 1 has group 1 (slot 0: group 1, slot 1: padding)
    assert bundle.group_ids.shape == (2, 2)
    assert bundle.group_positions.shape == (2, 2, 2)

    # Wave 0
    assert bundle.group_valid[0, 0]
    assert bundle.group_valid[0, 1]
    assert bundle.group_ids[0, 0] == 0
    assert bundle.group_ids[0, 1] == 2
    np.testing.assert_array_equal(bundle.group_positions[0, 0], [0, 3])
    np.testing.assert_array_equal(bundle.group_positions[0, 1], [2, 5])
    assert bool(bundle.position_valid[0, 0, 0])
    assert bool(bundle.position_valid[0, 0, 1])

    # Wave 1
    assert bundle.group_valid[1, 0]
    assert not bundle.group_valid[1, 1]
    assert bundle.group_ids[1, 0] == 1
    np.testing.assert_array_equal(bundle.group_positions[1, 0], [1, 4])


def test_wave_bundle_wiring_through_structure_bundle_kwargs():
    # Verify that a custom wave passed in shared reaches build_inference_bundle
    L = 4
    tie_group_map = jnp.arange(L, dtype=jnp.int32)
    group_colors = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    custom_wave = WaveScheduleBundle.from_colors(group_colors, tie_group_map)

    per_structure = {
        "coords": np.zeros((1, L, 4, 3), dtype=np.float32),
        "mask": np.ones((1, L), dtype=np.float32),
        "residue_index": np.tile(np.arange(L, dtype=np.int32), (1, 1)),
        "chain_index": np.zeros((1, L), dtype=np.int32),
        "fixed_mask": np.zeros((1, L), dtype=np.float32),
        "fixed_tokens": np.zeros((1, L), dtype=np.int32),
        "tie_group_map": tie_group_map[None, :],
        "state_position_map": None,
        "structure_mapping": None,
    }
    ligand_context = {
        "Y": None,
        "Y_t": None,
        "Y_m": None,
        "atom_37": None,
        "atom_37_mask": None,
        "chain_mask": None,
    }
    shared = {
        "bias": None,
        "state_weights": None,
        "mode": "sample_ar",
        "inference": True,
        "wave": custom_wave,
    }

    kwargs = _structure_bundle_kwargs(
        0, per_structure=per_structure, ligand_context=ligand_context, shared=shared
    )
    assert kwargs["wave"] is custom_wave

    bundle, _ = build_inference_bundle(**kwargs)
    assert bundle.wave is custom_wave
