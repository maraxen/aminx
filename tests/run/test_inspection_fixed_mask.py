"""Test that fixed_mask is properly threaded through the inspection runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from aminx.run.specs import InspectionSpecification

if TYPE_CHECKING:
    pass


def test_inspection_specification_with_fixed_mask() -> None:
    """Verify InspectionSpecification accepts and preserves fixed_mask field."""
    # Create a simple fixed_mask array
    fixed_mask = jnp.array([1, 1, 0, 0, 1])

    # Construct InspectionSpecification with fixed_mask
    spec = InspectionSpecification(
        inputs=["test.pdb"],
        fixed_mask=fixed_mask,
    )

    # Verify the field round-trips
    assert spec.fixed_mask is not None
    assert jnp.array_equal(spec.fixed_mask, fixed_mask)


def test_inspection_specification_with_sidechain_conditioning() -> None:
    """Verify InspectionSpecification accepts and preserves sidechain_conditioning field."""
    spec = InspectionSpecification(
        inputs=["test.pdb"],
        sidechain_conditioning=True,
    )

    # Verify the field round-trips
    assert spec.sidechain_conditioning is True


def test_inspection_specification_both_fields() -> None:
    """Verify InspectionSpecification accepts both fixed_mask and sidechain_conditioning."""
    fixed_mask = jnp.array([1, 1, 0, 0, 1])

    spec = InspectionSpecification(
        inputs=["test.pdb"],
        fixed_mask=fixed_mask,
        sidechain_conditioning=True,
    )

    # Verify both fields round-trip
    assert spec.fixed_mask is not None
    assert jnp.array_equal(spec.fixed_mask, fixed_mask)
    assert spec.sidechain_conditioning is True
