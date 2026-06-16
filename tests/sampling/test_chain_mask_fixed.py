"""Tests: spec.chain_mask_fixed flows through _prepare_ligand_context.

Invariant: when SamplingSpecification.chain_mask_fixed is set, the resulting
chain_mask must be exactly that value (not derived from fixed_mask).
Priority order: chain_mask_fixed > fixed_mask complement > all-ones fallback.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _prepare_ligand_context
from aminx.run.specs import SamplingSpecification


class _FakeBatch:
    """Minimal stand-in for a batched Protein with coordinates attribute."""

    def __init__(self, batch_size: int, seq_len: int) -> None:
        self.coordinates = np.zeros((batch_size, seq_len, 4, 3), dtype=np.float32)
        self.atom_mask = np.ones((batch_size, seq_len, 4), dtype=np.float32)


N = 10
BATCH = 1


@pytest.fixture
def fake_batch():
    return _FakeBatch(BATCH, N)


def _base_spec(**kwargs):
    return SamplingSpecification(
        inputs="test.pdb",
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        **kwargs
    )


def test_chain_mask_fixed_direct(fake_batch):
    """chain_mask_fixed alone must be reflected in chain_mask output."""
    mask = np.array([1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0], dtype=np.float32)
    spec = _base_spec(chain_mask_fixed=mask)
    result = _prepare_ligand_context(spec, batched_ensemble=fake_batch, batch_size=BATCH, seq_len=N)
    chain_mask = result["chain_mask"]
    np.testing.assert_array_equal(
        chain_mask[0],
        mask,
        err_msg="chain_mask_fixed not reflected in chain_mask output",
    )


def test_chain_mask_fixed_overrides_fixed_mask(fake_batch):
    """chain_mask_fixed must override fixed_mask (priority: chain_mask_fixed > fixed_mask)."""
    chain_mask_fixed = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    fixed_mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)  # would give complement [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    spec = _base_spec(chain_mask_fixed=chain_mask_fixed, fixed_mask=fixed_mask)
    result = _prepare_ligand_context(spec, batched_ensemble=fake_batch, batch_size=BATCH, seq_len=N)
    chain_mask = result["chain_mask"]
    np.testing.assert_array_equal(
        chain_mask[0],
        chain_mask_fixed,
        err_msg="chain_mask_fixed should override fixed_mask",
    )


def test_chain_mask_fixed_none_uses_fixed_mask(fake_batch):
    """When chain_mask_fixed is None, fixed_mask complement derivation should apply."""
    fixed_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    expected = 1.0 - fixed_mask  # [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    spec = _base_spec(fixed_mask=fixed_mask)
    result = _prepare_ligand_context(spec, batched_ensemble=fake_batch, batch_size=BATCH, seq_len=N)
    chain_mask = result["chain_mask"]
    np.testing.assert_array_equal(
        chain_mask[0],
        expected,
        err_msg="When chain_mask_fixed is None, fixed_mask complement should be used",
    )


def test_chain_mask_both_none_all_ones(fake_batch):
    """When both chain_mask_fixed and fixed_mask are None, chain_mask should be all-ones."""
    spec = _base_spec()
    result = _prepare_ligand_context(spec, batched_ensemble=fake_batch, batch_size=BATCH, seq_len=N)
    chain_mask = result["chain_mask"]
    expected = np.ones(N, dtype=np.float32)
    np.testing.assert_array_equal(
        chain_mask[0],
        expected,
        err_msg="chain_mask should be all-ones when both chain_mask_fixed and fixed_mask are None",
    )


def test_spec_roundtrip():
    """SamplingSpecification with chain_mask_fixed should round-trip correctly."""
    mask = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    spec = SamplingSpecification(inputs="test.pdb", chain_mask_fixed=mask)
    assert spec.chain_mask_fixed is not None
    np.testing.assert_array_equal(spec.chain_mask_fixed, mask)
