"""Tests for chain_mask computation from fixed_mask in _prepare_ligand_context.

Focus: Verify chain_mask (1=designable, 0=fixed) is computed as complement of fixed_mask.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _prepare_ligand_context
from aminx.run.specs import SamplingSpecification
from aminx.utils.data_structures import Protein


class _FakeProteinLigandMpnn:
  """Minimal Protein stand-in for ligandmpnn tests with sidechain conditioning."""

  def __init__(self, *, seq_len: int, batch_size: int) -> None:
    self.coordinates = jnp.zeros((batch_size, seq_len, 37, 3), dtype=jnp.float32)
    self.aatype = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    self.atom_mask = jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32)
    self.residue_index = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(
        batch_size, axis=0
    )
    self.chain_index = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    self.mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    self.mapping = None
    self.Y = jnp.zeros((batch_size, seq_len, 1, 3), dtype=jnp.float32)
    self.Y_t = jnp.zeros((batch_size, seq_len, 1), dtype=jnp.int32)
    self.Y_m = jnp.ones((batch_size, seq_len, 1), dtype=jnp.float32)
    self.full_atom_mask = None


def _make_fake_protein_ligandmpnn(
    batch_size: int = 1, seq_len: int = 10
):
  """Create a minimal Protein for ligandmpnn tests with sidechain conditioning."""
  return _FakeProteinLigandMpnn(seq_len=seq_len, batch_size=batch_size)


class TestChainMaskFromFixedMask:
  """Test suite for chain_mask computation from fixed_mask."""

  def test_chain_mask_all_ones_without_fixed_mask(self):
    """chain_mask should be all-ones when fixed_mask is None.

    Convention: chain_mask 1=designable, 0=fixed.
    Default (no fixed_mask) = all residues designable.
    """
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]

    # Should have correct shape
    assert chain_mask.shape == (batch_size, seq_len)

    # All ones (all designable)
    expected = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(chain_mask), np.asarray(expected))

  def test_chain_mask_complement_of_fixed_mask_1d(self):
    """chain_mask should be 1.0 - fixed_mask when fixed_mask is 1D array.

    Positions with fixed_mask=1 (fixed) should have chain_mask=0 (not designable).
    Positions with fixed_mask=0 (designable) should have chain_mask=1 (designable).
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    # fixed_mask: positions 2, 5, 7 are fixed (value=1)
    fixed_mask = np.array([0., 0., 1., 0., 0., 1., 0., 1., 0., 0.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]

    # Expected: positions 2, 5, 7 should have chain_mask=0 (fixed)
    # all others should have chain_mask=1 (designable)
    expected = np.array([[1., 1., 0., 1., 1., 0., 1., 0., 1., 1.]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(chain_mask), expected)

  def test_chain_mask_complement_of_fixed_mask_2d(self):
    """chain_mask should be 1.0 - fixed_mask for 2D fixed_mask.

    Each batch can have different fixed positions.
    """
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    # 2D fixed_mask: different per batch
    fixed_mask = np.array(
        [
            [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],  # batch 0
            [0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],  # batch 1
        ],
        dtype=np.float32,
    )

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]

    # Expected: complement of fixed_mask
    expected = np.array(
        [
            [1., 1., 0., 1., 1., 0., 1., 1., 1., 1.],
            [1., 1., 1., 1., 0., 1., 1., 0., 1., 1.],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(chain_mask), expected)

  def test_chain_mask_dtype_is_float32(self):
    """chain_mask should always have dtype float32."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    fixed_mask = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]
    assert chain_mask.dtype == jnp.float32

  def test_chain_mask_values_in_range(self):
    """All chain_mask values should be 0.0 or 1.0 (binary)."""
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    fixed_mask = np.array([0., 1., 0., 1., 0., 0., 1., 0., 0., 1.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]

    # All values should be exactly 0 or 1
    mask_array = np.asarray(chain_mask)
    assert np.all((mask_array == 0.0) | (mask_array == 1.0))

  def test_chain_mask_none_without_sidechain_conditioning(self):
    """chain_mask should be None when sidechain_conditioning=False."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    fixed_mask = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=False,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]
    assert chain_mask is None

  def test_chain_mask_fixed_priority_over_fixed_mask(self):
    """chain_mask_fixed should take priority over fixed_mask complement."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    # chain_mask_fixed: positions 2, 5, 8 are designable (1.0)
    chain_mask_fixed = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0.], dtype=np.float32)
    # fixed_mask: positions 3, 7 are fixed (would give complement chain_mask at those positions)
    fixed_mask = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        chain_mask_fixed=chain_mask_fixed,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]
    # Should be the verbatim chain_mask_fixed, not derived from fixed_mask
    expected_mask = chain_mask_fixed[None, :].repeat(batch_size, axis=0)
    np.testing.assert_array_equal(np.asarray(chain_mask), expected_mask)

  def test_chain_mask_fixed_overrides_fixed_mask(self):
    """chain_mask_fixed set; fixed_mask should be ignored."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein_ligandmpnn(batch_size=batch_size, seq_len=seq_len)

    # chain_mask_fixed: explicit designability mask
    chain_mask_fixed = np.array([1., 1., 1., 0., 0., 0., 1., 1., 1., 1.], dtype=np.float32)
    # fixed_mask: should be completely ignored
    fixed_mask = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)

    spec = SamplingSpecification(
        inputs=[],
        model_family="ligandmpnn",
        sidechain_conditioning=True,
        chain_mask_fixed=chain_mask_fixed,
        fixed_mask=fixed_mask,
    )

    ligand_ctx = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    chain_mask = ligand_ctx["chain_mask"]
    # Should be the chain_mask_fixed verbatim
    expected_mask = chain_mask_fixed[None, :].repeat(batch_size, axis=0)
    np.testing.assert_array_equal(np.asarray(chain_mask), expected_mask)
