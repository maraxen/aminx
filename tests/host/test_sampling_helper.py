"""Tests for _sampling_helper._prepare_fixed_controls and _prepare_ligand_context functions.

Focus: Fixed mask application correctness, no double-application bug; and that
model_family (checkpoint_id-derived per run/specs.py::RunSpecification.__post_init__) is
what actually gates real ligand/sidechain-atom injection.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import (
    _canonical_structure_ids_for_spec,
    _load_ligand_context_file,
    _prepare_fixed_controls,
    _prepare_ligand_context,
)
from aminx.run.specs import SamplingSpecification
from aminx.utils.data_structures import Protein


def _make_fake_protein(
    batch_size: int = 1, seq_len: int = 10
) -> Protein:
  """Create a minimal Protein namedtuple for testing."""
  return Protein(
      coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
      aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
      atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
      residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(
          batch_size, axis=0
      ),
      chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
      mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
      mapping=None,
  )


class TestPrepareFixedControlsFixedMask:
  """Test suite for fixed_mask application in _prepare_fixed_controls."""

  def test_float_mask_1d_shape_broadcast(self):
    """Float mask (L,) should broadcast to (batch, L) exactly once.

    Verifies: fixed_mask with shape (seq_len,) is broadcast to (batch_size, seq_len)
    without double-application.
    """
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # 1D float mask: positions 3, 5, 7 are fixed
    fixed_mask_1d = np.array([0., 0., 0., 1., 0., 1., 0., 1., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(
      inputs=[], fixed_mask=fixed_mask_1d,
      fixed_tokens=np.zeros(seq_len, dtype=np.int32),
    )

    # Call _prepare_fixed_controls
    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Verify shape
    assert fixed_mask_out.shape == (batch_size, seq_len)

    # Verify broadcast: both batches should have the same mask
    expected_mask = np.array([
        [0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
    ], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected_mask)

  def test_float_mask_2d_shape_no_broadcast(self):
    """Float mask (batch, L) should pass through without modification."""
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # 2D float mask: different mask per batch
    fixed_mask_2d = np.array(
        [
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
        ],
        dtype=np.float32,
    )
    spec = SamplingSpecification(
      inputs=[], fixed_mask=fixed_mask_2d,
      fixed_tokens=np.zeros(seq_len, dtype=np.int32),
    )

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    assert fixed_mask_out.shape == (batch_size, seq_len)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), fixed_mask_2d)

  def test_fixed_mask_and_fixed_positions_union(self):
    """fixed_mask and fixed_positions should be unioned (jnp.maximum).

    Verifies: when both are set, positions fixed by either should be 1.0 in output.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # fixed_mask: positions 2, 4 are fixed
    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    # fixed_positions: positions 4, 6, 8 are fixed (1D array means broadcast to all batches)
    fixed_positions = np.array([0., 0., 0., 0., 1., 0., 1., 0., 1., 0.], dtype=np.float32)

    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask, fixed_positions=fixed_positions, fixed_tokens=np.zeros(seq_len, dtype=np.int32))

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Expected: union of both (positions 2, 4, 6, 8 are fixed)
    expected = np.array([[0., 0., 1., 0., 1., 0., 1., 0., 1., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected)

  def test_idempotent_spec_application(self):
    """Applying same spec twice should produce identical result.

    Verifies: no double-application or accumulation of mask.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(
      inputs=[], fixed_mask=fixed_mask,
      fixed_tokens=np.zeros(seq_len, dtype=np.int32),
    )

    # Apply once
    fixed_mask_out1, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Apply again
    fixed_mask_out2, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Results should be identical
    np.testing.assert_array_equal(
        np.asarray(fixed_mask_out1), np.asarray(fixed_mask_out2)
    )

  def test_fixed_tokens_validation_with_mask(self):
    """Invalid fixed_tokens at masked positions should raise ValueError."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # fixed_mask: position 5 is fixed
    fixed_mask = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=np.float32)
    # fixed_tokens: position 5 has invalid token (25 > vocab size of 20)
    fixed_tokens = np.array([0, 0, 0, 0, 0, 25, 0, 0, 0, 0], dtype=np.int32)

    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask, fixed_tokens=fixed_tokens)

    with pytest.raises(ValueError, match="fixed_tokens must be in"):
      _prepare_fixed_controls(spec, batched_ensemble=protein)

  def test_no_double_application_bug(self):
    """Verify that fixed_mask is NOT applied twice (the double-application bug).

    This tests the core issue: the old code had TWO separate blocks applying fixed_mask:
    1. Lines 448-456: direct numpy broadcast
    2. Lines 485-493: via _broadcast_per_structure + jnp.maximum

    This would cause the mask to be applied twice, which could break cumulative behavior.
    The fix consolidates into a single application using _broadcast_per_structure.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # A mask that should be applied exactly once
    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(
      inputs=[], fixed_mask=fixed_mask,
      fixed_tokens=np.zeros(seq_len, dtype=np.int32),
    )

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # The mask should be broadcast and set exactly once, not accumulated
    # Before fix: mask might be (1, 10) with values potentially > 1 if applied twice
    # After fix: mask is (1, 10) with values in [0, 1] (no double-application)
    expected = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected)

    # Verify that all values are <= 1.0 (sanity check for double-application)
    assert np.all(np.asarray(fixed_mask_out) <= 1.0), "Mask values exceed 1.0 (possible double-application)"


class TestPrepareLigandContextModelFamilyGate:
  """_prepare_ligand_context gates BOTH real ligand-atom (Y/Y_t/Y_m) and real
  sidechain-atom (atom_37/atom_37_mask) injection on a single early-return keyed off
  spec.model_family == "ligandmpnn" -- not checkpoint_id, not sidechain_conditioning
  directly. A caller who set checkpoint_id to a LigandMPNN checkpoint but never touched
  model_family got that checkpoint's weights loaded correctly (get_topology_for_checkpoint
  already derives architecture from checkpoint_id for weight selection) while real
  ligand/sidechain context was silently never passed to it -- found live 2026-07-14 in
  tev_design's necklace P2 campaign (three of six model beads sampled with zero real
  ligand/sidechain atoms, no crash, no error).

  These tests exercise the REAL construction path a production caller uses -- kwargs into
  SamplingSpecification's __init__, letting __post_init__ derive model_family from
  checkpoint_id -- rather than a hand-constructed spec with model_family already set
  correctly (which is what every OTHER test touching _prepare_ligand_context does, and
  exactly the gap that let this bug ship: the unit was well-tested in isolation, the wiring
  from a realistic construction path was not).
  """

  def test_sidechain_atom_context_reaches_model_when_model_family_unset(self):
    """The exact regression: checkpoint_id set, model_family left unset,
    sidechain_conditioning=True -- atom_37/atom_37_mask must be real arrays, not None.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    spec = SamplingSpecification(
      inputs=[],
      checkpoint_id="ligandmpnn_v_32_020_25",
      sidechain_conditioning=True,
    )
    assert spec.model_family == "ligandmpnn", (
      "precondition: model_family must have been auto-derived from checkpoint_id"
    )

    ligand_context = _prepare_ligand_context(
      spec, batched_ensemble=protein, batch_size=batch_size, seq_len=seq_len,
    )

    assert ligand_context["atom_37"] is not None, (
      "sidechain_conditioning=True with a LigandMPNN checkpoint must produce real "
      "atom_37 coordinates -- got None, meaning the model_family gate silently closed."
    )
    assert ligand_context["atom_37_mask"] is not None
    np.testing.assert_array_equal(
      np.asarray(ligand_context["atom_37"]), np.asarray(protein.coordinates),
    )

  def test_sidechain_atom_context_is_none_without_sidechain_conditioning(self):
    """Sanity check on the OTHER half of the branch: even with model_family correctly
    'ligandmpnn', atom_37 stays None when sidechain_conditioning=False (default) -- this
    confirms the test above is exercising the real gate, not a fixture that always returns
    non-None regardless of configuration.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    spec = SamplingSpecification(
      inputs=[],
      checkpoint_id="ligandmpnn_v_32_020_25",
      sidechain_conditioning=False,
    )
    assert spec.model_family == "ligandmpnn"

    ligand_context = _prepare_ligand_context(
      spec, batched_ensemble=protein, batch_size=batch_size, seq_len=seq_len,
    )
    assert ligand_context["atom_37"] is None
    assert ligand_context["atom_37_mask"] is None

  def test_all_context_is_none_for_proteinmpnn_checkpoint(self):
    """Baseline: a genuinely proteinmpnn checkpoint (no ligand family at all) correctly
    gets None ligand/sidechain context regardless of sidechain_conditioning -- confirms the
    fix didn't flip this into an always-real-context bug in the other direction.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    spec = SamplingSpecification(
      inputs=[],
      checkpoint_id="proteinmpnn_v_48_020",
      sidechain_conditioning=True,
    )
    assert spec.model_family == "proteinmpnn"

    ligand_context = _prepare_ligand_context(
      spec, batched_ensemble=protein, batch_size=batch_size, seq_len=seq_len,
    )
    assert ligand_context["atom_37"] is None
    assert ligand_context["Y"] is None


def _write_keyed_npz(
    path: Path,
    payload: dict[str, dict[str, np.ndarray]],
) -> None:
  """payload: {structure_id: {"Y": arr, "Y_t": arr, "Y_m": arr}, ...}."""
  flat: dict[str, np.ndarray] = {}
  for structure_id, tensors in payload.items():
    for tensor_name, arr in tensors.items():
      flat[f"{structure_id}::{tensor_name}"] = arr
  np.savez(path, **flat)


def _make_ligand_tensors(
    rng: np.random.Generator, seq_len: int, atoms: int,
) -> dict[str, np.ndarray]:
  return {
      "Y": rng.standard_normal((seq_len, atoms, 3)).astype(np.float32),
      "Y_t": rng.integers(1, 20, size=(seq_len, atoms)).astype(np.int64),
      "Y_m": np.ones((seq_len, atoms), dtype=np.float32),
  }


class TestLoadLigandContextFileKeyedFormat:
  """Keyed npz format: '<structure_id>::Y' / '<structure_id>::Y_t' / '<structure_id>::Y_m'.

  Covers the round-trip stacking/ordering contract of _load_ligand_context_file
  (src/aminx/host/_sampling_helper.py:123-243) for the '::'-separated keying convention,
  and both its strict error paths (missing tensors, id mismatch between the npz payload
  and the canonical structure roster).
  """

  def test_round_trip_two_structures_preserves_values_and_shapes(self, tmp_path: Path):
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(0)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    tensors_b = _make_ligand_tensors(rng, seq_len, atoms)
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a, "struct_b": tensors_b})

    Y, Y_t, Y_m = _load_ligand_context_file(
        path,
        canonical_structure_ids=["struct_a", "struct_b"],
        batch_structure_ids=["struct_a", "struct_b"],
    )

    assert Y.shape == (2, seq_len, atoms, 3)
    assert Y_t.shape == (2, seq_len, atoms)
    assert Y_m.shape == (2, seq_len, atoms)
    np.testing.assert_array_equal(np.asarray(Y[0]), tensors_a["Y"])
    np.testing.assert_array_equal(np.asarray(Y[1]), tensors_b["Y"])
    np.testing.assert_array_equal(np.asarray(Y_t[0]), tensors_a["Y_t"])
    np.testing.assert_array_equal(np.asarray(Y_m[1]), tensors_b["Y_m"])

  def test_batch_structure_ids_controls_stacking_order_independent_of_canonical_order(
      self, tmp_path: Path,
  ):
    """batch_structure_ids (not canonical_structure_ids, and not npz insertion order)
    determines the order rows are stacked in -- this is the order kernel_dispatch.py's
    real per-chunk batch relies on to line up ligand context rows with structure rows.
    """
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(1)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    tensors_b = _make_ligand_tensors(rng, seq_len, atoms)
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a, "struct_b": tensors_b})

    # canonical roster order is [a, b]; this batch's row order is reversed: [b, a]
    Y, Y_t, Y_m = _load_ligand_context_file(
        path,
        canonical_structure_ids=["struct_a", "struct_b"],
        batch_structure_ids=["struct_b", "struct_a"],
    )

    np.testing.assert_array_equal(np.asarray(Y[0]), tensors_b["Y"])
    np.testing.assert_array_equal(np.asarray(Y[1]), tensors_a["Y"])

  def test_missing_tensor_key_raises_value_error(self, tmp_path: Path):
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(2)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    tensors_b = _make_ligand_tensors(rng, seq_len, atoms)
    del tensors_b["Y_m"]  # struct_b is missing its Y_m tensor
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a, "struct_b": tensors_b})

    with pytest.raises(ValueError, match="missing keyed tensors"):
      _load_ligand_context_file(
          path,
          canonical_structure_ids=["struct_a", "struct_b"],
          batch_structure_ids=None,
      )

  def test_extra_structure_id_in_npz_raises_value_error(self, tmp_path: Path):
    """npz has a structure the canonical roster doesn't -- must fail loudly, not silently
    ignore the extra data or silently accept a roster mismatch.
    """
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(3)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    tensors_b = _make_ligand_tensors(rng, seq_len, atoms)
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a, "struct_b": tensors_b})

    with pytest.raises(ValueError, match="must exactly match canonical structure IDs"):
      _load_ligand_context_file(
          path,
          canonical_structure_ids=["struct_a"],
          batch_structure_ids=None,
      )

  def test_missing_structure_id_in_npz_raises_value_error(self, tmp_path: Path):
    """canonical roster expects a structure the npz doesn't have -- must fail loudly."""
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(4)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a})

    with pytest.raises(ValueError, match="must exactly match canonical structure IDs"):
      _load_ligand_context_file(
          path,
          canonical_structure_ids=["struct_a", "struct_b"],
          batch_structure_ids=None,
      )


class TestLoadLigandContextFileLegacyFormat:
  """Legacy npz format: flat 'Y'/'Y_t'/'Y_m' arrays + a parallel 'structure_ids' array.

  _load_ligand_context_file supports this format as a fallback when no '<id>::Y'-style
  keys are found in the npz (src/aminx/host/_sampling_helper.py:213-244).
  """

  def test_round_trip_two_structures_preserves_values_and_reorders_by_batch_ids(
      self, tmp_path: Path,
  ):
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((2, seq_len, atoms, 3)).astype(np.float32)
    Y_t = rng.integers(1, 20, size=(2, seq_len, atoms)).astype(np.int64)
    Y_m = np.ones((2, seq_len, atoms), dtype=np.float32)
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        Y=Y,
        Y_t=Y_t,
        Y_m=Y_m,
        structure_ids=np.array(["struct_a", "struct_b"]),
    )

    out_Y, out_Y_t, out_Y_m = _load_ligand_context_file(
        path,
        canonical_structure_ids=["struct_a", "struct_b"],
        batch_structure_ids=["struct_b", "struct_a"],
    )

    assert out_Y.shape == (2, seq_len, atoms, 3)
    np.testing.assert_array_equal(np.asarray(out_Y[0]), Y[1])  # struct_b first
    np.testing.assert_array_equal(np.asarray(out_Y[1]), Y[0])  # struct_a second
    np.testing.assert_array_equal(np.asarray(out_Y_t[0]), Y_t[1])
    np.testing.assert_array_equal(np.asarray(out_Y_m[1]), Y_m[0])

  def test_missing_required_key_raises_value_error(self, tmp_path: Path):
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(6)
    Y = rng.standard_normal((1, seq_len, atoms, 3)).astype(np.float32)
    Y_t = rng.integers(1, 20, size=(1, seq_len, atoms)).astype(np.int64)
    path = tmp_path / "legacy_missing.npz"
    np.savez(path, Y=Y, Y_t=Y_t, structure_ids=np.array(["struct_a"]))  # Y_m omitted

    with pytest.raises(ValueError, match="missing required keys"):
      _load_ligand_context_file(path, canonical_structure_ids=None, batch_structure_ids=None)

  def test_structure_id_mismatch_raises_value_error(self, tmp_path: Path):
    seq_len, atoms = 4, 3
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((2, seq_len, atoms, 3)).astype(np.float32)
    Y_t = rng.integers(1, 20, size=(2, seq_len, atoms)).astype(np.int64)
    Y_m = np.ones((2, seq_len, atoms), dtype=np.float32)
    path = tmp_path / "legacy_mismatch.npz"
    np.savez(
        path,
        Y=Y,
        Y_t=Y_t,
        Y_m=Y_m,
        structure_ids=np.array(["struct_a", "struct_b"]),
    )

    with pytest.raises(ValueError, match="must exactly match canonical structure IDs"):
      _load_ligand_context_file(
          path,
          canonical_structure_ids=["struct_a", "struct_c"],
          batch_structure_ids=None,
      )


class TestPrepareLigandContextRequiresRealTensors:
  """_prepare_ligand_context's ligand_conditioning=True enforcement -- the exact original
  bug symptom this wiring fix addresses: a caller who asked for real ligand conditioning
  but supplied no way to get real Y/Y_t/Y_m tensors (no ligand_context_path, no Y already
  on the batch) must get a loud ValueError, not silently fall back to the zero-atom
  placeholder used for the "no ligand conditioning requested" case.
  """

  def test_ligand_conditioning_without_any_source_raises(self):
    batch_size, seq_len = 1, 4
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    spec = SamplingSpecification(
        inputs=["/tmp/struct_a.pdb"],
        checkpoint_id="ligandmpnn_v_32_020_25",
        ligand_conditioning=True,
    )
    assert spec.model_family == "ligandmpnn"
    assert spec.ligand_context_path is None

    with pytest.raises(ValueError, match="ligand_conditioning=True requires ligand context"):
      _prepare_ligand_context(spec, batched_ensemble=protein, batch_size=batch_size, seq_len=seq_len)

  def test_ligand_conditioning_with_valid_ligand_context_path_succeeds(self, tmp_path: Path):
    batch_size, seq_len, atoms = 1, 4, 3
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)
    rng = np.random.default_rng(8)
    tensors_a = _make_ligand_tensors(rng, seq_len, atoms)
    path = tmp_path / "ligand_context.npz"
    _write_keyed_npz(path, {"struct_a": tensors_a})

    spec = SamplingSpecification(
        inputs=["/tmp/struct_a.pdb"],
        checkpoint_id="ligandmpnn_v_32_020_25",
        ligand_conditioning=True,
        ligand_context_path=path,
    )
    assert spec.model_family == "ligandmpnn"
    # mirror the real caller in kernel_dispatch.py: derive canonical structure ids from
    # spec.inputs rather than relying on _load_ligand_context_file's sorted-keys fallback.
    canonical_ids = _canonical_structure_ids_for_spec(spec)
    assert canonical_ids == ["struct_a"]

    ligand_context = _prepare_ligand_context(
        spec,
        batched_ensemble=protein,
        batch_size=batch_size,
        seq_len=seq_len,
        canonical_structure_ids=canonical_ids,
        batch_structure_ids=canonical_ids,
    )

    assert ligand_context["Y"] is not None
    assert ligand_context["Y_t"] is not None
    assert ligand_context["Y_m"] is not None
    # #3510: the atom axis is end-padded with mask-0 atoms to the ligand atom rung.
    from aminx.host.bucketing import ligand_atom_rung

    assert ligand_context["Y"].shape == (batch_size, seq_len, ligand_atom_rung(atoms), 3)
    np.testing.assert_array_equal(np.asarray(ligand_context["Y"][0])[:, :atoms], tensors_a["Y"])
    np.testing.assert_array_equal(np.asarray(ligand_context["Y_t"][0])[:, :atoms], tensors_a["Y_t"])
    np.testing.assert_array_equal(np.asarray(ligand_context["Y_m"][0])[:, :atoms], tensors_a["Y_m"])
    assert not np.asarray(ligand_context["Y_m"][0])[:, atoms:].any()
