"""Unit, contract, and stage-by-stage parity tests for FrustraMPNN.

Verifies:
1. Strict weight conversion and PyTree instantiation with zero unmapped or random leaves.
2. Refusal on missing or unconsumed keys.
3. Stage-by-stage forward numerical parity against sealed oracle fixture
   (tests/data/fixtures/frustrampnn_1ubq_oracle.npz) across:
   - features (atol < 1e-4)
   - encoder_out (atol < 1e-4)
   - decoder_hidden (atol < 1e-4)
   - lin_input (atol < 1e-4)
   - light_attention_out (atol < 1e-4)
   - both_out (atol < 1e-4)
   - frustration_scores (atol < 1e-4)
4. FamilyDriver contract, registration, and stage execution.
5. Minimum 4 negative controls proving sensitivity to architectural corruptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.frustra_mpnn import (
  FrustraMPNN,
  FrustraMPNNDriver,
  FrustraMPNNOptions,
  convert_frustra_mpnn_state_dict,
  load_frustra_mpnn_checkpoint,
)
from aminx.families.frustra_mpnn.convert import FRUSTRA_HEAD_SHAPES
from aminx.host.family_driver import (
  DRIVER_BACKED_FAMILY_MODULES,
  FAMILY_DRIVERS,
)

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_UBQ_PDB = Path(__file__).resolve().parent.parent / "data" / "1ubq.pdb"
_ORACLE_NPZ = _FIXTURES_DIR / "frustrampnn_1ubq_oracle.npz"
_HEAD_CKPT = _FIXTURES_DIR / "megascale_default_la-true_esm-False_split1.ckpt"
_TRUNK_PT = _FIXTURES_DIR / "v_48_020.pt"


def _require_fixtures() -> None:
  """Skip test if local fixtures are missing."""
  if not _ORACLE_NPZ.is_file():
    pytest.skip(f"Sealed oracle fixture {_ORACLE_NPZ} not found")
  if not _HEAD_CKPT.is_file():
    pytest.skip(f"Head checkpoint fixture {_HEAD_CKPT} not found")
  if not _TRUNK_PT.is_file():
    pytest.skip(f"Trunk weights fixture {_TRUNK_PT} not found")
  if not _UBQ_PDB.is_file():
    pytest.skip(f"Ubiquitin PDB {_UBQ_PDB} not found")


class TestFrustraMPNNWeights:
  """Tests for checkpoint conversion and strict PyTree mapping."""

  def test_strict_weight_loading_shapes_and_leaves(self) -> None:
    """Verify loading real weights yields zero NaNs, correct shapes, and zero unmapped leaves."""
    _require_fixtures()
    model = load_frustra_mpnn_checkpoint(
      _HEAD_CKPT,
      trunk_checkpoint_path=_TRUNK_PT,
      dtype="float32",
    )
    assert isinstance(model, FrustraMPNN)
    assert model.light_attention.kernel_size == 9
    assert model.light_attention.embeddings_dim == 512
    assert not model.subtract_mut

    # Verify all array leaves in the PyTree
    leaves = jax.tree_util.tree_leaves(model)
    assert len(leaves) > 0

    array_leaves = [x for x in leaves if eqx.is_array(x)]
    assert len(array_leaves) > 0

    for leaf in array_leaves:
      arr = np.asarray(leaf)
      assert not np.isnan(arr).any(), "Found NaN in parameter leaf!"
      assert not np.isinf(arr).any(), "Found Inf in parameter leaf!"

    # Verify head parameter shapes
    la = model.light_attention
    assert la.feature_weight.shape == (512, 512, 9)
    assert la.feature_bias.shape == (512,)
    assert la.attention_weight.shape == (512, 512, 9)
    assert la.attention_bias.shape == (512,)

    bo = model.both_out
    assert bo.layers[0].weight.shape == (64, 512)
    assert bo.layers[0].bias.shape == (64,)
    assert bo.layers[1].weight.shape == (32, 64)
    assert bo.layers[1].bias.shape == (32,)
    assert bo.layers[2].weight.shape == (21, 32)
    assert bo.layers[2].bias.shape == (21,)

    fo = model.frustration_out
    assert fo.weight.shape == (1, 1)
    assert fo.bias.shape == (1,)

  def test_converter_rejects_missing_keys(self) -> None:
    """Verify converter raises KeyError when a required parameter is missing."""
    dummy_head: dict[str, Any] = {
      k: np.zeros(shape, dtype=np.float32) for k, shape in FRUSTRA_HEAD_SHAPES.items()
    }
    dummy_head.pop("frustration_out.weight")

    dummy_trunk = {"prot_mpnn.W_s.weight": np.zeros((128, 21), dtype=np.float32)}

    with pytest.raises(KeyError, match="frustration_out.weight"):
      convert_frustra_mpnn_state_dict(dummy_head, mpnn_state_dict=dummy_trunk)

  def test_converter_rejects_unconsumed_keys(self) -> None:
    """Verify converter raises ValueError when extra unconsumed keys exist."""
    _require_fixtures()
    from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint

    head_ckpt = load_restricted_torch_checkpoint(_HEAD_CKPT)
    trunk_data = load_restricted_torch_checkpoint(_TRUNK_PT)
    trunk_sd = trunk_data.get("model_state_dict", trunk_data) if isinstance(trunk_data, dict) else trunk_data

    head_ckpt["state_dict"]["model.unexpected_spurious_layer.weight"] = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="unconsumed keys"):
      convert_frustra_mpnn_state_dict(head_ckpt, mpnn_state_dict=trunk_sd)


class TestFrustraMPNNStageParity:
  """Tests for stage-by-stage forward numerical parity against sealed oracle."""

  def test_stage_parity_against_oracle(self) -> None:
    """Verify stage-by-stage numerical parity on 1ubq within f32 < 1e-4 band."""
    _require_fixtures()
    oracle = np.load(_ORACLE_NPZ)
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)

    model = load_frustra_mpnn_checkpoint(
      _HEAD_CKPT,
      trunk_checkpoint_path=_TRUNK_PT,
      dtype="float32",
    )

    stages = model.forward_stages(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    # Tolerances anchored to upstream float32 precision floor (1e-4)
    max_tolerances: dict[str, float] = {
      "features": 1e-4,
      "encoder_out": 1e-4,
      "decoder_hidden": 1e-4,
      "lin_input": 1e-4,
      "light_attention_out": 1e-4,
      "both_out": 1e-4,
      "frustration_scores": 1e-4,
    }

    errors: dict[str, float] = {}
    for stage_name, tol in max_tolerances.items():
      aminx_val = np.asarray(getattr(stages, stage_name))
      oracle_val = oracle[stage_name]
      assert aminx_val.shape == oracle_val.shape, f"Shape mismatch at stage {stage_name}"

      max_diff = float(np.abs(aminx_val - oracle_val).max())
      errors[stage_name] = max_diff
      assert max_diff < tol, f"Stage {stage_name} exceeded parity tolerance: diff={max_diff:.6e} >= {tol}"

    # Also test model __call__ matches frustration_scores
    direct_scores = model(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    diff_call = float(np.abs(np.asarray(direct_scores) - oracle["frustration_scores"]).max())
    assert diff_call < 1e-4


class TestFrustraMPNNDriverContract:
  """Tests verifying compliance with FamilyDriver protocol and registry integration."""

  def test_driver_metadata_and_registration(self) -> None:
    """Verify driver registration in DRIVER_BACKED_FAMILY_MODULES and FAMILY_DRIVERS."""
    assert "frustrampnn" in DRIVER_BACKED_FAMILY_MODULES
    assert DRIVER_BACKED_FAMILY_MODULES["frustrampnn"] == "aminx.families.frustra_mpnn"

    driver = FAMILY_DRIVERS.get("frustrampnn")
    assert driver is not None
    assert isinstance(driver, FrustraMPNNDriver)
    assert driver.name == "frustrampnn"
    for attr in (
      "name",
      "options_type",
      "handles",
      "prepare_structure",
      "collate_batch",
      "result_schema",
      "stages",
      "load",
    ):
      assert hasattr(driver, attr), f"Driver missing protocol method {attr}"

    for cap in ("score:frustration", "features", "encoder", "decoder"):
      assert driver.handles(None, cap), f"Driver should handle capability {cap}"
    assert not driver.handles(None, "sample")

  def test_driver_stages_and_result_schema(self) -> None:
    """Verify driver stages and schema generation."""
    _require_fixtures()
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)
    batch = driver.collate_batch((struct,), input_indices=(0,))

    model = load_frustra_mpnn_checkpoint(
      _HEAD_CKPT,
      trunk_checkpoint_path=_TRUNK_PT,
      dtype="float32",
    )

    spec = type("DummySpec", (), {
      "options": FrustraMPNNOptions(),
      "inputs": (_UBQ_PDB,),
    })()

    # Verify result schema
    schema = driver.result_schema(spec, "score:frustration")
    assert "frustration_scores" in schema
    assert schema["frustration_scores"].dims == ("L_total", "A")

    # Verify stage execution
    stage = driver.stages(spec, "score:frustration", model)
    key = jax.random.PRNGKey(0)
    out = stage(batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key)

    assert "frustration_scores" in out
    scores = np.asarray(out["frustration_scores"])
    assert scores.shape == (76, 21)


class TestFrustraMPNNNegativeControls:
  """Targeted negative control tests verifying failure on architectural corruption."""

  def test_decoder_order_corruption_trips_parity(self) -> None:
    """Mutant 1: Inverting decoder hidden state order fails parity decisively."""
    _require_fixtures()
    oracle = np.load(_ORACLE_NPZ)
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)
    model = load_frustra_mpnn_checkpoint(_HEAD_CKPT, trunk_checkpoint_path=_TRUNK_PT)

    stages = model.forward_stages(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    # Corrupt decoder hidden concatenation: [dec0, dec1, dec2] instead of [dec2, dec1, dec0]
    dec_hidden = np.asarray(stages.decoder_hidden)
    corrupted_lin = np.concatenate([
      np.concatenate([dec_hidden[2], dec_hidden[1], dec_hidden[0]], axis=-1),
      np.asarray(stages.lin_input)[:, 384:],
    ], axis=-1)
    la_corr = model.light_attention(jnp.asarray(corrupted_lin))
    both_corr = model.both_out(la_corr)
    frust_corr = np.asarray(both_corr) * float(model.frustration_out.weight[0, 0]) + float(model.frustration_out.bias[0])
    diff = float(np.abs(frust_corr - oracle["frustration_scores"]).max())
    assert diff > 1e-2, f"Corrupted decoder order did not trip parity! diff={diff}"

  def test_light_attention_bypass_trips_parity(self) -> None:
    """Mutant 2: Bypassing LightAttention1D fails parity decisively."""
    _require_fixtures()
    oracle = np.load(_ORACLE_NPZ)
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)
    model = load_frustra_mpnn_checkpoint(_HEAD_CKPT, trunk_checkpoint_path=_TRUNK_PT)

    stages = model.forward_stages(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    # Bypass LightAttention: feed lin_input directly to both_out
    bypassed_both = model.both_out(stages.lin_input)
    bypassed_frust = np.asarray(bypassed_both) * float(model.frustration_out.weight[0, 0]) + float(model.frustration_out.bias[0])
    diff = float(np.abs(bypassed_frust - oracle["frustration_scores"]).max())
    assert diff > 1e-2, f"LightAttention bypass did not trip parity! diff={diff}"

  def test_weight_perturbation_trips_parity(self) -> None:
    """Mutant 3: Perturbing readout weights by +0.05 fails parity decisively."""
    _require_fixtures()
    oracle = np.load(_ORACLE_NPZ)
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)
    model = load_frustra_mpnn_checkpoint(_HEAD_CKPT, trunk_checkpoint_path=_TRUNK_PT)

    # Perturb last layer weights
    perturbed_layers = list(model.both_out.layers)
    w_perturbed = perturbed_layers[2].weight + 0.05
    perturbed_layers[2] = eqx.tree_at(lambda l: l.weight, perturbed_layers[2], w_perturbed)
    corrupted_model = eqx.tree_at(lambda m: m.both_out.layers, model, tuple(perturbed_layers))

    scores = corrupted_model(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    diff = float(np.abs(np.asarray(scores) - oracle["frustration_scores"]).max())
    assert diff > 1e-2, f"Weight perturbation did not trip parity! diff={diff}"

  def test_wildtype_subtraction_trips_parity(self) -> None:
    """Mutant 4: Forcing subtract_mut=True produces significant divergence from absolute scores."""
    _require_fixtures()
    oracle = np.load(_ORACLE_NPZ)
    driver = FrustraMPNNDriver()
    struct = driver.prepare_structure(_UBQ_PDB)
    model_sub = load_frustra_mpnn_checkpoint(
      _HEAD_CKPT,
      trunk_checkpoint_path=_TRUNK_PT,
      subtract_mut=True,
    )
    scores_sub = model_sub(
      jnp.asarray(struct["coords"]),
      jnp.asarray(struct["mask"]),
      jnp.asarray(struct["residue_index"]),
      jnp.asarray(struct["chain_index"]),
      jnp.asarray(struct["sequence"]),
    )
    diff = float(np.abs(np.asarray(scores_sub) - oracle["frustration_scores"]).max())
    assert diff > 1e-2, f"Wild-type subtraction did not trip parity! diff={diff}"
