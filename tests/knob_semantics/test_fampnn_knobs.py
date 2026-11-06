# ruff: noqa: S101
"""Unit tests for FAMPNN knob semantics (spec 261030 Phase 4).

Covers all 9 knobs on ``FAMPNNOptions`` plus ``temperature``:
- checkpoint_id
- num_seq_steps
- num_scn_steps
- sigma_min
- sigma_max
- rho
- s_churn
- step_scale
- pack_each_step
- temperature
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.families.fampnn.driver import _checkpoint_id, mar_sample
from aminx.families.fampnn.model import FAMPNNModel
from aminx.run.options import FAMPNNOptions
from aminx.run.specs import SamplingSpecification


def _toy_fixture(num_l: int = 6) -> dict[str, jax.Array]:
  """Construct a minimal synthetic 6-residue backbone fixture."""
  res_idx = jnp.arange(1, num_l + 1, dtype=jnp.int32)
  chain_idx = jnp.zeros((num_l,), dtype=jnp.int32)
  mask = jnp.ones((num_l,), dtype=jnp.float32)

  # Synthetic C-alpha backbone spacing ~3.8 Angstroms
  atom37_xyz = jnp.zeros((num_l, 37, 3), dtype=jnp.float32)
  x_coords = jnp.arange(num_l, dtype=jnp.float32) * 3.8
  atom37_xyz = atom37_xyz.at[:, 1, 0].set(x_coords)
  atom37_xyz = atom37_xyz.at[:, 0, 0].set(x_coords - 1.2)
  atom37_xyz = atom37_xyz.at[:, 2, 0].set(x_coords + 1.2)
  atom37_xyz = atom37_xyz.at[:, 4, 1].set(1.2)

  atom37_mask = jnp.zeros((num_l, 37), dtype=jnp.float32)
  atom37_mask = atom37_mask.at[:, [0, 1, 2, 4]].set(1.0)
  aatype = jnp.zeros((num_l,), dtype=jnp.int32)
  seq_mask = jnp.zeros((num_l,), dtype=jnp.float32)

  return {
    "atom37_xyz": atom37_xyz,
    "atom37_mask": atom37_mask,
    "mask": mask,
    "residue_idx": res_idx,
    "chain_idx": chain_idx,
    "aatype": aatype,
    "seq_mask": seq_mask,
  }


def _toy_model() -> FAMPNNModel:
  """Construct a minimal FAMPNNModel for fast semantic testing."""
  return FAMPNNModel(
    hidden_dim=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    num_gvp_layers=1,
    scn_depth=1,
    has_confidence=False,
    key=jax.random.PRNGKey(42),
  )


def test_knob_semantics_fampnn_checkpoint_id() -> None:
  """FAMPNN checkpoint_id resolution and conflict checking."""
  # Default when neither is set
  spec_default = SamplingSpecification(inputs="test.pdb")
  assert _checkpoint_id(spec_default) == "fampnn_0_3"

  # Spec-level checkpoint_id overrides default
  spec_top = SamplingSpecification(inputs="test.pdb", checkpoint_id="fampnn_0_3_cath")
  assert _checkpoint_id(spec_top) == "fampnn_0_3_cath"

  # Options-level checkpoint_id overrides default
  opt = FAMPNNOptions(checkpoint_id="fampnn_0_0")
  spec_opt = SamplingSpecification(inputs="test.pdb", model_family="fampnn", fampnn=opt)
  assert _checkpoint_id(spec_opt) == "fampnn_0_0"

  # Agreeing spec-level and options-level succeeds
  spec_agree = SamplingSpecification(inputs="test.pdb", model_family="fampnn", checkpoint_id="fampnn_0_3", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_3"))
  assert _checkpoint_id(spec_agree) == "fampnn_0_3"

  # Conflicting spec-level and options-level raises ValueError
  spec_conflict = SamplingSpecification(inputs="test.pdb", model_family="fampnn", checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_3"))
  with pytest.raises(ValueError, match="conflicting FAMPNN checkpoint ids"):
    _checkpoint_id(spec_conflict)


def test_knob_semantics_num_seq_steps() -> None:
  """num_seq_steps controls the MAR sequence refinement schedule length."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(101)

  seq_2, _, _, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=2,
    key=key,
  )

  seq_6, _, _, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=6,
    num_scn_steps=2,
    key=key,
  )

  assert not jnp.array_equal(seq_2, seq_6), "num_seq_steps failed to alter sequence sampling trajectory"


def test_knob_semantics_num_scn_steps() -> None:
  """num_scn_steps controls the EDM sidechain diffusion integration step count."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(202)

  _, _, xyz_2, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=2,
    key=key,
  )

  _, _, xyz_8, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=8,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_2 - xyz_8))
  assert diff > 1e-4, f"num_scn_steps failed to alter sidechain coordinates (diff={diff})"


def test_knob_semantics_sigma_min() -> None:
  """sigma_min sets the minimum noise floor in EDM integration."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(303)

  _, _, xyz_low, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    sigma_min=0.002,
    key=key,
  )

  _, _, xyz_high, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    sigma_min=1.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_low - xyz_high))
  assert diff > 1e-3, f"sigma_min failed to alter diffused sidechain coordinates (diff={diff})"


def test_knob_semantics_sigma_max() -> None:
  """sigma_max sets the initial noise ceiling in EDM integration."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(404)

  _, _, xyz_80, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    sigma_max=80.0,
    key=key,
  )

  _, _, xyz_5, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    sigma_max=5.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_80 - xyz_5))
  assert diff > 1e-3, f"sigma_max failed to alter diffused sidechain coordinates (diff={diff})"


def test_knob_semantics_rho() -> None:
  """rho sets the Karras schedule power-law curvature exponent."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(505)

  _, _, xyz_rho7, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    rho=7.0,
    key=key,
  )

  _, _, xyz_rho1, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    rho=1.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_rho7 - xyz_rho1))
  assert diff > 1e-4, f"rho failed to alter EDM noise schedule trajectory (diff={diff})"


def test_knob_semantics_s_churn() -> None:
  """s_churn injects stochastic noise at each diffusion step."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(606)

  _, _, xyz_no_churn, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    s_churn=0.0,
    key=key,
  )

  _, _, xyz_churn, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    s_churn=5.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_no_churn - xyz_churn))
  assert diff > 1e-4, f"s_churn failed to inject stochastic noise into sampling (diff={diff})"


def test_knob_semantics_step_scale() -> None:
  """step_scale scales the Heun velocity update vector in EDM integration."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(707)

  _, _, xyz_scale15, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    step_scale=1.5,
    key=key,
  )

  _, _, xyz_scale05, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=2,
    num_scn_steps=4,
    step_scale=0.5,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_scale15 - xyz_scale05))
  assert diff > 1e-4, f"step_scale failed to alter velocity step updates (diff={diff})"


def test_knob_semantics_pack_each_step() -> None:
  """pack_each_step toggles intermediate sidechain diffusion during MAR unmasking."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(808)

  seq_pack, _, xyz_pack, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=3,
    num_scn_steps=2,
    pack_each_step=True,
    key=key,
  )

  seq_nopack, _, xyz_nopack, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=3,
    num_scn_steps=2,
    pack_each_step=False,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_pack - xyz_nopack))
  assert diff > 1e-4, f"pack_each_step failed to alter sidechain packing behavior (diff={diff})"


def test_knob_semantics_temperature() -> None:
  """temperature scales categorical logits during sequence sampling."""
  model = _toy_model()
  fx = _toy_fixture(6)
  key = jax.random.PRNGKey(909)

  seq_cold, _, _, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=3,
    num_scn_steps=2,
    temperature=0.01,
    key=key,
  )

  seq_hot, _, _, _ = mar_sample(
    model,
    fx["atom37_xyz"],
    fx["atom37_mask"],
    fx["aatype"],
    fx["seq_mask"],
    fx["mask"],
    fx["residue_idx"],
    fx["chain_idx"],
    num_seq_steps=3,
    num_scn_steps=2,
    temperature=10.0,
    key=key,
  )

  assert not jnp.array_equal(seq_cold, seq_hot), "temperature failed to alter sequence sampling randomness"
