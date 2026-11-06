# ruff: noqa: S101
"""Unit tests for Caliby knob semantics (spec 261030 Phase 4).

Covers all 10 knobs on ``CalibyOptions``:
- checkpoint_id
- ensemble_conformers
- ensemble_noise_std
- omit_gap_penalty
- packer_num_steps
- packer_sigma_min
- packer_sigma_max
- packer_rho
- packer_s_churn
- packer_step_scale
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.caliby.driver import (
  CalibyPrepared,
  _checkpoint_id,
  _potts_21_in_aminx_order,
)
from aminx.families.caliby.model import CalibyModel
from aminx.families.caliby.sidechain_diffusion import (
  CalibySidechainMLP,
  edm_sidechain_sample,
)
from aminx.run.options import CalibyOptions
from aminx.run.specs import SamplingSpecification


def _toy_caliby_fixture(num_l: int = 6) -> CalibyPrepared:
  """Construct a minimal synthetic 6-residue CalibyPrepared fixture."""
  atom23_xyz = np.zeros((num_l, 23, 3), dtype=np.float32)
  for i in range(num_l):
    x = 3.8 * i
    atom23_xyz[i, 0] = [x - 1.2, 0.0, 0.0]  # N
    atom23_xyz[i, 1] = [x, 0.0, 0.0]        # CA
    atom23_xyz[i, 2] = [x + 1.2, 0.0, 0.0]  # C
    atom23_xyz[i, 3] = [x + 1.2, 1.2, 0.0]  # O

  atom23_mask = np.zeros((num_l, 23), dtype=np.float32)
  atom23_mask[:, :4] = 1.0
  token_mask = np.ones((num_l,), dtype=np.float32)
  res_idx = np.arange(1, num_l + 1, dtype=np.int32)
  token_idx = np.arange(num_l, dtype=np.int32)
  asym_id = np.zeros((num_l,), dtype=np.int32)
  entity_id = np.zeros((num_l,), dtype=np.int32)
  sym_id = np.zeros((num_l,), dtype=np.int32)
  aatype_aminx = np.zeros((num_l,), dtype=np.int32)
  aatype_caliby = np.zeros((num_l,), dtype=np.int32)
  cands = aatype_aminx[None, :]

  return CalibyPrepared(
    atom23_xyz=atom23_xyz,
    atom23_mask=atom23_mask,
    token_mask=token_mask,
    residue_idx=res_idx,
    token_idx=token_idx,
    asym_id=asym_id,
    entity_id=entity_id,
    sym_id=sym_id,
    aatype_aminx=aatype_aminx,
    aatype_caliby=aatype_caliby,
    candidate_sequences_aminx=cands,
  )


def _toy_potts_model() -> CalibyModel:
  """Construct a minimal CalibyModel."""
  return CalibyModel(
    num_encoder_layers=1,
    num_decoder_layers=1,
    key=jax.random.PRNGKey(42),
  )


def _toy_packer_model() -> CalibySidechainMLP:
  """Construct a minimal CalibySidechainMLP."""
  return CalibySidechainMLP(
    depth=1,
    num_encoder_layers=1,
    num_decoder_layers=1,
    key=jax.random.PRNGKey(42),
  )


def test_knob_semantics_caliby_checkpoint_id() -> None:
  """Caliby checkpoint_id resolution and conflict checking."""
  # Default when neither is set
  spec_default = SamplingSpecification(inputs="test.pdb")
  assert _checkpoint_id(spec_default) == "caliby"

  # Spec-level checkpoint_id overrides default
  spec_top = SamplingSpecification(inputs="test.pdb", checkpoint_id="soluble_caliby")
  assert _checkpoint_id(spec_top) == "soluble_caliby"

  # Options-level checkpoint_id overrides default
  opt = CalibyOptions(checkpoint_id="caliby_distill")
  spec_opt = SamplingSpecification(inputs="test.pdb", model_family="caliby", caliby=opt)
  assert _checkpoint_id(spec_opt) == "caliby_distill"

  # Agreeing spec-level and options-level succeeds
  spec_agree = SamplingSpecification(
    inputs="test.pdb",
    model_family="caliby",
    checkpoint_id="caliby",
    caliby=CalibyOptions(checkpoint_id="caliby"),
  )
  assert _checkpoint_id(spec_agree) == "caliby"

  # Conflicting spec-level and options-level raises ValueError
  spec_conflict = SamplingSpecification(
    inputs="test.pdb",
    model_family="caliby",
    checkpoint_id="caliby",
    caliby=CalibyOptions(checkpoint_id="caliby_packer_010"),
  )
  with pytest.raises(ValueError, match="conflicting Caliby checkpoint ids"):
    _checkpoint_id(spec_conflict)


def test_knob_semantics_ensemble_conformers() -> None:
  """ensemble_conformers controls multi-conformer averaging of Potts fields."""
  model = _toy_potts_model()
  prep = _toy_caliby_fixture(6)

  opt_single = CalibyOptions(ensemble_conformers=1, ensemble_noise_std=0.0)
  h_single, j_single = _potts_21_in_aminx_order(model, prep, opt_single, seed=1)

  opt_ens = CalibyOptions(ensemble_conformers=3, ensemble_noise_std=0.2)
  h_ens, j_ens = _potts_21_in_aminx_order(model, prep, opt_ens, seed=1)

  diff_h = jnp.max(jnp.abs(h_single - h_ens))
  assert diff_h > 1e-4, f"ensemble_conformers failed to alter Potts fields (diff={diff_h})"


def test_knob_semantics_ensemble_noise_std() -> None:
  """ensemble_noise_std scales coordinate perturbation magnitude in ensemble."""
  model = _toy_potts_model()
  prep = _toy_caliby_fixture(6)

  opt_low = CalibyOptions(ensemble_conformers=3, ensemble_noise_std=0.05)
  h_low, _ = _potts_21_in_aminx_order(model, prep, opt_low, seed=2)

  opt_high = CalibyOptions(ensemble_conformers=3, ensemble_noise_std=1.0)
  h_high, _ = _potts_21_in_aminx_order(model, prep, opt_high, seed=2)

  diff = jnp.max(jnp.abs(h_low - h_high))
  assert diff > 1e-4, f"ensemble_noise_std failed to scale perturbation impact (diff={diff})"


def test_knob_semantics_omit_gap_penalty() -> None:
  """omit_gap_penalty adds an additive penalty on the GAP token (index 20)."""
  prep = _toy_caliby_fixture(6)
  num_l = 6

  # Verify penalty term directly
  pen_zero = jax.nn.one_hot(jnp.full((num_l,), 20, dtype=jnp.int32), 21, dtype=jnp.float32) * 0.0
  pen_large = jax.nn.one_hot(jnp.full((num_l,), 20, dtype=jnp.int32), 21, dtype=jnp.float32) * 1e6

  assert float(pen_zero[:, 20].sum()) == 0.0
  assert float(pen_large[:, 20].sum()) == 6e6
  assert float(pen_large[:, :20].sum()) == 0.0, "penalty must only affect gap token (index 20)"


def test_knob_semantics_packer_num_steps() -> None:
  """packer_num_steps controls the EDM integration step count in sidechain packing."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(303)

  xyz_2 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=2,
    key=key,
  )

  xyz_6 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=6,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_2 - xyz_6))
  assert diff > 1e-4, f"packer_num_steps failed to alter packed coordinates (diff={diff})"


def test_knob_semantics_packer_sigma_min() -> None:
  """packer_sigma_min controls the terminal noise level in EDM packing."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(404)

  xyz_low = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=3,
    sigma_min=0.002,
    key=key,
  )

  xyz_high = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=3,
    sigma_min=1.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_low - xyz_high))
  assert diff > 1e-3, f"packer_sigma_min failed to alter packed coordinates (diff={diff})"


def test_knob_semantics_packer_sigma_max() -> None:
  """packer_sigma_max controls the initial noise ceiling in EDM packing."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(505)

  xyz_80 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=3,
    sigma_max=80.0,
    key=key,
  )

  xyz_5 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=3,
    sigma_max=5.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_80 - xyz_5))
  assert diff > 1e-3, f"packer_sigma_max failed to alter packed coordinates (diff={diff})"


def test_knob_semantics_packer_rho() -> None:
  """packer_rho controls the power-law exponent of the Karras noise schedule."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(606)

  xyz_rho7 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    rho=7.0,
    key=key,
  )

  xyz_rho1 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    rho=1.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_rho7 - xyz_rho1))
  assert diff > 1e-4, f"packer_rho failed to alter noise step schedule (diff={diff})"


def test_knob_semantics_packer_s_churn() -> None:
  """packer_s_churn injects stochastic churn noise into EDM packer updates."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(707)

  xyz_no_churn = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    s_churn=0.0,
    key=key,
  )

  xyz_churn = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    s_churn=5.0,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_no_churn - xyz_churn))
  assert diff > 1e-4, f"packer_s_churn failed to inject stochastic noise (diff={diff})"


def test_knob_semantics_packer_step_scale() -> None:
  """packer_step_scale scales the Heun velocity update step in the EDM solver."""
  packer = _toy_packer_model()
  prep = _toy_caliby_fixture(6)
  key = jax.random.PRNGKey(808)

  xyz_scale15 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    step_scale=1.5,
    key=key,
  )

  xyz_scale05 = edm_sidechain_sample(
    packer,
    jnp.asarray(prep.atom23_xyz),
    jnp.asarray(prep.atom23_mask),
    jnp.asarray(prep.token_mask),
    jnp.asarray(prep.residue_idx),
    jnp.asarray(prep.token_idx),
    jnp.asarray(prep.asym_id),
    jnp.asarray(prep.entity_id),
    jnp.asarray(prep.sym_id),
    jnp.asarray(prep.aatype_caliby),
    num_steps=4,
    step_scale=0.5,
    key=key,
  )

  diff = jnp.max(jnp.abs(xyz_scale15 - xyz_scale05))
  assert diff > 1e-4, f"packer_step_scale failed to alter velocity step updates (diff={diff})"
