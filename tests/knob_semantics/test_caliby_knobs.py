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

from types import SimpleNamespace

from aminx.families.caliby.driver import (
  CalibyPrepared,
  _CalibyStages,
  _checkpoint_id,
  _potts_21_in_aminx_order,
)
from aminx.families.caliby.model import CalibyModel
from aminx.families.caliby.potts import aggregate_ensemble_potts_params, generate_gaussian_ensemble
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


def _sample(options: CalibyOptions, num_samples: int, temperature: float = 0.01) -> np.ndarray:
  """Run the real ``_CalibyStages`` sample path on the toy model and return the sequences array."""
  prep = _toy_caliby_fixture(6)
  spec = SimpleNamespace(
    caliby=options, random_seed=3, num_samples=num_samples, temperature=temperature, sequences_to_score=None
  )
  stages = _CalibyStages(_toy_potts_model(), spec, "sample")
  batch = SimpleNamespace(arrays={"prepared": (prep,)})
  out = stages(batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=jax.random.PRNGKey(0))
  return np.asarray(out["sequences"])


def test_knob_semantics_num_samples() -> None:
  """Upstream num_seqs_per_pdb (atom_mpnn_denoiser.py:227) draws N sequences; aminx num_samples is the N_samples axis."""
  for n in (1, 3, 7):
    seqs = _sample(CalibyOptions(), n)
    assert seqs.shape == (1, n, 1, 6), seqs.shape  # (structures, N_samples, D_temp, L)


def test_knob_semantics_ensemble_conformers() -> None:
  """ensemble_conformers: C conformers, averaged Potts parameters (upstream _aggregate_potts_params, mean)."""
  xyz = jnp.asarray(_toy_caliby_fixture(6).atom23_xyz)
  mask = jnp.asarray(_toy_caliby_fixture(6).atom23_mask)
  for c in (2, 5):
    ens = generate_gaussian_ensemble(xyz, mask, num_conformers=c, noise_std=0.3, key=jax.random.PRNGKey(0))
    assert ens.shape[0] == c
    # Documented divergence (TBD-CALIBY-5): conformer 0 is the unperturbed structure here; upstream
    # noises every conformer (atom_mpnn_denoiser.py:571).
    np.testing.assert_array_equal(np.asarray(ens[0]), np.asarray(xyz))
    assert float(jnp.max(jnp.abs(ens[1:] - xyz[None]))) > 0.0
  # Aggregation is the arithmetic mean over conformers, as upstream's use_mean=True.
  h = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 21))
  j = jax.random.normal(jax.random.PRNGKey(2), (4, 6, 6, 21, 21))
  h_m, j_m = aggregate_ensemble_potts_params(h, j)
  np.testing.assert_allclose(np.asarray(h_m), np.asarray(h).mean(0), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np.asarray(j_m), np.asarray(j).mean(0), rtol=1e-6, atol=1e-6)
  # ensemble_conformers == 1 disables the ensemble path in the driver (single model call).
  model, prep = _toy_potts_model(), _toy_caliby_fixture(6)
  a = _potts_21_in_aminx_order(model, prep, CalibyOptions(ensemble_conformers=1, ensemble_noise_std=0.5), seed=1)
  b = _potts_21_in_aminx_order(model, prep, CalibyOptions(), seed=1)
  np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_knob_semantics_ensemble_noise_std() -> None:
  """ensemble_noise_std is the std (Angstrom) of the coordinate noise; 0 disables (upstream :157)."""
  n = 300
  xyz = jnp.zeros((n, 23, 3), dtype=jnp.float32)
  mask = jnp.ones((n, 23), dtype=jnp.float32)
  for std in (0.1, 0.5, 2.0):
    ens = generate_gaussian_ensemble(xyz, mask, num_conformers=2, noise_std=std, key=jax.random.PRNGKey(7))
    emp = float(jnp.std(ens[1]))
    assert abs(emp - std) < 0.05 * std, (std, emp)
  masked = generate_gaussian_ensemble(xyz, mask.at[:, 5:].set(0.0), num_conformers=2, noise_std=1.0, key=jax.random.PRNGKey(7))
  assert float(jnp.max(jnp.abs(masked[1][:, 5:]))) == 0.0, "absent atoms must stay unperturbed"
  model, prep = _toy_potts_model(), _toy_caliby_fixture(6)
  off = _potts_21_in_aminx_order(model, prep, CalibyOptions(ensemble_conformers=3, ensemble_noise_std=0.0), seed=1)
  base = _potts_21_in_aminx_order(model, prep, CalibyOptions(), seed=1)
  np.testing.assert_array_equal(np.asarray(off[0]), np.asarray(base[0]))


def test_knob_semantics_omit_gap_penalty() -> None:
  """omit_gap_penalty keeps the GAP state (index 20) out of sampled sequences; 0 lets it compete."""
  seqs = _sample(CalibyOptions(omit_gap_penalty=1e6), 40, temperature=1.0)
  assert (seqs == 20).sum() == 0
  free = _sample(CalibyOptions(omit_gap_penalty=0.0), 40, temperature=1.0)
  assert free.shape == seqs.shape
  assert (free == 20).sum() > 0, "with no penalty and temperature 1 the gap state must be reachable"


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
