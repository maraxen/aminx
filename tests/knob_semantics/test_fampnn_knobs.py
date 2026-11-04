# ruff: noqa: S101, PLR2004
"""FAMPNN knob semantics, one test per live alias row (tests/knob_gate/alias_map.toml, ``fampnn__*``).

Each test pins the behaviour the *upstream* code defines, citing the upstream line, on a small random-weight
``FAMPNNModel`` (real-weight, real-structure agreement is the job of the sealed-oracle vehicle
``scripts/parity/fampnn_parity.py``; these tests show that each knob reaches the sampler and does what upstream says).
Upstream is ``~/repos/fampnn`` at aaf788b; ``sd_model.py`` is ``SeqDenoiser.sample`` / ``sidechain_pack``.

The sampler takes its randomness as explicit arrays (``SamplerNoise``), so "does this knob consume noise" is a direct
question: change the array, see whether the output moves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.fampnn import sampler
from aminx.families.fampnn.driver import FAMPNNPrepared, _checkpoint_id, _FAMPNNStages
from aminx.families.fampnn.model import FAMPNNModel
from aminx.host.family_driver import FamilyBatch
from aminx.run.options import FAMPNNOptions
from aminx.run.specs import SamplingSpecification

NUM_L = 8
S_STEPS = 3
SCN_STEPS = 4


def _model() -> FAMPNNModel:
  return FAMPNNModel(
    num_encoder_layers=1,
    num_decoder_layers=1,
    use_sidechain_features=True,
    num_gvp_layers=1,
    gvp_w_out_mode="gvp",
    gvp_rotate_out=False,
    has_w_out=True,
    scn_depth=1,
    has_f_embedder=True,
    has_confidence=True,
    conf_has_upstream_heads=True,
    conf_num_bins=33,
    scn_sigma_data=0.64,
    key=jax.random.PRNGKey(7),
  )


_MODEL = _model()


class _Fx:
  """A short backbone with a few side-chain atoms: finite, frames well defined, nothing symmetric."""

  def __init__(self) -> None:
    rng = np.random.default_rng(3)
    ca = np.stack([np.arange(NUM_L) * 3.8, 1.5 * np.sin(np.arange(NUM_L)), 0.8 * np.cos(np.arange(NUM_L))], axis=-1)
    xyz = np.zeros((NUM_L, 37, 3))
    xyz[:, 1] = ca
    xyz[:, 0] = ca + np.array([-1.2, 0.6, 0.0])
    xyz[:, 2] = ca + np.array([1.3, 0.5, 0.1])
    xyz[:, 4] = xyz[:, 2] + np.array([0.0, 1.2, 0.3])
    xyz[:, 3] = ca + np.array([0.2, -1.0, 1.1])  # CB
    xyz[:, 5] = ca + np.array([0.3, -2.0, 1.9])  # CG
    xyz += rng.normal(scale=0.05, size=xyz.shape) * (np.abs(xyz).sum(-1, keepdims=True) > 0)
    mask = np.zeros((NUM_L, 37))
    mask[:, [0, 1, 2, 3, 4, 5]] = 1.0
    self.x = jnp.asarray(xyz * mask[..., None], dtype=jnp.float32)
    self.atom_mask = jnp.asarray(mask, dtype=jnp.float32)
    self.aatype = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)  # A R N D C Q E G
    self.valid = jnp.ones((NUM_L,), dtype=jnp.float32)
    self.missing = jnp.zeros((NUM_L, 37), dtype=jnp.float32)
    self.ridx = jnp.arange(1, NUM_L + 1, dtype=jnp.int32)
    self.cidx = jnp.zeros((NUM_L,), dtype=jnp.int32)


FX = _Fx()


def _noise(seed: int = 0, *, churn_seed: int | None = None, calls: int = S_STEPS + 1) -> sampler.SamplerNoise:
  rng = np.random.default_rng(seed)
  churn = None
  if churn_seed is not None:
    churn = jnp.asarray(
      np.random.default_rng(churn_seed).standard_normal((calls, SCN_STEPS, NUM_L, 33, 3)), dtype=jnp.float32,
    )
  return sampler.SamplerNoise(
    decoding_u=jnp.asarray(rng.random(NUM_L), dtype=jnp.float32),
    categorical_u=jnp.asarray(rng.random((S_STEPS, NUM_L)), dtype=jnp.float32),
    prior_z=jnp.asarray(rng.standard_normal((calls, NUM_L, 33, 3)), dtype=jnp.float32),
    churn_z=churn,
  )


def _sample(noise: sampler.SamplerNoise | None = None, **kw: object) -> sampler.MarResult:
  """``SeqDenoiser.sample`` with ``seq_design.yaml`` defaults except small step counts."""
  p: dict[str, object] = {
    "S": S_STEPS, "T": SCN_STEPS, "temperature": 0.5, "seq_only": False, "repack_last": True, "psce": None,
    "aov": None, "sov": None, "step_scale": 1.5, "s_churn": 0.0, "s_noise": 1.0, "s_t_min": 0.01, "s_t_max": 50.0,
    "seq_mode": "linear", "seq_t0": 0.0, "seq_t1": 1.0, "scn_mode": "linear", "scn_t0": 0.0, "scn_t1": 1.0,
  }  # fmt: skip
  p.update(kw)
  ts = sampler.timestep_schedule(str(p["seq_mode"]), int(p["S"]), float(p["seq_t0"]), float(p["seq_t1"]))  # type: ignore[arg-type]
  ts_scn = sampler.timestep_schedule(str(p["scn_mode"]), int(p["T"]), float(p["scn_t0"]), float(p["scn_t1"]))  # type: ignore[arg-type]
  return sampler.mar_sample_upstream(
    _MODEL, FX.x, FX.aatype, FX.valid, FX.missing, FX.ridx, FX.cidx, ts, noise or _noise(),
    temperature=float(p["temperature"]), seq_only=bool(p["seq_only"]), repack_last=bool(p["repack_last"]),  # type: ignore[arg-type]
    psce_threshold=p["psce"],  # type: ignore[arg-type]
    aatype_override_mask=p["aov"], scn_override_mask=p["sov"],  # type: ignore[arg-type]
    timesteps_scn=ts_scn, step_scale=float(p["step_scale"]), s_churn=float(p["s_churn"]),  # type: ignore[arg-type]
    s_noise=float(p["s_noise"]), s_t_min=float(p["s_t_min"]), s_t_max=float(p["s_t_max"]),  # type: ignore[arg-type]
  )  # fmt: skip


def _differs(a: jax.Array, b: jax.Array, tol: float = 1e-6) -> bool:
  return bool(jnp.max(jnp.abs(jnp.asarray(a, jnp.float32) - jnp.asarray(b, jnp.float32))) > tol)


def _scn_loop(noise: sampler.SamplerNoise, **kw: object) -> jax.Array:
  """The sidechain EDM loop alone, on the model's own sequence-head embedding of the fixture."""
  p: dict[str, object] = {"T": SCN_STEPS, "step_scale": 1.5, "s_churn": 0.0, "s_noise": 1.0, "s_t_min": 0.01, "s_t_max": 50.0,
                          "mode": "linear", "t0": 0.0, "t1": 1.0}  # fmt: skip
  p.update(kw)
  h_v = jax.random.normal(jax.random.PRNGKey(11), (NUM_L, 128))
  ts = sampler.timestep_schedule(str(p["mode"]), int(p["T"]), float(p["t0"]), float(p["t1"]))  # type: ignore[arg-type]
  churn = None if noise.churn_z is None else noise.churn_z[0, : int(p["T"])]
  x, _ = sampler.scn_edm_loop(
    _MODEL.scn_diffusion, h_v, FX.aatype, FX.valid, ts, noise.prior_z[0], churn,
    step_scale=float(p["step_scale"]), s_churn=float(p["s_churn"]), s_noise=float(p["s_noise"]),  # type: ignore[arg-type]
    s_t_min=float(p["s_t_min"]), s_t_max=float(p["s_t_max"]),  # type: ignore[arg-type]
  )  # fmt: skip
  return x


def _stage(spec: SamplingSpecification) -> dict[str, jax.Array]:
  prep = FAMPNNPrepared(
    atom37_xyz=np.asarray(FX.x), atom37_mask=np.asarray(FX.atom_mask), mask=np.ones(NUM_L, np.float32),
    residue_idx=np.asarray(FX.ridx), chain_idx=np.asarray(FX.cidx),
    aatype_aminx=np.zeros(NUM_L, np.int32), aatype_fampnn=np.asarray(FX.aatype),
    missing_atom_mask=np.zeros((NUM_L, 37), np.float32), chain_id_mapping={"A": 0},
  )  # fmt: skip
  batch = FamilyBatch(input_indices=(0,), arrays={"prepared": (prep,)}, skipped=(), lengths=(NUM_L,))
  key = jax.random.PRNGKey(0)
  return dict(_FAMPNNStages(_MODEL, spec, "sample")(batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key))


def _spec(**kw: object) -> SamplingSpecification:
  opts = FAMPNNOptions(num_seq_steps=S_STEPS, num_scn_steps=SCN_STEPS, **{k: v for k, v in kw.items() if k in FAMPNNOptions.__dataclass_fields__})  # type: ignore[arg-type]
  rest = {k: v for k, v in kw.items() if k not in FAMPNNOptions.__dataclass_fields__}
  return SamplingSpecification(inputs=(), model_family="fampnn", fampnn=opts, **rest)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# run-level knobs
# ---------------------------------------------------------------------------------------------
def test_knob_semantics_seed() -> None:
  """``seed`` (sampling_utils.py:13-16 seeds every RNG): one seed is one result; another seed is another."""
  a = _stage(_spec(random_seed=1, temperature=1.0))
  b = _stage(_spec(random_seed=1, temperature=1.0))
  c = _stage(_spec(random_seed=2, temperature=1.0))
  np.testing.assert_array_equal(np.asarray(a["packed_atom37_xyz"]), np.asarray(b["packed_atom37_xyz"]))
  assert _differs(a["packed_atom37_xyz"], c["packed_atom37_xyz"])


def test_knob_semantics_checkpoint_id() -> None:
  """``checkpoint_path`` (:6) picks the weights: ``spec.checkpoint_id`` or ``fampnn.checkpoint_id``, default fampnn_0_3."""
  assert _checkpoint_id(SamplingSpecification(inputs="x.pdb")) == "fampnn_0_3"
  assert _checkpoint_id(SamplingSpecification(inputs="x.pdb", checkpoint_id="fampnn_0_3_cath")) == "fampnn_0_3_cath"
  opt = FAMPNNOptions(checkpoint_id="fampnn_0_0")
  assert _checkpoint_id(SamplingSpecification(inputs="x.pdb", model_family="fampnn", fampnn=opt)) == "fampnn_0_0"
  both = SamplingSpecification(
    inputs="x.pdb", model_family="fampnn", checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_3"),
  )
  with pytest.raises(ValueError, match="conflicting FAMPNN checkpoint ids"):
    _checkpoint_id(both)


def test_knob_semantics_num_samples() -> None:
  """``num_seqs_per_pdb`` (:9, seq_design.py:104 repeats each PDB): ``num_samples`` designs per input, each its own draw."""
  out = _stage(_spec(num_samples=3, temperature=1.0))
  assert out["sequences"].shape == (1, 3, 1, NUM_L)
  assert out["packed_atom37_xyz"].shape == (1, 3, 1, NUM_L, 37, 3)
  xyz = np.asarray(out["packed_atom37_xyz"])[0, :, 0]
  assert not np.allclose(xyz[0], xyz[1]) and not np.allclose(xyz[1], xyz[2])


# ---------------------------------------------------------------------------------------------
# fixed positions (fixed_pos_csv columns fixed_pos_seq, fixed_pos_scn; sampling_utils.py:92-190)
# ---------------------------------------------------------------------------------------------
def test_knob_semantics_fixed_pos_seq() -> None:
  """Fixed sequence positions keep the input residue; the syntax is ``[chain]start[-end]`` joined by commas."""
  idx = sampler.parse_fixed_positions("A1-2, A5", {"A": 0}, FX.ridx, FX.cidx)
  assert idx == [0, 1, 4]  # residue numbers are 1-based here
  assert sampler.parse_fixed_positions("", {"A": 0}, FX.ridx, FX.cidx) == []
  with pytest.raises(ValueError, match="Chain ID B not found"):
    sampler.parse_fixed_positions("B1", {"A": 0}, FX.ridx, FX.cidx)
  with pytest.raises(ValueError, match="Invalid position format"):
    sampler.parse_fixed_positions("A1:3", {"A": 0}, FX.ridx, FX.cidx)
  aov, sov = sampler.override_masks("A1-2,A5", "", {"A": 0}, FX.ridx, FX.cidx)
  assert np.asarray(aov).tolist() == [1, 1, 0, 0, 1, 0, 0, 0] and float(jnp.sum(sov)) == 0.0
  res = _sample(aov=aov.astype(jnp.float32), sov=sov.astype(jnp.float32), temperature=1.0)
  fixed = np.asarray(aov) > 0
  np.testing.assert_array_equal(np.asarray(res.aatype_final)[fixed], np.asarray(FX.aatype)[fixed])
  assert np.all(np.asarray(res.aatype_final) != sampler.X_TOKEN)  # every other position got designed


def test_knob_semantics_fixed_pos_scn() -> None:
  """Fixed sidechain positions keep the input sidechain through design and the final repack (sd_model.py:273)."""
  aov, sov = sampler.override_masks("A1-3", "A1-2", {"A": 0}, FX.ridx, FX.cidx)
  assert float(jnp.min(aov - sov)) >= 0  # upstream raises when a sidechain is fixed where the sequence is not (sd_model.py:230)
  res = _sample(aov=aov.astype(jnp.float32), sov=sov.astype(jnp.float32), temperature=1.0, repack_last=True)
  held = np.asarray(sov) > 0
  scn = np.asarray(res.x_final)[:, np.asarray(sampler.NON_BB_IDXS_ARR)]
  want = np.asarray(FX.x)[:, np.asarray(sampler.NON_BB_IDXS_ARR)]
  np.testing.assert_allclose(scn[held], want[held], atol=1e-5)
  assert not np.allclose(scn[~held], want[~held], atol=1e-3)


# ---------------------------------------------------------------------------------------------
# sequence design options
# ---------------------------------------------------------------------------------------------
def test_knob_semantics_seq_only() -> None:
  """``seq_only`` (:19): sidechains are not packed during design; only the provided ones stay (sd_model.py:247)."""
  res = _sample(seq_only=True, repack_last=False, temperature=1.0)
  scn = np.asarray(res.x_final)[:, np.asarray(sampler.NON_BB_IDXS_ARR)]
  assert np.all(scn == 0.0)  # nothing provided, nothing packed
  assert np.all(np.asarray(res.aatype_final) != sampler.X_TOKEN)
  packed = _sample(seq_only=False, repack_last=False, temperature=1.0)
  assert np.any(np.asarray(packed.x_final)[:, np.asarray(sampler.NON_BB_IDXS_ARR)] != 0.0)


def test_knob_semantics_temperature() -> None:
  """``temperature`` (:20, fampnn_denoiser.py:112-129): 0 is argmax and draws nothing; > 0 samples from softmax(l / T)."""
  base = _noise(0)
  other = base._replace(categorical_u=jnp.asarray(np.random.default_rng(99).random((S_STEPS, NUM_L)), dtype=jnp.float32))
  a, b = _sample(base, temperature=0.0), _sample(other, temperature=0.0)
  np.testing.assert_array_equal(np.asarray(a.aatype_final), np.asarray(b.aatype_final))
  hot_a, hot_b = _sample(base, temperature=1.0), _sample(other, temperature=1.0)
  assert np.any(np.asarray(hot_a.aatype_final) != np.asarray(hot_b.aatype_final))
  logits = jnp.asarray([[2.0, 1.0, 0.0] + [-5.0] * 17 + [9.0]])  # X (index 20) is huge but must never be sampled
  masked, probs = sampler.mask_logits_and_scale(logits, 0.5)
  assert float(masked[0, sampler.X_TOKEN]) == -1e9 and float(probs[0, sampler.X_TOKEN]) == 0.0
  want = jax.nn.softmax(jnp.asarray([2.0, 1.0, 0.0] + [-5.0] * 17 + [-1.0e9]) / 0.5)  # X masked first, then divided by T
  np.testing.assert_allclose(np.asarray(probs[0]), np.asarray(want), atol=1e-6)


def test_knob_semantics_repack_last() -> None:
  """``repack_last`` (:21): one more sidechain pass on the finished sequence, consuming its own prior and psce."""
  on, off = _sample(repack_last=True), _sample(repack_last=False)
  np.testing.assert_array_equal(np.asarray(on.aatype_final), np.asarray(off.aatype_final))  # sequence is unaffected
  assert _differs(on.x_final, off.x_final)
  changed = _noise(0)._replace(prior_z=_noise(0).prior_z.at[S_STEPS].set(_noise(5).prior_z[S_STEPS]))
  assert _differs(_sample(changed, repack_last=True).x_final, on.x_final)  # the extra pass reads prior_z[S]
  np.testing.assert_array_equal(np.asarray(_sample(changed, repack_last=False).x_final), np.asarray(off.x_final))


def test_knob_semantics_psce_threshold() -> None:
  """``psce_threshold`` (:22): re-mask sidechains whose mean psce exceeds it, except after the last step."""
  keep_all = _sample(psce=None, repack_last=False)
  huge = _sample(psce=1.0e9, repack_last=False)
  np.testing.assert_allclose(np.asarray(huge.x_final), np.asarray(keep_all.x_final), atol=1e-6)
  drop_all = _sample(psce=-1.0, repack_last=False)
  assert _differs(drop_all.x_final, keep_all.x_final)
  # after the second-to-last step everything was re-masked, so sidechains in the final state come from the last step only
  assert _differs(drop_all.xt_traj[0], keep_all.xt_traj[0]) or _differs(drop_all.xt_traj[1], keep_all.xt_traj[1])
  zeroed = np.asarray(drop_all.xt_traj[0])[:, np.asarray(sampler.NON_BB_IDXS_ARR)]
  assert np.all(zeroed == 0.0)  # threshold below any psce: every sidechain re-masked after step 0


# ---------------------------------------------------------------------------------------------
# sequence timestep schedule (timestep_schedule.*; sampling_utils.py:75-100, sd_model.py:233-234)
# ---------------------------------------------------------------------------------------------
def test_knob_semantics_num_seq_steps() -> None:
  """``timestep_schedule.num_steps`` (:25): that many denoiser steps; cumulative reveal count ends at L."""
  for n in (1, 2, 5):
    ts = sampler.timestep_schedule("linear", n, 0.0, 1.0)
    assert ts.shape == (n + 1,)
    k = sampler.unmask_targets(ts, jnp.asarray(float(NUM_L)), jnp.asarray(0))
    assert int(k[0]) == 0 and int(k[-1]) == NUM_L and bool(jnp.all(jnp.diff(k) >= 0))
  noise = _noise(0)
  res = _sample(noise, S=2)
  assert res.xt_traj.shape[0] == 2 and res.aatype_t_traj.shape[0] == 2


def test_knob_semantics_seq_timestep_mode() -> None:
  """``timestep_schedule.mode`` (:26): the warp changes how many residues are revealed early."""
  n, length = 6, jnp.asarray(float(NUM_L))
  lin = sampler.unmask_targets(sampler.timestep_schedule("linear", n, 0.0, 1.0), length, jnp.asarray(0))
  cos = sampler.unmask_targets(sampler.timestep_schedule("cosine", n, 0.0, 1.0), length, jnp.asarray(0))
  sq = sampler.unmask_targets(sampler.timestep_schedule("square", n, 0.0, 1.0), length, jnp.asarray(0))
  assert bool(jnp.all(cos <= lin)) and bool(jnp.all(sq <= lin)) and bool(jnp.any(cos < lin))  # 1-cos(pi t/2) <= t, t^2 <= t
  with pytest.raises(NotImplementedError):
    sampler.timestep_schedule("nope", 3, 0.0, 1.0)
  np.testing.assert_allclose(
    np.asarray(sampler.timestep_schedule("sqrt", 4, 0.0, 1.0)), np.sqrt(np.linspace(0, 1, 5)), atol=1e-6,
  )


def test_knob_semantics_seq_t_start() -> None:
  """``timestep_schedule.t_start`` (:27): the reveal schedule starts at ``ceil(t_start * L)`` residues, not at 0."""
  ts = sampler.timestep_schedule("linear", 4, 0.5, 1.0)
  assert float(ts[0]) == 0.5
  k = sampler.unmask_targets(ts, jnp.asarray(float(NUM_L)), jnp.asarray(0))
  assert int(k[0]) == NUM_L // 2 and int(k[-1]) == NUM_L


def test_knob_semantics_seq_t_end() -> None:
  """``timestep_schedule.t_end`` (:28): schedule end; below 1 some residues are never revealed and stay X."""
  ts = sampler.timestep_schedule("linear", S_STEPS, 0.0, 0.5)
  assert float(ts[-1]) == 0.5
  res = _sample(seq_t1=0.5, repack_last=False, temperature=1.0)
  assert int(jnp.sum(res.aatype_final == sampler.X_TOKEN)) == NUM_L // 2


# ---------------------------------------------------------------------------------------------
# sidechain diffusion (scn_diffusion.*; edm_interpolant.py, scn_diffusion_mlp.py)
# ---------------------------------------------------------------------------------------------
def test_knob_semantics_num_scn_steps() -> None:
  """``scn_diffusion.num_steps`` (:32): the Euler loop takes that many steps; one step differs from many."""
  noise = _noise(0)
  h_v = jax.random.normal(jax.random.PRNGKey(11), (NUM_L, 128))
  for n in (1, 3):
    ts = sampler.timestep_schedule("linear", n, 0.0, 1.0)
    _, trace = sampler.scn_edm_loop(
      _MODEL.scn_diffusion, h_v, FX.aatype, FX.valid, ts, noise.prior_z[0], None,
      step_scale=1.5, s_churn=0.0, s_noise=1.0, s_t_min=0.01, s_t_max=50.0,
    )  # fmt: skip
    assert trace.x_in.shape[0] == n
  assert _differs(_scn_loop(noise, T=1), _scn_loop(noise, T=3))


def test_knob_semantics_scn_timestep_mode() -> None:
  """``scn_diffusion.timestep_schedule.mode`` (:35): the same warps, applied to the EDM time axis."""
  noise = _noise(0)
  assert _differs(_scn_loop(noise, mode="linear"), _scn_loop(noise, mode="cosine"))
  lin = sampler.timestep_schedule("linear", SCN_STEPS, 0.0, 1.0)
  cub = sampler.timestep_schedule("cubic", SCN_STEPS, 0.0, 1.0)
  assert bool(jnp.all(cub <= lin))


def test_knob_semantics_scn_t_start() -> None:
  """``scn_diffusion.timestep_schedule.t_start`` (:36): the loop starts at ``sigma(t_start)``, but the prior is drawn at ``sigma(0)``."""
  noise = _noise(0)
  assert _differs(_scn_loop(noise, t0=0.0), _scn_loop(noise, t0=0.5))
  d = _MODEL.scn_diffusion
  s0 = sampler.edm_sigma(jnp.asarray(0.0), s_min=d.edm_s_min, s_max=d.edm_s_max, rho=d.edm_rho, sigma_data=d.sigma_data)
  s1 = sampler.edm_sigma(jnp.asarray(1.0), s_min=d.edm_s_min, s_max=d.edm_s_max, rho=d.edm_rho, sigma_data=d.sigma_data)
  np.testing.assert_allclose(float(s0), 80.0 * d.sigma_data, rtol=1e-5)  # t = 0 is pure noise (edm_interpolant.py:48-55)
  np.testing.assert_allclose(float(s1), d.edm_s_min * d.sigma_data, rtol=1e-5)  # t = 1 is clean data


def test_knob_semantics_scn_t_end() -> None:
  """``scn_diffusion.timestep_schedule.t_end`` (:37): stopping the schedule early leaves noise in the sample."""
  noise = _noise(0)
  full, early = _scn_loop(noise, t1=1.0), _scn_loop(noise, t1=0.5)
  assert _differs(full, early)
  assert float(sampler.timestep_schedule("linear", SCN_STEPS, 0.0, 0.5)[-1]) == 0.5


def test_knob_semantics_step_scale() -> None:
  """``scn_diffusion.step_scale`` (:38, edm_interpolant.py:179-180): scales the Euler drift; 0 freezes the state at the prior."""
  noise = _noise(0)
  d = _MODEL.scn_diffusion
  s0 = sampler.edm_sigma(jnp.asarray(0.0), s_min=d.edm_s_min, s_max=d.edm_s_max, rho=d.edm_rho, sigma_data=d.sigma_data)
  frozen = _scn_loop(noise, step_scale=0.0)
  np.testing.assert_allclose(np.asarray(frozen), np.asarray(noise.prior_z[0] * s0), atol=1e-4)
  assert _differs(_scn_loop(noise, step_scale=1.0), _scn_loop(noise, step_scale=1.5))


def test_knob_semantics_s_churn() -> None:
  """``churn_cfg.s_churn`` (:40, edm_interpolant.py:139-153): 0 is off and draws no noise; > 0 adds noise and shifts t."""
  base = _noise(0, churn_seed=1)
  other = _noise(0, churn_seed=2)
  off_a, off_b = _scn_loop(base, s_churn=0.0), _scn_loop(other, s_churn=0.0)
  np.testing.assert_array_equal(np.asarray(off_a), np.asarray(off_b))  # churn noise ignored when off
  on_a, on_b = _scn_loop(base, s_churn=30.0), _scn_loop(other, s_churn=30.0)
  assert _differs(on_a, off_a) and _differs(on_a, on_b)


def test_knob_semantics_s_noise() -> None:
  """``churn_cfg.s_noise`` (:41, edm_interpolant.py:149): scales the churn noise; 0 makes the result independent of it."""
  a, b = _noise(0, churn_seed=1), _noise(0, churn_seed=2)
  z_a, z_b = _scn_loop(a, s_churn=30.0, s_noise=0.0), _scn_loop(b, s_churn=30.0, s_noise=0.0)
  np.testing.assert_allclose(np.asarray(z_a), np.asarray(z_b), atol=1e-5)
  assert _differs(_scn_loop(a, s_churn=30.0, s_noise=1.0), _scn_loop(b, s_churn=30.0, s_noise=1.0))


def test_knob_semantics_s_t_min() -> None:
  """``churn_cfg.s_t_min`` (:42, edm_interpolant.py:142-143): churn only where sigma >= s_t_min * sigma_data."""
  noise = _noise(0, churn_seed=1)
  off = _scn_loop(noise, s_churn=0.0)
  window_closed = _scn_loop(noise, s_churn=30.0, s_t_min=1.0e6)
  np.testing.assert_allclose(np.asarray(window_closed), np.asarray(off), atol=1e-3)
  assert _differs(_scn_loop(noise, s_churn=30.0, s_t_min=0.01), off, 1e-2)


def test_knob_semantics_s_t_max() -> None:
  """``churn_cfg.s_t_max`` (:43, edm_interpolant.py:142-143): churn only where sigma <= s_t_max * sigma_data."""
  noise = _noise(0, churn_seed=1)
  off = _scn_loop(noise, s_churn=0.0)
  window_closed = _scn_loop(noise, s_churn=30.0, s_t_max=1.0e-6)
  np.testing.assert_allclose(np.asarray(window_closed), np.asarray(off), atol=1e-3)
  assert _differs(_scn_loop(noise, s_churn=30.0, s_t_max=50.0), off, 1e-2)
