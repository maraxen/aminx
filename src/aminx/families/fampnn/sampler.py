"""Upstream-faithful FAMPNN EDM sidechain diffusion and masked-autoregressive (MAR) sampler.

Each function cites the upstream line it reproduces (``richardshuai/fampnn``, local clone
``~/repos/fampnn`` at ``aaf788b``). Randomness is an explicit input (:class:`SamplerNoise`) so the
sampler can be graded against a sealed upstream oracle with injected noise; production callers draw
the noise with :func:`draw_sampler_noise`.

Conventions
-----------
* Single structure (no batch axis). ``valid`` is upstream's ``seq_mask`` (1 for real residues);
  ``seq_mlm`` / ``scn_mlm`` are upstream's ``seq_mlm_mask`` / ``scn_mlm_mask`` (1 = revealed).
* Tokens are in the FAMPNN 21-letter order (``X`` = 20); sidechain tensors carry the 33 non-backbone
  Atom37 slots (``NON_BB_IDXS``) in the structure's *local backbone frame*.
* EDM time ``t`` runs 0 (pure noise) to 1 (clean), as upstream (``edm_interpolant.py``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.fampnn.constants import (
  NON_BB_IDXS_ARR,
  RESTYPE_ATOM37_MASK,
  RESTYPE_REF_POS_ATOM37,
)
from aminx.families.fampnn.features import (
  _layer_norm,
  _linear,
  atom37_to_atom14,
  backbone_coords_to_frames,
)
from aminx.families.fampnn.model import FAMPNNModel
from aminx.families.fampnn.sidechain_diffusion import SidechainConfidenceModule, SidechainDiffusionModule

X_TOKEN = 20
NUM_SCN_ATOMS = 33
_BIG = 1.0e6  # sampling_utils.py:22 (padding decoded last)
_X_LOGIT = -1.0e9  # fampnn_denoiser.py:~84 ("do not sample mask/unknowns")


class SamplerNoise(NamedTuple):
  """Every random draw the upstream MAR sampler makes, as explicit arrays.

  Attributes
  ----------
  decoding_u : Float[Array, " L"]
    Uniforms behind the residue decoding order (``torch.rand_like`` at ``sampling_utils.py:25``).
  categorical_u : Float[Array, "S L"]
    One uniform per residue per MAR step for the inverse-CDF categorical draw
    (``torch.multinomial`` at ``fampnn_denoiser.py:129``). Unused when ``temperature == 0``.
  prior_z : Float[Array, "C L 33 3"]
    Unit normal behind the EDM prior of each denoiser call, ``C = S + repack_last``
    (``torch.randn`` in ``edm_interpolant.py::sample_prior``).
  churn_z : Float[Array, "C T L 33 3"] | None
    Unit normals for the Karras churn of each denoiser call and EDM step
    (``torch.randn_like`` in ``edm_interpolant.py::churn``); ``None`` when ``s_churn == 0``.
  """

  decoding_u: Float[Array, " L"]
  categorical_u: Float[Array, "S L"]
  prior_z: Float[Array, "C L 33 3"]
  churn_z: Float[Array, "C T L 33 3"] | None


def draw_sampler_noise(
  key: jax.Array,
  num_res: int,
  num_seq_steps: int,
  num_scn_steps: int,
  *,
  repack_last: bool,
  s_churn: float,
  dtype: jnp.dtype = jnp.float32,
) -> SamplerNoise:
  """Draw :class:`SamplerNoise` for a production run (not used by the parity vehicle)."""
  k_ord, k_cat, k_pri, k_chu = jax.random.split(key, 4)
  calls = num_seq_steps + int(repack_last)
  churn = (
    jax.random.normal(k_chu, (calls, num_scn_steps, num_res, NUM_SCN_ATOMS, 3), dtype=dtype)
    if s_churn != 0.0
    else None
  )
  return SamplerNoise(
    decoding_u=jax.random.uniform(k_ord, (num_res,), dtype=dtype),
    categorical_u=jax.random.uniform(k_cat, (num_seq_steps, num_res), dtype=dtype),
    prior_z=jax.random.normal(k_pri, (calls, num_res, NUM_SCN_ATOMS, 3), dtype=dtype),
    churn_z=churn,
  )


# ---------------------------------------------------------------------------------------------
# EDM time <-> noise level (edm_interpolant.py:48-67)
# ---------------------------------------------------------------------------------------------
def edm_sigma(
  t: Float[Array, "..."],
  *,
  s_min: float,
  s_max: float,
  rho: float,
  sigma_data: float,
) -> Float[Array, "..."]:
  """Noise level at EDM time ``t`` (``edm_interpolant.py:48-55``): Karras schedule scaled by sigma_data."""
  term1 = s_max ** (1.0 / rho)
  term2 = t * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
  return ((term1 + term2) ** rho) * sigma_data


def edm_sigma_inv(
  sigma: Float[Array, "..."],
  *,
  s_min: float,
  s_max: float,
  rho: float,
  sigma_data: float,
) -> Float[Array, "..."]:
  """Inverse of :func:`edm_sigma` (``edm_interpolant.py:58-65``)."""
  term_1_2 = (sigma / sigma_data) ** (1.0 / rho)
  term_1 = s_max ** (1.0 / rho)
  return (term_1_2 - term_1) / (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))


def _sigma(diffusion: SidechainDiffusionModule, t: Float[Array, "..."]) -> Float[Array, "..."]:
  return edm_sigma(
    t,
    s_min=diffusion.edm_s_min,
    s_max=diffusion.edm_s_max,
    rho=diffusion.edm_rho,
    sigma_data=diffusion.sigma_data,
  )


def _sigma_inv(diffusion: SidechainDiffusionModule, sigma: Float[Array, "..."]) -> Float[Array, "..."]:
  return edm_sigma_inv(
    sigma,
    s_min=diffusion.edm_s_min,
    s_max=diffusion.edm_s_max,
    rho=diffusion.edm_rho,
    sigma_data=diffusion.sigma_data,
  )


# ---------------------------------------------------------------------------------------------
# Timestep schedules (sampling_utils.py:75-100)
# ---------------------------------------------------------------------------------------------
_SCHEDULE_MODES = ("linear", "square", "cubic", "sqrt", "cbrt", "cosine")


def timestep_schedule(
  mode: str,
  num_steps: int,
  t_start: float = 0.0,
  t_end: float = 1.0,
  dtype: jnp.dtype = jnp.float32,
) -> Float[Array, " S1"]:
  """``num_steps + 1`` timesteps in ``[t_start, t_end]`` warped by ``mode`` (``get_timesteps_from_schedule``)."""
  ts = jnp.linspace(t_start, t_end, num_steps + 1, dtype=dtype)
  if mode == "linear":
    return ts
  if mode == "square":
    return ts**2
  if mode == "cubic":
    return ts**3
  if mode == "sqrt":
    return ts**0.5
  if mode == "cbrt":
    return ts ** (1.0 / 3.0)
  if mode == "cosine":
    return 1 - jnp.cos(ts * math.pi / 2)
  msg = f"timestep schedule mode {mode!r} not implemented (expected one of {_SCHEDULE_MODES})"
  raise NotImplementedError(msg)


def unmask_targets(
  timesteps: Float[Array, " S1"],
  length: Float[Array, ""],
  num_partial: Int[Array, ""],
) -> Int[Array, " S1"]:
  """Cumulative number of revealed residues at each timestep (``sd_model.py:233-234``).

  ``K_i = ceil(t_i * (L - n_partial)) + n_partial``: the schedule is defined over the masked residues.
  """
  k = jnp.ceil(timesteps * (length - num_partial)).astype(jnp.int32)
  return k + num_partial


def decoding_order(u: Float[Array, " L"], valid: Float[Array, " L"]) -> Int[Array, " L"]:
  """``sampling_utils.get_decoding_order``: ``argsort`` of uniforms, padding (valid == 0) last.

  Upstream returns the argsort *permutation* (``perm[j]`` = index of the j-th smallest), and
  :func:`update_mlm_mask` compares it elementwise to K. That is a uniform random K-subset, but it
  is not the rank of each residue, so it differs from a double argsort given the same uniforms.
  """
  return jnp.argsort(jnp.where(valid > 0, u, _BIG)).astype(jnp.int32)


def update_mlm_mask(
  seq_mlm: Float[Array, " L"],
  perm: Int[Array, " L"],
  k: Int[Array, ""],
) -> Float[Array, " L"]:
  """``sampling_utils.update_mlm_mask``: reveal masked residues j with ``perm[j] < K``."""
  newly = (seq_mlm <= 0) & (perm < k)
  return newly.astype(seq_mlm.dtype) + seq_mlm


def unmask(
  curr: Float[Array, "L ..."],
  pred: Float[Array, "L ..."],
  mlm_prev: Float[Array, " L"],
  mlm_new: Float[Array, " L"],
) -> Float[Array, "L ..."]:
  """``sampling_utils.unmask``: take ``pred`` where a residue was newly revealed, else keep ``curr``."""
  newly = (mlm_new - mlm_prev) > 0
  newly = newly.reshape(newly.shape + (1,) * (curr.ndim - 1))
  return jnp.where(newly, pred, curr)


def mask_logits_and_scale(
  logits: Float[Array, "L 21"],
  temperature: float,
) -> tuple[Float[Array, "L 21"], Float[Array, "L 21"]]:
  """``FAMPNNDenoiser.sample_aatype`` (``fampnn_denoiser.py:~80-100``) minus the draw.

  Returns ``(logits_with_X_masked, scaled_probs)``. X is set to ``-1e9`` *before* the temperature
  division; ``temperature == 0`` takes the argmax branch (probabilities are not scaled).
  """
  logits = logits.at[..., X_TOKEN].set(_X_LOGIT)
  if temperature == 0.0:
    return logits, jax.nn.softmax(logits, axis=-1)
  return logits, jax.nn.softmax(logits / temperature, axis=-1)


def categorical_inverse_cdf(probs: Float[Array, "L K"], u: Float[Array, " L"]) -> Int[Array, " L"]:
  """Inverse-CDF categorical draw: the first index whose cumulative probability reaches ``u``."""
  wide = jnp.float64 if jax.config.jax_enable_x64 else probs.dtype
  cdf = jnp.cumsum(probs.astype(wide), axis=-1)
  cdf = cdf / cdf[:, -1:]
  idx = jnp.sum(cdf < u.astype(wide)[:, None], axis=-1)
  return jnp.minimum(idx, probs.shape[-1] - 1).astype(jnp.int32)


# ---------------------------------------------------------------------------------------------
# Sidechain denoiser, one EDM loop, local <-> global frame
# ---------------------------------------------------------------------------------------------
def scn_denoise_local(
  diffusion: SidechainDiffusionModule,
  x_scn: Float[Array, "L 33 3"],
  t: Float[Array, ""],
  h_v: Float[Array, "L 128"],
  aatype: Int[Array, " L"],
  valid: Float[Array, " L"],
  x_self_cond: Float[Array, "L 33 3"] | None,
) -> Float[Array, "L 33 3"]:
  """``SidechainMLP.forward`` (``scn_diffusion_mlp.py:189-238``) with EDM preconditioning.

  ``c_in`` scales the input, the self-conditioning coordinates are divided by ``sigma_data`` (zeros
  when absent), ``c_noise = log(sigma) / 4`` drives the timestep embedding, the per-token reference
  conformer (all 37 atoms) and ``h_V`` are added, and the output is ``c_skip * x + c_out * F``.
  """
  dtype = diffusion.x_embedder.weight.dtype
  num_l = x_scn.shape[0]
  sigma = _sigma(diffusion, jnp.asarray(t, dtype=dtype))
  sd = diffusion.sigma_data
  var_x = sd**2 + sigma**2
  c_in = 1.0 / jnp.sqrt(var_x)
  c_skip = sd**2 / var_x
  c_out = sigma * sd / jnp.sqrt(var_x)
  c_noise = 0.25 * jnp.log(sigma)

  x_in = x_scn.astype(dtype) * c_in
  cond = jnp.zeros_like(x_in) if x_self_cond is None else x_self_cond.astype(dtype) / sd
  x = jnp.concatenate([x_in, cond], axis=-1).reshape(num_l, NUM_SCN_ATOMS * 6)
  x = jnp.concatenate([x, jax.nn.one_hot(jnp.clip(aatype, 0, 20), 21, dtype=dtype)], axis=-1)
  h = _linear(diffusion.x_embedder, x)

  assert diffusion.f_embedder is not None  # noqa: S101 (upstream checkpoints always carry f_embedder)
  ref_pos = RESTYPE_REF_POS_ATOM37[jnp.clip(aatype, 0, 20)].astype(dtype).reshape(num_l, 37 * 3)
  h = h + _linear(diffusion.f_embedder, ref_pos)

  c = diffusion.t_embedder(c_noise[None])  # (1, H): one scalar t per structure
  h_cond = _linear(diffusion.c_embedder, h_v.astype(dtype))
  c = c + h_cond
  h = h + h_cond
  for blk in diffusion.blocks:
    h = blk(h, c)
  out = diffusion.final_layer(h, c) * valid.astype(dtype)[:, None]
  out = out.reshape(num_l, NUM_SCN_ATOMS, 3)
  return c_skip * x_scn.astype(dtype) + c_out * out


class EdmTrace(NamedTuple):
  """Per-step denoiser inputs and outputs, for grading."""

  x_in: Float[Array, "T L 33 3"]
  x_out: Float[Array, "T L 33 3"]


def scn_edm_loop(
  diffusion: SidechainDiffusionModule,
  h_v: Float[Array, "L 128"],
  aatype: Int[Array, " L"],
  valid: Float[Array, " L"],
  timesteps: Float[Array, " T1"],
  prior_z: Float[Array, "L 33 3"],
  churn_z: Float[Array, "T L 33 3"] | None,
  *,
  step_scale: float,
  s_churn: float,
  s_noise: float,
  s_t_min: float,
  s_t_max: float,
) -> tuple[Float[Array, "L 33 3"], EdmTrace]:
  """The Euler EDM loop of ``SidechainDiffusionModule.sidechain_diffusion`` (``scn_diffusion_mlp.py:80-98``).

  Prior ``z * sigma(t=0)``; per step: optional Karras churn (``edm_interpolant.py::churn``), one
  denoiser call, ``x += (sigma(t_next) - sigma(t)) * step_scale * (x - x1_pred) / sigma(t)``, and
  the previous ``x1_pred`` becomes the next self-conditioning input.
  """
  dtype = diffusion.x_embedder.weight.dtype
  num_steps = timesteps.shape[0] - 1
  sd = diffusion.sigma_data
  xt0 = prior_z.astype(dtype) * _sigma(diffusion, jnp.zeros((), dtype=dtype))
  cond0 = jnp.zeros_like(xt0)
  churn_xs = jnp.zeros((num_steps, *xt0.shape), dtype=dtype) if churn_z is None else churn_z.astype(dtype)
  ts = timesteps.astype(dtype)

  def step(carry: tuple, xs: tuple) -> tuple:
    xt, x_cond = carry
    t, t_next, z = xs
    if s_churn != 0.0:
      sigma = _sigma(diffusion, t)
      churn_mask = ((s_t_min * sd) <= sigma) & (sigma <= (s_t_max * sd))
      gamma = (s_churn / num_steps) * churn_mask.astype(dtype)
      sigma_hat = sigma + gamma * sigma
      # outside the churn window the added noise is exactly zero (torch gives sqrt(0)); under XLA fusion
      # ``sigma_hat**2 - sigma**2`` can round to a tiny negative number, so select instead of subtracting blindly.
      noise_scale = jnp.where(churn_mask, jnp.sqrt(jnp.maximum(sigma_hat**2 - sigma**2, 0.0)), 0.0)
      xt = xt + (z * s_noise) * noise_scale
      t = _sigma_inv(diffusion, sigma_hat)
    cond_in = x_cond if diffusion.use_self_conditioning else None
    x1 = scn_denoise_local(diffusion, xt, t, h_v, aatype, valid, cond_in)
    sig = _sigma(diffusion, t)
    score = (xt - x1) / sig * step_scale
    x_next = xt + (_sigma(diffusion, t_next) - sig) * score
    return (x_next, x1), (xt, x1)

  (x_final, _), (xs_in, xs_out) = jax.lax.scan(step, (xt0, cond0), (ts[:-1], ts[1:], churn_xs))
  return x_final, EdmTrace(xs_in, xs_out)


def local_to_global_scn(
  x_local: Float[Array, "L 33 3"],
  x_bb: Float[Array, "L 4 3"],
  scn_atom_mask: Float[Array, "L 33"],
  bb_atom_mask: Float[Array, "L 4"],
) -> Float[Array, "L 33 3"]:
  """``transform_sidechain_frame(..., to_local=False)`` (``data.py:551-580``).

  The frame is built from the backbone (N, CA, C); sidechain atoms absent from the residue type, or
  residues whose N/CA/C are not all present, are placed at CA.
  """
  stub = jnp.zeros((x_bb.shape[0], 37, 3), dtype=x_bb.dtype)
  stub = stub.at[:, 0].set(x_bb[:, 0]).at[:, 1].set(x_bb[:, 1]).at[:, 2].set(x_bb[:, 2])
  rot, ca = backbone_coords_to_frames(stub)
  glob = jnp.einsum("laj,lij->lai", x_local, rot) + ca[:, None, :]
  frame_ok = jnp.min(bb_atom_mask[:, jnp.array([0, 1, 2])], axis=-1) > 0
  glob = jnp.where(scn_atom_mask[..., None] > 0, glob, ca[:, None, :])
  return jnp.where(frame_ok[:, None, None], glob, ca[:, None, :])


# ---------------------------------------------------------------------------------------------
# Confidence (psce) head (sidechain_confidence.py)
# ---------------------------------------------------------------------------------------------
def psce_from_local(
  conf: SidechainConfidenceModule,
  x_local: Float[Array, "L 33 3"],
  h_v: Float[Array, "L 128"],
  h_esv: Float[Array, "L K 512"],
  e_idx: Int[Array, "L K"],
  aatype: Int[Array, " L"],
  valid: Float[Array, " L"],
  *,
  min_bin: float,
  max_bin: float,
) -> Float[Array, "L 33"]:
  """``SidechainConfidenceModule.forward`` + ``compute_psce`` (``sidechain_confidence.py:~38-90``).

  Decoder layers refine the (masked) encoder edge states; node state adds the aatype and local
  sidechain embeddings; an MLP gives ``33 * n_bins`` logits; psce is the expectation over bin centres
  of ``linspace(min_bin, max_bin, n_bins) + step / 2``.
  """
  assert conf.aatype_embedder is not None  # noqa: S101
  assert conf.sidechain_encoder is not None  # noqa: S101
  assert conf.mlp_0 is not None  # noqa: S101
  assert conf.mlp_2 is not None  # noqa: S101
  dtype = conf.aatype_embedder.weight.dtype
  validf = valid.astype(dtype)
  ones_attend = jnp.ones(e_idx.shape, dtype=dtype)
  h_dummy = jnp.zeros_like(h_v)
  hv, he = h_v.astype(dtype), h_esv.astype(dtype)
  for layer in conf.decoder_layers:
    he = he * validf[:, None, None]
    hv, he = layer(hv, he, h_dummy, e_idx, validf, ones_attend)  # type: ignore[misc]
  hv = hv + _linear(conf.aatype_embedder, jax.nn.one_hot(jnp.clip(aatype, 0, 20), 21, dtype=dtype))
  hv = hv + _linear(conf.sidechain_encoder, x_local.astype(dtype).reshape(x_local.shape[0], NUM_SCN_ATOMS * 3))
  logits = _linear(conf.mlp_2, jax.nn.silu(_linear(conf.mlp_0, hv)))
  logits = logits.reshape(x_local.shape[0], NUM_SCN_ATOMS, conf.num_bins) * validf[:, None, None]
  lower = jnp.linspace(min_bin, max_bin, conf.num_bins, dtype=dtype)
  centers = lower + (lower[1] - lower[0]) / 2
  return jnp.sum(jax.nn.softmax(logits, axis=-1) * centers, axis=-1)


# ---------------------------------------------------------------------------------------------
# One denoiser call = FAMPNNDenoiser.forward (fampnn_denoiser.py:35-84)
# ---------------------------------------------------------------------------------------------
class DenoiseCall(NamedTuple):
  """Outputs of one :func:`denoiser_call`."""

  x1_pred: Float[Array, "L 37 3"]
  aatype_pred: Int[Array, " L"]
  seq_logits: Float[Array, "L 21"]
  seq_probs: Float[Array, "L 21"]
  psce: Float[Array, "L 33"]


def _atom_mask_noised(
  aatype_noised: Int[Array, " L"],
  valid: Float[Array, " L"],
  missing: Float[Array, "L 37"],
  scn_mlm: Float[Array, " L"],
  dtype: jnp.dtype,
) -> Float[Array, "L 37"]:
  """``sd_model.py:48-51`` / ``fampnn_denoiser.py:~40-43``: ghost, padding, missing and masked sidechain atoms off."""
  m = RESTYPE_ATOM37_MASK[jnp.clip(aatype_noised, 0, 20)].astype(dtype)
  m = m * valid.astype(dtype)[:, None] * (1 - missing.astype(dtype))
  return m.at[:, NON_BB_IDXS_ARR].multiply(scn_mlm.astype(dtype)[:, None])


def denoiser_call(  # noqa: PLR0913
  model: FAMPNNModel,
  x_noised: Float[Array, "L 37 3"],
  aatype_noised: Int[Array, " L"],
  scn_mlm: Float[Array, " L"],
  *,
  valid: Float[Array, " L"],
  missing: Float[Array, "L 37"],
  residue_idx: Int[Array, " L"],
  chain_idx: Int[Array, " L"],
  aatype_override: Int[Array, " L"],
  aatype_override_mask: Float[Array, " L"],
  temperature: float,
  categorical_u: Float[Array, " L"] | None,
  prior_z: Float[Array, "L 33 3"],
  churn_z: Float[Array, "T L 33 3"] | None,
  timesteps_scn: Float[Array, " T1"],
  step_scale: float,
  s_churn: float,
  s_noise: float,
  s_t_min: float,
  s_t_max: float,
  conf_min_bin: float,
  conf_max_bin: float,
) -> DenoiseCall:
  """Sequence head, aatype draw, EDM sidechain loop, local-to-global transform, psce."""
  dtype = x_noised.dtype
  amask = _atom_mask_noised(aatype_noised, valid, missing, scn_mlm, dtype)
  # the encoder reads the pad mask through ``mask`` and treats ``seq_mask`` as "revealed": all-ones here
  # because masked positions already carry the X token and masked sidechains are off in ``amask``.
  fwd = model(x_noised, amask, aatype_noised, jnp.ones_like(valid), valid, residue_idx, chain_idx)
  logits_x, scaled_probs = mask_logits_and_scale(fwd.logits_fampnn.astype(dtype), temperature)
  if temperature == 0.0:
    aatype_pred = jnp.argmax(logits_x, axis=-1).astype(jnp.int32)
  else:
    assert categorical_u is not None  # noqa: S101
    aatype_pred = categorical_inverse_cdf(scaled_probs, categorical_u)
  seq_probs = jax.nn.softmax(logits_x, axis=-1)

  aatype_used = jnp.where(aatype_override_mask > 0, aatype_override, aatype_pred)
  assert model.scn_diffusion is not None  # noqa: S101
  x_local, _ = scn_edm_loop(
    model.scn_diffusion,
    fwd.h_v,
    aatype_used,
    valid,
    timesteps_scn,
    prior_z,
    churn_z,
    step_scale=step_scale,
    s_churn=s_churn,
    s_noise=s_noise,
    s_t_min=s_t_min,
    s_t_max=s_t_max,
  )
  x_bb, bb_mask = encoder_backbone(x_noised, amask, aatype_noised)
  scn_atom_mask = jnp.take(RESTYPE_ATOM37_MASK[jnp.clip(aatype_used, 0, 20)].astype(dtype), NON_BB_IDXS_ARR, axis=1)
  x_scn = local_to_global_scn(x_local, x_bb, scn_atom_mask, bb_mask)
  x1_pred = jnp.zeros_like(x_noised).at[:, jnp.array([0, 1, 2, 4])].set(x_bb).at[:, NON_BB_IDXS_ARR].set(x_scn)

  if model.scn_confidence is not None:
    psce = psce_from_local(
      model.scn_confidence,
      x_local,
      fwd.h_v,
      fwd.h_e,
      fwd.e_idx,
      aatype_used,
      valid,
      min_bin=conf_min_bin,
      max_bin=conf_max_bin,
    )
  else:
    psce = jnp.zeros((x_noised.shape[0], NUM_SCN_ATOMS), dtype=dtype)
  return DenoiseCall(x1_pred, aatype_pred, logits_x, seq_probs, psce)


def encoder_backbone(
  atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask_noised: Float[Array, "L 37"],
  aatype_noised: Int[Array, " L"],
) -> tuple[Float[Array, "L 4 3"], Float[Array, "L 4"]]:
  """Atom14 backbone (N, CA, C, O) as the encoder sees it, missing atoms replaced by CA (``fampnn.py:88``)."""
  xyz14, mask14 = atom37_to_atom14(atom37_xyz, atom37_mask_noised, jnp.clip(aatype_noised, 0, 20))
  ca = xyz14[:, 1:2, :]
  xyz14 = jnp.where(mask14[..., None] > 0, xyz14, ca)
  return xyz14[:, :4], mask14[:, :4]


# ---------------------------------------------------------------------------------------------
# The MAR sampler (SeqDenoiser.sample / sidechain_pack, sd_model.py)
# ---------------------------------------------------------------------------------------------
class MarResult(NamedTuple):
  """Outputs of :func:`mar_sample_upstream`."""

  x_final: Float[Array, "L 37 3"]
  aatype_final: Int[Array, " L"]
  psce: Float[Array, "L 33"]
  seq_probs: Float[Array, "L 21"]
  xt_traj: Float[Array, "S L 37 3"]
  aatype_t_traj: Int[Array, "S L"]
  aatype_pred_traj: Int[Array, "S L"]
  psce_t_traj: Float[Array, "S L 33"]
  seq_logits_traj: Float[Array, "S L 21"]


def mar_sample_upstream(  # noqa: PLR0913, PLR0915
  model: FAMPNNModel,
  x: Float[Array, "L 37 3"],
  aatype: Int[Array, " L"],
  valid: Float[Array, " L"],
  missing: Float[Array, "L 37"],
  residue_idx: Int[Array, " L"],
  chain_idx: Int[Array, " L"],
  timesteps: Float[Array, " S1"],
  noise: SamplerNoise,
  *,
  temperature: float,
  seq_only: bool,
  repack_last: bool,
  psce_threshold: float | None,
  aatype_override_mask: Float[Array, " L"] | None = None,
  scn_override_mask: Float[Array, " L"] | None = None,
  timesteps_scn: Float[Array, " T1"],
  step_scale: float,
  s_churn: float,
  s_noise: float = 1.0,
  s_t_min: float = 0.01,
  s_t_max: float = 50.0,
  conf_min_bin: float = 0.0,
  conf_max_bin: float = 4.0,
) -> MarResult:
  """``SeqDenoiser.sample`` (``sd_model.py:148-291``): linear-K masked autoregressive design + repack.

  Per step: denoise (sequence head, aatype draw, 50-step sidechain EDM at every position), reveal the
  next K residues in decoding order, optionally re-mask low-confidence sidechains
  (``psce_threshold``), and finally repack the sidechains for the finished sequence (``repack_last``).
  """
  dtype = x.dtype
  num_l = x.shape[0]
  num_steps = timesteps.shape[0] - 1
  validf = valid.astype(dtype)
  aov = jnp.zeros((num_l,), dtype) if aatype_override_mask is None else aatype_override_mask.astype(dtype)
  sov = jnp.zeros((num_l,), dtype) if scn_override_mask is None else scn_override_mask.astype(dtype)

  x0 = x.at[:, NON_BB_IDXS_ARR].multiply(sov[:, None, None])
  aatype_noised = jnp.where(aov > 0, aatype, X_TOKEN) * (validf > 0).astype(jnp.int32)
  aatype_noised = aatype_noised.astype(jnp.int32)
  perm = decoding_order(noise.decoding_u, validf)
  length = jnp.sum(validf)
  k_all = unmask_targets(timesteps.astype(dtype), length, jnp.sum(aov).astype(jnp.int32))

  common = {
    "valid": validf,
    "missing": missing,
    "residue_idx": residue_idx,
    "chain_idx": chain_idx,
    "aatype_override": aatype_noised,
    "aatype_override_mask": aov,
    "timesteps_scn": timesteps_scn,
    "step_scale": step_scale,
    "s_churn": s_churn,
    "s_noise": s_noise,
    "s_t_min": s_t_min,
    "s_t_max": s_t_max,
    "conf_min_bin": conf_min_bin,
    "conf_max_bin": conf_max_bin,
  }

  seq_probs0 = jnp.zeros((num_l, 21), dtype)
  psce0 = jnp.zeros((num_l, NUM_SCN_ATOMS), dtype)
  churn_xs = (
    jnp.zeros((num_steps,), dtype) if noise.churn_z is None else noise.churn_z[:num_steps].astype(dtype)
  )
  cat_xs = noise.categorical_u[:num_steps].astype(dtype)
  is_last = jnp.arange(num_steps) == num_steps - 1

  def mar_step(carry: tuple, xs: tuple) -> tuple:
    xt, aatype_t, seq_probs_t, psce_t, seq_mlm, scn_mlm = carry
    k_next, cat_u, prior_z, churn_z, last = xs
    call = denoiser_call(
      model,
      xt,
      aatype_t,
      scn_mlm,
      temperature=temperature,
      categorical_u=None if temperature == 0.0 else cat_u,
      prior_z=prior_z,
      churn_z=None if noise.churn_z is None else churn_z,
      **common,
    )
    seq_prev, scn_prev = seq_mlm, scn_mlm
    seq_mlm = update_mlm_mask(seq_mlm, perm, k_next)
    scn_mlm = sov if seq_only else seq_mlm
    aatype_t = unmask(aatype_t, call.aatype_pred, seq_prev, seq_mlm)
    xt = unmask(xt, call.x1_pred, scn_prev, scn_mlm)
    seq_probs_t = unmask(seq_probs_t, call.seq_probs, seq_prev, seq_mlm)
    psce_t = unmask(psce_t, call.psce, scn_prev, scn_mlm)
    if psce_threshold is not None:
      # re-mask low-confidence sidechains, except after the last step (sd_model.py:255)
      m = RESTYPE_ATOM37_MASK[jnp.clip(aatype_t, 0, 20)][:, NON_BB_IDXS_ARR].astype(dtype)
      per_res = jnp.sum(psce_t * m, axis=-1) / jnp.maximum(jnp.sum(m, axis=-1), 1.0)
      keep = jnp.where(last, 1.0, (per_res <= psce_threshold).astype(dtype))
      scn_mlm = scn_mlm * keep
      xt = jnp.where(last, xt, xt.at[:, NON_BB_IDXS_ARR].multiply(scn_mlm[:, None, None]))
      psce_t = psce_t * jnp.where(last, 1.0, scn_mlm)[:, None]
    out = (xt, aatype_t, call.aatype_pred, psce_t, call.seq_logits)
    return (xt, aatype_t, seq_probs_t, psce_t, seq_mlm, scn_mlm), out

  carry0 = (x0, aatype_noised, seq_probs0, psce0, aov, sov)
  xs = (k_all[1:], cat_xs, noise.prior_z[:num_steps].astype(dtype), churn_xs, is_last)
  (xt, aatype_t, seq_probs_t, psce_t, seq_mlm, scn_mlm), traj = jax.lax.scan(mar_step, carry0, xs)

  if repack_last:
    # sidechain_pack(xt, aatype_t, ..., scn_override_mask, aatype_override_mask=seq_mlm): one denoiser call
    pack_aa = jnp.where(seq_mlm > 0, aatype_t, X_TOKEN) * (validf > 0).astype(jnp.int32)
    pack_aa = pack_aa.astype(jnp.int32)
    x_pack0 = xt.at[:, NON_BB_IDXS_ARR].multiply(sov[:, None, None])
    pack_common = {**common, "aatype_override": pack_aa, "aatype_override_mask": seq_mlm}
    call = denoiser_call(
      model,
      x_pack0,
      pack_aa,
      sov,
      temperature=0.0,  # pack's draw is discarded (every valid position is overridden); no uniform consumed
      categorical_u=None,
      prior_z=noise.prior_z[num_steps],
      churn_z=None if noise.churn_z is None else noise.churn_z[num_steps],
      **pack_common,
    )
    xt = unmask(x_pack0, call.x1_pred, sov, seq_mlm)
    psce_t = unmask(jnp.zeros((num_l, NUM_SCN_ATOMS), dtype), call.psce, sov, seq_mlm)

  return MarResult(
    x_final=xt,
    aatype_final=aatype_t,
    psce=psce_t,
    seq_probs=seq_probs_t,
    xt_traj=traj[0],
    aatype_t_traj=traj[1],
    aatype_pred_traj=traj[2],
    psce_t_traj=traj[3],
    seq_logits_traj=traj[4],
  )


# ---------------------------------------------------------------------------------------------
# Fixed-position strings (sampling_utils.py:107-150)
# ---------------------------------------------------------------------------------------------
def parse_fixed_positions(
  fixed_pos: str,
  chain_id_mapping: dict[str, int],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
) -> list[int]:
  """Absolute indices named by ``"A1,A10-25"``-style strings (``sampling_utils.parse_fixed_positions``).

  Items are ``[chain]start[-end]``; a missing chain letter means ``" "``. Unknown chains raise;
  residue numbers absent from the structure are skipped (upstream only prints a warning).
  """
  import re  # noqa: PLC0415

  import numpy as np  # noqa: PLC0415

  fixed: set[int] = set()
  text = fixed_pos.strip()
  if not text:
    return []
  ridx = np.asarray(residue_index)
  cidx = np.asarray(chain_index)
  for item in (p.strip() for p in text.split(",")):
    if not item:
      continue
    match = re.match(r"([A-Za-z])?(\d+)(?:-(\d+))?$", item)
    if not match:
      msg = f"Invalid position format: {item}"
      raise ValueError(msg)
    chain = match.group(1) if match.group(1) else " "
    start = int(match.group(2))
    end = int(match.group(3)) if match.group(3) else start
    if chain not in chain_id_mapping:
      msg = f"Chain ID {chain} not found in mapping."
      raise ValueError(msg)
    hit = (cidx == chain_id_mapping[chain]) & (ridx >= start) & (ridx <= end)
    fixed.update(int(j) for j in np.nonzero(hit)[0])
  return sorted(fixed)


def override_masks(
  fixed_pos_seq: str,
  fixed_pos_scn: str,
  chain_id_mapping: dict[str, int],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  *,
  mode: str = "seq_design",
) -> tuple[Float[Array, " L"], Float[Array, " L"]]:
  """``sampling_utils.get_override_masks`` for one structure.

  ``mode="seq_design"`` starts from no revealed sequence or sidechains; ``"packing"`` starts with the
  whole sequence revealed, and a non-empty ``fixed_pos_seq`` first clears it.
  """
  import numpy as np  # noqa: PLC0415

  num_l = int(np.asarray(residue_index).shape[0])
  aov = np.zeros(num_l) if mode == "seq_design" else np.ones(num_l)
  sov = np.zeros(num_l)
  if fixed_pos_seq:
    if mode == "packing":
      aov[:] = 0
    aov[parse_fixed_positions(fixed_pos_seq, chain_id_mapping, residue_index, chain_index)] = 1
  if fixed_pos_scn:
    sov[parse_fixed_positions(fixed_pos_scn, chain_id_mapping, residue_index, chain_index)] = 1
  return jnp.asarray(aov), jnp.asarray(sov)


__all__ = [
  "DenoiseCall",
  "EdmTrace",
  "MarResult",
  "SamplerNoise",
  "categorical_inverse_cdf",
  "decoding_order",
  "denoiser_call",
  "draw_sampler_noise",
  "edm_sigma",
  "edm_sigma_inv",
  "encoder_backbone",
  "local_to_global_scn",
  "mar_sample_upstream",
  "mask_logits_and_scale",
  "override_masks",
  "parse_fixed_positions",
  "psce_from_local",
  "scn_denoise_local",
  "scn_edm_loop",
  "timestep_schedule",
  "unmask",
  "unmask_targets",
  "update_mlm_mask",
]

# keep type checkers happy about imported-but-conditional names
_ = (eqx, _layer_norm)
