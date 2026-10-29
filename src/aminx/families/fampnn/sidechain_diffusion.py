"""FAMPNN conditional EDM sidechain diffusion module and optional pSCE confidence head.

Ports ``SidechainMLP``, ``SidechainDiffusionModule``, ``SidechainConfidenceModule``,
and Karras EDM sidechain sampling from ``richardshuai/fampnn``
(``fampnn/model/scn_diffusion.py`` and ``fampnn/model/confidence.py``).
"""

from __future__ import annotations

import math

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
  SidechainProteinFeatures,
  _layer_norm,
  _linear,
  atom37_to_atom14,
  global_to_local_atom37,
  local_to_global_atom37,
)
from aminx.families.fampnn.gvp import GVPEncoder


def _modulate(
  x: Float[Array, "... D"],
  shift: Float[Array, "... D"],
  scale: Float[Array, "... D"],
) -> Float[Array, "... D"]:
  """Apply DiT-style adaptive layer norm modulation ``x * (1 + scale) + shift``."""
  return x * (1.0 + scale) + shift


class FAMPNNTimestepEmbedder(eqx.Module):
  """Sinusoidal timestep embedder with 2-layer SiLU MLP for FAMPNN sidechain diffusion.

  Attributes
  ----------
  mlp_0 : eqx.nn.Linear
    First linear projection ``frequency_embedding_size -> hidden_size``.
  mlp_2 : eqx.nn.Linear
    Second linear projection ``hidden_size -> hidden_size``.
  frequency_embedding_size : int
    Sinusoidal embedding dimension (default ``256``).
  """

  mlp_0: eqx.nn.Linear
  mlp_2: eqx.nn.Linear
  frequency_embedding_size: int = eqx.field(static=True, default=256)

  def __init__(
    self,
    hidden_size: int = 256,
    frequency_embedding_size: int = 256,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNTimestepEmbedder``.

    Parameters
    ----------
    hidden_size : int, default=256
      Output hidden dimension.
    frequency_embedding_size : int, default=256
      Sinusoidal frequency embedding dimension.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    self.mlp_0 = eqx.nn.Linear(frequency_embedding_size, hidden_size, key=k1)
    self.mlp_2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
    self.frequency_embedding_size = frequency_embedding_size

  def __call__(self, t: Float[Array, " L"]) -> Float[Array, "L D"]:
    """Embed per-residue timesteps ``t`` into ``(L, hidden_size)``.

    Parameters
    ----------
    t : Float[Array, " L"]
      Timestep values ``c_noise(sigma)`` of shape ``(L,)``.

    Returns
    -------
    Float[Array, "L D"]
      Embedded timestep features ``(L, hidden_size)``.
    """
    dtype = self.mlp_0.weight.dtype
    half = self.frequency_embedding_size // 2
    freqs = jnp.exp(
      -math.log(10000.0) * jnp.arange(0, half, dtype=dtype) / half,
    )
    args = t.astype(dtype)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return _linear(self.mlp_2, jax.nn.silu(_linear(self.mlp_0, emb)))


class FAMPNNDenoisingMLPBlock(eqx.Module):
  """AdaLN-Zero modulated denoising MLP block with tanh-approximate GELU and ``eps=1e-6``.

  Attributes
  ----------
  norm : eqx.nn.LayerNorm
    Unparameterized LayerNorm (``use_weight=False, use_bias=False, eps=1e-6``).
  mlp_fc1 : eqx.nn.Linear
    First FFN linear ``hidden_size -> 4 * hidden_size``.
  mlp_fc2 : eqx.nn.Linear
    Second FFN linear ``4 * hidden_size -> hidden_size``.
  adaLN_modulation : eqx.nn.Linear
    Conditioning projection ``hidden_size -> 3 * hidden_size``.
  """

  norm: eqx.nn.LayerNorm
  mlp_fc1: eqx.nn.Linear
  mlp_fc2: eqx.nn.Linear
  adaLN_modulation: eqx.nn.Linear  # noqa: N815

  def __init__(
    self,
    hidden_size: int = 256,
    *,
    mlp_ratio: int = 4,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNDenoisingMLPBlock``.

    Parameters
    ----------
    hidden_size : int, default=256
      Block hidden dimension.
    mlp_ratio : int, default=4
      FFN expansion ratio.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    self.norm = eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False, eps=1e-6)
    self.mlp_fc1 = eqx.nn.Linear(hidden_size, hidden_size * mlp_ratio, key=k1)
    self.mlp_fc2 = eqx.nn.Linear(hidden_size * mlp_ratio, hidden_size, key=k2)
    self.adaLN_modulation = eqx.nn.Linear(hidden_size, 3 * hidden_size, key=k3)

  def __call__(
    self,
    x: Float[Array, "L D"],
    c: Float[Array, "L D"],
  ) -> Float[Array, "L D"]:
    """Apply AdaLN-Zero residual MLP block with ``approximate=True`` GELU.

    Parameters
    ----------
    x : Float[Array, "L D"]
      Token sidechain state ``(L, hidden_size)``.
    c : Float[Array, "L D"]
      Conditioning vector ``(L, hidden_size)``.

    Returns
    -------
    Float[Array, "L D"]
      Updated state ``(L, hidden_size)``.
    """
    mod = _linear(self.adaLN_modulation, jax.nn.silu(c))
    shift, scale, gate = jnp.split(mod, 3, axis=-1)
    h = _modulate(_layer_norm(self.norm, x), shift, scale)
    h = _linear(
      self.mlp_fc2,
      jax.nn.gelu(_linear(self.mlp_fc1, h), approximate=True),
    )
    return x + gate * h


class FAMPNNFinalLayer(eqx.Module):
  """AdaLN-Zero output projection layer for FAMPNN sidechain diffusion (``256 -> 99``).

  Attributes
  ----------
  norm_final : eqx.nn.LayerNorm
    Unparameterized LayerNorm (``use_weight=False, use_bias=False, eps=1e-6``).
  linear : eqx.nn.Linear
    Output projection ``hidden_size -> out_channels`` (``256 -> 99``).
  adaLN_modulation : eqx.nn.Linear
    Conditioning projection ``hidden_size -> 2 * hidden_size``.
  """

  norm_final: eqx.nn.LayerNorm
  linear: eqx.nn.Linear
  adaLN_modulation: eqx.nn.Linear  # noqa: N815

  def __init__(
    self,
    hidden_size: int = 256,
    out_channels: int = 99,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNFinalLayer``.

    Parameters
    ----------
    hidden_size : int, default=256
      Hidden state dimension.
    out_channels : int, default=99
      Flattened non-backbone Atom37 coordinate dimension (``33 * 3 = 99``).
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    self.norm_final = eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False, eps=1e-6)
    self.linear = eqx.nn.Linear(hidden_size, out_channels, key=k1)
    self.adaLN_modulation = eqx.nn.Linear(hidden_size, 2 * hidden_size, key=k2)

  def __call__(
    self,
    x: Float[Array, "L D"],
    c: Float[Array, "L D"],
  ) -> Float[Array, "L O"]:
    """Project modulated hidden state to ``(L, 99)`` denoised sidechain coordinates.

    Parameters
    ----------
    x : Float[Array, "L D"]
      Hidden state ``(L, hidden_size)``.
    c : Float[Array, "L D"]
      Conditioning vector ``(L, hidden_size)``.

    Returns
    -------
    Float[Array, "L O"]
      Output projection ``(L, 99)``.
    """
    mod = _linear(self.adaLN_modulation, jax.nn.silu(c))
    shift, scale = jnp.split(mod, 2, axis=-1)
    h = _modulate(_layer_norm(self.norm_final, x), shift, scale)
    return _linear(self.linear, h)


class SidechainDiffusionModule(eqx.Module):
  """FAMPNN conditional EDM sidechain diffusion module (``in_channels=219 -> 256 -> 99``).

  Matches ``fampnn/model/scn_diffusion.py::SidechainDiffusionModule``:
  - ``mlp_s_c``: ``Linear(21 + 128 = 149, 128) -> LayerNorm(128)``
  - ``denoiser.x_embedder``: ``Linear(33*3 (noisy) + 4*3 (bb local) + 33*3 (ref_pos) + 33 (atom_mask) - wait: 33*3 + 37*3 + 37 ? Wait: let's verify exact 219 breakdown: 33*3 (noisy scn = 99) + 4*3 (bb local N,CA,C,O = 12) + 36*3? No: 33*3 (99) + 33*3 (99) + 21 (aatype_onehot) = 219!``)
  - ``denoiser.t_embedder``: ``FAMPNNTimestepEmbedder(256)``
  - ``denoiser.c_embedder``: ``Linear(128, 256)``
  - ``denoiser.blocks``: 3 ``FAMPNNDenoisingMLPBlock(256)``
  - ``denoiser.final_layer``: ``FAMPNNFinalLayer(256, 99)``

  Attributes
  ----------
  mlp_s_c_lin : eqx.nn.Linear
    Conditioning fusion linear ``149 -> 128``.
  mlp_s_c_norm : eqx.nn.LayerNorm
    Conditioning fusion LayerNorm ``(128,)``.
  x_embedder : eqx.nn.Linear
    Input projection ``219 -> 256``.
  t_embedder : FAMPNNTimestepEmbedder
    Sinusoidal timestep embedder ``256 -> 256``.
  c_embedder : eqx.nn.Linear
    Conditioning projection ``128 -> 256``.
  blocks : tuple[FAMPNNDenoisingMLPBlock, ...]
    Tuple of 3 ``FAMPNNDenoisingMLPBlock`` layers.
  final_layer : FAMPNNFinalLayer
    Output projection ``256 -> 99``.
  sigma_data : float
    EDM coordinate standard deviation (default ``1.0`` after dividing by ``1.9``).
  """

  mlp_s_c_lin: eqx.nn.Linear
  mlp_s_c_norm: eqx.nn.LayerNorm
  x_embedder: eqx.nn.Linear
  t_embedder: FAMPNNTimestepEmbedder
  c_embedder: eqx.nn.Linear
  blocks: tuple[FAMPNNDenoisingMLPBlock, ...]
  final_layer: FAMPNNFinalLayer
  f_embedder: eqx.nn.Linear | None = None
  sigma_data: float = eqx.field(static=True, default=1.0)

  def __init__(
    self,
    in_channels: int = 219,
    hidden_size: int = 256,
    out_channels: int = 99,
    depth: int = 3,
    cond_dim: int = 128,
    *,
    has_f_embedder: bool = False,
    sigma_data: float = 1.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``SidechainDiffusionModule``.

    Parameters
    ----------
    in_channels : int
      Parameter ``in_channels``.
    hidden_size : int
      Parameter ``hidden_size``.
    out_channels : int
      Parameter ``out_channels``.
    depth : int
      Parameter ``depth``.
    cond_dim : int
      Parameter ``cond_dim``.
    has_f_embedder : bool
      Parameter ``has_f_embedder``.
    sigma_data : float
      Parameter ``sigma_data``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6 + depth)
    self.mlp_s_c_lin = eqx.nn.Linear(21 + cond_dim, cond_dim, key=keys[0])
    self.mlp_s_c_norm = eqx.nn.LayerNorm(cond_dim)
    self.x_embedder = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])
    self.t_embedder = FAMPNNTimestepEmbedder(hidden_size=hidden_size, key=keys[2])
    self.c_embedder = eqx.nn.Linear(cond_dim, hidden_size, key=keys[3])
    self.f_embedder = eqx.nn.Linear(111, hidden_size, key=keys[4]) if has_f_embedder else None
    self.blocks = tuple(
      FAMPNNDenoisingMLPBlock(hidden_size=hidden_size, key=k)
      for k in keys[5 : 5 + depth]
    )
    self.final_layer = FAMPNNFinalLayer(
      hidden_size=hidden_size,
      out_channels=out_channels,
      key=keys[-1],
    )
    self.sigma_data = sigma_data

  def compute_conditioning(
    self,
    h_v: Float[Array, "L 128"],
    aatype_onehot: Float[Array, "L 21"],
  ) -> Float[Array, "L 128"]:
    """Fuse encoder node embedding ``h_v`` and ``aatype_onehot`` via ``mlp_s_c``.

    Parameters
    ----------
    h_v : Float[Array, 'L 128']
      Parameter ``h_v``.
    aatype_onehot : Float[Array, 'L 21']
      Parameter ``aatype_onehot``.

    Returns
    -------
    Float[Array, 'L 128']
      Result of ``compute_conditioning``.
    """
    dtype = self.mlp_s_c_lin.weight.dtype
    cat = jnp.concatenate([aatype_onehot.astype(dtype), h_v.astype(dtype)], axis=-1)
    return _layer_norm(self.mlp_s_c_norm, _linear(self.mlp_s_c_lin, cat))

  def denoise_step(
    self,
    x_noisy_33: Float[Array, "L 33 3"],
    sigma: Float[Array, " L"],
    s_c: Float[Array, "L 128"],
    ref_pos_33: Float[Array, "L 33 3"],
    scn_mask_33: Float[Array, "L 33"],
    aatype_onehot: Float[Array, "L 21"],
    *,
    x_self_cond_33: Float[Array, "L 33 3"] | None = None,
    ref_pos_37: Float[Array, "L 37 3"] | None = None,
  ) -> Float[Array, "L 33 3"]:
    """Evaluate EDM-preconditioned denoiser ``D_theta(x_noisy_33, sigma)``.

    Parameters
    ----------
    x_noisy_33 : Float[Array, 'L 33 3']
      Parameter ``x_noisy_33``.
    sigma : Float[Array, ' L']
      Parameter ``sigma``.
    s_c : Float[Array, 'L 128']
      Parameter ``s_c``.
    ref_pos_33 : Float[Array, 'L 33 3']
      Parameter ``ref_pos_33``.
    scn_mask_33 : Float[Array, 'L 33']
      Parameter ``scn_mask_33``.
    aatype_onehot : Float[Array, 'L 21']
      Parameter ``aatype_onehot``.
    x_self_cond_33 : Float[Array, 'L 33 3'] | None, optional
      Optional self-conditioning sidechain coordinate estimate ``(L, 33, 3)``.
    ref_pos_37 : Float[Array, 'L 37 3'] | None, optional
      Optional full Atom37 reference conformer coordinates ``(L, 37, 3)``.

    Returns
    -------
    Float[Array, 'L 33 3']
      Result of ``denoise_step``.
    """
    dtype = self.x_embedder.weight.dtype
    num_l = x_noisy_33.shape[0]
    sigma_f = sigma.astype(dtype)
    sd2 = self.sigma_data * self.sigma_data
    sig2 = sigma_f * sigma_f

    c_skip = sd2 / (sig2 + sd2)
    c_out = sigma_f * self.sigma_data / jnp.sqrt(sig2 + sd2)
    c_in = 1.0 / jnp.sqrt(sd2 + sig2)
    c_noise = 0.25 * jnp.log(jnp.maximum(sigma_f, 1e-8))

    mask_f = scn_mask_33.astype(dtype)
    x_scaled = (c_in[:, None, None] * x_noisy_33.astype(dtype)) * mask_f[..., None]
    cond_source = ref_pos_33 if x_self_cond_33 is None else x_self_cond_33
    cond_scaled = (cond_source.astype(dtype) / self.sigma_data) * mask_f[..., None]
    x_scn = jnp.concatenate([x_scaled, cond_scaled], axis=-1).reshape(num_l, 33 * 6)

    in_219 = jnp.concatenate([x_scn, aatype_onehot.astype(dtype)], axis=-1)
    h = _linear(self.x_embedder, in_219)
    c_h = _linear(self.c_embedder, s_c.astype(dtype))
    if self.f_embedder is not None:
      if ref_pos_37 is not None:
        ref_111 = ref_pos_37.astype(dtype).reshape(num_l, 37 * 3)
      else:
        full_ref = jnp.zeros((num_l, 37, 3), dtype=dtype).at[:, NON_BB_IDXS_ARR, :].set(
          ref_pos_33.astype(dtype) * mask_f[..., None],
        )
        ref_111 = full_ref.reshape(num_l, 37 * 3)
      h = h + _linear(self.f_embedder, ref_111) + c_h
    c = self.t_embedder(c_noise) + c_h
    for blk in self.blocks:
      h = blk(h, c)
    f_x = self.final_layer(h, c).reshape(num_l, 33, 3)

    d_x = c_skip[:, None, None] * x_noisy_33.astype(dtype) + c_out[:, None, None] * f_x
    return d_x * mask_f[..., None]


def edm_sample(
  diffusion: SidechainDiffusionModule,
  atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask: Float[Array, "L 37"],
  h_v: Float[Array, "L 128"],
  aatype: Int[Array, " L"],
  *,
  num_steps: int = 18,
  sigma_min: float = 0.002,
  sigma_max: float = 80.0,
  rho: float = 7.0,
  s_churn: float = 0.0,
  s_min: float = 0.0,
  s_max: float = float("inf"),
  s_noise: float = 1.0,
  step_scale: float = 1.5,
  coord_scale: float = 1.9,
  key: jax.Array | None = None,
) -> Float[Array, "L 37 3"]:
  """Sample full-atom sidechain coordinates via Karras EDM schedule using ``lax.fori_loop``.

  Parameters
  ----------
  diffusion : SidechainDiffusionModule
    Parameter ``diffusion``.
  atom37_xyz : Float[Array, 'L 37 3']
    Parameter ``atom37_xyz``.
  atom37_mask : Float[Array, 'L 37']
    Parameter ``atom37_mask``.
  h_v : Float[Array, 'L 128']
    Parameter ``h_v``.
  aatype : Int[Array, ' L']
    Parameter ``aatype``.
  num_steps : int
    Parameter ``num_steps``.
  sigma_min : float
    Parameter ``sigma_min``.
  sigma_max : float
    Parameter ``sigma_max``.
  rho : float
    Parameter ``rho``.
  s_churn : float
    Parameter ``s_churn``.
  s_min : float
    Parameter ``s_min``.
  s_max : float
    Parameter ``s_max``.
  s_noise : float
    Parameter ``s_noise``.
  step_scale : float
    Parameter ``step_scale``.
  coord_scale : float
    Parameter ``coord_scale``.
  key : jax.Array | None
    Parameter ``key``.

  Returns
  -------
  Float[Array, 'L 37 3']
    Result of ``edm_sample``.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  dtype = diffusion.x_embedder.weight.dtype
  num_l = atom37_xyz.shape[0]
  aa_clamped = jnp.clip(aatype, 0, 20)
  aatype_oh = jax.nn.one_hot(aa_clamped, 21, dtype=dtype)

  std_mask37 = RESTYPE_ATOM37_MASK[aa_clamped].astype(dtype)
  scn_mask_33 = jnp.take(std_mask37, NON_BB_IDXS_ARR, axis=1)
  ref_pos_37 = RESTYPE_REF_POS_ATOM37[aa_clamped].astype(dtype) / coord_scale
  ref_pos_33 = jnp.take(ref_pos_37, NON_BB_IDXS_ARR, axis=1) * scn_mask_33[..., None]

  s_c = h_v.astype(dtype) if diffusion.f_embedder is not None else diffusion.compute_conditioning(h_v, aatype_oh)

  step_indices = jnp.arange(num_steps, dtype=dtype)
  inv_rho = 1.0 / rho
  s_max_rho = sigma_max**inv_rho
  s_min_rho = sigma_min**inv_rho
  denom = max(num_steps - 1, 1)
  t_steps = ((s_max_rho + step_indices / denom * (s_min_rho - s_max_rho)) ** rho) * diffusion.sigma_data
  t_steps_padded = jnp.concatenate([t_steps, jnp.zeros((1,), dtype=dtype)], axis=0)

  k_init, k_loop = jax.random.split(key, 2)
  x_init = (
    jax.random.normal(k_init, (num_l, 33, 3), dtype=dtype)
    * t_steps_padded[0]
    * scn_mask_33[..., None]
  )
  x_cond_init = jnp.zeros_like(x_init)
  gamma_val = min(s_churn / max(num_steps, 1), math.sqrt(2.0) - 1.0) if s_churn > 0.0 else 0.0

  def _body_fn(
    step_i: int,
    carry: tuple[Float[Array, "L 33 3"], Float[Array, "L 33 3"]],
  ) -> tuple[Float[Array, "L 33 3"], Float[Array, "L 33 3"]]:
    x_cur, x_cond_cur = carry
    t_cur = t_steps_padded[step_i]
    t_next = t_steps_padded[step_i + 1]
    step_key = jax.random.fold_in(k_loop, step_i)

    in_churn = (t_cur >= s_min) & (t_cur <= s_max)
    gamma = jnp.where(in_churn, gamma_val, 0.0)
    t_hat = t_cur + gamma * t_cur
    noise_std = jnp.sqrt(jnp.maximum(t_hat * t_hat - t_cur * t_cur, 0.0)) * s_noise
    eps = jax.random.normal(step_key, x_cur.shape, dtype=dtype) * scn_mask_33[..., None]
    x_hat = x_cur + noise_std * eps

    sigma_vec = jnp.broadcast_to(t_hat, (num_l,))
    denoised = diffusion.denoise_step(
      x_hat,
      sigma_vec,
      s_c,
      ref_pos_33,
      scn_mask_33,
      aatype_oh,
      x_self_cond_33=x_cond_cur,
      ref_pos_37=ref_pos_37 * std_mask37[..., None],
    )
    d_cur = (x_hat - denoised) / jnp.maximum(t_hat, 1e-8)
    x_next = x_hat + step_scale * (t_next - t_hat) * d_cur
    return x_next * scn_mask_33[..., None], denoised * scn_mask_33[..., None]

  x_local_33, _ = jax.lax.fori_loop(0, num_steps, _body_fn, (x_init, x_cond_init))

  # Assemble full local Atom37 and transform back to global coordinates
  local_bb = global_to_local_atom37(atom37_xyz, atom37_mask, scale=coord_scale)
  local_full = local_bb.at[:, NON_BB_IDXS_ARR, :].set(x_local_33)
  full_mask37 = jnp.maximum(atom37_mask.astype(dtype), std_mask37)
  global_full = local_to_global_atom37(
    local_full, atom37_xyz, full_mask37, scale=coord_scale,
  )
  # Preserve exact input backbone slots (0, 1, 2, 4)
  bb_slots = jnp.array([0, 1, 2, 4], dtype=jnp.int32)
  return global_full.at[:, bb_slots, :].set(atom37_xyz[:, bb_slots, :])


class SidechainConfidenceModule(eqx.Module):
  """Optional FAMPNN sidechain confidence (pSCE) module."""

  features: SidechainProteinFeatures
  gvp_encoder: GVPEncoder
  W_v: eqx.nn.Linear
  W_e: eqx.nn.Linear
  W_s: eqx.nn.Embedding
  encoder_layers: tuple[eqx.Module, ...]
  decoder_layers: tuple[eqx.Module, ...]
  W_out_0: eqx.nn.Linear
  W_out_2: eqx.nn.Linear
  W_out_4: eqx.nn.Linear
  aatype_embedder: eqx.nn.Linear | None = None
  sidechain_encoder: eqx.nn.Linear | None = None
  mlp_0: eqx.nn.Linear | None = None
  mlp_2: eqx.nn.Linear | None = None
  num_bins: int = eqx.field(static=True, default=64)

  def __init__(
    self,
    encoder_layers: tuple[eqx.Module, ...],
    decoder_layers: tuple[eqx.Module, ...],
    hidden_dim: int = 128,
    num_bins: int = 64,
    *,
    has_upstream_heads: bool = False,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``SidechainConfidenceModule``.

    Parameters
    ----------
    encoder_layers : tuple[eqx.Module, ...]
      Parameter ``encoder_layers``.
    decoder_layers : tuple[eqx.Module, ...]
      Parameter ``decoder_layers``.
    hidden_dim : int
      Parameter ``hidden_dim``.
    num_bins : int
      Parameter ``num_bins``.
    has_upstream_heads : bool
      Parameter ``has_upstream_heads``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 12)
    self.features = SidechainProteinFeatures(edge_features=hidden_dim, key=keys[0])
    self.gvp_encoder = GVPEncoder(key=keys[1])
    self.W_v = eqx.nn.Linear(hidden_dim + 176, hidden_dim, key=keys[2])
    self.W_e = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[3])
    self.W_s = eqx.nn.Embedding(21, hidden_dim, key=keys[4])
    self.encoder_layers = encoder_layers
    self.decoder_layers = decoder_layers
    self.W_out_0 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[5])
    self.W_out_2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[6])
    self.W_out_4 = eqx.nn.Linear(hidden_dim, 33 * num_bins, key=keys[7])
    if has_upstream_heads:
      self.aatype_embedder = eqx.nn.Linear(21, hidden_dim, use_bias=False, key=keys[8])
      self.sidechain_encoder = eqx.nn.Linear(99, hidden_dim, use_bias=False, key=keys[9])
      self.mlp_0 = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=keys[10])
      self.mlp_2 = eqx.nn.Linear(hidden_dim, 33 * num_bins, use_bias=False, key=keys[11])
    else:
      self.aatype_embedder = None
      self.sidechain_encoder = None
      self.mlp_0 = None
      self.mlp_2 = None
    self.num_bins = num_bins

  def __call__(
    self,
    atom37_xyz: Float[Array, "L 37 3"],
    atom37_mask: Float[Array, "L 37"],
    h_v_fa: Float[Array, "L 128"],
    aatype: Int[Array, " L"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
    *,
    h_esv: Float[Array, "L K 512"] | None = None,
    e_idx: Int[Array, "L K"] | None = None,
  ) -> Float[Array, "L 37"]:
    """Predict per-atom expected sidechain error (pSCE) in Atom37 representation.

    Parameters
    ----------
    atom37_xyz : Float[Array, 'L 37 3']
      Parameter ``atom37_xyz``.
    atom37_mask : Float[Array, 'L 37']
      Parameter ``atom37_mask``.
    h_v_fa : Float[Array, 'L 128']
      Parameter ``h_v_fa``.
    aatype : Int[Array, ' L']
      Parameter ``aatype``.
    mask : Float[Array, ' L']
      Parameter ``mask``.
    residue_idx : Int[Array, ' L']
      Parameter ``residue_idx``.
    chain_idx : Int[Array, ' L']
      Parameter ``chain_idx``.
    h_esv : Float[Array, 'L K 512'] | None
      Parameter ``h_esv``.
    e_idx : Int[Array, 'L K'] | None
      Parameter ``e_idx``.

    Returns
    -------
    Float[Array, 'L 37']
      Result of ``__call__``.
    """
    dtype = self.W_v.weight.dtype
    num_l = atom37_xyz.shape[0]
    aa_clamped = jnp.clip(aatype, 0, 20)
    aatype_oh = jax.nn.one_hot(aa_clamped, 21, dtype=dtype)
    mask_f = mask.astype(dtype)

    if (
      self.aatype_embedder is not None
      and self.sidechain_encoder is not None
      and self.mlp_0 is not None
      and self.mlp_2 is not None
    ):
      h_v_enc = h_v_fa.astype(dtype)
      if h_esv is not None and e_idx is not None:
        mask_2d = mask_f[:, None] * mask_f[None, :]
        mask_attend = jnp.take_along_axis(mask_2d, e_idx, axis=1)
        h_esv_cur = h_esv.astype(dtype)
        h_s_dummy = jnp.zeros_like(h_v_enc)
        for dec_layer in self.decoder_layers:
          h_v_enc, h_esv_cur = dec_layer(h_v_enc, h_esv_cur, h_s_dummy, e_idx, mask_f, mask_attend)  # type: ignore[misc]
      local_37 = global_to_local_atom37(atom37_xyz, atom37_mask)
      scn_flat = jnp.take(local_37, NON_BB_IDXS_ARR, axis=1).reshape(num_l, 99)
      h_v = (
        h_v_enc
        + _linear(self.aatype_embedder, aatype_oh)
        + _linear(self.sidechain_encoder, scn_flat)
      )
      logits_33 = _linear(self.mlp_2, jax.nn.silu(_linear(self.mlp_0, h_v))).reshape(
        num_l, 33, self.num_bins,
      )
    else:
      atom14_xyz, atom14_mask = atom37_to_atom14(atom37_xyz, atom37_mask, aa_clamped)
      e_init, e_idx = self.features(atom14_xyz, atom14_mask, mask, residue_idx, chain_idx)
      gvp_out = self.gvp_encoder(atom14_xyz, atom14_mask, aatype_oh, mask, residue_idx, chain_idx)

      h_v = _linear(self.W_v, jnp.concatenate([h_v_fa.astype(dtype), gvp_out], axis=-1))
      h_e = _linear(self.W_e, e_init)
      mask_2d = mask_f[:, None] * mask_f[None, :]
      mask_attend = jnp.take_along_axis(mask_2d, e_idx, axis=1)

      for enc_layer in self.encoder_layers:
        h_v, h_e = enc_layer(h_v, h_e, e_idx, mask_f, mask_attend)  # type: ignore[misc]

      h_s = self.W_s.weight[aa_clamped]
      h_es = jnp.concatenate([h_e, h_s[e_idx]], axis=-1)
      for dec_layer in self.decoder_layers:
        h_esv = jnp.concatenate([h_es, h_v[e_idx]], axis=-1)
        h_v = dec_layer(h_v, h_esv, mask_f, mask_attend)  # type: ignore[misc]

      h_out = jax.nn.relu(_linear(self.W_out_0, h_v))
      h_out = jax.nn.relu(_linear(self.W_out_2, h_out))
      logits_33 = _linear(self.W_out_4, h_out).reshape(num_l, 33, self.num_bins)

    probs = jax.nn.softmax(logits_33, axis=-1)
    lower = jnp.linspace(0.0, 8.0, self.num_bins, dtype=dtype)
    bin_centers = lower + (lower[1] - lower[0]) * 0.5
    psce_33 = jnp.sum(probs * bin_centers[None, None, :], axis=-1)

    psce_37 = jnp.zeros((num_l, 37), dtype=dtype).at[:, NON_BB_IDXS_ARR].set(psce_33)
    return psce_37 * atom37_mask.astype(dtype)
