"""Caliby conditional EDM sidechain diffusion packer and Karras ODE/SDE sampler.

Ports ``SidechainMLP``, ``DenoisingMLPBlock``, ``TimestepEmbedder``, ``FinalLayer``,
and ``EDM`` sidechain diffusion sampling from ``ProteinDesignLab/caliby``
(``caliby/model/seq_denoiser/denoisers/scn_diffusion_module/``).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.caliby.features import _layer_norm, _linear
from aminx.families.caliby.model import CalibyMPNN

# 33 non-backbone Atom37 sidechain indices (all except N=0, CA=1, C=2, O=4)
AF2_SCN_IDXS: tuple[int, ...] = (3, *range(5, 37))
_AF2_SCN_IDXS_ARR: jax.Array = jnp.array(AF2_SCN_IDXS, dtype=jnp.int32)


def _modulate(
  x: Float[Array, "... D"],
  shift: Float[Array, "... D"],
  scale: Float[Array, "... D"],
) -> Float[Array, "... D"]:
  """Apply DiT-style adaptive layer norm modulation ``x * (1 + scale) + shift``."""
  return x * (1.0 + scale) + shift


def rigid_from_3_points(
  c_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  n_xyz: Float[Array, "L 3"],
  eps: float = 1e-8,
) -> tuple[Float[Array, "L 3 3"], Float[Array, "L 3"]]:
  """Construct Gram-Schmidt local rigid frames ``(R, t)`` from ``(C, CA, N)``.

  Matches ``atomworks.ml.transforms.af2_util.rigid_from_3_points(x1=C, x2=CA, x3=N)``
  used by upstream ``SidechainMLP._transform_to_local_frame``: ``e1`` along ``C - CA``,
  ``e2`` orthogonalized from ``N - CA``, ``e3 = e1 x e2``, and translation ``t = CA``.

  Parameters
  ----------
  c_xyz : Float[Array, "L 3"]
    Backbone carbonyl carbon C coordinates ``(L, 3)``.
  ca_xyz : Float[Array, "L 3"]
    Backbone alpha carbon CA coordinates ``(L, 3)``.
  n_xyz : Float[Array, "L 3"]
    Backbone amide nitrogen N coordinates ``(L, 3)``.
  eps : float, default=1e-8
    Small numerical constant inside ``sqrt``.

  Returns
  -------
  tuple[Float[Array, "L 3 3"], Float[Array, "L 3"]]
    Tuple ``(rot, ca_xyz)`` where ``rot`` has columns ``[e1, e2, e3]`` of shape ``(L, 3, 3)``.
  """
  v1 = c_xyz - ca_xyz
  v2 = n_xyz - ca_xyz
  e1 = v1 / jnp.sqrt(jnp.sum(v1 * v1, axis=-1, keepdims=True) + eps)
  u2 = v2 - jnp.sum(e1 * v2, axis=-1, keepdims=True) * e1
  e2 = u2 / jnp.sqrt(jnp.sum(u2 * u2, axis=-1, keepdims=True) + eps)
  e3 = jnp.cross(e1, e2)
  rot = jnp.stack([e1, e2, e3], axis=-1)
  return rot, ca_xyz


def _extract_scn33_mask(atom_mask: Float[Array, "L A"]) -> Float[Array, "L 33"]:
  """Extract the 33-atom sidechain mask from Atom37 ``(L, 37)`` or Atom23 ``(L, 23)`` mask."""
  num_a = atom_mask.shape[1]
  if num_a == 37:
    return jnp.take(atom_mask, _AF2_SCN_IDXS_ARR, axis=1)
  # Atom23: slots 4..22 are the 19 sidechain atoms (CB + 18 sidechain atoms), pad to 33
  sc19_mask = atom_mask[:, 4:23]
  return jnp.pad(sc19_mask, ((0, 0), (0, 33 - sc19_mask.shape[1])))


def global_to_local_scn(
  atom_xyz: Float[Array, "L A 3"],
  atom_mask: Float[Array, "L A"],
) -> Float[Array, "L 33 3"]:
  """Transform global sidechain coordinates into the local ``rigid_from_3_points(C, CA, N)`` frame.

  Matches upstream ``SidechainMLP._transform_to_local_frame(..., to_local=True)``
  operating on 33 local rigid-frame sidechain coordinates ``(L, 33, 3)``.

  Parameters
  ----------
  atom_xyz : Float[Array, "L A 3"]
    Global Atom23 ``(L, 23, 3)`` or Atom37 ``(L, 37, 3)`` coordinates.
  atom_mask : Float[Array, "L A"]
    Atom presence mask ``(L, 23)`` or ``(L, 37)``.

  Returns
  -------
  Float[Array, "L 33 3"]
    Local rigid-frame sidechain coordinates ``(L, 33, 3)``.
  """
  n_xyz = atom_xyz[:, 0, :]
  ca_xyz = atom_xyz[:, 1, :]
  c_xyz = atom_xyz[:, 2, :]
  rot, t_ca = rigid_from_3_points(c_xyz, ca_xyz, n_xyz)

  num_a = atom_xyz.shape[1]
  if num_a == 37:
    scn_global = jnp.take(atom_xyz, _AF2_SCN_IDXS_ARR, axis=1)
    scn_mask = jnp.take(atom_mask, _AF2_SCN_IDXS_ARR, axis=1)
  else:
    sc19_global = atom_xyz[:, 4:23, :]
    scn_global = jnp.pad(sc19_global, ((0, 0), (0, 33 - sc19_global.shape[1]), (0, 0)))
    scn_mask = _extract_scn33_mask(atom_mask)

  scn_local = jnp.einsum("lai,lij->laj", scn_global - t_ca[:, None, :], rot)
  return scn_local * scn_mask[..., None]


def local_to_global_scn(
  scn_local: Float[Array, "L 33 3"],
  atom_xyz: Float[Array, "L A 3"],
  atom_mask: Float[Array, "L A"],
) -> Float[Array, "L A 3"]:
  """Transform 33 local rigid-frame sidechain coordinates back to global coordinates.

  Matches upstream ``SidechainMLP._transform_to_local_frame(..., to_local=False)``.

  Parameters
  ----------
  scn_local : Float[Array, "L 33 3"]
    Local rigid-frame sidechain coordinates ``(L, 33, 3)``.
  atom_xyz : Float[Array, "L A 3"]
    Reference Atom23 ``(L, 23, 3)`` or Atom37 ``(L, 37, 3)`` carrying backbone N, CA, C, O.
  atom_mask : Float[Array, "L A"]
    Atom presence mask ``(L, 23)`` or ``(L, 37)``.

  Returns
  -------
  Float[Array, "L A 3"]
    Reconstructed global coordinates ``(L, 23, 3)`` or ``(L, 37, 3)``.
  """
  n_xyz = atom_xyz[:, 0, :]
  ca_xyz = atom_xyz[:, 1, :]
  c_xyz = atom_xyz[:, 2, :]
  rot, t_ca = rigid_from_3_points(c_xyz, ca_xyz, n_xyz)
  scn_mask = _extract_scn33_mask(atom_mask)
  scn_global = (jnp.einsum("laj,lij->lai", scn_local, rot) + t_ca[:, None, :]) * scn_mask[..., None]

  num_a = atom_xyz.shape[1]
  if num_a == 37:
    out37 = atom_xyz.at[:, _AF2_SCN_IDXS_ARR, :].set(scn_global)
    return out37 * atom_mask[..., None]
  return jnp.concatenate([atom_xyz[:, :4, :], scn_global[:, :19, :]], axis=1)


class TimestepEmbedder(eqx.Module):
  """Sinusoidal diffusion timestep embedder followed by a 2-layer SiLU MLP.

  Attributes
  ----------
  mlp_0 : eqx.nn.Linear
    First linear projection ``frequency_embedding_size -> hidden_size``.
  mlp_2 : eqx.nn.Linear
    Second linear projection ``hidden_size -> hidden_size``.
  frequency_embedding_size : int
    Sinusoidal frequency embedding dimension (default ``256``).
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
    """Initialize ``TimestepEmbedder``.

    Parameters
    ----------
    hidden_size : int, default=256
      Output conditioning dimension.
    frequency_embedding_size : int, default=256
      Sinusoidal embedding dimension.
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
    """Embed scalar or per-token timesteps ``t`` into ``(L, hidden_size)``.

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


class DenoisingMLPBlock(eqx.Module):
  """AdaLN-Zero modulated denoising MLP block for Caliby sidechain diffusion.

  Attributes
  ----------
  norm : eqx.nn.LayerNorm
    Unparameterized LayerNorm (``use_weight=False, use_bias=False, eps=1e-6``).
  mlp_fc1 : eqx.nn.Linear
    First FFN linear ``hidden_size -> 4 * hidden_size``.
  mlp_fc2 : eqx.nn.Linear
    Second FFN linear ``4 * hidden_size -> hidden_size``.
  adaLN_modulation : eqx.nn.Linear
    Conditioning projection ``hidden_size -> 3 * hidden_size`` producing ``(shift, scale, gate)``.
  """

  norm: eqx.nn.LayerNorm
  mlp_fc1: eqx.nn.Linear
  mlp_fc2: eqx.nn.Linear
  adaLN_modulation: eqx.nn.Linear  # noqa: N815

  def __init__(self, hidden_size: int = 256, *, mlp_ratio: int = 4, key: jax.Array | None = None) -> None:
    """Initialize ``DenoisingMLPBlock``.

    Parameters
    ----------
    hidden_size : int, default=256
      Block hidden dimension.
    mlp_ratio : int, default=4
      Expansion factor for the inner FFN layer.
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
    """Apply AdaLN-modulated residual MLP step.

    Parameters
    ----------
    x : Float[Array, "L D"]
      Token sidechain state ``(L, hidden_size)``.
    c : Float[Array, "L D"]
      Conditioning vector ``(L, hidden_size)``.

    Returns
    -------
    Float[Array, "L D"]
      Updated state ``x + gate * MLP(modulate(norm(x), shift, scale))``.
    """
    mod = _linear(self.adaLN_modulation, jax.nn.silu(c))
    shift, scale, gate = jnp.split(mod, 3, axis=-1)
    h = _modulate(_layer_norm(self.norm, x), shift, scale)
    h = _linear(
      self.mlp_fc2,
      jax.nn.gelu(_linear(self.mlp_fc1, h), approximate=True),
    )
    return x + gate * h


class FinalLayer(eqx.Module):
  """AdaLN-modulated output projection layer for Caliby sidechain denoising.

  Attributes
  ----------
  norm_final : eqx.nn.LayerNorm
    Unparameterized LayerNorm (``use_weight=False, use_bias=False, eps=1e-6``).
  linear : eqx.nn.Linear
    Output projection ``hidden_size -> out_channels`` (``256 -> 99``).
  adaLN_modulation : eqx.nn.Linear
    Conditioning projection ``hidden_size -> 2 * hidden_size`` producing ``(shift, scale)``.
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
    """Initialize ``FinalLayer``.

    Parameters
    ----------
    hidden_size : int, default=256
      Hidden state dimension.
    out_channels : int, default=99
      Flattened sidechain coordinate dimension (``33 * 3 = 99``).
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
    """Project modulated hidden state to denoised coordinate update.

    Parameters
    ----------
    x : Float[Array, "L D"]
      Hidden state ``(L, hidden_size)``.
    c : Float[Array, "L D"]
      Conditioning vector ``(L, hidden_size)``.

    Returns
    -------
    Float[Array, "L O"]
      Output projection of shape ``(L, out_channels)``.
    """
    mod = _linear(self.adaLN_modulation, jax.nn.silu(c))
    shift, scale = jnp.split(mod, 2, axis=-1)
    h = _modulate(_layer_norm(self.norm_final, x), shift, scale)
    return _linear(self.linear, h)


class CalibySidechainMLP(eqx.Module):
  """Caliby sidechain denoiser combining a sequence-conditioned MPNN and AdaLN MLP blocks.

  Matches upstream ``caliby_packer_{000,010,030}.ckpt`` where ``denoiser.atom_mpnn.*``
  (normalized to ``mpnn.*``) computes trunk node embeddings and ``x_embedder``,
  ``t_embedder``, ``c_embedder``, ``blocks``, and ``final_layer`` denoise the
  33 local rigid-frame Atom37 sidechain coordinates ``(33 * 3 = 99)`` with
  ``99 (noisy) + 99 (self-cond) + 32 (sequence one-hot) = 230`` input channels.

  Attributes
  ----------
  mpnn : CalibyMPNN
    Sequence-conditioned MPNN trunk (``use_sequence=True``).
  x_embedder : eqx.nn.Linear
    Input projection ``in_channels -> hidden_size`` (``230 -> 256``).
  t_embedder : TimestepEmbedder
    Sinusoidal timestep embedder (``256 -> 256``).
  c_embedder : eqx.nn.Linear
    Trunk conditioning projection ``mpnn_hidden -> hidden_size`` (``128 -> 256``).
  blocks : tuple[DenoisingMLPBlock, ...]
    Stack of ``depth`` (default ``3``) AdaLN denoising MLP blocks.
  final_layer : FinalLayer
    Final AdaLN output layer (``256 -> 99``).
  in_channels : int
    Input feature dimension (default ``230``).
  out_channels : int
    Output coordinate dimension (default ``99``).
  sigma_data : float
    EDM data standard deviation (default ``1.0``).
  """

  mpnn: CalibyMPNN
  x_embedder: eqx.nn.Linear
  t_embedder: TimestepEmbedder
  c_embedder: eqx.nn.Linear
  blocks: tuple[DenoisingMLPBlock, ...]
  final_layer: FinalLayer
  in_channels: int = eqx.field(static=True, default=230)
  out_channels: int = eqx.field(static=True, default=99)
  sigma_data: float = eqx.field(static=True, default=1.0)

  def __init__(
    self,
    in_channels: int = 230,
    hidden_size: int = 256,
    out_channels: int = 99,
    depth: int = 3,
    mpnn_hidden: int = 128,
    *,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    model_version: int = 0,
    update_decoder_edges: bool = False,
    has_w_out: bool = False,
    sigma_data: float = 1.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibySidechainMLP``.

    Parameters
    ----------
    in_channels : int, default=230
      Input feature channels (``99 + 99 + 32 = 230``).
    hidden_size : int, default=256
      Denoising MLP hidden dimension.
    out_channels : int, default=99
      Output sidechain coordinate channels (``33 * 3 = 99``).
    depth : int, default=3
      Number of ``DenoisingMLPBlock`` layers.
    mpnn_hidden : int, default=128
      Trunk MPNN hidden dimension.
    num_encoder_layers : int, default=3
      Number of MPNN encoder layers.
    num_decoder_layers : int, default=3
      Number of MPNN decoder layers.
    model_version : int, default=0
      Feature layout version (``0`` or ``1``).
    update_decoder_edges : bool, default=False
      Whether decoder layers update edge features.
    has_w_out : bool, default=False
      Whether the MPNN trunk includes ``W_out``.
    sigma_data : float, default=1.0
      EDM data standard deviation.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5 + depth)
    self.mpnn = CalibyMPNN(
      node_features=mpnn_hidden,
      edge_features=mpnn_hidden,
      hidden_dim=mpnn_hidden,
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      k_neighbors=48,
      num_positional_embeddings=16,
      num_rbf=16,
      model_version=model_version,
      use_sequence=True,
      update_decoder_edges=update_decoder_edges,
      update_last_decoder_edge=False,
      return_decoder_edges=False,
      has_w_out=has_w_out,
      vocab=32,
      key=keys[0],
    )
    self.x_embedder = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])
    self.t_embedder = TimestepEmbedder(hidden_size=hidden_size, key=keys[2])
    self.c_embedder = eqx.nn.Linear(mpnn_hidden, hidden_size, key=keys[3])
    self.blocks = tuple(
      DenoisingMLPBlock(hidden_size=hidden_size, key=k) for k in keys[4 : 4 + depth]
    )
    self.final_layer = FinalLayer(hidden_size=hidden_size, out_channels=out_channels, key=keys[-1])
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.sigma_data = sigma_data

  def denoise_from_trunk(
    self,
    x_noisy: Float[Array, "L 33 3"],
    sigma: Float[Array, " L"],
    h_v: Float[Array, "L H"],
    scn_mask: Float[Array, "L 33"],
    x_cond: Float[Array, "L 33 3"] | None = None,
    aatype: Int[Array, " L"] | None = None,
  ) -> Float[Array, "L 33 3"]:
    """Run EDM-preconditioned denoising step ``D_theta(x_noisy, sigma)`` on 33 local sidechain atoms.

    Parameters
    ----------
    x_noisy : Float[Array, "L 33 3"]
      Noisy local rigid-frame sidechain coordinates ``(L, 33, 3)``.
    sigma : Float[Array, " L"]
      Per-token noise levels ``(L,)``.
    h_v : Float[Array, "L H"]
      Cached MPNN trunk node representations ``(L, 128)``.
    scn_mask : Float[Array, "L 33"]
      Sidechain atom validity mask ``(L, 33)``.
    x_cond : Float[Array, "L 33 3"] | None, optional
      Self-conditioning sidechain coordinate estimate ``(L, 33, 3)``.
    aatype : Int[Array, " L"] | None, optional
      Residue type indices in AF3 32-token order ``0..31``.

    Returns
    -------
    Float[Array, "L 33 3"]
      Denoised local rigid-frame sidechain coordinates ``(L, 33, 3)``.
    """
    dtype = self.x_embedder.weight.dtype
    num_l = x_noisy.shape[0]
    sigma_f = sigma.astype(dtype)
    sd2 = self.sigma_data * self.sigma_data
    sig2 = sigma_f * sigma_f

    c_skip = sd2 / (sig2 + sd2)
    c_out = sigma_f * self.sigma_data / jnp.sqrt(sig2 + sd2)
    c_in = 1.0 / jnp.sqrt(sd2 + sig2)
    c_noise = 0.25 * jnp.log(jnp.maximum(sigma_f, 1e-8))

    mask_f = scn_mask.astype(dtype)
    x_scaled = (c_in[:, None, None] * x_noisy.astype(dtype)) * mask_f[..., None]
    cond_scaled = (
      jnp.zeros_like(x_scaled)
      if x_cond is None
      else (x_cond.astype(dtype) / self.sigma_data) * mask_f[..., None]
    )
    x_scn = jnp.concatenate([x_scaled, cond_scaled], axis=-1).reshape(num_l, 33 * 6)
    seq_tokens = (
      jnp.zeros((num_l,), dtype=jnp.int32)
      if aatype is None
      else jnp.clip(aatype, 0, 31)
    )
    seq_oh = jax.nn.one_hot(seq_tokens, 32, dtype=dtype)

    in_feat = jnp.concatenate([x_scn, seq_oh], axis=-1)
    h_v_emb = _linear(self.c_embedder, h_v.astype(dtype))
    h = _linear(self.x_embedder, in_feat) + h_v_emb
    c = self.t_embedder(c_noise) + h_v_emb
    for block in self.blocks:
      h = block(h, c)
    f_x = self.final_layer(h, c).reshape(num_l, 33, 3)

    d_x = c_skip[:, None, None] * x_noisy.astype(dtype) + c_out[:, None, None] * f_x
    return d_x * mask_f[..., None]


def edm_sidechain_sample(
  packer: CalibySidechainMLP,
  atom23_xyz: Float[Array, "L 23 3"],
  atom23_mask: Float[Array, "L 23"],
  token_mask: Float[Array, " L"],
  residue_idx: Int[Array, " L"],
  token_idx: Int[Array, " L"],
  asym_id: Int[Array, " L"],
  entity_id: Int[Array, " L"],
  sym_id: Int[Array, " L"],
  aatype: Int[Array, " L"],
  *,
  num_steps: int = 25,
  sigma_min: float = 0.002,
  sigma_max: float = 80.0,
  rho: float = 7.0,
  s_churn: float = 0.0,
  s_min: float = 0.0,
  s_max: float = float("inf"),
  s_noise: float = 1.0,
  step_scale: float = 1.5,
  use_heun: bool = False,
  key: jax.Array | None = None,
  molecule_type_encoding: Float[Array, "L 5"] | None = None,
  is_ligand: Bool[Array, " L"] | None = None,
) -> Float[Array, "L 23 3"]:
  """Sample all-atom sidechain coordinates via Karras EDM discretized schedule.

  Uses ``jax.lax.scan`` over the ``num_steps`` Karras noise schedule levels
  (registered in ``tests/lint/l_drv_allowlist.toml`` with ``reason = "sequential_dependency"``).

  Parameters
  ----------
  packer : CalibySidechainMLP
    Loaded Caliby sidechain diffusion model.
  atom23_xyz : Float[Array, "L 23 3"]
    Input Atom23 coordinates (with backbone slots ``0..3`` populated).
  atom23_mask : Float[Array, "L 23"]
    Atom23 validity mask for the target sequence.
  token_mask : Float[Array, " L"]
    Token validity mask.
  residue_idx : Int[Array, " L"]
    Residue sequence numbers.
  token_idx : Int[Array, " L"]
    Token indices.
  asym_id : Int[Array, " L"]
    Chain IDs.
  entity_id : Int[Array, " L"]
    Entity IDs.
  sym_id : Int[Array, " L"]
    Symmetry IDs.
  aatype : Int[Array, " L"]
    Sequence amino-acid indices in AF3 token order ``0..31``.
  num_steps : int, default=25
    Number of EDM discretization steps.
  sigma_min : float, default=0.002
    Minimum noise level.
  sigma_max : float, default=80.0
    Maximum noise level.
  rho : float, default=7.0
    Karras schedule curvature exponent.
  s_churn : float, default=0.0
    Stochasticity churn parameter.
  s_min : float, default=0.0
    Minimum sigma for churn injection.
  s_max : float, default=inf
    Maximum sigma for churn injection.
  s_noise : float, default=1.0
    Noise inflation factor during churn.
  step_scale : float, default=1.5
    Step-size multiplier on drift ``d_cur``.
  use_heun : bool, default=False
    Whether to apply second-order Heun correction.
  key : jax.Array | None, optional
    PRNG key for initial noise and stochastic churn.
  molecule_type_encoding : Float[Array, "L 5"] | None, optional
    Optional molecule type encoding.
  is_ligand : Bool[Array, " L"] | None, optional
    Optional ligand token mask.

  Returns
  -------
  Float[Array, "L 23 3"]
    Global Atom23 coordinates ``(L, 23, 3)`` with packed sidechains.
  """
  dtype = packer.x_embedder.weight.dtype
  num_l = atom23_xyz.shape[0]
  scn_mask = _extract_scn33_mask(atom23_mask).astype(dtype)

  # Run backbone MPNN trunk once
  bb_only_mask = jnp.concatenate([atom23_mask[:, :5], jnp.zeros((num_l, 18), dtype=atom23_mask.dtype)], axis=1)
  bb_only_xyz = atom23_xyz * bb_only_mask[..., None]
  h_v, _, _, _ = packer.mpnn(
    atom23_xyz=bb_only_xyz,
    atom23_mask=bb_only_mask,
    token_mask=token_mask,
    residue_idx=residue_idx,
    token_idx=token_idx,
    asym_id=asym_id,
    entity_id=entity_id,
    sym_id=sym_id,
    aatype=aatype,
    molecule_type_encoding=molecule_type_encoding,
    is_ligand=is_ligand,
  )

  # Karras sigma schedule t_0 > t_1 > ... > t_{N-1} > t_N = 0 scaled by sigma_data
  step_indices = jnp.arange(num_steps, dtype=dtype)
  inv_rho = 1.0 / rho
  s_max_rho = sigma_max**inv_rho
  s_min_rho = sigma_min**inv_rho
  denom = float(max(num_steps - 1, 1))
  t_steps = ((s_max_rho + step_indices / denom * (s_min_rho - s_max_rho)) ** rho) * packer.sigma_data
  t_steps_padded = jnp.concatenate([t_steps, jnp.zeros((1,), dtype=dtype)], axis=0)

  if key is None:
    key = jax.random.PRNGKey(0)
  k_init, k_loop = jax.random.split(key, 2)
  x_init = (
    jax.random.normal(k_init, (num_l, 33, 3), dtype=dtype)
    * t_steps_padded[0]
    * scn_mask[..., None]
  )
  x_cond_init = jnp.zeros_like(x_init)
  step_keys = jax.random.split(k_loop, num_steps)
  gamma_val = min(s_churn / max(num_steps, 1), math.sqrt(2.0) - 1.0) if s_churn > 0.0 else 0.0

  def _scan_step(
    carry: tuple[Float[Array, "L 33 3"], Float[Array, "L 33 3"]],
    inputs: tuple[Float[Array, ""], Float[Array, ""], jax.Array],
  ) -> tuple[tuple[Float[Array, "L 33 3"], Float[Array, "L 33 3"]], None]:
    x_cur, x_cond_cur = carry
    t_cur, t_next, step_key = inputs
    in_churn_range = (t_cur >= s_min) & (t_cur <= s_max)
    gamma = jnp.where(in_churn_range, gamma_val, 0.0)
    t_hat = t_cur + gamma * t_cur
    noise_std = jnp.sqrt(jnp.maximum(t_hat * t_hat - t_cur * t_cur, 0.0)) * s_noise
    eps = jax.random.normal(step_key, x_cur.shape, dtype=dtype) * scn_mask[..., None]
    x_hat = x_cur + noise_std * eps

    sigma_vec = jnp.broadcast_to(t_hat, (num_l,))
    denoised = packer.denoise_from_trunk(
      x_hat, sigma_vec, h_v, scn_mask, x_cond=x_cond_cur, aatype=aatype,
    )
    d_cur = (x_hat - denoised) / jnp.maximum(t_hat, 1e-8)
    x_next = x_hat + step_scale * (t_next - t_hat) * d_cur
    x_cond_next = denoised

    if use_heun:
      sigma_next_vec = jnp.broadcast_to(t_next, (num_l,))
      denoised_next = packer.denoise_from_trunk(
        x_next, sigma_next_vec, h_v, scn_mask, x_cond=denoised, aatype=aatype,
      )
      d_prime = (x_next - denoised_next) / jnp.maximum(t_next, 1e-8)
      x_heun = x_hat + step_scale * (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
      x_next = jnp.where(t_next > 0.0, x_heun, x_next)
      x_cond_next = jnp.where(t_next > 0.0, denoised_next, denoised)

    return (x_next * scn_mask[..., None], x_cond_next * scn_mask[..., None]), None

  (x_final, _), _ = jax.lax.scan(
    _scan_step,
    (x_init, x_cond_init),
    (t_steps_padded[:-1], t_steps_padded[1:], step_keys),
  )
  return local_to_global_scn(x_final, atom23_xyz, atom23_mask)
