"""Caliby low-rank symmetric GraphPotts decoder head, vocabulary slicing, and ensemble averaging.

Ports ``GraphPotts`` and vocabulary projection utilities from
``ProteinDesignLab/caliby`` (``caliby/model/lr_potts_model/potts.py``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Upstream 21-state Potts slice indices out of 32 AF3 tokens:
# 0..19 are the 20 standard amino acids in AF3 order (A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V)
# 20 is UNK (index 20 in AF3), 31 is GAP ('-', mapped to slot 20 in Caliby 21-state Potts).
POTTS_AF3_TOKEN_IDXS: tuple[int, ...] = (*range(20), 31)
_POTTS_AF3_IDXS_ARR: jax.Array = jnp.array(POTTS_AF3_TOKEN_IDXS, dtype=jnp.int32)

# AF3 20-AA order + GAP: "ARNDCQEGHILKMFPSTWYVX" (where slot 20 is GAP/X)
CALIBY21_ALPHABET: str = "ARNDCQEGHILKMFPSTWYVX"
# Aminx canonical 21-AA order: "ACDEFGHIKLMNPQRSTVWYX"
AMINX21_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"

# Index permutation mapping Aminx 21-token index -> Caliby 21-token index
AMINX21_TO_CALIBY21_IDXS: tuple[int, ...] = tuple(
  CALIBY21_ALPHABET.index(ch) for ch in AMINX21_ALPHABET
)
# Index permutation mapping Caliby 21-token index -> Aminx 21-token index
CALIBY21_TO_AMINX21_IDXS: tuple[int, ...] = tuple(
  AMINX21_ALPHABET.index(ch) for ch in CALIBY21_ALPHABET
)

_AMINX_TO_CALIBY_ARR: jax.Array = jnp.array(AMINX21_TO_CALIBY21_IDXS, dtype=jnp.int32)
_CALIBY_TO_AMINX_ARR: jax.Array = jnp.array(CALIBY21_TO_AMINX21_IDXS, dtype=jnp.int32)


def _linear(layer: eqx.nn.Linear, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
  """Apply ``eqx.nn.Linear`` over the trailing axis of ``x`` without ``vmap``."""
  out = x @ layer.weight.T
  if layer.bias is not None:
    out = out + layer.bias
  return out


def _layer_norm(norm: eqx.nn.LayerNorm, x: Float[Array, "... H"]) -> Float[Array, "... H"]:
  """Apply ``eqx.nn.LayerNorm`` over the trailing axis of ``x`` without ``vmap``."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.var(x, axis=-1, keepdims=True)
  out = (x - mean) * jax.lax.rsqrt(variance + norm.eps)
  if norm.weight is not None:
    out = out * norm.weight
  if norm.bias is not None:
    out = out + norm.bias
  return out


def _symmetrize_J(  # noqa: N802
  j_knn: Float[Array, "L K S S"],
  e_idx: Int[Array, "L K"],
  valid_edge: Float[Array, "L K"],
  num_tokens: int,
  *,
  symmetric: bool = True,
) -> Float[Array, "L L S S"]:
  """Scatter kNN couplings onto ``(L, L, S, S)`` and symmetrize reciprocal edges.

  Matches upstream ``GraphPotts._symmetrize_J`` (``potts.py:256-288``): after
  truncating to ``k_neighbors_potts``, only reciprocal edges ``(i, j)`` where
  ``j in e_idx[i]`` and ``i in e_idx[j]`` are retained, and
  ``J_symm[i, j] = 0.5 * (J[i, j] + J[j, i]^T)``.

  Parameters
  ----------
  j_knn : Float[Array, "L K S S"]
    Per-edge coupling tensor on the truncated kNN graph.
  e_idx : Int[Array, "L K"]
    Truncated kNN neighbor index array of shape ``(L, K)``.
  valid_edge : Float[Array, "L K"]
    Valid non-self edge mask of shape ``(L, K)``.
  num_tokens : int
    Total number of residues/tokens ``L``.
  symmetric : bool, default=True
    Whether to enforce reciprocal-edge transpose symmetry.

  Returns
  -------
  Float[Array, "L L S S"]
    Dense symmetric coupling tensor of shape ``(L, L, S, S)``.
  """
  dtype = j_knn.dtype
  num_k = e_idx.shape[1]
  num_states = j_knn.shape[-1]
  row_idx = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
  rows_2d = jnp.broadcast_to(row_idx, (num_tokens, num_k))

  j_dense = jnp.zeros(
    (num_tokens, num_tokens, num_states, num_states),
    dtype=dtype,
  ).at[rows_2d, e_idx].add(j_knn)

  if not symmetric:
    return j_dense

  adj_dense = jnp.zeros((num_tokens, num_tokens), dtype=dtype).at[rows_2d, e_idx].add(valid_edge)
  reciprocal_mask = (adj_dense > 0).astype(dtype) * (jnp.swapaxes(adj_dense, 0, 1) > 0).astype(dtype)
  return 0.5 * (j_dense + jnp.transpose(j_dense, (1, 0, 3, 2))) * reciprocal_mask[:, :, None, None]


class CalibyGraphPotts(eqx.Module):
  """Symmetric low-rank Potts coupling and site-field head from Caliby.

  Computes site fields ``h`` of shape ``(L, K_vocab)`` and symmetric pairwise
  couplings ``J`` of shape ``(L, L, K_vocab, K_vocab)`` matching upstream
  ``caliby/model/seq_denoiser/denoisers/seq_design/potts.py``.

  Attributes
  ----------
  node_Wait_i : eqx.nn.Linear
    Source-node projection ``d_node -> num_factors * K_vocab``.
  node_Wait_j : eqx.nn.Linear
    Target-node projection ``d_node -> num_factors * K_vocab``.
  norm_J_i : eqx.nn.LayerNorm
    LayerNorm on projected factors ``(num_factors * K_vocab)``.
  norm_J_j : eqx.nn.LayerNorm
    LayerNorm on projected factors ``(num_factors * K_vocab)``.
  edge_Wait : eqx.nn.Linear
    Edge factor-gate projection ``d_edge -> num_factors``.
  node_to_h : eqx.nn.Linear
    Site-field projection ``d_node -> K_vocab``.
  norm_h : eqx.nn.LayerNorm
    LayerNorm on site fields ``(K_vocab,)``.
  log_scale : jax.Array | None
    Learnable log scale applied linearly to ``h`` and ``J``.
  W_h : eqx.nn.Linear | None
    Site-field projection ``d_node -> K_vocab`` for ``parameterization="factor"``.
  W_J_left : eqx.nn.Linear | None
    Left factor projection ``d_edge -> K_vocab * num_factors`` for ``parameterization="factor"``.
  W_J_right : eqx.nn.Linear | None
    Right factor projection ``d_edge -> K_vocab * num_factors`` for ``parameterization="factor"``.
  W_h_bg : eqx.nn.Linear | None
    Optional background site-field projection ``d_node -> bg_dim``.
  W_J_bg : eqx.nn.Linear | None
    Optional background coupling projection ``d_edge -> bg_dim``.
  num_states : int
    Vocabulary size ``K_vocab`` (default ``32`` for AF3 token vocabulary).
  num_factors : int
    Number of low-rank factors ``R``.
  k_neighbors_potts : int
    Maximum neighbors retained before coupling symmetrization (default ``30``).
  parameterization : str
    Parameterization mode (``"factor"`` or ``"factorized"``).
  symmetric_J : bool
    Whether to symmetrize couplings across reciprocal edges (``True``).
  bg_dim : int
    Output dimension of optional background projections (default ``1``).
  has_bg : bool
    Whether ``W_h_bg`` / ``W_J_bg`` are allocated. Set in ``__init__`` from the
    allocated projections, so it cannot drift from the weights.
  """

  node_Wait_i: eqx.nn.Linear  # noqa: N815
  node_Wait_j: eqx.nn.Linear  # noqa: N815
  norm_J_i: eqx.nn.LayerNorm  # noqa: N815
  norm_J_j: eqx.nn.LayerNorm  # noqa: N815
  edge_Wait: eqx.nn.Linear  # noqa: N815
  node_to_h: eqx.nn.Linear
  norm_h: eqx.nn.LayerNorm
  log_scale: jax.Array | None = None
  W_h: eqx.nn.Linear | None = None
  W_J_left: eqx.nn.Linear | None = None
  W_J_right: eqx.nn.Linear | None = None
  W_h_bg: eqx.nn.Linear | None = None
  W_J_bg: eqx.nn.Linear | None = None
  num_states: int = eqx.field(static=True, default=32)
  num_factors: int = eqx.field(static=True, default=64)
  k_neighbors_potts: int = eqx.field(static=True, default=30)
  parameterization: str = eqx.field(static=True, default="factorized")
  symmetric_J: bool = eqx.field(static=True, default=True)  # noqa: N815
  bg_dim: int = eqx.field(static=True, default=1)
  has_bg: bool = eqx.field(static=True, default=False)

  def __init__(
    self,
    d_node: int = 128,
    d_edge: int = 128,
    num_states: int = 32,
    num_factors: int = 64,
    *,
    k_neighbors_potts: int = 30,
    parameterization: str = "factorized",
    symmetric_J: bool = True,  # noqa: N803
    has_bg: bool = False,
    bg_dim: int = 1,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyGraphPotts``.

    Parameters
    ----------
    d_node : int, default=128
      Node hidden dimension.
    d_edge : int, default=128
      Edge hidden dimension.
    num_states : int, default=32
      Full token vocabulary size (``32`` in upstream Caliby).
    num_factors : int, default=32
      Low-rank factor count ``R``.
    k_neighbors_potts : int, default=30
      Maximum kNN edges retained for Potts couplings before ``_symmetrize_J``.
    parameterization : str, default="factorized"
      Coupling parameterization name (``"factorized"`` or ``"factor"``).
    symmetric_J : bool, default=True
      Whether to enforce symmetry on factors and couplings.
    has_bg : bool, default=False
      Whether to allocate optional ``W_h_bg`` and ``W_J_bg`` projections.
    bg_dim : int, default=1
      Output dimension for ``W_h_bg`` and ``W_J_bg`` (``1`` or ``num_states``).
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 9)
    proj_dim = num_factors * num_states
    self.node_Wait_i = eqx.nn.Linear(d_node, proj_dim, key=keys[0])
    self.node_Wait_j = eqx.nn.Linear(d_node, proj_dim, key=keys[1])
    self.norm_J_i = eqx.nn.LayerNorm(proj_dim)
    self.norm_J_j = eqx.nn.LayerNorm(proj_dim)
    self.edge_Wait = eqx.nn.Linear(d_edge, num_factors, key=keys[2])
    self.node_to_h = eqx.nn.Linear(d_node, num_states, key=keys[3])
    self.norm_h = eqx.nn.LayerNorm(num_states)
    if parameterization == "factor":
      self.log_scale = jnp.array([jnp.log(0.1)], dtype=jnp.float32)
      self.W_h = eqx.nn.Linear(d_node, num_states, key=keys[4])
      self.W_J_left = eqx.nn.Linear(d_edge, num_states * num_states, key=keys[5])
      self.W_J_right = eqx.nn.Linear(d_edge, num_states * num_states, key=keys[6])
      if has_bg:
        self.W_h_bg = eqx.nn.Linear(d_node, bg_dim, key=keys[7])
        self.W_J_bg = eqx.nn.Linear(d_edge, bg_dim, key=keys[8])
      else:
        self.W_h_bg = None
        self.W_J_bg = None
    else:
      self.log_scale = None
      self.W_h = None
      self.W_J_left = None
      self.W_J_right = None
      self.W_h_bg = None
      self.W_J_bg = None
    self.num_states = num_states
    self.num_factors = num_factors
    self.k_neighbors_potts = k_neighbors_potts
    self.parameterization = parameterization
    self.symmetric_J = symmetric_J
    self.bg_dim = bg_dim
    self.has_bg = self.W_h_bg is not None or self.W_J_bg is not None

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K E"],
    e_idx: Int[Array, "L K"],
    mask_i: Float[Array, " L"],
    mask_ij: Float[Array, "L K"],
  ) -> tuple[Float[Array, "L S"], Float[Array, "L L S S"]]:
    """Compute masked site fields ``h`` and symmetric dense couplings ``J``.

    Parameters
    ----------
    h_v : Float[Array, "L H"]
      Decoder node representations of shape ``(L, d_node)``.
    h_e : Float[Array, "L K E"]
      Encoder/decoder edge representations of shape ``(L, K, d_edge)``.
    e_idx : Int[Array, "L K"]
      kNN neighbor index array of shape ``(L, K)``.
    mask_i : Float[Array, " L"]
      Valid token mask of shape ``(L,)``.
    mask_ij : Float[Array, "L K"]
      Valid kNN edge mask of shape ``(L, K)`` (with self-edges zeroed if excluded).

    Returns
    -------
    tuple[Float[Array, "L S"], Float[Array, "L L S S"]]
      Tuple ``(h, J)`` where ``h`` has shape ``(L, num_states)`` and dense
      symmetric ``J`` has shape ``(L, L, num_states, num_states)``.
    """
    dtype = h_v.dtype
    num_tokens = h_v.shape[0]
    k_eff = min(self.k_neighbors_potts, e_idx.shape[1])
    h_e_trunc = h_e[:, :k_eff, :]
    e_idx_trunc = e_idx[:, :k_eff]
    mask_ij_trunc = mask_ij[:, :k_eff]

    row_idx = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    non_self = (e_idx_trunc != row_idx).astype(dtype)
    valid_edge = mask_ij_trunc.astype(dtype) * non_self

    if (
      self.parameterization == "factor"
      and self.log_scale is not None
      and self.W_h is not None
      and self.W_J_left is not None
      and self.W_J_right is not None
    ):
      scale = jnp.exp(self.log_scale.astype(dtype))
      mask_v = mask_i.astype(dtype)[:, None]
      h = scale * mask_v * _linear(self.W_h, h_v)

      num_factors_eff = self.W_J_left.weight.shape[0] // self.num_states
      j_left = (valid_edge[..., None] * _linear(self.W_J_left, h_e_trunc)).reshape(
        num_tokens,
        k_eff,
        self.num_states,
        num_factors_eff,
      )
      j_right = (valid_edge[..., None] * _linear(self.W_J_right, h_e_trunc)).reshape(
        num_tokens,
        k_eff,
        self.num_states,
        num_factors_eff,
      )
      j_knn = scale * jnp.matmul(j_left, jnp.swapaxes(j_right, -1, -2))
      # Zero-sum gauge
      h = (h - jnp.mean(h, axis=-1, keepdims=True)) * mask_v
      j_knn = (
        j_knn
        - jnp.mean(j_knn, axis=-1, keepdims=True)
        - jnp.mean(j_knn, axis=-2, keepdims=True)
        + jnp.mean(j_knn, axis=(-1, -2), keepdims=True)
      ) * valid_edge[..., None, None]
      if self.W_h_bg is not None:
        h = h + scale * mask_v * _linear(self.W_h_bg, h_v)
      if self.W_J_bg is not None:
        j_knn = j_knn + (scale * valid_edge[..., None] * _linear(self.W_J_bg, h_e_trunc))[..., None]
    else:
      # Site fields h
      h_raw = _layer_norm(self.norm_h, _linear(self.node_to_h, h_v))
      h = h_raw * mask_i.astype(dtype)[:, None]

      # Factorized couplings W_i, W_j
      w_i_flat = _layer_norm(self.norm_J_i, _linear(self.node_Wait_i, h_v))
      w_j_flat = _layer_norm(self.norm_J_j, _linear(self.node_Wait_j, h_v))
      w_i = w_i_flat.reshape(num_tokens, self.num_factors, self.num_states)
      w_j = w_j_flat.reshape(num_tokens, self.num_factors, self.num_states)

      if self.symmetric_J:
        w_sym = 0.5 * (w_i + w_j)
        w_i = w_sym
        w_j = w_sym

      w_j_neighbors = w_j[e_idx_trunc]
      s_edge = jax.nn.sigmoid(_linear(self.edge_Wait, h_e_trunc))
      gated_w_j = w_j_neighbors * (s_edge * valid_edge[..., None])[..., None]
      j_knn = jnp.einsum("ira,ikrb->ikab", w_i, gated_w_j)

    j_dense = _symmetrize_J(
      j_knn,
      e_idx_trunc,
      valid_edge,
      num_tokens,
      symmetric=self.symmetric_J,
    )
    return h, j_dense


def slice_potts_params(
  h: Float[Array, "... L 32"],
  j: Float[Array, "... L L 32 32"],
  *,
  to_aminx_alphabet: bool = False,
) -> tuple[Float[Array, "... L 21"], Float[Array, "... L L 21 21"]]:
  """Slice 32-token AF3 Potts parameters down to 21 amino-acid + gap states.

  Parameters
  ----------
  h : Float[Array, "... L 32"]
    Full 32-state site fields.
  j : Float[Array, "... L L 32 32"]
    Full 32-state pairwise couplings.
  to_aminx_alphabet : bool, default=False
    When ``True``, also permutes the 21 states from Caliby's AF3 order
    (``ARNDCQEGHILKMFPSTWYVX``) to Aminx's canonical alphabetical order
    (``ACDEFGHIKLMNPQRSTVWYX``).

  Returns
  -------
  tuple[Float[Array, "... L 21"], Float[Array, "... L L 21 21"]]
    Sliced ``(h_21, J_21)`` tensors.
  """
  h_21 = jnp.take(h, _POTTS_AF3_IDXS_ARR, axis=-1)
  j_21 = jnp.take(jnp.take(j, _POTTS_AF3_IDXS_ARR, axis=-2), _POTTS_AF3_IDXS_ARR, axis=-1)
  if to_aminx_alphabet:
    h_21 = jnp.take(h_21, _AMINX_TO_CALIBY_ARR, axis=-1)
    j_21 = jnp.take(jnp.take(j_21, _AMINX_TO_CALIBY_ARR, axis=-2), _AMINX_TO_CALIBY_ARR, axis=-1)
  return h_21, j_21


def generate_gaussian_ensemble(
  atom23_xyz: Float[Array, "L 23 3"],
  atom23_mask: Float[Array, "L 23"],
  *,
  num_conformers: int = 30,
  noise_std: float = 0.25,
  key: jax.Array | None = None,
) -> Float[Array, "C L 23 3"]:
  """Generate a synthetic conformational ensemble around a static input structure.

  In upstream Caliby, inference defaults to ``caliby_distill`` (single-structure) or
  accepts an ensemble of conformers (e.g., from Protpardelle-1c or Gaussian coordinate
  perturbation). The first conformer (index 0) is always the unperturbed structure.

  Parameters
  ----------
  atom23_xyz : Float[Array, "L 23 3"]
    Reference atom23 coordinates of shape ``(L, 23, 3)``.
  atom23_mask : Float[Array, "L 23"]
    Atom validity mask of shape ``(L, 23)``.
  num_conformers : int, default=30
    Number of ensemble conformers ``C`` to generate (``C >= 1``).
  noise_std : float, default=0.25
    Standard deviation of Gaussian coordinate noise (in Angstroms) for conformers ``1..C-1``.
  key : jax.Array
    PRNG key for noise sampling.

  Returns
  -------
  Float[Array, "C L 23 3"]
    Conformational ensemble coordinates of shape ``(C, L, 23, 3)``.
  """
  if num_conformers <= 1 or noise_std <= 0.0:
    return jnp.broadcast_to(atom23_xyz[None, ...], (max(1, num_conformers), *atom23_xyz.shape))
  if key is None:
    key = jax.random.PRNGKey(0)
  noise = jax.random.normal(key, (num_conformers - 1, *atom23_xyz.shape), dtype=atom23_xyz.dtype)
  perturbed = atom23_xyz[None, ...] + noise * noise_std * atom23_mask[None, ..., None]
  return jnp.concatenate([atom23_xyz[None, ...], perturbed], axis=0)


def aggregate_ensemble_potts_params(
  h_stack: Float[Array, "C L S"],
  j_stack: Float[Array, "C L K S S"],
  *,
  e_idx: Int[Array, "C L K"] | Int[Array, "L K"] | None = None,
  valid_edge: Float[Array, "C L K"] | Float[Array, "L K"] | None = None,
  symmetric: bool = True,
) -> tuple[Float[Array, "L S"], Float[Array, "L L S S"]]:
  """Aggregate per-conformer Potts parameters across a conformational ensemble by mean.

  Supports both already-densified ``(C, L, L, S, S)`` coupling stacks and sparse
  per-conformer kNN coupling stacks ``(C, L, K, S, S)`` with distinct neighbor
  tables ``e_idx`` of shape ``(C, L, K)`` (scattering each conformer onto the
  common ``(L, L, S, S)`` residue grid before averaging).

  Parameters
  ----------
  h_stack : Float[Array, "C L S"]
    Per-conformer site fields of shape ``(C, L, S)``.
  j_stack : Float[Array, "C L K S S"]
    Per-conformer pairwise couplings of shape ``(C, L, L, S, S)`` or sparse
    ``(C, L, K, S, S)``.
  e_idx : Int[Array, "C L K"] | Int[Array, "L K"] | None, optional
    Optional per-conformer or shared kNN neighbor index array when ``j_stack``
    is in sparse kNN layout ``(C, L, K, S, S)``.
  valid_edge : Float[Array, "C L K"] | Float[Array, "L K"] | None, optional
    Optional edge validity mask when ``e_idx`` is provided.
  symmetric : bool, default=True
    Whether to symmetrize each conformer when scattering from ``e_idx``.

  Returns
  -------
  tuple[Float[Array, "L S"], Float[Array, "L L S S"]]
    Ensemble-averaged ``(h_mean, J_mean)``.
  """
  h_mean = jnp.mean(h_stack, axis=0)
  if e_idx is None:
    return h_mean, jnp.mean(j_stack, axis=0)

  num_c, num_l, num_k, num_s, _ = j_stack.shape
  e_idx_3d = (
    jnp.broadcast_to(e_idx[None, ...], (num_c, num_l, num_k))
    if e_idx.ndim == 2
    else e_idx
  )
  c_idx = jnp.arange(num_c, dtype=jnp.int32)[:, None, None]
  row_idx = jnp.arange(num_l, dtype=jnp.int32)[None, :, None]
  if valid_edge is None:
    valid_3d = (e_idx_3d != row_idx).astype(j_stack.dtype)
  else:
    valid_3d = (
      jnp.broadcast_to(valid_edge[None, ...], (num_c, num_l, num_k)).astype(j_stack.dtype)
      if valid_edge.ndim == 2
      else valid_edge.astype(j_stack.dtype)
    )

  j_masked = j_stack * valid_3d[..., None, None]
  j_dense_stack = (
    jnp.zeros((num_c, num_l, num_l, num_s, num_s), dtype=j_stack.dtype)
    .at[c_idx, row_idx, e_idx_3d]
    .add(j_masked)
  )
  diag = jnp.arange(num_l, dtype=jnp.int32)
  j_dense_stack = j_dense_stack.at[:, diag, diag, :, :].set(0.0)
  if symmetric:
    j_dense_stack = 0.5 * (j_dense_stack + jnp.transpose(j_dense_stack, (0, 2, 1, 4, 3)))
  return h_mean, jnp.mean(j_dense_stack, axis=0)


def compute_caliby_potts_energy(
  h_21: Float[Array, "L 21"],
  j_21: Float[Array, "L L 21 21"],
  sequence: Int[Array, "... L"],
  mask: Float[Array, " L"],
) -> Float[Array, "..."]:
  """Evaluate the Caliby Potts Hamiltonian ``E(S) = sum_i h_i(s_i) + 0.5 * sum_{i != j} J_{ij}(s_i, s_j)``.

  Parameters
  ----------
  h_21 : Float[Array, "L 21"]
    21-state site fields ``(L, 21)``.
  j_21 : Float[Array, "L L 21 21"]
    21-state symmetric pairwise couplings ``(L, L, 21, 21)``.
  sequence : Int[Array, "... L"]
    Integer sequence indices in ``0..20`` of shape ``(L,)`` or ``(B, L)``.
  mask : Float[Array, " L"]
    Valid residue mask of shape ``(L,)``.

  Returns
  -------
  Float[Array, "..."]
    Total Potts energy per sequence.
  """
  dtype = h_21.dtype
  valid = mask.astype(dtype)
  one_hot_s = jax.nn.one_hot(sequence, 21, dtype=dtype) * valid[..., None]
  site_term = jnp.einsum("li,...li->...", h_21, one_hot_s)
  # Zero diagonal of J_21 to ensure i != j
  num_l = h_21.shape[0]
  off_diag = (1.0 - jnp.eye(num_l, dtype=dtype))[:, :, None, None]
  j_off = j_21 * off_diag
  pair_term = 0.5 * jnp.einsum("...ia,ijab,...jb->...", one_hot_s, j_off, one_hot_s)
  return site_term + pair_term
