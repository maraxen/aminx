"""LASEr protein and ligand encoder stacks.

Composes the B2 layers the way ``LASErMPNN.apply_encoding_layers`` does:
ligand HomoGAT, optional output DenseGVP, backbone-frame GVP, then the
protein HeteroGAT stack. Graphs are built on the host and padded to a fixed
ligand-atom bucket before the jitted body.

Math:
    Ligand atoms are one-hot period and group. Input vectors point from each
    atom to the mean of its ligand-ligand neighbours, self included.
    Protein scalars start at 0. Protein vectors are the normalised backbone
    frame (bisector, CA-CB, N-CA, CA-C) passed through a vector GVP and a
    vector-only layer norm. Edge features are RBFs of inter-atomic distances
    concatenated with frame-vector dot products.

Pseudocode:
    lig = output_gvp(ligand_gat(features, lig_lig_graph))
    frames = norm(gvp(backbone_frames))
    edges = edge_mlp(rbf(distances), dots(frames))
    for layer in protein_layers:
        prot, edges = layer(prot, lig, edges, graphs)
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xtrax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpy.typing import NDArray
from xtrax.tiling import AxisSpec, select_bucket

from aminx.model.laser.graphs import (
  GraphStructure,
  PackedEdges,
  build_laser_graphs,
  ligand_ligand_distances,
  ligand_protein_distances,
  pack_edges,
  protein_edge_distances,
  scatter_edge_values,
)
from aminx.model.laser.layers import (
  GVP,
  DenseGVP,
  EquivariantLayerNorm,
  HeteroGATv2,
  HomoGATv2,
  apply_linear,
  norm_no_nan,
)

if xtrax.__version__ != "0.4.0a10":
  msg = f"Expected xtrax 0.4.0a10, got {xtrax.__version__}"
  raise RuntimeError(msg)

# Smallest boundary >= the atom count. select_bucket rejects a longer ligand.
LIGAND_ATOM_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

LIGAND_ATOMS = AxisSpec(
  name="ligand_atoms",
  cardinality=LIGAND_ATOM_BUCKETS[-1],
  default_batch_size=1,
  bucket_boundaries=LIGAND_ATOM_BUCKETS,
)

_NODE = 256
_EDGE = 128
_V_PROT = 10
_V_LIG = 15
_HEADS_PROT = 1
_HEADS_LIG = 3
_UPSCALE = 4
_DROPOUT = 0.1
_N_PROT_LAYERS = 3
_N_LIG_LAYERS = 3
_LIG_IN = 26
_PERIOD_CLASSES = 7
_GROUP_CLASSES = 19
_PROT_RBF = (2.0, 22.0, 16)
_LIG_RBF = (0.0, 15.0, 75)


def ligand_atom_bucket(n_atoms: int) -> int:
  """Pad width for a ligand of ``n_atoms`` atoms."""
  return select_bucket(n_atoms, LIGAND_ATOM_BUCKETS)


def ligand_atom_features(
  atomic_numbers: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
) -> Float[NDArray[np.floating], "a 26"]:
  """Period and group one-hot, the ``LigandFeaturizer`` encoding of elemental Z."""
  numbers = np.asarray(atomic_numbers, dtype=np.int64)
  n_atoms = int(numbers.shape[0])
  features = np.zeros((n_atoms, _PERIOD_CLASSES + _GROUP_CLASSES), dtype=np.float64)
  if n_atoms == 0:
    return features
  period = np.asarray(period_index, dtype=np.int64)[numbers - 1]
  group = np.asarray(group_index, dtype=np.int64)[numbers - 1]
  rows = np.arange(n_atoms)
  features[rows, period] = 1.0
  features[rows, _PERIOD_CLASSES + group] = 1.0
  return features


class RBFEncoding(eqx.Module):
  """ProteinMPNN radial basis. ``D_mu`` is the checkpoint buffer."""

  D_mu: Float[Array, "1 bins"]
  num_bins: int = eqx.field(static=True)
  bin_min: float = eqx.field(static=True)
  bin_max: float = eqx.field(static=True)

  def __init__(self, bin_min: float, bin_max: float, num_bins: int) -> None:
    self.num_bins = num_bins
    self.bin_min = bin_min
    self.bin_max = bin_max
    self.D_mu = jnp.linspace(bin_min, bin_max, num_bins).reshape(1, num_bins)

  def __call__(self, distances: Float[Array, "..."]) -> Float[Array, "... bins"]:
    """``(exp(-((d - mu) / sigma)^2) + 1e-4)`` with ``sigma = (max - min) / bins``."""
    sigma = (self.bin_max - self.bin_min) / self.num_bins
    expanded = distances[..., None]
    return jnp.exp(-(((expanded - self.D_mu) / sigma) ** 2)) + 1e-4


class _LigandStack(eqx.Module):
  """``LigandEncoderModule`` plus the checkpoint's output DenseGVP."""

  ligand_input_layer: eqx.nn.Linear
  input_gvp: GVP
  ligand_edge_rbf_encoding: RBFEncoding
  ligand_edge_input_layer: eqx.nn.Linear
  gat_layers: tuple[HomoGATv2, HomoGATv2, HomoGATv2]

  def __init__(self, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, 6)
    self.ligand_input_layer = eqx.nn.Linear(_LIG_IN, _NODE, key=keys[0])
    self.input_gvp = GVP((_NODE, 1), (_NODE, _V_LIG), vector_gate=True, key=keys[1])
    self.ligand_edge_rbf_encoding = RBFEncoding(*_LIG_RBF)
    self.ligand_edge_input_layer = eqx.nn.Linear(_LIG_RBF[2], _EDGE, key=keys[2])
    built = [
      HomoGATv2(
        _NODE,
        _EDGE,
        _HEADS_LIG,
        _DROPOUT,
        update_edges=index < _N_LIG_LAYERS - 1,
        use_mlp_node_update=False,
        atten_head_aggr_layers=0,
        num_vectors=_V_LIG,
        atten_dimension_upscale_factor=None,
        key=keys[3 + index],
      )
      for index in range(_N_LIG_LAYERS)
    ]
    self.gat_layers = (built[0], built[1], built[2])

  def __call__(
    self,
    features: Float[Array, "a 26"],
    coords: Float[Array, "a 3"],
    neighbours: Int[Array, "a k"],
    mask: Bool[Array, "a k"],
    distance: Float[Array, "a k"],
  ) -> tuple[Float[Array, "a h"], Float[Array, "a vl 3"]]:
    """Ligand scalars and vectors before the output DenseGVP. Eval dropout is the identity."""
    scalars = apply_linear(self.ligand_input_layer, features)
    gathered = coords[neighbours]
    gathered = jnp.where(mask[..., None], gathered, jnp.zeros_like(gathered))
    count = jnp.sum(mask, axis=1)
    denom = jnp.where(count > 0, count, jnp.ones_like(count)).astype(coords.dtype)
    mean = jnp.sum(gathered, axis=1) / denom[:, None]
    direction = mean - coords
    direction = direction / norm_no_nan(direction, axis=-1, keepdims=True)
    vectors = direction[:, None, :]
    scalars, vectors = self.input_gvp(scalars, vectors)
    edges = apply_linear(
      self.ligand_edge_input_layer,
      self.ligand_edge_rbf_encoding(distance),
    )
    for gat in self.gat_layers:
      scalars, vectors, edges = gat(scalars, vectors, edges, neighbours, mask)
    return scalars, vectors


class _ProteinLayer(eqx.Module):
  """One ``LASErMPNN_Encoder``: HeteroGAT, then the two frame-dot linears."""

  hetgat: HeteroGATv2
  final_lig_pr_mlp: eqx.nn.Linear
  final_pr_pr_mlp: eqx.nn.Linear

  def __init__(self, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, 3)
    self.hetgat = HeteroGATv2(
      2,
      _NODE,
      _EDGE,
      0,
      _HEADS_PROT,
      _DROPOUT,
      num_vectors=_V_PROT,
      use_mlp_node_update=True,
      use_residual_node_update=True,
      compute_edge_updates=True,
      atten_dimension_upscale_factor=_UPSCALE,
      key=keys[0],
    )
    self.final_lig_pr_mlp = eqx.nn.Linear(_EDGE + _V_PROT, _EDGE, key=keys[1])
    self.final_pr_pr_mlp = eqx.nn.Linear(_EDGE + _V_PROT * _V_PROT, _EDGE, key=keys[2])

  def __call__(
    self,
    prot_s: Float[Array, "l h"],
    prot_v: Float[Array, "l v 3"],
    lig_s: Float[Array, "a h"],
    pr_neighbours: Int[Array, "l kp"],
    pr_mask: Bool[Array, "l kp"],
    pr_edges: Float[Array, "l kp e"],
    lp_neighbours: Int[Array, "l kl"],
    lp_mask: Bool[Array, "l kl"],
    lp_edges: Float[Array, "l kl e"],
    backbone: Float[Array, "l 5 3"],
    lig_coords: Float[Array, "a 3"],
  ) -> tuple[
    Float[Array, "l h"],
    Float[Array, "l v 3"],
    Float[Array, "l kp e"],
    Float[Array, "l kl e"],
  ]:
    """One protein encoder step. Ligand nodes are a source, not a sink."""
    prot_s, prot_v, edges = self.hetgat(
      prot_s,
      prot_v,
      (prot_s, lig_s),
      (pr_neighbours, lp_neighbours),
      (pr_edges, lp_edges),
      (pr_mask, lp_mask),
      (True, False),
    )
    pr_edges, lp_edges = edges
    lp_edges = apply_linear(
      self.final_lig_pr_mlp,
      jnp.concatenate(
        (lp_edges, _ligand_dots(prot_v, lp_neighbours, backbone, lig_coords)),
        axis=-1,
      ),
    )
    pr_edges = apply_linear(
      self.final_pr_pr_mlp,
      jnp.concatenate((pr_edges, _protein_dots(prot_v, pr_neighbours)), axis=-1),
    )
    return prot_s, prot_v, pr_edges, lp_edges


class LaserEncoder(eqx.Module):
  """Full ``apply_encoding_layers`` stack. Field names match the checkpoint."""

  ligand_encoder: _LigandStack
  ligand_encoder_output_gvp: DenseGVP
  backbone_frame_vec_input_layer: GVP
  backbone_frame_vec_norm: EquivariantLayerNorm
  protein_encoder_layers: tuple[_ProteinLayer, _ProteinLayer, _ProteinLayer]
  prot_prot_rbf_encoding: RBFEncoding
  lig_prot_rbf_encoding: RBFEncoding
  prot_prot_edge_input_layer: eqx.nn.Linear
  lig_prot_edge_input_layer: eqx.nn.Linear

  def __init__(self, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, 8)
    self.ligand_encoder = _LigandStack(key=keys[0])
    # Checkpoint key ``ligand_encoder_output_gvp.*``, applied once after the GAT stack.
    self.ligand_encoder_output_gvp = DenseGVP(
      (_NODE, _V_LIG),
      (_NODE, _V_LIG),
      (_NODE, 0),
      _DROPOUT,
      intermediate_norm=True,
      key=keys[5],
    )
    self.backbone_frame_vec_input_layer = GVP((0, 4), (0, _V_PROT), vector_gate=False, key=keys[1])
    self.backbone_frame_vec_norm = EquivariantLayerNorm((0, _V_PROT), vector_only=True)
    layers = tuple(_ProteinLayer(key=sub) for sub in jax.random.split(keys[2], _N_PROT_LAYERS))
    self.protein_encoder_layers = (layers[0], layers[1], layers[2])
    self.prot_prot_rbf_encoding = RBFEncoding(*_PROT_RBF)
    self.lig_prot_rbf_encoding = RBFEncoding(*_LIG_RBF)
    prot_in = _PROT_RBF[2] * 25 + _V_PROT * _V_PROT
    lig_in = _LIG_RBF[2] * 5 + _V_PROT
    self.prot_prot_edge_input_layer = eqx.nn.Linear(prot_in, _EDGE, key=keys[3])
    self.lig_prot_edge_input_layer = eqx.nn.Linear(lig_in, _EDGE, key=keys[4])

  def __call__(
    self,
    backbone: Float[Array, "l 5 3"],
    lig_coords: Float[Array, "a 3"],
    lig_features: Float[Array, "a 26"],
    pr_neighbours: Int[Array, "l kp"],
    pr_mask: Bool[Array, "l kp"],
    lp_neighbours: Int[Array, "l kl"],
    lp_mask: Bool[Array, "l kl"],
    ll_neighbours: Int[Array, "a kll"],
    ll_mask: Bool[Array, "a kll"],
    ll_distance: Float[Array, "a kll"],
    pr_distance: Float[Array, "l kp 25"],
    lp_distance: Float[Array, "l kl 5"],
    pr_sink: Int[Array, " epp"],
    pr_slots: Int[Array, " epp"],
    lp_sink: Int[Array, " elp"],
    lp_slots: Int[Array, " elp"],
  ) -> tuple[
    Float[Array, "a h"],
    Float[Array, "a vl 3"],
    Float[Array, "l h"],
    Float[Array, "l v 3"],
    Float[Array, "epp e"],
    Float[Array, "elp e"],
  ]:
    """Four ``apply_encoding_layers`` returns, edges gathered back to the edge list.

    Ligand scalars and vectors are the bucket-padded axis. The host slices the
    real atom prefix. Protein edges are the unpadded edge list.
    """
    lig_s, lig_v = self.ligand_encoder(
      lig_features,
      lig_coords,
      ll_neighbours,
      ll_mask,
      ll_distance,
    )
    lig_s, lig_v = self.ligand_encoder_output_gvp(lig_s, lig_v)
    n_res = backbone.shape[0]
    zeros = jnp.zeros((n_res, _NODE), dtype=backbone.dtype)
    frames = _backbone_frames(backbone)
    _scalars, prot_v = self.backbone_frame_vec_norm(
      *self.backbone_frame_vec_input_layer(zeros, frames),
    )
    prot_s = zeros
    pr_edges = _initial_protein_edges(self, prot_v, pr_neighbours, pr_distance)
    lp_edges = _initial_ligand_edges(self, prot_v, lp_neighbours, backbone, lig_coords, lp_distance)
    for layer in self.protein_encoder_layers:
      prot_s, prot_v, pr_edges, lp_edges = layer(
        prot_s,
        prot_v,
        lig_s,
        pr_neighbours,
        pr_mask,
        pr_edges,
        lp_neighbours,
        lp_mask,
        lp_edges,
        backbone,
        lig_coords,
      )
    return (
      lig_s,
      lig_v,
      prot_s,
      prot_v,
      pr_edges[pr_sink, pr_slots],
      lp_edges[lp_sink, lp_slots],
    )


@dataclass(frozen=True, slots=True)
class EncoderForward:
  """Unpadded ``apply_encoding_layers`` outputs plus the host edge lists."""

  lig_scalars: NDArray[np.floating]
  lig_vectors: NDArray[np.floating]
  prot_scalars: NDArray[np.floating]
  prot_vectors: NDArray[np.floating]
  pr_pr_eattr: NDArray[np.floating]
  lig_pr_eattr: NDArray[np.floating]
  pr_pr_idx: NDArray[np.int64]
  lig_pr_idx: NDArray[np.int64]


def encode_structure(
  model: LaserEncoder,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  *,
  protein_loop: bool = True,
) -> EncoderForward:
  """Host graphs, ligand-atom bucket, then one jitted encoder call."""
  dtype = model.prot_prot_edge_input_layer.weight.dtype
  numpy_dtype = np.float64 if dtype == jnp.float64 else np.float32
  backbone_np = np.asarray(backbone, dtype=numpy_dtype)
  ligand_np = np.asarray(ligand_coords, dtype=numpy_dtype)
  graphs = build_laser_graphs(
    backbone_np,
    ligand_np,
    ligand_subbatch,
    structure,
    protein_loop=protein_loop,
  )
  features = np.asarray(
    ligand_atom_features(ligand_atomic_numbers, period_index, group_index),
    dtype=numpy_dtype,
  )
  n_res = int(backbone_np.shape[0])
  n_atoms = int(ligand_np.shape[0])
  packed_pr = pack_edges(n_res, graphs.pr_pr)
  packed_lp = pack_edges(n_res, graphs.lig_pr)
  packed_ll = pack_edges(n_atoms, graphs.lig_lig)
  pr_distance = scatter_edge_values(
    n_res,
    packed_pr,
    protein_edge_distances(backbone_np, graphs.pr_pr),
  )
  lp_distance = scatter_edge_values(
    n_res,
    packed_lp,
    ligand_protein_distances(backbone_np, ligand_np, graphs.lig_pr),
  )
  ll_distance = scatter_edge_values(
    n_atoms,
    packed_ll,
    ligand_ligand_distances(ligand_np, graphs.lig_lig),
  )
  bucket = ligand_atom_bucket(n_atoms)
  lig_coords_j, lig_feat_j, ll_n, ll_m, ll_d = _pad_ligand(
    ligand_np,
    features,
    packed_ll,
    ll_distance,
    bucket,
  )
  outputs = _encode_jit(
    model,
    jnp.asarray(backbone_np),
    jnp.asarray(lig_coords_j),
    jnp.asarray(lig_feat_j),
    jnp.asarray(packed_pr.neighbours),
    jnp.asarray(packed_pr.mask),
    jnp.asarray(packed_lp.neighbours),
    jnp.asarray(packed_lp.mask),
    jnp.asarray(ll_n),
    jnp.asarray(ll_m),
    jnp.asarray(ll_d),
    jnp.asarray(pr_distance),
    jnp.asarray(lp_distance),
    jnp.asarray(packed_pr.sink),
    jnp.asarray(packed_pr.slots),
    jnp.asarray(packed_lp.sink),
    jnp.asarray(packed_lp.slots),
  )
  lig_s, lig_v, prot_s, prot_v, pr_e, lp_e = outputs
  return EncoderForward(
    lig_scalars=np.asarray(lig_s[:n_atoms]),
    lig_vectors=np.asarray(lig_v[:n_atoms]),
    prot_scalars=np.asarray(prot_s),
    prot_vectors=np.asarray(prot_v),
    pr_pr_eattr=np.asarray(pr_e),
    lig_pr_eattr=np.asarray(lp_e),
    pr_pr_idx=graphs.pr_pr,
    lig_pr_idx=graphs.lig_pr,
  )


@eqx.filter_jit
def _encode_jit(
  model: LaserEncoder,
  backbone: Float[Array, "l 5 3"],
  lig_coords: Float[Array, "a 3"],
  lig_features: Float[Array, "a 26"],
  pr_neighbours: Int[Array, "l kp"],
  pr_mask: Bool[Array, "l kp"],
  lp_neighbours: Int[Array, "l kl"],
  lp_mask: Bool[Array, "l kl"],
  ll_neighbours: Int[Array, "a kll"],
  ll_mask: Bool[Array, "a kll"],
  ll_distance: Float[Array, "a kll"],
  pr_distance: Float[Array, "l kp 25"],
  lp_distance: Float[Array, "l kl 5"],
  pr_sink: Int[Array, " epp"],
  pr_slots: Int[Array, " epp"],
  lp_sink: Int[Array, " elp"],
  lp_slots: Int[Array, " elp"],
) -> tuple[
  Float[Array, "a h"],
  Float[Array, "a vl 3"],
  Float[Array, "l h"],
  Float[Array, "l v 3"],
  Float[Array, "epp e"],
  Float[Array, "elp e"],
]:
  return model(
    backbone,
    lig_coords,
    lig_features,
    pr_neighbours,
    pr_mask,
    lp_neighbours,
    lp_mask,
    ll_neighbours,
    ll_mask,
    ll_distance,
    pr_distance,
    lp_distance,
    pr_sink,
    pr_slots,
    lp_sink,
    lp_slots,
  )


def _pad_ligand(
  coords: NDArray[np.floating],
  features: NDArray[np.floating],
  packed: PackedEdges,
  distance: NDArray[np.floating],
  bucket: int,
) -> tuple[
  NDArray[np.floating],
  NDArray[np.floating],
  NDArray[np.int64],
  NDArray[np.bool_],
  NDArray[np.floating],
]:
  """Pad the ligand node axis. Real rows keep their neighbours; pad rows are masked off."""
  n_atoms = int(coords.shape[0])
  width = int(packed.neighbours.shape[1])
  coords_out = np.zeros((bucket, 3), dtype=coords.dtype)
  features_out = np.zeros((bucket, features.shape[1]), dtype=features.dtype)
  neighbours = np.zeros((bucket, width), dtype=np.int64)
  mask = np.zeros((bucket, width), dtype=np.bool_)
  distance_out = np.zeros((bucket, width), dtype=distance.dtype)
  coords_out[:n_atoms] = coords
  features_out[:n_atoms] = features
  neighbours[:n_atoms] = packed.neighbours
  mask[:n_atoms] = packed.mask
  distance_out[:n_atoms] = distance
  return coords_out, features_out, neighbours, mask, distance_out


def _backbone_frames(backbone: Float[Array, "l 5 3"]) -> Float[Array, "l 4 3"]:
  """Bisector, CA to CB, N to CA, CA to C. Atom order is N, CA, CB, C, O."""
  n_ca = backbone[:, 0, :] - backbone[:, 1, :]
  ca_c = backbone[:, 3, :] - backbone[:, 1, :]
  n_ca = n_ca / jnp.linalg.norm(n_ca, axis=-1, keepdims=True)
  ca_c = ca_c / jnp.linalg.norm(ca_c, axis=-1, keepdims=True)
  total = n_ca + ca_c
  bisector = -total / jnp.linalg.norm(total, axis=-1, keepdims=True)
  ca_cb = backbone[:, 2, :] - backbone[:, 1, :]
  ca_cb = ca_cb / jnp.linalg.norm(ca_cb, axis=-1, keepdims=True)
  return jnp.stack((bisector, ca_cb, n_ca, ca_c), axis=1)


def _protein_dots(
  vectors: Float[Array, "l v 3"],
  neighbours: Int[Array, "l k"],
) -> Float[Array, "l k vv"]:
  """Pairwise dots of source and sink frame vectors, flattened ``V * V``."""
  normed = vectors / norm_no_nan(vectors, axis=-1, keepdims=True)
  src = normed[neighbours]
  dst = jnp.broadcast_to(normed[:, None, :, :], src.shape)
  dots = jnp.einsum("nkvc,nkwc->nkvw", src, dst)
  return dots.reshape(dots.shape[0], dots.shape[1], dots.shape[2] * dots.shape[3])


def _ligand_dots(
  vectors: Float[Array, "l v 3"],
  neighbours: Int[Array, "l k"],
  backbone: Float[Array, "l 5 3"],
  lig_coords: Float[Array, "a 3"],
) -> Float[Array, "l k v"]:
  """Dots of frame vectors with the unit vector from CA to the ligand atom."""
  disp = lig_coords[neighbours] - backbone[:, None, 1, :]
  disp = disp / norm_no_nan(disp, axis=-1, keepdims=True)
  normed = vectors / norm_no_nan(vectors, axis=-1, keepdims=True)
  normed = jnp.broadcast_to(normed[:, None, :, :], disp.shape[:2] + normed.shape[-2:])
  return jnp.einsum("nkvc,nkc->nkv", normed, disp)


def _flat_rbf(encoded: Float[Array, "l k pair bins"]) -> Float[Array, "l k feat"]:
  """Flatten pair and bin axes. An explicit width survives a zero-edge ``K``."""
  width = encoded.shape[2] * encoded.shape[3]
  return encoded.reshape(encoded.shape[0], encoded.shape[1], width)


def _initial_protein_edges(
  model: LaserEncoder,
  vectors: Float[Array, "l v 3"],
  neighbours: Int[Array, "l k"],
  distance: Float[Array, "l k 25"],
) -> Float[Array, "l k e"]:
  rbf = _flat_rbf(model.prot_prot_rbf_encoding(distance))
  return apply_linear(
    model.prot_prot_edge_input_layer,
    jnp.concatenate((rbf, _protein_dots(vectors, neighbours)), axis=-1),
  )


def _initial_ligand_edges(
  model: LaserEncoder,
  vectors: Float[Array, "l v 3"],
  neighbours: Int[Array, "l k"],
  backbone: Float[Array, "l 5 3"],
  lig_coords: Float[Array, "a 3"],
  distance: Float[Array, "l k 5"],
) -> Float[Array, "l k e"]:
  rbf = _flat_rbf(model.lig_prot_rbf_encoding(distance))
  return apply_linear(
    model.lig_prot_edge_input_layer,
    jnp.concatenate((rbf, _ligand_dots(vectors, neighbours, backbone, lig_coords)), axis=-1),
  )
