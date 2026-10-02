"""Host-side LASEr graphs.

Protein and ligand kNN follow ``torch_cluster.knn_graph(..., flow="source_to_target")``:
``edge_index[0]`` is the neighbour and ``edge_index[1]`` is the query. ``loop=True``
keeps the self-edge. Downstream decoding masks it (``rank < rank`` is false); the
encoder still receives it.

Ligand-protein edges follow ``compute_ligand_protein_knn_graph``: CA-ligand
distances in float64, a strict cutoff, then ``k`` nearest ligand atoms. Neighbours
are the stable argsort of squared (kNN) or float64 (ligand-protein) distance, ties
broken toward the lower index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GraphStructure:
  """Checkpoint ``model_params["graph_structure"]``."""

  pr_pr_knn_graph_k: int
  lig_pr_knn_graph_k: int
  lig_lig_knn_graph_k: int
  lig_pr_distance_cutoff: float


@dataclass(frozen=True, slots=True)
class PackedEdges:
  """Dense ``(N, K)`` neighbourhood plus the edge-list gather back to upstream order."""

  neighbours: Int[NDArray[np.int64], "n k"]
  mask: Bool[NDArray[np.bool_], "n k"]
  sink: Int[NDArray[np.int64], " e"]
  slots: Int[NDArray[np.int64], " e"]


@dataclass(frozen=True, slots=True)
class LaserGraphs:
  """The three graphs ``construct_graphs`` stores on one structure."""

  pr_pr: Int[NDArray[np.int64], "2 epp"]
  lig_pr: Int[NDArray[np.int64], "2 elp"]
  lig_lig: Int[NDArray[np.int64], "2 ell"]


def knn_graph(
  coords: Float[NDArray[np.floating], "n 3"],
  k: int,
  *,
  loop: bool,
  batch: Int[NDArray[np.integer], " n"] | None = None,
) -> Int[NDArray[np.int64], "2 e"]:
  """k nearest neighbours. ``loop=False`` drops the self-edge.

  ``batch`` restricts neighbours to atoms that share an id (one ligand). Ids are
  walked in ascending order, matching ``torch_cluster``'s batch pointer.
  """
  points = np.asarray(coords)
  n_points = int(points.shape[0])
  if n_points == 0 or k <= 0:
    return np.zeros((2, 0), dtype=np.int64)
  if batch is None:
    return _knn_block(points, np.arange(n_points, dtype=np.int64), k, loop=loop)
  labels = np.asarray(batch)
  blocks: list[NDArray[np.int64]] = []
  for label in np.unique(labels):
    members = np.flatnonzero(labels == label).astype(np.int64, copy=False)
    blocks.append(_knn_block(points[members], members, k, loop=loop))
  if not blocks:
    return np.zeros((2, 0), dtype=np.int64)
  return np.concatenate(blocks, axis=1)


def ligand_protein_knn(
  ca_coords: Float[NDArray[np.floating], "l 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  k: int,
  cutoff: float,
) -> Int[NDArray[np.int64], "2 e"]:
  """Ligand-atom source, protein-residue sink, for CA atoms inside ``cutoff``."""
  ca = np.asarray(ca_coords, dtype=np.float64)
  ligand = np.asarray(ligand_coords, dtype=np.float64)
  n_res = int(ca.shape[0])
  n_atoms = int(ligand.shape[0])
  if n_res == 0 or n_atoms == 0 or k <= 0:
    return np.zeros((2, 0), dtype=np.int64)
  delta = ca[:, None, :] - ligand[None, :, :]
  distances = np.linalg.norm(delta, axis=-1)
  connected = np.flatnonzero(np.sum(distances < cutoff, axis=1) > 0)
  width = min(k, n_atoms)
  if int(connected.shape[0]) == 0 or width == 0:
    return np.zeros((2, 0), dtype=np.int64)
  nearest = np.argsort(distances[connected], axis=1, kind="stable")[:, :width]
  source = nearest.reshape(-1).astype(np.int64, copy=False)
  sink = np.repeat(connected.astype(np.int64, copy=False), width)
  return np.stack((source, sink))


def build_laser_graphs(
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  structure: GraphStructure,
  *,
  protein_loop: bool = True,
) -> LaserGraphs:
  """Protein kNN on CA, per-ligand kNN, and ligand-to-protein kNN."""
  ca = np.asarray(backbone)[:, 1, :]
  pr_pr = knn_graph(ca, structure.pr_pr_knn_graph_k, loop=protein_loop)
  lig_lig = knn_graph(
    ligand_coords,
    structure.lig_lig_knn_graph_k,
    loop=True,
    batch=ligand_subbatch,
  )
  lig_pr = ligand_protein_knn(
    ca,
    ligand_coords,
    structure.lig_pr_knn_graph_k,
    structure.lig_pr_distance_cutoff,
  )
  return LaserGraphs(pr_pr=pr_pr, lig_pr=lig_pr, lig_lig=lig_lig)


def scatter_edge_values(
  n_nodes: int,
  packed: PackedEdges,
  values: NDArray[np.floating],
) -> NDArray[np.floating]:
  """Place edge-list values into the dense ``(N, K, ...)`` neighbourhood."""
  array = np.asarray(values)
  width = int(packed.neighbours.shape[1])
  dense = np.zeros((n_nodes, width, *array.shape[1:]), dtype=array.dtype)
  if int(array.shape[0]) > 0:
    dense[packed.sink, packed.slots] = array
  return dense


def protein_edge_distances(
  backbone: Float[NDArray[np.floating], "l 5 3"],
  edge_index: Int[NDArray[np.integer], "2 e"],
) -> Float[NDArray[np.floating], "e 25"]:
  """``cdist`` of the five backbone atoms, flattened with the destination atom fastest."""
  edges = np.asarray(edge_index)
  n_edges = int(edges.shape[1])
  coords = np.asarray(backbone)
  if n_edges == 0:
    return np.zeros((0, 25), dtype=coords.dtype)
  src = coords[edges[0]]
  dst = coords[edges[1]]
  delta = src[:, :, None, :] - dst[:, None, :, :]
  return np.linalg.norm(delta, axis=-1).reshape(n_edges, 25)


def ligand_protein_distances(
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  edge_index: Int[NDArray[np.integer], "2 e"],
) -> Float[NDArray[np.floating], "e 5"]:
  """Float64 CA-ligand ``cdist``, then the working dtype (upstream ``.float()``)."""
  edges = np.asarray(edge_index)
  n_edges = int(edges.shape[1])
  coords = np.asarray(backbone)
  if n_edges == 0:
    return np.zeros((0, 5), dtype=coords.dtype)
  protein = coords[edges[1]].astype(np.float64, copy=False)
  atom = np.asarray(ligand_coords)[edges[0]].astype(np.float64, copy=False)
  delta = protein - atom[:, None, :]
  distance = np.linalg.norm(delta, axis=-1)
  return np.asarray(distance, dtype=coords.dtype)


def ligand_ligand_distances(
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  edge_index: Int[NDArray[np.integer], "2 e"],
) -> Float[NDArray[np.floating], " e"]:
  """Float64 ligand-ligand ``cdist``, then the working dtype (upstream ``.float()``)."""
  edges = np.asarray(edge_index)
  n_edges = int(edges.shape[1])
  coords = np.asarray(ligand_coords)
  if n_edges == 0:
    return np.zeros((0,), dtype=coords.dtype)
  src = coords[edges[0]].astype(np.float64, copy=False)
  dst = coords[edges[1]].astype(np.float64, copy=False)
  distance = np.linalg.norm(src - dst, axis=-1)
  return np.asarray(distance, dtype=coords.dtype)


def pack_edges(n_nodes: int, edge_index: Int[NDArray[np.integer], "2 e"]) -> PackedEdges:
  """Scatter an edge list into a dense row-is-sink neighbourhood."""
  edges = np.asarray(edge_index)
  source = np.asarray(edges[0], dtype=np.int64)
  sink = np.asarray(edges[1], dtype=np.int64)
  n_edges = int(source.shape[0])
  if n_nodes == 0 or n_edges == 0:
    empty_i = np.zeros((n_nodes, 0), dtype=np.int64)
    empty_m = np.zeros((n_nodes, 0), dtype=np.bool_)
    empty_e = np.zeros((0,), dtype=np.int64)
    return PackedEdges(empty_i, empty_m, empty_e, empty_e)
  counts = np.bincount(sink, minlength=n_nodes)
  width = int(counts.max())
  slots = np.empty((n_edges,), dtype=np.int64)
  seen = np.zeros((n_nodes,), dtype=np.int64)
  for edge in range(n_edges):
    row = int(sink[edge])
    slots[edge] = seen[row]
    seen[row] = slots[edge] + 1
  neighbours = np.zeros((n_nodes, width), dtype=np.int64)
  mask = np.zeros((n_nodes, width), dtype=np.bool_)
  neighbours[sink, slots] = source
  mask[sink, slots] = True
  return PackedEdges(neighbours, mask, sink, slots)


def _knn_block(
  coords: NDArray[np.floating],
  global_index: NDArray[np.int64],
  k: int,
  *,
  loop: bool,
) -> NDArray[np.int64]:
  """One contiguous component. Queries are emitted in ascending global index."""
  n_local = int(coords.shape[0])
  if n_local == 0 or k <= 0:
    return np.zeros((2, 0), dtype=np.int64)
  requested = k if loop else k + 1
  take = min(requested, n_local)
  delta = coords[:, None, :] - coords[None, :, :]
  squared = np.sum(delta * delta, axis=-1)
  order = np.argsort(squared, axis=1, kind="stable")[:, :take]
  sources: list[int] = []
  sinks: list[int] = []
  for local in range(n_local):
    query = int(global_index[local])
    kept = 0
    for column in range(take):
      neighbour_local = int(order[local, column])
      if not loop and neighbour_local == local:
        continue
      if kept == k:
        break
      sources.append(int(global_index[neighbour_local]))
      sinks.append(query)
      kept += 1
  if not sources:
    return np.zeros((2, 0), dtype=np.int64)
  return np.stack(
    (
      np.asarray(sources, dtype=np.int64),
      np.asarray(sinks, dtype=np.int64),
    ),
  )
