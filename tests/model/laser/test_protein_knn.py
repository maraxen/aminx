"""Protein kNN keeps the self-edge. loop=False fails that parity check."""

from __future__ import annotations

import numpy as np
import pytest

from aminx.model.laser.encoders import LIGAND_ATOM_BUCKETS, LIGAND_ATOMS, ligand_atom_bucket
from aminx.model.laser.graphs import knn_graph


def _coords() -> np.ndarray:
  rng = np.random.default_rng(0)
  return rng.normal(size=(9, 3)).astype(np.float64)


def assert_self_edge_parity(coords: np.ndarray, edges: np.ndarray, *, k: int) -> None:
  """Match the loop=True protein kNN, including a self-edge on every row.

  This is the host graph B0 coordinates are consumed with. A ``loop=False``
  edge list fails it.
  """
  reference = knn_graph(coords, k, loop=True)
  if not np.array_equal(edges, reference):
    msg = "protein kNN does not match the loop=True reference"
    raise AssertionError(msg)
  queries = edges[1]
  neighbours = edges[0]
  n_rows = int(coords.shape[0])
  for row in range(n_rows):
    if row not in neighbours[queries == row]:
      msg = f"protein row {row} does not contain itself"
      raise AssertionError(msg)


def test_every_protein_row_contains_itself() -> None:
  """kNN with loop=True places each residue in its own neighbour list."""
  coords = _coords()
  edges = knn_graph(coords, 4, loop=True)
  assert_self_edge_parity(coords, edges, k=4)


def test_loop_false_fails_self_edge_parity() -> None:
  """Negative control: dropping the self-edge fails the loop=True parity check."""
  coords = _coords()
  edges = knn_graph(coords, 4, loop=False)
  with pytest.raises(AssertionError, match="loop=True reference"):
    assert_self_edge_parity(coords, edges, k=4)


def test_ligand_atom_axis_is_bucketed() -> None:
  """Two ragged ligand sizes in one bucket share one padded length."""
  assert LIGAND_ATOMS.bucket_boundaries == LIGAND_ATOM_BUCKETS
  assert ligand_atom_bucket(3) == ligand_atom_bucket(8) == 8
  assert ligand_atom_bucket(9) == 16
