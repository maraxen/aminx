"""The v2 CA-only vehicle forgives a tie-order slot swap but not a real neighbor-selection difference."""

from __future__ import annotations

import numpy as np

from scripts.parity import ca_only_parity as v


def _case(n: int = 6, k: int = 6, c: int = 3):
  rng = np.random.default_rng(0)
  idx = np.tile(np.arange(k), (n, 1))
  edges = rng.normal(size=(n, k, c))
  d_adj = rng.uniform(1.0, 9.0, size=(n, k))
  valid = np.ones(n, dtype=bool)
  return idx, edges, d_adj, valid


def test_tie_swap_is_forgiven_and_edge_diff_is_zero() -> None:
  idx, edges, d_adj, valid = _case()
  t_idx, t_edges = idx.copy(), edges.copy()
  for r in range(len(idx)):  # swap the last two slots and make them a D_adjust tie
    d_adj[r, -1] = d_adj[r, -2]
    t_idx[r, [-2, -1]] = t_idx[r, [-1, -2]]
    t_edges[r, [-2, -1]] = t_edges[r, [-1, -2]]
  raw = np.abs(edges - t_edges).max()
  assert raw > 0.1  # slot-by-slot comparison would report this
  je, te, n_set, n_untied = v._canonical_edge_diff(edges, t_edges, idx, t_idx, d_adj, valid)
  assert np.abs(je - te).max() == 0.0
  assert (n_set, n_untied) == (0, 0)


def test_untied_slot_difference_is_counted() -> None:
  idx, edges, d_adj, valid = _case()
  t_idx, t_edges = idx.copy(), edges.copy()
  t_idx[2, [0, 1]] = t_idx[2, [1, 0]]  # D_adjust of slots 0 and 1 differ: not a tie
  t_edges[2, [0, 1]] = t_edges[2, [1, 0]]
  _, _, n_set, n_untied = v._canonical_edge_diff(edges, t_edges, idx, t_idx, d_adj, valid)
  assert n_set == 0
  assert n_untied == 2


def test_different_neighbor_set_is_counted() -> None:
  idx, edges, d_adj, valid = _case(k=6)
  t_idx = idx.copy()
  t_idx[1, 5] = 99  # a neighbor upstream does not select
  d_adj_wide = np.pad(d_adj, ((0, 0), (0, 0)))
  # index 99 is outside d_adj; use a larger table so the untied check can read it
  big = np.zeros((6, 100))
  big[:, :6] = d_adj_wide
  big[1, 99] = 1e3
  _, _, n_set, n_untied = v._canonical_edge_diff(edges, edges.copy(), idx, t_idx, big, valid)
  assert n_set == 1
  assert n_untied == 1


def test_masked_rows_are_ignored() -> None:
  idx, edges, d_adj, valid = _case()
  valid[3] = False
  t_idx = idx.copy()
  t_idx[3] = t_idx[3][::-1]
  _, _, n_set, n_untied = v._canonical_edge_diff(edges, edges.copy(), idx, t_idx, d_adj, valid)
  assert (n_set, n_untied) == (0, 0)
