"""Conditional and block energies for ProtonPottsMPNN pH-design (pure JAX).

Operates on the merged Potts pair table ``table`` ``(L, K, V, V)`` and the neighbour
index ``e_idx`` ``(L, K)``. Slot 0 of every row is the position itself, so
``table[i, 0, a, a]`` is the single-site field. ``table[i, k, a, b]`` is the term for
the directed edge ``i -> e_idx[i, k]`` with token ``a`` at ``i`` and ``b`` at the
neighbour. The Hamiltonian is ``H(s) = sum_i sum_k table[i, k, s_i, s_{e_idx[i, k]}]``
over every directed edge (``aminx.families.potts_mpnn.etab.potts_energy``).

The graph is not symmetric, so a position's token enters ``H`` through its outgoing
rows and through incoming edges from other rows. Both are counted everywhere below.

Self-edges at ``k >= 1`` (``e_idx[i, k] == i``) are real terms of ``H`` whose value
depends on ``s_i``; they are counted on the diagonal ``table[i, k, a, a]``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def incoming_edges(
  e_idx: Int[Array, "L K"],
) -> tuple[Int[Array, " E"], Int[Array, " E"], Int[Array, " E"], Bool[Array, " E"]]:
  """Flat directed edges ``(src, slot, tgt, valid)`` for slots ``k >= 1``, ``E = L * (K - 1)``.

  Edge ``e`` is ``src[e] -> tgt[e] = e_idx[src[e], slot[e]]``. ``valid`` is False for
  self-edges (``tgt == src``): those are already counted by the outgoing sum of their
  own row, so they must not be counted again as incoming. Fixed shapes, so usable under
  ``jit``.
  """
  length, k_slots = e_idx.shape
  src = jnp.broadcast_to(jnp.arange(length)[:, None], (length, k_slots - 1)).reshape(-1)
  slot = jnp.broadcast_to(jnp.arange(1, k_slots)[None, :], (length, k_slots - 1)).reshape(-1)
  tgt = e_idx[:, 1:].reshape(-1).astype(jnp.int32)
  valid = tgt != src
  return src, slot, tgt, valid


def _outgoing(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  positions: Int[Array, " N"],
) -> Float[Array, "N V"]:
  """Sum over each listed row's own edges, with the row's token replaced by the candidate.

  For row ``p`` and candidate ``a``: ``sum_k table[p, k, a, seq[e_idx[p, k]]]``, except
  slots whose neighbour is ``p`` itself, which read ``table[p, k, a, a]``.
  """
  rows = e_idx[positions]  # (N, K)
  t_rows = table[positions]  # (N, K, V, V)
  nbr_tok = seq[rows]  # (N, K)
  index = jnp.broadcast_to(nbr_tok[:, :, None, None], (*t_rows.shape[:-1], 1))
  gathered = jnp.take_along_axis(t_rows, index, axis=-1)[..., 0]  # (N, K, V)
  diagonal = jnp.diagonal(t_rows, axis1=-2, axis2=-1)  # (N, K, V)
  is_self = rows == positions[:, None]
  return jnp.where(is_self[..., None], diagonal, gathered).sum(axis=1)


def _incoming_terms(
  table: Float[Array, "L K V V"],
  seq: Int[Array, " L"],
  src: Int[Array, " E"],
  slot: Int[Array, " E"],
) -> Float[Array, "E V"]:
  """Per incoming edge ``m -> p``, the term ``table[m, k, seq[m], a]`` for each candidate ``a``."""
  return table[src, slot, seq[src], :]


def candidate_energies(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
) -> Float[Array, "L V"]:
  """Token-dependent part of ``H`` at every position, all other positions fixed at ``seq``.

  ``E[i, a]`` collects the outgoing terms of row ``i`` (including its self field) and the
  incoming terms of every edge ``m -> i`` with ``m != i``. For any ``i`` and tokens
  ``a, b``: ``H(seq with seq[i]=a) - H(seq with seq[i]=b) == E[i, a] - E[i, b]``.
  """
  length = table.shape[0]
  out = _outgoing(table, e_idx, seq, jnp.arange(length))
  src, slot, tgt, valid = incoming_edges(e_idx)
  vals = jnp.where(valid[:, None], _incoming_terms(table, seq, src, slot), 0)
  return out.at[tgt].add(vals.astype(out.dtype))


def candidate_energies_at(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  positions: Int[Array, " N"],
) -> Float[Array, "N V"]:
  """Rows ``positions`` of :func:`candidate_energies`, without computing the other rows.

  The incoming part is a masked matmul over the flat edge list, so it costs
  ``O(N * L * K * V)`` instead of a full ``(L, V)`` table.
  """
  positions = jnp.asarray(positions, dtype=jnp.int32)
  out = _outgoing(table, e_idx, seq, positions)
  src, slot, tgt, valid = incoming_edges(e_idx)
  hit = (tgt[None, :] == positions[:, None]) & valid[None, :]  # (N, E)
  vals = _incoming_terms(table, seq, src, slot)  # (E, V)
  return out + hit.astype(vals.dtype) @ vals


def block_stability_potentials(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
) -> tuple[Float[Array, "B V"], Float[Array, "B B V V"]]:
  """Decompose the block-dependent part of ``H`` into unary and pairwise tables.

  ``block`` holds distinct positions; entries with ``block_valid`` False are ignored
  entirely (their position is treated as a fixed position at ``seq``). For any joint
  assignment ``x`` of the valid members, with all other positions fixed at ``seq``::

      H(seq with block set to x) - H(seq with block set to x0)
          == J(x) - J(x0),   J(x) = sum_b unary[b, x[b]] + sum_{bi<bj} pair[bi, bj, x[bi], x[bj]]

  ``unary[b]`` is the self field of ``block[b]`` plus every edge between it and a fixed
  position (outgoing from it, incoming to it). ``pair[bi, bj]`` (only ``bi < bj`` is
  filled) collects every edge between two members in either direction; an edge
  ``bj -> bi`` is transposed so that the first axis is always ``bi``'s token.
  """
  length, k_slots = e_idx.shape
  n_block = block.shape[0]
  vocab = table.shape[-1]
  dtype = table.dtype
  block = block.astype(jnp.int32)

  # Position -> index of its VALID block member, or -1 for fixed positions. Invalid
  # entries are sent out of range and dropped, so a padded entry never claims a position.
  member_slot = jnp.where(block_valid, block, length)
  pos_to_b = (
    jnp.full((length,), -1, dtype=jnp.int32)
    .at[member_slot]
    .set(jnp.arange(n_block, dtype=jnp.int32), mode="drop")
  )

  # Outgoing edges of every valid member (all slots, self-edges included exactly once).
  e_rows = e_idx[block]  # (B, K)
  t_rows = table[block]  # (B, K, V, V)
  src_ok = block_valid[:, None]  # (B, 1)
  is_self = e_rows == block[:, None]  # (B, K)
  tgt_b = pos_to_b[e_rows]  # (B, K), -1 when the neighbour is fixed
  diagonal = jnp.diagonal(t_rows, axis1=-2, axis2=-1)  # (B, K, V)
  nbr_tok = seq[e_rows]  # (B, K)
  index = jnp.broadcast_to(nbr_tok[:, :, None, None], (*t_rows.shape[:-1], 1))
  gathered = jnp.take_along_axis(t_rows, index, axis=-1)[..., 0]  # (B, K, V)
  out_vals = jnp.where(is_self[..., None], diagonal, gathered)

  unary_mask = src_ok & (is_self | (tgt_b < 0))  # (B, K)
  unary = jnp.where(unary_mask[..., None], out_vals, 0).sum(axis=1).astype(dtype)  # (B, V)

  pair_mask = src_ok & ~is_self & (tgt_b >= 0)  # (B, K) edges between two members
  b_idx = jnp.broadcast_to(jnp.arange(n_block, dtype=jnp.int32)[:, None], (n_block, k_slots))
  t_idx = jnp.maximum(tgt_b, 0)
  lo = jnp.where(pair_mask, jnp.minimum(b_idx, t_idx), 0).reshape(-1)
  hi = jnp.where(pair_mask, jnp.maximum(b_idx, t_idx), 0).reshape(-1)
  oriented = jnp.where((b_idx < t_idx)[..., None, None], t_rows, jnp.swapaxes(t_rows, -1, -2))
  oriented = jnp.where(pair_mask[..., None, None], oriented, 0).reshape(-1, vocab, vocab)
  pair = (
    jnp.zeros((n_block, n_block, vocab, vocab), dtype=dtype).at[lo, hi].add(oriented.astype(dtype))
  )

  # Incoming edges from fixed sources into a valid member.
  src, slot, tgt, valid = incoming_edges(e_idx)
  tgt_in = pos_to_b[tgt]
  in_ok = valid & (pos_to_b[src] < 0) & (tgt_in >= 0)
  in_vals = jnp.where(in_ok[:, None], _incoming_terms(table, seq, src, slot), 0)
  unary = unary.at[jnp.maximum(tgt_in, 0)].add(in_vals.astype(dtype))

  return unary, pair
