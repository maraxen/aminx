"""Pair-table symmetrization and Potts energies.

Ports ``etab_utils.merge_duplicate_pairE`` (geometric path), ``calc_eners``,
and ``positional_potts_energy``. Dtype follows the incoming tables: float64
inputs stay float64.

The etab alphabet is ``ACDEFGHIKLMNPQRSTVWY-X`` (``-`` = 20, ``X`` = 21).
The first 20 letters match the ProteinMPNN alphabet; model ``X`` (20) maps to
etab ``X`` (21). Energy tables used by ``potts_energy`` are 22x22 with the
``-`` and ``X`` rows and columns left at zero.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.potts_mpnn.featurize import MODEL_ALPHABET

ETAB_ALPHABET = f"{MODEL_ALPHABET[:-1]}-X"
N_AA = 20
N_ETAB = len(ETAB_ALPHABET)
ETAB_GAP = ETAB_ALPHABET.index("-")
ETAB_X = ETAB_ALPHABET.index("X")

# Model index p maps to etab index. Model X (last letter) lands on etab X.
_MODEL_TO_ETAB = jnp.asarray([*range(N_AA), ETAB_X], dtype=jnp.int32)
# etab gap has no model letter and reads back as -1.
_ETAB_TO_MODEL = jnp.asarray([*range(N_AA), -1, MODEL_ALPHABET.index("X")], dtype=jnp.int32)


def model_to_etab(seq: Int[Array, "..."]) -> Int[Array, "..."]:
  """Map ProteinMPNN indices onto the etab alphabet.

  Letters 0..19 are identical. Model ``X`` (20) becomes etab ``X`` (21).
  The result dtype matches ``seq``.
  """
  return _MODEL_TO_ETAB[seq].astype(seq.dtype)


def etab_to_model(seq: Int[Array, "..."]) -> Int[Array, "..."]:
  """Map etab indices back to the ProteinMPNN alphabet.

  Etabs ``-`` (20) is not a model letter and becomes ``-1``. Model letters
  round-trip: ``etab_to_model(model_to_etab(s)) == s``.
  """
  return _ETAB_TO_MODEL[seq].astype(seq.dtype)


def pad_etab_energy(etab: Float[Array, "... 20 20"]) -> Float[Array, "... 22 22"]:
  """Pad a 20x20 pair table to 22x22 with zero ``-`` and ``X`` slots.

  Matches ``F.pad(etab, (0, 2, 0, 2))`` on the amino-acid axes. Dtype is
  unchanged.
  """
  width = etab.ndim - 2
  return jnp.pad(etab, [(0, 0)] * width + [(0, N_ETAB - N_AA), (0, N_ETAB - N_AA)])


def merge_pair(
  etab: Float[Array, "L K A A"],
  e_idx: Int[Array, "L K"],
  pad_valid: Bool[Array, " L"],
  *,
  denom: int,
  exclude_self: bool,
  transpose: bool = True,
) -> Float[Array, "L K A A"]:
  """Average a directed pair table with its reverse edge.

  Ports ``merge_duplicate_pairE_geometric``. An edge ``i → j`` at slot ``k``
  updates the reverse entry ``j → i``:

  ``etab[j, k_rev] = (etab[j, k_rev] + etab[i, k].T) / denom``

  when that reverse slot exists and both endpoints are ``pad_valid``. Several
  writers to one slot keep the last edge in ``(i, k)`` order, and the
  right-hand side is read from the original table. Edges with no valid reverse
  stay undivided. ``exclude_self`` drops writes whose reverse slot is ``k == 0``
  (upstream ``denom != 2``).

  ``etab_forward`` uses ``denom=2, exclude_self=False``. ``etab_energy`` uses
  ``denom=4, exclude_self=True`` on ``etab_forward``. ``transpose=False`` adds
  the forward table without swapping the amino-acid axes (the megascale
  negative control).
  """
  # Neighbour indices from oracles are int64. Scatter updates require one
  # integer width, and int32 covers every residue index we index with.
  e_idx = e_idx.astype(jnp.int32)
  length, k, _, _ = etab.shape
  neighbour = e_idx
  in_range = (neighbour >= 0) & (neighbour < length)
  safe_neighbour = jnp.clip(neighbour, 0, length - 1)
  # e_idx[j] is (L, K, K): slot k2 of the neighbour of edge (i, k).
  source = jnp.arange(length)[:, None, None]
  reverse_hits = (e_idx[safe_neighbour] == source) & in_range[:, :, None]
  slots = jnp.arange(k)
  rev_slot = jnp.max(jnp.where(reverse_hits, slots, -1), axis=-1)
  has_reverse = rev_slot >= 0
  # ``in_range & x`` is ``where(in_range, x, False)`` for booleans. The ``&`` form avoids a
  # BOOL ``Where``, which ONNX Runtime has no kernel for (export spike run 1340fe84).
  both_valid = pad_valid[:, None] & in_range & pad_valid[safe_neighbour]
  merges = has_reverse & both_valid
  if exclude_self:
    merges = merges & (rev_slot != 0)

  safe_slot = jnp.clip(rev_slot, 0, k - 1)
  # One index array gathers a paired (neighbour, slot). Two index arrays would
  # not zip under JAX's indexing rules.
  alphabet = etab.shape[-1]
  paired = (safe_neighbour * k + safe_slot).reshape(-1)
  reverse_table = etab.reshape(length * k, alphabet, alphabet)[paired].reshape(
    length,
    k,
    alphabet,
    alphabet,
  )
  partner = jnp.swapaxes(etab, -1, -2) if transpose else etab
  updated = (reverse_table + partner) / jnp.asarray(denom, dtype=etab.dtype)

  flat_n = length * k
  order = jnp.arange(flat_n, dtype=jnp.int32).reshape(length, k)
  writer_order = jnp.where(merges, order, jnp.int32(-1))
  flat_target = (safe_neighbour * k + safe_slot).reshape(-1)
  winners = jnp.full((flat_n,), jnp.int32(-1))
  winners = winners.at[flat_target].max(writer_order.reshape(-1))
  written = winners >= 0
  source_index = jnp.clip(winners, 0, flat_n - 1)
  updated_flat = updated.reshape(flat_n, alphabet, alphabet)[source_index]
  original_flat = etab.reshape(flat_n, alphabet, alphabet)
  merged = jnp.where(written[:, None, None], updated_flat, original_flat)
  return merged.reshape(length, k, alphabet, alphabet)


def potts_energy(
  etab: Float[Array, "L K A A"],
  e_idx: Int[Array, "L K"],
  pad_valid: Bool[Array, " L"],
  seq: Int[Array, "... L"],
) -> Float[Array, "..."]:
  """Sum pair energies of one or more sequences.

  Ports ``calc_eners`` with ``filter=False`` and no factor of one half.
  ``seq`` is in the etab alphabet, shape ``(L,)`` or ``(N, L)``. A term is
  kept only when both endpoints are ``pad_valid``; gap rows still contribute
  when they are real A0 rows, through the zero ``-`` / ``X`` slots.

  Returns a scalar for a single sequence and shape ``(N,)`` for a batch.
  """
  single = seq.ndim == 1
  sequences = seq[None] if single else seq
  scores = _batch_energy(etab, e_idx.astype(jnp.int32), pad_valid, sequences)
  return scores[0] if single else scores


def positional_potts_energy(
  etab: Float[Array, "L K A A"],
  e_idx: Int[Array, "L K"],
  pad_valid: Bool[Array, " L"],
  seq: Int[Array, " L"],
  pos: int | Int[Array, ""],
) -> Float[Array, " A"]:
  """Energy of every amino acid at ``pos`` given the other sites.

  ``pos`` may be a static Python ``int`` or a traced scalar: A5 sweeps it inside
  ``lax`` control flow, while the etab tests index it with a literal.

  Ports ``positional_potts_energy`` for one unbatched protein (upstream reads
  batch index 0). Pair slots ``k > 0`` are summed; slot 0 contributes its
  diagonal. Terms with an endpoint outside ``pad_valid`` are zero. The result
  has one entry per column of ``etab``.
  """
  alphabet = etab.shape[-1]
  self_energy = jnp.diagonal(etab[pos, 0])
  neighbours = e_idx.astype(jnp.int32)[pos, 1:]
  length = seq.shape[0]
  in_range = (neighbours >= 0) & (neighbours < length)
  safe = jnp.clip(neighbours, 0, length - 1)
  amino = seq[safe]
  pair_etab = etab[pos, 1:]
  n_pairs = pair_etab.shape[0]
  column = jnp.broadcast_to(amino[:, None, None], (n_pairs, alphabet, 1))
  pair_energy = jnp.take_along_axis(pair_etab, column, axis=-1).squeeze(-1)
  neighbour_ok = in_range & pad_valid[safe] & pad_valid[pos]
  # Explicit-dtype zero: a weak-typed ``0`` lowers to an int32 ``Where`` branch under
  # jax2onnx, which ONNX type inference rejects (export spike run 1340fe84).
  pair_energy = jnp.where(neighbour_ok[:, None], pair_energy, jnp.zeros_like(pair_energy))
  total = self_energy + pair_energy.sum(axis=0)
  return jnp.where(pad_valid[pos], total, jnp.zeros_like(total))


def _batch_energy(
  etab: Float[Array, "L K A A"],
  e_idx: Int[Array, "L K"],
  pad_valid: Bool[Array, " L"],
  sequences: Int[Array, "N L"],
) -> Float[Array, " N"]:
  """Energies of ``N`` sequences by one broadcast gather of ``etab[i, k, s_i, s_j]``.

  Broadcasting over the sequence axis, not ``vmap``: L-DRV R1 bans vmap in
  ``families/``.
  """
  length, k, _, _ = etab.shape
  in_range = (e_idx >= 0) & (e_idx < length)
  safe = jnp.clip(e_idx, 0, length - 1)
  # ``&``, not a BOOL ``where``: see merge_pair.
  edge_ok = pad_valid[:, None] & in_range & pad_valid[safe]
  rows = jnp.arange(length)[:, None]
  slots = jnp.arange(k)[None, :]
  amino_i = sequences[:, :, None]
  amino_j = sequences[:, safe]
  terms = etab[rows, slots, amino_i, amino_j]
  zero = jnp.asarray(0, dtype=etab.dtype)
  return jnp.where(edge_ok[None], terms, zero).sum(axis=(-2, -1))
