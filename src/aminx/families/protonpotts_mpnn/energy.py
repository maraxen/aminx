"""ProtonPottsMPNN pair table and Hamiltonian over the 30-token v6 alphabet.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §35.4, §42. Upstream's ``compute_potts_context``
merges the head's table ONCE, reciprocally::

    merged[i, k] = 0.5 * (etab[i, k] + etab[j, reverse_k].T)       (j = E_idx[i, k], where the reverse edge exists)

and ``calc_potts_eners`` then sums every directed edge, ``H(s) = sum_i sum_k merged[i, k, s_i, s_{E_idx[i, k]}]``.

The shipped PottsMPNN path (``potts_mpnn.driver.absolute_energies``) merges TWICE (denominators 2 then 4,
with the self slot excluded the second time) and pads the table to the 22-symbol etab alphabet. Neither
applies here: the table is already indexed by the model token (spec §35.2), so this module reuses
``merge_pair`` and ``potts_energy`` unchanged, with different arguments. They are not edited (spec §35.4).

The sequence argument is in AMINX token order (``vocab.upstream_to_aminx_index``), the order of the converted
head's rows and columns (``convert.token_permutation``).
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.potts_mpnn.etab import merge_pair, potts_energy
from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states, cast_floating


def protonpotts_raw_table(
  model: PottsMPNN,
  coords: Float[Array, "L 4 3"],
  present: Float[Array, " L"],
  residue_idx: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  pad_valid: Bool[Array, " L"],
) -> tuple[Float[Array, "L K V V"], Int[Array, "L K"]]:
  """The head's table BEFORE the reciprocal merge, and ``E_idx``, in aminx token order.

  The result follows ``coords.dtype``.
  """
  dtype = coords.dtype
  scored = cast_floating(model, dtype)
  present_f = present.astype(dtype)
  valid = pad_valid.astype(jnp.bool_)
  _nodes, edges, neighbour_indices = _encoder_states(
    scored.mpnn,
    coords,
    present_f,
    residue_idx.astype(jnp.int32),
    chain_index.astype(jnp.int32),
    inference=True,
  )
  raw = scored.potts_head(edges[-1], neighbour_indices, present_f, valid)
  return cast("Float[Array, 'L K V V']", raw), neighbour_indices


def protonpotts_table(
  model: PottsMPNN,
  coords: Float[Array, "L 4 3"],
  present: Float[Array, " L"],
  residue_idx: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  pad_valid: Bool[Array, " L"],
) -> tuple[Float[Array, "L K V V"], Int[Array, "L K"]]:
  """The merged pair table (upstream's ``etab_out``) and ``E_idx``, in aminx token order."""
  raw, neighbour_indices = protonpotts_raw_table(
    model,
    coords,
    present,
    residue_idx,
    chain_index,
    pad_valid,
  )
  merged = merge_pair(
    raw,
    neighbour_indices,
    pad_valid.astype(jnp.bool_),
    denom=2,
    exclude_self=False,
  )
  return merged, neighbour_indices


def protonpotts_energies(
  model: PottsMPNN,
  coords: Float[Array, "L 4 3"],
  present: Float[Array, " L"],
  residue_idx: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  pad_valid: Bool[Array, " L"],
  sequences: Int[Array, "N L"],
) -> Float[Array, " N"]:
  """Potts energies of ``sequences`` (aminx token indices) on one structure.

  ``MPNNEncode -> PottsHead -> one reciprocal merge -> sum over every directed edge``.
  """
  table, neighbour_indices = protonpotts_table(
    model,
    coords,
    present,
    residue_idx,
    chain_index,
    pad_valid,
  )
  return potts_energy(
    table,
    neighbour_indices,
    pad_valid.astype(jnp.bool_),
    sequences.astype(jnp.int32),
  )
