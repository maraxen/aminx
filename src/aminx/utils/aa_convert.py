"""Utility functions for converting between AlphaFold and ProteinMPNN amino acid orders."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from proxide.chem.residues import (
  restype_order,
  restype_order_with_x,
  unk_restype_index,
)

from aminx.types.arrays import ProteinSequence

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

_AF_TO_MPNN_PERM = jnp.array(
  [MPNN_ALPHABET.index(k) for k in AF_ALPHABET],
)

_MPNN_TO_AF_PERM = jnp.array(
  [AF_ALPHABET.index(k) for k in MPNN_ALPHABET],
)


def af_to_mpnn(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a sequence of integer indices from AlphaFold's to ProteinMPNN's alphabet order."""
  return _AF_TO_MPNN_PERM[sequence].astype(jnp.int8)


def mpnn_to_af(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a sequence of integer indices from ProteinMPNN's to AlphaFold's alphabet order."""
  return _MPNN_TO_AF_PERM[sequence].astype(jnp.int8)


def training_labels(aatype: ProteinSequence) -> ProteinSequence:
  """Turn a batch's persisted ``aatype`` into the integer labels the model is trained against.

  ``aatype`` -- as parsed by proxide, and as persisted under the ``"aatype"`` key of
  preprocessed array_record datasets -- is **AF**-ordered (``ARNDCQEGHILKMFPSTWYVX``). The
  model's token space (logits, the decoder's one-hot embedding input, CE targets) is
  **MPNN**-ordered (``ACDEFGHIKLMNPQRSTVWYX``). Every value 0-20 is legal in both, so nothing
  downstream can detect a mix-up: it trains on permuted labels without any error or loss
  anomaly (issue #109). Every place that binds a loader batch's ``aatype`` to a trainer's
  ``sequence`` argument must go through this function.

  Values outside ``[0, 21)`` (e.g. a negative padding sentinel) are passed through unchanged,
  not remapped onto a real residue; the loss masks them out either way.

  Args:
    aatype: AF-ordered integer residue types, any shape.

  Returns:
    MPNN-ordered integer labels, same shape, dtype ``int32``.

  """
  aatype = jnp.asarray(aatype)
  in_range = (aatype >= 0) & (aatype < len(AF_ALPHABET))
  converted = _AF_TO_MPNN_PERM[jnp.clip(aatype, 0, len(AF_ALPHABET) - 1)]
  return jnp.where(in_range, converted, aatype).astype(jnp.int32)


def string_key_to_index(
  string_keys: np.ndarray,
  key_map: Mapping[str, int],
  unk_index: int | None = None,
) -> jax.Array:
  """Convert string keys to integer indices based on a mapping.

  Efficient vectorized implementation to convert a 1D array of string keys
  to a 1D array of integer indices using a provided mapping. If a key is not found in the mapping,
  it is replaced with a specified unknown index.

  Args:
    string_keys: A 1D array of string keys.
    key_map: A dictionary mapping string keys to integer indices.
    unk_index: The index to use for unknown keys not found in the mapping. If None, uses the
      length of the key_map as the unknown index.

  Returns:
    A 1D array of integer indices corresponding to the string keys.

  """
  if unk_index is None:
    unk_index = len(key_map)

  sorted_keys = np.array(sorted(key_map.keys()))
  sorted_values = np.array([key_map[k] for k in sorted_keys])
  indices = np.searchsorted(sorted_keys, string_keys)
  indices = np.clip(indices, 0, len(sorted_keys) - 1)

  found_keys = sorted_keys[indices]
  is_known = found_keys == string_keys

  return jnp.where(is_known, sorted_values[indices], unk_index)


def string_to_protein_sequence(
  sequence: str,
  aa_map: dict | None = None,
  unk_index: int | None = None,
) -> ProteinSequence:
  """Convert a string sequence to a ProteinSequence.

  Args:
    sequence: A string containing the protein sequence.
    aa_map: A dictionary mapping amino acid names to integer indices. If None, uses the default
      `restype_order` mapping.
    unk_index: The index to use for unknown amino acids not found in the mapping. If None, uses
      `unk_restype_index`.

  Returns:
    A ProteinSequence containing the amino acid type indices corresponding to the input string.

  """
  if unk_index is None:
    unk_index = unk_restype_index

  if aa_map is None:
    aa_map = restype_order
    # Corrected line: Split the string into a list of characters for string_key_to_index
    return af_to_mpnn(
      string_key_to_index(np.array(list(sequence)), aa_map, unk_index).astype(
        jnp.int8,
      ),
    )
  # This part was already correct for when aa_map is explicitly provided,
  # as it correctly uses list(sequence).
  return string_key_to_index(np.array(list(sequence)), aa_map, unk_index).astype(
    jnp.int8,
  )


def protein_sequence_to_string(
  sequence: ProteinSequence,
  aa_map: dict | None = None,
) -> str:
  """Convert a ProteinSequence to a string.

  Args:
    sequence: A ProteinSequence containing amino acid type indices.
    aa_map: A dictionary mapping amino acid type indices to their corresponding names. If None,
      uses the default `restype_order` mapping.

  Returns:
    A string representation of the protein sequence.

  """
  if aa_map is None:
    aa_map = {i: aa for aa, i in restype_order_with_x.items()}

  af_seq = mpnn_to_af(jnp.asarray(sequence)).astype(np.int32)

  return "".join([aa_map.get(int(aa), "X") for aa in af_seq])
