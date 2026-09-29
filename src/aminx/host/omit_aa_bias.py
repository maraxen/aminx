"""Compile ``omit_aa`` / ``omit_aa_per_position`` into a logit bias.

Upstream ProteinMPNN / PottsMPNN / LASEr write ``-1e8`` on omitted amino acids
at positions that are still being designed (fixed positions are left alone).
The constant is the upstream omit penalty. An empty omit specification returns
the caller's bias object unchanged so the MPNN kernel sees the same value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp

from aminx.utils.aa_convert import MPNN_ALPHABET

OMIT_AA_BIAS = -1e8
_ALPHABET_SIZE = len(MPNN_ALPHABET)


def omit_aa_is_active(
  omit_aa: Sequence[str],
  omit_aa_per_position: Mapping[int, str] | None,
) -> bool:
  """True when either omit channel names at least one amino acid."""
  if omit_aa_per_position:
    return True
  return any(token for token in omit_aa)


def compile_omit_aa_bias(
  bias: Any,  # noqa: ANN401 -- caller bias is None or an array of several ranks
  *,
  fixed_mask_row: Any,  # noqa: ANN401 -- concrete or traced (L,) mask; 1 marks fixed
  omit_aa: Sequence[str],
  omit_aa_per_position: Mapping[int, str] | None,
  seq_len: int,
) -> Any:  # noqa: ANN401
  """Return ``bias`` plus ``-1e8`` on omitted letters at designed positions.

  ``fixed_mask_row`` uses the host convention: values ``> 0`` are fixed and do
  not receive the omit penalty. ``None`` treats every position as designed.
  """
  if not omit_aa_is_active(omit_aa, omit_aa_per_position):
    return bias

  penalty = jnp.zeros((seq_len, _ALPHABET_SIZE), dtype=jnp.float32)
  for letter in _letters(omit_aa):
    penalty = penalty.at[:, _aa_index(letter)].set(jnp.float32(OMIT_AA_BIAS))
  if omit_aa_per_position is not None:
    for pos, token in omit_aa_per_position.items():
      position = int(pos)
      if position < 0 or position >= seq_len:
        msg = f"omit_aa_per_position key {position} is outside 0..{seq_len - 1}"
        raise ValueError(msg)
      for letter in token:
        penalty = penalty.at[position, _aa_index(letter)].set(jnp.float32(OMIT_AA_BIAS))
  if fixed_mask_row is not None:
    designed = jnp.asarray(fixed_mask_row, dtype=jnp.float32) <= 0
    penalty = jnp.where(designed[:, None], penalty, jnp.float32(0.0))
  return _broadcast_bias(bias, seq_len) + penalty


def _letters(omit_aa: Sequence[str]) -> list[str]:
  letters: list[str] = []
  for token in omit_aa:
    letters.extend(list(token))
  return letters


def _aa_index(letter: str) -> int:
  try:
    return MPNN_ALPHABET.index(letter)
  except ValueError as exc:
    msg = f"omit amino acid {letter!r} is not in the MPNN alphabet {MPNN_ALPHABET!r}"
    raise ValueError(msg) from exc


def _broadcast_bias(bias: Any, seq_len: int) -> Any:  # noqa: ANN401
  if bias is None:
    return jnp.zeros((seq_len, _ALPHABET_SIZE), dtype=jnp.float32)
  arr = jnp.asarray(bias, dtype=jnp.float32)
  if arr.ndim == 1 and arr.shape[0] == _ALPHABET_SIZE:
    return jnp.broadcast_to(arr, (seq_len, _ALPHABET_SIZE))
  if arr.ndim == 1 and arr.shape[0] == seq_len:
    return arr[:, None]
  return arr
