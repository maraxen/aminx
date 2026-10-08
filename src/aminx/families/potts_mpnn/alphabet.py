"""Potts alphabet descriptor: the single source of the Potts alphabet width.

``PottsAlphabet`` names one vocabulary together with the shape of the Potts pair
table that the head produces over it. Every consumer that needs the alphabet
width (the pair-table side, the flattened pair dimension, the energy-table
alphabet) reads it from here rather than repeating a literal ``20``.

The ``name`` field is required on purpose: two vocabularies can share a width
and collide index-for-index, so a code path must name the alphabet it uses and
must never infer it from the number of symbols.

This is plain ``src`` code. The ``alphex`` library is a dev-only dependency of
this repository and is not importable from ``src``, so nothing here depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PottsAlphabet:
  """Vocabulary and Potts pair-table layout for one model family.

  Attributes:
    name: Explicit identifier for the alphabet. Required and non-empty.
    symbols: Model-index order. Each entry is one token; a token may span
      several characters in future alphabets (e.g. ``"HIS-P"``).
    x_index: Model index of the unknown token ``X``.
    pair_side: Side length of the pair table the head produces. The table is
      ``pair_side x pair_side``, which need not equal ``size``.
    etab_symbols: Symbols of the energy-table alphabet, in etab index order.
  """

  name: str
  symbols: tuple[str, ...]
  x_index: int
  pair_side: int
  etab_symbols: tuple[str, ...]

  def __post_init__(self) -> None:
    """Reject inconsistent alphabet descriptions at construction time."""
    if not self.name:
      msg = "PottsAlphabet.name must be a non-empty string"
      raise ValueError(msg)
    if len(set(self.symbols)) != len(self.symbols):
      msg = f"PottsAlphabet {self.name!r}: symbols must be unique, got {self.symbols!r}"
      raise ValueError(msg)
    if not 0 <= self.x_index < self.size:
      msg = f"PottsAlphabet {self.name!r}: x_index={self.x_index} is outside [0, {self.size})"
      raise ValueError(msg)
    if not 0 < self.pair_side <= self.size:
      msg = (
        f"PottsAlphabet {self.name!r}: pair_side={self.pair_side} must satisfy "
        f"0 < pair_side <= size ({self.size})"
      )
      raise ValueError(msg)
    if len(set(self.etab_symbols)) != len(self.etab_symbols):
      msg = f"PottsAlphabet {self.name!r}: etab_symbols must be unique, got {self.etab_symbols!r}"
      raise ValueError(msg)

  @property
  def size(self) -> int:
    """Number of model tokens, ``len(symbols)``."""
    return len(self.symbols)

  @property
  def pair_dim(self) -> int:
    """Flattened pair-table width, ``pair_side * pair_side``."""
    return self.pair_side * self.pair_side


# Shipped PottsMPNN: 20 standard letters, then X. The etab alphabet adds "-" before X.
POTTS_MPNN = PottsAlphabet(
  name="pottsmpnn_21",
  symbols=tuple("ACDEFGHIKLMNPQRSTVWYX"),
  x_index=20,
  pair_side=20,
  etab_symbols=tuple("ACDEFGHIKLMNPQRSTVWY-X"),
)
