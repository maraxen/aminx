# ruff: noqa: S101
"""Tests for the PottsAlphabet descriptor and its shipped POTTS_MPNN instance."""

from __future__ import annotations

import pytest

from aminx.families.potts_mpnn import etab, potts_head
from aminx.families.potts_mpnn.alphabet import POTTS_MPNN, PottsAlphabet
from aminx.families.potts_mpnn.featurize import MODEL_ALPHABET


def test_potts_mpnn_reproduces_old_constants() -> None:
  """POTTS_MPNN reproduces every hardcoded constant it replaced."""
  assert etab.N_AA == 20
  assert potts_head.N_AA == 20
  assert potts_head.PAIR_DIM == 400
  assert etab.N_ETAB == 22
  assert etab.ETAB_GAP == 20
  assert etab.ETAB_X == 21
  assert etab.ETAB_ALPHABET == "ACDEFGHIKLMNPQRSTVWY-X"
  assert "".join(POTTS_MPNN.symbols) == MODEL_ALPHABET
  assert POTTS_MPNN.x_index == MODEL_ALPHABET.index("X")
  assert POTTS_MPNN.size == 21
  assert POTTS_MPNN.pair_dim == 400
  assert "".join(POTTS_MPNN.etab_symbols) == etab.ETAB_ALPHABET


def test_pair_side_is_single_source_for_etab_and_head() -> None:
  """etab.N_AA and potts_head.N_AA both come from POTTS_MPNN.pair_side."""
  assert etab.N_AA == POTTS_MPNN.pair_side
  assert potts_head.N_AA == POTTS_MPNN.pair_side
  assert etab.N_AA == potts_head.N_AA
  assert potts_head.PAIR_DIM == POTTS_MPNN.pair_dim


def test_empty_name_rejected() -> None:
  """An empty alphabet name raises ValueError."""
  with pytest.raises(ValueError, match="name"):
    PottsAlphabet(name="", symbols=("A", "X"), x_index=1, pair_side=1, etab_symbols=("A", "X"))


def test_duplicate_symbol_rejected() -> None:
  """Repeated model symbols raise ValueError."""
  with pytest.raises(ValueError, match="unique"):
    PottsAlphabet(name="dup", symbols=("A", "A", "X"), x_index=2, pair_side=2, etab_symbols=("A", "X"))


def test_x_index_out_of_range_rejected() -> None:
  """x_index outside [0, size) raises ValueError at both ends."""
  with pytest.raises(ValueError, match="x_index"):
    PottsAlphabet(name="xhi", symbols=("A", "X"), x_index=2, pair_side=1, etab_symbols=("A", "X"))
  with pytest.raises(ValueError, match="x_index"):
    PottsAlphabet(name="xlo", symbols=("A", "X"), x_index=-1, pair_side=1, etab_symbols=("A", "X"))


def test_pair_side_zero_rejected() -> None:
  """pair_side of zero raises ValueError."""
  with pytest.raises(ValueError, match="pair_side"):
    PottsAlphabet(name="p0", symbols=("A", "X"), x_index=1, pair_side=0, etab_symbols=("A", "X"))


def test_pair_side_above_size_rejected() -> None:
  """pair_side larger than size raises ValueError."""
  with pytest.raises(ValueError, match="pair_side"):
    PottsAlphabet(name="pbig", symbols=("A", "X"), x_index=1, pair_side=3, etab_symbols=("A", "X"))


def test_duplicate_etab_symbol_rejected() -> None:
  """Repeated etab symbols raise ValueError."""
  with pytest.raises(ValueError, match="etab_symbols"):
    PottsAlphabet(name="etabdup", symbols=("A", "X"), x_index=1, pair_side=1, etab_symbols=("A", "A"))


def test_multi_character_tokens_construct() -> None:
  """Multi-character tokens are accepted; no single-char symbol assumption."""
  alphabet = PottsAlphabet(
    name="multichar",
    symbols=("A", "HIS-P", "X"),
    x_index=2,
    pair_side=3,
    etab_symbols=("A", "HIS-P", "X"),
  )
  assert alphabet.size == 3
  assert alphabet.pair_dim == 9
  assert alphabet.symbols[1] == "HIS-P"
