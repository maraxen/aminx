# ruff: noqa: S101
"""Tests for the PottsAlphabet descriptor and its shipped POTTS_MPNN instance."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn import decode, etab, featurize, potts_head
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
    PottsAlphabet(name="", symbols=("A", "X"), x_index=1, pair_side=1, etab_symbols=("A", "X"), standard_symbols=("A",))


def test_duplicate_symbol_rejected() -> None:
  """Repeated model symbols raise ValueError."""
  with pytest.raises(ValueError, match="unique"):
    PottsAlphabet(name="dup", symbols=("A", "A", "X"), x_index=2, pair_side=2, etab_symbols=("A", "X"), standard_symbols=("A",))


def test_x_index_out_of_range_rejected() -> None:
  """x_index outside [0, size) raises ValueError at both ends."""
  with pytest.raises(ValueError, match="x_index"):
    PottsAlphabet(name="xhi", symbols=("A", "X"), x_index=2, pair_side=1, etab_symbols=("A", "X"), standard_symbols=("A",))
  with pytest.raises(ValueError, match="x_index"):
    PottsAlphabet(name="xlo", symbols=("A", "X"), x_index=-1, pair_side=1, etab_symbols=("A", "X"), standard_symbols=("A",))


def test_pair_side_zero_rejected() -> None:
  """pair_side of zero raises ValueError."""
  with pytest.raises(ValueError, match="pair_side"):
    PottsAlphabet(name="p0", symbols=("A", "X"), x_index=1, pair_side=0, etab_symbols=("A", "X"), standard_symbols=("A",))


def test_pair_side_above_size_rejected() -> None:
  """pair_side larger than size raises ValueError."""
  with pytest.raises(ValueError, match="pair_side"):
    PottsAlphabet(name="pbig", symbols=("A", "X"), x_index=1, pair_side=3, etab_symbols=("A", "X"), standard_symbols=("A",))


def test_duplicate_etab_symbol_rejected() -> None:
  """Repeated etab symbols raise ValueError."""
  with pytest.raises(ValueError, match="etab_symbols"):
    PottsAlphabet(name="etabdup", symbols=("A", "X"), x_index=1, pair_side=1, etab_symbols=("A", "A"), standard_symbols=("A",))


def test_multi_character_tokens_construct() -> None:
  """Multi-character tokens are accepted; no single-char symbol assumption."""
  alphabet = PottsAlphabet(
    name="multichar",
    symbols=("A", "HIS-P", "X"),
    x_index=2,
    pair_side=3,
    etab_symbols=("A", "HIS-P", "X"),
    standard_symbols=("A",),
  )
  assert alphabet.size == 3
  assert alphabet.pair_dim == 9
  assert alphabet.symbols[1] == "HIS-P"


def test_featurize_alphabet_reads_potts_alphabet() -> None:
  """featurize.MODEL_ALPHABET and X_INDEX come from POTTS_MPNN with the same types."""
  assert featurize.MODEL_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"
  assert isinstance(featurize.MODEL_ALPHABET, str)
  assert featurize.X_INDEX == 20
  assert isinstance(featurize.X_INDEX, int)


def test_etab_alphabet_reads_potts_alphabet() -> None:
  """etab.ETAB_ALPHABET and the remap tables keep their shipped values and dtype."""
  assert etab.ETAB_ALPHABET == "ACDEFGHIKLMNPQRSTVWY-X"
  assert np.asarray(etab._MODEL_TO_ETAB).tolist() == [*range(20), 21]
  assert np.asarray(etab._MODEL_TO_ETAB).dtype == np.int32
  assert np.asarray(etab._ETAB_TO_MODEL).tolist() == [*range(20), -1, 20]
  assert np.asarray(etab._ETAB_TO_MODEL).dtype == np.int32


def test_mask_refine_x_default_masks_index_20() -> None:
  """With the default x_index, only model X (index 20) receives the -1e8 offset."""
  masked = np.asarray(decode.mask_refine_x(jnp.zeros(21)))
  assert masked[20] == pytest.approx(-decode._OMIT_SCALE)
  assert np.all(masked[:20] == 0.0)


def test_mask_refine_x_explicit_index() -> None:
  """An explicit x_index masks only that slot on a length-5 vector."""
  masked = np.asarray(decode.mask_refine_x(jnp.zeros(5), x_index=3))
  assert masked[3] == pytest.approx(-decode._OMIT_SCALE)
  assert np.all(np.delete(masked, 3) == 0.0)


def test_parser_gap_index_is_upstream_gap() -> None:
  """The literal 20s in the parser index upstream's _ALPHA_3, where 20 is GAP."""
  assert featurize._AA_3_N["GAP"] == 20
  assert featurize._ALPHA_3[20] == "GAP"
