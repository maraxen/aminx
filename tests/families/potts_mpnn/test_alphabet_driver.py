"""P7 S9-S12: the sampler, DMS targets and fallback seam read the PottsAlphabet."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aminx.families.potts_mpnn import driver as driver_mod
from aminx.families.potts_mpnn import sample_host
from aminx.families.potts_mpnn.alphabet import POTTS_MPNN, PottsAlphabet
from aminx.run.options import PottsMPNNOptions


def test_standard_symbols_must_be_subset() -> None:
  """A standard residue that is not a model token is rejected."""
  with pytest.raises(ValueError, match="standard_symbols"):
    PottsAlphabet(
      name="bad",
      symbols=("A", "X"),
      x_index=1,
      pair_side=1,
      etab_symbols=("A", "X"),
      standard_symbols=("C",),
    )


def test_standard_symbols_must_be_non_empty() -> None:
  with pytest.raises(ValueError, match="standard_symbols"):
    PottsAlphabet(
      name="empty",
      symbols=("A", "X"),
      x_index=1,
      pair_side=1,
      etab_symbols=("A", "X"),
      standard_symbols=(),
    )


def test_shipped_constants_unchanged() -> None:
  """S9/S10: the derived constants equal the literals they replaced."""
  assert sample_host._ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"
  assert driver_mod._CANONICAL == "ACDEFGHIKLMNPQRSTVWY"


def test_etab_sink_dims_unchanged() -> None:
  """S12: the emitted etab keeps its (L_total, K, 20, 20) dims at V=21."""
  spec = SimpleNamespace(potts_mpnn=PottsMPNNOptions(emit_etab=True))
  schema = sample_host.sample_schema(spec)
  assert schema["potts_etab"].dims == ("L_total", "K", "20", "20")


def test_mpnn_core_returns_core_for_shipped_alphabet() -> None:
  core = object()
  model = SimpleNamespace(alphabet=POTTS_MPNN, mpnn=core)
  assert driver_mod.PottsMPNNDriver().mpnn_core(model) is core


def test_mpnn_core_refuses_other_alphabet() -> None:
  """S11: fallback purposes refuse a non-21-token model instead of misreading it."""
  wide = PottsAlphabet(
    name="wide3",
    symbols=("A", "C", "X"),
    x_index=2,
    pair_side=3,
    etab_symbols=("A", "C", "X"),
    standard_symbols=("A", "C"),
  )
  model = SimpleNamespace(alphabet=wide, mpnn=object())
  with pytest.raises(ValueError, match="wide3"):
    driver_mod.PottsMPNNDriver().mpnn_core(model)
