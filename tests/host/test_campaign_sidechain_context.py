"""A campaign sidechain row must state the model-level flag it depends on (#115).

`sidechain_conditioning=True` builds the atom_37 tensors; `ligand_mpnn_use_side_chain_context`
is what builds the model branch that consumes them. `prep.py` now implies the second from the
first at load time, but the campaign planner -- which owns the sidechain axis -- never wrote the
flag into the row, so a manifest row said "sidechain on" and left the model half of the
capability to an implication two modules away.

The planner now derives it per sidechain-on LigandMPNN row and never overrides an explicit
caller value. There is deliberately no campaign CLI flag: the 2x2 grid always contains both a
sidechain-on and a sidechain-off row, so any explicit campaign-wide value is wrong for half of it.

task_id: 260930_resolve-issues-debt
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aminx.host.campaign import plan_campaign_manifest
from aminx.host.prep import prep_protein_stream_and_model
from aminx.run.specs import SamplingSpecification

_LIGAND = "ligandmpnn_v_32_020_25"
_PROTEIN = "proteinmpnn_v_48_020"


def _rows(checkpoint: str = _LIGAND, **spec_kwargs: Any) -> list[dict[str, Any]]:
  return plan_campaign_manifest(
    base_spec=SamplingSpecification(inputs=["ref.pdb"], checkpoint_id=checkpoint, **spec_kwargs),
    campaign_id="sc",
    output_root="/tmp/sc",
    designs_per_library_type=1,
    samples_chunk_size=1,
  )


def _flag_by_sidechain(rows: list[dict[str, Any]]) -> dict[bool, set[Any]]:
  out: dict[bool, set[Any]] = {}
  for row in rows:
    out.setdefault(bool(row["sampling_spec"]["sidechain_conditioning"]), set()).add(
      row["sampling_spec"]["ligand_mpnn_use_side_chain_context"],
    )
  return out


def test_sidechain_on_rows_state_the_model_flag() -> None:
  """THE gap: a sidechain-on row carries the model-level flag in its own sampling_spec."""
  flags = _flag_by_sidechain(_rows())
  assert flags[True] == {True}
  assert flags[False] == {None}, "sidechain-off rows must be left as they were"


def test_explicit_caller_value_is_never_overridden() -> None:
  assert _flag_by_sidechain(_rows(ligand_mpnn_use_side_chain_context=True)) == {
    True: {True},
    False: {True},
  }
  assert _flag_by_sidechain(_rows(ligand_mpnn_use_side_chain_context=False)) == {
    True: {False},
    False: {False},
  }


def test_protein_family_rows_are_untouched() -> None:
  """The flag only matters on the LigandMPNN skeleton; don't rewrite other families' rows."""
  flags = _flag_by_sidechain(_rows(_PROTEIN))
  assert flags == {True: {None}, False: {None}}


def test_derivation_does_not_move_any_row_hash() -> None:
  """The model flag is not a hash input, so no existing done-marker is invalidated."""
  hashes = [r["manifest_row_hash"] for r in _rows()]
  explicit = [r["manifest_row_hash"] for r in _rows(ligand_mpnn_use_side_chain_context=True)]
  assert hashes == explicit


@pytest.mark.parametrize("sidechain_on", [False, True])
def test_row_reaches_load_model_with_the_matching_flag(sidechain_on: bool) -> None:
  """Plan -> manifest JSON -> worker spec -> prep -> load_model(use_side_chain_context=...)."""
  (row,) = [
    r for r in _rows()
    if r["sampling_spec"]["sidechain_conditioning"] is sidechain_on
    and r["sampling_spec"]["ligand_conditioning"] is False
  ]
  spec = SamplingSpecification(**json.loads(json.dumps(row["sampling_spec"])))
  with patch("aminx.host.prep.create_protein_dataset") as dataset, \
       patch("aminx.host.prep.load_model") as load_model:
    dataset.return_value = MagicMock()
    load_model.return_value = MagicMock()
    try:
      prep_protein_stream_and_model(spec)
    except Exception:  # noqa: BLE001 - downstream mocks raise; only the load_model call matters
      pass
  assert load_model.called
  assert load_model.call_args.kwargs["use_side_chain_context"] is sidechain_on
