# ruff: noqa: S101
"""Field-level parity against the sealed A0 oracle, when the orchestrator has written it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aminx.families.potts_mpnn.featurize import (
  ARRAY_FIELDS,
  JSON_FIELDS,
  parse_pdb_upstream,
  tied_featurize_port,
)

_ORACLE = (
  Path(__file__).resolve().parents[3]
  / "tests"
  / "port"
  / "reference"
  / "a0_featurize"
  / "oracle.npz"
)


def _json_value(features: object, field: str) -> object:
  value = getattr(features, field)
  if field == "tied_pos":
    return [list(group) for group in value]
  if field == "name":
    return value
  return list(value)


def test_featurize_matches_upstream_oracle(tmp_path: Path) -> None:
  """Compare every A0 field. ``E_idx`` is not part of this gate."""
  if not _ORACLE.is_file():
    pytest.skip(
      "tests/port/reference/a0_featurize/oracle.npz is absent; "
      "the orchestrator generates it on titanix",
    )
  oracle = np.load(_ORACLE)
  names = [str(name) for name in oracle["fixture_names"].tolist()]
  assert names, "oracle.npz has no fixtures"
  assert "manifest" in oracle
  for name in names:
    stored_name = json.loads(str(oracle[f"{name}__name"]))
    # ``name`` is the PDB stem (``parse_PDB`` strips the ``.pdb`` suffix), so the
    # temp file has to use that stem rather than the fixture key.
    pdb_path = tmp_path / f"{stored_name}.pdb"
    pdb_path.write_text(str(oracle[f"{name}__pdb"]))
    skip_gaps = bool(oracle[f"{name}__skip_gaps"])
    parsed = parse_pdb_upstream(pdb_path, None, skip_gaps)
    features = tied_featurize_port(parsed, None)[0]
    assert features.L_total == int(oracle[f"{name}__L_total"])
    for field in ARRAY_FIELDS:
      got = getattr(features, field)
      expected = oracle[f"{name}__{field}"]
      assert got.shape == expected.shape, (name, field, got.shape, expected.shape)
      assert got.dtype == expected.dtype, (name, field, got.dtype, expected.dtype)
      assert np.array_equal(got, expected), (name, field)
    for field in JSON_FIELDS:
      got = _json_value(features, field)
      expected = json.loads(str(oracle[f"{name}__{field}"]))
      assert got == expected, (name, field)
