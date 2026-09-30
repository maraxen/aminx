# ruff: noqa: S101
"""Field-level parity against the sealed B0 oracle, when the orchestrator has written it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.featurize import ARRAY_FIELDS, JSON_FIELDS, featurize

_ROOT = Path(__file__).resolve().parents[3] / "tests" / "port" / "reference" / "b0_featurize"
_ORACLE = _ROOT / "oracle.npz"
_SHA = _ROOT / "oracle.npz.sha256"
# Measured vs upstream (titanix 260929): coords <= 1.5e-5 A (2 f32 ULP), phi/psi and ligand
# coords exact, chi <= 3.4e-4 deg (f32 dihedral noise). chi alone gets the looser bound.
_FLOAT_ATOL = 1e-4
_FIELD_ATOL = {"chi_angles": 1e-3}
_GRAPH_FIELDS = (
  "pr_pr_edge_index",
  "pr_pr_edge_distance",
  "lig_pr_edge_index",
  "lig_pr_edge_distance",
  "lig_lig_edge_index",
  "lig_lig_edge_distance",
  "ligand_nodes",
  "ligand_coords",
  "ligand_atomic_number_indices",
  "first_shell_ligand_contact_mask",
)


def test_featurize_matches_upstream_oracle(tmp_path: Path) -> None:
  """Compare B0 fields. Exposure is stochastic and is not in this gate."""
  if not _ORACLE.is_file():
    pytest.skip(
      "tests/port/reference/b0_featurize/oracle.npz is absent; "
      "the orchestrator generates it on titanix",
    )
  digest = hashlib.sha256(_ORACLE.read_bytes()).hexdigest()
  expected = _SHA.read_text().strip()
  assert digest == expected, f"oracle.npz sha256 {digest} != {expected}"
  oracle = np.load(_ORACLE)
  names = [str(name) for name in oracle["fixture_names"].tolist()]
  assert names, "oracle.npz has no fixtures"
  assert len(names) == 5
  for name in names:
    for field in _GRAPH_FIELDS:
      drop0 = oracle[f"{name}__drop0__{field}"]
      drop6 = oracle[f"{name}__drop6__{field}"]
      assert drop0.shape == drop6.shape, (name, field)
      assert drop0.dtype == drop6.dtype, (name, field)
      assert np.array_equal(drop0, drop6), (name, field)
    pdb_path = tmp_path / f"{name}.pdb"
    pdb_path.write_text(str(oracle[f"{name}__pdb"]))
    features = featurize(pdb_path)
    for field in ARRAY_FIELDS:
      got = getattr(features, field)
      ref = oracle[f"{name}__{field}"]
      assert got.shape == ref.shape, (name, field, got.shape, ref.shape)
      assert got.dtype == ref.dtype, (name, field, got.dtype, ref.dtype)
      if np.issubdtype(got.dtype, np.floating):
        atol = _FIELD_ATOL.get(field, _FLOAT_ATOL)
        assert np.allclose(got, ref, rtol=0.0, atol=atol, equal_nan=True), (name, field)
      else:
        assert np.array_equal(got, ref), (name, field)
    for field in JSON_FIELDS:
      if field == "pdb_code":
        continue
      got = getattr(features, field)
      ref = json.loads(str(oracle[f"{name}__{field}"]))
      assert list(got) == ref, (name, field)
