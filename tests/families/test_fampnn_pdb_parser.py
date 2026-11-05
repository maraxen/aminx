"""aminx's FAMPNN PDB reader against upstream's loader on a stress input (altlocs, insertion codes, HETATM, X residues).

The reference is ``tests/data/fampnn_parser_oracle.npz``: the output of the real ``fampnn.data.data.load_feats_from_pdb``
+ ``process_single_pdb`` (Biopython underneath) on ``tests/data/fampnn_parser_stress.pdb``, produced by
``scripts/parity/dump_fampnn_parser_oracle.py`` on titanix (261105, upstream aaf788b). The input is derived from
``tests/data/1ubq.pdb`` by ``fampnn_fixtures.make_parser_stress_pdb``; the test regenerates it and checks its hash.
Numpy only: no torch, no JAX.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aminx.families.fampnn.pdb import read_fampnn_pdb

ROOT = Path(__file__).resolve().parents[2]
STRESS = ROOT / "tests/data/fampnn_parser_stress.pdb"
ORACLE = ROOT / "tests/data/fampnn_parser_oracle.npz"
STRESS_SHA256 = "f916ce38c30a4d376cca6e274c02cef31d83e8dc18b593abb2f41e091e3d9125"


@pytest.fixture(scope="module")
def oracle() -> dict[str, np.ndarray]:
  z = np.load(ORACLE)
  return {k: z[k] for k in z.files}


def test_stress_input_is_the_one_the_oracle_saw(oracle: dict[str, np.ndarray], tmp_path: Path) -> None:
  assert hashlib.sha256(STRESS.read_bytes()).hexdigest() == STRESS_SHA256
  assert bytes(oracle["pdb_sha256"]).decode() == STRESS_SHA256
  sys.path.insert(0, str(ROOT / "scripts" / "parity"))
  try:
    from fampnn_fixtures import make_parser_stress_pdb
  finally:
    sys.path.pop(0)
  assert make_parser_stress_pdb(ROOT / "tests/data/1ubq.pdb", tmp_path / "regen.pdb") == STRESS_SHA256


def test_reader_matches_upstream_loader(oracle: dict[str, np.ndarray]) -> None:
  st = read_fampnn_pdb(STRESS)
  np.testing.assert_array_equal(st.aatype, oracle["aatype"])
  np.testing.assert_array_equal(st.residue_index, oracle["residue_index"])
  np.testing.assert_array_equal(st.chain_index, oracle["chain_index"])
  np.testing.assert_array_equal(st.atom37_mask, oracle["atom_mask"])
  np.testing.assert_array_equal(st.missing_atom_mask, oracle["missing_atom_mask"])
  # Biopython stores coordinates as float32; the text values round identically
  np.testing.assert_array_equal(st.atom37_xyz.astype(np.float32), oracle["x"].astype(np.float32))
  assert st.chain_id_mapping == json.loads(bytes(oracle["chain_id_mapping_json"]).decode())


def test_stress_input_really_exercises_each_case(oracle: dict[str, np.ndarray]) -> None:
  """Guard against a fixture that silently stopped covering the behaviours it exists for."""
  ridx, aatype = oracle["residue_index"], oracle["aatype"]
  assert ridx[0] == 1 and aatype[0] == 20  # MSE (HETATM, mapped) kept and typed X
  assert (np.diff(ridx) > 1).any()  # insertion-code offsets open a numbering jump (35A/35B)
  assert 20 in aatype[40:60]  # residue 50 renamed ABC: unknown ATOM residue typed X
  assert oracle["x"].shape[0] == 76  # GOL (unmapped HETATM) and HOH are skipped, nothing else is
  # residue 7 (index 6): upstream keeps the *first* altloc in file order (occupancy 0.40), not the 0.60 one
  first_cb = float(next(line for line in STRESS.read_text().splitlines() if " CB ATHR A   7" in line)[30:38])
  assert float(oracle["x"][6, 3, 0]) == pytest.approx(first_cb, abs=1e-3)
