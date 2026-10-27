"""PDB-free unit tests for browser-validation fixture canonical parsing, targeted refresh, and stale-manifest guard."""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import aminx.io.parsing as aminx_parsing
import scripts.browser_validation.fixtures as fixtures
import scripts.browser_validation.layer_a_exact as lae
import scripts.browser_validation.layer_a_sampling as las
import scripts.browser_validation.parse_parity as parse_parity

if TYPE_CHECKING:
  from pathlib import Path


def _synthetic_manifest() -> dict[str, Any]:
  return {
    "created": "20260925",
    "fixtures": [
      {
        "L": 113,
        "kind": "protein",
        "ligand_heavy_atoms": 0,
        "n_chains": 3,
        "name": "1BC8",
        "near_tie_residues": {"32": [], "48": []},
        "path": "$REFERENCE_PATH/inputs/1BC8.pdb",
        "set": "B",
        "sha256": "d3e844d818f24e8a1b6a39f7d97816306f0b69433ec8b9ea2cb6af6ba08e1284",
        "tie_groups_knn_disjoint": [],
        "tie_groups_rejected": [{"members": [0, 10], "offending_pairs": [{"k": 48, "pair": [0, 10]}]}],
      },
      {
        "L": 3464,
        "kind": "protein",
        "ligand_heavy_atoms": 0,
        "n_chains": 16,
        "name": "2GFB",
        "near_tie_residues": {"32": [600], "48": [586]},
        "path": "$REFERENCE_PATH/inputs/2GFB.pdb",
        "set": "A",
        "sha256": "137fa4c5f11b93736ae16069038309bb33c414532c50a691c1093e3d84d0e060",
        "tie_groups_knn_disjoint": [
          {"designable_members": [0, 3463], "members": [0, 3463]},
        ],
        "tie_groups_rejected": [],
      },
      {
        "L": 416,
        "kind": "protein",
        "ligand_heavy_atoms": 0,
        "n_chains": 3,
        "name": "3HTN",
        "near_tie_residues": {"32": [], "48": []},
        "path": "$PROTEINMPNN_PATH/inputs/PDB_complexes/pdbs/3HTN.pdb",
        "set": "B",
        "sha256": "aaaaaaaa11111111222222223333333344444444555555556666666677777777",
        "tie_groups_knn_disjoint": [
          {"designable_members": [0, 276], "members": [0, 276]},
        ],
        "tie_groups_rejected": [],
      },
    ],
    "k_values": [48, 32],
    "min_protein_l": 64,
    "near_tie_gap_angstrom": 0.0001,
    "root_tokens": {
      "$PROTEINMPNN_PATH": "/fake/proteinmpnn",
      "$REFERENCE_PATH": "/fake/reference",
      "$WT": "/fake/wt",
    },
  }


def test_parse_fixture_uses_canonical_copy_and_pins_original_sha256(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """_parse_fixture parses parse_parity._canonical_copy output while computing sha256 on raw file."""
  raw_path = tmp_path / "3HTN.pdb"
  raw_path.write_bytes(b"RAW_SOURCE_PDB_BYTES")
  canonical_path = tmp_path / "canonical_3HTN.pdb"
  canonical_path.write_bytes(b"CANONICAL_COPY_BYTES")

  canonical_calls: list[tuple[Path, Path]] = []

  def fake_canonical_copy(path: Path, cache_dir: Path) -> Path:
    canonical_calls.append((path, cache_dir))
    return canonical_path

  parsed_paths: list[str] = []
  fake_protein = SimpleNamespace(
    aatype=np.array([0, 1, 2, 3], dtype=np.int32),
    mask=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
    coordinates=np.ones((4, 37, 3), dtype=np.float32),
    molecule_type=np.array([0, 0, 1, 1, 1], dtype=np.int32),
  )

  def fake_parse_structure(path_str: str) -> Any:
    parsed_paths.append(path_str)
    return fake_protein

  monkeypatch.setattr(parse_parity, "_canonical_copy", fake_canonical_copy)
  monkeypatch.setattr(aminx_parsing, "parse_structure", fake_parse_structure)

  out = fixtures._parse_fixture(raw_path)  # noqa: SLF001

  assert len(canonical_calls) == 1  # noqa: S101
  assert canonical_calls[0][0] == raw_path  # noqa: S101
  assert parsed_paths == [str(canonical_path)]  # noqa: S101
  assert out["path"] == raw_path  # noqa: S101
  assert out["sha256"] == fixtures.sha256_file(raw_path)  # noqa: S101
  assert out["sha256"] != fixtures.sha256_file(canonical_path)  # noqa: S101
  assert out["L"] == 4  # noqa: S101
  assert out["n_chains"] == 2  # noqa: S101
  assert out["ligand_heavy_atoms"] == 3  # noqa: S101
  assert np.array_equal(out["aatype"], fake_protein.aatype)  # noqa: S101
  assert np.array_equal(out["chain_index"], fake_protein.chain_index)  # noqa: S101
  assert np.array_equal(out["mask"], fake_protein.mask.astype(np.float64))  # noqa: S101
  assert np.array_equal(out["coordinates"], fake_protein.coordinates)  # noqa: S101


def test_refresh_mode_rewrites_only_named_entries_atomically(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """--refresh updates only named entries, preserves untouched entries and top-level keys, and writes via os.replace."""
  manifest_path = tmp_path / "manifest.json"
  initial = _synthetic_manifest()
  initial_text = json.dumps(initial, indent=2, sort_keys=True) + "\n"
  manifest_path.write_text(initial_text, encoding="utf-8")

  called_for: list[str] = []

  def fake_fixture_record(path: Path) -> dict[str, Any]:
    called_for.append(path.stem)
    return {
      "name": path.stem,
      "kind": "protein",
      "path": "$WT/overwritten_if_buggy.pdb",
      "set": "A",
      "sha256": "deadbeef",
      "L": 425,
      "n_chains": 3,
      "ligand_heavy_atoms": 9,
      "near_tie_residues": {"32": [14], "48": [14, 99]},
      "tie_groups_knn_disjoint": [
        {"designable_members": [0, 282], "members": [0, 282]},
      ],
      "tie_groups_rejected": [
        {"members": [10, 153], "offending_pairs": [{"k": 48, "pair": [10, 153]}]},
      ],
    }

  replace_calls: list[tuple[Path, Path]] = []
  real_replace = os.replace

  def spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    replace_calls.append((Path(src), Path(dst)))
    real_replace(src, dst)

  monkeypatch.setattr(fixtures, "_fixture_record", fake_fixture_record)
  monkeypatch.setattr(os, "replace", spy_replace)

  rc = fixtures.main(["--refresh", "3HTN", "--manifest", str(manifest_path)])
  assert rc == 0  # noqa: S101
  assert called_for == ["3HTN"]  # noqa: S101
  assert len(replace_calls) == 1  # noqa: S101
  assert replace_calls[0][1] == manifest_path  # noqa: S101
  assert not replace_calls[0][0].exists()  # noqa: S101

  after_text = manifest_path.read_text(encoding="utf-8")
  after = json.loads(after_text)

  for top_key in ("created", "k_values", "min_protein_l", "near_tie_gap_angstrom", "root_tokens"):
    assert after[top_key] == initial[top_key]  # noqa: S101

  assert after["fixtures"][0] == initial["fixtures"][0]  # noqa: S101
  assert after["fixtures"][1] == initial["fixtures"][1]  # noqa: S101

  refreshed_3htn = after["fixtures"][2]
  assert refreshed_3htn["name"] == "3HTN"  # noqa: S101
  assert refreshed_3htn["set"] == "B"  # noqa: S101
  assert refreshed_3htn["path"] == "$PROTEINMPNN_PATH/inputs/PDB_complexes/pdbs/3HTN.pdb"  # noqa: S101
  assert refreshed_3htn["sha256"] == initial["fixtures"][2]["sha256"]  # noqa: S101
  assert refreshed_3htn["L"] == 425  # noqa: S101
  assert refreshed_3htn["n_chains"] == 3  # noqa: S101
  assert refreshed_3htn["ligand_heavy_atoms"] == 9  # noqa: S101
  assert refreshed_3htn["near_tie_residues"] == {"32": [14], "48": [14, 99]}  # noqa: S101
  assert refreshed_3htn["tie_groups_knn_disjoint"] == [  # noqa: S101
    {"designable_members": [0, 282], "members": [0, 282]}
  ]
  assert refreshed_3htn["tie_groups_rejected"] == [  # noqa: S101
    {"members": [10, 153], "offending_pairs": [{"k": 48, "pair": [10, 153]}]}
  ]

  expected_manifest = copy.deepcopy(initial)
  expected_manifest["fixtures"][2] = refreshed_3htn
  expected_bytes = (json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
  assert manifest_path.read_bytes() == expected_bytes  # noqa: S101


def test_stale_manifest_guard_fires_on_length_mismatch_before_knn(
  capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Stale guard exits 2 with fixture name, both lengths, and refresh command before _reverify_knn_disjoint."""
  fixture: dict[str, Any] = {
    "name": "3HTN",
    "L": 416,
    "tie_groups_knn_disjoint": [
      {"designable_members": [0, 276], "members": [0, 276]},
    ],
  }
  batch = SimpleNamespace(fixture_name="3HTN", length=425, manifest_length=416)

  def forbidden_knn(*_args: Any, **_kwargs: Any) -> Any:
    msg = "_reverify_knn_disjoint must not run when manifest L mismatches parsed length"
    raise AssertionError(msg)

  monkeypatch.setattr(lae, "_reverify_knn_disjoint", forbidden_knn)

  with pytest.raises(SystemExit) as exc_info:
    lae.row_p09(fixture, batch, (None, None, None, None), "eqx")  # type: ignore[arg-type]
  assert exc_info.value.code == 2  # noqa: S101
  err = capsys.readouterr().err
  assert "3HTN" in err  # noqa: S101
  assert "425" in err  # noqa: S101
  assert "416" in err  # noqa: S101
  assert "fixtures.py --refresh 3HTN" in err  # noqa: S101

  monkeypatch.setattr(lae, "build_exact_batch", lambda _f, _du: batch)
  with pytest.raises(SystemExit) as exc_sampling:
    las.build_lane_batch(fixture, "P09-s@1.0", None)
  assert exc_sampling.value.code == 2  # noqa: S101
  err_sampling = capsys.readouterr().err
  assert "3HTN" in err_sampling  # noqa: S101
  assert "425" in err_sampling  # noqa: S101
  assert "416" in err_sampling  # noqa: S101
  assert "fixtures.py --refresh 3HTN" in err_sampling  # noqa: S101


def test_stale_manifest_guard_fires_on_out_of_range_member_before_knn(
  capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Stale guard exits 2 when any tie-group member >= parsed length before k-NN check runs."""
  fixture: dict[str, Any] = {
    "name": "2GFB",
    "L": 3416,
    "tie_groups_knn_disjoint": [
      {"designable_members": [0, 3463], "members": [0, 3463]},
    ],
  }
  batch = SimpleNamespace(fixture_name="2GFB", length=3416, manifest_length=3416)

  def forbidden_neighbors(*_args: Any, **_kwargs: Any) -> Any:
    msg = "_neighbor_indices must not run when a tie-group member is >= length"
    raise AssertionError(msg)

  monkeypatch.setattr(fixtures, "_neighbor_indices", forbidden_neighbors)

  with pytest.raises(SystemExit) as exc_info:
    lae._reverify_knn_disjoint(batch, [[0, 3463]], fixture)  # type: ignore[arg-type]  # noqa: SLF001
  assert exc_info.value.code == 2  # noqa: S101
  err = capsys.readouterr().err
  assert "2GFB" in err  # noqa: S101
  assert "3416" in err  # noqa: S101
  assert "3463" in err  # noqa: S101
  assert "fixtures.py --refresh 2GFB" in err  # noqa: S101

  monkeypatch.setattr(lae, "_reverify_knn_disjoint", forbidden_neighbors)
  monkeypatch.setattr(lae, "build_exact_batch", lambda _f, _du: batch)
  with pytest.raises(SystemExit) as exc_sampling:
    las.build_lane_batch(fixture, "P09-s@1.0", None)
  assert exc_sampling.value.code == 2  # noqa: S101
  err_sampling = capsys.readouterr().err
  assert "2GFB" in err_sampling  # noqa: S101
  assert "3416" in err_sampling  # noqa: S101
  assert "fixtures.py --refresh 2GFB" in err_sampling  # noqa: S101
