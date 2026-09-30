# ruff: noqa: S101
"""Unit tests for the PottsMPNN host featurizer (synthetic PDB strings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aminx.families.potts_mpnn.featurize import (
  MODEL_ALPHABET,
  X_INDEX,
  PottsFeatures,
  PottsInputError,
  knn_boundary_tie,
  pad,
  parse_pdb_upstream,
  tied_featurize_port,
)


def _atom(
  serial: int,
  atom: str,
  resname: str,
  chain: str,
  resseq: int,
  x: float,
  y: float,
  z: float,
  *,
  icode: str = " ",
  altloc: str = " ",
  record: str = "ATOM  ",
) -> str:
  return (
    f"{record:<6.6}{serial:5d} {atom:>4.4}{altloc:1.1}{resname:>3.3} {chain:1.1}"
    f"{resseq:4d}{icode:1.1}   {x:8.3f}{y:8.3f}{z:8.3f}"
  )


def _backbone(
  chain: str,
  resseq: int,
  resname: str,
  x0: float,
  *,
  icode: str = " ",
  altloc: str = " ",
  record: str = "ATOM  ",
  serial: int = 1,
  atoms: tuple[str, ...] = ("N", "CA", "C", "O"),
) -> list[str]:
  lines: list[str] = []
  for offset, atom in enumerate(atoms):
    lines.append(
      _atom(
        serial + offset,
        atom,
        resname,
        chain,
        resseq,
        x0 + offset,
        float(ord(chain)),
        float(resseq),
        icode=icode,
        altloc=altloc,
        record=record,
      ),
    )
  return lines


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
  path = tmp_path / name
  path.write_text("\n".join(lines) + "\n")
  return path


def _features(
  path: Path,
  chain_dict: dict[str, tuple[list[str], list[str]]] | None = None,
  *,
  skip_gaps: bool = False,
) -> tuple[list[dict[str, object]], PottsFeatures]:
  parsed = parse_pdb_upstream(path, None, skip_gaps=skip_gaps)
  return parsed, tied_featurize_port(parsed, chain_dict)[0]


def test_numbering_gap_inserted_and_removed(tmp_path: Path) -> None:
  """Missing residue numbers become a gap row unless skip_gaps is set."""
  path = _write(
    tmp_path,
    "gap.pdb",
    [*_backbone("A", 1, "ALA", 10.0), *_backbone("A", 3, "GLY", 20.0, serial=5)],
  )
  parsed_gap, feat_gap = _features(path, skip_gaps=False)
  assert parsed_gap[0]["seq_chain_A"] == "A-G"
  assert feat_gap.L_total == 3
  assert feat_gap.s.tolist() == [MODEL_ALPHABET.index("A"), X_INDEX, MODEL_ALPHABET.index("G")]
  assert feat_gap.present.tolist() == [1.0, 0.0, 1.0]
  assert np.all(feat_gap.x[1] == 0.0)
  assert feat_gap.x[0, 1, 0] == np.float32(11.0)

  _parsed_skip, feat_skip = _features(path, skip_gaps=True)
  assert feat_skip.L_total == 2
  assert feat_skip.s.tolist() == [MODEL_ALPHABET.index("A"), MODEL_ALPHABET.index("G")]
  assert feat_skip.present.tolist() == [1.0, 1.0]
  assert "-" not in _parsed_skip[0]["seq"]


def test_insertion_code_orders_blank_before_letter(tmp_path: Path) -> None:
  """One row per insertion code at the same residue number, sorted (blank then letter)."""
  path = _write(
    tmp_path,
    "icode.pdb",
    [
      *_backbone("A", 4, "ALA", 1.0, icode=" "),
      *_backbone("A", 4, "GLY", 8.0, icode="A", serial=5),
    ],
  )
  for skip in (False, True):
    _parsed, feat = _features(path, skip_gaps=skip)
    assert feat.L_total == 2
    assert feat.s.tolist() == [MODEL_ALPHABET.index("A"), MODEL_ALPHABET.index("G")]
    assert feat.x[0, 1, 0] == np.float32(2.0)
    assert feat.x[1, 1, 0] == np.float32(9.0)


def test_mse_hetatm_becomes_met(tmp_path: Path) -> None:
  path = _write(
    tmp_path,
    "mse.pdb",
    _backbone("A", 2, "MSE", 3.0, record="HETATM"),
  )
  parsed, feat = _features(path)
  assert parsed[0]["seq"] == "M"
  assert feat.s.tolist() == [MODEL_ALPHABET.index("M")]
  assert feat.present.tolist() == [1.0]


def test_altloc_and_repeat_atom_keep_first_occurrence(tmp_path: Path) -> None:
  """First ATOM line for an atom at (resnum, icode) wins; later altloc/MODEL copies do not."""
  first = _backbone("A", 1, "ALA", 1.0)
  second_ca = _atom(9, "CA", "ALA", "A", 1, 40.0, 0.0, 0.0, altloc="B")
  second_n = _atom(10, "N", "ALA", "A", 1, 50.0, 0.0, 0.0, altloc="B")
  path = _write(tmp_path, "alt.pdb", [*first, second_ca, "ENDMDL", "MODEL        2", second_n])
  _parsed, feat = _features(path)
  assert feat.L_total == 1
  assert feat.present.tolist() == [1.0]
  assert feat.x[0, 0, 0] == np.float32(1.0)
  assert feat.x[0, 1, 0] == np.float32(2.0)


def test_partial_backbone_is_not_present(tmp_path: Path) -> None:
  """A residue missing any backbone atom has present=0; finite atoms are kept and NaNs become 0."""
  path = _write(
    tmp_path,
    "partial.pdb",
    [
      *_backbone("A", 1, "ALA", 4.0, atoms=("N", "CA", "C")),
      *_backbone("A", 2, "SER", 30.0, serial=4),
    ],
  )
  _parsed, feat = _features(path)
  assert feat.present.tolist() == [0.0, 1.0]
  assert feat.x[0, 1, 0] == np.float32(5.0)
  assert np.all(feat.x[0, 3] == 0.0)
  assert feat.s.tolist() == [MODEL_ALPHABET.index("A"), MODEL_ALPHABET.index("S")]


def test_unknown_resname_is_dash_then_x(tmp_path: Path) -> None:
  path = _write(tmp_path, "unk.pdb", _backbone("A", 1, "XXX", 1.0))
  parsed, feat = _features(path)
  assert parsed[0]["seq_chain_A"] == "-"
  assert feat.s.tolist() == [X_INDEX]
  assert feat.present.tolist() == [1.0]


def test_chain_order_masks_residue_idx_and_annotations(tmp_path: Path) -> None:
  """Sorted designed chains, then sorted fixed chains; annotations are chain-local then concatenated."""
  lines = [
    *_backbone("A", 1, "ALA", 10.0),
    *_backbone("A", 3, "GLY", 20.0, serial=5),
    *_backbone("C", 1, "SER", 30.0, serial=9),
    *_backbone("B", 1, "VAL", 40.0, serial=13),
    *_backbone("B", 2, "LEU", 50.0, serial=17),
  ]
  path = _write(tmp_path, "demo.pdb", lines)
  parsed = parse_pdb_upstream(path, None, skip_gaps=False)
  assert parsed[0]["chain_order"] == ["A", "B", "C"]
  assert parsed[0]["seq_chain_A"] == "A-G"
  name = parsed[0]["name"]
  assert name == "demo"
  bias_c = np.zeros((1, 21), dtype=np.float64)
  bias_c[0, 4] = 0.25
  pssm_bias = np.zeros((1, 21), dtype=np.float64)
  pssm_bias[0, 0] = 3.0
  feat = tied_featurize_port(
    parsed,
    {name: (["C", "A"], ["B"])},
    fixed_position_dict={name: {"A": [3], "C": [1], "B": [1]}},
    omit_aa_dict={name: {"A": [], "C": [([1], "AW")]}},
    tied_positions_dict={
      name: [
        {"A": [[1], [0.2]], "C": [[1], [0.8]]},
        {"B": [2]},
      ],
    },
    pssm_dict={
      name: {
        "A": {},
        "C": {
          "pssm_coef": np.array([0.5]),
          "pssm_bias": pssm_bias,
          "pssm_log_odds": np.full((1, 21), 7.0),
        },
      },
    },
    bias_by_res_dict={
      name: {
        "A": np.zeros((3, 21)),
        "C": bias_c,
      },
    },
  )[0]
  assert feat.letter_list == ("A", "C", "B")
  assert feat.masked_list == ("A", "C")
  assert feat.visible_list == ("B",)
  assert feat.chain_lens == (3, 1, 2)
  assert feat.masked_chain_lengths == (3, 1)
  assert feat.L_total == 6
  assert feat.chain_m.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
  assert feat.chain_encoding.tolist() == [1, 1, 1, 2, 3, 3]
  assert feat.residue_idx.tolist() == [0, 1, 2, 103, 204, 205]
  # Chain-local 1-based index: A position 3 → global row 2; C position 1 → row 3.
  # Fixed chain B's fixed list is ignored, so row 4 stays designed-mask 1.
  assert feat.chain_m_pos.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
  assert feat.omit_aa_mask[3, MODEL_ALPHABET.index("A")] == 1.0
  assert feat.omit_aa_mask[3, MODEL_ALPHABET.index("W")] == 1.0
  assert feat.omit_aa_mask.sum() == 2.0
  assert feat.pssm_coef.tolist() == [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
  assert feat.pssm_bias[3, 0] == np.float32(3.0)
  assert feat.pssm_log_odds[0, 0] == np.float32(10000.0)
  assert feat.pssm_log_odds[3, 0] == np.float32(7.0)
  assert feat.bias_by_res[3, 4] == np.float32(0.25)
  assert feat.bias_by_res[0].sum() == 0.0
  assert feat.tied_pos == ((0, 3), (5,))
  assert feat.tied_beta.tolist() == [np.float32(0.2), 1.0, 1.0, np.float32(0.8), 1.0, 1.0]
  assert feat.s[1] == X_INDEX
  assert feat.present[1] == 0.0
  assert int(feat.lengths) == feat.L_total


def test_chains_argument_keeps_requested_order_and_drops_others(tmp_path: Path) -> None:
  path = _write(
    tmp_path,
    "two.pdb",
    [*_backbone("B", 1, "VAL", 1.0), *_backbone("A", 1, "ALA", 2.0, serial=5)],
  )
  parsed = parse_pdb_upstream(path, ["B"], skip_gaps=False)
  assert parsed[0]["chain_order"] == ["B"]
  assert "seq_chain_A" not in parsed[0]
  feat = tied_featurize_port(parsed, None)[0]
  assert feat.letter_list == ("B",)
  assert feat.s.tolist() == [MODEL_ALPHABET.index("V")]


def test_knn_boundary_tie_true_and_false() -> None:
  """#present <= min(K, L_total) < L_total, K default 48."""
  assert knn_boundary_tie(np.ones(60), 60) is False
  assert knn_boundary_tie(np.ones(30), 30) is False
  assert knn_boundary_tie(np.ones(48), 48) is False
  assert knn_boundary_tie(np.concatenate([np.ones(48), np.zeros(1)]), 49) is True
  assert knn_boundary_tie(np.concatenate([np.ones(10), np.zeros(50)]), 60) is True
  assert knn_boundary_tie(np.concatenate([np.ones(49), np.zeros(11)]), 60) is False
  present = np.concatenate([np.ones(10), np.zeros(50), np.ones(8)])
  assert knn_boundary_tie(present, 60, k=48) is True
  assert knn_boundary_tie(present, 60, k=8) is False


def test_pad_extends_with_x_and_absent_rows(tmp_path: Path) -> None:
  path = _write(tmp_path, "padme.pdb", _backbone("A", 1, "ALA", 1.0))
  _parsed, feat = _features(path)
  padded, pad_valid = pad(feat, 4)
  assert pad_valid.tolist() == [True, False, False, False]
  assert padded.L_total == 1
  assert padded.x.shape == (4, 4, 3)
  assert np.all(padded.x[1:] == 0.0)
  assert np.array_equal(padded.x[0], feat.x[0])
  assert padded.s.tolist() == [MODEL_ALPHABET.index("A"), X_INDEX, X_INDEX, X_INDEX]
  assert padded.present.tolist() == [1.0, 0.0, 0.0, 0.0]
  assert padded.chain_m.tolist() == [1.0, 0.0, 0.0, 0.0]
  assert padded.residue_idx[1] == -100
  assert padded.tied_beta[1] == np.float32(1.0)
  assert padded.pssm_log_odds[1, 0] == np.float32(0.0)
  same, same_valid = pad(feat, feat.L_total)
  assert same_valid.tolist() == [True]
  assert np.array_equal(same.x, feat.x)
  with pytest.raises(ValueError, match="shorter than L_total"):
    pad(feat, 0)


def test_non_pdb_raises(tmp_path: Path) -> None:
  cif = tmp_path / "struct.cif"
  cif.write_text("data_struct\n")
  with pytest.raises(PottsInputError, match="pottsmpnn_requires_pdb"):
    parse_pdb_upstream(cif, None, skip_gaps=False)


def test_parsing_module_does_not_import_jax() -> None:
  source = Path(parse_pdb_upstream.__code__.co_filename).read_text()
  assert "import jax" not in source
  assert "jax.numpy" not in source
