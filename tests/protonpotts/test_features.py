"""P5: the ProtonPotts v6 host featurizer, rule by rule and against the sealed upstream dumps.

Two kinds of test, because a rule can be right in isolation and still miss what upstream does:

* synthetic-PDB tests pin each rule the featurizer implements, one at a time, so a change to one rule
  fails exactly one test. They need nothing installed.
* conformance tests compare the featurizer with the SEALED upstream feature dumps (P4b run
  ``6a8ee503``, P4c run ``9a9b75bb``) on ten real structures. They need
  ``AMINX_PROTONPOTTS_FEATURES_DIR`` (the directory holding ``features_v6`` and ``features_v6_p4c``)
  and ``AMINX_POTTSMPNN_REPO`` (a PottsMPNN checkout holding the PDBs), and skip without them.

Every rule's provenance (upstream source line, or the cell that measured it) is in
``aminx.families.protonpotts_mpnn.features``'s docstring.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from aminx.families.protonpotts_mpnn.features import (
  ATOM37_ORDER,
  PdbFeatureError,
  featurize_pdb,
  loaded_residues,
)
from aminx.families.protonpotts_mpnn.vocab import token_index, upstream_to_aminx_index

# --- synthetic PDB helpers -----------------------------------------------------------


def _atom(  # noqa: PLR0913
  serial: int,
  name: str,
  resname: str,
  number: int,
  *,
  chain: str = "A",
  icode: str = " ",
  alt: str = " ",
  xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
  occ: float = 1.0,
  element: str | None = None,
  record: str = "ATOM  ",
) -> str:
  padded = f" {name:<3}" if len(name) < 4 else name  # noqa: PLR2004
  elem = element if element is not None else name[0]
  return (
    f"{record}{serial:>5} {padded}{alt}{resname:>3} {chain}{number:>4}{icode}   "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{occ:>6.2f}{20.0:>6.2f}          {elem:>2}"
  )


def _residue(
  start: int,
  resname: str,
  number: int,
  *,
  chain: str = "A",
  icode: str = " ",
  occ: dict[str, float] | None = None,
  extra: tuple[str, ...] = (),
  origin: float = 0.0,
) -> list[str]:
  occ = occ or {}
  names = ("N", "CA", "C", "O", *extra)
  return [
    _atom(
      start + i,
      n,
      resname,
      number,
      chain=chain,
      icode=icode,
      xyz=(origin + float(i), float(number), 0.0),
      occ=occ.get(n, 1.0),
    )
    for i, n in enumerate(names)
  ]


def _write(tmp_path: Path, lines: list[str]) -> Path:
  path = tmp_path / "t.pdb"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def _chain(tmp_path: Path, specs: list[tuple[str, int]]) -> Path:
  lines: list[str] = []
  for resname, number in specs:
    lines += _residue(len(lines) + 1, resname, number)
  return _write(tmp_path, lines)


# --- layout and masking --------------------------------------------------------------


def test_atoms_are_placed_by_name_in_foundry_order(tmp_path: Path) -> None:
  lines = _residue(1, "SER", 1, extra=("CB", "OG"))
  out = featurize_pdb(_write(tmp_path, lines))
  assert out["X"].shape == (1, 1, 37, 3)
  assert out["X"].dtype == np.float32
  assert out["X_m"][0, 0].nonzero()[0].tolist() == [
    ATOM37_ORDER.index(n) for n in ("N", "CA", "C", "O", "CB", "OG")
  ]
  assert ATOM37_ORDER[:5] == ("N", "CA", "C", "O", "CB")  # O before CB: not the legacy order


def test_atom_mask_is_strictly_above_half_and_keeps_the_coordinates(tmp_path: Path) -> None:
  lines = _residue(1, "VAL", 1, extra=("CB", "CG1", "CG2"), occ={"CB": 0.5, "CG1": 0.51, "CG2": 0.49})
  out = featurize_pdb(_write(tmp_path, lines))
  mask = out["X_m"][0, 0]
  assert not mask[ATOM37_ORDER.index("CB")]  # exactly 0.50: masked (measured on 1BVC)
  assert mask[ATOM37_ORDER.index("CG1")]
  assert not mask[ATOM37_ORDER.index("CG2")]
  assert out["X"][0, 0, ATOM37_ORDER.index("CB")].any()  # masked, but its coordinates are kept


def test_first_alternate_location_wins_and_hydrogens_and_hetatm_are_ignored(tmp_path: Path) -> None:
  lines = _residue(1, "SER", 1, extra=("CB",))
  lines[4] = _atom(5, "CB", "SER", 1, alt="A", xyz=(10.0, 0.0, 0.0), occ=0.6)
  lines.insert(5, _atom(6, "CB", "SER", 1, alt="B", xyz=(20.0, 0.0, 0.0), occ=0.4))
  lines.append(_atom(7, "HB2", "SER", 1, element="H"))
  lines.append(_atom(8, "O", "HOH", 900, record="HETATM"))
  out = featurize_pdb(_write(tmp_path, lines))
  assert out["S"].shape == (1, 1)  # the water is not a residue
  assert out["X"][0, 0, ATOM37_ORDER.index("CB"), 0] == pytest.approx(10.0)


def test_arginine_nh1_nh2_coordinates_are_exchanged_when_nh1_is_farther_from_cd(
  tmp_path: Path,
) -> None:
  def arg(nh1: tuple[float, float, float], nh2: tuple[float, float, float]) -> list[str]:
    lines = _residue(1, "ARG", 1)
    lines += [_atom(5 + i, n, "ARG", 1, xyz=xyz) for i, (n, xyz) in enumerate(
      (("CD", (0.0, 0.0, 0.0)), ("NH1", nh1), ("NH2", nh2))
    )]
    return lines

  far_first = featurize_pdb(_write(tmp_path, arg((5.0, 0.0, 0.0), (1.0, 0.0, 0.0))))
  near_first = featurize_pdb(_write(tmp_path, arg((1.0, 0.0, 0.0), (5.0, 0.0, 0.0))))
  i1, i2 = ATOM37_ORDER.index("NH1"), ATOM37_ORDER.index("NH2")
  assert far_first["X"][0, 0, i1, 0] == pytest.approx(1.0)  # exchanged: NH1 is now the near one
  assert far_first["X"][0, 0, i2, 0] == pytest.approx(5.0)
  assert near_first["X"][0, 0, i1, 0] == pytest.approx(1.0)  # already canonical: untouched


# --- which residues survive ----------------------------------------------------------


@pytest.mark.parametrize(
  ("occupancy", "kept"), [(0.81, True), (0.8, False), (0.79, False), (0.5, False), (1.0, True)]
)
def test_backbone_threshold_is_0_8_inclusive_of_unresolved(
  tmp_path: Path, occupancy: float, kept: bool
) -> None:
  """``occupancy <= 0.8`` is unresolved (atomworks), and it is NOT the 0.5 that masks atoms."""
  lines = _residue(1, "ALA", 1) + _residue(5, "GLY", 2, occ={"C": occupancy})
  if kept:
    assert featurize_pdb(_write(tmp_path, lines))["S"].shape == (1, 2)
  else:
    assert featurize_pdb(_write(tmp_path, lines))["S"].shape == (1, 1)


@pytest.mark.parametrize("atom", ["N", "CA", "C", "O"])
def test_each_of_n_ca_c_o_is_required(tmp_path: Path, atom: str) -> None:
  residue = [line for line in _residue(5, "GLY", 2) if line[12:16].strip() != atom]
  out = featurize_pdb(_write(tmp_path, _residue(1, "ALA", 1) + residue))
  assert out["S"].shape == (1, 1)


def test_a_residue_whose_backbone_passes_is_dropped_only_by_its_own_atoms(tmp_path: Path) -> None:
  """Side-chain occupancy never drops a residue (1BVC keeps VAL1 with CB/CG at 0.50)."""
  lines = _residue(1, "VAL", 1, extra=("CB", "CG1"), occ={"CB": 0.1, "CG1": 0.1})
  assert featurize_pdb(_write(tmp_path, lines))["S"].shape == (1, 1)


def test_all_residues_dropped_raises_like_upstream(tmp_path: Path) -> None:
  lines = _residue(1, "ALA", 1, occ={"CA": 0.3})
  with pytest.raises(PdbFeatureError, match="atom_array cannot be empty"):
    featurize_pdb(_write(tmp_path, lines))


# --- R_idx, chains, duplicates -------------------------------------------------------


def test_r_idx_counts_residues_it_does_not_follow_the_numbering(tmp_path: Path) -> None:
  """1EL1 is numbered -1, 1, 2 and upstream's R_idx is 0, 1, 2."""
  out = featurize_pdb(_chain(tmp_path, [("ALA", -1), ("GLY", 1), ("SER", 2), ("THR", 10)]))
  assert out["R_idx"][0].tolist() == [0, 1, 2, 3]
  assert out["R_idx"].dtype == np.int32


def test_r_idx_gaps_come_only_from_dropped_residues_and_are_counted_before_the_drop(
  tmp_path: Path,
) -> None:
  lines = (
    _residue(1, "ALA", 1)
    + _residue(5, "GLY", 2, occ={"CA": 0.5})
    + _residue(9, "SER", 3)
    + _residue(13, "THR", 4)
  )
  out = featurize_pdb(_write(tmp_path, lines))
  assert out["R_idx"][0].tolist() == [0, 2, 3]


def test_a_dropped_first_residue_does_not_shift_r_idx(tmp_path: Path) -> None:
  lines = _residue(1, "ALA", 1, occ={"N": 0.2}) + _residue(5, "GLY", 2)
  assert featurize_pdb(_write(tmp_path, lines))["R_idx"][0].tolist() == [1]


def test_chains_restart_r_idx_and_are_labelled_in_order_of_first_appearance(tmp_path: Path) -> None:
  lines = (
    _residue(1, "ALA", 1, chain="B")
    + _residue(5, "GLY", 2, chain="B")
    + _residue(9, "SER", 1, chain="A")
  )
  out = featurize_pdb(_write(tmp_path, lines))
  assert out["chain_labels"][0].tolist() == [0, 0, 1]  # B first, then A: file order, not alphabetical
  assert out["R_idx"][0].tolist() == [0, 1, 0]
  assert out["residue_mask"].all()


def test_reused_number_keeps_only_the_last_residue_name(tmp_path: Path) -> None:
  """1TPK: ``42 ALA`` then ``42A GLN``; upstream drops the ALA and counts R_idx after."""
  lines = (
    _residue(1, "SER", 41)
    + _residue(5, "ALA", 42)
    + _residue(9, "GLN", 42, icode="A")
    + _residue(13, "THR", 43)
  )
  path = _write(tmp_path, lines)
  assert [r[3] for r in loaded_residues(path)] == ["SER", "GLN", "THR"]
  out = featurize_pdb(path)
  assert out["S"][0].tolist() == [token_index(a) for a in ("S", "Q", "T")]
  assert out["R_idx"][0].tolist() == [0, 1, 2]


def test_same_name_at_the_same_number_with_different_codes_is_not_removed(tmp_path: Path) -> None:
  """The function groups by (chain, number, NAME), so identical names are both kept."""
  lines = _residue(1, "ALA", 42) + _residue(5, "ALA", 42, icode="A")
  assert featurize_pdb(_write(tmp_path, lines))["S"].shape == (1, 2)


# --- S and labels --------------------------------------------------------------------


def test_labels_align_to_loaded_residues_and_reach_s(tmp_path: Path) -> None:
  path = _chain(tmp_path, [("ALA", 1), ("HIS", 2), ("ASP", 3)])
  out = featurize_pdb(path, ["", "HIS-P", "ASP-D"])
  assert out["S"][0].tolist() == [token_index("A"), token_index("HIS-P"), token_index("ASP-D")]


def test_labels_of_dropped_residues_are_dropped_with_them(tmp_path: Path) -> None:
  lines = _residue(1, "HIS", 1, occ={"CA": 0.2}) + _residue(5, "ASP", 2)
  out = featurize_pdb(_write(tmp_path, lines), ["HIS-P", "ASP-D"])
  assert out["S"][0].tolist() == [token_index("ASP-D")]


def test_label_count_must_match_the_loaded_residues(tmp_path: Path) -> None:
  with pytest.raises(PdbFeatureError, match="labels for 2 loaded residues"):
    featurize_pdb(_chain(tmp_path, [("ALA", 1), ("GLY", 2)]), [""])


def test_a_v4_label_is_refused_through_the_featurizer(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="v4 labels"):
    featurize_pdb(_chain(tmp_path, [("HIS", 1)]), ["HID"])


# --- what is refused, not guessed ----------------------------------------------------


def test_hetatm_residues_are_skipped_even_with_a_full_backbone(tmp_path: Path) -> None:
  """A water would be dropped by the backbone rule anyway; a residue with N, CA, C, O is the real test."""
  lines = _residue(1, "ALA", 1)
  lines += [
    _atom(5 + i, n, "SEP", 2, record="HETATM", xyz=(float(i), 0.0, 0.0))
    for i, n in enumerate(("N", "CA", "C", "O"))
  ]
  assert featurize_pdb(_write(tmp_path, lines))["S"].shape == (1, 1)


def test_a_non_standard_atom_residue_is_refused_not_kept_as_x(tmp_path: Path) -> None:
  """Upstream atomizes it out of the residue tokens; no sealed cell shows how, so refuse."""
  with pytest.raises(PdbFeatureError, match="not one of the 20 standard residues or UNK"):
    featurize_pdb(_write(tmp_path, _residue(1, "SEP", 1)))


def test_unk_is_a_residue_token_and_maps_to_x(tmp_path: Path) -> None:
  out = featurize_pdb(_write(tmp_path, _residue(1, "ALA", 1) + _residue(5, "UNK", 2)))
  assert out["S"][0].tolist() == [token_index("A"), token_index("X")]


def test_hydrogens_contribute_nothing(tmp_path: Path) -> None:
  heavy = _residue(1, "SER", 1, extra=("CB",))
  with_h = [*heavy, _atom(9, "HB2", "SER", 1, element="H"), _atom(10, "H", "SER", 1, element="H")]
  a, b = featurize_pdb(_write(tmp_path, heavy)), featurize_pdb(_write(tmp_path, with_h))
  assert all(np.array_equal(a[k], b[k]) for k in a)


def test_selenomethionine_and_blank_chain_are_refused(tmp_path: Path) -> None:
  mse = _residue(1, "ALA", 1) + [_atom(5, "SE", "MSE", 2, record="HETATM")]
  with pytest.raises(PdbFeatureError, match="MSE"):
    featurize_pdb(_write(tmp_path, mse))
  with pytest.raises(PdbFeatureError, match="blank chain"):
    featurize_pdb(_write(tmp_path, _residue(1, "ALA", 1, chain=" ")))


# --- conformance with the SEALED upstream dumps --------------------------------------

_FEATURES_DIR = os.environ.get("AMINX_PROTONPOTTS_FEATURES_DIR")
_POTTS_REPO = os.environ.get("AMINX_POTTSMPNN_REPO")
_needs_dumps = pytest.mark.skipif(
  not (_FEATURES_DIR and _POTTS_REPO),
  reason="AMINX_PROTONPOTTS_FEATURES_DIR and AMINX_POTTSMPNN_REPO not set",
)
_FIRE = "energy_benchmark_datasets/fireprot_pdbs"
# (dump subdir, cell, pdb relative to the PottsMPNN checkout). P4b and P4c dump directories.
_CELLS = [
  ("features_v6/protonpotts_v6_features", "pkad_unlabelled", f"{_FIRE}/1BVC.pdb"),
  ("features_v6/protonpotts_v6_features", "pkad_labelled", f"{_FIRE}/1BVC.pdb"),
  ("features_v6/protonpotts_v6_features", "multichain", "energy_benchmark_datasets/covid_pdbs/6m0j.pdb"),
  ("features_v6/protonpotts_v6_features", "ligand", "inputs/example_pdbs/swe1_ligand.pdb"),
  ("features_v6_p4c/protonpotts_v6_features_p4c", "only_o", f"{_FIRE}/1CQW.pdb"),
  ("features_v6_p4c/protonpotts_v6_features_p4c", "gap", f"{_FIRE}/1EL1.pdb"),
  ("features_v6_p4c/protonpotts_v6_features_p4c", "many_gaps", f"{_FIRE}/1BAH.pdb"),
  ("features_v6_p4c/protonpotts_v6_features_p4c", "gap_and_drops", "inputs/example_pdbs/2yc3.pdb"),
  ("features_v6_p4c/protonpotts_v6_features_p4c", "icode", f"{_FIRE}/1TPK.pdb"),
  # P4d (run 7ec1f6b9): a residue where ONLY O fails the 0.8 backbone threshold; same dumper as P4c.
  ("features_v6_p4d/protonpotts_v6_features_p4c", "only_o_1olr", f"{_FIRE}/1OLR.pdb"),
]
# The dumper's labelling rule (dump_protonpotts_oracles.py:74): cycle by ordinal within residue type.
_CYCLE = {
  "HIS": ("HIS-P", "HIS-S", "HIS-A"),
  "ASP": ("ASP-P", "ASP-D", "ASP-A"),
  "GLU": ("GLU-P", "GLU-D", "GLU-A"),
}


def _cycling_labels(pdb: Path) -> list[str]:
  seen: dict[str, int] = {}
  out = []
  for _chain_id, _number, _icode, name in loaded_residues(pdb):
    options = _CYCLE.get(name)
    if options is None:
      out.append("")
      continue
    k = seen.get(name, 0)
    seen[name] = k + 1
    out.append(options[k % 3])
  return out


@_needs_dumps
@pytest.mark.parametrize(("subdir", "cell", "pdb"), _CELLS, ids=[c[1] for c in _CELLS])
def test_featurizer_is_exact_against_the_sealed_dump(subdir: str, cell: str, pdb: str) -> None:
  assert _FEATURES_DIR and _POTTS_REPO
  structure = Path(_POTTS_REPO).expanduser() / pdb
  ref = np.load(Path(_FEATURES_DIR).expanduser() / subdir / f"{cell}.npz")
  labels = _cycling_labels(structure) if cell == "pkad_labelled" else None
  got = featurize_pdb(structure, labels)
  for key in ("R_idx", "chain_labels", "residue_mask", "X_m"):
    assert got[key].shape == ref[key].shape, key
    assert np.array_equal(got[key], ref[key]), key
  want_s = np.vectorize(upstream_to_aminx_index)(ref["S"])
  assert np.array_equal(got["S"], want_s)
  assert got["X"].shape == ref["X"].shape
  mask = ref["X_m"][..., None] & np.ones(3, dtype=bool)
  assert np.array_equal(got["X"][mask], ref["X"][mask])  # bit-exact where the atom is unmasked
  assert np.array_equal(got["X"], ref["X"])  # and in fact everywhere, masked atoms included


@_needs_dumps
def test_input_upstream_refuses_is_refused_here_too() -> None:
  """1IFC: 115 of 131 residues have an unresolved backbone and upstream raises 'empty'.

  The sealed P4c manifest records that refusal; this checks the featurizer agrees on that input.
  """
  assert _POTTS_REPO
  with pytest.raises(PdbFeatureError, match="atom_array cannot be empty"):
    featurize_pdb(Path(_POTTS_REPO).expanduser() / f"{_FIRE}/1IFC.pdb")
