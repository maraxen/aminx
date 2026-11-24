"""Write one LASEr sample as a full-atom PDB through ``proxide.write_pdb``.

Divergences from upstream LASErMPNN (``~/repos/LASErMPNN/run_inference.py``).

Upstream ``output_protein_structure`` (the writer ``run_inference`` calls) sets
the B-factor to ``softmax(sequence_logits)`` gathered at the sampled token, one
value per residue, repeated across that residue's atoms. A missing tensor
becomes 1. Fixed positions (``--fix_beta``) still use that probability of the
kept token; the input B-factor of 1.0 is only the fix flag, not the written
column. Occupancy is 1.0. ``output_ligand_structure`` does not set betas. The
unused ``output_protein_structure_numpy`` writes 1.0 on every protein atom.

This writer follows that probability rule, including at positions the user
fixed, and writes ligand B-factors as 0.0. Further divergences:

- No backbone amide N-H imputation, so backbone H is absent.
- No titratable-H cleanup: no cysteine disulfide or HG handling, and no
  histidine tautomer choice. Rotatable hydrogens are written as
  ``build_rotamers`` placed them.
- Atom order within a residue is dataset slot order, not IUPAC order.
- Ligand coordinates are the input coordinates, not the noised backbone.
  Noncanonical methyl caps are synthetic and are not written.
- B-factors print with the PDB column's two decimals.
- Residue names are the canonical name of the sampled letter (``X`` is
  ``UNK``, histidine is ``HIS``). Input residue names are not preserved.
  A residue skipped because its backbone N, CA or C is missing is absent,
  which leaves a gap in the residue numbering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from proxide import write_pdb
from proxide.core.containers import Protein

from aminx.families.laser_mpnn.featurize import (
  LASER_ALPHABET,
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  LaserFeatures,
  canonical_resname,
  residue_atom_names,
)

_MAX_PDB_ATOMS = 99_999
_Provenance = Literal["none", "sidecar"]


def _name_table(*, protonated: bool) -> np.ndarray:
  width = MAX_PROTONATED_ATOMS if protonated else MAX_ATOMS
  table = np.full((len(LASER_ALPHABET), width), "", dtype=object)
  for index, letter in enumerate(LASER_ALPHABET):
    names = residue_atom_names(letter, protonated=protonated)
    table[index, : len(names)] = names
  return table


_HEAVY_NAMES = _name_table(protonated=False)
_PROTONATED_NAMES = _name_table(protonated=True)
_RESNAMES = np.array([canonical_resname(letter) for letter in LASER_ALPHABET], dtype=object)


def _first_character(names: np.ndarray) -> np.ndarray:
  """Element letter of each atom name. Dataset names start with that letter."""
  wide = np.asarray(names, dtype="U8")
  characters = wide.view(np.dtype("<U1")).reshape(wide.shape[0], -1)
  return characters[:, 0]


def _blank_to_space(codes: tuple[str, ...]) -> np.ndarray:
  raw = np.asarray(codes, dtype=object)
  text = np.array([str(code) for code in raw.tolist()], dtype="U1")
  return np.where(text == "", " ", text)


def _check_layout(
  features: LaserFeatures,
  sequence: np.ndarray,
  xyz: np.ndarray,
  probability: np.ndarray,
) -> np.ndarray:
  if xyz.ndim != 3 or xyz.shape[-1] != 3:
    msg = f"coords must be (L, atom, 3), got {xyz.shape}"
    raise ValueError(msg)
  length, width, _three = xyz.shape
  if sequence.shape != (length,) or probability.shape != (length,):
    msg = f"sequence {sequence.shape} and probability {probability.shape} must both be ({length},)"
    raise ValueError(msg)
  if int(np.asarray(features.pdb_resnums).shape[0]) != length:
    msg = f"pdb identity length {len(features.pdb_chain_ids)} does not match sequence length {length}"
    raise ValueError(msg)
  if width == MAX_PROTONATED_ATOMS:
    return _PROTONATED_NAMES
  if width == MAX_ATOMS:
    return _HEAVY_NAMES
  msg = f"atom width {width} is neither heavy ({MAX_ATOMS}) nor protonated ({MAX_PROTONATED_ATOMS})"
  raise ValueError(msg)


def _protein_rows(
  features: LaserFeatures,
  sequence: np.ndarray,
  xyz: np.ndarray,
  probability: np.ndarray,
  table: np.ndarray,
) -> tuple[np.ndarray, ...]:
  names = np.asarray(table[sequence], dtype=object)
  present = np.isfinite(xyz).all(axis=-1)
  if bool((present & (names == "")).any()):
    msg = "a placed atom has no dataset name"
    raise ValueError(msg)
  flat_present = present.reshape(-1)
  counts = present.sum(axis=1).astype(np.int64, copy=False)
  return (
    names.reshape(-1)[flat_present],
    xyz.reshape(-1, 3)[flat_present].astype(np.float32, copy=False),
    np.repeat(_RESNAMES[sequence], counts),
    np.repeat(np.asarray(features.pdb_chain_ids, dtype=object), counts),
    np.repeat(np.asarray(features.pdb_resnums, dtype=np.int64), counts),
    np.repeat(_blank_to_space(features.pdb_icodes), counts),
    np.repeat(probability, counts).astype(np.float32, copy=False),
  )


def _append_real_ligand(
  features: LaserFeatures,
  rows: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
  names, xyz, res_names, chains, resindex, insertion, b_factors = rows
  ligand_xyz = np.asarray(features.ligand_coords, dtype=np.float32)
  synthetic = np.asarray(features.ligand_synthetic, dtype=np.bool_)
  if ligand_xyz.shape[0] != synthetic.shape[0]:
    msg = "ligand_synthetic does not match ligand_coords"
    raise ValueError(msg)
  real = ~synthetic
  ligand_xyz = ligand_xyz[real]
  if ligand_xyz.size and not bool(np.isfinite(ligand_xyz).all()):
    msg = "ligand coordinates are non-finite"
    raise ValueError(msg)
  if ligand_xyz.shape[0] == 0:
    elements = _first_character(names)
    hetero = np.zeros(names.shape[0], dtype=np.bool_)
    return (names, xyz, res_names, chains, resindex, insertion, b_factors, elements, hetero)
  ligand_names = np.asarray(features.ligand_atom_names, dtype=object)[real]
  ligand_elements = np.asarray(features.ligand_elements, dtype=object)[real]
  if bool((ligand_names == "").any() or (ligand_elements == "").any()):
    msg = "a ligand atom is missing its name or element"
    raise ValueError(msg)
  return (
    np.concatenate([names, ligand_names]),
    np.concatenate([xyz, ligand_xyz]),
    np.concatenate([res_names, np.asarray(features.ligand_res_names, dtype=object)[real]]),
    np.concatenate([chains, np.asarray(features.ligand_chain_ids, dtype=object)[real]]),
    np.concatenate([resindex, np.asarray(features.ligand_resnums, dtype=np.int64)[real]]),
    np.concatenate([insertion, _blank_to_space(features.ligand_icodes)[real]]),
    np.concatenate([b_factors, np.zeros(ligand_xyz.shape[0], dtype=np.float32)]),
    np.concatenate([_first_character(names), ligand_elements.astype(str)]),
    np.concatenate(
      [
        np.zeros(names.shape[0], dtype=np.bool_),
        np.ones(ligand_xyz.shape[0], dtype=np.bool_),
      ],
    ),
  )


def _as_protein(rows: tuple[np.ndarray, ...]) -> Protein:
  names, xyz, res_names, chains, resindex, insertion, b_factors, elements, hetero = rows
  chain_ids = np.unique(np.asarray(chains, dtype=str))
  chain_index = np.searchsorted(chain_ids, np.asarray(chains, dtype=str)).astype(np.int32)
  n_atoms = int(xyz.shape[0])
  return Protein(
    coordinates=np.zeros((n_atoms, 37, 3), dtype=np.float32),
    aatype=np.zeros((n_atoms,), dtype=np.int32),
    residue_index=resindex.astype(np.int32, copy=False),
    chain_index=chain_index,
    format="Full",
    full_coordinates=xyz,
    atom_names=names.tolist(),
    res_names=res_names.tolist(),
    atom_res_index=resindex.astype(np.int32, copy=False),
    elements=elements.tolist(),
    atom_chain_ids=np.asarray(chains, dtype=str).tolist(),
    chain_ids=chain_ids.tolist(),
    b_factors=b_factors,
    occupancies=np.ones((n_atoms,), dtype=np.float32),
    insertion_codes=insertion.astype(str).tolist(),
    hetero=hetero,
  )


def build_full_protein(
  features: LaserFeatures,
  laser_sequence: Int[np.ndarray, " L"],
  coords: Float[np.ndarray, "L atom xyz"],
  token_prob: Float[np.ndarray, " L"],
) -> Protein:
  """Full-layout ``Protein`` for one sample. No I/O. NaN slots are dropped."""
  sequence = np.asarray(laser_sequence, dtype=np.int64)
  xyz = np.asarray(coords)
  probability = np.asarray(token_prob, dtype=np.float32)
  table = _check_layout(features, sequence, xyz, probability)
  rows = _protein_rows(features, sequence, xyz, probability, table)
  return _as_protein(_append_real_ligand(features, rows))


def write_sample_pdb(
  features: LaserFeatures,
  laser_sequence: Int[np.ndarray, " L"],
  coords: Float[np.ndarray, "L atom xyz"],
  token_prob: Float[np.ndarray, " L"],
  path: Path,
  *,
  structure_id: str,
  provenance: _Provenance,
) -> None:
  """Write ``path`` in place. A proxide ``ValueError`` is re-raised with ``structure_id``."""
  protein = build_full_protein(features, laser_sequence, coords, token_prob)
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  try:
    write_pdb(
      protein,
      destination,
      record_type="source",
      insertion_codes="write",
      provenance=provenance,
    )
  except ValueError as exc:
    msg = f"{structure_id}: {exc}"
    raise ValueError(msg) from exc
