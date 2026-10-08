"""ProtonPottsMPNN v6 host featurizer: a PDB file to the six upstream feature arrays.

Spec: ``.praxia/docs/specs/261007_protonpottsmpnn-support.md`` §3, §37-§39.

Upstream builds these with atomworks (biotite, CCD lookups, atom completion). aminx cannot depend on
atomworks (separate oracle environment, incompatible Python), so this is a small reader for the PDB
subset the oracle cells exercise. Every rule below was read off upstream's own source AND checked
against the sealed feature dumps (P4b run ``6a8ee503``, P4c run ``9a9b75bb``); anything not checked
raises instead of guessing.

The six arrays, each with a leading batch axis of 1 as upstream emits them:

* ``X``  ``(1, L, 37, 3)`` float32, foundry atom order (``N, CA, C, O, CB, CG, ...``); 0 where an atom
  is absent. Coordinates of an atom with several alternate locations come from the FIRST one.
* ``X_m`` ``(1, L, 37)`` bool, atom occupancy STRICTLY above 0.5 (``EncodePottsMPNNNonAtomizedTokens``
  ``occupancy_threshold``; measured on 1BVC: two alternate locations at exactly 0.50 are both
  masked). Coordinates are kept for masked atoms, so compare ``X`` only where ``X_m`` is true.
* ``S``  ``(1, L)`` int64, aminx v6 token indices from the pre-assigned protonation labels
  (``encode_sequence``), not upstream indices.
* ``R_idx`` ``(1, L)`` int32, the residue's index within its chain among the FILE's residues,
  counted BEFORE any residue is dropped. It is an ``arange``, not the residue number (1EL1 is
  numbered ``-1, 1, 2, ...`` and upstream's ``R_idx`` is ``0, 1, 2, ...``). Dropped residues leave
  gaps in it (6m0j: two), which is the only way a gap appears.
* ``chain_labels`` ``(1, L)`` int64, chain ids numbered in order of first appearance.
* ``residue_mask`` ``(1, L)`` bool, all true.

A residue is DROPPED when any of ``N, CA, C, O`` has occupancy ``<= 0.8`` or is absent
(``MaskResiduesWithSpecificUnresolvedAtoms`` with ``occupancy_threshold_backbone = 0.8`` then
``RemoveUnresolvedTokens``; pipelines/potts_mpnn.py:296-303). This is a different threshold from the
0.5 that masks atoms of KEPT residues. On 1CQW it removes the whole C-terminal run 306-309, including
a glycine whose backbone sits at 0.71-0.88.

Arginine naming is canonicalised the way atomworks does (``fix_arginines``): when ``NH1`` is farther
from ``CD`` than ``NH2``, the two COORDINATES are exchanged (occupancies stay with the names). Measured
on 6m0j: 18 atoms in 9 arginines differ without it and are exact with it.

Insertion codes: where a residue number is reused with a different residue NAME (1TPK: ``42 ALA``
then ``42A GLN``), only the LAST name at that number survives (atomworks ``keep_last_residue``, which
groups by ``(chain, res_id, res_name)`` and ignores the insertion code), and ``R_idx`` is counted
after that. Verified on 1TPK (three residues removed, ``R_idx`` exactly ``0..84``). Same name at the
same number with different insertion codes is NOT removed, by the same function; that case has no
sealed cell and is covered by a unit test of the rule only.

Hydrogens need no rule: their atom names are not in the 37-atom layout, so they are ignored.

An input in which EVERY residue is dropped raises, as upstream does ("atom_array cannot be empty",
sealed on 1IFC). Waters and other ``HETATM`` residues are skipped (verified on the ligand cell).

NOT verified, so refused: selenomethionine (``MSE``) conversion, a blank chain id, and any ATOM
residue that is not one of the 20 standard residues or ``UNK`` (upstream atomizes it out of the
residue tokens).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aminx.families.protonpotts_mpnn.sequence import encode_sequence
from aminx.families.protonpotts_mpnn.vocab import THREE_TO_ONE

# Foundry "new" atom order (ProtonPottsMPNN @ 09682ab, token_encodings.py atom_order). Every token
# shares this one 37-atom layout, so atoms are placed by NAME alone, whatever the residue.
ATOM37_ORDER: tuple[str, ...] = (
  "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
  "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
  "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)  # fmt: skip
_ATOM_INDEX = {name: i for i, name in enumerate(ATOM37_ORDER)}
assert len(ATOM37_ORDER) == 37  # noqa: S101

ATOM_OCCUPANCY_THRESHOLD = 0.5
BACKBONE_OCCUPANCY_THRESHOLD = 0.8
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")


class PdbFeatureError(ValueError):
  """The input uses something this reader has not been verified against upstream on."""


@dataclass(frozen=True)
class _Residue:
  chain: str
  number: int
  insertion_code: str
  name: str
  atoms: dict[str, tuple[float, float, float, float]]  # atom name -> x, y, z, occupancy


def _read_residues(path: Path) -> list[_Residue]:
  residues: list[_Residue] = []
  current: _Residue | None = None
  current_key: tuple[str, int, str, str] | None = None
  for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    record = raw[:6]
    if record.startswith("HETATM"):
      if raw[17:20] == "MSE":
        msg = "MSE (selenomethionine) conversion is not verified against upstream; refusing"
        raise PdbFeatureError(msg)
      continue
    if not record.startswith("ATOM"):
      continue
    chain = raw[21]
    if chain == " ":
      msg = "ATOM record with a blank chain id is not supported"
      raise PdbFeatureError(msg)
    name = raw[12:16].strip()
    key = (chain, int(raw[22:26]), raw[26], raw[17:20])
    if key != current_key:
      if key[3] not in THREE_TO_ONE and key[3] != "UNK":
        msg = (
          f"residue {key[3]!r} in an ATOM record is not one of the 20 standard residues or UNK. "
          "Upstream atomizes such residues out of the residue tokens (AtomizeByCCDName, "
          "res_names_to_ignore=STANDARD_AA+UNK); no sealed cell exercises that, so refusing."
        )
        raise PdbFeatureError(msg)
      current = _Residue(chain, key[1], key[2], key[3], {})
      residues.append(current)
      current_key = key
    assert current is not None  # noqa: S101
    if name in current.atoms:
      continue  # alternate locations repeat the atom name: the FIRST one wins
    occupancy = float(raw[54:60]) if raw[54:60].strip() else 1.0
    current.atoms[name] = (float(raw[30:38]), float(raw[38:46]), float(raw[46:54]), occupancy)
  return residues


def _fix_arginine(residue: _Residue) -> None:
  """Exchange NH1/NH2 coordinates when NH1 is the farther from CD (atomworks ``fix_arginines``)."""
  if residue.name != "ARG" or not {"CD", "NH1", "NH2"} <= residue.atoms.keys():
    return
  cd, nh1, nh2 = (np.asarray(residue.atoms[a][:3], dtype=np.float32) for a in ("CD", "NH1", "NH2"))
  if np.linalg.norm(cd - nh1) > np.linalg.norm(cd - nh2):
    first, second = residue.atoms["NH1"], residue.atoms["NH2"]
    residue.atoms["NH1"] = (*second[:3], first[3])
    residue.atoms["NH2"] = (*first[:3], second[3])


def _keep_last_by_number(residues: list[_Residue]) -> list[_Residue]:
  """atomworks ``keep_last_residue``: at one ``(chain, number)``, keep only the last residue NAME.

  Distinct names are taken in order of first appearance; every name but the last is removed, and
  removal is by ``(chain, number, name)``, so a repeated name at the same number is untouched.
  """
  names_at: dict[tuple[str, int], list[str]] = {}
  for residue in residues:
    seen = names_at.setdefault((residue.chain, residue.number), [])
    if residue.name not in seen:
      seen.append(residue.name)
  return [
    residue
    for residue in residues
    if residue.name == names_at[(residue.chain, residue.number)][-1]
  ]


def loaded_residues(path: str | Path) -> list[tuple[str, int, str, str]]:
  """``(chain, number, insertion code, name)`` of the residues ``protonation_labels`` must align to.

  These are the file's residues after duplicate-number removal and BEFORE any backbone-occupancy
  drop (labels for dropped residues are dropped with them).
  """
  return [
    (r.chain, r.number, r.insertion_code, r.name)
    for r in _keep_last_by_number(_read_residues(Path(path)))
  ]


def _backbone_resolved(residue: _Residue) -> bool:
  return all(
    residue.atoms.get(atom, (0.0, 0.0, 0.0, 0.0))[3] > BACKBONE_OCCUPANCY_THRESHOLD
    for atom in BACKBONE_ATOMS
  )


def featurize_pdb(
  path: str | Path,
  protonation_labels: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
  """The six upstream feature arrays for ``path`` (see the module docstring).

  ``protonation_labels`` is one entry per residue of ``loaded_residues(path)`` (``""`` for none),
  exactly as ``encode_sequence`` takes it; labels for dropped residues are dropped with them.
  """
  all_residues = _keep_last_by_number(_read_residues(Path(path)))
  if not all_residues:
    msg = f"{path}: no ATOM residues found"
    raise PdbFeatureError(msg)
  if protonation_labels is not None and len(protonation_labels) != len(all_residues):
    msg = f"{len(protonation_labels)} labels for {len(all_residues)} loaded residues"
    raise PdbFeatureError(msg)
  for residue in all_residues:
    _fix_arginine(residue)

  # R_idx is assigned over ALL loaded residues of a chain, before any backbone drop.
  file_index: list[int] = []
  counts: dict[str, int] = {}
  for residue in all_residues:
    file_index.append(counts.get(residue.chain, 0))
    counts[residue.chain] = counts.get(residue.chain, 0) + 1

  kept = [i for i, residue in enumerate(all_residues) if _backbone_resolved(residue)]
  if not kept:
    msg = f"{path}: every residue has an unresolved backbone (upstream: 'atom_array cannot be empty')"
    raise PdbFeatureError(msg)
  residues = [all_residues[i] for i in kept]
  labels = None if protonation_labels is None else [protonation_labels[i] for i in kept]

  length = len(residues)
  x = np.zeros((length, 37, 3), dtype=np.float32)
  x_m = np.zeros((length, 37), dtype=np.bool_)
  for i, residue in enumerate(residues):
    for atom, (px, py, pz, occupancy) in residue.atoms.items():
      slot = _ATOM_INDEX.get(atom)
      if slot is None:
        continue
      x[i, slot] = (px, py, pz)
      x_m[i, slot] = occupancy > ATOM_OCCUPANCY_THRESHOLD

  chain_order: dict[str, int] = {}
  chain_labels = np.empty(length, dtype=np.int64)
  for i, residue in enumerate(residues):
    chain_labels[i] = chain_order.setdefault(residue.chain, len(chain_order))
  r_idx = np.asarray([file_index[i] for i in kept], dtype=np.int32)

  s = encode_sequence([r.name for r in residues], labels)
  return {
    "X": x[None],
    "X_m": x_m[None],
    "S": s[None],
    "R_idx": r_idx[None],
    "chain_labels": chain_labels[None],
    "residue_mask": np.ones((1, length), dtype=np.bool_),
  }
