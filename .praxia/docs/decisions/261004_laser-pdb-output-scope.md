---
title: 'LASEr PDB output: the spec understates upstream''s output pipeline'
description: 'Decision request: upstream LASEr writes PDBs through backbone N-H imputation, titratable-H cleanup with a geometric H-bond detector, and probability B-factors -- none of which the spec names; choose minimal writer vs full parity port'
status: accepted
task_id: 260929_potts-laser-xtrax-compose
date: '261004'
supersedes: ''
backlog_ids: ''
---
# LASEr PDB output: the spec understates upstream's output pipeline

**Status: decided 2026-11-23. Option C, with the writer living in proxide** (see "Decision" at the end).
Nothing has been built yet.

## Where things stand

On side branch `wt/260929-laser-rotamers-wave`: aminx now ports upstream's
non-rotatable hydrogen placement (`2f803547`; the `laser_rotamers` wave passes
315/315 incl. hydrogens) and the LaserDriver `sample` purpose (`54216d92`),
which returns `sidechain_coords` (protonated when the checkpoint's
`build_hydrogens` is true). **No full-atom PDB writer exists anywhere in
`src/aminx`.**

The spec (5.6) asks for "PDB (hydrogens per checkpoint `build_hydrogens`)" and
says nothing more.

## What upstream actually does before writing a PDB

`run_inference.py:745-759` (and the batch/tied equivalents):

1. `build_rotamers(..., add_nonrotatable_hydrogens=True)` -- **ported.**
2. `impute_backbone_nh_coords(full_atom_coords, seq, phi)` --
   `utils/build_rotamers.py:388`: backbone amide H from phi. Not ported.
3. `cleanup_titratable_hydrogens(..., hbond_network_detector)` --
   `utils/build_rotamers.py:411`, ~150 lines: cysteine handling (disulfide
   distance 2.5 A) and histidine tautomer choice by H-bond distance, using
   `RigorousHydrogenBondNetworkDetector` (`utils/hbond_network.py`, ~390
   lines). The detector is **geometric** -- only `register_buffer` lookup
   tables, no learned weights -- so it is portable without new oracle weights,
   but it is a real port.
4. `output_protein_structure(..., sampled_probs)` -- per-residue **B-factor =
   the sampled token's probability**, residue identifiers from the input, then
   the input ligand appended; `pr.writePDB`.

None of 2-4 appears in the spec.

## Options

- **A. Minimal writer now.** Write heavy atoms + non-rotatable hydrogens with
  input residue identifiers and the ligand, document 2-4 as known divergences
  (no backbone N-H, no titratable-H cleanup, B-factors 0 or the probability).
  Small; aminx's PDBs would differ from upstream's in exactly the documented
  hydrogens.
- **B. Full parity port.** Port 2-4 (~600 upstream lines), with a new oracle
  wave over upstream's final full-atom coordinates (dumpable from the existing
  oracle env; the detector needs no weights). Largest; makes aminx's PDB output
  match upstream's atom for atom.
- **C. A now, B as a follow-up epic** -- A unblocks users of `sample`; B is
  scheduled separately with its own wave.

## Recommendation

C. The spec only asks for "PDB with hydrogens", which A delivers honestly with
its divergences named; the parity port is real work that deserves its own
pre-registered wave rather than being folded into this sprint at the end.

Related open decisions: `261003_laser-score-tier3-f32-design.md`,
`261004_sample-dist-pilot-fixtures-and-low-t.md`.


## Decision (2026-11-23, user)

**C, with one change: the full-atom PDB writer and its configurability belong in proxide, not aminx.**

- **Writer: proxide debt #3583.** A configurable writer: per-residue or per-atom B-factors and occupancy, input
  residue identifiers (chain, resnum, icode), optional HETATM ligand atoms, hydrogens, correct atom-name columns, and a
  loud failure on missing data. proxide's current `write_pdb` (`src/proxide/io/writing.py:87-160`) has none of these:
  it hard-codes B-factor 0.00, writes no HETATM, no insertion codes, and silently falls back to CA/UNK.
- **Source to port, not reinvent:** aminx local tag `archive/wt/260929-laser-rotamers-wave`, commit `095a49e0`.
  Its `src/aminx/io/laser_pdb.py` (314 lines, with ligand HETATM, per-residue B-factors and upstream atom names) and
  `tests/io/test_laser_pdb.py` move to proxide. Do not merge that commit into aminx.
- **B-factor: the sampled token's probability per residue**, as upstream LASErMPNN does. The aminx LASEr sample path
  passes it to the proxide writer. This removes one of A's documented divergences.
- **aminx side (debt #3565):** once #3583 lands, add a thin call from the LASEr sample path, honouring
  `output_fasta` / `output_fasta_only`, into the proxide writer. Document the remaining divergences: no backbone N-H,
  no titratable-H cleanup.
- **B (the parity port of steps 2-3 plus a new oracle wave) stays a follow-up epic in aminx**, pre-registered
  separately. It is about hydrogen placement, not writing.

### Where things stand, updated

Commits `2f803547` (non-rotatable hydrogen placement, `families/laser_mpnn/featurize.py:795 build_rotamers`) and
`54216d92` (the LaserDriver `sample` purpose) are now on main; the "Where things stand" section above predates their merge.
