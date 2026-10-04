---
title: 'LASEr PDB output: the spec understates upstream''s output pipeline'
description: 'Decision request: upstream LASEr writes PDBs through backbone N-H imputation, titratable-H cleanup with a geometric H-bond detector, and probability B-factors -- none of which the spec names; choose minimal writer vs full parity port'
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261004'
supersedes: ''
backlog_ids: ''
---
# LASEr PDB output: the spec understates upstream's output pipeline

**Status: needs the user's decision.** Nothing has been built for it.

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

