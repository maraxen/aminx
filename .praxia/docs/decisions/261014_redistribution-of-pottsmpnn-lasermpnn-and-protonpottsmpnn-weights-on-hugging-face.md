---
title: Redistribution of PottsMPNN, LASErMPNN and ProtonPottsMPNN weights on Hugging Face
description: Upstream licence findings (all MIT), the evidence-backed checkpoint set, open items and the redistribution decision for the 0.2.1 hub weights
status: accepted
task_id: 261014_hf-family-weights
date: '261014'
supersedes: ''
backlog_ids: ''
---
# Redistribution of PottsMPNN, LASErMPNN and ProtonPottsMPNN weights on Hugging Face

**Status: accepted 2026-10-14** by the reviewer (Marielle Russo), who approved the record as written in chat.
`redistribution = allowed`, conditional on shipping the notices in `NOTICE-weights.md`. The open items below
(training-data terms, debt #2698) were accepted as open, not resolved. Plan: `~/.claude/plans/parallel-honking-ocean.md`.

## Upstream licences (read from the pinned local clones under `~/repos`)

| Upstream | Commit | Licence | Copyright | Weights in scope? |
|---|---|---|---|---|
| PottsMPNN, <https://github.com/KeatingLab/PottsMPNN> | `0cb0a58874e373d664114f1735deab77614e4ab5` | MIT (`LICENSE`) | 2022 Justas Dauparas; 2026 Foster Birnbaum, Amy E. Keating | Weights ship in the repo under its MIT. The text says "Software and associated documentation files" and does not name weights. A Zenodo deposit also exists (DOI 10.5281/zenodo.18274667). |
| LASErMPNN, <https://github.com/polizzilab/LASErMPNN> | `e70f2c6d765416f7e29d51bfd6d4e08496438878` | MIT text, no SPDX header (`LICENSE`) | 2025 Benjamin Fry, Nicholas Polizzi | Weights ship in `model_weights/` under the repo MIT. Not named in the text. |
| ProtonPottsMPNN | `09682abf` (the oracle pin in `scripts/parity/convert_protonpotts_checkpoint.py`; the local clone has no `VENDOR_PIN.toml`) | MIT, scoped in `LICENSE` to include `checkpoints/`; `foundry/` is a verbatim IPD rc-foundry copy under BSD-3 | 2026 Christian P. Jacobsen | **Yes, explicitly** (`checkpoints/` is named). The v6 checkpoint is `checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt`. |

MIT permits redistribution and modification, so a `.pt`/`.ckpt` to `.eqx` conversion is permitted. The one
condition is that the copyright and permission notice travel with the copies (`NOTICE-weights.md`, and the
HF model card).

## Evidence-backed publish set

Rule: publish a checkpoint only if an in-repo parity test iterates it against a sealed upstream oracle.
Sources: `tests/port/reference/a1_potts/oracle_manifest.toml`, `~/projects/aminx-oracles/dumps/b3_laser_owned/oracle_manifest.toml`
(titanix), `scripts/parity/dump_{potts,laser}_oracles.py`, `tests/port/test_laser_*.py`.

| checkpoint id (proposed) | upstream file | upstream sha256 | status |
|---|---|---|---|
| `pottsmpnn_vanilla_20` | `vanilla_model_weights/pottsmpnn_20.pt` | `77e797fd...aa5c` | publish |
| `pottsmpnn_vanilla_30` | `vanilla_model_weights/pottsmpnn_30.pt` | `0491d76e...b92c` | publish |
| `pottsmpnn_sol_20` | `soluble_model_weights/sol_pottsmpnn_20.pt` | `f203a697...607f` | publish |
| `pottsmpnn_sol_30` | `soluble_model_weights/sol_pottsmpnn_30.pt` | `32533191...34f4` | publish |
| `pottsmpnn_ft` | `ft_model_weights/potts_ft.pt` | `b672ff4e...c71e` | publish |
| `lasermpnn_nothing_heldout` | `model_weights/laser_weights_0p1A_nothing_heldout.pt` | `304fe02a...1173` | publish |
| `lasermpnn_noise_ligandmpnn_split` | `model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt` | `4c76f879...1e1f` | publish |
| `lasermpnn_soluble_65000` | `model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt` | `5046dcfd...a41` | publish |
| `protonpottsmpnn_v6_30` | `checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt` | to record at conversion | publish |

Excluded, no parity evidence: `pottsmpnn_sol_msa_20` and `pottsmpnn_vanilla_msa_20` (present in `weights/pottsmpnn/`,
absent from every oracle manifest), `pretrained_ligand_encoder_weights.pt` (not a LASErMPNN checkpoint, see
`tests/families/laser_mpnn/test_weight_keys.py`), upstream `ca_model_weights/` and `proteinmpnn_compatible_model_weights/`.

Discrepancy to resolve in step 3: `PottsMPNN/VENDOR_PIN.toml` lists `proteinmpnn_compatible_model_weights/pottsmpnn_20.pt`
(`28b1620a...`) while the oracle graded `vanilla_model_weights/pottsmpnn_20.pt` (`77e797fd...`). These are different files.
The published bytes must be derived from the file the oracle used. The final upload table will carry the full sha256s,
the converted-file sha256s and the bathos run ids.

## Not verifiable from the repos (reviewer to accept or investigate)

1. Training-data terms (PDB, AFDB, ligand and protonated-PDB datasets for LASErMPNN; the pKa and AFDB data behind
   ProtonPottsMPNN v6). An MIT weights licence does not settle dataset terms.
2. Potts and LASErMPNN rely on the repo-level MIT covering bundled weights (normal reading, not stated in the text).
3. ProtonPottsMPNN: the clone has no pin file; the converter records `09682abf`. Confirm that commit contains the
   `LICENSE` read above and record the upstream URL and the checkpoint sha256 at conversion time.
4. Courtesy, not a legal requirement: tell the three upstream groups and cite their papers on the model card.

## Build record

- Run: bathos `354f98dc-0cbb-4f1a-b9f3-210dd2ddd1d2`, `scripts/weights/build_family_weights.py` at `06cb071c`, branch
  `wt/261014-hf-family-weights`, clean tree; cool-tier record `status completed, outcome pass, exit_code 0`.
  Sidecar `scripts/weights/build_family_weights.bth.toml` was committed before the first run and amended once before any
  run completed (ProtonPotts source hash added: it is named in the graded browser-parity sidecars).
- First attempt (same script, `--proton-checkpoint`) errored on ProtonPotts: unpickling the `.ckpt` needs `omegaconf`,
  absent from the gate-host env. The graded runs used the exported state file `~/scratch/v6_state.npz` (its `.json`
  records checkpoint sha256 `a3987225...fe7b` and pin `09682abf`); the second run built from it. Both are on record.
- Result: 9 of 9 built; every artifact reloads through the driver loader leaf-for-leaf identical to the converted model;
  second build byte-identical; all 9 source hashes match their recorded values; the flipped-byte control changed both
  the sha256 and the reloaded model for every artifact.
- Pinned hashes (artifact and manifest) are in `src/aminx/io/family_weights.json`.
- All three LASErMPNN checkpoints set `build_hydrogens = True`; before this change `_load_eqx` would have loaded such an
  artifact with it False. Fixed by reading the manifest (`tests/io/test_family_weights.py`).
- End to end on titanix (CPU): `aminx run --checkpoint-id <id> score ...` with only `AMINX_WEIGHTS_DIR` set resolves
  PottsMPNN (`energy`, 1ubq) and LASErMPNN (`nll`, 1ubq) and is bitwise equal to the explicit `--model-local-path` run.
- The tracked `weights/pottsmpnn/*.eqx.zst` are not these artifacts and do not load as `PottsMPNN` (debt #2697).

## Decision

`redistribution = allowed` for the nine checkpoints above, with notices shipped. Confirmed by the reviewer 2026-10-14.

## Follow-Up Family Additions (2026-10-27: Caliby, FAMPNN, ThermoMPNN / ThermoMPNN-D)

All published checkpoints below are licensed under MIT (see `NOTICE-weights.md`) and verified against upstream PyTorch checkpoints:
- **Caliby** (`ProteinDesignLab/caliby`, MIT): `caliby`, `soluble_caliby`, `soluble_caliby_v1`, `caliby_distill`, `soluble_caliby_distill`, `caliby_packer_000`, `caliby_packer_010`, `caliby_packer_030`
- **FAMPNN** (`richardshuai/fampnn`, MIT): `fampnn_0_0`, `fampnn_0_3`, `fampnn_0_3_cath`
- **ThermoMPNN & ThermoMPNN-D** (`Kuhlman-Lab/ThermoMPNN`, `Kuhlman-Lab/ThermoMPNN-D`, MIT): `thermo_mpnn_v1_default`, `thermo_mpnn_v2_ens1`, `thermo_mpnn_v2_ens2`, `thermo_mpnn_v2_ens3`, `thermo_mpnn_d_ens1`, `thermo_mpnn_d_ens2`, `thermo_mpnn_d_ens3`
