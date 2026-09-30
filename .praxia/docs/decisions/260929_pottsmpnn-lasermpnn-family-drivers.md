---
title: PottsMPNN and LASErMPNN are xtrax-composed FamilyDrivers; 260605 governs only aminx.potts
decision_id: 260929_pottsmpnn-lasermpnn-family-drivers
date: 2026-09-29
status: Accepted
decision_type: architectural
amends: 260605_potts-parallel-not-stageset
relates_to: 260605_potts-parallel-not-stageset
---

## Status: Accepted 2026-09-29

This decision amends [`260605_potts-parallel-not-stageset`](260605_potts-parallel-not-stageset.md) by narrowing its scope. It does not replace that decision for the TRW structure model, and it does not reopen Option III.

## Decision

`260605_potts-parallel-not-stageset` governs the TRW structure model (`aminx.potts.*`). That decision is unchanged, and its lint is unchanged: `aminx.potts` still must not import `aminx.inference.decode`, `aminx.host.plan`, `aminx.types.stages`, or `aminx.inference.logits`. The existing `aminx.potts.designer` exemption stands.

PottsMPNN sampling and energy purposes run in a FamilyDriver whose decode stage reuses aminx `DecoderLayer` modules. Its nll/logits scoring and jacobian/inspect run the unmodified MPNN paths on the embedded `Aminx` core. 260605 Option III (EncoderOutput widening) stays rejected.

LASErMPNN is a FamilyDriver with its own encoder-output and result types. MPNN contracts are not widened. Its own driver does not exempt it from composition: `LaserDriver` is xtrax-composed under the same normative definition and L-DRV lint as PottsMPNN (eqx stage modules; samples, temperatures, ligand_atoms, focus_residue, decoding_order, and dropout_seed as xtrax axes via `BatchPlanner`; proofread and tied reductions as `AxisBoundary` Fuses; export via xtrax `ZarrStagingSink`).

New modules: `aminx.model.potts_mpnn`, `aminx.families.potts_mpnn`, `aminx.model.laser`, `aminx.families.laser`, `aminx.host.family_driver`, `aminx.host.family_runner`. L-DRV rule R5 forbids an import of `aminx.potts` from those family, model, and host family modules.

`src/aminx/potts/model.py` carries a one-line disambiguation pointer: this module is the TRW structure model (`aminx.potts`); PottsMPNN is `aminx.families.potts_mpnn` (this ADR).

## Lint

The ruff TID251 entries whose message is "PottsModel is parallel, not a stageset consumer" stay in `[tool.ruff.lint.flake8-tidy-imports.banned-api]` and still apply to `src/aminx/potts`. `per-file-ignores` exempts `src/aminx/families/potts_mpnn`, `src/aminx/families/laser_mpnn` (the on-disk package for `aminx.families.laser`), `src/aminx/host/family_driver.py`, `src/aminx/model/potts_mpnn`, and `src/aminx/model/laser`. TID251 is a single rule code, so those files also skip the unrelated xtrax deep-import bans; those bans stay in force for `aminx.potts` and the rest of the tree. `src/aminx/host/family_runner.py` keeps its line-level `# noqa: TID251` on the `aminx.host.plan` import, because a file-level ignore would make that suppression unused.
