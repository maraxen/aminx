---
title: "tev_design AR state campaign: three data findings from the library EDA prototype"
task_id: "261106_campaign-eda-respec"
date: "2026-11-06"
status: "open — for the tev_design owner; nothing in tev_design was changed"
---

# tev_design AR state campaign: three findings

Found while prototyping knob-faceted library EDA (aminx `scripts/analysis/campaign_library_eda_prototype_v2.py`,
bathos run `6d38ebaf`, outcome pass) over
`tev_design/outputs/results/state_distinguishability_ar_manifest.json` (campaign
`260717_state-distinguishability-ar-verification`, 216 rows, finished 2026-07-21 on titanix). The campaign ran
aminx 0.1.0a19 (wheel) and proxide 0.1.0a11 (tev_design `uv.lock` at `1f40e01a`). These may change that project's
conclusions; they are not fixed here. Filed as a doc because the praxia debt DB was unreachable (2026-11-06).

## 1. All four LigandMPNN arms sampled near-chance libraries

| Arm | Native recovery | vs shuffled native | Composition entropy (bits) |
|---|---|---|---|
| ProteinMPNN | 0.412 | 0.061 | 4.03 |
| SolubleMPNN | 0.392 | 0.058 | 4.08 |
| LigandMPNN (4 arms) | 0.126-0.138 | 0.052-0.057 | 3.28-3.44 |

LigandMPNN tokens skew to L (about 0.25) and A (about 0.17); per-residue NLL of samples under their own stored
logits is about twice the other arms' (1.5-2.1 vs 0.77-1.03). aminx 0.1.0a19 (bumped 2026-07-19) predates
`82c4ea9b` "read the backbone O atom by name in LigandMPNN features (#2326)" (2026-10-01). Plausible cause, not
tested: re-sample one state with a fixed build and compare.

## 2. The catalytic_triad pins land on the wrong residues in five of nine states

`fixed_mask` indices are 38/73/143 in every state's store. Residue labels from the campaign's own parser:

| States | Residues pinned to H/D/C |
|---|---|
| reac1_TS, reac1_prod, reac1_reac | H46, D81, C151 (correct) |
| 1LVB | H53, D88, A158 (correct; crystal numbering +7, C151A mutant) |
| 1LVM | S39, V74, D144 (wrong) |
| reac2_Int, reac2_TS, reac2_prod, reac2_reac | K45, R80, Q150 (wrong; off by one) |

In those five states the real triad was free to be redesigned and three other residues were forced to H/D/C, which
can make states look more different than they are.

## 3. State JSD compares non-corresponding positions

`analyze_state_distinguishability_ar.py` averages per-position JSD over all 512 indices. Targets are 221-467
residues long, so padding (sampled, near-uniform logits, about 4.0 bits) is included, and multi-chain states
(1LVB 214/10/214/10, 1LVM 229/219/6/6/7) put different chains at the same index. The EDA prototype reproduced its
recorded numbers exactly, which shows the reader is faithful, not that the method is sound.

## Also relevant to anyone re-reading these stores

proxide 0.2.0 (current aminx main) reorders 1LVB's chains to 214/214/10/10; re-parsing with it misaligns every
residue after index 214. Use residue tables built with the campaign's parser
(`outputs/campaign_eda_prototype_v2/residue_tables/residue_tables.json` in the aminx prototype worktree, run `ca68e784`).
