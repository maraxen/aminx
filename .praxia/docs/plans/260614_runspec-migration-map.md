---
task_id: 260614_potts-runspec-xtrax-gates
backlog: RS-1 (#1620)
blocks: RS-2 (#1621)
date: 260614
status: superseded
superseded_by: 260827_runspec-sub-config-membership-rule-and-scaffolding-retirement.md
---

# RunSpec Host-Field Inventory (RS-1) [SUPERSEDED]

> [!WARNING]
> **SUPERSEDED on 2026-08-27** by [260827_runspec-sub-config-membership-rule-and-scaffolding-retirement.md](file:///home/marielle/projects/aminx/.claude/worktrees/wt-20260826-170332/.praxia/docs/decisions/260827_runspec-sub-config-membership-rule-and-scaffolding-retirement.md) and `.praxia/docs/specs/260827_runspec-scaffolding-remediation-migration-map-re-authoring-and-xtrax-transforms-adoption.md` (WS-B).
>
> This document is retained solely for historical auditability. It has drifted significantly from reality:
> - All 22 "already migrated" rows were stale; dead sub-configs (`TiedPositionsConfig`, `BatchingConfig`, `AveragingConfig`, `GridLineageConfig`, `LigandConfig` — 29 fields total) were either never read or already deleted.
> - Listed rows for files that do not exist (e.g. `_sampling_averaged.py`), removed fields (`use_arrayrecord`), and obsolete targets.
> - The human-maintained table format itself was prone to immediate rot upon subsequent refactors.
>
> Field-level membership is now governed by the two-clause membership rule in the decision record above and enforced at import time by `src/aminx/run/_runspec_coverage.py`. Do NOT use this document for scoping or implementation.

Complete map of every `spec.<field>` read in `src/aminx/host/`. Input for RS-2 PlannerTopology work.

## 1. Summary

| Category | Count |
|----------|-------|
| Unique fields read across host/ | 67 |
| Already on RunSpec sub-config | 22 |
| Needs migration to RunSpec | 16 |
| RS-gaps (no current home, RS-2/RS-3 additions needed) | 9 |
| Protein-only (stay flat on facade) | 21 |

Note: some fields appear in multiple files (backbone_noise, temperature) — counted once per unique field name.

## 2. Full Migration Table

| file | line(s) | field_name | already_on_run_spec | target_subconfig | notes |
|------|---------|------------|---------------------|------------------|-------|
| kernel_dispatch.py | 109 | backbone_noise | no | sampling | Sequence[float]; passed to jnp.asarray |
| kernel_dispatch.py | 110 | temperature | no | sampling | Sequence[float]; passed to jnp.asarray |
| kernel_dispatch.py | 125-128 | tie_group_map | YES → tied.tie_group_map | tied | already migrated |
| kernel_dispatch.py | 132-133 | structure_mapping | YES → tied.structure_mapping | tied | already migrated |
| kernel_dispatch.py | 158 | state_weights | YES → averaging.state_weights | averaging | already migrated |
| kernel_dispatch.py | 199,276,368,441 | bias | no | sampling | ArrayLike or None; per-position logit bias |
| kernel_dispatch.py | 171 | use_unified_driver | no | plan | getattr with wrong default=False; RS-5 bug (SamplingSpec default is True) — fixed in 26c9bb5 |
| kernel_dispatch.py | 303,468 | backbone_noise | no | sampling | len(spec.backbone_noise) used for noise_indices |
| kernel_dispatch.py | 540 | compute_pseudo_perplexity | no | sampling | bool flag |
| _sampling_helper.py | 37 | inputs | no | protein | stays flat — loader input |
| _sampling_helper.py | 255 | model_family | YES → ligand.model_family | ligand | already migrated |
| _sampling_helper.py | 269 | ligand_context_path | YES → ligand.context_path | ligand | already migrated |
| _sampling_helper.py | 280 | ligand_conditioning | YES → ligand.ligand_conditioning | ligand | already migrated |
| _sampling_helper.py | 315 | sidechain_conditioning | YES → ligand.sidechain_conditioning | ligand | already migrated |
| _sampling_helper.py | 448,449,485,487 | fixed_mask | no | sampling | float32 mask array; None or ArrayLike |
| _sampling_helper.py | 458,459 | fixed_positions | no | sampling | float32 position mask; None or ArrayLike |
| _sampling_helper.py | 470,471 | fixed_tokens | no | sampling | int32 token array; None or ArrayLike |
| runner.py | 164,312,492,742 | output_h5_path | partial (io.sink_kind derived) | io | should be stored directly as io.output_h5_path; currently only re-derived via _infer_sink_kind |
| runner.py | 170,202,232,331,352,404-438 | return_logits | no | sampling | bool; SamplingSpec/ScoringSpec |
| runner.py | 225,248 | grid_mode | YES → grid.grid_mode | grid | already migrated |
| runner.py | 341,527,741 | random_seed | no | sampling | int; used for PRNGKey init |
| runner.py | 333,353,406,412,418,442 | return_decoding_orders | no | sampling | bool; ScoringSpecification only |
| runner.py | 400 | multi_state_strategy | YES → multistate.combine_strategy | multistate | already migrated |
| runner.py | 518,545 | inspection_features | no | new | RS-gap: InspectionSpecification only |
| runner.py | 631,668 | distance_matrix | no | new | RS-gap: InspectionSpecification only |
| runner.py | 632,634,636,646 | distance_matrix_method | no | new | RS-gap: InspectionSpecification only |
| runner.py | 649,671 | cross_input_similarity | no | new | RS-gap: InspectionSpecification only |
| runner.py | 681,683,685,688 | similarity_metric | no | new | RS-gap: InspectionSpecification only |
| runner.py | 326 | sequences_to_score | no | new | RS-gap: ScoringSpecification only |
| runner.py | 714 | combine | no | new | RS-gap: JacobianSpecification only |
| runner.py | 718,770,817,824 | jacobian_mode | no | new | RS-gap: JacobianSpecification only |
| runner.py | 718,736 | compute_apc | no | new | RS-gap: JacobianSpecification only |
| runner.py | 796 | apc_residue_batch_size | YES → batching.apc_residue_batch_size | batching | already migrated |
| _sampling_averaged.py | 32 | random_seed | no | sampling | legacy deprecated path |
| _sampling_averaged.py | 32 | num_samples | no | sampling | count of samples to generate |
| _sampling_averaged.py | 34,160 | temperature | no | sampling | legacy deprecated path |
| _sampling_averaged.py | 100,161,180 | average_encoding_mode | YES → averaging.average_encoding_mode | averaging | already migrated |
| _sampling_averaged.py | 150,151 | structure_mapping | YES → tied.structure_mapping | tied | already migrated |
| _sampling_averaged.py | 158 | backbone_noise | no | sampling | legacy deprecated path |
| _sampling_averaged.py | 159 | noise_batch_size | YES → batching.noise_batch_size | batching | already migrated |
| _sampling_averaged.py | 173 | bias | no | sampling | legacy deprecated path |
| _sampling_averaged.py | 176 | multi_state_strategy | YES → multistate.combine_strategy | multistate | already migrated |
| _sampling_averaged.py | 177 | multi_state_temperature | YES → tied.multi_state_temperature | tied | already migrated |
| _sampling_averaged.py | 223 | compute_pseudo_perplexity | no | sampling | legacy deprecated path |
| prep.py | 37,39,42 | checkpoint_registry_path | no | protein | loader config; stays flat |
| prep.py | 39,49,55,56,132,134 | checkpoint_id | no | protein | loader config; stays flat |
| prep.py | 48,55,88,94,111,112 | model_family | YES → ligand.model_family | ligand | already migrated |
| prep.py | 96 | chain_id | no | protein | loader config; stays flat |
| prep.py | 97 | model | no | protein | loader model number; stays flat |
| prep.py | 98 | altloc | no | protein | stays flat |
| prep.py | 99 | topology | no | protein | stays flat |
| prep.py | 103 | inputs | no | protein | stays flat |
| prep.py | 104 | batch_size | YES → batching.batch_size | batching | already migrated |
| prep.py | 105 | foldcomp_database | no | protein | stays flat |
| prep.py | 107 | pass_mode | YES → tied.pass_mode | tied | already migrated |
| prep.py | 108 | use_preprocessed | no | protein | stays flat |
| prep.py | 109 | preprocessed_index_path | YES → io.manifest_path | io | already migrated |
| prep.py | 110 | split | no | protein | stays flat |
| prep.py | 111 | use_electrostatics | no | protein | stays flat |
| prep.py | 112 | estat_noise | no | protein | stays flat |
| prep.py | 113 | estat_noise_mode | no | protein | stays flat |
| prep.py | 114 | use_vdw | no | protein | stays flat |
| prep.py | 115 | vdw_noise | no | protein | stays flat |
| prep.py | 116 | vdw_noise_mode | no | protein | stays flat |
| prep.py | 117 | max_length | no | protein | stays flat |
| prep.py | 118 | truncation_strategy | no | protein | stays flat |
| prep.py | 124,125 | model_local_path | no | protein | stays flat |
| prep.py | 129,130 | cache_path | partial (io.output_dir) | io | cache_path feeds JAX compilation cache; partially captured but not directly stored |
| prep.py | 135,138,141,142 | model_weights | no | protein | loader config; stays flat |
| prep.py | 136,141 | model_version | no | protein | loader config; stays flat |
| streaming.py | 60,65 | use_arrayrecord | no | io | feeds io.sink_kind but not stored directly; candidate for io.use_arrayrecord |
| streaming.py | 60,65,117 | campaign_mode | YES → grid.campaign_mode | grid | already migrated |
| streaming.py | 87,248,398 | grid_mode | YES → grid.grid_mode | grid | already migrated |
| streaming.py | 88 | model_family | YES → ligand.model_family | ligand | already migrated |
| streaming.py | 89 | ligand_conditioning | YES → ligand.ligand_conditioning | ligand | already migrated |
| streaming.py | 90 | sidechain_conditioning | YES → ligand.sidechain_conditioning | ligand | already migrated |
| streaming.py | 141,215,352 | return_logits | no | sampling | bool |
| streaming.py | 166 | backbone_noise | no | sampling | len() used for metadata |
| streaming.py | 167 | temperature | no | sampling | len() used for metadata |
| streaming.py | 310 | run_spec.multistate.n_states | YES (is RunSpec read) | multistate | ONLY RunSpec read in non-training host code today |
| _sampling_grid_lineage.py | 14 | grid_mode | YES → grid.grid_mode | grid | already migrated |
| _sampling_grid_lineage.py | 16 | sample_count | YES → grid.sample_count | grid | already migrated |
| _sampling_grid_lineage.py | 16 | num_samples | no | sampling | fallback when sample_count is None |
| _sampling_grid_lineage.py | 20 | sample_start | YES → grid.sample_start | grid | already migrated |
| _sampling_grid_lineage.py | 24 | chunk_id | YES → grid.chunk_id | grid | already migrated |
| _sampling_grid_lineage.py | 28 | job_id | YES → grid.job_id | grid | already migrated |
| _sampling_grid_lineage.py | 94,111 | model_family | YES → ligand.model_family | ligand | already migrated |
| _sampling_grid_lineage.py | 95,112 | ligand_conditioning | YES → ligand.ligand_conditioning | ligand | already migrated |
| _sampling_grid_lineage.py | 96,113 | sidechain_conditioning | YES → ligand.sidechain_conditioning | ligand | already migrated |
| _sampling_grid_lineage.py | 97,114 | multi_state_strategy | YES → multistate.combine_strategy | multistate | already migrated |
| _sampling_grid_lineage.py | 98-99,115-116 | temperature, backbone_noise | no | sampling | canonical string for lineage hash |
| _sampling_grid_lineage.py | 135 | random_seed | no | sampling | key derivation |
| plan.py | 194 | num_samples | no | sampling | resolve_target_samples fallback |
| plan.py | 227,228 | samples_chunk_size | YES → batching.samples_chunk_size | batching | already migrated |
| campaign.py | 71,72 | temperature, backbone_noise | no | sampling | list conversion for campaign rows |
| campaign.py | 82,118 | model_family | YES → ligand.model_family | ligand | already migrated |
| campaign.py | 83,102,124 | ligand_conditioning | YES → ligand.ligand_conditioning | ligand | already migrated |
| campaign.py | 84,103 | sidechain_conditioning | YES → ligand.sidechain_conditioning | ligand | already migrated |
| campaign.py | 85,99,121 | multi_state_strategy | YES → multistate.combine_strategy | multistate | already migrated |
| campaign.py | 104 | checkpoint_id | no | protein | campaign manifest; stays flat |

## 3. Already Migrated (22 fields)

These fields are already readable via `spec.run_spec.<subconfig>.<field>` and should be the migration target for RS-6 hot-path callers:

| field | RunSpec path |
|-------|--------------|
| tie_group_map | tied.tie_group_map |
| structure_mapping | tied.structure_mapping |
| pass_mode | tied.pass_mode |
| multi_state_temperature | tied.multi_state_temperature |
| state_weights | averaging.state_weights |
| average_encoding_mode | averaging.average_encoding_mode |
| average_node_features | averaging.average_node_features |
| model_family | ligand.model_family |
| ligand_context_path | ligand.context_path |
| ligand_conditioning | ligand.ligand_conditioning |
| sidechain_conditioning | ligand.sidechain_conditioning |
| grid_mode | grid.grid_mode |
| campaign_mode | grid.campaign_mode |
| sample_count | grid.sample_count |
| sample_start | grid.sample_start |
| chunk_id | grid.chunk_id |
| job_id | grid.job_id |
| multi_state_strategy | multistate.combine_strategy |
| batch_size | batching.batch_size |
| apc_residue_batch_size | batching.apc_residue_batch_size |
| noise_batch_size | batching.noise_batch_size |
| samples_chunk_size | batching.samples_chunk_size |
| preprocessed_index_path | io.manifest_path |

## 4. Needs Migration (16 fields → new SamplingConfig + io additions)

Grouped by target sub-config:

### target: sampling (new SamplingConfig sub-config for RS-2)

| field | type | primary callers |
|-------|------|-----------------|
| backbone_noise | Sequence[float] | kernel_dispatch.py, streaming.py, _sampling_averaged.py, _sampling_grid_lineage.py, campaign.py |
| temperature | Sequence[float] | kernel_dispatch.py, streaming.py, _sampling_averaged.py, _sampling_grid_lineage.py, campaign.py |
| num_samples | int | _sampling_averaged.py, _sampling_grid_lineage.py, plan.py |
| random_seed | int | runner.py, _sampling_averaged.py, _sampling_grid_lineage.py |
| bias | ArrayLike or None | kernel_dispatch.py, _sampling_averaged.py |
| fixed_mask | ArrayLike or None | _sampling_helper.py |
| fixed_positions | ArrayLike or None | _sampling_helper.py |
| fixed_tokens | ArrayLike or None | _sampling_helper.py |
| compute_pseudo_perplexity | bool | kernel_dispatch.py, _sampling_averaged.py |
| return_logits | bool | runner.py, streaming.py |
| return_decoding_orders | bool | runner.py |
| use_arrayrecord | bool | streaming.py → feeds io.sink_kind; could fold into IOConfig instead |

### target: plan (new PlannerTopology sub-config for RS-2)

| field | type | primary callers |
|-------|------|-----------------|
| use_unified_driver | bool | kernel_dispatch.py (getattr default=False — RS-5 bug, fixed in 26c9bb5) |

### target: io (additions to existing IOConfig)

| field | type | notes |
|-------|------|-------|
| output_h5_path | Path or None | Currently only re-derived via _infer_sink_kind; should be stored directly |
| cache_path | Path or None | Partially via _infer_output_dir; feeds JAX compilation cache dir |

## 5. RS-Gaps (9 fields — no current RunSpec home)

All task-specific, no obvious generic sub-config. Candidate for RS-2/RS-3 task-scoped additions or thin `TaskConfig` wrapper:

| field | spec class | notes |
|-------|-----------|-------|
| sequences_to_score | ScoringSpecification | list of str; scoring-only |
| inspection_features | InspectionSpecification | list of feature name literals |
| distance_matrix | InspectionSpecification | bool flag |
| distance_matrix_method | InspectionSpecification | literal enum |
| cross_input_similarity | InspectionSpecification | bool flag |
| similarity_metric | InspectionSpecification | literal enum (rmsd, tm-score, cosine, etc.) |
| combine | JacobianSpecification | bool — combine Jacobians across structures |
| jacobian_mode | JacobianSpecification | literal: categorical or reverse |
| compute_apc | JacobianSpecification | bool — APC correction for Frobenius norm |

Recommendation: defer to RS-7 (scoring) and RS-8 (inspection). A thin `ScoringConfig`, `InspectionConfig`, `JacobianConfig` could be added to RunSpec in RS-2 with static fields. Mandatory for complete AC-RS-1a coverage.

## 6. Protein-Only Fields (21 — stay flat on facade forever)

These belong to the proxide loader / model weight resolution layer. They must NOT migrate to RunSpec (would couple loader config into the PyTree and force recompiles on loader changes):

`inputs`, `checkpoint_registry_path`, `checkpoint_id`, `model_weights`, `model_version`, `model_local_path`, `chain_id`, `model`, `altloc`, `topology`, `foldcomp_database`, `use_preprocessed`, `split`, `use_electrostatics`, `estat_noise`, `estat_noise_mode`, `use_vdw`, `vdw_noise`, `vdw_noise_mode`, `max_length`, `truncation_strategy`

Note: `checkpoint_id` also appears in campaign.py manifest rows — campaign manifest format uses full `spec_json` not RunSpec, so no migration pressure there (per spec wire-format contract: AS-RS2).

## Key finding: only one RunSpec read site today

`streaming.py:310` reads `spec.run_spec.multistate.n_states` — the ONLY place in non-training host code that currently goes through RunSpec rather than flat spec fields. All other 65+ field reads are flat. RS-6 will add `ruff banned-api` to prevent new flat reads after migration.

## RS-5 bug confirmed (fixed in 26c9bb5)

`kernel_dispatch.py:171`: `getattr(spec, 'use_unified_driver', False)` — default was `False`.
`SamplingSpecification.use_unified_driver` defaults to `True`.
When spec has the attribute, the right value is used. The getattr default only fires for non-SamplingSpec callers — but all callers pass SamplingSpec today, so this was latent. Fixed in commit `26c9bb5`: default corrected to `True`.

## Open Questions (for RS-2)

1. Should `use_arrayrecord` be stored directly on `IOConfig` (io.use_arrayrecord) or remain only as a `build_run_spec` input that sets `io.sink_kind`?
2. Should `cache_path` be stored directly on `IOConfig` alongside `output_dir`, or does the current `_infer_output_dir` logic cover all callers?
3. Should RS-gaps (`sequences_to_score`, `inspection_features`, etc.) get thin task-scoped sub-configs in RS-2 (`ScoringConfig`, `InspectionConfig`, `JacobianConfig`) or defer to RS-7/RS-8?
4. `_sampling_averaged.py` is fully deprecated — RS-6 lint should target this file for removal, not migration. Is there an explicit removal timeline?
5. `prep.py` reads 26 distinct spec fields but only 5 are already on RunSpec; the remaining 21 are protein-only. Should `prep.py` be explicitly exempted from the RS-6 flat-field ban?
6. `streaming.py:310` is the only current RunSpec read site — should this be used as the migration template/exemplar for RS-6?
