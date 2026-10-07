---
name: aminx-campaigns
description: Plan and run an aminx design campaign from a manifest. Covers plan, worker, run, gates, ramp-plan, ramp-evaluate, cluster use, and git provenance fields.
---

# aminx-campaigns

A campaign is a JSON manifest of sampling rows, then one or more processes that execute those rows. Single-structure sampling options are in `aminx-sampling`. There is no MCP campaign tool. The verbs are `aminx campaign`.

`plan` and `ramp-plan` take comma-separated local paths on `--inputs` (one option, not the repeatable `run` flag). Fixed arms, state-weight profiles, bias, and tie groups are applied while the manifest is written.

## Verbs

```bash
aminx campaign plan --inputs structure.pdb --campaign-id demo --manifest-path manifest.json --output-root out --designs-per-library-type 2 --samples-chunk-size 8
```

```bash
aminx campaign worker --manifest-path manifest.json --row-index 0
```

```bash
aminx campaign run --manifest-path manifest.json --continue-on-error
```

```bash
aminx campaign gates --manifest-path manifest.json --report-path gates.json
```

```bash
aminx campaign ramp-plan --inputs structure.pdb --campaign-id demo --manifest-dir manifests --output-root out --stage-designs-per-library-type 2,8 --samples-chunk-size 8
```

```bash
aminx campaign ramp-evaluate --report-path gates.json
```

```bash
aminx campaign adopt-legacy --manifest-path manifest.json --dry-run
```

- `plan` writes one manifest. Required: `--inputs`, `--campaign-id`, `--manifest-path`, `--output-root`, `--designs-per-library-type`, `--samples-chunk-size`.
- `worker` executes one row, selected by `--row-index` or `--row-hash`.
- `run` executes rows sequentially. Repeat `--row-hash` to filter. `--summary-path` writes the summary. Exit code 1 when `failed_rows` is greater than 0.
- `gates` writes a gate report. Exit code 2 when `promote` is false. `--allow-missing-rerun` skips the full-cell rerun requirement. Repeat `--rerun-manifest-path` for rerun manifests.
- `ramp-plan` writes one manifest per stage. `--stage-designs-per-library-type` is a comma-separated list of design counts. `--plan-path` writes the plan JSON.
- `ramp-evaluate` reads one or more `--report-path` gate reports. Exit code 2 when `promote` is false.
- `adopt-legacy` upgrades valid v2 done markers to v3. `--dry-run` reports without writing.

## Rows that change the design

`--fixed-arm` is repeatable, `LABEL=DECLARATION`. The declaration is residue letter plus 0-based index, pipe-separated (`catalytic_triad=H38|D73|C143`), or a path to a 1-D `.npy` mask. The letters set `fixed_tokens`. Omit it to fix nothing.

`--state-weight-profiles` is a comma-separated list of profile names. Only `equal` resolves by name. Any other name must also be declared with repeatable `--state-weight-profile LABEL=WEIGHTS` (pipe- or comma-separated weights, or a 1-D `.npy`). The default profile is `equal` unless a profile is given.

`--bias` is a path to a float `.npy` of shape `(L, 21)`, `L` the padded length (`max_length` 512). It is applied to every row. `--tie-group-map` (alias `--tie-group`) is a path to an integer `.npy` of shape `(L,)`; positions that share an id are decoded tied.

`--ligand-context-path` on `plan` is the keyed npz described in `aminx-sampling`. `--chain-id` on `plan` is JSON: a string, an array, or an object mapping each input path to chain ids. `ramp-plan` does not take `--chain-id` or `--ligand-context-path`.

## Cluster

The manifest is the unit you copy to a compute node. `worker` runs one row in one process. Structures in the manifest are local paths; this command does not fetch them. Resolve remote inputs on a machine that has network access before submitting workers (the `run` group's input cache is the fetch path; see `using-aminx`).

`--lock-backend` accepts `local_fs` or `distributed`, but `distributed` is rejected by every campaign verb. A distributed lock has to be passed to `run_manifest_row` or `execute_manifest` from Python. The CLI default is `local_fs`.

## Provenance

`plan_campaign_manifest` records `git_sha` and `git_provenance_source`. The default `git_sha=None` calls `aminx.agent.provenance.capture`. An explicit sha is kept with source `argument` unless `git_provenance_source` is passed as well. `write_campaign_manifest` takes `git_sha` only: it resolves the source (`argument` when a sha is passed, otherwise the capture source) and forwards both into the planner. Neither value is part of the row hash. The CLI verbs do not expose these flags, so a CLI plan uses the capture default.

`capture` tries `cisternal.provenance.capture_git_state`, then `git rev-parse HEAD` (`provenance_source` `builtin`), then a record with sha null and `provenance_source` `unknown`. The planner stores the string `unknown` when that sha is missing.

## Python

These functions are in `aminx.host.campaign`, not in `aminx.__all__`: `write_campaign_manifest`, `plan_campaign_manifest`, `run_manifest_row`, `execute_manifest`, `evaluate_campaign_gates`, `plan_scale_ramp`, `evaluate_scale_ramp_reports`, `adopt_legacy_done_markers`. The row spec type is the exported `SamplingSpecification`. Call `aminx.configure_multiprocessing` yourself only for a notebook or script; the campaign CLI already does.
