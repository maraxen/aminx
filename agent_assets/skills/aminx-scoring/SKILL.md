---
name: aminx-scoring
description: Score sequences, inspect conditional and unconditional logits, and compute a sequence Jacobian. Covers result shapes, npz side-file keys, and residue trimming.
---

# aminx-scoring

`score`, `inspect`, and `jacobian` read a structure (and, for score, sequences) without designing a new library. Sampling controls are in `aminx-sampling`. JSON specs for the same runs are in `aminx-run-specs`. Checkpoints are in `using-aminx`.

All three MCP tools take local structure paths, optional `checkpoint_id` and `chain_id`, an `options` dict of other specification fields, `output_dir`, and `inline_cap` (default 10000 elements). A key set both as a typed parameter and inside `options` is an error. Unknown keys are errors. Every model-run result has `kind`, `run_id`, `structure_ids`, `structure_lengths` when a structure parsed, `spec`, `spec_sha256`, `aminx_version`, `provenance`, `warnings`, and `side_file`.

## Score

`aminx-mcp:score` takes `sequences` and stores them as `sequences_to_score`. The CLI flag is `--sequences-to-score` (repeatable, required).

```bash
aminx run score --inputs structure.pdb --sequences-to-score ACDEFGHIKLMNPQRSTVWY
```

The shaped result adds `structures`. Each entry has `structure_id` and `scores`, a list of `{sequence, nll}`. `nll` is the negative log-likelihood. The runner `scores` array is two-dimensional, `(n_structures, n_sequences)`; a 1-D array is reshaped to one row. If `sequences_to_score` is shorter than that array, missing sequences are empty strings and `warnings` says so.

`--return-logits` defaults to false on `score`. Logits and other arrays over `inline_cap` go to the npz. `aminx.score` is the exported JAX function: it returns `(nll, logits, decoding order)` for one sequence array. `ScoringSpecification` is also exported. `aminx.host.runner.score` is what the CLI and MCP call.

Score temperature defaults to `1.0` and does not change the negative log-likelihood. `conformational_states` and `decoding_order_fn` cannot be set through the agent or through JSON.

## Inspect

`aminx-mcp:inspect` takes `features`, passed as `inspection_features`. The default is `unconditional_logits` only. Allowed names:

- `unconditional_logits`
- `conditional_logits`
- `batched_conditional_logits`
- `encoded_node_features`
- `decoded_node_features`
- `edge_features`

```bash
aminx run inspect --inputs structure.pdb --inspection-features unconditional_logits --inspection-features conditional_logits
```

Conditional logits are the features named `conditional_logits` and `batched_conditional_logits`. Unconditional logits are `unconditional_logits`. `aminx.make_conditional_logits_fn` is the exported JAX constructor for a conditional-logits function; the inspect runner is `aminx.host.runner.inspect` with an `InspectionSpecification`.

`--distance-matrix` adds a distance matrix (`ca`, `cb`, `backbone_average`, or `closest_atom`). `--cross-input-similarity` needs at least two inputs. The metric is `rmsd`, `tm-score`, `gdt_ts`, `gdt_ha`, or `cosine`.

## Jacobian

`aminx-mcp:jacobian` takes `mode` (`categorical` or `reverse`), stored as `jacobian_mode`. The default is `categorical`.

- `categorical`: full categorical Jacobian, shape `(L, 21, L, 21)`. `--compute-apc` defaults to true and writes `apc_frobenius_norm`.
- `reverse`: score gradient, shape `(L, 21)`.

```bash
aminx run jacobian --inputs structure.pdb --jacobian-mode categorical
```

```bash
aminx run jacobian --inputs structure.pdb --jacobian-mode reverse
```

`JacobianSpecification` is exported. The runner is `aminx.host.runner.jacobian`. `combine_fn` must be null for JSON; a non-null callable cannot be encoded.

## How to read the side file

Arrays with more than `inline_cap` elements are not inlined. The JSON value becomes `{"shape": [...], "dtype": "..."}`, and the array is written to `<output_dir>/<run_id>.npz` (`side_file.path`). `side_file.arrays` maps each logical key to `shape` and `dtype`. A slash in a logical key is stored as `__`.

When `output_dir` is omitted the directory is `$AMINX_AGENT_OUTPUT_DIR`, otherwise `$XDG_CACHE_HOME/aminx/agent`, otherwise `~/.cache/aminx/agent`.

Per-structure list results are trimmed to the parsed residue count before routing. A missing length leaves that structure untrimmed and adds a warning. Axes that are trimmed:

| Key | Residue axes |
| --- | --- |
| `unconditional_logits`, `conditional_logits`, `encoded_node_features`, `decoded_node_features`, `edge_features`, `score_gradients` | 0 |
| `batched_conditional_logits` | 2 |
| `distance_matrix`, `apc_frobenius_norm` | 0 and 1 |
| `categorical_jacobians` | 0 and 2 |

Sampled sequence strings are trimmed in `aminx-sampling`, not by this table. CLI `--output-h5-path` on inspect and score is an HDF5 path; on jacobian it is a Zarr store.
