---
name: aminx-sampling
description: Design sequences with aminx sample. Covers temperature, fixed and tied positions, bias, multi-state design, ligand context, and where outputs and side files go.
---

# aminx-sampling

`sample` draws amino-acid sequences for backbone structures. Checkpoints and the choice of MCP, CLI, or Python are in `using-aminx`. Reading scores, logits, and Jacobians is `aminx-scoring`. Campaign grids are `aminx-campaigns`.

## MCP

`aminx-mcp:sample` takes local structure paths. Typed parameters are `inputs`, `num_samples` (default 1), `temperature` (float or list of floats; default 0.1), `random_seed` (default 42), `checkpoint_id`, and `chain_id`. Anything else on `SamplingSpecification` goes in `options`. A key set both as a parameter and inside `options` is an error. Unknown keys are errors. `output_dir` and `inline_cap` (default 10000 elements) control the side file.

`options` can include `bias`, `fixed_positions`, `fixed_mask`, `fixed_tokens`, `tied_positions`, `tie_group_map`, `multi_state_strategy`, `state_weights`, `multi_state_temperature`, `multi_state_sharpness`, `ligand_conditioning`, and `ligand_context_path`. Pass arrays as nested lists. These fields are not accepted: `run_spec`, `decoding_order_fn`, `foldcomp_database`, `combine_fn`, `encoding_fusion`, `decoding_fusion`, `conformational_states`, `carry_specs`, `dedup_specs`, `noise`.

## Temperature

`temperature` is a sequence. The CLI flag is comma-separated and belongs on the `sample` verb. The specification default is `0.1`. `sampling_strategy` is `temperature` or `straight_through`. Straight-through requires `iterations` and `learning_rate`.

```bash
aminx run sample --inputs structure.pdb --num-samples 4 --temperature 0.1,0.2
```

## Fixed positions

`fixed_positions` and `fixed_mask` are masks of which positions stay frozen. They are unioned. `fixed_tokens` says which amino-acid token each frozen position keeps. A frozen position with no `fixed_tokens` is an error: the sampler will not default those sites to alanine. `aminx run sample` has no flags for these fields. Set them in MCP `options` or on `SamplingSpecification`.

`aminx.sample` (the exported JAX function) takes `fixed_mask` and `fixed_tokens` directly. It does not take `fixed_positions`.

## Tied positions

`tied_positions` is a list of index pairs, or the mode `auto` or `direct`. `auto` and `direct` require `pass_mode` `inter`. The CLI flag is repeatable and sits on the `run` callback, before the verb: `A:B` for two integer indices, or a single `auto` / `direct`. `tie_group_map` is a per-position integer array; positions that share an id are decoded together. That array is an `options` field, not a `run sample` flag.

```bash
aminx run --tied-position 10:20 sample --inputs structure.pdb
```

## Bias

`bias` is a per-position, per-token logit array. The campaign planner documents it as shape `(L, 21)`. On a single sample call, put the nested list in MCP `options` or pass the array to `aminx.sample`. `aminx run sample` does not take a bias file. A campaign-wide `.npy` bias is `aminx-campaigns`.

## Multi-state design

`multi_state_strategy` is `arithmetic_mean`, `geometric_mean`, or `product` (default `arithmetic_mean`). `state_weights` are mixing proportions across states. `multi_state_sharpness` scales the `product` strategy (default `1.0`); `None` keeps product-of-experts scaling. `multi_state_temperature` is a shared `run` option (default `1.0`).

```bash
aminx run --multi-state-temperature 1.0 sample --inputs structure.pdb --inputs other.pdb --multi-state-strategy arithmetic_mean
```

## Ligand context

Ligand checkpoints are listed by `aminx-mcp:list_checkpoints` (`using-aminx`). `ligand_context_path` points at an npz keyed `<structure_id>::Y`, `<structure_id>::Y_t`, and `<structure_id>::Y_m`, where `structure_id` is the structure path stem. `ligand_conditioning` is tri-state: unset uses ligand tensors when they are present; `true` requires them; `false` ignores them and uses a zero placeholder.

```bash
aminx run --checkpoint-id ligandmpnn_v_32_020_25 sample --inputs structure.pdb --ligand-context-path ligand.npz --ligand-conditioning
```

`aminx.sample` takes `ligand_coords`, `ligand_atom_types`, and `ligand_mask` instead of a context file.

## Outputs and side files

The MCP result includes `structures`: one entry per structure, each with `samples`. A sample has `sequence`, `sample_index`, `noise_index`, and `temperature_index`. Sequences are decoded with the aminx alphabet and cut to the parsed residue count when that count is known (`trimmed` is true). If it is not known, the sequence is left untrimmed and `warnings` says so. Set `compute_pseudo_perplexity` (it requires `return_logits`) to add `pseudo_perplexity` on each sample. The token array is shape `(B, N, noise, temperature, L)` before that decode.

Arrays larger than `inline_cap` are not inlined. They are stored in `<output_dir>/<run_id>.npz` under `side_file.path`, with `/` in the logical key replaced by `__`. `side_file` is null when every array fit inline. The result also carries `spec`, `spec_sha256`, and `provenance`. When `output_dir` is omitted the directory is `$AMINX_AGENT_OUTPUT_DIR`, otherwise `$XDG_CACHE_HOME/aminx/agent`, otherwise `~/.cache/aminx/agent`.

The CLI writes the runner output for the spec. `--output-h5-path` on `sample` is a Zarr store path. Shared options such as `--output-dir`, `--random-seed`, and `--checkpoint-id` go before the verb.

## Python

`SamplingSpecification` and `sample` are exported from `aminx`. `sample` is the JAX sampler: it returns `(sequence, logits, decoding order)`. `aminx.host.runner.sample` is what the CLI and MCP call; it takes a `SamplingSpecification`. Build a spec in JSON first with `aminx-run-specs` when the run must be reproducible.
