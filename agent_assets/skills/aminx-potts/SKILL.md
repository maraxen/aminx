---
name: aminx-potts
description: Emit and run an aminx Potts specification from a GeometryBundle. Covers aminx potts emit and run, k_neighbors, and the JSON result fields.
---

# aminx-potts

Potts inference is a separate command group from ProteinMPNN sampling. Sequence design, scoring, and campaign manifests are `aminx-sampling`, `aminx-scoring`, and `aminx-campaigns`. There is no Potts MCP tool. `aminx-mcp:spec_emit` does not take `kind` `potts`. `aminx-mcp:spec_validate` will still decode a Potts document (`aminx-run-specs`).

## Emit

`aminx potts emit` writes a `PottsRunSpec` JSON file. `--weights-path` must exist. `--k-neighbors` is required and is not stored in the checkpoint; it has to match the value used in training. `--out` is required. `--n-backbones` defaults to 1. `--caliby-path` is an optional calibration model. `--trw-backend` (default `dense_pinv`) and `--trw-iters` (default 10) are accepted and reserved; the spec stores `trw_spec` null so `__post_init__` fills the default TRW config.

```bash
aminx potts emit --weights-path weights.eqx --k-neighbors 48 --out potts.json
```

`aminx spec emit-potts` also writes `PottsRunSpec` JSON. It does not check that the weights file exists, and its `--k-neighbors` default is 48. Prefer `aminx potts emit` when the checkpoint is already on disk.

## Run

`aminx potts run` loads the JSON spec and a `GeometryBundle`, runs inference, and writes JSON.

```bash
aminx potts run --spec-path potts.json --geometry-path geometry.eqx --out result.json
```

`geometry.eqx` is an Equinox tree. A zstd-compressed file (magic bytes `28 b5 2f fd`, typically `.eqx.zst`) is decompressed first. The loader rebuilds a `GeometryBundle` with `coords` `(n_states, n_residues, 4, 3)`, `mask`, `residue_index`, `chain_index`, `n_states`, `n_canonical` 20, and `n_flat`. The CLI calls `run_potts` with `jax.random.PRNGKey(0)`.

The result JSON has `marginals`, `h`, `J`, `rho`, `calibrated_marginals`, `n_backbones`, and `samples` (`null` when the result has no samples).

## Python

`PottsRunSpec` and `PottsTRWRunSpec` are exported from `aminx.potts`. `GeometryBundle` is `aminx.types.bundles.GeometryBundle`. The runner is `aminx.potts.runner.run_potts(spec, geometry, key)`. It loads weights from `spec.weights_path`, uses identity calibration when `caliby_path` is null, and uses `PoeModel` when `n_backbones` is greater than 1. None of these names are in `aminx.__all__`. JSON helpers are `PottsRunSpec.to_json` and `PottsRunSpec.from_json`.
