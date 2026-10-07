---
name: aminx-run-specs
description: Build and check aminx run specs as JSON. Covers MCP spec_emit and spec_validate, the aminx spec verbs, and fields that cannot be serialized.
---

# aminx-run-specs

A run spec is JSON. It is not TOML. The document has `_spec_class` set to `SamplingSpecification`, `ScoringSpecification`, `InspectionSpecification`, or `JacobianSpecification`. A Potts document is separate: see `aminx-potts` and the `emit-potts` verb below. What the runs do is in `aminx-sampling` and `aminx-scoring`.

Encoded documents include integer `schema_version` 1 next to `_spec_class`. A document with no `schema_version` is legacy and still loads. A newer version, or a value that is not an integer, is rejected. A key that is not a field of the spec class is rejected: the error lists every unknown key and suggests a close match when there is one. Keys removed from the spec are migrated by a table instead of being ignored silently.

## MCP

`aminx-mcp:spec_emit` builds a spec and does not run the model. `kind` is `sample`, `score`, `inspect`, or `jacobian`. `inputs` are local structure paths that must already exist. `options` are dataclass fields. For `score`, put the sequences in `options` under `sequences_to_score`. Unknown, deprecated, and unsettable keys are reported together. The result is `{"spec": ...}`.

`aminx-mcp:spec_validate` decodes that object, or a Potts document. Success is `{"ok": true, "spec_class": "<class name>"}`. Failure is `{"ok": false, "errors": ["..."]}` and is a normal result, not a raised tool error. `kind` `potts` is not a `spec_emit` kind. A Potts object is recognized when `kind` is `potts`, `_spec_class` is `PottsRunSpec`, or the object has both `trw_spec` and `weights_path`.

## CLI

Shared model options sit on the `spec` group, before the verb, the same way as `aminx run`. `--out` writes a file. `--compact` is single-line JSON. Without `--out`, JSON goes to stdout.

```bash
aminx spec emit-sample --inputs structure.pdb --num-samples 4 --temperature 0.1
```

```bash
aminx spec emit-score --inputs structure.pdb --sequences-to-score ACDEFGHIKLMNPQRSTVWY
```

```bash
aminx spec emit-inspect --inputs structure.pdb --inspection-features unconditional_logits
```

```bash
aminx spec emit-jacobian --inputs structure.pdb --jacobian-mode categorical
```

```bash
aminx spec emit-potts --weights-path weights.eqx --k-neighbors 48 --n-backbones 1 --out potts.json
```

```bash
aminx spec validate spec.json
```

```bash
aminx spec roundtrip spec.json --out spec-roundtrip.json
```

```bash
aminx spec portable-roundtrip portable.json
```

`validate` and `roundtrip` load a file with `run_specification_from_json` and exit 0 when it constructs. `validate` prints `OK:` plus the class name. `roundtrip` re-encodes through the same codec. `portable-roundtrip` reloads a portable `RunSpec` subset, not a full sampling or scoring spec. `emit-potts` writes `PottsRunSpec.to_json()` and does not require the weights file to exist. `aminx potts emit` does require it (`aminx-potts`).

`aminx run sample --emit-json` (and the same flag on `score`, `inspect`, and `jacobian`) also writes the spec and exits 0 without running the model.

## Fields that cannot serialize

Encoding uses `run_specification_to_json_dict`. These fields must be null, or a callable (callables are omitted): `run_spec`, `decoding_order_fn`, `foldcomp_database`, `combine_fn`, `encoding_fusion`, `decoding_fusion`. Any other non-null value raises `SpecJSONEncodeError`. `noise` is derived from `backbone_noise` and the other noise fields, so it is omitted and rebuilt on decode. `inputs` must be a string or a list of strings.

The agent rejects those fields plus `conformational_states`, `carry_specs`, and `dedup_specs`, even before encoding. `bias`, `fixed_positions`, `fixed_mask`, `fixed_tokens`, `tied_positions`, `tie_group_map`, and `state_weights` do serialize: arrays become nested lists, and JSON lists are coerced back to arrays on decode.

## Python

Exported names: `SamplingSpecification`, `ScoringSpecification`, `InspectionSpecification`, `JacobianSpecification`, `RunSpecification`. JSON helpers live in `aminx.run.spec_json`: `run_specification_to_json_dict`, `run_specification_from_json`, `run_specification_from_json_dict`. The MCP path uses `aminx.agent.requests.build_spec`, `spec_to_json_dict`, and `spec_from_json`. Potts encoding is `PottsRunSpec.to_json` / `PottsRunSpec.from_json`.
