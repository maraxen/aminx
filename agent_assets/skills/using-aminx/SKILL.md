---
name: using-aminx
description: Router for aminx, the JAX and Equinox ProteinMPNN and LigandMPNN inverse-folding library. Use it to choose an install, pick MCP tools versus the CLI versus the Python API, select a checkpoint, and decide which task skill to load next.
---

# using-aminx

aminx designs, scores, and inspects protein sequences from structure files. It is a JAX and Equinox implementation of ProteinMPNN and LigandMPNN: inverse folding from a backbone (and, for ligand checkpoints, from ligand context) to amino-acid sequences. The same run is available to an agent, a shell, or Python.

## Install

MCP tools need the agent extra. The library and the `aminx` CLI do not.

```bash
pip install "aminx[agent]"
```

`pip install aminx` installs the library and the CLI only.

## Which surface

| Surface | When to use it |
| --- | --- |
| MCP tools | Inside an agent. This is the preferred surface. Tools take structure paths and return JSON plus side-file paths. |
| `aminx` CLI | A shell command or a file-driven, reproducible run. |
| Python API | Composing JAX code. Start with `aminx.parse_structure`, `aminx.load_model`, `aminx.sample`, and `aminx.score`. |

The first call for a given checkpoint and input shape compiles JAX and can take minutes. Later calls with the same shapes reuse that compilation.

## MCP tools

| Tool | What it does |
| --- | --- |
| `aminx-mcp:sample` | Design sequences for structure files. |
| `aminx-mcp:score` | Score sequences against structures. |
| `aminx-mcp:inspect` | Logits and other features (conditional or unconditional). |
| `aminx-mcp:jacobian` | Sequence Jacobian. Large arrays stay in a side file. |
| `aminx-mcp:spec_emit` | Build a run-spec JSON object without running the model. |
| `aminx-mcp:spec_validate` | Check that a run-spec JSON object decodes. |
| `aminx-mcp:list_checkpoints` | List checkpoint ids, whether each is installed locally, and the default id. |

A model-run result carries `spec`, `spec_sha256`, and `provenance`. Side files are written under `output_dir`. When that argument is omitted the directory is `$AMINX_AGENT_OUTPUT_DIR`, otherwise `$XDG_CACHE_HOME/aminx/agent`, otherwise `~/.cache/aminx/agent`.

## Checkpoints

Call `aminx-mcp:list_checkpoints` before choosing weights. The default checkpoint id is `proteinmpnn_v_48_020`. Soluble, membrane, and ligand checkpoints are in that listing. A weight file that is not installed locally downloads from the Hub on first use.

## CLI

`run` has four verbs: `sample`, `score`, `inspect`, and `jacobian`. Options shared by those verbs go before the verb.

```bash
aminx run --checkpoint-id proteinmpnn_v_48_020 sample --inputs structure.pdb --num-samples 4
```

```bash
aminx run score --inputs structure.pdb --sequences-to-score ACDEFGHIKLMNPQRSTVWY
```

```bash
aminx run inspect --inputs structure.pdb
```

```bash
aminx run jacobian --inputs structure.pdb
```

`aminx spec`, `aminx campaign`, and `aminx potts` are separate command groups. Use the task skill for the group you need.

## Load next

- `aminx-sampling` — temperature, fixed positions, tied positions, bias, multi-state design, ligand context, and output files.
- `aminx-scoring` — score, inspect, jacobian, and how to read the results.
- `aminx-run-specs` — JSON run specs, including emit, validate, and roundtrip.
- `aminx-campaigns` — campaign plan, worker, run, gates, and ramps.
- `aminx-potts` — Potts emit and run from a GeometryBundle.
