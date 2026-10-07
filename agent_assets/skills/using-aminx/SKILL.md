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

## Agent launch mode

Publish the plugin with `python -m aminx.agent.plugin publish`. The default MCP command is `uvx --from aminx[agent]==<version> aminx-mcp`. That version must be on PyPI. The first launch resolves the uvx environment, and the first tool call for a checkpoint and input shape compiles JAX.

`python -m aminx.agent.plugin publish --launch venv` switches to the local interpreter. The same choice is `AMINX_MCP_LAUNCH=venv` and `AMINX_MCP_PYTHON`, `[tool.aminx.agent]` in the nearest `pyproject.toml`, or `[agent]` in `${XDG_CONFIG_HOME:-~/.config}/aminx/config.toml`. The recorded command is an absolute interpreter path running `-m aminx.agent.mcp`, because Claude Code does not activate virtualenvs. `python -m aminx.agent.plugin info` prints the mode, command, and source. Publish again with `--launch uvx`, and remove an env or config setting if one was used, to switch back.

Details are in the docs page "Using aminx from an agent" (`docs/source/tutorials/agent.rst`).

## Which surface

| Surface | When to use it |
| --- | --- |
| MCP tools | Inside an agent. This is the preferred surface. Tools take structure paths and return JSON plus side-file paths. |
| `aminx` CLI | A shell command or a file-driven, reproducible run. |
| Python API | Composing JAX code. Start with `aminx.parse_structure`, `aminx.load_model`, `aminx.sample`, and `aminx.score`. |

Run time grows with the length a batch runs at. `sample` and `score` batches are trimmed by the runner to the next bucket on xtrax's ladder (64, 128, 256, 512), so `max_length` only caps them. Where the runner does not trim (`inspect`, `jacobian`, `pass_mode` inter, averaged or multi-state scoring, `length_bucketing` false), MCP tools fit `max_length` to the parsed structures on the same ladder unless `options` sets it, and record it in the returned `spec`. The first call for a checkpoint and shape compiles JAX; later calls in the same server reuse it.

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
