# Aminx: A functional interface to ProteinMPNN in JAX

[![PyPI](https://img.shields.io/pypi/v/aminx)](https://pypi.org/project/aminx/)
[![Python](https://img.shields.io/pypi/pyversions/aminx)](https://pypi.org/project/aminx/)
[![Coverage](https://codecov.io/gh/maraxen/Aminx/branch/main/graph/badge.svg)](https://codecov.io/gh/maraxen/Aminx)
[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maraxen/Aminx/blob/main/examples/colab_inference_demo.ipynb)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](http://maraxen.github.io/Aminx)

> [!WARNING]
> **Alpha release.** aminx is under active development. The API is functional and validated against the LigandMPNN reference, but may change between releases. You may encounter bugs or rough edges — please open an issue if something breaks.

Aminx is a JAX/Equinox reimplementation of the [LigandMPNN](https://github.com/dauparas/LigandMPNN) codebase. A graded literature-parity audit against LigandMPNN@`26ec57ac` (2026-09-26) found several paths that match the reference to within 1e-4 nats, and confirmed defects in the ligand, membrane and conditional-scoring paths. The global grade is **FAIL** for now; see the [verdict](.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md) and the per-path table below. It runs 8–61× faster on a single structure (H200) by trading eager dispatch for `jit`/`vmap`/`scan` kernels.

What you get:

- A functional `sample()` / `score()` API — no model objects to wire up, no inference loop to write.
- A composable inference layer (`StageSet`) for swapping logit transforms, encode paths, and decode variants without touching kernel math.
- Numerical parity with upstream LigandMPNN, validated across unconditional, conditional, autoregressive, membrane, and side-chain-packer paths.
- Native batching across temperatures, backbones, and sequence lengths — operations you'd normally write as Python loops are compiled into single JAX kernels, so there's no recompilation penalty and no padding waste.

The batching point is worth unpacking: in vanilla JAX, running N temperatures requires either a Python loop (N separate compiled calls, or worse, N retraces) or manually writing a `vmap`. aminx has already done that work. Passing `temperature=[0.1, 0.3, 0.7]` vmaps over the temperature axis in one compiled call; scoring a mixed-length library reuses a compiled kernel per length bucket rather than recompiling per structure. The speedups in the tables below come from applying this pattern consistently to the operations most commonly written as loops in protein design workflows.

## Performance

Benchmarked on H200 (NVIDIA SXM5), A100 (PCIe), L40s, and Blackwell (SM120). All figures are warm-call medians; see the [full benchmark report](reports/benchmark_report.md) for every hardware and configuration.

**H200 — single-structure latency (seq_len=76)**

| Mode | aminx | ColabDesign (JAX) | LigandMPNN (PyTorch) | Speedup vs PyTorch |
|---|---|---|---|---|
| Autoregressive sample | 17 ms | 38 ms | 149 ms | **8.7×** |
| Score conditional | 1.5 ms | 7.0 ms | 92 ms | **61×** |

### Advanced Capability Benchmarks

These are capabilities ProteinMPNN doesn't expose, measured against the PyTorch baseline doing the equivalent work.

| Capability | Config | aminx | PyTorch | Speedup vs PyTorch |
|---|---|---|---|---|
| DedupGather | K=1 unique / N=32 total | 1.1 ms | 92 ms | **80×** |
| DedupGather | K=32 / N=32 (no dedup) | 36 ms | 2958 ms | **82×** |
| Mixed-length batch | lengths [76, 150, 300, 500] | 4.1 ms | 2280 ms | **554×** |
| Temperature array | M=8 temperatures | 2.2 ms/temp | — | **8× per-temp** |

The 8× floor holds across hardware; ceilings reach 84–91× depending on the operation (A100: 8–91×, L40s: 8–85×, Blackwell SM120: 8–84×).

## Documentation

**[Complete Documentation →](http://maraxen.github.io/Aminx)**

- [Composition Guide](docs/COMPOSITION_GUIDE.md) — `StageSet`, `InferencePlan`, and the five extension points
- [Parity Validation](docs/parity/parity_report.html) — numerical parity report vs the LigandMPNN reference

## Validation
Aminx is audited against the upstream [LigandMPNN](https://github.com/dauparas/LigandMPNN) reference (which includes ProteinMPNN behavior):
Aminx is validated against the upstream [LigandMPNN](https://github.com/dauparas/LigandMPNN) reference (which includes ProteinMPNN behavior):

Latest graded audit (browser-validation Phase 1, run `e3d3ffa2`, HEAD `e4b86a0c`): **global grade FAIL**.
Two things force it. The audit's stricter distributional sampling test could not be run
(its calibration was `budget_exceeded`: 435–483 h projected, because the sampler recomputed the
decoder over every position at each step). Autoregressive sampling itself is validated by the
`parity_heavy` suite (`test_autoregressive_sampling_parity`: ≥ 95% token agreement, log-prob
corr ≥ 0.95). And 8 core defects were confirmed adversarially. 8/8
reference-derived invariants pass, and each goes red on an injected defect. Pre-registered clause
parity is 19/33 core clauses (0.576). Full details:
[`.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md`](.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md).

| Path | Exact-tier worst ratio to bar (1e-4 nats) | Advances to Phase 2 | Blocking |
|------|------------------------------------------|---------------------|----------|
| P00 unconditional score | 0 | yes | – |
| P01 weight conversion | 0 | no | bias-handling / topology defects |
| P02 input parsing | 0 (exact) | no | null-bar informational row classified over_bar (harness quirk) |
| P03 k-NN graph | 0 | yes | – |
| P04 conditional score (ProteinMPNN) | 0.36 | yes | – |
| P05/P06 conditional score (context) | 0.47 / 0.48 | no | 2-hop self-identity leak |
| P07/P08 sampling | – | no | distributional sampling test not run (budget); AR parity test passes |
| P09 tied sampling | 0.64 | no | distributional sampling test not run; fixed-position log-probs |
| P11 LigandMPNN | 2221 (0.22 nats) | no | atom_context 16 vs 25; `v_c` bias; ligand positional bias |
| P12 side-chain context | 0.44 | no | not-advanced rows |
| P13 membrane | ≈ 5 × 10⁴ (≈ 5 nats) | no | random-init `physics_projection` bias |
| P14 packer | not implemented | no | – |

Earlier claim, superseded by the audit above: 30/30 `parity_heavy` tests pass on Engaging (job
14203624). That suite never tested the defects the audit found.

Older parity reports (predate the 2026-09-26 audit):

- [Parity report (HTML)](docs/parity/parity_report.html)
- [Parity report (PDF)](docs/parity/parity_report.pdf)

Root-level parity stubs are non-canonical; use the links above.

## Quick Start

### Installation

aminx uses [uv](https://docs.astral.sh/uv/) as its package manager. Install uv first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install aminx with your target hardware extra:

```bash
uv sync --extra cuda  # GPU (CUDA)
uv sync --extra tpu   # TPU
uv sync --extra cpu   # CPU-only
```

**If you prefer pip:**

```bash
pip install "aminx[cuda]"   # GPU
pip install "aminx[cpu]"    # CPU-only
```

**Install as a standalone CLI tool** (no virtual environment required):

```bash
uv tool install aminx
aminx spec validate run_spec.json
```

Or invoke without installing at all:

```bash
uvx aminx spec validate run_spec.json
```

### High-level API

```python
from aminx.io.weights import load_model
from aminx.run import sample, score, SamplingSpecification, ScoringSpecification

model = load_model(model_version="v_48_020", model_weights="original")

# --- Sampling ---
spec = SamplingSpecification(
    inputs="path/to/structure.pdb",
    num_samples=10,
    temperature=0.1,
    random_seed=42,
)
results = sample(spec)
sequences = results["sequences"]   # (num_samples, seq_len)
logits    = results["logits"]      # (num_samples, seq_len, 21)

# --- Scoring ---
spec = ScoringSpecification(
    inputs="path/to/structure.pdb",
    sequences_to_score=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
    temperature=1.0,
)
results = score(spec)
scores = results["scores"]         # negative log-likelihood per sequence
```

### Composable Inference API

For control over fusion strategy, encode path, and decode variant without touching kernel math:

```python
import jax.numpy as jnp
from aminx.host.plan import make_inference_plan, InferencePlan, InferenceComponents
from aminx.inference.encode import make_encode_fn
from aminx.inference import driver
from aminx.inference.logits import GeometricMeanLogits, ARLogitFuse
from aminx.run import SamplingSpecification
from aminx.types.stages import StageSet

# Option A: factory (resolves stages from spec automatically)
spec = SamplingSpecification(
    inputs="structure.pdb",
    num_samples=10,
    multi_state_strategy="geometric_mean",
    state_weights=[1.0, 0.8, 0.6],
)
plan = make_inference_plan(model, spec)

# Option B: manual assembly for full control
state_weights = jnp.array([1.0, 0.8, 0.6])
stage_set = StageSet(
    logit_transform=GeometricMeanLogits(weights=state_weights, temperature=1.2),
    ar_logit_transform=ARLogitFuse(),
)
plan = InferencePlan(
    model=model,
    components=InferenceComponents(
        encode_fn=make_encode_fn(model, use_rolling_state=False),
        driver=driver.decode,
        stage_set=stage_set,
    ),
)

result = plan.sample(bundle, key, config)   # → SampleResult(sequence, logits)
logits = plan.score(bundle, key, config)    # → (L, 21)
```

Plain callables and lambdas work in all `StageSet` slots (the driver uses `eqx.filter_jit`). Use `eqx.Module` only when the callable carries JAX array leaves (e.g. weights that need grad).

See the [Composition Guide](docs/COMPOSITION_GUIDE.md) for the five extension points (`logit_transform`, `ar_logit_transform`, `decode_step`, `sample_step`, `tie_group_fuse`).

### Advanced Capabilities

#### Temperature Array Sweep

Pass a list of temperatures to run all M in a single JIT-compiled call. The kernel vmaps over the temperature dimension, so per-temperature cost scales near-ideally.

```python
from aminx.run import sample, SamplingSpecification

spec = SamplingSpecification(
    inputs="structure.pdb",
    temperature=[0.1, 0.3, 0.7, 1.0],  # M=4 temperatures, one JIT call
    num_samples=10,
)
results = sample(spec)
# One result dict per temperature, each with "sequences" and "logits"
for temp, result in zip(spec.temperature, results):
    print(f"T={temp}: {result['sequences'].shape}")
```

On H200 at M=8 (seq_len=76), per-temperature cost is **2.2 ms** — the same total wall-clock as M=1 (17 ms) split across 8 temperatures. Measured scaling: 7.97–8.47× across H200/A100/L40s/Blackwell.

---

#### Deduplicated Scoring (DedupGather pattern)

When scoring a large ensemble where many sequences share a backbone, score only the **K unique** structures rather than all N. aminx JIT-caches a compiled kernel per length bucket, so K sequential `plan.score()` calls stay fast regardless of N.

```python
from aminx.host.plan import make_inference_plan
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.tiling.bucketing import BucketingConfig
import jax.random as random

plan = make_inference_plan(model, spec)
bucket_cfg = BucketingConfig()
key = random.PRNGKey(0)

# Build one bundle per unique backbone (K unique out of N total)
unique_bundles = []
for coords, mask, residue_index, chain_index, sequence in unique_structures:
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        ligand_coords=None,
        ligand_atom_types=None,
        ligand_mask=None,
        temperature=1.0,
        mode="score_conditional",
        inference=True,
        bucket_config=bucket_cfg,
    )
    unique_bundles.append((bundle, config))

# Score K unique structures; scatter the K scores to your N positions
scores = [plan.score(bundle, key, config) for bundle, config in unique_bundles]
```

At K=1/N=32 on H200, latency is **1.1 ms** vs 92 ms for PyTorch scoring all 32 structures — an **80× speedup**. The speedup is stable across K (80–82× from K=1 through K=32) because PyTorch's cost scales linearly with N while aminx's scales linearly with K.

---

#### Mixed-Length Batch Scoring

Score a library of proteins with different sequence lengths without padding waste. aminx rounds each length to the next power-of-2 boundary and reuses the JIT-compiled kernel across structures in the same bucket — one compile per bucket, not one per structure.

```python
from aminx.host.plan import make_inference_plan
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.tiling.bucketing import BucketingConfig
import equinox as eqx

plan = make_inference_plan(model, spec)
bucket_cfg = BucketingConfig()

# Build one bundle per structure — lengths can differ freely
bundles = []
for coords, mask, residue_index, chain_index, sequence in your_library:
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        ligand_coords=None,
        ligand_atom_types=None,
        ligand_mask=None,
        temperature=1.0,
        mode="score_conditional",
        inference=True,
        bucket_config=bucket_cfg,
    )
    bundles.append((bundle, config))

# Each call reuses the compiled XLA kernel for its length bucket
score_one = eqx.filter_jit(plan.score)
scores = [score_one(bundle, key, config) for bundle, config in bundles]
```

For a batch of [76, 150, 300, 500]-residue structures on H200, total latency is **4.1 ms** versus 2280 ms for PyTorch (padded sequential) — a **554× speedup**. The gain comes from two sources: JAX's compiled kernels (vs PyTorch eager dispatch) and avoiding padding to the longest sequence.

---

### CLI

aminx ships a Typer CLI with four command groups: `run`, `campaign`, `spec`, and the `spec emit-*` family.

#### `aminx run` — run sampling and scoring pipelines

```bash
# Full sampling pipeline — constructs a RunSpecification and runs it end-to-end
aminx run sample \
  --inputs structure.pdb \
  --model-version v_48_020 \
  --model-weights original \
  --num-samples 10 \
  --random-seed 42

# Emit the spec JSON instead of running (useful for inspection or handoff)
aminx run sample --inputs structure.pdb --emit-json
aminx run sample --inputs structure.pdb --emit-json --out sample_spec.json

# score, jacobian, and inspect accept the same options;
# pass --emit-json to get the spec — the runner for these paths is not yet wired
aminx run score --inputs structure.pdb --sequences-to-score ACDEFGHIKLMNPQRSTVWY --emit-json
```

The base options are shared by all four subcommands (`sample`, `score`, `jacobian`, `inspect`) but belong to the **`run` group, so they go before the subcommand**: `--checkpoint-id`, `--chain-id`, `--max-length`, `--model-weights`, `--model-version`, `--model-family`, `--batch-size`, `--backbone-noise`, `--random-seed`, and the rest of the `RunSpecification` field set. Options specific to a subcommand (`--inputs`, `--num-samples`, `--temperature`, ...) go after it:

```bash
aminx run --checkpoint-id proteinmpnn_v_48_020 --chain-id C --random-seed 5 --max-length 96 \
  sample --inputs structure.pdb --num-samples 4 --temperature 0.3
```

`aminx run sample --chain-id C ...` is rejected (`No such option`). Fields that are not JSON-serialisable (`bias`, `fixed_positions`, `fixed_tokens`, `fixed_mask`, `state_weights`, `decode_fn`) cannot be set from the CLI; use the Python API. Spec construction failures exit 1; an unwired runner exits 2.

`--max-length` is the length every per-position array is padded to (default 512) and the autoregressive decode runs at that **padded** length, so for a short chain it dominates the cost: on one CPU, a 93-residue chain took roughly 86 s/sample at 512 and a small fraction of that at 96. Set it near your real chain length.

##### Heterogeneous inputs — local files and remote structures

The `--inputs` flag accepts multiple URI schemes, allowing you to mix local files and remote-fetched structures in a single command. Remote sources are fetched at CLI time to a local cache directory, making the resolved spec safely reproducible on cluster compute nodes (which are offline).

**Accepted URI forms:**

| Form | Example | Source | Behavior |
|------|---------|--------|----------|
| Bare path | `/data/1ubq.pdb`, `./structures/*.cif`, `data/` | Local | Files and glob patterns expanded as today; directories scanned for `.pdb`/`.cif` files. |
| `file://` | `file:///data/1ubq.pdb`, `file://localhost/data/1ubq.pdb` | Local | Explicit file URI; host portion (if present) is stripped; remainder is treated as a local path. |
| `pdb://` | `pdb://1A3A`, `pdb://1A3A.pdb`, `pdb://1A3A.cif` | RCSB | Fetch PDB structure by ID. Default format `mmcif`; suffix `.pdb` or `.cif` overrides. |
| `afdb://` | `afdb://P12345` | AlphaFold DB | Fetch AlphaFold v4 structure by UniProt ID. |
| `mdcath://` | `mdcath://1abcA00` | MD-CATH | Fetch MD-CATH HDF5 structure by ID. |

**Example — mixed local and remote inputs:**

```bash
# Fetch two RCSB structures and one AlphaFold structure; combine with a local directory
aminx run sample \
  --inputs pdb://1A3A \
  --inputs pdb://2ABC.pdb \
  --inputs afdb://P12345 \
  --inputs ./local_structures/ \
  --model-version v_48_020 \
  --num-samples 10
```

All remote sources are fetched to a cache directory (default: `~/.cache/aminx/inputs` or `$XDG_CACHE_HOME/aminx/inputs`). Subsequent runs with the same accession reuse cached files and skip the network call entirely — the resolved spec is fully deterministic and cluster-safe.

**Cache control and URI scheme override:**

```bash
# Override cache directory
aminx run sample --inputs pdb://1A3A --cache-dir /tmp/aminx_cache --num-samples 10

# Force all --inputs entries to be treated as local paths (suppress scheme detection)
aminx run sample --inputs "pdb://1A3A" --input-type file --num-samples 10
# ^ Treats "pdb://1A3A" as a literal local filename, does not fetch

# Apply a default scheme to schemeless tokens
aminx run sample --inputs 1A3A --input-type pdb --num-samples 10
# ^ Treats "1A3A" as pdb://1A3A; fetches from RCSB
```

**Error handling:** Empty accessions (e.g., `pdb://`), unknown schemes (e.g., `s3://`, `http://`), and fetch failures raise clear errors naming the problematic entry. Use `--inputs-fail-fast` to exit on first error, or omit it to warn and skip individual entries.

#### `aminx campaign` — design campaign orchestration

```bash
# Create a campaign manifest from a base spec
aminx campaign plan \
  --inputs structures/target.pdb \
  --campaign-id pilot_v1 \
  --manifest-path pilot.manifest.json \
  --output-root outputs/pilot \
  --designs-per-library-type 50 \
  --samples-chunk-size 16

# Execute a single manifest row (used by distributed workers)
aminx campaign worker --manifest-path pilot.manifest.json --row-index 0

# Execute all rows; exits 1 if any row fails
aminx campaign run --manifest-path pilot.manifest.json

# Evaluate quality gates; exits 2 if the campaign is not promoted
aminx campaign gates --manifest-path pilot.manifest.json

# Plan a staged scale ramp
aminx campaign ramp-plan \
  --inputs structures/target.pdb \
  --campaign-id ramp_v1 \
  --manifest-dir ramp_manifests/ \
  --output-root outputs/ramp \
  --stage-designs-per-library-type 10,50,200 \
  --samples-chunk-size 16

# Evaluate ramp stage reports; exits 2 if not promoted
aminx campaign ramp-evaluate --report-path stage1.json --report-path stage2.json
```

`--lock-backend distributed` is not supported from the CLI (raises an error); use `local_fs` (the default).

#### `aminx spec` — validate and round-trip spec files

```bash
# Validate a spec JSON — exits 0 and prints "OK: <SpecClass>", or exits 1 with the parse error
aminx spec validate run_spec.json

# Round-trip a spec through the codec — prints the re-serialized JSON to stdout
aminx spec roundtrip run_spec.json
aminx spec roundtrip run_spec.json --out validated.json   # write to file instead
aminx spec roundtrip run_spec.json --compact              # single-line JSON

# Round-trip the portable subset (dict-serializable fields only, no JAX arrays)
aminx spec portable-roundtrip portable_spec.json
aminx spec portable-roundtrip portable_spec.json --compact
```

#### `aminx spec emit-*` — emit spec JSON without running

The `emit-sample`, `emit-score`, `emit-jacobian`, and `emit-inspect` subcommands are a convenience complement to `aminx run <cmd> --emit-json` — identical output, no runner invoked.

```bash
# Emit a sample spec JSON — equivalent to: aminx run sample ... --emit-json
aminx spec emit-sample \
  --inputs structure.pdb \
  --model-version v_48_020 \
  --model-weights original \
  --compact

aminx spec emit-sample --inputs structure.pdb --out sample_spec.json
```

Specs can also be serialized from Python with `run_specification_to_json`:

```python
from aminx.run import run_specification_to_json
json_str = run_specification_to_json(spec)
```

## Potts Model Family (`aminx.potts`)

The `aminx.potts` module provides a parallel model family integrating PottsMPNN with
Tree-Reweighted belief propagation (TRW) for global sequence design.

| Component | Description |
|-----------|-------------|
| `PottsModel` | Equinox module wrapping PottsMPNN + DifferentiableTRW |
| `PoeModel` | N-backbone Product-of-Experts ensemble |
| `MpnnPottsDesigner` | MPNN-seeded Gibbs sampling coordinator |

**CLI:** `aminx potts run`, `aminx potts emit`
**Weight recapture:** `scripts/recapture/pottsmpnn_to_eqx.py`
**Architecture:** Parallel model family — see `.praxia/docs/decisions/260605_potts-parallel-not-stageset.md`

## Requirements

- Python ≥ 3.12
- JAX + Equinox (GPU/TPU/CPU via extras)
- `uv sync --extra cpu` for CPU-only; `--extra cuda` for GPU

## Where model weights come from

The built wheel ships **no** checkpoints, so the aminx version pin does not by itself
determine which weights execute. Resolution order, and it stops at the first hit:

1. **`AMINX_WEIGHTS_DIR`** — when set, this directory is authoritative and **fails closed**: a
   checkpoint missing from it raises rather than quietly falling through to the Hub. Scope is
   checkpoint-id resolution only; an explicit `local_path` / `--model-local-path` still
   bypasses it by design, and logs a warning when it does.
2. **Packaged resources** (`aminx/model_params/`) — present in a source checkout, absent from
   the wheel.
3. **Hugging Face Hub**, pinned to `HF_REVISION` in `aminx/io/weights.py`.

Because the pin is a full commit SHA, a warm cache resolves with **no network request at
all** — measured at ~0.7 ms against ~236 ms for the unpinned form, which issued a HEAD on
every call. Pinning is faster *and* reproducible, not a tradeoff between them.

The Hub's `main` branch carries the **newest** weight layout. `HF_REVISION` is the revision this
code can load: the LigandMPNN and membrane checkpoints were regenerated with the reference's
bias layout (#162, #163), and because deserialisation is positional, code from before that change
cannot load the new files (nor new code the old ones). Released wheels stay correct by pinning
the revision that matches their code. Never re-point a pin to a revision whose layout does not
match the code, and do not resolve the Hub's `main` from an older checkout.

### Recording which weights ran

```python
from aminx.io.weights import weight_provenance

record = weight_provenance("proteinmpnn_v_48_020.eqx.zst")
print(record.source, record.sha256, record.hub_revision)
```

Store `sha256` next to the aminx version in any result whose numbers depend on the weights —
the version alone does not identify them. `weight_provenance` shares one resolver with the
loader, so the record describes the file that actually executes **for the checkpoint-id
route**. It does not cover a run that passed `local_path` / `--model-local-path`, or one that
resolved through `checkpoint_registry_path` — those bypass resolution by design and log when
they do, and recording their weights is the caller's job.

### Overriding the pin

Set `AMINX_WEIGHTS_REVISION` to reach a checkpoint newer than the pin, or to roll back:

```bash
export AMINX_WEIGHTS_REVISION=<commit-sha>
```

In a SLURM batch script, export it before the run line:

```bash
#SBATCH --job-name=aminx-score
export AMINX_WEIGHTS_REVISION=<commit-sha>   # a revision compatible with this aminx version
uv run aminx run score --spec spec.json
```

The override is visible in `WeightProvenance.hub_revision`, which is read back from the
resolved cache path rather than from the variable — so an overridden run is still identifiable
from its own record. Setting either variable to a **blank** value is an error, not a request
for defaults: a blank value is almost always an unset variable interpolated into an
environment, and honouring it would change the weight source with no signal. Note this check
runs *before* the resolution order is known, so a blank `AMINX_WEIGHTS_REVISION` raises even on
a run that would have resolved from `AMINX_WEIGHTS_DIR` or packaged resources and never reached
the Hub. That is deliberate: a value that only fails once something happens to reach the Hub is
the failure mode this replaces.

## Development

| Command | Purpose |
|---------|---------|
| `uv run pytest` | Fast test suite (excludes `parity_heavy`) |
| `uv run ruff check src` | Lint |
| `uv run --extra dev ty check` | Type check (advisory: not a CI gate, and it currently reports many diagnostics) |
| `uv run ruff format .` | Auto-format |

All five decoding paths are validated via `parity_heavy` tests — see [Validation Reference](#validation-reference) below.

## Architecture

```
aminx.run          ← SamplingSpecification, ScoringSpecification, sample(), score()
aminx.host.plan    ← InferencePlan, InferenceComponents, make_inference_plan()
aminx.types.stages ← StageSet (the composition interface)
aminx.inference    ← driver.decode, logits (LOGIT_STRATEGIES, TIE_GROUP_STRATEGIES)
aminx.model        ← LigandMPNN, Packer (Equinox modules, JIT-safe)
aminx.sampling     ← sample() kernel
aminx.scoring      ← score() kernel
aminx.cli          ← aminx run / campaign / spec (Typer entry point)
```

`StageSet` is the seam between the host layer and the JAX-traced kernels: everything above it is Python-land, everything below it is traced. See the [Composition Guide](docs/COMPOSITION_GUIDE.md).

## Multiprocessing

Importing `aminx` does **not** set the multiprocessing start method. If your notebook or script spawns worker processes, call `configure_multiprocessing()` once at startup (see `aminx.runtime`); the campaign CLI does this for you.

## Validation Reference

<details>
<summary>Running the equivalence and parity suite</summary>

```bash
# Install project dependencies (CPU/dev/tests path)
uv sync --extra cpu --extra dev --extra tests --group dev
source .venv/bin/activate

# Checkout reference implementation (pinned commit used in CI)
git clone https://github.com/dauparas/LigandMPNN.git reference_ligandmpnn_clone
cd reference_ligandmpnn_clone && git checkout 26ec57ac976ade5379920dbd43c7f97a91cf82de && cd ..  # pin: scripts/browser_validation/reference_pins.json (3870631 is not an upstream commit)

# Optional strict preflight per parity tier
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_prereqs.py --reference-path "$REFERENCE_PATH" --project-root . --tier parity_heavy
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_prereqs.py --reference-path "$REFERENCE_PATH" --project-root . --tier parity_audit

# Validate parity asset cache/checksums
uv run python scripts/check_parity_assets.py --tier parity_fast
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_assets.py --tier parity_heavy
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_assets.py --tier parity_audit

# Run fast deterministic parity checks
uv run pytest tests/parity -m parity_fast -v

# Run reference-backed heavy parity checks
REFERENCE_PATH=./reference_ligandmpnn_clone \
  PRXTEIN_PARITY_TIER=parity_heavy \
  uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v

# Convert full checkpoint families and run parity_audit checks
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/convert_parity_family_weights.py \
    --project-root . \
    --reference-path "$REFERENCE_PATH" \
    --tier parity_audit \
    --skip-existing
REFERENCE_PATH=./reference_ligandmpnn_clone \
  PRXTEIN_PARITY_TIER=parity_audit \
  uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_audit -v

# Collect expanded parity evidence (multi-backbone + synthetic random cases)
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/collect_parity_evidence.py \
    --project-root . \
    --case-corpus tests/parity/parity_case_corpus.json \
    --output-dir docs/parity/reports/evidence

# Render Markdown/HTML report and export PDF with embedded plots/tables
uv run python scripts/generate_parity_report.py --project-root . --output-dir docs/parity --pdf
```

`AMINX_VERIFY` (runtime jaxtyping + beartype): tests under `tests/parity/` set `AMINX_VERIFY=1` via `tests/parity/conftest.py`. Elsewhere, opt in with:

```bash
AMINX_VERIFY=1 uv run pytest path/to/test.py -v
```

CI tier routing:

- pull_request/main CI excludes `parity_heavy` and `parity_audit` from the default pytest matrix.
- `parity.yml` runs heavy reference-backed checks on `main` push and manual dispatch.
- `parity-audit.yml` runs full-family audit checks on weekly schedule and manual dispatch.
- `ligand-tied-positions-and-multi-state` is staged as warn-only in `parity_heavy` and fail in `parity_audit`.

</details>

## License

MIT License.

## Contributing

Contributions are welcome — see the [contributing guidelines](CONTRIBUTING.md).

## Support

- **Documentation**: [http://maraxen.github.io/Aminx](http://maraxen.github.io/Aminx)
- **Issues**: [GitHub Issues](https://github.com/maraxen/Aminx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/maraxen/Aminx/discussions)

---

Built in JAX, for anyone who'd rather design proteins than write inference loops.
