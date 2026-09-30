# Running aminx ProteinMPNN sampling in the browser

This guide is for porting the exported ProteinMPNN sampler into a plain static site
(for example a local ProteinHunter.org build). No bundler, no server-side Python, no
WebGPU. The page loads one ONNX file per length bucket and runs it with
onnxruntime-web on the wasm execution provider.

Status as of 2026-09-29. Items marked **pending** are still running or not yet done.
Don't cite a number from them yet.

## What ships

| Piece | Path in repo | Role |
| :--- | :--- | :--- |
| Sampler module | `browser/aminx-sampler/aminx_sampler.mjs` | `createSampler`, bucket choice, padding, feeding the session |
| RunSpec core | `browser/aminx-sampler/runspec_core.mjs` | Builds the 11 input tensors from a structure and a RunSpec: PRNG, Gumbel noise, decoding order, bias/omit, fixed/tied positions. Has no `node:` imports. |
| Model, L ≤ 128 | `p07_sample_L128.onnx` (about 19.4 MB) | Built by the export step below. It isn't checked in yet (**pending**: persisted release files with a sha256 manifest) |
| Model, L ≤ 256 | `p07_sample_L256.onnx` (about 31.5 MB) | Same |
| Runtime | `onnxruntime-web@1.30.0`, wasm build (`ort.wasm.min.mjs`) | Pinned. This is the version and build every parity run used |

"P07" is the internal name of this export: the full autoregressive sampler inside
one ONNX graph. The encoder, the decoding loop and the sampling step all run
in-graph. The noise and every RunSpec setting are runtime inputs, so one file per
bucket serves every design configuration.

Browser scope is **ProteinMPNN sampling only** (checkpoint `proteinmpnn_v_48_020`).
LigandMPNN, membrane and other variants aren't exported to the browser.

## Files to copy into the static site

```
site/
  index.html
  js/aminx_sampler.mjs      <- browser/aminx-sampler/aminx_sampler.mjs
  js/runspec_core.mjs       <- browser/aminx-sampler/runspec_core.mjs (same dir; imported relatively)
  models/p07_sample_L128.onnx
  models/p07_sample_L256.onnx
```

Serve only the `.onnx` file. The export embeds its weights: `embed_external_data`
rewrites the jax2onnx output into one self-contained file. A stray `.onnx.data` file
next to it is stale. Don't ship it and don't rely on it.

## Minimal integration (plain ES modules)

```html
<script type="module">
  import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
  import { createSampler, pickBucket, MPNN_ALPHABET } from "./js/aminx_sampler.mjs";

  // Point ORT at its wasm binaries (same CDN version as the JS).
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

  // structure: backbone only, L residues (see "Structure format" below)
  const L = structure.coords.length;
  const bucket = pickBucket(L);                 // 128 or 256; throws if L > 256
  const sampler = await createSampler(ort, `./models/p07_sample_L${bucket}.onnx`, {
    numThreads: crossOriginIsolated ? navigator.hardwareConcurrency : 1,
  });

  try {
    const runspec = {
      seed: 42,
      temperature: 0.1,
      omit_AA: "C",
      // bias_AA: { A: -1.0 }, fixed_positions: { 0: "M" }, tied_positions: [[3, 40]], ...
    };
    const { outputs, nReal } = await sampler.sample(structure, runspec);

    const tokens = outputs["cond_out_0"].data;        // Int32Array [bucket]
    const logProbs = outputs["log_softmax_out_0"].data; // Float32Array [bucket * 21]
    const sequence = Array.from(tokens.slice(0, nReal), (t) => MPNN_ALPHABET[t]).join("");
    console.log(sequence);
  } finally {
    await sampler.release();                    // free wasm memory; one session per model
  }
</script>
```

Notes:

- **Outputs.** `cond_out_0` holds the sampled tokens (int32, `[bucket]`).
  `log_softmax_out_0` holds per-position log-probabilities (float32,
  `[bucket, 21]`). Use only the first `nReal` rows; the rest is padding. If you
  re-export, check `session.outputNames` to confirm the names.
- **Alphabet.** Token `i` is `"ACDEFGHIKLMNPQRSTVWYX"[i]`. That is the ProteinMPNN
  order, not alphabetical-by-name. Index 20 is `X`.
- **Reuse the sampler.** Session creation compiles the graph and is the slow part.
  Create one sampler per bucket, call `sample` many times, and release it on
  teardown.
- **Many designs.** Call `sample` again with a different `seed`. Randomness comes
  only from the `seed` field, through a SplitMix64 PRNG, so a given
  (structure, RunSpec, seed) always gives the same sequence.

## Structure format

`sampler.sample(structure, runspec)` takes plain arrays or typed arrays:

| Field | Shape / type | Meaning |
| :--- | :--- | :--- |
| `coords` | `[L][4][3]` numbers, or flat `Float32Array` with `coords_shape: [L,4,3]` | Backbone atoms in the order **N, CA, C, O** (no CB; the model computes it) |
| `mask` | `[L]` 0/1 | 1 = real residue |
| `residue_index` | `[L]` int | PDB residue numbering, used for relative-position features |
| `chain_index` | `[L]` int | Integer chain id per residue |
| `chain_ids` | `[L]` strings, optional | Chain letters. Needed only for `chains_to_design` |

Parse the PDB/mmCIF in the page with any JS parser. Take N/CA/C/O per residue in
file order. Drop residues missing any backbone atom, or keep them with `mask: 0`.

Padding up to the bucket is automatic: `padStructure` zero-pads coords and mask
and repeats the last residue/chain index. This matches the Python export's
convention exactly. **The maximum length is 256 residues.** Longer inputs throw.

## RunSpec settings (all runtime inputs)

| Key | Example | Effect |
| :--- | :--- | :--- |
| `seed` | `42` | Drives the Gumbel noise and the random decoding order |
| `temperature` | `0.1` (default) | Sampling temperature |
| `bias_AA` | `{ "A": -1.0 }` | Add to that residue type's logit everywhere |
| `bias_AA_per_residue` | `{ "5": { "W": 2.0 } }` | Per-position logit bias |
| `omit_AA` | `"CX"` | Forbid residue types everywhere |
| `omit_AA_per_residue` | `{ "5": "P" }` | Forbid residue types at a position |
| `chains_to_design` | `["A"]` | Other chains stay fixed at their native tokens (needs `chain_ids` and `native_tokens`) |
| `fixed_positions` | `{ "0": "M", "4": "G" }` | Pin these 0-based positions to the given residue |
| `tied_positions` | `[[3, 40], [4, 41]]` | Force groups of positions to the same residue |
| `decoding_order` | `"random"` or an explicit permutation | Autoregressive order. With `"random"`, fixed positions decode first |

`chains_to_design` fixes every other chain to `structure.native_tokens` (`[L]` int32 in
the alphabet above), so supply those when you use it. All positions are 0-based
indices into the structure arrays, not PDB residue numbers.

## Hosting requirements

- **Multithreading needs cross-origin isolation.** Serve the page with
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. Without these headers,
  `crossOriginIsolated` is false and ORT runs single-threaded. Output is the same,
  just slower. `browser/layer_c/serve.mjs` sets these headers for local testing.
  With COEP on, CDN assets must be CORS-enabled (jsdelivr is) or self-hosted.
- **Memory.** One L256 session plus its inputs fits comfortably in a desktop tab.
  Mobile hasn't been tested.
- **Execution provider.** Use wasm only. Don't pass `webgpu` (see below).

## Validation status

| Check | What it compares | Status |
| :--- | :--- | :--- |
| Browser vs native ORT, bitwise | ORT-Web (Chromium) tokens and log-probs vs onnxruntime CPU on the same `.onnx`, same inputs | Bitwise identical on the 12-cell resume smoke. Full gate **pending** (below) |
| RunSpec knobs gate | JAX vs ORT-CPU vs ORT-Web across every RunSpec knob, L128 and L256, 288 cells. Tokens exact; log-prob bound 2e-4 | **Pending**. The recovery run is in progress; 250 of 288 cells were carried over from the interrupted run by verified hash |
| JS RunSpec builder vs Python | `runspec_core.mjs` inputs vs the Python builder (PRNG, Gumbel, orders, bias) | Node unit tests pass (`runspec.test.mjs` 11/11, `aminx_sampler.test.mjs` 5/5) |
| Sampling distribution vs reference ProteinMPNN | Statistical comparison of aminx samples against LigandMPNN@26ec57ac | **Pending**. Calibration running on titanix; validate shards next on Engaging |
| Speed / memory profile | Wall time per design, session-create time, peak memory | **Not measured.** The one earlier benchmark failed its own timer control, so no timing number is trustworthy yet |
| WebGPU | — | **Not validated; out of scope for the first release** |

## WebGPU

Not covered in this release. The P07 graph keeps the whole autoregressive loop
in-graph (ONNX `Loop`/`If`/`Scan`). The ORT WebGPU provider tends to run control-flow
nodes on the CPU, which forces transfers every step. Any speedup would have to be
measured, not assumed. The plan is a later pre-registered probe on real GPU hardware
that checks output against the wasm path before any speed comparison. Until then,
wasm is the supported path.

## Reproducing the ONNX files

From an aminx checkout (Python 3.12, `uv`):

```bash
uv run --frozen --with jax2onnx==0.16.1 --with onnx --with onnxruntime==1.30.0 \
  python scripts/browser_validation/p07_knobs_gate.py --out /tmp/p07.json --budget-s 3600
```

The gate calls `aminx.export.wrappers.make_p07_sample` on
`load_model(checkpoint_id="proteinmpnn_v_48_020")`, converts with `jax2onnx.to_onnx`
for each bucket, and embeds the external data into one file. Export is
byte-deterministic: repeated exports of the same commit give the same sha256. A
standalone `export` entry point with a sha256 manifest is **pending**. Until it
lands, take the models from a gate run's work directory (`browser_L128/`,
`browser_L256/`).

## Licensing

The aminx code is MIT. The weights are converted from upstream ProteinMPNN
(dauparas) and distributed through Hugging Face `maraxen/aminx`. **Confirm the
upstream weights license allows redistribution before serving them publicly**, and
credit ProteinMPNN (Dauparas et al., *Science* 2022) on the page.
