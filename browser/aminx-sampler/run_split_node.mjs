// Drive browser/aminx-sampler/split_loop.mjs under Node with onnxruntime-web's wasm
// backend, against real exported graphs.
//
// This exists so the SHIPPING loop -- the same .mjs a browser would load -- is measured,
// not a Python re-implementation of it. G1 (run d2a06073) already established that the
// four-graph decomposition is correct by composing the graphs in Python; if that gate
// passes and this one fails, the fault is in the JavaScript, not the cut. That
// separation is the whole reason the two arms are distinct.
//
// It deliberately does NOT grade anything. It reads inputs, runs the loop, writes
// outputs, and leaves every comparison and threshold to the tracked Python gate that
// invokes it. A runner that graded itself could quietly decide what counts as agreement.
//
// Usage:
//   node run_split_node.mjs --cells <cells.json> --models <dir> --bucket 128 --out <out.json>

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { runSplitDecode } from "./split_loop.mjs";

function arg(name, fallback = undefined) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) {
    if (fallback === undefined) throw new Error(`missing --${name}`);
    return fallback;
  }
  return process.argv[i + 1];
}

const cellsPath = arg("cells");
const modelsDir = arg("models");
const bucket = Number(arg("bucket", "128"));
const outPath = arg("out");
const ortDir = arg("ort-dir");

// onnxruntime-web is resolved from an explicit directory rather than this module's own
// node_modules, because the sampler package intentionally has no dependency on it -- the
// browser supplies ORT. Keeping that true here means we measure the same module graph a
// page would.
const require = createRequire(pathToFileURL(join(ortDir, "package.json")));
const ortPkgDir = join(ortDir, "node_modules", "onnxruntime-web");
const ort = await import(pathToFileURL(join(ortPkgDir, "dist", "ort.wasm.min.mjs")).href);
void require;

ort.env.wasm.wasmPaths = pathToFileURL(join(ortPkgDir, "dist") + "/").href;
// Single-threaded: Node has no cross-origin isolation and the gate compares numbers, not
// speed. Thread count changes float accumulation order, so pinning it keeps this arm
// comparable with the ORT-CPU arm rather than introducing a second variable.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

function typedFrom(spec) {
  const { dtype, shape, data } = spec;
  if (dtype === "float32") return { shape, data: Float32Array.from(data) };
  if (dtype === "int32") return { shape, data: Int32Array.from(data) };
  if (dtype === "bool") return { shape, data: Uint8Array.from(data) };
  throw new Error(`unsupported dtype ${dtype}`);
}

// Infer the ONNX dtype from the TypedArray itself rather than carrying a dtype tag.
// Tensors reaching here come from three places -- the dumped cell JSON, intermediates
// built inside split_loop.mjs (Float32Array seqOh, Int32Array groupId, Uint8Array
// maskGroup), and ORT outputs fed straight back in -- and only the first has a tag. The
// constructor is unambiguous across all three, so this keeps one code path.
function toTensor(t) {
  const { shape, data } = t;
  if (data instanceof Float32Array) return new ort.Tensor("float32", data, shape);
  if (data instanceof Int32Array) return new ort.Tensor("int32", data, shape);
  if (data instanceof BigInt64Array) return new ort.Tensor("int64", data, shape);
  if (data instanceof Uint8Array) return new ort.Tensor("bool", data, shape);
  throw new Error(`unsupported tensor data type ${data?.constructor?.name ?? typeof data}`);
}

function fromTensor(t) {
  return { shape: Array.from(t.dims), data: t.data };
}

// Load from BYTES, not a path. onnxruntime-web treats a string argument as a URL and
// routes it through fetch(), which rejects a filesystem path under Node with
// "Failed to parse URL". Reading the file ourselves also mirrors what createSampler does
// in the browser (fetch -> arrayBuffer -> create), so both paths feed ORT the same shape
// of input.
const sessions = {};
for (const key of ["encoder", "wave", "decoder", "fuse"]) {
  const bytes = new Uint8Array(readFileSync(join(modelsDir, `p07_${key}_L${bucket}.onnx`)));
  sessions[key] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

/** Feed a session positionally: the graphs pin input ORDER, not names. */
async function runPositional(session, tensors) {
  const names = session.inputNames;
  if (names.length !== tensors.length) {
    throw new Error(`graph expects ${names.length} inputs, got ${tensors.length}`);
  }
  const feeds = {};
  names.forEach((n, i) => {
    feeds[n] = tensors[i];
  });
  const out = await session.run(feeds);
  return session.outputNames.map((n) => fromTensor(out[n]));
}

const cells = JSON.parse(readFileSync(cellsPath, "utf8"));
const results = [];

for (const cell of cells) {
  const a = Object.fromEntries(Object.entries(cell.arrays).map(([k, v]) => [k, typedFrom(v)]));

  const callbacks = {
    runEncoder: async ({ coords, mask, residue_index, chain_index }) => {
      const [node_features, edge_features, neighbor_indices] = await runPositional(
        sessions.encoder,
        [coords, mask, residue_index, chain_index].map(toTensor),
      );
      return { node_features, edge_features, neighbor_indices };
    },
    runWave: async ({ decoding_order, tie_group_map }) => {
      const o = await runPositional(
        sessions.wave,
        [decoding_order, tie_group_map].map(toTensor),
      );
      const [
        group_ids,
        group_positions,
        group_valid,
        position_valid,
        ar_mask,
        group_first_rank,
        pos_first_rank,
      ] = o;
      return {
        group_ids,
        group_positions,
        group_valid,
        position_valid,
        ar_mask,
        group_first_rank,
        pos_first_rank,
      };
    },
    runDecoder: async (inputs) => {
      const [logits] = await runPositional(
        sessions.decoder,
        [
          inputs.node_features,
          inputs.edge_features,
          inputs.neighbor_indices,
          inputs.mask,
          inputs.ar_mask,
          inputs.sequence_oh,
        ].map(toTensor),
      );
      return { logits };
    },
    runFuse: async (inputs) => {
      const [final_token, avg_stored] = await runPositional(
        sessions.fuse,
        [
          inputs.logits,
          inputs.cond_bias,
          inputs.mask_group,
          inputs.fixed_mask,
          inputs.fixed_tokens,
          inputs.group_id,
          inputs.temperature,
          inputs.gumbel_noise,
        ].map(toTensor),
      );
      return { final_token, avg_stored };
    },
  };

  const { tokens, logProbs } = await runSplitDecode(callbacks, {
    length: a.mask.shape[0],
    encoderInputs: {
      coords: a.coords,
      mask: a.mask,
      residue_index: a.residue_index,
      chain_index: a.chain_index,
    },
    waveInputs: { decoding_order: a.decoding_order, tie_group_map: a.tie_group_map },
    tieGroupMap: a.tie_group_map,
    mask: a.mask,
    condBias: a.bias,
    fixedMask: a.fixed_mask,
    fixedTokens: a.fixed_tokens,
    temperature: a.temperature,
    gumbelNoise: a.gumbel_noise,
  });
  results.push({
    name: cell.name,
    tokens: Array.from(tokens.data),
    log_probs: Array.from(logProbs.data),
  });
  process.stderr.write(`cell ${cell.name} done\n`);
}

for (const key of Object.keys(sessions)) await sessions[key].release();

writeFileSync(
  outPath,
  JSON.stringify(
    {
      bucket,
      n_cells: results.length,
      ort_version: ort.env.versions?.common ?? "unknown",
      num_threads: ort.env.wasm.numThreads,
      cells: results,
    },
    null,
    2,
  ),
);
process.stderr.write(`wrote ${outPath}\n`);
