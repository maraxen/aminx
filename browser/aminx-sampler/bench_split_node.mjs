// Time the split pipeline and the monolith on the same inputs, same backend.
//
// The first thing this does is prove its own clock works. A previous benchmark in this
// project produced timings that could not be trusted because its timer control was never
// established, so here a known delay is planted into a measured region and reported back;
// the Python gate refuses to publish any number unless the timer saw it. A stopwatch that
// cannot detect a deliberate 250 ms pause cannot be trusted to measure a 40 ms one.
//
// It measures, it does not judge. No thresholds, no verdicts, no "faster/slower" claim --
// those belong to the tracked gate, which is where the pre-registration lives.
//
// Usage:
//   node bench_split_node.mjs --cells <cells.json> --models <dir> --monolith <file>
//        --bucket 128 --reps 5 --planted-ms 250 --out <out.json> --ort-dir <dir>

import { readFileSync, writeFileSync } from "node:fs";
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
const monolithPath = arg("monolith");
const bucket = Number(arg("bucket", "128"));
const reps = Number(arg("reps", "5"));
const plantedMs = Number(arg("planted-ms", "250"));
const outPath = arg("out");
const ortDir = arg("ort-dir");
const numThreads = Number(arg("threads", "1"));

const ortPkgDir = join(ortDir, "node_modules", "onnxruntime-web");
const ort = await import(pathToFileURL(join(ortPkgDir, "dist", "ort.wasm.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(join(ortPkgDir, "dist") + "/").href;
// Threads are a measured VARIABLE, not a fixed 1. Thread count changes float
// accumulation order, so a run at one setting is only comparable with another at the
// same setting -- the gate records it and the sidecar scopes every figure to it.
ort.env.wasm.numThreads = numThreads;
ort.env.logLevel = "error";

const now = () => Number(process.hrtime.bigint()) / 1e6; // ms, monotonic
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- Timer control -----------------------------------------------------------------
// Measure a region containing a deliberate `plantedMs` pause with the SAME clock and the
// same await machinery the real measurements use. If this does not come back at least
// plantedMs, every number below is worthless and the gate must refuse them.
const ctrlStart = now();
await sleep(plantedMs);
const ctrlMeasuredMs = now() - ctrlStart;

function typedFrom(spec) {
  const { dtype, shape, data } = spec;
  if (dtype === "float32") return { shape, data: Float32Array.from(data) };
  if (dtype === "int32") return { shape, data: Int32Array.from(data) };
  if (dtype === "bool") return { shape, data: Uint8Array.from(data) };
  throw new Error(`unsupported dtype ${dtype}`);
}

function toTensor(t) {
  const { shape, data } = t;
  if (data instanceof Float32Array) return new ort.Tensor("float32", data, shape);
  if (data instanceof Int32Array) return new ort.Tensor("int32", data, shape);
  if (data instanceof Uint8Array) return new ort.Tensor("bool", data, shape);
  throw new Error(`unsupported tensor data type ${data?.constructor?.name}`);
}

const fromTensor = (t) => ({ shape: Array.from(t.dims), data: t.data });

async function runPositional(session, tensors) {
  const feeds = {};
  session.inputNames.forEach((n, i) => {
    feeds[n] = tensors[i];
  });
  const out = await session.run(feeds);
  return session.outputNames.map((n) => fromTensor(out[n]));
}

// ---- Session creation, timed individually ------------------------------------------
const sessionCreateMs = {};
const sessions = {};
for (const key of ["encoder", "wave", "decoder", "fuse"]) {
  const bytes = new Uint8Array(readFileSync(join(modelsDir, `p07_${key}_L${bucket}.onnx`)));
  const t0 = now();
  sessions[key] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  sessionCreateMs[key] = now() - t0;
}
const monoBytes = new Uint8Array(readFileSync(monolithPath));
const tMono0 = now();
const monoSession = await ort.InferenceSession.create(monoBytes, {
  executionProviders: ["wasm"],
});
sessionCreateMs.monolith = now() - tMono0;

const P07_KEYS = [
  "coords", "mask", "residue_index", "chain_index", "gumbel_noise", "decoding_order",
  "bias", "fixed_mask", "fixed_tokens", "temperature", "tie_group_map",
];

const cells = JSON.parse(readFileSync(cellsPath, "utf8"));
const perCell = [];

for (const cell of cells) {
  const a = Object.fromEntries(Object.entries(cell.arrays).map(([k, v]) => [k, typedFrom(v)]));

  const callbacks = {
    runEncoder: async (i) => {
      const [node_features, edge_features, neighbor_indices] = await runPositional(
        sessions.encoder,
        [i.coords, i.mask, i.residue_index, i.chain_index].map(toTensor),
      );
      return { node_features, edge_features, neighbor_indices };
    },
    runWave: async (i) => {
      const o = await runPositional(sessions.wave, [i.decoding_order, i.tie_group_map].map(toTensor));
      return {
        group_ids: o[0], group_positions: o[1], group_valid: o[2], position_valid: o[3],
        ar_mask: o[4], group_first_rank: o[5], pos_first_rank: o[6],
      };
    },
    runDecoder: async (i) => {
      const [logits] = await runPositional(sessions.decoder, [
        i.node_features, i.edge_features, i.neighbor_indices, i.mask, i.ar_mask, i.sequence_oh,
      ].map(toTensor));
      return { logits };
    },
    runFuse: async (i) => {
      const [final_token, avg_stored] = await runPositional(sessions.fuse, [
        i.logits, i.cond_bias, i.mask_group, i.fixed_mask, i.fixed_tokens, i.group_id,
        i.temperature, i.gumbel_noise,
      ].map(toTensor));
      return { final_token, avg_stored };
    },
  };

  const params = {
    length: a.mask.shape[0],
    encoderInputs: {
      coords: a.coords, mask: a.mask, residue_index: a.residue_index, chain_index: a.chain_index,
    },
    waveInputs: { decoding_order: a.decoding_order, tie_group_map: a.tie_group_map },
    tieGroupMap: a.tie_group_map,
    mask: a.mask,
    condBias: a.bias,
    fixedMask: a.fixed_mask,
    fixedTokens: a.fixed_tokens,
    temperature: a.temperature,
    gumbelNoise: a.gumbel_noise,
  };

  const splitMs = [];
  const monoMs = [];
  for (let r = 0; r < reps; r += 1) {
    const s0 = now();
    await runSplitDecode(callbacks, params);
    splitMs.push(now() - s0);

    const feeds = {};
    monoSession.inputNames.forEach((n, i) => {
      feeds[n] = toTensor(a[P07_KEYS[i]]);
    });
    const m0 = now();
    await monoSession.run(feeds);
    monoMs.push(now() - m0);
  }
  perCell.push({ name: cell.name, split_ms: splitMs, monolith_ms: monoMs });
  process.stderr.write(`bench ${cell.name} split=${splitMs.map((x) => x.toFixed(0))} mono=${monoMs.map((x) => x.toFixed(0))}\n`);
}

for (const k of Object.keys(sessions)) await sessions[k].release();
await monoSession.release();

writeFileSync(outPath, JSON.stringify({
  bucket,
  reps,
  planted_ms: plantedMs,
  ctrl_measured_ms: ctrlMeasuredMs,
  session_create_ms: sessionCreateMs,
  num_threads: ort.env.wasm.numThreads,
  ort_version: ort.env.versions?.common ?? "unknown",
  peak_rss_bytes: process.memoryUsage().rss,
  cells: perCell,
}, null, 2));
process.stderr.write(`wrote ${outPath}\n`);
