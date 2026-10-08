// Drive the SHIPPING potts_sampler.mjs under Node with onnxruntime-web (wasm, 1 thread).
//
// No grading here: read cells, run the loop, write raw outputs. The tracked Python gate
// (scripts/browser_validation/potts_loop_gate.py) does every comparison.
//
// Usage: node run_potts_loop_node.mjs --ort-dir <dir with node_modules/onnxruntime-web>
//          --models <dir with MANIFEST.json + .onnx> --cells <cells.json> --out <result.json>
// cells.json: {"cells": [{"id", "bucket", "out_dir", "refine": bool,
//   "inputs": {name: {file, dtype, dims}}, "noise": {randn|uniforms|refineUniforms: {file, dtype, dims}}}]}

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { createPottsSampler } from "./potts_sampler.mjs";

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) throw new Error(`missing --${name}`);
  return process.argv[i + 1];
}

const ortPkgDir = join(arg("ort-dir"), "node_modules", "onnxruntime-web");
const ort = await import(pathToFileURL(join(ortPkgDir, "dist", "ort.wasm.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(join(ortPkgDir, "dist") + "/").href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const ARRAYS = { float32: Float32Array, int32: Int32Array, bool: Uint8Array };

function readArray(spec) {
  const Ctor = ARRAYS[spec.dtype];
  if (!Ctor) throw new Error(`unsupported dtype ${spec.dtype}`);
  const bytes = readFileSync(spec.file);
  return Ctor.from(new Ctor(bytes.buffer, bytes.byteOffset, bytes.byteLength / Ctor.BYTES_PER_ELEMENT));
}

const modelsDir = arg("models");
const manifest = JSON.parse(readFileSync(join(modelsDir, "MANIFEST.json"), "utf8"));
const { cells } = JSON.parse(readFileSync(arg("cells"), "utf8"));
const samplers = new Map();
const results = [];
for (const cell of cells) {
  const record = { id: cell.id, ok: false, error: null, outputs: {} };
  try {
    if (!samplers.has(cell.bucket)) {
      const entry = manifest.buckets.find((b) => b.bucket === cell.bucket);
      const models = {};
      for (const key of ["encode", "energy", "decode", "refine"]) {
        models[key] = new Uint8Array(readFileSync(join(modelsDir, entry.graphs[key].file)));
      }
      samplers.set(cell.bucket, await createPottsSampler(ort, manifest, cell.bucket, models));
    }
    const inputs = {};
    for (const [name, spec] of Object.entries(cell.inputs)) inputs[name] = { data: readArray(spec) };
    const noise = {};
    for (const [name, spec] of Object.entries(cell.noise)) noise[name] = readArray(spec);
    const out = await samplers.get(cell.bucket).sample(inputs, noise, { refine: cell.refine });
    mkdirSync(cell.out_dir, { recursive: true });
    for (const [name, value] of Object.entries(out)) {
      if (typeof value === "number") {
        record.outputs[name] = { value };
        continue;
      }
      const file = join(cell.out_dir, `${cell.id}__${name}.bin`);
      writeFileSync(file, Buffer.from(value.buffer, value.byteOffset, value.byteLength));
      record.outputs[name] = { file, dtype: "int32", dims: [value.length] };
    }
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.stack ? err.stack : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ num_threads: ort.env.wasm.numThreads, results }, null, 2));
