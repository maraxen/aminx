// Full browser path under Node with onnxruntime-web (wasm, 1 thread), no Python inputs:
//   PDB text -> buildPottsInputs -> applyPottsControls -> createPottsSampler().sample
//
// No grading here; the tracked Python gate (scripts/browser_validation/potts_knobs_gate.py)
// compares every output against the JAX sampler.
//
// Usage: node run_potts_browser_node.mjs --ort-dir <dir> --models <dir> --cells <cells.json> --out <out.json>
// cells.json: {"cells": [{"id", "pdb", "bucket", "controls": {...}, "options": {refine, temperature,
//   optimizationTemperature}, "out_dir", "noise": {randn|uniforms|refineUniforms: {file, dtype, dims}}}]}

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { applyPottsControls } from "./potts_controls.mjs";
import { buildPottsInputs } from "./potts_inputs.mjs";
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

function readFloat32(spec) {
  const bytes = readFileSync(spec.file);
  return Float32Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
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
    const built = buildPottsInputs(readFileSync(cell.pdb, "utf8"), cell.bucket);
    const controlled = applyPottsControls(built, cell.controls || {});
    const inputs = {};
    for (const [name, t] of Object.entries(controlled.inputs ?? controlled)) inputs[name] = { data: t.data };
    const noise = {};
    for (const [name, spec] of Object.entries(cell.noise)) noise[name] = readFloat32(spec);
    const out = await samplers.get(cell.bucket).sample(inputs, noise, cell.options || {});
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
    record.l_total = built.l_total;
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.stack ? err.stack : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ num_threads: ort.env.wasm.numThreads, results }, null, 2));
