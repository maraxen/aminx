// Drive the SHIPPING decoder_sampler.mjs under Node with onnxruntime-web (wasm, 1 thread).
//
// No grading here: read cases, run ProtonPottsSampler.sample from PDB text, write the outputs. The tracked Python gate
// (scripts/browser_validation/protonpotts_sampler_gate.py) does every comparison.
//
// Usage: node run_sampler_node.mjs --ort-dir <dir with node_modules/onnxruntime-web> --models <decoder dir> --cases <cases.json> --out <result.json>
// cases.json: {"cases": [{"id", "bucket", "pdb_file", "designed": [0|1,..]|null, "temperature": number|[..],
//                          "bias": [..]|null, "uniforms": [..]|null, "noise": [..]|null, "seed": int}]}

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { ProtonPottsSampler } from "./decoder_sampler.mjs";

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

const dir = arg("models");
const manifest = JSON.parse(readFileSync(join(dir, "MANIFEST.json"), "utf8"));
const { cases } = JSON.parse(readFileSync(arg("cases"), "utf8"));

const bytes = (file) => new Uint8Array(readFileSync(join(dir, file)));
const samplers = new Map();
const results = [];
for (const c of cases) {
  const record = { id: c.id, ok: false, error: null };
  try {
    if (!samplers.has(c.bucket)) {
      const entry = manifest.buckets.find((b) => b.bucket === c.bucket);
      samplers.set(
        c.bucket,
        await ProtonPottsSampler.create(ort, manifest, c.bucket, {
          encode: bytes(entry.graphs.encode.file),
          decode: bytes(entry.graphs.decode.file),
        }),
      );
    }
    const options = { seed: c.seed ?? 0, temperature: Array.isArray(c.temperature) ? Float32Array.from(c.temperature) : c.temperature };
    if (c.designed) options.designed = c.designed.map((x) => x === 1);
    if (c.bias) options.bias = Float32Array.from(c.bias);
    if (c.uniforms) {
      options.uniforms = Float32Array.from(c.uniforms);
      options.noise = Float32Array.from(c.noise);
    }
    const out = await samplers.get(c.bucket).sample(readFileSync(c.pdb_file, "utf8"), null, options);
    record.sequence = Array.from(out.sequence);
    record.decoding_order = Array.from(out.decodingOrder);
    record.log_probs = Array.from(out.logProbs);
    record.l_total = out.lTotal;
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.stack ? err.stack : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ num_threads: ort.env.wasm.numThreads, results }));
