// Drive the SHIPPING protonpotts_scorer.mjs under Node with onnxruntime-web (wasm, 1 thread).
//
// No grading here: read cases, run the scorer from PDB text, write raw outputs. The tracked Python gate
// (scripts/browser_validation/protonpotts_score_gate.py) does every comparison.
//
// Usage: node run_protonpotts_node.mjs --ort-dir <dir with node_modules/onnxruntime-web>
//          --models <dir with MANIFEST.json + .onnx> --cases <cases.json> --out <result.json>
// cases.json: {"cases": [{"id", "bucket", "pdb_file", "labels": [..]|null, "variants": [[token name, ...], ...],
//                          "out_dir"}]}

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { createProtonPottsScorer } from "./protonpotts_scorer.mjs";

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

const modelsDir = arg("models");
const manifest = JSON.parse(readFileSync(join(modelsDir, "MANIFEST.json"), "utf8"));
const { cases } = JSON.parse(readFileSync(arg("cases"), "utf8"));
const scorers = new Map();
const results = [];
for (const c of cases) {
  const record = { id: c.id, ok: false, error: null, energies: null, tokens: null, l_total: null };
  try {
    if (!scorers.has(c.bucket)) {
      const entry = manifest.buckets.find((b) => b.bucket === c.bucket);
      const models = {};
      for (const key of ["table", "energy"]) {
        models[key] = new Uint8Array(readFileSync(join(modelsDir, entry.graphs[key].file)));
      }
      scorers.set(c.bucket, await createProtonPottsScorer(ort, manifest, c.bucket, models));
    }
    const out = await scorers.get(c.bucket).score(readFileSync(c.pdb_file, "utf8"), c.labels, c.variants);
    mkdirSync(c.out_dir, { recursive: true });
    const file = join(c.out_dir, `${c.id}__energies.bin`);
    writeFileSync(file, Buffer.from(out.energies.buffer, out.energies.byteOffset, out.energies.byteLength));
    record.energies = { file, dtype: "float32", dims: [out.energies.length] };
    record.tokens = out.tokens;
    record.l_total = out.lTotal;
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.stack ? err.stack : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ num_threads: ort.env.wasm.numThreads, results }, null, 2));
