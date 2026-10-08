// Run arbitrary ONNX graphs under onnxruntime-web's wasm backend (Node), one case at a time.
//
// Generic sibling of run_split_node.mjs: no model knowledge, no grading. Each case names a
// model file and its inputs as headerless raw buffers fed POSITIONALLY to
// session.inputNames; every output is written back as a raw buffer with its ORT dtype and
// dims. The tracked Python script that invokes this does all comparison, so the runner
// cannot decide what counts as agreement.
//
// Models are loaded from BYTES (InferenceSession.create(str) treats a string as a URL under
// Node), wasmPaths is an absolute file URL, and numThreads is pinned to 1 so float
// accumulation order is a fixed variable (reference_ort-web-browser-gotchas).
//
// Usage:
//   node run_onnx_io.mjs --ort-dir <dir with node_modules/onnxruntime-web> \
//     --cases <cases.json> --out <result.json>
// cases.json: {"cases": [{"id": str, "model": path, "out_dir": path,
//                         "inputs": [{"file": path, "dtype": str, "dims": [int]}]}]}

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

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

const ARRAYS = {
  float32: Float32Array,
  int32: Int32Array,
  int64: BigInt64Array,
  bool: Uint8Array,
  uint8: Uint8Array,
};

function readInput(spec) {
  const Ctor = ARRAYS[spec.dtype];
  if (!Ctor) throw new Error(`unsupported input dtype ${spec.dtype}`);
  const bytes = readFileSync(spec.file);
  const data = new Ctor(bytes.buffer, bytes.byteOffset, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
  return new ort.Tensor(spec.dtype, Ctor.from(data), spec.dims);
}

const { cases } = JSON.parse(readFileSync(arg("cases"), "utf8"));
const results = [];
for (const c of cases) {
  const record = { id: c.id, ok: false, error: null, outputs: [], num_threads: null };
  try {
    const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(c.model)));
    record.num_threads = ort.env.wasm.numThreads;
    if (session.inputNames.length !== c.inputs.length) {
      throw new Error(`${c.model} declares ${session.inputNames.length} inputs, case gives ${c.inputs.length}`);
    }
    const feeds = {};
    session.inputNames.forEach((name, i) => {
      feeds[name] = readInput(c.inputs[i]);
    });
    const out = await session.run(feeds);
    mkdirSync(c.out_dir, { recursive: true });
    session.outputNames.forEach((name, k) => {
      const t = out[name];
      const file = join(c.out_dir, `${c.id}__${k}.bin`);
      const view = t.data;
      writeFileSync(file, Buffer.from(view.buffer, view.byteOffset, view.byteLength));
      record.outputs.push({ file, dtype: t.type, dims: Array.from(t.dims) });
    });
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.message ? err.message : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ ort_web_version: ort.env.versions?.web ?? null, results }, null, 2));
