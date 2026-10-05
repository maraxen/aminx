// Module Web Worker: runs P07 ProteinMPNN sampling off the main thread.
//
// Import path note: this file ships from site/, next to the static page,
// but the sampler lives in browser/aminx-sampler/ in the source checkout.
// The DEV import below is the only path in site/ that tools/build_site.py
// rewrites -- it copies aminx-sampler beside this file into dist/ and
// replaces this exact specifier with "./aminx-sampler/aminx_sampler.mjs".
// Every other site/*.mjs module avoids that coupling on purpose (see
// pdb_parse.mjs's header); this file cannot, because it is the one place
// that has to import the sampler at all.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { createSampler, pickBucket } from "../browser/aminx-sampler/aminx_sampler.mjs"; // BUILD_REWRITE_IMPORT

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

// bucket:numThreads -> Promise<sampler>. Session creation compiles the
// graph and is the expensive part (browser_integration.md), so a sampler is
// reused across every design in a run and across runs at the same bucket +
// thread count.
const samplers = new Map();

function getSampler(bucket, modelBase, numThreads) {
  const key = `${bucket}:${numThreads}`;
  if (!samplers.has(key)) {
    const url = `${modelBase}/p07_sample_L${bucket}.onnx`;
    samplers.set(key, createSampler(ort, url, { numThreads }));
  }
  return samplers.get(key);
}

// Reads a named ORT output, falling back to session output ORDER when the
// name is not present -- the export's output names are pinned by convention
// (see aminx_sampler.mjs's P07_INPUT_ORDER note) but a re-export could in
// principle change them, and this worker is the one place actually holding
// the `outputs` object aminx_sampler.mjs hands back unmodified.
function pickOutput(outputs, preferredName, fallbackIndex) {
  if (outputs[preferredName]) return outputs[preferredName];
  const keys = Object.keys(outputs);
  const key = keys[fallbackIndex];
  return key ? outputs[key] : undefined;
}

async function runJob(message) {
  const { id, structure, runspecs, modelBase, numThreads } = message;
  const started = Date.now();
  const length = structure.coords_shape ? structure.coords_shape[0] : structure.coords.length;
  const bucket = pickBucket(length);
  const sampler = await getSampler(bucket, modelBase, numThreads);
  const total = runspecs.length;

  for (let i = 0; i < runspecs.length; i += 1) {
    const runspec = runspecs[i];
    const { outputs, nReal, bucket: usedBucket } = await sampler.sample(structure, runspec);

    const tokensTensor = pickOutput(outputs, "cond_out_0", 0);
    const logProbsTensor = pickOutput(outputs, "log_softmax_out_0", 1);
    if (!tokensTensor || !logProbsTensor) {
      throw new Error(
        `sampler produced no usable outputs; got [${Object.keys(outputs).join(", ")}]`,
      );
    }

    self.postMessage({
      type: "design",
      id,
      index: i,
      seed: runspec.seed,
      temperature: runspec.temperature === undefined ? 0.1 : runspec.temperature,
      nReal,
      bucket: usedBucket,
      tokens: Array.from(tokensTensor.data.slice(0, nReal)),
      logProbs: Array.from(logProbsTensor.data.slice(0, nReal * 21)),
      done: i + 1,
      total,
      elapsedMs: Date.now() - started,
    });
  }

  self.postMessage({ type: "done", id, total, elapsedMs: Date.now() - started });
}

async function disposeAll() {
  const pending = Array.from(samplers.values());
  samplers.clear();
  for (const promise of pending) {
    try {
      const sampler = await promise;
      await sampler.release();
    } catch {
      // Session never finished creating, or already released -- nothing
      // left to free.
    }
  }
  self.postMessage({ type: "disposed" });
}

self.onmessage = (event) => {
  const message = event.data;
  if (message.type === "dispose") {
    disposeAll();
    return;
  }
  if (message.type !== "run") return;
  runJob(message).catch((error) => {
    self.postMessage({
      type: "error",
      id: message.id,
      message: error && error.message ? error.message : String(error),
    });
  });
};
