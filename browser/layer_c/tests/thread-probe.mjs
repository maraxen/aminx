// Phase 2a, T5a: wasm-init thread-assertion probe page script (Definitions,
// "Verification test").
//
// Wraps `self.Worker` BEFORE importing onnxruntime-web, so every pthread Worker the wasm
// backend spawns for its thread pool is counted. `window.__workers` and
// `window.__createAndRun` are exposed so threads.spec.mjs can both read the current
// cumulative worker count and trigger a SECOND `InferenceSession.create` in the SAME
// page (to test whether a later `numThreads` write after the first create still changes
// the backend's actual thread pool, or only the JS-visible property).
const workers = [];
const NativeWorker = self.Worker;
self.Worker = function ProbeWorker(...args) {
  const w = new NativeWorker(...args);
  workers.push(w);
  return w;
};
self.Worker.prototype = NativeWorker.prototype;
window.__workers = workers;

import * as ort from "./ort/ort.wasm.min.mjs";

window.__ort = ort;

async function createAndRun(setNumThreadsTo) {
  if (typeof setNumThreadsTo === "number") {
    ort.env.wasm.numThreads = setNumThreadsTo;
  }
  const buf = await (await fetch("./models/topk_lattice.onnx")).arrayBuffer();
  const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  const x = new Float32Array(128 * 128);
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", x, [128, 128]) };
  await session.run(feeds);
  return {
    numThreadsProperty: ort.env.wasm.numThreads,
    workerCount: workers.length,
    // The observed thread count: 1 (main thread) plus every pthread Worker spawned so
    // far in this page. If ORT Web genuinely fixes the pool size at the FIRST
    // session-create (Definitions: "numThreads is fixed when the wasm backend
    // initialises"), this stays constant across later createAndRun() calls in the same
    // page regardless of what `setNumThreadsTo` requests.
    threadsObserved: workers.length + 1,
  };
}
window.__createAndRun = createAndRun;

async function main() {
  const params = new URLSearchParams(location.search);
  const requested = Number(params.get("numThreads") || "1");
  ort.env.wasm.numThreads = requested;
  ort.env.wasm.proxy = false;
  try {
    const result = await createAndRun();
    window.__THREAD_RESULT__ = { requested, ...result };
  } catch (err) {
    window.__THREAD_ERROR__ = String((err && err.stack) || err);
  } finally {
    window.__THREAD_DONE__ = true;
  }
}

main();
