// Phase 2a, T8: in-page ORT Web wasm benchmark cell.
//
// Loaded by run_bench.mjs inside one of exactly two contexts (ctx_t1 / ctx_tN).
// `numThreads` is taken from the query string and written to `ort.env.wasm.numThreads`
// before the first InferenceSession.create in this page (wasm-init is per context).
// Worker construction is wrapped first so `threadsObserved` counts pthread Workers
// spawned at InferenceSession.create (same pattern as tests/thread-probe.mjs).
const workers = [];
const NativeWorker = globalThis.Worker;
if (typeof NativeWorker === "function") {
  globalThis.Worker = function BenchWorker(...args) {
    const worker = new NativeWorker(...args);
    workers.push(worker);
    return worker;
  };
  globalThis.Worker.prototype = NativeWorker.prototype;
}

import * as ort from "./ort/ort.wasm.min.mjs";

const DTYPE_ARRAYS = {
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

function requestedThreads() {
  const params = new URLSearchParams(location.search);
  const requested = Number(params.get("numThreads") || "1");
  return Number.isFinite(requested) && requested > 0 ? requested : 1;
}

const threadsRequested = requestedThreads();
ort.env.wasm.numThreads = threadsRequested;
ort.env.wasm.proxy = false;

let threadsObserved = null;

function noteThreadsAfterInit() {
  if (threadsObserved === null) {
    threadsObserved = workers.length + 1;
  }
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`fetch ${url} failed: ${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

async function loadFeeds(session, inputs) {
  const feeds = {};
  for (let i = 0; i < inputs.length; i += 1) {
    const spec = inputs[i];
    const ArrayCtor = DTYPE_ARRAYS[spec.dtype];
    if (!ArrayCtor) {
      throw new Error(`unsupported input dtype: ${spec.dtype}`);
    }
    const buf = await fetchArrayBuffer(spec.dataFile);
    const data = new ArrayCtor(buf);
    feeds[session.inputNames[i]] = new ort.Tensor(spec.dtype, data, spec.dims);
  }
  return feeds;
}

function busyWait(ms) {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    // Planted busy-wait sits inside the timed region (T8 control arm).
  }
}

function wasmHeapBytes() {
  try {
    const memory = ort.wasm && ort.wasm.memory;
    if (memory && memory.buffer) {
      return memory.buffer.byteLength;
    }
  } catch (err) {
    return { available: false, error: String(err) };
  }
  return null;
}

async function peakMemory() {
  if (typeof performance.measureUserAgentSpecificMemory !== "function") {
    return { available: false, bytes: null };
  }
  try {
    const measured = await performance.measureUserAgentSpecificMemory();
    return { available: true, bytes: measured.bytes };
  } catch (err) {
    return { available: false, bytes: null, error: String(err) };
  }
}

async function timeCell(spec) {
  const nWarmup = spec.nWarmup;
  const nIter = spec.nIter;
  const plantedMs = Number(spec.plantedBusyWaitMs || 0);
  const fetchStart = performance.now();
  const modelBuf = await fetchArrayBuffer(spec.onnx);
  const fetchLoadMs = performance.now() - fetchStart;

  const createStart = performance.now();
  const session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });
  const sessionCreateMs = performance.now() - createStart;
  noteThreadsAfterInit();

  const feeds = await loadFeeds(session, spec.inputs);
  const firstStart = performance.now();
  await session.run(feeds);
  const firstInferenceMs = performance.now() - firstStart;

  const isolated = self.crossOriginIsolated === true;
  if (!isolated) {
    return {
      ok: false,
      error: "crossOriginIsolated is not true",
      crossOriginIsolated: false,
      threadsRequested,
      threadsObserved,
      threadsMismatch: threadsObserved !== threadsRequested,
      phases: {
        fetch_load_ms: fetchLoadMs,
        session_create_ms: sessionCreateMs,
        first_inference_ms: firstInferenceMs,
        steady_ms: [],
      },
    };
  }

  for (let i = 0; i < nWarmup; i += 1) {
    await session.run(feeds);
  }
  const steadyMs = [];
  for (let i = 0; i < nIter; i += 1) {
    const start = performance.now();
    await session.run(feeds);
    if (plantedMs > 0) {
      busyWait(plantedMs);
    }
    steadyMs.push(performance.now() - start);
  }

  const memory = await peakMemory();
  return {
    ok: true,
    crossOriginIsolated: true,
    threadsRequested,
    threadsObserved,
    threadsMismatch: threadsObserved !== threadsRequested,
    phases: {
      fetch_load_ms: fetchLoadMs,
      session_create_ms: sessionCreateMs,
      first_inference_ms: firstInferenceMs,
      steady_ms: steadyMs,
      n_warmup: nWarmup,
      n_iter: steadyMs.length,
    },
    peakMemory: memory,
    wasmHeapBytes: wasmHeapBytes(),
  };
}

window.__timeCell = timeCell;
window.__benchInfo = () => ({
  hardwareConcurrency: navigator.hardwareConcurrency || 1,
  crossOriginIsolated: self.crossOriginIsolated === true,
  threadsRequested,
  threadsObserved,
  numThreadsReadback: ort.env.wasm.numThreads,
});
window.__BENCH_READY__ = true;
