// Phase 2a, T5a: browser-side ORT Web (wasm EP) parity harness.
//
// Loaded by index.html inside a headless-Chromium page (run_parity.mjs). Reads a
// `cells.json` manifest (written by scripts/browser_validation/layer_c_common.py) whose
// `cells` entries are generic ({name, onnx, inputs, outputs}), same convention as Phase
// 0's smoke.mjs (inputs fed by POSITION -- `session.inputNames[i]` -- not by a hardcoded
// ONNX tensor name), so this harness runs both the P03 and P04 wrapper artifacts
// unchanged. For each cell it runs the model through onnxruntime-web's wasm execution
// provider and POSTs every output's RAW bytes to `/results/<cell.name>__<label>.bin`
// (not JSON -- the task spec asks for raw bytes so a P04 log-prob array at L=1024 does
// not round-trip through a JSON array). All numerical comparison against the ORT-CPU
// reference happens on the Python side (layer_c_calibrate.py) -- this module only
// produces evidence.
import * as ort from "./ort/ort.wasm.min.mjs";

const DTYPE_ARRAYS = {
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

function paramNumThreads() {
  const params = new URLSearchParams(location.search);
  const requested = Number(params.get("numThreads") || "1");
  return Number.isFinite(requested) && requested > 0 ? requested : 1;
}

async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  }
  return res.arrayBuffer();
}

async function loadRawTensor(spec) {
  const buf = await fetchArrayBuffer(spec.dataFile);
  const ArrayCtor = DTYPE_ARRAYS[spec.dtype];
  if (!ArrayCtor) {
    throw new Error(`unsupported input dtype in cell manifest: ${spec.dtype}`);
  }
  const data = new ArrayCtor(buf);
  return new ort.Tensor(spec.dtype, data, spec.dims);
}

function tensorRawBytes(tensor, expectedDtype) {
  if (tensor.type !== expectedDtype) {
    throw new Error(`output dtype mismatch: session says ${tensor.type}, manifest says ${expectedDtype}`);
  }
  return tensor.data.buffer.slice(
    tensor.data.byteOffset || 0,
    (tensor.data.byteOffset || 0) + tensor.data.byteLength,
  );
}

async function postResult(name, body) {
  const res = await fetch(`/results/${encodeURIComponent(name)}`, {
    method: "POST",
    body,
  });
  if (!res.ok) {
    throw new Error(`POST /results/${name} failed: ${res.status} ${res.statusText}`);
  }
}

async function runCell(cell) {
  const modelBuf = await fetchArrayBuffer(cell.onnx);
  const sessionOptions = { executionProviders: ["wasm"] };
  if (cell.profile) {
    // T6 (--profile mode): opt-in per-cell, via the cells.json manifest (NOT a URL
    // query param) -- so a caller that never sets `cell.profile` (every existing
    // layer_c_calibrate.py/layer_c_parity.py cell) takes the IDENTICAL code path as
    // before this change, byte-for-byte.
    sessionOptions.enableProfiling = true;
  }
  const session = await ort.InferenceSession.create(modelBuf, sessionOptions);
  if (session.inputNames.length !== cell.inputs.length) {
    throw new Error(
      `${cell.name}: session has ${session.inputNames.length} inputs, manifest declares ${cell.inputs.length}`,
    );
  }
  if (session.outputNames.length !== cell.outputs.length) {
    throw new Error(
      `${cell.name}: session has ${session.outputNames.length} outputs, manifest declares ${cell.outputs.length}`,
    );
  }
  const feeds = {};
  for (let i = 0; i < cell.inputs.length; i += 1) {
    feeds[session.inputNames[i]] = await loadRawTensor(cell.inputs[i]);
  }
  const outputMap = await session.run(feeds);
  for (let i = 0; i < cell.outputs.length; i += 1) {
    const outSpec = cell.outputs[i];
    const tensor = outputMap[session.outputNames[i]];
    const bytes = tensorRawBytes(tensor, outSpec.dtype);
    // eslint-disable-next-line no-await-in-loop
    await postResult(`${cell.name}__${outSpec.label}.bin`, bytes);
  }
  if (cell.profile) {
    // onnxruntime-web's wasm EP exposes NO JS-readable per-node profiling event array:
    // `endProfiling()` only frees an internal Emscripten MEMFS file handle (confirmed
    // against the bundled 1.30.0 dist, `ort.wasm.mjs`'s
    // `endProfiling = (sessionId) => { ... wasm2._OrtFree(profileFileName); }` --
    // there is no public API call that reads that file's bytes back into JS). We still
    // call it (so a future ORT Web version that DOES expose events is exercised
    // identically), but report unavailability honestly rather than fabricating
    // per-op data -- layer_b_profile.py reads `profileEventsAvailable` to decide its
    // own `web_per_op_available` result field.
    session.endProfiling();
    return { ok: true, profileEventsAvailable: false };
  }
  return { ok: true };
}

async function webgpuProbe() {
  // Capability probe only (D-E): record whether the API exists and whether
  // requestAdapter() resolves to a truthy adapter. No numbers, no further use.
  if (!("gpu" in navigator)) {
    return { available: false, adapter: false, error: null };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return { available: true, adapter: adapter != null, error: null };
  } catch (err) {
    return { available: true, adapter: false, error: String((err && err.stack) || err) };
  }
}

async function main() {
  const requested = paramNumThreads();
  // wasm-init thread assertion (Definitions): numThreads is fixed when the wasm backend
  // initialises, so it must be set before the FIRST InferenceSession.create in this
  // page. `proxy = false` keeps this the plain (non-proxy-worker) wasm EP, matching
  // Phase 0's smoke harness convention.
  ort.env.wasm.numThreads = requested;
  ort.env.wasm.proxy = false;

  const evidence = {
    sessionId: crypto.randomUUID(),
    userAgent: navigator.userAgent,
    crossOriginIsolated: self.crossOriginIsolated === true,
    ortWebVersion: (ort.env.versions && (ort.env.versions.web || ort.env.versions.common)) || null,
    ortVersions: ort.env.versions || null,
    executionProvider: "wasm",
    threadAssertion: { requested, readbackAfterInit: null },
    webgpu: null,
    cells: {},
  };

  try {
    const manifestRes = await fetch("./cells.json");
    const manifest = await manifestRes.json();
    for (const cell of manifest.cells) {
      try {
        // eslint-disable-next-line no-await-in-loop
        evidence.cells[cell.name] = await runCell(cell);
      } catch (err) {
        evidence.cells[cell.name] = { ok: false, error: String((err && err.stack) || err) };
      }
      if (evidence.threadAssertion.readbackAfterInit === null) {
        evidence.threadAssertion.readbackAfterInit = ort.env.wasm.numThreads;
      }
    }
    evidence.webgpu = await webgpuProbe();
    window.__PARITY_RESULT__ = evidence;
  } catch (err) {
    window.__PARITY_ERROR__ = String((err && err.stack) || err);
  } finally {
    try {
      await postResult(
        "evidence.json",
        new TextEncoder().encode(JSON.stringify(window.__PARITY_RESULT__ || evidence, null, 2)),
      );
    } catch {
      // best-effort: the harness driver also reads window.__PARITY_RESULT__ directly.
    }
    window.__PARITY_DONE__ = true;
  }
}

main();
