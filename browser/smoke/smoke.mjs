// Phase 0, T3: browser-side ORT Web (wasm EP) smoke harness.
//
// Loaded by index.html inside a headless-Chromium page (run_smoke.mjs). Reads
// a manifest of models to exercise (written by
// scripts/browser_validation/ort_web_smoke.py from the T2 jax2onnx-spike
// artifacts), runs each one through onnxruntime-web's wasm execution
// provider with `numThreads = 1`, and reports raw outputs plus environment
// facts (browser UA, ORT Web version, `self.crossOriginIsolated`) back to the
// Node/Playwright side. All comparison against the JAX baseline happens on
// the Python side (ort_web_smoke.py) -- this module only produces evidence.
//
// Model manifest entries are generic ({name, onnx, inputs, outputOrder}) so
// this same harness runs the P05 kernel and its perturbed-weight control
// twin unchanged, once a converted P05 onnx exists (Phase 3) -- inputs are
// fed by POSITION (`session.inputNames[i]`), matching jax2onnx's own
// argument-order-preserving naming, not by a hardcoded ONNX tensor name.
import * as ort from "./ort/ort.wasm.min.mjs";

// wasm-only backend (no webgl/webgpu): this is deliberately the "wasm EP"
// smoke, not a general ORT Web capability probe.
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const DTYPE_ARRAYS = {
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
  bool: Uint8Array,
};

async function loadJsonTensor(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  }
  const spec = await res.json();
  if (spec.dtype === "int64") {
    const data = BigInt64Array.from(spec.data.map((v) => BigInt(v)));
    return new ort.Tensor("int64", data, spec.dims);
  }
  const ArrayCtor = DTYPE_ARRAYS[spec.dtype];
  if (!ArrayCtor) {
    throw new Error(`unsupported dtype in manifest: ${spec.dtype}`);
  }
  const data = ArrayCtor.from(spec.data);
  return new ort.Tensor(spec.dtype, data, spec.dims);
}

function tensorToPlain(tensor) {
  const data = Array.from(tensor.data, (v) => (typeof v === "bigint" ? Number(v) : v));
  return { dims: tensor.dims, dtype: tensor.type, data };
}

async function runModel(modelSpec) {
  const session = await ort.InferenceSession.create(modelSpec.onnx, {
    executionProviders: ["wasm"],
  });
  if (session.inputNames.length !== modelSpec.inputs.length) {
    throw new Error(
      `${modelSpec.name}: session has ${session.inputNames.length} inputs, ` +
        `manifest declares ${modelSpec.inputs.length}`,
    );
  }
  const feeds = {};
  for (let i = 0; i < modelSpec.inputs.length; i += 1) {
    const sessionInputName = session.inputNames[i];
    feeds[sessionInputName] = await loadJsonTensor(modelSpec.inputs[i].dataFile);
  }
  const outputMap = await session.run(feeds);
  const outputs = {};
  session.outputNames.forEach((sessionOutputName, i) => {
    const label = (modelSpec.outputOrder && modelSpec.outputOrder[i]) || sessionOutputName;
    outputs[label] = tensorToPlain(outputMap[sessionOutputName]);
  });
  return outputs;
}

async function main() {
  const result = {
    userAgent: navigator.userAgent,
    crossOriginIsolated: self.crossOriginIsolated === true,
    ortWebVersion: (ort.env.versions && (ort.env.versions.web || ort.env.versions.common)) || null,
    ortVersions: ort.env.versions || null,
    numThreads: ort.env.wasm.numThreads,
    executionProvider: "wasm",
    models: {},
  };
  try {
    const manifestRes = await fetch("./data/manifest.json");
    const manifest = await manifestRes.json();
    for (const modelSpec of manifest.models) {
      try {
        // eslint-disable-next-line no-await-in-loop
        result.models[modelSpec.name] = { ok: true, outputs: await runModel(modelSpec) };
      } catch (err) {
        result.models[modelSpec.name] = {
          ok: false,
          error: String((err && err.stack) || err),
        };
      }
    }
    window.__SMOKE_RESULT__ = result;
  } catch (err) {
    window.__SMOKE_ERROR__ = String((err && err.stack) || err);
  } finally {
    window.__SMOKE_DONE__ = true;
  }
}

main();
