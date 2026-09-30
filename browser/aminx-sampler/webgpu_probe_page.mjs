// In-page half of the WebGPU capability probe. Reports capability; claims nothing.
//
// Deliberately staged, so a failure says WHERE it failed rather than just "no". Each
// stage is recorded independently: WebGPU may be absent entirely, present but unable to
// yield an adapter, or able to yield one that ORT still refuses. Those are different
// answers for someone deciding whether the GPU path is worth pursuing, and a single
// boolean would collapse them.

import * as ort from "./ort/ort.webgpu.min.mjs";

const qs = new URLSearchParams(window.location.search);
const bucket = Number(qs.get("bucket") || "128");

const out = {
  navigator_gpu_present: false,
  adapter_obtained: false,
  adapter_info: null,
  ort_webgpu_session_created: false,
  ort_webgpu_error: null,
  encoder_ran: false,
  encoder_outputs_finite: null,
};

async function main() {
  // Stage 1: does the browser expose WebGPU at all?
  out.navigator_gpu_present = Boolean(navigator.gpu);
  if (!navigator.gpu) return out;

  // Stage 2: can an adapter actually be obtained? Presence of navigator.gpu does not
  // imply a usable adapter — headless and software-rendering setups routinely have one
  // without the other.
  const adapter = await navigator.gpu.requestAdapter();
  out.adapter_obtained = Boolean(adapter);
  if (adapter) {
    try {
      const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
      out.adapter_info = info
        ? { vendor: info.vendor ?? null, architecture: info.architecture ?? null,
            device: info.device ?? null, description: info.description ?? null }
        : null;
    } catch (e) {
      out.adapter_info = { error: String(e) };
    }
  }
  if (!adapter) return out;

  // Stage 3: does ORT accept the webgpu provider for OUR graph? Use the encoder: it is
  // the graph carrying the k-NN sort, which lowers to the int64 TopK the WebGPU EP is
  // documented not to support, so it is the most informative single graph to try.
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.logLevel = "error";
  try {
    const buf = await (await fetch(`./models/p07_encoder_L${bucket}.onnx`)).arrayBuffer();
    const session = await ort.InferenceSession.create(new Uint8Array(buf), {
      executionProviders: ["webgpu"],
    });
    out.ort_webgpu_session_created = true;

    // Stage 4: run it once on trivial inputs. Finiteness only — no timing, no comparison
    // against any reference. Per ODQ-B3 this probe produces no numbers and no claims.
    const L = bucket;
    const coords = new ort.Tensor("float32", new Float32Array(L * 4 * 3), [L, 4, 3]);
    const mask = new ort.Tensor("float32", new Float32Array(L).fill(1), [L]);
    const residue = new ort.Tensor("int32", Int32Array.from({ length: L }, (_, i) => i), [L]);
    const chain = new ort.Tensor("int32", new Int32Array(L), [L]);
    const names = session.inputNames;
    const feeds = {};
    [coords, mask, residue, chain].forEach((t, i) => {
      feeds[names[i]] = t;
    });
    const res = await session.run(feeds);
    out.encoder_ran = true;
    out.encoder_outputs_finite = session.outputNames.every((n) => {
      const d = res[n].data;
      if (!(d instanceof Float32Array)) return true;
      for (let i = 0; i < d.length; i += 1) if (!Number.isFinite(d[i])) return false;
      return true;
    });
    await session.release();
  } catch (e) {
    out.ort_webgpu_error = String((e && e.message) || e);
  }
  return out;
}

main()
  .then((res) => {
    window.__PROBE_RESULT__ = res;
    window.__PROBE_DONE__ = true;
  })
  .catch((err) => {
    window.__PROBE_RESULT__ = { ...out, ort_webgpu_error: String((err && err.stack) || err) };
    window.__PROBE_DONE__ = true;
  });
