// In-page driver: ProtonPotts sequence sampling (ProtonPottsSampler), end to end, in a real browser.
//
// Loaded by sampler_index.html (copied to the site root as index.html). Protocol of browser/layer_c/run_p07.mjs: set
// `window.__PARITY_DONE__` when finished, the payload on `window.__PARITY_RESULT__`, any failure on `window.__PARITY_ERROR__`.
// `cells.json` (written by the tracked Python gate): cells[{id, bucket, pdb, designed, temperature, bias, uniforms, noise, seed}].
// This grades NOTHING; every comparison lives in the tracked Python gate.

import * as ort from "./ort/ort.wasm.min.mjs";

import { ProtonPottsSampler } from "./decoder_sampler.mjs";

const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.arrayBuffer();
}

async function main() {
  // ABSOLUTE, not "./ort/": ORT resolves wasmPaths relative to its own module URL (see split_page.mjs).
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = numThreads;
  ort.env.logLevel = "error";

  const cells = await (await fetch("./cells.json")).json();
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/MANIFEST.json")));
  const samplers = {};
  const sessionCreateMs = {};
  for (const bucket of [...new Set(cells.cells.map((c) => c.bucket))]) {
    const entry = manifest.buckets.find((b) => b.bucket === bucket);
    const t0 = performance.now();
    samplers[bucket] = await ProtonPottsSampler.create(ort, manifest, bucket, {
      encode: new Uint8Array(await fetchBytes(`./models/${entry.graphs.encode.file}`)),
      decode: new Uint8Array(await fetchBytes(`./models/${entry.graphs.decode.file}`)),
    });
    sessionCreateMs[String(bucket)] = performance.now() - t0;
  }
  const numThreadsAfterInit = ort.env.wasm.numThreads;

  const results = [];
  for (const cell of cells.cells) {
    const rec = { id: cell.id, ok: false, error: null, wall_ms: null };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      const options = { seed: cell.seed ?? 0, temperature: Array.isArray(cell.temperature) ? Float32Array.from(cell.temperature) : cell.temperature };
      if (cell.designed) options.designed = cell.designed.map((x) => x === 1);
      if (cell.bias) options.bias = Float32Array.from(cell.bias);
      if (cell.uniforms) {
        options.uniforms = Float32Array.from(cell.uniforms);
        options.noise = Float32Array.from(cell.noise);
      }
      const t0 = performance.now();
      const out = await samplers[cell.bucket].sample(pdbText, null, options);
      rec.wall_ms = performance.now() - t0;
      rec.sequence = Array.from(out.sequence);
      rec.decoding_order = Array.from(out.decodingOrder);
      rec.log_probs = Array.from(out.logProbs);
      rec.l_total = out.lTotal;
      rec.ok = true;
    } catch (err) {
      rec.error = String((err && err.stack) || err);
    }
    results.push(rec);
  }

  return {
    n_cells: results.length,
    num_threads_requested: numThreads,
    num_threads_after_init: numThreadsAfterInit,
    num_threads_effective: ort.env.wasm.numThreads,
    cross_origin_isolated: Boolean(window.crossOriginIsolated),
    user_agent: navigator.userAgent,
    session_create_ms: sessionCreateMs,
    cells: results,
  };
}

main()
  .then((res) => {
    window.__PARITY_RESULT__ = res;
    window.__PARITY_DONE__ = true;
  })
  .catch((err) => {
    window.__PARITY_ERROR__ = String((err && err.stack) || err);
    window.__PARITY_DONE__ = true;
  });
