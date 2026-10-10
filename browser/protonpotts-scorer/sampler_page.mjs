// In-page driver: ProtonPotts sequence sampling (ProtonPottsSampler), end to end, in a real browser.
//
// Loaded by sampler_index.html (copied to the site root as index.html). Protocol of browser/layer_c/run_p07.mjs: set
// `window.__PARITY_DONE__` when finished, the payload on `window.__PARITY_RESULT__`, any failure on `window.__PARITY_ERROR__`.
// `cells.json` (written by the tracked Python gate): cells[{id, bucket, pdb, designed, temperature, bias, uniforms, noise, seed}].
// For the benchmark gate cells.json also supplies reps, warmup, planted_ms (defaults 1, 0, 0): each cell runs `warmup + reps` times, the first
// `warmup` runs are discarded and the output of the first measured run is returned, and a planted delay is timed through the same wrapper.
// This grades NOTHING; every comparison lives in the tracked Python gate.

import * as ort from "./ort/ort.wasm.min.mjs";

import { ProtonPottsSampler } from "./decoder_sampler.mjs";

const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function timed(fn) {
  const t0 = performance.now();
  const value = await fn();
  return { value, ms: performance.now() - t0 };
}

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
  const reps = Number(cells.reps ?? 1);
  const warmup = Number(cells.warmup ?? 0);
  const plantedMs = Number(cells.planted_ms ?? 0);
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

  let ctrlMeasuredMs = -1;
  if (plantedMs > 0) ctrlMeasuredMs = (await timed(() => sleep(plantedMs))).ms;

  const results = [];
  for (const cell of cells.cells) {
    const rec = { id: cell.id, ok: false, error: null, wall_ms: null, wall_ms_all: [] };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      const options = { seed: cell.seed ?? 0, temperature: Array.isArray(cell.temperature) ? Float32Array.from(cell.temperature) : cell.temperature };
      if (cell.designed) options.designed = cell.designed.map((x) => x === 1);
      if (cell.bias) options.bias = Float32Array.from(cell.bias);
      if (cell.uniforms) {
        options.uniforms = Float32Array.from(cell.uniforms);
        options.noise = Float32Array.from(cell.noise);
      }
      let out = null;
      for (let k = 0; k < warmup + reps; k += 1) {
        const run = await timed(() => samplers[cell.bucket].sample(pdbText, null, options));
        if (k < warmup) continue;
        rec.wall_ms_all.push(run.ms);
        if (k === warmup) {
          out = run.value;
          rec.wall_ms = run.ms;
        }
      }
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
    reps,
    warmup,
    planted_ms: plantedMs,
    ctrl_measured_ms: ctrlMeasuredMs,
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
