// In-page driver: the PottsMPNN browser sampler, end to end, in a real browser.
//
// Loaded by potts_index.html (copied to the site root as index.html). Protocol, deliberately the
// one split_page.mjs and browser/layer_c/run_p07.mjs already use: set `window.__PARITY_DONE__` when
// finished, put the payload on `window.__PARITY_RESULT__`, any failure on `window.__PARITY_ERROR__`.
// Sample outputs are POSTed as raw little-endian bytes to `/results/<name>` (parity.mjs convention;
// browser/layer_c/serve.mjs writes them into the run's out-dir).
//
// Per cell the timed path is PDB text -> buildPottsInputs -> applyPottsControls -> sampler.sample
// (encode, decode, energy, refine). Each cell runs `warmup + reps` times; the first `warmup` runs are
// discarded, the outputs of the first measured run are posted. `prep_ms` (build + controls) and
// `sample_ms` (the four graphs) are split out as the cheap per-stage breakdown.
//
// `cells.json` (written by the tracked Python gate) supplies: bucket, cells[{id, pdb, bucket,
// controls, options, noise: {name: {file, dtype, dims}}}], reps, warmup, planted_ms. A planted delay
// (planted_ms > 0) is timed through the SAME wrapper as the real cells, so a timer that does not see
// it is detected by the gate.
//
// This grades NOTHING. Every comparison and threshold lives in the tracked Python gates.

import * as ort from "./ort/ort.wasm.min.mjs";

import { applyPottsControls } from "./potts_controls.mjs";
import { buildPottsInputs } from "./potts_inputs.mjs";
import { createPottsSampler } from "./potts_sampler.mjs";

const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// One timing wrapper for everything that is timed (planted control, prep, sample, full path).
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

async function postBytes(name, bytes) {
  const res = await fetch(`/results/${name}`, { method: "POST", body: bytes });
  if (!res.ok) throw new Error(`POST /results/${name} -> ${res.status}`);
}

async function main() {
  // ABSOLUTE, not "./ort/": ORT resolves wasmPaths relative to its own module URL, so a relative
  // path 404s every sibling .wasm (see split_page.mjs).
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = numThreads;
  ort.env.logLevel = "error";

  const cells = await (await fetch("./cells.json")).json();
  const reps = Number(cells.reps ?? 1);
  const warmup = Number(cells.warmup ?? 0);
  const plantedMs = Number(cells.planted_ms ?? 0);

  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/MANIFEST.json")));
  const buckets = [...new Set(cells.cells.map((c) => c.bucket))];
  const samplers = {};
  const sessionCreateMs = {};
  for (const bucket of buckets) {
    const entry = manifest.buckets.find((b) => b.bucket === bucket);
    if (!entry) throw new Error(`manifest has no bucket ${bucket}`);
    const models = {};
    for (const key of ["encode", "energy", "decode", "refine"]) {
      models[key] = new Uint8Array(await fetchBytes(`./models/${entry.graphs[key].file}`));
    }
    const created = await timed(() => createPottsSampler(ort, manifest, bucket, models));
    samplers[bucket] = created.value;
    sessionCreateMs[String(bucket)] = created.ms;
  }
  // What ORT reports after init, not what was requested: a silent single-thread fallback must
  // appear as data.
  const numThreadsAfterInit = ort.env.wasm.numThreads;

  // Planted-delay control: the same timing wrapper, the same await machinery as the real cells.
  let ctrlMeasuredMs = -1;
  if (plantedMs > 0) {
    ctrlMeasuredMs = (await timed(() => sleep(plantedMs))).ms;
  }

  const results = [];
  for (const cell of cells.cells) {
    const rec = {
      id: cell.id, ok: false, error: null, l_total: null, sample_energy: null,
      wall_ms_all: [], prep_ms_all: [], sample_ms_all: [], outputs: {},
    };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      const noise = {};
      for (const [name, spec] of Object.entries(cell.noise)) {
        noise[name] = new Float32Array(await fetchBytes(`./${spec.file}`));
      }
      const sampler = samplers[cell.bucket];
      for (let k = 0; k < warmup + reps; k += 1) {
        let out;
        let lTotal;
        let prepMs;
        let sampleMs;
        const full = await timed(async () => {
          const prep = await timed(() => {
            const built = buildPottsInputs(pdbText, cell.bucket);
            const controlled = applyPottsControls(built, cell.controls || {});
            const inputs = {};
            for (const [name, t] of Object.entries(controlled.inputs ?? controlled)) {
              inputs[name] = { data: t.data };
            }
            return { built, inputs };
          });
          lTotal = prep.value.built.l_total;
          const smp = await timed(() => sampler.sample(prep.value.inputs, noise, cell.options || {}));
          prepMs = prep.ms;
          sampleMs = smp.ms;
          out = smp.value;
          return out;
        });
        if (k < warmup) continue;
        // Only measured (post-warm-up) runs enter any timing array, so the three stay aligned.
        rec.wall_ms_all.push(full.ms);
        rec.prep_ms_all.push(prepMs);
        rec.sample_ms_all.push(sampleMs);
        if (k === warmup) {
          // Outputs come from the first measured run only; POSTing happens after its timing.
          rec.l_total = lTotal;
          rec.sample_energy = out.sample_energy;
          for (const name of ["sequence", "decoding_order", "refined_sequence"]) {
            const arr = out[name];
            if (arr === undefined) continue;
            const fname = `${cell.id}__${name}.bin`;
            await postBytes(fname, new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
            rec.outputs[name] = { file: fname, dtype: "int32", dims: [arr.length] };
          }
        }
      }
      rec.ok = true;
    } catch (err) {
      rec.error = String((err && err.stack) || err);
    }
    results.push(rec);
  }

  for (const b of buckets) {
    const s = samplers[b];
    if (s && s.release) await s.release();
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
