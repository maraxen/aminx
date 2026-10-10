// In-page driver: the ProtonPotts browser scorer, end to end, in a real browser.
//
// Loaded by protonpotts_index.html (copied to the site root as index.html). Protocol, deliberately the one
// potts_page.mjs / split_page.mjs and browser/layer_c/run_p07.mjs already use: set `window.__PARITY_DONE__` when
// finished, put the payload on `window.__PARITY_RESULT__`, any failure on `window.__PARITY_ERROR__`. Energies are
// POSTed as raw little-endian float32 bytes to `/results/<name>` (browser/layer_c/serve.mjs writes them into the
// run's out-dir).
//
// Per cell the timed path is PDB text -> buildProtonPottsInputs -> table graph -> chunked energy graph
// (scorer.score). Each cell runs `warmup + reps` times; the first `warmup` runs are discarded, the outputs of the
// first measured run are posted.
//
// `cells.json` (written by the tracked Python gate) supplies: cells[{id, pdb, bucket, labels, variants}], reps,
// warmup, planted_ms. A planted delay is timed through the SAME wrapper as the real cells, so a timer that does not
// see it is detected by the gate.
//
// This grades NOTHING. Every comparison and threshold lives in the tracked Python gates.

import * as ort from "./ort/ort.wasm.min.mjs";

import { createProtonPottsScorer } from "./protonpotts_scorer.mjs";

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

async function postBytes(name, bytes) {
  const res = await fetch(`/results/${name}`, { method: "POST", body: bytes });
  if (!res.ok) throw new Error(`POST /results/${name} -> ${res.status}`);
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
  const buckets = [...new Set(cells.cells.map((c) => c.bucket))];
  const scorers = {};
  const sessionCreateMs = {};
  for (const bucket of buckets) {
    const entry = manifest.buckets.find((b) => b.bucket === bucket);
    if (!entry) throw new Error(`manifest has no bucket ${bucket}`);
    const models = {};
    for (const key of ["table", "energy"]) {
      models[key] = new Uint8Array(await fetchBytes(`./models/${entry.graphs[key].file}`));
    }
    const created = await timed(() => createProtonPottsScorer(ort, manifest, bucket, models));
    scorers[bucket] = created.value;
    sessionCreateMs[String(bucket)] = created.ms;
  }
  // What ORT reports after init, not what was requested: a silent single-thread fallback must appear as data.
  const numThreadsAfterInit = ort.env.wasm.numThreads;

  let ctrlMeasuredMs = -1;
  if (plantedMs > 0) ctrlMeasuredMs = (await timed(() => sleep(plantedMs))).ms;

  const results = [];
  for (const cell of cells.cells) {
    const rec = {
      id: cell.id, ok: false, error: null, l_total: null, tokens: null, energies: null,
      wall_ms_all: [],
    };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      for (let k = 0; k < warmup + reps; k += 1) {
        const full = await timed(() => scorers[cell.bucket].score(pdbText, cell.labels, cell.variants));
        if (k < warmup) continue;
        rec.wall_ms_all.push(full.ms);
        if (k === warmup) {
          const { energies, tokens, lTotal } = full.value;
          rec.l_total = lTotal;
          rec.tokens = tokens;
          const fname = `${cell.id}__energies.bin`;
          await postBytes(fname, new Uint8Array(energies.buffer, energies.byteOffset, energies.byteLength));
          rec.energies = { file: fname, dtype: "float32", dims: [energies.length] };
        }
      }
      rec.ok = true;
    } catch (err) {
      rec.error = String((err && err.stack) || err);
    }
    results.push(rec);
  }

  for (const b of buckets) {
    const s = scorers[b];
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
