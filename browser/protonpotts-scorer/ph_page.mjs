// In-page driver: the ProtonPotts pH design loop (PhDesigner), end to end, in a real browser.
//
// Loaded by ph_index.html (copied to the site root as index.html). Protocol, the one potts_page.mjs / protonpotts_page.mjs and
// browser/layer_c/run_p07.mjs use: set `window.__PARITY_DONE__` when finished, put the payload on `window.__PARITY_RESULT__`, any failure
// on `window.__PARITY_ERROR__`. The designs are small, so they travel in the payload.
//
// `cells.json` (written by the tracked Python gate) supplies: cells[{id, bucket, pdb, labels, config, uniforms}]. This grades NOTHING;
// every comparison and threshold lives in the tracked Python gate.

import * as ort from "./ort/ort.wasm.min.mjs";

import { PhDesigner } from "./ph_design.mjs";

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
  const scoring = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/scoring/MANIFEST.json")));
  const ph = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/ph/MANIFEST.json")));
  const designers = {};
  const sessionCreateMs = {};
  for (const bucket of [...new Set(cells.cells.map((c) => c.bucket))]) {
    const scoringEntry = scoring.buckets.find((b) => b.bucket === bucket);
    const phEntry = ph.buckets.find((b) => b.bucket === bucket);
    const t0 = performance.now();
    const graphs = {};
    for (const [name, meta] of Object.entries(phEntry.graphs)) graphs[name] = new Uint8Array(await fetchBytes(`./models/ph/${meta.file}`));
    designers[bucket] = await PhDesigner.create(ort, ph, scoringEntry, bucket, {
      table: new Uint8Array(await fetchBytes(`./models/scoring/${scoringEntry.graphs.table.file}`)),
      energy: new Uint8Array(await fetchBytes(`./models/scoring/${scoringEntry.graphs.energy.file}`)),
      graphs,
    });
    sessionCreateMs[String(bucket)] = performance.now() - t0;
  }
  const numThreadsAfterInit = ort.env.wasm.numThreads;

  const results = [];
  for (const cell of cells.cells) {
    const rec = { id: cell.id, ok: false, error: null, designs: null, wall_ms: null };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      const uniforms = cell.uniforms ? cell.uniforms.map((u) => Float32Array.from(u)) : null;
      const t0 = performance.now();
      const designs = await designers[cell.bucket].design(pdbText, cell.labels, cell.config, uniforms);
      rec.wall_ms = performance.now() - t0;
      rec.designs = designs.map((d) => ({
        sequence: Array.from(d.sequence),
        sample: d.sample,
        method: d.method,
        label: d.label,
        designable: d.designable,
        pins: d.pins.map((p) => ({ position: p.position, protIdx: p.protIdx, depIdxs: p.depIdxs, resId: p.resId, type: p.protonationType })),
        final_potts_energy: d.finalPottsEnergy,
        selective_energy: d.selectiveEnergy,
        selective_energies: d.selectiveEnergies,
        n_draws: d.nDraws,
      }));
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
