// Drive the SHIPPING ph_design.mjs under Node with onnxruntime-web (wasm, 1 thread).
//
// No grading here: read cases, run PhDesigner.design from PDB text, write the designs. The tracked Python gate
// (scripts/browser_validation/protonpotts_ph_loop_gate.py) does every comparison.
//
// Usage: node run_ph_design_node.mjs --ort-dir <dir with node_modules/onnxruntime-web> --scoring-models <dir>
//          --ph-models <dir> --cases <cases.json> --out <result.json>
// cases.json: {"cases": [{"id", "bucket", "pdb_file", "labels": [..]|null, "config": {...PhDesigner config...},
//                          "uniforms": [[...], ...]|null}]}

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { PhDesigner } from "./ph_design.mjs";

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) throw new Error(`missing --${name}`);
  return process.argv[i + 1];
}

const ortPkgDir = join(arg("ort-dir"), "node_modules", "onnxruntime-web");
const ort = await import(pathToFileURL(join(ortPkgDir, "dist", "ort.wasm.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(join(ortPkgDir, "dist") + "/").href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const scoringDir = arg("scoring-models");
const phDir = arg("ph-models");
const scoring = JSON.parse(readFileSync(join(scoringDir, "MANIFEST.json"), "utf8"));
const ph = JSON.parse(readFileSync(join(phDir, "MANIFEST.json"), "utf8"));
const { cases } = JSON.parse(readFileSync(arg("cases"), "utf8"));

const bytes = (dir, file) => new Uint8Array(readFileSync(join(dir, file)));
const designers = new Map();
const results = [];
for (const c of cases) {
  const record = { id: c.id, ok: false, error: null, designs: null };
  try {
    if (!designers.has(c.bucket)) {
      const scoringEntry = scoring.buckets.find((b) => b.bucket === c.bucket);
      const phEntry = ph.buckets.find((b) => b.bucket === c.bucket);
      const graphs = {};
      for (const [name, meta] of Object.entries(phEntry.graphs)) graphs[name] = bytes(phDir, meta.file);
      designers.set(
        c.bucket,
        await PhDesigner.create(ort, ph, scoringEntry, c.bucket, {
          table: bytes(scoringDir, scoringEntry.graphs.table.file),
          energy: bytes(scoringDir, scoringEntry.graphs.energy.file),
          graphs,
        }),
      );
    }
    const uniforms = c.uniforms ? c.uniforms.map((u) => Float32Array.from(u)) : null;
    const designs = await designers.get(c.bucket).design(readFileSync(c.pdb_file, "utf8"), c.labels, c.config, uniforms);
    record.designs = designs.map((d) => ({
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
    record.ok = true;
  } catch (err) {
    record.error = String(err && err.stack ? err.stack : err);
  }
  results.push(record);
}
writeFileSync(arg("out"), JSON.stringify({ num_threads: ort.env.wasm.numThreads, results }, null, 2));
