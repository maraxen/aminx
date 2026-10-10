// ProtonPottsMPNN browser scorer: PDB text (+ protonation labels) -> energies of the native sequence and any
// number of variant rows, driving the two exported graphs (table, energy). Mirrors ProtonPottsDriver's score:energy
// for ONE structure: row 0 is the native sequence (with its labels), then the variants in the order given.
//
// task_id 261009_protonpotts-onnx (#5816). Graph contracts: scripts/browser_validation/protonpotts_export_gate.py
// (INPUT_NAMES) and the MANIFEST it writes. This module does no model arithmetic: it builds inputs, orders the
// graph calls, and chunks the variants through the energy graph's fixed row count (padding the last chunk).
//
// The caller supplies the onnxruntime-web namespace and the model bytes.

import { N_TOKENS, ProtonPottsInputError, VOCABULARY, buildProtonPottsInputs } from "./protonpotts_inputs.mjs";

const TOKEN_INDEX = new Map(VOCABULARY.map((symbol, index) => [symbol, index]));

function bucketEntry(manifest, bucket) {
  const entry = (manifest.buckets || []).find((b) => b.bucket === bucket);
  if (!entry) throw new Error(`manifest has no bucket ${bucket}`);
  for (const key of ["table", "energy"]) {
    if (!entry.graphs || !entry.graphs[key]) throw new Error(`manifest bucket ${bucket} lacks graph ${key}`);
  }
  return entry;
}

export async function run(ort, session, graph, named) {
  // Positional binding: jax2onnx names graph inputs in_0..in_N; the manifest gives the logical order.
  const names = graph.input_names;
  if (session.inputNames.length !== names.length) {
    throw new Error(`${graph.file}: session has ${session.inputNames.length} inputs, manifest lists ${names.length}`);
  }
  const feeds = {};
  names.forEach((logical, i) => {
    const t = named[logical];
    if (t === undefined) throw new Error(`${graph.file}: no value for input ${logical}`);
    feeds[session.inputNames[i]] = new ort.Tensor(graph.input_dtypes[i], t.data, graph.input_shapes[i]);
  });
  const out = await session.run(feeds);
  return session.outputNames.map((n) => out[n]);
}

/** A variant row given as token names or indices -> Int32Array of token indices, length-checked. */
export function rowToIndices(row, length) {
  if (row.length !== length) throw new ProtonPottsInputError(`variant has ${row.length} tokens for ${length} kept residues`);
  return Int32Array.from(row, (token) => {
    const index = typeof token === "number" ? token : TOKEN_INDEX.get(token);
    if (index === undefined || !Number.isInteger(index) || index < 0 || index >= N_TOKENS) {
      throw new ProtonPottsInputError(`unknown token ${JSON.stringify(token)}`);
    }
    return index;
  });
}

/**
 * @param {object} ort onnxruntime-web namespace
 * @param {object} manifest parsed MANIFEST.json (family "protonpottsmpnn")
 * @param {number} bucket 256 or 1024
 * @param {{table: Uint8Array, energy: Uint8Array}} models
 */
export async function createProtonPottsScorer(ort, manifest, bucket, models) {
  if (manifest.family !== "protonpottsmpnn") throw new Error(`manifest family ${manifest.family} is not protonpottsmpnn`);
  if (manifest.n_tokens !== N_TOKENS) throw new Error(`manifest n_tokens ${manifest.n_tokens} != ${N_TOKENS}`);
  if (JSON.stringify(manifest.vocabulary) !== JSON.stringify(VOCABULARY)) {
    throw new Error("manifest vocabulary differs from the one this module encodes tokens with");
  }
  const entry = bucketEntry(manifest, bucket);
  const sessions = {
    table: await ort.InferenceSession.create(models.table),
    energy: await ort.InferenceSession.create(models.energy),
  };
  const rowsPerCall = entry.graphs.energy.input_shapes[3][0];

  /**
   * @param {string} pdbText
   * @param {(string|null)[]|null} labels  one entry per loaded residue, '' for none; null for unlabelled
   * @param {Array<Array<string|number>>} variants  token rows (names or indices), one token per KEPT residue
   * @returns {{energies: Float32Array, tokens: number[], kept: Array, lTotal: number}} energies[0] is the native row
   */
  async function score(pdbText, labels, variants = []) {
    const built = buildProtonPottsInputs(pdbText, bucket, labels);
    const [table, eIdx] = await run(ort, sessions.table, entry.graphs.table, built.inputs);
    const rows = [Int32Array.from(built.tokens), ...variants.map((v) => rowToIndices(v, built.lTotal))];
    const energies = new Float32Array(rows.length);
    for (let start = 0; start < rows.length; start += rowsPerCall) {
      const chunk = rows.slice(start, start + rowsPerCall);
      const seqs = new Int32Array(rowsPerCall * bucket); // unused rows and the pad tail stay token 0
      chunk.forEach((row, r) => seqs.set(row, r * bucket));
      const [out] = await run(ort, sessions.energy, entry.graphs.energy, {
        table,
        e_idx: eIdx,
        pad_valid: built.inputs.pad_valid,
        sequences: { data: seqs },
      });
      for (let r = 0; r < chunk.length; r += 1) energies[start + r] = out.data[r];
    }
    return { energies, tokens: built.tokens, kept: built.kept, lTotal: built.lTotal };
  }

  async function release() {
    for (const s of Object.values(sessions)) await s.release?.();
  }

  return { score, release, bucket, rowsPerCall };
}
