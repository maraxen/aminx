// PottsMPNN browser sampler: drives the four exported graphs (encode, energy, decode, refine)
// for one structure, mirroring aminx sample_host._decode_chunk for ONE sample at a time.
//
// task_id 261007_potts-onnx-export (#5813). Graph contracts: scripts/browser_validation/
// potts_export_gate.py (INPUT_NAMES) and the MANIFEST it writes. This module does no model
// arithmetic. It orders graph calls, maps model -> etab indices for the energy graph, and picks
// the refine visit order exactly as upstream_refine_order does. All randomness (decode randn and
// uniforms, refine uniforms) is an INPUT, so a gate can feed identical noise to JAX and to this
// loop and demand token-exact agreement (the P07 split-export convention).
//
// The caller supplies the onnxruntime-web namespace (this package has no dependency on it) and
// model bytes. Inputs come from potts_inputs.mjs (buildPottsInputs) or any producer of the same
// named, typed arrays.

// Model alphabet index -> etab index: 0..19 identical, model X (20) -> etab X (21).
// aminx/families/potts_mpnn/etab.py model_to_etab.
function modelToEtab(token) {
  return token === 20 ? 21 : token;
}

// pad_etab_energy: zero-pad the 20x20 pair table to 22x22 ("-" and "X" slots). Layout only,
// identical to F.pad(etab, (0, 2, 0, 2)) -- no arithmetic on the values.
function padEtab(forward) {
  const [L, K, A, A2] = forward.dims;
  if (A !== 20 || A2 !== 20) throw new Error(`forward etab must be [L,K,20,20], got ${forward.dims}`);
  const P = 22;
  const out = new Float32Array(L * K * P * P);
  const src = forward.data;
  for (let lk = 0; lk < L * K; lk += 1) {
    for (let a = 0; a < A; a += 1) {
      out.set(src.subarray(lk * A * A + a * A, lk * A * A + (a + 1) * A), lk * P * P + a * P);
    }
  }
  return { data: out, dims: [L, K, P, P] };
}

const GRAPH_KEYS = Object.freeze(["encode", "energy", "decode", "refine"]);

function bucketEntry(manifest, bucket) {
  const entry = (manifest.buckets || []).find((b) => b.bucket === bucket);
  if (!entry) throw new Error(`manifest has no bucket ${bucket}`);
  for (const key of GRAPH_KEYS) {
    if (!entry.graphs || !entry.graphs[key]) throw new Error(`manifest bucket ${bucket} lacks graph ${key}`);
  }
  return entry;
}

function tensor(ort, dtype, dims, data) {
  return new ort.Tensor(dtype, data, dims);
}

async function run(ort, session, graph, named) {
  // Positional binding: jax2onnx names graph inputs in_0..in_N, so the manifest's logical
  // input_names give the order and session.inputNames gives the ONNX names.
  const names = graph.input_names;
  if (session.inputNames.length !== names.length) {
    throw new Error(`${graph.file}: session has ${session.inputNames.length} inputs, manifest lists ${names.length}`);
  }
  const feeds = {};
  names.forEach((logical, i) => {
    const t = named[logical];
    if (t === undefined) throw new Error(`${graph.file}: no value for input ${logical}`);
    const dtype = graph.input_dtypes[i] === "bool" ? "bool" : graph.input_dtypes[i];
    feeds[session.inputNames[i]] = tensor(ort, dtype, graph.input_shapes[i], t.data);
  });
  const out = await session.run(feeds);
  return session.outputNames.map((n) => out[n]);
}

/**
 * Create a Potts sampler for one bucket.
 *
 * @param {object} ort onnxruntime-web namespace
 * @param {object} manifest parsed MANIFEST.json (family "pottsmpnn")
 * @param {number} bucket 128 or 256
 * @param {{encode: Uint8Array, energy: Uint8Array, decode: Uint8Array, refine: Uint8Array}} models
 */
export async function createPottsSampler(ort, manifest, bucket, models) {
  if (manifest.family !== "pottsmpnn") throw new Error(`manifest family ${manifest.family} is not pottsmpnn`);
  const entry = bucketEntry(manifest, bucket);
  const sessions = {};
  for (const key of GRAPH_KEYS) {
    sessions[key] = await ort.InferenceSession.create(models[key]);
  }
  const energyRows = entry.graphs.energy.input_shapes[3][0];

  /**
   * One sample. `inputs` maps logical names to {data: TypedArray} (structure arrays from
   * buildPottsInputs). `noise` = {randn: Float32Array[B], uniforms: Float32Array[B],
   * refineUniforms: Float32Array[8*B]}. `refine` toggles the default one-sweep refine
   * (optimization_mode "potts" is aminx's default, so a faithful sampler refines).
   * `numSamples` reproduces upstream_refine_order's keying quirk (>1 -> N-to-C sweep).
   * `temperature` (decode) and `optimizationTemperature` (refine) are graph inputs; 0 is
   * floored to 1e-6 inside the graphs, as upstream does.
   */
  async function sample(
    inputs,
    noise,
    { refine = true, numSamples = 1, temperature = 0.1, optimizationTemperature = 0.0 } = {},
  ) {
    const B = bucket;
    const [hV, hE, eIdx, forward, table] = await run(ort, sessions.encode, entry.graphs.encode, inputs);
    const enc = { h_v: hV, h_e: hE, e_idx: eIdx, forward, table };

    const [sequence, rankFlat, decodingOrder] = await run(ort, sessions.decode, entry.graphs.decode, {
      ...inputs,
      h_v: enc.h_v,
      h_e: enc.h_e,
      e_idx: enc.e_idx,
      randn: { data: noise.randn },
      uniforms: { data: noise.uniforms },
      temperature: { data: Float32Array.of(temperature) },
    });

    // sample_energy = potts_energy(table, e_idx, pad_valid, model_to_etab(sequence)); the energy
    // graph has a fixed row count, so the sequence goes in row 0 and the rest repeat it.
    const seqs = new Int32Array(energyRows * B);
    for (let r = 0; r < energyRows; r += 1) {
      for (let i = 0; i < B; i += 1) seqs[r * B + i] = modelToEtab(sequence.data[i]);
    }
    const [energies] = await run(ort, sessions.energy, entry.graphs.energy, {
      table: enc.table,
      e_idx: enc.e_idx,
      pad_valid: inputs.pad_valid,
      sequences: { data: seqs },
    });
    const result = {
      sequence: Int32Array.from(sequence.data),
      rank_flat: Int32Array.from(rankFlat.data),
      decoding_order: Int32Array.from(decodingOrder.data),
      sample_energy: energies.data[0],
    };
    if (!refine) return result;

    // upstream_refine_order with stored_orders_present=true and no chain suffix:
    // one sample -> the AR order; more than one -> N-to-C (the _i / str(i) keying quirk).
    const order = numSamples === 1 ? Int32Array.from(decodingOrder.data) : Int32Array.from({ length: B }, (_, i) => i);
    const [refined] = await run(ort, sessions.refine, entry.graphs.refine, {
      ...inputs,
      sequence: { data: result.sequence },
      etab: padEtab(enc.forward),
      e_idx: enc.e_idx,
      order: { data: order },
      uniforms: { data: noise.refineUniforms },
      h_v: enc.h_v,
      h_e: enc.h_e,
      temperature: { data: Float32Array.of(optimizationTemperature) },
    });
    result.refined_sequence = Int32Array.from(refined.data);
    return result;
  }

  async function release() {
    for (const key of GRAPH_KEYS) await sessions[key].release?.();
  }

  return { sample, release, bucket, energyRows };
}

export { modelToEtab, padEtab };
