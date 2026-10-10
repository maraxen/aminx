// ProtonPotts 30-token sequence sampling in the browser (ADR decision 12, spec §53-54): PDB text -> encode graph -> decode graph.
//
// The graphs (protonpotts_decode_export_gate.py; manifest family "protonpottsmpnn_decoder") do the arithmetic. This module owns what
// the graphs deliberately take as inputs: the decoding-order noise and the draw uniforms (injected for replay, otherwise a seeded
// generator), the per-residue temperature / bias / designed mask, the padding to the bucket, and the trimming of the outputs.
//
// STREAM ALIGNMENT. The decoder orders padding LAST (decoding_order_from_noise takes pad_valid), so the real residues keep decode steps
// 0..L-1 whatever the padding, and the caller's length-L uniform stream (u[k] = the k-th decode step, as in the unpadded decode) goes at
// U[0..L-1]; noise is indexed by position. (An earlier version shifted the stream by the pad count on the belief that padding decodes
// first; the X8b exploratory smoke showed every case wrong and a padded-vs-unpadded JAX spike showed the unshifted stream is exact.)

import { ProtonPottsInputError, N_TOKENS, VOCABULARY, buildProtonPottsInputs } from "./protonpotts_inputs.mjs";
import { run } from "./protonpotts_scorer.mjs";

export const DEFAULT_TEMPERATURE = Math.fround(0.1); // upstream's default, float32, as the decoder wave graded it

/** mulberry32: small, seedable, good enough for draw uniforms; the replay contract is the injected-uniform path. */
export function seededStreams(seed, length) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const uniforms = new Float32Array(length);
  const noise = new Float32Array(length);
  for (let i = 0; i < length; i += 1) uniforms[i] = Math.fround(next());
  for (let i = 0; i < length; i += 1) {
    const a = Math.max(next(), 1e-12);
    noise[i] = Math.fround(Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * next()));
  }
  return { uniforms, noise };
}

export class ProtonPottsSampler {
  constructor(ort, manifest, bucket, sessions) {
    this.ort = ort;
    this.manifest = manifest;
    this.bucket = bucket;
    this.sessions = sessions;
    this.entry = manifest.buckets.find((b) => b.bucket === bucket);
    if (!this.entry) throw new Error(`decoder manifest has no bucket ${bucket}`);
  }

  static async create(ort, manifest, bucket, models) {
    if (manifest.family !== "protonpottsmpnn_decoder") throw new Error(`manifest family ${manifest.family} is not protonpottsmpnn_decoder`);
    if (manifest.n_tokens !== N_TOKENS) throw new Error(`manifest n_tokens ${manifest.n_tokens} != ${N_TOKENS}`);
    const sessions = { encode: await ort.InferenceSession.create(models.encode), decode: await ort.InferenceSession.create(models.decode) };
    return new ProtonPottsSampler(ort, manifest, bucket, sessions);
  }

  /**
   * One sample.
   * @param {string} pdbText
   * @param {(string|null)[]|null} labels  as buildProtonPottsInputs
   * @param {object} options
   *   designed:    boolean[] over kept residues (default all true)
   *   temperature: number | Float32Array(L)        (default DEFAULT_TEMPERATURE)
   *   bias:        Float32Array(L * 30) in aminx token order, row-major (default zeros)
   *   uniforms, noise: Float32Array(L) — injected streams (both or neither); else seeded from `seed`
   * @returns {{sequence: Int32Array, symbols: string, decodingOrder: Int32Array, logProbs: Float32Array, lTotal: number, kept: Array}}
   */
  async sample(pdbText, labels = null, options = {}) {
    const B = this.bucket;
    const built = buildProtonPottsInputs(pdbText, B, labels);
    const L = built.lTotal;
    const V = N_TOKENS;
    const { designed = null, temperature = DEFAULT_TEMPERATURE, bias = null, seed = 0 } = options;
    let { uniforms = null, noise = null } = options;
    if ((uniforms === null) !== (noise === null)) throw new ProtonPottsInputError("uniforms and noise are injected together or not at all");
    if (uniforms === null) ({ uniforms, noise } = seededStreams(seed, L));
    if (uniforms.length !== L || noise.length !== L) throw new ProtonPottsInputError(`streams must have length ${L}`);
    if (designed !== null && designed.length !== L) throw new ProtonPottsInputError(`designed has ${designed.length} entries for ${L} kept residues`);
    if (bias !== null && bias.length !== L * V) throw new ProtonPottsInputError(`bias must have ${L * V} entries`);

    const [hV, hE, eIdx] = await run(this.ort, this.sessions.encode, this.entry.graphs.encode, built.inputs);

    const designedPad = new Uint8Array(B);
    for (let i = 0; i < L; i += 1) designedPad[i] = designed === null || designed[i] ? 1 : 0;
    const temperaturePad = new Float32Array(B).fill(DEFAULT_TEMPERATURE);
    for (let i = 0; i < L; i += 1) temperaturePad[i] = typeof temperature === "number" ? Math.fround(temperature) : temperature[i];
    const biasPad = new Float32Array(B * V);
    if (bias !== null) biasPad.set(bias);
    const sTrue = new Int32Array(B);
    sTrue.set(built.tokens);
    const uniformsPad = new Float32Array(B);
    uniformsPad.set(uniforms);
    const noisePad = new Float32Array(B);
    noisePad.set(noise);

    const [seq, order, logProbs] = await run(this.ort, this.sessions.decode, this.entry.graphs.decode, {
      h_v: hV, h_e: hE, e_idx: eIdx, present: built.inputs.present, pad_valid: built.inputs.pad_valid,
      s_true: { data: sTrue }, designed: { data: designedPad }, temperature: { data: temperaturePad },
      bias: { data: biasPad }, uniforms: { data: uniformsPad }, noise: { data: noisePad },
    });
    const sequence = Int32Array.from(seq.data.subarray(0, L));
    return {
      sequence,
      symbols: Array.from(sequence, (t) => VOCABULARY[t]).join(""),
      decodingOrder: Int32Array.from(Array.from(order.data).filter((p) => p < L)),
      logProbs: Float32Array.from(logProbs.data.subarray(0, L * V)),
      lTotal: L,
      kept: built.kept,
    };
  }
}
