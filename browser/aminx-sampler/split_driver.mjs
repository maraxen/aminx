// Browser-ready ES module driving the four-graph split ProteinMPNN export
// (task_id 260926_browser-export-loop / T13). NO node: imports, NO bundler
// required -- same convention as ./aminx_sampler.mjs.
//
// The monolithic P07 ONNX export kept the whole autoregressive loop in-graph.
// This module drives the loop-free split instead: Graph E (encoder, once per
// structure), Graph W (wave schedule, once per sample), Graph D (decoder
// step, once per wave), Graph F (fuse + sample, once per wave). The loop
// itself -- which graph to call in what order, with which tensors -- lives
// in ./split_loop.mjs as pure, ORT-free logic; this file only adds the
// onnxruntime-web session plumbing and the RunSpec -> tensor conversion
// (reusing ./runspec_core.mjs and the bucket/pad conventions of
// ./aminx_sampler.mjs so the three modules can never diverge by hand-copy).
//
// Model-family knobs (alphabet, token count, omit bias, per-graph input order
// and dtypes) come from the export MANIFEST when it supplies them, so a second
// family (PottsMPNN) reuses this sampler by exporting its own manifest. When
// the manifest omits them (the committed ProteinMPNN MANIFEST.json predates
// them) the ProteinMPNN constants below apply, so behaviour is unchanged.
import { pickBucket, padStructure } from "./aminx_sampler.mjs";
import { buildP07TypedInputs, MPNN_ALPHABET, OMIT_BIAS } from "./runspec_core.mjs";
import { runSplitDecode } from "./split_loop.mjs";

// Positional input orders per the graph contracts (T13 spec). Pinned here the
// same way P07_INPUT_ORDER pins the monolithic model's order in
// aminx_sampler.mjs -- each graph's `session.inputNames[i]` is expected to
// correspond to the i-th name below. Used only when the manifest does not
// declare `input_names` for the graph.
export const ENCODER_INPUT_ORDER = Object.freeze(["coords", "mask", "residue_index", "chain_index"]);
export const WAVE_INPUT_ORDER = Object.freeze(["decoding_order", "tie_group_map"]);
export const DECODER_INPUT_ORDER = Object.freeze([
  "node_features",
  "edge_features",
  "neighbor_indices",
  "mask",
  "ar_mask",
  "sequence_oh",
]);
export const FUSE_INPUT_ORDER = Object.freeze([
  "logits",
  "cond_bias",
  "mask_group",
  "fixed_mask",
  "fixed_tokens",
  "group_id",
  "temperature",
  "gumbel_noise",
]);

const ENCODER_OUTPUT_NAMES = Object.freeze(["node_features", "edge_features", "neighbor_indices"]);
const WAVE_OUTPUT_NAMES = Object.freeze([
  "group_ids",
  "group_positions",
  "group_valid",
  "position_valid",
  "ar_mask",
  "group_first_rank",
  "pos_first_rank",
]);
const DECODER_OUTPUT_NAMES = Object.freeze(["logits"]);
const FUSE_OUTPUT_NAMES = Object.freeze(["final_token", "avg_stored"]);

const ENCODER_INPUT_DTYPES = Object.freeze(["float32", "float32", "int32", "int32"]);
const WAVE_INPUT_DTYPES = Object.freeze(["int32", "int32"]);
const DECODER_INPUT_DTYPES = Object.freeze(["float32", "float32", "int32", "float32", "float32", "float32"]);
const FUSE_INPUT_DTYPES = Object.freeze([
  "float32",
  "float32",
  "bool",
  "float32",
  "int32",
  "int32",
  "float32",
  "float32",
]);

// The ProteinMPNN fallback for each graph, used when the manifest is silent.
const PINNED_GRAPHS = Object.freeze({
  encoder: { order: ENCODER_INPUT_ORDER, dtypes: ENCODER_INPUT_DTYPES },
  wave: { order: WAVE_INPUT_ORDER, dtypes: WAVE_INPUT_DTYPES },
  decoder: { order: DECODER_INPUT_ORDER, dtypes: DECODER_INPUT_DTYPES },
  fuse: { order: FUSE_INPUT_ORDER, dtypes: FUSE_INPUT_DTYPES },
});

async function loadSession(ort, urlOrBuffer) {
  let modelSource = urlOrBuffer;
  if (typeof urlOrBuffer === "string") {
    const res = await fetch(urlOrBuffer);
    if (!res.ok) {
      throw new Error(`fetch ${urlOrBuffer} failed: ${res.status} ${res.statusText}`);
    }
    modelSource = await res.arrayBuffer();
  }
  return ort.InferenceSession.create(modelSource, { executionProviders: ["wasm"] });
}

function checkInputCount(session, name, expected) {
  if (session.inputNames.length !== expected.length) {
    throw new Error(
      `${name} session has ${session.inputNames.length} inputs, expected ${expected.length} (${expected.join(", ")})`,
    );
  }
}

/**
 * Family knobs for the sampler. The manifest wins when it supplies them;
 * otherwise the ProteinMPNN values apply. The token count is always the
 * alphabet length, so a manifest can only state it redundantly, never differ.
 *
 * @param {object|undefined} manifest parsed MANIFEST.json (top level)
 * @returns {{alphabet: string, nTokens: number, omitBias: number}}
 */
export function resolveFamily(manifest) {
  if (!manifest) {
    return { alphabet: MPNN_ALPHABET, nTokens: MPNN_ALPHABET.length, omitBias: OMIT_BIAS };
  }
  const alphabet = manifest.alphabet ?? MPNN_ALPHABET;
  if (typeof alphabet !== "string" || alphabet.length === 0) {
    throw new Error("manifest.alphabet must be a non-empty string");
  }
  const nTokens = alphabet.length;
  if (manifest.n_tokens !== undefined && manifest.n_tokens !== nTokens) {
    throw new Error(`manifest.n_tokens ${manifest.n_tokens} != alphabet length ${nTokens}`);
  }
  const omitBias = manifest.omit_bias ?? OMIT_BIAS;
  if (typeof omitBias !== "number" || !Number.isFinite(omitBias)) {
    throw new Error(`manifest.omit_bias must be a finite number, got ${omitBias}`);
  }
  return { alphabet, nTokens, omitBias };
}

/**
 * Resolve one graph's positional input order and dtypes. A declared
 * `input_names` list must have one entry per session input, else this throws.
 * Names are logical (coords, mask, ...); jax2onnx gives the ONNX tensors
 * positional names (in_0, in_1, ...), so the declared list is checked by count
 * against `session.inputNames` rather than by name.
 */
function resolveGraph(key, session, bucketGraphs) {
  const pinned = PINNED_GRAPHS[key];
  const entry = bucketGraphs ? bucketGraphs[key] : undefined;
  const declared = entry ? entry.input_names : undefined;
  if (declared === undefined) {
    checkInputCount(session, key, pinned.order);
    return pinned;
  }
  if (!Array.isArray(declared) || declared.length === 0) {
    throw new Error(`manifest ${key}.input_names must be a non-empty array`);
  }
  if (!declared.every((name) => typeof name === "string" && name.length > 0)) {
    throw new Error(`manifest ${key}.input_names must contain non-empty strings`);
  }
  if (new Set(declared).size !== declared.length) {
    throw new Error(`manifest ${key}.input_names contains duplicates: ${declared.join(", ")}`);
  }
  checkInputCount(session, `${key} (manifest input_names)`, declared);
  const dtypes = entry.input_dtypes ?? pinned.dtypes;
  if (dtypes.length !== declared.length) {
    throw new Error(`manifest ${key}.input_dtypes has ${dtypes.length} entries for ${declared.length} inputs`);
  }
  return { order: Object.freeze([...declared]), dtypes: Object.freeze([...dtypes]) };
}

function toOrtTensor(ort, dtype, tensor) {
  return new ort.Tensor(dtype, tensor.data, tensor.shape);
}

function fromOrtTensor(tensor) {
  return { shape: Array.from(tensor.dims), data: tensor.data };
}

function buildFeeds(ort, session, order, dtypes, inputs) {
  const feeds = {};
  for (let i = 0; i < order.length; i += 1) {
    const tensor = inputs[order[i]];
    if (tensor === undefined) {
      throw new Error(`no input tensor for graph input "${order[i]}" (session input ${session.inputNames[i]})`);
    }
    feeds[session.inputNames[i]] = toOrtTensor(ort, dtypes[i], tensor);
  }
  return feeds;
}

function readOutputs(session, names, result) {
  const out = {};
  for (let i = 0; i < names.length; i += 1) {
    out[names[i]] = fromOrtTensor(result[session.outputNames[i]]);
  }
  return out;
}

/**
 * Load the four split-graph ONNX Runtime Web sessions and wrap them with the
 * RunSpec -> tensor plumbing + wave loop, mirroring ./aminx_sampler.mjs's
 * `createSampler` shape.
 *
 * @param {typeof import("onnxruntime-web")} ort caller-supplied onnxruntime-web namespace
 * @param {{encoderUrl: string|ArrayBuffer|Uint8Array, waveUrl: string|ArrayBuffer|Uint8Array,
 *   decoderUrl: string|ArrayBuffer|Uint8Array, fuseUrl: string|ArrayBuffer|Uint8Array}} urls
 * @param {{numThreads?: number, manifest?: object, manifestBucket?: number}} [opts]
 *   `manifest` is the parsed MANIFEST.json; `manifestBucket` selects the bucket
 *   entry whose per-graph input_names/input_dtypes apply (required when the
 *   manifest has buckets). Omit both to use the ProteinMPNN constants.
 * @returns {Promise<{sample: (structure: object, runspec: object) => Promise<object>, release: () => Promise<void>}>}
 */
export async function createSplitSampler(
  ort,
  { encoderUrl, waveUrl, decoderUrl, fuseUrl },
  { numThreads, manifest, manifestBucket } = {},
) {
  if (numThreads !== undefined) {
    ort.env.wasm.numThreads = numThreads;
  }

  const family = resolveFamily(manifest);
  let bucketGraphs;
  if (manifest && Array.isArray(manifest.buckets)) {
    if (manifestBucket === undefined) {
      throw new Error("createSplitSampler: opts.manifestBucket is required when the manifest has buckets");
    }
    const entry = manifest.buckets.find((b) => b.bucket === manifestBucket);
    if (!entry) {
      throw new Error(`manifest has no bucket ${manifestBucket}`);
    }
    bucketGraphs = entry.graphs;
  }

  const [encoderSession, waveSession, decoderSession, fuseSession] = await Promise.all([
    loadSession(ort, encoderUrl),
    loadSession(ort, waveUrl),
    loadSession(ort, decoderUrl),
    loadSession(ort, fuseUrl),
  ]);

  const plan = {
    encoder: resolveGraph("encoder", encoderSession, bucketGraphs),
    wave: resolveGraph("wave", waveSession, bucketGraphs),
    decoder: resolveGraph("decoder", decoderSession, bucketGraphs),
    fuse: resolveGraph("fuse", fuseSession, bucketGraphs),
  };

  async function runEncoder(inputs) {
    const feeds = buildFeeds(ort, encoderSession, plan.encoder.order, plan.encoder.dtypes, inputs);
    const result = await encoderSession.run(feeds);
    return readOutputs(encoderSession, ENCODER_OUTPUT_NAMES, result);
  }

  async function runWave(inputs) {
    const feeds = buildFeeds(ort, waveSession, plan.wave.order, plan.wave.dtypes, inputs);
    const result = await waveSession.run(feeds);
    return readOutputs(waveSession, WAVE_OUTPUT_NAMES, result);
  }

  async function runDecoder(inputs) {
    const feeds = buildFeeds(ort, decoderSession, plan.decoder.order, plan.decoder.dtypes, inputs);
    const result = await decoderSession.run(feeds);
    return readOutputs(decoderSession, DECODER_OUTPUT_NAMES, result);
  }

  async function runFuse(inputs) {
    const feeds = buildFeeds(ort, fuseSession, plan.fuse.order, plan.fuse.dtypes, inputs);
    const result = await fuseSession.run(feeds);
    return readOutputs(fuseSession, FUSE_OUTPUT_NAMES, result);
  }

  return {
    async sample(structure, runspec) {
      const nReal = structure.coords_shape ? structure.coords_shape[0] : structure.coords.length;
      const bucket = pickBucket(nReal);
      const padded = padStructure(structure, bucket);
      const built = buildP07TypedInputs(padded, runspec, {
        alphabet: family.alphabet,
        omitBias: family.omitBias,
      });

      const result = await runSplitDecode(
        { runEncoder, runWave, runDecoder, runFuse },
        {
          length: bucket,
          nTokens: family.nTokens,
          encoderInputs: {
            coords: built.coords,
            mask: built.mask,
            residue_index: built.residue_index,
            chain_index: built.chain_index,
          },
          waveInputs: {
            decoding_order: built.decoding_order,
            tie_group_map: built.tie_group_map,
          },
          tieGroupMap: built.tie_group_map,
          mask: built.mask,
          condBias: built.bias,
          fixedMask: built.fixed_mask,
          fixedTokens: built.fixed_tokens,
          temperature: built.temperature,
          gumbelNoise: built.gumbel_noise,
        },
      );

      return { ...result, nReal, bucket };
    },
    async release() {
      await Promise.all([
        encoderSession.release(),
        waveSession.release(),
        decoderSession.release(),
        fuseSession.release(),
      ]);
    },
  };
}
