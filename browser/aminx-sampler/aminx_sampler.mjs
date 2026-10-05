// Browser-ready ES module for the exported P07 ProteinMPNN sampler.
//
// NO node: imports, NO bundler required -- a plain static site can
// `import { createSampler } from "./aminx_sampler.mjs"` directly (plus its
// sibling `runspec_core.mjs`, and onnxruntime-web) with a `<script
// type="module">` tag, the same way browser/layer_c/parity.mjs imports
// onnxruntime-web today.
//
// The PRNG/gumbel/shuffle/buildP07Inputs logic lives in ./runspec_core.mjs,
// shared byte-for-byte with the Node CLI/test harness at
// browser/layer_c/runspec.mjs -- see that file's header. This module adds
// only what the Node side never needed: length-bucket selection and padding
// (mirroring src/aminx/export/buckets.py's `pad_inputs` +
// scripts/browser_validation/p07_knobs_gate.py's `_pad_geometry`), and a
// thin onnxruntime-web session wrapper.
import { MPNN_ALPHABET, asFloat32, asInt32, buildP07TypedInputs } from "./runspec_core.mjs";

export { MPNN_ALPHABET };

// The 11 P07 ONNX input names, in the model's positional input order.
// Mirrors P07_KEYS in scripts/browser_validation/p07_knobs_gate.py exactly
// (that ordering is itself pinned to the model's actual `session.inputNames`
// order via `js_build`'s parity check on the Python/JAX side).
export const P07_INPUT_ORDER = Object.freeze([
  "coords",
  "mask",
  "residue_index",
  "chain_index",
  "gumbel_noise",
  "decoding_order",
  "bias",
  "fixed_mask",
  "fixed_tokens",
  "temperature",
  "tie_group_map",
]);

// Pinned P07 export bucket ladder (D-C's FULL_BUCKETS in p07_knobs_gate.py;
// distinct from src/aminx/export/buckets.py's full EXPORT_BUCKETS ladder,
// which also serves P512/P1024 -- the browser build only ships 128/256).
export const BUCKETS = Object.freeze([128, 256]);

/**
 * Smallest bucket in BUCKETS that is >= L.
 *
 * @param {number} length real (unpadded) residue count
 * @returns {number}
 */
export function pickBucket(length) {
  for (const bucket of BUCKETS) {
    if (length <= bucket) return bucket;
  }
  const largest = BUCKETS[BUCKETS.length - 1];
  throw new Error(`length ${length} exceeds the largest browser export bucket ${largest}`);
}

function structureLength(structure) {
  return structure.coords_shape ? structure.coords_shape[0] : structure.coords.length;
}

/**
 * Pad a real-length structure up to `bucket`, matching
 * src/aminx/export/buckets.py's `pad_inputs` + p07_knobs_gate.py's
 * `_pad_geometry` convention exactly: coords pad with 0, mask pads with 0
 * (padded rows are invalid), residue_index/chain_index pad by repeating the
 * last real value, chain_ids pads by repeating the last real chain id, and
 * native_tokens (if present) pads with 0.
 *
 * @param {object} structure {coords, coords_shape?, mask, residue_index,
 *   chain_index, chain_ids?, native_tokens?}
 * @param {number} bucket target padded length; must be >= the real length
 * @returns {object} a new structure object, padded to `bucket` on axis 0
 */
export function padStructure(structure, bucket) {
  const nReal = structureLength(structure);
  if (bucket < nReal) {
    throw new Error(`bucket ${bucket} is smaller than n_real ${nReal}`);
  }
  if (bucket === nReal) {
    return structure;
  }

  const coords = asFloat32(structure.coords, structure.coords_shape || [nReal, 4, 3]);
  const mask = asFloat32(structure.mask, [nReal]);
  const residueIndex = asInt32(structure.residue_index, [nReal]);
  const chainIndex = asInt32(structure.chain_index, [nReal]);

  const coordsOut = new Float32Array(bucket * 4 * 3);
  coordsOut.set(coords, 0);

  const maskOut = new Float32Array(bucket);
  maskOut.set(mask, 0);

  const residueOut = new Int32Array(bucket);
  residueOut.set(residueIndex, 0);
  const lastResidue = nReal > 0 ? residueIndex[nReal - 1] : 0;
  residueOut.fill(lastResidue, nReal);

  const chainOut = new Int32Array(bucket);
  chainOut.set(chainIndex, 0);
  const lastChain = nReal > 0 ? chainIndex[nReal - 1] : 0;
  chainOut.fill(lastChain, nReal);

  const padded = {
    ...structure,
    coords: coordsOut,
    coords_shape: [bucket, 4, 3],
    mask: maskOut,
    residue_index: residueOut,
    chain_index: chainOut,
  };

  const chainIds = structure.chain_ids;
  if (Array.isArray(chainIds) && chainIds.length === nReal) {
    const lastChainId = nReal > 0 ? chainIds[nReal - 1] : undefined;
    padded.chain_ids = chainIds.concat(Array(bucket - nReal).fill(lastChainId));
  }

  if (structure.native_tokens) {
    const nativeTokens = asInt32(structure.native_tokens, [nReal]);
    const nativeOut = new Int32Array(bucket);
    nativeOut.set(nativeTokens, 0);
    padded.native_tokens = nativeOut;
  }

  return padded;
}

/**
 * The 11 P07 input tensors as live typed arrays (no Array.from copy) -- each
 * `.data` is handed straight to `new ort.Tensor(dtype, data, shape)`.
 *
 * @param {object} structure
 * @param {object} [runspec]
 * @returns {Record<string, {dtype: string, shape: number[], data: (Float32Array|Int32Array)}>}
 */
export function buildInputs(structure, runspec) {
  return buildP07TypedInputs(structure, runspec);
}

/**
 * Load an ONNX Runtime Web session for the P07 sampler and wrap it with the
 * RunSpec -> tensor plumbing, so a caller only ever hands over a structure +
 * runspec and gets amino-acid logits/samples back.
 *
 * @param {typeof import("onnxruntime-web")} ort the onnxruntime-web module
 *   namespace (caller-supplied so this file has no bundler-visible import of
 *   its own -- same convention as browser/layer_c/parity.mjs).
 * @param {string|ArrayBuffer|Uint8Array} modelUrlOrBuffer a URL to fetch (a
 *   plain string) or an already-loaded model buffer.
 * @param {{numThreads?: number}} [options]
 * @returns {Promise<{sample: (structure: object, runspec: object) => Promise<{outputs: Record<string, unknown>, nReal: number, bucket: number}>, release: () => Promise<void>}>}
 */
export async function createSampler(ort, modelUrlOrBuffer, { numThreads } = {}) {
  if (numThreads !== undefined) {
    ort.env.wasm.numThreads = numThreads;
  }
  let modelSource = modelUrlOrBuffer;
  if (typeof modelUrlOrBuffer === "string") {
    const res = await fetch(modelUrlOrBuffer);
    if (!res.ok) {
      throw new Error(`fetch ${modelUrlOrBuffer} failed: ${res.status} ${res.statusText}`);
    }
    modelSource = await res.arrayBuffer();
  }

  const session = await ort.InferenceSession.create(modelSource, { executionProviders: ["wasm"] });
  if (session.inputNames.length !== P07_INPUT_ORDER.length) {
    throw new Error(
      `session has ${session.inputNames.length} inputs, P07_INPUT_ORDER declares ${P07_INPUT_ORDER.length}`,
    );
  }

  return {
    async sample(structure, runspec) {
      const nReal = structureLength(structure);
      const bucket = pickBucket(nReal);
      const padded = padStructure(structure, bucket);
      const inputs = buildInputs(padded, runspec);

      const feeds = {};
      for (let i = 0; i < P07_INPUT_ORDER.length; i += 1) {
        const name = P07_INPUT_ORDER[i];
        const tensor = inputs[name];
        feeds[session.inputNames[i]] = new ort.Tensor(tensor.dtype, tensor.data, tensor.shape);
      }

      const outputs = await session.run(feeds);
      return { outputs, nReal, bucket };
    },
    async release() {
      await session.release();
    },
  };
}
