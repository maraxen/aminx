// RunSpec -> P07 input tensors: browser-safe core (no node: imports).
//
// This module holds the numeric logic shared by the Node CLI/test harness
// (browser/layer_c/runspec.mjs, which re-exports everything here and adds
// node:fs/node:path-based file helpers on top) and the browser-only sampler
// (browser/aminx-sampler/aminx_sampler.mjs). Keeping the PRNG, gumbel,
// shuffle/fixedFirstShuffle, tieGroupMap and buildP07Inputs logic in exactly
// ONE place means the two callers can never diverge by hand-copying.
//
// Alphabet is aminx's MPNN_ALPHABET (src/aminx/utils/aa_convert.py):
// "ACDEFGHIKLMNPQRSTVWYX".
//
// Randomness is one SplitMix64 stream per (structure, runspec, seed):
//   1. unbiased Fisher-Yates, then a stable fixed-first partition, when
//      decoding_order is "random" (ProteinMPNN argsort of chain_mask)
//   2. then Gumbel noise (L, 21) in row-major order
// Uniforms are in (0, 1). Gumbel is g = -log(-log(u)) stored as float32.
// Tie-group ids are the smallest member of each group so every id stays in 0..L-1
// (ONNX Gather of the noise by group id is undefined out of range).

export const MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";
export const OMIT_BIAS = -1e8;

const MASK64 = (1n << 64n) - 1n;
const GOLDEN = 0x9E3779B97F4A7C15n;
const MIX_1 = 0xBF58476D1CE4E5B9n;
const MIX_2 = 0x94D049BB133111EBn;
const U32_SPAN = 0x100000000;
const MANTISSA_DENOM = 16777217; // 2^24 + 1, so (mant + 1) / denom is in (0, 1)

const _f32 = new Float32Array(1);

export function toFloat32(value) {
  _f32[0] = value;
  return _f32[0];
}

export function makePrng(seed) {
  let state = BigInt(seed >>> 0);
  return {
    nextU64() {
      state = (state + GOLDEN) & MASK64;
      let z = state;
      z = ((z ^ (z >> 30n)) * MIX_1) & MASK64;
      z = ((z ^ (z >> 27n)) * MIX_2) & MASK64;
      return (z ^ (z >> 31n)) & MASK64;
    },
    nextU32() {
      return Number(this.nextU64() >> 32n);
    },
  };
}

export function uniformFromU32(u32) {
  const mant = (u32 >>> 8) & 0xffffff;
  return toFloat32((mant + 1) / MANTISSA_DENOM);
}

export function gumbelFromUniform(u) {
  const uu = toFloat32(u);
  const inner = toFloat32(-Math.log(uu));
  return toFloat32(-Math.log(inner));
}

export function badGumbelFromUniform(u) {
  const uu = toFloat32(u);
  return toFloat32(-Math.log(uu));
}

export function nextBelow(prng, bound) {
  if (bound <= 0 || bound > U32_SPAN) {
    throw new Error(`nextBelow bound out of range: ${bound}`);
  }
  const limit = U32_SPAN - (U32_SPAN % bound);
  let draw = prng.nextU32();
  while (draw >= limit) {
    draw = prng.nextU32();
  }
  return draw % bound;
}

export function shuffle(length, prng) {
  const order = new Int32Array(length);
  for (let i = 0; i < length; i += 1) order[i] = i;
  for (let i = length - 1; i > 0; i -= 1) {
    const j = nextBelow(prng, i + 1);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

export function designedFlags(length, mask, fixedMask, tie) {
  const designed = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    if (mask[i] !== 0 && fixedMask[i] === 0) designed[i] = 1;
  }
  const group = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    if (designed[i]) group[tie[i]] = 1;
  }
  for (let i = 0; i < length; i += 1) {
    if (group[tie[i]]) designed[i] = 1;
  }
  return designed;
}

// One Fisher-Yates, then a stable partition: design-mask 0 (fixed, chain not
// designed, or padding) first, designed positions after. A tie group that
// contains any designed position is designed, so `designed` must already be
// expanded through tie groups.
//
// Within-group uniformity: shuffle() is an unbiased Fisher-Yates, so π is
// uniform on S_L. Let F = {i | designed[i] === 0} and D its complement. The
// stable partition writes F in the order those positions appear in π, then D
// the same way, and it draws no further random numbers. In a uniform random
// permutation the relative order of any fixed subset is itself uniform: each
// of the |F|! orders of F is paired with the same number of interleavings and
// orders of D, namely L! / |F|!, so each has probability 1/|F|!, and each
// order of D has probability 1/|D|!. The PRNG stream after this call matches
// a plain shuffle of the same length.
export function fixedFirstShuffle(length, prng, designed) {
  const order = shuffle(length, prng);
  let nFixed = 0;
  for (let i = 0; i < length; i += 1) {
    if (designed[order[i]] === 0) nFixed += 1;
  }
  const out = new Int32Array(length);
  let head = 0;
  let tail = nFixed;
  for (let i = 0; i < length; i += 1) {
    const pos = order[i];
    if (designed[pos] === 0) {
      out[head] = pos;
      head += 1;
    } else {
      out[tail] = pos;
      tail += 1;
    }
  }
  return out;
}

export function biasedShuffle(length, prng) {
  const order = new Int32Array(length);
  for (let i = 0; i < length; i += 1) order[i] = i;
  for (let i = 0; i < length; i += 1) {
    const j = nextBelow(prng, length);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

// Returns letter -> column index for `alphabet`, throwing on letters outside it.
function makeLetterIndex(alphabet) {
  const indexOf = Object.fromEntries([...alphabet].map((letter, index) => [letter, index]));
  return (letter) => {
    const index = indexOf[letter];
    if (index === undefined) {
      throw new Error(`letter ${letter} is not in alphabet ${alphabet}`);
    }
    return index;
  };
}

export function tieGroupMap(length, groups) {
  const ids = new Int32Array(length);
  for (let i = 0; i < length; i += 1) ids[i] = i;
  for (const group of groups || []) {
    if (group.length === 0) continue;
    let gid = group[0];
    for (const member of group) {
      if (member < gid) gid = member;
    }
    for (const member of group) {
      if (member < 0 || member >= length) {
        throw new Error(`tie member ${member} outside 0..${length - 1}`);
      }
      ids[member] = gid;
    }
  }
  return ids;
}

// Flattens nested plain arrays (the Node/test convention) AND flat typed
// arrays (the browser convention, e.g. a Float32Array fetched over the wire)
// into a plain JS array of numbers, so asFloat32/asInt32 accept either.
export function flattenNumeric(values) {
  const out = [];
  const walk = (node) => {
    if (Array.isArray(node)) {
      for (const item of node) walk(item);
    } else if (ArrayBuffer.isView(node) && !(node instanceof DataView)) {
      for (let i = 0; i < node.length; i += 1) out.push(node[i]);
    } else {
      out.push(Number(node));
    }
  };
  walk(values);
  return out;
}

export function asFloat32(values, shape) {
  const flat = flattenNumeric(values);
  const data = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) data[i] = flat[i];
  const expect = shape.reduce((prod, dim) => prod * dim, 1);
  if (data.length !== expect) {
    throw new Error(`float tensor has ${data.length} values, shape ${shape} wants ${expect}`);
  }
  return data;
}

export function asInt32(values, shape) {
  const flat = flattenNumeric(values);
  const data = new Int32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) data[i] = flat[i];
  const expect = shape.reduce((prod, dim) => prod * dim, 1);
  if (data.length !== expect) {
    throw new Error(`int tensor has ${data.length} values, shape ${shape} wants ${expect}`);
  }
  return data;
}

function positionEntries(mapping) {
  if (!mapping) return [];
  return Object.entries(mapping).map(([key, value]) => [Number(key), value]);
}

// Wraps a typed array as {dtype, shape, data} with NO copy -- the data is
// handed to callers (e.g. `new ort.Tensor(dtype, data, shape)`) as-is.
function typedTensor(dtype, data, shape) {
  return { dtype, shape, data };
}

export function inferCoordsShape(coords) {
  return [coords.length, coords[0].length, coords[0][0].length];
}

/**
 * Core of buildP07Inputs: identical logic, but every tensor's `.data` is the
 * live typed array (Float32Array/Int32Array) with no Array.from copy. Used
 * directly by the browser sampler (aminx-sampler/aminx_sampler.mjs), which
 * feeds `.data` straight into `new ort.Tensor(...)`.
 *
 * @param {object} structure
 * @param {object} [runspec]
 * @param {{alphabet?: string, omitBias?: number}} [options] model family knobs;
 *   default to ProteinMPNN (MPNN_ALPHABET, OMIT_BIAS). The token count is alphabet.length.
 */
export function buildP07TypedInputs(structure, runspec, options = {}) {
  const spec = runspec || {};
  const alphabet = options.alphabet ?? MPNN_ALPHABET;
  const nTokens = alphabet.length;
  const letterIndex = makeLetterIndex(alphabet);
  const coords = asFloat32(structure.coords, structure.coords_shape || inferCoordsShape(structure.coords));
  const length = structure.coords_shape ? structure.coords_shape[0] : structure.coords.length;
  const mask = asFloat32(structure.mask, [length]);
  const residueIndex = asInt32(structure.residue_index, [length]);
  const chainIndex = asInt32(structure.chain_index, [length]);
  const chainIds = structure.chain_ids || [];
  const nativeTokens = structure.native_tokens
    ? asInt32(structure.native_tokens, [length])
    : new Int32Array(length);

  if (chainIds.length !== 0 && chainIds.length !== length) {
    throw new Error(`chain_ids length ${chainIds.length} != L ${length}`);
  }

  const temperature = toFloat32(spec.temperature === undefined ? 0.1 : spec.temperature);
  const bias = new Float32Array(length * nTokens);
  for (const [letter, value] of Object.entries(spec.bias_AA || {})) {
    const column = letterIndex(letter);
    const delta = toFloat32(value);
    for (let pos = 0; pos < length; pos += 1) {
      bias[pos * nTokens + column] = toFloat32(bias[pos * nTokens + column] + delta);
    }
  }
  for (const [pos, letters] of positionEntries(spec.bias_AA_per_residue)) {
    for (const [letter, value] of Object.entries(letters)) {
      const column = letterIndex(letter);
      bias[pos * nTokens + column] = toFloat32(bias[pos * nTokens + column] + toFloat32(value));
    }
  }
  const omitValue = toFloat32(options.omitBias ?? OMIT_BIAS);
  for (const letter of spec.omit_AA || "") {
    const column = letterIndex(letter);
    for (let pos = 0; pos < length; pos += 1) bias[pos * nTokens + column] = omitValue;
  }
  for (const [pos, letters] of positionEntries(spec.omit_AA_per_residue)) {
    for (const letter of letters) {
      bias[pos * nTokens + letterIndex(letter)] = omitValue;
    }
  }

  const fixedMask = new Float32Array(length);
  const fixedTokens = new Int32Array(length);
  if (spec.chains_to_design) {
    if (chainIds.length !== length) {
      throw new Error("chains_to_design requires structure.chain_ids of length L");
    }
    const design = new Set(spec.chains_to_design);
    for (let pos = 0; pos < length; pos += 1) {
      if (!design.has(chainIds[pos])) {
        fixedMask[pos] = 1;
        fixedTokens[pos] = nativeTokens[pos];
      }
    }
  }
  for (const [pos, letter] of positionEntries(spec.fixed_positions)) {
    fixedMask[pos] = 1;
    fixedTokens[pos] = letterIndex(letter);
  }

  const tie = tieGroupMap(length, spec.tied_positions || []);
  const seed = spec.seed >>> 0;
  const prng = makePrng(seed);
  let decodingOrder;
  const requested = spec.decoding_order === undefined ? "random" : spec.decoding_order;
  if (requested === "random") {
    const designed = designedFlags(length, mask, fixedMask, tie);
    decodingOrder = fixedFirstShuffle(length, prng, designed);
  } else {
    decodingOrder = Int32Array.from(requested);
    if (decodingOrder.length !== length) {
      throw new Error(`explicit decoding_order length ${decodingOrder.length} != L ${length}`);
    }
    const seen = new Set(decodingOrder);
    if (seen.size !== length) {
      throw new Error("explicit decoding_order is not a permutation");
    }
  }

  const gumbel = new Float32Array(length * nTokens);
  for (let i = 0; i < gumbel.length; i += 1) {
    gumbel[i] = gumbelFromUniform(uniformFromU32(prng.nextU32()));
  }

  return {
    coords: typedTensor("float32", coords, [length, 4, 3]),
    mask: typedTensor("float32", mask, [length]),
    residue_index: typedTensor("int32", residueIndex, [length]),
    chain_index: typedTensor("int32", chainIndex, [length]),
    gumbel_noise: typedTensor("float32", gumbel, [length, nTokens]),
    decoding_order: typedTensor("int32", decodingOrder, [length]),
    bias: typedTensor("float32", bias, [length, nTokens]),
    fixed_mask: typedTensor("float32", fixedMask, [length]),
    fixed_tokens: typedTensor("int32", fixedTokens, [length]),
    temperature: typedTensor("float32", Float32Array.of(temperature), []),
    tie_group_map: typedTensor("int32", tie, [length]),
  };
}

/**
 * Node/test-harness convention: identical to buildP07TypedInputs, but each
 * tensor's `.data` is a plain Array (Array.from copy) so the result can be
 * JSON.stringify'd (the Node CLI path, consumed by
 * scripts/browser_validation/p07_knobs_gate.py) and compared with
 * assert.deepEqual against plain-array fixtures (runspec.test.mjs).
 *
 * @param {object} structure
 * @param {object} [runspec]
 * @param {{alphabet?: string, omitBias?: number}} [options] see buildP07TypedInputs
 */
export function buildP07Inputs(structure, runspec, options = {}) {
  const typed = buildP07TypedInputs(structure, runspec, options);
  const out = {};
  for (const [key, t] of Object.entries(typed)) {
    out[key] = { dtype: t.dtype, shape: t.shape, data: Array.from(t.data) };
  }
  return out;
}
