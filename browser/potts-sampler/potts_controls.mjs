/**
 * Design controls -> PottsMPNN graph inputs (browser port of prepare_sample's control steps).
 *
 * Reproduces, on top of a buildPottsInputs() result (no controls), the three control steps of
 * aminx's prepare_sample (src/aminx/families/potts_mpnn/sample_host.py, ~lines 270-355):
 *
 *   fixed_positions  chain_m_pos[i] = 0 for each listed global 0-based row i with 0 <= i < B
 *                    (B = padded bucket length, i.e. chain_m_pos.shape[0]; indices outside the
 *                    bucket are ignored).
 *   omit_aa          omit[k] = 1 for each letter's index in MODEL_ALPHABET (omit_letter_indices).
 *   bias             _split_bias: a 1-D vector of length L_total is per-position and is added to
 *                    every column of bias_by_res rows [0, L_total); a 1-D vector of length 21 is
 *                    the global bias (it replaces the zero vector); a (L_total, 21) array is
 *                    per-residue and is added to bias_by_res rows [0, L_total). Anything else
 *                    throws. When L_total == 21 a 1-D vector is per-position (that branch is
 *                    tested first in _split_bias).
 *
 * Only rows < L_total receive per-position/per-residue bias: aminx's `by_res[:length] += ...`
 * uses L_total, not the bucket.
 *
 * Pure ES module: no DOM, no dependencies. Never mutates its input.
 */

export const MODEL_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";
const N_AA = MODEL_ALPHABET.length;

export class PottsControlsError extends Error {
  constructor(message) {
    super(message);
    this.name = "PottsControlsError";
  }
}

function cloneInputs(inputs) {
  const out = {};
  for (const [name, entry] of Object.entries(inputs)) {
    out[name] = { dtype: entry.dtype, dims: [...entry.dims], data: entry.data.slice() };
  }
  return out;
}

/** fixed_positions: flat array of global row indices. Truncated to int like np.int32. */
function applyFixedPositions(chainMPos, fixed) {
  if (fixed === undefined || fixed === null) return;
  if (!Array.isArray(fixed)) {
    throw new PottsControlsError("fixed_positions must be an array of integer row indices");
  }
  const rows = chainMPos.length;
  for (const value of fixed) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new PottsControlsError(`fixed_positions entry ${JSON.stringify(value)} is not a finite number`);
    }
    const index = Math.trunc(value);
    if (index >= 0 && index < rows) chainMPos[index] = 0;
  }
}

/** omit_aa: a string of letters (or an array of strings, each split into letters). */
function omitLetterIndices(omit) {
  if (omit === undefined || omit === null) return [];
  let tokens;
  if (typeof omit === "string") {
    tokens = [omit];
  } else if (Array.isArray(omit) && omit.every((token) => typeof token === "string")) {
    tokens = omit;
  } else {
    throw new PottsControlsError("omit_aa must be a string of amino-acid letters");
  }
  const indices = [];
  for (const token of tokens) {
    for (const letter of token) {
      const index = MODEL_ALPHABET.indexOf(letter);
      if (index < 0) {
        throw new PottsControlsError(
          `omit amino acid ${JSON.stringify(letter)} is not in the MPNN alphabet ${JSON.stringify(MODEL_ALPHABET)}`,
        );
      }
      indices.push(index);
    }
  }
  return indices;
}

/** Shape of a bias spec: [n] for a vector, [n, m] for a rectangular matrix; throws otherwise. */
function biasShape(bias) {
  const fail = () => {
    throw new PottsControlsError(
      "bias must be a flat array of numbers or a rectangular array of rows of numbers",
    );
  };
  if (!Array.isArray(bias)) fail();
  if (bias.length === 0) return [0];
  if (bias.every((value) => typeof value === "number")) {
    if (!bias.every((value) => Number.isFinite(value))) {
      throw new PottsControlsError("bias contains a non-finite value");
    }
    return [bias.length];
  }
  if (bias.every((row) => Array.isArray(row))) {
    const cols = bias[0].length;
    for (const row of bias) {
      if (row.length !== cols || !row.every((value) => typeof value === "number")) fail();
      if (!row.every((value) => Number.isFinite(value))) {
        throw new PottsControlsError("bias contains a non-finite value");
      }
    }
    return [bias.length, cols];
  }
  return fail();
}

/**
 * Port of _split_bias. Returns the global Float32Array(21) bias; byRes (the caller's copy of
 * bias_by_res) is updated in place. `length` is L_total.
 */
function applyBias(bias, length, byRes) {
  const globalBias = new Float32Array(N_AA);
  if (bias === undefined || bias === null) return globalBias;
  const shape = biasShape(bias);
  const isVector = shape.length === 1;
  if (isVector && shape[0] === length) {
    // per-position: by_res[:length] += arr[:, None]
    for (let row = 0; row < length; row += 1) {
      for (let aa = 0; aa < N_AA; aa += 1) byRes[row * N_AA + aa] += bias[row];
    }
    return globalBias;
  }
  if (isVector && shape[0] === N_AA) {
    globalBias.set(bias);
    return globalBias;
  }
  if (shape.length === 2 && shape[0] === length && shape[1] === N_AA) {
    // per-residue: by_res[:length] += arr
    for (let row = 0; row < length; row += 1) {
      for (let aa = 0; aa < N_AA; aa += 1) byRes[row * N_AA + aa] += bias[row][aa];
    }
    return globalBias;
  }
  const shown = shape.length === 1 ? `(${shape[0]},)` : `(${shape[0]}, ${shape[1]})`;
  throw new PottsControlsError(
    `bias shape ${shown} is not (${N_AA},), (${length},), or (${length}, ${N_AA})`,
  );
}

/**
 * Apply design controls to built PottsMPNN inputs.
 *
 * @param {{inputs: Object<string, {dtype: string, dims: number[], data: ArrayLike}>,
 *          l_total: number, bucket: number}} built  result of buildPottsInputs (not mutated)
 * @param {{fixed_positions?: number[] | null, omit_aa?: string | null,
 *          bias?: number[] | number[][] | null}} [controls]
 * @returns {Object<string, {dtype: string, dims: number[], data: ArrayLike}>}
 *   a NEW inputs object with fresh data arrays (unchanged inputs are copied too).
 */
export function applyPottsControls(built, controls = {}) {
  if (!built || typeof built !== "object" || !built.inputs) {
    throw new PottsControlsError("applyPottsControls needs a buildPottsInputs() result");
  }
  const c = controls ?? {};
  const inputs = cloneInputs(built.inputs);
  const chainMPos = inputs.chain_m_pos.data;
  const byRes = inputs.bias_by_res.data;
  const length = built.l_total;
  if (!Number.isInteger(length) || length < 0) {
    throw new PottsControlsError(`built.l_total must be a non-negative integer, got ${length}`);
  }

  applyFixedPositions(chainMPos, c.fixed_positions);

  const omit = new Float32Array(N_AA);
  for (const index of omitLetterIndices(c.omit_aa)) omit[index] = 1;
  inputs.omit = { dtype: "float32", dims: [N_AA], data: omit };

  const globalBias = applyBias(c.bias, length, byRes);
  inputs.bias = { dtype: "float32", dims: [N_AA], data: globalBias };
  return inputs;
}
