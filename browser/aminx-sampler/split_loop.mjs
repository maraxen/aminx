// Pure JS port of the four-graph autoregressive wave loop (T13 / task_id
// 260926_browser-export-loop). NO onnxruntime import, NO .onnx file, NO
// node: import -- this module is exercised entirely through injected
// `runEncoder`/`runWave`/`runDecoder`/`runFuse` callbacks so the loop
// semantics are unit-testable before the real split ONNX graphs exist.
//
// Ported line-for-line from src/aminx/inference/decode/autoregressive.py:
//   - step_fn (lines 469-594): per-wave group identity, mask_group,
//     decode-once/fuse-per-group, sequence update.
//   - post-hoc gather (lines 811-820): each position takes its group's
//     logits at the group's FIRST (wave, slot) occurrence.
//   - init_sequence (lines 430-437): UNDRAWN_TOKEN (-1) everywhere except
//     fixed positions, which start at fixed_tokens.
//
// Tensor convention: every tensor here is a plain object `{ shape, data }`
// with `data` a flat, row-major array-like (matches onnxruntime-web's
// `Tensor.dims`/`Tensor.data`, and runspec_core.mjs's `typedTensor` shape).
// `makeTensor`/`at` below let tests build/read these with nested arrays
// instead of hand-flattening indices.

export const UNDRAWN_TOKEN = -1;
// ProteinMPNN's token count, used when a caller does not pass `nTokens`. The loop
// is alphabet-agnostic: every per-token dimension comes from runSplitDecode's
// `nTokens` parameter.
export const DEFAULT_N_TOKENS = 21;

/** Flatten nested arrays / typed arrays into a flat plain Array, row-major. */
function flattenValues(node) {
  if (Array.isArray(node)) {
    const out = [];
    for (const item of node) out.push(...flattenValues(item));
    return out;
  }
  if (ArrayBuffer.isView(node) && !(node instanceof DataView)) {
    return Array.from(node);
  }
  return [node];
}

/**
 * Build a `{shape, data}` tensor from nested arrays (or an already-flat
 * array-like), asserting the element count matches `shape`.
 *
 * @param {number[]} shape
 * @param {unknown} values nested arrays, a flat array, or a typed array
 * @returns {{shape: number[], data: Array<unknown>}}
 */
export function makeTensor(shape, values) {
  const data = flattenValues(values);
  const expect = shape.reduce((prod, dim) => prod * dim, 1);
  if (data.length !== expect) {
    throw new Error(`tensor has ${data.length} values, shape ${shape} wants ${expect}`);
  }
  return { shape, data };
}

/** Row-major flat index of `indices` into a tensor of shape `shape`. */
function flatIndex(shape, indices) {
  if (indices.length !== shape.length) {
    throw new Error(`index arity ${indices.length} != shape arity ${shape.length} (shape ${shape})`);
  }
  let flat = 0;
  for (let axis = 0; axis < shape.length; axis += 1) {
    flat = flat * shape[axis] + indices[axis];
  }
  return flat;
}

/** Read one element of a `{shape, data}` tensor by full multi-index. */
export function at(tensor, ...indices) {
  return tensor.data[flatIndex(tensor.shape, indices)];
}

function zerosTensor(shape) {
  const size = shape.reduce((prod, dim) => prod * dim, 1);
  return { shape, data: new Float32Array(size) };
}

function logSoftmaxRow(row) {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sumExp = 0;
  for (const v of row) sumExp += Math.exp(v - max);
  const logSumExp = Math.log(sumExp) + max;
  return row.map((v) => v - logSumExp);
}

/**
 * Run the four-graph autoregressive wave loop.
 *
 * @param {object} callbacks
 * @param {(inputs: object) => Promise<{node_features, edge_features, neighbor_indices}>} callbacks.runEncoder
 * @param {(inputs: object) => Promise<{group_ids, group_positions, group_valid, position_valid, ar_mask, group_first_rank, pos_first_rank}>} callbacks.runWave
 * @param {(inputs: object) => Promise<{logits}>} callbacks.runDecoder
 * @param {(inputs: object) => Promise<{final_token, avg_stored}>} callbacks.runFuse
 * @param {object} params
 * @param {number} params.length structure length L (post-padding, bucket length)
 * @param {object} params.encoderInputs passed through verbatim to runEncoder
 * @param {object} params.waveInputs passed through verbatim to runWave
 * @param {{shape:number[],data}} params.tieGroupMap int32[L] tie-group id per position
 * @param {{shape:number[],data}} params.mask float32[L] decoder mask input (loop invariant)
 * @param {{shape:number[],data}} params.condBias float32[L,nTokens] additive logit bias
 * @param {{shape:number[],data}} params.fixedMask float32[L]
 * @param {{shape:number[],data}} params.fixedTokens int32[L]
 * @param {{shape:number[],data}} params.temperature float32[]
 * @param {{shape:number[],data}} params.gumbelNoise float32[L,nTokens]
 * @param {number} [params.nTokens=DEFAULT_N_TOKENS] alphabet size (21 for ProteinMPNN)
 * @returns {Promise<{tokens: {shape:number[],data:Int32Array}, logProbs: {shape:number[],data:Float32Array}}>}
 */
export async function runSplitDecode(callbacks, params) {
  const { runEncoder, runWave, runDecoder, runFuse } = callbacks;
  const {
    length: L,
    encoderInputs,
    waveInputs,
    tieGroupMap,
    mask,
    condBias,
    fixedMask,
    fixedTokens,
    temperature,
    gumbelNoise,
  } = params;
  const N_TOKENS = params.nTokens ?? DEFAULT_N_TOKENS;
  if (!Number.isInteger(N_TOKENS) || N_TOKENS < 1) {
    throw new Error(`nTokens must be a positive integer, got ${N_TOKENS}`);
  }
  if (condBias.shape[1] !== N_TOKENS) {
    throw new Error(`condBias has ${condBias.shape[1]} tokens per position, nTokens is ${N_TOKENS}`);
  }

  const encOut = await runEncoder(encoderInputs);
  const { node_features, edge_features, neighbor_indices } = encOut;

  const waveOut = await runWave(waveInputs);
  const { group_ids, group_positions, group_valid, ar_mask, group_first_rank, pos_first_rank } = waveOut;
  // `position_valid` is part of Graph W's output contract (used by the
  // incremental-cache decode path, not by this full-recompute JS loop) --
  // deliberately not destructured/consumed here.

  const [nWaves, G] = group_ids.shape;
  const noOccurrenceSentinel = nWaves * G;

  const sequence = new Int32Array(L).fill(UNDRAWN_TOKEN);
  for (let pos = 0; pos < L; pos += 1) {
    if (at(fixedMask, pos) > 0.5) sequence[pos] = at(fixedTokens, pos);
  }

  const logitsStack = []; // nWaves entries, each {shape:[G,nTokens], data}

  for (let w = 0; w < nWaves; w += 1) {
    const groupId = new Int32Array(G);
    const isFirst = new Uint8Array(G);
    for (let g = 0; g < G; g += 1) {
      const pos0 = at(group_positions, w, g, 0);
      groupId[g] = at(tieGroupMap, pos0);
      const isActive = at(group_valid, w, g) ? 1 : 0;
      const thisRank = w * G + g;
      isFirst[g] = isActive && at(group_first_rank, groupId[g]) === thisRank ? 1 : 0;
    }

    const maskGroup = new Uint8Array(G * L);
    for (let g = 0; g < G; g += 1) {
      if (!isFirst[g]) continue;
      for (let pos = 0; pos < L; pos += 1) {
        if (at(tieGroupMap, pos) === groupId[g]) maskGroup[g * L + pos] = 1;
      }
    }

    const waveHasActiveGroup = isFirst.some((v) => v);
    if (!waveHasActiveGroup) {
      logitsStack.push(zerosTensor([G, N_TOKENS]));
      continue;
    }

    const seqOh = new Float32Array(L * N_TOKENS);
    for (let pos = 0; pos < L; pos += 1) {
      const tok = sequence[pos];
      if (tok >= 0 && tok < N_TOKENS) seqOh[pos * N_TOKENS + tok] = 1;
    }

    const decOut = await runDecoder({
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      sequence_oh: { shape: [L, N_TOKENS], data: seqOh },
    });
    const logits3d = { shape: [1, L, N_TOKENS], data: decOut.logits.data };

    const fuseOut = await runFuse({
      logits: logits3d,
      cond_bias: condBias,
      mask_group: { shape: [G, L], data: maskGroup },
      fixed_mask: fixedMask,
      fixed_tokens: fixedTokens,
      group_id: { shape: [G], data: groupId },
      temperature,
      gumbel_noise: gumbelNoise,
    });
    const { final_token, avg_stored } = fuseOut;

    for (let pos = 0; pos < L; pos += 1) {
      let covers = false;
      let tokenSum = 0;
      for (let g = 0; g < G; g += 1) {
        if (maskGroup[g * L + pos]) {
          covers = true;
          tokenSum += at(final_token, g);
        }
      }
      if (covers) sequence[pos] = tokenSum;
    }

    logitsStack.push(avg_stored);
  }

  const logProbsData = new Float32Array(L * N_TOKENS);
  for (let pos = 0; pos < L; pos += 1) {
    const rank = at(pos_first_rank, pos);
    const scheduled = rank < noOccurrenceSentinel;
    const safeRank = scheduled ? rank : 0;
    const w = Math.floor(safeRank / G);
    const g = safeRank % G;

    const row = new Array(N_TOKENS).fill(0);
    if (scheduled) {
      for (let k = 0; k < N_TOKENS; k += 1) row[k] = at(logitsStack[w], g, k);
    }
    const logProbRow = logSoftmaxRow(row);
    for (let k = 0; k < N_TOKENS; k += 1) logProbsData[pos * N_TOKENS + k] = logProbRow[k];
  }

  return {
    tokens: { shape: [L], data: Int32Array.from(sequence) },
    logProbs: { shape: [L, N_TOKENS], data: logProbsData },
  };
}
