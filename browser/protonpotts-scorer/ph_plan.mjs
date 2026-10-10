/**
 * Placement planning for ProtonPotts pH design: the JS port of src/aminx/families/protonpotts_mpnn/ph_plan.py.
 *
 * Pure integer/graph logic, no model arithmetic. Every function mirrors its Python namesake, including the
 * tie-breaking: ranked positions resolve ties to ascending position (np.argsort kind="stable"), knn_rank sorts on
 * (rank, position). Indices are aminx's v6 token order (protonpotts_inputs.mjs VOCABULARY). The neighbour table
 * `eIdx` is a flat Int32Array of shape [L, K] (slot 0 = the position itself); `field` is a flat array [L, V] of
 * conditional energies of the NATIVE sequence. Binder masks are Uint8Array/boolean arrays of length L.
 *
 * Python raises ValueError where this throws PhPlanError; a plan function returns null where Python returns None.
 */

import { VOCABULARY } from "./protonpotts_inputs.mjs";

export const CENTRE_TYPES = ["HIS-P", "ASP-P", "GLU-P"];
export const INFILL_SCOPES = ["neighbourhood", "chain"];
export const DEFAULT_DEP_MAP = { "HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"] };
const UNRANKED = 1_000_000;
const UNK_SYMBOL = "X";
const INDEX = new Map(VOCABULARY.map((symbol, i) => [symbol, i]));

export class PhPlanError extends Error {
  constructor(message) {
    super(message);
    this.name = "PhPlanError";
  }
}

/** Boolean (V,) mask of tokens the sampler may emit: everything except X and each forbidden token that exists. */
export function validTokenMask(forbiddenTokens) {
  const mask = new Uint8Array(VOCABULARY.length).fill(1);
  mask[INDEX.get(UNK_SYMBOL)] = 0;
  for (const name of forbiddenTokens) {
    const symbol = name === "UNK" ? UNK_SYMBOL : name;
    const idx = INDEX.get(symbol);
    if (idx !== undefined) mask[idx] = 0;
  }
  return mask;
}

/** Designable neighbourhood of one centre: coupled kNN in both graph directions, intersected with the binder. */
export function neighbourMask(eIdx, length, slots, centre, binderMask, neighbourK) {
  const kEff = neighbourK > 0 ? Math.min(slots, neighbourK + 1) : slots;
  const mask = new Uint8Array(length);
  for (let s = 1; s < kEff; s += 1) {
    const j = eIdx[centre * slots + s];
    if (j !== centre && binderMask[j]) mask[j] = 1;
  }
  for (let i = 0; i < length; i += 1) {
    if (i === centre || !binderMask[i]) continue;
    for (let s = 1; s < kEff; s += 1) {
      if (eIdx[i * slots + s] === centre) {
        mask[i] = 1;
        break;
      }
    }
  }
  return mask;
}

/** Designable positions ordered closest-coupled first (upstream _knn_rank). */
export function knnRank(eIdx, slots, designable, pins) {
  const best = new Map(designable.map((j) => [j, UNRANKED]));
  for (const pin of pins) {
    for (let s = 1; s < slots; s += 1) {
      const j = eIdx[pin.position * slots + s];
      if (best.has(j)) best.set(j, Math.min(best.get(j), s - 1));
    }
  }
  return [...best.keys()].sort((a, b) => best.get(a) - best.get(b) || a - b);
}

/** Block for position p: p plus its nearest designable partners, at most blockSize (upstream _block_partners). */
export function blockPartners(nbrRow, p, designableSet, blockSize) {
  const block = [p];
  if (blockSize <= 1) return block;
  for (const j of nbrRow) {
    if (j !== p && designableSet.has(j) && !block.includes(j)) {
      block.push(j);
      if (block.length === blockSize) break;
    }
  }
  return block;
}

/** Padded blocks for every designable position: { blocks: Int32Array(N*B) padded with -1, valid: Uint8Array(N*B) }. */
export function blockTable(eIdx, slots, designable, blockSize) {
  if (blockSize < 1) throw new PhPlanError(`block_size must be >= 1, got ${blockSize}`);
  const n = designable.length;
  const set = new Set(designable);
  const blocks = new Int32Array(n * blockSize).fill(-1);
  const valid = new Uint8Array(n * blockSize);
  designable.forEach((p, row) => {
    const nbr = Array.from(eIdx.subarray(p * slots + 1, (p + 1) * slots));
    const block = blockPartners(nbr, p, set, blockSize);
    block.forEach((q, col) => {
      blocks[row * blockSize + col] = q;
      valid[row * blockSize + col] = 1;
    });
  });
  return { blocks, valid };
}

/** Selective placement score per position, lower is better; +Infinity outside the free binder mask. */
export function placementScores(field, vocab, protIdx, depIdxs, binderFreeMask) {
  if (depIdxs.length === 0) throw new PhPlanError("placementScores needs at least one deprotonated contrast token");
  const length = field.length / vocab;
  const out = new Float64Array(length);
  for (let j = 0; j < length; j += 1) {
    if (!binderFreeMask[j]) {
      out[j] = Infinity;
      continue;
    }
    let low = Infinity;
    for (const d of depIdxs) low = Math.min(low, field[j * vocab + d]);
    out[j] = field[j * vocab + protIdx] - low;
  }
  return out;
}

/** Finite-score positions, best first; ties resolve to ascending position. */
export function rankedPositions(scores) {
  const idx = [];
  for (let p = 0; p < scores.length; p += 1) if (Number.isFinite(scores[p])) idx.push(p);
  return idx.sort((a, b) => scores[a] - scores[b] || a - b);
}

function pinIndices(protonationType, depMap) {
  if (!CENTRE_TYPES.includes(protonationType)) {
    throw new PhPlanError(`unknown centre protonation type ${JSON.stringify(protonationType)}; expected one of ${CENTRE_TYPES}`);
  }
  if (!(protonationType in depMap)) throw new PhPlanError(`dep_map has no contrast tokens for centre ${JSON.stringify(protonationType)}`);
  const deps = depMap[protonationType];
  if (deps.length === 0) throw new PhPlanError(`dep_map[${JSON.stringify(protonationType)}] is empty`);
  const missing = deps.filter((t) => !INDEX.has(t));
  if (missing.length) throw new PhPlanError(`dep_map[${JSON.stringify(protonationType)}] names ${missing}, absent from the vocabulary`);
  return [INDEX.get(protonationType), deps.map((t) => INDEX.get(t))];
}

function makePin(position, protonationType, depMap, resId) {
  const [protIdx, depIdxs] = pinIndices(protonationType, depMap);
  return { position, protonationType, protIdx, depIdxs, resId: resId[position] };
}

/** Pins plus their designable set, or null when nothing is designable (upstream _finalize_plan). */
export function finalizePlan(eIdx, length, slots, binderMask, pins, { infillScope, neighbourK, maxMutations }) {
  if (!INFILL_SCOPES.includes(infillScope)) throw new PhPlanError(`infill_scope must be one of ${INFILL_SCOPES}, got ${infillScope}`);
  if (maxMutations < 0) throw new PhPlanError(`max_mutations must be >= 0 (0 = no cap), got ${maxMutations}`);
  const sorted = [...pins].sort((a, b) => a.position - b.position);
  const pinPos = new Set(sorted.map((p) => p.position));
  const designable = new Set();
  if (infillScope === "chain") {
    for (let p = 0; p < length; p += 1) if (binderMask[p]) designable.add(p);
  } else {
    for (const pin of sorted) {
      const mask = neighbourMask(eIdx, length, slots, pin.position, binderMask, neighbourK);
      for (let p = 0; p < length; p += 1) if (mask[p]) designable.add(p);
    }
  }
  for (const p of pinPos) designable.delete(p);
  if (designable.size === 0) return null;
  let ordered = [...designable].sort((a, b) => a - b);
  if (maxMutations && ordered.length > maxMutations) {
    ordered = knnRank(eIdx, slots, ordered, sorted).slice(0, maxMutations).sort((a, b) => a - b);
  }
  const label = sorted.map((p) => p.protonationType).sort().join("+");
  return { pins: sorted, designable: ordered, label };
}

/** One plan with one distinct, best-ranked centre per requested type (upstream center_types branch). */
export function planFromCenterTypes(field, vocab, eIdx, length, slots, binderMask, resId, centerTypes, depMap, options) {
  let nBinder = 0;
  for (let p = 0; p < length; p += 1) nBinder += binderMask[p] ? 1 : 0;
  if (nBinder < centerTypes.length) return null;
  const used = new Set();
  const pins = [];
  for (const ptype of centerTypes) {
    const [protIdx, depIdxs] = pinIndices(ptype, depMap);
    const scores = placementScores(field, vocab, protIdx, depIdxs, binderMask);
    const pos = rankedPositions(scores).find((p) => !used.has(p));
    if (pos === undefined) return null;
    used.add(pos);
    pins.push(makePin(pos, ptype, depMap, resId));
  }
  return finalizePlan(eIdx, length, slots, binderMask, pins, options);
}

/** Pins at hand-chosen residue numbers, given as [resId, protonationType] pairs (upstream explicit_centers branch). */
export function planFromExplicitCenters(eIdx, length, slots, binderMask, resId, centers, depMap, options) {
  const posOf = new Map();
  for (let p = 0; p < length; p += 1) if (binderMask[p]) posOf.set(resId[p], p); // the last position wins
  const pins = centers.map(([rid, ptype]) => {
    const pos = posOf.get(rid);
    if (pos === undefined) throw new PhPlanError(`res_id ${rid} is not a free binder position`);
    return makePin(pos, ptype, depMap, resId);
  });
  return finalizePlan(eIdx, length, slots, binderMask, pins, options);
}

/** Plan with no pins and every free binder position designable (centre-free greedy). */
export function planCentreFree(binderMask) {
  const designable = [];
  for (let p = 0; p < binderMask.length; p += 1) if (binderMask[p]) designable.push(p);
  return designable.length ? { pins: [], designable, label: "" } : null;
}
