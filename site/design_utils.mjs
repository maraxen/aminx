// RunSpec input parsing + design scoring, for app.js and its worker.
//
// Imports MPNN_ALPHABET from ./pdb_parse.mjs rather than
// browser/aminx-sampler/runspec_core.mjs -- same reason pdb_parse.mjs gives
// at its own top: site/*.mjs (excluding worker.js) has no path coupling to
// browser/, so nothing here needs a build-time rewrite.
import { MPNN_ALPHABET } from "./pdb_parse.mjs";

/**
 * Parse a comma-separated PDB-numbering position list ("A10,A11" or, with an
 * insertion code, "A10A") into a RunSpec `fixed_positions` map, pinning each
 * resolved index to the structure's OWN native residue at that position
 * (i.e. "keep this position as-is" -- there is no way to type a target
 * letter from this field; use bias/omit for that).
 *
 * Chain ids are assumed to be a single run of letters/digits that does not
 * itself end in a digit, so "A10" splits as chain "A" + residue 10. A
 * multi-character chain id ending in a digit (rare) is not resolvable this
 * way; the caller gets a clear error rather than a silent misparse.
 *
 * @param {string} text
 * @param {object} structure a structure built by pdb_parse.buildStructure
 *   (must carry `residue_keys` and `native_tokens`)
 * @returns {Record<number, string>}
 */
export function parseFixedPositions(text, structure) {
  const result = {};
  if (!text || !text.trim()) return result;
  for (const rawToken of text.split(",")) {
    const token = rawToken.trim();
    if (!token) continue;
    const index = resolvePdbPosition(token, structure);
    const tokenIndex = structure.native_tokens[index];
    result[index] = MPNN_ALPHABET[tokenIndex] ?? "X";
  }
  return result;
}

/**
 * Parse a ';'-separated list of '='-joined PDB-numbering position groups
 * ("A10=A40; A11=A41") into a RunSpec `tied_positions` array of 0-based
 * index groups.
 *
 * @param {string} text
 * @param {object} structure
 * @returns {number[][]}
 */
export function parseTiedPositions(text, structure) {
  const groups = [];
  if (!text || !text.trim()) return groups;
  for (const rawGroup of text.split(";")) {
    const group = rawGroup.trim();
    if (!group) continue;
    const indices = group.split("=").map((t) => t.trim()).filter(Boolean)
      .map((token) => resolvePdbPosition(token, structure));
    if (indices.length > 1) groups.push(indices);
  }
  return groups;
}

function resolvePdbPosition(token, structure) {
  const match = token.match(/^([A-Za-z0-9]+?)(-?\d+)([A-Za-z]?)$/);
  if (!match) {
    throw new Error(`cannot parse position "${token}" (expected e.g. "A10" or, with an insertion code, "A10A")`);
  }
  const [, chainId, resSeqText, iCodeRaw] = match;
  const resSeq = Number(resSeqText);
  const iCode = iCodeRaw || "";
  const keys = structure.residue_keys;
  if (!keys) {
    throw new Error("structure has no residue_keys; build it with pdb_parse.buildStructure");
  }
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    if (key.chainId === chainId && key.resSeq === resSeq && (key.iCode || "") === iCode) {
      return i;
    }
  }
  throw new Error(`position "${token}" not found in the loaded structure`);
}

/**
 * Parse a comma-separated "LETTER:value" list ("A:-1,W:0.5") into a RunSpec
 * `bias_AA` map. Letter is upper-cased; value is a bare number (no units).
 *
 * @param {string} text
 * @returns {Record<string, number>}
 */
export function parseBias(text) {
  const result = {};
  if (!text || !text.trim()) return result;
  for (const rawToken of text.split(",")) {
    const token = rawToken.trim();
    if (!token) continue;
    const parts = token.split(":");
    if (parts.length !== 2) {
      throw new Error(`cannot parse bias entry "${token}" (expected e.g. "A:-1")`);
    }
    const [letterRaw, valueText] = parts.map((part) => part.trim());
    const letter = letterRaw.toUpperCase();
    if (!letter) {
      throw new Error(`cannot parse bias entry "${token}" (expected e.g. "A:-1")`);
    }
    const value = Number(valueText);
    if (!Number.isFinite(value)) {
      throw new Error(`bias value for "${letter}" is not a number: "${valueText}"`);
    }
    result[letter] = value;
  }
  return result;
}

/**
 * Which structure positions a RunSpec actually designs: real (mask=1),
 * not individually fixed, and -- when `chains_to_design` is set -- on one
 * of the listed chains. Does NOT expand tied_positions groups (a UI-level
 * simplification: a tie group is scored/recovered position-by-position, not
 * promoted as a whole the way the sampler's own decoding order does).
 *
 * @param {object} structure
 * @param {object} runspec
 * @returns {Uint8Array}
 */
export function designedMask(structure, runspec) {
  const length = structure.mask.length;
  const mask = new Uint8Array(length);
  const fixed = new Set(Object.keys(runspec.fixed_positions || {}).map(Number));
  const chainsToDesign = runspec.chains_to_design ? new Set(runspec.chains_to_design) : null;
  for (let i = 0; i < length; i += 1) {
    if (!structure.mask[i]) continue;
    if (fixed.has(i)) continue;
    if (chainsToDesign && !chainsToDesign.has(structure.chain_ids[i])) continue;
    mask[i] = 1;
  }
  return mask;
}

/**
 * Mean negative log-probability of the sampled tokens over designed
 * positions only (ProteinMPNN-style per-design "score").
 *
 * @param {ArrayLike<number>} tokens sampled token indices, length >= nReal
 * @param {ArrayLike<number>} logProbs flat [nReal_or_more * 21] log-probs
 * @param {ArrayLike<number>} designedMaskArray 0/1 per position, length nReal
 * @param {number} nReal
 * @returns {number}
 */
export function scoreDesign(tokens, logProbs, designedMaskArray, nReal) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < nReal; i += 1) {
    if (!designedMaskArray[i]) continue;
    const token = tokens[i];
    sum += -logProbs[i * 21 + token];
    count += 1;
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Sequence recovery (fraction identical to native) over designed positions
 * only.
 *
 * @param {ArrayLike<number>} tokens sampled token indices
 * @param {ArrayLike<number>} native native_tokens (same alphabet)
 * @param {ArrayLike<number>} designedMaskArray 0/1 per position
 * @returns {number}
 */
export function recovery(tokens, native, designedMaskArray) {
  let match = 0;
  let count = 0;
  for (let i = 0; i < designedMaskArray.length; i += 1) {
    if (!designedMaskArray[i]) continue;
    count += 1;
    if (tokens[i] === native[i]) match += 1;
  }
  return count > 0 ? match / count : 0;
}

/**
 * Render a list of designs as ProteinMPNN-style FASTA text.
 *
 * @param {Array<{index: number, temperature: number, seed: number,
 *   score: number, recovery: number, sequence: string}>} designs
 * @returns {string}
 */
export function toFasta(designs) {
  const lines = [];
  for (const design of designs) {
    lines.push(
      `>design_${design.index}, T=${design.temperature}, seed=${design.seed}, `
      + `score=${design.score.toFixed(4)}, recovery=${design.recovery.toFixed(4)}`,
    );
    lines.push(design.sequence);
  }
  return lines.length ? `${lines.join("\n")}\n` : "";
}
