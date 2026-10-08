/**
 * PDB text -> PottsMPNN ONNX graph inputs (browser port of aminx's host-numpy path).
 *
 * Reproduces, for one structure with aminx's default PottsMPNNOptions and a spec whose
 * fixed_positions=None, omit_aa=(), bias=None:
 *
 *   parse_pdb_upstream(path, skip_gaps=False)[0]              featurize.py:182
 *   _featurize_one(parsed, PottsMPNNOptions(), name)          driver.py:770
 *     = tied_featurize_port([parsed], None, pssm_dict=None, bias_by_res_dict=None)
 *   prepare_sample(features, chains, options, spec, l_pad=B)  sample_host.py:270
 *
 * Pure ES module: no DOM, no dependencies.
 */

export const MODEL_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";
export const N_AA = 21;
export const X_INDEX = 20;
export const BUCKETS = [128, 256];

const ALPHA_1 = "ARNDCQEGHILKMFPSTWYV-";
const AA3 = [
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
  "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];
const AA3_INDEX = new Map(AA3.map((name, index) => [name, index]));
const BACKBONE_COUNT = 4; // N, CA, C, O
const PSSM_THRESHOLD = 0.0; // PottsMPNNOptions.pssm_threshold
const PSSM_DEFAULT_LOG_ODDS = 10000.0; // tied_featurize_port: pssm_log_odds default when no pssm_dict

// Upstream's chain alphabet is A-Z, a-z, then "0".."299". Only single characters can match
// the PDB chain column (line[21:22]), so the multi-character entries never select anything
// and are omitted here. The order of this list is irrelevant: chain order is re-sorted below.
const CHAIN_LETTERS = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"];
const CHAIN_SET = new Set(CHAIN_LETTERS);

const COORD_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const INT_RE = /^[+-]?\d+$/;

export class PottsInputError extends Error {
  constructor(message) {
    super(message);
    this.name = "PottsInputError";
  }
}

/** Python str ordering for the ASCII chain/icode strings used here. */
function byCodePoint(a, b) {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

function coordField(line, start) {
  // Python float(line[start:start+8]) strips whitespace and raises on an empty field.
  const text = line.slice(start, start + 8).trim();
  if (!COORD_RE.test(text)) {
    throw new PottsInputError(`bad coordinate field ${JSON.stringify(text)}`);
  }
  return Number(text);
}

function residueNumber(text) {
  if (!INT_RE.test(text)) {
    throw new PottsInputError(`bad residue number ${JSON.stringify(text)}`);
  }
  return Number(text);
}

/**
 * Split PDB text into per-chain ATOM record lists, applying parse_PDB_biounits' rewrites:
 * HETATM records whose residue is MSE become ATOM records with MET. Other HETATM records
 * are dropped, as are records whose chain column is not in the upstream alphabet (blank
 * chain IDs included).
 */
function splitByChain(pdbText) {
  const buckets = new Map();
  for (const raw of pdbText.split("\n")) {
    let line = raw.replace(/\s+$/u, "");
    if (line.slice(0, 6) === "HETATM" && line.slice(17, 20) === "MSE") {
      line = line.split("HETATM").join("ATOM  ").split("MSE").join("MET");
    }
    if (line.slice(0, 4) !== "ATOM") continue;
    const letter = line.slice(21, 22);
    if (!CHAIN_SET.has(letter)) continue;
    let lines = buckets.get(letter);
    if (!lines) {
      lines = [];
      buckets.set(letter, lines);
    }
    lines.push(line);
  }
  return buckets;
}

/**
 * Port of _parse_biounits for one chain (skip_gaps as in parse_pdb_upstream).
 * Returns null when the chain has no atoms, else { coords, sequence } where coords is an
 * array of [x, y, z] rows (4 per residue: N, CA, C, O; NaN where absent) and sequence is
 * the upstream one-letter string in which '-' marks gaps and nonstandard residue names.
 */
function parseChain(lines, skipGaps) {
  const xyz = new Map(); // resn -> Map(icode -> Map(atom -> [x, y, z]))
  const seq = new Map(); // resn -> Map(icode -> resname)
  let minResn = Infinity;
  let maxResn = -Infinity;
  for (const line of lines) {
    const atom = line.slice(12, 16).trim();
    const resname = line.slice(17, 20);
    const token = line.slice(22, 27).trim();
    if (token.length === 0) throw new PottsInputError("empty residue number field");
    const last = token.slice(-1);
    let icode;
    let resn;
    if (/^[A-Za-z]$/.test(last)) {
      icode = last;
      resn = residueNumber(token.slice(0, -1)) - 1;
    } else {
      icode = "";
      resn = residueNumber(token) - 1;
    }
    if (resn < minResn) minResn = resn;
    if (resn > maxResn) maxResn = resn;

    let byIcode = xyz.get(resn);
    if (!byIcode) {
      byIcode = new Map();
      xyz.set(resn, byIcode);
    }
    let atoms = byIcode.get(icode);
    if (!atoms) {
      atoms = new Map();
      byIcode.set(icode, atoms);
    }
    let names = seq.get(resn);
    if (!names) {
      names = new Map();
      seq.set(resn, names);
    }
    // First record wins for both the residue name and each atom name (altlocs included).
    if (!names.has(icode)) names.set(icode, resname);
    if (!atoms.has(atom)) {
      atoms.set(atom, [coordField(line, 30), coordField(line, 38), coordField(line, 46)]);
    }
  }
  if (xyz.size === 0) return null;

  const seqIdx = [];
  const rows = [];
  for (let resn = minResn; resn <= maxResn; resn += 1) {
    if (seq.has(resn)) {
      const names = seq.get(resn);
      for (const icode of [...names.keys()].sort(byCodePoint)) {
        seqIdx.push(AA3_INDEX.get(names.get(icode)) ?? 20);
        const atoms = xyz.get(resn).get(icode);
        for (const atom of ["N", "CA", "C", "O"]) {
          rows.push(atoms.get(atom) ?? [NaN, NaN, NaN]);
        }
      }
    } else if (!skipGaps) {
      seqIdx.push(20);
      for (let i = 0; i < BACKBONE_COUNT; i += 1) rows.push([NaN, NaN, NaN]);
    }
  }
  return { coords: rows, sequence: seqIdx.map((index) => ALPHA_1[index]).join("") };
}

/** Parse every chain present, returned as a Map letter -> { coords, sequence }. */
function parseChains(pdbText, skipGaps) {
  const out = new Map();
  for (const [letter, lines] of splitByChain(pdbText)) {
    const parsed = parseChain(lines, skipGaps);
    if (parsed !== null) out.set(letter, parsed);
  }
  return out;
}

/**
 * Build the PottsMPNN graph inputs for one PDB text at a fixed padded length.
 *
 * @param {string} pdbText  PDB file contents.
 * @param {number} bucket   Padded length B (the ONNX graphs are compiled for 128 or 256).
 * @returns {{inputs: Object<string, {dtype: string, dims: number[], data: ArrayBufferView}>,
 *            l_total: number, sequence: string, chains: string[], bucket: number}}
 *   `sequence` is the native sequence as aminx passes it to prepare_sample: '-' for gaps
 *   and nonstandard residues. `inputs` holds the 17 graph inputs. pad_valid is a Uint8Array
 *   tagged dtype "bool".
 */
export function buildPottsInputs(pdbText, bucket) {
  if (!Number.isInteger(bucket) || bucket <= 0) {
    throw new PottsInputError(`bucket must be a positive integer, got ${bucket}`);
  }
  const parsed = parseChains(pdbText, false);
  // Chain order: upstream sorts the designed chains (all present chains here) by Python str order.
  const chainOrder = [...parsed.keys()].sort(byCodePoint);
  let lTotal = 0;
  for (const letter of chainOrder) lTotal += parsed.get(letter).sequence.length;
  if (bucket < lTotal) {
    throw new PottsInputError(`l_pad (${bucket}) is shorter than L_total (${lTotal})`);
  }

  const B = bucket;
  const coords = new Float32Array(B * BACKBONE_COUNT * 3);
  const present = new Float32Array(B);
  const residueIdx = new Int32Array(B).fill(-100); // pad tail
  const chainIndex = new Int32Array(B); // pad 0
  const padValid = new Uint8Array(B);
  const sTrue = new Int32Array(B).fill(X_INDEX);
  const chainMask = new Float32Array(B);
  const chainMPos = new Float32Array(B);
  const tieGroups = new Int32Array(B).fill(-1); // untied: singleton per real row, then -1
  const tiedBeta = new Float32Array(B).fill(1.0);
  const omit = new Float32Array(N_AA);
  const bias = new Float32Array(N_AA);
  const biasByRes = new Float32Array(B * N_AA);
  const pssmCoef = new Float32Array(B);
  const pssmBias = new Float32Array(B * N_AA);
  const pssmLogOddsMask = new Float32Array(B * N_AA);
  const omitAaMask = new Float32Array(B * N_AA);

  const native = [];
  let row = 0; // global residue row (gap rows included)
  let cursor = 0; // upstream row_cursor: cumulative across chains, never reset
  chainOrder.forEach((letter, chainPos) => {
    const chain = parsed.get(letter);
    native.push(chain.sequence);
    const xSeq = chain.sequence.split("-").join("X"); // _dash_to_x
    for (let i = 0; i < xSeq.length; i += 1, row += 1) {
      for (let atom = 0; atom < BACKBONE_COUNT; atom += 1) {
        const point = chain.coords[i * BACKBONE_COUNT + atom];
        for (let axis = 0; axis < 3; axis += 1) {
          const value = point[axis];
          // Float32Array assignment rounds to nearest, as numpy astype(float32) does.
          coords[row * 12 + atom * 3 + axis] = Number.isNaN(value) ? 0 : value;
        }
      }
      present[row] = chain.coords
        .slice(i * BACKBONE_COUNT, (i + 1) * BACKBONE_COUNT)
        .every((point) => point.every((value) => Number.isFinite(value)))
        ? 1
        : 0;
      sTrue[row] = MODEL_ALPHABET.indexOf(xSeq[i]);
      residueIdx[row] = 100 * chainPos + cursor + i;
      chainIndex[row] = chainPos + 1; // 1-based chain_encoding
      padValid[row] = 1;
      chainMask[row] = 1; // every present chain is designed (chain_dict is None)
      chainMPos[row] = 1; // fixed_positions is None
      tieGroups[row] = row;
      pssmLogOddsMask.fill(
        PSSM_DEFAULT_LOG_ODDS > PSSM_THRESHOLD ? 1 : 0,
        row * N_AA,
        (row + 1) * N_AA,
      );
    }
    cursor += xSeq.length;
  });

  return {
    inputs: {
      coords: { dtype: "float32", dims: [B, 4, 3], data: coords },
      present: { dtype: "float32", dims: [B], data: present },
      residue_idx: { dtype: "int32", dims: [B], data: residueIdx },
      chain_index: { dtype: "int32", dims: [B], data: chainIndex },
      pad_valid: { dtype: "bool", dims: [B], data: padValid },
      s_true: { dtype: "int32", dims: [B], data: sTrue },
      chain_mask: { dtype: "float32", dims: [B], data: chainMask },
      chain_m_pos: { dtype: "float32", dims: [B], data: chainMPos },
      tie_groups: { dtype: "int32", dims: [B, 1], data: tieGroups },
      tied_beta: { dtype: "float32", dims: [B], data: tiedBeta },
      omit: { dtype: "float32", dims: [N_AA], data: omit },
      bias: { dtype: "float32", dims: [N_AA], data: bias },
      bias_by_res: { dtype: "float32", dims: [B, N_AA], data: biasByRes },
      pssm_coef: { dtype: "float32", dims: [B], data: pssmCoef },
      pssm_bias: { dtype: "float32", dims: [B, N_AA], data: pssmBias },
      pssm_log_odds_mask: { dtype: "float32", dims: [B, N_AA], data: pssmLogOddsMask },
      omit_aa_mask: { dtype: "float32", dims: [B, N_AA], data: omitAaMask },
    },
    l_total: lTotal,
    sequence: native.join(""),
    chains: chainOrder,
    bucket: B,
  };
}

/**
 * Smallest bucket that holds l_total residues, or null when none does.
 * (aminx's gate runs a structure in every bucket it fits; choosing the smallest is this
 * helper's policy, not something the Python code states.)
 */
export function pickPottsBucket(lTotal, buckets = BUCKETS) {
  const fit = [...buckets].sort((a, b) => a - b).find((size) => size >= lTotal);
  return fit === undefined ? null : fit;
}
