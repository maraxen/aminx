/**
 * PDB text (+ optional protonation labels) -> ProtonPottsMPNN ONNX graph inputs.
 *
 * Browser port of aminx's host path, line for line:
 *
 *   featurize_pdb(path, labels)         src/aminx/families/protonpotts_mpnn/features.py
 *   encode_sequence(names, labels)      src/aminx/families/protonpotts_mpnn/sequence.py
 *   ProtonPottsDriver._prepare          .../driver.py  (X[:, :4], present = 1, R_idx, chain_labels)
 *   padding                             potts_mpnn.featurize.pad  (coords 0, present 0, chain 0, residue_idx -100)
 *
 * The same refusals as the Python reader: MSE, a blank chain id, an ATOM residue that is not one of the 20
 * standard residues or UNK. A label outside the nine v6 tokens, or whose parent residue disagrees with the
 * residue name, throws. Pure ES module: no DOM, no dependencies.
 */

export const STANDARD_LETTERS = "ACDEFGHIKLMNPQRSTVWY";
export const V6_PROTONATION_TOKENS = [
  "HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A", "GLU-P", "GLU-D", "GLU-A",
];
export const VOCABULARY = [...STANDARD_LETTERS, "X", ...V6_PROTONATION_TOKENS];
export const N_TOKENS = VOCABULARY.length; // 30
export const X_INDEX = 20;
export const BUCKETS = [256, 1024];

const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
};
const TOKEN_INDEX = new Map(VOCABULARY.map((symbol, index) => [symbol, index]));
const PARENT = new Map(V6_PROTONATION_TOKENS.map((token) => [token, THREE_TO_ONE[token.slice(0, 3)]]));

const ATOM37_ORDER = [
  "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
  "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
  "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];
const ATOM_INDEX = new Map(ATOM37_ORDER.map((name, index) => [name, index]));
const BACKBONE = ["N", "CA", "C", "O"];
const BACKBONE_OCCUPANCY_THRESHOLD = 0.8;

const NUMBER_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const INT_RE = /^[+-]?\d+$/;

export class ProtonPottsInputError extends Error {
  constructor(message) {
    super(message);
    this.name = "ProtonPottsInputError";
  }
}

function numberField(text, what) {
  const trimmed = text.trim();
  if (!NUMBER_RE.test(trimmed)) throw new ProtonPottsInputError(`bad ${what} field ${JSON.stringify(trimmed)}`);
  return Number(trimmed);
}

function readResidues(pdbText) {
  const residues = [];
  let current = null;
  let currentKey = null;
  for (const raw of pdbText.split(/\r?\n/u)) {
    const record = raw.slice(0, 6);
    if (record.startsWith("HETATM")) {
      if (raw.slice(17, 20) === "MSE") {
        throw new ProtonPottsInputError("MSE (selenomethionine) conversion is not verified against upstream; refusing");
      }
      continue;
    }
    if (!record.startsWith("ATOM")) continue;
    const chain = raw[21];
    if (chain === " " || chain === undefined) {
      throw new ProtonPottsInputError("ATOM record with a blank chain id is not supported");
    }
    const name = raw.slice(12, 16).trim();
    const numberText = raw.slice(22, 26).trim();
    if (!INT_RE.test(numberText)) throw new ProtonPottsInputError(`bad residue number ${JSON.stringify(numberText)}`);
    const resName = raw.slice(17, 20);
    const key = `${chain}\u0000${Number(numberText)}\u0000${raw[26]}\u0000${resName}`;
    if (key !== currentKey) {
      if (!(resName in THREE_TO_ONE) && resName !== "UNK") {
        throw new ProtonPottsInputError(
          `residue ${JSON.stringify(resName)} in an ATOM record is not one of the 20 standard residues or UNK`,
        );
      }
      current = { chain, number: Number(numberText), icode: raw[26], name: resName, atoms: new Map() };
      residues.push(current);
      currentKey = key;
    }
    if (current.atoms.has(name)) continue; // alternate locations: the FIRST one wins
    const occText = raw.slice(54, 60);
    const occupancy = occText.trim() === "" ? 1.0 : numberField(occText, "occupancy");
    current.atoms.set(name, [
      numberField(raw.slice(30, 38), "x"),
      numberField(raw.slice(38, 46), "y"),
      numberField(raw.slice(46, 54), "z"),
      occupancy,
    ]);
  }
  return residues;
}

function f32Norm(a, b) {
  const f = Math.fround;
  const dx = f(f(a[0]) - f(b[0]));
  const dy = f(f(a[1]) - f(b[1]));
  const dz = f(f(a[2]) - f(b[2]));
  return f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz))));
}

/** Exchange NH1/NH2 coordinates when NH1 is the farther from CD (atomworks fix_arginines). */
function fixArginine(residue) {
  const { atoms } = residue;
  if (residue.name !== "ARG" || !atoms.has("CD") || !atoms.has("NH1") || !atoms.has("NH2")) return;
  const cd = atoms.get("CD");
  const nh1 = atoms.get("NH1");
  const nh2 = atoms.get("NH2");
  if (f32Norm(cd, nh1) > f32Norm(cd, nh2)) {
    atoms.set("NH1", [nh2[0], nh2[1], nh2[2], nh1[3]]);
    atoms.set("NH2", [nh1[0], nh1[1], nh1[2], nh2[3]]);
  }
}

/** atomworks keep_last_residue: at one (chain, number) keep only the last residue NAME. */
function keepLastByNumber(residues) {
  const namesAt = new Map();
  for (const residue of residues) {
    const key = `${residue.chain}\u0000${residue.number}`;
    if (!namesAt.has(key)) namesAt.set(key, []);
    const seen = namesAt.get(key);
    if (!seen.includes(residue.name)) seen.push(residue.name);
  }
  return residues.filter((residue) => {
    const seen = namesAt.get(`${residue.chain}\u0000${residue.number}`);
    return residue.name === seen[seen.length - 1];
  });
}

function backboneResolved(residue) {
  return BACKBONE.every((atom) => (residue.atoms.get(atom) ?? [0, 0, 0, 0])[3] > BACKBONE_OCCUPANCY_THRESHOLD);
}

/** [chain, number, icode, name] of the residues protonation labels must align to (before the backbone drop). */
export function loadedResidues(pdbText) {
  return keepLastByNumber(readResidues(pdbText)).map((r) => [r.chain, r.number, r.icode, r.name]);
}

/** encode_sequence: aminx v6 token indices from residue names and optional labels ('' / null = no label). */
export function encodeSequence(names, labels = null) {
  if (labels !== null && labels.length !== names.length) {
    throw new ProtonPottsInputError(`${labels.length} labels for ${names.length} residues`);
  }
  return names.map((name, i) => {
    const label = labels === null ? "" : (labels[i] ?? "");
    if (label) {
      if (!V6_PROTONATION_TOKENS.includes(label)) {
        throw new ProtonPottsInputError(`residue ${i}: label ${JSON.stringify(label)} is not a v6 protonation token`);
      }
      const parent = THREE_TO_ONE[name];
      if (parent !== undefined && PARENT.get(label) !== parent) {
        throw new ProtonPottsInputError(`residue ${i}: label ${JSON.stringify(label)} belongs to ${PARENT.get(label)} but the residue is ${name}`);
      }
      return TOKEN_INDEX.get(label);
    }
    const letter = THREE_TO_ONE[name];
    return letter === undefined ? X_INDEX : TOKEN_INDEX.get(letter);
  });
}

/**
 * Build the ProtonPotts table-graph inputs for one PDB text at a fixed padded length.
 *
 * @param {string} pdbText
 * @param {number} bucket   Padded length B (the graphs are compiled for 256 or 1024).
 * @param {(string|null)[]|null} labels  One entry per loadedResidues(pdbText), '' for none; or null.
 * @returns {{inputs: Object, lTotal: number, tokens: number[], kept: Array, loaded: Array, bucket: number}}
 *   `inputs` holds the five table-graph inputs; `tokens` is the unpadded native sequence (aminx token order).
 */
export function buildProtonPottsInputs(pdbText, bucket, labels = null) {
  if (!Number.isInteger(bucket) || bucket <= 0) {
    throw new ProtonPottsInputError(`bucket must be a positive integer, got ${bucket}`);
  }
  const all = keepLastByNumber(readResidues(pdbText));
  if (all.length === 0) throw new ProtonPottsInputError("no ATOM residues found");
  if (labels !== null && labels.length !== all.length) {
    throw new ProtonPottsInputError(`${labels.length} labels for ${all.length} loaded residues`);
  }
  for (const residue of all) fixArginine(residue);

  const fileIndex = [];
  const counts = new Map();
  for (const residue of all) {
    const n = counts.get(residue.chain) ?? 0;
    fileIndex.push(n);
    counts.set(residue.chain, n + 1);
  }
  const keptIdx = [];
  all.forEach((residue, i) => {
    if (backboneResolved(residue)) keptIdx.push(i);
  });
  if (keptIdx.length === 0) throw new ProtonPottsInputError("every residue has an unresolved backbone");
  const residues = keptIdx.map((i) => all[i]);
  const keptLabels = labels === null ? null : keptIdx.map((i) => labels[i]);
  const length = residues.length;
  if (bucket < length) throw new ProtonPottsInputError(`bucket (${bucket}) is shorter than L_total (${length})`);

  const coords = new Float32Array(bucket * 4 * 3);
  const present = new Float32Array(bucket);
  const residueIdx = new Int32Array(bucket).fill(-100);
  const chainIndex = new Int32Array(bucket);
  const padValid = new Uint8Array(bucket);
  const chainOrder = new Map();
  residues.forEach((residue, row) => {
    BACKBONE.forEach((atom, a) => {
      const point = residue.atoms.get(atom);
      if (point !== undefined) {
        for (let axis = 0; axis < 3; axis += 1) coords[row * 12 + a * 3 + axis] = point[axis]; // rounds to f32
      }
    });
    present[row] = 1;
    residueIdx[row] = fileIndex[keptIdx[row]];
    if (!chainOrder.has(residue.chain)) chainOrder.set(residue.chain, chainOrder.size);
    chainIndex[row] = chainOrder.get(residue.chain);
    padValid[row] = 1;
  });
  const tokens = encodeSequence(residues.map((r) => r.name), keptLabels);
  return {
    inputs: {
      coords: { dtype: "float32", dims: [bucket, 4, 3], data: coords },
      present: { dtype: "float32", dims: [bucket], data: present },
      residue_idx: { dtype: "int32", dims: [bucket], data: residueIdx },
      chain_index: { dtype: "int32", dims: [bucket], data: chainIndex },
      pad_valid: { dtype: "bool", dims: [bucket], data: padValid },
    },
    lTotal: length,
    tokens,
    kept: residues.map((r) => [r.chain, r.number, r.icode, r.name]),
    loaded: all.map((r) => [r.chain, r.number, r.icode, r.name]),
    bucket,
  };
}

/** Smallest bucket that holds lTotal residues, or null when none does. */
export function pickBucket(lTotal, buckets = BUCKETS) {
  const fit = [...buckets].sort((a, b) => a - b).find((size) => size >= lTotal);
  return fit === undefined ? null : fit;
}
