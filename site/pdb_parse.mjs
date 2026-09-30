// Pure PDB / mmCIF parsing and structure building -- no DOM.
//
// This module deliberately does NOT import browser/aminx-sampler/runspec_core.mjs
// even though it needs the same amino-acid alphabet: site/ ships as a
// standalone static-site source tree (see tools/build_site.py), and only
// site/worker.js's import of the sampler is rewritten by the build to point
// at the copied-in dist/aminx-sampler/. Every other site/*.mjs file, this one
// included, keeps zero path coupling to browser/ so it never needs a build-time
// rewrite. The alphabet below MUST match aminx's MPNN_ALPHABET
// (src/aminx/utils/aa_convert.py / browser/aminx-sampler/runspec_core.mjs)
// byte-for-byte.
export const MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";

// Three-letter -> one-letter residue code, standard 20 plus MSE
// (selenomethionine, a HETATM record that stands in for Met in most
// crystal structures -- treated here as plain M, never dropped as a
// hetero group).
const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E",
  GLY: "G", HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F",
  PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
  MSE: "M",
};

// Backbone atoms a residue must carry, in the order aminx's structure format
// wants them: N, CA, C, O (see browser_integration.md's "Structure format").
const BACKBONE = ["N", "CA", "C", "O"];

function newResidue(chainId, resSeq, iCode, resName) {
  return { chainId, resSeq, iCode, resName, atoms: new Map() };
}

function residueKey(chainId, resSeq, iCode) {
  return `${chainId}\u0000${resSeq}\u0000${iCode || ""}`;
}

/**
 * Parse a PDB-format text into an intermediate residue list. First MODEL
 * only (stops at the first ENDMDL); HETATM records are dropped except MSE;
 * only the primary altLoc ("" or "A") is kept.
 *
 * @param {string} text
 * @returns {{format: "pdb", residues: Array<object>}}
 */
export function parsePdb(text) {
  const residues = [];
  const byKey = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("ENDMDL")) break;
    const record = line.slice(0, 6).trim();
    if (record !== "ATOM" && record !== "HETATM") continue;
    if (line.length < 54) continue; // not enough columns for coordinates

    const resName = line.slice(17, 20).trim();
    if (record === "HETATM" && resName !== "MSE") continue;

    const altLoc = line[16] || " ";
    if (altLoc !== " " && altLoc !== "A") continue;

    const name = line.slice(12, 16).trim();
    const chainId = (line[21] || " ").trim();
    const resSeq = parseInt(line.slice(22, 26), 10);
    const iCode = (line[26] || " ").trim();
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    const element = line.length >= 78 ? line.slice(76, 78).trim() : name.slice(0, 1);
    const bfactor = line.length >= 66 ? parseFloat(line.slice(60, 66)) : NaN;

    const key = residueKey(chainId, resSeq, iCode);
    let residue = byKey.get(key);
    if (!residue) {
      residue = newResidue(chainId, resSeq, iCode, resName);
      byKey.set(key, residue);
      residues.push(residue);
    }
    residue.atoms.set(name, { x, y, z, element, bfactor });
  }
  return { format: "pdb", residues };
}

// Splits one mmCIF loop data row into tokens, honoring '...' / "..." quoting
// (whitespace inside a quote is not a separator). Good enough for atom_site
// rows, which never carry the multi-line ';'-quoted text mmCIF also allows.
function tokenizeCifRow(line) {
  const tokens = [];
  let i = 0;
  while (i < line.length) {
    while (i < line.length && /\s/.test(line[i])) i += 1;
    if (i >= line.length) break;
    if (line[i] === "'" || line[i] === '"') {
      const quote = line[i];
      let end = line.indexOf(quote, i + 1);
      if (end === -1) end = line.length;
      tokens.push(line.slice(i + 1, end));
      i = end + 1;
    } else {
      let end = i;
      while (end < line.length && !/\s/.test(line[end])) end += 1;
      tokens.push(line.slice(i, end));
      i = end;
    }
  }
  return tokens;
}

// Finds the first `loop_` block whose column tags are `_atom_site.*` and
// returns its columns + raw data rows (tokenized). Only the FIRST atom_site
// loop is used, which is all a single-model deposition ever has.
function findAtomSiteLoop(text) {
  const lines = text.split(/\r?\n/);
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim() === "loop_") {
      let j = i + 1;
      const columns = [];
      while (j < lines.length && lines[j].trim().startsWith("_")) {
        columns.push(lines[j].trim());
        j += 1;
      }
      if (columns.length > 0 && columns[0].startsWith("_atom_site.")) {
        const rows = [];
        while (j < lines.length) {
          const trimmed = lines[j].trim();
          if (trimmed === "" || trimmed === "#" || trimmed.startsWith("_") ||
              trimmed === "loop_" || trimmed.startsWith("data_")) {
            break;
          }
          rows.push(tokenizeCifRow(trimmed));
          j += 1;
        }
        return { columns, rows };
      }
      i = j;
      continue;
    }
    i += 1;
  }
  return null;
}

function cifColumnIndex(columns, candidates) {
  for (const name of candidates) {
    const index = columns.indexOf(name);
    if (index !== -1) return index;
  }
  return -1;
}

/**
 * Parse an mmCIF text's `_atom_site` loop into the same intermediate residue
 * list parsePdb produces. Prefers `auth_*` fields (PDB-compatible numbering)
 * over `label_*`, falling back to `label_*` when `auth_*` is absent. First
 * `pdbx_PDB_model_num` only.
 *
 * @param {string} text
 * @returns {{format: "mmcif", residues: Array<object>}}
 */
export function parseMmcif(text) {
  const table = findAtomSiteLoop(text);
  if (!table) return { format: "mmcif", residues: [] };
  const { columns, rows } = table;

  const iGroup = cifColumnIndex(columns, ["_atom_site.group_PDB"]);
  const iAtom = cifColumnIndex(columns, ["_atom_site.auth_atom_id", "_atom_site.label_atom_id"]);
  const iComp = cifColumnIndex(columns, ["_atom_site.auth_comp_id", "_atom_site.label_comp_id"]);
  const iAsym = cifColumnIndex(columns, ["_atom_site.auth_asym_id", "_atom_site.label_asym_id"]);
  const iSeq = cifColumnIndex(columns, ["_atom_site.auth_seq_id", "_atom_site.label_seq_id"]);
  const iIns = cifColumnIndex(columns, ["_atom_site.pdbx_PDB_ins_code"]);
  const iX = cifColumnIndex(columns, ["_atom_site.Cartn_x"]);
  const iY = cifColumnIndex(columns, ["_atom_site.Cartn_y"]);
  const iZ = cifColumnIndex(columns, ["_atom_site.Cartn_z"]);
  const iModel = cifColumnIndex(columns, ["_atom_site.pdbx_PDB_model_num"]);
  const iElem = cifColumnIndex(columns, ["_atom_site.type_symbol"]);
  const iAlt = cifColumnIndex(columns, ["_atom_site.label_alt_id"]);
  const iBfactor = cifColumnIndex(columns, ["_atom_site.B_iso_or_equiv"]);

  const residues = [];
  const byKey = new Map();
  let firstModel = null;
  for (const row of rows) {
    const model = iModel !== -1 ? row[iModel] : "1";
    if (firstModel === null) firstModel = model;
    if (model !== firstModel) continue;

    const group = iGroup !== -1 ? row[iGroup] : "ATOM";
    const resName = iComp !== -1 ? row[iComp] : "";
    if (group === "HETATM" && resName !== "MSE") continue;

    const altLoc = iAlt !== -1 ? row[iAlt] : ".";
    if (altLoc !== "." && altLoc !== "?" && altLoc !== "A") continue;

    const atomName = iAtom !== -1 ? row[iAtom] : "";
    const chainId = iAsym !== -1 ? row[iAsym] : "";
    const resSeq = iSeq !== -1 ? parseInt(row[iSeq], 10) : 0;
    const iCodeRaw = iIns !== -1 ? row[iIns] : ".";
    const iCode = iCodeRaw === "." || iCodeRaw === "?" ? "" : iCodeRaw;
    const x = parseFloat(row[iX]);
    const y = parseFloat(row[iY]);
    const z = parseFloat(row[iZ]);
    const element = iElem !== -1 ? row[iElem] : atomName.slice(0, 1);
    const bfactor = iBfactor !== -1 ? parseFloat(row[iBfactor]) : NaN;

    const key = residueKey(chainId, resSeq, iCode);
    let residue = byKey.get(key);
    if (!residue) {
      residue = newResidue(chainId, resSeq, iCode, resName);
      byKey.set(key, residue);
      residues.push(residue);
    }
    residue.atoms.set(atomName, { x, y, z, element, bfactor });
  }
  return { format: "mmcif", residues };
}

/**
 * Build an aminx structure object (coords/mask/residue_index/chain_index/
 * chain_ids/native_tokens, see browser_integration.md's "Structure format")
 * from a parsed residue list. Residues missing any of N/CA/C/O are dropped.
 * Adds `residue_keys` (PDB chainId/resSeq/iCode per row, in the same order
 * as every other array) and `insertion_codes` -- extra fields the sampler
 * ignores, used by design_utils.mjs to map PDB numbering back to indices.
 *
 * @param {{residues: Array<object>}} parsed
 * @param {{chains?: string[]}} [options] restrict to these chain ids (all
 *   chains if omitted)
 */
export function buildStructure(parsed, options = {}) {
  const chainFilter = options.chains ? new Set(options.chains) : null;

  const coords = [];
  const mask = [];
  const residueIndex = [];
  const chainIndex = [];
  const chainIds = [];
  const nativeTokens = [];
  const residueKeys = [];
  const insertionCodes = [];

  const chainOrderIndex = new Map();
  const chainOrdinal = (id) => {
    if (!chainOrderIndex.has(id)) chainOrderIndex.set(id, chainOrderIndex.size);
    return chainOrderIndex.get(id);
  };

  for (const residue of parsed.residues) {
    if (chainFilter && !chainFilter.has(residue.chainId)) continue;
    if (!BACKBONE.every((name) => residue.atoms.has(name))) continue;

    coords.push(BACKBONE.map((name) => {
      const atom = residue.atoms.get(name);
      return [atom.x, atom.y, atom.z];
    }));
    mask.push(1);
    residueIndex.push(residue.resSeq);
    chainIndex.push(chainOrdinal(residue.chainId));
    chainIds.push(residue.chainId);
    const letter = THREE_TO_ONE[residue.resName];
    nativeTokens.push(letter ? MPNN_ALPHABET.indexOf(letter) : 20);
    residueKeys.push({ chainId: residue.chainId, resSeq: residue.resSeq, iCode: residue.iCode });
    insertionCodes.push(residue.iCode || "");
  }

  return {
    coords,
    mask,
    residue_index: residueIndex,
    chain_index: chainIndex,
    chain_ids: chainIds,
    native_tokens: nativeTokens,
    residue_keys: residueKeys,
    insertion_codes: insertionCodes,
  };
}

/**
 * Lists the distinct chain ids present in a parsed residue list, in file
 * order, restricted to residues with a complete backbone (i.e. the chains
 * `buildStructure` would actually be able to design).
 *
 * @param {{residues: Array<object>}} parsed
 * @returns {string[]}
 */
export function listChains(parsed) {
  const seen = new Set();
  const order = [];
  for (const residue of parsed.residues) {
    if (!BACKBONE.every((name) => residue.atoms.has(name))) continue;
    if (!seen.has(residue.chainId)) {
      seen.add(residue.chainId);
      order.push(residue.chainId);
    }
  }
  return order;
}

function pad(value, width) {
  const text = String(value);
  return text.length >= width ? text.slice(0, width) : " ".repeat(width - text.length) + text;
}

function formatAtomLine(serial, atomName, resName, chainId, resSeq, iCode, x, y, z, bfactor, element) {
  const name4 = atomName.length >= 4 ? atomName.slice(0, 4) : (` ${atomName}`).padEnd(4);
  const ser = pad(serial, 5);
  const rn = pad(resName, 3);
  const ch = (chainId || "A").slice(0, 1);
  const rs = pad(resSeq, 4);
  const ic = iCode || " ";
  const xs = pad(x.toFixed(3), 8);
  const ys = pad(y.toFixed(3), 8);
  const zs = pad(z.toFixed(3), 8);
  const occ = pad("1.00", 6);
  const bf = pad(bfactor.toFixed(2), 6);
  const el = pad(element || "", 2);
  return `ATOM  ${ser} ${name4} ${rn} ${ch}${rs}${ic}   ${xs}${ys}${zs}${occ}${bf}          ${el}`;
}

/**
 * Re-emit a parsed structure as PDB text with each residue's atoms' B-factor
 * column replaced by a supplied per-residue value (for py2Dmol's
 * `setColor('plddt')`, which reads B-factor as a 0-100 value off the CA
 * atom). Residues with no matching entry keep a B-factor of 0.
 *
 * @param {{residues: Array<object>}} parsed
 * @param {Array<{chainId: string, resSeq: number, iCode?: string, value: number}>} perResidueValues
 *   `value` is expected in [0, 1]; it is written out as `value * 100`,
 *   clamped to [0, 100].
 * @returns {string} PDB text, one ATOM record per atom, terminated by END.
 */
export function writePdbWithBfactors(parsed, perResidueValues) {
  const values = new Map();
  for (const entry of perResidueValues) {
    values.set(residueKey(entry.chainId, entry.resSeq, entry.iCode || ""), entry.value);
  }

  const lines = [];
  let serial = 1;
  for (const residue of parsed.residues) {
    const raw = values.get(residueKey(residue.chainId, residue.resSeq, residue.iCode));
    const bfactor = raw === undefined ? 0 : Math.max(0, Math.min(100, raw * 100));
    for (const [name, atom] of residue.atoms) {
      lines.push(formatAtomLine(
        serial, name, residue.resName, residue.chainId, residue.resSeq, residue.iCode,
        atom.x, atom.y, atom.z, bfactor, atom.element,
      ));
      serial += 1;
    }
  }
  lines.push("END");
  return `${lines.join("\n")}\n`;
}
