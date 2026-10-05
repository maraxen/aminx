import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildStructure,
  listChains,
  parseMmcif,
  parsePdb,
  writePdbWithBfactors,
} from "../pdb_parse.mjs";

// Independent fixed-column PDB line writer (deliberately NOT reusing
// pdb_parse.mjs's own formatter) so this fixture is a genuine external
// input, not a round-trip of the code under test.
function pdbLine(serial, name, resName, chainId, resSeq, iCode, x, y, z, bfactor, element, record = "ATOM") {
  const name4 = name.length >= 4 ? name.padEnd(4) : ` ${name}`.padEnd(4);
  const rec = record.padEnd(6);
  const ser = String(serial).padStart(5);
  const rn = resName.padStart(3);
  const rs = String(resSeq).padStart(4);
  const ic = iCode || " ";
  const xs = x.toFixed(3).padStart(8);
  const ys = y.toFixed(3).padStart(8);
  const zs = z.toFixed(3).padStart(8);
  const bf = bfactor.toFixed(2).padStart(6);
  const el = element.padStart(2);
  return `${rec}${ser} ${name4} ${rn} ${chainId}${rs}${ic}   ${xs}${ys}${zs}  1.00${bf}          ${el}`;
}

// Fixture: chain A has ALA@10 (full backbone), GLY@11 (missing O, must be
// dropped), MSE@12 as a HETATM (must be kept and mapped to M). Chain B has
// one SER at residue 1 with insertion code "A".
function samplePdbText() {
  const lines = [
    pdbLine(1, "N", "ALA", "A", 10, "", 0.0, 0.0, 0.0, 10.0, "N"),
    pdbLine(2, "CA", "ALA", "A", 10, "", 1.0, 0.0, 0.0, 20.0, "C"),
    pdbLine(3, "C", "ALA", "A", 10, "", 2.0, 0.0, 0.0, 30.0, "C"),
    pdbLine(4, "O", "ALA", "A", 10, "", 3.0, 0.0, 0.0, 40.0, "O"),
    pdbLine(5, "N", "GLY", "A", 11, "", 4.0, 0.0, 0.0, 10.0, "N"),
    pdbLine(6, "CA", "GLY", "A", 11, "", 5.0, 0.0, 0.0, 20.0, "C"),
    pdbLine(7, "C", "GLY", "A", 11, "", 6.0, 0.0, 0.0, 30.0, "C"),
    // GLY@11's O is deliberately omitted -- this residue must be dropped.
    pdbLine(8, "N", "MSE", "A", 12, "", 7.0, 0.0, 0.0, 10.0, "N", "HETATM"),
    pdbLine(9, "CA", "MSE", "A", 12, "", 8.0, 0.0, 0.0, 20.0, "C", "HETATM"),
    pdbLine(10, "C", "MSE", "A", 12, "", 9.0, 0.0, 0.0, 30.0, "C", "HETATM"),
    pdbLine(11, "O", "MSE", "A", 12, "", 10.0, 0.0, 0.0, 40.0, "O", "HETATM"),
    pdbLine(12, "SE", "MSE", "A", 12, "", 11.0, 0.0, 0.0, 50.0, "SE", "HETATM"),
    pdbLine(13, "N", "SER", "B", 1, "A", 12.0, 0.0, 0.0, 10.0, "N"),
    pdbLine(14, "CA", "SER", "B", 1, "A", 13.0, 0.0, 0.0, 20.0, "C"),
    pdbLine(15, "C", "SER", "B", 1, "A", 14.0, 0.0, 0.0, 30.0, "C"),
    pdbLine(16, "O", "SER", "B", 1, "A", 15.0, 0.0, 0.0, 40.0, "O"),
    // A water: HETATM, not MSE -- must be skipped entirely.
    pdbLine(17, "O", "HOH", "A", 101, "", 20.0, 0.0, 0.0, 0.0, "O", "HETATM"),
  ];
  return `${lines.join("\n")}\n`;
}

test("parsePdb groups residues in file order, keeps MSE as a residue, skips other HETATM", () => {
  const parsed = parsePdb(samplePdbText());
  assert.equal(parsed.format, "pdb");
  // 4 real residues in the file (ALA, GLY, MSE, SER) + 1 water grouped as
  // its own "residue" that build/listChains will reject for lacking backbone.
  const names = parsed.residues.map((r) => r.resName);
  assert.deepEqual(names, ["ALA", "GLY", "MSE", "SER"]);
});

test("buildStructure keeps ALA/MSE/SER, drops GLY (missing O), maps MSE -> M", () => {
  const parsed = parsePdb(samplePdbText());
  const structure = buildStructure(parsed);

  assert.equal(structure.coords.length, 3);
  assert.deepEqual(structure.residue_index, [10, 12, 1]);
  assert.deepEqual(structure.chain_ids, ["A", "A", "B"]);
  assert.deepEqual(structure.chain_index, [0, 0, 1]);
  assert.deepEqual(Array.from(structure.mask), [1, 1, 1]);

  // MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX" -> A=0, M=10, S=15
  assert.deepEqual(structure.native_tokens, [0, 10, 15]);

  // residue_keys / insertion_codes carry PDB numbering, including "B1A".
  assert.deepEqual(structure.residue_keys, [
    { chainId: "A", resSeq: 10, iCode: "" },
    { chainId: "A", resSeq: 12, iCode: "" },
    { chainId: "B", resSeq: 1, iCode: "A" },
  ]);
  assert.deepEqual(structure.insertion_codes, ["", "", "A"]);

  // Backbone coords: N, CA, C, O in that order, for the ALA residue.
  assert.deepEqual(structure.coords[0], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
});

test("buildStructure({chains}) restricts to the given chain ids", () => {
  const parsed = parsePdb(samplePdbText());
  const structure = buildStructure(parsed, { chains: ["B"] });
  assert.equal(structure.coords.length, 1);
  assert.deepEqual(structure.chain_ids, ["B"]);
});

test("listChains reports only chains with a designable (full-backbone) residue", () => {
  const parsed = parsePdb(samplePdbText());
  assert.deepEqual(listChains(parsed), ["A", "B"]);
});

test("writePdbWithBfactors round-trips through parsePdb with the requested B-factors", () => {
  const parsed = parsePdb(samplePdbText());
  const rewritten = writePdbWithBfactors(parsed, [
    { chainId: "A", resSeq: 10, iCode: "", value: 1.0 },
    { chainId: "A", resSeq: 12, iCode: "", value: 0.5 },
    { chainId: "B", resSeq: 1, iCode: "A", value: 0.25 },
  ]);
  assert.match(rewritten, /\nEND\n?$/);

  const reparsed = parsePdb(rewritten);
  const ala = reparsed.residues.find((r) => r.resName === "ALA");
  const mse = reparsed.residues.find((r) => r.resName === "MSE");
  const ser = reparsed.residues.find((r) => r.resName === "SER");
  assert.equal(ala.atoms.get("CA").bfactor, 100);
  assert.equal(mse.atoms.get("CA").bfactor, 50);
  assert.equal(ser.atoms.get("CA").bfactor, 25);

  // GLY (never in perResidueValues) still comes through with B-factor 0.
  const gly = reparsed.residues.find((r) => r.resName === "GLY");
  assert.equal(gly.atoms.get("CA").bfactor, 0);
});

test("writePdbWithBfactors clamps values outside [0, 1]", () => {
  const parsed = parsePdb(samplePdbText());
  const rewritten = writePdbWithBfactors(parsed, [
    { chainId: "A", resSeq: 10, iCode: "", value: 5 },
    { chainId: "A", resSeq: 12, iCode: "", value: -5 },
  ]);
  const reparsed = parsePdb(rewritten);
  assert.equal(reparsed.residues.find((r) => r.resName === "ALA").atoms.get("CA").bfactor, 100);
  assert.equal(reparsed.residues.find((r) => r.resName === "MSE").atoms.get("CA").bfactor, 0);
});

function sampleMmcifText() {
  return `data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.type_symbol
_atom_site.pdbx_PDB_model_num
ATOM 1 N   ALA A N   ALA A 5 ? 0.000 0.000 0.000 10.00 N 1
ATOM 2 CA  ALA A CA  ALA A 5 ? 1.000 0.000 0.000 20.00 C 1
ATOM 3 C   ALA A C   ALA A 5 ? 2.000 0.000 0.000 30.00 C 1
ATOM 4 O   ALA A O   ALA A 5 ? 3.000 0.000 0.000 40.00 O 1
ATOM 5 N   VAL A N   VAL A 6 ? 4.000 0.000 0.000 10.00 N 2
ATOM 6 CA  VAL A CA  VAL A 6 ? 5.000 0.000 0.000 20.00 C 2
#
`;
}

test("parseMmcif reads the atom_site loop, preferring auth_* fields, first model only", () => {
  const parsed = parseMmcif(sampleMmcifText());
  assert.equal(parsed.format, "mmcif");
  assert.equal(parsed.residues.length, 1); // model 2's VAL is excluded
  const [residue] = parsed.residues;
  assert.equal(residue.resName, "ALA");
  assert.equal(residue.chainId, "A");
  assert.equal(residue.resSeq, 5);
  assert.deepEqual(residue.atoms.get("CA"), { x: 1, y: 0, z: 0, element: "C", bfactor: 20 });
});

test("parseMmcif -> buildStructure produces a valid single-residue structure", () => {
  const parsed = parseMmcif(sampleMmcifText());
  const structure = buildStructure(parsed);
  assert.equal(structure.coords.length, 1);
  assert.equal(structure.residue_index[0], 5);
  assert.equal(structure.chain_ids[0], "A");
});
