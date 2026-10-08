import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { PottsInputError, buildPottsInputs, pickPottsBucket } from "../potts_inputs.mjs";

/** One fixed-column PDB ATOM/HETATM record (PDB v3 columns). */
function record({ rec = "ATOM  ", serial, name, alt = " ", res, chain, num, icode = " ", xyz }) {
  const field = (value) => value.toFixed(3).padStart(8);
  return (
    rec.padEnd(6) +
    String(serial).padStart(5) +
    " " +
    name +
    alt +
    res.padStart(3) +
    " " +
    chain +
    String(num).padStart(4) +
    icode +
    "   " +
    field(xyz[0]) +
    field(xyz[1]) +
    field(xyz[2]) +
    "  1.00  0.00           " +
    name.trim().slice(0, 1)
  );
}

/** Backbone N, CA, C, O at (base,0,0), (base,1,0), (base,2,0), (base,3,0). */
function backbone({ startSerial, res, chain, num, icode = " ", base, atoms = ["N", "CA", "C", "O"] }) {
  return atoms.map((atom, index) =>
    record({
      serial: startSerial + index,
      name: ` ${atom.padEnd(2)} `,
      res,
      chain,
      num,
      icode,
      xyz: [base, index, 0],
    }),
  );
}

// Synthetic 2-chain structure, chain B written FIRST so chain order must come from sorting.
//   chain A: residues 1 ALA, 2 GLY (O missing), 3 SER, [4 absent], 5 LEU  -> 5 rows, one gap
//   chain B: residues 1 VAL (CA altloc A then B), 2 UNK (nonstandard), 3 MET; HETATM HEM dropped
const SYNTHETIC_PDB = [
  record({ serial: 1, name: " CA ", alt: "A", res: "VAL", chain: "B", num: 1, xyz: [5, 5, 5] }),
  record({ serial: 2, name: " CA ", alt: "B", res: "VAL", chain: "B", num: 1, xyz: [9, 9, 9] }),
  record({ serial: 3, name: " N  ", res: "VAL", chain: "B", num: 1, xyz: [1, 0, 0] }),
  record({ serial: 4, name: " C  ", res: "VAL", chain: "B", num: 1, xyz: [2, 0, 0] }),
  record({ serial: 5, name: " O  ", res: "VAL", chain: "B", num: 1, xyz: [3, 0, 0] }),
  record({ serial: 6, name: " N  ", res: "UNK", chain: "B", num: 2, xyz: [0.1, 0, 0] }),
  record({ serial: 7, name: " CA ", res: "UNK", chain: "B", num: 2, xyz: [0, 1, 0] }),
  record({ serial: 8, name: " C  ", res: "UNK", chain: "B", num: 2, xyz: [0, 2, 0] }),
  record({ serial: 9, name: " O  ", res: "UNK", chain: "B", num: 2, xyz: [0, 3, 0] }),
  ...backbone({ startSerial: 10, res: "MET", chain: "B", num: 3, base: 70 }),
  record({ rec: "HETATM", serial: 14, name: " C1 ", res: "HEM", chain: "B", num: 4, xyz: [7, 7, 7] }),
  ...backbone({ startSerial: 15, res: "ALA", chain: "A", num: 1, base: 10 }),
  ...backbone({ startSerial: 19, res: "GLY", chain: "A", num: 2, base: 20, atoms: ["N", "CA", "C"] }),
  ...backbone({ startSerial: 22, res: "SER", chain: "A", num: 3, base: 30 }),
  ...backbone({ startSerial: 26, res: "LEU", chain: "A", num: 5, base: 50 }),
].join("\n");

const BUCKET = 16;
const B = BUCKET;
const M = 21;

function pad(values, length, fill) {
  return [...values, ...Array(length - values.length).fill(fill)];
}

function repeat(value, count) {
  return Array(count).fill(value);
}

test("synthetic 2-chain PDB: residue_idx, chain_index, s_true, present, pad rows", () => {
  const out = buildPottsInputs(SYNTHETIC_PDB, BUCKET);
  const { inputs } = out;

  assert.equal(out.l_total, 8);
  assert.equal(out.bucket, BUCKET);
  assert.deepEqual(out.chains, ["A", "B"]);
  // "-" marks the gap row and the UNK residue (aminx's native string, not the X form).
  assert.equal(out.sequence, "AGS-LV-M");

  // Chain A (chain_number 1): cursor 0..4. Chain B (chain_number 2): 100 + cursor 5..7.
  assert.deepEqual(
    [...inputs.residue_idx.data],
    pad([0, 1, 2, 3, 4, 105, 106, 107], B, -100),
  );
  assert.deepEqual(inputs.residue_idx.dtype, "int32");
  // chain_encoding is 1-based; pad rows are 0.
  assert.deepEqual([...inputs.chain_index.data], pad([1, 1, 1, 1, 1, 2, 2, 2], B, 0));
  // A:ALA=0 G=5 S=15 gap->X=20 L=9; B:V=17 UNK->X=20 M=10; pad X=20.
  assert.deepEqual([...inputs.s_true.data], pad([0, 5, 15, 20, 9, 17, 20, 10], B, 20));
  // GLY missing O and the gap row are not present; UNK has all four atoms and is present.
  assert.deepEqual([...inputs.present.data], pad([1, 0, 1, 0, 1, 1, 1, 1], B, 0));
  assert.deepEqual(
    [...inputs.pad_valid.data],
    pad(repeat(1, 8), B, 0),
  );
  assert.equal(inputs.pad_valid.dtype, "bool");
  assert.deepEqual([...inputs.chain_mask.data], pad(repeat(1, 8), B, 0));
  assert.deepEqual([...inputs.chain_m_pos.data], pad(repeat(1, 8), B, 0));
  // Untied: one singleton per real row, then -1 pad; M = 1.
  assert.deepEqual(inputs.tie_groups.dims, [B, 1]);
  assert.deepEqual([...inputs.tie_groups.data], pad([0, 1, 2, 3, 4, 5, 6, 7], B, -1));
  assert.deepEqual([...inputs.tied_beta.data], repeat(1, B));
  assert.deepEqual([...inputs.omit.data], repeat(0, M));
  assert.deepEqual([...inputs.bias.data], repeat(0, M));
  assert.deepEqual([...inputs.bias_by_res.data], repeat(0, B * M));
  assert.deepEqual([...inputs.pssm_coef.data], repeat(0, B));
  assert.deepEqual([...inputs.pssm_bias.data], repeat(0, B * M));
  // No pssm: log_odds default 10000 > threshold 0 on real rows; pad log_odds 0 -> 0.
  assert.deepEqual([...inputs.pssm_log_odds_mask.data], pad(repeat(1, 8 * M), B * M, 0));
  assert.deepEqual([...inputs.omit_aa_mask.data], repeat(0, B * M));
});

test("synthetic 2-chain PDB: coords float32, altloc first-wins, missing O and gap zero-filled", () => {
  const { inputs } = buildPottsInputs(SYNTHETIC_PDB, BUCKET);
  const coords = inputs.coords;
  assert.deepEqual(coords.dims, [B, 4, 3]);
  assert.ok(coords.data instanceof Float32Array);

  const rows = [
    [10, 0, 0, 10, 1, 0, 10, 2, 0, 10, 3, 0], // A1 ALA
    [20, 0, 0, 20, 1, 0, 20, 2, 0, 0, 0, 0], // A2 GLY, O absent -> 0 (NaN replaced)
    [30, 0, 0, 30, 1, 0, 30, 2, 0, 30, 3, 0], // A3 SER
    repeat(0, 12), // gap row, all NaN -> 0
    [50, 0, 0, 50, 1, 0, 50, 2, 0, 50, 3, 0], // A5 LEU
    [1, 0, 0, 5, 5, 5, 2, 0, 0, 3, 0, 0], // B1 VAL, CA altloc A (first) wins over B
    [Math.fround(0.1), 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0], // B2 UNK, 0.1 rounded to float32
    [70, 0, 0, 70, 1, 0, 70, 2, 0, 70, 3, 0], // B3 MET
  ];
  const expected = new Float32Array(B * 12);
  rows.flat().forEach((value, index) => {
    expected[index] = value;
  });
  assert.deepEqual([...coords.data], [...expected]);
  assert.equal(coords.data[5 * 12 + 3], 5); // B1 CA x = 5 (altloc A line)
  assert.equal(coords.data[5 * 12 + 4], 5); // B1 CA y = 5
  assert.equal(coords.data[5 * 12 + 5], 5); // B1 CA z = 5
  assert.equal(coords.data[6 * 12], Math.fround(0.1));
  assert.notEqual(coords.data[6 * 12], 0.1);
});

test("nonstandard residue, HETATM filtering, MSE, insertion codes, blank chain ID", () => {
  const pdb = [
    // chain C: residue 1 is HETATM MSE (-> MET), residue 2 has insertion code, residue 2A
    // is ALA, so icode "" (GLY) sorts before "A".
    ...backbone({ startSerial: 1, res: "MSE", chain: "C", num: 1, base: 1 }).map((line) =>
      line.replace("ATOM  ", "HETATM"),
    ),
    ...backbone({ startSerial: 5, res: "GLY", chain: "C", num: 2, base: 2 }),
    ...backbone({ startSerial: 9, res: "ALA", chain: "C", num: 2, icode: "A", base: 3 }),
    // residue 2 has icode "" (GLY) and icode "A" (ALA); "" sorts first, so GLY comes first
    // blank chain ID: dropped entirely
    record({ serial: 13, name: " CA ", res: "ALA", chain: " ", num: 9, xyz: [9, 9, 9] }),
  ].join("\n");
  const out = buildPottsInputs(pdb, 8);
  assert.equal(out.sequence, "MGA");
  assert.equal(out.l_total, 3);
  assert.deepEqual(out.chains, ["C"]);
  assert.deepEqual([...out.inputs.s_true.data], [10, 5, 0, 20, 20, 20, 20, 20]);
  assert.deepEqual([...out.inputs.present.data], [1, 1, 1, 0, 0, 0, 0, 0]);
  assert.deepEqual([...out.inputs.residue_idx.data], [0, 1, 2, -100, -100, -100, -100, -100]);
});

test("pickPottsBucket returns the smallest fitting bucket, or null", () => {
  assert.equal(pickPottsBucket(8), 128);
  assert.equal(pickPottsBucket(128), 128);
  assert.equal(pickPottsBucket(129), 256);
  assert.equal(pickPottsBucket(256), 256);
  assert.equal(pickPottsBucket(257), null);
  assert.equal(pickPottsBucket(5, [16, 8]), 8);
});

test("bucket smaller than L_total and non-integer buckets are rejected", () => {
  assert.throws(() => buildPottsInputs(SYNTHETIC_PDB, 4), PottsInputError);
  assert.throws(() => buildPottsInputs(SYNTHETIC_PDB, 0), PottsInputError);
  assert.throws(() => buildPottsInputs(SYNTHETIC_PDB, 16.5), PottsInputError);
});

// ---- Fixture comparison against aminx's Python dumps (potts_dump_inputs.py) -----------
// Set POTTS_INPUTS_FIXTURE_DIR to a directory holding <pdb>_L<bucket>.json dumps and the
// matching <pdb>.pdb text. Skipped when the variable is absent.
const FIXTURE_DIR = process.env.POTTS_INPUTS_FIXTURE_DIR;
const FIXTURE_NAME = /^(.+)_L(\d+)\.json$/u;

function sha256(text) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function sameValue(actual, expected) {
  // Float dumps carry NaN as null. Anything else must match bit-for-bit (Object.is), so
  // -0 versus 0 also counts as a difference. Integers and bools use strict equality.
  if (expected === null) return Number.isNaN(actual);
  return Object.is(actual, expected);
}

if (FIXTURE_DIR === undefined) {
  test("fixture comparison against aminx Python dumps", {
    skip: "POTTS_INPUTS_FIXTURE_DIR not set; no Python dumps to compare against",
  }, () => {});
} else {
  test("fixture comparison against aminx Python dumps (exact, NaN-equal)", () => {
    assert.ok(existsSync(FIXTURE_DIR), `fixture dir missing: ${FIXTURE_DIR}`);
    const dumps = readdirSync(FIXTURE_DIR).filter((file) => FIXTURE_NAME.test(file)).sort();
    assert.ok(dumps.length > 0, `no <pdb>_L<bucket>.json dumps in ${FIXTURE_DIR}`);

    for (const file of dumps) {
      const [, stem, bucketText] = FIXTURE_NAME.exec(file);
      const bucket = Number(bucketText);
      const dump = JSON.parse(readFileSync(join(FIXTURE_DIR, file), "utf8"));
      const pdbPath = join(FIXTURE_DIR, `${stem}.pdb`);
      assert.ok(existsSync(pdbPath), `matching PDB missing for ${file}: ${pdbPath}`);
      const pdbText = readFileSync(pdbPath, "utf8");
      if (dump.pdb_sha256 !== undefined) {
        assert.equal(sha256(pdbText), dump.pdb_sha256, `${stem}.pdb does not match its dump`);
      }
      assert.equal(dump.bucket, bucket, `${file}: bucket field disagrees with file name`);

      const ours = buildPottsInputs(pdbText, bucket);
      const where = `${stem} L${bucket}`;
      assert.equal(ours.l_total, dump.l_total, `${where}: l_total`);
      assert.equal(ours.sequence, dump.sequence, `${where}: sequence`);
      assert.deepEqual(ours.chains, dump.chains, `${where}: chains`);
      assert.deepEqual(Object.keys(ours.inputs).sort(), Object.keys(dump.inputs).sort(), `${where}: input names`);

      for (const [name, want] of Object.entries(dump.inputs)) {
        const got = ours.inputs[name];
        assert.equal(got.dtype, want.dtype, `${where}/${name}: dtype`);
        assert.deepEqual(got.dims, want.dims, `${where}/${name}: dims`);
        assert.equal(got.data.length, want.data.length, `${where}/${name}: length`);
        const bad = [];
        for (let index = 0; index < want.data.length; index += 1) {
          if (!sameValue(got.data[index], want.data[index])) {
            bad.push({ index, got: got.data[index], want: want.data[index] });
            if (bad.length >= 5) break;
          }
        }
        assert.deepEqual(bad, [], `${where}/${name}: mismatched entries (first 5)`);
      }
    }
  });
}
