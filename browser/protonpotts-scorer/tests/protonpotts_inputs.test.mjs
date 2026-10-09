import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  N_TOKENS,
  ProtonPottsInputError,
  VOCABULARY,
  buildProtonPottsInputs,
  encodeSequence,
  loadedResidues,
  pickBucket,
} from "../protonpotts_inputs.mjs";

function record({ rec = "ATOM  ", serial, name, res, chain, num, icode = " ", xyz, occ = 1.0 }) {
  const field = (value) => value.toFixed(3).padStart(8);
  return (
    rec.padEnd(6) + String(serial).padStart(5) + " " + name + " " + res.padStart(3) + " " + chain +
    String(num).padStart(4) + icode + "   " + field(xyz[0]) + field(xyz[1]) + field(xyz[2]) +
    occ.toFixed(2).padStart(6) + "  0.00           " + name.trim().slice(0, 1)
  );
}

function backbone({ start, res, chain, num, base, occ = 1.0 }) {
  return ["N", "CA", "C", "O"].map((atom, i) =>
    record({ serial: start + i, name: ` ${atom.padEnd(2)} `, res, chain, num, xyz: [base, i, 0], occ }),
  );
}

const SYNTHETIC = [
  ...backbone({ start: 1, res: "HIS", chain: "A", num: 1, base: 1 }),
  ...backbone({ start: 5, res: "ALA", chain: "A", num: 2, base: 5 }),
  ...backbone({ start: 9, res: "GLY", chain: "A", num: 3, base: 9, occ: 0.7 }), // dropped: backbone <= 0.8
  ...backbone({ start: 13, res: "ASP", chain: "B", num: 1, base: 13 }),
].join("\n");

test("vocabulary is the 30-token v6 alphabet", () => {
  assert.equal(N_TOKENS, 30);
  assert.equal(VOCABULARY[20], "X");
  assert.deepEqual(VOCABULARY.slice(21), ["HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A", "GLU-P", "GLU-D", "GLU-A"]);
});

test("encodeSequence: names, labels, X for UNK, label wins", () => {
  assert.deepEqual(encodeSequence(["ALA", "UNK", "HIS"]), [0, 20, 6]);
  assert.deepEqual(encodeSequence(["HIS", "ASP"], ["HIS-A", ""]), [23, 2]);
  assert.throws(() => encodeSequence(["HIS"], ["HID"]), ProtonPottsInputError);
  assert.throws(() => encodeSequence(["HIS"], ["ASP-P"]), ProtonPottsInputError);
  assert.throws(() => encodeSequence(["HIS", "ALA"], ["HIS-P"]), ProtonPottsInputError);
});

test("synthetic: backbone drop, R_idx counted before the drop, chain order, padding", () => {
  const built = buildProtonPottsInputs(SYNTHETIC, 8, ["HIS-S", "", "", "ASP-D"]);
  assert.equal(built.lTotal, 3);
  assert.deepEqual(built.tokens, [22, 0, 25]);
  assert.deepEqual(Array.from(built.inputs.residue_idx.data), [0, 1, 0, -100, -100, -100, -100, -100]);
  assert.deepEqual(Array.from(built.inputs.chain_index.data), [0, 0, 1, 0, 0, 0, 0, 0]);
  assert.deepEqual(Array.from(built.inputs.present.data), [1, 1, 1, 0, 0, 0, 0, 0]);
  assert.deepEqual(Array.from(built.inputs.pad_valid.data), [1, 1, 1, 0, 0, 0, 0, 0]);
  assert.equal(built.loaded.length, 4);
  assert.equal(built.kept.length, 3);
  assert.throws(() => buildProtonPottsInputs(SYNTHETIC, 2), ProtonPottsInputError);
});

test("refusals: MSE, blank chain, nonstandard ATOM residue, wrong label count", () => {
  const mse = record({ rec: "HETATM", serial: 1, name: " N  ", res: "MSE", chain: "A", num: 1, xyz: [0, 0, 0] });
  assert.throws(() => loadedResidues(mse), ProtonPottsInputError);
  const blank = record({ serial: 1, name: " N  ", res: "ALA", chain: " ", num: 1, xyz: [0, 0, 0] });
  assert.throws(() => loadedResidues(blank), ProtonPottsInputError);
  const odd = record({ serial: 1, name: " N  ", res: "SEP", chain: "A", num: 1, xyz: [0, 0, 0] });
  assert.throws(() => loadedResidues(odd), ProtonPottsInputError);
  assert.throws(() => buildProtonPottsInputs(SYNTHETIC, 8, ["", ""]), ProtonPottsInputError);
});

test("keep-last rule: only the last residue NAME at one number survives", () => {
  const text = [
    ...backbone({ start: 1, res: "ALA", chain: "A", num: 42, base: 1 }),
    ...backbone({ start: 5, res: "GLN", chain: "A", num: 42, base: 5 }),
  ].join("\n");
  assert.deepEqual(loadedResidues(text).map((r) => r[3]), ["GLN"]);
});

test("pickBucket", () => {
  assert.equal(pickBucket(153), 256);
  assert.equal(pickBucket(789), 1024);
  assert.equal(pickBucket(2000), null);
});

// ---- fixture comparison against the Python dumps (POTTS... set by the gate; skipped otherwise) ----
const FIXTURE_DIR = process.env.PROTONPOTTS_INPUTS_FIXTURE_DIR;
const fixtures = FIXTURE_DIR && existsSync(FIXTURE_DIR)
  ? readdirSync(FIXTURE_DIR).filter((f) => f.endsWith(".json")).sort()
  : [];

function compareToFixture(fixture, pdbText) {
  const built = buildProtonPottsInputs(pdbText, fixture.bucket, fixture.labels);
  const problems = [];
  for (const [name, spec] of Object.entries(fixture.inputs)) {
    const got = built.inputs[name];
    if (got.dtype !== spec.dtype) problems.push(`${name}: dtype ${got.dtype} != ${spec.dtype}`);
    if (JSON.stringify(got.dims) !== JSON.stringify(spec.dims)) problems.push(`${name}: dims`);
    const data = Array.from(got.data);
    if (data.length !== spec.data.length || data.some((v, i) => v !== spec.data[i])) problems.push(`${name}: data`);
  }
  if (JSON.stringify(built.tokens) !== JSON.stringify(fixture.tokens)) problems.push("tokens");
  if (JSON.stringify(built.kept) !== JSON.stringify(fixture.kept)) problems.push("kept");
  if (JSON.stringify(built.loaded) !== JSON.stringify(fixture.loaded)) problems.push("loaded");
  return problems;
}

test("fixtures: at least one Python dump was supplied", { skip: FIXTURE_DIR === undefined }, () => {
  assert.ok(fixtures.length > 0, `no fixtures in ${FIXTURE_DIR}`);
});

for (const file of fixtures) {
  test(`bit-exact against the Python path: ${file}`, () => {
    const fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, file), "utf8"));
    const pdbText = readFileSync(join(FIXTURE_DIR, fixture.pdb), "utf8");
    assert.deepEqual(compareToFixture(fixture, pdbText), []);
  });
}

if (fixtures.length > 0) {
  test("negative control: a corrupted residue_idx entry is caught", () => {
    const fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, fixtures[0]), "utf8"));
    const pdbText = readFileSync(join(FIXTURE_DIR, fixture.pdb), "utf8");
    fixture.inputs.residue_idx.data[3] += 1;
    assert.ok(compareToFixture(fixture, pdbText).includes("residue_idx: data"));
  });
  test("negative control: a wrong label changes the tokens", () => {
    const fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, fixtures[0]), "utf8"));
    const pdbText = readFileSync(join(FIXTURE_DIR, fixture.pdb), "utf8");
    fixture.tokens[0] = (fixture.tokens[0] + 1) % 30;
    assert.ok(compareToFixture(fixture, pdbText).includes("tokens"));
  });
}
