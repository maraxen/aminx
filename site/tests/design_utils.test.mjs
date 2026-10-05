import assert from "node:assert/strict";
import { test } from "node:test";

import {
  designedMask,
  parseBias,
  parseFixedPositions,
  parseTiedPositions,
  recovery,
  scoreDesign,
  toFasta,
} from "../design_utils.mjs";

function sampleStructure() {
  // 3 positions: A10 (native A=0), A11 (native M=11), B1A (native S=15,
  // insertion code "A") -- mirrors the pdb_parse fixture.
  return {
    mask: [1, 1, 1],
    chain_ids: ["A", "A", "B"],
    native_tokens: [0, 10, 15],
    residue_keys: [
      { chainId: "A", resSeq: 10, iCode: "" },
      { chainId: "A", resSeq: 11, iCode: "" },
      { chainId: "B", resSeq: 1, iCode: "A" },
    ],
  };
}

test("parseFixedPositions resolves PDB numbering (incl. insertion codes) to native residue letters", () => {
  const structure = sampleStructure();
  const result = parseFixedPositions("A10,B1A", structure);
  assert.deepEqual(result, { 0: "A", 2: "S" });
});

test("parseFixedPositions returns {} for empty input", () => {
  assert.deepEqual(parseFixedPositions("", sampleStructure()), {});
  assert.deepEqual(parseFixedPositions("   ", sampleStructure()), {});
});

test("parseFixedPositions throws on an unresolvable position", () => {
  assert.throws(() => parseFixedPositions("A999", sampleStructure()), /not found/);
});

test("parseFixedPositions throws on a malformed token", () => {
  assert.throws(() => parseFixedPositions("not-a-position", sampleStructure()), /cannot parse/);
});

test("parseBias parses LETTER:value pairs, upper-casing the letter", () => {
  assert.deepEqual(parseBias("A:-1,w:0.5"), { A: -1, W: 0.5 });
  assert.deepEqual(parseBias(""), {});
});

test("parseBias throws on a non-numeric value", () => {
  assert.throws(() => parseBias("A:abc"), /not a number/);
});

test("parseTiedPositions resolves '=' groups separated by ';'", () => {
  const structure = sampleStructure();
  const groups = parseTiedPositions("A10=A11; B1A=A10", structure);
  assert.deepEqual(groups, [[0, 1], [2, 0]]);
});

test("parseTiedPositions ignores single-member groups and empty input", () => {
  assert.deepEqual(parseTiedPositions("", sampleStructure()), []);
  assert.deepEqual(parseTiedPositions("A10", sampleStructure()), []);
});

test("designedMask excludes fixed positions and non-designed chains", () => {
  const structure = sampleStructure();
  const maskAll = designedMask(structure, {});
  assert.deepEqual(Array.from(maskAll), [1, 1, 1]);

  const maskFixed = designedMask(structure, { fixed_positions: { 1: "M" } });
  assert.deepEqual(Array.from(maskFixed), [1, 0, 1]);

  const maskChain = designedMask(structure, { chains_to_design: ["A"] });
  assert.deepEqual(Array.from(maskChain), [1, 1, 0]);
});

test("scoreDesign is the mean negative log-prob over designed positions only", () => {
  // 2 positions, alphabet width 21. Position 0 designed, token 0,
  // logProbs[0*21+0] = -1 (so -logp = 1). Position 1 NOT designed, its
  // logProbs must be ignored even though present.
  const tokens = [0, 5];
  const logProbs = new Array(2 * 21).fill(-9); // sentinel; position 1 must be ignored
  logProbs[0 * 21 + 0] = -1;
  const mask = [1, 0];
  assert.equal(scoreDesign(tokens, logProbs, mask, 2), 1);
});

test("scoreDesign averages over multiple designed positions", () => {
  const tokens = [0, 1];
  const logProbs = new Array(2 * 21).fill(0);
  logProbs[0 * 21 + 0] = -2; // -logp = 2
  logProbs[1 * 21 + 1] = -4; // -logp = 4
  const mask = [1, 1];
  assert.equal(scoreDesign(tokens, logProbs, mask, 2), 3); // mean(2, 4)
});

test("scoreDesign returns 0 when nothing is designed", () => {
  assert.equal(scoreDesign([0], [0], [0], 1), 0);
});

test("recovery is the fraction identical to native over designed positions only", () => {
  const tokens = [0, 5, 2];
  const native = [0, 6, 2];
  const mask = [1, 1, 0]; // position 2 differs but is not designed -> ignored
  assert.equal(recovery(tokens, native, mask), 0.5); // 1 of 2 designed match
});

test("recovery returns 0 when nothing is designed", () => {
  assert.equal(recovery([0], [0], [0]), 0);
});

test("toFasta renders ProteinMPNN-style headers", () => {
  const designs = [
    { index: 0, temperature: 0.1, seed: 42, score: 1.23456, recovery: 0.5, sequence: "AC/GH" },
    { index: 1, temperature: 0.1, seed: 43, score: 0.5, recovery: 1, sequence: "AC/GH" },
  ];
  const fasta = toFasta(designs);
  const lines = fasta.trimEnd().split("\n");
  assert.equal(lines.length, 4);
  assert.equal(lines[0], ">design_0, T=0.1, seed=42, score=1.2346, recovery=0.5000");
  assert.equal(lines[1], "AC/GH");
  assert.equal(lines[2], ">design_1, T=0.1, seed=43, score=0.5000, recovery=1.0000");
  assert.equal(lines[3], "AC/GH");
});

test("toFasta returns empty string for no designs", () => {
  assert.equal(toFasta([]), "");
});
