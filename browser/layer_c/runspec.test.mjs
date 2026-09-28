import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MPNN_ALPHABET,
  badGumbelFromUniform,
  biasedShuffle,
  buildP07Inputs,
  gumbelFromUniform,
  makePrng,
  shuffle,
  tieGroupMap,
  toFloat32,
  uniformFromU32,
} from "./runspec.mjs";

function structure(length = 6) {
  const coords = [];
  for (let i = 0; i < length; i += 1) {
    coords.push([
      [i, 0, 0],
      [i, 1, 0],
      [i, 1, 1],
      [i, 0, 1],
    ]);
  }
  return {
    coords,
    mask: Array.from({ length }, () => 1),
    residue_index: Array.from({ length }, (_, i) => i),
    chain_index: Array.from({ length }, () => 0),
    chain_ids: Array.from({ length }, (_, i) => (i < length / 2 ? "A" : "B")),
    native_tokens: Array.from({ length }, (_, i) => i % 20),
  };
}

test("alphabet is the aminx MPNN order", () => {
  assert.equal(MPNN_ALPHABET, "ACDEFGHIKLMNPQRSTVWYX");
  assert.equal(MPNN_ALPHABET.length, 21);
});

test("gumbelFromUniform is finite float32 and differs from -log(u)", () => {
  const u = uniformFromU32(0x12345678);
  const g = gumbelFromUniform(u);
  const bad = badGumbelFromUniform(u);
  assert.equal(typeof g, "number");
  assert.ok(Number.isFinite(g));
  assert.notEqual(g, bad);
  assert.equal(g, toFloat32(g));
  assert.ok(u > 0 && u < 1);
});

test("prng and shuffle are deterministic permutations", () => {
  const a = shuffle(8, makePrng(99));
  const b = shuffle(8, makePrng(99));
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.equal(new Set(a).size, 8);
  const other = shuffle(8, makePrng(100));
  assert.notDeepEqual(Array.from(a), Array.from(other));
});

test("biased shuffle is a different stream from Fisher-Yates", () => {
  const fair = shuffle(5, makePrng(7));
  const biased = biasedShuffle(5, makePrng(7));
  assert.notDeepEqual(Array.from(fair), Array.from(biased));
});

test("default temperature is 0.1 and explicit order is kept", () => {
  const built = buildP07Inputs(structure(), { seed: 3, decoding_order: [5, 4, 3, 2, 1, 0] });
  assert.equal(built.temperature.data[0], toFloat32(0.1));
  assert.deepEqual(built.decoding_order.data, [5, 4, 3, 2, 1, 0]);
  assert.equal(built.temperature.shape.length, 0);
});

test("bias, omit, fixed, chains, and ties map onto the P07 tensors", () => {
  const built = buildP07Inputs(structure(), {
    seed: 11,
    temperature: 1,
    bias_AA: { A: 1.5 },
    bias_AA_per_residue: { 2: { G: 0.25 } },
    omit_AA: "CX",
    omit_AA_per_residue: { 1: "W" },
    fixed_positions: { 0: "A" },
    chains_to_design: ["A"],
    tied_positions: [[4, 1, 3]],
    decoding_order: "random",
  });
  const omit = toFloat32(-1e8);
  assert.equal(built.bias.data[0], toFloat32(1.5));
  assert.equal(built.bias.data[MPNN_ALPHABET.indexOf("C")], omit);
  assert.equal(built.bias.data[MPNN_ALPHABET.indexOf("X")], omit);
  assert.equal(built.bias.data[1 * 21 + MPNN_ALPHABET.indexOf("W")], omit);
  assert.equal(built.bias.data[2 * 21 + MPNN_ALPHABET.indexOf("G")], toFloat32(0.25));
  assert.equal(built.fixed_mask.data[0], 1);
  assert.equal(built.fixed_tokens.data[0], 0);
  assert.equal(built.fixed_mask.data[4], 1);
  assert.equal(built.fixed_tokens.data[4], 4);
  assert.equal(built.fixed_mask.data[1], 0);
  const ties = tieGroupMap(6, [[4, 1, 3]]);
  assert.deepEqual(Array.from(built.tie_group_map.data), Array.from(ties));
  assert.ok(Math.min(...ties) >= 0);
  assert.ok(Math.max(...ties) < 6);
  assert.equal(ties[4], 1);
  assert.equal(ties[1], 1);
});

test("omit overrides an additive bias on the same letter", () => {
  const built = buildP07Inputs(structure(2), {
    seed: 1,
    bias_AA: { C: 4 },
    omit_AA: "C",
    decoding_order: [0, 1],
  });
  assert.equal(built.bias.data[MPNN_ALPHABET.indexOf("C")], toFloat32(-1e8));
});

function assertGroup(order, first, second) {
  const head = [...order.slice(0, first.length)].sort((a, b) => a - b);
  const tail = [...order.slice(first.length)].sort((a, b) => a - b);
  const expectHead = [...first].sort((a, b) => a - b);
  const expectTail = [...second].sort((a, b) => a - b);
  assert.deepEqual(head, expectHead, `order ${[...order]} head ${head}`);
  assert.deepEqual(tail, expectTail, `order ${[...order]} tail ${tail}`);
}

test("random order decodes fixed, padding, and undesigned chains first", () => {
  const n = 200;
  for (let seed = 0; seed < n; seed += 1) {
    const fixed = buildP07Inputs(structure(), {
      seed,
      fixed_positions: { 0: "A", 4: "G" },
    });
    assertGroup(fixed.decoding_order.data, [0, 4], [1, 2, 3, 5]);

    const padded = structure();
    padded.mask = [1, 1, 1, 1, 1, 0];
    const padBuilt = buildP07Inputs(padded, { seed });
    assertGroup(padBuilt.decoding_order.data, [5], [0, 1, 2, 3, 4]);

    const chains = buildP07Inputs(structure(), {
      seed,
      chains_to_design: ["A"],
    });
    assertGroup(chains.decoding_order.data, [3, 4, 5], [0, 1, 2]);

    const tied = buildP07Inputs(structure(4), {
      seed,
      fixed_positions: { 0: "A", 2: "C" },
      tied_positions: [[2, 1]],
    });
    assert.equal(
      tied.decoding_order.data[0],
      0,
      `tied seed ${seed} order ${tied.decoding_order.data}`,
    );
  }
});

test("designed-group order is uniform for a two-position group", () => {
  const n = 4000;
  let leading = 0;
  for (let seed = 0; seed < n; seed += 1) {
    const built = buildP07Inputs(structure(4), {
      seed,
      fixed_positions: { 0: "A", 1: "C" },
    });
    const order = built.decoding_order.data;
    assertGroup(order, [0, 1], [2, 3]);
    if (order[2] === 2) leading += 1;
  }
  assert.ok(leading > 1800 && leading < 2200, `designed-pair count ${leading} of ${n}`);
});

test("explicit decoding order is unchanged when positions are fixed", () => {
  const order = [5, 4, 3, 2, 1, 0];
  const built = buildP07Inputs(structure(), {
    seed: 3,
    fixed_positions: { 0: "A", 1: "C" },
    chains_to_design: ["A"],
    decoding_order: order,
  });
  assert.deepEqual(built.decoding_order.data, order);
});

test("the same seed rebuilds identical noise", () => {
  const spec = { seed: 42, temperature: 0.2 };
  const a = buildP07Inputs(structure(), spec);
  const b = buildP07Inputs(structure(), spec);
  assert.deepEqual(a.gumbel_noise.data, b.gumbel_noise.data);
  assert.deepEqual(a.decoding_order.data, b.decoding_order.data);
});
