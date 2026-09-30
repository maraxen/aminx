import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BUCKETS,
  P07_INPUT_ORDER,
  buildInputs,
  padStructure,
  pickBucket,
} from "./aminx_sampler.mjs";
import { buildP07Inputs } from "./runspec_core.mjs";

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
    chain_ids: Array.from({ length }, () => "A"),
    native_tokens: Array.from({ length }, (_, i) => i % 20),
  };
}

test("pickBucket picks the smallest fitting bucket and refuses overflow", () => {
  assert.equal(pickBucket(1), 128);
  assert.equal(pickBucket(128), 128);
  assert.equal(pickBucket(129), 256);
  assert.equal(pickBucket(256), 256);
  assert.throws(() => pickBucket(257), /exceeds/);
  assert.deepEqual(BUCKETS, [128, 256]);
});

test("padStructure mirrors buckets.py pad_inputs: zero coords/mask, repeat-last index/chain_id", () => {
  const s = structure(3);
  s.residue_index = [10, 11, 12];
  s.chain_index = [0, 0, 1];
  s.chain_ids = ["A", "A", "B"];
  const padded = padStructure(s, 6);

  assert.deepEqual(Array.from(padded.mask), [1, 1, 1, 0, 0, 0]);
  assert.deepEqual(Array.from(padded.residue_index), [10, 11, 12, 12, 12, 12]);
  assert.deepEqual(Array.from(padded.chain_index), [0, 0, 1, 1, 1, 1]);
  assert.deepEqual(padded.chain_ids, ["A", "A", "B", "B", "B", "B"]);
  assert.deepEqual(Array.from(padded.native_tokens), [0, 1, 2, 0, 0, 0]);
  // padded coords rows are all zero
  for (let i = 3 * 4 * 3; i < 6 * 4 * 3; i += 1) {
    assert.equal(padded.coords[i], 0);
  }
  assert.deepEqual(padded.coords_shape, [6, 4, 3]);
});

test("padStructure is a no-op when already at the bucket length", () => {
  const s = structure(4);
  assert.equal(padStructure(s, 4), s);
});

test("buildInputs returns live typed arrays, not Array.from copies, matching buildP07Inputs numerically", () => {
  const s = structure(4);
  const runspec = { seed: 7, temperature: 0.3 };
  const typed = buildInputs(s, runspec);
  const plain = buildP07Inputs(s, runspec);

  for (const name of P07_INPUT_ORDER) {
    assert.ok(
      typed[name].data instanceof Float32Array || typed[name].data instanceof Int32Array,
      `${name}.data should be a typed array`,
    );
    assert.deepEqual(Array.from(typed[name].data), plain[name].data, `${name} mismatch`);
    assert.deepEqual(typed[name].shape, plain[name].shape);
  }
});

test("pickBucket + padStructure + buildInputs compose for an unpadded structure", () => {
  const s = structure(5);
  const bucket = pickBucket(s.coords.length);
  const padded = padStructure(s, bucket);
  const inputs = buildInputs(padded, { seed: 1 });
  assert.equal(inputs.mask.shape[0], bucket);
  assert.equal(inputs.coords.data.length, bucket * 4 * 3);
});
