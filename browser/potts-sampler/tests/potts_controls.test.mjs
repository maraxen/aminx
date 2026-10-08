import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { MODEL_ALPHABET as INPUTS_ALPHABET, buildPottsInputs } from "../potts_inputs.mjs";
import { MODEL_ALPHABET, PottsControlsError, applyPottsControls } from "../potts_controls.mjs";

const N = 21;

// Synthetic built-inputs object with the shape buildPottsInputs() returns for a chain of
// lTotal real rows padded to `bucket`. Only the three inputs the controls touch are needed.
//   chain_m_pos: 1 on real rows, 0 on pad rows (as buildPottsInputs writes it).
//   bias_by_res: `byResFill` everywhere, bias: zeros.
function makeBuilt({ bucket, lTotal, byResFill = 0 }) {
  const chainMPos = new Float32Array(bucket);
  chainMPos.fill(1, 0, lTotal);
  return {
    l_total: lTotal,
    bucket,
    inputs: {
      coords: { dtype: "float32", dims: [bucket, 4, 3], data: new Float32Array(bucket * 12) },
      chain_m_pos: { dtype: "float32", dims: [bucket], data: chainMPos },
      omit: { dtype: "float32", dims: [N], data: new Float32Array(N) },
      bias: { dtype: "float32", dims: [N], data: new Float32Array(N) },
      bias_by_res: {
        dtype: "float32",
        dims: [bucket, N],
        data: new Float32Array(bucket * N).fill(byResFill),
      },
    },
  };
}

/** Float32 vector of length `n` with ones at `indices`. */
function oneHot(n, indices) {
  const out = new Float32Array(n);
  for (const index of indices) out[index] = 1;
  return out;
}

/** Float32 array (rows x N) where entry (r, a) is fn(r, a). */
function table(rows, fn) {
  const out = new Float32Array(rows * N);
  for (let r = 0; r < rows; r += 1) {
    for (let a = 0; a < N; a += 1) out[r * N + a] = fn(r, a);
  }
  return out;
}

function sameArray(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    if (!Object.is(actual[i], expected[i])) {
      assert.fail(`index ${i}: got ${actual[i]}, want ${expected[i]}`);
    }
  }
}

test("MODEL_ALPHABET matches the MPNN alphabet and potts_inputs.mjs", () => {
  assert.equal(MODEL_ALPHABET, "ACDEFGHIKLMNPQRSTVWYX");
  assert.equal(MODEL_ALPHABET.length, N);
  assert.equal(MODEL_ALPHABET, INPUTS_ALPHABET);
});

test("no controls: returns an equal but independent inputs object", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8, byResFill: 0.5 });
  const out = applyPottsControls(built, {});
  assert.deepEqual(Object.keys(out), Object.keys(built.inputs));
  for (const name of Object.keys(out)) {
    assert.equal(out[name].dtype, built.inputs[name].dtype);
    assert.deepEqual(out[name].dims, built.inputs[name].dims);
    sameArray(out[name].data, built.inputs[name].data);
    assert.notEqual(out[name].data, built.inputs[name].data, `${name} must not share its buffer`);
  }
  // null and undefined controls are the same as absent.
  const nulls = applyPottsControls(built, { fixed_positions: null, omit_aa: null, bias: null });
  sameArray(nulls.bias_by_res.data, built.inputs.bias_by_res.data);
  sameArray(nulls.chain_m_pos.data, built.inputs.chain_m_pos.data);
  sameArray(applyPottsControls(built).omit.data, new Float32Array(N));
});

test("fixed_positions zero chain_m_pos at in-bucket rows; out-of-range ignored", () => {
  // bucket 16, L 8: rows 0..7 real (chain_m_pos 1), rows 8..15 pad (0).
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  const out = applyPottsControls(built, { fixed_positions: [0, 3, 7, -1, 16, 100, 15] });
  // -1, 16, 100 are outside 0..bucket-1 and ignored. 15 is a pad row, already 0.
  sameArray(out.chain_m_pos.data, Float32Array.from([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  // The input is untouched.
  sameArray(built.inputs.chain_m_pos.data, Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]));
  // Nothing else changes.
  sameArray(out.omit.data, new Float32Array(N));
  sameArray(out.bias_by_res.data, new Float32Array(16 * N));
});

test("fixed_positions: empty list is a no-op; floats truncate like np.int32; NaN throws", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  sameArray(applyPottsControls(built, { fixed_positions: [] }).chain_m_pos.data, built.inputs.chain_m_pos.data);
  const trunc = applyPottsControls(built, { fixed_positions: [2.7] });
  assert.equal(trunc.chain_m_pos.data[2], 0);
  assert.equal(trunc.chain_m_pos.data[3], 1);
  assert.throws(() => applyPottsControls(built, { fixed_positions: [Number.NaN] }), PottsControlsError);
  assert.throws(() => applyPottsControls(built, { fixed_positions: 3 }), PottsControlsError);
  assert.throws(() => applyPottsControls(built, { fixed_positions: [[1]] }), PottsControlsError);
});

test("omit_aa sets omit[k]=1 for each letter's MODEL_ALPHABET index", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  // A=0, C=1, W=18
  sameArray(applyPottsControls(built, { omit_aa: "ACW" }).omit.data, oneHot(N, [0, 1, 18]));
  // X is a valid omit letter (index 20).
  sameArray(applyPottsControls(built, { omit_aa: "X" }).omit.data, oneHot(N, [20]));
  // repeated letters are idempotent; "" omits nothing.
  sameArray(applyPottsControls(built, { omit_aa: "AA" }).omit.data, oneHot(N, [0]));
  sameArray(applyPottsControls(built, { omit_aa: "" }).omit.data, new Float32Array(N));
  // an array of tokens is split into letters, as _letters does.
  sameArray(applyPottsControls(built, { omit_aa: ["CW"] }).omit.data, oneHot(N, [1, 18]));
  // omit_aa does not touch the per-position mask or chain_m_pos.
  const out = applyPottsControls(built, { omit_aa: "W" });
  sameArray(out.chain_m_pos.data, built.inputs.chain_m_pos.data);
});

test("omit_aa rejects letters outside the MPNN alphabet", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  assert.throws(() => applyPottsControls(built, { omit_aa: "Z" }), PottsControlsError);
  assert.throws(() => applyPottsControls(built, { omit_aa: "a" }), PottsControlsError);
  assert.throws(() => applyPottsControls(built, { omit_aa: "-" }), PottsControlsError);
  assert.throws(() => applyPottsControls(built, { omit_aa: 7 }), PottsControlsError);
});

test("bias length 21 is the global bias (zeros in bias_by_res), float32-rounded", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  const v = Array.from({ length: N }, (_, a) => a * 0.1);
  const out = applyPottsControls(built, { bias: v });
  sameArray(out.bias.data, Float32Array.from(v));
  assert.equal(out.bias.data[3], Math.fround(0.30000000000000004));
  sameArray(out.bias_by_res.data, new Float32Array(16 * N));
  assert.deepEqual(out.bias.dims, [N]);
});

test("bias length L is per-position, added to all 21 columns of rows [0, L_total) only", () => {
  // bucket 16, L 8. Rows 8..15 are pad and must not receive the bias.
  const built = makeBuilt({ bucket: 16, lTotal: 8, byResFill: 0.5 });
  const v = [1, 2, 3, 4, 5, 6, 7, 8];
  const out = applyPottsControls(built, { bias: v });
  sameArray(out.bias.data, new Float32Array(N));
  sameArray(
    out.bias_by_res.data,
    table(16, (r) => (r < 8 ? 0.5 + v[r] : 0.5)),
  );
});

test("bias (L, 21) is per-residue, added to rows [0, L_total)", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  const m = Array.from({ length: 8 }, (_, r) => Array.from({ length: N }, (_, a) => r * 100 + a));
  const out = applyPottsControls(built, { bias: m });
  sameArray(out.bias.data, new Float32Array(N));
  sameArray(out.bias_by_res.data, table(16, (r, a) => (r < 8 ? r * 100 + a : 0)));
});

test("L == 21: a 1-D vector is per-position, not global (per-position branch wins)", () => {
  // bucket 32, L 21. The 21-vector must land in bias_by_res rows 0..20, not in bias.
  const built = makeBuilt({ bucket: 32, lTotal: 21 });
  const v = Array.from({ length: N }, (_, r) => r + 1);
  const out = applyPottsControls(built, { bias: v });
  sameArray(out.bias.data, new Float32Array(N));
  sameArray(out.bias_by_res.data, table(32, (r) => (r < 21 ? r + 1 : 0)));
});

test("L == 21: a (21, 21) matrix is per-residue", () => {
  const built = makeBuilt({ bucket: 32, lTotal: 21 });
  const m = Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, a) => r * N + a));
  const out = applyPottsControls(built, { bias: m });
  sameArray(out.bias.data, new Float32Array(N));
  sameArray(out.bias_by_res.data, table(32, (r, a) => (r < 21 ? r * N + a : 0)));
});

test("bias adds onto existing bias_by_res values (not replace)", () => {
  const built = makeBuilt({ bucket: 8, lTotal: 4, byResFill: 2 });
  const out = applyPottsControls(built, { bias: [10, 20, 30, 40] });
  sameArray(out.bias_by_res.data, table(8, (r) => (r < 4 ? 2 + (r + 1) * 10 : 2)));
});

test("bad bias shapes and values throw", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  const bad = [
    3, // scalar
    "abc", // not an array
    [], // empty
    [1, 2, 3, 4, 5], // length neither 21 nor 8
    Array.from({ length: 8 }, () => Array(20).fill(0)), // (8, 20)
    Array.from({ length: 7 }, () => Array(N).fill(0)), // (7, 21)
    [[[1]]], // rank 3
    [Array(N).fill(0), Array(N - 1).fill(0)], // ragged rows
    [1, [2]], // mixed
    [Number.NaN, ...Array(7).fill(0)], // non-finite entry in a per-position vector
    Array.from({ length: 8 }, (_, r) => (r === 2 ? [Number.POSITIVE_INFINITY, ...Array(N - 1).fill(0)] : Array(N).fill(0))),
  ];
  for (const bias of bad) {
    assert.throws(() => applyPottsControls(built, { bias }), PottsControlsError, JSON.stringify(bias).slice(0, 60));
  }
  // A shape error names the expected shapes.
  assert.throws(
    () => applyPottsControls(built, { bias: [1, 2, 3, 4, 5] }),
    /bias shape \(5,\) is not \(21,\), \(8,\), or \(8, 21\)/u,
  );
});

test("a bias with L != 21 and L != 21-vector is rejected even when L_total < bucket", () => {
  // bucket 16 with L 8: a 16-vector (the bucket length) is NOT per-position.
  const built = makeBuilt({ bucket: 16, lTotal: 8 });
  assert.throws(() => applyPottsControls(built, { bias: Array(16).fill(0) }), PottsControlsError);
});

test("applyPottsControls never mutates its input", () => {
  const built = makeBuilt({ bucket: 16, lTotal: 8, byResFill: 0.25 });
  const snapshot = structuredClone(built);
  const out = applyPottsControls(built, {
    fixed_positions: [0, 1, 2],
    omit_aa: "ACDEFGHIKLMNPQRSTVWYX",
    bias: Array.from({ length: 8 }, (_, r) => r),
  });
  assert.deepEqual(built, snapshot);
  // The output does change, so the no-mutation check is not vacuous.
  assert.equal(out.chain_m_pos.data[0], 0);
  assert.equal(out.omit.data[20], 1);
  assert.equal(out.bias_by_res.data[7 * N], 0.25 + 7);
});

test("the built object passed without inputs is rejected", () => {
  assert.throws(() => applyPottsControls({}, {}), PottsControlsError);
});

// ---- Fixture comparison against aminx Python dumps (potts_dump_inputs.py --controls) ----
// Set POTTS_CONTROLS_FIXTURE_DIR to a directory holding dumps written by potts_dump_inputs.py
// that carry a "controls" field, plus the PDB named by each dump's "pdb" field. Skipped when
// the variable is unset.
const FIXTURE_DIR = process.env.POTTS_CONTROLS_FIXTURE_DIR;

function sameValue(actual, expected) {
  // NaN is dumped as null. Anything else must match bit-for-bit (Object.is).
  if (expected === null) return Number.isNaN(actual);
  return Object.is(actual, expected);
}

if (FIXTURE_DIR === undefined) {
  test("controls fixture comparison against aminx Python dumps", {
    skip: "POTTS_CONTROLS_FIXTURE_DIR not set; no Python dumps with controls to compare against",
  }, () => {});
} else {
  test("controls fixture comparison against aminx Python dumps (exact, NaN-equal)", () => {
    assert.ok(existsSync(FIXTURE_DIR), `fixture dir missing: ${FIXTURE_DIR}`);
    const dumps = readdirSync(FIXTURE_DIR)
      .filter((file) => file.endsWith(".json"))
      .sort()
      .map((file) => ({ file, dump: JSON.parse(readFileSync(join(FIXTURE_DIR, file), "utf8")) }))
      .filter(({ dump }) => dump.controls !== undefined);
    assert.ok(dumps.length > 0, `no dumps with a "controls" field in ${FIXTURE_DIR}`);

    for (const { file, dump } of dumps) {
      const pdbPath = join(FIXTURE_DIR, dump.pdb);
      assert.ok(existsSync(pdbPath), `${file}: PDB missing: ${pdbPath}`);
      const pdbText = readFileSync(pdbPath, "utf8");
      const where = `${file} (${dump.pdb} L${dump.bucket})`;

      const built = buildPottsInputs(pdbText, dump.bucket);
      const ours = applyPottsControls(built, dump.controls);
      assert.deepEqual(Object.keys(ours).sort(), Object.keys(dump.inputs).sort(), `${where}: input names`);

      for (const [name, want] of Object.entries(dump.inputs)) {
        const got = ours[name];
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
