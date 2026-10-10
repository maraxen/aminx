import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  DEFAULT_DEP_MAP,
  PhPlanError,
  blockPartners,
  blockTable,
  finalizePlan,
  knnRank,
  placementScores,
  planCentreFree,
  planFromCenterTypes,
  planFromExplicitCenters,
  rankedPositions,
  validTokenMask,
} from "../ph_plan.mjs";

// ---- unit tests that need no Python ------------------------------------------------------------
test("validTokenMask: everything except X and the forbidden tokens that exist (HIS-D is ignored)", () => {
  const mask = validTokenMask(["HIS-A", "ASP-A", "GLU-A", "UNK", "HIS-D"]);
  assert.equal(mask.length, 30);
  assert.deepEqual([20, 23, 26, 29].map((i) => mask[i]), [0, 0, 0, 0]);
  assert.equal(mask.reduce((a, b) => a + b, 0), 26);
});

test("blockPartners: p plus nearest designable partners, shorter when few exist, [p] for size 1", () => {
  assert.deepEqual(blockPartners([5, 6, 7, 8], 3, new Set([3, 6, 8]), 3), [3, 6, 8]);
  assert.deepEqual(blockPartners([5, 6, 7, 8], 3, new Set([3, 6]), 3), [3, 6]);
  assert.deepEqual(blockPartners([5, 6], 3, new Set([3, 5, 6]), 1), [3]);
});

test("rankedPositions: ascending score, ties to ascending position, infinities dropped", () => {
  assert.deepEqual(rankedPositions(Float64Array.of(1, 0, 0, Infinity, 0.5)), [1, 2, 4, 0]);
});

test("placementScores: field[prot] - min field[deps], +Infinity outside the binder", () => {
  const field = Float32Array.of(0, 5, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const s = placementScores(field, 30, 1, [2, 4], [1]);
  assert.equal(s[0], 5 - 1);
  assert.throws(() => placementScores(field, 30, 1, [], [1]), PhPlanError);
});

test("blockTable pads with -1 and marks validity", () => {
  const slots = 3;
  const eIdx = Int32Array.from([0, 1, 2, 1, 0, 2, 2, 0, 1]);
  const { blocks, valid } = blockTable(eIdx, slots, [0, 1], 3);
  // position 2 is not designable, so each block holds only its own position and the one designable partner
  assert.deepEqual(Array.from(blocks), [0, 1, -1, 1, 0, -1]);
  assert.deepEqual(Array.from(valid), [1, 1, 0, 1, 1, 0]);
  const small = blockTable(eIdx, slots, [0], 3);
  assert.deepEqual(Array.from(small.blocks), [0, -1, -1]);
});

test("refusals: unknown centre type, empty and unknown contrast tokens, unknown scope", () => {
  const eIdx = Int32Array.from([0, 1, 1, 0]);
  const binder = [1, 1];
  const opts = { infillScope: "chain", neighbourK: 0, maxMutations: 0 };
  assert.throws(() => planFromExplicitCenters(eIdx, 2, 2, binder, [10, 11], [[10, "LYS-P"]], DEFAULT_DEP_MAP, opts), PhPlanError);
  assert.throws(() => planFromExplicitCenters(eIdx, 2, 2, binder, [10, 11], [[10, "HIS-P"]], { "HIS-P": [] }, opts), PhPlanError);
  assert.throws(() => planFromExplicitCenters(eIdx, 2, 2, binder, [10, 11], [[10, "HIS-P"]], { "HIS-P": ["HID"] }, opts), PhPlanError);
  assert.throws(() => finalizePlan(eIdx, 2, 2, binder, [], { ...opts, infillScope: "everything" }), PhPlanError);
  assert.throws(() => planFromExplicitCenters(eIdx, 2, 2, binder, [10, 11], [[99, "HIS-P"]], DEFAULT_DEP_MAP, opts), PhPlanError);
});

test("knnRank puts unranked positions last and breaks ties on position", () => {
  // L=3, K=3. Row 0 (the pin) lists 2 then 1, so position 2 ranks 0 and position 1 ranks 1.
  const eIdx = Int32Array.from([0, 2, 1, 1, 0, 2, 2, 1, 0]);
  assert.deepEqual(knnRank(eIdx, 3, [1, 2], [{ position: 0 }]), [2, 1]);
  // a designable position in no pin row is unranked and sorts last, ties on position
  assert.deepEqual(knnRank(eIdx, 3, [1, 2], []), [1, 2]);
});

// ---- comparison with the Python functions --------------------------------------------------------
const FIXTURE = process.env.PROTONPOTTS_PLAN_FIXTURE;
const cases = FIXTURE && existsSync(FIXTURE) ? JSON.parse(readFileSync(FIXTURE, "utf8")).cases : [];

function normalisePlan(plan) {
  return plan === null ? null : JSON.parse(JSON.stringify(plan));
}

function run(c) {
  const i = c.inputs;
  const eIdx = Int32Array.from(i.e_idx ?? []);
  const binder = Uint8Array.from(i.binder ?? []);
  switch (c.kind) {
    case "center_types":
      return normalisePlan(planFromCenterTypes(Float64Array.from(i.field), i.vocab, eIdx, i.length, i.slots, binder, i.res_id, i.types, i.dep_map, i.opts));
    case "explicit":
      return normalisePlan(planFromExplicitCenters(eIdx, i.length, i.slots, binder, i.res_id, i.centers, i.dep_map, i.opts));
    case "centre_free":
      return normalisePlan(planCentreFree(binder));
    case "block_table": {
      const { blocks, valid } = blockTable(eIdx, i.slots, i.designable, i.block_size);
      return { blocks: Array.from(blocks), valid: Array.from(valid) };
    }
    case "valid_mask":
      return Array.from(validTokenMask(i.forbidden));
    default:
      throw new Error(`unknown case kind ${c.kind}`);
  }
}

test("fixtures: a Python dump was supplied", { skip: FIXTURE === undefined }, () => {
  assert.ok(cases.length > 0, `no cases in ${FIXTURE}`);
});

for (const c of cases) {
  test(`matches ph_plan.py: ${c.name}`, () => {
    assert.deepEqual(run(c), c.expected);
  });
}

if (cases.length > 0) {
  test("negative control: a corrupted expected plan is caught", () => {
    const c = cases.find((x) => x.kind === "center_types" && x.expected !== null);
    const bad = JSON.parse(JSON.stringify(c));
    bad.expected.designable[0] += 1;
    assert.notDeepEqual(run(bad), bad.expected);
  });
}
