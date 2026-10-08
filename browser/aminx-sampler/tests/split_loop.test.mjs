// Node tests for split_loop.mjs -- the pure four-graph autoregressive wave
// loop (task_id 260926_browser-export-loop / T13). Uses hand-built FAKE
// graph callbacks: no onnxruntime, no .onnx file. Graph D/F's actual
// numerics (attention, softmax-with-bias, categorical sampling) are covered
// separately by the G0/G0b/G0c export-parity gates; these tests exercise
// only the JS loop's control flow -- which graph is called when, with what
// tensors, and how the results are threaded into the evolving sequence and
// the final per-position gather.
import assert from "node:assert/strict";
import { test } from "node:test";

import { at, makeTensor, runSplitDecode, UNDRAWN_TOKEN } from "../split_loop.mjs";
import { buildP07TypedInputs, MPNN_ALPHABET, OMIT_BIAS } from "../runspec_core.mjs";
import { createSplitSampler, resolveFamily } from "../split_driver.mjs";

const N_TOKENS = 21;

function expectedLogSoftmax(row) {
  const max = Math.max(...row);
  const sumExp = row.reduce((sum, v) => sum + Math.exp(v - max), 0);
  const logSumExp = Math.log(sumExp) + max;
  return row.map((v) => v - logSumExp);
}

function logProbRow(tensor, pos) {
  const start = pos * N_TOKENS;
  return Array.from(tensor.data.slice(start, start + N_TOKENS));
}

function assertRowClose(actual, expected, tol, message) {
  assert.equal(actual.length, expected.length, message);
  for (let i = 0; i < actual.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < tol,
      `${message}: index ${i}: got ${actual[i]}, want ${expected[i]}`,
    );
  }
}

function uniformStructureTensors(L) {
  return {
    mask: makeTensor([L], Array.from({ length: L }, () => 1)),
    condBias: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)),
    fixedMask: makeTensor([L], Array.from({ length: L }, () => 0)),
    fixedTokens: makeTensor([L], Array.from({ length: L }, () => 0)),
    temperature: makeTensor([], [1]),
    gumbelNoise: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)),
    encoderInputs: { marker: "encoder-inputs" },
  };
}

function makeEncoderStub(calls) {
  const nodeFeatures = makeTensor([2, 2], [1, 2, 3, 4]);
  const edgeFeatures = makeTensor([2, 2], [5, 6, 7, 8]);
  const neighborIndices = makeTensor([2, 2], [0, 1, 1, 0]);
  return async (inputs) => {
    calls.push(inputs);
    return { node_features: nodeFeatures, edge_features: edgeFeatures, neighbor_indices: neighborIndices };
  };
}

// ---------------------------------------------------------------------------
// Test A + E fixture: L=8, G=2, 2 waves. Wave 0 has one active slot (group 0)
// and one inactive/padded slot. Wave 1 has two active slots (groups 1, 2).
// Positions 3-7 are never scheduled by any slot in this schedule.
// ---------------------------------------------------------------------------
function buildPaddedSlotFixture() {
  const L = 8;
  const G = 2;
  const nWaves = 2;
  const tieGroupMap = makeTensor([L], [0, 1, 2, 3, 4, 5, 6, 7]); // no ties, identity
  const groupPositions = makeTensor(
    [nWaves, G, 1],
    [
      [[0], [3]], // wave 0: slot0 -> pos0=0 (active); slot1 -> pos0=3 (inactive, value unused)
      [[1], [2]], // wave 1: slot0 -> pos0=1; slot1 -> pos0=2
    ],
  );
  const groupValid = makeTensor(
    [nWaves, G],
    [
      [1, 0],
      [1, 1],
    ],
  );
  const sentinel = nWaves * G; // 4
  // group ids that appear: 0 (rank 0), 1 (rank 2), 2 (rank 3); 3..7 never scheduled.
  const groupFirstRank = makeTensor([L], [0, 2, 3, sentinel, sentinel, sentinel, sentinel, sentinel]);
  const posFirstRank = groupFirstRank; // tieGroupMap is identity here
  const arMask = makeTensor([L, L], Array.from({ length: L * L }, () => 0));

  const decoderCalls = [];
  const fuseCalls = [];
  let activeWaveIndex = 0;

  const encoderCalls = [];
  const runEncoder = makeEncoderStub(encoderCalls);

  const runWave = async (inputs) => {
    void inputs;
    return {
      group_ids: makeTensor([nWaves, G], [0, 0, 0, 0]), // contract shape only; loop reads group ids via tieGroupMap
      group_positions: groupPositions,
      group_valid: groupValid,
      position_valid: makeTensor([nWaves, G, 1], [[[1], [1]], [[1], [1]]]),
      ar_mask: arMask,
      group_first_rank: groupFirstRank,
      pos_first_rank: posFirstRank,
    };
  };

  const runDecoder = async (inputs) => {
    decoderCalls.push({
      node_features: inputs.node_features,
      edge_features: inputs.edge_features,
      neighbor_indices: inputs.neighbor_indices,
      mask: inputs.mask,
      ar_mask: inputs.ar_mask,
    });
    return { logits: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)) };
  };

  const runFuse = async (inputs) => {
    const w = activeWaveIndex;
    fuseCalls.push({ w, inputs });
    const G_ = inputs.group_id.shape[0];
    const finalToken = [];
    const avgStored = [];
    for (let g = 0; g < G_; g += 1) {
      finalToken.push(5 + w * 2 + g);
      for (let k = 0; k < N_TOKENS; k += 1) avgStored.push(w * 100 + g * 10 + k);
    }
    activeWaveIndex += 1;
    return {
      final_token: makeTensor([G_], finalToken),
      avg_stored: makeTensor([G_, N_TOKENS], avgStored),
    };
  };

  const structureTensors = uniformStructureTensors(L);
  return {
    L,
    G,
    tieGroupMap,
    structureTensors,
    callbacks: { runEncoder, runWave, runDecoder, runFuse },
    decoderCalls,
    fuseCalls,
  };
}

test("wave with an inactive/padded slot: only the active slot updates the sequence, padded slot and never-scheduled positions stay undrawn", async () => {
  const fx = buildPaddedSlotFixture();
  const result = await runSplitDecode(fx.callbacks, {
    length: fx.L,
    encoderInputs: fx.structureTensors.encoderInputs,
    waveInputs: { marker: "wave-inputs" },
    tieGroupMap: fx.tieGroupMap,
    mask: fx.structureTensors.mask,
    condBias: fx.structureTensors.condBias,
    fixedMask: fx.structureTensors.fixedMask,
    fixedTokens: fx.structureTensors.fixedTokens,
    temperature: fx.structureTensors.temperature,
    gumbelNoise: fx.structureTensors.gumbelNoise,
  });

  const seq = Array.from(result.tokens.data);
  // group0 (pos0=0) scheduled wave0/slot0 -> final_token = 5 + 0*2 + 0 = 5
  assert.equal(seq[0], 5, "position 0 (active slot) should take the fused token");
  // group1 (pos0=1) scheduled wave1/slot0 -> final_token = 5 + 1*2 + 0 = 7
  assert.equal(seq[1], 7, "position 1 should take wave1/slot0's token");
  // group2 (pos0=2) scheduled wave1/slot1 -> final_token = 5 + 1*2 + 1 = 8
  assert.equal(seq[2], 8, "position 2 should take wave1/slot1's token");
  // position 3 was wave0's PADDED slot's pos0 value, but that slot was inactive
  // (group_valid=0) so it must never be written -- it stays UNDRAWN, same as
  // positions 4-7 which no slot ever references.
  for (const pos of [3, 4, 5, 6, 7]) {
    assert.equal(seq[pos], UNDRAWN_TOKEN, `position ${pos} was never scheduled and must stay undrawn`);
  }

  // Only 2 fuse calls happened (one per wave; both waves had >=1 active slot).
  assert.equal(fx.fuseCalls.length, 2);

  // log-probs: scheduled positions gather their (wave, slot)'s avg_stored row;
  // never-scheduled positions gather an all-zero row (log_softmax of zeros).
  assertRowClose(logProbRow(result.logProbs, 0), expectedLogSoftmax(Array.from({ length: 21 }, (_, k) => k)), 1e-5, "pos0 logits");
  assertRowClose(
    logProbRow(result.logProbs, 1),
    expectedLogSoftmax(Array.from({ length: 21 }, (_, k) => 100 + k)),
    1e-5,
    "pos1 logits",
  );
  assertRowClose(
    logProbRow(result.logProbs, 2),
    expectedLogSoftmax(Array.from({ length: 21 }, (_, k) => 110 + k)),
    1e-5,
    "pos2 logits",
  );
  const uniform = expectedLogSoftmax(new Array(21).fill(0));
  for (const pos of [3, 4, 5, 6, 7]) {
    assertRowClose(logProbRow(result.logProbs, pos), uniform, 1e-5, `pos${pos} unscheduled logits`);
  }
});

test("the five loop-invariant decoder inputs (node/edge/neighbor features, mask, ar_mask) are passed identically on every wave; only sequence_oh varies", async () => {
  const fx = buildPaddedSlotFixture();
  await runSplitDecode(fx.callbacks, {
    length: fx.L,
    encoderInputs: fx.structureTensors.encoderInputs,
    waveInputs: { marker: "wave-inputs" },
    tieGroupMap: fx.tieGroupMap,
    mask: fx.structureTensors.mask,
    condBias: fx.structureTensors.condBias,
    fixedMask: fx.structureTensors.fixedMask,
    fixedTokens: fx.structureTensors.fixedTokens,
    temperature: fx.structureTensors.temperature,
    gumbelNoise: fx.structureTensors.gumbelNoise,
  });

  assert.equal(fx.decoderCalls.length, 2, "both waves had an active slot, so decoder runs twice");
  const [callA, callB] = fx.decoderCalls;
  // Reference-identical: the loop must bind these once, not rebuild them per wave.
  assert.equal(callA.node_features, callB.node_features);
  assert.equal(callA.edge_features, callB.edge_features);
  assert.equal(callA.neighbor_indices, callB.neighbor_indices);
  assert.equal(callA.mask, callB.mask);
  assert.equal(callA.ar_mask, callB.ar_mask);
});

// ---------------------------------------------------------------------------
// Test B: a multi-member tie group -- all tied positions receive the same
// sampled token, untied/never-scheduled positions stay undrawn.
// ---------------------------------------------------------------------------
test("multi-member tie group: all tied positions receive the same token", async () => {
  const L = 5;
  const G = 1;
  const nWaves = 1;
  // positions 1, 3, 4 are tied into group id 1 (smallest member); 0 and 2 are singletons.
  const tieGroupMap = makeTensor([L], [0, 1, 2, 1, 1]);
  const groupPositions = makeTensor([nWaves, G, 1], [[[1]]]);
  const groupValid = makeTensor([nWaves, G], [[1]]);
  const sentinel = nWaves * G; // 1
  const groupFirstRank = makeTensor([L], [sentinel, 0, sentinel, sentinel, sentinel]);
  const posFirstRank = makeTensor(
    [L],
    [0, 1, 2, 3, 4].map((pos) => at(groupFirstRank, at(tieGroupMap, pos))),
  );
  const arMask = makeTensor([L, L], Array.from({ length: L * L }, () => 0));

  const encoderCalls = [];
  const runEncoder = makeEncoderStub(encoderCalls);
  const runWave = async () => ({
    group_ids: makeTensor([nWaves, G], [0]),
    group_positions: groupPositions,
    group_valid: groupValid,
    position_valid: makeTensor([nWaves, G, 1], [[[1]]]),
    ar_mask: arMask,
    group_first_rank: groupFirstRank,
    pos_first_rank: posFirstRank,
  });
  const runDecoder = async () => ({ logits: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)) });
  const runFuse = async () => ({
    final_token: makeTensor([1], [9]),
    avg_stored: makeTensor([1, N_TOKENS], Array.from({ length: N_TOKENS }, (_, k) => 100 + k)),
  });

  const structureTensors = uniformStructureTensors(L);
  const result = await runSplitDecode(
    { runEncoder, runWave, runDecoder, runFuse },
    {
      length: L,
      encoderInputs: structureTensors.encoderInputs,
      waveInputs: { marker: "wave-inputs" },
      tieGroupMap,
      mask: structureTensors.mask,
      condBias: structureTensors.condBias,
      fixedMask: structureTensors.fixedMask,
      fixedTokens: structureTensors.fixedTokens,
      temperature: structureTensors.temperature,
      gumbelNoise: structureTensors.gumbelNoise,
    },
  );

  const seq = Array.from(result.tokens.data);
  assert.deepEqual(seq, [UNDRAWN_TOKEN, 9, UNDRAWN_TOKEN, 9, 9], "tied positions 1,3,4 all get token 9");

  const expected = expectedLogSoftmax(Array.from({ length: 21 }, (_, k) => 100 + k));
  for (const pos of [1, 3, 4]) {
    assertRowClose(logProbRow(result.logProbs, pos), expected, 1e-5, `pos${pos} logits`);
  }
  const uniform = expectedLogSoftmax(new Array(21).fill(0));
  for (const pos of [0, 2]) {
    assertRowClose(logProbRow(result.logProbs, pos), uniform, 1e-5, `pos${pos} unscheduled logits`);
  }
});

// ---------------------------------------------------------------------------
// Test C: a fixed position overriding the sample. Graph F (fake here) is
// responsible for the override, mirroring _fuse_and_sample lines 181-188;
// this test verifies split_loop.mjs plumbs fixed_mask/fixed_tokens/group_id
// through to runFuse and trusts its returned final_token verbatim.
// ---------------------------------------------------------------------------
test("a fixed position overrides the sample via Graph F's final_token", async () => {
  const L = 3;
  const G = 1;
  const nWaves = 1;
  const tieGroupMap = makeTensor([L], [0, 1, 2]);
  const groupPositions = makeTensor([nWaves, G, 1], [[[1]]]);
  const groupValid = makeTensor([nWaves, G], [[1]]);
  const sentinel = nWaves * G;
  const groupFirstRank = makeTensor([L], [sentinel, 0, sentinel]);
  const posFirstRank = groupFirstRank;
  const arMask = makeTensor([L, L], Array.from({ length: L * L }, () => 0));

  const fixedMask = makeTensor([L], [0, 1, 0]);
  const fixedTokens = makeTensor([L], [0, 7, 0]);

  const encoderCalls = [];
  const runEncoder = makeEncoderStub(encoderCalls);
  const runWave = async () => ({
    group_ids: makeTensor([nWaves, G], [0]),
    group_positions: groupPositions,
    group_valid: groupValid,
    position_valid: makeTensor([nWaves, G, 1], [[[1]]]),
    ar_mask: arMask,
    group_first_rank: groupFirstRank,
    pos_first_rank: posFirstRank,
  });
  const runDecoder = async () => ({ logits: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)) });

  // Mirrors _fuse_and_sample's fixed-position override (lines 181-188): a
  // group whose masked positions include ANY fixed position is forced to
  // that fixed token, overriding whatever was "sampled".
  const runFuse = async (inputs) => {
    const G_ = inputs.group_id.shape[0];
    const finalToken = [];
    for (let g = 0; g < G_; g += 1) {
      const WRONG_SAMPLE = 3; // stands in for whatever categorical sampling would draw
      let isGroupFixed = false;
      let groupFixedToken = 0;
      for (let pos = 0; pos < L; pos += 1) {
        if (at(inputs.mask_group, g, pos) && at(inputs.fixed_mask, pos) > 0.5) {
          isGroupFixed = true;
          groupFixedToken = Math.max(groupFixedToken, at(inputs.fixed_tokens, pos));
        }
      }
      finalToken.push(isGroupFixed ? groupFixedToken : WRONG_SAMPLE);
    }
    return {
      final_token: makeTensor([G_], finalToken),
      avg_stored: makeTensor([G_, N_TOKENS], Array.from({ length: G_ * N_TOKENS }, () => 0)),
    };
  };

  const structureTensors = uniformStructureTensors(L);
  const result = await runSplitDecode(
    { runEncoder, runWave, runDecoder, runFuse },
    {
      length: L,
      encoderInputs: structureTensors.encoderInputs,
      waveInputs: { marker: "wave-inputs" },
      tieGroupMap,
      mask: structureTensors.mask,
      condBias: structureTensors.condBias,
      fixedMask,
      fixedTokens,
      temperature: structureTensors.temperature,
      gumbelNoise: structureTensors.gumbelNoise,
    },
  );

  const seq = Array.from(result.tokens.data);
  assert.equal(seq[1], 7, "fixed position's group must end up at the fixed token, not the wrong sample");
  assert.notEqual(seq[1], 3, "override must actually replace the sampled value");
  assert.equal(seq[0], UNDRAWN_TOKEN);
  assert.equal(seq[2], UNDRAWN_TOKEN);
});

// ---------------------------------------------------------------------------
// Test D: the final per-position gather picks each group's FIRST occurrence.
// Positions 2 and 5 are tied into group 2. Group 2 is scheduled (as the
// group's real first occurrence) at wave 0/slot 0. A SECOND, later slot at
// wave 1/slot 0 also nominally covers position 5 (as WaveScheduleBundle's
// per-position `empty()` schedules can, per autoregressive.py's own comment
// at lines 438-445) but is NOT the first occurrence, so it must contribute
// nothing -- both positions must gather wave 0's logits, not wave 1's.
// ---------------------------------------------------------------------------
test("final per-position gather uses each group's first (wave, slot) occurrence, not a later duplicate", async () => {
  const L = 6;
  const G = 1;
  const nWaves = 2;
  const tieGroupMap = makeTensor([L], [0, 1, 2, 3, 4, 2]); // positions 2 and 5 tied into group 2
  const groupPositions = makeTensor([nWaves, G, 1], [[[2]], [[5]]]);
  const groupValid = makeTensor([nWaves, G], [[1], [1]]);
  const sentinel = nWaves * G; // 2
  // group 2's first occurrence is wave0/slot0 -> rank 0. All other group ids unused (sentinel).
  const groupFirstRank = makeTensor([L], [sentinel, sentinel, 0, sentinel, sentinel, sentinel]);
  const posFirstRank = makeTensor(
    [L],
    Array.from({ length: L }, (_, pos) => at(groupFirstRank, at(tieGroupMap, pos))),
  );
  const arMask = makeTensor([L, L], Array.from({ length: L * L }, () => 0));

  const encoderCalls = [];
  const runEncoder = makeEncoderStub(encoderCalls);
  const runWave = async () => ({
    group_ids: makeTensor([nWaves, G], [0, 0]),
    group_positions: groupPositions,
    group_valid: groupValid,
    position_valid: makeTensor([nWaves, G, 1], [[[1]], [[1]]]),
    ar_mask: arMask,
    group_first_rank: groupFirstRank,
    pos_first_rank: posFirstRank,
  });
  const runDecoder = async () => ({ logits: makeTensor([L, N_TOKENS], Array.from({ length: L * N_TOKENS }, () => 0)) });

  const fuseCalls = [];
  const runFuse = async (inputs) => {
    fuseCalls.push(inputs);
    // Distinguishable marker per call: call 0 -> all 10s, call 1 (should never
    // happen, since wave1's slot is not a first-occurrence) -> all 20s.
    const value = (fuseCalls.length - 1 + 1) * 10;
    return {
      final_token: makeTensor([1], [5]),
      avg_stored: makeTensor([1, N_TOKENS], Array.from({ length: N_TOKENS }, () => value)),
    };
  };

  const structureTensors = uniformStructureTensors(L);
  const result = await runSplitDecode(
    { runEncoder, runWave, runDecoder, runFuse },
    {
      length: L,
      encoderInputs: structureTensors.encoderInputs,
      waveInputs: { marker: "wave-inputs" },
      tieGroupMap,
      mask: structureTensors.mask,
      condBias: structureTensors.condBias,
      fixedMask: structureTensors.fixedMask,
      fixedTokens: structureTensors.fixedTokens,
      temperature: structureTensors.temperature,
      gumbelNoise: structureTensors.gumbelNoise,
    },
  );

  // Wave 1's slot is not a first-occurrence (is_first_occurrence gates on
  // group_first_rank, not just group_valid), so the loop must skip it
  // entirely -- only ONE fuse call total.
  assert.equal(fuseCalls.length, 1, "the later duplicate occurrence must not trigger a second decode/fuse");

  const seq = Array.from(result.tokens.data);
  assert.equal(seq[2], 5, "position 2 (real first occurrence) gets the fused token");
  assert.equal(seq[5], 5, "position 5 (tied, later duplicate slot) gets the SAME token via the tie, not its own slot");

  const expectedFirst = expectedLogSoftmax(new Array(21).fill(10));
  assertRowClose(logProbRow(result.logProbs, 2), expectedFirst, 1e-5, "pos2 must gather wave0's logits");
  assertRowClose(logProbRow(result.logProbs, 5), expectedFirst, 1e-5, "pos5 must gather wave0's logits, not a later wave's");

  const uniform = expectedLogSoftmax(new Array(21).fill(0));
  for (const pos of [0, 1, 3, 4]) {
    assertRowClose(logProbRow(result.logProbs, pos), uniform, 1e-5, `pos${pos} unscheduled logits`);
  }
});

// ---------------------------------------------------------------------------
// Model-family plumbing (manifest-driven alphabet / token count / input names).
// ---------------------------------------------------------------------------

test("toy 5-token alphabet: one-hot, logits and log-probs are [L,5]; omitting nTokens (default 21) rejects the 5-wide condBias", async () => {
  const L = 3;
  const K = 5;
  const G = 1;
  const nWaves = L;
  const seqOhShapes = [];
  let waveCall = 0;

  const runWave = async () => ({
    group_ids: makeTensor([nWaves, G], [0, 0, 0]),
    group_positions: makeTensor([nWaves, G, 1], [[[0]], [[1]], [[2]]]),
    group_valid: makeTensor([nWaves, G], [[1], [1], [1]]),
    position_valid: makeTensor([nWaves, G, 1], [[[1]], [[1]], [[1]]]),
    ar_mask: makeTensor([L, L], new Array(L * L).fill(0)),
    group_first_rank: makeTensor([L], [0, 1, 2]),
    pos_first_rank: makeTensor([L], [0, 1, 2]),
  });
  const runDecoder = async (inputs) => {
    seqOhShapes.push(inputs.sequence_oh.shape);
    return { logits: makeTensor([L, K], new Array(L * K).fill(0)) };
  };
  const runFuse = async (inputs) => {
    const wave = waveCall;
    waveCall += 1;
    assert.deepEqual(inputs.logits.shape, [1, L, K]);
    assert.deepEqual(inputs.cond_bias.shape, [L, K]);
    assert.deepEqual(inputs.gumbel_noise.shape, [L, K]);
    const avg = [];
    for (let k = 0; k < K; k += 1) avg.push(wave * 10 + k);
    return { final_token: makeTensor([G], [wave + 1]), avg_stored: makeTensor([G, K], avg) };
  };
  const callbacks = { runEncoder: makeEncoderStub([]), runWave, runDecoder, runFuse };
  const params = {
    length: L,
    nTokens: K,
    encoderInputs: {},
    waveInputs: {},
    tieGroupMap: makeTensor([L], [0, 1, 2]),
    mask: makeTensor([L], [1, 1, 1]),
    condBias: makeTensor([L, K], new Array(L * K).fill(0)),
    fixedMask: makeTensor([L], [0, 0, 0]),
    fixedTokens: makeTensor([L], [0, 0, 0]),
    temperature: makeTensor([], [1]),
    gumbelNoise: makeTensor([L, K], new Array(L * K).fill(0)),
  };

  const result = await runSplitDecode(callbacks, params);
  assert.deepEqual(seqOhShapes, [[L, K], [L, K], [L, K]], "sequence_oh is one-hot over the 5 tokens");
  assert.deepEqual(Array.from(result.tokens.data), [1, 2, 3], "each wave's fused token lands at its position");
  assert.deepEqual(result.logProbs.shape, [L, K]);
  for (let pos = 0; pos < L; pos += 1) {
    const row = Array.from({ length: K }, (_, k) => pos * 10 + k);
    const actual = Array.from(result.logProbs.data.slice(pos * K, pos * K + K));
    assertRowClose(actual, expectedLogSoftmax(row), 1e-5, `pos${pos} log-probs`);
  }

  await assert.rejects(
    runSplitDecode(callbacks, { ...params, nTokens: undefined }),
    /condBias has 5 tokens per position, nTokens is 21/,
    "without nTokens the loop assumes ProteinMPNN's 21 and must refuse a 5-wide condBias",
  );
});

test("buildP07TypedInputs with a toy alphabet: bias and gumbel are [L,nTokens] and omit/bias letters index that alphabet", () => {
  const L = 3;
  const structure = {
    coords: Array.from({ length: L }, () => [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
    mask: [1, 1, 1],
    residue_index: [0, 1, 2],
    chain_index: [0, 0, 0],
  };
  const runspec = { decoding_order: [0, 1, 2], omit_AA: "A", bias_AA: { C: 0.5 } };
  const built = buildP07TypedInputs(structure, runspec, { alphabet: "ABCDE" });
  assert.deepEqual(built.bias.shape, [L, 5]);
  assert.deepEqual(built.gumbel_noise.shape, [L, 5]);
  for (let pos = 0; pos < L; pos += 1) {
    assert.equal(built.bias.data[pos * 5 + 0], -1e8, `pos${pos} omits A (index 0), at OMIT_BIAS`);
    assert.equal(built.bias.data[pos * 5 + 2], 0.5, `pos${pos} biases C (index 2)`);
  }
  assert.throws(
    () => buildP07TypedInputs(structure, { decoding_order: [0, 1, 2], omit_AA: "Z" }, { alphabet: "ABCDE" }),
    /letter Z is not in alphabet ABCDE/,
  );

  // Default options reproduce the ProteinMPNN shapes exactly.
  const mpnn = buildP07TypedInputs(structure, { decoding_order: [0, 1, 2] });
  assert.deepEqual(mpnn.bias.shape, [L, MPNN_ALPHABET.length]);
  assert.equal(MPNN_ALPHABET.length, 21);
  assert.equal(OMIT_BIAS, -1e8);
});

test("resolveFamily: manifest values win; absent values fall back to ProteinMPNN; n_tokens must equal the alphabet length", () => {
  assert.deepEqual(resolveFamily(undefined), { alphabet: MPNN_ALPHABET, nTokens: 21, omitBias: -1e8 });
  // The committed MANIFEST.json shape: alphabet only, no n_tokens / omit_bias / input_names.
  assert.deepEqual(resolveFamily({ alphabet: MPNN_ALPHABET, buckets: [] }), {
    alphabet: MPNN_ALPHABET,
    nTokens: 21,
    omitBias: -1e8,
  });
  assert.deepEqual(resolveFamily({ alphabet: "ABCDE", n_tokens: 5, omit_bias: -50 }), {
    alphabet: "ABCDE",
    nTokens: 5,
    omitBias: -50,
  });
  assert.throws(() => resolveFamily({ alphabet: "ABCDE", n_tokens: 21 }), /n_tokens 21 != alphabet length 5/);
});

// Fake onnxruntime-web: sessions are keyed by the model source object, so a
// test controls each graph's input count without any .onnx file.
function fakeOrt(sessionsBySource) {
  return {
    env: { wasm: {} },
    Tensor: class {
      constructor(dtype, data, dims) {
        this.dtype = dtype;
        this.data = data;
        this.dims = dims;
      }
    },
    InferenceSession: {
      create: async (source) => {
        const spec = sessionsBySource.get(source);
        if (!spec) throw new Error("fake ort: unknown model source");
        return {
          inputNames: spec.inputNames,
          outputNames: [],
          run: async () => ({}),
          release: async () => {},
        };
      },
    },
  };
}

// ONNX-side input counts match the ProteinMPNN graphs (encoder 4, wave 2,
// decoder 6, fuse 8). jax2onnx names the ONNX inputs in_0, in_1, ...
function fakeSplitSessions(counts = { encoder: 4, wave: 2, decoder: 6, fuse: 8 }) {
  const urls = {
    encoderUrl: new Uint8Array([1]),
    waveUrl: new Uint8Array([2]),
    decoderUrl: new Uint8Array([3]),
    fuseUrl: new Uint8Array([4]),
  };
  const map = new Map();
  for (const [urlKey, graph] of [
    ["encoderUrl", "encoder"],
    ["waveUrl", "wave"],
    ["decoderUrl", "decoder"],
    ["fuseUrl", "fuse"],
  ]) {
    map.set(urls[urlKey], { inputNames: Array.from({ length: counts[graph] }, (_, i) => `in_${i}`) });
  }
  return { sources: urls, map };
}

function manifestFixture({ alphabet = MPNN_ALPHABET, encoderNames, graphOverrides = {} } = {}) {
  const encoder = {
    file: "p07_encoder_L128.onnx",
    sha256: "0",
    bytes: 1,
    input_shapes: [],
    input_dtypes: ["float32", "float32", "int32", "int32"],
    ...graphOverrides,
  };
  if (encoderNames !== undefined) encoder.input_names = encoderNames;
  return {
    checkpoint_id: "toy",
    git_hash: "0",
    alphabet,
    buckets: [{ bucket: 128, graphs: { encoder } }],
  };
}

test("createSplitSampler: a manifest input_names list whose length differs from the encoder session throws", async () => {
  const { sources, map } = fakeSplitSessions();
  const manifest = manifestFixture({ encoderNames: ["coords", "mask", "residue_index"] });
  await assert.rejects(
    createSplitSampler(fakeOrt(map), sources, { manifest, manifestBucket: 128 }),
    /encoder \(manifest input_names\) session has 4 inputs, expected 3 \(coords, mask, residue_index\)/,
  );
});

test("createSplitSampler: duplicate manifest input_names throw before any session is trusted", async () => {
  const { sources, map } = fakeSplitSessions();
  const manifest = manifestFixture({ encoderNames: ["coords", "coords", "residue_index", "chain_index"] });
  await assert.rejects(
    createSplitSampler(fakeOrt(map), sources, { manifest, manifestBucket: 128 }),
    /input_names contains duplicates/,
  );
});

test("createSplitSampler: manifest input_names of the right count are accepted", async () => {
  const { sources, map } = fakeSplitSessions();
  const manifest = manifestFixture({
    alphabet: "ABCDE",
    encoderNames: ["coords", "mask", "residue_index", "chain_index"],
  });
  const sampler = await createSplitSampler(fakeOrt(map), sources, { manifest, manifestBucket: 128 });
  assert.equal(typeof sampler.sample, "function");
  await sampler.release();
});

test("createSplitSampler: a manifest without the new fields (committed MANIFEST shape) loads exactly as before", async () => {
  const { sources, map } = fakeSplitSessions();
  const committedShape = manifestFixture();
  delete committedShape.buckets[0].graphs.encoder.input_names;
  const sampler = await createSplitSampler(fakeOrt(map), sources, { manifest: committedShape, manifestBucket: 128 });
  await sampler.release();

  // No manifest at all: the pinned ProteinMPNN count still gates the load.
  const bad = fakeSplitSessions({ encoder: 3, wave: 2, decoder: 6, fuse: 8 });
  await assert.rejects(
    createSplitSampler(fakeOrt(bad.map), bad.sources),
    /encoder session has 3 inputs, expected 4/,
  );
});

test("createSplitSampler: a manifest with buckets needs manifestBucket, and an unknown bucket throws", async () => {
  const { sources, map } = fakeSplitSessions();
  const manifest = manifestFixture();
  await assert.rejects(createSplitSampler(fakeOrt(map), sources, { manifest }), /manifestBucket is required/);
  await assert.rejects(
    createSplitSampler(fakeOrt(map), sources, { manifest, manifestBucket: 256 }),
    /manifest has no bucket 256/,
  );
});
