// RunSpec -> P07 input tensors for the browser ProteinMPNN sampler.
//
// Node CLI/test-harness wrapper. The numeric logic (PRNG, gumbel, shuffle,
// tieGroupMap, buildP07Inputs) lives in browser/aminx-sampler/runspec_core.mjs
// (no node: imports, browser-safe) so it is exactly the same code the
// browser sampler uses -- this module just re-exports it and adds the
// node:fs/node:path-based CLI entry point + debug ops (`gumbel`, `orders`,
// `fixed_order`) used by scripts/browser_validation/p07_knobs_gate.py.
//
// ES module, no dependencies beyond node builtins. Node 24 and Chromium.
// Alphabet is aminx's MPNN_ALPHABET (src/aminx/utils/aa_convert.py):
// "ACDEFGHIKLMNPQRSTVWYX".
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  MPNN_ALPHABET,
  OMIT_BIAS,
  badGumbelFromUniform,
  biasedShuffle,
  buildP07Inputs,
  designedFlags,
  fixedFirstShuffle,
  gumbelFromUniform,
  makePrng,
  nextBelow,
  shuffle,
  tieGroupMap,
  toFloat32,
  uniformFromU32,
} from "../aminx-sampler/runspec_core.mjs";

export {
  MPNN_ALPHABET,
  OMIT_BIAS,
  badGumbelFromUniform,
  biasedShuffle,
  buildP07Inputs,
  designedFlags,
  fixedFirstShuffle,
  gumbelFromUniform,
  makePrng,
  nextBelow,
  shuffle,
  tieGroupMap,
  toFloat32,
  uniformFromU32,
};

function lexRank(perm) {
  const avail = [];
  for (let i = 0; i < perm.length; i += 1) avail.push(i);
  const fact = [1];
  for (let i = 1; i <= perm.length; i += 1) fact[i] = fact[i - 1] * i;
  let rank = 0;
  for (let i = 0; i < perm.length; i += 1) {
    const idx = avail.indexOf(perm[i]);
    rank += idx * fact[perm.length - 1 - i];
    avail.splice(idx, 1);
  }
  return rank;
}

function writeGumbel(spec) {
  const rows = spec.rows;
  const cols = spec.cols;
  const transform = spec.transform === "bad" ? badGumbelFromUniform : gumbelFromUniform;
  const prng = makePrng(spec.seed >>> 0);
  const out = Buffer.alloc(rows * cols * 4);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  for (let i = 0; i < rows * cols; i += 1) {
    view.setFloat32(i * 4, transform(uniformFromU32(prng.nextU32())), true);
  }
  fs.writeFileSync(spec.out, out);
  return { ok: true, bytes: out.length, rows, cols };
}

function violatesFixedFirst(order, designed) {
  let seenDesigned = false;
  for (let i = 0; i < order.length; i += 1) {
    if (designed[order[i]]) seenDesigned = true;
    else if (seenDesigned) return true;
  }
  return false;
}

function subpermRank(order, designedPositions) {
  const rankOf = new Map();
  for (let i = 0; i < designedPositions.length; i += 1) {
    rankOf.set(designedPositions[i], i);
  }
  const sub = [];
  for (let i = 0; i < order.length; i += 1) {
    const rank = rankOf.get(order[i]);
    if (rank !== undefined) sub.push(rank);
  }
  return lexRank(sub);
}

function writeFixedOrder(spec) {
  const length = spec.length;
  const fixed = new Set(spec.fixed);
  const mask = new Float32Array(length);
  mask.fill(1);
  const fixedMask = new Float32Array(length);
  for (const pos of fixed) fixedMask[pos] = 1;
  const tie = tieGroupMap(length, []);
  const designed = designedFlags(length, mask, fixedMask, tie);
  const designedPositions = [];
  for (let i = 0; i < length; i += 1) {
    if (designed[i]) designedPositions.push(i);
  }
  const counts = new Array(factSmall(designedPositions.length)).fill(0);
  let violations = 0;
  let uniformViolations = 0;
  for (let i = 0; i < spec.n; i += 1) {
    const seed = (spec.seed + i) >>> 0;
    const order = fixedFirstShuffle(length, makePrng(seed), designed);
    if (violatesFixedFirst(order, designed)) violations += 1;
    counts[subpermRank(order, designedPositions)] += 1;
    const uniform = shuffle(length, makePrng(seed));
    if (violatesFixedFirst(uniform, designed)) uniformViolations += 1;
  }
  return { violations, uniform_violations: uniformViolations, counts, n: spec.n, length };
}

function writeOrders(spec) {
  const length = spec.length;
  const nPerm = factSmall(length);
  const counts = new Array(nPerm).fill(0);
  const draw = spec.kind === "biased" ? biasedShuffle : shuffle;
  for (let i = 0; i < spec.n; i += 1) {
    const prng = makePrng((spec.seed + i) >>> 0);
    counts[lexRank(draw(length, prng))] += 1;
  }
  return { counts, n: spec.n, length };
}

function factSmall(n) {
  let value = 1;
  for (let i = 2; i <= n; i += 1) value *= i;
  return value;
}

function invokedDirectly() {
  const entry = process.argv[1];
  if (!entry) return false;
  return import.meta.url === pathToFileURL(path.resolve(entry)).href;
}

if (invokedDirectly()) {
  const specPath = process.argv[2];
  if (!specPath) {
    process.stderr.write("usage: node runspec.mjs <spec.json>\n");
    process.exit(2);
  }
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  let result;
  if (spec.op === "gumbel") result = writeGumbel(spec);
  else if (spec.op === "orders") result = writeOrders(spec);
  else if (spec.op === "fixed_order") result = writeFixedOrder(spec);
  else result = buildP07Inputs(spec.structure, spec.runspec);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}
