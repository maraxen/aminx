// T11d: static guard that browser/layer_c/parity.mjs's runCell() actually releases its
// ORT Web session. A missing release() leaks wasm heap allocations across cells in the
// same page until the fixed heap fills, then every later InferenceSession.create() fails
// with "Can't create a session. ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc" (or a bare
// numeric wasm abort). This reads the source as text rather than importing it, so it runs
// in plain Node with no ORT Web / wasm runtime required.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SOURCE_PATH = path.join(__dirname, "parity.mjs");

function extractBalancedBraceBlock(source, braceStart) {
  let depth = 0;
  for (let i = braceStart; i < source.length; i += 1) {
    if (source[i] === "{") {
      depth += 1;
    } else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(braceStart, i + 1);
      }
    }
  }
  throw new Error(`unbalanced braces starting at index ${braceStart}`);
}

function extractFunctionBody(source, functionSignature) {
  const startIdx = source.indexOf(functionSignature);
  if (startIdx === -1) {
    throw new Error(`could not find "${functionSignature}" in ${SOURCE_PATH}`);
  }
  const braceStart = source.indexOf("{", startIdx);
  if (braceStart === -1) {
    throw new Error(`could not find opening brace for "${functionSignature}"`);
  }
  return extractBalancedBraceBlock(source, braceStart);
}

function extractFinallyBlock(functionBody) {
  const idx = functionBody.lastIndexOf("finally");
  if (idx === -1) {
    return null;
  }
  const braceStart = functionBody.indexOf("{", idx);
  if (braceStart === -1) {
    return null;
  }
  return extractBalancedBraceBlock(functionBody, braceStart);
}

test("runCell releases the ORT session inside a finally block", () => {
  const source = fs.readFileSync(SOURCE_PATH, "utf8");
  const runCellBody = extractFunctionBody(source, "async function runCell(cell) {");

  assert.ok(
    runCellBody.includes("finally"),
    "runCell() has no finally block -- session.release() is not guaranteed to run",
  );

  const finallyBlock = extractFinallyBlock(runCellBody);
  assert.ok(finallyBlock, "could not isolate runCell()'s finally block");
  assert.match(
    finallyBlock,
    /session\.release\(\)/,
    "runCell()'s finally block must call session.release() so ORT Web frees the wasm-side session on every path, including a thrown cell error",
  );
});
