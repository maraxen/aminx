// WebGPU capability probe. CAPABILITY ONLY — no timings, no parity claims.
//
// Spec ODQ-B3 scopes the WebGPU execution provider to a capability probe precisely
// because a half-measured GPU number is worse than none: ORT partitions graphs across
// providers, so a "WebGPU run" can silently execute most of its work on CPU, and the
// int64 TopK indices ONNX mandates for sort are a dtype the WebGPU EP does not support
// (xtrax research 260914 S4). This answers only: is WebGPU reachable here at all, does
// ORT accept the provider, and does the graph load.
//
// It lives in layer_c rather than beside the sampler because that is where
// node_modules is: @playwright/test is not resolvable from browser/aminx-sampler,
// and deliberately so -- the sampler package a site copies must not depend on
// Playwright or onnxruntime.
//
// It needs its own driver rather than reusing browser/layer_c/run_p07.mjs because
// headless Chromium requires explicit flags to expose WebGPU, and run_p07.mjs is shared
// with gates that must not have their launch conditions changed underneath them.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

import { createStaticServer, listenEphemeral } from "./serve.mjs";

function arg(name, fallback = undefined) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) {
    if (fallback === undefined) throw new Error(`missing --${name}`);
    return fallback;
  }
  return process.argv[i + 1];
}

const site = arg("site");
const outPath = arg("out");
const bucket = Number(arg("bucket", "128"));

const probe = {
  probe_ok: false,
  probe_error: null,
  chromium_version: null,
  navigator_gpu_present: false,
  adapter_obtained: false,
  adapter_info: null,
  ort_webgpu_session_created: false,
  ort_webgpu_error: null,
  encoder_ran: false,
  encoder_outputs_finite: null,
  console_errors: [],
};

const server = createStaticServer(site, { isolate: true, resultsDir: path.dirname(outPath) });
const port = await listenEphemeral(server);
let browser;
try {
  // Flags that expose WebGPU in headless Chromium. SwiftShader is allowed deliberately:
  // this box may have no usable GPU, and "WebGPU works via a software adapter" is a
  // materially different answer from "WebGPU is unavailable" — both are worth recording,
  // and conflating them would misinform anyone deciding whether to pursue the GPU path.
  browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-unsafe-swiftshader",
      "--enable-features=Vulkan",
      "--use-angle=swiftshader",
    ],
  });
  probe.chromium_version = browser.version();
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") probe.console_errors.push(m.text());
  });
  page.on("pageerror", (e) => probe.console_errors.push(String(e)));

  await page.goto(`http://127.0.0.1:${port}/index.html?bucket=${bucket}`, {
    waitUntil: "load",
  });
  await page.waitForFunction(() => window.__PROBE_DONE__ === true, undefined, {
    timeout: 300_000,
  });
  const res = await page.evaluate(() => window.__PROBE_RESULT__ || null);
  Object.assign(probe, res || {});
  probe.probe_ok = true;
} catch (err) {
  probe.probe_ok = false;
  probe.probe_error = String((err && err.stack) || err);
} finally {
  if (browser) await browser.close();
  await new Promise((r) => server.close(r));
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(probe, null, 2));
process.stdout.write(`${JSON.stringify(probe)}\n`);
void fileURLToPath;
process.exit(0);
