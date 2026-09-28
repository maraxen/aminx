// Phase 2a, T5a: wasm-init thread-assertion verification test (Definitions,
// "Verification test").
//
// Two contexts in one session, requesting 1 and 4 threads. Asserts:
//   - threadsObserved = 1 and 4 respectively (the pthread-Worker-count signal --
//     `workers.length + 1`, see thread-probe.mjs);
//   - in the 1-thread page, setting numThreads = 4 AFTER the first session-create and
//     then creating a second session still reports 1 (documents the init-time fixing);
//   - with --no-isolation (no COOP/COEP -> no SharedArrayBuffer), a request for 4
//     reports 1.
//
// If the worker-count signal does not actually track ORT Web's real thread pool size
// (i.e. requesting 4 does NOT yield threadsObserved = 4), this test's own assertion
// fails loudly rather than silently accepting an unvalidated mechanism -- per the task
// spec, "the mechanism is to be verified for ORT Web 1.30.0 in this task" and "if the
// observed-thread signal cannot be validated, threads_observed = null,
// threads_observed_available = false".
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";
import { createStaticServer, listenEphemeral } from "../serve.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LAYER_C_DIR = path.dirname(__dirname);
const REPO_ROOT = path.resolve(LAYER_C_DIR, "..", "..");
const SITE_DIR = path.join(LAYER_C_DIR, ".test-site");

function buildSite() {
  fs.rmSync(SITE_DIR, { recursive: true, force: true });
  fs.mkdirSync(path.join(SITE_DIR, "models"), { recursive: true });
  fs.cpSync(path.join(LAYER_C_DIR, "node_modules", "onnxruntime-web", "dist"), path.join(SITE_DIR, "ort"), {
    recursive: true,
  });
  fs.copyFileSync(path.join(__dirname, "thread-probe.html"), path.join(SITE_DIR, "thread-probe.html"));
  fs.copyFileSync(path.join(__dirname, "thread-probe.mjs"), path.join(SITE_DIR, "thread-probe.mjs"));
  const onnxSrc = path.join(REPO_ROOT, "outputs", "browser_validation", "phase0", "topk_lattice.onnx");
  fs.copyFileSync(onnxSrc, path.join(SITE_DIR, "models", "topk_lattice.onnx"));
}

test.beforeAll(() => {
  buildSite();
});

test.afterAll(() => {
  fs.rmSync(SITE_DIR, { recursive: true, force: true });
});

test("threads_observed = 1 and 4 across two isolated contexts; init-time fixing holds", async ({
  browser,
}) => {
  const server = createStaticServer(SITE_DIR, { isolate: true });
  const port = await listenEphemeral(server);
  try {
    // -- Context requesting 1 thread. --
    const ctx1 = await browser.newContext();
    const page1 = await ctx1.newPage();
    await page1.goto(`http://127.0.0.1:${port}/thread-probe.html?numThreads=1`);
    await page1.waitForFunction(() => window.__THREAD_DONE__ === true);
    const err1 = await page1.evaluate(() => window.__THREAD_ERROR__ || null);
    expect(err1).toBeNull();
    const r1 = await page1.evaluate(() => window.__THREAD_RESULT__);
    expect(r1.requested).toBe(1);
    expect(r1.threadsObserved).toBe(1);

    // Setting numThreads = 4 AFTER the first session-create, then creating a SECOND
    // session in the SAME (1-thread-initialised) page, still reports 1 -- the thread
    // pool is fixed at first init, not re-read per session.
    const second = await page1.evaluate(() => window.__createAndRun(4));
    expect(second.threadsObserved).toBe(1);
    await ctx1.close();

    // -- Context requesting 4 threads (fresh context -> fresh wasm module instance). --
    const ctx4 = await browser.newContext();
    const page4 = await ctx4.newPage();
    await page4.goto(`http://127.0.0.1:${port}/thread-probe.html?numThreads=4`);
    await page4.waitForFunction(() => window.__THREAD_DONE__ === true);
    const err4 = await page4.evaluate(() => window.__THREAD_ERROR__ || null);
    expect(err4).toBeNull();
    const r4 = await page4.evaluate(() => window.__THREAD_RESULT__);
    expect(r4.requested).toBe(4);
    expect(r4.threadsObserved).toBe(4);
    await ctx4.close();
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("--no-isolation: a request for 4 threads reports 1 (no SharedArrayBuffer)", async ({ browser }) => {
  const server = createStaticServer(SITE_DIR, { isolate: false });
  const port = await listenEphemeral(server);
  try {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await page.goto(`http://127.0.0.1:${port}/thread-probe.html?numThreads=4`);
    await page.waitForFunction(() => window.__THREAD_DONE__ === true);
    const isolated = await page.evaluate(() => self.crossOriginIsolated === true);
    expect(isolated).toBe(false);
    const err = await page.evaluate(() => window.__THREAD_ERROR__ || null);
    // Either ORT Web falls back to a single-threaded build cleanly (threadsObserved ===
    // 1, no error), or it raises trying to spawn a SharedArrayBuffer-backed pool without
    // cross-origin isolation -- either outcome documents "no SAB -> not multi-threaded",
    // so both are accepted, but at most one may hold.
    if (err === null) {
      const r = await page.evaluate(() => window.__THREAD_RESULT__);
      expect(r.threadsObserved).toBe(1);
    } else {
      expect(err).toMatch(/SharedArrayBuffer|shared|thread/i);
    }
    await ctx.close();
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
