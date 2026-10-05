// Phase 2a, T5a: Playwright headless-Chromium driver for the browser layer-(c) harness.
//
// Usage: node run_parity.mjs --site <dir> --out-dir <results-dir> [--num-threads N]
//                             [--no-isolation] [--timeout-ms N]
//
// Serves `--site` (index.html + parity.mjs + a copied onnxruntime-web dist + the
// model/data files layer_c_common.py assembled) with serve.mjs, navigates headless
// Chromium to it with `?numThreads=<--num-threads>`, waits for parity.mjs to set
// `window.__PARITY_DONE__`, and writes the collected result (plus the Chromium build
// version) to `<out-dir>/harness.json`. Per-cell outputs and evidence.json land in
// `--out-dir` via parity.mjs's own `POST /results/<name>` calls (serve.mjs).
//
// Exits non-zero ONLY if the harness itself failed to produce a result (browser launch,
// page crash, navigation error, or timeout) -- a per-cell `ok: false` inside the result
// is NOT a harness failure and is reported as data (result-emission rule: the Python
// driver turns any harness/page state into a result row; it does not propagate this
// exit code).
import fs from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";
import { createStaticServer, listenEphemeral } from "./serve.mjs";

function parseArgs(argv) {
  const args = {
    site: null,
    outDir: null,
    timeoutMs: 60_000,
    numThreads: 1,
    isolate: true,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const tok = argv[i];
    if (tok === "--site") {
      args.site = argv[++i];
    } else if (tok === "--out-dir") {
      args.outDir = argv[++i];
    } else if (tok === "--timeout-ms") {
      args.timeoutMs = Number(argv[++i]);
    } else if (tok === "--num-threads") {
      args.numThreads = Number(argv[++i]);
    } else if (tok === "--no-isolation") {
      args.isolate = false;
    }
  }
  if (!args.site || !args.outDir) {
    throw new Error(
      "usage: node run_parity.mjs --site <dir> --out-dir <results-dir> [--num-threads N] [--no-isolation]",
    );
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  fs.mkdirSync(args.outDir, { recursive: true });
  const server = createStaticServer(args.site, { isolate: args.isolate, resultsDir: args.outDir });
  const port = await listenEphemeral(server);

  let browser;
  const harness = {
    harnessOk: false,
    harnessError: null,
    chromiumVersion: null,
    playwrightVersion: null,
    isolate: args.isolate,
    numThreadsRequested: args.numThreads,
    result: null,
  };
  try {
    browser = await chromium.launch({ headless: true });
    harness.chromiumVersion = browser.version();
    try {
      const pkg = JSON.parse(
        fs.readFileSync(
          new URL("./node_modules/@playwright/test/package.json", import.meta.url),
          "utf8",
        ),
      );
      harness.playwrightVersion = pkg.version;
    } catch {
      harness.playwrightVersion = null;
    }

    const page = await browser.newPage();
    const consoleErrors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));

    await page.goto(`http://127.0.0.1:${port}/index.html?numThreads=${args.numThreads}`, {
      waitUntil: "load",
    });
    await page.waitForFunction(() => window.__PARITY_DONE__ === true, undefined, {
      timeout: args.timeoutMs,
    });
    const outcome = await page.evaluate(() => ({
      result: window.__PARITY_RESULT__ || null,
      error: window.__PARITY_ERROR__ || null,
    }));
    if (outcome.error) {
      harness.harnessOk = false;
      harness.harnessError = outcome.error;
    } else {
      harness.harnessOk = true;
      harness.result = outcome.result;
    }
    if (consoleErrors.length > 0) {
      harness.consoleErrors = consoleErrors;
    }
  } catch (err) {
    harness.harnessOk = false;
    harness.harnessError = String((err && err.stack) || err);
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }

  const outPath = path.join(args.outDir, "harness.json");
  fs.writeFileSync(outPath, JSON.stringify(harness, null, 2));
  process.stdout.write(`${JSON.stringify(harness)}\n`);
  process.exit(harness.harnessOk ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`run_parity.mjs fatal: ${(err && err.stack) || err}\n`);
  process.exit(1);
});
