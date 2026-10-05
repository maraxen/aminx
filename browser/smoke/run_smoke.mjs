// Phase 0, T3: Playwright headless-Chromium driver for the ORT Web smoke.
//
// Usage: node run_smoke.mjs --site <dir> --out <result.json>
//
// Serves `--site` (index.html + smoke.mjs + a copied onnxruntime-web dist +
// the model/data files ort_web_smoke.py assembled) with the COOP/COEP
// headers from serve.mjs, navigates headless Chromium to it, waits for
// smoke.mjs to set `window.__SMOKE_DONE__`, and writes the collected result
// (plus the Chromium build version) to `--out` as JSON. Exits non-zero only
// if the harness itself failed to produce a result (browser launch, page
// crash, navigation error, or timeout) -- a per-model `ok: false` inside the
// result is NOT a harness failure and is reported as data.
import fs from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";
import { createStaticServer, listenEphemeral } from "./serve.mjs";

function parseArgs(argv) {
  const args = { site: null, out: null, timeoutMs: 60_000 };
  for (let i = 0; i < argv.length; i += 1) {
    const tok = argv[i];
    if (tok === "--site") {
      args.site = argv[++i];
    } else if (tok === "--out") {
      args.out = argv[++i];
    } else if (tok === "--timeout-ms") {
      args.timeoutMs = Number(argv[++i]);
    }
  }
  if (!args.site || !args.out) {
    throw new Error("usage: node run_smoke.mjs --site <dir> --out <result.json>");
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const server = createStaticServer(args.site);
  const port = await listenEphemeral(server);

  let browser;
  const harness = {
    harnessOk: false,
    harnessError: null,
    chromiumVersion: null,
    playwrightVersion: null,
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

    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: "load" });
    await page.waitForFunction(
      () => window.__SMOKE_DONE__ === true,
      undefined,
      { timeout: args.timeoutMs },
    );
    const outcome = await page.evaluate(() => ({
      result: window.__SMOKE_RESULT__ || null,
      error: window.__SMOKE_ERROR__ || null,
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

  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(harness, null, 2));
  process.stdout.write(`${JSON.stringify(harness)}\n`);
  process.exit(harness.harnessOk ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`run_smoke.mjs fatal: ${(err && err.stack) || err}\n`);
  process.exit(1);
});
