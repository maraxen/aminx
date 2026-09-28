// Phase 2a, T8: one Chromium session, two contexts, JSON-lines stdin/stdout.
//
// Protocol (one JSON object per line):
//   → {"cmd":"init"}
//   ← {"cmd":"init","ok":true,"sessionId","hardwareConcurrency","threadsN",
//      "chromiumVersion","contexts":[...],"crossOriginIsolated"}
//   → {"cmd":"time","context":"ctx_t1"|"ctx_tN","spec":{...}}
//   ← {"cmd":"time","ok":true,"context","result":{...}}
//   → {"cmd":"close"}
//   ← {"cmd":"close","ok":true}
//
// Headless Chromium only. Exactly two contexts are created during init and kept
// for the life of the process (one Python driver process = one session).
import readline from "node:readline";
import { chromium } from "@playwright/test";
import { createStaticServer, listenEphemeral } from "./serve.mjs";

function parseArgs(argv) {
  let site = null;
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--site") {
      site = argv[i + 1];
      i += 1;
    }
  }
  if (!site) {
    throw new Error("usage: node run_bench.mjs --site <dir>");
  }
  return { site };
}

function writeLine(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

async function openContext(browser, port, name, numThreads) {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:${port}/index.html?numThreads=${numThreads}`, {
    waitUntil: "load",
  });
  await page.waitForFunction(() => window.__BENCH_READY__ === true);
  const info = await page.evaluate(() => window.__benchInfo());
  return { name, context, page, numThreads, info };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const server = createStaticServer(args.site, { isolate: true, resultsDir: null });
  const port = await listenEphemeral(server);
  let browser = null;
  const pages = {};
  let sessionId = null;

  async function onInit() {
    browser = await chromium.launch({ headless: true });
    const crypto = await import("node:crypto");
    sessionId = crypto.randomUUID();
    const probe = await openContext(browser, port, "ctx_t1", 1);
    const hardware = Number(probe.info.hardwareConcurrency) || 1;
    const threadsN = Math.min(8, hardware);
    pages.ctx_t1 = probe;
    pages.ctx_tN = await openContext(browser, port, "ctx_tN", threadsN);
    const contexts = [pages.ctx_t1, pages.ctx_tN].map((entry) => ({
      name: entry.name,
      threadsRequested: entry.numThreads,
      hardwareConcurrency: entry.info.hardwareConcurrency,
      crossOriginIsolated: entry.info.crossOriginIsolated,
      numThreadsReadback: entry.info.numThreadsReadback,
    }));
    const isolated = contexts.every((entry) => entry.crossOriginIsolated === true);
    writeLine({
      cmd: "init",
      ok: true,
      sessionId,
      hardwareConcurrency: hardware,
      threadsN,
      chromiumVersion: browser.version(),
      crossOriginIsolated: isolated,
      contexts,
    });
  }

  async function onTime(msg) {
    const entry = pages[msg.context];
    if (!entry) {
      writeLine({ cmd: "time", ok: false, error: `unknown context ${msg.context}` });
      return;
    }
    const result = await entry.page.evaluate(async (spec) => window.__timeCell(spec), msg.spec);
    writeLine({ cmd: "time", ok: true, context: msg.context, result });
  }

  async function onClose() {
    if (browser) {
      await browser.close();
      browser = null;
    }
    await new Promise((resolve) => server.close(resolve));
    writeLine({ cmd: "close", ok: true });
  }

  const rl = readline.createInterface({ input: process.stdin });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch (err) {
      writeLine({ cmd: "error", ok: false, error: `bad json: ${err}` });
      continue;
    }
    try {
      if (msg.cmd === "init") {
        await onInit();
      } else if (msg.cmd === "time") {
        await onTime(msg);
      } else if (msg.cmd === "close") {
        await onClose();
        break;
      } else {
        writeLine({ cmd: msg.cmd || "error", ok: false, error: "unknown cmd" });
      }
    } catch (err) {
      writeLine({
        cmd: msg.cmd || "error",
        ok: false,
        error: String((err && err.stack) || err),
      });
    }
  }
}

main().catch((err) => {
  writeLine({ cmd: "fatal", ok: false, error: String((err && err.stack) || err) });
  process.exit(1);
});
