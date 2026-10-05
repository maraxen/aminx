// Phase 2a, T5a: static file server for the browser layer-(c) harness.
//
// Extends browser/smoke/serve.mjs (Phase 0 T3) with two things that harness needed not
// have:
//   1. An `isolate` toggle. Isolated (default) sets Cross-Origin-Opener-Policy:
//      same-origin and Cross-Origin-Embedder-Policy: require-corp -- required for
//      `self.crossOriginIsolated` and SharedArrayBuffer-backed multi-threaded wasm
//      (Definitions, "wasm-init thread assertion"). `isolate: false` omits both headers
//      -- the `--no-isolation` negative control (AC-C1: "a no-COOP/COEP negative control
//      yields crossOriginIsolated === false"; Definitions' thread test: "with
//      --no-isolation, a request for 4 reports 1 (no SAB)").
//   2. `POST /results/<name>` (only meaningful when `resultsDir` is given): reads the
//      raw request body and writes it verbatim to `<resultsDir>/<sanitized name>`. The
//      name is sanitized to its basename with any non `[A-Za-z0-9._-]` character
//      stripped, so a crafted `name` cannot escape `resultsDir` (path-sanitised name,
//      per the task spec).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".map": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".bin": "application/octet-stream",
};

/**
 * Sanitize an untrusted result filename to a bare basename made only of
 * `[A-Za-z0-9._-]`, so it cannot escape `resultsDir` via `..` or an absolute path.
 *
 * @param {string} name
 * @returns {string}
 */
export function sanitizeResultName(name) {
  const base = path.basename(String(name || ""));
  const cleaned = base.replace(/[^A-Za-z0-9._-]/g, "_");
  return cleaned || "unnamed";
}

/**
 * Create (but do not start) a static HTTP server rooted at `rootDir`, optionally also
 * accepting `POST /results/<name>` writes into `resultsDir`.
 *
 * @param {string} rootDir absolute path to serve at "/"
 * @param {{isolate?: boolean, resultsDir?: string|null}} [opts]
 * @returns {http.Server}
 */
export function createStaticServer(rootDir, opts = {}) {
  const isolate = opts.isolate !== false;
  const resultsDir = opts.resultsDir || null;
  const root = path.resolve(rootDir);

  return http.createServer((req, res) => {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);

    if (req.method === "POST" && urlPath.startsWith("/results/") && resultsDir) {
      const name = sanitizeResultName(urlPath.slice("/results/".length));
      const chunks = [];
      req.on("data", (chunk) => chunks.push(chunk));
      req.on("end", () => {
        try {
          fs.mkdirSync(resultsDir, { recursive: true });
          fs.writeFileSync(path.join(resultsDir, name), Buffer.concat(chunks));
          res.writeHead(200, { "Content-Type": "text/plain" }).end("ok");
        } catch (err) {
          res.writeHead(500, { "Content-Type": "text/plain" }).end(String(err));
        }
      });
      return;
    }

    const rel = urlPath === "/" ? "/index.html" : urlPath;
    const filePath = path.join(root, rel);
    if (!filePath.startsWith(root)) {
      res.writeHead(403, { "Content-Type": "text/plain" }).end("forbidden");
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404, { "Content-Type": "text/plain" }).end(`not found: ${rel}`);
        return;
      }
      const ext = path.extname(filePath);
      const headers = {
        "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
        "Cache-Control": "no-store",
      };
      if (isolate) {
        headers["Cross-Origin-Opener-Policy"] = "same-origin";
        headers["Cross-Origin-Embedder-Policy"] = "require-corp";
      }
      res.writeHead(200, headers);
      res.end(data);
    });
  });
}

/**
 * Start `server` on an ephemeral loopback port and resolve with the port.
 *
 * @param {http.Server} server
 * @returns {Promise<number>}
 */
export function listenEphemeral(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve(typeof address === "object" && address ? address.port : 0);
    });
  });
}
