// Phase 0, T3: minimal static file server for the headless ORT Web smoke.
//
// Sets Cross-Origin-Opener-Policy: same-origin and
// Cross-Origin-Embedder-Policy: require-corp on every response -- the pair
// required for `self.crossOriginIsolated` to be true (AC-3 records this
// flag), independent of whether the smoke actually uses more than one
// wasm thread.
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
};

/**
 * Create (but do not start) a static HTTP server rooted at `rootDir`.
 *
 * @param {string} rootDir absolute path to serve at "/"
 * @returns {http.Server}
 */
export function createStaticServer(rootDir) {
  const root = path.resolve(rootDir);
  return http.createServer((req, res) => {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
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
      res.writeHead(200, {
        "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cache-Control": "no-store",
      });
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
