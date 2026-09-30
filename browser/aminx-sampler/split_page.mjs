// In-page driver: runs the four-graph split in a real browser.
//
// Loaded by split_index.html and driven by browser/layer_c/run_p07.mjs, whose protocol
// this deliberately reuses rather than inventing a parallel one: set
// `window.__PARITY_DONE__` when finished, put the payload on `window.__PARITY_RESULT__`,
// and any failure on `window.__PARITY_ERROR__`. Reusing that contract means the existing
// driver -- which already serves with COOP/COEP, captures console errors, and records
// Chromium and Playwright versions -- needs no changes at all.
//
// This grades NOTHING. It runs the loop and reports outputs; every comparison and
// threshold lives in the tracked Python gate. The page is the artifact under test, so it
// must not be the thing deciding whether its own numbers are acceptable.
//
// `numThreads` comes from the query string so the same page serves the single-threaded
// and multithreaded measurements. Threaded wasm is exactly why this must run over HTTP:
// above one thread ORT-Web spawns Workers that fetch the wasm binary, which fails from a
// file:// path (run 98a62501).

import * as ort from "./ort/ort.wasm.min.mjs";

import { runSplitDecode } from "./split_loop.mjs";

const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");

function typedFrom(spec) {
  const { dtype, shape, data } = spec;
  if (dtype === "float32") return { shape, data: Float32Array.from(data) };
  if (dtype === "int32") return { shape, data: Int32Array.from(data) };
  if (dtype === "bool") return { shape, data: Uint8Array.from(data) };
  throw new Error(`unsupported dtype ${dtype}`);
}

function toTensor(t) {
  const { shape, data } = t;
  if (data instanceof Float32Array) return new ort.Tensor("float32", data, shape);
  if (data instanceof Int32Array) return new ort.Tensor("int32", data, shape);
  if (data instanceof Uint8Array) return new ort.Tensor("bool", data, shape);
  throw new Error(`unsupported tensor data ${data?.constructor?.name}`);
}

const fromTensor = (t) => ({ shape: Array.from(t.dims), data: t.data });

async function runPositional(session, tensors) {
  const feeds = {};
  session.inputNames.forEach((n, i) => {
    feeds[n] = tensors[i];
  });
  const out = await session.run(feeds);
  return session.outputNames.map((n) => fromTensor(out[n]));
}

async function main() {
  // ABSOLUTE, not "./ort/". ORT resolves wasmPaths relative to its own module URL,
  // not the page, so a relative "./ort/" from /ort/ort.wasm.min.mjs resolves to
  // /ort/ort/ and every sibling .wasm 404s with an unhelpful "no available backend".
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = numThreads;
  ort.env.logLevel = "error";

  const cells = await (await fetch("./cells.json")).json();
  const bucket = cells.bucket;

  const sessions = {};
  const sessionCreateMs = {};
  for (const key of ["encoder", "wave", "decoder", "fuse"]) {
    const buf = await (await fetch(`./models/p07_${key}_L${bucket}.onnx`)).arrayBuffer();
    const t0 = performance.now();
    sessions[key] = await ort.InferenceSession.create(new Uint8Array(buf), {
      executionProviders: ["wasm"],
    });
    sessionCreateMs[key] = performance.now() - t0;
  }

  const results = [];
  for (const cell of cells.cells) {
    const a = Object.fromEntries(Object.entries(cell.arrays).map(([k, v]) => [k, typedFrom(v)]));
    const callbacks = {
      runEncoder: async (i) => {
        const [node_features, edge_features, neighbor_indices] = await runPositional(
          sessions.encoder,
          [i.coords, i.mask, i.residue_index, i.chain_index].map(toTensor),
        );
        return { node_features, edge_features, neighbor_indices };
      },
      runWave: async (i) => {
        const o = await runPositional(
          sessions.wave,
          [i.decoding_order, i.tie_group_map].map(toTensor),
        );
        return {
          group_ids: o[0], group_positions: o[1], group_valid: o[2], position_valid: o[3],
          ar_mask: o[4], group_first_rank: o[5], pos_first_rank: o[6],
        };
      },
      runDecoder: async (i) => {
        const [logits] = await runPositional(sessions.decoder, [
          i.node_features, i.edge_features, i.neighbor_indices, i.mask, i.ar_mask, i.sequence_oh,
        ].map(toTensor));
        return { logits };
      },
      runFuse: async (i) => {
        const [final_token, avg_stored] = await runPositional(sessions.fuse, [
          i.logits, i.cond_bias, i.mask_group, i.fixed_mask, i.fixed_tokens, i.group_id,
          i.temperature, i.gumbel_noise,
        ].map(toTensor));
        return { final_token, avg_stored };
      },
    };

    const t0 = performance.now();
    const { tokens, logProbs } = await runSplitDecode(callbacks, {
      length: a.mask.shape[0],
      encoderInputs: {
        coords: a.coords, mask: a.mask,
        residue_index: a.residue_index, chain_index: a.chain_index,
      },
      waveInputs: { decoding_order: a.decoding_order, tie_group_map: a.tie_group_map },
      tieGroupMap: a.tie_group_map,
      mask: a.mask,
      condBias: a.bias,
      fixedMask: a.fixed_mask,
      fixedTokens: a.fixed_tokens,
      temperature: a.temperature,
      gumbelNoise: a.gumbel_noise,
    });
    results.push({
      name: cell.name,
      wall_ms: performance.now() - t0,
      tokens: Array.from(tokens.data),
      log_probs: Array.from(logProbs.data),
    });
  }

  for (const k of Object.keys(sessions)) await sessions[k].release();

  return {
    bucket,
    n_cells: results.length,
    // What ORT actually used, not what was asked for: a silent fallback to one thread
    // must show up as data rather than be mistaken for a threading result.
    num_threads_effective: ort.env.wasm.numThreads,
    num_threads_requested: numThreads,
    cross_origin_isolated: Boolean(window.crossOriginIsolated),
    user_agent: navigator.userAgent,
    session_create_ms: sessionCreateMs,
    cells: results,
  };
}

main()
  .then((res) => {
    window.__PARITY_RESULT__ = res;
    window.__PARITY_DONE__ = true;
  })
  .catch((err) => {
    window.__PARITY_ERROR__ = String((err && err.stack) || err);
    window.__PARITY_DONE__ = true;
  });
