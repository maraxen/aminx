# aminx ProteinMPNN, in the browser

A static, bundler-free site that runs the P07 ProteinMPNN sampler entirely
client-side via `onnxruntime-web`. See `../docs/browser_integration.md` for
the underlying sampler API, structure/RunSpec formats, and hosting
requirements this site is built on top of -- this file only covers the site
itself.

## Local dev

Serve `site/` (or a built `dist/`, see below) over plain HTTP -- ES modules
and module Web Workers need `http(s)://`, not `file://`. Either works:

```bash
# plain, single-threaded wasm (no COOP/COEP headers)
python3 -m http.server 8000 --directory site

# cross-origin isolated, multithreaded wasm -- sets the headers
# browser_integration.md's "Hosting requirements" section describes
node browser/layer_c/serve.mjs --dir site --port 8000
```

Then open `http://localhost:8000/`. You'll need model weights in
`site/models/` (or wherever `site/config.js`'s `MODEL_BASE` points) --
see below.

## Where the models come from

`p07_sample_L128.onnx` and `p07_sample_L256.onnx` are built by
`scripts/browser_validation/p07_knobs_gate.py`'s export step (see
`../docs/browser_integration.md`). They are not checked into this repo.
`tools/build_site.py --models-dir <dir>` copies them into the built site and
writes `dist/models/MANIFEST.json` with each file's sha256 + size, so a
deployment's weights are always independently verifiable against that
manifest. A stray `<name>.onnx.data` beside them is stale (the export embeds
its weights into one self-contained `.onnx` file) and is never copied.

## Threads / COOP+COEP

`onnxruntime-web`'s wasm execution provider only runs multithreaded under
cross-origin isolation (`crossOriginIsolated === true`, which needs the
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` response headers). Without
them the page still works, just single-threaded (`numThreads: 1`) -- see the
"threads: N" note under Settings, and `index.html`'s commented-out
coi-serviceworker hook if you want isolation on a host (like GitHub Pages)
that cannot set response headers itself.

## Reusing just the sampler

Everything under `site/` is this demo UI. If you only want the sampler
itself -- e.g. to embed ProteinMPNN sampling into another page, such as
ProteinHunter -- you need exactly two files plus a Worker:

- `browser/aminx-sampler/aminx_sampler.mjs` (+ its sibling
  `runspec_core.mjs`, imported relatively)
- `site/worker.js` as a template for running it off the main thread (module
  Web Worker, one cached `createSampler` session per length bucket, release
  on teardown)

`site/pdb_parse.mjs` and `site/design_utils.mjs` are optional conveniences
(PDB/mmCIF parsing, RunSpec text-field parsing, scoring) -- reuse them or
write your own; neither one imports the sampler, so they carry no coupling
to how you run inference.

## Validation status

See `../docs/browser_integration.md`'s "Validation status" table for what is
and is not verified about the underlying sampler (browser-vs-native parity,
the RunSpec knobs gate, distributional comparison against reference
ProteinMPNN). This site adds a UI on top of that sampler; it does not add
new validation of the sampler's numerics.
