---
title: "S5: hub - static site, catalog, executors, viewer (working slug aminx-hub)"
description: "A hub project separate from the aminx package (aminx keeps its name; working slug aminx-hub, final repo name, PyPI policy and domain are the S3-01 user decision): a static GitHub Pages site whose catalog is generated from S4 manifests, with ORT-Web, Pyodide, localfold and remote executors and py2Dmol as the viewer."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - aminx-hub
  - aminx
  - isochore
  - py2Dmol
  - localfold
  - xtrax
related_specs:
  - S1  # RunSpec unification (optional consumer: RunSpec JSON schema)
  - S2  # EBM extraction (future catalog source, not v1)
  - S3  # aminx identity kept: hub identity ADR (S3-01), release guard, stale-name checker, custom domain
  - S4  # xtrax model contract (manifests, ports, graph IR, ONNX route)
  - S6  # pipeline editor component
---

# S5: hub - static site, catalog, executors, viewer

**Naming (no-rename revision, 261001).** `aminx` always means the MPNN package: PyPI `aminx`, import `aminx`, repo
`maraxen/aminx`. It keeps its name (revised D2); there is no rename, shim or cutover. "The hub" means the separate
project this spec builds. Its working slug is `aminx-hub`, used as a placeholder in examples, paths, sidecar
commands and the bathos slug. The final repo name (the user also floated a name derived from the domain
`praxia.science`; a Pages custom domain works with any repo slug), the PyPI policy (S3 recommends none) and the
domain are the user decision S3-01. The hub never takes the name `aminx`. **Flag:** a hub repo, PyPI, npm, CLI or
bathos name containing `praxia` would share a namespace with the user's praxia agent orchestrator; S3-01 carries
the collision policy, and a DNS name under `praxia.science` does not collide.

## 2. Goal and non-goals

**Goal.** One static site, a project separate from the aminx package (D1, revised D2) and deployed to GitHub Pages under the identity S3-01 records (working slug `aminx-hub`), that lists models
and tools from their published S4 manifests, runs each on the best executor the visitor's browser (or
own compute) can offer, and shows results in py2Dmol. The hub imports **no Python model package**: its
build reads JSON manifests and released bytes only.

**Non-goals.**
- No hub-operated compute, accounts, job queue, server-side storage, or telemetry. The hub is files.
- No absorbing model code into the hub (D1). Model-specific browser code (host prep, RunSpec mirror)
  stays in the model repo and reaches the hub as a hash-pinned release asset (the *driver module*).
- No JAX in the browser. JAX-in-Pyodide is not available (A9); JAX models reach the browser only via
  S4's ONNX route (ORT-Web) or a remote executor.
- No hosting of third-party weights whose license forbids it (AF3 params, see section 5).
- No EBM in v1 (S2 may publish a manifest later; the catalog would accept it unchanged).
- No hub dependency edge for proteinsmc (D6): the hub is the *dependent* party; it never asks a model
  repo to import it, and a model enters the catalog only by publishing a manifest.
- No rename of aminx and no use of the name `aminx` by the hub (repo, PyPI project, importable package or npm
  name): S3-24's reserved-name mode runs in the hub's CI (S5-01). No S5 item touches aminx's PyPI project, HF
  repo or consumers.
- This spec executes nothing now (D7); items are DAG nodes.

## 3. Current state (anchored; tagged verified / unverified)

**Verified by reading, 2026-10-01** (file:line, or an external read named with tool and target):

- The existing browser demo has **never been deployed**: `gh run list --workflow pages.yml` returns `[]`;
  `pages.yml:8` is `workflow_dispatch` only; zero repository variables are set (`gh api
  .../actions/variables`), so `AMINX_MODELS_RELEASE_TAG` / `AMINX_MODEL_BASE_URL` (`pages.yml:41,60`)
  are both empty. There is no URL back-compat obligation for the demo.
- `maraxen.github.io/aminx/` **already serves something else**: Pages is `build_type: legacy`, source
  branch `gh-pages`, containing a Sphinx build whose last commit is 2025-11-10; no workflow in
  `.github/workflows/` produces it. That URL belongs to the aminx repo, which keeps its name; the hub does not inherit it (a hub project site lives under its own `/<slug>/`, never `/aminx/`).
- The site loads ORT-Web 1.30.0 wasm from jsdelivr (`site/worker.js:11,14`, `site/index.html:28`),
  py2Dmol from `py2dmol.solab.org` (`site/app.js:25`), and wasm-only EP (`aminx_sampler.mjs:167`).
  Threads: `isolated ? min(4, cores) : 1` (`site/app.js:81`); GitHub Pages cannot send COOP/COEP, and
  `site/index.html:17-22` only *documents* a coi-serviceworker hook, it is not wired.
- The site uses the **monolith** graphs (`MODEL_FILES = p07_sample_L128/L256.onnx`,
  `tools/build_site.py:67`; `site/worker.js:12` `createSampler`). The release **v0.2.0a3 carries only
  the split graphs** plus `MANIFEST.json` (`gh release view v0.2.0a3`: decoder/encoder/fuse/wave for
  L128 and L256, no `p07_sample_*`; `release/browser/` also holds `p07_monolith_*` under a different
  name). The split is 6.64 / 6.71 MB against the monolith's 19.43 / 31.52 MB
  (`docs/browser_integration.md:249-250`). The site as written cannot be fed from the release.
- Browser scope of aminx today is **ProteinMPNN sampling only** (`checkpoint proteinmpnn_v_48_020`);
  LigandMPNN, membrane and other variants are not exported (`docs/browser_integration.md:26-28`).
- `release/browser/MANIFEST.json` already records, per graph, `file, bytes, sha256, input_shapes,
  input_dtypes`, plus top-level `alphabet, checkpoint_id, git_hash`. This is the de-facto manifest S4
  generalises.
- The repo's JS tests (`site/tests/*.test.mjs`, `browser/aminx-sampler/*.test.mjs`) are **not run in CI**:
  the only node reference in `.github/workflows/` is `ci.yml:43` (`npm install -g @ast-grep/cli`).
- xtrax v0.4.0a11 `export/onnx.py` (module docstring): jax2onnx leaks a `jnp` patch (restored by
  `convert_to_onnx`); **JAX RNG is refused for the onnx target** (RandomUniform is attribute-seeded, so
  noise must be an input, which is why the sampler's Gumbel noise is made in JS); int64 index graphs
  matter because ORT-Web's WebGPU EP has none; `targets.py` states the ONNX target is executed on
  **ORT CPU only** - "the same file on ORT Web (wasm or WebGPU) is not executed here". Browser-side
  verification is therefore the hub's job (S5-15).
- **Weights hosting is not "point at a release URL".** Measured with `curl -H 'Origin:
  https://maraxen.github.io'` on 2026-10-01: GitHub release assets (302 to
  `release-assets.githubusercontent.com`, then 206 on a ranged GET) carry **no
  `Access-Control-Allow-Origin`**, so a Pages page cannot `fetch()` them. Hugging Face `resolve/`
  returns `ACAO: <origin>` on the 302 and `ACAO: *` on the CDN 206 for a real LFS file
  (`maraxen/aminx` `proteinmpnn_v_48_020.eqx.zst`). jsdelivr: `ACAO: *`, `CORP: cross-origin`.
  `py2dmol.solab.org`: `ACAO: *`, no CORP header.
  This invalidates `pages.yml`'s "point the built site at a remote model base (e.g. a release asset URL)"
  branch for browser fetch.
- localfold (`~/repos/localfold`, HEAD ab5bd38, npm `localfold` 0.0.1, Beerware code):
  - `package.json` `exports` = `.` (`src/index.js`), `./node`, `./src/*`. `src/index.js` re-exports the
    **AF2 kernels and runtime only**; there is no `foldAf3`, MPNN or ESMFold2 export. `docs/RUNNING.md:82-84`:
    "the `0.0.x` version is meant literally: treat it as unstable until a single sequence-in-structure-out
    entry point exists." The AF3 entry is `foldBatch(device, batch, weights, options)` (`src/af3/fold.js:641`),
    below featurisation.
  - **Its MPNN is not WebGPU.** `rg 'navigator.gpu|GPUDevice|createBuffer|WebGPU' src/design` matches
    only a comment in `hunter-loop.js`; `src/design/mpnn/accel.js` is an optional **wasm-SIMD** dense kernel
    over a JS reference (`ops.js`). It is a **mirror** of an external checkout (`SOURCE.md`: upstream
    `/Users/mini/Documents/GitHub/mpnn/mpnn` @ a80942f, "DO NOT EDIT"), ships float16 `.mpnn` weights
    (3.3-5.3 MB each) for four families (`designers.js:29-52`: soluble, protein, ligand, NA-MPNN), and
    `web/public/mpnn/` holds them. This corrects the brief's "its own WebGPU ProteinMPNN" (D5 text).
  - **No cross-window API.** The only `postMessage` in `src/` and `web/` is an internal scheduler yield
    (`src/runtime/yield.js:35`); URL params read are `model` and `graph` (`web/app.js:354,4554`). An
    iframe handoff is not possible without an upstream change. A Colab bridge exists
    (`web/colab-bridge.js`): the runtime runs the *same page* headlessly and a broker relays events;
    the token arrives in the URL and is then dropped with `replaceState`.
  - Wire formats that do exist: AF3 job JSON, both dialects (`web/job-json.js` header), and the
    AF3-server-layout `.zip` ("Download All"; dropping it back re-folds, `README.md:41-49`).
  - Privacy statement to emulate (`README.md:27-39`): the model runs in the browser; only MSA search
    (ColabFold server) and template fetch (RCSB / AFDB) use the network, only when asked; weights are
    fetched once and cached. Licenses (`README.md:109-117,127-135`): AF2 params CC BY 4.0; **AF3 params
    carry DeepMind's Prohibited Use Policy and are not redistributed**; ESM-C, ESMFold2, OpenBind-0,
    OpenDDE, Boltz-2, Protenix-v2 under their own; "a bundle names the Hugging Face revision it fetches
    from". Sizes (`docs/HOSTING.md:3-5`): AF2 monomer 227 MB, AF3 150 MB; Pages publishes at most ~1 GB.
- py2Dmol: `main` is 67cbe73; `plugin-system-impl` is main + 5 commits (acb1cfc tip), and
  `git merge-base main plugin-system-impl` equals main, so the merge is a **fast-forward**. It adds
  `src/parts/plugins.js` (`PLUGIN_API_VERSION = 1` at :27, registry `window.py2dmolPlugins`), the `volume`
  plugin (`py2Dmol/resources/plugins/volume.js`, "shipped as written, in NO bundle"), rebuilt tracked
  bundles (`embed.min.js` 664,186 -> 686,221 bytes); 38 files changed (+7,056/-55) including the new tests. Latest tag v1.6.5; license Beerware.
- Pyodide (npm `pyodide` 314.0.7, `pyodide-lock.json` read from the tarball): Python 3.14.2, wasm32, 357
  packages. **Present:** numpy 2.4.6, scipy 1.18.0, h5py 3.13.0, scikit-learn, pandas, micropip, ml-dtypes.
  **Absent:** jax, jaxlib, onnx, onnxruntime, mdtraj, torch. PyPI `jaxlib` JSON (43 releases, 845 files,
  latest 0.11.2): **zero** emscripten/pyodide/wasm wheels. This settles the brief's unverified "JAX cannot run
  under Pyodide" as *no, with the packaged and PyPI routes*; building jaxlib for wasm yourself is out of scope.
- **isochore is not numpy-only at package level.** `pyproject.toml` requires `numpy, mdtraj, scipy, h5py`
  (Apache-2.0). `import isochore` does not import mdtraj (lazy, inside functions in `gfe.py` and
  `traj_cache.py`), and `grid_io.py`, `hotspots.py`, `kde_density.py` need only numpy/scipy. So a Pyodide
  executor can run `.dx` read/write, density smoothing and hotspot validation; **trajectory ingestion
  (mdtraj) cannot run in Pyodide**, and a `micropip` install must pass `deps=False` unless the dependency
  declaration changes (S5-16).
- License facts: aminx code MIT (`LICENSE`); `dauparas/ProteinMPNN` and `dauparas/LigandMPNN` repos are
  MIT (GitHub license API); coi-serviceworker 0.1.7 is MIT (npm). Whether the *weights* are covered for
  public redistribution is exactly what `docs/browser_integration.md:403` and `pages.yml:31-36` say must be
  confirmed first; it stays a user sign-off (S5-07).
- The hub bathos convention: global CLAUDE.md requires `bth run` (not `uv run bth`) and the wrapped form
  `bth run --project-slug <slug> -- uv run --no-sync python3 scripts/foo.py`; a `bash -c` wrapper breaks
  outcome resolution (memory: feedback_bth-wrapper-breaks-outcome).
- **isochore's tool signatures do not compose into a pipeline** (A19). `read_dx(path) -> (data, origin, delta)`
  takes a *path* (`grid_io.py:80`) and returns a **GFE** grid. `smooth_density(counts, spacing_angstrom,
  sigma_angstrom)` takes a pooled voxel-**count** histogram (`kde_density.py:20`), not a `read_dx` grid.
  `validate(protein, gfe_dir, ref_pdb, out_path, threshold, cutoff_ang)` takes a **directory** of
  `map_agfe_*.dx` files, a reference PDB *path* and a `KNOWN_HOTSPOTS` key, writes JSON to `out_path` and
  returns the dict (`hotspots.py:132-145,234-236`). They are three independent tools; the Pyodide executor
  must place files in the Emscripten FS (4.4).
- **aminx's driver is a multi-file graph.** `split_driver.mjs:14-16` imports `./aminx_sampler.mjs`,
  `./runspec_core.mjs` and `./split_loop.mjs`; its entry is `createSplitSampler(ort, {encoderUrl, waveUrl,
  decoderUrl, fuseUrl}, {numThreads})` (`:126`); `loadSession` (`:70`) accepts a URL **or** an
  `ArrayBuffer`/`Uint8Array`, so CacheStore bytes can replace URLs. A single `.mjs` release asset cannot
  resolve those relative imports without bundling.
- **aminx already holds ORT-Web evidence** (A21): `docs/browser_integration.md:264-265` records exact
  tokens 8/8 in headless Chromium (bathos `13e06211`) and 8/8 with 4 threads under COOP/COEP, log-probs
  bit-identical to 1 thread (`da02516e`); `:168` records the 288-cell knob gate bitwise against ORT-CPU and
  ORT-Web; `:368` warns that backends need not agree on tie order for the same file.
- **The hub's identity is a user decision S3 records, not something S5 can assume** (A22): revised D2 keeps
  `maraxen/aminx` for the MPNN package, so the hub is a separate repository. S3-01 records the repo slug (S3
  recommends `aminx-hub`; the alternative is a praxia.science-derived slug), the PyPI policy (recommended: none;
  never `aminx`), the Pages custom domain (recommended: subdomain `hub.praxia.science`, wired after the first
  deployment by S3-29), the bathos slug (recommended `aminx-hub`) and the policy for the name collision with the
  user's praxia agent orchestrator (`261001_aminx-identity-and-hub-naming.md:270-286`, inventory at `:168-181`).
  S5 therefore cannot assume the repo slug; the URL layout is `base_path`-driven (4.1).
- **Pages configuration is per repository.** The legacy `gh-pages` source and the stale Sphinx site belong to
  `maraxen/aminx`, which stays as it is; a new hub repo's Pages source starts unconfigured, and a project Pages site
  is served under `/<repo>/` (aminx's own is `/aminx/`, A1; a custom domain serves from `/`, A40).
- **py2Dmol embed API in use today:** `py2Dmol.show(el, text, {width,height})`, `py2Dmol.fetch(id)` and
  `viewer.setColor("plddt")` (`site/app.js:120,146,440`); the existing site fetches structures from
  `files.rcsb.org` (`site/app.js:142`). The plugin registry (`window.py2dmolPlugins`) is created if absent and
  a plugin script may load before or after the bundle (`py2Dmol-plugin-spec` `src/parts/plugins.js:4-60`).
- **S4's manifest as it stands (round 2, A18/A31/A32; re-cited in the convergence check).** Every `S4 :<line>`
  below is a line of `261001_xtrax-model-contract.md` at sha256 prefix `44c4d220971dc0c1` (the working-tree file read 261001;
  S4 is revised concurrently, so on any other revision re-cite by section: 4.2, 4.3, 4.5). Section 4.3 defines
  `schema_version` (`:536`), `id, kind (model|tool), role, family, version` (`:537-541`), `ports[]` (`:542-547`),
  `artifacts[]` (`:549-556`, each with `sha256`, `bytes`, `urls`, per-format `details`), `executors[]`
  (`:557-562`, with `driver {url, sha256, export}` and `determinism {rng_scheme, cross_executor_equivalent}` on the
  `onnx` entry), `license.redistribution` `allowed|gated|forbidden` (`:563-564`), `provenance` (`:565-566`),
  `validation` and `limits`. Every `hf://` URL carries a 40-hex revision and every artifact has a sha256
  (`:599-600`). `OnnxBundle@1` is a **fragment of a manifest** (4.5, `:677-684`), so the graph hashes are inside
  `artifacts[]` and there is no second document to fetch. The 4.5 table gives the `pyodide` entry as
  `wheel: URL + sha256` plus `entry` only (`:668`) and the `remote` entry `protocol: "xtrax-http@1"` (`:669`).
  `catalog_view(manifest)` is a Python function (`:604-606`) and S4-09 commits its snapshot corpus under
  `conformance/catalog_view/` (`:606-609`; S4-09 at `:1416-1422`). Section 4.2 defines the grid port types
  `energy-map` (alias `gfe-grid`), `count-grid` and `density-grid`, each refining `grid` (`:474-477`), and compat
  rule 2 rejects a `count-grid` wired to an `energy-map` (`:495-496`). **What S4 does not carry** and the hub needs:
  the four gaps G1-G4 in 4.2. My round-1 field names (`executors[]`, `artifacts[]`, `driver`, `determinism`,
  `redistribution` enum) matched S4; the round-2 objection's field names (`contract_version`, `operations[]`,
  `bindings`) were from an earlier S4 draft and are not in the file.
- **aminx's `loadSession` contains a `fetch` call.** `split_driver.mjs:70-79` calls `fetch(urlOrBuffer)` for a
  string argument; `aminx_sampler.mjs:160` does the same. A bundle of the unmodified source therefore contains
  `fetch` (A30), so the driver rule "no fetch of its own" needs a bytes-only entry (S5-14).
- **ORT-Web already loads inside a module worker in the existing site** (`site/worker.js:11,14`: `import * as ort`
  at the worker top level, `ort.env.wasm.wasmPaths` set), so the driver host can import vendored ORT itself (A29).
- **The split loop returns per-position log-probs and accepts fixed tokens**: `split_loop.mjs:104` documents
  `{tokens, logProbs}`, `:208-226` fills `logProbs[L, 21]`, `:13,186` pass `fixed_tokens` for fixed positions.
  Whether *all* positions fixed yields teacher-forced scores is not read (A36).
- **localfold's MPNN families by file name** (`src/design/designers.js:31,37,42,48`): `solublempnn_v_48_020.mpnn`,
  `proteinmpnn_v_48_020.mpnn`, `ligandmpnn_v_32_020_25.mpnn`, `na_mpnn_design.mpnn`. The ProteinMPNN file has the
  same *name* as aminx's checkpoint id; that is not evidence the weights are the same (A35).

### Scope map: what exists, where it goes

| Today (aminx @ origin/main d1210e4a) | Destination | Item |
|---|---|---|
| `site/` (demo UI, 477-line `app.js`, `pdb_parse.mjs`, `design_utils.mjs`, `site/tests/*.test.mjs`) | hub `apps/mpnn-designer/` (bespoke panel) | S5-29 |
| `browser/aminx-sampler/` (`aminx_sampler.mjs`, `runspec_core.mjs`, `split_*.mjs`); keeps its name and path | aminx, shipped as release driver module | S5-14, S5-52 |
| `browser/layer_c/` (Node CLI, parity, bench, `serve.mjs`) | stays in aminx (model validation harness); header-controlled dev server re-implemented in hub | S5-02 |
| `browser/smoke/` (Playwright 1.63.0 + ORT-Web 1.30.0 Phase-0 harness) | superseded by hub e2e harness; left in aminx as history | S5-02 |
| `tools/build_site.py` (allow-list copy, `--check`, MANIFEST) | hub catalog builder keeps its two invariants (explicit allow-list, refuse stray `.onnx.data`) | S5-04, S5-06 |
| `.github/workflows/pages.yml` (manual dispatch, repo-var weights) | hub deploy workflow (push to main, never bundles unreviewed weights) | S5-32 |
| `docs/browser_integration.md` (+ `.html`) | split: executor-agnostic parts to hub docs, MPNN specifics stay | S5-49, S5-34 |
| Browser assets of an aminx release (native manifest, split graphs, driver, golden vectors); v0.2.0a3 carries only the old `MANIFEST.json` and split graphs | A new aminx release carries the S4 manifest, driver and golden vectors; hub pins it in `catalog/sources.toml` | S5-52, S5-53 |
| `release/browser/*.onnx` (git-LFS tracked, ~10 files) | stays in aminx; hub never vendors it, hub CI copies verified bytes at build | S5-06 |
| `gh-pages` branch (stale Sphinx API docs) and its legacy Pages config | Belong to aminx and stay there untouched; the docs URL is recorded by S3-21. The hub deployment does not depend on any of it | none |

### Assumption Ledger

No spike harness exists in this workspace (`scripts/loop/` is absent) and this task is read-only, so there
are no `S<n>` spikes. "External read" evidence is a read-only probe of a non-workspace source, named by
tool, target and date (261001); S5-18 turns the re-checkable ones into a recorded, pre-registered probe. The no-rename revision re-read A1, A22 and A28 against the current files (their `read:` citations) and added A40 and A41; no new spike was possible for the same reason.

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | The legacy `gh-pages` branch and Pages config belong to `maraxen/aminx` (which keeps its name); `maraxen.github.io/aminx/` serves a stale Sphinx build from it and the demo was never deployed | S5-33 and S5-34 would have to coordinate with a live aminx Pages deployment | VERIFIED | read: .github/workflows/pages.yml:8; read: external gh api repos/maraxen/aminx/pages, gh run list --workflow pages.yml (261001); changed: no rename, so the hub never touches this site and S5-33 has no dependency on where aminx's docs live |
| A2 | GitHub release-asset downloads are not CORS-fetchable from a Pages origin | Release assets could be the direct hosting tier and S5-06 staging is optional | VERIFIED | read: external curl -H Origin on release v0.2.0a3 asset, 302 then 206, no ACAO (261001) |
| A3 | Hugging Face `resolve/` incl. the LFS CDN hop is CORS-fetchable from the Pages origin | `hf` hosting tier is unusable; large weights need `origin` or `pages` | VERIFIED | read: external curl on maraxen/aminx proteinmpnn_v_48_020.eqx.zst: ACAO echo then `*` (261001) |
| A4 | The release v0.2.0a3 split graphs + MANIFEST.json are what a hub can consume; the monolith is not released | S5-29 could keep the monolith path | VERIFIED | read: external gh release view v0.2.0a3 (asset list); read: tools/build_site.py:67 |
| A5 | localfold's `src/index.js` is AF2 kernels only and AF3/MPNN need deep imports; no stable entry point | Library adapter could skip the upstream request | VERIFIED | read: /home/marielle/repos/localfold/src/index.js:1-50; read: /home/marielle/repos/localfold/docs/RUNNING.md:82-84 |
| A6 | localfold exposes no postMessage/URL job API, so an iframe handoff needs an upstream change | S5-23 can become an iframe embed | VERIFIED | read: external rg 'postMessage' in localfold src/ web/ (only src/runtime/yield.js:35); read: /home/marielle/repos/localfold/web/app.js:354 |
| A7 | localfold's MPNN is CPU (JS + optional wasm SIMD), not WebGPU | "Duplicate MPNN" decision framing changes (both would be WebGPU/CPU peers) | VERIFIED | read: /home/marielle/repos/localfold/src/design/mpnn/accel.js:1-10; read: external rg GPU in src/design |
| A8 | py2Dmol `plugin-system-impl` fast-forwards onto `main` | Merge item needs a rebase/merge and a re-run of bundle builds | VERIFIED | read: external git merge-base main plugin-system-impl == 67cbe73 (main tip) in /home/marielle/repos/py2Dmol-plugin-spec |
| A9 | JAX/jaxlib cannot run under Pyodide (no packaged or PyPI wasm wheel) | A Pyodide JAX executor becomes possible and ONNX export would not be required for browser | VERIFIED | read: external pyodide-lock.json from npm pyodide@314.0.7; read: external PyPI jaxlib JSON, 845 files (261001) |
| A10 | isochore's `import isochore` does not import mdtraj; only `gfe` trajectory paths do | The Pyodide executor cannot import isochore at all without S5-16 | VERIFIED | read: /home/marielle/projects/isochore/src/isochore/__init__.py:19-24; read: /home/marielle/projects/isochore/src/isochore/gfe.py:28-38 |
| A11 | xtrax's ONNX route is verified on ORT CPU only; ORT-Web execution is unverified upstream | Hub gate S5-15 would be redundant | VERIFIED | read: xtrax v0.4.0a11 src/xtrax/export/targets.py (module docstring, `onnx` bullet) |
| A12 | coi-serviceworker, scoped to `/run/`, yields `crossOriginIsolated` on a header-less Pages origin, and the assets the run pages load survive `COEP: require-corp` | Single-thread only (7 s vs 16-20 s per L128 design) or a different host for the isolated runner | UNVERIFIED | deferred: needs a deployed Pages origin; local header-less server proves the mechanism only (S5-11), real origin proven in S5-33 |
| A13 | `actions/deploy-pages` requires the **hub repo's** Pages source to be "GitHub Actions"; a new hub repo's source starts unconfigured | S5-33 drops the enable-Pages-with-Actions step | UNVERIFIED | deferred: only testable by a real deploy attempt on the live repo, which this spec-only task must not do |
| A14 | Per-origin quotas (Cache API / OPFS) accommodate ~10-250 MB of weights in current Chromium, Firefox and Safari | Cache tier falls back to `none` with a warning for large models | UNVERIFIED | deferred: browser-specific quota behaviour; S5-09 measures `navigator.storage.estimate()` in Chromium (required), S5-40 is the follow-up item that runs Firefox and WebKit |
| A15 | `meta http-equiv` CSP `connect-src` is enforceable on the hub's non-remote pages as an egress allow-list | Egress is only audited (Playwright), not enforced | UNVERIFIED | deferred: standard behaviour, but interplay with the isolating service worker is not read anywhere; S5-28 gate tests it in Chromium |
| A16 | Headless Chromium in CI has no WebGPU adapter (so WebGPU executors cannot be gated in CI) | A CI WebGPU lane exists and S5-26 gates there | UNVERIFIED | deferred: the v0.2.0a3 release notes report "no adapter on the development machine"; CI runner behaviour is not measured; WebGPU work is isolated in S5-47, which stays open until a lane has run |
| A17 | Node-only scripts under bathos need a Python wrapper to resolve their sidecar | Hub can track node scripts directly and drop the Python shim | UNVERIFIED | deferred: only the Python-wrapped form is documented; S5-59 is the one-run check (first item needing bathos), the design defaults to a stdlib-only shim |
| A18 | The S4 manifest carries the field set the catalog record needs (HUB-REQ-1..10: `driver`, `determinism`, `redistribution` enum, `artifacts[]`, `executors[]`), with the gaps G1-G4 of 4.2 | The catalog builder needs a per-manifest-version adapter | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:536; read: .praxia/docs/specs/261001_xtrax-model-contract.md:542-547; read: .praxia/docs/specs/261001_xtrax-model-contract.md:549-556; read: .praxia/docs/specs/261001_xtrax-model-contract.md:557-562; read: .praxia/docs/specs/261001_xtrax-model-contract.md:563-564; read: .praxia/docs/specs/261001_xtrax-model-contract.md:599-600; read: .praxia/docs/specs/261001_xtrax-model-contract.md:604-609; changed: round 2 maps fields explicitly (4.2) and keeps no legacy adapter (r2 R2-C1); the convergence check re-cited every line against S4 sha256 prefix 44c4d220971dc0c1 (the round-2 line numbers had drifted) and cut the gaps to G1-G4 (G5, G6 removed, see A39) |
| A19 | isochore's tools are `read_dx(path)`, `smooth_density(counts, spacing, sigma)` on a count histogram, and `validate(protein, gfe_dir, ref_pdb, out_path, ...)`; they do not chain | The 4.4 Pyodide tool entries and S5-17/S5-42 gates are wrong | VERIFIED | read: /home/marielle/projects/isochore/src/isochore/grid_io.py:80; read: /home/marielle/projects/isochore/src/isochore/kde_density.py:20-24; read: /home/marielle/projects/isochore/src/isochore/hotspots.py:132-145; read: /home/marielle/projects/isochore/src/isochore/hotspots.py:234-236 (r1 C1) |
| A20 | aminx's driver is a multi-file ES graph whose sessions can be created from bytes | Driver packaging (4.3) needs a hash-listed file set or a bundler, and CacheStore bytes cannot feed it | VERIFIED | read: browser/aminx-sampler/split_driver.mjs:14-16; read: browser/aminx-sampler/split_driver.mjs:70-78; read: browser/aminx-sampler/split_driver.mjs:126 (r1 C5) |
| A21 | aminx already records exact-token ORT-Web parity (wasm, headless Chromium, 1 and 4 threads) and warns about tie order across backends | S5-15 becomes first-time ORT-Web validation, not a hub re-check | VERIFIED | read: docs/browser_integration.md:168; read: docs/browser_integration.md:264-265; read: docs/browser_integration.md:368 (r1 C18) |
| A22 | The hub's repo slug, PyPI policy, custom domain, bathos slug and orchestrator-collision policy are not decided by S5: S3-01 records them (S3 recommends `aminx-hub`, no PyPI project, `hub.praxia.science`, `aminx-hub`) and the hub never takes the name `aminx` (S3-24 reserved-name mode) | S5-01, S5-33 and the e2e prefixes assume a slug S3-01 did not choose | VERIFIED | read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:270-286; read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:321-324; read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:500-506 (r1 C9); changed: replaces the rename-era row (S3 option 1, P1/P2, S3-22), whose items are retired |
| A23 | The existing site embeds py2Dmol via `show`, `fetch` and `setColor`, and fetches structures from files.rcsb.org | `mountViewer` is built on an API nobody uses today | VERIFIED | read: site/app.js:120; read: site/app.js:146; read: site/app.js:440; read: site/app.js:142 (r1 C13) |
| A24 | `mountViewer({structure, scores, overlays})` maps onto py2Dmol's JS embed API (scores -> colouring, overlays -> plugin primitives) | S5-12 adapter contract changes shape | UNVERIFIED | deferred: only the Python `viewer.add(plddts=...)` (`py2Dmol/viewer.py:2761`) and the three JS calls in A23 were read; S5-12's first step reads the pinned bundle's JS API and amends the contract before coding |
| A25 | The aminx ONNX split graphs are CORS-fetchable from a Pages origin on Hugging Face (A3 verified only the `.eqx.zst`) | `hf` is not a tier for ONNX; only `pages` staging (needs S5-07) works | UNVERIFIED | deferred: no ONNX file is known to exist on HF, and huggingface.co egress was denied to this task; S5-33 therefore depends on S5-07 and its gate loads a real artifact from the deployed origin |
| A26 | A service-worker-added `Content-Security-Policy: connect-src` header on a worker *script* response is enforced inside that worker (driver worker, Pyodide code sandbox) | Driver and `code.python` egress is audited only, and the S6 Q4 fallback (untrusted code runs only for in-session documents) applies | UNVERIFIED | deferred: needs a real browser; S5-43 gate is a negative test (a canary `fetch` must fail, with a control showing the canary is reachable without the header) |
| A27 | titanix has Playwright's Chromium (and possibly a WebGPU adapter) for browser lanes | Browser studies run locally only; S5-47 cannot run on titanix | UNVERIFIED | deferred: titanix browser/display/adapter setup was not probed (remote host, outside this read-only task); S5-47's first step is the probe |
| A28 | The S3/S4/S6 item ids bound in section 6 exist with the stated roles | The section 6 id map points at wrong items | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:1371-1372 (S4-04); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1389-1390 (S4-06); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1416-1417 (S4-09); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1425-1426 (S4-10); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1461-1462 (S4-14); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1506-1507 (S4-19); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1596-1597 (S4-30); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1632-1633 (S4-34); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:500 (S3-01); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:509 (S3-24); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:527 (S3-12); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:545 (S3-21); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:563 (S3-29); read: .praxia/docs/specs/261001_pipeline-editor.md:1103-1104 (S6-03); read: .praxia/docs/specs/261001_pipeline-editor.md:1193-1194 (S6-10); read: .praxia/docs/specs/261001_pipeline-editor.md:1283-1284 (S6-18); other specs are revised concurrently, so ids and lines are as of this revision (S4 sha256 prefix 44c4d220971dc0c1) (r1 C2; re-cited in the convergence check); changed: the no-rename revision cites the S3 ids that still exist (S3-01, S3-12, S3-21, S3-24, S3-29) and drops S3-02 and S3-22, which are retired |
| A29 | ORT-Web can be imported inside a module Worker on a Pages origin, so the driver host can import vendored ORT itself and hand the namespace to the driver by reference inside the worker | The driver host design (4.3) needs another ORT hand-off | VERIFIED | read: site/worker.js:11-14 (r2 R2-C5) |
| A30 | A straight bundle of aminx's split driver contains a `fetch` call (string-URL branch of `loadSession`, and of the monolith `createSampler`) | The "no fetch" scan needs no source change | VERIFIED | read: browser/aminx-sampler/split_driver.mjs:70-79; read: browser/aminx-sampler/aminx_sampler.mjs:160-165 (r2 R2-C5) |
| A31 | `OnnxBundle@1` is a manifest fragment, so the graph hashes are in the manifest's `artifacts[]` and no second document is fetched | S5-04 must fetch and validate a second document | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:677-684 (r2 R2-C1; re-cited in the convergence check) |
| A32 | S4 lacks: a weights-optional tool kind (G1), artifacts that are externally hosted or user-supplied without a pinned sha256/URL (G2), per-tool file mappings on the `pyodide` entry (G3), and `determinism` on non-onnx executors (G4). It does not lack the grid distinction (A39) or the `catalog_view` snapshot corpus (S4-09) | A subset of S5-51 is unneeded | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:538 (`kind` is `model` or `tool` and no rule on a weights-less tool appears in 4.3, `:536-566`, `:599-600`, or in 4.5); read: .praxia/docs/specs/261001_xtrax-model-contract.md:599-600 (every `hf://` URL carries a 40-hex revision and every artifact a sha256, no external or byo flag); read: .praxia/docs/specs/261001_xtrax-model-contract.md:668 (`pyodide` is `wheel`, `entry` only); read: .praxia/docs/specs/261001_xtrax-model-contract.md:665 (`determinism` is listed on the `onnx` row); read: .praxia/docs/specs/261001_xtrax-model-contract.md:668-669 (no `determinism` on the `pyodide` or `remote` rows); read: .praxia/docs/specs/261001_xtrax-model-contract.md:561 (`determinism` shown on the `onnx` example only); S4 is revised concurrently (sha256 prefix 44c4d220971dc0c1), so S5-51's first step re-reads the then-current file (r2 R2-C1); changed: the convergence check removed G5 and G6, which S4 now covers in 4.2 and S4-09 |
| A33 | IndexedDB (RunStore) behaves acceptably in the supported browsers: it works in private browsing, survives eviction pressure for small unit records, and accepts the structured-clone size of one unit's outputs | RunStore needs a memory fallback or chunked unit records | UNVERIFIED | deferred: browser-specific; S5-10 gate measures it in Chromium (required), S5-40 repeats it in Firefox and WebKit (r2 R2-C14) |
| A34 | The coi-serviceworker's script must sit at or above its scope (`<base_path>run/`), and the first visit needs one reload before `crossOriginIsolated` becomes true | Scope or first-load behaviour in 4.4 and S5-11 is wrong | UNVERIFIED | deferred: needs a real browser; S5-11's gate asserts both (first load not yet isolated, second load isolated) against a header-less server (r2 R2-C14) |
| A35 | localfold's `proteinmpnn_v_48_020.mpnn` holds the same weights as aminx's `proteinmpnn_v_48_020` up to float16 rounding | The S5-27 comparison is between different checkpoints and must be reported as descriptive only | UNVERIFIED | deferred: only the file name matches (`/home/marielle/repos/localfold/src/design/designers.js:37`); S5-27 cell 0 is the tensor-wise identity check, run before any comparison cell (r2 R2-C11) |
| A36 | aminx's split loop, with every position fixed to a given sequence, returns teacher-forced per-position log-probs usable as the S5-27 metric | The metric falls back to an aminx-side scorer for the ONNX arm or the ONNX arm is dropped | UNVERIFIED | deferred: `logProbs` output and `fixed_tokens` are read (`browser/aminx-sampler/split_loop.mjs:104,186,208-226`) but all-fixed use is not; S5-61 first step runs it on a toy input (r2 R2-C11) |
| A37 | The Pyodide runtime plus the lock packages numpy, scipy and h5py can be self-hosted on Pages within a `runtime` budget and load under COEP `require-corp` | Runtime stays on a CDN (jsdelivr, `CORP: cross-origin`), which weakens the pinned-hosts story | UNVERIFIED | deferred: package sizes were not measured; S5-62 records them and enforces the budget (r2 R2-C10) |
| A38 | A GitHub release asset can be the source for `py-wheel` staging but not a browser host (same CORS fact as A2) | A `py-wheel` tier could be a direct release URL | VERIFIED | read: external curl -H Origin on release v0.2.0a3 asset, no ACAO (261001), same fetch path for any release asset (r2 R2-C10) |
| A39 | S4 4.2 already carries the three isochore grid kinds as distinct port types (`energy-map` with alias `gfe-grid`, `count-grid`, `density-grid`, each refining `grid`), and compat rule 2 rejects a `count-grid` wired to an `energy-map`, so no `grid.meaning`-in-`compat` amendment is needed | G5 returns as an S5-51 amendment and the S5-42 manifests would declare `grid` ports with a `meaning` descriptor | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:474-477; read: .praxia/docs/specs/261001_xtrax-model-contract.md:490-496; read: .praxia/docs/specs/261001_xtrax-model-contract.md:613-629 (S4 records G5 as withdrawn); S4 sha256 prefix 44c4d220971dc0c1 (convergence check, R2-C1) |
| A40 | A GitHub Pages custom domain is a per-site setting that works with any repository slug, and a custom-domain site is served from `/` | The "repo name and domain are independent" statement and S3-29's `base_path` flip are wrong, and the repo slug would have to follow the domain | UNVERIFIED | deferred: GitHub's documentation says so (recorded as S3 A51, spike S18 read the docs), but the live behaviour is first exercised by S3-29, which needs DNS and account control; S5's design does not depend on it (`base_path` is one config line, e2e runs under `/` and two prefixes) |
| A41 | S5-52 releases through the existing `release.yml`: it fires on `release: published` and its publish job uses PyPI trusted publishing (environment `pypi`, `id-token: write`, no token), so no new distribution, token or workflow is introduced by the hub | S5-52 needs a token secret or a second workflow, which S3-12's guard would have to cover | VERIFIED | read: .github/workflows/release.yml:3-5; read: .github/workflows/release.yml:27-41 |
| A42 | aminx CI has an AST inventory test that fails on unmapped and on stale public top-level symbols in `src/aminx/{inference,sampling,scoring,model,host,tiling,ebm,potts}` and on every name in `aminx.run.__all__`; `tests/parity/browser_validation_paths.json` is its mapping, so S5-21 and S5-61 must map any public def they add there, and S2-18, S1 and S4 edit the same JSON (unordered textual merges) | S5-21 or S5-61 turns the test red, or collides with S2-18/S1-20 in the JSON | VERIFIED | read: tests/parity/test_browser_validation_inventory.py:18-29; read: tests/parity/test_browser_validation_inventory.py:101-137; read: tests/parity/browser_validation_paths.json:2-21 |

## 4. Design

### 4.1 Shape of the hub

Repo (working name `aminx-hub`; the slug is the S3-01 user decision, recorded again in `hub.toml` by S5-01; section 4.9; every URL is `base_path`-relative):

```text
aminx-hub/                       # working slug; the final name is the S3-01 decision
  hub.toml                       # config: [site] base_path, sources, hosting, budgets, cache (section 4.7)
  catalog/
    sources.toml                 # [[source]] repo, tag, manifest asset, asset_sha256 (download integrity)
    catalog.lock.json            # resolved manifest + artifact sha256, written by `catalog update`
    overrides/<model>.toml       # human copy only: title, blurb, citation, attribution text
    licenses/<model>.review.toml # sign-off record (section 4.8)
    reviews/<sha256>.toml        # driver / host-list review record (4.3, S5-50)
  schemas/                       # pinned copy of S4 JSON Schemas + hub schemas
  hub-python-allow.toml          # allow-list for the dev/shim pyproject (initially empty)
  tools/catalog/                 # build.mjs, update.mjs, review.mjs (driver + host-list diff gate)
  tools/hygiene/                 # check_stale_names.py vendored from aminx S3-24 (reserved-name mode), source commit in its header
  src/
    executors/{core,ort-web,pyodide,webgpu-localfold,remote}.mjs
    plan/                        # plan(), value refs, transfer encodings, consent hook (4.3)
    workers/                     # driver-host.mjs (model drivers), pyodide-worker.mjs
    cache/, runstore/, viewer/, config/
    shell/                       # generated-page templates, no framework
  apps/mpnn-designer/            # ported site/ (bespoke panel), built to <base_path>run/mpnn-designer/
  e2e/                           # Playwright
  notebooks/                     # Colab launcher
  vendor/vendor.lock.json        # py2Dmol bundle, ORT-Web, Pyodide, coi-serviceworker: tag + sha256
```

Build is **Node (ESM), bundler-free**: generated multi-page static HTML (`/`, `/models/<id>/`,
`/run/<id>/`, `/run/mpnn-designer/`, `/pipelines/`, `/privacy/`, `/executors/`; all relative to `base_path`), ES modules + an import map, vendored third-party
bundles by hash. One toolchain with the e2e tests; schema validation by `ajv`. The only Python in the repo
is a stdlib-only bathos shim (A17; `bth` is the installed tool, not a dependency) with a dev-only
`pyproject.toml` whose dependencies are an **allow-list** (`hub-python-allow.toml`, initially empty). The
`no-model-imports` test (S5-01) fails the build on any dependency not on the list, so it catches `torch`,
`proxide` and `aminx` (the MPNN package: the hub reads manifests and released bytes and never imports a model
package) alike, with no hardcoded deny-list to go stale. A second guard in S5-01 runs S3-24's reserved-name mode
(`check_stale_names.py --reserved aminx`): the hub's `pyproject.toml` name, any `package.json` name and any
top-level package directory may never be `aminx`, so a second importable `aminx` cannot defeat the
`find_spec("aminx")` availability guards elsewhere in the ecosystem. S6's editor arrives as a
prebuilt ES module / custom element (consumed: S6-10, section 6), so the hub needs no
bundler for it.

Why multi-page: COOP/COEP isolation has to be scoped (4.4); the service worker's scope is its directory on a
host that cannot send headers, so only `<base_path>run/` is isolated and the viewer / link-out pages are
untouched. The MPNN designer lives at `run/mpnn-designer/`, inside the isolated scope, because it runs ORT-Web.

**Base path.** A GitHub *project* Pages site is served under `/<repo>/` (aminx's own is `/aminx/`, A1, which is
never a hub prefix), the repo slug is the S3-01 decision (`aminx-hub` or a praxia.science-derived name), and a
custom domain later serves from `/` (S3-29, A40). So `hub.toml [site] base_path`
(e.g. `/aminx-hub/`, `/praxia-science/`, `/` for a custom domain) is the only place the prefix exists: one `url()`
helper emits every link, import-map entry, vendored-asset path and the service-worker scope, and no source
file contains a root-absolute path literal (a lint in S5-01's guard suite). The e2e harness serves the site
under `/`, `/aminx-hub/` and `/praxia-science/` (S5-02, S5-11, S5-32: a custom domain, the working slug and the
alternative slug), so a base-path bug appears in CI and not on the real origin.

### 4.2 Catalog: generated at build time from S4 manifests

Input: `catalog/sources.toml` lists `(repo, release tag, manifest asset name, asset_sha256)`. **Two different hashes, never
conflated (coherence round 2, CH2-05).** `asset_sha256` is the sha256 of the manifest *file bytes* as released; it is
download integrity only and is checked by `update.mjs` and `build.mjs --check`. `manifest_sha256` is S4's manifest hash,
`sha256(JCS(manifest))` over the parsed document in RFC 8785 canonical form (S4 "Canonical hash"; S4-30 owns the
Python function and the shared vector file, and "the same function hashes manifests"). `manifest_sha256` is what the
catalog record carries, what the wire protocol v1 echoes (`S5:executor-wire-v1`) and what the 4.6 unit key uses; it is
independent of whitespace and key order in the released file. The hub computes it in Node with the TypeScript RFC 8785
canonicaliser of S6-35 (tested against the S4-30 vector file), so **S5-04 depends on S6-35**; it is computed once, at
`catalog update` time, written into `catalog.lock.json`, and recomputed and compared by `--check`. The runtime server
(S5-21) computes the same value in Python with `xtrax_contract.canonical` (S4-30).
`tools/catalog/update.mjs` (run by a maintainer as `npm run catalog:update`; the lock change goes in an ordinary PR
together with its review records, S5-53; no bot workflow exists, round 2 R2-C2) fetches each manifest, validates it
against the pinned S4 schema, resolves every artifact's declared sha256/bytes, and rewrites `catalog.lock.json`.
`build.mjs --check` fails if the lock is stale or any sha256 disagrees (the `build_site.py --check`
discipline, kept). Output: `dist/catalog/catalog.json` plus `dist/catalog/models/<id>.json`, byte-deterministic
(sorted keys, no timestamps; `generated_from` carries the lock sha256 and hub commit). The file is named
`catalog.json`, the name S6 consumes (its C4); each model record has `availability[]`, one entry per declared
executor with the probe `requires` and its evidence level, so the editor can grey out unavailable executors.
`tools/catalog/update.mjs` downloads through the Node cache (4.7, S5-38).

**Mapping from the S4 manifest to the catalog record (HUB-REQ; owner S5-04).** The builder reads exactly one
document per model, the S4 `ModelManifest` at the pinned `schema_version` (A31: `OnnxBundle@1` artifacts live inside
`artifacts[]`, so there is no second fetch). Field names are S4's (A18, verified):

| Catalog record field (hub) | Derived from S4 field | Notes |
|---|---|---|
| `manifest_sha256` | computed: `sha256(JCS(manifest))` (S4-30 vector semantics; Node form from S6-35) | **not** the asset file hash; echoed by wire v1 and used in the 4.6 unit key |
| `description`, `params` (manifest params schema) | `description`, `params` | passed through verbatim; S6's palette form model reads them (S6 A51) |
| `id, kind, role, family, version, title` | same names | `kind` is `model|tool`; S4 `role` (`sampler|scorer|...`) is carried, not reinterpreted |
| `ports[]` (name, direction, type, dtype, shape, required, cardinality, `accepts_encodings`, `alphabet`) | `ports[]` | dtype/shape derived by the type table; `accepts_encodings` and `alphabet` pass through when declared (S6 A51); S5-04 reimplements `catalog_view` in JS and is tested against S4-09's `conformance/catalog_view/` snapshots (mandatory, no hub-written substitute) |
| `artifacts[]` (+ `hosting.tier`, `staged_path`) | `artifacts[]` (`id, format, role, sha256, bytes, urls, bucket, details`) | `hosting.tier` is **hub-decided** (4.7), never in the manifest |
| `availability[]` per executor | `executors[]` (`kind` -> hub executor kind per S4 4.5, `requires`, `artifacts`) | `python` entries yield no browser availability; `remote` yields `remote` only on opt-in |
| `driver {url, sha256, export}` | `executors[].driver` | staged to `dist/drivers/<sha256>/` after the review gate (S5-50) |
| `determinism {rng_scheme, cross_executor_equivalent}` | `executors[].determinism` | feeds the unit key (4.6); **missing = `undeclared`, cross-executor reuse refused** (G4) |
| `license {code, weights, redistribution, attribution}` | `license` | `redistribution` is the S4 enum `allowed|gated|forbidden` (S4 4.3, `license` object, `.praxia/docs/specs/261001_xtrax-model-contract.md:563`) |
| `evidence[]`, golden vectors | `validation.evidence`, `validation.golden_vectors` (artifact refs) | golden-vector artifacts are fetched and sha256-verified like any artifact |
| `limits`, `provenance` | same | `generated_from` carries the lock sha256 and hub commit |
| remote availability | `executors[]` entry with `protocol == "xtrax-http@1"` (runtime pointer, `determinism.rng_scheme`) | the entry is **added to aminx's source manifests by S5-21**; a manifest without it shows no remote availability |
| remote protocol | `executors[].protocol == "xtrax-http@1"` | **`xtrax-http@1` is this spec's wire protocol v1 (`S5:executor-wire-v1`, S5-19)**; the string is S4's token, the contract is S5's |

The hub never invents numeric facts; `overrides/` carries prose only. An executor badge on every model page is
derived from `validation.evidence` (`scope` keeps "ORT CPU (xtrax)" separate from "ORT-Web wasm (hub gate S5-57)",
A11), never merged.

**Gaps in S4 (four named requests G1-G4, one item: S5-51).** The hub needs these and S4 as read does not carry them (A32).
Until S5-51 lands, items that depend on a gap depend on S5-51; the rest validate against S4 unamended.

| Gap | Request to S4 | Needed by |
|---|---|---|
| G1 weights-less tool | state in the schema that a `kind: tool` manifest may have no `role: weights` artifact (isochore has none) | S5-42 |
| G2 external or user-supplied artifacts | an artifact flag `external` (`origin` or `byo`) that waives `urls`/40-hex revision (localfold bundles, AF3 params, user files) and allows `sha256` only for `byo`; the 40-hex rule stays for everything else | S5-24, S5-45 |
| G3 file mappings on tool entries | `pyodide` entry (or its ports) carries `call` plus `fs_in`/`fs_out` (path template, encoding), the shape 4.4 uses | S5-42, S5-17 |
| G4 `determinism` on every executor kind | allow `determinism` on `pyodide`, `npm`, `remote` entries (default `undeclared`) | S5-17, S5-21 |

**Not gaps (convergence check, R2-C1).** Two requests from earlier rounds are withdrawn because S4 already covers them.
*Grid meaning (old G5):* S4 4.2 defines `energy-map` (alias `gfe-grid`), `count-grid` and `density-grid` as distinct
port types, each refining `grid`, and rule 2 of `compat` rejects a `count-grid` wired to an `energy-map`
(`.praxia/docs/specs/261001_xtrax-model-contract.md:474-477`, `:495-496`; A39). The hub therefore uses those type strings (4.3, 4.4, S5-42) and files no
`meaning` amendment; S4 4.3 records the same withdrawal (`:613-629`). *`catalog_view` corpus (old G6):* S4-09 itself
commits `conformance/catalog_view/` (`:606-609`, `:1416-1422`), so it is a dependency of S5-04, not a request.
**Pin rule (agrees with S4 4.3, `:613-629`).** S5-04 deliberately builds against the S4-09 commit, which carries the
schemas and the `catalog_view` corpus it needs but not the G1-G4 amendments, so the catalog builder does not wait for
the S4-34 release; S5-64 then re-pins `schemas/` to the S4-34 release and re-runs the derivation against the released
corpus. S5-24, S5-42 and S5-45 depend on S5-64 and therefore only ever see the S4-34 release.

**No legacy adapter (round 1, C14; kept in round 2).** Nothing downstream could use an adapter before S4 exists,
and S4's manifest now carries the fields in the table above (A18), so there is none. Until aminx's native
manifest is **released** (S5-52 after S4-19), hub items test against fixture manifests that validate against S4's
committed schemas (S4-09); the first item that needs the real model depends on S5-53, which adds the aminx source
to `catalog/sources.toml` and writes the first real `catalog.lock.json`.

### 4.3 Contracts the hub owns

`Executor` (JS, JSDoc + JSON Schema, contract `S5:executor-interface`):

```js
/** @typedef {{ok:boolean, reasons:string[], details:object}} Capability */
export class Executor {
  static kind = "ort-web";                 // ort-web | pyodide | webgpu-localfold | remote
  probe(env) {}                            // -> Capability; "no adapter" is a reason code, not an exception
  privacy() {}                             // -> PrivacyDescriptor (4.5), static per executor + artifact set
  async load(entry, {cache, runStore, signal, onProgress, consent}) {}   // -> Session; bytes via CacheStore
  async *run(session, inputs, {signal, unitKey, trusted}) {}   // yields {type: "progress"|"unit"|"frame"|"done", ...}
  async cancel(session, unitKey) {}        // explicit cancel of one unit (kills its worker); `signal` is the same path
  async dispose(session) {}
}
```

- **Inputs/outputs are `PortValue`s** typed by S4 port types (consumes S4-04): typed arrays with
  dims, text for `structure`/`af3-job`, files for tool inputs (4.4), or a `ValueRef` (below). isochore needs
  three distinct grid kinds, not one, and S4 4.2 defines them as distinct port types that each refine `grid`
  (`.praxia/docs/specs/261001_xtrax-model-contract.md:474-477`): `energy-map` (alias `gfe-grid`, `meaning` gfe; what `read_dx`/`write_dx` carry), `count-grid`
  (an integer voxel-count histogram plus `spacing`; the `smooth_density` input) and `density-grid` (f32; what
  `smooth_density` returns). Compat rule 2 (`:495-496`) rejects a `count-grid` wired to an `energy-map`, so no
  `meaning`-in-`compat` request is needed (A39). **The S5-42 tool manifests declare these exact type strings,
  with `energy-map` as the one spelling (never the alias `gfe-grid`), and never a bare `grid` with a `meaning`
  field.**
- **Planning (`S5:executor-plan`, S5-37).** `plan(document, availability, prefs) -> ExecutionPlan`, a pure
  function (no I/O): `{segments: [{id, executor_kind, nodes[], requires[]}], transfers: [{edge, from, to,
  encoding}], consent_required: [segmentId]}`. A segment is a maximal run of nodes on one executor chosen by a
  deterministic preference (local executors first; **`remote` is never auto-selected**, matching S4's risk
  table, only by explicit user opt-in per segment).
- **Value refs and transfer encodings.** Large values do not cross executors by copy through the page:
  `ValueRef {id, dtype, dims, bytes, sha256, home: "worker"|"runstore"}`. Movement between executors goes
  through RunStore bytes addressed by sha256, using a named encoding from a closed registry:
  `raw-transferable` (same-origin workers), `json`, `text`, `npy-b64` (remote wire). A transfer to the remote
  executor is a `consent_required` event.
- **Consent hook.** `hooks.consent(segment) -> Promise<ConsentGrant>`; a `ConsentGrant` is an opaque object
  minted by the shell after the consent screen (4.5) and bound to `{segment, origin, input_sha256}`. Every
  executor that needs consent derives from `GrantCheckedExecutor` (S5-37), whose `load` and `run` throw without a
  grant, with a grant bound to a different origin, and with a grant whose `input_sha256` differs from the inputs
  being run; a caller cannot skip consent by omitting a boolean. S5-37 tests the base class through a stub subclass;
  the real remote executor (S5-20) re-runs the same three negative controls against itself.
- **ExecutionClient ownership (S6 Q9, corrected in round 2, R2-C7).** S6 owns the editor-facing `ExecutionClient`
  port, `ExecutionPlan`/`RunEvent` types and the `runExecutionClientContract` suite (S6-07; S6-11 publishes the suite
  with a fake client). **S5 owns the adapter**: `HubExecutionClient`, written in S5-30 over `plan()`, the executors
  and `RunStore`, and it must pass `runExecutionClientContract`. Round 1 wrongly recorded S6-11 as the adapter; S6-11
  is the demo app with a fake client (`261001_pipeline-editor.md:883-884`) and S6 Q9 itself says S5 writes the
  adapter (`:806`). S6's A31 (`:233`) is the row to correct on S6's side.
- **Driver module** (`S5:driver-module-contract`, decided in round 1, C5; reworded in round 2, R2-C5). aminx's
  source is a multi-file graph with relative imports (A20), so the model repo ships **one pre-bundled,
  self-contained ES module**, built in aminx by a dev-dependency bundler (esbuild; the hub itself stays
  bundler-free because it consumes a finished file; user decision Q10). Contract: `export async function
  createSession(ctx)` with `ctx = {ort, artifacts, manifest, env}`, where `ctx.ort` is the hub's vendored ORT-Web
  (external to the bundle) and `ctx.artifacts[id]` is a **`Uint8Array` already verified by CacheStore**, never a
  URL. For aminx, `createSession` is a thin adapter over the **bytes-only** variant of `createSplitSampler`
  (S5-14 removes the string-URL branch of `loadSession`, which contains a `fetch`, A30, from the code reachable from
  the bundle entry; URL-accepting wrappers stay in aminx's non-bundled Node/smoke entry points), mapping
  `run(inputs, opts) -> sample(structure, runspec)` and `dispose() -> release()`. The catalog stages the file into
  `dist/drivers/<sha256>/` (same origin, no CORS) and the executor loads it from there.
- **Driver trust boundary (round 1 C6, round 2 R2-C5).** A sha256 pin proves the bytes are the reviewed bytes, not
  that they are safe. Rules:
  (a) The driver runs in a dedicated module Worker (`src/workers/driver-host.mjs`). `ort` is a module namespace of
  functions and classes and **cannot cross `postMessage`**, so the **host module imports the vendored ORT itself,
  inside the worker (A29)**, and passes the namespace to `createSession` by reference. What crosses `postMessage`
  is only `{artifact bytes (transferred), inputs, run options}` in and `{events, outputs}` out. The worker holds
  **no** reference to the remote token, RunStore, CacheStore or page DOM, and receives no URL it may fetch.
  (b) The staged driver file is scanned on its **parsed AST** (S5-50, not a text grep): zero import specifiers, and
  zero references to `fetch`, `XMLHttpRequest`, dynamic `import()`, `importScripts`, `WebSocket`, `EventSource`,
  `navigator.sendBeacon`, `eval`/`Function`, or to those names as properties of `self`, `globalThis`, `window` or
  an alias. The rule list is a data file, and the gate carries a planted-violation file for each rule (negative
  control) plus a clean driver (positive control). There is no allow-list for the dead string-URL branch: S5-14
  deletes it.
  (c) The worker script is served with a generated `connect-src 'self'` header applied by the isolation service
  worker; enforcement is unverified (A26), so egress is also audited by request interception.
  (d) A driver enters `catalog.lock.json` only through the review gate (S5-50): a `catalog update` PR carries a
  unified diff of the driver against the previously reviewed sha256 and the diff of every host in
  `artifacts[].urls`/`driver.url` that feeds CSP, and CI fails if a driver sha256 or host list changed without a
  reviewer record (`catalog/reviews/<sha256>.toml`: reviewer, date, diff url).
  Residual risk stated plainly: a driver necessarily sees the user's inputs, so review, not isolation, is the
  primary control until A26 is verified.
- **Units, not whole runs.** `run` emits one `unit` event per independently reproducible piece (one design,
  one grid, one node). See 4.6.

### 4.4 Executors

**ORT-Web (`ort-web`)** - aminx ONNX and any S4 ONNX export.
- ORT-Web **1.30.0, wasm EP only**, vendored same-origin (`vendor.lock.json`), not jsdelivr: reproducible and
  safe under COEP (the jsdelivr `CORP: cross-origin` is fine, but a vendored copy removes the dependency).
  WebGPU EP is out for the same reason as in `docs/browser_integration.md` (int64 TopK in the encoder); the
  probe still reports adapter availability.
- Threads: `isolated ? min(4, cores) : 1` (carry over `site/app.js:81`). Measured upstream: COOP/COEP takes an
  L=128 design from ~16 s to under 7 s (`docs/browser_integration.md`), so isolation is worth wiring, not
  optional polish.
- Isolation: `coi-serviceworker` (MIT, 0.1.7) vendored and registered **only under `/run/`**; every
  subresource on those pages is self-hosted or `crossorigin`-attributed (py2Dmol has ACAO `*` and no CORP, so
  its `<script>` needs `crossorigin="anonymous"`). Unverified on a real Pages origin until the first deployment (A12, S5-33).
- Model path: manifest -> artifacts -> `CacheStore` (sha256 keyed) -> bytes -> driver worker (4.3) -> split graph
  E/W/D/F. The first catalog entry is aminx's P07 split export at buckets 128 and 256.
- **What is already proven, and what S5-15 adds** (A21). aminx records exact-token parity in headless
  Chromium at 1 and 4 threads and a 288-cell knob gate against ORT-Web, all on the wasm path, plus a warning
  that backends need not agree on tie order. S5-15 is therefore a **hub-integration re-check** (hub-served bytes,
  hub driver worker, hub-set threads and isolation), not first-time validation. Golden vectors are produced on
  the **ORT-Web wasm path** (the S4-18 Node-wasm/Chromium record), so S5-15 compares exact tokens on that same
  backend; no exactness is claimed across backends.

**Pyodide (`pyodide`)** - numpy/scipy/h5py tools only (A9, A10).
- Pyodide 314.x pinned, in a **module Web Worker**. **Hosting (round 2, R2-C10):** the Pyodide runtime and the lock
  packages the registered tools use (numpy, scipy, h5py) are **self-hosted** under `dist/runtime/pyodide/<version>/`,
  verified against `vendor.lock.json`, so the run page needs no third-party origin under COEP; they count against a
  separate `[budgets] runtime_mib` (proposed 250, Pages publishes at most ~1 GB), not against the 200 MiB artifact
  budget, and their measured sizes are recorded by S5-62 (A37). Tool wheels (isochore) are a `py-wheel` artifact:
  built and released by S5-54, staged by S5-62 to `dist/artifacts/<sha256>/` (a release asset is not a browser host,
  A38), installed with `micropip` from that same-origin URL (`deps=False` unless the wheel metadata is clean, S5-16).
- Inputs cross the worker boundary as transferable typed arrays / text. Every call has its own timeout and the
  worker is **killed and restarted** on timeout, so a hung call loses one unit, never the session.
- **Tool entries (the real isochore call graph, A19).** The S5-42 tool manifests declare three independent entries
  (S4 is one manifest per node type, so three manifests sharing `family = isochore`); a `pyodide` binding has `call`
  (dotted callable) plus `fs_in`/`fs_out` file mappings. **S4's `pyodide` entry is only `{wheel, entry}` (A32), so
  the mappings are request G3 (S5-51)**, and S5-42 depends on it. The worker writes
  `file` inputs into the Emscripten FS with `pyodide.FS.writeFile` under `/work/<call-id>/` (a fresh dir per
  call, removed after), passes the path, and reads declared outputs back:
  1. `isochore.grid_io.read_dx`: in `dx` (file, `.dx` bytes) -> out `grid` of port type `energy-map` (`meaning = gfe`: `data` float32 `(nx,ny,nz)`,
     `origin` float64 `(3,)`, `spacing` float).
  2. `isochore.kde_density.smooth_density`: in `counts` (port type `count-grid`, integer 3D array), `spacing_angstrom`,
     `sigma_angstrom` (params) -> out `density` (port type `density-grid`). Not fed by (1). `write_dx` takes an
     `energy-map`.
  3. `isochore.hotspots.validate`: in `protein` (string key of `KNOWN_HOTSPOTS`), `gfe` (a map of probe name ->
     `.dx` bytes, written as `/work/<call-id>/gfe/map_agfe_<probe>.dx` because the callee globs that name),
     `ref_pdb` (text, written to a file), params `threshold`, `cutoff_ang` -> out `result` (the returned dict, JSON;
     the executor passes `out_path=/work/<call-id>/out.json` and reads it back). An unknown `protein` key makes
     isochore log a warning and report raw stats only; the tool manifest restricts the `protein` port to the
     `KNOWN_HOTSPOTS` keys.
  Pipelines that combine them (e.g. `write_dx` of a smoothed grid, then `validate`) are S6 documents, not a tool.
- Capability matrix is explicit in the UI: the three entries plus `write_dx` **yes**; trajectory ingestion
  **no (mdtraj not available in Pyodide)**; any JAX package **no (A9)**. A typed
  `UnavailableInBrowser{package}` error is returned instead of an ImportError traceback.
- **`code.python` (S5-43, consumed by S6).** The Pyodide executor also runs a user-authored Python node. The
  import allow-list is derived from the pinned Pyodide lock plus hub wheels; any other import yields
  `E_CODE_IMPORT_UNAVAILABLE` before execution. `run(..., {trusted})`: a document that is not trusted does not
  execute code nodes (the executor refuses; the Trust UI is S6-17). Sandbox mode runs the node in a dedicated
  worker whose script is served with `connect-src 'none'` (A26, negative-tested), with the pinned Pyodide
  version reported in the run record.
- JupyterLite: a `/lab/` page built with the same Pyodide pin, hub wheels preinstalled, blocks from S6
  (consumes S6-18, S5-35). The notebook kernel and the executor share one Pyodide version.

**localfold (`webgpu-localfold`)** - AF2/AF3/ESMFold2/ESM-C; **third-party, unstable API (A5, A6)**.
Three integration options exist (user decision Q3); the plan layers them:
1. *v1, no coupling:* **link-out plus job-JSON handoff** (S5-23). The hub builds an AF3 job JSON (server
   dialect for proteins/ligands the server dialect can express, open dialect otherwise, per `web/job-json.js`)
   and offers a download and a link to localfold.org. localfold reads only the `model` and `graph` URL params (A6), so
   the job file cannot ride the link: the page and privacy panel tell the user to drag the downloaded file into
   localfold (S5-23 gate). Results come back as the AF3-layout zip, which the viewer can ingest. Honest about the leave-the-site step in the privacy panel.
2. *Upstream request:* ask for a single entry point (sequence/job JSON in, structure out) and a postMessage or
   module API (S5-25).
3. *After 2:* a **pinned-commit library adapter** (`localfold/src/*` deep imports are allowed by its `exports`
   map) behind the same `Executor` interface (S5-26), gated on a WebGPU lane (A16).
Requires WebGPU; `probe` reports why not (no adapter, no `shader-f16`, memory). The hub never copies
localfold's weights; the bundle names its own Hugging Face revision (README:134-135) and the hub's
`hosting.tier = "origin"` records that.

**Remote (`remote`)** - two variants sharing one wire protocol:
- **Own endpoint:** the user supplies `https://host` + token. **Colab:** a hub notebook (S5-22) starts a runtime
  implementing the protocol and prints URL + token. Colab pattern borrowed from localfold's bridge (the
  runtime serves the contract; the token starts in the URL and is removed after load) but the hub puts the
  token in the **URL fragment**, so it is never sent to a server or leaked in `Referer`.
- **Wire protocol v1** (`S5:executor-wire-v1`, S5-19): `GET /v1/capabilities` (executors, `manifest_sha256`
  per model, versions); `POST /v1/run {model_id, manifest_sha256, inputs, unit_keys?}` (`manifest_sha256` = `sha256(JCS(manifest))`, the catalog record's field, 4.2) answered as
  `text/event-stream` of the same event types as `Executor.run`; `POST /v1/cancel`. Bearer token required;
  server must answer with an **exact** `Access-Control-Allow-Origin` (never `*`) matching an allow-list it
  holds, and must echo the `manifest_sha256` it actually ran so every result records provenance. The hub
  rejects an endpoint whose echoed hash differs from the catalog's.
- Mixed content / network rules, stated plainly in the UI: `https` required except loopback; plain-http LAN
  addresses are blocked by browsers; Chrome's Private Network Access preflight may also apply to
  public-page -> local-network requests (unverified, documented as "if it fails, use a tunnel").
- Reference runtime: the hub ships a **mock** server for the conformance suite; the real runtime
  (`aminx serve`, S5-21) lives in the model repo because it imports the Python model.

**Viewer (`py2Dmol`, S5-12, S5-36).** `mountViewer(el, {structure, scores, overlays})` wraps the embed API the
existing site already uses (`show`, `fetch`, `setColor`, A23); how `scores` and `overlays` map is unread and is
the first step of S5-12 (A24). The bundle is self-hosted, hash-pinned, loaded with `crossorigin="anonymous"` and
must render **on the isolated `run/` route**, because results are shown on that page: S5-12's gate is on that
route with zero COEP-blocked requests. Structure fetches by id (`files.rcsb.org`, AFDB) are `connect-src` hosts
contributed by the viewer (4.5) and only enabled when the user types an id; if a CORS or COEP failure occurs the
viewer shows "paste or upload a file" rather than failing silently. The `volume` plugin is **not in any bundle**
(section 3), so S5-13's release provides it as a separate file, listed in `vendor/vendor.lock.json` with its
sha256 and loaded by `<script crossorigin>` before mount; it registers itself on `window.py2dmolPlugins`
(registry created if absent, load order free, `PLUGIN_API_VERSION = 1`).

### 4.5 Auth and privacy model (honest)

Principles: the hub is static, so it holds no secrets, no accounts, and sees no user data. Everything below
is what a user could be exposed to, listed per executor and rendered from a machine-readable
`PrivacyDescriptor {executor, sends:[{what, to, when, optional}], stores:[{what, where, ttl}], thirdParties[]}`.

| Executor | What leaves the machine | Who can see it | Persistence |
|---|---|---|---|
| ort-web (aminx) | Weight/driver download (host list from the catalog); optional structure fetch from RCSB/AFDB by the id the user types | GitHub Pages / HF / RCSB (request metadata only) | Weights in the browser cache (4.7); designs in the run store until cleared |
| pyodide | Pyodide runtime + wheels from the pinned host; nothing about the data | CDN request metadata | Same |
| localfold link-out | Whatever the user then enters on localfold.org (MSA search sends the sequence to ColabFold's server unless single-sequence; templates fetch by id) - localfold's own statement | localfold.org, ColabFold, RCSB/AFDB | localfold's |
| localfold library (later) | Weights from its Hugging Face revision; optional MSA/template fetches as above | HF, ColabFold, RCSB | Browser cache |
| remote | **The full input** (structure text, sequence, RunSpec) to the user's endpoint; with a tunnel (typical for Colab) the tunnel provider terminates TLS and **can read it** | The endpoint's operator (the user, or whoever hosts it) plus the tunnel provider | Whatever the runtime keeps; the hub keeps nothing |

Rules the build and tests enforce:
- **Consent before first byte:** a remote run shows exactly what will be sent and to which origin; no `POST`
  precedes it (Playwright request-interception gate, S5-20).
- **Token handling:** fragment-delivered, held in memory, optional `sessionStorage` on explicit opt-in, never
  `localStorage`, never in URLs the page requests, sent only as `Authorization` to the endpoint origin it was
  issued for.
- **Egress allow-list:** each non-remote page gets a generated `<meta http-equiv="Content-Security-Policy">`
  whose `connect-src` is the union of the hosts its executors declare, the hosts of the artifacts in the models it
  lists, and the viewer's structure-fetch hosts (`files.rcsb.org`, AFDB) when id fetch is enabled (enforcement
  unverified, A15, so it is *also* audited: the Playwright suite fails on any request outside the declared set). The remote page is the
  one place `connect-src` must open to a user-entered origin; it carries the consent screen instead.
- **Integrity:** every artifact is verified against the manifest sha256 before use (build time for staged
  files, load time for `hf`/`origin` fetches); a mismatch is a typed error, never a silent fallback.
- **Gated weights:** for `redistribution: gated|forbidden` artifacts the hub offers *bring your own weights*
  (file picker, sha256 verified against the manifest, stored locally, never uploaded; S5-06/S5-09 hooks).
- No analytics, no third-party scripts beyond those in `vendor.lock.json` **and the reviewed model drivers**
  (4.3 trust boundary: drivers are third-party code and are held to the review gate, not exempted from this rule).

### 4.6 Long jobs are resumable and chunked (from the first write)

- A **run** is a list of **units** (a design, a grid, a graph node). Each unit has a key
  `sha256(manifest_sha256 (JCS form, 4.2) || executor kind || driver sha256 || rng_scheme || canonical inputs || unit params)`
  (round 1, C7): the ORT-Web sampler draws Gumbel noise in JS while the remote runtime uses a JAX RNG, so the
  same seed gives different tokens. **Cross-executor reuse is refused** unless the manifest attests
  `cross_executor_equivalent` for that pair; a unit stored by executor A is recomputed when the run resumes on
  executor B and is listed as "not reused: executor changed".
- `RunStore` (S5-10): after each unit completes, persist `{unit_key, outputs, input_hash, artifact_sha256s,
  completed_at}` to IndexedDB **before** emitting `unit`. On reopen, "resume" reuses a stored unit only when its
  input hash and artifact hashes equal the current run's; anything else is recomputed. The result lists
  reused units (source run id, hashes) separately from computed ones.
- **Blast radius:** a per-unit timeout and a separate worker per executor session; a timeout kills one worker
  and loses at most the unit in flight (ORT-Web L=128 is ~7-20 s a design, so a 50-design job is minutes and
  must not be a single call).
- Node/CI studies (S5-57, S5-58, S5-27, S5-59) write one JSON per cell with a completion stamp (inputs sha256,
  artifact sha256s, cell result) under the **study output root**, which resolves through the 4.7 path resolver
  (explicit `--study-dir` > `AMINX_HUB_STUDY_DIR` > `hub.toml [paths] study_dir` > user config > **fail loudly when
  unresolved**; no default path), and resume the same way: rerunning skips cells whose stamp matches and recomputes
  any whose inputs or artifact hashes differ, recording which cells were reused and from where. Each cell runs in
  its own process with its own timeout; a timeout loses one cell. Cell counts and timeouts are fixed in the
  pre-registered sidecar (S5-57: 8 designs at L=128, 120 s per cell; S5-58: 3 parity cells, 60 s each; S5-27: cell 0
  plus 12 comparison cells, 300 s each) and every resume path has its own gate.
- **Implementation and study are separate items (round 2, R2-C9).** The executor PR (S5-15, S5-17) closes on
  CI/Playwright assertions against committed expected files. The pre-registered study (S5-57, S5-58) is a second
  item: sidecar PR first, then the run, then the cited record. Items that need only a working executor (S5-29, S5-33)
  depend on the implementation item alone.

### 4.7 Weights hosting, caching, size budgets, licenses

Hosting tiers, recorded per artifact in the catalog (`hosting.tier`):

| Tier | Meaning | Allowed when |
|---|---|---|
| `pages` | CI downloads the released asset, verifies sha256, stages `dist/artifacts/<sha256>/<file>` | `redistribution: allowed`, a sign-off record exists (4.8), under the per-file and total budget |
| `hf` | Fetched from a pinned Hugging Face revision at run time | CORS verified (A3); sha256 verified on load |
| `origin` | Hub never copies; the third party's own source (localfold bundles, AF3) | Always; record only |
| `byo` | User supplies the file; hash-checked | `gated`/`forbidden` |
| `dev` | **Never deployable.** `build.mjs --mode dev` downloads the lock's release bytes in Node, verifies sha256, and stages them into a local build dir for the local file server and Playwright; it writes `DEV-ONLY.marker` and `deployable: false` into `hub.config.json`, and the deploy workflow fails if either is present (S5-06, S5-32) | Any artifact, no review record needed, because the bytes never leave the machine or CI job. Used by S5-15, S5-29 and the studies S5-57, S5-27 |

`py-wheel` artifacts (isochore) are not a tier of their own: they use `pages` (A38) under the same sign-off rule, with
the wheel's licence reviewed like a model's (S5-62). A fourth host kind is added only if a CORS probe (S5-18) shows
a release asset gaining `Access-Control-Allow-Origin`.

GitHub release assets are **not** a browser hosting tier (A2); they are the *source* for `pages` staging.
For aminx's ONNX graphs no browser-fetchable host exists until either S5-07 (sign-off, then `pages`) or an
HF ONNX upload with a CORS probe (A25, unverified): the first public deployment (S5-33) therefore depends on
S5-07 (the decision, approved or declined) and on S5-56 (the HF-upload fallback, a conditional item that runs only
when S5-07 records a decline and is closed as not applicable otherwise), so the DAG is complete under both outcomes.
Pre-sign-off development and CI use the `dev` tier, never `pages`.
Proposed budgets (config in `hub.toml`, user-adjustable): staged artifacts total <= 200 MiB, each <= 50 MiB;
show size before first download, confirm above 100 MiB; the build fails over budget. For scale: aminx
split bucket pair ~13.4 MB; localfold MPNN families 3.3-5.3 MB; AF2 227 MB and AF3 150 MB stay `origin`.

Cache (derived-data rule): the browser cache is *derived data*, so
- `CacheStore` interface with backends `none | cache-api | opfs`; library/executor code has **no default**:
  the caller injects a store.
- The shell resolves the backend and namespace by: explicit argument > URL param `?cache=` / user setting in
  `localStorage` > the generated `hub.config.json` (from `hub.toml [cache]`) > off. `cacheRootSource()`
  reports which layer decided; a malformed config fails loudly.
- Node/CI caches (downloaded manifests and artifacts during `catalog update` and e2e): `--cache-dir` >
  `AMINX_HUB_CACHE` (`none`/empty disables) > `hub.toml [cache]` > `${XDG_CONFIG_HOME:-~/.config}/aminx-hub/config.toml`
  > off. (A JS repo has no `pyproject.toml`, so `hub.toml` is the "project config table" layer; implemented and
  gated by S5-38.) The browser chain (URL param > localStorage > `hub.config.json` > off) is a separate resolver;
  S5-03 gates both. Never written
  into another project's directories; keyed by **sha256 of the bytes**, never by URL or temp name; validated on
  read, stale entries rejected; written to a per-writer temp then renamed.
- **Other derived-data directories (round 2, R2-C13)** resolve through the same chain
  (explicit arg > env var > `hub.toml [paths]` > `${XDG_CONFIG_HOME:-~/.config}/aminx-hub/config.toml` > unresolved,
  which fails loudly where the directory is needed; no default path in code, a malformed file raises, the deciding
  layer is reported):

  | Directory | Argument | Env var | `hub.toml [paths]` key |
  |---|---|---|---|
  | study outputs (per-cell JSON + completion stamps, 4.6) | `--study-dir` | `AMINX_HUB_STUDY_DIR` | `study_dir` |
  | build output and staging scratch (`dist/`, `dist-dev/`) | `--out-dir` | `AMINX_HUB_BUILD_DIR` | `build_dir` |
  | Playwright browser download path (passed to Playwright as `PLAYWRIGHT_BROWSERS_PATH`) | `--browsers-dir` | `AMINX_HUB_BROWSERS_DIR` | `browsers_dir` |
  | e2e traces, screenshots, videos | `--e2e-artifacts-dir` | `AMINX_HUB_E2E_DIR` | `e2e_dir` |

  The Node cache (`AMINX_HUB_CACHE`) is one more row of the same family. S5-03 gates the whole table; S5-57, S5-58,
  S5-27 and the e2e harness (S5-02) cite it and write nowhere else. Never inside another project's directories;
  check quota and inode limits of the chosen tier (Playwright browsers and `node_modules` are inode-heavy).
- Browser entries are content-addressed by sha256 (the same bytes from two URLs share one entry), verified on
  read, with an evict-and-clear UI. Storage limits are checked with `navigator.storage.estimate()` (A14).

### 4.8 Licenses and sign-off

`catalog/licenses/<model>.review.toml` records reviewer, date, source URL and the decision for each
redistributed model. The build refuses `hosting.tier = "pages"` without one (S5-06) and never copies
`forbidden`. ProteinMPNN/LigandMPNN code is MIT (verified) but the weights question is the user's sign-off
(`browser_integration.md:403`; Q8). Attribution text from the manifest is rendered on every page that
runs the model (ProteinMPNN: Dauparas et al., Science 2022; py2Dmol: Ovchinnikov; ORT-Web; localfold).
AF3: listed with link-out and `byo` only (Q4).

### 4.9 Repo strategy and first deployment (relationship to S3)

The hub is a **separate project** from the aminx package (revised D2). aminx keeps `maraxen/aminx`, the PyPI
project `aminx`, the import `aminx`, the HF weights `maraxen/aminx`, the `aminx` CLI and its docs; nothing there
moves and no consumer (mpnn_ext, asr, hautespout, tev_design) is edited (S3 4.1). The hub is a new repository whose
Pages source starts unconfigured (A13). Its identity is the user decision S3-01 records: repo slug (recommended
`maraxen/aminx-hub`; the alternative is a name derived from the domain praxia.science), PyPI policy (recommended:
no project; never `aminx`), custom domain (recommended: the subdomain `hub.praxia.science`, attached after the first
deployment by S3-29), bathos slug (recommended `aminx-hub`, independent of the repo slug) and the policy for the
collision with the user's praxia agent orchestrator (a hub repo, PyPI, npm, CLI or bathos name containing `praxia`
would share a namespace with the orchestrator's CLI, repo, crate family, bathos slug and registered PyPI project; a
DNS name under praxia.science would not). A Pages custom domain works with any repo slug (A40), so the repo name and
the domain are independent choices. S5 builds to any answer: every URL is `base_path`-relative and nothing but
`hub.toml` and the repo settings knows the slug.

Sequence. S5-33 deploys the hub on its project Pages URL (`base_path = /<slug>/`) after: the deploy workflow
(S5-32), the designer port (S5-29), the sign-off or fallback (S5-07, S5-56), the real release in the lock (S5-53)
and the identity ADR (S3-01). Steps: (1) the user creates the hub repo with the slug S3-01 recorded (S5-01); (2)
enable Pages with source **Actions** on it (A13); (3) deploy via S5-32; (4) smoke and isolation probe on the real
origin (A12) *including a real model artifact load*. Attaching a custom domain is **S3-29** (depends on S3-01 and
S5-33): domain verification, the Pages custom-domain setting, a DNS-only CNAME and one `hub.toml` line
(`base_path = "/"`). It replaces the earlier P2 item (the hub taking the name `aminx`), which revised D2 voids:
S5-55 is retired. **No S5 item publishes to PyPI or npm**: the hub's only Python is the stdlib bathos shim and the
dev-only `pyproject.toml`, and any later hub publication is a decision S3-01's re-open table regenerates.
**The hub reserves no `aminx.*` Python namespace and never uses the name `aminx` for anything it publishes**,
enforced in hub CI by S3-24's reserved-name mode (S5-01). Because everything is `base_path`-relative and e2e-tested
under three prefixes (4.1), choosing the slug or attaching the domain changes one `hub.toml` line.

Hazards. aminx's release URLs (`github.com/maraxen/aminx/releases/...`) and HF weights (`HF_REPO_ID =
"maraxen/aminx"`, `src/aminx/io/weights.py:23`; `HF_REVISION` at :34) are stable, so the catalog pins them as they
are and nothing is re-pointed. The hazards are the hub's own: a slug or domain that collides with the
orchestrator's namespace (S3-01 policy), a dangling CNAME if the Pages site is later unpublished (S3-29 deletes the
record with the site), and the hub taking the name `aminx` (S3-24). Repo creation and Pages enablement are user
actions (S5-01 and S5-33 are decision items).

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Release assets are not CORS-fetchable (A2); copying `pages.yml`'s remote-base branch fails silently in browsers | Tier model 4.7; `pages` staging from verified release bytes; S5-18 re-checks A2/A3 on a schedule |
| ONNX graphs may have no CORS-capable host (A25): a deployed hub whose only model cannot load | S5-33 depends on the sign-off decision S5-07 and the conditional HF fallback S5-56 (CORS probe gate); its gate loads and verifies a real artifact from the deployed origin; development uses the never-deployable `dev` tier |
| The hub is created before its identity is decided, or takes the name `aminx` (which would defeat every `find_spec("aminx")` guard) | S5-01 depends on S3-01 (slug, PyPI policy, domain) and on S3-24; S3-24's reserved-name mode runs in hub CI (S5-01); everything is `base_path`-relative and e2e-tested under three prefixes, so the slug is one config line |
| The hub name collides with the user's praxia agent orchestrator (CLI, repo, crates, bathos slug, registered PyPI project) | S3-01 records the collision policy (recommended: no `praxia` in any non-DNS name; the domain is DNS only); the slug is independent of the domain (A40) |
| Two aminx releases race for a PyPI version (S5-52, S1-25, any S2 release) | S3-12's build-job guard (tag equals built version, PyPI JSON must be 404, fail closed) and next-free-alpha rule; S5-52 reads the version at cut time and S1-25 is ordered behind it (A41) |
| `maraxen.github.io/aminx/` serves stale Sphinx docs | That site belongs to aminx and is untouched; the hub is another repo, never uses `/aminx/` and enables Pages with Actions itself (A1, A13); docs URLs are recorded by S3-21 |
| Project Pages sub-path breaks root-absolute links, service-worker scope or the import map (C8) | `base_path` in `hub.toml`, one `url()` helper, lint for root-absolute literals, e2e under `/`, `/aminx-hub/`, `/praxia-science/` |
| Model drivers are third-party JS inside the hub origin (C6) | Driver worker with no token/store handles, `connect-src 'self'`, review gate with driver and host-list diff (S5-50); residual risk stated in 4.3; A26 unverified |
| Driver cannot be loaded as one hashed file, or contains a fetch (multi-file source, C5; A30) | Pre-bundled single module built in aminx from a bytes-only entry, parsed-AST scan with planted negatives (S5-50), loads from `dist/drivers/<sha256>/` (S5-14, S5-15) |
| S4 lacks fields the hub needs (A32, gaps G1-G4) | One named request item, S5-51; items that need a gap depend on it; the rest validate against S4 as it stands; missing `determinism` is treated as `undeclared` (reuse refused) |
| No published aminx release carries the native manifest and driver (v0.2.0a3 has only the old MANIFEST.json) | S5-52 cuts the browser-assets release; S5-53 pins it in `sources.toml` and writes the first real lock; real-model gates (S5-15, S5-33) depend on S5-53 |
| isochore wheel has no host or CORS-capable URL (A38) | S5-54 builds and releases the wheel; S5-62 stages it to the `pages` tier with sha256 and hosts the Pyodide runtime under a `runtime` budget (A37) |
| Hub-side `HubExecutionClient` has no owner (S5/S6 text disagreed) | S5-30 owns it and must pass S6's `runExecutionClientContract`; S6's A31 is the row to correct |
| Vendored third-party licences (Pyodide MPL-2.0, ORT-Web, coi-serviceworker MIT, py2Dmol Beerware) lack attribution | `vendor.lock.json` entries require `license` and `notice`; NOTICE generated and checked (S5-63); the hub's own licence is Q11 |
| localfold is third-party, unstable (0.0.x), mirrors an external repo | v1 link-out only; upstream request; library adapter pinned to a commit with contract tests; never in the critical path of other items |
| AF3 weights terms (Prohibited Use Policy) | `origin`/`byo` only, no hosting; the AF3 entry is its own user-decision item (S5-45) |
| Weights redistribution rights unconfirmed | No `pages` tier without a sign-off record; the `dev` tier cannot be deployed (marker checked by S5-32); S5-33 depends on S5-07 and S5-56 |
| Isolation (COOP/COEP) breaks cross-origin assets or iframes | Scope to `run/`; self-host/vendor; `crossorigin` attrs; e2e asserts no COEP-blocked requests including the viewer on the isolated route; fallback single thread (A12) |
| Pyodide cannot run JAX or mdtraj | Executor matrix states it; typed unavailable errors; JAX models use ORT-Web or remote (A9, A10) |
| isochore tools do not compose as the first draft assumed (C1) | Tool entries follow the real signatures with FS mappings (4.4, A19); parity gates are three independent cells |
| Dependency drift: hub imports a Python package "just for the schema" | Allow-list guard (`hub-python-allow.toml`); hub reads JSON only; schemas vendored by hash |
| ORT-Web results differ from reference (RNG is host-side) | Golden vectors from the wasm path; S5-15 compares exactly on that backend; ties across backends not claimed (A21) |
| A resumed run mixes outputs of executors with different RNG (C7) | Unit key includes executor kind, driver sha256 and rng_scheme; cross-executor reuse refused; negative-control gate in S5-10 |
| Remote endpoint exfiltration / token leakage | Consent grant object, exact-origin CORS, fragment token, no persistence, egress audit, provenance echo |
| Two browser MPNNs diverge silently, or are different checkpoints (A35) | S5-27 pre-registered, cell 0 weight identity first, frozen reference from aminx (S5-61), localfold node harness (S5-60); decision item S5-46 after the numbers |
| Node scripts do not resolve bathos sidecars (A17); CI cannot write a bathos catalog | Stdlib-only Python shim spawns the node/Playwright cell; bathos studies run locally or on titanix; CI runs plain assertions only (section 7) |
| A gate that accepts NOT RUN closes an item with nothing verified (C15) | Required (CI) part and optional lane are separate items (S5-40, S5-47) that stay open until the lane has run |
| Old JS tests never ran in CI; they may already be red | S5-01 CI runs them first; S5-29 ports only passing suites and records any that fail |
| S5-21 or S5-61 adds a public def in the aminx root packages and turns the inventory test red, or its JSON edit merges badly with S2-18/S1-20 (A42) | Both gates run `tests/parity/test_browser_validation_inventory.py` and map new symbols in `browser_validation_paths.json` in the same PR; keep `aminx serve` outside the scanned packages where possible; rebase on the JSON before merge |
| Browser quota/eviction deletes cached weights mid-use | Content-addressed re-fetch; `RunStore` unit outputs are independent of weights; quota check before download |
| S4/S6 contract names differ from those used here | `depends_on` holds item ids only; section 6 has the token-to-id map (A28) and the S5 id map that S6 rebinds to |

Rollback: every item is additive in `aminx-hub` or a copy-then-delete in aminx; S5-34 (deleting the old
site from aminx) is the only deletion and is gated on a deployed, smoke-tested hub; revert that PR to
restore it.

## 6. Interfaces

### Provides

| Contract | What | Consumers |
|---|---|---|
| `S5:catalog-index` | Schema + generated `dist/catalog/catalog.json` (name S6 consumes): models, tools, ports, executors with per-executor availability, hosting, license, evidence | S6 (palette), hub shell |
| `S5:executor-interface` | The `Executor`/`Session`/event contract incl. `cancel` (4.3) | S6, model driver modules |
| `S5:executor-plan` | `plan()`, `ValueRef`, transfer-encoding registry, `ConsentGrant` hook, and the `HubExecutionClient` adapter that passes S6's contract suite (4.3; S5-37, S5-30) | S6 (S6-15, S6-19 evidence) |
| `S5:driver-module-contract` | One pre-bundled ES module, `createSession(ctx)` over verified bytes, driver worker, review gate | S4 (manifest `executors[].driver`), aminx |
| `S5:code-python-capability` | Pyodide `code.python` with import allow-list from the lock, `trusted` flag, sandbox worker | S6 (S6-16, S6-17) |
| `S5:executor-wire-v1` (= the manifest token `xtrax-http@1`, S4 4.5) | Remote protocol, schemas, conformance suite, mock server | aminx runtime, any future runtime (S2's EBM project), S4 `remote` entries |
| `S5:privacy-descriptor` | Per-executor "what leaves your machine" record | S6 (show on nodes), hub pages |
| `S5:cache-store` | Content-addressed browser cache interface + config resolution (plus the Node cache, S5-38) | S6, praxis browser mode if it wants it |
| `S5:run-store` | Unit persistence/resume contract with executor-aware keys | S6 pipeline runner |
| `S5:viewer-adapter` | `mountViewer(el, {structure, scores, overlays})` over py2Dmol with graceful fallback (A24) | S6 preview nodes |
| `S5:hub-config` | `hub.toml` schema (incl. `base_path`) and resolvers | internal; S3-29 flips `base_path` to `/` when the custom domain is attached |

### Consumes

`depends_on` carries item ids only. The right-hand column is the binding this revision uses (A28); the middle
column is the contract name this spec used in round 0.

| Contract | Provider | Bound to item(s) | Use |
|---|---|---|---|
| `S4:model-manifest` | S4 | S4-06 (dataclasses, `catalog_view`), S4-09 (committed JSON Schemas, validators); S4-19 (the MPNN manifest); **S5-51 (hub-requested amendments G1-G4, an S5 item in the xtrax repo)** | Catalog input; schema version pinned; mapping table in 4.2 (A18, A31, A32) |
| `S4:port-types` | S4 | S4-04 | `PortValue` typing; isochore ports use S4's `energy-map`, `count-grid`, `density-grid` types (A39), no amendment |
| `S4:onnx-route` | S4 | S4-14 | The shared export that produces aminx's ONNX |
| `S4:graph-ir` | S4 | S4-10 | Pipeline runner (S5-30), planner (S5-37); `callable_ref` names a Python callable, so the runner binds by manifest/executor |
| `S3:aminx-identity` | S3 | S3-21 (README hub pointer; the ADR record of aminx's docs URLs, whose stale `maraxen.github.io/aminx/` Sphinx build stays where it is) | aminx keeps `maraxen/aminx`, HF `maraxen/aminx` and `browser/aminx-sampler`; S5-14, S5-21, S5-52 consume the names and release URLs, S5-34 consumes the docs URL record |
| `S3:release-guard` | S3 | S3-12 (build-job guard, dry run that never publishes, next-free-alpha train rule) | S5-14, S5-52 release through it (A41) |
| `S3:stale-name-guard` | S3 | S3-24 (`check_stale_names.py`, reserved-name mode) | S5-01 vendors it into `tools/hygiene/` and runs `--reserved aminx` |
| `S3:hub-identity` | S3 | S3-01 (hub identity ADR: repo slug, PyPI policy, custom domain, bathos slug, orchestrator-collision policy, the rule that the hub never takes `aminx`); S3-29 attaches the custom domain after S5-33 and nothing in S5 depends on it | Precondition of S5-01 (the slug is known before the repo is created) and of S5-33; S5 sidecars read the bathos slug from it |
| `S6:graph-document` | S6 | S6-03 | Runner input (S5-30) |
| `S6:execution-client-port` | S6 | S6-07 (port, `runExecutionClientContract`), S6-11 (suite published with a fake client) | `HubExecutionClient` adapter (S5-30) must pass it |
| `S6:pipeline-editor-component` | S6 | S6-10 (pipe-list element) | `/pipelines/` page (S5-31) |
| `S6:jupyterlite-blocks` | S6 | S6-18 (ipynb export for JupyterLite) | `/lab/` page (S5-35) |
| `S1:runspec-schema` (optional) | S1 | none (no item depends on it) | Generic form rendering later; `runspec_core.mjs` stays aminx-owned and ships in the driver |
| py2Dmol tag with plugin API | user-owned external | S5-13 | Viewer (core works with `main`; density overlay needs the plugin merge) |
| localfold pinned commit | third-party external | S5-23 | Link-out v1; library adapter later |

### Ordering records (restated for the no-rename world)

The earlier revision argued several orderings from the rename (the hub against the rename PR and the shim-removal
item, "rename first vs EBM first", S2 running on a renamed `ebm` tree). With no rename, no shim and no deprecation
window those arguments are void; this is what holds now.

- **No S5 item waits for a rename.** The S3 edges that remain are S3-01 (S5-01, S5-33), S3-24 (S5-01), S3-12 (S5-14,
  S5-52) and S3-21 (S5-34). The first public deployment (S5-33) waits for S3-01 only among S3 items (S3-12 arrives
  transitively through S5-53 and S5-52, S3-24 through S5-01), and does not wait for EBM close-out or the S1 flip chain.
- **EBM first vs hub first.** The hub imports no Python model package, no EBM manifest exists in v1 (non-goal) and
  nothing outside `src/aminx/ebm/` imports it, so the hub's first deployment is independent of S2 in either order.
  S5-14 and S5-52 build and release aminx at whatever state main is in; an S2 release that carries the EBM removal is
  ordered against S5-52 and S1-25 by S3-28's release-order check.
- **Release order (CH2-01, kept).** S1-25 depends_on S5-52 (edge owned by S1-25); both depend on S3-12, which owns the
  guard and the next-free-alpha rule. S5-52 takes the next unused alpha read from PyPI at cut time (0.2.0a4 on 261001),
  does not contain S1-25, and S1-25 then takes the following alpha (0.2.0a5). The first hub deployment (S5-33 -> S5-53
  -> S5-52) never waits for the S1 flip chain, and S3-28 sees S1-25 and S5-52 ordered. S5-52 need not contain the
  S4-20 generated knob schema.
- **Manifest finalisation order:** S4-19 (source manifest, placeholders) -> S5-14 (bundle, sha256) -> S5-52 (rewritten,
  validated, finalised manifest).
- **Weights redistribution:** S4-37 decides the manifest value; S5-07 records the hub's sign-off and must agree.
- **localfold:** S5-25 is the only upstream request; S4-27's request half is redundant and S4's iframe interim (Q6) is
  impossible without it (A6).

### Requests from S6 (coherence round 2, CH2-06)

S6 files these against S5 items; S5 answers here so none is silent.

| S6 item or ask | S5 answer |
|---|---|
| S6-33 (catalog record carries params, description, port `accepts_encodings`/`alphabet`, `manifest_sha256`) | **Accepted into S5-04.** The 4.2 mapping table lists all four and S5-04's gate asserts them, so S6-33 is a verification no-op against the S5-04 builder. S5-04 depends on S6-35 for the Node canonicaliser |
| S6-34, S6-27, S6-36 | No S5 contract change; they bind only through `S5:catalog-index` and the S5-31 page. S5 names no further edge |
| Q17(a): does S5-01's dependency guard scan `package.json` | **Open, S5 recommendation: no.** The guard stays scoped to the dev-only Python `pyproject.toml`; any npm allow-list is a committed file S6-02 adds (S5-01 gate unchanged apart from "passes with the package present") |
| Q17(b): a workspace-sourced `vendor.lock.json` entry before extraction | **Open, S5 recommendation: yes.** A hash-pinned entry whose `source` is the workspace build with the sha256 from `checksums.txt`; S5-31 is the first consumer |

### S5 id map for S6 and other specs to bind to

S6 section 6 expects S5 role-bound placeholders (S5-01 scaffold, S5-02 catalog, S5-03 executor API, S5-04 Pyodide,
S5-05 viewer) and also binds tokens in its `depends_on` (`S5:catalog-index`, `S5:executor-interface`,
`S5:privacy-descriptor`, `S5:pyodide-code-exec`). S5 keeps its own ids, so S6 rebinds as below. Recorded outcome of S6
Q9 (round 2): S6 owns the `ExecutionClient` port and contract suite (S6-07, S6-11); **S5 owns `HubExecutionClient`
(S5-30)**.

| S6 expectation or token | Real S5 item(s) |
|---|---|
| C8 hub repo scaffold (S6 "S5-01") | S5-01 (same id) |
| C4 `catalog.json` (S6 "S5-02"); token `S5:catalog-index` | S5-04 builder; pages in S5-28. S6 `:430,751` still says `dist/catalog/index.json`: **S5 emits `catalog.json` only, no alias**; S6 rebinds the file name |
| C5 executor API: plan/run/cancel/resume, capabilities, value refs, transfer encodings, consent hook (S6 "S5-03"); token `S5:executor-interface` | S5-08 (contract, cancel, capabilities), S5-37 (plan, value refs, encodings, consent), S5-10 (resume), S5-30 (`HubExecutionClient`) |
| Token `S5:privacy-descriptor` | S5-08 (schema); per-executor descriptors in S5-15, S5-17, S5-20, S5-23 |
| C6 Pyodide executor with `code.python` and sandbox/untrusted mode (S6 "S5-04"); token `S5:pyodide-code-exec` (not an S5 name) = `S5:code-python-capability` | S5-17 (executor), S5-43 (`code.python`, sandbox, `trusted`) |
| C7 viewer contract (S6 "S5-05"); token `S5:viewer-adapter` | S5-12 |
| Token `S5:run-store` | S5-10 |
| `/pipelines/` page, runner | S5-31, S5-30 |
| `/lab/` page | S5-35 |

## 7. Verification gates

Cross-cutting rules:
- **Pre-registration and where `bth` runs.** Anything that yields a finding (parity numbers, tolerances, the MPNN
  comparison, the executor-matrix probe) gets a tracked script plus a `.bth.toml` sidecar committed *before* the
  run, with `[outcomes]` criteria, verified by the record (`bth sql ...`, after `bth compact`), never by exit
  code. Items: S5-57, S5-58, S5-27, S5-59, S5-61 and the manual probe of S5-18. Executor implementation items (S5-15, S5-17) are *not* in this list: they close on CI assertions and a separate study item follows (R2-C9). **Venue: local or titanix, never CI** (a CI job
  cannot write the user's `~/.bth` catalog and bathos is not on PyPI), under the bathos slug S3-01 records (recommended `aminx-hub`; `aminx` is the package's own slug, 145 runs), with the
  installed `bth` (not `uv run bth`): `bth run --project-slug aminx-hub -- uv run --no-sync python3
  scripts/studies/<study>.py` where the stdlib-only shim spawns each node/Playwright cell, so `script_path`
  resolves to a `.py` and the sidecar `<stem>.bth.toml` resolves (never a `bash -c` or `npx` wrapper). The
  record lives in that machine's bathos catalog; the spec's gate text cites the run id. CI runs plain assertion
  tests only.
- **Positive and negative controls.** Every comparison gate pairs a case that must pass with one that must
  fail (planted corrupt artifact, a mock without an Origin check, a tampered unit hash, a perturbed weight
  tensor, a resume on a different executor). A gate without the negative is not accepted.
- **Name what did not run; NOT RUN is never a pass.** Each gate states a required part (what CI or the local
  Chromium run must pass) and, where relevant, an optional lane that is a **separate item** (S5-40 other
  engines, S5-47 WebGPU on titanix) which stays open until the lane has run once and its record is cited. GPU
  or JAX work goes to titanix; nothing in this spec asks for a local JAX suite. Hub tests (`node --test`,
  Playwright Chromium) are light and may run locally.
- **Chunked and resumable**: every pre-registered study gates its per-cell stamp and resume (4.6), under the resolved study root (4.7); S5-10's
  resume test is a gate for the browser RunStore.

Per-item gates are the `gate` field in section 9. Headline gates:

| Concern | Gate |
|---|---|
| No model imports, name reserved | Allow-list guard test in hub CI: planted `aminx` and unlisted `torch` fail, a listed package passes; S3-24's `--reserved aminx` fails a planted project name `aminx` and passes the hub tree (S5-01) |
| Base path | e2e harness and deploy dry-run pass under `/`, `/aminx-hub/`, `/praxia-science/`; planted root-absolute link fails the lint (S5-01, S5-02, S5-32) |
| Resolvers and caches | Both resolver chains tested; Node cache rejects a poisoned entry, survives concurrent writers (S5-03, S5-38) |
| Catalog determinism and integrity | Build twice -> equal sha256 of `catalog.json`; schema-invalid and sha-mismatch fixtures fail (S5-04) |
| Hosting policy | Forbidden-redistribution `pages` fixture fails the build; over-budget fails (S5-06) |
| Driver trust | Parsed-AST scan finds none of import specifiers, fetch, XMLHttpRequest, dynamic import, importScripts, WebSocket, sendBeacon in the staged driver, each rule has a planted-violation negative; sha256 matches; a changed driver or host without a review record fails (S5-50, re-run on the real bundle by S5-53; S5-14 builds it) |
| ORT-Web correctness | Implementation (S5-15, CI/Playwright on dev-staged real bytes from the lock): split P07 bucket 128 tokens equal the manifest golden vector exactly (wasm path), corrupted-artifact negative. Study (S5-57, pre-registered): 8 cells at 1 and 4 threads, resumable |
| Pyodide parity | Implementation (S5-17): three isochore cells equal committed native-CPython expected files (assertion). Study (S5-58, pre-registered): the same three cells within the pre-registered tolerance, resumable; `import mdtraj` yields the typed unavailable error |
| code.python | numpy runs; `import jax` gives `E_CODE_IMPORT_UNAVAILABLE`; sandbox canary `fetch` fails with a reachable-without-sandbox control (S5-43) |
| Unit-key semantics | A unit stored by executor A is recomputed when resumed on B (S5-10); server and ORT-Web keys differ (S5-21) |
| Privacy and consent | Request-interception audit: no request outside the declared host set; consent grant precedes the first remote POST; the real remote executor throws without a grant, with a wrong-origin grant, and with a different `input_sha256` (S5-20, S5-28; base class S5-37) |
| Resume | Close the page after 2 of 4 units, reopen: 2 reused with matching hashes, 2 computed, tampered unit recomputed (S5-10) |
| Isolation | `run/` isolated, other pages not (negative control), no COEP-blocked subresources incl. the viewer (S5-11, S5-12); real origin at first deployment (S5-33) |
| Duplicate-MPNN evidence | Pre-registered study (S5-27): cell 0 weight-identity check, then 12 resumable comparison cells on teacher-forced per-position log-probs against a frozen reference, with a planted-perturbation negative and a self-comparison positive |
| ExecutionClient adapter | `HubExecutionClient` passes S6's `runExecutionClientContract` (S5-30) |
| Release and lock | The real aminx tag's assets match the manifest sha256s; `catalog update` against it writes a lock that `--check` accepts (S5-52, S5-53) |
| Never-deployable dev tier | The deploy workflow fails on a build carrying `DEV-ONLY.marker` (S5-32) |
| First deployment | Pages source Actions on the hub repo; deployed-origin smoke passes under the base path in `hub.toml`; one real model artifact loaded and hash-verified; the only direct S3 dependency is S3-01 (S5-33) |

## 8. Open questions for the user

**Q1. Hub identity: repo slug, PyPI, domain, timing.** Superseded by S3-01 (revised D2: aminx keeps its name, the hub
is separate; S3 Q16-Q20). S3's recommendations: repo `maraxen/aminx-hub` (alternative: a praxia.science-derived slug
such as `praxia-science`); no PyPI project, never `aminx`; custom domain as the subdomain `hub.praxia.science` after
the first deployment (S3-29); bathos slug `aminx-hub`; no `praxia` in any non-DNS name because of the orchestrator
collision. S5 builds to any of them (`base_path`-agnostic, A40). Recommendation: create the hub repo only after S3-01
is answered (S5-01 depends on it), deploy on the project URL first (S5-33), then attach the domain (S3-29). Nothing in
the hub waits on the S1, S2 or S4 code items beyond the `depends_on` edges.

**Q2. The duplicate browser MPNN (aminx ONNX vs localfold).** Facts changed the question: localfold's MPNN is
CPU (JS + optional wasm SIMD), not WebGPU (A7); it mirrors an external checkout; it already ships SolubleMPNN,
LigandMPNN and NA-MPNN, which aminx does not export to the browser; its numerical agreement with reference
ProteinMPNN is not recorded anywhere I could read. aminx's ONNX path has recorded parity and benchmarks.
Options: (A) keep both, listed as two executors of one family; (B) hub exposes only localfold's; (C) hub
exposes only aminx's; (D) aminx ONNX is the validated default for ProteinMPNN/SolubleMPNN, localfold's is
offered for Ligand/NA families and as an opt-in cross-check. **Recommendation: D, decided after the
pre-registered study S5-27**, at the decision item S5-46. Nothing is deleted before then.
The study compares *teacher-forced per-position log-probs against a frozen reference*, after a weight-identity check; if
the two checkpoints are not the same weights (A35), it is descriptive only and cannot support option D on parity grounds.

**Q3. localfold integration.** (A) link-out with job JSON, (B) deep-import library adapter now, (C) request an
upstream entry point first. **Recommendation: A now, C in parallel (S5-23, S5-25), B only after C.** B today
couples to a 0.0.x API below featurisation and an iframe route is impossible (A6). S5-23 carries the decision
flag; the link cannot carry the job, so the user drags the file in (4.4).

**Q4. AF3 in the hub.** AF3 params are under DeepMind's Prohibited Use Policy and not redistributed by
localfold. Should the hub list AF3 at all? **Recommendation: yes, as link-out/`byo` only with the terms
shown; the hub never hosts or proxies the params.** Its own item (S5-45) carries the flag.

**Q5. Build toolchain.** Node/ESM (recommended: one toolchain with Playwright and the S6 component, `ajv` for
schemas) versus keeping the Python stdlib `build_site.py` style. **Recommendation: Node**, Python only as the
bathos shim. S5-01 carries the flag; every hub item depends on it transitively.

**Q6. isochore.** Move `mdtraj` (and `h5py` if unused at import) to an extra so the wheel is pure and
`deps=False` is unnecessary (S5-16)? **Recommendation: yes.** It changes a published dependency contract of
your own package.

**Q7. py2Dmol plugin-system merge.** Fast-forward `plugin-system-impl` into `main`, run py2Dmol's own suite,
tag a release the hub pins (S5-13)? **Recommendation: yes**, after the suite passes; the hub core does not wait
on it (only the density overlay S5-36 does). It rebuilds tracked bundles (+3.3% on `embed.min.js`).

**Q8. Weights on Pages versus elsewhere.** Recommendation: the hub carries code and catalog only by default;
small, sign-off-approved ONNX goes to Pages via staging, bigger or third-party weights stay on HF/`origin`.
Requires your explicit redistribution sign-off per model (S5-07), as `pages.yml` already says; S5-33 depends on
it. If you decline, the fallback is the conditional item S5-56 (HF ONNX upload with a CORS probe, A25), which is in
the DAG and closes as not applicable if you approve. Until the decision, development and CI use the never-deployable
`dev` tier (4.7). This also retires the repo-variable switches (`AMINX_MODELS_RELEASE_TAG`, `AMINX_MODEL_BASE_URL`), which
were never set.

**Q9. Tool vs model manifests.** S4 now has `kind: "model|tool"` (A18), but a weights-less tool, an external
(`origin`/`byo`) artifact and a `pyodide` file mapping are not expressible (A32, G1-G3). **Recommendation: accept
the four amendments G1-G4 as S5-51** (an S5-authored change to the S4 contract, so it needs your yes and the S4
author's). S5-42 and S5-51 carry the flag.

**Q10. Driver packaging.** (A) one pre-bundled single ES module built in aminx with a dev-dependency bundler
(recommended; the hub stays bundler-free), (B) a hash-listed multi-file set, (C) the hub bundles drivers.
**Recommendation: A**, with the parsed-AST scan (S5-50) as the guard. S5-14 carries the flag.

**Q11. Hub licence, vendored notices and repo creation (round 2, R2-C14e).** (a) Which licence does the hub's own
code carry (aminx is MIT; py2Dmol is Beerware; localfold Beerware)? **Recommendation: MIT**, to match aminx. (b) The hub
vendors Pyodide (MPL-2.0: source-availability notice), ORT-Web (MIT), coi-serviceworker (MIT), py2Dmol (Beerware) and
reviewed model drivers: **Recommendation: `vendor.lock.json` requires `license` + `notice` per entry and a NOTICE
file is generated and checked (S5-63)**; legal review of MPL-2.0 obligations is yours. (c) Who creates
the hub repo (slug from S3-01) and sets Pages to Actions: you (a user action); S5-01 records the slug and licence.
- **Requests from S6 (CH2-06).** Q17(a) and Q17(b) (dependency guard scope; workspace-sourced vendor entry) are answered with recommendations in section 6, "Requests from S6"; both are visible, reversible change requests. Confirm or override them there.

## 9. Backlog items

Every `depends_on` entry is an item id (S3/S4/S6 contract tokens were resolved in round 1; the binding table is
in section 6). Item ids are stable; S5-05 was withdrawn in round 1 and S5-55 was retired in the no-rename revision, and neither id is reused. Order of work is
expressed only through `depends_on`.

Conventions (round 2): items marked conditional (S5-56) are in the DAG under every outcome and close with a
recorded not-applicable note when their condition is false; a dependency on them is satisfied by that note. Items that
publish a release or create a repo carry `user_decision = true` because the user performs them. Study items (S5-57,
S5-58, S5-27, S5-59, S5-61) are separate from the implementation items they follow (4.6). Implementation items that
need real model bytes use the never-deployable `dev` tier (4.7) until S5-07 and S5-33.

```toml
[[item]]
id = "S5-01"
title = "Hub repo skeleton: layout, Node toolchain, CI running node --test, base-path url() lint, dependency allow-list guard (no-model-imports), reserved-name guard (S3-24 check_stale_names.py --reserved aminx, vendored into tools/hygiene), LICENSE per Q11; the user creates the repo under the slug S3-01 recorded and S5-01 records it in hub.toml"
repo = "aminx-hub"
size = "M"
depends_on = ["S3-01", "S3-24"]
gate = "CI on the skeleton PR runs node --test green; the guard fails for a planted aminx dependency and for an unlisted torch dependency and passes for a listed one; the reserved-name run (--reserved aminx) fails for a planted pyproject.toml project name of aminx and for a planted package.json name of aminx and passes on the hub tree (negative and positive controls); the lint fails on a planted root-absolute href; LICENSE exists and hub.toml records the slug that S3-01 recorded"
user_decision = true

[[item]]
id = "S5-63"
title = "Vendored licence notices: vendor.lock.json entries require license and notice, NOTICE generated and checked in CI"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-11"]
gate = "the generator fails on a planted vendor.lock.json entry lacking license or notice, passes on a complete one, and the NOTICE lists every entry (Pyodide MPL-2.0, ORT-Web, coi-serviceworker, py2Dmol, reviewed drivers)"
user_decision = false

[[item]]
id = "S5-02"
title = "Playwright e2e harness with a header-controlled dev server (COOP/COEP per route) served under /, /aminx-hub/ and /praxia-science/ prefixes, pinned Chromium"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-01", "S5-03"]
gate = "npx playwright test e2e/harness passes under each prefix with traces and browsers resolved through the S5-03 path table: crossOriginIsolated true on the isolated route and false on the plain route (negative control)"
user_decision = false

[[item]]
id = "S5-03"
title = "hub.toml schema (incl. [site] base_path) and the layered resolvers (Node cache: arg > env > hub.toml > XDG > off; browser: URL param > localStorage > hub.config.json > off; derived-data path table study_dir/build_dir/browsers_dir/e2e_dir: arg > env > hub.toml [paths] > XDG > fail loudly) with cacheRootSource() reporting"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-01"]
gate = "unit tests cover every layer of every chain, none/empty disables the cache, an unresolved study_dir raises with a message naming the four layers, a malformed file raises, the deciding layer is reported, and a malformed base_path is rejected"
user_decision = false

[[item]]
id = "S5-59"
title = "Bathos shim check for node/Playwright cells: stdlib-only Python shim spawns a node cell so the sidecar resolves (A17), pre-registered, with a failing-criterion negative"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-02", "S5-03"]
gate = "with the sidecar committed first, bth run --project-slug aminx-hub -- uv run --no-sync python3 scripts/studies/shim_check.py yields a record whose outcome evaluated to pass (bth sql after bth compact), and a second sidecar with a deliberately failing criterion yields outcome fail"
user_decision = false

[[item]]
id = "S5-38"
title = "Node/CI cache store: sha256-keyed, validate-on-read, stale rejection, atomic per-writer write, resolved through the S5-03 Node chain"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-03"]
gate = "tests: a poisoned entry is rejected and refetched, two concurrent writers do not collide, a URL or temp name never becomes a key, and the value none disables the cache"
user_decision = false

[[item]]
id = "S5-04"
title = "Catalog builder: sources.toml -> schema-validated S4 manifests -> catalog record derivation per the 4.2 mapping table -> catalog.lock.json -> deterministic dist/catalog/catalog.json with per-executor availability, with --check"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-03", "S5-38", "S4-06", "S4-09", "S6-35"]
gate = "building twice yields byte-identical catalog.json (sha256 equal) from fixture manifests that validate against the pinned S4 schema_version recorded in schemas/; S5-04 reimplements catalog_view in JS (the hub has no Python), schemas/ records the S4-09 commit it was built against (S5-64 later re-pins to the S4-34 release), and for every fixture in S4-09's conformance/catalog_view/ corpus the derived port dtype/shape equals the committed snapshot (mandatory: S5-04 depends on S4-09, a missing or empty corpus fails the gate, and no hub-written snapshot substitutes for it); missing determinism yields 'undeclared'; a schema-invalid manifest, a sha256 mismatch and a stale lock each fail with a named error; catalog records carry params, description, port accepts_encodings/alphabet (when declared) and manifest_sha256 (JCS form via the S6-35 canonicaliser) so S6-33 is a verification no-op; the computed manifest_sha256 equals the S4-30 Python value on a shared fixture manifest, and reformatting that fixture (whitespace, key order) leaves manifest_sha256 unchanged while changing asset_sha256 (negative control); sources.toml fields are asset_sha256 and the lock holds both hashes"
user_decision = false

[[item]]
id = "S5-51"
title = "Request amendments G1-G4 to the S4 contract (weights-less tool, external/byo artifacts, pyodide fs mappings, determinism on every executor kind) and land them in xtrax-contract"
repo = "xtrax"
size = "M"
depends_on = ["S4-06", "S4-09"]
gate = "xtrax-contract emit --check exits 0 and its tests accept: a kind tool manifest with no weights artifact, an external artifact without urls/revision, a pyodide entry with fs_in/fs_out, determinism on pyodide/npm/remote, and reject a planted non-external artifact missing sha256; regression check only (the grid types and the catalog_view corpus are S4-04 and S4-09 deliverables, not this item): the compat table still rejects a count-grid wired to an energy-map after the amendments"
user_decision = true

[[item]]
id = "S5-64"
title = "Re-pin schemas/ to the released xtrax-contract that carries the S5-51 amendments (G1-G4) and re-run the JS catalog_view derivation against the released snapshot corpus"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-04", "S5-51", "S4-34"]
gate = "schemas/ records an xtrax-contract release tag whose history contains the S5-51 commit (if the S4-34 release predates it the item is blocked and the user is asked for a new xtrax-contract release); the hub build validates a weights-less tool fixture and an external-artifact fixture; every released catalog_view snapshot matches the JS derivation; the old unamended pin is gone"
user_decision = true

[[item]]
id = "S5-06"
title = "Hosting policy: tiers pages/hf/origin/byo/dev, size budgets, content-addressed staging of verified release bytes, never-deployable dev mode (DEV-ONLY.marker), refusal of stray .onnx.data and of unreviewed or forbidden redistribution"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-04"]
gate = "fixtures: a forbidden-redistribution artifact at tier pages fails the build, an over-budget artifact fails, a planted stray .onnx.data fails, an artifact at tier pages without a review record fails while the same artifact under --mode dev stages and writes DEV-ONLY.marker, an allowed+reviewed artifact is staged with sha256 verified"
user_decision = false

[[item]]
id = "S5-07"
title = "Redistribution decision and sign-off records per model the hub may host (ProteinMPNN/SolubleMPNN/LigandMPNN weights, isochore wheel): approved or declined, consistent with S4-37"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-06", "S4-37"]
gate = "catalog/licenses/<model>.review.toml exists with decision approved|declined, reviewer, date and source URL for every candidate model, the build reads it, and a declined model cannot reach tier pages; for each MPNN weight set an approved record exists only where S4-37's license.redistribution is allowed (gated or forbidden forces declined or a recorded S4-37 amendment) and names the same reviewer, and a planted mismatch fails the build"
user_decision = true

[[item]]
id = "S5-56"
title = "Conditional HF fallback: if S5-07 records a decline, upload the aminx ONNX files to a pinned Hugging Face revision and probe CORS from the Pages origin; otherwise close as not applicable"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-07", "S5-52"]
gate = "if S5-07 declined: every ONNX file is on HF at a 40-hex revision, a CORS probe (Origin set to the hub origin) shows ACAO on the redirect and the CDN hop for each, catalog tier is hf with sha256 verified on load; if S5-07 approved: closed with a one-line not-applicable record"
user_decision = true

[[item]]
id = "S5-08"
title = "Executor core: Executor/Session/event contract with explicit cancel, capability probe (WebGPU adapter, isolation, memory, SIMD) and PrivacyDescriptor schema"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-02", "S5-03", "S4-04"]
gate = "unit tests with a mocked navigator cover each capability branch (no adapter is a reason code, not a throw), cancel kills only the named unit's worker, and a Playwright probe returns a schema-valid report"
user_decision = false

[[item]]
id = "S5-09"
title = "CacheStore: none/cache-api/opfs backends, content-addressed by sha256, verified on read, quota check, evict/clear, config-resolved namespace"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-02", "S5-03"]
gate = "Playwright Chromium (required part): round-trip passes, a corrupted entry is rejected and refetched, the same bytes from two URLs share one entry, backend none stores nothing; other engines are not claimed here (S5-40)"
user_decision = false

[[item]]
id = "S5-40"
title = "Cross-engine storage lane: run the CacheStore gate in Firefox and WebKit and record storage.estimate() quotas"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-09"]
gate = "Playwright firefox and webkit projects pass the CacheStore gate, or the failing engine is recorded and the backend falls back to none with a visible warning (tested); the item stays open until both engines have run"
user_decision = false

[[item]]
id = "S5-10"
title = "RunStore: per-unit IndexedDB persistence with executor-aware unit keys, input/artifact hashes and resume that reports reused units"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-08", "S5-09"]
gate = "Playwright closes the page after 2 of 4 units and reopens: units 1-2 reused with matching hashes, 3-4 computed, a tampered stored unit is recomputed, and a unit stored under executor A is recomputed when resumed on executor B (negative control)"
user_decision = false

[[item]]
id = "S5-37"
title = "Planner and consent: plan() over segments, ValueRef and transfer-encoding registry, ConsentGrant hook and the GrantCheckedExecutor base class"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-08", "S5-10", "S4-10"]
gate = "unit tests: plan() of a fixture three-node document is deterministic, remote is never auto-selected, every cross-segment edge carries a named encoding, and a stub subclass of GrantCheckedExecutor throws on load/run without a ConsentGrant, with a grant bound to the wrong origin, and with a different input_sha256 (the real remote executor repeats these in S5-20)"
user_decision = false

[[item]]
id = "S5-11"
title = "coi-serviceworker vendored and scoped to run/ under base_path, with a COEP subresource audit of the run pages"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-02", "S5-03"]
gate = "Playwright against a header-less server under /aminx-hub/ and /praxia-science/ with the service-worker script under run/: the first visit is not yet isolated and the second is (A34), run/ is crossOriginIsolated, other pages are not, and no subresource is blocked by COEP"
user_decision = false

[[item]]
id = "S5-12"
title = "Viewer adapter over a self-hosted, hash-pinned py2Dmol bundle (first step: read the bundle's JS API and amend mountViewer), crossorigin attribute, graceful fallback"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-02", "S5-11"]
gate = "Playwright on the isolated run/ route renders a fixture structure to a non-blank canvas with zero COEP-blocked requests; an id fetch to a reachable host succeeds AND an id fetch with the host blocked (and a planted COEP-blocked response) shows the paste-a-file fallback while run controls stay enabled; a blocked bundle shows the fallback; the vendor.lock.json sha256 check passes"
user_decision = false

[[item]]
id = "S5-13"
title = "py2Dmol: fast-forward plugin-system-impl into main, run its suite, tag a release for the hub to pin, ship the volume plugin as a separate file"
repo = "py2Dmol"
size = "M"
depends_on = []
gate = "git merge-base --is-ancestor plugin-system-impl main exits 0, py2Dmol tests/run.sh passes on the merged main with tracked bundles rebuilt, and the tagged release exposes the volume plugin file"
user_decision = true

[[item]]
id = "S5-14"
title = "aminx publishes its browser driver as one pre-bundled ES module built from a bytes-only entry (createSession over bytes, adapter over createSplitSampler, string-URL branch of loadSession removed from the bundled path) as a release asset, referenced from the native manifest"
repo = "aminx"
size = "L"
depends_on = ["S3-12", "S4-19", "S4-14"]
gate = "a release dry run (S3-12's workflow_dispatch path, which never publishes) builds the single driver .mjs and records its sha256 (build output and release-notes template; the S4-19 source manifest keeps a release-asset: placeholder for driver.url and driver.sha256, which S5-52 fills); a text check in aminx finds no fetch/XMLHttpRequest/importScripts/import( in the bundle (the authoritative parsed-AST scan is S5-50, re-run on the real bundle by S5-53); a Node test creates a session from the bundle with byte artifacts and reproduces the golden tokens; node --test browser/aminx-sampler passes; the URL-accepting Node/smoke entry points still work"
user_decision = true

[[item]]
id = "S5-52"
title = "aminx browser-assets release: tag carrying the S4 native manifests, split graphs for buckets 128 and 256, the driver asset and golden vectors; a script rewrites every release-asset: URL to the real release URL, fills driver.url and driver.sha256 from the S5-14 bundle, and lists every asset sha256 in the finalised manifest"
repo = "aminx"
size = "M"
depends_on = ["S5-14", "S4-18", "S4-19", "S3-12"]
gate = "gh release view <tag> lists the manifest, graphs, driver and golden-vector assets and the rewrite script leaves no release-asset: placeholder, the manifest validates against the S4 schema in non-source mode, each asset's sha256 equals its manifest entry and the driver sha256 equals the S5-14 bundle's; the version is the next unused alpha read from the PyPI JSON for aminx at cut time under S3-12's release-train rule (0.2.0a4 on 261001; never v0.2.0a3, whose asset is the old MANIFEST.json shape), stated in the PR, and the release goes through the guarded workflow; S1-25 depends_on S5-52 (edit lands in S1) and takes the next alpha (0.2.0a5), so this tag does not contain the S1-25 flip; the gate states whether the tag contains the S4-20 generated knob schema (it need not), and every python module:symbol string in the manifest resolves in the tagged tree"
user_decision = true

[[item]]
id = "S5-53"
title = "Hub adds the aminx source to catalog/sources.toml, runs catalog update against the real tag, commits catalog.lock.json with the S5-50 review records, and re-runs the driver scan on the real bundle"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-52", "S5-50", "S5-04"]
gate = "build.mjs --check passes against the real tag; sources.toml pins asset_sha256 and the lock records manifest_sha256; catalog.json has the aminx entry with ort-web availability; catalog/reviews/<driver sha256>.toml exists and the S5-50 scan passes on the real driver"
user_decision = false

[[item]]
id = "S5-50"
title = "Driver and host-list review gate: parsed-AST driver scan (import specifiers, fetch, XMLHttpRequest, dynamic import, importScripts, WebSocket, EventSource, sendBeacon, eval/Function, self/globalThis aliases), catalog update PR carries driver and host-list diffs, CI requires a reviewer record per driver sha256"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-04", "S5-08"]
gate = "fixtures: one planted driver per scan rule fails (negative controls) and a clean driver passes (positive control); an update that changes a driver sha256 or adds a host without a catalog/reviews record fails CI, the same update with a record passes, and the update artifact contains both diffs"
user_decision = false

[[item]]
id = "S5-15"
title = "ORT-Web executor (implementation): vendored ORT-Web 1.30.0 wasm EP, hub-staged driver in the driver-host worker (ORT imported inside the worker), per-design units, PrivacyDescriptor"
repo = "aminx-hub"
size = "L"
depends_on = ["S5-08", "S5-09", "S5-10", "S5-06", "S5-11", "S5-53"]
gate = "CI/Playwright on dev-mode staged real artifacts from the lock: the driver loads from dist/drivers/<sha256>/, split P07 bucket 128 tokens equal the manifest golden vector exactly (wasm path) at 1 and 4 threads, a corrupted artifact is refused, a planted driver that calls fetch is rejected by the S5-50 scan, a killed run resumes through RunStore, and the ort-web PrivacyDescriptor validates against its schema"
user_decision = false

[[item]]
id = "S5-57"
title = "ORT-Web hub re-check study (pre-registered bathos, local or titanix): 8 per-design cells at 1 and 4 threads, each in its own process, resumable, with a corrupted-artifact negative"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-15", "S5-59"]
gate = "sidecar (metric: exact token equality to the manifest golden vector on the wasm path, 120 s per cell) committed before the run, the bathos record shows outcome evaluated (bth sql after bth compact), a killed study resumes reusing stamped cells under the resolved study_dir, and the corrupted-artifact cell is refused"
user_decision = false

[[item]]
id = "S5-16"
title = "isochore: pure-python wheel (mdtraj moved to an extra) so Pyodide installs it without deps=False"
repo = "isochore"
size = "M"
depends_on = []
gate = "isochore CI with the mdtraj extra installed passes, and the built wheel installs in a clean venv without mdtraj where import isochore and read_dx on a fixture work"
user_decision = true

[[item]]
id = "S5-54"
title = "isochore wheel release: build the pure py3-none-any wheel from S5-16, publish it as a release asset, record its sha256"
repo = "isochore"
size = "S"
depends_on = ["S5-16"]
gate = "the built wheel's tag is py3-none-any, it installs in a clean venv without mdtraj and read_dx works on a fixture, and the release asset list shows the wheel whose sha256 equals the one recorded for the tool manifests"
user_decision = true

[[item]]
id = "S5-62"
title = "Hub staging of the isochore wheel (tier pages, sha256 verified) and self-hosted Pyodide runtime with the lock packages under a runtime budget, with measured sizes"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-06", "S5-54", "S5-11"]
gate = "the wheel is staged to dist/artifacts/<sha256>/ and an over-runtime-budget fixture fails the build; Playwright on the isolated run/ route loads the Pyodide runtime and the wheel same-origin with zero COEP-blocked requests; measured sizes of numpy/scipy/h5py and the runtime are recorded in docs/executor-matrix.md (A37)"
user_decision = false

[[item]]
id = "S5-42"
title = "isochore tool manifests (kind tool, no weights): read_dx, smooth_density, validate (and write_dx) as separate pyodide entries with the file-system mappings of section 4.4 and the wheel sha256"
repo = "isochore"
size = "S"
depends_on = ["S5-16", "S5-64", "S5-54"]
gate = "the manifests validate against the S4 ModelManifest schema at the S5-64 pin carrying the S5-51 amendments (tool without weights, pyodide fs mappings), declare exactly those entries with exactly these port type strings (read_dx out energy-map, write_dx in energy-map, smooth_density counts in count-grid and out density-grid, never a bare grid with a meaning field), S4 compat applied to the declared port types accepts density-grid to grid and energy-map to grid (positive control) and rejects count-grid to energy-map (negative control), and a native CPython test calls each through its declared mapping on fixtures"
user_decision = true

[[item]]
id = "S5-17"
title = "Pyodide executor (implementation): pinned self-hosted Pyodide in a worker, micropip of the staged wheel, FS-mounted tool entries, per-call timeout with worker restart, typed UnavailableInBrowser errors, PrivacyDescriptor"
repo = "aminx-hub"
size = "L"
depends_on = ["S5-08", "S5-09", "S5-10", "S5-42", "S5-62"]
gate = "Playwright: three independent cells (read_dx on .dx bytes, smooth_density on a count histogram, validate on a map_agfe_*.dx set plus reference PDB text) equal committed native-CPython expected files within the tolerance recorded in each file; import mdtraj returns the typed unavailable error; a hung call is killed and the worker restarts losing one unit; the pyodide PrivacyDescriptor validates"
user_decision = false

[[item]]
id = "S5-58"
title = "Pyodide parity study (pre-registered bathos, local or titanix): three isochore cells against native CPython, each in its own process, resumable"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-17", "S5-59"]
gate = "sidecar (metric: max abs difference per output, tolerance fixed in the sidecar, 60 s per cell) committed before the run, the bathos record shows outcome evaluated, a planted perturbed expected file is detected (negative), and a killed study resumes reusing stamped cells"
user_decision = false

[[item]]
id = "S5-43"
title = "Pyodide code.python capability: import allow-list from the lock, trusted flag, sandbox worker with connect-src none"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-17", "S5-37"]
gate = "Playwright: a numpy node runs; import jax gives E_CODE_IMPORT_UNAVAILABLE; an untrusted run refuses code nodes; a canary fetch fails in sandbox mode while the same fetch succeeds without the sandbox (control that the canary is reachable)"
user_decision = false

[[item]]
id = "S5-18"
title = "Capability and hosting probe: scheduled CI assertion test over the Pyodide lock, jaxlib wasm wheels, ORT-Web version and CORS of release/HF hosts, plus a manually run pre-registered bathos probe; generates docs/executor-matrix.md"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-02"]
gate = "CI part: a fixture with a planted jaxlib emscripten wheel flips the matrix entry (positive control), real data reports no wasm wheel, and the CORS probe covers the release, HF and staged py-wheel hosts; manual part: a local bathos record of the probe (sidecar committed first) shows outcome evaluated"
user_decision = false

[[item]]
id = "S5-19"
title = "Remote executor wire protocol v1: JSON Schemas, SSE events, error taxonomy, capabilities carry rng_scheme and runtime id, conformance suite and mock server"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-08"]
gate = "the conformance suite passes against the mock and fails against a mock without an Origin check and against a mock echoing a wrong manifest_sha256 (two negative controls)"
user_decision = false

[[item]]
id = "S5-20"
title = "Remote client: endpoint/token entry (fragment token, memory only), consent screen minting a ConsentGrant, provenance echo check"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-19", "S5-37"]
gate = "Playwright with request interception: no request leaves to an undeclared host, no POST precedes the consent screen, the token appears in no requested URL or localStorage, and the real remote executor throws without a ConsentGrant, with a grant for a different origin, and with a different input_sha256 (negative controls)"
user_decision = false

[[item]]
id = "S5-21"
title = "aminx runtime server implementing wire v1 (aminx serve) with allow-listed exact CORS origin, manifest hash echo and rng_scheme declaration; owns adding the remote executor entry to the S4-19 source manifests"
repo = "aminx"
size = "L"
depends_on = ["S5-19", "S4-19"]
gate = "wire conformance suite passes against the server on titanix, the server echoes manifest_sha256 equal to sha256(JCS(manifest)) from xtrax_contract.canonical, the S4-19 source manifests gain a remote executor entry (protocol xtrax-http@1, runtime pointer to aminx serve, determinism.rng_scheme) that validates against the S4 schema (manifest diff attached), a fixed-seed request returns the same tokens as a native sample, and the declared rng_scheme differs from ORT-Web's so cross-executor unit keys differ, and `uv run pytest tests/parity/test_browser_validation_inventory.py` is green: any public top-level def the item adds under `src/aminx/{inference,sampling,scoring,model,host,tiling,ebm,potts}` or to `aminx.run.__all__` (the `aminx serve` runtime is kept out of those packages, or) is mapped in `tests/parity/browser_validation_paths.json` in the same PR, with the internal-symbol baseline file updated if it counts (A42)"
user_decision = false

[[item]]
id = "S5-22"
title = "Colab launcher notebook that installs the runtime named in the manifest's remote executor entry (added by S5-21), starts it, and prints endpoint URL and token with the tunnel-provider caveat"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-19", "S5-21"]
gate = "notebook JSON lints, its launch cell runs against the mock executor in CI, and the printed endpoint passes the wire capabilities check"
user_decision = false

[[item]]
id = "S5-23"
title = "localfold link-out executor: AF3 job JSON builder (server and open dialects) tested against localfold's job-json reader vendored at a pinned commit with its licence in vendor/localfold, download plus link with the drag-the-file-in step stated, result-zip ingestion into the viewer; privacy panel states the leave-the-site step"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-08", "S5-12"]
gate = "generated job JSON round-trips through the vendored web/job-json.js (with its src/chem/smiles.js and entities.js dependencies, sha256 in vendor.lock.json) in a node test, a SMILES ligand fixture is emitted in the open dialect, the UI text states the manual drop-in step (asserted in Playwright), and the localfold-link PrivacyDescriptor validates"
user_decision = true

[[item]]
id = "S5-60"
title = "localfold MPNN node harness: run the vendored localfold JS MPNN (ProteinMPNN family) in Node on a fixture structure with fixed tokens and emit per-position log-probs"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-23"]
gate = "the harness emits per-position log-probs for a fixture, two runs are byte-identical (positive control), and a copy with one perturbed weight tensor differs beyond a stated threshold (negative control); no finding is claimed"
user_decision = false

[[item]]
id = "S5-61"
title = "aminx frozen reference for the cross-implementation study: teacher-forced per-position log-probs and a float32 weights tensor dump for proteinmpnn_v_48_020 on four fixtures, hashed, produced by a pre-registered script (first step probes A36 on a toy input)"
repo = "aminx"
size = "M"
depends_on = ["S4-19"]
gate = "sidecar committed before the run; the bathos record shows outcome evaluated; reference_scores.json and weights dump are published with their sha256; the toy probe records whether all-fixed split-loop input reproduces the Python scorer, and if it does not the ONNX arm is dropped and S5-27 says so, and `uv run pytest tests/parity/test_browser_validation_inventory.py` is green: any public top-level def the item adds in the root packages is mapped in `tests/parity/browser_validation_paths.json` in the same PR (A42)"
user_decision = false

[[item]]
id = "S5-24"
title = "Catalog entries for localfold AF2, ESMFold2 and ESM-C as hand-authored manifests with origin hosting, license text and sizes"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-04", "S5-23", "S5-64"]
gate = "entries validate against the S4 ModelManifest schema at the S5-64 pin carrying the S5-51 external-artifact amendment, hosting is origin only, and the build stages no localfold weights"
user_decision = true

[[item]]
id = "S5-45"
title = "AF3 catalog entry: link-out/byo only with the terms shown, no hosting or proxying of params"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-24"]
gate = "the entry validates against the S4 schema at the S5-64 pin carrying the S5-51 amendment (byo artifact, redistribution forbidden), the build stages no AF3 file, and the page shows the DeepMind terms link"
user_decision = true

[[item]]
id = "S5-25"
title = "Upstream request to localfold (sole owner; S4-27 duplicates it and should drop its request half): a single sequence/job-JSON-in structure-out entry point and a postMessage or module API"
repo = "localfold"
size = "S"
depends_on = ["S5-23"]
gate = "an issue or PR URL is recorded in docs/ext-requests.md with the exact API asked for"
user_decision = true

[[item]]
id = "S5-26"
title = "localfold library executor: pinned-commit adapter behind the Executor interface and WebGPU probe, contract-tested against a stub"
repo = "aminx-hub"
size = "L"
depends_on = ["S5-23", "S5-25", "S5-09", "S5-10"]
gate = "contract tests against a stubbed localfold module pass in CI (required part; real-adapter execution is S5-47); a recorded user decision on vendoring web/af3-model.js (pinned commit, author consent) exists, folded in from S4-27"
user_decision = true

[[item]]
id = "S5-47"
title = "titanix WebGPU lane for the localfold executor: probe the browser setup, then fold a fixture"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-26"]
gate = "a recorded titanix run with a real WebGPU adapter folds a fixture and its record is cited; with no adapter the item stays open and the probe result is recorded"
user_decision = false

[[item]]
id = "S5-27"
title = "MPNN cross-implementation study (pre-registered): cell 0 weight identity of localfold vs aminx checkpoint, then 12 comparison cells (4 fixtures x 3 pairings among localfold JS, aminx ONNX, frozen reference) on teacher-forced per-position log-probs"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-57", "S5-60", "S5-61"]
gate = "sidecar committed before the run states the hypothesis, primary metric (max and median per-position abs log-prob difference), tolerance, cell count and 300 s timeouts; cell 0 (tensor-wise comparison at float16 rounding) runs first and a failure makes the study descriptive; the bathos record shows outcome evaluated; a planted weight perturbation is detected, an identical-implementation self-comparison passes, and a killed run resumes reusing stamped cells"
user_decision = false

[[item]]
id = "S5-46"
title = "Decision: which MPNN implementations the hub exposes (Q2), from the S5-27 record"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-27"]
gate = "a decision doc cites the S5-27 bathos run id and records option A/B/C/D, and the catalog reflects it"
user_decision = true

[[item]]
id = "S5-28"
title = "Hub shell: generated pages (home, model, run, privacy, executor matrix) from the catalog under base_path, per-page generated CSP (executor, artifact and viewer hosts)"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-04", "S5-08", "S5-12", "S5-11"]
gate = "Playwright renders every catalog entry's pages under /aminx-hub/, a link checker finds no dead internal link, a PrivacyDescriptor fixture per executor kind is rendered on each page (real descriptors are asserted in S5-15/17/20/23), a request outside the declared host set fails the audit, and run pages are isolated"
user_decision = false

[[item]]
id = "S5-48"
title = "Evidence badges derived from validation.evidence and per-page attribution text"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-28"]
gate = "snapshot tests: ORT CPU and ORT-Web evidence render as separate badges, a model without evidence shows none, and every page that runs a model shows its manifest attribution"
user_decision = false

[[item]]
id = "S5-49"
title = "Docs split: executor-agnostic parts of browser_integration.md move to hub docs, MPNN specifics stay in aminx"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-28"]
gate = "the hub docs build, a link checker passes, and a grep shows no MPNN-specific section duplicated between hub docs and the aminx file"
user_decision = false

[[item]]
id = "S5-29"
title = "Port the MPNN designer (site/*) into apps/mpnn-designer at run/mpnn-designer/ on the split-graph path with its node test suites"
repo = "aminx-hub"
size = "L"
depends_on = ["S5-15", "S5-12", "S5-28"]
gate = "the ported design_utils and pdb_parse node tests pass and an offline fixture structure designed at a fixed seed returns the golden tokens in Playwright on the isolated route, served from the dev-mode build"
user_decision = false

[[item]]
id = "S5-30"
title = "Browser pipeline runner for the S4 graph IR plus the HubExecutionClient adapter: topological execution over planned segments, per-node unit persistence, resume; passes S6's runExecutionClientContract"
repo = "aminx-hub"
size = "L"
depends_on = ["S5-15", "S5-10", "S5-37", "S4-10", "S6-03", "S6-07", "S6-11"]
gate = "a two-node fixture DAG runs deterministically and, with the second node killed mid-run, resumes reusing the first node's output by matching hash; HubExecutionClient passes runExecutionClientContract from S6 and a seeded-broken adapter fails it"
user_decision = false

[[item]]
id = "S5-31"
title = "Pipelines page embedding the S6 pipe-list element with a palette generated from catalog.json"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-30", "S5-28", "S6-10"]
gate = "Playwright loads /pipelines/, adds two catalog nodes from the generated palette, runs the graph through HubExecutionClient and sees both outputs"
user_decision = false

[[item]]
id = "S5-32"
title = "Deploy workflow: build on push to main, Pages via Actions, artifact manifest check that no non-allowed weights are staged, size budget enforced, base_path from hub.toml"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-28", "S5-06", "S5-02", "S5-63"]
gate = "workflow dry-run on a PR uploads an artifact under budget whose file listing contains zero files with redistribution other than allowed and includes the generated NOTICE; a planted build carrying DEV-ONLY.marker makes the workflow fail (negative control); the built site passes the e2e smoke under /aminx-hub/ and /praxia-science/"
user_decision = false

[[item]]
id = "S5-33"
title = "First public deployment: after the deploy workflow, the designer port, the sign-off or fallback and the real release in the lock, enable Pages with Actions on the hub repo (slug from S3-01), deploy, smoke and isolation probe on the real origin with a real model artifact; the user records the slug"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-32", "S5-29", "S5-07", "S5-56", "S5-53", "S3-01"]
gate = "gh api repos/maraxen/<slug>/pages shows build_type workflow, the deployed-origin Playwright smoke passes under the base_path in hub.toml, crossOriginIsolated is true on run/, the deployed origin loads, sha256-verifies and runs one real aminx artifact, and no S3 item beyond S3-01 is a direct dependency (S3-12 arrives only transitively through S5-53 and S5-52, and S3-24 through S5-01)"
user_decision = true

[[item]]
id = "S5-34"
title = "Retire the old browser site from aminx: remove site/, tools/build_site.py, pages.yml and re-point docs; keep browser/aminx-sampler and layer_c"
repo = "aminx"
size = "M"
depends_on = ["S5-33", "S5-14", "S5-49", "S3-21"]
gate = "aminx CI is green without site/, tools/build_site.py and pages.yml, the docs URL recorded by S3-21 still resolves (aminx.readthedocs.io and the stale Sphinx build are not touched by this item), and a repo search for build_site finds no references"
user_decision = false

[[item]]
id = "S5-35"
title = "JupyterLite page /lab/ on the pinned Pyodide with hub wheels preinstalled and S6 notebook export"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-17", "S6-18"]
gate = "Playwright opens /lab/, runs a notebook cell importing isochore and reading a fixture .dx, and the kernel Pyodide version equals the pinned one"
user_decision = false

[[item]]
id = "S5-36"
title = "GFE density viewer: isochore grid from the Pyodide executor rendered through the py2Dmol volume plugin, plugin file hash-pinned in vendor.lock.json"
repo = "aminx-hub"
size = "M"
depends_on = ["S5-13", "S5-17", "S5-12"]
gate = "Playwright loads a fixture .dx, renders an isosurface via the volume plugin (registered on window.py2dmolPlugins, sha256 from vendor.lock.json verified) to a non-blank canvas, and the plugin-absent path falls back to a message"
user_decision = false
```

## Revision log

> Naming note for the entries below. Every entry before "No-rename revision" was written before the 2026-10-01
> decision that aminx keeps its name. In those entries `molxmpnn` means today's `aminx`, `S3-02/03/07/13/22/23/25`
> are retired S3 items, "P1/P2" and S5-55 are void, and `browser/molxmpnn-sampler` is `browser/aminx-sampler`. They are
> kept as history; where one conflicts with the body above or with the last entry, the body and the last entry win.

**Round 1** (adversarial objections C1-C22). Applied everything conceded or partly conceded; the edits are in
place above.

Changed:
- **C1 (isochore chain).** Re-read `read_dx`, `smooth_density`, `validate` (A19). Three independent tool entries with
  FS mappings (4.4), new grid port types, S5-17 gate rewritten as three parity cells, tool manifest split into S5-42.
- **C2 (depends_on tokens).** Every token bound to a concrete S3/S4/S6 item id (section 6 map, A28); published the S5
  id map for S6 to rebind to (S5 ids kept, S6 rebinds). The S3 spec now has parseable items; ids are as of this
  revision and the coherence pass re-checks them.
- **C3 (contract surface for S6).** Added `cancel`, `plan()`, `ValueRef`/encodings, `ConsentGrant` (4.3, S5-37), the
  `code.python` capability (S5-43), and renamed the artifact to `catalog.json` with per-executor availability.
  Q9 outcome recorded: S6 owns `ExecutionClient` and its adapter (S6-11), S5 owns the planner.
- **C4 (no CORS host for ONNX).** S5-33 now depends on S5-07 and gates on loading a real artifact from the deployed
  origin; A25 added (unverified, HF egress denied here); Q8 states the HF-upload fallback.
- **C5 (driver packaging).** Single pre-bundled module built in molxmpnn, import-specifier scan, byte artifacts,
  adapter over `createSplitSampler` (A20); Q10 added; S5-14 is a user decision and size L.
- **C6 (driver trust).** Driver worker, CSP, review gate item S5-50, reworded the no-third-party-scripts rule.
- **C7 (unit key).** Key includes executor kind, driver sha256, rng_scheme; cross-executor reuse refused; negative
  control in S5-10; HUB-REQ-10 added.
- **C8 (base path).** `base_path` in `hub.toml`, `url()` helper, lint, e2e under three prefixes; designer placed at
  `run/mpnn-designer/`.
- **C9 (Pages/rename).** 4.9 and S5-33 rewritten around the old repo owning gh-pages; the new hub repo enables Pages
  with Actions; also reconciled with S3's option 1 (A22), which the first draft missed: the hub slug is now an S3-22
  outcome (P1/P2) and S5-33 depends on S3-02 and S3-22.
- **C10 (bathos venue).** Section 7 states local/titanix, slug `aminx-hub`, stdlib shim, CI assertions only; S5-18
  split into a CI part and a manual bathos part; the A17 shim check moved to S5-15.
- **C11 (Node cache).** New S5-38; S5-03 gates both resolver chains.
- **C12 (silent decisions).** `user_decision = true` on S5-23, S5-24, S5-42; new decision items S5-45 (AF3), S5-46
  (Q2); S5-27 is a study, not a decision.
- **C13 (viewer under isolation).** S5-12 depends on S5-11 and gates on the isolated route with zero COEP-blocked
  requests; viewer hosts feed CSP; A23/A24 added.
- **C14 (legacy adapter).** Withdrew S5-05 (id not reused); 4.2 and the scope map corrected.
- **C15 (NOT RUN passes).** Gates split into required part and separate lane items S5-40 and S5-47; A27 added.
- **C16 (S5-16 split).** S5-16 is the pure wheel with no S4 dependency; S5-42 is the manifest.
- **C17 (S5-02, S5-28).** Shim check moved to S5-15; S5-28 split into pages+CSP (S5-28), badges+attribution (S5-48)
  and docs split (S5-49).
- **C18 (ORT-Web evidence).** Cited existing evidence (A21); S5-15 is a hub re-check; golden vectors on the wasm path.
- **C19 (chunking in gates).** Cell counts, timeouts and resume checks added to S5-15, S5-17, S5-27; S5-27 now
  depends on S5-23 and names titanix for the reference run.
- **C20 (localfold details, volume plugin).** Manual drop-in step in S5-23's gate; volume plugin file pinned in
  `vendor.lock.json` and S5-13/S5-36, registration via `window.py2dmolPlugins` (read in `plugins.js:4-60`).
- **C21 (guard).** Allow-list guard with tests for a denied and an unlisted package.
- **C22 (numbering).** The scope map is folded into section 3 and the Goal is numbered 2, so sections match the
  required template (frontmatter counts as 1) and the `section 6` / `section 9` references stay valid.

Declined or limited:
- **C22, `owner_repos` comments.** Inline YAML comments are valid and did not alter the parsed list; they were removed
  anyway as tidying (no behaviour change).
- **C10, "Python shim needs a bathos dependency".** Not adopted: bathos is the installed `bth` tool; the shim is
  stdlib-only and the hub `pyproject.toml` has no bathos dependency.
- **C4, A25 spike.** Not run: no ONNX file is known on HF and huggingface.co egress was denied to this task, so the row
  stays UNVERIFIED with a deferred reason rather than a recorded spike. The same holds for A24, A26 and A27 (browser,
  titanix or pinned-bundle reads outside a read-only spec task); the spike harness is absent from this workspace.

### Round 2 (adversarial objections R2-C1 to R2-C14)

Changed:
- **R2-C1 (S4 shape, partial).** The objection cited an earlier S4 draft: the current S4 already has `schema_version`,
  `artifacts[]`, `executors[]`, `driver {url, sha256, export}`, `determinism`, and `redistribution` as the enum
  (line cites since re-done against the current file, see the convergence check). A18 flipped to VERIFIED; 4.2 now has an
  explicit mapping table (S4 field -> catalog record), S5-04 pins the S4 `schema_version`. The real gaps are filed as
  G1-G6 (weights-less tool, external/byo artifacts, pyodide fs mappings, determinism on non-onnx kinds, grid
  `meaning` in compat, JS-usable `catalog_view` corpus) with one request item S5-51, and S5-24, S5-42, S5-45 depend
  on it; their gates now name the amended schema version. A31/A32 added. `xtrax-http@1` is mapped to wire v1.
- **R2-C2.** New S5-52 (molxmpnn browser-assets release) and S5-53 (add source, `catalog update`, lock, review
  records). The "bot PR" claim was dropped: `catalog update` is a maintainer command. S5-15 and S5-33 depend on S5-53.
- **R2-C3.** S5-14 and S5-21 depend on S3-07 (and S3-12 for S5-14); every path and gate uses `browser/molxmpnn-sampler`
  (S3 Q10); S5-34 depends on S3-07 and S3-21. The scope map shows the rename.
- **R2-C4.** S5-33 no longer depends on S3-02 or S3-22 (needs S3-01 only); the P2 rename is S5-55 behind S3-22,
  S3-23 and S5-33; 4.9, risks, Q1 and section 6 rewritten; stated that no S5 item publishes to PyPI.
- **R2-C5.** 4.3 reworded: the driver host imports vendored ORT inside the worker and passes the namespace by
  reference; only bytes and inputs cross `postMessage`. S5-14 builds from a bytes-only entry (the `fetch` in
  `loadSession` is verified, A30); S5-50 owns a parsed-AST scan over seven rule groups with planted negatives.
- **R2-C6.** S5-37 gates on a stub `GrantCheckedExecutor` subclass; the real negative controls (no grant, wrong
  origin, different `input_sha256`) moved into S5-20.
- **R2-C7.** One owner: S5-30 writes `HubExecutionClient` and must pass S6's `runExecutionClientContract` (depends
  on S6-07, S6-11); 4.3 and section 6 corrected; the id map covers `S5:privacy-descriptor`, `S5:run-store`,
  `S5:pyodide-code-exec` (= `S5:code-python-capability`, S5-43) and flags S6's stale `index.json` file name.
- **R2-C8.** New never-deployable `dev` tier and `DEV-ONLY.marker` (S5-06, checked by S5-32); S5-15/S5-29 use it
  instead of S5-07. New conditional S5-56 (HF fallback with CORS probe); S5-07 is now an approve/decline decision.
- **R2-C9.** S5-15 and S5-17 are implementation items closing on CI assertions; new study items S5-57 (ORT-Web) and
  S5-58 (Pyodide) with sidecar-then-run; the A17 shim check is its own item S5-59.
- **R2-C10.** New S5-54 (isochore wheel release) and S5-62 (stage the wheel, self-host the Pyodide runtime under a
  `runtime` budget); `py-wheel` hosting stated (A38), CORS probe in S5-18; A37 added.
- **R2-C11.** S5-27 now states hypothesis, metric (teacher-forced per-position log-probs), a cell-0 weight-identity
  check (A35) and a frozen reference; new S5-60 (localfold node harness) and S5-61 (molxmpnn reference plus weights
  dump); the dependency moved from S5-23 to S5-60/S5-61/S5-57. A36 added.
- **R2-C12.** Gate fixes in S5-12 (forced-failure case), S5-23 (localfold source vendored at a pin with licence),
  S5-28 (descriptor fixtures), S5-06 (planted `.onnx.data`).
- **R2-C13.** 4.7 path table (study, build, browsers, e2e dirs) with arg > env > `hub.toml` > user config > fail
  loudly; gated in S5-03, cited by S5-02/S5-57/S5-58/S5-27.
- **R2-C14.** Ledger rows A33 (IndexedDB), A34 (SW scope and first-load reload), A29/A30 (ORT in worker, fetch); the
  `xtrax-http@1` mapping; Q11 (hub licence, NOTICE, repo creation) and new S5-63 (notice generation).

Declined or limited:
- **R2-C1, "fetch the OnnxBundle as a second document".** Declined: `OnnxBundle@1` is a manifest fragment (A31), so no
  second document exists. "S4 lacks `determinism`, `redistribution` enum, driver `{url,sha256,export}`" also declined
  (stale draft); only G1-G6 are filed.
- **R2-C1, "flip A18 to VERIFIED-DIVERGENT".** The ledger vocabulary has no such status; A18 is VERIFIED with a
  `changed:` note, A32 records the divergences.
- **R2-C14(c), "SW script must sit under `run/`".** Recorded as A34 (unverified) with a gate in S5-11 rather than
  asserted; not spiked because this task is spec-only and no browser harness exists in this workspace.
- **R2-C14 spikes generally.** A33-A37 stay UNVERIFIED with `deferred:` reasons (browser, titanix or measured-size
  work); no `S<n>` spikes were run because the spike harness is absent here and the task forbids running builds.
- **R2-C8, "alternatively add S5-07 to S5-15's depends_on".** Not adopted; the `dev` tier keeps S5-15 independent of
  the user's sign-off.
- S6's `index.json` versus S5's `catalog.json`: not changed unilaterally; S5 keeps `catalog.json` and asks S6 to rebind.

### Coherence round 1 (cross-spec fixes C2, C4, C8, C9, C11, C13, C14)

Changed:
- **C2.** 4.9 states that no hub artifact reserves `aminx.ebm` or any `aminx.*` namespace; S3 owns the sole shim.
- **C4.** S5-33 gate reworded: no S3 item beyond S3-01 is a direct dependency. Section 6 records that
  S5-14/21/34/52/61 are deliberately not ordered against S3-20 and the user-decision alternative.
- **C8.** S5-14 publishes the bundle and its sha256; S5-52 title and gate now own the release-asset: URL rewrite,
  driver.url/sha256 fill, non-source re-validation and asset hash check. Order recorded in section 6.
- **C9.** New S5-64 re-pins schemas/ to the released amended contract (depends on S5-04, S5-51, S4-34); S5-24, S5-42,
  S5-45 depend on it; S5-04 states it reimplements catalog_view in JS and is tested against S4's snapshots.
- **C11.** S5-07 depends on S4-37 and its gate requires agreement with license.redistribution and the same reviewer.
- **C13.** S5-25 is the sole upstream-request owner; S5-26 gate absorbs the af3-model.js vendoring consent decision.
- **C14.** S5-52 gate states the version (later than S3-13's 0.2.0a4, unused on PyPI, user-chosen) and whether it
  contains S4-20 (not required). Revised in coherence round 2 (CH2-01): S1-25 depends_on S5-52, S5-52 is the first
  alpha after 0.2.0a4 (0.2.0a5 unless S3's table says otherwise) and does not contain S1-25.

Not changed: S5 does not edit S2, S3 or S4 (S2-18/19, S3-28, S4-27, S4-34 fixes belong to those specs).

### Coherence round 2 (CH2-01, CH2-05, CH2-06, CH2-07, CH2-08)

Changed:
- **CH2-01.** Replaced "deliberately not ordered against S1-25" (S5-52 gate, section 6 "molxmpnn versions", the round 1 C14 note) with: S1-25 depends_on S5-52 (edit in S1); S5-52 is the first alpha after 0.2.0a4 (0.2.0a5 unless S3's table says otherwise) and does not contain S1-25. "Need not contain the S4-20 generated knob schema" kept. S5-52 gains no S1 dependency, so the first deployment does not wait for the S1 flip chain. Note: the round 1 C14 line above is superseded by this entry.
- **CH2-05.** sources.toml field renamed `asset_sha256` (download integrity). `manifest_sha256 = sha256(JCS(manifest))` added to the 4.2 record and mapping table, the wire protocol v1 echo, and the 4.6 unit key. The Node computation uses S6-35's canonicaliser; S5-04 now depends on S6-35 (no cycle: S6-35 <- S6-03 <- S6-02 <- S5-01/S5-02). S5-04 and S5-52/S5-53 gates updated.
- **CH2-06.** 4.2 mapping table and S5-04 gate now carry params, description, port accepts_encodings/alphabet and manifest_sha256, making S6-33 a verification no-op. New section 6 "Requests from S6" answers Q17(a)/(b); pointer added to section 8.
- **CH2-07.** S5-34 gate reworded to "the docs URL recorded by S3-21 resolves (or no stub is required if S3-21 records none is possible)"; section 6 Consumes row for S3-21 aligned. Not changed: S3-03's `MOLXMPNN_MODELS_RELEASE_TAG` requirement conflicts with Q8 retiring those variables; that is S3's item to reconcile (S5 does not edit S3).
- **CH2-08.** S5-21 now owns adding the remote executor entry (protocol xtrax-http@1, runtime pointer to `molxmpnn serve`, determinism.rng_scheme) to the S4-19 source manifests and depends on S4-19; S5-22 reads the runtime from that entry. Consequence stated: S5-52 is not ordered behind S5-21, so the first release may lack the entry and remote availability appears at the next molxmpnn release via an ordinary catalog update PR.

### Convergence check (R2-C1 residue; touched: this spec and `261001_xtrax-model-contract.md`)

Changed:
- **Line cites re-done (A18, A28, A31, A32; the context bullet in section 3; the license row of the 4.2 table).** The
  round-2 cites into S4 (`:444`, `:468-471`, `:484-489`, `:515-524`, `:396,411-418`, `:810-868`) pointed at unrelated
  lines of the then-current S4 file, so VERIFIED rows rested on text a reader could not find. Every cite is now a
  `read:` line of `261001_xtrax-model-contract.md` at sha256 prefix `44c4d220971dc0c1` (schema_version `:536`, ports `:542-547`,
  artifacts `:549-556`, executors `:557-562`, license `:563-564`, sha256 and 40-hex rule `:599-600`, catalog_view
  `:604-609`, tables `:665,668-669`, OnnxBundle `:677-684`, grid types `:474-477`, compat `:490-496`). The S3 and S6 cites in
  A28 (S3-02, S3-22, S6-03, S6-10, S6-18) were stale too and are re-done. The S4 revision is named in each row, and a
  reader on a different revision re-cites by section (4.2, 4.3, 4.5).
- **G5 withdrawn, grid types fixed (A39, A32).** S4 4.2 now defines `energy-map` (alias `gfe-grid`), `count-grid`,
  `density-grid` and rejects a `count-grid` to `energy-map` edge, so the sentence "S4 compares only dtype, rank, axes and
  alphabet" and the "replaces my round-1 names" claim were wrong. Option (a) taken: 4.3 and 4.4 declare isochore ports as
  `energy-map` (read_dx out, write_dx in), `count-grid` (smooth_density in) and `density-grid` (out); the S5-42 gate
  names the exact type strings with a positive and a negative compat control; the S5-51 gate keeps only a regression
  check on the compat rule. S4 4.3 (`:613-629`), its section 3.5 S5 row and Q13 were amended to the same position.
- **G6 removed.** S4-09 already commits `conformance/catalog_view/` (`:606-609`, `:1416-1422`), so G6 leaves the gap table,
  the S5-51 title and gate, A32, the risk row, Q9 and the section 6 rows. The amendment is G1-G4 everywhere (S4-34's
  title and gate say the same).
- **Pin rule reconciled (S5-04 versus S4-34).** Option 2 taken, stated in 4.2 and in S4 4.3: S5-04 deliberately builds on
  the S4-09 commit (it needs schemas and the corpus, not the amendments) and S5-64 re-pins it to the S4-34 release;
  S5-24, S5-42 and S5-45 go through S5-64 and pin the S4-34 release. S4's text, which listed S5-04 among the S4-34 pinners,
  was reworded. No `depends_on` changed, so no cycle is possible.
- **S5-04 gate hedge deleted.** The "else a hub snapshot, flagged" fallback is gone: equality with every snapshot in
  S4-09's `conformance/catalog_view/` is mandatory, and an empty or missing corpus fails the gate.
- **Ledger.** A39 added (VERIFIED by read). A18, A28, A31 and A32 re-cited; none of them was spiked, and none needs to be
  (all are code-reading claims about a file in this workspace).

### No-rename revision (261001, user decision revising D2)

Trigger: the user decided that the MPNN package, released on PyPI as `aminx`, keeps that name, and that the hub is a
separate project holding the combined implementations (working slug `aminx-hub`; the user also floated naming it
after the domain praxia.science, which works with any repo slug because a Pages custom domain is independent of the
slug). S3 was rewritten to match (`261001_aminx-identity-and-hub-naming.md`); this entry lists what S5 changed. The
hub's final repo name, PyPI policy and domain stay an explicit user decision (S3-01), and the collision with the
user's praxia agent orchestrator is flagged in the naming note after the title, in 4.9, in the risks and in S3-01.

Changed:
- **Names.** Every `molxmpnn` in sections 1-9 that meant the MPNN package or repo is `aminx` again: `repo = "aminx"`
  in the toml for S5-14, S5-21, S5-34, S5-52 and S5-61; `browser/aminx-sampler` keeps its name and path; the S5-21
  runtime is `aminx serve`. Where `aminx` used to mean the hub's brand ("the hub that inherits the aminx name",
  "brand `aminx`") the text now says the hub is a separate project with working slug `aminx-hub`, final name per
  S3-01. Front matter (title, description, owner_repos, related_specs) rewritten and a naming note added after the title.
  This was not a blind search-replace: the rename-only text below was rewritten or deleted instead.
- **Section 3 and ledger.** The "S3 recommends against reusing the slug" bullet became "the hub's identity is a user
  decision S3 records"; the scope-map rows for the sampler directory and for `gh-pages` no longer mention a rename;
  the Pages bullet says the stale Sphinx site stays with aminx. A22 re-meant (hub identity is S3-01's; cites S3 4.2, the
  reserved-name mode and S3-01), A1 re-stated, A28 re-cited to the S3 ids that exist (S3-01, S3-24, S3-12, S3-21,
  S3-29) with S3-02 and S3-22 dropped. New A40 (a Pages custom domain is independent of the repo slug; UNVERIFIED,
  deferred to S3-29) and A41 (S5-52 uses the existing trusted-publishing `release.yml`; VERIFIED by read). No spike
  harness exists in this workspace, so no `S<n>` spike was run for these rows.
- **4.1 base path and guards.** The e2e prefixes are `/`, `/aminx-hub/` and `/praxia-science/` (a custom domain, the
  working slug and the alternative slug); `/aminx/` is aminx's own docs site and is never a hub prefix (it was the P2
  test shape). The dependency allow-list now catches a planted `aminx` (was `molxmpnn` plus the "ambiguous name after
  cutover" text), and S5-01 additionally runs S3-24's reserved-name mode (`--reserved aminx`) from a vendored copy in
  `tools/hygiene/`, so the hub can never become a second importable `aminx`.
- **4.9 rewritten.** Removed: the in-place rename, the P1/P2 split, the redirect hazard, the deprecation window, the
  PyPI handover. Now: the hub is a separate repo whose identity is S3-01's; first deployment on the project URL
  (S5-33); the custom domain is S3-29, a one-line `base_path` flip; no S5 item publishes to PyPI or npm.
- **Risks.** The slug-reuse row and the Pages-URL row were replaced; added: hub identity/naming drift (reserved name),
  the orchestrator name collision, and the release version race (S3-12).
- **Section 6.** Consumes rows `S3:molxmpnn-repo` and `S3:aminx-name-freed` replaced by `S3:aminx-identity`,
  `S3:release-guard`, `S3:stale-name-guard` and `S3:hub-identity`; `S5:hub-config` records S3-29 as a consumer.
- **Q1** re-pointed to S3-01 (S3 Q16-Q20); Q11(c) no longer names the slug.
- **Ordering restated for the no-rename world** (section 6, "Ordering records"). "Rename first vs hub first" and the
  not-ordered-against-S3-20 bullets are void (no rename, no shim removal item). "EBM first vs hub first": the hub's
  first deployment is independent of S2 in either order (the hub imports no model package, no EBM manifest in v1, and
  nothing outside `src/aminx/ebm/` imports it); an S2 release carrying the EBM removal is ordered against S5-52 and
  S1-25 by S3-28's release-order check. Release order kept: S3-12 first in the train, S5-52 (next unused alpha, 0.2.0a4
  on 261001) before S1-25 (0.2.0a5), S5-33 off the S1 flip chain.

Edges rewired (S3 map from the rewritten S3):
- **S5-01** gains `S3-01` (the slug is known before the repo exists; S3-28 C6) and `S3-24` (the vendored reserved-name
  checker it runs); `depends_on = ["S3-01", "S3-24"]`. Consequence: every hub item, and the S6 items that wait on S5-01
  and S5-02, now wait transitively on the user's S3-01 answer.
- **S5-14**: `S3-07` dropped (rename ordering); keeps `S3-12`, `S4-19`, `S4-14`. Gate says the dry run is S3-12's.
- **S5-52**: `S3-13` dropped (S3-13 replaced by `S3-12`, already an edge); gate's version rule is now "next unused
  alpha read from PyPI at cut time" (0.2.0a4 on 261001); the S3-25 mention is gone. S1-25's own `depends_on` (S5-52,
  S3-12) is edited in S1.
- **S5-21**: `S3-07` dropped; `depends_on = ["S5-19", "S4-19"]`. **S5-61**: `S3-07` dropped; `["S4-19"]`.
- **S5-34**: `S3-07` dropped; keeps `S3-21` (docs URL record). Gate no longer talks about an old-URL stub.
- **S5-33**: keeps `S3-01` only; the gate text about "S3-02, S3-12, S3-13 arrive transitively" now reads S3-12 through
  S5-53 and S5-52, and S3-24 through S5-01. Title drops "(P1)".
- No edge to `S3-02`, `S3-03`, `S3-04`, `S3-05`, `S3-06`, `S3-08`, `S3-10`, `S3-11`, `S3-14`..`S3-17`, `S3-19`, `S3-20`,
  `S3-22`, `S3-23`, `S3-25`..`S3-27` remains in any item field. S3-09, S3-18, S3-28 and S3-29 have no S5 edge (S3-29
  depends on S5-33 from S3's side).

Retired item (id out of the toml, never reused): **S5-55** ("P2 only: hub takes the aminx name"). P2 contradicts
revised D2; the useful residue, flipping `base_path` to `/` when a custom domain is attached, is S3-29. No other spec
had an edge to S5-55.

Checked mechanically after the edit (stdlib `tomllib`, throwaway script): the single toml block parses; 59 S5 items;
ids unique across all six specs; every S5 `depends_on` resolves; no retired S3 id or `molxmpnn` string in any S5
`title`, `gate` or `repo`; no cycle in the union graph; no other spec depends on a removed S5 id.

Not changed: S5's design (catalog, executors, privacy, resume, hosting tiers, driver trust boundary), items other than
those named above, and the history in the earlier entries. Other specs still carry their own stale edges to retired S3
ids until their authors apply S3 section 2.1.

### No-rename coherence r1 (261001, CH1-02)

- **CH1-02.** aminx CI has an AST inventory test (`tests/parity/test_browser_validation_inventory.py`) that fails on unmapped and stale public symbols in the eight root packages and on every `aminx.run.__all__` name, with `tests/parity/browser_validation_paths.json` as the mapping. S5 did not know it. Changed: new ledger row A42 (VERIFIED by read); the S5-21 and S5-61 gates now require the test green and any new public def mapped in the JSON in the same PR; one risk row added. The JSON is shared with S2-18, S1-20, S1-07, S1-23 and S4-21/S4-40, so edits are unordered textual merges; the cross-spec ordering fix belongs to those specs. No `depends_on` changed.
