---
title: "Ecosystem hub: shared recon brief for the 261001 spec set"
description: "Verified facts, user decisions and open questions shared by the six 261001 ecosystem specs (spec unification, EBM extraction, molxmpnn rename, xtrax contract, aminx hub, pipeline editor)."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
---

# Ecosystem hub: shared recon brief (261001)

Shared input to the six specs in this set. **Every fact below was checked by reading code or
git refs on 2026-10-01** unless tagged *unverified*. Spec authors should still re-read any
file:line they make load-bearing (anchors can drift); do not re-derive what is here.

## User decisions (2026-10-01)

- **D1.** The *user-facing* one-stop shop is a hub (static site + catalog + executors + editor).
  Model code stays in separate repos; it is not absorbed into one monorepo.
- **D2 (REVISED 2026-10-01, supersedes the earlier rename decision).** The MPNN package **keeps the
  name `aminx`** — it is released on PyPI as `aminx`, so there is **no rename** (no `molxmpnn`, no
  `mpnnx`, no shim distribution, no repo/HF/PyPI cutover). The hub is a **separate** project holding
  the combined implementations; working slug **`aminx-hub`**. Alternative the user raised: name it
  after the domain **praxia.science** the user owns. Note a GitHub Pages custom domain works with any
  repo slug, so domain and repo name are independent choices; also note the name collision with the
  user's existing praxia agent orchestrator. Final hub name/domain is a user decision.
- **D3.** ProteinEBM (`src/aminx/ebm/`) moves out of aminx into **its own project** (name TBD;
  molxmpnn is NOT the EBM name). Filed as praxia tech debt **#2369**.
- **D4.** aminx's two spec systems must become **one**, regardless of anything else.
- **D5.** AF3 **can** run in the browser — see localfold (below). The earlier claim that AF3
  must be a remote node was wrong.
- **D6.** proteinsmc must not take a runtime dependency on aminx/molxmpnn (ecosystem partition).
- **D7.** Spec everything for a backlog DAG. **Nothing executes yet.**
- xtrax pin is already bumped to 0.4.0a11 in draft PR maraxen/aminx#174 (not merged).

## Verified facts

### aminx (this repo, origin/main @ d1210e4a, v0.2.0a3)
- ~47k LOC in `src/aminx`. Sizes (lines): ebm 6.6k, host 8.9k, inference 5.0k, model 4.5k,
  utils 3.5k, potts 3.0k, run 2.1k, training 2.1k, sampling 1.8k, types 1.7k, io 1.6k,
  parity 1.3k, tiling 1.2k, export 0.7k, scoring 0.4k.
- **Two spec systems coexist:**
  - `src/aminx/run/spec.py:138` `class RunSpec(_XtraxRunSpec)` (eqx.Module; subclasses
    `xtrax.run.RunSpec`) with sub-configs `io, resource, multistate, ligand, grid, precision,
    plan, sampling` + `encoding_fusion` / `decoding_fusion`. Module docstring: "`run.specs.
    RunSpecification` remains the public dataclass façade for one minor version."
  - `src/aminx/run/specs.py:190` `class RunSpecification` with subclasses `ScoringSpecification`
    (:527), `SamplingSpecification` (:582), `JacobianSpecification` (:718),
    `InspectionSpecification` (:749); `src/aminx/training/specs.py:18`
    `TrainingSpecification(RunSpecification)`. Each façade subclass calls
    `self._sync_run_spec()` (`specs.py:329`, called at :522/:577/:713/:741) to mirror into RunSpec.
  - Blast radius: **95 files** across src/tests/scripts reference a `*Specification` class;
    **25 files** in src/tests reference `RunSpec` / `aminx.run.spec`.
  - Also `src/aminx/potts/spec.py:36` `PottsRunSpec` and `src/aminx/potts/_trw_spec.py:14`
    `PottsTRWRunSpec` — separate from both systems.
  - `PottsTRWRunSpec`: the first 40 lines of `aminx/src/aminx/potts/_trw_spec.py` and
    `mistypotts/src/mistypotts/potts_trw_spec.py` are **byte-identical** (copy-paste fork).
  - Memory note (project_epic1541): ~27 dead fields in the RS sub-config scaffolding and a stale
    migration map remain open. *Unverified count — re-measure.*
- `src/aminx/registry.py`: `SAMPLERS`, `OUTPUT_SINKS` registries, `_COMBINE_INDEX`.
- **EBM coupling (measured):** nothing outside `src/aminx/ebm/` imports `aminx.ebm`. `ebm`
  imports from the rest of aminx only: `aminx.utils.safe_scan` (2), `aminx.utils.aa_convert` (2),
  `aminx.utils.safe_map` (1), `aminx.io.parsing` (1), `aminx.model.diffusion_mpnn` (1 —
  `SwiGLU`, `ebm/trunk.py:113`, reused verbatim, also referenced by `ebm/checkpoint.py:68,178`).
  EBM modules: checkpoint, conformational_biasing, contracts, ddg_stability, decoy_ranking,
  diffusion, dispatch, langevin, langevin_schedule, model, plan, readout,
  structure_prediction, training, trunk. EBM uses orbax (`ebm/training.py`, `ebm/model.py`).
  Root dirs `ebm_benchmarks/`, `ebm_benchmarks_h200/` exist.
- **Export:** `src/aminx/export/{wrappers,buckets,rng_audit}.py`. `buckets.py:15,74` delegates
  selection to `xtrax.tiling.select_bucket`; aminx keeps only the pinned
  `EXPORT_BUCKETS = (128, 256, 512, 1024)` and an MPNN-specific `k_neighbors` guard.
  The ONNX route is in `scripts/browser_validation/*.py`, importing `jax2onnx` /
  `onnxruntime` directly; **jax2onnx is not in uv.lock** (ad-hoc install).
  `scripts/browser_validation/layer_b_iree.py:802` uses `xtrax.export` for IREE safety only.
- **Browser/site:** `site/` (vanilla ES modules, no bundler; `app.js` loads py2Dmol from CDN with
  graceful fallback), `browser/aminx-sampler/` (`aminx_sampler.mjs`, `runspec_core.mjs` — a
  hand-written JS mirror of RunSpec), `browser/layer_c`, `browser/smoke` (Playwright, ORT-Web
  1.30.0 via importmap). Built by `tools/build_site.py`; deployed by
  `.github/workflows/pages.yml` (manual `workflow_dispatch` only). *(agent-reported, spot-check)*
- Published identity that a rename touches: PyPI `aminx`, the `aminx` CLI (typer,
  `src/aminx/cli.py`), HF weight repo + `HF_REVISION` (memory: PR #168, revision aa80d0fd),
  the `using-aminx` skill, site URLs.

### xtrax (~/projects/xtrax)
- **Local checkout is on branch `skills/activation-parity-260929` and local `main` is stale
  (35c5100). Read `origin/main` (6ccc913) or tag `v0.4.0a11`.**
- `xtrax.run.RunSpec(eqx.Module)` is domain-agnostic: `seed`, `axes: list[AxisSpec]`,
  `carry_specs`, `boundaries`, `run_id` (static). Also `StageBundle`, `InputResolver`,
  `SinkSpec`. *(agent-reported on the skills branch; re-check on v0.4.0a11)*
- Graph IR: `xtrax.composition.graph.HostPrepGraph` (nodes + edges, pre-JIT DAG; node
  `callable_ref` serialised as `module.path:symbol`), CLI verbs `graph-validate`,
  `graph-plan`, `graph-author`. *(agent-reported; re-check)* — note `callable_ref` names a
  **Python** callable, which a browser executor cannot bind.
- **v0.4.0a11 has `xtrax.export.onnx`**: `convert_to_onnx`, `run_onnx`, `verify_onnx_parity`,
  `find_onnx_rng_ops`, `onnx_dtype_census`, plus a JAX-namespace-restoring guard (handles
  jax2onnx patching `jnp` at import). Also `export/{compile,spirv,targets,pipeline,composer,
  parity,divergence,rings,safety,hf_weights}.py`. `v0.4.0a10` has **no** `onnx.py`.
- xtrax extras (a11): `export = [xtrax[export-runtime], iree-base-compiler>=3.11,<4]`;
  `onnx = [jax2onnx>=0.17.0,<0.18, onnx>=1.23,<2, onnxruntime>=1.30,<2]`.
- `xtrax.tiling` exports `Bucket`, `select_bucket`, `bucketize`, `SafeMap`, `Scan`, `Vmap`.
- **Dependency conflict (measured):** adding `xtrax[onnx]` to aminx pulls jax2onnx 0.17 →
  orbax-export 0.0.8 → caps orbax-checkpoint; through the universal uv lock this downgrades
  the **base** install's orbax-checkpoint 0.12.0 → 0.11.36. Without the extra it resolves to
  0.12.6. Unresolved; PR #174 deliberately omits the extra.

### Sibling repos (agent-reported, ~/projects; spot-check before relying on)
- **None import aminx.** xtrax users: demistify, plegadx, prolix, denxity, colliculix.
  proxide users: plegadx, prolix, mistypotts.
- plegadx (~49k LOC): Equinox AlphaFold3, StableHLO export (`src/plegadx/export.py`), deps
  torch + xtrax + proxide.
- prolix (~34k LOC): JAX MD/physics, StableHLO export with IREE WASM/WebGPU flags
  (`src/prolix/export.py`), deps jax-md, openmmtools, rdkit, parmed.
- demistify (~16k LOC): von Mises mixtures + MIST over MD trajectories; deps torch, mdtraj,
  xtrax.
- mistypotts (~6.5k): Potts/TRW; has `PottsTRWRunSpec` (forked copy, above).
- isochore (~2k): numpy-only grid free energy maps (Pyodide candidate).
- proteinsmc (~8.6k): SMC sequence design over fitness oracles (ESM / ProteinMPNN); flax +
  blackjax + alphex. **Verified:** `proteinsmc/pyproject.toml:20-25` deliberately does not
  depend on aminx because "that would create a proteinsmc -> aminx dependency edge, which the
  ecosystem partition exists to flip"; `scoring/mpnn.py` degrades when the MPNN package is absent.
- alphex (numpy-only alphabets), proxide (Rust/maturin protein I/O), denxity, colliculix.
- **Ecosystem partition (verified):** `alphex/.praxia/docs/specs/260814_alphabet-contract.md`
  D2/D4 — the partition's concern is "heavyweight inbound coupling (jax / proxide / Rust /
  maturin)"; aminx and proteinsmc are "mid-partition and should not take a new edge while that
  is in flight"; dev-only test dependencies do not count as coupling edges.

### Browser-side neighbours
- **localfold** (`~/repos/localfold`, sokrypton, npm `localfold`, localfold.org): AF2, AF3,
  ESMFold2, ESM-C as **hand-written WebGPU kernels in JS** (`src/kernels`, `src/af2`, `src/af3`,
  `src/esmfold2`, `src/esmc`), usable as a library ("using the kernels as a library",
  `docs/RUNNING.md`). Also has a **WebGPU ProteinMPNN**: `src/design/{mpnn/, mpnn-bridge.js,
  sample-sequence.js, designers.js, hunter-loop.js}`. Not a JAX export. Needs WebGPU.
  ⇒ a second browser MPNN implementation exists alongside aminx's ONNX one.
- **py2Dmol**: source of truth `~/repos/py2Dmol` branch `main` (67cbe73) has **no** plugin
  registry. The plugin system (`window.py2dmolPlugins`, `PLUGIN_API_VERSION = 1`, `draw(ctx, r)
  -> Primitive[]`) exists only on `~/repos/py2Dmol-plugin-spec` branch `plugin-system-impl`
  (5 commits ahead of its main, latest acb1cfc "Add the volume plugin..."). Plugins are
  render-side (emit primitives), not inference backends. Read py2Dmol's own CLAUDE.md.
- **praxis** (`~/projects/praxis`): lab-automation platform (PyLabRobot), FastAPI + Pyodide
  browser mode, **Angular 21** front end with Formly forms; no node/graph editor exists.
  praxia (Rust agent orchestrator) has no graph UI either. *(agent-reported)*
- *Unverified:* JAX cannot run under Pyodide (no official jaxlib wasm wheel) — check before a
  spec depends on it.

## Spec set (task_id 261001_aminx-hub-ecosystem-specs)

| ID | Spec file (`.praxia/docs/specs/`) | Owner repo(s) |
|---|---|---|
| S1 | `261001_spec-system-unification.md` | aminx; mistypotts for the TRW dedupe |
| S2 | `261001_ebm-extraction.md` | aminx → new EBM repo |
| S3 | `261001_molxmpnn-rename.md` → retitled "aminx identity kept: hub naming + stale-name cleanup" (rename retired per revised D2) | aminx, hub |
| S4 | `261001_xtrax-model-contract.md` | xtrax (+ adopters) |
| S5 | `261001_aminx-hub.md` | new hub repo (working slug aminx-hub; site, catalog, executors) |
| S6 | `261001_pipeline-editor.md` | shared editor component (hub + praxis) |

Backlog item ids are `S<n>-<two digits>` (e.g. `S4-03`) and may depend on other specs' ids.
