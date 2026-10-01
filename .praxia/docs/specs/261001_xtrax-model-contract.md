---
title: "xtrax model contract: ports, manifests, schemas, executor-agnostic graph IR, shared ONNX route"
description: "L1 contract (port types, model manifest, param JSON Schema, graph IR v2 with per-executor bindings, scorer protocols, shared ONNX export route) that lets models compose and be served by the aminx hub without importing each other."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - xtrax
  - aminx (becomes molxmpnn after S3)
  - proteinsmc
  - plegadx
  - prolix
related_specs: [S1, S2, S3, S5, S6]
---

# S4: xtrax model contract

Spec only. Nothing here has run, nothing is implemented, and no number in this document is a
measurement unless it carries a read anchor. Items that will produce a number are
pre-registered (section 7). User decisions D1-D7 from
`.praxia/docs/research/261001_ecosystem-hub-recon-brief.md` are taken as given. This spec
defines what D1 ("model code stays in separate repos") needs in order to be true: a small
contract that sits *between* the repos.

## 1. Goal and non-goals

**Goal.** Define the L1 contract that lets a model be described once and then (a) composed
with other models in a typed graph, (b) rendered as a form, and (c) executed by whichever
executor can run it (Python, ONNX Runtime, an npm kernel library, Pyodide, a remote
endpoint), without the models, the hub, or a consumer such as proteinsmc importing each
other. Concretely: ten deliverables (a)-(j) from the scope, mapped to sections 4.2-4.11.

**Acceptance criteria** (each is gated in section 7):

- AC1. A `ModelManifest` for each MPNN node type validates against a generated JSON Schema,
  round-trips through the Python dataclasses, carries every field the hub catalog asked for
  (HUB-REQ 1-10, section 3.5), and every `module:symbol` it names resolves.
- AC2. Graph IR v1 documents still load; v2 documents round-trip; unknown schema versions are
  rejected without default-fill (the existing PM3 discipline).
- AC3. The conformance corpus and the RFC 8785 hash vectors exist and are owned by S4 (S4-30,
  S4-31); the Python evaluator passes the corpus, a planted port-type-table mutation is caught,
  and canonical-JSON hashes match the shared vectors where naive `json.dumps` does not.
  Agreement of the TypeScript evaluator on the same corpus is gated by S6-04, not by an S4 item.
- AC4. The portable MPNN knob document has one Python source (a projection of S1's unified
  RunSpec), its pure lowering has golden vectors, and its schema, TS types, constants, defaults
  and validator are generated from it (not hand-written); a planted wrong default is caught.
- AC5. proteinsmc obtains an MPNN scorer by discovery with no import edge to
  aminx/molxmpnn and no new runtime dependency (D6).
- AC6. The four-graph ONNX split exports through `xtrax.export.onnx` and passes the already
  established parity chain under bounds inherited from earlier runs, under pre-registered
  bathos sidecars.
- AC7. After the dependency decision (Q3), the MPNN repo's *base* lock resolves
  `orbax-checkpoint >= 0.12`, and the record of the resolution is kept.
- AC8. The plegadx / prolix export audit is recorded (section 3.4); plegadx's shape contract is
  importable as `PortSpec`s (S4-24, code, not a decision record); the prolix adoption decision is
  made on the S4-25 measurement (S4-26, a user decision).

**Non-goals.**

- Executor implementations, catalog, site: S5. Editor UI: S6. Rename: S3. Spec-system
  unification and the field registry itself: S1 (S1 declares the nested-field metadata; S4 owns
  the browser knob document built on it, S1 decision D10). EBM extraction: S2. This spec defines
  the *shapes* those consume.
- A general workflow engine. The graph is a DAG of typed operations; no loops, no control flow.
- Making JAX run under Pyodide, or any WebGPU claim. Executors record what was verified and at
  what level; they do not promise more (section 4.5).
- Porting the browser host-prep numerics (PRNG, shuffles, bias building) into ONNX. They stay
  hand-written JS, gated by cross-language golden vectors from the Python lowering (S4-41,
  section 4.7).
- Hosting weights. The manifest *points at* a pinned weights revision.
- Changing `xtrax.run.RunSpec` (`seed, axes, carry_specs, boundaries, run_id`, `from_spec`).
  S1 asked that these stay backward compatible; this spec does not touch them and S4-10 pins
  them with a regression test.

## 2. How to read the tags

Every load-bearing assumption is in the ledger in section 3.6 as VERIFIED (a `read:` anchor to
the line that does the thing), REFUTED (with the design change), or UNVERIFIED (with
`deferred:`). The spike harness named in the specialist process
(`scripts/loop/adversarial_metrics.py`, `.praxia/spikes/`) does **not exist in this worktree**
(`ls` returned no such path), and this task forbids running tests, builds, resolvers or heavy
commands. So no spike was run. Rows that only a run can settle are UNVERIFIED and each is
pre-registered as a spike or a gate in section 9 rather than silently trusted. xtrax anchors are
`xtrax@v0.4.0a11:<path>:<line>` read through `git show` (the local xtrax checkout is stale).
Agent-reported facts in the brief were re-read for every row the design leans on.

The sibling specs (S1, S3, S5, S6) were being written and revised while this spec was written.
Their asks of S4 are recorded in section 3.5 with anchors into their drafts as of 261001 round 1
(line numbers may drift). **S5 and S1 already cite S4 item ids** (S4-04, S4-09, S4-10, S4-14,
S4-18, S4-19, S4-20), so those ids and their meaning are frozen; later additions are appended as
S4-30 and up rather than renumbering (ledger A41). Round 1 appended S4-34 to S4-42 and re-scoped
S4-33 (see the Revision log); S4-18 keeps its meaning as the ORT-CPU cell of the parity chain.
Round 2 appended S4-43 to S4-49 and narrowed S4-10 (it keeps its id and its S1-facing meaning: the
contract-side graph wire format plus the `xtrax.run.RunSpec` pin; the live-graph, emitter and
audit work moved to S4-44 and S4-45).

## 3. Current state

### 3.1 xtrax at v0.4.0a11

- **Graph IR.** `HostPrepGraphNode` carries `id`, `callable_ref` (a live Python callable),
  `metadata`, `frozen` (`composition/graph.py:42`). `GraphEdge` is `src`, `dst` only, no ports
  (`graph.py:59-63`). `serialize_node` emits the callable as a `module:symbol` string
  (`composition/serialize.py:108-112`). `GRAPH_SCHEMA_VERSION = 1` and `deserialize_graph`
  *rejects* any newer version (`serialize.py:42`, `:175`). Nothing under the sibling repos
  other than xtrax imports `xtrax.composition` or `HostPrepGraph` (rg over aminx, plegadx,
  prolix, demistify, denxity, colliculix, mistypotts, mpnn_ext, asr, tev_design, proteinsmc,
  proxide, isochore, maraxiom, naurmalade returned nothing), so v2's blast radius is
  xtrax-internal: 17 source files at a11, namely `cli/{graph_plan_verb,graph_verb,loader,plan}.py`,
  `composition/{author,validate,node_metadata,graph,serialize,errors,__init__}.py`,
  `inference/ir_schema.py`, `loop/{admission,diversity_quota,__init__}.py`,
  `run/component_binding.py`, `stages/evaluate.py`.
- **A schema emitter already exists.** `inference/ir_schema.py:223` `emit_ir_schema()` emits a
  JSON Schema 2020-12 document (`$id xtrax://composition-ir`) derived from live types. Its
  type mapper (`:93-144`) handles bool/int/float/str, enums, `Optional`, `list/tuple/set`
  (all as homogeneous arrays, fixed-arity tuples are not distinguished), `dict`, and
  `Callable` (as an import-path string). It **raises** `IRSchemaTypeError` for everything else:
  no `Literal`, no `Mapping`/`Sequence` (origin is not `dict`/`list`), no `Path`, no nested
  dataclass `$ref`. It imports `jax.ShapeDtypeStruct` at module level (`:43`).
- **Prior decision on schema ownership (fork 13).** The 260702 design spec accepted
  "inference-layer-owned, emitted from E1 types, one schema two consumers" and rejected a
  *standalone shared schema package* "precisely because that is what creates two competing
  graph formats" (`xtrax@v0.4.0a11:.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md:54`).
  This spec has to honour the reason, not just the letter (section 4.1).
- **Export.** `xtrax.export.onnx` (new in a11): `convert_to_onnx(callable, abstract_inputs,
  target)` flattens pytrees to leaves, so graph inputs are the leaves in order
  (`export/onnx.py:147`); it refuses `jax_enable_x64` (`:310`), restores the JAX namespaces
  jax2onnx patches (`:324`), refuses graphs containing ONNX RNG ops (`:330`), and serialises
  the in-memory model (`:347`). Opset is pinned at 23 (`:68`). `export_pipeline` requires a
  `BatchPlan` (`export/pipeline.py:175-177`, `:284`), so a caller that has no plan (aminx's
  split graphs) can only use `convert_to_onnx` directly. The `onnx` target is EXECUTED on ORT
  *CPU* only (`export/targets.py:296`, docstring there); `wasm32` is `CODEGEN_ONLY` with
  triple `wasm32-unknown-emscripten` and `+simd128,+atomics,+bulk-memory` (`targets.py:217-226`).
- **Weights.** `load_hf_weights` calls `hf_hub_download(repo_id=, filename=)` with no
  `revision` (`export/hf_weights.py:109`), so it cannot pin what a manifest would pin.
- **Entry points.** xtrax *deferred* a plugin entry-point hook for CLI verbs and wrote down the
  rule that any `importlib.metadata.entry_points` scan must be lazy and per-entry isolated
  (`cli/registry.py:16-22`). `xtrax/__init__.py` is lazy (`:68`, `:135`), and
  `composition/graph.py` imports only stdlib plus `composition` siblings (`:20-25`).
- **Packaging.** `requires-python = ">=3.13"` and hard dependencies on jax, jaxlib, equinox,
  optax, orbax-checkpoint, numpy (`pyproject.toml:6-7`). The `onnx` extra is
  `jax2onnx>=0.17.0,<0.18, onnx, onnxruntime` (`:92-96`). xtrax carries
  `[tool.uv] override-dependencies = ["orbax-checkpoint>=0.11.17"]` (`:149`, lock `uv.lock:14`)
  and says in a comment that this lifts the cap "for this repo's lock only"
  (`pyproject.toml:97-101`). jax2onnx 0.17 itself pulls flax, dm-pix, netron, huggingface-hub,
  einops, orbax-export (`uv.lock:1392-1407`).

### 3.2 The MPNN package (aminx, `origin/main` d1210e4a)

- **Two spec systems** (D4; S1 owns the fix). `RunSpec` sub-configs use `eqx.field(static=True)`
  (`src/aminx/run/spec.py:97-131`), several array-carrying fields are annotated `Any`
  (`bias :119`, `fixed_positions :121`, `decoding_order_fn :131`), and the module uses
  `from __future__ import annotations` (`:10`), so annotations are strings that
  `typing.get_type_hints` must resolve. S1's plan replaces hand lists with a field registry
  (section 3.5) whose facts reach S4 as opt-in field metadata on the unified RunSpec.
- **The browser "RunSpec" is not a RunSpec.** `browser/aminx-sampler/runspec_core.mjs` is
  366 lines, of which the contract-shaped part is small: constants (`MPNN_ALPHABET :21`,
  `OMIT_BIAS :22`) and defaults (`temperature` 0.1 at `:265`, `decoding_order` "random" at
  `:314`). The rest is **numerics**: SplitMix64 PRNG, Fisher-Yates and fixed-first shuffles,
  Gumbel noise, tie-group maps, bias-tensor building (`buildP07TypedInputs`, `:249`). The
  letter-keyed knob document it consumes (`bias_AA`, `omit_AA`, `tied_positions`,
  `chains_to_design`, `fixed_positions`, `decoding_order`, `seed`, `temperature`) has **no
  Python class**: `rg bias_AA` finds it only in `browser/`, `scripts/browser_validation/`
  and `tests/parity/test_p07_knobs_gate.py`. Python has a second, script-only
  re-implementation of the same compile step (`scripts/browser_validation/p07_knobs_gate.py:297`
  `build_p07_inputs`) that exists to cross-check the JS. So "make `runspec_core.mjs` generated"
  is only partly achievable; see ledger A14 (REFUTED) and section 4.7.
- **JS binds ONNX inputs by hand-pinned position.** `split_driver.mjs:22-65` pins per-graph
  input order, output names and dtypes as constants, and feeds `session.inputNames[i]`
  (`:102`). That pinned table is exactly what manifest port lists plus recorded graph names replace.
- **Production export route.** `scripts/browser_validation/p07_split_export.py` converts the
  four graphs E/W/D/F with `jax2onnx.to_onnx(..., return_mode="file")` then
  `embed_external_data` (`:233-242`; the helper is `p07_knobs_gate.py:1071`, a workaround for
  Loop-subgraph constants that jax2onnx writes as external data, which ORT-Web cannot load).
  It writes a sha256 manifest per graph with `file, bytes, sha256, input_shapes, input_dtypes`
  (`:317-332`), buckets `(128, 256)` (`:54`). Nine scripts reference `to_onnx`
  (`p07_split_{export,feasibility,feasibility_wave,feasibility_fuse}`, `p07_knobs_gate`,
  `layer_b_{build,ort_calibrate}`, `layer_c_calibrate`, `jax2onnx_spike`). aminx keeps its own
  ONNX RNG walker (`scripts/browser_validation/onnx_audit.py:62`) duplicating
  `xtrax.export.onnx.find_onnx_rng_ops`.
- **Pins.** `xtrax[io,export]==0.4.0a10` is a *base* dependency (`pyproject.toml:26`), so the
  IREE compiler already rides every install; the lock resolves `orbax-checkpoint 0.12.0`
  (`uv.lock:2481-2482`). `HF_REVISION = "aa80d0fd..."` is env-overridable
  (`src/aminx/io/weights.py:34`, `:202`). `alphex` is dev-group only by decision D4
  (`pyproject.toml:269-273`).
- **Validated parity chain** (run ids from memory note `project_browser-split-export-validated`,
  *to be re-verified by record*, not cited as evidence here): reference-vs-split teacher-forced
  bound 1e-4 nats, split-vs-monolith bound 2e-4 nats with tokens exact
  (`scripts/browser_validation/p07_split_parity.bth.toml:7` states the 2e-4 bound and that it
  is inherited, not invented).

### 3.3 Siblings

- **proteinsmc** (`requires-python >=3.11`, no xtrax dependency): the consumer-side function
  type is `FitnessFn = Callable[[PRNGKeyArray | None, EvoSequence, PyTree | Array | None],
  Float]` (`src/proteinsmc/models/fitness.py:17`). Its MPNN scorer guards on
  `find_spec("prxteinmpnn")` (`scoring/mpnn.py:14`), a name that no longer resolves, so the
  scorer is **dead code today** and the guard silently reports "unavailable". The pyproject
  documents the partition reason for not depending on aminx (`pyproject.toml:20-25`).
- **alphex** (numpy-only, `requires-python >=3.11`, `pyproject.toml:14,41-42`): known alphabet
  names include `MPNN_X_21` (`src/alphex/known.py:50`). Its own spec **cut** the entry-point
  registry (D5, `.praxia/docs/specs/260814_alphabet-contract.md:308-324`) for two reasons that
  bind this design: resolving `module:attr` imports the plugin's package and so inverts the
  dependency at runtime (`:312`), and entry points need an installed distribution (`:324`).
  Its D4 pattern (dev-only conformance tests, no `src/` import, no coupling edge) is reused in
  section 4.9.
- **localfold** (npm `localfold` 0.0.1, Beerware): published `files` are `src`, README,
  LICENSE only (`package.json:26-30`); `src/index.js` exports AF2 kernels. The AF3 driver
  `foldAf3(options)` lives in `web/af3-model.js:486`, **outside the published package**, and the
  docs say the API is "unstable until a single sequence-in-structure-out entry point exists"
  (`docs/RUNNING.md:82-84`). Its MPNN is a mirror of a different checkout, not aminx
  (`src/design/mpnn/SOURCE.md:1-12`), and its inter-model interface is a PDB string
  (`src/design/mpnn-bridge.js:5-12`), a useful precedent for the Structure port's text encoding.
- **Prior-art check (reuse before inventing).** `rg` for `ModelManifest`, `PortType`,
  `port_type`, `xtrax.models`, `xtrax_contract` over `~/projects` and `~/repos` (excluding
  venvs, `node_modules`, `.git`) found no existing model-manifest or port-type contract; the
  only hits were unrelated third-party files. The nearest precedents are reused, not
  duplicated: plegadx's `export_shape_contract.json` (ports with named axes), aminx's
  per-graph export manifest (`p07_split_export.py:317-332`), xtrax's `BundleSchema` /
  `extract_schema` (`xtrax@v0.4.0a11:src/xtrax/inference/schema.py`, output-leaf names and
  shapes from `jax.eval_shape`, which the provider conformance kit can use to check a manifest's
  declared output ports against the real callable), and alphex's alphabet names.

### 3.4 Export audit: plegadx and prolix against `xtrax.export.{compile,targets}` (scope item i)

Neither repo imports `xtrax.export` (rg for `xtrax.export` / `from xtrax import` over both
`src/` trees: no match; `grep -c xtrax` on each `export.py` is 0).

| Aspect | xtrax a11 | prolix | plegadx |
|---|---|---|---|
| Lowering | `jax.export.export(jax.jit(callable))` then IREE or jax2onnx | jax lowering, `Lowered.as_text()` (`export.py:195-204`) | `jax.export.export(jax.jit(fn, static_argnums=...))` (`export.py:141`) |
| Compile | `iree.compiler.tools.compile_str(..., input_type="stablehlo", extra_args=...)` (`compile.py:191`), `--iree-hal-target-backends=<b>` (`:157`) | `iree-compile` subprocess with `--iree-input-type=stablehlo --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu` (`export.py:22-28`, `:190`) | none: writes StableHLO `.mlir` text only (`export.py:141-150`) |
| WASM flags | triple `wasm32-unknown-emscripten`, cpu `generic`, features `+simd128,+atomics,+bulk-memory` (`targets.py:217-226`) | triple `wasm32-unknown-unknown`, cpu `generic`, **no feature flags** (`export.py:26-27`) | n/a |
| Safety / dtype gate | `validate_export_safe` + per-target dtype envelope | none | none |
| Verification | parity vs independent oracle for EXECUTED targets | none; size gate only: `WASM_ARTIFACT_MAX_BYTES = 50 MiB` (`export.py:19`) | none |
| Static args / layout | no `static_argnums` in `export_pipeline` | n/a | `static_argnums`; gather layout closed over (`export_shape_contract.json` `gather_info_policy`) |
| Contract metadata | none | none | `export_shape_contract.json` v1.0.0: per input `name, dtype, shape_spec, variability_axis, static_arg_required`, sha256-linked to a spec |
| xtrax dependency | n/a | `xtrax>=0.4.0a6,<0.5` (`pyproject.toml:32`) | `xtrax[cli]`, path-editable source (`pyproject.toml:22,132`) |

Findings: (1) prolix's WASM flags differ from xtrax's target in triple, feature flags and HAL
flag spelling; which is right for IREE 3.11 is **unmeasured** (S4-25). (2) prolix has a size
gate xtrax lacks; plegadx has a shape contract xtrax lacks. Both are things the contract
wants (a size budget in `limits`, port shapes with named axes). (3) plegadx's
`export_shape_contract.json` is already a port list in all but name, so S4-24 imports it
rather than inventing a second format. (4) Neither repo gets executed verification from xtrax
today; only `compile_for_target(mlir_text, target)` (`compile.py:117`, plan-free) is a drop-in,
`export_pipeline` is not (plan-bound).

### 3.5 What the sibling drafts ask of S4

Anchors are into the drafts as of round 1; they are asks, not verified facts about shipped code.

| From | Ask | Where it lands here |
|---|---|---|
| S5 | Manifest fields HUB-REQ 1-10: `schema_version`; `id, kind (model or tool), family, version, title`; `ports[]` (name, direction, semantic type, dtype, symbolic shape); `artifacts[]` (id, format, urls, sha256, bytes, bucket); `executors[]` (kind, artifact ids, `requires`, `driver {url, sha256, export}`, `determinism {rng_scheme, cross_executor_equivalent}`); `license` (SPDX, weights text, `redistribution allowed or gated or forbidden`, attribution); `provenance`; `validation` (golden vectors, evidence `{level, executor, bathos_id}`); `limits` (`261001_aminx-hub.md:276-292`) | 4.3 |
| S5 | `S4:model-manifest`, `S4:port-types`, `S4:onnx-route`, `S4:graph-ir` tokens; fixtures validating against "S4's committed schemas (S4-09)", native manifest "(S4-19)" (`261001_aminx-hub.md:296-297`, `:624-627`) | token map in section 6 |
| S5 | Driver module is one pre-bundled ES module, no `import` specifiers, sha256 equals `driver.sha256`, `export async function createSession(ctx)` returning `{run(inputs, opts), dispose()}` (`261001_aminx-hub.md:338-350`) | 4.5 (S4 carries only the reference; S5 owns the ABI) |
| S5 | `kind: "tool"` for isochore-like entries (`261001_aminx-hub.md:705`) | 4.3 (`kind` is model or tool) |
| S5 | Three distinct grid port types (`gfe-grid` `{data f32, origin f64[3], spacing scalar}`, `count-grid`, `density-grid`) and `file` inputs for tools such as `read_dx` (`261001_aminx-hub.md:316-321`, `:394-400`) | 4.2 (`energy-map` with alias `gfe-grid`, `count-grid`, `density-grid`, `file`, `json`) |
| S6 | Nodes addressed by `ref` (manifest `id@version`), no `callable_ref`; port-addressed edges with ids and `kind`; inline `builtin:code.python` node with `source`, `ports`, `requirements`; own `schema_version` gate (`261001_pipeline-editor.md:295-300`, `:675`) | 4.6 |
| S6 | Declarative port-type table with a compat relation `exact or widen or reject`, a JSON conformance corpus `{document, expected codes}` and JCS hash vectors, a shared diagnostic-code registry, params JSON Schema subset, port `required` and `cardinality` (`261001_pipeline-editor.md:319-349`, `:674-676`) | 4.2, 4.4, S4-30, S4-31 |
| S1 | A registry (one row per task and flat name, with `portable` and `browser_key`) is the single source; each nested leaf field of the unified RunSpec carries registry-derived metadata that S4's schemagen reads (the "field-metadata hook"); `run_spec_fields.json` is a derived snapshot for non-Python consumers only (`261001_spec-system-unification.md:351`, `:520-522`) | 4.7, S4-07, S4-32 |
| S1 | **S4-20 owns the browser knob document, its defaults and validators, and the pure lowering** of letter-keyed structure-relative constraints (`bias`, `fixed_mask`, `fixed_tokens`, `tie_group_map`, the `chains_to_design` vs `chain_id` mismatch); it depends on S1-05 and S1-07, and on S1-18 to delete the interim drift test; S1 builds no `DesignConstraints` (decision D10, Q6) (`261001_spec-system-unification.md:481-491`, `:531`, `:547`, `:610`) | 4.7, S4-20, S4-33 |
| S1 | Keep xtrax `RunSpec` base fields and identity `from_spec` backward compatible (`261001_spec-system-unification.md:545`) | S4-10 pins them |
| S6 | **Superseded asks.** S6's draft files S6-25 (the `builtin:` scheme and port `required`/`cardinality`, in xtrax) and S6-26 (conformance corpus, JCS vectors, diagnostic codes, in xtrax) and says S4 has none (`261001_pipeline-editor.md:24`, `:83`, `:232`). S4 now provides all of them (`builtin:` in S4-10, `required`/`cardinality` in S4-06, JCS vectors in S4-30, corpus and codes in S4-31). **S4 is the sole owner**; S6-25 and S6-26 are superseded and must be removed by the coherence pass (section 6 lists the re-points) | 4.2, 4.6, S4-06, S4-10, S4-30, S4-31 |
| S3 | Interim proteinsmc guard `find_spec("molxmpnn.scoring.score")` (S3-18); rename and weights cutover items S3-07, S3-11. S3 provides only the root name `molxmpnn` (`S3:molxmpnn-manifest-namespace`) and the repo/release facts (`S3:molxmpnn-repo`); S4 owns the id and entry-point grammar (4.3, 4.8, C12) | 4.3, 4.8, 4.9, S4-01, S4-22 |
| S5 | **Accepted inbound amendment S5-51 (repo xtrax, `user_decision`)**: G1 `kind: tool` manifest may carry no weights artifact; G2 `external`/`byo` artifacts waive `urls` and the 40-hex rule; G3 `pyodide` entry gains `fs_in`/`fs_out` mappings; G4 `determinism` on every executor kind (`261001_aminx-hub.md`, 4.2 gap table). The hub's earlier G5 (`grid.meaning` in `compat`) is **withdrawn**: 4.2 already defines `energy-map` (alias `gfe-grid`), `count-grid` and `density-grid`, and compat rule 2 rejects a `count-grid` wired to an `energy-map`. The hub's earlier G6 (a `catalog_view` snapshot corpus usable from Node) is S4's own deliverable in S4-09, not an amendment. G1-G4 are S5-51's edits to schemas S4 owns. Decision recorded in 4.3 (additive within `model-manifest.v1`, lands before the first release, S4-34 depends on S5-51) | 4.3, S4-09, S4-34 |

### 3.6 Assumption Ledger

Evidence format: `read: <repo>@<ref>:<path>:<line>` or a workspace path; `deferred:` where only
a run can settle it (see section 2 for why no spike ran).

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | HostPrepGraphNode.callable_ref is a live Python callable serialised as a module:symbol string | v2 node shape and v1 upgrade rule change | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/composition/graph.py:42; read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:108-112 |
| A2 | GraphEdge has only src and dst, no port names | typed-port edges would not need a schema change | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/composition/graph.py:59-63; read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:201 |
| A3 | deserialize_graph rejects schema_version greater than GRAPH_SCHEMA_VERSION (1), so v2 needs a bump plus a v1 reader | v2 could ride under v1 | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:42; read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:175 |
| A4 | xtrax already emits a composition-IR JSON Schema from live types and its mapper raises on Literal, Mapping, Path and nested dataclasses | a new emitter would be needed instead of an extension | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/inference/ir_schema.py:93-144; read: xtrax@v0.4.0a11:src/xtrax/inference/ir_schema.py:223 |
| A5 | Fork 13 rejected a standalone shared schema package because it creates two competing graph formats | a standalone package is free to be hand-authored | VERIFIED | read: xtrax@v0.4.0a11:.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md:54 |
| A6 | Importing an xtrax submodule does not import jax (lazy __init__), and composition.graph imports only stdlib plus composition siblings | a jax-free in-xtrax contract subpackage is impossible | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/__init__.py:68; read: xtrax@v0.4.0a11:src/xtrax/__init__.py:135; read: xtrax@v0.4.0a11:src/xtrax/composition/graph.py:20-25 |
| A7 | The xtrax distribution requires Python 3.13 and hard-depends on jax, jaxlib, equinox, optax, orbax-checkpoint | an in-xtrax contract would be installable by 3.11 and non-JAX consumers | VERIFIED | read: xtrax@v0.4.0a11:pyproject.toml:6-7 |
| A8 | proteinsmc and alphex support Python 3.11 and proteinsmc has no xtrax dependency | the Python floor of the contract could be 3.13 | VERIFIED | read: /home/marielle/projects/proteinsmc/pyproject.toml:9; read: /home/marielle/projects/alphex/pyproject.toml:14; read: /home/marielle/projects/proteinsmc/pyproject.toml:12-30 |
| A9 | proteinsmc FitnessFn is (key, sequence, context) to float, and its MPNN scorer guards on the stale name prxteinmpnn | the Scorer protocol shape and the proteinsmc item change | VERIFIED | read: /home/marielle/projects/proteinsmc/src/proteinsmc/models/fitness.py:17; read: /home/marielle/projects/proteinsmc/src/proteinsmc/scoring/mpnn.py:14 |
| A10 | alphex cut its entry-point registry because resolving module:attr imports the plugin package (runtime dependency inversion) and entry points need an installed distribution | entry points could point at module attributes freely | VERIFIED | read: /home/marielle/projects/alphex/.praxia/docs/specs/260814_alphabet-contract.md:312; read: /home/marielle/projects/alphex/.praxia/docs/specs/260814_alphabet-contract.md:324 |
| A11 | importlib.metadata.EntryPoint stores a non-module:attr value without validation and exposes .dist, so a distribution-relative data path can be an entry-point value read through dist.locate_file | the discovery design must fall back to a module-attribute value or a sidecar file scan | VERIFIED | read: /home/marielle/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/lib/python3.13/importlib/metadata/__init__.py:170-171; read: /home/marielle/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/lib/python3.13/importlib/metadata/__init__.py:201-203 |
| A12 | The same EntryPoint behaviour holds on Python 3.11 (the contract floor): the value is stored as given and `.module`/`.attr` are computed only on access | discovery needs a 3.11 shim | VERIFIED | read: /home/marielle/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/lib/python3.11/importlib/metadata/__init__.py:149-221 (r1 C17); the S4-11 gate still runs on 3.11 and 3.13 |
| A13 | convert_to_onnx flattens inputs positionally, refuses RNG ops and x64, and restores jax namespaces, so a manifest must record graph input names read from the ModelProto | binding by name could rely on convert_to_onnx | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/export/onnx.py:147; read: xtrax@v0.4.0a11:src/xtrax/export/onnx.py:310-347 |
| A14 | runspec_core.mjs is a hand-written mirror of the RunSpec schema | n/a | REFUTED | read: browser/aminx-sampler/runspec_core.mjs:249; read: scripts/browser_validation/p07_knobs_gate.py:297; changed: only constants, defaults, validation and types are generated (from the knob document, S4-40, S4-41 and S4-20), the PRNG and shuffle kernels stay hand-written and are gated by golden vectors (section 4.7) |
| A15 | aminx run-spec sub-configs are eqx.Module classes using static fields with some array fields typed Any | the emitter could ignore non-static array fields | VERIFIED | read: src/aminx/run/spec.py:97-131; read: src/aminx/run/spec.py:10 |
| A16 | eqx.field accepts a metadata kwarg (reserved keys static and converter) so UI hints can ride on fields of classes that are not in the S1 registry | hints need a side table keyed by field name | VERIFIED | read: /home/marielle/projects/aminx/.venv/lib/python3.13/site-packages/equinox/_module/_field.py:57-83 |
| A17 | eqx.Module classes are dataclasses, so dataclasses.fields works for a stdlib-only mapper | the mapper needs equinox-aware introspection | VERIFIED | read: /home/marielle/projects/aminx/.venv/lib/python3.13/site-packages/equinox/_module/_module.py:339-340 |
| A18 | typing.get_type_hints resolves every annotation of the real MPNN spec classes | emitter needs a fallback for TYPE_CHECKING-only names | UNVERIFIED | deferred: needs importing the aminx package (jax), forbidden locally, pre-registered as S4-02; S4-02 characterises the pre-S1-07 classes only, so S4-32 re-runs the same script on the unified classes (r2 R2-8) |
| A19 | safetensors accepts a str-to-str metadata map on save, so a port descriptor can ride in the file header | descriptor travels in a sidecar file | VERIFIED | read: /home/marielle/projects/aminx/.venv/lib/python3.13/site-packages/safetensors/__init__.pyi:37; read: /home/marielle/projects/aminx/.venv/lib/python3.13/site-packages/safetensors/numpy.py:29 |
| A20 | The safetensors byte layout (u64 little-endian header length, JSON header with __metadata__ and data_offsets) is what a 30-line JS reader needs | the browser reader needs another container | UNVERIFIED | deferred: the layout is in a compiled extension with no readable doc here, S4-04 gate cross-checks against the safetensors reference writer |
| A21 | Production split export uses jax2onnx directly with return_mode file plus embed_external_data, and records a per-graph sha256 manifest | the adoption item is a smaller swap | VERIFIED | read: scripts/browser_validation/p07_split_export.py:233-242; read: scripts/browser_validation/p07_knobs_gate.py:1071; read: scripts/browser_validation/p07_split_export.py:317-332 |
| A22 | The in-memory ModelProto from xtrax convert_to_onnx contains no external-data tensors for these graphs | the bundle exporter needs an embed step like embed_external_data | UNVERIFIED | deferred: needs a jax2onnx run, forbidden locally; pre-registered spike S4-43 settles it on the four real graphs before S4-14 and S4-17 depend on it, with conditional fix item S4-46 (r2 R2-5) |
| A23 | JS binds each ONNX input by hand-pinned position and dtype | a manifest cannot replace the pinned table | VERIFIED | read: browser/aminx-sampler/split_driver.mjs:22-65; read: browser/aminx-sampler/split_driver.mjs:102 |
| A24 | The split-vs-monolith bound 2e-4 nats and token-exact agreement are inherited bounds, not invented | adoption gates need freshly derived bounds | VERIFIED | read: scripts/browser_validation/p07_split_parity.bth.toml:7 |
| A25 | An xtrax-route artifact will meet the inherited bounds | the xtrax route regresses and the adoption item blocks | UNVERIFIED | deferred: requires conversion plus ORT runs on titanix, pre-registered as S4-17 and S4-18 |
| A26 | xtrax's [tool.uv] override-dependencies protects a consuming repo's lock | the consumer must carry its own override or an isolated env | REFUTED | read: xtrax@v0.4.0a11:pyproject.toml:97-101; changed: option A (consumer-side override restating the core floor) or D (isolated export env) is required, never inheritance |
| A27 | A consumer-side override of orbax-checkpoint keeps the base lock at 0.12.x and the ONNX scripts working | option A fails and option D or B is needed | UNVERIFIED | deferred: needs a resolver run plus ORT tests, forbidden locally, pre-registered as S4-03 |
| A28 | Adding xtrax[onnx] to the base install of the MPNN package pulls flax, dm-pix, netron and orbax-export | the extra could stay in base | VERIFIED | read: xtrax@v0.4.0a11:uv.lock:1392-1407 |
| A29 | localfold's published npm package exposes an AF3 sequence-to-structure entry point | an npm binding for AF3 can be written now | REFUTED | read: /home/marielle/repos/localfold/package.json:26-30; read: /home/marielle/repos/localfold/web/af3-model.js:486; read: /home/marielle/repos/localfold/docs/RUNNING.md:82-84; changed: the AF3 npm executor is gated behind S5's localfold items (upstream request S5-25, vendored job-json S5-23, adapter S5-26; Q6) and the contract only reserves the shape; S4-27 retired (r3 C13) |
| A30 | prolix WASM flags and xtrax WASM32 produce equivalent artifacts | adoption is a flag swap | UNVERIFIED | deferred: needs iree-compile runs, forbidden locally, pre-registered as S4-25 |
| A31 | plegadx export_shape_contract.json carries name, dtype, shape_spec, variability_axis and static_arg_required per input | the port importer needs a different source | VERIFIED | read: /home/marielle/projects/plegadx/src/plegadx/export_shape_contract.json:1-30 |
| A32 | Neither prolix nor plegadx imports xtrax.export | adoption has no existing call sites to preserve | VERIFIED | read: /home/marielle/projects/prolix/src/prolix/export.py:9-16; read: /home/marielle/projects/plegadx/src/plegadx/export.py:40-61 |
| A33 | The MPNN score function returns a negative log-likelihood (`nll`, lower is better), reduced as a masked mean | none, the contract declares direction and units explicitly rather than assuming | VERIFIED | read: src/aminx/scoring/score.py:209; read: src/aminx/scoring/score.py:86 (r1 C17); S4-21 keeps a sign-flip negative control |
| A34 | JAX cannot run under Pyodide | none, no item depends on it, the pyodide executor entry is shape-only | UNVERIFIED | deferred: needs a Pyodide session and network, not a design input |
| A35 | The aminx dependency-groups table exists, so an export-only group is expressible without a new extra | the toolchain would have to ride in an extra | VERIFIED | read: pyproject.toml:264 |
| A36 | importlib.util.find_spec of a dotted name imports the parent package, so a guard like find_spec("molxmpnn.scoring.score") runs the package __init__ and propagates its exceptions | the interim S3-18 guard is import-free and S4-22 is only a refinement | VERIFIED | read: /home/marielle/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/lib/python3.13/importlib/util.py:80-91 |
| A37 | S1 makes the registry the single source, exposes it to S4 as opt-in nested-field metadata on the unified RunSpec (snapshot JSON only for non-Python consumers), and assigns the knob document and lowering to S4-20 | schemagen would have to read the JSON snapshot instead of field metadata | VERIFIED | read: .praxia/docs/specs/261001_spec-system-unification.md:351; read: .praxia/docs/specs/261001_spec-system-unification.md:481-491; read: .praxia/docs/specs/261001_spec-system-unification.md:531 |
| A38 | S5's catalog needs the manifest field set HUB-REQ 1-10 with ports at the manifest level (one manifest is one node type) | operations[] inside one manifest would be acceptable to the catalog builder | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:276-292 |
| A39 | S6 needs ref-addressed nodes, port-addressed edges with ids and kind, an inline code node, a declarative port-type table with exact/widen/reject plus a corpus and hash vectors | the IR v2 shape could stay close to HostPrepGraph v1 | VERIFIED | read: .praxia/docs/specs/261001_pipeline-editor.md:295-300; read: .praxia/docs/specs/261001_pipeline-editor.md:319-327; read: .praxia/docs/specs/261001_pipeline-editor.md:674-676 |
| A40 | Python float formatting via repr followed by ES6-style reformatting reproduces RFC 8785 number text for the vectors S6 lists (0.1, 1e-5, 1e16, 1.0, 2^53 ints) | a vetted JCS library or a hand-written dtoa is needed | UNVERIFIED | deferred: needs running code, S4-30 gate runs the shared vectors with naive json.dumps as the negative control |
| A41 | S4 item ids can be renumbered freely | none, ids stay stable | REFUTED | read: .praxia/docs/specs/261001_aminx-hub.md:210; read: .praxia/docs/specs/261001_aminx-hub.md:296-297; read: .praxia/docs/specs/261001_spec-system-unification.md:531; changed: S4-04, S4-09, S4-10, S4-14, S4-18, S4-19, S4-20 keep their ids, new work is appended as S4-30 to S4-33, and the part of S1's "S4-20" that is large (the knob document and lowering) is split out as S4-33 which S4-20 depends on, so S1's statement that S4-20 deletes the drift test stays literally true |
| A42 | `dist.locate_file(<relative path>)` finds package data for every install mode, including editable installs | the entry-point value cannot be a distribution-relative data path | REFUTED | read: /home/marielle/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/lib/python3.13/importlib/metadata/__init__.py:921-922 (`self._path.parent / path`, relative to site-packages); changed: the entry-point value is a top-level package name located with `find_spec`, not a path (4.8); r1 C5 |
| A43 | Build backends accept a path-valued entry-point value such as `molxmpnn/manifests/x.json` | the entry-point value cannot be a path | REFUTED | read: .venv/lib/python3.13/site-packages/setuptools/config/_validate_pyproject/formats.py:336-358 (module part must be dotted identifiers); read: .venv/lib/python3.13/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_validations.py:35 (entry-point values carry `format: python-entrypoint-reference`); read: pyproject.toml:3 (aminx builds with `setuptools.build_meta`); changed: the value is the bare module form `importable.module` (a top-level package name), which the grammar accepts (4.8); r1 C6 |
| A44 | `importlib.util.find_spec("<top-level name>")` does not import the package or run its `__init__` (the parent import happens only for dotted names) | the discovery step runs model code and the alphex D5 inversion returns | VERIFIED | read: /home/marielle/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/lib/python3.13/importlib/util.py:89-100 (r1 C5) |
| A45 | `find_spec` of the top-level package resolves its directory (`submodule_search_locations`) for editable installs (setuptools `.pth` and finder modes), not only for built wheels | pre-stated fallback: PEP 610 `direct_url.json` source root for editable dists, `dist.files` scan for wheel installs | UNVERIFIED | deferred: needs an editable install, which this task forbids; gated by S4-11 (synthetic `.pth` and finder-mode editable fixtures) and S4-21/S4-23 (real editable dev install) (r1 C5) |
| A46 | import-linter (grimp) treats a function-level `import numpy` as an import edge | n/a: the design avoids the question, `wire.py` imports no array library and numpy conversion lives xtrax-side | UNVERIFIED | deferred: needs running lint-imports; the S4-04 negative control (a planted function-level `import numpy` must fail the lint) settles it (r1 C15) |
| A47 | xtrax is published on PyPI and the MPNN repo resolves it from there; plegadx uses a path-editable xtrax | the release channel for `xtrax-contract` differs by consumer | VERIFIED | read: pyproject.toml:311-313; read: /home/marielle/projects/plegadx/pyproject.toml:22 (r1 C3); the channel itself is Q11 |
| A48 | S6's draft files S6-25 and S6-26 for deliverables (builtin scheme, required/cardinality, corpus, JCS vectors) that S4 also files | no duplicate work | VERIFIED | read: .praxia/docs/specs/261001_pipeline-editor.md:24; read: .praxia/docs/specs/261001_pipeline-editor.md:83; read: .praxia/docs/specs/261001_pipeline-editor.md:232; changed: S4 is sole owner, S6-25 and S6-26 superseded (section 6); r1 C4 |
| A49 | v1 `deserialize_node` resolves `callable_ref` eagerly by import, and `_callable_to_import_path` accepts a callable only if `module.qualname` resolves back to the same object | the v2 serialise/upgrade rule would have to be invented rather than reused | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:46-75; read: xtrax@v0.4.0a11:src/xtrax/composition/serialize.py:77-133 (r1 C11) |
| A50 | xtrax at a11 has a single-source gate for the IR schema: `scripts/audit_ir_schema_single_source.py` scans `src/xtrax` and allows `$schema` only in `inference/ir_schema.py`; `tests/audit/test_ir_schema_single_source.py::test_real_src_tree_is_clean` runs it; `emit_ir_schema` hand-builds the `Node` and `Edge` schemas plus a `NodeMetadata` `$defs` entry; `node_metadata_schema.toml` and `load_node_metadata_schema` live in `xtrax.composition` | a new package outside `src/xtrax` would pass the gate mechanically while breaking its intent, and nothing says who owns Node/Edge | VERIFIED | read: xtrax@v0.4.0a11:scripts/audit_ir_schema_single_source.py:19-20; read: xtrax@v0.4.0a11:tests/audit/test_ir_schema_single_source.py:112; read: xtrax@v0.4.0a11:src/xtrax/inference/ir_schema.py:223-264; read: xtrax@v0.4.0a11:src/xtrax/composition/node_metadata.py:137 (r2 R2-1) |
| A51 | Exactly 17 files under `src/` at a11 mention `callable_ref` or `HostPrepGraph` (list in 4.11); the extra hits seen in a stale checkout (`inference/__init__.py`, `cli/graph_author_verb.py`) are not at a11 | the audit scope of S4-44 is larger | VERIFIED | read: `grep -l -E 'callable_ref\|HostPrepGraph'` over the v0.4.0a11 tree, `src` only, in ~/projects/xtrax (17 paths listed in 4.11; r2 R2-6) |
| A52 | xtrax builds with hatchling, `version` is dynamic from `src/xtrax/__init__.py`, `requires-python >=3.13`, and the a11 root pyproject has no uv workspace table | the packaging wiring (S4-04) would already exist or differ | VERIFIED | read: xtrax@v0.4.0a11:pyproject.toml:3; read: xtrax@v0.4.0a11:pyproject.toml:6; read: xtrax@v0.4.0a11:pyproject.toml:16-17; read: xtrax@v0.4.0a11:pyproject.toml:155-156 (r2 R2-2; no `workspace` hit in a `grep -n` of the file) |
| A53 | plegadx `export_shape_contract.json` has top-level `version`, `spec_markdown`, `static_shape_policy`, `gather_info_policy` and `exports`; each export carries `symbol` and `traced_inputs[]` whose entries have `name` (a dotted pytree path such as `token_features.aatype`), `dtype`, `shape_spec` (a string such as `(N_tokens, 447)`), `example_shape`, `variability_axis` and `static_arg_required` | the importer needs only dtype and rank | VERIFIED | read: /home/marielle/projects/plegadx/src/plegadx/export_shape_contract.json:1-40 (r2 R2-13; refines A31, which called the list `inputs`) |
| A54 | proteinsmc's public MPNN scorer is `make_mpnn_score(mpnn_model_params, inputs, decoding_settings)` registered as `"mpnn"`; its guarded import branch still names `prxteinmpnn` and `proxide`; proteinsmc depends on `alphex` | the S4-48 adapter could keep the old signature | VERIFIED | read: /home/marielle/projects/proteinsmc/src/proteinsmc/scoring/mpnn.py:64-68; read: /home/marielle/projects/proteinsmc/src/proteinsmc/scoring/mpnn.py:17-19; read: /home/marielle/projects/proteinsmc/src/proteinsmc/utils/fitness.py:30; read: /home/marielle/projects/proteinsmc/pyproject.toml:30 (r2 R2-3) |
| A55 | proteinsmc treats a higher fitness as better (it reports `max_fitness` as the maximum of the combined fitness over particles), so a `minimize` NLL must be negated at the adapter | the sign rule in 4.9 would invert the optimisation | UNVERIFIED | deferred: the max-reporting anchor is read (`proteinsmc/sampling/initialization_factory.py:377`) but the SMC weight update that consumes fitness was not traced; S4-48 begins by anchoring it with a `path:line`, and its gate carries a sign-flip negative control (r2 R2-3) |
| A56 | RFC 8785 sorts object members by UTF-16 code units, which differs from Python's code-point `sorted()` for an astral key against a BMP key above U+D800 (U+FF5E against U+10000) | the extra S4-30 vectors are unnecessary | UNVERIFIED | deferred: the spike harness is absent from this worktree (section 2); an inline probe printed the BMP key first under `sorted()` and the astral key first under a UTF-16-BE key sort, a sanity check and not a citation; the S4-30 vectors settle it against the RFC text (r2 R2-12) |
| A57 | `node_metadata_schema.toml` defines `nl_description` as required, type string, `min_length = 1`, plus optional slots (`mathjax_label`, `citations`, `script_usage`) | the stdlib validator needs more than one slot rule | VERIFIED | read: xtrax@v0.4.0a11:src/xtrax/composition/node_metadata_schema.toml:8-30 (r2 R2-1) |

## 4. Design

### 4.0 Principles

- **P1. One source, derived artifacts.** Every wire format has exactly one Python definition.
  JSON Schemas, TS types, generated validators and the per-model param schemas are *emitted*
  from it and committed, with a drift test (same discipline as `ir_schema`'s drift tests).
  This is the reading of fork 13 that this spec keeps: no hand-authored second schema.
- **P2. The contract is data; behaviour is bound.** A manifest never contains code. Code is
  reached through an *executor entry* (an import path, artifact ids plus a driver, an npm
  specifier).
- **P3. The contract imports nothing heavy.** Stdlib only, Python >= 3.11, so proteinsmc-class
  and browser-tooling consumers can use it (A7, A8).
- **P4. Fail loud on versions.** Reuse the PM3 rule from `serialize.py`: no default-fill of
  `schema_version`; older-than-minimum and newer-than-known both raise. Unknown executor kinds
  and unknown port types are errors unless prefixed `x-` or namespaced.
- **P5. Claims never exceed verification.** Each manifest carries `validation.evidence` records
  mirroring xtrax's `VerificationLevel` (`executed`, `codegen_only`, `validated`) plus the
  executor and a scope string. A UI may only render a claim the records support.
- **P6. Long work is resumable by construction** (section 4.10, section 7).

### 4.1 Where it lives (scope item f)

Options evaluated against facts read above:

| | F1: `xtrax.contract` inside xtrax | F2: separate dist `xtrax-contract` built from the xtrax repo | F3: generated JSON Schema + TS only, no Python package |
|---|---|---|---|
| Single source of truth (P1, fork 13) | yes | yes, if schemas are emitted from its types with a drift gate | only if schemas are generated, Python consumers then hand-copy types |
| Installable by Python 3.11 / non-JAX consumers (A7, A8) | **no**: xtrax needs Python 3.13 and jax | **yes** (`requires-python >=3.11`, no dependencies) | n/a |
| Browser / TS consumers | extract from wheel | generated `ts/` shipped | yes |
| Release cost | one train | two trains in one repo, `xtrax` depends on `xtrax-contract` | one |
| Failure mode | proteinsmc-class consumers cannot take it | version skew between the two dists | "two competing formats" via hand-maintained Python types |

**Recommendation: F2**, with `xtrax.contract` kept as a thin re-export shim so existing xtrax
users see one namespace. The deciding fact is A7/A8: an in-xtrax contract is uninstallable on
the Python floor of the consumers D6 is protecting. Fork 13 is respected because the contract
types are the single source and `emit_ir_schema` becomes a composer over them (S4-45, with its mapping delegated in S4-08). This is
Q1 and the first backlog item (S4-01, user decision), so that no later item builds on an
unconfirmed layout.

```
xtrax/                                          (repo, unchanged identity)
  packages/xtrax-contract/                      NEW distribution, import name xtrax_contract
    pyproject.toml                              requires-python >=3.11, dependencies = []
    src/xtrax_contract/
      __init__.py  version.py  errors.py
      canonical.py    RFC 8785 JCS dumps + sha256 (S4-30)
      ports.py        PortSpec, port-type table loader, compat() evaluator
      port_types.v1.json   declarative port-type table (S4-04)
      wire.py         PortValue descriptor + safetensors-layout codec over bytes/memoryview (imports no array library)
      executors.py    ExecutorEntry kinds, ArtifactEntry (+ per-format details)
      manifest.py     ModelManifest, catalog_view()
      graphdoc.py     GraphDoc / NodeDoc / EdgeDoc  (graph IR v2 wire dataclasses)
      validate.py     pure-data graph validator + diagnostic code registry (S4-31)
      profile_validate.py  param-profile@1 subset validator, stdlib (S4-47); E_PARAM_SCHEMA uses it
      emit.py         the ONLY module that writes schemas/ and ts/ (S4-09; `--check` drift gate)
      schemagen.py    dataclass / eqx.Module to param-profile schema, FieldMeta hook (S4-07)
      registry.py     discover_models(), NodeTypeRegistry (lazy, isolated)
      protocols.py    Scorer, Sampler, StructurePredictor, EnergyModel (typing.Protocol)
      testing.py      assert_conforms(...)
      schemas/        GENERATED, committed:  model-manifest.v1, graph-ir.v2, port-types.v1 ...
    ts/               GENERATED, committed:  index.d.ts, schemas.mjs, validators.mjs
                      hand-written once: profile_validate.mjs (S4-47), package.json (dev: typescript)
    conformance/      graph-validate.v1.json (corpus), canonical/jcs.v1.json (hash vectors),
                      param-profile.v1.json (profile-subset corpus, S4-47)
    tests/
  src/xtrax/contract/__init__.py                re-export shim
  src/xtrax/contract/numpy_codec.py             NEW: numpy <-> PortValue conversion (xtrax side, may import numpy)
  src/xtrax/composition/                        HostPrepGraph v2 built on xtrax_contract.graphdoc
  src/xtrax/export/bundle.py                    NEW: export_onnx_bundle (section 4.10)
  importlinter contract (new, in xtrax)           xtrax_contract must not import jax, numpy,
                                                equinox, xtrax
```

**Numpy reconciliation (r1 C15).** `wire.py` writes and reads the safetensors layout (8-byte
little-endian header length, JSON header, raw bytes) over `bytes`/`memoryview` and imports no
array library, so the import-linter contract needs no `ignore_imports` entry. Array conversion is
`xtrax.contract.numpy_codec` in the xtrax distribution. S4-04 carries the negative control: a
planted function-level `import numpy` in the contract package must fail `lint-imports` (A46).

**Release and pin policy (r1 C3).** `xtrax-contract` is a new distribution, so nothing can import
it until it is released. S4-34 releases `xtrax-contract` and a new `xtrax` (which pins
`xtrax-contract>=X,<next minor`); the channel is Q11 (recommendation: PyPI, matching how the MPNN
repo already resolves xtrax, A47). Consumers add the dependency in explicit pin items: S4-35 (MPNN
repo: `xtrax` bump plus `xtrax-contract`), S4-36 (plegadx). proteinsmc takes **no** pin (D6).
Every item that imports `xtrax_contract` in a consumer repo depends on its pin item.

**Two-distribution build wiring (r2 R2-2, owned by S4-04).** At a11 the root builds with
hatchling, a dynamic version read from `src/xtrax/__init__.py`, and has no uv workspace (A52).
S4-04 therefore owns, and its gate checks, all of the following, so that S4-08 and S4-10 can
import `xtrax_contract` the moment they land:

- root `pyproject.toml`: `[tool.uv.workspace] members = ["packages/xtrax-contract"]`,
  `[tool.uv.sources] xtrax-contract = { workspace = true }`, and `xtrax-contract>=<initial>,<next
  minor>` appended to `[project] dependencies` (initial version `0.1.0a1`, so the range is
  `>=0.1.0a1,<0.2`);
- sub-dist `packages/xtrax-contract/pyproject.toml`: hatchling backend (the same backend as the
  root, so one build toolchain), `requires-python = ">=3.11"`, `dependencies = []`,
  `[tool.hatch.version] path = "src/xtrax_contract/version.py"`, `packages = ["src/xtrax_contract"]`
  with the committed `schemas/` and `ts/` files force-included;
- **version and tag policy (recorded in the S4-01 ADR):** the two dists version independently.
  `xtrax-contract` follows the schema versions it carries: a change to any committed schema
  version bumps its minor; additive changes bump its patch. The root keeps its existing
  `v<version>` tags; the contract is tagged `xtrax-contract-v<version>`. S4-34 pushes both tags
  in one release, and the lower bound `X` of the `xtrax` dependency is rewritten to the
  contract version being released in the same commit. Between releases the workspace source
  resolves the dependency inside the repo, which is why the range can name a version that is not
  yet on the channel.

**Generation toolchain (r2 R2-7, decided in the S4-01 ADR).** Generation of schemas, TS types and
the validator data bundle is **stdlib Python** (`xtrax_contract.emit`); Node is only a dev and CI
dependency for checking (`tsc --noEmit`, `node --test`), declared in
`packages/xtrax-contract/ts/package.json` as devDependencies (`typescript` only), so the Python
xtrax repo gains a Node toolchain for checking but none for building a wheel. The TS declaration
generator walks the param-profile subset (4.4), which is small enough for a ~150-line walker; no
`json-schema-to-typescript`, no `ajv`. The validator is **one hand-written engine per language for
the fixed keyword subset** (S4-47: `profile_validate.py` and `profile_validate.mjs`), both run
against one shared corpus (`conformance/param-profile.v1.json`). `validators.mjs` is a thin
generated binding of each committed schema to the JS engine. An engine for a fixed keyword
subset is not a second definition of any wire format (P1 concerns formats), and the corpus
arbitrates between the two engines. S4-20 reuses the same generator entry point and the same JS
engine for the knob schema; it adds no generator of its own.

If the user prefers less machinery (Q1 alternative): land F1 first and carve out later; the
package-relative layout above makes that a pure move, but proteinsmc-class consumers stay
blocked until the carve-out.

### 4.2 Port vocabulary and the type table (scope item a)

The vocabulary is a **declarative table** (`port_types.v1.json`, `port_types_version = 1`), not
code, because S6 implements a TypeScript evaluator over the same table and both must pass one
corpus (A39). Core type names are bare (`structure`), matching what S5 and S6 already use;
extension types are namespaced `<ns>/<name>` and must declare an adapter route to a core type.

| Port type | Meaning | Tensors (name: dtype[shape]) | Descriptor fields |
|---|---|---|---|
| `structure` | one macromolecular structure | encoding `backbone4`: `coords: f32[L,4,3]` (N, CA, C, O), `mask: f32[L]`, `residue_index: i32[L]`, `chain_index: i32[L]`, optional `native_tokens: i32[L]` | `encoding` in {`backbone4`, `pdb`, `mmcif`}; `chain_ids: str[L]`; `alphabet` for `native_tokens`. Text encodings carry UTF-8 text, no tensors. **Every structure producer MUST be able to emit `pdb`** (lowest common denominator, matches localfold's PDB-string interface) |
| `sequence` | residue tokens | `tokens: i32[L]` or `[B,L]` | `alphabet: {name, symbols}` (alphex name such as `MPNN_X_21` plus the literal symbols, so an alias cannot hide a different ordering), optional `chain_breaks`, text view `fasta` |
| `logits` | per-position scores | `logits: f32[L,A]` | `alphabet`, `scale` in {`logit`, `log_prob`}, `temperature_applied: bool` |
| `ensemble` | structures sharing a topology (ensemble or trajectory) | `frames: f32[T,N,3]`, optional `weights: f32[T]` | `axis` in {`ensemble`, `time`}, `reference` structure descriptor, optional `dt` |
| `grid` | scalar field on a regular grid | `data: <dtype>[X,Y,Z]`, `origin: f64[3]`, `spacing: f64` (scalar) or `f64[3]` | `units`, `meaning` |
| `energy-map` | refines `grid`, widens to `grid`; the scope's "EnergyMap". Alias `gfe-grid` (S5's isochore `read_dx`/`write_dx` shape) | as `grid`, `data: f32` | `meaning` in {`gfe`, `pmf`} |
| `count-grid` | refines `grid`, widens to `grid` (voxel-count histogram, S5's `smooth_density` input) | as `grid`, `data: integer` | `meaning = count` |
| `density-grid` | refines `grid`, widens to `grid` (S5's `smooth_density` output) | as `grid`, `data: f32` | `meaning = density` |
| `file` | opaque bytes with a name, for tool inputs such as `.dx` or a reference PDB | `bytes: u8[N]` | `name`, `media_type`; a map of files is a `file` port with `cardinality: many` and per-element `key` |
| `json` | a JSON value, for tool results | JSON | optional `schema_ref` |
| `alignment` | MSA | either `tokens: i32[N,L]` or text (`a3m`/`fasta`), optional `weights: f32[N]` | `alphabet` |
| `scalar` | named scores | `values: f32[K]` | `names: str[K]`, per-name `direction` in {`maximize`, `minimize`}, `units` (A33: the sign convention is declared, not assumed) |
| `tensor` | generic array | `values` | optional `dtype`, `rank`, named `axes` |
| `runspec` | refines `json`: a params document typed by an operation's params schema | JSON | `schema_ref` (required) |

`af3-job` (S5's AF3 job JSON) is an extension type `localfold/af3-job`, registered by S5 in the
table's `extensions` section with an adapter route from `sequence`; nothing else in the contract
depends on it.

**`compat(src, dst) -> exact | widen | reject` (v1, no coercion; matches S6's evaluator contract):**
1. Resolve names and aliases. Same name: compare `dtype` (both given must be equal), `rank` (both
   given must be equal), named `axes` (both given must be equal), and, for `sequence`/`logits`,
   **`alphabet` name (both given must be equal)**. Equal on every given attribute is `exact`; the
   source being more specific than an unspecified destination attribute is `widen`; the source
   less specific than a destination that specifies it is `reject`.
2. Different names: `widen` iff `dst` is reachable through the table's `widens_to` edges from
   `src` (for example `energy-map` to `grid`); otherwise `reject`.
3. The alphabet rule exists because alphex's census found sentinel placement and ordering to be
   the real source of shipped bugs (a length-20 table serving a 21-valued domain). Two `sequence`
   ports with different alphabets must not connect silently.
Symbolic-dimension unification is **reserved** (the table carries named axes, evaluator v1
checks concrete dimensions only, as S6's v1 does); adding it is a corpus version bump.
Encoding mismatch (producer and consumer share no encoding) is a warning that an adapter node is
needed, never a silent conversion.

**Port names and extension keys (r2 R2-13).** A port name matches
`[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*`: a dot-separated path, because plegadx's traced
inputs are named by pytree path (`token_features.aatype`, A53). Manifests written by hand use
single-segment names; the dotted form is legal everywhere a port name is (edges, `catalog_view`).
Port objects accept `x-<ns>` extension keys, preserved verbatim and ignored by `compat`. The
plegadx importer (S4-24) maps `name` to the port name, `dtype` to `dtype`, the rank to the
number of dimensions parsed from `shape_spec`, and keeps what has no home under `x-plegadx`:
`shape_spec` **verbatim** as a string (`x-plegadx.shape_spec`), `example_shape`,
`variability_axis` and `static_arg_required`. The two policy blocks (`static_shape_policy`,
`gather_info_policy`) ride on the manifest as `x-plegadx.policies`. "Round-trips equal to the
source JSON" in S4-24 means: for every `traced_inputs` entry, `export(import(entry)) == entry`
as parsed JSON (key order irrelevant), with `x-plegadx` as the only carrier of the fields the
port model has no slot for. The port model does not grow slots for them.

**Wire format of a port value.** One self-describing file: a safetensors container whose
`__metadata__["xtrax.port"]` holds the descriptor JSON (A19), tensors under the fixed names
above. Text encodings are the descriptor with a `text` field and no container. The Python codec
lives in `wire.py` and works on `bytes`/`memoryview` with no array library (4.1); numpy
conversion is `xtrax.contract.numpy_codec`. The JS reader is the generated `ts/` package plus a ~30-line safetensors header reader. The
byte-layout assumption is A20 and is checked by S4-04's gate against the reference writer.

### 4.3 Model manifest (scope item b)

**One manifest is one node type** (A38): HUB-REQ puts `ports[]` at the manifest level, and S6's
`ref` is `<id>@<version>`. A model with several operations (MPNN `sample` and `score`) ships
several manifests that share a `family` and repeat the same artifact entries (the catalog
dedupes by sha256). The model repo's build generates all of them from one source and a drift
test asserts the shared artifact entries are identical. Shape (dataclasses in `manifest.py`):

```jsonc
{
  "schema_version": 1,                          // required, never default-filled
  "id": "<namespace>/<family>.<op>",            // e.g. molxmpnn/proteinmpnn.sample; grammar owned by S4-01, see below
  "kind": "model",                              // model | tool (isochore-class entries are tools)
  "role": "sampler",                            // scorer | sampler | predictor | energy | transform | adapter
  "family": "proteinmpnn", "version": "<model package version>",
  "title": "...", "description": "...",
  "ports": [
    {"name": "structure", "direction": "in", "type": "structure", "required": true,
     "cardinality": "one", "accepts_encodings": ["backbone4", "pdb"], "shape": {"coords": ["L", 4, 3]}},
    {"name": "sequences", "direction": "out", "type": "sequence", "alphabet": "MPNN_X_21"},
    {"name": "scores", "direction": "out", "type": "scalar"}
  ],
  "params": {"$ref": "<param-profile schema>"}, // section 4.4; inline or an artifact id
  "artifacts": [
    {"id": "enc-128", "format": "onnx", "role": "graph", "bucket": 128, "sha256": "...", "bytes": 0,
     "urls": ["..."], "details": {"graph": "encoder", "opset": 23, "self_contained": true,
        "inputs": [{"name": "<as read from the ModelProto>", "port": "coords", "dtype": "float32", "shape": ["L",4,3]}],
        "outputs": [], "dtype_census": {}, "rng_ops": []}},
    {"id": "weights", "format": "eqx-zst", "role": "weights", "sha256": "...", "bytes": 0,
     "urls": ["hf://<org>/<repo>@<40-hex revision>/<path>"]}
  ],
  "executors": [
    {"id": "py", "kind": "python", "factory": "module:symbol", "jittable": true, "requires": []},
    {"id": "ort", "kind": "onnx", "artifacts": ["enc-128"], "requires": ["wasm"],
     "driver": {"url": "...", "sha256": "...", "export": "createSession"},
     "determinism": {"rng_scheme": "js-gumbel", "cross_executor_equivalent": false}}
  ],
  "license": {"code": "<SPDX>", "weights": "<text or URL>", "redistribution": "allowed|gated|forbidden",
              "attribution": "..."},
  "provenance": {"git_sha": "...", "checkpoint_id": "...", "xtrax": "...", "jax": "...",
                 "jax2onnx": "...", "onnxruntime": "..."},
  "validation": {
    "golden_vectors": [{"artifact": "gv-1", "tolerance": {"atol": 0, "rtol": 0}}],
    "evidence": [{"level": "executed", "executor": "ort", "scope": "ort-cpu",
                  "claim": "tokens exact vs JAX reference", "bathos_id": "<run id>"}]
  },
  "limits": {"max_length": 256, "buckets": [128, 256], "min_memory_mb": 0}
}
```

**Id, entry-point and directory grammar (owned here, r3 C12).** The S4-01 ADR fixes it; S3
provides only the root name `molxmpnn` (`S3:molxmpnn-manifest-namespace`) and the repo/release
facts (`S3:molxmpnn-repo`), and defines no manifest strings. An `id` is
`<namespace>/<family>.<op>`: `namespace` is the providing project's root name, `family` is the
model family, `op` is the operation (`sample`, `score`; not the `role` enum). The MPNN ids are
`molxmpnn/proteinmpnn.sample` and `molxmpnn/proteinmpnn.score`. The `xtrax.models` entry-point
name is the manifest file stem, derived from the id as the part after `/` with `.` replaced by `-`
(`proteinmpnn-sample`, `proteinmpnn-score`); manifests live in `<package>/manifests/<stem>.json`
(4.8). S4-19, S4-21 and the S6 `ref` values (`<id>@<version>`) use these strings verbatim, and the
S4-19 resolve test fails if an id and its entry-point stem disagree.

**Source mode versus release mode (r1 C7, r3 C8).** An artifact's `urls` entries are real
locations in a released manifest. A manifest committed in a model repo before its artifacts are
published is a *source-mode* manifest (validated with `allow_placeholders=True`), and source mode
permits exactly two pending states: the placeholder scheme `release-asset:<filename>` in any
`urls` entry, and the literal `"pending"` in `executors[].driver.sha256` (and only there: a
pending `sha256` on any artifact entry is rejected in both modes). Release mode rejects both.
`catalog_view` refuses placeholders and a pending driver hash unless the caller passes
`allow_placeholders=True` (dev). A source-mode `onnx` entry listing more than one graph therefore
validates with `driver.url` a `release-asset:` placeholder and `driver.sha256` pending; the driver
bundle is produced downstream (S5-14 fills the driver hash in the source manifest), and S5-52 owns
rewriting placeholders to the real release URLs and the driver `url`/`sha256` when it publishes,
after which the manifest validates in release mode (hosting itself stays a non-goal here, r1 C7;
Q12). S4-19 does not yield a final `onnx` entry. Every `hf://` URL must carry a 40-hex revision (a manifest that cannot pin its weights
cannot be reproduced; contrast `hf_weights.py:109`, fixed by S4-15) and every artifact has a
sha256. `provenance` is written by the producer, not typed by hand. `validation.evidence` is the
only source a UI may use for "verified" claims (P5), and `scope` keeps "ORT CPU (xtrax)"
distinct from "ORT-Web wasm" exactly as S5's badge rule requires. `ports[].required` and
`cardinality` (`one` or `many`) are the fields S6 needs. `xtrax_contract.catalog_view(manifest)`
returns the flattened shape S5's catalog builder consumes (derived dtype and shape per port from
the type table). S5's catalog builder (S5-04) runs in Node with no Python, so it re-derives the
same fields in JS; it is tested against the `catalog_view` snapshot corpus S4-09 commits as
conformance data (`conformance/catalog_view/`), so the two derivations cannot silently
diverge.

**Manifest hash (coherence round 2, CH2-05).** `manifest_sha256 = sha256(JCS(manifest))` (RFC 8785, `xtrax_contract.canonical`, S4-30) over the manifest document. This one definition is what the graph IR `models` lock table stores, what S6 uses for spec keys, and what S5 echoes as `manifest_sha256` on the remote wire and in the RunStore unit key; a server that computes any other hash (for example over the file bytes) makes a correct endpoint fail the echo check. The sha256 of a manifest *release asset* (the bytes pinned in `catalog/sources.toml`) is a separate artifact-integrity value and is never called `manifest_sha256`. S4-30's shared vector file includes a manifest-hash vector (a small manifest document and its expected `sha256(JCS(..))`) that the Python and the TypeScript (S6-35, S5 Node builder) canonicalisers must both reproduce.

**Inbound amendment S5-51 (r3 C9, narrowed in the convergence check).** G1-G4 (weights-less `tool`,
`external`/`byo` artifacts that waive `urls` and the 40-hex rule, `pyodide` `fs_in`/`fs_out`,
`determinism` on every executor kind) are additive relaxations and additions, and S5-51 lands them in
`xtrax-contract`. Two further hub requests are not amendments. G5 (`grid.meaning` in `compat`) is
withdrawn, because the type table in 4.2 already carries the distinction as distinct types
(`energy-map` with alias `gfe-grid`, `count-grid`, `density-grid`, each widening to `grid`, with compat
rule 2 rejecting `count-grid` to `energy-map`). G6 (the `catalog_view` snapshot corpus) is S4-09's own
deliverable (`conformance/catalog_view/`, above). Decision (S4 owns the schema, so S4 records it):
**additive within `model-manifest.v1`, because nothing is released yet.** S4-34 depends on S5-51 so the
first published `v1` already carries the amendments and there is never a second meaning of
`model-manifest.v1`. **Pin rule:** S5-24, S5-42 and S5-45 (through S5-64) pin the S4-34 release, not the
S4-09 commit. S5-04 is the one deliberate exception: it builds against the S4-09 commit, which already
carries the schemas and the `catalog_view` corpus it needs and none of the amendments, so the catalog
builder does not wait for the S4-34 release; S5-64 then re-pins it to the S4-34 release and re-runs the
derivation against the released corpus. If S5-51 is declined or lands after S4-34, the amendments are a
`schema_version` 2 bump (P4), never a silent change to `v1`. `emit --check` (S4-09) guards the amended
schemas like the rest.

### 4.4 Param schemas (scope item c, first half)

`params` is a JSON Schema in the **xtrax-param-profile@1**: a deliberately small keyword subset
(`type`, `enum`, `const`, `minimum`, `maximum`, `items`, `prefixItems`, `properties`,
`required`, `additionalProperties`, `oneOf` with a required discriminator, `default`, `title`,
`description`, `examples`, `$ref` within the document) plus an `x-xtrax` annotation object
(`unit`, `group`, `advanced`, `widget`, `portable`). The subset is the contract with S6: a form
renderer and a generated validator only have to understand it, and S6's palette already fails
loudly on an unsupported keyword. An emitted schema using any other keyword fails emission.

**Committed schemas (what S4-09 emits).** Exactly four, plus TS types and validators for each:
`model-manifest.v1`, `port-types.v1`, `param-profile.v1`, `graph-ir.v2`. The diagnostic code
registry and conformance corpus (S4-31) and the JCS vectors (S4-30) are data files, not emitted
schemas. S4-09 therefore depends on S4-10 (which creates `graphdoc`) so the drift gate covers
the graph schema (r1 C2).

**Checking `params` against the profile (r2 R2-7).** `E_PARAM_SCHEMA` (4.6) and the browser's
form validation need an engine for exactly this keyword subset, not full JSON Schema. S4-47
delivers it as its own item: `xtrax_contract.profile_validate` (stdlib, `dependencies = []`) and
`ts/profile_validate.mjs`, both run against `conformance/param-profile.v1.json` (documents,
schema, expected error paths, including `oneOf` discriminators, `prefixItems`, `$ref` and
`additionalProperties`, and a planted unsupported keyword that must be refused). The corpus is
the arbiter between the two languages. `validate.py` (S4-31) and the generated `validators.mjs`
(S4-09) both call these engines; nothing else in the repo validates params.

### 4.5 Executor entries (scope item d, first half)

`executors[]` entries have a closed set of kinds (P4). They name *what artifact shape* an entry
is; S5's hub `Executor` classes name *who runs it*, and each declares which manifest kinds it
handles:

| Manifest kind | Fields | Run by (S5 executor kind) | Notes |
|---|---|---|---|
| `python` | `factory: "module:symbol"`, `jittable`, `requires` | none in the hub (local / remote) | resolved lazily at *call* time, never at discovery |
| `onnx` | `artifacts: [ids]`, `driver {url, sha256, export}`, `requires`, `determinism` | `ort-web` (and ORT CPU in Python) | `driver` is **required** when more than one graph is listed: MPNN's loop is JS-driven over four graphs (`split_driver.mjs`). In source mode `driver.url` may be a `release-asset:` placeholder and `driver.sha256` may be `"pending"` (only there, 4.3); release mode requires both real |
| `iree` | `target: <xtrax Target name>`, `artifacts`, `size_bytes` | none today | levels copied from xtrax (`targets.py`): `wasm32` is `codegen_only` |
| `npm` | `module: <specifier or URL + integrity>`, `export`, `requires: ["webgpu"]` | `webgpu-localfold` | localfold-style kernel libraries. AF3 is not bindable today (A29) |
| `pyodide` | `wheel: URL + sha256`, `entry: "module:symbol"` | `pyodide` | pure-Python / numpy tools (isochore-class). No JAX assumption (A34) |
| `remote` | `endpoint`, `protocol: "xtrax-http@1"`, `requires_network: true` | `remote` | never auto-selected; the hub's consent hook gates it. For the MPNN manifests this entry (protocol, runtime reference, `determinism.rng_scheme`, gap G4) is added by S5-21, not by S4-19, whose gate validates only `python` and `onnx`; source-mode validation accepts a `remote` entry without a `driver` |

**The driver ABI is S5's** (`createSession(ctx)` returning `{run(inputs, opts), dispose()}` over a
pre-bundled module with no imports, A38/S5:338-350). S4 carries only the reference
(`url`, `sha256`, `export`) and requires that a driver's `run` takes and returns `PortValue`s
keyed by the manifest's port names. The Python-side equivalent is the `Scorer` / `Sampler`
protocols (4.9), which are typed conveniences over the same `inputs -> outputs` shape.

**OnnxBundle@1** is a *fragment of a manifest*, not a separate format: `export_onnx_bundle`
(4.10) emits `{artifacts: [...], producer: {...}}` where each artifact has the per-format
`details` block shown in 4.3. It generalises aminx's existing per-graph manifest (A21): per graph
`file, bytes, sha256, bucket`, graph input/output *names as read from the ModelProto* with port
names, dtypes and shapes, dtype census (int64 producers matter for ORT Web WebGPU), RNG audit,
`self_contained`, plus `opset`, producer versions and `jax_enable_x64: false`. This replaces the
hand-pinned order constants in `split_driver.mjs` (A23) with data, bound by recorded name rather
than position, because `convert_to_onnx` only promises leaf order (A13).

### 4.6 Graph IR v2 (scope item d, second half)

`GRAPH_SCHEMA_VERSION = 2`, `MIN_SUPPORTED_GRAPH_SCHEMA_VERSION = 1` (A3). The wire dataclasses
(`GraphDoc`, `NodeDoc`, `EdgeDoc`) live in `xtrax_contract.graphdoc`; `HostPrepGraph` stays the
live Python representation and converts to and from them (a round-trip property test is the
guard against "two formats"). Shape follows S6's embedded-IR contract (A39):

```jsonc
{
  "schema_version": 2,
  "models": {"<id>@<version>": "<sha256(JCS(manifest))>"},        // optional lock table
  "nodes": [
    {"id": "n_k3f9a2qd", "ref": "<manifest id>@<version>", "params": {}, "executor_hint": null,
     "metadata": {"nl_description": "..."}, "frozen": false},
    {"id": "n_p0x7c1mw", "ref": "builtin:code.python", "source": "def run(inputs, params): ...",
     "ports": {"in": [], "out": []}, "requirements": ["numpy"],
     "metadata": {"nl_description": "..."}}
  ],
  "edges": [
    {"id": "e_81hd0z", "src": {"node": "n_k3f9a2qd", "port": "sequences"},
     "dst": {"node": "n_p0x7c1mw", "port": "seqs"}, "kind": "data"}
  ]
}
```

- `ref` is `<manifest id>@<version>`, `builtin:<name>` (only `builtin:code.python` in v2), or
  `py:<module:symbol>` (the v1 upgrade scheme, Python executor only, untyped ports, refused by the
  hub). A *wire* node (`NodeDoc`) has exactly one of `ref` or the builtin payload. A live
  `HostPrepGraphNode` has a `callable_ref` (a live object, `Callable | None`; positional field
  order is kept so existing constructors keep working) or a non-`py:` `ref`. The live-node
  change and the audit of its use sites are S4-44; the wire dataclasses are S4-10.
- **Serialising a live callable (r1 C11).** `HostPrepGraph -> GraphDoc` maps a live
  `callable_ref` to `ref = "py:<module>:<qualname>"` using the rule v1 already enforces: the
  callable must resolve back to the same object through `module.qualname` (A49,
  `serialize.py:46-75`). Lambdas, closures, `functools.partial` objects and anything else that
  does not round-trip raise `GraphSerializationError` naming the node id, exactly as v1 does today;
  nothing that serialised under v1 stops serialising.
- **Loading `py:` refs.** `GraphDoc` keeps the `py:` string and never imports (the validator and
  the hub must not run code). `GraphDoc -> HostPrepGraph` for the Python executor resolves `py:`
  refs **eagerly**, as v1's `deserialize_node` does (`serialize.py:133`), raising
  `GraphSerializationError` on an unimportable or non-callable target, so `graph-plan`, `loader.py`
  and the other xtrax consumers keep receiving live callables. Nodes whose `ref` is a manifest
  reference or `builtin:` keep `callable_ref = None` and are resolved by whichever executor runs
  them.
- **Writer version.** `dump_graph(graph, path, schema_version=2)` defaults to 2.
  `schema_version=1` is permitted only when every node serialises as a `py:` ref and every edge has
  null ports and default `id`/`kind`; otherwise it raises (older readers cannot represent the
  graph). The reader accepts v1 and v2.
- **v1 upgrade:** a v1 node loads as `ref = "py:<module:symbol>"`; a v1 edge `{src, dst}` loads as
  `{id: "e_v1_<n>", src: {node, port: null}, dst: {node, port: null}, kind: "data"}`. The reader
  accepts v1 and v2, rejects `< 1` and `> 2`, never default-fills a missing version.
- Unknown keys: `x-` prefixed keys are preserved verbatim at node and edge level; any other
  unknown key is an error (P4). `metadata.nl_description` stays required (an xtrax invariant,
  `node_metadata_schema.toml`); S6's example nodes omit `metadata`, so see Q10.
- **Canonical hash.** `graph_sha256 = sha256(JCS(graph))` (RFC 8785), implemented in
  `xtrax_contract.canonical` with a shared vector file; the same function hashes manifests for the
  lock table. Number formatting is the known divergence between naive Python and JS (A40), so the
  gate is the vector file with naive `json.dumps` as the negative control.
- **Validation** is a pure-data function over `GraphDoc` plus a resolver `ref -> manifest`
  (`validate.py`): `E_DUP_ID`, `E_EDGE_ENDPOINT`, `E_UNKNOWN_PORT`, `E_TYPE_MISMATCH` (via
  `compat`), `E_CARDINALITY`, `E_CYCLE`, `E_REQUIRED_UNCONNECTED`, `E_UNKNOWN_REF`,
  `E_PARAM_SCHEMA` (checked with the S4-47 profile engine), plus additive codes
  `E_ALPHABET_MISMATCH`, `E_NODE_METADATA`, `E_SCHEMA_VERSION`, `E_REF_GRAMMAR`
  and warning `W_ENCODING_ADAPTER`. `W_NO_EXECUTOR` and `W_TRANSFER_LARGE` need executor capability
  info and belong to S6's pack, not to this corpus. The code registry
  (`diagnostic_codes.v1.json`) and the conformance corpus (`{document, catalog fixture, expected
  codes}`) ship in the contract (S4-31); S6's TS evaluator must pass the same corpus, with a
  mutation control on the type table. The xtrax `graph-validate` verb (S4-12) calls this function
  and adds the JAX-side check (`extract_schema` consistency, jaxlint). Nodes with no Python
  executor entry are reported "unplannable on the python executor", not as errors.
- **Who owns the graph schema (r2 R2-1).** The contract's committed `graph-ir.v2`, generated by
  `xtrax_contract.emit` from `graphdoc`, is the **only definition of `Node`, `Edge`,
  `NodeMetadata` and the document envelope**. It keeps the `$id` that the a11 emitter uses. The
  a11 `emit_ir_schema` hand-builds `Node` and `Edge` literals and `$defs` for BundleSchema,
  AxisSpec, AxisOverride, AxisBoundary, Fuse, Tap and Sink (A50); those last types are jax-bound
  xtrax types and cannot live in the stdlib-only package. So `emit_ir_schema()` becomes a thin
  **composer** (S4-45): it loads the contract's `graph-ir.v2`, adds the xtrax-side `$defs`
  (built by `schemagen` with the jax handlers, 4.7), and writes no `Node` or `Edge` of its own.
  The result is one document extended, not two formats; a test asserts the composed
  `$defs.Node` and `$defs.Edge` equal the contract's after JCS.
- **The single-source gate follows the schema (r2 R2-1).** At a11 `audit_ir_schema_single_source`
  scans only `src/xtrax` and permits `$schema` only in `inference/ir_schema.py` (A50). A new
  package outside that root would pass mechanically while defeating its intent, so S4-09 extends
  the audit (script and `test_real_src_tree_is_clean`) to also scan
  `packages/xtrax-contract/src` and `packages/xtrax-contract/ts`, with exactly this allow-list:
  `xtrax_contract/emit.py` plus the generated `xtrax_contract/schemas/*.json`; a `$schema` marker
  in any other file, in either tree, fails. The S4-01 ADR amends the fork-13 note: "one emitter"
  now means one emitter module in the contract package, composed by one module in xtrax.
- **Node metadata (r2 R2-1).** `node_metadata_schema.toml` and `load_node_metadata_schema` stay in
  `xtrax.composition` (A50). The stdlib contract enforces only the one rule it can state
  without reading that file: a wire node's `metadata.nl_description` is a non-empty string
  (`required = true`, `min_length = 1` in the toml, A57), as a constant in `graphdoc` with error
  code `E_NODE_METADATA`. The remaining slots (`mathjax_label`, `citations`, `script_usage`) are
  validated xtrax-side by the `graph-validate` verb (S4-12) through `load_node_metadata_schema`,
  and a test in S4-44 asserts the contract's constant equals the toml's `nl_description` slot, so
  the two cannot drift. At the contract level `NodeMetadata` allows the optional slots as free-form
  `x-`-tolerant objects.
- The drift tests (`tests/inference/test_ir_schema.py`) are kept and extended by S4-45.

### 4.7 Schema emission and the browser knob document (scope item c, second half)

**Emitter** (`schemagen.py`, stdlib-only, S4-07). It walks a dataclass or `eqx.Module` with
`dataclasses.fields` (A17) and `typing.get_type_hints` (A18), emits the param profile (4.4), and
**fails loudly** on anything unmappable rather than degrading to `{}`, the discipline
`IRSchemaTypeError` already encodes. It extends the existing mapper with `Literal`, `Mapping`,
`Sequence`, fixed-arity `tuple` (`prefixItems`), `Path` (string with `x-xtrax.path`), nested
dataclasses (`$defs`/`$ref`), and discriminated unions. `Any` is rejected unless the field's
metadata says `opaque`. `xtrax.inference.ir_schema` then delegates its type mapping to this module
(S4-08), so there is one mapper, not two.

*Domain-type hook and tuple mode (r1 C10).* The existing mapper emits every tuple as a homogeneous
array (A4) and imports `jax.ShapeDtypeStruct` at module level (`ir_schema.py:43`), which a
stdlib-only module cannot know. `schemagen` therefore exposes (a) `register_type_handler(predicate,
handler)`, so `ir_schema` registers `jax.ShapeDtypeStruct` and `Callable` (import-path string)
from the xtrax side, and (b) `tuple_mode`: `"homogeneous"` (the v1-compatible behaviour, used by
`ir_schema`) or `"fixed"` (`prefixItems`, the default for the param profile). With
`tuple_mode="homogeneous"` the v1 emitted schema is byte-identical by construction, which is the
S4-08 gate. S4-02 additionally reports whether any IR field is a fixed-arity tuple, by walking the xtrax
a11 sources with `ast` at a pinned ref passed as an argument (no import of xtrax, so it needs no
JAX environment); if one is,
S4-08's gate is relaxed to semantic equivalence plus a changelog entry rather than silently
changing bytes.

*What the S4-02 spike encodes and what it claims (r2 R2-8).* S4-07 cannot be the instrument for
the spike that precedes it, and the a11 `ir_schema` mapper raises on `Literal`, `Mapping` and
`Path` that S4-07 will support. So the spike script carries its own classifier built from the
**S4-07 target table**, written into the sidecar before the run: `mapped` means the type is in
that table, `opaque-flagged` means `Any` or `Callable` with `metadata["xtrax"]["opaque"]`,
anything else is `unmapped`. S4-07's gate then asserts its mapping table equals the sidecar's
table, so the spike and the schemagen cannot disagree about what "mapped" means. The spike runs
against the aminx tree at a recorded commit, **before** S1-07 flips the spec classes, so its
claim is limited to "what the pre-flip `RunSpec` and `*Specification` classes look like" (and
A18 stays UNVERIFIED for the unified classes). S4-32 re-runs the same script, as a new bathos
run, on the unified classes after S1-07; zero unmapped exposed fields there is part of S4-32's
gate.

**The field-metadata hook (the contract with S1).** Only fields carrying
`metadata={"xtrax": {...}}` are emitted (opt-in, A16): RunSpec has array-valued `Any` fields
(A15) that must never leak into a form. `xtrax_contract.meta(...)` builds the dict, used as
`eqx.field(metadata=meta(...))` (any `static=True` in an example is illustrative only: each
field keeps its own static classification, S1 A28; the hook adds metadata and never changes
`static`); its keys are `expose` (bool), `browser_key`
(the key a browser request carries), `title`, `description`, `unit`, `group`, `advanced`,
`widget`, `minimum`, `maximum`, `examples`, `opaque`. S1 generates this dict from the registry
row (`portable` becomes `expose`, `browser_key` is copied), so S4's schemagen "sees the same
facts as the JSON snapshot" (S1 D4). **Shape, published before S1-06 consumes it (r3 C10).**
S4-07 publishes `FieldMeta` as a documented dict shape: a committed `field_meta.v1.json` listing
exactly the twelve keys above with their types, which `meta()` and schemagen both load.
schemagen reads only `metadata["xtrax"]`; any other metadata key (including a
`metadata={"aminx_field": ...}` style key) is invisible to it, and an unknown key inside
`metadata["xtrax"]` fails emission. S1-06 therefore emits a plain dict of that shape under the
`"xtrax"` key (no import of `xtrax_contract`, hence no S4-34 edge) and depends on S4-07 for the
shape (section 6). S4-32 is the guard against the two sources disagreeing: it validates exactly
that shape for every exposed field and asserts the schema emitted from the classes has exactly
the properties the committed `run_spec_fields.json` marks portable, keyed by `browser_key`.

**Who owns what for the browser (resolves the REFUTED A14).** The letter-keyed knob document the
browser reads has no Python class and a script-only Python twin (3.2). S1 decided S4-20 owns it
(D10, Q6), as a projection of the surviving RunSpec so D4's "one spec system" holds:

0. **S4-33 settles the semantics first** (a short decision/semantics note, a user decision): the
   `chains_to_design` versus `chain_id` mapping (Q2) and the constraint-lowering conventions, so the
   pre-registered controls of S4-41 can state expected outputs.
1. **S4-40 builds the knob document and the pure lowering in library code**: a typed class
   with field metadata (letter-keyed `bias_AA`, `omit_AA`, `tied_positions`, `fixed_positions`,
   `chains_to_design`, `decoding_order`, `seed`, `temperature`, and the rest of the ten browser
   keys), and the lowering to `bias`, `fixed_mask`, `fixed_tokens`, `tie_group_map` that today
   exists only as `scripts/browser_validation/p07_knobs_gate.py:297` `build_p07_inputs`. It
   becomes the source of truth in place of that script twin. **S4-41** produces the golden
   vectors from it and cross-checks the JS kernels (pre-registered); **S4-42** retires the script
   twin after the shared cases agree. The `chains_to_design` (browser, by chain letter) versus
   `chain_id` (Python) difference is settled by S4-33 (Q2).
2. **S4-20 generates from it**: `knobs.schema.json`, `knobs.d.ts`, and `knobs.generated.mjs`
   (constants such as `MPNN_ALPHABET` from the alphex `MPNN_X_21` symbols and `OMIT_BIAS`,
   defaults, and a validator and normaliser for the param profile). It calls the S4-09 generator
   entry point and binds the S4-47 JS profile engine; it adds no generator or validator engine of
   its own (r2 R2-7). `runspec_core.mjs` imports
   the generated module for constants, defaults and validation and keeps its numeric kernels
   (PRNG, shuffles, Gumbel, tie groups, bias building) hand-written, gated by the S4-41 golden
   vectors. S4-20 then **deletes S1-18's key-set drift test**, replacing it with a
   generated-vs-hand equivalence check, with planted-wrong-default and planted-shuffle
   off-by-one negative controls.

If the user does not confirm S4 as the owner (S1 Q6 alternative), the letter-keyed knobs stay
hand-written, S4-33, S4-40 to S4-42 and S4-20 shrink to the keys the registry already carries, and S1 builds the
`DesignConstraints` sub-config instead.

### 4.8 Discovery and the registry (scope item j)

- **Group:** `xtrax.models`, named for the contract owner, not the hub (the contract is hub
  independent).
- **Entry-point name is the manifest file stem; the value is the bare top-level package name of
  the providing distribution.** Why it is not `module:attr`: that forces `import <pkg>`, which runs
  the package `__init__` (jax, weights tables) and is exactly the runtime inversion alphex cut its
  registry over (A10). Why it is not a path: the first draft read `ep.dist.locate_file(<path>)`, but
  `PathDistribution.locate_file` resolves against the dist-info's parent (site-packages), so an
  editable install, where package data lives in `src/`, would not resolve (A42, REFUTED); and
  setuptools, which builds the MPNN repo, rejects a path-valued entry point at build time because
  values must match `importable.module[:attr]` (A43, REFUTED). The bare module form passes that
  grammar.
  The registry resolves a manifest in three steps: (1) `importlib.util.find_spec(value)` with a
  **dotless** `value`, which locates the package without importing it or running its `__init__`
  (A44; a dotted value is rejected because `find_spec` would import the parent); (2) take
  `spec.submodule_search_locations[0]`; (3) read `<that dir>/manifests/<ep.name>.json` with stdlib
  `json`. Discovering a model therefore executes no model code. One distribution registers one
  entry point per manifest (one manifest is one node type, 4.3). Zip and egg installs, where
  `submodule_search_locations` is not a real directory, are reported as `DiscoveryError`.
  ```text
  # in the model distribution's pyproject.toml (illustrative, not a backlog block)
  [project.entry-points."xtrax.models"]
  proteinmpnn-sample = "molxmpnn"     # file: molxmpnn/manifests/proteinmpnn-sample.json
  proteinmpnn-score  = "molxmpnn"     # stems derive from ids by the S4-01 grammar (4.3); "molxmpnn" is S3's root name
  ```
  Whether `find_spec` resolves the package directory under every editable mode is A45
  (UNVERIFIED, deferred to the S4-11 fixtures and the real editable dev installs in S4-21/S4-23).
  **Pre-stated fallback if A45 fails:** for an editable dist read PEP 610 `direct_url.json`
  (`dist.read_text("direct_url.json")`, `dir_info.editable`) to get the source root and join
  `<package>/manifests/<ep.name>.json`; for a wheel install scan `dist.files` for that path and
  resolve it with `locate_file`. The fallback keeps data-only discovery; a `module:attr` value is
  not an acceptable fallback because it reintroduces the D5 inversion.
- **Lazy and isolated**, following the rule xtrax already wrote down (`cli/registry.py:16-22`):
  the scan runs inside `discover_models()`, never at import; a malformed manifest from one
  distribution yields a recorded `DiscoveryError` for that entry and does not stop the rest.
- **Collision guard:** two entries claiming the same `id@version` with different manifest hashes
  fail registration (alphex's symbols-collision guard, applied to ids).
- `NodeTypeRegistry` indexes manifests by `ref`. The hub does not use entry points: it builds a
  static catalog from manifests collected in CI (S5), because the browser has no installed
  distributions (the second alphex reason, A10).

### 4.9 Protocols and the proteinsmc partition (scope item e, D6)

`protocols.py` defines structural `typing.Protocol`s over port types. Arrays are typed as a
minimal `SupportsShapeDtype` protocol so the module imports no array library.

```python
StructureInput = str | os.PathLike[str] | PdbText     # stdlib-only forms, see conventions below

class PdbText(TypedDict):
    pdb_text: str                                      # the PDB or mmCIF file contents

class ScoreFn(Protocol):
    # Call shape deliberately identical to proteinsmc FitnessFn (A9).
    def __call__(self, key: Any | None, sequence: SupportsShapeDtype,
                 context: Any | None = None) -> SupportsShapeDtype: ...

class Scorer(Protocol):
    manifest: ModelManifest
    def bind(self, structure: StructureInput, *, options: Mapping[str, Any] | None = None) -> ScoreFn: ...
```

**Python-side conventions (r2 R2-3).** The protocols are typed conveniences over the same
`inputs -> outputs` shape as the port model, so the conventions below are what a consumer that
cannot build a `PortValue` relies on. They are part of `Protocols@1`, tested by `assert_conforms`
on the provider and by the adapter tests on the consumer; S4-21 (provider) and S4-48 (consumer)
both cite this list.

1. **`structure` is a path (`str` or `os.PathLike`) to a PDB or mmCIF file, or a `PdbText`
   dict.** Never backbone arrays and never a `PortValue`: those need a structure parser or the
   contract package, which a D6 consumer has neither of. The provider parses (molxmpnn already
   depends on proxide); the old `inputs: str | Path | Sequence` of `make_mpnn_score` is a subset
   of this form. `assert_conforms` binds both forms on a tiny shared fixture PDB and requires equal
   scores.
2. **`sequence` is an integer token array in the alphabet the manifest names on its input port**
   (for MPNN: `MPNN_X_21`, order given by the port's literal `symbols`, 4.2). The provider never
   guesses an alphabet. **The consumer maps tokens**: proteinsmc already depends on `alphex`
   (A54), so S4-48 builds the index map from its own alphex alphabet to the port's `symbols` by
   symbol, and refuses (raises) an unknown symbol. A test permutes the destination ordering and
   requires the mapped tokens, and therefore the score, to change, so a wrong ordering cannot pass
   silently.
3. **Direction.** The return value is the provider's native score in the unit and direction
   declared on the manifest's `scalar` output port (`direction` in `maximize`/`minimize`). For
   the MPNN scorer that is an NLL, `minimize` (A33). A consumer that maximises (proteinsmc reports
   `max_fitness`, A55) negates a `minimize` score in its adapter and leaves a `maximize` score
   alone; the adapter reads `direction` from the manifest, it does not hard-code the sign. S4-48
   anchors the proteinsmc convention with a `path:line` before writing the adapter and tests it
   with a sign-flip negative control: an adapter that skips the negation must fail the
   ordering test (a lower-NLL sequence must receive a higher fitness).
4. **Fate of `make_mpnn_score` (a user decision, Q9).** Its `mpnn_model_params` argument came from
   a package that no longer exists and its whole branch is guarded on the stale name
   `prxteinmpnn` (A54), so it has no working caller. The recommendation is to keep the name and
   the `"mpnn"` registry key in `utils/fitness.py` but change the signature to
   `make_mpnn_score(structure, decoding_settings="random", *, scorer_id=None)`, forwarding
   `decoding_settings` through `options`; the old three-argument form raises a `TypeError` with
   a message naming the new form. The alternative is to leave the old function and add a new
   registry key. S4-48 is a user decision for this reason.

`Sampler`, `StructurePredictor` and `EnergyModel` follow the same `bind(...)` shape; a manifest's
`role` says which protocol its python executor entry implements. `bind` is the
`make_mpnn_score(params, inputs, decoding_settings)` pattern proteinsmc already used
(`scoring/mpnn.py`), with the structure arrays closed over, which is also how aminx's
`score_sequence` is shaped (it takes `structure_coordinates, mask, residue_index, chain_index`,
`src/aminx/scoring/score.py:117-124`). Sign and units of the returned score are declared in the
manifest's `scalar` output port (`direction`, `units`), not assumed (A33). An xtrax-side
helper (`xtrax.contract.check_ports`, delivered by S4-13, imports jax, so it is not in the stdlib-only package) runs
`extract_schema` (`jax.eval_shape`) over a provider's callable and fails when the declared output
port names or shapes disagree with what the callable actually returns, so a manifest cannot
drift from the code it describes.

**How proteinsmc consumes it with no dependency edge (D6):**

1. proteinsmc scans `importlib.metadata.entry_points(group="xtrax.models")` itself, locates each
   manifest with stdlib `importlib.util.find_spec(ep.value)` plus `manifests/<ep.name>.json` (4.8),
   reads it with stdlib `json`, and filters manifests whose `role == "scorer"` and whose
   ports match what it needs. About 50 lines, no new runtime dependency, no import of
   `xtrax_contract`, `aminx` or `molxmpnn`. **Version discipline (r2 R2-15).** Because this
   reader bypasses the contract package, it carries P4 itself: it accepts exactly
   `schema_version == 1` and a known `role`, and skips, with a logged warning naming the entry
   point, any manifest whose version is missing, `0`, `2` or any other value, or whose `role` is
   absent; it never default-fills. S4-22's gate runs fixtures for versions 0, 1, 2 and missing.
2. Only when the user selects that scorer does proteinsmc call
   `importlib.import_module(factory_module)` from the `python` executor entry, the same late
   string binding as `callable_ref`. This replaces the stale `find_spec("prxteinmpnn")` guard,
   and also the interim guard S3-18 plans (a root-only `find_spec("molxmpnn")` guard in S3's
   version; the earlier `find_spec("molxmpnn.scoring.score")` form is shown here because it is
   the one that demonstrates the problem): a dotted `find_spec` imports the
   parent package `molxmpnn` (A36), so it runs the package `__init__` at proteinsmc import time
   and raises instead of returning `None` if that import fails. That is the alphex D5 inversion
   again (Q9).
3. Conformance is checked on the **provider** side: `xtrax_contract.testing.assert_conforms`
   runs in the MPNN repo's CI. proteinsmc's own tests use a fake scorer written to the
   Protocol by structure. This is alphex's D4 pattern: no coupling edge, and a mechanical check
   against drift.

D6 gate (r1 C18): `rg -n "(import|from) +(aminx|molxmpnn|xtrax_contract)|import_module\(.(aminx|molxmpnn|xtrax_contract)" proteinsmc/src`
is empty, and a TOML check of `proteinsmc/pyproject.toml` finds no `aminx`, `molxmpnn`, `xtrax` or
`xtrax-contract` in `[project.dependencies]` or `[project.optional-dependencies]` (the tables a
published wheel exposes). `[dependency-groups]` is **exempt**: a non-default group is a dev-only
test dependency, not a coupling edge (alphex D4 pattern, A10), so S3-18's Q13 mechanism and
S4-23's own group do not trip the check; a planted `molxmpnn` in `[project.dependencies]` or
`[project.optional-dependencies]` must fail it, a planted one in a `[dependency-groups]` entry
must pass. The only string-based import is the factory string read from the manifest, bound late
(step 2).

**How the provider reaches proteinsmc's test environment (r3 C5).** S4-23 states its own
mechanism and does not rely on S3-18: a non-default `[dependency-groups]` entry (for example
`mpnn-e2e`, not installed by `uv sync` or by CI's default job) used by two runs, a wheel arm
(`uv build` of the molxmpnn checkout, installed from that wheel) and an editable arm (editable
install of the same checkout into a throwaway env). If S4-49 answers "supersede" and S3-18 is closed
unmerged, S4-23 is unaffected.

### 4.10 Shared ONNX route (scope items g and h)

**`xtrax.export.bundle.export_onnx_bundle`** (new, S4-14): takes named graphs
`{name: (callable, abstract_inputs, port_names)}` and a bucket list, calls
`convert_to_onnx` for each, then records the OnnxBundle@1 artifact entries (4.5): sha256, bytes,
graph input/output names read from the ModelProto, dtype census, RNG audit via
`find_onnx_rng_ops`, and a *self-contained* assertion (no external-data tensors).

*A22 is probably false for Loop subgraphs, so it is spiked first (r2 R2-5).* aminx's
`embed_external_data` exists because jax2onnx writes Loop-subgraph constants as external data
that ORT-Web cannot load (A21). A toy two-graph test in S4-14 would not reproduce that, and the
first real export would be the L-sized S4-17. So **S4-43** (pre-registered, on titanix) runs
`convert_to_onnx` on the four real MPNN graphs at one bucket and counts external-data tensors in
the in-memory `ModelProto`, with a planted external-data fixture as the negative control that must
make the assertion and the sidecar outcome fail. S4-14 depends on that record. **S4-46** is the
conditional fix: if the record shows any external-data tensor, S4-46 ports the embed step into
`xtrax.export.bundle` (and S4-14 uses it); if the record shows none, S4-46 is closed by citing
the record and S4-14 keeps only the assertion. Either way the exporter is correct for the real
graphs before S4-17 starts.

*Preemption-safe by construction.* The unit of work is one graph at one bucket. Each unit is
written atomically (per-writer temp dir, then rename) with a completion stamp holding the
sha256 of its inputs (weights file hash, git SHA of the callable's package, xtrax / jax /
jax2onnx / onnxruntime versions, opset, bucket). A rerun reuses a unit only if the stamp
matches and the artifact hash re-verifies; otherwise it recomputes. The result records which
units were reused, from where, with hashes. Each unit runs in its own process with its own
timeout; there is no whole-run timeout. Because jax2onnx leaves global patches (and A13's
guard only restores namespaces), JAX reference arms are evaluated *before* the first
conversion, per the existing rule in the aminx scripts.

*Cache and output locations are configuration, not constants.* Resolution order for the
resume/cache root: explicit `--cache-dir` / argument, then env `XTRAX_EXPORT_CACHE` (`none` or
empty disables), then `[tool.xtrax.export] cache_dir` in the nearest `pyproject.toml`, then
`${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml` key `export.cache_dir`, else off. The library
has no default path, exposes `export_cache_source()` reporting which layer decided, fails
loudly on a malformed config file, never writes inside another project's data directory, and
keys entries by the real source identity (resolved weights path plus content hash), never a
staging name. Reference implementation named in the user's rules: `isochore.traj_cache`.
The output directory is always an explicit argument.

**Adoption in the MPNN repo** (S4-17, then the parity cells S4-18, S4-38, S4-39): replace
`_convert` in `p07_split_export.py` **only** with the bundle exporter (the eight other scripts that
reference `to_onnx` are out of scope for this DAG and stay on direct jax2onnx until their own gates
are next re-run under their own items, r1 C9);
delete the private `onnx_audit.find_onnx_rng_ops` in favour of xtrax's, keeping the planted
controls. Artifact bytes will change (opset pinned at 23, model name `xtrax_export`, no
external-data rewrite), so the sha256 manifest is **re-baselined**, and equivalence is argued by
the *parity chain*, not by byte identity. The chain is re-run, not assumed, one executor per item
and each under its own pre-registered sidecar: ORT-CPU token-exact and within the inherited 2e-4
nats (A24; S4-18), Node/wasm (S4-38), headless Chromium (S4-39, the record S5-15 compares
against). Bounds are inherited unchanged; if the xtrax route needs a looser bound,
that is a finding recorded as such.

**The orbax conflict (scope item h).** Measured by the brief, not re-run (A27): adding
`xtrax[onnx]` to the MPNN lock moves base `orbax-checkpoint` 0.12.0 to 0.11.36 through
jax2onnx to orbax-export. Options:

| | Option | Pros | Cons |
|---|---|---|---|
| A | consumer-side `[tool.uv] override-dependencies = ["orbax-checkpoint>=0.11.17"]` (restating only the core floor, exactly as xtrax does, `pyproject.toml:137-149`) and the toolchain in a **dependency group**, not a published extra | in-repo precedent with CI evidence in xtrax; small | lock-only: a PyPI install of any `[onnx]` extra still hits the cap (xtrax says so itself, A26), so no such extra may be published; relies on orbax 0.12 + jax2onnx 0.17 compatibility, which xtrax reports testing |
| B | uv `conflicts` to fork the lock | isolates the base | needs a second conflicting extra/group to express; most invasive |
| D | isolated export environment (script env outside the lock) | base lock untouched | env drift; the sidecar must record versions of an environment the lock does not pin |

**Recommendation: A, plus move `xtrax[export]` (IREE compiler, runtime, hf hub) out of the base
dependencies** (`pyproject.toml:26` has it in base today) into an export group, with D as the
fallback if S4-03's measurement shows A breaks the ONNX scripts. Never publish an `onnx` extra
for the MPNN package. Decision item: Q3 and S4-16.

### 4.11 Compatibility and migration

| Surface | Change | Compat |
|---|---|---|
| `HostPrepGraphNode` | `callable_ref` becomes optional, adds `ref`, `params`, builtin-node payload | positional order kept; v1 docs load via upgrade rule |
| `GraphEdge` | optional `src_port`, `dst_port`, `id`, `kind` | v1 edges upgrade with null ports |
| `GRAPH_SCHEMA_VERSION` | 1 to 2 | reader accepts 1 and 2, writer emits 2 by default (`schema_version=1` only when representable, 4.6); the 17 xtrax source files that mention `callable_ref` or `HostPrepGraph` at a11 (A51: `cli/graph_plan_verb.py`, `cli/graph_verb.py`, `cli/loader.py`, `cli/plan.py`, `composition/__init__.py`, `composition/author.py`, `composition/errors.py`, `composition/graph.py`, `composition/node_metadata.py`, `composition/serialize.py`, `composition/validate.py`, `inference/ir_schema.py`, `loop/__init__.py`, `loop/admission.py`, `loop/diversity_quota.py`, `run/component_binding.py`, `stages/evaluate.py`) are audited by S4-44: its gate lists every use site and a `None` test for each, and re-greps the tree so a new hit fails the audit |
| `xtrax.run.RunSpec` | untouched | S4-10 pins field names and identity `from_spec` with a regression test (S1's ask) |
| `emit_ir_schema` | becomes a composer of the contract's `graph-ir.v2` plus xtrax-side `$defs` (S4-45) | same `$id`, new version, drift tests extended; single-source audit extended to the contract package (S4-09) |
| xtrax root `pyproject.toml` | uv workspace member, `xtrax-contract` dependency and workspace source (S4-04) | released range pinned by S4-34 |
| `xtrax[onnx]` | unchanged | MPNN repo consumes via group + override (Q3) |
| `scripts/browser_validation/p07_split_export.py` | moves behind the bundle exporter (S4-17); the other eight `to_onnx` sites are unchanged | sidecars re-registered before re-run; old run ids stay as history |
| `runspec_core.mjs` | imports generated constants/defaults/validator | kernels untouched; S4-41 golden vectors gate |
| Distribution/import names | manifest strings (`aminx.contract...`) must follow S3's rename | S4-19 depends on S3-07 and S3-11; every `module:symbol` in a manifest is resolved by a test, so a missed rename fails loudly |
| Naming | `xtrax.models` entry-point group | independent of the hub name |
| Repo naming in this backlog | `repo = "aminx"` for every MPNN-tree item with no S3-07 ancestor: the spikes S4-02, S4-03, S4-43 (which take the package name as an argument) and the two decision notes S4-33 and S4-37, which are dated docs that S3-07 carries across the rename (r2 R2-9); every other MPNN-tree item says `repo = "molxmpnn"` and is ordered after S3-07 by `depends_on` | one rule for pre- and post-rename items, matching S1/S3 |

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A second graph format appears (fork 13's fear): `GraphDoc` vs `HostPrepGraph` | one wire definition, `HostPrepGraph` converts to/from it, round-trip property test, `emit_ir_schema` derived from `graphdoc`, drift tests |
| Python and TS type engines drift (S6's "most important anti-drift control") | the table is data, one corpus, a mutation control (flip one table entry, corpus must fail), every code produced by at least one vector |
| Contract becomes a dumping ground | closed executor kinds and port vocabulary, `x-` / namespace escape hatch only, profile subset enforced at emission |
| Two release trains (xtrax, xtrax-contract) skew | `xtrax` pins `xtrax-contract>=X,<next minor`; one release item (S4-34) publishes both together; consumers pin in S4-35/S4-36; CI job installs both from the repo; Q1 offers F1 as the lighter alternative |
| Workspace wiring made up differently by two items (r2 R2-2) | S4-04 alone owns workspace membership, the sub-dist backend and version source, and xtrax's dependency plus workspace source; its gate is `uv lock --check` and an import from the workspace; the tag policy is in the S4-01 ADR |
| The single-source audit passes while its intent is broken, or two emitters disagree on Node/Edge (r2 R2-1) | the contract's `graph-ir.v2` is the only Node/Edge definition, `emit_ir_schema` composes it (S4-45), the audit scans both trees with a two-entry allow-list (S4-09), and a test equates the composed and contract `$defs` |
| Hidden toolchain: TS generator and validator engine (r2 R2-7) | generation is stdlib Python, Node is check-only, one hand-written engine per language for the fixed profile subset, both run against one corpus (S4-47); decided in the S4-01 ADR |
| proteinsmc adapter silently inverts the optimisation or misorders tokens (r2 R2-3) | direction read from the manifest and tested with a sign-flip control; alphabet mapped by symbol with a permuted-ordering control (S4-48) |
| No installable contract dist for consumers (r1 C3) | S4-34 is the only release item and every consumer item that imports `xtrax_contract` depends on a pin item (S4-35, S4-36); channel is Q11 |
| Manifest-as-data discovery is unconventional and the first design (path value, `locate_file`) was refuted (A42, A43) | entry value is a bare top-level package name located by `find_spec` without import (A44); wheel and editable installs both tested (S4-11, S4-21, S4-23); pre-stated data-only fallback via `direct_url.json`/`dist.files` (4.8) |
| jax2onnx 0.17 behaves differently from the 0.16.1 the aminx gates were first validated on (xtrax records the patch happening after the first call, not at import; the aminx memory note says at import) | the pre-registered gates re-run the controls (planted perturbation, RNG plants); JAX references evaluated before the first conversion |
| xtrax route is not byte-identical, so "same artifact" cannot be claimed | claim equivalence only through the parity chain with inherited bounds (A24, A25); sha256 manifest re-baselined and labelled as such |
| Self-contained assumption (A22) fails, ORT-Web cannot load external data | spike S4-43 settles it on the real graphs before S4-14 and S4-17; conditional item S4-46 ports `embed_external_data` into xtrax; the exporter asserts it per graph regardless, with a planted-external-data negative control (S4-14, S4-17, S4-43) |
| Schema emitter leaks array-valued `Any` fields into forms | opt-in exposure (`metadata={"xtrax": ...}` with `expose`); `Any` rejected unless `opaque`; S4-32 checks the exposed set equals the registry snapshot's portable rows |
| Knob document turns into a third spec system (D4) | it is a projection of the surviving RunSpec built on S1's field metadata, built by S4-40 per S1 D10; no second authored source of field facts |
| Generated knob schema and the hand-written JS kernels disagree | S4-41's golden vectors from the Python lowering are the arbiter; planted-default and planted-shuffle controls in S4-20 |
| JCS number formatting diverges between Python and JS (A40) | shared vector file owned solely by S4-30 (S6-26 superseded), naive `json.dumps` is the negative control, S6-35 runs the TS side |
| Manifest strings break on S3's rename or S2's extraction | S4-19 depends on S3-07 and S3-11; test resolving every executor `module:symbol`; S3's stale-name guard sweep should include `rg "aminx\.contract"` |
| Weights license / redistribution mistakes surface in a public hub | `license` and `redistribution` are required manifest fields, hub CI refuses a manifest without them (Q8); one decision (S4-37) with one reviewer and one record in the aminx/molxmpnn tree, consumed by S5-07 and S5-06 so the hub cannot record a second, different answer (r3 C11) |
| Two owners of the xtrax 0.4.0a11 pin disagree on ordering (r3 C7) | S2-02 owns the merge of draft PR #174; S4-16 is only the Q3 resolution on top of it and depends on S2-02, S3-07 and S4-03 |
| Manifest completeness circular between S4-19 and S5-14 (r3 C8) | source mode permits only `driver.url` placeholder and `driver.sha256` pending; S5-14 fills the hash, S5-52 rewrites placeholders and validates in release mode |
| Hub-requested amendment G1-G4 creates two meanings of `v1` (r3 C9) | additive within `v1` before the first release; S4-34 depends on S5-51; a later amendment is a version bump |
| molxmpnn version sequence unowned (r3 C14) | S4-19 resolves `version` and every `module:symbol` against the S3 release-train table (S3 4.9, the owner of the molxmpnn version sequence) entry of the tag the manifests are released from; no S4 item publishes molxmpnn itself |
| Remote executor exfiltrates user structures | `remote` is never auto-selected; `requires_network` shown; consent hook is S5's (`261001_aminx-hub.md:338-350` area) |
| AF3 cannot be bound as a library today (A29) | contract reserves the `npm` shape only; S4-27 is retired and the localfold upstream request, vendored job-json and adapter belong to S5 (S5-25, S5-23, S5-26; Beerware vendoring consent is S5-26's) |
| Rebase conflicts with S3 (rename) and S1 on `pyproject.toml`, `scripts/browser_validation` | S4 aminx-repo items avoid hard-coded import paths in generated artifacts; the manifest resolve test catches misses; every MPNN-tree item is ordered behind S3-07 by `depends_on` (S4-16, S4-35 directly, the rest through S4-35 or S4-16); S4-16 also depends on S3-13, because it changes base dependencies and `uv.lock` and would otherwise be able to merge between S3-07 and S3-12 and break the S3-13 pinned-tag diff (S3 4.9), so S4-35, S4-17..S4-21, S4-32 and S4-40..S4-42 follow the release, S4-20/S4-40 also behind S1 items; spikes S4-02/S4-03 only add new files |
| S4-16 and S4-35 change the MPNN lock (base `orbax-checkpoint`, `xtrax[export]` out of base, xtrax bump) after the S2-02 merge, and nothing orders them against S2-05..S2-34 (S2 A0/B1 compare environment stamps; S1-17 may also touch the lock) | intentionally unordered, no dependency edge is added: it is safe only because S2 runs A0 from a pinned F' checkout, not from the moving MPNN tree. The S2 environment stamp should add `orbax-checkpoint` (EBM restore uses it) so any lock difference between A0 and B1 is visible in the record |
| Sibling drafts change under S4 (S5 and S1 already cite S4 ids) | ids frozen, new work appended (A41); the token map in section 6 is the interface ledger the coherence pass reconciles |
| Local-machine memory limits | contract package tests are stdlib-only and may run locally; every JAX/ONNX gate runs on titanix or CI, never a whole suite locally |

## 6. Interfaces

### Provides (named contracts)

| Contract | What | Items | Consumed by |
|---|---|---|---|
| `PortTypes@1` (token `S4:port-types`) | declarative port-type table, `compat` evaluator, port-value wire format, diagnostic code registry, conformance corpus, JCS hash vectors (S4 is sole owner; supersedes S6-25/S6-26) | S4-04, S4-30, S4-31 | S5, S6 |
| `ModelManifest@1` (token `S4:model-manifest`) | HUB-REQ 1-10 manifest dataclasses, generated JSON Schema, TS types, `catalog_view`; the native MPNN manifests (ids and entry-point stems follow the S4-01 grammar, 4.3); source-mode manifests may carry a `release-asset:` placeholder and a pending driver hash; `catalog_view` snapshot corpus (S4-09); the amended schemas once S5-51 lands (inbound) | S4-01, S4-05, S4-06, S4-09, S4-19 (S5-51 amends; S4-34 releases) | S5 (catalog builder), S6, S2, S3 |
| `WeightsLicenseDecision@1` | per shipped weight set: license text, `redistribution` value, one named reviewer, one record in the aminx/molxmpnn tree | S4-37 | S4-19, S5-07 (cites it from `catalog/licenses/<model>.review.toml`), S5-06 |
| `ExecutorEntry@1` | closed executor kinds, artifact entries with per-format details, driver reference (ABI is S5's) | S4-05 | S5 (executors), S2 |
| `GraphIR@2` (token `S4:graph-ir`) | v2 document, v1 upgrade rule, `ref` grammar, canonical hash, pure-data validator, `graph-validate` verb | S4-10, S4-44, S4-45, S4-12, S4-30, S4-31 | S5 (runner), S6 (editor read/write) |
| `ParamProfile@1` | param-schema keyword subset, `x-xtrax` hints, the `FieldMeta` metadata hook (`metadata["xtrax"]`, twelve keys published as `field_meta.v1.json` by S4-07 before S1-06 consumes it), schema emitter, the MPNN knob document with its pure lowering and golden vectors, generated knob schema, TS types, defaults, validator | S4-07, S4-09, S4-47, S4-20, S4-32, S4-33, S4-40, S4-41, S4-42 | S6 (form renderer), S5, S1 (emits `FieldMeta` from its registry) |
| `Protocols@1` | `Scorer`, `Sampler`, `StructurePredictor`, `EnergyModel`, `assert_conforms`, `check_ports`, the Python-side conventions of 4.9 (structure form, token alphabet, direction) | S4-13, S4-21, S4-48 | proteinsmc (structurally, no import), S2 |
| `ModelDiscovery@1` | `xtrax.models` group semantics, manifest-as-data entry value (bare package name, file `manifests/<name>.json`) | S4-11, S4-21 | S5 (local dev mode), proteinsmc, S2, S3 |
| `OnnxBundle@1` (token `S4:onnx-route`) | manifest-fragment artifacts and `export_onnx_bundle`; molxmpnn's parity evidence (ORT-CPU, Node wasm, Chromium) | S4-43, S4-46, S4-14, S4-17, S4-18, S4-38, S4-39 | S5 (S5-14, S5-15), S3 |
| `ContractArtifacts@1` | committed schemas (`model-manifest.v1`, `port-types.v1`, `param-profile.v1`, `graph-ir.v2`), TS package, versioning policy, the released `xtrax-contract` and `xtrax` dists | S4-04, S4-09, S4-28, S4-34 | S5, S6, S1 |
| `XtraxRunSpecBase` stability | `seed, axes, carry_specs, boundaries, run_id`, identity `from_spec` unchanged | S4-10 | S1 |

### Consumes

| Contract | Provided by | Used for |
|---|---|---|
| Surviving unified RunSpec with registry-derived field metadata (S1 P1, items S1-05, S1-06, S1-07) | S1 | S4-40 builds the knob document on it, S4-32 cross-checks metadata against the snapshot |
| `run_spec_fields.json` derived snapshot (S1 P2) | S1 | S4-32 agreement test only (the snapshot is not an authored source) |
| Key-set drift test (S1-18) | S1 | deleted by S4-20 once the generated schema replaces the hand copy |
| `S3:molxmpnn-repo` (repo, release URLs, HF weights repo and final revision, PyPI names) and `S3:molxmpnn-manifest-namespace` (the root name `molxmpnn` and the version source only) | S3 (items S3-07 rename, S3-11 weights cutover, S3-13 first versioned release; release-train table S3 4.9) | the namespace for ids and `python` factory strings, `hf://` revision, and the `version` S4-19 records (S4-19). S3 defines no manifest strings: the id/entry-point grammar and manifests directory are S4-01's (4.3) |
| Interim proteinsmc guard (S3-18) | S3 | the backlog does not decide whether S4-22 supersedes it: decision item S4-49 records the Q9 answer and S4-22 depends on it, so exactly one guard implementation lands (r2 R2-4). S4-22 gains a dependency on S3-18 only when S4-49 answers "sequence" (the S4-49 record adds the edge); S4-23 does not depend on S3-18 either way (4.9) |
| Hub-requested contract amendment S5-51 (G1-G4) | S5 | accepted as an inbound amendment before the first release (4.3); S4-34 depends on it; S4-09 commits the `catalog_view` snapshot corpus (S4's own deliverable) |
| Release-train table (molxmpnn version sequence) | S3 | S4-19 resolves its `version` and `module:symbol` strings in the tagged tree it is released from; S4 publishes no molxmpnn version |
| `EbmProjectIdentity` and EBM descriptor facts (`ebm-model-facts`) | S2 | second registrant of `xtrax.models`, later adoption item the DAG integrator adds (no blocking item here) |
| Driver module ABI `createSession(ctx)` and `Executor` kinds | S5 | `executors[].driver` shape and the kind mapping in 4.5; no item dependency |
| UI hint vocabulary and S6's evaluator | S6 | extends `x-xtrax` keys, TS evaluator passes the S4-31 corpus; no item dependency |

**Token map for the coherence pass** (S5 and S6 use tokens in `depends_on`): `S4:model-manifest`
is satisfied by S4-06 (schema and dataclasses) and, for committed fixtures, S4-09; the native MPNN
manifests are S4-19. `S4:port-types` is S4-04 (table and `compat`) plus S4-31 (corpus and codes).
`S4:graph-ir` is S4-10 (wire format) plus S4-44 (live graph) and S4-30 (canonical hash); the composed xtrax schema is S4-45. `S4:onnx-route` is S4-14, with the molxmpnn
evidence in S4-17, S4-18, S4-38 and S4-39 (one per executor).

### Edges owed by other specs (coherence-pass actions, r1 C4, C14)

S4 cannot edit another spec's `depends_on`; these are the changes it needs recorded there.

| Owed by | Change | Why |
|---|---|---|
| S1 | S1-06 `depends_on` gains `S4-07` (the published `FieldMeta` key list, `field_meta.v1.json`), and S1-06 emits a plain dict of exactly that shape under `metadata["xtrax"]` with no import, so it needs no S4-34 edge; S1's `metadata={"aminx_field": ...}`-style key is replaced or supplemented by it, because schemagen reads nothing else (4.7). S4-32 validates the shape | S1-06 consumes the metadata hook S4 publishes; without the key, S4-32 and S4-40 fail on S1's output. S1 declined this edge in its D4 (S4-32 validates the dict); the row stays owed because without it S1-06 and S1-31 assert the dict shape against prose and a drifted key list is caught only at S4-32 after S1-07 has flipped. The edge is acyclic: S4-07 depends only on S4-01 and S4-02, and S4-02 is the pre-flip spike, so with the edge S4-02 precedes S1-07 by construction |
| S5, S6 | Use `manifest_sha256 = sha256(JCS(manifest))` (4.3) for the remote echo (S5-19), the RunStore unit key and S6 spec keys; give the Node builder (S5-04) and the editor (S6-33) the JS canonicaliser (S6-35) by `depends_on`, since they have no edge to it today (S5-04 and S6-33 depend on S4-06 only); both pass the manifest-hash vector in S4-30 | one hash definition across three specs; the release-asset byte hash stays a separate value |
| S6 | Remove S6-25 and S6-26 (superseded: S4-10 carries `builtin:`, S4-06 carries port `required`/`cardinality`, S4-30 the JCS vectors, S4-31 the corpus and codes). Re-point every S6 item that depends on them (S6-03, S6-04, S6-16 and any other) to S4-06, S4-10, S4-30, S4-31; S6's rows A24, A34, A36 are stale | one owner for one deliverable in one repo |
| S3 | S3-18 gains `depends_on` on S4-49 (the Q9 decision) and is closed unmerged if S4-49 answers "supersede"; otherwise S4-22 lands after it (S4-49's record adds that S3-18 edge to S4-22). S3's Consumes drops "no contract is consumed from S4" and lists the S4-49 decision. Refresh S3-20's leaf set from this graph (or let S3-28 compute it). Here "leaf" means an S4 item with `repo` molxmpnn or aminx that no other molxmpnn or aminx item depends on; for S4 that set is **S4-20, S4-21, S4-42** (S4-18 is a dependency of S4-19, S4-32 of S4-40, S4-38 and S4-39 of S4-19, S4-48 is a proteinsmc item; S4-33 and S4-37 sit in the aminx tree and are carried by S3-07). Also: S3-05's environment stamp includes the xtrax version and S3-07 refuses to compare across a mismatch (S3 A41), so S3-05 and S3-07 should depend on S2-02 (merged #174) to keep the xtrax version fixed across the capture and the rename | S4-22 replaces the guard S3-18 would write; the S3-20 quiescent window must follow every S4 item that edits the tree |
| S5 | Re-point the ORT-Web comparison (the draft's S5-15 `depends_on` and its prose naming "the S4-18 Node-wasm/Chromium record") to S4-38 (Node wasm) and S4-39 (Chromium); S4-18 is the ORT-CPU cell only. `S4:onnx-route` also means S4-38 and S4-39 (and S4-43 and S4-46 for the exporter's correctness); `S4:model-manifest` fixtures also need S4-09. **Manifest completion (r3 C8):** S5-14 fills `driver.sha256` in the source manifest; S5-52 rewrites `release-asset:` placeholders, fills `driver.url`/`driver.sha256`, and its gate validates in release mode (non-source). **Inbound amendment:** S5-51 stays in the xtrax repo, gains no new S4 dependency beyond S4-06/S4-09, and S5-24, S5-42, S5-45 (through S5-64) pin the S4-34 release (which carries it), not the S4-09 commit; S5-04 deliberately builds against the S4-09 commit and S5-64 re-pins it (4.3). **Weights decision:** S5-07 depends on S4-37 and its `catalog/licenses/<model>.review.toml` cites the S4-37 record (same reviewer, same redistribution value); S5-06 reads that file. **Localfold:** S5-25, S5-23, S5-26 own the AF3 work (S4-27 retired) | evidence per executor; artifact hosting; one answer to the licensing question |

## 7. Verification gates per item

Where things run: contract-package tests are stdlib-only (narrow local runs are fine);
anything importing JAX or onnxruntime runs on titanix (rsync the worktree, `ssh`) or CI, never
as a whole suite locally.

**Pre-registration rule for every item that produces a number or pass/fail finding** (S4-02,
S4-03, S4-43, S4-17 (re-baselined manifest), S4-18, S4-38, S4-39, S4-20 (generated-vs-hand
equivalence), S4-25, S4-41 (golden vectors)): a
tracked script plus a `.bth.toml` sidecar stating the hypothesis and `[outcomes]` is **committed
before the run**; run with
`bth run --project-slug <slug> -- uv run --no-sync python3 scripts/<x>.py ...` (never
`uv run bth`); verify by record, after `bth compact`:
`bth sql "SELECT id, status, outcome, exit_code, command FROM runs WHERE id LIKE '<prefix>%'"`,
and spot-check that claimed paths resolve. Each such item also states a **negative control that
must fail** (a positive control that can only pass is not a check). Deterministic fixtures
(the conformance corpus, JCS vectors) are test data, not findings, but each still carries a
mutation control.

| Item | Gate (the full command / observable) | Controls and resumability |
|---|---|---|
| S4-01 | ADR file exists in xtrax `.praxia/docs/decisions/`, records Q1 and Q4 (discovery shipped now), names, floor, the version and tag policy for the two dists (4.1), the generation toolchain (stdlib generator, Node check-only, one engine per language, 4.1), the manifest id grammar `<namespace>/<family>.<op>`, the entry-point stem rule and the `manifests/` directory (4.3; examples `molxmpnn/proteinmpnn.sample`, `molxmpnn/proteinmpnn.score`), and amends the fork-13 note (one emitter module in the contract package, composed by one in xtrax, 4.6) | n/a |
| S4-02 | bathos record with an evaluated `[outcomes]`: every field of every real spec class (aminx tree at a recorded commit, pre-S1-07) is classified `mapped`, `opaque-flagged` or `unmapped` against the **S4-07 target table written in the sidecar**, the unmapped names are recorded, and any fixed-arity tuple field of the xtrax IR is found by `ast` over the a11 sources at a pinned ref (no xtrax import); outcome is `pass` only if both controls behave. The claim is limited to pre-flip classes (A18 stays UNVERIFIED for the unified ones; S4-32 re-runs it) | positive: a toy dataclass maps fully; negative: a field annotated `Callable` without a ref flag is reported as unmapped. One process per class, per-class result persisted |
| S4-03 | bathos record: resolved `orbax-checkpoint` per arm for the base install, plus onnx script import smoke per arm | negative control: the unmodified `xtrax[onnx]` arm must reproduce the known downgrade or the instrument is distrusted. One arm per process, per-arm result persisted |
| S4-04 | on CI or titanix: `cd packages/xtrax-contract && uv run pytest -q tests/test_ports.py tests/test_wire.py` and `uv run lint-imports` (xtrax_contract forbidden from jax, numpy, equinox, xtrax); at the xtrax root `uv lock --check` exits 0 with the workspace member, the `xtrax-contract` dependency and the workspace source in place, and `uv run python -c "import xtrax_contract, xtrax; print(xtrax_contract.__file__)"` prints a path inside `packages/xtrax-contract` (the workspace copy, not a PyPI one); a build of the sub-dist yields a wheel with `requires-python >=3.11` and no `Requires-Dist` | wire test round-trips through the real `safetensors` writer and a pure-Python header reader (settles A20); `compat` table tests include the alphabet-mismatch case; negative control: a planted function-level `import numpy` must fail `lint-imports` (settles A46) |
| S4-05 | `pytest tests/test_executors.py`: unknown kind rejected, `onnx` with more than one artifact graph and no driver rejected, `iree` level limited to xtrax levels, source mode accepts `driver.url` as a `release-asset:` placeholder with `driver.sha256` `"pending"` and release mode rejects both | planted `"pending"` on a non-driver artifact `sha256` must be rejected in both modes |
| S4-06 | `pytest tests/test_manifest.py`: schema-validates a fixture, rejects an `hf://` URL without a 40-hex revision, rejects older and newer `schema_version`, accepts `release-asset:` URLs only in source mode (`catalog_view` refuses them unless `allow_placeholders`), port `required`/`cardinality` round-trip, `catalog_view` output matches a snapshot, source mode accepts a pending `driver.sha256` only on an executor `driver` and release mode rejects it, `catalog_view` refuses placeholders and a pending driver hash unless `allow_placeholders` | planted pending `sha256` on an artifact entry must be rejected |
| S4-07 | `pytest tests/test_schemagen.py`: mapping table for every supported type including `Callable` (via handler), a registered domain-type handler is honoured, both tuple modes tested, unmappable type raises, `Any` rejected unless opaque, only metadata-exposed fields emitted, keywords outside the profile fail emission, the module's mapping table equals the target table committed in the S4-02 sidecar, and `field_meta.v1.json` (the twelve-key `FieldMeta` shape, published for S1-06) is committed and loaded by both `meta()` and schemagen, an unknown key inside `metadata["xtrax"]` fails emission and a metadata key other than `"xtrax"` is ignored | negative: planted `Any` field must raise; a planted extra entry in the table must fail the equality check; a planted unknown `FieldMeta` key must fail emission. S1-06 consumes `field_meta.v1.json` from this item, so the file, not prose, is the shape S1 builds against |
| S4-08 | on titanix or CI: `uv run pytest tests/inference/test_ir_schema.py` passes and the emitted v1 schema JSON is byte-identical before and after the refactor (semantic equivalence plus changelog only if S4-02 recorded a fixed-arity IR field) | n/a |
| S4-09 | `uv run python -m xtrax_contract.emit --check` exits 0 for all four schemas (`model-manifest.v1`, `port-types.v1`, `param-profile.v1`, `graph-ir.v2`) and the `conformance/catalog_view` snapshots usable from Node (no Python needed to read them); `npx tsc --noEmit -p packages/xtrax-contract/ts` exits 0; `uv run python scripts/audit_ir_schema_single_source.py` and `pytest tests/audit/test_ir_schema_single_source.py` pass with the audit scanning `packages/xtrax-contract/src` and `ts` and allowing `$schema` only in `emit.py` and `schemas/*.json`; the generator is stdlib Python and `package.json` lists only `typescript` as a dev dependency | planted stale schema file must fail `--check`, for the graph schema as well as the manifest schema; a planted `$schema` marker in a contract-package module other than `emit.py` must fail the audit; the `catalog_view` snapshot corpus (`conformance/catalog_view/*.json`) is committed, guarded by `--check`, and a planted altered snapshot must fail it |
| S4-10 | `cd packages/xtrax-contract && uv run pytest -q tests/test_graphdoc.py` (stdlib only): `GraphDoc`/`NodeDoc`/`EdgeDoc` round-trip, v1 fixtures load through the upgrade rule, v2 round-trips, versions 0 and 3 rejected without default-fill, `ref` grammar (`<id>@<version>`, `builtin:code.python`, `py:`), `metadata.nl_description` constant enforced (`E_NODE_METADATA`), `x-` keys preserved; and on titanix or CI `pytest tests/run/test_runspec_base_pinned.py`: `xtrax.run.RunSpec` field names and identity `from_spec` pinned | planted unknown non-`x-` key must be rejected; planted renamed `RunSpec` field must fail the pin test |
| S4-44 | on titanix or CI: `pytest tests/composition tests/cli tests/loop tests/run tests/stages`: v1 fixtures still load; a live callable serialises to `py:mod:qualname`; a lambda, a closure and a `functools.partial` each raise `GraphSerializationError` naming the node; `py:` refs resolve eagerly on load for the Python executor; `schema_version=1` writer succeeds for a py-only graph and raises for a typed-port graph; the audit test lists all 17 a11 files (4.11), has a `callable_ref=None` case for each use site, and fails if a new file mentions `callable_ref` or `HostPrepGraph` without being listed; the contract's `nl_description` constant equals the slot in `node_metadata_schema.toml` | planted extra file with an unlisted `callable_ref` use must fail the audit; planted changed toml slot must fail the equality test |
| S4-45 | on titanix or CI: `uv run pytest tests/inference/test_ir_schema.py tests/audit/test_ir_schema_single_source.py`: composed `$defs.Node` and `$defs.Edge` equal the contract's after JCS, the xtrax-side `$defs` (BundleSchema, AxisSpec, AxisOverride, AxisBoundary, Fuse, Tap, Sink) are still emitted, `$id` unchanged from a11, `emit_ir_schema` contains no hand-written Node or Edge literal (checked by an `ast` test) | planted divergence in the contract's Node schema must fail the equality test |
| S4-11 | `pytest tests/test_registry.py` on Python 3.11 and 3.13 with synthetic dists: a regular dist-info, a `.pth`-style editable layout and a finder-style editable layout each discover the manifest by bare package name without importing the package (sentinel module stays unimported), a dotted value is rejected, a zipped dist yields `DiscoveryError`, malformed sibling isolated, id collision rejected | settles A12 on 3.11 at runtime and, for synthetic layouts, A45 |
| S4-12 | `pytest tests/cli/test_graph_validate_contract.py`: the contract validator is invoked, an output-name mismatch found by `check_ports` (S4-13) fails the verb naming the node (exit non-zero), a node without a python executor is reported unplannable and the verb exits 0, node-metadata slots beyond `nl_description` are validated through `load_node_metadata_schema` | toy registry fixture, no JAX models needed |
| S4-13 | `pytest tests/test_protocols.py`: a conforming fake passes, swapped argument order and missing direction/units both fail `assert_conforms`, and on titanix or CI `check_ports` fails a callable whose output names disagree with the declared ports | negative controls built in |
| S4-14 | on CI or titanix: `uv run pytest tests/export/test_bundle.py` (export extra): toy two-graph bundle has recorded names and hashes, interrupted run resumes with reused units listed, tampered artifact is recomputed, cache root resolves arg then `XTRAX_EXPORT_CACHE` then pyproject then user config, a planted external-data tensor fails the self-contained assertion, and the embed step from S4-46 is used if that item landed | planted tampered artifact must be recomputed, not reused; planted external-data graph must fail; per-unit process and timeout |
| S4-15 | `pytest tests/export/test_hf_weights.py` with a mocked hub: revision forwarded to `hf_hub_download` and present in the report | mocked hub |
| S4-16 | after the S3-13 release, on top of the merged #174 pin (S2-02): the recorded resolution (S4-03 record) shows base `orbax-checkpoint >= 0.12`, `xtrax[export]` is out of the base dependencies, the toolchain is in a dependency group, and import smoke of the package and the export scripts passes on titanix | records which option (A, B, D) was applied |
| S4-17 | sidecar committed first, then on titanix `bth run ... scripts/browser_validation/p07_split_export.py --buckets 128 256`: record shows all graphs exported, `self_contained` true, manifest written, `jax_enable_x64` false; `scripts/browser_validation/lint_sidecars.py` passes. Scope is `p07_split_export.py` only | negative control (r2 R2-11): the S4-43 planted external-data fixture is run through the same assertion in the same sidecar and must make the assertion and the outcome fail; a tampered artifact must be recomputed. Per-bucket per-graph units with completion stamps, reuse only on matching sha256 inputs, reused units listed in the result |
| S4-18 | pre-registered sidecar for the ORT-CPU cell of the chain (split-vs-reference and split-vs-monolith) against the xtrax-route manifest: tokens exact, log-prob gap within inherited 2e-4 nats, teacher-forced within inherited 1e-4 nats | existing planted-perturbation controls kept and must still be detected; per-cell results persisted, each cell its own process and timeout |
| S4-38 | pre-registered sidecar for the Node/wasm parity cell on the xtrax-route artifacts, same inherited bounds and controls as S4-18 | one cell per process and timeout, results persisted |
| S4-39 | pre-registered sidecar for the headless-Chromium parity cell on the xtrax-route artifacts, same inherited bounds and controls as S4-18; this is the record S5-15 compares against | one cell per process and timeout, results persisted |
| S4-19 | `pytest tests/contract/test_manifest.py` on titanix: both manifests validate, every executor `module:symbol` resolves via `find_spec`, shared artifact entries identical across the two manifests and equal to the S4-17 record, golden-vector entries have sha256, every `validation.evidence[].bathos_id` resolves via `bth sql` to a record with a passing outcome whose executor and scope match the S4-18/S4-38/S4-39 cell, `license.redistribution` equals the S4-37 decision, both manifests validate in source mode (`allow_placeholders`): every `urls` entry is a `release-asset:` placeholder or a pinned `hf://` URL, `driver.url` is a placeholder and `driver.sha256` is `"pending"` (the final `onnx` entry exists only after S5-14 and S5-52), ids and entry-point stems follow the S4-01 grammar (`molxmpnn/proteinmpnn.sample` is `proteinmpnn-sample`), and the manifest `version` and every `module:symbol` resolve in the tree of the tag the manifests are released from, as listed in the S3 release-train table | resolve test is the rename tripwire; a planted evidence entry with a made-up id must fail; a planted id/stem mismatch must fail |
| S4-20 | sidecar-registered equivalence run: generated JS constants, defaults and validator agree with the S4-41 golden vectors, `node --test browser/molxmpnn-sampler` passes, S1-18's drift test is deleted and its key-set property is now enforced by generation | planted wrong default and planted shuffle off-by-one must both fail |
| S4-21 | `pytest tests/contract/test_scorer.py` on titanix: `assert_conforms` passes, declared direction and units match the code, sign-flip control fails; discovery works from (a) the wheel built by the repo's actual backend (setuptools), proving the bare-module entry value passes build-time validation, and (b) a real editable install of the dev checkout | negative: a path-valued entry point must be rejected by the build (confirms A43) |
| S4-22 | proteinsmc `uv run pytest tests/scoring/test_mpnn_discovery.py` with a fake scorer: discovery imports no provider package (sentinel unimported), manifests with `schema_version` 0, 2 or missing, or with a missing or unknown `role`, are skipped with a warning and version 1 is accepted, the extended D6 check (imports, `import_module` of provider names, `[project.dependencies]` and `[project.optional-dependencies]`; `[dependency-groups]` exempt) is empty, and `rg -n "find_spec\(.(molxmpnn|prxteinmpnn)" proteinsmc/src` is empty (exactly one guard implementation survives; if S4-49 answered "sequence", S3-18's root-only guard is replaced here and this item depends on S3-18) | fake scorer, no provider installed; planted version-2 manifest must be skipped, planted `import molxmpnn` must fail the D6 check, a planted `molxmpnn` in `[project.optional-dependencies]` must fail it and in `[dependency-groups]` must pass |
| S4-48 | proteinsmc `uv run pytest tests/scoring/test_mpnn_adapter.py` with a fake scorer: a path and a `PdbText` structure both reach `bind` unchanged; alphex tokens are mapped to the port's `symbols` by symbol (an unknown symbol raises, a permuted destination ordering changes the mapped tokens); the declared `direction` is applied; a lower NLL yields a higher fitness; `make_mpnn_score` and the `"mpnn"` registry entry follow the Q9 decision; the proteinsmc fitness-sign convention is cited in the test with a `path:line` | negative controls: an adapter that skips the negation fails the ordering test; an adapter that skips the token mapping fails the permutation test |
| S4-49 | decision recorded (in the S3 or S4 tree, linked from both): supersede S3-18 or sequence S4-22 after it, and the `make_mpnn_score` fate (Q9); if "sequence", the record adds `S3-18` to S4-22's `depends_on` (and S3-18 keeps its S4-49 edge) | n/a |
| S4-23 | dev-only e2e in proteinsmc: a non-default `[dependency-groups]` entry owned by this item (not S3-18's) installs the MPNN package from a wheel built from the molxmpnn checkout, and a second run uses an editable install of the same checkout; in each, discovery returns the scorer and one `bind` + call through the S4-48 adapter returns a finite scalar; the extended D6 check (groups exempt) and `git diff pyproject.toml` show no new entry in `[project.dependencies]` or `[project.optional-dependencies]` | skipped cleanly when provider absent; a planted `molxmpnn` in `[project.dependencies]` must fail the D6 check |
| S4-24 | `pytest tests/test_export_shape_contract_ports.py` in plegadx (with `xtrax-contract` pinned by S4-36): every `traced_inputs` entry of every export becomes a valid `PortSpec` (dotted names accepted), and `export(import(entry)) == entry` as parsed JSON with `shape_spec`, `example_shape`, `variability_axis`, `static_arg_required` and the two policy blocks carried under `x-plegadx` (4.2) | planted entry with a malformed `shape_spec` must fail |
| S4-25 | sidecar-registered differential compile of one fixed StableHLO module under prolix flags and under xtrax `WASM32`: success, size, and whether either flagset is rejected | negative: a deliberately invalid triple must fail in both arms. One flagset per process |
| S4-26 | prolix tests pass after adoption and the 50 MiB size gate still fails an over-limit artifact | per the S4-25 outcome |
| S4-27 | retired (id not reused): the localfold upstream request, vendoring and adapter are S5-25, S5-23 and S5-26 | n/a |
| S4-28 | decision recorded; if yes, `npm pack --dry-run` shows only generated files | n/a |
| S4-29 | xtrax `pytest tests/contract/test_toy_graph.py`: three-node toy graph over fake manifests discovers, validates and plans end to end with no JAX model | n/a |
| S4-30 | `pytest tests/test_canonical.py`: output equals the shared vector file (`0.1`, `1e-5`, `1e16`, `1.0`, 2^53 ints, key ordering, unicode escapes, **key ordering by UTF-16 code unit with an astral key against a BMP key above U+D800, e.g. U+10000 against U+FF5E**, and rejection of a lone surrogate and of `NaN`/`Infinity`), a manifest-hash vector (`manifest_sha256 = sha256(JCS(manifest))`, 4.3), hash unchanged under key reordering; S4 is the sole owner of the vector file (S6-26 superseded) and S6-03 runs the TS side | negative controls: naive `json.dumps` must fail the vectors (settles A40), and an implementation that sorts keys with plain `sorted()` (code-point order) must fail the astral-key vector (settles A56) |
| S4-31 | `pytest tests/test_validate.py tests/test_conformance_corpus.py`: every code in the registry is produced by at least one vector, `compat` cases incl. alphabet mismatch, v1 upgrade documents, `E_PARAM_SCHEMA` raised through the S4-47 engine | mutation control: flipping one table entry must fail the corpus |
| S4-32 | on titanix: `pytest tests/run/test_xtrax_metadata_vs_snapshot.py`: the schema emitted from the unified RunSpec classes has exactly the properties the committed `run_spec_fields.json` marks portable, keyed by `browser_key`; the S4-02 classifier script, re-run as a new bathos run on the unified classes after S1-07, reports zero unmapped exposed fields; every exposed field's `metadata["xtrax"]` dict validates against exactly the `field_meta.v1.json` shape from S4-07 (no other key; S1's emitter needs no import of `xtrax_contract`) | negative: a planted portable row with no field metadata, a planted exposed field with no row, and a planted unknown metadata key must each fail |
| S4-33 | decision/semantics note merged: `chains_to_design` versus `chain_id` mapping, constraint-lowering conventions, expected outputs for the chain cases (inputs to S4-41's pre-registration) | n/a |
| S4-34 | `xtrax-contract` and `xtrax` released on the Q11 channel with tags `xtrax-contract-v<ver>` and `v<ver>` (4.1). Two checks: (1) in a clean Python 3.11 venv the released `xtrax-contract` installs with no dependencies and `import xtrax_contract` succeeds; (2) in a clean Python 3.13 venv `pip install xtrax` resolves, and `importlib.metadata.version("xtrax-contract")` lies inside the range pinned by the released `xtrax`; the released schemas include the S5-51 amendments (G1-G4) and the `catalog_view` snapshots, so there is one meaning of `model-manifest.v1` | n/a |
| S4-35 | on titanix: `uv lock --check` exits 0 with the new `xtrax` and `xtrax-contract`, the lock's base `orbax-checkpoint` is `>= 0.12` (S4-03 record method), and `uv run python -c "import xtrax_contract"` exits 0 | n/a |
| S4-36 | in plegadx: `uv lock --check` exits 0 after the dependency is added, and `uv run python -c "import xtrax_contract"` exits 0 | n/a |
| S4-37 | decision recorded in the aminx tree (carried across the rename by S3-07): weights license text and `redistribution` value (allowed/gated/forbidden) for each shipped weight set, with one named reviewer and one record location in the aminx/molxmpnn tree; S5-07 and S5-06 consume this record, and `catalog/licenses/<model>.review.toml` cites it and must agree (same redistribution value and reviewer), which S5's gate tests | n/a |
| S4-40 | `pytest tests/contract/test_knob_document.py` on titanix: the knob class round-trips its ten browser keys, lowering of simple cases equals hand-computed arrays, the `FieldMeta` exposure set matches S1's snapshot (S4-32 reused) | negative: a planted wrong key mapping must fail |
| S4-41 | sidecar committed first, then on titanix `bth run` of the lowering gate: Python lowering of every golden knob case equals the vectors, the JS kernels (`node`) equal the same vectors, the script twin `p07_knobs_gate` agrees on the shared cases | planted off-by-one in a shuffle and planted chain-semantics swap (per S4-33) must each fail; per-case results persisted, each case its own process and timeout |
| S4-42 | `build_p07_inputs` and its test references deleted; `rg build_p07_inputs` is empty; S4-41's gate re-run passes unchanged | n/a |
| S4-43 | sidecar committed first, then on titanix `bth run` of the spike over the four real MPNN graphs at one bucket: external-data tensor count is 0 for each graph (or the count per graph is recorded and S4-46 is required); per-graph result persisted, one graph per process and timeout | negative control: a planted ModelProto with a tensor stored as external data (fixture in the aminx tree, reused by S4-17) must make the assertion and the sidecar outcome fail; verified by record not exit code |
| S4-46 | conditional on the S4-43 record. If external data was found: `pytest tests/export/test_embed.py` shows the embed step makes a planted external-data graph self-contained and its output loads in ORT. If none was found: item closed with the S4-43 run id cited and no code change | planted external-data graph must fail before the embed and pass after |
| S4-47 | `pytest packages/xtrax-contract/tests/test_profile_validate.py` and `node --test packages/xtrax-contract/ts/test` both pass over `conformance/param-profile.v1.json`; an emitted schema containing a keyword outside the profile is refused by emission | a planted keyword outside the subset must be refused in both languages; a planted corpus case whose expected error path is wrong must fail both |

## 8. Open questions for the user

1. **Q1. Contract packaging, answered together with Q4 (S4-01).** F2 (separate `xtrax-contract` dist built from the xtrax
   repo, Python >= 3.11, no dependencies, `xtrax.contract` shim) versus F1 (inside xtrax,
   carve out later). *Recommendation: F2*, because xtrax needs Python 3.13 and jax (A7) and the
   consumers D6 protects do not. Also confirm the names `xtrax-contract` / `xtrax_contract`
   and the group `xtrax.models`. The same ADR fixes the two-dist version and tag policy and the
   generation toolchain (4.1: stdlib generator, Node check-only, `typescript` as the only dev
   dependency); *recommendation: accept as written*.
2. **Q2. Owner of the browser knob document, and its chain semantics (S4-33, S4-40, S4-20).** S1 D10 and
   S1 Q6 recommend that S4 owns the knob document and the pure lowering, as a projection of the
   surviving RunSpec, and that S1 builds no `DesignConstraints`. *Recommendation: accept
   (S4-40 builds it, S4-41 vectors it, S4-20 generates from it); keep `runspec_core.mjs` numeric kernels
   hand-written and golden-vector-gated.* Also decide the semantic gap S4-33 has to settle:
   the browser's `chains_to_design` (chain letters to design) against Python's `chain_id`; the
   recommendation is to keep the browser vocabulary at the portable layer and map it in the
   lowering, never to rename the browser key.
3. **Q3. orbax conflict (S4-16).** Option A (consumer-side override, toolchain in a dependency
   group, no published `onnx` extra) versus B or D. *Recommendation: A, plus move
   `xtrax[export]` out of the MPNN package's base dependencies, with D as fallback if S4-03
   shows A breaks the ONNX scripts.*
4. **Q4. Ship discovery now?** alphex cut its registry for lack of a second registrant (A10).
   Here there are at least two likely registrants (MPNN, EBM, later plegadx). *Recommendation:
   ship, with the manifest-as-data entry value (a bare package name, 4.8) so discovery imports no
   model code.* Answered in the S4-01 ADR; S4-11, S4-21 and S4-22 depend on S4-01 (r1 C12).
5. **Q5. Publish the generated TS package to npm (S4-28)?** Needs an npm scope and a version
   policy. *Recommendation: keep the artifacts in-repo and consumed by URL or vendoring until the
   hub (S5) needs a pinned release; publish later.*
6. **Q6. AF3 in the hub (owned by S5; S4-27 retired, r3 C13).** localfold's AF3 driver is not in
   its published package (A29), and localfold has no postMessage or URL job API, so an
   embed/iframe handoff is not an option without an upstream change. The upstream request
   (S5-25), the vendored job-json at a pinned commit (S5-23) and the adapter that needs the
   Beerware vendoring consent (S5-26) are S5's items; the S4 contract only reserves the `npm`
   executor shape. *Recommendation: as S5 sequences it: link-out with a manual drop-in first,
   upstream request, vendoring only with the author's consent.* No S4 item remains.
7. **Q7. prolix WASM flags (S4-26).** Decide after S4-25 measures, not before: adopt xtrax's
   `WASM32` target, amend the xtrax target, or keep prolix's flags. *Recommendation: let the
   differential result pick; keep prolix's 50 MiB gate regardless.*
8. **Q8. Weights licensing (S4-37).** Who audits license and redistribution terms of each shipped
   weight set before a public catalog lists it? *Recommendation: required manifest fields plus a
   hub CI check; a human sign-off for each first listing.* S4-19 depends on the S4-37 decision item. It is one decision with one reviewer and one record in the aminx/molxmpnn tree (r3 C11): S5-07 depends on S4-37 and cites the record from `catalog/licenses/<model>.review.toml`, and S5-06 stages to public tiers only when the manifest value and that record agree, so the hub cannot hold a second, different answer.
9. **Q9. S3-18 versus S4-22, and the fate of `make_mpnn_score` (S4-49, S4-22, S4-48).** Both
   S3-18 and S4-22 edit the same proteinsmc guard (S3-18 installs a root-only
   `find_spec("molxmpnn")` guard, which is import-free; the dotted form that A36 shows importing the
   parent was S3's earlier draft). If S4-49 answers "supersede", S3-18 is closed unmerged and
   S4-23 supplies its own non-default dependency group so the end-to-end test does not need S3-18's
   (4.9). Superseding a sibling spec's item is the user's decision, so
   the backlog does **not** bake in an answer (r2 R2-4): decision item S4-49 records it, S4-22
   depends on S4-49 and is `user_decision = true`, and S3-18 is owed a dependency on S4-49
   (section 6). *Recommendation: supersede, so S3-18 is closed unmerged and S4-22 is the only guard
   (one guard, manifest-driven, no name coupling to the provider package).* If the user prefers to keep S3-18, S4-22
   lands after it and replaces its guard; either way exactly one guard survives (S4-22 gate).
   Second half of the question: change the public `make_mpnn_score` signature to
   `(structure, decoding_settings="random", *, scorer_id=None)` and keep the `"mpnn"` registry key
   (recommended: the old form has no working caller, A54), or leave it and add a new key (4.9).
10. **Q10. `metadata.nl_description` in v2 documents.** xtrax requires it per node; S6's example
    nodes omit `metadata`. *Recommendation: keep the xtrax invariant, have S6's editor prefill it
    from the manifest `description` on insertion, so no default-fill happens in the loader.* Not
    blocking: S4-10 implements the recommendation (keep the invariant).
11. **Q11. Release channel for `xtrax-contract` and `xtrax` (S4-34).** PyPI, a git-tag pin, or a
    path/editable source. *Recommendation: PyPI for both (the MPNN repo already resolves xtrax from
    PyPI, A47; plegadx may keep a path-editable xtrax in dev and pin the contract from PyPI).*
    Publishing is a user action.
12. **Q12. Where the ONNX graphs and golden vectors are hosted** (the `urls` of S4-19's artifacts).
    Hosting is a non-goal here; S4-19 ships `release-asset:` placeholders and S5-52 rewrites them.
    *Recommendation: GitHub release assets of the MPNN repo, as S5-52 already assumes.*
13. **Q13. Accept the S5-51 amendments G1-G4 into `v1` before the first release (r3 C9).** The
    decision is S5-51's (`user_decision`); S4's side is that S4-34 depends on it, so a "yes" means
    the first published `model-manifest.v1` carries G1-G4 and the `catalog_view` corpus, and a
    "no" (declined, treated as satisfied for ordering) means the hub works from the unamended `v1`
    and any later amendment is a `schema_version` 2 bump. *Recommendation: accept; they are
    additive and nothing is published yet.*

## 9. Backlog items

```toml
[[item]]
id = "S4-01"
title = "ADR: contract packaging (xtrax-contract dist vs in-xtrax), names, Python floor, ship-discovery-now (Q4), two-dist version and tag policy, generation toolchain (stdlib generator, Node check-only); amend the 260702 fork-13 note (one emitter module in the contract package, composed by one in xtrax)"
repo = "xtrax"
size = "S"
depends_on = []
gate = "ADR merged under xtrax .praxia/docs/decisions/ recording the Q1 and Q4 answers, dist and import names, python floor 3.11, the entry-point group name, the manifest id grammar <namespace>/<family>.<op> with the entry-point stem rule and manifests/ directory (molxmpnn/proteinmpnn.sample and molxmpnn/proteinmpnn.score as worked examples), the version and tag policy of the two dists, the generation toolchain and the amended fork-13 note"
user_decision = true

[[item]]
id = "S4-02"
title = "Spike (pre-registered bathos): which annotations of the real pre-S1-07 MPNN spec classes the S4-07 target mapper table can and cannot express (table written in the sidecar), and whether any xtrax a11 IR field is a fixed-arity tuple (read by ast at a pinned ref)"
repo = "aminx"
size = "S"
depends_on = []
gate = "bth record on titanix with evaluated outcome: aminx tree commit recorded, every field classified mapped, opaque-flagged or unmapped against the sidecar's target table with unmapped names recorded, xtrax a11 IR tuple fields listed from an ast walk at the pinned ref, positive control (toy dataclass maps fully) and negative control (bare Callable field is flagged) both behave; claim limited to pre-flip classes"
user_decision = false

[[item]]
id = "S4-03"
title = "Spike (pre-registered bathos): onnx-extra resolution arms (consumer override, isolated env, uv conflicts) and resolved orbax-checkpoint per arm"
repo = "aminx"
size = "S"
depends_on = []
gate = "bth record shows resolved orbax-checkpoint for the base install per arm and an onnx-script import smoke per arm, with the unmodified xtrax[onnx] arm reproducing the known downgrade as negative control"
user_decision = false

[[item]]
id = "S4-04"
title = "xtrax-contract skeleton (layout, errors, version discipline, import-linter contract), uv workspace membership, sub-dist pyproject (hatchling, version source, requires-python 3.11, no dependencies), xtrax dependency on xtrax-contract with workspace source, declarative port-type table, PortSpec and port-name grammar, compat evaluator with alphabet rule, port-value wire format with no array-library import"
repo = "xtrax"
size = "M"
depends_on = ["S4-01"]
gate = "on CI or titanix: cd packages/xtrax-contract && uv run pytest -q tests/test_ports.py tests/test_wire.py && uv run lint-imports (jax, numpy, equinox, xtrax forbidden); at the xtrax root uv lock --check exits 0 and uv run python -c 'import xtrax_contract, xtrax' resolves xtrax_contract from packages/xtrax-contract; sub-dist wheel has requires-python >=3.11 and no Requires-Dist; wire test round-trips via the real safetensors writer, alphabet-mismatch compat case rejects, planted function-level import numpy fails lint-imports"
user_decision = false

[[item]]
id = "S4-05"
title = "ExecutorEntry vocabulary (python, onnx, iree, npm, pyodide, remote), ArtifactEntry with per-format details, driver reference (source mode permits a release-asset: driver.url and a pending driver.sha256, only there), determinism field"
repo = "xtrax"
size = "S"
depends_on = ["S4-04"]
gate = "pytest tests/test_executors.py: unknown kind rejected, onnx entry with more than one graph and no driver rejected, iree level limited to xtrax verification levels, source mode accepts a release-asset: driver.url with driver.sha256 pending and release mode rejects both, a pending sha256 on a non-driver artifact rejected in both modes"
user_decision = false

[[item]]
id = "S4-06"
title = "ModelManifest (HUB-REQ 1-10: kind, family, ports with required and cardinality, artifacts, executors, license, provenance, validation, limits) with JSON (de)serialisation, schema_version gate, source mode (release-asset placeholder URLs and a pending driver.sha256, driver only) versus release mode, and catalog_view"
repo = "xtrax"
size = "M"
depends_on = ["S4-04", "S4-05"]
gate = "pytest tests/test_manifest.py: fixture validates, hf url without a 40-hex revision rejected, schema_version older and newer than known both rejected without default-fill, release-asset URLs and a pending driver.sha256 accepted only in source mode (a pending artifact sha256 rejected in both), catalog_view equals its snapshot"
user_decision = false

[[item]]
id = "S4-07"
title = "schemagen classes front-end: stdlib-only dataclass / eqx.Module to JSON Schema (param profile), type-handler registration hook, homogeneous and fixed tuple modes, loud failure on unmappable types, opt-in field exposure via metadata[xtrax], and the published FieldMeta key list (field_meta.v1.json, twelve keys) that S1-06 emits as a plain dict"
repo = "xtrax"
size = "M"
depends_on = ["S4-01", "S4-02"]
gate = "pytest tests/test_schemagen.py: mapping table per supported type incl. Callable via handler, both tuple modes, unmappable type raises, Any rejected unless opaque, only metadata-exposed fields emitted, keywords outside the profile fail emission, the mapping table equals the target table committed in the S4-02 sidecar (a planted extra entry fails the equality check), and field_meta.v1.json is committed and loaded by meta() and schemagen with a planted unknown FieldMeta key failing emission; S1-06 consumes field_meta.v1.json from this item (an edge owed, section 6), so its key list is the authoritative source for S1-06 and S1-31"
user_decision = false

[[item]]
id = "S4-08"
title = "xtrax ir_schema delegates its type mapping to the shared mapper (homogeneous tuple mode, jax types via handlers) with no change to the emitted v1 schema"
repo = "xtrax"
size = "S"
depends_on = ["S4-07"]
gate = "on titanix or CI: uv run pytest tests/inference/test_ir_schema.py passes and the emitted v1 schema JSON is byte-identical before and after (semantic equivalence plus changelog only if S4-02 recorded a fixed-arity IR field)"
user_decision = false

[[item]]
id = "S4-09"
title = "Generated artifacts: committed JSON Schemas (model-manifest.v1, port-types.v1, param-profile.v1, graph-ir.v2) from xtrax_contract.emit, stdlib TS declaration generator and generated validators bound to the S4-47 engine, CI drift gate (emit --check), the catalog_view snapshot corpus committed as conformance data usable from Node, and the single-source audit extended to the contract package"
repo = "xtrax"
size = "L"
depends_on = ["S4-06", "S4-07", "S4-10", "S4-47"]
gate = "uv run python -m xtrax_contract.emit --check exits 0 for all four schemas, npx tsc --noEmit -p packages/xtrax-contract/ts exits 0, scripts/audit_ir_schema_single_source.py and tests/audit/test_ir_schema_single_source.py pass scanning both trees with $schema allowed only in emit.py and schemas/*.json, ts/package.json lists only typescript as a dev dependency; conformance/catalog_view snapshots are committed and guarded by --check; a planted stale graph-ir.v2 file or an altered catalog_view snapshot fails --check and a planted $schema marker in another contract module fails the audit"
user_decision = false

[[item]]
id = "S4-10"
title = "Graph IR v2 wire format in the contract (stdlib): graphdoc dataclasses, ref-addressed nodes, builtin code nodes, port-addressed edges with id and kind, schema_version 2, v1 reader with upgrade rule, nl_description constant (E_NODE_METADATA), plus the xtrax RunSpec base pin test"
repo = "xtrax"
size = "M"
depends_on = ["S4-04", "S4-05"]
gate = "cd packages/xtrax-contract && uv run pytest -q tests/test_graphdoc.py: v1 fixtures load via the upgrade rule, v2 round-trips, versions 0 and 3 rejected without default-fill, ref grammar, nl_description constant enforced, x- keys preserved, planted unknown non-x- key rejected; and on titanix or CI pytest tests/run/test_runspec_base_pinned.py pins RunSpec field names and identity from_spec (planted renamed field fails)"
user_decision = false

[[item]]
id = "S4-11"
title = "Model discovery and node-type registry: xtrax.models entry-point group with manifest-as-data values (bare package name, manifests/<name>.json located by find_spec without import), lazy scan, per-entry isolation, id collision guard"
repo = "xtrax"
size = "M"
depends_on = ["S4-01", "S4-06"]
gate = "pytest tests/test_registry.py on Python 3.11 and 3.13 with synthetic regular, pth-editable and finder-editable dists: manifest discovered without importing the package (sentinel module stays unimported), dotted value rejected, zipped dist reported, malformed sibling isolated, id collision rejected"
user_decision = false

[[item]]
id = "S4-12"
title = "xtrax graph-validate and graph-plan verbs call the contract validator, add extract_schema consistency via check_ports and report nodes without a python executor as unplannable"
repo = "xtrax"
size = "M"
depends_on = ["S4-10", "S4-44", "S4-11", "S4-13", "S4-31"]
gate = "pytest tests/cli/test_graph_validate_contract.py: contract validator invoked, output-name mismatch found by check_ports fails the verb naming the node, node without python executor reports unplannable and exits 0, node-metadata slots beyond nl_description validated via load_node_metadata_schema"
user_decision = false

[[item]]
id = "S4-13"
title = "Scorer, Sampler, StructurePredictor, EnergyModel protocols, provider-side conformance kit (assert_conforms) and the xtrax-side check_ports helper over extract_schema"
repo = "xtrax"
size = "M"
depends_on = ["S4-04", "S4-06"]
gate = "pytest tests/test_protocols.py: a conforming fake passes, swapped argument order and missing direction/units both fail assert_conforms, and on titanix or CI check_ports fails a callable whose output names disagree with the declared ports"
user_decision = false

[[item]]
id = "S4-14"
title = "xtrax.export.bundle.export_onnx_bundle: resumable per-graph per-bucket ONNX export on convert_to_onnx emitting manifest-fragment artifacts with names, sha256, census, RNG audit and self-contained assertion; export cache resolved via config"
repo = "xtrax"
size = "L"
depends_on = ["S4-04", "S4-05", "S4-43", "S4-46"]
gate = "on CI or titanix: uv run pytest tests/export/test_bundle.py (export extra): toy two-graph bundle has recorded names and hashes, interrupted run resumes with reused units listed, tampered artifact is recomputed, cache root resolves arg then XTRAX_EXPORT_CACHE then pyproject then user config, planted external-data tensor fails the self-contained assertion, S4-46's embed step used if that item landed"
user_decision = false

[[item]]
id = "S4-15"
title = "hf_weights: add a revision pin parameter and report it in WeightReport"
repo = "xtrax"
size = "S"
depends_on = []
gate = "pytest tests/export/test_hf_weights.py with a mocked hub: revision forwarded to hf_hub_download and present in the report"
user_decision = false

[[item]]
id = "S4-16"
title = "MPNN repo: Q3 resolution on top of merged #174 (dependency group, xtrax[export] out of base; changes the MPNN lock after S2-02, intentionally unordered against S2-05..S2-34, which run A0 from a pinned F' checkout)"
repo = "molxmpnn"
size = "S"
depends_on = ["S4-03", "S3-07", "S3-13", "S2-02"]
gate = "on titanix, on top of the merged #174 pin: uv lock keeps base orbax-checkpoint >= 0.12 per the S4-03 record, import smoke of the package and of the export scripts passes, no onnx extra is published"
user_decision = true

[[item]]
id = "S4-17"
title = "MPNN repo: route p07_split_export.py (only) through the xtrax bundle exporter, drop the private jax2onnx wrappers and duplicate RNG walker, re-baseline the sha256 manifest (pre-registered)"
repo = "molxmpnn"
size = "L"
depends_on = ["S4-14", "S4-16", "S4-35", "S4-43"]
gate = "sidecar committed first, then on titanix bth run p07_split_export --buckets 128 256: record shows all graphs exported, self_contained true, manifest written, x64 false, lint_sidecars.py passes, reused units listed; the S4-43 planted external-data fixture run through the same assertion makes assertion and outcome fail"
user_decision = false

[[item]]
id = "S4-18"
title = "MPNN repo: ORT-CPU cell of the split parity chain on xtrax-route artifacts under a pre-registered sidecar with inherited bounds"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-17"]
gate = "bth record shows tokens exact, log-prob gap within inherited 2e-4 nats and teacher-forced within inherited 1e-4 nats on ORT-CPU, planted-perturbation controls still detected, verified by record not exit code"
user_decision = false

[[item]]
id = "S4-19"
title = "MPNN ModelManifests (sample and score): stdlib-only source-mode manifest files with ids molxmpnn/proteinmpnn.sample and molxmpnn/proteinmpnn.score per the S4-01 grammar, ports, params schema refs, python and onnx executor entries only (driver.url a release-asset: placeholder, driver.sha256 pending; the remote executor entry is added by S5-21), weights pinned at the post-cutover HF revision, golden-vector artifacts, placeholder URLs and validation evidence from S4-18/S4-38/S4-39"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-06", "S4-17", "S4-18", "S4-38", "S4-39", "S4-37", "S4-35", "S3-07", "S3-11"]
gate = "pytest tests/contract/test_manifest.py on titanix: both manifests validate in source mode (allow_placeholders; python and onnx entries only, the remote entry is S5-21's) with driver.url a release-asset: placeholder and driver.sha256 explicitly pending (the final onnx entry comes from S5-14 and S5-52), ids and entry-point stems follow the S4-01 grammar, manifest version and every executor module:symbol resolve in the tree of the tag released from as listed in the S3 release-train table (find_spec), shared artifact entries identical and equal to the S4-17 record, golden-vector entries have sha256, every evidence bathos_id resolves via bth sql to a passing record of the matching executor, license matches the S4-37 decision, a planted id/stem mismatch fails"
user_decision = false

[[item]]
id = "S4-20"
title = "Generate the portable MPNN knob schema, TS types, constants, defaults and validator from the knob document via the S4-09 generator and S4-47 engine (no second generator), make runspec_core.mjs import them, delete S1-18's drift test"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-09", "S4-47", "S4-41", "S1-18", "S4-35"]
gate = "sidecar-registered equivalence run: generated JS constants, defaults and validator agree with the S4-41 golden vectors, node --test browser/molxmpnn-sampler passes, S1-18 drift test deleted, planted wrong default and planted shuffle off-by-one both fail"
user_decision = false

[[item]]
id = "S4-21"
title = "MPNN Scorer provider: bind(structure) to a FitnessFn-shaped ScoreFn, direction and units declared, xtrax.models entry-point registration with bare-package value, conformance test"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-01", "S4-11", "S4-13", "S4-19", "S4-35"]
gate = "on titanix: pytest tests/contract/test_scorer.py: assert_conforms passes, bind accepts a path and a PdbText structure and returns equal scores on the shared fixture PDB, declared direction (minimize) and units match the code, sign-flip control fails, discovery works from the setuptools-built wheel and from an editable install, a path-valued entry point is rejected by the build"
user_decision = false

[[item]]
id = "S4-22"
title = "proteinsmc: discover scorers via the xtrax.models entry-point group with stdlib only, version-check manifests (skip unknown or missing schema_version and role with a warning), replace the stale prxteinmpnn guard (and S3-18's root-only guard if S4-49 answered sequence and it landed) so exactly one guard survives, fake-scorer tests; depends on S3-18 only when S4-49 answers sequence (the S4-49 record adds that edge)"
repo = "proteinsmc"
size = "M"
depends_on = ["S4-01", "S4-06", "S4-49"]
gate = "uv run pytest tests/scoring/test_mpnn_discovery.py passes with a fake scorer: no provider package imported, manifests with schema_version 0, 2 or missing and with missing or unknown role are skipped with a warning, version 1 accepted, extended D6 check (imports, import_module of provider names, [project.dependencies] and [project.optional-dependencies]; [dependency-groups] exempt as dev-only, alphex D4) empty, a planted molxmpnn in optional-dependencies fails it and in a dependency group passes, and rg for find_spec of molxmpnn or prxteinmpnn in proteinsmc/src is empty"
user_decision = true

[[item]]
id = "S4-23"
title = "proteinsmc end-to-end (dev-only, skipped when absent): its own non-default dependency group installs the MPNN package (wheel built from the molxmpnn checkout, and an editable install), discover and call the scorer, verify D6 holds in pyproject; independent of S3-18"
repo = "proteinsmc"
size = "S"
depends_on = ["S4-21", "S4-22", "S4-48"]
gate = "dev-only test returns a finite scalar from bind plus call through the S4-48 adapter for both install modes (the item-owned non-default dependency group, wheel and editable) when the provider is installed, skips cleanly otherwise, and git diff shows no new entry in [project.dependencies] or [project.optional-dependencies] of proteinsmc/pyproject.toml"
user_decision = false

[[item]]
id = "S4-24"
title = "plegadx: adapter importing export_shape_contract.json into PortSpec (new code, no behaviour change to export.py)"
repo = "plegadx"
size = "S"
depends_on = ["S4-04", "S4-36"]
gate = "pytest tests/test_export_shape_contract_ports.py: every traced_inputs entry of every export converts to a valid PortSpec (dotted names accepted) and export(import(entry)) equals the entry as parsed JSON with shape_spec, example_shape, variability_axis, static_arg_required and policy blocks carried under x-plegadx, a planted malformed shape_spec fails"
user_decision = false

[[item]]
id = "S4-25"
title = "Spike (pre-registered bathos): differential iree-compile of one StableHLO module under prolix WASM flags and under xtrax WASM32"
repo = "prolix"
size = "M"
depends_on = []
gate = "bth record per flagset with success, artifact size and rejection reason, a deliberately invalid triple fails in both arms as negative control, one flagset per process"
user_decision = false

[[item]]
id = "S4-26"
title = "prolix: adopt xtrax WASM32 / compile_for_target or amend per the S4-25 outcome, keep the 50 MiB size gate as a manifest limits budget"
repo = "prolix"
size = "M"
depends_on = ["S4-25"]
gate = "prolix export tests pass after adoption and the size gate still fails an over-limit artifact"
user_decision = true

[[item]]
id = "S4-28"
title = "Decide and, if yes, publish the generated contract TS package to npm"
repo = "xtrax"
size = "S"
depends_on = ["S4-09"]
gate = "decision recorded and, if yes, npm pack --dry-run lists only generated files and the declared version"
user_decision = true

[[item]]
id = "S4-29"
title = "Toy end-to-end fixture: three-node graph over fake manifests discovers, validates and plans with no JAX model"
repo = "xtrax"
size = "S"
depends_on = ["S4-12", "S4-13"]
gate = "pytest tests/contract/test_toy_graph.py passes: discovery, graph-validate v2 and graph-plan over the toy registry"
user_decision = false

[[item]]
id = "S4-30"
title = "xtrax_contract.canonical: RFC 8785 JCS dumps and sha256 plus the shared hash vector file including a manifest-hash vector (S4 sole owner; supersedes S6-26's vectors)"
repo = "xtrax"
size = "S"
depends_on = ["S4-04"]
gate = "pytest tests/test_canonical.py: output equals every vector (0.1, 1e-5, 1e16, 1.0, 2^53 ints, key order, unicode escapes, UTF-16 code-unit key order with an astral key against a BMP key above U+D800, lone-surrogate and NaN rejection), a manifest-hash vector (manifest_sha256 = sha256(JCS(manifest))), hash stable under key reordering; naive json.dumps and a code-point sorted() implementation each fail the vectors as negative controls"
user_decision = false

[[item]]
id = "S4-31"
title = "Pure-data graph validator with the diagnostic code registry and the conformance corpus {document, catalog fixture, expected codes} that S6's TypeScript evaluator must also pass (S4 sole owner; supersedes S6-26's corpus)"
repo = "xtrax"
size = "L"
depends_on = ["S4-06", "S4-10", "S4-30", "S4-47"]
gate = "pytest tests/test_validate.py tests/test_conformance_corpus.py: every registry code produced by at least one vector, compat and alphabet-mismatch cases, v1 upgrade documents, flipping one port-type table entry makes the corpus fail"
user_decision = false

[[item]]
id = "S4-32"
title = "Metadata-versus-snapshot agreement: the schema emitted from the unified RunSpec classes has exactly the properties S1's run_spec_fields.json marks portable, keyed by browser_key"
repo = "molxmpnn"
size = "S"
depends_on = ["S4-07", "S1-06", "S1-07", "S4-35"]
gate = "on titanix: pytest tests/run/test_xtrax_metadata_vs_snapshot.py passes, every exposed field's metadata[xtrax] dict validates against exactly the field_meta.v1.json shape published by S4-07, the S4-02 classifier re-run as a new bathos run on the unified classes reports zero unmapped exposed fields, a planted portable row with no field metadata, a planted exposed field with no row and a planted unknown metadata key each fail"
user_decision = false

[[item]]
id = "S4-33"
title = "Decision and semantics note for the portable knob document: chains_to_design versus chain_id mapping and lowering conventions, with expected outputs for the chain cases (Q2)"
repo = "aminx"
size = "S"
depends_on = ["S4-02"]
gate = "note merged recording the Q2 answer, the chain mapping rule and at least one worked input/expected-output pair per rule, which S4-41 pre-registers against"
user_decision = true

[[item]]
id = "S4-34"
title = "Release xtrax-contract and a new xtrax (pinning xtrax-contract>=X,<next minor) on the Q11 channel, carrying the S5-51 amendments (G1-G4) and the catalog_view snapshots so the first published v1 has one meaning"
repo = "xtrax"
size = "S"
depends_on = ["S4-04", "S4-09", "S4-10", "S4-11", "S4-13", "S4-14", "S4-15", "S4-30", "S4-31", "S4-44", "S4-45", "S4-47", "S5-51"]
gate = "xtrax-contract-v<ver> and v<ver> tags released on the Q11 channel; in a clean Python 3.11 venv the released xtrax-contract installs with no dependencies and imports; in a separate clean Python 3.13 venv the released xtrax installs and importlib.metadata shows an xtrax-contract version inside the range the released xtrax pins"
user_decision = true

[[item]]
id = "S4-35"
title = "MPNN repo: bump xtrax to the S4-34 release and add xtrax-contract as a dependency (changes the MPNN lock after S2-02; intentionally unordered against S2-05..S2-34, which run A0 from a pinned F' checkout)"
repo = "molxmpnn"
size = "S"
depends_on = ["S4-16", "S4-34", "S3-07"]
gate = "on titanix: uv lock --check exits 0 with the new xtrax and xtrax-contract, the lock's base orbax-checkpoint is >= 0.12 (S4-03 record method), uv run python -c 'import xtrax_contract' exits 0"
user_decision = false

[[item]]
id = "S4-36"
title = "plegadx: add xtrax-contract as a dependency (pin from the S4-34 release)"
repo = "plegadx"
size = "S"
depends_on = ["S4-34"]
gate = "in plegadx: uv lock --check exits 0 after the dependency is added and uv run python -c 'import xtrax_contract' exits 0"
user_decision = false

[[item]]
id = "S4-37"
title = "Decision: weights license text and redistribution value (allowed, gated, forbidden) for each shipped MPNN weight set, with a named reviewer (Q8)"
repo = "aminx"
size = "S"
depends_on = []
gate = "decision recorded in the aminx tree (carried across the rename by S3-07) naming license text, redistribution value and one reviewer per weight set, at one record location that S5-07 cites from catalog/licenses/<model>.review.toml and S5-06 reads (the hub holds no second decision)"
user_decision = true

[[item]]
id = "S4-38"
title = "MPNN repo: Node/wasm cell of the split parity chain on xtrax-route artifacts under a pre-registered sidecar with inherited bounds"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-17"]
gate = "bth record shows tokens exact and the inherited bounds met under Node wasm, planted-perturbation controls still detected, verified by record not exit code"
user_decision = false

[[item]]
id = "S4-39"
title = "MPNN repo: headless-Chromium cell of the split parity chain on xtrax-route artifacts under a pre-registered sidecar with inherited bounds"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-17"]
gate = "bth record shows tokens exact and the inherited bounds met in headless Chromium, planted-perturbation controls still detected, verified by record not exit code"
user_decision = false

[[item]]
id = "S4-40"
title = "Portable MPNN knob document as a projection of the unified RunSpec with field metadata, and the pure library lowering to bias, fixed_mask, fixed_tokens and tie_group_map"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-33", "S4-07", "S4-32", "S4-35", "S1-05", "S1-07"]
gate = "on titanix: pytest tests/contract/test_knob_document.py passes, the ten browser keys round-trip, simple lowering cases equal hand-computed arrays, a planted wrong key mapping fails"
user_decision = false

[[item]]
id = "S4-41"
title = "Golden vectors from the Python lowering and JS kernel cross-check under a pre-registered sidecar (planted shuffle off-by-one and chain-semantics swap as negative controls)"
repo = "molxmpnn"
size = "M"
depends_on = ["S4-40"]
gate = "sidecar committed first, then on titanix bth run of the lowering gate: Python lowering equals every golden vector, node-run JS kernels equal the same vectors, the script twin agrees on shared cases, both planted faults fail, per-case results persisted"
user_decision = false

[[item]]
id = "S4-42"
title = "Retire the p07_knobs_gate script twin (build_p07_inputs) once S4-41 agrees on the shared cases"
repo = "molxmpnn"
size = "S"
depends_on = ["S4-41"]
gate = "rg build_p07_inputs over the repo is empty and the S4-41 gate re-run passes unchanged"
user_decision = false

[[item]]
id = "S4-44"
title = "HostPrepGraph v2 (live): optional callable_ref plus ref and params, builtin-node payload, optional edge ports/id/kind, GraphDoc conversion, callable-to-py-ref serialisation, eager py: resolution on load, schema_version=1 writer option, and the audit of the 17 a11 files that mention callable_ref or HostPrepGraph"
repo = "xtrax"
size = "L"
depends_on = ["S4-10"]
gate = "on titanix or CI: pytest tests/composition tests/cli tests/loop tests/run tests/stages: v1 fixtures load, live callable serialises to py:mod:qualname, lambda/closure/partial each raise GraphSerializationError naming the node, py: refs resolve eagerly on load, schema_version=1 writer works only for representable graphs, the audit test lists all 17 a11 files with a callable_ref=None case per use site and fails on an unlisted new use, contract nl_description constant equals node_metadata_schema.toml"
user_decision = false

[[item]]
id = "S4-45"
title = "emit_ir_schema becomes a composer: contract graph-ir.v2 plus xtrax-side defs (BundleSchema, AxisSpec, AxisOverride, AxisBoundary, Fuse, Tap, Sink) built by schemagen with jax handlers, same $id, drift tests extended"
repo = "xtrax"
size = "M"
depends_on = ["S4-08", "S4-09", "S4-44"]
gate = "on titanix or CI: uv run pytest tests/inference/test_ir_schema.py tests/audit/test_ir_schema_single_source.py: composed Node and Edge defs equal the contract's after JCS, xtrax-side defs still emitted, $id unchanged from a11, an ast test finds no hand-written Node or Edge literal in emit_ir_schema, planted divergence in the contract Node schema fails"
user_decision = false

[[item]]
id = "S4-48"
title = "proteinsmc adapter over the discovered scorer: structure passed as path or PdbText, alphex tokens mapped to the manifest port symbols by symbol, declared direction applied (minimize negated for a maximising consumer), make_mpnn_score and the mpnn registry entry changed per the Q9 decision, fitness sign convention anchored with path:line"
repo = "proteinsmc"
size = "M"
depends_on = ["S4-22"]
gate = "uv run pytest tests/scoring/test_mpnn_adapter.py with a fake scorer: path and PdbText reach bind unchanged, unknown alphex symbol raises, permuted destination ordering changes the mapped tokens, lower NLL gives higher fitness, an adapter skipping the negation fails and an adapter skipping the token map fails, make_mpnn_score follows the Q9 decision"
user_decision = true

[[item]]
id = "S4-49"
title = "Decision (Q9): supersede S3-18 with S4-22 or sequence S4-22 after it, and the fate of the public make_mpnn_score signature and mpnn registry key"
repo = "proteinsmc"
size = "S"
depends_on = []
gate = "decision recorded in proteinsmc .praxia/docs/decisions and linked from S3 and S4: supersede-or-sequence answer for S3-18 (if sequence, S3-18 is added to S4-22 depends_on in the same change), make_mpnn_score signature answer"
user_decision = true

[[item]]
id = "S4-43"
title = "Spike (pre-registered bathos): convert_to_onnx on the four real MPNN graphs at one bucket, count external-data tensors in the in-memory ModelProto (settles A22), planted external-data fixture as negative control"
repo = "aminx"
size = "S"
depends_on = ["S4-03"]
gate = "bth record on titanix with evaluated outcome: external-data tensor count per real graph (0 means S4-46 closes as not needed), one graph per process, planted external-data ModelProto makes the assertion and the outcome fail, verified by record not exit code"
user_decision = false

[[item]]
id = "S4-46"
title = "Conditional on the S4-43 record: port aminx's embed_external_data step into xtrax.export.bundle (closed with the S4-43 run id cited if no external data was found)"
repo = "xtrax"
size = "S"
depends_on = ["S4-43"]
gate = "if S4-43 found external data: pytest tests/export/test_embed.py shows a planted external-data graph is self-contained after the embed and loads in ORT; otherwise the item is closed citing the S4-43 run id with no code change"
user_decision = false

[[item]]
id = "S4-47"
title = "Param-profile engine: stdlib profile_validate.py and hand-written ts/profile_validate.mjs for the xtrax-param-profile@1 keyword subset, shared conformance/param-profile.v1.json corpus, emission refuses keywords outside the subset"
repo = "xtrax"
size = "M"
depends_on = ["S4-04"]
gate = "pytest packages/xtrax-contract/tests/test_profile_validate.py and node --test packages/xtrax-contract/ts/test both pass over the shared corpus; a planted keyword outside the subset is refused in both languages and a planted wrong expected-error-path corpus case fails both"
user_decision = false
```

## Revision log

### Round 1 (adversarial objections C1-C18)

Applied (conceded or partial, in full unless noted):

- **C1** S4-12 now depends on S4-13 (`check_ports`).
- **C2** S4-09 depends on S4-10; section 4.4 states the four schemas it emits (`model-manifest.v1`, `port-types.v1`, `param-profile.v1`, `graph-ir.v2`); the stale-file control covers the graph schema.
- **C3** Added S4-34 (release `xtrax-contract` and `xtrax`, user decision, Q11), S4-35 (MPNN repo pin) and S4-36 (plegadx pin); every consumer item that imports `xtrax_contract` (S4-17, 19, 20, 21, 24, 32, 40) now depends on a pin item. Section 4.1 states the release and pin policy; ledger A47.
- **C4** S4 is sole owner of the corpus, codes, JCS vectors, `builtin:` scheme and `required`/`cardinality`; S6-25/S6-26 listed as superseded with the re-points in section 6 ("Edges owed by other specs"); "owned jointly" removed from S4-30; ledger A48.
- **C5, C6** The first discovery design (path value read with `dist.locate_file`) is REFUTED (A42: `locate_file` is relative to site-packages; A43: setuptools rejects non-`module:attr` values, read in the vendored validator). Replaced by a bare top-level package name located with `find_spec` (A44, verified: no import for a dotless name) plus `manifests/<ep.name>.json`; the fallback is pre-stated (4.8) and the editable case (A45, UNVERIFIED, deferred) is gated in S4-11, S4-21, S4-23 with a built-wheel check against setuptools and a path-valued negative control.
- **C7** S4-19 depends on S4-18, S4-38, S4-39 and gates on each `bathos_id` resolving; `release-asset:` placeholder URL policy (4.3, S5-52 rewrites, Q12).
- **C8** S4-33 is now the small decision/semantics note; knob document and lowering split to S4-40, golden vectors and JS cross-check (pre-registered) to S4-41, script-twin retirement to S4-42; S4-20 depends on S4-41. S4-33's id and user_decision flag are kept.
- **C9 (partial)** S4-17 scoped to `p07_split_export.py` only and the nine-site claim dropped (4.10, 4.11); S4-18 split per executor (S4-18 ORT-CPU, S4-38 Node wasm, S4-39 Chromium). S4-10 re-sized M to L (17 files plus schema, upgrade path and RunSpec pin).
- **C10 (partial)** Type-handler hook and `tuple_mode` added to schemagen (4.7); `Callable` added to the S4-07 gate; S4-02 reports fixed-arity IR fields and S4-08's gate relaxes only if one exists.
- **C11** Callable-to-`py:` serialisation rule reusing v1's round-trip check, errors for lambdas/closures/partials, eager `py:` resolution on load, `schema_version=1` writer option (4.6, S4-10 gate); ledger A49.
- **C12 (partial)** S4-01 now records Q4 and S4-11/S4-21/S4-22 depend on it; new decision item S4-37 (Q8) gates S4-19; Q9 resolved in the backlog (S4-22 has no S3-18 edge). Q10 left as is: its recommendation keeps the xtrax invariant and nothing in S4-10 needs a user answer.
- **C13** All MPNN-tree items other than the spikes are ordered after S3-07 (S4-16, S4-35 directly; S4-17, 18, 19, 20, 21, 32, 38, 39, 40 through them) and use `repo = "molxmpnn"`; the naming rule is in 4.11.
- **C14 (partial)** Edges owed by S1 (S1-06 to S4-07 and S4-34), S6, S3 and S5 are recorded in section 6; S4-side edges were already present.
- **C15** `wire.py` imports no array library; numpy conversion moves to `xtrax.contract.numpy_codec`; planted function-level `import numpy` control in S4-04 (A46).
- **C16** AC3 narrowed to the S4-owned corpus, vectors and Python evaluator; S6-04 cited as the TS-side gate.
- **C17 (partial)** A33 and A12 read and VERIFIED; S4-02 now has an evaluated `[outcomes]`.
- **C18 (partial)** S4-22 trimmed to depend on S4-01 and S4-06 only; S4-24 depends on the plegadx pin, retitled as code (adapter) and AC8 reworded; D6 gate extended to `import_module` and pyproject dependency tables.

Declined, with reason:

- **C9, S4-04 re-size or split:** one coherent gate (package skeleton, table, compat, wire) and each part is small; left at M.
- **C17, A20 (safetensors layout) stays deferred:** the layout is in a compiled extension with no readable reference in the installed package; the S4-04 gate checks it against the real writer. A18 stays pre-registered as S4-02.
- **C18, S4-27 depends on S4-05:** kept. The upstream request cites the `npm` executor entry shape that S4-05 defines, so the edge has a reason; it costs nothing because S4-05 is small.
- **C12, Q10:** no user_decision flag added (see above).

### Round 2 (adversarial objections R2-1 to R2-16)

The objector read a stale xtrax checkout; every xtrax claim below was re-read at `v0.4.0a11`
(ledger A50 to A52, A57) before it was applied.

Applied (conceded or partial):

- **R2-1** The contract's `graph-ir.v2` is the only Node/Edge definition; `emit_ir_schema` becomes a composer that adds the jax-bound `$defs` (new S4-45). `audit_ir_schema_single_source` and `test_real_src_tree_is_clean` exist at a11 (A50); S4-09 extends them to the contract package with a two-entry allow-list, and the S4-01 ADR amends the fork-13 note. `node_metadata_schema.toml` stays in `xtrax.composition`; the contract enforces only `nl_description` (`E_NODE_METADATA`) and S4-44 asserts it equals the toml (4.6, A57).
- **R2-2** S4-04 now owns workspace membership, the hatchling sub-dist (version source, `requires-python >=3.11`), xtrax's dependency plus workspace source, and a `uv lock --check` gate; the tag and version policy is in the S4-01 ADR (4.1, A52).
- **R2-3** Python-side conventions added to 4.9: `structure` is a path or `PdbText`, the consumer maps alphex tokens by symbol (proteinsmc depends on alphex, A54), direction is read from the manifest and tested with a sign-flip control, and the fate of `make_mpnn_score` is a stated user decision. S4-22 is now M (discovery) and the adapter is new S4-48 (M). The SMC weight use of fitness was not traced, so A55 is UNVERIFIED and S4-48 begins by anchoring it.
- **R2-4** Q9 is no longer baked in: new decision item S4-49, S4-22 depends on it and carries `user_decision = true`, S3-18 is owed a dependency on S4-49, and the S4-22 gate requires exactly one surviving guard.
- **R2-5** New pre-registered spike S4-43 (A22 on the four real graphs, planted external-data negative control) and conditional S4-46 (port `embed_external_data`); S4-14 and S4-17 depend on them.
- **R2-6** S4-10 is split: S4-10 keeps its id and is the contract-side wire format plus the RunSpec pin (M); S4-44 is the live graph, serialisation and the audit of the 17 a11 files (listed in 4.11 from a `grep -l`, A51); S4-45 is the composer.
- **R2-7** Toolchain decided in the ADR text: stdlib generation, Node check-only, one hand-written engine per language for the fixed profile subset as its own item S4-47 with a shared corpus; S4-20 reuses the generator and engine. S4-09 re-sized M to L.
- **R2-8 (partial)** S4-02 states its target table (in the sidecar), its refs, that xtrax is read by `ast` at a pinned ref, and that its claim covers pre-flip classes; S4-07 asserts table equality; S4-32 re-runs the classifier post-S1-07.
- **R2-9** S4-33 and S4-37 now `repo = "aminx"` (notes carried by S3-07); rule in 4.11.
- **R2-10** S4-34 gate split into a 3.11 contract-only check and a 3.13 xtrax check.
- **R2-11** S4-17 carries the planted external-data negative control (shared with S4-43).
- **R2-12** S4-30 vectors add UTF-16 code-unit key order, lone-surrogate and NaN rejection, with a code-point `sorted()` implementation as a second negative control (A56).
- **R2-13 (partial)** Port-name grammar and the `x-plegadx` carrier defined in 4.2 and the S4-24 gate (A53); verbatim `shape_spec`; "equals" defined.
- **R2-14 (partial)** Edge rows added for S3-20's leaf set (S3-28 can compute it) and for re-pointing the ORT-Web comparison from S4-18 to S4-38/S4-39. The S3 and S5 item numbers are as read in the drafts; the owning specs confirm.
- **R2-15** The proteinsmc reader accepts exactly `schema_version == 1` and a known `role`, skipping with a warning otherwise; fixtures for 0, 1, 2 and missing in S4-22.
- **R2-16** Gates for S4-12, S4-35 and S4-36 name runnable commands.

Declined, with reason:

- **R2-6, separate RunSpec-pin item:** the pin stays in S4-10. S1 cites "S4-10 pins them" (ledger A41), and with the graph work moved out S4-10 is small enough to carry it.
- **R2-8, make S4-02 depend on the S1 flip:** S4-02 is a cheap early spike that feeds S4-07; instead its claim is narrowed and S4-32 repeats it after S1-07.
- **R2-4, only flipping the flag:** a flag alone does not order S3-18 against S4-22, so a decision item and an edge were added.
- **R2-2 and R2-6 sizes:** S4-04 stays M (the wiring is a few config files); S4-44 is sized L because the 17-file audit is the bulk of it.

### Coherence round 1 (cross-spec fixes C5 to C14, r3)

Applied to this spec only; the edges owed to other specs are in section 6.

- **C5** S4-22's D6 check scans `[project.dependencies]` and `[project.optional-dependencies]` and exempts `[dependency-groups]` (dev-only, alphex D4); S4-23 owns a non-default dependency group (wheel built from the checkout, plus an editable install) and no longer relies on S3-18; S4-22 depends on S3-18 only when S4-49 answers "sequence" (S4-49's record adds the edge); Q9 and 4.9 describe S3-18 as the root-only guard. S4-49 stays the user decision (recommendation: supersede).
- **C6** Section 6 "Edges owed" S3 row now defines "leaf" and lists S4-20, S4-21, S4-42 (S4-18, S4-32, S4-38, S4-39 are dependencies; S4-48 is a proteinsmc item); also asks S3-05/S3-07 to depend on S2-02 so the xtrax version in the environment stamp is fixed.
- **C7** S4-16 retitled "Q3 resolution on top of merged #174 (dependency group, xtrax[export] out of base)", `depends_on = ["S4-03", "S3-07", "S2-02"]`; "rebased on the rename" and "land the pin" removed.
- **C8** Source mode (4.3, 4.5) permits a `release-asset:` `driver.url` and a pending `driver.sha256` (driver only; a pending artifact hash is rejected in both modes); S4-05, S4-06 and S4-19 gate on it; S4-19 yields no final `onnx` entry; S5-14 fills the hash and S5-52 rewrites placeholders and validates in release mode (edge owed to S5).
- **C9** S5-51 recorded as an accepted inbound amendment (3.5, 4.3, section 6, Q13): additive within `v1` before the first release, S4-34 depends on S5-51, S4-09 commits the `catalog_view` snapshot corpus, "S5 does not reimplement `catalog_view`" deleted (S5-04 re-derives in Node against the snapshots).
- **C10** S4-07 publishes `field_meta.v1.json` (twelve keys) before S1-06 consumes it; `static=True` in the example is illustrative; S4-32 validates exactly that shape; the S1 edge owed is S1-06 on S4-07 with a plain dict under `metadata["xtrax"]`.
- **C11** S4-37, Q8 and the Provides table state one decision, one reviewer, one record location, consumed by S5-07 and S5-06 (edge owed to S5).
- **C12** S4-01 owns the id grammar `<namespace>/<family>.<op>`, the entry-point stem rule and the `manifests/` directory; "names set by S3 / come from S3" deleted; the `MolxmpnnIdentity` row is replaced by `S3:molxmpnn-repo` and `S3:molxmpnn-manifest-namespace`.
- **C13** S4-27 retired (id not reused); Q6, the AF3 risk row and A29 defer to S5-25, S5-23 and S5-26; the iframe/embed option is removed; `localfold` leaves `owner_repos`.
- **C14** S4-19's gate resolves `version` and every `module:symbol` against the S3 release-train table entry of the tag released from; S4 publishes no molxmpnn version.

### Coherence round 2 (cross-spec fixes CH2-02 to CH2-08)

Applied to this spec only.

- **CH2-02** S4-07's gate and the section 7 row state that S1-06 consumes `field_meta.v1.json` from this item; the owed-edge row (section 6) records why S1's decline leaves a gap (prose-asserted shape caught only at S4-32 after S1-07) and that the edge is acyclic, so S4-02 (pre-flip) precedes S1-07 by construction. No TOML change.
- **CH2-03** S4-16 `depends_on` gains S3-13 (it changes base dependencies and `uv.lock`, so S4-35, S4-17..S4-21, S4-32 and S4-40..S4-42 follow the release); the section 5 risk row names S3-13 for the dependency change; the S4-16 gate row says "after the S3-13 release".
- **CH2-04** Section 5 row and the S4-16/S4-35 titles state that these items change the MPNN lock after S2-02 and are intentionally unordered against S2-05..S2-34, safe only because S2 runs A0 from a pinned F' checkout; no dependency edge is added; S2's environment stamp should add `orbax-checkpoint`.
- **CH2-05** Section 4.3 defines `manifest_sha256 = sha256(JCS(manifest))` (separate from the release-asset byte hash); S4-30 gains a manifest-hash vector; an edge owed to S5/S6 gives the Node builder and editor the JS canonicaliser (S6-35) by `depends_on`.
- **CH2-08** Section 4.5 and S4-19 state that the MPNN `remote` executor entry is added by S5-21, S4-19's gate validates only `python` and `onnx`, and source-mode validation accepts a `remote` entry without a `driver`.

### Convergence check (R2-C1 residue from the S5 review; applied to this spec only)

- **Amendment narrowed to G1-G4.** The hub's G5 (`grid.meaning` in `compat`) is withdrawn: 4.2 already defines `energy-map` (alias `gfe-grid`), `count-grid` and `density-grid`, and compat rule 2 rejects `count-grid` to `energy-map`. The hub's G6 (the `catalog_view` snapshot corpus) is S4-09's own deliverable, not an amendment. Edited: 3.5 S5 row, 4.3 amendment paragraph, section 5 risk row, section 6 Provides row, section 6 S5 re-point row, Q13, S4-34 title and gate row. No `depends_on` changed.
- **Pin rule made consistent with the S5 spec.** 4.3 and the section 6 S5 row now say S5-24, S5-42 and S5-45 (through S5-64) pin the S4-34 release, while S5-04 deliberately builds against the S4-09 commit (it needs the schemas and the `catalog_view` corpus, not the amendments) and S5-64 re-pins it. Before this edit the text listed S5-04 among the S4-34 pinners, which its `depends_on` (S4-06, S4-09) never did.

### Convergence check 2 (S6 objection C2-1 residue; applied to this spec only)

- **S6-35, not S6-03, runs the TypeScript side of the JCS vectors.** Section 5's JCS number-formatting risk row said "S6-03 runs the TS side"; S6 files the TypeScript canonicaliser as S6-35 (S6-03 is the document envelope). Corrected the row. The section 6 "Edges owed" rows already name S6-35. No `depends_on` or id changed.
- **For S6's reviewers:** this spec grew while S6 re-derived its citations (1854 to 1866 lines during the pass); S6 now cites this file by section and quoted text with line numbers as hints only.
