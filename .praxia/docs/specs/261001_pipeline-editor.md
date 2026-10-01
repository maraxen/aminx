---
title: "S6 Pipeline editor: shared node-graph component (hub + praxis + JupyterLite blocks)"
description: "A framework-neutral custom-element node-graph editor (list view first, DAG canvas second, praxis adoption gated) over a versioned pipeline document whose hub graph is S4's graph IR v2 verbatim, with a TypeScript evaluator of S4's compat rules, a catalog-generated palette (one node per manifest) and Pyodide code-block nodes."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - aminx-hub
  - pipecanvas
  - praxis
related_specs:
  - S1
  - S4
  - S5
---

# S6: Pipeline editor (shared node-graph component)

**Status: DRAFT. SPEC ONLY. Nothing here has been built or run.** Working name for the
component is `pipecanvas` (open question Q1); every tag/package name below is a placeholder
that lives in one constants module so a rename is mechanical. `owner_repos` lists `aminx-hub` (renamed by
S5-33; items say `aminx-hub` for the repo as it is called today), `pipecanvas` (the repo the user creates in
S6-34, populated by S6-20) and `praxis`. S6 lands no xtrax work: round 2 dropped the two xtrax-side items
(S6-25, S6-26) because S4-06/S4-10/S4-30/S4-31 already own that territory (Revision log).

Round 1 re-anchored the document on a graph IR shape; **round 2 found that shape was not the one in S4's
file** and rebuilt every S4 anchor from S4's current text (`.praxia/docs/specs/261001_xtrax-model-contract.md`,
cited below as "S4 file"). S4 and S5 are concurrent drafts and may move again; every cross-spec line citation
is as of this revision and the coherence pass re-checks them. Convergence check: every S4 citation was re-derived against
the S4 file at 1866 lines and now names the S4 section plus a verbatim quote, with the line only as a hint. Cross-spec ids that S5 names are listed in A49.
S5 line citations (`261001_aminx-hub.md:NNN`, `S5 file :NNN`) were NOT re-derived in that check: S5 has grown to about 1600 lines since they were taken, so treat them as leads located by the quoted text, not as current line numbers, until the coherence pass re-derives them (Revision log, convergence-check).

Evidence files (read-only probes captured during spec writing, all under
`.praxia/spikes/261001_pipeline-editor/`): `evidence.txt` (exact source lines copied from xtrax
`v0.4.0a11`, praxis and aminx), `probe_libs.tsv` (npm registry / code-host REST metadata and the two
Pyodide lock files shipped inside praxis), `probe_misc.tsv` (PyPI wheel tags for jax/jaxlib/onnxruntime
and the praxis graph fixture shape), `probe_rete_jcs.tsv` (type definitions shipped in the Rete tarballs,
ECMAScript-vs-Python number formatting, praxis zone configuration), `recheck_rete_socket_r1.txt` (rete `Socket`
class body and a NOHIT count, A13) and `recheck_praxis_fixtures_r1.txt` (the 28 praxis graph fixtures,
A37/A38). They are throwaway probes with their outputs persisted, **not** `spike-run` records: the
`scripts/loop/adversarial_metrics.py` harness does not exist in this worktree, and `npm`/`node` are not
installed on this machine (only `bun`, used for the number-format probe), so every claim about how a JS
library *behaves at runtime* is `deferred` to the pre-registered bake-off (S6-08, run by S6-36); claims about
what a package *ships* (type definitions, licenses, peers) were read from the tarballs and registry. Round 2
read the S4 and S5 spec files directly and cites them by line (no new probe files were needed).

## 1. Goal and non-goals

### Goal

Give the aminx hub (S5) and, later and conditionally, praxis a **visual pipeline editor** in the
Blender-geometry-nodes / Unreal-blueprints / Max-for-Live idiom: typed ports, wires, a searchable
palette generated from model manifests (S4), per-node parameter inspectors, run-state overlays, and
code-block nodes. It is one reusable component built as **web components over a framework-free core**,
because the hub is vanilla ES modules and praxis is Angular 21 (A6, A7).

Delivered in three stages, each shippable alone:

1. **Stage 1, ordered step list.** A document editor where each step picks its inputs from earlier
   steps (default: the previous step). Linear by default, DAG-expressive by construction (a step may
   pick any earlier output, several steps may pick the same one), no canvas, accessible, works on a
   phone.
2. **Stage 2, DAG canvas.** The same document on a pan/zoom canvas with typed connect, groups,
   comments, executor/transfer overlays. Adds spatial editing, not new expressiveness.
3. **Stage 3, praxis adoption**, gated on the hub editor having real use (S6-19), and starting
   read-only.

### Non-goals

- **Not an execution engine, not the Pipelines page.** The editor never runs anything, and S6 does not
  own the hub page that embeds it (S5-31), the pipeline runner and its `HubExecutionClient` adapter (S5-30) or the
  remote-consent screen (S5-20). Execution is S5's executors behind an `ExecutionClient` port (4.6). The editor renders plans,
  privacy descriptors and run-state.
- **Not a model of lab semantics.** Praxis shares the canvas core, the document envelope and the
  type-engine; it does **not** share semantics (physical resources, time, irreversibility). Those live
  in a praxis domain pack (4.9).
- **No JAX in the browser.** Code-block nodes run Pyodide; JAX is not available there (4.7, A14, A15).
  This spec does not try to change that.

- **No new hub graph IR.** For the hub domain the document's `graph` member is S4's graph IR v2
  (`xtrax_contract.graphdoc`, S4 file section 4.6) and nothing else, byte for byte; the editor adds only a `layout`
  member and namespaced `extensions` outside the hashed graph (4.2). S6 asks S4 for **no** schema change: S4 already
  has `ref`, edge `id`/`kind`, the `models` lock, `builtin:code.python`, and port `required`/`cardinality` (A42,
  A43). The praxis domain has its own pack-defined `graph` schema (4.9); that is a separate domain, not a fork of
  the hub IR.

- **No server.** Sharing is by file or URL fragment; a gallery service is out of scope.
- **No backend for praxis authoring.** Round-tripping an edited graph back into praxis Python protocol
  code is a research problem and is excluded; praxis stage is read-only (4.9, Q6).
- **No telemetry.** The hub is static; "real use" evidence for the praxis gate is collected by hand
  (S6-19), never by instrumenting users.

## 2. Terms

- **Document**: one `*.pipeline.json` file (4.2). **Graph**: its executable, layout-free part.
  **Layout**: ignorable view state.
- **ref**: S4's node identity field. `<manifest id>@<version>` for a model or tool node (for example
  `molxmpnn/proteinmpnn.sample@0.2.0a4`), or `builtin:code.python`, or `py:<module:symbol>` (Python-only, refused by
  the hub). One manifest is one node type, so the palette has one entry per manifest (A43). The manifest hash comes
  from the graph's optional `models` lock table (`{"<id>@<version>": sha256(JCS(manifest))}`). The core treats
  `ref` as an opaque string resolved by the catalog.
- **Edge id**: S4's edges carry their own `id` (`e_` prefix) and a `kind` (always `data` in the hub). Layout,
  diagnostics and transfers address edges by that id.
- **Domain pack**: the object that tells the generic core what port types, edge kinds and extra rules
  a domain has (4.9). Hub pack is derived from S4/S5; praxis pack is praxis-owned.
- **Executor**: S5's unit that runs nodes (ORT-Web wasm, ORT-Web WebGPU, Pyodide worker, remote...).

## 3. Current state

### 3.1 What exists (anchored)

- **xtrax graph IR is not usable as the editor model yet.** `HostPrepGraph` has `nodes` and `edges`;
  `GraphEdge` is just `src: str, dst: str` node ids, **no ports**; each node carries
  `callable_ref`, a live Python callable serialised as `module.path:symbol`; the document is
  `{schema_version, nodes, edges}` with `GRAPH_SCHEMA_VERSION = 1` and a loader that rejects newer
  versions (`evidence.txt:4,10-11,20,27,29`; xtrax `v0.4.0a11:src/xtrax/composition/graph.py:59-63`,
  `serialize.py:139-145,175`). Typed ports and a browser-bindable node identity therefore have to be
  added by S4 (A1, A2). Each node also has a required `nl_description` metadata slot
  (`v0.4.0a11:src/xtrax/composition/node_metadata_schema.toml:11-14`), which maps naturally to palette
  descriptions.
- **xtrax proves ONNX only on ORT CPU.** "That is evidence about ORT CPU only: the same file on ORT Web
  (wasm or WebGPU) is not executed here" (`evidence.txt:35-37`). The editor must therefore not show an
  unqualified "verified" badge for browser executors derived from S4's manifest alone (A4).
- **Hub is vanilla ES modules with an import map, no bundler** (`evidence.txt:91-97`,
  `site/index.html:25-29`, `site/app.js:1`). A component for the hub must ship as prebuilt ESM loadable
  from a `<script type="module">` / import map with no build step in the consumer (A6).
- **A model "node" is not one graph.** The P07 sampler was split into encoder/decoder graphs with the
  autoregressive loop in JavaScript (`evidence.txt:108`; `browser/aminx-sampler/split_loop.mjs`). The
  editor therefore treats a node as an opaque unit whose internal orchestration belongs to the
  executor (A16).
- **No SharedArrayBuffer on the current host**: the hub's own page comment says GitHub Pages cannot set
  COOP/COEP headers (`evidence.txt:88-90`), so cross-worker value movement must be transfer-based (A17).
- **Praxis front end**: Angular `^21.0.0` and `pyodide ^0.29.0` are direct dependencies; a
  `jupyterlite:build` script and a `@jupyterlite/pyodide-kernel-extension` config already exist
  (`evidence.txt:77-82`). No node/graph editor exists (brief; agent-reported, not re-verified beyond the
  feature directory listing). The production build configures **no zone.js polyfill** and no
  `provideZone*` (`angular.json` has zero `polyfills` entries; `zone.js` appears only in
  `src/test-setup.ts`; `probe_rete_jcs.tsv:29-30`), so production is zoneless while the unit-test
  environment loads zone.js: a custom element's events must write signals, and the praxis gate must test
  both configurations (A8).
- **Praxis already has a graph, and it is not a dataflow DAG.** `ProtocolComputationGraph` is *derived
  from protocol Python source* (`build_graph`, `evidence.txt:86`), holds `OperationNode`s with node types
  `static/dynamic/conditional/foreach/region`, loop bodies (`foreach_body`) and branch lists
  (`true_branch`), plus resource nodes, state preconditions (e.g. `tips_loaded`) and a total
  `execution_order` (`evidence.txt:46-67`). Praxis scheduling carries `estimated_duration_ms`,
  `max_retries` and asset reservations with `released_at` (`evidence.txt:71-75`). A real fixture exists:
  4 operations, `execution_order = [op_1..op_4]`, `has_loops = False` (`probe_misc.tsv:7`). Conclusion:
  praxis *code* is the source of truth, its graph is a derived artefact, and its semantics (resources,
  time, preconditions, structured control flow) are not hub dataflow semantics (A18, A19).

- **The S5 spec fixes the hub-side shapes S6 must fit** (re-read at S5's current state; line numbers below are
  `.praxia/docs/specs/261001_aminx-hub.md`; A28-A30, A47-A50, A52-A55). Hub toolchain is Node ESM, bundler-free; its only
  Python is a stdlib-only bathos shim with a dev-only `pyproject.toml` whose dependencies are an allow-list, and S6
  "arrives as a prebuilt ES module / custom element" (`:277-285`). The `Executor` contract has `probe`, `privacy`,
  `load` and an async-generator `run(session, inputs, {signal, unitKey, trusted})` yielding
  `progress | unit | frame | done` (`:352-364`); `plan(document, availability, prefs)` is S5-37 and the consent hook is
  minted by the shell (`:375-390`). **S5 owns `HubExecutionClient`** (written in S5-30, which must pass S6's
  `runExecutionClientContract`); S6 owns the `ExecutionClient` port and the contract suite (S6-07, published by
  S6-11) (`:391-396`, item `:1368-1374`). The catalog artifact is `dist/catalog/catalog.json` with `availability[]`
  per executor, not `index.json` (`:306-310`, `:313-332`, S5-04 at `:963-969`). The Pyodide executor gains a
  `code.python` capability in S5-43 (import allow-list from the pinned lock, `trusted` flag, sandbox worker with
  `connect-src 'none'`, canary negative test; `:484-489`, item `:1188-1194`), and S5 names the Trust UI as S6-17.
  S5 owns the pipeline runner and the `/pipelines/` page (S5-30, S5-31). The bathos shim check for node/Playwright
  cells is S5-59, not S5-02 (`:945-950`), and bathos studies run locally or on titanix under slug `aminx-hub`,
  never in CI (`:787-795`). S5 binds `S6:graph-document` to S6-03, `S6:pipeline-editor-component` to S6-10 and
  `S6:jupyterlite-blocks` to S6-18 (`:755-758`), and S5-30 already depends on S6-03, S6-07 and S6-11, so those
  item ids and meanings are frozen in this revision.
- **S4's contract as it stands in its file (re-read in round 2, line anchors re-derived at the convergence check
  against S4 at 1866 lines; it supersedes the round-1 rows A24, A34, A36).** Citations to the S4 file are by section
  heading and a verbatim quote; the `:NNN` after a quote is only a hint as of that check, because S4 and S5 are
  drafts that keep growing. One manifest is one node type ("**One manifest is one node type**", S4 4.3, hint `:528`);
  ports live at manifest level with `required`, `cardinality` and `accepts_encodings` (S4 4.3 manifest shape,
  `"required": true, "cardinality": "one", "accepts_encodings": [...]`, hint `:543-544`, and "`ports[].required` and
  `cardinality` (`one` or `many`) are the fields S6 needs", hint `:603-604`). A graph v2 node is addressed by
  `ref: "<manifest id>@<version>"`; `builtin:code.python` carries `source`, `ports` and `requirements` at node
  level; edges carry `id`, `src {node, port}`, `dst {node, port}` and `kind`; the optional `models` lock is
  `{"<id>@<version>": sha256(JCS(manifest))}` (S4 4.6 "Graph IR v2", hint `:693-709`; `ref` grammar hint `:711-713`;
  "`x-` prefixed keys are preserved verbatim", hint `:737-738`). `compat` is `exact | widen | reject` with an alphabet
  rule, symbolic-dimension unification is **reserved, not implemented**, and an encoding mismatch is the warning
  `W_ENCODING_ADAPTER`, never a silent conversion (S4 4.2 "`compat(src, dst) -> exact | widen | reject`", hint
  `:489-503`; the warning code is in S4 4.6 "Validation", hint `:744-755`). Params are the `xtrax-param-profile@1`
  keyword subset plus `x-xtrax` hints (S4 4.4, hint `:633-637`). The manifest hash is "`manifest_sha256 =
  sha256(JCS(manifest))`" (S4 4.3 "Manifest hash", hint `:611`) and the graph hash is "`graph_sha256 =
  sha256(JCS(graph))`" (S4 4.6 "Canonical hash", hint `:740-743`). S4 owns the JCS implementation and hash vectors
  (S4-30, item hint `:1596-1602`) and the validator, diagnostic code registry and conformance corpus that S6's
  TypeScript evaluator must also pass (S4-31, item hint `:1605-1611`; S4 4.6 "Validation", hint `:744-755`; data files
  `conformance/canonical/jcs.v1.json` and `conformance/graph-validate.v1.json` in the S4 4.1 layout, hint
  `:397-398`). S4 lists "UI hint vocabulary and S6's evaluator" as S6's, with no item dependency (S4 6 "Consumes"
  table, hint `:1176`). S4 records S6-25 and S6-26 as superseded and lists the S6 items to re-point (S6-03, S6-04,
  S6-16) to S4-06, S4-10, S4-30 and S4-31 (S4 6 "Edges owed by other specs", S6 row, hint `:1192`; S4 3.5, hint
  `:265`), which this revision does.

- **Pyodide contents (what code blocks can import).** Both Pyodide lock files shipped inside praxis
  contain numpy, scipy, pandas, scikit-learn, biopython, ml-dtypes, sympy, pydantic, and **neither
  contains jax, jaxlib, onnxruntime or torch**; they differ in versions (numpy 2.2.5 vs 2.4.3, Python
  3.13.2 vs 3.14.0), and S5 pins its own Pyodide (314.x), so the allowed-import list is derived from the
  hub's own lock, never from these two
  (`probe_libs.tsv:24-53`) (A14). PyPI hosts no wasm/emscripten-tagged wheel for `jaxlib` in any release
  (`probe_misc.tsv:3-4`) (A15).

### 3.2 Library evaluation (evidence: `probe_libs.tsv`, probed 2026-10-01)

Criteria: (L) license, (F) framework coupling, (T) typed sockets, (M) maintenance, (R) rendering model.
`n/v` = not verified here (needs `npm install`; deferred to S6-36, the run of the S6-08 pre-registered bake-off).

| Candidate | L | F | T | M (latest publish / repo push) | R | Verdict |
|---|---|---|---|---|---|---|
| **Rete.js v2** (`rete` 2.0.6) | MIT (`:1`) | Core has no framework peers; render plugins for Lit (`rete-lit-plugin` 2.0.3, peers are rete only) and Angular (`rete-angular-plugin` 2.7.1, `@angular/core >=12 <23`) (`:1,4,5`) | No socket-level compatibility hook in the shipped type definitions; `rete-connection-plugin` 2.0.5 exposes a `canMakeConnection(from, to)` veto in its classic flow (`probe_rete_jcs.tsv:1,3`; A13, A26); pre-commit timing and drag highlighting **n/v** (A27) | core 2025-06-30 (15 months), area plugin 2026-07-08, repo pushed 2026-09-27 (`:1,2,20`); Lit plugin last published 2023-08-21 (`:4`) | Pluggable: DOM render plugins | **Provisional pick.** Own thin render plugin over a vanilla/Lit element; keep behind the `CanvasEngine` adapter |
| **@xyflow/system** (+ own renderer) | MIT (`:10`) | Framework-agnostic (no peers), but `0.0.83`, pre-1.0 API; the React/Svelte layers carry react / svelte peers (`:8,9`) | `isValidConnection`-style callback in React Flow: **n/v** | very active (pushed 2026-09-29, 38.5k stars) (`:10,21`) | Vanilla core gives drag/zoom/handle math; we write the node/edge renderer | **Bake-off challenger**: most control, most code |
| **@antv/x6** | MIT (`:15`) | No framework peers, deps `dom-align, lodash-es, mousetrap` (`:15`) | port groups and connection validation: **n/v** | 3.1.8 published 2026-08-11 (`:15,23`) | SVG + HTML nodes | **Bake-off challenger** |
| **LiteGraph** (`litegraph.js` 0.7.18; `@comfyorg/litegraph` 0.17.2) | MIT both packages; **ComfyUI_frontend itself is GPL-3.0** (`:6,7,18,19,22`) | None; canvas-drawn | Native typed slots (from prior knowledge, **n/v**) | upstream npm last 2024-01-08, repo 2024-08-01 (`:6,22`); the Comfy-Org fork was last published 2025-08-06 and its repo pushed 2026-01-14 (`:7,19`) | Canvas 2D: widgets are canvas-drawn, so text inputs and a11y are weak | **Excluded as primary**: canvas widgets conflict with DOM inspectors and a11y; GPL monorepo adjacency is a licensing trap |
| React Flow / Svelte Flow / Vue Flow / BaklavaJS | MIT (`:8,9,11,12`) | Hard framework coupling (react / svelte / vue peers) | yes | active | framework DOM | **Excluded**: would force a framework runtime into the hub or a custom-element wrapper around a framework |
| Drawflow | MIT (`:13`) | None | no typed ports | last published 2024-09-03, `0.0.60` (`:13`) | DOM | **Excluded**: no typed ports, thin maintenance |
| JointJS core | **MPL-2.0** (`:14`) | None | yes | active | SVG | **Excluded**: file-level copyleft on a library we would modify, and the editor affordances are in the commercial tier (prior knowledge, n/v) |
| Blockly | Apache-2.0 (`:16`) | None | typed connections | active | block stack | **Excluded**: expression-tree metaphor, one output per block, no fan-out DAG |

Provisional pick and why: Rete.js v2 has a **framework-neutral core**, first-party Lit and Angular
render plugins (so the praxis stage has a native path if the custom-element route disappoints), a
plugin model that lets our own core own the document, and an active repo. Its weaknesses (core last
published 15 months ago; bus factor unknown; Lit plugin stale) are exactly why the pick is
**behind an adapter and confirmed by a pre-registered bake-off (pre-registered in S6-08, run in S6-36) and a user decision (S6-09)**.

### 3.3 Assumption Ledger

Format: five cells; evidence items separated by `;`. `probe_libs.tsv` is shortened to `libs`,
`probe_misc.tsv` to `misc`, `evidence.txt` to `ev` (all under `.praxia/spikes/261001_pipeline-editor/`).

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | xtrax `HostPrepGraph` edges carry only `src`/`dst` node ids, no ports (v0.4.0a11) | If ports already existed, S6 would reuse them and S4-02 shrinks | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:10-11 |
| A2 | xtrax node identity is a live Python callable serialised as `module:symbol`, which a browser executor cannot bind, so the editor graph cannot be `HostPrepGraph` as-is | If a manifest-id route existed, S6 would consume it directly | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:4; read: .praxia/spikes/261001_pipeline-editor/evidence.txt:20 |
| A3 | xtrax's graph loader rejects a newer `schema_version` (the version-gate discipline S6 copies for its own document) | none (S6 adopts the discipline regardless; this only anchors the prior art) | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:29-31 |
| A4 | xtrax executes the `onnx` target only on ORT CPU; ORT Web wasm/WebGPU execution is not evidenced by S4's current manifests | Executor badges could be shown straight from S4 verification level | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:35-37 |
| A5 | ORT Web WebGPU has no int64 support (asserted by an xtrax docstring about a third party) | none (capability table is S5-owned; S6 renders what the capability probe returns) | UNVERIFIED | deferred: third-party behaviour asserted only in a docstring, S5-owned capability probe is the authority |
| A6 | The hub is plain ES modules + import map with no bundler, so the component must ship as prebuilt ESM | If a bundler existed, the component could ship as source | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:91-97 |
| A7 | Praxis front end is Angular ^21 with `pyodide ^0.29.0` as a direct dependency and a JupyterLite build script/config already present | Praxis stage would need to add Pyodide/JupyterLite plumbing | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:77-82 |
| A8 | Praxis's production build configures no zone.js polyfill and no `provideZone*` (zone.js appears only in the unit-test setup), so custom-element events must write signals and the gate must test both prod and test configurations | If prod were zone-based, wrapper would need `NgZone.run` and tests would match prod | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_rete_jcs.tsv:29-30 |
| A9 | Rete core is MIT with no framework peers, and has MIT Lit and Angular render plugins; core last published 2025-06-30, Lit plugin 2023-08-21 | Pick would move to a challenger | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:1; read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:4-5 |
| A10 | `@xyflow/react` and `@xyflow/svelte` carry react/svelte peers; `@xyflow/system` has no peers but is `0.0.x` | A framework-free xyflow route would be mature enough to pick by default | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:8-10 |
| A11 | ComfyUI_frontend is GPL-3.0 while `jagenjo/litegraph.js` and `Comfy-Org/litegraph.js` are MIT; upstream litegraph npm was last published 2024-01-08 | LiteGraph would be a safer primary candidate | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:6-7; read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:18-19; read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:22 |
| A12 | The Comfy-Org litegraph fork has been folded into the GPL ComfyUI_frontend repo | none (LiteGraph is not the primary candidate; S6-08 reads the exact tarball LICENSE if it is ever reconsidered) | UNVERIFIED | deferred: only matters if LiteGraph is reconsidered; the repo push date alone does not establish it |
| A13 | Rete v2 `Socket` objects expose a compatibility hook usable for connect-time typed checking | If true, per-socket compat objects could carry the type check | REFUTED | read: .praxia/spikes/261001_pipeline-editor/recheck_rete_socket_r1.txt:8-15 (the whole `Socket` class body is `name` plus a constructor); read: .praxia/spikes/261001_pipeline-editor/recheck_rete_socket_r1.txt:16-17 (explicit NOHIT: 0 occurrences of `isCompatibleWith` in `package/_types/*` of rete 2.0.6; r1 C14); changed: type compatibility is evaluated by our core and applied through the engine adapter's connection veto (A26), not by Socket objects |
| A26 | `rete-connection-plugin` 2.0.5 exposes a `canMakeConnection(from, to)` veto in its classic flow (type-level existence) | The adapter would have to intercept connections some other way | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_rete_jcs.tsv:3 |
| A27 | That veto runs before the edge is committed and supports hover highlighting while dragging | Adapter wraps pointer events itself for highlighting; rubric G1 may fail Rete | UNVERIFIED | deferred: needs the library running in a browser, barred here; resolved by S6-36 gate G1 |
| A14 | Neither Pyodide lock shipped in praxis lists jax, jaxlib, onnxruntime or torch; both list numpy, scipy, pandas, scikit-learn, biopython | Code blocks could offer more libraries and the lint allow-list changes | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:24-30; read: .praxia/spikes/261001_pipeline-editor/probe_libs.tsv:39-46 |
| A15 | No wasm/emscripten-tagged `jaxlib` wheel exists on PyPI in any release, so micropip cannot install JAX into Pyodide from PyPI | If a wasm jaxlib appeared, the lock-driven lint auto-relaxes; no schema change | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_misc.tsv:3-4 |
| A16 | A browser model node is a JS-orchestrated multi-graph unit (encoder/decoder split, AR loop in JS), so nodes must be opaque to the editor | Editor could expose sub-graph structure | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:108 |
| A17 | Cross-executor value movement in the browser uses transferable ArrayBuffers (correct with or without SharedArrayBuffer); the hub's current host cannot set COOP/COEP | none (correctness does not depend on it; only a zero-copy optimisation would) | UNVERIFIED | deferred: platform behaviour asserted in a page comment; S5-owned executor design, S6 only renders the plan |
| A18 | Praxis's graph is derived from Python source with loops, conditionals, preconditions and a total execution order, so its source of truth is code, not the graph | Praxis could be an authoring target and stage 3 would not need to be read-only | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:46-67; read: .praxia/spikes/261001_pipeline-editor/evidence.txt:86 |
| A19 | Praxis scheduling models time and physical asset reservations (estimated duration, retries, reservation release), i.e. semantics a hub dataflow DAG lacks | Sharing semantics with praxis would be cheaper than assumed | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/evidence.txt:71-75 |
| A20 | A real praxis graph fixture exists (4 operations, ordered, no loops) usable as adapter test data | S6-21 would have to generate fixtures through the extractor | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_misc.tsv:7 |
| A21 | A custom element with property binding and event listeners works inside Angular 21 under `CUSTOM_ELEMENTS_SCHEMA` | A small Angular wrapper package becomes mandatory instead of optional | UNVERIFIED | deferred: needs an Angular build (barred here); resolved by S6-22 gate |
| A22 | The chosen canvas engine works inside a shadow root (pointer capture, CSS, focus) | Use light DOM with namespaced CSS, or pick another engine | UNVERIFIED | deferred: needs a browser + npm install; resolved by S6-36 gate G2 |
| A23 | A JCS (RFC 8785) implementation in Python agrees with the TypeScript one on the number forms that appear in params (floats like 0.1, 1e-5, large ints) | Hash keys diverge across languages; params floats would need string-encoded decimals | UNVERIFIED | deferred: needs running both serializers (barred here); the Python side and the vectors are S4-30's, the TypeScript side is S6-35, resolved by S6-35 running S4-30's vector file (see A56) |
| A25 | Python's default float serialisation (`json.dumps`) is not RFC 8785 / ECMAScript number formatting (e.g. `1e-5`, `1e16`, `1.0`), so a real JCS implementation is required on the Python side | A plain `json.dumps` would suffice for hashing | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/probe_rete_jcs.tsv:20; read: .praxia/spikes/261001_pipeline-editor/probe_rete_jcs.tsv:23; read: .praxia/spikes/261001_pipeline-editor/probe_rete_jcs.tsv:25 |
| A24 | (Round-1 row, withdrawn.) S4's graph IR v2 is `node_type` + flat port-addressed edges (`src`, `src_port`, `dst`, `dst_port`) + a `models` lock table, with no edge id, no edge `kind`, no `ref@version` and no inline code node | If S4's IR had edge ids, `kind`, `ref` and a code node, S6 would use them directly and S6-25 shrinks | REFUTED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:693-713 (S4 4.6 "Graph IR v2" shape block and `ref` grammar; S4's file has `ref`, edge `id`/`kind`, `models` as `{"<id>@<ver>": sha}`, `builtin:code.python` with `source`/`ports`/`requirements`; `node_type` appears nowhere in the file); changed: 4.2 embeds that shape verbatim (A42), the derived `edgeKey` and `extensions["x-pipecanvas"].executor_hints` are removed, and S6-25 is dropped (r2 C2-1) |
| A36 | (Round-1 row, withdrawn.) S4 provides no conformance corpus and no JCS/hash vectors | If S4 already shipped them, S6-26 would be a no-op | REFUTED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:1596-1602 (S4-30: JCS plus shared hash vectors); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1192 (S4 6 "Edges owed by other specs", S6 row: S4 records S6-25 and S6-26 as superseded and asks S6 to re-point); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1605-1611 (S4-31: validator, code registry, conformance corpus S6's TS evaluator must pass); changed: S6-26 is dropped; S6-03/S6-35 and S6-04 depend on S4-30 and S4-31 (r2 C2-1) |
| A28 | The S5 hub is Node ESM and bundler-free and consumes S6 as a prebuilt ES module / custom element | If the hub required source-level integration, packaging changes | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:277-285 |
| A29 | S5's `Executor.run` is an async generator yielding progress/unit/frame/done with one `unit` per independently reproducible piece, and each executor exposes `privacy()` | The run-state mapping in 4.6 changes | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:352-364 |
| A30 | S5 specifies a `code.python` capability on the Pyodide executor (S5-43): allow-list from the pinned lock, `trusted` flag, sandbox worker with `connect-src 'none'`, canary negative test | If S5-43 were absent S6-28 would need an S6-owned request | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:484-489; read: .praxia/docs/specs/261001_aminx-hub.md:1188-1194 |
| A31 | (Round-1 row, withdrawn.) S6 owns the `ExecutionClient` port and the `HubExecutionClient` adapter (S6-11) | S6 would need its own hub-integration items | REFUTED | read: .praxia/docs/specs/261001_aminx-hub.md:391-396 (S5 states the adapter is S5's, and that recording S6-11 as the adapter was wrong); read: .praxia/docs/specs/261001_aminx-hub.md:1368-1374 (S5-30 writes it and passes the S6 suite); changed: S5-30 owns `HubExecutionClient` (A47); S6-11 becomes the fake-client demo plus published contract suite; S6-28/S6-29 re-point to S5-30 (r2 C2-2) |
| A32 | Code-node network egress can be blocked inside the Pyodide sandbox worker (script served with `connect-src 'none'`) | Code nodes run only in documents authored in the current session (Q4 fallback) | UNVERIFIED | deferred: S5 itself keeps its own row (A26) unverified and gates it with a negative test in S5-43; needs a real browser; S6-28 re-checks it end to end through the editor |
| A34 | (Round-1 row, withdrawn.) S4's operation inputs carry `name`, `port`, `accepts_encodings`, `dims` but not `required` or `cardinality`, so S6 must request them | If S4 carried `required`/`cardinality`, the request is dead | REFUTED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:543-544 (S4 4.3 manifest shape: manifest ports carry `required`, `cardinality`, `accepts_encodings`); read: .praxia/docs/specs/261001_xtrax-model-contract.md:603-604 (S4 4.3: "`ports[].required` and `cardinality` (`one` or `many`) are the fields S6 needs"); changed: no request to S4; the palette reads `required`/`cardinality` from the manifest port, there is no `operations[]` and no fallback default (r2 C2-1) |
| A35 | (Round-1 row, partly withdrawn.) S5-02 is the Playwright harness "with pinned Chromium and sidecar shim check" | A depends_on id points at the wrong item | REFUTED | read: .praxia/docs/specs/261001_aminx-hub.md:927-935 (S5-02's gate is the header-controlled dev server with `crossOriginIsolated` true/false under three prefixes, no sidecar check); read: .praxia/docs/specs/261001_aminx-hub.md:945-950 (the shim check is S5-59); changed: S6-08 and S6-27 depend on S5-59 as well as S5-02 (r2 C2-3); the other roles in this row are re-verified in A49 |
| A37 | `ProtocolComputationGraph` has `operations`, `resources`, `preconditions`, `execution_order` and no dataflow edges or ports; loop/branch containment is `foreach_body`/`true_branch`/`false_branch` lists of operation ids on the operation | A pack-defined `graph` would need no invented edges | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/recheck_praxis_fixtures_r1.txt:34 (models.py lines cited there: 504-511, 525-569, 572-596, 623-671) |
| A38 | There are exactly 28 praxis graph fixtures matching `plr-sema/tests/fixtures/*_graph.json`, each with `operations` and `execution_order`, every contained child id resolves to an `operations` entry, and `execution_order` is a subset of `operations` (it omits nested nodes in 12 of 28 files) | The S6-21/S6-22 fixture predicate and count rule change | VERIFIED | read: .praxia/spikes/261001_pipeline-editor/recheck_praxis_fixtures_r1.txt:4-31; read: .praxia/spikes/261001_pipeline-editor/recheck_praxis_fixtures_r1.txt:32-33; changed: the node-count rule is the number of `operations` entries (regions and bodies included), not the length of `execution_order` |
| A39 | A deflate+base64url fragment of up to 32 KiB survives the targets we care about (browser address bar, issue trackers, chat clients) without truncation | A smaller cap or file-only sharing for large documents | UNVERIFIED | deferred: client-specific URL handling (chat clients, link shorteners) cannot be probed from this machine; the cap is a constant, the typed error and the file-download fallback make a wrong value harmless |
| A40 | (Round-1 row, withdrawn.) A stdlib-only RFC 8785 serializer written by S6 in Python can match ECMAScript number formatting for the number forms in `params` | S6-26 falls back to string-encoded decimals | REFUTED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:1596-1602 (the Python serializer and vectors are S4-30's, with S4's own ledger row for it); changed: S6 writes no Python serializer; the TypeScript side is S6-35 and the open question is A56 |
| A41 | titanix (or another single named host) can run the pinned-Chromium Playwright harness for frame-time measurements | The perf budget runs on the local workstation and the host is recorded in the sidecar | UNVERIFIED | deferred: S5 keeps the same question open (S5 A27) and titanix browser setup was not probed; resolved by S6-36/S6-27 recording the host |
| A42 | S4's graph IR v2 nodes are `{id, ref, params, executor_hint, metadata, frozen}` with `ref` = `<manifest id>@<version>` or `builtin:<name>` or `py:<module:symbol>`; edges are `{id, src {node, port}, dst {node, port}, kind}`; the optional `models` table is `{"<id>@<version>": sha256(JCS(manifest))}`; `builtin:code.python` carries `source`, `ports {in, out}` and `requirements` at node level; `x-` keys are preserved and any other unknown key is an error | The document example (4.2), edge identity, spec keys and the lossless-round-trip rule change | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:693-713 (S4 4.6 "Graph IR v2" shape block and `ref` grammar); read: .praxia/docs/specs/261001_xtrax-model-contract.md:737-738 (S4 4.6 "Unknown keys: `x-` prefixed keys are preserved verbatim"); read: .praxia/docs/specs/261001_xtrax-model-contract.md:611 (S4 4.3 "Manifest hash", the `models` value form) |
| A43 | One S4 manifest is one node type; ports are at manifest level and carry `name`, `direction`, `type`, `required`, `cardinality`, `accepts_encodings`, optional `alphabet` and `shape`; a multi-operation model ships several manifests sharing a `family` | The palette would need an operations layer and a `required`/`cardinality` fallback | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:528-532 (S4 4.3 "One manifest is one node type", `family`); read: .praxia/docs/specs/261001_xtrax-model-contract.md:543-546 (S4 4.3 manifest shape, ports); read: .praxia/docs/specs/261001_xtrax-model-contract.md:603-604 (S4 4.3 `required`/`cardinality`) |
| A44 | S4-30 (JCS dumps, sha256 and the shared vector file, S4 the sole owner) and S4-31 (pure-data validator, diagnostic code registry, conformance corpus `{document, catalog fixture, expected codes}` that S6's TypeScript evaluator must also pass, with a mutation control on the type table) exist; S4-31 depends on S4-06, S4-10, S4-30, S4-47; S4-30's vector file includes a manifest-hash vector | S6 would have to author the corpus and vectors itself | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:1596-1602 (S4-30); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1605-1611 (S4-31, `depends_on`); read: .praxia/docs/specs/261001_xtrax-model-contract.md:744-755 (S4 4.6 "Validation", corpus and codes); read: .praxia/docs/specs/261001_xtrax-model-contract.md:611 (S4 4.3 "Manifest hash"); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1192 (S4 6 "Edges owed by other specs", S6 row) |
| A45 | S4's `compat` is `exact \| widen \| reject` with an alphabet rule (two `sequence` ports with different alphabets must not connect); symbolic-dimension unification is reserved (evaluator v1 checks concrete dimensions only, adding it is a corpus version bump); an encoding mismatch is the warning `W_ENCODING_ADAPTER`; `W_NO_EXECUTOR` and `W_TRANSFER_LARGE` belong to S6's pack, not S4's corpus | The type engine would have accepts lists, R3 dimension unification and an `E_DIM_MISMATCH` code | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:489-503 (S4 4.2 "`compat(src, dst) -> exact \| widen \| reject`", alphabet rule, reserved dimension unification, encoding warning); read: .praxia/docs/specs/261001_xtrax-model-contract.md:744-755 (S4 4.6 "Validation": `W_ENCODING_ADAPTER`, `W_NO_EXECUTOR` and `W_TRANSFER_LARGE` belong to S6's pack) |
| A46 | S4's validator takes the ports of a `builtin:code.python` node from its node-level `ports` table (entries shaped like manifest ports) and S4-31's corpus contains a code-node vector | A code node would be rejected as `E_UNKNOWN_PORT`, or S6 has to carry its own non-normative vector | UNVERIFIED | deferred: S4 shows the node shape (S4 4.6 "Graph IR v2", `:700-702`, `"ports": {"in": [], "out": []}`; re-checked at the convergence check) but states neither the port-entry fields nor the validator's treatment of `builtin:` nodes, and S4-31 is unbuilt; resolved by the S6-16 gate, which reads S4-31's corpus and files the request in Q14 if the vector is absent |
| A47 | S5 owns `HubExecutionClient` (S5-30, passing S6's `runExecutionClientContract`); S6 owns the `ExecutionClient` port, types and the contract suite (S6-07), and S6-11 publishes the suite with a fake client; S5-30 `depends_on` already lists S6-03, S6-07 and S6-11 | S6 would carry an adapter item and a dependency on S5-30 | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:391-396; read: .praxia/docs/specs/261001_aminx-hub.md:1368-1374; read: .praxia/docs/specs/261001_aminx-hub.md:756 |
| A48 | The bathos shim check for node/Playwright cells is S5-59 (depends on S5-02 and S5-03); S5's studies run under slug `aminx-hub` with `bth run --project-slug aminx-hub -- uv run --no-sync python3 scripts/studies/<study>.py`, locally or on titanix, never in CI; S5's own A17 (a Python wrapper is needed to resolve the sidecar) is still UNVERIFIED there and S5-59 is its one-run check | S6's bake-off could not rely on the shim, and its sidecar might resolve to nothing | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:945-950; read: .praxia/docs/specs/261001_aminx-hub.md:787-795; read: .praxia/docs/specs/261001_aminx-hub.md:224 |
| A49 | S5 ids with the roles S6 binds: S5-01 repo skeleton and CI with the dependency allow-list guard; S5-02 Playwright harness; S5-03 `hub.toml` and layered path resolvers; S5-04 catalog builder; S5-08 Executor core and PrivacyDescriptor; S5-10 RunStore; S5-17 Pyodide executor; S5-30 runner plus `HubExecutionClient`; S5-31 Pipelines page; S5-33 first public deployment; S5-37 planner and consent; S5-43 `code.python`; S5-59 bathos shim check | A depends_on id points at the wrong item | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:909-912; read: .praxia/docs/specs/261001_aminx-hub.md:927-937; read: .praxia/docs/specs/261001_aminx-hub.md:945-946; read: .praxia/docs/specs/261001_aminx-hub.md:963-964; read: .praxia/docs/specs/261001_aminx-hub.md:1008-1009; read: .praxia/docs/specs/261001_aminx-hub.md:1035-1036; read: .praxia/docs/specs/261001_aminx-hub.md:1044-1045; read: .praxia/docs/specs/261001_aminx-hub.md:1170-1171; read: .praxia/docs/specs/261001_aminx-hub.md:1188-1189; read: .praxia/docs/specs/261001_aminx-hub.md:1368-1369; read: .praxia/docs/specs/261001_aminx-hub.md:1377-1381; read: .praxia/docs/specs/261001_aminx-hub.md:1395-1401 |
| A50 | S5 emits `dist/catalog/catalog.json` (plus `dist/catalog/models/<id>.json`), byte-deterministic, with `availability[]` per executor carrying the probe `requires` and an evidence level; there is no `index.json`; run-time probe results come from S5-08's capability probe, not from the catalog | The palette input name and the source of `available`/`reasons` change | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:306-310; read: .praxia/docs/specs/261001_aminx-hub.md:322; read: .praxia/docs/specs/261001_aminx-hub.md:352-358 |
| A51 | S5-04's catalog record passes through the fields the palette needs that its mapping table does not list: the manifest `params` schema, `description`, port `accepts_encodings` and `alphabet`, and the manifest hash for the `models` lock, which is the S4 form sha256(JCS(manifest)), not the sha256 of the release asset bytes that S5's `catalog/sources.toml` pins | The palette has no form model, no encoding warning and no lock entry for model nodes; S6-33 must amend the builder | UNVERIFIED | deferred: S5's mapping table lists ports as (name, direction, type, dtype, shape, required, cardinality) and no params, description, `accepts_encodings`, `alphabet` or manifest hash (`:313-332`); S5 names the hash its wire protocol echoes as `manifest_sha256` (`:513-518`, S5-19) without saying which hash it is, so the JCS form is the contract S6-33 states and S5 must confirm (Q18); S5-04 and S4's `catalog_view` are unbuilt, so S6-33 is the item whose gate settles it (it is a no-op when the fixture already carries the fields) |
| A52 | S5-01's `no-model-imports` guard checks the dependencies of the dev-only Python `pyproject.toml` against `hub-python-allow.toml` (initially empty), so a package with its own npm `package.json` and no Python is outside it, and adding a `[tool.*]` table to that pyproject adds no dependency | S6-02 may fail S5-01's CI, and an npm allow-list policy becomes an S5-01 change request | UNVERIFIED | deferred: the guard is unbuilt and S5's text (`:263`, `:279-283`) does not say whether `package.json` manifests are scanned; resolved by the S6-02 gate (S5-01's guard passes with the package present), with the policy filed as Q17 |
| A53 | Number-producing hub studies are Python drivers under the hub's `scripts/studies/` using only the stdlib, spawning the node/Playwright cells, so `script_path` resolves to a `.py` and `<stem>.bth.toml` resolves; never a `bash -c` or `npx` wrapper | The bake-off driver would need its own pyproject and allow-list entries | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:787-795; read: .praxia/docs/specs/261001_aminx-hub.md:279-283 |
| A54 | S5-03 defines `hub.toml` with a `[paths]` table and a derived-data path table (`study_dir`, `build_dir`, `browsers_dir`, `e2e_dir`) resolving explicit argument > environment variable > `hub.toml [paths]` > XDG config > fail loudly, with a malformed file raising | S6 would need a resolver of its own with no hub layer | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:936-941 |
| A55 | Before extraction the hub can consume the package's built bundle as a hash-pinned vendored input (a `vendor.lock.json` entry whose source is the workspace build), although S5's vendor lock currently lists py2Dmol, ORT-Web, Pyodide and coi-serviceworker as tag-plus-sha256 entries | S5-31 would need the bundle committed to the hub, or the extraction would have to precede the Pipelines page | UNVERIFIED | deferred: S5 names S6 as "a prebuilt ES module" (`:284-285`) and lists the vendor lock content (`:274`) but does not describe a workspace source; filed as Q17, settled by S6-02's gate (the build emits `dist/` plus `checksums.txt`) and the first consumer, S5-31 |
| A56 | A TypeScript RFC 8785 implementation reproduces S4-30's vectors for the number forms in `params` (0.1, 1e-5, 1e16, 1.0, 2^53 integers), because RFC 8785 number serialization is ECMAScript's | S6-35 hand-writes number formatting, or float params must be string-encoded decimals (S4 decides) | UNVERIFIED | deferred: needs running the vector file in a JS engine (the barred step is installing the package toolchain; the bun-based number-format probe at `probe_rete_jcs.tsv:20,23,25` shows ECMAScript and Python formatting differ, which is why Python needs a real JCS); resolved by the S6-35 gate |

## 4. Design

### 4.1 Packages and where it lives

One path, used everywhere in this spec: the package root is `packages/pipecanvas/` inside the hub repo
(working name), and it is the repository root after the S6-20 subtree split. Paths below are relative to it.

```
packages/pipecanvas/              (package root; working name; becomes the repo root of `pipecanvas` at S6-20)
  package.json                    devDependencies live here (typescript, bundler, test tooling, engine candidates)
  core/                           @pipecanvas/core      no DOM, no framework; runs in Node/worker/browser
    src/document/                 schema.json, types.ts, migrate.ts, gate.ts          (S6-03)
                                  canonical.ts (JCS + graph_sha256), share.ts          (S6-35)
    src/types/                    TS evaluator of S4's compat over S4's generated port-type table
    src/validate/                 structural rules, diagnostics registry (codes.ts, mirrors S4's registry)
    src/palette/                  paletteFromCatalog, paramsToFormModel (xtrax-param-profile@1)
    src/commands/                 invertible edit commands, undo/redo stack, patch events
    src/run/                      RunState model, ExecutionClient port, plan view-model, spec keys, contract suite
    src/domain/                   DomainPack + GraphAccess interfaces (no implementations)
    src/code/                     allowed-imports lint (consumes an allowed-imports JSON; reads no lock itself)
  elements/                       @pipecanvas/elements  custom elements, ESM, shadow DOM
    src/pipe-list.ts              <pipe-list>       stage 1
    src/pipe-canvas.ts            <pipe-canvas>     stage 2 (CanvasEngine adapter inside)
    src/engine/                   CanvasEngine interface + the selected engine's adapter (one file set)
    src/pipe-palette.ts, pipe-inspector.ts
  angular/                        @pipecanvas/angular   OPTIONAL, created only if A21 is false
  demo/                           in-package demo page (S6-10 harness; fake client and catalog fixture in S6-11)
  fixtures/                       golden documents (v1/*.json), S5-04's emitted catalog fixture,
                                  xtrax-conformance.lock.json + pinned copies of S4's port_types.v1.json,
                                  graph-validate.v1.json (S4-31 corpus) and jcs.v1.json (S4-30 vectors)
  scripts/bakeoff/specs/          Playwright specs and scenario definitions only (no Python in the package)
  scripts/allowed_imports/        Node CLI (.mjs) that turns a pyodide-lock.json (explicit argument) into allowed-imports JSON
```

The bake-off and performance drivers (Python) are **not** in the package. They are stdlib-only scripts in the
hub's `scripts/studies/` (`pipecanvas_bakeoff.py`, `pipecanvas_perf_budget.py`, each with a stem-named
`.bth.toml`), exactly the shim shape S5 prescribes (A53), so the hub keeps its one dev-only `pyproject.toml` and
its empty `hub-python-allow.toml` (4.11, 4.12).

**Toolchain and coexistence with S5-01's guard.** TypeScript, compiled with `tsc` to ESM plus `.d.ts` (praxis is
TypeScript/Angular, so typed contracts matter to it). The hub itself stays bundler-free (A28), so the package
ships **prebuilt**: `core` as plain ESM, and `elements` as one self-contained ESM file with the canvas engine
inlined (so a consumer needs one import-map entry; gate G3 below). The bundler is a **devDependency of the
package**, runs only in the package's own build (`npm run build` in `packages/pipecanvas/`, a CI step before the
hub build), and the hub's build never invokes it. The package adds **no Python**, so the allow-list guard on the
hub's Python dependencies is untouched; whether S5-01's guard also scans npm manifests is unverified (A52). S6-02's
gate therefore runs S5-01's guard with the package present, and if the guard flags the package's
devDependencies the same PR adds them to a visible npm allow-list file, which is a **change request to S5-01**
(Q17), never a silent exemption. Before extraction the hub consumes the build output (`dist/pipecanvas-elements.js`
plus `checksums.txt`) as a hash-pinned vendored input, a `vendor.lock.json` entry whose source is the workspace
build (A55, Q17); after extraction it consumes the release tarball (S6-30).

**Home and extraction (recommendation, Q1).** Start as a workspace *inside the hub repo*
(`packages/pipecanvas/`) with an enforced boundary (core and elements may not import anything from the
hub; checked in CI with a seeded-violation negative control). Extract to its own repo only when the S6-19
checkpoint records `outcome = "go"`, via `git subtree split` to preserve history. Reasoning: a standalone repo
before there is a second consumer is exactly the premature generalization the brief warns about, while the
boundary lint keeps extraction cheap.

**Extraction is a DAG with a go-only branch.** S6-19 is a *decision* item whose result is one of go / no-go /
defer; a DAG only knows "done", so every item on the extraction branch (S6-34, S6-20, S6-21) carries the check
`check_checkpoint.py --require go` in its gate and fails on a no-go or defer record (4.10). (1) S6-34 (user-owned)
creates the empty `pipecanvas` repo and records its slug. (2) S6-20 (hub-side; the subtree split is produced in
the hub repo and pushed to the new repo) cuts a tag with a **release tarball** (built `core` ESM, the one-file
`elements` bundle, `.d.ts`, `checksums.txt` with sha256s, and the checkpoint outcome file); the tag plus the
recorded tarball sha256 is the verifiable state. (3) S6-30 migrates the hub to that release (vendored bundle with
a `vendor.lock.json` sha256 entry, import-map entry, in-repo copy deleted, CI updated). (4) S6-21/S6-22 make
praxis consume the same release tarball by URL plus sha256 (an npm tarball dependency), so no step waits on npm.
Publishing to npm is a separate optional, user-owned item (S6-32) that nothing depends on.

**Records across the move.** Bathos runs made before S6-20 (S6-36, S6-27) live in the hub repo's bathos
project (slug `aminx-hub`, A48) and are cited by run id; they are not migrated. The `--project-slug` in commands
is whatever the repo that holds the script has configured at run time, never a slug for a repo that does not
exist yet.

### 4.2 Pipeline document format (`pipecanvas.document` v1)

```jsonc
{
  "format": "pipecanvas.document",
  "schema_version": 1,                 // int. Load gate below.
  "domain": { "id": "aminx.hub", "version": 1 },   // which DomainPack interprets it
  "graph": {                           // hub domain: S4 graph IR v2, EXACTLY as the S4 file section 4.6 defines it
    "schema_version": 2,               // S4's own gate, independent of the envelope's
    "models": { "molxmpnn/proteinmpnn.sample@0.2.0a4": "<sha256(JCS(manifest)), 64 hex>" },   // optional lock table
    "nodes": [
      { "id": "n_k3f9a2qd", "ref": "molxmpnn/proteinmpnn.sample@0.2.0a4",
        "params": { "temperature": 0.1, "seed": 7 }, "executor_hint": null,
        "metadata": { "nl_description": "..." }, "frozen": false },
      { "id": "n_p0x7c1mw", "ref": "builtin:code.python",
        "source": "def run(inputs, params):\n    ...",
        "ports": { "in":  [ { "name": "seqs", "type": "sequence" } ], "out": [] },
        "requirements": ["numpy"], "metadata": { "nl_description": "..." } }
    ],
    "edges": [
      { "id": "e_81hd0z3k", "src": { "node": "n_k3f9a2qd", "port": "sequences" },
        "dst": { "node": "n_p0x7c1mw", "port": "seqs" }, "kind": "data" }
    ]
  },
  "layout": {                          // ignorable view state; never affects execution or hashes
    "order": ["n_k3f9a2qd", "n_p0x7c1mw"],       // list-view order hint (must be a valid topological order)
    "nodes": { "n_k3f9a2qd": { "x": 80, "y": 120, "w": 220, "collapsed": false, "color": null, "parent": null } },
    "viewport": { "x": 0, "y": 0, "zoom": 1 },
    "groups": [ { "id": "g_1", "title": "Design", "members": ["n_k3f9a2qd"] } ],
    "comments": [ { "id": "c_1", "x": 40, "y": 10, "text": "..." } ]
  },
  "extensions": {                      // namespaced, opaque to core, preserved verbatim, NEVER hashed
    "x-praxis": { }
  }
}
```

The `graph` member is S4's shape with nothing added and nothing renamed (A42). The shape of the code-node
port entries is taken from the manifest port (`name`, `type`, `required`, `cardinality`); S4 does not spell it out
(A46, Q14).

What S6 does with that shape:

- **Node identity** is S4's `ref`. For a model or tool node the model version and manifest hash come from
  `graph.models`; the editor writes the lock entry (`"<id>@<version>"` to the manifest hash) when a node of a new
  manifest is added, from the palette entry's manifest hash (A51, S6-33; the value is the JCS hash that the `models` lock expects, A51). If the catalog record carries no hash,
  the editor omits the optional lock (S4 allows that) and the palette marks the node "unlocked".
- **Edge identity** is S4's edge `id` (`e_` plus 8 base32 characters for edges the editor creates; any id S4
  accepts is preserved). Layout, diagnostics and plan `transfers` address edges by that id. A second edge with the same
  `(src, dst)` pair is refused by the connect command and, when loaded, reported as the editor pack warning
  `W_EDGE_DUPLICATE` (not an S4 code).
- **Edge kind**: hub edges are all `data`. A praxis `order` edge exists only in the praxis domain's own graph
  schema (4.9), never in the hub IR.
- **Executor hints** are S4's `executor_hint` node field and are passed to S5's `plan()` as `prefs`. Because the field
  is inside `graph`, it **is** part of `graph_sha256` (S4 hashes the whole graph); the consequence is that changing a
  hint changes the hash and revokes per-hash code trust (4.7), which is conservative, and is Q19. Spec keys exclude it.
- **Inline code nodes** are S4's `builtin:code.python` with `source`, `ports` and `requirements` at node level
  (4.7).
- **`metadata.nl_description`** is required by xtrax (S4 Q10); the editor prefills it from the manifest
  `description` (the palette's `description`) on insertion, and the loader never default-fills it.

Rules:

- **Load gate (copies xtrax's discipline, A3).** Envelope `schema_version` missing, non-int, older than
  `MIN_SUPPORTED`, or newer than `SCHEMA_VERSION` => refuse to load with a typed error; never default-fill.
  Migrations are pure `vN -> vN+1` functions with a golden fixture per version. The embedded `graph.schema_version`
  is gated separately: document v1 embeds graph **v2 only**, and an embedded graph with another version is refused
  with S4's `E_SCHEMA_VERSION`, because a v1 graph's nodes are untyped `py:` refs the hub refuses; upgrading v1 is
  S4's Python-side job.
- **Lossless round trip, with S4's rule inside `graph`.** Unknown fields in the envelope, `layout` (at any level)
  and `extensions` survive load/save after canonicalisation. Inside `graph`, S4 preserves `x-` prefixed keys at node and
  edge level and treats any other unknown key as an error (A42); the editor therefore keeps `x-` keys verbatim and
  refuses to load a graph with another unknown key (a typed load error naming the key). It never silently drops a
  field it does not understand.
- **Layout is separable.** Deleting `layout` yields a valid, executable document. `graph_sha256 =
  sha256(JCS(graph))` (S4 4.6 "Canonical hash", hint `:740-743`; the manifest form is S4 4.3 "Manifest hash", hint `:611`), implemented in TypeScript by S6-35 and required to reproduce S4-30's vector
  file (A44, A56). `layout` and `extensions` are outside the hash.
- **Node ids are stable and never reused** within a document (new nodes get `n_` + 8 random base32 chars; any
  id S4 accepts is preserved). Layout item ids use `g_` / `c_`. Copy/paste remaps node and edge ids; the remap is
  part of the command so undo is exact.
- **Edges are port-addressed** by S4's `src {node, port}` and `dst {node, port}`.
- **Spec keys** (owned by S6-07; a plan-time hint, **not** the reuse authority): `spec_key(node) =
  sha256(JCS({ref, params, lock: graph.models["<ref>"] (a builtin uses sha256(JCS({source, ports, requirements}))),
  inputs: sorted([[dst.port, src.port, spec_key(upstream)] for each incoming edge])}))`. The key uses the lock's
  manifest hash because the manifest hash already covers its `weights.files[]` hashes. Sorting is by content (not by
  edge id), so regenerated edge ids change no key. `executor_hint`, `metadata` and `frozen` are excluded. Layout edits
  change no key; a param edit changes that node's key and every downstream key, and no sibling or upstream key
  (S6-07 gate, with a negative control). Their use is *display and prediction* ("this node would be reused"). The
  authority for actually reusing a result is S5's value-based unit key (`sha256(manifest sha || canonical inputs ||
  unit params)`, S5 section 4.6) checked by `RunStore`; S6 never decides reuse.
- **Sharing** (owned by S6-35): file download/upload, or a URL fragment holding a deflate+base64url document.
  Limits and failure behaviour (A39): the encoded fragment is capped at `SHARE_FRAGMENT_MAX = 32768` bytes (one
  constant in `share.ts`). Encoding a larger document returns the typed error `E_SHARE_TOO_LARGE {bytes, max}` and
  the UI offers the file download instead; it never truncates. A fragment that fails to decode or inflate yields
  `E_SHARE_DECODE {stage}` shown as a visible message, and nothing is loaded; a fragment that decodes then goes
  through the load gate like a file. A document loaded from a URL or foreign file is untrusted (4.7).

**Prior art and what we deliberately copy or avoid** (prior knowledge, **not load-bearing**, n/v):
ComfyUI keeps an editor "workflow" JSON (positions, widgets) separate from an executable "prompt"
(the API graph): we copy the separation as `layout` vs `graph`. Galaxy's `.ga` workflows and KNIME's
workflow files both keep tool/node identity plus typed connections and tolerate unknown tool versions;
we copy "an unresolvable `ref` is preserved and flagged, never deleted" (S4's `E_UNKNOWN_REF`, rendered as a
"missing" placeholder node). Neither is adopted as a format because neither is executor-agnostic nor carries our
type table.

### 4.3 Typed ports and validation

S6 defines **no type model of its own**. The type model is S4's (`port_types.v1.json`, S4-04; S4 4.2
"Port vocabulary and the type table", hint `:461-503`): a declarative table of port types (`structure`, `sequence`, `logits`, `grid`, `energy-map`, ...), named
`widens_to` edges, named axes, an `alphabet` attribute on `sequence`/`logits`, and `compat(src, dst) -> exact | widen |
reject` (A45). S6 implements an *evaluator* of that rule in TypeScript over the table S4 generates and commits
(S4-09; the table is the data, the evaluator holds no type names); S4's `xtrax_contract.validate` is the Python
evaluator. Both must pass **one conformance corpus**, which S4 owns and ships (`conformance/graph-validate.v1.json`,
S4-31, A44); S6 does not author it. This is the single most important anti-drift control: the browser must never
accept a graph the executor side rejects, or the reverse. The package carries a pinned copy of the corpus and of the
port-type table (`fixtures/xtrax-conformance.lock.json` records the xtrax tag and sha256s; a refresh script takes the
source path or tag as an explicit argument), and a test fails when the pinned table differs from S4-09's generated
artifact.

`compat(srcPort, dstPort) -> { result: "exact" | "widen" | "reject", reason, code? }` is S4's rule, nothing more:

- Same type name: `dtype`, `rank`, named `axes` and, for `sequence`/`logits`, `alphabet` name must be equal where
  both are given; equal on every given attribute is `exact`; source more specific than an unspecified destination
  attribute is `widen`; source less specific than a destination that specifies it is `reject`.
- Different type names: `widen` iff the destination is reachable from the source through the table's `widens_to`
  edges (for example `energy-map` to `grid`); otherwise `reject`. An alphabet mismatch is `reject` with
  `E_ALPHABET_MISMATCH`; any other reject is `E_TYPE_MISMATCH`.
- **Dimensions.** Symbolic-dimension unification is reserved by S4 (evaluator v1 checks concrete dimensions only;
  adding it is a corpus version bump). S6 therefore implements none: there is no `E_DIM_MISMATCH` and no
  `W_DIM_UNKNOWN`. When S4 adds it, S6 re-pins the corpus and the evaluator follows.
- **Encodings.** A producer/consumer pair that shares no encoding yields the warning `W_ENCODING_ADAPTER` (S4);
  the edge is allowed and the warning is shown on it. The editor **never inserts an adapter silently** (S4); an
  explicit "insert adapter node" command is deferred (Q5).
- **Adapter suggestions** are a UI hint, not part of `compat`: when the result is `reject`, the rejection message
  lists catalog manifests with `role: adapter` whose input accepts the source type and whose output widens to the
  destination type (S4 4.3 manifest shape reserves `role: adapter` in "`scorer | sampler | predictor | energy | transform | adapter`", hint `:539`; S4 4.2 "adapter route", hint `:485-487`).

Connect-time behaviour (both elements): while dragging from an output, every input port is marked
compatible (`exact`/`widen`, with a warning chip when `W_ENCODING_ADAPTER` applies) or incompatible (`reject`) via
`compat`; dropping on an incompatible port is vetoed before the edge exists, with the reason in the
`connect-rejected` event and a live region message. In `<pipe-list>` the per-input picker only lists compatible
earlier outputs.

Validation in core returns diagnostics; none executes anything. The registry (`codes.ts`) has two parts, and only
the first is covered by S4's corpus.

| Source | Codes |
|---|---|
| **S4's registry** (evaluated in TS, must match S4-31's corpus; severities as S4 defines them) | `E_DUP_ID`, `E_EDGE_ENDPOINT`, `E_UNKNOWN_PORT`, `E_TYPE_MISMATCH`, `E_ALPHABET_MISMATCH`, `E_CARDINALITY` (a `one` input has at most one incoming edge), `E_CYCLE`, `E_REQUIRED_UNCONNECTED`, `E_UNKNOWN_REF` (document preserved, node rendered as a "missing" placeholder), `E_PARAM_SCHEMA` (params violate the manifest's `xtrax-param-profile@1`), `E_SCHEMA_VERSION`, `E_REF_GRAMMAR`, `W_ENCODING_ADAPTER` |
| **S6 pack-level** (not in S4's corpus; S4 4.6 "Validation": "`W_NO_EXECUTOR` and `W_TRANSFER_LARGE` need executor capability info and belong to S6's pack", hint `:749-750`) | `W_NO_EXECUTOR` (no available executor can run the node, from capability info), `W_TRANSFER_LARGE` (the plan moves more than `pack.transferWarnBytes` across executors), `W_EDGE_DUPLICATE`, `E_CODE_IMPORT_UNAVAILABLE{package}` (S5-43's code, 4.7) |

`E_REQUIRED_UNCONNECTED` is shown as a quiet hint while editing and as a blocker at run time; that is presentation
policy, the code and its severity are S4's. Whole-graph validation is debounced and incremental; the executor side
re-runs the authoritative S4 validator at run time.

### 4.4 Node palette generated from the catalog

S4 manifests are aggregated by S5's catalog builder (S5-04) into `dist/catalog/catalog.json` plus
`dist/catalog/models/<id>.json` (`S5:catalog-index`, A50). Each record carries `id`, `kind` (model|tool), `role`,
`family`, `version`, `title`, `ports[]` (name, direction, type, dtype, shape, `required`, `cardinality`),
`availability[]` (one entry per declared executor with the probe `requires` and an evidence level), `license`,
`evidence[]`, `limits`, `provenance`. **One manifest is one node type** (A43), so the palette has **one entry per
manifest**, keyed by `ref = "<id>@<version>"`; there is no operations layer. The fields the palette also needs and
S5's mapping table does not list (the `params` schema, `description`, port `accepts_encodings` and `alphabet`, a
manifest hash for the `models` lock) are A51 and the amendment item S6-33, which is a no-op when S5-04's emitted
fixture already carries them. S6-33 also owns the pass-through of `manifest_sha256` as sha256(JCS(manifest)), computed
with S6-35's canonicaliser (the hub's Node builder has no other way to produce the S4 form), so the value S5-19's wire
protocol echoes and the RunStore unit key use is the same hash the `models` lock holds; the sha256 of the manifest
release asset that `catalog/sources.toml` pins is a separate integrity check and is never used as a lock value.

`paletteFromCatalog(catalog, probeReport) -> PaletteEntry[]` joins the static `availability[].requires` with the
run-time capability report from S5-08's probe to compute `available` and `reasons` (the catalog is static, the
probe is not). Its snapshot test uses **S5-04's emitted fixture**, not an S6-authored one.

```
PaletteEntry { ref ("<id>@<version>" | "builtin:<name>"), modelId, version, manifestSha256?,
               title, family, kind: "model"|"tool"|"builtin", role, description, category,
               inputs:  [{ name, type, required, cardinality, acceptsEncodings, alphabet?, shape? }],
               outputs: [{ name, type, alphabet?, shape? }],
               paramsSchema? (xtrax-param-profile@1 document, resolved $refs),
               executors: [{ kind, requires, available, reasons[], privacy (PrivacyDescriptor), evidence[] }],
               license: { code, weights, redistribution }, limits?, deprecated? }
```

What S4 provides (A43, A42):

1. **Params: provided.** A manifest's `params` is a schema in `xtrax-param-profile@1` (S4 4.4, "`params` is a JSON Schema in the **xtrax-param-profile@1**", hint `:633-639`). There is
   no raw-JSON fallback for a model that has a schema. A raw JSON editor flagged "unvalidated" remains only for a
   manifest whose `params` is absent.
2. **Port `required` and `cardinality`: provided** at manifest level (S4 4.3 manifest shape, hint `:543-544`; "`ports[].required` and `cardinality` (`one` or `many`) are the fields S6 needs", hint `:603-604`). The palette copies
   them; there is no default-filling fallback (round 1 had one; A34 is withdrawn).
3. **Prose.** `description` comes from the manifest (S6-33) or, when absent, from S5's `overrides/<model>.toml`;
   the editor prefills the node's `metadata.nl_description` from it (4.2).

- The inspector form is generated from `paramsSchema` by `paramsToFormModel`, which implements **exactly the
  S4 profile**: `type`, `enum`, `const`, `minimum`, `maximum`, `items`, `prefixItems`, `properties`,
  `required`, `additionalProperties`, `oneOf` with a required discriminator (rendered as a variant selector),
  `default`, `title`, `description`, `examples`, `$ref` within the document, and the `x-xtrax` annotations
  `unit`, `group`, `advanced`, `widget`, `portable`. The editor-specific widget vocabulary (`structure-picker`,
  `residue-selection`, grouping) is S6's contribution that S4 lists as "UI hint vocabulary" (S4 6 "Consumes" table, row "UI hint vocabulary and S6's evaluator", hint `:1176`), expressed as
  additional values of `x-xtrax.widget` and documented in the S6-05 form-model README. A keyword outside the S4
  profile is a **loud, named error at palette-build time**, never a silently missing field; because S4's emitter
  already fails emission on any other keyword, a legitimate S4 manifest cannot trigger it, and the S6-05 gate proves
  that with a `oneOf`+`$ref` fixture.
- Search is local (fuzzy over title, description, family, ref) and filtered by executor availability;
  unavailable nodes are shown greyed with the probe's `reasons` (for example no WebGPU adapter), not hidden.
  Manifests with `role: adapter` appear under their own category.
- One catalog entry may list several executors (ONNX via ORT-Web, a remote runtime, a localfold WebGPU
  adapter). The palette shows one node; the **plan** picks the executor and the node shows its chip.
  Evidence badges are **per executor** from `evidence[]` and are never merged, matching S5's rule
  that "ORT CPU (xtrax)" and "ORT-Web wasm (hub gate)" stay separate (A4).
- Entries with `redistribution: gated | forbidden` render a license chip and a "bring your own weights"
  hint; the palette never hides them (hiding is a catalog decision).

### 4.5 `<pipe-*>` custom-element API (framework-neutral contract)

All elements: shadow DOM, theming through CSS custom properties and `::part()`, no global CSS, no
framework runtime in the public API, ES module build with no top-level side effects except
`customElements.define` (guarded against double registration).

| Member | Kind | Notes |
|---|---|---|
| `document` | property | set: load-gate + validate, then render; get: current document (deep copy) |
| `catalog` | property | palette entries built by `paletteFromCatalog` (host builds them from S5's `catalog.json` and the capability probe report) |
| `domain` | property | a `DomainPack` (4.9); when absent, structural rules only |
| `plan` | property | `ExecutionPlan` from the host: segments, per-segment `PrivacyDescriptor`, transfers |
| `runState` | property | `Record<nodeId, NodeRunState>` set by the host; the element only renders it |
| `codeTrusted` | property | boolean set by the host; code nodes are inert and flagged while false (4.7) |
| `allowedImports` | property | allowed-imports JSON for the code-node lint (4.7); the element never reads a lock itself; absent means the lint is off and says so |
| `readonly` | attribute | disables all edits |
| `storage-key` | attribute | if set, autosave to IndexedDB under this key; **absent means no persistence** (no default key in library code) |
| `getDocument() / validate() / undo() / redo() / fitView() / focusNode(id)` | methods | |
| `pipeline-change` | event | `{ document, patch }` (patch is the inverse-capable command record) |
| `diagnostics` | event | `{ diagnostics[] }` after each validation |
| `connect-rejected` | event | `{ from, to, code, reason }` |
| `run-requested` | event | `{ scope: "all" \| "node" \| "selection", nodeIds?, codeTrusted }`; the host decides what happens (S5's page calls its runner and, for remote segments, its own consent screen) |
| `trust-requested` | event | `{ graph_sha256 }`; the host persists trust, sets `codeTrusted` |
| `node-open` | event | `{ nodeId, port? }` (host decides: open viewer, inspector, notebook) |

Slots: `slot="body-<nodeId>"` lets the host render a custom body inside a node (for example a preview node
mounted through S5's `mountViewer`, `S5:viewer-adapter`) without the editor knowing what a viewer is.

Integration shapes: the hub's page does `import ".../pipecanvas-elements.js"` (S5 vendors it by hash like its
other third-party bundles) then property assignment and `addEventListener`; Angular uses the element under
`CUSTOM_ELEMENTS_SCHEMA` with property bindings and event handlers that write signals, tested under both zone
configurations (A8, A21). The optional `@pipecanvas/angular` wrapper (typed inputs/outputs,
`ControlValueAccessor`) is created **only if A21 proves false**.

### 4.6 Execution is delegated: the `ExecutionClient` port

S5 owns executors, the runner (S5-30), `plan()` (S5-37) and the page (S5-31). **S5 also owns the adapter**:
`HubExecutionClient` is written in S5-30 over `plan()`, the executors and `RunStore`, and must pass S6's contract
suite (S5 file `:391-396`, item `:1368-1374`; A47). **S6 owns the editor-facing port and its types**
(`ExecutionClient`, `ExecutionPlan`, `RunEvent`, `NodeRunState`) and the **contract suite**
(`runExecutionClientContract(client)`, S6-07). S6-11 publishes the suite as a package export together with a fake
client and the demo app, so S5-30 can import it; S5-30 already depends on S6-03, S6-07 and S6-11, and S6-11 does not
depend on S5-30, so the graph has no cycle. S6 therefore has no adapter item. The editor-side use of the real
adapter is S6-28 (code-node end to end) and S6-29 (canvas on the deployed page), both of which depend on S5-30.

```ts
interface ExecutionClient {
  plan(doc: Document): Promise<ExecutionPlan>;
  run(doc: Document, opts: { scope: "all"|"node"|"selection"; nodeIds?: string[]; resume?: boolean;
                             codeTrusted: boolean }): AsyncIterable<RunEvent>;
  cancel(runId: string): Promise<void>;
}
ExecutionPlan { segments: { id: string; executor: string; nodes: string[]; privacy: PrivacyDescriptor }[];
                transfers?: { edge: string /* S4 edge id */; from: string; to: string; encoding?: string;
                              estBytes?: number }[];
                consentRequired: string[] /* segment ids */;
                blockers: Diagnostic[] }
RunEvent      { type: "node-state"; nodeId: string; state: NodeRunState }
              | { type: "unit"; nodeId: string; unitKey: string; reused: boolean }
              | { type: "done" | "error"; code?: string; message?: string }
NodeRunState  { status: "idle"|"queued"|"running"|"done"|"error"|"skipped"; executor?: string;
                progress?: { done: number; total: number };
                cache?: { reused: boolean; unitKey?: string; source?: string; specKeyHit?: boolean };
                error?: { code: string; message: string };
                outputs?: Record<string, { type: PortType; bytes?: number; sha256?: string }> }
```

Mapping from S5's plan (`plan()` returns `{segments: [{id, executor_kind, nodes[], requires[]}], transfers:
[{edge, from, to, encoding}], consent_required: [segmentId]}`, S5 file `:375-378`) to this port, performed by S5's
`HubExecutionClient` (S5-30) and asserted by the contract suite: `executor_kind -> executor`, `consent_required ->
consentRequired`, and the segment's `PrivacyDescriptor` is **added** from the executor registry (S5 gives it per
executor, not per plan), so the editor never needs S5's executor classes; `transfers[].edge` is the S4 edge `id`.
Mapping from S5's `Executor.run` events (A29): `progress` -> `NodeRunState.progress`; `unit` -> a `unit` event
whose `reused` flag comes from `RunStore`'s resume report (S5-10: reuse only when input hash and artifact
hashes match); `done` -> `status: "done"`. Unit and cache keys shown in the UI are S5's value-based keys; the
spec key (4.2) is only the plan-time prediction.

**Privacy and cross-executor movement.** The editor moves no data and holds no consent logic. It makes
movement and exposure *visible*: each plan segment carries S5's `PrivacyDescriptor`, rendered as a chip on
every node in the segment (a `remote` segment reads "leaves your machine: sends the full input to
<endpoint>"); an edge whose endpoints sit in different segments renders dashed, with the `encoding` and
`estBytes` chips when the plan supplies them (`transfers` is optional in the port: S5's runner decides what it
can report); `W_TRANSFER_LARGE` fires over a pack threshold. The consent screen before the first byte is S5's
(S5-20); `run-requested` is just a request. The hub's public promise ("runs entirely in your browser",
`site/index.html` header) is therefore enforced where the data is sent, not duplicated here.

**Resume.** Preemption/refresh safety belongs to S5 (`RunStore` persists each unit before emitting it and
resume reports reused units; S5 section 4.6). The editor's side: (a) autosave the document when `storage-key`
is set, with a recovery prompt after a crash; (b) a "Resume" affordance that sets `resume: true`; (c) render
`cache.reused` nodes distinctly with the unit key's short hash, so reuse is visible, never silent.

### 4.7 Code-block nodes: Pyodide in a worker, JupyterLite as an escape hatch

- **Node contract.** S4's `ref = "builtin:code.python"` with `source`, `ports {in, out}` and `requirements` at node
  level (4.2; S4 4.6 "Graph IR v2", the `builtin:code.python` node in the shape block, hint `:700-702`; A42). Ports are declared explicitly in a table in the inspector (name, S4 port
  type, `required`, `cardinality`, direction; the entry fields follow the manifest port, which S4 does not spell out
  for code nodes, A46); the source is the body of `def run(inputs: dict, params: dict) -> dict`
  returning a dict keyed by output port. Signature inference from annotations is a later enhancement
  (it would need a Python parser in the editor).
- **Execution is S5's Pyodide executor, `code.python` capability (S5-43).** S5 specifies the fresh-namespace
  run, the `trusted` flag (an untrusted run refuses code nodes), the lock-derived allow-list and a sandbox
  worker served with `connect-src 'none'` with a canary negative test (`261001_aminx-hub.md:484-489`, item `:1188-1194`;
  A30). S6 specifies nothing about how it executes, only the node contract and the Trust UX.
- **What code can import.** The hub pins its own Pyodide (S5: 314.x). S5-43 derives the executor-side
  allow-list from that lock plus hub wheels. The editor's lint is **lock-agnostic**: `@pipecanvas/core` takes an
  *allowed-imports JSON* (`{pyodide, packages: [...], generated_from_sha256}`) through the `allowedImports`
  property and never reads a lock or a hub path (that would violate the boundary rule). The JSON is produced by
  the `scripts/allowed_imports/` CLI from a `pyodide-lock.json` given by **explicit argument** (resolution
  order for callers: `--lock` argument > `PIPECANVAS_PYODIDE_LOCK` environment variable > fail loudly; no
  default path in the package). The hub wires its real pinned lock in with S6-31, which depends on S5-17 and
  S5-43 and asserts the editor's list and S5-43's list agree on the same lock. The two locks shipped in praxis
  differ (A14) and neither is the hub's, so nothing is hard-coded. A disallowed import yields
  `E_CODE_IMPORT_UNAVAILABLE{package}` (S5-43's code, the edit-time twin of S5's run-time
  `UnavailableInBrowser{package}`).
- **JAX limitation, stated plainly.** Code blocks cannot `import jax`, `jaxlib`, `onnxruntime` or `torch`:
  none is in the Pyodide locks we can see (A14) and PyPI has no wasm/emscripten-tagged jaxlib wheel in any
  release (A15). So: (1) code blocks are for glue and analysis (numpy, scipy, pandas, scikit-learn, biopython,
  sympy, hub wheels); (2) JAX-derived models reach the browser only as exported artifacts (ONNX via ORT-Web,
  S4/S5) or run on a remote executor, always as **model nodes**, never from inside a code block; (3) the lint
  returns a named diagnostic that points to the catalog's model nodes instead of failing at run time; (4) we do
  not ship a fake `jax` shim. If a wasm jaxlib ever appears, only the lock changes. (S5's own capability probe
  CI, S5-18, re-checks this on a schedule.)
- **Data crossing.** Typed arrays and text cross the worker boundary exactly as S5 specifies for `PortValue`s
  (transferable typed arrays / text, S5 section 4.4); large-tensor paths are S5's concern and show up in
  the plan's `transfers` when S5 reports them.
- **Trust and sandbox.** A document with code nodes is arbitrary code. Rules: the editor treats a document as
  **untrusted until the host sets `codeTrusted`**; documents loaded from a URL fragment or a foreign file start
  untrusted, with code nodes rendered inert and flagged; "Trust this pipeline" emits `trust-requested
  {graph_sha256}` and the host persists trust per hash (never globally). `run-requested` always carries
  `codeTrusted`, and S5's executor must refuse code nodes when it is false (S5-43's `trusted` flag). Whether
  a trusted code node can exfiltrate an uploaded structure over the network is an **executor-sandbox
  question owned by S5**: S5-43 serves the sandbox worker with `connect-src 'none'`, which is **unverified**
  until its negative test runs (A32). Delivery is split: **S6-17** is the Trust UX against the fake client
  (no S5 dependency); **S6-28** is the end-to-end integration through the editor and `HubExecutionClient`,
  including its own canary check. If the canary cannot be made to fail, adopting the fallback "code nodes
  only run for documents authored in the current session" changes product behaviour and is a **user
  decision** (Q4), which is why S6-28 is marked `user_decision = true` and S6-17 is not.
- **JupyterLite (`S6:jupyterlite-blocks`).** Not embedded as the execution kernel (it is interactive and
  stateful and carries a large bundle). S6 provides (a) the `code.python` node contract above and (b) an
  `.ipynb` export (S6-18: one cell per node in topological order, inputs snapshotted) that opens in S5's
  `/lab/` page (S5-35), which already shares one Pyodide version with the executor. An interactive
  `<pipe-list>` widget *inside* a notebook is an optional later bridge (S6-23, Q13).

### 4.8 Staged delivery and the canvas engine

**Stage 1: `<pipe-list>`** (S6-10, with the fake-client demo in S6-11), embedded in the hub by S5-31
(S5 binds `S6:pipeline-editor-component` to S6-10; the canvas upgrade of that page is S6-29, Q10). Steps in topological order; each input port has a picker
(default: previous step's compatible output, otherwise first compatible earlier output, otherwise
"unset"); add-step opens the palette; reorder moves only among valid topological positions; delete warns
about dependants; steps can also be added by dragging an entry from the palette (S5-31's gate drags catalog
nodes). Because a step can reference any earlier step, this *is* a DAG editor without a canvas,
and it stays permanently as the accessible/mobile view of the same document. Imports of any valid
document; cycles are impossible by construction.

**Stage 2: `<pipe-canvas>`** (S6-12 to S6-15, perf budget S6-27, hub adoption S6-29) behind a `CanvasEngine` interface:

```ts
interface CanvasEngine {
  mount(host: HTMLElement, hooks: { canConnect(from: PortRef, to: PortRef): Compat;
                                    onMove, onConnect, onDisconnect, onSelect, onViewport }): void;
  setGraph(model: RenderModel): void;      // full set; diffed internally
  patch(change: RenderPatch): void;        // incremental
  setNodeBody(nodeId: string, el: HTMLElement): void;
  destroy(): void;
}
```

Only one file set (`src/engine/<name>/`) knows the library. The **bake-off** (pre-registered in S6-08, run in S6-36) picks the engine; a
user decision (S6-09) confirms. Hard gates (pass/fail, must all pass): **G1** typed connect veto fires
before the edge is committed (for Rete this means the connection plugin's `canMakeConnection`, A26, A27); **G2** works inside a shadow root with pointer interaction intact (A22);
**G3** consumable from a plain `<script type=module>`/import map with no framework runtime and no bundler
step in the consumer; **G4** the full production dependency tree is MIT/Apache-2.0/BSD/ISC (no GPL/AGPL;
MPL flagged); **G5** programmatic add/move/remove round-trips to the exact document. Soft metrics
(reported and ranked, thresholds fixed in the sidecar before the run): gzip size of the engine +
adapter, median frame time for pan/zoom on a 200-node / 400-edge fixture, adapter lines of code.
Selection rule (pre-stated): pass all G; then maintenance (latest publish, repo activity); then
performance; ties go to Rete.js v2.

**Stage 3: praxis** (S6-19 to S6-22): see 4.9.

### 4.9 Praxis crossover: share the canvas, not the semantics

What is shared: `@pipecanvas/core` mechanics (document envelope, ids, undo, layout, validation
framework, type engine), the elements, the engine adapter. What is **not** shared and stays out of core:
resources and their linearity (a plate cannot be in two places, so fan-out of a physical resource is
illegal while fan-out of data is free), time (incubation durations, the schedule's
`estimated_duration_ms`), irreversibility (no caching/memoisation, "dry-run" instead), structured control
flow (loops/branches in `ProtocolComputationGraph`), and state preconditions (`tips_loaded` etc.)
(`evidence.txt:46-75`).

Mechanism: the `DomainPack` interface. **The `graph` member's schema is pack-defined**: the hub pack's graph
is S4's IR v2; the praxis pack's graph is its own schema (below). The envelope (`format`, `schema_version`,
`domain`, `layout`, `extensions`), the load gate, ids, undo and layout are shared; the core reaches into `graph`
only through the pack's `GraphAccess`, so it never assumes S4's field names.

```ts
interface GraphAccess {                     // how core reads a pack's graph without knowing its schema
  schemaId: string;                         // hub: "xtrax.graphdoc@2"; praxis: "praxis.pcg-view@1"
  nodes(g): { id: string; type: string }[];
  edges(g): { key: string; from: { node: string; port: string }; to: { node: string; port: string };
              kind: string }[];             // hub: kind always "data"; key = the S4 edge id
  apply(g, command): g;                     // pack-owned edits (hub: S4 shapes; praxis: read-only, throws)
}
interface DomainPack {
  id: string; version: number;
  graph: GraphAccess;
  portTypes: PortTypeTable;                 // hub: S4's generated table; praxis: single "flow" port type
  edgeKinds: string[];                      // hub: ["data"]; praxis: ["order","contains","uses","satisfies"]
  compat(src, dst): Compat;                 // default: table-driven S4 compat (4.3)
  rules: Rule[];                            // (doc, ctx) => Diagnostic[]; e.g. praxis "unsatisfied precondition"
  allowCycles?: boolean; transferWarnBytes?: number;
  nodeVocabulary?(): PaletteEntry[];        // when no catalog
}
```

Praxis specifics go in `extensions["x-praxis"]` and pack-level rules. The core gets **no** praxis field.
Containment for loop/region nodes uses the optional `parent` id on `layout.nodes` entries (reserved in
v1, **no semantics in core**, rendering only).

First praxis use is **read-only visualisation** of the already-produced `ProtocolComputationGraph`
(adapter `pcgToDocument`, S6-21; host component in the protocol detail view, S6-22). Authoring is
excluded because praxis code is the source of truth and graph-to-code round-tripping is out of scope.

**`pcgToDocument` mapping (S6-21; models read at `praxis/backend/utils/plr_static_analysis/models.py`, A37).**
`ProtocolComputationGraph` has `operations`, `resources`, `preconditions` and `execution_order` and **no
dataflow edges or ports**, so the adapter derives a view and invents no data semantics. Output document:
`domain = {id: "praxis.protocol", version: 1}`, `graph.schema = "praxis.pcg-view@1"`, defined by a JSON
Schema in the praxis pack (the praxis repo owns it; it is not S4's IR):

- **Nodes.** One node per `operations[]` entry: `id = op.id`, `type = "praxis.op/" + op.node_type`
  (`static | dynamic | conditional | foreach | region`), `params = {method_name, receiver_variable,
  receiver_type, arguments, line_number, condition_expr, foreach_source, trip}`. One node per `resources[var]`:
  `id = "res:" + variable_name`, `type = "praxis.resource"`. Region headers and loop-body children are
  ordinary `praxis.op/...` nodes because they are entries of `operations[]`.
- **Edges (all with an explicit `kind`, one port named `flow` on every node).**
  `order`: for each container (the top level, and each `foreach_body`, `true_branch`, `false_branch` list)
  consecutive members in that list's own order; the top level uses `execution_order`, which is a **subset** of
  `operations[]` that omits nested nodes in 12 of the 28 fixtures (A38), so nested containers use the list
  order they carry. `contains`: container operation -> each child id, with attribute `branch` in
  `body | true | false`; the same relation sets `layout.nodes[child].parent` (render only). `uses`: operation
  -> `res:<var>` when `receiver_variable` or an `arguments` value equals the resource's `variable_name`
  (exact string match; anything else is not guessed). `satisfies`: for each precondition with a non-null
  `satisfied_by`, edge `satisfied_by` op -> every op whose `preconditions` lists that precondition id.
- **Unsatisfied preconditions** (`satisfied_by = null`) stay as data in `x-praxis` and raise the pack rule
  `W_PRECONDITION_UNSATISFIED` on the operations that list them; the adapter adds no node for them.
- **Losslessness.** The complete original PCG JSON is stored verbatim at `extensions["x-praxis"].pcg`, so no
  PCG field is dropped (unknown fields included). The derived graph is a view; nothing is read back from it.
- **Layout.** Absent in the adapter output; the deterministic auto-layout (S6-13) places nodes, so the
  document is identical for identical input.
- **Counting rule (used by S6-22).** `count(op nodes) == len(pcg.operations)`, including region headers and
  loop/branch bodies; resource nodes are excluded from the comparison and counted separately
  (`len(pcg.resources)`). `len(execution_order)` is **not** the node count.
- **Fixture predicate (used by S6-21).** The 28 files matching `plr-sema/tests/fixtures/*_graph.json` in the
  praxis repo (all have `operations` and `execution_order`; `derived_contracts_pre_increment.json` is not
  matched and is not a graph; A38). The gate enumerates by this glob and asserts the count is 28, so a new
  fixture surfaces as a deliberate test change.

Guards against premature generalization:

1. Praxis stage cannot start before S6-19 records `outcome = "go"`, a **user decision** against criteria that are
   written and committed before the checkpoint (4.10); the gates of S6-34, S6-20 and S6-21 each enforce it.
2. No core field, rule or schema change is made "for praxis" before S6-21; after it, any such change
   requires a short ADR and a hub regression run.
3. A synthetic `toy.lab` domain pack in core's tests (S6-04) proves the core takes non-hub semantics via
   rules only, so domain-neutrality is exercised early without building praxis stage.
4. The praxis pack and adapter live in the **praxis repo**, not in pipecanvas.

### 4.10 S6-19 checkpoint (proposed criteria, user sets them before the review)

Proposed, to be edited and committed *before* evaluating: (a) the Stage 2 canvas has been live on the hub's
**public** Pipelines page for at least 60 days, counted from a **recorded deploy date**; (b) at least 3 distinct
non-trivial pipelines (fan-out or fan-in) saved and re-run by the maintainer in real work, or 2 by other people
(counted by hand from shared files and issues, no telemetry); (c) no open P1 bug against the document schema
or load gate; (d) a written praxis first-use case (read-only graph view) approved. The outcome is a recorded
go / no-go / defer.

**How the time gate becomes DAG-checkable (Q10).** S5-31 embeds the stage 1 `<pipe-list>` (S6-10). The item
**S6-29** upgrades that page to `<pipe-canvas>`. It depends on **S5-33**, the first public deployment (S5 file
`:1395-1401`, gated on its own user decision), so a non-public deploy cannot start the clock. S6-29's gate
verifies against the **live origin** (the origin and `base_path` come from `hub.toml`): the served
`pipecanvas-elements.js` has a sha256 equal to the release build's `checksums.txt`, and only then does it write
`deploy_date` (the date of that passing live check) and the served sha256 into
`packages/pipecanvas/docs/checkpoint_criteria.toml`. S6-19 depends on S6-29 and S5-31 (not S6-17: the criteria do not
need code nodes).

**A DAG knows "done", not "go".** S6-19 is the *decision* item. Its gate runs `scripts/check_checkpoint.py`, which
fails unless `review_date >= deploy_date + 60 days` (a 59-day fixture is the negative control), and it writes
`docs/checkpoint_outcome.toml` with `outcome = "go" | "no-go" | "defer"`, the review date and the evidence links.
Any of the three outcomes completes S6-19. The items that must only run on a *go* (S6-34, S6-20, S6-21) each run
`check_checkpoint.py --require go` (S6-21 reads the copy of `checkpoint_outcome.toml` shipped in the release tarball)
as the first step of their gate and fail on no-go or defer, so a no-go leaves them blocked rather than executable.
The negative control for that gate is a fixture outcome file with `no-go`.

### 4.11 Accessibility, performance, storage

- Node-graph canvases are inherently hard to use with a keyboard or screen reader. The list view is the
  accessible path and stays; the canvas gets a keyboard connect flow (select output port, Enter,
  choose a compatible input from a list) and a live region (S6-14). Axe-core check on both elements.
- Performance budget (pre-registered in S6-27, a separate item from the functional canvas work in S6-12):
  200 nodes / 400 edges, pan and zoom. The thresholds are fixed in S6-27's sidecar from the S6-36 record id
  (its baseline), and S6-27 depends on S6-36 and S6-12 explicitly. **Where it runs:** frame-time numbers are
  only comparable within one host, so S6-36 and S6-27 each run on **one named host** recorded in the sidecar
  and result (host id, CPU/GPU, Chromium build from the S5-02 pin, device scale factor), and a threshold is only
  compared with a number measured on that same host. S5 runs bathos studies locally or on titanix, never in CI
  (A48); CI here runs the functional gates and the smoke of the harness, never the frame-time thresholds. Which host
  that is (titanix versus the local workstation) is Q15; titanix's browser setup is unverified (A41).
- Storage: no default location anywhere in library code. Browser autosave only with `storage-key`. The bake-off
  and perf drivers are stdlib-only Python in the hub's `scripts/studies/` (A53), so the **output-directory
  resolver is Python and lives in `scripts/studies/pipecanvas_paths.py`**, a small module both drivers import. It
  reuses S5-03's path table instead of inventing a second one (A54): explicit `--out-dir` > environment variable
  `PIPECANVAS_BENCH_DIR` (`none`/empty disables) > `hub.toml [paths] study_dir` (read with stdlib `tomllib`, the
  same file S5-03's Node resolver reads) > `${XDG_CONFIG_HOME:-~/.config}/pipecanvas/config.toml` > off (fail
  loudly). It reports the deciding layer (`bench_dir_source()`), raises on a malformed config file (a unit test with a
  malformed fixture is the gate), and never writes into another project's data. It copies S5-03's pattern and
  tests, not its code, because S5's resolver is Node; a cross-check test feeds one `hub.toml` fixture to this module
  and asserts it resolves the same `study_dir` S5-03's tests expect.

### 4.12 Pre-registration and resume design for number-producing items

S6-36 (bake-off run) and S6-27 (performance budget) produce findings. Both use the hub's shim shape (A48, A53):
a tracked stdlib-only Python driver, `scripts/studies/pipecanvas_bakeoff.py` and
`scripts/studies/pipecanvas_perf_budget.py`, each with a stem-named sidecar (`pipecanvas_bakeoff.bth.toml`,
`pipecanvas_perf_budget.bth.toml`) stating the hypothesis and `[outcomes]`, **written and committed before the
run**. S6-08 is the pre-registration PR (harness, resolver, controls, committed sidecar); S6-36 is the run;
S6-27 commits its own sidecar first and reads its threshold from the S6-36 record id the sidecar names. Run under
slug `aminx-hub` with the installed `bth` (not `uv run bth`): `bth run --project-slug aminx-hub -- uv run --no-sync
python3 scripts/studies/pipecanvas_bakeoff.py ...`, so `script_path` resolves to a `.py` and the sidecar resolves
(a `node`, `npx` or `bash -c` wrapper would leave the sidecar unresolved and the record outcome empty). The
Node-side pieces (the Playwright specs under `packages/pipecanvas/scripts/bakeoff/specs/` that the driver
launches) are the only JavaScript. **Whether a Python driver reaches a Playwright cell and the sidecar resolves
through `bth` is proven by S5-59** (A48), which S6-08 and S6-27 depend on; S5-02 only proves the pinned Chromium and
the header-controlled dev server. Both items execute on the single named host of 4.11.
Verify by record: `bth compact`, then `bth sql "SELECT id,status,outcome,exit_code,command FROM runs WHERE id
LIKE '<prefix>%'"` and spot-check that claimed paths resolve; if the warm index still shows a stale `running` row,
check the cool-tier parquet. Controls: a **positive** stub engine that passes G1 and a **negative** stub whose veto
is a no-op that must fail G1; the harness is not trusted until the negative control fails. Resume: units are
(candidate, scenario) pairs; each runs in its own process with its own timeout (a timeout loses one unit);
a unit's result is persisted with a completion stamp as it finishes and is reused on resume only if the
sha256 of the candidate's lockfile and of the scenario definition match, with reused units and hashes
recorded in the result. Resume changes scheduling, never the `[outcomes]`.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| S4's file moves again (it changed between two reads in round 2; its round 1 landed while this spec was being revised) | The hub `graph` is S4's IR v2 verbatim (A42), the type engine and corpus are S4's, and every S4 dependency is an item id (S4-04, S4-06, S4-07, S4-09, S4-10, S4-30, S4-31). Line citations are as of this revision and the coherence pass re-checks them |
| S4's validator does not handle `builtin:code.python` ports the way the editor assumes, or S4-31's corpus has no code-node vector (A46) | The S6-16 gate reads the S4-31 corpus; if the vector is absent the request is filed (Q14) and S6-16 carries a local vector flagged non-normative, so the gap is visible and not a silent divergence |
| Browser and executor validators drift | One corpus, owned by S4 (S4-31), run by S4's Python validator and S6-04's TS evaluator; mutation control (flip one table entry, corpus must fail); a pinned-table equality test against S4-09's generated artifact; S4-30's hash vectors run by S6-35 |
| S5-04's catalog record lacks the fields the palette needs (A51) | S6-33 amends the builder (no-op if the fixture already carries them); the S6-05 snapshot is built from S5-04's emitted fixture, so a missing field fails there, not in production |
| The package breaks S5-01's CI guard or the hub's vendoring model (A52, A55) | No Python in the package; drivers are stdlib-only shims in the hub's `scripts/studies/`; S6-02's gate runs S5-01's guard with the package present; the npm allow-list and workspace-source vendoring are filed as visible change requests (Q17) |
| Library pick is wrong or the library stalls (Rete core published 15 months ago, A9) | Engine behind `CanvasEngine`; pre-registered bake-off; user confirmation; the list view works with no engine at all |
| Premature generalization for praxis | Praxis gated on S6-19 `outcome = "go"` (checked by every extraction-branch gate); read-only first; no praxis fields in core; praxis pack/adapter in praxis repo; `toy.lab` pack proves neutrality cheaply (4.9) |
| A no-go or defer unlocks the extraction items because a DAG only knows "done" | S6-34, S6-20 and S6-21 gates start with `check_checkpoint.py --require go` and fail on a no-go fixture (negative control) (4.10) |
| Code nodes run untrusted code and can exfiltrate uploaded structures | Untrusted-by-default for foreign documents (`codeTrusted` set by the host, per-hash trust, S6-17), S5-43's sandbox worker with `connect-src 'none'` and its canary test, an end-to-end canary through the editor in S6-28, fallback "session-authored only" decided by the user (Q4). Residual risk until the sandbox mechanism is verified (it is **unverified**, A32) |
| An `executor_hint` edit changes `graph_sha256` and silently revokes code trust | Intended and conservative (4.2, Q19): trust is per graph hash; the UI says why trust was lost; spec keys ignore hints |
| Users expect JAX in code blocks | Allow-list-driven lint with a named diagnostic pointing to model nodes; stated plainly in UI and docs (4.7) |
| Editor allow-list and S5-43's executor allow-list disagree | The editor lint is lock-agnostic and fed by a generated JSON; S6-31 asserts equality with S5-43 on the same pinned lock |
| Cross-language hashing mismatch (floats) | Naive Python float formatting provably differs from ECMAScript (A25); S4-30 owns the Python JCS and vectors, S6-35 passes the same vectors in TypeScript (A56); if they disagree, S4 decides whether float params become decimal strings |
| A shared URL is truncated by a chat client or exceeds browser limits | 32 KiB cap with typed `E_SHARE_TOO_LARGE` and a file-download fallback; decode failure shows `E_SHARE_DECODE` and loads nothing (A39) |
| Extraction leaves the hub on a stale in-repo copy or praxis waiting on a manual publish | S6-30 migrates the hub to the release tarball and deletes the in-repo copy; S6-20 produces a tag plus checksummed tarball, so npm publishing (S6-32) is optional and nothing depends on it |
| Custom element inside Angular misbehaves (zoneless, events, forms) (A8, A21) | S6-22 gate tests both the zoneless production and zone.js test configurations; `@pipecanvas/angular` wrapper exists as the fallback |
| Browser data leaves the machine silently | The editor renders S5's `PrivacyDescriptor` on every node and dashed transfer edges; consent before the first byte stays in S5's remote client (S5-20); `W_TRANSFER_LARGE` |
| Bake-off or perf runs lose work to a crash/timeout | Per-(candidate, scenario) units, each its own process and timeout; persisted on completion; resume reuses a unit only when candidate lockfile hash and scenario hash match, recording reused units and hashes (4.12) |
| Canvas inaccessible | List view permanent; keyboard connect flow (S6-14); axe gate |
| S5's hub CI and deploy shape differ from what S6-02 assumes | S6-02 depends on S5-01 and S5-02; the package adds a job to that CI rather than a parallel workflow |
| Rollback | Everything is additive in new packages; the editor reaches the site only through S5-31's page, so reverting that page removes it without touching the packages; praxis items are separate PRs in the praxis repo |

## 6. Interfaces

`depends_on` in section 9 carries **item ids only** (`S<n>-<NN>`). The contract names below are labels for
humans; the "Bound to" column is the id binding. Provider ids were re-read at the current state of S4 and S5
(A47-A50); both are concurrent specs, so ids are as of this revision and the coherence pass re-checks them.
Ids S6-24, S6-25 and S6-26 are retired and never reused.

### Provides

| Contract | What | Satisfied by | Consumers |
|---|---|---|---|
| `S6:graph-document` | `pipecanvas.document/1`: JSON Schema, TS types, load gate, migrations (4.2); canonical hash and share codec in S6-35 | S6-03, S6-35 | S5-30 (runner input, binds S6-03), S5-31, praxis |
| `S6:pipeline-editor-component` | `<pipe-list>` (S6-10, stage 1), then `<pipe-canvas>` (S6-12, adopted by the hub at S6-29): property/event/slot/CSS contract (4.5) | S6-10, S6-12 | S5-31 (binds S6-10), praxis host |
| `S6:jupyterlite-blocks` | `code.python` node contract + allowed-imports lint + `.ipynb` export (4.7) | S6-16, S6-18 | S5-35 (binds S6-18) |
| `S6:domain-pack` | `DomainPack` and `GraphAccess` interfaces (4.9) | S6-04 | S5 (hub pack derived from S4), praxis (S6-21) |
| `S6:validator` | TS evaluator of S4's `compat` + structural rules, stable diagnostic codes; passes S4's corpus (4.3) | S6-04 | S5-30 (may reuse for pre-run checks) |
| `S6:palette-model` | `paletteFromCatalog`, `paramsToFormModel` over xtrax-param-profile@1 (4.4) | S6-05 | S5-31, S5-28 (catalog pages may reuse) |
| `S6:execution-client-port` | `ExecutionClient`, `ExecutionPlan`, `RunEvent`, `NodeRunState`, and the `runExecutionClientContract` suite (4.6); S6-11 publishes the suite with a fake client | S6-07, S6-11 | S5-30 (the adapter must pass it), S6-11 |
| `S6:release-artifact` | tag + tarball + `checksums.txt` + checkpoint outcome of the extracted package (4.1) | S6-20 | S6-30 (hub), S6-21/S6-22 (praxis), S6-32 |
| `S6:node-body-slots` | `slot="body-<nodeId>"` contract | S6-10, S6-12 | S5 viewer integration |
| `S6:ui-hint-vocabulary` | additional `x-xtrax.widget` values the form model renders (4.4) | S6-05 | S4 (documents them next to the param profile; no item dependency, S4 6 "Consumes", hint `:1176`) |

The conformance corpus and the JCS hash vectors are **S4's** (`S4:port-types`, S4-30, S4-31). S6 consumes them and
adds no corpus of its own.

### Consumes

| Contract | Provider spec | Bound to item(s) | What S6 needs |
|---|---|---|---|
| `S4:port-types` (`PortTypes@1`, table, `compat`) | S4 | S4-04 | the port-type table the TS `compat` evaluator reads |
| `S4:model-manifest` + `S4:contract-artifacts` | S4 | S4-06 (manifest dataclasses with port `required`/`cardinality`), S4-09 (committed JSON Schemas incl. `graph-ir.v2` and `port-types.v1`, TS types, validators) | one node type per manifest with typed ports and `params`; generated TS and schemas the evaluator and form model build on |
| `S4:graph-ir` (`GraphIR@2`) | S4 | S4-10 (graph IR v2 with `ref`, edge `id`/`kind`, `builtin:code.python`) | the hub `graph` member, verbatim (A42) |
| `S4:jcs-vectors` | S4 | S4-30 | the hash vector file S6-35's TypeScript JCS must reproduce |
| `S4:conformance-corpus` + diagnostic registry | S4 | S4-31 | the corpus S6-04's TS evaluator must pass; the code registry `codes.ts` mirrors |
| `S4:param-profile` (`ParamProfile@1`) | S4 | S4-07 (schemagen and the profile), S4-09 | the keyword subset `paramsToFormModel` implements |
| `S5:catalog-index` | S5 | S5-04 (catalog builder) | `dist/catalog/catalog.json` plus `models/<id>.json`, per-executor `availability[]` (A50); fields beyond S5's mapping table via S6-33 (A51) |
| `S5:executor-interface` + `S5:privacy-descriptor` | S5 | S5-08 (Executor core, PrivacyDescriptor, capability probe) | `Executor` event vocabulary, `probe()` results for `available`/`reasons`, per-executor privacy |
| `S5:executor-plan` + consent hook | S5 | S5-37 | `plan()`, value refs, transfer encodings, `consent_required` |
| `S5:hub-execution-client` | S5 | S5-30 (runner plus `HubExecutionClient`) | the adapter that passes S6's suite; what S6-28 and S6-29 drive |
| `S5:run-store` | S5 | S5-10 | reuse report (reused units, hashes) for the "reused" display |
| `S5:viewer-adapter` | S5 | S5-12 | `mountViewer(el, {structure, scores, overlays})` for preview-node bodies (body slot only, not a hard dependency) |
| `S5:code-python-capability` + `S5:pyodide-executor` | S5 | S5-17 (executor), S5-43 (`code.python`, trusted flag, sandbox, allow-list) | code-node execution and the executor-side allow-list (A30) |
| `S5:hub-skeleton`, `S5:e2e-harness`, `S5:paths` | S5 | S5-01 (repo, CI, dependency guard), S5-02 (Playwright harness, pinned Chromium), S5-03 (`hub.toml`, `[paths]`) | where the package starts; browser plumbing; the path table the driver resolver reads |
| `S5:bathos-shim` | S5 | S5-59 (shim check for node/Playwright cells) | proof that a stdlib Python driver reaches a Playwright cell and its sidecar resolves |
| `S5:pipelines-page` | S5 | S5-31 | the page S6-29 upgrades and S6-19's evidence comes from |
| `S5:first-deploy` | S5 | S5-33 | the public origin whose live check starts the 60-day clock (4.10) |
| `S5:lab-page` | S5 | S5-35 | the `/lab/` page, the precondition for the optional notebook bridge (S6-23) |
| `S1` registry / knob schema (optional, no dependency) | S1 via S4 | none (S4-20 produces the MPNN knob schema; S6 uses it only as a form fixture if present) | an MPNN RunSpec form example |
| `ProtocolComputationGraph` JSON | praxis (existing; `models.py:623-671`, A37) | none (existing code) | the shape the praxis adapter reads |

### Requests that stay with the owning spec (not S6 items)

| To | Request | Tracked as |
|---|---|---|
| S4 (S4-31) | the corpus includes `builtin:code.python` vectors and states the port-entry fields of a code node (A46) | Q14 |
| S5 (S5-01) | state whether the dependency guard scans `package.json`, and give a visible npm allow-list policy | Q17, S6-02 gate |
| S5 (S5-31/S5-04) | accept a workspace-sourced `vendor.lock.json` entry before extraction | Q17, A55 |
| S5 (S5-04) | pass `params`, `description`, port `accepts_encodings`/`alphabet` and `manifest_sha256` (defined as sha256(JCS(manifest)), the S4 form) through the catalog record; state in S5 section 6 that its wire `manifest_sha256` and RunStore unit key use that same hash | Q18, S6-33 |
| S5 | S5-30 keeps `S6-07` and `S6-11` in its `depends_on` and drops no edge; S6-11 now has no S5-30 dependency | none needed (S5 file `:1372`) |

## 7. Verification gates per item

Local runs are narrow only (a single package test file / one Playwright spec); these are small JS suites, not
JAX suites, and full runs go to CI or titanix. Nothing here is executed by this spec.

Where things run. Unit and contract suites run in CI (S5-01's CI). Playwright specs use the pinned Chromium
from S5-02 and run in CI for functional gates. The two number-producing runs (S6-36, S6-27) run on the one named
host of 4.11 (Q15), under slug `aminx-hub`, never in CI (A48); a frame-time threshold is never compared across
hosts. Every number-producing item has a tracked stdlib driver plus a `.bth.toml` sidecar committed **before** the
run, is verified by its record (`outcome` evaluated), and is chunked and resumable per 4.12. Praxis builds and
tests (S6-21, S6-22) run in praxis CI or on titanix, never as a whole-suite local run.

| Item | Gate (how we prove done) |
|---|---|
| S6-01 | Decision record committed with chosen home, name, extraction trigger; names exist only in `constants.ts` (grep test) |
| S6-02 | CI (a job added to S5-01's CI, Chromium from S5-02) green on a clean checkout: `tsc --strict`, `node --test` smoke, dependency-boundary check **fails** on a seeded forbidden import (negative control), license allowlist check **fails** on a seeded GPL fixture (negative control); **S5-01's dependency guard passes with the package present** (and still fails S5-01's own planted unlisted dependency, so it was not bypassed); `npm run build` emits `dist/pipecanvas-elements.js` plus `checksums.txt`; if the guard scans `package.json`, the same PR adds the devDependencies to the visible allow-list file (Q17) |
| S6-03 | Tests: golden v1 documents embedding an S4 graph IR v2 validate against S4-09's `graph-ir.v2` JSON Schema and deep-equal after load/save; the load gate rejects missing/non-int/older/newer envelope versions and never default-fills; an embedded graph version other than 2 gives `E_SCHEMA_VERSION`; unknown fields preserved in envelope, `layout` and `extensions`; `x-` keys preserved inside `graph` and a non-`x-` unknown key refused with a typed error naming it; `layout`/`extensions` edits leave the graph member unchanged |
| S6-35 | Tests: every vector in the pinned copy of S4-30's `jcs.v1.json` (`0.1`, `1e-5`, `1e16`, `1.0`, 2^53 ints, key order, unicode) matches; a deliberately wrong number formatter fails them (negative control); `graph_sha256` ignores `layout` and `extensions` and changes with a param edit; share-fragment tests: a document over `SHARE_FRAGMENT_MAX` yields `E_SHARE_TOO_LARGE` with no truncation, a corrupted fragment yields `E_SHARE_DECODE` and loads nothing |
| S6-04 | The TS evaluator passes the full pinned S4-31 corpus; mutation control (flip one port-table entry, corpus must fail); pinned-table equality against S4-09's generated artifact; a `toy.lab` pack test shows a linear-resource rule works with zero core changes; every S4-registry code used by the editor is produced by at least one corpus vector, and the S6 pack-level codes have local tests |
| S6-33 | S5-04's builder, run on its fixture manifests, emits records carrying `params`, `description`, port `accepts_encodings`/`alphabet` and `manifest_sha256` equal to sha256(JCS(manifest)) (checked against S4-30's Python value for each fixture manifest); two builds are byte-identical; S5-04's own `--check` and drift gates still pass; the item is a no-op (gate passes on inspection) when the fixture already carries the fields |
| S6-05 | Tests: palette built from **S5-04's emitted `catalog.json` fixture** (one entry per manifest) matches a snapshot; `available`/`reasons` come from joining `availability[].requires` with an S5-08-shaped probe report; a manifest fixture using `oneOf` with a discriminator, `$ref`, `prefixItems` and `x-xtrax` hints builds a form model with no error; a keyword outside the S4 profile raises the named error; unavailable executors are greyed with the probe reasons, not dropped; a manifest without `params` falls back to the flagged raw-JSON editor; `required`/`cardinality` are copied from manifest ports (no defaults) |
| S6-06 | Property test: random command sequences followed by full undo restore the exact document (hash-equal via S6-35), with a negative control (a deliberately non-invertible command is caught); patch events replay to the same document |
| S6-07 | Tests: spec keys stable under layout edits and regenerated edge ids; a param edit changes only the edited node and its descendants (negative control: siblings/upstream unchanged); spec-key input is the `models` lock hash (changing it changes the key); a fake `ExecutionClient` drives run-state transitions from S5-shaped events (progress/unit/done); `runExecutionClientContract` passes the fake and **fails** a seeded-broken client (negative control) |
| S6-08 | The **pre-registration PR**: harness, the Python output-directory resolver (`pipecanvas_paths.py`) with a malformed-config test and the `hub.toml` cross-check, positive and negative stub engines (the negative fails G1), scenario definitions with their sha256, and the sidecar `pipecanvas_bakeoff.bth.toml` (hypothesis, `[outcomes]`, host, Chromium pin, thresholds) committed **before any run**; the stdlib-only drivers pass S5-01's guard; the smoke of the harness runs in CI |
| S6-36 | `bth run --project-slug aminx-hub -- uv run --no-sync python3 scripts/studies/pipecanvas_bakeoff.py` on the named host: G1-G5 recorded per candidate; soft metrics recorded; per-(candidate, scenario) units persisted with completion stamps and resumable (reuse only on matching lockfile and scenario sha256, reused units listed); verified by record (`bth compact`, then `bth sql`: `outcome` evaluated, not exit code) |
| S6-09 | User decision recorded in a decision doc: engine confirmed or overridden, with the S6-36 record id cited |
| S6-10 | Playwright on the in-package demo harness: build structure -> model -> viewer from the palette, including by dragging a palette entry; the inspector edits params and `E_PARAM_SCHEMA` fires on a bad value; document validates; fan-out (two steps pick the same output) and fan-in work; reorder only among valid topological positions; keyboard-only completion; axe reports zero serious violations; cycles unrepresentable |
| S6-11 | Playwright on the demo app: a list pipeline runs with the fake `ExecutionClient` and node states render from S5-shaped events; the document survives a URL-fragment round trip with identical graph hash (via S6-35); the published contract suite is importable by a separate package (an S5-30-shaped consumer fixture imports it and fails a seeded-broken client); the catalog fixture is S5-04's emitted one |
| S6-12 | Playwright: `reject` drop is vetoed before the edge exists (`connect-rejected` emitted, alphabet mismatch reads `E_ALPHABET_MISMATCH`), `exact`/`widen` drop commits, an encoding mismatch commits with the `W_ENCODING_ADAPTER` warning shown; shadow-DOM pointer interaction passes (G2 re-checked on the production adapter); layout round-trips exactly. No performance threshold in this item |
| S6-13 | Playwright: multi-select move, copy/paste with node and edge id remap (undo restores exactly), group and comment round trip; documents without layout get a deterministic auto-layout (same input, same output) |
| S6-14 | Playwright + axe: complete a connection with keyboard only; connect-rejected reason announced in the live region; zero serious axe violations on `<pipe-canvas>` |
| S6-15 | Playwright on the demo with fake plans: segment/privacy chips come from `PrivacyDescriptor` fixtures (a remote-segment fixture reads "leaves your machine"); cross-segment edges (by edge id) dashed; reused nodes marked with the unit-key short hash; list/canvas switch preserves the graph hash |
| S6-16 | Tests with an allowed-imports JSON fixture: `import jax` / `onnxruntime` / `torch` produce `E_CODE_IMPORT_UNAVAILABLE`, `import numpy` passes; port-table edits update node ports and validation; the pinned S4-31 corpus's `builtin:code.python` vectors pass (A46), and if the corpus has none the item records that, files the request (Q14) and adds a local vector marked non-normative; an absent `allowedImports` turns the lint off visibly; the Node CLI turns a lock fixture into the JSON and changing the fixture changes the list |
| S6-17 | Playwright with the fake client (no S5 dependency): a URL-loaded document opens with code nodes inert, `run-requested` carries `codeTrusted: false`, `trust-requested` fires on "Trust" with the graph hash, trust is per hash (an edited graph, including an `executor_hint` edit, is untrusted again) |
| S6-18 | Exported `.ipynb` validates against the nbformat schema; cell order is a valid topological order of the document; the notebook opens in a JupyterLite smoke test |
| S6-19 | The decision item. Criteria file committed before review (4.10) with `deploy_date` taken from S6-29's live-origin record; `scripts/check_checkpoint.py` passes only when `review_date >= deploy_date + 60 days` (a 59-day fixture fails); `docs/checkpoint_outcome.toml` records go / no-go / defer. Completing the item is **not** a go |
| S6-34 | User-owned. `check_checkpoint.py --require go` passes on the committed outcome file (and fails on a `no-go` fixture); the empty `pipecanvas` repo exists and its slug is recorded (`git ls-remote` resolves) |
| S6-20 | Hub-side. `check_checkpoint.py --require go` first; then the subtree split is pushed to the S6-34 repo with history preserved; a tag and release tarball (core ESM, one-file elements bundle, `.d.ts`, `checksums.txt`, `docs/checkpoint_outcome.toml`) are published; `git ls-remote` shows the tag; clean-directory consumer smoke test from the tarball (import core in Node, register elements via import map) |
| S6-21 | praxis tests (CI or titanix): the release tarball's `docs/checkpoint_outcome.toml` has `outcome = "go"`; the adapter converts the real 4-operation fixture and **all 28** `plr-sema/tests/fixtures/*_graph.json` files (glob count asserted equal to 28) to documents that pass core validation with the praxis pack; the mapping rules of 4.9 hold on a nested-regions fixture (`contains` edges, `order` per container, `parent` layout); the complete PCG is preserved at `extensions["x-praxis"].pcg` |
| S6-22 | praxis build (CI or titanix, not a local whole-suite run) includes the release tarball; for each of 3 chosen fixtures the protocol detail view shows op-node count equal to `len(operations)` (regions and bodies included, not `len(execution_order)`) and resource-node count equal to `len(resources)`; an event from the element updates a signal and re-renders under **both** the production (zoneless) configuration and the unit-test (zone.js) configuration (resolves A21; A8 is already read); decides whether `@pipecanvas/angular` is needed |
| S6-23 | (Optional, on user go) in the `/lab/` page a notebook cell renders `<pipe-list>` for a document and a Python-side handle receives edited document JSON; round trip preserves the graph hash |
| S6-27 | **Pre-registered** bathos run on the same named host as S6-36: sidecar committed first, names the S6-36 record id and the threshold derived from it; 200-node / 400-edge pan/zoom units persisted and resumable; a stub engine with an artificial delay fails the budget (negative control) and a trivial engine passes it (positive control); verified by record |
| S6-28 | End to end through the editor on the hub page with S5-30's `HubExecutionClient`: an untrusted document is refused (`run-requested.codeTrusted == false` and the executor refuses code nodes); a trusted numpy node runs; a code node that `fetch`es a canary origin fails while the same fetch succeeds in a control without the sandbox (the control proves the canary is reachable). If the canary cannot be made to fail, the item records that and stops for the Q4 user decision |
| S6-29 | Playwright on the **live public origin** (after S5-33): `<pipe-canvas>` replaces the list as the default with the list view still available; two catalog nodes dragged and wired, run via `HubExecutionClient`, both outputs seen; the served `pipecanvas-elements.js` sha256 equals the release build's `checksums.txt`; `deploy_date` (date of that passing check) and the served sha256 are written to the criteria file |
| S6-30 | Hub CI green with the pipecanvas release tarball vendored: `vendor.lock.json` sha256 matches the tarball, import-map entry resolves, `rg "packages/pipecanvas"` in the hub finds nothing, the Pipelines page Playwright spec still passes |
| S6-31 | Hub build emits allowed-imports JSON from the pinned Pyodide lock (explicit `--lock`); a test asserts the editor's list equals S5-43's executor-side list for the same lock, and a planted extra package in a copy of the lock changes both |
| S6-32 | (Optional, user-owned) `npm pack --dry-run` lists only built files and the declared version; publish performed by the user; nothing in the DAG depends on this item |

## 8. Open questions for the user

| # | Question | Recommendation |
|---|---|---|
| Q1 | Component name and home. Working name `pipecanvas`; where does it live? | Keep `pipecanvas` as a placeholder (rename is one constants file). Start inside the hub repo with an enforced boundary; extract to its own repo only on a recorded praxis "go" (S6-34 you create the repo, S6-20 the hub-side split, tag and checksummed tarball), then migrate the hub to the release (S6-30). npm publishing is a separate optional item (S6-32, Q16). Repo creation and publishing are yours. |
| Q2 | Is the DAG canvas a goal in itself, given that the stage 1 list (per-input pickers over earlier steps) is already DAG-expressive and accessible? | Yes, build the canvas (you asked for the geometry-nodes feel and it gives spatial overview, grouping and layout), but ship the list first and keep it permanently as the accessible/mobile view. |
| Q3 | JupyterLite: embed a real JupyterLite kernel as the execution engine for code blocks, or use S5's Pyodide worker and offer export-to-notebook? | S5's Pyodide worker for execution, export-to-notebook for the JupyterLite experience (S6-18). Embedding the kernel couples execution to a Jupyter session and adds a large bundle. |
| Q4 | Trust model for code nodes in shared documents. If the sandbox cannot be made to block network access, what is acceptable? | Untrusted by default for foreign documents with per-hash trust; if the `fetch` negative test cannot pass, code nodes run only for documents authored in the current session. |
| Q5 | Type system scope (revised in round 2): the editor implements exactly S4's `compat` (`exact`/`widen`/`reject`, alphabet rule, `W_ENCODING_ADAPTER` warning) and nothing more; symbolic-dimension unification is reserved by S4 and absent here. Should it also offer an explicit "insert adapter node" command? | S4's rule as it is. Defer the insert-adapter command (the rejection message already names adapter-role manifests); add it only if real pipelines hit the limit. It is an explicit user edit, never silent, matching S4. |
| Q6 | Praxis stage scope: read-only graph view first, no authoring? | Yes. Authoring needs graph-to-code round-tripping that this spec explicitly excludes. |
| Q7 | Criteria for the praxis go/no-go (4.10): are the proposed 60 days (from the live-origin deploy) / 3 maintainer pipelines / no P1 / written first use case right? | Edit them before S6-19 and commit; they are proposals, not requirements. |
| Q8 | Engine pick: confirm Rete.js v2 provisionally, or short-list differently? | Confirm provisionally; make S6-09 the real decision after the pre-registered bake-off run (S6-36). |
| Q9 | (Resolved in round 2; confirm.) Who owns the `ExecutionClient` port and its adapter? | S6 owns the port, its types and the contract suite (S6-07, published by S6-11); **S5 owns the adapter** `HubExecutionClient` (S5-30), as S5 records. No request to S5 is outstanding. |
| Q10 | (Resolved in the DAG; confirm.) How does S5-31 avoid being blocked on the canvas while the praxis checkpoint needs a deployed canvas? | S5-31 binds to the stage 1 list (S6-10). S6-29 upgrades the page to `<pipe-canvas>` after S5-33's public deploy and records the deploy date from a live-origin check; S6-19 depends on S6-29 and checks the 60-day rule from that date (4.10). |
| Q11 | (Narrowed: S5-43 specifies `code.python`.) Is S5-43's design acceptable as the code-node executor, and what happens if its sandbox canary cannot be made to fail (A32)? | Accept S5-43. If the canary cannot be made to fail, adopt "code nodes only run for documents authored in the current session" (Q4); S6-28 stops for that decision. |
| Q12 | (Resolved in round 2.) Request to S4 for port `required`/`cardinality` and a `builtin:` scheme? | Withdrawn: S4 already carries both (A42, A43). Nothing is requested; S6-25 is dropped. |
| Q13 | Is an interactive pipeline widget *inside* JupyterLite notebooks (S6-23) wanted, or is export-to-notebook (S6-18) enough for "JupyterLite blocks"? | Export is enough for now. S6-23 is optional and gated on your go because it adds a notebook-side bridge to maintain. |
| Q14 | (Reworded in round 2.) Who carries the conformance corpus, diagnostic registry and JCS vectors, and what small asks remain of S4? | S4 is the sole owner (S4-30, S4-31); S6-26 is dropped. Two asks stay with S4: S4-31's corpus includes `builtin:code.python` vectors and S4 states the port-entry fields of a code node (A46); S4's `W_NO_EXECUTOR`/`W_TRANSFER_LARGE` stay S6 pack codes. S6-16 files the request if the vectors are absent. |
| Q15 | Which single host runs the bake-off and the performance budget (S6-36, S6-27)? | titanix if its pinned-Chromium Playwright lane is verified (S5 A27, A41 here); otherwise the local workstation. Whichever is chosen is recorded in the sidecar and never mixed. |
| Q16 | How does praxis consume the extracted package: a release-tarball URL plus sha256 (default), a git URL, or an npm publish (S6-32)? | Release tarball by URL plus sha256: verifiable in the DAG without a publish. npm publishing stays optional and yours. |
| Q17 | (New.) Change requests to S5: (a) does S5-01's dependency guard scan `package.json`, and if so what is the visible npm allow-list policy for the package's devDependencies; (b) may `vendor.lock.json` carry a workspace-sourced, hash-pinned entry for the package build before extraction (A52, A55)? | Yes to both, as visible changes (one decision with the S5 section 6 row for S6's requests, so S5 answers Q17, Q18 and the hash definition together): the guard stays Python-scoped unless S5 says otherwise, any npm allow-list is a committed file, and the workspace vendoring entry records the sha256 from `checksums.txt`. S6-02's gate is the check. |
| Q18 | (New.) May S5-04 be amended (S6-33) to pass `params`, `description`, port `accepts_encodings`/`alphabet` and a manifest sha256 through the catalog record? | Yes. They are manifest fields S4 already defines, so no S5 design changes; S6-33 is a no-op if the fixture already has them. The hash is sha256(JCS(manifest)) so the S5-19 echo check and the models lock agree; answered with Q17 in the S5 section 6 row. |
| Q19 | (New.) `executor_hint` is a field of S4's hashed graph, so changing a hint changes `graph_sha256` and revokes per-hash code trust. Accept? | Accept: it is conservative and keeps one IR. Spec keys ignore hints, so cache prediction is unaffected. The alternative (hints outside the graph) would diverge from S4's shape. |

## 9. Backlog items

Order of work is expressed **only** through `depends_on`, and every entry is an item id (`S<n>-<NN>`). Cross-spec
ids were re-read in the current S4 and S5 specs (A47-A49, section 6). `repo = "aminx-hub"` means the in-hub workspace
`packages/pipecanvas/` (and the hub's `scripts/studies/`) for work before S6-20 and the hub repo itself afterwards;
`pipecanvas` is used only for work whose first commit lands in the new repo (S6-34, S6-32); `praxis` items land in the
praxis repo. **Retired ids, never reused:** S6-24 (folded into S6-11), S6-25 and S6-26 (superseded by S4-06, S4-10,
S4-30 and S4-31, as S4 itself records). Ids 33 to 36 are new in round 2. `HubExecutionClient` is S5-30's, not an S6
item.

```toml
[[item]]
id = "S6-01"
title = "Decide pipecanvas name, home (inside hub vs own repo) and extraction trigger; record decision"
repo = "aminx-hub"
size = "S"
depends_on = []
gate = "Decision doc committed with name, home and extraction trigger; names exist only in one constants module (grep test)"
user_decision = true

[[item]]
id = "S6-02"
title = "Scaffold pipecanvas workspace (core + elements, TypeScript build to prebuilt ESM) in the hub repo with a CI job, dependency-boundary and license-allowlist checks, and the S5-01 guard passing with the package present"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-01", "S5-01", "S5-02"]
gate = "CI green on clean checkout using the S5-02 pinned Chromium; boundary check and license check each fail on a seeded violation; S5-01's dependency guard passes with the package present; npm run build emits dist bundle plus checksums.txt"
user_decision = false

[[item]]
id = "S6-03"
title = "Document envelope v1 embedding S4 graph IR v2: JSON Schema, TS types, load gate, migrations harness, lossless round trip with S4's x- key rule"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-02", "S4-09", "S4-10"]
gate = "Tests: golden v1 documents validate against the S4-09 graph-ir.v2 schema and deep-equal after load/save; version gate rejects missing/older/newer and never default-fills; embedded graph version other than 2 gives E_SCHEMA_VERSION; x- keys preserved and other unknown graph keys refused with a typed error"
user_decision = false

[[item]]
id = "S6-35"
title = "TypeScript RFC 8785 canonical form and graph_sha256 passing S4-30's vector file, plus the share-fragment codec with size limit"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-03", "S4-30"]
gate = "Tests: every vector in the pinned S4-30 file matches and a wrong number formatter fails them; graph hash ignores layout and extensions; oversize fragment gives E_SHARE_TOO_LARGE without truncation and a corrupt fragment gives E_SHARE_DECODE and loads nothing"
user_decision = false

[[item]]
id = "S6-04"
title = "TS evaluator of S4 compat over the generated port-type table, structural validator and diagnostic registry mirroring S4, DomainPack and GraphAccess interfaces; pass the S4-31 corpus"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-03", "S4-04", "S4-09", "S4-31"]
gate = "Tests pass the full pinned S4-31 corpus; mutation control (flip one port-table entry) fails the corpus; pinned table equals the S4-09 artifact; toy.lab pack rule works with zero core changes"
user_decision = false

[[item]]
id = "S6-33"
title = "Amend the S5-04 catalog builder so catalog records carry the palette fields (params schema, description, port accepts_encodings and alphabet) and manifest_sha256 = sha256(JCS(manifest)) computed with S6-35's canonicaliser; no-op if already present"
repo = "aminx-hub"
size = "S"
depends_on = ["S5-04", "S4-06", "S6-35"]
gate = "S5-04's builder on its fixture manifests emits records with those fields, manifest_sha256 equals the Python sha256(JCS(manifest)) of S4-30 for each fixture, two builds are byte-identical and S5-04's own check and drift gates still pass"
user_decision = false

[[item]]
id = "S6-05"
title = "Palette model (one entry per manifest) from S5-04's catalog.json plus a probe report, and params form model over xtrax-param-profile@1 with x-xtrax hints and loud unsupported-keyword errors"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-03", "S4-06", "S4-07", "S4-09", "S5-04", "S6-33"]
gate = "Tests: palette snapshot from S5-04's emitted catalog fixture; availability joined with a probe report gives available and reasons; oneOf/$ref/prefixItems fixture builds without error; keyword outside the profile raises named error; unavailable executors greyed with reasons; required/cardinality copied from manifest ports"
user_decision = false

[[item]]
id = "S6-06"
title = "Command stack: invertible edit commands, undo/redo, patch events"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-03", "S6-35"]
gate = "Property test: random command sequences then full undo restore hash-equal document; non-invertible negative control is caught"
user_decision = false

[[item]]
id = "S6-07"
title = "Run-state model, ExecutionClient port and types, plan view-model, spec keys from the models lock, and the runExecutionClientContract test suite"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-35", "S5-08"]
gate = "Tests: spec keys stable under layout edits and edge-id regeneration and change only for the edited node and descendants; fake client driven by S5-shaped events; contract suite passes the fake and fails a seeded-broken client"
user_decision = false

[[item]]
id = "S6-08"
title = "Pre-register the canvas-library bake-off (Rete.js v2, xyflow-system, X6): stdlib driver and output-dir resolver in hub scripts/studies, positive and negative stub controls, scenario hashes, committed sidecar, Playwright specs"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-02", "S6-04", "S5-02", "S5-59"]
gate = "Sidecar committed before any run; negative stub fails G1 and positive stub passes; resolver malformed-config test and hub.toml cross-check pass; drivers are stdlib-only and pass S5-01's guard; harness smoke green in CI"
user_decision = false

[[item]]
id = "S6-36"
title = "Run the pre-registered bake-off on one named host and record the bathos run id"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-08"]
gate = "bth run under slug aminx-hub on the named host; G1-G5 and soft metrics recorded per candidate; units resumable by lockfile and scenario hash with reused units listed; verified by run record outcome after bth compact"
user_decision = false

[[item]]
id = "S6-09"
title = "Confirm or override the canvas engine from the bake-off record"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-36"]
gate = "Decision doc cites the S6-36 run id and names the chosen engine"
user_decision = true

[[item]]
id = "S6-10"
title = "pipe-list element with in-package demo harness: ordered step list, per-input pickers, palette (drag or click add), inspector, reorder, body slots"
repo = "aminx-hub"
size = "L"
depends_on = ["S6-04", "S6-05", "S6-06", "S5-02"]
gate = "Playwright on demo harness: structure->model->viewer incl. palette drag, inspector param errors, fan-out and fan-in, keyboard-only completion, axe zero serious violations, cycles unrepresentable"
user_decision = false

[[item]]
id = "S6-11"
title = "Demo app with fake ExecutionClient and S5-04-shaped catalog fixture, URL-fragment and file sharing, and the published contract suite for external packages"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-10", "S6-07", "S6-35"]
gate = "Playwright on demo: list pipeline runs with fake client and node states render; URL round trip keeps graph hash; contract suite importable by a separate S5-30-shaped consumer fixture that fails a seeded-broken client"
user_decision = false

[[item]]
id = "S6-12"
title = "pipe-canvas element on the chosen engine behind CanvasEngine adapter with typed connect veto, encoding warning and shadow-DOM interaction"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-09", "S6-04", "S6-05", "S6-06", "S5-02"]
gate = "Playwright: reject drop vetoed before edge exists, exact/widen drop commits, encoding mismatch commits with warning, shadow-DOM pointer interaction works on the production adapter, layout round-trips exactly"
user_decision = false

[[item]]
id = "S6-27"
title = "Pre-registered canvas performance budget (200 nodes, 400 edges, pan and zoom) with threshold derived from the S6-36 record, positive and negative engine controls, resumable per unit"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-36", "S6-12", "S5-59"]
gate = "Sidecar committed before the run on the same named host as S6-36 naming its record id; delayed stub engine fails and trivial engine passes; verified by run record outcome"
user_decision = false

[[item]]
id = "S6-13"
title = "Canvas ergonomics: multi-select, copy/paste with node and edge id remap, groups, comments, deterministic auto-layout for layout-less documents"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-12"]
gate = "Playwright: paste remaps ids and undo restores exactly; groups/comments round-trip; auto-layout deterministic"
user_decision = false

[[item]]
id = "S6-14"
title = "Keyboard and screen-reader connect flow with live-region announcements"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-12"]
gate = "Playwright + axe: keyboard-only connection completes, rejection reason announced, zero serious violations"
user_decision = false

[[item]]
id = "S6-15"
title = "Run overlays in both elements: executor and PrivacyDescriptor chips, dashed transfer edges, reuse badges, list/canvas switch"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-12", "S6-11"]
gate = "Playwright on demo with fake plans: remote-segment fixture reads leaves-your-machine, cross-segment edges dashed, reused nodes marked, view switch preserves graph hash"
user_decision = false

[[item]]
id = "S6-16"
title = "code.python node: inspector, explicit port table, allowed-imports lint with E_CODE_IMPORT_UNAVAILABLE, lock-to-JSON Node CLI with explicit --lock argument, checked against S4-31's code-node vectors"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-10", "S4-31"]
gate = "Tests with allowed-imports fixture: jax/onnxruntime/torch flagged, numpy passes, absent allowedImports turns the lint off visibly, CLI output changes with the lock fixture, pinned S4-31 builtin vectors pass or the missing-vector request is recorded"
user_decision = false

[[item]]
id = "S6-17"
title = "Code-node trust UX against the fake client: inert code nodes for foreign documents, codeTrusted, per-hash trust-requested"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-16", "S6-11", "S6-35"]
gate = "Playwright with fake client: URL-loaded doc opens inert, run-requested carries codeTrusted false, trust-requested fires with the graph hash, an edited graph or executor_hint is untrusted again"
user_decision = false

[[item]]
id = "S6-28"
title = "Code-node end-to-end through S5-30's HubExecutionClient and S5 code.python: untrusted refusal, trusted numpy run, egress canary with reachable control"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-17", "S5-30", "S5-43"]
gate = "Playwright on the hub page: untrusted doc refused, trusted numpy node runs, canary fetch fails while the same fetch succeeds without the sandbox; if the canary cannot be made to fail the item stops for the Q4 user decision"
user_decision = true

[[item]]
id = "S6-18"
title = "Export pipeline to ipynb for JupyterLite (cell per node in topological order, inputs snapshotted)"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-16"]
gate = "Exported notebook validates against nbformat schema, cell order is a valid topological order, opens in JupyterLite smoke test"
user_decision = false

[[item]]
id = "S6-31"
title = "Hub build step emitting allowed-imports JSON from the pinned Pyodide lock and asserting agreement with the S5-43 executor allow-list"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-16", "S5-17", "S5-43"]
gate = "Editor list equals S5-43 executor-side list for the same lock, and a planted extra package in a copy of the lock changes both"
user_decision = false

[[item]]
id = "S6-29"
title = "Upgrade the public hub Pipelines page from pipe-list to pipe-canvas (list kept as the accessible view) and record the live-origin deploy date, served sha256 and release tag in the checkpoint criteria file"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-12", "S6-15", "S6-27", "S5-30", "S5-31", "S5-33"]
gate = "Playwright on the live public origin: canvas default with list available, two catalog nodes dragged, wired and run via HubExecutionClient with both outputs seen, served bundle sha256 equals checksums.txt; deploy_date and served sha256 written to the criteria file"
user_decision = false

[[item]]
id = "S6-19"
title = "Praxis go/no-go decision against pre-committed criteria (canvas live on the public origin for at least 60 days from the recorded deploy date); records outcome go, no-go or defer"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-29", "S5-31"]
gate = "Criteria file committed before review with deploy_date; check_checkpoint.py passes only when review date is at least 60 days after deploy_date (59-day fixture fails); checkpoint_outcome.toml records go, no-go or defer"
user_decision = true

[[item]]
id = "S6-34"
title = "User creates the empty pipecanvas repository and records its slug (go outcome required)"
repo = "pipecanvas"
size = "S"
depends_on = ["S6-19"]
gate = "check_checkpoint.py --require go passes on the committed outcome file and fails on a no-go fixture; git ls-remote resolves the new repo and its slug is recorded"
user_decision = true

[[item]]
id = "S6-20"
title = "Hub-side extraction: subtree split pushed to the new repo, tag, and checksummed release tarball including the checkpoint outcome file"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-34", "S6-13", "S6-14"]
gate = "check_checkpoint.py --require go first; git ls-remote shows the tag, tarball and checksums.txt exist, clean-directory consumer smoke test from the tarball passes (core in Node, elements via import map), history preserved"
user_decision = false

[[item]]
id = "S6-30"
title = "Hub consumes the extracted pipecanvas release: vendor.lock.json entry, import map, CI, delete the in-repo copy"
repo = "aminx-hub"
size = "S"
depends_on = ["S6-20"]
gate = "Hub CI green with vendor.lock.json sha256 equal to the release tarball, no references to packages/pipecanvas remain, Pipelines page Playwright spec still passes"
user_decision = false

[[item]]
id = "S6-21"
title = "Praxis domain pack with pack-defined graph schema and read-only pcgToDocument adapter over the 28 graph fixtures, consuming the release tarball (go outcome required)"
repo = "praxis"
size = "M"
depends_on = ["S6-20", "S6-13"]
gate = "Tarball's checkpoint_outcome.toml says go; adapter converts all 28 plr-sema/tests/fixtures/*_graph.json files (glob count asserted 28) to documents that pass core validation with the praxis pack; nested-regions fixture yields contains/order/parent as specified; full PCG preserved at x-praxis.pcg; tests run in praxis CI or on titanix"
user_decision = false

[[item]]
id = "S6-22"
title = "Praxis host: read-only pipe-canvas in protocol detail view; resolve zoneless and CUSTOM_ELEMENTS_SCHEMA assumptions"
repo = "praxis"
size = "M"
depends_on = ["S6-21"]
gate = "Build and tests in praxis CI or on titanix: op-node count equals len(operations) and resource-node count equals len(resources) on three fixtures; element event updates signal and re-renders under both zoneless prod and zone.js test configs"
user_decision = false

[[item]]
id = "S6-23"
title = "Optional notebook bridge: render pipe-list inside a JupyterLite notebook cell and return edited document JSON to Python"
repo = "aminx-hub"
size = "M"
depends_on = ["S6-10", "S6-18", "S5-35"]
gate = "In the /lab/ page a notebook cell renders pipe-list for a document and Python receives edited JSON; round trip preserves graph hash"
user_decision = true

[[item]]
id = "S6-32"
title = "Optional npm publish of the pipecanvas packages (publish performed by the user); nothing depends on it"
repo = "pipecanvas"
size = "S"
depends_on = ["S6-20"]
gate = "npm pack --dry-run lists only built files and the declared version; publish performed by the user"
user_decision = true
```

## Revision log

### Round 1 (adversarial spec coherence cycle)

Applied (all 16 objections; C13 and C14 partial as ruled):

- **C1 (tokens in depends_on).** Every `S<n>:<contract>` token is gone from section 9; `depends_on` holds item
  ids only. Bindings (section 6): S4-04, S4-06, S4-07, S4-09, S4-10, S4-12; S5-01, S5-02, S5-04, S5-08, S5-10,
  S5-17, S5-30, S5-31, S5-35, S5-37, S5-43 (S5-12, the viewer adapter, is consumed through the body slot only and
  is not a hard dependency). `S5:pyodide-code-exec` is S5-43, which now exists in S5 (verified, A35); Q11
  narrowed accordingly.
- **C2 (IR shape).** 4.2 re-anchored on S4's real v2 fields (`node_type`, flat `src/src_port/dst/dst_port`
  edges, `models` lock). Edge ids dropped (derived `edgeKey`), edge `kind` removed from the hub IR, executor hints
  moved to `extensions["x-pipecanvas"]` (unhashed), inline code node expressed as the reserved `builtin:` scheme
  (S6-25). A24 re-statused VERIFIED with S4 citations (new row A36 for the missing corpus).
- **C3 (corpus, type model).** 4.3 rewritten to implement S4 R1-R3 (`accepts`, encodings, symbolic dims,
  explicit adapters); the nominal/dtype/rank relation is dropped. New item S6-26 authors the corpus and JCS
  vectors (S4 has none, A36); S6-03 and S6-04 depend on it; Q5 revised; Q14 asks who carries S6-25/S6-26.
- **C4 (adapter).** New explicit `HubExecutionClient` item S6-11 (owner S6, which matches S5's own record at
  `261001_aminx-hub.md:335-337`); depends on S6-07, S5-37, S5-30, S5-10. Q9 resolved as a stated default. Request
  to S5: add S6-11 to S5-31's `depends_on` (S5 edits its own block). The old S6-11 (demo app) is now S6-24.
- **C5 (param profile).** 4.4 binds the form model to `xtrax-param-profile@1` and `x-xtrax` keys; `x-widget`
  replaced by `x-xtrax.widget` with S6's `EditorUiHints` values. A34 split: params provided, `required` and
  `cardinality` still requested (S6-25), with fallback defaults.
- **C6 (browser harness edge).** S5-02 added to S6-02, S6-08, S6-10, S6-12. Where perf and bake-off run: one
  named host recorded in the sidecar, frame-time never compared across hosts, CI runs functional gates only (4.11,
  Q15, A41).
- **C7 (S6-12 split).** S6-12 is now engine adapter plus element plus veto and shadow-DOM gate (size M); the
  pre-registered performance budget is S6-27, depending on S6-08 (record id) and S6-12.
- **C8 (S6-17 split).** S6-17 is the Trust UX against the fake client (no S5 dependency); S6-28 is the end-to-end
  integration and canary, depends on S5-43, and carries `user_decision = true` for the Q4 fallback.
- **C9 (checkpoint).** New S6-29 upgrades the Pipelines page to `<pipe-canvas>` and records the deploy date;
  S6-19 depends on S6-29 and S5-31 (not S6-17) and its gate checks the 60-day rule from the recorded date. Q10
  resolved in the DAG.
- **C10 (extraction).** `pipecanvas` and `xtrax` added to `owner_repos`. One path (`packages/pipecanvas/`, repo
  root after the split). S6-20 now ends in a tag plus checksummed release tarball (verifiable); new S6-30 migrates
  the hub to it and deletes the in-repo copy; praxis consumes the same tarball by URL plus sha256 (Q16); npm
  publish is optional S6-32. Bathos slug is the slug of the repo holding the script at run time, never a repo that
  does not exist yet.
- **C11 (praxis mapping).** 4.9 specifies `pcgToDocument` (nodes, `order`/`contains`/`uses`/`satisfies` edges,
  `parent` layout, `x-praxis.pcg` verbatim), makes `graph` pack-defined via `GraphAccess`, fixes the fixture
  predicate to the 28 `*_graph.json` files, and defines the S6-22 count rule as `len(operations)` (regions and
  bodies included). Read from the real fixtures: `execution_order` omits nested nodes in 12 of 28 files (A38).
- **C12 (allow-list source).** The lint takes an allowed-imports JSON and reads no lock; the CLI takes `--lock`
  explicitly (env fallback, no default). New S6-31 wires the hub's pinned lock and asserts agreement with S5-43;
  it depends on S5-17 and S5-43.
- **C13 (partial).** `related_specs` is now S1, S4, S5 (S3 dropped, no S3 contract is consumed); `pipecanvas` and
  `xtrax` added to `owner_repos`. Declined: renaming `aminx-hub` in the frontmatter, because S5-33 renames the
  repo later; a note in the header explains the name.
- **C14 (partial).** A13 now cites a persisted re-read of the rete 2.0.6 `Socket` class body and an explicit NOHIT
  count (`recheck_rete_socket_r1.txt`); A24 and A34 re-statused from S4 line citations. Declined: editing
  `probe_rete_jcs.tsv` itself, because it is the original probe record; the re-check is a new file.
- **C15 (bench resolver).** The resolver is Python, in `scripts/bakeoff/bench_dir.py`, following S5-03's layered
  pattern (no `package.json` layer; a `[tool.pipecanvas-bench]` table in the bake-off directory's
  `pyproject.toml`), with a malformed-config test as part of the S6-08 gate.
- **C16 (spec key, share limit).** Spec-key input is `graph.models[...].manifest_sha256`; the fragment cap is
  32 KiB with `E_SHARE_TOO_LARGE`, `E_SHARE_DECODE` and a file-download fallback (the cap itself is an unverified
  constant, A39).

Declined or not applied: none of the 16 objections was rebutted in full; the two partial declines are noted under
C13 and C14.

Also re-verified in this round: S5's ledger citations A28-A31 now point at S5's current line numbers (S5 moved
under revision); the S4 line citations are as of S4's current text. Both specs are concurrent and may move
again, so the coherence pass re-checks ids.

### Round 2 note on ids

In the round-1 log above, item ids and S4 anchors are as of round 1; round 2 below supersedes them where they differ.

### Round 2 (adversarial spec coherence cycle)

Before editing, S4 and S5 were re-read in full where S6 leans on them. S4's file changed between two reads during
this round (its own round 1 landed: it now lists S6-25/S6-26 as superseded and re-points S6-03/S6-04/S6-16), so all S4
line citations were re-derived against the later state. All S5 citations were re-derived against its current text.

Applied:

- **C2-1 (conceded, blocking).** Rebuilt every S4 anchor from S4's file. The document (4.2) is S4's graph IR v2
  verbatim: `ref`, edge `id`/`kind`, `models` lock `{"<id>@<ver>": sha}`, `builtin:code.python` with node-level
  `source`/`ports`/`requirements`, `executor_hint` (hashed), `x-` keys. Removed `node_type`, `#operation`, `edgeKey`,
  `E_DUP_EDGE` (now the pack warning `W_EDGE_DUPLICATE`), `E_UNKNOWN_NODE_TYPE` (now S4's `E_UNKNOWN_REF`),
  `extensions["x-pipecanvas"].executor_hints`, R1-R3, `accepts`, R3 dimension unification and `E_DIM_MISMATCH`. The
  palette is one entry per manifest; compat is `exact | widen | reject`; `W_ENCODING_ADAPTER` is a warning. Ledger rows
  A24, A34, A36 are REFUTED with `changed:` notes and replaced by A42-A45. **Dropped S6-25 and S6-26** (ids retired);
  S6-03/S6-04/S6-16 depend on S4-09, S4-10, S4-30, S4-31 instead; the new S6-35 owns only the TypeScript JCS and share
  codec. Q12 and Q14 reworded.
- **C2-2 (conceded, blocking).** S5-30 owns `HubExecutionClient` (A47; A31 REFUTED). S6-11 is now the fake-client demo
  plus the published contract suite (the old S6-24 folded in; S6-24 retired) and does not depend on S5-30, so there is
  no cycle with S5-30's existing `depends_on` of S6-03, S6-07, S6-11. S6-28 and S6-29 depend on S5-30. 4.6, Q9 and the
  Provides table corrected; the request to change S5-31's `depends_on` is dropped. One detail of the objection was
  stale: S5-30 already lists S6-07 and S6-11, so S5-30 needs no new edge.
- **C2-3 (conceded).** The sidecar-resolution proof is S5-59, not S5-02 (verified in S5; A35 REFUTED, A48 added).
  S6-08 and S6-27 depend on S5-59. The stdlib Python shim shape, slug `aminx-hub`, and the "never a `bash -c` or
  `npx` wrapper" rule are taken from S5 section 7.
- **C2-4 (partial, conceded the gap).** 4.1 now states how the package coexists with S5-01: no Python in the package;
  the bake-off and perf drivers and the resolver are stdlib-only scripts in the hub's `scripts/studies/`; the bundler is
  a package devDependency never run by the hub build; the output-dir resolver reuses S5-03's `hub.toml [paths]`
  (A54). S6-02's gate now runs S5-01's guard with the package present. Whether the guard scans `package.json` and the
  workspace-sourced vendor entry are unverified (A52, A55) and filed as Q17, not assumed.
- **C2-5 (conceded).** S6-29 depends on S5-33 and its gate checks the live origin (served sha256 equals the release
  build) before writing `deploy_date`. S6-19 is the decision item; new S6-34 and the gates of S6-20/S6-21 run
  `check_checkpoint.py --require go` (with a `no-go` fixture as negative control), so only a go unlocks extraction. S6-20
  also depends on S6-13/S6-14 so the release includes the canvas feature set.
- **C2-6 (conceded).** Palette input is `dist/catalog/catalog.json` with per-executor `availability[]` (A50); `available`
  and `reasons` come from joining that with S5-08's probe report. S6-05's snapshot uses S5-04's emitted fixture. Fields
  S5's mapping table does not list (params, description, port `accepts_encodings`/`alphabet`, manifest sha256) are A51
  (UNVERIFIED) and the new amendment item S6-33, with Q18.
- **C2-7 (conceded).** Spec keys belong to S6-07 only. The share codec belongs to S6-35 only (S6-11 wires its UI). S6-03
  was split: S6-03 is document and gate, S6-35 is canonical hash and share.
- **C2-8 (partial).** S6-08 split into S6-08 (pre-registration PR) and S6-36 (the run); S6-09 and S6-27 re-point to
  S6-36. S6-10 re-sized to L rather than split: S5 binds `S6:pipeline-editor-component` to S6-10 as the whole stage-1
  component and S5-31's gate needs the palette from it, so splitting would force an edit to S5-31's `depends_on`.
- **C2-9 (conceded).** S6-20 is now `repo = "aminx-hub"` (the split, tag and tarball are produced in the hub); the
  user-owned repo creation is the new S6-34. S6-21 and S6-22 name their venue (praxis CI or titanix, no local whole
  suite).

Declined or adjusted:

- Splitting S6-10 (C2-8), reason above.
- The ruling on C2-3 suggested optionally a first-step sidecar check inside S6-08; declined in favour of the explicit
  S5-59 dependency, which is S5's own pre-registered check and avoids duplicating it.
- The objection's mention of S5-15 as the shim check for C2-3 was outdated: S5 now has the check as S5-59.

New or changed ledger rows: A24, A31, A34, A35, A36, A40 REFUTED (with `changed:`); A23, A28-A30 restated against
current lines; A42-A56 added (A46, A51, A52, A55, A56 UNVERIFIED with reasons; the rest VERIFIED by `read:`). No
spike was run: nothing here is runnable without the barred package toolchain, and every VERIFIED row cites the line
that states the fact.

### Coherence round 1 (cross-spec)

- **C12 (manifest id and entry-point vocabulary).** Example refs and the `models` lock key now use S4's id
  `molxmpnn/proteinmpnn.sample` (S4 section 4.3, `261001_xtrax-model-contract.md:537`) at version `0.2.0a4`: `0.2.0a3`
  was the last aminx release and the first molxmpnn release is `0.2.0a4` (S3 section 4.9). The ref is an opaque string
  to S6 (section 2); S6 quotes it only in examples and fixtures, and fixtures re-read it from S4-19's committed
  manifest instead of carrying a literal. S4 owns the final id and entry-point strings (S4-19); S3 does not yet provide
  the `MolxmpnnIdentity` contract S4 consumes. That gap is open on the S3/S4 side, not in S6.

### Coherence round 2 (cross-spec)

- **CH2-05 (manifest sha256 meaning).** Three meanings existed: S4's sha256(JCS(manifest)) for the `models` lock, the
  sha256 of the release asset S5 pins in `catalog/sources.toml`, and an unstated hash in S5-19's `manifest_sha256` echo.
  A51 now says the palette/lock value is the JCS hash that the `models` lock expects and is distinct from the asset
  hash. S6-33 now depends on S6-35 (the TypeScript canonicaliser, the only way the hub's Node builder can compute the S4
  form) and owns the pass-through of `manifest_sha256`; its gate compares the value with S4-30's Python value per
  fixture manifest. The request that S5 state its wire echo and RunStore unit key use this same hash is filed in the
  Q18 row and section 6. A51 stays UNVERIFIED (deferred) until S5 lists the fields.
- **CH2-06 (shared builder).** S6-33 keeps depends_on S5-04 and S4-06 and adds S6-35. S5's mapping table (4.2) does not
  yet list the fields, so A51 remains UNVERIFIED and is to be re-statused once S5 lists them. Q17 now points at the S5
  section 6 row so one decision answers Q17, Q18 and the hash definition. Not edited here: S5 itself (patched
  concurrently); the S5 side must add S6-33 to its Consumes/Provides notes.

### Convergence check (round 2 residue C2-1: stale S4 line citations)

- **C2-1 (S4 line citations).** The substance of C2-1 was already fixed in round 2 (4.2 embeds S4's graph IR v2 with node
  `ref`, edge `id`/`kind`, the `models` lock and `builtin:code.python`; S6-25 and S6-26 are gone). What remained was that
  the `:NNN` citations to the S4 file had been taken before S4 grew and no longer showed the facts they were cited for
  (ledger rows marked VERIFIED with `read:` lines that landed on unrelated S4 text). Every S4 citation was re-derived
  against the S4 file at 1866 lines: section 3.1 (the "S4's contract as it stands" bullet), 4.2 (hash line), 4.3 (type
  model, adapter, pack-level codes, params, port `required`/`cardinality`, UI hint vocabulary), 4.7 (code-node
  contract), section 6 (UI hint token) and ledger rows A24, A34, A36, A40, A42, A43, A44, A45, A46. In prose, each S4
  citation now names the S4 section and a verbatim quote, with the line as a hint only (the words "hint" and the number); in the
  ledger the `read:` lines are current line numbers with a parenthetical naming the S4 section, because the ledger
  format requires a line.
- **Anchors used** (S4 file, as of this check): Graph IR v2 shape block and `ref` grammar (S4 4.6), the `x-` key rule,
  "Canonical hash", "Validation" (codes, corpus, `W_NO_EXECUTOR`/`W_TRANSFER_LARGE` belong to S6's pack), the manifest
  hash definition (S4 4.3 "Manifest hash", previously uncited here), manifest ports with `required`/`cardinality`/
  `accepts_encodings`, `compat` with the alphabet rule (S4 4.2), S4-30, S4-31, the "UI hint vocabulary" Consumes row, and
  the "Edges owed by other specs" S6 row.
- **Two content corrections found while re-reading.** A44 said S4-31 depends on S4-06, S4-10, S4-30; S4-31 now also
  depends on S4-47 (the profile engine), so A44 lists it (no S6 item changes: no S6 item depends on S4-47). A44 now also
  names S4-30's manifest-hash vector, which S6-35 and S6-33 must reproduce (already required in 4.2 and CH2-05).
  A46 stays UNVERIFIED, re-checked against the current S4 text: S4 4.6 shows `"ports": {"in": [], "out": []}` for the
  code node, S4-10's title mentions "builtin code nodes", and S4-31's gate has no code-node vector.
- **S4 moved during the check.** The S4 file went from 1854 to 1866 lines while this pass ran (another revision was
  editing it), so the line numbers above are valid for that instant only. The quotes are the stable anchors; the coherence
  pass should re-derive the numbers rather than trust them.
- **Not done, left to the coherence pass.** (1) The S5 line citations (A28-A31, A35, A47-A55, the S5 bullet in 3.1, and
  the `S5 file :NNN` mentions in 4.6 and section 6) were not re-derived; S5 has grown to about 1600 lines, so they are
  stale in the same way and are now flagged as such in the opening note. (2) A51 is still UNVERIFIED with the reason
  "S5's mapping table lists no params, description, `accepts_encodings`, `alphabet` or manifest hash"; S5's current 4.2
  mapping table appears to list them, so A51 can probably be re-statused by a `read:` once its line is re-derived.
  (3) The matching S4 text fix: S4's risk row said S6-03 runs the TS side of the JCS vectors, which is S6-35; corrected in
  the S4 file in the same pass.
- No backlog item, `depends_on` or id changed; the single `[[item]]` TOML block is untouched.
