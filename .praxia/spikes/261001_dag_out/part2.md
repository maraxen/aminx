
## 4. Critical path

The script computes the longest chain two ways. Both chains start with the rename (S3-01 to S3-13), so the
rename gates everything else.

**By item count: 28 items, size weight 44 (S=1, M=2, L=3).**

S3-01* → S3-24 → S3-06 → S3-07 → S3-08 → S3-10 → S3-12 → S3-13* → S2-03 → S2-21 → S2-04 → S2-05 → S2-32 →
S2-25* → S2-06 → S2-22 → S2-08 → S2-27 → S2-28 → S2-10 → S2-11 → S2-13 → S2-24 → S2-14 → S2-15 → S2-18* →
S2-20* → S3-20*

This is the rename, then the full EBM extraction:
- freeze;
- A0 baseline on the GPU;
- filter-repo into `ebmx`;
- seams and isolation;
- tier-1 goldens;
- B1 on the GPU;
- cutover;
- close-out;
- the checkout-rename quiescent window.

Six of its items are your decisions (`*`). It also needs side gates that are not on the chain:
- S2-30, GPU budget, for S2-05, S2-32 and S2-14;
- S2-01, name and licence, for S2-06;
- S2-31, converter branch, and S2-17, remote creation, for S2-18.

Four items are GPU campaigns on the node class NC: S2-05, S2-32, S2-13 and S2-14. S2 estimates about 7.7
GPU-hours for the two full arms, plus the floor. Each item is chunked, resumable and pre-registered.

**By size weight: 25 items, weight 46.**

S3-01* → S3-24 → S3-06 → S3-07 → S3-08 → S3-10 → S3-12 → S3-13* → S4-16* → S4-35 → S4-17 → S4-18 → S4-19 →
S5-14* → S5-52* → S5-53 → S5-15 → S5-30 → S5-31 → S6-29 → S6-19* → S6-34* → S6-20 → S6-21 → S6-22

This is the rename, then:
- the MPNN ONNX route through the xtrax bundle exporter;
- the native manifests;
- the molxmpnn browser-assets release;
- the hub ORT-Web executor and pipeline runner;
- the public Pipelines page;
- the praxis go/no-go;
- extraction of `pipecanvas`;
- the praxis host.

S4-35 also waits for the xtrax release S4-34. S4-34 fans in 13 items, including S5-51, the G1-G6 amendments.

**Wall-clock caveat.** S6-19 includes a pre-committed soak: the canvas must be live on the public origin for at
least 60 days (S6 4.10). In calendar time that soak dominates everything after it (S6-34, S6-20, S6-21, S6-22,
S6-30, S6-32), whatever the item sizes. Nothing outside S6 depends on those items.

## 5. Items that can start immediately

There are 14 items with no `depends_on`. The count of items that transitively depend on each one is in
parentheses, as a measure of leverage.

Ten roots are decision items, so you have to act before any work follows from them:

| Root | What you decide | Downstream |
|---|---|---|
| S4-01 | xtrax contract ADR: separate `xtrax-contract` dist, names, ship discovery now, version and tag policy, toolchain | 137 |
| S3-01 | Rename ADR: name, repo strategy, PyPI policy, R/E ordering, bathos slug, HF weights option | 127 |
| S2-02 | Confirm that maraxen/aminx#174 (xtrax 0.4.0a11 pin) is merged and record its SHA. The executor does not merge it | 120 |
| S5-01 | Hub repo skeleton. You create `maraxen/aminx-hub` and set its licence and Node toolchain | 84 |
| S6-01 | pipecanvas name, home and extraction trigger | 62 |
| S4-37 | Weights licence and redistribution value per MPNN weight set, with a named reviewer | 34 |
| S2-01 | EBM project name, licence, visibility, install channel, clone location | 23 |
| S5-16 | isochore: move `mdtraj` to an extra (changes your package's published dependency contract) | 11 |
| S4-49 | Whether S4-22 supersedes S3-18, and the fate of `make_mpnn_score` | 7 |
| S5-13 | Fast-forward py2Dmol `plugin-system-impl` into `main` and tag a release | 1 |

Four roots need no decision and can be dispatched now:

| Root | What it is | Downstream |
|---|---|---|
| S4-02 | Pre-registered bathos spike: which annotations of the real pre-S1-07 spec classes the S4-07 mapper can express, and whether any xtrax a11 IR field is a fixed-arity tuple. It must run **before S1-07** because it reads the pre-flip classes | 113 |
| S4-03 | Pre-registered bathos spike: onnx-extra resolution arms and the resolved orbax-checkpoint for each arm | 61 |
| S4-15 | xtrax `hf_weights`: add a revision-pin parameter and report it in `WeightReport` | 57 |
| S4-25 | Pre-registered bathos spike: differential `iree-compile` of one StableHLO module under prolix's flags and under xtrax's WASM32 flags | 1 |

All three spikes commit their `.bth.toml` sidecar with `[outcomes]` before they run (`bth run`, not
`uv run bth`). The most leverage per decision comes from answering S3-01, S4-01 and S2-02, which unblock 127,
137 and 120 descendants respectively (the sets overlap). S2-02 is close to free: it only records a merge you
may already have done.

## 6. Open questions for the user (consolidated)

The six specs ask 84 numbered questions in total (S1 11, S2 15, S3 15, S4 13, S5 11, S6 19). Duplicates are merged here. Each row lists its source
questions (for example `S3 Q4`), the recommendation as the specs state it, and the items it blocks or
parameterises. No answer here overrides D1-D7. Where an answer is recorded inside a decision item (such as
S3-01's ADR), that item is listed first.

### 6.1 Ordering, naming and repos

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-01 | Rename first (R) or EBM cutover and release first (E)? | S1 Q10, S2 Q7, S3 Q4 | **R.** The rename is mechanical and its codemod is reusable. S1 already assumes R. R also keeps a single publisher of PyPI `aminx` (S3-13). Under R, S2 runs on `src/molxmpnn/ebm`, S2-03 follows S3-13, and S2-18 follows S3-07 | S3-01. It parameterises S1-01/02/03/17, S2-03, S2-18, S3-07 and S1-25 |
| OQ-02 | Final MPNN package name | S3 Q1 (D2 proposes `molxmpnn`) | `molxmpnn`. Probe availability (S3-01) and reserve the name with a real `0.0.0.dev0` upload (S3-27) on the same day | S3-01, S3-27 |
| OQ-03 | Repo strategy and hub slug | S3 Q2, S5 Q1 | Option 1: rename `maraxen/aminx` in place to `maraxen/molxmpnn` and create `maraxen/aminx-hub` now. Never recreate `maraxen/aminx` | S3-01, S3-02, S5-01, S5-33 |
| OQ-04 | After the deprecation window, does the hub take PyPI `aminx` (P2) or does the shim keep it (P1)? Are the P2 soak and download thresholds right? | S3 Q3, S3 Q14, S5 Q1 | **P1**: the shim stays `aminx`, and any hub Python tooling is `aminx-hub`. The thresholds (>= 90 days, <= 50 weekly downloads, zero consumer hits) are accepted as written, but P1 needs none of them | S3-22, S3-23, S5-55 |
| OQ-05 | Bathos slug for the MPNN repo | S3 Q5 | New runs go under `molxmpnn`. The `aminx` slug is frozen (144 runs) and the hub gets `aminx-hub` | S3-01, S3-20 |
| OQ-06 | Rename the local checkout `~/projects/aminx` to `~/projects/molxmpnn`? | S3 Q7 | Yes, once, through the S3-20 runbook with no open worktrees | S3-20 |
| OQ-07 | HF weights repo: (a) copy to new, (b) keep `maraxen/aminx`, or (c) rename in place? | S3 Q8 | (c) if the S3-11 probe confirms the redirect and that SHAs are kept, otherwise (a). Never (b) | S3-01, S3-11, S4-19 |
| OQ-08 | Shim warning class | S3 Q6 | `DeprecationWarning`, plus the always-visible CLI banner and the README/CHANGELOG notice | S3-01, S3-10 |
| OQ-09 | Fold the earlier `prxteinmpnn` rename's residue (64 files) into this rename? | S3 Q9 | Yes (S3-09) | S3-09 |
| OQ-10 | Rename `browser/aminx-sampler` to `molxmpnn-sampler` in the codemod? | S3 Q10 | Yes. S5 consumes the new names | S3-06, S3-07 |
| OQ-11 | Fate of the legacy Pages URL `maraxen.github.io/aminx/` | S3 Q15 | Accept that it goes dead, unless S3-21's probe finds a cheap redirect that does not recreate `maraxen/aminx` | S3-21, S5-34 |
| OQ-12 | EBM project name | S2 Q1 (D3: not molxmpnn) | `ebmx`, always naming the checkpoint by its filename | S2-01 (and every `ebmx` item) |
| OQ-13 | EBM licence and attribution. Upstream `jproney/ProteinEBM` has no LICENSE locally and is derived from Boltz-1 | S2 Q2 | Do not publish until upstream's GitHub licence has been read. If it is permissive, license the new code MIT with a NOTICE. If there is none, ask upstream first | S2-01, S2-16, S2-17 |
| OQ-14 | EBM remote, visibility and install channel | S2 Q3 | Private `maraxen/ebmx`, installed by git URL, until OQ-13 is resolved. Until then, a local branch plus rsync | S2-01, S2-17, S3-25 |
| OQ-15 | pipecanvas name and home | S6 Q1 | Keep `pipecanvas` as a placeholder. Start inside the hub repo behind an enforced boundary, and extract it only on a recorded praxis "go" (you create the repo at S6-34) | S6-01, S6-34, S6-20 |

### 6.2 Spec unification (S1)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-16 | Which representation survives: nested `RunSpec` (A) or the facade (B)? | S1 Q1 | A: nested `RunSpec` is the model, and the flat vocabulary becomes a permanent codec. Confirm at S1-04, after the S1-02 and S1-03 records exist. Fall back to B if the S1-03 probes refute A | S1-04 |
| OQ-17 | When do flat reads and `.run_spec` break? | S1 Q2 | The shims survive at least one published alpha (S1-25), and consumers are migrated first (S1-26..29). Only then does S1-20 remove the shims | S1-20, S1-25 |
| OQ-18 | Fix the `or`-coalescing quirks (`random_seed=0` becomes 42, and so on)? | S1 Q3 | Preserve them in S1. Fix them separately in S1-22, with a changelog entry, because the fix changes sampled outputs for seed 0 | S1-22 |
| OQ-19 | Owner of the Potts TRW spec and its conformance mechanism | S1 Q4 | `mistypotts` owns it. aminx vendors a copy pinned by sha256 and tests it without importing it. The dev-dependency variant is used only if `uv lock --check` passes | S1-17, S1-30 |
| OQ-20 | Fold Potts into `RunSpec`? | S1 Q5 | Not now. Share a `SpecFamily` protocol only, and revisit with S4 | none (no item) |
| OQ-21 | Pruning knobs that are accepted but ignored | S1 Q7 | Delete scaffolding fields outright. Provably inert public kwargs go through warn-and-ignore for one window. S1-16 asks item by item | S1-16 |
| OQ-22 | Fate of portable v2 (no consumer) | S1 Q8 | Freeze it and mark it deprecated until S4 ships the schema-generated format (S4-20), then delete it | S1-08, S4-20 |
| OQ-23 | `RunSpec.run_id` is never populated | S1 Q9 | Follow-on outside this DAG: derive it deterministically from the canonical payload plus the code SHA. S1 keeps the field wired | none (file as a follow-on) |
| OQ-24 | Consumer migration scope. Does hautespout get bumped or stay pinned? | S1 Q11; S3-17 | S1 owns the migration branches for mpnn_ext, asr and hautespout, on the same branches as S3-14/15/17. tev_design gets only a handoff (S1-29, S3-16). If S3-17 leaves hautespout pinned, S1-28 is moot | S3-17, S1-28, S1-20 |

### 6.3 EBM extraction evidence and gates (S2)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-25 | Approve the GPU budget and the node class NC | S2 Q8 | Approve about 7.7 GPU-hours for the A0 and B1 arms, plus about 1 hour for the floor. Use `pi_so3` with an explicit `--gres` type and no `mit_preemptable` fallback, chunked and preemption-safe | S2-30. Gates S2-05, S2-32, S2-33, S2-34, S2-14 |
| OQ-26 | How to handle the A0 outcome | S2 Q12 | On a `fail`, investigate dependency drift against the July baseline. On a `marginal`, proceed with A0 as the reference. The move gate is A0 vs B1 | S2-25 |
| OQ-27 | Converter mismatch (`converter_matches_legacy = false`) | S2 Q14 | Fix the converter first if the differing leaves are few and explainable. Otherwise ship the legacy-tree path only, with the limitation documented | S2-31, S2-17, S2-18 |
| OQ-28 | Near-threshold parity misses (above tolerance but within 10x) | S2 Q15 | Accept only if the delta traces to the `chunked_map`/`scan` op order and A0-vs-B1 stays within TOL_AGG | S2-13, S2-14 (decided at review) |
| OQ-29 | Close the old claim campaign `aba3bfe6` with A0/B1 | S2 Q11 | Yes, but only through S2-29. The old claim stays byte-identical until then | S2-29 |
| OQ-30 | Disposition of the untracked evidence | S2 Q9, S2 Q13 | Keep `outputs/ebm_benchmarks*` tracked as they are, and import the extras under `untracked_import/`. Delete the root directories only after the pushed copy re-hashes clean, and keep the tarball outside every repo | S2-07, S2-20 |
| OQ-31 | Publish the converted orbax weights? | S2 Q10 | Not until OQ-13 is settled. Ship the converter and `weights.lock.toml` instead | S2-12, S2-17 |
| OQ-32 | Should `aminx.ebm` forward to `ebmx` after the cutover? | S2 Q4 | Flip the shim row to "forward if installed" in S3-25, using the install command recorded at S2-01 | S3-25 |
| OQ-33 | alphex as a runtime dependency of ebmx | S2 Q5 | Dev-only conformance test now, and the runtime edge after the partition work | S2-27 |
| OQ-34 | SwiGLU | S2 Q6 | Copy it verbatim now, and file a debt item for a bias-free `Transition` | S2-26 |

### 6.4 xtrax contract (S4)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-35 | Contract packaging and names, and whether to ship discovery now | S4 Q1, S4 Q4 | F2: a separate `xtrax-contract` dist with Python >= 3.11 and no dependencies, plus `xtrax.contract` as a shim. Ship `xtrax.models` discovery with manifest-as-data entry values. Accept the version, tag and toolchain policy | S4-01 (137 descendants) |
| OQ-36 | orbax conflict on the onnx extra | S4 Q3 | Option A: a consumer-side override plus a dependency group, with `xtrax[export]` moved out of the base deps. Fall back to D if S4-03 shows A breaks the scripts | S4-16 |
| OQ-37 | Owner of the browser knob document, and `chains_to_design` vs `chain_id` | S1 Q6, S4 Q2 | S4 owns it (S4-40 builds it, S4-41 gives it golden vectors, S4-20 generates from it). S1 builds no `DesignConstraints`. Keep the browser vocabulary and map it in the lowering | S4-33, S4-40, S4-20 |
| OQ-38 | prolix WASM flags | S4 Q7 | Let the S4-25 differential result decide. Keep the 50 MiB gate either way | S4-26 |
| OQ-39 | Release channel for `xtrax-contract` and `xtrax` | S4 Q11 | PyPI for both. Publishing is your action | S4-34 |
| OQ-40 | Hosting of the MPNN ONNX graphs and golden vectors | S4 Q12 | GitHub release assets of the MPNN repo (S5-52 rewrites the placeholders) | S4-19, S5-52 |
| OQ-41 | Accept S5's G1-G6 contract amendments into v1 before the first release? | S4 Q13, S5 Q9 | Accept. They are additive and nothing has been published yet | S5-51, S4-34, S5-42, S5-64 |
| OQ-42 | S3-18 vs S4-22 (the proteinsmc guard), the fate of `make_mpnn_score`, and how the end-to-end test gets molxmpnn without a runtime edge (D6) | S3 Q11, S3 Q13, S4 Q9 | S4-22 supersedes S3-18 (S3-18 is closed unmerged), so a single manifest-driven guard survives. Change `make_mpnn_score` to `(structure, decoding_settings="random", *, scorer_id=None)` and keep the `"mpnn"` key. Put the e2e install in a non-default dependency group. If you keep S3-18, use the root-only `find_spec("molxmpnn")` guard | S4-49. Gates S3-18, S4-22, S4-48, S4-23 |
| OQ-43 | `metadata.nl_description` is required per node | S4 Q10 | Keep the invariant. The S6 editor prefills it from the manifest `description`. Not blocking | S4-10, S6-10 |

### 6.5 Hub (S5)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-44 | Hub build toolchain, licence, vendored notices and repo creation | S5 Q5, S5 Q11 | Node/ESM, with Python only as the bathos shim. The hub code is MIT. `vendor.lock.json` requires `license` and `notice` for each entry, and NOTICE is generated and checked (S5-63). The MPL-2.0 legal review and creating the repo are yours | S5-01, S5-63 |
| OQ-45 | Weights licensing review, and weights on Pages vs elsewhere | S4 Q8, S5 Q8 | One decision, one reviewer, one record (S4-37), cited by S5-07. By default the hub carries code and catalog only, and small signed-off ONNX goes to Pages. If you decline, the HF fallback is S5-56 | S4-37, S5-07, S5-56. Gates S4-19 and S5-33 |
| OQ-46 | Two browser MPNN implementations (molxmpnn ONNX vs localfold) | S5 Q2 | Option D: molxmpnn ONNX is the validated default for ProteinMPNN and SolubleMPNN, and localfold is offered for the Ligand/NA families and as a cross-check. Decide after the pre-registered study S5-27. It is descriptive only if the weights differ | S5-46 |
| OQ-47 | localfold integration mode, and whether to list AF3 | S5 Q3, S5 Q4, S4 Q6 (D5) | Link-out with job JSON now (S5-23) and an upstream entry-point request in parallel (S5-25). The library adapter comes only after that, and only with the author's vendoring consent (S5-26). List AF3 as link-out/`byo` only with its terms shown, and never host the params. D5 still holds: the decision is about licensing, not capability | S5-23, S5-25, S5-26, S5-45 |
| OQ-48 | isochore pure-python wheel | S5 Q6 | Yes, move `mdtraj` to an extra | S5-16, S5-54, S5-42 |
| OQ-49 | Merge the py2Dmol plugin system into `main` | S5 Q7 | Yes, after its own suite passes. The hub core does not wait for it; only S5-36 does | S5-13, S5-36 |
| OQ-50 | Driver packaging | S5 Q10 | A single pre-bundled ES module built in molxmpnn, guarded by the S5-50 AST scan | S5-14 |
| OQ-51 | Publish the generated contract TS package to npm, and publish pipecanvas to npm | S4 Q5, S6 Q16 | Not yet for either. Keep contract artifacts in-repo or vendored. Praxis consumes pipecanvas as a release tarball URL plus sha256 | S4-28, S6-32 (nothing depends on either) |

### 6.6 Pipeline editor (S6)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-52 | Is the DAG canvas a goal, given that the list is already DAG-expressive? | S6 Q2 | Yes, build it, but ship the list first and keep it as the accessible view | S6-12 and the canvas chain |
| OQ-53 | How JupyterLite is used, and whether to add an in-notebook widget | S6 Q3, S6 Q13 | Execute in S5's Pyodide worker and export to a notebook (S6-18). The widget S6-23 is optional and not needed now | S6-18, S5-35, S6-23 |
| OQ-54 | Code-node trust model, and the S5-43 sandbox | S6 Q4, S6 Q11 | Foreign documents are untrusted by default, with trust granted per hash. Accept S5-43. If the egress canary cannot be made to fail, run code nodes only for documents authored in the current session | S6-28, S5-43 |
| OQ-55 | Type-system scope | S6 Q5 | S4's `compat` exactly. Defer an "insert adapter" command | S6-04 |
| OQ-56 | Praxis scope | S6 Q6 | A read-only view first, with no authoring | S6-21, S6-22 |
| OQ-57 | Praxis go/no-go criteria (60 days, 3 pipelines, no P1, a written use case) | S6 Q7 | Edit and commit them before S6-19. They are proposals | S6-19 |
| OQ-58 | Canvas engine | S6 Q8 | Rete.js v2 provisionally. The real decision is S6-09, made after the S6-36 bake-off | S6-09 |
| OQ-59 | Which single host runs the bake-off and the performance budget | S6 Q15 | titanix if its pinned-Chromium Playwright lane is verified, otherwise the local workstation. Record the host and never mix hosts | S6-36, S6-27 |
| OQ-60 | Change requests to S5: dependency-guard scope, a workspace-sourced vendor entry, and catalog palette fields with `manifest_sha256` | S6 Q17, S6 Q18; S5 section 6 | Yes to all three, as visible changes (S6-33 is a no-op if the fields are already there) | S6-02, S6-33, S5-01, S5-04 |
| OQ-61 | `executor_hint` is inside the hashed graph, so changing a hint revokes per-hash trust | S6 Q19 | Accept. This is conservative and keeps one IR | S6-17, S6-35 |
| OQ-62 | Confirm-only items: S6 owns the `ExecutionClient` port and S5 owns the adapter; S5-31 binds to the list; S4 is the sole owner of the corpus and JCS vectors; the S4 port-field request is withdrawn | S6 Q9, S6 Q10, S6 Q12, S6 Q14 | Confirm as written | S6-07, S5-30, S5-31, S4-30, S4-31 |

## 7. Mechanical DAG check (by code)

The checker is `.praxia/spikes/261001_dag_check.py` (stdlib only):

```
uv run --no-project python3 .praxia/spikes/261001_dag_check.py --emit-dir .praxia/spikes/261001_dag_out
uv run --no-project python3 .praxia/spikes/261001_dag_check.py --controls
```

It reads the authoritative item list (transcribed to `.praxia/spikes/261001_dag_given_edges.txt`) and,
separately, the single `toml` block in each spec's backlog section. It generated sections 2 and 3 of this file.
Result on 2026-10-01:

| Check | Result |
|---|---|
| Items / edges | 231 / 553 (S1 31, S2 32, S3 27, S4 48, S5 60, S6 33) |
| Exactly one `toml` block per spec | yes, 6/6 |
| Authoritative list vs spec `toml` (id, repo, size, user_decision, depends_on) | **0 differences** (231 = 231) |
| Duplicate ids | none |
| Dangling `depends_on` | none |
| Self-dependencies | none |
| Acyclic (Kahn) | **yes**, all 231 items ordered (the spec `toml` union is also acyclic on its own) |
| S3 assembly row 6: computed MPNN-tree leaf set equals S3-20's non-S3 `depends_on` | **yes** (S1-16, S1-17, S1-20, S1-22, S2-01, S2-20, S2-25, S4-20, S4-21, S4-42) |
| Negative controls: a planted cycle (S3-01 depends on S3-13), a planted dangling id, a planted duplicate, and planted list-vs-toml size drift | all four detected |

The check does not compare titles. Section 3 takes its titles from the spec `toml` blocks. Other S3-28
assembly rows can be read straight off the graph and hold: S2-18 depends on S3-07; there is no S2-19; S2-03
depends on S3-13; S3-05 depends on S2-02; S1-25 depends on S5-52; S1-05, S1-17 and S4-16 depend directly on
S3-13. S3-28 itself remains the item that turns these into a CI-enforced script.
