---
title: "Ecosystem backlog DAG (261001 spec set)"
description: "Combined backlog DAG of the six 261001 ecosystem specs (S1-S6) after the no-rename decision: 213 PR-sized items, mermaid graph, topological order, critical path, startable items, consolidated user questions (hub identity first), what the no-rename decision removed, and a code-checked acyclicity result."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
---

# Ecosystem backlog DAG (261001 spec set)

> **Regenerated after the no-rename decision (2026-10-01).** This file was rewritten from scratch. It replaces the
> earlier 231-item DAG, which assumed the MPNN package would be renamed from `aminx` to `molxmpnn`. Under revised
> D2, the package keeps the name `aminx`, because it is released on PyPI under that name. The hub is a separate
> project with the working slug `aminx-hub`. Nothing from the rename-first plan carries over unless it is restated here.

## 1. Overview and spec index

This plan merges the backlogs of the six 261001 specs into one dependency graph:
- **D4:** unify aminx's two spec systems into one.
- **D3, praxia debt #2369:** move ProteinEBM into its own project.
- **S3:** keep the `aminx` identity, settle the hub's name and clean up stale names.
- **S4:** the xtrax model contract.
- **D1:** the static hub, a separate project.
- **S6:** the shared pipeline editor.

Nothing here runs yet (D7). Only `depends_on` sets the order of work. Each item is one PR or one recorded decision.

**Totals.**
- **213 items**: 102 S, 96 M and 15 L.
- **478 edges.**
- **53 items are `user_decision = true`.** Each one records a choice of yours or needs an action from you, such
  as publishing, creating a repo, wiring DNS, deleting something or approving GPU time.

Section 8 shows the code check. The authoritative item list from the workflow matches every spec's `toml` block
on id, title, repo, size, `user_decision` and `depends_on`.

**What changed in the revision.**
- **No rename.** There is no `molxmpnn`, no `mpnnx` and no `aminx` shim distribution. The GitHub repo, the HF
  weights repo and the PyPI name all stay as they are, and nothing changes hands on PyPI.
- **Consumers keep their imports.** mpnn_ext, asr, hautespout and tev_design still import `aminx`.
- **All other user decisions stand.** D1 and D3-D7 still hold: EBM still moves out (debt #2369), the two spec
  systems still become one, proteinsmc still takes no runtime edge to aminx (D6), and nothing executes yet.

**The hub's identity is the first open question** (OQ-01, decided at S3-01). The options are:
- the repo slug `aminx-hub`, or a name derived from your domain `praxia.science`;
- whether the hub publishes to PyPI at all (never as `aminx`);
- the custom domain.

A GitHub Pages custom domain works with any repo slug, so the repo name and the domain are independent choices.

**Name collision with your praxia orchestrator (flagged).** S3 3.3 lists where the orchestrator already uses
`praxia`: the CLI binary, 138 `praxia-*` crates, the GitHub repo `maraxen/praxia`, a bathos slug with 61 runs, a
registered PyPI project and the `.praxia/` directory in 55 projects. `praxis`, the lab app that consumes S6, is a
near-name. A DNS name under `praxia.science` collides with none of these. A hub repo, PyPI, npm, CLI or bathos name
containing `praxia` would collide with them.

| Spec | File | Items | S / M / L | Decisions | Owner repos (from items) |
|---|---|---|---|---|---|
| S1 spec-system unification | [261001_spec-system-unification.md](../specs/261001_spec-system-unification.md) | 31 | 16 / 14 / 1 | 6 | aminx, mistypotts, mpnn_ext, asr, hautespout |
| S2 ProteinEBM extraction | [261001_ebm-extraction.md](../specs/261001_ebm-extraction.md) | 34 | 22 / 11 / 1 | 10 | aminx, ebmx |
| S3 aminx identity kept: hub naming and stale-name cleanup | [261001_aminx-identity-and-hub-naming.md](../specs/261001_aminx-identity-and-hub-naming.md) | 8 | 7 / 1 / 0 | 2 | aminx, aminx-hub, proteinsmc |
| S4 xtrax model contract | [261001_xtrax-model-contract.md](../specs/261001_xtrax-model-contract.md) | 48 | 22 / 21 / 5 | 10 | xtrax, aminx, proteinsmc, plegadx, prolix |
| S5 hub (working slug aminx-hub) | [261001_aminx-hub.md](../specs/261001_aminx-hub.md) | 59 | 23 / 29 / 7 | 18 | aminx-hub, aminx, xtrax, py2Dmol, isochore, localfold |
| S6 pipeline editor | [261001_pipeline-editor.md](../specs/261001_pipeline-editor.md) | 33 | 12 / 20 / 1 | 7 | aminx-hub, pipecanvas, praxis |

S3 was renamed from `261001_molxmpnn-rename.md`. Its full path is
`/home/marielle/projects/aminx/.claude/worktrees/261001-xtrax-a11-pin/.praxia/docs/specs/261001_aminx-identity-and-hub-naming.md`.
Shared input: [261001_ecosystem-hub-recon-brief.md](../research/261001_ecosystem-hub-recon-brief.md) (D1-D7 with
revised D2).

**Repo labels are partly placeholders.** Items count by repo as follows:
- `aminx`: 70 items. This is the MPNN package and the repo as it is today.
- `aminx-hub`: 78 items. This is the hub's working slug until S3-01 records the real one.
- `ebmx`: 20 items. This is S2-01's recommended EBM name, not yet decided.
- `pipecanvas`: 2 items. This is S6-01's placeholder.

If one of these names changes, the edit is a single pass over the `repo` fields. No edge changes.

## 2. Flowchart

There is one subgraph per spec. Each edge points from a dependency to the item that waits for it. `*` and the amber
fill mark `user_decision = true`. A red border marks the critical path (section 4). The diagram is generated by
`.praxia/spikes/261001_dag_check_v2.py --emit-dir`.

```mermaid
@@FLOW@@
```

## 3. Topological order

This is a deterministic Kahn order: whenever several items are ready, the lowest id goes first. Every item appears
after everything it depends on. The table is generated by the same script, and its titles come from the
authoritative list, which matches the spec `toml` blocks.

@@TOPO@@

## 4. Critical path

The script computes the longest chain twice: once by item count, and once by size weight (S=1, M=2, L=3). Both
measures give the **same chain: 23 items, size weight 44.**

S4-01\* → S4-04 → S4-05 → S4-06 → S4-09 → S4-45 → S4-34\* → S4-35 → S4-17 → S4-18 → S4-19 → S5-14\* → S5-52\* →
S5-53 → S5-15 → S5-30 → S5-31 → S6-29 → S6-19\* → S6-34\* → S6-20 → S6-21 → S6-22

The chain runs in this order:
1. The xtrax contract: the ADR, skeleton, executor vocabulary, `ModelManifest`, generated artifacts and IR composer.
2. The first `xtrax-contract`/xtrax release, and aminx bumping to it.
3. The MPNN ONNX route through the xtrax bundle exporter and its ORT-CPU parity cell.
4. The native manifests.
5. The aminx browser driver and the browser-assets release.
6. The hub catalog entry, then the ORT-Web executor and the pipeline runner.
7. The public Pipelines page and its upgrade to the canvas.
8. The praxis go/no-go, extraction of `pipecanvas`, and the read-only praxis host.

The rename no longer heads the critical path. The previous plan's 28-item chain started with S3-01 to S3-13. **The
contract ADR S4-01 now gates the most work: 136 descendants.**

**Six items on the chain are yours.** S4-01 is the contract ADR. S4-34 is publishing the contract release. S5-14 is
driver packaging. S5-52 is the browser-assets release. S6-19 is the go/no-go. S6-34 is creating the repo.

These gates are not on the chain, but chain items wait for them:
- **S4-34 waits for 13 items.** They include S5-51\* (the G1-G4 amendments), S4-31 (L, the validator and corpus),
  S4-44 (L) and S4-14 (L). S4-14 in turn waits for the spike S4-43 and the conditional S4-46.
- **S4-17 waits for S4-16\*, the orbax resolution.** S4-16 needs the spike S4-03 and S2-02\*, which confirms that
  #174 is merged.
- **S4-19 waits for three items:** S4-37\* (the weights licence) and the Node/wasm and Chromium parity cells (S4-38,
  S4-39).
- **S5-14 waits for S3-12**, the release guard.
- **S5-30 waits for the S6 document and client chain:** S6-03, S6-07 and S6-11.
- **S6-29 waits for three things:**
  - the canvas element S6-12, which waits for the engine decision S6-09\* after the bake-off S6-36;
  - the performance budget S6-27;
  - the first public deployment S5-33\*.

**Wall-clock caveat.** S6-19 has a pre-committed soak: the canvas must stay live on the public origin for at least
60 days. In calendar time, that soak outweighs every item after it, whatever their sizes: S6-34, S6-20, S6-21,
S6-22, S6-30 and S6-32. Nothing outside S6 depends on those items.

**Other long chains.** The script reports these from the longest chain into each sink:

| Ends at | Items / weight | Chain |
|---|---|---|
| S2-20 EBM close-out | 21 / 32 | S2-02\* → S2-35\* → S2-03 → S2-21 → S2-04 → S2-05 → S2-32 → S2-25\* → S2-06 → S2-22 → S2-08 → S2-27 → S2-28 → S2-10 → S2-11 → S2-13 → S2-24 → S2-14 → S2-15 → S2-18\* → S2-20\* |
| S3-29 custom domain (via S5-33, first public deploy) | 18 / 35 | the S4 chain to S5-15, then S5-29 → S5-33\* → S3-29\* |
| S5-46 which MPNN implementations the hub exposes | 18 / 34 | the S4 chain to S5-15, then S5-57 → S5-27 → S5-46\* |
| S1-20 shim removal | 16 / 29 | the S4 chain to S4-19, then S5-14\* → S5-52\* → S1-25 → S1-29 → S1-20\* |
| S1-16 / S1-22 (inside S1 only) | 12 / 22 | S1-01 → S1-02 → S1-04\* → S1-05 → S1-06 → S1-31 → S1-32 → S1-07 → S1-08 → S1-10 → S1-12 or S1-15 → S1-16\* / S1-22\* |

What these chains show:
- **The EBM extraction is self-contained and no longer waits on any S3 item.** It starts as soon as you confirm
  S2-02 and decide S2-35. It needs these side decisions:
  - S2-30, the GPU budget, for S2-05, S2-32, S2-33, S2-34 and S2-14;
  - S2-01, the name and licence, for S2-06;
  - S2-31, the converter branch, and S2-17, creating the remote, for S2-18.
- **The EBM GPU work is chunked, resumable and pre-registered.** Six items run on node class NC: S2-05, S2-32,
  S2-33, S2-34, S2-13 and S2-14. S2 estimates about 7.7 GPU-hours for the two full arms, plus about one hour for
  the floor.
- **S1's own work is 12 items deep.** The S1 shim removal is longer because the release that carries the flip
  (S1-25) follows the browser-assets release (S5-52). That is the release order in OQ-03. If you reverse it, S1-20
  drops back to S1's own depth, and S5-33 inherits the S1 chain instead.

## 5. Items that can start now

**19 items have no `depends_on`.** The number in parentheses after each one counts the items that transitively
depend on it, as a measure of leverage.

### 5.1 No decision needed (9): can be dispatched now

| Root | What it is | Downstream |
|---|---|---|
| S4-02 | Pre-registered bathos spike: which annotations of the real pre-S1-07 spec classes the S4-07 mapper can express, and whether any xtrax a11 IR field is a fixed-arity tuple. **It must run before S1-07**, because it reads the pre-flip classes | 112 |
| S3-24 | Stale-name checker `check_stale_names.py` (retired-name, guard-name and reserved-name modes). S5-01 vendors its reserved-name mode so the hub can never be named `aminx` | 87 |
| S4-03 | Pre-registered bathos spike: onnx-extra resolution arms and the orbax-checkpoint version each arm resolves | 60 |
| S4-15 | xtrax `hf_weights`: a revision-pin parameter, reported in `WeightReport` | 56 |
| S1-01 | Pre-registered characterization goldens on **current, unchanged main**. It must land before any S1 code item | 32 |
| S1-03 | Feasibility probes for design A on the installed equinox, producing a go/no-go record | 31 |
| S3-12 | Release guard and train rule in `release.yml` (tag equals the built version, PyPI 404 check, a dry run that never publishes). It orders S5-52 and S1-25 and any S2 release | 26 |
| S4-25 | Pre-registered bathos spike: differential `iree-compile` of one module under prolix's flags and under xtrax's WASM32 | 1 |
| S3-28 | Assembly-consistency check over the six spec `toml` blocks. Nothing depends on it; it turns section 8 into CI | 0 |

The four spikes (S4-02, S4-03, S4-25, S1-03) and S1-01 commit their `.bth.toml` sidecar with `[outcomes]` before
they run (`bth run`, not `uv run bth`).

### 5.2 Your decision or action needed first (10)

| Root | What you decide or do | Downstream |
|---|---|---|
| S4-01 | xtrax contract ADR: a separate `xtrax-contract` dist, names, Python floor, shipping discovery now, version and tag policy, toolchain (OQ-29) | 136 |
| S3-01 | **Hub identity:** slug, PyPI, domain, bathos slug, praxia collision policy, docs URL, release order (OQ-01 to OQ-03) | 86 |
| S2-02 | Confirm that maraxen/aminx#174 (xtrax 0.4.0a11 pin) is merged and record its SHA. The executor does not merge it | 77 |
| S6-01 | pipecanvas name, home and extraction trigger (OQ-45) | 62 |
| S4-37 | Weights licence and redistribution value for each MPNN weight set, with a named reviewer (OQ-34) | 33 |
| S2-01 | EBM project name, licence, visibility, install channel and clone location (OQ-16 to OQ-18) | 22 |
| S5-16 | isochore: move `mdtraj` to an extra. This changes your package's published dependency contract (OQ-41) | 11 |
| S4-49 | Whether S4-22 supersedes S3-18, and the fate of `make_mpnn_score` (OQ-05) | 4 |
| S1-17 | Owner and conformance mechanism for `PottsTRWRunSpec` (OQ-10) | 1 |
| S5-13 | Fast-forward py2Dmol `plugin-system-impl` into `main` and tag a release (OQ-42) | 1 |

**Answer S4-01, S3-01 and S2-02 first.** They unblock the most work: 136, 86 and 77 descendants respectively (the
sets overlap). S2-02 costs almost nothing, because it only records a merge you may already have made. S2-35 is the
next decision on the EBM path (the hp-lattice branch), and it becomes answerable as soon as S2-02 is recorded.

## 6. Open questions for the user (consolidated)

**What is counted.** The six specs currently ask 80 live questions:

| Spec | Live questions | Retired, numbers kept |
|---|---|---|
| S1 | 11 | Q10 |
| S2 | 14 | Q7 |
| S3 | 11 | Q1-Q8, Q10, Q14, Q15 |
| S4 | 13 | none |
| S5 | 11, of which Q1 is a pointer to S3-01 | none |
| S6 | 20 | none |

Section 7 lists the retired questions. Duplicates are merged here into 56 rows. Each row gives:
- its source questions;
- the recommendation as the specs state it;
- the items it blocks or parameterises.

Where a decision item records the answer, that item is listed first. No answer here overrides D1-D7.

### 6.1 Hub identity and the aminx name (S3, decided at S3-01)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-01 | **Hub identity.** Five independent layers, recorded in one ADR with availability probes and a re-open table. **Repo slug:** (a) `maraxen/aminx-hub`, (b) a praxia.science-derived slug such as `maraxen/praxia-science`, or (c) a new org. **PyPI:** whether the hub publishes and under what name. **Custom domain:** none, `hub.praxia.science` or the apex `praxia.science`. **Bathos slug.** **praxia-orchestrator collision policy.** | S3 Q16, Q17, Q18, Q19, Q20; S5 Q1; S6 Q20 | **Repo:** (a) `maraxen/aminx-hub`. Both (a) and (b) are free. (a) is what every S5/S6 item and sidecar already carries, and it shares no namespace with the orchestrator. **PyPI:** no project, because the hub ships no Python package. If one is ever needed, use `aminx-hub`. **Never `aminx`**, as a PyPI, import or npm name; S3-24's reserved-name mode enforces this in hub CI. **Domain:** the subdomain `hub.praxia.science`, wired after the first deployment (S3-29: verify the domain with GitHub first, add no CNAME file, create a DNS-only CNAME, enforce HTTPS, flip `base_path` to `/`, never leave a dangling record). The apex stays free for an umbrella or orchestrator site. **Bathos slug:** `aminx-hub`, kept even if the repo is renamed. **Collision:** keep `praxia` out of repo, PyPI, npm, CLI and bathos names, and allow it in DNS and prose. Site brand text is still to be decided | S3-01. Gates S5-01 (and through it every hub and editor item), S5-33, S3-21, S3-29 |
| OQ-02 | aminx's API docs URL: keep `aminx.readthedocs.io` canonical and leave the stale `maraxen.github.io/aminx/` Sphinx build, or retire or replace that build | S3 Q21 | Leave both as they are and revisit at S5-34. The Pages content is stale and has no demo behind it | S3-01, S3-21, S5-34 |
| OQ-03 | Release order: the browser-assets release S5-52 (next free alpha, `0.2.0a4` as read on 261001) before the spec-flip release S1-25 (`0.2.0a5`) | S3 Q22, S1 Q12 | Yes. The first hub deployment then does not wait for the S1 flip chain. Keep the coordination note that the S5-52 tag should not fall between the S1-07 merge and the S1-15 equivalence pass. Reversing the order flips one edge (S5-52 `depends_on` S1-25), and S5-33 then inherits the S1 chain | S3-01, S3-12, S5-52, S1-25 |
| OQ-04 | Include the previous rename's residue in this spec? `prxteinmpnn` appears in 75 tracked files, 32 of them outside dated history | S3 Q9 | Yes. It is the same defect class, and fixing it in one pass also revives or retires the dead `check_model_boundary.sh` guard | S3-24, S3-09 |
| OQ-05 | proteinsmc guard. Does S4-22 (manifest-driven discovery) supersede S3-18 (a root-only `find_spec("aminx")` guard with deferred imports), or follow it? What happens to the public `make_mpnn_score` signature and the `"mpnn"` registry key? How does the end-to-end test get `aminx` without a runtime edge (D6)? | S3 Q11, S3 Q13, S4 Q9 | **Supersede.** S3-18 is closed unmerged, and S4-22 becomes the only guard. Change the signature to `make_mpnn_score(structure, decoding_settings="random", *, scorer_id=None)` and keep the `"mpnn"` key (the old form has no working caller). S4-23 uses its own non-default dependency group, which alphex D4 does not count as a coupling edge. If you keep S3-18, S4-22 lands after it and replaces its guard. Either way, exactly one guard survives | S4-49. Gates S3-18, S4-22, S4-48, S4-23 |
| OQ-06 | asr carries a nested clone of proteinsmc at `vendor/proteinsmc` (its own `.git`, not in `.gitmodules`), and that clone's `mpnn.py` differs from proteinsmc main. Keep it, or depend on proteinsmc normally? | S3 Q12 | Out of scope for this set. File it as a D6-adjacent debt item for the ecosystem-partition work | none (file as debt) |

### 6.2 Spec unification (S1)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-07 | Which representation survives: the nested `RunSpec` (A) or the facade (B)? | S1 Q1 | **A.** The nested `RunSpec` becomes the model, and the flat vocabulary becomes a permanent codec and constructor signature. Confirm at S1-04, once the S1-02 and S1-03 records exist. Fall back to B if the S1-03 probes refute A | S1-04 |
| OQ-08 | When do flat attribute reads and `.run_spec` stop working? | S1 Q2 | The shims survive at least one published alpha (S1-25). Consumers migrate first (S1-26 to S1-29). Only then does S1-20 remove the shims. The alternative is permanent read shims | S1-25, S1-20 |
| OQ-09 | Fix the `or`-coalescing quirks? Today `random_seed=0` becomes 42, `multi_state_temperature=0.0` becomes 1.0 and `num_samples=0` becomes 1 | S1 Q3 | Keep them in S1 (raw storage plus `resolved_*` accessors). Fix them separately in S1-22 with a changelog entry, because the fix changes sampled outputs for seed 0 | S1-22 |
| OQ-10 | Owner of the Potts TRW spec, and how conformance is checked | S1 Q4 | `mistypotts` owns it. aminx vendors a copy pinned by sha256 and tests it without importing it, with a mutate-a-copy negative control. The dev-dependency variant is used only if `uv lock --check` passes | S1-17, S1-30 |
| OQ-11 | Fold Potts into `RunSpec`? | S1 Q5 | Not now. Share a `SpecFamily` protocol only, and revisit with S4 | none |
| OQ-12 | Who owns the browser knob document, and how do the browser's `chains_to_design` and Python's `chain_id` map onto each other? | S1 Q6, S4 Q2 | S4 owns it: S4-40 builds it, S4-41 gives it golden vectors and S4-20 generates from it. S1 builds no `DesignConstraints`. Keep the browser vocabulary at the portable layer and map it in the lowering; never rename the browser key | S4-33, S4-40, S4-20 |
| OQ-13 | How to prune knobs that are accepted but ignored | S1 Q7 | Delete scaffolding fields outright. Provably inert public kwargs go through warn-and-ignore for one window. S1-16 asks item by item | S1-16 |
| OQ-14 | Fate of portable v2, which has no consumer | S1 Q8 | Freeze it and mark it deprecated until S4-20 ships the schema-generated format, then delete it | S1-08, S4-20 |
| OQ-15 | `RunSpec.run_id` is never populated | S1 Q9 | A follow-on outside this DAG: derive it from the canonical payload plus the code SHA. S1 keeps the field wired | none (file as follow-on) |
| OQ-16 | Consumer migration scope, and whether hautespout is bumped or left pinned | S1 Q11 | S1 owns migration branches for mpnn_ext, asr and hautespout. Each is a branch off the consumer's own main that bumps the aminx submodule SHA; nothing is renamed and nothing is shared with S3. tev_design gets only a handoff (S1-29). S1-28 itself decides bump or pin; if pinned, S1-28 is recorded moot | S1-28, S1-26, S1-27, S1-29, S1-21, S1-20 |

### 6.3 EBM extraction (S2)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-17 | EBM project name | S2 Q1 | `ebmx`, always naming the checkpoint by its filename. The other options were `proteinebm-jax` and `proteinxebm`. PyPI and GitHub availability have not been checked yet | S2-01 (and the 20 `ebmx` items) |
| OQ-18 | EBM licence and attribution. The local `jproney/ProteinEBM` checkout has no LICENSE, and the trunk is derived from Boltz-1 | S2 Q2 | Do not publish publicly until upstream's GitHub licence has been read. If it is permissive, license the new code MIT and add a NOTICE. If there is none, ask upstream first | S2-01, S2-16, S2-17 |
| OQ-19 | EBM remote, visibility and install channel | S2 Q3 | A private `maraxen/ebmx`, installed by git URL, until OQ-18 is resolved. Until S2-17 creates the remote, work happens on a local branch synced by rsync | S2-01, S2-17, S2-18 |
| OQ-20 | How to tell `aminx.ebm` users after the cutover (there is no shim), and what happens to the branch `hp-lattice-sanity-check` and its hautespout WIP consumer | S2 Q4 | Use a CHANGELOG pointer only. `import aminx.ebm` then raises `ModuleNotFoundError`. If you know of outside users, add a five-line stub that raises `ImportError` naming `ebmx`. The removal rides the next release cut. The branch's fate is S2-35's: merge it before the freeze, abandon it, or port it into `ebmx` (S2-36) | S2-35, S2-18, S2-36 |
| OQ-21 | alphex as a runtime dependency of `ebmx` | S2 Q5 | A dev-only conformance test now. Take the runtime edge after the partition work | S2-27 |
| OQ-22 | SwiGLU | S2 Q6 | Copy it verbatim now. File a debt item for a bias-free `Transition`; that change alters the orbax tree and needs its own parity run | S2-26 |
| OQ-23 | Approve the GPU budget and node class NC | S2 Q8 | Approve about 7.7 GPU-hours for the two full arms, plus about 1 hour for the floor, a small tier-1 cost and a one-time compile. Use `pi_so3` with an explicit `--gres` type and no `mit_preemptable` fallback, chunked and preemption-safe. An optional upgrade costs about +7.7 h: a second whole-population A0 replicate | S2-30. Gates S2-05, S2-32, S2-33, S2-34, S2-14 |
| OQ-24 | Disposition of the tracked and untracked evidence | S2 Q9, S2 Q13 | Keep `outputs/ebm_benchmarks*` tracked as it is, and import the untracked extras under `untracked_import/`. Delete the root directories only after the pushed copy re-hashes clean. Keep the tarball outside every repo | S2-07, S2-20 |
| OQ-25 | Publish the converted orbax weights? | S2 Q10 | Not until OQ-18 is settled. Ship the converter and `weights.lock.toml` instead | S2-12, S2-17 |
| OQ-26 | Should A0/B1 close the old claim campaign `aba3bfe6`? | S2 Q11 | Yes, but only through S2-29. The old claim stays open and byte-identical until you approve | S2-29 |
| OQ-27 | How to handle the A0 outcome | S2 Q12 | On `fail`, investigate dependency drift against the July baseline. On `marginal`, proceed with A0 as the reference. The move gate is A0 vs B1, not A0 vs the paper | S2-25 |
| OQ-28 | What to do if `converter_matches_legacy = false` | S2 Q14 | Fix the converter first, and re-run tier 1, if the differing leaves are few and explainable. Otherwise ship the legacy-tree path only, with the limitation documented | S2-31, S2-17, S2-18 |
| OQ-29 | Near-threshold parity misses: above tolerance but within 10x | S2 Q15 | Accept only if the delta traces to the `chunked_map`/`scan` op order and the A0-vs-B1 mean difference stays within TOL_AGG | S2-13, S2-14 (decided at review) |

### 6.4 xtrax contract (S4)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-30 | Contract packaging and names, and whether to ship discovery now | S4 Q1, S4 Q4 | **F2:** a separate `xtrax-contract` dist with Python >= 3.11 and no dependencies, plus an `xtrax.contract` shim. Accept the group `xtrax.models` and the id root `aminx`, which is the providing package's own name and never the hub's. Ship discovery with manifest-as-data entry values. Accept the version, tag and toolchain policy | S4-01 (136 descendants) |
| OQ-31 | The orbax conflict on the onnx extra | S4 Q3 | Option A: a consumer-side override with the toolchain in a dependency group, and `xtrax[export]` moved out of aminx's base dependencies. Fall back to D if S4-03 shows that A breaks the ONNX scripts | S4-16 |
| OQ-32 | Publish the generated contract TS package to npm? Publish pipecanvas to npm? How does praxis consume pipecanvas? | S4 Q5, S6 Q16 | Not yet, for either package. Keep the contract artifacts in-repo or vendored. Praxis consumes pipecanvas as a release-tarball URL plus sha256 | S4-28, S6-32 (nothing depends on either) |
| OQ-33 | localfold integration mode, and AF3 in the hub | S4 Q6, S5 Q3, S5 Q4 | Now: link-out with job JSON, where the user drags the file in (S5-23). In parallel: an upstream entry-point request (S5-25). The library adapter comes only after that, and only with the author's vendoring consent (S5-26). List AF3 as link-out/`byo` only, with its terms shown, and never host or proxy its params. D5 still holds; the limit is licensing, not capability | S5-23, S5-25, S5-26, S5-45 |
| OQ-34 | prolix WASM flags | S4 Q7 | Let the S4-25 differential result decide. Keep the 50 MiB gate either way | S4-26 |
| OQ-35 | Weights licensing review, and weights on Pages versus elsewhere | S4 Q8, S5 Q8 | One decision with one reviewer and one record in the aminx tree (S4-37), cited by S5-07. By default the hub carries only code and the catalog. Small ONNX files that you have signed off go to Pages through staging. Larger or third-party weights stay on HF or `origin`. If you decline, the HF fallback is S5-56 | S4-37, S5-07, S5-56. Gates S4-19, S5-33 |
| OQ-36 | `metadata.nl_description` is required on every node | S4 Q10 | Keep the invariant. The S6 editor prefills it from the manifest `description`. Not blocking | S4-10, S6-10 |
| OQ-37 | Release channel for `xtrax-contract` and `xtrax` | S4 Q11 | PyPI for both. Publishing is your action | S4-34 |
| OQ-38 | Where the MPNN ONNX graphs and golden vectors are hosted | S4 Q12 | GitHub release assets of the aminx repo. S4-19 ships `release-asset:` placeholders, and S5-52 rewrites them | S4-19, S5-52 |
| OQ-39 | Accept S5's amendments G1-G4 into `v1` before the first release? They cover a weights-less tool, external/`byo` artifacts, pyodide file-system mappings, and determinism on every executor kind | S4 Q13, S5 Q9 | Accept. They are additive and nothing has been published yet. If you decline, the hub works from the unamended `v1`, and any later amendment becomes a `schema_version` 2 bump | S5-51. Gates S4-34, S5-64, S5-42 |

### 6.5 Hub (S5)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-40 | Two browser MPNN implementations: aminx's ONNX path and localfold's JS path | S5 Q2 | **Option D:** aminx's ONNX path is the validated default for ProteinMPNN and SolubleMPNN, and localfold's is offered for the Ligand and NA families and as an opt-in cross-check. Decide after the pre-registered study S5-27. If the two checkpoints are not the same weights, the study is descriptive only | S5-46 |
| OQ-41 | isochore as a pure-python wheel | S5 Q6 | Yes. Move `mdtraj` to an extra, and `h5py` too if it is unused at import | S5-16, S5-54, S5-42 |
| OQ-42 | Merge the py2Dmol plugin system into `main` | S5 Q7 | Yes, after py2Dmol's own suite passes. Only S5-36 waits for it. It rebuilds the tracked bundles (+3.3% on `embed.min.js`) | S5-13, S5-36 |
| OQ-43 | Driver packaging | S5 Q10 | One pre-bundled ES module built in aminx with a dev-dependency bundler, guarded by the S5-50 AST scan. The hub stays bundler-free | S5-14 |
| OQ-44 | Hub build toolchain, licence, vendored notices and creating the repo | S5 Q5, S5 Q11 | Node/ESM, with Python only as the bathos shim. Hub code under MIT. `vendor.lock.json` requires `license` and `notice` for each entry, and NOTICE is generated and checked (S5-63). The MPL-2.0 legal review is yours. So are creating the repo under the S3-01 slug and setting Pages to Actions | S5-01, S5-63 |

### 6.6 Pipeline editor (S6)

| # | Question | Sources | Recommendation | Blocks |
|---|---|---|---|---|
| OQ-45 | pipecanvas name and home | S6 Q1 | Keep `pipecanvas` as a placeholder. Start inside the hub repo behind an enforced boundary. Extract it only on a recorded praxis "go": you create the repo (S6-34), the hub-side split follows (S6-20), and the hub then consumes the release (S6-30) | S6-01, S6-34, S6-20, S6-30 |
| OQ-46 | Is the DAG canvas a goal in itself, given that the list is already DAG-expressive? | S6 Q2 | Yes, build it. Ship the list first and keep it permanently as the accessible and mobile view | S6-12 and the canvas chain |
| OQ-47 | How JupyterLite is used, and whether to add a widget inside the notebook | S6 Q3, S6 Q13 | Execute in S5's Pyodide worker and export to a notebook (S6-18). The widget (S6-23) is optional and not needed now | S6-18, S5-35, S6-23 |
| OQ-48 | Code-node trust model, and whether to accept the S5-43 sandbox | S6 Q4, S6 Q11 | Documents from elsewhere are untrusted by default, with trust granted per hash. Accept S5-43. If the egress canary cannot be made to fail, code nodes run only for documents authored in the current session | S6-17, S6-28, S5-43 |
| OQ-49 | Scope of the type system | S6 Q5 | Implement exactly S4's `compat` rule. Defer an "insert adapter" command | S6-04 |
| OQ-50 | Praxis scope | S6 Q6 | A read-only view first, with no authoring | S6-21, S6-22 |
| OQ-51 | Praxis go/no-go criteria: 60 days from the live-origin deploy, 3 maintainer pipelines, no P1, and a written first use case | S6 Q7 | These are proposals. Edit them and commit them before S6-19. A custom-domain change after the deploy date does not restart the clock | S6-19 |
| OQ-52 | Canvas engine | S6 Q8 | Rete.js v2 provisionally. The real decision is S6-09, after the bake-off S6-36 | S6-09 |
| OQ-53 | Which single host runs the bake-off and the performance budget | S6 Q15 | titanix, if its pinned-Chromium Playwright lane is verified. Otherwise the local workstation. Record the host and never mix hosts | S6-36, S6-27 |
| OQ-54 | Change requests to S5: what the dependency guard scans, a workspace-sourced vendor entry, and catalog palette fields plus `manifest_sha256 = sha256(JCS(manifest))` | S6 Q17, S6 Q18 (S5 section 6, "Requests from S6") | Yes to all three, as visible changes. S6-33 is a no-op if the fields are already present | S6-02, S6-33, S5-01, S5-04 |
| OQ-55 | `executor_hint` is inside the hashed graph, so changing a hint revokes per-hash code trust | S6 Q19 | Accept. This is conservative and keeps one IR. Spec keys ignore hints, so cache prediction is unaffected | S6-17, S6-35 |
| OQ-56 | Confirm-only items. S6 owns the `ExecutionClient` port, and S5 owns the adapter. S5-31 binds to the list. S4 is the sole owner of the corpus and JCS vectors. The S4 port-field request is withdrawn | S6 Q9, S6 Q10, S6 Q12, S6 Q14 | Confirm as written | S6-07, S5-30, S5-31, S4-30, S4-31 |

## 7. What the no-rename decision removed

The previous plan's authoritative list is kept at `.praxia/spikes/261001_dag_given_edges.txt`. The checker compares
it with the current list.

| Spec | Items before | Retired | Added | Items now | Edges to retired ids removed |
|---|---|---|---|---|---|
| S1 | 31 | 0 | 0 | 31 | 13 |
| S2 | 32 | 0 | 2 (S2-35, S2-36; hp-lattice branch, not from the rename decision) | 34 | 2 |
| S3 | 27 | **20** | 1 (S3-29) | 8 | 6 (between S3 items) |
| S4 | 48 | 0 | 0 | 48 | 5 |
| S5 | 60 | **1** (S5-55) | 0 | 59 | 5 |
| S6 | 33 | 0 | 0 | 33 | 0 |
| **Total** | **231** | **21** | **3** | **213** | **31** (553 → 478 edges overall) |

**Retired S3 items (20).** S3 section 10 gives the reason and replacement for each. S3-26 was never allocated, and
retired ids are never reused.

| Item | What it was |
|---|---|
| S3-02 | Repo rename and redirect probe |
| S3-03 | Trusted publishers for the shim and molxmpnn |
| S3-04 | Old-name contract test |
| S3-05 | Pre-rename goldens |
| S3-06 | Rename codemod |
| S3-07 | Atomic rename PR |
| S3-08 | `MOLXMPNN_*` compat layer |
| S3-10 | `aminx` shim distribution |
| S3-11 | HF weights cutover |
| S3-13 | Publishing `molxmpnn` and the shim |
| S3-14 | mpnn_ext migration |
| S3-15 | asr migration |
| S3-16 | tev_design note |
| S3-17 | hautespout decision |
| S3-19 | `using-molxmpnn` skill |
| S3-20 | Checkout-rename runbook |
| S3-22 | Closing the deprecation window and deciding P1/P2 |
| S3-23 | PyPI handover |
| S3-25 | Shim `aminx.ebm` forwarding |
| S3-27 | Reserving `molxmpnn` |

**Retired S5 item (1).** S5-55, "P2 only: hub takes the aminx name". It contradicts revised D2. No S5 item publishes
to PyPI or npm any more.

**S3 items kept and re-scoped (7), plus one new.**
- **S3-01** is now the hub identity decision.
- **S3-24** is the stale-name checker. It gained a reserved-name mode so the hub can never be named `aminx`.
- **S3-09** fixes the `prxteinmpnn` residue, re-measured at 32 non-history files.
- **S3-12** is now the release guard and train rule. It was re-meant because S5-52, S1-25 and any S2 release still
  share one workflow.
- **S3-18** guards `aminx` in proteinsmc with deferred imports.
- **S3-21** covers aminx's public surfaces and the docs-URL record.
- **S3-28** is the assembly check, rewritten for the retired set.
- **New: S3-29**, the custom domain wiring.

**Edges.**
- Edges that only ordered work against the rename were dropped, with no replacement:
  - S1-01, S1-02, S1-03, S1-05 and S1-17;
  - S1-26, S1-27 and S1-28 (their edges to S3-10, S3-14, S3-15 and S3-17);
  - S2-03 (its edge to S3-13) and S2-18 (its edge to S3-07);
  - S4-16, S4-19 and S4-35;
  - S5-14, S5-21, S5-34, S5-52 and S5-61.
- Edges that still carry a real need now point at kept items:
  - S1-25 and S5-52 point at S3-12;
  - S5-01 and S5-33 point at S3-01;
  - S5-34 points at S3-21;
  - S4-49 sits upstream of S3-18;
  - S5-01 also gains S3-24.
- **64 surviving items changed their `repo` from `molxmpnn` back to `aminx`.**

**Questions retired (13).** The ordering question shared by S1 Q10, S2 Q7 and S3 Q4 (rename first or EBM first) no
longer exists. S3 also retired these:

| S3 question | What it asked |
|---|---|
| Q1 | The name `molxmpnn` |
| Q2 | Repo strategy |
| Q3 | PyPI handover |
| Q5 | Bathos slug rename (superseded by Q19) |
| Q6 | Shim warning class |
| Q7 | Checkout rename |
| Q8 | HF weights option |
| Q10 | Renaming `browser/aminx-sampler` |
| Q14 | Soak thresholds |
| Q15 | Legacy Pages URL (superseded by Q21) |

The previous plan's questions OQ-01 to OQ-11 have either disappeared or been folded into the new OQ-01 to OQ-03.

**Critical path.** The previous plan's longest chain had 28 items: the rename (S3-01 to S3-13), then the whole EBM
extraction, then the checkout-rename window. The longest chain now has 23 items and runs through the contract and
the hub. The EBM extraction is a separate chain of 21 items that waits on no S3 item.

## 8. DAG check (by code)

The checker is `.praxia/spikes/261001_dag_check_v2.py`, stdlib only. It is a throwaway spike that produces this
table and sections 2 and 3, not a research finding:

```
uv run --no-project python3 .praxia/spikes/261001_dag_check_v2.py --emit-dir .praxia/spikes/261001_dag_out_v2
uv run --no-project python3 .praxia/spikes/261001_dag_check_v2.py --controls
```

It reads the authoritative item list, transcribed verbatim to `.praxia/spikes/261001_dag_given_items.json`. It also
reads, separately, the single `toml` block of `[[item]]` tables in each of the six spec files. Result on 2026-10-01:

| Check | Result |
|---|---|
| Items / edges | 213 / 478 (S1 31, S2 34, S3 8, S4 48, S5 59, S6 33) |
| Exactly one `[[item]]` toml block per spec, parseable | yes, 6/6 |
| Authoritative list vs spec `toml` (id, title, repo, size, user_decision, depends_on) | **0 differences** (213 = 213) |
| Duplicate ids | **none** |
| Dangling `depends_on` (an id that is not an item) | **none** |
| Self-dependencies | none |
| Acyclic (Kahn) | **yes**: all 213 items ordered, no node left in a cycle |
| Retired names (`molxmpnn`, `mpnnx`) in any item field | none. S3-28 is allow-listed, because its title names the string it forbids |
| Roots / sinks | 19 / 34 |
| Negative controls, each planted in a copy | all six detected: a cycle (S3-01 made to depend on S5-33), a dangling id, a duplicate id, a self-dependency, a `molxmpnn` string in a title, and list-vs-toml size drift |

S3-28 is still the item that turns these checks into CI and adds the remaining assembly rows from S3 2.2. Those
rows hold on the current graph:
- every release item sits behind S3-12 (S1-25 and S5-52 depend on it);
- S3-01 comes before S5-01 and S5-33;
- S3-29 comes after S5-33;
- no item depends on a retired S3 id (the dangling check would catch one).

## Revision log

### no-rename revision (2026-10-01, user decision revising D2)

- The file was regenerated from scratch for the no-rename item set: 213 items and 478 edges, down from 231 and 553.
  Sections 2, 3 and 8 come from the new checker, `.praxia/spikes/261001_dag_check_v2.py`, run over
  `261001_dag_given_items.json`. The old checker and its outputs (`261001_dag_check.py`, `261001_dag_out/`) are
  left as history.
- The S3 link now points at `261001_aminx-identity-and-hub-naming.md`. The `molxmpnn` repo labels are gone: every
  MPNN-tree item says `aminx`.
- The R/E ordering premise was removed. The critical path was recomputed, and S4-01 now heads it instead of the
  rename.
- The open questions were rebuilt from the six specs' current question lists, with hub identity first. The new
  section 7 records what the decision removed.
