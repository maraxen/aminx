---
title: "S3: aminx -> molxmpnn rename, freeing the aminx name for the hub"
description: "Rename the aminx MPNN package to molxmpnn without breaking consumers (shim, PyPI, repo, weights, provenance), and release the aminx name for the hub."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - aminx (-> molxmpnn)
  - mpnn_ext
  - asr
  - tev_design
  - hautespout
  - proteinsmc
related_specs:
  - S1
  - S2
  - S4
  - S5
---

# S3: aminx -> molxmpnn rename, freeing the aminx name for the hub

Spec only. Nothing here executes yet (user decision D7). Decisions D1-D7 in
`.praxia/docs/research/261001_ecosystem-hub-recon-brief.md` are not overridden; D2 (rename, keep
the name `aminx` for the hub) is implemented here, with one honest limit on what "free the name"
can mean for the GitHub repo slug (section 4.1).

## 1. Goal and non-goals

**Goal.** After this spec's items are done:

1. The MPNN package is published and imported as `molxmpnn` (the user's proposed name: "might be the
   right name"; confirmation is Q1, answered in S3-01), with the same behaviour as `aminx 0.2.0a3` and
   bit-identical outputs on a frozen fixture set.
2. Every existing consumer (`mpnn_ext`, `asr`, `tev_design`, `hautespout`, anything doing
   `import aminx`) keeps working during a deprecation window through a transitional `aminx`
   release that re-exports `molxmpnn` and warns.
3. The `aminx` name is free for the hub (S5) at the layers that can be freed without destroying old
   URLs: **brand, site, CLI (after the window), skill name**. Two layers are *not* freed under the
   recommended policy, so D2 is only partly satisfied: the **GitHub slug `maraxen/aminx`** stays a
   redirect to molxmpnn forever (Q2, option 1), and the **PyPI name `aminx`** stays the MPNN shim
   forever (Q3, P1). Handing either to the hub is a separate, user-gated, irreversible decision
   (S3-22/S3-23), not an outcome of this spec.
4. Nothing keyed by the old name silently stops working: import-boundary bans, `find_spec` guards,
   package-data keys, persisted provenance, weight pins, bathos history.

**Non-goals.**

- No behaviour change, API redesign or spec-system change (S1), and no EBM extraction (S2).
- No new hub functionality (S5). S3 only releases the name and states the contract.
- No rewrite of git history, and no rewrite of dated docs, CHANGELOG entries, bathos run
  records, or `outputs/` artifacts (append-only logs).
- No layered cache-dir resolver for the whole library. The one cache dir this spec touches
  (`aminx` inputs cache, `cli.py:108-130`) gets the documented resolution order (S3-08); the rest
  is existing behaviour and is out of scope.
- No GitHub Pages / custom-domain design for the hub (S5).

## 2. Spec set coordination

S3 is the DAG root against S1: S1 code items carry `depends_on` edges *to* S3 items (S1-01/02/03/17
-> S3-07, S1-25 -> S3-13, shim consumers -> S3-10), and S3 has no edge to S1. S2 is the contested
edge, settled here (round 1, C3):

**Recommended ordering (Q4, option R): rename first.** S3-07 lands before S1's refactor and before
S2's EBM cutover. S2 then runs on the renamed tree. Consequences S2 must adopt (S2 is revised
separately; these are the contract S3 provides, not edits S3 makes to S2):

| Topic | Rename first (R, recommended; S1 already assumes it) | S2 first (alternative, option E) |
|---|---|---|
| Edges | S2-18 `depends_on` S3-07; **no edge from any S3 item to S2 except S3-25** (below) | S3-07 `depends_on` S2-18 and S2-19, and S1's four S3-07 edges flip per S1's own flip path |
| Who publishes `aminx 0.2.0a4` | **S3-13 only** (molxmpnn 0.2.0a4 plus namespace shim aminx 0.2.0a4). **S2-19 is deleted** (C2); its dependents are re-pointed to S2-18 and the `aminx.ebm` row is S3-25's | S2-19 publishes `aminx 0.2.0a4` (old MPNN dist, `aminx.ebm` forwards to `ebmx`); S3 then publishes **0.2.0a5** for both dists |
| `aminx.ebm` in the shim | Initial row: refuse with "moved, see CHANGELOG". S3-25 (after S2-18) flips it to *forward to `ebmx` if installed, else the install message* and releases the shim as a new shim-only version | Row is "refuse with pointer to `ebmx`"; the 0.2.0a4 forwarding shim from S2 is already out and is not carried into molxmpnn |
| Owner of `gone_modules.toml` | **S3 (S3-10, S3-25).** S2 supplies only the project name and release (S2-01 ADR, ebmx release) | S3 (S3-10) |
| PyPI uniqueness | One upload per (project, version) is guaranteed because only S3 publishes `aminx` | Guaranteed by version bump to 0.2.0a5 |
| EBM `filter-repo` and the A0 freeze | Runs on `src/molxmpnn/ebm` (plus `src/aminx/ebm` as history paths) and **S2-03 (freeze SHA) follows S3-13**, mandatory, not optional (C3, assembly row 8) | Runs on `src/aminx/ebm` only |

If the user chooses option E, replace every `0.2.0a4` below with `0.2.0a5`, add `S2-19` and `S2-18`
to S3-07's `depends_on`, and drop S3-25 (its work is already in S2-19). The DAG is acyclic under both.

**Bathos slug (settled, contradicts S2 line 564).** `runs` has no project-id column [A8], so a slug
"rename that keeps the project id" does not exist. Policy is Q5: new slug `molxmpnn`, legacy `aminx`
frozen, union query. S2 must reword its mitigation; `ebmx` takes its own slug via `bth init`.

**Assembly checklist (round 2, R2-4/R2-5; verified mechanically by S3-28).** S3 states contracts that
the other specs' text does not yet satisfy (S2 in particular, as read in round 2: S2-18 lists
`depends_on = [S2-14, S2-15, S2-17, S2-31]` with no S3 edge; S2-19 "Release aminx 0.2.0a4 containing
the aminx.ebm forwarding shim" depends on S2-18; S2's text recommends E). Revising S2 is outside this
spec. The ordering question behind rows 1-3 and 8 is one user question asked three times (S1 Q10 = S3 Q4 = S2 Q7) and is answered **once, at S3-01**; if S2 still recommends E when S3-01 closes, the answer there wins and S2 is revised, not S3. The required edits are listed here and checked by a script instead of living in prose:

| # | Required edit in another spec | Why | S3-28 check |
|---|---|---|---|
| 1 | S2-18 `depends_on` gains `S3-07` (option R) | S2 edits the post-rename tree | S2-18 deps contain S3-07 |
| 2 | **S2-19 is deleted** (round 1, C2): its dependents re-point to S2-18, and S2-19 is removed from the leaf lists in S2 and from every `depends_on`. S3-25 is the single owner of the `aminx.ebm` row and S3-13 the sole publisher of `aminx 0.2.0a4` | PyPI rejects a second upload of `aminx 0.2.0a4`; under rename-first S2's sys.meta_path shim has no tree to live in, and its justification ("after the rename `import aminx` resolves to the hub") is false under P1 and S5 (the hub ships no Python package) | no item id `S2-19` exists or appears in any `depends_on`; no item outside S3 whose title matches `release.*\baminx\b` |
| 3 | S2 deletes its in-tree meta-path shim from S2-18's scope | S3 owns the only shim (`gone_modules.toml`) | S2-18 title has no "shim" |
| 4 | S2-04/S2-05/S2-25 name the old-arm package `molxmpnn` | tree is renamed first | manual line in the S3-28 report (not machine-checkable) |
| 5 | S2 mitigation wording on the bathos slug follows Q5 | no project-id column [A8] | manual |
| 6 | Every S1/S2/S4 item whose `repo` is `molxmpnn`/`aminx` and that is a **leaf** is in S3-20's `depends_on`. **Leaf** means: no *direct* `depends_on` edge from another S1/S2/S4 item whose repo is `molxmpnn`/`aminx` (edges through items of other repos, e.g. `ebmx`, are ignored). S3-20 is the last item among those | quiescent window must be a computed property | S3-28 **computes** the leaf set from the assembled backlog and compares it to S3-20's non-S3 `depends_on`; the ids are not repeated in prose |
| 7 | **Withdrawn (round 1, C4).** S5 items on repo `molxmpnn` are intentionally **unordered** against S3-20: forcing them behind S3-20 would put S5-52, S5-53, S5-15, S5-29 and S5-33 behind S1-16/17/20/22, S2-20 (after the multi-hour B1 parity) and S4-20/21/42, so the first public hub deployment would wait for shim removal and EBM close-out. Protection is the fail-closed first runbook step (`git worktree list` shows only the main checkout); S3-20 blocks until S5 worktrees close | quiescence without coupling the hub rollout to S3 close-out | none (no check) |
| 8 | **S2-03 `depends_on` contains S3-13** (round 1, C3) | S2 section 4.7 requires `git diff F' <sha> --` to be empty over `src/aminx/ebm`, `utils`, `io`, `parity` and `scripts/ebm` at S2-32..34 and S2-06; S3-07 renames every one of those paths, S3-08/S3-11 edit `utils`/`io`, S3-09 sweeps 64 files. Freezing after S3-13 means the A0 evidence is captured on the code the extraction moves. S1 items that edit the same paths (e.g. S1-11) must be ordered against S2-03..S2-06 by S1/S2, not by S3 | S2-03 deps contain S3-13 |
| 9 | S2-02 (xtrax 0.4.0a11 pin confirmation, draft PR 174) precedes the golden capture: **S3-05 `depends_on` S2-02**; S4-16 must not claim to land #174 "rebased on the rename" (merge-before-rename and rebase-on-the-rename cannot both hold); it keeps only the orbax resolution and moving `xtrax[export]` out of base dependencies (C7) | S3-05's environment stamp includes the xtrax version and S3-07 refuses to compare on any mismatch (A41); both must run on a11 | S3-05 deps contain S2-02; the S4-16 wording is a manual line in the report |
| 10 | S1-25 and S5-52 (both cut molxmpnn releases, 4.9) are **ordered**: one is an ancestor of the other. **Order: S5-52 first (`0.2.0a5`), then S1-25 (`0.2.0a6`)**, i.e. S1-25 `depends_on` gains S5-52 (an edge S1 adds; S3 does not edit S1). The later alpha carries both sets of work. Reason: the first hub deployment (S5-33 -> S5-53 -> S5-52) must not wait for the whole S1 flip chain (S1-07 size L, S1-10..S1-15), the same coupling row 7 withdrew for S3-20; S1-25 is already a tail item (it feeds only S1-29, S1-20 and S3-20), so ordering it behind S5-52 costs nothing. S1-25's own gate that claims `0.2.0a5` and says it 'does not contain S5-52's work' must be restated by S1 as 'next free alpha' (4.9 rule), and S5's 'deliberately not ordered against S1-25' wording gives way to this edge. If the user prefers S1-25 first, the edge goes the other way (S5-52 `depends_on` S1-25) and S5-33 inherits the S1 chain; that cost is recorded in the S3-01 re-open table | two unordered release items race for the next free version and PyPI forbids re-upload (C14); the three specs disagreed on intent, which this row resolves | the two are comparable in the union DAG (default direction S5-52 an ancestor of S1-25; either direction passes) |
| 11 | The 0.2.0a4 pin is a content rule (4.9), so items that change `src/molxmpnn/` or the dependency set must not be mergeable before the tag: **S1-05** (new `run/fields.py`), **S1-17** (potts file header, a `src/` edit) and **S4-16** (`pyproject.toml`, `uv.lock`, base-dependency change) each have **S3-13 as an ancestor** in the union DAG. Additive files under `tests/`, `scripts/`, `docs/`, `.praxia/` from items with no S3-13 ancestor (S1-01..03, S4-02/03/33/37/43 as read on 261001) are allowed in the tag's history | git ancestry of the tag includes everything merged to main before the S3-07..S3-12 merge; S3-12 waits on S3-03 <- S3-02 <- S3-27 (user actions), so the window is realistic and the diff gate would otherwise fail or force cherry-picks | S1-05, S1-17 and S4-16 each have S3-13 as an ancestor in the union DAG |

S3-28 is a read-only stdlib script (no heavy compute, runnable locally) that parses every
`261001_*.md` backlog block, asserts rows 1, 2, 3, 6, 8, 9 (S3-05 edge), 10 and 11, the acyclicity of the union DAG, and prints
rows 4 and 5 and the S4-16 wording of row 9 for a human (row 7 is withdrawn). S3-13 and S3-20 depend on it; the script run is the evidence, so a stale
S2 fails S3's DAG instead of surviving as a footnote.

## 3. Current state (anchored; tags are in the Assumption Ledger below)

Paths are relative to the aminx worktree root unless prefixed `../../../../` (that is `~/projects/`).

### 3.1 Published identity that the rename touches

- PyPI project `aminx`: ten releases, `0.1.0a1` .. `0.2.0a3`, **all pre-releases** [A3, spike S1].
  PyPI project `molxmpnn`: **404, unregistered** at 2026-10-01 [A2, spike S1].
- `pyproject.toml:6-7` name `aminx` version `0.2.0a3`; `:34` console script `aminx = "aminx.cli:main"`;
  `:37-38` repository `https://github.com/maraxen/Aminx.git` (capital A, already relying on
  case-insensitive resolution) and documentation `https://aminx.readthedocs.io` [A21].
- Build config keyed by the package name: `package-data` (`:285`), `exclude-package-data`
  (`:292`, the guard that stopped the 88.8 MB wheel incident, see comment at `:288-291`), ruff
  `banned-api` (`:153`, four `aminx.*` keys protecting the Potts boundary), `ty` root (`:245`) [A21].
- Release is PyPI trusted publishing: `.github/workflows/release.yml:30-41` (`environment: pypi`,
  `id-token: write`, no token) [A11]. Pages is manual dispatch only: `.github/workflows/pages.yml:7-8`
  [A24].
- Weights: `src/aminx/io/weights.py:23` `HF_REPO_ID = "maraxen/aminx"`, `:34` `HF_REVISION = aa80d0fd...`;
  released wheels pin this repo and SHA; env overrides `AMINX_WEIGHTS_DIR` / `AMINX_WEIGHTS_REVISION`
  (`:39-40`) [A17]. 15 weight files are Git LFS under `src/aminx/model_params/` and 10 ONNX under
  `release/browser/` (LFS patterns are by extension, `.gitattributes:1-6`) [A22].

### 3.2 Things that carry the package name at runtime or on disk

- Runtime strings (not just imports): `files("aminx.model_params")` (`io/weights.py:216`),
  `importlib.metadata.version("aminx")` (`io/sink_provenance.py:201`, used by `resolve_aminx_version`),
  `importlib.import_module("aminx.sampling.sample")` (`host/sampling_driver.py:36`), typer app name
  and XDG cache dir (`cli.py:37,127-129`) [A14]. Plain-text rewrite does not find these as imports.
- Persisted artifacts: every design/sampling store stamps the root attr `aminx_version`
  (`io/designs.py:221`, `host/runner.py:1643`) [A15]. Readers outside this repo (tev_design, asr)
  key off it.
- `potts/calibration.py:159` `pickle.loads(...)` of calibration checkpoints, which embed class
  module paths, so a renamed package orphans every existing checkpoint of that kind [A16, spike S3].
- Env vars: `AMINX_*` names across src/scripts/tests/tools/.github/browser/site (`AMINX_CACHE_DIR` 11
  uses, `AMINX_VERIFY` 8, `AMINX_MODEL_BASE_URL` 6, ...). **Library code under `src/` reads exactly
  four: `AMINX_CACHE_DIR`, `AMINX_VERIFY`, `AMINX_WEIGHTS_DIR`, `AMINX_WEIGHTS_REVISION`** [A29].
  `AMINX_VERIFY` is read at import time (`utils/typing.py:15`) and silently gates beartype/jaxtyping
  verification, so it is a runtime knob, not a test-harness one. The rest (`AMINX_BV_*`,
  `AMINX_CHECKPOINTS`, `AMINX_AMBER_TEST_DIR`, `AMINX_MODEL_BASE_URL`, `AMINX_MODELS_RELEASE_TAG`,
  `AMINX_EBM_ORBAX_CHECKPOINT`, ...) are read by scripts, tests, browser JS or workflows; S3-08
  publishes the full classified table (name, read site, class, fallback) as
  `docs/migration/aminx_to_molxmpnn_env.md`. `pages.yml` reads the **GitHub repository variable**
  `vars.AMINX_MODELS_RELEASE_TAG`.
- Blast radius: **812** tracked files mention the word `aminx`: 143 under `src/`, 253 `tests/`,
  130 `scripts/`, 19 `docs/`, plus dated docs and `outputs/` that must *not* be rewritten
  [A7, spike S5].
- The previous rename (prxteinmpnn -> aminx, 260813) is the best evidence for how this goes
  wrong: **64 tracked files still contain `prxteinmpnn`**, and `scripts/check_model_boundary.sh:18`
  greps only for the dead name, so it passes while checking nothing [A19, spike S5].
  `proteinsmc/src/proteinsmc/scoring/mpnn.py:14` still guards `find_spec("prxteinmpnn")`, so MPNN
  scoring there is unreachable [A20].
- The boundary script is vacuous for a second reason beyond the dead name: it guards
  `model/_inference/` and `model.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand)`, and
  `src/aminx/model/` has **no `_inference` module** (files: `capabilities, decoder, diffusion_mpnn,
  dropout, encoder, features, ligand_*, mpnn*, packer, versions`); the only mentions are the
  docstring at `model/__init__.py:9-10` [A28]. Renaming its patterns would leave it checking a
  directory that does not exist.
- Other tracked files carry the name in their **path**, not only content: `.ast-grep/rules/aminx-xtrax-internals-boundary.yml`,
  `.bth/claims/aminx-bv-layer-{a,b}.claim.toml` (bathos claim anchors), `.praxia/audit_chain_vendor/aminx_chain_fields.json`,
  `browser/aminx-sampler/*`, and many dated docs/handoffs/sprint plans. Provenance helpers
  `resolve_aminx_version` / `aminx_version` are used at `sink_provenance.py:171`, `designs.py:116,221`,
  `runner.py:1562,1643`, `multistate_poe.py:570,704`, `streaming.py:96` [A31, by grep in this revision].
- EBM isolation, re-measured: **0** importers of `aminx.ebm` outside `src/aminx/ebm` [A18, spike S5].
  The shim therefore has one module family (`aminx.ebm`) it must refuse to alias once S2 lands.

### 3.3 Consumers (the recon brief said "none import aminx"; that is wrong here)

- Declared dependencies on `aminx`: `mpnn_ext` (`../../../../mpnn_ext/pyproject.toml:8`, workspace
  source `:12`, member `:15`), `asr` (`../../../../asr/pyproject.toml:23`, source `:94`, member `:134`),
  `tev_design` (`../../../../tev_design/pyproject.toml:9`, pinned to a local `aminx-0.1.0a24` wheel
  `:66`, which the rename cannot affect) [A1].
- Git submodules fetched from `https://github.com/maraxen/aminx.git`: `asr`
  (`../../../../asr/.gitmodules:3`), `mpnn_ext` (`../../../../mpnn_ext/.gitmodules:3`), `hautespout`
  (`../../../../hautespout/.gitmodules:3`) [A10]. Their pinned SHAs resolve only while those SHAs are
  fetchable from that URL. This single fact drives the repo strategy (4.1).
- `asr/vendor/proteinsmc/.../scoring/mpnn.py:14` guards `find_spec("aminx")`. Spike S4: once the hub
  is installed under the name `aminx`, that guard reports MPNN available and then fails at the call
  site with `No module named 'aminx.scoring'` instead of the friendly "not installed" error [A6, A20].
- **asr's `vendor/proteinsmc` is a plain directory copy, not a submodule**: `asr/.gitmodules` lists
  only `aminx` and `vendor/euclidean_fast_attention`, and the vendored `scoring/mpnn.py:14` already
  differs from `proteinsmc` main (`AMINX_AVAILABLE`/`find_spec("aminx")` vs main's
  `PRXTEINMPNN_AVAILABLE`/`find_spec("prxteinmpnn")`) [A27]. Two codebases, two edits (S3-15, S3-18);
  a proteinsmc change does not propagate to asr's copy. That a lab project vendors another lab
  project at all is a D6-adjacent smell; it is filed as a question (Q12), not fixed here.
- hautespout vendors via submodule `vendor/aminx` (`hautespout/.gitmodules:1-3`). Whether its
  `pyproject.toml` also declares `aminx` is **not confirmed**: the main-checkout `pyproject.toml`
  contains no `aminx` string; the declaration the review cited was seen only in a worktree. S3-04
  scans the vendored surface directly, so the item does not lean on the declaration [A32].
- Comments/citations only (no change needed beyond wording): `sweetprots`, `alphex/known.py`,
  `praxia/experiments/colliculix`.

### Assumption Ledger

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | mpnn_ext, asr and tev_design declare aminx as a dependency, and mpnn_ext, asr and hautespout vendor it as git submodules; the recon brief's 'none import aminx' is wrong for this partition. | Consumer-migration items S3-14..S3-18 shrink or vanish | VERIFIED | spike: S9 |
| A2 | PyPI project name molxmpnn is unregistered (the JSON API returns 404). | Name decision (Q1) reopens | VERIFIED | spike: S1 |
| A3 | Every published aminx release on PyPI is an alpha or other pre-release version. | Shim/handover version policy needs a stable-release story | VERIFIED | spike: S1 |
| A4 | A sys.meta_path alias finder whose loader returns the real module object gives identical module and class objects under aminx.* and molxmpnn.* names, while aliasing only the package in sys.modules yields duplicate module objects. | Shim must become a full re-export copy instead of an alias finder | VERIFIED | spike: S2 |
| A5 | An old aminx.* class path embedded in a pickle fails to load with ModuleNotFoundError after the rename, and a find_class prefix remap restores loading. | No remap item needed, or checkpoints must be re-serialised | VERIFIED | spike: S3 |
| A6 | A find_spec('aminx') availability guard returns True for any installed package named aminx, including a hub with no scoring module, so the guard's friendly ImportError path is bypassed. | Guard rewrite in S3-15/S3-18 is unnecessary | VERIFIED | spike: S4 |
| A7 | The rename touches at least 800 tracked files (case-insensitive word aminx), about 140 of them under src. | Item sizing and the single-atomic-PR decision change | VERIFIED | spike: S5 |
| A8 | Bathos run provenance is keyed by a project_slug column (144 runs under aminx) and the runs table has no separate project id column. | A slug rename could be done safely via a project id, changing S3-20 | VERIFIED | spike: S6 |
| A9 | ruff banned-api matches the literal module string, so a ban on aminx.X silently stops firing once code imports molxmpnn.X. | The stale-name gate (S3-09) is unnecessary for banned-api | VERIFIED | spike: S8; changed: S7 was an invalid probe (flag clash), S8 re-ran it and measured the positive control (1 violation, exit 1) and the rename victim (0 violations, exit 0) |
| A10 | asr, mpnn_ext and hautespout fetch the aminx submodule from https://github.com/maraxen/aminx.git. | Repo-strategy risk to consumers is smaller than stated | VERIFIED | spike: S9 |
| A11 | release.yml publishes through PyPI trusted publishing (OIDC id-token, environment pypi, no API token). | S3-03 / S3-12 publisher registration steps change to token handling | VERIFIED | read: .github/workflows/release.yml:30-41 |
| A12 | PyPI trusted-publisher registrations bind to the GitHub owner, repo name, workflow file and environment, so the existing `aminx` publisher stops matching once the repo is renamed (S3-02) and the new identity must be added to the existing project. | S3-03's publisher step is unnecessary | UNVERIFIED | deferred: this is PyPI's documented behaviour (r1 C2) but docs.pypi.org is blocked from this sandbox and the registration state needs a PyPI login; S3-02 records the old publisher's state before the rename and S3-03 verifies the new entry. Consequence assumed in the DAG: no PyPI release of `aminx` or `molxmpnn` is possible between S3-02 and S3-03 |
| A13 | GitHub keeps redirecting a renamed repository's old URL (web, git fetch, submodule clone) until a new repository is created under the old name, which removes the redirect. | Option 1 loses its main advantage and option 2 becomes cheap, reopening Q2 | UNVERIFIED | deferred: proving it needs two throwaway repositories on the user's account and api.github.com is blocked from this sandbox (probes returned curl exit 7), and it is GitHub's documented behaviour, so S3-02 includes a redirect probe on the real repo |
| A14 | Runtime strings carry the package name beyond imports: files('aminx.model_params'), importlib.metadata.version('aminx'), importlib.import_module('aminx.sampling.sample'), the typer app name and XDG cache dir. | Codemod can be import-only | VERIFIED | read: src/aminx/io/weights.py:216; read: src/aminx/io/sink_provenance.py:201; read: src/aminx/host/sampling_driver.py:36; read: src/aminx/cli.py:37-129 |
| A15 | Every design and sampling store written by the runner stamps an aminx_version root attribute. | No dual-stamp item needed | VERIFIED | read: src/aminx/io/designs.py:221; read: src/aminx/host/runner.py:1643 |
| A16 | potts/calibration.py unpickles calibration checkpoints with pickle.loads. | No remap item needed | VERIFIED | read: src/aminx/potts/calibration.py:159 |
| A17 | Released wheels pin weights to HF repo maraxen/aminx at commit aa80d0fd..., overridable by AMINX_WEIGHTS_REVISION and AMINX_WEIGHTS_DIR. | Weights can move without freezing the old repo | VERIFIED | read: src/aminx/io/weights.py:23-40 |
| A19 | The previous rename left stale names: 64 tracked files still contain prxteinmpnn and scripts/check_model_boundary.sh guards only the dead name. | Stale-name sweep (S3-09) is less urgent | VERIFIED | spike: S5; read: scripts/check_model_boundary.sh:18 |
| A18 | No file under src outside src/aminx/ebm imports aminx.ebm. | Shim alias table and S2 decoupling assumptions break | VERIFIED | spike: S5 |
| A20 | asr's vendored proteinsmc guards find_spec('aminx'), and proteinsmc main guards the dead name find_spec('prxteinmpnn'). | S3-18 would not be needed | VERIFIED | spike: S9 |
| A21 | pyproject keys embed the package name: console script, package-data, exclude-package-data (88.8 MB wheel guard), ruff banned-api and ty root. | Codemod needs fewer targets | VERIFIED | read: pyproject.toml:34; read: pyproject.toml:153; read: pyproject.toml:245; read: pyproject.toml:285; read: pyproject.toml:292 |
| A22 | 15 weight files are LFS-tracked under src/aminx/model_params and LFS patterns are extension-based, so a git mv keeps LFS object ids. | LFS migration step needed | VERIFIED | read: .gitattributes:1-6 |
| A23 | Claude auto-memory is keyed by the checkout path (-home-marielle-projects-aminx), so renaming the checkout directory orphans it. | S3-20 needs no memory step | VERIFIED | spike: S9 |
| A24 | The browser demo was never deployed (`pages.yml` has no runs, zero repo variables set), and `maraxen.github.io/aminx/` serves a stale Sphinx build from the old repo's `gh-pages` branch (legacy Pages, last commit 2025-11-10), so there is no demo URL to preserve, only a stale docs URL. Reworded in coherence round 2 (CH2-07) from "the demo is live"; the old claim is refuted. | S3-21's Pages work is a docs-URL decision, not a demo redirect | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:46-53 (S5 A1: `gh run list --workflow pages.yml` returns `[]`, `pages.yml:8` is `workflow_dispatch` only); read: .github/workflows/pages.yml:7-8; changed: S3-21 now records where the Sphinx site lands after S3-02 and whether any `/aminx/` redirect is possible; the demo-redirect framing is dropped |
| A43 | GitHub does not redirect a project-site Pages URL (`<user>.github.io/<old-repo>/`) after the repo is renamed, and a stub at `/aminx/` would need a repo (or user-site path) of that name, which option 1 of 4.1 forbids recreating as `maraxen/aminx`. | S3-21 can promise a redirect | UNVERIFIED | deferred: Pages behaviour on rename needs api.github.com and a throwaway repo, blocked here; S3-21 probes it on a throwaway repo before choosing between "redirect from a user-site repo" and "accept /aminx/ as dead" |
| A25 | External (non-lab) installs of aminx from PyPI are few. | Deprecation window in S3-22 must be longer | UNVERIFIED | deferred: PyPI download statistics are not reachable from this sandbox so S3-22 requires the owner to read them |
| A26 | praxia backlog/scope identity is unaffected by the package rename (praxia.toml carries a workspace_id UUID). | S3-20 needs an extra praxia re-keying step | UNVERIFIED | deferred: scope-name keying lives in the praxia DB which is reachable only via MCP outside this session so S3-20 verifies it |
| A27 | asr's `vendor/proteinsmc` is not a git submodule and its `scoring/mpnn.py` differs from proteinsmc main (AMINX_AVAILABLE/find_spec('aminx') vs PRXTEINMPNN_AVAILABLE/find_spec('prxteinmpnn')). | S3-15 could bump it by submodule update | VERIFIED | read: ../../../../asr/.gitmodules:1-6 (r1 C6); read: ../../../../asr/vendor/proteinsmc/src/proteinsmc/scoring/mpnn.py:14 (r1 C6); read: ../../../../proteinsmc/src/proteinsmc/scoring/mpnn.py:14 |
| A28 | `src/aminx/model/` has no `_inference` module, so `scripts/check_model_boundary.sh` guards a nonexistent boundary even after its names are fixed. | S3-09 only needs a name fix for that script | VERIFIED | read: src/aminx/model/__init__.py:9-10 (r1 C9); read: scripts/check_model_boundary.sh:14-23 (r1 C9) |
| A29 | Library code under `src/` reads exactly four `AMINX_*` env vars (CACHE_DIR, VERIFY, WEIGHTS_DIR, WEIGHTS_REVISION), and `AMINX_VERIFY` is read at module import. | Fallback list and the env table change | VERIFIED | read: src/aminx/utils/typing.py:15 (r1 C7); read: src/aminx/io/weights.py:39-40 (r1 C7); read: src/aminx/cli.py:108-120 (r1 C7) |
| A30 | `importlib.util.find_spec('pkg.sub.mod')` raises ModuleNotFoundError when `pkg` is not installed, whereas `find_spec('pkg')` returns None, so guards must test the root package first. (Round 2: this covers only the root-absent case; the root-present case is A36.) | The Q11 guard form works as first written | VERIFIED | spike: S10 |
| A31 | Version-provenance helpers (`resolve_aminx_version`, `aminx_version` attr) are used at sink_provenance.py:171, designs.py:116/221, runner.py:1562/1643, multistate_poe.py:570/704 and streaming.py:96, and some names must survive the rename (legacy stamp) while others are renamed. | Codemod needs no provenance rule | VERIFIED | read: src/aminx/host/runner.py:1562; read: src/aminx/host/runner.py:1643; read: src/aminx/sampling/multistate_poe.py:570; read: src/aminx/sampling/multistate_poe.py:704 |
| A32 | hautespout's `pyproject.toml` declares `aminx` as a dependency (the review saw it in a worktree). | S3-17 is a pure submodule bump | UNVERIFIED | deferred: the main-checkout hautespout/pyproject.toml has no `aminx` string and the worktree is outside this sandbox's write scope; S3-04 scans `vendor/aminx` consumers directly so no item leans on it |
| A33 | A PyPI pending publisher can only be created for a project name that does not yet exist and does not reserve the name; `aminx` already exists with 10 releases so it needs an ordinary publisher entry on the existing project. | S3-27 placeholder upload is unnecessary | UNVERIFIED | deferred: PyPI documentation is blocked here and creating a publisher needs a login; the `aminx` half is settled by A3 (project exists); S3-27 sidesteps the question by uploading a real placeholder release. Consequence (r2 R2-3): once S3-27 has created project `molxmpnn`, S3-03 registers an **ordinary** publisher on it, not a pending one |
| A34 | Renaming a Hugging Face model repo redirects the old name for web and `hf_hub_download`, keeps commit SHAs, and the redirect disappears if the old name is later recreated. | Q8 option (c) drops out and copy is the only path | UNVERIFIED | deferred: huggingface.co is blocked from this sandbox; S3-11 probes it on a throwaway repo before Q8 is answered |
| A35 | `release.yml` fires on every `release: published` event and has **two** jobs, `build` (no environment, no permissions) and `publish` (`needs: build`, `environment: pypi`, `id-token: write`); the tag filter must therefore be on both. | Release routing needs no change | VERIFIED | read: .github/workflows/release.yml:3-5; read: .github/workflows/release.yml:8; read: .github/workflows/release.yml:27-31 (r2 R2-16; r1 wrongly said one job) |
| A36 | `importlib.util.find_spec('pkg.sub')` imports the parent package `pkg` (executing its `__init__`) before looking up `sub`, and aminx's `__init__` eagerly imports io, run, sampling and scoring (the JAX stack). So a dotted-name guard is not import-free and any non-ImportError raised by the parent breaks the caller. | The guard may test the dotted submodule | VERIFIED | read: /usr/lib/python3.12/importlib/util.py:88-92; read: src/aminx/__init__.py:13-24 (r2 R2-6); changed: S3-18/Q11 guard is root-only `find_spec("molxmpnn")`, submodule check at call time |
| A37 | `uv.lock` embeds the root package as `name = "aminx"` / `source = { editable = "." }` (one entry), so the rename edits the lock, and a regenerated lock can also move third-party versions. | Lockfile needs no handling | VERIFIED | read: uv.lock:67-69 (r2 R2-10) |
| A38 | A first upload to a PyPI project name that does not exist needs a credential (API token); OIDC trusted publishing for a not-yet-existing project needs a pending publisher, which S3-27 deliberately does not use. | S3-27 needs no token step | UNVERIFIED | deferred: docs.pypi.org and a PyPI login are not reachable from this sandbox; S3-27 is a user_decision item and records the credential type, holder and revocation in the ADR (r2 R2-3) |
| A39 | TestPyPI projects `aminx` and `molxmpnn` may or may not exist; a pending publisher applies only where absent, an ordinary one where present. | S3-03's TestPyPI step is uniform | UNVERIFIED | deferred: test.pypi.org is blocked here; S3-01 probes both names and S3-03 branches on the result (r2 R2-3) |
| A40 | Read the Docs slug `molxmpnn` is available, and an old RTD project can redirect to a new URL (a project-level redirect rule on `aminx.readthedocs.io`). | S3-21's RTD step changes (different slug; stub page instead of a redirect) | UNVERIFIED | deferred: readthedocs.org is blocked here; S3-01 probes slug availability and S3-21 probes the redirect on a throwaway project first (r2 R2-15) |
| A41 | Titanix GPU fixtures are bit-reproducible only on an identical device, driver, jaxlib and XLA-flag set; a different set can change hashes without any rename effect. | Manifest pinning in S3-05/S3-07 is unnecessary | UNVERIFIED | deferred: titanix is reachable only via ssh outside this sandbox; S3-05 measures run-to-run variance and records the environment, S3-07 refuses to compare on mismatch (r2 R2-11) |
| A42 | `[tool.pytest.ini_options]` has `pythonpath = ["src","tests"]` and `addopts = "-m 'not parity_heavy and not slow'"`, so a default non-heavy run collects `tests/compat/**`; a new marker must be registered under `markers` or pytest warns. | A compat test could be skipped by path alone | VERIFIED | read: pyproject.toml:226-229 (r2 R2-1) |
| A44 | A module-level `pytest.importorskip("aminx")` makes a missing `aminx` produce one collect-time skip per module, with reason beginning `could not import 'aminx'`, not one skip per test; so skip accounting must be keyed by file, not test count. | Gate (e) would be keyed on the wrong unit | VERIFIED | read: .venv/lib/python3.14/site-packages/_pytest/outcomes.py:272-276; read: .venv/lib/python3.14/site-packages/_pytest/runner.py:411-412 (convergence-check on R2-1) |

Spike tooling: `scripts/loop/adversarial_metrics.py` is not in this workspace, so spikes were run with
`python3.13 <praxia triage-worktrees>/scripts/loop/adversarial_metrics.py spike-run --workspace <this worktree>`
(stdlib only, no `uv`). Records are in `.praxia/spikes/261001_molxmpnn-rename/S*/`. S7 is an invalid probe
(it passed `--isolated` with `--config`, which ruff rejects, so it measured nothing). S8 is the corrected
re-run. The spikes ran on Python 3.13.12 while the sandbox default is 3.12.3, matching `requires-python >=3.13`.

## 4. Design

### 4.1 Repository strategy (the critical choice)

Facts that constrain it: three repos pin SHAs of `maraxen/aminx` through submodules [A10]; GitHub
redirects a renamed repo's old name only until a new repo takes that name [A13, documented,
unverified here]; the repo holds 504 MiB of packed history, 47 LFS files, 33 tags (measured with
`git count-objects`, `git lfs ls-files`, `git tag`), the commits that 144 bathos runs pin, and the
`pypi` environment.

| Option | What it is | Old URLs / SHAs | History, issues, PRs, tags, LFS | Verdict |
|---|---|---|---|---|
| **1. Rename in place; hub gets another slug** | `maraxen/aminx` -> `maraxen/molxmpnn`. Hub lives in `maraxen/aminx-hub` (or an org). **Never create `maraxen/aminx` again.** | Redirect lives indefinitely; every submodule pin and `git+https` URL keeps resolving | All preserved (same repository) | **Recommended.** Zero consumer breakage, zero history work. Cost: the hub's repo slug is not `aminx`; `github.com/maraxen/aminx` shows molxmpnn forever via redirect |
| 2. Rename, then new hub repo at `maraxen/aminx` | The "obvious" reading of D2 | **Redirect destroyed** the moment the hub repo exists; asr/mpnn_ext/hautespout fresh clones now fetch the hub and fail with "not our ref" on their pinned SHAs; every external link now lands on the hub | Preserved in molxmpnn | Rejected unless the consumer inventory is zero (S3-22) and the owner accepts permanent breakage of unknown external URLs. Irreversible |
| 3. Keep `maraxen/aminx` for the hub; push full history to a new `maraxen/molxmpnn` | Mirror clone (same SHAs, `git lfs push --all`), then delete the MPNN tree in `aminx` | Pins keep resolving as long as the MPNN history stays reachable (needs a `legacy/mpnn` branch) | The hub repo inherits 504 MiB of MPNN history, 33 MPNN tags and the `release: published` trigger (which would publish the hub as an MPNN wheel); PRs and issues (numbering reaches #174) stay on the wrong repo and PRs cannot be transferred; molxmpnn starts with no PR/issue history | Viable but strictly worse than 1 for a first-order goal (nothing breaks). Choose only if the repo slug `aminx` is non-negotiable |
| 4. `git filter-repo` a fresh molxmpnn | Rewrites SHAs | All old pins dead in the new repo | Large history loss/rewrite risk | Rejected. S2 uses filter-repo for the *EBM* repo, not for this |

Consequence of option 1 for D2: "the name `aminx` becomes the hub" is satisfied for brand, site,
CLI, skill and (later) PyPI. The one resource that stays molxmpnn's is the GitHub slug
`maraxen/aminx` (as a redirect). Q2 asks the user to confirm; the design does not depend on the
answer until S3-21/S3-22.

Guard (S3-02): the hub repo's CI (S5) and molxmpnn's `CONTRIBUTING.md` carry a one-line check
`gh repo view maraxen/aminx --json url` (S3-02 first verifies that `gh` reports the redirect target
rather than an error) that must resolve to `.../molxmpnn`; if someone creates a repo at the old
name it fails loudly instead of silently re-pointing three submodules.

### 4.2 Names

| Layer | molxmpnn (this spec) | aminx (freed) |
|---|---|---|
| Import package | `molxmpnn` (`src/molxmpnn/`) | `aminx` becomes the shim, later the hub's namespace if any |
| PyPI dist | `molxmpnn` 0.2.0a4 (continues the 0.2.0 alpha line) | `aminx 0.2.0a4` shim now; handover in S3-23 is conditional (Q3) |
| Console script | `molxmpnn = molxmpnn.cli:main` | `aminx` is owned by the shim dist (two dists cannot both own it) |
| GitHub | `maraxen/molxmpnn` (rename of the existing repo) | `maraxen/aminx` redirects; hub is `maraxen/aminx-hub` |
| HF weights | `maraxen/molxmpnn` by in-place rename (c) if the S3-11 probe passes, else copy (a) with new commit SHAs (Q8) | old `maraxen/aminx` redirected or frozen forever, never repurposed, never recreated |
| Docs | `molxmpnn.readthedocs.io` | `aminx.readthedocs.io` redirects to it |
| Skill | `using-molxmpnn` | `using-aminx` becomes a pointer until S5 writes the hub skill |
| Bathos slug | new runs `molxmpnn`; legacy `aminx` frozen (Q5) | hub uses `aminx-hub` |
| Local checkout | `~/projects/molxmpnn` (Q7) | `~/projects/aminx` is the hub checkout later |

### 4.3 The `aminx` shim distribution

Lives in this repo at `packaging/aminx-shim/` and is built and published by its **own workflow**
`.github/workflows/release-aminx-shim.yml` (see 4.9; so hub work in another repo never has to touch
PyPI `aminx` before S3-23, and a molxmpnn release can never re-upload the shim).

```
packaging/aminx-shim/
  pyproject.toml          name="aminx" version="0.2.0a4" (shim line, versioned independently of molxmpnn) requires-python=">=3.13"
                          dependencies=["molxmpnn>=0.2.0a4,<1"]    # range, not ==: an exact pin
                                                                   # makes `pip install -U molxmpnn` conflict
                          extras forwarded 1:1: cpu, cuda12, benchmark, tests, foldcomp, docs, dev
                                                (each -> "molxmpnn[<extra>]")
                          [project.scripts] aminx = "aminx.cli:main"
  src/aminx/__init__.py   imports molxmpnn eagerly (same cost as today), emits ONE DeprecationWarning
                          (stacklevel=2), sets __path__ = [], forwards attributes/__all__/__dir__ and
                          __version__ to molxmpnn, installs the finder
  src/aminx/_alias.py     AliasFinder + AliasLoader (below)
  src/aminx/gone_modules.toml   OWNED BY S3 (S3-10 creates, S3-25 edits). Row kinds:
                          [[gone]] prefix = "aminx.ebm" message = "moved out of aminx, see CHANGELOG"
                                   when = "target-absent"   # refuse only if molxmpnn.ebm is not
                                   # importable; at the 0.2.0a4 tag commit it still is (4.9)
                          [[forward]] prefix = "aminx.ebm" target = "<S2 project>" (S3-25, lazy,
                          only if installed, else the install message)
  src/aminx/cli.py        prints a stderr notice, then `raise SystemExit(molxmpnn.cli.main())`
  src/aminx/py.typed
  tests/
```

`AliasFinder` is inserted at `sys.meta_path[0]`. For `aminx.<x>` it (a) raises
`ModuleNotFoundError(message)` if `<x>` matches a `gone_modules.toml` prefix, else (b) imports
`molxmpnn.<x>` and returns a `ModuleSpec` whose loader's `create_module` returns **the real module
object**. `_init_module_attrs` overwrites `__spec__` with the alias spec, so the loader saves the real
spec in `create_module` and restores it in `exec_module` (spike S2 asserts `real.__spec__.name ==
real.__name__` afterwards). Spike S2 measured: module identity, class identity, `isinstance` across
names, pickle round trip, one warning, informative error for a gone module: all true; and the
negative control (`sys.modules["aminx"] = molxmpnn` with a shared `__path__`) produced distinct
module and class objects, so the naive approach is not sufficient [A4].

Warning visibility: `DeprecationWarning` is hidden by Python's default filters unless raised
directly in `__main__`. The spec keeps `DeprecationWarning` as requested; the CLI banner on stderr
is the always-visible nudge (Q6).

### 4.4 Compatibility inside molxmpnn (S3-08)

- **Env vars.** One helper `molxmpnn._compat.env(name)` reads `MOLXMPNN_<X>` first, then `AMINX_<X>`
  with a once-per-process `DeprecationWarning` that names the new variable. **Rule: every variable
  read from `src/` goes through `env()` with the fallback**, which is the four names in A29
  (`CACHE_DIR`, `VERIFY`, `WEIGHTS_DIR`, `WEIGHTS_REVISION`). `AMINX_VERIFY` is included because it is
  read at import (`utils/typing.py:15`) and a silent rename would turn verification off for anyone
  with it set in CI, slurm or bathos sidecars. Variables read outside `src/` fall in three classes,
  all listed per name (read site, class, fallback yes/no) in `docs/migration/aminx_to_molxmpnn_env.md`
  by S3-08: **(i) user-facing runtime/CI** (`MODEL_BASE_URL`, `MODELS_RELEASE_TAG`, `CHECKPOINTS`:
  read by browser JS, scripts and workflows) get the same fallback in the reader, and
  `vars.AMINX_MODELS_RELEASE_TAG` is read by `pages.yml` (no repo variable is set today and S5-34
  deletes `pages.yml`, so no `MOLXMPNN_MODELS_RELEASE_TAG` variable is created; the `||` fallback is
  added only if `pages.yml` survives until a variable is set); **(ii) test/harness knobs** (`BV_*PERTURB`, `AMBER_TEST_DIR`) rename
  with no fallback because they are set next to the code that reads them; **(iii) EBM-owned**
  (`EBM_ORBAX_CHECKPOINT`) leave with S2. Test: with only the old name set, the value is honoured and
  the warning fires once; with both set, the new name wins.
- **Cache dir** (`cli.py:108-130`): resolution order explicit arg > `MOLXMPNN_CACHE_DIR` >
  `AMINX_CACHE_DIR` (deprecated) > `[tool.molxmpnn] cache_dir` in the nearest `pyproject.toml` >
  `${XDG_CONFIG_HOME:-~/.config}/molxmpnn/config.toml` > the existing XDG cache default; expose
  `cache_dir_source()`. The legacy `~/.cache/aminx/inputs` is never written to; entries are keyed by
  input content hash, so the old dir is simply abandoned and may be deleted. **This one resolver
  (`molxmpnn._compat.resolve_dir(name, explicit=None)`, same layers, same `*_source()` reporting) is
  reused by every derived-data directory this spec introduces**: the S3-05 golden `--out`/work dir
  (`MOLXMPNN_GOLDEN_DIR`, `[tool.molxmpnn] golden_dir`) and the S3-11 staging dir
  (`MOLXMPNN_STAGING_DIR`, `[tool.molxmpnn] staging_dir`). S3-11 runs after S3-08 and imports it
  (`depends_on` S3-08). S3-05 runs on the *pre-rename* tree, where `molxmpnn` does not exist, so it
  uses a stdlib-only copy of the same function at `scripts/rename/_resolve_dir.py` (created in S3-05,
  carried unchanged through S3-07), and an S3-08 test asserts the two implementations agree on a table
  of layer combinations. No default path is hardcoded in either.
- **Provenance stamp.** Writers stamp both `molxmpnn_version` (primary) and `aminx_version` (legacy,
  same value) for one minor version; readers accept either; `SINK_PROVENANCE_VERSION` bumps additively.
  `resolve_aminx_version` becomes `resolve_version` reading `importlib.metadata.version("molxmpnn")`
  (reading `"aminx"` under the shim would return the shim's version). **The codemod does not touch
  this surface** (see 4.5, class P2): S3-07 leaves `aminx_version` keys and the `resolve_aminx_version`
  name alone, so the tree after S3-07 still stamps only the legacy key; S3-08 then introduces the
  dual stamp and the rename of the helper. The intermediate state is therefore well defined and tested
  (S3-07 gate: store written by renamed tree has `aminx_version`; S3-08 gate: has both).
- **Pickles.** `load_calibration` uses a `pickle.Unpickler` subclass whose `find_class` remaps module
  prefix `aminx` -> `molxmpnn` (spike S3: plain load fails with `No module named`, remap loads and
  leaves stdlib classes alone) [A5].
- **Weights.** `HF_REPO_ID`/`HF_REVISION` cut over in S3-11; old pins stay valid for old wheels because
  the old repo is frozen [A17].

### 4.5 The rename PR (S3-07) and its tooling (S3-06)

Atomic, mechanical, one PR: `git mv src/aminx src/molxmpnn`, then the codemod. It cannot be split
without leaving an uninstallable intermediate tree. The codemod `scripts/rename/rename_aminx.py`:

- Idempotent, `--dry-run` default, driven by `scripts/rename/rename_rules.toml`. Every tracked
  path/line falls into exactly one class; unmatched occurrences land in **review** and make `--apply`
  exit non-zero until a human classifies them (so nothing is silently skipped or silently rewritten):
  - **R (rewrite):** imports, dotted-string module paths, `files()`/`import_module`/`metadata.version`
    literals, pyproject keys (name, scripts, package-data, exclude-package-data, banned-api, ty root),
    **`uv.lock`** (handled by a dedicated step, not by text rewrite: the codemod renames only the
    root entry `name = "aminx"` / `source = { editable = "." }` [A37]; it then runs `uv lock --check`
    and fails if `--check` wants more than that edit; no `uv lock --upgrade`, no re-resolve; the
    S3-07 gate asserts every third-party `version` and hash line is unchanged),
    `AMINX_` -> `MOLXMPNN_` for the *readers* in classes (i)/(ii) of 4.4, `browser/aminx-sampler` ->
    `browser/molxmpnn-sampler`, CI `path: aminx`, **case variants**: `Aminx` in prose/docstrings ->
    `Molxmpnn` only where it names the package (brand), and the URL `maraxen/Aminx.git` ->
    `maraxen/molxmpnn.git` (GitHub is case-insensitive; the form is canonicalised once).
  - **P (preserve), listed in `rename_rules.toml` and each with a fixture:**
    P1 history (CHANGELOG entries, dated `.praxia/docs/**`, `.praxia/handoffs/**`,
    `.praxia/sprint_plans/**`, `outputs/**`, `*.bth.toml` recorded commands of past runs);
    **P2 the provenance legacy surface**: the string `"aminx_version"`, `_aminx_version` and
    `resolve_aminx_version` (renamed by S3-08, not S3-07, per 4.4) and `SINK_PROVENANCE_VERSION`
    readers; **P3 `tests/compat/old_names/**` and `tests/compat/path_map.toml`** (S3-04's consumer-surface
    test deliberately lists *old* `aminx.*` paths so it exercises the shim; the codemod must not
    rewrite them, and S3-13 requires that test to pass through both names). The glob is narrow on
    purpose (convergence-check on R2-1): `tests/compat/new_names/**` is **not** P3. It is generated
    output, listed in `rename_rules.toml` as class `G` (the codemod's `--apply` skips it; the
    generator below owns it; a fixture asserts both), so no preserve glob matches a directory whose
    content is meant to change. **Collection rule (round 2, R2-1; reworked in the convergence-check).**
    `pytest` collects `tests/compat` in a default run [A42], and between S3-07 and S3-10 nothing
    provides `aminx.*`. So S3-04 splits the directory by what each test needs:
    `tests/compat/old_names/` (every test carries the registered marker `consumer_shim` and a
    module-level guard `pytest.importorskip("aminx")`: before the rename the real package answers and
    the tests run (S3-04 gate); after S3-07 nothing answers to `aminx` and each module is skipped, not
    failed; once the shim is installed they run again) and `tests/compat/new_names/` (the same surface
    as `molxmpnn.*` paths; it runs everywhere).
    - **Generator (named, owned by S3-06).** `scripts/rename/gen_compat_new_names.py` reads the
      checked-in `tests/compat/path_map.toml` (old -> new) and the `old_names/` modules and writes
      `new_names/`; `--write` is deterministic and idempotent, `--check` regenerates in memory and
      exits non-zero on any difference. It exits non-zero when an old path in an `old_names` module
      has no map row or a map row is never applied. S3-06 builds it with a *synthetic* map and
      synthetic old_names modules (`molxmpnn` does not exist when S3-04 runs); S3-07 runs `--write`
      on the real map and commits the result; CI runs `--check`.
    - **Skip accounting is by file, against a measured baseline, never by a count.** A module-level
      `importorskip` produces one collect-time skip per *module* [A44], so "number of old_names
      tests" is the wrong unit, and the non-heavy suite already contains skips unrelated to the
      rename (75 `skip` sites in 37 test files, `tests/conftest.py:41` when weights and HF are both
      unreachable, `:62`/`:66` when no parser backend is installed). The S3-07 titanix bathos record
      therefore contains **two runs on the same node under the same environment stamp**:
      `before` = the non-heavy suite at the S3-07 parent commit, `after` = at the PR head, each with
      `-rs --junitxml`. Each yields a set of (file path, reason) skip records, with paths normalised
      through the `[[path]]` table. Let `O` be `git ls-files 'tests/compat/old_names/test_*.py'` at
      the PR head. The pre-registered outcome is: (1) `before` has zero failures, and every `O`
      file ran with zero skips in it (S3-04's property, re-checked); (2) every file in `O` appears
      in `after` with a reason beginning `could not import 'aminx'`; (3) `after` minus the `O` records
      is a subset of `before` (same file, same reason string): no skip outside `old_names` is new,
      and a skip that disappears is allowed (it ran and passed); (4) `tests/compat/new_names`
      passes with zero skips. Skips present in `before` (weights/HF unreachable, no parser backend,
      the rest) are allowed to *remain*, never to grow; the gate does not assert any total, and if the
      environment stamps of the two runs differ the record is "not evaluated". Negative controls in the
      sidecar: a seeded extra skip in a non-compat test must fail (3); an `old_names` module whose
      `importorskip` is removed must fail (2).
    - After S3-10 (shim installed): `pytest -m consumer_shim tests/compat/old_names` has **zero
      skips** and passes (S3-10 gate, repeated post-publish by S3-13 over all of `tests/compat`).
      `consumer_shim` is registered under `pyproject.toml` `markers` by S3-04 and is **not** excluded
      by `addopts`. Gate (g) is "`tests/compat/old_names/**` and `path_map.toml` are byte-unchanged,
      `gen_compat_new_names.py --check` is clean" (nothing edits the old files; `new_names/` is the
      one new directory);
    **P4 the shim** (`packaging/aminx-shim/**`) and `_compat.py`;
    **P5 bathos anchors**: `.bth/claims/**` contents and every path a sidecar or claim hash-anchors.
  - **V (review):** anything else with the word, printed with file:line.
- **Path renames are a separate, explicit table** (`[[path]]` rules), not a side effect of
  `git mv src/aminx`: `src/aminx -> src/molxmpnn` and `browser/aminx-sampler -> browser/molxmpnn-sampler`
  (rename); `.ast-grep/rules/aminx-xtrax-internals-boundary.yml` (rename, its content is R);
  `.praxia/audit_chain_vendor/aminx_chain_fields.json` (V review item, default preserve: who
  reads this file by name is not established, so it is not renamed until the owner answers); `.bth/claims/aminx-*.claim.toml` (**preserve path and bytes**: a
  claim's hash anchors and past-run provenance point at them; new claims are created under the new
  slug, old ones are read-only history, and the S3-07 gate asserts their sha256 unchanged); dated
  docs and handoffs with `aminx` in the filename (preserve). Benchmarks named `*aminx*` are rename
  with an alias note in the CHANGELOG if a sidecar references the old script path (the sidecar's
  recorded command is P1, so the old path is kept as a one-line forwarding stub).
- Reusable on in-flight branches (`--apply` after a merge) so S1/S2 work conflicts cost minutes.
- Fixtures (S3-06): a synthetic mini-package where every R rule fires; one preserve fixture **per P
  class** (P1..P5), each asserted byte-identical after `--apply`; a case-variant fixture; a
  path-rename fixture including a claim file whose sha256 must not change; an idempotence fixture
  (second `--apply` is a no-op); and a V fixture that must make `--apply` exit non-zero.

**Standing stale-name guard, built before and used by the rename (round 1, C1/C9).**
`scripts/rename/check_stale_names.py` is **S3-24**, built on the *pre-rename* tree and
parameterised by the retired name(s) (`--retired aminx --retired prxteinmpnn`), the replacement
package dir, and an allowlist file. It fails if any non-allowlisted occurrence of a retired name
remains, and additionally checks that every **guard name** resolves: root packages named in ruff
`banned-api` keys, `find_spec(...)`/`import_module(...)` literals and `metadata.version(...)`
literals must be a directory under `src/`, a declared dependency, or an allowlisted external. Its red
fixtures run against synthetic trees (dead banned-api key, dead `find_spec` literal, residue), and
its real-tree run before S3-07 is *expected to report* the aminx occurrences (a positive control that
the checker sees the old name). S3-07's gate then requires it clean with `--retired aminx`. S3-09
(after S3-07) uses it for the previous rename's residue (`--retired prxteinmpnn`) and wires CI.
Spike S8 shows the ban case fails silently without it [A9, A19].

**One source of truth for "legitimate remaining occurrence" (round 2, R2-12).** The allowlist is a
**generated artifact of `rename_rules.toml`**, not a second hand-written list. S3-24 defines and
documents the allowlist schema (`[[allow]] glob = ..., retired = ..., reason = ..., class = "P1".."P5"|"external"`)
and consumes it; S3-06 owns `rename_rules.toml` and adds an `--emit-allowlist` mode that renders the
P1-P5 rules (and the path-preserve rows) into that schema. `scripts/rename/allowlist.toml` is that
output, committed, with a header naming its generator. Hand-written rows are only the `external`
class (third-party text that names the old word) and each needs a reason. S3-06 gate adds a
consistency test: every P-class rule appears in the emitted allowlist and every non-`external`
allowlist row maps back to exactly one rule. S3-24 therefore needs no knowledge of the post-rename
tree, only of the schema; its pre-rename fixtures use a hand-written synthetic allowlist.

**Boundary scripts: decide per script, do not rename blindly (C9).** `scripts/check_model_boundary.sh`
guards `model/_inference/`, which does not exist [A28]. S3-09 decides per script among: (a) delete it
(the boundary is gone), (b) revive it against a real boundary, (c) move the rule into ruff
`banned-api` where it is enforced by the linter. Recommended: (c) for any boundary that still
exists (e.g. keep consumers off `molxmpnn.model.*` internals other than the documented `model/__init__`
exports) and (a) for `_inference`. The gate must show the guard firing on a **real, pre-existing
boundary** in a red/green pair (a temporary test module imports a real internal symbol from
outside), not on a fabricated path.

### 4.6 Output-equivalence proof (S3-05, S3-07)

The rename is claimed to be behaviour-preserving; that is a finding and is pre-registered:

- S3-05 captures, on the **pre-rename** tree at a recorded SHA, run-to-run determinism first
  (same fixture twice) and then per-fixture sha256 of `score`/`logits` outputs for >= 4 model
  families (`LEGACY_ALIAS_MAP` families: proteinmpnn, ligandmpnn, solublempnn, membrane) on titanix.
  Tolerance for the post-rename comparison is **fixed by the measured run-to-run difference before
  S3-07 runs** (0.0 if deterministic).
- **Pinned environment (round 2, R2-11) [A41].** The manifest records device model, driver version,
  CUDA/jaxlib/jax/xtrax/equinox versions, `XLA_FLAGS`, the node name and the number of determinism
  reruns (**5** per fixture; variance is the max pairwise difference over the five). S3-07 first
  compares its own environment stamp with the manifest's and **refuses to compare (exit non-zero,
  outcome "not evaluated", not "pass")** if device class, driver, jaxlib, jax, xtrax, equinox or
  `XLA_FLAGS` differ; the remedy is to re-capture on the matching node, never to widen tolerance.
  The tolerance rule is: 0.0 if all five reruns are hash-identical, else
  `tol = 4 x max pairwise max-abs difference` measured on that same device class.
- S3-07's gate re-runs the same fixtures on the renamed tree and compares. Negative control: the
  same comparison with one weight perturbed must report a difference on every fixture. The
  perturbation is fixed in the sidecar: add `+1e-3 x std(w)` to one element of the first-layer
  weight (or, if deterministic, any nonzero), and the pre-registered requirement is that the
  resulting output difference exceeds `max(10 x tol, 1e-6)` on every fixture; if a fixture cannot
  reach that separation the perturbation is enlarged in S3-05 (before the run), not after.
- Resume/chunking: one process and one timeout per fixture; each writes `<fixture>.done.json`
  (input sha256, code SHA, output sha256, tool versions); a rerun skips a fixture only if all hashes
  match and the result records which fixtures were reused and from which run. The output directory
  resolves through the 4.4 layers (explicit `--out` > `MOLXMPNN_GOLDEN_DIR` > `[tool.molxmpnn]
  golden_dir` > `~/.config/molxmpnn/config.toml` > off, failing loudly if none); no default path;
  only the small hash manifest is committed.
- **Independence of S3-05 from S1/S2 ordering.** The recorded SHA need not be S3-07's parent. S3-07's
  gate asserts `git diff --stat <recorded-sha>..<S3-07 parent> -- src/aminx ':!src/aminx/ebm'` is empty
  for the code the fixtures exercise (MPNN, not EBM), so S2 landing in between does not invalidate the
  goldens, and the item needs no edge to S2.
- **xtrax pin ordering (round 1, C7).** The stamp includes the xtrax version and S3-07 refuses to compare across a mismatch (A41). Draft PR #174 (xtrax 0.4.0a11) would change that version between S3-05 and S3-07 if it merged in between, so **S3-05 `depends_on` S2-02** (the confirmation that #174 merged, whose merged SHA is also S2's A0 base): goldens and the S3-07 comparison are both on a11. The alternative (merge #174 after S3-07 and stamp a10) was rejected because S2 and S4 need a11.
- **Measured baselines recorded as data, not asserted in prose.** S3-05 also records the pre-rename
  `uv build` wheel size (bytes, sha256) and sdist size into the manifest; S3-07's size cap is
  `baseline_wheel_bytes * 1.10` read from that manifest (the a24 figure of 499,612 bytes is
  history, the tree has grown since).

### 4.7 Weights (S3-11)

Three options (Q8): **(a) copy** to a new repo `maraxen/molxmpnn` (new commit SHAs; old repo frozen);
**(b) keep** `maraxen/aminx` as the weights repo forever; **(c) rename in place** `maraxen/aminx` ->
`maraxen/molxmpnn` on the Hub. Option (c) mirrors the GitHub decision (a rename keeps the old name
redirecting and, if Hub semantics match, keeps commit SHAs so `HF_REVISION = aa80d0fd...` and every
released wheel's pin keep working with no copy and no sha256 proof). Its costs: it is unverified here
[A34], the redirect dies if anyone recreates `maraxen/aminx` (the same "never recreate" rule as the
GitHub slug applies, and an HF org/user can enforce it only by policy), and the old name cannot be
used for anything else. **Recommended: (c) if the S3-11 probe on a throwaway repo shows redirect +
SHA preservation for both web and `hf_hub_download`; otherwise (a).** (a) is retained as the fallback
because it needs no behaviour of the Hub that we have not measured, and it leaves the old repo
byte-frozen.

Under (a): create `maraxen/molxmpnn`, copy every file at `aa80d0fd...`, prove file-level sha256
equality against the old repo; `HF_REPO_ID`/`HF_REVISION` point at the new repo; the old repo is
untouched so every released `aminx` wheel keeps loading its pinned revision. Under (c): after the
rename, prove that `hf_hub_download(old_id, ..., revision=aa80d0fd...)` for each file returns the
same sha256 as via the new id. Either way per-file work is resumable (stamp per file, hash-checked
reuse, per-file timeout). The staging dir resolves through the 4.4 layers
(`MOLXMPNN_STAGING_DIR`, `[tool.molxmpnn] staging_dir`, `~/.config/molxmpnn/config.toml`), is never
hardcoded, and is never inside `~/.cache/huggingface` or another tool's data.

**Name reservation depends on the answer (round 2, R2-14).** A Hub name cannot be reserved without
creating the repo, and under (c) the rename target must not exist. So S3-27's HF step is
conditional on S3-01's recorded Q8: under **(a)** it creates `maraxen/molxmpnn`; under **(c)** it only
verifies the name is unregistered (HTTP 404) and S3-11 performs the rename (the squat window between
S3-27 and S3-11 is accepted and recorded in the ADR; the PyPI name is the one S3-27 really protects);
under **(b)** it does nothing for HF.

### 4.8 Bathos, praxia, local environment (S3-20)

- Bathos history is keyed by `project_slug` only [A8]: renaming the slug orphans the legacy runs.
  Policy (Q5): new runs use slug `molxmpnn`; `aminx` stays as a read-only legacy slug; documented union
  query `SELECT ... WHERE project_slug IN ('aminx','molxmpnn')`; hub gets `aminx-hub`. Run ids and
  claims are unaffected.
- **Slug used by items before S3-20.** Every bathos-tracked S3 item (S3-04, S3-05, S3-07, S3-11)
  runs while the checkout and slug are still `aminx`, so each sidecar states its slug explicitly
  (`--project-slug aminx` for items that run before S3-20) and the S3-20 union query includes them.
  Nothing relies on a fixed run count: S3-20 records `N_before` (union count at S3-20 start), and its
  gate is `N_after == N_before + 1` (the one new trivial run under `molxmpnn`).
- Checkout rename (Q7) must move together: `.bth.toml` `root` and `remote_root` (`~/projects/aminx` on
  engaging; `/home/solab/bv/catalog-aminx` on titanix), the Claude memory directory
  `~/.claude/projects/-home-marielle-projects-aminx` [A23], praxia workspace/scope registration [A26],
  `~/projects/project_status.toml`, **and the environment the move breaks**: every open git worktree
  stores an absolute `gitdir` (repair with `git worktree repair` from the moved main checkout) and
  the editable install bakes the old path (`.venv/.../__editable__.aminx-*.pth`), so the runbook
  deletes and rebuilds the venv (`uv sync`) on every machine that has one (local, titanix, engaging).
- **Quiescent window.** S3-20 is the only item that moves a directory other work lives in. It must
  not run while any worktree of the repo is open. Round 2 (R2-4) replaced "the last items" by a
  **computed** rule; round 1 of the coherence cycle (C4, C6) narrowed it. (1) **S3-20 is the last
  among the S1/S2/S4 items on the MPNN tree.** Its `depends_on` is S3-02, S3-13, S3-28 plus every
  *leaf* of the set {items of S1, S2, S4 whose `repo` is `molxmpnn` or `aminx`}, where **leaf means no
  direct `depends_on` edge from another member of that set** (edges through items of other repos are
  ignored). As read on 261001 the leaves are S1-16, S1-17, S1-20, S1-22, S2-01, S2-20, S2-25, S4-20,
  S4-21, S4-42 (S2-19 is deleted, assembly row 2; S4-18 and S4-32 are dependencies of S4-19 and S4-40,
  so they are not leaves; S4-42 has no dependents). This list is a snapshot for readability, not a
  contract: **S3-28 computes the leaf set from the assembled backlog** and fails if S3-20's list
  differs, so a newly added or revised S1/S2/S4 item cannot silently break the window. Every other
  member of the set is an ancestor of a leaf, so S3-20 transitively waits for all of them.
  (2) **S5 items on repo `molxmpnn` are deliberately not ordered against S3-20** (withdrawn assembly
  row 7, C4). They run when the hub rolls out; making them descendants of S3-20 would hold the first
  public hub deployment behind shim removal and EBM close-out. Their protection is (3): the runbook's
  first step is `git worktree list` showing only the main checkout, and S3-20 **fails closed** while
  any S5 worktree of the repo is open, i.e. it blocks until those close. This is a scheduling
  statement, not a race: S3-20 does not start an S5 item and no S5 item waits for S3-20. The edges
  stay acyclic: no item depends on S3-20 (checked by S3-28 on the union DAG).

### 4.9 Release and version strategy

- **Two dists, two workflows, two tag namespaces (C4).** `release.yml` fires on every
  `release: published` event [A35]; after the rename it is changed to build and publish **only
  molxmpnn**, and only when the tag matches `v*`. `release.yml` has **two** jobs, `build` and
  `publish` [A35]; the `if: startsWith(github.event.release.tag_name, 'v')` guard goes on **both**
  (a guard only on `build` would skip `publish` via `needs`, but a future edit that drops the `needs`
  would reopen it, and the `pypi` environment approval gate lives on `publish`). The S3-12 gate
  seeds a wrong tag (`aminx-shim-v9`) and asserts neither job runs. The shim has its own `release-aminx-shim.yml`, triggered by tags `aminx-shim-v*`, environment
  `pypi-aminx-shim`, building only `packaging/aminx-shim/`. A molxmpnn release therefore never
  attempts to re-upload the shim, and vice versa. Both publish steps set `skip-existing: false`
  (a duplicate must fail loudly, not be silently skipped) and each workflow carries a pre-publish
  step that asks the PyPI JSON API whether `<name>/<version>` already exists and exits with a clear
  message if so.
- **Version policy.** molxmpnn continues the 0.2.x line (`0.2.0a4` is the first release under the new
  name). The shim is versioned **independently**: `0.2.0a4` first (so it sorts above the last real
  `aminx 0.2.0a3`), bumped only when `gone_modules.toml`, the alias finder, or a dependency bound
  changes (S3-25 is the first planned bump). Its dependency `molxmpnn>=0.2.0a4,<1` will exclude
  molxmpnn 1.0; at that point a shim re-release raising the cap is a required step of molxmpnn's 1.0
  checklist (recorded in S3-22's ADR). Tags use a separate namespace so the two are never confused.
- **What 0.2.0a4 contains: the tag commit is pinned (round 2, R2-2).** Goal 1 promises output
  equivalence with `aminx 0.2.0a3`, so the release must not carry S1's spec flip or S2's EBM removal.
  The tag commit is defined, not "main HEAD": it is the merge of the last of
  {S3-07, S3-08, S3-09, S3-10, S3-11, S3-12}, on a release branch `release/molxmpnn-0.2.0a4` cut from
  that merge (if main has since moved, the branch is cut from the merge commit, not from HEAD).
  **The pin is a content rule, not an ancestry rule (coherence round 2, CH2-03).** Git ancestry
  includes anything merged to main before the S3-07..S3-12 merge, and S3-12 waits on user actions
  (S3-03 <- S3-02 <- S3-27), so other items can land in that window; "nothing orders S1 items after
  S3-13" only covers commits after the merge. S3-13's gate therefore asserts: nothing under
  `src/molxmpnn/` changes except S3-owned paths (`src/molxmpnn/_compat.py` and its callers'
  env/provenance lines, the rename itself, weights constants); `pyproject.toml` and `uv.lock` change
  only as S3 requires; additive files under `tests/`, `scripts/`, `docs/` and `.praxia/` from items
  with no S3-13 ancestor are allowed; `scripts/rename/**`, `packaging/aminx-shim/**`,
  `.github/workflows/release*.yml` and docs/migration are S3-owned. Mechanically:
  `git diff <S3-07 merge>..<tag> --name-only` lists only those, and
  `git diff <S3-07 merge>..<tag> -- src/molxmpnn/run src/molxmpnn/model` is empty. To make this
  satisfiable, assembly row 11 gives S1-05, S1-17 and S4-16 (the items that edit `src/` or the
  dependency set) S3-13 as an ancestor, checked by S3-28. The gate then re-runs the S3-05 fixtures
  **against the built wheel installed from the tagged artifact** (a bathos run, sidecar committed
  before, one process per fixture, same environment-pinning rule as 4.6) and records the run id in the
  release notes. The row-11 edges are required, not optional; the tag-commit rule alone does not make
  them unnecessary. **`gone_modules.toml` and the tag commit:** because 0.2.0a4 is cut
  before S2-18, `src/molxmpnn/ebm` still exists then, so the initial `aminx.ebm` row is conditional
  (`when = "target-absent"`): the finder refuses `aminx.ebm` only if `molxmpnn.ebm` is not importable,
  otherwise it aliases normally. S3-25 later flips the row.
- **Release train (round 1, C14).** S3 owns the molxmpnn version sequence, because `release.yml`
  publishes to PyPI on **every** `v*` tag (S3-12) and PyPI forbids re-upload. The shim has its own
  namespace (`aminx-shim-v*`) and is not in this table.

  | Version | Cut by | Contains | Notes |
  |---|---|---|---|
  | `0.2.0a4` | S3-13 | S3-07..S3-12 only (pinned tag commit above) | first molxmpnn release; output-equivalent to `aminx 0.2.0a3` |
  | next free alpha in tag order (`0.2.0a5`) | S5-52 (the first of the two, assembly row 10) | everything merged to main at its tag, which must include `0.2.0a4`'s release branch | release notes record the version and the item branches contained; the first hub deployment does not wait for the S1 flip chain |
  | the following alpha (`0.2.0a6`) | S1-25 | the previous release plus its own work (carries both sets of work) | S1-25 and S5-52 are ordered by assembly row 10 (S1-25 `depends_on` S5-52), so this is never a race; if the user reverses the order, the edge flips and S5-33 inherits the S1 chain (S3-01 re-open table) |

  Rules: the item cutting a release reads the PyPI JSON for the next unused version (no number is
  hardcoded in an item), states it in the PR, and records the contained branches in the release
  notes. S4-19 manifests record `version` and python `module:symbol` strings; the factory-resolution
  test (S4-19, S5-52 gates) runs **against the tagged wheel**, not the tree, so a string that does
  not resolve in the tag fails before upload. S3-12's duplicate pre-check still fails loudly at
  release time as the last line of defence.
- **Window between S3-02 and S3-03.** After the repo rename the existing `aminx` trusted publisher no
  longer matches [A12], so **no release of either dist is possible until S3-03 completes**. S3-02
  records the old publisher's state (screenshot or `pypi.org/manage/project/aminx/settings/publishing/`
  text in the ADR) before renaming, so a rollback (rename back) is possible if S3-03 stalls.
- **First upload and credentials (round 2, R2-3) [A38].** S3-27's `molxmpnn 0.0.0.dev0` upload
  creates the PyPI project, so by S3-03 the correct action for `molxmpnn` is an **ordinary**
  publisher on the existing project (not a pending one; A33). The first upload cannot use OIDC
  without a pending publisher, so it uses a **project-scoped API token created by the owner** (an
  account-scoped token only if PyPI offers no project scope for a not-yet-existing project, recorded
  in the ADR). Token handling is part of S3-27: the owner holds it, it is used from a local shell
  (never stored in repo secrets or CI). S3-27 records holder and scope; **revocation is S3-03's
  gate** (S3-03 depends on S3-27): after the publisher is registered, the owner deletes the token and records in the ADR the token name and
  deletion date (confirmed on the PyPI account page). No later release uses a token.
- **TestPyPI.** TestPyPI has its own accounts, project names and trusted-publisher registrations
  [A39]; S3-03 registers both there too: a **pending publisher where the project is absent on
  TestPyPI, an ordinary publisher where it exists** (S3-01 probes both names). TestPyPI filenames cannot be re-uploaded, so dry runs use a
  dev-suffixed version derived from the run (`0.2.0a4.dev<github.run_number>` for molxmpnn,
  `aminx 0.2.0a4.dev<run_number>` for the shim); the real versions are never dry-run.
- If the hub later takes PyPI `aminx` (Q3, option P2) its first version must exceed the shim's:
  **`1.0.0a1`**. `pip install -U aminx` then deliberately stops being MPNN; that is why S3-22 gates
  it on a clean consumer scan and a soak period. **Numbers (round 2, R2-9), all overridable by the
  owner in Q14:** P2 requires (i) at least **90 days** since S3-13 *and* since the first shim release
  that prints the CLI stderr banner (the same release, S3-13, unless a later shim re-release
  changes the banner); (ii) PyPI downloads of `aminx` versions >= `0.2.0a4` (shim) of at most
  **50 per week excluding mirrors** averaged over the last 4 weeks of the soak, read from the
  `pypistats`/BigQuery numbers by the owner and pasted into the ADR; (iii) a consumer scan with
  **zero** hits outside the allowlist over roots `~/projects`, `~/repos` and the GitHub code search
  `org/user:maraxen "import aminx"` / `"aminx>="` (external users are the actual risk behind A25 and
  are covered only by (ii)). **P1 (the recommendation) needs none of this**: it is decided at S3-22
  with no soak. Recommended (P1): the shim stays as `aminx` forever
  (a stale 20 KB tombstone is cheap; yanking is hostile), and any hub Python package is `aminx-hub`.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A new repo at `maraxen/aminx` destroys the redirect and breaks three consumers' fresh clones (A10, A13) | Option 1; redirect-alive CI guard (S3-02); slug reclaim only through S3-22 with the owner's explicit decision |
| Silent loss of enforcement: ruff bans, `find_spec` guards, boundary scripts (S8, A19, A20) | `check_stale_names.py` guard-name resolution gate, built before the rename (S3-24) and required clean by S3-07; residue fixed and CI wired in S3-09; boundary scripts decided per script against a real boundary |
| Hub installed as `aminx` makes `find_spec("aminx")` guards true and fail late (S4) | S3-15/S3-18 move guards to `molxmpnn.scoring.score`; shim keeps the name occupied until handover |
| Duplicate module objects break `isinstance`, equinox pytrees, pickles through the shim (S2 negative control) | Alias finder returns the real module; identity tests are a release gate (S3-10) |
| Old calibration pickles and old stores stop loading | `find_class` remap (S3-08, S3 spike); dual-stamp provenance; read accepts both keys |
| Rename merge conflicts with S1/S2 and any open PR (812 files) | Idempotent reusable codemod; S3-07 lands first (Q4); freeze window announced |
| Wheel regressions from keyed build config: py.typed missing or 88.8 MB weights shipped (A21) | S3-07 gate builds both wheels and asserts size cap, `py.typed`, no `model_params/*.eqx.zst` |
| PyPI trusted publishing stops working after the repo rename (A12); no release possible between S3-02 and S3-03 | S3-02 records the old publisher state first (rollback = rename back); S3-03 adds the new identity as an ordinary publisher on existing `aminx` and an ordinary publisher on `molxmpnn` (the project exists after S3-27), plus the TestPyPI registrations (pending where absent, ordinary where present); S3-12 dry-runs with dev-suffixed versions |
| `molxmpnn` squatted on PyPI/HF between decision and publish (a pending publisher does not reserve a name, A33) | S3-27 uploads a real `molxmpnn 0.0.0.dev0` placeholder with a one-time scoped token (revoked in S3-03, A38) right after S3-01, before the repo rename; the HF repo is created there only under Q8 (a), otherwise the name is only verified free (4.7) |
| The placeholder-upload token outlives its purpose | Revocation is S3-03's gate; the ADR records token scope, holder and deletion date |
| 0.2.0a4 ships S1's spec flip or S2's EBM removal because main moved before the tag | Tag-commit pin and path-diff assertion in S3-13 (4.9); fixtures re-run against the tagged wheel |
| `uv lock` regeneration moves third-party versions, so equivalence compares different stacks | Codemod edits only the root lock entry; S3-07 asserts third-party lines unchanged and environment stamps equal [A37, A41] |
| Equivalence compared across different GPU/driver/jaxlib | S3-07 refuses to compare (not evaluated) on a stamp mismatch; 5 determinism reruns; negative control separation >= 10 x tolerance (4.6) |
| Other specs still describe a different DAG (S2-19 publishing `aminx`, S2-18 without an S3 edge) | Assembly checklist in section 2 enforced by S3-28; S3-13 depends on it |
| A molxmpnn release re-uploads the shim and fails, or the shim is silently skipped | Separate workflows and tag namespaces (4.9); no `skip-existing`; a second molxmpnn release is a gate (S3-12) |
| S3 and S2 both publish `aminx 0.2.0a4`, or form a dependency cycle (C3) | Ordering settled in section 2: rename first, S3 sole publisher of `aminx`, S3-25 is the only S3-to-S2 edge and S2-18 depends on S3-07; option E is documented with its version and edge changes |
| Codemod rewrites the compat test or legacy stamp, or renames claim files and breaks bathos anchors (C8) | Explicit P1-P5 preserve classes with one fixture each and a path-rename table; S3-07 gate asserts claim-file sha256 unchanged |
| `AMINX_VERIFY` (or another src-read var) silently stops working after the rename (C7) | Every src-read var goes through `env()` with fallback; classified table; fallback test |
| proteinsmc guard raises, or imports all of molxmpnn, when molxmpnn is absent or broken (A30, A36) | Root-only `find_spec("molxmpnn")`; the submodule is checked at call time inside `make_mpnn_score` under `try/except ImportError`; gate with a stub `molxmpnn` whose `__init__` raises `RuntimeError` (S3-18) |
| Renaming the checkout breaks open worktrees and the editable install (C10) | Quiescent window, `git worktree list` fail-closed, `git worktree repair` and venv rebuild in S3-20 |
| New HF repo drifts from the pinned weights | File-level sha256 equality gate (S3-11); old repo frozen so old wheels cannot regress |
| DeprecationWarning never seen (hidden by default) | CLI banner on stderr; CHANGELOG + README banner; Q6 |
| Local checkout rename orphans Claude memory, bathos remote roots, worktrees (A23, A26) | One runbook, executed with no open worktrees, verified item by item (S3-20) |
| Hub takes PyPI `aminx` while an unknown external user pins `aminx>=0.2` (A25) | Default policy P1 (shim stays); P2 only after S3-22 evidence |
| The stale Sphinx docs at `maraxen.github.io/aminx/` stop resolving on repo rename, with no redirect (A24, A43). The demo was never deployed, so only an old API-docs URL is at stake | S3-21 records where the `gh-pages` site lands (`maraxen.github.io/molxmpnn/`) and decides explicitly between a redirect from a user-site repo and accepting `/aminx/` as dead (Q15); S5-34's gate does not require a stub |

## 6. Interfaces

### Provides

| Contract | What it guarantees | Where | Downstream use |
|---|---|---|---|
| **`molxmpnn` namespace** | Package `molxmpnn` with the API of `aminx 0.2.0a3` (before S1/S2 changes), output-equivalent on the frozen fixtures | S3-07 (code), S3-13 (published) | S1 code items and S2-18 `depends_on` **S3-07**; S2's new EBM repo depends on `molxmpnn>=0.2.0a4` (S3-13) |
| **`S3:molxmpnn-repo`** (symbolic id used by S5 and S4) | Canonical repo `maraxen/molxmpnn` (old slug a permanent redirect), release URLs, HF weights repo (name per Q8 outcome), and the PyPI project names | **S3-02** (repo), **S3-27** (names reserved), **S3-11** (HF weights final), **S3-13** (first release); a consumer needing "the repo exists and releases are published" depends on S3-02 and S3-13, one needing "weights final" also on S3-11 | S5 (release URLs, catalog provenance), S4 |
| **`S3:molxmpnn-manifest-namespace`** | The manifest/id namespace owner. S3 **provides only the root name `molxmpnn`** and the version source (`molxmpnn.__version__` and the PyPI dist `molxmpnn`). Ids such as `molxmpnn/proteinmpnn`, the entry-point group member names and the `molxmpnn/manifest.json` layout are **defined by S4-01** (S4 Consumes this row and fixes its own ids; S4's `MolxmpnnIdentity` reads the root name and the HF repo from S3 and adds the rest); S3 defines no manifest items and no entry-point names | S3-07 (name exists in code), S3-13 (versioned release) | S4 manifest ids, S5 catalog |
| **`S3:aminx-name-freed`** | Preconditions for the first public hub deployment **under P1** (the recommendation): the name-release policy is recorded (S3-01) and the molxmpnn release exists (S3-13, transitively via S5's own edges). The repo-slug rename (S3-02) is a transitive prerequisite of S3-13. **Only S5-55** (anything taking the PyPI name or reclaiming the slug) waits for S3-22/S3-23 | S3-01 (policy), S3-13 (release), S3-22/S3-23 (only for P2 handover) | S5 first deployment needs S3-01 and nothing later in S3's close-out |
| **Rename codemod** | Idempotent `scripts/rename/rename_aminx.py` + rules for rebasing in-flight branches | S3-06 | S1/S2 in-flight branches |
| **`aminx` compat shim** | `aminx>=0.2.0a4,<1` re-exports `molxmpnn`, warns, alias-table-driven; independent shim versioning (4.9) | S3-10, S3-13 | Consumers during the window. The `gone_modules.toml` file is **owned by S3**: S2 does not edit it; S3-25 applies S2's row once S2-18 exists |
| **Name-release policy** | Which `aminx` layers are free, when, and the do-not-recreate guard (repo slug redirect, PyPI policy, skill/CLI/brand, bathos slug) | S3-01 (decision), S3-02 (slug), S3-22/S3-23 (PyPI) | S5: hub repo/site/CLI naming items `depends_on` **S3-01** and **S3-02**; any S5 item that publishes to PyPI under `aminx` must `depends_on` **S3-23** |
| **Env/provenance key contract** | `MOLXMPNN_*` env vars (AMINX_* fallback), `molxmpnn_version` + legacy `aminx_version` store attrs | S3-08 | Anything reading stores (S5 catalog provenance, tev_design) |
| **Bathos slug policy** | New slug `molxmpnn`, legacy `aminx` frozen, union query; no project-id column exists [A8] | S3-20 | S2 (must drop its "rename the slug keeping the project id" wording), S5 (`aminx-hub`) |
| **Stale-name guard** | `check_stale_names.py` + allowlist format, parameterised by retired name | S3-24 (built), S3-09 (CI) | Reusable by S2's new repo and S5 by vendoring the single stdlib file |
| **Checkout-rename quiescent window** | The runbook that renames `~/projects/aminx` and repairs worktrees/venvs; S3-20 is the last S1/S2/S4 MPNN-tree item, with `depends_on` = the computed leaf set of those items on the `molxmpnn`/`aminx` repo (4.8, leaf = no direct edge from another member) | S3-20, S3-28 (recomputes the set) | S1/S2/S4 need no edit, they are ancestors. S5 `molxmpnn` items are intentionally unordered against S3-20 and protected by the fail-closed first step |
| **Release train** | The molxmpnn version sequence and which branches each release contains (4.9): 0.2.0a4 (S3-13), then the next free alpha per release item in tag order | S3-13, 4.9 table | S1-25, S5-52, S4-19 (manifest `version`), S3-25 (shim versions are a separate namespace) |

### Consumes

| Contract | From | Used for |
|---|---|---|
| EBM project name (S2-01 ADR) and the cutover that removes `aminx.ebm` from the tree (**S2-18**) | **S2** (EBM extraction) | S3-25 only: the `gone_modules.toml` forward row and CHANGELOG pointer. Until then the row says "moved, see CHANGELOG". This is the single S3 -> S2 edge; S2-18 depends on S3-07, so it is acyclic |
| Publication of a name `aminx` release only by S3 | **S2** deletes S2-19 (a requirement S3 places on S2, not a dependency) | Avoids two uploads of `aminx 0.2.0a4` (PyPI forbids re-upload of a version) |
| Every item of S1, S2 and S4 whose `repo` is `molxmpnn` or `aminx`; its leaf set (no direct edge from another member) is computed by S3-28 and listed in S3-20: S1-16, S1-17, S1-20, S1-22, S2-01, S2-20, S2-25, S4-20, S4-21, S4-42 as read on 261001 | **S1, S2, S4** | S3-20 quiescent window (it waits for all of them; none of them waits for S3-20) |
| S2-18 `depends_on` S3-07, S2-19 is deleted, S2 has no in-tree shim, S2-03 `depends_on` S3-13 (assembly rows 1-3 and 8) | **S2** (a requirement S3 places on S2, checked by S3-28, not a dependency) | One owner of the shim and of PyPI `aminx 0.2.0a4`; A0 freeze on the renamed, released tree |
| xtrax 0.4.0a11 pin confirmation (**S2-02**, draft PR 174) | **S2** | S3-05 `depends_on` S2-02: golden capture, the S3-05 environment stamp and the S3-07 comparison all run on xtrax a11 (assembly row 9) |
| Fate of the proteinsmc guard: decision **S4-49** (supersede S3-18 with S4-22, or sequence S4-22 after it) | **S4** | S3-18 `depends_on` S4-49. On "supersede" S3-18 is **closed unmerged** and S3-22 treats it as complete; Q13 (dependency group) applies only if S3-18 survives. S4-22 owns the guard and its D6 scan otherwise |
| Public-API removal timetable for the `RunSpecification` facade and other S1 deprecations | **S1** (spec-system unification) | Ensures the shim's window is not confused with S1's own deprecation window; the shim is namespace-wide so it needs no per-name list |
| Hub canonical URL / slug and whether the hub ships a Python package | **S5** (aminx hub) | Deprecation messages, README banner, site redirect (S3-21), and Q3 |

Only the S4-49 decision above is consumed from S4, and nothing from S6 (S4 consumes `S3:molxmpnn-manifest-namespace` above).

## 7. Verification gates per item

All test runs go to titanix or CI (never local, per the compute rules); only `rg`, `git`, JSON/TOML
parsing and the pure-stdlib codemod fixtures are cheap enough to run narrowly local. Every item that
produces a number or pass/fail finding (S3-04, S3-05, S3-07 equality, S3-11) has its `.bth.toml`
sidecar (hypothesis plus `[outcomes]`) committed **before** the run, runs as
`bth run --project-slug <slug> -- uv run --no-sync python3 scripts/...` (never `uv run bth`), and is
verified by its record (`bth compact` then `bth sql`), not by exit code.

| Item | Gate |
|---|---|
| S3-01 | ADR `.praxia/docs/decisions/261001_molxmpnn-name-and-repo-strategy.md` records an answer for **every** of Q1-Q14 (any not put to the user is written as "accepted at the recommendation" with the date, never left blank) and the availability probes: PyPI (`molxmpnn`, `aminx`), TestPyPI (both names, A39), GitHub, HF, npm, and the Read the Docs slug `molxmpnn` (A40). The ADR ends with a re-open table: for each Q the named items a different answer regenerates (Q4=E: every `0.2.0a4` string becomes `0.2.0a5`, S3-07 gains S2-18/S2-19, S3-25 is dropped, S1's four S3-07 edges flip; Q6: S3-10 warning class; Q8: S3-11 and S3-27 HF step; Q9: S3-09; Q10: S3-06/S3-07 R rule and S5's use of the sampler path; Q11 and Q13: S3-18; Q12: S3-15 scope; Q14: S3-22/S3-23). Every other item depends on S3-01 transitively, so none starts before the answers exist |
| S3-27 | PyPI JSON for `molxmpnn` returns 200 with release `0.0.0.dev0`; the upload credential type, scope and holder are recorded in the ADR (A38); HF step by Q8: under (a) `huggingface.co/maraxen/molxmpnn` exists, under (c) the name returns 404 and the squat-window acceptance is recorded, under (b) no HF action; GitHub `maraxen/molxmpnn` returns 404 (free) or is the existing repo, recorded in the ADR |
| S3-02 | Old publisher state for PyPI `aminx` recorded in the ADR before the rename; after it, `git ls-remote https://github.com/maraxen/aminx.git` returns the molxmpnn refs; each consumer's pinned submodule SHA is `git fetch`-able through the old URL; repo variables, environments (`pypi`) and Pages settings are **verified preserved** (listed before and after, no re-creation unless a diff shows one missing); redirect-alive check wired |
| S3-03 | PyPI `aminx` lists a trusted publisher bound to the renamed repo + `release-aminx-shim.yml` + env `pypi-aminx-shim`; PyPI `molxmpnn` (created by S3-27) lists an **ordinary** publisher for `release.yml` + env `pypi`; TestPyPI: a pending publisher where S3-01's probe found the project absent, an ordinary one where it exists, for both names; no `MOLXMPNN_MODELS_RELEASE_TAG` variable is required (S5 verified none are set; S5-34 retires them); **the S3-27 upload token is revoked** (deletion date and token name recorded in the ADR) |
| S3-04 | Sidecar pre-registered; scan covers mpnn_ext, asr and hautespout **own code and the `vendor/aminx` or `aminx` submodule consumers' import sites**, excluding vendored third-party copies (identified by `.gitmodules` entries and `vendor/` directories, listed in the output); tev_design is excluded because it pins a prebuilt 0.1.0a24 wheel the rename cannot affect (justification recorded); `tests/compat/old_names/` carries the registered `consumer_shim` marker and `pytest.importorskip("aminx")`, plus `tests/compat/path_map.toml`; `pytest tests/compat/old_names` passes against the tree at the pinned pre-rename SHA with zero skips; negative control: an injected nonexistent path fails the test; a second control runs the suite in a venv where `aminx` is absent and asserts every old_names test is *skipped*, not failed |
| S3-05 | Sidecar pre-registered with the fixture list, tolerance rule, the baseline wheel/sdist sizes, the negative-control perturbation (size and the `>= max(10 x tol, 1e-6)` separation requirement) and the **environment stamp** (device model, driver, jaxlib, jax, xtrax, equinox, `XLA_FLAGS`, node); determinism measured over 5 reruns per fixture on the stamped node; manifest committed; goldens dir resolves through the layered resolver (no default); resume test: kill mid-run, rerun, only unfinished fixtures recompute and reuse is recorded with hashes |
| S3-24 | Built on the pre-rename tree: red fixtures fail (dead banned-api key, dead `find_spec` literal, residual retired name) and a green fixture passes, using a hand-written synthetic allowlist in the documented schema (`[[allow]] glob, retired, reason, class`); real-tree run with `--retired aminx` reports occurrences (positive control) and with `--retired prxteinmpnn` reports the known residue count; `pytest tests/rename/test_check_stale_names.py -q` on titanix |
| S3-06 | `pytest tests/rename -q` on titanix: every R rule fires on the positive fixture; each P1..P5 fixture byte-identical after `--apply`; case-variant and path-rename fixtures pass (claim file sha256 unchanged); a `uv.lock` fixture where only the root entry changes and a third-party version line is byte-identical; a V fixture makes `--apply` exit non-zero; second `--apply` is a no-op (idempotent); `--emit-allowlist` output validates against S3-24's schema and a consistency test shows every P-class rule appears in it and every non-`external` row maps back to exactly one rule; `tests/compat/new_names/**` is class `G` (`--apply` leaves it byte-identical); `gen_compat_new_names.py` fixtures with a synthetic `path_map.toml` and synthetic old_names modules: `--write` yields the expected new_names, a second `--write` is a no-op, `--check` fails on a hand-edited output, and an unmapped old path or an unused map row exits non-zero |
| S3-07 | **Split by where it can run (round 2, R2-13).** *CI, clean checkout, no GPU:* (a) `check_stale_names.py --check --retired aminx` clean against the generated allowlist; (b) ruff red check: a file importing a banned `molxmpnn.inference.decode` from the Potts package fails `ruff check`; (c) `uv build` + wheel inspection: `molxmpnn/py.typed` present, no `model_params/*.eqx.zst`, wheel <= `baseline_wheel_bytes * 1.10` from the S3-05 manifest; (d) `git lfs ls-files -l` oid set identical to the pre-rename set; (g) `.bth/claims/*` sha256 unchanged, `tests/compat/old_names/**` and `path_map.toml` byte-unchanged, `tests/compat/new_names/` written by `scripts/rename/gen_compat_new_names.py --write` and `--check` clean; (h) `uv lock --check` clean and a lock diff restricted to the root package entry (script asserts every third-party `version` and hash line unchanged, A37). *Titanix, bathos-tracked, sidecar committed first (never on CI runners):* (e) non-heavy suite (`-m 'not parity_heavy and not slow'`) run twice in the one record, `before` at the S3-07 parent and `after` at the PR head, same node and stamp, both with zero failures; skips compared **by (file, reason) set, not by count** (4.5, A44): every `tests/compat/old_names` module is skipped in `after` with reason `could not import 'aminx'`, `after` minus those records is a subset of `before`, and `tests/compat/new_names` passes with zero skips (negative controls: a seeded new skip elsewhere fails, an old_names module without `importorskip` fails); (f) the S3-05 environment stamp equals this run's (else the run reports "not evaluated" and the PR cannot merge), fixtures bit-equal within the pre-registered tolerance, negative control fires with the stated separation, `git diff <golden-sha>..<parent> -- src/aminx ':!src/aminx/ebm'` is empty, and a store written by the renamed tree still carries `aminx_version`. **Merge condition: CI green and the titanix bathos run id, with an evaluated outcome (`bth compact` then `bth sql`), cited in the PR** |
| S3-08 | Tests: env fallback + warning once for all four src-read vars including `AMINX_VERIFY` (old name alone is honoured, both set -> new wins); `docs/migration/aminx_to_molxmpnn_env.md` lists every `AMINX_*` found by a repo scan with class and fallback (a test fails if a scanned name is missing from the table); cache-dir `source` reports the deciding layer and a malformed config fails loudly; the stdlib copy in `scripts/rename/_resolve_dir.py` agrees with `_compat.resolve_dir` on a layer table; store carries both version attrs and old readers/new readers both pass; synthetic old-path pickle loads via remap and fails without it |
| S3-09 | Using S3-24: zero `prxteinmpnn` outside the allowlist; for each of `scripts/check_model_boundary.sh` and any other boundary script a recorded decision (delete / revive / move to ruff banned-api); the surviving guard fires in a red/green pair on a **real existing boundary**; the checker runs in CI |
| S3-10 | Shim tests on titanix: identity of module/class/isinstance across names, pickle round trip, exactly one `DeprecationWarning`, `-W error::DeprecationWarning` raises, gone-module error text including the `when = "target-absent"` behaviour (with `molxmpnn.ebm` importable `aminx.ebm` aliases; with it hidden the refusal message appears), CLI forwards exit code and prints the banner, extras present in wheel metadata, wheel < 100 KB, `python -I -c "import aminx.host.runner"` works with only the shim and molxmpnn installed; `pytest -m consumer_shim tests/compat/old_names` with the shim installed passes with **zero skips** |
| S3-11 | First, the HF rename probe on a throwaway repo (redirect, SHA preservation, behaviour after recreating the old name) recorded as a bathos-tracked run with its sidecar committed first; then per the Q8 answer: copy path (sidecar hypothesis: every file sha256-identical across repos; per-file stamps; resumable) or rename path (old id and new id return identical sha256 per file at `aa80d0fd...`); `bth sql` record shows outcome evaluated; molxmpnn pinned load of each checkpoint family succeeds; old wheel's pinned load still succeeds (`pip download aminx==0.2.0a3` in a scratch venv) |
| S3-12 | `release.yml` publishes only molxmpnn and ignores `aminx-shim-v*` tags; the tag guard is on **both** `build` and `publish`, and a seeded wrong tag asserts neither job runs and the `pypi` environment approval is not requested (A35); `release-aminx-shim.yml` publishes only the shim; `workflow_dispatch` dry run publishes both dists to TestPyPI with dev-suffixed versions; **a second dispatch run with a new dev suffix also succeeds** and a deliberate duplicate version fails loudly; wheel gates fail on a seeded violation; two environments exist |
| S3-13 | **Pre-publish:** S3-28 passes (so no other spec publishes `aminx 0.2.0a4`: S2-19 is gone, the PyPI JSON for `aminx` has no `0.2.0a4` and no item outside S3 is titled as an `aminx` release; and S2-03 depends on S3-13), and the tag commit is the S3-07..S3-12 merge on `release/molxmpnn-0.2.0a4` with `git diff <S3-07 merge>..<tag> --name-only` obeying the 4.9 content rule (nothing under `src/molxmpnn/` outside S3-owned paths, `pyproject.toml`/`uv.lock` only as S3 requires, only additive `tests/`/`scripts/`/`docs/`/`.praxia/` files from items with no S3-13 ancestor) and empty for `src/molxmpnn/run` and `src/molxmpnn/model`, and S1-05, S1-17 and S4-16 have S3-13 as an ancestor (assembly row 11). **Post-publish:** in a fresh venv `pip install --pre molxmpnn==0.2.0a4` and `aminx==0.2.0a4`; both import; `pytest -m consumer_shim tests/compat` passes through both names with zero skips; the S3-05 fixtures re-run against the installed tagged wheel (bathos run, sidecar first, same environment-stamp rule) equal the golden manifest and the run id is in the release notes; PyPI JSON lists both versions; a following molxmpnn patch tag in TestPyPI/dry mode does not touch the shim |
| S3-14..S3-17 | Per consumer: `uv lock --check` clean after the dependency/source/submodule change; the consumer's own narrow smoke on its remote passes; `git grep -n -w aminx` shows only intentional historical mentions. S3-15 additionally runs the narrow test on the *vendored* `proteinsmc/scoring/mpnn.py` of asr itself |
| S3-18 | Closed unmerged, without gate, if S4-49 answers "supersede". Otherwise new proteinsmc tests: (1) **molxmpnn absent**: `import proteinsmc.scoring.mpnn` succeeds without error and `make_mpnn_score` raises the friendly ImportError; (2) **molxmpnn present but broken**: a stub `molxmpnn` package on `sys.path` whose `__init__` raises `RuntimeError` still lets `import proteinsmc.scoring.mpnn` succeed (guard is root-only `find_spec("molxmpnn")`, no parent import, A36), the flag is true, and the failure surfaces at the `make_mpnn_score` call as an ImportError-wrapped message; (3) a stub hub package `aminx` on `sys.path` leaves the flag false; (4) with real molxmpnn installed through the mechanism fixed by Q13 (default: a non-default `[dependency-groups]` entry used only by this test on titanix) the flag is true and one structure is scored end to end; the TYPE_CHECKING and call-site import paths are rewritten to molxmpnn's layout (`aminx.types` / `utils.types` -> `molxmpnn.types`, `make_score_sequence` -> `molxmpnn.scoring.score` public name); the `pyproject.toml` diff leaves `[project].dependencies` and `[project.optional-dependencies]` unchanged (D6) and adds at most the one non-default dependency group named in Q13 |
| S3-19 | Skill frontmatter parses; `using-molxmpnn` has no `import aminx` example except in the migration note; `using-aminx` is a pointer under 20 lines |
| S3-20 | S3-28 passes and its recomputed leaf set equals this item's non-S3 `depends_on`; the first runbook step `git worktree list` shows only the main checkout (this also covers any open S5 worktree: S3-20 blocks until they close); runbook checklist ticked with evidence per line; `N_before` recorded then `bth sql` union query over `('aminx','molxmpnn')` returns `N_before + 1` after one new trivial run under `molxmpnn`; `git worktree repair` run and `git status` clean on each former worktree; venv rebuilt and `python -c "import molxmpnn"` from the new path on local, titanix and engaging; Claude memory dir present at the new path |
| S3-25 | `aminx.ebm` row flipped to forward-if-installed; shim test: with `ebmx` absent the install message is raised, with a stub `ebmx` present `import aminx.ebm.model` resolves to it; shim released as a new shim-only version via its own tag; molxmpnn is untouched |
| S3-28 | Pure stdlib script `scripts/rename/check_assembly.py` (runs locally, no JAX): parses every `.praxia/docs/specs/261001_*.md` backlog block; asserts assembly rows 1, 2, 3, 6, 8, 9 (S3-05 edge), 10 (S5-52 then S1-25 by default, either direction passes) and 11 (S1-05, S1-17, S4-16 have S3-13 as an ancestor) of section 2 (row 7 is withdrawn), no duplicate ids, every `depends_on` resolves, and the union DAG is acyclic; **computes** the row-6 leaf set (no direct edge from another S1/S2/S4 member on repo `molxmpnn`/`aminx`) rather than reading a list; prints rows 4, 5 and the S4-16 wording for a human. Red fixtures (synthetic spec dir): S2-18 without S3-07, an S2-19 present or titled "Release aminx", a leaf missing from S3-20, an S2-03 without S3-13, an S3-05 without S2-02, unordered S1-25 and S5-52 (edge removed), an S1-05/S1-17/S4-16 without S3-13 as ancestor, a cycle; green fixture (S1-25 depends on S5-52, row-11 edges present) passes; real run exits 0 only after S2 is revised, and its output is attached to S3-13 and S3-20 |
| S3-21 | README/CHANGELOG/RTD/Pages checked (the legacy gh-pages Sphinx URL after S3-02 is recorded, and the A43 probe plus the Q15 outcome are in the ADR): no broken link (`rg` for old URLs only in the allowlist); RTD redirect mechanism proven first on a throwaway project (A40), then the old project redirects; README install extras match `pyproject` (`cuda12`, `cpu`: the current README says `cuda`, which does not exist) |
| S3-22 | ADR records the consumer scan (roots `~/projects`, `~/repos` and GitHub code search over `maraxen`; zero hits outside the allowlist), elapsed days since S3-13, and the decision **P1 or P2**. **P1 needs no soak** and no S5 deployment item waits for it: S5-33 needs only S3-01, S5-55 waits for this item (`S3:aminx-name-freed`, 6). If P2 is chosen the ADR records "P2 pending" and S3-23 enforces the thresholds (4.9: >= 90 days, <= 50 weekly downloads of `aminx` >= 0.2.0a4 averaged over the last 4 weeks, owner-read) |
| S3-23 | Only if P2: the 4.9 thresholds are met and pasted into the ADR (days since S3-13 and since the banner release, weekly download figures, scan result); `aminx==0.2.0a4` still installable; hub wheel version >= `1.0.0a1`; trusted publisher moved; shim yank policy recorded (no yank) |

## 8. Open questions for the user

| # | Question | Recommendation |
|---|---|---|
| Q1 | Confirm the final name `molxmpnn`. PyPI `molxmpnn` and `mpnnx` both returned 404 on 2026-10-01 [A2]; GitHub `maraxen/molxmpnn`, HF `maraxen/molxmpnn` and npm were not checkable from this sandbox. | Yes, `molxmpnn`. Do S3-01's availability probe and S3-27's reservation (a real `0.0.0.dev0` placeholder upload; a pending publisher would not reserve the name, A33) the same day to avoid squatting |
| Q2 | Repo strategy: option 1 (rename in place, hub at another slug) vs 2 (new hub repo at `maraxen/aminx`, destroying the redirect) vs 3 (keep aminx for hub, mirror MPNN out). And the hub slug under option 1: `maraxen/aminx-hub` vs a new GitHub org. | Option 1, hub `maraxen/aminx-hub`. Never reclaim `maraxen/aminx` unless S3-22 shows zero consumers and you accept permanent breakage of unknown external links. This is the one place D2 is satisfied at brand level, not slug level |
| Q3 | After the window, does the hub take PyPI `aminx` (P2, first version `1.0.0a1`) or does the shim stay forever (P1)? | P1: the shim stays as `aminx`; hub Python tooling, if any, is `aminx-hub`. Decide with S5's answer on whether the hub ships a Python package at all |
| Q4 | Order against S1 and S2: **R (rename first)** or **E (S2's EBM cutover and release first)**? Consequences are tabulated in section 2: who publishes `aminx 0.2.0a4` (R: S3 only; E: S2-19, and S3 uses 0.2.0a5), what the shim does with `aminx.ebm` (R: refuse, then forward via S3-25; E: refuse with pointer), who owns `gone_modules.toml` (S3 in both), and which edges flip (E: S3-07 gains S2-18/S2-19, S1's four S3-07 edges flip). S1 is already written for R; S2 recommends E and must be revised either way (drop S2-19's `aminx` release under R). | **R.** It is mechanical and codemod-reusable, S1 already assumes it, and it keeps exactly one publisher of PyPI `aminx`. S2's `filter-repo` runs on `src/molxmpnn/ebm`, with `src/aminx/ebm` as history paths, and **S2-03 must follow S3-13** (assembly row 8); the order is not free, because the A0 freeze window forbids S3-07..S3-11 touching the frozen paths after the GPU evidence exists. Pick E only if you want the EBM removed from the rename's 812-file blast radius badly enough to hold the rename behind the whole EBM extraction |
| Q5 | Bathos slug: new runs under `molxmpnn` with `aminx` frozen, or keep `aminx` for the MPNN repo forever? | New runs under `molxmpnn`; hub gets `aminx-hub`. Keeping `aminx` would later collide with the hub's identity; slug rename would orphan 144 runs [A8] |
| Q6 | Shim warning class: `DeprecationWarning` (as requested, hidden by default outside `__main__`) or `FutureWarning` (always shown)? | Keep `DeprecationWarning` plus the always-visible CLI banner and README/CHANGELOG notice |
| Q7 | Rename the local checkout `~/projects/aminx` to `~/projects/molxmpnn`? | Yes, once, via the S3-20 runbook with no open worktrees; otherwise `~/projects/aminx` keeps meaning "molxmpnn" while the hub needs that name |
| Q8 | HF weights: (a) new repo `maraxen/molxmpnn` (copy, new SHAs, old frozen), (b) keep `maraxen/aminx` as the weights repo forever, or (c) rename `maraxen/aminx` to `maraxen/molxmpnn` in place (redirect, SHAs kept if the Hub behaves as assumed, A34)? | (c) if the S3-11 probe confirms redirect and SHA preservation, mirroring the GitHub choice at much lower cost (no copy, no sha256 proof, pins in released wheels keep working); else (a). Not (b): it puts the hub's brand name on MPNN weights. Under (c) and (a) the old name is never recreated |
| Q9 | Include the previous rename's residue (64 files naming `prxteinmpnn`, dead guards) in this spec? | Yes (S3-09). Fixing the same defect class in one pass is cheaper than rediscovering it |
| Q10 | Rename `browser/aminx-sampler` and `aminx_sampler.mjs` to `molxmpnn-sampler` in the codemod, given S5 will vendor it as an executor? | Yes; S5 consumes the new names (the sampler is MPNN-specific) |
| Q11 | proteinsmc guard: a root-only `find_spec` guard (optional, no dependency edge) or an entry-point provider? | The root-only form: `find_spec("molxmpnn") is not None`, with the submodule resolved at call time inside `make_mpnn_score` under `try/except ImportError`. Round 2 withdrew the earlier dotted form `find_spec("molxmpnn.scoring.score")`: it raises when the root is absent (A30) and, when the root is present, imports molxmpnn's heavy `__init__` (A36), so a version-skewed install would break `import proteinsmc.scoring.mpnn`. S4-22's later fix, if any, refines this. It matches the existing precedent and D6. An entry-point provider is a bigger proteinsmc change that belongs to the ecosystem-partition work |
| Q12 | asr carries a diverged plain-directory copy of proteinsmc (`vendor/proteinsmc`, not a submodule, A27). Is that vendored fork still wanted, or should asr depend on proteinsmc normally? | Out of S3's scope. S3-15 patches the copy in place (names and import paths) and tests it; file the fork itself as a separate D6 debt item for the ecosystem-partition work |
| Q13 | How does molxmpnn get into proteinsmc's test environment for the end-to-end score test (S3-18) without a runtime dependency (D6)? (a) a non-default `[dependency-groups]` entry (dev-only, used by one titanix test), (b) install out-of-band on titanix and record the exact command in the bathos sidecar | Applies only if S3-18 survives S4-49 (on "supersede" S3-18 is closed unmerged and S4-22 supplies its own test mechanism). (a). alphex `260814_alphabet-contract.md:283-289` (D4) treats a dev-only test dependency as not a coupling edge, and `:278` lists proteinsmc, aminx and proxide as remaining dev-only. The gate then reads: no change to `[project].dependencies` or `[project.optional-dependencies]`, one non-default group added. (b) is acceptable but leaves the test unreproducible from the repo |
| Q14 | Soak and download thresholds before the hub may take PyPI `aminx` (P2): the defaults in 4.9 are >= 90 days since S3-13, <= 50 weekly downloads of `aminx` >= 0.2.0a4 averaged over the last 4 weeks (mirrors excluded), zero consumer-scan hits across `~/projects`, `~/repos` and GitHub code search over `maraxen`. Accept, tighten or loosen? | Accept as written; but choose P1, which needs none of them. Under P1, S5-33 waits only for S3-01 (the recorded policy; S3-13 arrives transitively via S5-53/S5-52), not for S3-22 and not for 90 days. Only S5-55 waits for S3-22/S3-23 |
| Q15 | The legacy Pages URL `maraxen.github.io/aminx/` (stale Sphinx build from `gh-pages`, A24) stops resolving when S3-02 renames the repo. Accept it as dead, or keep a redirect from a user-site repo (`maraxen/maraxen.github.io`) or from the hub repo? | **Accept as dead** unless S3-21's throwaway probe (A43) shows a cheap redirect that does not recreate `maraxen/aminx` (option 1 forbids that). The content is stale API docs with no known inbound links; the new Sphinx location (`maraxen.github.io/molxmpnn/`, or RTD) is recorded in the README and CHANGELOG. S5-34's gate is restated by S5 to "S3-21 recorded the docs URL decision", not "the stub resolves" |

## 9. Backlog items

Items use the target repo names; the `molxmpnn` repo is the existing `maraxen/aminx` checkout until
S3-02. Order of work is expressed only through `depends_on`.

```toml
[[item]]
id = "S3-01"
title = "Decide final name, repo strategy, PyPI policy, ordering against S1/S2 (Q4), bathos slug, checkout rename, HF weights option and the remaining open questions Q1-Q14; record ADR with availability probes and a re-open table"
repo = "molxmpnn"
size = "S"
depends_on = []
gate = "ADR .praxia/docs/decisions/261001_molxmpnn-name-and-repo-strategy.md exists with an answer (or dated accepted-at-recommendation) for every Q1-Q14, PyPI/TestPyPI/GitHub/HF/npm/RTD availability results, and a re-open table naming the items each different answer regenerates"
user_decision = true

[[item]]
id = "S3-27"
title = "Reserve names: upload a real molxmpnn 0.0.0.dev0 placeholder to PyPI with a one-time scoped token (holder and scope recorded); HF repo created only under Q8 (a), otherwise the name is only verified free"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-01"]
gate = "PyPI JSON for molxmpnn returns 200 with release 0.0.0.dev0 and the token scope and holder are in the ADR; the HF step matches the recorded Q8 (a: repo exists, c: name returns 404 and squat window accepted, b: none)"
user_decision = true

[[item]]
id = "S3-02"
title = "Rename GitHub repo maraxen/aminx to maraxen/molxmpnn after recording the old PyPI publisher state; probe redirect and pinned submodule SHAs, verify repo variables/environments/Pages preserved, add redirect-alive guard and do-not-recreate note"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-01", "S3-27"]
gate = "old publisher state recorded in the ADR; git ls-remote on the old URL resolves and the pinned SHAs of asr, mpnn_ext and hautespout are fetchable through it; repo variables and the pypi environment verified present before and after the rename"
user_decision = true

[[item]]
id = "S3-03"
title = "Register trusted publishers: ordinary publishers on existing PyPI aminx (release-aminx-shim.yml, env pypi-aminx-shim) and on molxmpnn (created by S3-27; release.yml, env pypi); TestPyPI pending where the project is absent and ordinary where present; revoke the S3-27 upload token"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-02", "S3-27"]
gate = "PyPI and TestPyPI settings pages list both publishers bound to the renamed repo and the named workflows and environments (no MOLXMPNN_MODELS_RELEASE_TAG repo variable is required: S5 verified zero are set and retires them with pages.yml in S5-34); the S3-27 token deletion date is recorded in the ADR"
user_decision = true

[[item]]
id = "S3-04"
title = "Consumer import-surface inventory (AST scan of mpnn_ext, asr and hautespout own code and submodule import sites, vendored third-party copies identified and excluded; tev_design excluded with justification) plus contract test under tests/compat/old_names (registered consumer_shim marker, importorskip(aminx), path_map.toml), pre-registered with bathos sidecar and negative controls"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-01"]
gate = "tests/compat/old_names passes at the pre-rename SHA on titanix with zero skips, fails when a nonexistent path is injected, and is skipped (not failed) in a venv without aminx; consumer_shim marker registered; bathos record evaluated"
user_decision = false

[[item]]
id = "S3-05"
title = "Capture pre-rename golden outputs (on xtrax 0.4.0a11, after S2-02): environment stamp (device, driver, jaxlib, XLA flags), 5-rerun determinism then per-fixture output hashes for four model families, negative-control perturbation spec, baseline wheel/sdist sizes, chunked per fixture and resumable, layered-resolver output dir, sidecar pre-registered"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-01", "S2-02"]
gate = "bathos record shows environment stamp (xtrax 0.4.0a11), 5-rerun determinism and golden manifest committed; kill-and-rerun test recomputes only unfinished fixtures and records reuse with hashes"
user_decision = false

[[item]]
id = "S3-24"
title = "Build the stale-name checker check_stale_names.py on the pre-rename tree, parameterised by retired name, with guard-name resolution (banned-api keys, find_spec/import_module/metadata.version literals), allowlist format and red/green fixtures"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-01"]
gate = "pytest tests/rename/test_check_stale_names.py -q on titanix: dead banned-api key, dead find_spec literal and residue fixtures fail, green fixture passes against a synthetic allowlist in the documented schema; real tree with --retired aminx reports occurrences"
user_decision = false

[[item]]
id = "S3-06"
title = "Rename codemod with rewrite/preserve(P1-P5)/generated(G)/review classes, explicit path-rename table, uv.lock root-entry-only step, case variants, --emit-allowlist (generated allowlist for S3-24 schema), dry-run default, idempotent, plus scripts/rename/gen_compat_new_names.py (write/check) for tests/compat/new_names, with positive, per-class preserve, path, lock, generator and review fixtures"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-01", "S3-24"]
gate = "pytest tests/rename -q on titanix: all R rules fire, every P fixture byte-identical, claim file sha256 unchanged, uv.lock fixture changes only the root entry, review fixture exits non-zero, second apply is a no-op, emitted allowlist validates against the S3-24 schema and matches the P rules one to one, tests/compat/new_names is class G and untouched by --apply, gen_compat_new_names.py writes the expected output from a synthetic path map, is idempotent, fails --check on drift and fails on an unmapped path or unused map row"
user_decision = false

[[item]]
id = "S3-07"
title = "Atomic rename PR: git mv src/aminx to src/molxmpnn, apply codemod (imports, runtime strings, pyproject keys, uv.lock root entry, CI, tests, scripts, browser sampler names), keep dated docs, tests/compat/old_names, path_map.toml, claims and legacy provenance names untouched; write tests/compat/new_names with gen_compat_new_names.py"
repo = "molxmpnn"
size = "L"
depends_on = ["S3-06", "S3-04", "S3-05", "S3-24"]
gate = "CI: stale-name check clean with --retired aminx, ruff banned-api red check fires, wheel has py.typed and no model_params weights within baseline+10%, LFS oid set unchanged, claims, tests/compat/old_names and path_map.toml byte-unchanged, gen_compat_new_names.py --check clean, uv lock --check clean with third-party lines unchanged. Titanix (bathos run id with evaluated outcome cited in the PR): non-heavy suite run before (S3-07 parent) and after (PR head) on the same node and stamp, both with zero failures, skips compared as (file, reason) sets not counts: every old_names module skipped after with reason could not import 'aminx', no other new skip, tests/compat/new_names passes with zero skips; environment stamp equals S3-05 else not-evaluated, golden fixtures equal within the pre-registered tolerance and the negative control fires with the stated separation"
user_decision = false

[[item]]
id = "S3-08"
title = "Compat layer in molxmpnn: env() helper for all four src-read AMINX_* vars with warned fallback, classified env table, shared layered dir resolver with source reporting, dual molxmpnn_version/aminx_version store stamp and resolve_version, legacy pickle find_class remap"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-07"]
gate = "pytest on titanix: env fallback warns once incl. AMINX_VERIFY, env table covers every scanned AMINX_ name, cache_dir_source reports deciding layer and malformed config fails loudly, stores carry both attrs, old-path pickle loads via remap and fails without it"
user_decision = false

[[item]]
id = "S3-09"
title = "Fix the previous rename's residue with the S3-24 checker (prxteinmpnn in 64 files), decide per boundary script to delete, revive or move to ruff banned-api, and wire the checker into CI"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-07", "S3-24"]
gate = "zero prxteinmpnn outside the allowlist; a recorded decision per boundary script; the surviving guard fires in a red/green pair on a real existing boundary; checker runs in CI"
user_decision = false

[[item]]
id = "S3-10"
title = "Build the aminx shim distribution (alias finder, forwarded extras and CLI, DeprecationWarning, S3-owned gone_modules.toml) with identity and pickle tests"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-07", "S3-08"]
gate = "shim tests on titanix: module/class identity and isinstance across names, one warning, gone-module error text, CLI exit code forwarded, extras in metadata, wheel under 100 KB, pytest -m consumer_shim tests/compat/old_names with the shim installed passes with zero skips"
user_decision = false

[[item]]
id = "S3-11"
title = "Weights cutover: probe HF rename semantics on a throwaway repo, then per the Q8 answer rename in place or copy HF maraxen/aminx@aa80d0fd with per-file sha256 equality (resumable, pre-registered, layered staging dir), repoint HF_REPO_ID/HF_REVISION, never recreate the old name"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-07", "S3-08"]
gate = "bathos records for the probe and the per-file equality are evaluated; pinned load of every checkpoint family succeeds from the new id and the released aminx 0.2.0a3 pinned load still succeeds"
user_decision = true

[[item]]
id = "S3-12"
title = "Release plumbing: release.yml (tag guard on both build and publish jobs) publishes only molxmpnn on v* tags, release-aminx-shim.yml publishes only the shim on aminx-shim-v* tags, two environments, pre-publish duplicate check, wheel size/py.typed/no-weights gates, TestPyPI dry runs with dev-suffixed versions"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-03", "S3-10"]
gate = "two consecutive workflow_dispatch dry runs both succeed on TestPyPI, a seeded duplicate version fails loudly, a seeded aminx-shim-v tag runs neither build nor publish of release.yml, a molxmpnn release run never touches the shim, and wheel gates fail on a seeded violation"
user_decision = false

[[item]]
id = "S3-13"
title = "Publish molxmpnn 0.2.0a4 and aminx 0.2.0a4 shim from a pinned tag commit (S3-07..S3-12 only) with CHANGELOG migration notes; re-run the golden fixtures against the tagged wheel; sole publisher of aminx 0.2.0a4 and first row of the release train"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-08", "S3-09", "S3-10", "S3-11", "S3-12", "S3-28"]
gate = "S3-28 passes and PyPI aminx has no 0.2.0a4 before upload; tag diff vs the S3-07 merge obeys the 4.9 content rule (nothing under src/molxmpnn/ outside S3-owned paths, empty for run/ and model/, only additive tests/scripts/docs/.praxia files from items with no S3-13 ancestor) and S1-05, S1-17, S4-16 descend from S3-13; fresh venv installs both from PyPI, both import, consumer_shim tests pass through both names with zero skips, golden fixtures against the tagged wheel equal the manifest (bathos run id in release notes), PyPI JSON lists both versions"
user_decision = true

[[item]]
id = "S3-14"
title = "Migrate mpnn_ext: submodule URL and path to molxmpnn, workspace member and source, imports in scripts and sidecars"
repo = "mpnn_ext"
size = "S"
depends_on = ["S3-13"]
gate = "uv lock --check clean and an import smoke on the remote passes; git grep -w aminx shows only historical mentions"
user_decision = false

[[item]]
id = "S3-15"
title = "Migrate asr: submodule and workspace member to molxmpnn, dependency name and imports, patch asr's own vendored proteinsmc copy directly (guard, names, import paths), regenerate uv.lock"
repo = "asr"
size = "M"
depends_on = ["S3-13"]
gate = "uv lock --check clean and the narrow asr tests importing asr.mpnn_loss plus a test of the vendored scoring/mpnn.py pass on the remote; vendored guard no longer names aminx"
user_decision = false

[[item]]
id = "S3-16"
title = "tev_design: widen the aminx dependency note to molxmpnn for new pins, keep the pinned a24 wheel, update CLAUDE.md aminx section and version-trap text"
repo = "tev_design"
size = "S"
depends_on = ["S3-13"]
gate = "uv lock --check clean; the CLAUDE.md aminx section names molxmpnn and the aminx_version/molxmpnn_version attrs"
user_decision = false

[[item]]
id = "S3-17"
title = "hautespout: decide bump versus leave pinned for vendor/aminx, update submodule URL and path if bumped"
repo = "hautespout"
size = "S"
depends_on = ["S3-13"]
gate = "git submodule update --init resolves on a fresh clone and the repo imports its vendored package"
user_decision = true

[[item]]
id = "S3-18"
title = "proteinsmc main (closed unmerged if S4-49 answers supersede): replace dead find_spec('prxteinmpnn') guard and imports with a root-only optional guard on molxmpnn (submodule resolved at call time), rewrite import paths to molxmpnn's layout, no runtime dependency edge"
repo = "proteinsmc"
size = "S"
depends_on = ["S3-13", "S4-49"]
gate = "tests: import succeeds and friendly ImportError raised with molxmpnn absent, import still succeeds with a stub molxmpnn whose __init__ raises RuntimeError, stub hub aminx leaves flag false, real molxmpnn (via the Q13 mechanism) scores one structure end to end on titanix; pyproject [project] dependencies and optional-dependencies unchanged"
user_decision = false

[[item]]
id = "S3-19"
title = "Skills: create using-molxmpnn from using-aminx content and reduce using-aminx to a pointer until the hub skill exists"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-13"]
gate = "both skill files parse; using-molxmpnn has no import aminx example outside the migration note; using-aminx is under 20 lines"
user_decision = false

[[item]]
id = "S3-20"
title = "Local environment runbook in a quiescent window: checkout rename, .bth.toml root and remote_roots, git worktree repair, venv rebuild on local/titanix/engaging, bathos slug policy and union query, Claude memory dir, praxia scope check, project_status.toml"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-02", "S3-13", "S3-28", "S1-16", "S1-17", "S1-20", "S1-22", "S2-01", "S2-20", "S2-25", "S4-20", "S4-21", "S4-42"]
gate = "S3-28 passes and its recomputed leaf set equals this depends_on; git worktree list shows only the main checkout; each checklist line has evidence; union query count is N_before plus the one new molxmpnn run; former worktrees repaired; import molxmpnn works from the new path on all three machines; memory dir exists at the new path"
user_decision = true

[[item]]
id = "S3-21"
title = "Docs and public surfaces: README and extras drift fix, CHANGELOG entry, RTD new project with redirect from the old one, record where the legacy gh-pages Sphinx site lands after the rename and decide redirect-or-accept-dead for maraxen.github.io/aminx/ (Q15)"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-02", "S3-13"]
gate = "rg for old URLs finds only allowlisted history; RTD redirect proven on a throwaway project then old project redirects; README install extras match pyproject; the ADR records the post-rename gh-pages Sphinx URL and the Q15 outcome (redirect probed on a throwaway repo, or maraxen.github.io/aminx/ accepted as dead), and S5-34 does not depend on a stub resolving"
user_decision = false

[[item]]
id = "S3-22"
title = "Close the deprecation window: consumer scan, PyPI download statistics, decide P1 (shim stays, hub is aminx-hub) or P2 (hub takes aminx 1.0.0a1) and whether to ever reclaim the repo slug; record the shim cap-raise step for molxmpnn 1.0"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-14", "S3-15", "S3-16", "S3-17", "S3-18", "S3-19", "S3-21"]
gate = "ADR records the consumer scan (zero hits outside the allowlist over the stated roots), elapsed days, and the decision P1 or P2-pending; P1 needs no soak; this item alone releases S3:aminx-name-freed together with S3-02"
user_decision = true

[[item]]
id = "S3-23"
title = "PyPI handover mechanics, only if P2: move the trusted publisher, publish hub as aminx 1.0.0a1 or later, verify the last shim stays installable, record no-yank policy"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-22"]
gate = "thresholds met and recorded (at least 90 days since S3-13, at most 50 weekly downloads of aminx >= 0.2.0a4 averaged over 4 weeks, zero scan hits), last shim version still installable, hub wheel version is at least 1.0.0a1, publisher registration points at the hub repo"
user_decision = true

[[item]]
id = "S3-25"
title = "Apply S2's EBM outcome to the shim: flip the aminx.ebm row to forward-if-installed (else install message), release the shim as a new shim-only version via its own tag"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-13", "S3-10", "S2-18"]
gate = "shim tests: ebmx absent gives the install message, stub ebmx present makes import aminx.ebm.model resolve to it; shim published via aminx-shim-v* tag and molxmpnn untouched"
user_decision = false

[[item]]
id = "S3-28"
title = "Assembly consistency check: stdlib script that parses all 261001 spec backlogs and asserts the section 2 assembly rows (S2-18 edge to S3-07, S2-19 absent and no non-S3 release of aminx, no second shim, S3-20 depends on exactly the computed leaf set, S2-03 depends on S3-13, S3-05 depends on S2-02, S1-25 and S5-52 ordered, S1-05/S1-17/S4-16 descend from S3-13, union DAG acyclic)"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-01"]
gate = "python3 scripts/rename/check_assembly.py exits 0 on the assembled backlog and non-zero on each seeded red fixture (missing S2-18 edge, S2-19 present or aminx-release title, missing leaf, S2-03 without S3-13, S3-05 without S2-02, unordered S1-25/S5-52, S1-05/S1-17/S4-16 without S3-13 ancestor, cycle)"
user_decision = false
```

## References

- Recon brief: `.praxia/docs/research/261001_ecosystem-hub-recon-brief.md` (D1-D7)
- Spike records: `.praxia/spikes/261001_molxmpnn-rename/S1`..`S10` (S7 invalid, superseded by S8; S10 is round 1, A30)
- Prior rename and its residue: `asr/CLAUDE.md:113`, `scripts/check_model_boundary.sh:18`,
  `proteinsmc/pyproject.toml:20-25`
- Ecosystem partition: `alphex/.praxia/docs/specs/260814_alphabet-contract.md` (D2/D4)
- Sibling specs: S1 `261001_spec-system-unification.md`, S2 `261001_ebm-extraction.md`,
  S5 `261001_aminx-hub.md`

## Revision log

### Round 1 (adversarial coherence cycle)

Applied (conceded or partial, each checked against the tree where it could be):

- **C1** S3-09 split. New S3-24 builds `check_stale_names.py` on the pre-rename tree (retired name is a parameter) and is a dependency of S3-07; S3-09 now only fixes the prxteinmpnn residue, decides the boundary scripts and wires CI. S3-07 gate (a) is satisfiable.
- **C2** Name reservation rewritten: new S3-27 uploads a real `molxmpnn 0.0.0.dev0` placeholder (a pending publisher does not reserve a name, A33). S3-02 now depends on S3-27 and records the old publisher state first. S3-03 registers an ordinary publisher on existing `aminx`, a pending one for `molxmpnn`, and both on TestPyPI. Dry runs use dev-suffixed versions. No release is possible between S3-02 and S3-03 (stated). A12 reworded and stays UNVERIFIED/deferred (docs.pypi.org is blocked in this sandbox); "re-create variables" became "verify preserved".
- **C3** Ordering settled in section 2 with a table of consequences; Q4 now offers R (rename first, recommended, matches S1) and E (S2 first). Under R, S3 is the sole publisher of `aminx 0.2.0a4` and owns `gone_modules.toml`; S2 must drop S2-19's release. The only S3 to S2 edge is S3-25 (and S3-20 to S2-18), and S2-18 depends on S3-07, so the graph is acyclic. Bathos slug disagreement resolved in favour of A8 (no project-id column): S2 must reword.
- **C4** Two workflows and tag namespaces (`v*` vs `aminx-shim-v*`), no `skip-existing`, a duplicate pre-check, independent shim versioning and a cap-raise step for molxmpnn 1.0; S3-12 gate requires a second dry run to succeed.
- **C5** Guard form is parent-first (spike S10 / A30 confirmed the raise); S3-18 gate gains the molxmpnn-absent import case, a real end-to-end score and the import-path rewrites.
- **C6** (verified: `asr/.gitmodules` lists two submodules; the vendored file differs from main, A27). S3-15 patches asr's copy directly and no longer depends on S3-18; the vendored fork is surfaced as Q12.
- **C7** Verified `AMINX_VERIFY` is read at import (A29). Every src-read var goes through `env()` with fallback; a classified table is a deliverable of S3-08 and a test guards its completeness. The brief's "16 names" is corrected: four are read from `src/`.
- **C8** Codemod gets preserve classes P1-P5 (history, provenance legacy surface, `tests/compat`, shim, bathos anchors), case variants, an explicit path-rename table (claims preserved, sha256 gate in S3-07) and a review class that blocks `--apply`; S3-08 owns the provenance rename so the intermediate state is defined.
- **C9** (verified: `model/_inference` does not exist, A28). S3-09 decides delete/revive/move-to-ruff per boundary script and its gate needs a guard firing on a real boundary.
- **C10** Slug recorded per sidecar before S3-20; count gate is `N_before + 1`; S3-20 adds `git worktree repair`, venv rebuild, a quiescent-window rule and edges to S1-25 and S2-18.
- **C11** HF in-place rename added as Q8 option (c) with ledger row A34 and a probe in S3-11.
- **C12** "user-confirmed" removed; Goal 3 states that the repo slug and PyPI name stay with molxmpnn under the recommended policies (D2 only partly satisfied).
- **C13** Scan now covers hautespout and drops tev_design with a justification. A32 records that hautespout's declared dependency was not confirmed in the main checkout.
- **C14** Provides gains `S3:molxmpnn-repo`, `S3:molxmpnn-manifest-namespace` (S3 provides only the root name; S4 defines ids), `S3:aminx-name-freed`, the slug policy and the quiescent-window contract.
- **C15** S3-05 and S3-11 directories use the layered resolver (stdlib copy for the pre-rename tree, agreement test in S3-08); the wheel cap is read from a measured baseline recorded in S3-05; "re-create" reworded.

Declined, with reason:

- **C15(d), split S3-07 into a mechanical commit and a separate gating item.** The gates must pass on the PR before merge; a separate item would merge an unverified rename. S3-07 stays one atomic L item and its heavy fixture runs are bounded by S3-05's per-fixture chunking.
- **C10, "fix the slug for pre-S3-20 runs" as a new slug.** Adopted as "state the slug explicitly in each sidecar" (they stay `aminx`) rather than introducing a transitional slug.
- **C6 (partial), "re-justify the S3-15 to S3-18 edge".** Dropped rather than re-justified: no mechanism connects the two.

### Round 2 (adversarial coherence cycle)

Applied (R2-n is the objection id):

- **R2-1** `tests/compat` split into `old_names/` (registered `consumer_shim` marker, `importorskip("aminx")`: runs and passes pre-rename, skipped after S3-07, runs again with the shim) and a generated `new_names/` (path map). S3-07 gate (e) requires exactly the old_names tests skipped; S3-10 and S3-13 require zero skips. Gate (g) now says the old_names files and `path_map.toml` are byte-unchanged. A42 added.
- **R2-2** Tag commit pinned (4.9): the merge of S3-07..S3-12 on `release/molxmpnn-0.2.0a4`; S3-13 asserts the tag diff touches only those items' paths and nothing under `run/` or `model/`, and re-runs the S3-05 fixtures against the tagged wheel. Optional S1 edges noted but not required. Because the tag precedes S2-18, the `aminx.ebm` gone row became conditional (`when = "target-absent"`).
- **R2-3** S3-03 now registers an ordinary publisher on `molxmpnn` (project exists after S3-27); TestPyPI is pending-where-absent, ordinary-where-present (A39). S3-27 records token scope and holder; revocation is S3-03's gate (A38).
- **R2-4** Quiescence is computed: S3-20 depends on S3-02, S3-13, S3-28 and the leaf set of S1/S2/S4 items on the molxmpnn/aminx repo (12 ids). S5 molxmpnn-repo items must descend from S3-20 (assembly row 7). S3-20 waits for S4's aminx-repo items.
- **R2-5** (partial) Assembly checklist in section 2 listing the exact S2 edits; new item S3-28 (stdlib check, runs locally) verifies it and is a dependency of S3-13 and S3-20. S3-13 gate names the `aminx 0.2.0a4` collision. Rows 4 and 5 are manual lines in the report. I re-read S2 (S2-18 deps, S2-19 title, lines 564-598) and the objection holds.
- **R2-6** Guard is root-only `find_spec("molxmpnn")`; submodule resolved at call time. A36 added (read: `importlib/util.py:88-92`, `__init__.py:13-24`); S3-18 gate adds a stub whose `__init__` raises.
- **R2-7** Q13 added with recommendation (a) non-default `[dependency-groups]`, citing alphex `260814_alphabet-contract.md:278,283-289`; gate reworded to leave `[project].dependencies` and `optional-dependencies` unchanged.
- **R2-8** S3-01 gate extended to all of Q1-Q14 with a dated accepted-at-recommendation fallback and a re-open table naming the items each answer regenerates.
- **R2-9** Numbers added (4.9, Q14): >= 90 days, <= 50 weekly downloads over the last 4 weeks, named scan roots including GitHub code search. P1 needs no soak; S3-22 releases `S3:aminx-name-freed` together with S3-02, and S3-23 enforces the P2 thresholds. S5-33 already depends on S3-02 and S3-22, so no S5 edit is needed.
- **R2-10** `uv.lock` handled by a dedicated root-entry-only codemod step (A37, read: `uv.lock:67-69`); S3-07 gate asserts third-party lines unchanged and `uv lock --check` clean.
- **R2-11** Environment stamp, 5 determinism reruns, refuse-to-compare on mismatch, tolerance rule and negative-control separation (>= max(10 x tol, 1e-6)) added to 4.6, S3-05 and S3-07. A41 added (deferred).
- **R2-12** (partial) Allowlist is a generated artifact of `rename_rules.toml` (`--emit-allowlist` in S3-06) with a schema owned by S3-24 and a consistency test. S3-06 now depends on S3-24 for the schema.
- **R2-13** S3-07 gate split: CI runs (a)-(d),(g),(h); titanix runs (e),(f); merge needs CI green plus a cited bathos run id with evaluated outcome.
- **R2-14** S3-27's HF step is conditional on Q8 (4.7).
- **R2-15** RTD slug probe added to S3-01; redirect mechanism is ledger row A40 (UNVERIFIED, deferred) with a throwaway-project probe in S3-21.
- **R2-16** A35 corrected (two jobs); tag guard on both, seeded wrong-tag gate in S3-12.
- **R2-17** (partial) S3-03 dropped from S3-11 (weights do not need the PyPI publisher). S3-04 dropped from S3-06 (the P3 rule needs only a path glob, the fixture is synthetic); S3-07 keeps its direct S3-04 edge.

Declined: none. Not spiked this round: A38-A41 stay UNVERIFIED with `deferred:` reasons (PyPI, TestPyPI, RTD, titanix unreachable from this sandbox). Not independently re-verified: the cited S2 line numbers beyond the ones listed in the R2-5 entry.

### Coherence round 1 (cross-spec adversarial cycle, 261001)

Applied to S3 only; other specs are patched by their own agents. C-n is the objection id.

- **C1** Rename-vs-extraction ordering is one user question (S1 Q10 = S3 Q4 = S2 Q7) answered once at S3-01. Assembly row 8 (S2-03 `depends_on` S3-13) added and enforced by S3-28, so a not-yet-revised S2 is red mechanically. No TOML change for this issue.
- **C2** Row 2 now states S2-19 is deleted (dependents re-point to S2-18, removed from every leaf list); S3-25 is the sole owner of the `aminx.ebm` row and S3-13 the sole publisher of `aminx 0.2.0a4`. S2-19 removed from S3-20 `depends_on`; S3-28 checks S2-19 is absent.
- **C3** Replaced the Q4 "filter-repo includes both" remark: S2-03 must follow S3-13 (row 8, section 2 table, S3-28, S3-13 gate). S1 items touching the frozen paths (e.g. S1-11) remain S1/S2's to order against S2-03..S2-06.
- **C4** Row 7 and 4.8(2) withdrawn. S5 `molxmpnn` items are intentionally unordered against S3-20 and protected by the fail-closed `git worktree list` step. `S3:aminx-name-freed`, Q14 and the S3-22 gate now say first deployment under P1 needs S3-01 (S3-13 transitively); only S5-55 waits for S3-22/S3-23. The S3-28 row-7 check is deleted. Round 2 entries R2-4 and R2-9 above are superseded on these points (the log is append-only and is not rewritten).
- **C5** S3-18 `depends_on` gains S4-49 and a Consumes row; it closes unmerged on "supersede" and S3-22 treats it as complete. Q13 applies only if S3-18 survives; Consumes no longer says nothing is consumed from S4.
- **C6** Leaf defined as "no direct edge from another S1/S2/S4 member on the MPNN repo". S3-20 now lists S1-16, S1-17, S1-20, S1-22, S2-01, S2-20, S2-25, S4-20, S4-21, S4-42 (S4-18/S4-32 are not leaves, S4-42 added, S2-19 dropped); S3-28 computes the set instead of trusting prose. If S2's revision leaves S2-18 without a dependent, S3-28 will flag it and the list must gain it.
- **C7** S3-05 `depends_on` gains S2-02 (golden capture and S3-07 comparison both on xtrax a11); assembly row 9 states S4-16 must not claim to land #174 rebased on the rename. Alternative (merge after S3-07, stamp a10) rejected.
- **C12** The manifest ids and entry-point names are pointed at S4-01; S3 keeps the root-name-only Provides row.
- **C14** Release-train table added to 4.9 (0.2.0a4 by S3-13, then the next free alpha per release item in tag order, each recording its contained branches; factory-resolution tests run against the tagged wheel); assembly row 10 requires S1-25 and S5-52 to be ordered.

Declined: none. Assumption ledger unchanged: no new assumption leans on unread code (the leaf set and the S4/S2/S5 edges were recomputed by parsing the current backlog blocks, a read of the specs on 261001, not a spike).

### Coherence round 2 (cross-spec adversarial cycle, 261001)

- **CH2-01** Release order reversed: assembly row 10 and the 4.9 release table now give `0.2.0a5` to S5-52 and `0.2.0a6` to S1-25, with S1-25 `depends_on` gaining S5-52 (an edge S1 adds; S3 does not edit S1). Reason: the first hub deployment (S5-33 -> S5-53 -> S5-52) must not wait for the S1 flip chain, the coupling row 7 already withdrew for S3-20; S1-25 is a tail item. "Recommended: S5-52 after S1-25" deleted. The S3-28 row-10 check is unchanged in kind (comparable in the union DAG, either direction passes) and its red fixture is now "edge removed". Reversal cost recorded in row 10 for the S3-01 re-open table. S1's `0.2.0a5` gate wording and S5's "deliberately not ordered" wording must give way; flagged in row 10, not edited here.
- **CH2-03** The 0.2.0a4 pin restated as a content rule (4.9, S3-13 gate): nothing under `src/molxmpnn/` outside S3-owned paths, `pyproject.toml`/`uv.lock` only as S3 requires, only additive `tests/`/`scripts/`/`docs/`/`.praxia/` files from items with no S3-13 ancestor. New assembly row 11 (checked by S3-28): S1-05, S4-16 and, added by me, S1-17 (a `src/` edit) have S3-13 as an ancestor. The claim that the tag-commit rule makes the S1 edges unnecessary is removed. S1-05 (`src/aminx/run/fields.py`) and S4-16 (`depends_on` S4-03, S3-07, S2-02) read in the S1 and S4 specs on 261001.
- **CH2-07** A24 replaced by S5's verified fact (demo never deployed; `maraxen.github.io/aminx/` serves a stale `gh-pages` Sphinx build). New unverified A43 (Pages does not redirect after rename; stub would need a repo named `aminx`, which option 1 forbids) with a throwaway-repo probe deferred to S3-21. S3-21 now records where the Sphinx site lands and decides redirect-or-accept-dead (new Q15, recommendation: accept dead); its title and gate no longer assume a live demo or a stub. The `MOLXMPNN_MODELS_RELEASE_TAG` variable is dropped from the S3-03 title, gate and section 7 row, and from the 4.4 `pages.yml` fallback (conditional on `pages.yml` surviving). S5-34's "S3-21 old-URL stub still resolves" gate must be restated by S5 (flagged in Q15).

Declined: none. TOML block unchanged in ids and edges (S3-28 title/gate and S3-03/S3-21 text only).

### Convergence-check (cross-spec adversarial cycle, 261001)

Objection R2-1 re-verified as still open on three points and fixed in 4.5, section 7 (S3-06, S3-07 rows) and the TOML blocks of S3-06, S3-07 and S3-10:

- **Skip assertion was unsatisfiable.** "Skip count equals the old_names test count" cannot hold in a suite that already skips for unrelated reasons (`tests/conftest.py:41`, `:62`, `:66`, 75 skip sites in 37 files), and a module-level `importorskip` skips per module, not per test (A44, read). Replaced by a (file, reason) set comparison between a `before` run (S3-07 parent) and an `after` run (PR head) inside the one titanix bathos record, same node and stamp: every `old_names` module skipped with the `could not import 'aminx'` reason, no other new skip, `new_names` passes with zero skips; two negative controls. No baseline is carried from S3-05, so S3-05 is unchanged and cannot go stale against the S3-07 parent.
- **Prose gates mirrored into the machine-readable block.** The S3-07 TOML gate now carries `path_map.toml` byte-unchanged, `gen_compat_new_names.py --check` and the set comparison; the S3-10 TOML gate carries the zero-skip `pytest -m consumer_shim tests/compat/old_names` run with the shim installed.
- **P3 narrowed and the generator named.** P3 is now `tests/compat/old_names/**` plus `tests/compat/path_map.toml`; `tests/compat/new_names/**` is class `G` (generated, skipped by `--apply`). `scripts/rename/gen_compat_new_names.py` (`--write`/`--check`) is built and fixture-tested by S3-06 (synthetic map, no dependency on S3-04), run by S3-07 and checked in CI.

Declined: none. Ledger: A44 added (VERIFIED by reading the installed pytest, no run). Ids and `depends_on` edges unchanged.
