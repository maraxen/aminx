---
title: "S3: aminx identity kept: hub naming and stale-name cleanup"
description: "aminx keeps its name and PyPI identity (no rename). S3 records the separate hub's identity (repo slug, PyPI policy, praxia.science domain, bathos slug, orchestrator-name collision), cleans the previous rename's stale names, guards release uniqueness, and checks the spec set's assembly mechanically."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - aminx
  - aminx-hub
  - proteinsmc
related_specs:
  - S1
  - S2
  - S4
  - S5
---

# S3: aminx identity kept: hub naming and stale-name cleanup

Spec only. Nothing here executes yet (user decision D7). This file was `261001_molxmpnn-rename.md`,
then `261001_aminx-identity-and-hub-naming.md` (git rename). **User decision 2026-10-01, revised D2 in
`.praxia/docs/research/261001_ecosystem-hub-recon-brief.md`:** the MPNN package keeps the name `aminx`
(it is released on PyPI as `aminx`), so the rename, the `molxmpnn` name, the `aminx` shim distribution and
every cutover that came with them are **retired** (section 10 lists each retired item with its reason). The
hub is a separate project holding the combined implementations. D1 and D3-D7 are unchanged: EBM still moves
to its own project (debt #2369), the two spec systems still become one, proteinsmc still takes no runtime
edge to aminx (D6), nothing executes yet.

## 1. Goal and non-goals

**Goal.** After this spec's items are done:

1. **aminx is unchanged in identity and this is a written contract** (`S3:aminx-identity`, section 6):
   PyPI `aminx`, import `aminx`, repo `maraxen/aminx`, HF weights `maraxen/aminx`, CLI `aminx`, skill
   `using-aminx`, bathos slug `aminx`, env vars `AMINX_*`, store attribute `aminx_version`. No consumer
   (`mpnn_ext`, `asr`, `tev_design`, `hautespout`) needs an edit.
2. **The hub's identity is a recorded user decision** (S3-01): repo slug, whether it publishes to PyPI and
   under what name, the GitHub Pages custom domain, the bathos slug, and a policy for the collision with
   the user's praxia agent orchestrator. S3 gives a recommendation for each and decides none of them.
3. **The hub can never take the name `aminx`** as a PyPI project, an importable top-level package or a
   package.json name, enforced by a checker mode (S3-24) because a second package importable as `aminx`
   defeats every `find_spec("aminx")` availability guard [A6].
4. **The custom domain, if chosen, is wired safely** (S3-29): domain verification first, record created only
   after the Pages site exists, base path flipped in one place.
5. **The previous rename's stale names are removed and guarded** (S3-24, S3-09, S3-18) and the public
   surface drift is fixed (S3-21).
6. **No release can collide on a version** (S3-12): three specs cut releases through one workflow and PyPI
   forbids re-upload [A56].
7. **The spec set's assembly is checked by a script, not by prose** (S3-28).

**Non-goals.**

- No rename of anything, no shim distribution, no codemod, no repository, HF or PyPI cutover.
- No behaviour change to the MPNN code, no spec-system change (S1), no EBM extraction (S2).
- No hub functionality (S5): S3 records the hub's identity, S5 builds it.
- No decision on the hub name, the PyPI name or the domain: those are explicit user decisions (S3-01).
- No rewrite of git history, dated docs, CHANGELOG entries, bathos records or `outputs/` (append-only logs).
- **Resume and derived-data rules.** No item here runs longer than a few minutes and none writes a cache,
  index, staging directory or venv, so the preemption-safe chunking rule and the derived-data-directory
  rule are not engaged; every item is re-runnable from scratch and the checker items only read trees.

## 2. Spec set coordination

With no rename, S3 is no longer the DAG root that S1, S2 and S4 code items wait behind. S3 still orders
four things: **S3-12 before any release item** (S1-25, S5-52, any S2 release), **S3-01 before the hub repo
exists and is first deployed** (S5-01, S5-33), **S4-49 before S3-18**, and **S5-33 before S3-29**.

### 2.1 What each other-spec edge to a retired S3 item was for, and where it goes

"Resulting deps" is what the edge list becomes if only the S3 edges are removed or re-pointed. Ids were read
in the other specs' toml blocks and prose on 261001. The owning spec makes its own edit; S3-28 fails until
it has.

| Item (spec) | Old S3 edge | What the edge was for | Disposition and resulting deps |
|---|---|---|---|
| S1-01 | S3-07 | characterization goldens "on the renamed but otherwise unchanged code": ordering against the rename only | retired, no replacement; S1-01 becomes a root; goldens are captured on current main |
| S1-02 | S3-07 | ordering against the rename only | retired, no replacement; deps `[S1-01]` |
| S1-03 | S3-07 | ordering against the rename only | retired, no replacement; becomes a root |
| S1-05 | S3-13 | content rule of the pinned 0.2.0a4 tag: a `src/` edit must not predate it | retired, no replacement; deps `[S1-02, S1-04]` |
| S1-17 | S3-07, S3-13 | same two reasons (rename ordering, pinned-tag content rule) | retired, no replacement; S1-17 becomes a root (`depends_on = []`) |
| S1-25 | S3-13 | first row of the release train: S1-25 must not race the first release for a version | replaced by S3-12 (the release guard and train rule); deps `[S1-12, S1-13, S1-14, S1-15, S3-12, S5-52]`; the gate's version text becomes "next free alpha per S3-12" (0.2.0a5 on 261001, after S5-52's 0.2.0a4) |
| S1-26, S1-27, S1-28 | S3-10 | the shim keeps consumers on the old name working while they migrate | retired, no replacement (there is no old name); each keeps `[S1-13]` |
| S1-26 | S3-14 | "stack on the S3-14 consumer branch" | retired; S1-26 branches from mpnn_ext main |
| S1-27 | S3-15 | "stack on the S3-15 consumer branch" | retired; S1-27 branches from asr main |
| S1-28 | S3-17 | stack on S3-17's bump-or-leave-pinned decision | retired; S1-28 owns the bump-or-leave-pinned decision itself (it is already `user_decision = true`) |
| S2-02 | S3-05 (title only) | "S3-05 and S4-16 depend on this": S3-05 needed the xtrax pin to capture rename goldens | retired; the title keeps S4-16 only |
| S2-03 | S3-13 | A0 freeze on the renamed, released tree | retired, no replacement; deps `[S2-02]`; the title "Freeze after the rename" is reworded by S2 |
| S2-18 | S3-07 | S2 edited the post-rename tree | retired, no replacement; the EBM removal notice (CHANGELOG pointer) is S2-18's; there is no shim and no `gone_modules.toml` to edit |
| S4-16 | S3-07, S3-13 | rename ordering and pinned-tag content rule | retired, no replacement; deps `[S4-03, S2-02]` |
| S4-19 | S3-07 | manifest module paths in the renamed tree | retired; manifest ids use the root name `aminx` (S4-01 owns the grammar) |
| S4-19 | S3-11 | weights pinned at the post-cutover HF revision | retired; weights stay at the current HF repo and revision (`io/weights.py`); no cutover exists |
| S4-35 | S3-07 | rename ordering | retired, no replacement; deps `[S4-16, S4-34]` |
| S4-22, S4-23, S4-49 | S3-18 | proteinsmc guard | **kept**: S3-18 survives, now guarding `aminx` (section 4.4); S4-49 decides supersede-or-sequence as before |
| S5-14 | S3-07 | browser sampler directory rename (old Q10) | retired; the directory stays `browser/aminx-sampler` |
| S5-14, S5-52 | S3-12 | release plumbing | **kept, re-meant**: S3-12 is now the release guard, dry-run and train rule (section 4.5) |
| S5-52 | S3-13 | "first alpha after 0.2.0a4" | retired; the version is the next unused alpha read from PyPI by S3-12's rule (0.2.0a4 on 261001) |
| S5-21, S5-61 | S3-07 | module paths in the renamed tree | retired, no replacement; S5-61 deps `[S4-19]`, S5-21 deps `[S5-19, S4-19]` |
| S5-33 | S3-01 | naming ADR before first deployment | **kept**: S3-01 is now the hub identity ADR; S5-33's gate wording on "S3-02, S3-12, S3-13 arrive only transitively" is reworded |
| S5-34 | S3-07 | removing `site/` from the renamed tree | retired, no replacement |
| S5-34 | S3-21 | the docs URL record | **kept**: S3-21 still records the docs URL (section 4.4) |
| S5-55 | S3-22, S3-23 | "P2: hub takes the aminx name" and the PyPI handover | retired: P2 contradicts revised D2. Its useful residue (flip `base_path` when a custom domain is attached) is S3-29; S5 retires S5-55 or rewrites it to depend on S3-29 |
| S5-01 | none today | creates the repo and records its slug | **request**: S5-01 gains S3-01 so the slug is decided before the repo exists |

Prose-only mentions of retired ids (S2 A33/A34/A37 and Q7, S1 A31 and Q10, S5 sections 4.9 and 6, S4 section 6
contract tokens `S3:molxmpnn-repo`, `S3:molxmpnn-manifest-namespace`, `S3:aminx-name-freed`) are the owning
specs' to reword; S3-28 lists them with file and line as warnings. **Q10 (S1) = Q4 (S3) = Q7 (S2), the
"ordering against the rename" question, no longer exists.**

### 2.2 Assembly checklist (checked by S3-28)

| # | Required state of the assembled backlog | S3-28 check |
|---|---|---|
| C1 | ids unique, every `depends_on` resolves, union DAG acyclic | fails on a missing id, duplicate or cycle |
| C2 | no retired S3 id (section 10) appears in any item's `depends_on`, `title`, `gate` or `repo` in any spec | fails and prints the field; prose mentions elsewhere are printed as warnings with file:line |
| C3 | no item has `repo` equal to `molxmpnn` or `mpnnx`, and none of those strings occurs in an item field outside S3 | fails; prose hits printed as warnings |
| C4 | **release items are ordered**: S5-52 and S1-25 are comparable in the union DAG (default S5-52 an ancestor of S1-25; either direction passes), and every other item on repo `aminx` whose title starts with `Release` or `Publish` or contains `release cut` is comparable with both | fails if two release items are unordered; prints the set |
| C5 | S3-12 is an ancestor of S1-25 and of S5-52 | fails otherwise (no release without the guard) |
| C6 | S3-01 is an ancestor of S5-01 and of S5-33, and S5-33's direct S3 dependencies are a subset of {S3-01}; S3-29 has S3-01 and S5-33 as ancestors | fails otherwise |
| C7 | S3-18 `depends_on` contains S4-49 | self-check of this spec's block |
| C8 | manual lines printed for a human: S2-18's CHANGELOG pointer for `aminx.ebm`; S1-28's bump-or-pin decision text; S4-37's "carried across the rename" wording; S5-55's retirement or repointing | printed, not machine-checkable |

S3-28 is a read-only stdlib script (no heavy compute, imports no project code) so it can run anywhere. On the
converged specs a real-tree run is expected to exit 0; a non-zero exit names a defect in a spec to fix.

**Scope of S3-28 (CH1-01).** It checks only the single `toml` block of each `261001_*.md` spec. It does not read
the generated backlog-DAG plan (`.praxia/docs/plans/261001_ecosystem-backlog-dag.md`), the checker input
`.praxia/spikes/261001_dag_given_edges.txt`, `.praxia/docs/INDEX.md` or the brief's front matter. Those are
derived from the specs, so a green S3-28 says nothing about them. The plan is still the rename-era graph (231
items, 553 edges, retired S3 ids, repo label `molxmpnn`, OQ-01..OQ-11 shim and cutover questions, a link to the
renamed file) and the given-edges file is a hand transcription of it; neither may be used to file backlog items.
After the plan is regenerated from the revised blocks, S3-28 is run against it as well (`--plan` mode: same
C1-C7 over the plan's item and edge tables, plus a diff of the plan's edge set against the union of the six
blocks, which must be empty), and the given-edges file is regenerated from the specs rather than retyped. The
brief's front-matter description is corrected and `INDEX.md` is regenerated with
`docs(action="index")`, never hand-edited.

**Owner and trigger of the regeneration (R2-05).** The owner is the integrator step that follows convergence of
this cycle; it is not an S1..S6 backlog item. In order: (1) regenerate the plan from the six blocks (expected
211 items: 31+32+8+48+59+33); (2) regenerate `.praxia/spikes/261001_dag_given_edges.txt` from the specs; (3) run
`docs(action="index")` so `INDEX.md` drops the deleted `261001_molxmpnn-rename.md` entry; (4) correct the
brief's description and spec-table row (new file name, no `molxmpnn`); (5) run
`check_assembly.py --plan` and require exit 0. No backlog item is filed before step 5 passes.

## 3. Current state (anchored; tags are in the Assumption Ledger)

Paths are relative to the aminx worktree root unless prefixed `../../../../` (that is `~/projects/`).

### 3.1 aminx identity that stays

- PyPI project `aminx`: ten releases, `0.1.0a1` .. `0.2.0a3`, **all pre-releases**; `0.2.0a3` is the latest
  and `0.2.0a4` is unpublished on 261001 [A3, A58, spike S11, S19].
- `pyproject.toml:6-7` name `aminx`, version `0.2.0a3` as a **static string** (not derived from a tag, no
  `dynamic` version) [A54]; `:34` console script `aminx = "aminx.cli:main"`; `:37-38` repository URL
  `https://github.com/maraxen/Aminx.git` (capital A, relying on case-insensitive resolution) and
  documentation `https://aminx.readthedocs.io`.
- Release is PyPI trusted publishing: `.github/workflows/release.yml:30-41` (`environment: pypi`,
  `id-token: write`, no token) [A11]; the workflow fires on every `release: published` and has two jobs,
  `build` and `publish` [A63]. It contains no PyPI duplicate check, no tag-to-version comparison, no
  `skip-existing` setting and no manual trigger [A55].
- Carried from the earlier revision, not re-verified here and **not load-bearing** (nothing is renamed):
  `mpnn_ext`, `asr`, `tev_design` declare `aminx` as a dependency; `mpnn_ext`, `asr`, `hautespout` fetch the
  `aminx` submodule from `https://github.com/maraxen/aminx.git`. They keep working because the repo, the
  import name and the distribution name do not change.

### 3.2 Facts about the hub's candidate names

- PyPI `aminx-hub` and `praxia-science` return 404 (unregistered); GitHub `maraxen/aminx-hub`,
  `maraxen/praxia-science` and `maraxen/praxia.science` return 404 to an unauthenticated request (a private
  repository would too) [A46, A47, spike S11]. PyPI `praxia` **is registered** (latest `0.1.0a0`, author
  field `GenArch, Praxia Contributors`, summary "Specialized multi-agent orchestrator ..."; whether the user
  owns it is not established here) and `maraxen/praxia` returns 404 unauthenticated (spike S11).
- DNS on 261001 (public resolver, spike S18): `praxia.science` is a live zone delegated to two Cloudflare
  nameservers; it has **no A, AAAA, CNAME or TXT records at the apex**; `hub.praxia.science`,
  `aminx.praxia.science` and `www.praxia.science` are NXDOMAIN [A52]. Nothing would be displaced by a
  subdomain record.
- GitHub's Pages docs: a custom domain is a per-site setting; a subdomain uses a CNAME record to
  `<owner>.github.io`; GitHub recommends **verifying the domain before adding it to a repository, to avoid
  takeover attacks** [A51]; when the site is published from a custom Actions workflow **no CNAME file is
  created** and any existing one is ignored [A62].
- The hub's own config already isolates the URL prefix: `hub.toml [site] base_path` (`/aminx-hub/`, `/aminx/`
  or `/` for a custom domain) is the only place the prefix exists, and the hub "ships no Python package"
  [A53].
- Bathos run provenance is keyed by the `project_slug` column only (no project id column); the slug `aminx`
  holds 145 runs on 261001, `praxia` 61, `aminx-hub` none [A61, spike S21]. A slug chosen for the hub
  therefore has to be fixed before its first tracked run, and is independent of the repo slug.

### 3.3 The orchestrator collision (the user flagged it; inventory from spike S14 and S11)

| Name in use by the praxia orchestrator | Evidence |
|---|---|
| CLI binary `praxia` (crate `praxia-cli`) and 138 workspace crates named `praxia-*` | `../../../../praxia/crates/praxia-cli/Cargo.toml`, `../../../../praxia/Cargo.toml` via S14 |
| GitHub repository `maraxen/praxia` (origin of that checkout; 404 unauthenticated) | S14, S11 |
| bathos project slug `praxia` (61 runs) | S14, S21 |
| the `.praxia/` directory convention: 55 of the user's projects have one | S14 |
| PyPI project `praxia` (registered; ownership not established) | S11 |
| near-name `praxis` (sibling lab-automation project, and a consumer of the S6 shared editor) | S14 |

Consequence: a hub whose repo, PyPI, npm, CLI or bathos name contains `praxia` would share a namespace with
the orchestrator in every one of those tables. A **DNS name** under `praxia.science` does not: DNS labels do not
collide with any of the above.

### 3.4 Stale names left by the previous rename (prxteinmpnn to aminx, 260813) [A49, A28, A27, A57, A60]

- 75 tracked files contain `prxteinmpnn` (case-insensitive); **32 are outside dated history** (`.praxia/`,
  `outputs/`, `.claude/`): 24 under `scripts/engaging`, 2 in `src/aminx` (a comment at
  `potts/__init__.py:31` and a docstring at `sampling/__init__.py:1`), one each in `scripts/cluster`,
  `scripts/recapture`, `scripts/test_refactoring_validation.sh`, `scripts/check_model_boundary.sh`,
  `examples/colab_inference_demo.ipynb` and `docs/parity` (spike S17).
- `scripts/check_model_boundary.sh` has 6 lines naming the dead package and 0 naming `aminx`, and it guards
  `model/_inference/`, which does not exist in `src/aminx/model/` [A28, A49]; it passes while checking
  nothing.
- `../../../../proteinsmc/src/proteinsmc/scoring/mpnn.py` guards the dead name
  (`PRXTEINMPNN_AVAILABLE = find_spec("prxteinmpnn")`) and imports `prxteinmpnn.*` at module level behind it,
  so MPNN scoring there is unreachable (spike S12) [A27].
- asr's `vendor/proteinsmc` is not in `asr/.gitmodules` (paths: `aminx`, `vendor/euclidean_fast_attention`)
  and its `scoring/mpnn.py` already guards `find_spec("aminx")` with the `aminx` imports **deferred into
  `make_mpnn_score`'s body** (module level has only a `TYPE_CHECKING` import), for the reason its own comment
  gives (a module-level import drags in `proxide.ops.dataset`, `jax_md` and `haiku`; asr backlog #3713)
  [A27, A57, spike S12]. Spike S12 also found that this directory has its own `.git` directory whose origin
  is `maraxen/proteinsmc.git`: it is a nested clone, not the "plain directory copy" the earlier revision
  called it. Nothing in S3 edits it.
- README line 114 documents `pip install "aminx[cuda]"`; `pyproject.toml` defines `cpu` and `cuda12`, not
  `cuda` [A60].
- The legacy docs site: `maraxen.github.io/aminx/` serves a stale Sphinx build from the `gh-pages` branch
  (legacy Pages, last commit 2025-11-10); the browser demo workflow is `workflow_dispatch` only and has never
  run, so there is no demo URL to preserve [A45].

## Assumption Ledger

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A3 | Every published aminx release on PyPI is an alpha or other pre-release version. | The release-train rule in S3-12 needs a stable-release story | VERIFIED | spike: S11 |
| A6 | A find_spec('aminx') availability guard returns True for any installed package named aminx, including a hub with no scoring module, so the guard's friendly ImportError path is bypassed. | The reserved-name mode in S3-24 and the call-time error in S3-18 are unnecessary | VERIFIED | spike: S13 |
| A11 | release.yml publishes through PyPI trusted publishing (OIDC id-token, environment pypi, no API token). | S3-12 must handle a token and the publisher registration | VERIFIED | read: .github/workflows/release.yml:30-41 |
| A27 | asr's `vendor/proteinsmc` is not a git submodule and its `scoring/mpnn.py` differs from proteinsmc main (AMINX_AVAILABLE/find_spec('aminx') vs PRXTEINMPNN_AVAILABLE/find_spec('prxteinmpnn')). | S3-18 would be a submodule bump or would not be needed | VERIFIED | spike: S12 |
| A28 | `src/aminx/model/` has no `_inference` module, so `scripts/check_model_boundary.sh` guards a nonexistent boundary even after its names are fixed. | S3-09 only needs a name fix for that script | VERIFIED | read: src/aminx/model/__init__.py:9-10; read: scripts/check_model_boundary.sh:14-23 |
| A30 | `importlib.util.find_spec('pkg.sub.mod')` raises ModuleNotFoundError when `pkg` is not installed, whereas `find_spec('pkg')` returns None, so guards must test the root package first. (Round 2: this covers only the root-absent case; the root-present case is A36.) | The Q11 guard form works as a dotted name | VERIFIED | spike: S15 |
| A36 | `importlib.util.find_spec('pkg.sub')` imports the parent package `pkg` (executing its `__init__`) before looking up `sub`, and aminx's `__init__` eagerly imports io, run, sampling and scoring (the JAX stack). So a dotted-name guard is not import-free and any non-ImportError raised by the parent breaks the caller. | The guard may test the dotted submodule | VERIFIED | spike: S15; read: src/aminx/__init__.py:13-24 |
| A45 | `maraxen.github.io/aminx/` serves a stale Sphinx build from the legacy gh-pages branch, and the browser demo workflow pages.yml is workflow_dispatch only and has never run, so no demo URL exists to preserve. | S3-21's docs-URL record changes | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:46-53; read: .github/workflows/pages.yml:7-8 |
| A46 | PyPI project names aminx-hub and praxia-science are unregistered (the JSON API returns 404). | S3-01's recommendation moves to the next candidate name | VERIFIED | spike: S11 |
| A47 | No public GitHub repository exists at maraxen/aminx-hub or maraxen/praxia-science (HTTP 404; a private repository would also return 404, so this proves only that no public repo exists). | S3-01 must pick another slug or the user must rename an existing repository | VERIFIED | spike: S11 |
| A48 | ruff banned-api matches the literal module string and emits no warning for a ban key that no code imports, so a dead key (a module path that no longer exists) never fires and is never reported. | The guard-name check in S3-24 is unnecessary for banned-api | VERIFIED | spike: S16 |
| A49 | The previous rename (prxteinmpnn to aminx) left stale names: tracked files outside dated history (.praxia/, outputs/, .claude/) still contain prxteinmpnn, and scripts/check_model_boundary.sh names only the dead package. | S3-09 reduces to a note | VERIFIED | spike: S17 |
| A50 | The user's praxia agent orchestrator already owns the names praxia (CLI binary, GitHub repository maraxen/praxia, bathos project slug) and the .praxia directory convention used across the user's projects, and a separate project named praxis exists; the bathos catalog has no project slug aminx-hub. | The collision inventory in 3.3 and Q20 change | VERIFIED | spike: S14 |
| A51 | GitHub's Pages documentation describes a custom domain as a per-site setting with a DNS CNAME record (for a subdomain) pointing at <owner>.github.io, and describes domain verification as protection against domain takeover; none of this ties the domain to the repository name. | The order of operations in S3-29 is rewritten from what the page says | VERIFIED | spike: S18 |
| A52 | praxia.science is a live DNS zone (it has NS records), and the subdomains hub.praxia.science and aminx.praxia.science have no A or CNAME records today, so a hub subdomain would not displace an existing service. | The recommended label in Q18 moves | VERIFIED | spike: S18 |
| A53 | The hub ships no Python package, and hub.toml `[site] base_path` is the only place the URL prefix exists. | Q17's no-PyPI recommendation and S3-29's one-line base_path change both change | VERIFIED | read: .praxia/docs/specs/261001_aminx-hub.md:298-301; read: .praxia/docs/specs/261001_aminx-hub.md:709-712 |
| A54 | aminx's version is a static string in pyproject.toml, not derived from the git tag, so a release tag can disagree with the built version. | The tag-to-version check in S3-12 is unnecessary | VERIFIED | read: pyproject.toml:7 |
| A55 | release.yml contains no step that queries PyPI for an existing version, compares the release tag with the built version, or sets skip-existing, and has no workflow_dispatch trigger. | S3-12 shrinks to the guards that are missing | VERIFIED | spike: S19 |
| A56 | PyPI does not allow a distribution filename to be reused once uploaded (stated on pypi.org/help), so a second upload of the same version fails permanently. | The release-train ordering rule and the duplicate check are unnecessary | VERIFIED | spike: S19 |
| A57 | asr's vendored proteinsmc defers its aminx imports into the body of make_mpnn_score (none at module level outside TYPE_CHECKING), so importing the module does not import aminx. | S3-18 must not copy that form and needs another way to keep import light | VERIFIED | spike: S12 |
| A58 | The PyPI JSON endpoint /pypi/aminx/<version>/json answers 200 for a published version (0.2.0a3) and 404 for an unpublished one (0.2.0a4 on 261001), which is enough for a pre-publish duplicate check. | S3-12 must use another endpoint | VERIFIED | spike: S19 |
| A59 | With Cloudflare as the DNS host, the Pages CNAME record must be DNS-only (not proxied) for GitHub to see it and provision the HTTPS certificate. | S3-29's DNS-only instruction is stricter than needed | UNVERIFIED | deferred: needs the user's Cloudflare account and a live Pages site; S3-29 observes the Pages HTTPS certificate state after the record is created and records what it saw |
| A60 | README.md documents the install extra `cuda`, but pyproject.toml defines the extras `cpu` and `cuda12`, not `cuda`. | The README fix in S3-21 is unnecessary | VERIFIED | read: README.md:114; read: pyproject.toml:96 |
| A61 | Bathos run provenance is keyed by a project_slug column, the runs table has no separate project id column, and the aminx slug holds more than 100 runs, so changing a slug after runs exist orphans them. | The decide-the-slug-first ordering in S3-01 is unnecessary | VERIFIED | spike: S21 |
| A62 | GitHub's Pages documentation states that when a site is published from a custom GitHub Actions workflow no CNAME file is created, so the custom domain is a repository setting rather than a file in the deployed artifact. | S3-29 must add a CNAME file to the deploy artifact | VERIFIED | spike: S22 |
| A63 | `release.yml` fires on every `release: published` event and has two jobs, `build` (no environment, no permissions) and `publish` (`needs: build`, `environment: pypi`, `id-token: write`). | S3-12 puts the duplicate check in the wrong job | VERIFIED | read: .github/workflows/release.yml:3-5; read: .github/workflows/release.yml:8; read: .github/workflows/release.yml:27-31 |

Spike tooling: `scripts/loop/adversarial_metrics.py` is not in this workspace, so spikes ran with
`python3.13 ~/projects/praxia/scripts/loop/adversarial_metrics.py spike-run --workspace <this worktree> --spec <this file>`.
Records are in `.praxia/spikes/261001_aminx-identity-and-hub-naming/S11`..`S22` (the validator keys them by
this file's stem; the earlier S1..S10 under `261001_molxmpnn-rename/` belong to the retired design and no row
cites them). S20 is superseded by S21 (its claim text embedded a run count that changed by one between runs).
S12 pre-registered "no `.git` entry in the vendored directory"; the result had one. The row it carries (A27)
claims only "not a submodule" and "differs from main", both of which held, so the row stands and the nested
clone is recorded in 3.4 as a finding instead of being hidden. Of the earlier revision's ledger only A3, A6,
A11, A27, A28, A30 and A36 are carried (text unchanged). Every other earlier id was rename-only and is retired
with the items that used it; ids are never reused, and rows whose claim was reworded got new ids (earlier A8 to
A61, A9 to A48, A19 to A49, A24 to A45, A35 to A63). The earlier rows that cited external-repo paths
(`../../../../asr/...`) could not pass the validator's workspace-bound `read:` rule, so the ones still needed
(A27) are re-established by spike.

## 4. Design

### 4.1 What stays (the `S3:aminx-identity` contract)

| Layer | State | Note |
|---|---|---|
| Import package, PyPI distribution, CLI, console script | `aminx` | unchanged; `pyproject.toml:6,34` |
| GitHub repo | `maraxen/aminx` | submodule URLs of asr, mpnn_ext, hautespout stay valid; no redirect is involved |
| HF weights | `maraxen/aminx`, `HF_REVISION` as pinned in `io/weights.py` | no cutover; S4-19 pins this revision |
| Env vars, store attribute, provenance helpers | `AMINX_*`, `aminx_version`, `resolve_aminx_version` | unchanged; no compat layer, no dual stamp |
| Docs | `aminx.readthedocs.io` plus the legacy Pages Sphinx build | S3-21 records them (4.4) |
| Skill | `using-aminx` | the hub skill is S5's and has its own name |
| Bathos slug | `aminx` (145 runs) | the hub gets its own slug (Q19); no slug renamed |
| Local checkout, Claude memory, `.bth.toml` roots | `~/projects/aminx` | untouched |
| Manifest/id namespace root (S4) | `aminx` | S3 provides only the root name; S4-01 owns ids and entry-point stems |

### 4.2 Hub identity (S3-01): options and recommendation

Repo name, PyPI name, domain, bathos slug and npm name are five independent layers. The user's two
candidates for the repo name are the working slug **`aminx-hub`** and a name derived from the owned domain
**praxia.science**. A GitHub Pages custom domain works with any repository slug [A51], so choosing the domain
does not force the repo name.

| Layer | Options | Recommendation | Why |
|---|---|---|---|
| Repo slug (Q16) | (a) `maraxen/aminx-hub`; (b) `maraxen/praxia-science` (or `praxia.science`); (c) a new GitHub org | **(a)** | Both (a) and (b) are free as public repos and PyPI names [A46, A47]. (a) is what every S5 item, `hub.toml` and sidecar already carries, it names what the hub fronts (the MPNN family first), and it shares no namespace with the orchestrator (3.3). The repo is new, so it has no pinned submodule consumers; renaming it later is cheap in git terms but **not free for bathos**, which keys runs by slug [A61]: that is why the *bathos slug* is decided separately and first (Q19). Choose (b) only if you want the hub to be the front door of everything under the domain, not the MPNN family |
| PyPI (Q17) | no project; or `aminx-hub`; never `aminx` | **no PyPI project** | The hub ships no Python package [A53]. If you later publish, `aminx-hub` is free [A46]. The import name `aminx` is reserved for the MPNN package for good (A6); S3-24's reserved-name mode enforces it in the hub's CI |
| Custom domain (Q18) | none; subdomain `hub.praxia.science`; apex `praxia.science` | **subdomain `hub.praxia.science`, wired after the first deployment (S3-29)** | The apex has no records today and keeps its option value for an umbrella or orchestrator site; a subdomain is one CNAME and displaces nothing [A52]. `hub` is neutral (`aminx.praxia.science` would put the MPNN package under the orchestrator's brand). Deploy first on the project URL, then attach the domain: the base path is the one `hub.toml` line [A53] |
| Bathos slug (Q19) | `aminx-hub`; a name matching a different repo slug | **`aminx-hub`**, kept even if the repo is renamed | Free [A50]; S5 sidecars already use it; the slug need not follow the repo [A61] |
| Orchestrator collision policy (Q20) | avoid `praxia` in repo, PyPI, npm, CLI and bathos names but allow it in DNS and prose; or accept the shared namespace | **avoid in all non-DNS names** | 3.3: `praxia` is a CLI, a repo, a crate family, a bathos slug, a registered PyPI project and a directory convention in 55 projects, and `praxis` is a near-name in the same lab. A subdomain of `praxia.science` collides with none of those |
| Site brand text | "aminx hub" | not decided here | S5's page titles; the domain does not force it |

The ADR also records the layer-by-layer answers as a table S5 reads (slug, PyPI, domain, bathos slug).

### 4.3 Custom domain mechanics (S3-29)

Order of operations (GitHub docs, A51, A62; every step is a user action because it needs DNS and account control):

1. **Verify the domain with GitHub first** (account or organization Pages settings; GitHub gives a TXT
   challenge to add in Cloudflare). This is GitHub's own recommendation "to avoid takeover attacks" [A51] and
   is why no CNAME is created before step 3.
2. S5-33 has deployed the Pages site (source: Actions) on the project URL; the hub is public (Pages on a
   free plan needs a public repository; confirm for the account at step 2).
3. Set the custom domain in the repository's Pages settings. With an Actions deployment **no CNAME file is
   created and none is needed** [A62]; do not add one to the artifact.
4. Create the Cloudflare record `hub CNAME maraxen.github.io` (subdomain, A51), **DNS-only, not proxied**
   [A59, unverified: the step is the conservative choice and S3-29 records what the Pages certificate state
   showed]. Then wait for the certificate and enable "Enforce HTTPS".
5. In the hub repo set `hub.toml [site] base_path = "/"` and redeploy; the e2e harness already passes under
   `/` (S5 section 4.1) [A53].
6. Record the record values, the verification date and the observed behaviour of the old project URL.
7. **Never leave a dangling record**: if the Pages site is unpublished or the repo deleted, delete the CNAME
   record in the same change (that is the takeover risk step 1 exists for).

### 4.4 Stale-name cleanup (S3-24, S3-09, S3-18, S3-21)

**The checker (S3-24).** `scripts/hygiene/check_stale_names.py`, stdlib only, one file, vendorable into
other repos. Three modes, each with red and green fixtures on synthetic trees:

1. `--retired <name>` (repeatable): fails on any occurrence not covered by the allowlist. Walks tracked files
   (`git ls-files`) or, with `--root`, a plain directory.
2. **Guard-name resolution.** Every root package named by a ruff `banned-api` key, by a `find_spec(...)` or
   `import_module(...)` string literal, or by a `metadata.version(...)` literal must resolve: for an
   in-repo root the full dotted path must be a file or package directory under `src/`; for an external root,
   the root must be a declared dependency (`[project].dependencies`, optional dependencies or dependency
   groups) or an allowlisted external; stdlib roots resolve via `sys.stdlib_module_names`. Needed because a
   dead banned-api key is silent in ruff [A48] and the previous rename left exactly such a dead guard [A49, A28].
3. `--reserved <name>` (repeatable): fails if any `pyproject.toml` `[project].name`, `package.json` `name`, or
   top-level package directory under `src/` equals a reserved name. The hub's CI runs it with
   `--reserved aminx` (S5 vendors the file). Matching is exact (`aminx-hub` or `aminx-browser-smoke` do not
   match `aminx`). On the real aminx tree the same flag **must** report exactly two legitimate hits:
   `pyproject.toml:6` (`[project].name`) and `src/aminx` (top-level package directory; the only
   `src/*/__init__.py`). The tree's `package.json` names (`aminx-browser-smoke`, `aminx-browser-layer-c`) give
   none. This is the positive control that the mode sees both rules. A synthetic green fixture, a hub tree with
   `src/executors` and a `package.json` named `aminx-hub`, must exit 0 and proves the exact-match semantics.

Allowlist schema (hand-written, no generator since there is no codemod): `[[allow]] glob = ..., retired = ...,
reason = ..., class = "history" | "external"`. `history` covers append-only logs (`.praxia/docs/**`,
`.praxia/handoffs/**`, `.praxia/sprint_plans/**`, `.praxia/spikes/**`, `outputs/**`, recorded commands in
`*.bth.toml`); every row needs a reason. Each class is exercised by one fixture.

**The residue (S3-09).** Uses S3-24 with `--retired prxteinmpnn`. For each of the 32 non-history files
(3.4) decide rewrite, delete as dead, or allowlist with a reason; recorded sidecar commands stay as
history. For `scripts/check_model_boundary.sh` and any other boundary script (found by the guard-name mode and
by `rg -l '_inference|internal-import' scripts`), record one decision: (a) delete (the boundary is gone), (b)
revive against a real boundary, or (c) move the rule into ruff `banned-api` where the linter enforces it.
Recommended: (a) for `_inference` [A28]. A surviving guard must fire in a red/green pair on a **real,
pre-existing boundary** (a temporary test module importing a real internal symbol from outside), never on a
fabricated path. The checker is wired into `ci.yml`.

**The proteinsmc guard (S3-18).** In `proteinsmc` main, replace
`PRXTEINMPNN_AVAILABLE = find_spec("prxteinmpnn")` and the module-level `prxteinmpnn.*` imports with the form
asr's nested clone already uses [A27, A57]: `AMINX_AVAILABLE = importlib.util.find_spec("aminx") is not None`
(root only, [A30, A36]), `TYPE_CHECKING`-only imports at module level, and the runtime imports
(`aminx.scoring.score.make_score_sequence`, `aminx.utils.decoding_order`) inside `make_mpnn_score`, which
raises a friendly `ImportError` ("aminx is not installed. ...") when the flag is false and wraps any
non-ImportError raised while importing `aminx` into an `ImportError` that names the cause. Import paths map
as: `prxteinmpnn.scoring.score` -> `aminx.scoring.score` (`make_score_sequence` is an alias of `make_score_fn`,
`scoring/score.py:214`), `prxteinmpnn.utils.decoding_order` -> `aminx.utils.decoding_order`,
`prxteinmpnn.utils.types` -> `aminx.types`. No dependency edge (D6): `[project].dependencies` and
`[project.optional-dependencies]` are unchanged; Q13 fixes how the end-to-end test gets `aminx`. S3-18 is
**closed unmerged** if S4-49 answers "supersede" (S4-22 then owns the guard) and S3-09 does not wait for it.

A hub that installed a top-level package named `aminx` would make this guard true and fail at the call
site [A6]; that is the reason for the reserved-name mode.

**Public surfaces (S3-21).** README: fix the `aminx[cuda]` install line to the real extras [A60], add one
sentence pointing to the hub at the URL recorded by S3-01; CHANGELOG: one entry stating that aminx keeps its
name and the browser hub is a separate project; `pyproject.toml` `[project.urls] repository`:
canonicalise `maraxen/Aminx.git` to `maraxen/aminx` (GitHub is case-insensitive; the capital letter is
residue). **Docs URL record** (the one S5-34 depends on): canonical API docs stay on
`aminx.readthedocs.io`; the legacy `maraxen.github.io/aminx/` Sphinx build is stale and nothing in the demo
needs it [A45]; default is "leave as is, no redirect work" unless Q21 says otherwise. S3-21 records the state
of both URLs (HTTP status, last build date) in the ADR so S5-34's gate "the docs URL recorded by S3-21
resolves" is checkable.

### 4.5 Release guard and release train (S3-12)

**Why it survives the rename's retirement.** Three items in three specs cut releases through the one
`release.yml` that fires on every `release: published` [A63]: S5-52 (browser assets), S1-25 (spec flip) and
any S2 release carrying the EBM removal. PyPI forbids re-upload of a filename [A56], `pyproject.toml` pins the
version as a static string [A54], and `release.yml` verifies nothing before publishing [A55]. Two of them
tagging the same number, or a tag disagreeing with the built version, is a permanent failure at the worst
moment (after the environment approval).

**Changes to `release.yml` (S3-12).**

- A `Release guards` step in the `build` job (so it fails before the environment-gated `publish`,
  [A63]) runs `scripts/release/check_release.py` (stdlib): (1) for a `release` event, the tag must equal
  `v<built version>` [A54]; (2) `https://pypi.org/pypi/aminx/<version>/json` must return **404**; 200 fails
  with "already published", and any other status or a transport error also fails (fail closed) [A58];
  (3) wheel gates: `aminx/py.typed` present, no `aminx/model_params/*.eqx.zst` (the 88.8 MB incident that
  `exclude-package-data` at `pyproject.toml:292` guards), wheel size at most `baseline_wheel_bytes * 1.10`
  where the baseline is measured by S3-12 and committed as `release/wheel_baseline.json` (bytes and sha256;
  not asserted in prose).
- `workflow_dispatch` trigger added: it runs `build` and every guard and **never** `publish`
  (`publish` gets `if: github.event_name == 'release'`). No TestPyPI, so no new publisher registration and no
  user action is needed; the existing trusted publisher [A11] is untouched.
- `skip-existing: false` is set explicitly on the publish step.

**Release-train rule (an assembly row, C4/C5).** The item cutting a release reads the next unused alpha from
the PyPI JSON for `aminx` at cut time (no number is hard-coded in an item), states it in the PR, and records
the contained branches in the release notes.

| Release item | Version (as read on 261001) | Contains | Order |
|---|---|---|---|
| S5-52 browser-assets release | next unused alpha: `0.2.0a4` | everything merged to main at its tag | first (Q22 recommendation: keeps the first hub deployment off the S1 flip chain) |
| S1-25 spec-flip release | the following alpha: `0.2.0a5` | the previous release plus S1's work | after S5-52 (S1-25 `depends_on` S5-52) |
| any S2 release carrying the EBM removal | next unused alpha at its cut | whatever is merged | S2 adds the item and orders it against the two above (C4); with no S2 release item the removal rides the next release cut after S2-18 |

The factory-resolution test in S4-19 and S5-52's gate runs against the tagged wheel, not the tree, as before.
If you reverse Q22, the edge flips (S5-52 `depends_on` S1-25) and S5-33 inherits the S1 chain; C4 passes in
either direction.

### 4.6 The assembly check (S3-28)

`scripts/spec_set/check_assembly.py` parses the single `toml` block of every `.praxia/docs/specs/261001_*.md`
with `tomllib`, builds the union graph and applies C1-C7 of section 2.2. It reads the retired ids from the
first column of this file's section 10 table, so the retired set has one source of truth. Output is a list of
failures (field, item, file) plus warnings (prose mentions with file:line) plus the C8 manual lines.

Scope: toml blocks only (see the scope note under section 2.2). The `--plan` extension named there is added to
S3-28's gate (section 7) so a stale generated plan fails the same way a stale spec does.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Other specs still depend on retired S3 ids or still say `molxmpnn` | Section 2.1 maps every edge; S3-28 C2/C3 fail until each spec is revised, so a stale edge cannot survive as a footnote |
| The hub takes the import or PyPI name `aminx` (user decision drift, or a copied template) | `--reserved aminx` in the hub's CI (S3-24) [A6]; S3-01 records the rule as part of the ADR |
| The hub is named so that it shares a namespace with the orchestrator (`praxia`) | Section 3.3 inventory in the ADR; Q20 recommendation; the domain stays DNS-only |
| Bathos runs orphaned because the hub slug changes after the first tracked run [A61] | S3-01 fixes the slug (Q19) before S5 sidecars run; the slug is independent of the repo slug |
| Domain takeover or a dangling CNAME after the Pages site goes away | Verify the domain first and delete the record with the site (4.3, S3-29) [A51] |
| Cloudflare proxying blocks GitHub's certificate provisioning [A59, unverified] | DNS-only record; S3-29 records the observed certificate state |
| Two releases race for a PyPI version and one fails after approval [A56] | S3-12 duplicate check in `build`, fail closed; C4/C5 order the release items |
| Release tag disagrees with the built version [A54] | tag-to-version check in S3-12 |
| An 88.8 MB weights wheel ships again | wheel gates in S3-12 (size cap from a measured baseline, no `model_params/*.eqx.zst`) |
| Silent loss of enforcement from dead guards (ruff keys, `find_spec` literals, boundary scripts) [A48, A49, A28] | S3-24 guard-name resolution; S3-09 requires a guard firing on a real boundary |
| proteinsmc import pulls in the MD stack or breaks on a broken `aminx` install [A36, A57] | root-only guard, deferred imports, stub-package tests (S3-18) |
| S3-18 collides with S4-22 (two guards) | S3-18 depends on S4-49; closed unmerged on "supersede" |
| The checker flags legitimate history | allowlist `history` class with a reason per row and one fixture per class |
| README keeps an install line that does not work [A60] | S3-21 gate compares README extras to `pyproject.toml` |

## 6. Interfaces

### Provides

| Contract | What it guarantees | Where | Downstream use |
|---|---|---|---|
| **`S3:aminx-identity`** | The names in 4.1 do not change: PyPI, import, repo, HF repo, CLI, env vars, store attribute, bathos slug, id-namespace root `aminx`. Replaces `S3:molxmpnn-repo` and `S3:molxmpnn-manifest-namespace` (S3 provides only the root name; S4-01 defines ids and entry-point stems) | this file (no item) | S4 manifest ids, S5 catalog provenance and release URLs, all consumers |
| **`S3:hub-identity`** | The user's recorded answers for repo slug, PyPI policy, domain, bathos slug, collision policy (and the rule that the hub never takes `aminx`). Replaces `S3:aminx-name-freed` | **S3-01** | S5-01 (slug), S5-33, S3-29 |
| **`S3:release-guard`** | `release.yml` refuses a duplicate version, a tag/version mismatch and a bad wheel; the release-train ordering rule | **S3-12**, 4.5 | S5-14, S5-52, S1-25, any S2 release item, S4-19 (factory-resolution test runs against the tagged wheel) |
| **`S3:stale-name-guard`** | `check_stale_names.py`: retired names, guard-name resolution, reserved names, allowlist schema | **S3-24**; CI wiring in S3-09 | S2's new EBM repo and S5 vendor the single stdlib file |
| **`S3:assembly-check`** | The script and the checklist of section 2.2 | **S3-28** | every spec's author; run before any revision is declared converged |
| **`S3:hub-domain`** | The custom domain, if chosen, resolves to the hub with HTTPS and `base_path = "/"` | **S3-29** | S5-55 (if kept as the base_path flip) |

### Consumes

| Contract | From | Used for |
|---|---|---|
| Decision supersede-or-sequence for the proteinsmc guard (**S4-49**), and the fate of `make_mpnn_score`'s signature and the `mpnn` registry key | **S4** | S3-18 `depends_on` S4-49; on "supersede" S3-18 is closed unmerged |
| The first successful hub deployment on the Pages project URL (**S5-33**) | **S5** | S3-29 step 3 onward |
| `hub.toml [site] base_path` as the single prefix location | **S5** (A53) | S3-29 step 5 |
| The release items **S1-25** and **S5-52** and any S2 release item | **S1, S5, S2** | C4/C5 ordering; they are not dependencies of any S3 item |
| The other specs' toml blocks | **S1, S2, S4, S5, S6** | S3-28 only (read-only) |

## 7. Verification gates per item

Narrow test runs for aminx go to titanix or CI (never a local pytest, per the compute rules); the stdlib
scripts (S3-24, S3-28, the release checker) import no project code and may run anywhere. Every item that
produces a pass/fail finding about a real tree (S3-24 real-tree runs, S3-09 residue, S3-28 real run, S3-18
end-to-end) has its `.bth.toml` sidecar (hypothesis plus `[outcomes]`) committed **before** the run, runs as
`bth run --project-slug <slug> -- uv run --no-sync python3 scripts/...` (never `uv run bth`), and is verified
by its record (`bth compact` then `bth sql`), not by exit code. Each checker is validated on synthetic ground
truth with a positive control that must pass and a negative control that must fail.

| Item | Gate |
|---|---|
| S3-01 | ADR `.praxia/docs/decisions/261001_hub-identity-and-aminx-name.md` records an answer (or a dated accepted-at-recommendation) for every question in section 8, the availability probes (PyPI and GitHub names, DNS records, re-run with the date), the collision inventory of 3.3, the rule "the hub never takes `aminx`", and a re-open table naming the items each different answer regenerates. S5-01 and S5-33 read the answers from it |
| S3-24 | `pytest tests/hygiene -q` on titanix: red fixtures fail (dead banned-api key that ruff itself ignores [A48], dead `find_spec` literal, dead `import_module` literal, residual retired name, reserved name in a synthetic hub tree), green fixtures pass (including a synthetic hub tree with `src/executors` and a `package.json` named `aminx-hub`, which proves exact-match semantics), the allowlist classes each have one fixture, a malformed allowlist exits non-zero. Real tree (bathos run, sidecar first): `--retired prxteinmpnn` reports the known residue (positive control, expected non-zero before S3-09: 32 non-history files on 261001); `--reserved aminx` on the real tree reports exactly two hits, `pyproject.toml:6` (`[project].name`) and `src/aminx` (top-level package directory); guard-name mode reports zero unresolved guards on the current `pyproject.toml` or lists each with its line |
| S3-09 | Using S3-24: zero `prxteinmpnn` outside the allowlist, every allowlist row has a reason, no recorded `*.bth.toml` command rewritten; a recorded decision (delete, revive, ruff) for `scripts/check_model_boundary.sh` and every other boundary script found; the surviving guard fires in a red/green pair on a real existing boundary; `ci.yml` runs the checker (a seeded residue file in a throwaway PR fails the job) |
| S3-12 | CI: `workflow_dispatch` run builds and runs every guard and the `publish` job does not run; seeded failures each fail the `build` job (a built version already on PyPI, a tag `v9.9.9` against a built `0.2.0a3`, a wheel containing a `model_params/*.eqx.zst`, a wheel over the cap); on a real unpublished version the guards pass; `release/wheel_baseline.json` exists with bytes and sha256 and its cap is read from it; `tests/release/test_check_release.py` on titanix covers the PyPI answers 200, 404, 500 and a transport error (only 404 passes) with a stubbed fetcher. The release-train rule is in `docs/` and referenced by S1-25 and S5-52 |
| S3-18 | Closed unmerged, without gate, if S4-49 answers "supersede". Otherwise new proteinsmc tests: (1) **aminx absent**: `import proteinsmc.scoring.mpnn` succeeds and `make_mpnn_score` raises the friendly ImportError; (2) **aminx present but broken**: a stub `aminx` on `sys.path` whose `__init__` raises `RuntimeError` still lets the module import (root-only guard, no parent import, A36), the flag is true, and the failure surfaces at the `make_mpnn_score` call as an ImportError that names the cause; (3) a stub package `aminx` with no `scoring` module: the flag is true (A6) and the call raises an ImportError naming `aminx.scoring`, never a bare traceback at import; (4) real `aminx` installed through the mechanism fixed by Q13 (default: a non-default `[dependency-groups]` entry used only by this test on titanix): the flag is true and one structure is scored end to end (bathos run, sidecar first); `rg PRXTEINMPNN_AVAILABLE` finds no remaining importer; S3-24 `--retired prxteinmpnn` is clean on `proteinsmc/src`; the `pyproject.toml` diff leaves `[project].dependencies` and `[project.optional-dependencies]` unchanged (D6) and adds at most the one dependency group named in Q13 |
| S3-21 | `rg` for `aminx[cuda]` finds nothing outside history; the README install extras equal the extras in `pyproject.toml`; the README names the hub URL recorded in S3-01; the CHANGELOG entry exists; `[project.urls] repository` is `https://github.com/maraxen/aminx`; the ADR records HTTP status and last build date of `aminx.readthedocs.io` and `maraxen.github.io/aminx/` and the Q21 outcome, and S5-34's gate can cite it |
| S3-28 | `pytest tests/spec_set -q` on titanix over synthetic spec directories: red fixtures (a retired id in `depends_on`, a retired id in a title, `repo = "molxmpnn"`, S5-52 and S1-25 unordered, S1-25 without S3-12 as ancestor, S5-01 without S3-01, S5-33 with a direct dependency on S3-12, a cycle, a duplicate id, an unresolvable dependency) each exit non-zero with the failing field named; a green fixture exits 0; a `--plan` fixture whose edge set differs from its blocks, or that carries a retired id or `repo = "molxmpnn"`, exits non-zero. The real-tree run (`python3 scripts/spec_set/check_assembly.py --specs .praxia/docs/specs`) is a bathos run with sidecar first; once the specs are converged it is expected to exit 0 (a failure is a defect to fix, not a to-do list); after the integrator regeneration (CH1-01) `--plan` must also exit 0, and both outputs are attached to the revision that declares the set converged |
| S3-29 | The verification TXT challenge was completed and is recorded; DNS (public resolver) returns a CNAME for the chosen host to `<owner>.github.io`; `curl -sI https://<host>/` returns 200 over HTTPS with a certificate valid for the host; the Pages settings show the custom domain and HTTPS enforced; `hub.toml [site] base_path` is `/` and the deployed-origin Playwright smoke passes with `crossOriginIsolated` true on `run/`; the observed behaviour of the old project URL and of the proxy setting [A59] are recorded in the ADR; no CNAME file exists in the artifact [A62] |

## 8. Open questions for the user

Numbering continues the earlier revision (Q1-Q15), so no number changes meaning. Retired questions are in
section 10.

| # | Question | Recommendation |
|---|---|---|
| Q9 | Include the previous rename's residue (75 tracked files, 32 outside history) in this spec? | Yes (S3-09, S3-24): the same defect class in one pass |
| Q11 | proteinsmc guard: a root-only `find_spec` guard (optional, no dependency edge) or an entry-point provider? | Root-only: `find_spec("aminx") is not None`, imports deferred into `make_mpnn_score`, submodule resolved at call time [A30, A36, A57]. S4-22's entry-point work, if it lands, supersedes it (S4-49). It matches the existing precedent in asr's clone and D6 |
| Q12 | asr carries a nested clone of proteinsmc at `vendor/proteinsmc` (own `.git`, origin `maraxen/proteinsmc.git`, not in `.gitmodules`; spike S12) whose `mpnn.py` differs from proteinsmc main. Is that still wanted, or should asr depend on proteinsmc normally? | Out of S3's scope. Nothing in S3 edits it. File it as a D6-adjacent debt item for the ecosystem-partition work |
| Q13 | How does `aminx` get into proteinsmc's test environment for S3-18's end-to-end test without a runtime dependency (D6)? (a) a non-default `[dependency-groups]` entry (dev-only, one titanix test), (b) install out-of-band on titanix and record the command in the sidecar | Applies only if S3-18 survives S4-49. (a): alphex `260814_alphabet-contract.md:283-289` (D4) treats a dev-only test dependency as not a coupling edge. (b) is acceptable but leaves the test unreproducible from the repo |
| Q16 | Hub repo slug: (a) `maraxen/aminx-hub`, (b) a praxia.science-derived `maraxen/praxia-science`, (c) a new org? | (a), see 4.2. Both (a) and (b) are free [A46, A47] |
| Q17 | Does the hub publish to PyPI, and under what name? | No PyPI project (the hub ships no Python package [A53]). If ever, `aminx-hub` [A46]. Never `aminx` as a PyPI name, import package or npm name |
| Q18 | Custom domain: none, subdomain `hub.praxia.science`, or the apex `praxia.science`? And when? | Subdomain `hub.praxia.science`, after the first deployment (S3-29). The apex has no records [A52] and stays free for an umbrella or orchestrator site |
| Q19 | Bathos project slug for the hub | `aminx-hub`, fixed before the first S5 tracked run, kept even if the repo is renamed [A61] |
| Q20 | The hub name versus the praxia orchestrator: avoid `praxia` in repo, PyPI, npm, CLI and bathos names (allowing it in DNS and prose), or accept the shared namespace? | Avoid in all non-DNS names (3.3). Also decide whether site text may say "praxia" |
| Q21 | aminx's API docs URL: leave `aminx.readthedocs.io` canonical and the stale `maraxen.github.io/aminx/` Sphinx build as is, or retire/replace the latter? | Leave as is; revisit at S5-34. The content is stale, there is no demo behind it [A45] |
| Q22 | Release order: S5-52 (browser assets) before S1-25 (spec flip)? | Yes, as in the earlier revision (4.5): the first hub deployment does not wait for the S1 flip chain. Reversing it flips one edge |

## 9. Backlog items

Items use the target repo names. Order of work is expressed only through `depends_on`. Retired ids are not
in this block (section 10).

```toml
[[item]]
id = "S3-01"
title = "Hub identity decision (user): hub repo slug (aminx-hub or a praxia.science-derived name), whether the hub publishes to PyPI and under what name (never aminx), GitHub Pages custom domain (praxia.science or a subdomain such as hub.praxia.science, with the DNS and verification steps), bathos project slug, the praxia-orchestrator name collision policy, the aminx docs URL and the release order; record an ADR with availability probes and a re-open table"
repo = "aminx"
size = "S"
depends_on = []
gate = "ADR .praxia/docs/decisions/261001_hub-identity-and-aminx-name.md exists with an answer (or a dated accepted-at-recommendation) for every question in section 8, PyPI/GitHub/DNS availability results with dates, the collision inventory, the rule that the hub never takes the name aminx, and a re-open table naming the items each different answer regenerates"
user_decision = true

[[item]]
id = "S3-24"
title = "Stale-name checker check_stale_names.py (stdlib, vendorable): retired-name mode, guard-name resolution (banned-api keys, find_spec/import_module/metadata.version literals) and reserved-name mode (hub never named aminx), hand-written allowlist with history/external classes, red and green fixtures"
repo = "aminx"
size = "S"
depends_on = []
gate = "pytest tests/hygiene -q on titanix: dead banned-api key, dead find_spec literal, residue and reserved-name fixtures fail, green fixture passes, each allowlist class has a fixture; real-tree bathos run (sidecar first) reports the prxteinmpnn residue with --retired prxteinmpnn and exactly two hits with --reserved aminx (pyproject.toml:6 and the src/aminx package directory); a synthetic hub tree with src/executors and a package.json named aminx-hub passes"
user_decision = false

[[item]]
id = "S3-09"
title = "Fix the previous rename's residue with the S3-24 checker (prxteinmpnn in the 32 non-history files), decide per boundary script to delete, revive or move to ruff banned-api, and wire the checker into ci.yml"
repo = "aminx"
size = "M"
depends_on = ["S3-24"]
gate = "zero prxteinmpnn outside the allowlist with a reason per allowlist row; a recorded decision per boundary script; the surviving guard fires in a red/green pair on a real existing boundary; ci.yml runs the checker and a seeded residue file fails it"
user_decision = false

[[item]]
id = "S3-12"
title = "Release guard and train rule: release.yml gains a build-job guard step (tag equals v<built version>, PyPI JSON for aminx/<version> must be 404, fail closed, wheel py.typed/no model_params weights/size cap from a measured baseline), a workflow_dispatch dry run that never publishes, explicit skip-existing false; document the next-free-alpha release-train rule"
repo = "aminx"
size = "S"
depends_on = []
gate = "CI: a dispatch run builds and runs the guards and skips publish; seeded failures (already-published version, tag/version mismatch, weights in the wheel, oversize wheel) each fail the build job; guards pass on a real unpublished version; release/wheel_baseline.json exists and the cap is read from it; tests/release on titanix cover PyPI answers 200, 404, 500 and a transport error"
user_decision = false

[[item]]
id = "S3-18"
title = "proteinsmc main (closed unmerged if S4-49 answers supersede): replace the dead find_spec('prxteinmpnn') guard and module-level imports with a root-only optional guard on aminx and imports deferred into make_mpnn_score (the form asr's nested clone already uses), rewrite import paths to aminx's layout, no runtime dependency edge"
repo = "proteinsmc"
size = "S"
depends_on = ["S4-49", "S3-24"]
gate = "tests: import succeeds and a friendly ImportError is raised with aminx absent; import still succeeds with a stub aminx whose __init__ raises RuntimeError; a stub aminx without scoring leaves the flag true and the call raises an ImportError naming aminx.scoring; real aminx via the Q13 mechanism scores one structure end to end on titanix; no remaining PRXTEINMPNN_AVAILABLE importer; pyproject [project] dependencies and optional-dependencies unchanged"
user_decision = false

[[item]]
id = "S3-21"
title = "Public surfaces: README install-extras fix and hub pointer, CHANGELOG entry that aminx keeps its name, canonical repository URL in pyproject, and an ADR record of the aminx docs URLs (RTD and the stale legacy Pages Sphinx build) with the Q21 outcome"
repo = "aminx"
size = "S"
depends_on = ["S3-01"]
gate = "README extras equal pyproject extras and no aminx[cuda] remains outside history; README names the hub URL from S3-01; CHANGELOG entry exists; pyproject repository URL is lowercase; the ADR records HTTP status and last build date of both docs URLs and the Q21 outcome so S5-34 can cite it"
user_decision = false

[[item]]
id = "S3-28"
title = "Assembly consistency check scripts/spec_set/check_assembly.py: stdlib script that parses every 261001 spec's toml block and asserts the section 2.2 rows (unique ids and resolvable acyclic edges, no retired S3 id or molxmpnn string in any item field, release items ordered and behind S3-12, S3-01 before S5-01 and S5-33, S3-29 behind S5-33), warns on prose mentions and prints the manual lines"
repo = "aminx"
size = "S"
depends_on = []
gate = "pytest tests/spec_set -q on titanix: each red fixture (retired id in depends_on, retired id in a title, repo molxmpnn, unordered S5-52 and S1-25, S1-25 without S3-12, S5-01 without S3-01, S5-33 depending on S3-12 directly, cycle, duplicate id, unresolvable edge) exits non-zero naming the field, green fixture exits 0; real-tree bathos run (sidecar first) over the six converged blocks exits 0 (expected 211 items); once the integrator has regenerated the plan, `--plan` over it also exits 0 with an empty edge-set diff, and that exit 0 is required before any backlog item is filed"
user_decision = false

[[item]]
id = "S3-29"
title = "Custom domain wiring for the hub (user, DNS and account control): verify the domain with GitHub, set the custom domain in the Pages settings after S5-33 (no CNAME file), create the DNS-only CNAME record, enforce HTTPS, flip hub.toml base_path to /, record the observed behaviour, never leave a dangling record"
repo = "aminx-hub"
size = "S"
depends_on = ["S3-01", "S5-33"]
gate = "TXT verification recorded; public DNS returns the CNAME to <owner>.github.io; https://<host>/ returns 200 with a valid certificate and HTTPS is enforced; hub.toml base_path is / and the deployed-origin Playwright smoke passes with crossOriginIsolated true on run/; the old project URL behaviour and the proxy setting observed are recorded in the ADR; the artifact contains no CNAME file"
user_decision = true
```

## 10. Retired items

These ids are **not** in the toml block. Each is listed with the reason and where its purpose went. Ids are
never reused. S3-28 reads the first column of this table as the retired set.

| Old id | What it was | Reason retired | Replacement or owner |
|---|---|---|---|
| S3-02 | rename GitHub repo to molxmpnn, redirect probe | no rename; the repo stays `maraxen/aminx` | none |
| S3-03 | register trusted publishers for the shim and molxmpnn, revoke the placeholder token | no new dists, no repo rename; the existing publisher [A11] keeps matching | none |
| S3-04 | consumer import-surface inventory and `tests/compat/old_names` contract test | existed to prove the shim; there is no old name | none |
| S3-05 | pre-rename golden outputs and environment stamp | existed to prove the rename is behaviour-preserving; nothing is renamed | none (S1-01 captures its own goldens) |
| S3-06 | rename codemod, preserve classes, `gen_compat_new_names.py` | no rename | none; the allowlist idea survives as a hand-written file in S3-24 |
| S3-07 | atomic rename PR (`src/aminx` to `src/molxmpnn`) | no rename | none; every edge to it was rename ordering (2.1) |
| S3-08 | compat layer: `MOLXMPNN_*` env fallback, dir resolver, dual version stamp, pickle remap | no rename, so no old names to honour | none |
| S3-10 | build the `aminx` shim distribution | no `molxmpnn` for it to forward to | none; S1-26/27/28's edge to it was ordering only |
| S3-11 | weights cutover to a new HF repo | no rename; weights stay at `maraxen/aminx` | none; S4-19 pins the existing revision |
| S3-13 | publish `molxmpnn 0.2.0a4` and the shim from a pinned tag | nothing to publish for a rename | the release guard and train rule: **S3-12**; the releases themselves are S5-52 and S1-25 |
| S3-14 | migrate mpnn_ext | consumers keep importing `aminx` | none |
| S3-15 | migrate asr and patch its vendored proteinsmc | same; asr's nested clone already guards `aminx` [A27] | none (Q12 files the nested clone as debt) |
| S3-16 | tev_design dependency note | same | none |
| S3-17 | hautespout bump-or-pin decision | the rename reason is gone | S1-28 owns the bump-or-pin decision for the S1 migration |
| S3-19 | `using-molxmpnn` skill, `using-aminx` pointer | the skill keeps its name | none; the hub skill is S5's |
| S3-20 | checkout rename runbook, bathos slug policy, worktree repair | no checkout, slug or `.bth.toml` root changes | none; the hub's slug is Q19 |
| S3-22 | close the deprecation window, decide P1/P2 | no window; P2 (hub takes `aminx`) contradicts revised D2 | S3-01 records the rule that the hub never takes `aminx` |
| S3-23 | PyPI handover mechanics (P2 only) | P2 is void | none |
| S3-25 | flip the shim's `aminx.ebm` row to forward to the EBM project | no shim | S2-18 owns the CHANGELOG pointer for the removed `aminx.ebm` |
| S3-26 | never allocated (the earlier revision skipped the number) | n/a | none |
| S3-27 | reserve `molxmpnn` on PyPI and HF | no `molxmpnn`; the hub names are free [A46, A47] | if Q17 chooses to publish, S3-01's re-open table adds a reservation item |

Retired questions (numbers kept so references stay readable): Q1 (name `molxmpnn`), Q2 (repo strategy),
Q3 (PyPI handover), Q4 (order against S1/S2; the shared S1 Q10 = S2 Q7), Q5 (bathos slug rename; superseded by
Q19), Q6 (shim warning class), Q7 (checkout rename), Q8 (HF weights option), Q10 (rename `browser/aminx-sampler`),
Q14 (soak thresholds), Q15 (legacy Pages URL; superseded by Q21).

Retired contracts: `S3:molxmpnn-repo` and `S3:molxmpnn-manifest-namespace` -> `S3:aminx-identity`;
`S3:aminx-name-freed` -> `S3:hub-identity`; "Rename codemod", "`aminx` compat shim", "Env/provenance key
contract", "Bathos slug policy" (rename form), "Checkout-rename quiescent window", "Release train"
(rename form) -> none, or `S3:release-guard` for the train.

## References

- Recon brief: `.praxia/docs/research/261001_ecosystem-hub-recon-brief.md` (D1-D7, revised D2)
- Spike records: `.praxia/spikes/261001_aminx-identity-and-hub-naming/S11`..`S22` (S20 superseded by S21);
  retired-design spikes: `.praxia/spikes/261001_molxmpnn-rename/S1`..`S10`
- Previous rename's residue: `asr/CLAUDE.md:113`, `scripts/check_model_boundary.sh:18`,
  `proteinsmc/pyproject.toml:20-25`, `proteinsmc/src/proteinsmc/scoring/mpnn.py:14`
- Ecosystem partition: `alphex/.praxia/docs/specs/260814_alphabet-contract.md` (D2/D4)
- GitHub Pages custom domains and verification: docs.github.com, "Managing a custom domain for your GitHub
  Pages site" and "Verifying your custom domain for GitHub Pages" (fetched in spikes S18, S22)
- Sibling specs: S1 `261001_spec-system-unification.md`, S2 `261001_ebm-extraction.md`,
  S4 `261001_xtrax-model-contract.md`, S5 `261001_aminx-hub.md`, S6 `261001_pipeline-editor.md`

## Revision log

Entries before the "no-rename revision" describe the retired rename design (molxmpnn, shim, cutovers). They
are kept verbatim as history and are not a description of this spec's current content.

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

### No-rename revision (261001, user decision revising D2)

Trigger: the user decided that the MPNN package, released on PyPI as `aminx`, keeps that name, and that the hub
is a separate project (working slug `aminx-hub`; the user also floated a name derived from the domain
praxia.science, which works with any repo slug). The spec was rewritten, not patched; sections 1-10 above are
the new content. Changes:

- **Retitled** "aminx identity kept: hub naming and stale-name cleanup"; front matter, owner repos and
  description rewritten; the file keeps the name it was renamed to (`261001_aminx-identity-and-hub-naming.md`).
- **Retired (ids out of the toml block, listed with reasons in section 10):** S3-02, S3-03, S3-04, S3-05, S3-06,
  S3-07, S3-08, S3-10, S3-11, S3-13, S3-14, S3-15, S3-16, S3-17, S3-19, S3-20, S3-22, S3-23, S3-25, S3-26
  (never allocated), S3-27. That is the codemod, the atomic rename PR, the compat env layer, the shim
  distribution, the repo rename, the trusted-publisher changes, the weights cutover, the molxmpnn/shim release
  plumbing and publish, every consumer migration, the `using-molxmpnn` skill, the local checkout rename
  runbook, the deprecation-window close, the PyPI handover and the shim's EBM forwarding.
- **Kept and adapted:** S3-01 (now the hub identity decision, `user_decision = true`), S3-24 (stale-name checker,
  with the hand-written allowlist, plus a new reserved-name mode so the hub can never take `aminx`), S3-09
  (the prxteinmpnn residue, now re-measured: 32 non-history files), S3-18 (proteinsmc guard, now guarding
  `aminx` with the deferred-import form asr's clone already uses, and a dependency on S3-24), S3-21 (public
  surfaces and the docs-URL record that S5-34 needs; README `cuda` extra defect added), S3-12 (re-meant: the
  release guard, dry run and train rule, because S5-52, S1-25 and any S2 release still race for PyPI versions
  through one workflow), S3-28 (assembly check rewritten for the new edges and the retired set).
- **New:** S3-29 (custom domain wiring for the hub, user action, after S5-33).
- **Edge map:** section 2.1 lists every edge the other specs had to S3 items, what it was for, and the
  disposition. Edges that existed only to order work against the rename are retired with no replacement
  (S1-01/02/03/05/17, S1-26/27/28's S3-10/14/15/17, S2-02/03/18, S4-16/19/35, S5-14's and S5-21/34/61's S3-07,
  S5-52's S3-13, S5-55's S3-22/23). Edges that still carry a real need point at kept items: S1-25 and S5-52 at
  S3-12, S5-33 at S3-01, S5-34 at S3-21, S4-22/23/49 at S3-18. Two requested additions: S5-01 gains S3-01; S1-25
  gains S3-12.
- **Questions:** Q9, Q11-Q13 kept; Q1-Q8, Q10, Q14, Q15 retired; Q16-Q22 added (hub slug, PyPI, domain, bathos
  slug, orchestrator collision, docs URL, release order). The shared "ordering against the rename" question
  (S1 Q10, S3 Q4, S2 Q7) no longer exists.
- **Contracts:** `S3:molxmpnn-repo`, `S3:molxmpnn-manifest-namespace` -> `S3:aminx-identity`;
  `S3:aminx-name-freed` -> `S3:hub-identity`; added `S3:release-guard`, `S3:stale-name-guard`,
  `S3:assembly-check`, `S3:hub-domain`.
- **Ledger:** A3, A6, A11, A27, A28, A30, A36 carried; A45-A63 added; every other earlier row retired.
  Spikes S11-S22 ran under this file's stem (S20 superseded by S21). Findings that changed the design:
  S12 showed asr's `vendor/proteinsmc` is a nested clone with its own `.git` (the pre-registered "no `.git`
  entry" expectation failed, A27 still holds, the earlier "plain directory copy" wording is withdrawn) and that
  its guard uses imports deferred into `make_mpnn_score`, which S3-18 now copies; S11 showed PyPI `praxia` is
  already registered and GitHub `maraxen/praxia` is not publicly visible, which entered the collision
  inventory; S18 showed `praxia.science` is a live Cloudflare zone with no apex records and no `hub` label;
  S17 re-measured the prxteinmpnn residue. The docs and DNS spikes needed outbound access to
  `docs.github.com` and `dns.google`, requested for those runs only. A59 stays UNVERIFIED with a recorded
  reason (needs the user's Cloudflare account and a live Pages site).
- **Not done, by design:** no hub name, PyPI name or domain is decided (S3-01 is the user's); no other spec
  was edited, so S3-28's real run is expected to fail until S1, S2, S4 and S5 apply section 2.1; no code was
  touched and no test was run.

Declined: none.

## Revision log: no-rename coherence r1

- **CH1-01** Accepted as a scope statement, not a toml change. The generated plan
  `261001_ecosystem-backlog-dag.md` (231 items, 553 edges, rename-era S3 with retired ids, `molxmpnn` repo label,
  OQ-01..OQ-11, link to the old file name), the hand-made `261001_dag_given_edges.txt`, the generated `INDEX.md`
  and the brief's front-matter description are all stale relative to the six specs, so the plan's earlier
  "0 differences, acyclic" no longer describes them. Section 2.2 now says S3-28 checks only the toml blocks and
  that a green run says nothing about the plan; section 4.6 and the S3-28 gate add a `--plan` mode (C1-C7 over
  the plan plus an empty edge-set diff against the blocks) to run after the plan is regenerated. The plan must
  not be used to file backlog items until then. Regenerating the plan, the given-edges file, the brief
  front matter and `INDEX.md` (via `docs(action="index")`) belong to their owners, not this file. Hand check
  recorded: every `depends_on` in the six blocks resolves and I found no cycle, but that is not a recorded
  script run. TOML block unchanged (ids, edges and fields); S3-28's title is unchanged and only its gate text grew.

## Revision log: no-rename coherence r2

- **R2-02** Accepted. The `--reserved aminx` real-tree result is now exactly two hits, `pyproject.toml:6` and the
  `src/aminx` package directory (section 4.4 point 3, the section 7 S3-24 row, the S3-24 toml gate). Red fixtures
  kept; added a green fixture (a synthetic hub tree with `src/executors` and a `package.json` named `aminx-hub`)
  proving exact-match semantics. S5-01's gate is unchanged, since the hub tree has no `src/aminx`.
- **R2-05** Accepted. Section 2.2 and the S3-28 gate name the owner (the integrator step after convergence, not a
  backlog item) and the ordered steps: regenerate the plan (expect 211 items), regenerate the given-edges file,
  `docs(action="index")` for `INDEX.md`, fix the brief's description and table row, then `check_assembly.py --plan`
  must exit 0 before any filing. The S3-28 gate and the section 2.2 closing paragraph now state that the real-tree
  run is expected to exit 0 once converged, not a one-off to-do list. TOML: only the S3-24 and S3-28 gate text
  changed; ids, edges and fields are unchanged.
