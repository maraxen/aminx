---
title: RunSpec scaffolding remediation, migration-map re-authoring, and xtrax transforms adoption
description: Implementable remediation spec for the 260707 audit's three remaining workstreams (delete 2 dead RunSpec sub-configs, re-author the stale migration map + anti-drift gate, xtrax safe_scan/safe_map/Engine.fit() adoption decisions); reviewed through one full adversarial challenger/defender cycle.
status: ready
task_id: 260827_runspec-scaffolding-remediation-spec
date: '260827'
backlog_ids: ''
adversarial_review: ''
---
**task_id:** `260827_runspec-scaffolding-remediation-spec`
**Status:** READY — reviewed
**Remediates:** `.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md` ("Remediation", items 1/2/4)
**Also supersedes claims in:** `.praxia/docs/audits/260817_aminx-deeper-work-assessment.md` §4, §6
**Baseline:** worktree `/home/marielle/projects/aminx/.claude/worktrees/wt-20260826-170332`, branch `audit/260826-chain-selection-vendor-superset`

**Revision note:** this spec incorporates one full adversarial review cycle (18 challenges, each independently re-verified against the tree); all engineering verdicts survived, argumentation and gate precision were tightened.

---

## 0. Re-baseline: the audit is stale, and one of its central claims is wrong

**Read this section before reading anything else.** Two of the three workstreams have materially different scope than the brief and the source audit describe, because the tree moved and because the audit mis-attributed a mechanism. Every claim below is file:line-checkable in the worktree named above.

### 0.1 Three of the five dead sub-configs are already deleted

`BatchingConfig` (9 fields), `TiedPositionsConfig` (5), and `AveragingConfig` (4) — 18 of the audit's 27 dead fields — **no longer exist**. `src/aminx/run/spec.py` now defines 8 sub-configs, not 11. The deletion is documented in two places in-tree:

- `src/aminx/run/spec.py:100-108` (`SamplingConfig` docstring): *"the prior attempt at this (`TiedPositionsConfig`/`AveragingConfig`/`BatchingConfig`) went 100% dead and was deleted (`dd0e952`) rather than fixed -- fields live directly on `SamplingConfig` this time, not a separate sub-config, since that's the one that stuck."*
- `CHANGELOG.md:203-213` (Unreleased → Removed), which also records the deliberate decision **not** to remove `GridLineageConfig`/`LigandConfig`, on the grounds that *"each has one live field (`grid_mode`, `model_family`) plus existing partial-migration fallback logic worth finishing rather than discarding."* §0.3 below shows that justification does not survive checking.

Consequences:
- The 260817 assessment's §4 claim — *"**27 of 72** `RunSpec` sub-config fields are populated at construction and never read… every finding still holds today"* (`260817_aminx-deeper-work-assessment.md:233-237`) — **is stale against this worktree.** The true remaining figure is **11 fields across 2 sub-configs** (`GridLineageConfig` 6 + `LigandConfig` 5), all of which §0.3 shows to be dead.
- The precedent the audit asked about (its Open Question 3: *"is `average_encoding_mode`'s dead-on-purpose comment a convention?"*) is answered by events: the project's actual convention turned out to be **delete, and re-land only the fields that have a real consumer, directly on `SamplingConfig`.** That is the precedent WS-A should follow.

**Blocking precondition — verify by exit code, not by text.**

```bash
git merge-base --is-ancestor dd0e952 HEAD; echo "exit=$?"
```

- `exit=0` → `dd0e952` is an ancestor of the target branch. §0.1's re-baseline holds. **Proceed.**
- non-zero → the deletion commit is **not** on this branch. §0.1 is wrong, WS-A's scope reverts toward the audit's original 27 fields, and §1/§2 must be re-scoped before any edit. **STOP** (risk R1).

Prefer exit-code-based git inspection throughout this spec's execution. A text-transforming proxy in this environment has been observed to corrupt `git log --format=...` / `--oneline` output (fabricated hashes and subjects); `git merge-base --is-ancestor` and `git rev-parse --verify` are unaffected because they communicate through the exit status.

### 0.2 `jax.lax.map` natively supports both behaviours the audit attributes to aminx's fork

This is the finding that changes WS-C's recommendation. From `.venv/lib/python3.14/site-packages/jax/_src/lax/control_flow/loops.py`:

- **`batch_size=0` is a documented vmap sentinel *in JAX itself*.** `loops.py:2691`: *"``batch_size=0`` is equivalent to applying a ``vmap``. That is, it uses a full batch."* Implemented at `loops.py:2647-2648` (`if batch_size == 0: num_batches, remainder = 0, leaves[0].shape[0]`), which routes the whole array into the remainder branch, which is vmapped (`loops.py:2726`).
- **Non-divisible cardinalities are natively supported.** `loops.py:2685-2689`: *"If the axis is not divisible by the batch size, the remainder is processed in a separate ``vmap`` and concatenated to the result."* Implemented in `_batch_and_remainder` (`loops.py:2643-2663`).

Therefore:

| Audit claim (`260707…md:198-213`) | Actual |
|---|---|
| aminx's fork "*added* a `batch_size == 0` 'always vmap' branch **on top of xtrax's semantics**" | aminx's `safe_map.py:49` `batch_size == 0` branch is a **redundant fast path** for behaviour `jax.lax.map` already implements. Deleting the branch alone changes nothing observable: `batch_size=0` would still vmap, via `loops.py:2647`. It substitutes a `lax.map` call (which reduces to a bare `vmap` over one full-range slice) for a Python-level short-circuit — cosmetic, not a behaviour risk, and not worth spending a change on (§4.3). |
| "passing `0` hits `n % batch_size` and raises `ZeroDivisionError`… at the `lax.map` boundary" | The `ZeroDivisionError` originates in **xtrax's own added pre-check** (`xtrax/transforms/map.py:33`, `if n % batch_size != 0`), *before* `lax.map` is reached (`map.py:40`). It is not a `lax.map` boundary behaviour. |
| Adopting xtrax's `safe_map` is "a strictly cleaner fix… the one most consistent with how xtrax itself expects this primitive to be called" | Adoption imports a **new hard failure mode**: `xtrax/transforms/map.py:33-37` raises `ValueError` whenever `n % batch_size != 0`, which both `jax.lax.map` and aminx's fork accept. This is a behaviour regression on production shapes, not defence-in-depth. |

Concretely: a run with `num_samples=100` and a planner-chosen `samples_bs=32` works today and would raise `ValueError` after adoption. `100 % 32 == 4`.

**The existing test suite cannot catch this.** Every `lax.map`-path case in `tests/utils/test_safe_map.py` happens to be divisible: `n=10,bs=5` (`:34`), `n=10,bs=1` (`:38`), `n=10,bs=2` (`:58`), `n=6,bs=2` (`:74`). A non-divisible case must be added as gate zero of WS-C (Step C0).

### 0.3 `GridLineageConfig` and `LigandConfig` are dead in production too — their "live" fields are structurally unreachable guards

`CHANGELOG.md:210-212` kept these two because each "has one live field." Both live reads are in `src/aminx/run/run_spec_portable_json.py` — and a grep for `run_spec\.(grid|ligand)\.` across all of `src/` returns **exactly these two lines and nothing else**:

- `:145` — `if run_spec.grid.grid_mode:` → raise "not representable in v2 wire format"
- `:153` — `if run_spec.ligand.model_family == "ligandmpnn":` → same

Tracing the callers:

1. `run_spec_portable_to_dict` has exactly **one** production caller: `src/aminx/cli.py:1620`, inside `aminx spec portable-roundtrip` (`cli.py:1605-1621`).
2. That caller's input is the return of `run_spec_portable_from_dict` on the preceding line (`cli.py:1619`); the command's own input is raw JSON read from a file (`cli.py:1615`), so no flat `SamplingSpecification` is ever in scope there.
3. `run_spec_portable_from_dict` always returns `_placeholder_run_spec(...)` (`run_spec_portable_json.py:256`).
4. `_placeholder_run_spec` **hardcodes** `grid_mode=False` (`:119`) and `model_family="proteinmpnn"` (`:112`).

So in production, both guards are evaluated against constants that can never trip them. No `build_run_spec()`-produced `RunSpec` — the only kind carrying real grid/ligand values — is passed to `run_spec_portable_to_dict` anywhere in `src/`. The only code that reaches either guard is tests that synthesise the state with `eqx.tree_at` (`tests/run/test_run_spec_portable_json.py:101-116`, `:125-139`, and `tests/cli/test_inputs_integration.py:338-360`).

**Net: 11 of 11 remaining sub-config fields have zero production read sites.** `GridLineageConfig` and `LigandConfig` are 100% dead, not 5/6 and 4/5 dead.

### 0.4 The portable-JSON import direction has an unimplemented pre-registered guard

`run_spec_portable_from_dict`'s docstring says *"unknown top-level keys are ignored"* (`run_spec_portable_json.py:183`), and the implementation honours that: it validates `version ∈ {1,2}` (`:184-187`) and then reads only `multistate`/`resource`/`precision`/`io`. Any additional top-level key is silently discarded.

**Stating the gap precisely, because an overstated version of it is easy to reach.** A *v3* payload cannot cause silent data loss here: `:185-187` raises `ValueError` on any version outside `_SUPPORTED_PORTABLE_VERSIONS = {1, 2}` **before** any field parsing runs. The real, narrower gap is that a payload declaring `version: 2` while carrying extra top-level blocks (e.g. `grid`, `ligand`) is accepted and the extra blocks are dropped without a word. No producer of such a payload exists today. The guard is therefore **cheap forward insurance against a v3 authorship error** — someone extending the format who forgets to bump `_SUPPORTED_PORTABLE_VERSIONS`, or who hand-writes a v2 file with v3 content — not the closure of an active data-loss path.

It is also a **pre-registered acceptance criterion that was never implemented.** `.praxia/docs/specs/260611_runspec-unification.md:91` (RS-8) states two guards, one per direction:

> **Guard (RS-8):** `run_spec_portable_to_dict` must **raise** `ValueError` when serializing a spec with `grid.grid_mode=True` or `ligand.model_family='ligandmpnn'` until v3 exists (the lossy boundary — v2 silently drops these fields today). `run_spec_portable_from_dict` must reject unknown/lossy top-level keys instead of ignoring them.

The export-direction guard exists (and is unreachable, §0.3). The import-direction guard does not exist at all. Step A3 implements it, and makes an explicit, on-the-record decision about the export-direction one, which A1 renders structurally unsatisfiable.

---

## 1. Scope

### In scope

| ID | Workstream |
|---|---|
| **WS-A** | Decide and execute fix-or-abandon for the 2 remaining write-only `RunSpec` sub-configs (`GridLineageConfig`, `LigandConfig`), per-sub-config, with justification. |
| **WS-B** | Re-author `.praxia/docs/plans/260614_runspec-migration-map.md` from current code state, and install a mechanical anti-drift gate so it cannot silently rot again. |
| **WS-C** | Migrate `aminx.utils.safe_scan` → `xtrax.transforms.safe_scan`; make an evidence-backed **decision** on `safe_map` (adopt / adopt-with-upstream-change / keep fork with documented rationale); produce a written feasibility assessment (no migration) of `xtrax.engine.Engine.fit()` vs `training/trainer.py`. |

### Out of scope — named explicitly so review does not treat these as omissions

| Item | Why out | Where it belongs |
|---|---|---|
| Consolidating the three divergent multistate paths in `host/runner.py` / `sampling/multistate_poe.py` | Separate scope, already filed | **praxia debt #1500** |
| Replacing hardcoded `average_node_features` averaging with a composable fusion transform | Explicitly deferred pending in-development xtrax caching work | **praxia debt #1501** — see §2.4 for the landing spot so this need not be rediscovered |
| `xtrax.safety.safe_norm` / `safe_reciprocal` adoption across the 10+ ad-hoc epsilon guards | Style/consolidation only; the audit found no wrong or missing guard (`260707…md:225-233`) | Future backlog item; not blocking anything |
| Actually migrating `trainer.py` onto `Engine.fit()` | Brief scopes WS-C's Engine leg to feasibility assessment only | Follow-on, gated on WS-C3's verdict |
| Whether `IOConfig`/`ResourceConfig`/`MultistateConfig`/`PrecisionConfig` should survive a retirement of the portable-JSON wire format | They are retained under §2.1 clause 2 today, which is dispositive for this spec; the question of what happens if that clause's premise is removed is a different, larger question | §8 open question 1; WS-B Step B2 records it |
| Fixing `RunSpecification`-family flat fields themselves | Nothing here proposes changing the public dataclass façade | — |
| `tiling/dispatch.py`'s stale docstring and the other doc-drift items in `260817…md:252-259` | Cheap and unrelated to these three workstreams | Separate doc-hygiene pass |

### Non-goals (behavioural)

- **No production behaviour change is intended by WS-A's deletions or by WS-B.** Both are dead-code removal and documentation. Any observable behaviour difference is a bug in the change. (Step A3 does introduce one deliberate, recorded Breaking Change — see §2.5.)
- **WS-C's `safe_scan` migration is intended to be behaviour-preserving except for strictly stronger input validation** (§4.2). WS-C's `safe_map` leg is decision-only in this spec; if the decision is ever "adopt," that adoption is a *separate* spec, because §0.2 shows it is a breaking change.

---

## 2. WS-A — `RunSpec` sub-config fix-or-abandon

### 2.1 The decision rule

The RS track's purpose (`.praxia/docs/specs/260611_runspec-unification.md:50-51,99-100`) is a pytree-shaped config: `RunSpec` subclasses `xtrax.run.RunSpec` (`src/aminx/run/spec.py:21,138`) and its sub-configs are `eqx.Module`s with `eqx.field(static=True)` leaves, so they can cross a jit boundary. The flat `SamplingSpecification` cannot — it holds loader handles and `Path`s.

That gives a principled test for *why a sub-config exists*, and it is two-clause:

> **Rule.** A sub-config earns a seat on `RunSpec` iff **either**
> **(1)** at least one of its fields has a trace-adjacent consumer — read at or near a traced/jitted boundary — **or**
> **(2)** at least one of its fields is required to cross the portable-JSON wire boundary (`run_spec_portable_to_dict`).
> A sub-config satisfying neither is scaffolding and should be deleted. Fields read only in host-side orchestration, metadata-dict construction, or loader configuration do not by themselves earn a seat.

Clause 2 is not a concession bolted on — it is the same distinction B3 formalises as its `SERIALIZATION_ONLY` bucket (§3.3), and the two must use one vocabulary or they will drift apart.

**Applying it to the current roster:**

| Sub-config | Clause 1 (trace-adjacent) | Clause 2 (wire) | Verdict |
|---|---|---|---|
| `SamplingConfig` | **yes** — `host/kernel_dispatch.py:500,533` traced dispatch; `host/plan.py:941-950` | — | retain |
| `PlannerTopology` | **yes** — drives dispatch-path selection | — | retain |
| `IOConfig` | no — *all* live reads are host-side (`prep.py:129-130`, `runner.py:183`, `streaming.py:79`, `multistate_poe.py:610`) | **yes** — `run_spec_portable_json.py:163-165` | retain |
| `ResourceConfig` | no | **yes** — `:172-177` | retain |
| `MultistateConfig` | no | **yes** — `:167-171` | retain |
| `PrecisionConfig` | no | **yes** — `:178` | retain |
| `GridLineageConfig` | **no** | **no** | **delete** (§2.2) |
| `LigandConfig` | **no** | **no** | **delete** (§2.3) |

Grid and ligand fail a weaker test than either clause: per §0.3 they have **no reachable read site at all**. The rule is what says they should not be *fixed* into having one; their deadness is what says they must go now.

One clarification so the rule is not over-read: it governs whether a *sub-config* exists, not whether every field inside a retained one is live. `IOConfig.output_h5_path` and `IOConfig.cache_path` have real, unique readers (`streaming.py:79`, `runner.py:183`, `multistate_poe.py:610`, `prep.py:129-130`) and are live by the ordinary definition — B3 classifies them `MIGRATED`. Per-field classification is B3's job; §2.1's job is the seat.

This rule is not invented here — it is what the tree already did. `SamplingConfig` survived because it satisfies clause 1. The three sub-configs deleted in `dd0e952` satisfied neither.

### 2.2 `GridLineageConfig` — **ABANDON (delete)**

**Definition:** `src/aminx/run/spec.py:63-71`. **Constructed:** `spec.py:328-335`. **6 fields.**

Read-site inventory (all reads in `src/`, verified by grep):

| Field | Sub-config reads | Flat reads |
|---|---|---|
| `grid_mode` | `run_spec_portable_json.py:145` — structurally unreachable (§0.3) | `_sampling_grid_lineage.py:14`, `streaming.py:83,227`, `runner.py:245`, `multistate_poe.py:614,663` |
| `campaign_mode` | none | `streaming.py:115` |
| `job_id` | none | `_sampling_grid_lineage.py:28` |
| `chunk_id` | none | `_sampling_grid_lineage.py:24` |
| `sample_start` | none | `_sampling_grid_lineage.py:20` |
| `sample_count` | none | `_sampling_grid_lineage.py:16` |

**Reasoning for abandon over fix — four independent arguments:**

1. **The decision rule (§2.1) excludes it on both clauses.** Every flat read above is host-side. `_sampling_grid_lineage.py:94-113` builds a `dict[str, str]` for a lineage hash. `streaming.py:83-86`, `runner.py:245`, and `multistate_poe.py:614-617,663` build metadata dicts written to Zarr `.attrs`. None is inside a traced function; none benefits from being a static pytree leaf; and no grid field appears in `run_spec_portable_to_dict`'s output.

2. **There is already a second, better-guarded home for exactly these fields.** `host/spec_partition.py:46-56` defines `CAMPAIGN_OWNED_KEYS = {grid_mode, samples_chunk_size, job_id, chunk_id, sample_start, sample_count, output_h5_path}` — five of `GridLineageConfig`'s six fields, plus `grid_mode`. That module has an import-time exhaustiveness assertion (`spec_partition.py:111-157,236`) enforcing that the set stays in sync with `dataclasses.fields(SamplingSpecification)`. `GridLineageConfig` is an unenforced third copy of a concept that already has one enforced copy and one source of truth. Completing the migration would make it a *load-bearing* third copy.

3. **Fixing collides head-on with debt #1500.** Three of the `grid_mode` flat reads (`streaming.py:83`, `runner.py:245`, `multistate_poe.py:614,663`) are inside near-identical metadata-dict literals — `{"schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION, "model_family": …, "ligand_conditioning": …, "sidechain_conditioning": …}`. That duplication *is* part of what #1500 exists to consolidate. Migrating those reads now means editing three blocks that #1500 will rewrite or delete, then re-reviewing the same lines twice. Deleting the sub-config touches none of them.

4. **`CHANGELOG.md:210-212`'s stated reason for keeping it does not hold — in both of its clauses.** It cites "one live field (`grid_mode`)"; §0.3 shows that read is unreachable in production. It also cites "existing partial-migration fallback logic worth finishing"; the counter-argument paragraph immediately below identifies that logic precisely (`_sampling_grid_lineage.py:16,28`) and shows it is already correct under §2.1's rule, with nothing left to finish. Both premises of the earlier decision are falsified, so the decision should be revisited — which is what this spec does.

**Counter-argument, stated and answered.** `_sampling_grid_lineage.py:16` and `:28` already mix idioms on a single line each:
```python
sample_count = int(spec.sample_count if spec.sample_count is not None else spec.run_spec.sampling.num_samples)
job_id = spec.job_id or f"grid_{spec.run_spec.sampling.random_seed}"
```
One could argue this half-migrated state is itself the defect — and this is exactly the "partial-migration fallback logic worth finishing" the CHANGELOG appeals to — so that finishing is cheaper than justifying permanent inconsistency. **Answer:** the mixture is not accidental and is not inconsistent under §2.1's rule. `num_samples`/`random_seed` are `SamplingConfig` fields with real traced consumers (clause 1); `sample_count`/`job_id` are host-side lineage with neither clause. The line reads oddly but each half is already on the correct side of the rule. There is no unfinished migration here — only a correctly-mixed one. Deleting `GridLineageConfig` leaves these two lines untouched and correct.

### 2.3 `LigandConfig` — **ABANDON (delete)**

**Definition:** `src/aminx/run/spec.py:53-60`. **Constructed:** `spec.py:319-326`. **5 fields.**

| Field | Sub-config reads | Flat reads (flat name, where it differs) |
|---|---|---|
| `model_family` | `run_spec_portable_json.py:153` — structurally unreachable (§0.3) | `prep.py:48,55`, `_sampling_helper.py:255`, `_sampling_grid_lineage.py:94,111`, `streaming.py:84`, `campaign.py:95`, `multistate_poe.py:615` |
| `use_side_chain_context` | none | `prep.py:166,177` (flat name: `ligand_mpnn_use_side_chain_context`) |
| `ligand_conditioning` | none | `_sampling_helper.py:280`, `_sampling_grid_lineage.py:95,112`, `streaming.py:85`, `campaign.py:97,115`, `multistate_poe.py:616` |
| `sidechain_conditioning` | none | `_sampling_helper.py:315`, `prep.py:166,178`, `runner.py:957`, `_sampling_grid_lineage.py:96,113`, `streaming.py:86`, `campaign.py:96,116`, `multistate_poe.py:617` |
| `context_path` | none | `_sampling_helper.py:269,271` (flat name: `ligand_context_path`) |

**Reasoning for abandon — the case here is stronger than for grid:**

1. **Every real `model_family` read is flat and host-side, and the migration map's own "key finding" shows the sub-config path was never built out.** `prep.py:48` uses `model_family` to select a checkpoint-registry entry (`entry.get("model_family") == spec.model_family`); `prep.py:166,177-178` uses `ligand_mpnn_use_side_chain_context` to build loader kwargs. Both are loader configuration, read nowhere near a trace, and neither crosses the wire — so `LigandConfig` fails both clauses of §2.1. This is not a case where the migration was half-done and needs finishing: the map itself, at `260614_runspec-migration-map.md:220-222`, records that when it was written *"only one RunSpec read site today"* existed at all (`streaming.py:310`, for `multistate.n_states`), and that "all other 65+ field reads are flat." The `ligand.*` reads it optimistically lists as "already migrated" (`:143`) never materialised. Independently verified: `run_spec.ligand.` has exactly **one** read site anywhere in `src/` — the unreachable guard at `run_spec_portable_json.py:153` (§0.3). There is no partial migration to complete; there is a table entry that was aspirational when written and is now six weeks of drift old (WS-B).

2. **Two fields were renamed in the sub-config** (`ligand_mpnn_use_side_chain_context` → `use_side_chain_context`, `ligand_context_path` → `context_path`). Completing the migration means every reader must learn a second name for the same value, with no compensating benefit. Rename-during-migration is also the most common way a mechanical migration silently reads the wrong field.

3. **`sidechain_conditioning` has 11 flat read sites across 7 files**, including `runner.py:957` inside a hot sampling path and `prep.py:166,178` inside loader setup. This is the single largest fan-out of any remaining dead field. Migrating it — which is what the "fix" half of fix-or-abandon means — is the most expensive edit available in WS-A and buys the least, since none of the sites is trace-adjacent. The fan-out count is the price tag on the rejected alternative, quoted here for exactly that reason.

4. **Same #1500 collision as §2.2 argument 3** — `streaming.py:84-86`, `campaign.py:95-97`, `multistate_poe.py:615-617`, `_sampling_grid_lineage.py:94-96,111-113` are the same duplicated metadata literal.

5. **`CHANGELOG.md:210-212`'s "partial-migration fallback logic worth finishing" has no ligand-side referent at all.** For grid, that clause at least points at something real (§2.2's counter-argument paragraph). For ligand it points at nothing: there are zero `run_spec.ligand.*` reads in `src/` outside the one unreachable guard, so there is no fallback idiom, half-migrated or otherwise. The clause was written once, covering both sub-configs, and is only ever true of one of them.

**Counter-argument, stated and answered.** `_sampling_helper.py:255-315`'s ligand-context loading is arguably "near a boundary" — it prepares arrays that then enter a traced kernel. **Answer:** it prepares them by reading a file from `spec.ligand_context_path` (`:269-271`). File IO is definitionally host-side; the *outputs* cross the boundary, the *config* does not. Under §2.1 clause 1 the config is not trace-adjacent.

### 2.4 `AveragingConfig` — already deleted; do **not** resurrect it. Landing spot for praxia debt #1501

The brief asked for a call on `AveragingConfig` and warned against a deletion that would force debt #1501's implementer to reinvent the sub-config from scratch. **`AveragingConfig` was already deleted before this spec** (§0.1). So the actionable guidance is not "delete or keep" but "where does #1501 land, now that the sub-config is gone?" — recorded here so it need not be rediscovered:

- **The thing #1501 replaces** is `src/aminx/host/plan.py:960-970`: a hardcoded `bool → class` branch,
  ```python
  if getattr(spec, "average_node_features", False):
      from aminx.host.averaging import ArithmeticMeanEncodingFusion
      stage_set = eqx.tree_at(lambda s: s.encoding_fusion, stage_set,
                              ArithmeticMeanEncodingFusion(), is_leaf=lambda x: x is None)
  ```
  documented at `plan.py:916-917`. `ArithmeticMeanEncodingFusion` and the `average_encoding_mode` ladder live in `src/aminx/host/averaging.py:77,197-199`.
- **The composable slot #1501 wants already exists.** `RunSpec.encoding_fusion: EncodingFusionFn | None` (`src/aminx/run/spec.py:149`) and its `decoding_fusion` sibling (`:150`) are already caller-supplied composable transforms, already wired through `plan.py:972-982` by the identical `eqx.tree_at` mechanism. #1501's "pure JAX fn or a venv-resolvable path to one" is therefore **a resolver in front of the existing `encoding_fusion` slot**, not a new sub-config.
- **`average_node_features` is a live flat field**, read at `plan.py:960` and `runner.py:532`, declared at `run/specs.py:550,594`, CLI-exposed at `cli.py:679,796,1198,1308`. It is not scaffolding and must not be touched by WS-A.
- **`average_encoding_mode` is already soft-deprecated** at the constructor: `run/specs.py:164-168` pops the kwarg and emits a deprecation warning, describing the intended `average_encoding_mode → encoding_aggregation_fn` migration (`:153`). That is the same direction #1501 points.
- **Explicit recommendation to #1501's implementer:** do **not** reintroduce an `AveragingConfig` sub-config. It went 100% dead once (`spec.py:100-108`). Extend `SamplingConfig` (`spec.py:97-135`) or the existing `encoding_fusion` slot, matching the pattern the project already found to stick.

### 2.5 WS-A steps

**Precondition (blocking):** `git merge-base --is-ancestor dd0e952 HEAD` exits 0 (§0.1). If not, stop and re-scope per risk R1.

---

**Step A1 — Delete `GridLineageConfig` and `LigandConfig` and their construction; update the one test outside `tests/run/` that names them.**

Files: `src/aminx/run/spec.py` (modify), `tests/cli/test_inputs_integration.py` (modify)

In `src/aminx/run/spec.py`:
- Delete class `LigandConfig` (`:53-60`) and class `GridLineageConfig` (`:63-71`).
- Delete fields `ligand` (`:144`) and `grid` (`:145`) from `RunSpec`.
- Delete their construction in `build_run_spec` (`:319-326`, `:328-335`) and the now-unused local `ctx_path` (`:319`).
- Delete `ligand=ligand,` / `grid=grid,` from the `RunSpec(...)` return (`:381-382`).

In `tests/cli/test_inputs_integration.py`:
- **Delete `test_portable_guards_still_work` in its entirety** (`:338-360`). Its subject — the two `to_dict` guards — is being removed; there is no rewrite that preserves its meaning, and A3 installs the replacement coverage on the `from_dict` side. Note its `from aminx.run.spec import GridLineageConfig` is a *function-local* import (`:341`), so leaving it would break at **run** time, not collection time — which is precisely why it is easy to miss.
- Change `test_portable_roundtrip_succeeds`'s final assertion (`:325`, `assert roundtripped.ligand.model_family == "proteinmpnn"`) to something that survives: assert on a field the wire format actually carries, e.g. `roundtripped.multistate.mode == baseline_v2_spec["multistate"]["mode"]`.

Scope estimate: ~35 LOC deleted in `spec.py`, ~25 LOC deleted in the test.

**Gate A1:**
```bash
uv run ruff check src/aminx/run/spec.py            # no F401/F841 on removed locals
uv run ty check                                     # must be clean
uv run pytest tests/run/ tests/cli/test_inputs_integration.py -q
rg -n 'tree_at\([^)]*\.(grid|ligand)|getattr\([^,]*run_spec' src tests scripts
```
The pytest target is deliberately wider than `tests/run/`: `tests/cli/test_inputs_integration.py` is the only file outside `tests/run/` that constructs these classes, and a `tests/run/`-only gate would pass on a tree that is broken there.

**What `ty check` does and does not prove.** It proves no stale *static* attribute access to `.grid`/`.ligand` survives anywhere, including in code this spec did not enumerate. It has **zero visibility** into dynamic access. `RunSpec` is an `eqx.Module`; deleting two fields changes its treedef, and any consumer using `eqx.tree_at`, `tree_flatten`, or `getattr` against a `RunSpec` is invisible to the type checker — `tests/cli/test_inputs_integration.py:346-357` is exactly such a call. The `rg` line above is therefore part of the gate block, not a suggestion in the risk table; it must return only hits that A1 itself is removing.

---

**Step A2 — Update `run_spec_portable_json.py` construction sites, and fix two latent defects in the lines being touched.**

Files: `src/aminx/run/run_spec_portable_json.py` (modify)
- Remove `GridLineageConfig`/`LigandConfig` from the import block (`:24-33`).
- Delete their construction inside `_placeholder_run_spec` (`:111-125`) and the corresponding `RunSpec(...)` kwargs (`:135-136`).

**Two pre-existing defects live in the exact lines this step edits. Do not edit around them silently.** Both are latent today (no consumer reads either value), which is why they have survived, but Step A2 is rewriting this constructor and is the natural place to close them:

1. **`_placeholder_run_spec` passes no `sampling=` kwarg** (`:127-139`). `RunSpec.sampling` is declared `field(default_factory=lambda: None)` (`spec.py:148`), so **every `from_dict`-produced `RunSpec` has `sampling is None`** — a shape `build_run_spec` never produces. Any future consumer that reaches for `.sampling` on a deserialized spec gets `None`, not a default `SamplingConfig`.
2. **`_placeholder_run_spec` passes `carry_specs={}`** (`:130`) where `build_run_spec` passes `[]` (`spec.py:376`) — a dict where the rest of the codebase has a list.

**Required handling — pick one and record it:**
- **(preferred)** Fix both in this commit: add a `sampling=SamplingConfig(...)` built from the same defaults `build_run_spec` would produce for a bare spec, and change `carry_specs={}` → `carry_specs=[]`. Add one test asserting `run_spec_portable_from_dict(baseline).sampling is not None` and `isinstance(..., list)`.
- **(acceptable)** Scope out explicitly by filing a praxia debt item naming both, referencing `run_spec_portable_json.py:127-139` and `spec.py:148,376`, and add a code comment at the constructor pointing at it.

Silently leaving them unremarked is not acceptable — the next reader of this constructor will assume the divergence was reviewed.

Scope estimate: ~20 LOC deleted, ~8 LOC added if the fix path is taken.

**Gate A2:** `uv run ty check` clean; `uv run pytest tests/run/test_run_spec_portable_json.py -q` (will still fail on the two guard tests — Step A3 fixes those; **A2 and A3 land as one commit**).

---

**Step A3 — Implement RS-8's import-direction guard, and decide on the record about the export-direction one.**

Files: `src/aminx/run/run_spec_portable_json.py` (modify), `tests/run/test_run_spec_portable_json.py` (modify)

**The change:**
- Delete both guards from `run_spec_portable_to_dict` (`:144-158`) — they are unreachable (§0.3) and, after A1, unwritable (the fields they read no longer exist).
- In `run_spec_portable_from_dict`, after the version check (`:185-187`), add an unknown-block guard: raise `ValueError` if the payload carries any top-level key outside the v2 vocabulary (`version`, `io`, `multistate`, `resource`, `precision`) — naming `grid` and `ligand` explicitly in the message as "requires wire format v3, not yet defined."
- Update the module docstring (`:1-16`) and `run_spec_portable_from_dict`'s docstring (`:183`, currently *"unknown top-level keys are ignored"*) to state the new contract.

**The decision that must be recorded, not glossed.** RS-8 (`260611_runspec-unification.md:91`) pre-registered **two** acceptance criteria, one per direction. This step implements the import-direction one and **retires the export-direction one.** That is a net trade, not a relocation, and it must be stated as such: once A1 removes `RunSpec.grid`/`.ligand`, `run_spec_portable_to_dict` structurally *cannot* detect that its input describes a grid or ligandmpnn run — the information is no longer in the object it receives.

**Chosen resolution: accept the loss (option b), with the following argument on the record.**

`run_spec_portable_to_dict` has exactly one production caller, `cli.py:1620`, and that caller's input is `run_spec_portable_from_dict(raw)` from the line above (`cli.py:1619`), whose own input is raw JSON read from a file (`cli.py:1615`). **No flat `SamplingSpecification` is in scope anywhere in that command** — there is nothing to consult for the originating `grid_mode`/`model_family` even if a guard wanted to. The alternative (option a: pass the originating flat spec into `to_dict` so it can guard on `spec.grid_mode`/`spec.model_family`) would require changing `to_dict`'s signature to add a parameter that its only caller cannot supply. That is not a defensible shape.

Furthermore, once the import-direction guard lands, the export-direction property is **implied for the actual call graph**: a v2 payload carrying `grid`/`ligand` now raises at `from_dict`, so a `RunSpec` reaching `to_dict` via the only path that reaches it can never carry grid/ligand content. The criterion is not silently dropped; it is discharged by the other guard for every reachable case, and genuinely lost only for a hypothetical future caller that feeds `to_dict` a `build_run_spec`-produced spec — which no code does today and which, after A1, would be carrying no grid/ligand data to lose.

**This retirement must appear in three places:** the CHANGELOG entry (A4), the B2 decision record (§3.2 Part 1's history), and a one-line note added to `260611_runspec-unification.md:91` marking the export-direction criterion superseded with a pointer here.

Scope estimate: ~25 LOC net.

**Gate A3:**
```bash
uv run pytest tests/run/test_run_spec_portable_json.py tests/cli/test_inputs_integration.py -q
```
Rewrite `test_portable_to_dict_raises_on_grid_mode` (`tests/run/test_run_spec_portable_json.py:94-116`) and `test_portable_to_dict_raises_on_ligandmpnn` (`:118-139`) as `from_dict` guard tests — feed a dict with a `"grid": {...}` / `"ligand": {...}` block, assert `pytest.raises(ValueError, match="v3")`. These tests get *stronger*: they now exercise a path production can reach, which the originals could not. Also drop the now-unused `GridLineageConfig`/`LigandConfig` imports from the test module.

**Additional gate — must be added, does not exist today:** a regression test asserting that a v2 payload with an arbitrary unknown top-level key (not just `grid`/`ligand`) is rejected rather than ignored. Without it, Step A3's whole justification is untested.

---

**Step A4 — CHANGELOG.**

Files: `CHANGELOG.md` (modify) — extend the existing Unreleased → Removed entry at `:203-213`, whose final sentence (*"`GridLineageConfig` and `LigandConfig` were NOT removed — each has one live field…"*) is now false and must be corrected in place, with a pointer to §0.3's tracing. Under Breaking Changes, record **both**: (1) `run_spec_portable_from_dict` now rejects unknown top-level keys (a payload that previously round-tripped now raises), and (2) `run_spec_portable_to_dict` no longer raises on grid/ligand specs, with RS-8's export-direction criterion retired and the §2.5-A3 argument summarised in one sentence.

**Gate A4:** manual read; `uv run ruff check .` unaffected.

---

**WS-A completion gate (all steps):**
```bash
uv run ty check && uv run ruff check .
uv run pytest tests/run/ tests/cli/ tests/host/ -q
rg -n 'run_spec\.(grid|ligand)\.|GridLineageConfig|LigandConfig' src tests    # expect: no hits
rg -n 'tree_at\([^)]*\.(grid|ligand)|getattr\([^,]*run_spec' src tests scripts
```
Plus the **behaviour-invariance argument**, which review should demand explicitly: no production code path read `.grid.*` or `.ligand.*` (§0.3), so deleting them cannot change behaviour. `ty check` proves the no-static-reader claim mechanically; the two `rg` lines cover the dynamic/treedef surface `ty` cannot see. State all three in the PR description, and state Step A3's Breaking Change separately — WS-A is behaviour-preserving *except* for that one deliberate change.

---

## 3. WS-B — Re-author the migration map, and stop it drifting

### 3.1 How wrong the map currently is

`.praxia/docs/plans/260614_runspec-migration-map.md` carries `status: complete` (`:6`). Beyond the audit's finding (all 22 "already migrated" rows stale; ≥6 "to migrate" rows already done), this worktree adds these further classes of error the audit did not record:

| Map claim | Line | Reality |
|---|---|---|
| Rows for `_sampling_averaged.py` (10 rows) | `:62-72` | **The file does not exist.** `src/aminx/host/` contains 18 modules; `_sampling_averaged.py` is not among them. |
| `use_arrayrecord` → candidate for `io.use_arrayrecord` | `:99,179` | Field and CLI flag **removed** (`CHANGELOG.md:158-159`). |
| `output_h5_path` "should be stored directly" on `IOConfig` | `:46,191` | **Already is** — `spec.py:32`. Its semantics also changed (Zarr store dir, not HDF5 file — `CHANGELOG.md:26-29`). |
| Targets `batching.*`, `tied.*`, `averaging.*` (11 rows) | `:31-33,61,65-66,68,71,81,83,122,136-142,154-157` | **Those sub-configs no longer exist** (§0.1). |
| `ligand.*` rows listed as "already migrated" | `:143` and neighbours | Never materialised. `run_spec.ligand.` has exactly one read site in all of `src/`, and it is unreachable (§0.3). |
| "only one RunSpec read site today" (`streaming.py:310`) | `:220-222` | True when written; contradicted today by `plan.py:941-950`, `kernel_dispatch.py:500,533`, `_sampling_grid_lineage.py:16,28`, and more. Useful now only as *evidence of the map's vintage* — which is how §2.3 uses it. |

A reader scoping T4.1 from this document is misled on essentially every axis. The 260817 assessment already recommended the minimum action (`260817…md:245-246`: flip to `status: outdated`); this spec does that **and** replaces it, because a document that is merely marked outdated still has no successor and T4.1 still has nothing correct to read.

### 3.2 What the corrected map should be — and what it should *not* be

**Explicit design decision: do not re-author the map as another 100-row `file:line → field` table.** That shape is *why* it rotted. It encodes line numbers, which change on every edit, and file names, which change on refactor — it was guaranteed to be wrong within weeks and it was. Re-authoring the same shape buys a document that is correct for one day.

The replacement has two parts, and the second is the load-bearing one:

**Part 1 — a short, stable, human-authored decision record** (`.praxia/docs/decisions/`, not `plans/`), containing:
- The §2.1 two-clause decision rule, stated once, as the project's standing answer to "should this field go on `RunSpec`?"
- The current sub-config roster (6 after WS-A: `io`, `resource`, `multistate`, `precision`, `plan`, `sampling`, plus `encoding_fusion`/`decoding_fusion` as top-level slots) and one sentence per sub-config naming which clause of §2.1 it satisfies.
- The history in four lines: RS-1 populated 11 sub-configs; 5 went write-only; 3 were deleted in `dd0e952`; 2 were deleted in WS-A. **This is the institutional memory that stops a future contributor re-adding a `BatchingConfig`.**
- The RS-8 export-direction criterion's retirement (§2.5 Step A3), with its argument, so a future reader of `260611_runspec-unification.md:91` finds the disposition rather than an apparently-unmet criterion.
- An explicit statement that the field-level inventory is *generated*, not maintained by hand, with a pointer to Part 2.
- The open question from §8 (does the portable-JSON wire format itself still earn its keep?), recorded as open rather than silently dropped.

**Part 2 — a partly-generated inventory plus an import-time drift gate.**

### 3.3 The anti-drift mechanism

**Reuse the in-repo precedent rather than inventing a CI grep.** `src/aminx/host/spec_partition.py` already solves this exact problem class. Its docstring (`:1-26`) describes the identical failure: a hand-written literal that had no coupling to `dataclasses.fields(...)`, drifting silently, producing eleven separate bugs over five weeks, none found by a sweep. Its fix — `assert_partition_is_exhaustive()` (`:111-157`), run **at import** (`:236`), requiring every dataclass field to be classified into one of three named buckets or fail the build — is directly transferable.

**Which of `spec_partition.py`'s guarantees transfer, and which do not.** This matters because the new gate will be weaker than its model in one specific way, and pretending otherwise is how gates get trusted past their remit:

| Guarantee | Mechanism there | Transfers to B3? |
|---|---|---|
| **Exhaustiveness over fields** | `actual - classified` must be empty (`:139-147`), where `actual` comes from `dataclasses.fields` (`:107-108`) | **Yes, cleanly.** Same reflection, applied per sub-config. |
| **Staleness check** | classified names that are no longer fields raise (`:149-152`) | **Yes, cleanly.** |
| **Vague-reason check** | reasons under 4 words raise (`:154-157`) | **Yes, cleanly.** This is what makes "classify it to shut it up" cost more than doing it right. |
| **Executed serializer as source of truth** | the "serialized" bucket is not hand-written — it is `set(run_specification_to_json_dict(probe))` on a realistic probe (`:128-136`) | **Partly.** It transfers to the `SERIALIZATION_ONLY` bucket (see below) and **not** to the `MIGRATED` bucket, which stays hand-maintained. |

`spec_partition.py:123-127` is emphatic about why the probe must resemble a real spec rather than take defaults: a bare-defaults probe passed while failing to reach the path that actually breaks, and *"a probe that cannot reach the failing path is a false green, which is the exact bug class this module exists to prevent."* B3's probe inherits that requirement.

**Proposed `src/aminx/run/_runspec_coverage.py`:**

- Compute, at import, the set of every `RunSpec` sub-config field as `"<subconfig>.<field>"` (reflectively, via `dataclasses.fields` on each sub-config — `spec.py:273` already uses this pattern in `topology_hash`).
- Derive the `SERIALIZATION_ONLY` bucket **by execution, not by hand**, mirroring `spec_partition.py:136` exactly:
  ```python
  probe = _placeholder_run_spec(...)          # or a build_run_spec-produced spec; see below
  dumped = run_spec_portable_to_dict(probe)
  SERIALIZATION_ONLY = {
      f"{block}.{field}"
      for block, value in dumped.items()
      if isinstance(value, dict)
      for field in value
  }
  ```
  This yields `{io.sink_kind, io.output_dir, io.manifest_path, multistate.mode, multistate.n_states, multistate.combine_strategy, resource.n_devices, resource.sample_batch_size, resource.structure_batch_size, resource.max_buffer_size, precision.compute}` from the executed serializer. If someone drops a field from `to_dict`, the bucket shrinks automatically and the field falls through to `unclassified` unless it is also classified elsewhere — the drift is caught by construction rather than by a reviewer noticing.
  Per `spec_partition.py:123-127`'s warning, the probe must be a **realistic** spec, not a bare-defaults one; prefer a `build_run_spec`-produced spec over `_placeholder_run_spec` if one can be constructed cheaply at import, and document the choice in the module docstring.
- Require every remaining field to be classified in exactly one of two **hand-written** buckets:
  - **`MIGRATED`** — the sub-config field is the canonical read path; the flat equivalent (if any) is legacy. Entry must name at least one reader module.
  - **`MIRRORED`** — deliberately duplicated with a live flat field, **with a reason naming a mechanism** (the `EXCLUDED_WITH_REASON` discipline, `spec_partition.py:58-59,154-157` — *"'it contains a callable' is a reason; 'not applicable' is not"*).
- Fail with a `RunSpecCoverageError` on any unclassified field, and on any hand-written classification naming something that is no longer a field.

**What this does and does not catch — stated plainly, because review will probe it:**
- It **does** catch: a new sub-config field added with no classification (the WS-A failure mode — 27 fields accreting unnoticed); a field deleted while its documentation entry survives; a field silently dropped from the wire format (via the derived bucket).
- It **does not** catch: a field classified `MIGRATED` whose last real reader is later deleted, silently making it dead again. Detecting *that* needs read-site analysis, which is what rots.
- **Mitigation for the residual gap, and the honest limit of this mechanism:** require the `MIGRATED` classification to name at least one reader module, and add one test that imports each named module and asserts it is importable. This turns "the reader was deleted" into a failure, without pinning line numbers. It does *not* catch "the reader still exists but no longer reads this field." `MIGRATED` is and remains the weaker, hand-maintained bucket; `SERIALIZATION_ONLY` is derived and does not have this problem. That residual is accepted for `MIGRATED`; the alternative is a static analyser, which is disproportionate for 6 sub-configs.

**Rejected alternative:** a CI job that greps read-sites and diffs against a checked-in table. Rejected because (a) it reintroduces line-number brittleness, (b) it runs in CI rather than at import, so a developer sees the failure minutes later instead of immediately, and (c) the project already has a working, proven instance of the import-time pattern and no instance of the grep pattern. Consistency with `spec_partition.py` matters more than marginal coverage.

### 3.4 WS-B steps

**Step B1 — Retire the old map.**
Files: `.praxia/docs/plans/260614_runspec-migration-map.md` (modify)
- Frontmatter `status: complete` → `status: superseded`; add `superseded_by:` pointing at the B2 decision record.
- Prepend a short banner: what was wrong, in which directions, citing §3.1's table and `260707…md` Finding B. **Do not delete the body** — it is the evidence for the drift, and deleting it destroys the ability to check this spec's claims (§2.3 argument 1 cites `:220-222` as vintage evidence).

**Gate B1:** `docs(action="check", payload={strict: true})` passes; `docs(action="index")` regenerates cleanly. (MCP form preferred per the internal-docs convention.)

---

**Step B2 — Author the decision record.**
Files: `.praxia/docs/decisions/260827_runspec-subconfig-membership-rule.md` (create) — via `docs(action="add", payload={category: "decisions", …, task_id: "260827_runspec-scaffolding-remediation-spec"})`.
Also: one-line supersession note appended at `.praxia/docs/specs/260611_runspec-unification.md:91` (§2.5 Step A3).
Content: §3.2 Part 1.

**Gate B2:** doc exists at the registry-resolved path; `docs check --strict` clean; INDEX regenerated. Content gate: a reader who knows nothing of RS-1 can answer "should my new field go on `RunSpec`?" from this document alone — verify by having the reviewer answer it for `average_node_features` (expected answer: no — host-side, no wire crossing, fails both clauses; and §2.4 is where its future lives).

---

**Step B3 — Implement the coverage gate.**
Files: `src/aminx/run/_runspec_coverage.py` (create), `src/aminx/run/__init__.py` (modify — import it so the assertion runs), `tests/run/test_runspec_coverage.py` (create).
Scope estimate: ~130 LOC + ~70 LOC tests.

**Gate B3 — five required tests, four mirroring `spec_partition.py`'s own guarantees plus one for the derived bucket:**
1. Passing state: `import aminx.run` succeeds against the current tree.
2. Unclassified field → `RunSpecCoverageError`. Inject via a synthetic sub-config in the test, not by editing `spec.py`.
3. Stale classification (a hand-written name that is no longer a field) → `RunSpecCoverageError`.
4. Vague reason (`< 4` words, matching `spec_partition.py:154-157`) → `RunSpecCoverageError`.
5. Derived-bucket coupling: monkeypatch `run_spec_portable_to_dict` to drop one block from its output, assert the now-unclassified fields raise `RunSpecCoverageError`. This is what proves `SERIALIZATION_ONLY` is genuinely derived and not a hand-written list wearing a probe as decoration.

```bash
uv run pytest tests/run/test_runspec_coverage.py -q
uv run python -c "import aminx.run"   # assertion runs at import; must be silent
```

**Ordering constraint:** B3 must land **after** WS-A, or the gate will immediately fail on the 11 fields WS-A deletes and someone will be tempted to classify them `MIRRORED` to make it pass — permanently blessing the scaffolding this spec exists to remove.

---

## 4. WS-C — xtrax transforms adoption and Engine feasibility

### 4.1 Step C0 (blocking, do first) — characterisation tests that expose the real divergence

Before any migration decision is executed, pin current behaviour. §0.2 shows the existing suite cannot see the difference that matters.

Files: `tests/utils/test_safe_map.py` (modify), `tests/utils/test_safe_scan.py` (create if absent)

1. **Non-divisible cardinality (new test)** — `safe_map(f, jnp.arange(100), batch_size=32)`; assert correct result, length 100. *This is the case xtrax's `map.py:33-37` rejects.* Every existing `lax.map`-path test is divisible (`:34,38,58,74`).
2. **`batch_size=0` sentinel provenance (extend an existing test)** — `test_batch_size_zero_routes_to_vmap` already exists (`tests/utils/test_safe_map.py:112-117`) and already compares `batch_size=0` against `batch_size=None`; `test_batch_size_zero_pytree` (`:120-123`) covers the pytree case. Do not write fresh coverage. **Extend `test_batch_size_zero_routes_to_vmap` with a third comparison leg** against `jax.lax.map(f, xs, batch_size=0)`, documenting that the sentinel is JAX's (`loops.py:2691`), not aminx's.
   **"Equivalence" here means exactly three assertions, and no more:** `jnp.allclose(a, b)`, `a.shape == b.shape`, `a.dtype == b.dtype`. It explicitly does **not** mean jaxpr equality — verified, `jax.lax.map(f, xs, batch_size=0)` and `jax.vmap(f)(xs)` produce different jaxprs (the former adds one full-range slice via `_remainder_leaf`) while being numerically identical. A test asserting jaxpr identity would fail today for a reason that does not matter.
3. **Empty-pytree rejection (new test)** — aminx `safe_map.py:43-45` raises `ValueError`; xtrax `map.py:26` would raise `IndexError`. Pin the current exception type.
4. **`safe_scan` leading-axis-0 (new test)** — `xs` of shape `(0, N)`. aminx `safe_scan.py:50-53` only checks for *no leaves* and **passes this through to `lax.scan`**; xtrax `scan.py:43-44` raises `ValueError`. Pin current behaviour so the change in Step C1 is visible as an intentional diff, not an accident.

**Gate C0:** `uv run pytest tests/utils/ -q` — all new and extended tests pass against the *current* implementations before any change.

---

### 4.2 Step C1 — `safe_scan` → `xtrax.transforms.safe_scan`: **ADOPT**

This is the clean win, and it should not be bundled with the `safe_map` question.

| | aminx (`utils/safe_scan.py`) | xtrax (`transforms/scan.py`) |
|---|---|---|
| Signature | `safe_scan(f, xs, *, init)` (`:22-27`) | `safe_scan(fn, init, xs, length=None, reverse=False, unroll=1)` (`:6-13`) |
| Validation | empty pytree only (`:50-53`) | `effective_length == 0`, inferred from `xs` **or** explicit `length` (`:36-44`) — catches empty pytree *and* shape-`(0, N)` |
| Extra capability | none | `length` / `reverse` / `unroll` passthrough |

xtrax's validation is a **strict superset** of aminx's: an empty pytree yields `effective_length = 0` (`scan.py:39-40`) and raises, so nothing is lost. The only real work is the argument-order change.

**Call sites — all three, exhaustively:**

| Site | Current call | Migrated call |
|---|---|---|
| `src/aminx/host/kernel_dispatch.py:88` | `safe_scan(scan_body, xs, init=init)` | `safe_scan(scan_body, init, xs)` |
| `src/aminx/ebm/plan.py:207` | `safe_scan(_scan_body, xs, init=init)` | `safe_scan(_scan_body, init, xs)` |
| `src/aminx/ebm/langevin_schedule.py:459` | `safe_scan(carry_spec.transition, xs, init=carry_spec.init)` | `safe_scan(carry_spec.transition, carry_spec.init, xs)` |

Import sites to update: `kernel_dispatch.py:65` (function-local import), `ebm/plan.py:66`, `ebm/langevin_schedule.py:201`.

**One documentation reference must be updated in the same commit, or it becomes a dangling path.** `ebm/langevin_schedule.py:355` names the module by dotted path inside a prose docstring:

> *"…and executes it directly via ``aminx.utils.safe_scan.safe_scan`` -- provably identical to what ``BatchPlanner.plan()`` would have produced."*

Rewrite the path to `xtrax.transforms.safe_scan`. Then delete `src/aminx/utils/safe_scan.py`.

**The "provably identical" comment is resolved, not a watch item.** Read in full (`:345-356`), it argues **ceremony versus benefit**: building an `AxisSpec`+`BatchPlan` "purely to immediately unwrap it back to the same `(init, transition)` pair would be ceremony with no behavioral difference," so the function builds a `CarrySpec` and executes it directly. The equivalence it asserts is to *what `BatchPlanner.plan()` would have produced* — a code path, not a golden artifact. **It names no golden test and depends on no stored reference output.** Changing `safe_scan`'s argument order does not touch this reasoning: it is the same `lax.scan` call reached through a differently-ordered signature. **Safe to proceed; no golden test blocks this edit; only the dotted path in the prose needs updating.**

**Behaviour-change acceptance, stated explicitly:** a shape-`(0, N)` `xs` that today reaches `lax.scan` and produces empty outputs will after this change raise `ValueError`. This is the intended improvement (it is the `safe_scan` half of the audit's Finding E, `260707…md:205-208`). Test 4 from C0 must be **inverted** in the same commit, with a comment naming the change.

**Also worth flagging while in this file (do not fix here):** `kernel_dispatch.py:67-97`'s `_dispatch_axis` dispatches on `type(strategy).__name__ == "Vmap"` (string comparison, `:69,71,79,90`), whereas `ebm/plan.py:197-208` does the same job with `isinstance` against the imported xtrax types. The string form silently falls through to `batch_size_fallback` (`:97`) if a class is ever renamed. Out of scope; file as a separate debt item.

**Gate C1:**
```bash
uv run ty check && uv run ruff check .
uv run pytest tests/utils/ tests/host/ tests/ebm/ -q
uv run pytest -k "safe_scan or langevin or dispatch" -q     # narrow; do not run the full suite locally
rg -n 'aminx\.utils\.safe_scan|from aminx.utils import safe_scan' src tests docs
```
The `rg` must return nothing — note the `docs` path in the target list, which is what catches the `langevin_schedule.py:355`-class of prose reference if a similar one exists elsewhere.

Scope estimate: ~15 LOC changed, ~58 LOC deleted.

---

### 4.3 Step C2 — `safe_map`: **DECISION REQUIRED, three options, with a recommendation**

§0.2 establishes that this is not the low-risk defence-in-depth change the audit described. The blocking issue is xtrax's divisibility hard-reject (`transforms/map.py:33-37`), which `jax.lax.map` (`loops.py:2685-2689`) and aminx's fork both accept.

**Blast radius — 15 `safe_map` call sites, exhaustively:**

| File | Lines | `batch_size` source | Can it be non-divisible? |
|---|---|---|---|
| `host/kernel_dispatch.py` | 78 | `strategy.tile` / `strategy.batch_size` | **Yes** — planner-chosen tile vs. real cardinality |
| `host/kernel_dispatch.py` | 94, 384, 566 | `None` | No (vmap path) |
| `host/kernel_dispatch.py` | 97 | `batch_size_fallback` | Yes |
| `host/kernel_dispatch.py` | 461, 561 | `samples_bs` | **Yes** — `num_samples` vs. planner tile; this is the cardinality-bug axis |
| `host/kernel_dispatch.py` | 468, 563 | `temps_bs` | Yes |
| `host/kernel_dispatch.py` | 470, 544 | `noises_bs` | Yes |
| `host/kernel_dispatch.py` | 473, 578 | `structures_bs` | Yes |
| `tiling/iterator.py` | 157 | `self.tile` | Yes |
| `ebm/plan.py` | 200 | `strategy.batch_size` (xtrax `SafeMap`) | Yes |

All of `samples_bs`/`temps_bs`/`noises_bs`/`structures_bs` come from `extract_batch_sizes` (`host/plan.py:324-344`) → `_legacy_batch_size` (`:306-321`). The `0` value is not incidental — it is deliberate, and its own docstring says why (`plan.py:307-317`), quoted here in full so nobody has to reconstruct the reasoning from a line reference:

> *"safe_map treats batch_size=0 (or None) as "no chunking, run everything at once" -- the Vmap-equivalent -- and any positive value as "chunk into groups of that size." aminx's retired local BatchPlanner always set batch_size=0 for Vmap decisions to match; xtrax's BatchPlanner never does (Vmap decisions carry batch_size=spec.default_batch_size, e.g. 1). Feeding that straight to safe_map would silently turn a parallel Vmap axis into a fully serial per-element loop for aminx's legacy (use_unified_driver=False) dispatch path in host/kernel_dispatch.py, which is not dead code (a real, still-tested CLI flag) -- EPIC #1541 T-PLANNER.2 finding, 2026-07-06."*

So **both** divergent behaviours (the `0` sentinel and non-divisible tiles) are live on the default sampling path, and the `0` sentinel specifically exists to prevent a known, previously-diagnosed serialisation bug.

**Option 1 — Adopt xtrax's `safe_map` as-is.**
Requires eliminating the `0` sentinel (rewrite `_legacy_batch_size`, `plan.py:306-321`, to return `None` for `Vmap`) **and** guaranteeing divisibility at all 15 sites. The second is not achievable without either padding cardinalities or constraining the planner — a large change with real performance consequences (padding a samples axis wastes compute proportional to the remainder).

There is a **second, independent** reason this is not viable, and it is already sitting in the test suite: `test_batch_size_zero_routes_to_vmap` (`tests/utils/test_safe_map.py:112-117`) and `test_batch_size_zero_pytree` (`:120-123`) both call `safe_map(..., batch_size=0)`. xtrax's `safe_map` has no `== 0` branch at all, so both would reach `n % batch_size` at `map.py:33` and raise **`ZeroDivisionError`** — a straight adoption breaks two existing green tests on contact, before any production shape is considered.

*Verdict: not viable as scoped.* The audit's recommendation assumed only the first half of the sentinel work existed and did not account for the divisibility reject at all.

**Option 2 — Adopt, after an upstream xtrax change.**
File an xtrax issue to relax `map.py:33-37` from a hard `ValueError` to delegating to `jax.lax.map`, which already handles the remainder correctly (`loops.py:2685-2689`, `_batch_and_remainder` `:2643-2663`), and to accept `0` as the vmap sentinel `lax.map` itself documents (`loops.py:2691`). xtrax's check is stricter than the primitive it wraps, for no stated reason — plausibly an oversight. If upstream accepts, Option 1 becomes a small change: delete `utils/safe_map.py`, rewrite `_legacy_batch_size` to emit `None` (or keep `0`, if upstream takes the sentinel too), update 15 call sites' import.
*Verdict: **recommended path**, but it is gated on a third party and must not block WS-A or WS-B.*

**Option 3 — Keep the fork, fix its docstring, and stop there.**
Keep `utils/safe_map.py` unchanged in behaviour. Rewrite its docstring, which is currently wrong on two counts: it claims to mirror *jaxbeans* semantics (`safe_map.py:3-5` — a dependency that does not appear in `pyproject.toml`), and it does not mention xtrax at all. Replace with: the divergence table from §0.2, the reason (`jax.lax.map` supports what xtrax rejects), a cross-reference to `plan.py:307-317`'s sentinel rationale, and a link to the upstream issue from Option 2.

**Explicitly not doing:** simplifying `:49` from `batch_size is None or batch_size == 0 or num_elements <= batch_size` to drop the `== 0` clause. It is technically redundant (`jax.lax.map` handles `0` itself, `loops.py:2647`), but removing it buys nothing measurable — it substitutes a `lax.map` call that reduces to a bare `vmap` over one slice for a Python short-circuit — while touching a branch whose presence is load-bearing documentation of a diagnosed bug (`plan.py:307-317`). A change with no measurable benefit and a nonzero reading cost is not worth a commit. **Option 3 is docstring-only.**
*Verdict: the correct interim state while Option 2 is pending. Cheap, honest, and it removes the misleading docstring immediately.*

**Recommendation: execute Option 3 now; file the xtrax issue for Option 2; revisit when upstream answers.** Do **not** execute Option 1.

**File a praxia debt item for the Option-2 deferral**, alongside #1500 and #1501 rather than leaving it as prose: *"revisit `safe_map` adoption once xtrax relaxes `transforms/map.py:33-37`'s divisibility check"*, with a **90-day check-date**, linking the upstream issue and this section. Linking the issue from the docstring is a forcing artifact for a reader of that file; a dated debt item is the forcing artifact for the project. Both are needed — the docstring link alone has no clock.

**The defence-in-depth argument the audit made does not survive either.** Its claim was that xtrax's `safe_map` would have turned the cardinality-divergence bug into a loud crash. But (a) that bug is fixed (`CHANGELOG.md:72-90`, `n_samples_override`), (b) the "loud crash" would have been a `ZeroDivisionError` from a divisibility check that also crashes on *correct* inputs, and (c) the genuinely dangerous path the audit itself identified — `_dispatch_axis`'s unconditional `jax.vmap` at `kernel_dispatch.py:69-70` — **does not go through `safe_map` at all**, so no `safe_map` change defends it. If defence-in-depth on that path is wanted, the right change is a cardinality assertion in `_dispatch_axis`, which is a different (small, self-contained) piece of work. **Flagging it here as the thing the audit was actually reaching for; not folding it into this spec** (§8 open question 3).

**Gate C2 (Option 3):**
```bash
uv run pytest tests/utils/test_safe_map.py -q      # incl. the C0 non-divisible test
uv run ruff check src/aminx/utils/safe_map.py
```
Plus a documentation gate: the docstring names the three concrete divergences (`0`-sentinel origin, divisibility, empty-pytree exception type) with file:line, cross-references `plan.py:307-317`, and does not mention jaxbeans. Plus: the xtrax issue exists and is linked from the docstring; the praxia debt item exists with a 90-day check-date. Plus: `git diff --stat src/aminx/utils/safe_map.py` shows docstring lines only — no change to `:49` or any other executable line.

---

### 4.4 Step C3 — `xtrax.engine.Engine.fit()` vs `training/trainer.py`: feasibility assessment only

**Deliverable: a written assessment, not a migration.** Based on reading both, here is the assessment this step should confirm and formalise — the audit flagged this "unassessed" (`260707…md:216-223`), so producing the assessment *is* the closure.

**Verdict: `Engine.fit()` is not a drop-in for `trainer.train()`. Five capability gaps, three of which need upstream xtrax changes.**

| # | Requirement | `trainer.py` | `Engine` | Bridgeable in aminx alone? |
|---|---|---|---|---|
| 1 | Step function contract | `train_step` takes **19 positional args** (`trainer.py:772-792`) | `trainer.step(state, batch) -> (state, metrics)` (`engine.py:34,132`) | **Yes** — a `TrainStepLike` adapter closing over `optimizer`, `spec`, `noise_schedule`, `compute_dtype`. Mechanical. |
| 2 | Checkpoint cadence | every `spec.checkpoint_every` **steps** (`trainer.py:852-853`) **plus** a second `permanent_manager` for `spec.save_at_epochs` (`:856-858`) | one `checkpoint_dir`, saved **per-epoch only** (`engine.py:150-151`) | **No.** Step cadence would have to move into an `on_step_end` callback — which `Engine` fires **asynchronously** through `BoundedCallbackHandler(max_concurrent=4)` (`engine.py:107,135-140`), giving no ordering guarantee between concurrent checkpoint writes. Unsafe for checkpointing. The dual-manager requirement has no expression at all. |
| 3 | Mid-epoch eval | every `spec.eval_every` steps, inside the batch loop (`trainer.py:808-838`) | `Engine.eval()` is a **separate top-level method**; `fit()` never calls it (`engine.py:160`) | **Partially** — a callback could close over `val_loader`, since `fit()` does not pass `data` to hooks. Fragile, and lands in the same async-ordering problem as #2. |
| 4 | Early stopping | patience counter + `break` (`trainer.py:840-850`) | **no stop protocol** — no callback return value is inspected, no exception contract | **No.** Only expressible by raising from a callback and catching outside `fit()`; `fit()`'s `finally` (`engine.py:153-156`) would fire `on_train_end`, so it is *survivable*, but it is an abuse of control flow, not a supported path. |
| 5 | Data contract | `create_protein_dataset` iterables, attribute-style batches (`batch.coordinates`, `trainer.py:776`) | `train_iter()`/`eval_iter()` (`engine.py:41-43`); `eval()` additionally assumes **dict** batches (`batch["inputs"]`, `batch["targets"]`, `engine.py:204-205`) when `loss_fn` is given | **Yes** — a thin `DataIterLike` wrapper; the dict assumption is avoidable by not passing `loss_fn`. |

Non-issues worth recording so they are not re-litigated: `setup_mixed_precision` (`trainer.py:728`) is a one-shot call before the loop and `Engine` does not interfere; `ResumableState` is already the shared currency (`trainer.py:795-801`, `engine.py:65`); the final test loop (`trainer.py:862-934`) is outside `fit()`'s remit either way.

**Recommendation: do not migrate.** Gaps 2 and 4 are correctness-relevant (checkpoint durability, training termination) and both need upstream `Engine` changes — a synchronous `on_step_end` option or an explicit checkpoint-cadence parameter, and a stop protocol. File those as xtrax feature requests. Re-assess if they land. The 939 lines are not 939 lines of duplicated orchestration: a large fraction is `train_step`/`eval_step` bodies (`trainer.py:293-706`, ~410 lines) that would survive any migration unchanged.

**Incidental defect found during this assessment — flag, do not fix here.** `trainer.py:760-762` constructs `eqx.filter_jit(train_step)` and `eqx.filter_jit(eval_step)` **inside** the `for epoch` loop (`:755`). A fresh `filter_jit` wrapper per epoch is a fresh jit cache key, which would mean a full recompile at the start of every epoch. Hoisting them above `:755` is a one-line change if confirmed. **Verify before filing** — do not take this on faith:
```bash
JAX_LOG_COMPILES=1 uv run python -c "<2-epoch, 2-batch smoke>" 2>&1 | grep -c "Compiling train_step"
```
Expected if the defect is real: compile count scales with epochs, not with distinct input shapes. File as a separate backlog item with that evidence attached.

**Gate C3:** the assessment is filed via `docs(action="add", payload={category: "research", title: "xtrax Engine.fit feasibility vs aminx trainer", task_id: "260827_runspec-scaffolding-remediation-spec"})`; every row of the gap table cites file:line on both sides; the two xtrax feature requests exist and are linked. No source file changes.

---

## 5. Sequencing

```
A1 → A2+A3 (one commit) → A4 ─┐
                              ├→ B3   (B3 must follow A; see §3.4 ordering constraint)
B1 → B2 ──────────────────────┘

C0 → C1 (independent of A/B; can run in parallel)
C0 → C2 (Option 3)
C3      (no code changes; any time)
```

**Rationale for this order:**
1. **WS-A first.** It deletes code, shrinking WS-B's surface. Running B3's coverage gate before A would force the 11 doomed fields to be classified, which risks blessing them permanently.
2. **B1 immediately, in parallel.** Flipping the map's frontmatter is the single cheapest correction available and the 260817 assessment already asked for it (`:245-246`). It should not wait behind engineering.
3. **WS-C is independent.** C1 (`safe_scan`) is a clean, self-contained win. C2 (Option 3) is documentation plus an upstream issue plus a debt item. C3 is a document. None blocks or is blocked by A/B.
4. **The one hard ordering constraint** is A before B3.

**Recommended commit boundaries:** `A1`; `A2+A3`; `A4`; `B1`; `B2`; `B3`; `C0`; `C1`; `C2`; `C3`. Ten commits, each independently revertible.

---

## 6. Risks

| # | Risk | Likelihood | Impact | Mitigation / rollback |
|---|---|---|---|---|
| R1 | `dd0e952` is not on the target branch, so §0.1's re-baseline is wrong and WS-A's scope is 27 fields, not 11 | Low (docstring + CHANGELOG + tree all agree) | **High** — invalidates all of WS-A | Blocking precondition, §0.1/§2.5. `git merge-base --is-ancestor dd0e952 HEAD` must exit 0 before Step A1. If not, stop and re-scope. |
| R2 | A `.grid.*`/`.ligand.*` reference exists somewhere this spec did not enumerate (dynamic access, `eqx.tree_at`, a docstring path, another repo) | Low in `src/`; **needs an explicit gate outside it** | High if a consumer breaks | See the **named gate R2-GREP** below; it replaces a hand-wave with an executable check. Rollback: revert A1-A3 (near-pure deletion, clean revert). |
| R3 | Step A3's new unknown-key rejection breaks an existing caller passing extra keys | Low — one production caller (`cli.py:1620`), fed by `from_dict` | Medium | It is a Breaking Change and recorded as one (A4). Gate: `tests/cli/test_inputs_integration.py` exercises the round-trip at `:311-336`. Rollback: restore the ignore behaviour; keep the sub-config deletions (independent). |
| R4 | WS-C1's stricter `safe_scan` validation raises on a shape-`(0, N)` input that some path legitimately produces | Medium — `ebm/langevin_schedule.py:459` scans over a schedule that could be empty at a degenerate config | Medium — a run that silently no-ops today would now crash | C0 test 4 pins current behaviour first. Before C1, check whether any config can produce a zero-length schedule (`langevin_schedule.py` around `:440-460`). If yes, that is a *bug being surfaced*, not a regression — but it must be surfaced deliberately, with a clear error, not as a mystery `ValueError`. Rollback: `utils/safe_scan.py` is 58 lines; restore and revert 3 call sites plus the `:355` docstring path. |
| R5 | Adopting xtrax `safe_map` (Option 1) breaks production on non-divisible cardinalities | **Certain, if Option 1 is executed** | High — silent-working configs start raising | Option 1 is explicitly **not recommended** (§4.3). C0 test 1 makes the breakage visible immediately if anyone attempts it; the two existing `batch_size=0` tests would additionally raise `ZeroDivisionError` on contact. |
| R6 | The B3 coverage gate becomes a nuisance and gets weakened or `# noqa`'d | Medium — this is how such gates usually die | Medium — drift returns | Mirror `spec_partition.py` exactly, including the vague-reason check (`:154-157`), which is what makes "classify it to shut it up" cost more than doing it right. Keep the hand-written buckets small (6 sub-configs, one bucket derived). Cite `spec_partition.py`'s track record in the module docstring. |
| R7 | WS-A conflicts with in-flight debt #1500 work on the same metadata-dict blocks | Medium if #1500 starts concurrently | Low — mechanical conflicts | WS-A deliberately does *not* touch those blocks (§2.2 argument 3). That is a design choice specifically to avoid this. Sequence #1500 after WS-A if both are active. |
| R8 | This spec's own read-site inventories go stale before execution | Medium — the audit it replaces went stale in six weeks | Medium | Every table cites file:line and the grep that produced it. Re-run R2-GREP and the `safe_map`/`safe_scan` greps at execution start; if counts differ from §2.2/§2.3/§4.3, re-scope. **This spec should be treated as valid for weeks, not months.** |
| R9 | The `filter_jit`-in-loop finding (§4.4) is wrong and gets filed as a real bug | Medium — asserted from reading, not measured | Low — wasted triage | Flagged as unverified with a specific verification command. Do not file without the `JAX_LOG_COMPILES=1` evidence. |
| R10 | Step A2's two latent constructor defects (`sampling=None`, `carry_specs={}`) are noticed by a future reader who assumes A2 blessed them | Medium | Low | §2.5 Step A2 requires either fixing both or filing a debt item with a code comment. Neither option leaves them unremarked. |

**Gate R2-GREP (named, executable, run before merging WS-A):**

```bash
rg -n --glob '!mpnn_ext/external/aminx/**' \
      --glob '!tev_design/prxteinmpnn/**' \
      --glob '!**/.claude/worktrees/**' \
      --glob '!docs/_build/**' \
      --glob '!.praxia/**' \
   'run_spec\.(grid|ligand)\.|GridLineageConfig|LigandConfig' \
   src tests
```

**Pass criterion: zero hits** (after A1-A3 have landed their own removals).

**Why those exclusions specifically** — a naive cross-repo grep returns ~19 files of pure noise and would stall the merge on nothing:
- `mpnn_ext/external/aminx/` is a **vendored copy of aminx itself** (it contains its own `src/aminx/run/spec.py`). It resyncs from upstream; it needs no separate handling and its hits are this change's own code seen twice.
- `tev_design/prxteinmpnn/` is an **independent fork** with its own `spec.py` defining its own `LigandConfig`/`GridLineageConfig`. It is not downstream of this `RunSpec` and is unaffected by this change.
- `.claude/worktrees/`, `docs/_build/`, `.praxia/` are working copies, build output, and the docs that *describe* this change (including this spec) — all expected to mention the names.

The genuine downstream-breakage question is therefore narrower than the risk statement suggests: there is no consumer that imports aminx's `RunSpec` and reads `.grid`/`.ligand` outside this repo. If a new vendoring relationship appears later, extend the exclusion list deliberately rather than widening the pattern.

---

## 7. Consolidated verification

**Per this project's local-compute rule, never run the full `pytest` suite locally** — it exhausts swap. All gates below are narrow (single file, directory, or `-k` selector). If a full-suite run is needed before merge, it goes to titanix.

| Step | Gate |
|---|---|
| A1 | `uv run ty check && uv run ruff check src/aminx/run/spec.py && uv run pytest tests/run/ tests/cli/test_inputs_integration.py -q`; dynamic-access grep clean |
| A2+A3 | `uv run pytest tests/run/test_run_spec_portable_json.py tests/cli/test_inputs_integration.py -q` + new unknown-key rejection test; A2's two latent defects fixed or filed |
| A4 | manual read of `CHANGELOG.md:203-213`; both Breaking Changes recorded; RS-8 retirement noted |
| WS-A total | `uv run ty check && uv run ruff check . && uv run pytest tests/run/ tests/cli/ tests/host/ -q`; **gate R2-GREP** zero hits; dynamic-access grep clean |
| B1 | `docs(action="check", payload={strict: true})` |
| B2 | doc at registry path; INDEX regenerated; reader-comprehension check (§3.4); `260611_runspec-unification.md:91` annotated |
| B3 | `uv run pytest tests/run/test_runspec_coverage.py -q`; `uv run python -c "import aminx.run"` silent; 5 required tests present |
| C0 | `uv run pytest tests/utils/ -q` green **against unchanged sources**; test 2 is an extension of `test_batch_size_zero_routes_to_vmap`, with equivalence defined as allclose + shape + dtype |
| C1 | `uv run ty check`; `uv run pytest tests/utils/ tests/host/ tests/ebm/ -q`; `rg -n 'aminx\.utils\.safe_scan' src tests docs` empty (incl. the `langevin_schedule.py:355` docstring path); C0 test 4 inverted with a comment |
| C2 | `uv run pytest tests/utils/test_safe_map.py -q`; docstring names 3 divergences with file:line and cross-references `plan.py:307-317`, no jaxbeans reference; `git diff --stat` shows docstring-only change; xtrax issue filed and linked; praxia debt item filed with 90-day check-date |
| C3 | assessment doc filed; every gap-table row cites file:line on both sides; 2 xtrax feature requests linked |

---

## 8. Open questions this spec deliberately does not answer

Recorded so review can rule on them rather than treating them as gaps:

1. **Does the portable-JSON wire format still earn its keep — and if not, what happens to the four sub-configs that exist only to cross it?** `IOConfig`/`ResourceConfig`/`MultistateConfig`/`PrecisionConfig` are retained under §2.1 clause 2, and three of the four have *no* trace-adjacent consumer at all (verified: every live `run_spec.io.*` read — `prep.py:129-130`, `runner.py:183`, `streaming.py:79`, `multistate_poe.py:610` — is host-side). Their fields are not dead: `run_spec_portable_json.py:163-178` reads them, and B3 will derive that bucket by execution. **The open question is not whether they are dead — it is whether clause 2's premise survives.** The portable v2 format has exactly one production consumer (`aminx spec portable-roundtrip`, `cli.py:1605-1621`), which is a round-trip self-test. If that command is the format's only user, the format may not be worth a sub-config each; if the format is retired, clause 2 empties and all four sub-configs would need to re-justify under clause 1, which none currently satisfies. **That is a big enough consequence to be worth chasing next**, and it is a strictly larger question than WS-A's. WS-B Step B2 records it.
2. **Will xtrax relax `map.py:33-37`'s divisibility check?** Gates WS-C's Option 2. Third-party dependency; tracked by the §4.3 debt item with a 90-day check-date.
3. **Is `_dispatch_axis`'s unconditional `jax.vmap` (`kernel_dispatch.py:69-70`) worth an independent cardinality assertion** now that the planner fix (`n_samples_override`) makes its input trustworthy? §4.3 argues this is what the audit's defence-in-depth instinct was actually pointing at. Small, self-contained; deliberately not folded in.
4. **Does `spec.run_spec` ever go stale?** `_sync_run_spec` (`run/specs.py:329-333`) memoises on `_run_spec_synced` and is called at the end of each subclass's `__post_init__` (`:522,566,695,723,795`). No post-construction mutation of a spec was found (`rg 'object\.__setattr__\(spec|setattr\(spec|replace\(spec' src` → no matches), and `dataclasses.replace` re-runs `__post_init__`. So it appears safe today — but nothing *enforces* it, and a future `object.__setattr__` on a constructed spec would silently desync every `run_spec` read. Not a WS-A blocker (WS-A removes readers rather than adding them), but it is a latent hazard that argues generally against widening `run_spec` read surface. **Worth its own guard eventually.**

---

## 9. References

- `.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md` — the audit remediated here (Findings A, B, E)
- `.praxia/docs/audits/260817_aminx-deeper-work-assessment.md` §4, §6, §7 — sequencing input; §4's "27 of 72 still holds" is corrected by §0.1
- `.praxia/docs/plans/260614_runspec-migration-map.md` — the document WS-B retires; `:220-222` is load-bearing evidence of its vintage (§2.3)
- `.praxia/docs/specs/260611_runspec-unification.md` — RS track intent; `:50-51,99-100` sub-config taxonomy; `:91` RS-8's two-direction guard, one of which §2.5 Step A3 implements and one of which it retires
- `.praxia/docs/specs/260706_samples-axis-planner-cardinality-mismatch.md` — the fixed bug whose mechanism §0.2 re-attributes
- `CHANGELOG.md:203-213` — the `dd0e952` removal record, whose final sentence A4 corrects
- `src/aminx/host/spec_partition.py` — the anti-drift pattern WS-B reuses; `:111-157` the assertion, `:123-136` the executed-probe discipline B3 inherits for its derived bucket
- `src/aminx/host/plan.py:306-321` — `_legacy_batch_size` and its rationale for the `0` sentinel (§4.3)
- praxia debt **#1500** (multistate path consolidation) — adjacent, out of scope, sequenced after WS-A
- praxia debt **#1501** (composable fusion transform) — landing spot documented in §2.4
- praxia debt **(new, filed by §4.3)** — revisit `safe_map` adoption after upstream xtrax relaxes `transforms/map.py:33-37`; 90-day check-date

---

**Files this spec would touch, by workstream:**

- **WS-A:** `src/aminx/run/spec.py`, `src/aminx/run/run_spec_portable_json.py`, `tests/run/test_run_spec_portable_json.py`, `tests/cli/test_inputs_integration.py`, `CHANGELOG.md` (all modify)
- **WS-B:** `.praxia/docs/plans/260614_runspec-migration-map.md` (modify), `.praxia/docs/specs/260611_runspec-unification.md` (modify — one-line RS-8 supersession note), `.praxia/docs/decisions/260827_runspec-subconfig-membership-rule.md` (create), `src/aminx/run/_runspec_coverage.py` (create), `src/aminx/run/__init__.py` (modify), `tests/run/test_runspec_coverage.py` (create)
- **WS-C:** `src/aminx/utils/safe_scan.py` (delete), `src/aminx/utils/safe_map.py` (modify — docstring only), `src/aminx/host/kernel_dispatch.py`, `src/aminx/ebm/plan.py`, `src/aminx/ebm/langevin_schedule.py` (modify — call sites plus the `:355` docstring path), `tests/utils/test_safe_map.py` (modify), `tests/utils/test_safe_scan.py` (create), plus one research doc (create)

