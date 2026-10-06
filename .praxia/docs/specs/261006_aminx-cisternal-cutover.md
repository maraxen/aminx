---
title: 'S7: aminx cisternal cutover - agent assets, MCP surface, provenance'
description: 'Make aminx agent-native through cisternal: a manifest-driven plugin bundle (using-aminx moved into the repo), an aminx-mcp server wired with @cisternal.tool over host.runner, cisternal provenance/telemetry for campaign runs; the Typer-to-cyclopts CLI port is a user decision, recommended deferred.'
status: draft
task_id: 261006_cisternal-cutover-spec
date: '261006'
backlog_ids: ''
adversarial_review: ''
owner_repos:
  - aminx
  - cisternal  # consumed at >=0.1.1a15; no change requested of it unless S7-Q4 says so
related_specs:
  - S4  # xtrax model contract: ModelManifest ids (S4-19) are the later naming target for MCP tools
  - S5  # hub: `aminx serve` (S5-21) is the HTTP sibling of aminx-mcp; both are thin adapters over host.runner
---
# S7: aminx cisternal cutover - agent assets, MCP surface, provenance

The 261001 ecosystem spec set (S1-S6, 213 items in `plans/261001_ecosystem-backlog-dag.md`) contains no
agent-facing work: `cistern`, `MCP`, `agentic` and `agent-native` each have 0 hits across the DAG and all
seven 2610xx specs (checked on origin/main cdee29b1). S7 fills that gap. It is numbered as the next series
so its item ids (`S7-NN`) cannot collide with the existing DAG, and it can be folded into a regenerated
DAG later (S7-15).

"Cutover" follows what it meant for the sibling projects that already did it (bathos, naurmalade, maraxiom,
myxcel, contemplex). It has three legs, and S7 takes them in increasing order of blast radius:

| Leg | Replaces | aminx today | S7 stance |
|---|---|---|---|
| A. Agent assets | hand-placed skill files, no plugin | `using-aminx` exists only as `~/.claude/skills/using-aminx/SKILL.md`, untracked and partly stale | **do** (additive) |
| B. Tool surface | hand-rolled FastMCP + separate CLI | no MCP server at all; `.mcp.json` tracked but empty | **do** the MCP leg (additive) |
| C. Provenance + telemetry | private logging / "unknown" git fields | `campaign.py` git_sha defaults to `"unknown"`; stdlib logging only | **do** (additive, guarded) |
| D. CLI port Typer -> cyclopts via `wire()` | the Typer CLI | 2117-line Typer CLI | **user decision**, recommended *defer* (section 4.7) |

## 1. Goal and non-goals

**Goal.** An agent in Claude Code (and the other cisternal surfaces) can install aminx as a plugin and get:
an up-to-date `using-aminx` skill that ships from the aminx repo, plus an `aminx-mcp` server whose tools run
sampling, scoring, inspection and spec validation on structure files and return JSON. Campaign runs record
the real code commit instead of `"unknown"`.

**Non-goals.**
- Not a new inference path. Every tool is an adapter over `aminx.host.runner` and the spec codec; no
  numerics change, so no parity re-run is triggered.
- Not the hub's remote executor. `aminx serve` (S5-21, HTTP wire v1) stays S5's. S7 only requires that
  both adapters share the same request normalization (section 6).
- No change to the published `aminx` CLI's flags or command names unless the user picks D (S7-Q1).
- No hosted service, accounts or remote state. The MCP server is a local stdio process.

## 2. How to read the tags

`[verified]` = read at the cited file:line or probed by a spike recorded here. `[unverified]` = from a
recon report or a doc, not yet checked against code; each one that would change the design has a spike
item. Line numbers for cisternal are against the **0.1.1a15 release** (tag `v0.1.1a15`, read from the
installed wheel in maraxiom's venv). The local `~/projects/cisternal` checkout is 53 commits behind
origin/main at 0.1.1a7 and must not be cited for current API.

## 3. Current state

### 3.1 aminx

- `[verified]` One console script, `aminx = "aminx.cli:main"` (`pyproject.toml:33-34`). Framework is
  **Typer** (`pyproject.toml:24`, `src/aminx/cli.py:14,36`). Groups: `run`, `spec`, `campaign`, `potts`.
- `[verified]` The `run` group declares its shared options once on a Typer callback (`cli.py:385-397`,
  `@run_app.callback()` at `cli.py:448`) and passes them through `ctx.obj`. `cli.py` has 285 `Annotated[`
  parameter declarations.
- `[verified]` `aminx run sample` (without `--emit-json`) calls `host.runner.sample(spec)` and **discards
  the returned dict** (`cli.py:764-768`). Results persist only through the spec's own output sinks
  (`output_h5_path` / `output_dir`, `run/specs.py:217-221,536`).
- `[verified]` `host.runner.{sample,score,inspect,jacobian}` take `(spec=None, **kwargs)` and return
  `dict[str, Any]` (`host/runner.py:166,658,1186,1525`). The dict holds JAX arrays, e.g.
  `results["sample_indices"] = jnp.asarray(...)` (`host/runner.py:387`), so it is not JSON-serializable as is.
- `[verified]` JSON spec codec exists: `aminx.run.spec_json` (`run_specification_to_json_dict` used at
  `cli.py:371`). Array-valued spec fields (bias, fixed_mask, ...) raise `SpecJSONEncodeError` `[unverified]`.
- `[verified]` `campaign.py` takes `git_sha` as a caller-supplied string defaulting to `"unknown"`
  (`host/campaign.py:82,144,1100,1301`); the CLI never passes it.
- `[verified]` The browser-validation inventory guard scans only
  `inference, sampling, scoring, model, host, tiling, ebm, potts`
  (`tests/parity/test_browser_validation_inventory.py:20-29`). A new `aminx.agent` package is outside it.
- `[verified]` `using-aminx` has one copy, at `~/.claude/skills/using-aminx/SKILL.md`, not tracked in
  any repo. Its frontmatter carries `triggers:`; at least one statement is stale ("run inspect not yet
  wired" while `cli.py:950` wires it) `[unverified: recon-reported]`.
- `[verified]` No hardcoded `aminx run|spec|campaign` subprocess invocations in `src/`, `scripts/` or
  `.github/`. The command strings appear in docstrings, error messages, and the parity lane
  `scripts/parity/e2e_run_api_parity.py:30`, which drives `aminx run sample --emit-json`. A CLI rename
  therefore breaks parity lanes and docs, not cluster jobs.

### 3.2 cisternal 0.1.1a15

- `[verified]` Hard deps `cyclopts>=4.18.0,<5`, `fastmcp>=4`, `opentelemetry-api`, `pyyaml`; only extra is
  `otlp` (wheel METADATA). `requires-python >=3.13` matches aminx.
- `[verified]` **Spike S-1 (lock):** adding `agent = ["cisternal>=0.1.1a15"]` to a scratch copy of aminx's
  `pyproject.toml` and running `uv lock` with *no* `prerelease` setting exits 0 and resolves
  cisternal 0.1.1a15, fastmcp 4.0.11, cyclopts 4.25.3, typer 0.25.1. The xtrax "no cisternal extra" rule
  (xtrax `pyproject.toml:53-70`) was written against the old `fastmcp==4.0.0a2` pin (a3-a11) and does not
  apply at a12+.
- `[unverified: recon]` `@cisternal.tool(registry=, name=, cli_group=, cli_name=, cli_contract=)` is a pure
  marker (`registration/decorator.py:54-72`). `wire(server, app=None, *, registry, expected, ...)` registers
  a whole registry partition on a FastMCP server and/or a **cyclopts** App (`registration/wired.py:408-421`).
  Its `adapter=` argument is accepted and ignored. MCP telemetry needs
  `server.add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))` (maraxiom
  `telemetry_bridge.py:88-113`).
- `[unverified: recon]` The MCP callable is always `async def`; a sync tool body runs **on the event loop**
  (`registration/shim.py` dispatch). A JAX call made there blocks the server for the whole compile+run.
- `[unverified: recon]` Manifest (`assets/manifest.py`): `[plugin]`, `[[plugin.skills]]` (`name`, `path`,
  `description`, `triggers`; manifest triggers win), `[[plugin.agents]]`, `[[plugin.hook_specs]]`, one
  `[plugin.mcp]` (`command`, `launch = "path"|"uvx"`, `uvx_from`), `[plugin.marketplace]`. Paths resolve
  relative to the directory **containing** `.praxia/`.
- `[unverified: recon]` `assets export` / `inspect` exit 0 even when they drop assets; `assets validate`
  and `assets snapshot` fail on warnings. For a non-cisternal manifest `validate` is structural only.
- `[unverified: recon]` `cisternal.provenance.capture_git_state(cwd=None) -> GitState` never raises and
  records `provenance_source` (`provenance/channels.py:525`). It is pure stdlib.
- `[unverified: recon]` `cisternal.init()` must be called by the adopter; without it every
  `span`/`emit_event` is a no-op.

### 3.3 Reference adopters

`[unverified: recon]` **alynxr** is the cleanest layout (greenfield, single registry, both legs from one
tools module, packaged snapshot with a CI `--check`): `src/alynxr/{tools,mcp,cli,provenance}.py`,
`agent_assets/skills/*/SKILL.md`, `.praxia/manifest.toml`, `src/alynxr/agent_plugin.json`,
`tests/test_agent_plugin.py`, `tests/test_tool_surface.py`. It installs **no** MCP middleware; that part
should be copied from maraxiom. For a Typer CLI the precedents are maraxiom (CliContract, 261002) and bathos /
naurmalade (full ports, 260828-29). bathos only found about 15 regressions by driving every command through
the real binary.

### 3.4 Assumption ledger

| # | Assumption the design leans on | Tag | Spike |
|---|---|---|---|
| A1 | aminx locks with a cisternal extra without prerelease=allow | verified (S-1) | done |
| A2 | `wire(server, app=None, ...)` registers MCP-only, with no cyclopts App needed | unverified | S7-02 |
| A3 | Sync tool bodies block the FastMCP loop; `asyncio.to_thread` in an async body fixes it | unverified | S7-02 |
| A4 | `host.runner.*` can be driven from kwargs alone (no Typer context) for every knob the tools expose | unverified | S7-02 |
| A5 | Manifest skill paths resolve from repo root; `snapshot --check` fails on a stale skill | unverified | S7-03 |
| A6 | `capture_git_state` returns a 40-hex sha in a clean checkout and a non-"git" `provenance_source` in a wheel install | unverified | S7-09 |
| A7 | `plugin_app` (cyclopts) cannot mount under a Typer app, so the `plugin` verbs need a separate entry point | unverified | S7-06 |
| A8 | A long-lived server keeps the jit cache warm across calls with the same shapes | unverified | S7-08 |

## 4. Design

### 4.1 Package layout (additive)

```
pyproject.toml                         # + extra  agent = ["cisternal>=0.1.1a15"]
                                       # + script aminx-mcp = "aminx.agent.mcp:main"
.praxia/manifest.toml                  # new: [plugin] aminx, skills, [plugin.mcp]
agent_assets/skills/using-aminx/SKILL.md   # moved in from ~/.claude/skills, refreshed
agent_assets/skills/<more>/SKILL.md    # optional, S7-Q3
src/aminx/agent/__init__.py            # import-guarded: clear error if the extra is missing
src/aminx/agent/tools.py               # REGISTRY = "aminx"; @cisternal.tool bodies; TOOL_NAMES
src/aminx/agent/shaping.py             # results dict -> JSON summary + npz/zarr side file
src/aminx/agent/mcp.py                 # FastMCP("aminx") + wire(server, registry=..., expected=TOOL_NAMES)
                                       #   + CisternalMiddleware(PassthroughAdapter, reraise=True); main()
src/aminx/agent/provenance.py          # guarded capture_git_state, builtin fallback
src/aminx/agent_plugin.json            # packaged snapshot (package data)
tests/agent/                           # surface, shaping, snapshot, provenance tests
```

Rule (from the cisternal onboarding guide): `tools.py` never imports the CLI; `mcp.py` calls `wire()`
once. Nothing outside `aminx.agent` imports cisternal or fastmcp, except the guarded provenance hook in
`campaign.py` (4.6). `import aminx` must not get slower or gain a dependency (gate G-IMPORT).

### 4.2 Leg A: agent assets

- Move `using-aminx` into `agent_assets/skills/using-aminx/SKILL.md`. Refresh it against origin/main: CLI
  verbs, `inspect` now wired, JSON (not TOML) specs, the MCP tools. Put `description` and `triggers` in
  the **manifest**, because cisternal exports manifest triggers and drops frontmatter ones.
- `.praxia/manifest.toml`: `[plugin] name = "aminx"` (the package keeps its name, per revised D2; avoid
  `praxia` in any non-DNS name, per OQ-01), `skills_delivered_by_plugin = true` (stops praxia's legacy copy
  into `~/.claude/skills`), `[plugin.mcp] command = ["aminx-mcp"]` with `launch` per S7-Q2.
- Packaged snapshot `src/aminx/agent_plugin.json` via `cisternal assets snapshot`, with a CI `--check`
  test, so wheel installs (which have no `.praxia/`) can still install the plugin.
- Retire `~/.claude/skills/using-aminx` only after the plugin copy is verified loading (S7-14, user step:
  that directory is outside the repo and write-protected in agent sessions).

### 4.3 Leg B: MCP tool surface (v1)

Tools take **paths and JSON**, never arrays. They return a JSON summary plus paths to side files.

| Tool | Wraps | Returns |
|---|---|---|
| `sample` | `host.runner.sample` | sequences (decoded strings), per-sample score/perplexity, `outputs` paths, `spec` (JSON), `provenance` |
| `score` | `host.runner.score` | per-sequence scores, `outputs`, `spec`, `provenance` |
| `inspect` | `host.runner.inspect` | per-structure summary (lengths, chains, masks) and a logits side file |
| `jacobian` | `host.runner.jacobian` | side-file path + shape summary only (too large for JSON) |
| `spec_emit` | spec builders + `run_specification_to_json_dict` | spec JSON for `kind` in sample/score/inspect/jacobian/potts |
| `spec_validate` | `run_specification_from_json` | `{ok, errors}` |
| `list_checkpoints` | weights registry + `weight_provenance` | checkpoint ids, families, sha256, hub revision |

- **Arguments** mirror `SamplingSpecification` etc. as typed keyword parameters, so FastMCP derives a
  JSON schema from annotations. The ~38 shared run options become one optional `options: dict` validated
  by constructing the spec. Unknown keys are errors, not silently dropped. Exposing every knob as a top-level
  parameter would make a schema agents cannot use.
- **Execution.** Tool bodies are `async def` and run the runner via `asyncio.to_thread`, so the server
  stays responsive (A3). One call at a time per process (an `asyncio.Lock`). JAX device memory is not shared
  safely across concurrent runs, and agents rarely need concurrency here.
- **Output shaping** (`agent/shaping.py`), single owner: scalars and short lists go inline; any array over
  a size cap (default 10k elements) goes to `<output_dir>/<run_id>.npz` and is referenced by path; token
  arrays are decoded to sequence strings with aminx's own alphabet. Every result carries
  `{"aminx_version", "spec_sha256", "provenance"}`. `output_dir` defaults to an XDG cache dir, resolved by
  the same rule as `cli.py:127-129`. It is never a cwd write.
- **Naming.** v1 tool names are plain verbs (`sample`, `score`, ...). FastMCP namespaces them under the
  server name. When S4-19 lands, S7-13 adds manifest ids (`aminx/proteinmpnn.sample`) to each tool's
  description and result, so agents and the hub catalog agree without a rename.
- **Errors.** Tool bodies raise; `CisternalMiddleware(PassthroughAdapter(), reraise=True)` reports them.
  No body returns `{"error": ...}`.

### 4.4 Shared request normalization with `aminx serve`

S5-21 (`aminx serve`, HTTP wire v1) and `aminx-mcp` both turn a JSON request into a spec, run
`host.runner`, and shape the result. S7-05 puts that in `aminx.agent.shaping` +
`aminx.agent.requests` without importing cisternal there, so S5-21 can import it with no agent extra.
Whichever lands second must reuse it (an interface note in section 6, not a dependency edge).

### 4.5 Packaging

- `agent` extra as in spike S-1. `aminx-mcp` imports fail fast with
  `aminx-mcp needs the agent extra: pip install "aminx[agent]"`.
- `[plugin.mcp] launch`: `"uvx"` (rewrites to `uvx --from aminx[agent]==<ver> aminx-mcp`, works with no
  local venv) vs `"path"` (uses the project venv; right for dev). S7-Q2.
- Add the `--extra agent` lane to CI's sync line (`ci.yml:46`) for `tests/agent/` only.

### 4.6 Leg C: provenance and telemetry

- **Provenance.** `aminx.agent.provenance.capture()` tries `cisternal.provenance.capture_git_state`,
  falls back to a builtin `git rev-parse` shellout, then to `"unknown"`, and always records
  `provenance_source` so the difference is visible (the same contract xtrax uses,
  `xtrax/telemetry/record.py:350`). `campaign.py` call sites that default `git_sha="unknown"` take
  `capture().sha` when the caller passes nothing. Capture happens **per run, from the aminx source tree**,
  not per process from cwd. A long-lived MCP server would otherwise stamp a stale commit.
- **Telemetry.** `aminx-mcp` calls `cisternal.init()` at start (MCP leg only). The `aminx` CLI gets no
  telemetry unless `CISTERNAL_TELEMETRY` is in `{aminx, all, 1, true}`, the convention bathos, contemplex
  and maraxiom use. Spans: `aminx.tool.<name>` with `checkpoint_id`, `n_inputs`, `wall_s`,
  `compile_cache_hit`. Never sequences or structure contents.
- No overlap with bathos: bathos records *experiments*; this records *tool calls*. Neither replaces the other.

### 4.7 Leg D: the CLI port (user decision S7-Q1)

The bathos/naurmalade/maraxiom pattern deletes the Typer CLI and serves both legs from one `@tool`
function set via `wire()`. For aminx it costs more than it did there:

- The `run` group's shared options live on a Typer **callback** (`cli.py:448`) and are positioned *before*
  the subcommand (`aminx run --backbone-noise 0.1 sample ...`). cisternal's CliContract has no
  global-options mechanism (onboarding guide §7), so either the 38 options repeat per command or the
  flag order changes. Either way it breaks the published CLI (released on PyPI) and the
  `e2e_run_api_parity` lane.
- 285 annotated parameters; bathos needed a per-command drive-through to find ~15 regressions at a
  smaller size.
- The payoff is one function set for CLI and MCP. S7 already gets most of that by having both
  call `host.runner` and share `requests`/`shaping`.

**Recommendation: (a) defer.** Keep Typer. Revisit only if a second agent-facing CLI surface appears or
cisternal gains group-level shared options. Options offered in S7-Q1: (a) defer; (b) port the small groups
(`spec`, `potts`) to cyclopts via `wire()` under a new `aminx-agent` entry point that also hosts `plugin`
verbs, leaving `aminx` untouched; (c) full port with a deprecation release. Items S7-16..S7-18 exist only
for (b)/(c) and are marked `user_decision`.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A sync JAX call stalls the MCP loop; the client times out during first compile (minutes) | async + `to_thread` (S7-02 spike); tool description states cold-compile latency; progress notifications if FastMCP 4 supports them |
| Agents pass huge `options` and get silent defaults | spec construction validates; unknown keys raise (G-SURFACE negative control) |
| `assets export` silently drops a skill | CI uses `assets snapshot --check` / `validate` (both fail on warnings), never `export` exit code |
| Snapshot goes stale after a skill edit | `tests/agent/test_agent_plugin.py` runs `snapshot --check` |
| Skill drifts from the CLI again | S7-04 adds a test that every `aminx ...` command in the skill parses with `--help` |
| cisternal is alpha and APIs move | pin `>=0.1.1a15,<0.2`; surface tests assert against `await server.list_tools()`, not the registry |
| `import aminx` gains fastmcp | G-IMPORT: `python -X importtime -c "import aminx"` has no `fastmcp`/`cisternal`/`cyclopts` modules |
| Stale commit stamped by a long-lived server | provenance captured per run from the package source tree (A6 spike) |
| Heavy tests run locally and kill the box | all `tests/agent` JAX-touching tests run on titanix (repo rule); local runs are tiny-structure smoke only |

## 6. Interfaces

**Provides.** `aminx-mcp` (stdio MCP server); the `aminx` plugin bundle (Claude Code first, other cisternal
surfaces via `assets export --surface`); `aminx.agent.requests` + `aminx.agent.shaping` (cisternal-free;
reusable by S5-21); `aminx.agent.provenance.capture()`.

**Consumes.** cisternal >=0.1.1a15 (tool, wire, CisternalMiddleware, provenance, assets/plugin CLI);
`aminx.host.runner`, `aminx.run.spec_json`, weights registry.

**Ordering records (not hard edges).** S7-13 depends on S4-19 (manifest ids). S5-21 should consume S7-05
if S7-05 lands first (and vice versa). Neither blocks the S7 critical path.

## 7. Verification gates

Each gate pairs a positive control with a negative control that must fail.

- **G-LOCK** (S7-01): `uv lock --check` passes with the extra; `uv sync` without `--extra agent` installs
  no fastmcp. Negative: `python -c "import aminx.agent.mcp"` in the no-extra env raises the fail-fast message.
- **G-IMPORT** (S7-01): `-X importtime` of `import aminx` lists no cisternal/fastmcp/cyclopts. Negative:
  a test that injects `import cisternal` into a core module is caught by the same check.
- **G-SURFACE** (S7-07): `await server.list_tools()` names == `TOOL_NAMES`; each schema has the declared
  params. Negatives: `wire(..., expected=TOOL_NAMES + ["ghost"])` raises `CisternalWireError`; a tool
  call with an unknown `options` key raises.
- **G-PARITY** (S7-07, titanix): for a fixed seed and two structures (one multi-chain), the `sample`
  and `score` tool results equal `host.runner.sample/score` called directly with the same spec (sequences
  identical, scores to `pytest.approx(rel=1e-6)`). Negative: changing the seed changes the tool result,
  so the comparison is not vacuous.
- **G-SHAPE** (S7-05): every tool result passes `json.dumps`; an array above the cap goes to a side
  file that round-trips with equal contents. Negative: a cap of 0 forces every array to a side file, and
  the inline-only assertion fails.
- **G-SNAPSHOT** (S7-03/04): `cisternal assets snapshot --check` passes; `assets validate` exits 0.
  Negative: deleting a manifest-listed skill file makes both exit 1 (proves the check fires, unlike `export`).
- **G-PROV** (S7-09): in a clean checkout `capture()` gives a 40-hex sha with `provenance_source == "git"`;
  in a wheel install / no-cisternal env it gives the builtin or `"unknown"` source, never a fabricated sha.
  Negative: a dirty tree sets `dirty=True`.
- **G-LOAD** (S7-14, user-run): after `cisternal assets install` (or `aminx-agent plugin install claude`), a
  fresh Claude Code session lists the `using-aminx` skill from the plugin and `aminx-mcp` tools, and the
  `~/.claude/skills` copy is gone.

No gate produces a research finding. G-PARITY is an equality test on code paths, not a measurement,
so it is a test, not a bathos run.

## 8. Open questions for the user

| # | Question | Recommendation | Blocks |
|---|---|---|---|
| S7-Q1 | CLI port: (a) defer, (b) port `spec`/`potts` + `plugin` under a new `aminx-agent` entry, (c) full Typer->cyclopts port with deprecation release | **(a)**, revisit later | S7-16..18 |
| S7-Q2 | `[plugin.mcp] launch`: `uvx` (zero-setup, pins released version) or `path` (project venv) | `uvx` in the shipped snapshot, `path` documented for dev | S7-06 |
| S7-Q3 | Skills beyond `using-aminx` (e.g. `aminx-campaigns`, `aminx-potts`), and any agents | v1: only `using-aminx`, refreshed; split later if it grows past ~500 lines | S7-04 |
| S7-Q4 | Distribution: `[plugin.marketplace]` in aminx (own marketplace) or `cisternal assets publish-shared` into the shared local marketplace | shared marketplace (what most adopters do); no marketplace table | S7-06 |
| S7-Q5 | Should `jacobian` be in v1 (large outputs, slow) | yes, side-file only | S7-07 |

## 9. Backlog items

```toml
[[item]]
id = "S7-01"
title = "agent extra (cisternal>=0.1.1a15,<0.2) + aminx.agent package skeleton with fail-fast import guard; G-LOCK and G-IMPORT tests"
repo = "aminx"
size = "S"
depends_on = []
gate = "G-LOCK, G-IMPORT"
user_decision = false

[[item]]
id = "S7-02"
title = "Spike: wire() MCP-only registration, async+to_thread JAX body responsiveness, kwargs-only host.runner drive for every exposed knob (A2-A4); record answers in this spec"
repo = "aminx"
size = "S"
depends_on = ["S7-01"]
gate = "spike answers recorded with file:line; spike code deleted"
user_decision = false

[[item]]
id = "S7-03"
title = ".praxia/manifest.toml ([plugin] aminx, skills_delivered_by_plugin, [plugin.mcp]) + packaged agent_plugin.json snapshot + snapshot --check test"
repo = "aminx"
size = "S"
depends_on = ["S7-01"]
gate = "G-SNAPSHOT"
user_decision = false

[[item]]
id = "S7-04"
title = "Move using-aminx into agent_assets/skills, refresh against origin/main (inspect wired, JSON specs, MCP tools), manifest description+triggers; test that every `aminx ...` command in the skill parses with --help"
repo = "aminx"
size = "M"
depends_on = ["S7-03"]
gate = "G-SNAPSHOT; skill-command parse test with a negative control (a bogus verb fails)"
user_decision = true  # S7-Q3 scope

[[item]]
id = "S7-05"
title = "aminx.agent.requests + aminx.agent.shaping (cisternal-free): JSON request -> spec, results dict -> JSON summary + npz side files, token decoding, XDG output_dir"
repo = "aminx"
size = "M"
depends_on = ["S7-02"]
gate = "G-SHAPE"
user_decision = false

[[item]]
id = "S7-06"
title = "aminx-mcp entry point: FastMCP('aminx') + wire(registry='aminx', expected=TOOL_NAMES) + CisternalMiddleware(PassthroughAdapter, reraise=True) + cisternal.init(); plugin install path (cisternal assets install or aminx-agent plugin, per A7) and launch mode"
repo = "aminx"
size = "M"
depends_on = ["S7-03", "S7-05"]
gate = "server starts over stdio and answers list_tools in a subprocess test"
user_decision = true  # S7-Q2, S7-Q4

[[item]]
id = "S7-07"
title = "v1 tools: sample, score, inspect, jacobian, spec_emit, spec_validate, list_checkpoints (async, to_thread, single-run lock)"
repo = "aminx"
size = "L"
depends_on = ["S7-05", "S7-06"]
gate = "G-SURFACE, G-PARITY (titanix)"
user_decision = false

[[item]]
id = "S7-08"
title = "Warm-cache behaviour: model + jit cache reuse across calls in one server process; document cold-compile latency in tool descriptions (A8)"
repo = "aminx"
size = "S"
depends_on = ["S7-07"]
gate = "second identical-shape call skips compile (compile_cache_hit span field true); negative: a new shape reports false"
user_decision = false

[[item]]
id = "S7-09"
title = "aminx.agent.provenance.capture(): cisternal capture_git_state -> builtin git -> 'unknown', always recording provenance_source; per-run, from the package source tree"
repo = "aminx"
size = "S"
depends_on = ["S7-01"]
gate = "G-PROV"
user_decision = false

[[item]]
id = "S7-10"
title = "campaign.py: fill git_sha from capture() when the caller passes none (sites at host/campaign.py:1100,1301); record provenance_source beside it"
repo = "aminx"
size = "S"
depends_on = ["S7-09"]
gate = "campaign manifest test asserts a real sha in a clean checkout; negative: explicit git_sha argument still wins"
user_decision = false

[[item]]
id = "S7-11"
title = "Opt-in CLI telemetry: cisternal.init() + aminx.cli.* spans only when CISTERNAL_TELEMETRY in {aminx,all,1,true}; no-op otherwise"
repo = "aminx"
size = "S"
depends_on = ["S7-09"]
gate = "unset env -> no events file written; set -> one start/end pair per command"
user_decision = false

[[item]]
id = "S7-12"
title = "CI: --extra agent lane for tests/agent (non-JAX parts); JAX-touching agent tests marked for titanix"
repo = "aminx"
size = "S"
depends_on = ["S7-07"]
gate = "lane green; lane fails when the snapshot is made stale on purpose"
user_decision = false

[[item]]
id = "S7-13"
title = "Align tool metadata with S4 ModelManifest ids (aminx/proteinmpnn.sample|score) in descriptions and results; no rename"
repo = "aminx"
size = "S"
depends_on = ["S7-07", "S4-19"]
gate = "each tool result's manifest_id resolves in the S4-19 manifests"
user_decision = false

[[item]]
id = "S7-14"
title = "Install + load check in a fresh Claude Code session; retire ~/.claude/skills/using-aminx (user step)"
repo = "aminx"
size = "S"
depends_on = ["S7-04", "S7-07"]
gate = "G-LOAD"
user_decision = true

[[item]]
id = "S7-15"
title = "Fold S7 into a regenerated ecosystem DAG (ids, edges, topological order) alongside S1-S6"
repo = "aminx"
size = "S"
depends_on = []
gate = "praxia docs check passes; DAG acyclicity re-checked by code"
user_decision = false

[[item]]
id = "S7-16"
title = "(only if S7-Q1 = b/c) aminx-agent cyclopts entry point hosting plugin_app + wired spec/potts groups"
repo = "aminx"
size = "M"
depends_on = ["S7-07"]
gate = "every ported command driven through the real binary; flag-for-flag diff vs Typer help"
user_decision = true

[[item]]
id = "S7-17"
title = "(only if S7-Q1 = c) port run/campaign groups to cyclopts via wire(); shared-options strategy; deprecation release"
repo = "aminx"
size = "L"
depends_on = ["S7-16"]
gate = "e2e_run_api_parity lane green on the new CLI; old invocations print a deprecation pointer"
user_decision = true

[[item]]
id = "S7-18"
title = "(only if S7-Q1 = c) drop typer from base deps after one release with both CLIs"
repo = "aminx"
size = "S"
depends_on = ["S7-17"]
gate = "uv lock --check; no typer import in src"
user_decision = true
```

### 9.1 Order

| # | id | title | repo | size | depends_on | user_decision |
|---|---|---|---|---|---|---|
| 1 | S7-01 | agent extra + aminx.agent skeleton | aminx | S | - | no |
| 2 | S7-15 | fold S7 into ecosystem DAG | aminx | S | - | no |
| 3 | S7-02 | spike A2-A4 | aminx | S | S7-01 | no |
| 4 | S7-03 | manifest + snapshot | aminx | S | S7-01 | no |
| 5 | S7-09 | provenance.capture() | aminx | S | S7-01 | no |
| 6 | S7-04 | using-aminx into repo, refreshed | aminx | M | S7-03 | yes |
| 7 | S7-05 | requests + shaping | aminx | M | S7-02 | no |
| 8 | S7-10 | campaign git_sha | aminx | S | S7-09 | no |
| 9 | S7-11 | opt-in CLI telemetry | aminx | S | S7-09 | no |
| 10 | S7-06 | aminx-mcp entry + install path | aminx | M | S7-03, S7-05 | yes |
| 11 | S7-07 | v1 tools | aminx | L | S7-05, S7-06 | no |
| 12 | S7-08 | warm cache | aminx | S | S7-07 | no |
| 13 | S7-12 | CI agent lane | aminx | S | S7-07 | no |
| 14 | S7-14 | install + load check | aminx | S | S7-04, S7-07 | yes |
| 15 | S7-13 | align with S4 manifest ids | aminx | S | S7-07, S4-19 | no |
| 16 | S7-16 | aminx-agent cyclopts entry (Q1 b/c) | aminx | M | S7-07 | yes |
| 17 | S7-17 | full CLI port (Q1 c) | aminx | L | S7-16 | yes |
| 18 | S7-18 | drop typer (Q1 c) | aminx | S | S7-17 | yes |

Critical path: S7-01 -> S7-02 -> S7-05 -> S7-06 -> S7-07 -> S7-14 (six items, two of them M and one L).
Under the recommended S7-Q1 = (a), S7-16..18 are not filed, leaving 15 items.

## Revision log

- 261006: first draft (task 261006_cisternal-cutover-spec). Recon from two read-only agent passes, with
  load-bearing claims re-read at source; spike S-1 (lock) run on a scratch copy. Not yet adversarially
  reviewed.
