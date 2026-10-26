---
title: 'Incremental CI test selection via target-aware static import closures'
description: 'Select only diff-implicated pytest modules plus a 20-file always-run guard set on pull_request CI runs, with fail-open soundness, XLA compilation caching, and concurrency/timeout hardening'
status: implemented
task_id: 261026_incremental-ci-test-selection
date: '261026'
backlog_ids: 'aminx debt #2691, #2606, #2585, #2582'
---
# Incremental CI test selection via target-aware static import closures

## 1. Problem Statement

Every `pull_request` run of `.github/workflows/ci.yml` executes the entire fast unit-test suite (`uv run pytest -n auto -m "not slow and not parity_heavy and not parity_audit and not redsox_gate"`) across all **414** test files on a 4-vCPU `ubuntu-latest` GitHub Actions runner:

1. **Runtime & Runner OOM Shutdowns (Debt #2582, #2585, #2712)**: Full-suite PR runs take 30–40+ minutes and repeatedly suffer `The runner has received a shutdown signal` around ~91% completion due to accumulated JAX/XLA JIT compilation caches and worker RSS across 4 `pytest-xdist` workers.
2. **Unnecessary Re-execution on Scoped PRs (Debt #2691)**: A change isolated to `src/aminx/potts/`, `src/aminx/families/laser_mpnn/`, `src/aminx/ebm/`, `src/aminx/agent/`, or documentation currently compiles and executes every unrelated model family, port wave, host campaign test, and ONNX export test.
3. **Cold XLA Compilation & Uncancelled Superseded Runs (Debt #2606)**: Every PR run compiles identical JAX/Equinox kernels from scratch without a persistent `JAX_COMPILATION_CACHE_DIR`, and pushing a follow-up commit to an open PR leaves superseded runs consuming runner capacity with no `concurrency` cancellation or job `timeout-minutes`.

Meanwhile, `tests/knob_gate/_closure.py` and `tests/knob_gate/closure_edges.toml` (accepted in ADR `.praxia/docs/decisions/261008_per-row-closure-staleness.md`) already proved that static AST import closures + explicit reviewed edge declarations can soundly isolate cross-family dependencies for the 9 graded parity vehicles.

## 2. Evaluation of Dependency Sources

We evaluated four approaches for mapping a git diff (`origin/<base_ref>...HEAD`) to affected pytest modules:

| Criterion | Generalized Static AST Closure (`select_affected_tests.py`) | `pytest-testmon` (Coverage.py DB) | `pytest-picked` / Naive File Name Matching | Bazel (`py_test` targets) |
| :--- | :--- | :--- | :--- | :--- |
| **Prior CI state required** | **None** — computed deterministically from the git checkout in ~1.5 s | Requires a warm `.testmondata` SQLite artifact from the exact base commit; cache miss or schema drift forces full run | None | None |
| **Runtime overhead** | **~1.5 s** pre-pytest step (0% pytest execution overhead) | `sys.monitoring` / `coverage.py` tracing overhead (~15–35% slower per test; interacts poorly with JAX JIT & `xdist`) | Negligible | High migration and maintenance cost |
| **Soundness on transitive imports** | **Sound** (full transitive import graph + declared soft/data edges + fail-open rules) | Sound only for Python lines traced on the previous run; misses branches guarded by platform/env or C++/XLA boundaries | **Unsound** — misses transitive callers completely | Sound only if `BUILD` files are manually kept in sync with imports |
| **Compatibility with `pytest -n auto`** | **Native** — passes explicit test file list to `pytest -n auto` | Fragile across `xdist` workers and dynamic fixture collection | Native | Replaces `pytest` runner |

**Decision**: Generalize the static AST closure engine from `tests/knob_gate/_closure.py` + `tests/knob_gate/closure_edges.toml` into `scripts/ci/select_affected_tests.py` and `scripts/ci/test_closure_edges.toml`.

## 3. Soundness Model & Target-Aware Closure Mechanics

### 3.1 Why Naive Package-Ancestor Expansion Over-Approximates `aminx`

In `tests/knob_gate/_closure.py`, any `from a.b.c import X` adds `_ancestors("a.b.c")` (`a`, `a.b`, `a.b.c`) to the BFS expansion queue. For `aminx`, `src/aminx/__init__.py` eagerly re-exports `.io.parsing`, `.io.weights`, `.run.specs`, `.runtime`, `.sampling`, and `.scoring`; furthermore, `src/aminx/run/__init__.py` imports `src/aminx/run/_exports.py`, which lazily imports `aminx.host.runner`. Under naive ancestor expansion, any test that imports `from aminx.potts.runner import X` expands `src/aminx/__init__.py` and pulls `sampling`, `scoring`, and `host/runner.py` into **363 of 414** test closures.

### 3.2 Target-Aware Import Closure with Ancestor File Tracking

In Python, executing `from aminx.potts.runner import X` executes `src/aminx/__init__.py` and `src/aminx/potts/__init__.py`, and then loads `src/aminx/potts/runner.py`. Consequently:

1. **Ancestor `__init__.py` File Recording**: For every imported module `a.b.c`, the `__init__.py` of every ancestor package (`src/aminx/__init__.py`, `src/aminx/potts/__init__.py`) is added to `closure.files`. Any direct edit to an ancestor `__init__.py` (or anything `src/aminx/__init__.py` itself directly imports when `src/aminx/__init__.py` is the target of `import aminx` / `from aminx import ...`) invalidates all tests whose closure contains that `__init__.py`.
2. **Target-Only BFS Expansion**: The BFS queue recurses into the resolved target module (`a.b.c` for `import a.b.c`; `base.name` if `base.name` is a submodule/subpackage for `from base import name`, or `base` if `name` is a symbol defined/re-exported in `base/__init__.py`).
3. **Python 3 Package Import Scoping**: Inside `src/aminx/`, absolute imports (`node.level == 0`, e.g. `import typing`) resolve only against `_ROOTS` and stdlib, never implicitly against the current subpackage directory (`src/aminx/utils/typing.py`), matching Python 3's import semantics.

With target-aware import closure:
- Modifying `src/aminx/__init__.py` directly selects **364 / 414** test files.
- Modifying `src/aminx/potts/runner.py` implicates **21** test files (**39 / 414** total with guards, **9.4%** of suite).
- Modifying `src/aminx/families/laser_mpnn/driver.py` implicates **14** test files (**33 / 414** total with guards, **8.0%** of suite).
- Modifying `src/aminx/ebm/model.py` implicates **11** test files (**31 / 414** total with guards, **7.5%** of suite).
- Modifying `src/aminx/agent/mcp.py` implicates **2** test files (**22 / 414** total with guards, **5.3%** of suite).

### 3.3 Static Resolution of Non-Import Dependencies

Beyond `import` and `from ... import` statements (at any AST depth, including lazy imports inside functions), `scripts/ci/select_affected_tests.py` statically resolves:

1. **Subdirectory `conftest.py` and `__init__.py` Hierarchy**: Every test file `tests/<subpath>/test_*.py` automatically includes all ancestor `tests/<subpath>/conftest.py` and `tests/<subpath>/__init__.py` files in its closure. Root `tests/conftest.py` is a global fallback trigger (`mode = "full"`).
2. **`Path(__file__)` & `BinOp(op=Div)` Constant Propagation**: Top-level and local `Path(__file__).resolve().parents[k] / "a" / "b"`, `.with_name(...)`, `.joinpath(...)`, and repo-relative path literals (`scripts/...`, `tests/...`, `src/...`, `.praxia/...`, `agent_assets/...`, `tools/...`, `NOTICE-weights.md`) are evaluated statically. When passed to `spec_from_file_location` or `runpy.run_path`, the target `.py` script is added to the BFS queue and its transitive imports are scanned automatically (resolving 36/47 dynamic load sites across the repo without manual TOML entries).
3. **`@pytest.mark.port_wave("<wave>")` Resolution**: Every port wave test (`tests/port/test_*.py`) is automatically linked to `tests/port/targets/<wave>.toml`, its `wave_id` manifest `tests/port/manifests/<wave_id>.toml`, and its `[oracle].reference_import` module `tests/port/<reference_subtree>/algo.py` + directory prefix `tests/port/<reference_subtree>/`.
4. **Dotted `"aminx.<pkg>.<mod>..."` String Literals in Tests**: String constants matching `^aminx(\.[A-Za-z_][A-Za-z0-9_]*)+$` in test modules (e.g. `monkeypatch.setattr("aminx.host.kernel_dispatch._safe_map", ...)`, `unittest.mock.patch(...)`, `sys.modules["aminx.families..."]`, or `subprocess.run([..., "-m", "aminx.agent.mcp"])`) resolve to their longest existing module under `src/aminx/`.
5. **Reviewed Edge Declarations (`scripts/ci/test_closure_edges.toml`)**: Covers remaining dynamic dispatch sites (`_DRIVER_BACKED_FAMILIES` in `src/aminx/host/runner.py`, `importlib.resources.files` in `src/aminx/io/weights.py` and `src/aminx/agent/tools.py`, `spec_from_file_location` in `src/aminx/families/potts_mpnn/driver.py`, and external reference loaders in `tests/parity/` and `scripts/browser_validation/`).

### 3.4 Fail-Open Rules

The selector operates in three modes (`full`, `incremental`, `guards_only`) and **never fails closed**:

1. **Git Diff Failure**: If `git diff --name-only <base>...<head>` exits non-zero (e.g. shallow clone or missing ref), fail open to `mode = "full"`.
2. **Unsound Closure (`not closure.sound`)**: Any file in any test closure containing an unresolved dynamic call (`import_module(<non-literal>)`, `__import__(<non-literal>)`, `spec_from_file_location(<non-resolvable>)`, `run_path(<non-resolvable>)`, `entry_points`, `iter_modules`, `files`) without `dynamic_ok = true` in `scripts/ci/test_closure_edges.toml` forces `mode = "full"` and fails `tests/ci/test_select_affected_tests.py`.
3. **Global Fallback Triggers (`[global]` in `test_closure_edges.toml`)**: Modifying `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `tests/conftest.py`, `scripts/ci/select_affected_tests.py`, `scripts/ci/test_closure_edges.toml`, or anything under `src/aminx/model_params/` forces `mode = "full"`.
4. **Deleted or Renamed Code/Test Files**: Any path in the diff under `src/`, `tests/`, or `scripts/` that no longer exists on disk at `HEAD` forces `mode = "full"` so broken imports across the suite cannot be missed.
5. **Zero-Closure Fail-Open on `src/` and `tests/`**: Any modified file under `src/` (including `.py` modules in `[unreferenced_src].allowlist` or non-`.py` package data) or any non-Markdown file under `tests/` that matches **zero** test closures forces `mode = "full"`. Furthermore, `.py` files under `src/` must match `closure.files` directly (a directory `data_prefix` from an AST lint test does not mask an unreferenced `src/aminx/*.py` module).

## 4. Always-Run Guard Set

Twenty fast repository-wide guard, boundary, and AST lint test modules run on **every** PR (`incremental` and `guards_only` modes) because they inspect the codebase via `Path.rglob` / `ast.parse` / `ast-grep` without importing every module they audit:

1. `tests/ci/test_select_affected_tests.py` (selector soundness invariants + positive/negative controls)
2. `tests/lint/test_l_drv.py`
3. `tests/lint/test_options_plumbing.py`
4. `tests/lint/test_port_tolerances_match_targets.py`
5. `tests/lint/test_potts_import_boundary.py`
6. `tests/lint/test_rs6b_flat_field_gate.py`
7. `tests/lint/test_sidecar_conditions.py`
8. `tests/lint/test_xtrax_public_imports.py`
9. `tests/test_alphabet_conformance.py`
10. `tests/test_import_isolation.py`
11. `tests/test_port_guards_controls.py`
12. `tests/test_xtrax_boundary_lint.py`
13. `tests/audit/test_F001_version_invariant.py`
14. `tests/host/test_schema_versions_consolidation.py`
15. `tests/inference/decode/test_library_surface.py`
16. `tests/parity/test_browser_validation_inventory.py`
17. `tests/parity/test_litparity_schema.py`
18. `tests/potts/test_import_boundaries.py`
19. `tests/sampling/test_x_draw_registry.py`
20. `tests/training/test_trainer_alphabet.py`

Additionally, `.github/workflows/ci.yml` continues to run the collection-time ONNX toolchain verification (`import onnx, onnxruntime, jax2onnx`) before invoking `pytest`, and root `tests/conftest.py` continues to enforce the per-test JAX `x64` leak guard (`_guard_jax_x64`) and port tolerance verification (`pytest_collection_modifyitems`).

## 5. CI Workflow Hardening & Caching (`.github/workflows/ci.yml`)

1. **Concurrency Cancellation**:
   ```yaml
   concurrency:
     group: ci-${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: ${{ github.event_name == 'pull_request' }}
   ```
   Cancels superseded PR runs immediately when new commits are pushed to a branch, while never cancelling `push` runs on `main`.
2. **Job Timeout**: `timeout-minutes: 45` on `unit-tests` prevents hung workers from blocking runners for the 6-hour GitHub Actions default (Debt #2582).
3. **Full Git History on PRs**: `fetch-depth: 0` on `actions/checkout@v4` ensures `origin/${{ github.base_ref }}...HEAD` is always available for `select_affected_tests.py`.
4. **Persistent XLA Compilation Cache & `uv` Cache (Debt #2606)**:
   - Sets `JAX_COMPILATION_CACHE_DIR: ${{ runner.temp }}/jax_compilation_cache` and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS: "0.5"`.
   - Restores/saves `${{ runner.temp }}/jax_compilation_cache` via `actions/cache@v4` keyed on `runner.os`, Python version (`3.13`), and `hashFiles('uv.lock')`.
5. **Marker Orthogonality & `main` Safety Net**:
   - On `pull_request`: runs `scripts/ci/select_affected_tests.py --base "origin/${{ github.base_ref }}" --head HEAD --explain --write-pytest-args .pytest-targets.txt` and passes those targets to `uv run pytest -n auto -m "$MARKER"`.
   - On `push` to `main`: continues to run the **entire** test suite with `--cov=src --cov-report=xml --cov-report=term` as the post-merge safety net.

## 6. Empirical Historical Replay & Control Verification

Replaying `scripts/ci/select_affected_tests.py --replay 200` over the last **200 first-parent commits** on `origin/main` (`a6e1573b`):

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total repository test files** | **414** | `tests/**/test_*.py` (413 existing + `tests/ci/test_select_affected_tests.py`) |
| **Commits analyzed (`--first-parent -n 200`)** | **200** | Merged PRs and direct commits on `origin/main` |
| **Mode breakdown** | **147 `incremental` (73.5%)**, **33 `guards_only` (16.5%)**, **20 `full` (10.0%)** | Full runs triggered on `pyproject.toml`, `uv.lock`, `tests/conftest.py`, or `.github/workflows/ci.yml` edits |
| **Median selected test files (overall)** | **37 / 414 (8.9%)** | **11.2x reduction** in median test files executed per commit |
| **Mean selected test files (overall)** | **87.3 / 414 (21.1%)** | **4.7x reduction** in mean test files executed (including the 10% full-suite fallback runs) |
| **Mean selected test files (`incremental` only)** | **58.0 / 414 (14.0%)** | Median `incremental` run selects **37 / 414 (8.9%)** test files |
| **Co-change recall (`src/` diff -> co-changed `tests/` file)** | **42 / 42 (100.0%)** | Every commit that modified both `src/` and an existing `tests/` file selected the modified test from the `src/` paths alone |

### Sample Replay Rows (Most Recent Commits on `origin/main`)

| Commit | Subject | Mode | Changed Files | Selected Tests | % of Suite | Reason |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `a6e1573b` | `feat(decode): opt-in AR wave loop on xtrax.intent.loop` | `full` | 10 | 414 / 414 | 100.0% | Global trigger: `pyproject.toml` |
| `7eabce4c` | `feat(potts): implement multi-state PoE inference` | `incremental` | 3 | 39 / 414 | 9.4% | 21 implicated + 20 guards |
| `cfab450c` | `fix(potts): refuse multi-state input instead of state 0` | `incremental` | 2 | 38 / 414 | 9.2% | 19 implicated + 20 guards |
| `0729c458` | `chore(deps): bump xtrax pins 0.4.1b2 -> 0.4.1b3 (#229)` | `full` | 2 | 414 / 414 | 100.0% | Global trigger: `pyproject.toml` |
| `f940fb4e` | `fix(potts): forward the k-NN adjacency W to PottsResult (#228)` | `incremental` | 1 | 38 / 414 | 9.2% | 19 implicated + 20 guards |
| `11a0f150` | `feat(potts): condition_on_fixed_sites (#227)` | `incremental` | 4 | 56 / 414 | 13.5% | 41 implicated + 20 guards |
| `782dba6c` | `fix(run)!: rename RunSpec.precision to precision_config` | `incremental` | 6 | 248 / 414 | 59.9% | 234 implicated + 20 guards |
| `b2dbb2ba` | `Merge PR #221 from maraxen/harness/261016` | `guards_only` | 2 | 20 / 414 | 4.8% | Non-code/doc paths only |
| `cb06fbbc` | `Merge PR #217 from maraxen/wt/261013-laser-f32` | `incremental` | 36 | 86 / 414 | 20.8% | 68 implicated + 20 guards |
| `6c6a1245` | `Merge PR #210 from maraxen/wt/261008-protonpotts-feat` | `incremental` | 5 | 36 / 414 | 8.7% | 17 implicated + 20 guards |
| `24fd3d28` | `Merge PR #209 from maraxen/wt/261008-protonpotts-p4b` | `guards_only` | 3 | 20 / 414 | 4.8% | Non-code/doc paths only |
| `eeb58303` | `Merge PR #208 from maraxen/wt/261008-protonpotts-p5` | `incremental` | 5 | 48 / 414 | 11.6% | 29 implicated + 20 guards |
