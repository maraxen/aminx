---
title: 'Incremental CI test selection via target-aware static import closures'
description: 'Select only diff-implicated pytest modules plus a 20-file always-run guard set on pull_request CI runs, with fail-open soundness, XLA compilation caching, and concurrency/timeout hardening'
status: implemented
adversarial_review: 'research subagent 261026 + 3-subagent bigger-picture audit (Static Analysis, CI Pipeline, Architecture): all findings addressed'
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

### 3.1 Why Naive Package-Ancestor & `TYPE_CHECKING` Expansion Over-Approximates `aminx`

In `tests/knob_gate/_closure.py`, any `from a.b.c import X` adds `_ancestors("a.b.c")` (`a`, `a.b`, `a.b.c`) to the BFS expansion queue, and `ast.walk` visits `if TYPE_CHECKING:` blocks. For `aminx`:
1. `src/aminx/__init__.py` eagerly re-exports `.io.parsing`, `.io.weights`, `.run.specs`, `.runtime`, `.sampling`, and `.scoring`; `src/aminx/run/__init__.py` imports `src/aminx/run/_exports.py` (`aminx.host.runner`); `src/aminx/model/__init__.py` re-exports all model classes; and `src/aminx/types/__init__.py` re-exports all type submodules.
2. `from __future__ import annotations` is active across `src/aminx/`, and modules such as `src/aminx/types/arrays.py`, `src/aminx/types/protocols.py`, `src/aminx/types/stages.py`, `src/aminx/utils/autoregression.py`, and `src/aminx/inference/optimize_ste.py` contain `if TYPE_CHECKING:` upward imports that never execute at runtime. Walking `if TYPE_CHECKING:` blocks or expanding climbed ancestor `__init__.py` files on relative imports fuses 78 `src/aminx/` modules into a single strongly connected component (SCC) implicated in 350/414 tests.

### 3.2 Target-Aware Runtime Import Closure with Ancestor File Tracking

In Python, executing `from aminx.potts.runner import X` executes `src/aminx/__init__.py` and `src/aminx/potts/__init__.py`, and then loads `src/aminx/potts/runner.py`. Consequently:

1. **Ancestor `__init__.py` File Recording Without Star-Hub Expansion**: For every imported module `a.b.c` (and every climbed package in a relative import `from .spec import ...`), the `__init__.py` of every ancestor package is added to `closure.files` so any direct edit to that `__init__.py` invalidates all tests whose closure traverses that package. However, star-hub ancestor packages in `_NON_EXPANDED_ANCESTOR_PKGS = {"aminx", "aminx.run", "aminx.model", "aminx.types"}` are only expanded in the BFS queue when imported directly as the target of an import (not merely traversed as an ancestor path prefix).
2. **`if TYPE_CHECKING:` Skipping**: `_iter_runtime_ast` skips the body of `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` blocks (while still walking `orelse`) because those imports never execute at runtime (`TYPE_CHECKING` is `False`).
3. **Python 3 Package Import Scoping & Stdlib Non-Shadowing**: Top-level stdlib modules (`parts[0] in sys.stdlib_module_names`, e.g. `import io`, `import types`, `import logging`) never resolve to same-named test directories (`tests/io/`, `tests/types/`), and imports inside `src/aminx/` (`importer_dir is None`) search only `[repo / "src", repo]`. Relative imports (`node.level > 0`) resolve directly against `file_resolved.parents[node.level - 1]`.

With these rules, the 78-file mega-SCC is completely eliminated (largest SCC is just 12 core model/io files):
- Modifying `src/aminx/__init__.py` directly implicates **362** test files (**376 / 414** total with guards, **90.8%** of suite).
- Modifying `src/aminx/io/designs.py` or `src/aminx/utils/catjac.py` implicates **1** test file (**21 / 414** total with guards, **5.1%** of suite — down from 350+ before breaking the SCC).
- Modifying `src/aminx/potts/runner.py` implicates **18** test files (**38 / 414** total with guards, **9.2%** of suite).
- Modifying `src/aminx/families/laser_mpnn/driver.py` implicates **25** test files (**45 / 414** total with guards, **10.9%** of suite).
- Modifying `src/aminx/ebm/model.py` implicates **11** test files (**31 / 414** total with guards, **7.5%** of suite).
- Modifying `src/aminx/agent/mcp.py` implicates **3** test files (**23 / 414** total with guards, **5.6%** of suite).

### 3.3 Static Resolution of Non-Import Dependencies

Beyond `import` and `from ... import` statements (at any runtime AST depth, including lazy imports inside functions), `scripts/ci/select_affected_tests.py` statically resolves:

1. **Subdirectory `conftest.py` and `__init__.py` Hierarchy**: Every test file `tests/<subpath>/test_*.py` automatically includes all ancestor `tests/<subpath>/conftest.py` and `tests/<subpath>/__init__.py` files in its closure. Root `tests/conftest.py` (along with its imported helpers `tests/__init__.py`, `tests/port_guards.py`, and `tests/port/mutant_registry.py`) is a global fallback trigger (`mode = "full"`).
2. **Lexically Scoped `Path(__file__)` & `BinOp(op=Div)` Constant Propagation**: Single-assignment module-level and function-local `Path(__file__).resolve().parents[k] / "a" / "b"`, `.with_name(...)`, `.joinpath(...)` expressions are evaluated per lexical scope (`_build_scope_path_envs`), with function parameters and reassigned variables tainted so cross-function variable names can never shadow or silence dynamic calls. When passed to `spec_from_file_location` or `runpy.run_path`, the target `.py` script is added to the BFS queue and its transitive imports are scanned automatically.
3. **`@pytest.mark.port_wave("<wave>")` Resolution**: Every port wave test (`tests/port/test_*.py`) is automatically linked via `_resolve_port_wave` (reading the `[port]` table in `tests/port/targets/<wave>.toml`) to its target TOML, `tests/port/manifests/<wave_id>.toml`, `tests/port/<reference_subtree>/*.py` + directory prefix, `symbol_qualname` under `src/aminx/`, and `tests/port/mutants/<wave>.py`.
4. **Dotted `"aminx.<pkg>.<mod>..."`, Inline `import aminx...`, & Family Name Strings in Non-Guard Tests**: String constants matching `^aminx(\.[A-Za-z_][A-Za-z0-9_]*)+$`, inline subprocess snippets `import aminx...` / `from aminx...`, or family identifier strings / checkpoint prefixes (`DRIVER_BACKED_FAMILY_MODULES`, e.g. `"laser-mpnn"`, `"potts-mpnn"`, `"protonpotts"`, `"thermompnn"`) in non-guard test modules resolve directly to their corresponding modules under `src/aminx/`. String literals inside `always_run` guard tests are deliberately excluded so guard/self-test assertions do not pollute `src/` coverage or mask uncovered modules.
5. **Exact-Count Reviewed Edge Declarations (`scripts/ci/test_closure_edges.toml`)**: Covers remaining dynamic dispatch sites (`_DRIVER_BACKED_FAMILIES` in `src/aminx/host/runner.py` — also verified by `test_family_driver_modules_and_conftest_global_invariants` to stay in 1-to-1 sync with `DRIVER_BACKED_FAMILY_MODULES` — `importlib.resources.files` in `src/aminx/io/weights.py` and `src/aminx/agent/tools.py`, `spec_from_file_location` in `src/aminx/families/potts_mpnn/driver.py`, and external reference loaders in `tests/parity/` and `scripts/browser_validation/`). Each file with unresolved dynamic calls pins `dynamic_count = N` (exact integer count of reviewed call sites); adding or removing even a single dynamic call site causes a count mismatch and fails open.

### 3.4 Fail-Open Rules

The selector operates in three modes (`full`, `incremental`, `guards_only`) and **never fails closed**:

1. **Git Diff Failure**: If `git diff --name-only <base>...<head>` exits non-zero (e.g. shallow clone or missing ref), fail open to `mode = "full"`.
2. **Unsound Closure (`not closure.sound`) or `SyntaxError`**: Any file in any test closure containing an unparseable `SyntaxError` or a number of unresolved dynamic calls (`len(scan.dynamic_lines)`) not equal to its declared `dynamic_count` in `scripts/ci/test_closure_edges.toml` forces `mode = "full"` and fails `tests/ci/test_select_affected_tests.py`.
3. **Global Fallback Triggers (`[global]` in `test_closure_edges.toml`)**: Modifying `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `tests/__init__.py`, `tests/conftest.py`, `tests/port_guards.py`, `tests/port/mutant_registry.py`, `scripts/ci/select_affected_tests.py`, `scripts/ci/test_closure_edges.toml`, or anything under `src/aminx/model_params/` forces `mode = "full"`.
4. **Deleted or Renamed Code/Test/Script Files**: Any path in the diff under `src/`, `tests/`, `scripts/`, or `tools/` that no longer exists on disk at `HEAD` forces `mode = "full"` so broken imports across the suite cannot be missed.
5. **Zero PR-Runnable Closure Fail-Open on `src/` and `tests/`**: Any modified `.py` file under `src/` that matches **zero PR-runnable non-guard test closures** (`not is_pr_excluded_test_file`, i.e., not 100% deselected by `-m "not slow and not parity_heavy and not parity_audit and not redsox_gate"`, including files in `[unreferenced_src].allowlist` such as `src/aminx/potts/designer.py` and `src/aminx/types/host_protocols.py`), any non-`.py` file under `src/` matching zero closures, or any non-Markdown file under `tests/` matching zero closures forces `mode = "full"`. Furthermore, `.py` files under `src/` or `tests/` must match `closure.files` directly (a directory `data_prefix` cannot mask an unreferenced `.py` module).

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
     group: ci-${{ github.workflow }}-${{ github.event_name == 'pull_request' && github.ref || github.sha }}
     cancel-in-progress: ${{ github.event_name == 'pull_request' }}
   ```
   Cancels superseded PR runs immediately when new commits are pushed to a PR branch, while keying `push` runs on `main` by `github.sha` so queued `main` runs are never dropped.
2. **Job Timeout & Removed Redundant `npm install -g @ast-grep/cli`**: `timeout-minutes: 45` on `unit-tests` (and `timeout-minutes: 10` on `epic-attribution-lint`, which also scopes its commit-range scan to `origin/${{ github.base_ref }}..${{ github.event.pull_request.head.sha }}` on PRs) prevents hung workers from blocking runners for the 6-hour GitHub Actions default (Debt #2582). Removed the redundant `npm install -g @ast-grep/cli` step since `ast-grep-cli>=0.45` is already installed by `uv sync --extra dev`.
3. **Full Git History on PRs**: `fetch-depth: 0` on `actions/checkout@v4` ensures `origin/${{ github.base_ref }}...HEAD` is always available for `select_affected_tests.py`.
4. **Separated `uv` Cache & Read-On-PR / Save-On-Main XLA Compilation Cache (Debt #2606)**:
   - Sets `JAX_COMPILATION_CACHE_DIR: ${{ runner.temp }}/jax_compilation_cache` and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS: "0.5"`.
   - Caches `~/.cache/uv` under `uv-${{ runner.os }}-py3.13-${{ hashFiles('uv.lock') }}`.
   - On `pull_request`, uses `actions/cache/restore@v4` (read-only restore from `main`'s XLA cache) so PR commits do not upload a new multi-hundred-MB XLA archive on every push and evict `~/.cache/uv` from GitHub Actions' 10 GB repo cache budget. On `push` to `main`, uses `actions/cache@v4` keyed by `xla-${{ runner.os }}-py3.13-${{ hashFiles('uv.lock') }}-${{ github.sha }}` with `restore-keys: xla-${{ runner.os }}-py3.13-${{ hashFiles('uv.lock') }}-` to keep `main`'s XLA cache warm.
5. **Adaptive Worker Sizing (`--dist loadscope`) & `main` Safety Net (Debt #2585, #2582)**:
   - On `pull_request`: runs `scripts/ci/select_affected_tests.py --base "origin/${{ github.base_ref }}" --head HEAD --explain --write-pytest-args .pytest-targets.txt` and adapts `pytest-xdist` workers to prevent 4-worker JAX/PyTorch OOM kills on 16 GB runners: `-n 2 --dist loadscope` for small runs (`<= 25` files, avoiding 4-worker startup overhead) and large/full runs (`> 150` files or `mode == "full"`, capping peak RSS), and `-n auto --dist loadscope` for medium incremental runs (`26..150` files).
   - On `push` to `main`: runs the **entire** test suite with `-n 2 --dist loadscope --cov=src --cov-report=xml --cov-report=term` as the post-merge safety net.

## 6. Empirical Historical Replay & Control Verification

Replaying `scripts/ci/select_affected_tests.py --replay 200` over the last **200 first-parent commits** on `main`:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total repository test files** | **414** | `tests/**/test_*.py` (413 existing + `tests/ci/test_select_affected_tests.py`), 100% sound closures |
| **Commits analyzed (`--first-parent -n 200`)** | **200** | Merged PRs and direct commits on `main` |
| **Mode breakdown** | **114 `incremental` (57.0%)**, **67 `guards_only` (33.5%)**, **19 `full` (9.5%)** | Full runs triggered on `pyproject.toml`, `uv.lock`, `tests/conftest.py`, or `.github/workflows/ci.yml` edits |
| **Median selected test files (overall)** | **21 / 414 (5.1%)** | **19.7x reduction (94.9% fewer files)** in median test files executed per commit |
| **Mean selected test files (overall)** | **80.8 / 414 (19.5%)** | **5.1x reduction (80.5% fewer files)** in mean test files executed (including full-suite fallback runs) |
| **Median selected test files (`incremental` only)** | **24 / 414 (5.8%)** | Mean `incremental` run selects **61.1 / 414 (14.8%)** test files |
| **Per-file co-change recall (non-test diff -> co-changed `tests/` files)** | **257 / 259 (99.2%)** | Across 63 co-change commits (62/63 any-hit, 61/63 all-hit), the only 2 un-recalled files were unrelated test hygiene edits bundled with non-code or unrelated commits |
