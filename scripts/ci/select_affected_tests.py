#!/usr/bin/env python3
"""Incremental CI test selector based on target-aware static import closures.

Maps a git diff (`--base <ref> --head <ref>`) or explicit file list (`--paths ...`)
to the subset of pytest test files (`tests/**/test_*.py`, `tests/**/*_test.py`) whose
static dependency closure intersects the diff, plus the Always-Run Guard Set.

Design & Soundness Model (`.praxia/docs/specs/261026_incremental-ci-test-selection.md`):
- Generalizes `tests/knob_gate/_closure.py` and `tests/knob_gate/closure_edges.toml`
  from the 9 redsox parity vehicles to all repository test modules.
- Uses target-aware import expansion: importing `from aminx.potts.runner import X`
  records `src/aminx/__init__.py` and `src/aminx/potts/__init__.py` as file
  dependencies of the test (so editing an `__init__.py` invalidates all tests
  importing through that package), expands subpackage `__init__.py` files in BFS,
  and avoids expanding root `src/aminx/__init__.py` or `src/aminx/run/__init__.py`
  unless imported directly.
- Evaluates lexically scoped `Path(__file__)`-derived expressions passed to
  `spec_from_file_location` and `run_path`, literal `import_module` calls,
  `@pytest.mark.port_wave("<wave>")` markers, and dotted `"aminx.<pkg>..."` or
  inline `import aminx...` strings in non-guard test modules.
- Fails open (`mode = "full"`, running the entire pytest suite) whenever:
  1. `git diff` fails or refs are unavailable;
  2. Any closure in the repository has an unreviewed dynamic call site or syntax
     error (`not closure.sound`);
  3. Any changed file matches `[global].files` or `[global].prefixes` in
     `scripts/ci/test_closure_edges.toml`;
  4. Any changed file under `src/`, `tests/`, or `scripts/` was deleted or renamed
     away;
  5. Any changed file under `src/` matches zero non-guard test closures, or any
     code/fixture file under `tests/` matches zero test closures.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDGES_FILE = Path(__file__).with_name("test_closure_edges.toml")
_ROOTS = ("src", "tests", "scripts", "tools")
_DYNAMIC_CALLS = frozenset(
    {
        "import_module",
        "__import__",
        "spec_from_file_location",
        "run_path",
        "entry_points",
        "iter_modules",
        "files",  # importlib.resources.files
    }
)
# Root `aminx`, `aminx.run`, `aminx.model`, and `aminx.types` re-export cross-cutting
# symbols; their `__init__.py` files are recorded in `closure.files` on every import
# through them, and expanded in BFS only when imported directly (`import aminx`,
# `from aminx import ...`, `from aminx.run import ...`, `from aminx.model import ...`).
# All other subpackage `__init__.py` files are expanded in BFS.
_NON_EXPANDED_ANCESTOR_PKGS = frozenset(
    {"aminx", "aminx.io", "aminx.run", "aminx.model", "aminx.types"}
)
_NON_EXPANDED_ANCESTOR_FILES = frozenset(
    {
        "src/aminx/__init__.py",
        "src/aminx/io/__init__.py",
        "src/aminx/run/__init__.py",
        "src/aminx/model/__init__.py",
        "src/aminx/types/__init__.py",
    }
)
_PR_EXCLUDED_MARKERS = frozenset(
    {"slow", "parity_heavy", "parity_audit", "redsox_gate"}
)
_FAMILY_STRING_TO_MODULE: Mapping[str, str] = {
    "pottsmpnn": "aminx.families.potts_mpnn",
    "protonpottsmpnn": "aminx.families.protonpotts_mpnn",
    "lasermpnn": "aminx.families.laser_mpnn",
    "caliby": "aminx.families.caliby",
    "fampnn": "aminx.families.fampnn",
}
_BROAD_DIRS = frozenset(
    {
        "",
        ".",
        "src",
        "src/aminx",
        "tests",
        "tests/port",
        "tests/port/targets",
        "tests/port/manifests",
        "tests/port/reference",
        "scripts",
        "tools",
        ".praxia",
        ".praxia/docs",
        ".praxia/docs/specs",
        "agent_assets",
    }
)
_DOTTED_AMINX_RE = re.compile(r"^aminx(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_INLINE_AMINX_IMPORT_RE = re.compile(
    r"\b(?:from|import)\s+(aminx(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
)
_RESOLVE_CACHE: dict[tuple[Path, str, Path | None], str | None] = {}
_PR_EXCLUDED_TEST_CACHE: dict[tuple[Path, str], bool] = {}


@dataclass(frozen=True)
class EdgeConfig:
    """Parsed contents of `scripts/ci/test_closure_edges.toml`."""

    global_files: frozenset[str]
    global_prefixes: tuple[str, ...]
    always_run: tuple[str, ...]
    unreferenced_src_allowlist: frozenset[str]
    modules: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class TestClosure:
    """Static dependency closure of a single test file."""

    test_file: str
    files: frozenset[str]
    data_prefixes: tuple[str, ...] = ()
    undeclared_dynamic: tuple[str, ...] = field(default=())

    @property
    def sound(self) -> bool:
        return not self.undeclared_dynamic


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of mapping a set of changed paths to affected tests."""

    mode: str  # "full" | "incremental" | "guards_only"
    reason: str
    changed_paths: tuple[str, ...]
    selected_tests: tuple[str, ...]
    implicated_tests: tuple[str, ...]
    guard_tests: tuple[str, ...]
    per_path_hits: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _FileScan:
    """Cached single-file AST scan result."""

    target_modules: frozenset[str]
    ancestor_modules: frozenset[str]
    ancestor_files: frozenset[str]
    path_files: frozenset[str]
    data_prefixes: frozenset[str]
    dynamic_lines: tuple[int, ...]


def norm_path(raw: str) -> str:
    """Normalize a repo-relative path string."""
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/") if path != "/" else path


def _string_list(body: Mapping[str, object], key: str, context: str) -> list[str]:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{context}: {key!r} must be a list of strings, got {value!r}"
        raise ValueError(msg)
    return [norm_path(item) for item in value]


def load_edge_config(path: Path = _EDGES_FILE) -> EdgeConfig:
    """Load and validate `scripts/ci/test_closure_edges.toml`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    global_tbl = raw.get("global", {})
    guards_tbl = raw.get("guards", {})
    unref_tbl = raw.get("unreferenced_src", {})
    mod_tbl = raw.get("module", {})

    global_files = frozenset(_string_list(global_tbl, "files", "[global]"))
    global_prefixes = tuple(
        p.rstrip("/") + "/" for p in _string_list(global_tbl, "prefixes", "[global]")
    )
    always_run = tuple(_string_list(guards_tbl, "always_run", "[guards]"))
    unreferenced_src = frozenset(
        _string_list(unref_tbl, "allowlist", "[unreferenced_src]")
    )
    modules: dict[str, Mapping[str, object]] = {}
    if isinstance(mod_tbl, dict):
        for k, v in mod_tbl.items():
            if isinstance(v, dict):
                modules[norm_path(str(k))] = v

    return EdgeConfig(
        global_files=global_files,
        global_prefixes=global_prefixes,
        always_run=always_run,
        unreferenced_src_allowlist=unreferenced_src,
        modules=modules,
    )


def is_test_file(path: str) -> bool:
    """Return True if `path` is a pytest test module under `tests/`."""
    normed = norm_path(path)
    if not normed.startswith("tests/") or not normed.endswith(".py"):
        return False
    name = normed.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def discover_test_files(repo: Path = _REPO_ROOT) -> list[str]:
    """Return all pytest test files under `tests/` in sorted order."""
    tests_root = repo / "tests"
    if not tests_root.is_dir():
        return []
    found: list[str] = []
    for p in tests_root.rglob("*.py"):
        rel = p.relative_to(repo).as_posix()
        if is_test_file(rel):
            found.append(rel)
    return sorted(found)


def resolve_module_file(
    repo: Path,
    module: str,
    *,
    importer_dir: Path | None = None,
) -> str | None:
    """Resolve a dotted module name to a repo-relative `.py` file if it exists."""
    if not module:
        return None
    cache_key = (repo, module, importer_dir)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]

    parts = [p for p in module.split(".") if p]
    if not parts or parts[0] in sys.stdlib_module_names:
        _RESOLVE_CACHE[cache_key] = None
        return None

    if importer_dir is None:
        search_bases: list[Path] = [repo / "src", repo]
    else:
        search_bases = [repo / root for root in _ROOTS]
        search_bases.append(repo)
        search_bases.append(importer_dir)

    result: str | None = None
    for base in search_bases:
        as_file = base.joinpath(*parts).with_suffix(".py")
        if as_file.is_file():
            try:
                result = as_file.relative_to(repo).as_posix()
                break
            except ValueError:
                pass
        as_pkg = base.joinpath(*parts, "__init__.py")
        if as_pkg.is_file():
            try:
                result = as_pkg.relative_to(repo).as_posix()
                break
            except ValueError:
                pass

    _RESOLVE_CACHE[cache_key] = result
    return result


def _resolve_longest_aminx_module(repo: Path, dotted: str) -> tuple[str | None, list[str]]:
    """Resolve the longest module prefix of `aminx.foo.bar.attr` under `src/`."""
    parts = dotted.split(".")
    for end in range(len(parts), 1, -1):
        cand = ".".join(parts[:end])
        resolved = resolve_module_file(repo, cand)
        if resolved is not None:
            ancestors = [".".join(parts[:i]) for i in range(1, end)]
            return cand, ancestors
    return None, []


def _ancestors(module: str) -> list[str]:
    parts = [p for p in module.split(".") if p]
    return [".".join(parts[:end]) for end in range(1, len(parts) + 1)]


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _eval_path_expr(
    node: ast.AST | None,
    *,
    repo_resolved: Path,
    file_resolved: Path,
    env: Mapping[str, Path],
) -> Path | None:
    """Statically evaluate `Path(__file__)` and constant path expressions within a lexical scope."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return file_resolved
        return env.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        val = node.value
        if len(val) < 200 and " " not in val and "\n" not in val:
            cand = (repo_resolved / val.strip("/")).resolve()
            if cand.exists() and cand.is_relative_to(repo_resolved):
                return cand
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_path_expr(node.left, repo_resolved=repo_resolved, file_resolved=file_resolved, env=env)
        if (
            left is not None
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
            and len(node.right.value) < 200
        ):
            cand = (left / node.right.value).resolve()
            if cand.is_relative_to(repo_resolved):
                return cand
        return None
    if isinstance(node, ast.Attribute):
        base = _eval_path_expr(node.value, repo_resolved=repo_resolved, file_resolved=file_resolved, env=env)
        if base is not None and node.attr == "parent":
            return base.parent
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            base = _eval_path_expr(
                node.value.value,
                repo_resolved=repo_resolved,
                file_resolved=file_resolved,
                env=env,
            )
            if base is not None and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                idx = node.slice.value
                if 0 <= idx < len(base.parents):
                    return base.parents[idx]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"Path", "str"} and node.args:
            return _eval_path_expr(
                node.args[0],
                repo_resolved=repo_resolved,
                file_resolved=file_resolved,
                env=env,
            )
        if isinstance(node.func, ast.Attribute):
            base = _eval_path_expr(
                node.func.value,
                repo_resolved=repo_resolved,
                file_resolved=file_resolved,
                env=env,
            )
            if base is not None:
                if node.func.attr == "resolve" and not node.args:
                    return base.resolve()
                if (
                    node.func.attr == "with_name"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and len(node.args[0].value) < 200
                ):
                    cand = base.with_name(node.args[0].value).resolve()
                    if cand.is_relative_to(repo_resolved):
                        return cand
                if (
                    node.func.attr == "joinpath"
                    and node.args
                    and all(
                        isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) < 200
                        for a in node.args
                    )
                ):
                    cand = base.joinpath(*(str(a.value) for a in node.args)).resolve()
                    if cand.is_relative_to(repo_resolved):
                        return cand
    return None


def _extract_div_chain_suffix(node: ast.BinOp) -> str | None:
    """Extract trailing string segments from `expr / 'a' / 'b'`."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        if isinstance(cur.right, ast.Constant) and isinstance(cur.right.value, str):
            parts.append(cur.right.value.strip("/"))
            cur = cur.left
        else:
            break
    if isinstance(cur, ast.Constant) and isinstance(cur.value, str):
        parts.append(cur.value.strip("/"))
    parts = [p for p in reversed(parts) if p]
    if not parts:
        return None
    return "/".join(parts)


def _register_candidate_repo_path(
    repo: Path,
    candidate: str,
    path_files: set[str],
    data_prefixes: set[str],
) -> None:
    """If `candidate` resolves to an existing repo file or narrow directory, record it."""
    cand = norm_path(candidate)
    if not cand or cand in _BROAD_DIRS or len(cand) >= 200 or " " in cand or "\n" in cand:
        return
    full = repo / cand
    if full.is_file():
        path_files.add(full.relative_to(repo).as_posix())
        return
    as_py = repo / (cand + ".py")
    if as_py.is_file():
        path_files.add(as_py.relative_to(repo).as_posix())
        return
    if full.is_dir() and cand not in _BROAD_DIRS:
        data_prefixes.add(cand.rstrip("/") + "/")


def _resolve_port_wave(
    repo: Path,
    wave_name: str,
    target_modules: set[str],
    ancestor_modules: set[str],
    path_files: set[str],
    data_prefixes: set[str],
) -> None:
    """Resolve `@pytest.mark.port_wave('<wave>')` to target TOML, manifest, mutants, symbol, and oracle."""
    target_rel = f"tests/port/targets/{wave_name}.toml"
    target_path = repo / target_rel
    if not target_path.is_file():
        return
    path_files.add(target_rel)

    mutant_rel = f"tests/port/mutants/{wave_name}.py"
    if (repo / mutant_rel).is_file():
        path_files.add(mutant_rel)

    try:
        target_doc = tomllib.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        return

    port_tbl = target_doc.get("port", target_doc)
    if not isinstance(port_tbl, dict):
        port_tbl = target_doc

    wave_id = port_tbl.get("wave_id") or target_doc.get("wave_id")
    if isinstance(wave_id, str) and wave_id:
        manifest_rel = f"tests/port/manifests/{wave_id}.toml"
        if (repo / manifest_rel).is_file():
            path_files.add(manifest_rel)

    ref_sub = port_tbl.get("reference_subtree")
    if isinstance(ref_sub, str) and ref_sub:
        ref_dir = repo / "tests" / "port" / ref_sub.strip("/")
        if ref_dir.is_dir():
            data_prefixes.add(ref_dir.relative_to(repo).as_posix().rstrip("/") + "/")
            for py_f in ref_dir.rglob("*.py"):
                path_files.add(py_f.relative_to(repo).as_posix())

    symbol_qualname = port_tbl.get("symbol_qualname")
    if isinstance(symbol_qualname, str) and symbol_qualname.startswith("aminx."):
        mod_target, mod_ancs = _resolve_longest_aminx_module(repo, symbol_qualname)
        for anc in mod_ancs:
            ancestor_modules.add(anc)
            if anc not in _NON_EXPANDED_ANCESTOR_PKGS:
                target_modules.add(anc)
        if mod_target is not None and mod_target != "aminx":
            target_modules.add(mod_target)

    oracle = target_doc.get("oracle", {})
    if isinstance(oracle, dict):
        ref_import = oracle.get("reference_import")
        if isinstance(ref_import, str) and ref_import:
            parts = ref_import.split(".")
            algo_path = repo.joinpath("tests", "port", *parts, "algo.py")
            if algo_path.is_file():
                path_files.add(algo_path.relative_to(repo).as_posix())
            ref_dir = repo.joinpath("tests", "port", *parts)
            if ref_dir.is_dir():
                data_prefixes.add(ref_dir.relative_to(repo).as_posix().rstrip("/") + "/")


def _is_type_checking_guard(test: ast.AST) -> bool:
    """Return True if `test` is `TYPE_CHECKING` or `typing.TYPE_CHECKING`."""
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _iter_runtime_ast(root: ast.AST) -> Iterable[ast.AST]:
    """Walk `root` skipping the dead `body` of `if TYPE_CHECKING:` blocks."""
    stack: list[ast.AST] = [root]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, ast.If) and _is_type_checking_guard(cur.test):
            for child in reversed(cur.orelse):
                stack.append(child)
            continue
        for child in reversed(list(ast.iter_child_nodes(cur))):
            stack.append(child)


def _iter_scope_nodes(body: Sequence[ast.stmt]) -> Iterable[ast.AST]:
    """Yield runtime AST nodes within `body` without descending into nested function/class definitions."""
    stack: list[ast.AST] = list(reversed(body))
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, ast.If) and _is_type_checking_guard(cur.test):
            for child in reversed(cur.orelse):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    stack.append(child)
            continue
        for child in reversed(list(ast.iter_child_nodes(cur))):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child)


def _extract_assign_name_and_value(node: ast.AST) -> tuple[str, ast.AST | None] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return node.target.id, node.value
    return None


def _build_scope_path_envs(
    tree: ast.Module,
    *,
    repo: Path,
    repo_resolved: Path,
    file_resolved: Path,
    register_paths: bool,
    path_files: set[str],
    data_prefixes: set[str],
) -> dict[ast.AST, Mapping[str, Path]]:
    """Build lexically scoped `Path` constant environments per module/function without cross-function shadowing."""
    module_env: dict[str, Path] = {}
    module_tainted: set[str] = set()

    for node in _iter_scope_nodes(tree.body):
        assign = _extract_assign_name_and_value(node)
        if assign is None:
            continue
        var_name, val_node = assign
        if var_name in module_tainted:
            continue
        if var_name in module_env:
            # Reassigned at module level -> taint so we do not guess which value reaches a call
            del module_env[var_name]
            module_tainted.add(var_name)
            continue
        val_path = _eval_path_expr(
            val_node,
            repo_resolved=repo_resolved,
            file_resolved=file_resolved,
            env=module_env,
        )
        if val_path is not None and val_path.is_relative_to(repo_resolved):
            module_env[var_name] = val_path
            if register_paths and val_path.exists():
                _register_candidate_repo_path(
                    repo,
                    val_path.relative_to(repo_resolved).as_posix(),
                    path_files,
                    data_prefixes,
                )
        else:
            module_tainted.add(var_name)

    node_env: dict[ast.AST, Mapping[str, Path]] = {}
    for node in _iter_scope_nodes(tree.body):
        node_env[node] = module_env

    for node in _iter_runtime_ast(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_env = dict(module_env)
            func_tainted: set[str] = set()
            # Function parameters shadow module bindings and are dynamic
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                func_env.pop(arg.arg, None)
                func_tainted.add(arg.arg)
            if node.args.vararg is not None:
                func_env.pop(node.args.vararg.arg, None)
                func_tainted.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                func_env.pop(node.args.kwarg.arg, None)
                func_tainted.add(node.args.kwarg.arg)

            local_assigned: set[str] = set()
            for sub in _iter_scope_nodes(node.body):
                assign = _extract_assign_name_and_value(sub)
                if assign is None:
                    continue
                var_name, val_node = assign
                if var_name in func_tainted:
                    continue
                if var_name in local_assigned:
                    func_env.pop(var_name, None)
                    func_tainted.add(var_name)
                    continue
                local_assigned.add(var_name)
                val_path = _eval_path_expr(
                    val_node,
                    repo_resolved=repo_resolved,
                    file_resolved=file_resolved,
                    env=func_env,
                )
                if val_path is not None and val_path.is_relative_to(repo_resolved):
                    func_env[var_name] = val_path
                    if register_paths and val_path.exists():
                        _register_candidate_repo_path(
                            repo,
                            val_path.relative_to(repo_resolved).as_posix(),
                            path_files,
                            data_prefixes,
                        )
                else:
                    func_env.pop(var_name, None)
                    func_tainted.add(var_name)

            for sub in _iter_scope_nodes(node.body):
                node_env[sub] = func_env

    return node_env


def _resolve_relative_import(
    node: ast.ImportFrom,
    *,
    file_resolved: Path,
    repo_resolved: Path,
    ancestor_files: set[str],
    path_files: set[str],
) -> None:
    """Resolve `from . import x` / `from ..sub import y` directly on the filesystem."""
    # level == 1 -> file_resolved.parent; level == 2 -> file_resolved.parent.parent
    up = node.level - 1
    parents = file_resolved.parents
    if up < 0 or up >= len(parents):
        return
    rel_base = parents[up]
    if not rel_base.is_relative_to(repo_resolved):
        return

    # Record intermediate package `__init__.py` files climbed by relative import
    # in `ancestor_files` so hub `__init__.py` files (`src/aminx/__init__.py`,
    # `src/aminx/run/__init__.py`, etc.) are tracked in `closure.files` without
    # unconditionally expanding them in BFS.
    for idx in range(0, up + 1):
        pkg_init = parents[idx] / "__init__.py"
        if pkg_init.is_file() and pkg_init.is_relative_to(repo_resolved):
            ancestor_files.add(pkg_init.relative_to(repo_resolved).as_posix())

    mod_parts = [p for p in (node.module or "").split(".") if p]
    for end in range(1, len(mod_parts) + 1):
        sub_init = rel_base.joinpath(*mod_parts[:end], "__init__.py")
        if sub_init.is_file() and sub_init.is_relative_to(repo_resolved):
            ancestor_files.add(sub_init.relative_to(repo_resolved).as_posix())

    target_dir = rel_base.joinpath(*mod_parts) if mod_parts else rel_base
    if mod_parts:
        mod_file = target_dir.with_suffix(".py")
        if mod_file.is_file() and mod_file.is_relative_to(repo_resolved):
            path_files.add(mod_file.relative_to(repo_resolved).as_posix())

    target_init = target_dir / "__init__.py"
    for alias in node.names:
        if alias.name == "*":
            if target_init.is_file() and target_init.is_relative_to(repo_resolved):
                path_files.add(target_init.relative_to(repo_resolved).as_posix())
            continue
        cand_file = target_dir.joinpath(alias.name).with_suffix(".py")
        cand_pkg = target_dir.joinpath(alias.name, "__init__.py")
        matched_sub = False
        if cand_file.is_file() and cand_file.is_relative_to(repo_resolved):
            path_files.add(cand_file.relative_to(repo_resolved).as_posix())
            matched_sub = True
        if cand_pkg.is_file() and cand_pkg.is_relative_to(repo_resolved):
            path_files.add(cand_pkg.relative_to(repo_resolved).as_posix())
            matched_sub = True
        if not matched_sub and target_init.is_file() and target_init.is_relative_to(repo_resolved):
            t_rel = target_init.relative_to(repo_resolved).as_posix()
            if not mod_parts or t_rel not in _NON_EXPANDED_ANCESTOR_FILES:
                path_files.add(t_rel)


def _scan_python_file(
    repo: Path,
    path: str,
    *,
    scan_string_literals: bool,
) -> _FileScan:
    """Parse `path` and extract target imports, ancestor packages, paths, and dynamic calls."""
    full_path = repo / path
    repo_resolved = repo.resolve()
    file_resolved = full_path.resolve()
    try:
        source = full_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        # Unparseable Python file: mark line 1 as an unreviewed dynamic site so
        # any closure reaching it becomes unsound and fails open to `mode = "full"`.
        return _FileScan(
            target_modules=frozenset(),
            ancestor_modules=frozenset(),
            ancestor_files=frozenset(),
            path_files=frozenset(),
            data_prefixes=frozenset(),
            dynamic_lines=(1,),
        )

    # Python 3 has no implicit relative imports inside packages (`src/aminx/`);
    # only `tests/` and `scripts/` put their own directory on sys.path.
    importer_dir = None if path.startswith("src/") else full_path.parent

    target_modules: set[str] = set()
    ancestor_modules: set[str] = set()
    ancestor_files: set[str] = set()
    path_files: set[str] = set()
    data_prefixes: set[str] = set()
    dynamic_lines: list[int] = []

    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in _iter_runtime_ast(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    node_envs = _build_scope_path_envs(
        tree,
        repo=repo,
        repo_resolved=repo_resolved,
        file_resolved=file_resolved,
        register_paths=scan_string_literals,
        path_files=path_files,
        data_prefixes=data_prefixes,
    )

    def add_target_module(mod_name: str) -> None:
        if not mod_name:
            return
        ancs = _ancestors(mod_name)
        for anc in ancs[:-1]:
            if resolve_module_file(repo, anc, importer_dir=importer_dir) is not None:
                ancestor_modules.add(anc)
                if anc not in _NON_EXPANDED_ANCESTOR_PKGS:
                    target_modules.add(anc)
        if resolve_module_file(repo, mod_name, importer_dir=importer_dir) is not None:
            target_modules.add(mod_name)

    for node in _iter_runtime_ast(tree):
        scope_env = node_envs.get(node, {})
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_target_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                _resolve_relative_import(
                    node,
                    file_resolved=file_resolved,
                    repo_resolved=repo_resolved,
                    ancestor_files=ancestor_files,
                    path_files=path_files,
                )
            else:
                base = node.module or ""
                if base:
                    for anc in _ancestors(base):
                        if resolve_module_file(repo, anc, importer_dir=importer_dir) is not None:
                            ancestor_modules.add(anc)
                            if anc not in _NON_EXPANDED_ANCESTOR_PKGS:
                                target_modules.add(anc)
                    # If `base` is not in `_NON_EXPANDED_ANCESTOR_PKGS`, importing
                    # `from base import X` also expands `base/__init__.py` in case `base`
                    # defines or re-exports `X` alongside a submodule of the same name.
                    if (
                        base not in _NON_EXPANDED_ANCESTOR_PKGS
                        and resolve_module_file(repo, base, importer_dir=importer_dir) is not None
                    ):
                        target_modules.add(base)
                    for alias in node.names:
                        cand = f"{base}.{alias.name}"
                        if resolve_module_file(repo, cand, importer_dir=importer_dir) is not None:
                            target_modules.add(cand)
                        elif resolve_module_file(repo, base, importer_dir=importer_dir) is not None:
                            target_modules.add(base)
        elif scan_string_literals and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parent = parent_map.get(node)
            if not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)):
                eval_p = _eval_path_expr(
                    node,
                    repo_resolved=repo_resolved,
                    file_resolved=file_resolved,
                    env=scope_env,
                )
                if eval_p is not None and eval_p.exists() and eval_p.is_relative_to(repo_resolved):
                    _register_candidate_repo_path(
                        repo,
                        eval_p.relative_to(repo_resolved).as_posix(),
                        path_files,
                        data_prefixes,
                    )
                else:
                    suffix = _extract_div_chain_suffix(node)
                    if suffix is not None:
                        _register_candidate_repo_path(repo, suffix, path_files, data_prefixes)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "port_wave" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    _resolve_port_wave(
                        repo,
                        arg0.value,
                        target_modules,
                        ancestor_modules,
                        path_files,
                        data_prefixes,
                    )
            elif name in _DYNAMIC_CALLS:
                literal_str = (
                    str(node.args[0].value)
                    if (
                        bool(node.args)
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    )
                    else None
                )
                if name in {"import_module", "__import__"} and literal_str is not None:
                    add_target_module(literal_str)
                elif name in {"spec_from_file_location", "run_path"}:
                    arg_idx = 1 if name == "spec_from_file_location" else 0
                    arg_node = node.args[arg_idx] if len(node.args) > arg_idx else None
                    resolved_p = _eval_path_expr(
                        arg_node,
                        repo_resolved=repo_resolved,
                        file_resolved=file_resolved,
                        env=scope_env,
                    )
                    if (
                        resolved_p is not None
                        and resolved_p.is_file()
                        and resolved_p.is_relative_to(repo_resolved)
                    ):
                        path_files.add(resolved_p.relative_to(repo_resolved).as_posix())
                    else:
                        dynamic_lines.append(node.lineno)
                else:
                    dynamic_lines.append(node.lineno)
        elif scan_string_literals and isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            val_stripped = val.strip()
            if (
                "\n" not in val
                and " " not in val
                and len(val) < 200
                and (
                    val.startswith(
                        (
                            "scripts/",
                            "tests/",
                            "src/",
                            ".praxia/",
                            "agent_assets/",
                            "tools/",
                        )
                    )
                    or val == "NOTICE-weights.md"
                )
            ):
                _register_candidate_repo_path(repo, val, path_files, data_prefixes)
            elif _DOTTED_AMINX_RE.match(val):
                mod_target, mod_ancs = _resolve_longest_aminx_module(repo, val)
                for anc in mod_ancs:
                    ancestor_modules.add(anc)
                    if anc not in _NON_EXPANDED_ANCESTOR_PKGS:
                        target_modules.add(anc)
                if mod_target is not None and mod_target != "aminx":
                    target_modules.add(mod_target)
            elif ("import " in val or "from " in val) and "aminx" in val:
                for match in _INLINE_AMINX_IMPORT_RE.finditer(val):
                    mod_target, mod_ancs = _resolve_longest_aminx_module(repo, match.group(1))
                    for anc in mod_ancs:
                        ancestor_modules.add(anc)
                        if anc not in _NON_EXPANDED_ANCESTOR_PKGS:
                            target_modules.add(anc)
                    if mod_target is not None and mod_target != "aminx":
                        target_modules.add(mod_target)
            elif val_stripped in _FAMILY_STRING_TO_MODULE:
                add_target_module(_FAMILY_STRING_TO_MODULE[val_stripped])
            else:
                for fam_key, fam_mod in _FAMILY_STRING_TO_MODULE.items():
                    if val_stripped.startswith(f"{fam_key}_"):
                        add_target_module(fam_mod)
                        break

    return _FileScan(
        target_modules=frozenset(target_modules),
        ancestor_modules=frozenset(ancestor_modules),
        ancestor_files=frozenset(ancestor_files),
        path_files=frozenset(path_files),
        data_prefixes=frozenset(data_prefixes),
        dynamic_lines=tuple(sorted(set(dynamic_lines))),
    )


def _scoped_conftests_for(repo: Path, test_file: str) -> list[str]:
    """Return subdirectory `conftest.py` and `__init__.py` files applicable to `test_file` (excluding root)."""
    parts = norm_path(test_file).split("/")
    scoped: list[str] = []
    for depth in range(2, len(parts)):
        for special in ("conftest.py", "__init__.py"):
            rel = "/".join([*parts[:depth], special])
            if (repo / rel).is_file():
                scoped.append(rel)
    return scoped


def _expected_dynamic_count(mod_decl: Mapping[str, object], context: str) -> int | None:
    """Return expected dynamic site count (`int`), or `None` for legacy boolean `dynamic_ok = true`."""
    if "dynamic_count" in mod_decl:
        val = mod_decl["dynamic_count"]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            msg = f"{context}: 'dynamic_count' must be a non-negative integer, got {val!r}"
            raise ValueError(msg)
        return val
    if mod_decl.get("dynamic_ok") is True:
        return None
    return 0


def is_pr_excluded_test_file(repo: Path, test_file: str) -> bool:
    """Return True if 100% of test items in `test_file` carry PR-excluded markers (`slow`, `parity_heavy`, `parity_audit`, `redsox_gate`)."""
    cache_key = (repo, test_file)
    if cache_key in _PR_EXCLUDED_TEST_CACHE:
        return _PR_EXCLUDED_TEST_CACHE[cache_key]

    full_path = repo / test_file
    try:
        tree = ast.parse(full_path.read_text(encoding="utf-8"), filename=test_file)
    except Exception:
        _PR_EXCLUDED_TEST_CACHE[cache_key] = False
        return False

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    text = ast.unparse(node.value)
                    if any(f"mark.{m}" in text for m in _PR_EXCLUDED_MARKERS):
                        _PR_EXCLUDED_TEST_CACHE[cache_key] = True
                        return True

    test_fns: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, tuple[ast.expr, ...]]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            test_fns.append((node, tuple(node.decorator_list)))
        elif isinstance(node, ast.ClassDef):
            cls_decs = tuple(node.decorator_list)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                    test_fns.append((sub, (*cls_decs, *sub.decorator_list)))

    if not test_fns:
        _PR_EXCLUDED_TEST_CACHE[cache_key] = False
        return False

    all_excluded = True
    for _fn, decs in test_fns:
        dec_texts = [ast.unparse(d) for d in decs]
        if not any(f"mark.{m}" in dt for dt in dec_texts for m in _PR_EXCLUDED_MARKERS):
            all_excluded = False
            break

    _PR_EXCLUDED_TEST_CACHE[cache_key] = all_excluded
    return all_excluded


def compute_all_test_closures(
    repo: Path = _REPO_ROOT,
    config: EdgeConfig | None = None,
) -> dict[str, TestClosure]:
    """Compute the static dependency closure for every test file in `repo`."""
    if repo != _REPO_ROOT:
        _RESOLVE_CACHE.clear()
        _PR_EXCLUDED_TEST_CACHE.clear()
    config = load_edge_config(repo / "scripts/ci/test_closure_edges.toml") if config is None else config
    always_run_set = set(config.always_run)
    scan_cache: dict[str, _FileScan] = {}

    def get_scan(rel_path: str) -> _FileScan:
        if rel_path not in scan_cache:
            # Extract string-literal repo paths and dotted `aminx.*` strings only
            # from non-guard test modules under `tests/` (never from `always_run`
            # guards or `src/`/`scripts/` docstrings).
            scan_literals = rel_path.startswith("tests/") and rel_path not in always_run_set
            scan_cache[rel_path] = _scan_python_file(
                repo,
                rel_path,
                scan_string_literals=scan_literals,
            )
        return scan_cache[rel_path]

    closures: dict[str, TestClosure] = {}
    for test_file in discover_test_files(repo):
        visited_py: set[str] = set()
        files: set[str] = set()
        data_prefixes: set[str] = set()
        undeclared: list[str] = []

        queue: list[str] = [test_file, *_scoped_conftests_for(repo, test_file)]

        while queue:
            cur = queue.pop()
            if cur in visited_py:
                continue
            if not (repo / cur).is_file():
                continue
            visited_py.add(cur)
            files.add(cur)
            importer_dir = None if cur.startswith("src/") else (repo / cur).parent

            mod_decl = config.modules.get(cur, {})
            expected_dyn = _expected_dynamic_count(mod_decl, f"[module.{cur!r}]")
            for soft_mod in _string_list(mod_decl, "soft", f"[module.{cur!r}]"):
                for anc in _ancestors(soft_mod)[:-1]:
                    anc_file = resolve_module_file(repo, anc, importer_dir=importer_dir)
                    if anc_file is not None:
                        files.add(anc_file)
                        if anc not in _NON_EXPANDED_ANCESTOR_PKGS and anc_file not in visited_py:
                            queue.append(anc_file)
                soft_file = resolve_module_file(repo, soft_mod, importer_dir=importer_dir)
                if soft_file is None:
                    msg = f"test_closure_edges.toml: soft edge {soft_mod!r} on {cur!r} resolves to no file"
                    raise ValueError(msg)
                if soft_file not in visited_py:
                    queue.append(soft_file)

            for decl_file in _string_list(mod_decl, "files", f"[module.{cur!r}]"):
                files.add(decl_file)
                if decl_file.endswith(".py") and (repo / decl_file).is_file() and decl_file not in visited_py:
                    queue.append(decl_file)

            for decl_prefix in _string_list(mod_decl, "data_prefixes", f"[module.{cur!r}]"):
                data_prefixes.add(decl_prefix.rstrip("/") + "/")

            if not cur.endswith(".py"):
                continue

            scan = get_scan(cur)
            if expected_dyn is not None and len(scan.dynamic_lines) != expected_dyn:
                if scan.dynamic_lines:
                    undeclared.extend(f"{cur}:{line}" for line in scan.dynamic_lines)
                else:
                    undeclared.append(f"{cur}:stale_dynamic_count({expected_dyn})")

            for anc_mod in scan.ancestor_modules:
                anc_file = resolve_module_file(repo, anc_mod, importer_dir=importer_dir)
                if anc_file is not None:
                    files.add(anc_file)

            for anc_file in scan.ancestor_files:
                files.add(anc_file)
                if (
                    anc_file.endswith(".py")
                    and anc_file not in _NON_EXPANDED_ANCESTOR_FILES
                    and (repo / anc_file).is_file()
                    and anc_file not in visited_py
                ):
                    queue.append(anc_file)

            for tgt_mod in scan.target_modules:
                tgt_file = resolve_module_file(repo, tgt_mod, importer_dir=importer_dir)
                if tgt_file is not None and tgt_file not in visited_py:
                    queue.append(tgt_file)

            for pf in scan.path_files:
                files.add(pf)
                if pf.endswith(".py") and (repo / pf).is_file() and pf not in visited_py:
                    queue.append(pf)

            for dp in scan.data_prefixes:
                data_prefixes.add(dp)

        closures[test_file] = TestClosure(
            test_file=test_file,
            files=frozenset(files),
            data_prefixes=tuple(sorted(data_prefixes)),
            undeclared_dynamic=tuple(sorted(set(undeclared))),
        )

    return closures


def select_tests(
    changed_paths: Iterable[str],
    *,
    repo: Path = _REPO_ROOT,
    config: EdgeConfig | None = None,
    closures: Mapping[str, TestClosure] | None = None,
) -> SelectionResult:
    """Select affected test files for `changed_paths` with fail-open soundness."""
    config = load_edge_config(repo / "scripts/ci/test_closure_edges.toml") if config is None else config
    closures = compute_all_test_closures(repo, config) if closures is None else closures

    all_tests = tuple(sorted(closures))
    guard_tests = tuple(sorted(t for t in config.always_run if (repo / t).is_file()))
    guard_set = frozenset(guard_tests)
    normed_paths = tuple(dict.fromkeys(norm_path(p) for p in changed_paths if norm_path(p)))

    # 1. Fail open if any test closure in the repository is unsound.
    unsound = [c.test_file for c in closures.values() if not c.sound]
    if unsound:
        return SelectionResult(
            mode="full",
            reason=f"unsound test closures ({len(unsound)}): {', '.join(unsound[:5])}",
            changed_paths=normed_paths,
            selected_tests=all_tests,
            implicated_tests=all_tests,
            guard_tests=guard_tests,
            per_path_hits={},
        )

    # 2. If no paths changed, run the guard set.
    if not normed_paths:
        return SelectionResult(
            mode="guards_only",
            reason="empty diff; running always-run guard set",
            changed_paths=normed_paths,
            selected_tests=guard_tests,
            implicated_tests=(),
            guard_tests=guard_tests,
            per_path_hits={},
        )

    implicated: set[str] = set()
    per_path_hits: dict[str, tuple[str, ...]] = {}

    for path in normed_paths:
        # Global exact file or prefix trigger -> full suite
        if path in config.global_files or path.startswith(config.global_prefixes):
            return SelectionResult(
                mode="full",
                reason=f"global fallback trigger modified: {path}",
                changed_paths=normed_paths,
                selected_tests=all_tests,
                implicated_tests=all_tests,
                guard_tests=guard_tests,
                per_path_hits=per_path_hits,
            )

        # Deleted or renamed-away file under src/, tests/, scripts/, or tools/ -> full suite
        if path.startswith(("src/", "tests/", "scripts/", "tools/")) and not (repo / path).exists():
            return SelectionResult(
                mode="full",
                reason=f"deleted or renamed code/test file: {path}",
                changed_paths=normed_paths,
                selected_tests=all_tests,
                implicated_tests=all_tests,
                guard_tests=guard_tests,
                per_path_hits=per_path_hits,
            )

        # Subdirectory conftest.py -> selects all tests under that directory
        if path.startswith("tests/") and path.endswith("/conftest.py"):
            sub_prefix = path[: -len("conftest.py")]
            sub_hits = sorted(t for t in all_tests if t.startswith(sub_prefix))
            if not sub_hits:
                return SelectionResult(
                    mode="full",
                    reason=f"conftest modified with no matching subtree tests: {path}",
                    changed_paths=normed_paths,
                    selected_tests=all_tests,
                    implicated_tests=all_tests,
                    guard_tests=guard_tests,
                    per_path_hits=per_path_hits,
                )
            implicated.update(sub_hits)
            per_path_hits[path] = tuple(sub_hits)
            continue

        # Match against all test closures.
        # `.py` modules under `src/` or `tests/` must match `closure.files`
        # directly (not a directory `data_prefix`), so an unreferenced `.py`
        # module under `src/` or `tests/` cannot be masked by a directory prefix.
        is_src_or_test_py = path.startswith(("src/", "tests/")) and path.endswith(".py")
        hits: list[str] = []
        if is_test_file(path) and path in closures:
            hits.append(path)
        for test_file, closure in closures.items():
            if test_file == path:
                continue
            if path in closure.files or (
                not is_src_or_test_py
                and closure.data_prefixes
                and path.startswith(closure.data_prefixes)
            ):
                hits.append(test_file)

        hits = sorted(set(hits))
        per_path_hits[path] = tuple(hits)

        # Fail-open check for `src/**/*.py`: must match at least one PR-runnable
        # non-guard test closure (an always-run AST/lint guard or a 100%
        # marker-excluded test file alone will not execute unit tests on PRs).
        if path.startswith("src/") and path.endswith(".py"):
            pr_runnable_hits = [
                h
                for h in hits
                if h not in guard_set and not is_pr_excluded_test_file(repo, h)
            ]
            if not pr_runnable_hits:
                return SelectionResult(
                    mode="full",
                    reason=f"fail-open: {path} is not covered by any PR-runnable non-guard test closure",
                    changed_paths=normed_paths,
                    selected_tests=all_tests,
                    implicated_tests=all_tests,
                    guard_tests=guard_tests,
                    per_path_hits=per_path_hits,
                )

        if hits:
            implicated.update(hits)
            continue

        # Fail-open check: any file under `src/` or non-doc file under `tests/`
        # that matched zero test closures must trigger the full suite.
        if path.startswith("src/") or (path.startswith("tests/") and not path.endswith(".md")):
            return SelectionResult(
                mode="full",
                reason=f"fail-open: {path} is not covered by any static test closure",
                changed_paths=normed_paths,
                selected_tests=all_tests,
                implicated_tests=all_tests,
                guard_tests=guard_tests,
                per_path_hits=per_path_hits,
            )

    if implicated:
        selected = tuple(sorted(set(guard_tests) | implicated))
        return SelectionResult(
            mode="incremental",
            reason=(
                f"selected {len(implicated)} implicated test(s) + "
                f"{len(guard_tests)} always-run guard(s) = {len(selected)}/{len(all_tests)} test files"
            ),
            changed_paths=normed_paths,
            selected_tests=selected,
            implicated_tests=tuple(sorted(implicated)),
            guard_tests=guard_tests,
            per_path_hits=per_path_hits,
        )

    return SelectionResult(
        mode="guards_only",
        reason=(
            f"diff touched only non-code/doc paths ({len(normed_paths)} file(s)); "
            f"running {len(guard_tests)} always-run guard tests"
        ),
        changed_paths=normed_paths,
        selected_tests=guard_tests,
        implicated_tests=(),
        guard_tests=guard_tests,
        per_path_hits=per_path_hits,
    )


def git_changed_paths(
    base: str,
    head: str = "HEAD",
    *,
    repo: Path = _REPO_ROOT,
) -> list[str] | None:
    """Return changed file paths in `base...head`, or `None` if git fails."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...{head}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return [norm_path(line) for line in proc.stdout.splitlines() if line.strip()]


def select_from_git_diff(
    base: str,
    head: str = "HEAD",
    *,
    repo: Path = _REPO_ROOT,
    config: EdgeConfig | None = None,
    closures: Mapping[str, TestClosure] | None = None,
) -> SelectionResult:
    """Run `git diff --name-only base...head` and select affected tests (fail-open on git error)."""
    config = load_edge_config(repo / "scripts/ci/test_closure_edges.toml") if config is None else config
    closures = compute_all_test_closures(repo, config) if closures is None else closures
    paths = git_changed_paths(base, head, repo=repo)
    if paths is None:
        all_tests = tuple(sorted(closures))
        guard_tests = tuple(sorted(t for t in config.always_run if (repo / t).is_file()))
        return SelectionResult(
            mode="full",
            reason=f"fail-open: git diff {base}...{head} failed",
            changed_paths=(),
            selected_tests=all_tests,
            implicated_tests=all_tests,
            guard_tests=guard_tests,
            per_path_hits={},
        )
    return select_tests(paths, repo=repo, config=config, closures=closures)


def run_historical_replay(
    limit: int = 200,
    *,
    repo: Path = _REPO_ROOT,
) -> dict[str, object]:
    """Replay the selector across the last `limit` first-parent commits on HEAD."""
    config = load_edge_config(repo / "scripts/ci/test_closure_edges.toml")
    closures = compute_all_test_closures(repo, config)
    total_tests = len(closures)

    log_proc = subprocess.run(
        ["git", "log", "--first-parent", f"-n{limit}", "--format=%H %s"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    commits: list[tuple[str, str]] = []
    for line in log_proc.stdout.splitlines():
        if line.strip():
            sha, _, subj = line.partition(" ")
            commits.append((sha, subj))

    mode_counts = {"full": 0, "incremental": 0, "guards_only": 0}
    selected_counts: list[int] = []
    incremental_counts: list[int] = []
    cochange_commits = 0
    cochange_any_recalled = 0
    cochange_all_recalled = 0
    cochange_total_test_files = 0
    cochange_recalled_test_files = 0
    samples: list[dict[str, object]] = []

    for sha, subj in commits:
        show_proc = subprocess.run(
            ["git", "show", "--first-parent", "--name-only", "--format=", sha],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        paths = list(dict.fromkeys(norm_path(p) for p in show_proc.stdout.splitlines() if p.strip()))
        res = select_tests(paths, repo=repo, config=config, closures=closures)
        mode_counts[res.mode] += 1
        selected_counts.append(len(res.selected_tests))
        if res.mode == "incremental":
            incremental_counts.append(len(res.selected_tests))

        # Co-change recall check: when a commit modifies both non-test files and
        # existing test files, do the non-test files alone select the co-changed
        # test files?
        non_test_paths = [p for p in paths if not is_test_file(p) and (repo / p).exists()]
        existing_test_paths = [p for p in paths if is_test_file(p) and (repo / p).exists()]
        if non_test_paths and existing_test_paths:
            cochange_commits += 1
            non_test_res = select_tests(non_test_paths, repo=repo, config=config, closures=closures)
            selected_set = set(non_test_res.selected_tests)
            test_set = set(existing_test_paths)
            hits = selected_set & test_set
            cochange_total_test_files += len(test_set)
            cochange_recalled_test_files += len(hits)
            if hits:
                cochange_any_recalled += 1
            if test_set <= selected_set:
                cochange_all_recalled += 1

        if len(samples) < 25:
            samples.append(
                {
                    "sha": sha[:8],
                    "subject": subj[:72],
                    "mode": res.mode,
                    "changed_files": len(paths),
                    "selected_tests": len(res.selected_tests),
                    "pct_of_suite": round(100.0 * len(res.selected_tests) / max(total_tests, 1), 1),
                    "reason": res.reason,
                }
            )

    selected_sorted = sorted(selected_counts)
    inc_sorted = sorted(incremental_counts)
    n = len(selected_sorted)
    n_inc = len(inc_sorted)
    return {
        "commits_analyzed": n,
        "total_test_files": total_tests,
        "mode_counts": mode_counts,
        "mean_selected_tests_overall": round(sum(selected_sorted) / max(n, 1), 1),
        "mean_pct_overall": round(100.0 * (sum(selected_sorted) / max(n, 1)) / max(total_tests, 1), 1),
        "median_selected_tests_overall": selected_sorted[n // 2] if n else 0,
        "median_pct_overall": round(
            100.0 * (selected_sorted[n // 2] if n else 0) / max(total_tests, 1), 1
        ),
        "mean_selected_tests_incremental": round(sum(inc_sorted) / max(n_inc, 1), 1) if n_inc else 0,
        "median_selected_tests_incremental": inc_sorted[n_inc // 2] if n_inc else 0,
        "cochange_commits": cochange_commits,
        "cochange_any_recalled": cochange_any_recalled,
        "cochange_all_recalled": cochange_all_recalled,
        "cochange_all_recall_rate": round(100.0 * cochange_all_recalled / max(cochange_commits, 1), 1),
        "cochange_total_test_files": cochange_total_test_files,
        "cochange_recalled_test_files": cochange_recalled_test_files,
        "cochange_file_recall_rate": round(
            100.0 * cochange_recalled_test_files / max(cochange_total_test_files, 1), 1
        ),
        "recent_samples": samples,
    }


def _format_explanation(result: SelectionResult, total_tests: int) -> str:
    pct = 100.0 * len(result.selected_tests) / max(total_tests, 1)
    lines = [
        "=== Incremental CI Test Selection ===",
        f"Mode:             {result.mode}",
        f"Reason:           {result.reason}",
        f"Changed files:    {len(result.changed_paths)}",
        f"Implicated tests: {len(result.implicated_tests)}",
        f"Always-run guards:{len(result.guard_tests)}",
        f"Selected total:   {len(result.selected_tests)} / {total_tests} ({pct:.1f}%)",
    ]
    if result.changed_paths:
        lines.append("\nChanged paths -> implicated tests:")
        for p in result.changed_paths[:40]:
            hits = result.per_path_hits.get(p, ())
            lines.append(f"  - {p}: {len(hits)} test(s)")
        if len(result.changed_paths) > 40:
            lines.append(f"  ... ({len(result.changed_paths) - 40} more changed paths)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Base git ref for `git diff <base>...<head>`")
    parser.add_argument("--head", default="HEAD", help="Head git ref (default: HEAD)")
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Explicit repo-relative changed paths (overrides --base/--head)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print human-readable selection summary to stderr",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON selection summary to stdout",
    )
    parser.add_argument(
        "--pytest-args",
        action="store_true",
        help="Print selected pytest target arguments (`.` for full suite, or test file paths)",
    )
    parser.add_argument(
        "--write-pytest-args",
        type=Path,
        help="Write newline-separated pytest target arguments to a file",
    )
    parser.add_argument(
        "--replay",
        type=int,
        metavar="N",
        help="Replay selector over the last N first-parent commits and print report JSON",
    )
    args = parser.parse_args(argv)

    if args.replay is not None:
        report = run_historical_replay(limit=args.replay, repo=_REPO_ROOT)
        print(json.dumps(report, indent=2))
        return 0

    config = load_edge_config()
    closures = compute_all_test_closures(_REPO_ROOT, config)
    total_tests = len(closures)

    if args.paths is not None:
        result = select_tests(args.paths, repo=_REPO_ROOT, config=config, closures=closures)
    elif args.base:
        result = select_from_git_diff(
            args.base,
            args.head,
            repo=_REPO_ROOT,
            config=config,
            closures=closures,
        )
    else:
        parser.error("Must provide either --base <ref>, --paths <path>..., or --replay <N>")

    if args.explain:
        print(_format_explanation(result, total_tests), file=sys.stderr)

    pytest_targets = ["."] if result.mode == "full" else list(result.selected_tests)

    if args.write_pytest_args is not None:
        args.write_pytest_args.write_text("\n".join(pytest_targets) + "\n", encoding="utf-8")

    if args.json:
        payload = {
            "mode": result.mode,
            "reason": result.reason,
            "changed_paths": list(result.changed_paths),
            "selected_tests": list(result.selected_tests),
            "implicated_tests": list(result.implicated_tests),
            "guard_tests": list(result.guard_tests),
            "total_tests": total_tests,
            "pytest_targets": pytest_targets,
        }
        print(json.dumps(payload, indent=2))
    elif args.pytest_args or args.write_pytest_args is None:
        print("\n".join(pytest_targets))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
