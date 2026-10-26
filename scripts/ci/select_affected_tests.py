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
  importing through that package), while expanding transitive imports only through
  the target module (`src/aminx/potts/runner.py`).
- Evaluates `Path(__file__)`-derived expressions passed to `spec_from_file_location`
  and `run_path`, literal `import_module` / `importlib.resources.files("<pkg>")`
  calls, `@pytest.mark.port_wave("<wave>")` markers, and dotted `"aminx.<pkg>..."`
  strings in test modules.
- Fails open (`mode = "full"`, running the entire pytest suite) whenever:
  1. `git diff` fails or refs are unavailable;
  2. Any closure in the repository has an undeclared dynamic call site
     (`not closure.sound`);
  3. Any changed file matches `[global].files` or `[global].prefixes` in
     `scripts/ci/test_closure_edges.toml`;
  4. Any changed file under `src/`, `tests/`, or `scripts/` was deleted or renamed
     away;
  5. Any changed file under `src/` or code/fixture file under `tests/` matches
     zero test closures.
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
_ROOTS = ("src", "tests", "scripts")
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
_BROAD_DIRS = frozenset(
    {
        "",
        ".",
        "src",
        "src/aminx",
        "tests",
        "scripts",
        ".praxia",
        ".praxia/docs",
        "agent_assets",
    }
)
_DOTTED_AMINX_RE = re.compile(r"^aminx(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_RESOLVE_CACHE: dict[tuple[Path, str, Path | None], str | None] = {}


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
    if not parts:
        _RESOLVE_CACHE[cache_key] = None
        return None

    search_bases: list[Path] = [repo / root for root in _ROOTS]
    search_bases.append(repo)
    if importer_dir is not None:
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


def _module_name_for_path(path: str) -> tuple[str, bool]:
    """Return `(dotted_name, is_package_init)` for a repo-relative `.py` path."""
    for root in _ROOTS:
        prefix = root + "/"
        if path.startswith(prefix):
            rel = path[len(prefix) :].removesuffix(".py")
            parts = rel.split("/")
            is_pkg = parts[-1] == "__init__"
            if is_pkg:
                parts = parts[:-1]
            return ".".join(parts), is_pkg
    return "", False


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
    """Statically evaluate `Path(__file__)` and constant path expressions within a file."""
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
            return (left / node.right.value).resolve()
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
                    return base.with_name(node.args[0].value).resolve()
                if (
                    node.func.attr == "joinpath"
                    and node.args
                    and all(
                        isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) < 200
                        for a in node.args
                    )
                ):
                    return base.joinpath(*(str(a.value) for a in node.args)).resolve()
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
    path_files: set[str],
    data_prefixes: set[str],
) -> None:
    """Resolve `@pytest.mark.port_wave('<wave>')` to target TOML, manifest, and oracle."""
    target_rel = f"tests/port/targets/{wave_name}.toml"
    target_path = repo / target_rel
    if not target_path.is_file():
        return
    path_files.add(target_rel)
    try:
        target_doc = tomllib.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        return

    wave_id = target_doc.get("wave_id")
    if isinstance(wave_id, str) and wave_id:
        manifest_rel = f"tests/port/manifests/{wave_id}.toml"
        if (repo / manifest_rel).is_file():
            path_files.add(manifest_rel)

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


def _scan_python_file(
    repo: Path,
    path: str,
    *,
    scan_dotted_strings: bool,
) -> _FileScan:
    """Parse `path` and extract target imports, ancestor packages, paths, and dynamic calls."""
    full_path = repo / path
    repo_resolved = repo.resolve()
    file_resolved = full_path.resolve()
    source = full_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    # Python 3 has no implicit relative imports inside packages (`src/aminx/`);
    # only `tests/` and `scripts/` put their own directory on sys.path.
    importer_dir = None if path.startswith("src/") else full_path.parent

    own, is_pkg = _module_name_for_path(path)
    package = own if is_pkg else own.rpartition(".")[0]

    target_modules: set[str] = set()
    ancestor_modules: set[str] = set()
    path_files: set[str] = set()
    data_prefixes: set[str] = set()
    dynamic_lines: list[int] = []

    parent_map: dict[ast.AST, ast.AST] = {}
    path_env: dict[str, Path] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            val_path = _eval_path_expr(
                parent.value,
                repo_resolved=repo_resolved,
                file_resolved=file_resolved,
                env=path_env,
            )
            if val_path is not None:
                path_env[parent.targets[0].id] = val_path
                if val_path.exists() and val_path.is_relative_to(repo_resolved):
                    _register_candidate_repo_path(
                        repo,
                        val_path.relative_to(repo_resolved).as_posix(),
                        path_files,
                        data_prefixes,
                    )
        elif isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name) and parent.value is not None:
            val_path = _eval_path_expr(
                parent.value,
                repo_resolved=repo_resolved,
                file_resolved=file_resolved,
                env=path_env,
            )
            if val_path is not None:
                path_env[parent.target.id] = val_path
                if val_path.exists() and val_path.is_relative_to(repo_resolved):
                    _register_candidate_repo_path(
                        repo,
                        val_path.relative_to(repo_resolved).as_posix(),
                        path_files,
                        data_prefixes,
                    )

    def add_target_module(mod_name: str) -> None:
        if not mod_name:
            return
        ancs = _ancestors(mod_name)
        for anc in ancs[:-1]:
            if resolve_module_file(repo, anc, importer_dir=importer_dir) is not None:
                ancestor_modules.add(anc)
        if resolve_module_file(repo, mod_name, importer_dir=importer_dir) is not None:
            target_modules.add(mod_name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_target_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg_parts = package.split(".") if package else []
                keep = len(pkg_parts) - (node.level - 1)
                base = ".".join(pkg_parts[: max(keep, 0)])
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""

            if base:
                for anc in _ancestors(base):
                    if resolve_module_file(repo, anc, importer_dir=importer_dir) is not None:
                        ancestor_modules.add(anc)
                for alias in node.names:
                    cand = f"{base}.{alias.name}"
                    if resolve_module_file(repo, cand, importer_dir=importer_dir) is not None:
                        target_modules.add(cand)
                    elif resolve_module_file(repo, base, importer_dir=importer_dir) is not None:
                        target_modules.add(base)
            elif node.level:
                for alias in node.names:
                    sibling = importer_dir / f"{alias.name}.py"
                    if sibling.is_file():
                        path_files.add(sibling.relative_to(repo).as_posix())
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parent = parent_map.get(node)
            if not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)):
                eval_p = _eval_path_expr(
                    node,
                    repo_resolved=repo_resolved,
                    file_resolved=file_resolved,
                    env=path_env,
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
                    _resolve_port_wave(repo, arg0.value, path_files, data_prefixes)
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
                        env=path_env,
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
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
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
            elif scan_dotted_strings and _DOTTED_AMINX_RE.match(val):
                mod_target, mod_ancs = _resolve_longest_aminx_module(repo, val)
                for anc in mod_ancs:
                    ancestor_modules.add(anc)
                if mod_target is not None and mod_target != "aminx":
                    target_modules.add(mod_target)

    return _FileScan(
        target_modules=frozenset(target_modules),
        ancestor_modules=frozenset(ancestor_modules),
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


def compute_all_test_closures(
    repo: Path = _REPO_ROOT,
    config: EdgeConfig | None = None,
) -> dict[str, TestClosure]:
    """Compute the static dependency closure for every test file in `repo`."""
    config = load_edge_config(repo / "scripts/ci/test_closure_edges.toml") if config is None else config
    always_run_set = set(config.always_run)
    scan_cache: dict[str, _FileScan] = {}

    def get_scan(rel_path: str) -> _FileScan:
        if rel_path not in scan_cache:
            scan_dotted = rel_path.startswith("tests/") and rel_path not in always_run_set
            scan_cache[rel_path] = _scan_python_file(
                repo,
                rel_path,
                scan_dotted_strings=scan_dotted,
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
            dynamic_ok = bool(mod_decl.get("dynamic_ok", False))
            for soft_mod in _string_list(mod_decl, "soft", f"[module.{cur!r}]"):
                for anc in _ancestors(soft_mod)[:-1]:
                    anc_file = resolve_module_file(repo, anc, importer_dir=importer_dir)
                    if anc_file is not None:
                        files.add(anc_file)
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
            if scan.dynamic_lines and not dynamic_ok:
                undeclared.extend(f"{cur}:{line}" for line in scan.dynamic_lines)

            for anc_mod in scan.ancestor_modules:
                anc_file = resolve_module_file(repo, anc_mod, importer_dir=importer_dir)
                if anc_file is not None:
                    files.add(anc_file)

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

        # Deleted or renamed-away file under src/, tests/, or scripts/ -> full suite
        if path.startswith(("src/", "tests/", "scripts/")) and not (repo / path).exists():
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
        # Note: `.py` modules under `src/` must match `closure.files` directly
        # (not a directory `data_prefix` from an AST lint test), so an unreferenced
        # `src/aminx/*.py` module cannot be masked by a directory prefix.
        is_src_py = path.startswith("src/") and path.endswith(".py")
        hits: list[str] = []
        if is_test_file(path) and path in closures:
            hits.append(path)
        for test_file, closure in closures.items():
            if test_file == path:
                continue
            if path in closure.files or (
                not is_src_py
                and closure.data_prefixes
                and path.startswith(closure.data_prefixes)
            ):
                hits.append(test_file)

        hits = sorted(set(hits))
        per_path_hits[path] = tuple(hits)

        if hits:
            implicated.update(hits)
            continue

        # Fail-open check: any file under `src/` or non-doc file under `tests/`
        # that matched zero test closures must trigger the full suite.
        if path.startswith("src/") or (
            path.startswith("tests/") and not path.endswith(".md")
        ):
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
        ["git", "diff", "--name-only", f"{base}...{head}"],
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
    cochange_recalled = 0
    samples: list[dict[str, object]] = []

    for sha, subj in commits:
        diff_proc = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "--first-parent", sha],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        paths = [norm_path(p) for p in diff_proc.stdout.splitlines() if p.strip()]
        res = select_tests(paths, repo=repo, config=config, closures=closures)
        mode_counts[res.mode] += 1
        selected_counts.append(len(res.selected_tests))
        if res.mode == "incremental":
            incremental_counts.append(len(res.selected_tests))

        src_paths = [p for p in paths if p.startswith("src/") and (repo / p).exists()]
        existing_test_paths = [p for p in paths if is_test_file(p) and (repo / p).exists()]
        if src_paths and existing_test_paths:
            cochange_commits += 1
            src_only_res = select_tests(src_paths, repo=repo, config=config, closures=closures)
            if set(src_only_res.selected_tests) & set(existing_test_paths):
                cochange_recalled += 1

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
        "cochange_recalled": cochange_recalled,
        "cochange_recall_rate": round(100.0 * cochange_recalled / max(cochange_commits, 1), 1),
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
