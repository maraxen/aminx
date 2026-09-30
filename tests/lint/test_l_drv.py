"""L-DRV lint (spec §2.4): xtrax composition of family drivers.

Stdlib ast only. One planted-violation test per rule, a clean-tree test, and
an allowed-pattern fixture (enumerate/zip/shape/len).
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).with_name("l_drv_allowlist.toml")

_R1_ATTRS = frozenset({"vmap", "pmap", "filter_vmap", "shard_map"})
_JIT_ATTRS = frozenset({"jit", "filter_jit", "scan", "while_loop", "fori_loop", "cond"})
_LAX_LOOP_ATTRS = frozenset({"scan", "while_loop", "fori_loop"})
_HOST_GET_ATTRS = frozenset({"item", "tolist"})


@dataclass(frozen=True)
class _Hit:
    rule: str
    path: str
    detail: str


def _scope(rel: str) -> str | None:
    if rel.startswith("src/aminx/families/") and rel.endswith(".py"):
        return "F"
    if rel.startswith(("src/aminx/model/potts_mpnn/", "src/aminx/model/laser/")) and rel.endswith(".py"):
        return "M"
    name = Path(rel).name
    if rel.startswith("src/aminx/host/") and name.startswith("family_") and name.endswith(".py"):
        return "H"
    return None


def _scope_files() -> list[Path]:
    found: list[Path] = []
    root = _REPO / "src" / "aminx"
    for base in (root / "families", root / "model" / "potts_mpnn", root / "model" / "laser"):
        if base.exists():
            found.extend(sorted(base.rglob("*.py")))
    host = root / "host"
    if host.exists():
        found.extend(sorted(host.glob("family_*.py")))
    return found


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_REPO).as_posix()


def _load_allowlist() -> list[dict[str, str]]:
    data = tomllib.loads(_ALLOWLIST.read_text(encoding="utf-8"))
    rows = data.get("entries", [])
    if not isinstance(rows, list):
        msg = "l_drv_allowlist.toml entries must be a list"
        raise TypeError(msg)
    return rows


def _is_self_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _iter_allowed(node: ast.AST) -> bool:
    if _is_self_attr(node):
        return True
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or not node.args:
        return False
    name = node.func.id
    if name == "enumerate":
        return _is_self_attr(node.args[0])
    if name == "zip":
        return _is_self_attr(node.args[0])
    if name == "range" and len(node.args) == 1:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return arg.value <= 4
        return (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id == "len"
            and bool(arg.args)
            and _is_self_attr(arg.args[0])
        )
    return False


def _annotation_is_tuple(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "tuple"
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name):
            return base.id == "tuple"
        if isinstance(base, ast.Attribute):
            return base.attr == "Tuple"
    return False


def _is_static_field(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_field = (isinstance(func, ast.Attribute) and func.attr == "field") or (
        isinstance(func, ast.Name) and func.id == "field"
    )
    if not is_field:
        return False
    return any(
        kw.arg == "static" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in node.keywords
    )


def _static_fields(class_def: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            static = _annotation_is_tuple(stmt.annotation) or (
                stmt.value is not None and _is_static_field(stmt.value)
            )
            if static:
                names.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign) and _is_static_field(stmt.value):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_eqx_module(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            return True
    return False


def _receiver_is_lax(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "lax"
    if isinstance(node, ast.Attribute):
        return node.attr == "lax" or _receiver_is_lax(node.value)
    return False


def _mentions_attr(node: ast.AST, names: frozenset[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
        if isinstance(child, ast.Attribute) and child.attr in names:
            return True
    return False


def _r4_arg_ok(arg: ast.AST, static_fields: set[str]) -> bool:
    if isinstance(arg, ast.Constant):
        return True
    if (
        isinstance(arg, ast.Subscript)
        and isinstance(arg.value, ast.Attribute)
        and arg.value.attr == "shape"
        and isinstance(arg.slice, ast.Constant)
        and isinstance(arg.slice.value, int)
    ):
        return True
    if isinstance(arg, ast.Attribute) and arg.attr in {"ndim", "size"}:
        return True
    return _is_self_attr(arg) and arg.attr in static_fields


class _Indexer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.functions: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.classes[".".join(self.stack)] = node
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.functions[".".join(self.stack)] = node
        self.generic_visit(node)
        self.stack.pop()


def _r2_bodies(index: _Indexer) -> dict[str, set[str]]:
    """qualname -> static field names of the enclosing eqx.Module, if any."""
    bodies: dict[str, set[str]] = {}
    for qual, class_def in index.classes.items():
        if _is_eqx_module(class_def) and f"{qual}.__call__" in index.functions:
            bodies[f"{qual}.__call__"] = _static_fields(class_def)
    for qual, fn in index.functions.items():
        if any(_mentions_attr(dec, _JIT_ATTRS) for dec in fn.decorator_list):
            bodies.setdefault(qual, _fields_for(index, qual))
    for fn in index.functions.values():
        for call in ast.walk(fn):
            _note_passed(call, index, bodies)
    for call in _module_level_calls(index):
        _note_passed(call, index, bodies)
    return bodies


def _fields_for(index: _Indexer, qual: str) -> set[str]:
    parts = qual.split(".")
    for end in range(len(parts) - 1, 0, -1):
        class_qual = ".".join(parts[:end])
        class_def = index.classes.get(class_qual)
        if class_def is not None and _is_eqx_module(class_def):
            return _static_fields(class_def)
    return set()


def _module_level_calls(index: _Indexer) -> list[ast.Call]:
    calls: list[ast.Call] = []
    # Re-walk is done by the caller via the functions' trees plus a stored module.
    return calls


def _note_passed(call: ast.AST, index: _Indexer, bodies: dict[str, set[str]]) -> None:
    if not isinstance(call, ast.Call) or not call.args or not _mentions_attr(call.func, _JIT_ATTRS):
        return
    arg0 = call.args[0]
    target = arg0.id if isinstance(arg0, ast.Name) else arg0.attr if isinstance(arg0, ast.Attribute) else None
    if target is None:
        return
    for qual in index.functions:
        if qual == target or qual.endswith(f".{target}"):
            bodies.setdefault(qual, _fields_for(index, qual))


def _comprehension_iters(node: ast.AST) -> list[ast.AST] | None:
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [gen.iter for gen in node.generators]
    if isinstance(node, ast.DictComp):
        return [gen.iter for gen in node.generators]
    return None


def _lint_tree(tree: ast.AST, rel: str, scope: str) -> tuple[list[_Hit], set[tuple[str, str]]]:
    hits: list[_Hit] = []
    lax_sites: set[tuple[str, str]] = set()
    index = _Indexer()
    index.visit(tree)
    if scope in {"F", "H"}:
        hits.extend(_lint_r1(tree, rel))
    if scope in {"F", "M"}:
        bodies = _r2_bodies(index)
        # Module-level calls are not inside functions; scan the whole module.
        for call in ast.walk(tree):
            _note_passed(call, index, bodies)
        hits.extend(_lint_r2_r4(index, bodies, rel))
        sites = _lint_r3_sites(tree, index)
        lax_sites |= {(rel, qual) for qual in sites}
        allowed = {
            (row.get("path"), row.get("qualname"))
            for row in _load_allowlist()
            if row.get("reason") == "sequential_dependency"
        }
        for qual in sites:
            if (rel, qual) not in allowed:
                hits.append(_Hit("R3", rel, f"lax loop at {qual} is not allowlisted"))
    if scope in {"F", "M", "H"}:
        hits.extend(_lint_r5(tree, rel))
    return hits, lax_sites


def _lint_r1(tree: ast.AST, rel: str) -> list[_Hit]:
    hits: list[_Hit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        banned = False
        if isinstance(func, ast.Name) and func.id in _R1_ATTRS:
            banned = True
        if isinstance(func, ast.Attribute) and func.attr in _R1_ATTRS:
            banned = True
        if banned:
            hits.append(_Hit("R1", rel, f"banned map call at line {node.lineno}"))
    return hits


def _lint_r2_r4(
    index: _Indexer,
    bodies: dict[str, set[str]],
    rel: str,
) -> list[_Hit]:
    hits: list[_Hit] = []
    for qual, static_fields in bodies.items():
        fn = index.functions[qual]
        for node in ast.walk(fn):
            if isinstance(node, ast.While):
                hits.append(_Hit("R2", rel, f"while in {qual}"))
            elif isinstance(node, ast.For) and not _iter_allowed(node.iter):
                hits.append(_Hit("R2", rel, f"for in {qual}"))
            else:
                iters = _comprehension_iters(node)
                if iters is not None and any(not _iter_allowed(it) for it in iters):
                    hits.append(_Hit("R2", rel, f"comprehension in {qual}"))
            if isinstance(node, ast.Call):
                hits.extend(_lint_r4_call(node, rel, qual, static_fields))
    return hits


def _lint_r4_call(node: ast.Call, rel: str, qual: str, static_fields: set[str]) -> list[_Hit]:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _HOST_GET_ATTRS:
        return [_Hit("R4", rel, f"{func.attr}() in {qual}")]
    if isinstance(func, ast.Attribute) and func.attr == "asarray":
        return [_Hit("R4", rel, f"asarray() in {qual}")]
    if isinstance(func, ast.Name) and func.id == "asarray":
        return [_Hit("R4", rel, f"asarray() in {qual}")]
    if isinstance(func, ast.Attribute) and func.attr == "device_get":
        return [_Hit("R4", rel, f"device_get() in {qual}")]
    if isinstance(func, ast.Name) and func.id == "device_get":
        return [_Hit("R4", rel, f"device_get() in {qual}")]
    if isinstance(func, ast.Name) and func.id in {"int", "float", "len"} and node.args:
        if not _r4_arg_ok(node.args[0], static_fields):
            return [_Hit("R4", rel, f"{func.id}() on a traced value in {qual}")]
    return []


def _lint_r3_sites(tree: ast.AST, index: _Indexer) -> set[str]:
    sites: set[str] = set()
    for qual, fn in index.functions.items():
        for node in ast.walk(fn):
            if _is_lax_loop(node):
                sites.add(qual)
    for node in ast.walk(tree):
        if _is_lax_loop(node):
            # Covered when the call sits inside a function. A module-level call
            # still counts, under the synthetic qualname "<module>".
            if not any(node in ast.walk(fn) for fn in index.functions.values()):
                sites.add("<module>")
    return sites


def _is_lax_loop(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LAX_LOOP_ATTRS
        and _receiver_is_lax(node.func.value)
    )


def _lint_r5(tree: ast.AST, rel: str) -> list[_Hit]:
    hits: list[_Hit] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "aminx.potts" or alias.name.startswith("aminx.potts."):
                    hits.append(_Hit("R5", rel, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "aminx.potts" or node.module.startswith("aminx.potts."):
                hits.append(_Hit("R5", rel, f"from {node.module} import"))
    return hits


def lint_paths(paths: list[Path], *, check_stale: bool) -> list[_Hit]:
    """Lint ``paths`` (each must be inside an L-DRV scope)."""
    hits: list[_Hit] = []
    live: set[tuple[str, str]] = set()
    for path in paths:
        rel = _rel(path)
        scope = _scope(rel)
        if scope is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        file_hits, sites = _lint_tree(tree, rel, scope)
        hits.extend(file_hits)
        live |= sites
    if check_stale:
        hits.extend(_stale_hits(live))
    return hits


def _stale_hits(live: set[tuple[str, str]]) -> list[_Hit]:
    hits: list[_Hit] = []
    for row in _load_allowlist():
        reason = row.get("reason")
        path = row.get("path", "")
        qual = row.get("qualname", "")
        if reason != "sequential_dependency":
            hits.append(_Hit("R3", path, f"allowlist reason {reason!r} is not sequential_dependency"))
            continue
        if (path, qual) not in live:
            hits.append(_Hit("R3", path, f"stale allowlist entry {qual}"))
    return hits


def _write(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)
    parent = path.parent
    if parent.exists() and parent != _REPO and not any(parent.iterdir()):
        parent.rmdir()


def test_r1_planted_vmap_in_host_family_module() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "host",
        "family_ldr_v_r1_probe.py",
        "import jax\n\ndef f(x):\n    return jax.vmap(lambda y: y)(x)\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert any(hit.rule == "R1" for hit in hits)


def test_r2_planted_for_in_module_call() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "families",
        "_ldr_v_r2_probe.py",
        "import equinox as eqx\n"
        "class M(eqx.Module):\n"
        "    def __call__(self, x):\n"
        "        total = 0\n"
        "        for i in range(8):\n"
        "            total += i\n"
        "        return total\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert any(hit.rule == "R2" for hit in hits)


def test_r3_planted_unlisted_lax_scan() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "families",
        "_ldr_v_r3_probe.py",
        "import jax\n"
        "def step(carry, x):\n"
        "    return carry, x\n"
        "def body(xs):\n"
        "    return jax.lax.scan(step, 0, xs)\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert any(hit.rule == "R3" for hit in hits)


def test_r4_planted_int_of_traced_value() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "families",
        "_ldr_v_r4_probe.py",
        "import equinox as eqx\n"
        "class M(eqx.Module):\n"
        "    def __call__(self, x):\n"
        "        return int(x)\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert any(hit.rule == "R4" for hit in hits)


def test_r5_planted_potts_import() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "host",
        "family_ldr_v_r5_probe.py",
        "import aminx.potts\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert any(hit.rule == "R5" for hit in hits)


def test_allowed_patterns_pass() -> None:
    path = _write(
        _REPO / "src" / "aminx" / "families",
        "_ldr_v_allowed_probe.py",
        "import equinox as eqx\n"
        "class Decoder(eqx.Module):\n"
        "    layers: tuple[eqx.Module, ...]\n"
        "    def __call__(self, x, keys):\n"
        "        for layer in enumerate(self.layers):\n"
        "            pass\n"
        "        for layer, key in zip(self.layers, keys):\n"
        "            pass\n"
        "        n = int(x.shape[0])\n"
        "        m = len(self.layers)\n"
        "        return n + m\n",
    )
    try:
        hits = lint_paths([path], check_stale=False)
    finally:
        _remove(path)
    assert hits == []


def test_clean_tree() -> None:
    hits = lint_paths(_scope_files(), check_stale=True)
    assert hits == [], "\n".join(f"{hit.rule} {hit.path}: {hit.detail}" for hit in hits)
