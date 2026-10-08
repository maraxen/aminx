"""Per-row import closure: which changed paths can alter a ledger row's result.

The original staleness test (``_coverage._touches_scoped``) is a file-prefix test:
one commit anywhere under ``src/aminx/`` invalidates every row. This module narrows
that to the files a row's vehicle can actually execute, and fails SAFE everywhere it
cannot prove a file is irrelevant.

What a row's closure is
-----------------------
The static import closure of ``scripts/parity/<slug>.py`` over two roots
(``src/`` and ``scripts/parity/``): every ``import`` / ``from ... import`` statement
at any nesting depth (lazy imports inside functions count), plus the ``__init__.py``
of every package on each import path, plus the modules named in the slug's declared
``soft`` list in ``closure_edges.toml``.

It is deliberately an OVER-approximation of what runs. A lazy import that a given run
never reaches still counts.

What it cannot see, and what happens then
-----------------------------------------
- Dynamic imports (``importlib.import_module(<non-literal>)``, ``__import__``,
  ``spec_from_file_location``, ``run_path``, entry points, ``pkgutil``) and package-data
  reads (``importlib.resources.files``). A closure file containing one of these is
  UNSOUND unless the slug lists that file under ``dynamic_ok`` in
  ``closure_edges.toml``; an unsound closure makes the row fall back to the old global
  prefix test. Declaring it is a reviewed claim that the declared ``soft`` and ``data``
  entries cover what the site loads for this vehicle.
- Imports that resolve to neither root (third-party, stdlib, upstream checkouts) are
  ignored here; ``pyproject.toml`` / ``uv.lock`` stay globally scoped.

Paths that stay GLOBAL (any change invalidates every row), unchanged from before:
``pyproject.toml``, ``uv.lock``, ``scripts/recapture/``, ``aminx-oracles/``, and any
non-``.py`` file under ``src/aminx/`` or ``scripts/parity/`` not named in the slug's
``data`` list (except a vehicle's own ``.bth.toml``, which ``sidecar_sha256`` already
pins per row).

``tests/port/`` is no longer scoped for rows: no graded vehicle imports from it, and
the port suite is re-run by the gate's step 2 on every gate run, so a change there
cannot make a recorded vehicle run wrong. ``test_closure_selftest.py`` pins this.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_ROOTS = ("src", "scripts/parity")
_NARROWED_PY_PREFIXES = ("src/aminx/", "scripts/parity/")
_ALWAYS_GLOBAL_FILES = frozenset({"pyproject.toml", "uv.lock"})
_ALWAYS_GLOBAL_PREFIXES = ("scripts/recapture/", "aminx-oracles/")
# Calls that load code or data the import graph cannot see.
_DYNAMIC_CALLS = frozenset(
  {
    "import_module",
    "__import__",
    "spec_from_file_location",
    "run_path",
    "entry_points",
    "iter_modules",
    "files",  # importlib.resources.files: package data
  }
)
_EDGES_FILE = Path(__file__).with_name("closure_edges.toml")


@dataclass(frozen=True)
class Closure:
  """The files one row depends on, and whether that set can be trusted."""

  slug: str
  files: frozenset[str]
  data_prefixes: tuple[str, ...] = ()
  undeclared_dynamic: tuple[str, ...] = field(default=())

  @property
  def sound(self) -> bool:
    return not self.undeclared_dynamic


def _norm(raw: str) -> str:
  path = raw.replace("\\", "/")
  while path.startswith("./"):
    path = path[2:]
  return path


def _module_file(repo: Path, module: str) -> str | None:
  """Repo-relative path for a dotted module name under the closure roots, if any."""
  parts = module.split(".")
  for root in _ROOTS:
    base = repo / root
    as_file = base.joinpath(*parts).with_suffix(".py")
    if as_file.is_file():
      return as_file.relative_to(repo).as_posix()
    as_pkg = base.joinpath(*parts, "__init__.py")
    if as_pkg.is_file():
      return as_pkg.relative_to(repo).as_posix()
  return None


def _module_name(path: str) -> tuple[str, bool]:
  """Dotted name for a repo-relative file under a root, and whether it is a package."""
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


def _ancestors(module: str) -> Iterable[str]:
  parts = module.split(".")
  for end in range(1, len(parts) + 1):
    yield ".".join(parts[:end])


def _call_name(node: ast.Call) -> str | None:
  func = node.func
  if isinstance(func, ast.Attribute):
    return func.attr
  if isinstance(func, ast.Name):
    return func.id
  return None


def scan_file(repo: Path, path: str) -> tuple[set[str], list[int]]:
  """Modules a file imports (resolvable ones only) and its dynamic-site line numbers."""
  tree = ast.parse((repo / path).read_text(encoding="utf-8"), filename=path)
  own, is_pkg = _module_name(path)
  package = own if is_pkg else own.rpartition(".")[0]
  found: set[str] = set()
  dynamic: list[int] = []

  def add(name: str) -> None:
    for part in _ancestors(name):
      if _module_file(repo, part) is not None:
        found.add(part)

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        add(alias.name)
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        pkg_parts = package.split(".") if package else []
        keep = len(pkg_parts) - (node.level - 1)
        base = ".".join(pkg_parts[: max(keep, 0)])
        base = f"{base}.{node.module}" if node.module else base
      else:
        base = node.module or ""
      if base:
        add(base)
        for alias in node.names:
          add(f"{base}.{alias.name}")
    elif isinstance(node, ast.Call):
      name = _call_name(node)
      if name in _DYNAMIC_CALLS:
        literal = bool(node.args) and isinstance(node.args[0], ast.Constant)
        if name in {"import_module", "__import__"} and literal:
          add(str(node.args[0].value))
        else:
          dynamic.append(node.lineno)
  return found, dynamic


def load_edges(path: Path = _EDGES_FILE) -> Mapping[str, Mapping[str, object]]:
  if not path.is_file():
    return {}
  table = tomllib.loads(path.read_text(encoding="utf-8")).get("slug", {})
  return {str(slug): body for slug, body in table.items() if isinstance(body, dict)}


def _strings(body: Mapping[str, object], key: str) -> list[str]:
  value = body.get(key, [])
  if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    msg = f"closure_edges.toml: {key!r} must be a list of strings, got {value!r}"
    raise ValueError(msg)
  return list(value)


def compute_closure(
  repo: Path,
  slug: str,
  edges: Mapping[str, Mapping[str, object]] | None = None,
) -> Closure:
  """Static closure of ``scripts/parity/<slug>.py`` plus the slug's declared edges."""
  edges = load_edges() if edges is None else edges
  body = edges.get(slug, {})
  entry = _norm(str(body.get("entry", f"scripts/parity/{slug}.py")))
  if not (repo / entry).is_file():
    msg = f"closure entry for {slug!r} does not exist: {entry}"
    raise FileNotFoundError(msg)

  dynamic_ok = {_norm(item) for item in _strings(body, "dynamic_ok")}
  data = tuple(_norm(item) for item in _strings(body, "data"))
  files: set[str] = set()
  undeclared: list[str] = []
  queue: list[str] = [entry]
  for soft in _strings(body, "soft"):
    target = _module_file(repo, soft)
    if target is None:
      msg = f"closure_edges.toml: soft edge {soft!r} for {slug!r} resolves to no file"
      raise ValueError(msg)
    queue.append(target)

  while queue:
    path = queue.pop()
    if path in files:
      continue
    files.add(path)
    modules, dynamic = scan_file(repo, path)
    if dynamic and path not in dynamic_ok:
      undeclared.extend(f"{path}:{line}" for line in dynamic)
    for module in modules:
      target = _module_file(repo, module)
      if target is not None and target not in files:
        queue.append(target)
  return Closure(slug, frozenset(files), data, tuple(sorted(undeclared)))


def row_is_stale(
  paths: Sequence[str],
  closure: Closure | None,
) -> bool:
  """True when any changed path can alter the closure's row result.

  ``closure=None`` (or an unsound closure) reproduces the old global behaviour for
  every narrowed prefix, so a missing or unreviewed declaration can only ever make a
  row stale, never fresh.
  """
  for raw in paths:
    path = _norm(raw)
    if path in _ALWAYS_GLOBAL_FILES or path.startswith(_ALWAYS_GLOBAL_PREFIXES):
      return True
    if path.startswith("tests/port/"):
      continue
    if not path.startswith(_NARROWED_PY_PREFIXES):
      continue
    if closure is None or not closure.sound:
      return True
    if path in closure.files or path.startswith(closure.data_prefixes):
      return True
    if path.endswith(".py"):
      continue
    if path.startswith("scripts/parity/") and path.endswith(".bth.toml"):
      continue  # the row's own sidecar is pinned by sidecar_sha256
    return True  # non-Python data under a narrowed prefix and not declared
  return False


def loaded_outside_closure(loaded: Iterable[str], closure: Closure) -> list[str]:
  """Files a real run loaded that the row's closure does not cover.

  Only ``.py`` files under the narrowed prefixes matter: those are the ones whose
  change the closure rule would ignore. A non-empty result means the closure (or a
  declaration in ``closure_edges.toml``) is too small and rows that depend on it
  could grade stale evidence.
  """
  return sorted(
    path
    for path in (_norm(p) for p in loaded)
    if path.endswith(".py") and path.startswith(_NARROWED_PY_PREFIXES) and path not in closure.files
  )
