"""Inventory which public xtrax symbols aminx uses, by area (src, tests, scripts).

Static analysis only (``ast``); neither xtrax nor aminx is imported, so no JAX
initialises. Re-exports inside xtrax are resolved to the defining module, so
``from xtrax.tiling import select_bucket`` counts against
``xtrax.tiling.bucket.select_bucket``.

Surface: for each non-private xtrax module, ``__all__`` when present,
otherwise its public top-level classes, functions and UPPER_CASE constants;
plus anything a public module re-exports. Usage: every ``import xtrax...`` /
``from xtrax... import ...`` in aminx (including function-local imports), and
attribute chains through an imported xtrax module (``tiling.select_bucket``).

Controls (all must hold, or the inventory is not trusted):
- positive: ``xtrax.tiling.bucket.select_bucket`` is used in src (export/buckets.py);
- negative: ``xtrax.tiling.iterator.BucketIterator`` is unused everywhere;
- planted: a synthetic file importing ``BucketIterator`` IS detected by the scanner.

Usage:
  uv run --no-project python3 scripts/audit/xtrax_usage_inventory.py \
    --xtrax-root /path/to/site-packages/xtrax --out outputs/audit/xtrax_usage.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("xtrax_usage_inventory")

AREAS = ("src", "tests", "scripts")
_CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
POSITIVE = ("xtrax.tiling.bucket", "select_bucket")
NEGATIVE = ("xtrax.tiling.iterator", "BucketIterator")


@dataclass
class Module:
  """One parsed xtrax module."""

  name: str
  path: Path
  is_package: bool
  tree: ast.Module
  defs: dict[str, str] = field(default_factory=dict)  # name -> kind
  all_names: list[str] | None = None
  reexports: dict[str, tuple[str, str]] = field(default_factory=dict)  # local -> (module, name)


def _module_name(root: Path, path: Path) -> tuple[str, bool]:
  rel = path.relative_to(root.parent).with_suffix("")
  parts = list(rel.parts)
  is_package = parts[-1] == "__init__"
  if is_package:
    parts = parts[:-1]
  return ".".join(parts), is_package


def _resolve_relative(module: Module, node: ast.ImportFrom) -> str | None:
  if node.level == 0:
    return node.module
  base = module.name.split(".")
  if not module.is_package:
    base = base[:-1]
  if node.level > 1:
    base = base[: -(node.level - 1)]
  if node.module:
    base = [*base, *node.module.split(".")]
  return ".".join(base)


def load_xtrax(root: Path) -> dict[str, Module]:
  """Parse every xtrax module under ``root`` (the package directory)."""
  modules: dict[str, Module] = {}
  for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts or "tests" in path.relative_to(root).parts:
      continue
    name, is_package = _module_name(root, path)
    try:
      tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
      logger.warning("xtrax parse error %s: %s", path, exc)
      continue
    mod = Module(name=name, path=path, is_package=is_package, tree=tree)
    for stmt in tree.body:
      if isinstance(stmt, ast.ClassDef):
        mod.defs[stmt.name] = "class"
      elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        mod.defs[stmt.name] = "function"
      elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
          if isinstance(target, ast.Name):
            if target.id == "__all__" and isinstance(stmt.value, (ast.List, ast.Tuple)):
              mod.all_names = [
                elt.value for elt in stmt.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
              ]
            elif _CONSTANT.match(target.id):
              mod.defs[target.id] = "constant"
      elif isinstance(stmt, ast.ImportFrom):
        source = _resolve_relative(mod, stmt)
        if source and source.split(".")[0] == "xtrax":
          for alias in stmt.names:
            if alias.name != "*":
              mod.reexports[alias.asname or alias.name] = (source, alias.name)
    modules[name] = mod
  return modules


def canonical(modules: dict[str, Module], module: str, name: str, depth: int = 0) -> tuple[str, str] | None:
  """Resolve ``module.name`` to its defining (module, name); None if unknown or a module."""
  if depth > 20:
    return None
  mod = modules.get(module)
  if mod is None:
    return None
  if name in mod.defs:
    return (module, name)
  if name in mod.reexports:
    src_mod, src_name = mod.reexports[name]
    if f"{src_mod}.{src_name}" in modules:  # re-export of a submodule
      return None
    return canonical(modules, src_mod, src_name, depth + 1)
  return None


def public_surface(modules: dict[str, Module]) -> dict[tuple[str, str], str]:
  """Canonical public symbols -> kind."""
  surface: dict[tuple[str, str], str] = {}
  for mod in modules.values():
    if any(part.startswith("_") and part != "__init__" for part in mod.name.split(".")):
      continue
    names = mod.all_names if mod.all_names is not None else [n for n in mod.defs if not n.startswith("_")]
    if mod.all_names is None:
      names += [n for n in mod.reexports if not n.startswith("_") and mod.is_package]
    for name in names:
      target = canonical(modules, mod.name, name)
      if target is not None:
        surface[target] = modules[target[0]].defs[target[1]]
  return surface


@dataclass
class Usage:
  """Per-canonical-symbol usage, keyed by area."""

  files: dict[tuple[str, str], dict[str, set[str]]] = field(
    default_factory=lambda: defaultdict(lambda: {area: set() for area in AREAS}),
  )
  modules: dict[str, dict[str, set[str]]] = field(
    default_factory=lambda: defaultdict(lambda: {area: set() for area in AREAS}),
  )
  unresolved: list[dict[str, str]] = field(default_factory=list)
  parse_errors: list[str] = field(default_factory=list)


def _attr_chain(node: ast.Attribute) -> list[str] | None:
  parts: list[str] = []
  cur: ast.expr = node
  while isinstance(cur, ast.Attribute):
    parts.append(cur.attr)
    cur = cur.value
  if not isinstance(cur, ast.Name):
    return None
  parts.append(cur.id)
  return parts[::-1]


def scan_file(path: Path, area: str, rel: str, modules: dict[str, Module], usage: Usage) -> None:
  """Record every xtrax symbol ``path`` uses."""
  try:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  except (SyntaxError, UnicodeDecodeError) as exc:
    usage.parse_errors.append(f"{rel}: {exc}")
    return
  module_aliases: dict[str, str] = {}  # local name -> xtrax module

  def record(module: str, name: str, how: str) -> None:
    target = canonical(modules, module, name)
    if target is not None:
      usage.files[target][area].add(rel)
    elif f"{module}.{name}" in modules:
      usage.modules[f"{module}.{name}"][area].add(rel)
    else:
      usage.unresolved.append({"file": rel, "area": area, "import": f"{module}.{name}", "how": how})

  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] == "xtrax":
      for alias in node.names:
        if alias.name == "*":
          usage.unresolved.append({"file": rel, "area": area, "import": f"{node.module}.*", "how": "star"})
          continue
        full = f"{node.module}.{alias.name}"
        if full in modules:
          module_aliases[alias.asname or alias.name] = full
          usage.modules[full][area].add(rel)
        else:
          record(node.module, alias.name, "from-import")
    elif isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name.split(".")[0] == "xtrax":
          local = alias.asname or alias.name.split(".")[0]
          module_aliases[local] = alias.name if alias.asname else "xtrax"
          usage.modules[alias.name][area].add(rel)
  for node in ast.walk(tree):
    if not isinstance(node, ast.Attribute):
      continue
    chain = _attr_chain(node)
    if not chain or chain[0] not in module_aliases:
      continue
    module = module_aliases[chain[0]]
    rest = chain[1:]
    while rest and f"{module}.{rest[0]}" in modules:
      module = f"{module}.{rest[0]}"
      rest = rest[1:]
    if rest:
      target = canonical(modules, module, rest[0])
      if target is not None:
        usage.files[target][area].add(rel)


def scan_repo(repo: Path, modules: dict[str, Module]) -> Usage:
  """Scan src/, tests/ and scripts/ of the aminx checkout."""
  usage = Usage()
  for area in AREAS:
    base = repo / area
    if not base.is_dir():
      continue
    for path in sorted(base.rglob("*.py")):
      if "__pycache__" in path.parts:
        continue
      scan_file(path, area, path.relative_to(repo).as_posix(), modules, usage)
  return usage


def xtrax_version(root: Path) -> str:
  """Version from the sibling dist-info, else from pyproject, else unknown."""
  for meta in sorted(root.parent.glob("xtrax-*.dist-info/METADATA")):
    for line in meta.read_text(encoding="utf-8").splitlines():
      if line.startswith("Version:"):
        return line.split(":", 1)[1].strip()
  pyproject = root.parent.parent / "pyproject.toml"
  if pyproject.is_file():
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    if match:
      return match.group(1) + " (pyproject)"
  return "unknown"


def planted_control(modules: dict[str, Module]) -> bool:
  """The scanner must detect a use it was planted with."""
  with tempfile.TemporaryDirectory() as tmp:
    planted = Path(tmp) / "planted.py"
    planted.write_text("from xtrax.tiling import BucketIterator\n", encoding="utf-8")
    usage = Usage()
    scan_file(planted, "src", "planted.py", modules, usage)
    return bool(usage.files.get(NEGATIVE, {}).get("src"))


def main() -> int:
  parser = argparse.ArgumentParser(description="Inventory which public xtrax symbols aminx uses.")
  parser.add_argument("--xtrax-root", type=Path, required=True, help="the xtrax package directory")
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
  parser.add_argument("--out", type=Path, required=True, help="full inventory JSON")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  modules = load_xtrax(args.xtrax_root.resolve())
  surface = public_surface(modules)
  usage = scan_repo(args.repo.resolve(), modules)

  symbols = []
  for (module, name), kind in sorted(surface.items()):
    areas = usage.files.get((module, name), {area: set() for area in AREAS})
    symbols.append({
      "module": module,
      "name": name,
      "kind": kind,
      **{f"{area}_files": sorted(areas[area]) for area in AREAS},
    })
  internal_used = sorted(
    f"{m}.{n}" for (m, n) in usage.files if (m, n) not in surface
  )
  by_package: dict[str, dict[str, int]] = defaultdict(lambda: {"symbols": 0, "used_src": 0, "used_tests_or_scripts_only": 0, "unused": 0})
  for sym in symbols:
    parts = str(sym["module"]).split(".")
    package = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
    row = by_package[package]
    row["symbols"] += 1
    if sym["src_files"]:
      row["used_src"] += 1
    elif sym["tests_files"] or sym["scripts_files"]:
      row["used_tests_or_scripts_only"] += 1
    else:
      row["unused"] += 1

  def used_in(key: tuple[str, str], area: str) -> bool:
    return bool(usage.files.get(key, {}).get(area))

  positive_ok = used_in(POSITIVE, "src")
  negative_ok = POSITIVE in surface and NEGATIVE in surface and not any(used_in(NEGATIVE, a) for a in AREAS)
  planted_ok = planted_control(modules)
  result = {
    "xtrax_version": xtrax_version(args.xtrax_root.resolve()),
    "n_modules": len(modules),
    "n_symbols": len(symbols),
    "n_used_src": sum(1 for s in symbols if s["src_files"]),
    "n_used_tests_or_scripts_only": sum(1 for s in symbols if not s["src_files"] and (s["tests_files"] or s["scripts_files"])),
    "n_unused": sum(1 for s in symbols if not (s["src_files"] or s["tests_files"] or s["scripts_files"])),
    "n_unresolved_imports": len(usage.unresolved),
    "n_internal_symbols_used": len(internal_used),
    "parse_errors": len(usage.parse_errors),
    "positive_control_ok": positive_ok,
    "negative_control_ok": negative_ok,
    "planted_control_detected": planted_ok,
    "controls_ok": positive_ok and negative_ok and planted_ok,
  }
  full = {
    **result,
    "by_package": dict(sorted(by_package.items())),
    "symbols": symbols,
    "modules_imported": {m: {a: sorted(v[a]) for a in AREAS} for m, v in sorted(usage.modules.items())},
    "internal_symbols_used": internal_used,
    "unresolved_imports": usage.unresolved,
    "parse_error_detail": usage.parse_errors,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(full, indent=1) + "\n", encoding="utf-8")
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(result) + "\n", encoding="utf-8")
  logger.info("%s", json.dumps(result))
  return 0 if result["controls_ok"] else 1


if __name__ == "__main__":
  sys.exit(main())
