"""Shared path and wave rules for the redsox gate scripts.

Wave assignment is by path first: anything outside ``tests/port/`` is
``__nonport__`` so that wave never loads ``tests/port/conftest.py``. Port
ids take the ``port_wave`` marker, and ``port_selftest`` is just that marker
plus ``tests/port/targets/port_selftest.toml``.
"""

from __future__ import annotations

import ast
from pathlib import Path

NONPORT = "__nonport__"


def repo_root(start: Path) -> Path:
  """First ancestor of ``start`` that contains ``pyproject.toml``.

  Never uses the process cwd: the gate is invoked from whatever directory
  bathos happens to launch.
  """
  resolved = start.resolve()
  candidates = (
    (resolved.parent, *resolved.parents) if resolved.is_file() else (resolved, *resolved.parents)
  )
  for directory in candidates:
    if (directory / "pyproject.toml").is_file():
      return directory
  msg = f"no pyproject.toml ancestor of {start}"
  raise RuntimeError(msg)


def target_wave_names(repo: Path) -> list[str]:
  """Basenames of ``tests/port/targets/*.toml``, sorted, as step 1 iterates them."""
  targets = repo / "tests" / "port" / "targets"
  if not targets.is_dir():
    return []
  return sorted(path.stem for path in targets.glob("*.toml"))


def nodeid_path(nodeid: str) -> str:
  """File part of a pytest nodeid, posix, without parameters."""
  return nodeid.split("::", 1)[0].replace("\\", "/")


def declared_wave(repo: Path, nodeid: str) -> str:
  """Wave a nodeid belongs to.

  Ids outside ``tests/port/`` are ``__nonport__`` even if a marker is present.
  Port ids must carry ``port_wave``; a missing marker is a harness error
  rather than a silent drop into ``__nonport__``.
  """
  relative = nodeid_path(nodeid)
  if not relative.startswith("tests/port/"):
    return NONPORT
  source = repo / relative
  if not source.is_file():
    msg = f"port id {nodeid} has no source file at {relative}"
    raise RuntimeError(msg)
  wave = _marker_wave(source, nodeid)
  if wave is None:
    msg = f"port id {nodeid} has no port_wave marker"
    raise RuntimeError(msg)
  return wave


def _port_wave_call(node: ast.AST) -> str | None:
  if not isinstance(node, ast.Call):
    return None
  func = node.func
  if not isinstance(func, ast.Attribute) or func.attr != "port_wave":
    return None
  if not node.args or not isinstance(node.args[0], ast.Constant):
    return None
  value = node.args[0].value
  if not isinstance(value, str) or not value:
    return None
  return value


def _module_wave(tree: ast.AST) -> str | None:
  for node in tree.body if isinstance(tree, ast.Module) else []:
    if not isinstance(node, ast.Assign):
      continue
    if not any(
      isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
    ):
      continue
    found = _wave_in_expr(node.value)
    if found is not None:
      return found
  return None


def _wave_in_expr(expr: ast.AST) -> str | None:
  direct = _port_wave_call(expr)
  if direct is not None:
    return direct
  if isinstance(expr, ast.List | ast.Tuple):
    for element in expr.elts:
      found = _port_wave_call(element)
      if found is not None:
        return found
  return None


def _function_name(nodeid: str) -> str:
  tail = nodeid.rsplit("::", maxsplit=1)[-1]
  return tail.split("[", 1)[0]


def _function_wave(tree: ast.AST, func_name: str) -> str | None:
  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
      continue
    if node.name != func_name:
      continue
    for decorator in node.decorator_list:
      found = _port_wave_call(decorator)
      if found is not None:
        return found
  return None


def _marker_wave(source: Path, nodeid: str) -> str | None:
  tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
  # A function-level marker wins; the module mark covers the current port files,
  # which declare pytestmark once per module.
  on_function = _function_wave(tree, _function_name(nodeid))
  if on_function is not None:
    return on_function
  return _module_wave(tree)
