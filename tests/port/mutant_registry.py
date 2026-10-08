"""Resolve AMINX_PORT_MUTANT to a context manager in tests/port/mutants/."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType

MUTANTS_ROOT = Path(__file__).resolve().parent / "mutants"


def _load_module(path: Path) -> ModuleType:
  spec = importlib.util.spec_from_file_location(f"port_mutant_{path.stem}", path)
  if spec is None or spec.loader is None:
    msg = f"unable to load mutant module {path}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_mutant(name: str) -> AbstractContextManager[None]:
  """Return the context manager produced by the uniquely named mutant function."""
  if not MUTANTS_ROOT.is_dir():
    msg = f"mutant registry directory missing: {MUTANTS_ROOT}"
    raise FileNotFoundError(msg)
  found: list[object] = []
  for path in sorted(MUTANTS_ROOT.glob("*.py")):
    if path.name.startswith("_") or path.name == "__init__.py":
      continue
    module = _load_module(path)
    if hasattr(module, name):
      found.append(getattr(module, name))
  if len(found) != 1:
    msg = f"AMINX_PORT_MUTANT={name!r} matched {len(found)} functions in {MUTANTS_ROOT}"
    raise RuntimeError(msg)
  factory = found[0]
  if not callable(factory):
    msg = f"AMINX_PORT_MUTANT={name!r} is not callable"
    raise TypeError(msg)
  context = factory()
  if not isinstance(context, AbstractContextManager):
    msg = f"AMINX_PORT_MUTANT={name!r} did not return a context manager"
    raise TypeError(msg)
  return context


def iter_mutant_modules() -> Iterator[Path]:
  if not MUTANTS_ROOT.is_dir():
    return
  for path in sorted(MUTANTS_ROOT.glob("*.py")):
    if not path.name.startswith("_"):
      yield path
