"""The parity evidence collector must import, and the matrix must name files that exist (#159).

At v0.2.0a1 ``scripts/collect_parity_evidence.py`` imported four module paths that did not exist
in the wheel built from the same tag (three one-word path errors and one module deleted in that
release), and ``parity_matrix.json`` still listed the deleted file as a ``code_path``. Nothing
noticed, because the reference-parity gate only runs on demand, so it silently went unexercised
for months. These tests are cheap and always-on, so the same drift fails CI instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_COLLECTOR = _ROOT / "scripts" / "collect_parity_evidence.py"
_MATRIX = _ROOT / "tests" / "parity" / "parity_matrix.json"


def test_every_declared_code_path_exists() -> None:
  matrix = json.loads(_MATRIX.read_text())
  missing = sorted(
    {
      f"{path_entry['id']}: {code_path}"
      for path_entry in matrix["paths"]
      for code_path in path_entry.get("code_paths", [])
      # A path may be a glob (``model/*.py``); it must match at least one file.
      if not (_ROOT / code_path).exists() and not list(_ROOT.glob(code_path))
    },
  )
  assert not missing, f"parity_matrix.json names code paths that do not exist: {missing}"


def test_collector_imports_against_the_installed_package() -> None:
  """Importing the module executes every top-level ``from aminx... import ...``."""
  name = "collect_parity_evidence_under_test"
  spec = importlib.util.spec_from_file_location(name, _COLLECTOR)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  # ``@dataclass(slots=True)`` looks its own module up in ``sys.modules`` while decorating.
  sys.modules[name] = module
  try:
    spec.loader.exec_module(module)
  except ModuleNotFoundError as exc:
    if exc.name is not None and not exc.name.startswith("aminx"):
      pytest.skip(f"optional dependency of the collector is not installed: {exc.name}")
    raise
  finally:
    sys.modules.pop(name, None)
