"""AST inventory sweep mapping every public aminx symbol to a P-ID (AC-5).

Walks the public top-level ``def``/``class`` names under the browser-validation
inference-path packages plus ``aminx.run.__all__`` and checks each one is
accounted for in ``browser_validation_paths.json``: either mapped to a P-ID or
explicitly marked ``{"internal": "<reason>"}``. Fails on unmapped symbols,
stale mappings (a JSON entry whose symbol no longer exists), empty internal
reasons, and an internal count above the committed baseline.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import aminx.run as aminx_run

_ROOT_PACKAGES: tuple[str, ...] = (
  "inference",
  "sampling",
  "scoring",
  "model",
  "host",
  "tiling",
  "ebm",
  "potts",
)


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _src_root() -> Path:
  return _repo_root() / "src" / "aminx"


def _mapping_path() -> Path:
  return Path(__file__).with_name("browser_validation_paths.json")


def _baseline_path() -> Path:
  return Path(__file__).with_name("browser_validation_internal_baseline.txt")


def _public_top_level_names(source: str) -> list[str]:
  """Return public (non-underscore) top-level def/class names in `source`."""
  tree = ast.parse(source)
  return [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    and not node.name.startswith("_")
  ]


def discover_symbols(
  *,
  src_root: Path | None = None,
  package_prefix: str = "src/aminx",
  root_packages: tuple[str, ...] = _ROOT_PACKAGES,
  run_all: tuple[str, ...] | None = None,
) -> set[str]:
  """Discover public top-level symbols, keyed as ``<repo-relative path>::<name>``.

  ``aminx.run.__all__`` symbols (no single owning file) are keyed as
  ``aminx.run::<name>``. Parameters are overridable so the control test below
  can point the sweep at a synthetic fixture tree instead of the real source.
  """
  root = src_root if src_root is not None else _src_root()
  symbols: set[str] = set()
  for pkg in root_packages:
    pkg_dir = root / pkg
    if not pkg_dir.is_dir():
      continue
    for f in sorted(pkg_dir.rglob("*.py")):
      if "__pycache__" in f.parts:
        continue
      rel = f.relative_to(root)
      key_prefix = f"{package_prefix}/{rel.as_posix()}"
      for name in _public_top_level_names(f.read_text(encoding="utf-8")):
        symbols.add(f"{key_prefix}::{name}")
  names = run_all if run_all is not None else tuple(aminx_run.__all__)
  for name in names:
    symbols.add(f"aminx.run::{name}")
  return symbols


@dataclass(frozen=True)
class InventoryReport:
  """Findings from comparing discovered symbols against the mapping file."""

  unmapped: tuple[str, ...]
  stale: tuple[str, ...]
  empty_reason: tuple[str, ...]
  internal_count: int


def check_inventory(symbols: set[str], mapping: dict[str, object]) -> InventoryReport:
  """Compare discovered `symbols` against a loaded `browser_validation_paths.json`."""
  mapped = set(mapping)
  unmapped = tuple(sorted(symbols - mapped))
  stale = tuple(sorted(mapped - symbols))
  empty_reason: list[str] = []
  internal_count = 0
  for key, value in mapping.items():
    if isinstance(value, dict):
      internal_count += 1
      reason = value.get("internal", "")
      if not isinstance(reason, str) or not reason.strip():
        empty_reason.append(key)
    elif not isinstance(value, str) or not value.strip():
      empty_reason.append(key)
  return InventoryReport(
    unmapped=unmapped,
    stale=stale,
    empty_reason=tuple(empty_reason),
    internal_count=internal_count,
  )


def _load_mapping() -> dict[str, object]:
  return json.loads(_mapping_path().read_text(encoding="utf-8"))


def _load_baseline() -> int:
  return int(_baseline_path().read_text(encoding="utf-8").strip())


def test_inventory_fully_mapped_with_no_stale_entries() -> None:
  """Every discovered public symbol maps to a P-ID or a non-empty internal reason."""
  report = check_inventory(discover_symbols(), _load_mapping())
  assert not report.unmapped, f"Unmapped public symbols: {list(report.unmapped)}"
  assert not report.stale, f"Stale mappings (symbol no longer exists): {list(report.stale)}"
  assert not report.empty_reason, f"Empty internal reasons: {list(report.empty_reason)}"


def test_internal_count_does_not_exceed_baseline() -> None:
  """The committed internal-symbol count must not grow silently."""
  report = check_inventory(discover_symbols(), _load_mapping())
  baseline = _load_baseline()
  assert report.internal_count <= baseline, (
    f"internal count {report.internal_count} exceeds committed baseline {baseline}; "
    "map the new symbol(s) to a P-ID, or update the baseline deliberately"
  )


def test_control_injected_public_function_is_reported_unmapped(tmp_path: Path) -> None:
  """Control: a fake public function injected into the sweep is reported unmapped.

  Builds a synthetic package tree containing one public function that cannot
  possibly be in the committed mapping file, then asserts the sweep reports it.
  """
  fake_root = tmp_path / "aminx"
  fake_pkg = fake_root / "inference"
  fake_pkg.mkdir(parents=True)
  (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
  (fake_pkg / "injected_fixture_module.py").write_text(
    "def totally_fake_injected_symbol():\n    return 1\n",
    encoding="utf-8",
  )

  symbols = discover_symbols(
    src_root=fake_root,
    root_packages=("inference",),
    run_all=(),
  )
  report = check_inventory(symbols, _load_mapping())

  expected_key = "src/aminx/inference/injected_fixture_module.py::totally_fake_injected_symbol"
  assert expected_key in report.unmapped
