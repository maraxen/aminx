# ruff: noqa: S101
"""xtrax deep-import ban for the TID251-exempt FamilyDriver paths.

ADR 260929 exempts the family/model/host-family modules from ruff ``TID251`` so the
PottsModel stageset bans of ADR 260605 stop applying to them. ``TID251`` is one rule
code, so that exemption also drops the unrelated ``xtrax.<sub>.<module>`` deep-import
bans on exactly those files. This test restores that half: the exempt paths must import
xtrax through its public ``__init__`` (``from xtrax.tiling import AxisSpec``), never a
private submodule (``from xtrax.tiling.plan import AxisSpec``).

The banned module list is read from ``pyproject.toml`` so it cannot drift from the ruff
configuration.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The per-file-ignores globs of ADR 260929, plus family_runner.py, which keeps a
# line-level noqa for host.plan but must still obey the xtrax rule.
EXEMPT_GLOBS = (
  "src/aminx/families/potts_mpnn/**/*.py",
  "src/aminx/families/laser_mpnn/**/*.py",
  "src/aminx/model/potts_mpnn/**/*.py",
  "src/aminx/model/laser/**/*.py",
  "src/aminx/host/family_driver.py",
  "src/aminx/host/family_runner.py",
)


def _banned_xtrax_modules() -> frozenset[str]:
  pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
  banned = pyproject["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
  return frozenset(name for name in banned if name.startswith("xtrax."))


def _exempt_files() -> list[Path]:
  found: set[Path] = set()
  for glob in EXEMPT_GLOBS:
    found.update(path for path in REPO_ROOT.glob(glob) if path.is_file())
  return sorted(found)


def _imported_modules(tree: ast.Module) -> set[str]:
  names: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      names.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
      names.add(node.module)
  return names


def test_banned_list_is_not_empty() -> None:
  """A typo in the pyproject path would make this whole test vacuous."""
  assert len(_banned_xtrax_modules()) >= 10


def test_exempt_paths_import_xtrax_from_public_init() -> None:
  """No TID251-exempt file deep-imports a banned xtrax submodule."""
  banned = _banned_xtrax_modules()
  violations = [
    f"{path.relative_to(REPO_ROOT)}: {module}"
    for path in _exempt_files()
    for module in sorted(_imported_modules(ast.parse(path.read_text(encoding="utf-8"))))
    if module in banned
  ]
  assert violations == []


def test_detects_a_planted_deep_import(tmp_path: Path) -> None:
  """Negative control: the checker flags a banned import it is shown directly."""
  banned = _banned_xtrax_modules()
  planted = tmp_path / "planted.py"
  offender = next(iter(sorted(banned)))
  planted.write_text(f"from {offender} import Thing\n", encoding="utf-8")
  modules = _imported_modules(ast.parse(planted.read_text(encoding="utf-8")))
  assert modules & banned == {offender}
