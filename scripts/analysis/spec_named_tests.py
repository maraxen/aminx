#!/usr/bin/env python3
"""Which tests does the spec NAME, and which of them do not exist?

The sprint's remaining test work has been tracked from memory ("the knob tests
the spec names but that do not exist"). This derives it instead, so the list is
reproducible and its size is a measurement rather than a recollection.

WHAT COUNTS AS EXISTING is the whole difficulty, and getting it wrong inflates
the answer. A bare ``test_[a-z0-9_]+`` sweep of the spec matched against
``^def test_`` reports 35 missing; 11 of those are artefacts:

* **prefix fragments.** The spec writes families, e.g. ``test_knob_semantics_``
  and ``test_family_sink_ragged_``. A trailing underscore means the token is a
  prefix the prose completed in words, not a name.
* **module references.** ``test_knob_superset`` is
  ``tests/knob_gate/test_knob_superset.py``, a file, not a function.
* **indented definitions.** Anchoring on ``^def`` misses every test defined as
  a method.
* **parametrised or suffixed names.** A spec name that is a unique prefix of
  exactly one defined test resolves to it; an ambiguous prefix does not, since
  guessing which of several it meant would be fabrication.

Emits ``{n_named, n_present, n_missing, missing, fragments}`` to
``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger("spec_named_tests")

SPEC = ".praxia/docs/specs/260929_pottsmpnn-lasermpnn-xtrax-composition.md"


def _repo_root(start: Path) -> Path:
  for candidate in (start, *start.parents):
    if (candidate / "pyproject.toml").is_file():
      return candidate
  msg = f"no pyproject.toml above {start}"
  raise SystemExit(msg)


def _defined_tests(tests: Path) -> dict[str, str]:
  """Test function name -> file. No ``^`` anchor: methods are indented."""
  found: dict[str, str] = {}
  for path in sorted(tests.rglob("*.py")):
    if "__pycache__" in path.parts:
      continue
    for match in re.finditer(r"def (test_[a-z0-9_]+)", path.read_text(encoding="utf-8")):
      found.setdefault(match.group(1), str(path))
  return found


def _modules(tests: Path) -> dict[str, str]:
  return {
    path.stem: str(path)
    for path in sorted(tests.rglob("test_*.py"))
    if "__pycache__" not in path.parts
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", type=Path, default=None)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

  repo = args.repo or _repo_root(Path(__file__).resolve())
  spec_path = repo / SPEC
  if not spec_path.is_file():
    msg = f"spec not found: {spec_path}"
    raise SystemExit(msg)

  raw = sorted(set(re.findall(r"\btest_[a-z0-9_]+", spec_path.read_text(encoding="utf-8"))))
  fragments = [name for name in raw if name.endswith("_")]
  named = [name for name in raw if not name.endswith("_")]

  tests = repo / "tests"
  defined = _defined_tests(tests)
  modules = _modules(tests)

  def resolve(name: str) -> str | None:
    if name in defined:
      return defined[name]
    if name in modules:
      return f"{modules[name]} (module)"
    hits = sorted(other for other in defined if other.startswith(name))
    if len(hits) == 1:
      return f"{defined[hits[0]]} (as {hits[0]})"
    return None

  resolved = {name: resolve(name) for name in named}
  missing = sorted(name for name, where in resolved.items() if where is None)

  for name in missing:
    log.info("MISSING %s", name)
  log.info(
    "named=%d present=%d missing=%d fragments=%d",
    len(named), len(named) - len(missing), len(missing), len(fragments),
  )

  results = os.environ.get("BTH_RESULTS_PATH")
  if not results:
    msg = "spec_named_tests requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(results).write_text(
    json.dumps(
      {
        "n_named": len(named),
        "n_present": len(named) - len(missing),
        "n_missing": len(missing),
        "missing": missing,
        "fragments": fragments,
      },
      indent=1,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
