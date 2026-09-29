"""Test for Debt #1718: schema version consolidation.

Verifies that all schema version constants are imported from a single
source of truth (schema_versions.py) and that no duplicate definitions exist.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_schema_versions_single_source_of_truth() -> None:
  """Assert all schema versions are the same object across modules."""
  # Import from the authoritative source
  from aminx.host.schema_versions import (
    GRID_SCHEMA_VERSION,
    INSPECTION_SCHEMA_VERSION,
    JACOBIAN_SCHEMA_VERSION,
    SAMPLING_SCHEMA_VERSION,
    SCORING_SCHEMA_VERSION,
  )

  # Import from old locations to verify backward compat
  from aminx.host._sampling_grid_lineage import (
    GRID_SCHEMA_VERSION as GRID_SV_OLD,
  )
  from aminx.host.streaming import (
    GRID_SCHEMA_VERSION as GRID_SV_STREAM,
    SAMPLING_SCHEMA_VERSION as SAMPLING_SV_STREAM,
  )
  from aminx.host.runner import (
    GRID_SCHEMA_VERSION as GRID_SV_RUNNER,
    INSPECTION_SCHEMA_VERSION as INSPECTION_SV_RUNNER,
    JACOBIAN_SCHEMA_VERSION as JACOBIAN_SV_RUNNER,
    SAMPLING_SCHEMA_VERSION as SAMPLING_SV_RUNNER,
    SCORING_SCHEMA_VERSION as SCORING_SV_RUNNER,
  )

  # Verify they are the SAME OBJECTS (not just equal strings)
  assert GRID_SV_OLD is GRID_SCHEMA_VERSION, (
    "GRID_SCHEMA_VERSION in _sampling_grid_lineage should be the same object "
    "as in schema_versions"
  )
  assert GRID_SV_STREAM is GRID_SCHEMA_VERSION, (
    "GRID_SCHEMA_VERSION in streaming should be the same object "
    "as in schema_versions"
  )
  assert SAMPLING_SV_STREAM is SAMPLING_SCHEMA_VERSION, (
    "SAMPLING_SCHEMA_VERSION in streaming should be the same object "
    "as in schema_versions"
  )
  assert GRID_SV_RUNNER is GRID_SCHEMA_VERSION, (
    "GRID_SCHEMA_VERSION in runner should be the same object "
    "as in schema_versions"
  )
  assert SAMPLING_SV_RUNNER is SAMPLING_SCHEMA_VERSION, (
    "SAMPLING_SCHEMA_VERSION in runner should be the same object "
    "as in schema_versions"
  )
  assert SCORING_SV_RUNNER is SCORING_SCHEMA_VERSION, (
    "SCORING_SCHEMA_VERSION in runner should be the same object "
    "as in schema_versions"
  )
  assert INSPECTION_SV_RUNNER is INSPECTION_SCHEMA_VERSION, (
    "INSPECTION_SCHEMA_VERSION in runner should be the same object "
    "as in schema_versions"
  )
  assert JACOBIAN_SV_RUNNER is JACOBIAN_SCHEMA_VERSION, (
    "JACOBIAN_SCHEMA_VERSION in runner should be the same object "
    "as in schema_versions"
  )


def test_no_duplicate_schema_version_definitions() -> None:
  """Grep for duplicates of the consolidated schema versions in src/aminx.

  These specific versions should only be defined in schema_versions.py:
  - GRID_SCHEMA_VERSION
  - SAMPLING_SCHEMA_VERSION
  - SCORING_SCHEMA_VERSION
  - INSPECTION_SCHEMA_VERSION
  - JACOBIAN_SCHEMA_VERSION

  Other *_SCHEMA_VERSION constants (e.g. LOCK_SCHEMA_VERSION in campaign.py)
  are unrelated and outside the scope of Debt #1718.
  """
  src_dir = Path(__file__).parent.parent.parent / "src" / "aminx"
  if not src_dir.is_dir():
    raise RuntimeError(f"src/aminx not found at {src_dir}")

  # Check for each of the consolidated versions
  versions_to_check = [
    "GRID_SCHEMA_VERSION",
    "SAMPLING_SCHEMA_VERSION",
    "SCORING_SCHEMA_VERSION",
    "INSPECTION_SCHEMA_VERSION",
    "JACOBIAN_SCHEMA_VERSION",
  ]

  for version_name in versions_to_check:
    result = subprocess.run(
      [
        "grep",
        "-r",
        f"{version_name}\\s*=\\s*['\"]",
        str(src_dir),
        "--include=*.py",
        "--exclude-dir=__pycache__",
      ],
      capture_output=True,
      text=True,
    )

    # Should only find one match: in schema_versions.py
    violations = []
    for line in result.stdout.split("\n"):
      if not line.strip():
        continue
      if "schema_versions.py" in line:
        continue
      violations.append(line.strip())

    assert not violations, (
      f"Found duplicate definition of {version_name} outside schema_versions.py:\n"
      + "\n".join(violations)
    )
