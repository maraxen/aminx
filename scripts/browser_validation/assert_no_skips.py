#!/usr/bin/env python3
"""Fail if a JUnit XML report has any skipped tests, or ran zero tests.

A "no hollow tests" guard for the browser-validation metric gate (T5): a green pytest run
whose suite was silently skipped, or that collected nothing, must not be read as a passing
synthetic-truth check. Both counts are always printed so a failure is diagnosable at a glance.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize_junit(path: Path) -> tuple[int, int]:
  """Return (tests, skipped) totals summed across every <testsuite> in the report."""
  root = ET.parse(path).getroot()
  suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
  tests = sum(int(suite.get("tests", 0)) for suite in suites)
  skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
  return tests, skipped


def main(argv: list[str] | None = None) -> int:
  """Print (tests, skipped) for the given JUnit XML file and return an exit code."""
  args = sys.argv[1:] if argv is None else argv
  if len(args) != 1:
    print("usage: assert_no_skips.py <junit_xml>", file=sys.stderr)
    return 2

  tests, skipped = summarize_junit(Path(args[0]))
  print(f"tests={tests} skipped={skipped}")
  if skipped > 0 or tests == 0:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
