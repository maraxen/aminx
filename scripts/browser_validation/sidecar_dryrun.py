"""Sidecar dry-run harness (T2, common context "Sidecar dry-run", r1 M4).

For every synthetic case in `--cases`, calls
`bathos.sidecar.evaluate_outcome(bathos.sidecar.parse_sidecar(Path(--sidecar)),
case["result"])` and asserts the returned label equals `case["expect"]`. Runs under
`uv run --frozen --no-sync $BATHOSW` (needs the `bathos` overlay import, not the bare
project venv). Every sidecar this spec creates ships a tracked `<stem>.cases.json` with
>= 1 case per declared outcome, each a full schema-valid result.

Usage: `sidecar_dryrun.py --sidecar S.bth.toml --cases S.cases.json`
Exit 0 if every case's evaluated outcome matches its expectation, else exit 1 with every
mismatch printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_cases(sidecar_path: Path, cases_path: Path) -> list[str]:
  """Return a list of failure descriptions (empty means every case matched)."""
  import bathos.sidecar as bathos_sidecar  # noqa: PLC0415 (only importable under $BATHOSW)

  sidecar = bathos_sidecar.parse_sidecar(sidecar_path)
  with cases_path.open() as fh:
    cases = json.load(fh)

  failures: list[str] = []
  for i, case in enumerate(cases):
    result = case["result"]
    expected = case["expect"]
    actual = bathos_sidecar.evaluate_outcome(sidecar, result)
    if actual != expected:
      name = case.get("name", "<unnamed>")
      failures.append(f"case {i} ({name}): expected {expected!r}, got {actual!r}")
  return failures


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sidecar", required=True, type=Path)
  parser.add_argument("--cases", required=True, type=Path)
  args = parser.parse_args(argv)

  failures = run_cases(args.sidecar, args.cases)
  if failures:
    for failure in failures:
      print(failure, file=sys.stderr)
    print(f"sidecar_dryrun: {len(failures)} case(s) FAILED for {args.sidecar}", file=sys.stderr)
    return 1

  print(f"sidecar_dryrun: all cases OK for {args.sidecar}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
