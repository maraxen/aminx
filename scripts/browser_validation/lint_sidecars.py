"""Lint `scripts/browser_validation/*.bth.toml` sidecars (T7, step 5).

Checks (Bathos staging's "Outcome labels" paragraph):

1. Every `[outcomes.<label>]` table whose label is NOT in `NON_RESIDUAL_WHITELIST`
   must have `is_residual = true`. bathos treats every non-residual label except
   `marginal`/`error`/`unknown` as pass-direction and picks the first matching
   branch in declaration order, so a mislabeled outcome silently becomes a claimed
   pass. `ctrl_unsized` is whitelisted per this task's step 5 (the calibration-twin
   outcome that legitimately declares `pass` even though `ctrl_unsized` itself is
   residual by declaration order -- see `layer_a_exact_calibrate.bth.toml`,
   Calibration-twin outcomes).
2. No `stage_name = "calibration"` sidecar has a `[differential]` table or a
   `claim_discriminates` key under `[experiment]` (calibration sidecars are never
   the ones proving sensitivity; validate sidecars are).
3. Every `stage_name = "validation"` sidecar with a `[differential]` table has
   `min_effect > 0`, UNLESS `--allow-placeholder` is passed, in which case
   `min_effect == 0.0` is tolerated as the documented sampling placeholder before
   T8's params commit. `--allow-placeholder` must be used ONLY in fixer lint
   gates, never in a smoke path (a real `bth run` always runs this script WITHOUT
   the flag).

Exit 0 if every sidecar passes every check, else exit 1 with every violation
printed (one line each, `<file>: <check> -- <detail>`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Outcome labels bathos allows to be non-residual by declaration (Bathos staging /
# Outcome labels): `pass`, `partial`, `partial_headroom`, `blocked`, `tie_unstable`,
# `not_converted` are the spec's list; `ctrl_unsized` is this task's own addition
# (step 5) for the calibration-twin outcomes.
NON_RESIDUAL_WHITELIST = frozenset(
  {
    "pass",
    "partial",
    "partial_headroom",
    "blocked",
    "tie_unstable",
    "not_converted",
    "ctrl_unsized",
  }
)


def _sidecar_paths(script_dir: Path) -> list[Path]:
  return sorted(script_dir.glob("*.bth.toml"))


def _check_outcome_residuality(path: Path, doc: dict[str, Any]) -> list[str]:
  violations = []
  outcomes = doc.get("outcomes")
  if not isinstance(outcomes, dict):
    return violations
  for label, outcome in outcomes.items():
    if label in NON_RESIDUAL_WHITELIST:
      continue
    if not isinstance(outcome, dict):
      violations.append(f"{path}: outcome-residuality -- [outcomes.{label}] is not a table")
      continue
    if outcome.get("is_residual") is not True:
      violations.append(
        f"{path}: outcome-residuality -- [outcomes.{label}] must have is_residual = true "
        f"(label not in the whitelist {sorted(NON_RESIDUAL_WHITELIST)!r})"
      )
  return violations


def _check_calibration_no_differential(path: Path, doc: dict[str, Any]) -> list[str]:
  violations = []
  experiment = doc.get("experiment", {})
  if experiment.get("stage_name") != "calibration":
    return violations
  if "differential" in doc:
    violations.append(
      f"{path}: calibration-no-differential -- calibration sidecar has a [differential] table"
    )
  if "claim_discriminates" in experiment:
    violations.append(
      f"{path}: calibration-no-differential -- calibration sidecar has "
      "experiment.claim_discriminates"
    )
  return violations


def _check_validation_min_effect(
  path: Path, doc: dict[str, Any], *, allow_placeholder: bool
) -> list[str]:
  violations = []
  experiment = doc.get("experiment", {})
  differential = doc.get("differential")
  if experiment.get("stage_name") != "validation" or not isinstance(differential, dict):
    return violations
  min_effect = differential.get("min_effect")
  if not isinstance(min_effect, (int, float)):
    violations.append(
      f"{path}: validation-min-effect -- [differential].min_effect is missing or non-numeric"
    )
    return violations
  if min_effect > 0:
    return violations
  if allow_placeholder and min_effect == 0.0:
    logger.warning(
      "%s: [differential].min_effect == 0.0 tolerated via --allow-placeholder "
      "(documented sampling placeholder before T8's params commit)",
      path,
    )
    return violations
  violations.append(
    f"{path}: validation-min-effect -- [differential].min_effect must be > 0 "
    f"(got {min_effect!r}; pass --allow-placeholder only in a fixer lint gate, never a smoke path)"
  )
  return violations


def lint_sidecar(path: Path, *, allow_placeholder: bool) -> list[str]:
  with path.open("rb") as fh:
    doc = tomllib.load(fh)
  violations: list[str] = []
  violations.extend(_check_outcome_residuality(path, doc))
  violations.extend(_check_calibration_no_differential(path, doc))
  violations.extend(_check_validation_min_effect(path, doc, allow_placeholder=allow_placeholder))
  return violations


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--allow-placeholder",
    action="store_true",
    help="Tolerate [differential].min_effect == 0.0 on validation sidecars. Fixer lint gates only.",
  )
  parser.add_argument(
    "--script-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="Directory to glob *.bth.toml from (default: this script's own directory).",
  )
  args = parser.parse_args(argv)

  sidecars = _sidecar_paths(args.script_dir)
  if not sidecars:
    logger.warning("no *.bth.toml sidecars found under %s", args.script_dir)
    return 0

  all_violations: list[str] = []
  for path in sidecars:
    all_violations.extend(lint_sidecar(path, allow_placeholder=args.allow_placeholder))

  if all_violations:
    for violation in all_violations:
      print(violation, file=sys.stderr)
    print(
      f"lint_sidecars: {len(all_violations)} violation(s) across {len(sidecars)} sidecar(s)",
      file=sys.stderr,
    )
    return 1

  logger.info("lint_sidecars: %d sidecar(s) OK", len(sidecars))
  return 0


if __name__ == "__main__":
  sys.exit(main())
