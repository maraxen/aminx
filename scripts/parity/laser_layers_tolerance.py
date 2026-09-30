#!/usr/bin/env python3
"""Is the laser_layers f64 bound satisfiable as pre-registered?

Spec section 7.2 pre-registers ``laser_layers`` as ``rtol=1e-9 / rtol=1e-5`` with
no ``atol``, and the B2 wave implements that literally as ``err > rtol * |ref|``.
This script asks whether that bound is satisfiable by ANY implementation, using
the sealed oracle alone -- no aminx code is involved, so the answer cannot be
confounded by a port defect.

The argument is arithmetic, not statistical: where a reference element is exactly
0.0, a pure relative bound admits an error of exactly 0, i.e. it demands
bit-exactness of a floating-point result. Counting those elements settles it.

Both controls run on the oracle against itself, so the instrument is tested
rather than assumed:

* positive control -- an identical copy must be accepted everywhere. If the
  comparison rejects the oracle against itself, the instrument is broken.
* negative control -- a 1e-6 RELATIVE perturbation must still be rejected on
  most elements under the proposed mixed bound. A bound that accepts that is too
  loose to detect a real port defect, and the amendment must not be made.

Usage (graded):
  bth run --project-slug aminx -- uv run --no-sync python3 \
      scripts/parity/laser_layers_tolerance.py --payload-out <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("laser_layers_tolerance")

RTOL_F64 = 1e-9
PROPOSED_ATOL = 1e-12
PERTURBATION = 1e-6
CONTROL_REJECT_FLOOR = 0.9
NEAR_ZERO = 1e-7
# The key the wave still fails on after the b5c9e274 pre-hook dump fix.
LAYER = "protein_encoder_layers_0_hetgat"
LEAF = "out__0__scalars"


def _oracle_root() -> Path:
  raw = os.environ.get("AMINX_LASER_ORACLE_DIR", "~/projects/aminx-oracles/dumps/laser")
  return Path(raw).expanduser()


def _names(raw: np.ndarray) -> list[str]:
  return [str(value) for value in np.atleast_1d(raw)]


def _pure_rtol_rejects(got: np.ndarray, ref: np.ndarray) -> int:
  return int(np.sum(np.abs(got - ref) > RTOL_F64 * np.abs(ref)))


def _mixed_rejects(got: np.ndarray, ref: np.ndarray, atol: float) -> int:
  return int(np.sum(np.abs(got - ref) > atol + RTOL_F64 * np.abs(ref)))


def _measure(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
  n_zero = 0
  n_near_zero = 0
  n_total = 0
  identity_rejected = 0
  perturb_rejected = 0
  pure_rtol_rejected = 0
  mixed_rejected = 0
  keys_seen = 0
  worst_ref_of_zero_key = ""
  rng = np.random.default_rng(0)
  del rng  # perturbation is deterministic; kept explicit that nothing is sampled

  for checkpoint in _names(data["checkpoint_ids"]):
    for fixture in _names(data["fixture_names"]):
      key = f"{checkpoint}__{fixture}__layer__{LAYER}__{LEAF}"
      if key not in data.files:
        continue
      keys_seen += 1
      ref = np.asarray(data[key]).astype(np.float64)
      n_total += int(ref.size)
      zeros = int(np.sum(ref == 0.0))
      n_zero += zeros
      if zeros and not worst_ref_of_zero_key:
        worst_ref_of_zero_key = key
      n_near_zero += int(np.sum((ref != 0.0) & (np.abs(ref) < NEAR_ZERO)))

      # Positive control: the oracle against itself.
      identity_rejected += _mixed_rejects(ref, ref, PROPOSED_ATOL)
      # Negative control: a 1e-6 relative perturbation of the oracle.
      perturbed = ref * (1.0 + PERTURBATION)
      perturb_rejected += _mixed_rejects(perturbed, ref, PROPOSED_ATOL)
      # How the two bounds treat a round-off-scale error, the thing actually at issue.
      round_off = ref + np.sign(ref) * 7.2e-15
      pure_rtol_rejected += _pure_rtol_rejects(round_off, ref)
      mixed_rejected += _mixed_rejects(round_off, ref, PROPOSED_ATOL)

  return {
    "keys_compared": keys_seen,
    "elements_total": n_total,
    "elements_exactly_zero": n_zero,
    "elements_near_zero": n_near_zero,
    "identity_control_rejected": identity_rejected,
    "perturb_control_rejected": perturb_rejected,
    "perturb_control_rejected_frac": (perturb_rejected / n_total) if n_total else 0.0,
    "roundoff_rejected_pure_rtol": pure_rtol_rejected,
    "roundoff_rejected_mixed": mixed_rejected,
    "rtol": RTOL_F64,
    "proposed_atol": PROPOSED_ATOL,
    "first_key_with_exact_zero": worst_ref_of_zero_key,
  }


def _band(m: dict[str, Any]) -> str:
  if m["keys_compared"] == 0 or m["elements_total"] == 0:
    return "fail"
  # Instrument checks first: a broken instrument invalidates the claim either way.
  if m["identity_control_rejected"] != 0:
    return "fail"
  if m["perturb_control_rejected_frac"] < CONTROL_REJECT_FLOOR:
    return "fail"
  # The substantive claim: pure rtol rejects round-off, the mixed bound does not.
  if m["elements_exactly_zero"] > 0 and m["roundoff_rejected_pure_rtol"] > 0:
    return "pass" if m["roundoff_rejected_mixed"] == 0 else "inconclusive"
  if m["elements_near_zero"] > 0 and m["roundoff_rejected_pure_rtol"] > 0:
    return "inconclusive"
  return "fail"


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--payload-out", type=Path, default=None)
  args = parser.parse_args()

  path = _oracle_root() / "laser_layers" / "oracle_f64.npz"
  if not path.is_file():
    msg = f"oracle dump absent: {path}"
    raise SystemExit(msg)
  data = np.load(path)
  try:
    measured = _measure(data)
  finally:
    data.close()
  measured["band"] = _band(measured)
  logger.info("band=%s", measured["band"])
  for name, value in measured.items():
    logger.info("  %s = %s", name, value)
  payload = json.dumps(measured, indent=1, sort_keys=True)
  if args.payload_out is not None:
    args.payload_out.parent.mkdir(parents=True, exist_ok=True)
    args.payload_out.write_text(payload + "\n", encoding="utf-8")
  else:
    print(payload)


if __name__ == "__main__":
  main()
