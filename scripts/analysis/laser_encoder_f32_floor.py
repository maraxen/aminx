"""Measure the float32 parity floor for the ``laser_encoder`` port wave.

WHY THIS EXISTS. ``tests/port/targets/laser_encoder.toml`` declares BOTH of its
bands unmeasured -- ``f64_atol_basis`` and ``f32_atol_basis`` each read
"house-paired and NOT measured" -- and its header says outright:

    atol is the house pairing for the declared rtol (assert_close / potts_head
    style). It is NOT a measured deviation. The orchestrator sets it from the
    graded run.

This is that graded run. The condition that calls for it has fired: the wave's
f32 tier was executed for the first time and failed on every pair, while the
~1e4x tighter f64 tier passed on every pair.

WHY IT IS A NEW FILE AND NOT A FLAG ON THE SHARED INSTRUMENT. Nearly all of the
machinery here is reused verbatim from ``laser_f32_tolerance_floor``, which is
already generic: it resolves ``port.test_<wave>`` and ``port.reference.<wave>``
by name and reads the bands off the test module. Only its ``_WAVES`` tuple
scopes it to two waves. Widening that tuple would have been one token -- but
bathos resolves a sidecar by script stem, so a third wave would have had to ride
``laser_f32_tolerance_floor.bth.toml``, and that file is the PRE-registration of
two runs that have already been graded (e47bfaab, 3af3be02). Editing a
pre-registration after its run is the exact failure the sidecar discipline
exists to prevent. A new stem gets a new pre-registration, and the shared
instrument is left byte-identical so neither graded run's provenance moves.

WHY THE laser_layers BESPOKE MACHINERY IS NOT NEEDED HERE. The sibling script
``laser_layers_f32_floor.py`` had to re-derive and then VERIFY a field-name list
because ``tests/port/test_laser_layers.py`` has no ``_FLOATS``. This wave is not
like that, and the difference was read out of the source rather than assumed:

  - ``tests/port/test_laser_encoder.py:55`` declares ``_FLOATS`` (7 fields).
  - its ``_run`` (``:231``) loops over ``_EXACT + _FLOATS``, but ``_rows_over``
    is reached ONLY through the ``elif numeric:`` branch (``:268``), which the
    ``_EXACT`` fields never enter because they are caught by the preceding
    ``if field in _EXACT`` branch.

So ``_rows_over`` fires exactly once per ``_FLOATS`` entry, in ``_FLOATS``
order, and the shared ``_measure_unit`` already refuses to proceed unless the
recorded call count equals ``len(_FLOATS)``. The positional pairing that had to
be checked element-for-element in the sibling script is structurally sound here.

THE DERIVATION RULE IS THE SHARED ONE, UNCHANGED, and is fixed before the run:
whichever of ``atol``/``rtol`` the measurement shows to bind is moved to
``ceil_one_significant_figure(10 x floor)``; the other is left alone. The
proposed band must admit EVERY element of EVERY field of EVERY pair, must reject
an injected 1e-2 relative defect, and must sit at least 10x below that defect's
absolute size. Nothing here chooses a number.

NOT RATIFIED BY THIS RUN: nothing in ``laser_encoder.toml`` is amended by
running this. ``tests/port/`` is a scoped path under the freeze, so any
amendment rides the single post-merge re-wave.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from laser_f32_tolerance_floor import (  # noqa: E402
  OracleAbsent,
  _add_tests_to_path,
  _grade,
  _repo_root,
  _run_wave,
  _self_test,
)

if TYPE_CHECKING:  # pragma: no cover
  from laser_f32_tolerance_floor import WaveResult

_LOG = logging.getLogger("laser_encoder_f32_floor")

_WAVE = "laser_encoder"

#: The shared instrument whose code is part of this measurement's identity.
_SHARED = _HERE / "laser_f32_tolerance_floor.py"


def _digest(shared: bytes, driver: bytes) -> str:
  """Combine the two halves of this measurement's code identity.

  Taking BYTES rather than reading the files makes this testable on synthetic
  input: the self-test can prove the digest actually depends on each half and
  on their order, which it cannot do while the inputs are two fixed files whose
  contents it has no way to vary.
  """
  digest = hashlib.sha256()
  digest.update(shared)
  digest.update(driver)
  return digest.hexdigest()


def _combined_sha() -> str:
  """Hash BOTH files, so editing either recomputes every cached unit.

  The resume cache key is built from this value. If only this driver were
  hashed, a change to the shared derivation rule would silently reuse units
  measured under the old rule -- which is the one way a resume can corrupt a
  result rather than merely waste time.
  """
  return _digest(_SHARED.read_bytes(), Path(__file__).resolve().read_bytes())


def _driver_self_test() -> dict[str, Any]:
  """Check the code THIS file adds, on synthetic input, touching no oracle.

  The shared instrument's own self-test already covers the derivation, the
  admission forms and the grader's negative controls. What it cannot cover is
  the glue below: the combined hash and the payload assembly. Both are checked
  here so that neither is trusted merely because it is short.
  """
  failed: list[str] = []
  checks = 0

  # 1. The digest is deterministic.
  checks += 1
  if _digest(b"shared", b"driver") != _digest(b"shared", b"driver"):
    failed.append("digest_not_deterministic")

  # 2. It depends on the SHARED half. Without this, a change to the shared
  #    derivation rule would reuse units measured under the old rule.
  checks += 1
  if _digest(b"shared", b"driver") == _digest(b"shared!", b"driver"):
    failed.append("digest_ignores_shared_half")

  # 3. It depends on the DRIVER half.
  checks += 1
  if _digest(b"shared", b"driver") == _digest(b"shared", b"driver!"):
    failed.append("digest_ignores_driver_half")

  # 4. It is order-sensitive, so the pinned order (shared first) is part of the
  #    cache identity rather than an accident of two update() calls.
  checks += 1
  if _digest(b"a", b"b") == _digest(b"b", b"a"):
    failed.append("digest_order_insensitive")

  # 5. The real combined hash is wired to _digest with both files present.
  checks += 1
  if _combined_sha() != _digest(
    _SHARED.read_bytes(), Path(__file__).resolve().read_bytes(),
  ):
    failed.append("combined_sha_not_wired_to_digest")

  # 6/7. THE PAYLOAD ASSEMBLER MUST CARRY THE GRADER'S VERDICT, NOT PRODUCE ONE.
  #    Two fakes with DIFFERENT grades are required, and this is the whole point:
  #    an earlier version of this check used only the passing fake, so an
  #    assembler that hardcoded "pass" -- precisely the dangerous hardcode --
  #    satisfied it. Checking a refusing fake as well means no constant can
  #    satisfy both.
  passing = _FakeResult(wave=_WAVE)
  refusing = _FakeResult(wave=_WAVE, f64_passes=False)
  if _grade(passing) == _grade(refusing):  # pragma: no cover - guards the pair
    failed.append("fakes_do_not_grade_differently")
  checks += 1
  if _payload_for(passing, verdict=_grade(passing))["verdict"] != "pass":
    failed.append("payload_does_not_carry_pass_verdict")
  checks += 1
  if (
    _payload_for(refusing, verdict=_grade(refusing))["verdict"]
    != "defect_not_tolerance"
  ):
    failed.append("payload_does_not_carry_refusal_verdict")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
  }


class _FakeResult:
  """A minimal stand-in carrying only the attributes the grader reads.

  Deliberately NOT a real ``WaveResult``: the point of checks 4 and 5 is that
  the payload assembler and the grader agree on an object they are both handed,
  not that a dataclass can be constructed.
  """

  def __init__(self, *, wave: str, f64_passes: bool = True) -> None:
    self.wave = wave
    self.f64_passes = f64_passes
    self.f32_violated = True
    self.below_control_scale = True
    self.all_admitted = True
    self.control_rejected = True
    self.current_rtol_f32 = 1e-4
    self.current_atol_f32 = 1e-7
    self.current_rtol_f64 = 1e-8
    self.current_atol_f64 = 1e-11
    self.n_pairs = 0
    self.n_fields = 0
    self.floor_abs = 0.0
    self.floor_rel_unconditioned = 0.0
    self.floor_rel_conditioned = 0.0
    self.floor_abs_in_eps = 0.0
    self.binding_term = ""
    self.binding_ref = 0.0
    self.binding_atol_share = 0.0
    self.proposed_rtol = 0.0
    self.proposed_atol = 0.0
    self.n_not_admitted = 0
    self.n_zero_mag_blocked = 0
    self.n_control_admitted = 0
    self.control_scale_ratio = 0.0
    self.control_1e3_margin = 0.0
    self.reused_units: list[str] = []
    self.computed_units: list[str] = []


def _payload_for(result: WaveResult | _FakeResult, *, verdict: str) -> dict[str, Any]:
  """Flatten one wave result. The verdict is CARRIED, never re-derived here."""
  return {
    "verdict": verdict,
    "wave": result.wave,
    "current_rtol_f32": result.current_rtol_f32,
    "current_atol_f32": result.current_atol_f32,
    "current_rtol_f64": result.current_rtol_f64,
    "current_atol_f64": result.current_atol_f64,
    "n_pairs": result.n_pairs,
    "n_fields": result.n_fields,
    "f64_passes": result.f64_passes,
    "f32_violated": result.f32_violated,
    "floor_abs": result.floor_abs,
    "floor_rel_unconditioned": result.floor_rel_unconditioned,
    "floor_rel_conditioned": result.floor_rel_conditioned,
    "floor_abs_in_eps": result.floor_abs_in_eps,
    "binding_term": result.binding_term,
    "binding_ref": result.binding_ref,
    "binding_atol_share": result.binding_atol_share,
    "proposed_rtol": result.proposed_rtol,
    "proposed_atol": result.proposed_atol,
    "all_admitted": result.all_admitted,
    "n_not_admitted": result.n_not_admitted,
    "n_zero_mag_blocked": result.n_zero_mag_blocked,
    "control_rejected": result.control_rejected,
    "n_control_admitted": result.n_control_admitted,
    "below_control_scale": result.below_control_scale,
    "control_scale_ratio": result.control_scale_ratio,
    "control_1e3_margin": result.control_1e3_margin,
    "n_reused": len(result.reused_units),
    "n_computed": len(result.computed_units),
  }


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text)
    _LOG.info("emitted result to BTH_RESULTS_PATH=%s", results)


def main() -> int:
  parser = argparse.ArgumentParser(description="Measure the laser_encoder f32 floor.")
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--self-test-only",
    action="store_true",
    help="run both synthetic instrument checks and stop, touching no oracle",
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
  )
  args.work_dir.mkdir(parents=True, exist_ok=True)

  root = _repo_root()
  _add_tests_to_path(root)
  script_sha = _combined_sha()

  shared = _self_test()
  driver = _driver_self_test()
  passed = bool(shared["self_test_passed"]) and bool(driver["self_test_passed"])
  n_checks = int(shared["self_test_n_checks"]) + int(driver["self_test_n_checks"])
  all_failed = list(shared["self_test_failed"]) + list(driver["self_test_failed"])

  base: dict[str, Any] = {
    "wave": _WAVE,
    "script_sha256": script_sha,
    "shared_instrument_sha256": hashlib.sha256(_SHARED.read_bytes()).hexdigest(),
    "work_dir": str(args.work_dir),
    "self_test_passed": passed,
    "self_test_n_checks": n_checks,
    "self_test_failed": all_failed,
    "oracle_absent": [],
  }

  if not passed:
    _emit(args.out, {**base, "verdict": "instrument_unverified"})
    _LOG.error("instrument unverified: %s", all_failed)
    return 1
  _LOG.info("self-test passed (%d checks)", n_checks)

  if args.self_test_only:
    _emit(args.out, {**base, "verdict": "self_test_only"})
    return 0

  # Import for the record only: the pair count sizes the job and belongs in the
  # payload, and an import failure here is an environment fact worth naming
  # before any measurement is attributed to the port.
  importlib.import_module(f"port.test_{_WAVE}")

  try:
    result = _run_wave(_WAVE, args.work_dir, script_sha)
  except OracleAbsent as exc:
    _LOG.error("%s: %s", _WAVE, exc)
    _emit(args.out, {**base, "verdict": "oracle_absent", "oracle_absent": [_WAVE]})
    return 1

  verdict = _grade(result)
  payload = {**base, **_payload_for(result, verdict=verdict)}
  _emit(args.out, payload)
  _LOG.info(
    "%s: verdict=%s binding=%s proposed_atol=%g proposed_rtol=%g",
    _WAVE,
    verdict,
    result.binding_term,
    result.proposed_atol,
    result.proposed_rtol,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
