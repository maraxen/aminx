"""Find every sealed-oracle array that is STRUCTURALLY EMPTY, across all port waves.

WHY THIS EXISTS. The laser_encoder f32 floor run (graded 91dfbf20) incidentally
revealed that ``lig_vectors`` has shape ``(N, 0, 3)`` for all 63 pairs: the
ligand atoms are present (N = 31..93, ``lig_scalars`` is ``(N, 256)``) but the
vector channel count is zero. One of the seven declared float fields therefore
compares an empty array against an empty array, at BOTH precisions, for EVERY
pair. That assertion cannot fail.

That was found by accident. This run asks the same question on purpose, for
every wave: which comparisons in the port suite are vacuous?

The project's own rule is the reason this matters -- "a positive control that
can only pass is not a check". A tier that reports N passing assertions, some
of which are over zero elements, overstates its own coverage, and the overstatement
is invisible in a green run.

WHAT COUNTS AS EMPTY. An array with any zero-length axis, hence zero elements.
``(93, 0, 3)`` qualifies; ``(93, 256)`` does not. This is a STRUCTURAL property
read off the array header -- it does not depend on the values, on which fixture
was used, or on any tolerance.

WHAT THIS DELIBERATELY DOES NOT MEASURE. Arrays that are non-empty but
identically zero are a DIFFERENT and weaker problem: they still assert that the
port also produces zeros, so they are not vacuous, though they do defeat a
scale-relative band. That case is already measured and recorded elsewhere --
graded run 7384a821 found 126 such entries in laser_layers
(``backbone_frame_vec_input_layer__out__scalars`` and
``backbone_frame_vec_norm__out__scalars``, max_ref_magnitude exactly 0.0 across
all 252 entries at both precisions) -- so it is cited rather than re-derived.
Detecting it would also require decompressing roughly 30 GB of dumps, where
emptiness needs only the headers.

HOW IT STAYS CHEAP. The dumps total ~30 GB and individual files reach 900 MB,
so no array is ever materialised. Each ``.npy`` member's shape is parsed from
its header alone via ``numpy.lib.format.read_array_header``, which consumes only
the leading bytes of the member's stream. The whole audit reads kilobytes.

This run amends nothing and proposes no band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("port_dump_vacuity_audit")

_PRECISIONS = ("f32", "f64")


def _header_shape(stream: Any) -> tuple[int, ...]:
  """Shape of one ``.npy`` member, from its header only.

  Dispatches on the PUBLIC per-version readers rather than numpy's private
  ``_read_array_header``: this runs against whatever numpy the project venv
  resolves, and a private helper that moves would turn a shape audit into a
  traceback. An unknown version raises instead of guessing.
  """
  major, minor = np.lib.format.read_magic(stream)
  if (major, minor) == (1, 0):
    shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(stream)
  elif (major, minor) == (2, 0):
    shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(stream)
  else:
    msg = f"unsupported .npy header version {major}.{minor}"
    raise RuntimeError(msg)
  return tuple(shape)


def _scan_npz(path: Path) -> dict[str, tuple[int, ...]]:
  """Every member's shape, without decompressing a single array."""
  shapes: dict[str, tuple[int, ...]] = {}
  with zipfile.ZipFile(path) as archive:
    for name in archive.namelist():
      if not name.endswith(".npy"):
        continue
      with archive.open(name) as member:
        shapes[name[: -len(".npy")]] = _header_shape(member)
  return shapes


def _is_empty(shape: tuple[int, ...]) -> bool:
  """True when the array holds no elements, i.e. some axis has length 0."""
  return int(np.prod(shape)) == 0 if shape else False


def _grade(payload: dict[str, Any]) -> str:
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload["dumps_absent"]:
    return "dumps_absent"
  if payload["n_arrays"] == 0:
    return "no_arrays_scanned"
  if payload["n_empty"] > 0:
    return "vacuous_fields_present"
  return "no_vacuous_fields"


def _self_test() -> dict[str, Any]:
  """Synthetic checks on a real npz written here, before any dump is opened."""
  failed: list[str] = []
  checks = 0

  with tempfile.TemporaryDirectory() as tmp:
    probe = Path(tmp) / "probe.npz"
    np.savez(
      probe,
      full=np.ones((4, 3)),
      empty_middle=np.ones((5, 0, 3)),
      empty_first=np.ones((0, 7)),
      scalar=np.array(1.0),
      zeros_but_not_empty=np.zeros((2, 2)),
    )
    shapes = _scan_npz(probe)

    # 1. Every member is seen.
    checks += 1
    if set(shapes) != {
      "full", "empty_middle", "empty_first", "scalar", "zeros_but_not_empty",
    }:
      failed.append(f"scan_missed_members:{sorted(shapes)}")

    # 2. Shapes are read correctly from headers alone.
    checks += 1
    if shapes.get("empty_middle") != (5, 0, 3):
      failed.append(f"header_shape_wrong:{shapes.get('empty_middle')}")
    checks += 1
    if shapes.get("full") != (4, 3):
      failed.append(f"header_shape_wrong_full:{shapes.get('full')}")

    # Lookups below are defaulted rather than indexed. A scan that drops a
    # member must leave this function REPORTING a complete failure list, not
    # raising a KeyError partway through -- a self-test that aborts tells you
    # less than one that finishes.
    def shape_of(name: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
      return shapes.get(name, fallback)

    # 3. Emptiness is detected on the axis that is actually zero, wherever it is.
    checks += 1
    if not _is_empty(shape_of("empty_middle", (1,))):
      failed.append("missed_empty_middle_axis")
    checks += 1
    if not _is_empty(shape_of("empty_first", (1,))):
      failed.append("missed_empty_first_axis")

    # 4. NEGATIVE CONTROLS. A populated array must NOT be called empty, and --
    #    the distinction this instrument exists to keep -- neither must an
    #    all-zero array, which is a different and weaker problem.
    checks += 1
    if _is_empty(shape_of("full", (0,))):
      failed.append("false_positive_on_full")
    checks += 1
    if _is_empty(shape_of("zeros_but_not_empty", (0,))):
      failed.append("false_positive_on_all_zero_array")

    # 5. A 0-d scalar holds one element and is not empty.
    checks += 1
    if _is_empty(shape_of("scalar", (0,))):
      failed.append("false_positive_on_scalar")

  # 6-9. Grader controls: it must be able to return each verdict.
  base = {"self_test_passed": True, "dumps_absent": [], "n_arrays": 10}
  checks += 1
  if _grade({**base, "n_empty": 0}) != "no_vacuous_fields":
    failed.append("grader_misses_no_vacuous_fields")
  checks += 1
  if _grade({**base, "n_empty": 1}) != "vacuous_fields_present":
    failed.append("grader_misses_vacuous_fields_present")
  checks += 1
  if _grade({**base, "n_arrays": 0, "n_empty": 0}) != "no_arrays_scanned":
    failed.append("grader_does_not_refuse_on_empty_scan")
  checks += 1
  if _grade({**base, "dumps_absent": ["x"], "n_empty": 0}) != "dumps_absent":
    failed.append("dumps_absent_does_not_outrank")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
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
  parser = argparse.ArgumentParser(description="Audit port dumps for empty arrays.")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--dump-root",
    type=Path,
    action="append",
    required=True,
    help="a dumps/<set> directory holding <wave>/oracle_<precision>.npz (repeatable)",
  )
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  self_test = _self_test()
  base: dict[str, Any] = {
    **self_test,
    "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    "dump_roots": [str(r) for r in args.dump_root],
    "dumps_absent": [],
    "n_arrays": 0,
    "n_empty": 0,
  }
  if not self_test["self_test_passed"]:
    _emit(args.out, {**base, "verdict": "instrument_unverified"})
    _LOG.error("instrument unverified: %s", self_test["self_test_failed"])
    return 1
  _LOG.info("self-test passed (%d checks)", self_test["self_test_n_checks"])
  if args.self_test_only:
    _emit(args.out, {**base, "verdict": "self_test_only"})
    return 0

  absent: list[str] = []
  per_wave: dict[str, Any] = {}
  n_arrays = 0
  n_empty = 0

  for root in args.dump_root:
    if not root.is_dir():
      absent.append(str(root))
      continue
    for wave_dir in sorted(p for p in root.iterdir() if p.is_dir()):
      for precision in _PRECISIONS:
        npz = wave_dir / f"oracle_{precision}.npz"
        if not npz.is_file():
          continue
        shapes = _scan_npz(npz)
        empties = {k: list(v) for k, v in shapes.items() if _is_empty(v)}
        n_arrays += len(shapes)
        n_empty += len(empties)
        slot = per_wave.setdefault(
          f"{root.name}/{wave_dir.name}",
          {"n_arrays": 0, "n_empty": 0, "empty_examples": {}, "empty_fields": {}},
        )
        slot["n_arrays"] += len(shapes)
        slot["n_empty"] += len(empties)
        for key, shape in sorted(empties.items())[:5]:
          slot["empty_examples"].setdefault(f"{precision}:{key}", shape)
        # Field name = the suffix after the last "__", which is how the port
        # tests name what they compare; counting by field shows whether an
        # emptiness is one odd pair or a whole column of the tier.
        for key in empties:
          field = key.rsplit("__", 1)[-1]
          slot["empty_fields"][field] = slot["empty_fields"].get(field, 0) + 1

  payload = {
    **base,
    "dumps_absent": absent,
    "n_arrays": n_arrays,
    "n_empty": n_empty,
    "n_waves_scanned": len(per_wave),
    "n_waves_with_empties": sum(1 for v in per_wave.values() if v["n_empty"]),
    "per_wave": per_wave,
  }
  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  _LOG.info(
    "verdict=%s  %d empty of %d arrays across %d waves",
    payload["verdict"],
    n_empty,
    n_arrays,
    len(per_wave),
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
