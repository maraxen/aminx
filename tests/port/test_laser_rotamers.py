"""laser_rotamers wave.

One item per (checkpoint, fixture). Ids are read from the dump at collection
time; an absent oracle yields no pairs and the wave skips.

The build is a pure function of the dump's backbone, the fixture sequence, and
the dump's chi angles. No decode and no sampling.

NaN masks are compared before finite values. A build that zero-fills absent
atom slots fails that mask check even when the finite atoms look right.

f64 uses rtol=1e-8, atol=1e-11. f32 uses rtol=1e-4, atol=1e-7. Those bounds
are pre-registered. A miss reports the measured deviation and does not widen
them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from port.a1_compare import open_dump
from port.reference.laser_rotamers.algo import OracleAbsentError, load as load_rotamer_dump

from aminx.families.laser_mpnn.featurize import featurize
from aminx.model.laser.rotamers import build_rotamers

pytestmark = [pytest.mark.port_wave("laser_rotamers"), pytest.mark.parity_heavy]

_RTOL = {"f64": 1e-8, "f32": 1e-4}
# House pairing for the declared rtol. Not a measured deviation.
_ATOL = {"f64": 1e-11, "f32": 1e-7}
_REPO = Path(__file__).resolve().parents[2]
_SEQ: dict[str, np.ndarray] = {}


def _laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _fixture_path(name: str) -> Path:
  if name == "4jnj-1_prot":
    return _laser_root() / "example_pdbs" / "4jnj-1_prot.pdb"
  return _REPO / "tests" / "fixtures" / "laser" / f"{name}.pdb"


def _sequence(fixture: str) -> np.ndarray:
  """Sequence is not a rotamer-dump field. The fixture's featurized indices are."""
  cached = _SEQ.get(fixture)
  if cached is not None:
    return cached
  path = _fixture_path(fixture)
  if not path.is_file():
    pytest.skip(f"LASEr fixture absent: {path}")
  features = featurize(path)
  sequence = np.asarray(features.sequence_indices)
  _SEQ[fixture] = sequence
  return sequence


def _names(values: np.ndarray) -> list[str]:
  return [str(item) for item in values.tolist()]


def _oracle_pairs() -> list[tuple[str, str]]:
  """Read checkpoint and fixture ids from the dump at collection time."""
  data: np.lib.npyio.NpzFile | None = None
  for precision in ("f64", "f32"):
    try:
      data = load_rotamer_dump(precision)
    except OracleAbsentError:
      continue
    break
  if data is None:
    return []
  try:
    checkpoints = _names(data["checkpoint_ids"])
    fixtures = _names(data["fixture_names"])
  finally:
    data.close()
  return [(checkpoint, fixture) for checkpoint in checkpoints for fixture in fixtures]


_PAIRS = _oracle_pairs()
_PAIR_IDS = [f"{checkpoint}-{fixture}" for checkpoint, fixture in _PAIRS]


def _skip_absent_oracle() -> None:
  if not _PAIRS:
    pytest.skip("laser_rotamers oracle absent")


def _compare_sidechains(
  got: np.ndarray,
  ref: np.ndarray,
  *,
  rtol: float,
  atol: float,
  numeric: bool,
) -> tuple[int, list[str], float]:
  """NaN masks first, then finite values. Returns comparisons, failures, worst abs."""
  compared = 0
  failures: list[str] = []
  worst = 0.0
  if got.shape != ref.shape:
    failures.append(f"shape {got.shape} vs {ref.shape}")
    return compared, failures, worst
  got_nan = np.isnan(got)
  ref_nan = np.isnan(ref)
  compared += 1
  if not np.array_equal(got_nan, ref_nan):
    n_bad = int(np.sum(got_nan != ref_nan))
    failures.append(f"NaN mask: {n_bad} slots differ")
    worst = max(worst, 1.0)
    # Finite comparison would subtract NaN against a number. Stop after the mask.
    return compared, failures, worst
  if not numeric:
    return compared, failures, worst
  finite = ~ref_nan
  compared += 1
  if not bool(np.any(finite)):
    return compared, failures, worst
  # Index the finite slots. Do not nan_to_num either side: that would hide a
  # mask bug the check above already refused to ignore.
  delta = np.abs(got[finite].astype(np.float64) - ref[finite].astype(np.float64))
  limit = atol + rtol * np.abs(ref[finite].astype(np.float64))
  bad = delta > limit
  n_bad = int(np.sum(bad))
  max_abs = float(np.max(delta)) if delta.size else 0.0
  worst = max(worst, max_abs)
  if n_bad:
    failures.append(
      f"sidechain_coords: {n_bad} finite values exceed atol={atol:g}+rtol={rtol:g}; "
      f"max abs {max_abs:.6e}"
    )
  return compared, failures, worst


def _run(
  data: np.lib.npyio.NpzFile,
  precision: str,
  checkpoint: str,
  fixture: str,
  *,
  numeric: bool,
) -> int:
  dtype = np.float64 if precision == "f64" else np.float32
  rtol = _RTOL[precision]
  atol = _ATOL[precision]
  prefix = f"{checkpoint}__{fixture}__"
  ref = np.asarray(data[prefix + "sidechain_coords"])
  backbone = np.asarray(data[prefix + "backbone_coords"], dtype=dtype)
  chi = np.asarray(data[prefix + "chi_angles"], dtype=dtype)
  sequence = _sequence(fixture)
  if sequence.shape[0] != backbone.shape[0]:
    msg = f"{prefix} sequence length {sequence.shape[0]} != backbone {backbone.shape[0]}"
    raise AssertionError(msg)
  hydrogens = ref.shape[1] == 24
  got = build_rotamers(
    backbone,
    chi,
    sequence,
    dtype=dtype,
    add_nonrotatable_hydrogens=hydrogens,
  )
  got = np.asarray(got)
  if got.dtype != dtype:
    msg = f"{prefix} build dtype {got.dtype} != {dtype}"
    raise AssertionError(msg)
  compared, failures, worst = _compare_sidechains(
    got,
    ref,
    rtol=rtol,
    atol=atol,
    numeric=numeric,
  )
  if failures:
    report = "\n".join(failures)
    msg = f"Max absolute difference: {worst:.6e}\n{report}"
    raise AssertionError(msg)
  return compared


@pytest.mark.tier_1
@pytest.mark.parametrize("precision", ("f64", "f32"))
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_1_dtype_shape(
  oracle: object,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> None:
  """Shapes match and the NaN mask matches, before any finite tolerance."""
  _skip_absent_oracle()
  data = open_dump(oracle, precision)
  try:
    assert _run(data, precision, checkpoint, fixture, numeric=False) > 0
  finally:
    data.close()


@pytest.mark.tier_2
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_2_f64(oracle: object, checkpoint: str, fixture: str) -> None:
  """f64 sidechains match at rtol=1e-8, atol=1e-11."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_3_f32(oracle: object, checkpoint: str, fixture: str) -> None:
  """f32 sidechains match at rtol=1e-4, atol=1e-7. A miss reports the deviation."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_5_still_compares(
  oracle: object,
  max_traces: int,
  checkpoint: str,
  fixture: str,
) -> None:
  """The placement is host NumPy, so the trace budget is the comparison itself."""
  del max_traces
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()
