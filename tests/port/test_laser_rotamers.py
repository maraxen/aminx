"""laser_rotamers wave.

One item per (checkpoint, fixture). Ids are read from the dump at collection
time; an absent oracle yields no pairs and the wave skips.

Upstream ``RotamerBuilder.build_rotamers`` is called with the featurized
batch's backbone, chi angles, and ``sequence_indices``
(``dump_laser_oracles.py``). The rotamer npz stores backbone and chi. It does
not store the sequence, so the residue identity is ``featurize(fixture)``'s
``sequence_indices``, the same batch field the dump passed in.

Absent atoms stay NaN in the dumped coordinates (ideal-geometry padding, and
``F.pad(..., torch.nan)`` when nonrotatable hydrogens are added). Those
positions are compared by NaN-mask equality and left out of the angstrom
check. They are not filled with zero.

f64 uses rtol=0, atol=1e-9. f32 uses rtol=0, atol=1e-4. Both bounds are the
spec table's absolute angstrom limits, not a measured deviation.

The dump passes ``add_nonrotatable_hydrogens=model_params["build_hydrogens"]``.
The three LASEr checkpoints set that flag true, so the sealed tensor is the
protonated atom axis. ``build_rotamers`` in aminx has no such flag and returns
the 14-atom axis. This wave compares the full ``sidechain_coords`` array.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_shape_dtype, names, open_dump, x64_context
from port.reference.laser_rotamers.algo import OracleAbsentError, load as load_rotamer_dump

from aminx.families.laser_mpnn.featurize import build_rotamers, featurize

pytestmark = [pytest.mark.port_wave("laser_rotamers"), pytest.mark.parity_heavy]

# Spec table absolute bounds. rtol is 0 because the table states no relative
# band. Not a measured deviation. See targets/laser_rotamers.toml.
_RTOL = {"f64": 0.0, "f32": 0.0}
_ATOL = {"f64": 1e-9, "f32": 1e-4}
_REPO = Path(__file__).resolve().parents[2]
_TRACE_LENGTH = 4


def _laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _fixture_path(name: str) -> Path:
  if name == "4jnj-1_prot":
    return _laser_root() / "example_pdbs" / "4jnj-1_prot.pdb"
  return _REPO / "tests" / "fixtures" / "laser" / f"{name}.pdb"


_SEQUENCE_CACHE: dict[str, np.ndarray] = {}


def _sequence(fixture: str) -> np.ndarray:
  cached = _SEQUENCE_CACHE.get(fixture)
  if cached is not None:
    return cached
  path = _fixture_path(fixture)
  if not path.is_file():
    pytest.skip(f"LASEr fixture absent: {path}")
  features = featurize(path)
  sequence = np.asarray(features.sequence_indices, dtype=np.int64)
  _SEQUENCE_CACHE[fixture] = sequence
  return sequence


def _host_rotamers(
  backbone: np.ndarray,
  chi: np.ndarray,
  sequence: np.ndarray,
) -> np.ndarray:
  """Static-shape host call used by the tier-5 trace budget."""
  return np.asarray(
    build_rotamers(
      np.asarray(backbone, dtype=np.float32),
      np.asarray(chi, dtype=np.float32),
      np.asarray(sequence, dtype=np.int64),
      np.float32,
    ),
    dtype=np.float32,
  )


def _predict(
  data: np.lib.npyio.NpzFile,
  prefix: str,
  fixture: str,
  precision: str,
) -> np.ndarray:
  backbone_key = prefix + "backbone_coords"
  chi_key = prefix + "chi_angles"
  if backbone_key not in data.files or chi_key not in data.files:
    msg = f"laser_rotamers dump has no {backbone_key} or {chi_key}"
    raise AssertionError(msg)
  backbone = np.asarray(data[backbone_key])
  chi = np.asarray(data[chi_key])
  sequence = _sequence(fixture)
  if sequence.shape[0] != backbone.shape[0]:
    msg = (
      f"{prefix}sequence length {sequence.shape[0]} != "
      f"dumped backbone length {backbone.shape[0]}"
    )
    raise AssertionError(msg)
  numpy_dtype = np.float64 if precision == "f64" else np.float32
  return np.asarray(
    build_rotamers(backbone, chi, sequence, numpy_dtype),
    dtype=numpy_dtype,
  )


def _compare(
  got: np.ndarray,
  ref: np.ndarray,
  *,
  rtol: float,
  atol: float,
  numeric: bool,
  label: str,
) -> float:
  """NaN masks must match. Finite atoms are compared; NaNs are not zeroed."""
  if got.shape != ref.shape or got.dtype != ref.dtype:
    assert_shape_dtype(got, ref, label)
  got_absent = np.isnan(got)
  ref_absent = np.isnan(ref)
  if not np.array_equal(got_absent, ref_absent):
    n_bad = int(np.sum(got_absent != ref_absent))
    msg = f"{label}: {n_bad} absent-atom mask entries differ\nMax absolute difference: inf"
    raise AssertionError(msg)
  if not numeric:
    return 0.0
  present = ~ref_absent
  if not bool(np.any(present)):
    return 0.0
  delta = np.abs(got[present].astype(np.float64) - ref[present].astype(np.float64))
  worst = float(np.max(delta)) if delta.size else 0.0
  limit = atol + rtol * np.abs(ref[present].astype(np.float64))
  n_bad = int(np.sum(delta > limit))
  if n_bad:
    msg = (
      f"{label}: {n_bad}/{int(delta.size)} finite atoms exceed "
      f"atol={atol:g}+rtol={rtol:g}\nMax absolute difference: {worst:.6e}"
    )
    raise AssertionError(msg)
  return worst


def _run(
  data: np.lib.npyio.NpzFile,
  precision: str,
  checkpoint: str,
  fixture: str,
  *,
  numeric: bool,
) -> int:
  prefix = f"{checkpoint}__{fixture}__"
  key = prefix + "sidechain_coords"
  with x64_context(precision):
    if key not in data.files:
      msg = f"laser_rotamers dump has no {key}"
      raise AssertionError(msg)
    got = _predict(data, prefix, fixture, precision)
    ref = np.asarray(data[key])
    _compare(
      got,
      ref,
      rtol=_RTOL[precision],
      atol=_ATOL[precision],
      numeric=numeric,
      label=f"laser_rotamers {precision} {key}",
    )
  return 1


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
    checkpoints = names(data["checkpoint_ids"])
    fixtures = names(data["fixture_names"])
  finally:
    data.close()
  return [(checkpoint, fixture) for checkpoint in checkpoints for fixture in fixtures]


_PAIRS = _oracle_pairs()
_PAIR_IDS = [f"{checkpoint}-{fixture}" for checkpoint, fixture in _PAIRS]


def _skip_absent_oracle() -> None:
  if not _PAIRS:
    pytest.skip("laser_rotamers oracle absent")


@pytest.mark.tier_1
@pytest.mark.parametrize("precision", ("f64", "f32"))
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_1_dtype_shape(
  oracle: object,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> None:
  """Shapes, dtypes, and absent-atom NaN masks match."""
  _skip_absent_oracle()
  data = open_dump(oracle, precision)
  try:
    assert _run(data, precision, checkpoint, fixture, numeric=False) > 0
  finally:
    data.close()


@pytest.mark.tier_2
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_2_f64(oracle: object, checkpoint: str, fixture: str) -> None:
  """f64 sidechain coordinates match at rtol=0, atol=1e-9 under x64."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_3_f32(oracle: object, checkpoint: str, fixture: str) -> None:
  """f32 sidechain coordinates match at rtol=0, atol=1e-4.

  The bound is the spec table's absolute angstrom limit. This tier does not
  widen it.
  """
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_5_trace_budget(
  oracle: object,
  max_traces: int,
  checkpoint: str,
  fixture: str,
) -> None:
  """One static residue length traces the host builder at most ``max_traces`` times.

  ``build_rotamers`` is host NumPy, so the traced kernel is a ``pure_callback``
  at a fixed length. Two calls of that shape count as one trace.
  """
  import chex

  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    prefix = f"{checkpoint}__{fixture}__"
    assert prefix + "sidechain_coords" in data.files
  finally:
    data.close()
  dtype = jnp.float32
  backbone = jnp.zeros((_TRACE_LENGTH, 5, 3), dtype=dtype)
  # N, CA, CB form a non-degenerate frame. C and O sit off that plane.
  backbone = backbone.at[:, 1, 0].set(1.0)
  backbone = backbone.at[:, 2, 1].set(1.0)
  backbone = backbone.at[:, 3, 2].set(1.0)
  backbone = backbone.at[:, 4, 0].set(1.0)
  chi = jnp.zeros((_TRACE_LENGTH, 4), dtype=dtype)
  sequence = jnp.zeros((_TRACE_LENGTH,), dtype=jnp.int32)
  result = jax.ShapeDtypeStruct((_TRACE_LENGTH, 14, 3), jnp.float32)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    bb: jax.Array,
    angles: jax.Array,
    seq: jax.Array,
  ) -> jax.Array:
    return jax.pure_callback(_host_rotamers, result, bb, angles, seq)

  compiled = jax.jit(kernel)
  compiled(backbone, chi, sequence)
  compiled(backbone, chi, sequence)
