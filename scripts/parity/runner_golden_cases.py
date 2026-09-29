"""Shared in-memory MPNN runner golden cases (T0.5a).

The capture script and ``tests/golden/test_runner_goldens.py`` both import this
module so the matrix cannot drift between bathos and pytest.

Matrix
------
Operations ``sample`` (``batch_size`` 1 and 2), ``score`` (NLL), ``jacobian``,
and ``inspect``, crossed with two packaged checkpoint ids and two fixtures.
Every combination is valid. Zarr rows are not in this matrix: they are T0.5b,
after the external sink ``run_id`` fix. ``score`` and ``inspect`` still raise
``NotImplementedError`` when ``output_h5_path`` is set, so a Zarr row would not
be a faithful freeze of today's in-memory runner.

Checkpoints (the same bare ids the host tests pass to ``load_model``):

- ``proteinmpnn_v_48_020`` — ``load_model`` default (``io/weights.py``).
- ``ligandmpnn_v_32_020_25`` — ``LIGAND_DEFAULT_CHECKPOINT``, used by
  ``tests/model/test_ligandmpnn_equivalence.py`` and
  ``tests/inference/test_side_chain_context.py``.

Fixtures already used by host/runner tests:

- ``tests/data/1ubq.pdb`` — 76 residues, waters only (apo). LigandMPNN runs
  with an empty ligand context.
- ``tests/data/1mbn.pdb`` — 153 residues plus a HEM ligand, so the ligandmpnn
  id actually conditions on ligand atoms. ProteinMPNN parses the protein chain.

``5awl`` (10 residues, waters) and ``3pgk`` (ATP, 415 residues) are in
``tests/data`` but are not the fixtures the host/runner tests load, so they are
not part of the freeze. ``1SMD`` and ``1BC8`` are larger and likewise unused
by those tests.

Fixed settings: ``random_seed=0``, ``num_samples=4``, ``temperature=0.1``.
Score receives the native sequence and a single-site mutant. Jacobian and
inspect use the dataclass defaults (categorical Jacobian with APC; inspect
feature ``unconditional_logits`` only) plus ``batch_size=1`` and no output
path — the minimal valid in-memory specs.

The negative control reruns ``sample`` at ``batch_size=1`` on proteinmpnn/1ubq
with ``random_seed=1``. Sampling consumes the seed; score, jacobian, and
inspect can be invariant to it when backbone noise is off, so the control has
to be a sample case.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from aminx.host.runner import inspect, jacobian, sample, score
from aminx.io.parsing import parse_structure
from aminx.run.spec_json import run_specification_to_json_dict
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "runner_v0"

PROTEIN_CHECKPOINT_ID = "proteinmpnn_v_48_020"
LIGAND_CHECKPOINT_ID = "ligandmpnn_v_32_020_25"
CHECKPOINT_IDS: tuple[str, ...] = (PROTEIN_CHECKPOINT_ID, LIGAND_CHECKPOINT_ID)

FIXTURE_APO = "tests/data/1ubq.pdb"
FIXTURE_LIGAND = "tests/data/1mbn.pdb"
FIXTURES: tuple[str, ...] = (FIXTURE_APO, FIXTURE_LIGAND)

RANDOM_SEED = 0
NEGATIVE_CONTROL_SEED = 1
NUM_SAMPLES = 4
TEMPERATURE = 0.1
PAYLOAD_KEY = "golden_payload"

# Dropped on both sides before metadata equality. Specification fields are
# compared separately, and only for names present in the captured snapshot.
PROVENANCE_TIME_KEYS = frozenset(
  {
    "provenance",
    "created_at",
    "timestamp",
    "captured_at",
    "wall_time",
    "wall_time_s",
    "elapsed",
    "elapsed_s",
    "time",
    "git_sha",
  },
)

Operation = Literal["sample", "score", "jacobian", "inspect"]
ModelFamily = Literal["proteinmpnn", "ligandmpnn"]

# Seed-sensitive sample row. See module docstring.
NEGATIVE_CONTROL_CASE_ID = f"sample_bs1__{PROTEIN_CHECKPOINT_ID}__1ubq"


@dataclass(frozen=True, slots=True)
class GoldenCase:
  """One in-memory runner invocation frozen by T0.5a."""

  case_id: str
  operation: Operation
  checkpoint_id: str
  model_family: ModelFamily
  fixture: str
  batch_size: int

  def to_manifest_entry(self) -> dict[str, str | int]:
    """JSON object stored under ``manifest["cases"]``."""
    return {
      "case_id": self.case_id,
      "operation": self.operation,
      "checkpoint_id": self.checkpoint_id,
      "model_family": self.model_family,
      "fixture": self.fixture,
      "batch_size": self.batch_size,
      "npz": f"{self.case_id}.npz",
    }


def _model_family(checkpoint_id: str) -> ModelFamily:
  if checkpoint_id.startswith("ligandmpnn"):
    return "ligandmpnn"
  return "proteinmpnn"


def _case_id(operation: Operation, checkpoint_id: str, fixture: str, batch_size: int) -> str:
  stem = Path(fixture).stem
  if operation == "sample":
    return f"sample_bs{batch_size}__{checkpoint_id}__{stem}"
  return f"{operation}__{checkpoint_id}__{stem}"


def _one_case(
  operation: Operation,
  checkpoint_id: str,
  family: ModelFamily,
  fixture: str,
  batch_size: int,
) -> GoldenCase:
  return GoldenCase(
    case_id=_case_id(operation, checkpoint_id, fixture, batch_size),
    operation=operation,
    checkpoint_id=checkpoint_id,
    model_family=family,
    fixture=fixture,
    batch_size=batch_size,
  )


def build_cases() -> tuple[GoldenCase, ...]:
  """Return the full in-memory matrix, stable order."""
  cases: list[GoldenCase] = []
  for checkpoint_id in CHECKPOINT_IDS:
    family = _model_family(checkpoint_id)
    for fixture in FIXTURES:
      cases.extend(
        _one_case("sample", checkpoint_id, family, fixture, batch_size) for batch_size in (1, 2)
      )
      cases.extend(
        _one_case(operation, checkpoint_id, family, fixture, 1)
        for operation in ("score", "jacobian", "inspect")
      )
  return tuple(cases)


CASES: tuple[GoldenCase, ...] = build_cases()


def cases_by_id() -> dict[str, GoldenCase]:
  """Map ``case_id`` to its definition."""
  return {case.case_id: case for case in CASES}


def negative_control_case() -> GoldenCase:
  """The single sample row rerun at ``random_seed=1``."""
  found = cases_by_id().get(NEGATIVE_CONTROL_CASE_ID)
  if found is None:
    msg = f"negative-control case {NEGATIVE_CONTROL_CASE_ID} is not in the matrix"
    raise KeyError(msg)
  return found


def current_device_kind() -> str:
  """``platform:device_kind`` of the first JAX device.

  Capture records this string. The pytest skips when the live device differs,
  because runner outputs are not byte-identical across CPU and GPU.
  """
  import jax  # noqa: PLC0415

  device = jax.devices()[0]
  platform = str(device.platform)
  kind = str(getattr(device, "device_kind", platform))
  return f"{platform}:{kind}"


def native_sequence(fixture: str) -> str:
  """Residue sequence of ``fixture`` (repo-relative or absolute), one-letter."""
  from proxide.chem.residues import restype_order_with_x  # noqa: PLC0415

  path = Path(fixture)
  if not path.is_file():
    path = REPO_ROOT / fixture
  structure = parse_structure(path)
  idx_to_aa = {index: aa for aa, index in restype_order_with_x.items()}
  return "".join(idx_to_aa.get(int(index), "X") for index in np.asarray(structure.aatype))


def native_and_mutant(fixture: str) -> tuple[str, str]:
  """Native sequence plus a single substitution at the first residue."""
  native = native_sequence(fixture)
  if not native:
    msg = f"{fixture} parsed to an empty sequence"
    raise ValueError(msg)
  replacement = "A" if native[0] != "A" else "C"
  return native, replacement + native[1:]


def _sample_spec(case: GoldenCase, random_seed: int) -> SamplingSpecification:
  return SamplingSpecification(
    inputs=case.fixture,
    checkpoint_id=case.checkpoint_id,
    model_family=case.model_family,
    random_seed=random_seed,
    batch_size=case.batch_size,
    num_samples=NUM_SAMPLES,
    temperature=TEMPERATURE,
  )


def _score_spec(case: GoldenCase, random_seed: int) -> ScoringSpecification:
  native, mutant = native_and_mutant(case.fixture)
  return ScoringSpecification(
    inputs=case.fixture,
    checkpoint_id=case.checkpoint_id,
    model_family=case.model_family,
    random_seed=random_seed,
    batch_size=case.batch_size,
    sequences_to_score=(native, mutant),
    temperature=TEMPERATURE,
  )


def _jacobian_spec(case: GoldenCase, random_seed: int) -> JacobianSpecification:
  return JacobianSpecification(
    inputs=case.fixture,
    checkpoint_id=case.checkpoint_id,
    model_family=case.model_family,
    random_seed=random_seed,
    batch_size=case.batch_size,
  )


def _inspect_spec(case: GoldenCase, random_seed: int) -> InspectionSpecification:
  return InspectionSpecification(
    inputs=case.fixture,
    checkpoint_id=case.checkpoint_id,
    model_family=case.model_family,
    random_seed=random_seed,
    batch_size=case.batch_size,
  )


def run_case(case: GoldenCase, *, random_seed: int = RANDOM_SEED) -> dict[str, Any]:
  """Run one in-memory runner entry point and return its result dict."""
  if case.operation == "sample":
    return sample(_sample_spec(case, random_seed))
  if case.operation == "score":
    return score(_score_spec(case, random_seed))
  if case.operation == "jacobian":
    return jacobian(_jacobian_spec(case, random_seed))
  if case.operation == "inspect":
    return inspect(_inspect_spec(case, random_seed))
  msg = f"unknown operation {case.operation}"
  raise ValueError(msg)


def _jsonable(value: Any) -> Any:  # noqa: ANN401
  """Convert metadata leaves to JSON values. Unknown types raise."""
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  if isinstance(value, Path):
    return value.as_posix()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in value.items()}
  msg = f"metadata value of type {type(value).__name__} is not JSON-stable"
  raise TypeError(msg)


def metadata_snapshot(metadata: Mapping[str, Any]) -> dict[str, Any]:
  """JSON snapshot of runner metadata, provenance/time keys removed.

  ``specification`` is the ``run_specification_to_json_dict`` payload so later
  code can compare only the field names that existed at capture.
  """
  spec = metadata["specification"]
  snapshot: dict[str, Any] = {
    "specification": run_specification_to_json_dict(spec),
  }
  for key, value in metadata.items():
    if key == "specification" or key in PROVENANCE_TIME_KEYS:
      continue
    snapshot[key] = _jsonable(value)
  return snapshot


def _is_scalar(value: object) -> bool:
  return value is None or isinstance(value, (str, bool, int, float))


def split_result(result: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Split a runner result into dtype-preserving arrays and a JSON payload.

  Nested lists of arrays (jacobian, inspect features) become
  ``{key}__{index}`` entries. Non-array result fields (schema version, mode,
  counts) live under ``payload["scalars"]``.
  """
  arrays: dict[str, np.ndarray] = {}
  scalars: dict[str, Any] = {}
  for key, value in result.items():
    if key == "metadata":
      continue
    if _is_scalar(value):
      scalars[key] = _jsonable(value)
      continue
    _collect_arrays(value, key, arrays)
  payload = {
    "scalars": scalars,
    "metadata": metadata_snapshot(result["metadata"]),
  }
  return arrays, payload


def _collect_arrays(value: Any, prefix: str, out: dict[str, np.ndarray]) -> None:  # noqa: ANN401
  if isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _collect_arrays(item, f"{prefix}__{index}", out)
    return
  if _is_scalar(value) or isinstance(value, (str, bytes, Mapping)):
    msg = f"{prefix} is not an array ({type(value).__name__})"
    raise TypeError(msg)
  array = np.asarray(value)
  if array.dtype == object:
    msg = f"{prefix} is an object array; refusing to golden it"
    raise TypeError(msg)
  out[prefix] = array


def save_case_npz(path: Path, arrays: Mapping[str, np.ndarray], payload: Mapping[str, Any]) -> None:
  """Write one case ``.npz``: every result array, plus a JSON payload."""
  path.parent.mkdir(parents=True, exist_ok=True)
  stored = {key: np.asarray(array) for key, array in arrays.items()}
  stored[PAYLOAD_KEY] = np.asarray(json.dumps(payload, sort_keys=True))
  # numpy's stub types `allow_pickle: bool` ahead of `**kwds: ArrayLike`, and ty
  # binds the unpacked arrays to that bool parameter.
  np.savez(path, **stored)  # ty: ignore[invalid-argument-type]


def load_case_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Load arrays and the JSON payload written by :func:`save_case_npz`."""
  with np.load(path, allow_pickle=False) as archive:
    payload_raw = archive[PAYLOAD_KEY]
    payload = json.loads(str(payload_raw.item()))
    arrays = {key: archive[key] for key in archive.files if key != PAYLOAD_KEY}
  return arrays, payload


def arrays_byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
  """True when dtype, shape, and values all match."""
  return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def compare_arrays(
  captured: Mapping[str, np.ndarray],
  fresh: Mapping[str, np.ndarray],
) -> list[str]:
  """Byte-exact array diff messages. Empty when the two maps match."""
  captured_keys = set(captured)
  fresh_keys = set(fresh)
  messages: list[str] = [
    *(f"missing array {key}" for key in sorted(captured_keys - fresh_keys)),
    *(f"unexpected array {key}" for key in sorted(fresh_keys - captured_keys)),
  ]
  for key in sorted(captured_keys & fresh_keys):
    left = captured[key]
    right = fresh[key]
    if left.dtype != right.dtype:
      messages.append(f"{key} dtype {left.dtype} != {right.dtype}")
    elif left.shape != right.shape:
      messages.append(f"{key} shape {left.shape} != {right.shape}")
    elif not np.array_equal(left, right):
      messages.append(f"{key} values differ")
  return messages


def compare_payload(captured: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
  """Compare scalars and metadata.

  Metadata keys other than ``specification`` must match exactly after
  provenance/time keys are dropped. ``specification`` is compared only on
  field names present in the captured snapshot; fields added after capture
  are ignored.
  """
  messages: list[str] = []
  captured_scalars = captured.get("scalars", {})
  fresh_scalars = fresh.get("scalars", {})
  if captured_scalars != fresh_scalars:
    messages.append(f"scalars differ: captured={captured_scalars!r} fresh={fresh_scalars!r}")

  captured_meta = {
    key: value
    for key, value in captured.get("metadata", {}).items()
    if key not in PROVENANCE_TIME_KEYS
  }
  fresh_meta = {
    key: value
    for key, value in fresh.get("metadata", {}).items()
    if key not in PROVENANCE_TIME_KEYS
  }
  captured_spec = captured_meta.pop("specification", {})
  fresh_spec = fresh_meta.pop("specification", {})
  if set(captured_meta) != set(fresh_meta):
    messages.append(
      f"metadata keys {sorted(captured_meta)} != {sorted(fresh_meta)}",
    )
  else:
    messages.extend(
      f"metadata[{key}] differs"
      for key in sorted(captured_meta)
      if captured_meta[key] != fresh_meta[key]
    )
  if not isinstance(captured_spec, Mapping) or not isinstance(fresh_spec, Mapping):
    messages.append("specification snapshot is not an object")
    return messages
  for name in sorted(captured_spec):
    if name not in fresh_spec:
      messages.append(f"specification field {name} missing on rerun")
    elif captured_spec[name] != fresh_spec[name]:
      messages.append(f"specification field {name} differs")
  return messages


def compare_case(
  captured_arrays: Mapping[str, np.ndarray],
  captured_payload: Mapping[str, Any],
  fresh: Mapping[str, Any],
) -> list[str]:
  """Diff a fresh runner result against a captured case."""
  fresh_arrays, fresh_payload = split_result(fresh)
  # Captured payloads were json.dumps'd. Round-trip the fresh side so tuples
  # and key order cannot disagree with the file.
  fresh_json = json.loads(json.dumps(fresh_payload, sort_keys=True))
  return [
    *compare_arrays(captured_arrays, fresh_arrays),
    *compare_payload(captured_payload, fresh_json),
  ]
