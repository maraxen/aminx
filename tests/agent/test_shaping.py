"""JSON shaping of synthetic runner results (G-SHAPE)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from aminx.agent.shaping import (
  DEFAULT_INLINE_CAP,
  default_output_dir,
  new_run_id,
  shape_result,
)
from aminx.host.logit_aggregation import LogitFingerprint
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  SamplingSpecification,
  ScoringSpecification,
)
from aminx.utils.aa_convert import string_to_protein_sequence

_PDB = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
_PROVENANCE = {"sha": "abc", "provenance_source": "test"}
_KNOWN = "ACDE"
_PADDED = "ACDEAA"


def _contains(node: object, target: object) -> bool:
  """Return whether ``target`` appears anywhere in a nested JSON value."""
  if node == target:
    return True
  if isinstance(node, dict):
    return any(_contains(item, target) for item in node.values())
  if isinstance(node, list):
    return any(_contains(item, target) for item in node)
  return False


def _tokens() -> np.ndarray:
  """Return a length-6 MPNN row whose first four letters are ACDE."""
  encoded = np.asarray(string_to_protein_sequence(_KNOWN))
  row = np.zeros((6,), dtype=np.int32)
  row[: encoded.shape[0]] = encoded
  return row


def _fingerprint(fill: float) -> LogitFingerprint:
  """Return a fingerprint whose arrays carry ``fill``."""
  mean = np.full((1, 1, 1, 1, 6, 21), fill, dtype=np.float32)
  line = np.full((1, 1, 1, 1, 6), fill, dtype=np.float32)
  top = np.zeros((1, 1, 1, 1, 6, 1), dtype=np.int32)
  return LogitFingerprint(
    mean_prob=mean,
    entropy_mean=line,
    entropy_std=line,
    top_k=top,
    top_k_prob=np.full((1, 1, 1, 1, 6, 1), fill, dtype=np.float32),
    argmax=np.zeros((1, 1, 1, 1, 6), dtype=np.int32),
    confidence_mean=line,
    confidence_std=line,
    jsd_to_reference=line,
    schedule_variance=line,
  )


def _sample_results(fill: float = 123456.0) -> dict[str, object]:
  """Return a synthetic non-streaming sample result. No model is run."""
  row = _tokens()
  return {
    "sequences": row.reshape(1, 1, 1, 1, 6),
    "logits": np.full((1, 1, 1, 1, 6, 21), fill, dtype=np.float32),
    "logit_fingerprint": _fingerprint(fill),
    "pseudo_perplexity": np.full((1, 1, 1, 1), 2.5, dtype=np.float32),
    "mask": np.ones((1, 1, 1, 1, 6), dtype=np.int32),
    "schema_version": "sampling_v1",
    "metadata": {
      "structure_ids": ["1ubq"],
      "skipped_inputs": [],
      "specification": SamplingSpecification(inputs=[str(_PDB)]),
    },
  }


def _shape(
  tmp_path: Path,
  results: dict[str, object],
  *,
  kind: str = "sample",
  spec: SamplingSpecification | ScoringSpecification | InspectionSpecification | JacobianSpecification | None = None,
  lengths: dict[str, int] | None = None,
  inline_cap: int = DEFAULT_INLINE_CAP,
  run_id: str = "run1",
) -> dict[str, object]:
  """Shape ``results`` against the 1ubq sampling spec unless another is given."""
  chosen = spec if spec is not None else SamplingSpecification(inputs=[str(_PDB)])
  shaped = shape_result(
    kind,
    results,
    spec=chosen,
    lengths=lengths,
    output_dir=tmp_path,
    run_id=run_id,
    inline_cap=inline_cap,
    provenance=_PROVENANCE,
  )
  json.dumps(shaped)
  return shaped


def test_default_output_dir_follows_xdg(tmp_path: Path) -> None:
  """Explicit dir wins, then XDG cache, then ~/.cache."""
  explicit = tmp_path / "explicit"
  xdg = tmp_path / "xdg"
  home = tmp_path / "home"
  assert default_output_dir({"AMINX_AGENT_OUTPUT_DIR": str(explicit), "XDG_CACHE_HOME": str(xdg)}) == explicit
  assert default_output_dir({"XDG_CACHE_HOME": str(xdg), "HOME": str(home)}) == xdg / "aminx" / "agent"
  assert default_output_dir({"HOME": str(home)}) == home / ".cache" / "aminx" / "agent"


def test_new_run_id_is_a_utc_stamp_plus_hex() -> None:
  """Run ids carry a UTC timestamp and 8 hex characters."""
  assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", new_run_id())


def test_sequences_are_decoded_and_trimmed(tmp_path: Path) -> None:
  """Padding positions are dropped when the residue length is known."""
  shaped = _shape(tmp_path, _sample_results(), lengths={"1ubq": 4})
  structure = shaped["structures"][0]
  assert structure["structure_id"] == "1ubq"
  assert structure["trimmed"] is True
  assert structure["length"] == 4
  assert structure["samples"][0]["sequence"] == _KNOWN
  assert structure["samples"][0]["pseudo_perplexity"] == 2.5
  assert structure["samples"][0]["sample_index"] == 0
  assert structure["samples"][0]["noise_index"] == 0
  assert structure["samples"][0]["temperature_index"] == 0


def test_missing_length_does_not_trim(tmp_path: Path) -> None:
  """Without a length the padded sequence is kept and a warning is recorded."""
  shaped = _shape(tmp_path, _sample_results(), lengths=None, run_id="untrimmed")
  structure = shaped["structures"][0]
  assert structure["trimmed"] is False
  assert structure["samples"][0]["sequence"] == _PADDED
  assert any("1ubq" in warning and "not trimmed" in warning for warning in shaped["warnings"])


def test_logit_fingerprint_namedtuple_is_a_dict(tmp_path: Path) -> None:
  """Each fingerprint field is addressable by name."""
  shaped = _shape(tmp_path, _sample_results(), lengths={"1ubq": 4}, run_id="fingerprint")
  fingerprint = shaped["logit_fingerprint"]
  assert isinstance(fingerprint, dict)
  assert set(LogitFingerprint._fields) <= set(fingerprint)
  assert fingerprint["mean_prob"] == _fingerprint(123456.0).mean_prob.tolist()


def test_array_above_the_cap_roundtrips_through_npz(tmp_path: Path) -> None:
  """An array larger than the cap is stored in the npz under a slash-free key."""
  logits = np.full((1, 1, 1, 1, 6, 21), 123456.0, dtype=np.float32)
  shaped = _shape(tmp_path, _sample_results(), lengths={"1ubq": 4}, inline_cap=10, run_id="capped")
  side = shaped["side_file"]
  assert side is not None
  assert "logits" in side["arrays"]
  assert "logit_fingerprint/mean_prob" in side["arrays"]
  loaded = np.load(side["path"])
  np.testing.assert_array_equal(loaded["logits"], logits)
  np.testing.assert_array_equal(loaded["logit_fingerprint__mean_prob"], logits)


def test_inline_cap_zero_side_file_and_huge_cap_inlines(tmp_path: Path) -> None:
  """Cap 0 parks every array; a huge cap inlines the same arrays and writes nothing."""
  results = _sample_results(fill=123456.0)
  parked = _shape(tmp_path, results, lengths={"1ubq": 4}, inline_cap=0, run_id="cap0")
  side = parked["side_file"]
  assert side is not None
  arrays = side["arrays"]
  assert "logits" in arrays
  assert "sequences" in arrays
  assert "mask" in arrays
  assert "pseudo_perplexity" in arrays
  assert "logit_fingerprint/mean_prob" in arrays
  for key in ("logits", "sequences", "mask", "pseudo_perplexity"):
    assert not _contains(parked, np.asarray(results[key]).tolist())
  assert not _contains(parked, np.asarray(results["logit_fingerprint"].mean_prob).tolist())
  loaded = np.load(side["path"])
  np.testing.assert_array_equal(loaded["logits"], results["logits"])

  inlined = _shape(tmp_path, results, lengths={"1ubq": 4}, inline_cap=10**9, run_id="caphuge")
  assert inlined["side_file"] is None
  assert not (tmp_path / "caphuge.npz").exists()
  assert _contains(inlined, np.asarray(results["logits"]).tolist())
  assert _contains(inlined, np.asarray(results["sequences"]).tolist())
  assert _contains(inlined, np.asarray(results["logit_fingerprint"].mean_prob).tolist())


def test_score_pairs_sequences_and_parks_logits(tmp_path: Path) -> None:
  """Score rows follow sequences_to_score and fused ids are copied through."""
  logits = np.full((1, 2, 3, 21), 7.0, dtype=np.float32)
  spec = ScoringSpecification(inputs=[str(_PDB)], sequences_to_score=["AAAA", "CCCC"])
  results = {
    "scores": np.array([[1.0, 2.0]], dtype=np.float32),
    "logits": logits,
    "schema_version": "scoring_v1",
    "metadata": {
      "structure_ids": ["fused_multistate"],
      "fused_structure_ids": ["1ubq", "1mbn"],
      "skipped_inputs": [],
    },
  }
  shaped = _shape(tmp_path, results, kind="score", spec=spec, inline_cap=10, run_id="score")
  assert shaped["structures"] == [
    {
      "structure_id": "fused_multistate",
      "scores": [
        {"sequence": "AAAA", "nll": 1.0},
        {"sequence": "CCCC", "nll": 2.0},
      ],
    },
  ]
  assert shaped["fused_structure_ids"] == ["1ubq", "1mbn"]
  loaded = np.load(shaped["side_file"]["path"])
  np.testing.assert_array_equal(loaded["logits"], logits)


def test_inspect_features_use_structure_keys(tmp_path: Path) -> None:
  """Inspect arrays are keyed by feature and structure id."""
  logits = np.arange(4 * 21, dtype=np.float32).reshape(4, 21)
  distance = np.eye(4, dtype=np.float32)
  similarity = np.arange(4, dtype=np.float32).reshape(2, 2)
  spec = InspectionSpecification(inputs=[str(_PDB)])
  results = {
    "unconditional_logits": [logits],
    "distance_matrix": [distance],
    "cross_input_similarity": similarity,
    "schema_version": "inspection_v1",
    "metadata": {"structure_ids": ["1ubq"], "skipped_inputs": []},
  }
  shaped = _shape(tmp_path, results, kind="inspect", spec=spec, inline_cap=3, run_id="inspect")
  arrays = shaped["side_file"]["arrays"]
  assert "unconditional_logits/1ubq" in arrays
  assert "distance_matrix/1ubq" in arrays
  assert "cross_input_similarity" in arrays
  loaded = np.load(shaped["side_file"]["path"])
  np.testing.assert_array_equal(loaded["unconditional_logits__1ubq"], logits)
  np.testing.assert_array_equal(loaded["cross_input_similarity"], similarity)

  inlined = _shape(tmp_path, results, kind="inspect", spec=spec, inline_cap=10**9, run_id="inspect-inline")
  assert inlined["side_file"] is None
  assert _contains(inlined, logits.tolist())
  assert _contains(inlined, similarity.tolist())


def test_jacobian_arrays_and_streaming_digest(tmp_path: Path) -> None:
  """Jacobian tensors are side-filed; a streaming result keeps path and digest inline."""
  jac = np.ones((2, 21, 2, 21), dtype=np.float32)
  apc = np.eye(2, dtype=np.float32)
  spec = JacobianSpecification(inputs=[str(_PDB)])
  results = {
    "categorical_jacobians": [jac],
    "apc_frobenius_norm": [apc],
    "jacobian_mode": "categorical",
    "n_structures": 1,
    "schema_version": "jacobian_v1",
    "metadata": {"structure_ids": ["1ubq"], "skipped_inputs": []},
  }
  shaped = _shape(tmp_path, results, kind="jacobian", spec=spec, inline_cap=3, run_id="jac")
  assert shaped["jacobian_mode"] == "categorical"
  assert shaped["n_structures"] == 1
  loaded = np.load(shaped["side_file"]["path"])
  np.testing.assert_array_equal(loaded["categorical_jacobians__1ubq"], jac)
  np.testing.assert_array_equal(loaded["apc_frobenius_norm__1ubq"], apc)

  streaming = {
    "categorical_jacobians": [],
    "jacobian_mode": "reverse",
    "n_structures": 0,
    "output_path": "/tmp/out.zarr",
    "output_digest": "abc123",
    "schema_version": "jacobian_v1",
    "metadata": {"structure_ids": [], "skipped_inputs": []},
  }
  streamed = _shape(tmp_path, streaming, kind="jacobian", spec=spec, run_id="jac-stream")
  assert streamed["side_file"] is None
  assert streamed["output_path"] == "/tmp/out.zarr"
  assert streamed["output_digest"] == "abc123"


def test_streaming_sample_reports_the_zarr_path(tmp_path: Path) -> None:
  """A streaming sample result has no decoded structures."""
  results = {
    "output_zarr_path": "/tmp/sample.zarr",
    "schema_version": "sampling_v1",
    "metadata": {"structure_ids": ["1ubq"], "skipped_inputs": [{"path": "skip.pdb"}]},
  }
  shaped = _shape(tmp_path, results, run_id="stream")
  assert shaped["output_zarr_path"] == "/tmp/sample.zarr"
  assert "structures" not in shaped
  assert shaped["skipped_inputs"] == [{"path": "skip.pdb"}]
  assert shaped["side_file"] is None
