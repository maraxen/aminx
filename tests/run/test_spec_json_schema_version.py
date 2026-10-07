"""Spec JSON schema_version, unknown-key rejection, and removed-key migration."""

from __future__ import annotations

import pytest

from aminx.host.spec_partition import CAMPAIGN_OWNED_KEYS, campaign_sampling_spec_payload
from aminx.run import SPEC_JSON_SCHEMA_VERSION
from aminx.run.spec_json import (
  _REMOVED_SPEC_KEYS,
  RemovedSpecKey,
  SpecJSONDecodeError,
  run_specification_from_json_dict,
  run_specification_to_json_dict,
)
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

_SPEC_CLASSES = (
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
  InspectionSpecification,
  JacobianSpecification,
)


def _example(cls: type[RunSpecification]) -> RunSpecification:
  """A small spec whose JSON round trip keeps the same Python values."""
  if cls is ScoringSpecification:
    return cls(inputs="test.pdb", sequences_to_score=["ACDEFG"])
  if cls is InspectionSpecification:
    # Defaults for these two are tuples; JSON lists would not compare equal.
    return cls(
      inputs="test.pdb",
      inspection_features=["unconditional_logits"],
      candidate_sequences=[],
    )
  return cls(inputs="test.pdb")


@pytest.mark.parametrize("cls", _SPEC_CLASSES, ids=lambda cls: cls.__name__)
def test_encode_writes_schema_version_and_round_trips(cls: type[RunSpecification]) -> None:
  spec = _example(cls)
  encoded = run_specification_to_json_dict(spec)
  assert encoded["schema_version"] == SPEC_JSON_SCHEMA_VERSION == 1
  assert encoded["_spec_class"] == cls.__name__
  assert run_specification_from_json_dict(encoded) == spec


def test_legacy_document_without_schema_version_decodes() -> None:
  spec = SamplingSpecification(inputs="test.pdb")
  encoded = run_specification_to_json_dict(spec)
  encoded.pop("schema_version")
  assert run_specification_from_json_dict(encoded) == spec


def test_newer_schema_version_raises() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb"))
  encoded["schema_version"] = 2
  with pytest.raises(SpecJSONDecodeError, match="2") as exc_info:
    run_specification_from_json_dict(encoded)
  message = str(exc_info.value)
  assert "1" in message


def test_non_integer_schema_version_raises() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb"))
  encoded["schema_version"] = "1"
  with pytest.raises(SpecJSONDecodeError) as exc_info:
    run_specification_from_json_dict(encoded)
  message = str(exc_info.value)
  assert "1" in message


def test_unknown_key_names_the_key_and_suggests_a_close_match() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb"))
  encoded["temprature"] = 0.5
  with pytest.raises(SpecJSONDecodeError, match="temprature") as exc_info:
    run_specification_from_json_dict(encoded)
  assert "temperature" in str(exc_info.value)


def test_two_unknown_keys_are_both_listed() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb"))
  encoded["temprature"] = 0.5
  encoded["zzzz_not_a_field"] = 1
  with pytest.raises(SpecJSONDecodeError) as exc_info:
    run_specification_from_json_dict(encoded)
  message = str(exc_info.value)
  assert message.index("temprature") < message.index("zzzz_not_a_field")
  assert "temperature" in message


def test_legacy_average_logits_is_dropped_with_a_deprecation_warning() -> None:
  spec = SamplingSpecification(inputs="test.pdb")
  encoded = run_specification_to_json_dict(spec)
  encoded.pop("schema_version")
  with_removed = dict(encoded)
  with_removed["average_logits"] = True
  with pytest.warns(DeprecationWarning, match="average_logits"):
    restored = run_specification_from_json_dict(with_removed)
  assert restored == run_specification_from_json_dict(encoded)


def test_error_policy_raises_with_its_note(monkeypatch: pytest.MonkeyPatch) -> None:
  note = "retired_knob used to change sampling and must not be dropped"
  table = dict(_REMOVED_SPEC_KEYS)
  table["retired_knob"] = RemovedSpecKey(policy="error", note=note)
  monkeypatch.setattr("aminx.run.spec_json._REMOVED_SPEC_KEYS", table)
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb"))
  encoded["retired_knob"] = True
  with pytest.raises(
    SpecJSONDecodeError, match="retired_knob used to change sampling"
  ) as exc_info:
    run_specification_from_json_dict(encoded)
  assert note in str(exc_info.value)


def test_campaign_payload_strips_codec_metadata() -> None:
  spec = SamplingSpecification(inputs="test.pdb")
  encoded = run_specification_to_json_dict(spec)
  owned = {key: encoded[key] for key in CAMPAIGN_OWNED_KEYS}
  payload = campaign_sampling_spec_payload(spec, campaign_owned=owned)
  assert "_spec_class" not in payload
  assert "schema_version" not in payload
  assert set(payload) == set(encoded) - {"_spec_class", "schema_version"}


def test_sampling_specification_rejects_average_logits() -> None:
  with pytest.raises(TypeError, match="average_logits"):
    SamplingSpecification(inputs="test.pdb", average_logits=True)
