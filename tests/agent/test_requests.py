"""Request normalization for agent specifications."""

from __future__ import annotations

from pathlib import Path

import pytest

from aminx.agent.requests import (
  build_spec,
  spec_from_json,
  spec_to_json_dict,
  structure_lengths,
)
from aminx.host._sampling_helper import _canonical_structure_id
from aminx.potts.spec import PottsRunSpec
from aminx.run.specs import SamplingSpecification

_PDB = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"


def test_unknown_option_names_the_key_and_a_suggestion() -> None:
  """An unknown option is rejected and a close field name is suggested."""
  with pytest.raises(ValueError, match=r"num_sample") as exc_info:
    build_spec("sample", [str(_PDB)], {"num_sample": 2})
  message = str(exc_info.value)
  assert "unknown option" in message
  assert "num_samples" in message


def test_deprecated_option_is_named_deprecated() -> None:
  """Deprecated keys are errors, unlike the runner which drops them silently."""
  with pytest.raises(ValueError, match="deprecated") as exc_info:
    build_spec("sample", [str(_PDB)], {"average_logits": True})
  assert "average_logits" in str(exc_info.value)


def test_every_offending_key_is_listed_together() -> None:
  """Unknown and deprecated keys share one ValueError."""
  with pytest.raises(ValueError, match="average_logits") as exc_info:
    build_spec(
      "sample",
      [str(_PDB)],
      {"average_logits": True, "not_a_real_field": 1},
    )
  message = str(exc_info.value)
  assert "not_a_real_field" in message
  assert "deprecated" in message


def test_unsettable_field_is_rejected() -> None:
  """Non-JSON callables cannot be set through agent options."""
  with pytest.raises(ValueError, match="decoding_order_fn"):
    build_spec("sample", [str(_PDB)], {"decoding_order_fn": lambda _n: None})


def test_missing_input_path_is_rejected(tmp_path: Path) -> None:
  """Missing local paths are named and not fetched."""
  missing = tmp_path / "missing.pdb"
  with pytest.raises(ValueError, match="missing.pdb"):
    build_spec("sample", [str(missing)])


def test_valid_sample_options_build_a_sampling_specification() -> None:
  """Accepted options are stored on the constructed specification."""
  spec = build_spec(
    "sample",
    [str(_PDB)],
    {"num_samples": 3, "temperature": 0.2, "random_seed": 7},
  )
  assert isinstance(spec, SamplingSpecification)
  assert spec.num_samples == 3
  assert spec.temperature == (0.2,)
  assert spec.random_seed == 7
  assert spec.inputs == [str(_PDB.resolve())]


def test_score_without_sequences_to_score_raises() -> None:
  """ScoringSpecification.__post_init__ rejects an empty sequence list."""
  with pytest.raises(ValueError, match="sequences_to_score"):
    build_spec("score", [str(_PDB)])


def test_structure_lengths_uses_the_runner_id() -> None:
  """1ubq's residue count is keyed by the same id the runner would use."""
  spec = build_spec("sample", [str(_PDB)])
  lengths = structure_lengths(spec)
  structure_id = _canonical_structure_id(spec.inputs[0], 0)
  assert structure_id == "1ubq"
  assert lengths == {structure_id: 76}


def test_spec_json_roundtrip_and_potts_kind() -> None:
  """Run specs and Potts specs survive spec_to_json_dict / spec_from_json."""
  spec = build_spec("sample", [str(_PDB)], {"num_samples": 2})
  restored = spec_from_json(spec_to_json_dict(spec))
  assert isinstance(restored, SamplingSpecification)
  assert restored.num_samples == 2

  potts = PottsRunSpec(k_neighbors=8, weights_path="weights.npz")
  payload = spec_to_json_dict(potts)
  assert payload["kind"] == "potts"
  decoded = spec_from_json(payload)
  assert isinstance(decoded, PottsRunSpec)
  assert decoded.k_neighbors == 8
  assert decoded.weights_path == "weights.npz"


def test_shared_file_stem_is_rejected(tmp_path: Path) -> None:
  """Two inputs named model.pdb in different directories cannot share one id."""
  import shutil  # noqa: PLC0415

  from aminx.agent.requests import check_unique_structure_ids  # noqa: PLC0415

  first = tmp_path / "run1" / "model.pdb"
  second = tmp_path / "run2" / "model.pdb"
  for target in (first, second):
    target.parent.mkdir()
    shutil.copy(_PDB, target)
  spec = build_spec("sample", [str(first), str(second)], {})
  with pytest.raises(ValueError, match="'model'"):
    check_unique_structure_ids(spec)
  # Negative control: distinct stems pass.
  distinct = tmp_path / "run2" / "other.pdb"
  shutil.copy(_PDB, distinct)
  check_unique_structure_ids(build_spec("sample", [str(first), str(distinct)], {}))


@pytest.mark.parametrize(("chain", "expected"), [(None, 86), ("A", 76), ("B", 10)])
def test_structure_lengths_match_the_runner_loader(
  tmp_path: Path,
  chain: str | None,
  expected: int,
) -> None:
  """``structure_lengths`` counts what the runner's loader keeps, chain filter included.

  ``fitted_max_length`` sizes the padded shape from these counts, so an
  undercount would truncate residues.
  """
  import numpy as np  # noqa: PLC0415
  from proxide.ops.dataset import create_protein_dataset  # noqa: PLC0415

  ubq = [ln for ln in _PDB.read_text().splitlines() if ln.startswith("ATOM")]
  awl_path = _PDB.parent / "5awl.pdb"
  awl = [ln[:21] + "B" + ln[22:] for ln in awl_path.read_text().splitlines() if ln.startswith("ATOM")]
  two_chain = tmp_path / "two_chain.pdb"
  two_chain.write_text("\n".join([*ubq, "TER", *awl, "TER", "END"]) + "\n")
  spec = build_spec("sample", [str(two_chain)], {} if chain is None else {"chain_id": chain})
  assert structure_lengths(spec) == {"two_chain": expected}
  dataset = create_protein_dataset(
    [str(two_chain)],
    batch_size=1,
    parse_kwargs={
      "chain_id": spec.chain_id,
      "model": spec.model,
      "altloc": spec.altloc,
      "topology": spec.topology,
    },
    max_length=512,
  )
  batch = next(iter(dataset))
  assert int(np.asarray(batch.mask).sum()) == expected
