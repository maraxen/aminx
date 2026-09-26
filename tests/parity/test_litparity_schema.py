"""Tests for the literature-parity clause/adjudication schema validator (T10 step 5)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scripts.browser_validation.litparity_schema import (
  DESIGNED_DEVIATIONS,
  MANDATORY_ELEMENTS,
  compute_ambiguity,
  main,
  validate,
  validate_adjudication,
  validate_clauses,
)

# One in-scope P-ID (P00-P14, P19, P26) and one out-of-scope P-ID (P15) per
# element, used to build a minimal-but-complete valid `clauses.json`.
_ELEMENT_PATHS = {
  "decoding_order_default": ["P00"],
  "temperature_on_logits_plus_bias": ["P01"],
  "x_omit_at_sampling": ["P02"],
  "omit_as_bias": ["P03"],
  "augment_eps": ["P04"],
  "knn_k_clamp": ["P05"],
  "topk_tie_break": ["P06"],
  "membrane_label_encoding": ["P07"],
  "ligand_atom_context": ["P08"],
  "side_chain_context_toggle": ["P09"],
  "score_only_multi_order_average": ["P10"],
  "packer_chi_sampling": ["P11"],
  "tied_fixed_override": ["P12"],
  "tied_last_member_bias": ["P13"],
  "tied_fusion_weights": ["P14"],
  "tied_intra_group_visibility": ["P19"],
}


def _build_clause(element: str) -> dict[str, Any]:
  paths = _ELEMENT_PATHS[element]
  is_designed = element in DESIGNED_DEVIATIONS
  return {
    "element": element,
    "source": "reference_packet/run.py",
    "code_location": f"src/aminx/inference/{element}.py:1",
    "verdict": "DEVIATION" if is_designed else "MATCH",
    "core": not is_designed,
    "notes": f"designed deviation: {element}" if is_designed else "",
    "paths": paths,
  }


def _valid_clauses_obj() -> dict[str, Any]:
  return {
    "schema_version": 1,
    "reference": "LigandMPNN@26ec57ac",
    "clauses": [_build_clause(e) for e in sorted(MANDATORY_ELEMENTS)],
  }


def _valid_adjudication_obj(clauses: list[dict[str, Any]]) -> dict[str, Any]:
  n_ambiguous, ambiguity_load = compute_ambiguity(clauses)
  return {
    "defects_confirmed": [],
    "adversarial_survived": True,
    "ambiguity_load": ambiguity_load,
    "n_ambiguous": n_ambiguous,
  }


def _write_litparity_dir(
  tmp_path: Path,
  clauses_obj: dict[str, Any],
  adjudication_obj: dict[str, Any] | None,
  *,
  n_recon: int = 3,
) -> Path:
  d = tmp_path / "litparity"
  d.mkdir()
  (d / "clauses.json").write_text(json.dumps(clauses_obj))
  if adjudication_obj is not None:
    (d / "adjudication.json").write_text(json.dumps(adjudication_obj))
  for i in range(n_recon):
    (d / f"recon_{i}.md").write_text(f"# recon {i}\n")
  return d


def test_valid_minimal_full_set_passes(tmp_path: Path) -> None:
  """A minimal but complete clauses.json + adjudication.json + 3 recon files is valid."""
  clauses_obj = _valid_clauses_obj()
  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  d = _write_litparity_dir(tmp_path, clauses_obj, adjudication_obj)

  errors = validate(d)

  assert errors == []


def test_valid_clauses_only_no_adjudication_yet(tmp_path: Path) -> None:
  """adjudication.json is optional -- clauses.json alone (plus recon files) is valid."""
  clauses_obj = _valid_clauses_obj()
  d = _write_litparity_dir(tmp_path, clauses_obj, adjudication_obj=None)

  errors = validate(d)

  assert errors == []


def test_designed_deviation_marked_core_true_is_rejected() -> None:
  """A designed-deviation element with core=True violates the core rule."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "decoding_order_default":
      clause["core"] = True

  errors = validate_clauses(clauses_obj)

  assert any("decoding_order_default" in e and "core=True but expected False" in e for e in errors)


def test_in_scope_clause_marked_core_false_is_rejected() -> None:
  """A non-designed-deviation, in-scope-path clause with core=False violates the rule."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "augment_eps":
      clause["core"] = False

  errors = validate_clauses(clauses_obj)

  assert any("augment_eps" in e and "core=False but expected True" in e for e in errors)


def test_both_core_violations_reported_together() -> None:
  """Both a false-positive and false-negative core flag are each reported."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "decoding_order_default":
      clause["core"] = True
    if clause["element"] == "knn_k_clamp":
      clause["core"] = False

  errors = validate_clauses(clauses_obj)

  core_errors = [e for e in errors if "core=" in e]
  assert len(core_errors) == 2


def test_inconsistent_ambiguity_load_in_adjudication_is_reported() -> None:
  """A hand-set ambiguity_load that disagrees with the recomputed value is rejected."""
  clauses_obj = _valid_clauses_obj()
  # Introduce one AMBIGUOUS clause that lists an in-scope P-ID -> load_bearing.
  clauses_obj["clauses"][0]["verdict"] = "AMBIGUOUS"
  clauses_obj["clauses"][0]["notes"] = "unclear from reference source"

  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  # Deliberately corrupt it to disagree with the recomputed value.
  adjudication_obj["ambiguity_load"] = "none"

  errors = validate_adjudication(adjudication_obj, clauses_obj["clauses"])

  assert any("ambiguity_load" in e and "load_bearing" in e for e in errors)


def test_inconsistent_n_ambiguous_in_adjudication_is_reported() -> None:
  """A hand-set n_ambiguous that disagrees with the recomputed count is rejected."""
  clauses_obj = _valid_clauses_obj()
  clauses_obj["clauses"][0]["verdict"] = "AMBIGUOUS"
  clauses_obj["clauses"][0]["notes"] = "unclear from reference source"

  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  adjudication_obj["n_ambiguous"] = 0

  errors = validate_adjudication(adjudication_obj, clauses_obj["clauses"])

  assert any("n_ambiguous=0" in e for e in errors)


def test_deviation_with_empty_notes_is_rejected() -> None:
  """A DEVIATION clause must carry a non-empty (stripped) reconciliation note."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "x_omit_at_sampling":
      clause["notes"] = "   "

  errors = validate_clauses(clauses_obj)

  assert any("x_omit_at_sampling" in e and "non-empty" in e for e in errors)


def test_missing_mandatory_element_is_rejected() -> None:
  """Dropping a mandatory element from clauses.json is a violation."""
  clauses_obj = _valid_clauses_obj()
  clauses_obj["clauses"] = [
    c for c in clauses_obj["clauses"] if c["element"] != "packer_chi_sampling"
  ]

  errors = validate_clauses(clauses_obj)

  assert any("missing mandatory element 'packer_chi_sampling'" in e for e in errors)


def test_non_bool_core_is_rejected() -> None:
  """core=1 (int, not bool) is rejected even though it is truthy."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "augment_eps":
      clause["core"] = 1

  errors = validate_clauses(clauses_obj)

  assert any("augment_eps" in e and "'core' must be a bool" in e for e in errors)


def test_adversarial_survived_inconsistent_with_core_defect_is_rejected() -> None:
  """adversarial_survived=True with a confirmed core-severity defect is rejected."""
  clauses_obj = _valid_clauses_obj()
  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  adjudication_obj["defects_confirmed"] = [
    {"id": "D1", "paths": ["P02"], "severity": "core"},
  ]
  # adversarial_survived left at True, which is now inconsistent.

  errors = validate_adjudication(adjudication_obj, clauses_obj["clauses"])

  assert any("adversarial_survived=True but expected False" in e for e in errors)


def test_adversarial_survived_false_with_only_minor_defect_is_rejected() -> None:
  """A confirmed non-core defect must NOT flip adversarial_survived to False."""
  clauses_obj = _valid_clauses_obj()
  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  adjudication_obj["defects_confirmed"] = [
    {"id": "D1", "paths": ["P02"], "severity": "minor"},
  ]
  adjudication_obj["adversarial_survived"] = False

  errors = validate_adjudication(adjudication_obj, clauses_obj["clauses"])

  assert any("adversarial_survived=False but expected True" in e for e in errors)


def test_invalid_path_format_is_rejected() -> None:
  """paths entries must match ^P\\d{2}$."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "augment_eps":
      clause["paths"] = ["P4"]

  errors = validate_clauses(clauses_obj)

  assert any("invalid path id" in e for e in errors)


def test_empty_paths_is_rejected() -> None:
  """paths must be a non-empty list."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "augment_eps":
      clause["paths"] = []

  errors = validate_clauses(clauses_obj)

  assert any("augment_eps" in e and "'paths' must be a non-empty list" in e for e in errors)


def test_duplicate_element_is_rejected() -> None:
  """Elements must be unique."""
  clauses_obj = _valid_clauses_obj()
  clauses_obj["clauses"].append(copy.deepcopy(clauses_obj["clauses"][0]))

  errors = validate_clauses(clauses_obj)

  assert any("duplicate element" in e for e in errors)


def test_wrong_defect_severity_is_rejected() -> None:
  """severity must be one of core/major/minor/accepted."""
  clauses_obj = _valid_clauses_obj()
  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  adjudication_obj["defects_confirmed"] = [
    {"id": "D1", "paths": ["P02"], "severity": "critical"},
  ]

  errors = validate_adjudication(adjudication_obj, clauses_obj["clauses"])

  assert any("'severity' must be one of" in e for e in errors)


def test_missing_recon_files_is_rejected(tmp_path: Path) -> None:
  """Exactly 3 recon_*.md files must be present in the directory."""
  clauses_obj = _valid_clauses_obj()
  d = _write_litparity_dir(tmp_path, clauses_obj, adjudication_obj=None, n_recon=2)

  errors = validate(d)

  assert any("expected exactly 3 recon_*.md files" in e for e in errors)


def test_missing_clauses_file_is_rejected(tmp_path: Path) -> None:
  """Missing clauses.json entirely is reported, not silently skipped."""
  d = tmp_path / "litparity"
  d.mkdir()
  for i in range(3):
    (d / f"recon_{i}.md").write_text("x")

  errors = validate(d)

  assert any("clauses.json: file not found" in e for e in errors)


def test_cli_exits_zero_on_valid_dir(tmp_path: Path, capsys: Any) -> None:
  """CLI returns exit code 0 and prints an OK message for a valid directory."""
  clauses_obj = _valid_clauses_obj()
  adjudication_obj = _valid_adjudication_obj(clauses_obj["clauses"])
  d = _write_litparity_dir(tmp_path, clauses_obj, adjudication_obj)

  rc = main([str(d)])

  out = capsys.readouterr().out
  assert rc == 0
  assert "OK: schema valid" in out


def test_cli_exits_one_and_prints_every_violation(tmp_path: Path, capsys: Any) -> None:
  """CLI returns exit code 1 and prints all violations for an invalid directory."""
  clauses_obj = _valid_clauses_obj()
  for clause in clauses_obj["clauses"]:
    if clause["element"] == "decoding_order_default":
      clause["core"] = True
    if clause["element"] == "knn_k_clamp":
      clause["core"] = False
  d = _write_litparity_dir(tmp_path, clauses_obj, adjudication_obj=None)

  rc = main([str(d)])

  out = capsys.readouterr().out
  assert rc == 1
  assert "decoding_order_default" in out
  assert "knn_k_clamp" in out
  assert "violation(s) found" in out


def test_ambiguity_load_rule_covers_every_branch() -> None:
  """R3-C5: none / non_load_bearing (out-of-scope, non-core) / load_bearing (in-scope)."""
  base = [
    {"element": "a", "verdict": "MATCH", "core": True, "paths": ["P07"]},
  ]
  assert compute_ambiguity(base) == (0, "none")
  out_of_scope = [*base, {"element": "b", "verdict": "AMBIGUOUS", "core": False, "paths": ["P20"]}]
  assert compute_ambiguity(out_of_scope) == (1, "non_load_bearing")
  in_scope = [
    *out_of_scope,
    {"element": "c", "verdict": "AMBIGUOUS", "core": False, "paths": ["P09"]},
  ]
  assert compute_ambiguity(in_scope) == (2, "load_bearing")
  core_only = [*base, {"element": "d", "verdict": "AMBIGUOUS", "core": True, "paths": ["P20"]}]
  assert compute_ambiguity(core_only) == (1, "load_bearing")
