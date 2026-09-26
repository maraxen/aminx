"""Schema + core-flag + ambiguity-load validator for the literature-parity protocol (T10 step 5).

Validates a ``litparity/`` directory produced by the orchestrator-driven Phase 1-4
literature-parity protocol (spec "Grade inputs", R2-C8/R3-C11/R3-C5):

- ``clauses.json`` (always required): required keys, the verdict vocabulary, every
  mandatory clause present, ``core`` present and boolean on every clause, a
  non-empty reconciliation note on every ``DEVIATION``, and a non-empty list of
  ``P``-ID ``paths`` on every clause. It also mechanically ENFORCES
  ``core == (paths ∩ IN_SCOPE_LAYER_A != {} and element not in DESIGNED_DEVIATIONS)``
  -- the six-item ``DESIGNED_DEVIATIONS`` list and ``IN_SCOPE_LAYER_A`` set are
  hard-coded from the spec's "Grade inputs" section and must not be re-derived
  per run.
- ``adjudication.json`` (validated only if present -- it is written in a later
  phase): ``severity`` on every confirmed defect, ``paths`` on every defect, and
  ``ambiguity_load``/``n_ambiguous`` recomputed from ``clauses.json`` (R3-C5) and
  ``adversarial_survived`` recomputed as "no confirmed core-severity defect".
- Exactly 3 ``recon_*.md`` files in the directory.

CLI: ``python litparity_schema.py <litparity_dir>`` -- exit 0 if valid, exit 1 with
every violation printed otherwise. Importable: ``validate(dir) -> list[str]``, plus
the pure helpers ``validate_clauses``, ``validate_adjudication``, and
``compute_ambiguity``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

VERDICTS = {"MATCH", "DEVIATION", "MISSING", "AMBIGUOUS"}
SEVERITIES = {"core", "major", "minor", "accepted"}
AMBIGUITY_LOADS = {"none", "non_load_bearing", "load_bearing"}

MANDATORY_ELEMENTS = {
  "decoding_order_default",
  "temperature_on_logits_plus_bias",
  "x_omit_at_sampling",
  "omit_as_bias",
  "augment_eps",
  "knn_k_clamp",
  "topk_tie_break",
  "membrane_label_encoding",
  "ligand_atom_context",
  "side_chain_context_toggle",
  "score_only_multi_order_average",
  "packer_chi_sampling",
  "tied_fixed_override",
  "tied_last_member_bias",
  "tied_fusion_weights",
  "tied_intra_group_visibility",
}

# Hard-coded per spec "Grade inputs" (R3-C11) -- never re-derive at runtime.
IN_SCOPE_LAYER_A = {f"P{i:02d}" for i in range(15)} | {"P19", "P26"}

DESIGNED_DEVIATIONS = {
  "x_omit_at_sampling",
  "decoding_order_default",
  "tied_fixed_override",
  "tied_last_member_bias",
  "tied_fusion_weights",
  "tied_intra_group_visibility",
}

_PATH_RE = re.compile(r"^P\d{2}$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _expected_paths(paths: object) -> set[str]:
  """Return the subset of ``paths`` that are well-formed P-ID strings."""
  if not isinstance(paths, list):
    return set()
  return {p for p in paths if isinstance(p, str) and _PATH_RE.match(p)}


def validate_clauses(obj: Any) -> list[str]:  # noqa: ANN401
  """Validate a parsed ``clauses.json`` object; return a list of violation strings."""
  errors: list[str] = []
  if not isinstance(obj, dict):
    return ["clauses.json: root must be an object"]

  if obj.get("schema_version") != 1:
    errors.append(f"clauses.json: schema_version must be 1, got {obj.get('schema_version')!r}")

  reference = obj.get("reference")
  if not isinstance(reference, str) or not reference:
    errors.append("clauses.json: 'reference' must be a non-empty string")

  clauses = obj.get("clauses")
  if not isinstance(clauses, list):
    errors.append("clauses.json: 'clauses' must be a list")
    return errors

  seen_elements: set[str] = set()

  for i, clause in enumerate(clauses):
    prefix = f"clauses[{i}]"
    if not isinstance(clause, dict):
      errors.append(f"{prefix}: must be an object")
      continue

    element = clause.get("element")
    if not isinstance(element, str) or not element:
      errors.append(f"{prefix}: 'element' must be a non-empty string")
      element = None
    elif not _SNAKE_RE.match(element):
      errors.append(f"{prefix} ({element}): 'element' must be snake_case")

    label = element if element is not None else f"index {i}"

    if element is not None:
      if element in seen_elements:
        errors.append(f"{prefix} ({label}): duplicate element '{element}'")
      seen_elements.add(element)

    source = clause.get("source")
    if not isinstance(source, str) or not source:
      errors.append(f"{prefix} ({label}): 'source' must be a non-empty string")

    code_location = clause.get("code_location")
    if not isinstance(code_location, str) or not code_location:
      errors.append(f"{prefix} ({label}): 'code_location' must be a non-empty string")

    verdict = clause.get("verdict")
    if verdict not in VERDICTS:
      errors.append(
        f"{prefix} ({label}): 'verdict' must be one of {sorted(VERDICTS)}, got {verdict!r}"
      )

    core = clause.get("core")
    if not isinstance(core, bool):
      errors.append(f"{prefix} ({label}): 'core' must be a bool, got {core!r}")

    notes = clause.get("notes")
    if not isinstance(notes, str):
      errors.append(f"{prefix} ({label}): 'notes' must be a string, got {notes!r}")
      notes = ""
    if verdict == "DEVIATION" and not notes.strip():
      errors.append(f"{prefix} ({label}): DEVIATION clause must have a non-empty 'notes'")

    paths = clause.get("paths")
    if not isinstance(paths, list) or not paths:
      errors.append(f"{prefix} ({label}): 'paths' must be a non-empty list")
    else:
      for p in paths:
        if not isinstance(p, str) or not _PATH_RE.match(p):
          errors.append(f"{prefix} ({label}): invalid path id {p!r}, must match ^P\\d{{2}}$")

    if isinstance(core, bool) and element is not None:
      path_set = _expected_paths(paths)
      expected_core = bool(path_set & IN_SCOPE_LAYER_A) and element not in DESIGNED_DEVIATIONS
      if core != expected_core:
        errors.append(
          f"{prefix} ({label}): core={core} but expected {expected_core} "
          f"(paths={sorted(path_set)}, designed_deviation={element in DESIGNED_DEVIATIONS})"
        )

  missing = MANDATORY_ELEMENTS - seen_elements
  for m in sorted(missing):
    errors.append(f"clauses.json: missing mandatory element '{m}'")

  return errors


def compute_ambiguity(clauses: list[dict[str, Any]]) -> tuple[int, str]:
  """Recompute ``(n_ambiguous, ambiguity_load)`` from a parsed clause list (R3-C5).

  ``ambiguity_load`` is 'load_bearing' if any AMBIGUOUS clause is ``core = true``
  or lists an in-scope layer-(a) P-ID; 'non_load_bearing' if at least one AMBIGUOUS
  clause exists and none is load-bearing; else 'none'.
  """
  ambiguous = [c for c in clauses if isinstance(c, dict) and c.get("verdict") == "AMBIGUOUS"]
  n_ambiguous = len(ambiguous)
  if n_ambiguous == 0:
    return 0, "none"

  load_bearing = any(
    bool(c.get("core")) or bool(_expected_paths(c.get("paths")) & IN_SCOPE_LAYER_A)
    for c in ambiguous
  )
  return n_ambiguous, "load_bearing" if load_bearing else "non_load_bearing"


def validate_adjudication(adj: Any, clauses: list[dict[str, Any]]) -> list[str]:  # noqa: ANN401
  """Validate a parsed ``adjudication.json`` object against the given clause list."""
  errors: list[str] = []
  if not isinstance(adj, dict):
    return ["adjudication.json: root must be an object"]

  defects = adj.get("defects_confirmed")
  if not isinstance(defects, list):
    errors.append("adjudication.json: 'defects_confirmed' must be a list")
    defects = []

  any_core_defect = False
  for i, d in enumerate(defects):
    prefix = f"adjudication.defects_confirmed[{i}]"
    if not isinstance(d, dict):
      errors.append(f"{prefix}: must be an object")
      continue

    d_id = d.get("id")
    if not isinstance(d_id, str) or not d_id:
      errors.append(f"{prefix}: 'id' must be a non-empty string")
    label = d_id if isinstance(d_id, str) and d_id else f"index {i}"

    paths = d.get("paths")
    if (
      not isinstance(paths, list)
      or not paths
      or not all(isinstance(p, str) and _PATH_RE.match(p) for p in paths)
    ):
      errors.append(f"{prefix} ({label}): 'paths' must be a non-empty list of P-IDs")

    severity = d.get("severity")
    if severity not in SEVERITIES:
      errors.append(
        f"{prefix} ({label}): 'severity' must be one of {sorted(SEVERITIES)}, got {severity!r}"
      )
    elif severity == "core":
      any_core_defect = True

  adversarial_survived = adj.get("adversarial_survived")
  if not isinstance(adversarial_survived, bool):
    errors.append("adjudication.json: 'adversarial_survived' must be a bool")
  else:
    expected_survived = not any_core_defect
    if adversarial_survived != expected_survived:
      errors.append(
        f"adjudication.json: adversarial_survived={adversarial_survived} but expected "
        f"{expected_survived} (any confirmed core-severity defect={any_core_defect})"
      )

  expected_n, expected_load = compute_ambiguity(clauses)

  n_ambiguous = adj.get("n_ambiguous")
  if not isinstance(n_ambiguous, int) or isinstance(n_ambiguous, bool):
    errors.append(f"adjudication.json: 'n_ambiguous' must be an int, got {n_ambiguous!r}")
  elif n_ambiguous != expected_n:
    errors.append(
      f"adjudication.json: n_ambiguous={n_ambiguous} but recomputed from clauses.json = "
      f"{expected_n}"
    )

  ambiguity_load = adj.get("ambiguity_load")
  if ambiguity_load not in AMBIGUITY_LOADS:
    errors.append(
      f"adjudication.json: 'ambiguity_load' must be one of {sorted(AMBIGUITY_LOADS)}, "
      f"got {ambiguity_load!r}"
    )
  elif ambiguity_load != expected_load:
    errors.append(
      f"adjudication.json: ambiguity_load={ambiguity_load!r} but recomputed from "
      f"clauses.json = {expected_load!r}"
    )

  return errors


def validate(directory: str | Path) -> list[str]:
  """Validate a litparity directory; return every violation (empty means valid).

  ``clauses.json`` is always validated. ``adjudication.json`` is validated only
  if it exists (it is written in a later phase). Prints which files were
  validated.
  """
  d = Path(directory)
  errors: list[str] = []
  validated: list[str] = []

  clauses_path = d / "clauses.json"
  clauses_list: list[dict[str, Any]] = []
  if not clauses_path.exists():
    errors.append(f"{clauses_path}: file not found")
  else:
    validated.append("clauses.json")
    try:
      clauses_obj = json.loads(clauses_path.read_text())
    except json.JSONDecodeError as exc:
      errors.append(f"{clauses_path}: invalid JSON ({exc})")
      clauses_obj = None
    if clauses_obj is not None:
      errors.extend(validate_clauses(clauses_obj))
      if isinstance(clauses_obj, dict) and isinstance(clauses_obj.get("clauses"), list):
        clauses_list = [c for c in clauses_obj["clauses"] if isinstance(c, dict)]

  adjudication_path = d / "adjudication.json"
  if adjudication_path.exists():
    validated.append("adjudication.json")
    try:
      adjudication_obj = json.loads(adjudication_path.read_text())
    except json.JSONDecodeError as exc:
      errors.append(f"{adjudication_path}: invalid JSON ({exc})")
      adjudication_obj = None
    if adjudication_obj is not None:
      errors.extend(validate_adjudication(adjudication_obj, clauses_list))

  recon_files = sorted(d.glob("recon_*.md"))
  if len(recon_files) != 3:
    errors.append(
      f"{d}: expected exactly 3 recon_*.md files, found {len(recon_files)} "
      f"({[p.name for p in recon_files]})"
    )

  print(f"validated: {', '.join(validated) if validated else '(none)'}")
  return errors


def main(argv: list[str] | None = None) -> int:
  """CLI entry point: validate the litparity directory and print violations."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("litparity_dir", help="Directory containing clauses.json / adjudication.json")
  args = parser.parse_args(argv)

  errors = validate(args.litparity_dir)
  if errors:
    print(f"{len(errors)} violation(s) found:")
    for err in errors:
      print(f"  - {err}")
    return 1

  print("OK: schema valid")
  return 0


if __name__ == "__main__":
  sys.exit(main())
