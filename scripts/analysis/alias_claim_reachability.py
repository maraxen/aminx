#!/usr/bin/env python3
"""Are the alias map's equivalence claims reachable in the implementation?

``tests/knob_gate/alias_map.toml`` records, for every upstream knob, the aminx
field it corresponds to. ``test_parity_ids_passed`` then enshrines those claims by
requiring each live row to name a passing test. A row whose target field the
implementation never consumes asserts a correspondence the code does not make --
the knob is accepted and silently ignored.

REVISION 2. Revision 1 is void by its own pre-registered adversarial check: it
fired ``family_knob_handled_generically`` while ``noise`` -- the one instance
declared in advance as already confirmed -- was absent from its hit list, and the
sidecar said that means the probe is broken, not the codebase sound. The check
found two defects:

  * A false negative. Revision 1 asked its family question only of fields mapped
    from exactly ONE family, and ``noise`` is mapped from both
    (``pottsmpnn_cfg__inference__noise`` and the LASEr ``backbone_noise`` rows),
    so it was skipped outright. Revision 2 asks it of every (family, field) PAIR
    a row creates, which is what the claim is actually about.
  * A missing question, and the sharpest one. Chasing a suspected false positive
    on ``chi_temp`` showed the opposite: the laser subtree handles the chi
    temperature under the parameter name ``chi_temperature``, while the exact
    token ``chi_temp`` appears nowhere but its own dataclass line
    (``run/options.py:42``). The rename is where the plumbing stops, so the
    Options field is inert. Neither of revision 1's questions could see that.

Three questions, ordered by how little judgement each needs:

  A. Is the target field read ANYWHERE under ``src/aminx/``? Zero reads means the
     claim names a consumer that does not exist. No judgement.
  B. Is a family Options field (``LaserOptions`` / ``PottsMPNNOptions``) read
     anywhere outside ``run/options.py``? Zero reads means the knob is declared
     and inert: accepted by the API, forwarded to nothing. No judgement.
  C. For each (family, field) pair, is the field read inside that family's own
     subtree? Reads only outside it are the ``noise`` shape -- handled by the
     generic path, which spec 6.3 says is the wrong rule for LASEr. Genuinely
     generic runner knobs legitimately live outside, so C is reported for review
     rather than failed on.

task_id 260929_potts-laser-xtrax-compose
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import tomllib
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

logger = logging.getLogger("alias_claim_reachability")

_FAMILY_TREES: dict[str, tuple[str, ...]] = {
  "laser": ("src/aminx/families/laser_mpnn", "src/aminx/model/laser"),
  "potts": ("src/aminx/families/potts_mpnn", "src/aminx/potts"),
}

# Runner-level concerns the host resolves before any driver runs, so living
# outside a family subtree is correct for them. Excluded from question C only,
# never from A or B, and recorded in the payload so the call can be audited.
_RUNNER_KNOBS = frozenset({
  "inputs", "output_dir", "output_h5_path", "cache_path", "overwrite_cache",
  "batch_size", "samples_batch_size", "noise_batch_size", "temperature_batch_size",
  "max_length", "max_workers", "max_buffer_size", "n_devices", "ram_budget_mb",
  "num_samples", "sample_count", "sample_start", "random_seed", "split",
  "checkpoint_id", "checkpoint_registry_path", "model_local_path", "model_weights",
  "model_version", "model_family", "model", "run_spec", "job_id", "chunk_id",
  "host_resource_allocation_strategy", "truncation_strategy", "topology",
  "preprocessed_index_path", "use_preprocessed", "foldcomp_database",
  "sequences_to_score", "output_kind",
})


def _family_of(ref: str) -> str | None:
  if ref.startswith("laser_"):
    return "laser"
  if ref.startswith(("potts_", "pottsmpnn_")):
    return "potts"
  return None


def _read_names(path: Path) -> set[str]:
  """Every identifier and attribute name mentioned in one Python file.

  An ast walk, not a substring grep. The distinction is load-bearing twice over:
  a grep for ``noise`` credits any file saying ``backbone_noise_mode``, and a grep
  for ``chi_temp`` credits ``chi_temperature`` -- which is precisely the rename
  that hides an unplumbed field.
  """
  try:
    tree = ast.parse(path.read_text(encoding="utf-8"))
  except (SyntaxError, UnicodeDecodeError):
    logger.warning("skipping unparsable %s", path)
    return set()
  names: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      names.add(node.id)
    elif isinstance(node, ast.Attribute):
      names.add(node.attr)
    elif isinstance(node, ast.keyword) and node.arg:
      names.add(node.arg)
    elif isinstance(node, ast.arg):
      names.add(node.arg)
  return names


def _names_in(root: Path, subtrees: tuple[str, ...] | None, skip: Path | None) -> set[str]:
  bases = [root / sub for sub in subtrees] if subtrees else [root / "src" / "aminx"]
  names: set[str] = set()
  for base in bases:
    if not base.exists():
      logger.warning("missing subtree %s", base)
      continue
    for path in sorted(base.rglob("*.py")):
      if "__pycache__" in path.parts or (skip and path.resolve() == skip):
        continue
      names |= _read_names(path)
  return names


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
  parser.add_argument("--payload-out", type=Path, required=True)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  repo: Path = args.repo
  sys.path.insert(0, str(repo / "src"))
  from aminx.run.options import LaserOptions, PottsMPNNOptions  # noqa: PLC0415

  alias = repo / "tests" / "knob_gate" / "alias_map.toml"
  rows = tomllib.loads(alias.read_text(encoding="utf-8"))["row"]
  live = [r for r in rows if r["equivalence"] != "exclusion"]

  pairs: set[tuple[str, str]] = set()
  rows_of: dict[str, list[str]] = defaultdict(list)
  targets: set[str] = set()
  for row in live:
    family = _family_of(str(row["ref"]))
    for raw in row["targets"]:
      target = str(raw)
      targets.add(target)
      rows_of[target].append(str(row["ref"]))
      if family:
        pairs.add((family, target))

  options_file = (repo / "src" / "aminx" / "run" / "options.py").resolve()
  everywhere = _names_in(repo, None, None)
  outside_options = _names_in(repo, None, options_file)
  per_family = {
    fam: _names_in(repo, trees, None) for fam, trees in _FAMILY_TREES.items()
  }

  option_fields = {
    "laser": {f.name for f in fields(LaserOptions)},
    "potts": {f.name for f in fields(PottsMPNNOptions)},
  }

  unreachable = sorted(t for t in targets if t not in everywhere)
  inert_options = sorted(
    {
      target
      for fam, members in option_fields.items()
      for target in members & targets
      if target not in outside_options
      for _ in (fam,)
    },
  )
  outside_family = sorted(
    {
      f"{fam}:{target}"
      for fam, target in pairs
      if target in everywhere
      and target not in _RUNNER_KNOBS
      and fam in per_family
      and target not in per_family[fam]
    },
  )

  if unreachable:
    outcome = "unreachable_targets"
  elif inert_options:
    outcome = "inert_option_fields"
  elif outside_family:
    outcome = "family_knob_handled_generically"
  else:
    outcome = "all_claims_reachable"

  payload = {
    "outcome": outcome,
    "revision": 2,
    "n_rows": len(rows),
    "n_live_rows": len(live),
    "n_target_fields": len(targets),
    "n_family_field_pairs": len(pairs),
    "n_unreachable": len(unreachable),
    "n_inert_options": len(inert_options),
    "n_outside_family": len(outside_family),
    "unreachable": {t: rows_of[t] for t in unreachable},
    "inert_options": {t: rows_of[t] for t in inert_options},
    "inert_options_by_family": {
      fam: sorted(m & set(inert_options)) for fam, m in option_fields.items()
    },
    "outside_family": {
      key: rows_of[key.split(":", 1)[1]] for key in outside_family
    },
    "noise_is_a_hit": any(k.endswith(":noise") for k in outside_family),
    "runner_knobs_excluded_from_question_c": sorted(_RUNNER_KNOBS),
  }
  args.payload_out.parent.mkdir(parents=True, exist_ok=True)
  args.payload_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  logger.info(
    "outcome=%s pairs=%d unreachable=%d inert_options=%d outside_family=%d "
    "noise_is_a_hit=%s",
    outcome, len(pairs), len(unreachable), len(inert_options),
    len(outside_family), payload["noise_is_a_hit"],
  )
  for target in unreachable:
    logger.info("UNREACHABLE      %s <- %s", target, rows_of[target][:3])
  for target in inert_options:
    logger.info("INERT-OPTION     %s <- %s", target, rows_of[target][:3])
  for key in outside_family:
    logger.info("OUTSIDE-FAMILY   %s", key)
  return 0


if __name__ == "__main__":
  sys.exit(main())
