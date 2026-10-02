#!/usr/bin/env python3
"""Are the alias map's equivalence claims reachable in the implementation?

``tests/knob_gate/alias_map.toml`` records, for every upstream knob, the aminx
field it corresponds to. ``test_parity_ids_passed`` then enshrines those claims by
requiring each live row to name a passing test. A row whose target field is never
read by the family it is mapped from asserts a correspondence the code does not
make -- the knob is accepted and ignored.

One such row class is already confirmed by hand: the seven ``backbone_noise`` /
``bb_noise`` rows map to ``noise``, and the LASEr path never reads it (aminx debt
2434). This probe exists to find whether that is an isolated case or a pattern,
and to enumerate it exhaustively rather than one grep at a time. The known
instance is declared in the sidecar so it cannot be mistaken for a prediction.

Two questions, deliberately separated because only the first is mechanical:

  A. Is the target field read ANYWHERE under ``src/aminx/``? Zero reads means the
     claim is unsupported outright, with no judgement needed.
  B. Is a field that is mapped ONLY from one family's reference surfaces read
     inside that family's own subtree? Reads only outside it are the ``noise``
     shape: handled by the generic path, which the spec says is the wrong rule for
     LASEr. Genuinely generic runner knobs (paths, batching, device) legitimately
     live outside, so these are reported for review, not failed on.

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
from pathlib import Path

logger = logging.getLogger("alias_claim_reachability")

# Family implementation subtrees. A field "belongs" to a family when every
# reference row that maps to it comes from that family's surfaces.
_FAMILY_TREES: dict[str, tuple[str, ...]] = {
  "laser": ("src/aminx/families/laser_mpnn", "src/aminx/model/laser"),
  "potts": ("src/aminx/families/potts_mpnn", "src/aminx/potts", "src/aminx/model/potts"),
}

# Runner-level concerns that are correctly handled outside any family subtree:
# the host resolves them before a driver is reached. Listed explicitly so the
# question-B report is interpretable instead of being dominated by them.
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

  An AST walk, not a substring grep: ``noise`` must not be credited to a file
  that only says ``backbone_noise_mode`` or mentions it in prose. Attribute
  access (``spec.noise``), plain names and keyword arguments (``noise=``) all
  count as a read, since any of them is the field being consumed.
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


def _names_in(root: Path, subtrees: tuple[str, ...] | None = None) -> set[str]:
  bases = [root / sub for sub in subtrees] if subtrees else [root / "src" / "aminx"]
  names: set[str] = set()
  for base in bases:
    if not base.exists():
      logger.warning("missing subtree %s", base)
      continue
    for path in sorted(base.rglob("*.py")):
      if "__pycache__" in path.parts:
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
  alias = repo / "tests" / "knob_gate" / "alias_map.toml"
  rows = tomllib.loads(alias.read_text(encoding="utf-8"))["row"]
  live = [r for r in rows if r["equivalence"] != "exclusion"]

  # target field -> set of families whose rows map to it
  families_of: dict[str, set[str]] = defaultdict(set)
  rows_of: dict[str, list[str]] = defaultdict(list)
  for row in live:
    family = _family_of(str(row["ref"]))
    for target in row["targets"]:
      families_of[str(target)].add(family or "?")
      rows_of[str(target)].append(str(row["ref"]))

  everywhere = _names_in(repo)
  per_family = {fam: _names_in(repo, trees) for fam, trees in _FAMILY_TREES.items()}

  unreachable = sorted(t for t in families_of if t not in everywhere)
  outside_family = sorted(
    target
    for target, fams in families_of.items()
    if target in everywhere
    and target not in _RUNNER_KNOBS
    and len(fams) == 1
    and (fam := next(iter(fams))) in per_family
    and target not in per_family[fam]
  )

  if unreachable and outside_family:
    outcome = "both"
  elif unreachable:
    outcome = "unreachable_targets"
  elif outside_family:
    outcome = "family_knob_handled_generically"
  else:
    outcome = "no_unreachable"

  payload = {
    "outcome": outcome,
    "n_rows": len(rows),
    "n_live_rows": len(live),
    "n_target_fields": len(families_of),
    "n_unreachable": len(unreachable),
    "n_outside_family": len(outside_family),
    "unreachable": {t: sorted(families_of[t]) for t in unreachable},
    "unreachable_rows": {t: rows_of[t] for t in unreachable},
    "outside_family": {
      t: {"family": sorted(families_of[t]), "rows": rows_of[t]}
      for t in outside_family
    },
    "runner_knobs_excluded_from_question_b": sorted(_RUNNER_KNOBS),
  }
  args.payload_out.parent.mkdir(parents=True, exist_ok=True)
  args.payload_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  logger.info(
    "outcome=%s target_fields=%d unreachable=%d outside_family=%d",
    outcome, len(families_of), len(unreachable), len(outside_family),
  )
  for target in unreachable:
    logger.info("UNREACHABLE %s <- %s", target, rows_of[target][:3])
  for target in outside_family:
    logger.info("OUTSIDE-FAMILY %s <- %s", target, rows_of[target][:3])
  return 0


if __name__ == "__main__":
  sys.exit(main())
