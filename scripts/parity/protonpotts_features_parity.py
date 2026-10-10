"""The ``protonpotts_features`` wave: the aminx featurizer against the sealed upstream feature dumps.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §6, §7, §39. Run in an environment with
aminx importable (the sprint venv on titanix)::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_features_parity.py \\
        --features-dir ~/projects/aminx-oracles-protonpotts --potts-repo ~/repos/PottsMPNN

WHAT IS COMPARED. Ten structures that upstream featurizes (P4b run 6a8ee503, P4c run 9a9b75bb, P4d run
7ec1f6b9, whose 1OLR cell discriminates the O rule): every
key, EXACTLY, with ``X`` compared everywhere (masked atoms included). One structure (1IFC) that upstream
REFUSES: the featurizer must refuse it too.

WHAT MAKES A PASS MEAN SOMETHING (spec §6). A wave is an instrument only if a deliberate perturbation of
the component is REJECTED by the same comparison. So the run also applies eleven perturbations, six that
make a featurizer RULE wrong and five that corrupt an OUTPUT, and every one must be detected by at least
one cell. A rule the dumps cannot discriminate would show up here as an undetected perturbation.

Reads the dumps; writes nothing but the bathos results file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the featurizer needs no accelerator

import numpy as np  # noqa: E402

from aminx.families.protonpotts_mpnn import features  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_features_parity")

FIRE = "energy_benchmark_datasets/fireprot_pdbs"
P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"  # same dumper as P4c, different cells
# (dump subdir, cell, pdb relative to the PottsMPNN checkout)
CELLS = [
  (P4B, "pkad_unlabelled", f"{FIRE}/1BVC.pdb"),
  (P4B, "pkad_labelled", f"{FIRE}/1BVC.pdb"),
  (P4B, "multichain", "energy_benchmark_datasets/covid_pdbs/6m0j.pdb"),
  (P4B, "ligand", "inputs/example_pdbs/swe1_ligand.pdb"),
  (P4C, "only_o", f"{FIRE}/1CQW.pdb"),
  (P4C, "gap", f"{FIRE}/1EL1.pdb"),
  (P4C, "many_gaps", f"{FIRE}/1BAH.pdb"),
  (P4C, "gap_and_drops", "inputs/example_pdbs/2yc3.pdb"),
  (P4C, "icode", f"{FIRE}/1TPK.pdb"),
  (P4D, "only_o_1olr", f"{FIRE}/1OLR.pdb"),
]
REFUSED_CELL = ("many_drops", f"{FIRE}/1IFC.pdb")
KEYS = ("R_idx", "chain_labels", "residue_mask", "X_m", "S", "X")
# The dumper's labelling rule (dump_protonpotts_oracles.py:74): cycle by ordinal within residue type.
CYCLE = {
  "HIS": ("HIS-P", "HIS-S", "HIS-A"),
  "ASP": ("ASP-P", "ASP-D", "ASP-A"),
  "GLU": ("GLU-P", "GLU-D", "GLU-A"),
}


def cycling_labels(pdb: Path) -> list[str]:
  seen: dict[str, int] = {}
  out = []
  for _chain, _number, _icode, name in features.loaded_residues(pdb):
    options = CYCLE.get(name)
    if options is None:
      out.append("")
      continue
    k = seen.get(name, 0)
    seen[name] = k + 1
    out.append(options[k % 3])
  return out


def mismatching_keys(got: dict[str, np.ndarray], ref: np.lib.npyio.NpzFile) -> list[str]:
  bad = []
  for key in KEYS:
    want = np.vectorize(upstream_to_aminx_index)(ref[key]) if key == "S" else ref[key]
    if got[key].shape != want.shape or not np.array_equal(got[key], want):
      bad.append(key)
  return bad


# --- the perturbations ---------------------------------------------------------------


@contextmanager
def _patched(name: str, value: object) -> Generator[None]:
  original = getattr(features, name)
  setattr(features, name, value)
  try:
    yield
  finally:
    setattr(features, name, original)


def _identity(residues: list) -> list:  # type: ignore[type-arg]
  return residues


RULE_MUTANTS: dict[str, Callable[[], AbstractContextManager[None]]] = {
  "backbone_threshold_0_5": lambda: _patched("BACKBONE_OCCUPANCY_THRESHOLD", 0.5),
  "atom_threshold_just_below_half": lambda: _patched("ATOM_OCCUPANCY_THRESHOLD", 0.4999),
  "backbone_without_o": lambda: _patched("BACKBONE_ATOMS", ("N", "CA", "C")),
  "no_arginine_fix": lambda: _patched("_fix_arginine", lambda _residue: None),
  "no_keep_last_by_number": lambda: _patched("_keep_last_by_number", _identity),
  "atom_threshold_0_8": lambda: _patched("ATOM_OCCUPANCY_THRESHOLD", 0.8),
}


def _bump_r_idx(out: dict[str, np.ndarray]) -> None:
  out["R_idx"] = out["R_idx"].copy()
  out["R_idx"][0, -1] += 1


def _change_s(out: dict[str, np.ndarray]) -> None:
  out["S"] = out["S"].copy()
  out["S"][0, 0] = (out["S"][0, 0] + 1) % 21


def _swap_chain_labels(out: dict[str, np.ndarray]) -> None:
  out["chain_labels"] = 1 - out["chain_labels"]


def _flip_x_m(out: dict[str, np.ndarray]) -> None:
  out["X_m"] = out["X_m"].copy()
  out["X_m"][0, 0, 0] = ~out["X_m"][0, 0, 0]


def _nudge_x(out: dict[str, np.ndarray]) -> None:
  out["X"] = out["X"].copy()
  out["X"][0, 0, 1, 0] += np.float32(1e-4)


OUTPUT_MUTANTS: dict[str, Callable[[dict[str, np.ndarray]], None]] = {
  "r_idx_last_plus_one": _bump_r_idx,
  "s_first_token_changed": _change_s,
  "chain_labels_inverted": _swap_chain_labels,
  "x_m_first_atom_flipped": _flip_x_m,
  "x_nudged_by_1e-4": _nudge_x,
}


# --- the wave ------------------------------------------------------------------------


def run_cells(
  features_dir: Path,
  potts_repo: Path,
  perturb: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> tuple[dict[str, list[str]], bool]:
  """Mismatching keys per featurized cell, and whether the refused cell was refused."""
  bad: dict[str, list[str]] = {}
  for subdir, cell, pdb in CELLS:
    structure = potts_repo / pdb
    ref = np.load(features_dir / subdir / f"{cell}.npz")
    labels = cycling_labels(structure) if cell == "pkad_labelled" else None
    try:
      got = features.featurize_pdb(structure, labels)
    except features.PdbFeatureError:
      bad[cell] = ["raised"]
      continue
    if perturb is not None:
      perturb(got)
    keys = mismatching_keys(got, ref)
    if keys:
      bad[cell] = keys
  try:
    features.featurize_pdb(potts_repo / REFUSED_CELL[1])
    refused = False
  except features.PdbFeatureError:
    refused = True
  return bad, refused


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--potts-repo", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  features_dir, potts_repo = args.features_dir.expanduser(), args.potts_repo.expanduser()

  clean_bad, refused = run_cells(features_dir, potts_repo)
  for cell, keys in clean_bad.items():
    logger.info("CLEAN MISMATCH %s: %s", cell, keys)
  clean = "pass" if not clean_bad and refused else "fail"

  mutants: dict[str, str] = {}
  for name, make in RULE_MUTANTS.items():
    with make():
      bad, still_refused = run_cells(features_dir, potts_repo)
    mutants[name] = "failed" if bad or not still_refused else "passed"
  for name, perturb in OUTPUT_MUTANTS.items():
    bad, _ = run_cells(features_dir, potts_repo, perturb)
    mutants[name] = "failed" if bad else "passed"
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  results = {
    "clean": clean,
    "n_cells_exact": len(CELLS) - len(clean_bad),
    "n_cells": len(CELLS),
    "refusal_matched": refused,
    "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"),
  }
  logger.info(json.dumps(results, indent=1))
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_features_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
