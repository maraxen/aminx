#!/usr/bin/env python3
"""Sanity check for the Lau & Dill (1989) 2D-lattice HP model.

Reproduces two invariants directly from the primary source (Lau, K.F. &
Dill, K.A. "A Lattice Statistical Mechanics Model of the Conformational and
Sequence Spaces of Proteins." Macromolecules 22(10), 3986-3997, 1989):

  1. Enumeration count: the paper states the full conformational space for a
     chain of length 10 on the 2D square lattice has exactly 2034 distinct
     conformations (p. 3988). This is counted up to the lattice's dihedral
     symmetry (4 rotations x 2 reflections, order 8) -- confirmed here by
     direct computation: raw self-avoiding-walk enumeration for a 9-bond,
     10-vertex chain gives 16268 (matches OEIS A001411, a(9)), and reducing
     by the dihedral-8 canonical form gives exactly 2034.
  2. Folding classification: the paper's Figure 2 gives six labeled n=10
     sequences with published fold / non-fold behavior in the epsilon ->
     -infinity limit (Sec. "Stability and Denaturation", eq. 6-8): the
     native state(s) are the conformation(s) maximizing HH contact count m,
     and a sequence "folds" iff its native m equals the geometric
     compactness ceiling t_max (i.e. <rho>_ns = 1, maximal compactness).

This is Tier 0 of the HP-model coarse-init exploration for aminx's
ProteinEBM: pure discrete combinatorics, no aminx/JAX dependency, checked
against the primary source before anything continuous is built on top of it
(see src/aminx/ebm/hp_energy.py, and ~/.claude/rules/BATHOS.md's "verify
your measurement pipeline before trusting any research conclusion").

DANGEROUS: none of this touches JAX/GPU. --dry-run validates args only.
"""

from __future__ import annotations

import argparse
import logging
import sys

Site = tuple[int, int]
Walk = tuple[Site, ...]

_MOVES: tuple[Site, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Figure 2 of Lau & Dill (1989): six labeled n=10 sequences with published
# fold / non-fold classification at strong H-H attraction.
_FIGURE_2_SEQUENCES: dict[str, bool] = {
  "HHHHHHHHHH": True,  # (a) Phi=1.0
  "HPHPPHPPHH": True,  # (b) Phi=0.5
  "HPPHPPHPHH": True,  # (c) Phi=0.5
  "PPPPPHPPHH": False,  # (d) Phi=0.2
  "PPPPPHHHHH": False,  # (e) Phi=0.5
  "PPPPPPPPPP": False,  # (f) Phi=0.0
}

_EXPECTED_N10_CONFORMATION_COUNT = 2034


def enumerate_saws(n: int) -> list[Walk]:
  """Enumerate every self-avoiding walk of `n` residues on the 2D square lattice.

  Depth-first search from a fixed origin, matching the paper's own
  description of its enumeration procedure (p. 3988): "a depth-first
  algorithm, which seeks the longest branch of the directed graph... The
  algorithm backtracks either when the full chain... has been generated or
  there is a dead end due to an excluded-volume violation." The starting
  site and orientation are not fixed, so this raw count matches OEIS
  A001411 (walks of `n - 1` steps), not the paper's own dihedral-reduced
  figure -- see `canonical_form` for that reduction.
  """
  walks: list[Walk] = []

  def extend(walk: Walk, occupied: frozenset[Site]) -> None:
    if len(walk) == n:
      walks.append(walk)
      return
    x, y = walk[-1]
    for dx, dy in _MOVES:
      site = (x + dx, y + dy)
      if site not in occupied:
        extend((*walk, site), occupied | {site})

  extend(((0, 0),), frozenset({(0, 0)}))
  return walks


def canonical_form(walk: Walk) -> Walk:
  """Canonicalize a walk under the square lattice's dihedral-8 symmetry group.

  Two conformations related purely by a global rotation or reflection have
  identical energy and represent the same physical shape; the paper's
  stated conformation counts are up to this symmetry (see module docstring
  point 1). Returns the lexicographically smallest of the 8 symmetry images.
  """
  forms: list[Walk] = []
  pts = walk
  for _ in range(4):
    pts = tuple((y, -x) for x, y in pts)
    forms.append(pts)
    forms.append(tuple((x, -y) for x, y in pts))
  return min(forms)


def hh_contacts(walk: Walk, sequence: str) -> int:
  """Count H-H topological-neighbor contacts for one conformation.

  A "topological neighbor" pair (Lau & Dill's term) is lattice-adjacent but
  not sequence-adjacent -- i.e. a non-covalent contact. Every other contact
  type (H-P, P-P, or solvent) contributes 0 to the energy in this model.
  """
  site_index = {site: i for i, site in enumerate(walk)}
  count = 0
  for i, site in enumerate(walk):
    if sequence[i] != "H":
      continue
    x, y = site
    for dx, dy in _MOVES:
      j = site_index.get((x + dx, y + dy))
      if j is not None and abs(j - i) > 1 and sequence[j] == "H":
        count += 1
  return count // 2  # each contact counted from both ends


def native_m(sequence: str, walks: list[Walk]) -> int:
  """Max H-H contact count achieved by any conformation.

  This is the sequence's native state population in the epsilon ->
  -infinity limit of eq. 6-8: Z_infinity = g(s) where s is exactly this
  maximum.
  """
  return max(hh_contacts(walk, sequence) for walk in walks)


def folds(sequence: str, walks: list[Walk], t_max: int) -> bool:
  """A sequence "folds" iff its native state achieves maximal compactness (<rho>_ns == 1)."""
  return native_m(sequence, walks) == t_max


def setup_logging(level: int = logging.INFO) -> None:
  """Configure logging to stderr with timestamps."""
  logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
  )


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="HP-lattice sanity check: reproduce Lau & Dill (1989)'s own numbers exactly."
  )
  parser.add_argument(
    "--n",
    type=int,
    default=10,
    help="Chain length (residues). Default 10 matches the paper's own exhaustive-enumeration example.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate args only, skip enumeration, then exit 0.",
  )
  args = parser.parse_args()

  if args.n <= 1:
    parser.error(f"--n must be > 1, got {args.n}")

  if args.dry_run:
    print("dry-run: args OK", file=sys.stderr)
    return 0

  setup_logging()
  logger = logging.getLogger(__name__)

  logger.info(f"Enumerating self-avoiding walks: n={args.n}")
  walks = enumerate_saws(args.n)
  logger.info(f"  raw SAW count = {len(walks)}")

  t_max = native_m("H" * args.n, walks)
  logger.info(f"  t_max (compactness ceiling, via all-H sequence) = {t_max}")

  # ==========================================================================
  # ASSERTION 1: exact conformation count (dihedral-8-reduced), n=10 only
  # ==========================================================================
  assertion_1_pass = True
  if args.n == 10:
    canonical_count = len({canonical_form(walk) for walk in walks})
    assertion_1_pass = canonical_count == _EXPECTED_N10_CONFORMATION_COUNT
    logger.info(
      f"ASSERTION 1 (n=10 exact conformation count): "
      f"{'PASS' if assertion_1_pass else 'FAIL'} "
      f"({canonical_count} {'==' if assertion_1_pass else '!='} {_EXPECTED_N10_CONFORMATION_COUNT})"
    )
  else:
    logger.info(f"ASSERTION 1: skipped (n={args.n} != 10, paper's exact count is n=10-specific)")

  # ==========================================================================
  # ASSERTION 2: Figure 2 fold/non-fold classification, n=10 only
  # ==========================================================================
  assertion_2_pass = True
  if args.n == 10:
    logger.info("Classifying Figure 2's six labeled sequences...")
    mismatches = []
    for sequence, expected_folds in _FIGURE_2_SEQUENCES.items():
      actual_folds = folds(sequence, walks, t_max)
      status = "PASS" if actual_folds == expected_folds else "FAIL"
      logger.info(f"  {sequence}  expected={expected_folds}  actual={actual_folds}  [{status}]")
      if actual_folds != expected_folds:
        mismatches.append(sequence)
    assertion_2_pass = not mismatches
    logger.info(
      f"ASSERTION 2 (Figure 2 classification): {'PASS' if assertion_2_pass else 'FAIL'}"
      + (f" -- mismatches: {mismatches}" if mismatches else "")
    )
  else:
    logger.info(f"ASSERTION 2: skipped (n={args.n} != 10, Figure 2 sequences are all n=10)")

  logger.info("")
  logger.info("=" * 70)
  if assertion_1_pass and assertion_2_pass:
    logger.info("ALL ASSERTIONS PASSED")
    logger.info("=" * 70)
    return 0
  logger.error("ONE OR MORE ASSERTIONS FAILED")
  logger.error("=" * 70)
  return 1


if __name__ == "__main__":
  sys.exit(main())
