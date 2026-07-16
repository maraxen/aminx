"""Tests for the continuous HP contact energy (``aminx.ebm.hp_energy``).

**Honest scope statement (read this before reading any assertion below).**
This is an *internal consistency* check between two energy formulations of
the same rule, not a validation against experimental or even simulated
biophysical ground truth (same pattern as ``test_ddg_stability.py``'s scope
statement). ``scripts/validate/hp_lattice_sanity.py`` (Tier 0) reproduces
Lau & Dill (1989)'s own discrete 2D-lattice HP model exactly, and gives a
published fold / non-fold classification for six labeled n=10 sequences
(their Figure 2). A 2D discrete lattice and a continuous 3D coordinate space
are not numerically comparable -- there is no exact energy value this
module's output should match.

What these tests establish instead: for each of Tier 0's six labeled
sequences, this module embeds that *specific sequence's own native lattice
conformation* (found by Tier 0's exhaustive search -- a genuine, self-
avoiding, H-H-contact-maximizing conformation, not an arbitrary shape) as
continuous coordinates, and checks that ``hp_contact_energy`` puts a clear
gap between the folding sequences (native_m == t_max == 4, i.e. their best
lattice conformation realizes every contact the geometry allows) and the
non-folding ones (native_m in {0, 1}, well short of t_max). An earlier draft
of this test used a single composition-blind "compact" shape shared across
all six sequences and found it *couldn't* reproduce the classification --
unsurprising in hindsight, since a non-folding sequence's failure in the
discrete model is a *global* self-avoiding-walk constraint (the P residues
can't simultaneously be routed AND leave the H residues maximally compact),
not a local absence of nearby H residues. Using each sequence's own native
walk sidesteps that by construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aminx.ebm.hp_energy import hp_contact_energy

Site = tuple[int, int]
Walk = tuple[Site, ...]
_MOVES: tuple[Site, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Mirrors scripts/validate/hp_lattice_sanity.py exactly (Lau & Dill 1989,
# Macromolecules 22(10), 3986-3997, Figure 2) -- duplicated rather than
# imported, since scripts/ is excluded from the aminx package (ruff's own
# exclude list treats it as non-library code) and this keeps the two
# validation tiers independently runnable.
_FIGURE_2_SEQUENCES: dict[str, bool] = {
  "HHHHHHHHHH": True,
  "HPHPPHPPHH": True,
  "HPPHPPHPHH": True,
  "PPPPPHPPHH": False,
  "PPPPPHHHHH": False,
  "PPPPPPPPPP": False,
}

_BOND_LENGTH = 0.38  # real 3.8 Angstrom CA-CA bond, coordinate_scaling=0.1-scaled.
_EXTENDED_SPACING = 3.0  # >> 2x the default 0.8 cutoff: no pair ever registers a contact.


def _enumerate_saws(n: int) -> list[Walk]:
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


def _hh_contacts(walk: Walk, sequence: str) -> int:
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
  return count // 2


def _native_walk(sequence: str, walks: list[Walk]) -> Walk:
  """A native conformation achieving the sequence's own maximum H-H contact count."""
  return max(walks, key=lambda w: _hh_contacts(w, sequence))


_ALL_WALKS_N10 = _enumerate_saws(10)


def _walk_to_coords(walk: Walk) -> jnp.ndarray:
  """Embed integer lattice sites as continuous 3D coordinates (z=0), bond-length scaled."""
  xy = jnp.array(walk, dtype=jnp.float32) * _BOND_LENGTH
  z = jnp.zeros((xy.shape[0], 1), dtype=jnp.float32)
  return jnp.concatenate([xy, z], axis=-1)


def _extended_coords(n: int) -> jnp.ndarray:
  """A widely spaced straight-line backbone: no pair is ever within the contact cutoff."""
  x = jnp.arange(n, dtype=jnp.float32) * _EXTENDED_SPACING
  return jnp.stack([x, jnp.zeros(n), jnp.zeros(n)], axis=-1)


def _hp_labels(sequence: str) -> jnp.ndarray:
  return jnp.array([c == "H" for c in sequence], dtype=bool)


def test_native_walk_energy_gap_separates_folding_from_nonfolding() -> None:
  """Native-vs-extended energy gap, folding sequences vs. non-folding ones.

  Folding sequences realize t_max=4 H-H contacts in their native walk;
  non-folding ones realize at most 1 (per Tier 0's own search -- see
  ``hp_lattice_sanity.py``'s printed ``native_m`` values). The comparison is
  group-relative (min folding gap vs. max non-folding gap) rather than a
  fixed numeric threshold, since the exact gap magnitude is a function of
  this module's cutoff/sharpness/epsilon defaults, not a property either
  model claims to predict precisely.
  """
  gaps: dict[str, float] = {}
  for sequence in _FIGURE_2_SEQUENCES:
    n = len(sequence)
    hp_labels = _hp_labels(sequence)
    mask = jnp.ones(n, dtype=bool)

    native_walk = _native_walk(sequence, _ALL_WALKS_N10)
    native_energy = hp_contact_energy(_walk_to_coords(native_walk), hp_labels, mask)
    extended_energy = hp_contact_energy(_extended_coords(n), hp_labels, mask)
    gaps[sequence] = float(extended_energy - native_energy)

  folding_gaps = [gaps[s] for s, folds in _FIGURE_2_SEQUENCES.items() if folds]
  nonfolding_gaps = [gaps[s] for s, folds in _FIGURE_2_SEQUENCES.items() if not folds]

  assert min(folding_gaps) > max(nonfolding_gaps), (
    f"expected every folding sequence's native-vs-extended gap to exceed every "
    f"non-folding sequence's gap; got folding={gaps and {s: gaps[s] for s in gaps if _FIGURE_2_SEQUENCES[s]}}, "
    f"non-folding={ {s: gaps[s] for s in gaps if not _FIGURE_2_SEQUENCES[s]} }"
  )


def test_all_polar_sequence_is_energy_free() -> None:
  """A sequence with no H residues at all has zero HP contact energy regardless of geometry."""
  n = 10
  hp_labels = jnp.zeros(n, dtype=bool)
  mask = jnp.ones(n, dtype=bool)
  native_walk = _native_walk("P" * n, _ALL_WALKS_N10)
  assert float(hp_contact_energy(_walk_to_coords(native_walk), hp_labels, mask)) == pytest.approx(0.0)
  assert float(hp_contact_energy(_extended_coords(n), hp_labels, mask)) == pytest.approx(0.0)


def test_masked_residues_excluded() -> None:
  """Masking out every H residue removes their contribution regardless of proximity."""
  n = 10
  hp_labels = _hp_labels("HHHHHHHHHH")
  all_masked_out = jnp.zeros(n, dtype=bool)
  native_walk = _native_walk("H" * n, _ALL_WALKS_N10)
  energy = hp_contact_energy(_walk_to_coords(native_walk), hp_labels, all_masked_out)
  assert float(energy) == pytest.approx(0.0)
