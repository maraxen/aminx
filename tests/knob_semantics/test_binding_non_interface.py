# ruff: noqa: S101, SLF001
"""Spec §4.4: non-interface sites keep the absolute energy, and tied skips the reference.

The converge-stop half of ``test_knob_semantics_binding_converge_stop`` is
already covered: ``test_refine_modes_and_converge_stop``
(``test_sample.py:253``) pins ``n_iters == 1`` when a sweep changes nothing,
the ``max_iters`` cap, and a ``binding="both"`` converge. The BINDING half is
not, and the reason is structural -- every refine fixture in the suite builds
``inter_mask=jnp.ones(...)`` (``test_potts_knobs.py:441``,
``test_sample.py:194``), so the ``jnp.where(tables.inter_mask[position],
adjusted, base)`` at ``refine.py:496`` has only ever taken its TRUE branch.

Two properties pinned here, both reachable from ``_energy_vector`` without a
model:

1. A NON-INTERFACE position keeps ``base``, the absolute complex energy, while
   an interface position gets the binding-adjusted vector. Mixing the mask is
   the only way to see this; an all-True mask cannot.

2. TIED BINDING DOES NOT SUBTRACT THE CURRENT-IDENTITY REFERENCE. Untied is
   ``(base - base[current]) - (unbound - unbound[current])``; tied is plain
   ``base - unbound`` (``refine.py:488-493``). The module docstring records
   this as an upstream quirk. The difference between them is therefore a
   CONSTANT across the alphabet axis -- ``base[current] - unbound[current]`` --
   which is exactly the signature asserted below, so the test needs no
   hand-derived energies.

``base`` is never hand-computed: ``binding="none"`` returns it by definition
(``:492``), so the test reads it back out of the function under test rather
than re-deriving arithmetic that could drift.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.refine import BindingTables, _energy_vector

_L = 2
_A = 4
_SEQ = jnp.asarray(np.asarray([1, 2], dtype=np.int32))

# Complex and unbound tables differ, or `base - unbound` is zero and every
# comparison below collapses.
_ETAB = jnp.asarray(np.arange(_L * 1 * _A * _A, dtype=np.float64).reshape(_L, 1, _A, _A))
_UNBOUND = jnp.asarray(
  (np.arange(_L * 1 * _A * _A, dtype=np.float64).reshape(1, _L, 1, _A, _A) * 0.25) + 3.0,
)


def _tables(inter: list[bool]) -> BindingTables:
  return BindingTables(
    etab=_UNBOUND,
    e_idx=jnp.zeros((1, _L, 1), dtype=jnp.int32),
    pad_valid=jnp.ones((1, _L), dtype=jnp.bool_),
    complex_index=jnp.arange(_L, dtype=jnp.int32)[None, :],
    partition_of=jnp.zeros((_L,), dtype=jnp.int32),
    local_of=jnp.arange(_L, dtype=jnp.int32),
    inter_mask=jnp.asarray(np.asarray(inter, dtype=bool)),
  )


def _vector(position: int, tables: BindingTables, *, binding: str, tied: bool) -> np.ndarray:
  return np.asarray(
    _energy_vector(
      _SEQ,
      jnp.asarray(position, dtype=jnp.int32),
      _ETAB,
      jnp.zeros((_L, 1), dtype=jnp.int32),
      jnp.ones((_L,), dtype=jnp.bool_),
      tables,
      binding=binding,
      tied=tied,
    ),
  )


def test_knob_semantics_binding_non_interface_keeps_absolute_energy() -> None:
  """Position 1 is non-interface, so it keeps ``base`` while position 0 does not."""
  mixed = _tables([True, False])

  # binding="none" IS base, by definition at refine.py:492.
  base_0 = _vector(0, mixed, binding="none", tied=False)
  base_1 = _vector(1, mixed, binding="none", tied=False)

  adjusted_0 = _vector(0, mixed, binding="both", tied=False)
  adjusted_1 = _vector(1, mixed, binding="both", tied=False)

  # The non-interface row is untouched by the binding adjustment.
  np.testing.assert_array_equal(adjusted_1, base_1)
  # The interface row is not -- without this the test passes for an
  # implementation that ignores `binding` entirely.
  assert not np.allclose(adjusted_0, base_0), (
    f"fixture broken: the binding adjustment must move position 0, got {adjusted_0}"
  )


def test_binding_non_interface_control_all_interface_adjusts_both() -> None:
  """CONTROL: flip the mask to all-True and position 1 is adjusted too.

  This is what shows the previous test is reading ``inter_mask`` rather than
  treating position 1 specially for some other reason -- its index, its
  partition, or its place in the sequence.
  """
  everything = _tables([True, True])
  base_1 = _vector(1, everything, binding="none", tied=False)
  adjusted_1 = _vector(1, everything, binding="both", tied=False)
  assert not np.allclose(adjusted_1, base_1), adjusted_1


def test_knob_semantics_tied_binding_skips_the_current_identity_reference() -> None:
  """Tied binding differs from untied by a CONSTANT, which is the quirk.

  Untied subtracts the current residue's own energy from both terms; tied does
  not (``refine.py:488-493``, recorded in the module docstring as an upstream
  quirk). So the two differ by ``base[current] - unbound[current]``, the same
  value at every amino acid -- a constant offset, not a reshaping.

  Asserting constancy rather than a hand-computed number means the test states
  the structural property and cannot drift with the fixture's arithmetic.
  """
  mixed = _tables([True, False])
  untied = _vector(0, mixed, binding="both", tied=False)
  tied = _vector(0, mixed, binding="both", tied=True)

  delta = tied - untied
  np.testing.assert_allclose(delta, np.full_like(delta, delta[0]), rtol=0, atol=1e-12)

  # And the offset is NONZERO, or "differs by a constant" is satisfied by the
  # two paths being identical and the quirk would be invisible here.
  assert abs(float(delta[0])) > 1e-9, delta

  # The non-interface row ignores `tied` too, because it never reaches the
  # adjustment at all.
  np.testing.assert_array_equal(
    _vector(1, mixed, binding="both", tied=True),
    _vector(1, mixed, binding="both", tied=False),
  )
