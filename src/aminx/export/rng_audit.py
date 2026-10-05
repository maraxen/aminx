"""Structural jaxpr walker proving a traced function contains no RNG primitives.

The xtrax export-safety gate (V5) is not an RNG-freedom check: its threefry rule only
fires on input-derived keys, and a trace failure returns ``[]`` rather than raising, so a
wrapper could carry a planted RNG primitive and still pass Phase-0's census clean. This
module is the RNG-freedom check that gate does not provide, used by
``tests/export/test_export_wrappers.py`` test group (e) to assert
``find_rng_primitives(make_jaxpr(wrapper)) == []`` for both P03 and P04 wrappers, and that
the walk actually flags a real RNG primitive when one is present (kernel jaxpr, a planted
``normal``, RNG nested inside a ``cond`` inside a ``jit``, and a planted
``jax.lax.rng_bit_generator``).
"""

from __future__ import annotations

from typing import Any

#: Primitive names that consume or produce PRNG state anywhere in a jaxpr.
RNG_PRIMITIVES: frozenset[str] = frozenset(
  {
    "random_bits",
    "random_seed",
    "random_wrap",
    "random_unwrap",
    "random_split",
    "random_fold_in",
    "random_clone",
    "threefry2x32",
    "rng_bit_generator",
    "rng_uniform",
  },
)


def _sub_jaxprs(value: Any) -> list[Any]:  # noqa: ANN401
  """Return every jaxpr-like object directly reachable from an eqn param value.

  A param value may itself be a ``Jaxpr``/``ClosedJaxpr`` (``pjit``, ``scan``,
  ``while``'s ``cond_jaxpr``/``body_jaxpr``, ``custom_jvp``/``custom_vjp``'s
  ``call_jaxpr``), or a tuple/list of them (``cond``'s ``branches``). Anything else
  (arrays, dtypes, plain Python scalars) yields nothing.
  """
  candidates = value if isinstance(value, (list, tuple)) else (value,)
  found = []
  for candidate in candidates:
    inner = getattr(candidate, "jaxpr", candidate)
    if hasattr(inner, "eqns"):
      found.append(inner)
  return found


def find_rng_primitives(closed_jaxpr: Any) -> list[str]:  # noqa: ANN401
  """Recursively collect every RNG primitive name reachable from ``closed_jaxpr``.

  Recurses into every eqn param that holds a nested ``Jaxpr``/``ClosedJaxpr`` (or a
  tuple of them), so RNG use hidden inside a ``cond`` branch, a ``scan`` body, a nested
  ``pjit``, a ``while`` loop's cond/body, or a ``custom_jvp``/``custom_vjp`` rule is not
  missed just because it is not a top-level equation.

  Args:
    closed_jaxpr: A ``ClosedJaxpr`` (e.g. from ``jax.make_jaxpr(fn)(*avals)``) or a bare
      ``Jaxpr``.

  Returns:
    The RNG primitive names found, in traversal order, one entry per occurrence
    (duplicates included). Empty means no RNG primitive anywhere in the traced program.
  """
  found: list[str] = []

  def walk(jaxpr: Any) -> None:  # noqa: ANN401
    for eqn in jaxpr.eqns:
      if eqn.primitive.name in RNG_PRIMITIVES:
        found.append(eqn.primitive.name)
      for value in eqn.params.values():
        for sub in _sub_jaxprs(value):
          walk(sub)

  root = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
  walk(root)
  return found
