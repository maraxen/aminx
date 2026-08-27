"""Safe mapping over arrays and PyTrees, dispatching between vmap and lax.map.

This local fork is retained (Option 3 in
`.praxia/docs/specs/260827_runspec-scaffolding-remediation-migration-map-re-authoring-and-xtrax-transforms-adoption.md` §4.3,
praxia debt #1515) because ``xtrax.transforms.map.safe_map`` enforces a strict divisibility check
(``num_elements % batch_size == 0``, ``xtrax/transforms/map.py:33-37``) that rejects non-divisible
batch sizes and raises ``ZeroDivisionError`` on ``batch_size=0``. In contrast, ``jax.lax.map``
natively supports non-divisible cardinality via ``_remainder_leaf`` (``jax/_src/lax/loops.py:2647``).

Concrete divergences from ``xtrax.transforms.map.safe_map``:

+----------------------------+---------------------------------+--------------------------------------+
| Behaviour                  | ``aminx.utils.safe_map``        | ``xtrax.transforms.map.safe_map``   |
+============================+=================================+======================================+
| ``batch_size=0`` sentinel  | Routes to ``vmap`` (:49;        | Raises ``ZeroDivisionError``         |
|                            | see ``plan.py:307-317``)        | (``map.py:33`` divisibility modulo)  |
+----------------------------+---------------------------------+--------------------------------------+
| Non-divisible batch size   | Supported (``lax.map`` handles  | Raises ``ValueError``                |
| (e.g. N=100, batch=32)     | remainder slices natively)      | (``map.py:37`` divisibility check)   |
+----------------------------+---------------------------------+--------------------------------------+
| Empty PyTree input (``{}``)| Raises ``ValueError`` (:45)     | Raises ``IndexError`` (``map.py:26``)|
+----------------------------+---------------------------------+--------------------------------------+

Adoption of upstream xtrax ``safe_map`` is deferred until xtrax relaxes its divisibility requirement
(tracked in praxia debt #1515).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import jax

if TYPE_CHECKING:
  from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")


def safe_map(
  f: Callable[[T], U],
  xs: Any,  # noqa: ANN401
  batch_size: int | None = None,
) -> Any:  # noqa: ANN401
  """Map a function over the first axis of xs.

  This function dispatches to jax.vmap if the input size is smaller than or equal to
  the batch size (or if batch_size is None), avoiding the overhead and potential
  XLA issues of jax.lax.map's loop construct for single-batch or small inputs.
  Otherwise, it falls back to jax.lax.map.

  Args:
      f: The function to map.
      xs: The input array(s).
      batch_size: The batch size for processing.

  Returns:
      The result of mapping f over xs.

  """
  leaves = jax.tree_util.tree_leaves(xs)
  if not leaves:
    msg = "Input xs must not be an empty PyTree"
    raise ValueError(msg)

  num_elements = leaves[0].shape[0]

  if batch_size is None or batch_size == 0 or num_elements <= batch_size:
    return jax.vmap(f)(xs)

  return jax.lax.map(f, xs, batch_size=batch_size)
