"""Utility for safe mapping over arrays, avoiding XLA loop issues.

In-repo implementation mirrors jaxbeans ``utils/mapping.safe_map`` semantics (roadmap §3.6 **DEPEND**).
Swapping to an explicit ``jaxbeans`` dependency is deferred until workspace packaging stabilizes;
behavioral parity for JIT is gated by ``tests/utils/test_safe_map.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jax.numpy as jnp

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

  It never lets XLA compile a vmapped chunk of size exactly 1 (aminx #2391). On one GPU stack
  (TITAN RTX, jax 0.10.2, CUDA 12.9) a jitted size-1 batch around a matmul whose intermediate is
  square returned wrong activations silently. A chunk of 1 arises three ways, all handled here:
  a size-1 axis, ``batch_size == 1`` (lax.map vmaps each chunk of 1), and a remainder of 1
  (``n % batch_size == 1``; lax.map vmaps the remainder). Results are identical on every backend.
  Temporary home: the same guard belongs in xtrax's ``chunked_map``, after which this helper is a
  re-export and then removed (aminx debt #2371).

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

  if num_elements == 1:
    return _call_on_single_element(f, xs)

  if batch_size is None or batch_size == 0 or num_elements <= batch_size:
    return jax.vmap(f)(xs)

  if batch_size == 1:
    return jax.lax.map(f, xs)  # a plain scan: one unbatched call per element

  if num_elements % batch_size == 1:
    # lax.map would vmap the trailing remainder of 1; peel it off and run it unbatched.
    head = jax.lax.map(f, jax.tree_util.tree_map(lambda x: x[:-1], xs), batch_size=batch_size)
    tail = _call_on_single_element(f, jax.tree_util.tree_map(lambda x: x[-1:], xs))
    return jax.tree_util.tree_map(lambda h, t: jnp.concatenate([h, t], axis=0), head, tail)

  return jax.lax.map(f, xs, batch_size=batch_size)


def _call_on_single_element(f: Callable[[Any], Any], xs: Any) -> Any:  # noqa: ANN401
  """``jax.vmap(f)(xs)`` for a leading axis of size 1, without a batched XLA program."""
  element = jax.tree_util.tree_map(lambda x: x[0], xs)
  return jax.tree_util.tree_map(lambda y: jnp.asarray(y)[None], f(element))
