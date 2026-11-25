"""Jaxpr loop-scaling guards shared by the debt #1981 complexity tests.

``loop_bodies`` only traces. Work is an explicit proxy (one matmul term plus
output sizes), so these helpers compare ratios of the same loop across extents.
They never execute the program. A ``while`` lowered from ``fori_loop`` reports
its trip count when both bounds are static integer literals (including a
``fori`` upper bound that is a traced argument closed over a literal); every
other ``while`` stays ``None``.

Criterion A: a loop that iterates once per element of an axis must do
extent-independent work per iteration (ratio < 1.5). Loops whose trip count
grows slower than that (``jnp.searchsorted``) are exempt.

Criterion B: a fixed-trip loop over an L-sized state must grow at most linearly
(ratio < 3.0), and its trip count must not start depending on L.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching
from xtrax.profiling.jaxpr import sub_jaxprs
from xtrax.profiling.loop_scaling import LoopStructureMismatchError, loop_bodies

#: Per-iteration work ratio at or above which criterion A fails. Same split
#: xtrax uses between "independent" (~1) and "recomputes the extent" (~2).
A_THRESHOLD = 1.5
#: Per-iteration work ratio at or above which criterion B fails. Sits between
#: linear (~2) and quadratic (~4).
B_THRESHOLD = 3.0
#: A must-fire mutant has to clear the threshold with this margin.
A_MUTANT_MIN = 1.7
#: A criterion-B mutant has to land strictly above this.
B_MUTANT_MIN = 3.5
#: Trip-count ratio at or above which a loop is treated as iterating over the swept axis.
TRIP_SCALING_RATIO = 1.5

#: Identity primitive a test inserts where a full model call stands in for the encoder.
ENCODER_MARK = "aminx_encoder_mark"

_encoder_mark_p = Primitive(ENCODER_MARK)


def _encoder_mark_impl(value: jax.Array) -> jax.Array:
  return value


_encoder_mark_p.def_impl(_encoder_mark_impl)
_encoder_mark_p.def_abstract_eval(lambda value: value)
# Encode (and a chunked pair map) vmaps the model call. The mark is an identity,
# so it keeps the batch axis of its input.
batching.defvectorized(_encoder_mark_p)


def mark_array(value: jax.Array) -> jax.Array:
  """Return ``value`` unchanged, leaving ``ENCODER_MARK`` in the jaxpr.

  Multiplying the mark by zero still keeps the primitive, so a later trace can
  see that the marked call ran inside a loop.
  """
  return _encoder_mark_p.bind(value)


@dataclass(frozen=True)
class LoopRow:
  """One loop body compared at every requested extent, in extent order."""

  path: str
  primitive: str
  trips: list[int | None]
  work: list[int]


def _scalar_value(val: Any) -> int | bool | None:
  """Python int or bool for a scalar literal. Arrays and tracers stay unknown."""
  if isinstance(val, bool):
    return val
  if isinstance(val, int):
    return val
  if isinstance(val, jax.Array):
    return None
  dtype = getattr(val, "dtype", None)
  if dtype is None or getattr(val, "ndim", 0) != 0:
    return None
  label = str(getattr(dtype, "name", dtype))
  if label == "bool":
    return bool(val)
  if "int" in label:
    return int(val)
  return None


def _atom_value(env: dict[int, int | bool], atom: Any) -> int | bool | None:
  """Value of a jaxpr atom: a literal, or a var already folded into ``env``."""
  if hasattr(atom, "val") and hasattr(atom, "aval"):
    literal = _scalar_value(atom.val)
    if literal is not None:
      return literal
  return env.get(id(atom))


def _is_int(val: object) -> bool:
  return isinstance(val, int) and not isinstance(val, bool)


def _fold_scalar(
  name: str, vals: list[int | bool | None], params: dict[str, Any]
) -> int | bool | None:
  """Fold one scalar integer or boolean eqn. Unknown or non-scalar inputs stay unknown."""
  if not vals or any(val is None for val in vals):
    return None
  if name in {"add", "sub", "mul", "max", "min"} and _is_int(vals[0]) and _is_int(vals[1]):
    left, right = int(vals[0]), int(vals[1])
    if name == "add":
      return left + right
    if name == "sub":
      return left - right
    if name == "mul":
      return left * right
    if name == "max":
      return max(left, right)
    return min(left, right)
  if name in {"div", "rem"} and _is_int(vals[0]) and _is_int(vals[1]):
    left, right = int(vals[0]), int(vals[1])
    if right == 0 or left < 0 or right < 0:
      return None
    return left // right if name == "div" else left % right
  if name == "neg" and _is_int(vals[0]):
    return -int(vals[0])
  if name == "sign" and _is_int(vals[0]):
    value = int(vals[0])
    return (value > 0) - (value < 0)
  if name in {"eq", "ne", "lt", "le", "gt", "ge"} and len(vals) == 2:
    left, right = vals[0], vals[1]
    same_int = _is_int(left) and _is_int(right)
    same_bool = isinstance(left, bool) and isinstance(right, bool)
    if not same_int and not same_bool:
      return None
    compared = int(left) if same_int else bool(left)
    other = int(right) if same_int else bool(right)
    return {
      "eq": compared == other,
      "ne": compared != other,
      "lt": compared < other,
      "le": compared <= other,
      "gt": compared > other,
      "ge": compared >= other,
    }[name]
  if name == "and" and len(vals) == 2:
    if isinstance(vals[0], bool) and isinstance(vals[1], bool):
      return bool(vals[0] and vals[1])
    if _is_int(vals[0]) and _is_int(vals[1]):
      return int(vals[0]) & int(vals[1])
  if name == "or" and len(vals) == 2:
    if isinstance(vals[0], bool) and isinstance(vals[1], bool):
      return bool(vals[0] or vals[1])
    if _is_int(vals[0]) and _is_int(vals[1]):
      return int(vals[0]) | int(vals[1])
  if name == "select_n" and isinstance(vals[0], bool) and len(vals) == 3:
    return vals[2] if vals[0] else vals[1]
  if name == "select_n" and _is_int(vals[0]):
    index = int(vals[0])
    cases = vals[1:]
    if 0 <= index < len(cases):
      return cases[index]
  if name == "convert_element_type" and (isinstance(vals[0], bool) or _is_int(vals[0])):
    dtype = params.get("new_dtype")
    if dtype is None:
      return None
    try:
      integer = bool(jnp.issubdtype(jnp.dtype(dtype), jnp.integer))
    except (TypeError, ValueError):
      return None
    if integer:
      return int(vals[0])
  return None


def _open_jaxpr(closed: Any) -> Any:
  inner = getattr(closed, "jaxpr", None)
  if inner is not None and inner is not closed and hasattr(inner, "eqns"):
    return inner
  return closed


def _seed_call(eqn: Any, inner: Any, env: dict[int, int | bool]) -> dict[int, int | bool]:
  """Bind a called jaxpr's invars to the scalar values of the call operands."""
  closed = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
  consts = list(getattr(closed, "consts", ()) or ())
  const_vals = [_scalar_value(const) for const in consts]
  arg_vals = [_atom_value(env, atom) for atom in eqn.invars]
  if len(inner.invars) == len(arg_vals):
    bound = arg_vals
  elif len(inner.invars) == len(const_vals) + len(arg_vals):
    bound = const_vals + arg_vals
  else:
    return {}
  seeded: dict[int, int | bool] = {}
  for var, val in zip(inner.invars, bound, strict=True):
    if val is not None:
      seeded[id(var)] = val
  return seeded


def _loop_body_jaxpr(eqn: Any) -> Any | None:
  key = "body_jaxpr" if eqn.primitive.name == "while" else "jaxpr"
  closed = eqn.params.get(key)
  if closed is None:
    return None
  return _open_jaxpr(closed)


def _seed_while_body(eqn: Any, body: Any, env: dict[int, int | bool]) -> dict[int, int | bool]:
  n_cond = int(eqn.params["cond_nconsts"])
  n_body = int(eqn.params["body_nconsts"])
  seeded: dict[int, int | bool] = {}
  for offset, var in enumerate(body.invars[:n_body]):
    val = _atom_value(env, eqn.invars[n_cond + offset])
    if val is not None:
      seeded[id(var)] = val
  return seeded


def _carry_index(invars: Sequence[Any], atom: Any, n_const: int) -> int | None:
  for index, var in enumerate(invars):
    if var is atom and index >= n_const:
      return index - n_const
  return None


def _literal_one(atom: Any) -> bool:
  return hasattr(atom, "val") and hasattr(atom, "aval") and _scalar_value(atom.val) == 1


def _body_is_unit_step(body: Any, n_body: int, index_carry: int, upper_carry: int) -> bool:
  """``fori`` body: the index carry becomes ``index + 1`` and the upper carry is forwarded."""
  if n_body + max(index_carry, upper_carry) >= len(body.invars):
    return False
  if max(index_carry, upper_carry) >= len(body.outvars):
    return False
  if body.outvars[upper_carry] is not body.invars[n_body + upper_carry]:
    return False
  index_in = body.invars[n_body + index_carry]
  index_out = body.outvars[index_carry]
  for eqn in body.eqns:
    if not any(out is index_out for out in eqn.outvars):
      continue
    if eqn.primitive.name != "add" or len(eqn.invars) != 2:
      return False
    left, right = eqn.invars
    return (left is index_in and _literal_one(right)) or (right is index_in and _literal_one(left))
  return False


def _static_fori_trip(eqn: Any, env: dict[int, int | bool]) -> int | None:
  """Trip count of a ``fori_loop``-shaped ``while`` whose bounds are static integers.

  ``fori_loop`` with a traced upper bound lowers to ``while`` whose cond is
  ``index < upper`` and whose body is ``index + 1``, forwarding ``upper``.
  ``upper - index`` is the trip count when both are non-negative integer
  constants. Any other ``while`` stays unmeasured.
  """
  cond = _open_jaxpr(eqn.params["cond_jaxpr"])
  body = _open_jaxpr(eqn.params["body_jaxpr"])
  n_cond = int(eqn.params["cond_nconsts"])
  n_body = int(eqn.params["body_nconsts"])
  cond_env: dict[int, int | bool] = {}
  for index, var in enumerate(cond.invars):
    src = index if index < n_cond else index + n_body
    if src >= len(eqn.invars):
      return None
    val = _atom_value(env, eqn.invars[src])
    if val is not None:
      cond_env[id(var)] = val
  for ceqn in cond.eqns:
    folded = _fold_scalar(
      ceqn.primitive.name,
      [_atom_value(cond_env, atom) for atom in ceqn.invars],
      ceqn.params,
    )
    if folded is not None and len(ceqn.outvars) == 1:
      cond_env[id(ceqn.outvars[0])] = folded
  if len(cond.outvars) != 1:
    return None
  producer = next(
    (ceqn for ceqn in cond.eqns if any(out is cond.outvars[0] for out in ceqn.outvars)),
    None,
  )
  if producer is None or producer.primitive.name != "lt" or len(producer.invars) != 2:
    return None
  lhs, rhs = producer.invars
  index_carry = _carry_index(cond.invars, lhs, n_cond)
  upper_carry = _carry_index(cond.invars, rhs, n_cond)
  if index_carry is None or upper_carry is None:
    return None
  if not _body_is_unit_step(body, n_body, index_carry, upper_carry):
    return None
  lower_src = n_cond + n_body + index_carry
  upper_src = n_cond + n_body + upper_carry
  if upper_src >= len(eqn.invars):
    return None
  lower = _atom_value(env, eqn.invars[lower_src])
  upper = _atom_value(env, eqn.invars[upper_src])
  if not _is_int(lower) or not _is_int(upper) or int(lower) < 0 or int(upper) < 0:
    return None
  return max(int(upper) - int(lower), 0)


def _walk_static_trips(
  jaxpr: Any,
  path: str,
  env: dict[int, int | bool],
  found: dict[str, int],
) -> list[int | bool | None]:
  """Walk ``jaxpr`` with ``loop_bodies`` path strings, recording static ``fori`` trips."""
  local = dict(env)
  for idx, eqn in enumerate(jaxpr.eqns):
    name = eqn.primitive.name
    here = f"{path}/{idx}:{name}"
    if name in {"scan", "while"}:
      if name == "while":
        trip = _static_fori_trip(eqn, local)
        if trip is not None:
          found[here] = trip
      body = _loop_body_jaxpr(eqn)
      if body is not None:
        body_env = _seed_while_body(eqn, body, local) if name == "while" else {}
        _walk_static_trips(body, here, body_env, found)
      continue
    subs = list(sub_jaxprs(eqn.params))
    if subs:
      returned: list[int | bool | None] | None = None
      for sub_index, sub in enumerate(subs):
        outs = _walk_static_trips(sub, f"{here}[{sub_index}]", _seed_call(eqn, sub, local), found)
        if returned is None:
          returned = outs
      if len(subs) == 1 and returned is not None and len(returned) == len(eqn.outvars):
        for var, val in zip(eqn.outvars, returned, strict=True):
          if val is not None:
            local[id(var)] = val
      continue
    folded = _fold_scalar(
      name,
      [_atom_value(local, atom) for atom in eqn.invars],
      eqn.params,
    )
    if folded is not None and len(eqn.outvars) == 1:
      local[id(eqn.outvars[0])] = folded
  return [_atom_value(local, var) for var in jaxpr.outvars]


def _static_fori_trips(fn: Callable[..., Any], args: tuple[Any, ...]) -> dict[str, int]:
  """Static ``fori_loop`` trip counts keyed by the same path ``loop_bodies`` uses.

  A second trace after ``loop_bodies`` hits ``make_jaxpr``'s cache. Scalar
  integer arguments seed the root invars, so ``fori_loop(0, n)(3)`` reports 3
  even when ``n`` is not a literal inside the jaxpr. Computed bounds
  (``iterations * L``, tiling's chunk count) fold when every operand is a
  static integer.
  """
  closed = jax.make_jaxpr(fn)(*args)
  jaxpr = closed.jaxpr
  flat, _ = jax.tree_util.tree_flatten(args)
  env: dict[int, int | bool] = {}
  if len(flat) == len(jaxpr.invars):
    for var, leaf in zip(jaxpr.invars, flat, strict=True):
      val = _scalar_value(leaf)
      if val is not None:
        env[id(var)] = val
  found: dict[str, int] = {}
  _walk_static_trips(jaxpr, "", env, found)
  return found


def _rows_from_bodies(
  found: list[list[Any]],
  extents: Sequence[int],
  static_trips: Sequence[dict[str, int]] | None = None,
) -> list[LoopRow]:
  paths = [[body.path for body in row] for row in found]
  if any(row != paths[0] for row in paths[1:]):
    msg = f"loop structure differs across extents {list(extents)}: {paths}"
    raise LoopStructureMismatchError(msg)
  rows: list[LoopRow] = []
  for index, body in enumerate(found[0]):
    trips: list[int | None] = []
    for extent_index, row in enumerate(found):
      trip = row[index].trip_count
      recovered = None if static_trips is None else static_trips[extent_index].get(row[index].path)
      if trip is None and row[index].primitive == "while" and recovered is not None:
        trip = recovered
      trips.append(trip)
    rows.append(
      LoopRow(
        path=str(body.path),
        primitive=str(body.primitive),
        trips=trips,
        work=[int(row[index].iteration_work) for row in found],
      )
    )
  return rows


def loop_work_by_extent(
  fn: Callable[..., Any],
  make_args: Callable[[int], tuple[Any, ...]],
  extents: Sequence[int],
) -> list[LoopRow]:
  """Trace ``fn(*make_args(extent))`` at each extent and align loops by path.

  ``jax.clear_caches`` runs first: a cached ``filter_jit`` jaxpr would hide a
  monkeypatched body of the same shape. Path mismatch across extents is an
  error, not a skip. The extent has to arrive as an argument of ``fn``. A
  no-argument wrapper that closes over the extent is cached by ``make_jaxpr``
  and every extent then reports the first trace (``measure_rows`` avoids that
  by tracing a fresh callable per extent).

  Args:
    fn: Program to trace. Never executed.
    make_args: Maps an extent to the positional arguments of ``fn``.
    extents: Two or more increasing sizes of the swept axis.

  Returns:
    One row per loop, outermost first, the same order ``loop_bodies`` uses.
  """
  if len(extents) < 2:
    msg = f"need at least two extents, got {list(extents)}"
    raise ValueError(msg)
  jax.clear_caches()
  found: list[list[Any]] = []
  static_trips: list[dict[str, int]] = []
  for extent in extents:
    args = make_args(int(extent))
    found.append(loop_bodies(fn, *args))
    static_trips.append(_static_fori_trips(fn, args))
  return _rows_from_bodies(found, extents, static_trips)


def per_extent(
  trace_at: Callable[[int], tuple[Callable[..., Any], tuple[Any, ...]]],
) -> tuple[Callable[..., Any], Callable[[int], tuple[Any, ...]]]:
  """Adapt a per-extent ``(fn, args)`` builder to ``loop_work_by_extent``.

  Mutants close over the extent-sized tensor, so the traced callable has to be
  rebuilt at every extent (the shape used by the graded probe). ``make_args``
  stores that callable; ``fn`` forwards to it. Call ``make_args`` before ``fn``.
  """
  box: dict[str, Callable[..., Any]] = {}

  def fn(*args: Any, **kwargs: Any) -> Any:
    return box["fn"](*args, **kwargs)

  def make_args(extent: int) -> tuple[Any, ...]:
    built, args = trace_at(int(extent))
    box["fn"] = built
    return args

  return fn, make_args


def measure_rows(
  trace_at: Callable[[int], tuple[Callable[..., Any], tuple[Any, ...]]],
  extents: Sequence[int],
) -> list[LoopRow]:
  """Trace a fresh callable from ``trace_at`` at each extent.

  ``make_jaxpr`` caches a no-argument function, so a mutant that closes over
  the extent-sized tensor has to be a new function object at every extent (the
  shape the graded probe traces).
  """
  if len(extents) < 2:
    msg = f"need at least two extents, got {list(extents)}"
    raise ValueError(msg)
  jax.clear_caches()
  found: list[list[Any]] = []
  static_trips: list[dict[str, int]] = []
  for extent in extents:
    fn, args = trace_at(int(extent))
    found.append(loop_bodies(fn, *args))
    static_trips.append(_static_fori_trips(fn, args))
  return _rows_from_bodies(found, extents, static_trips)


def work_ratios(row: LoopRow) -> list[float]:
  """Per-iteration work ratios between consecutive extents."""
  ratios: list[float] = []
  for small, large in zip(row.work, row.work[1:], strict=False):
    if small <= 0 or large <= 0:
      msg = f"non-positive iteration work {row.work} at {row.path}"
      raise ValueError(msg)
    ratios.append(large / small)
  return ratios


def trips_scale(trips: Sequence[int | None], threshold: float = TRIP_SCALING_RATIO) -> bool:
  """Whether ``trips`` grow with the swept axis.

  A ``None`` count (a ``while``, or an unmeasured trip count) counts as
  scaling. A static count scales only when every consecutive ratio reaches
  ``threshold``. A binary-search scan (one extra trip when the extent doubles)
  stays below the threshold and is exempt from criterion A.
  """
  if len(trips) < 2:
    return False
  saw = False
  for small, large in zip(trips, trips[1:], strict=False):
    saw = True
    if small is None or large is None:
      continue
    if small <= 0 or large / small < threshold:
      return False
  return saw


def scaling_summary(rows: Sequence[LoopRow]) -> str:
  """One block of path, trips, work and ratios for an assertion message."""
  lines: list[str] = []
  for row in rows:
    lines.append(
      f"{row.path or '<root>'} {row.primitive} trips={row.trips} work={row.work} "
      f"ratios={work_ratios(row)}"
    )
  return "\n".join(lines)


def independence_violations(
  rows: Sequence[LoopRow],
  *,
  only_trips_scaling: bool = True,
  threshold: float = A_THRESHOLD,
  require_any: bool = True,
) -> list[str]:
  """Criterion A failures: trip-scaling loops whose per-iteration work grows.

  Args:
    rows: Loops aligned across the same extents.
    only_trips_scaling: Skip loops whose trip count does not scale. That is the
      ``searchsorted`` exemption.
    threshold: A ratio at or above this fails.
    require_any: When true, zero trip-scaling loops is itself a violation, so a
      refactor that deletes the loop cannot pass silently.

  Returns:
    Human-readable violations. Empty means the rows pass criterion A.
  """
  violations: list[str] = []
  checked = 0
  for row in rows:
    if only_trips_scaling and not trips_scale(row.trips, threshold):
      continue
    checked += 1
    for ratio in work_ratios(row):
      if ratio >= threshold:
        violations.append(
          f"{row.path or '<root>'} {row.primitive} work ratio {ratio:.4f} "
          f">= {threshold} trips={row.trips} work={row.work}"
        )
  if require_any and checked == 0:
    violations.append("no trip-scaling loop; the guard would be vacuous")
  return violations


def assert_work_independent(
  rows: Sequence[LoopRow],
  *,
  only_trips_scaling: bool = True,
  threshold: float = A_THRESHOLD,
  require_any: bool = True,
) -> None:
  """Assert criterion A. Raises ``AssertionError`` listing every failing loop."""
  violations = independence_violations(
    rows,
    only_trips_scaling=only_trips_scaling,
    threshold=threshold,
    require_any=require_any,
  )
  assert not violations, scaling_summary(rows) + "\n" + "\n".join(violations)


def linearity_violations(
  rows: Sequence[LoopRow],
  *,
  threshold: float = B_THRESHOLD,
) -> list[str]:
  """Criterion B failures: fixed-trip work above ``threshold``, or trips that move.

  Returns:
    Human-readable violations. Empty means the rows pass criterion B.
  """
  if not rows:
    return ["no loop; the guard would be vacuous"]
  violations: list[str] = []
  for row in rows:
    if any(trip != row.trips[0] for trip in row.trips):
      violations.append(f"trips must not depend on L: {row.path} trips={row.trips}")
    for ratio in work_ratios(row):
      if ratio >= threshold:
        violations.append(
          f"{row.path or '<root>'} {row.primitive} work ratio {ratio:.4f} "
          f">= {threshold} trips={row.trips} work={row.work}"
        )
  return violations


def assert_work_at_most_linear(
  rows: Sequence[LoopRow],
  *,
  threshold: float = B_THRESHOLD,
) -> None:
  """Assert criterion B. Raises ``AssertionError`` listing every failing loop."""
  violations = linearity_violations(rows, threshold=threshold)
  assert not violations, scaling_summary(rows) + "\n" + "\n".join(violations)


def pick_outermost_while(rows: Sequence[LoopRow]) -> LoopRow:
  """First ``while`` (outermost; rows are parent-before-child)."""
  for row in rows:
    if row.primitive == "while":
      return row
  msg = f"no while loop among {[row.primitive for row in rows]}"
  raise LookupError(msg)


def _expected_trip_list(expected_trips: Sequence[int]) -> list[int]:
  return [int(trip) for trip in expected_trips]


def pick_trips_scaling(rows: Sequence[LoopRow], *, expected_trips: Sequence[int]) -> LoopRow:
  """Outermost static-doubling loop, which must have ``expected_trips``.

  The guarded loop is named by its trip counts. Two substitutions are refused.
  A loop that still doubles, but at the wrong counts, is not replaced by an
  inner loop that happens to have ``expected_trips``. A heavier loop listed
  earlier (it contains the candidate) whose trips no longer double is not
  skipped in favour of that inner loop.
  """
  expected = _expected_trip_list(expected_trips)
  chosen: LoopRow | None = None
  for row in rows:
    if any(trip is None or trip <= 0 for trip in row.trips):
      continue
    if not trips_scale(row.trips):
      continue
    ratios = [
      large / small
      for small, large in zip(row.trips, row.trips[1:], strict=False)
      if small is not None and large is not None and small > 0
    ]
    if ratios and all(ratio >= TRIP_SCALING_RATIO for ratio in ratios):
      chosen = row
      break
  if chosen is None:
    msg = f"no trip-scaling loop among {[row.trips for row in rows]}"
    raise LookupError(msg)
  if list(chosen.trips) != expected:
    msg = (
      f"outermost trip-scaling loop {chosen.path} has trips {chosen.trips}, "
      f"not {expected}; refusing to select a different loop"
    )
    raise LookupError(msg)
  for row in rows:
    if row is chosen:
      break
    if row.primitive in {"scan", "while"} and row.work[0] > chosen.work[0]:
      msg = (
        f"refusing to substitute {chosen.path} for heavier loop {row.path} "
        f"trips={row.trips} work={row.work[0]}"
      )
      raise LookupError(msg)
  return chosen


def pick_fixed_scan_largest(rows: Sequence[LoopRow], *, expected_trips: Sequence[int]) -> LoopRow:
  """Largest fixed-trip scan, which must have ``expected_trips``.

  Scans whose trip count changes across extents are not candidates. If one of
  those has at least as much per-iteration work as the selected scan, the
  intended loop changed shape and this raises instead of returning a smaller
  fixed scan that still has constant trips.
  """
  expected = _expected_trip_list(expected_trips)
  best: LoopRow | None = None
  best_work = -1
  varying: list[LoopRow] = []
  for row in rows:
    if row.primitive != "scan" or row.trips[0] is None:
      continue
    if any(trip != row.trips[0] for trip in row.trips):
      varying.append(row)
      continue
    if row.work[0] > best_work:
      best = row
      best_work = row.work[0]
  if best is None:
    msg = "no fixed-trip scan"
    raise LookupError(msg)
  if list(best.trips) != expected:
    msg = (
      f"largest fixed scan {best.path} has trips {best.trips}, not {expected}; "
      "refusing to select a different loop"
    )
    raise LookupError(msg)
  heavier = [row for row in varying if row.work[0] >= best.work[0]]
  if heavier:
    detail = ", ".join(f"{row.path} trips={row.trips} work={row.work[0]}" for row in heavier)
    msg = f"refusing to substitute a smaller fixed scan for a loop whose trips changed: {detail}"
    raise LookupError(msg)
  return best


def _count_primitive(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive:
      total += 1
    for sub in sub_jaxprs(eqn.params):
      total += _count_primitive(sub, primitive)
  return total


def _count_in_loops(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in {"scan", "while"}:
      for sub in sub_jaxprs(eqn.params):
        total += _count_primitive(sub, primitive)
    else:
      for sub in sub_jaxprs(eqn.params):
        total += _count_in_loops(sub, primitive)
  return total


def _aval_numel(aval: Any) -> int:
  shape = getattr(aval, "shape", ())
  return math.prod(int(dim) for dim in shape)


def _numel_of_primitive(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive:
      total += sum(_aval_numel(var.aval) for var in eqn.outvars)
    for sub in sub_jaxprs(eqn.params):
      total += _numel_of_primitive(sub, primitive)
  return total


def _numel_in_loops(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in {"scan", "while"}:
      for sub in sub_jaxprs(eqn.params):
        total += _numel_of_primitive(sub, primitive)
    else:
      for sub in sub_jaxprs(eqn.params):
        total += _numel_in_loops(sub, primitive)
  return total


def count_primitive_in_loops(
  fn: Callable[..., Any],
  args: tuple[Any, ...],
  primitive: str,
) -> int:
  """How many times ``primitive`` appears inside any ``scan`` or ``while`` body.

  The walk enters ``pjit`` and ``cond`` so a marked call nested in the pair-map
  kernel is visible. Marks outside every loop are ignored.

  ``jax.clear_caches`` runs first so a jaxpr cached by an earlier test cannot
  change this count.
  """
  jax.clear_caches()
  closed = jax.make_jaxpr(fn)(*args)
  jaxpr = closed.jaxpr
  return _count_in_loops(jaxpr, primitive)


def assert_no_primitive_in_loops(
  fn: Callable[..., Any],
  args: tuple[Any, ...],
  primitive: str,
) -> None:
  """Assert ``primitive`` does not run inside a loop body."""
  count = count_primitive_in_loops(fn, args, primitive)
  assert count == 0, f"primitive {primitive} ran inside a loop ({count} times)"


def primitive_numel_in_loops(
  fn: Callable[..., Any],
  args: tuple[Any, ...],
  primitive: str,
) -> int:
  """Total output elements of ``primitive`` inside any ``scan`` or ``while``.

  A ``vmap`` over the marked call is one equation whose output shape includes
  the batch axis, so the element count grows with that axis even though the
  equation is not replicated. Marks outside every loop are ignored.

  ``jax.clear_caches`` runs first so a jaxpr cached by an earlier test cannot
  change this count.
  """
  jax.clear_caches()
  closed = jax.make_jaxpr(fn)(*args)
  return _numel_in_loops(closed.jaxpr, primitive)


def assert_mark_numel_independent(
  counts: Sequence[int],
  *,
  threshold: float = A_THRESHOLD,
) -> None:
  """Assert marked output volume inside loops does not grow across extents.

  Criterion A on whole-loop work misses a fold whose tensor is small next to
  the rest of the body. The mark's own output size is the measurement that
  sees it. Zero at any extent is a violation: the mark never reached a loop.
  """
  if len(counts) < 2 or any(count <= 0 for count in counts):
    msg = f"encoder mark numel inside loops is missing: {list(counts)}"
    raise AssertionError(msg)
  violations: list[str] = []
  for small, large in zip(counts, counts[1:], strict=False):
    ratio = large / small
    if ratio >= threshold:
      violations.append(f"mark numel ratio {ratio:.4f} >= {threshold} numel={list(counts)}")
  assert not violations, "\n".join(violations)


def fold_over(fn: Callable[..., Any], n: int, arg_index: int) -> Callable[..., Any]:
  """Re-run ``fn`` on ``n`` perturbed copies and keep element 0.

  This is the shape of the autoregressive bug, and the only mutation that moves
  the work ratio (A12). A small injected einsum changes the proxy by about a
  tenth of a percent and does not clear the must-fire margin. The index is
  added into argument ``arg_index`` so the ``vmap`` reaches the body: a ``vmap``
  that ignores its index is deleted by ``make_jaxpr`` when only element 0 is
  kept, and the fold then costs nothing.

  Args:
    fn: Body to recompute. Called with the same positional arguments.
    n: Number of copies. Use the swept extent.
    arg_index: Which positional argument receives ``+ index``.

  Returns:
    A function with ``fn``'s positional signature.
  """

  def folded(*args: Any) -> Any:
    def one(index: jax.Array) -> Any:
      replaced = list(args)

      def shift(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.number):
          return leaf + index.astype(leaf.dtype)
        return leaf

      replaced[arg_index] = jax.tree.map(shift, replaced[arg_index])
      return fn(*replaced)

    outs = jax.vmap(one)(jnp.arange(n))
    return jax.tree.map(lambda leaf: leaf[0], outs)

  return folded
