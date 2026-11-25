"""score_pairs loop complexity (debt #1981, resolved Q2).

Three guards on the split path (``make_score_split_fns``). The averaged path
re-encodes by design and is not traced here.

(a) Criterion A, separately over K with C fixed and over C with K fixed.
    Per-iteration pair work must not grow with the pair count. Each axis has
    its own whole-body fold mutant (re-run the candidate kernel over every mask,
    or over every sequence).
(b) Criterion B over L: the same loop's per-iteration work stays linear.
(c) The encoder does not run inside the pair map. Criterion A is blind to a
    per-pair re-encode (the K ratio stays ~1 while the body inflates), so this
    guard counts an encoder mark inside the scan. The must-fire mutant calls
    the model once per pair.

Work is a jaxpr proxy, not wall time. The score module is imported by
``importlib`` because the package re-exports a ``score`` function that shadows
the submodule. Mutants are installed with ``monkeypatch``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from aminx.model.mpnn import Aminx
from tests.helpers.aminx_small import make_ar_model, structure
from tests.helpers.loop_scaling import (
  A_MUTANT_MIN,
  A_THRESHOLD,
  B_THRESHOLD,
  ENCODER_MARK,
  LoopRow,
  assert_no_primitive_in_loops,
  assert_work_at_most_linear,
  assert_work_independent,
  count_primitive_in_loops,
  mark_array,
  measure_rows,
  per_extent,
  pick_fixed_scan_largest,
  pick_trips_scaling,
  work_ratios,
)

_MODEL = make_ar_model()
_SENTINEL = {"score_k": 0, "score_c": 0, "reencode": 0}
_SCORE_AXIS: dict[str, jax.Array] = {}


@pytest.fixture(autouse=True)
def _mark_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
  """Tag ``Aminx.__call__`` so a per-pair re-encode shows up inside the pair scan.

  The decoder reads ``model.decoder`` rather than ``model(...)``, so the mark
  stays off the clean pair map. A wrapper module would hide ``.decoder``.
  """
  original = Aminx.__call__

  def marked(self: Aminx, *args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    nodes, edges, neighbors = original(self, *args, **kwargs)
    scale = jnp.float32(0) * jnp.sum(mark_array(nodes))
    return nodes + scale, edges, neighbors

  monkeypatch.setattr(Aminx, "__call__", marked)


def _inputs(length: int, n_masks: int, n_seq: int) -> tuple[Any, ...]:
  coords = structure(length)["coords"]
  mask = jnp.ones((length,), dtype=jnp.float32)
  residue_index = jnp.arange(length, dtype=jnp.int32)
  chain_index = jnp.zeros((length,), dtype=jnp.int32)
  sequences = jnp.zeros((n_seq, length), dtype=jnp.int32)
  masks = jnp.broadcast_to(1.0 - jnp.eye(length, dtype=jnp.float32), (n_masks, length, length))
  keys = jax.random.split(jax.random.PRNGKey(0), n_seq)
  _SCORE_AXIS["masks"] = masks
  _SCORE_AXIS["sequences"] = sequences
  return coords, mask, residue_index, chain_index, sequences, masks, keys


def _zeros_like(shaped: Any) -> Any:
  def one(leaf: Any) -> Any:
    if leaf is None or not hasattr(leaf, "shape"):
      return leaf
    return jnp.zeros(leaf.shape, leaf.dtype)

  return jax.tree.map(one, shaped, is_leaf=lambda leaf: leaf is None or hasattr(leaf, "shape"))


def _trace(extent: int, *, mode: str) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  """``extent`` is K, C, or L depending on ``mode``. Encoding is supplied, not computed."""
  if mode == "k":
    length, n_masks, n_seq = 24, extent, 2
  elif mode == "c":
    length, n_masks, n_seq = 24, 2, extent
  else:
    length, n_masks, n_seq = extent, 2, 2
  coords, mask, residue_index, chain_index, sequences, masks, keys = _inputs(length, n_masks, n_seq)
  score_mod = importlib.import_module("aminx.scoring.score")
  fns = score_mod.make_score_split_fns(_MODEL)
  shaped = jax.eval_shape(fns.encode_structure, coords, mask, residue_index, chain_index)
  enc = _zeros_like(shaped)

  def fn() -> Any:
    return fns.score_pairs(
      keys,
      sequences,
      masks,
      enc,
      coords,
      mask,
      residue_index,
      chain_index,
      chunk_size=2,
    )

  return fn, ()


def _rows(mode: str, extents: tuple[int, ...]) -> list[LoopRow]:
  return measure_rows(lambda n: _trace(n, mode=mode), extents)


def _first_of(outs: Any) -> tuple[Any, Any, Any]:
  return outs[0][0], outs[1][0], outs[2][0]


def _mask_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    _SENTINEL["score_k"] += 1
    masks = _SCORE_AXIS["masks"]

    def one(mask: jax.Array) -> Any:
      replaced = list(args)
      replaced[-1] = mask
      return real(*replaced)

    return _first_of(jax.vmap(one)(masks))

  return bad


def _seq_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    _SENTINEL["score_c"] += 1
    one_hot = jax.nn.one_hot(_SCORE_AXIS["sequences"], 21)

    def one(sequence: jax.Array) -> Any:
      replaced = list(args)
      replaced[3] = sequence
      return real(*replaced)

    return _first_of(jax.vmap(one)(one_hot))

  return bad


def _reencode(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    _SENTINEL["reencode"] += 1
    model, coords, mask, residue_index, chain_index = args[0], args[5], args[6], args[7], args[8]
    if getattr(coords, "ndim", 0) == 4:
      coords = coords[0]
      mask = mask[0]
      residue_index = residue_index[0]
      chain_index = chain_index[0]
    node, edge, neigh = model(coords, mask, residue_index, chain_index, inference=True)
    nll, logits, order = real(*args)
    bump = jnp.float32(0) * (jnp.sum(node) + jnp.sum(edge) + jnp.sum(neigh.astype(jnp.float32)))
    return nll + bump, logits, order

  return bad


def _assert_pair_axis(mode: str, extents: tuple[int, ...]) -> LoopRow:
  rows = _rows(mode, extents)
  row = pick_trips_scaling(rows)
  assert row.primitive == "scan"
  assert_work_independent([row])
  assert max(work_ratios(row)) < A_THRESHOLD
  return row


def test_score_pairs_work_independent_of_k() -> None:
  _assert_pair_axis("k", (5, 10, 20))


def test_score_pairs_work_independent_of_c() -> None:
  _assert_pair_axis("c", (5, 10, 20))


def test_score_pairs_work_at_most_linear_in_length() -> None:
  rows = _rows("l", (32, 64, 128))
  row = pick_fixed_scan_largest(rows)
  assert row.primitive == "scan"
  assert row.trips[0] == row.trips[1] == row.trips[2]
  assert_work_at_most_linear([row])
  assert max(work_ratios(row)) < B_THRESHOLD


def _assert_axis_mutant(clean: list[LoopRow], mutant: list[LoopRow], sentinel: str) -> None:
  assert _SENTINEL[sentinel] > 0
  clean_row = pick_trips_scaling(clean)
  mutant_row = pick_trips_scaling(mutant)
  clean_ratios = work_ratios(clean_row)
  mutant_ratios = work_ratios(mutant_row)
  assert max(clean_ratios) < A_THRESHOLD
  assert min(mutant_ratios) >= A_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_independent([mutant_row])


def test_score_pairs_mask_fold_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  clean = _rows("k", (5, 10, 20))
  score_mod = importlib.import_module("aminx.scoring.score")
  monkeypatch.setattr(score_mod, "_score_candidate_kernel", _mask_fold(score_mod._score_candidate_kernel))
  _SENTINEL["score_k"] = 0
  mutant = _rows("k", (5, 10, 20))
  _assert_axis_mutant(clean, mutant, "score_k")


def test_score_pairs_sequence_fold_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  clean = _rows("c", (5, 10, 20))
  score_mod = importlib.import_module("aminx.scoring.score")
  monkeypatch.setattr(score_mod, "_score_candidate_kernel", _seq_fold(score_mod._score_candidate_kernel))
  _SENTINEL["score_c"] = 0
  mutant = _rows("c", (5, 10, 20))
  _assert_axis_mutant(clean, mutant, "score_c")


def test_score_pairs_encoder_is_outside_the_pair_map() -> None:
  """Clean split path: the supplied encoding is reused, so the encoder mark is absent."""
  fn, make_args = per_extent(lambda n: _trace(n, mode="k"))
  args = make_args(5)
  assert count_primitive_in_loops(fn, args, ENCODER_MARK) == 0
  assert_no_primitive_in_loops(fn, args, ENCODER_MARK)


def test_score_pairs_per_pair_reencode_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  """A is blind to this mutant; the encoder mark inside the pair scan is what fires."""
  clean = _rows("k", (5, 10, 20))
  score_mod = importlib.import_module("aminx.scoring.score")
  monkeypatch.setattr(score_mod, "_score_candidate_kernel", _reencode(score_mod._score_candidate_kernel))
  _SENTINEL["reencode"] = 0
  mutant = _rows("k", (5, 10, 20))
  assert _SENTINEL["reencode"] > 0
  clean_row = pick_trips_scaling(clean)
  mutant_row = pick_trips_scaling(mutant)
  assert max(work_ratios(mutant_row)) < A_THRESHOLD
  assert mutant_row.work[0] / clean_row.work[0] >= A_MUTANT_MIN
  fn, make_args = per_extent(lambda n: _trace(n, mode="k"))
  args = make_args(5)
  assert count_primitive_in_loops(fn, args, ENCODER_MARK) > 0
  with pytest.raises(AssertionError, match=ENCODER_MARK):
    assert_no_primitive_in_loops(fn, args, ENCODER_MARK)
