"""Trace-only loop-complexity probe for debt #1981 (spec 261124, Task 0).

Reproduces the A5-A12 ratio tables with ``xtrax.profiling.loop_scaling``:
``loop_bodies`` for every in-scope loop (three extents) and
``extent_scaling_report`` for the instrument self-test. Both toys are scans, so
the report never falls through to eager ``while`` trip measurement.

Nothing in the graded paths is executed. ``--self-test`` checks the instrument
and does not trace aminx. Every exit path writes one JSON object that contains
every ``[result_schema]`` field.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from xtrax.profiling.loop_scaling import extent_scaling_report, loop_bodies
from xtrax.tiling import VmapIterator

from aminx.host.kernel_dispatch import _chunked_sample_decode, _dispatch_axis
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.decode.ste import STEDecode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.inference.optimize_ste import make_optimize_sequence_fn
from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi
from aminx.model.mpnn import Aminx
from aminx.tiling.strategy import Scan

logger = logging.getLogger("loop_complexity_probe")

A_CLEAN = (
  "tiling_clean",
  "tiling_multi_clean",
  "tiling_ragged_clean",
  "tiling_multi_ragged_clean",
  "score_k_clean",
  "score_c_clean",
  "seam_clean",
  "scan_clean",
  "encode_s_clean",
)
B_CLEAN = (
  "ste_untied_clean",
  "ste_tied_clean",
  "optimize_ste_clean",
  "score_l_clean",
  "encode_l_clean",
)
A_MUT = (
  "tiling_mut",
  "score_k_mut",
  "score_c_mut",
  "seam_mut",
  "scan_mut",
  "encode_s_mut",
)
B_MUT = (
  "ste_untied_mut",
  "ste_tied_mut",
  "optimize_ste_mut",
)
SENTINEL_NAMES = (
  "tiling",
  "score_k",
  "score_c",
  "reencode",
  "seam",
  "scan",
  "ste_untied",
  "ste_tied",
  "optimize_ste",
  "encode",
)
META_FLOATS = (
  "score_reencode_k_r1",
  "score_reencode_k_r2",
  "score_reencode_inflation",
  "self_test_o1_ratio",
  "self_test_fold_ratio",
  "elapsed_s",
)
RATIO_NAMES = A_CLEAN + B_CLEAN + A_MUT + B_MUT
FLOAT_KEYS = tuple(f"{name}_{which}" for name in RATIO_NAMES for which in ("r1", "r2")) + META_FLOATS
STRING_KEYS = ("mode", "refusal_reason", "xtrax_version", "jax_version", "loop_paths")
BOOL_KEYS = ("refused", "sentinels_fired")
SCHEMA_KEYS = set(FLOAT_KEYS) | set(STRING_KEYS) | set(BOOL_KEYS)

ORDER_CFG = (True, 0, None, None, "none")
SENTINEL: dict[str, int] = dict.fromkeys(SENTINEL_NAMES, 0)
_LIGAND: list[ProteinFeaturesLigand] = []
_MODEL: list[Aminx] = []
# Filled by each score trace so a patched kernel folds over THAT extent, not a fixed one.
_SCORE_AXIS: dict[str, jax.Array] = {}

TraceAt = Callable[[int], tuple[Callable[..., Any], tuple[Any, ...]]]
Pick = Callable[[list[list[Any]]], int]


@dataclass(frozen=True)
class Series:
  """Two consecutive per-iteration work ratios and the loop they came from."""

  r1: float
  r2: float
  path: str
  works: tuple[int, int, int]
  trips: tuple[Any, Any, Any]


def _versions() -> tuple[str, str]:
  import xtrax

  return str(xtrax.__version__), str(jax.__version__)


def blank_payload(*, mode: str) -> dict[str, Any]:
  """One schema-complete payload. Unmeasured ratios are -1, never null or NaN."""
  xtrax_version, jax_version = ("unknown", "unknown")
  try:
    xtrax_version, jax_version = _versions()
  except Exception:
    logger.exception("version probe failed")
  payload: dict[str, Any] = {key: -1.0 for key in FLOAT_KEYS}
  payload.update(
    {
      "mode": mode,
      "refused": False,
      "refusal_reason": "",
      "xtrax_version": xtrax_version,
      "jax_version": jax_version,
      "loop_paths": "",
      "sentinels_fired": False,
      "elapsed_s": 0.0,
    }
  )
  return payload


def _finite(payload: dict[str, Any]) -> None:
  bad = [key for key in FLOAT_KEYS if not math.isfinite(float(payload[key]))]
  if bad:
    msg = f"non-finite result fields: {bad}"
    raise RuntimeError(msg)
  missing = SCHEMA_KEYS - payload.keys()
  extra = payload.keys() - SCHEMA_KEYS
  if missing or extra:
    msg = f"schema drift missing={sorted(missing)} extra={sorted(extra)}"
    raise RuntimeError(msg)


def _emit_text(out: str, text: str) -> None:
  """Write text. ``/dev/stdout`` goes through sys.stdout (opening that path is ENXIO here)."""
  if out in {"/dev/stdout", "-"}:
    sys.stdout.write(text)
    sys.stdout.flush()
    return
  with open(out, "w", encoding="utf-8") as handle:
    handle.write(text)


def write_payload(payload: dict[str, Any], out: str) -> None:
  """Write payload to --out and to BTH_RESULTS_PATH when that env var is set."""
  _finite(payload)
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  _emit_text(out, text)
  mirror = os.environ.get("BTH_RESULTS_PATH")
  if mirror and mirror not in {out, "/dev/stdout", "-"}:
    _emit_text(mirror, text)


def _ratios(works: Sequence[int]) -> tuple[float, float]:
  if works[0] <= 0 or works[1] <= 0:
    msg = f"non-positive iteration work {works}"
    raise RuntimeError(msg)
  return works[1] / works[0], works[2] / works[1]


def measure(name: str, trace_at: TraceAt, extents: Sequence[int], pick: Pick) -> Series:
  """Trace ``trace_at`` at three extents and return the picked loop's ratios.

  ``filter_jit`` reuses a cached jaxpr for the same shapes, which would hide a
  monkeypatched body. Drop the cache at the start of every series.
  """
  jax.clear_caches()
  found: list[list[Any]] = []
  for extent in extents:
    fn, args = trace_at(int(extent))
    logger.info("tracing %s at extent %s", name, extent)
    found.append(loop_bodies(fn, *args))
  paths = [[body.path for body in row] for row in found]
  if any(row != paths[0] for row in paths[1:]):
    msg = f"{name}: loop structure differs across extents {list(extents)}: {paths}"
    raise RuntimeError(msg)
  if not paths[0]:
    msg = f"{name}: no scan/while in the trace"
    raise RuntimeError(msg)
  index = pick(found)
  works = tuple(int(row[index].iteration_work) for row in found)
  trips = tuple(row[index].trip_count for row in found)
  r1, r2 = _ratios(works)
  series = Series(r1, r2, str(found[0][index].path), works, trips)  # type: ignore[arg-type]
  logger.info(
    "%s ratios %.4f %.4f path=%s trips=%s works=%s",
    name,
    series.r1,
    series.r2,
    series.path,
    series.trips,
    series.works,
  )
  return series


def _put(payload: dict[str, Any], notes: list[str], name: str, series: Series) -> None:
  payload[f"{name}_r1"] = float(series.r1)
  payload[f"{name}_r2"] = float(series.r2)
  notes.append(
    f"{name} path={series.path} trips={series.trips} works={series.works} "
    f"ratios=({series.r1:.6f},{series.r2:.6f})"
  )


def pick_outermost_while(found: list[list[Any]]) -> int:
  """First ``while`` (outermost; ``loop_bodies`` lists parents before children)."""
  for index, body in enumerate(found[0]):
    if body.primitive == "while":
      return index
  msg = "no while loop"
  raise LookupError(msg)


def pick_trips_scaling(found: list[list[Any]]) -> int:
  """Outermost loop whose static trip count at least doubles at each step."""
  for index, body in enumerate(found[0]):
    trips = [row[index].trip_count for row in found]
    if any(trip is None or trip <= 0 for trip in trips):
      continue
    if trips[1] / trips[0] >= 1.5 and trips[2] / trips[1] >= 1.5:
      return index
  msg = f"no trip-scaling loop among {[body.trip_count for body in found[0]]}"
  raise LookupError(msg)


def pick_fixed_scan_largest(found: list[list[Any]]) -> int:
  """Fixed-trip scan with the largest per-iteration work (the body that contains the rest)."""
  best_index = -1
  best_work = -1
  for index, body in enumerate(found[0]):
    trips = [row[index].trip_count for row in found]
    if body.primitive != "scan" or trips[0] is None:
      continue
    if any(trip != trips[0] for trip in trips):
      continue
    if body.iteration_work > best_work:
      best_index = index
      best_work = int(body.iteration_work)
  if best_index < 0:
    msg = "no fixed-trip scan"
    raise LookupError(msg)
  return best_index


def _control_ratio(fn: Callable[..., Any], make_args: Callable[[int], tuple[Any, ...]], extent: int) -> float:
  """One 2x ratio from ``extent_scaling_report``, checked against ``loop_bodies``."""
  report = extent_scaling_report(fn, make_args, extent)
  small = loop_bodies(fn, *make_args(extent))
  large = loop_bodies(fn, *make_args(2 * extent))
  if [body.path for body in small] != [finding.path for finding in report.findings]:
    msg = "loop_bodies paths diverged from extent_scaling_report"
    raise RuntimeError(msg)
  if len(report.findings) != 1:
    msg = f"self-test expected one loop, found {len(report.findings)}"
    raise RuntimeError(msg)
  direct = large[0].iteration_work / small[0].iteration_work
  reported = float(report.findings[0].ratio)
  if not math.isfinite(reported) or abs(direct - reported) > 1e-6:
    msg = f"self-test ratio mismatch direct={direct} report={reported}"
    raise RuntimeError(msg)
  return reported


def self_test_ratios() -> tuple[float, float]:
  """O(1) scan body (ratio ~1, must not fire) and an (L, d) recompute (ratio ~2, must fire)."""

  def o1(xs: jax.Array) -> jax.Array:
    def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
      return carry + x, x

    carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
    return carry

  def fold(xs: jax.Array) -> jax.Array:
    width = xs.shape[1]

    def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
      recomputed = xs @ jnp.ones((width, width), dtype=xs.dtype)
      updated = carry + jnp.sum(x) + jnp.float32(0) * jnp.sum(recomputed)
      return updated, x

    carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
    return carry

  o1_ratio = _control_ratio(o1, lambda n: (jnp.ones((n,), dtype=jnp.float32),), 32)
  fold_ratio = _control_ratio(fold, lambda n: (jnp.ones((n, 16), dtype=jnp.float32),), 32)
  logger.info("self-test o1=%.4f fold=%.4f", o1_ratio, fold_ratio)
  return o1_ratio, fold_ratio


def controls_ok(payload: dict[str, Any]) -> bool:
  """Must-not-fire window around 1.0 and must-fire window around 2.0."""
  o1 = float(payload["self_test_o1_ratio"])
  fold = float(payload["self_test_fold_ratio"])
  return 0.8 < o1 < 1.2 and 1.7 <= fold < 2.5


def _ligand() -> ProteinFeaturesLigand:
  if not _LIGAND:
    _LIGAND.append(
      ProteinFeaturesLigand(
        node_features=128,
        edge_features=128,
        k_neighbors=8,
        ligand_l_chunk=-1,
        key=jax.random.PRNGKey(0),
      )
    )
  return _LIGAND[0]


def _model() -> Aminx:
  if not _MODEL:
    model = Aminx(
      node_features=64,
      edge_features=64,
      hidden_features=64,
      num_encoder_layers=2,
      num_decoder_layers=2,
      k_neighbors=5,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(0),
    )
    _MODEL.append(eqx.tree_inference(model, value=True))
  return _MODEL[0]


def _trace_tiling(extent: int, *, multi: bool, mutant: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  feat = _ligand()
  embed = feat._y_edges_coords_to_embed
  if multi:
    left = jnp.ones((extent, 2), dtype=jnp.float32)
    right = jnp.ones((extent, 5), dtype=jnp.float32)

    def synthetic(left_slab: jax.Array, right_slab: jax.Array) -> tuple[jax.Array, jax.Array]:
      return left_slab + jnp.float32(0.25), jnp.sin(right_slab)

    def traced_multi(left: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
      return map_chunks_axis0_multi(synthetic, 8, (left, right))

    return traced_multi, (left, right)

  y = jnp.ones((extent, 12, 3), dtype=jnp.float32)

  def clean(slab: jax.Array) -> jax.Array:
    return embed(slab)

  def bad(slab: jax.Array, y: jax.Array = y, embed: Any = embed) -> jax.Array:
    SENTINEL["tiling"] += 1
    return embed(slab) + jnp.float32(0) * jnp.sum(embed(y))

  fn = bad if mutant else clean

  def traced(y: jax.Array) -> jax.Array:
    return map_chunks_axis0(y, chunk_size=8, fn=fn)

  return traced, (y,)


def _ste_bundle(length: int, *, tied: bool) -> tuple[Any, Any]:
  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(1, length, 4, 3)), dtype=jnp.float32)
  mask = jnp.ones((1, length), dtype=jnp.float32)
  residue_index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (1, length))
  chain_index = jnp.zeros((1, length), dtype=jnp.int32)
  tokens = jnp.asarray(rng.integers(0, 20, size=(length,)), dtype=jnp.int32)
  tie = jnp.arange(length, dtype=jnp.int32)
  if tied:
    tie = tie.at[5].set(0).at[9].set(0).at[17].set(2)
  return build_inference_bundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    sequence=jax.nn.one_hot(tokens, 21),
    state_weights=jnp.ones((1,), dtype=jnp.float32),
    mode="score_conditional",
    tie_group_map=tie,
  )


class FoldConditional(eqx.Module):
  """Call the inner decode on ``n_fold`` copies and keep element 0 (D5 STE mutant)."""

  inner: ConditionalDecode
  n_fold: int = eqx.field(static=True)
  which: str = eqx.field(static=True)

  @property
  def model(self) -> Aminx:
    return self.inner.model

  def __call__(self, key: jax.Array, enc: Any, bundle: Any, config: Any, stage_set: Any) -> jax.Array:
    SENTINEL[self.which] += 1

    def one(index: jax.Array) -> jax.Array:
      # The index has to reach the body. A vmap that ignores it is deleted by
      # make_jaxpr when only element 0 is kept, and the fold then costs nothing.
      shifted = eqx.tree_at(
        lambda item: item.node_features,
        enc,
        enc.node_features + index.astype(enc.node_features.dtype),
      )
      return self.inner(key=key, enc=shifted, bundle=bundle, config=config, stage_set=stage_set)

    logits = jax.vmap(one)(jnp.arange(self.n_fold))
    return logits[0] + jnp.float32(0) * jnp.sum(logits)


class FoldEncoder(eqx.Module):
  """Re-run the whole per-state model ``n_fold`` times and keep copy 0.

  Spec D5 names a wrapper on ``model.encoder``. On this 64-d model the feature
  stage, not the encoder stack, is almost all of the jaxpr proxy, so folding
  only ``.encoder`` left the S-ratio at 1.001. Folding the model call is the
  same whole-body recompute and clears 1.7 (D5: strengthen the mutant).
  """

  wrapped: Aminx
  n_fold: int = eqx.field(static=True)

  def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    SENTINEL["encode"] += 1

    def one(index: jax.Array) -> tuple[Any, Any, Any]:
      shifted = args[0] + index.astype(args[0].dtype)
      return self.wrapped(shifted, *args[1:], **kwargs)

    nodes, edges, neighbors = jax.vmap(one)(jnp.arange(self.n_fold))
    scale = jnp.float32(0)
    return nodes[0] + scale * jnp.sum(nodes), edges[0] + scale * jnp.sum(edges), neighbors[0]


def _trace_ste(extent: int, *, tied: bool, mutant: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  bundle, config = _ste_bundle(extent, tied=tied)
  inner: Any = ConditionalDecode(model=_model(), state_iterator=VmapIterator())
  if mutant:
    which = "ste_tied" if tied else "ste_untied"
    inner = FoldConditional(inner=inner, n_fold=extent, which=which)
  ste = STEDecode(inner=inner, iterations=3, optimizer=optax.adam(1e-3))
  stage = make_stage_set(state_weights=bundle.conditioning.state_weights)
  key = jax.random.PRNGKey(0)

  def fn(key: jax.Array, bundle: Any, config: Any) -> Any:
    return ste(key, bundle, config, 1.0, stage)

  return fn, (key, bundle, config)


def _trace_optimize(extent: int, *, tied: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  bundle, config = _ste_bundle(extent, tied=tied)
  stage = make_stage_set(state_weights=bundle.conditioning.state_weights)
  opt_fn = make_optimize_sequence_fn(_model(), stage)
  key = jax.random.PRNGKey(0)

  def fn(key: jax.Array, bundle: Any, config: Any) -> Any:
    return opt_fn(key, bundle, config, 3, 1e-3, 1.0)

  return fn, (key, bundle, config)


def _encode_bundle(length: int, n_states: int) -> tuple[Any, Any]:
  key = jax.random.PRNGKey(1)
  steps = jax.random.normal(key, (n_states, length, 1, 3))
  ca = jnp.cumsum(steps, axis=1)
  coords = ca + jnp.zeros((1, 1, 4, 3))
  if n_states == 1:
    coords = coords[0]
    mask = jnp.ones((length,), dtype=jnp.float32)
    residue_index = jnp.arange(length, dtype=jnp.int32)
    chain_index = jnp.zeros((length,), dtype=jnp.int32)
  else:
    mask = jnp.ones((n_states, length), dtype=jnp.float32)
    residue_index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (n_states, length))
    chain_index = jnp.zeros((n_states, length), dtype=jnp.int32)
  return build_inference_bundle(
    coords,
    mask,
    residue_index,
    chain_index,
    mode="sample_autoregressive",
  )


def _trace_encode(
  extent: int, *, length: int, n_states: int, mutant: bool
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  """``extent`` is the swept axis; the other of length/S is the fixed one."""
  if n_states == 0:
    n_states = extent
  else:
    length = extent
  model: Any = _model()
  if mutant:
    model = FoldEncoder(wrapped=model, n_fold=n_states)
  encode = make_encode_fn(model, use_rolling_state=True)
  bundle, config = _encode_bundle(length, n_states)
  key = jax.random.PRNGKey(0)

  def fn(bundle: Any, key: jax.Array, config: Any) -> Any:
    return encode(bundle, key, config)

  return fn, (bundle, key, config)


def _score_inputs(length: int, n_masks: int, n_seq: int) -> tuple[Any, ...]:
  coords = jnp.ones((length, 4, 3), dtype=jnp.float32)
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

  return jax.tree_util.tree_map(one, shaped, is_leaf=lambda leaf: leaf is None or hasattr(leaf, "shape"))


def _trace_score(extent: int, *, mode: str) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  """Build a closed-over ``score_pairs`` trace. ``extent`` is K, C, or L per ``mode``."""
  if mode == "k":
    length, n_masks, n_seq = 24, extent, 2
  elif mode == "c":
    length, n_masks, n_seq = 24, 2, extent
  else:
    length, n_masks, n_seq = extent, 2, 2
  coords, mask, residue_index, chain_index, sequences, masks, keys = _score_inputs(length, n_masks, n_seq)
  score_mod = importlib.import_module("aminx.scoring.score")
  fns = score_mod.make_score_split_fns(_model())
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


def _first_of(outs: Any) -> tuple[Any, Any, Any]:
  return outs[0][0], outs[1][0], outs[2][0]


@contextmanager
def _patched(module: Any, name: str, value: Any) -> Any:
  previous = getattr(module, name)
  setattr(module, name, value)
  try:
    yield
  finally:
    setattr(module, name, previous)


def _mask_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    SENTINEL["score_k"] += 1
    masks = _SCORE_AXIS["masks"]

    def one(mask: jax.Array) -> Any:
      replaced = list(args)
      replaced[-1] = mask
      return real(*replaced)

    return _first_of(jax.vmap(one)(masks))

  return bad


def _seq_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    SENTINEL["score_c"] += 1
    one_hot = jax.nn.one_hot(_SCORE_AXIS["sequences"], 21)

    def one(sequence: jax.Array) -> Any:
      replaced = list(args)
      replaced[3] = sequence
      return real(*replaced)

    return _first_of(jax.vmap(one)(one_hot))

  return bad


def _reencode(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(*args: Any) -> tuple[Any, Any, Any]:
    SENTINEL["reencode"] += 1
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


def _optimize_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  def bad(model: Any, prng_key: jax.Array, bundle: Any, config: Any, stage_set: Any) -> jax.Array:
    SENTINEL["optimize_ste"] += 1
    length = bundle.geometry.coords.shape[1]

    def one(index: jax.Array) -> jax.Array:
      coords = bundle.geometry.coords + index.astype(bundle.geometry.coords.dtype)
      shifted = eqx.tree_at(lambda item: item.geometry.coords, bundle, coords)
      return real(model, prng_key, shifted, config, stage_set)

    logits = jax.vmap(one)(jnp.arange(length))
    return logits[0] + jnp.float32(0) * jnp.sum(logits)

  return bad


def _seam_fold(real: Callable[..., Any]) -> Callable[..., Any]:
  """Per-chunk body also runs ``f`` over all ``xs``.

  One pass leaves the N=16→32 ratio at ~1.58, under the 1.7 margin, because the
  chunk's own work is still a large share of a 16-wide vmap. Three passes keep
  the same whole-body shape and put both ratios above 1.7 (spec D5: strengthen
  the mutant, do not lower the threshold).
  """

  def bad(fn: Callable[..., Any], xs: Any, batch_size: int | None = None) -> Any:
    def wrapped(x: Any) -> Any:
      SENTINEL["seam"] += 1
      y = fn(x)
      extra = jax.vmap(fn)(xs)
      for _ in range(2):
        extra = jax.tree_util.tree_map(lambda left, right: left + right, extra, jax.vmap(fn)(xs))

      def mix(left: jax.Array, right: jax.Array) -> jax.Array:
        return left + jnp.float32(0) * jnp.sum(right)

      return jax.tree_util.tree_map(mix, y, extra)

    return real(wrapped, xs, batch_size=batch_size)

  return bad


@dataclass
class _Result:
  """Stand-in decode result (a pytree of the two arrays the seam returns)."""

  sequence: jax.Array
  logits: jax.Array


class _StubDecode(eqx.Module):
  """Cheap decode stand-in so the seam guard traces the chunk loop only (A8)."""

  scale: jax.Array

  def __call__(self, enc: jax.Array, bundle: jax.Array, key: jax.Array, config: Any) -> _Result:
    del config
    seq = (enc * self.scale + bundle + jax.random.randint(key, enc.shape, 0, 1)).astype(jnp.int32)
    return _Result(sequence=seq, logits=jnp.stack([seq.astype(jnp.float32)] * 3, axis=-1))


def _trace_seam_real(extent: int) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  enc = jnp.ones((4,), dtype=jnp.float32)
  bundle = jnp.ones((4,), dtype=jnp.float32)
  keys = jax.random.split(jax.random.PRNGKey(0), extent)
  decode = _StubDecode(scale=jnp.float32(2.0))

  def fn(enc: jax.Array, bundle: jax.Array, keys: jax.Array) -> Any:
    return _chunked_sample_decode(decode, enc, bundle, None, keys, ORDER_CFG, batch_size=8)

  return fn, (enc, bundle, keys)


def _trace_scan(extent: int, *, mutant: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  xs = jnp.ones((extent, 4), dtype=jnp.float32)
  weight = jnp.ones((4, 4), dtype=jnp.float32)

  def clean(x: jax.Array) -> jax.Array:
    return x @ weight

  def bad(x: jax.Array, xs: jax.Array = xs, weight: jax.Array = weight) -> jax.Array:
    SENTINEL["scan"] += 1
    return x @ weight + jnp.float32(0) * jnp.sum(xs @ weight)

  body = bad if mutant else clean

  def fn(xs: jax.Array) -> jax.Array:
    return _dispatch_axis(Scan(), body, xs)

  return fn, (xs,)


def _fill_self_test(payload: dict[str, Any]) -> None:
  o1, fold = self_test_ratios()
  payload["self_test_o1_ratio"] = float(o1)
  payload["self_test_fold_ratio"] = float(fold)


def run_self_test() -> dict[str, Any]:
  """Instrument controls only. In-scope ratios stay at the -1 sentinel."""
  payload = blank_payload(mode="self_test")
  _fill_self_test(payload)
  payload["loop_paths"] = "self-test only"
  payload["sentinels_fired"] = False
  return payload


def run_probe() -> dict[str, Any]:
  """Clean and mutant ratios for every in-scope loop, plus the instrument controls."""
  for name in SENTINEL_NAMES:
    SENTINEL[name] = 0
  payload = blank_payload(mode="probe")
  notes: list[str] = []
  _fill_self_test(payload)
  notes.append(
    f"self_test o1={payload['self_test_o1_ratio']:.6f} fold={payload['self_test_fold_ratio']:.6f}"
  )

  triples = (
    ("tiling_clean", lambda n: _trace_tiling(n, multi=False, mutant=False), (32, 64, 128), pick_outermost_while),
    ("tiling_multi_clean", lambda n: _trace_tiling(n, multi=True, mutant=False), (32, 64, 128), pick_outermost_while),
    (
      "tiling_ragged_clean",
      lambda n: _trace_tiling(n, multi=False, mutant=False),
      (37, 74, 148),
      pick_outermost_while,
    ),
    (
      "tiling_multi_ragged_clean",
      lambda n: _trace_tiling(n, multi=True, mutant=False),
      (37, 74, 148),
      pick_outermost_while,
    ),
    ("tiling_mut", lambda n: _trace_tiling(n, multi=False, mutant=True), (32, 64, 128), pick_outermost_while),
  )
  for name, trace_at, extents, pick in triples:
    _put(payload, notes, name, measure(name, trace_at, extents, pick))

  clean_k = measure("score_k_clean", lambda n: _trace_score(n, mode="k"), (5, 10, 20), pick_trips_scaling)
  _put(payload, notes, "score_k_clean", clean_k)
  _put(
    payload,
    notes,
    "score_c_clean",
    measure("score_c_clean", lambda n: _trace_score(n, mode="c"), (5, 10, 20), pick_trips_scaling),
  )
  _put(
    payload,
    notes,
    "score_l_clean",
    measure("score_l_clean", lambda n: _trace_score(n, mode="l"), (32, 64, 128), pick_fixed_scan_largest),
  )

  score_mod = importlib.import_module("aminx.scoring.score")
  with _patched(score_mod, "_score_candidate_kernel", _mask_fold(score_mod._score_candidate_kernel)):
    _put(
      payload,
      notes,
      "score_k_mut",
      measure("score_k_mut", lambda n: _trace_score(n, mode="k"), (5, 10, 20), pick_trips_scaling),
    )
  with _patched(score_mod, "_score_candidate_kernel", _seq_fold(score_mod._score_candidate_kernel)):
    _put(
      payload,
      notes,
      "score_c_mut",
      measure("score_c_mut", lambda n: _trace_score(n, mode="c"), (5, 10, 20), pick_trips_scaling),
    )

  with _patched(score_mod, "_score_candidate_kernel", _reencode(score_mod._score_candidate_kernel)):
    reencode = measure(
      "score_reencode", lambda n: _trace_score(n, mode="k"), (5, 10, 20), pick_trips_scaling
    )
  payload["score_reencode_k_r1"] = float(reencode.r1)
  payload["score_reencode_k_r2"] = float(reencode.r2)
  payload["score_reencode_inflation"] = float(reencode.works[0] / clean_k.works[0])
  notes.append(
    f"score_reencode path={reencode.path} ratios=({reencode.r1:.6f},{reencode.r2:.6f}) "
    f"inflation={payload['score_reencode_inflation']:.6f} "
    f"works_mut={reencode.works} works_clean={clean_k.works}"
  )

  import aminx.host.kernel_dispatch as dispatch_mod

  _put(payload, notes, "seam_clean", measure("seam_clean", _trace_seam_real, (16, 32, 64), pick_trips_scaling))
  with _patched(dispatch_mod, "chunked_map", _seam_fold(dispatch_mod.chunked_map)):
    _put(payload, notes, "seam_mut", measure("seam_mut", _trace_seam_real, (16, 32, 64), pick_trips_scaling))

  _put(
    payload,
    notes,
    "scan_clean",
    measure("scan_clean", lambda n: _trace_scan(n, mutant=False), (32, 64, 128), pick_trips_scaling),
  )
  _put(
    payload,
    notes,
    "scan_mut",
    measure("scan_mut", lambda n: _trace_scan(n, mutant=True), (32, 64, 128), pick_trips_scaling),
  )

  _put(
    payload,
    notes,
    "encode_s_clean",
    measure(
      "encode_s_clean",
      lambda n: _trace_encode(n, length=24, n_states=0, mutant=False),
      (2, 4, 8),
      pick_trips_scaling,
    ),
  )
  _put(
    payload,
    notes,
    "encode_l_clean",
    measure(
      "encode_l_clean",
      lambda n: _trace_encode(n, length=0, n_states=2, mutant=False),
      (32, 64, 128),
      pick_fixed_scan_largest,
    ),
  )
  _put(
    payload,
    notes,
    "encode_s_mut",
    measure(
      "encode_s_mut",
      lambda n: _trace_encode(n, length=24, n_states=0, mutant=True),
      (2, 4, 8),
      pick_trips_scaling,
    ),
  )

  for tied, clean_name, mut_name in (
    (False, "ste_untied_clean", "ste_untied_mut"),
    (True, "ste_tied_clean", "ste_tied_mut"),
  ):
    _put(
      payload,
      notes,
      clean_name,
      measure(
        clean_name,
        lambda n, tied=tied: _trace_ste(n, tied=tied, mutant=False),
        (32, 64, 128),
        pick_fixed_scan_largest,
      ),
    )
    _put(
      payload,
      notes,
      mut_name,
      measure(
        mut_name,
        lambda n, tied=tied: _trace_ste(n, tied=tied, mutant=True),
        (32, 64, 128),
        pick_fixed_scan_largest,
      ),
    )

  import aminx.inference.optimize_ste as opt_mod

  _put(
    payload,
    notes,
    "optimize_ste_clean",
    measure(
      "optimize_ste_clean",
      lambda n: _trace_optimize(n, tied=False),
      (32, 64, 128),
      pick_outermost_while,
    ),
  )
  with _patched(opt_mod, "score_conditional", _optimize_fold(opt_mod.score_conditional)):
    _put(
      payload,
      notes,
      "optimize_ste_mut",
      measure(
        "optimize_ste_mut",
        lambda n: _trace_optimize(n, tied=False),
        (32, 64, 128),
        pick_outermost_while,
      ),
    )

  payload["sentinels_fired"] = all(SENTINEL[name] > 0 for name in SENTINEL_NAMES)
  notes.append("sentinels " + ",".join(f"{name}={SENTINEL[name]}" for name in SENTINEL_NAMES))
  payload["loop_paths"] = "\n".join(notes)
  return payload


def _refuse(out: str, mode: str, reason: str, started: float) -> int:
  payload = blank_payload(mode="refused")
  payload["refused"] = True
  payload["refusal_reason"] = reason[:2000]
  payload["elapsed_s"] = time.perf_counter() - started
  payload["loop_paths"] = f"refused during {mode}"
  try:
    write_payload(payload, out)
  except Exception:
    logger.exception("failed to write refusal payload")
    return 1
  return 1


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry: ``--self-test`` or the full trace-only probe, always writing ``--out``."""
  parser = argparse.ArgumentParser(description="Trace-only loop complexity probe (debt #1981 Task 0)")
  parser.add_argument("--out", required=True, help="JSON results path")
  parser.add_argument("--self-test", action="store_true", help="Instrument controls only")
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args(argv)
  logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(message)s")
  logger.info("OMP_NUM_THREADS=%s JAX_PLATFORMS=%s", os.environ.get("OMP_NUM_THREADS"), os.environ.get("JAX_PLATFORMS"))
  started = time.perf_counter()
  mode = "self_test" if args.self_test else "probe"
  try:
    payload = run_self_test() if args.self_test else run_probe()
    payload["elapsed_s"] = time.perf_counter() - started
    write_payload(payload, args.out)
  except Exception as exc:
    logger.error("%s", traceback.format_exc())
    return _refuse(args.out, mode, f"{type(exc).__name__}: {exc}", started)
  if args.self_test and not controls_ok(payload):
    logger.error("self-test controls failed: o1=%s fold=%s", payload["self_test_o1_ratio"], payload["self_test_fold_ratio"])
    return 1
  logger.info("wrote %s mode=%s", args.out, payload["mode"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
