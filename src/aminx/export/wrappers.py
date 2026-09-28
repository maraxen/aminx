"""RNG-free P03/P04 export wrappers (D-A, D-B, D-H).

``Aminx.__call__`` always consumes a PRNG key (injecting ``PRNGKey(0)`` when none is
given, V17), which makes it unsuitable to trace for export: an exported graph that
splits/consumes RNG state is neither deterministic across export targets nor legal for
some of them. These wrappers reproduce the P03 (edge-feature) and P04 (unconditional
logit) computations without ever creating or consuming a key (D-B): they compute the
k-NN graph and RBF features directly on the host-side pipeline
(``compute_backbone_coordinates`` -> ``compute_backbone_distance`` ->
``aminx.model.features.select_neighbors`` -> ``compute_radial_basis``), then hand the
precomputed ``rbf_features``/``neighbor_indices`` to
``ProteinFeatures.forward_edge_stages`` (which takes the precomputed branch, V3, and
never touches ``apply_noise_to_coordinates``), and call the encoder/decoder with
``key=None``. With ``key=None`` the encoder forces ``inference=True``
(``encoder.py:369-370``) and every ``Dropout`` returns its input unchanged
(``dropout.py:52-54``), so the wrapper is dropout-free by construction (V17) --
``zero_dropout`` is not applied to the model these wrappers close over; it exists here
only to build the deterministic "native" comparator used in tests (D-H), and to record
``n_dropout``/``max_p_before`` on the wrapper's own ``.meta``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.factory import make_decode_fn  # noqa: TID251
from aminx.inference.decode.mode import UnconditionalMode  # noqa: TID251
from aminx.io.weights import get_topology_for_checkpoint
from aminx.model.dropout import Dropout
from aminx.model.features import select_neighbors
from aminx.tiling.strategy import Vmap
from aminx.types.bundles import EncoderOutput
from aminx.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from aminx.utils.radial_basis import compute_radial_basis

if TYPE_CHECKING:
  # Type-only: aminx.types.stages / aminx.inference.decode.unconditional are exempt from
  # the PottsModel-parallel banned-api rule (ADR 260605_potts-parallel-not-stageset) at
  # runtime already (this wrapper never imports them for a value, only a type), same
  # status as `aminx.potts.designer`'s existing TID251 exemption.
  from aminx.inference.decode.unconditional import UnconditionalDecode  # noqa: TID251
  from aminx.model import Aminx
  from aminx.types.stages import StageSet  # noqa: TID251

#: ``forward_edge_stages``/``UnconditionalDecode.__call__`` type their PRNG-key
#: parameter as non-Optional (out of this task's scope to change, per the T1 spec's
#: "No other change to features.py" and no decode-module changes at all), even though
#: both genuinely accept ``None`` at runtime (V17: ``key is None`` forces
#: ``inference=True`` and skips every ``split``/``fold_in``). ``_NO_KEY`` names that gap
#: at each RNG-free call site instead of a bare, unexplained ``None``.
_NO_KEY = cast("PRNGKeyArray", None)

#: The checkpoint this export ladder is pinned to (V10, common context "Fixtures").
#: P01 (checkpoint-agnostic topology inference) does not advance this sprint, so wrappers
#: cross-check ``model.features.k_neighbors`` against this checkpoint's own topology
#: instead of trusting either the model instance or a bare constant alone.
PINNED_CHECKPOINT_ID = "proteinmpnn_v_48_020"

#: T4 (IREE stack-allocation fix): row-chunk size passed to ``select_neighbors`` ->
#: ``top_k`` for the compiled export path only (the eager/training call in
#: ``ProteinFeatures.forward_edge_stages`` is unaffected -- it does not pass
#: ``row_chunk`` and keeps its pre-T4 trace exactly).
#:
#: Measured 260928 (see ``top_k``'s docstring): IREE's llvm-cpu codegen for the
#: unchunked ``lax.sort`` stack-allocates a scratch buffer that starts exceeding the
#: default 32768-byte limit around L=360-400 and keeps growing with L (131072 bytes
#: at L=512, 262144 at L=1024) -- this is what made P03/P04 fail to compile
#: (native AND wasm32) at the L=512/1024 export buckets. Capping the batch axis
#: actually processed per compiled call at 32 rows keeps the allocation bounded
#: (~19KB native / ~16KB wasm32) at every export bucket (128/256/512/1024), with
#: comfortable headroom under both IREE's own default limit and Emscripten's
#: default 64KiB linked wasm stack -- a real reduction in per-call footprint, not
#: a raised ceiling (``--iree-llvmcpu-stack-allocation-limit`` was rejected for
#: wasm32 specifically because it only raises IREE's compile-time guard without
#: changing the actual stack frame size, which would silently overflow
#: Emscripten's default stack at runtime; nothing here executes wasm32 to verify
#: that empirically, so it is not an acceptable fix for that target).
EXPORT_TOP_K_ROW_CHUNK = 32


class _MetaCallable:
  """A JAX-traceable callable carrying a ``.meta`` dict (``n_dropout``, ``max_p_before``).

  Plain function objects can have arbitrary attributes bolted on at runtime, but a
  strict type checker has no way to know a bare ``Callable``-typed return value carries
  one. This tiny wrapper makes ``.meta`` part of the type instead.
  """

  def __init__(self, fn: Callable[..., Any], meta: dict[str, int | float]) -> None:
    self._fn = fn
    self.meta = meta

  def __call__(self, *args: Any) -> Any:  # noqa: ANN401
    return self._fn(*args)


def zero_dropout(model: Aminx) -> tuple[Aminx, dict[str, int | float]]:
  """Return a copy of ``model`` with every ``Dropout.p`` set to ``0.0`` (D-H).

  Collects every ``aminx.model.dropout.Dropout`` leaf via
  ``jax.tree_util.tree_leaves(model, is_leaf=...)`` (so ``Dropout`` -- itself an
  ``eqx.Module`` pytree node -- is treated as an opaque leaf rather than flattened
  through) and replaces each one's ``p`` with ``eqx.tree_at``.

  Args:
    model: An ``Aminx`` model.

  Returns:
    ``(zeroed_model, stats)`` where ``stats`` is ``{"n_dropout": int, "max_p_before":
    float}``. ``n_dropout`` is the count of ``Dropout`` modules found (not whether any
    had ``p > 0``); ``max_p_before`` is the largest ``p`` observed before zeroing, or
    ``0.0`` if there were none.
  """

  def _dropouts(m: Aminx) -> list[Dropout]:
    leaves = jax.tree_util.tree_leaves(m, is_leaf=lambda x: isinstance(x, Dropout))
    return [leaf for leaf in leaves if isinstance(leaf, Dropout)]

  dropouts = _dropouts(model)
  n_dropout = len(dropouts)
  max_p_before = max((float(d.p) for d in dropouts), default=0.0)

  def where(m: Aminx) -> list[jax.Array]:
    return [d.p for d in _dropouts(m)]

  replace = [jnp.zeros_like(d.p) for d in dropouts]
  zeroed = eqx.tree_at(where, model, replace=replace)
  return zeroed, {"n_dropout": n_dropout, "max_p_before": max_p_before}


def _assert_pinned_topology(model: Aminx) -> None:
  """Assert V10: ``model.features.k_neighbors`` matches the pinned checkpoint's topology."""
  expected_k = get_topology_for_checkpoint(PINNED_CHECKPOINT_ID)["k_neighbors"]
  actual_k = model.features.k_neighbors
  if actual_k != expected_k:
    msg = (
      f"model.features.k_neighbors={actual_k!r} does not match "
      f"get_topology_for_checkpoint({PINNED_CHECKPOINT_ID!r})['k_neighbors']="
      f"{expected_k!r} (V10); refusing to build an export wrapper for a model whose "
      "topology disagrees with the checkpoint this export ladder is pinned to."
    )
    raise ValueError(msg)


def make_p03_featurize(
  model: Aminx,
) -> _MetaCallable:
  """Build an RNG-free P03 (edge-feature) wrapper closed over ``model`` (D-B).

  Args:
    model: The model to wrap. Must satisfy the V10 topology assertion.

  Returns:
    A callable ``(coords, mask, residue_index, chain_index) -> (neighbor_indices,
    edge_features)`` with ``neighbor_indices`` ``int32 (L, k)`` and ``edge_features``
    ``float32 (L, k, H)``, plus a ``.meta`` dict (``n_dropout``, ``max_p_before``,
    recorded from ``zero_dropout(model)`` -- the wrapper itself closes over the
    unmodified ``model``, per D-B/D-H).

  Raises:
    ValueError: If ``model.features.k_neighbors`` disagrees with the pinned
      checkpoint's topology (V10).
  """
  _assert_pinned_topology(model)
  _, stats = zero_dropout(model)

  def p03_featurize(
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
  ) -> tuple[Int[Array, "L k"], Float[Array, "L k H"]]:
    backbone_coords = compute_backbone_coordinates(coords)
    distances = compute_backbone_distance(backbone_coords)
    neighbor_indices = select_neighbors(
      distances,
      mask,
      model.features.k_neighbors,
      row_chunk=EXPORT_TOP_K_ROW_CHUNK,
    )
    rbf = compute_radial_basis(backbone_coords, neighbor_indices)
    stages = model.features.forward_edge_stages(
      _NO_KEY,
      coords,
      mask,
      residue_index,
      chain_index,
      None,
      rbf_features=rbf,
      neighbor_indices=neighbor_indices,
    )
    return (
      jnp.asarray(stages.neighbor_indices, dtype=jnp.int32),
      jnp.asarray(stages.final, dtype=jnp.float32),
    )

  return _MetaCallable(p03_featurize, dict(stats))


def make_p04_unconditional(
  model: Aminx,
  stage_set: StageSet,
) -> _MetaCallable:
  """Build an RNG-free P04 (unconditional-logit) wrapper closed over ``model`` (D-B).

  Args:
    model: The model to wrap. Must satisfy the V10 topology assertion.
    stage_set: A ``StageSet`` (e.g. ``aminx.inference.logits.make_stage_set()``)
      supplying the logit fusion used by ``UnconditionalDecode``.

  Returns:
    A callable ``(coords, mask, residue_index, chain_index) -> (logits,
    neighbor_indices)`` with ``logits`` ``float32 (L, 21)`` and ``neighbor_indices``
    ``int32 (L, k)``, plus a ``.meta`` dict (``n_dropout``, ``max_p_before``).

  Raises:
    ValueError: If ``model.features.k_neighbors`` disagrees with the pinned
      checkpoint's topology (V10).
  """
  _assert_pinned_topology(model)
  _, stats = zero_dropout(model)
  # make_decode_fn's declared return type is a union over every decode mode; UnconditionalMode
  # always resolves to UnconditionalDecode at runtime (factory.py's isinstance dispatch), so
  # this narrows what ty cannot infer across that dispatch boundary.
  decode_fn = cast(
    "UnconditionalDecode",
    make_decode_fn(model, mode=UnconditionalMode(), strategy=Vmap()),
  )

  def p04_unconditional(
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
  ) -> tuple[Float[Array, "L 21"], Int[Array, "L k"]]:
    backbone_coords = compute_backbone_coordinates(coords)
    distances = compute_backbone_distance(backbone_coords)
    neighbor_indices = select_neighbors(
      distances,
      mask,
      model.features.k_neighbors,
      row_chunk=EXPORT_TOP_K_ROW_CHUNK,
    )
    rbf = compute_radial_basis(backbone_coords, neighbor_indices)
    stages = model.features.forward_edge_stages(
      _NO_KEY,
      coords,
      mask,
      residue_index,
      chain_index,
      None,
      rbf_features=rbf,
      neighbor_indices=neighbor_indices,
    )
    node_features, edge_features = model.encoder(
      stages.final,
      stages.neighbor_indices,
      mask,
      key=None,
    )
    enc = EncoderOutput(
      node_features=jnp.asarray(node_features)[None, ...],
      edge_features=jnp.asarray(edge_features)[None, ...],
      neighbor_indices=stages.neighbor_indices[None, ...],
      mask=jnp.asarray(mask)[None, ...],
    )
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      mode="score_unconditional",
      inference=True,
    )
    logits = decode_fn(_NO_KEY, enc, bundle, config, stage_set)
    return (
      jnp.asarray(logits, dtype=jnp.float32),
      jnp.asarray(stages.neighbor_indices, dtype=jnp.int32),
    )

  return _MetaCallable(p04_unconditional, dict(stats))
