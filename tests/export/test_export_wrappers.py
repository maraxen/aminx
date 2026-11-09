"""Tests for RNG-free P03/P04 export wrappers, the export bucket ladder, and the jaxpr
RNG walker (task 260926_browser-export-loop, T1; spec 260926_aminx-browser-export-phase2a).

CPU only, synthetic structures (random coordinates x 10 Angstrom, seed fixed). The shared
checkpoint fixture loads the real pinned checkpoint (``proteinmpnn_v_48_020``, V10) so
this file's Gate also surfaces finding F-D1b: whether the loaded checkpoint's
``Dropout.p`` values are ever > 0 (see ``TestZeroDropoutInvariant`` and the module-level
``CHECKPOINT_MAX_P_BEFORE`` capture below, quoted in the T1 commit body).
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.export.buckets import (
  EXPORT_BUCKETS,
  LengthAboveMaxBucketError,
  LengthBelowNeighborsError,
  pad_inputs,
  select_export_bucket,
)
from aminx.export.rng_audit import find_rng_primitives
from aminx.export.wrappers import (
  PINNED_CHECKPOINT_ID,
  make_p03_featurize,
  make_p04_unconditional,
  p07_bundle,
  zero_dropout,
)
from aminx.inference import score_unconditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.io.weights import get_topology_for_checkpoint, load_model
from aminx.model import Aminx
from aminx.model.dropout import Dropout
from aminx.model.features import select_neighbors, top_k
from aminx.utils.autoregression import (
  ar_mask_from_decoding_order,
  full_context_ar_mask,
  generate_ar_mask,
  generate_wave_ar_mask,
)
from scripts.browser_validation._lattice import cubic_lattice_ca

# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _synthetic_structure(
  num_residues: int,
  seed: int = 0,
  scale: float = 10.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Deterministic random (L, 4, 3) backbone coordinates (N, CA, C, O order; D-C)."""
  rng = np.random.default_rng(seed)
  coords = jnp.asarray((rng.normal(size=(num_residues, 4, 3)) * scale).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)
  return coords, mask, residue_index, chain_index


def _avals_for(num_residues: int) -> tuple[jax.ShapeDtypeStruct, ...]:
  return (
    jax.ShapeDtypeStruct((num_residues, 4, 3), jnp.float32),
    jax.ShapeDtypeStruct((num_residues,), jnp.float32),
    jax.ShapeDtypeStruct((num_residues,), jnp.int32),
    jax.ShapeDtypeStruct((num_residues,), jnp.int32),
  )


def _all_dropout_p(model: object) -> list[float]:
  leaves = jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, Dropout))
  return [float(leaf.p) for leaf in leaves if isinstance(leaf, Dropout)]


def _with_k_neighbors(model: Aminx, k: int) -> Aminx:
  """Return a copy of ``model`` with ``features.k_neighbors`` patched to ``k``.

  ``k_neighbors`` is an ``eqx.field(static=True)``, so it lives in the pytree's aux data,
  not among its leaves -- ``eqx.tree_at`` cannot reach it. ``object.__setattr__`` bypasses
  ``eqx.Module``'s frozen-dataclass ``__setattr__`` directly, on a shallow copy so the
  original ``model`` is left untouched.
  """
  patched_features = copy.copy(model.features)
  object.__setattr__(patched_features, "k_neighbors", k)
  patched_model = copy.copy(model)
  object.__setattr__(patched_model, "features", patched_features)
  return patched_model


def _log_softmax64(logits: np.ndarray) -> np.ndarray:
  """``log_softmax`` computed in numpy float64 on the host (bar table's ``float64 host``)."""
  x = np.asarray(logits, dtype=np.float64)
  shifted = x - x.max(axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _frozen_select_neighbors(
  distances: jax.Array,
  mask: jax.Array,
  k: int,
  structure_mapping: jax.Array | None = None,
) -> jax.Array:
  """Frozen copy of ``ProteinFeatures.forward_edge_stages``'s pre-extraction (260926) block.

  Independent of ``aminx.model.features.select_neighbors`` (only ``top_k`` is shared, and
  ``top_k`` itself is not part of this task's refactor) so a regression introduced by the
  extraction -- or by a later edit to either copy -- shows up as a bitwise mismatch here.
  """
  distances_masked = jnp.array(
    jnp.where(
      (mask[:, None] * mask[None, :]).astype(jnp.bool_),
      distances,
      jnp.inf,
    ),
  )
  if structure_mapping is not None:
    same_structure = structure_mapping[:, jnp.newaxis] == structure_mapping[jnp.newaxis, :]
    distances_masked = jnp.array(
      jnp.where(
        same_structure.astype(jnp.bool_),
        distances_masked,
        jnp.inf,
      ),
    ).squeeze()
  k_clamped = min(k, distances.shape[0])
  _, neighbor_indices = top_k(-distances_masked, k_clamped)
  return jnp.array(neighbor_indices, dtype=jnp.int32)


def _pairwise_distance(coords: jax.Array) -> jax.Array:
  """Same distance convention as ``aminx.utils.coordinates.compute_backbone_distance``."""
  diff = coords[:, None, :] - coords[None, :, :]
  return jnp.sqrt(1e-6 + jnp.sum(jnp.square(diff), axis=-1))


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint_model() -> Aminx:
  """The pinned checkpoint (V10), loaded once for the whole module."""
  return load_model(checkpoint_id=PINNED_CHECKPOINT_ID)


@pytest.fixture(scope="module")
def native_and_stats(checkpoint_model: Aminx) -> tuple[Aminx, dict[str, int | float]]:
  """``zero_dropout(checkpoint_model)`` -- the D-H "native" comparator, built once."""
  return zero_dropout(checkpoint_model)


# --------------------------------------------------------------------------------------
# Shared-fixture invariants (not vacuous; complete zeroing; F-D1b capture)
# --------------------------------------------------------------------------------------


@pytest.mark.requires_weights
class TestZeroDropoutInvariant:
  """The shared fixture's own preconditions, asserted directly rather than assumed."""

  def test_zeroing_is_not_vacuous(
    self,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
  ) -> None:
    _, stats = native_and_stats
    assert stats["n_dropout"] > 0

  def test_every_dropout_p_is_zero_after_zeroing(
    self,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
  ) -> None:
    native_model, _ = native_and_stats
    assert _all_dropout_p(native_model) == [0.0] * len(_all_dropout_p(native_model))
    assert all(p == 0.0 for p in _all_dropout_p(native_model))

  def test_f_d1b_checkpoint_max_p_before(
    self,
    checkpoint_model: Aminx,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
  ) -> None:
    """F-D1b (conditional): report the loaded checkpoint's Dropout.p values.

    Raised only if ``max_p_before > 0`` -- this test never fails either way, it exists to
    put the measured values into the test log/report for the commit body regardless of
    which way it comes out.
    """
    _, stats = native_and_stats
    p_values = _all_dropout_p(checkpoint_model)
    print(f"F-D1b: checkpoint {PINNED_CHECKPOINT_ID!r} Dropout.p values: {p_values}")
    print(f"F-D1b: max_p_before = {stats['max_p_before']!r}")
    assert stats["max_p_before"] == max(p_values, default=0.0)


# --------------------------------------------------------------------------------------
# (a) W-equivalence: wrapper vs native, P03 and P04, at every export bucket
# --------------------------------------------------------------------------------------


@pytest.mark.requires_weights
class TestWEquivalence:
  """D-H: ``native_model.features(PRNGKey(0), ..., 0.0)`` /
  ``score_unconditional.kernel(native_model, PRNGKey(0), ...)`` vs the wrappers."""

  @pytest.mark.parametrize("bucket", EXPORT_BUCKETS)
  def test_p03_matches_native(
    self,
    checkpoint_model: Aminx,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
    bucket: int,
  ) -> None:
    native_model, _ = native_and_stats
    coords, mask, residue_index, chain_index = _synthetic_structure(bucket, seed=bucket)

    p03 = make_p03_featurize(checkpoint_model)
    idx_w, edge_w = p03(coords, mask, residue_index, chain_index)

    edge_n, idx_n, _, _ = native_model.features(
      jax.random.PRNGKey(0),
      coords,
      mask,
      residue_index,
      chain_index,
      0.0,
    )

    assert np.array_equal(np.asarray(idx_w), np.asarray(idx_n))
    max_abs = float(np.max(np.abs(np.asarray(edge_w) - np.asarray(edge_n))))
    assert max_abs <= 1e-6, max_abs

  @pytest.mark.parametrize("bucket", EXPORT_BUCKETS)
  def test_p04_matches_native(
    self,
    checkpoint_model: Aminx,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
    bucket: int,
  ) -> None:
    native_model, _ = native_and_stats
    coords, mask, residue_index, chain_index = _synthetic_structure(bucket, seed=bucket + 1)

    stage_set = make_stage_set()
    p04 = make_p04_unconditional(checkpoint_model, stage_set)
    logits_w, idx_w = p04(coords, mask, residue_index, chain_index)

    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      mode="score_unconditional",
    )
    assert config.inference is True
    logits_n = score_unconditional.kernel(
      native_model,
      jax.random.PRNGKey(0),
      bundle,
      config,
      stage_set,
    )
    _, idx_n, _, _ = native_model.features(
      jax.random.PRNGKey(0),
      coords,
      mask,
      residue_index,
      chain_index,
      0.0,
    )

    assert np.array_equal(np.asarray(idx_w), np.asarray(idx_n))
    max_abs = float(np.max(np.abs(np.asarray(logits_w) - np.asarray(logits_n))))
    assert max_abs <= 1e-6, max_abs


# --------------------------------------------------------------------------------------
# (b) P27: padded vs unpadded native, k=48 (checkpoint) and a random-init k=32 model (X1)
# --------------------------------------------------------------------------------------


class TestP27PaddedVsUnpaddedNative:
  """Bar table: P04 log-probs (real rows) TOL <= 1e-4 nats; indices EXACT 0 mismatches."""

  @staticmethod
  def _check(model: Aminx, l_real: int, bucket: int) -> None:
    native_model, _ = zero_dropout(model)
    coords, mask, residue_index, chain_index = _synthetic_structure(l_real, seed=l_real)
    stage_set = make_stage_set()

    bundle_u, config_u = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      mode="score_unconditional",
    )
    assert config_u.inference is True
    logits_u = score_unconditional.kernel(
      native_model,
      jax.random.PRNGKey(0),
      bundle_u,
      config_u,
      stage_set,
    )
    _, idx_u, _, _ = native_model.features(
      jax.random.PRNGKey(0),
      coords,
      mask,
      residue_index,
      chain_index,
      0.0,
    )

    padded = pad_inputs(
      np.asarray(coords),
      np.asarray(mask),
      np.asarray(residue_index),
      np.asarray(chain_index),
      bucket,
    )
    coords_p = jnp.asarray(padded["coords"])
    mask_p = jnp.asarray(padded["mask"])
    residue_index_p = jnp.asarray(padded["residue_index"])
    chain_index_p = jnp.asarray(padded["chain_index"])

    bundle_p, config_p = build_inference_bundle(
      coords=coords_p,
      mask=mask_p,
      residue_index=residue_index_p,
      chain_index=chain_index_p,
      mode="score_unconditional",
    )
    assert config_p.inference is True
    logits_p = score_unconditional.kernel(
      native_model,
      jax.random.PRNGKey(0),
      bundle_p,
      config_p,
      stage_set,
    )
    _, idx_p, _, _ = native_model.features(
      jax.random.PRNGKey(0),
      coords_p,
      mask_p,
      residue_index_p,
      chain_index_p,
      0.0,
    )

    assert np.array_equal(np.asarray(idx_u), np.asarray(idx_p)[:l_real])

    log_probs_u = _log_softmax64(np.asarray(logits_u))
    log_probs_p = _log_softmax64(np.asarray(logits_p)[:l_real])
    max_abs = float(np.max(np.abs(log_probs_u - log_probs_p)))
    assert max_abs <= 1e-4, max_abs

  @pytest.mark.requires_weights
  def test_checkpoint_k48(self, checkpoint_model: Aminx) -> None:
    self._check(checkpoint_model, l_real=100, bucket=128)

  def test_random_init_k32(self) -> None:
    model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=2,
      num_decoder_layers=2,
      k_neighbors=32,
      dropout_rate=0.1,
      key=jax.random.PRNGKey(11),
    )
    self._check(model, l_real=40, bucket=128)


# --------------------------------------------------------------------------------------
# (c) Export-bucket refusal/acceptance boundaries, k=48 and k=32
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("k_neighbors", "l_refuse_low", "l_accept_low", "l_refuse_high", "l_accept_high"),
  [
    (48, 47, 48, 1025, 1024),
    (32, 31, 32, 1025, 1024),
  ],
)
def test_export_bucket_boundaries(
  k_neighbors: int,
  l_refuse_low: int,
  l_accept_low: int,
  l_refuse_high: int,
  l_accept_high: int,
) -> None:
  with pytest.raises(LengthBelowNeighborsError):
    select_export_bucket(l_refuse_low, k_neighbors)
  select_export_bucket(l_accept_low, k_neighbors)  # must not raise
  with pytest.raises(LengthAboveMaxBucketError):
    select_export_bucket(l_refuse_high, k_neighbors)
  select_export_bucket(l_accept_high, k_neighbors)  # must not raise


# --------------------------------------------------------------------------------------
# (d) select_neighbors bitwise vs a frozen pre-refactor copy
# --------------------------------------------------------------------------------------


class TestSelectNeighborsBitwiseVsFrozen:
  """No model or weights needed -- pure array-in, array-out."""

  @staticmethod
  def _assert_matches(
    distances: jax.Array,
    mask: jax.Array,
    k: int,
    structure_mapping: jax.Array | None = None,
  ) -> None:
    got = select_neighbors(distances, mask, k, structure_mapping)
    want = _frozen_select_neighbors(distances, mask, k, structure_mapping)
    assert np.array_equal(np.asarray(got), np.asarray(want))

  def test_cubic_lattice_ties(self) -> None:
    coords = cubic_lattice_ca(96)
    distances = _pairwise_distance(coords)
    mask = jnp.ones((96,), dtype=jnp.float32)
    self._assert_matches(distances, mask, 48)

  def test_random_structure(self) -> None:
    rng = np.random.default_rng(5)
    coords = jnp.asarray((rng.normal(size=(96, 3)) * 10.0).astype(np.float32))
    distances = _pairwise_distance(coords)
    mask = jnp.ones((96,), dtype=jnp.float32)
    self._assert_matches(distances, mask, 48)

  def test_random_structure_last_20_masked(self) -> None:
    rng = np.random.default_rng(5)
    coords = jnp.asarray((rng.normal(size=(96, 3)) * 10.0).astype(np.float32))
    distances = _pairwise_distance(coords)
    mask = jnp.asarray(np.concatenate([np.ones(76), np.zeros(20)]).astype(np.float32))
    self._assert_matches(distances, mask, 48)

  def test_two_structure_mapping(self) -> None:
    rng = np.random.default_rng(6)
    coords = jnp.asarray((rng.normal(size=(96, 3)) * 10.0).astype(np.float32))
    distances = _pairwise_distance(coords)
    mask = jnp.ones((96,), dtype=jnp.float32)
    structure_mapping = jnp.asarray(
      np.concatenate([np.zeros(48), np.ones(48)]).astype(np.int32),
    )
    self._assert_matches(distances, mask, 48, structure_mapping)

  def test_clamp_is_live_l40_k48(self) -> None:
    """L=40 < k=48: below the export refusal layer (select_neighbors has no such refusal)."""
    rng = np.random.default_rng(7)
    coords = jnp.asarray((rng.normal(size=(40, 3)) * 10.0).astype(np.float32))
    distances = _pairwise_distance(coords)
    mask = jnp.ones((40,), dtype=jnp.float32)
    got = select_neighbors(distances, mask, 48)
    assert got.shape == (40, 40)  # clamp: min(48, 40) == 40
    self._assert_matches(distances, mask, 48)


# --------------------------------------------------------------------------------------
# (e) find_rng_primitives: clean on both wrappers, flags real RNG use
# --------------------------------------------------------------------------------------


class TestFindRngPrimitives:
  @pytest.mark.requires_weights
  def test_p03_wrapper_is_clean(self, checkpoint_model: Aminx) -> None:
    p03 = make_p03_featurize(checkpoint_model)
    jaxpr = jax.make_jaxpr(p03)(*_avals_for(128))
    assert find_rng_primitives(jaxpr) == []

  @pytest.mark.requires_weights
  def test_p04_wrapper_is_clean(self, checkpoint_model: Aminx) -> None:
    stage_set = make_stage_set()
    p04 = make_p04_unconditional(checkpoint_model, stage_set)
    jaxpr = jax.make_jaxpr(p04)(*_avals_for(128))
    assert find_rng_primitives(jaxpr) == []

  @pytest.mark.requires_weights
  def test_kernel_jaxpr_is_flagged(
    self,
    native_and_stats: tuple[Aminx, dict[str, int | float]],
  ) -> None:
    """Control: the native kernel DOES consume RNG, so the walker must see it."""
    native_model, _ = native_and_stats
    coords, mask, residue_index, chain_index = _synthetic_structure(64)
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      mode="score_unconditional",
    )
    stage_set = make_stage_set()

    def _traced(key: jax.Array) -> jax.Array:
      return score_unconditional.kernel(native_model, key, bundle, config, stage_set)

    jaxpr = jax.make_jaxpr(_traced)(jax.random.PRNGKey(0))
    assert find_rng_primitives(jaxpr) != []

  def test_planted_normal_is_flagged(self) -> None:
    def fn(x: jax.Array) -> jax.Array:
      return x + jax.random.normal(jax.random.PRNGKey(0), x.shape)

    jaxpr = jax.make_jaxpr(fn)(jnp.zeros((4,)))
    assert find_rng_primitives(jaxpr) != []


# --------------------------------------------------------------------------------------
# (f) check_export_safety: no sort-stability blocker on the P03/P04 wrappers, for both
# NATIVE and WASM32 (T4, task 260926_browser-export-loop). Until this fix,
# `make_p04_unconditional`'s traced jaxpr carried a `sort` op with `is_stable=True`
# (xtrax's `sort-stability` rule): NOT from `inference/decode/autoregressive.py`'s wave
# order (P04 goes through `UnconditionalDecode`, which never calls
# `AutoregressiveDecode.__call__`), but from `build_inference_bundle` ->
# `generate_ar_mask`'s tied branch, whose `group_decoding_order = jnp.argsort(...)` relied
# on the default `stable=True` over a key where every "group not present" slot shares one
# sentinel. That branch is gone: the mask now comes from the scatter-based wave path, which
# traces no sort at all (see
# `tests/utils/test_autoregression.py::test_generate_ar_mask_traces_no_sort`). This class's
# own red-check (documented in the T4 report) confirms this test fails on the pre-fix code.
# --------------------------------------------------------------------------------------


class TestCheckExportSafetyNoSortStability:
  @staticmethod
  def _blockers(fn: object, bucket: int) -> dict[str, list[str]]:
    from xtrax.export import NATIVE, WASM32, check_export_safety

    specs = _avals_for(bucket)
    out = {}
    for target_name, target in (("NATIVE", NATIVE), ("WASM32", WASM32)):
      blockers = check_export_safety([], {}, specs, fn, target)
      out[target_name] = [b.rule for b in blockers]
    return out

  @pytest.mark.requires_weights
  @pytest.mark.parametrize("bucket", [128, 256, 512, 1024])
  def test_p04_wrapper_has_no_sort_stability_blocker(
    self, checkpoint_model: Aminx, bucket: int
  ) -> None:
    stage_set = make_stage_set()
    p04 = make_p04_unconditional(checkpoint_model, stage_set)
    blockers = self._blockers(p04, bucket)
    for target_name, rules in blockers.items():
      assert "sort-stability" not in rules, (target_name, rules)

  @pytest.mark.requires_weights
  @pytest.mark.parametrize("bucket", [128, 256, 512, 1024])
  def test_p03_wrapper_has_no_sort_stability_blocker(
    self, checkpoint_model: Aminx, bucket: int
  ) -> None:
    p03 = make_p03_featurize(checkpoint_model)
    blockers = self._blockers(p03, bucket)
    for target_name, rules in blockers.items():
      assert "sort-stability" not in rules, (target_name, rules)

  def test_rng_in_cond_in_jit_is_flagged(self) -> None:
    def cond_rng(pred: jax.Array, x: jax.Array) -> jax.Array:
      def true_branch(x: jax.Array) -> jax.Array:
        return x + jax.random.normal(jax.random.PRNGKey(0), x.shape)

      def false_branch(x: jax.Array) -> jax.Array:
        return x

      return jax.lax.cond(pred, true_branch, false_branch, x)

    jaxpr = jax.make_jaxpr(jax.jit(cond_rng))(jnp.array(True), jnp.zeros((4,)))
    assert find_rng_primitives(jaxpr) != []

  def test_planted_rng_bit_generator_is_flagged(self) -> None:
    def fn(key: jax.Array) -> tuple[jax.Array, jax.Array]:
      return jax.lax.rng_bit_generator(key, shape=(4,))

    jaxpr = jax.make_jaxpr(fn)(jnp.zeros((2,), dtype=jnp.uint32))
    assert find_rng_primitives(jaxpr) != []


# --------------------------------------------------------------------------------------
# (f) Topology assertion fires on a patched k_neighbors
# --------------------------------------------------------------------------------------


@pytest.mark.requires_weights
class TestTopologyAssertion:
  def test_p03_refuses_mismatched_k_neighbors(self, checkpoint_model: Aminx) -> None:
    expected = get_topology_for_checkpoint(PINNED_CHECKPOINT_ID)["k_neighbors"]
    patched = _with_k_neighbors(checkpoint_model, expected - 1)
    with pytest.raises(ValueError, match="k_neighbors"):
      make_p03_featurize(patched)

  def test_p04_refuses_mismatched_k_neighbors(self, checkpoint_model: Aminx) -> None:
    expected = get_topology_for_checkpoint(PINNED_CHECKPOINT_ID)["k_neighbors"]
    patched = _with_k_neighbors(checkpoint_model, expected - 1)
    stage_set = make_stage_set()
    with pytest.raises(ValueError, match="k_neighbors"):
      make_p04_unconditional(patched, stage_set)


# --------------------------------------------------------------------------------------
# (g) Key-independence of native (D-H): checkpoint (p likely 0) AND a random-init p=0.1
# model, so the red-check mutation ("use model instead of zero_dropout(model)") is
# caught even if the checkpoint's own dropout happens to already be zero (V17 F-D1b).
# --------------------------------------------------------------------------------------


@pytest.mark.requires_weights
class TestKeyIndependenceOfNative:
  @pytest.mark.parametrize("model_kind", ["checkpoint", "random_init_p01"])
  def test_bitwise_key_independent_at_bucket_128(
    self,
    checkpoint_model: Aminx,
    model_kind: str,
  ) -> None:
    if model_kind == "checkpoint":
      model = checkpoint_model
    else:
      model = Aminx(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=48,
        dropout_rate=0.1,
        key=jax.random.PRNGKey(21),
      )

    # Red-check mutation target (D-H): replacing `zero_dropout(model)[0]` with `model`
    # here is the "use model instead of zero_dropout(model) for native" mutation.
    native_model, _ = zero_dropout(model)

    coords, mask, residue_index, chain_index = _synthetic_structure(128, seed=42)
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      mode="score_unconditional",
    )
    assert config.inference is True
    stage_set = make_stage_set()

    logits_key0 = score_unconditional.kernel(
      native_model,
      jax.random.PRNGKey(0),
      bundle,
      config,
      stage_set,
    )
    logits_key1 = score_unconditional.kernel(
      native_model,
      jax.random.PRNGKey(1),
      bundle,
      config,
      stage_set,
    )
    assert np.array_equal(np.asarray(logits_key0), np.asarray(logits_key1))


# --------------------------------------------------------------------------------------
# (f) wave_from_decoding_order's sorts must be tie-free.
#
# This function had NO unit coverage before this test, and it is the one baked into the
# split export's Graph W (spec 260929_p07-split-export.md): its outputs are the wave
# schedule and ar_mask the JavaScript loop drives. G0c and G1 validate it end to end
# against JAX, but an end-to-end comparison cannot see a stability regression, because
# both sides would regress together.
#
# It holds TWO sorts (wrappers.py:316 and :328) and both are deliberately tie-free:
#   - the group-order sort carries an explicit index as a second key
#   - order_in_wave's key is unique by construction (position vs length + position), so
#     `stable=False` is safe there
# IREE does not honour JAX's stable-sort tie order, so a revert to a stable sort on a
# tied key would change the generated design silently and identically on both sides of
# every parity comparison. Asserting on the traced jaxpr is the only thing that catches
# it -- the same reasoning, and the same guard shape, as
# test_generate_ar_mask_traces_no_sort in tests/utils/test_autoregression.py.
# --------------------------------------------------------------------------------------


def _all_eqns(closed_jaxpr):
  """Every eqn in the traced program, descending into nested jaxprs."""
  from aminx.export.rng_audit import _sub_jaxprs

  out = []

  def walk(jaxpr):
    for eqn in jaxpr.eqns:
      out.append(eqn)
      for value in eqn.params.values():
        for sub in _sub_jaxprs(value):
          walk(sub)

  walk(getattr(closed_jaxpr, "jaxpr", closed_jaxpr))
  return out


@pytest.mark.parametrize("length", [8, 32])
def test_wave_from_decoding_order_sorts_are_tie_free(length: int) -> None:
  """Every sort Graph W bakes in must be tie-free, not stability-dependent."""
  from aminx.export.wrappers import wave_from_decoding_order

  decoding_order = jnp.arange(length, dtype=jnp.int32)
  tie_group_map = jnp.arange(length, dtype=jnp.int32).at[1].set(0).at[3].set(2)

  jaxpr = jax.make_jaxpr(wave_from_decoding_order)(decoding_order, tie_group_map)
  sorts = [eqn for eqn in _all_eqns(jaxpr) if eqn.primitive.name == "sort"]

  assert sorts, "wave_from_decoding_order traced no sort; this guard would be vacuous"
  for eqn in sorts:
    assert eqn.params.get("is_stable") is False, (
      "wave_from_decoding_order lowered a STABLE sort. Graph W embeds this function, "
      "and IREE does not honour JAX's stable-sort tie order, so a tied key here would "
      "change the design silently -- see the comment block above."
    )


# --------------------------------------------------------------------------------------
# (g) p07_bundle ar_mask pinning tests: causality and tied positions on reverse and
# custom decoding orders (Defect 9 / task 261030_family-parity-coverage).
# --------------------------------------------------------------------------------------


class TestP07BundleArMaskPinning:
  """Pinning unit tests for p07_bundle: causality and tied-position mutual visibility."""

  @staticmethod
  def _dummy_inputs(
    length: int,
    decoding_order: jax.Array,
    tie_group_map: jax.Array,
  ) -> tuple[jax.Array, ...]:
    coords = jnp.zeros((length, 4, 3), dtype=jnp.float32)
    mask = jnp.ones((length,), dtype=jnp.float32)
    residue_index = jnp.arange(length, dtype=jnp.int32)
    chain_index = jnp.zeros((length,), dtype=jnp.int32)
    bias = jnp.zeros((length, 21), dtype=jnp.float32)
    fixed_mask = jnp.zeros((length,), dtype=jnp.float32)
    fixed_tokens = jnp.zeros((length,), dtype=jnp.int32)
    temperature = jnp.array(1.0, dtype=jnp.float32)
    return (
      coords,
      mask,
      residue_index,
      chain_index,
      decoding_order,
      bias,
      fixed_mask,
      fixed_tokens,
      temperature,
      tie_group_map,
    )

  def test_p07_bundle_ar_mask_reverse_decoding_order_causality(self) -> None:
    """Reverse order [L-1, ..., 0] must yield strictly upper-triangular causal mask."""
    length = 6
    tie_group_map = jnp.arange(length, dtype=jnp.int32)
    decoding_order = jnp.array([5, 4, 3, 2, 1, 0], dtype=jnp.int32)

    bundle, _ = p07_bundle(*self._dummy_inputs(length, decoding_order, tie_group_map))
    ar_mask = bundle.conditioning.ar_mask[0]

    assert ar_mask.shape == (length, length)
    # Self-exclusion: diagonal is strictly 0.0
    assert np.all(np.diag(np.asarray(ar_mask)) == 0.0)

    # In reverse order [5, 4, 3, 2, 1, 0]:
    # pos 5 decodes at step 0 (sees nobody)
    # pos 4 decodes at step 1 (sees pos 5)
    # ...
    # pos 0 decodes at step 5 (sees pos 5, 4, 3, 2, 1)
    # Position i decodes at step (5 - i). Pos i sees pos j iff step(j) < step(i) <=> j > i.
    expected = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1)
    assert np.array_equal(np.asarray(ar_mask), np.asarray(expected))

  def test_p07_bundle_ar_mask_custom_order_tied_positions(self) -> None:
    """Custom order with tied positions verifies mutual visibility and causal precedence."""
    length = 6
    # Group 0: {0, 3}; Group 1: {1, 4}; Group 2: {2}; Group 3: {5}
    tie_group_map = jnp.array([0, 1, 2, 0, 1, 3], dtype=jnp.int32)
    # Order visits: 2 (grp 2, wave 0), 4 (grp 1, wave 1), 0 (grp 0, wave 2),
    # 5 (grp 3, wave 3), then duplicate group positions 1 and 3.
    decoding_order = jnp.array([2, 4, 0, 5, 1, 3], dtype=jnp.int32)

    bundle, _ = p07_bundle(*self._dummy_inputs(length, decoding_order, tie_group_map))
    ar_mask = np.asarray(bundle.conditioning.ar_mask[0])

    assert ar_mask.shape == (length, length)
    # Self-exclusion on diagonal
    assert np.all(np.diag(ar_mask) == 0.0)

    # Mutual visibility between tied positions in Group 1 ({1, 4})
    assert ar_mask[1, 4] == 1.0, "pos 1 must see tied partner pos 4"
    assert ar_mask[4, 1] == 1.0, "pos 4 must see tied partner pos 1"

    # Mutual visibility between tied positions in Group 0 ({0, 3})
    assert ar_mask[0, 3] == 1.0, "pos 0 must see tied partner pos 3"
    assert ar_mask[3, 0] == 1.0, "pos 3 must see tied partner pos 0"

    # Causality checks across waves:
    # Wave 0 (pos 2) decodes first -> sees nobody
    assert np.sum(ar_mask[2]) == 0.0, "earliest position must have zero visible context"

    # Wave 1 ({1, 4}) sees wave 0 (pos 2) + each other
    assert ar_mask[1, 2] == 1.0
    assert ar_mask[4, 2] == 1.0
    # and cannot see later waves (pos 0, 3, 5)
    assert ar_mask[1, 0] == 0.0 and ar_mask[1, 3] == 0.0 and ar_mask[1, 5] == 0.0
    assert ar_mask[4, 0] == 0.0 and ar_mask[4, 3] == 0.0 and ar_mask[4, 5] == 0.0

    # Wave 3 (pos 5) decodes last -> sees all positions 0, 1, 2, 3, 4
    assert np.sum(ar_mask[5]) == 5.0, "latest position must see all previously decoded positions"

    # Full exact expected mask check
    expected = np.array([
      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # pos 0 (grp 0, wave 2): sees 2 (w0), 1,4 (w1), 3 (same grp)
      [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # pos 1 (grp 1, wave 1): sees 2 (w0), 4 (same grp)
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pos 2 (grp 2, wave 0): sees nobody
      [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # pos 3 (grp 0, wave 2): sees 2 (w0), 1,4 (w1), 0 (same grp)
      [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # pos 4 (grp 1, wave 1): sees 2 (w0), 1 (same grp)
      [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # pos 5 (grp 3, wave 3): sees 2 (w0), 1,4 (w1), 0,3 (w2)
    ], dtype=np.float32)
    assert np.array_equal(ar_mask, expected)

  def test_p07_bundle_exported_wave_and_ar_mask_describe_one_schedule(self) -> None:
    """The exported wave and the exported ar_mask must imply the same visibility.

    p07_bundle builds its wave with ``wave_from_decoding_order`` and its ar_mask with
    ``ar_mask_from_decoding_order`` (which goes through ``WaveScheduleBundle.from_decoding_order``).
    The two constructors number their waves differently (one wave per tie group in
    first-step order, versus one wave per step), so the comparison is on what each implies:
    the mask generated from the exported wave must equal the exported ar_mask. Random
    permutations with tied groups, including groups whose members are far apart in the order.
    """
    rng = np.random.default_rng(20261030)
    for length in (6, 9):
      for _ in range(16):
        decoding_order = jnp.asarray(rng.permutation(length), dtype=jnp.int32)
        # Tie ids stay in 0 .. L-1 (the smallest L here is 6, so 0..4 is in range), the
        # convention both wave constructors index by.
        tie_group_map = jnp.asarray(rng.integers(0, 5, size=length), dtype=jnp.int32)

        bundle, _ = p07_bundle(*self._dummy_inputs(length, decoding_order, tie_group_map))
        exported_mask = np.asarray(bundle.conditioning.ar_mask[0])
        wave_implied_mask = np.asarray(generate_wave_ar_mask(bundle.wave, tie_group_map))

        np.testing.assert_array_equal(
          wave_implied_mask,
          exported_mask,
          err_msg=(
            f"order={np.asarray(decoding_order).tolist()} "
            f"ties={np.asarray(tie_group_map).tolist()}"
          ),
        )

  @pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32, jnp.bool_])
  def test_builder_normalises_a_caller_supplied_ar_mask_to_int8(self, dtype: jnp.dtype) -> None:
    length = 5
    order = jnp.arange(length, dtype=jnp.int32)
    tie = jnp.zeros(length, dtype=jnp.int32)
    coords, mask, residue_index, chain_index, _, bias, fixed_mask, fixed_tokens, temperature, _ = (
      self._dummy_inputs(length, order, tie)
    )
    caller_mask = full_context_ar_mask(length).astype(dtype)
    bundle, _ = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      ar_mask=caller_mask,
      bias=bias,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
      tie_group_map=tie,
      temperature=temperature,
      mode="sample_autoregressive",
      inference=True,
    )
    assert bundle.conditioning.ar_mask.dtype == jnp.int8
    assert bundle.conditioning.ar_mask.shape == (1, length, length)

  def test_builder_reads_a_caller_mask_as_nonzero_means_visible(self) -> None:
    length = 4
    order = jnp.arange(length, dtype=jnp.int32)
    tie = jnp.zeros(length, dtype=jnp.int32)
    coords, mask, residue_index, chain_index, _, bias, fixed_mask, fixed_tokens, temperature, _ = (
      self._dummy_inputs(length, order, tie)
    )
    soft = jnp.full((length, length), 0.5, dtype=jnp.float32).at[0, 0].set(0.0)
    bundle, _ = build_inference_bundle(
      coords=coords, mask=mask, residue_index=residue_index, chain_index=chain_index,
      ar_mask=soft, bias=bias, fixed_mask=fixed_mask, fixed_tokens=fixed_tokens,
      tie_group_map=tie, temperature=temperature, mode="sample_autoregressive", inference=True,
    )  # fmt: skip
    got = bundle.conditioning.ar_mask[0]
    assert got.dtype == jnp.int8
    assert int(got[0, 0]) == 0
    assert int(got.sum()) == length * length - 1  # 0.5 is visible, not truncated to 0

  def test_helper_dtype_argument_changes_dtype_not_values(self) -> None:
    order = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    tie = jnp.arange(4, dtype=jnp.int32)
    as_int8 = ar_mask_from_decoding_order(order, tie_group_map=tie)
    as_f32 = ar_mask_from_decoding_order(order, tie_group_map=tie, dtype=jnp.float32)
    assert as_int8.dtype == jnp.int8
    assert as_f32.dtype == jnp.float32
    assert bool(jnp.all(as_int8.astype(jnp.float32) == as_f32))
    # The other two builder-path helpers honour the dtype argument too.
    assert full_context_ar_mask(4).dtype == jnp.int8
    assert full_context_ar_mask(4, dtype=jnp.float32).dtype == jnp.float32
    assert generate_ar_mask(order).dtype == jnp.int8
    assert generate_ar_mask(order, dtype=jnp.float32).dtype == jnp.float32

  def test_exported_p07_graph_carries_no_int8_mask_ops(self) -> None:
    length = 6
    order = jnp.arange(length, dtype=jnp.int32)
    tie = jnp.zeros(length, dtype=jnp.int32)
    args = self._dummy_inputs(length, order, tie)

    def f(decoding_order: jax.Array, tie_group_map: jax.Array) -> jax.Array:
      a = list(args)
      a[4], a[9] = decoding_order, tie_group_map
      bundle, _ = p07_bundle(*a)
      return bundle.conditioning.ar_mask

    jaxpr = str(jax.make_jaxpr(f)(order, tie))
    assert "i8[" not in jaxpr
    assert "u8[" not in jaxpr

  def test_p07_bundle_exported_ar_mask_dtype_contract(self) -> None:
    """Pin the AR mask contract: mask helpers return int8; exported p07_bundle ar_mask stays float32 on purpose (export boundary)."""
    length = 6
    order = jnp.arange(length, dtype=jnp.int32)
    tie = jnp.zeros(length, dtype=jnp.int32)
    bundle, _ = p07_bundle(*self._dummy_inputs(length, order, tie))
    assert bundle.conditioning.ar_mask.dtype == jnp.float32

    wave = bundle.wave
    assert generate_ar_mask(order).dtype == jnp.int8
    assert generate_ar_mask(order, tie_group_map=tie).dtype == jnp.int8
    assert ar_mask_from_decoding_order(order, tie_group_map=tie).dtype == jnp.int8
    assert generate_wave_ar_mask(wave, tie).dtype == jnp.int8
    assert full_context_ar_mask(length).dtype == jnp.int8
