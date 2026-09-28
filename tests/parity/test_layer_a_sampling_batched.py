"""T9 (sampling track): batched-harness AR mask staleness (F-S1) + incremental plumbing.

Covers, in one file (spec "Task 9", r1 M12 gate):

- (a) **Structural** (`test_per_draw_ar_mask_structural`): `layer_a_sampling._bundle_for_draw`
  is the exact seam `_vmapped_sample`'s jitted `one(key, wave)` calls per draw -- this test
  calls it directly (host-side, un-jitted, un-vmapped) for 3 draws with distinct decoding
  orders and asserts each draw's rebuilt bundle carries an `ar_mask` recomputed from ITS OWN
  wave (`generate_wave_ar_mask(waves[i], tie_group_map)`), not the wave that built the shared
  bundle. Also demonstrates, as a permanent regression guard, that the OLD construction
  (`eqx.tree_at(lambda x: x.wave, bundle, wave)` alone -- pre-fix `_vmapped_sample`) produces a
  DIFFERENT (stale) `ar_mask` for `i >= 1`, which is F-S1 itself (V15).
- (b) **Empirical** (`test_batched_vs_unbatched_empirical`): batched `aminx_sample_batch` draws
  vs the unbatched `sample_autoregressive.kernel` called per-draw with a freshly-built bundle,
  on a synthetic L=96 structure (falls back to synthetic always; 5L33 only if
  `$PROTEINMPNN_PATH` is set, per spec -- never skipped).
- **Host predicate mirror** (`test_host_predicates_match_device`,
  `test_force_equals_off_when_host_true`, `test_planted_inconsistent_mask_not_forced`):
  `layer_a_sampling.incremental_predicates_host` against a from-scratch jnp re-derivation of
  the same three device expressions (`autoregressive.py:591-756`, D-F/V19).
- **Regression** (`test_kernel_incremental_default_bitwise`): `sample_autoregressive.kernel`'s
  new `incremental` kwarg defaults to `"auto"` -- the pre-existing call shape is bitwise
  unchanged.

A small synthetic `Aminx` model is used throughout (no real checkpoint/torch dependency),
mirroring `tests/inference/decode/test_incremental_autoregressive.py`'s own convention -- the
bug this task fixes is in harness WIRING (which `ar_mask`/`wave` reaches the kernel), not model
numerics, so a tiny deterministic model is sufficient and keeps this file free of the heavy
`--extra=benchmark`/reference-checkout dependency chain.
"""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scripts.browser_validation.layer_a_sampling as las
from aminx.inference import sample_autoregressive
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.model.mpnn import Aminx
from aminx.types.bundles import WaveScheduleBundle
from aminx.types.configs import InferenceConfig
from aminx.utils.autoregression import generate_wave_ar_mask

L_DEFAULT = 96
LOGIT_ATOL = 1e-5


@pytest.fixture(scope="module")
def model() -> Aminx:
  m = Aminx(
    node_features=48,
    edge_features=48,
    hidden_features=48,
    num_encoder_layers=2,
    num_decoder_layers=3,
    k_neighbors=12,
    key=jax.random.PRNGKey(0),
  )
  # Amplify the sequence embedding, as test_incremental_autoregressive.py's own model
  # fixture does: at random init a sequence-channel wiring bug is invisible at 1x scale.
  m = eqx.tree_at(lambda mm: mm.w_s_embed.weight, m, m.w_s_embed.weight * 30.0)
  return eqx.tree_inference(m, value=True)


def _synthetic_batch(
  length: int = L_DEFAULT, seed: int = 1, n_fixed: int = 0, n_real: int | None = None
) -> las.LaneBatch:
  """A synthetic `LaneBatch` (random-walk backbone, no real fixture I/O).

  `n_real < length` zeroes `mask` for the trailing `length - n_real` positions
  (structural padding, distinct from `chain_mask=0` fixed-but-real positions) --
  needed to exercise `valid_nbr` (an invalid/padded position CAN be selected as a
  top-k neighbour when there are fewer real neighbours than `k`, `features.py`'s
  `k = min(self.k_neighbors, L)` clamp, V25).
  """
  key = jax.random.PRNGKey(seed)
  steps = jax.random.normal(key, (length, 1, 3)) * 2.0
  ca = jnp.cumsum(steps, axis=0)
  offsets = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 3)) * 0.8
  x4 = np.asarray(ca + offsets, dtype=np.float32)
  mask = np.ones(length, dtype=np.float32)
  if n_real is not None and n_real < length:
    mask[n_real:] = 0.0
  chain_mask = np.ones(length, dtype=np.float32)
  if n_fixed:
    rng = np.random.default_rng(seed + 100)
    idx = rng.choice(length, size=n_fixed, replace=False)
    chain_mask[idx] = 0.0
  fixed_mask = 1.0 - mask * chain_mask
  seq_ref = np.zeros(length, dtype=np.int32)
  tie_group_map = np.arange(length, dtype=np.int64)
  comparison_positions = np.where((mask > 0) & (chain_mask > 0))[0].astype(np.int64)
  return las.LaneBatch(
    lane="P07@0.1",
    fixture_name="synthetic96",
    length=length,
    x4=x4,
    atom37=np.zeros((length, 37, 3), dtype=np.float32),
    atom37_mask=np.zeros((length, 37), dtype=np.float32),
    seq_ref=seq_ref,
    mask=mask,
    residue_index=np.arange(length, dtype=np.int64),
    chain_index=np.zeros(length, dtype=np.int32),
    chain_mask=chain_mask,
    fixed_mask=fixed_mask,
    bias=las._x_omit_bias(length),  # noqa: SLF001 -- reuse, not reimplement
    reference_bias=np.zeros((length, 21), dtype=np.float32),
    tie_group_map=tie_group_map,
    groups=None,
    use_side_chain_context=False,
    comparison_positions=comparison_positions,
  )


def _bundle_kwargs(batch: las.LaneBatch, wave: WaveScheduleBundle, *, temperature: float) -> dict:
  """Mirrors `layer_a_sampling.aminx_sample_batch`'s own `build_inference_bundle` kwargs."""
  return {
    "coords": jnp.asarray(batch.x4),
    "mask": jnp.asarray(batch.mask),
    "residue_index": jnp.asarray(batch.residue_index, dtype=jnp.int32),
    "chain_index": jnp.asarray(batch.chain_index, dtype=jnp.int32),
    "chain_mask": jnp.asarray(batch.chain_mask),
    "bias": jnp.asarray(batch.bias),
    "fixed_mask": jnp.asarray(batch.fixed_mask),
    "fixed_tokens": jnp.asarray(batch.seq_ref, dtype=jnp.int32),
    "tie_group_map": jnp.asarray(batch.tie_group_map),
    "wave": wave,
    "temperature": float(temperature),
    "mode": "sample",
  }


def _order(length: int, seed: int) -> np.ndarray:
  return np.asarray(jax.random.permutation(jax.random.PRNGKey(seed), length))


# --------------------------------------------------------------------------------------
# (a) Structural: the per-draw bundle passed to the kernel carries its OWN ar_mask (F-S1)
# --------------------------------------------------------------------------------------


def test_per_draw_ar_mask_structural() -> None:
  batch = _synthetic_batch(length=32, seed=2)
  orders = [_order(32, seed) for seed in (11, 12, 13)]
  waves = [las.wave_from_tie_groups_np(batch.tie_group_map, order) for order in orders]

  bundle, _config = build_inference_bundle(**_bundle_kwargs(batch, waves[0], temperature=1.0))
  tie_group_map_row = bundle.conditioning.tie_group_map[0]

  for i, wave in enumerate(waves):
    rebuilt = las._bundle_for_draw(bundle, wave)  # noqa: SLF001 -- the seam under test
    expected_2d = generate_wave_ar_mask(wave, tie_group_map_row)
    expected = jnp.broadcast_to(expected_2d[None, ...], bundle.conditioning.ar_mask.shape)
    np.testing.assert_array_equal(np.asarray(rebuilt.conditioning.ar_mask), np.asarray(expected))
    # Also confirm the wave itself was swapped (not just left at wave[0]).
    np.testing.assert_array_equal(np.asarray(rebuilt.wave.group_ids), np.asarray(wave.group_ids))
    if i >= 1:
      # F-S1 regression guard: the PRE-FIX construction (swap `.wave` only, leave
      # `.conditioning.ar_mask` at whatever built the shared `bundle` -- draw 0's wave)
      # must disagree with the correct per-draw ar_mask for every later draw. If this
      # assertion ever starts failing, the two waves stopped being distinguishable and
      # the test fixture (not the fix) needs revisiting -- it would silently stop
      # exercising F-S1 at all.
      stale = eqx.tree_at(lambda x: x.wave, bundle, wave)  # noqa: B023
      assert not np.array_equal(np.asarray(stale.conditioning.ar_mask), np.asarray(expected)), (
        f"draw {i}: stale (pre-fix) ar_mask unexpectedly already matches the correct one"
      )


# --------------------------------------------------------------------------------------
# (b) Empirical: batched draws vs the unbatched per-draw kernel call
# --------------------------------------------------------------------------------------


def _unbatched_draw(
  model: Aminx, batch: las.LaneBatch, order: np.ndarray, seed: int, *, temperature: float
) -> np.ndarray:
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  bundle, config = build_inference_bundle(**_bundle_kwargs(batch, wave, temperature=temperature))
  result = sample_autoregressive.kernel(
    model, jax.random.PRNGKey(seed), bundle, config, make_stage_set(), inference_only=True
  )
  return np.asarray(result.sequence)


def test_batched_vs_unbatched_empirical(model: Aminx) -> None:
  proteinmpnn_path = os.environ.get("PROTEINMPNN_PATH")
  fixtures: list[tuple[str, las.LaneBatch]] = [("synthetic96", _synthetic_batch(length=96))]
  if proteinmpnn_path and Path(proteinmpnn_path).exists():
    # Real-fixture arm is best-effort/opportunistic; never a hard dependency (spec:
    # "never skipped, it falls back to synthetic"). Silently skip only THIS arm if
    # any part of the real-fixture loading path is unavailable.
    try:  # noqa: SIM105
      import json as _json

      manifest_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "browser_validation"
        / "fixtures"
        / "manifest.json"
      )
      manifest = _json.loads(manifest_path.read_text())
      fixture = next(f for f in manifest["fixtures"] if f["name"] == "5L33")
      import scripts.browser_validation.layer_a_exact as lae  # noqa: PLC0415

      data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001
      real_batch = las.build_lane_batch(fixture, "P07@0.1", data_utils_module)
      fixtures.append(("5L33", real_batch))
    except Exception:  # noqa: BLE001
      pass

  n = 3
  temperature = 0.1
  for name, batch in fixtures:
    seed_base = las._seed_for(f"{name}batchedvsunbatched")  # noqa: SLF001
    batched = las.aminx_sample_batch(model, batch, n, seed_base, temperature=temperature)
    max_mismatch = 0
    for i in range(n):
      _randn, order_i = las._draw_order_for(batch, seed_base + i)  # noqa: SLF001
      unbatched = _unbatched_draw(model, batch, order_i, seed_base + i, temperature=temperature)
      mismatch = int(np.sum(batched[i] != unbatched))
      max_mismatch = max(max_mismatch, mismatch)
      # Pre-registered bar (spec T9 step 1(b)): <= 1 position per draw after the fix
      # (fused-XLA-vmap vs eager rounding differences may still exist, per
      # aminx_sample_batch's own docstring -- "same sampler, not bit-identical").
      assert mismatch <= 1, (
        f"{name} draw {i}: {mismatch} mismatched positions between batched and "
        f"unbatched (bar: <= 1)"
      )
    # Recorded (not asserted further): pre-fix this was e.g. "0/85, 15/85, 10/85" on
    # 5L33 (V15) -- draw 0 exact, every other draw wrong at many positions.
    assert max_mismatch <= 1


# --------------------------------------------------------------------------------------
# incremental=default is unchanged (regression guard for the new kernel kwarg)
# --------------------------------------------------------------------------------------


def test_kernel_incremental_default_bitwise(model: Aminx) -> None:
  batch = _synthetic_batch(length=32, seed=3)
  order = _order(32, 21)
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  bundle, config = build_inference_bundle(**_bundle_kwargs(batch, wave, temperature=1.0))
  key = jax.random.PRNGKey(7)
  stage_set = make_stage_set()

  default_call = sample_autoregressive.kernel(model, key, bundle, config, stage_set)
  explicit_auto = sample_autoregressive.kernel(
    model, key, bundle, config, stage_set, incremental="auto"
  )
  np.testing.assert_array_equal(
    np.asarray(default_call.sequence), np.asarray(explicit_auto.sequence)
  )
  np.testing.assert_array_equal(np.asarray(default_call.logits), np.asarray(explicit_auto.logits))


# --------------------------------------------------------------------------------------
# Host predicate mirror vs. a from-scratch jnp re-derivation of the device expressions
# --------------------------------------------------------------------------------------


def _device_predicates_jax(bundle, enc) -> dict[str, bool]:
  """A from-scratch jnp re-derivation of `autoregressive.py`'s three predicates
  (independent of `layer_a_sampling.incremental_predicates_host`, so this test does
  not just compare the host function against itself)."""
  cond = bundle.conditioning
  wave = bundle.wave
  n_waves, max_groups_per_wave = wave.group_ids.shape
  L = cond.tie_group_map.shape[-1]

  pos0_grid = wave.group_positions[:, :, 0]
  real_group_id_grid = cond.tie_group_map[0, pos0_grid]
  wave_index_grid = jnp.broadcast_to(
    jnp.arange(n_waves, dtype=jnp.int32)[:, None], (n_waves, max_groups_per_wave)
  )
  slot_grid = jnp.broadcast_to(
    jnp.arange(max_groups_per_wave, dtype=jnp.int32)[None, :], (n_waves, max_groups_per_wave)
  )
  combined_rank_grid = wave_index_grid * max_groups_per_wave + slot_grid
  no_occurrence_sentinel = n_waves * max_groups_per_wave
  flat_group_id = jnp.where(wave.group_valid, real_group_id_grid, 0).reshape(-1)
  flat_rank = jnp.where(wave.group_valid, combined_rank_grid, no_occurrence_sentinel).reshape(-1)
  group_first_rank = (
    jnp.full((L,), no_occurrence_sentinel, dtype=jnp.int32).at[flat_group_id].min(flat_rank)
  )
  pos_rank = group_first_rank[cond.tie_group_map[0]]
  decode_wave = jnp.where(
    pos_rank < no_occurrence_sentinel, pos_rank // max_groups_per_wave, n_waves
  ).astype(jnp.int32)
  order_pos = jnp.argsort(decode_wave, stable=True)
  wave_start = jnp.searchsorted(
    decode_wave[order_pos], jnp.arange(n_waves + 1, dtype=jnp.int32), side="left"
  ).astype(jnp.int32)

  nbr_all = enc.neighbor_indices
  ar_neighbors = jnp.take_along_axis(cond.ar_mask, nbr_all, axis=2)
  valid_row = enc.mask > 0.5
  valid_nbr = (
    jnp.take_along_axis(enc.mask.reshape(1, -1), nbr_all.reshape(1, -1), axis=1).reshape(
      nbr_all.shape
    )
    > 0.5
  )
  reads = valid_row[..., None] & valid_nbr & (ar_neighbors > 0.5)
  wave_row = decode_wave[None, :, None]
  wave_nbr = decode_wave[nbr_all]
  consistent = bool(~jnp.any(reads & (wave_nbr > wave_row)))

  identity_frame = bool(
    jnp.all(cond.state_position_map == jnp.arange(L, dtype=cond.state_position_map.dtype)[None, :])
  )

  slab = max_groups_per_wave * wave.group_positions.shape[2]
  slab = min(slab, L)
  fits_slab = bool(jnp.all((wave_start[1:] - wave_start[:-1]) <= slab))

  return {"consistent": consistent, "identity_frame": identity_frame, "fits_slab": fits_slab}


def _host_predicates_for_bundle(bundle, enc, *, max_positions_per_wave: int | None = None):
  return las.incremental_predicates_host(
    wave=bundle.wave,
    tie_group_map=np.asarray(bundle.conditioning.tie_group_map[0]),
    ar_mask=np.asarray(bundle.conditioning.ar_mask[0]),
    neighbor_indices=np.asarray(enc.neighbor_indices[0]),
    valid_mask=np.asarray(enc.mask[0]),
    state_position_map=np.asarray(bundle.conditioning.state_position_map[0]),
    max_positions_per_wave=max_positions_per_wave,
  )


def _encode(model: Aminx, bundle, config: InferenceConfig):
  return make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)


@pytest.mark.parametrize(
  "case",
  [
    "no_fixed",  # (i)
    "fixed",  # (ii)
    "inconsistent_mask",  # (iii)
    "non_identity_frame",  # (iv)
    "wave_exceeds_slab",  # (v)
    "valid_nbr_load_bearing",  # extra: isolates the `valid_nbr` term (red-check target)
  ],
)
def test_host_predicates_match_device(model: Aminx, case: str) -> None:
  if case == "wave_exceeds_slab":
    # (v): a wave whose real occupancy (per tie_group_map) exceeds its own declared
    # per-wave capacity -- a malformed/inconsistent (tie_group_map, wave) pair, built
    # by hand since `WaveScheduleBundle.from_tie_groups` never emits an inconsistent
    # one itself. 5 positions share a tie group id the WAVE schedule (built untied,
    # one position per group/wave) never grouped together, so `decode_wave` (computed
    # from `tie_group_map` alone) assigns all 5 to the wave of their earliest member,
    # while the wave's own `group_positions.shape[2] * max_groups_per_wave` capacity
    # is 1. Bypasses build_inference_bundle/make_encode_fn (no ordinary construction
    # reaches this malformed state); `fits_slab` depends only on `wave`/`tie_group_map`.
    length = 16
    tie_group_map = np.arange(length, dtype=np.int64)
    tied = [0, 1, 2, 3, 4]
    tie_group_map[tied] = tied[0]
    untied_map_for_wave = np.arange(length, dtype=np.int64)
    wave = las.wave_from_tie_groups_np(untied_map_for_wave, np.arange(length, dtype=np.int64))
    assert wave.group_positions.shape[2] == 1
    host = las.incremental_predicates_host(
      wave=wave,
      tie_group_map=tie_group_map,
      ar_mask=np.zeros((length, length), dtype=np.float32),
      neighbor_indices=np.zeros((length, 1), dtype=np.int64),
      valid_mask=np.ones(length, dtype=np.float32),
      state_position_map=np.arange(length, dtype=np.int64),
    )
    assert host["fits_slab"] is False
    return

  if case == "valid_nbr_load_bearing":
    # Isolates the `valid_nbr` term with hand-built, fully-controlled inputs (the
    # natural encode pipeline's top-k neighbour search does not let us place a
    # violation at a specific, chosen (row, slot) pair deterministically). L=4,
    # untied, identity order -> decode_wave = [0, 1, 2, 3] (position i decodes at
    # wave i). ar_mask is the correct causal mask (lower triangular: j visible to i
    # iff j < i) EXCEPT ar_mask[0, 3] = 1, a deliberately planted "row 0 can see
    # position 3" entry -- position 3 decodes at wave 3 > row 0's wave 0. Row 0's
    # neighbour slot 0 points at position 3, which is INVALID (`valid_mask[3] = 0`,
    # padding, not merely fixed). With `valid_nbr` in place this pair is correctly
    # excluded from `reads` (an invalid neighbour contributes no evidence either
    # way) and `consistent` is True; drop `valid_nbr` and this pair alone flips it
    # to False -- this is exactly the mutation the T9 gate's red-check names.
    length = 4
    tie_group_map = np.arange(length, dtype=np.int64)
    order = np.arange(length, dtype=np.int64)
    wave = las.wave_from_tie_groups_np(tie_group_map, order)
    ar_mask = np.tril(np.ones((length, length), dtype=np.float32), k=-1)
    ar_mask[0, 3] = 1.0  # the planted violation, only visible if valid_nbr is dropped
    neighbor_indices = np.array([[3, 1], [0, 2], [0, 1], [0, 1]], dtype=np.int64)
    valid_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # position 3 is padding
    state_position_map = np.arange(length, dtype=np.int64)
    host = las.incremental_predicates_host(
      wave=wave,
      tie_group_map=tie_group_map,
      ar_mask=ar_mask,
      neighbor_indices=neighbor_indices,
      valid_mask=valid_mask,
      state_position_map=state_position_map,
    )
    assert host["consistent"] is True, (
      "an invalid (padded) neighbour must not count as evidence of an "
      f"inconsistent schedule: {host}"
    )
    return

  length = 32
  n_real = None
  batch = _synthetic_batch(
    length=length, seed=4, n_fixed=10 if case == "fixed" else 0, n_real=n_real
  )
  order = _order(length, 31)
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  kwargs = _bundle_kwargs(batch, wave, temperature=1.0)

  if case == "inconsistent_mask":
    # (iii): an ar_mask from a DIFFERENT order than the wave -- read-before-decode.
    other_order = _order(length, 987)
    rank = np.empty_like(other_order)
    rank[other_order] = np.arange(length)
    kwargs["ar_mask"] = jnp.asarray((rank[None, :] < rank[:, None]).astype(np.float32))

  bundle, config = build_inference_bundle(**kwargs)

  if case == "non_identity_frame":
    # (iv): a non-identity state_position_map (reverse the reference frame).
    reversed_map = jnp.asarray(np.arange(length)[::-1].copy())
    bundle = eqx.tree_at(
      lambda b: b.conditioning.state_position_map,
      bundle,
      jnp.broadcast_to(reversed_map[None, :], bundle.conditioning.state_position_map.shape),
    )

  enc = _encode(model, bundle, config)
  host = _host_predicates_for_bundle(bundle, enc)
  device = _device_predicates_jax(bundle, enc)
  assert host == device, f"case={case}: host={host} device={device}"


def test_force_equals_off_when_host_true(model: Aminx) -> None:
  """Whenever the host verdict is True, one draw decoded with `force` matches `off`
  (tokens exact, logits <= 1e-5) -- the incremental fastpath is exact when its
  preconditions genuinely hold."""
  length = 32
  batch = _synthetic_batch(length=length, seed=5)
  order = _order(length, 41)
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  bundle, config = build_inference_bundle(**_bundle_kwargs(batch, wave, temperature=1.0))
  enc = _encode(model, bundle, config)
  host = _host_predicates_for_bundle(bundle, enc)
  assert all(host.values()), f"expected an all-True host verdict on this construction: {host}"

  stage_set = make_stage_set()
  key = jax.random.PRNGKey(9)
  off = sample_autoregressive.kernel(model, key, bundle, config, stage_set, incremental="off")
  force = sample_autoregressive.kernel(model, key, bundle, config, stage_set, incremental="force")
  np.testing.assert_array_equal(np.asarray(off.sequence), np.asarray(force.sequence))
  np.testing.assert_allclose(
    np.asarray(off.logits), np.asarray(force.logits), atol=LOGIT_ATOL, rtol=0
  )


def test_planted_inconsistent_mask_not_forced(model: Aminx) -> None:
  """(iii): a planted inconsistent ar_mask gets `"off"` -- `auto` must reproduce full
  recompute rather than the (silently wrong) incremental cache path. `force` on the
  same input is the negative control proving the precondition is load-bearing."""
  length = 32
  batch = _synthetic_batch(length=length, seed=6)
  order = _order(length, 51)
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  kwargs = _bundle_kwargs(batch, wave, temperature=1.0)
  other_order = _order(length, 777)
  rank = np.empty_like(other_order)
  rank[other_order] = np.arange(length)
  kwargs["ar_mask"] = jnp.asarray((rank[None, :] < rank[:, None]).astype(np.float32))
  bundle, config = build_inference_bundle(**kwargs)
  enc = _encode(model, bundle, config)
  host = _host_predicates_for_bundle(bundle, enc)
  assert host["consistent"] is False

  stage_set = make_stage_set()
  key = jax.random.PRNGKey(13)
  full = sample_autoregressive.kernel(model, key, bundle, config, stage_set, incremental="off")
  auto = sample_autoregressive.kernel(model, key, bundle, config, stage_set, incremental="auto")
  np.testing.assert_array_equal(np.asarray(full.sequence), np.asarray(auto.sequence))
  np.testing.assert_allclose(
    np.asarray(full.logits), np.asarray(auto.logits), atol=LOGIT_ATOL, rtol=0
  )
  forced = sample_autoregressive.kernel(model, key, bundle, config, stage_set, incremental="force")
  assert not np.allclose(np.asarray(forced.logits), np.asarray(full.logits), atol=LOGIT_ATOL)


# --------------------------------------------------------------------------------------
# T9b: `aminx_sample_batch` routes `incremental` on the HOST (never "auto" under vmap)
# --------------------------------------------------------------------------------------


def test_choose_incremental_mode_all_true_selects_force(model: Aminx) -> None:
  """On a normal harness-constructed batch, every draw's host verdict is all-True by
  this harness's own invariants: `ar_mask` is always regenerated FROM the same wave
  (`generate_wave_ar_mask`, never a stale/foreign one -- F-S1), `state_position_map` is
  always identity (no rolling state), and `fits_slab`'s slab is sized from the SAME
  `wave_from_tie_groups_np` construction that built the wave (one tie group per wave, by
  that function's own docstring) -- this is the T9 budget-floor's own empirical finding
  (`n_draws_forced_off == 0` on every real lane). Locks in that invariant at the
  `_choose_incremental_mode` aggregation level, with the REAL (unmocked) predicate."""
  batch = _synthetic_batch(length=24, seed=61)
  waves = [las.wave_from_tie_groups_np(batch.tie_group_map, _order(24, s)) for s in (71, 72, 73)]
  mode = las._choose_incremental_mode(model, batch, waves, temperature=1.0)  # noqa: SLF001
  assert mode == "force"


def test_choose_incremental_mode_any_false_selects_off(
  model: Aminx, monkeypatch: pytest.MonkeyPatch
) -> None:
  """(T9b routing test) `_choose_incremental_mode` must return exactly what the
  aggregation rule says: "force" iff EVERY draw's own wave verdict is all-True, else
  "off". The real predicate is always all-True on this harness's own wave construction
  (previous test), so a normal harness-constructed batch can never naturally exercise
  the OFF branch -- this test controls the verdict directly via monkeypatch (the same
  technique `test_planted_inconsistent_mask_not_forced` uses at the kernel level, here
  applied at the aggregation level) to prove the "any draw fails -> off for the whole
  chunk" rule, matching `_vmapped_sample_at`'s own docstring (`incremental` is a single
  static field shared by the whole vmapped batch, so it cannot be chosen per-draw)."""
  batch = _synthetic_batch(length=24, seed=62)
  waves = [las.wave_from_tie_groups_np(batch.tie_group_map, _order(24, s)) for s in (81, 82, 83)]

  calls = {"n": 0}

  def flaky(**_kwargs: object) -> dict[str, bool]:
    calls["n"] += 1
    # The SECOND draw's wave fails `consistent`; every other draw's wave passes -- a
    # single failing draw among several passing ones must still force "off" for the
    # whole (homogeneous) chunk, not skip just that one draw.
    return {"consistent": calls["n"] != 2, "identity_frame": True, "fits_slab": True}

  monkeypatch.setattr(las, "incremental_predicates_host", flaky)
  mode = las._choose_incremental_mode(model, batch, waves, temperature=1.0)  # noqa: SLF001
  assert mode == "off"

  monkeypatch.setattr(
    las,
    "incremental_predicates_host",
    lambda **_kwargs: {"consistent": True, "identity_frame": True, "fits_slab": True},
  )
  mode_all_true = las._choose_incremental_mode(model, batch, waves, temperature=1.0)  # noqa: SLF001
  assert mode_all_true == "force"


def test_aminx_sample_batch_exact_vs_off_when_force_chosen(model: Aminx) -> None:
  """(T9b exactness, force-chosen lane) `aminx_sample_batch`'s host-routed output is
  TOKEN-IDENTICAL to an explicit `incremental="off"` call with the same seeds -- the
  fastpath must be exact, not merely close, when its own preconditions genuinely hold
  (mirrors `test_force_equals_off_when_host_true`'s kernel-level guarantee, at the full
  `aminx_sample_batch` plumbing level, where routing is real/unmocked)."""
  batch = _synthetic_batch(length=24, seed=63)
  seed_base = las._seed_for("t9b_force_lane")  # noqa: SLF001
  n = 3
  routed = las.aminx_sample_batch(model, batch, n, seed_base, temperature=1.0)
  forced_off = las.aminx_sample_batch_at_incremental(
    model, batch, n, seed_base, temperature=1.0, incremental="off"
  )
  np.testing.assert_array_equal(routed, forced_off)


def test_aminx_sample_batch_exact_vs_off_when_off_chosen(
  model: Aminx, monkeypatch: pytest.MonkeyPatch
) -> None:
  """(T9b exactness, off-chosen lane) With the host predicate monkeypatched to fail (the
  only way to exercise this branch on a harness-constructed batch, see
  `test_choose_incremental_mode_any_false_selects_off`), `aminx_sample_batch`'s routed
  output is STILL token-identical to an explicit `incremental="off"` call with the same
  seeds -- the delegation to `aminx_sample_batch_at_incremental` reproduces the exact
  same waves/keys/chunking, never drifting the draw sequence."""
  batch = _synthetic_batch(length=24, seed=64)
  seed_base = las._seed_for("t9b_off_lane")  # noqa: SLF001
  n = 3
  monkeypatch.setattr(
    las,
    "incremental_predicates_host",
    lambda **_kwargs: {"consistent": False, "identity_frame": True, "fits_slab": True},
  )
  routed = las.aminx_sample_batch(model, batch, n, seed_base, temperature=1.0)
  forced_off = las.aminx_sample_batch_at_incremental(
    model, batch, n, seed_base, temperature=1.0, incremental="off"
  )
  np.testing.assert_array_equal(routed, forced_off)


# --------------------------------------------------------------------------------------
# T9 remediation regression: `layer_a_sampling_budget_floor._n_draws_forced_off` must not
# raise for ANY lane, on a real fixture -- the synthetic model above never exercises
# `use_side_chain_context`, so it could not have caught the P11-s bug this guards.
# --------------------------------------------------------------------------------------

_ONE_FIXTURE_ALLOCATION_CACHE_NAME = "1BC8"  # smallest set-B fixture resolvable WITHOUT
# $PROTEINMPNN_PATH (only needs $REFERENCE_PATH, present on every dev box that has the
# LigandMPNN reference clone at all) -- L=113, real PDB, real atom_37 data.


@pytest.mark.parity_heavy
@pytest.mark.parametrize("lane", las.LANE_KEYS)
def test_budget_floor_n_draws_forced_off_every_lane(lane: str) -> None:
  """`layer_a_sampling_budget_floor._n_draws_forced_off` -- the function whose per-lane
  bundle construction this task's root cause lives in -- must not raise for ANY lane in
  `las.LANE_KEYS`, on a real fixture. Builds the model + per-lane batch via the EXACT
  same reused helpers the budget-floor script's own `run()` calls
  (`las.full_model_bundle_for_lane`, `layer_a_sampling_calibrate._lane_fixture_batches`)
  rather than re-deriving lane inputs here, then calls `_n_draws_forced_off` itself (the
  function under test, not a reimplementation of it) for one allocated draw.

  Regression guard for T9: pre-fix, the P11-s lane's per-lane bundle kwargs omitted
  `atom_37`/`atom_37_mask`/`ligand_*` (now `las.side_chain_context_kwargs`), crashing
  with `ValueError: atom_37 and atom_37_mask must be provided when use_side_chains=True`
  (`ligand_features.py:359`) the first time a real lane -- rather than this file's other
  tests' synthetic, non-side-chain-context `Aminx` model -- exercised it. `P09-s@1.0` on
  this fixture has no k-NN-disjoint qualifying tie group (`tie_groups_knn_disjoint: []`
  in the manifest), so `_lane_fixture_batches` filters it out and the loop below runs
  zero iterations for that lane -- still a valid "did not raise" pass, matching what the
  real `run()` does when no set-B fixture qualifies for a lane.

  Marked `parity_heavy`: needs the real reference checkpoint + torch
  (`load_full_model`/`load_sidechain_context_models`), unlike this file's synthetic-model
  tests above. Run locally with a narrow selection, e.g.:
  `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run pytest
  tests/parity/test_layer_a_sampling_batched.py -k test_budget_floor_n_draws_forced_off_every_lane
  -m parity_heavy`.
  """
  import json

  import scripts.browser_validation.layer_a_exact as lae  # noqa: PLC0415
  import scripts.browser_validation.layer_a_sampling_budget_floor as lasbf  # noqa: PLC0415
  import scripts.browser_validation.layer_a_sampling_calibrate as lasc  # noqa: PLC0415

  manifest_path = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "browser_validation"
    / "fixtures"
    / "manifest.json"
  )
  manifest = json.loads(manifest_path.read_text())
  fixture = next(f for f in manifest["fixtures"] if f["name"] == _ONE_FIXTURE_ALLOCATION_CACHE_NAME)
  data_utils_module = lae._load_reference_data_utils()  # noqa: SLF001 -- reuse, not reimplement

  lane_model = las.full_model_bundle_for_lane(lane, "eqx")
  lane_jax = lane_model[0]
  batches = lasc._lane_fixture_batches([fixture], lane, data_utils_module)  # noqa: SLF001
  n_off = lasbf._n_draws_forced_off(  # noqa: SLF001 -- the function under test, not reimplemented
    lane_jax, batches, lane, {fixture["name"]: 1}
  )
  assert isinstance(n_off, int)
  assert n_off >= 0
