"""Incremental AR decode is exactly the full-recompute decode, at O(1) decoder work per wave.

`AutoregressiveDecode(incremental="off")` runs the decoder over all L positions at every wave
and keeps the wave's rows (O(L^2 k) per sequence). The incremental path decodes only the wave's
positions against a per-layer cache (O(L k) per sequence, like the reference's `h_V_stack`).
These tests pin the two to the same tokens and logits on every schedule shape the sampler
supports, check that `"auto"` falls back when the incremental preconditions fail (with a
negative control showing the fallback is load-bearing), and guard the complexity itself: no
matmul inside the wave loop may grow with L.
"""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.profiling.loop_scaling import extent_scaling_report, loop_bodies

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.model.mpnn import Aminx
from aminx.tiling.strategy import SafeMap, Vmap
from aminx.types.bundles import WaveScheduleBundle
from aminx.utils.autoregression import generate_ar_mask

L_DEFAULT = 24
LOGIT_ATOL = 1e-5


@pytest.fixture(scope="module")
def model() -> Aminx:
  m = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=3,
    k_neighbors=8,
    key=jax.random.PRNGKey(0),
  )
  # Amplify the sequence embedding. At random init one neighbour's token moves a row's
  # hidden features by ~4e-3 (the /30 message scale plus LayerNorm), and a second hop --
  # a later row reading a tied row's CACHED layer-1 features -- by ~4e-6, below any sane
  # logit tolerance. Measured: dropping the post-draw cache refresh moved logits by
  # 2.9e-6 at 1x (invisible) but 2.6e-4 at 30x; with the refresh the gap stays 9.5e-7
  # (float noise) at every scale. 30x makes sequence-channel bugs visible at 1e-5.
  m = eqx.tree_at(lambda mm: mm.w_s_embed.weight, m, m.w_s_embed.weight * 30.0)
  return eqx.tree_inference(m, value=True)


def _structure(length: int, n_states: int = 1, seed: int = 1) -> dict[str, jax.Array]:
  """A random-walk backbone (distinct neighbours, no degenerate ties)."""
  key = jax.random.PRNGKey(seed)
  steps = jax.random.normal(key, (n_states, length, 1, 3)) * 2.0
  ca = jnp.cumsum(steps, axis=1)
  offsets = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 4, 3)) * 0.8
  coords = ca + offsets
  return {
    "coords": coords if n_states > 1 else coords[0],
    "mask": jnp.ones((n_states, length)) if n_states > 1 else jnp.ones((length,)),
    "residue_index": jnp.broadcast_to(jnp.arange(length), (n_states, length))
    if n_states > 1
    else jnp.arange(length),
    "chain_index": jnp.zeros((n_states, length) if n_states > 1 else (length,), dtype=jnp.int32),
  }


def _decode(
  model,
  bundle,
  config,
  *,
  incremental: str,
  inference_only: bool = False,
  seed: int = 3,
  strategy=None,
):
  key = jax.random.PRNGKey(seed)
  k_enc, k_dec = jax.random.split(key)
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, k_enc, config)
  decode_fn = make_decode_fn(
    model=model,
    mode=AutoregressiveMode(),
    strategy=strategy if strategy is not None else Vmap(),
    autoregressive_config=AutoregressiveConfig(
      inference_only=inference_only, incremental=incremental
    ),
  )
  return eqx.filter_jit(decode_fn)(k_dec, enc, bundle, config, make_stage_set())


def _assert_same(a, b) -> None:
  np.testing.assert_array_equal(np.asarray(a.sequence), np.asarray(b.sequence))
  np.testing.assert_allclose(np.asarray(a.logits), np.asarray(b.logits), atol=LOGIT_ATOL, rtol=0)


def _bundle(length: int = L_DEFAULT, **kw):
  s = _structure(length, n_states=kw.pop("n_states", 1))
  return build_inference_bundle(
    s["coords"], s["mask"], s["residue_index"], s["chain_index"], mode="sample_autoregressive", **kw
  )


def _random_wave(length: int, tie_group_map: jax.Array | None = None, seed: int = 11):
  order = jax.random.permutation(jax.random.PRNGKey(seed), length)
  tie = tie_group_map if tie_group_map is not None else jnp.arange(length)
  return WaveScheduleBundle.from_tie_groups(tie, order)


def _tie_map(length: int) -> jax.Array:
  tie = np.arange(length)
  for a, b in ((0, 5), (5, 9), (2, 17), (11, 12)):  # {0,5,9}, {2,17}, {11,12}
    tie[b] = tie[a]
  return jnp.asarray(tie, dtype=jnp.int32)


@pytest.mark.parametrize("incremental", ["force", "auto"])
def test_default_schedule_matches_full_recompute(model, incremental) -> None:
  bundle, config = _bundle(temperature=0.7)
  _assert_same(
    _decode(model, bundle, config, incremental="off"),
    _decode(model, bundle, config, incremental=incremental),
  )


def test_random_order_matches_full_recompute(model) -> None:
  bundle, config = _bundle(wave=_random_wave(L_DEFAULT), temperature=1.0)
  _assert_same(
    _decode(model, bundle, config, incremental="off"),
    _decode(model, bundle, config, incremental="force"),
  )


def test_tied_groups_match_full_recompute(model) -> None:
  """Same-wave visibility exercises the post-draw cache refresh."""
  tie = _tie_map(L_DEFAULT)
  bundle, config = _bundle(tie_group_map=tie, wave=_random_wave(L_DEFAULT, tie), temperature=1.0)
  full = _decode(model, bundle, config, incremental="off")
  inc = _decode(model, bundle, config, incremental="force")
  _assert_same(full, inc)
  seq = np.asarray(inc.sequence)
  assert seq[0] == seq[5] == seq[9] and seq[2] == seq[17] and seq[11] == seq[12]


def test_fixed_positions_and_bias_match_full_recompute(model) -> None:
  fixed_mask = jnp.zeros((L_DEFAULT,)).at[jnp.array([1, 7, 13])].set(1.0)
  fixed_tokens = (
    jnp.zeros((L_DEFAULT,), dtype=jnp.int32).at[jnp.array([1, 7, 13])].set(jnp.array([4, 9, 15]))
  )
  bias = jax.random.normal(jax.random.PRNGKey(5), (L_DEFAULT, 21))
  bundle, config = _bundle(
    wave=_random_wave(L_DEFAULT),
    fixed_mask=fixed_mask,
    fixed_tokens=fixed_tokens,
    bias=bias,
    temperature=0.5,
  )
  full = _decode(model, bundle, config, incremental="off")
  inc = _decode(model, bundle, config, incremental="force")
  _assert_same(full, inc)
  assert [int(inc.sequence[i]) for i in (1, 7, 13)] == [4, 9, 15]


def test_while_loop_matches_full_recompute(model) -> None:
  bundle, config = _bundle(wave=_random_wave(L_DEFAULT), temperature=1.0)
  _assert_same(
    _decode(model, bundle, config, incremental="off", inference_only=True),
    _decode(model, bundle, config, incremental="force", inference_only=True),
  )


def test_multistate_identity_map_matches_full_recompute(model) -> None:
  bundle, config = _bundle(n_states=2, wave=_random_wave(L_DEFAULT), temperature=1.0)
  _assert_same(
    _decode(model, bundle, config, incremental="off"),
    _decode(model, bundle, config, incremental="auto"),
  )


def test_single_decoder_layer_under_safemap_matches_full_recompute() -> None:
  """A 1-layer decoder has no cached layers; SafeMap (lax.map) must not see a zero-size cache."""
  m = Aminx(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=jax.random.PRNGKey(4),
  )
  m = eqx.tree_inference(
    eqx.tree_at(lambda mm: mm.w_s_embed.weight, m, m.w_s_embed.weight * 30.0), value=True
  )
  bundle, config = _bundle(n_states=2, wave=_random_wave(L_DEFAULT), temperature=1.0)
  _assert_same(
    _decode(m, bundle, config, incremental="off", strategy=SafeMap(tile=1)),
    _decode(m, bundle, config, incremental="force", strategy=SafeMap(tile=1)),
  )


def test_auto_falls_back_on_inconsistent_ar_mask(model) -> None:
  """An ar_mask from a different order than the wave breaks the cache's premise.

  `auto` must detect it and reproduce full recompute. The negative control -- `force` on the
  same input differs -- proves the precondition is load-bearing, not vacuous.
  """
  other = jax.random.permutation(jax.random.PRNGKey(99), L_DEFAULT)
  rank = jnp.empty_like(other).at[other].set(jnp.arange(L_DEFAULT))
  inconsistent = generate_ar_mask(rank)
  bundle, config = _bundle(wave=_random_wave(L_DEFAULT), ar_mask=inconsistent, temperature=1.0)
  full = _decode(model, bundle, config, incremental="off")
  _assert_same(full, _decode(model, bundle, config, incremental="auto"))
  forced = _decode(model, bundle, config, incremental="force")
  assert not np.allclose(np.asarray(forced.logits), np.asarray(full.logits), atol=LOGIT_ATOL)


# --------------------------------------------------------------------------------------
# Complexity guard: per-wave decoder work must not grow with L
# --------------------------------------------------------------------------------------


GUARD_EXTENT = 32
# A loop whose trip count grows less than this between L and 2L is not a per-position loop.
# jnp.searchsorted's binary-search scan runs about log2 L trips (one more at 2L) and its body
# carries O(L) work, so xtrax flags its per-iteration ratio (~1.97); it is O(L log L) in
# total, not the O(L^2) this guard looks for (xtrax debt #2505).
TRIP_SCALING_RATIO = 1.5


def _decode_scaling(model, incremental: str) -> tuple[list[tuple[object, int | None, int | None]], int]:
  """Per-loop scaling of the decode between L=GUARD_EXTENT and 2L, with trip counts.

  Returns ``(rows, n_length_loops)``: one ``(finding, trips_at_L, trips_at_2L)`` per loop
  body, and how many loops have a trip count that scales with L (a ``while`` has no static
  trip count and counts as scaling).
  """
  decode_fn = make_decode_fn(
    model=model,
    mode=AutoregressiveMode(),
    strategy=Vmap(),
    autoregressive_config=AutoregressiveConfig(incremental=incremental),
  )

  def make_args(length: int):
    bundle, config = _bundle(length=length, wave=_random_wave(length), temperature=1.0)
    enc = make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)
    return (jax.random.PRNGKey(1), enc, bundle, config)

  def fn(key, enc, bundle, config):
    return decode_fn(key, enc, bundle, config, make_stage_set())

  report = extent_scaling_report(fn, make_args, GUARD_EXTENT)
  # The report pairs bodies by structural path but drops trip counts; recover them from
  # the same two traces (findings follow loop_bodies order, outermost first).
  small = loop_bodies(fn, *make_args(GUARD_EXTENT))
  large = loop_bodies(fn, *make_args(2 * GUARD_EXTENT))
  assert [b.path for b in small] == [f.path for f in report.findings]
  rows = [
    (finding, s.trip_count, g.trip_count)
    for finding, s, g in zip(report.findings, small, large, strict=True)
  ]
  n_length_loops = sum(1 for _, s, g in rows if _trips_scale(s, g))
  return rows, n_length_loops


def _trips_scale(small: int | None, large: int | None) -> bool:
  if small is None or large is None:
    return True
  return small > 0 and large / small >= TRIP_SCALING_RATIO


def _quadratic_loops(rows) -> list[str]:
  """Loops whose trip count AND per-iteration work both grow with L (O(L^2) total)."""
  return [
    f"{f.path or '<root>'} {f.primitive} work {f.work_at_extent}->{f.work_at_double_extent} "
    f"(x{f.ratio:.2f}), trips {s}->{g}"
    for f, s, g in rows
    if f.flagged and _trips_scale(s, g)
  ]


def test_incremental_wave_body_work_is_independent_of_length(model) -> None:
  rows, n_length_loops = _decode_scaling(model, "force")
  assert n_length_loops, "no loop whose trip count scales with L -- the guard would be vacuous"
  assert not _quadratic_loops(rows), _quadratic_loops(rows)


def test_full_recompute_wave_body_work_grows_with_length(model) -> None:
  """Negative control: the guard above must be able to see O(L) per-wave work."""
  rows, _ = _decode_scaling(model, "off")
  assert _quadratic_loops(rows), "full recompute was not flagged; the guard cannot fail"
