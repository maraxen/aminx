"""Standalone aminx-only reproduction: which factor produces the fused-vs-single probe-logit residual?

Pre-registered in fused_vs_single_order_repro.bth.toml. Needs only aminx (>= 0.4.0, #266 branch for ``fixed_position_context``), jax,
equinox and numpy. No tev_design import, no checkpoint. Run under bathos; the result goes to $BTH_RESULTS_PATH.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.sampling.multistate_poe import sample_states_fused

log = logging.getLogger("fused_vs_single_order_repro")

LENGTH = 16
PAD = 8
PROBE = 7
WEIGHT_SCALE = 4.0
GEOMETRIES = (5, 6, 7)
K = 3
N_BATCH = 4
SEED = 11
TOL_REPRODUCE = 1e-3
TOL_REMOVE = 1e-4
MIN_VERSION = (0, 4, 0)


def _model():
  model = Aminx(
    node_features=64, edge_features=64, hidden_features=64, num_encoder_layers=2, num_decoder_layers=2,
    k_neighbors=5, dropout_rate=0.0, key=jax.random.PRNGKey(3),
  )
  scaled = jax.tree_util.tree_map(lambda x: x * WEIGHT_SCALE if eqx.is_inexact_array(x) else x, model)
  return eqx.tree_inference(scaled, value=True)


def _bundle(num_states: int, geometry: int, pad: int = 0):
  rng = np.random.default_rng(geometry)
  coords = rng.normal(size=(LENGTH, 4, 3)).astype(np.float32)
  tokens = rng.integers(0, 20, size=(LENGTH,))
  n = LENGTH + pad
  if pad:
    pad_coords = np.random.default_rng(1000 + geometry).normal(size=(pad, 4, 3)).astype(np.float32) + 50.0
    coords = np.concatenate([coords, pad_coords], axis=0)
    tokens = np.concatenate([tokens, np.zeros(pad, dtype=tokens.dtype)])
  mask = np.ones(n, dtype=np.float32)
  mask[LENGTH:] = 0.0
  fixed = np.ones(n, dtype=np.float32)
  fixed[PROBE] = 0.0
  stack = lambda x: jnp.stack([jnp.asarray(x)] * num_states, axis=0)  # noqa: E731
  return build_inference_bundle(
    coords=stack(coords), mask=stack(mask), residue_index=stack(np.arange(n, dtype=np.int32)),
    chain_index=stack(np.zeros(n, dtype=np.int32)), state_weights=jnp.ones(num_states),
    fixed_mask=jnp.asarray(fixed), fixed_tokens=jnp.asarray(tokens, dtype=jnp.int32), sequence=None, mode="sample_ar",
  ), int((fixed == 0).sum())


def _logits(model, geometry, k, key, mode, *, n_samples=1, pad=0, seed=None):
  (bundle, config), n_free = _bundle(k, geometry, pad)
  stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(k))
  kwargs = {"fixed_position_context": mode}
  if seed is not None:
    kwargs["decoding_order_seed"] = seed
  _, lg = sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(key), n_samples, **kwargs)
  lg = np.asarray(lg, dtype=np.float64)
  assert lg.ndim == 3, lg.shape
  v = lg[:, PROBE, :]
  return v - v.mean(axis=-1, keepdims=True), n_free


def _rel(a, b):
  return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def _residual(model, g, mode, key_single, key_fused, **kw):
  single, nf1 = _logits(model, g, 1, key_single, mode, **kw)
  fused, nfk = _logits(model, g, K, key_fused, mode, **kw)
  return _rel(fused[0] / K, single[0]), single[0], (nf1, nfk)


def _distribution_ok(version: str) -> bool:
  import aminx

  try:
    dist = md.distribution("aminx")
    root = Path(str(dist.locate_file(""))).resolve()
    mod = Path(aminx.__file__).resolve()
    if mod.is_relative_to(root):
      return True
    direct = dist.read_text("direct_url.json")
    if direct:
      url = json.loads(direct).get("url", "")
      if url.startswith("file://"):
        return mod.is_relative_to(Path(url[7:]).resolve())
  except Exception:  # noqa: BLE001
    log.exception("distribution check failed")
  return False


def main() -> int:
  """Run every arm on every geometry and write the graded booleans and magnitudes."""
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--aminx-commit", required=True, help="aminx commit under test, recorded as a CLAIM")
  args = ap.parse_args()

  version = md.version("aminx")
  version_ok = tuple(int(p) for p in version.split(".")[:3] if p.isdigit()) >= MIN_VERSION
  model = _model()

  r_ctl, r_key, r_ord, r_seed, spread_r, spread_c, r_bs, r_pad, live = [], [], [], [], [], [], [], [], []
  r_pad_random_info, norms, singles, free_ok, finite = [], [], [], True, True
  for g in GEOMETRIES:
    rc, s_rand, nf = _residual(model, g, "random_order", 0, 1)
    free_ok &= all(n == 1 for n in nf)
    r_ctl.append(rc)
    r_key.append(_residual(model, g, "random_order", 0, 0)[0])
    r_ord.append(_residual(model, g, "canonical_order", 0, 1)[0])
    r_seed.append(_residual(model, g, "random_order", 0, 1, seed=SEED)[0])
    singles.append(_logits(model, g, 1, 0, "canonical_order")[0][0])
    live.append(_rel(singles[-1], s_rand))
    norms.append(float(np.linalg.norm(singles[-1])))
    finite &= bool(np.isfinite(s_rand).all() and np.isfinite(singles[-1]).all())
    for mode, sink in (("random_order", spread_r), ("canonical_order", spread_c)):
      batch, _ = _logits(model, g, 1, 0, mode, n_samples=N_BATCH)
      sink.append(max(_rel(batch[i], batch[0]) for i in range(1, N_BATCH)))
    batch_c, _ = _logits(model, g, 1, 0, "canonical_order", n_samples=N_BATCH)
    one_c, _ = _logits(model, g, 1, 0, "canonical_order", n_samples=1)
    r_bs.append(_rel(batch_c[0], one_c[0]))
    padded, _ = _logits(model, g, 1, 0, "canonical_order", pad=PAD)
    r_pad.append(_rel(padded[0], one_c[0]))
    r_pad_random_info.append(_rel(_logits(model, g, 1, 0, "random_order", pad=PAD)[0][0], _logits(model, g, 1, 0, "random_order")[0][0]))
    log.info("geometry %d: control %.3e key_pinned %.3e order_pinned %.3e seed %.3e spread_r %.3e spread_c %.3e batch %.3e pad %.3e live %.3e",
             g, r_ctl[-1], r_key[-1], r_ord[-1], r_seed[-1], spread_r[-1], spread_c[-1], r_bs[-1], r_pad[-1], live[-1])

  distinct = all(_rel(singles[i], singles[j]) > 1e-3 for i in range(3) for j in range(i + 1, 3))
  ge2 = lambda xs, t: sum(x > t for x in xs) >= 2  # noqa: E731
  result = {
    "instrument_ok": bool(finite and free_ok and min(norms) > 1e-6 and version_ok and _distribution_ok(version) and distinct),
    "reproduces_control": bool(ge2(r_ctl, TOL_REPRODUCE)),
    "modes_are_live": bool(ge2(live, TOL_REPRODUCE)),
    "key_alone_removes": bool(max(r_key) <= TOL_REMOVE),
    "pinned_order_removes": bool(max(r_ord) <= TOL_REMOVE),
    "seed_removes": bool(max(r_seed) <= TOL_REMOVE),
    "batch_spread_is_order_induced": bool(ge2(spread_r, TOL_REPRODUCE) and max(spread_c) <= TOL_REMOVE),
    "batch_size_inert": bool(max(r_bs) <= TOL_REMOVE),
    "padding_inert": bool(max(r_pad) <= TOL_REMOVE),
    "min_residual_control": float(min(r_ctl)),
    "max_residual_key_pinned": float(max(r_key)),
    "max_residual_order_pinned": float(max(r_ord)),
    "max_residual_seed": float(max(r_seed)),
    "min_spread_random": float(min(spread_r)),
    "max_spread_canonical": float(max(spread_c)),
    "max_residual_batch_size": float(max(r_bs)),
    "max_residual_padding": float(max(r_pad)),
    "min_mode_liveness": float(min(live)),
    # provenance and ungraded reporting
    "provenance_aminx_commit_claimed": args.aminx_commit,
    "provenance_aminx_version": version,
    "provenance_jax_backend": jax.default_backend(),
    "provenance_jax_version": jax.__version__,
    "info_padding_random_order_unpadded_vs_padded_max": float(max(r_pad_random_info)),
  }
  out = Path(os.environ.get("BTH_RESULTS_PATH", "fused_vs_single_order_repro.json"))
  out.write_text(json.dumps(result, indent=2))
  log.info(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
