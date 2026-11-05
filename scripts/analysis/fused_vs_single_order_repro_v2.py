"""Aminx-only repro v2: which factor produces the fused-vs-single probe-logit residual (synthetic one-free-probe fixture).

Pre-registered in fused_vs_single_order_repro_v2.bth.toml. Needs aminx (>= 0.4.0, with ``fixed_position_context``), jax, equinox, numpy.
Every arm runs the FUSED k=3 path; padding sits inside the coordinate cloud and has a detection control. Result goes to $BTH_RESULTS_PATH.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import os
import re
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

log = logging.getLogger("fused_vs_single_order_repro_v2")

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


def _bundle(num_states: int, geometry: int, pad: int = 0, pad_is_real: bool = False):
  rng = np.random.default_rng(geometry)
  coords = rng.normal(size=(LENGTH, 4, 3)).astype(np.float32)
  tokens = rng.integers(0, 20, size=(LENGTH,))
  n = LENGTH + pad
  if pad:
    # inside the cloud: same distribution as the real residues, so a mask-ignoring k-NN would pick them
    pad_coords = np.random.default_rng(1000 + geometry).normal(size=(pad, 4, 3)).astype(np.float32)
    coords = np.concatenate([coords, pad_coords], axis=0)
    tokens = np.concatenate([tokens, np.zeros(pad, dtype=tokens.dtype)])
  mask = np.ones(n, dtype=np.float32)
  if not pad_is_real:
    mask[LENGTH:] = 0.0
  fixed = np.ones(n, dtype=np.float32)
  fixed[PROBE] = 0.0
  stack = lambda x: jnp.stack([jnp.asarray(x)] * num_states, axis=0)  # noqa: E731
  bundle, config = build_inference_bundle(
    coords=stack(coords), mask=stack(mask), residue_index=stack(np.arange(n, dtype=np.int32)),
    chain_index=stack(np.zeros(n, dtype=np.int32)), state_weights=jnp.ones(num_states),
    fixed_mask=jnp.asarray(fixed), fixed_tokens=jnp.asarray(tokens, dtype=jnp.int32), sequence=None, mode="sample_ar",
  )
  fm = np.asarray(bundle.conditioning.fixed_mask).reshape(-1, n)[0]  # read back from the bundle, not a local array
  return bundle, config, int((fm == 0).sum())


def _logits(model, geometry, k, key, mode, *, n_samples=1, pad=0, pad_is_real=False, seed=None):
  bundle, config, n_free = _bundle(k, geometry, pad, pad_is_real)
  stage_set = make_stage_set(strategy="product", state_weights=jnp.ones(k))
  kwargs = {"fixed_position_context": mode}
  if seed is not None:
    kwargs["decoding_order_seed"] = seed
  _, lg = sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(key), n_samples, **kwargs)
  lg = np.asarray(lg, dtype=np.float64)
  assert lg.ndim == 3, lg.shape
  v = lg[:, PROBE, :]
  return v - v.mean(axis=-1, keepdims=True), n_free


class _Checks:
  """Accumulates the instrument conditions across every logits call."""

  def __init__(self) -> None:
    self.finite = True
    self.free_ok = True
    self.norm_ok = True

  def take(self, arr, n_free):
    self.finite &= bool(np.isfinite(arr).all())
    self.free_ok &= n_free == 1
    self.norm_ok &= bool(np.linalg.norm(arr) > 1e-6)
    return arr


def _rel(a, b):
  return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def _version_ok(version: str) -> bool:
  m = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
  return bool(m) and tuple(int(x) for x in m.groups()) >= MIN_VERSION


def _distribution_ok() -> bool:
  import aminx

  try:
    dist = md.distribution("aminx")
    mod = Path(aminx.__file__).resolve()
    if mod.is_relative_to(Path(str(dist.locate_file(""))).resolve()):
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
  model = _model()
  ck = _Checks()
  r_ctl, r_key, r_ord, r_seed, spr_r, spr_c, r_bs, r_pad, r_padc, live, singles = ([] for _ in range(11))

  def fused_resid(g, mode, ks, kf, **kw):
    single = ck.take(*_logits(model, g, 1, ks, mode, **kw))
    fused = ck.take(*_logits(model, g, K, kf, mode, **kw))
    return _rel(fused[0] / K, single[0])

  for g in GEOMETRIES:
    r_ctl.append(fused_resid(g, "random_order", 0, 1))
    r_key.append(fused_resid(g, "random_order", 0, 0))
    r_ord.append(fused_resid(g, "canonical_order", 0, 1))
    r_seed.append(fused_resid(g, "random_order", 0, 1, seed=SEED))
    s_rand = ck.take(*_logits(model, g, 1, 0, "random_order"))
    singles.append(ck.take(*_logits(model, g, 1, 0, "canonical_order")))
    live.append(_rel(singles[-1][0], s_rand[0]))
    for mode, sink in (("random_order", spr_r), ("canonical_order", spr_c)):
      b = ck.take(*_logits(model, g, K, 0, mode, n_samples=N_BATCH))
      sink.append(max(_rel(b[i], b[0]) for i in range(1, N_BATCH)))
    bc = ck.take(*_logits(model, g, K, 0, "canonical_order", n_samples=N_BATCH))
    one = ck.take(*_logits(model, g, K, 0, "canonical_order", n_samples=1))
    r_bs.append(_rel(bc[0], one[0]))
    padded = ck.take(*_logits(model, g, K, 0, "canonical_order", pad=PAD))
    r_pad.append(_rel(padded[0], one[0]))
    real_arr, real_free = _logits(model, g, K, 0, "canonical_order", pad=PAD, pad_is_real=True)
    ck.finite &= bool(np.isfinite(real_arr).all())
    ck.free_ok &= real_free == 1
    r_padc.append(_rel(real_arr[0], one[0]))
    log.info("g%d ctl %.2e key %.2e ord %.2e seed %.2e spr_r %.2e spr_c %.2e bs %.2e pad %.2e padctl %.2e live %.2e",
             g, r_ctl[-1], r_key[-1], r_ord[-1], r_seed[-1], spr_r[-1], spr_c[-1], r_bs[-1], r_pad[-1], r_padc[-1], live[-1])

  distinct = all(_rel(singles[i][0], singles[j][0]) > 1e-3 for i in range(3) for j in range(i + 1, 3))
  all3 = lambda xs, t: all(x > t for x in xs)  # noqa: E731
  result = {
    "instrument_ok": bool(ck.finite and ck.free_ok and ck.norm_ok and _version_ok(version) and _distribution_ok() and distinct),
    "reproduces_control": all3(r_ctl, TOL_REPRODUCE),
    "modes_are_live": all3(live, TOL_REPRODUCE),
    "key_alone_removes": bool(max(r_key) <= TOL_REMOVE),
    "pinned_order_removes": bool(max(r_ord) <= TOL_REMOVE),
    "seed_removes": bool(max(r_seed) <= TOL_REMOVE),
    "batch_spread_is_order_induced": bool(all3(spr_r, TOL_REPRODUCE) and max(spr_c) <= TOL_REMOVE),
    "batch_size_inert": bool(max(r_bs) <= TOL_REMOVE),
    "padding_control_detects": all3(r_padc, TOL_REPRODUCE),
    "padding_inert": bool(max(r_pad) <= TOL_REMOVE),
    "min_residual_control": float(min(r_ctl)),
    "max_residual_key_pinned": float(max(r_key)),
    "max_residual_order_pinned": float(max(r_ord)),
    "max_residual_seed": float(max(r_seed)),
    "min_spread_random": float(min(spr_r)),
    "max_spread_canonical": float(max(spr_c)),
    "max_residual_batch_size": float(max(r_bs)),
    "max_residual_padding": float(max(r_pad)),
    "min_residual_padding_control": float(min(r_padc)),
    "min_mode_liveness": float(min(live)),
    "provenance_aminx_commit_claimed": args.aminx_commit,
    "provenance_aminx_version": version,
    "provenance_jax_backend": jax.default_backend(),
    "provenance_jax_devices": ",".join(str(d) for d in jax.devices()),
    "provenance_jax_version": jax.__version__,
  }
  out = Path(os.environ.get("BTH_RESULTS_PATH", "fused_vs_single_order_repro_v2.json"))
  out.write_text(json.dumps(result, indent=2))
  log.info(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
