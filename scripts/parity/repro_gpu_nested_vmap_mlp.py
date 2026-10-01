"""Exploratory repro (aminx debt #2391): does a plain Linear-GELU-Linear MLP give a different answer on GPU than on CPU when it
sits under nested size-1 ``jax.vmap`` axes?  No aminx code is imported -- only jax + equinox.

Context: the e2e run-API parity run on a TITAN RTX showed the sampler's encoder output differing by ~0.17 from CPU
only when the encode is vmapped over a size-1 noise axis (the runner's default), localised to the encoder layer's
``dense`` MLP (128 -> 512 -> 128, exact GELU).  This script strips everything else away.

EXPLORATORY: numbers it prints are diagnostics, not findings.  Anything cited from it must be re-run under a bathos sidecar.

Usage (a CPU device must stay visible, hence ``cuda,cpu``):
  CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda,cpu uv run python scripts/parity/repro_gpu_nested_vmap_mlp.py [--length 512]
"""

import argparse
import logging
import os
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("repro_gpu_nested_vmap_mlp")

ACTIVATIONS = {
  "gelu_exact": partial(jax.nn.gelu, approximate=False),
  "gelu_tanh": partial(jax.nn.gelu, approximate=True),
  "relu": jax.nn.relu,
  "identity": lambda x: x,
}
# (outer vmap sizes applied outermost-first, e.g. (1, 1) = vmap(vmap(f)) over two size-1 axes); () = no extra vmap.
NESTINGS = [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2), (1, 1, 1)]


def build_mlp(width: int, hidden: int, activation: Callable[[jax.Array], jax.Array]) -> eqx.nn.MLP:
  """The aminx EncoderLayer.dense shape: in=out=128, one hidden layer of 512."""
  return eqx.nn.MLP(in_size=width, out_size=width, width_size=hidden, depth=1, activation=activation, key=jax.random.PRNGKey(0))


def run_once(mlp: eqx.nn.MLP, x: np.ndarray, nesting: tuple[int, ...], device: jax.Device, *, jitted: bool) -> np.ndarray:
  """Apply ``vmap(mlp)`` over residues, wrapped in one size-N vmap per entry of ``nesting``; return the un-batched result."""
  with jax.default_device(device):
    mlp_d = jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, device) if eqx.is_array(leaf) else leaf, mlp)
    xd = jax.device_put(jnp.asarray(x), device)

    def inner(h: jax.Array) -> jax.Array:
      return jax.vmap(mlp_d)(h)

    f = inner
    for _ in nesting:
      f = jax.vmap(f)
    batched = xd
    for n in reversed(nesting):  # tile outermost-last so the leading axes line up with the vmaps
      batched = jnp.broadcast_to(batched, (n, *batched.shape))
    fn = jax.jit(f) if jitted else f
    out = np.asarray(fn(batched))
  for _ in nesting:
    out = out[0]
  return out


def numpy_truth(mlp: eqx.nn.MLP, x: np.ndarray) -> np.ndarray:
  """float64 forward of the identity-activation MLP (two affine layers), independent of any JAX backend."""
  l0, l1 = mlp.layers
  h = x.astype(np.float64) @ np.asarray(l0.weight, np.float64).T + np.asarray(l0.bias, np.float64)
  return h @ np.asarray(l1.weight, np.float64).T + np.asarray(l1.bias, np.float64)


def diagnose(gpu: jax.Device, cpu: jax.Device, width: int, hidden: int) -> None:
  """Who is wrong (vs a float64 numpy truth), and where: error by length, and the shape of the wrong region."""
  logger.info("--- diagnose: identity MLP, jitted vmap over one size-1 axis vs un-vmapped, float64 numpy truth ---")
  logger.info("%-7s %-12s %-12s %-12s  wrong-rows (|err|>1e-3 of GPU vmap(1))", "length", "GPU vmap(1)", "GPU plain", "CPU vmap(1)")
  mlp = build_mlp(width, hidden, ACTIVATIONS["identity"])
  rng = np.random.default_rng(0)
  lengths = [int(v) for v in os.environ["LENGTHS"].split(",")] if os.environ.get("LENGTHS") else [1, 8, 64, 96, 128, 256, 511, 512, 513, 1024]
  for length in lengths:
    x = rng.standard_normal((length, width)).astype(np.float32)
    truth = numpy_truth(mlp, x)
    g1 = run_once(mlp, x, (1,), gpu, jitted=True)
    gp = run_once(mlp, x, (), gpu, jitted=True)
    c1 = run_once(mlp, x, (1,), cpu, jitted=True)
    err = np.abs(g1 - truth)
    rows = np.where(err.max(1) > 1e-3)[0]
    where = f"{len(rows)} rows [{rows.min()}..{rows.max()}]" if len(rows) else "none"
    logger.info("%-7d %-12.3g %-12.3g %-12.3g  %s", length, err.max(), np.abs(gp - truth).max(), np.abs(c1 - truth).max(), where)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--length", type=int, default=512, help="residue axis length (the failing case used 512)")
  parser.add_argument("--width", type=int, default=int(os.environ.get("WIDTH", 128)))
  parser.add_argument("--hidden", type=int, default=int(os.environ.get("HIDDEN", 512)))
  parser.add_argument("--quick", action="store_true", default=bool(os.environ.get("QUICK")), help="identity + exact GELU, nestings (), (1,), (2,), jitted only")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  gpu, cpu = jax.devices()[0], jax.devices("cpu")[0]
  if gpu.platform == "cpu":
    msg = "needs an accelerator as devices()[0]; run with JAX_PLATFORMS=cuda,cpu"
    raise SystemExit(msg)
  logger.info("devices: %s vs %s | jax %s | length %d", gpu, cpu, jax.__version__, args.length)

  if os.environ.get("DIAGNOSE"):
    diagnose(gpu, cpu, args.width, args.hidden)
    return
  rng = np.random.default_rng(0)
  x = rng.standard_normal((args.length, args.width)).astype(np.float32)
  logger.info("%-11s %-10s %-6s  max|GPU-CPU|", "activation", "nesting", "jit")
  acts = {k: v for k, v in ACTIVATIONS.items() if k in ("identity", "gelu_exact")} if args.quick else ACTIVATIONS
  nestings = [(), (1,), (2,)] if args.quick else NESTINGS
  for act_name, act in acts.items():
    mlp = build_mlp(args.width, args.hidden, act)
    for nesting in nestings:
      for jitted in ((True,) if args.quick else (True, False)):
        g = run_once(mlp, x, nesting, gpu, jitted=jitted)
        c = run_once(mlp, x, nesting, cpu, jitted=jitted)
        logger.info("%-11s %-10s %-6s  %.3g", act_name, str(nesting), jitted, float(np.abs(g - c).max()))


if __name__ == "__main__":
  main()
