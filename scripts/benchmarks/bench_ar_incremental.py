"""Incremental vs full-recompute autoregressive decode: per-sequence time against L.

Pre-registered in ``bench_ar_incremental.bth.toml``. For each length, one encoder pass is
shared, then the AR decode is timed (jit-compiled, compile excluded, median of
``--repeats``, ``block_until_ready``) under ``incremental="off"`` (decoder over all L rows
every wave), ``"force"`` (wave rows only, per-layer cache) and ``"auto"`` (the on-device
precondition check, which must pick the incremental branch on this input). Tokens from
``off`` and ``force`` must be identical at every length.

The model is production-shaped (128-dim, 3+3 layers, k = 48) with random weights: decode
cost depends on shapes only, and equivalence is checked on the same weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.model.mpnn import Aminx
from aminx.tiling.strategy import Vmap

logger = logging.getLogger("bench_ar_incremental")
MODES = ("off", "force", "auto")


def _structure(length: int, seed: int) -> tuple[jax.Array, ...]:
  key = jax.random.PRNGKey(seed)
  ca = jnp.cumsum(jax.random.normal(key, (length, 1, 3)) * 2.2, axis=0)
  coords = ca + jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 3)) * 0.8
  return coords, jnp.ones((length,)), jnp.arange(length), jnp.zeros((length,), dtype=jnp.int32)


def _time_length(model: Aminx, length: int, repeats: int) -> dict[str, Any]:
  coords, mask, residue_index, chain_index = _structure(length, seed=length)
  bundle, config = build_inference_bundle(
    coords, mask, residue_index, chain_index, mode="sample_autoregressive", temperature=1.0
  )
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)
  stage_set = make_stage_set()
  row: dict[str, Any] = {}
  sequences = {}
  for mode in MODES:
    decode = eqx.filter_jit(
      make_decode_fn(
        model=model,
        mode=AutoregressiveMode(),
        strategy=Vmap(),
        autoregressive_config=AutoregressiveConfig(inference_only=True, incremental=mode),
      )
    )
    t0 = time.perf_counter()
    jax.block_until_ready(decode(jax.random.PRNGKey(7), enc, bundle, config, stage_set))
    compile_s = time.perf_counter() - t0
    times = []
    for r in range(repeats):
      t0 = time.perf_counter()
      out = jax.block_until_ready(decode(jax.random.PRNGKey(7 + r), enc, bundle, config, stage_set))
      times.append(time.perf_counter() - t0)
      if r == 0:
        sequences[mode] = np.asarray(out.sequence)
    row[mode] = float(np.median(times))
    row[f"{mode}_compile"] = compile_s
    logger.info("L=%d %-5s median %.4fs (compile %.1fs)", length, mode, row[mode], compile_s)
  row["tokens_equal"] = bool(np.array_equal(sequences["off"], sequences["force"]))
  return row


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lengths", default="128,256,512")
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  lengths = sorted(int(x) for x in args.lengths.split(","))
  model = eqx.tree_inference(
    Aminx(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      num_encoder_layers=3,
      num_decoder_layers=3,
      k_neighbors=48,
      key=jax.random.PRNGKey(0),
    ),
    value=True,
  )
  rows = {length: _time_length(model, length, args.repeats) for length in lengths}
  lo, hi = lengths[0], lengths[-1]
  result: dict[str, Any] = {
    "lengths": lengths,
    "per_length": {str(k): v for k, v in rows.items()},
    "speedup_max_len": rows[hi]["off"] / rows[hi]["force"],
    "inc_scaling": rows[hi]["force"] / rows[lo]["force"],
    "full_scaling": rows[hi]["off"] / rows[lo]["off"],
    "length_ratio": hi / lo,
    "auto_overhead": rows[hi]["auto"] / rows[hi]["force"],
    "tokens_equal": all(r["tokens_equal"] for r in rows.values()),
    "backend": jax.default_backend(),
    "git_hash": subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip(),
  }
  for key in ("speedup_max_len", "inc_scaling", "full_scaling", "auto_overhead"):
    if not math.isfinite(result[key]):
      msg = f"{key} is not finite: {result[key]}"
      raise ValueError(msg)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
  logger.info(
    "speedup@%d=%.1fx inc_scaling=%.2f full_scaling=%.2f auto_overhead=%.2f tokens_equal=%s",
    hi,
    result["speedup_max_len"],
    result["inc_scaling"],
    result["full_scaling"],
    result["auto_overhead"],
    result["tokens_equal"],
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
