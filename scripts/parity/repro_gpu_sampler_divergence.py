#!/usr/bin/env python3
"""Reproduce the GPU-only divergence of aminx's sampler logits (aminx #2326 GPU run, bathos 239505fd).

EXPLORATORY DIAGNOSTIC, not a pre-registered experiment: it exists so the divergence can be reproduced and bisected.
Numbers quoted from it are uncounted; the registered evidence is the GPU run of scripts/parity/e2e_run_api_parity.py.

What it does: in ONE process, runs the same ``aminx.run.sample`` call (same seed, same injected decoding order, the
reference's chain-C feature dict) once under ``jax.default_device(gpu)`` and once under ``jax.default_device(cpu)``, then
reports whether sequences and per-position log-probabilities agree.  Observed on a TITAN RTX (jax 0.10.2, CUDA):

* KNOB=base   N<=384, padded length 96 : GPU == CPU == reference (max |dlogp| ~3e-5)
* KNOB=base   N=512,  padded length 96 : GPU disagrees with the reference by ~1 nat; CPU agrees
* KNOB=pad512 N=4,    padded length 512: GPU disagrees by up to ~2 nats from the FIRST decoded position (0.087) onward

Not the cause (each tested, results unchanged): memory pool size, matmul precision (highest), XLA autotuning off,
CUDA-graph command buffers off, aminx's top_k (correct on GPU for every failing shape), the encoder (GPU == CPU to
1e-6 with the same padding), the planner's axis strategy (identical on both platforms), mode="drop" scatters.

Usage (titanix, from a tree with the aminx fix applied, GPU index chosen explicitly):
  CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda,cpu REFERENCE_PATH=/home/solab/bv/ref/LigandMPNN \\
    KNOB=pad512 N=4 python scripts/parity/repro_gpu_sampler_divergence.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import jax
import numpy as np

_SPEC = importlib.util.spec_from_file_location("e2e", Path(__file__).with_name("e2e_run_api_parity.py"))
e2e = importlib.util.module_from_spec(_SPEC)
sys.modules["e2e"] = e2e
_SPEC.loader.exec_module(e2e)


def main() -> None:
  knob = os.environ.get("KNOB", "pad512")
  n = int(os.environ.get("N", "4"))
  ctx = e2e.ref_context("pmpnn", knob)
  fd, out = ctx["fd"], ctx["out"]
  order = out["decoding_order"][0].numpy()
  designed = e2e._designed(fd)
  gpu, cpu = jax.devices()[0], jax.devices("cpu")[0]
  if gpu.platform == "cpu":
    msg = "no GPU visible: set JAX_PLATFORMS=cuda,cpu and CUDA_VISIBLE_DEVICES"
    raise SystemExit(msg)
  results = {}
  for name, dev in (("gpu", gpu), ("cpu", cpu)):
    with jax.default_device(dev):
      results[name] = e2e.shared_sample("pmpnn", knob, ctx, n_draws=n)
  (seq_g, log_g), (seq_c, log_c) = results["gpu"], results["cpu"]
  print(f"knob={knob} n={n} padded_length={e2e._pad(knob)} gpu={gpu.device_kind}")
  print("sequences identical:", bool((seq_g == seq_c).all()), "| fraction of tokens equal:", float((seq_g == seq_c).mean()))
  diff = np.abs(e2e._lsm(log_g[0])[:, :20] - e2e._lsm(log_c[0])[:, :20]).max(-1)
  by_rank = np.array([diff[p] for p in order if designed[p]])
  print(f"sample-0 log-prob max|GPU-CPU| = {by_rank.max():.4f}; positions > 1e-3: {int((by_rank > 1e-3).sum())} of {len(by_rank)}")
  print("per-decoding-rank error, first 20:", np.round(by_rank[:20], 3).tolist())


if __name__ == "__main__":
  main()
