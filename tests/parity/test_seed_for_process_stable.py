"""`_seed_for` must not depend on the per-process str-hash seed (PYTHONHASHSEED).

Regression for the builtin-``hash()`` seed derivation in the layer-(a) exact and sampling
tiers: each fresh interpreter produced different seeds, so runs were not reproducible and a
checkpoint/resume (a new process) would silently change its randomness.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _seed_in_fresh_process(module: str, name: str, hash_seed: str) -> int:
  env = {**os.environ, "PYTHONHASHSEED": hash_seed, "JAX_PLATFORMS": "cpu"}
  code = f"import {module} as m; print(m._seed_for({name!r}))"
  proc = subprocess.run(  # noqa: S603
    [sys.executable, "-c", code],
    cwd=_ROOT,
    env=env,
    capture_output=True,
    text=True,
    check=True,
    timeout=300,
  )
  return int(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
  ("module", "name"),
  [
    ("scripts.browser_validation.layer_a_exact", "1BC8"),
    ("scripts.browser_validation.layer_a_sampling", "1BC8P07@1.0testA1"),
  ],
)
def test_seed_for_is_stable_across_hash_seeds(module: str, name: str) -> None:
  seeds = {_seed_in_fresh_process(module, name, hash_seed) for hash_seed in ("1", "2", "3")}
  assert len(seeds) == 1, f"{module}._seed_for({name!r}) varied with PYTHONHASHSEED: {seeds}"
