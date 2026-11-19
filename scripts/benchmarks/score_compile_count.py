"""Count XLA backend compiles across repeated identical-shape ``runner.score`` calls (debt #3484).

tev_design's teacher-forced screen calls ``aminx.host.runner.score`` once per decoding order, with
``ar_mask=(L, L)``, ``return_logits=True`` and ``backbone_noise=0``. On 0.5.3 each call took about
3.1 s on CPU at L=214 (padded to 512), of which about 3.0 s looked like recompilation. This probe
measures that directly: it makes ``--calls`` identical-shape calls in one process (the ar_mask
VALUES change between calls, as they do across orders; the shapes do not) and counts backend
compiles per call through ``jax.monitoring``'s ``BACKEND_COMPILE_EVENT``.

The instrument control is call 1: a cold process MUST compile, so ``compiles_call1 == 0`` means
the counter is blind and nothing else in the record is interpretable.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("score_compile_count")

_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


class _CompileCounter:
  """Counts backend compiles; read ``count`` before and after a call."""

  def __init__(self) -> None:
    self.count = 0
    jax.monitoring.register_event_duration_secs_listener(self._on_event)

  def _on_event(self, event: str, _duration: float, **_kwargs: object) -> None:
    if event == _BACKEND_COMPILE_EVENT:
      self.count += 1


def _git_hash() -> str:
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--structure", default="tests/data/1ubq.pdb")
  parser.add_argument("--checkpoint", default="proteinmpnn_v_48_020")
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--calls", type=int, default=3)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415
  from aminx.utils.autoregression import ar_mask_from_decoding_order  # noqa: PLC0415

  counter = _CompileCounter()
  length = len(_UBIQUITIN)
  rng = np.random.default_rng(0)
  per_call: list[dict[str, float | int]] = []
  for call in range(args.calls):
    order = rng.permutation(length).astype(np.int32)
    ar_mask = np.asarray(ar_mask_from_decoding_order(jnp.asarray(order)))
    spec = ScoringSpecification(
      inputs=[args.structure],
      checkpoint_id=args.checkpoint,
      sequences_to_score=[_UBIQUITIN],
      max_length=args.max_length,
      return_logits=True,
      ar_mask=ar_mask,
      backbone_noise=0.0,
    )
    before = counter.count
    start = time.perf_counter()
    result = score(spec)
    jax.block_until_ready(result["logits"])
    wall = time.perf_counter() - start
    compiles = counter.count - before
    per_call.append({"call": call + 1, "compiles": compiles, "wall_s": wall})
    logger.info("call %d: %d backend compiles, %.3f s", call + 1, compiles, wall)

  payload = {
    "compiles_call1": int(per_call[0]["compiles"]),
    "compiles_call2": int(per_call[1]["compiles"]),
    "compiles_call3": int(per_call[2]["compiles"]) if len(per_call) > 2 else -1,
    "wall_s_call1": float(per_call[0]["wall_s"]),
    "wall_s_call2": float(per_call[1]["wall_s"]),
    "per_call": per_call,
    "length": length,
    "max_length": args.max_length,
    "checkpoint": args.checkpoint,
    "backend": jax.default_backend(),
    "jax_version": jax.__version__,
    "git_hash": _git_hash(),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(payload))
  logger.info("wrote %s", args.out)


if __name__ == "__main__":
  main()
