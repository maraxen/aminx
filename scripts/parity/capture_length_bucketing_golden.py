"""Capture the S8 opt-out golden: runner sample/score outputs at the pre-bucketing base.

S8 (``.praxia/docs/specs/261006_runner-length-bucketing-xtrax-adoption.md``) makes the
runner bucket residue length by default. Its G-OPTOUT gate requires that
``length_bucketing=False`` reproduces today's outputs bit for bit, so this script must
be run on the commit BEFORE any S8 code change. It records that commit in the
metadata, and the gate test refuses a fixture captured anywhere else.

Writes ``<out>/optout_golden.npz`` and ``<out>/optout_golden.json`` (cases, base
commit, versions).

Usage (titanix, CPU):
  uv run --no-sync python scripts/parity/capture_length_bucketing_golden.py \
    --out tests/fixtures/length_bucketing
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("capture_length_bucketing_golden")

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
CASES = (
  {"name": "1ubq", "pdb": "1ubq.pdb", "num_samples": 2, "temperature": 0.1, "random_seed": 7},
  {"name": "5awl", "pdb": "5awl.pdb", "num_samples": 2, "temperature": 0.1, "random_seed": 7},
)
MAX_LENGTH = 512
SCORE_SEED = 42


def _commit() -> str:
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=_DATA
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def main() -> int:
  parser = argparse.ArgumentParser(description="Capture the S8 opt-out runner golden.")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--base-commit", default=None, help="override when the tree is not a git checkout")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  import jax

  from aminx import __version__ as aminx_version
  from aminx.host.runner import sample, score
  from aminx.utils.aa_convert import protein_sequence_to_string

  arrays: dict[str, np.ndarray] = {}
  meta: dict[str, object] = {
    "base_commit": args.base_commit or _commit(),
    "aminx_version": aminx_version,
    "jax_version": jax.__version__,
    "backend": jax.default_backend(),
    "max_length": MAX_LENGTH,
    "score_seed": SCORE_SEED,
    "cases": [],
  }
  for case in CASES:
    pdb = str(_DATA / str(case["pdb"]))
    logger.info("sample %s", case["name"])
    sampled = sample(
      inputs=[pdb],
      num_samples=case["num_samples"],
      temperature=case["temperature"],
      random_seed=case["random_seed"],
      max_length=MAX_LENGTH,
    )
    for key in ("sequences", "logits"):
      if key in sampled and sampled[key] is not None:
        arrays[f"{case['name']}__sample__{key}"] = np.asarray(sampled[key])
    tokens = np.asarray(sampled["sequences"]).reshape(-1, np.asarray(sampled["sequences"]).shape[-1])
    sequences = [protein_sequence_to_string(row) for row in tokens]
    logger.info("score %s", case["name"])
    scored = score(
      inputs=[pdb],
      sequences_to_score=sequences,
      random_seed=SCORE_SEED,
      max_length=MAX_LENGTH,
    )
    for key in ("scores", "logits"):
      if key in scored and scored[key] is not None:
        arrays[f"{case['name']}__score__{key}"] = np.asarray(scored[key])
    meta["cases"].append({**case, "scored_sequences": sequences})
  args.out.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.out / "optout_golden.npz", **arrays)
  meta["arrays"] = {key: list(value.shape) for key, value in arrays.items()}
  (args.out / "optout_golden.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
  logger.info("wrote %d arrays at base %s", len(arrays), meta["base_commit"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
