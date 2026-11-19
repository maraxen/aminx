"""Profile one ``score_conditionals`` unit at tev_design's production shapes (#3485 / #3486).

Production: K = 32-64 masks, N = 2-7 sequences, L = 224 (214 protein + 10 peptide), padded to 512
and trimmed to its length rung by the runner. This scores K x N (mask, sequence) pairs against one
encoding in ONE ``runner.score`` call (``ar_masks``), twice with the same shapes and different
masks, and records wall time per call, backend compiles per call, and the planned dispatch.

A surrogate structure is used: the first ``--length`` residues of ``--structure`` (3pgk has 415,
so L=224 is a contiguous prefix). The sequence is the structure's own at those positions; the rest
of the chain is scored as context it does not see (masks only cover the first L positions).

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

logger = logging.getLogger("score_conditionals_unit_profile")
_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


def _git_hash() -> str:
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--structure", default="tests/data/3pgk.pdb")
  parser.add_argument("--checkpoint", default="proteinmpnn_v_48_020")
  parser.add_argument("--length", type=int, default=224)
  parser.add_argument("--k", type=int, default=64)
  parser.add_argument("--n", type=int, default=7)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  import jax  # noqa: PLC0415
  import numpy as np  # noqa: PLC0415

  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  compiles = [0]

  def _on_event(event: str, _d: float, **_k: object) -> None:
    if event == _BACKEND_COMPILE_EVENT:
      compiles[0] += 1

  jax.monitoring.register_event_duration_secs_listener(_on_event)

  # The structure's own residues for the first L positions, as MPNN letters.
  three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
           "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
           "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
  residues = [three.get(line[17:20], "X") for line in Path(args.structure).read_text().splitlines()
              if line.startswith("ATOM") and line[12:16].strip() == "CA"]
  length = args.length
  native = "".join(residues[:length])
  rng = np.random.default_rng(0)
  letters = "ACDEFGHIKLMNPQRSTVWY"
  seqs = [native] + ["".join(rng.choice(list(letters), size=length)) for _ in range(args.n - 1)]
  design = np.ones(length, np.int8)
  design[-10:] = 0

  def _masks(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    out = np.zeros((args.k, length, length), np.int8)
    idx = np.flatnonzero(design)
    for j in range(args.k):
      rank = np.full(length, -1)
      rank[idx] = r.permutation(idx.size)
      see = (rank[None, :] < rank[:, None]) & (design[:, None] == 1) & (design[None, :] == 1)
      see |= design[None, :] == 0
      see |= (design[:, None] == 0) & (design[None, :] == 1)
      np.fill_diagonal(see, False)
      out[j] = see
    return out

  calls = []
  for call, seed in enumerate((1, 2)):
    spec = ScoringSpecification(
      inputs=[args.structure], checkpoint_id=args.checkpoint, sequences_to_score=seqs,
      ar_masks=_masks(seed), designable_mask=design, return_logits=True, backbone_noise=0.0,
      max_length=args.max_length,
    )
    before = compiles[0]
    t0 = time.perf_counter()
    out = score(spec)
    jax.block_until_ready(out["logits"])
    wall = time.perf_counter() - t0
    shape = list(np.asarray(out["logits"]).shape)
    calls.append({"call": call + 1, "wall_s": wall, "compiles": compiles[0] - before, "logits_shape": shape})
    logger.info("call %d: %.2f s, %d compiles, logits %s", call + 1, wall, compiles[0] - before, shape)

  pairs = args.k * args.n
  payload = {
    "wall_s_call1": calls[0]["wall_s"],
    "wall_s_call2": calls[1]["wall_s"],
    "compiles_call1": calls[0]["compiles"],
    "compiles_call2": calls[1]["compiles"],
    "ms_per_pair_warm": 1e3 * calls[1]["wall_s"] / pairs,
    "logits_shape_ok": calls[1]["logits_shape"][:3] == [1, args.k, args.n],
    "k": args.k, "n": args.n, "length": length, "pairs": pairs,
    "backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
    "jax_version": jax.__version__, "git_hash": _git_hash(), "calls": calls,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  if results := os.environ.get("BTH_RESULTS_PATH"):
    Path(results).write_text(json.dumps({k: v for k, v in payload.items() if k != "calls"}))


if __name__ == "__main__":
  main()
