"""GPU memory and wall time of ``score_pairs`` by tile size: the evidence for ``_MAX_PAIR_TILE`` (#3485).

One unit (K=64 masks x N=7 sequences = 448 pairs, L=224 on 3pgk at the 512 rung) is scored with
the pair tile forced to each of 6, 12, 24 and 32, and once with the cap lifted so the device-sized
budget picks the tile ("uncapped"). Each tile runs in its OWN process, so peaks do not carry over:
a cold call, then a warm call with different masks; the process reports the warm wall time and the
device's peak bytes in use, or ``oom`` if the call ran out of memory.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("score_pairs_tile_sweep")
_TILES = (6, 12, 24, 32)


def _git_hash() -> str:
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _one_tile(tile: int | None) -> dict[str, object]:
  """Run inside the child process: score the unit twice at ``tile`` (None = uncapped)."""
  import jax  # noqa: PLC0415
  import numpy as np  # noqa: PLC0415

  import aminx.host.runner as runner  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  if tile is None:
    runner._MAX_PAIR_TILE = 10**9  # noqa: SLF001 -- the device-sized plan, uncapped
  else:
    runner._MAX_PAIR_TILE = tile  # noqa: SLF001
    os.environ["AMINX_MEMORY_BUDGET_BYTES"] = str(10**13)  # planner allows any tile; the cap decides
  length, k, n = 224, 64, 7
  three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
           "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
           "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
  res = [three.get(line[17:20], "X") for line in Path("tests/data/3pgk.pdb").read_text().splitlines()
         if line.startswith("ATOM") and line[12:16].strip() == "CA"]
  seqs = ["".join(res[:length])] * n

  def masks(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    out = np.zeros((k, length, length), np.int8)
    for j in range(k):
      rank = np.empty(length, int)
      rank[r.permutation(length)] = np.arange(length)
      out[j] = rank[None, :] < rank[:, None]
    return out

  def call(seed: int) -> float:
    spec = ScoringSpecification(inputs=["tests/data/3pgk.pdb"], checkpoint_id="proteinmpnn_v_48_020",
                                sequences_to_score=seqs, ar_masks=masks(seed), return_logits=True,
                                backbone_noise=0.0, max_length=512)
    t0 = time.perf_counter()
    jax.block_until_ready(score_out := runner.score(spec)["logits"])
    del score_out
    return time.perf_counter() - t0

  try:
    call(1)
    warm = call(2)
    status = "ok"
  except Exception as exc:  # noqa: BLE001 -- an OOM is a result here, not a crash
    if "RESOURCE_EXHAUSTED" not in str(exc):
      raise
    warm, status = float("nan"), "oom"
  stats = jax.devices()[0].memory_stats() or {}
  return {"tile": tile, "status": status, "warm_s": warm, "peak_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
          "device_kind": jax.devices()[0].device_kind}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path)
  parser.add_argument("--child-tile", default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  if args.child_tile is not None:
    tile = None if args.child_tile == "uncapped" else int(args.child_tile)
    print("RESULT " + json.dumps(_one_tile(tile)), flush=True)
    return

  rows = []
  for tile in (*_TILES, "uncapped"):
    proc = subprocess.run([sys.executable, __file__, "--child-tile", str(tile)], capture_output=True, text=True, check=False)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
      logger.error("tile %s produced no result (exit %s): %s", tile, proc.returncode, proc.stderr[-2000:])
      rows.append({"tile": tile, "status": "crash"})
      continue
    rows.append(json.loads(line.removeprefix("RESULT ")))
    logger.info("tile %s: %s", tile, rows[-1])

  ok = [r for r in rows if r.get("tile") != "uncapped" and r.get("tile") is not None and r["status"] == "ok"]
  capped = {r["tile"]: r for r in ok}
  unc = rows[-1]
  payload = {
    "all_capped_ok": len(ok) == len(_TILES),
    "peak_gib_tile32": float(capped[32]["peak_gib"]) if 32 in capped else float("nan"),
    "warm_s_tile32": float(capped[32]["warm_s"]) if 32 in capped else float("nan"),
    "warm_s_min_capped": min((float(r["warm_s"]) for r in ok), default=float("nan")),
    "warm_ratio_tile32_vs_best": (float(capped[32]["warm_s"]) / min(float(r["warm_s"]) for r in ok)) if 32 in capped else float("nan"),
    "uncapped_status": str(unc.get("status")),
    "uncapped_peak_gib": float(unc.get("peak_gib", float("nan"))),
    "device_kind": str(next((r.get("device_kind") for r in rows if r.get("device_kind")), "unknown")),
    "git_hash": _git_hash(),
    "rows": rows,
  }
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
  if results := os.environ.get("BTH_RESULTS_PATH"):
    Path(results).write_text(json.dumps({k: v for k, v in payload.items() if k != "rows"}))


if __name__ == "__main__":
  main()
