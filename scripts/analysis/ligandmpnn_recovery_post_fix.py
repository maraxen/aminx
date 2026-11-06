"""Native-sequence recovery of LigandMPNN on current aminx (after the backbone-O fix, #2326).

tev_design's 260717 state campaign ran aminx 0.1.0a19, whose LigandMPNN features read CB as the
backbone O; its LigandMPNN arms recovered about 0.13 of native residues against about 0.41 for
ProteinMPNN (aminx run 6d38ebaf). This re-samples the same structures, seeds and temperatures with
the current build, LigandMPNN with no ligand or side-chain context (isolating the backbone feature
path) against ProteinMPNN as the instrument control. No fixed positions; recovery is over every
real residue (the runner's validity mask), native taken from the same parse the sampler uses.
Task 261106_campaign-eda-respec.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

AF2_ORDER = "ARNDCQEGHILKMFPSTWYVX"
MPNN_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AF2_TO_MPNN = np.array([MPNN_ORDER.index(c) for c in AF2_ORDER])
ARMS = {"LigandMPNN": "ligandmpnn_v_32_020_25", "ProteinMPNN": "proteinmpnn_v_48_020"}
STATES = {"1LVB": "1LVB.cif", "reac1_TS": "noWat-reac1-TS_0.6_-0.4_full.cif", "reac2_TS": "noWat-reac2-TS_0.2_0.4_full.cif"}
SEEDS = (1001, 1002, 1003, 1004)
TEMPERATURES = (0.1, 0.3)
N_SAMPLES = 4

log = logging.getLogger("ligandmpnn_recovery_post_fix")


def native_tokens(path: str) -> np.ndarray:
  from aminx.io.parsing.dispatch import parse_input  # noqa: PLC0415

  proteins = list(parse_input(path))
  if len(proteins) != 1:
    raise ValueError(f"{path}: expected one model, got {len(proteins)}")
  return AF2_TO_MPNN[np.asarray(proteins[0].aatype).astype(int)]


def main() -> int:  # noqa: PLR0915
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--structures-dir", type=Path, required=True)
  ap.add_argument("--out-dir", type=Path, required=True)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  args.out_dir.mkdir(parents=True, exist_ok=True)

  from aminx.host.runner import sample  # noqa: PLC0415
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  rng = np.random.default_rng(0)
  rows = []
  for state, fname in STATES.items():
    path = str(args.structures_dir / fname)
    native = native_tokens(path)
    L = native.shape[0]
    for arm, ckpt in ARMS.items():
      for seed in SEEDS:
        t0 = time.time()
        spec = SamplingSpecification(
          inputs=[path], checkpoint_id=ckpt, num_samples=N_SAMPLES, temperature=list(TEMPERATURES),
          random_seed=seed, backbone_noise=0.0, return_logits=True,
        )
        res = sample(spec)
        seqs = np.asarray(res["sequences"])[0, :, 0]  # (N, T, Lpad)
        mask = np.asarray(res["mask"])[0, :, 0].astype(bool)  # (N, T, Lpad)
        logits = np.asarray(res["logits"])[0, :, 0]  # (N, T, Lpad, 21)
        real = mask[0, 0]
        if int(real.sum()) != L:
          raise ValueError(f"{state}: runner mask covers {int(real.sum())} residues, parser gives {L}")
        idx = np.where(real)[0]
        s = seqs[..., idx]  # (N, T, L)
        lg = logits[..., idx, :]
        shuffled = native.copy()
        shuffled = shuffled[rng.permutation(L)]
        lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
        nll = lse - np.take_along_axis(lg, s[..., None].astype(np.int64), axis=-1)[..., 0]
        for ti, temp in enumerate(TEMPERATURES):
          toks = s[:, ti]
          comp = np.bincount(toks.ravel(), minlength=21)[:21] / toks.size
          rows.append({
            "arm": arm, "state": state, "seed": seed, "temperature": temp,
            "recovery": round(float((toks == native).mean()), 6),
            "recovery_shuffled_native": round(float((toks == shuffled).mean()), 6),
            "composition_entropy_bits": round(float(-(comp[comp > 0] * np.log2(comp[comp > 0])).sum()), 4),
            "nll_per_residue": round(float(nll[:, ti].mean()), 6),
            "top_token_fraction": round(float(comp.max()), 4),
          })
        log.info("%s %s seed %d: recovery@T=%.1f %.3f (%.0f s)", arm, state, seed, TEMPERATURES[0], rows[-2]["recovery"], time.time() - t0)

  with (args.out_dir / "recovery_by_seed.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  def cell_mean(arm: str, state: str, key: str = "recovery", temp: float = TEMPERATURES[0]) -> float:
    v = [r[key] for r in rows if r["arm"] == arm and r["state"] == state and r["temperature"] == temp]
    return float(np.mean(v))

  lig = {s: cell_mean("LigandMPNN", s) for s in STATES}
  pro = {s: cell_mean("ProteinMPNN", s) for s in STATES}
  results = {
    "ligandmpnn_recovery_min_state": min(lig.values()),
    "ligandmpnn_recovery_mean": float(np.mean(list(lig.values()))),
    "proteinmpnn_recovery_min_state": min(pro.values()),
    "proteinmpnn_recovery_mean": float(np.mean(list(pro.values()))),
    "ligandmpnn_composition_entropy_mean": float(np.mean([cell_mean("LigandMPNN", s, "composition_entropy_bits") for s in STATES])),
    "proteinmpnn_composition_entropy_mean": float(np.mean([cell_mean("ProteinMPNN", s, "composition_entropy_bits") for s in STATES])),
    "ligandmpnn_nll_mean": float(np.mean([cell_mean("LigandMPNN", s, "nll_per_residue") for s in STATES])),
    "proteinmpnn_nll_mean": float(np.mean([cell_mean("ProteinMPNN", s, "nll_per_residue") for s in STATES])),
    "shuffled_recovery_max": max(r["recovery_shuffled_native"] for r in rows),
    "aminx_version": md.version("aminx"),
    "proxide_version": md.version("proxide"),
    "n_rows": len(rows),
  }
  results["per_state_ligandmpnn"] = json.dumps({k: round(v, 4) for k, v in lig.items()})
  results["per_state_proteinmpnn"] = json.dumps({k: round(v, 4) for k, v in pro.items()})
  log.info("results: %s", json.dumps(results))
  (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
  if rp := os.environ.get("BTH_RESULTS_PATH"):
    Path(rp).write_text(json.dumps(results))
  return 0


if __name__ == "__main__":
  sys.exit(main())
