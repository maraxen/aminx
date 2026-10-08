"""The ``protonpotts_ph_driver`` wave: the driver's ``sample`` purpose end to end, against the upstream engine at T = 0.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §45-§48. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/protonpotts_ph_driver_parity.py \\
        --state-npz <v6_state.npz> --ph-dir <P4g out dir> \\
        --cell pkad_unlabelled=<1BVC.pdb>,A --cell multichain=<6m0j.pdb>,E --cell only_o_1olr=<1OLR.pdb>,A --cell gap=<1EL1.pdb>,A

Criteria are pre-registered in ``protonpotts_ph_driver_parity.bth.toml`` (committed before this script). For each cell
the REAL ``aminx.host.runner.sample`` runs on the structure file with the converted checkpoint (float32), at temperature
0 so no random numbers are consumed, once for the block-descent configuration and once for the centre-free greedy
configuration of the P4g dump. Centres, final sequence, final energy and selectivity gap must match the dump's T = 0 calls.
Five deliberate errors must each be rejected by the same comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402
from protonpotts_driver_parity import _load, _load_converter  # noqa: E402

from aminx.families.protonpotts_mpnn.vocab import token_index, upstream_to_aminx_index  # noqa: E402
from aminx.host.runner import sample  # noqa: E402
from aminx.run.options import ProtonPottsOptions  # noqa: E402
from aminx.run.specs import SamplingSpecification  # noqa: E402

logger = logging.getLogger("protonpotts_ph_driver_parity")

F32_FACTOR = 10.0
F32_MIN_REL = 1e-6
CONTROL_CELL = "pkad_unlabelled"
CELLS = ("pkad_unlabelled", "multichain", "only_o_1olr", "gap")


def _to_aminx(sequence: np.ndarray) -> np.ndarray:
  return np.vectorize(upstream_to_aminx_index)(sequence).astype(np.int32)


def block_options(chain: str) -> ProtonPottsOptions:
  """Production block descent at T = 0 (the dump's `block_T0`)."""
  return ProtonPottsOptions(binder_chain=chain, design_method="block_descent", temperature=0.0, samples_per_site=1)


def greedy_options(chain: str) -> ProtonPottsOptions:
  """The dump's centre-free greedy at T = 0 (`greedy_T0`; upstream's neighbour_k / max_mutations stay 0)."""
  return ProtonPottsOptions(
    binder_chain=chain, design_method="greedy_energy_block", infill_scope="chain", center_types=(),
    neighbour_k=0, max_mutations=0, cv_patience=1, cv_max=2, temperature=0.0, samples_per_site=1,
  )  # fmt: skip


def run_driver(model_path: Path, pdb: Path, options: ProtonPottsOptions) -> dict[str, np.ndarray]:
  spec = SamplingSpecification(
    inputs=str(pdb), model_family="protonpottsmpnn", model_local_path=model_path, protonpotts=options
  )
  return sample(spec)["structures"]["0"]["arrays"]


class Dump:
  """One cell's recorded T = 0 calls, with the f32-vs-f64 floors of their energies."""

  def __init__(self, ph_dir: Path, cell: str) -> None:
    root = ph_dir / "protonpotts_v6_ph"
    self.arrays = {p: _load(root / f"{cell}_{p}.npz") for p in ("f64", "f32")}
    self.meta = {p: json.loads((root / f"{cell}_{p}.json").read_text(encoding="utf-8")) for p in ("f64", "f32")}

  def call(self, method: str) -> dict:
    return next(c for c in self.meta["f64"]["calls"] if c["method"] == method and c["temperature"] == 0.0)

  def design(self, precision: str, index: int) -> dict:
    return next(d for d in self.meta[precision]["designs"] if d["call_index"] == index)


def compare(dump: Dump, method: str, got: dict[str, np.ndarray]) -> dict:
  call = dump.call(method)
  index = call["index"]
  want_seq = _to_aminx(dump.arrays["f64"][f"c{index}_S_out"])
  tokens_ok = bool(np.array_equal(got["sequence"][0], want_seq))
  want_pos = [p["position"] for p in call["pins"]] or [-1]
  want_types = [token_index(p["protonation_type"]) for p in call["pins"]] or [-1]
  centres_ok = list(got["center_positions"][0]) == want_pos and list(got["center_types"][0]) == want_types
  want, other = dump.design("f64", index), dump.design("f32", index)
  scale = max(1.0, abs(want["final_potts_energy"]))
  floor_h = abs(want["final_potts_energy"] - other["final_potts_energy"])
  band_h = max(F32_FACTOR * floor_h, F32_MIN_REL * scale)
  diff_h = abs(float(got["final_potts_energy"][0]) - want["final_potts_energy"])
  scale_s = max(1.0, abs(want["selective_energy"] or 0.0))
  floor_s = abs((want["selective_energy"] or 0.0) - (other["selective_energy"] or 0.0))
  band_s = max(F32_FACTOR * floor_s, F32_MIN_REL * scale_s)
  diff_s = abs(float(got["selective_energy"][0]) - (want["selective_energy"] or 0.0))
  scores_ok = diff_h <= band_h and diff_s <= band_s
  return {
    "centres_ok": centres_ok, "tokens_ok": tokens_ok, "scores_ok": scores_ok, "ok": centres_ok and tokens_ok and scores_ok,
    "energy_over_band": diff_h / band_h, "selective_over_band": diff_s / band_s,
  }  # fmt: skip


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--ph-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PDB,CHAIN")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  cells: dict[str, tuple[Path, str]] = {}
  for item in args.cell:
    name, _, rest = item.partition("=")
    pdb, _, chain = rest.rpartition(",")
    cells[name] = (Path(pdb), chain)

  state = _load(args.state_npz)
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  manifest = json.loads((args.ph_dir / "ph_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == manifest["checkpoint_sha256"]

  per_call: dict[str, dict] = {}
  counts = {"ok": 0, "centres": 0, "tokens": 0, "scores": 0}
  worst = 0.0
  mutants: dict[str, str] = {}
  with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "model.eqx"
    eqx.tree_serialise_leaves(model_path, _load_converter()(state))
    dumps = {name: Dump(args.ph_dir, name) for name in cells}
    for name, (pdb, chain) in cells.items():
      for method, options in (("block", block_options(chain)), ("greedy", greedy_options(chain))):
        verdict = compare(dumps[name], method, run_driver(model_path, pdb, options))
        per_call[f"{name}/{method}"] = verdict
        counts["ok"] += verdict["ok"]
        counts["centres"] += verdict["centres_ok"]
        counts["tokens"] += verdict["tokens_ok"]
        counts["scores"] += verdict["scores_ok"]
        worst = max(worst, verdict["energy_over_band"], verdict["selective_over_band"])
        logger.info("%s/%s %s", name, method, "ok" if verdict["ok"] else verdict)

    pdb, chain = cells[CONTROL_CELL]
    base = block_options(chain)
    swapped_dep = (("HIS-P", ("HIS-A",)), ("ASP-P", ("ASP-D",)), ("GLU-P", ("GLU-D",)))
    variants = {
      "combined_lambda_0p4": dataclasses.replace(base, combined_lambda=0.4),
      "block_size_2": dataclasses.replace(base, block_size=2),
      "forbidden_tokens_reduced_to_unk": dataclasses.replace(base, forbidden_tokens=("UNK",)),
      "repetitive_window_off": dataclasses.replace(base, repetitive_window_weight=0.0),
      "wrong_contrast_token": dataclasses.replace(base, dep_map=swapped_dep),
    }
    for control, variant in variants.items():
      rejected = False
      greedy_variant = dataclasses.replace(
        greedy_options(chain),
        **{k: getattr(variant, k) for k in ("block_size", "forbidden_tokens", "repetitive_window_weight")},
      )
      for method, options in (("block", variant), ("greedy", greedy_variant)):
        try:
          rejected |= not compare(dumps[CONTROL_CELL], method, run_driver(model_path, pdb, options))["ok"]
        except ValueError:
          rejected = True
      mutants[control] = "failed" if rejected else "passed"
  for control, verdict in mutants.items():
    logger.info("mutant %-34s %s", control, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  n_calls = len(per_call)
  clean = "pass" if counts["ok"] == n_calls == 2 * len(CELLS) and checkpoint_matches else "fail"
  results = {
    "clean": clean, "n_cells": len(cells), "n_calls": n_calls, "n_calls_ok": int(counts["ok"]),
    "n_centres_ok": int(counts["centres"]), "n_tokens_exact": int(counts["tokens"]), "n_scores_ok": int(counts["scores"]),
    "checkpoint_matches": checkpoint_matches, "worst_energy_over_band": worst, "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()), "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_ph_driver_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
