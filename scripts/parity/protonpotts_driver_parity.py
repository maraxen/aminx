"""The ``protonpotts_driver`` wave: ``ProtonPottsDriver`` end to end, from a PDB file, against upstream's energies.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §43. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_driver_parity.py \\
        --state-npz <v6_state.npz> --energy-dir <P4f out dir> --features-dir ~/projects/aminx-oracles-protonpotts \\
        --cell pkad_unlabelled=<1BVC.pdb> --cell multichain=<6m0j.pdb> --cell only_o_1olr=<1OLR.pdb> --cell gap=<1EL1.pdb>

WHAT IS COMPARED. ``protonpotts_energy`` graded the merge and the energy on sealed FEATURE dumps. This runs the
whole user path instead: ``aminx.host.runner.score`` on the structure file, through the driver's PDB reader,
variant handling, model call and result arrays. The ten sealed P4f sequences per cell (native, eight uniform
over all 30 tokens, one cyclic sweep) go in as token lists, so the protonation tokens are exercised by name.
The candidate energies are compared with upstream's sealed energies, and the reference row's tokens with the
sealed native ``S``, which also checks that the driver featurised the same residues upstream did.

TOLERANCE, fixed before any result exists: float32 (the driver's precision); each energy within 10x the MEASURED
upstream f32-vs-f64 spread of that cell's energies, never below 1e-6 relative to max(1, max |E|) (decision 11e).

WHAT MAKES A PASS MEAN SOMETHING (spec §6): three errors in the user path must each be REJECTED by the same
comparison: token names spelled in upstream's index order (a vocabulary mix-up), candidate rows reversed, and a
correct output nudged to 4x the band.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, upstream_to_aminx_index  # noqa: E402
from aminx.host.runner import score  # noqa: E402
from aminx.run.options import ProtonPottsOptions  # noqa: E402
from aminx.run.specs import ScoringSpecification  # noqa: E402

logger = logging.getLogger("protonpotts_driver_parity")

P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"
FEATURES = {"pkad_unlabelled": P4B, "multichain": P4B, "gap": P4C, "only_o_1olr": P4D}
F32_FACTOR = 10.0
F32_MIN_REL = 1e-6
CONTROL_CELL = "pkad_unlabelled"


def _load_converter() -> Callable:
  path = Path(__file__).resolve().parent / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_state


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def _names(rows_upstream: np.ndarray, *, upstream_order: bool = False) -> list[list[str]]:
  """Token-name lists for upstream-index rows. ``upstream_order`` is the deliberate vocabulary mix-up."""
  convert = (lambda i: int(i)) if upstream_order else upstream_to_aminx_index
  return [[PROTONPOTTS_V6.symbols[convert(int(i))] for i in row] for row in rows_upstream]


def run_driver(
  model_path: Path, pdb: Path, rows_upstream: np.ndarray, workdir: Path, *, upstream_order: bool = False
) -> tuple[np.ndarray, np.ndarray]:
  """(energy per row, token indices per row) from the real ``score`` path. Row 0 is the reference."""
  variants = {f"row{i}": names for i, names in enumerate(_names(rows_upstream[1:], upstream_order=upstream_order))}
  variants_path = workdir / "variants.json"
  variants_path.write_text(json.dumps(variants), encoding="utf-8")
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind="energy",
    protonpotts=ProtonPottsOptions(variants_json=str(variants_path)),
  )
  arrays = score(spec)["structures"]["0"]["arrays"]
  return np.asarray(arrays["energy"], dtype=np.float64), np.asarray(arrays["candidate_tokens"])


def _band(energy_up: np.ndarray, floor: float) -> float:
  return max(F32_FACTOR * floor, F32_MIN_REL * max(1.0, float(np.abs(energy_up).max())))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--energy-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  cells = {n: Path(p) for n, _, p in (c.partition("=") for c in args.cell)}

  state = _load(args.state_npz)
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  energy_meta = json.loads((args.energy_dir / "energy_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == energy_meta["encoder_checkpoint_sha256"]

  per_cell: dict[str, dict] = {}
  n_ok = n_tokens_ok = 0
  worst_ratio = 0.0
  mutants: dict[str, str] = {}
  with tempfile.TemporaryDirectory() as tmp:
    workdir = Path(tmp)
    model_path = workdir / "model.eqx"
    eqx.tree_serialise_leaves(model_path, _load_converter()(state))

    def up(cell: str, precision: str) -> dict[str, np.ndarray]:
      return _load(args.energy_dir / "protonpotts_v6_energy" / f"{cell}_{precision}.npz")

    def native_aminx(cell: str) -> np.ndarray:
      feats = _load(args.features_dir / FEATURES[cell] / f"{cell}.npz")
      return np.vectorize(upstream_to_aminx_index)(feats["S"][0])

    for cell, pdb in cells.items():
      u32, u64 = up(cell, "f32"), up(cell, "f64")
      floor = float(np.abs(u64["energy"] - u32["energy"].astype(np.float64)).max())
      energy, tokens = run_driver(model_path, pdb, u32["seqs"], workdir)
      diff = float(np.abs(energy - u32["energy"].astype(np.float64)).max())
      band = _band(u32["energy"], floor)
      tokens_ok = bool(np.array_equal(tokens[0], native_aminx(cell))) and tokens.shape == (
        u32["seqs"].shape[0],
        u32["seqs"].shape[1],
      )
      per_cell[cell] = {"max_diff": diff, "band": band, "floor": floor, "tokens_ok": tokens_ok}
      n_ok += diff <= band
      n_tokens_ok += tokens_ok
      worst_ratio = max(worst_ratio, diff / band)
      logger.info("%s diff=%.3g band=%.3g tokens_ok=%s", cell, diff, band, tokens_ok)

      if cell == CONTROL_CELL:
        want = u32["energy"].astype(np.float64)
        wrong_vocab, _ = run_driver(model_path, pdb, u32["seqs"], workdir, upstream_order=True)
        mutants["token_names_in_upstream_order"] = (
          "passed" if float(np.abs(wrong_vocab - want).max()) <= band else "failed"
        )
        mutants["candidate_rows_reversed"] = (
          "passed" if float(np.abs(energy[::-1] - want).max()) <= band else "failed"
        )
        nudged = energy.copy()
        nudged[1] += 4.0 * band
        mutants["energy_nudged_4x_band"] = (
          "passed" if float(np.abs(nudged - want).max()) <= band else "failed"
        )
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  clean = "pass" if (n_ok == len(cells) and n_tokens_ok == len(cells) and checkpoint_matches) else "fail"
  results = {
    "clean": clean,
    "n_cells": len(cells),
    "n_energy_ok": int(n_ok),
    "n_tokens_ok": int(n_tokens_ok),
    "checkpoint_matches": checkpoint_matches,
    "worst_over_band": worst_ratio,
    "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"),
    "per_cell": per_cell,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_driver_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
