"""The ``protonpotts_ddg`` wave: ``ProtonPottsDriver`` ``score:ddg`` end to end, from a PDB file, against upstream.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §43, §52; debt #2618 item 1. Sidecar:
``protonpotts_ddg_parity.bth.toml`` (committed before this script). Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ddg_parity.py \\
        --state-npz <v6_state.npz> --energy-dir <P4f out dir> --features-dir ~/projects/aminx-oracles-protonpotts \\
        --cell pkad_unlabelled=<1BVC.pdb> --cell multichain=<6m0j.pdb> --cell only_o_1olr=<1OLR.pdb> --cell gap=<1EL1.pdb>

WHAT IS COMPARED. ``protonpotts_driver`` graded ``score:energy``. Here the nine sealed P4f variant sequences per cell go in
as token lists with ``output_kind="ddg"``; the driver's ``ddg`` must equal upstream's E(variant) - E(reference) (float32,
tolerance 10x the MEASURED upstream f32-vs-f64 spread of the cell's ddg values, never below 1e-6 relative to
max(1, max|ddg|)), and ``mutant_tokens`` must equal the sealed variants in aminx order.

WHAT MAKES A PASS MEAN SOMETHING: five errors in the path must each be REJECTED by the same comparison: ddg sign flipped,
token names in upstream's index order, mutant rows reversed, variant ENERGIES compared as if they were ddg (reference not
subtracted), and a correct output nudged to 4x the band.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402

from aminx.host.runner import score  # noqa: E402
from aminx.run.options import ProtonPottsOptions  # noqa: E402
from aminx.run.specs import ScoringSpecification  # noqa: E402

logger = logging.getLogger("protonpotts_ddg_parity")


def _load_driver_wave():  # noqa: ANN202
  """The energy wave's helpers (converter, loaders, name lists, band rule), loaded from its file."""
  path = Path(__file__).resolve().parents[1] / "parity" / "protonpotts_driver_parity.py"
  spec = importlib.util.spec_from_file_location("protonpotts_driver_parity", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


W = _load_driver_wave()


def run_ddg(
  model_path: Path, pdb: Path, rows_upstream: np.ndarray, workdir: Path, *, upstream_order: bool = False
) -> tuple[np.ndarray, np.ndarray]:
  """(ddg per variant, token indices per variant) from the real ``score`` path, ``output_kind="ddg"``."""
  variants = {f"row{i}": names for i, names in enumerate(W._names(rows_upstream[1:], upstream_order=upstream_order))}
  variants_path = workdir / "variants.json"
  variants_path.write_text(json.dumps(variants), encoding="utf-8")
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind="ddg",
    protonpotts=ProtonPottsOptions(variants_json=str(variants_path)),
  )
  arrays = score(spec)["structures"]["0"]["arrays"]
  return np.asarray(arrays["ddg"], dtype=np.float64), np.asarray(arrays["mutant_tokens"])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--energy-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  cells = {n: Path(p) for n, _, p in (c.partition("=") for c in args.cell)}

  state = W._load(args.state_npz)
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
    eqx.tree_serialise_leaves(model_path, W._load_converter()(state))

    def up(cell: str, precision: str) -> dict[str, np.ndarray]:
      return W._load(args.energy_dir / "protonpotts_v6_energy" / f"{cell}_{precision}.npz")

    for cell, pdb in cells.items():
      u32, u64 = up(cell, "f32"), up(cell, "f64")
      ref32 = u32["energy"].astype(np.float64)
      want = ref32[1:] - ref32[0]
      want64 = u64["energy"][1:] - u64["energy"][0]
      floor = float(np.abs(want64 - want).max())
      ddg, tokens = run_ddg(model_path, pdb, u32["seqs"], workdir)
      diff = float(np.abs(ddg - want).max())
      band = max(W.F32_FACTOR * floor, W.F32_MIN_REL * max(1.0, float(np.abs(want).max())))
      expected_tokens = np.vectorize(W.upstream_to_aminx_index)(u32["seqs"][1:])
      tokens_ok = bool(np.array_equal(tokens, expected_tokens)) and tokens.shape[0] == want.shape[0] == 9
      per_cell[cell] = {"max_diff": diff, "band": band, "floor": floor, "tokens_ok": tokens_ok}
      n_ok += diff <= band
      n_tokens_ok += tokens_ok
      worst_ratio = max(worst_ratio, diff / band)
      logger.info("%s diff=%.3g band=%.3g tokens_ok=%s", cell, diff, band, tokens_ok)

      if cell == W.CONTROL_CELL:

        def verdict(candidate: np.ndarray, *, _want: np.ndarray = want, _band: float = band) -> str:
          return "passed" if float(np.abs(candidate - _want).max()) <= _band else "failed"

        mutants["ddg_sign_flipped"] = verdict(-ddg)
        wrong_vocab, _ = run_ddg(model_path, pdb, u32["seqs"], workdir, upstream_order=True)
        mutants["token_names_in_upstream_order"] = verdict(wrong_vocab)
        mutants["mutant_rows_reversed"] = verdict(ddg[::-1])
        mutants["energies_without_reference_subtracted"] = verdict(ref32[1:])
        nudged = ddg.copy()
        nudged[1] += 4.0 * band
        mutants["ddg_nudged_4x_band"] = verdict(nudged)
  for name, v in mutants.items():
    logger.info("mutant %-40s %s", name, "DETECTED" if v == "failed" else "NOT DETECTED")

  clean = "pass" if (n_ok == len(cells) and n_tokens_ok == len(cells) and checkpoint_matches) else "fail"
  results = {
    "clean": clean,
    "n_cells": len(cells),
    "n_ddg_ok": int(n_ok),
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
    msg = "protonpotts_ddg_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
