"""The ``protonpotts_energy`` wave: aminx's reciprocal merge and Potts energy against upstream.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §35.4, §42. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/protonpotts_energy_parity.py \\
        --state-npz <v6_state.npz> --encoder-dir <P4e out dir> --energy-dir <P4f out dir> \\
        --features-dir ~/projects/aminx-oracles-protonpotts

WHAT IS COMPARED, per cell (4) and precision (f32, f64):
  * the MERGED pair table (``etab_postmerge`` in the sealed P4e dump) against ``protonpotts_table``;
  * ``E_idx``, exact;
  * the energy of ten fixed sequences per cell (sealed P4f dump: native, eight uniform draws over all 30
    tokens, one cyclic sweep) against ``protonpotts_energies`` fed the same sequences mapped to aminx token
    order.

TOLERANCES, fixed before any result exists:
  table, f64     1e-9 absolute;
  table, f32     10x the MEASURED upstream f32-vs-f64 spread of the table, never below 1e-7;
  energy, f64    1e-9 relative to max(1, max |E| of the cell);
  energy, f32    10x the MEASURED upstream f32-vs-f64 spread of that cell's energies, never below
                 1e-6 relative to max(1, max |E|) (a sum of ~10^4 float32 terms cannot do better);
  E_idx          exactly equal.

WHAT MAKES A PASS MEAN SOMETHING (spec §6): seven perturbations must each be REJECTED by the same comparison:
the shipped Potts path's double merge, no merge, a merge that forgets the transpose, sequences left in
upstream token order, an energy off by a factor of two, and two nudges just outside each band.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.potts_mpnn.etab import merge_pair, potts_energy  # noqa: E402
from aminx.families.protonpotts_mpnn import convert  # noqa: E402
from aminx.families.protonpotts_mpnn.energy import protonpotts_raw_table  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_energy_parity")

P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"
CELLS = {"pkad_unlabelled": P4B, "multichain": P4B, "gap": P4C, "only_o_1olr": P4D}
TABLE_F64_TOL = 1e-9
ENERGY_F64_REL = 1e-9
F32_FACTOR = 10.0
TABLE_F32_MIN = 1e-7
ENERGY_F32_MIN_REL = 1e-6
CONTROL_CELL = "pkad_unlabelled"
VARIANTS = ("double_merge", "no_merge", "merge_without_transpose", "upstream_token_order", "energy_halved")


def _load_converter() -> Callable:
  path = Path(__file__).resolve().parent / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_state


def _reorder_table(table: np.ndarray) -> np.ndarray:
  take = convert.token_permutation()
  return table[:, :, take][:, :, :, take]


def run_aminx(model, features: dict[str, np.ndarray], seqs_upstream: np.ndarray, dtype, variant: str = "good"):  # noqa: ANN001, ANN201
  """(table, E_idx, energies) for one cell; ``variant`` is a deliberate error, ``good`` is the real path."""
  coords = jnp.asarray(features["X"][0, :, :4, :], dtype=dtype)
  length = coords.shape[0]
  valid = jnp.ones(length, dtype=bool)
  raw, e_idx = protonpotts_raw_table(
    model,
    coords,
    jnp.ones(length, dtype=dtype),
    jnp.asarray(features["R_idx"][0]),
    jnp.asarray(features["chain_labels"][0]),
    valid,
  )
  if variant == "no_merge":
    table = raw
  elif variant == "merge_without_transpose":
    table = merge_pair(raw, e_idx, valid, denom=2, exclude_self=False, transpose=False)
  else:
    table = merge_pair(raw, e_idx, valid, denom=2, exclude_self=False)
  if variant == "double_merge":
    table = merge_pair(table, e_idx, valid, denom=4, exclude_self=True)
  mapped = np.vectorize(upstream_to_aminx_index)(seqs_upstream)
  sequences = jnp.asarray(seqs_upstream if variant == "upstream_token_order" else mapped)
  energy = potts_energy(table, e_idx, valid, sequences)
  if variant == "energy_halved":
    energy = energy * 0.5
  return np.asarray(table), np.asarray(e_idx), np.asarray(energy)


def compare(got, up_table, up_idx, up_energy, floor):  # noqa: ANN001, ANN201
  """Max table diff, max energy diff, E_idx exact, and whether table and energy are both within their bands."""
  table, e_idx, energy = got
  want = _reorder_table(up_table).astype(np.float64)
  table_diff = float(np.abs(table.astype(np.float64) - want).max())
  energy_diff = float(np.abs(energy.astype(np.float64) - up_energy.astype(np.float64)).max())
  scale = max(1.0, float(np.abs(up_energy).max()))
  if floor is None:
    table_band, energy_band = TABLE_F64_TOL, ENERGY_F64_REL * scale
  else:
    table_band = max(F32_FACTOR * floor["table"], TABLE_F32_MIN)
    energy_band = max(F32_FACTOR * floor["energy"], ENERGY_F32_MIN_REL * scale)
  idx_ok = bool(np.array_equal(e_idx, up_idx))
  return table_diff, energy_diff, idx_ok, table_diff <= table_band and energy_diff <= energy_band and idx_ok, (table_band, energy_band)


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--energy-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  state = _load(args.state_npz)
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  energy_meta = json.loads((args.energy_dir / "energy_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == energy_meta["encoder_checkpoint_sha256"]
  model = _load_converter()(state)

  def up(cell: str, precision: str) -> tuple[dict, dict]:
    return (
      _load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_{precision}.npz"),
      _load(args.energy_dir / "protonpotts_v6_energy" / f"{cell}_{precision}.npz"),
    )

  def floors(cell: str) -> dict[str, float]:
    (t64, e64), (t32, e32) = up(cell, "f64"), up(cell, "f32")
    return {
      "table": float(np.abs(t64["etab_postmerge"] - t32["etab_postmerge"].astype(np.float64)).max()),
      "energy": float(np.abs(e64["energy"] - e32["energy"].astype(np.float64)).max()),
    }

  per_cell: dict[str, dict] = {}
  n64 = n32 = n_idx = 0
  worst_table = worst_energy_rel = worst_ratio = 0.0
  for cell, rel in CELLS.items():
    features = _load(args.features_dir / rel / f"{cell}.npz")
    floor = floors(cell)
    row: dict = {"floor": floor}
    for name, dtype, tol in (("f64", jnp.float64, None), ("f32", jnp.float32, floor)):
      table_up, energy_up = up(cell, name)
      got = run_aminx(model, features, energy_up["seqs"], dtype)
      t_diff, e_diff, idx_ok, ok, bands = compare(
        got, table_up["etab_postmerge"], table_up["E_idx"], energy_up["energy"], tol
      )
      row[name] = {"table_max": t_diff, "energy_max": e_diff, "e_idx_exact": idx_ok, "ok": ok, "bands": bands}
      if name == "f64":
        n64 += ok
        worst_table = max(worst_table, t_diff)
        worst_energy_rel = max(worst_energy_rel, e_diff / max(1.0, float(np.abs(energy_up["energy"]).max())))
      else:
        n32 += ok
        worst_ratio = max(worst_ratio, t_diff / bands[0], e_diff / bands[1])
      n_idx += idx_ok
    per_cell[cell] = row
    logger.info("%s f64_ok=%s f32_ok=%s", cell, row["f64"]["ok"], row["f32"]["ok"])

  clean = "pass" if (n64 == len(CELLS) and n32 == len(CELLS) and n_idx == 2 * len(CELLS) and checkpoint_matches) else "fail"

  mutants: dict[str, str] = {}
  features = _load(args.features_dir / CELLS[CONTROL_CELL] / f"{CONTROL_CELL}.npz")
  table64, energy64 = up(CONTROL_CELL, "f64")
  for variant in VARIANTS:
    got = run_aminx(model, features, energy64["seqs"], jnp.float64, variant)
    ok = compare(got, table64["etab_postmerge"], table64["E_idx"], energy64["energy"], None)[3]
    mutants[variant] = "passed" if ok else "failed"
  good = run_aminx(model, features, energy64["seqs"], jnp.float64)
  nudged = good[2].copy()
  nudged[0] += 2.0 * ENERGY_F64_REL * max(1.0, float(np.abs(energy64["energy"]).max()))
  mutants["energy_nudged_just_outside_f64"] = (
    "passed" if compare((good[0], good[1], nudged), table64["etab_postmerge"], table64["E_idx"], energy64["energy"], None)[3] else "failed"
  )
  table32, energy32 = up(CONTROL_CELL, "f32")
  floor32 = floors(CONTROL_CELL)
  good32 = run_aminx(model, features, energy32["seqs"], jnp.float32)
  _, _, _, _, bands32 = compare(good32, table32["etab_postmerge"], table32["E_idx"], energy32["energy"], floor32)
  nudged32 = good32[2].copy()
  nudged32[0] += bands32[1] * 4.0
  mutants["energy_nudged_4x_band_f32"] = (
    "passed" if compare((good32[0], good32[1], nudged32), table32["etab_postmerge"], table32["E_idx"], energy32["energy"], floor32)[3] else "failed"
  )
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  results = {
    "clean": clean,
    "n_cells": len(CELLS),
    "n_f64_ok": int(n64),
    "n_f32_ok": int(n32),
    "n_e_idx_exact": int(n_idx),
    "checkpoint_matches": checkpoint_matches,
    "worst_f64_table_abs": worst_table,
    "worst_f64_energy_rel": worst_energy_rel,
    "worst_f32_over_band": worst_ratio,
    "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"),
    "per_cell": per_cell,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_energy_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
