"""The ``protonpotts_ph_greedy`` wave: aminx centre-free greedy pH design against the upstream engine, on replayed uniforms.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §45-§47. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ph_greedy_parity.py \\
        --encoder-dir <P4e out dir> --ph-dir <P4g out dir> --features-dir ~/projects/aminx-oracles-protonpotts

Criteria are pre-registered in ``protonpotts_ph_greedy_parity.bth.toml`` (committed before this script). Shares its loader
and conventions with the graded block-descent wave (``protonpotts_ph_block_parity.py``): the aminx side is given the sealed
table and ``E_idx``, the native sequence, the binder mask, the configuration and the dump's uniforms (replayed in
upstream's CDF token order), REDERIVES the centre-free plan, and must match upstream's designable set, final sequence,
draw count and final energy. Six deliberate errors must each be rejected by the same comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from protonpotts_ph_block_parity import (  # noqa: E402
  CELLS,
  CONTROL_CELL,
  F32_FACTOR,
  F32_MIN_REL,
  F64_REL,
  SENTINEL,
  UPSTREAM_CDF_ORDER,
  Cell,
  _to_aminx,
)

from aminx.families.potts_mpnn.etab import potts_energy  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_descent import rep_class_mask  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_greedy import greedy_energy_block  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_plan import plan_centre_free, valid_token_mask  # noqa: E402

logger = logging.getLogger("protonpotts_ph_greedy_parity")


def greedy_config() -> PHDesignConfig:
  """The dump's centre-free configuration (upstream's neighbour_k / max_mutations stay at their 0 defaults)."""
  return PHDesignConfig(
    method="greedy_energy_block", center_count=0, center_types=(), infill_scope="chain", cv_patience=1, cv_max=2,
    neighbour_k=0, max_mutations=0,
  )  # fmt: skip


def run_call(cell: Cell, call: dict, config: PHDesignConfig, *, shift_uniforms: bool = False, valid=None, cdf_order="upstream"):  # noqa: ANN001, ANN201
  """Rederive the centre-free plan, replay the call's uniforms, and run ``greedy_energy_block``. Returns (plan, result)."""
  plan = plan_centre_free(cell.binder)
  designable = np.asarray(plan.designable if plan is not None else (), dtype=np.int64)
  recorded = cell.arrays[f"c{call['index']}_uniforms"]
  need = max(1, config.cv_max * len(designable))
  uniforms = np.zeros(max(need, recorded.shape[0]), dtype=cell.dtype)
  uniforms[: recorded.shape[0]] = recorded
  if shift_uniforms and recorded.shape[0] > 1:
    uniforms[: recorded.shape[0]] = np.roll(recorded, -1)
  residue_number = np.where(cell.binder, cell.res_id, -SENTINEL - 100 * np.arange(cell.length))
  result = greedy_energy_block(
    cell.table, cell.e_idx, jnp.asarray(_to_aminx(cell.arrays[f"c{call['index']}_S_in"])), jnp.asarray(designable, dtype=jnp.int32),
    jnp.asarray(valid_token_mask(config.forbidden_tokens) if valid is None else valid),
    jnp.asarray(rep_class_mask(config.repetitive_window_parents)), jnp.asarray(residue_number, dtype=jnp.int32),
    config=config, uniforms=jnp.asarray(uniforms),
    cdf_order=jnp.asarray(UPSTREAM_CDF_ORDER) if cdf_order == "upstream" else None,
  )  # fmt: skip
  return plan, result


def compare_call(cell: Cell, other: Cell, call: dict, got, *, f32: bool) -> dict:  # noqa: ANN001
  plan, result = got
  index = call["index"]
  plan_ok = plan is not None and list(plan.designable) == call["neigh"] and not plan.pins and not call["pins"]
  tokens_ok = bool(np.array_equal(np.asarray(result.seq), _to_aminx(cell.arrays[f"c{index}_S_out"])))
  draws_ok = int(result.n_draws) == int(cell.arrays[f"c{index}_uniforms"].shape[0])
  design = cell.designs[index]
  floor_h = abs(design["final_potts_energy"] - other.designs[index]["final_potts_energy"])
  scale_h = max(1.0, abs(design["final_potts_energy"]))
  band = max(F32_FACTOR * floor_h, F32_MIN_REL * scale_h) if f32 else F64_REL * scale_h
  energy = float(potts_energy(cell.table, cell.e_idx, jnp.ones(cell.length, dtype=bool), jnp.asarray(np.asarray(result.seq), dtype=jnp.int32)))
  scores_ok = abs(energy - design["final_potts_energy"]) <= band
  return {
    "plan_ok": plan_ok, "tokens_ok": tokens_ok, "draws_ok": draws_ok, "scores_ok": scores_ok,
    "ok": plan_ok and tokens_ok and draws_ok and scores_ok,
    "energy_rel_diff": abs(energy - design["final_potts_energy"]) / scale_h, "steps": int(result.steps),
  }  # fmt: skip


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--ph-dir", type=Path, required=True)
  parser.add_argument(
    "--features-dir", type=Path, required=True,
    help="accepted for the registered invocation; the dump already verified its native S against the sealed features",
  )  # fmt: skip
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  manifest = json.loads((args.ph_dir / "ph_manifest.json").read_text(encoding="utf-8"))
  flags = manifest["flags"]
  dump_flags_pass = flags["n_cells"] == len(CELLS) and all(v for k, v in flags.items() if k != "n_cells")
  base = greedy_config()

  per_call: dict[str, dict] = {}
  counts = {"ok": 0, "plan": 0, "tokens": 0, "draws": 0, "scores": 0}
  worst_e = 0.0
  for cell_name in CELLS:
    cells = {p: Cell(args.encoder_dir, args.ph_dir, cell_name, p) for p in ("f64", "f32")}
    for precision, cell in cells.items():
      other = cells["f32" if precision == "f64" else "f64"]
      for call in [c for c in cell.meta["calls"] if c["method"] == "greedy"]:
        config = dataclasses.replace(base, temperature=float(call["temperature"]))
        verdict = compare_call(cell, other, call, run_call(cell, call, config), f32=precision == "f32")
        per_call[f"{cell_name}/{precision}/{call['label']}/{call['index']}"] = verdict
        counts["ok"] += verdict["ok"]
        counts["plan"] += verdict["plan_ok"]
        counts["tokens"] += verdict["tokens_ok"]
        counts["draws"] += verdict["draws_ok"]
        counts["scores"] += verdict["scores_ok"]
        if precision == "f64":
          worst_e = max(worst_e, verdict["energy_rel_diff"])
        logger.info("%s/%s/%s %s", cell_name, precision, call["label"], "ok" if verdict["ok"] else verdict)
  n_calls = len(per_call)

  cell = Cell(args.encoder_dir, args.ph_dir, CONTROL_CELL, "f64")
  other = Cell(args.encoder_dir, args.ph_dir, CONTROL_CELL, "f32")
  all_but_x = np.ones(30, dtype=bool)
  all_but_x[20] = False
  control_runs = {
    "uniforms_shifted_by_one": (base, {"shift_uniforms": True}),
    "block_size_2": (dataclasses.replace(base, block_size=2), {}),
    "forbidden_tokens_not_applied": (base, {"valid": all_but_x}),
    "repetitive_window_off": (dataclasses.replace(base, repetitive_window_weight=0.0), {}),
    "cdf_in_aminx_token_order": (base, {"cdf_order": None}),
  }  # fmt: skip
  calls = [c for c in cell.meta["calls"] if c["method"] == "greedy"]
  mutants: dict[str, str] = {}
  for name, (config_variant, kwargs) in control_runs.items():
    rejected = False
    for call in calls:
      config = dataclasses.replace(config_variant, temperature=float(call["temperature"]))
      try:
        verdict = compare_call(cell, other, call, run_call(cell, call, config, **kwargs), f32=False)
      except (ValueError, IndexError):
        rejected = True
        break
      rejected |= not verdict["ok"]
    mutants[name] = "failed" if rejected else "passed"
  call = calls[0]
  good = run_call(cell, call, dataclasses.replace(base, temperature=float(call["temperature"])))
  design = cell.designs[call["index"]]
  energy = float(potts_energy(cell.table, cell.e_idx, jnp.ones(cell.length, dtype=bool), jnp.asarray(np.asarray(good[1].seq), dtype=jnp.int32)))
  band = F64_REL * max(1.0, abs(design["final_potts_energy"]))
  mutants["energy_nudged_outside_f64"] = "passed" if abs(energy + 2.0 * band - design["final_potts_energy"]) <= band else "failed"
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  clean = "pass" if counts["ok"] == n_calls == 24 and dump_flags_pass else "fail"
  results = {
    "clean": clean, "n_cells": len(CELLS), "n_calls": n_calls, "n_calls_ok": int(counts["ok"]),
    "n_plans_ok": int(counts["plan"]), "n_tokens_exact": int(counts["tokens"]), "n_draws_ok": int(counts["draws"]),
    "n_scores_ok": int(counts["scores"]), "dump_flags_pass": bool(dump_flags_pass), "worst_energy_rel_f64": worst_e,
    "n_listed": len(mutants), "n_failed": sum(v == "failed" for v in mutants.values()), "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_ph_greedy_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
