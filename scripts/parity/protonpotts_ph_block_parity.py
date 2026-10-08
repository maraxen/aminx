"""The ``protonpotts_ph_block`` wave: aminx block-descent pH design against the upstream engine, on replayed uniforms.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §45-§46. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ph_block_parity.py \\
        --encoder-dir <P4e out dir> --ph-dir <P4g out dir> --features-dir ~/projects/aminx-oracles-protonpotts

Criteria are pre-registered in ``protonpotts_ph_block_parity.bth.toml`` (committed before this script). Per
(cell, precision, block-descent call) the aminx side is given only the sealed table, ``E_idx``, the native sequence,
the binder mask, the production configuration and the dump's recorded uniforms, REDERIVES the placement plan, runs the
sweep, and must match upstream's plan, pooled z-scales, final sequence, draw count and scores. Eight deliberate errors
must each be rejected by the same comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.potts_mpnn.etab import potts_energy  # noqa: E402
from aminx.families.protonpotts_mpnn import convert  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_descent import block_descent, rep_class_mask  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_plan import (  # noqa: E402
  block_table,
  plan_from_center_types,
  valid_token_mask,
)
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies, candidate_energies_at  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_ph_block_parity")

CELLS = ("pkad_unlabelled", "multichain", "only_o_1olr", "gap")
CONTROL_CELL = "pkad_unlabelled"
F64_REL = 1e-9
F32_FACTOR = 10.0
F32_MIN_REL = 1e-6
SENTINEL = 10**6
# Upstream accumulates its sampling CDF in ITS token order, so the same uniform selects a different entry than a
# CDF in aminx order would. Entry j is the aminx index of upstream token j. A parity detail, not a modelling choice:
# any order is a valid sampler (found by the first smoke, which matched every T=0 call but not the sampled ones).
UPSTREAM_CDF_ORDER = np.asarray([upstream_to_aminx_index(j) for j in range(30)], dtype=np.int32)


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def _to_aminx(sequence: np.ndarray) -> np.ndarray:
  return np.vectorize(upstream_to_aminx_index)(sequence).astype(np.int32)


def _reorder_table(table: np.ndarray) -> np.ndarray:
  take = convert.token_permutation()
  return table[:, :, take][:, :, :, take]


class Cell:
  """One cell at one precision: the sealed table, the dump's context, and the recorded calls."""

  def __init__(self, encoder_dir: Path, ph_dir: Path, cell: str, precision: str) -> None:
    tables = _load(encoder_dir / "protonpotts_v6_encoder" / f"{cell}_{precision}.npz")
    dtype = np.float64 if precision == "f64" else np.float32
    self.table = jnp.asarray(_reorder_table(tables["etab_postmerge"]).astype(dtype))
    self.e_idx = jnp.asarray(tables["E_idx"], dtype=jnp.int32)
    self.arrays = _load(ph_dir / "protonpotts_v6_ph" / f"{cell}_{precision}.npz")
    self.meta = json.loads((ph_dir / "protonpotts_v6_ph" / f"{cell}_{precision}.json").read_text(encoding="utf-8"))
    self.native = _to_aminx(self.arrays["ctx_S_native"])
    self.binder = np.asarray(self.arrays["ctx_free_mask"], dtype=bool) & np.asarray(self.arrays["ctx_chainA"], dtype=bool)
    self.res_id = np.asarray(self.arrays["ctx_res_id"], dtype=np.int64)
    self.length = int(self.native.shape[0])
    self.dtype = dtype
    self.designs = {d["call_index"]: d for d in self.meta["designs"]}

  def block_calls(self) -> list[dict]:
    return [c for c in self.meta["calls"] if c["method"] == "block"]


def derive_plan(cell: Cell, config: PHDesignConfig):  # noqa: ANN201
  """The placement plan, rederived from the table and the native sequence alone."""
  e_idx = np.asarray(cell.e_idx)
  field = np.asarray(candidate_energies(cell.table, cell.e_idx, jnp.asarray(cell.native)))
  return plan_from_center_types(
    field, e_idx, cell.binder, cell.res_id, config.center_types, config.dep_map_dict(),
    infill_scope=config.infill_scope, neighbour_k=config.neighbour_k, max_mutations=config.max_mutations,
  )  # fmt: skip


def run_call(cell: Cell, call: dict, config: PHDesignConfig, *, shift_uniforms: bool = False, valid=None):  # noqa: ANN001, ANN201
  """Rederive the plan, replay the call's uniforms, and run ``block_descent``. Returns (plan, blocks, result)."""
  plan = derive_plan(cell, config)
  e_idx = np.asarray(cell.e_idx)
  blocks, block_valid = block_table(e_idx, plan.designable, config.block_size)
  pins = plan.pins
  depth = max(len(p.dep_idxs) for p in pins)
  pin_dep = np.zeros((len(pins), depth), dtype=np.int32)
  pin_dep_valid = np.zeros((len(pins), depth), dtype=bool)
  for i, pin in enumerate(pins):
    pin_dep[i, : len(pin.dep_idxs)] = pin.dep_idxs
    pin_dep_valid[i, : len(pin.dep_idxs)] = True
  recorded = cell.arrays[f"c{call['index']}_uniforms"]
  need = max(1, config.block_max_rounds * len(plan.designable))
  uniforms = np.zeros(max(need, recorded.shape[0]), dtype=cell.dtype)
  uniforms[: recorded.shape[0]] = recorded
  if shift_uniforms and recorded.shape[0] > 1:
    uniforms[: recorded.shape[0]] = np.roll(recorded, -1)
  residue_number = np.where(cell.binder, cell.res_id, -SENTINEL - 100 * np.arange(cell.length))
  result = block_descent(
    cell.table, cell.e_idx, jnp.asarray(_to_aminx(cell.arrays[f"c{call['index']}_S_in"])),
    jnp.asarray(plan.designable, dtype=jnp.int32), jnp.asarray(blocks), jnp.asarray(block_valid),
    jnp.asarray([p.position for p in pins], dtype=jnp.int32), jnp.asarray([p.prot_idx for p in pins], dtype=jnp.int32),
    jnp.asarray(pin_dep), jnp.asarray(pin_dep_valid),
    jnp.asarray(valid_token_mask(config.forbidden_tokens) if valid is None else valid),
    jnp.asarray(rep_class_mask(config.repetitive_window_parents)), jnp.asarray(residue_number, dtype=jnp.int32),
    config=config, uniforms=jnp.asarray(uniforms), cdf_order=jnp.asarray(UPSTREAM_CDF_ORDER),
  )  # fmt: skip
  return plan, blocks, result


def scores(cell: Cell, plan, seq: np.ndarray) -> tuple[float, float]:  # noqa: ANN001
  """(final Potts energy, summed selectivity gap) of a sequence, computed independently of the optimiser."""
  valid = jnp.ones(cell.length, dtype=bool)
  energy = float(potts_energy(cell.table, cell.e_idx, valid, jnp.asarray(seq, dtype=jnp.int32)))
  rows = np.asarray(
    candidate_energies_at(
      cell.table, cell.e_idx, jnp.asarray(seq, dtype=jnp.int32),
      jnp.asarray([p.position for p in plan.pins], dtype=jnp.int32),
    )
  )  # fmt: skip
  gap = sum(float(rows[i, p.prot_idx] - np.mean([rows[i, d] for d in p.dep_idxs])) for i, p in enumerate(plan.pins))
  return energy, gap


def _close(got: float, want: float, band: float) -> bool:
  return abs(got - want) <= band


def compare_call(cell: Cell, other: Cell, call: dict, got, *, f32: bool) -> dict:  # noqa: ANN001
  """Compare one replayed call with the dump. ``other`` is the same cell at the other precision (for the f32 floors)."""
  plan, blocks, result = got
  index = call["index"]
  want_pins = [(p["position"], p["protonation_type"]) for p in call["pins"]]
  got_pins = [(p.position, p.protonation_type) for p in plan.pins]
  want_blocks = {int(k): v for k, v in call["blocks"].items()}
  got_blocks = {int(b[0]): [int(x) for x in b[v]] for b, v in zip(blocks, blocks >= 0, strict=True)}
  plan_ok = got_pins == want_pins and list(plan.designable) == call["neigh"] and got_blocks == want_blocks

  tokens_ok = bool(np.array_equal(np.asarray(result.seq), _to_aminx(cell.arrays[f"c{index}_S_out"])))
  draws_ok = int(result.n_draws) == int(cell.arrays[f"c{index}_uniforms"].shape[0])

  want_z = np.asarray(call["zscales"], dtype=np.float64)
  floor_z = np.abs(want_z - np.asarray(other.meta["calls"][index]["zscales"], dtype=np.float64))
  got_z = np.asarray(result.zscales, dtype=np.float64)
  rel = F32_MIN_REL if f32 else F64_REL
  z_band = np.maximum(F32_FACTOR * floor_z, rel * np.abs(want_z)) if f32 else rel * np.maximum(1.0, np.abs(want_z))
  z_diff = np.abs(got_z - want_z)
  z_ok = bool(np.all(z_diff <= z_band))

  design = cell.designs[index]
  floor_h = abs(design["final_potts_energy"] - other.designs[index]["final_potts_energy"])
  floor_s = abs((design["selective_energy"] or 0.0) - (other.designs[index]["selective_energy"] or 0.0))
  energy, gap = scores(cell, plan, np.asarray(result.seq))
  scale_h = max(1.0, abs(design["final_potts_energy"]))
  scale_s = max(1.0, abs(design["selective_energy"] or 0.0))
  band_h = max(F32_FACTOR * floor_h, F32_MIN_REL * scale_h) if f32 else F64_REL * scale_h
  band_s = max(F32_FACTOR * floor_s, F32_MIN_REL * scale_s) if f32 else F64_REL * scale_s
  scores_ok = _close(energy, design["final_potts_energy"], band_h) and _close(gap, design["selective_energy"], band_s)
  return {
    "plan_ok": plan_ok, "tokens_ok": tokens_ok, "draws_ok": draws_ok, "zscales_ok": z_ok, "scores_ok": scores_ok,
    "ok": plan_ok and tokens_ok and draws_ok and z_ok and scores_ok,
    "zscale_rel_diff": float(np.max(z_diff / np.maximum(1.0, np.abs(want_z)))),
    "energy_rel_diff": abs(energy - design["final_potts_energy"]) / scale_h,
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
  base = PHDesignConfig()

  per_call: dict[str, dict] = {}
  counts = {"ok": 0, "plan": 0, "tokens": 0, "draws": 0, "zscales": 0, "scores": 0}
  worst_z = worst_e = 0.0
  for cell_name in CELLS:
    cells = {p: Cell(args.encoder_dir, args.ph_dir, cell_name, p) for p in ("f64", "f32")}
    for precision, cell in cells.items():
      other = cells["f32" if precision == "f64" else "f64"]
      for call in cell.block_calls():
        config = dataclasses.replace(base, temperature=float(call["temperature"]))
        verdict = compare_call(cell, other, call, run_call(cell, call, config), f32=precision == "f32")
        per_call[f"{cell_name}/{precision}/{call['label']}/{call['index']}"] = verdict
        counts["ok"] += verdict["ok"]
        counts["plan"] += verdict["plan_ok"]
        counts["tokens"] += verdict["tokens_ok"]
        counts["draws"] += verdict["draws_ok"]
        counts["zscales"] += verdict["zscales_ok"]
        counts["scores"] += verdict["scores_ok"]
        if precision == "f64":
          worst_z = max(worst_z, verdict["zscale_rel_diff"])
          worst_e = max(worst_e, verdict["energy_rel_diff"])
        logger.info("%s/%s/%s %s", cell_name, precision, call["label"], "ok" if verdict["ok"] else verdict)
  n_calls = len(per_call)

  # Deliberate errors on the control cell (float64): each must make the same comparison fail on at least one call.
  cell = Cell(args.encoder_dir, args.ph_dir, CONTROL_CELL, "f64")
  other = Cell(args.encoder_dir, args.ph_dir, CONTROL_CELL, "f32")
  all_but_x = np.ones(30, dtype=bool)
  all_but_x[20] = False
  swapped_dep = tuple((k, ("HIS-A",) if k == "HIS-P" else v) for k, v in base.dep_map)
  control_runs = {
    "uniforms_shifted_by_one": (base, {"shift_uniforms": True}),
    "combined_lambda_0p4": (dataclasses.replace(base, combined_lambda=0.4), {}),
    "block_size_2": (dataclasses.replace(base, block_size=2), {}),
    "forbidden_tokens_not_applied": (base, {"valid": all_but_x}),
    "repetitive_window_off": (dataclasses.replace(base, repetitive_window_weight=0.0), {}),
    "wrong_contrast_token": (dataclasses.replace(base, dep_map=swapped_dep), {}),
  }  # fmt: skip
  mutants: dict[str, str] = {}
  for name, (config_variant, kwargs) in control_runs.items():
    rejected = False
    for call in cell.block_calls():
      config = dataclasses.replace(config_variant, temperature=float(call["temperature"]))
      try:
        verdict = compare_call(cell, other, call, run_call(cell, call, config, **kwargs), f32=False)
      except (ValueError, IndexError):
        rejected = True
        break
      rejected |= not verdict["ok"]
    mutants[name] = "failed" if rejected else "passed"
  call = cell.block_calls()[0]
  good = run_call(cell, call, dataclasses.replace(base, temperature=float(call["temperature"])))
  nudged = good[2]._replace(zscales=good[2].zscales * (1.0 + 2e-9 * 2.0))
  mutants["zscale_nudged_outside_f64"] = (
    "passed" if compare_call(cell, other, call, (good[0], good[1], nudged), f32=False)["zscales_ok"] else "failed"
  )
  design = cell.designs[call["index"]]
  energy, _gap = scores(cell, good[0], np.asarray(good[2].seq))
  nudged_ok = _close(energy + 2.0 * F64_REL * max(1.0, abs(design["final_potts_energy"])), design["final_potts_energy"],
                     F64_REL * max(1.0, abs(design["final_potts_energy"])))  # fmt: skip
  mutants["energy_nudged_outside_f64"] = "passed" if nudged_ok else "failed"
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  clean = (
    "pass"
    if counts["ok"] == n_calls == 24 and dump_flags_pass
    else "fail"
  )
  results = {
    "clean": clean, "n_cells": len(CELLS), "n_calls": n_calls, "n_calls_ok": int(counts["ok"]),
    "n_plans_ok": int(counts["plan"]), "n_zscales_ok": int(counts["zscales"]), "n_tokens_exact": int(counts["tokens"]),
    "n_draws_ok": int(counts["draws"]), "n_scores_ok": int(counts["scores"]), "dump_flags_pass": bool(dump_flags_pass),
    "worst_zscale_rel_f64": worst_z, "worst_energy_rel_f64": worst_e, "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()), "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_ph_block_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
