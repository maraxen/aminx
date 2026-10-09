"""Dump placement-planning cases from aminx's own ph_plan, as the reference for ``browser/protonpotts-scorer/ph_plan.mjs``.

task_id 261009_protonpotts-onnx, backlog #5816 (B1). A deterministic equality test, not a measured finding: random kNN
graphs, fields, binder masks and residue numbers go through the real Python functions and their outputs are written next to
the inputs. Output: ``plan_cases.json`` = ``{"cases": [{"name", "kind", "inputs", "expected"}]}``.

Run with the project environment, e.g.:
  uv run python scripts/browser_validation/protonpotts_dump_plan.py --out plan_cases.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

logger = logging.getLogger("protonpotts_dump_plan")

V = 30
DEP_MAP = {"HIS-P": ("HIS-S",), "ASP-P": ("ASP-D",), "GLU-P": ("GLU-D",)}


def _problem(rng: np.random.Generator, length: int, slots: int) -> dict[str, Any]:
  e_idx = np.zeros((length, slots), dtype=np.int32)
  for i in range(length):
    others = np.delete(np.arange(length), i)
    e_idx[i, 0] = i
    e_idx[i, 1:] = rng.choice(others, size=slots - 1, replace=False)
  field = rng.normal(size=(length, V)).astype(np.float32)
  field[rng.random((length, V)) < 0.05] = field.min()  # exact ties
  binder = np.zeros(length, dtype=bool)
  start = int(rng.integers(0, length // 3))
  binder[start : start + int(rng.integers(length // 3, 2 * length // 3))] = True
  res_id = (np.arange(length) + int(rng.integers(-3, 5))).astype(np.int64)
  if length > 10:
    res_id[int(np.flatnonzero(binder)[2])] = res_id[int(np.flatnonzero(binder)[1])]  # a repeated residue number
  return {"length": length, "slots": slots, "e_idx": e_idx, "field": field, "binder": binder, "res_id": res_id}


def _pin(pin: Any) -> dict[str, Any]:  # noqa: ANN401
  return {"position": int(pin.position), "protonationType": pin.protonation_type, "protIdx": int(pin.prot_idx),
          "depIdxs": [int(d) for d in pin.dep_idxs], "resId": int(pin.res_id)}


def _plan(plan: Any) -> dict[str, Any] | None:  # noqa: ANN401
  if plan is None:
    return None
  return {"pins": [_pin(p) for p in plan.pins], "designable": [int(d) for d in plan.designable], "label": plan.label}


def main() -> int:  # noqa: C901
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  from aminx.families.protonpotts_mpnn.ph_plan import (  # noqa: PLC0415
    block_table,
    plan_centre_free,
    plan_from_center_types,
    plan_from_explicit_centers,
    valid_token_mask,
  )

  rng = np.random.default_rng(args.seed)
  cases: list[dict[str, Any]] = []
  for n in range(24):
    length = int(rng.integers(30, 90))
    slots = int(rng.integers(6, 17))
    p = _problem(rng, length, slots)
    base = {"length": length, "slots": slots, "e_idx": p["e_idx"].reshape(-1).tolist(),
            "field": p["field"].reshape(-1).astype(float).tolist(), "binder": p["binder"].astype(int).tolist(),
            "res_id": p["res_id"].tolist(), "vocab": V}
    types = [("HIS-P",), ("HIS-P", "ASP-P", "GLU-P"), ("ASP-P", "ASP-P"), ("GLU-P", "HIS-P")][n % 4]
    scope = ("neighbourhood", "chain")[n % 2]
    neighbour_k = (0, 4, 16)[n % 3]
    max_mut = (0, 3, 20)[(n // 2) % 3]
    opts = {"infillScope": scope, "neighbourK": neighbour_k, "maxMutations": max_mut}
    plan = plan_from_center_types(p["field"], p["e_idx"], p["binder"], p["res_id"], types, DEP_MAP,
                                  infill_scope=scope, neighbour_k=neighbour_k, max_mutations=max_mut)
    cases.append({"name": f"center_types_{n}", "kind": "center_types",
                  "inputs": {**base, "types": list(types), "opts": opts, "dep_map": {k: list(v) for k, v in DEP_MAP.items()}},
                  "expected": _plan(plan)})
    if plan is not None:
      block_size = (1, 2, 3)[n % 3]
      blocks, valid = block_table(p["e_idx"], plan.designable, block_size)
      cases.append({"name": f"block_table_{n}", "kind": "block_table",
                    "inputs": {**base, "designable": [int(d) for d in plan.designable], "block_size": block_size},
                    "expected": {"blocks": blocks.reshape(-1).astype(int).tolist(), "valid": valid.reshape(-1).astype(int).tolist()}})
    free = np.flatnonzero(p["binder"])
    explicit = [(int(p["res_id"][free[1]]), "HIS-P"), (int(p["res_id"][free[-1]]), "ASP-P")]
    plan_e = plan_from_explicit_centers(p["e_idx"], p["binder"], p["res_id"], explicit, DEP_MAP,
                                        infill_scope=scope, neighbour_k=neighbour_k, max_mutations=max_mut)
    cases.append({"name": f"explicit_{n}", "kind": "explicit",
                  "inputs": {**base, "centers": [list(c) for c in explicit], "opts": opts,
                             "dep_map": {k: list(v) for k, v in DEP_MAP.items()}},
                  "expected": _plan(plan_e)})
    cases.append({"name": f"centre_free_{n}", "kind": "centre_free", "inputs": {**base},
                  "expected": _plan(plan_centre_free(p["binder"]))})
  cases.append({"name": "valid_token_mask", "kind": "valid_mask",
                "inputs": {"forbidden": ["HIS-A", "ASP-A", "GLU-A", "UNK", "HIS-D"]},
                "expected": valid_token_mask(("HIS-A", "ASP-A", "GLU-A", "UNK", "HIS-D")).astype(int).tolist()})
  args.out.write_text(json.dumps({"cases": cases}))
  logger.info("wrote %d cases to %s", len(cases), args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
