"""The ``protonpotts_sample`` wave: aminx ``sample_rows`` (the engine's ``mpnn_sample``) against upstream's batched call, from the P4i dump.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §55; debt #2617. Sidecar: ``protonpotts_sample_parity.bth.toml``
(committed before this script). Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_sample_parity.py \\
        --state-npz <v6_state.npz> --sample-dir <dir holding protonpotts_v6_sample/ and sample_manifest.json> \\
        --features-dir ~/projects/aminx-oracles-protonpotts

WHAT IS COMPARED, per cell (4), precision (f64, f32) and row (3): logits, log_probs and the renormalised probs_sample at every position
in a FORCED-TOKEN replay (the dump's tokens as each row's prefix, so one fragile draw cannot cascade); every step's own draw except at
fragile positions; the free-running sampled sequence (exact, a row with a fragile draw exempt); and that non-designed positions keep
their native token. Tolerances, the fragile-draw rule and the eight controls are fixed in the sidecar.
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

from aminx.families.potts_mpnn.model import _encoder_states, cast_floating  # noqa: E402
from aminx.families.protonpotts_mpnn import convert  # noqa: E402
from aminx.families.protonpotts_mpnn import decode as decode_mod  # noqa: E402
from aminx.families.protonpotts_mpnn.decode import ProtonPottsARDecode, sample_rows  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_sample_parity")

P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"
CELLS = {"pkad_unlabelled": P4B, "multichain": P4B, "gap": P4C, "only_o_1olr": P4D}
CONTROL_CELL = "multichain"
F64_TOL = 1e-9
F32_FACTOR = 10.0
F32_MIN_BAND = 1e-6
FRAGILE_MARGIN = 1e-5
DEFAULT_TEMPERATURE = float(np.float32(0.1))  # prepare_potts_input's 0.1, stored float32 in the features and cast up in f64
V = 30
ARRAYS = ("logits", "log_probs", "probs_sample")
TAKE = convert.token_permutation()  # TAKE[k] = the upstream index of the token at aminx index k
INV = np.argsort(TAKE)  # INV[u] = the aminx index of upstream token u


def _load_converter() -> Callable:
  path = Path(__file__).resolve().parent / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_state


def _npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def band(floor: float | None) -> float:
  return F64_TOL if floor is None else max(F32_FACTOR * floor, F32_MIN_BAND)


class Cell:
  """One cell in one precision: the aminx encoder states and the dump's arrays for that precision."""

  def __init__(self, model, features: dict[str, np.ndarray], dtype, up: dict[str, np.ndarray]) -> None:  # noqa: ANN001
    self.dtype = dtype
    self.up = up
    scored = cast_floating(model, dtype)
    coords = jnp.asarray(features["X"][0, :, :4, :], dtype=dtype)
    self.n = int(coords.shape[0])
    self.present = jnp.ones(self.n, dtype=dtype)
    nodes, edges, e_idx = _encoder_states(
      scored.mpnn, coords, self.present, jnp.asarray(features["R_idx"][0]), jnp.asarray(features["chain_labels"][0]),
      inference=True,
    )
    self.h_v, self.h_e, self.e_idx = nodes[-1], edges[-1], e_idx
    self.native = np.vectorize(upstream_to_aminx_index)(features["S"][0]).astype(np.int32)
    self.designed = up["designed"].astype(bool)
    self.decoder = ProtonPottsARDecode(
      layers=scored.mpnn.decoder.layers, w_s_embed=scored.mpnn.w_s_embed, w_out=scored.mpnn.w_out,
    )

  def rows(self, *, forced: bool, order=None, uniforms=None, designed=None, temperature=None,  # noqa: ANN001
           upstream_cdf: bool = True) -> decode_mod.DecodeResult:
    order = self.up["decoding_order"] if order is None else order
    uniforms = self.up["uniforms"] if uniforms is None else uniforms
    designed = self.designed if designed is None else designed
    temperature = np.full(self.n, DEFAULT_TEMPERATURE) if temperature is None else temperature
    forced_tokens = INV[self.up["S_sampled"]].astype(np.int32) if forced else None
    return sample_rows(
      self.decoder, self.h_v, self.h_e, self.e_idx, self.present, jnp.ones(self.n, dtype=bool), jnp.asarray(self.native),
      jnp.asarray(designed), jnp.asarray(temperature, dtype=self.dtype), jnp.zeros((self.n, V), dtype=self.dtype),
      jnp.asarray(uniforms, dtype=self.dtype), decoding_order=jnp.asarray(order),
      forced_tokens=None if forced_tokens is None else jnp.asarray(forced_tokens),
      cdf_order=jnp.asarray(INV) if upstream_cdf else None,
    )


def grade(cell: Cell, floors: dict[str, float] | None, *, override: dict | None = None, nudge: float = 0.0) -> dict:
  """Everything the sidecar asks of one cell and precision; ``override`` feeds a deliberate error."""
  override = override or {}
  up = cell.up
  fragile = up["draw_margin"] < FRAGILE_MARGIN  # (N, L)
  replay = cell.rows(forced=True, **override)
  got = {"logits": np.asarray(replay.logits), "log_probs": np.asarray(replay.log_probs),
         "probs_sample": np.asarray(replay.probs_sample)}
  diffs, in_band = {}, True
  for key in ARRAYS:
    if nudge and key == "log_probs":
      got[key] = got[key].copy()
      got[key].flat[0] += nudge
    want = up[key][..., TAKE]
    d = float(np.abs(got[key].astype(np.float64) - want.astype(np.float64)).max())
    diffs[key] = d
    in_band &= d <= band(None if floors is None else floors[key])
  want_draws = INV[up["drawn_token"]]
  draws_exact = bool(np.array_equal(np.asarray(replay.drawn_token)[~fragile], want_draws[~fragile]))
  free = cell.rows(forced=False, **override)
  free_seq = np.asarray(free.sequence)
  want_seq = INV[up["S_sampled"]]
  row_exempt = fragile.any(axis=1)
  seq_exact = all(bool(np.array_equal(free_seq[r], want_seq[r])) for r in range(free_seq.shape[0]) if not row_exempt[r])
  nondesigned_hold = bool(np.all(free_seq[:, ~cell.designed] == cell.native[~cell.designed]))
  return {"diffs": diffs, "in_band": in_band, "draws_exact": draws_exact, "sequence_exact": seq_exact,
          "nondesigned_hold": nondesigned_hold, "n_fragile": int(fragile.sum()), "n_rows_exempt": int(row_exempt.sum()),
          "ok": in_band and draws_exact and seq_exact and nondesigned_hold}


def main() -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--sample-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  state = _npz(args.state_npz)
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  manifest = json.loads((args.sample_dir / "sample_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == manifest["checkpoint_sha256"]
  model = _load_converter()(state)

  per_cell: dict[str, dict] = {}
  f64_in_band = f32_in_band = draws_exact = sequences_exact = nondesigned_hold = True
  n_fragile = n_exempt = 0
  worst_ratio = 0.0
  cells_by_precision: dict[tuple[str, str], Cell] = {}
  for cell_name, root in CELLS.items():
    features = _npz(args.features_dir / root / f"{cell_name}.npz")
    up64 = _npz(args.sample_dir / "protonpotts_v6_sample" / f"{cell_name}_f64.npz")
    up32 = _npz(args.sample_dir / "protonpotts_v6_sample" / f"{cell_name}_f32.npz")
    floors = {k: float(np.abs(up64[k].astype(np.float64) - up32[k].astype(np.float64)).max()) for k in ARRAYS}
    entry: dict = {}
    for precision, dtype, up, fl in (("f64", jnp.float64, up64, None), ("f32", jnp.float32, up32, floors)):
      cell = Cell(model, features, dtype, up)
      cells_by_precision[(cell_name, precision)] = cell
      graded = grade(cell, fl)
      entry[precision] = graded
      draws_exact &= graded["draws_exact"]
      sequences_exact &= graded["sequence_exact"]
      nondesigned_hold &= graded["nondesigned_hold"]
      n_fragile += graded["n_fragile"]
      n_exempt += graded["n_rows_exempt"]
      for key, d in graded["diffs"].items():
        worst_ratio = max(worst_ratio, d / band(None if fl is None else fl[key]))
      if precision == "f64":
        f64_in_band &= graded["in_band"]
      else:
        f32_in_band &= graded["in_band"]
      logger.info("%s %s: ok=%s diffs=%s", cell_name, precision, graded["ok"], {k: f"{v:.2e}" for k, v in graded["diffs"].items()})
    per_cell[cell_name] = entry

  # ---- the eight deliberate errors, on the control cell in float64 ----------------------------------
  cell = cells_by_precision[(CONTROL_CELL, "f64")]
  up = cell.up
  n_rows = up["decoding_order"].shape[0]
  mutants: dict[str, str] = {}

  def verdict(ok: bool) -> str:  # noqa: FBT001
    return "passed" if ok else "failed"

  mutants["shared_order_across_rows"] = verdict(grade(
    cell, None, override={"order": np.broadcast_to(up["decoding_order"][:1], up["decoding_order"].shape)})["ok"])
  swapped = up["uniforms"][[1, 0, *range(2, n_rows)]]
  mutants["uniforms_rows_swapped"] = verdict(grade(cell, None, override={"uniforms": swapped})["ok"])
  mutants["all_residues_designed"] = verdict(grade(cell, None, override={"designed": np.ones(cell.n, dtype=bool)})["ok"])
  mutants["cdf_in_aminx_order"] = verdict(grade(cell, None, override={"upstream_cdf": False})["ok"])
  mutants["uniforms_shifted_one_step"] = verdict(grade(cell, None, override={"uniforms": np.roll(up["uniforms"], 1, axis=1)})["ok"])
  mutants["single_row_repeated"] = verdict(grade(cell, None, override={
    "order": np.broadcast_to(up["decoding_order"][:1], up["decoding_order"].shape),
    "uniforms": np.broadcast_to(up["uniforms"][:1], up["uniforms"].shape)})["ok"])
  mutants["temperature_one"] = verdict(grade(cell, None, override={"temperature": np.ones(cell.n)})["ok"])
  mutants["nudged_4x_band"] = verdict(grade(cell, None, nudge=4.0 * F64_TOL)["ok"])
  for name, v in mutants.items():
    logger.info("mutant %-28s %s", name, "DETECTED" if v == "failed" else "NOT DETECTED")

  clean = "pass" if (f64_in_band and f32_in_band and draws_exact and sequences_exact and nondesigned_hold and checkpoint_matches) else "fail"
  results = {
    "clean": clean, "n_cells": len(per_cell), "f64_in_band": bool(f64_in_band), "f32_in_band": bool(f32_in_band),
    "draws_exact": bool(draws_exact), "sequences_exact": bool(sequences_exact), "nondesigned_hold": bool(nondesigned_hold),
    "checkpoint_matches": checkpoint_matches, "n_fragile_steps": int(n_fragile), "n_rows_exempt_from_free_run": int(n_exempt),
    "worst_over_band": float(worst_ratio), "n_listed": len(mutants), "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants, "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_cell": per_cell,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_sample_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
