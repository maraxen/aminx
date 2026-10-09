"""The ``protonpotts_decoder`` wave: the aminx 30-token decoder against upstream's, per step, from the P4h dump.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §53; debt #2617. Sidecar: ``protonpotts_decoder_parity.bth.toml``
(committed before this script). Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_decoder_parity.py \\
        --state-npz <v6_state.npz> --decoder-dir <dir holding protonpotts_v6_decoder/ and decoder_manifest.json> \\
        --features-dir ~/projects/aminx-oracles-protonpotts

WHAT IS COMPARED, per cell (4), precision (f64, f32) and autoregressive configuration (ar_default, ar_bias_temp, ar_fixed): logits,
log_probs, the renormalised probs_sample and each decoder layer's output at every position, in a FORCED-TOKEN replay (the dump's
tokens as the prefix, so one fragile draw cannot cascade); every step's own draw in that replay; and the free-running sequence. Per
teacher-forced pattern (auto_regressive, conditional, conditional_minus_self): logits and log_probs. Tolerances, the fragile-draw
rule and the eight controls are fixed in the sidecar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.potts_mpnn.model import _encoder_states, cast_floating  # noqa: E402
from aminx.families.protonpotts_mpnn import convert  # noqa: E402
from aminx.families.protonpotts_mpnn import decode as decode_mod  # noqa: E402
from aminx.families.protonpotts_mpnn.decode import (  # noqa: E402
  TEACHER_FORCING_PATTERNS,
  ProtonPottsARDecode,
  teacher_forced,
)
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_decoder_parity")

P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"
CELLS = {"pkad_unlabelled": P4B, "multichain": P4B, "gap": P4C, "only_o_1olr": P4D}
CONTROL_CELL = "pkad_unlabelled"
AR_CONFIGS = ("ar_default", "ar_bias_temp", "ar_fixed")
F64_TOL = 1e-9
F32_FACTOR = 10.0
F32_MIN_BAND = 1e-6
FRAGILE_MARGIN = 1e-5
DEFAULT_TEMPERATURE = 0.1  # prepare_potts_input's settings["temperature"]
V = 30
AR_ARRAYS = ("logits", "log_probs", "probs_sample", "decoder_layers")
TF_ARRAYS = ("logits", "log_probs")


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


def regenerate_inputs(config: str, n: int, seed: int, fixed_every: int, fixed_offset: int, temperature_default: float):  # noqa: ANN201
  """Per-residue temperature, bias and designed mask, built exactly as the dumper's ``_edits`` did."""
  designed = np.ones(n, dtype=bool)
  temperature = np.full(n, temperature_default)
  bias = np.zeros((n, V))
  if config == "ar_bias_temp":
    rng = np.random.default_rng(seed + 17)
    temperature = rng.uniform(0.3, 1.0, size=(1, n))[0]
    bias = 0.5 * rng.standard_normal(size=(1, n, V))[0]
  elif config == "ar_fixed":
    designed[fixed_offset::fixed_every] = False
  return temperature, bias, designed


class Cell:
  """One cell in one precision: the aminx encoder states, and the dump's arrays for that precision."""

  def __init__(self, model, features: dict[str, np.ndarray], dtype, up: dict[str, np.ndarray]) -> None:  # noqa: ANN001
    self.dtype = dtype
    self.up = up
    scored = cast_floating(model, dtype)
    self.scored = scored
    coords = jnp.asarray(features["X"][0, :, :4, :], dtype=dtype)
    self.n = int(coords.shape[0])
    self.present = jnp.ones(self.n, dtype=dtype)
    nodes, edges, e_idx = _encoder_states(
      scored.mpnn, coords, self.present, jnp.asarray(features["R_idx"][0]), jnp.asarray(features["chain_labels"][0]),
      inference=True,
    )
    self.h_v, self.h_e, self.e_idx = nodes[-1], edges[-1], e_idx
    self.native = np.vectorize(upstream_to_aminx_index)(features["S"][0]).astype(np.int32)
    self.decoder = ProtonPottsARDecode(
      layers=scored.mpnn.decoder.layers, w_s_embed=scored.mpnn.w_s_embed, w_out=scored.mpnn.w_out,
    )

  def ar(self, config: str, regen: tuple, *, forced: bool, bias=None, temperature=None, designed=None,  # noqa: ANN001
         uniforms_shift: int = 0) -> decode_mod.DecodeResult:
    temp, b, des = regen
    temp = temp if temperature is None else temperature
    b = b if bias is None else bias
    des = des if designed is None else designed
    uniforms = np.roll(self.up[f"{config}__uniforms"], uniforms_shift)
    forced_tokens = self.up[f"{config}__S_sampled"].astype(np.int32) if forced else None
    return self.decoder(
      self.h_v, self.h_e, self.e_idx, self.present, jnp.ones(self.n, dtype=bool), jnp.asarray(self.native),
      jnp.asarray(des), jnp.asarray(temp, dtype=self.dtype), jnp.asarray(b, dtype=self.dtype),
      jnp.asarray(uniforms, dtype=self.dtype), decoding_order=jnp.asarray(self.up[f"{config}__decoding_order"]),
      forced_tokens=None if forced_tokens is None else jnp.asarray(forced_tokens),
    )

  def tf(self, pattern: str, regen: tuple, *, seq: np.ndarray | None = None, mask_pattern: str | None = None):  # noqa: ANN201
    temp, b, _ = regen
    return teacher_forced(
      self.scored.mpnn.decoder, self.scored.mpnn.w_s_embed, self.scored.mpnn.w_out, self.h_v, self.h_e, self.e_idx,
      self.present, jnp.asarray(self.native if seq is None else seq), jnp.asarray(temp, dtype=self.dtype),
      jnp.asarray(b, dtype=self.dtype), pattern=mask_pattern or pattern,
      decoding_order=jnp.asarray(self.up[f"tf_{pattern}__decoding_order"]),
    )


def band(key_diff_floor: float | None) -> float:
  return F64_TOL if key_diff_floor is None else max(F32_FACTOR * key_diff_floor, F32_MIN_BAND)


def ar_arrays(result: decode_mod.DecodeResult) -> dict[str, np.ndarray]:
  return {
    "logits": np.asarray(result.logits), "log_probs": np.asarray(result.log_probs),
    "probs_sample": np.asarray(result.probs_sample), "decoder_layers": np.asarray(result.h_v_stack)[1:],
  }


def grade_ar(cell: Cell, config: str, regen: tuple, floors: dict[str, float] | None, *, override: dict | None = None,
             nudge: float = 0.0) -> dict:
  """Everything the sidecar asks of one autoregressive configuration; ``override`` feeds a deliberate error."""
  override = override or {}
  up = cell.up
  margin = up[f"{config}__draw_margin"]
  fragile = margin < FRAGILE_MARGIN
  replay = cell.ar(config, regen, forced=True, **override)
  got = ar_arrays(replay)
  diffs, in_band = {}, True
  for key in AR_ARRAYS:
    if nudge and key == "log_probs":
      got[key] = got[key].copy()
      got[key].flat[0] += nudge
    d = float(np.abs(got[key].astype(np.float64) - up[f"{config}__{key}"].astype(np.float64)).max())
    diffs[key] = d
    in_band &= d <= band(None if floors is None else floors[f"{config}__{key}"])
  draws_exact = bool(np.array_equal(np.asarray(replay.drawn_token)[~fragile], up[f"{config}__drawn_token"][~fragile]))
  free = cell.ar(config, regen, forced=False, **override)
  exempt = bool(fragile.any())
  seq_exact = exempt or bool(np.array_equal(np.asarray(free.sequence), up[f"{config}__S_sampled"]))
  return {"diffs": diffs, "in_band": in_band, "draws_exact": draws_exact, "sequence_exact": seq_exact,
          "n_fragile": int(fragile.sum()), "exempt_free_run": exempt,
          "ok": in_band and draws_exact and seq_exact}


def grade_tf(cell: Cell, pattern: str, regen: tuple, floors: dict[str, float] | None, *, seq=None,  # noqa: ANN001
             mask_pattern: str | None = None) -> dict:
  logits, log_probs = cell.tf(pattern, regen, seq=seq, mask_pattern=mask_pattern)
  diffs, in_band = {}, True
  for key, value in (("logits", logits), ("log_probs", log_probs)):
    d = float(np.abs(np.asarray(value, dtype=np.float64) - cell.up[f"tf_{pattern}__{key}"].astype(np.float64)).max())
    diffs[key] = d
    in_band &= d <= band(None if floors is None else floors[f"tf_{pattern}__{key}"])
  return {"diffs": diffs, "in_band": in_band, "ok": in_band}


@contextmanager
def _x_not_zeroed():  # noqa: ANN202
  original = decode_mod._unknown_mask  # noqa: SLF001
  decode_mod._unknown_mask = lambda vocab, dtype: jnp.zeros((vocab,), dtype=dtype)  # noqa: SLF001
  try:
    yield
  finally:
    decode_mod._unknown_mask = original  # noqa: SLF001


def main() -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--decoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  state = _npz(args.state_npz)
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  manifest = json.loads((args.decoder_dir / "decoder_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == manifest["checkpoint_sha256"]
  seed, fixed_every, fixed_offset = manifest["seed"], manifest["fixed_every"], manifest["fixed_offset"]
  model = _load_converter()(state)

  per_cell: dict[str, dict] = {}
  f64_in_band = f32_in_band = draws_exact = sequences_exact = True
  n_fragile = n_exempt = 0
  worst_ratio = 0.0
  cells_by_precision: dict[tuple[str, str], Cell] = {}
  for cell_name, root in CELLS.items():
    features = _npz(args.features_dir / root / f"{cell_name}.npz")
    up64 = _npz(args.decoder_dir / "protonpotts_v6_decoder" / f"{cell_name}_f64.npz")
    up32 = _npz(args.decoder_dir / "protonpotts_v6_decoder" / f"{cell_name}_f32.npz")
    floors = {k: float(np.abs(up64[k].astype(np.float64) - up32[k].astype(np.float64)).max())
              for k in up64 if k.rsplit("__", 1)[-1] in (*AR_ARRAYS, *TF_ARRAYS)}
    entry: dict = {}
    for precision, dtype, up, fl in (("f64", jnp.float64, up64, None), ("f32", jnp.float32, up32, floors)):
      cell = Cell(model, features, dtype, up)
      cells_by_precision[(cell_name, precision)] = cell
      result: dict = {"ar": {}, "tf": {}}
      temp_default = DEFAULT_TEMPERATURE  # the prepared features' per-residue temperature, as dumped
      for config in AR_CONFIGS:
        regen = regenerate_inputs(config, cell.n, seed, fixed_every, fixed_offset, temp_default)
        graded = grade_ar(cell, config, regen, fl)
        result["ar"][config] = graded
        draws_exact &= graded["draws_exact"]
        sequences_exact &= graded["sequence_exact"]
        n_fragile += graded["n_fragile"]
        n_exempt += int(graded["exempt_free_run"])
        for key, d in graded["diffs"].items():
          b = band(None if fl is None else fl[f"{config}__{key}"])
          worst_ratio = max(worst_ratio, d / b)
        if precision == "f64":
          f64_in_band &= graded["in_band"]
        else:
          f32_in_band &= graded["in_band"]
      regen_tf = regenerate_inputs("ar_default", cell.n, seed, fixed_every, fixed_offset, temp_default)
      for pattern in TEACHER_FORCING_PATTERNS:
        graded = grade_tf(cell, pattern, regen_tf, fl)
        result["tf"][pattern] = graded
        for key, d in graded["diffs"].items():
          worst_ratio = max(worst_ratio, d / band(None if fl is None else fl[f"tf_{pattern}__{key}"]))
        if precision == "f64":
          f64_in_band &= graded["in_band"]
        else:
          f32_in_band &= graded["in_band"]
      entry[precision] = result
      logger.info("%s %s: ar %s tf %s", cell_name, precision,
                  {c: r["ok"] for c, r in result["ar"].items()}, {p: r["ok"] for p, r in result["tf"].items()})
    per_cell[cell_name] = entry

  # ---- the eight deliberate errors, on the control cell in float64 -----------------------------------
  cell = cells_by_precision[(CONTROL_CELL, "f64")]
  n = cell.n
  regen = {c: regenerate_inputs(c, n, seed, fixed_every, fixed_offset, DEFAULT_TEMPERATURE) for c in AR_CONFIGS}
  mutants: dict[str, str] = {}

  def verdict(ok: bool) -> str:  # noqa: FBT001
    return "passed" if ok else "failed"

  temp_bias = regen["ar_bias_temp"][0]
  mutants["bias_ignored"] = verdict(grade_ar(cell, "ar_bias_temp", regen["ar_bias_temp"], None,
                                             override={"bias": np.zeros((n, V))})["ok"])
  mutants["scalar_temperature"] = verdict(grade_ar(cell, "ar_bias_temp", regen["ar_bias_temp"], None,
                                                   override={"temperature": np.full(n, temp_bias.mean())})["ok"])
  with _x_not_zeroed():
    mutants["x_not_zeroed"] = verdict(grade_ar(cell, "ar_default", regen["ar_default"], None)["ok"])
  mutants["fixed_treated_as_designed"] = verdict(grade_ar(cell, "ar_fixed", regen["ar_fixed"], None,
                                                          override={"designed": np.ones(n, dtype=bool)})["ok"])
  upstream_order = np.argsort(convert.token_permutation())[cell.native].astype(np.int32)
  mutants["token_order_upstream"] = verdict(grade_tf(cell, "conditional_minus_self", regen["ar_default"], None,
                                                     seq=upstream_order)["ok"])
  mutants["cms_uses_conditional_mask"] = verdict(grade_tf(cell, "conditional_minus_self", regen["ar_default"], None,
                                                          mask_pattern="conditional")["ok"])
  mutants["uniforms_shifted_one_step"] = verdict(grade_ar(cell, "ar_default", regen["ar_default"], None,
                                                          override={"uniforms_shift": 1})["ok"])
  mutants["nudged_4x_band"] = verdict(grade_ar(cell, "ar_default", regen["ar_default"], None, nudge=4.0 * F64_TOL)["ok"])
  for name, v in mutants.items():
    logger.info("mutant %-30s %s", name, "DETECTED" if v == "failed" else "NOT DETECTED")

  clean = "pass" if (f64_in_band and f32_in_band and draws_exact and sequences_exact and checkpoint_matches) else "fail"
  results = {
    "clean": clean, "n_cells": len(per_cell), "f64_in_band": bool(f64_in_band), "f32_in_band": bool(f32_in_band),
    "draws_exact": bool(draws_exact), "sequences_exact": bool(sequences_exact), "checkpoint_matches": checkpoint_matches,
    "n_fragile_steps": int(n_fragile), "n_configs_exempt_from_free_run": int(n_exempt),
    "worst_over_band": float(worst_ratio), "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants, "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_cell": per_cell,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_decoder_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
