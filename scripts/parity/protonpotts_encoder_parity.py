"""The ``protonpotts_encoder`` wave: the converted aminx encoder and Potts head against upstream, layer by layer.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §7, §41. Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_encoder_parity.py \\
        --state-npz <v6_state.npz> --encoder-dir <dir holding protonpotts_v6_encoder/> \\
        --features-dir ~/projects/aminx-oracles-protonpotts

WHAT IS COMPARED, per cell (4) and precision (f32, f64): the layer-0 input and every encoder layer's output
(``h_V``, ``h_E``), ``E_idx``, and the head table BEFORE the reciprocal merge (``etab_premerge``), which is
what aminx's ``PottsHead`` computes. The upstream arrays are the sealed P4e dump; the inputs are the sealed
feature dumps (P4b/P4c/P4d), whose hashes P4e verified.

TOLERANCES, fixed before any result exists:
  f64       every array within 1e-9 absolute;
  f32       every array within 10x the MEASURED upstream f32-vs-f64 spread for that array (decision 11e: a
            band from a measured floor, not a guess), with no floor below 1e-7;
  E_idx     exactly equal.

WHAT MAKES A PASS MEAN SOMETHING (spec §6): seven perturbations must each be REJECTED by the same comparison:
the three orderings the conversion must permute (pair order, token order, both etab axes), each in the ways
they are classically wrong, and two nudges of a correct output that sit just outside each band.

``etab_premerge`` is where a wrong token or etab permutation shows: the encoder's activations do not depend
on token order at all, so an encoder-only comparison cannot catch them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.protonpotts_mpnn import convert  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import upstream_to_aminx_index  # noqa: E402

logger = logging.getLogger("protonpotts_encoder_parity")

P4B = "features_v6/protonpotts_v6_features"
P4C = "features_v6_p4c/protonpotts_v6_features_p4c"
P4D = "features_v6_p4d/protonpotts_v6_features_p4c"
CELLS = {
  "pkad_unlabelled": P4B,
  "multichain": P4B,
  "gap": P4C,
  "only_o_1olr": P4D,
}
F64_TOL = 1e-9
F32_FACTOR = 10.0
F32_MIN_BAND = 1e-7
ENCODER_LAYERS = 3
H_KEYS = ("enc_in_h_E", *(f"enc_L{i}_h_{n}" for i in range(ENCODER_LAYERS) for n in ("V", "E")))
ARRAY_KEYS = ("enc_in_h_V", *H_KEYS, "etab_premerge")
CONTROL_CELL = "pkad_unlabelled"


def _load_converter() -> Callable:
  path = Path(__file__).resolve().parent / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_state


def _reorder_table(table: np.ndarray) -> np.ndarray:
  """Upstream's ``(L, K, V, V)`` table in foundry token order -> aminx token order, both axes."""
  take = convert.token_permutation()
  return table[:, :, take][:, :, :, take]


def run_aminx(model, features: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:  # noqa: ANN001
  coords = jnp.asarray(features["X"][0, :, :4, :], dtype=dtype)
  length = coords.shape[0]
  sequence = np.vectorize(upstream_to_aminx_index)(features["S"][0])
  out = model(
    coords,
    jnp.ones(length, dtype=dtype),
    jnp.asarray(features["R_idx"][0]),
    jnp.asarray(features["chain_labels"][0]),
    jnp.asarray(sequence),
    jnp.arange(length),
    inference=True,
  )
  got = {"enc_in_h_V": np.asarray(out.enc_h_v[0]), "enc_in_h_E": np.asarray(out.enc_h_e[0])}
  for i in range(ENCODER_LAYERS):
    got[f"enc_L{i}_h_V"] = np.asarray(out.enc_h_v[i + 1])
    got[f"enc_L{i}_h_E"] = np.asarray(out.enc_h_e[i + 1])
  got["E_idx"] = np.asarray(out.neighbor_indices)
  got["etab_premerge"] = np.asarray(out.etab_raw)
  return got


def compare(
  got: dict[str, np.ndarray],
  up: dict[str, np.ndarray],
  floor: dict[str, float] | None,
) -> tuple[dict[str, float], bool, bool]:
  """Max abs diff per array, whether E_idx is exact, and whether every array is in its band."""
  diffs: dict[str, float] = {}
  in_band = True
  for key in ARRAY_KEYS:
    want = _reorder_table(up[key]) if key == "etab_premerge" else up[key]
    diff = float(np.abs(got[key].astype(np.float64) - want.astype(np.float64)).max())
    diffs[key] = diff
    band = F64_TOL if floor is None else max(F32_FACTOR * floor[key], F32_MIN_BAND)
    in_band &= diff <= band
  e_idx_ok = bool(np.array_equal(got["E_idx"], up["E_idx"]))
  return diffs, e_idx_ok, in_band and e_idx_ok


# --- the perturbations ---------------------------------------------------------------


@contextmanager
def _patched(**replacements: object) -> Generator[None]:
  originals = {name: getattr(convert, name) for name in replacements}
  for name, value in replacements.items():
    setattr(convert, name, value)
  try:
    yield
  finally:
    for name, value in originals.items():
      setattr(convert, name, value)


def _identity(array: np.ndarray) -> np.ndarray:
  return array


def _inverse_pair_edge_embedding(weight: np.ndarray) -> np.ndarray:
  """The classic mistake: apply the inverse of the right pair permutation."""
  inverse = np.argsort(convert.pair_permutation())
  rbf = weight[:, convert.N_POSITIONAL :].reshape(weight.shape[0], convert.N_ATOM_PAIRS, convert.N_RBF)
  return np.concatenate(
    [weight[:, : convert.N_POSITIONAL], rbf[:, inverse, :].reshape(weight.shape[0], -1)], axis=1
  )


def _etab_rows_only(array: np.ndarray) -> np.ndarray:
  """Permute only the first token axis of the pair table."""
  take = convert.token_permutation()
  table = array.reshape(convert.V, convert.V, *array.shape[1:])
  return table[take].reshape(array.shape)


CONVERSION_MUTANTS: dict[str, dict[str, object]] = {
  "pair_order_not_permuted": {"permute_edge_embedding": _identity},
  "pair_order_inverse_permutation": {"permute_edge_embedding": _inverse_pair_edge_embedding},
  "token_order_not_permuted": {"token_permutation": lambda: np.arange(convert.V)},
  "etab_not_permuted": {"permute_etab": _identity},
  "etab_first_axis_only": {"permute_etab": _etab_rows_only},
}


# --- the wave ------------------------------------------------------------------------


def _load_up(directory: Path, cell: str, precision: str) -> dict[str, np.ndarray]:
  with np.load(directory / "protonpotts_v6_encoder" / f"{cell}_{precision}.npz") as z:
    return {key: z[key] for key in z.files}


def _load_features(root: Path, cell: str) -> dict[str, np.ndarray]:
  with np.load(root / CELLS[cell] / f"{cell}.npz") as z:
    return {key: z[key] for key in z.files}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  with np.load(args.state_npz) as z:
    state = {key: z[key] for key in z.files}
  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  dump_meta = json.loads((args.encoder_dir / "encoder_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == dump_meta["checkpoint_sha256"]

  convert_state = _load_converter()
  model = convert_state(state)

  per_cell: dict[str, dict] = {}
  n_f64 = n_f32 = n_idx = 0
  worst_f64 = 0.0
  worst_f32_ratio = 0.0
  for cell in CELLS:
    features = _load_features(args.features_dir, cell)
    up64, up32 = _load_up(args.encoder_dir, cell, "f64"), _load_up(args.encoder_dir, cell, "f32")
    floor = {k: float(np.abs(up64[k].astype(np.float64) - up32[k].astype(np.float64)).max()) for k in ARRAY_KEYS}
    d64, idx64, ok64 = compare(run_aminx(model, features, jnp.float64), up64, None)
    d32, idx32, ok32 = compare(run_aminx(model, features, jnp.float32), up32, floor)
    n_f64 += ok64
    n_f32 += ok32
    n_idx += idx64 and idx32
    worst_f64 = max(worst_f64, *d64.values())
    worst_f32_ratio = max(worst_f32_ratio, *(d32[k] / max(floor[k], F32_MIN_BAND) for k in ARRAY_KEYS))
    per_cell[cell] = {"f64_max": d64, "f32_max": d32, "floor": floor, "ok_f64": ok64, "ok_f32": ok32}
    logger.info("%s f64_ok=%s f32_ok=%s", cell, ok64, ok32)

  clean = "pass" if (n_f64 == len(CELLS) and n_f32 == len(CELLS) and n_idx == len(CELLS) and checkpoint_matches) else "fail"

  mutants: dict[str, str] = {}
  features = _load_features(args.features_dir, CONTROL_CELL)
  up64 = _load_up(args.encoder_dir, CONTROL_CELL, "f64")
  for name, replacement in CONVERSION_MUTANTS.items():
    with _patched(**replacement):
      mutated = convert_state(state)
    _, _, ok = compare(run_aminx(mutated, features, jnp.float64), up64, None)
    mutants[name] = "passed" if ok else "failed"
  good = run_aminx(model, features, jnp.float64)
  nudged = dict(good)
  nudged["enc_L2_h_E"] = good["enc_L2_h_E"].copy()
  nudged["enc_L2_h_E"].flat[0] += 1e-6
  mutants["output_nudged_1e-6_f64"] = "passed" if compare(nudged, up64, None)[2] else "failed"
  up32 = _load_up(args.encoder_dir, CONTROL_CELL, "f32")
  floor32 = {k: float(np.abs(up64[k].astype(np.float64) - up32[k].astype(np.float64)).max()) for k in ARRAY_KEYS}
  good32 = run_aminx(model, features, jnp.float32)
  nudged32 = dict(good32)
  nudged32["enc_L2_h_E"] = good32["enc_L2_h_E"].copy()
  nudged32["enc_L2_h_E"].flat[0] += max(F32_FACTOR * floor32["enc_L2_h_E"], F32_MIN_BAND) * 4.0
  mutants["output_nudged_4x_band_f32"] = "passed" if compare(nudged32, up32, floor32)[2] else "failed"
  for name, verdict in mutants.items():
    logger.info("mutant %-34s %s", name, "DETECTED" if verdict == "failed" else "NOT DETECTED")

  results = {
    "clean": clean,
    "n_cells": len(CELLS),
    "n_f64_ok": int(n_f64),
    "n_f32_ok": int(n_f32),
    "n_e_idx_exact": int(n_idx),
    "checkpoint_matches": checkpoint_matches,
    "worst_f64_abs": worst_f64,
    "worst_f32_over_band_floor": worst_f32_ratio,
    "n_listed": len(mutants),
    "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"),
    "per_cell": per_cell,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_encoder_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
