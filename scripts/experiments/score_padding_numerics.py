"""Do #3510's shape paddings change scores beyond SCORE_BATCH_COMPOSITION_ATOL? (PR #300 review)

0.5.5 pads two axes so units share compiled kernels:

* N, the sequences per score_conditionals unit, padded with copies of row 0 (the copies are
  dropped);
* M, LigandMPNN's per-residue ligand atom axis, padded at the end with mask-0 atoms. Below
  ``atom_context_num`` (25), this changes how many atoms each residue reads (k_y).

Both should change real outputs only by float reassociation, which #3487 bounded at 1e-5 for
ProteinMPNN on CPU. This run measures the deltas for the three families tev_design's screen uses
(ProteinMPNN, SolubleMPNN, LigandMPNN with a ligand), on whatever backend ``JAX_PLATFORMS`` selects,
so the GPU numbers are recorded rather than assumed.

Each measurement goes through ``runner.score``, the function the campaign unit calls, with the
explicit ``ar_masks`` path (K masks):

* N padding: score [s0, s1, s2] and [s0, s1, s2, s0]; compare the three real rows.
* M padding (LigandMPNN; M in 8, 20, 27): score with the ligand ladder on versus off
  (``_sampling_helper.ligand_atom_rung`` replaced by the identity), at the same residue rung.
* Planted (instrument): M=8 real atoms plus 9 mask-0 atoms placed at real-looking coordinates
  with oxygen types, against the unpadded M=8 reference.

Controls:

* Positive: moving the real (mask-1) ligand atoms must move the logits by more than 1e-3. If
  not, an M delta of 0 says nothing.
* Negative: an identical call repeated must give a delta of exactly 0. If not, the backend is
  nondeterministic at these shapes and padding deltas cannot be separated from run-to-run noise.

task_id: 261119_aminx-3510-compile-shape-classes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("score_padding_numerics")

_STRUCTURE = "tests/data/1ubq.pdb"
_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_L = len(_SEQ)
_MAX_LENGTH = 96
_FAMILIES = {
  "proteinmpnn": "proteinmpnn_v_48_020",
  "solublempnn": "solublempnn_v_48_020",
  "ligandmpnn": "ligandmpnn_v_32_020_25",
}
_ATOMS = (8, 20, 27)


def _git_hash() -> str:
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _designable() -> np.ndarray:
  design = np.ones(_L, np.int8)
  design[_L - 10 :] = 0
  return design


def _masks(k: int, seed: int = 0) -> np.ndarray:
  design = _designable().astype(bool)
  rng = np.random.default_rng(seed)
  out = np.zeros((k, _L, _L), np.int8)
  idx = np.flatnonzero(design)
  for j in range(k):
    order = np.full(_L, -1)
    order[idx] = rng.permutation(idx.size)
    see = (order[None, :] < order[:, None]) & design[:, None] & design[None, :]
    see |= ~design[None, :]
    see |= (~design[:, None]) & design[None, :]
    np.fill_diagonal(see, False)
    out[j] = see.astype(np.int8)
  return out


def _mutants(n: int) -> list[str]:
  letters = "ACDEFGHIKLMNPQRSTVWY"
  return [_SEQ[:5] + letters[(3 * i) % 20] + _SEQ[6:] for i in range(n)]


def _ligand_npz(path: Path, n_atoms: int, *, n_masked: int = 0, seed: int = 0) -> str:
  from aminx.host._sampling_helper import _canonical_structure_ids_for_spec
  from aminx.run.specs import ScoringSpecification

  (structure_id,) = _canonical_structure_ids_for_spec(
    ScoringSpecification(inputs=[_STRUCTURE], sequences_to_score=[_SEQ], max_length=_MAX_LENGTH),
  )
  real = np.random.default_rng(seed).normal(scale=6.0, size=(_MAX_LENGTH, n_atoms, 3))
  planted = np.random.default_rng(seed + 1000).normal(scale=6.0, size=(_MAX_LENGTH, n_masked, 3))
  total = n_atoms + n_masked
  y_t = np.full((_MAX_LENGTH, total), 6, dtype=np.int32)
  y_t[:, n_atoms:] = 8
  y_m = np.zeros((_MAX_LENGTH, total), dtype=np.float32)
  y_m[:, :n_atoms] = 1.0
  np.savez(
    path,
    **{
      f"{structure_id}::Y": np.concatenate([real, planted], axis=1).astype(np.float32),
      f"{structure_id}::Y_t": y_t,
      f"{structure_id}::Y_m": y_m,
    },
  )
  return str(path)


def _score(checkpoint: str, sequences: list[str], lig: str | None, masks: np.ndarray) -> dict[str, np.ndarray]:
  from aminx.host.runner import score
  from aminx.run.specs import ScoringSpecification

  out = score(
    ScoringSpecification(
      inputs=[_STRUCTURE], checkpoint_id=checkpoint, sequences_to_score=sequences, ar_masks=masks,
      designable_mask=_designable(), return_logits=True, backbone_noise=0.0, max_length=_MAX_LENGTH,
      ligand_context_path=lig, ligand_conditioning=bool(lig),
    ),
  )
  return {"logits": np.asarray(out["logits"])[0][..., :_L, :], "nll": np.asarray(out["scores"])[0]}


def _max_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], rows: int | None = None) -> float:
  sl = (slice(None), slice(0, rows)) if rows is not None else (slice(None),)
  return float(
    max(np.abs(a["logits"][sl] - b["logits"][sl]).max(), np.abs(a["nll"][sl] - b["nll"][sl]).max()),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)

  import jax

  from aminx.host import _sampling_helper
  from aminx.scoring.score import SCORE_BATCH_COMPOSITION_ATOL

  args.work_dir.mkdir(parents=True, exist_ok=True)
  masks = _masks(2)
  real_ladder = _sampling_helper.ligand_atom_rung
  lig20 = _ligand_npz(args.work_dir / "lig20.npz", 20)

  result: dict[str, Any] = {}
  n_deltas = {}
  for family, ckpt in _FAMILIES.items():
    lig = lig20 if family == "ligandmpnn" else None
    three = _score(ckpt, _mutants(3), lig, masks)
    four = _score(ckpt, [*_mutants(3), _mutants(3)[0]], lig, masks)
    n_deltas[family] = _max_delta(four, three, rows=3)
    result[f"n_pad_{family}"] = n_deltas[family]
    logger.info("N padding %s: max |delta| %.3g", family, n_deltas[family])

  ckpt = _FAMILIES["ligandmpnn"]
  m_deltas = {}
  for m in _ATOMS:
    lig = _ligand_npz(args.work_dir / f"lig{m}.npz", m)
    padded = _score(ckpt, _mutants(2), lig, masks)
    _sampling_helper.ligand_atom_rung = lambda n: n
    try:
      unpadded = _score(ckpt, _mutants(2), lig, masks)
      if m == 8:
        planted = _score(ckpt, _mutants(2), _ligand_npz(args.work_dir / "planted.npz", 8, n_masked=9), masks)
        repeat = _score(ckpt, _mutants(2), lig, masks)
        moved = _score(ckpt, _mutants(2), _ligand_npz(args.work_dir / "moved.npz", 8, seed=1), masks)
        result["planted_m8"] = _max_delta(planted, unpadded)
        result["repeat_delta"] = _max_delta(repeat, unpadded)
        result["positive_control_delta"] = _max_delta(moved, unpadded)
    finally:
      _sampling_helper.ligand_atom_rung = real_ladder
    m_deltas[m] = _max_delta(padded, unpadded)
    result[f"m_pad_{m}"] = m_deltas[m]
    logger.info("M padding M=%d: max |delta| %.3g", m, m_deltas[m])

  worst = max([*n_deltas.values(), *m_deltas.values(), result["planted_m8"]])
  device = jax.devices()[0]
  result.update(
    {
      "max_abs_delta": worst,
      "atol": float(SCORE_BATCH_COMPOSITION_ATOL),
      "backend": jax.default_backend(),
      "device_kind": str(getattr(device, "device_kind", device.platform)),
      "matmul_precision": str(jax.config.jax_default_matmul_precision),
      "git_hash": _git_hash(),
    },
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2))
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(result))
  logger.info("result: %s", json.dumps(result))


if __name__ == "__main__":
  main()
