"""Dump ProtonPottsMPNN (v6) parity oracles from the pinned upstream checkout.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md`, P4. Run in the
`aminx-oracles-protonpotts` environment. Imports upstream `mpnn`, torch and numpy only --
**never aminx**, same rule as `scripts/parity/dump_potts_oracles.py`::

    uv run --no-sync python scripts/protonpotts/dump_protonpotts_oracles.py \\
        --checkpoint <ckpt> --pdb <pdb> --out <dir>

WHY THIS LIVES IN `scripts/protonpotts/` AND NOT `scripts/parity/`.
`scripts/parity/` is a *scoped* prefix (`tests/knob_gate/_coverage.py:21-28`), so a commit
there invalidates every ledger row globally. ProtonPotts is a new port with no ledger rows of
its own, so keeping it unscoped costs nothing and keeps it from entangling with the frozen
Potts/LASEr port. Revisit once the freeze lifts; §25.5 of the spec records the question.

WHAT THIS CAPTURES AND WHY EACH ONE.

  S            the featurized sequence. The only place protonation labels appear at all.
  etab_out     the Potts energy table. MEASURED INVARIANT TO S (§23.2) -- captured because it
               is real PottsMPNN structure parity, NOT because it validates protonation.
  E_idx        neighbour indices. Same status as etab_out.
  log_probs    the ONLY sequence-dependent observable here, so the only one that can detect a
               protonation-handling defect at all (§23.3, tightened by §26.4).

TWO TRAPS THIS SCRIPT EXISTS TO NOT FALL INTO, both measured, both in the spec.

1. `_get_potts_pipeline` memoises on (device, build_bond_labels, hbond_scope, extended_vocab,
   use_salt_bridge, deterministic, protonation_seed) and NOT on the annotation block (§23.4).
   Building a second cell without clearing `_PIPELINE_CACHE` silently reuses the first
   pipeline, so the two cells come out identical and every invariance check "passes"
   trivially. We clear it explicitly between cells.

2. A labelled cell that covers only some protonation tokens can pass a wave while the P/S/A
   discrimination is entirely broken (§26). We assert 9/9 coverage and refuse to write a dump
   otherwise, rather than recording a partial dump that reads as complete later.

THE LABELS ARE NOT CHEMISTRY. `CyclingProtonationLabels` assigns states by position in an
enumeration. The output is wrong as chemistry and predicts nothing. That is adequate for
parity and ONLY for parity: a parity wave compares two implementations on identical inputs, so
labels need to be valid tokens exercising the vocabulary, not correct ones. No number produced
from these dumps may be presented as a protonation-state result (§26.5).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("dump_protonpotts_oracles")

PREFIX_LEN = 21
V6_PROTONATION = (
  "HIS-P", "HIS-S", "HIS-A",
  "ASP-P", "ASP-D", "ASP-A",
  "GLU-P", "GLU-D", "GLU-A",
)
# Deterministic cycling, by ordinal within residue type. NOT chemistry -- see module docstring.
CYCLE = {
  "HIS": ("HIS-P", "HIS-S", "HIS-A"),
  "ASP": ("ASP-P", "ASP-D", "ASP-A"),
  "GLU": ("GLU-P", "GLU-D", "GLU-A"),
}
MIN_PER_TYPE = 3  # §26.3's cell-selection criterion
EXTENDED_VOCAB = "v6"


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _toml_str(value: str) -> str:
  return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _make_labeller():  # noqa: ANN202
  """Build the cycling labeller. Imported lazily so --help works without the oracle env."""
  from atomworks.ml.transforms.base import Transform

  class CyclingProtonationLabels(Transform):
    """Cycle protonation labels on the ordinal of each titratable residue within its type.

    Ordinal-within-type rather than `res_id % 3`: the latter's coverage depends on where
    residues happen to sit in the numbering, and was measured giving 7/9 on a structure whose
    every HIS landed on the same value mod 3 (§26.1).
    """

    counts: dict[str, int]

    def __init__(self) -> None:
      super().__init__()
      self.counts = {}

    def check_input(self, data: dict) -> None:
      if "atom_array" not in data:
        raise KeyError("needs atom_array")

    def forward(self, data: dict) -> dict:
      aa = data["atom_array"]
      seen: dict[tuple[str, tuple[object, int]], int] = {}
      order: dict[str, int] = {}
      labels = []
      for res_name, res_id, chain_id in zip(
        aa.res_name, aa.res_id, aa.chain_id, strict=True
      ):
        options = CYCLE.get(res_name)
        if options is None:
          labels.append("")
          continue
        key = (res_name, (chain_id, int(res_id)))
        if key not in seen:
          seen[key] = order.get(res_name, 0)
          order[res_name] = order.get(res_name, 0) + 1
        labels.append(options[seen[key] % 3])
      aa.set_annotation("protonation_label", np.array(labels, dtype="U8"))
      self.counts = {name: order.get(name, 0) for name in CYCLE}
      return data

  return CyclingProtonationLabels()


def _build_cell(checkpoint: Path, pdb: Path, *, labelled: bool):  # noqa: ANN202
  """Featurize and run one cell. Clears the pipeline cache first -- see trap 1."""
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference

  labeller = _make_labeller() if labelled else None
  pipelines.get_protonation_state_transforms = (
    (lambda **_: [labeller]) if labelled else (lambda **_: [])
  )
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1; no public API for this

  model = inference.load_model(str(checkpoint))
  batch = inference.prepare_potts_input(str(pdb), extended_vocab=EXTENDED_VOCAB)
  features = batch["network_input"]["input_features"]
  out = inference.run_forward(model, batch, num_sequences=1)
  counts = dict(labeller.counts) if labeller is not None else {}
  return features, out, counts


def _as_numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _covered_indices(seq: np.ndarray) -> list[int]:
  return sorted({int(v) for v in seq.ravel() if v >= PREFIX_LEN})


def _check_cell_is_adequate(counts: dict[str, int], covered: list[int]) -> None:
  """Refuse to dump a cell that cannot support a non-partial wave (§26.4)."""
  short = {name: n for name, n in counts.items() if n < MIN_PER_TYPE}
  if short:
    msg = (
      f"cell has too few titratable residues {short} (need >= {MIN_PER_TYPE} of each); "
      f"it cannot cover all nine v6 tokens. Pick another cell -- §26.3."
    )
    raise SystemExit(msg)
  expected = list(range(PREFIX_LEN, PREFIX_LEN + len(V6_PROTONATION)))
  missing = [i for i in expected if i not in covered]
  if missing:
    names = [V6_PROTONATION[i - PREFIX_LEN] for i in missing]
    msg = f"cell covers {len(covered)}/9 protonation tokens; missing {missing} = {names}"
    raise SystemExit(msg)


def _check_non_vacuous(plain_logprobs: np.ndarray, labelled_logprobs: np.ndarray) -> float:
  """Positive control, run before the dump is written, not asserted about afterwards.

  If log_probs does not move between the unlabelled and labelled cells, the labels are not
  reaching the model and every downstream comparison is vacuous -- in exactly the way that
  still looks like a clean pass.
  """
  delta = float(np.abs(plain_logprobs - labelled_logprobs).max())
  if delta == 0.0:
    raise SystemExit(
      "log_probs is identical between the unlabelled and labelled cells: the labels are not "
      "reaching the model, so these dumps would be vacuous. Refusing to write."
    )
  return delta


def _check_etab_invariant(plain_etab: np.ndarray, labelled_etab: np.ndarray) -> None:
  """NEGATIVE control, paired with the positive one above.

  `etab_out` was measured bit-identical across labelled and unlabelled cells (§23.2) -- a
  Potts model emits an energy function from the STRUCTURE encoder, and the sequence selects
  entries from it afterwards. A positive control that fires on its own proves only that
  something changed; pairing it with a quantity that must NOT change is what distinguishes
  "the labels reached the model" from "the two runs differ for some unrelated reason".

  If this fires, do not relax it. Either the architecture is not what §23.2 measured, or the
  two cells differ in some way beyond their labels.
  """
  if not np.array_equal(plain_etab, labelled_etab):
    delta = float(np.abs(plain_etab - labelled_etab).max())
    raise SystemExit(
      f"etab_out differs between the unlabelled and labelled cells (max|delta|={delta:.6g}), "
      "but §23.2 measured it invariant to S. The cells differ by more than their labels, or "
      "the architecture changed. Refusing to write."
    )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

  logger.info("unlabelled cell")
  plain_features, plain_out, _ = _build_cell(args.checkpoint, args.pdb, labelled=False)
  logger.info("labelled cell (cycling, ordinal-within-type)")
  lab_features, lab_out, counts = _build_cell(args.checkpoint, args.pdb, labelled=True)

  seq = _as_numpy(lab_features["S"])
  covered = _covered_indices(seq)
  logger.info("titratable counts %s; covered %d/9 %s", counts, len(covered), covered)
  _check_cell_is_adequate(counts, covered)

  delta = _check_non_vacuous(
    _as_numpy(plain_out.log_probs), _as_numpy(lab_out.log_probs)
  )
  logger.info("positive control: log_probs max|delta| labelled vs unlabelled = %.6g", delta)
  _check_etab_invariant(_as_numpy(plain_out.etab_out), _as_numpy(lab_out.etab_out))
  logger.info("negative control: etab_out bit-identical across the two cells, as §23.2 found")

  payload = {
    "S_unlabelled": _as_numpy(plain_features["S"]),
    "S_labelled": seq,
    "X": _as_numpy(lab_features["X"]),
    "X_m": _as_numpy(lab_features["X_m"]),
    "etab_out": _as_numpy(lab_out.etab_out),
    "E_idx": _as_numpy(lab_out.E_idx),
    "log_probs_unlabelled": _as_numpy(plain_out.log_probs),
    "log_probs_labelled": _as_numpy(lab_out.log_probs),
  }
  out_path = args.out / "protonpotts_v6" / "oracle_f32.npz"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **payload)
  sha = _sha256_file(out_path)

  manifest = args.out / "oracle_manifest.toml"
  manifest.write_text(
    "\n".join(
      [
        f"extended_vocab = {_toml_str(EXTENDED_VOCAB)}",
        f"torch_version = {_toml_str(torch.__version__)}",
        f"checkpoint = {_toml_str(str(args.checkpoint))}",
        f"pdb = {_toml_str(str(args.pdb))}",
        "labels_are_chemistry = false",
        f"labeller = {_toml_str('cycling, ordinal-within-type')}",
        f"titratable_counts = {{ {', '.join(f'{k} = {v}' for k, v in sorted(counts.items()))} }}",
        f"covered_protonation_indices = [{', '.join(str(i) for i in covered)}]",
        f"positive_control_logprobs_delta = {delta!r}",
        "",
        "[[dump]]",
        f"wave = {_toml_str('protonpotts_v6')}",
        f"precision = {_toml_str('f32')}",
        f"path = {_toml_str(str(out_path))}",
        f"sha256 = {_toml_str(sha)}",
        "",
      ]
    )
  )
  logger.info("wrote %s (sha256 %s)", out_path, sha)
  logger.info("wrote %s", manifest)
  return 0


if __name__ == "__main__":
  sys.exit(main())
