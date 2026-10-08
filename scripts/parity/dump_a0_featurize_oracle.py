"""Dump the A0 featurizer oracle from upstream PottsMPNN.

Run in the torch oracle environment (this script does not import aminx)::

    python3 scripts/parity/dump_a0_featurize_oracle.py --potts-root <path> --out <dir>

Writes ``<out>/oracle.npz`` (every tied_featurize field, per fixture) and
``<out>/manifest.json`` (upstream commit and per-fixture sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ARRAY_FIELDS = (
  "x",
  "s",
  "present",
  "lengths",
  "chain_m",
  "chain_m_pos",
  "chain_encoding",
  "residue_idx",
  "omit_aa_mask",
  "dihedral_mask",
  "tied_beta",
  "pssm_coef",
  "pssm_bias",
  "pssm_log_odds",
  "bias_by_res",
)
JSON_FIELDS = (
  "letter_list",
  "visible_list",
  "masked_list",
  "masked_chain_lengths",
  "tied_pos",
  "chain_lens",
  "name",
)


def _atom(
  serial: int,
  atom: str,
  resname: str,
  chain: str,
  resseq: int,
  x: float,
  *,
  icode: str = " ",
  record: str = "ATOM  ",
) -> str:
  return (
    f"{record:<6.6}{serial:5d} {atom:>4.4} {resname:>3.3} {chain:1.1}"
    f"{resseq:4d}{icode:1.1}   {x:8.3f}{1.0:8.3f}{2.0:8.3f}"
  )


def _backbone(
  chain: str,
  resseq: int,
  resname: str,
  x0: float,
  *,
  icode: str = " ",
  record: str = "ATOM  ",
  serial: int = 1,
  atoms: tuple[str, ...] = ("N", "CA", "C", "O"),
) -> list[str]:
  return [
    _atom(serial + offset, atom, resname, chain, resseq, x0 + offset, icode=icode, record=record)
    for offset, atom in enumerate(atoms)
  ]


def _pdb(lines: list[str]) -> str:
  return "\n".join(lines) + "\n"


def _synthetic_pdbs() -> dict[str, str]:
  """Gapped fixtures the A0 gate compares at both skip_gaps settings."""
  gap_le48 = _pdb(
    [
      *_backbone("A", 1, "ALA", 1.0),
      *_backbone("A", 4, "GLY", 10.0, serial=5),
    ],
  )
  # PDB numbers 1 and 70 → L_total=70 (>48) with 2 present residues when gaps are kept.
  gap_gt48 = _pdb(
    [
      *_backbone("A", 1, "ALA", 1.0),
      *_backbone("A", 70, "VAL", 20.0, serial=5),
    ],
  )
  icode = _pdb(
    [
      *_backbone("A", 8, "ALA", 1.0, icode=" "),
      *_backbone("A", 8, "GLY", 10.0, icode="A", serial=5),
    ],
  )
  partial = _pdb(
    [
      *_backbone("A", 1, "ALA", 1.0, atoms=("N", "CA", "C")),
      *_backbone("A", 2, "SER", 10.0, serial=4),
    ],
  )
  return {
    "gap_le48": gap_le48,
    "gap_gt48": gap_gt48,
    "icode": icode,
    "partial": partial,
  }


def _upstream_commit(potts_root: Path) -> str:
  try:
    return subprocess.check_output(  # noqa: S603
      ["git", "-C", str(potts_root), "rev-parse", "HEAD"],  # noqa: S607
      text=True,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    pin = potts_root / "VENDOR_PIN.toml"
    for line in pin.read_text().splitlines():
      if line.startswith("commit = "):
        return line.split("=", 1)[1].strip().strip('"')
    msg = f"could not resolve an upstream commit under {potts_root}"
    raise SystemExit(msg) from None


def _featurize_one(potts: object, path: Path, *, skip_gaps: bool) -> dict[str, object]:
  device = torch.device("cpu")
  parsed = potts.parse_PDB(str(path), input_chain_list=None, ca_only=False, skip_gaps=skip_gaps)
  (
    x,
    s,
    mask,
    lengths,
    chain_m,
    chain_encoding,
    letter_list,
    visible_list,
    masked_list,
    masked_chain_lengths,
    chain_m_pos,
    omit_aa_mask,
    residue_idx,
    dihedral_mask,
    tied_pos,
    pssm_coef,
    pssm_bias,
    pssm_log_odds,
    bias_by_res,
    tied_beta,
    chain_lens,
  ) = potts.tied_featurize(parsed, device, None)

  def _np(value: object) -> np.ndarray:
    if torch.is_tensor(value):
      return value.detach().cpu().numpy()
    return np.asarray(value)

  x_np = _np(x)[0]
  return {
    "x": x_np,
    "s": _np(s)[0],
    "present": _np(mask)[0],
    "lengths": np.asarray(_np(lengths)[0], dtype=np.int32),
    "chain_m": _np(chain_m)[0],
    "chain_m_pos": _np(chain_m_pos)[0],
    "chain_encoding": _np(chain_encoding)[0],
    "residue_idx": _np(residue_idx)[0],
    "omit_aa_mask": _np(omit_aa_mask)[0],
    "dihedral_mask": _np(dihedral_mask)[0],
    "tied_beta": _np(tied_beta),
    "pssm_coef": _np(pssm_coef)[0],
    "pssm_bias": _np(pssm_bias)[0],
    "pssm_log_odds": _np(pssm_log_odds)[0],
    "bias_by_res": _np(bias_by_res)[0],
    "letter_list": list(letter_list[0]),
    "visible_list": list(visible_list[0]),
    "masked_list": list(masked_list[0]),
    "masked_chain_lengths": [int(v) for v in masked_chain_lengths[0]],
    "tied_pos": [[int(v) for v in group] for group in tied_pos[0]],
    "chain_lens": [int(v) for v in chain_lens],
    "name": str(parsed[0]["name"]),
    "L_total": int(x_np.shape[0]),
  }


def _collect_fixtures(potts_root: Path, work: Path) -> list[tuple[str, Path, bool]]:
  fixtures: list[tuple[str, Path, bool]] = []
  example_dir = potts_root / "inputs" / "example_pdbs"
  for stem in ("2yc3", "3dkm"):
    source = example_dir / f"{stem}.pdb"
    if not source.is_file():
      msg = f"missing upstream example pdb: {source}"
      raise SystemExit(msg)
    fixtures.extend((f"ex_{stem}_gaps{int(skip)}", source, skip) for skip in (False, True))
  synthetic = _synthetic_pdbs()
  for kind, text in synthetic.items():
    path = work / f"{kind}.pdb"
    path.write_text(text)
    fixtures.extend((f"{kind}_gaps{int(skip)}", path, skip) for skip in (False, True))
  return fixtures


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--potts-root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  potts_root = args.potts_root.resolve()
  sys.path.insert(0, str(potts_root))
  import potts_mpnn_utils as potts  # noqa: PLC0415

  out = args.out
  out.mkdir(parents=True, exist_ok=True)
  work = out / "_fixtures"
  work.mkdir(exist_ok=True)
  payload: dict[str, np.ndarray] = {}
  manifest_fixtures: list[dict[str, object]] = []
  names: list[str] = []
  for name, path, skip in _collect_fixtures(potts_root, work):
    pdb_bytes = path.read_bytes()
    digest = hashlib.sha256(pdb_bytes).hexdigest()
    fields = _featurize_one(potts, path, skip_gaps=skip)
    names.append(name)
    payload[f"{name}__pdb"] = np.asarray(pdb_bytes.decode("utf-8", "replace"))
    payload[f"{name}__skip_gaps"] = np.asarray(skip)
    for key in ARRAY_FIELDS:
      payload[f"{name}__{key}"] = np.asarray(fields[key])
    for key in JSON_FIELDS:
      payload[f"{name}__{key}"] = np.asarray(json.dumps(fields[key]))
    payload[f"{name}__L_total"] = np.asarray(fields["L_total"], dtype=np.int64)
    manifest_fixtures.append(
      {
        "name": name,
        "sha256": digest,
        "skip_gaps": skip,
        "L_total": fields["L_total"],
      },
    )
  manifest = {
    "upstream_commit": _upstream_commit(potts_root),
    "fixtures": manifest_fixtures,
  }
  payload["fixture_names"] = np.asarray(names)
  payload["manifest"] = np.asarray(json.dumps(manifest))
  np.savez(out / "oracle.npz", **payload)
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
