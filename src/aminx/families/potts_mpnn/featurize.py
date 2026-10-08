"""Host-numpy PottsMPNN featurizer.

Ports ``parse_PDB`` / ``parse_PDB_biounits`` (``potts_mpnn_utils.py:73-205``) and
``tied_featurize`` (``:293-512``). The parsing path is NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from jaxtyping import Bool, Float, Int

from aminx.families.potts_mpnn.alphabet import POTTS_MPNN

MODEL_ALPHABET = "".join(POTTS_MPNN.symbols)
X_INDEX = POTTS_MPNN.x_index

_ALPHA_1 = "ARNDCQEGHILKMFPSTWYV-"
_ALPHA_3 = [
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
  "GAP",
]
_AA_3_N = {name: index for index, name in enumerate(_ALPHA_3)}
_AA_N_1 = dict(enumerate(_ALPHA_1))
_BACKBONE = ("N", "CA", "C", "O")
_CHAIN_ALPHABET = [
  *"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
  *[str(item) for item in range(300)],
]

# Oracle / parity field names. Array fields are the numeric ``tied_featurize`` outputs.
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


class PottsInputError(ValueError):
  """Raised when a PottsMPNN input is not a ``.pdb`` file."""


@dataclass(frozen=True, slots=True, eq=False)
class PottsFeatures:
  """One structure featurized the way upstream ``tied_featurize`` packs a length-``L`` batch item.

  Arrays are unpadded (length ``L_total``, gap rows included). ``pad`` appends the host pad.
  """

  x: Float[np.ndarray, "L atom xyz"]
  s: Int[np.ndarray, " L"]
  present: Float[np.ndarray, " L"]
  lengths: Int[np.ndarray, " "]
  chain_m: Float[np.ndarray, " L"]
  chain_m_pos: Float[np.ndarray, " L"]
  chain_encoding: Int[np.ndarray, " L"]
  residue_idx: Int[np.ndarray, " L"]
  omit_aa_mask: Float[np.ndarray, "L alphabet"]
  dihedral_mask: Float[np.ndarray, "L 3"]
  tied_beta: Float[np.ndarray, " L"]
  pssm_coef: Float[np.ndarray, " L"]
  pssm_bias: Float[np.ndarray, "L vocab"]
  pssm_log_odds: Float[np.ndarray, "L vocab"]
  bias_by_res: Float[np.ndarray, "L vocab"]
  letter_list: tuple[str, ...]
  visible_list: tuple[str, ...]
  masked_list: tuple[str, ...]
  masked_chain_lengths: tuple[int, ...]
  tied_pos: tuple[tuple[int, ...], ...]
  chain_lens: tuple[int, ...]
  name: str
  L_total: int


def _model_alphabet(vocab: int) -> str:
  if vocab == 22:
    return f"{MODEL_ALPHABET}-"
  return MODEL_ALPHABET


def _parse_biounits(
  path: Path,
  atoms: Sequence[str],
  chain: str | None,
  *,
  skip_gaps: bool,
) -> tuple[np.ndarray, str] | tuple[str, str]:
  """Parse one chain. Returns ``('no_chain', 'no_chain')`` when the chain is absent."""
  xyz: dict[int, dict[str, dict[str, np.ndarray]]] = {}
  seq: dict[int, dict[str, str]] = {}
  min_resn = 10**6
  max_resn = -(10**6)
  with path.open("rb") as handle:
    for raw in handle:
      line = raw.decode("utf-8", "ignore").rstrip()
      if line[:6] == "HETATM" and line[17:20] == "MSE":
        line = line.replace("HETATM", "ATOM  ")
        line = line.replace("MSE", "MET")
      if line[:4] != "ATOM":
        continue
      if chain is not None and line[21:22] != chain:
        continue
      atom = line[12:16].strip()
      resname = line[17:20]
      resn_token = line[22:27].strip()
      if resn_token[-1].isalpha():
        icode, resn = resn_token[-1], int(resn_token[:-1]) - 1
      else:
        icode, resn = "", int(resn_token) - 1
      min_resn = min(min_resn, resn)
      max_resn = max(max_resn, resn)
      xyz.setdefault(resn, {}).setdefault(icode, {})
      seq.setdefault(resn, {}).setdefault(icode, resname)
      if atom not in xyz[resn][icode]:
        coords = [float(line[start : start + 8]) for start in (30, 38, 46)]
        xyz[resn][icode][atom] = np.asarray(coords, dtype=np.float64)

  if not xyz:
    return "no_chain", "no_chain"

  # The literal 20s below index UPSTREAM's three-letter parse alphabet (_ALPHA_3),
  # where 20 is "GAP". They are NOT the model alphabet's X (POTTS_MPNN.x_index).
  # Do not "fix" them to x_index; they are part of a faithful port of the parser.
  seq_idx: list[int] = []
  xyz_rows: list[np.ndarray] = []
  for resn in range(min_resn, max_resn + 1):
    if resn in seq:
      for icode in sorted(seq[resn]):
        seq_idx.append(_AA_3_N.get(seq[resn][icode], 20))
        for atom in atoms:
          coords = xyz[resn][icode].get(atom)
          xyz_rows.append(np.full(3, np.nan) if coords is None else coords)
    elif skip_gaps:
      continue
    else:
      seq_idx.append(20)
      xyz_rows.extend(np.full(3, np.nan) for _atom in atoms)
  coords_out = np.asarray(xyz_rows, dtype=np.float64).reshape(-1, len(atoms), 3)
  sequence = "".join(_AA_N_1.get(index, "-") for index in seq_idx)
  return coords_out, sequence


def parse_pdb_upstream(
  path: str | Path,
  chains: Sequence[str] | None = None,
  skip_gaps: bool = False,  # noqa: FBT001, FBT002
  *,
  ca_only: bool = False,
) -> list[dict[str, Any]]:
  """Port of ``parse_PDB`` / ``parse_PDB_biounits``.

  Absent chains are dropped. Chain order is ``chains`` when given, otherwise the upstream
  alphabet (A-Z, a-z, ``"0"`` .. ``"299"``). Non-``.pdb`` paths raise ``PottsInputError``.
  """
  pdb_path = Path(path)
  if pdb_path.suffix.lower() != ".pdb":
    raise PottsInputError("pottsmpnn_requires_pdb")
  chain_alphabet = list(chains) if chains else list(_CHAIN_ALPHABET)
  atoms = ("CA",) if ca_only else _BACKBONE
  parsed: dict[str, Any] = {}
  concat_seq = ""
  chain_order: list[str] = []
  for letter in chain_alphabet:
    xyz, seq = _parse_biounits(pdb_path, atoms, letter, skip_gaps=skip_gaps)
    if isinstance(xyz, str):
      continue
    concat_seq += seq
    parsed[f"seq_chain_{letter}"] = seq
    coord_map: dict[str, list[list[float]]] = {}
    if ca_only:
      coord_map[f"CA_chain_{letter}"] = xyz.reshape(-1, 3).tolist()
    else:
      for index, atom in enumerate(_BACKBONE):
        coord_map[f"{atom}_chain_{letter}"] = xyz[:, index, :].tolist()
    parsed[f"coords_chain_{letter}"] = coord_map
    chain_order.append(letter)
  text = pdb_path.as_posix()
  parsed["name"] = text[text.rfind("/") + 1 : -4]
  parsed["num_of_chains"] = len(chain_order)
  parsed["seq"] = concat_seq
  parsed["chain_order"] = chain_order
  return [parsed]


def _dash_to_x(sequence: str) -> str:
  return "".join("X" if aa == "-" else aa for aa in sequence)


def _chain_coordinates(
  pdb: Mapping[str, Any],
  letter: str,
  *,
  ca_only: bool,
) -> tuple[np.ndarray, str]:
  sequence = _dash_to_x(cast("str", pdb[f"seq_chain_{letter}"]))
  coords = cast("dict[str, Any]", pdb[f"coords_chain_{letter}"])
  if ca_only:
    x_chain = np.asarray(coords[f"CA_chain_{letter}"], dtype=np.float64)
    if x_chain.ndim == 2:
      x_chain = x_chain[:, None, :]
    return x_chain, sequence
  stacked = np.stack(
    [np.asarray(coords[f"{atom}_chain_{letter}"], dtype=np.float64) for atom in _BACKBONE],
    axis=1,
  )
  return stacked, sequence


def _omit_mask(
  items: Sequence[Any],
  chain_length: int,
  alphabet: str,
) -> np.ndarray:
  mask = np.zeros((chain_length, len(alphabet)), dtype=np.float64)
  letters = np.array(list(alphabet))
  for item in items:
    idx_aa = np.asarray(item[0], dtype=np.int64) - 1
    aa_idx = np.array(
      [int(np.argwhere(letters == aa)[0][0]) for aa in item[1]],
      dtype=np.int64,
    ).repeat(idx_aa.shape[0])
    pairs = np.array([[row, col] for row in idx_aa for col in aa_idx], dtype=np.int64)
    mask[pairs[:, 0], pairs[:, 1]] = 1.0
  return mask


def _tied_groups(
  tied_positions: Sequence[Mapping[str, Any]] | None,
  letter_list: Sequence[str],
  global_idx_start: Sequence[int],
  l_total: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
  tied_beta = np.ones(l_total, dtype=np.float64)
  if not tied_positions:
    return (), tied_beta
  letters = np.array(list(letter_list))
  groups: list[tuple[int, ...]] = []
  for tied_item in tied_positions:
    one: list[int] = []
    for chain_key, raw in tied_item.items():
      start = int(global_idx_start[int(np.argwhere(letters == chain_key)[0][0])])
      if isinstance(raw[0], list):
        positions = cast("list[int]", raw[0])
        betas = cast("list[float]", raw[1])
        for index, pos in enumerate(positions):
          row = start + pos - 1
          one.append(row)
          tied_beta[row] = float(betas[index])
      else:
        one.extend(start + pos - 1 for pos in cast("list[int]", raw))
    groups.append(tuple(one))
  return tuple(groups), tied_beta


def _dihedral_mask(residue_idx: np.ndarray) -> np.ndarray:
  if residue_idx.shape[0] == 0:
    return np.zeros((0, 3), dtype=np.float64)
  jumps = ((residue_idx[1:] - residue_idx[:-1]) == 1).astype(np.float64)
  phi = np.pad(jumps, (1, 0))
  psi = np.pad(jumps, (0, 1))
  omega = np.pad(jumps, (0, 1))
  return np.stack([phi, psi, omega], axis=-1)


def _default_designed_chains(pdb: Mapping[str, Any]) -> list[str]:
  # Upstream keeps only the last character of ``seq_chain_*`` keys (``:325``).
  return [key[-1:] for key in pdb if key[:10] == "seq_chain_"]


def _resolve_chain_order(
  batch: Sequence[Mapping[str, Any]],
  chain_dict: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None,
) -> tuple[list[str], list[str]]:
  """Last batch item wins, matching the two-loop ``all_chains`` assignment in upstream."""
  masked: list[str] = []
  visible: list[str] = []
  for pdb in batch:
    name = cast("str", pdb["name"])
    if chain_dict is not None and len(chain_dict[name][0]) > 0:
      masked = list(chain_dict[name][0])
      visible = list(chain_dict[name][1])
    else:
      masked = _default_designed_chains(pdb)
      visible = []
    masked.sort()
    visible.sort()
  return masked, visible


def tied_featurize_port(  # noqa: PLR0915
  batch: Sequence[Mapping[str, Any]],
  chain_dict: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None,
  fixed_position_dict: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
  omit_aa_dict: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
  tied_positions_dict: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
  pssm_dict: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
  bias_by_res_dict: Mapping[str, Mapping[str, Any]] | None = None,
  ca_only: bool = False,  # noqa: FBT001, FBT002
  vocab: int = 21,
) -> tuple[PottsFeatures, ...]:
  """Port of ``tied_featurize`` for a numpy host (no ``device`` tensor move).

  Chain order is sorted designed chains then sorted fixed chains. Returns one
  ``PottsFeatures`` per batch element, cropped to that element's ``L_total``.
  ``tied_beta`` and ``chain_lens`` are per element (upstream returns only the last
  element's copies, which agrees for the single-structure batches A0 featurizes).
  """
  if vocab <= 0:
    msg = f"vocab must be positive, got {vocab}"
    raise ValueError(msg)
  alphabet = _model_alphabet(vocab)
  masked_chains, visible_chains = _resolve_chain_order(batch, chain_dict)
  all_chains = masked_chains + visible_chains
  features: list[PottsFeatures] = []
  for pdb in batch:
    name = cast("str", pdb["name"])
    x_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    seq_parts: list[str] = []
    encoding_parts: list[np.ndarray] = []
    residue_parts: list[np.ndarray] = []
    fixed_parts: list[np.ndarray] = []
    omit_parts: list[np.ndarray] = []
    pssm_coef_parts: list[np.ndarray] = []
    pssm_bias_parts: list[np.ndarray] = []
    pssm_log_odds_parts: list[np.ndarray] = []
    bias_parts: list[np.ndarray] = []
    letter_list: list[str] = []
    visible_list: list[str] = []
    masked_list: list[str] = []
    masked_chain_lengths: list[int] = []
    chain_lens: list[int] = []
    global_idx_start = [0]
    chain_number = 1
    row_cursor = 0
    for letter in all_chains:
      x_chain, chain_seq = _chain_coordinates(pdb, letter, ca_only=ca_only)
      chain_length = len(chain_seq)
      letter_list.append(letter)
      global_idx_start.append(global_idx_start[-1] + chain_length)
      designed = letter in masked_chains
      if designed:
        masked_list.append(letter)
        masked_chain_lengths.append(chain_length)
        chain_mask = np.ones(chain_length, dtype=np.float64)
      else:
        visible_list.append(letter)
        chain_mask = np.zeros(chain_length, dtype=np.float64)
      fixed_mask = np.ones(chain_length, dtype=np.float64)
      if designed and fixed_position_dict is not None:
        fixed_pos = fixed_position_dict[name][letter]
        if fixed_pos:
          fixed_mask[np.asarray(fixed_pos, dtype=np.int64) - 1] = 0.0
      if designed and omit_aa_dict is not None:
        omit = _omit_mask(omit_aa_dict[name][letter], chain_length, alphabet)
      else:
        omit = np.zeros((chain_length, len(alphabet)), dtype=np.float64)
      if designed and pssm_dict and pssm_dict[name][letter]:
        pssm = pssm_dict[name][letter]
        pssm_coef = np.asarray(pssm["pssm_coef"], dtype=np.float64)
        pssm_bias = np.asarray(pssm["pssm_bias"], dtype=np.float64)
        pssm_log_odds = np.asarray(pssm["pssm_log_odds"], dtype=np.float64)
      else:
        pssm_coef = np.zeros(chain_length, dtype=np.float64)
        pssm_bias = np.zeros((chain_length, vocab), dtype=np.float64)
        pssm_log_odds = np.full((chain_length, vocab), 10000.0, dtype=np.float64)
      if designed and bias_by_res_dict:
        bias = np.asarray(bias_by_res_dict[name][letter], dtype=np.float64)
      else:
        bias = np.zeros((chain_length, vocab), dtype=np.float64)
      residue_parts.append(
        100 * (chain_number - 1) + np.arange(row_cursor, row_cursor + chain_length),
      )
      row_cursor += chain_length
      chain_number += 1
      x_parts.append(x_chain)
      mask_parts.append(chain_mask)
      seq_parts.append(chain_seq)
      encoding_parts.append(np.full(chain_length, chain_number - 1, dtype=np.int64))
      fixed_parts.append(fixed_mask)
      omit_parts.append(omit)
      pssm_coef_parts.append(pssm_coef)
      pssm_bias_parts.append(pssm_bias)
      pssm_log_odds_parts.append(pssm_log_odds)
      bias_parts.append(bias)
      chain_lens.append(chain_length)

    sequence = "".join(seq_parts)
    l_total = len(sequence)
    x_cat = np.concatenate(x_parts, axis=0) if x_parts else np.zeros((0, 1 if ca_only else 4, 3))
    present = np.isfinite(np.sum(x_cat, axis=(1, 2))).astype(np.float32)
    x_out = np.where(np.isnan(x_cat), 0.0, x_cat).astype(np.float32)
    s = np.asarray([alphabet.index(aa) for aa in sequence], dtype=np.int64)
    residue_idx = (
      np.concatenate(residue_parts).astype(np.int64)
      if residue_parts
      else np.zeros(0, dtype=np.int64)
    )
    if tied_positions_dict is None:
      tied_pos, tied_beta = (), np.ones(l_total, dtype=np.float64)
    else:
      tied_pos, tied_beta = _tied_groups(
        tied_positions_dict[name],
        letter_list,
        global_idx_start,
        l_total,
      )
    features.append(
      PottsFeatures(
        x=x_out,
        s=s,
        present=present,
        lengths=np.asarray(l_total, dtype=np.int32),
        chain_m=np.concatenate(mask_parts).astype(np.float32)
        if mask_parts
        else np.zeros(0, np.float32),
        chain_m_pos=np.concatenate(fixed_parts).astype(np.float32)
        if fixed_parts
        else np.zeros(0, np.float32),
        chain_encoding=(
          np.concatenate(encoding_parts).astype(np.int64)
          if encoding_parts
          else np.zeros(0, np.int64)
        ),
        residue_idx=residue_idx,
        omit_aa_mask=(
          np.concatenate(omit_parts).astype(np.float32)
          if omit_parts
          else np.zeros((0, len(alphabet)), np.float32)
        ),
        dihedral_mask=_dihedral_mask(residue_idx).astype(np.float32),
        tied_beta=tied_beta.astype(np.float32),
        pssm_coef=np.concatenate(pssm_coef_parts).astype(np.float32)
        if pssm_coef_parts
        else np.zeros(0, np.float32),
        pssm_bias=(
          np.concatenate(pssm_bias_parts).astype(np.float32)
          if pssm_bias_parts
          else np.zeros((0, vocab), np.float32)
        ),
        pssm_log_odds=(
          np.concatenate(pssm_log_odds_parts).astype(np.float32)
          if pssm_log_odds_parts
          else np.zeros((0, vocab), np.float32)
        ),
        bias_by_res=(
          np.concatenate(bias_parts).astype(np.float32)
          if bias_parts
          else np.zeros((0, vocab), np.float32)
        ),
        letter_list=tuple(letter_list),
        visible_list=tuple(visible_list),
        masked_list=tuple(masked_list),
        masked_chain_lengths=tuple(masked_chain_lengths),
        tied_pos=tied_pos,
        chain_lens=tuple(chain_lens),
        name=name,
        L_total=l_total,
      ),
    )
  return tuple(features)


def pad(
  features: PottsFeatures,
  l_pad: int,
) -> tuple[PottsFeatures, Bool[np.ndarray, " L_pad"]]:
  """Pad to ``l_pad``. Pad rows are X=0, S=X, present=0. ``pad_valid`` marks real rows."""
  if l_pad < features.L_total:
    msg = f"l_pad ({l_pad}) is shorter than L_total ({features.L_total})"
    raise ValueError(msg)
  n_pad = l_pad - features.L_total

  def _tail(row: np.ndarray, fill: float) -> np.ndarray:
    if n_pad == 0:
      return row
    extra_shape = (n_pad, *row.shape[1:])
    return np.concatenate([row, np.full(extra_shape, fill, dtype=row.dtype)])

  padded = PottsFeatures(
    x=_tail(features.x, 0.0),
    s=_tail(features.s, X_INDEX),
    present=_tail(features.present, 0.0),
    lengths=features.lengths,
    chain_m=_tail(features.chain_m, 0.0),
    chain_m_pos=_tail(features.chain_m_pos, 0.0),
    chain_encoding=_tail(features.chain_encoding, 0),
    residue_idx=_tail(features.residue_idx, -100),
    omit_aa_mask=_tail(features.omit_aa_mask, 0.0),
    dihedral_mask=_tail(features.dihedral_mask, 0.0),
    tied_beta=_tail(features.tied_beta, 1.0),
    pssm_coef=_tail(features.pssm_coef, 0.0),
    pssm_bias=_tail(features.pssm_bias, 0.0),
    pssm_log_odds=_tail(features.pssm_log_odds, 0.0),
    bias_by_res=_tail(features.bias_by_res, 0.0),
    letter_list=features.letter_list,
    visible_list=features.visible_list,
    masked_list=features.masked_list,
    masked_chain_lengths=features.masked_chain_lengths,
    tied_pos=features.tied_pos,
    chain_lens=features.chain_lens,
    name=features.name,
    L_total=features.L_total,
  )
  pad_valid = np.arange(l_pad) < features.L_total
  return padded, pad_valid


def knn_boundary_tie(present: np.ndarray, l_total: int, k: int = 48) -> bool:
  """True when ``#present <= K_eff < L_total`` with ``K_eff = min(k, l_total)``.

  Those structures are the §6.5b ``knn_boundary_tie`` case: upstream ``topk`` breaks the
  row-``D_max`` tie arbitrarily, so the neighbour set is not reproducible.
  """
  if l_total < 0:
    msg = f"l_total must be non-negative, got {l_total}"
    raise ValueError(msg)
  head = np.asarray(present)[:l_total]
  n_present = int(np.sum(head > 0))
  k_eff = min(k, l_total)
  return n_present <= k_eff < l_total
