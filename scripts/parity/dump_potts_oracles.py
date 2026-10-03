"""Dump PottsMPNN parity oracles (f64 and f32) from the pinned upstream checkout.

Run in the torch oracle environment. This script imports upstream PottsMPNN, torch,
and numpy only (never aminx)::

    python3 scripts/parity/dump_potts_oracles.py \\
        --potts-root <path> --fixtures-dir <dir> --out <dir> \\
        [--checkpoints vanilla_20 ...] [--precisions f64 f32]

``--selfcheck`` asserts the draw shim is a no-op when its context manager is off
(bit-identical decoder tokens). Distributional shim-vs-unshimmed sampling is a
later sidecar and is not run here.

``--only order`` writes ``<out>/potts_order/order_f64.npz`` and appends new sha
entries (the out-dir manifest, and one new line in ``oracles.sha256``). It does
not regenerate any existing npz or rewrite any existing sha line. Orders are
integers, so the f64 dump is the whole batch.

Writes ``<out>/<wave>/oracle_<prec>.npz`` for each §7.2 Potts wave, ``<out>/declayer_f64.npz``,
and ``<out>/oracle_manifest.toml``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from oracle_shims.potts import (
  SHIM_SITES,
  DrawCursor,
  injected_tied_randn,
  injected_uniform_draws,
  inverse_cdf_index,
  rbf_follows_input_dtype,
  shim_sha256,
  untied_decoding_order,
  weight_dtype_positional_encodings,
)

logger = logging.getLogger("dump_potts_oracles")

SEED = 0
VOCAB = 21
REFINE_OMIT_INDEX = 20  # X in the ProteinMPNN alphabet
N_RANDOM = 200
AR_TEMPERATURE = 0.1
REFINE_TEMPERATURE = 0.5
PSSM_MULTI = 0.5
BINDING_CUTOFF = 8.0
CONVERGE_MAX_L = 16
N_SWEEPS = 1000

WAVES: tuple[str, ...] = (
  "potts_head",
  "potts_merge_pair_d2",
  "potts_merge_pair_d4",
  "potts_energy",
  "potts_ar_decode",
  "potts_refine",
  "pottsmpnn_full",
)

CHECKPOINTS: dict[str, str] = {
  "vanilla_20": "vanilla_model_weights/pottsmpnn_20.pt",
  "vanilla_30": "vanilla_model_weights/pottsmpnn_30.pt",
  "soluble_20": "soluble_model_weights/sol_pottsmpnn_20.pt",
  "soluble_30": "soluble_model_weights/sol_pottsmpnn_30.pt",
  "ft": "ft_model_weights/potts_ft.pt",
}

A0_PDBS: tuple[str, ...] = ("gap_le48", "gap_gt48", "icode", "partial")
EXAMPLE_PDBS: tuple[str, ...] = ("2yc3", "3dkm")

SHIMMED_WAVES = frozenset({"potts_ar_decode", "potts_refine"})


@dataclass
class Fixture:
  """One featurization of a PDB."""

  name: str
  path: Path
  skip_gaps: bool
  tied_positions: list[dict[str, list[int]]] | None = None
  fixed_positions: dict[str, list[int]] | None = None
  run_tied_decode: bool = False
  run_tied_refine: bool = False
  run_pssm: bool = False
  run_binding: bool = False
  binding_partitions: list[list[str]] | None = None


@dataclass
class Feat:
  """Tensors returned by upstream ``tied_featurize`` (batch axis kept)."""

  parsed: list[dict[str, object]]
  x: torch.Tensor
  s: torch.Tensor
  mask: torch.Tensor
  chain_m: torch.Tensor
  chain_m_pos: torch.Tensor
  chain_encoding: torch.Tensor
  residue_idx: torch.Tensor
  omit_aa_mask: torch.Tensor
  bias_by_res: torch.Tensor
  pssm_coef: torch.Tensor
  pssm_bias: torch.Tensor
  pssm_log_odds: torch.Tensor
  tied_beta: torch.Tensor
  tied_pos: list[list[int]]
  chain_order: list[str]
  seq: str


@dataclass
class Encoded:
  """Encoder outputs plus the three etab conventions."""

  h_v: torch.Tensor
  h_e: torch.Tensor
  e_idx: torch.Tensor
  etab_flat: torch.Tensor
  etab_raw: torch.Tensor
  etab_forward: torch.Tensor
  etab_energy: torch.Tensor
  enc_h_v: list[torch.Tensor]
  enc_h_e: list[torch.Tensor]


@dataclass
class Buckets:
  """Per-wave numpy payloads for one precision."""

  precision: str
  waves: dict[str, dict[str, np.ndarray]] = field(
    default_factory=lambda: {wave: {} for wave in WAVES},
  )
  declayer: dict[str, np.ndarray] | None = None


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _np(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _toml_str(value: str) -> str:
  return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_list(values: Iterable[str]) -> str:
  return "[" + ", ".join(_toml_str(item) for item in values) + "]"


def _load_pin(potts_root: Path) -> tuple[str, dict[str, str]]:
  pin_path = potts_root / "VENDOR_PIN.toml"
  commit = ""
  weights: dict[str, str] = {}
  in_weights = False
  for line in pin_path.read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("commit") and "=" in stripped and not in_weights:
      commit = stripped.split("=", 1)[1].strip().strip('"')
    if stripped == "[weights_sha256]":
      in_weights = True
      continue
    if in_weights:
      if stripped.startswith("["):
        break
      if "=" in stripped:
        key, val = stripped.split("=", 1)
        weights[key.strip().strip('"')] = val.strip().strip('"')
  if not commit or not weights:
    msg = f"could not parse commit and weights_sha256 from {pin_path}"
    raise SystemExit(msg)
  return commit, weights


def _upstream_commit(potts_root: Path, pin_commit: str) -> str:
  try:
    head = subprocess.check_output(  # noqa: S603
      ["git", "-C", str(potts_root), "rev-parse", "HEAD"],  # noqa: S607
      text=True,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    logger.info("no git HEAD under %s; manifest commit is the VENDOR_PIN", potts_root)
    return pin_commit
  if head != pin_commit:
    msg = f"PottsMPNN HEAD {head} != VENDOR_PIN commit {pin_commit}"
    raise SystemExit(msg)
  return pin_commit


def _atom(
  serial: int,
  atom: str,
  resname: str,
  chain: str,
  resseq: int,
  x: float,
  y: float,
  z: float,
) -> str:
  return (
    f"ATOM  {serial:5d} {atom:>4.4} {resname:>3.3} {chain:1.1}"
    f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
  )


def _backbone(
  chain: str,
  resseq: int,
  resname: str,
  x: float,
  y: float,
  *,
  serial: int,
) -> list[str]:
  atoms = ("N", "CA", "C", "O")
  return [
    _atom(serial + offset, atom, resname, chain, resseq, x + offset, y, 0.0)
    for offset, atom in enumerate(atoms)
  ]


def _write_generated(work: Path) -> dict[str, Path]:
  """Small PDBs the A0 set does not contain: a tied mix, and a 2-chain interface."""
  work.mkdir(parents=True, exist_ok=True)
  tied_rows: list[str] = []
  serial = 1
  # Resnums 1,2,4,5,6,7 leave a gap row at chain position 3 (mask == 0).
  for resseq, resname, x in (
    (1, "ALA", 0.0),
    (2, "SER", 4.0),
    (4, "GLY", 8.0),
    (5, "VAL", 12.0),
    (6, "LEU", 16.0),
    (7, "THR", 20.0),
  ):
    tied_rows.extend(_backbone("A", resseq, resname, x, 0.0, serial=serial))
    serial += 4
  two_rows: list[str] = []
  serial = 1
  for chain, y in (("A", 0.0), ("B", 4.0)):
    for index, resname in enumerate(("ALA", "GLY", "VAL"), start=1):
      two_rows.extend(_backbone(chain, index, resname, float(index) * 3.8, y, serial=serial))
      serial += 4
  paths = {
    "tied_mix": work / "tied_mix.pdb",
    "two_chain": work / "two_chain.pdb",
  }
  paths["tied_mix"].write_text("\n".join(tied_rows) + "\n")
  paths["two_chain"].write_text("\n".join(two_rows) + "\n")
  return paths


def _collect_fixtures(potts_root: Path, fixtures_dir: Path, work: Path) -> list[Fixture]:
  fixtures: list[Fixture] = []
  generated = _write_generated(work)
  sources: list[tuple[str, Path]] = [(name, fixtures_dir / f"{name}.pdb") for name in A0_PDBS]
  example_dir = potts_root / "inputs" / "example_pdbs"
  sources.extend((name, example_dir / f"{name}.pdb") for name in EXAMPLE_PDBS)
  sources.append(("tied_mix", generated["tied_mix"]))
  sources.append(("two_chain", generated["two_chain"]))
  for _name, path in sources:
    if not path.is_file():
      msg = f"missing fixture pdb: {path}"
      raise SystemExit(msg)
  for name, path in sources:
    if name in {"tied_mix", "two_chain"}:
      continue
    fixtures.extend(
      Fixture(name=f"{name}_gaps{int(skip)}", path=path, skip_gaps=skip) for skip in (False, True)
    )
  fixtures.append(
    Fixture(
      name="tied_mix",
      path=generated["tied_mix"],
      skip_gaps=False,
      # 1-based chain positions. [1, 2] = designed then fixed. [4, 3, 5] = present, gap, present.
      tied_positions=[{"A": [1, 2]}, {"A": [4, 3, 5]}],
      fixed_positions={"A": [2]},
      run_tied_decode=True,
      run_tied_refine=True,
    ),
  )
  fixtures.append(
    Fixture(
      name="gap_le48_pssm",
      path=fixtures_dir / "gap_le48.pdb",
      skip_gaps=False,
      run_pssm=True,
    ),
  )
  fixtures.append(
    Fixture(
      name="two_chain",
      path=generated["two_chain"],
      skip_gaps=False,
      run_binding=True,
      binding_partitions=[["A"], ["B"]],
    ),
  )
  # Mark the short gapped fixture (and every other small one is detected by length).
  for fixture in fixtures:
    if fixture.name == "gap_le48_gaps0":
      fixture.run_pssm = False
  return fixtures


def _featurize(potts: object, fixture: Fixture) -> Feat:
  device = torch.device("cpu")
  parsed = potts.parse_PDB(  # type: ignore[attr-defined]
    str(fixture.path),
    input_chain_list=None,
    ca_only=False,
    skip_gaps=fixture.skip_gaps,
  )
  name = str(parsed[0]["name"])
  fixed = None if fixture.fixed_positions is None else {name: fixture.fixed_positions}
  tied = None if fixture.tied_positions is None else {name: fixture.tied_positions}
  (
    x,
    s,
    mask,
    _lengths,
    chain_m,
    chain_encoding,
    _letters,
    _visible,
    _masked,
    _masked_lengths,
    chain_m_pos,
    omit_aa_mask,
    residue_idx,
    _dihedral,
    tied_pos_batch,
    pssm_coef,
    pssm_bias,
    pssm_log_odds,
    bias_by_res,
    tied_beta,
    _chain_lens,
  ) = potts.tied_featurize(  # type: ignore[attr-defined]
    [parsed[0]],
    device,
    None,
    fixed,
    None,
    tied,
    None,
    None,
    ca_only=False,
    vocab=VOCAB,
  )
  import etab_utils  # noqa: PLC0415

  seq = "".join(etab_utils.ints_to_seq(s[0].detach().cpu().tolist()))
  tied_pos = [[int(v) for v in group] for group in tied_pos_batch[0]]
  return Feat(
    parsed=parsed,
    x=x,
    s=s,
    mask=mask,
    chain_m=chain_m,
    chain_m_pos=chain_m_pos,
    chain_encoding=chain_encoding,
    residue_idx=residue_idx,
    omit_aa_mask=omit_aa_mask,
    bias_by_res=bias_by_res,
    pssm_coef=pssm_coef,
    pssm_bias=pssm_bias,
    pssm_log_odds=pssm_log_odds,
    tied_beta=tied_beta,
    tied_pos=tied_pos,
    chain_order=[str(c) for c in parsed[0]["chain_order"]],
    seq=seq,
  )


def _cast_pair(
  x: torch.Tensor,
  mask: torch.Tensor,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  return x.to(dtype=dtype), mask.to(dtype=dtype)


def _load_model(
  potts: object,
  checkpoint_path: Path,
  *,
  precision: str,
) -> tuple[torch.nn.Module, list[str], list[str]]:
  torch.manual_seed(SEED)
  model = potts.PottsMPNN(  # type: ignore[attr-defined]
    ca_only=False,
    num_letters=VOCAB,
    vocab=VOCAB,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    potts_dim=400,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    augment_eps=0.0,
  )
  blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  incompat = model.load_state_dict(blob["model_state_dict"], strict=False)
  missing = list(incompat.missing_keys)
  unexpected = list(incompat.unexpected_keys)
  if missing:
    msg = f"missing keys loading {checkpoint_path}: {missing}"
    raise SystemExit(msg)
  if precision == "f64":
    model.double()
  elif precision != "f32":
    msg = f"unknown precision {precision}"
    raise SystemExit(msg)
  model.eval()
  for param in model.parameters():
    param.requires_grad_(requires_grad=False)
  for module in model.modules():
    if module.__class__.__name__ in {"Dropout", "_VDropout"} and module.training:
      msg = f"{module.__class__.__name__} is in training mode"
      raise SystemExit(msg)
  dtype = next(model.parameters()).dtype
  original = model.run_encoder

  def run_encoder(
    x: torch.Tensor,
    mask: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding: torch.Tensor,
  ) -> object:
    return original(x.to(dtype=dtype), mask.to(dtype=dtype), residue_idx, chain_encoding)

  model.run_encoder = run_encoder  # type: ignore[method-assign]
  return model, missing, unexpected


def _encode(potts: object, model: torch.nn.Module, feat: Feat) -> Encoded:
  import etab_utils  # noqa: PLC0415

  enc_h_v: list[torch.Tensor] = []
  enc_h_e: list[torch.Tensor] = []

  def hook(
    _module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor],
  ) -> None:
    enc_h_v.append(inputs[0].detach())
    enc_h_e.append(inputs[1].detach())
    enc_h_v.append(output[0].detach())
    enc_h_e.append(output[1].detach())

  handles = [layer.register_forward_hook(hook) for layer in model.encoder_layers]
  captured: list[torch.Tensor] = []
  real_merge = potts.merge_duplicate_pairE  # type: ignore[attr-defined]

  def wrapped(etab: torch.Tensor, e_idx: torch.Tensor, denom: int = 2) -> torch.Tensor:
    captured.append(etab.detach().clone())
    return real_merge(etab, e_idx, denom=denom)

  potts.merge_duplicate_pairE = wrapped  # type: ignore[attr-defined]
  try:
    h_v, e_idx, h_e, etab_flat = model.run_encoder(
      feat.x,
      feat.mask,
      feat.residue_idx,
      feat.chain_encoding,
    )
  finally:
    potts.merge_duplicate_pairE = real_merge  # type: ignore[attr-defined]
    for handle in handles:
      handle.remove()
  if len(captured) != 1:
    msg = f"expected one pre-merge etab capture, got {len(captured)}"
    raise RuntimeError(msg)
  etab_raw = captured[0]
  etab_forward = etab_flat.view(*etab_flat.shape[:3], 20, 20).contiguous()
  etab_energy = etab_utils.functionalize_etab(etab_flat.clone(), e_idx)
  # Hooks append (input, output) per layer, so even slots are inputs. Keep input of
  # layer 0 and the output of every layer.
  kept_v = [enc_h_v[0], *enc_h_v[1::2]]
  kept_e = [enc_h_e[0], *enc_h_e[1::2]]
  return Encoded(
    h_v=h_v,
    h_e=h_e,
    e_idx=e_idx,
    etab_flat=etab_flat,
    etab_raw=etab_raw,
    etab_forward=etab_forward,
    etab_energy=etab_energy,
    enc_h_v=kept_v,
    enc_h_e=kept_e,
  )


def _put(
  buckets: Buckets,
  waves: Iterable[str],
  key: str,
  value: torch.Tensor | np.ndarray,
) -> None:
  array = value if isinstance(value, np.ndarray) else _np(value)
  for wave in waves:
    buckets.waves[wave][key] = array


def _order_from_randn(score: torch.Tensor, randn: torch.Tensor) -> torch.Tensor:
  weights = (score + 1e-4) * torch.abs(randn)
  return torch.argsort(weights, dim=-1)


def _decoder_kwargs(feat: Feat, dtype: torch.dtype, *, pssm: bool) -> dict[str, object]:
  coef = feat.pssm_coef.to(dtype=dtype)
  bias = feat.pssm_bias.to(dtype=dtype)
  multi = 0.0
  flag = False
  if pssm:
    coef = torch.ones_like(coef)
    bias = torch.zeros_like(bias)
    bias[..., 0] = 1
    multi = PSSM_MULTI
    flag = True
  return {
    "temperature": AR_TEMPERATURE,
    "omit_AAs_np": np.zeros(VOCAB, dtype=np.float64),
    "bias_AAs_np": np.zeros(VOCAB, dtype=np.float64),
    "chain_M_pos": feat.chain_m_pos.to(dtype=dtype),
    "omit_AA_mask": feat.omit_aa_mask.to(dtype=dtype),
    "pssm_coef": coef,
    "pssm_bias": bias,
    "pssm_multi": multi,
    "pssm_log_odds_flag": False,
    "pssm_log_odds_mask": None,
    "pssm_bias_flag": flag,
    "bias_by_res": feat.bias_by_res.to(dtype=dtype),
  }


def _stream(rng: np.random.Generator, n_step: int) -> np.ndarray:
  return rng.random((1, n_step), dtype=np.float64)


def _refine_constants(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
  """Refine ``constant`` omits X (``omit_AAs=['X']``).

  ``optimize_sequence`` / ``tied_optimize_sequence`` build 20 candidates but sample over
  ``predicted_E[:vocab]`` (21 entries); drawing X indexes ``sort_seqs`` out of bounds
  (``run_utils.py:175,423``). Masking X keeps every draw inside the candidate set.
  """
  constant = torch.zeros(VOCAB, dtype=dtype)
  constant[REFINE_OMIT_INDEX] = 1.0
  return constant, torch.zeros(VOCAB, dtype=dtype)


def _call_optimize(
  run_utils: object,
  model: torch.nn.Module,
  *,
  tied: bool,
  seq: str,
  encoded: Encoded,
  feat: Feat,
  dtype: torch.dtype,
  opt_type: str,
  decoding_order: list[int],
  uniforms: np.ndarray,
  binding: str | None,
  partition_etabs: dict[int, tuple[torch.Tensor, torch.Tensor, str]] | None,
  partition_index: torch.Tensor | None,
  inter_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, DrawCursor]:
  import etab_utils  # noqa: PLC0415

  constant, constant_bias = _refine_constants(dtype)
  h_ex = run_utils.cat_neighbors_nodes(  # type: ignore[attr-defined]
    torch.zeros_like(encoded.h_v),
    encoded.h_e,
    encoded.e_idx,
  )
  h_exv = run_utils.cat_neighbors_nodes(encoded.h_v, h_ex, encoded.e_idx)  # type: ignore[attr-defined]
  mask = feat.mask.to(dtype=dtype) * feat.chain_m_pos.to(dtype=dtype)
  kwargs: dict[str, object] = {
    "constant": constant,
    "constant_bias": constant_bias,
    "bias_by_res": feat.bias_by_res.to(dtype=dtype),
    "pssm_bias_flag": False,
    "pssm_coef": feat.pssm_coef.to(dtype=dtype),
    "pssm_bias": feat.pssm_bias.to(dtype=dtype),
    "pssm_multi": 0.0,
    "pssm_log_odds_flag": False,
    "pssm_log_odds_mask": None,
    "omit_AA_mask": feat.omit_aa_mask.to(dtype=dtype),
    "model": model,
    "h_E": encoded.h_e,
    "h_EXV_encoder": h_exv,
    "h_V": encoded.h_v,
    "decoding_order": decoding_order,
    "partition_etabs": partition_etabs,
    "partition_index": partition_index,
    "inter_mask": inter_mask,
    "binding_optimization": binding,
    "vocab": VOCAB,
  }
  with injected_uniform_draws(uniforms) as cursor:
    if tied:
      seq_out = run_utils.tied_optimize_sequence(  # type: ignore[attr-defined]
        seq,
        encoded.etab_flat,
        encoded.e_idx,
        mask,
        feat.chain_m.to(dtype=dtype),
        opt_type,
        etab_utils.seq_to_ints,
        REFINE_TEMPERATURE,
        tied_pos=feat.tied_pos,
        tied_beta=feat.tied_beta.to(dtype=dtype),
        tied_epistasis=True,
        **kwargs,
      )
    else:
      seq_out = run_utils.optimize_sequence(  # type: ignore[attr-defined]
        seq,
        encoded.etab_flat,
        encoded.e_idx,
        mask,
        feat.chain_m.to(dtype=dtype),
        opt_type,
        etab_utils.seq_to_ints,
        REFINE_TEMPERATURE,
        **kwargs,
      )
  return seq_out, cursor


def _binding_context(
  run_utils: object,
  model: torch.nn.Module,
  feat: Feat,
  fixture: Fixture,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, str]], torch.Tensor, torch.Tensor]:
  cfg = SimpleNamespace(dev=torch.device("cpu"), model=SimpleNamespace(vocab=VOCAB))
  partitions = fixture.binding_partitions
  if partitions is None:
    msg = f"{fixture.name} has no binding partitions"
    raise RuntimeError(msg)
  partition_etabs: dict[int, tuple[torch.Tensor, torch.Tensor, str]] = {}
  for index, partition in enumerate(partitions):
    partition_etabs[index] = run_utils.get_etab(model, feat.parsed, cfg, partition)  # type: ignore[attr-defined]
  partition_index = run_utils.chain_to_partition_map(  # type: ignore[attr-defined]
    feat.chain_encoding,
    feat.chain_order,
    partitions,
  )
  inter_mask = run_utils.inter_partition_contact_mask(  # type: ignore[attr-defined]
    feat.x[:, :, 1],
    partition_index,
    BINDING_CUTOFF,
  )
  return partition_etabs, partition_index, inter_mask


def _dump_deterministic(
  buckets: Buckets,
  *,
  checkpoint: str,
  fixture: Fixture,
  feat: Feat,
  encoded: Encoded,
  log_probs: torch.Tensor,
  decoding_order: torch.Tensor,
  dec_h_v: list[torch.Tensor],
  dec_h_e: list[torch.Tensor],
) -> None:
  import etab_utils  # noqa: PLC0415

  prefix = f"{checkpoint}__{fixture.name}__"
  length = int(feat.s.shape[1])
  _put(buckets, ("potts_head", "pottsmpnn_full"), prefix + "X", feat.x)
  _put(
    buckets,
    ("potts_head", "pottsmpnn_full", "potts_ar_decode", "potts_refine"),
    prefix + "S",
    feat.s,
  )
  _put(
    buckets,
    (
      "potts_head",
      "potts_merge_pair_d2",
      "potts_merge_pair_d4",
      "potts_energy",
      "potts_ar_decode",
      "potts_refine",
      "pottsmpnn_full",
    ),
    prefix + "mask",
    feat.mask,
  )
  _put(buckets, ("potts_head", "pottsmpnn_full"), prefix + "residue_idx", feat.residue_idx)
  _put(buckets, ("potts_head", "pottsmpnn_full"), prefix + "chain_encoding", feat.chain_encoding)
  _put(
    buckets,
    (
      "potts_head",
      "potts_merge_pair_d2",
      "potts_merge_pair_d4",
      "potts_energy",
      "potts_ar_decode",
      "potts_refine",
      "pottsmpnn_full",
    ),
    prefix + "E_idx",
    encoded.e_idx,
  )
  _put(buckets, ("potts_head",), prefix + "h_E", encoded.h_e)
  _put(buckets, ("potts_head", "potts_merge_pair_d2"), prefix + "etab_raw", encoded.etab_raw)
  _put(
    buckets,
    ("potts_merge_pair_d2", "potts_merge_pair_d4", "pottsmpnn_full"),
    prefix + "etab_forward",
    encoded.etab_forward,
  )
  _put(buckets, ("potts_merge_pair_d4",), prefix + "etab_energy", encoded.etab_energy)
  padded = torch.nn.functional.pad(encoded.etab_energy, (0, 2, 0, 2))
  generator = torch.Generator()
  generator.manual_seed(SEED)
  seq_random = torch.randint(0, VOCAB, (N_RANDOM, length), generator=generator)
  seq_native = feat.s[:, :length].to(dtype=torch.int64)
  seqs = torch.cat([seq_native.unsqueeze(1), seq_random.unsqueeze(0)], dim=1)
  scores, _, _ = etab_utils.calc_eners(padded, encoded.e_idx, seqs, None, filter=False)
  _put(buckets, ("potts_energy",), prefix + "etab_energy_padded", padded)
  _put(buckets, ("potts_energy",), prefix + "seq_native", seq_native)
  _put(buckets, ("potts_energy",), prefix + "seq_random", seq_random)
  _put(buckets, ("potts_energy",), prefix + "energy_native", scores[:, 0])
  _put(buckets, ("potts_energy",), prefix + "energy_random", scores[:, 1:])
  _put(buckets, ("pottsmpnn_full", "potts_ar_decode"), prefix + "chain_M", feat.chain_m)
  _put(buckets, ("potts_ar_decode", "potts_refine"), prefix + "chain_M_pos", feat.chain_m_pos)
  _put(buckets, ("potts_ar_decode", "potts_refine"), prefix + "bias_by_res", feat.bias_by_res)
  _put(buckets, ("potts_ar_decode", "potts_refine"), prefix + "omit_AA_mask", feat.omit_aa_mask)
  _put(buckets, ("potts_ar_decode",), prefix + "pssm_coef", feat.pssm_coef)
  _put(buckets, ("potts_ar_decode",), prefix + "pssm_bias", feat.pssm_bias)
  _put(buckets, ("potts_ar_decode",), prefix + "pssm_log_odds", feat.pssm_log_odds)
  _put(
    buckets,
    ("potts_ar_decode", "potts_refine"),
    prefix + "skip_gaps",
    np.asarray(fixture.skip_gaps),
  )
  _put(buckets, ("pottsmpnn_full", "potts_ar_decode"), prefix + "decoding_order", decoding_order)
  _put(buckets, ("pottsmpnn_full",), prefix + "log_probs", log_probs)
  _put(buckets, ("potts_ar_decode", "potts_refine"), prefix + "h_V", encoded.h_v)
  _put(buckets, ("potts_ar_decode", "potts_refine"), prefix + "h_E", encoded.h_e)
  _put(buckets, ("potts_refine",), prefix + "etab_flat", encoded.etab_flat)
  for index, (h_v, h_e) in enumerate(zip(encoded.enc_h_v, encoded.enc_h_e, strict=True)):
    _put(buckets, ("pottsmpnn_full",), prefix + f"enc_h_V_{index}", h_v)
    _put(buckets, ("pottsmpnn_full",), prefix + f"enc_h_E_{index}", h_e)
  for index, h_e in enumerate(dec_h_e):
    _put(buckets, ("pottsmpnn_full",), prefix + f"dec_h_E_{index}", h_e)
  for index, h_v in enumerate(dec_h_v):
    _put(buckets, ("pottsmpnn_full",), prefix + f"dec_h_V_{index}", h_v)


def _dump_ar(
  buckets: Buckets,
  potts: object,
  model: torch.nn.Module,
  *,
  checkpoint: str,
  fixture: Fixture,
  feat: Feat,
  encoded: Encoded,
  dtype: torch.dtype,
  rng: np.random.Generator,
  decoding_order: torch.Tensor,
  ar_randn: torch.Tensor,
) -> None:
  prefix = f"{checkpoint}__{fixture.name}__"
  length = int(feat.s.shape[1])
  _put(buckets, ("potts_ar_decode",), prefix + "ar_randn", ar_randn)
  randn = torch.zeros(feat.chain_m.shape, dtype=dtype)
  kwargs = _decoder_kwargs(feat, dtype, pssm=False)
  uniforms = _stream(rng, length)
  with injected_uniform_draws(uniforms) as cursor:
    output, probs = model.decoder(  # type: ignore[operator]
      encoded.h_v,
      encoded.e_idx,
      encoded.h_e,
      randn,
      feat.s,
      feat.chain_m.to(dtype=dtype),
      feat.chain_encoding,
      feat.residue_idx,
      mask=feat.mask.to(dtype=dtype),
      decoding_order=untied_decoding_order(decoding_order),
      **kwargs,
    )
  _put(buckets, ("potts_ar_decode",), prefix + "ar_uniforms", uniforms)
  _put(
    buckets,
    ("potts_ar_decode",),
    prefix + "ar_uniforms_consumed",
    np.asarray(cursor.consumed_steps),
  )
  _put(buckets, ("potts_ar_decode",), prefix + "ar_temperature", np.asarray(AR_TEMPERATURE))
  _put(buckets, ("potts_ar_decode",), prefix + "ar_S", output["S"])
  _put(buckets, ("potts_ar_decode",), prefix + "ar_probs", probs)
  if fixture.run_pssm:
    pssm_kwargs = _decoder_kwargs(feat, dtype, pssm=True)
    pssm_u = _stream(rng, length)
    with injected_uniform_draws(pssm_u) as cursor:
      pssm_out, pssm_probs = model.decoder(  # type: ignore[operator]
        encoded.h_v,
        encoded.e_idx,
        encoded.h_e,
        randn,
        feat.s,
        feat.chain_m.to(dtype=dtype),
        feat.chain_encoding,
        feat.residue_idx,
        mask=feat.mask.to(dtype=dtype),
        decoding_order=untied_decoding_order(decoding_order),
        **pssm_kwargs,
      )
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_uniforms", pssm_u)
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_coef", pssm_kwargs["pssm_coef"])  # type: ignore[arg-type]
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_bias", pssm_kwargs["pssm_bias"])  # type: ignore[arg-type]
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_multi", np.asarray(PSSM_MULTI))
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_S", pssm_out["S"])
    _put(buckets, ("potts_ar_decode",), prefix + "ar_pssm_probs", pssm_probs)
    _put(
      buckets,
      ("potts_ar_decode",),
      prefix + "ar_pssm_uniforms_consumed",
      np.asarray(cursor.consumed_steps),
    )
  if fixture.run_tied_decode:
    tied_randn = torch.as_tensor(rng.standard_normal(tuple(feat.chain_m.shape)), dtype=dtype)
    tied_u = _stream(rng, length)
    with (
      injected_uniform_draws(tied_u) as cursor,
      injected_tied_randn(potts.PottsMPNN, tied_randn) as orders,  # type: ignore[attr-defined]
    ):
      tied_out, tied_probs = model.tied_decoder(  # type: ignore[operator]
        encoded.h_v,
        encoded.e_idx,
        encoded.h_e,
        tied_randn,
        feat.s,
        feat.chain_m.to(dtype=dtype),
        feat.chain_encoding,
        feat.residue_idx,
        mask=feat.mask.to(dtype=dtype),
        tied_pos=feat.tied_pos,
        tied_beta=feat.tied_beta.to(dtype=dtype),
        **kwargs,
      )
    if len(orders) != 1:
      msg = f"tied_decoder recorded {len(orders)} orders"
      raise RuntimeError(msg)
    _put(buckets, ("potts_ar_decode",), prefix + "tied_randn", tied_randn)
    _put(buckets, ("potts_ar_decode",), prefix + "tied_decoding_order", orders[0])
    _put(buckets, ("potts_ar_decode",), prefix + "tied_pos", np.asarray(json.dumps(feat.tied_pos)))
    _put(buckets, ("potts_ar_decode",), prefix + "tied_beta", feat.tied_beta)
    _put(buckets, ("potts_ar_decode",), prefix + "tied_uniforms", tied_u)
    _put(
      buckets,
      ("potts_ar_decode",),
      prefix + "tied_uniforms_consumed",
      np.asarray(cursor.consumed_steps),
    )
    _put(buckets, ("potts_ar_decode",), prefix + "tied_S", tied_out["S"])
    _put(buckets, ("potts_ar_decode",), prefix + "tied_probs", tied_probs)


def _dump_refine(
  buckets: Buckets,
  run_utils: object,
  model: torch.nn.Module,
  *,
  checkpoint: str,
  fixture: Fixture,
  feat: Feat,
  encoded: Encoded,
  dtype: torch.dtype,
  rng: np.random.Generator,
) -> None:
  prefix = f"{checkpoint}__{fixture.name}__"
  length = int(feat.s.shape[1])
  refine_randn = torch.as_tensor(rng.standard_normal((1, length)), dtype=dtype)
  order = _order_from_randn(feat.chain_m.to(dtype=dtype), refine_randn)
  order_list = [int(v) for v in order[0].detach().cpu().tolist()]
  _put(buckets, ("potts_refine",), prefix + "refine_randn", refine_randn)
  _put(buckets, ("potts_refine",), prefix + "refine_decoding_order", order)
  _put(buckets, ("potts_refine",), prefix + "refine_temperature", np.asarray(REFINE_TEMPERATURE))
  _put(buckets, ("potts_refine",), prefix + "refine_omit_index", np.asarray(REFINE_OMIT_INDEX))
  _put(buckets, ("potts_refine",), prefix + "seq", np.asarray(feat.seq))
  modes = ["potts", "nodes"]
  if length <= CONVERGE_MAX_L:
    modes.append("potts_converge")
  bindings: list[str | None] = [None]
  partition_etabs = None
  partition_index = None
  inter_mask = None
  if fixture.run_binding:
    partition_etabs, partition_index, inter_mask = _binding_context(run_utils, model, feat, fixture)
    bindings = ["both", "only"]
    _put(buckets, ("potts_refine",), prefix + "partition_index", partition_index)
    _put(buckets, ("potts_refine",), prefix + "inter_mask", inter_mask)
    for index, (etab, e_idx, _wt) in partition_etabs.items():
      _put(buckets, ("potts_refine",), prefix + f"partition_{index}_etab", etab)
      _put(buckets, ("potts_refine",), prefix + f"partition_{index}_E_idx", e_idx)
  for binding in bindings:
    tag = "none" if binding is None else binding
    for mode in modes:
      n_step = length if "converge" not in mode else N_SWEEPS * length
      uniforms = _stream(rng, n_step)
      seq_out, cursor = _call_optimize(
        run_utils,
        model,
        tied=fixture.run_tied_refine,
        seq=feat.seq,
        encoded=encoded,
        feat=feat,
        dtype=dtype,
        opt_type=mode,
        decoding_order=order_list,
        uniforms=uniforms,
        binding=binding,
        partition_etabs=partition_etabs,
        partition_index=partition_index,
        inter_mask=inter_mask,
      )
      stem = f"refine_{mode}_{tag}"
      _put(buckets, ("potts_refine",), prefix + stem + "_S", seq_out)
      _put(buckets, ("potts_refine",), prefix + stem + "_uniforms", uniforms)
      _put(
        buckets,
        ("potts_refine",),
        prefix + stem + "_uniforms_consumed",
        np.asarray(cursor.consumed_steps),
      )
      _put(buckets, ("potts_refine",), prefix + stem + "_tied", np.asarray(fixture.run_tied_refine))


def _teacher_forced(
  model: torch.nn.Module,
  feat: Feat,
  dtype: torch.dtype,
  decoding_order: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
  dec_h_v: list[torch.Tensor] = []
  dec_h_e: list[torch.Tensor] = []

  def hook(
    _module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
  ) -> None:
    dec_h_v.append(inputs[0].detach())
    dec_h_e.append(inputs[1].detach())
    dec_h_v.append(output.detach())

  handles = [layer.register_forward_hook(hook) for layer in model.decoder_layers]
  try:
    log_probs, _, _ = model(
      feat.x.to(dtype=dtype),
      feat.s,
      feat.mask.to(dtype=dtype),
      feat.chain_m.to(dtype=dtype),
      feat.residue_idx,
      feat.chain_encoding,
      torch.zeros_like(feat.chain_m, dtype=dtype),
      use_input_decoding_order=True,
      decoding_order=decoding_order,
    )
  finally:
    for handle in handles:
      handle.remove()
  # input of layer 0, then each layer output (outputs sit on the odd slots).
  kept_v = [dec_h_v[0], *dec_h_v[1::2]]
  return log_probs, kept_v, dec_h_e


def _maybe_declayer(
  buckets: Buckets,
  *,
  checkpoint: str,
  fixture: str,
  precision: str,
  model: torch.nn.Module,
  dec_h_v: list[torch.Tensor],
  dec_h_e: list[torch.Tensor],
  mask: torch.Tensor,
) -> None:
  if buckets.declayer is not None or precision != "f64":
    return
  if checkpoint != "vanilla_20" or fixture != "gap_le48_gaps0":
    return
  layer = model.decoder_layers[0]
  payload: dict[str, np.ndarray] = {
    "checkpoint": np.asarray(checkpoint),
    "fixture": np.asarray(fixture),
    "h_V": _np(dec_h_v[0]),
    "h_E": _np(dec_h_e[0]),
    "mask_V": _np(mask),
    "output": _np(dec_h_v[1]),
    "scale": np.asarray(layer.scale),
  }
  for name, tensor in layer.state_dict().items():
    payload["weight__" + name.replace(".", "__")] = _np(tensor)
  buckets.declayer = payload


def _dump_fixture(
  buckets: Buckets,
  potts: object,
  run_utils: object,
  model: torch.nn.Module,
  *,
  checkpoint: str,
  fixture: Fixture,
  precision: str,
) -> None:
  dtype = torch.float64 if precision == "f64" else torch.float32
  feat = _featurize(potts, fixture)
  encoded = _encode(potts, model, feat)
  length = int(feat.s.shape[1])
  rng = np.random.Generator(np.random.PCG64(SEED))
  ar_randn = torch.as_tensor(rng.standard_normal((1, length)), dtype=dtype)
  score = (
    feat.chain_m.to(dtype=dtype) * feat.chain_m_pos.to(dtype=dtype) * feat.mask.to(dtype=dtype)
  )
  decoding_order = _order_from_randn(score, ar_randn)
  log_probs, dec_h_v, dec_h_e = _teacher_forced(model, feat, dtype, decoding_order)
  _dump_deterministic(
    buckets,
    checkpoint=checkpoint,
    fixture=fixture,
    feat=feat,
    encoded=encoded,
    log_probs=log_probs,
    decoding_order=decoding_order,
    dec_h_v=dec_h_v,
    dec_h_e=dec_h_e,
  )
  _maybe_declayer(
    buckets,
    checkpoint=checkpoint,
    fixture=fixture.name,
    precision=precision,
    model=model,
    dec_h_v=dec_h_v,
    dec_h_e=dec_h_e,
    mask=feat.mask.to(dtype=dtype),
  )
  _dump_ar(
    buckets,
    potts,
    model,
    checkpoint=checkpoint,
    fixture=fixture,
    feat=feat,
    encoded=encoded,
    dtype=dtype,
    rng=rng,
    decoding_order=decoding_order,
    ar_randn=ar_randn,
  )
  _dump_refine(
    buckets,
    run_utils,
    model,
    checkpoint=checkpoint,
    fixture=fixture,
    feat=feat,
    encoded=encoded,
    dtype=dtype,
    rng=rng,
  )
  logger.info("dumped %s %s %s (L=%s)", checkpoint, precision, fixture.name, length)


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **payload)
  return _sha256_file(path)


def _write_manifest(
  path: Path,
  *,
  commit: str,
  checkpoints: list[dict[str, object]],
  dumps: list[dict[str, object]],
) -> None:
  lines = [
    f"upstream_commit = {_toml_str(commit)}",
    f"torch_version = {_toml_str(torch.__version__)}",
    f"seed = {SEED}",
    f"shim_sha256 = {_toml_str(shim_sha256())}",
    f"shim_sites = {_toml_list(SHIM_SITES)}",
    "shimmed = true",
    "",
  ]
  for row in checkpoints:
    lines.append("[[checkpoint]]")
    lines.append(f"id = {_toml_str(str(row['id']))}")
    lines.append(f"path = {_toml_str(str(row['path']))}")
    lines.append(f"weights_sha256 = {_toml_str(str(row['weights_sha256']))}")
    lines.append(f"missing_keys = {_toml_list(row['missing_keys'])}")  # type: ignore[arg-type]
    lines.append(f"unexpected_keys = {_toml_list(row['unexpected_keys'])}")  # type: ignore[arg-type]
    lines.append("")
  for row in dumps:
    lines.append("[[dump]]")
    lines.append(f"wave = {_toml_str(str(row['wave']))}")
    lines.append(f"precision = {_toml_str(str(row['precision']))}")
    lines.append(f"path = {_toml_str(str(row['path']))}")
    lines.append(f"sha256 = {_toml_str(str(row['sha256']))}")
    lines.append(f"shimmed = {'true' if row['shimmed'] else 'false'}")
    lines.append("")
  path.write_text("\n".join(lines))


def _assert_inverse_cdf_edges() -> None:
  probs = torch.tensor([0.0, 0.25, 0.75, 0.0], dtype=torch.float64)
  almost_one = torch.tensor(1.0 - (2.0**-53), dtype=torch.float64)
  last = int(inverse_cdf_index(probs, almost_one))
  if last != 2:
    msg = f"u=1-2**-53 returned {last}, expected last positive index 2"
    raise RuntimeError(msg)
  for unit in (0.0, 0.1, 0.5, 0.9, float(almost_one)):
    drawn = int(inverse_cdf_index(probs, torch.tensor(unit, dtype=torch.float64)))
    if drawn == 3 or probs[drawn] == 0:
      msg = f"inverse-CDF drew a zero-probability index {drawn} at u={unit}"
      raise RuntimeError(msg)


def _selfcheck(potts: object, potts_root: Path, fixtures_dir: Path) -> None:
  """Bit-identical decoder outputs with the shim context manager off."""
  _assert_inverse_cdf_edges()
  pin_commit, pin_weights = _load_pin(potts_root)
  _upstream_commit(potts_root, pin_commit)
  rel = CHECKPOINTS["vanilla_20"]
  path = potts_root / rel
  digest = _sha256_file(path)
  if digest != pin_weights[rel]:
    msg = f"sha256 mismatch for {rel}"
    raise SystemExit(msg)
  model, _missing, _unexpected = _load_model(potts, path, precision="f32")
  fixture = Fixture(name="gap_le48_gaps0", path=fixtures_dir / "gap_le48.pdb", skip_gaps=False)
  feat = _featurize(potts, fixture)
  encoded = _encode(potts, model, feat)
  order = torch.arange(feat.s.shape[1], dtype=torch.long).view(1, -1)
  kwargs = _decoder_kwargs(feat, torch.float32, pssm=False)
  randn = torch.zeros_like(feat.chain_m)

  def once() -> torch.Tensor:
    torch.manual_seed(2)
    output, _probs = model.decoder(  # type: ignore[operator]
      encoded.h_v,
      encoded.e_idx,
      encoded.h_e,
      randn,
      feat.s,
      feat.chain_m,
      feat.chain_encoding,
      feat.residue_idx,
      mask=feat.mask,
      decoding_order=order,
      **kwargs,
    )
    return output["S"]

  before = torch.multinomial
  original_tied = potts.PottsMPNN.tied_decoder  # type: ignore[attr-defined]
  first = once()
  second = once()
  if not torch.equal(first, second):
    msg = "unshimmed decoder is not bit-identical across seeded repeats"
    raise RuntimeError(msg)
  with injected_uniform_draws(np.full((1, 64), 0.5, dtype=np.float64)):
    pass
  with injected_tied_randn(potts.PottsMPNN, torch.zeros(1, int(feat.s.shape[1]))):  # type: ignore[attr-defined]
    pass
  third = once()
  if not torch.equal(first, third):
    msg = "shim leaked: decoder tokens differ with the context manager off"
    raise RuntimeError(msg)
  if torch.multinomial is not before:
    msg = "torch.multinomial was not restored"
    raise RuntimeError(msg)
  if potts.PottsMPNN.tied_decoder is not original_tied:  # type: ignore[attr-defined]
    msg = "tied_decoder was not restored"
    raise RuntimeError(msg)
  if len(shim_sha256()) != 64:
    msg = "shim_sha256 is not a sha256 hex digest"
    raise RuntimeError(msg)
  logger.info("selfcheck passed (shim is a no-op when disabled)")


def _run_dumps(
  potts: object,
  run_utils: object,
  args: argparse.Namespace,
  commit: str,
  pin_weights: dict[str, str],
) -> None:
  if args.out is None:
    msg = "--out is required unless --selfcheck is set"
    raise SystemExit(msg)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  fixtures = _collect_fixtures(Path(args.potts_root), Path(args.fixtures_dir), out / "_generated")
  checkpoint_rows: list[dict[str, object]] = []
  dump_rows: list[dict[str, object]] = []
  for precision in args.precisions:
    buckets = Buckets(precision=precision)
    for checkpoint_id in args.checkpoints:
      if checkpoint_id not in CHECKPOINTS:
        msg = f"unknown checkpoint id {checkpoint_id}"
        raise SystemExit(msg)
      rel = CHECKPOINTS[checkpoint_id]
      path = Path(args.potts_root) / rel
      digest = _sha256_file(path)
      pinned = pin_weights[rel]
      if digest != pinned:
        msg = f"sha256 mismatch for {rel}: file {digest} pin {pinned}"
        raise SystemExit(msg)
      model, missing, unexpected = _load_model(potts, path, precision=precision)
      checkpoint_rows.append(
        {
          "id": f"{checkpoint_id}_{precision}",
          "path": rel,
          "weights_sha256": digest,
          "missing_keys": missing,
          "unexpected_keys": unexpected,
        },
      )
      with contextlib.ExitStack() as precision_shims, torch.no_grad():
        if precision == "f64":
          precision_shims.enter_context(
            weight_dtype_positional_encodings(potts.PositionalEncodings),  # type: ignore[attr-defined]
          )
          precision_shims.enter_context(rbf_follows_input_dtype(potts.ProteinFeatures))  # type: ignore[attr-defined]
        for fixture in fixtures:
          _dump_fixture(
            buckets,
            potts,
            run_utils,
            model,
            checkpoint=checkpoint_id,
            fixture=fixture,
            precision=precision,
          )
    for wave in WAVES:
      payload = buckets.waves[wave]
      payload["precision"] = np.asarray(precision)
      payload["seed"] = np.asarray(SEED)
      payload["fixture_names"] = np.asarray([fixture.name for fixture in fixtures])
      payload["checkpoint_ids"] = np.asarray(list(args.checkpoints))
      dest = out / wave / f"oracle_{precision}.npz"
      digest = _save_npz(dest, payload)
      dump_rows.append(
        {
          "wave": wave,
          "precision": precision,
          "path": str(dest.relative_to(out)),
          "sha256": digest,
          "shimmed": wave in SHIMMED_WAVES,
        },
      )
      logger.info("wrote %s (%s)", dest, digest)
    if precision == "f64" and buckets.declayer is None:
      logger.warning(
        "declayer_f64.npz was not written (needs checkpoint vanilla_20 and fixture gap_le48_gaps0)",
      )
    if buckets.declayer is not None:
      dest = out / "declayer_f64.npz"
      digest = _save_npz(dest, buckets.declayer)
      dump_rows.append(
        {
          "wave": "declayer_f64",
          "precision": "f64",
          "path": dest.name,
          "sha256": digest,
          "shimmed": False,
        },
      )
      logger.info("wrote %s (%s)", dest, digest)
  _write_manifest(
    out / "oracle_manifest.toml",
    commit=commit,
    checkpoints=checkpoint_rows,
    dumps=dump_rows,
  )


def _rows(value: torch.Tensor) -> torch.Tensor:
  """Drop a leading batch of 1. ``E_idx`` / masks arrive as ``(1, L, ...)``."""
  array = value.detach().cpu()
  if array.ndim >= 1 and array.shape[0] == 1 and array.ndim >= 2:
    array = array[0]
  return array


def _e_idx_rows(value: torch.Tensor) -> torch.Tensor:
  array = _rows(value).to(dtype=torch.int64)
  if array.ndim != 2:
    msg = f"E_idx must be (L, K) after squeezing the batch, got {tuple(array.shape)}"
    raise RuntimeError(msg)
  return array


def _same_neighbour_sets(left: torch.Tensor, right: torch.Tensor) -> bool:
  if left.shape[0] != right.shape[0]:
    return False
  for index in range(int(left.shape[0])):
    if set(int(v) for v in left[index].tolist()) != set(int(v) for v in right[index].tolist()):
      return False
  return True


def _order_mask_backward(decoding_order: torch.Tensor, length: int) -> torch.Tensor:
  """Upstream ``order_mask_backward`` (``potts_mpnn_utils.py:1423-1424``)."""
  order = decoding_order.detach().to(dtype=torch.long).reshape(1, length)
  lower = 1.0 - torch.triu(torch.ones(length, length, dtype=torch.float64))
  perm = torch.nn.functional.one_hot(order, num_classes=length).to(dtype=torch.float64)
  mask = torch.einsum("ij,biq,bjp->bqp", lower, perm, perm)
  return mask[0]


def _append_text_line(path: Path, line: str, *, identity: str) -> None:
  """Append ``line`` unless ``identity`` is already recorded. Never rewrite a line."""
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  for existing in text.splitlines():
    stripped = existing.strip()
    if not stripped:
      continue
    if stripped == line.strip():
      return
    tokens = stripped.split()
    if identity in tokens or stripped == identity:
      msg = f"refusing to modify existing sha entry {identity} in {path}"
      raise SystemExit(msg)
  prefix = "" if text.endswith("\n") or text == "" else "\n"
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(prefix + line + "\n")


def _append_dump_row(path: Path, row: dict[str, object]) -> None:
  """Append one ``[[dump]]`` table. An existing path keeps its recorded sha."""
  rel = str(row["path"])
  digest = str(row["sha256"])
  if path.is_file():
    text = path.read_text(encoding="utf-8")
    needle = f"path = {_toml_str(rel)}"
    if needle in text:
      if f"sha256 = {_toml_str(digest)}" not in text:
        msg = f"refusing to modify existing manifest sha for {rel}"
        raise SystemExit(msg)
      return
  else:
    text = ""
  block = "\n".join(
    [
      "[[dump]]",
      f"wave = {_toml_str(str(row['wave']))}",
      f"precision = {_toml_str(str(row['precision']))}",
      f"path = {_toml_str(rel)}",
      f"sha256 = {_toml_str(digest)}",
      f"shimmed = {'true' if row['shimmed'] else 'false'}",
      "",
    ],
  )
  prefix = "" if text.endswith("\n") or text == "" else "\n"
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(prefix + block)


def _write_chain(path: Path, residues: list[tuple[str, int, str, float, float]]) -> str:
  rows: list[str] = []
  serial = 1
  for chain, resseq, resname, x, y in residues:
    rows.extend(_backbone(chain, resseq, resname, x, y, serial=serial))
    serial += 4
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(rows) + "\n")
  return _sha256_file(path)


def _write_order_pdbs(work: Path) -> dict[str, tuple[Path, str]]:
  """Synthetic PDBs for the order batch. Straight chains so kNN distances are unique."""
  written: dict[str, tuple[Path, str]] = {}

  def straight(
    length: int,
    *,
    chain: str = "A",
    y: float = 0.0,
  ) -> list[tuple[str, int, str, float, float]]:
    return [(chain, index + 1, "ALA", float(index) * 3.8, y) for index in range(length)]

  specs: dict[str, list[tuple[str, int, str, float, float]]] = {
    "L30": straight(30),
    "L48": straight(48),
    "L49": straight(49),
    "partition_a": straight(20),
    "binding": straight(20) + straight(12, chain="B", y=6.0),
    # resseq 1,2,4,5 leaves residue 3 as a gap row. Residue 1 is fixed below.
    "gap_fixed": [
      ("A", resseq, "ALA", float(index) * 3.8, 0.0) for index, resseq in enumerate((1, 2, 4, 5))
    ],
    "tied": straight(6),
  }
  for name, residues in specs.items():
    path = work / f"{name}.pdb"
    written[name] = (path, _write_chain(path, residues))
  return written


def _record_graph(
  payload: dict[str, np.ndarray],
  potts: object,
  model: torch.nn.Module,
  *,
  name: str,
  path: Path,
  pdb_sha: str,
) -> torch.Tensor:
  fixture = Fixture(name=name, path=path, skip_gaps=False)
  feat = _featurize(potts, fixture)
  encoded = _encode(potts, model, feat)
  e_idx = _e_idx_rows(encoded.e_idx)
  coords = _rows(feat.x).to(dtype=torch.float64)
  if coords.ndim != 3 or coords.shape[-2:] != (4, 3):
    msg = f"{name} coords have shape {tuple(coords.shape)}, expected (L, 4, 3)"
    raise RuntimeError(msg)
  if int(coords.shape[0]) != int(e_idx.shape[0]):
    msg = f"{name} L={coords.shape[0]} but E_idx has {e_idx.shape[0]} rows"
    raise RuntimeError(msg)
  payload[f"{name}__coords"] = coords.numpy()
  payload[f"{name}__mask"] = _rows(feat.mask).to(dtype=torch.float64).numpy()
  payload[f"{name}__residue_idx"] = _rows(feat.residue_idx).to(dtype=torch.int64).numpy()
  payload[f"{name}__chain_encoding"] = _rows(feat.chain_encoding).to(dtype=torch.int64).numpy()
  payload[f"{name}__E_idx"] = e_idx.numpy()
  payload[f"{name}__pdb_sha256"] = np.asarray(pdb_sha)
  return e_idx


def _tied_randn(length: int, group: list[int], non_member: int, scale: float) -> np.ndarray:
  """Raw order visits the listing-last member, then ``non_member``, then the rest of the group."""
  used = [group[-1], non_member, *group[:-1]]
  if len(set(used)) != len(used):
    msg = f"tied randn reused a row in {used}"
    raise RuntimeError(msg)
  tail = [index for index in range(length) if index not in used]
  order = used + tail
  if len(order) != length:
    msg = f"tied randn order covers {len(order)} of {length}"
    raise RuntimeError(msg)
  values = np.zeros(length, dtype=np.float64)
  for rank, index in enumerate(order):
    values[index] = scale * float(rank + 1)
  return values


def _assert_tie_separates(raw: torch.Tensor, group: list[int]) -> None:
  rank = torch.empty(raw.numel(), dtype=torch.long)
  rank[raw.to(dtype=torch.long)] = torch.arange(raw.numel())
  by_rank = sorted(group, key=lambda index: int(rank[index]))
  if list(group) == by_rank:
    msg = "tied listing order matches raw-rank order; the raw-rank control would not fire"
    raise RuntimeError(msg)
  member_ranks = [int(rank[index]) for index in group]
  lo, hi = min(member_ranks), max(member_ranks)
  between = [
    index
    for index in range(int(raw.numel()))
    if index not in set(group) and lo < int(rank[index]) < hi
  ]
  if not between:
    msg = "no non-member raw rank lies between two tied members"
    raise RuntimeError(msg)


def _run_order_dumps(
  potts: object,
  run_utils: object,
  args: argparse.Namespace,
  commit: str,
  pin_weights: dict[str, str],
) -> None:
  """Neighbour sets, tied rank, and AR/refine orders. New files only."""
  if args.out is None:
    msg = "--out is required"
    raise SystemExit(msg)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  checkpoint_id = "vanilla_20"
  rel = CHECKPOINTS[checkpoint_id]
  weights = Path(args.potts_root) / rel
  digest = _sha256_file(weights)
  if digest != pin_weights[rel]:
    msg = f"sha256 mismatch for {rel}: file {digest} pin {pin_weights[rel]}"
    raise SystemExit(msg)
  pdbs = _write_order_pdbs(out / "_order_generated")
  model, _missing, _unexpected = _load_model(potts, weights, precision="f64")
  payload: dict[str, np.ndarray] = {
    "precision": np.asarray("f64"),
    "upstream_commit": np.asarray(commit),
    "checkpoint_id": np.asarray(checkpoint_id),
    "weights_sha256": np.asarray(digest),
    "neighbour_cases": np.asarray(["L30", "L48", "L49", "partition_Lp20"]),
  }
  expected_length = {"L30": 30, "L48": 48, "L49": 49}
  # The same f64 shims the main dump applies (see _run_dumps): upstream's
  # PositionalEncodings hardcodes .float() and its RBF centres are f32, so an
  # f64 model without them dies in F.linear on a Float/Double mismatch.
  with contextlib.ExitStack() as precision_shims, torch.no_grad():
    precision_shims.enter_context(
      weight_dtype_positional_encodings(potts.PositionalEncodings),  # ty: ignore[unresolved-attribute]
    )
    precision_shims.enter_context(rbf_follows_input_dtype(potts.ProteinFeatures))  # ty: ignore[unresolved-attribute]
    for name, length in expected_length.items():
      path, pdb_sha = pdbs[name]
      e_idx = _record_graph(payload, potts, model, name=name, path=path, pdb_sha=pdb_sha)
      if int(e_idx.shape[0]) != length:
        msg = f"{name} featurized to L={e_idx.shape[0]}, expected {length}"
        raise RuntimeError(msg)
    solo_path, solo_sha = pdbs["partition_a"]
    solo_idx = _record_graph(
      payload,
      potts,
      model,
      name="partition_Lp20",
      path=solo_path,
      pdb_sha=solo_sha,
    )
    binding_path, binding_sha = pdbs["binding"]
    binding = Fixture(
      name="order_binding",
      path=binding_path,
      skip_gaps=False,
      run_binding=True,
      binding_partitions=[["A"], ["B"]],
    )
    binding_feat = _featurize(potts, binding)
    partition_etabs, _partition_index, _inter = _binding_context(
      run_utils,
      model,
      binding_feat,
      binding,
    )
    partition_idx = _e_idx_rows(partition_etabs[0][1])
    if int(partition_idx.shape[0]) >= 48:
      msg = f"binding partition L_p={partition_idx.shape[0]} is not < 48"
      raise RuntimeError(msg)
    if not _same_neighbour_sets(partition_idx, solo_idx):
      msg = "get_etab partition E_idx does not match the chain-A encoder graph"
      raise RuntimeError(msg)
    payload["partition_Lp20__E_idx"] = partition_idx.numpy()
    payload["partition_Lp20__binding_pdb_sha256"] = np.asarray(binding_sha)
    payload["partition_Lp20__L_p"] = np.asarray(partition_idx.shape[0])

    gap_path, gap_sha = pdbs["gap_fixed"]
    gap = Fixture(
      name="order_gap_fixed",
      path=gap_path,
      skip_gaps=False,
      fixed_positions={"A": [1]},
    )
    gap_feat = _featurize(potts, gap)
    gap_randn = torch.linspace(0.2, 1.1, int(gap_feat.s.shape[1]), dtype=torch.float64).view(1, -1)
    present = gap_feat.mask
    chain_m = gap_feat.chain_m
    chain_m_pos = gap_feat.chain_m_pos
    ar = _order_from_randn(chain_m * chain_m_pos * present, gap_randn)
    refine = _order_from_randn(chain_m, gap_randn)
    dropped = _order_from_randn(chain_m * present, gap_randn)
    if not bool((present == 0).any()):
      msg = "gap fixture has no mask==0 row"
      raise RuntimeError(msg)
    fixed_present = (chain_m_pos == 0) & (present == 1)
    if not bool(fixed_present.any()):
      msg = "gap fixture has no fixed present row (chain_M_pos==0); drop-chain_M_pos would not fire"
      raise RuntimeError(msg)
    if torch.equal(ar, refine) or torch.equal(ar, dropped):
      msg = "AR / refine / dropped-chain_M_pos orders are not separated"
      raise RuntimeError(msg)
    payload["knob_randn"] = _rows(gap_randn).numpy()
    payload["knob_chain_M"] = _rows(chain_m).to(dtype=torch.float64).numpy()
    payload["knob_chain_M_pos"] = _rows(chain_m_pos).to(dtype=torch.float64).numpy()
    payload["knob_mask"] = _rows(present).to(dtype=torch.float64).numpy()
    payload["knob_ar_order"] = _rows(ar).to(dtype=torch.int64).numpy()
    payload["knob_refine_order"] = _rows(refine).to(dtype=torch.int64).numpy()
    payload["knob_pdb_sha256"] = np.asarray(gap_sha)

    tied_path, tied_sha = pdbs["tied"]
    tied = Fixture(
      name="order_tied",
      path=tied_path,
      skip_gaps=False,
      tied_positions=[{"A": [3, 1]}],
      run_tied_decode=True,
    )
    tied_feat = _featurize(potts, tied)
    tied_encoded = _encode(potts, model, tied_feat)
    groups = [group for group in tied_feat.tied_pos if len(group) > 1]
    if len(groups) != 1:
      msg = f"expected one multi-member tied group, got {tied_feat.tied_pos}"
      raise RuntimeError(msg)
    group = [int(index) for index in groups[0]]
    length = int(tied_feat.s.shape[1])
    score = _rows(tied_feat.chain_m * tied_feat.chain_m_pos * tied_feat.mask)
    if not bool(torch.all(score == score.reshape(-1)[0])) or float(score.reshape(-1)[0]) == 0.0:
      msg = f"tied fixture score is not a positive constant: {score}"
      raise RuntimeError(msg)
    non_members = [index for index in range(length) if index not in group]
    if len(non_members) < 2:
      msg = "tied fixture needs two non-members so the three seeds can move the between-row"
      raise RuntimeError(msg)
    randn_rows = [
      _tied_randn(length, group, non_members[0], 0.10),
      _tied_randn(length, group, non_members[1], 0.13),
      _tied_randn(length, group, non_members[0], 0.17),
    ]
    orders: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dtype = torch.float64
    kwargs = _decoder_kwargs(tied_feat, dtype, pssm=False)
    for values in randn_rows:
      randn = torch.as_tensor(values, dtype=dtype).view(1, -1)
      raw = _order_from_randn(
        tied_feat.chain_m * tied_feat.chain_m_pos * tied_feat.mask,
        randn,
      )[0]
      _assert_tie_separates(raw, group)
      uniforms = np.full((1, length * 4), 0.5, dtype=np.float64)
      model_cls = getattr(potts, "PottsMPNN")
      with (
        injected_uniform_draws(uniforms),
        injected_tied_randn(model_cls, randn) as recorded,
      ):
        tied_feat_mask = tied_feat.mask.to(dtype=dtype)
        # Look the method up inside the shim, or the captured original skips it.
        tied_decoder = getattr(model, "tied_decoder")
        tied_decoder(
          tied_encoded.h_v,
          tied_encoded.e_idx,
          tied_encoded.h_e,
          randn,
          tied_feat.s,
          tied_feat.chain_m.to(dtype=dtype),
          tied_feat.chain_encoding,
          tied_feat.residue_idx,
          mask=tied_feat_mask,
          tied_pos=tied_feat.tied_pos,
          tied_beta=tied_feat.tied_beta.to(dtype=dtype),
          **kwargs,
        )
      if len(recorded) != 1:
        msg = f"tied_decoder recorded {len(recorded)} orders"
        raise RuntimeError(msg)
      flat = recorded[0].detach().to(dtype=torch.long).reshape(-1)
      if int(flat.numel()) != length:
        msg = f"flattened decoding_order has length {flat.numel()}, expected {length}"
        raise RuntimeError(msg)
      if torch.equal(flat, raw.to(dtype=torch.long)):
        msg = "grouped decoding order equals the raw order; the control would not fire"
        raise RuntimeError(msg)
      orders.append(flat.cpu().numpy())
      masks.append(_order_mask_backward(flat, length).numpy())
    payload["tied_randn"] = np.stack(randn_rows, axis=0)
    payload["tied_decoding_order"] = np.stack(orders, axis=0)
    payload["tied_order_mask_backward"] = np.stack(masks, axis=0)
    payload["tied_E_idx"] = _e_idx_rows(tied_encoded.e_idx).numpy()
    payload["tied_chain_M"] = _rows(tied_feat.chain_m).to(dtype=torch.float64).numpy()
    payload["tied_chain_M_pos"] = _rows(tied_feat.chain_m_pos).to(dtype=torch.float64).numpy()
    payload["tied_mask"] = _rows(tied_feat.mask).to(dtype=torch.float64).numpy()
    payload["tied_group"] = np.asarray(group, dtype=np.int64)
    payload["tied_pdb_sha256"] = np.asarray(tied_sha)

  dest = out / "potts_order" / "order_f64.npz"
  file_digest = _save_npz(dest, payload)
  relative = "potts_order/order_f64.npz"
  sidecar = dest.with_suffix(".npz.sha256")
  if sidecar.is_file() and sidecar.read_text(encoding="utf-8").strip() != file_digest:
    msg = f"refusing to modify existing sha entry {sidecar}"
    raise SystemExit(msg)
  if not sidecar.is_file():
    sidecar.write_text(file_digest + "\n", encoding="utf-8")
  repo_sha = Path(__file__).resolve().parents[2] / "tests/port/reference/a1_potts/oracles.sha256"
  _append_text_line(
    repo_sha,
    f"{file_digest}  ./{relative}",
    identity=f"./{relative}",
  )
  _append_dump_row(
    out / "oracle_manifest.toml",
    {
      "wave": "potts_order",
      "precision": "f64",
      "path": relative,
      "sha256": file_digest,
      "shimmed": True,
    },
  )
  logger.info("wrote %s (%s)", dest, file_digest)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--potts-root", type=Path, required=True)
  parser.add_argument("--fixtures-dir", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--checkpoints", nargs="*", default=list(CHECKPOINTS))
  parser.add_argument("--precisions", nargs="+", default=["f64", "f32"], choices=["f64", "f32"])
  parser.add_argument("--selfcheck", action="store_true")
  parser.add_argument(
    "--only",
    choices=("order",),
    help=(
      "Write potts_order/order_f64.npz and new sha entries, and do not regenerate existing dumps."
    ),
  )
  args = parser.parse_args()
  potts_root = Path(args.potts_root).resolve()
  sys.path.insert(0, str(potts_root))
  import potts_mpnn_utils as potts  # noqa: PLC0415
  import run_utils  # noqa: PLC0415

  pin_commit, pin_weights = _load_pin(potts_root)
  commit = _upstream_commit(potts_root, pin_commit)
  if args.selfcheck:
    if args.fixtures_dir is None:
      msg = "--fixtures-dir is required for --selfcheck"
      raise SystemExit(msg)
    _selfcheck(potts, potts_root, Path(args.fixtures_dir).resolve())
    return
  if args.only == "order":
    _run_order_dumps(potts, run_utils, args, commit, pin_weights)
    return
  if args.fixtures_dir is None:
    msg = "--fixtures-dir is required unless --only order is set"
    raise SystemExit(msg)
  _run_dumps(potts, run_utils, args, commit, pin_weights)


if __name__ == "__main__":
  main()
