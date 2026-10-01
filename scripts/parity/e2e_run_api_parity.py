#!/usr/bin/env python3
"""End-to-end parity of the public run API (spec -> runner -> sample()/score()) against upstream LigandMPNN.

Pre-registered: scripts/parity/e2e_run_api_parity.bth.toml.  Why this exists: the old
"end-to-end-run-apis" parity path called run_sample twice and compared the two results to each other
(self-consistency), and no test drove RunSpec / the runner against the upstream reference (aminx debt
#2326).  This drives BOTH implementations on identical inputs and compares their outputs.

The oracle is upstream's OWN code, in-process: ``run.py`` is executed with its real argument parsing
(--omit_AA, --bias_AA, --fixed_residues, --redesigned_residues, --temperature, ...), with
``ProteinMPNN.sample`` wrapped only to (a) inject the decoding-order randomness and (b) capture the
feature dict and outputs.  The aminx side goes through ``aminx.run.sample`` / ``score`` with a
``SamplingSpecification`` / ``ScoringSpecification``.

RNG streams cannot be matched across JAX and torch, so no lane compares draws seed-for-seed:

* lane T  (teacher-forced): aminx samples with the reference's decoding order injected through
  ``decoding_order_fn``; the reference then scores the SAME sequences under the SAME order; per-position
  conditional log-probabilities are compared.  Sees: fixed/redesigned residues, ligand context, checkpoint.
* lane D  (distributional): the first designed decoding step has identical context on both sides, so
  aminx's empirical token frequencies over N draws are tested (chi-square) against the reference's exact
  ``sampling_probs`` at that position.  Sees: temperature, omit_AA, bias_AA (the stored logits are
  bias-free, so lane T cannot).
* lane C  (collapsed sampler): temperature 1e-3 on both sides with the same injected order; tokens must
  match exactly up to the first near-tie (reported).  Full-sequence check of the whole decode loop.
* lane S  (score path): aminx ``score()`` is FULL-context teacher forcing (every position but itself
  visible).  The oracle is the reference's own decoder layers run with that explicit mask.  The stock
  ``single_aa_score`` is NOT that definition (its logits depend on the injected randn), which is the
  lane's negative control.
* lane CLI: ``aminx run sample --emit-json`` must carry every CLI-reachable knob through to the spec.

Every lane has a must-fail control (kind="control"); lane D has an instrument cell that measures the
test's false-reject rate and power on synthetic draws.

Resume: one process per cell; ``.started``/``.done`` stamps; a cell is reused only when the sha256 of
its inputs (code commit, script, weights, reference head, knobs) and of its result file match.  The
reference stage is persisted as ``fixtures/<cell>.ref.npz`` and listed, with its hash, in the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("e2e_run_api_parity")

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PDB_NAME = "1BC8.pdb"
CHAIN = "C"
ORDER_SEED = 7  # seeds the injected decoding-order randn (both sides consume the same row)
SAMPLE_SEED = 3
DIST_SEED = 5
# aminx pads every per-position array to RunSpecification.max_length (default 512) and the AR decode runs at the
# PADDED length, so a 93-residue chain at 512 costs ~86 s/sample on CPU (measured: 4 samples = ~345 s, warm or cold;
# compile is not the cost).  The experiment buckets to the smallest multiple of 32 that fits chain C (93 -> 96).
# That is itself a declared knob: one cell (pmpnn__pad512) runs the default 512 path.
N_PAD = 96
PAD_DEFAULT = 512
N_TEACHER = 4
N_DIST = 512
N_DIST_SMOKE = 32
COLLAPSED_T = 1e-3

# Pre-registered thresholds (sidecar [outcomes] refer to these by name).
TOL_LOGPROB = 1e-3  # lane T: max |log p_aminx - log p_ref| at designed positions
TOL_SCORE = 1e-3  # lane S: max |logit diff| and |NLL diff|
ALPHA = 1e-3  # lane D: chi-square rejection level, per cell
TIE_GAP = 1e-2  # lane C: top1-top2 gap under which a position is a near-tie
REJECT_T = 1e-2  # control T: rejected when max |diff| exceeds this
REJECT_S = 2e-2  # control S: rejected when mean |logit diff| vs the stock single_aa_score exceeds this
INSTRUMENT_FALSE_REJECT_MAX = 0.005
INSTRUMENT_POWER_MIN = 0.9
INSTRUMENT_TV = 0.10

MODELS: dict[str, dict[str, Any]] = {
  "pmpnn": {"model_type": "protein_mpnn", "checkpoint": "proteinmpnn_v_48_020",
            "ref_flag": "--checkpoint_protein_mpnn", "ligand": False},
  "lmpnn": {"model_type": "ligand_mpnn", "checkpoint": "ligandmpnn_v_32_020_25",
            "ref_flag": "--checkpoint_ligand_mpnn", "ligand": True},
}

# knob set -> how each side is configured.  "ref" are run.py flags; aminx side is derived from the
# reference's own feature dict (chain_mask, bias), so both sides are configured from ONE source of truth.
KNOBS: dict[str, dict[str, Any]] = {
  "base": {"ref": [], "temperature": None},
  "temp05": {"ref": ["--temperature", "0.5"], "temperature": 0.5},
  "omit_CW": {"ref": ["--omit_AA", "CW"], "temperature": None},
  "bias_AE": {"ref": ["--bias_AA", "A:1.5,E:-2.0"], "temperature": None},
  "fixed_1_10": {"ref": ["--fixed_residues", " ".join(f"{CHAIN}{i}" for i in range(1, 11))], "temperature": None},
  "redesign_20_60": {"ref": ["--redesigned_residues", " ".join(f"{CHAIN}{i}" for i in range(20, 61))],
                     "temperature": None},
  # The default padded length users actually run.  Too slow for the 512-draw lane (see N_PAD), so lanes T and C
  # only, at N_TEACHER samples.
  "pad512": {"ref": [], "temperature": None, "max_length": PAD_DEFAULT, "lanes": "TC"},
}


def _pad(knob: str) -> int:
  return int(KNOBS.get(knob, {}).get("max_length", N_PAD))


def _lanes(knob: str) -> str:
  return str(KNOBS.get(knob, {}).get("lanes", "TDC"))

DEFAULT_T = 0.1  # run.py and SamplingSpecification both default to 0.1

# Knobs the reference's run.py exposes that this experiment does NOT vary, stated not omitted.
KNOBS_DECLARED_UNVARIED = (
  "batch_size", "number_of_batches", "symmetry_residues", "symmetry_weights", "chains_to_design",
  "bias_AA_per_residue", "omit_AA_per_residue", "ligand_mpnn_cutoff_for_score",
  "ligand_mpnn_use_atom_context", "ligand_mpnn_use_side_chain_context", "pack_side_chains",
  "global_transmembrane_label", "checkpoints: solublempnn/membrane/sc", "structures other than 1BC8 chain C",
  "ligand parsing/atom selection/cutoff from PDB HETATM (aminx is GIVEN the reference's featurized per-residue "
  "ligand windows through ligand_context_path, because ligand_conditioning=True refuses to run without them)",
)
KNOBS_VARIED = ("temperature", "omit_AA", "bias_AA", "fixed_residues", "redesigned_residues",
                "model_type+checkpoint (protein_mpnn, ligand_mpnn)", "decoding order (injected)",
                "max_length (96 bucket everywhere; the default 512 in one protein_mpnn cell, lanes T and C only)")

KNOB_CELLS = {f"{m}__{k}": (m, k) for m in MODELS for k in KNOBS if k != "pad512"}
KNOB_CELLS["pmpnn__pad512"] = ("pmpnn", "pad512")
SCORE_CELLS = {f"{m}__score": (m, "score") for m in MODELS}
CONTROL_CELLS = {
  "ctl_T_wrong_order": ("pmpnn", "fixed_1_10"),
  "ctl_D_wrong_temperature": ("pmpnn", "temp05"),
  "ctl_C_wrong_bias": ("pmpnn", "base"),
  "ctl_S_stock_single_aa": ("pmpnn", "score"),
}
OTHER_CELLS = ("cli_equivalence", "instrument")
ALL_CELLS = (*KNOB_CELLS, *SCORE_CELLS, *CONTROL_CELLS, *OTHER_CELLS)

CLI_KNOBS = {"checkpoint_id": "proteinmpnn_v_48_020", "num_samples": 4, "temperature": 0.3,
             "chain_id": CHAIN, "random_seed": 5, "max_length": N_PAD}


# --------------------------------------------------------------------------- hashing / provenance
def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
  return _sha256_bytes(path.read_bytes())


def _ref_root() -> Path:
  root = os.environ.get("REFERENCE_PATH")
  if not root or not Path(root).is_dir():
    msg = "REFERENCE_PATH must point at a LigandMPNN checkout (e.g. /home/solab/bv/ref/LigandMPNN)"
    raise RuntimeError(msg)
  return Path(root)


def _ref_head(root: Path) -> str:
  try:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
  except Exception:  # noqa: BLE001 - provenance only
    return "unknown"


def _weights_sha(checkpoint: str) -> str:
  from aminx.io.weights import weight_provenance

  return weight_provenance(checkpoint).sha256


def _inputs_hash(cell: str, code_commit: str, smoke: bool) -> str:
  payload: dict[str, Any] = {"cell": cell, "code_commit": code_commit, "smoke": smoke,
                             "script_sha256": _sha256_file(Path(__file__)), "thresholds":
                             [TOL_LOGPROB, TOL_SCORE, ALPHA, TIE_GAP, N_DIST, N_TEACHER]}
  spec = KNOB_CELLS.get(cell) or SCORE_CELLS.get(cell) or CONTROL_CELLS.get(cell)
  if spec:
    model, knob = spec
    payload["model"] = MODELS[model]
    payload["knob"] = KNOBS.get(knob, knob)
    payload["weights_sha256"] = "smoke" if smoke else _weights_sha(MODELS[model]["checkpoint"])
    payload["ref_head"] = _ref_head(_ref_root())
  return _sha256_bytes(json.dumps(payload, sort_keys=True).encode())


# --------------------------------------------------------------------------- reference side
def _np_aliases() -> None:
  # upstream's openfold predates the numpy-2 removal of the np.int/float/... aliases
  for name, typ in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)):
    if name not in np.__dict__:
      setattr(np, name, typ)


def _import_reference_torch():
  _np_aliases()
  root = _ref_root()
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  import torch  # noqa: PLC0415

  import model_utils  # noqa: PLC0415  (upstream)

  return torch, model_utils, root


def ref_context(model_key: str, knob: str, *, temperature: float | None = None, batch: int = 1) -> dict[str, Any]:
  """Run upstream ``run.py`` in-process with randn injected; return its feature dict, outputs and model."""
  torch, model_utils, root = _import_reference_torch()
  spec = MODELS[model_key]
  knobs = KNOBS[knob]
  cap: dict[str, Any] = {}
  orig = model_utils.ProteinMPNN.sample

  def patched(self, feature_dict):  # noqa: ANN001
    length = feature_dict["mask"].shape[1]
    row = torch.randn((1, length), generator=torch.Generator().manual_seed(ORDER_SEED))
    feature_dict["randn"] = row.repeat(feature_dict["batch_size"], 1)
    out = orig(self, feature_dict)
    cap.update(model=self, fd=feature_dict, out=out, randn_row=row)
    return out

  model_utils.ProteinMPNN.sample = patched
  cwd = os.getcwd()
  try:
    os.chdir(root)
    argv = ["run.py", "--model_type", spec["model_type"], spec["ref_flag"],
            f"{root}/model_params/{spec['checkpoint']}.pt", "--pdb_path", f"{root}/inputs/{PDB_NAME}",
            "--out_folder", tempfile.mkdtemp(prefix="e2e_ref_"), "--seed", "11", "--batch_size", str(batch),
            "--number_of_batches", "1", "--parse_these_chains_only", CHAIN, *knobs["ref"]]
    if temperature is not None:  # argparse keeps the LAST --temperature, so an override beats the knob set's own
      argv += ["--temperature", str(temperature)]
    old_argv, sys.argv = sys.argv, argv
    try:
      runpy.run_path(f"{root}/run.py", run_name="__main__")
    finally:
      sys.argv = old_argv
  finally:
    os.chdir(cwd)
    model_utils.ProteinMPNN.sample = orig
  if spec["ligand"] and not all(k in cap["fd"] for k in ("Y", "Y_t", "Y_m")):
    msg = "ligand_mpnn reference run exposed no ligand tensors (Y/Y_t/Y_m); cannot build a ligand context"
    raise RuntimeError(msg)
  return cap


def ligand_npz(ctx: dict[str, Any], pad: int) -> str:
  """A ligand_context_path file holding the reference's own per-residue ligand windows, zero-padded to ``pad``.

  aminx's loader wants Y (N, L, M, 3), Y_t and Y_m (N, L, M) at the PADDED length plus structure_ids.  Handing it
  the reference's featurized windows means ligand PARSING and atom selection are not compared (declared).
  """
  cache = ctx.setdefault("_ligand_npz", {})
  if pad not in cache:
    fd = ctx["fd"]
    arrays = {k: fd[k].numpy() for k in ("Y", "Y_t", "Y_m")}
    width = pad - arrays["Y"].shape[1]

    def pad_len(a: np.ndarray) -> np.ndarray:
      return np.pad(a, [(0, 0), (0, width)] + [(0, 0)] * (a.ndim - 2))

    path = Path(tempfile.mkdtemp(prefix="e2e_ligand_")) / f"{PDB_NAME[:-4]}.ligand.npz"
    np.savez(path, Y=pad_len(arrays["Y"]).astype(np.float32), Y_t=pad_len(arrays["Y_t"]).astype(np.int32),
             Y_m=pad_len(arrays["Y_m"]).astype(np.float32), structure_ids=np.array([PDB_NAME[:-4]]))
    cache[pad] = str(path)
  return cache[pad]


def ref_full_context_logits(model, fd, seq_tensor):  # noqa: ANN001
  """Reference decoder layers with an explicit full-context mask: every neighbour except self visible."""
  torch, model_utils, _ = _import_reference_torch()
  fd2 = dict(fd)
  fd2["S"] = seq_tensor
  with torch.no_grad():
    h_v, h_e, e_idx = model.encode(fd2)
    length = e_idx.shape[1]
    not_self = (e_idx != torch.arange(length)[None, :, None]).float().unsqueeze(-1)
    mask = fd2["mask"]
    mask_1d = mask.view([1, length, 1, 1])
    mask_bw, mask_fw = mask_1d * not_self, mask_1d * (1.0 - not_self)
    h_s = model.W_s(fd2["S"])
    h_es = model_utils.cat_neighbors_nodes(h_s, h_e, e_idx)
    h_ex_enc = model_utils.cat_neighbors_nodes(torch.zeros_like(h_s), h_e, e_idx)
    h_exv_enc = model_utils.cat_neighbors_nodes(h_v, h_ex_enc, e_idx)
    h_exv_fw = mask_fw * h_exv_enc
    for layer in model.decoder_layers:
      h_esv = model_utils.cat_neighbors_nodes(h_v, h_es, e_idx)
      h_esv = mask_bw * h_esv + h_exv_fw
      h_v = layer(h_v, h_esv, mask)
    return model.W_out(h_v).numpy()[0]


def _lsm(x: np.ndarray) -> np.ndarray:
  x = x.astype(np.float64)
  m = x.max(-1, keepdims=True)
  return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


# --------------------------------------------------------------------------- aminx side
_ORDER_FNS: dict[bytes, Any] = {}


def _order_fn(order: np.ndarray, length: int):
  """One function object per distinct order.

  ``decoding_order_fn`` is a STATIC field of the compiled sampler, so a fresh closure per call forces
  a retrace and recompile even when the order is unchanged.  The order is a constant of the program, so
  a different order is a legitimately different program; the same order must reuse the same object.
  """
  import jax.numpy as jnp

  key_bytes = np.asarray(order, dtype=np.int32).tobytes() + length.to_bytes(4, "little")
  if key_bytes in _ORDER_FNS:
    return _ORDER_FNS[key_bytes]

  def fn(key, n, tie_map=None, num_groups=None):  # noqa: ANN001, ARG001
    tail = jnp.arange(length, n, dtype=jnp.int32)
    return jnp.concatenate([jnp.asarray(order, dtype=jnp.int32), tail]), key

  _ORDER_FNS[key_bytes] = fn
  return fn


def aminx_kwargs(model_key: str, ctx: dict[str, Any], *, temperature: float, bias_override: np.ndarray | None = None,
                 pad: int = N_PAD) -> dict[str, Any]:
  """aminx spec kwargs derived from the reference's feature dict, so both sides share one source of truth."""
  root = _ref_root()
  fd = ctx["fd"]
  chain_mask = fd["chain_mask"][0].numpy()
  s_native = fd["S"][0].numpy()
  kw: dict[str, Any] = {"inputs": [str(root / "inputs" / PDB_NAME)], "checkpoint_id": MODELS[model_key]["checkpoint"],
                        "chain_id": CHAIN, "temperature": temperature, "max_length": pad}
  length = chain_mask.shape[0]
  fixed = chain_mask == 0
  if fixed.any():
    fm = np.zeros(pad, np.float32)
    ft = np.zeros(pad, np.int32)
    fm[:length][fixed] = 1.0
    ft[:length][fixed] = s_native[fixed]
    kw["fixed_mask"], kw["fixed_tokens"] = fm, ft
  ref_bias = fd["bias"][0].numpy()  # (L, 21) = -1e8*omit + bias_AA
  bias = np.zeros((pad, 21), np.float32)
  bias[:length] = ref_bias
  if bias_override is not None:
    bias = bias_override
  if np.any(bias):
    kw["bias"] = bias
  if MODELS[model_key]["ligand"]:
    kw["ligand_conditioning"] = True
    kw["ligand_context_path"] = ligand_npz(ctx, pad)
  return kw


def aminx_sample(kw: dict[str, Any], *, num_samples: int, seed: int, order: np.ndarray, length: int):
  from aminx.run import SamplingSpecification, sample

  res = sample(SamplingSpecification(**kw, num_samples=num_samples, random_seed=seed,
                                     decoding_order_fn=_order_fn(order, length)))
  seqs = np.asarray(res["sequences"])[0, :, 0, 0, :length]
  logits = np.asarray(res["logits"])[0, :, 0, 0, :length, :]
  return seqs, logits


# --------------------------------------------------------------------------- statistics
def chisq_test(counts: np.ndarray, probs: np.ndarray, *, min_expected: float = 5.0) -> dict[str, Any]:
  from scipy.stats import chi2

  n = float(counts.sum())
  exp = n * probs
  keep = exp >= min_expected
  obs_b = np.append(counts[keep], counts[~keep].sum())
  exp_b = np.append(exp[keep], exp[~keep].sum())
  if exp_b[-1] <= 0:
    if obs_b[-1] > 0:
      # draws landed on tokens the reference gives EXACTLY zero probability (omit_AA, underflow): that is a
      # rejection, not an empty bin to drop -- dropping it would let a forbidden token pass lane D unseen.
      return {"chi2": None, "dof": max(len(obs_b) - 1, 1), "p_value": 0.0,
              "tv": float(0.5 * np.abs(counts / n - probs).sum()), "n_bins": int(len(obs_b)),
              "forbidden_token_draws": float(obs_b[-1])}
    obs_b, exp_b = obs_b[:-1], exp_b[:-1]
  stat = float(((obs_b - exp_b) ** 2 / np.maximum(exp_b, 1e-12)).sum())
  dof = max(len(obs_b) - 1, 1)
  return {"chi2": stat, "dof": dof, "p_value": float(chi2.sf(stat, dof)),
          "tv": float(0.5 * np.abs(counts / n - probs).sum()), "n_bins": int(len(obs_b))}


# --------------------------------------------------------------------------- lanes
def _designed(fd) -> np.ndarray:  # noqa: ANN001
  return (fd["chain_mask"][0] * fd["mask"][0]).numpy() > 0


def shared_sample(model_key: str, knob: str, ctx: dict[str, Any], *, n_draws: int, temperature_scale: float = 1.0,
                  order_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
  """The one aminx sample() call that lanes T and D share (same spec, temperature, order, sample count)."""
  fd, out = ctx["fd"], ctx["out"]
  order = out["decoding_order"][0].numpy() if order_override is None else order_override
  temperature = (KNOBS[knob]["temperature"] or DEFAULT_T) * temperature_scale
  return aminx_sample(aminx_kwargs(model_key, ctx, temperature=temperature, pad=_pad(knob)), num_samples=n_draws,
                      seed=DIST_SEED, order=order, length=fd["mask"].shape[1])


def lane_teacher(ctx: dict[str, Any], seqs: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
  """Teacher-force the reference on the first N_TEACHER of aminx's own samples, under the injected order."""
  torch, _, _ = _import_reference_torch()
  fd, model = ctx["fd"], ctx["model"]
  designed = _designed(fd)
  worst = 0.0
  for i in range(N_TEACHER):
    fd2 = dict(fd)
    fd2["S"] = torch.as_tensor(seqs[i : i + 1].astype(np.int64))
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    with torch.no_grad():
      ref_lp = model.score(fd2, use_sequence=True)["log_probs"].numpy()[0]
    diff = np.abs(_lsm(logits[i])[designed, :20] - ref_lp[designed, :20])
    worst = max(worst, float(diff.max()))
  s_native = fd["S"][0].numpy()
  fixed_ok = bool((seqs[:, ~designed] == s_native[~designed]).all()) if (~designed).any() else True
  return {"max_abs_logprob": worst, "n_designed": int(designed.sum()), "fixed_tokens_ok": fixed_ok,
          "passed": bool(worst <= TOL_LOGPROB and fixed_ok)}


def lane_dist(ctx: dict[str, Any], seqs: np.ndarray) -> dict[str, Any]:
  """Chi-square of aminx's first-designed-step token counts against the reference's exact sampling_probs."""
  fd, out = ctx["fd"], ctx["out"]
  order = out["decoding_order"][0].numpy()
  n_fixed = int((fd["chain_mask"][0] * fd["mask"][0]).numpy().size - _designed(fd).sum())
  pos0 = int(order[n_fixed])  # first DESIGNED step: identical context on both sides
  probs = out["sampling_probs"][0, pos0].numpy().astype(np.float64)
  probs = probs / probs.sum()
  counts = np.bincount(seqs[:, pos0].astype(int), minlength=21)[:20].astype(np.float64)
  res = chisq_test(counts, probs)
  res.update({"position": pos0, "n_draws": int(seqs.shape[0]), "ref_top_prob": float(probs.max()),
              "passed": bool(res["p_value"] > ALPHA)})
  return res


def lane_collapsed(model_key: str, knob: str, *, bias_poke: bool = False) -> dict[str, Any]:
  ctx = ref_context(model_key, knob, temperature=COLLAPSED_T)
  fd, out = ctx["fd"], ctx["out"]
  length = fd["mask"].shape[1]
  order = out["decoding_order"][0].numpy()
  designed = _designed(fd)
  ref_seq = out["S"][0].numpy()
  bias_override = None
  if bias_poke:  # negative control: forbid the reference's first-designed-step token on the aminx side only
    base = aminx_kwargs(model_key, ctx, temperature=COLLAPSED_T, pad=_pad(knob)).get("bias")
    bias_override = np.zeros((_pad(knob), 21), np.float32) if base is None else np.array(base)
    pos_first = int(order[int((~designed).sum())])
    bias_override[pos_first, int(ref_seq[pos_first])] = -1e8
  kw = aminx_kwargs(model_key, ctx, temperature=COLLAPSED_T, bias_override=bias_override, pad=_pad(knob))
  seqs, _ = aminx_sample(kw, num_samples=1, seed=SAMPLE_SEED, order=order, length=length)
  # first near-tie in decode order (ties make later contexts legitimately diverge)
  lp = out["log_probs"][0].numpy() + fd["bias"][0].numpy()
  lp = lp[:, :20]
  top2 = np.sort(lp, axis=-1)[:, -2:]
  gap = top2[:, 1] - top2[:, 0]
  first_tie_rank = length
  for rank, pos in enumerate(order):
    if designed[pos] and gap[pos] < TIE_GAP:
      first_tie_rank = rank
      break
  compared = [int(p) for r, p in enumerate(order) if r < first_tie_rank and designed[p]]
  mism = [p for p in compared if int(seqs[0, p]) != int(ref_seq[p])]
  return {"n_compared": len(compared), "first_tie_rank": int(first_tie_rank), "n_mismatch": len(mism),
          "first_mismatch_position": mism[0] if mism else None, "min_gap": float(gap[designed].min()),
          "passed": bool(len(mism) == 0 and len(compared) > 0)}


def lane_score(model_key: str, ctx: dict[str, Any], *, vs_stock: bool) -> dict[str, Any]:
  torch, _, _ = _import_reference_torch()
  from aminx.run import ScoringSpecification, score

  fd, model, out = ctx["fd"], ctx["model"], ctx["out"]
  length = fd["mask"].shape[1]
  native = fd["S"][0].numpy()
  other = out["S"][0].numpy()
  strings = ["".join(ALPHABET[i] for i in s) for s in (native, other)]
  kw: dict[str, Any] = {"inputs": [str(_ref_root() / "inputs" / PDB_NAME)],
                        "checkpoint_id": MODELS[model_key]["checkpoint"], "chain_id": CHAIN,
                        "sequences_to_score": strings, "return_logits": True, "max_length": N_PAD}
  if MODELS[model_key]["ligand"]:
    kw["ligand_conditioning"] = True
    kw["ligand_context_path"] = ligand_npz(ctx, N_PAD)
  res = score(ScoringSpecification(**kw))
  lg = np.asarray(res["logits"])[0]  # (n_seq, N_PAD, 21) raw logits
  scores = np.asarray(res["scores"])[0]
  worst_logit, worst_nll, mean_logit = 0.0, 0.0, []
  for i, seq in enumerate((native, other)):
    tensor = torch.as_tensor(seq[None].astype(np.int64))
    if vs_stock:
      fd2 = dict(fd)
      fd2["S"], fd2["batch_size"] = tensor, 1
      fd2["randn"] = ctx["randn_row"]
      with torch.no_grad():
        ref_lg = model.single_aa_score(fd2, use_sequence=False)["logits"].numpy()[0]
    else:
      ref_lg = ref_full_context_logits(model, fd, tensor)
    d = np.abs(lg[i, :length, :20] - ref_lg[:, :20])
    worst_logit = max(worst_logit, float(d.max()))
    mean_logit.append(float(d.mean()))
    ref_nll = float(-_lsm(ref_lg)[np.arange(length), seq].mean())
    worst_nll = max(worst_nll, abs(ref_nll - float(scores[i])))
  record = {"max_abs_logit": worst_logit, "mean_abs_logit": float(np.mean(mean_logit)),
            "max_abs_nll": worst_nll, "oracle": "stock single_aa_score(use_sequence=False)" if vs_stock
            else "reference decoder layers, explicit full-context mask (every neighbour but self)"}
  record["passed"] = bool(worst_logit <= TOL_SCORE and worst_nll <= TOL_SCORE)
  return record


def lane_cli() -> dict[str, Any]:
  """`aminx run sample --emit-json` must carry every CLI-reachable knob into the spec."""
  root = _ref_root()
  # Base options (checkpoint, chain, seed, max_length, ...) live on the `run` GROUP; sampling options on `sample`.
  cmd = [sys.executable, "-c", "from aminx.cli import main; main()", "run",
         "--checkpoint-id", str(CLI_KNOBS["checkpoint_id"]), "--chain-id", str(CLI_KNOBS["chain_id"]),
         "--random-seed", str(CLI_KNOBS["random_seed"]), "--max-length", str(CLI_KNOBS["max_length"]),
         "sample", "--inputs", str(root / "inputs" / PDB_NAME),
         "--num-samples", str(CLI_KNOBS["num_samples"]), "--temperature", str(CLI_KNOBS["temperature"]),
         "--emit-json"]
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  record: dict[str, Any] = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-400:]}
  if proc.returncode != 0:
    record["passed"] = False
    return record
  start = proc.stdout.find("{")
  payload = json.loads(proc.stdout[start:])

  def find(obj: Any, key: str) -> list[Any]:  # noqa: ANN401
    found: list[Any] = []
    if isinstance(obj, dict):
      for k, v in obj.items():
        if k == key:
          found.append(v)
        found += find(v, key)
    elif isinstance(obj, list):
      for v in obj:
        found += find(v, key)
    return found

  def norm(v: Any) -> Any:  # noqa: ANN401
    while isinstance(v, list) and len(v) == 1:
      v = v[0]
    return v

  results, ok = {}, True
  for knob, expected in CLI_KNOBS.items():
    values = [norm(v) for v in find(payload, knob) if v is not None]
    hit = any(v == expected or (isinstance(v, (int, float)) and isinstance(expected, (int, float))
                                and abs(v - expected) < 1e-9) for v in values)
    results[knob] = {"expected": expected, "found": values[:3], "ok": hit}
    ok &= hit
  record["knobs"] = results
  record["passed"] = bool(ok)
  return record


def lane_instrument(seed: int = 0) -> dict[str, Any]:
  """False-reject rate and power of lane D's chi-square test on synthetic draws (positive + null)."""
  rng = np.random.default_rng(seed)
  reps = 2000
  false_rejects = 0
  for _ in range(reps):
    p = rng.dirichlet(np.full(20, 0.5))
    counts = rng.multinomial(N_DIST, p).astype(np.float64)
    false_rejects += chisq_test(counts, p)["p_value"] <= ALPHA
  hits, used = 0, 0
  for _ in range(reps * 2):
    p = rng.dirichlet(np.full(20, 0.5))
    q = rng.dirichlet(np.full(20, 0.5))
    tv_pq = 0.5 * np.abs(q - p).sum()
    lam = INSTRUMENT_TV / tv_pq
    if lam > 1:
      continue
    p_alt = (1 - lam) * p + lam * q
    counts = rng.multinomial(N_DIST, p_alt).astype(np.float64)
    hits += chisq_test(counts, p)["p_value"] <= ALPHA
    used += 1
    if used >= reps:
      break
  fr, power = false_rejects / reps, hits / max(used, 1)
  return {"false_reject_rate": fr, "power_at_tv": power, "tv": INSTRUMENT_TV, "n_draws": N_DIST,
          "alpha": ALPHA, "reps": reps, "reps_alt": used,
          "passed": bool(fr <= INSTRUMENT_FALSE_REJECT_MAX and power >= INSTRUMENT_POWER_MIN)}


# --------------------------------------------------------------------------- cells
def _persist_fixture(out_dir: Path, cell: str, ctx: dict[str, Any]) -> dict[str, str]:
  fx = out_dir / "fixtures"
  fx.mkdir(parents=True, exist_ok=True)
  path = fx / f"{cell}.ref.npz"
  fd, out = ctx["fd"], ctx["out"]
  extra: dict[str, str] = {}
  if "Y" in fd:  # persist the exact ligand windows both sides were given (the padded file aminx reads)
    lig_path = fx / f"{cell}.ligand.npz"
    lig_path.write_bytes(Path(ligand_npz(ctx, N_PAD)).read_bytes())
    extra["ligand_path"], extra["ligand_sha256"] = str(lig_path), _sha256_file(lig_path)
  np.savez_compressed(
    path, randn=ctx["randn_row"].numpy(), S=out["S"].numpy(), log_probs=out["log_probs"].numpy(),
    sampling_probs=out["sampling_probs"].numpy(), decoding_order=out["decoding_order"].numpy(),
    chain_mask=fd["chain_mask"].numpy(), mask=fd["mask"].numpy(), bias=fd["bias"].numpy(), native=fd["S"].numpy())
  return {"path": str(path), "sha256": _sha256_file(path), **extra}


def _run_cell_body(cell: str, out_dir: Path, smoke: bool) -> dict[str, Any]:
  n_draws = N_DIST_SMOKE if smoke else N_DIST
  if cell == "instrument":
    return {"kind": "instrument", "instrument": lane_instrument()}
  if cell == "cli_equivalence":
    return {"kind": "cli", "cli": lane_cli()}
  if cell in KNOB_CELLS:
    model_key, knob = KNOB_CELLS[cell]
    ctx = ref_context(model_key, knob, temperature=KNOBS[knob]["temperature"])
    fixture = _persist_fixture(out_dir, cell, ctx)
    t0 = time.perf_counter()
    wanted = _lanes(knob)
    n_call = n_draws if "D" in wanted else N_TEACHER
    seqs, logits = shared_sample(model_key, knob, ctx, n_draws=n_call)  # ONE aminx call feeds lanes T and D
    LOG.info("%s: aminx sample(%d, pad %d) %.1fs", cell, n_call, _pad(knob), time.perf_counter() - t0)
    lanes = {"T": lane_teacher(ctx, seqs, logits)}
    if "D" in wanted:
      lanes["D"] = lane_dist(ctx, seqs)
    LOG.info("%s: lanes %s done %.1fs", cell, "T,D" if "D" in wanted else "T", time.perf_counter() - t0)
    lanes["C"] = lane_collapsed(model_key, knob)
    LOG.info("%s: lane C done %.1fs", cell, time.perf_counter() - t0)
    return {"kind": "knob", "model": model_key, "knob": knob, "fixture": fixture, "lanes": lanes,
            "passed": all(v["passed"] for v in lanes.values())}
  if cell in SCORE_CELLS:
    model_key, _ = SCORE_CELLS[cell]
    ctx = ref_context(model_key, "base")
    fixture = _persist_fixture(out_dir, cell, ctx)
    lane = lane_score(model_key, ctx, vs_stock=False)
    return {"kind": "score", "model": model_key, "fixture": fixture, "lanes": {"S": lane}, "passed": lane["passed"]}
  model_key, knob = CONTROL_CELLS[cell]
  if cell == "ctl_T_wrong_order":
    ctx = ref_context(model_key, knob)
    order = ctx["out"]["decoding_order"][0].numpy()[::-1].copy()
    seqs, logits = shared_sample(model_key, knob, ctx, n_draws=N_TEACHER, order_override=order)
    lane = lane_teacher(ctx, seqs, logits)  # the reference still scores under ITS order; aminx used the reversed one
    lane["rejected"] = bool(lane["max_abs_logprob"] > REJECT_T)
  elif cell == "ctl_D_wrong_temperature":
    ctx = ref_context(model_key, knob, temperature=KNOBS[knob]["temperature"])
    seqs, _ = shared_sample(model_key, knob, ctx, n_draws=n_draws, temperature_scale=2.0)
    lane = lane_dist(ctx, seqs)
    lane["rejected"] = bool(lane["p_value"] <= ALPHA)
  elif cell == "ctl_C_wrong_bias":
    lane = lane_collapsed(model_key, knob, bias_poke=True)
    lane["rejected"] = bool(lane["n_mismatch"] > 0)
  else:  # ctl_S_stock_single_aa
    ctx = ref_context(model_key, "base")
    lane = lane_score(model_key, ctx, vs_stock=True)
    lane["rejected"] = bool(lane["mean_abs_logit"] > REJECT_S)
  return {"kind": "control", "model": model_key, "lanes": {"control": lane}, "rejected": lane["rejected"]}


def _can_reuse(done: Path, result_path: Path, inputs_hash: str) -> bool:
  """A prior cell is reusable only if it COMPLETED (status ok) with matching inputs and an unmodified result file.

  An errored record is a harness failure (OOM, import error, a killed process that still wrote a record), not a
  measurement, so it is retried instead of being frozen in by its own stamp.
  """
  stamp = json.loads(done.read_text())
  return bool(stamp.get("status") == "ok" and stamp.get("inputs_hash") == inputs_hash
              and stamp.get("result_sha256") == _sha256_file(result_path))


def run_cell(cell: str, out_dir: Path, *, code_commit: str, smoke: bool) -> int:
  out_dir.mkdir(parents=True, exist_ok=True)
  result_path, started, done = out_dir / f"{cell}.json", out_dir / f"{cell}.started", out_dir / f"{cell}.done"
  inputs_hash = _inputs_hash(cell, code_commit, smoke)
  if done.exists() and result_path.exists():
    if _can_reuse(done, result_path, inputs_hash):
      print(json.dumps({"reused": cell, "from": str(result_path), **json.loads(done.read_text())}))
      return 0
    LOG.warning("stale, mismatched or errored stamp for %s; recomputing", cell)
  for stale in (done, result_path):
    stale.unlink(missing_ok=True)
  started.write_text(json.dumps({"cell": cell, "pid": os.getpid(), "t": time.time()}))
  record: dict[str, Any] = {"cell": cell, "smoke": smoke, "code_commit": code_commit}
  t0 = time.perf_counter()
  try:
    import jax

    import aminx

    record.update({"aminx_file": aminx.__file__, "jax_version": jax.__version__,
                   "device": str(jax.devices()[0]), "ref_head": _ref_head(_ref_root()),
                   "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR")})
    record.update(_run_cell_body(cell, out_dir, smoke))
    record["status"] = "ok"
  except BaseException as exc:  # noqa: BLE001 - a crash IS the measurement; record it
    record["status"] = "error"
    record["error_type"] = type(exc).__name__
    record["error_head"] = str(exc)[:600]
    record["traceback_tail"] = traceback.format_exc()[-1500:]
  record["seconds"] = round(time.perf_counter() - t0, 2)
  result_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
  done.write_text(json.dumps({"cell": cell, "inputs_hash": inputs_hash, "result_sha256": _sha256_file(result_path),
                              "status": record["status"]}))
  print(json.dumps({"cell": cell, "status": record["status"], "seconds": record["seconds"]}))
  return 0  # the record, not the exit code, carries the outcome


# --------------------------------------------------------------------------- verdict from files
def _cell_record(out_dir: Path, cell: str) -> dict[str, Any]:
  path = out_dir / f"{cell}.json"
  if not path.exists():
    return {"cell": cell, "state": "crashed_or_killed" if (out_dir / f"{cell}.started").exists() else "absent"}
  rec = json.loads(path.read_text())
  rec["state"] = rec.get("status", "error")
  # A verdict rests only on records that are (a) from a counted run, not --smoke, and (b) verified against their own
  # .done stamp, so a smoke record, a record without a stamp, or a hand-edited file cannot contribute to a pass.
  done = out_dir / f"{cell}.done"
  stamp_ok = False
  if done.exists():
    stamp = json.loads(done.read_text())
    stamp_ok = stamp.get("result_sha256") == _sha256_file(path)
  if rec.get("smoke") or not stamp_ok:
    rec["state"] = "invalid_record"
    rec["passed"] = None
    rec["rejected"] = None
  return rec


def aggregate(out_dir: Path) -> dict[str, Any]:
  recs = {cell: _cell_record(out_dir, cell) for cell in ALL_CELLS}
  real = [*KNOB_CELLS, *SCORE_CELLS]
  ok_cells = [c for c in real if recs[c]["state"] == "ok" and recs[c].get("passed") is True]
  failing = {c: {ln: {k: v for k, v in lane.items() if k in ("max_abs_logprob", "p_value", "tv", "n_mismatch",
                                                             "first_mismatch_position", "max_abs_logit",
                                                             "max_abs_nll", "fixed_tokens_ok", "passed")}
                 for ln, lane in (recs[c].get("lanes") or {}).items() if not lane.get("passed")}
             or {"state": recs[c]["state"], "error_head": recs[c].get("error_head")} for c in real if c not in ok_cells}
  controls = {c: bool(recs[c]["state"] == "ok" and recs[c].get("rejected") is True) for c in CONTROL_CELLS}
  cli = recs["cli_equivalence"]
  inst = recs["instrument"]
  commits = {r.get("code_commit") for r in recs.values() if r.get("code_commit")}
  return {
    "cells_ok": len(ok_cells) == len(real),
    "n_cells_ok": len(ok_cells),
    "n_cells_expected": len(real),
    "controls_rejected": all(controls.values()),
    "n_controls_rejected": sum(controls.values()),
    "instrument_ok": bool(inst["state"] == "ok" and inst.get("instrument", {}).get("passed") is True),
    "cli_ok": bool(cli["state"] == "ok" and cli.get("cli", {}).get("passed") is True),
    "single_code_commit": len(commits) <= 1,
    "knobs_varied": list(KNOBS_VARIED),
    "knobs_declared_unvaried": list(KNOBS_DECLARED_UNVARIED),
    "failing_cells": failing,
    "controls": controls,
    "cells": {c: {"state": r["state"], "passed": r.get("passed"), "rejected": r.get("rejected"),
                  "seconds": r.get("seconds")} for c, r in recs.items()},
  }


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--cell", choices=ALL_CELLS)
  parser.add_argument("--out-dir", type=Path, default=Path("outputs/results/e2e_run_api_parity"))
  parser.add_argument("--code-commit", default=os.environ.get("E2E_CODE_COMMIT", "unknown"))
  parser.add_argument("--dry-run", action="store_true", help="L1: cell table, reference path, imports")
  parser.add_argument("--smoke", action="store_true", help="L2: small draw counts (uncounted dev run)")
  parser.add_argument("--aggregate", type=Path, metavar="DIR", help="summarise a results dir; prints JSON")
  parser.add_argument("--out", type=Path, default=None, help="with --aggregate: also write the JSON to this file")
  args = parser.parse_args()
  if args.aggregate is not None:
    text = json.dumps(aggregate(args.aggregate), indent=2, sort_keys=True)
    if args.out is not None:  # bth evaluates [outcomes] from a result FILE; same JSON as stdout
      args.out.write_text(text)
    print(text)
    return 0
  if args.dry_run:
    root = _ref_root()
    import aminx

    print(json.dumps({"dry_run": "ok", "aminx": aminx.__file__, "ref": str(root), "ref_head": _ref_head(root),
                      "cells": list(ALL_CELLS)}))
    return 0
  if args.cell is None:
    parser.error("--cell is required unless --aggregate / --dry-run")
  return run_cell(args.cell, args.out_dir, code_commit=args.code_commit, smoke=args.smoke)


if __name__ == "__main__":
  sys.exit(main())
