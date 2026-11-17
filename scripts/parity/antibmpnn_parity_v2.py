"""AntiBMPNN parity vehicle v2: aminx against a SEALED dump of the real upstream.

Supersedes ``antibmpnn_parity.py`` (v1, kept untouched with its sidecar). What changed and why:

* The v1 dump drew the decoding-order ``randn`` under ``set_default_dtype(float64)`` in its f64 pass, so the f64
  and f32 passes used different orders and no cross-precision logit comparison was valid. The v2 dump
  (``dump_antibmpnn_oracle.py``) draws one float32 ``randn`` and one float32 order shared by both passes.
* The float32 band is no longer the flat 1e-4. The dump holds the REAL upstream in float32 AND float64 on the same
  inputs, so each stage has a measured conditioning floor ``floor = max|up_f32 - up_f64|`` (checkpoint 000 has a
  logit floor ~2.8e-4, one to two orders of magnitude above the later checkpoints). aminx float32 is graded
  against the float64 upstream with a tolerance that scales with that floor (see ``TOL_*``); a flat 1e-4 would
  fail the upstream's own float32 against its float64 at checkpoint 000.
* A new ``entry`` stage drives the REAL entry point ``aminx.host.runner.score`` (``ar_mask`` + ``return_logits``,
  published Hub weights) against the same dump, and against upstream ``ProteinMPNN.conditional_probs`` (the code
  behind ``--conditional_probs_only``), full-context and backbone-only.
* ``E_idx`` (neighbour indices) is graded in both precisions (``nbr_mismatch``).
* Mutants are labelled weight-level (``zero_edge_embedding``, ``swap_checkpoint_noise``: they edit the loaded
  parameters, so they show the comparison is sensitive to the weights, not that aminx code is exercised) or
  code-level (everything else: a perturbation of aminx source behaviour).

Stages (clean arm), aminx value vs upstream value:

* ``parse``       ``aminx.io.upstream_pdb.parse_pdb_upstream`` vs upstream ``parse_PDB`` (HuJ3, chain B).
* ``features`` / ``encoder`` / ``logits``  hand-composed forward (``_forward``) for the 11 ``.pt`` checkpoints
                  converted by ``convert_mpnn_checkpoint``, float32 and float64.
* ``entry``       ``runner.score`` on the published ``antibmpnn_v_48_{000,010,020,030}`` weights, teacher-forced
                  under the dump's decoding order, plus ``conditional_probs`` parity (checkpoint 000).
* ``sampler``     per-position sampling distribution and Gumbel-drawn tokens vs upstream ``ProteinMPNN.sample``
                  (two configs, float32), recomposed from the hand-composed forward logits with aminx's
                  ``compile_omit_aa_bias``.  ``runner.sample`` is NOT compared bit-wise: it draws its decoding
                  order and categorical noise from ``jax.random`` and has no hook to inject the dump's
                  ``randn``/Gumbel arrays (the lower-level ``inference.sample_autoregressive.kernel`` has a
                  ``gumbel_noise`` hook but needs a full InferenceBundle; not wired here), so sampled TOKENS cannot
                  be reproduced through the entry point.
* ``descriptors`` ``compute_cdr_physicochemical_descriptors`` vs ``peptides`` (CSV rounded to 4 dp and full precision).

Results go to ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
  os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "scripts", _REPO / "scripts" / "parity"):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from artifact_key import stable_artifact_key  # noqa: E402

log = logging.getLogger("antibmpnn_parity_v2")

SIDECAR = Path(__file__).with_suffix(".bth.toml")
CHECKPOINTS = tuple(f"antibmpnn_{n:03d}" for n in range(0, 101, 10))
PRIMARY = "antibmpnn_000"
PUBLISHED = {"antibmpnn_000": "v_48_000", "antibmpnn_010": "v_48_010", "antibmpnn_020": "v_48_020", "antibmpnn_030": "v_48_030"}
CONFIGS = ("cdr5", "knobs")
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PASS_F64 = 1e-7
PASS_SAMPLER = 1e-4
FAIL_MIN = 1e-3
#: float32 tolerance for one (checkpoint, stage) cell, scaled by that cell's measured conditioning floor
#: ``floor = max|upstream_f32 - upstream_f64|``.  Two independent float32 implementations of one function each sit
#: about one floor from the float64 truth, so ``3 x floor`` passes and ``10 x floor`` is marginal; the absolute
#: terms stop a cell with a vanishing floor from demanding sub-ulp agreement (2e-5 ~ 170 float32 ulp at 1).
TOL_PASS_MULT, TOL_PASS_ABS = 3.0, 2e-5
TOL_MARG_MULT, TOL_MARG_ABS = 10.0, 1e-4
#: mutant -> (designated stage, level).  ``weight`` mutants edit loaded parameters (they show the comparison is
#: sensitive to the weights, not that aminx code is exercised); ``code`` mutants perturb aminx behaviour.
MUTANTS = {
  "reversed_ar_mask": ("logits", "code"),
  "offset_shift": ("features", "code"),
  "entry_ar_mask_transposed": ("entry", "code"),
  "omit_penalty_dropped": ("sampler", "code"),
  "lehninger_pka_shifted": ("descriptors", "code"),
  "barley_tables_swapped": ("descriptors", "code"),
  "parser_wrong_atom_order": ("parse", "code"),
  "zero_edge_embedding": ("features", "weight"),
  "swap_checkpoint_noise": ("logits", "weight"),
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _pinned() -> dict[str, Any]:
  return tomllib.loads(SIDECAR.read_text(encoding="utf-8"))["data"]


def _maxabs(a: Any, b: Any) -> float:  # noqa: ANN401
  x = np.asarray(a, dtype=np.float64)
  y = np.asarray(b, dtype=np.float64)
  if x.shape != y.shape:
    msg = f"shape mismatch {x.shape} vs {y.shape}"
    raise ValueError(msg)
  return float(np.max(np.abs(x - y))) if x.size else 0.0


def _tols(floor: float) -> tuple[float, float]:
  return max(TOL_PASS_MULT * floor, TOL_PASS_ABS), max(TOL_MARG_MULT * floor, TOL_MARG_ABS)


def _verify_inputs(oracle_dir: Path, root: Path, weights_dir: Path) -> dict[str, str]:
  """Fail loudly unless every sealed input matches the sha256 pinned in the sidecar."""
  pinned = _pinned()
  checks = {
    oracle_dir / "antibmpnn_oracle.npz": pinned["oracle_dump_sha256"],
    oracle_dir / "antibmpnn_descriptors_oracle.npz": pinned["descriptors_dump_sha256"],
    root / "inputs" / "HuJ3" / "HuJ3.pdb": pinned["fixture_sha256"],
  }
  for name in CHECKPOINTS:
    checks[root / "antibmpnn_model_weights" / f"{name}.pt"] = pinned["checkpoints"][name]
  for version in PUBLISHED.values():
    checks[weights_dir / f"antibmpnn_{version}.eqx.zst"] = pinned["published"][f"antibmpnn_{version}"]
  weights: dict[str, str] = {}
  for path, want in checks.items():
    got = _sha256(path)
    if got != want:
      msg = f"{path} sha256 {got} != pinned {want}"
      raise SystemExit(msg)
    if path.suffix in (".pt", ".zst"):
      weights[stable_artifact_key(path)] = got
  return weights


# --------------------------------------------------------------------------------------------------
# aminx side
# --------------------------------------------------------------------------------------------------
def _forward(model: Any, x: Any, mask: Any, ridx: Any, cidx: Any, seq: Any, order: Any) -> dict[str, Any]:  # noqa: ANN401
  """features -> encoder -> teacher-forced conditional decoder, exposing every stage."""
  import jax
  import jax.numpy as jnp

  from aminx.model.decoder import conditional_decoder_layer_edge_features, pack_conditional_decoder_static_edges
  from aminx.utils import autoregression

  edge, nbr, _, _ = model.features(jax.random.PRNGKey(0), x, mask, ridx, cidx, backbone_noise=jnp.asarray(0.0, dtype=x.dtype))
  node, enc_edge = model.encoder(edge, nbr, mask, inference=True)
  one_hot = jax.nn.one_hot(seq, 21, dtype=x.dtype)
  ar = autoregression.ar_mask_from_decoding_order(order.astype(jnp.int32))
  seq_edge, mask_bw, masked = pack_conditional_decoder_static_edges(node, enc_edge, nbr, one_hot, model.w_s_embed.weight, ar, mask)
  h = node
  for layer in model.decoder.layers:
    h = layer(h, conditional_decoder_layer_edge_features(h, seq_edge, nbr, mask_bw, masked), mask, inference=True)
  logits = h @ model.w_out.weight.T + model.w_out.bias
  return {"edge": edge, "nbr": nbr, "node": node, "enc_edge": enc_edge, "logits": logits, "log_probs": jax.nn.log_softmax(logits, axis=-1)}


def _np_ar_mask(order: Any) -> np.ndarray:  # noqa: ANN401
  """``mask[i, j] == 1`` iff ``j`` is decoded before ``i`` (upstream ``order_mask_backward``), built without aminx."""
  order = np.asarray(order)
  rank = np.empty(order.shape[0], dtype=np.int64)
  rank[order] = np.arange(order.shape[0])
  return (rank[None, :] < rank[:, None]).astype(np.float32)


def _load_pt(root: Path, name: str, dtype: Any) -> Any:  # noqa: ANN401
  import jax
  import jax.numpy as jnp

  from convert_weights import convert_mpnn_checkpoint

  if os.environ.get("ANTIBMPNN_MUTANT") == "swap_checkpoint_noise" and name == PRIMARY:
    name = "antibmpnn_030"
  model = convert_mpnn_checkpoint(root / "antibmpnn_model_weights" / f"{name}.pt")
  import equinox as eqx

  model = jax.tree_util.tree_map(lambda v: v.astype(dtype) if eqx.is_inexact_array(v) else v, model)
  if os.environ.get("ANTIBMPNN_MUTANT") == "zero_edge_embedding":
    model = eqx.tree_at(lambda m: m.features.w_e.weight, model, jnp.zeros_like(model.features.w_e.weight))
  return model


def _inputs(d: Any, cfg: str, dtype: Any) -> tuple[Any, ...]:  # noqa: ANN401
  import jax.numpy as jnp

  g = lambda k: d[f"feat/{cfg}/{k}"][0]  # noqa: E731
  return (
    jnp.asarray(g("X"), dtype=dtype),
    jnp.asarray(g("mask"), dtype=dtype),
    jnp.asarray(g("residue_idx"), dtype=jnp.int32),
    jnp.asarray(g("chain_encoding_all"), dtype=jnp.int32),
    jnp.asarray(g("S"), dtype=jnp.int32),
  )


def _floors(d: Any, name: str, cfg: str = "cdr5") -> dict[str, float]:  # noqa: ANN401
  """Upstream float32-vs-float64 conditioning floor per stage for one checkpoint (dump only, no aminx)."""
  a, b = f"{name}/f32/{cfg}/", f"{name}/f64/{cfg}/"
  out = {"enc_node": _maxabs(d[a + "h_V_enc"], d[b + "h_V_enc"]), "logits": _maxabs(d[a + "log_probs_tf"], d[b + "log_probs_tf"])}
  if name == PRIMARY and cfg == "cdr5":
    out["features"] = _maxabs(d[a + "h_E0"], d[b + "h_E0"])
    out["enc_edge"] = _maxabs(d[a + "h_E_enc"], d[b + "h_E_enc"])
  return out


def _stage_forward(d: Any, root: Path, ckpts: tuple[str, ...], dtype: Any, prec: str) -> dict[str, Any]:  # noqa: ANN401
  """features / encoder / logits against the dump for one precision; per-cell gaps for the f32 floor rule."""
  import jax.numpy as jnp

  cells: list[dict[str, Any]] = []
  nbr_mismatch = 0
  cfg = "cdr5"
  for name in ckpts:
    model = _load_pt(root, name, dtype)
    x, mask, ridx, cidx, seq = _inputs(d, cfg, dtype)
    p = f"{name}/{prec}/{cfg}/"
    p64 = f"{name}/f64/{cfg}/"
    order = jnp.asarray(d[p + "order_tf"][0])
    r = _forward(model, x, mask, ridx, cidx, seq, order)
    nbr_mismatch += int(np.sum(np.asarray(r["nbr"]) != d[p + "E_idx"][0]))
    floors = _floors(d, name)
    pairs = {"enc_node": (r["node"], "h_V_enc"), "logits": (r["log_probs"], "log_probs_tf")}
    if name == PRIMARY:
      pairs["features"] = (r["edge"], "h_E0")
      pairs["enc_edge"] = (r["enc_edge"], "h_E_enc")
    for stage, (val, key) in pairs.items():
      cells.append({
        "ckpt": name, "stage": stage, "gap_same": _maxabs(val, d[p + key][0]),
        "gap_f64": _maxabs(val, d[p64 + key][0]), "floor": floors[stage],
      })
  return {"cells": cells, "nbr_mismatch": nbr_mismatch}


def _entry_logp(pdb: Path, checkpoint_id: str, seq: str, ar: Any, length: int) -> np.ndarray:  # noqa: ANN401
  """Per-position log-probs from the REAL entry point ``aminx.host.runner.score`` (``return_logits=True``)."""
  import jax
  import jax.numpy as jnp

  from aminx.host.runner import score

  kwargs: dict[str, Any] = {}
  if ar is not None:
    kwargs["ar_mask"] = ar
  result = score(
    inputs=str(pdb), checkpoint_id=checkpoint_id, sequences_to_score=[seq], return_logits=True, output_kind="nll",
    random_seed=0, **kwargs,
  )
  logits = np.asarray(result["logits"], dtype=np.float32).reshape(-1, 21)[:length]
  return np.asarray(jax.nn.log_softmax(jnp.asarray(logits), axis=-1))


def _stage_entry(d: Any, root: Path, ckpts: tuple[str, ...], *, with_cond: bool = True) -> dict[str, Any]:  # noqa: ANN401
  """``runner.score`` on the published weights vs the dump: teacher-forced (all given ckpts) and conditional_probs."""
  pdb = root / "inputs" / "HuJ3" / "HuJ3.pdb"
  seq = "".join(ALPHABET[int(i)] for i in d["feat/cdr5/S"][0])
  length = len(seq)
  cells: list[dict[str, Any]] = []
  for name in ckpts:
    if name not in PUBLISHED:
      continue
    ar = _np_ar_mask(d[f"{name}/f32/cdr5/order_tf"][0])
    lp = _entry_logp(pdb, f"antibmpnn_{PUBLISHED[name]}", seq, ar, length)
    cells.append({
      "ckpt": name, "stage": "entry_tf", "gap_same": _maxabs(lp, d[f"{name}/f32/cdr5/log_probs_tf"][0]),
      "gap_f64": _maxabs(lp, d[f"{name}/f64/cdr5/log_probs_tf"][0]), "floor": _floors(d, name)["logits"],
    })
  if PRIMARY in ckpts and with_cond:
    # --conditional_probs_only (Running_AntiBMPNN_run.py:271-282): upstream ProteinMPNN.conditional_probs
    floor = _floors(d, PRIMARY, "knobs")["logits"]
    # Upstream decodes each position LAST (protein_mpnn_utils.py:1330: order key (order_mask + 1e-4) * |randn|, order_mask
    # = e_i), so row i sees every other residue and no other position sees s_i.  aminx's default full-context mask is a
    # different estimand (every position sees s_i, which leaks back into row i through later layers), so each probed
    # position gets its own ar_mask built from the same stored randn and fed through runner.score.
    randn = np.asarray(d["condprobs/randn"][0], dtype=np.float32)
    probe = list(range(0, length, 10))
    full = np.zeros((length, 21), dtype=np.float32)
    for i in probe:
      key = (np.eye(length, dtype=np.float32)[i] + np.float32(0.0001)) * np.abs(randn)
      full[i] = _entry_logp(pdb, f"antibmpnn_{PUBLISHED[PRIMARY]}", seq, _np_ar_mask(np.argsort(key, kind="stable")), length)[i]
    bb =_entry_logp(pdb, f"antibmpnn_{PUBLISHED[PRIMARY]}", seq, np.zeros((length, length), dtype=np.float32), length)
    for stage, got, key in (("entry_cond_full", full, "condprobs/full"), ("entry_cond_backbone", bb, "condprobs/backbone_only")):
      rows = probe if stage == "entry_cond_full" else list(range(length))
      gap = _maxabs(got[rows], d[key][0][rows])
      cells.append({"ckpt": PRIMARY, "stage": stage, "gap_same": gap, "gap_f64": gap, "floor": floor})
  return {"cells": cells}


def _stage_sampler(d: Any, root: Path, dtype: Any) -> tuple[float, int]:  # noqa: ANN401
  """Sampling distribution + drawn tokens vs upstream ``ProteinMPNN.sample`` (float32 only)."""
  import jax
  import jax.numpy as jnp

  from aminx.host import omit_aa_bias

  model = _load_pt(root, PRIMARY, dtype)
  worst = 0.0
  token_mismatch = 0
  for cfg in CONFIGS:
    x, mask, ridx, cidx, _ = _inputs(d, cfg, dtype)
    p = f"{PRIMARY}/f32/{cfg}/"
    s_sample = jnp.asarray(d[p + "S_sample"][0], dtype=jnp.int32)
    order = jnp.asarray(d[p + "order_sample"][0])
    r = _forward(model, x, mask, ridx, cidx, s_sample, order)
    designed = (d[f"feat/{cfg}/chain_M"][0] * d[f"feat/{cfg}/chain_M_pos"][0] * d[f"feat/{cfg}/mask"][0]).astype(np.float32)
    temp = float(d[f"feat/{cfg}/temperature"])
    omit_letters = tuple(str(d[f"feat/{cfg}/omit"]))
    length = designed.shape[0]
    omit_mask = d[f"feat/{cfg}/omit_AA_mask"][0]
    per_pos = {int(i): "".join(ALPHABET[j] for j in np.nonzero(omit_mask[i])[0]) for i in range(length) if omit_mask[i].any()}
    bias = jnp.asarray(d[f"feat/{cfg}/bias_aa"], dtype=dtype)[None, :] + jnp.asarray(
      d[f"feat/{cfg}/bias_by_res_all"][0], dtype=dtype
    )
    fixed_row = jnp.asarray(1.0 - designed)
    # NB: cfg "cdr5" has no per-position omit; its omit_AA_mask is all zero (upstream passes None).
    total = omit_aa_bias.compile_omit_aa_bias(
      bias, fixed_mask_row=fixed_row, omit_aa=("".join(omit_letters),), omit_aa_per_position=per_pos or None, seq_len=length
    )
    probs = jax.nn.softmax((r["logits"] + total) / temp, axis=-1) * jnp.asarray(designed)[:, None]
    worst = max(worst, _maxabs(probs, d[p + "probs_sample"][0]))
    # token each step draws: argmax(log p + g[step]) at designed positions, native elsewhere
    g = d[p + "gumbel"]
    native = d[f"feat/{cfg}/S"][0]
    ord_np = np.asarray(order)
    lp = np.log(np.maximum(np.asarray(probs, dtype=np.float64), 1e-300))
    for step, pos in enumerate(ord_np):
      tok = int(np.argmax(lp[pos] + g[step])) if designed[pos] > 0 else int(native[pos])
      token_mismatch += int(tok != int(d[p + "S_sample"][0][pos]))
  return worst, token_mismatch


def _stage_descriptors(dd: Any) -> dict[str, float]:  # noqa: ANN401
  import jax.numpy as jnp

  from aminx.model import cdr_bias

  def tokens(seq: str) -> Any:  # noqa: ANN401
    """MPNN alphabet token ids for a one-letter sequence."""
    return jnp.asarray([ALPHABET.index(c) for c in seq], dtype=jnp.int32)

  csv = {"PD1_volume": 0.0, "PD2_hydro": 0.0, "PD3_charge": 0.0}
  for i, seq in enumerate(dd["csv/combination"]):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(tokens(str(seq)), ph=7.4)
    for key, val in (("PD1_volume", got.pd1_volume), ("PD2_hydro", got.pd2_hydro), ("PD3_charge", got.pd3_charge)):
      csv[key] = max(csv[key], abs(float(val) - float(dd[f"csv/{key}"][i])))
  direct = 0.0
  for i, seq in enumerate(dd["direct/seq"]):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(tokens(str(seq)), ph=7.4)
    direct = max(
      direct,
      abs(float(got.pd1_volume) - float(dd["direct/PD1"][i])),
      abs(float(got.pd2_hydro) - float(dd["direct/PD2"][i])),
      abs(float(got.pd3_charge) - float(dd["direct/charge74"][i])),
    )
  probe = tokens(str(dd["probe/seq"]))
  sweep = 0.0
  for ph, want in zip(dd["probe/ph"], dd["probe/charge"], strict=True):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(probe, ph=float(ph))
    sweep = max(sweep, abs(float(got.pd3_charge) - float(want)))
  return {"csv": max(csv.values()), "direct": direct, "ph_sweep": sweep}


def _stage_parse(d: Any, root: Path) -> float:  # noqa: ANN401
  from aminx.io import upstream_pdb

  parsed = upstream_pdb.parse_pdb_upstream(root / "inputs" / "HuJ3" / "HuJ3.pdb")[0]
  worst = 0.0
  for atom in ("N", "CA", "C", "O"):
    worst = max(worst, _maxabs(parsed["coords_chain_B"][f"{atom}_chain_B"], d[f"parse/{atom}_chain_B"]))
  if str(parsed["seq_chain_B"]) != str(d["parse/seq_chain_B"]):
    worst = max(worst, 1.0)
  return worst


def _apply_patch(mutant: str | None) -> None:
  if mutant in (None, "clean"):
    return
  if mutant not in MUTANTS:
    msg = f"unknown mutant {mutant}"
    raise SystemExit(msg)
  os.environ["ANTIBMPNN_MUTANT"] = mutant
  if mutant == "reversed_ar_mask":
    import jax.numpy as jnp

    from aminx.utils import autoregression

    original = autoregression.ar_mask_from_decoding_order
    autoregression.ar_mask_from_decoding_order = lambda order, *a, **k: original(jnp.flip(order, -1), *a, **k)
  elif mutant == "offset_shift":
    # code-level: every sequence-offset embedding index moves by one (an off-by-one in the relative-position one-hot)
    from aminx.model import features

    original_offsets = features.compute_neighbor_offsets
    features.compute_neighbor_offsets = lambda *a, **k: original_offsets(*a, **k) + 1
  elif mutant == "entry_ar_mask_transposed":
    # code-level: the real entry point hands the decoder the transposed visibility mask
    from aminx.host import runner

    original_pad = runner._pad_scoring_ar_mask  # noqa: SLF001
    runner._pad_scoring_ar_mask = lambda ar, *a, **k: original_pad(np.asarray(ar).T, *a, **k)  # noqa: SLF001
  elif mutant == "omit_penalty_dropped":
    from aminx.host import omit_aa_bias

    omit_aa_bias.compile_omit_aa_bias = lambda bias, **_k: bias
  elif mutant == "lehninger_pka_shifted":
    from aminx.model import cdr_bias

    # real pKa edits (a 1.4 shift on His moves its pH 7.4 charge from 0.04 to 0.5)
    for key, delta in (("H", 1.4), ("C", 1.0), ("N_term", -1.5)):
      cdr_bias.LEHNINGER_PKA[key] += delta
  elif mutant == "barley_tables_swapped":
    from aminx.model import cdr_bias

    cdr_bias._PD1_TABLE, cdr_bias._PD2_TABLE = cdr_bias._PD2_TABLE, cdr_bias._PD1_TABLE  # noqa: SLF001
  elif mutant == "parser_wrong_atom_order":
    from aminx.io import upstream_pdb

    original = upstream_pdb._parse_biounits  # noqa: SLF001

    def swapped(*a: Any, **k: Any) -> Any:  # noqa: ANN401
      """Return N and CA columns exchanged."""
      xyz, seq = original(*a, **k)
      if isinstance(xyz, str):
        return xyz, seq
      return (xyz[:, [1, 0, 2, 3], :] if xyz.shape[1] == 4 else xyz), seq

    upstream_pdb._parse_biounits = swapped  # noqa: SLF001
  # zero_edge_embedding / swap_checkpoint_noise act in _load_pt via ANTIBMPNN_MUTANT


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import contextlib

  import jax
  import jax.numpy as jnp

  _apply_patch(args.arm)
  oracle_dir, root = args.oracle_dir, args.antibmpnn_root
  d = np.load(oracle_dir / "antibmpnn_oracle.npz")
  dd = np.load(oracle_dir / "antibmpnn_descriptors_oracle.npz")

  @contextlib.contextmanager
  def x64(enabled: bool):  # noqa: ANN202
    """Toggle ``jax_enable_x64`` for one precision pass."""
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
      yield
    finally:
      jax.config.update("jax_enable_x64", prev)

  clean = args.arm in (None, "clean")
  ckpts = CHECKPOINTS if clean else (PRIMARY,)
  t0 = time.time()
  metrics: dict[str, Any] = {}
  with x64(False):
    metrics["f32"] = _stage_forward(d, root, ckpts, jnp.float32, "f32")
    # the 14 conditional_probs score() calls run in the clean arm and in the entry mutant only (wall time)
    metrics["entry"] = _stage_entry(d, root, ckpts, with_cond=clean or args.arm == "entry_ar_mask_transposed")
    metrics["sampler"], metrics["sampler_token_mismatch"] = _stage_sampler(d, root, jnp.float32)
    metrics["descriptors"] = _stage_descriptors(dd)
  if clean:
    with x64(True):
      metrics["f64"] = _stage_forward(d, root, ckpts, jnp.float64, "f64")
  metrics["parse"] = _stage_parse(d, root)
  import aminx

  metrics["aminx_file"] = str(Path(aminx.__file__).resolve())  # which tree was graded
  metrics["seconds"] = time.time() - t0
  return metrics


def _stage_value(metrics: dict[str, Any], stage: str) -> float:
  """Raw aminx-vs-upstream (same precision) gap used to decide whether a mutant left the pass region."""
  if stage in ("features", "logits"):
    return max(c["gap_same"] for c in metrics["f32"]["cells"] if c["stage"] == stage)
  if stage == "entry":
    return max(c["gap_same"] for c in metrics["entry"]["cells"])
  if stage == "descriptors":
    return float(max(metrics["descriptors"].values()))
  if stage == "sampler":
    return float(metrics["sampler"])
  return float(metrics[stage])


def _grade_cells(cells: list[dict[str, Any]]) -> tuple[int, int, float]:
  """(# cells above the pass tolerance, # above the marginal tolerance, worst gap/floor ratio) against the f64 truth."""
  over_pass = over_marg = 0
  worst_ratio = 0.0
  for c in cells:
    tol_pass, tol_marg = _tols(c["floor"])
    over_pass += int(c["gap_f64"] > tol_pass)
    over_marg += int(c["gap_f64"] > tol_marg)
    worst_ratio = max(worst_ratio, c["gap_f64"] / max(c["floor"], 1e-12))
  return over_pass, over_marg, worst_ratio


def _parent(args: argparse.Namespace) -> dict[str, Any]:
  weights = _verify_inputs(args.oracle_dir, args.antibmpnn_root, args.weights_dir)
  work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "antibmpnn_parity_v2"
  work.mkdir(parents=True, exist_ok=True)
  script = str(Path(__file__).resolve())
  mutants = [m for m in args.mutants.split(",") if m]
  measured: dict[str, dict[str, Any]] = {}
  for arm in ("clean", *mutants):
    out = work / f"payload_{arm}.json"
    out.unlink(missing_ok=True)
    cmd = [
      sys.executable, script, "--arm", arm, "--oracle-dir", str(args.oracle_dir), "--antibmpnn-root", str(args.antibmpnn_root),
      "--weights-dir", str(args.weights_dir), "--payload-out", str(out),
    ]
    done = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=args.arm_timeout)
    if done.returncode != 0:
      log.error("arm %s failed: %s", arm, done.stderr[-800:])
      measured[arm] = {"error": done.stderr[-800:]}
      continue
    measured[arm] = json.loads(out.read_text(encoding="utf-8"))
  clean = measured["clean"]
  flat: dict[str, Any] = {}
  decomposition: dict[str, Any] = {}
  if "error" in clean:
    verdict = "error"
  else:
    f64_cells = clean["f64"]["cells"]
    f32_cells = clean["f32"]["cells"]
    entry_cells = clean["entry"]["cells"]
    max_f64 = max(c["gap_same"] for c in f64_cells)
    f32_pass, f32_marg, f32_ratio = _grade_cells(f32_cells)
    entry_pass, entry_marg, entry_ratio = _grade_cells(entry_cells)
    nbr = clean["f32"]["nbr_mismatch"] + clean["f64"]["nbr_mismatch"]
    parse = clean["parse"]
    desc = max(clean["descriptors"].values())
    ok = (
      max_f64 < PASS_F64 and f32_pass == 0 and entry_pass == 0 and nbr == 0 and parse < 1e-9
      and clean["sampler"] < PASS_SAMPLER and clean["sampler_token_mismatch"] == 0 and desc < PASS_SAMPLER
    )
    verdict = "pass" if ok else "other"
    for stage in ("features", "enc_node", "enc_edge", "logits"):
      sel = [c for c in f32_cells if c["stage"] == stage]
      decomposition[stage] = {
        "floor_up_f32_vs_f64": max(c["floor"] for c in sel),
        "aminx_f32_vs_up_f32": max(c["gap_same"] for c in sel),
        "aminx_f32_vs_up_f64": max(c["gap_f64"] for c in sel),
        "aminx_f64_vs_up_f64": max(c["gap_same"] for c in f64_cells if c["stage"] == stage),
      }
    decomposition["per_cell_f32"] = f32_cells
    decomposition["per_cell_entry"] = entry_cells
    flat = {
      "max_abs_f64": max_f64, "f32_pass_violations": f32_pass, "f32_marginal_violations": f32_marg, "f32_worst_ratio": f32_ratio,
      "entry_pass_violations": entry_pass, "entry_marginal_violations": entry_marg, "entry_worst_ratio": entry_ratio,
      "entry_n_cells": len(entry_cells), "entry_tf_max_abs_f64": max(c["gap_f64"] for c in entry_cells if c["stage"] == "entry_tf"),
      "nbr_mismatch": nbr, "parse_max_abs": parse, "sampler_f32": clean["sampler"],
      "sampler_token_mismatch": clean["sampler_token_mismatch"], "descriptors_max": desc,
      "descriptors_csv": clean["descriptors"]["csv"], "descriptors_direct": clean["descriptors"]["direct"],
      "descriptors_ph_sweep": clean["descriptors"]["ph_sweep"], "n_checkpoints": len({c["ckpt"] for c in f32_cells}),
    }
  statuses: dict[str, str] = {}
  for m in mutants:
    payload = measured[m]
    if "error" in payload:
      statuses[m] = "error"
    else:
      stage = MUTANTS[m][0]
      value = _stage_value(payload, stage)
      statuses[m] = "failed" if (value > FAIL_MIN or (stage == "sampler" and payload["sampler_token_mismatch"] > 0)) else "passed"
      log.info("mutant %s (%s) stage %s value %.3e -> %s", m, MUTANTS[m][1], stage, value, statuses[m])
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(json.dumps({"clean": verdict, "mutants": statuses, "weights": weights}, indent=2) + "\n", encoding="utf-8")
  summary = {
    "clean": verdict, "n_listed": len(mutants), "n_failed": sum(s == "failed" for s in statuses.values()),
    "n_code_level": sum(MUTANTS[m][1] == "code" for m in mutants),
    "n_code_level_failed": sum(MUTANTS[m][1] == "code" and statuses[m] == "failed" for m in mutants),
    "oracle_verified": 1, "weight_source": "upstream_checkpoint", **flat,
    "decomposition": decomposition,
    "mutant_values": {
      m: {"stage": MUTANTS[m][0], "level": MUTANTS[m][1], "value": (_stage_value(measured[m], MUTANTS[m][0]) if "error" not in measured[m] else None)}
      for m in mutants
    },
    "arm_seconds": {a: v.get("seconds") for a, v in measured.items()},
    "aminx_file": {a: v.get("aminx_file") for a, v in measured.items()},
  }
  log.info("summary %s", json.dumps({k: v for k, v in summary.items() if k != "decomposition"}, default=str))
  return summary


def main() -> None:
  """Entry point: parent (arms + verdict) or, with ``--arm``, one fresh-interpreter arm."""
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--oracle-dir", type=Path, default=Path(os.environ["ANTIBMPNN_ORACLE_DIR"]) if os.environ.get("ANTIBMPNN_ORACLE_DIR") else None)
  parser.add_argument("--antibmpnn-root", type=Path, default=Path(os.environ.get("AMINX_ANTIBMPNN_ROOT", "~/repos/AntiBMPNN")).expanduser())
  parser.add_argument("--weights-dir", type=Path, default=Path(os.environ["AMINX_WEIGHTS_DIR"]) if os.environ.get("AMINX_WEIGHTS_DIR") else None)
  parser.add_argument("--mutants", default=",".join(MUTANTS))
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--arm-timeout", type=float, default=3600.0)
  args = parser.parse_args()
  if args.oracle_dir is None:
    msg = "set --oracle-dir or $ANTIBMPNN_ORACLE_DIR"
    raise SystemExit(msg)
  if args.weights_dir is None:
    msg = "set --weights-dir or $AMINX_WEIGHTS_DIR (the published antibmpnn_v_48_*.eqx.zst snapshot; the entry stage loads them)"
    raise SystemExit(msg)
  os.environ["AMINX_WEIGHTS_DIR"] = str(args.weights_dir)
  if args.arm is not None:
    payload = _run_arm(args)
    args.payload_out.write_text(json.dumps(payload), encoding="utf-8")
    return
  summary = _parent(args)
  results = os.environ.get("BTH_RESULTS_PATH")
  if not results:
    msg = "antibmpnn_parity_v2 requires $BTH_RESULTS_PATH (run under bth)"
    raise SystemExit(msg)
  Path(results).write_text(json.dumps(summary, default=str), encoding="utf-8")


if __name__ == "__main__":
  main()
