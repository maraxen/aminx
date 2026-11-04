"""Bathos parity vehicle for FAMPNN: aminx against a sealed dump of the real upstream code.

The reference is ``dump_fampnn_oracle.py`` output (real ``~/repos/fampnn`` code and released weights,
injected RNG), never an aminx-derived value. Stages, each graded per checkpoint and fixture:

  parse   aminx structure prep against upstream ``load_feats_from_pdb`` (backbone, aatype, chain, numbering)
  enc     encoder logits / h_V / h_ESV / E_idx / atom14 backbone under five mask states
  scn1    one sidechain-denoiser call, with and without self-conditioning; the sigma(t) grid
  edm     the EDM loop on an injected prior, per-step inputs/outputs, local and global coordinates, psce
  sample  the masked-autoregressive sampler (linear-K schedule, psce re-mask, overrides, repack_last)
  fixed   ``"A1-20,B45-50"`` fixed-position parsing against upstream ``get_override_masks``

Bands: float64 arm 1e-7 absolute on every continuous array; float32 arm
``max(1e-4, 5 * floor)`` where ``floor`` is the sealed upstream's own float32-vs-float64 deviation on the
same array (the oracle cannot be closer to itself than that). Discrete arrays (aatype, indices, orders)
are exact in both arms. Negative controls are aminx-side code perturbations; one is "failed" (the
control works) when its targeted stage leaves the float64 band by more than 1e-4 or flips a discrete
value.

Run on titanix. Required env: ``AMINX_FAMPNN_ORACLE_DIR`` (holds ``f32/`` and ``f64/``),
``AMINX_FAMPNN_ROOT`` (upstream clone; default ``~/repos/fampnn``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "scripts" / "parity"):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from artifact_key import stable_artifact_key  # noqa: E402
from fampnn_fixtures import make_split_ab_pdb  # noqa: E402

from aminx.families.fampnn import sampler  # noqa: E402
from aminx.families.fampnn.constants import NON_BB_IDXS_ARR, RESTYPE_ATOM37_MASK, RESTYPE_REF_POS_ATOM37  # noqa: E402
from aminx.families.fampnn.features import positional_embedding_table  # noqa: E402
from aminx.families.fampnn import sidechain_diffusion as sd_mod  # noqa: E402
from aminx.families.fampnn.convert import convert_fampnn_state_dict  # noqa: E402
from aminx.families.fampnn.driver import _prepare_structure  # noqa: E402
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint  # noqa: E402

LOG = logging.getLogger("fampnn_parity")

# ---- pinned identities (the sealed oracle and the released weights) ---------------------------------
PINNED_ORACLE_CONTENT_SHA256 = {
  "f32": "c903b3808b3f30fc491e4feb8407a9c1d60492836a325185e3712c8fed456b4c",
  "f64": "9494f1531e7b6a7c482883d82f933cfdeda5f360dd0ceeca626b66d601a9e553",
}
PINNED_ORACLE_FILE_SHA256 = {
  "f32": "d8b38894a48c09d6808008e13d7c7ee1c02aa8cd898b017863d35870289f5a2c",
  "f64": "c68eaac50213c3eb85530940e4816ae841c5457c4540ad583039eff7c9596fe2",
}
PINNED_WEIGHTS_SHA256 = {
  "fampnn_0_0": "afbdfda29e6f2a1bd340971bb226638afb1bf460cfbc29f115d3b5964622c006",
  "fampnn_0_3": "8969b3f1f3c941178076c7800952595a18b56fd3828d15bb993d3ef537938a05",
  "fampnn_0_3_cath": "81112a9b8d436d9baf5233a3603bac911c75b9782ced59dae5a2726316802218",
}
CKPTS = tuple(PINNED_WEIGHTS_SHA256)
SAMPLER_CKPTS = ("fampnn_0_3", "fampnn_0_3_cath")
FIXTURES = ("1ubq", "1ubq_AB")
STATES = ("all_masked", "half", "seqonly", "partial", "full")
EDM_TAGS = ("s50", "s12_sc1", "s12_churn")
SAMPLE_TAGS = ("A_default", "B_fixed", "C_seqonly_argmax", "D_churn_hot")

F64_BAND = 1.0e-7
F64_BAND_LOOP = 2.0e-6  # edm and sample stages: see the sidecar [data].bands derivation
LOOP_STAGES = ("edm", "sample")
F32_BAND_FLOOR = 1.0e-4
F32_FLOOR_MULT = 5.0
MUTANT_TRIP = 1.0e-4

DEFAULT_MUTANTS = (
  "sigma_min_old_default",
  "cosine_unmask_schedule",
  "rank_decoding_order",
  "selfcond_off",
  "step_scale_one",
  "conf_bins_0_8",
  "ghost_atoms_to_origin",
  "chain_embedding_off",
  "categorical_reversed_cdf",
)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
  """Canonical digest over (key, dtype, shape, bytes); mirrors ``dump_fampnn_oracle.content_sha256``."""
  h = hashlib.sha256()
  for k in sorted(arrays):
    a = np.ascontiguousarray(arrays[k])
    h.update(k.encode())
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
  return h.hexdigest()


def _cast(model: Any, dtype: Any) -> Any:  # noqa: ANN401
  def f(a: Any) -> Any:  # noqa: ANN401
    return a.astype(dtype) if eqx.is_inexact_array(a) else a

  return jax.tree_util.tree_map(f, model)


class Ledger:
  """Collects (arm, stage, name) -> deviation, with the arm's band."""

  def __init__(self, floors: dict[str, float]) -> None:
    self.floors = floors
    self.rows: list[dict[str, Any]] = []

  def band(self, arm: str, stage: str, key: str) -> float:
    if arm == "f64":
      return F64_BAND_LOOP if stage in LOOP_STAGES else F64_BAND
    return max(F32_BAND_FLOOR, F32_FLOOR_MULT * self.floors.get(key, 0.0))

  def cont(self, arm: str, stage: str, key: str, ours: Any, ref: np.ndarray) -> float:  # noqa: ANN401
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    if a.shape != b.shape:
      self.rows.append({"arm": arm, "stage": stage, "key": key, "dev": float("inf"), "band": self.band(arm, stage, key), "kind": "shape"})
      return float("inf")
    dev = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.isfinite(dev):
      dev = float("inf")
    self.rows.append({"arm": arm, "stage": stage, "key": key, "dev": dev, "band": self.band(arm, stage, key), "kind": "cont"})
    return dev

  def disc(self, arm: str, stage: str, key: str, ours: Any, ref: np.ndarray) -> int:  # noqa: ANN401
    a = np.asarray(ours)
    b = np.asarray(ref)
    if a.shape != b.shape:
      n = max(a.size, b.size, 1)
    else:
      n = int(np.sum(a != b))
    self.rows.append({"arm": arm, "stage": stage, "key": key, "dev": float(n), "band": 0.0, "kind": "disc"})
    return n


def _worst(rows: list[dict[str, Any]], arm: str, stage: str) -> float:
  v = [r["dev"] for r in rows if r["arm"] == arm and r["stage"] == stage and r["kind"] == "cont"]
  return max(v) if v else float("nan")


def _n_over(rows: list[dict[str, Any]], arm: str) -> int:
  return sum(1 for r in rows if r["arm"] == arm and r["kind"] == "cont" and not r["dev"] <= r["band"])


def _n_disc(rows: list[dict[str, Any]], arm: str) -> int:
  return int(sum(r["dev"] for r in rows if r["arm"] == arm and r["kind"] == "disc"))


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
class Oracle:
  """One sealed dump (``f32`` or ``f64``) plus its manifest."""

  def __init__(self, directory: Path, arm: str) -> None:
    self.arm = arm
    npz = directory / arm / f"fampnn_oracle_{arm}.npz"
    man = json.loads((directory / arm / f"fampnn_oracle_{arm}.manifest.json").read_text())
    self.file_sha256 = _sha256(npz)
    z = np.load(npz)
    self.d = {k: z[k] for k in z.files}
    self.content_sha256 = content_sha256(self.d)
    self.meta = man["meta"]
    self.sealed = (
      self.content_sha256 == PINNED_ORACLE_CONTENT_SHA256[arm]
      and self.file_sha256 == PINNED_ORACLE_FILE_SHA256[arm]
      and man["content_sha256"] == self.content_sha256
    )
    self.dtype = jnp.float64 if arm == "f64" else jnp.float32


def _floors(o32: Oracle, o64: Oracle) -> dict[str, float]:
  out: dict[str, float] = {}
  for k, a in o32.d.items():
    if a.dtype.kind != "f" or k not in o64.d:
      continue
    out[k] = float(np.max(np.abs(a.astype(np.float64) - o64.d[k].astype(np.float64)))) if a.size else 0.0
  return out


def _j(o: Oracle, key: str, dtype: Any | None = None) -> jax.Array:  # noqa: ANN401
  a = o.d[key]
  if dtype is not None:
    return jnp.asarray(a, dtype=dtype)
  return jnp.asarray(a)


def stage_parse(led: Ledger, o: Oracle, fx: str, pdb: Path) -> None:
  """aminx structure prep against upstream ``load_feats_from_pdb`` + ``process_single_pdb`` (all 37 atoms)."""
  prep = _prepare_structure(pdb)
  P = f"fampnn_0_3/{fx}/parse"
  led.cont(o.arm, "parse", f"{fx}/atom37_xyz", prep.atom37_xyz, o.d[f"{P}/x"])
  led.disc(o.arm, "parse", f"{fx}/atom37_mask", prep.atom37_mask, o.d[f"{P}/atom_mask"])
  led.disc(o.arm, "parse", f"{fx}/missing_atom_mask", prep.missing_atom_mask, o.d[f"{P}/missing_atom_mask"])
  led.disc(o.arm, "parse", f"{fx}/aatype", prep.aatype_fampnn, o.d[f"{P}/aatype"])
  led.disc(o.arm, "parse", f"{fx}/chain_index", prep.chain_idx, o.d[f"{P}/chain_index"])
  led.disc(o.arm, "parse", f"{fx}/residue_index", prep.residue_idx, o.d[f"{P}/residue_index"])
  led.disc(o.arm, "parse", f"{fx}/valid_mask", prep.mask, o.d[f"{P}/seq_mask"])
  led.disc(o.arm, "parse", f"{fx}/chain_id_mapping", np.asarray(sorted(prep.chain_id_mapping.items())), np.asarray(sorted(o.meta["chain_id_mapping"][fx].items())))


def stage_enc(led: Ledger, o: Oracle, model: Any, ck: str, fx: str, *, states: tuple[str, ...] = STATES, flat_chain: bool = False) -> None:  # noqa: ANN401
  P = f"{ck}/{fx}"
  dt = o.dtype
  ridx = _j(o, f"{P}/parse/residue_index").astype(jnp.int32)
  cidx = _j(o, f"{P}/parse/chain_index").astype(jnp.int32)
  if flat_chain:
    cidx = jnp.zeros_like(cidx)
  valid = jnp.ones(ridx.shape, dtype=dt)
  for st in states:
    S = f"{P}/enc/{st}"
    x_in = _j(o, f"{S}/x_in", dt)
    amask = _j(o, f"{S}/atom_mask_noised", dt)
    aa = _j(o, f"{S}/aatype_noised").astype(jnp.int32)
    fwd = model(x_in, amask, aa, valid, valid, ridx, cidx)
    led.cont(o.arm, "enc", f"{S}/logits", fwd.logits_fampnn, o.d[f"{S}/logits"])
    led.cont(o.arm, "enc", f"{S}/h_V", fwd.h_v, o.d[f"{S}/h_V"])
    led.disc(o.arm, "enc", f"{S}/E_idx", fwd.e_idx, o.d[f"{S}/E_idx"])
    if f"{S}/h_ESV" in o.d:
      led.cont(o.arm, "enc", f"{S}/h_ESV", fwd.h_e, o.d[f"{S}/h_ESV"])
    x_bb, bb_mask = sampler.encoder_backbone(x_in, amask, aa)
    led.cont(o.arm, "enc", f"{S}/X14_bb", x_bb, o.d[f"{S}/X14"][:, :4])
    led.cont(o.arm, "enc", f"{S}/atom14_mask_bb", bb_mask, o.d[f"{S}/atom14_mask"][:, :4])
    led.cont(o.arm, "enc", f"{S}/log_probs", jax.nn.log_softmax(fwd.logits_fampnn, axis=-1), o.d[f"{S}/log_probs"])


def stage_scn1(led: Ledger, o: Oracle, model: Any, ck: str, fx: str) -> None:  # noqa: ANN401
  P = f"{ck}/{fx}"
  dt = o.dtype
  S = f"{P}/scn1"
  aa = _j(o, f"{P}/parse/aatype").astype(jnp.int32)
  valid = jnp.ones(aa.shape, dtype=dt)
  h_v = _j(o, f"{P}/enc/half/h_V", dt)
  t = _j(o, f"{S}/t", dt)[0]
  diff = model.scn_diffusion
  y_sc = sampler.scn_denoise_local(diff, _j(o, f"{S}/x_scn", dt), t, h_v, aa, valid, _j(o, f"{S}/x_self_cond", dt))
  y_no = sampler.scn_denoise_local(diff, _j(o, f"{S}/x_scn", dt), t, h_v, aa, valid, None)
  led.cont(o.arm, "scn1", f"{S}/y_selfcond", y_sc, o.d[f"{S}/y_selfcond"])
  led.cont(o.arm, "scn1", f"{S}/y_noselfcond", y_no, o.d[f"{S}/y_noselfcond"])
  grid_t = _j(o, f"{S}/sigma_grid_t", dt)
  sig = sampler.edm_sigma(grid_t, s_min=diff.edm_s_min, s_max=diff.edm_s_max, rho=diff.edm_rho, sigma_data=diff.sigma_data)
  led.cont(o.arm, "scn1", f"{S}/sigma_grid", sig, o.d[f"{S}/sigma_grid"])


def stage_edm(led: Ledger, o: Oracle, model: Any, ck: str, fx: str, *, tags: tuple[str, ...] = EDM_TAGS) -> None:  # noqa: ANN401
  P = f"{ck}/{fx}"
  dt = o.dtype
  aa = _j(o, f"{P}/parse/aatype").astype(jnp.int32)
  ridx = _j(o, f"{P}/parse/residue_index").astype(jnp.int32)
  cidx = _j(o, f"{P}/parse/chain_index").astype(jnp.int32)
  valid = jnp.ones(aa.shape, dtype=dt)
  half = f"{P}/enc/half"
  x_in = _j(o, f"{half}/x_in", dt)
  amask = _j(o, f"{half}/atom_mask_noised", dt)
  aa_noised = _j(o, f"{half}/aatype_noised").astype(jnp.int32)
  fwd = model(x_in, amask, aa_noised, valid, valid, ridx, cidx)
  x_bb, bb_mask = sampler.encoder_backbone(x_in, amask, aa_noised)
  for tag in tags:
    S = f"{P}/edm/{tag}"
    cfg = o.meta["edm_cfg"][tag]
    n = int(cfg["num_steps"])
    ts = sampler.timestep_schedule("linear", n, 0.0, 1.0, dt)
    led.cont(o.arm, "edm", f"{S}/timesteps", ts, o.d[f"{S}/timesteps"])
    churn = None
    if f"{S}/churn_z" in o.d:
      churn = _j(o, f"{S}/churn_z", dt)[:, 0]
    s_churn = float(cfg["s_churn"])

    def run(model: Any, h_v: Any, h_e: Any, e_idx: Any, prior: Any, churn: Any, ts: Any, step_scale: float = float(cfg["step_scale"]), s_churn: float = s_churn) -> tuple[Any, ...]:  # noqa: ANN401
      x_local, trace = sampler.scn_edm_loop(
        model.scn_diffusion, h_v, aa, valid, ts, prior, churn,
        step_scale=step_scale, s_churn=s_churn, s_noise=1.0, s_t_min=0.01, s_t_max=50.0,
      )  # fmt: skip
      scn_atom_mask = jnp.take(RESTYPE_ATOM37_MASK[aa].astype(dt), NON_BB_IDXS_ARR, axis=1)
      x_glob = sampler.local_to_global_scn(x_local, x_bb, scn_atom_mask, bb_mask)
      psce = sampler.psce_from_local(
        model.scn_confidence, x_local, h_v, h_e, e_idx, aa, valid,
        min_bin=model.conf_min_bin, max_bin=model.conf_max_bin,
      )  # fmt: skip
      return x_local, trace.x_in, trace.x_out, x_glob, psce

    x_local, tr_in, tr_out, x_glob, psce = eqx.filter_jit(run)(model, fwd.h_v, fwd.h_e, fwd.e_idx, _j(o, f"{S}/prior_z", dt)[0], churn, ts)
    led.cont(o.arm, "edm", f"{S}/x_local_final", x_local, o.d[f"{S}/x_local_final"])
    led.cont(o.arm, "edm", f"{S}/traj_in", tr_in, o.d[f"{S}/traj_in"])
    led.cont(o.arm, "edm", f"{S}/traj_out", tr_out, o.d[f"{S}/traj_out"])
    led.cont(o.arm, "edm", f"{S}/x_scn_global", x_glob, o.d[f"{S}/x_scn_global"])
    led.cont(o.arm, "edm", f"{S}/psce", psce, o.d[f"{S}/psce"])


def stage_consts(led: Ledger, o: Oracle) -> None:
  """Module-level constants: the CCD reference conformers (rows 0-19) and the float32 positional table."""
  led.cont(o.arm, "consts", "ref_pos_atom37[:20]", np.asarray(RESTYPE_REF_POS_ATOM37)[:20], o.d["consts/ref_pos_atom37"][:20])
  num_l = (o.d["consts/posemb_offsets"].shape[0] + 1) // 2
  led.cont(o.arm, "consts", "posemb_table", positional_embedding_table(num_l, 16), o.d["consts/posemb_table"])


def stage_fixed(led: Ledger, o: Oracle, ck: str, fx: str) -> None:
  P = f"{ck}/{fx}"
  fp = o.meta["fixed_pos"][fx]
  aov, sov = sampler.override_masks(
    fp["fixed_pos_seq"], fp["fixed_pos_scn"], o.meta["chain_id_mapping"][fx],
    o.d[f"{P}/parse/residue_index"], o.d[f"{P}/parse/chain_index"],
  )  # fmt: skip
  S = f"{P}/sample/B_fixed"
  led.disc(o.arm, "fixed", f"{S}/aatype_override_mask", np.asarray(aov).astype(np.int64), o.d[f"{S}/aatype_override_mask"])
  led.disc(o.arm, "fixed", f"{S}/scn_override_mask", np.asarray(sov).astype(np.int64), o.d[f"{S}/scn_override_mask"])


def _sample_inputs(o: Oracle, ck: str, fx: str, tag: str) -> tuple[dict[str, Any], dict[str, Any]]:
  dt = o.dtype
  P = f"{ck}/{fx}"
  S = f"{P}/sample/{tag}"
  cfg = o.meta["sample_cfg"][tag]
  num_s = int(cfg["S"])
  calls = num_s + int(cfg["repack_last"])
  tsteps = int(cfg["scn_steps"])
  churn = None
  if f"{S}/inj_churn_z" in o.d:
    churn = _j(o, f"{S}/inj_churn_z", dt)[:, 0].reshape(calls, tsteps, *o.d[f"{S}/inj_prior_z"].shape[2:])
  cat = _j(o, f"{S}/inj_multinomial_u", dt)[:num_s] if f"{S}/inj_multinomial_u" in o.d else jnp.zeros((num_s, o.d[f"{P}/parse/aatype"].shape[0]), dt)
  noise = sampler.SamplerNoise(
    decoding_u=_j(o, f"{S}/inj_decoding_u", dt)[0],
    categorical_u=cat,
    prior_z=_j(o, f"{S}/inj_prior_z", dt)[:, 0],
    churn_z=churn,
  )
  fp = o.meta.get("fixed_pos", {}).get(fx)
  if cfg["fixed"]:
    aov, sov = sampler.override_masks(
      fp["fixed_pos_seq"], fp["fixed_pos_scn"], o.meta["chain_id_mapping"][fx],
      o.d[f"{P}/parse/residue_index"], o.d[f"{P}/parse/chain_index"],
    )  # fmt: skip
    aov, sov = aov.astype(dt), sov.astype(dt)
  else:
    aov = sov = None
  inputs = {
    "x": _j(o, f"{P}/parse/x", dt),
    "aatype": _j(o, f"{P}/parse/aatype").astype(jnp.int32),
    "valid": _j(o, f"{P}/parse/seq_mask", dt),
    "missing": _j(o, f"{P}/parse/missing_atom_mask", dt),
    "residue_idx": _j(o, f"{P}/parse/residue_index").astype(jnp.int32),
    "chain_idx": _j(o, f"{P}/parse/chain_index").astype(jnp.int32),
    "noise": noise,
    "aatype_override_mask": aov,
    "scn_override_mask": sov,
  }
  return inputs, cfg


def stage_sample(led: Ledger, o: Oracle, model: Any, ck: str, fx: str, tags: tuple[str, ...] = SAMPLE_TAGS) -> None:  # noqa: ANN401
  P = f"{ck}/{fx}"
  dt = o.dtype
  for tag in tags:
    S = f"{P}/sample/{tag}"
    inp, cfg = _sample_inputs(o, ck, fx, tag)
    ts = sampler.timestep_schedule("linear", int(cfg["S"]), 0.0, 1.0, dt)
    ts_scn = sampler.timestep_schedule("linear", int(cfg["scn_steps"]), 0.0, 1.0, dt)
    led.cont(o.arm, "sample", f"{S}/timesteps", ts, o.d[f"{S}/timesteps"])

    def run(model: Any, inp: dict[str, Any], ts: Any, ts_scn: Any, cfg: dict[str, Any] = cfg) -> Any:  # noqa: ANN401
      return sampler.mar_sample_upstream(
        model, inp["x"], inp["aatype"], inp["valid"], inp["missing"], inp["residue_idx"], inp["chain_idx"], ts, inp["noise"],
        temperature=float(cfg["T"]), seq_only=bool(cfg["seq_only"]), repack_last=bool(cfg["repack_last"]),
        psce_threshold=None if cfg["psce"] is None else float(cfg["psce"]),
        aatype_override_mask=inp["aatype_override_mask"], scn_override_mask=inp["scn_override_mask"],
        timesteps_scn=ts_scn, step_scale=float(cfg["step_scale"]), s_churn=float(cfg["s_churn"]),
        conf_min_bin=model.conf_min_bin, conf_max_bin=model.conf_max_bin,
      )  # fmt: skip

    r = eqx.filter_jit(run)(model, inp, ts, ts_scn)
    led.cont(o.arm, "sample", f"{S}/x_final", r.x_final, o.d[f"{S}/x_final"])
    led.cont(o.arm, "sample", f"{S}/psce", r.psce, o.d[f"{S}/psce"])
    led.cont(o.arm, "sample", f"{S}/seq_probs", r.seq_probs, o.d[f"{S}/seq_probs"])
    led.cont(o.arm, "sample", f"{S}/xt_traj", r.xt_traj, o.d[f"{S}/xt_traj"])
    led.cont(o.arm, "sample", f"{S}/psce_t_traj", r.psce_t_traj, o.d[f"{S}/psce_t_traj"])
    led.cont(o.arm, "sample", f"{S}/seq_logits_traj", r.seq_logits_traj, o.d[f"{S}/seq_logits_traj"])
    led.disc(o.arm, "sample", f"{S}/aatype_final", r.aatype_final, o.d[f"{S}/aatype_final"])
    led.disc(o.arm, "sample", f"{S}/aatype_t_traj", r.aatype_t_traj, o.d[f"{S}/aatype_t_traj"])
    led.disc(o.arm, "sample", f"{S}/aatype_pred_traj", r.aatype_pred_traj, o.d[f"{S}/aatype_pred_traj"])


# ---------------------------------------------------------------------------------------------
# negative controls: aminx-side code perturbations
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _patched(mod: Any, name: str, repl: Callable[..., Any]) -> Iterator[None]:  # noqa: ANN401
  orig = getattr(mod, name)
  setattr(mod, name, repl)
  try:
    yield
  finally:
    setattr(mod, name, orig)


def _mutant_ctx(name: str) -> contextlib.AbstractContextManager[None]:
  if name == "sigma_min_old_default":
    orig = sampler.edm_sigma
    return _patched(sampler, "edm_sigma", lambda t, **k: orig(t, **{**k, "s_min": 0.002}))
  if name == "cosine_unmask_schedule":
    orig_ts = sampler.timestep_schedule
    return _patched(sampler, "timestep_schedule", lambda mode, *a, **k: orig_ts("cosine" if mode == "linear" else mode, *a, **k))
  if name == "rank_decoding_order":
    return _patched(sampler, "decoding_order", lambda u, valid: jnp.argsort(jnp.argsort(jnp.where(valid > 0, u, 1.0e6))).astype(jnp.int32))
  if name == "selfcond_off":
    orig_d = sampler.scn_denoise_local
    return _patched(sampler, "scn_denoise_local", lambda diffusion, x, t, h, aa, v, cond: orig_d(diffusion, x, t, h, aa, v, None))
  if name == "step_scale_one":
    orig_l = sampler.scn_edm_loop
    return _patched(sampler, "scn_edm_loop", lambda *a, **k: orig_l(*a, **{**k, "step_scale": 1.0}))
  if name == "conf_bins_0_8":
    orig_p = sampler.psce_from_local
    return _patched(sampler, "psce_from_local", lambda *a, **k: orig_p(*a, **{**k, "max_bin": 8.0}))
  if name == "ghost_atoms_to_origin":
    orig_g = sampler.local_to_global_scn

    def ghost0(x_local: Any, x_bb: Any, scn_atom_mask: Any, bb_mask: Any) -> Any:  # noqa: ANN401
      out = orig_g(x_local, x_bb, scn_atom_mask, bb_mask)
      return jnp.where(scn_atom_mask[..., None] > 0, out, 0.0)

    return _patched(sampler, "local_to_global_scn", ghost0)
  if name == "chain_embedding_off":
    return contextlib.nullcontext()  # applied through stage_enc(flat_chain=True)
  if name == "categorical_reversed_cdf":
    orig_c = sampler.categorical_inverse_cdf
    return _patched(sampler, "categorical_inverse_cdf", lambda probs, u: orig_c(probs, 1.0 - u))
  msg = f"unknown mutant {name}"
  raise KeyError(msg)


_MUTANT_STAGES: dict[str, tuple[str, ...]] = {
  "sigma_min_old_default": ("scn1", "edm"),
  "cosine_unmask_schedule": ("sample",),
  "rank_decoding_order": ("sample",),
  "selfcond_off": ("scn1", "edm"),
  "step_scale_one": ("edm",),
  "conf_bins_0_8": ("edm",),
  "ghost_atoms_to_origin": ("edm",),
  "chain_embedding_off": ("enc",),
  "categorical_reversed_cdf": ("sample",),
}


def run_mutant(name: str, o64: Oracle, models64: dict[str, Any]) -> tuple[str, dict[str, float]]:
  """Run the targeted stages with the perturbation applied; "failed" means the control tripped."""
  led = Ledger({})
  stages = _MUTANT_STAGES[name]
  ck, fx = "fampnn_0_3", "1ubq"
  with _mutant_ctx(name):
    if "enc" in stages:
      stage_enc(led, o64, models64[ck], ck, "1ubq_AB", states=("half", "full"), flat_chain=(name == "chain_embedding_off"))
    if "scn1" in stages:
      stage_scn1(led, o64, models64[ck], ck, fx)
    if "edm" in stages:
      stage_edm(led, o64, models64[ck], ck, fx, tags=("s50",))
    if "sample" in stages:
      stage_sample(led, o64, models64[ck], ck, fx, tags=("A_default",))
  dev = max([r["dev"] for r in led.rows if r["kind"] == "cont"], default=0.0)
  nd = int(sum(r["dev"] for r in led.rows if r["kind"] == "disc"))
  status = "failed" if (dev > MUTANT_TRIP or nd > 0) else "passed"
  return status, {"max_abs_diff": dev, "discrete_mismatches": float(nd)}


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def leaf_provenance(raw: Any, o32: Oracle) -> float:  # noqa: ANN401
  """Two conversions with different init keys must agree on every graded output: no random leaf is live."""
  m1 = convert_fampnn_state_dict(raw, key=jax.random.PRNGKey(1))
  m2 = convert_fampnn_state_dict(raw, key=jax.random.PRNGKey(2))
  led = Ledger({})
  # reuse the graded stages with model 2 compared to model 1's outputs via the oracle: both must equal the oracle;
  # here we additionally require bit-identical outputs between the two conversions.
  ck, fx = "fampnn_0_3", "1ubq"
  P = f"{ck}/{fx}"
  dt = jnp.float32
  ridx = _j(o32, f"{P}/parse/residue_index").astype(jnp.int32)
  cidx = _j(o32, f"{P}/parse/chain_index").astype(jnp.int32)
  valid = jnp.ones(ridx.shape, dtype=dt)
  worst = 0.0
  for st in ("half", "full"):
    S = f"{P}/enc/{st}"
    args = (_j(o32, f"{S}/x_in", dt), _j(o32, f"{S}/atom_mask_noised", dt), _j(o32, f"{S}/aatype_noised").astype(jnp.int32), valid, valid, ridx, cidx)
    a, b = m1(*args), m2(*args)
    for u, v in zip((a.logits_fampnn, a.h_v, a.h_e), (b.logits_fampnn, b.h_v, b.h_e), strict=True):
      worst = max(worst, float(jnp.max(jnp.abs(u - v))))
  aa = _j(o32, f"{P}/parse/aatype").astype(jnp.int32)
  S = f"{P}/scn1"
  h_v = _j(o32, f"{P}/enc/half/h_V", dt)
  t = _j(o32, f"{S}/t", dt)[0]
  for sc in (_j(o32, f"{S}/x_self_cond", dt), None):
    y1 = sampler.scn_denoise_local(m1.scn_diffusion, _j(o32, f"{S}/x_scn", dt), t, h_v, aa, valid, sc)
    y2 = sampler.scn_denoise_local(m2.scn_diffusion, _j(o32, f"{S}/x_scn", dt), t, h_v, aa, valid, sc)
    worst = max(worst, float(jnp.max(jnp.abs(y1 - y2))))
  half = f"{P}/enc/half"
  fwd = m1(_j(o32, f"{half}/x_in", dt), _j(o32, f"{half}/atom_mask_noised", dt), _j(o32, f"{half}/aatype_noised").astype(jnp.int32), valid, valid, ridx, cidx)
  x_local = _j(o32, f"{P}/edm/s50/x_local_final", dt)
  p1 = sampler.psce_from_local(m1.scn_confidence, x_local, fwd.h_v, fwd.h_e, fwd.e_idx, aa, valid, min_bin=0.0, max_bin=4.0)
  p2 = sampler.psce_from_local(m2.scn_confidence, x_local, fwd.h_v, fwd.h_e, fwd.e_idx, aa, valid, min_bin=0.0, max_bin=4.0)
  worst = max(worst, float(jnp.max(jnp.abs(p1 - p2))))
  del led
  return worst


def _json_safe(v: Any) -> Any:  # noqa: ANN401
  if isinstance(v, float):
    return None if not math.isfinite(v) else v
  if isinstance(v, dict):
    return {k: _json_safe(x) for k, x in v.items()}
  if isinstance(v, list):
    return [_json_safe(x) for x in v]
  return v


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
  t0 = time.time()
  odir = Path(args.oracle_dir)
  froot = Path(args.fampnn_root)
  o32, o64 = Oracle(odir, "f32"), Oracle(odir, "f64")
  led = Ledger(_floors(o32, o64))

  registry = {e["artifact_path"]: e["sha256"] for e in json.loads((_REPO_ROOT / "tests/knob_gate/checkpoint_registry.json").read_text())["entries"]}
  weights: dict[str, str] = {}
  weights_ok = True
  raws: dict[str, Any] = {}
  models32: dict[str, Any] = {}
  for ck in CKPTS:
    path = froot / "weights" / f"{ck}.pt"
    sha = _sha256(path)
    key = stable_artifact_key(path)
    weights[key] = sha
    weights_ok &= sha == PINNED_WEIGHTS_SHA256[ck] and registry.get(key) == sha
    raws[ck] = load_restricted_torch_checkpoint(path)
    models32[ck] = convert_fampnn_state_dict(raws[ck])
  models64 = {ck: _cast(m, jnp.float64) for ck, m in models32.items()}

  ab_pdb = Path(args.work_dir) / "1ubq_split_AB.pdb"
  ab_sha = make_split_ab_pdb(_REPO_ROOT / "tests/data/1ubq.pdb", ab_pdb)
  fixture_ok = (
    ab_sha == o64.meta["pdb_sha256"]["1ubq_AB"]
    and _sha256(_REPO_ROOT / "tests/data/1ubq.pdb") == o64.meta["pdb_sha256"]["1ubq"]
  )
  pdbs = {"1ubq": _REPO_ROOT / "tests/data/1ubq.pdb", "1ubq_AB": ab_pdb}

  for o, models in ((o64, models64), (o32, models32)):
    stage_consts(led, o)
    for fx in FIXTURES:
      stage_parse(led, o, fx, pdbs[fx])
      stage_fixed(led, o, "fampnn_0_3", fx)
      for ck in CKPTS:
        LOG.info("arm=%s %s %s enc/scn1", o.arm, ck, fx)
        stage_enc(led, o, models[ck], ck, fx)
        stage_scn1(led, o, models[ck], ck, fx)
      for ck in SAMPLER_CKPTS:
        LOG.info("arm=%s %s %s edm/sample", o.arm, ck, fx)
        stage_edm(led, o, models[ck], ck, fx)
        stage_sample(led, o, models[ck], ck, fx)

  leaf_worst = leaf_provenance(raws["fampnn_0_3"], o32)

  mutant_list = [m.strip() for m in args.mutants.split(",") if m.strip()]
  mutants: dict[str, str] = {}
  mutant_dev: dict[str, dict[str, float]] = {}
  for m in mutant_list:
    LOG.info("mutant %s", m)
    mutants[m], mutant_dev[m] = run_mutant(m, o64, models64)

  rows = led.rows
  f64_over, f32_over = _n_over(rows, "f64"), _n_over(rows, "f32")
  disc64, disc32 = _n_disc(rows, "f64"), _n_disc(rows, "f32")
  f32_ratio = max(
    (r["dev"] / r["band"] for r in rows if r["arm"] == "f32" and r["kind"] == "cont" and r["band"] > 0),
    default=0.0,
  )
  parse_dev = max((r["dev"] for r in rows if r["stage"] == "parse" and r["kind"] == "cont"), default=0.0)
  parse_disc = int(sum(r["dev"] for r in rows if r["stage"] == "parse" and r["kind"] == "disc"))
  fixed_disc = int(sum(r["dev"] for r in rows if r["stage"] == "fixed" and r["kind"] == "disc"))
  all_ok = (
    o32.sealed and o64.sealed and weights_ok and fixture_ok and leaf_worst == 0.0
    and f64_over == 0 and f32_over == 0 and disc64 == 0 and disc32 == 0
  )  # fmt: skip
  n_failed = sum(1 for s in mutants.values() if s == "failed")
  result: dict[str, Any] = {
    "clean": "pass" if all_ok else "fail",
    "n_listed": len(mutant_list),
    "n_failed": n_failed,
    "weight_source": "upstream_checkpoint",
    "oracle_sealed": int(o32.sealed and o64.sealed),
    "oracle_f32_content_sha256": o32.content_sha256,
    "oracle_f64_content_sha256": o64.content_sha256,
    "oracle_f32_file_sha256": o32.file_sha256,
    "oracle_f64_file_sha256": o64.file_sha256,
    "weights_sha_ok": int(weights_ok),
    "fixture_sha_ok": int(fixture_ok),
    "leaf_provenance_max_abs": leaf_worst,
    "n_checks_f64": sum(1 for r in rows if r["arm"] == "f64"),
    "n_checks_f32": sum(1 for r in rows if r["arm"] == "f32"),
    "f64_n_over_band": f64_over,
    "f32_n_over_band": f32_over,
    "f64_discrete_mismatches": disc64,
    "f32_discrete_mismatches": disc32,
    "f32_worst_dev_over_band": f32_ratio,
    "parse_max_abs_diff": parse_dev,
    "parse_discrete_mismatches": parse_disc,
    "fixed_pos_mismatches": fixed_disc,
    "wall_seconds": time.time() - t0,
  }
  for arm in ("f64", "f32"):
    for stage in ("consts", "enc", "scn1", "edm", "sample"):
      result[f"{stage}_{arm}_worst_max_abs"] = _worst(rows, arm, stage)
  for m, dev in mutant_dev.items():
    result[f"mutant_{m}_max_abs_diff"] = dev["max_abs_diff"]
  result["mutants_detail"] = {m: mutant_dev[m] for m in mutant_dev}
  result["mutants"] = mutants

  # per-array breakdown goes to a side file, not the bth result
  breakdown = Path(args.out).with_suffix(".rows.json")
  breakdown.parent.mkdir(parents=True, exist_ok=True)
  breakdown.write_text(json.dumps(_json_safe(rows), indent=0))

  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(
      json.dumps({"clean": result["clean"], "mutants": mutants, "weights": weights}, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
  return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--out", type=Path, default=Path("outputs/parity/fampnn_parity.json"))
  ap.add_argument("--oracle-dir", default=os.environ.get("AMINX_FAMPNN_ORACLE_DIR"))
  ap.add_argument("--fampnn-root", default=os.environ.get("AMINX_FAMPNN_ROOT", str(Path.home() / "repos/fampnn")))
  ap.add_argument("--mutants", default=",".join(DEFAULT_MUTANTS), help="comma-separated negative controls")
  ap.add_argument("--controls-out", type=Path, default=None, help="branch_controls.json path")
  ap.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "fampnn_parity")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
  if not args.oracle_dir:
    print("ERROR: set AMINX_FAMPNN_ORACLE_DIR or --oracle-dir", file=sys.stderr)
    return 2
  args.work_dir.mkdir(parents=True, exist_ok=True)
  result = _json_safe(run(args))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  bth = os.environ.get("BTH_RESULTS_PATH")
  if bth:
    Path(bth).parent.mkdir(parents=True, exist_ok=True)
    Path(bth).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  LOG.info("clean=%s n_failed=%s/%s", result["clean"], result["n_failed"], result["n_listed"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

_ = sd_mod
