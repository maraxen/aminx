"""Graded Potts autoregressive decode (do not run from the fixer).

Exact-token match against upstream ``decoder`` on the example PDBs. Pass when
every token matches. ``[0.99, 1)`` is inconclusive. Below ``0.99`` is fail.

Negative control ``ar_mask_present_chain_m_pos`` multiplies the autoregressive
mask by ``chain_M_pos`` and must land in the fail band on its own. The example
PDBs have no fixed positions, so the job applies the pinned sidecar in
``fixtures/potts_ar_fixed_positions.json``; otherwise that product is the
identity. Results go to ``--payload-out`` (never stdout) and ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

import graded_resume
import potts_graded_common as common

MUTANT_ID = "ar_mask_present_chain_m_pos"
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
EXAMPLE_SHA256 = {
  "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
  "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
  "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
  "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
  "6w25.pdb": "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086",
  "swe1_ligand.pdb": "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6",
}
_WORK = "potts_ar_decode"

# ``forward_context`` does not take ``chain_M_pos``. The control closes over the
# structure currently being decoded.
_CONTROL_CHAIN_M_POS: Any = None


def _parse(argv: list[str]) -> Any:
  parser = common.build_parser(__doc__)
  parser.set_defaults(mutants=MUTANT_ID)
  return parser.parse_args(argv)


def _install_mutant(arm: str | None) -> None:
  """Multiply AR ``m`` by ``chain_M_pos``.

  The clean mask is ``present``. Scaling ``mask_bw`` and the forward context
  by ``chain_M_pos`` drops neighbors at fixed positions, which moves the
  decoder logits and therefore the inverse-CDF token. Exact match must break
  wherever ``chain_M_pos`` is not identically 1 on present rows.
  """
  if arm in (None, "clean"):
    return
  if arm != MUTANT_ID:
    msg = f"unknown mutant {arm}"
    raise SystemExit(msg)
  import jax.numpy as jnp

  from aminx.families.potts_mpnn import decode as decode_mod

  original = decode_mod.forward_context

  def _scaled(h_v, h_e, e_idx, rank_flat, present):  # noqa: ANN001
    mask_bw, h_exv = original(h_v, h_e, e_idx, rank_flat, present)
    if _CONTROL_CHAIN_M_POS is None:
      return mask_bw, h_exv
    scale = jnp.asarray(_CONTROL_CHAIN_M_POS, dtype=present.dtype)
    return mask_bw * scale[:, None], h_exv * scale[:, None, None]

  decode_mod.forward_context = _scaled  # ty: ignore[invalid-assignment]


def _units(job: dict[str, Any], arm: str, checkpoint: Path) -> list[graded_resume.Unit]:
  return common.units(
    job,
    arm,
    checkpoint,
    Path(__file__).resolve(),
    lambda name, _payload: [name],
  )


def _build_job(root: Path) -> dict[str, Any]:
  # The sidecar is what makes chain_M_pos drop rows. The PDBs alone cannot.
  return {"structures": common.example_structures(root, EXAMPLE_SHA256)}


def _load_oracle(root: Path, checkpoint: Path) -> dict[str, Any]:
  return common.load_potts_oracle(root, checkpoint, double=False)


def _bank(root: Path, job: dict[str, Any], work: Path) -> dict[str, common.OracleFeat]:
  feats: dict[str, common.OracleFeat] = {}
  cells: dict[str, dict[str, Any]] = {}
  for name, payload in job["structures"].items():
    feat = common.featurize_upstream(root, Path(payload["pdb"]), common.structure_fixed(payload))
    feats[str(name)] = feat
    rng = common.draw_rng(str(name))
    cells[str(name)] = {
      "randn": rng.standard_normal(feat.length),
      "uniforms": rng.random(feat.length),
    }
  common.write_draws(work, cells)
  return feats


def _oracle_one(
  prepared: dict[str, Any],
  feat: common.OracleFeat,
  cell: dict[str, Any],
) -> list[int]:
  import numpy as np
  import torch
  from oracle_shims.potts import injected_uniform_draws

  model = prepared["model"]
  dtype = torch.float64 if prepared["double"] else torch.float32
  x = common.as_torch(feat.x, floating=dtype)
  mask = common.as_torch(feat.mask, floating=dtype)
  chain_m = common.as_torch(feat.chain_m, floating=dtype)
  chain_m_pos = common.as_torch(feat.chain_m_pos, floating=dtype)
  residue_idx = common.as_torch(feat.residue_idx, floating=None)
  chain_encoding = common.as_torch(feat.chain_encoding, floating=None)
  s_true = common.as_torch(feat.s, floating=None)
  omit = common.as_torch(feat.omit_aa_mask, floating=dtype)
  bias_by_res = common.as_torch(feat.bias_by_res, floating=dtype)
  pssm_coef = common.as_torch(feat.pssm_coef, floating=dtype)
  pssm_bias = common.as_torch(feat.pssm_bias, floating=dtype)
  h_v, e_idx, h_e, _etab = model.run_encoder(x, mask, residue_idx, chain_encoding)
  randn = common.as_torch(np.asarray(cell["randn"]).reshape(1, -1), floating=dtype)
  uniforms = np.asarray(cell["uniforms"], dtype=np.float64).reshape(1, -1)
  zeros = np.zeros(common.VOCAB, dtype=np.float64)
  with injected_uniform_draws(uniforms):
    output, _probs = model.decoder(
      h_v,
      e_idx,
      h_e,
      randn,
      s_true,
      chain_m,
      chain_encoding,
      residue_idx,
      mask=mask,
      temperature=common.AR_TEMPERATURE,
      omit_AAs_np=zeros,
      bias_AAs_np=zeros,
      chain_M_pos=chain_m_pos,
      omit_AA_mask=omit,
      pssm_coef=pssm_coef,
      pssm_bias=pssm_bias,
      pssm_multi=0.0,
      pssm_log_odds_flag=False,
      pssm_log_odds_mask=None,
      pssm_bias_flag=False,
      bias_by_res=bias_by_res,
    )
  return common.ints(output["S"][0].detach().cpu().tolist())


def _oracle_worker(args: Any) -> None:
  if args.payload_out is None or args.job is None:
    msg = "oracle worker requires --job and --payload-out"
    raise SystemExit(msg)
  checkpoint = common.checkpoint_path(args)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  work = common.work_dir(args, _WORK)
  feats = _bank(Path(args.potts_root), job, work)
  units = _units(job, graded_resume.ORACLE_ARM, checkpoint)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  prepared = _load_oracle(Path(args.potts_root), checkpoint) if remaining > 0 else None

  def compute(unit: graded_resume.Unit) -> list[int]:
    if prepared is None:
      msg = "oracle model was not loaded"
      raise RuntimeError(msg)
    cell = common.read_draws(work, unit.unit_id)
    return _oracle_one(prepared, feats[unit.unit_id], cell)

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  scored = {unit.unit_id: payload for unit, payload in zip(units, payloads, strict=True)}
  common.write_payload(Path(args.payload_out), scored)


def decode_tokens(
  decode: Any,
  arrays: dict[str, Any],
  randn: Any,
  uniforms: Any,
  *,
  temperature: float,
) -> list[int]:
  """Call ``PottsARDecode`` and return its tokens. Draws are the banked stream."""
  import jax.numpy as jnp

  length = int(arrays["length"])
  result = decode(
    jnp.asarray(arrays["h_v"]),
    jnp.asarray(arrays["h_e"]),
    jnp.asarray(arrays["e_idx"]),
    jnp.asarray(arrays["present"]),
    jnp.asarray(arrays["pad_valid"]),
    jnp.asarray(arrays["s_true"]),
    jnp.asarray(arrays["chain_mask"]),
    jnp.asarray(arrays["chain_m_pos"]),
    jnp.asarray(arrays["tie_groups"]),
    jnp.asarray(arrays["tied_beta"]),
    jnp.asarray(randn),
    jnp.asarray(uniforms),
    jnp.asarray(arrays["omit"]),
    jnp.asarray(arrays["bias"]),
    jnp.asarray(arrays["bias_by_res"]),
    jnp.asarray(arrays["pssm_coef"]),
    jnp.asarray(arrays["pssm_bias"]),
    jnp.asarray(arrays["pssm_log_odds_mask"]),
    jnp.asarray(arrays["omit_aa_mask"]),
    temperature=temperature,
    pssm_multi=0.0,
    pssm_bias_flag=False,
    pssm_log_odds_flag=False,
  )
  return common.ints(result.sequence[:length])


def _encoded(model: Any, ready: dict[str, Any]) -> dict[str, Any]:
  import jax.numpy as jnp

  from aminx.families.potts_mpnn.sample_host import _encode

  h_v, h_e, e_idx, _forward, _table = _encode(
    model,
    jnp.asarray(ready["coords"]),
    jnp.asarray(ready["present"]),
    jnp.asarray(ready["residue_idx"]),
    jnp.asarray(ready["chain_index"]),
    jnp.asarray(ready["pad_valid"]),
  )
  encoded = dict(ready)
  encoded["h_v"] = h_v
  encoded["h_e"] = h_e
  encoded["e_idx"] = e_idx
  return encoded


def _run_arm(args: Any) -> dict[str, Any]:
  global _CONTROL_CHAIN_M_POS

  from types import SimpleNamespace

  from aminx.families.potts_mpnn.decode import PottsARDecode
  from aminx.families.potts_mpnn.driver import PottsMPNNDriver

  _install_mutant(args.arm)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  upstream = json.loads(Path(args.upstream).read_text(encoding="utf-8"))
  checkpoint = common.checkpoint_path(args)
  arm_id = str(args.arm or "clean")
  units = _units(job, arm_id, checkpoint)
  work = common.work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  model = None
  ready: dict[str, dict[str, Any]] = {}
  if remaining > 0:
    model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=checkpoint))
    for name, payload in job["structures"].items():
      ready[str(name)] = _encoded(
        model,
        common.prepare_aminx(Path(payload["pdb"]), common.structure_fixed(payload)),
      )

  def compute(unit: graded_resume.Unit) -> list[int]:
    global _CONTROL_CHAIN_M_POS  # noqa: PLW0603

    if model is None:
      msg = "aminx model was not loaded"
      raise RuntimeError(msg)
    arrays = ready[unit.unit_id]
    _CONTROL_CHAIN_M_POS = arrays["chain_m_pos"]
    decode = PottsARDecode(
      layers=model.mpnn.decoder.layers,
      w_s_embed=model.mpnn.w_s_embed,
      w_out=model.mpnn.w_out,
    )
    cell = common.read_draws(work, unit.unit_id)
    return decode_tokens(
      decode,
      arrays,
      cell["randn"],
      cell["uniforms"],
      temperature=common.AR_TEMPERATURE,
    )

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  got = [list(payload) for payload in payloads]
  ref = [list(upstream[unit.unit_id]) for unit in units]
  match = common.token_match(got, ref)
  return {"band": common.band(match), "match": match}


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  if args.oracle_worker:
    _oracle_worker(args)
    return
  if args.arm:
    if args.payload_out is None:
      msg = "arm requires --payload-out"
      raise SystemExit(msg)
    common.write_payload(Path(args.payload_out), _run_arm(args))
    return
  logger = common.logger_for("potts_ar_decode")
  results = common.parent(
    args,
    logger,
    script=Path(__file__).resolve(),
    work_name=_WORK,
    checkpoint_sha=CHECKPOINT_SHA256,
    build_job=_build_job,
    units_for=_units,
  )
  path = common.results_path("potts_ar_decode")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
