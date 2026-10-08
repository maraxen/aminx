"""Graded Potts refine parity (do not run from the fixer).

Every optimization mode, with injected uniforms, must match upstream tokens
exactly. Pass is match 1.0, inconclusive is ``[0.99, 1)``, fail is below.

Negative control ``unmasked_x_logit`` leaves letter X drawable and must FAIL
because that mutation moved the tokens. Payload via ``--payload-out``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

import graded_resume
import potts_graded_common as common

MUTANT_ID = "unmasked_x_logit"
MODES = ("potts", "potts_converge", "nodes")
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
EXAMPLE_SHA256 = {
  "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
  "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
  "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
  "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
  "6w25.pdb": "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086",
  "swe1_ligand.pdb": "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6",
}
_WORK = "potts_refine"


def _parse(argv: list[str]) -> Any:
  parser = common.build_parser(__doc__)
  parser.set_defaults(mutants=MUTANT_ID)
  return parser.parse_args(argv)


def _install_mutant(arm: str | None) -> None:
  """Stop masking logit 20.

  ``mask_refine_x`` subtracts ``1e8`` from X so that letter is not in the
  simplex. The identity leaves X drawable, which moves every inverse-CDF
  boundary. A shared uniform then lands on a different token.
  """
  if arm in (None, "clean"):
    return
  if arm != MUTANT_ID:
    msg = f"unknown mutant {arm}"
    raise SystemExit(msg)
  from aminx.families.potts_mpnn import decode as decode_mod

  def _identity(logits):  # noqa: ANN001
    return logits

  decode_mod.mask_refine_x = _identity  # ty: ignore[invalid-assignment]


def _units(job: dict[str, Any], arm: str, checkpoint: Path) -> list[graded_resume.Unit]:
  return common.units(
    job,
    arm,
    checkpoint,
    Path(__file__).resolve(),
    lambda name, _payload: [f"{name}:{mode}" for mode in MODES],
  )


def _build_job(root: Path) -> dict[str, Any]:
  folder = root / "inputs" / "example_pdbs"
  structures: dict[str, Any] = {}
  for name, digest in EXAMPLE_SHA256.items():
    path = folder / name
    common.check_pdb(path, digest)
    structures[path.stem] = {"pdb": str(path), "sha256": digest}
  return {"structures": structures}


def _torch_order(chain_mask: Any, randn: Any) -> Any:
  """Upstream ``argsort((chain_mask + 1e-4) · |randn|)`` from ``sample_seqs.py``."""
  import torch

  mask = torch.as_tensor(chain_mask, dtype=torch.float64).reshape(-1)
  noise = torch.as_tensor(randn, dtype=torch.float64).reshape(-1)
  return torch.argsort((mask + 1.0e-4) * noise.abs()).cpu().numpy()


def _load_oracle(root: Path, checkpoint: Path) -> dict[str, Any]:
  return common.load_potts_oracle(root, checkpoint, double=False)


def _bank(root: Path, job: dict[str, Any], work: Path) -> dict[str, common.OracleFeat]:
  feats: dict[str, common.OracleFeat] = {}
  cells: dict[str, dict[str, Any]] = {}
  for name, payload in job["structures"].items():
    feat = common.featurize_upstream(root, Path(payload["pdb"]))
    feats[str(name)] = feat
    width = common.refine_width(feat.mask, feat.chain_m, feat.chain_m_pos)
    rng = common.draw_rng(str(name))
    randn = rng.standard_normal(feat.length)
    cells[str(name)] = {
      "randn": randn,
      "uniforms": rng.random((common.REFINE_SWEEPS, width)),
      # Upstream argsort, banked once, so the aminx sweep does not re-break ties.
      "order": _torch_order(feat.chain_m, randn),
    }
  common.write_draws(work, cells)
  return feats


def _oracle_one(
  prepared: dict[str, Any],
  feat: common.OracleFeat,
  cell: dict[str, Any],
  mode: str,
) -> list[int]:
  import numpy as np
  import torch
  from oracle_shims.potts import injected_uniform_draws

  if str(prepared["root"]) not in sys.path:
    sys.path.insert(0, str(prepared["root"]))
  import run_utils

  model = prepared["model"]
  dtype = torch.float64 if prepared["double"] else torch.float32
  x = common.as_torch(feat.x, floating=dtype)
  mask = common.as_torch(feat.mask, floating=dtype)
  chain_m = common.as_torch(feat.chain_m, floating=dtype)
  chain_m_pos = common.as_torch(feat.chain_m_pos, floating=dtype)
  residue_idx = common.as_torch(feat.residue_idx, floating=None)
  chain_encoding = common.as_torch(feat.chain_encoding, floating=None)
  bias_by_res = common.as_torch(feat.bias_by_res, floating=dtype)
  pssm_coef = common.as_torch(feat.pssm_coef, floating=dtype)
  pssm_bias = common.as_torch(feat.pssm_bias, floating=dtype)
  omit_aa = common.as_torch(feat.omit_aa_mask, floating=dtype)
  h_v, e_idx, h_e, etab = model.run_encoder(x, mask, residue_idx, chain_encoding)
  h_ex = run_utils.cat_neighbors_nodes(torch.zeros_like(h_v), h_e, e_idx)
  h_exv = run_utils.cat_neighbors_nodes(h_v, h_ex, e_idx)
  order = [int(index) for index in np.asarray(cell["order"]).reshape(-1)]
  constant = torch.zeros(common.VOCAB, dtype=dtype)
  constant[20] = 1.0
  constant_bias = torch.zeros(common.VOCAB, dtype=dtype)
  import etab_utils

  stream = np.asarray(cell["uniforms"], dtype=np.float64).reshape(1, -1)
  with injected_uniform_draws(stream):
    seq_out = run_utils.optimize_sequence(
      feat.seq,
      etab,
      e_idx,
      mask * chain_m_pos,
      chain_m,
      mode,
      etab_utils.seq_to_ints,
      common.REFINE_TEMPERATURE,
      constant=constant,
      constant_bias=constant_bias,
      bias_by_res=bias_by_res,
      pssm_bias_flag=False,
      pssm_coef=pssm_coef,
      pssm_bias=pssm_bias,
      pssm_multi=0.0,
      pssm_log_odds_flag=False,
      pssm_log_odds_mask=None,
      omit_AA_mask=omit_aa,
      model=model,
      h_E=h_e,
      h_EXV_encoder=h_exv,
      h_V=h_v,
      decoding_order=order,
      partition_etabs=None,
      partition_index=None,
      inter_mask=None,
      binding_optimization=None,
      vocab=common.VOCAB,
    )
  return common.ints(seq_out.detach().cpu().tolist())


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
    name, mode = unit.unit_id.split(":", 1)
    cell = common.read_draws(work, name)
    return _oracle_one(prepared, feats[name], cell, mode)

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  scored = {unit.unit_id: payload for unit, payload in zip(units, payloads, strict=True)}
  common.write_payload(Path(args.payload_out), scored)


def refine_tokens(
  refiner: Any,
  arrays: dict[str, Any],
  order: Any,
  uniforms: Any,
  *,
  mode: str,
  temperature: float,
) -> list[int]:
  """Call ``PottsRefine`` on one mode. ``uniforms`` is the banked stream."""
  import jax.numpy as jnp

  from aminx.families.potts_mpnn.etab import pad_etab_energy
  from aminx.families.potts_mpnn.sample_host import _dummy_binding

  length = int(arrays["length"])
  dtype = jnp.asarray(arrays["h_v"]).dtype
  tables = _dummy_binding(length, 22, dtype)
  result = refiner(
    mode,
    jnp.asarray(arrays["s_true"], dtype=jnp.int32),
    pad_etab_energy(jnp.asarray(arrays["forward"])),
    jnp.asarray(arrays["e_idx"]),
    jnp.asarray(arrays["pad_valid"]),
    jnp.asarray(arrays["present"]),
    jnp.asarray(arrays["chain_mask"]),
    jnp.asarray(arrays["chain_m_pos"]),
    jnp.asarray(order, dtype=jnp.int32),
    jnp.asarray(uniforms),
    jnp.asarray(arrays["omit"]),
    jnp.asarray(arrays["bias"]),
    jnp.asarray(arrays["bias_by_res"]),
    jnp.asarray(arrays["pssm_coef"]),
    jnp.asarray(arrays["pssm_bias"]),
    jnp.asarray(arrays["pssm_log_odds_mask"]),
    jnp.asarray(arrays["omit_aa_mask"]),
    jnp.asarray(arrays["tie_groups"]),
    jnp.asarray(arrays["tied_beta"]),
    jnp.asarray(arrays["h_v"]),
    jnp.asarray(arrays["h_e"]),
    tables,
    temperature=temperature,
    pssm_multi=0.0,
    pssm_bias_flag=False,
    pssm_log_odds_flag=False,
    binding="none",
    tied=False,
    tied_epistasis=False,
    max_iters=common.REFINE_SWEEPS,
  )
  return common.ints(result.sequence[:length])


def _encoded(model: Any, ready: dict[str, Any]) -> dict[str, Any]:
  import jax.numpy as jnp

  from aminx.families.potts_mpnn.sample_host import _encode

  h_v, h_e, e_idx, forward, _table = _encode(
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
  encoded["forward"] = forward
  return encoded


def _run_arm(args: Any) -> dict[str, Any]:
  from aminx.families.potts_mpnn.driver import PottsMPNNDriver
  from aminx.families.potts_mpnn.refine import PottsRefine

  _install_mutant(args.arm)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  upstream = json.loads(Path(args.upstream).read_text(encoding="utf-8"))
  checkpoint = common.checkpoint_path(args)
  arm_id = str(args.arm or "clean")
  units = _units(job, arm_id, checkpoint)
  work = common.work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  encoded: dict[str, dict[str, Any]] = {}
  if remaining > 0:
    model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=checkpoint))
    for name, payload in job["structures"].items():
      encoded[str(name)] = _encoded(model, common.prepare_aminx(Path(payload["pdb"])))
  else:
    model = None

  def compute(unit: graded_resume.Unit) -> list[int]:
    if model is None:
      msg = "aminx model was not loaded"
      raise RuntimeError(msg)
    name, mode = unit.unit_id.split(":", 1)
    arrays = encoded[name]
    refiner = PottsRefine(
      layers=model.mpnn.decoder.layers,
      w_s_embed=model.mpnn.w_s_embed,
      w_out=model.mpnn.w_out,
    )
    cell = common.read_draws(work, name)
    order = cell["order"]
    return refine_tokens(
      refiner,
      arrays,
      order,
      cell["uniforms"],
      mode=mode,
      temperature=common.REFINE_TEMPERATURE,
    )

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  got = [list(payload) for payload in payloads]
  ref = [list(upstream[unit.unit_id]) for unit in units]
  match = common.token_match(got, ref)
  return {"band": common.band(match), "match": match, "modes": list(MODES)}


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
  logger = common.logger_for("potts_refine")
  results = common.parent(
    args,
    logger,
    script=Path(__file__).resolve(),
    work_name=_WORK,
    checkpoint_sha=CHECKPOINT_SHA256,
    build_job=_build_job,
    units_for=_units,
  )
  path = common.results_path("potts_refine")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
