"""Graded exact AR-then-refine parity (do not run from the fixer).

Example PDBs, 50 seeds, f64, exact token match. Pass is 1.0, inconclusive is
``[0.99, 1)``, fail is below.

Negative controls, each of which must fail because it moved the measured
tokens: ``ar_mask_present_chain_m_pos``, ``wrong_partition_sign``,
``ntoc_refine_order``. The mask control needs the pinned fixed-position
sidecar; on these PDBs ``chain_M_pos`` is otherwise identically 1. Payload
via ``--payload-out``.
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

MUTANTS = ("ar_mask_present_chain_m_pos", "wrong_partition_sign", "ntoc_refine_order")
N_SEEDS = 50
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
EXAMPLE_SHA256 = {
  "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
  "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
  "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
  "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
  "6w25.pdb": "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086",
  "swe1_ligand.pdb": "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6",
}
_WORK = "potts_ar_refine_exact"

# ``forward_context`` has no ``chain_M_pos`` argument. The mask control reads
# the structure the arm is decoding.
_CONTROL_CHAIN_M_POS: Any = None


def _parse(argv: list[str]) -> Any:
  parser = common.build_parser(__doc__)
  parser.set_defaults(mutants=",".join(MUTANTS))
  return parser.parse_args(argv)


def _install_mutant(arm: str | None) -> None:
  """Install one negative control. Each one moves a different observable.

  ``ar_mask_present_chain_m_pos`` multiplies the autoregressive mask by
  ``chain_M_pos``. The clean mask is ``present``; the extra factor changes
  decoder logits wherever a position is fixed.

  ``wrong_partition_sign`` negates ``_energy_vector``. Refine logits are
  ``-energy / temperature``, so the sign flip inverts the softmax even when
  binding is ``none`` (the function returns the positional energy).

  ``ntoc_refine_order`` replaces the autoregressive refine order with
  ``arange``. The sweep visits residues N-to-C instead of in the sampled
  order, so later positions condition on a different prefix.
  """
  if arm in (None, "clean"):
    return
  if arm not in MUTANTS:
    msg = f"unknown mutant {arm}"
    raise SystemExit(msg)
  if arm == "ar_mask_present_chain_m_pos":
    import jax.numpy as jnp

    from aminx.families.potts_mpnn import decode as decode_mod

    context = decode_mod.forward_context

    def _scaled(h_v, h_e, e_idx, rank_flat, present):  # noqa: ANN001
      mask_bw, h_exv = context(h_v, h_e, e_idx, rank_flat, present)
      if _CONTROL_CHAIN_M_POS is None:
        return mask_bw, h_exv
      scale = jnp.asarray(_CONTROL_CHAIN_M_POS, dtype=present.dtype)
      return mask_bw * scale[:, None], h_exv * scale[:, None, None]

    decode_mod.forward_context = _scaled  # ty: ignore[invalid-assignment]
    return
  if arm == "wrong_partition_sign":
    from aminx.families.potts_mpnn import refine as refine_mod

    energy = refine_mod._energy_vector

    def _flipped(
      seq,
      position,
      etab,
      e_idx,
      pad_valid,
      tables,
      *,
      binding,
      tied,
    ):  # noqa: ANN001
      return -energy(
        seq,
        position,
        etab,
        e_idx,
        pad_valid,
        tables,
        binding=binding,
        tied=tied,
      )

    refine_mod._energy_vector = _flipped  # ty: ignore[invalid-assignment]
    return
  from aminx.families.potts_mpnn import sample_host as host_mod

  def _ntoc(ar_order, chain_mask, randn, *, num_samples, chain_suffix, stored_orders_present):  # noqa: ANN001
    import numpy as np

    del ar_order, randn, num_samples, chain_suffix, stored_orders_present
    return np.arange(len(chain_mask), dtype=np.int32)

  host_mod.upstream_refine_order = _ntoc  # ty: ignore[invalid-assignment]


def _units(job: dict[str, Any], arm: str, checkpoint: Path) -> list[graded_resume.Unit]:
  return common.units(
    job,
    arm,
    checkpoint,
    Path(__file__).resolve(),
    lambda name, _payload: [f"{name}:seed{seed}" for seed in range(N_SEEDS)],
  )


def _build_job(root: Path) -> dict[str, Any]:
  # Same sidecar as potts_ar_decode. chain_M_pos has to be non-degenerate here too.
  return {"structures": common.example_structures(root, EXAMPLE_SHA256)}


def _load_oracle(root: Path, checkpoint: Path) -> dict[str, Any]:
  # The sidecar asks for an f64 comparison. ``model.double()`` is the upstream
  # float64 forward; the dtype shims keep the positional encoding and RBF in
  # that dtype.
  return common.load_potts_oracle(root, checkpoint, double=True)


def _bank(root: Path, job: dict[str, Any], work: Path) -> dict[str, common.OracleFeat]:
  feats: dict[str, common.OracleFeat] = {}
  cells: dict[str, dict[str, Any]] = {}
  for name, payload in job["structures"].items():
    feat = common.featurize_upstream(root, Path(payload["pdb"]), common.structure_fixed(payload))
    feats[str(name)] = feat
    width = common.refine_width(feat.mask, feat.chain_m, feat.chain_m_pos)
    for seed in range(N_SEEDS):
      cell_id = f"{name}:seed{seed}"
      rng = common.draw_rng(cell_id)
      cells[cell_id] = {
        "randn": rng.standard_normal(feat.length),
        "uniforms": rng.random(feat.length),
        "refine_uniforms": rng.random((1, width)),
      }
  common.write_draws(work, cells)
  return feats


def _oracle_one(
  prepared: dict[str, Any],
  feat: common.OracleFeat,
  cell: dict[str, Any],
) -> list[int]:
  import numpy as np
  import potts_mpnn_utils as potts
  import torch
  from oracle_shims.potts import (
    injected_uniform_draws,
    rbf_follows_input_dtype,
    weight_dtype_positional_encodings,
  )

  if str(prepared["root"]) not in sys.path:
    sys.path.insert(0, str(prepared["root"]))
  import etab_utils
  import run_utils

  model = prepared["model"]
  dtype = torch.float64
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
  zeros = np.zeros(common.VOCAB, dtype=np.float64)
  randn = common.as_torch(np.asarray(cell["randn"]).reshape(1, -1), floating=dtype)
  ar_stream = np.asarray(cell["uniforms"], dtype=np.float64).reshape(1, -1)
  with (
    weight_dtype_positional_encodings(potts.PositionalEncodings),
    rbf_follows_input_dtype(potts.ProteinFeatures),
  ):
    h_v, e_idx, h_e, etab = model.run_encoder(x, mask, residue_idx, chain_encoding)
    with injected_uniform_draws(ar_stream):
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
    ar_tokens = common.ints(output["S"][0].detach().cpu().tolist())
    order = [int(index) for index in output["decoding_order"].detach().cpu().reshape(-1)]
    seq = "".join(etab_utils.ints_to_seq(ar_tokens))
    h_ex = run_utils.cat_neighbors_nodes(torch.zeros_like(h_v), h_e, e_idx)
    h_exv = run_utils.cat_neighbors_nodes(h_v, h_ex, e_idx)
    constant = torch.zeros(common.VOCAB, dtype=dtype)
    constant[20] = 1.0
    constant_bias = torch.zeros(common.VOCAB, dtype=dtype)
    refine_stream = np.asarray(cell["refine_uniforms"], dtype=np.float64).reshape(1, -1)
    with injected_uniform_draws(refine_stream):
      seq_out = run_utils.optimize_sequence(
        seq,
        etab,
        e_idx,
        mask * chain_m_pos,
        chain_m,
        "potts",
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
        omit_AA_mask=omit,
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
    name = unit.unit_id.split(":seed", 1)[0]
    cell = common.read_draws(work, unit.unit_id)
    return _oracle_one(prepared, feats[name], cell)

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  scored = {unit.unit_id: payload for unit, payload in zip(units, payloads, strict=True)}
  common.write_payload(Path(args.payload_out), scored)


def exact_tokens(
  decode: Any,
  refiner: Any,
  arrays: dict[str, Any],
  cell: dict[str, Any],
) -> list[int]:
  """AR-decode, then one ``potts`` refine sweep, both on the banked stream."""
  import jax.numpy as jnp
  import numpy as np

  from aminx.families.potts_mpnn.etab import pad_etab_energy
  from aminx.families.potts_mpnn.sample_host import _dummy_binding, upstream_refine_order

  global _CONTROL_CHAIN_M_POS  # noqa: PLW0603

  _CONTROL_CHAIN_M_POS = arrays["chain_m_pos"]
  length = int(arrays["length"])
  decoded = decode(
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
    jnp.asarray(cell["randn"]),
    jnp.asarray(cell["uniforms"]),
    jnp.asarray(arrays["omit"]),
    jnp.asarray(arrays["bias"]),
    jnp.asarray(arrays["bias_by_res"]),
    jnp.asarray(arrays["pssm_coef"]),
    jnp.asarray(arrays["pssm_bias"]),
    jnp.asarray(arrays["pssm_log_odds_mask"]),
    jnp.asarray(arrays["omit_aa_mask"]),
    temperature=common.AR_TEMPERATURE,
    pssm_multi=0.0,
    pssm_bias_flag=False,
    pssm_log_odds_flag=False,
  )
  order = upstream_refine_order(
    np.asarray(decoded.decoding_order),
    np.asarray(arrays["chain_mask"]),
    np.asarray(cell["randn"]),
    num_samples=1,
    chain_suffix="",
    stored_orders_present=True,
  )
  dtype = jnp.asarray(arrays["h_v"]).dtype
  tables = _dummy_binding(length, 22, dtype)
  refined = refiner(
    "potts",
    decoded.sequence.astype(jnp.int32),
    pad_etab_energy(jnp.asarray(arrays["forward"])),
    jnp.asarray(arrays["e_idx"]),
    jnp.asarray(arrays["pad_valid"]),
    jnp.asarray(arrays["present"]),
    jnp.asarray(arrays["chain_mask"]),
    jnp.asarray(arrays["chain_m_pos"]),
    jnp.asarray(order, dtype=jnp.int32),
    jnp.asarray(cell["refine_uniforms"]),
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
    temperature=common.REFINE_TEMPERATURE,
    pssm_multi=0.0,
    pssm_bias_flag=False,
    pssm_log_odds_flag=False,
    binding="none",
    tied=False,
    tied_epistasis=False,
    max_iters=1,
  )
  return common.ints(refined.sequence[:length])


def _cast_ready(ready: dict[str, Any]) -> dict[str, Any]:
  import numpy as np

  casted: dict[str, Any] = {}
  for key, value in ready.items():
    if key == "length":
      casted[key] = value
      continue
    array = np.asarray(value)
    if array.dtype.kind == "f":
      casted[key] = array.astype(np.float64)
    else:
      casted[key] = array
  return casted


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
  import jax
  import jax.numpy as jnp

  from aminx.families.potts_mpnn.decode import PottsARDecode
  from aminx.families.potts_mpnn.driver import PottsMPNNDriver
  from aminx.families.potts_mpnn.model import cast_floating
  from aminx.families.potts_mpnn.refine import PottsRefine

  # f64 is the pre-registered comparison. Enabling it here stays out of ``src/``.
  jax.config.update("jax_enable_x64", True)
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
  model = None
  if remaining > 0:
    loaded = PottsMPNNDriver().load(SimpleNamespace(model_local_path=checkpoint))
    model = cast_floating(loaded, jnp.float64)
    for name, payload in job["structures"].items():
      encoded[str(name)] = _encoded(
        model,
        _cast_ready(common.prepare_aminx(Path(payload["pdb"]), common.structure_fixed(payload))),
      )

  def compute(unit: graded_resume.Unit) -> list[int]:
    if model is None:
      msg = "aminx model was not loaded"
      raise RuntimeError(msg)
    name = unit.unit_id.split(":seed", 1)[0]
    arrays = encoded[name]
    decode = PottsARDecode(
      layers=model.mpnn.decoder.layers,
      w_s_embed=model.mpnn.w_s_embed,
      w_out=model.mpnn.w_out,
    )
    refiner = PottsRefine(
      layers=model.mpnn.decoder.layers,
      w_s_embed=model.mpnn.w_s_embed,
      w_out=model.mpnn.w_out,
    )
    cell = common.read_draws(work, unit.unit_id)
    return exact_tokens(decode, refiner, arrays, cell)

  payloads, _stats = graded_resume.run_units(units, compute, cache, resume=bool(args.resume))
  got = [list(payload) for payload in payloads]
  ref = [list(upstream[unit.unit_id]) for unit in units]
  match = common.token_match(got, ref)
  return {"band": common.band(match), "match": match, "n_seeds": N_SEEDS}


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
  logger = common.logger_for("potts_ar_refine_exact")
  results = common.parent(
    args,
    logger,
    script=Path(__file__).resolve(),
    work_name=_WORK,
    checkpoint_sha=CHECKPOINT_SHA256,
    build_job=_build_job,
    units_for=_units,
  )
  path = common.results_path("potts_ar_refine_exact")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
