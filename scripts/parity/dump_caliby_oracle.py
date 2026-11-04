"""Sealed upstream oracle dump for Caliby (ProteinDesignLab/caliby), real code and real weights.

Runs INSIDE a torch environment that can import the upstream repo (torch, lightning, omegaconf,
hydra, einops, biotite, atomworks, torchtyping); it imports no JAX and no aminx code. It loads a
released Caliby checkpoint through the upstream ``load_from_checkpoint(LitSeqDenoiser, ...)``, builds
the batch through the upstream ``get_sd_batch`` / ``initialize_sampling_masks`` (the same calls as
``caliby.api.CalibyModel.score``), and records every stage through forward hooks so no upstream file
is edited:

  feat    atom14 coordinates and conditioning masks the model sees (add_atom14_feats)
  tf      TokenFeatures output (edge features E, E_idx, D_neighbors)
  enc     per-layer encoder (h_V, h_E)
  dec     per-layer decoder (h_V, h_ESV)
  potts   raw GraphPotts output (32 AF3 states) and the reduced (22 state) aux the sampler consumes
  score   ``SeqDenoiser.score_samples`` U and U_i for the wild type, plus compute_potts_energy for
          seeded alternative sequences
  ens     Gaussian-conformer ensemble: ``_generate_gaussian_conformers`` with a seeded global RNG
          (the injection point is torch.randn_like at caliby/model/seq_denoiser/denoisers/
          atom_mpnn_denoiser.py:571), the per-conformer atom14 coordinates, and the tied-aggregate
          Potts parameters from ``compute_potts_params`` (``_aggregate_potts_params``)
  pk      (packer checkpoints) mpnn trunk h_V, the first SidechainMLP denoiser call with the seeded
          prior x0 (edm_interpolant.py:119) and its output

The dump is written as one uncompressed ``.npz`` plus ``<name>.manifest.json`` holding sha256 of the
dump, of the checkpoint, the upstream commit and porcelain status, library versions, dtype, seed
and every injection site. Float64 mode sets the torch default dtype before the model is built and
asserts every captured floating tensor is float64.

Usage (titanix)::

  PYTHONPATH=<pylib-with-torchtyping> <venv>/bin/python \\
    scripts/parity/dump_caliby_oracle.py --caliby-root /home/solab/repos/caliby \\
    --ckpt /home/solab/repos/caliby_weights/caliby/caliby.ckpt --pdb tests/data/1ubq.pdb \\
    --dtype float64 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PDB_MIRROR_PATH", "")
os.environ.setdefault("CCD_MIRROR_PATH", "")
os.environ.setdefault("MODEL_PARAMS_DIR", "model_params")

log = logging.getLogger("dump_caliby_oracle")

# Injection sites, file:line at upstream commit 41d31560 (read, not assumed).
INJECTION_SITES = {
  "gaussian_conformers": "caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py:571 torch.randn_like(batch['coords'])",
  "scn_prior": "caliby/model/seq_denoiser/denoisers/scn_diffusion_module/edm_interpolant.py:119 torch.randn(*shape)",
  "eval_training_noise": "caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py:132-135 (eval mode: noise = zeros)",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
  return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _np(x: Any, want_float: str) -> Any:  # noqa: ANN401
  """Detach a tensor to numpy; assert floating tensors have the requested precision."""
  import numpy as np  # noqa: PLC0415
  import torch  # noqa: PLC0415

  arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
  if arr.dtype.kind == "f" and str(arr.dtype) != want_float:
    if not bool(np.all((arr == 0) | (arr == 1))):
      msg = f"captured float tensor is {arr.dtype}, expected {want_float}, and is not an exact 0/1 mask"
      raise AssertionError(msg)
    arr = arr.astype(want_float)
  return arr


def main() -> None:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--caliby-root", type=Path, required=True)
  parser.add_argument("--ckpt", type=Path, required=True)
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=20261104)
  parser.add_argument("--noise-std", type=float, default=0.5)
  parser.add_argument("--n-conformers", type=int, default=3)
  parser.add_argument("--n-alt-seqs", type=int, default=4)
  parser.add_argument("--scn-steps", type=int, default=2)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  sys.path.insert(0, str(args.caliby_root))
  import numpy as np  # noqa: PLC0415
  import torch  # noqa: PLC0415

  torch.set_num_threads(4)
  want = args.dtype
  torch_dtype = torch.float64 if want == "float64" else torch.float32
  torch.set_default_dtype(torch_dtype)

  import hydra  # noqa: PLC0415
  from omegaconf import OmegaConf  # noqa: PLC0415

  import caliby.data.const as const  # noqa: PLC0415
  from caliby.checkpoint_utils import get_cfg_from_ckpt, load_from_checkpoint  # noqa: PLC0415
  from caliby.data.data import to  # noqa: PLC0415
  from caliby.eval.eval_utils.inference_dataloader import _InferenceDataset  # noqa: F401, PLC0415
  from caliby.eval.eval_utils.seq_des_utils import get_sd_batch, initialize_sampling_masks  # noqa: PLC0415
  from caliby.model.seq_denoiser.denoisers import atom_mpnn_denoiser as amd  # noqa: PLC0415
  from caliby.model.seq_denoiser.denoisers.seq_design.potts import compute_potts_energy  # noqa: PLC0415
  from caliby.model.seq_denoiser.lit_sd_model import LitSeqDenoiser  # noqa: PLC0415

  cfg, _ = get_cfg_from_ckpt(str(args.ckpt))
  task = str(cfg.model.task)
  model_version = int(cfg.model.denoiser.get("model_version", 0) or 0)
  lit = load_from_checkpoint(LitSeqDenoiser, str(args.ckpt)).eval()
  model = lit.model
  compiled = hasattr(model, "_orig_mod")
  if compiled:
    model = model._orig_mod  # noqa: SLF001
  model = model.to(torch_dtype).eval()
  data_cfg = hydra.utils.instantiate(cfg.data)
  denoiser = model.denoiser
  mpnn = denoiser.mpnn

  def cast(batch: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
      out[key] = value.to(torch_dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
    return out

  # ---- hooks -------------------------------------------------------------------------------------
  rec: dict[str, Any] = {}
  calls: dict[str, int] = {}

  def keyed(name: str) -> str:
    calls[name] = calls.get(name, 0) + 1
    return f"{name}#{calls[name]}"

  def hook_tf(_m: Any, inp: Any, out: Any) -> None:  # noqa: ANN401
    batch = inp[0]
    rec[keyed("tf_in")] = {"atom14_coords": batch["atom14_coords"], "atom14_cond_mask": batch["atom14_cond_mask"]}
    rec[keyed("tf")] = {"E": out[0], "E_idx": out[1], "D_neighbors": out[2]}

  def hook_ws(_m: Any, _i: Any, out: Any) -> None:  # noqa: ANN401
    rec[keyed("W_s")] = {"h_S": out}

  def hook_we(_m: Any, _i: Any, out: Any) -> None:  # noqa: ANN401
    rec[keyed("W_e")] = {"h_E": out}

  def make_layer_hook(tag: str):  # noqa: ANN202
    def _hook(_m: Any, _i: Any, out: Any) -> None:  # noqa: ANN401
      rec[keyed(tag)] = {"h_V": out[0], "h_E": out[1]}

    return _hook

  def hook_potts(_m: Any, inp: Any, out: Any) -> None:  # noqa: ANN401
    rec[keyed("potts")] = {"h": out[0], "J": out[1], "edge_idx": inp[2], "mask_i": inp[3], "mask_ij": inp[4]}

  # Upstream builds one-hot inputs with ``.float()`` (atom_mpnn.py:105-107 for W_s, :331 for the
  # positional encoding), which is float32 even in a float64 model. Upcasting an exact 0/1 tensor loses
  # nothing. Every Linear input that is not already model-dtype is checked to be exactly 0/1 and the
  # run fails otherwise, so this intervention cannot hide a real float32 computation.
  upcast_log: dict[str, int] = {}

  def make_upcast(name: str):  # noqa: ANN202
    def _pre(_m: Any, inp: Any) -> tuple[Any, ...]:  # noqa: ANN401
      x = inp[0]
      if x.is_floating_point() and x.dtype != torch_dtype:
        if not bool(((x == 0) | (x == 1)).all()):
          msg = f"{name}: non-one-hot {x.dtype} input into a {torch_dtype} Linear"
          raise AssertionError(msg)
        upcast_log[name] = upcast_log.get(name, 0) + 1
        x = x.to(torch_dtype)
      return (x, *inp[1:])

    return _pre

  for mod_name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
      mod.register_forward_pre_hook(make_upcast(mod_name))
  mpnn.token_features.register_forward_hook(hook_tf)
  mpnn.W_s.register_forward_hook(hook_ws)
  mpnn.W_e.register_forward_hook(hook_we)
  for i, layer in enumerate(mpnn.encoder_layers):
    layer.register_forward_hook(make_layer_hook(f"enc{i}"))
  for i, layer in enumerate(mpnn.decoder_layers):
    layer.register_forward_hook(make_layer_hook(f"dec{i}"))
  if getattr(mpnn, "use_potts", False):
    mpnn.decoder_S_potts.register_forward_hook(hook_potts)
  if task == "scn_pack":
    sdm = denoiser.scn_diffusion_module
    def hook_pk(_m: Any, args_: Any, kwargs_: Any, out_: Any) -> None:  # noqa: ANN401
      rec[keyed("pk_call")] = {
        "x_scn": args_[0],
        "aatype": kwargs_["aatype"],
        "t": kwargs_["t"],
        "h_V": kwargs_["h_V"],
        "x_self_cond": kwargs_.get("x_scn_self_cond"),
        "x_out": out_[0],
      }

    sdm.scn_denoiser.register_forward_hook(hook_pk, with_kwargs=True)

  # ---- batch -------------------------------------------------------------------------------------
  def fresh_batch() -> dict[str, Any]:
    batch = get_sd_batch([str(args.pdb)], data_cfg=data_cfg, device="cpu")
    return cast(initialize_sampling_masks(batch))

  out: dict[str, Any] = {}
  gap_idx = int(const.AF3_ENCODING.token_to_idx["<G>"])
  base = fresh_batch()
  n_tok = int(base["restype"].shape[1])
  s_wt = base["restype"].argmax(-1)
  out["feat/restype_idx"] = _np(s_wt[0], want)
  out["feat/residue_index"] = _np(base["residue_index"][0], want)
  out["feat/asym_id"] = _np(base["asym_id"][0], want)
  out["feat/token_pad_mask"] = _np(base["token_pad_mask"][0], want)
  out["feat/seq_cond_mask"] = _np(base["seq_cond_mask"][0], want)
  out["feat/atom_cond_mask_atoms"] = _np(base["atom_cond_mask"][0], want)
  out["feat/coords_atoms"] = _np(base["coords"][0], want)
  out["feat/atom_to_token_map"] = _np(base["atom_to_token_map"][0], want)
  out["feat/atom_pad_mask"] = _np(base["atom_pad_mask"][0], want)

  sampling_inputs = OmegaConf.to_container(OmegaConf.load(args.caliby_root / "caliby/configs/seq_des/inference.yaml"), resolve=True)

  def take(prefix: str, key: str, store: str) -> None:
    for field, tensor in rec[key].items():
      out[f"{store}/{prefix}{field}"] = _np(tensor[0] if tensor.shape[0] == 1 else tensor, want)

  if task != "scn_pack":
    # ---- score path (caliby.api.CalibyModel.score -> SeqDenoiser.score_samples) -----------------
    calls.clear()
    rec.clear()
    scored = model.score_samples(fresh_batch(), sampling_inputs=sampling_inputs)
    aux = next(iter(scored.values()))
    out["score/U"] = np.asarray(aux["U"], dtype=want)
    out["score/U_i"] = _np(aux["U_i"], want)
    for key, name in (("tf#1", "tf"), ("W_s#1", "ws"), ("W_e#1", "we")):
      take("", key, name)
    out["feat/atom14_coords"] = _np(rec["tf_in#1"]["atom14_coords"][0], want)
    out["feat/atom14_cond_mask"] = _np(rec["tf_in#1"]["atom14_cond_mask"][0], want)
    for i in range(len(mpnn.encoder_layers)):
      take("", f"enc{i}#1", f"enc{i}")
    for i in range(len(mpnn.decoder_layers)):
      take("", f"dec{i}#1", f"dec{i}")
    p = rec["potts#1"]
    out["potts/h_full"] = _np(p["h"][0], want)
    out["potts/J_full"] = _np(p["J"][0], want)
    out["potts/edge_idx"] = _np(p["edge_idx"][0], want)
    out["potts/mask_i"] = _np(p["mask_i"][0], want)
    out["potts/mask_ij"] = _np(p["mask_ij"][0], want)

    # Reduced-space aux exactly as the sampler consumes it, and alternative-sequence energies.
    pd_aux, batch_after, _ = denoiser.compute_potts_params(fresh_batch(), dict(sampling_inputs))
    out["potts/h_red"] = _np(pd_aux["h"][0], want)
    out["potts/J_red"] = _np(pd_aux["J"][0], want)
    rng = np.random.default_rng(args.seed)
    aa_idx = np.asarray(const.AF3_ENCODING.encode_aa_seq(list("ARNDCQEGHILKMFPSTWYV")), dtype=np.int64)
    alt = aa_idx[rng.integers(0, 20, size=(args.n_alt_seqs, n_tok))]
    out["score/alt_S"] = alt
    alt_u = []
    for row in alt:
      u, _ = compute_potts_energy(
        S=torch.as_tensor(row)[None],
        h=pd_aux["h"],
        J=pd_aux["J"],
        edge_idx=pd_aux["edge_idx"],
        mask_i=pd_aux["mask_i"],
        mask_ij=pd_aux["mask_ij"],
      )
      alt_u.append(float(u[0]))
    out["score/alt_U"] = np.asarray(alt_u, dtype=want)

    # ---- Gaussian-conformer ensemble (the potts_sample prefix up to the sampler) ----------------
    calls.clear()
    rec.clear()
    ens_batch = fresh_batch()
    torch.manual_seed(args.seed)
    amd._generate_gaussian_conformers(batch=ens_batch, noise_std=args.noise_std, n_gaussian_conformers=args.n_conformers)  # noqa: SLF001
    sampling_ens = dict(sampling_inputs)
    sampling_ens["batch_size"] = args.n_conformers
    pd_ens, _, _ = denoiser.compute_potts_params(ens_batch, sampling_ens)
    out["ens/atom14_coords"] = _np(rec["tf_in#1"]["atom14_coords"], want)
    out["ens/atom14_cond_mask"] = _np(rec["tf_in#1"]["atom14_cond_mask"], want)
    out["ens/h"] = _np(pd_ens["h"][0], want)
    out["ens/J"] = _np(pd_ens["J"][0], want)
    out["ens/edge_idx"] = _np(pd_ens["edge_idx"][0], want)
    out["ens/mask_i"] = _np(pd_ens["mask_i"][0], want)
    out["ens/mask_ij"] = _np(pd_ens["mask_ij"][0], want)
    u_ens, _ = compute_potts_energy(
      S=s_wt.cpu(),
      h=pd_ens["h"],
      J=pd_ens["J"],
      edge_idx=pd_ens["edge_idx"],
      mask_i=pd_ens["mask_i"],
      mask_ij=pd_ens["mask_ij"],
    )
    out["ens/U"] = np.asarray(float(u_ens[0]), dtype=want)
  else:
    # ---- packer: trunk and first denoiser calls under a seeded prior ----------------------------
    calls.clear()
    rec.clear()
    pk_inputs = dict(sampling_inputs)
    pk_inputs["scn_packing_cfg"] = {"num_steps": args.scn_steps, "step_scale": float(sampling_inputs["scn_packing_cfg"]["step_scale"])}
    batch = fresh_batch()
    # run_sidechain_packing (seq_des_utils.py:343): pack with the full sequence conditioned.
    batch["seq_cond_mask"] = batch["token_resolved_mask"].clone()
    torch.manual_seed(args.seed)
    prior = sdm.scn_interpolant.sample_prior((1, n_tok, len(const.AF2_SCN_IDXS), 3), torch.device("cpu"))
    torch.manual_seed(args.seed)
    model.sidechain_pack(batch, sampling_inputs=pk_inputs)
    out["pk/prior_x0"] = _np(prior[0], want)
    out["feat/atom14_coords"] = _np(rec["tf_in#1"]["atom14_coords"][0], want)
    out["feat/atom14_cond_mask"] = _np(rec["tf_in#1"]["atom14_cond_mask"][0], want)
    take("", "tf#1", "tf")
    for i in range(len(mpnn.encoder_layers)):
      take("", f"enc{i}#1", f"enc{i}")
    for i in range(len(mpnn.decoder_layers)):
      take("", f"dec{i}#1", f"dec{i}")
    for call in range(1, args.scn_steps + 1):
      for field, tensor in rec[f"pk_call#{call}"].items():
        if tensor is not None:
          out[f"pk/call{call}_{field}"] = _np(tensor[0], want)
    out["pk/sigma_data"] = np.asarray(float(sdm.scn_interpolant.sigma_data), dtype=want)
    for name in ("s_min", "s_max", "rho"):
      out[f"pk/{name}"] = np.asarray(float(getattr(sdm.scn_interpolant, name)), dtype=want)

  # ---- seal --------------------------------------------------------------------------------------
  args.out_dir.mkdir(parents=True, exist_ok=True)
  stem = f"{args.ckpt.stem}_{Path(args.pdb).stem}_{want}"
  dump = args.out_dir / f"{stem}.npz"
  np.savez(dump, **out)
  import biotite  # noqa: PLC0415

  manifest = {
    "dump": dump.name,
    "dump_sha256": _sha256(dump),
    "checkpoint": str(args.ckpt),
    "checkpoint_sha256": _sha256(args.ckpt),
    "task": task,
    "model_version": model_version,
    "dtype": want,
    "seed": args.seed,
    "noise_std": args.noise_std,
    "n_conformers": args.n_conformers,
    "torch_compile_unwrapped": compiled,  # loaded compiled (checkpoint keys carry _orig_mod), then run eager via ._orig_mod
    "pdb": str(args.pdb),
    "pdb_sha256": _sha256(args.pdb),
    "gap_token_idx": gap_idx,
    "n_tokens": n_tok,
    "injection_sites": INJECTION_SITES,
    "dtype_intervention": "Linear forward_pre_hooks upcast exact 0/1 float32 one-hot inputs (atom_mpnn.py:105-107, :331); any other float32 input aborts the run",
    "upcast_log": upcast_log,
    "upstream_commit": _git(args.caliby_root, "rev-parse", "HEAD"),
    "upstream_porcelain": _git(args.caliby_root, "status", "--porcelain"),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "biotite": biotite.__version__,
    "atomworks_note": "venv atomworks is 2.2.1; upstream pins the atomworks-caliby fork (featurization provenance caveat)",
    "keys": sorted(out),
  }
  (args.out_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
  log.info("wrote %s sha256=%s keys=%d", dump, manifest["dump_sha256"], len(out))
  del to


if __name__ == "__main__":
  main()
