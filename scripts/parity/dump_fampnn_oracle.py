"""Sealed upstream oracle for FAMPNN (real ``~/repos/fampnn`` code, real released weights).

Runs the upstream ``SeqDenoiser`` (fampnn/model/sd_model.py) on real fixtures and writes one sealed
``.npz`` per dtype. Nothing from aminx is imported. Randomness is *injected* at the torch RNG
primitives that upstream calls, so every stochastic step is deterministic and recorded:

  torch.rand_like   fampnn/sampling_utils.py:21     get_decoding_order  (residue decoding order)
  torch.multinomial fampnn/model/fampnn_denoiser.py:95  sample_aatype   (per-position categorical draw)
  torch.randn       fampnn/interpolants/edm_interpolant.py:~118 sample_prior (EDM prior, scaled by sigma(0))
  torch.randn_like  fampnn/interpolants/edm_interpolant.py:~139 churn   (Karras stochastic churn; s_churn>0)

Any RNG call that is not served from an injected queue raises, so no randomness is unaccounted for.
``torch.multinomial`` is replaced by an inverse-CDF draw on injected uniforms (the only change of
mechanism); the surrounding upstream code (temperature scaling, X masking) runs unmodified.

Stages (all graded by scripts/parity/fampnn_parity.py):
  parse    upstream load_feats_from_pdb + process_single_pdb
  enc      seq_design_module logits / h_V / h_ESV / E_idx / atom14 X under 5 mask states
  scn1     one SidechainMLP call at fixed (x_t, t), with and without self-conditioning
  edm      the full EDM loop (sidechain_diffusion) with injected prior, incl. per-step x_in / x_out
  sample   SeqDenoiser.sample (MAR loop + repack_last + psce_threshold + overrides) under 4 configs

f64: ``--dtype f64`` sets torch's default dtype to float64 after import, casts the model and the
module-level float32 constants, and routes ``Tensor.float()`` to float64 (upstream hard-codes
``.float()`` at fampnn/model/fampnn.py:360 and elsewhere). Values are unchanged; only precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

LOG = logging.getLogger("dump_fampnn_oracle")

CKPTS = ("fampnn_0_3", "fampnn_0_3_cath", "fampnn_0_0")
SAMPLER_CKPTS = ("fampnn_0_3", "fampnn_0_3_cath")
FIXTURES = ("1ubq", "1ubq_AB")

# (name, aatype_visible fn, scn_visible fn, zero_masked_coords)
STATES = ("all_masked", "half", "seqonly", "partial", "full")


def _sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
  """Canonical digest over (key, dtype, shape, bytes), independent of zip timestamps."""
  h = hashlib.sha256()
  for k in sorted(arrays):
    a = np.ascontiguousarray(arrays[k])
    h.update(k.encode())
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
  return h.hexdigest()


class _Queues:
  """Injected-RNG queues. Every torch RNG primitive upstream uses is served from here."""

  def __init__(self, torch, dtype, rng: np.random.Generator):
    self.torch = torch
    self.dtype = dtype
    self.rng = rng
    self.log: dict[str, list[np.ndarray]] = {"rand_like": [], "multinomial_u": [], "randn": [], "randn_like": []}
    self.orig = {
      "rand_like": torch.rand_like,
      "multinomial": torch.multinomial,
      "randn": torch.randn,
      "randn_like": torch.randn_like,
      "rand": torch.rand,
      "randint": torch.randint,
      "randperm": torch.randperm,
      "normal": torch.normal,
      "bernoulli": torch.bernoulli,
    }
    self.preset: dict[str, list[np.ndarray]] = {}

  def install(self):
    t = self.torch

    def rand_like(x, *a, **k):
      u = self._draw("rand_like", x.shape)
      return t.as_tensor(u, dtype=x.dtype, device=x.device)

    def randn(*shape, **k):
      if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
      z = self._draw("randn", tuple(shape), normal=True)
      return t.as_tensor(z, dtype=self.dtype, device=k.get("device"))

    def randn_like(x, *a, **k):
      z = self._draw("randn_like", x.shape, normal=True)
      return t.as_tensor(z, dtype=x.dtype, device=x.device)

    def multinomial(p, num_samples=1, replacement=False, **k):
      assert num_samples == 1, "oracle injects exactly one categorical draw per row"
      u = self._draw("multinomial_u", (p.shape[0],))
      cdf = np.cumsum(p.detach().cpu().double().numpy(), axis=-1)
      cdf = cdf / cdf[:, -1:]
      idx = (cdf < u[:, None]).sum(-1)
      idx = np.minimum(idx, p.shape[-1] - 1)
      return t.as_tensor(idx, dtype=t.long, device=p.device).unsqueeze(-1)

    def forbidden(name):
      def _f(*a, **k):
        raise RuntimeError(f"unaccounted RNG primitive torch.{name} called during oracle run")

      return _f

    t.rand_like, t.randn, t.randn_like, t.multinomial = rand_like, randn, randn_like, multinomial
    for n in ("rand", "randint", "randperm", "normal", "bernoulli"):
      setattr(t, n, forbidden(n))

  def uninstall(self):
    t = self.torch
    t.rand_like, t.multinomial = self.orig["rand_like"], self.orig["multinomial"]
    t.randn, t.randn_like = self.orig["randn"], self.orig["randn_like"]
    for n in ("rand", "randint", "randperm", "normal", "bernoulli"):
      setattr(t, n, self.orig[n])

  def _draw(self, kind: str, shape: tuple, normal: bool = False) -> np.ndarray:
    pre = self.preset.get(kind)
    if pre:
      arr = pre.pop(0)
      assert tuple(arr.shape) == tuple(shape), f"{kind}: injected {arr.shape} != requested {shape}"
    else:
      arr = self.rng.standard_normal(shape) if normal else self.rng.random(shape)
    arr = np.asarray(arr, dtype=np.float64)
    # round-trip through the run dtype so the stored array IS the injected tensor
    arr = arr.astype(np.float32 if self.dtype == self.torch.float32 else np.float64)
    self.log[kind].append(arr.copy())
    return arr.astype(np.float64)

  def drain_log(self) -> dict[str, list[np.ndarray]]:
    out = {k: list(v) for k, v in self.log.items()}
    for v in self.log.values():
      v.clear()
    return out


def _np(x) -> np.ndarray:
  return x.detach().cpu().numpy()


def _stage(out: dict, prefix: str, **kw):
  for k, v in kw.items():
    out[f"{prefix}/{k}"] = _np(v) if hasattr(v, "detach") else np.asarray(v)


def build(args) -> int:  # noqa: PLR0915
  import torch

  fam_root = Path(args.fampnn_root).resolve()
  sys.path.insert(0, str(fam_root))
  # The oracle must run upstream code only: this script never imports aminx; fail if anything did.
  assert "aminx" not in sys.modules, "aminx must not be loaded in the oracle process"

  from fampnn import sampling_utils
  from fampnn.data import life
  from fampnn.data import residue_constants as rc
  from fampnn.data.data import get_rc_tensor, load_feats_from_pdb, process_single_pdb
  from fampnn.model.sd_model import SeqDenoiser

  dt = torch.float64 if args.dtype == "f64" else torch.float32
  if dt == torch.float64:
    torch.set_default_dtype(torch.float64)
    _orig_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self.double()  # type: ignore[method-assign]
    life.RESTYPE_REF_POS_ATOM37 = life.RESTYPE_REF_POS_ATOM37.double()
  torch.set_grad_enabled(False)
  torch.use_deterministic_algorithms(True)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # fixtures --------------------------------------------------------------
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from fampnn_fixtures import make_split_ab_pdb

  pdb_1ubq = Path(args.pdb).resolve()
  pdb_ab = out_dir / "1ubq_split_AB.pdb"
  ab_sha = make_split_ab_pdb(pdb_1ubq, pdb_ab)
  pdbs = {"1ubq": pdb_1ubq, "1ubq_AB": pdb_ab}

  arrays: dict[str, np.ndarray] = {}
  meta: dict = {
    "dtype": args.dtype,
    "torch": torch.__version__,
    "fampnn_commit": args.fampnn_commit,
    "pdb_sha256": {k: _sha256_file(v) for k, v in pdbs.items()},
    "ab_sha256_written": ab_sha,
    "weights_sha256": {},
    "injection_points": {
      "rand_like": "fampnn/sampling_utils.py:21 (get_decoding_order)",
      "multinomial": "fampnn/model/fampnn_denoiser.py:95 (sample_aatype); replaced by inverse-CDF on injected uniforms",
      "randn": "fampnn/interpolants/edm_interpolant.py:~118 (sample_prior)",
      "randn_like": "fampnn/interpolants/edm_interpolant.py:~139 (churn)",
    },
  }

  rng_master = np.random.default_rng(20261104)
  for ckpt_name in CKPTS:
    wpath = fam_root / "weights" / f"{ckpt_name}.pt"
    meta["weights_sha256"][ckpt_name] = _sha256_file(wpath)
    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    model = SeqDenoiser(ck["model_cfg"]).eval()
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(dt)
    den = model.denoiser
    scn_mod = den.scn_diffusion_module
    interp = scn_mod.scn_interpolant
    meta.setdefault("ckpt_cfg", {})[ckpt_name] = {
      "n_layers": int(den.seq_design_module.num_encoder_layers),
      "k_neighbors": int(den.seq_design_module.k_neighbors),
      "rho": float(interp.rho),
      "s_min": float(interp.s_min),
      "s_max": float(interp.s_max),
      "sigma_data": float(interp.sigma_data),
      "use_self_conditioning": bool(scn_mod.use_self_conditioning),
      "scn_mean": float(model.scn_mean),
      "scn_std": float(model.scn_std),
    }

    for fx in FIXTURES:
      data = load_feats_from_pdb(str(pdbs[fx]))
      single = process_single_pdb(data)
      L = int(single["x"].shape[0])
      P = f"{ckpt_name}/{fx}"
      b = {k: single[k].unsqueeze(0) for k in ("x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index")}
      x = b["x"].to(dt)
      sm = b["seq_mask"].to(dt)
      mam = b["missing_atom_mask"].to(dt)
      aat = b["aatype"]
      ridx, cidx = b["residue_index"], b["chain_index"]
      _stage(
        arrays,
        f"{P}/parse",
        x=x[0],
        aatype=aat[0],
        seq_mask=sm[0],
        missing_atom_mask=mam[0],
        residue_index=ridx[0],
        chain_index=cidx[0],
        atom_mask=single["atom_mask"],
        interface_residue_mask=single["interface_residue_mask"],
      )
      meta.setdefault("chain_id_mapping", {})[fx] = {k: int(v) for k, v in data["chain_id_mapping"].items()}

      # ---- enc: 5 mask states ------------------------------------------------
      idx = torch.arange(L)
      vis_aa = {
        "all_masked": torch.zeros(L),
        "half": (idx % 2 == 0).to(dt),
        "seqonly": torch.ones(L),
        "partial": (idx % 2 == 0).to(dt),
        "full": torch.ones(L),
      }
      vis_scn = {
        "all_masked": torch.zeros(L),
        "half": (idx % 2 == 0).to(dt),
        "seqonly": torch.zeros(L),
        "partial": (idx % 4 == 0).to(dt),
        "full": torch.ones(L),
      }
      captured: dict = {}

      def _hook(_m, _inp, outp, captured=captured):
        captured["logits"], captured["feat"] = outp

      hh = den.seq_design_module.register_forward_hook(_hook)
      state_h_V = {}
      for st in STATES:
        va, vs = vis_aa[st][None].to(dt), vis_scn[st][None].to(dt)
        aat_noised = torch.where(va.bool(), aat, torch.full_like(aat, rc.restype_order_with_x["X"]))
        x_in = x.clone()
        if st in ("all_masked", "partial"):
          x_in[:, :, rc.non_bb_idxs, :] = x_in[:, :, rc.non_bb_idxs, :] * vs[:, :, None, None]
        logp = model.score(x_in, aat_noised, sm, mam, vs, ridx, cidx)
        feat = captured["feat"]
        _stage(
          arrays,
          f"{P}/enc/{st}",
          aatype_noised=aat_noised[0],
          scn_vis=vs[0],
          x_in=x_in[0],
          logits=captured["logits"][0],
          log_probs=logp[0],
          h_V=feat["h_V"][0],
          X14=feat["X"][0],
          atom14_mask=feat["atom14_mask"][0],
          E_idx=feat["E_idx"][0],
        )
        if ckpt_name == "fampnn_0_3" and st in ("half", "partial"):
          _stage(arrays, f"{P}/enc/{st}", h_ESV=feat["h_ESV"][0])
        state_h_V[st] = {k: feat[k] for k in ("h_V", "h_ESV", "X", "atom14_mask", "E_idx", "S")}
      hh.remove()

      # ---- scn1: one SidechainMLP call ---------------------------------------
      hv = state_h_V["half"]["h_V"]
      g = np.random.default_rng(rng_master.integers(1 << 31))
      x_scn = torch.as_tensor(g.standard_normal((1, L, 33, 3)) * 2.0, dtype=dt)
      x_sc = torch.as_tensor(g.standard_normal((1, L, 33, 3)) * 0.5, dtype=dt)
      t_val = torch.tensor([0.37], dtype=dt)
      y_sc, _ = scn_mod.scn_denoiser(x_scn, aat, t_val, hv, sm, x_scn_self_cond=x_sc)
      y_no, _ = scn_mod.scn_denoiser(x_scn, aat, t_val, hv, sm, x_scn_self_cond=None)
      tt = torch.linspace(0, 1, 11, dtype=dt)
      _stage(
        arrays,
        f"{P}/scn1",
        x_scn=x_scn[0],
        x_self_cond=x_sc[0],
        t=t_val,
        y_selfcond=y_sc[0],
        y_noselfcond=y_no[0],
        sigma_grid_t=tt,
        sigma_grid=interp.sigma(tt),
      )

      if ckpt_name not in SAMPLER_CKPTS or args.skip_sampler:
        continue

      # ---- edm: full loop with injected prior ----------------------------------
      for tag, nsteps, step_scale, s_churn in (("s50", 50, 1.5, 0.0), ("s12_sc1", 12, 1.0, 0.0), ("s12_churn", 12, 1.5, 30.0)):
        q = _Queues(torch, dt, np.random.default_rng(rng_master.integers(1 << 31)))
        q.install()
        traj_in, traj_out = [], []

        def _rec(_m, _args, _kwargs, outp, traj_in=traj_in, traj_out=traj_out):
          traj_in.append(_np(_args[0])[0].copy())
          traj_out.append(_np(outp[0])[0].copy())

        hk = scn_mod.scn_denoiser.register_forward_hook(_rec, with_kwargs=True)
        try:
          ts = sampling_utils.get_timesteps_from_schedule("linear", nsteps, 0.0, 1.0)[None].to(dt)
          churn_cfg = {"s_churn": s_churn, "s_noise": 1.0, "s_t_min": 0.01, "s_t_max": 50.0, "num_steps": nsteps}
          scd = {
            "num_steps": nsteps,
            "timesteps": ts,
            "churn_cfg": churn_cfg,
            "step_scale": step_scale,
            "aatype_override_mask": torch.zeros_like(aat),
            "aatype_override": aat,
          }
          x1, aux = scn_mod.sidechain_diffusion(
            state_h_V["half"], aat, seq_mask=sm, residue_index=ridx, chain_index=cidx, aux_inputs={"scd": scd}, is_sampling=True
          )
        finally:
          hk.remove()
          q.uninstall()
        lg = q.drain_log()
        _stage(
          arrays,
          f"{P}/edm/{tag}",
          timesteps=ts[0],
          x_scn_global=x1[0],
          psce=aux["psce"][0],
          prior_z=lg["randn"][0],
          traj_in=np.stack(traj_in),
          traj_out=np.stack(traj_out),
        )
        if lg["randn_like"]:
          _stage(arrays, f"{P}/edm/{tag}", churn_z=np.stack(lg["randn_like"]))
        meta.setdefault("edm_cfg", {})[tag] = {"num_steps": nsteps, "step_scale": step_scale, "s_churn": s_churn}

      # ---- sample: full MAR loop under injected RNG ------------------------------
      import pandas as pd

      fixed_seq = "A1-20,A31-35" if fx == "1ubq" else "A1-20,B45-50"
      fixed_df = pd.DataFrame({"fixed_pos_seq": [fixed_seq], "fixed_pos_scn": ["A1-20"]}, index=[fx])
      meta.setdefault("fixed_pos", {})[fx] = {"fixed_pos_seq": fixed_seq, "fixed_pos_scn": "A1-20"}
      sample_cfgs = {
        "A_default": dict(S=4, scn_steps=50, step_scale=1.5, s_churn=0.0, T=0.1, seq_only=False, repack_last=True, psce=0.3, fixed=False),
        "B_fixed": dict(S=4, scn_steps=50, step_scale=1.5, s_churn=0.0, T=0.1, seq_only=False, repack_last=True, psce=0.3, fixed=True),
        "C_seqonly_argmax": dict(S=3, scn_steps=50, step_scale=1.5, s_churn=0.0, T=0.0, seq_only=True, repack_last=False, psce=None, fixed=False),
        "D_churn_hot": dict(S=3, scn_steps=12, step_scale=1.0, s_churn=20.0, T=1.0, seq_only=False, repack_last=True, psce=None, fixed=False),
      }
      for tag, c in sample_cfgs.items():
        q = _Queues(torch, dt, np.random.default_rng(rng_master.integers(1 << 31)))
        q.install()
        try:
          if c["fixed"]:
            chain_map = {fx: data["chain_id_mapping"]}
            batch_for_masks = {"residue_index": ridx, "chain_index": cidx, "aatype": aat}
            ao, so = sampling_utils.get_override_masks(batch_for_masks, [fx], [data["chain_id_mapping"]], fixed_df, mode="seq_design")
            del chain_map
          else:
            ao, so = None, None
          timesteps = sampling_utils.get_timesteps_from_schedule("linear", c["S"], 0.0, 1.0)[None].to(dt)
          t_scd = sampling_utils.get_timesteps_from_schedule("linear", c["scn_steps"], 0.0, 1.0)[None].to(dt)
          scd_inputs = {
            "num_steps": c["scn_steps"],
            "timesteps": t_scd,
            "step_scale": c["step_scale"],
            "churn_cfg": {"s_churn": c["s_churn"], "s_noise": 1.0, "s_t_min": 0.01, "s_t_max": 50.0, "num_steps": c["scn_steps"]},
          }
          xs, aas, saux = model.sample(
            x,
            aatype=aat,
            seq_mask=sm,
            missing_atom_mask=mam,
            residue_index=ridx,
            chain_index=cidx,
            timesteps=timesteps,
            seq_only=c["seq_only"],
            temperature=c["T"],
            repack_last=c["repack_last"],
            psce_threshold=c["psce"],
            aatype_override_mask=ao,
            scn_override_mask=so,
            scd_inputs=scd_inputs,
          )
        finally:
          q.uninstall()
        lg = q.drain_log()
        _stage(
          arrays,
          f"{P}/sample/{tag}",
          timesteps=timesteps[0],
          x_final=xs[0],
          aatype_final=aas[0],
          psce=saux["psce"][0],
          seq_probs=saux["seq_probs"][0],
          xt_traj=saux["xt_traj"][0],
          aatype_t_traj=saux["aatype_t_traj"][0],
          aatype_pred_traj=saux["aatype_pred_traj"][0],
          psce_t_traj=saux["psce_t_traj"][0],
          seq_logits_traj=saux["seq_logits_traj"][0],
          inj_decoding_u=lg["rand_like"][0],
          inj_prior_z=np.stack(lg["randn"]),
        )
        if lg["multinomial_u"]:  # temperature == 0 takes the argmax branch: no categorical draw
          _stage(arrays, f"{P}/sample/{tag}", inj_multinomial_u=np.stack(lg["multinomial_u"]))
        if ao is not None:
          _stage(arrays, f"{P}/sample/{tag}", aatype_override_mask=ao[0], scn_override_mask=so[0])
        if lg["randn_like"]:
          _stage(arrays, f"{P}/sample/{tag}", inj_churn_z=np.stack(lg["randn_like"]))
        meta.setdefault("sample_cfg", {})[tag] = {k: (None if v is None else v) for k, v in c.items()}
        LOG.info("sample %s %s %s done (rng calls: %s)", ckpt_name, fx, tag, {k: len(v) for k, v in lg.items()})
      LOG.info("done %s %s", ckpt_name, fx)

  assert "aminx" not in sys.modules, "aminx got loaded during the oracle run"
  out = out_dir / f"fampnn_oracle_{args.dtype}.npz"
  np.savez(out, **arrays)
  manifest = {
    "file": out.name,
    "file_sha256": _sha256_file(out),
    "content_sha256": content_sha256(arrays),
    "n_arrays": len(arrays),
    "meta": meta,
  }
  (out_dir / f"fampnn_oracle_{args.dtype}.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
  LOG.info("sealed %s file_sha256=%s content_sha256=%s", out, manifest["file_sha256"], manifest["content_sha256"])
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--fampnn-root", required=True, help="path to upstream fampnn checkout")
  ap.add_argument("--fampnn-commit", default="aaf788b")
  ap.add_argument("--pdb", required=True, help="tests/data/1ubq.pdb")
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--dtype", choices=("f32", "f64"), default="f32")
  ap.add_argument("--skip-sampler", action="store_true")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
  return build(args)


if __name__ == "__main__":
  raise SystemExit(main())
