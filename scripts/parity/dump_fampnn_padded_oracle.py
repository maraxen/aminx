"""Upstream FAMPNN encoder on a padded input (``seq_mask = 0`` pad residues with junk coordinates), float32.

Runs the real ``SeqDenoiser.score`` of ``~/repos/fampnn`` (released ``fampnn_0_3`` weights) on 1ubq, once unpadded and
once with ``PAD`` trailing pad residues whose coordinates are random, residue/chain index 0 and aatype X, in two mask
states (all masked, half revealed). Records the real-residue logits and node states of both runs and the padded run's
inputs, so a test can (a) check upstream itself is pad-invariant on the real residues and (b) grade aminx on the same
padded input. Writes ``tests/data/fampnn_padded_oracle.json.gz`` (gzipped JSON, not an LFS-tracked type). Nothing from aminx is imported.
Run on titanix in the oracle environment (``PYTHONPATH=<P4_ROOT>/stubs:<P4_ROOT>/extra``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PAD = 11


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--fampnn-root", required=True)
  ap.add_argument("--pdb", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  root = Path(args.fampnn_root).resolve()
  sys.path.insert(0, str(root))
  assert "aminx" not in sys.modules
  import torch

  from fampnn.data import residue_constants as rc
  from fampnn.data.data import load_feats_from_pdb, process_single_pdb
  from fampnn.model.sd_model import SeqDenoiser

  torch.set_grad_enabled(False)
  ck = torch.load(root / "weights" / "fampnn_0_3.pt", map_location="cpu", weights_only=False)
  model = SeqDenoiser(ck["model_cfg"]).eval()
  model.load_state_dict(ck["state_dict"], strict=True)

  data = load_feats_from_pdb(args.pdb)
  single = process_single_pdb(data)
  n = int(single["x"].shape[0])
  rng = np.random.default_rng(7)

  def pad(t: torch.Tensor, fill) -> torch.Tensor:  # noqa: ANN001
    extra = torch.full((PAD, *t.shape[1:]), fill, dtype=t.dtype) if np.isscalar(fill) else fill
    return torch.cat([t, extra], dim=0)

  junk_x = torch.tensor(rng.normal(size=(PAD, 37, 3)) * 3.0, dtype=torch.float32)
  feats = {
    "x": single["x"].float(),
    "aatype": single["aatype"],
    "seq_mask": single["seq_mask"].float(),
    "missing_atom_mask": data["missing_atom_mask"].float(),
    "residue_index": single["residue_index"],
    "chain_index": single["chain_index"],
  }
  padded = {
    "x": pad(feats["x"], junk_x),
    "aatype": pad(feats["aatype"], rc.restype_order_with_x["X"]),
    "seq_mask": pad(feats["seq_mask"], 0.0),
    "missing_atom_mask": pad(feats["missing_atom_mask"], 0.0),
    "residue_index": pad(feats["residue_index"], 0),
    "chain_index": pad(feats["chain_index"], 0),
  }
  captured: dict = {}

  def hook(_m, _inp, outp):  # noqa: ANN001
    captured["logits"], captured["feat"] = outp
    captured["amask"] = _inp[3]

  model.denoiser.seq_design_module.register_forward_hook(hook)
  out: dict = {"n": n, "pad": PAD, "pdb_sha256": hashlib.sha256(Path(args.pdb).read_bytes()).hexdigest()}
  out["junk_x"] = junk_x.numpy().tolist()
  diffs: dict = {}
  for name, vis_mod in (("all_masked", 0), ("half", 2)):
    res = {}
    for tag, f in (("unpadded", feats), ("padded", padded)):
      length = f["x"].shape[0]
      idx = torch.arange(length)
      real = idx < n
      va = (idx % 2 == 0) & real if vis_mod else torch.zeros(length, dtype=torch.bool)
      aat_noised = torch.where(va, f["aatype"], torch.full_like(f["aatype"], rc.restype_order_with_x["X"]))
      vs = va.float()
      x_in = f["x"].clone()
      x_in[:, rc.non_bb_idxs, :] = x_in[:, rc.non_bb_idxs, :] * vs[:, None, None]
      b = {k: v[None] for k, v in f.items()}
      model.score(x_in[None], aat_noised[None], b["seq_mask"], b["missing_atom_mask"], vs[None], b["residue_index"], b["chain_index"])
      res[tag] = {
        "x_in": x_in.numpy().tolist(),
        "aatype_noised": aat_noised.numpy().tolist(),
        "scn_vis": vs.numpy().tolist(),
        "logits": captured["logits"][0].numpy().tolist(),
        "h_V": captured["feat"]["h_V"][0].numpy().tolist(),
        "E_idx": captured["feat"]["E_idx"][0].numpy().tolist(),
        "atom_mask_noised": captured["amask"][0].numpy().tolist(),
      }
    out[name] = res
    lg_u, lg_p = np.asarray(res["unpadded"]["logits"]), np.asarray(res["padded"]["logits"])[:n]
    hv_u, hv_p = np.asarray(res["unpadded"]["h_V"]), np.asarray(res["padded"]["h_V"])[:n]
    diffs[name] = {"logits": float(np.abs(lg_u - lg_p).max()), "h_V": float(np.abs(hv_u - hv_p).max())}
    print(name, "upstream padded-vs-unpadded:", diffs[name])
  for name in ("all_masked", "half"):
    out[name].pop("unpadded")  # only the numbers above are kept from the unpadded run
  out["upstream_padded_vs_unpadded_real_residues"] = diffs
  out["padded_inputs"] = {k: v.numpy().tolist() for k, v in padded.items()}
  with gzip.open(args.out, "wt") as fh:
    json.dump(out, fh)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
