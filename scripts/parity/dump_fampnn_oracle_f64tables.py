"""DIAGNOSTIC oracle variant (not a sealed dump): upstream FAMPNN with its two float32 trig tables in float64.

Upstream builds two sinusoidal tables in float32 whatever the model dtype:
  fampnn/data/data.py:427-442        positional_embeddings (edge positional features)
  fampnn/model/sidechain_diffusion/dit_utils.py:266-269  TimestepEmbedder.timestep_embedding (frequencies)
This wrapper replaces both by float64 versions (same formulas, float64 arange/exp/angles/cos/sin) and then
runs ``dump_fampnn_oracle.build`` unchanged, so the injected noise (same seed, same draw order), weights and
fixtures are identical to the sealed f64 dump; only the table precision differs. It also writes
``diag_tables.npz`` with the *torch float32* tables (the real upstream values) for offsets -(127)..127 and the
timestep frequencies, so aminx can be fed upstream's exact table values in the diagnostic.

Output goes to ``--out-dir`` (never the sealed oracle directory). Run on titanix via launch_fampnn_oracle.sh-style env.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOG = logging.getLogger("dump_fampnn_oracle_f64tables")
L_TAB = 128  # offsets -(L_TAB-1)..(L_TAB-1) covers every fixture (3e1r has ~100 residues)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--fampnn-root", required=True)
  ap.add_argument("--pdb", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--fampnn-commit", default="aaf788b")
  ap.add_argument("--skip-sampler", action="store_true")
  args = ap.parse_args()
  args.dtype = "f64"
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

  import torch

  sys.path.insert(0, str(Path(args.fampnn_root).resolve()))
  import fampnn.data.data as fdata
  import fampnn.model.gvp.gvp_modules as gvp_modules
  from fampnn.model.sidechain_diffusion import dit_utils

  # --- real upstream float32 tables, captured before patching ---------------------------------------
  offs = torch.arange(-(L_TAB - 1), L_TAB)
  pos_f32 = fdata.positional_embeddings(torch.stack([offs, torch.zeros_like(offs)]), num_positional_embeddings=16)
  half = 128  # frequency_embedding_size 256 (dit_utils.py:245)
  ts_freq_f32 = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  def pos64(edge_index, num_embeddings=None, num_positional_embeddings=16, period_range=(2, 1000)):  # noqa: ARG001
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float64, device=edge_index.device) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1).to(torch.float64) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)

  def ts64(t, dim, max_period=10000):
    h = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=h, dtype=torch.float64) / h).to(device=t.device)
    a = t[:, None].double() * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
      emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

  fdata.positional_embeddings = pos64
  gvp_modules.positional_embeddings = pos64
  dit_utils.TimestepEmbedder.timestep_embedding = staticmethod(ts64)

  np.savez(
    out_dir / "diag_tables.npz",
    pos_f32_torch=pos_f32.numpy(),
    pos_offsets=offs.numpy(),
    ts_freq_f32_torch=ts_freq_f32.numpy(),
    pos_f64=pos64(torch.stack([offs, torch.zeros_like(offs)])).numpy(),
  )

  import dump_fampnn_oracle as dump

  LOG.info("tables patched to float64; running the unchanged oracle build")
  return dump.build(args)


if __name__ == "__main__":
  raise SystemExit(main())
