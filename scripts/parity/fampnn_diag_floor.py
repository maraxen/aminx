"""DIAGNOSTIC (plain verification run, not a pre-registered finding): where does the FAMPNN f64 loop deviation come from?

Questions answered (task 261104_p4-review-fixes, FAMPNN):
  tables    aminx positional table vs upstream torch float32 table, max abs diff by |offset| (E features, exact)
  oracles   sealed f32 / sealed f64 / f64-table oracle against each other, per sample-stage array (the reference's own floor)
  noise     injected-noise consistency across the three dumps (the shared-draw defect class)
  stages    aminx (table variant) vs oracle (sealed f64 or f64-table), per stage and fixture, fampnn_0_3, f64 arm

aminx table variants (patched at the call sites, jit caches cleared between variants):
  as_is        features.py as committed (correctly rounded float32 exp; float64 cos/sin of float32 angles)
  torch_f32    upstream's own torch float32 table values (from diag_tables.npz), sliced per offset
  f64          float64 frequencies and float64 angles (matches the f64-table oracle)

usage: fampnn_diag_floor.py --sealed-dir D --f64tab-dir E --variant {as_is,torch_f32,f64,none} --out out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import fampnn_parity as fp  # noqa: E402

from aminx.families.fampnn import features, gvp, sidechain_diffusion  # noqa: E402
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint  # noqa: E402
from aminx.families.fampnn.convert import convert_fampnn_state_dict  # noqa: E402

LOG = logging.getLogger("fampnn_diag_floor")
CK = "fampnn_0_3"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
  z = np.load(path)
  return {k: z[k] for k in z.files}


def tables_report(tabs: dict[str, np.ndarray]) -> dict[str, Any]:
  """aminx positional table against upstream torch f32 (and an f64 truth), by |offset|."""
  offs = tabs["pos_offsets"]
  up = tabs["pos_f32_torch"].astype(np.float64)
  truth = tabs["pos_f64"].astype(np.float64)
  num_l = (offs.shape[0] + 1) // 2
  am = features.positional_embedding_table(num_l, 16).astype(np.float64)
  d = np.abs(offs)
  out: dict[str, Any] = {"num_l": int(num_l)}
  bins = [(0, 20), (21, 40), (41, 60), (61, 79), (80, 100), (101, 127)]
  for name, a, b in (("aminx_vs_torch_f32", am, up), ("torch_f32_vs_f64truth", up, truth), ("aminx_vs_f64truth", am, truth)):
    diff = np.abs(a - b)
    out[name] = {
      "max": float(diff.max()),
      "by_abs_offset": {f"{lo}-{hi}": float(diff[(d >= lo) & (d <= hi)].max()) for lo, hi in bins},
      "n_nonzero": int((diff > 0).sum()),
      "n": int(diff.size),
    }
  freq_t = tabs["ts_freq_f32_torch"].astype(np.float64)
  freq_a = features.timestep_frequency_table(128).astype(np.float64)
  out["timestep_freq_aminx_vs_torch_f32"] = {
    "max_abs": float(np.max(np.abs(freq_a - freq_t))),
    "max_rel": float(np.max(np.abs(freq_a - freq_t) / np.abs(freq_t))),
    "n_nonzero": int((freq_a != freq_t).sum()),
    "n": 128,
  }
  return out


def oracles_report(o32: dict, o64: dict, otab: dict) -> dict[str, Any]:
  """Per-array max abs difference between oracle dumps (sample/edm continuous arrays of fampnn_0_3)."""
  res: dict[str, Any] = {}
  keys = [k for k in o64 if k.startswith(f"{CK}/") and o64[k].dtype.kind == "f" and ("/sample/" in k or "/edm/" in k) and not k.split("/")[-1].startswith("inj_")]
  for fx in fp.FIXTURES:
    for pair, (a, b) in {"f32_vs_f64": (o32, o64), "f64tab_vs_f64": (otab, o64), "f64tab_vs_f32": (otab, o32)}.items():
      w: dict[str, float] = {}
      for k in keys:
        if f"/{fx}/" not in k or k not in a or k not in b:
          continue
        name = k.split(f"/{fx}/")[1]
        w[name] = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))))
      res[f"{fx}/{pair}"] = w
  return res


def noise_report(o32: dict, o64: dict, otab: dict) -> dict[str, Any]:
  """Injected-noise consistency: f32 dump should equal float32(f64 dump) (same draws, rounded); f64tab == f64 exactly."""
  out: dict[str, Any] = {}
  inj = [k for k in o64 if "/inj_" in k or k.endswith(("prior_z", "churn_z", "decoding_u", "multinomial_u"))]
  w32 = w64tab = 0.0
  n_bad32 = n_badtab = 0
  worst32_key = ""
  for k in inj:
    a64 = o64[k].astype(np.float64)
    if k in o32:
      d = float(np.max(np.abs(o32[k].astype(np.float64) - a64.astype(np.float32).astype(np.float64)))) if a64.size else 0.0
      if d > 0:
        n_bad32 += 1
        if d > w32:
          w32, worst32_key = d, k
    if k in otab:
      d2 = float(np.max(np.abs(otab[k].astype(np.float64) - a64))) if a64.size else 0.0
      w64tab = max(w64tab, d2)
      n_badtab += int(d2 > 0)
  out.update(n_inj_keys=len(inj), f32_vs_cast_f64_max=w32, f32_vs_cast_f64_n_differ=n_bad32, worst_key=worst32_key, f64tab_vs_f64_max=w64tab, f64tab_vs_f64_n_differ=n_badtab)
  return out


def install_variant(variant: str, tabs: dict[str, np.ndarray]) -> None:
  """Patch the table providers where aminx calls them; ``as_is`` restores the committed ones."""
  if variant == "as_is":
    return
  half_ts = 128
  if variant == "torch_f32":
    pos = tabs["pos_f32_torch"]
    ts = tabs["ts_freq_f32_torch"]
  elif variant == "f64":
    pos = tabs["pos_f64"]
    e = 16
    ts = np.exp(-math.log(10000.0) * np.arange(0, half_ts, dtype=np.float64) / half_ts)
    _ = e
  else:
    raise KeyError(variant)
  num_l_tab = (pos.shape[0] + 1) // 2

  def pos_table(num_res: int, num_embeddings: int = 16) -> np.ndarray:
    assert num_embeddings == 16
    c = num_l_tab - 1
    return np.asarray(pos[c - (num_res - 1) : c + num_res])

  gvp.positional_embedding_table = pos_table
  sidechain_diffusion.timestep_frequency_table = lambda half, max_period=10000.0: np.asarray(ts)[:half]  # noqa: ARG005


def stage_report(o: fp.Oracle, model: Any, tags_edm: tuple[str, ...], tags_sample: tuple[str, ...], only_sample: bool = False) -> list[dict[str, Any]]:  # noqa: ANN401
  led = fp.Ledger({})
  pdbs_unused = None
  _ = pdbs_unused
  for fx in fp.FIXTURES:
    if not only_sample:
      fp.stage_enc(led, o, model, CK, fx)
      fp.stage_scn1(led, o, model, CK, fx)
      fp.stage_edm(led, o, model, CK, fx, tags=tags_edm)
    fp.stage_sample(led, o, model, CK, fx, tags=tags_sample)
  return led.rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """Worst dev by (fixture, stage, array-name), continuous only."""
  w: dict[str, float] = {}
  for r in rows:
    if r["kind"] != "cont":
      continue
    parts = r["key"].split("/")
    fx = next((p for p in parts if p in fp.FIXTURES), "?")
    name = "/".join(parts[1:]) if fx in parts else f"?/{r['stage']}/{parts[-1]}"
    w[name] = max(w.get(name, 0.0), r["dev"])
  return w


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--sealed-dir", required=True)
  ap.add_argument("--f64tab-dir", required=True)
  ap.add_argument("--variant", choices=("as_is", "torch_f32", "f64", "none"), default="none")
  ap.add_argument("--fampnn-root", default="/home/solab/repos/fampnn")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--quick", action="store_true", help="only s50 edm and the D_churn_hot sample config")
  ap.add_argument("--only-sample", action="store_true", help="skip enc/scn1/edm (already shown to agree to ~1e-13 with matching tables)")
  ap.add_argument("--oracles", default="sealed_f64,f64tab")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
  sealed, tabdir = Path(args.sealed_dir), Path(args.f64tab_dir)
  tabs = _load_npz(tabdir / "f64" / "diag_tables.npz")
  res: dict[str, Any] = {"variant": args.variant}

  if args.variant == "none":
    o32 = _load_npz(sealed / "f32" / "fampnn_oracle_f32.npz")
    o64 = _load_npz(sealed / "f64" / "fampnn_oracle_f64.npz")
    otab = _load_npz(tabdir / "f64" / "fampnn_oracle_f64.npz")
    res["tables"] = tables_report(tabs)
    res["oracles"] = oracles_report(o32, o64, otab)
    res["noise"] = noise_report(o32, o64, otab)
  else:
    raw = load_restricted_torch_checkpoint(Path(args.fampnn_root) / "weights" / f"{CK}.pt")
    model = fp._cast(convert_fampnn_state_dict(raw), jax.numpy.float64)  # noqa: SLF001
    install_variant(args.variant, tabs)
    tags_e = ("s50",) if args.quick else fp.EDM_TAGS
    tags_s = ("D_churn_hot",) if args.quick else fp.SAMPLE_TAGS
    for name, d in (("sealed_f64", sealed), ("f64tab", tabdir)):
      if name not in args.oracles.split(","):
        continue
      jax.clear_caches()
      o = fp.Oracle(d, "f64")
      LOG.info("variant=%s oracle=%s", args.variant, name)
      rows = stage_report(o, model, tags_e, tags_s, args.only_sample)
      res[f"vs_{name}"] = summarise(rows)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(res, indent=1, sort_keys=True))
  LOG.info("wrote %s", args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
