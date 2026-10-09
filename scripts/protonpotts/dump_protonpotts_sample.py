"""Dump upstream ProtonPottsMPNN (v6) N-row decoder sampling as the engine's ``mpnn_sample`` makes it (P4i).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §55; debt #2617. Sidecar: ``dump_protonpotts_sample.bth.toml``
(committed before this script). Run in the `aminx-oracles-protonpotts` environment; imports upstream `mpnn`, torch and numpy
only, never aminx::

    bth run --project-slug aminx -- <oracle-env python> scripts/protonpotts/dump_protonpotts_sample.py \\
        --checkpoint <ckpt> --out <dir> --cell pkad_unlabelled=<1BVC.pdb>:A --cell multichain=<6m0j.pdb>:E \\
        --cell only_o_1olr=<1OLR.pdb>:A --cell gap=<1EL1.pdb>:A --sealed p4b=<...> --sealed p4c=<...> --sealed p4d=<...>

The engine's call (potts_mpnn_ph.py:1226-1231) is ``ni["input_features"]["repeat_sample_num"] = N`` then
``model(ni)["decoder_features"]["S_sampled"]`` on an input prepared with ``designed_chains=[binder_chain]``. Here
``torch.multinomial`` is replaced by inverse-CDF on INJECTED uniforms: row r, decoding step k draws with ``uniforms[r, k]`` (the P4h
convention, per row). The f64 run makes the same three changes P4h made (see dump_protonpotts_decoder.py). Threads are pinned to 1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_decoder import _f64_changes  # noqa: E402
from dump_protonpotts_encoder import EXTENDED_VOCAB, PRECISIONS, SEED, _sealed_hash, _to_numpy  # noqa: E402
from dump_protonpotts_features import _content_hash  # noqa: E402

logger = logging.getLogger("dump_protonpotts_sample")

N_ROWS = 3
V = 30


class _DrawsN:
  """``torch.multinomial`` on a [N, V] batch replaced by per-row inverse-CDF on injected uniforms [N, L]."""

  def __init__(self, uniforms: np.ndarray) -> None:
    self.uniforms = uniforms
    self.probs: list[torch.Tensor] = []
    self.index: list[list[int]] = []
    self.margin: list[list[float]] = []

  def __call__(self, probs: torch.Tensor, num_samples: int, replacement: bool = False, generator=None):  # noqa: ANN001, ANN204, ARG002, FBT001, FBT002
    n = self.uniforms.shape[0]
    if probs.shape[0] != n or num_samples != 1:
      msg = f"shim expects {n} rows and one sample, got {tuple(probs.shape)} x {num_samples}"
      raise RuntimeError(msg)
    k = len(self.probs)
    indices, margins = [], []
    for r in range(n):
      row = probs[r]
      cdf = torch.cumsum(row, dim=-1)
      target = torch.tensor(float(self.uniforms[r, k]), dtype=row.dtype) * cdf[-1]
      last = int(torch.nonzero(row > 0)[-1])
      indices.append(min(int((cdf <= target).sum()), last))
      margins.append(float((cdf - target).abs().min() / cdf[-1]))
    self.probs.append(probs.detach().clone())
    self.index.append(indices)
    self.margin.append(margins)
    return torch.tensor(indices, dtype=torch.long).unsqueeze(1)


def _prepare(pdb: Path, binder: str | None):  # noqa: ANN202
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference

  pipelines.get_protonation_state_transforms = lambda **_: []
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (spec §23.4)
  torch.manual_seed(SEED)
  torch.set_num_threads(1)
  kwargs = {"designed_chains": [binder]} if binder else {}
  return inference, inference.prepare_potts_input(str(pdb), extended_vocab=EXTENDED_VOCAB, **kwargs)


def _tensors(batch: dict) -> dict[str, np.ndarray]:
  features = batch["network_input"]["input_features"]
  return {k: _to_numpy(v) for k, v in sorted(features.items()) if isinstance(v, torch.Tensor)}


def _forward(model, batch, dtype, uniforms):  # noqa: ANN001, ANN202
  import mpnn.model.mpnn as mm

  network_input = copy.deepcopy(batch["network_input"])
  features = network_input["input_features"]
  if dtype == torch.float64:
    for key, value in list(features.items()):
      if isinstance(value, torch.Tensor) and value.is_floating_point():
        features[key] = value.double()
  features["repeat_sample_num"] = N_ROWS

  captured: dict[str, list] = {"order": [], "wout": []}
  handle = model.W_out.register_forward_hook(lambda _m, _i, out: captured["wout"].append(out.detach().clone()))
  original_setup = mm.ProteinMPNN.setup_causality_masks

  def wrapped(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    out = original_setup(self, *args, **kwargs)
    captured["order"].append(out["decoding_order"].detach().clone())
    return out

  mm.ProteinMPNN.setup_causality_masks = wrapped
  draws = _DrawsN(uniforms)
  original_multinomial = torch.multinomial
  torch.multinomial = draws
  try:
    torch.manual_seed(SEED)
    with torch.no_grad():
      out = model(network_input)
  finally:
    torch.multinomial = original_multinomial
    mm.ProteinMPNN.setup_causality_masks = original_setup
    handle.remove()

  decoder_features = out["decoder_features"]
  order = _to_numpy(captured["order"][0])  # [N, L]
  n, length = order.shape
  log_probs = _to_numpy(decoder_features["log_probs"])  # [N, L, V]
  probs_by_pos = np.zeros((n, length, V), dtype=log_probs.dtype)
  logits_by_pos = np.zeros((n, length, V), dtype=log_probs.dtype)
  drawn = np.zeros((n, length), dtype=np.int64)
  margin = np.zeros((n, length), dtype=np.float64)
  for step in range(length):
    wout = _to_numpy(captured["wout"][step])  # [N, 1, V]
    for r in range(n):
      pos = int(order[r, step])
      probs_by_pos[r, pos] = _to_numpy(draws.probs[step][r])
      logits_by_pos[r, pos] = wout[r, 0]
      drawn[r, pos] = draws.index[step][r]
      margin[r, pos] = draws.margin[step][r]
  return {
    "decoding_order": order, "uniforms": np.asarray(uniforms, dtype=np.float64), "S_sampled": _to_numpy(decoder_features["S_sampled"]),
    "log_probs": log_probs, "probs_sample": probs_by_pos, "logits": logits_by_pos, "drawn_token": drawn, "draw_margin": margin,
  }


def _run(inference, checkpoint: Path, batch: dict, dtype: torch.dtype, uniforms: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ANN001
  import mpnn.model.mpnn as mm

  previous_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    model = inference.load_model(str(checkpoint))
    if dtype == torch.float64:
      model = model.double()
    positional = model.graph_featurization_module.positional_embedding.embed_positional_features
    if dtype == torch.float64:
      with _f64_changes(mm, positional):
        return _forward(model, batch, dtype, uniforms)
    return _forward(model, batch, dtype, uniforms)
  finally:
    torch.set_default_dtype(previous_default)


def _draw_matches(arrays: dict[str, np.ndarray], designed: np.ndarray) -> bool:
  probs, order, uniforms, sampled = arrays["probs_sample"], arrays["decoding_order"], arrays["uniforms"], arrays["S_sampled"]
  for r in range(order.shape[0]):
    for step, pos in enumerate(order[r]):
      if not designed[pos]:
        continue
      row = probs[r, pos]
      cdf = np.cumsum(row)
      target = uniforms[r, step].astype(row.dtype) * cdf[-1]
      if int(sampled[r, pos]) != min(int((cdf <= target).sum()), int(np.nonzero(row > 0)[0][-1])):
        return False
  return True


def _dump(args: argparse.Namespace, out: Path, manifests: dict[str, dict]) -> dict[str, dict]:
  out.mkdir(parents=True, exist_ok=True)
  report: dict[str, dict] = {}
  for cell, (pdb, binder) in args.cells.items():
    _inference, default_batch = _prepare(pdb, None)
    default_arrays = _tensors(default_batch)
    default_hash = _content_hash(default_arrays)
    sealed = _sealed_hash(manifests, cell)
    inference, batch = _prepare(pdb, binder)
    arrays_in = _tensors(batch)
    from atomworks.ml.utils.token import get_token_starts  # noqa: PLC0415

    atom_array = batch["atom_array"]
    chains = np.asarray(atom_array[get_token_starts(atom_array)].chain_id)
    designed = arrays_in["designed_residue_mask"][0].astype(bool)
    only_mask = (
      set(default_arrays) <= set(arrays_in) and set(arrays_in) - set(default_arrays) <= {"designed_residue_mask"}
      and all(np.array_equal(arrays_in[k], default_arrays[k]) for k in default_arrays)
      and bool(np.array_equal(designed, chains == binder)) and bool(designed.any())
      and (len(set(chains.tolist())) == 1 or not bool(designed.all()))  # single chain: the binder is everything
    )
    n = int(arrays_in["S"].shape[-1])
    uniforms = np.random.default_rng(SEED + 200).uniform(size=(N_ROWS, n))
    entry: dict = {
      "feature_content_sha256": default_hash, "sealed_content_sha256": sealed,
      "pipeline_is_sealed": sealed is not None and sealed == default_hash, "only_designed_mask_differs": bool(only_mask),
      "n_residues": n, "native": arrays_in["S"][0].tolist(), "designed": designed.tolist(), "precisions": {},
    }
    for name, dtype in PRECISIONS:
      arrays = _run(inference, args.checkpoint, batch, dtype, uniforms)
      arrays["designed"] = designed
      np.savez(out / f"{cell}_{name}.npz", **arrays)
      entry["precisions"][name] = {
        "content_sha256": _content_hash(arrays),
        "dtypes": sorted({str(a.dtype) for k, a in arrays.items() if a.dtype.kind == "f" and "uniforms" not in k
                          and "draw_margin" not in k}),
        "keys": sorted(arrays),
      }
    report[cell] = entry
  return report


def _child(args: argparse.Namespace) -> dict[str, str]:
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
               "--out", str(Path(tmp) / "child"), "--child-hashes-to", str(result)]
    for name, (pdb, binder) in args.cells.items():
      command += ["--cell", f"{name}={pdb}:{binder}"]
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603
    return json.loads(result.read_text(encoding="utf-8"))


def _flatten(report: dict[str, dict]) -> dict[str, str]:
  return {f"{cell}/{p}": v["content_sha256"] for cell, e in report.items() for p, v in e["precisions"].items()}


def main() -> int:  # noqa: C901
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH:CHAIN")
  parser.add_argument("--sealed", action="append", default=[], metavar="ID=MANIFEST")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {}
  for item in args.cell:
    name, _, rest = item.partition("=")
    path, _, chain = rest.rpartition(":")
    args.cells[name] = (Path(path), chain)
  manifests = {i: json.loads(Path(p).read_text(encoding="utf-8")) for i, _, p in (s.partition("=") for s in args.sealed)}

  directory = args.out / "protonpotts_v6_sample"
  report = _dump(args, directory, manifests)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(_flatten(report)), encoding="utf-8")
    return 0

  reproducible = _child(args) == _flatten(report)
  complete = f64_dtype_ok = precisions_differ = order_same = rows_independent = shim_ok = nondesigned_ok = True
  floor: dict[str, float] = {}
  min_margin: dict[str, float] = {}
  rows_differ_at: dict[str, int] = {}
  keys = ("decoding_order", "uniforms", "S_sampled", "log_probs", "probs_sample", "logits", "drawn_token", "draw_margin", "designed")
  for cell, entry in report.items():
    n = entry["n_residues"]
    native = np.asarray(entry["native"])
    designed = np.asarray(entry["designed"], dtype=bool)
    a32 = dict(np.load(directory / f"{cell}_f32.npz"))
    a64 = dict(np.load(directory / f"{cell}_f64.npz"))
    shapes_ok = (a32["decoding_order"].shape == (N_ROWS, n) and a32["S_sampled"].shape == (N_ROWS, n)
                 and a32["log_probs"].shape == (N_ROWS, n, V) and a32["probs_sample"].shape == (N_ROWS, n, V))
    complete &= shapes_ok and all(k in a32 and k in a64 for k in keys)
    o32, o64 = a32["decoding_order"], a64["decoding_order"]
    order_same &= bool(np.array_equal(o32, o64)) and all(sorted(o32[r].tolist()) == list(range(n)) for r in range(N_ROWS))
    rows_independent &= all(not np.array_equal(o32[i], o32[j]) for i in range(N_ROWS) for j in range(i + 1, N_ROWS))
    shim_ok &= _draw_matches(a32, designed) and _draw_matches(a64, designed)
    for a in (a32, a64):
      nondesigned_ok &= all(bool(np.array_equal(a["S_sampled"][r][~designed], native[~designed])) for r in range(N_ROWS))
    min_margin[cell] = float(a32["draw_margin"][:, designed].min())
    rows_differ_at[cell] = int((a32["S_sampled"] != a32["S_sampled"][0]).any(axis=0).sum())
    for key in ("log_probs", "logits"):
      floor[f"{cell}/{key}"] = float(np.abs(a64[key] - a32[key].astype(np.float64)).max())
    f64_dtype_ok &= all(a64[k].dtype == np.float64 for k in a64 if a64[k].dtype.kind == "f")
    precisions_differ &= float(np.abs(a64["log_probs"] - a32["log_probs"].astype(np.float64)).max()) > 0.0

  flags = {
    "pipeline_is_sealed": all(e["pipeline_is_sealed"] for e in report.values()),
    "only_designed_mask_differs": all(e["only_designed_mask_differs"] for e in report.values()),
    "complete": bool(complete), "f64_is_float64": bool(f64_dtype_ok), "precisions_differ": bool(precisions_differ),
    "order_same_across_precisions": bool(order_same), "rows_independent": bool(rows_independent),
    "shim_is_the_draw": bool(shim_ok), "nondesigned_hold": bool(nondesigned_ok), "reproducible": bool(reproducible),
    "n_cells": len(report),
  }
  (args.out / "sample_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "torch_version": torch.__version__,
       "cells": {n: {"pdb": str(p), "binder": c} for n, (p, c) in args.cells.items()}, "report": report,
       "f32_vs_f64_floor": floor, "min_draw_margin_f32": min_margin, "positions_where_rows_differ": rows_differ_at,
       "flags": flags, "n_rows": N_ROWS, "seed": SEED},
      indent=2, sort_keys=True),
    encoding="utf-8",
  )
  logger.info("flags %s", flags)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8")
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
