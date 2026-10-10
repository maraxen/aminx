"""Dump upstream ProtonPottsMPNN (v6) ENCODER activations and head output, in f32 and f64 (P4e).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §41. Run in the `aminx-oracles-protonpotts`
environment; imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_encoder.py --checkpoint <ckpt> --out <dir> \\
        --cell pkad_unlabelled=<1BVC.pdb> --cell multichain=<6m0j.pdb> \\
        --cell only_o_1olr=<1OLR.pdb> --cell gap=<1EL1.pdb> \\
        --sealed p4b=<features_manifest.json> --sealed p4c=<features_p4c_manifest.json> \\
        --sealed p4d=<features_p4c_manifest.json of the P4d run>

WHY. The P4 oracle holds only the post-merge `etab_out`, so the aminx encoder and head cannot be compared
with it layer by layer, and it exists in f32 only, so there is nothing to separate an aminx defect from
float32 noise. This dumps, per cell and per precision: the layer-0 input and every encoder layer's output
(so a mismatch is localised to a layer), the final `h_V`/`h_E`, `E_idx`, and the head's table BEFORE the
reciprocal merge (what aminx's `PottsHead` computes) and AFTER it (upstream's `etab_out`).

The f64 run is genuine double precision: the model is cast with `.double()` and every floating input
feature is cast too. A control checks it (the two precisions must differ, and the f64 arrays must be
float64), because a silent f32 run labelled f64 would make every later "exact" claim vacuous.

The features each run consumes must be the SEALED ones: their content hash is compared with the P4b/P4c/
P4d manifests, so the aminx wave can feed the sealed feature dumps to its own model knowing upstream saw
the same inputs.

Threads are pinned to 1 (spec §32).
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
from dump_protonpotts_features import _content_hash  # noqa: E402

logger = logging.getLogger("dump_protonpotts_encoder")

EXTENDED_VOCAB = "v6"
SEED = 0
PRECISIONS = (("f32", torch.float32), ("f64", torch.float64))
# cell name -> (manifest id, how to read its sealed content hash)
SEALED_LOOKUP = {
  "pkad_unlabelled": ("p4b", "flat"),
  "multichain": ("p4b", "flat"),
  "gap": ("p4c", "results"),
  "only_o_1olr": ("p4d", "results"),
}
ENCODER_LAYERS = 3


class _StopAfterHead(Exception):  # noqa: N818 -- control flow, not an error
  """Raised once the head output is captured: the decoder is not needed and not dtype-polymorphic."""


def _sealed_hash(manifests: dict[str, dict], cell: str) -> str | None:
  which, mode = SEALED_LOOKUP[cell]
  manifest = manifests.get(which)
  if manifest is None:
    return None
  if mode == "flat":
    return manifest["content_sha256"].get(cell)
  return manifest["results"].get(cell, {}).get("content_sha256")


def _prepare(pdb: Path):  # noqa: ANN202
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference

  pipelines.get_protonation_state_transforms = lambda **_: []
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (spec §23.4)
  torch.manual_seed(SEED)
  torch.set_num_threads(1)
  return inference, inference.prepare_potts_input(str(pdb), extended_vocab=EXTENDED_VOCAB)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _run(inference, checkpoint: Path, batch: dict, dtype: torch.dtype) -> dict[str, np.ndarray]:  # noqa: ANN001
  # Upstream builds some tensors at torch's DEFAULT dtype even in a double model; the RBF centres
  # (`torch.linspace` in graph_embeddings.py:398) are the one that matters, because float32-rounded centres
  # put ~3e-6 into the edge features of an otherwise-double run. The default dtype is therefore set to double
  # for the f64 run, so the oracle is as double as upstream's code can be, and restored afterwards.
  previous_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    return _run_inner(inference, checkpoint, batch, dtype)
  finally:
    torch.set_default_dtype(previous_default)


def _run_inner(inference, checkpoint: Path, batch: dict, dtype: torch.dtype) -> dict[str, np.ndarray]:  # noqa: ANN001
  import mpnn.model.pottsmpnn as pottsmpnn

  model = inference.load_model(str(checkpoint))
  if dtype == torch.float64:
    model = model.double()
  network_input = copy.deepcopy(batch["network_input"])
  features = network_input["input_features"]
  features["repeat_sample_num"] = 1
  if dtype == torch.float64:
    for key, value in list(features.items()):
      if isinstance(value, torch.Tensor) and value.is_floating_point():
        features[key] = value.double()

  captured: dict[str, torch.Tensor] = {}

  def _pre(_module, args):  # noqa: ANN001, ANN202
    h_v = args[0].detach().clone()
    if h_v.any():
      msg = "layer-0 h_V is not all zeros, so its float32 dtype in the f64 run would not be value-neutral"
      raise RuntimeError(msg)
    # Upstream builds the layer-0 node features as float32 ZEROS even in a double model (measured). Casting
    # zeros to the run dtype changes no value, and the check above refuses to run if they were not zeros.
    captured["enc_in_h_V"] = h_v.to(dtype)
    captured["enc_in_h_E"] = args[1].detach().clone()

  def _post(index: int):  # noqa: ANN202
    def hook(_module, _inputs, output):  # noqa: ANN001, ANN202
      h_v, h_e = output[0], output[1]
      captured[f"enc_L{index}_h_V"] = h_v.detach().clone()
      captured[f"enc_L{index}_h_E"] = h_e.detach().clone()

    return hook

  handles = [model.encoder_layers[0].register_forward_pre_hook(_pre)]
  if dtype == torch.float64:
    # Upstream is not dtype-polymorphic: positional_encoding.py:89 builds its one-hot with `.float()` and
    # feeds it to a (now double) linear layer. The one-hot is exactly 0/1, so casting it to the weight dtype
    # changes no value; this is the only change made to upstream's computation in the f64 run.
    positional = model.graph_featurization_module.positional_embedding.embed_positional_features
    handles.append(
      positional.register_forward_pre_hook(lambda m, args: (args[0].to(m.weight.dtype),))
    )
  handles += [
    layer.register_forward_hook(_post(i)) for i, layer in enumerate(model.encoder_layers)
  ]
  raw: dict[str, torch.Tensor] = {}
  handles.append(
    model.etab_out.register_forward_hook(lambda _m, _i, out: raw.__setitem__("linear", out.detach().clone()))
  )

  original = pottsmpnn.PottsMPNN.compute_potts_context

  def wrapped(self, input_features, graph_features, encoder_features):  # noqa: ANN001, ANN202
    e_mask = self.compute_edge_mask(input_features, graph_features)
    raw["e_mask"] = e_mask.detach().clone()
    raw["E_idx"] = graph_features["E_idx"].detach().clone()
    context = original(self, input_features, graph_features, encoder_features)
    raw["etab_postmerge"] = context.etab_out.detach().clone()
    raise _StopAfterHead

  pottsmpnn.PottsMPNN.compute_potts_context = wrapped
  try:
    with torch.no_grad():
      model(network_input)
  except _StopAfterHead:
    pass
  finally:
    pottsmpnn.PottsMPNN.compute_potts_context = original
    for handle in handles:
      handle.remove()

  # The head's table BEFORE the reciprocal merge: the same two steps upstream applies to the linear output
  # (compute_potts_context: edge mask, then the slot-0 diagonal). Replicated literally from the source.
  b, l, k = raw["E_idx"].shape
  v = model.potts_vocab_size
  premerge = raw["linear"] * raw["e_mask"].unsqueeze(-1).to(dtype=raw["linear"].dtype)
  premerge = premerge.view(b, l, k, v, v).clone()
  premerge[:, :, 0, :, :] = premerge[:, :, 0, :, :] * torch.eye(v, dtype=premerge.dtype)

  arrays = {name: _to_numpy(tensor)[0] for name, tensor in captured.items()}
  arrays["h_V"] = arrays[f"enc_L{ENCODER_LAYERS - 1}_h_V"]
  arrays["h_E"] = arrays[f"enc_L{ENCODER_LAYERS - 1}_h_E"]
  arrays["E_idx"] = _to_numpy(raw["E_idx"])[0]
  arrays["etab_premerge"] = _to_numpy(premerge)[0]
  arrays["etab_postmerge"] = _to_numpy(raw["etab_postmerge"])[0]
  return arrays


def _dump(args: argparse.Namespace, out: Path, manifests: dict[str, dict]) -> dict[str, dict]:
  out.mkdir(parents=True, exist_ok=True)
  report: dict[str, dict] = {}
  for cell, pdb in args.cells.items():
    inference, batch = _prepare(pdb)
    features = batch["network_input"]["input_features"]
    feature_arrays = {k: _to_numpy(v) for k, v in sorted(features.items()) if isinstance(v, torch.Tensor)}
    feature_hash = _content_hash(feature_arrays)
    sealed = _sealed_hash(manifests, cell)
    entry: dict = {
      "feature_content_sha256": feature_hash,
      "sealed_content_sha256": sealed,
      "features_match_sealed": sealed is not None and sealed == feature_hash,
      "n_residues": int(feature_arrays["S"].shape[-1]),
      "precisions": {},
    }
    for name, dtype in PRECISIONS:
      arrays = _run(inference, args.checkpoint, batch, dtype)
      np.savez(out / f"{cell}_{name}.npz", **arrays)
      entry["precisions"][name] = {
        "content_sha256": _content_hash(arrays),
        "dtypes": sorted({str(a.dtype) for k, a in arrays.items() if k != "E_idx"}),
        "keys": sorted(arrays),
      }
    report[cell] = entry
  return report


def _child(args: argparse.Namespace) -> dict[str, str]:
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
               "--out", str(Path(tmp) / "child"), "--child-hashes-to", str(result)]
    for name, pdb in args.cells.items():
      command += ["--cell", f"{name}={pdb}"]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def _flatten(report: dict[str, dict]) -> dict[str, str]:
  return {f"{cell}/{p}": v["content_sha256"] for cell, e in report.items() for p, v in e["precisions"].items()}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
  parser.add_argument("--sealed", action="append", default=[], metavar="ID=MANIFEST")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {n: Path(p) for n, _, p in (c.partition("=") for c in args.cell)}
  manifests = {
    i: json.loads(Path(p).read_text(encoding="utf-8"))
    for i, _, p in (s.partition("=") for s in args.sealed)
  }

  report = _dump(args, args.out / "protonpotts_v6_encoder", manifests)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(_flatten(report)), encoding="utf-8")
    return 0

  reproducible = _child(args) == _flatten(report)
  complete = True
  f64_dtype_ok = True
  precisions_differ = True
  e_idx_stable = True
  floor: dict[str, float] = {}
  for cell in args.cells:
    a32 = np.load(args.out / "protonpotts_v6_encoder" / f"{cell}_f32.npz")
    a64 = np.load(args.out / "protonpotts_v6_encoder" / f"{cell}_f64.npz")
    complete &= all(k in a32.files for k in ("h_V", "h_E", "E_idx", "etab_premerge", "etab_postmerge"))
    complete &= a32["h_E"].shape == (a32["E_idx"].shape[0], a32["E_idx"].shape[1], 128)
    f64_dtype_ok &= all(a64[k].dtype == np.float64 for k in a64.files if k != "E_idx")
    precisions_differ &= float(np.abs(a64["h_E"] - a32["h_E"].astype(np.float64)).max()) > 0.0
    e_idx_stable &= bool(np.array_equal(a32["E_idx"], a64["E_idx"]))
    for key in ("h_V", "h_E", "etab_premerge", "etab_postmerge"):
      floor[f"{cell}/{key}"] = float(np.abs(a64[key] - a32[key].astype(np.float64)).max())

  flags = {
    "features_match_sealed": all(e["features_match_sealed"] for e in report.values()),
    "complete": bool(complete),
    "f64_is_float64": bool(f64_dtype_ok),
    "precisions_differ": bool(precisions_differ),
    "reproducible": bool(reproducible),
    "e_idx_precision_stable": bool(e_idx_stable),
    "n_cells": len(report),
  }
  (args.out / "encoder_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
       "torch_version": torch.__version__, "cells": {n: str(p) for n, p in args.cells.items()},
       "report": report, "f32_vs_f64_floor": floor, "flags": flags},
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
