"""Dump upstream ProtonPottsMPNN (v6) DECODER behaviour: autoregressive sampling with injected uniforms, and three
teacher-forced modes, in f32 and f64 (P4h).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §53; debt #2617. Sidecar:
``dump_protonpotts_decoder.bth.toml`` (committed before this script). Run in the `aminx-oracles-protonpotts` environment;
imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- <oracle-env python> scripts/protonpotts/dump_protonpotts_decoder.py \\
        --checkpoint <ckpt> --out <dir> --cell pkad_unlabelled=<1BVC.pdb> --cell multichain=<6m0j.pdb> \\
        --cell only_o_1olr=<1OLR.pdb> --cell gap=<1EL1.pdb> \\
        --sealed p4b=<features_manifest.json> --sealed p4c=<features_p4c_manifest.json> --sealed p4d=<...>

WHAT IS DUMPED. See the sidecar. In short: per cell and precision, three autoregressive configurations (ar_default, ar_bias_temp,
ar_fixed) with ``torch.multinomial`` replaced by inverse-CDF on INJECTED uniforms, and three teacher-forced patterns
(auto_regressive, conditional, conditional_minus_self).

THE f64 RUN changes upstream's computation in exactly three ways, all listed in the sidecar: the positional one-hot is cast
to the weight dtype; the ``torch.float32`` / ``.float()`` literals of three decode-path methods are widened by rewriting their
source; the decoding-order noise is drawn in float32 and cast, so both precisions see the same order. The originals are restored
after every run.

Threads are pinned to 1 (spec §32).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_encoder import PRECISIONS, _prepare, _sealed_hash, _to_numpy  # noqa: E402
from dump_protonpotts_features import _content_hash  # noqa: E402

logger = logging.getLogger("dump_protonpotts_decoder")

SEED = 0
AR_CONFIGS = ("ar_default", "ar_bias_temp", "ar_fixed")
TF_PATTERNS = ("auto_regressive", "conditional", "conditional_minus_self")
DECODER_LAYERS = 3
FIXED_EVERY, FIXED_OFFSET = 7, 3
WIDEN = ("setup_causality_masks", "decode_auto_regressive", "decode_teacher_forcing")


@contextmanager
def _f64_changes(mm, positional):  # noqa: ANN001, ANN202
  """The three f64-only changes to upstream's computation (see the module docstring); restored on exit."""
  saved = {}
  for name in WIDEN:
    original = getattr(mm.ProteinMPNN, name)
    saved[name] = original
    src = textwrap.dedent(inspect.getsource(original))
    namespace = dict(vars(mm))
    exec(src.replace("torch.float32", "torch.float64").replace(".float()", ".double()"), namespace)  # noqa: S102
    setattr(mm.ProteinMPNN, name, namespace[name])
  original_randn = torch.randn

  def randn32(*size, **kwargs):  # noqa: ANN002, ANN003, ANN202
    if "dtype" not in kwargs:
      return original_randn(*size, **{**kwargs, "dtype": torch.float32}).to(torch.get_default_dtype())
    return original_randn(*size, **kwargs)

  torch.randn = randn32
  handle = positional.register_forward_pre_hook(lambda m, args: (args[0].to(m.weight.dtype),))
  try:
    yield
  finally:
    handle.remove()
    torch.randn = original_randn
    for name, original in saved.items():
      setattr(mm.ProteinMPNN, name, original)


class _Draws:
  """``torch.multinomial`` replaced by inverse-CDF on injected uniforms (aminx ``decode.py`` convention)."""

  def __init__(self, uniforms: np.ndarray) -> None:
    self.uniforms = uniforms
    self.probs: list[torch.Tensor] = []
    self.index: list[int] = []
    self.margin: list[float] = []

  def __call__(self, probs: torch.Tensor, num_samples: int, replacement: bool = False, generator=None):  # noqa: ANN001, ANN204, ARG002, FBT001, FBT002
    if probs.shape[0] != 1 or num_samples != 1:
      msg = f"shim expects one row and one sample, got {tuple(probs.shape)} x {num_samples}"
      raise RuntimeError(msg)
    k = len(self.probs)
    row = probs[0]
    cdf = torch.cumsum(row, dim=-1)
    target = torch.tensor(float(self.uniforms[k]), dtype=row.dtype) * cdf[-1]
    last = int(torch.nonzero(row > 0)[-1])
    index = min(int((cdf <= target).sum()), last)
    self.probs.append(row.detach().clone())
    self.index.append(index)
    self.margin.append(float((cdf - target).abs().min() / cdf[-1]))
    return torch.tensor([[index]], dtype=torch.long)


def _forward(model, batch, dtype, mode, *, uniforms=None, edits=None):  # noqa: ANN001, ANN202, C901
  """One upstream forward; returns its numpy arrays. ``mode`` is an AR config name or a teacher-forcing pattern."""
  import mpnn.model.mpnn as mm

  network_input = copy.deepcopy(batch["network_input"])
  features = network_input["input_features"]
  if dtype == torch.float64:
    for key, value in list(features.items()):
      if isinstance(value, torch.Tensor) and value.is_floating_point():
        features[key] = value.double()
  features["repeat_sample_num"] = 1
  for key, value in (edits or {}).items():
    features[key] = value.to(dtype) if value.is_floating_point() else value
  autoregressive = mode in AR_CONFIGS
  if not autoregressive:
    features.update({
      "decode_type": "teacher_forcing", "causality_pattern": mode,
      "initialize_sequence_embedding_with_ground_truth": True,
    })

  captured: dict[str, list] = {"order": [], "wout": [], **{f"layer{i}": [] for i in range(DECODER_LAYERS)}}
  handles = [model.W_out.register_forward_hook(lambda _m, _i, out: captured["wout"].append(out.detach().clone()))]
  if autoregressive:
    for i, layer in enumerate(model.decoder_layers):
      handles.append(layer.register_forward_hook(
        lambda _m, _i, out, i=i: captured[f"layer{i}"].append((out[0] if isinstance(out, tuple) else out).detach().clone())
      ))
  original_setup = mm.ProteinMPNN.setup_causality_masks

  def wrapped(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    out = original_setup(self, *args, **kwargs)
    captured["order"].append(out["decoding_order"].detach().clone())
    return out

  mm.ProteinMPNN.setup_causality_masks = wrapped
  draws = _Draws(uniforms) if autoregressive else None
  original_multinomial = torch.multinomial
  if draws is not None:
    torch.multinomial = draws
  try:
    torch.manual_seed(SEED)
    with torch.no_grad():
      out = model(network_input)
  finally:
    torch.multinomial = original_multinomial
    mm.ProteinMPNN.setup_causality_masks = original_setup
    for handle in handles:
      handle.remove()

  decoder_features = out["decoder_features"]
  arrays: dict[str, np.ndarray] = {
    "log_probs": _to_numpy(decoder_features["log_probs"])[0],
    "decoding_order": _to_numpy(captured["order"][0])[0],
  }
  order = arrays["decoding_order"]
  if autoregressive:
    n = order.shape[0]
    arrays["S_sampled"] = _to_numpy(decoder_features["S_sampled"])[0]
    arrays["S_argmax"] = _to_numpy(decoder_features["S_argmax"])[0]
    arrays["uniforms"] = np.asarray(uniforms, dtype=np.float64)
    v = arrays["log_probs"].shape[-1]
    by_pos = {name: np.zeros((n, v), dtype=arrays["log_probs"].dtype) for name in ("probs_sample", "logits")}
    layers = np.zeros((DECODER_LAYERS, n, captured["layer0"][0].shape[-1]), dtype=arrays["log_probs"].dtype)
    drawn = np.zeros(n, dtype=np.int64)
    margin = np.zeros(n, dtype=np.float64)
    for step in range(n):
      pos = int(order[step])
      by_pos["probs_sample"][pos] = _to_numpy(draws.probs[step])
      by_pos["logits"][pos] = _to_numpy(captured["wout"][step])[0, 0]
      drawn[pos] = draws.index[step]
      margin[pos] = draws.margin[step]
      for i in range(DECODER_LAYERS):
        layers[i, pos] = _to_numpy(captured[f"layer{i}"][step])[0, 0]
    arrays.update(probs_sample=by_pos["probs_sample"], logits=by_pos["logits"], decoder_layers=layers,
                  drawn_token=drawn, draw_margin=margin)
  else:
    arrays["logits"] = _to_numpy(captured["wout"][-1])[0]
  return arrays


def _edits(config: str, features: dict, dtype: torch.dtype) -> dict[str, torch.Tensor]:
  n = int(features["S"].shape[-1])
  v = 30
  rng = np.random.default_rng(SEED + 17)
  if config == "ar_bias_temp":
    return {
      "temperature": torch.tensor(rng.uniform(0.3, 1.0, size=(1, n)), dtype=dtype),
      "bias": torch.tensor(0.5 * rng.standard_normal(size=(1, n, v)), dtype=dtype),
    }
  if config == "ar_fixed":
    # None means every residue is designed (the prepared default); start from all-true either way.
    given = features.get("designed_residue_mask")
    designed = torch.ones(features["S"].shape, dtype=torch.bool) if given is None else given.clone()
    designed[:, FIXED_OFFSET::FIXED_EVERY] = False
    return {"designed_residue_mask": designed}
  return {}


def _uniforms(config: str, n: int) -> np.ndarray:
  return np.random.default_rng(SEED + 100 + AR_CONFIGS.index(config)).uniform(size=n)


def _run(inference, checkpoint: Path, batch: dict, dtype: torch.dtype) -> dict[str, np.ndarray]:  # noqa: ANN001
  import mpnn.model.mpnn as mm

  previous_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    model = inference.load_model(str(checkpoint))
    if dtype == torch.float64:
      model = model.double()
    positional = model.graph_featurization_module.positional_embedding.embed_positional_features
    arrays: dict[str, np.ndarray] = {}
    n = int(batch["network_input"]["input_features"]["S"].shape[-1])

    def go() -> None:
      for config in AR_CONFIGS:
        edits = _edits(config, batch["network_input"]["input_features"], dtype)
        out = _forward(model, batch, dtype, config, uniforms=_uniforms(config, n), edits=edits)
        arrays.update({f"{config}__{k}": v for k, v in out.items()})
      for pattern in TF_PATTERNS:
        out = _forward(model, batch, dtype, pattern)
        arrays.update({f"tf_{pattern}__{k}": v for k, v in out.items()})

    if dtype == torch.float64:
      with _f64_changes(mm, positional):
        go()
    else:
      go()
    return arrays
  finally:
    torch.set_default_dtype(previous_default)


def _draw_matches(arrays: dict[str, np.ndarray], config: str, native: np.ndarray) -> bool:
  """Recompute every step's token from the recorded probabilities and uniform, in the run's own dtype."""
  probs = arrays[f"{config}__probs_sample"]
  order = arrays[f"{config}__decoding_order"]
  uniforms = arrays[f"{config}__uniforms"]
  sampled = arrays[f"{config}__S_sampled"]
  for step, pos in enumerate(order):
    row = probs[pos]
    cdf = np.cumsum(row)
    target = uniforms[step].astype(row.dtype) * cdf[-1]
    index = min(int((cdf <= target).sum()), int(np.nonzero(row > 0)[0][-1]))
    if config == "ar_fixed" and (int(pos) - FIXED_OFFSET) % FIXED_EVERY == 0 and int(pos) >= FIXED_OFFSET:
      if int(sampled[pos]) != int(native[pos]):
        return False
    elif int(sampled[pos]) != index:
      return False
  return True


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
      "feature_content_sha256": feature_hash, "sealed_content_sha256": sealed,
      "features_match_sealed": sealed is not None and sealed == feature_hash,
      "n_residues": int(feature_arrays["S"].shape[-1]), "native": feature_arrays["S"][0].tolist(), "precisions": {},
    }
    for name, dtype in PRECISIONS:
      arrays = _run(inference, args.checkpoint, batch, dtype)
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
    for name, pdb in args.cells.items():
      command += ["--cell", f"{name}={pdb}"]
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603
    return json.loads(result.read_text(encoding="utf-8"))


def _flatten(report: dict[str, dict]) -> dict[str, str]:
  return {f"{cell}/{p}": v["content_sha256"] for cell, e in report.items() for p, v in e["precisions"].items()}


def main() -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
  parser.add_argument("--sealed", action="append", default=[], metavar="ID=MANIFEST")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {n: Path(p) for n, _, p in (c.partition("=") for c in args.cell)}
  manifests = {i: json.loads(Path(p).read_text(encoding="utf-8")) for i, _, p in (s.partition("=") for s in args.sealed)}

  directory = args.out / "protonpotts_v6_decoder"
  report = _dump(args, directory, manifests)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(_flatten(report)), encoding="utf-8")
    return 0

  reproducible = _child(args) == _flatten(report)
  complete = f64_dtype_ok = precisions_differ = order_same = shim_ok = fixed_ok = not_inert = True
  floor: dict[str, float] = {}
  tokens_equal: dict[str, bool] = {}
  min_margin: dict[str, float] = {}
  ar_keys = ("decoding_order", "uniforms", "S_sampled", "S_argmax", "log_probs", "probs_sample", "logits",
             "decoder_layers", "drawn_token", "draw_margin")
  tf_keys = ("log_probs", "logits", "decoding_order")
  for cell, entry in report.items():
    n = entry["n_residues"]
    native = np.asarray(entry["native"])
    a32 = dict(np.load(directory / f"{cell}_f32.npz"))
    a64 = dict(np.load(directory / f"{cell}_f64.npz"))
    for config in AR_CONFIGS:
      complete &= all(f"{config}__{k}" in a32 and f"{config}__{k}" in a64 for k in ar_keys)
      order32, order64 = a32[f"{config}__decoding_order"], a64[f"{config}__decoding_order"]
      order_same &= bool(np.array_equal(order32, order64)) and sorted(order32.tolist()) == list(range(n))
      shim_ok &= _draw_matches(a32, config, native) and _draw_matches(a64, config, native)
      tokens_equal[f"{cell}/{config}"] = bool(np.array_equal(a32[f"{config}__S_sampled"], a64[f"{config}__S_sampled"]))
      min_margin[f"{cell}/{config}"] = float(a32[f"{config}__draw_margin"].min())
      for key in ("log_probs", "logits"):
        floor[f"{cell}/{config}/{key}"] = float(
          np.abs(a64[f"{config}__{key}"] - a32[f"{config}__{key}"].astype(np.float64)).max())
    fixed_pos = np.arange(n)[FIXED_OFFSET::FIXED_EVERY]
    for a in (a32, a64):
      fixed_ok &= bool(np.array_equal(a["ar_fixed__S_sampled"][fixed_pos], native[fixed_pos]))
    for pattern in TF_PATTERNS:
      complete &= all(f"tf_{pattern}__{k}" in a32 and f"tf_{pattern}__{k}" in a64 for k in tf_keys)
      for key in ("log_probs", "logits"):
        floor[f"{cell}/tf_{pattern}/{key}"] = float(
          np.abs(a64[f"tf_{pattern}__{key}"] - a32[f"tf_{pattern}__{key}"].astype(np.float64)).max())
    f64_dtype_ok &= all(a64[k].dtype == np.float64 for k in a64
                        if a64[k].dtype.kind == "f")
    precisions_differ &= float(np.abs(a64["ar_default__log_probs"] - a32["ar_default__log_probs"].astype(np.float64)).max()) > 0.0
    not_inert &= float(np.abs(a32["ar_bias_temp__log_probs"] - a32["ar_default__log_probs"]).max()) > 0.0
    not_inert &= not np.array_equal(a32["ar_fixed__decoding_order"], a32["ar_default__decoding_order"])

  flags = {
    "features_match_sealed": all(e["features_match_sealed"] for e in report.values()),
    "complete": bool(complete), "f64_is_float64": bool(f64_dtype_ok), "precisions_differ": bool(precisions_differ),
    "order_same_across_precisions": bool(order_same), "shim_is_the_draw": bool(shim_ok),
    "fixed_positions_hold": bool(fixed_ok), "configs_not_inert": bool(not_inert),
    "reproducible": bool(reproducible), "n_cells": len(report),
  }
  (args.out / "decoder_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "torch_version": torch.__version__,
       "cells": {n: str(p) for n, p in args.cells.items()}, "report": report, "f32_vs_f64_floor": floor,
       "tokens_equal_across_precisions": tokens_equal, "min_draw_margin_f32": min_margin, "flags": flags,
       "fixed_every": FIXED_EVERY, "fixed_offset": FIXED_OFFSET, "seed": SEED},
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
