"""G1n: does the SHIPPING JavaScript loop reproduce the monolith, under ORT-Web wasm?

G1 (run d2a06073) composed the four graphs in Python and matched the monolith exactly on
56/56 cells. That established the DECOMPOSITION is right. It did not touch
``browser/aminx-sampler/split_loop.mjs``, which is the code a page actually loads.

This gate closes that gap: it drives the real .mjs through ``run_split_node.mjs`` with
onnxruntime-web's wasm backend, against the same four exported graphs, on the same
fixtures, knobs and seeds, and compares to the same references. The two arms are kept
separate on purpose -- a pass in G1 with a failure here localises the fault to the
JavaScript rather than the cut, which a single combined gate could not do.

The Node runner deliberately grades nothing; it reads inputs, runs the loop and writes
outputs. Every comparison and threshold lives here, in the tracked script with the
sidecar, so the thing being measured cannot decide what counts as agreement.

Threading is pinned to 1 in the runner. Thread count changes float accumulation order, so
leaving it free would introduce a second variable between this arm and the ORT-CPU arm.

Honest naming: this is ORT-Web's wasm backend under **Node**, not a browser. It is the
same wasm module a page would load, but not the same environment. A Chromium run is a
separate, stronger gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("p07_split_node_parity")

N_TOKENS = 21
DEFAULT_LOGPROB_BOUND = 2e-4
CTRL_PERTURB = 1.0
_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")


def _git_state(repo: Path) -> tuple[str, bool, str]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo,
      capture_output=True,
      text=True,
      check=True,
      timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  ), _run("status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS)


def _spec(a: np.ndarray) -> dict[str, Any]:
  arr = np.asarray(a)
  dtype = {"float32": "float32", "int32": "int32", "bool": "bool"}.get(str(arr.dtype))
  if dtype is None:
    msg = f"unsupported dtype for JSON transport: {arr.dtype}"
    raise ValueError(msg)
  return {"dtype": dtype, "shape": list(arr.shape), "data": arr.reshape(-1).tolist()}


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--monolith", type=Path, required=True)
  parser.add_argument("--ort-dir", type=Path, required=True, help="dir whose node_modules has ORT")
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--logprob-bound", type=float, default=DEFAULT_LOGPROB_BOUND)
  parser.add_argument("--max-cells", type=int, default=0)
  parser.add_argument("--node-bin", type=str, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
  )

  git_hash, git_clean, git_dirty_paths = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "logprob_bound": args.logprob_bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "git_dirty_paths": git_dirty_paths,
    "jax_enable_x64": False,
    "cells_total": 0,
    "cells_tokens_exact_vs_monolith": 0,
    "cells_tokens_exact_vs_jax": 0,
    "max_logprob_abs_diff_vs_monolith": 0.0,
    "max_logprob_abs_diff_vs_jax": 0.0,
    "worst_cell": "",
    "node_ok": False,
    "ort_version": "",
    "num_threads": -1,
    "controls_total": 1,
    "controls_detected": 0,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is True; refusing."
      raise RuntimeError(msg)

    import onnxruntime as ort  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import (  # noqa: PLC0415
      FULL_KNOBS,
      FULL_SEEDS,
      P07_KEYS,
      _call_p07,
      _load_manifest,
      _node_bin,
      _pad_geometry,
      _parse_geometry,
      _select_fixtures,
      _structure_payload,
      _with_chain_split,
      js_build,
      runspec_for,
    )
    from aminx.export.wrappers import make_p07_sample  # noqa: PLC0415
    from aminx.inference.logits import make_stage_set  # noqa: PLC0415, TID251
    from aminx.io.weights import load_model  # noqa: PLC0415

    node = _node_bin(args.node_bin)
    work = Path(tempfile.mkdtemp(prefix="p07_split_node_"))
    geometries = [
      _parse_geometry(row, work / "canonical")
      for row in _select_fixtures(_load_manifest(), [args.bucket])
    ]
    p07 = make_p07_sample(load_model(checkpoint_id="proteinmpnn_v_48_020"), make_stage_set())
    mono = ort.InferenceSession(str(args.monolith), providers=["CPUExecutionProvider"])
    mono_names = [i.name for i in mono.get_inputs()]

    cells: list[dict[str, Any]] = []
    refs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for geom in geometries:
      if int(geom["n_real"]) > args.bucket:
        continue
      padded = _pad_geometry(geom, args.bucket)
      chain_split = _pad_geometry(_with_chain_split(geom), args.bucket)
      for knob in FULL_KNOBS:
        host = chain_split if knob in ("chains_to_design", "all_combined") else padded
        for seed in FULL_SEEDS:
          if args.max_cells and len(cells) >= args.max_cells:
            break
          spec = runspec_for(knob, seed, host)
          arrays = js_build(_structure_payload(host), spec, node)
          name = f"{geom['name']}_{knob}_s{seed}"
          tok_jax, lp_jax = _call_p07(p07, arrays)
          mono_out = mono.run(
            None, {n: np.asarray(arrays[k]) for n, k in zip(mono_names, P07_KEYS, strict=True)}
          )
          refs[name] = (
            np.asarray(tok_jax),
            np.asarray(lp_jax),
            np.asarray(mono_out[0]),
            np.asarray(mono_out[1]),
          )
          cells.append(
            {"name": name, "arrays": {k: _spec(np.asarray(arrays[k])) for k in P07_KEYS}}
          )
        if args.max_cells and len(cells) >= args.max_cells:
          break
      if args.max_cells and len(cells) >= args.max_cells:
        break

    cells_path = work / "cells.json"
    node_out_path = work / "node_out.json"
    cells_path.write_text(json.dumps(cells))
    logger.info("dumped %d cells -> %s", len(cells), cells_path)

    runner = _REPO_ROOT / "browser" / "aminx-sampler" / "run_split_node.mjs"
    proc = subprocess.run(  # noqa: S603
      [
        node,
        str(runner),
        "--cells",
        str(cells_path),
        "--models",
        str(args.release_dir),
        "--bucket",
        str(args.bucket),
        "--out",
        str(node_out_path),
        "--ort-dir",
        str(args.ort_dir),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=7200,
    )
    if proc.returncode != 0 or not node_out_path.is_file():
      msg = f"node runner failed rc={proc.returncode}: {proc.stderr[-2000:]}"
      raise RuntimeError(msg)
    node_out = json.loads(node_out_path.read_text())
    result["node_ok"] = True
    result["ort_version"] = str(node_out.get("ort_version", ""))
    result["num_threads"] = int(node_out.get("num_threads", -1))

    for i, cell in enumerate(node_out["cells"]):
      name = cell["name"]
      tok_jax, lp_jax, tok_mono, lp_mono = refs[name]
      tok_js = np.asarray(cell["tokens"], dtype=np.int32)
      lp_js = np.asarray(cell["log_probs"], dtype=np.float64).reshape(-1, N_TOKENS)

      result["cells_total"] += 1
      if np.array_equal(tok_js, tok_mono):
        result["cells_tokens_exact_vs_monolith"] += 1
      if np.array_equal(tok_js, tok_jax):
        result["cells_tokens_exact_vs_jax"] += 1
      d_mono = float(np.max(np.abs(lp_js - lp_mono.astype(np.float64))))
      d_jax = float(np.max(np.abs(lp_js - lp_jax.astype(np.float64))))
      if d_mono > result["max_logprob_abs_diff_vs_monolith"]:
        result["max_logprob_abs_diff_vs_monolith"] = d_mono
        result["worst_cell"] = name
      result["max_logprob_abs_diff_vs_jax"] = max(result["max_logprob_abs_diff_vs_jax"], d_jax)

      if i == 0:
        poisoned = lp_js.copy()
        poisoned.flat[0] += CTRL_PERTURB
        result["ctrl_perturb_detected"] = bool(
          np.max(np.abs(poisoned - lp_mono.astype(np.float64))) > args.logprob_bound
        )
      logger.info(
        "%s: tok_mono=%s tok_jax=%s d_mono=%.3e",
        name,
        np.array_equal(tok_js, tok_mono),
        np.array_equal(tok_js, tok_jax),
        d_mono,
      )

    result["controls_detected"] = int(result["ctrl_perturb_detected"])

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G1n node parity failed")

  total = result["cells_total"]
  result["outcome"] = (
    "incomplete"
    if result["jax_enable_x64"]
    or total < 32
    or not result["git_clean"]
    or result["failure"]
    or not result["node_ok"]
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["cells_tokens_exact_vs_monolith"] == total
    and result["cells_tokens_exact_vs_jax"] == total
    and result["max_logprob_abs_diff_vs_monolith"] <= args.logprob_bound
    and result["max_logprob_abs_diff_vs_jax"] <= args.logprob_bound
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
