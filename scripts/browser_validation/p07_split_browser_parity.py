"""G1c: does the split reproduce the monolith in a REAL browser?

Everything before this ran the shipping JavaScript under Node. That executes the same
wasm module a page loads, but it is not the same environment: no cross-origin isolation,
no browser APIs, a different module loader, and — as run 98a62501 showed — threaded wasm
does not even initialise from a file:// path. This gate closes that gap by serving the
graphs over HTTP with COOP/COEP and driving headless Chromium.

It reuses `browser/layer_c/run_p07.mjs` unchanged rather than writing a second driver.
That driver already serves with isolation, captures console errors, and records Chromium
and Playwright versions; the page simply honours its existing contract (set
`window.__PARITY_DONE__`, put the payload on `window.__PARITY_RESULT__`). Reusing it also
means the split arm and the monolith arm are exercised through identical harness code, so
a difference between them cannot be a difference in harnesses.

The page grades nothing. It runs the loop and reports outputs; every comparison and
threshold lives here.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
_BV = _REPO_ROOT / "scripts" / "browser_validation"
if str(_BV) not in sys.path:
  sys.path.insert(0, str(_BV))

logger = logging.getLogger("p07_split_browser_parity")

N_TOKENS = 21
DEFAULT_BOUND = 2e-4
CTRL_PERTURB = 1.0
_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")
_SAMPLER = _REPO_ROOT / "browser" / "aminx-sampler"


def _git_state(repo: Path) -> tuple[str, bool]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  )


def _spec(a: np.ndarray) -> dict[str, Any]:
  arr = np.asarray(a)
  dtype = {"float32": "float32", "int32": "int32", "bool": "bool"}.get(str(arr.dtype))
  if dtype is None:
    msg = f"unsupported dtype {arr.dtype}"
    raise ValueError(msg)
  return {"dtype": dtype, "shape": list(arr.shape), "data": arr.reshape(-1).tolist()}


def _assemble_site(site: Path, release_dir: Path, ort_dir: Path, bucket: int) -> None:
  """Lay out exactly what the page fetches: itself, the loop, the ORT dist, the graphs."""
  site.mkdir(parents=True, exist_ok=True)
  (site / "models").mkdir(exist_ok=True)
  (site / "ort").mkdir(exist_ok=True)

  shutil.copy(_SAMPLER / "split_index.html", site / "index.html")
  shutil.copy(_SAMPLER / "split_page.mjs", site / "split_page.mjs")
  shutil.copy(_SAMPLER / "split_loop.mjs", site / "split_loop.mjs")
  for key in ("encoder", "wave", "decoder", "fuse"):
    name = f"p07_{key}_L{bucket}.onnx"
    shutil.copy(release_dir / name, site / "models" / name)
  # The whole ORT dist, not a hand-picked subset: the threaded build fetches sibling
  # .wasm and worker files by name at runtime, and guessing which ones is how a
  # multithreaded run fails with an unhelpful "no available backend".
  dist = ort_dir / "node_modules" / "onnxruntime-web" / "dist"
  for f in dist.iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--monolith", type=Path, required=True)
  parser.add_argument("--ort-dir", type=Path, required=True)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--cells", type=int, default=8)
  parser.add_argument("--threads", type=int, default=1)
  parser.add_argument("--bound", type=float, default=DEFAULT_BOUND)
  parser.add_argument("--timeout-ms", type=int, default=1_800_000)
  parser.add_argument("--node-bin", type=str, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
  )

  git_hash, git_clean = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "bound": args.bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "cells_total": 0,
    "cells_tokens_exact_vs_monolith": 0,
    "cells_tokens_exact_vs_jax": 0,
    "max_logprob_abs_diff_vs_monolith": 0.0,
    "max_logprob_abs_diff_vs_jax": 0.0,
    "worst_cell": "",
    "browser_ok": False,
    "cross_origin_isolated": False,
    "num_threads_requested": args.threads,
    "num_threads_effective": -1,
    "chromium_version": "",
    "median_wall_ms": -1.0,
    "controls_total": 1,
    "controls_detected": 0,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import onnxruntime as ort  # noqa: PLC0415

    from p07_knobs_gate import (  # noqa: PLC0415
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
    work = Path(tempfile.mkdtemp(prefix="p07_split_browser_"))
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
        for seed in FULL_SEEDS[:1]:
          if len(cells) >= args.cells:
            break
          spec = runspec_for(knob, seed, host)
          arrays = js_build(_structure_payload(host), spec, node)
          name = f"{geom['name']}_{knob}_s{seed}"
          tok_jax, lp_jax = _call_p07(p07, arrays)
          mono_out = mono.run(
            None, {n: np.asarray(arrays[k]) for n, k in zip(mono_names, P07_KEYS, strict=True)}
          )
          refs[name] = (
            np.asarray(tok_jax), np.asarray(lp_jax),
            np.asarray(mono_out[0]), np.asarray(mono_out[1]),
          )
          cells.append(
            {"name": name, "arrays": {k: _spec(np.asarray(arrays[k])) for k in P07_KEYS}}
          )
        if len(cells) >= args.cells:
          break
      if len(cells) >= args.cells:
        break

    site = work / "site"
    out_dir = work / "results"
    _assemble_site(site, args.release_dir, args.ort_dir, args.bucket)
    (site / "cells.json").write_text(json.dumps({"bucket": args.bucket, "cells": cells}))
    logger.info("assembled site with %d cells at %s", len(cells), site)

    runner = _REPO_ROOT / "browser" / "layer_c" / "run_p07.mjs"
    cmd = [
      node, str(runner),
      "--site", str(site),
      "--out-dir", str(out_dir),
      "--num-threads", str(args.threads),
      "--timeout-ms", str(args.timeout_ms),
    ]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, timeout=args.timeout_ms // 1000 + 600)  # noqa: S603
    harness_path = out_dir / "harness.json"
    if not harness_path.is_file():
      msg = f"browser harness wrote no harness.json (rc={proc.returncode})"
      raise RuntimeError(msg)
    harness = json.loads(harness_path.read_text())
    result["chromium_version"] = str(harness.get("chromiumVersion") or "")
    if not harness.get("harnessOk"):
      msg = f"browser harness failed: {str(harness.get('harnessError'))[:800]}"
      raise RuntimeError(msg)

    payload = harness["result"]
    result["browser_ok"] = True
    result["cross_origin_isolated"] = bool(payload.get("cross_origin_isolated"))
    result["num_threads_effective"] = int(payload.get("num_threads_effective", -1))

    walls = []
    for i, cell in enumerate(payload["cells"]):
      name = cell["name"]
      tok_jax, lp_jax, tok_mono, lp_mono = refs[name]
      tok_js = np.asarray(cell["tokens"], dtype=np.int32)
      lp_js = np.asarray(cell["log_probs"], dtype=np.float64).reshape(-1, N_TOKENS)
      walls.append(float(cell["wall_ms"]))

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
          np.max(np.abs(poisoned - lp_mono.astype(np.float64))) > args.bound
        )
      logger.info(
        "%s: tok_mono=%s tok_jax=%s d_mono=%.3e wall=%.0f ms",
        name, np.array_equal(tok_js, tok_mono), np.array_equal(tok_js, tok_jax), d_mono,
        float(cell["wall_ms"]),
      )

    if walls:
      result["median_wall_ms"] = float(np.median(walls))
    result["controls_detected"] = int(result["ctrl_perturb_detected"])

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G1c browser parity failed")

  total = result["cells_total"]
  result["outcome"] = (
    "incomplete"
    if not result["browser_ok"] or total < 4 or not result["git_clean"] or result["failure"]
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["cells_tokens_exact_vs_monolith"] == total
    and result["cells_tokens_exact_vs_jax"] == total
    and result["max_logprob_abs_diff_vs_monolith"] <= args.bound
    and result["max_logprob_abs_diff_vs_jax"] <= args.bound
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
