"""Layer (c) shared plumbing: browser-site assembly + Playwright driving (T5a).

Layer (c) reuses layer (b)'s already-converted, tracked ONNX artifacts
(`outputs/browser_validation/layer_b/artifact_manifest.json`, D-G) and runs them a
SECOND time -- through onnxruntime-web's wasm execution provider inside headless
Chromium (`browser/layer_c/`), instead of onnxruntime's native CPU execution provider
(`layer_b_ort_calibrate.py`'s "b-ORT" arm). All comparison numerics
(`layer_b_common.near_tie_rows`/`compare_neighbor_indices`/`classify_headroom`) are
reused unchanged; this module only adds what is NEW for a browser-driven arm:

- **`find_node`** -- locate a node binary (explicit arg, `$NODE_BIN`, `$PATH`, nvm dirs),
  same discovery order as Phase 0's `ort_web_smoke.py`.
- **`write_raw_input`** / **`read_raw_output`** -- raw (headerless) little-endian
  buffers, matching `parity.mjs`'s `fetch(...).arrayBuffer()` / `POST` convention (NOT
  the JSON-tensor convention `ort_web_smoke.py`'s `smoke.mjs` uses -- the task spec asks
  for raw bytes, so a P04 log-prob array at L=1024 does not round-trip through a JSON
  array).
- **`assemble_site`** -- copies `browser/layer_c/{index.html,parity.mjs}` plus the
  onnxruntime-web dist (from `browser/layer_c/node_modules`) and every cell's model +
  input files into a fresh static-server root, and writes that root's `cells.json`.
- **`run_browser`** -- invokes `node run_parity.mjs --site ... --out-dir ...` (headless
  Chromium via Playwright) and returns its parsed harness JSON. Non-zero ONLY on a
  harness-level failure (browser launch, page crash, navigation error, timeout) -- a
  per-cell `ok: false` inside the result is data, not a harness failure (`run_parity.mjs`
  docstring).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
LAYER_C_DIR = _WORKTREE_ROOT / "browser" / "layer_c"

DTYPE_TO_NP: dict[str, Any] = {
  "float32": np.float32,
  "int32": np.int32,
  "int8": np.int8,
  "uint8": np.uint8,
}


def find_node(explicit: str | None = None) -> str | None:
  """Locate a node binary: explicit arg, then `$NODE_BIN`, then `$PATH`, then nvm dirs.

  Same discovery order as Phase 0's `ort_web_smoke.py:find_node` (this project's node
  toolchain is not reliably on `$PATH`; nvm-managed installs are the common case).
  """
  candidates: list[str] = []
  if explicit:
    candidates.append(explicit)
  env_bin = os.environ.get("NODE_BIN")
  if env_bin:
    candidates.append(env_bin)
  which = shutil.which("node")
  if which:
    candidates.append(which)
  nvm_dir = Path(os.environ.get("NVM_DIR", str(Path.home() / ".nvm"))) / "versions" / "node"
  if nvm_dir.is_dir():
    for version_dir in sorted(nvm_dir.iterdir(), reverse=True):
      candidate = version_dir / "bin" / "node"
      if candidate.is_file():
        candidates.append(str(candidate))
  for candidate in candidates:
    if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
      return str(candidate)
  return None


def write_raw_input(arr: np.ndarray, path: Path, dtype: str) -> dict[str, Any]:
  """Write `arr` (cast to `dtype`, C-contiguous) as headerless raw bytes to `path`.

  Returns the `{"dataFile", "dtype", "dims"}` spec `parity.mjs`'s `cells.json` expects
  for this input (dataFile is `path.name`, relative to the eventual site root -- the
  caller is responsible for placing `path` under a `data/` directory of that root).
  """
  np_dtype = DTYPE_TO_NP[dtype]
  contiguous = np.ascontiguousarray(arr, dtype=np_dtype)
  path.parent.mkdir(parents=True, exist_ok=True)
  contiguous.tofile(path)
  return {"dataFile": f"data/{path.name}", "dtype": dtype, "dims": list(contiguous.shape)}


def read_raw_output(path: Path, dtype: str, dims: tuple[int, ...]) -> np.ndarray:
  """Read a headerless raw buffer `parity.mjs` POSTed back, matching `write_raw_input`."""
  np_dtype = DTYPE_TO_NP[dtype]
  data = np.frombuffer(path.read_bytes(), dtype=np_dtype)
  return data.reshape(dims)


def assemble_site(
  cells: list[dict[str, Any]], model_files: dict[str, Path], site_dir: Path
) -> None:
  """Build a static-server root at `site_dir`: harness files + models + `cells.json`.

  Args:
    cells: Already-built `cells.json`-shaped entries (each `{"name", "onnx", "inputs",
      "outputs"}`, with `"onnx"` set to `f"models/{model_files[key].name}"` by the
      caller and every input's `dataFile` already written under `site_dir/data/` via
      `write_raw_input`).
    model_files: `{onnx_basename: source_path}` for every distinct model this run's
      cells reference; copied once each into `site_dir/models/`.
    site_dir: Destination root (created fresh; caller owns cleanup).
  """
  site_dir.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(LAYER_C_DIR / "index.html", site_dir / "index.html")
  shutil.copyfile(LAYER_C_DIR / "parity.mjs", site_dir / "parity.mjs")
  ort_dist_src = LAYER_C_DIR / "node_modules" / "onnxruntime-web" / "dist"
  ort_dist_dst = site_dir / "ort"
  if ort_dist_dst.exists():
    shutil.rmtree(ort_dist_dst)
  shutil.copytree(ort_dist_src, ort_dist_dst)

  models_dir = site_dir / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  for basename, src in model_files.items():
    shutil.copyfile(src, models_dir / basename)

  (site_dir / "cells.json").write_text(json.dumps({"cells": cells}, indent=2))


def run_browser(
  *,
  site_dir: Path,
  out_dir: Path,
  node_bin: str | None = None,
  num_threads: int = 1,
  isolate: bool = True,
  timeout_s: float = 120.0,
) -> dict[str, Any]:
  """Run `run_parity.mjs` (headless Chromium via Playwright) against `site_dir`.

  Returns the harness's own parsed JSON: `{harnessOk, harnessError, chromiumVersion,
  playwrightVersion, isolate, numThreadsRequested, result, consoleErrors?}`. Raises
  `RuntimeError` only on a harness-launch failure this function itself cannot recover a
  JSON result from (missing node, subprocess timeout, unparseable stdout) -- an
  `harnessOk: false` WITH a parsed JSON body is returned normally, as data (the caller
  decides how that folds into its own result-emission rule).
  """
  resolved_node = find_node(node_bin)
  if resolved_node is None:
    msg = "run_browser: no node binary found (checked --node-bin, $NODE_BIN, $PATH, nvm)"
    raise RuntimeError(msg)

  out_dir.mkdir(parents=True, exist_ok=True)
  cmd = [
    resolved_node,
    "run_parity.mjs",
    "--site",
    str(site_dir),
    "--out-dir",
    str(out_dir),
    "--num-threads",
    str(num_threads),
    "--timeout-ms",
    str(int(timeout_s * 1000)),
  ]
  if not isolate:
    cmd.append("--no-isolation")

  proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell, args are ints/paths we built
    cmd,
    cwd=LAYER_C_DIR,
    capture_output=True,
    text=True,
    timeout=timeout_s + 30.0,
    check=False,
  )
  stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
  if stdout_lines:
    try:
      return json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
      pass

  # Fall back to the file run_parity.mjs always writes before exiting, if stdout itself
  # did not parse (e.g. Playwright/node emitted extra diagnostic lines).
  harness_path = out_dir / "harness.json"
  if harness_path.is_file():
    with harness_path.open() as fh:
      return json.load(fh)

  msg = (
    f"run_browser: harness produced no parseable result (exit={proc.returncode}); "
    f"stderr tail: {proc.stderr[-2000:]!r}"
  )
  raise RuntimeError(msg)


def node_env_report(node_bin: str) -> dict[str, Any]:
  """`{node, playwright, onnxruntime-web}` version strings for the sidecar's `versions`."""
  info: dict[str, Any] = {}
  try:
    proc = subprocess.run(  # noqa: S603
      [node_bin, "--version"], capture_output=True, text=True, timeout=30, check=False
    )
    info["node"] = proc.stdout.strip() if proc.returncode == 0 else None
  except OSError:
    info["node"] = None
  for pkg in ("@playwright/test", "onnxruntime-web"):
    pkg_json = LAYER_C_DIR / "node_modules" / pkg / "package.json"
    if pkg_json.is_file():
      with pkg_json.open() as fh:
        info[pkg] = json.load(fh).get("version")
    else:
      info[pkg] = None
  return info


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  logger.info("layer_c_common: node=%s", find_node())
  sys.exit(0)
