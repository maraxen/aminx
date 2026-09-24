"""Phase 0, T3: ORT Web (wasm EP) headless-Chromium smoke (AC-3).

Precondition: every artifact T2 (``jax2onnx_spike.py``) recorded in
``outputs/browser_validation/phase0/jax2onnx_spike.json['artifacts']`` must
exist on disk with a matching sha256 -- a mismatch or a missing file means the
artifact is stale or tampered and this script refuses to build browser
evidence from it (exit 3). T2's own conversion outcome (``converted``) is NOT
a precondition failure: T2 recorded ``converted = false`` (jax2onnx 0.16.1 has
no plugin for the ``random_split`` primitive that
``score_conditional.kernel`` calls unconditionally), so ``p05_L128.onnx`` and
``p05_L128_perturbed.onnx`` were never produced and are correctly absent from
T2's artifact manifest -- their absence is expected, not a mismatch.

Independently of whether P05 converted, this script still runs the ONE
Phase-0 artifact that DID convert -- ``topk_lattice.onnx`` -- through
ONNX Runtime Web's wasm execution provider inside real headless Chromium
(Playwright), with ``numThreads = 1``, and compares the resulting top_k
neighbor indices against the JAX baseline recorded in
``topk_lattice_io.npz``. This is Phase 0's one piece of genuine in-browser
evidence: T2's ORT Python CPU EP is explicitly NOT ORT Web.

When P05 is unavailable (T2 ``converted = false``), the result is written
with ``status = "BLOCKED"`` and a ``reason`` naming T2's verbatim conversion
error -- AC-3's "BLOCKED produces no compatibility statement" -- but the
browser-environment facts that the smoke DID produce (browser/ORT-Web
versions, ``crossOriginIsolated``, thread count, and
``tie_indices_identical_browser``) are still recorded rather than discarded,
since step 3/4 of this task record that field, they do not gate it. The same
``status = "BLOCKED"`` is used, for a different reason, when the
node/Playwright-Chromium toolchain itself is unavailable and cannot be
installed -- in that case no browser facts exist to record.

The browser-side harness (``browser/smoke/``) is written generically: model
manifest entries are matched to ONNX graph inputs/outputs BY POSITION
(``session.inputNames``/``session.outputNames``, in jax2onnx's own
argument-order-preserving naming), not by a hardcoded ONNX tensor name, so
the same harness runs the P05 kernel and its perturbed-weight control twin
unchanged once a converted P05 onnx exists (Phase 3) -- this script already
assembles their manifest entries whenever the ``.onnx`` files are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE0_DIR = Path("outputs/browser_validation/phase0")
T2_RESULT_NAME = "jax2onnx_spike.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "browser" / "smoke"
RUN_SMOKE_JS = "run_smoke.mjs"

ORT_MAX_ABS_BAR = 1e-4
CONTROL_MIN_MAX_ABS = 2e-4  # matches jax2onnx_spike.py's CONTROL_MIN_MAX_ABS
NODE_TIMEOUT_S = 120

P05_INPUT_ORDER = ["coords", "mask", "residue_index", "chain_index", "sequence"]

DTYPE_TO_ORT = {
  np.dtype("float32"): "float32",
  np.dtype("int32"): "int32",
  np.dtype("int8"): "int8",
  np.dtype("int64"): "int64",
  np.dtype("uint8"): "uint8",
  np.dtype("bool"): "bool",
}


def _sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
  result: dict[str, Any] = {
    "status": "BLOCKED",
    "reason": reason,
    "browser": None,
    "browser_version": None,
    "ort_web_version": None,
    "cross_origin_isolated": False,
    "threads": None,
    "execution_provider": "wasm",
    "node": None,
    "max_abs": None,
    "idx_exact": False,
    "control_detected": False,
    "tie_indices_identical_browser": False,
    "versions": {},
    "models": {},
  }
  result.update(extra)
  return result


def load_t2_result(phase0_dir: Path) -> dict[str, Any] | None:
  path = phase0_dir / T2_RESULT_NAME
  if not path.is_file():
    return None
  return json.loads(path.read_text())


def verify_t2_artifacts(t2: dict[str, Any], phase0_dir: Path) -> None:
  """Exit 3 if any T2-recorded artifact is missing or sha256-mismatched.

  Absent artifacts that T2 never recorded (e.g. the P05 .onnx files, when T2
  failed to convert) are not checked here -- there is nothing recorded to
  verify against, and their absence is expected (see module docstring).
  """
  artifacts = t2.get("artifacts", {})
  if not artifacts:
    logger.warning("T2 result recorded no artifacts at all")
    return
  for name, info in artifacts.items():
    path = phase0_dir / name
    if not path.is_file():
      logger.error("T2 artifact missing: %s", path)
      sys.exit(3)
    actual = _sha256_file(path)
    expected = info.get("sha256")
    if actual != expected:
      logger.error(
        "T2 artifact sha256 mismatch for %s: recorded=%s actual=%s", name, expected, actual
      )
      sys.exit(3)
  logger.info("verified %d T2 artifact(s) against recorded sha256", len(artifacts))


def find_node(explicit: str | None) -> str | None:
  """Locate a node binary: explicit arg, then $NODE_BIN, then $PATH, then nvm dirs."""
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


def playwright_browsers_path(smoke_dir: Path) -> Path:
  """Where Playwright's Chromium download lives -- kept inside the repo, gitignored.

  `~/.cache/ms-playwright` (Playwright's own default) is not writable under
  this project's sandbox policy; pinning `PLAYWRIGHT_BROWSERS_PATH` here
  keeps the install self-contained and reproducible regardless of ambient
  environment.
  """
  override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
  if override:
    return Path(override)
  return smoke_dir / ".cache" / "ms-playwright"


def node_env(smoke_dir: Path) -> dict[str, str]:
  env = dict(os.environ)
  env["PLAYWRIGHT_BROWSERS_PATH"] = str(playwright_browsers_path(smoke_dir))
  return env


def ensure_toolchain(node_bin: str, smoke_dir: Path) -> tuple[bool, str | None, dict[str, Any]]:
  """Verify node_modules + a runnable Chromium exist; best-effort `npm install`/install if not.

  Returns (ok, reason_if_not_ok, versions_dict).
  """
  versions: dict[str, Any] = {}
  node_version = subprocess.run(
    [node_bin, "--version"], capture_output=True, text=True, timeout=30, check=False
  )
  if node_version.returncode != 0:
    return False, f"node --version failed: {node_version.stderr.strip()}", versions
  versions["node"] = node_version.stdout.strip()

  node_modules = smoke_dir / "node_modules"
  if (
    not (node_modules / "@playwright" / "test").is_dir()
    or not (node_modules / "onnxruntime-web").is_dir()
  ):
    logger.info("browser/smoke/node_modules incomplete; attempting `npm install`")
    npm_bin = str(Path(node_bin).parent / "npm") if Path(node_bin).parent.name else "npm"
    if not Path(npm_bin).is_file():
      npm_bin = shutil.which("npm") or "npm"
    install = subprocess.run(
      [npm_bin, "install", "--no-audit", "--no-fund"],
      cwd=smoke_dir,
      capture_output=True,
      text=True,
      timeout=300,
      env=node_env(smoke_dir),
      check=False,
    )
    if install.returncode != 0 or not (node_modules / "@playwright" / "test").is_dir():
      tail = "\n".join(install.stdout.splitlines()[-20:] + install.stderr.splitlines()[-20:])
      return False, f"npm install failed (exit {install.returncode}): {tail}", versions

  pkg_playwright = json.loads((node_modules / "@playwright" / "test" / "package.json").read_text())
  pkg_ort = json.loads((node_modules / "onnxruntime-web" / "package.json").read_text())
  versions["playwright"] = pkg_playwright.get("version")
  versions["onnxruntime-web"] = pkg_ort.get("version")

  probe = subprocess.run(
    [
      node_bin,
      "-e",
      "console.log(require('@playwright/test').chromium.executablePath())",
    ],
    cwd=smoke_dir,
    capture_output=True,
    text=True,
    timeout=30,
    env=node_env(smoke_dir),
    check=False,
  )
  if probe.returncode != 0:
    return False, f"could not resolve Chromium executable path: {probe.stderr.strip()}", versions
  chromium_path = Path(probe.stdout.strip())
  if not chromium_path.is_file():
    logger.info(
      "Chromium not installed at %s; attempting `npx playwright install chromium`", chromium_path
    )
    npx_bin = str(Path(node_bin).parent / "npx") if Path(node_bin).parent.name else "npx"
    if not Path(npx_bin).is_file():
      npx_bin = shutil.which("npx") or "npx"
    install_browser = subprocess.run(
      [npx_bin, "playwright", "install", "chromium"],
      cwd=smoke_dir,
      capture_output=True,
      text=True,
      timeout=600,
      env=node_env(smoke_dir),
      check=False,
    )
    if install_browser.returncode != 0 or not chromium_path.is_file():
      tail = "\n".join(
        install_browser.stdout.splitlines()[-20:] + install_browser.stderr.splitlines()[-20:]
      )
      return (
        False,
        f"npx playwright install chromium failed (exit {install_browser.returncode}): {tail}",
        versions,
      )
  return True, None, versions


def _write_tensor_json(arr: np.ndarray, out_path: Path) -> None:
  ort_dtype = DTYPE_TO_ORT.get(arr.dtype)
  if ort_dtype is None:
    msg = f"unsupported dtype for browser tensor: {arr.dtype}"
    raise ValueError(msg)
  payload = {"dims": list(arr.shape), "dtype": ort_dtype, "data": arr.reshape(-1).tolist()}
  out_path.write_text(json.dumps(payload))


def build_topk_lattice_entry(phase0_dir: Path, site_dir: Path) -> dict[str, Any] | None:
  onnx_src = phase0_dir / "topk_lattice.onnx"
  io_src = phase0_dir / "topk_lattice_io.npz"
  if not onnx_src.is_file() or not io_src.is_file():
    return None
  models_dir = site_dir / "models"
  data_dir = site_dir / "data"
  models_dir.mkdir(parents=True, exist_ok=True)
  data_dir.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(onnx_src, models_dir / "topk_lattice.onnx")
  io = np.load(io_src)
  _write_tensor_json(np.asarray(io["x"]), data_dir / "topk_lattice_x.json")
  return {
    "name": "topk_lattice",
    "onnx": "models/topk_lattice.onnx",
    "inputs": [{"name": "x", "dataFile": "data/topk_lattice_x.json"}],
    "outputOrder": ["values", "indices"],
  }


def build_p05_entry(
  name: str, onnx_filename: str, phase0_dir: Path, site_dir: Path
) -> dict[str, Any] | None:
  """Manifest entry for a P05 (or perturbed-control) model, IF its onnx exists.

  Not exercised while T2's `converted = false` (no such file exists yet) --
  kept generic so Phase 3's converted P05 runs through this same harness
  unchanged.
  """
  onnx_src = phase0_dir / onnx_filename
  inputs_src = phase0_dir / "p05_inputs.npz"
  if not onnx_src.is_file() or not inputs_src.is_file():
    return None
  models_dir = site_dir / "models"
  data_dir = site_dir / "data"
  models_dir.mkdir(parents=True, exist_ok=True)
  data_dir.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(onnx_src, models_dir / onnx_filename)
  npz = np.load(inputs_src)
  inputs = []
  for key in P05_INPUT_ORDER:
    data_path = data_dir / f"{name}_{key}.json"
    _write_tensor_json(np.asarray(npz[key]), data_path)
    inputs.append({"name": key, "dataFile": f"data/{name}_{key}.json"})
  return {
    "name": name,
    "onnx": f"models/{onnx_filename}",
    "inputs": inputs,
    "outputOrder": ["logits", "neighbor_indices"],
  }


def assemble_site(phase0_dir: Path, smoke_dir: Path, site_dir: Path) -> dict[str, Any]:
  """Copy the static harness + onnxruntime-web dist + model/data files into `site_dir`."""
  shutil.copyfile(smoke_dir / "index.html", site_dir / "index.html")
  shutil.copyfile(smoke_dir / "smoke.mjs", site_dir / "smoke.mjs")
  ort_dist_src = smoke_dir / "node_modules" / "onnxruntime-web" / "dist"
  shutil.copytree(ort_dist_src, site_dir / "ort")

  models: list[dict[str, Any]] = []
  topk_entry = build_topk_lattice_entry(phase0_dir, site_dir)
  if topk_entry is not None:
    models.append(topk_entry)
  p05_entry = build_p05_entry("p05_L128", "p05_L128.onnx", phase0_dir, site_dir)
  if p05_entry is not None:
    models.append(p05_entry)
  p05_perturbed_entry = build_p05_entry(
    "p05_L128_perturbed", "p05_L128_perturbed.onnx", phase0_dir, site_dir
  )
  if p05_perturbed_entry is not None:
    models.append(p05_perturbed_entry)

  manifest = {"models": models}
  (site_dir / "data").mkdir(parents=True, exist_ok=True)
  (site_dir / "data" / "manifest.json").write_text(json.dumps(manifest))
  return manifest


def run_node_smoke(
  node_bin: str, smoke_dir: Path, site_dir: Path, out_path: Path
) -> dict[str, Any]:
  """Invoke `run_smoke.mjs`; returns its harness JSON (raises on a harness process crash)."""
  proc = subprocess.run(
    [node_bin, RUN_SMOKE_JS, "--site", str(site_dir), "--out", str(out_path)],
    cwd=smoke_dir,
    capture_output=True,
    text=True,
    timeout=NODE_TIMEOUT_S,
    env=node_env(smoke_dir),
    check=False,
  )
  if out_path.is_file():
    harness = json.loads(out_path.read_text())
  else:
    harness = {
      "harnessOk": False,
      "harnessError": (
        f"run_smoke.mjs exited {proc.returncode} without writing {out_path}: "
        f"stdout={proc.stdout[-2000:]!r} stderr={proc.stderr[-2000:]!r}"
      ),
      "result": None,
    }
  logger.info("run_smoke.mjs: exit=%s harnessOk=%s", proc.returncode, harness.get("harnessOk"))
  return harness


def compare_topk_lattice(
  model_out: dict[str, Any], phase0_dir: Path
) -> tuple[bool, dict[str, Any]]:
  io = np.load(phase0_dir / "topk_lattice_io.npz")
  jax_indices = np.asarray(io["jax_indices"])
  if not model_out.get("ok"):
    return False, {"error": model_out.get("error")}
  outputs = model_out["outputs"]
  browser_indices = np.array(outputs["indices"]["data"]).reshape(outputs["indices"]["dims"])
  identical = bool(np.array_equal(browser_indices, jax_indices))
  return identical, {"browser_dims": outputs["indices"]["dims"]}


def compare_p05(
  model_out: dict[str, Any] | None, phase0_dir: Path
) -> tuple[float | None, bool, dict[str, Any]]:
  if model_out is None:
    return None, False, {"note": "P05 model not present in this run's manifest"}
  jax_outputs = np.load(phase0_dir / "p05_jax_outputs.npz")
  jax_logits = np.asarray(jax_outputs["logits"])
  jax_neighbor_indices = np.asarray(jax_outputs["neighbor_indices"])
  if not model_out.get("ok"):
    return None, False, {"error": model_out.get("error")}
  outputs = model_out["outputs"]
  browser_logits = np.array(outputs["logits"]["data"]).reshape(outputs["logits"]["dims"])
  browser_neighbor_indices = np.array(outputs["neighbor_indices"]["data"]).reshape(
    outputs["neighbor_indices"]["dims"]
  )
  max_abs = float(np.max(np.abs(browser_logits - jax_logits.reshape(browser_logits.shape))))
  idx_exact = bool(
    np.array_equal(browser_neighbor_indices.reshape(-1), jax_neighbor_indices.reshape(-1))
  )
  return max_abs, idx_exact, {}


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump(result, f, indent=2, default=str)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    with Path(results_path).open("w") as f:
      json.dump(result, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--node-bin", default=None, help="Explicit node binary (else $NODE_BIN/$PATH/nvm)."
  )
  args = parser.parse_args(argv)

  phase0_dir = PHASE0_DIR
  t2 = load_t2_result(phase0_dir)
  if t2 is None:
    logger.error("T2 result missing: %s", phase0_dir / T2_RESULT_NAME)
    sys.exit(3)
  verify_t2_artifacts(t2, phase0_dir)

  node_bin = find_node(args.node_bin)
  if node_bin is None:
    result = _blocked("node binary not found (checked --node-bin, $NODE_BIN, $PATH, nvm dirs)")
    _write_result(args.out, result)
    logger.info("BLOCKED: no node binary")
    return 0

  ok, reason, toolchain_versions = ensure_toolchain(node_bin, SMOKE_DIR)
  if not ok:
    result = _blocked(reason or "toolchain unavailable", node=node_bin, versions=toolchain_versions)
    _write_result(args.out, result)
    logger.info("BLOCKED: %s", reason)
    return 0

  with tempfile.TemporaryDirectory(prefix="aminx-bv-smoke-") as tmp:
    site_dir = Path(tmp) / "site"
    site_dir.mkdir(parents=True)
    manifest = assemble_site(phase0_dir, SMOKE_DIR, site_dir)
    harness_out = Path(tmp) / "harness_result.json"
    harness = run_node_smoke(node_bin, SMOKE_DIR, site_dir, harness_out)

  if not harness.get("harnessOk"):
    result = _blocked(
      f"headless-Chromium harness failed to run: {harness.get('harnessError')}",
      node=node_bin,
      browser_version=harness.get("chromiumVersion"),
      versions={**toolchain_versions, "playwright_reported": harness.get("playwrightVersion")},
    )
    _write_result(args.out, result)
    logger.info("BLOCKED: harness failure")
    return 0

  browser_result = harness["result"]
  models_out = browser_result.get("models", {})

  tie_identical, tie_detail = compare_topk_lattice(models_out.get("topk_lattice", {}), phase0_dir)

  p05_max_abs, p05_idx_exact, p05_detail = compare_p05(models_out.get("p05_L128"), phase0_dir)
  control_max_abs, control_detected, control_detail = compare_p05(
    models_out.get("p05_L128_perturbed"), phase0_dir
  )
  # The control is a DIFFERENCE check (perturbed vs UNPERTURBED jax baseline),
  # not a parity check -- reuse compare_p05's max_abs against the same
  # unperturbed jax_logits, then threshold it like jax2onnx_spike.py does.
  if control_max_abs is not None:
    control_detected = control_max_abs >= CONTROL_MIN_MAX_ABS
  else:
    control_detected = False

  t2_converted = bool(t2.get("converted", False))
  if not t2_converted:
    status = "BLOCKED"
    reason = f"T2 not_converted: {t2.get('conversion_error', 'unknown conversion error')}"
  else:
    status = "OK"
    reason = None

  result: dict[str, Any] = {
    "status": status,
    "reason": reason,
    "browser": "chromium",
    "browser_version": harness.get("chromiumVersion"),
    "ort_web_version": (
      browser_result.get("ortWebVersion") or toolchain_versions.get("onnxruntime-web")
    ),
    "cross_origin_isolated": bool(browser_result.get("crossOriginIsolated")),
    "threads": browser_result.get("numThreads"),
    "execution_provider": browser_result.get("executionProvider", "wasm"),
    "node": node_bin,
    "max_abs": p05_max_abs,
    "idx_exact": p05_idx_exact,
    "control_detected": control_detected,
    "control_max_abs": control_max_abs,
    "tie_indices_identical_browser": tie_identical,
    "versions": {
      **toolchain_versions,
      "playwright_reported": harness.get("playwrightVersion"),
      "ort_versions_from_browser": browser_result.get("ortVersions"),
      "user_agent": browser_result.get("userAgent"),
    },
    "models": {
      "topk_lattice": {"ok": models_out.get("topk_lattice", {}).get("ok"), **tie_detail},
      "p05_L128": {"present": "p05_L128" in models_out, **p05_detail},
      "p05_L128_perturbed": {"present": "p05_L128_perturbed" in models_out, **control_detail},
    },
    "t2": {
      "converted": t2_converted,
      "conversion_error": t2.get("conversion_error"),
      "manifest_models": [m["name"] for m in manifest["models"]],
    },
  }
  if harness.get("consoleErrors"):
    result["browser_console_errors"] = harness["consoleErrors"]

  _write_result(args.out, result)
  logger.info(
    "ort_web_smoke complete: status=%s max_abs=%s idx_exact=%s control_detected=%s "
    "tie_indices_identical_browser=%s",
    result["status"],
    result["max_abs"],
    result["idx_exact"],
    result["control_detected"],
    result["tie_indices_identical_browser"],
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
