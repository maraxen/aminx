"""X5b browser parity: the PottsMPNN browser page in REAL headless Chromium, token-exact vs JAX.

task_id 261007_potts-onnx-export, backlog #5815 (X5 part B). Sidecar: potts_browser_parity.bth.toml
(pre-registered).

The X5 knobs gate (potts_knobs_gate.py) proved the full browser path under Node, which is not a
browser: no cross-origin isolation, no browser APIs, a different module loader, and threaded wasm
is not expected to initialise from a file:// path. This gate serves the shipping page
(browser/potts-sampler/potts_index.html + potts_page.mjs) over HTTP with COOP/COEP and drives it
through browser/layer_c/run_p07.mjs UNCHANGED, the same harness the P07 split gates use, at 1 and at
4 ORT threads.

The JAX reference, per-sample grading and the cross-seed comparator control are imported from
potts_knobs_gate, so this gate grades against exactly the reference the knobs gate grades against.
Cases: default and combined. Cells: 3dkm L128, 3gg7 L256. Seeds 21 and 22. That is 8 samples per
thread setting. The page grades nothing; every comparison and threshold lives here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit
from scripts.browser_validation.layer_c_common import (
  LAYER_C_DIR,
  find_node,
  node_env_report,
)
from scripts.browser_validation.potts_knobs_gate import (
  FLOAT_REL_BAR,
  _require,
  cross_seed_hits,
  grade_sample,
  jax_reference,
)
from scripts.browser_validation.potts_loop_gate import _verify_artifacts, _write

logger = logging.getLogger("potts_browser_parity")

SAMPLER_DIR = REPO / "browser" / "potts-sampler"
RUNNER = LAYER_C_DIR / "run_p07.mjs"
PAGE_FILES = ("potts_page.mjs", "potts_inputs.mjs", "potts_controls.mjs", "potts_sampler.mjs")
CASES = frozenset({"default", "combined"})
CELLS = (("3dkm.pdb", 128), ("3gg7.pdb", 256))
SEEDS = (21, 22)
THREADS = (1, 4)
N_EXPECTED = len(CASES) * len(CELLS) * len(SEEDS)
_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")


def git_state(repo: Path) -> tuple[str, bool]:
  """HEAD sha and whether the code paths are clean (untracked files included)."""

  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS,
  )


def noise_cell(
  cid: str,
  pdb_path: Path,
  bucket: int,
  controls: dict[str, Any],
  options: dict[str, Any],
  seed: int,
  work_dir: Path,
) -> dict[str, Any]:
  """One browser cell with its noise written as float32 files, in the node/page cell format."""
  rng = np.random.default_rng(seed)
  randn = rng.standard_normal(bucket).astype(np.float32)
  uniforms = rng.uniform(size=bucket).astype(np.float32)
  refine_uniforms = rng.uniform(size=(8, bucket)).astype(np.float32)
  cdir = work_dir / cid
  cdir.mkdir(parents=True, exist_ok=True)
  return {
    "id": cid, "pdb": str(pdb_path), "bucket": bucket, "controls": controls, "options": options,
    "out_dir": str(cdir),
    "noise": {"randn": _write(randn, cdir / "randn.bin"), "uniforms": _write(uniforms, cdir / "uniforms.bin"),
              "refineUniforms": _write(refine_uniforms, cdir / "refine_uniforms.bin")},
  }


def assemble_potts_site(
  site: Path,
  models: Path,
  ort_dir: Path,
  cells: list[dict[str, Any]],
  *,
  reps: int,
  warmup: int,
  planted_ms: float,
) -> None:
  """Lay out exactly what the page fetches: the page, its modules, ORT, models, data, cells.json.

  PDB files and noise buffers are copied under `data/` and the cell entries are rewritten to
  site-relative paths; noise is renamed `<cell id>__<name>.bin` so the same basename in two
  cell directories cannot collide.
  """
  if site.exists():
    shutil.rmtree(site)
  (site / "models").mkdir(parents=True)
  (site / "ort").mkdir()
  (site / "data").mkdir()
  shutil.copy(SAMPLER_DIR / "potts_index.html", site / "index.html")
  for name in PAGE_FILES:
    shutil.copy(SAMPLER_DIR / name, site / name)

  manifest_path = models / "MANIFEST.json"
  shutil.copy(manifest_path, site / "models" / "MANIFEST.json")
  manifest = json.loads(manifest_path.read_text())
  # The page fetches `./models/<graph file>` with the manifest's own relative name, so the graph
  # files keep the manifest's names exactly.
  for graph_file in sorted({g["file"] for b in manifest["buckets"] for g in b["graphs"].values()}):
    dst = site / "models" / graph_file
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(models / graph_file, dst)

  dist = ort_dir / "node_modules" / "onnxruntime-web" / "dist"
  for f in dist.iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)

  site_cells = []
  for cell in cells:
    pdb_src = Path(cell["pdb"])
    if not (site / "data" / pdb_src.name).exists():
      shutil.copy(pdb_src, site / "data" / pdb_src.name)
    noise = {}
    for name, spec in cell["noise"].items():
      dst_name = f"{cell['id']}__{name}.bin"
      shutil.copy(spec["file"], site / "data" / dst_name)
      noise[name] = {"file": f"data/{dst_name}", "dtype": spec["dtype"], "dims": spec["dims"]}
    site_cells.append({
      "id": cell["id"], "pdb": f"data/{pdb_src.name}", "bucket": cell["bucket"],
      "controls": cell["controls"], "options": cell["options"], "noise": noise,
    })
  (site / "cells.json").write_text(json.dumps({
    "cells": site_cells, "reps": reps, "warmup": warmup, "planted_ms": planted_ms,
  }))


def run_harness(
  node: str, site: Path, out_dir: Path, threads: int, timeout_ms: int,
) -> dict[str, Any]:
  """Run browser/layer_c/run_p07.mjs against `site` and return its parsed harness.json.

  Raises only when no harness.json was written at all; a page error comes back as harnessOk false.
  """
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  cmd = [
    node, str(RUNNER), "--site", str(site), "--out-dir", str(out_dir),
    "--num-threads", str(threads), "--timeout-ms", str(timeout_ms),
  ]
  with (out_dir / "run_p07.log").open("w") as log:
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      cmd, stdout=log, stderr=subprocess.STDOUT, check=False,
      timeout=timeout_ms // 1000 + 600,
    )
  harness_path = out_dir / "harness.json"
  if not harness_path.is_file():
    msg = f"browser harness wrote no harness.json (rc={proc.returncode})"
    raise RuntimeError(msg)
  return json.loads(harness_path.read_text())


def browser_samples(payload: dict[str, Any], out_dir: Path) -> dict[str, dict[str, Any]]:
  """Turn the page's per-cell records into the `{ok, error, outputs}` shape the knobs gate grades."""
  got: dict[str, dict[str, Any]] = {}
  for cell in payload.get("cells", []):
    sid = cell["id"]
    if not cell.get("ok"):
      got[sid] = {"ok": False, "error": cell.get("error"), "outputs": {}}
      continue
    if cell.get("sample_energy") is None:
      got[sid] = {"ok": False, "error": "sample_energy missing or non-finite", "outputs": {}}
      continue
    outs: dict[str, Any] = {
      name: np.fromfile(out_dir / spec["file"], dtype=np.int32) for name, spec in cell["outputs"].items()
    }
    outs["sample_energy"] = float(cell["sample_energy"])
    got[sid] = {"ok": True, "error": None, "outputs": outs}
  return got


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True)
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))),
                      help="directory whose node_modules/onnxruntime-web/dist supplies the ORT files")
  parser.add_argument("--timeout-ms", type=int, default=1_800_000)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean,
    "manifest_sha256": "", "artifacts_verified": False, "jax_arm_ok": False,
    "jax_version": "", "onnxruntime_web_version": "", "playwright_version": "", "node_version": "",
    "n_expected_per_thread": N_EXPECTED, "threads_requested": list(THREADS), "runs": [],
    "browser_complete": False, "isolated_at_4": False, "control_tokens_detected": False,
    "all_exact": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415

    result["jax_version"] = str(jax.__version__)
    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True
    result["manifest_sha256"] = args.manifest_sha256
    refs, node_cells = jax_reference(
      args.potts_root, args.work_dir / "jax", cells=CELLS, seeds=SEEDS, case_names=CASES,
    )
    _require(len(refs) == N_EXPECTED, f"JAX arm built {len(refs)} samples, expected {N_EXPECTED}")
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d samples", len(refs))

    node = find_node(args.node_bin)
    _require(node is not None, "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)")
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    site = args.work_dir / "site"
    assemble_potts_site(site, args.models, args.ort_dir, node_cells, reps=1, warmup=0, planted_ms=0)
    logger.info("assembled site at %s", site)

    for threads in THREADS:
      run: dict[str, Any] = {"threads_requested": threads, "harness_ok": False, "harness_error": ""}
      result["runs"].append(run)
      try:
        harness = run_harness(node, site, args.work_dir / f"browser_t{threads}", threads, args.timeout_ms)
        run["chromium_version"] = str(harness.get("chromiumVersion") or "")
        if not harness.get("harnessOk"):
          run["harness_error"] = str(harness.get("harnessError"))[:800]
          continue
        payload = harness["result"]
        run["harness_ok"] = True
        run["cross_origin_isolated"] = bool(payload.get("cross_origin_isolated"))
        run["num_threads_after_init"] = int(payload.get("num_threads_after_init", -1))
        run["num_threads_effective"] = int(payload.get("num_threads_effective", -1))
        got = browser_samples(payload, args.work_dir / f"browser_t{threads}")
        rows = [grade_sample(sid, ref, got.get(sid, {"ok": False, "error": "missing", "outputs": {}}))
                for sid, ref in refs.items()]
        run["samples"] = rows
        run["n_samples"] = len(rows)
        run["n_pass"] = sum(bool(r.get("passed")) for r in rows)
        hits, n = cross_seed_hits(refs, got)
        run["control_hits"] = hits
        run["control_n"] = n
        run["control_tokens_detected"] = n > 0 and hits == n
        logger.info(
          "threads %d (effective %d, isolated %s): exact %d/%d, control %d/%d",
          threads, run["num_threads_effective"], run["cross_origin_isolated"],
          run["n_pass"], run["n_samples"], hits, n,
        )
      except Exception as exc:
        run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
        logger.exception("threads %d run failed", threads)

    runs = result["runs"]
    result["browser_complete"] = len(runs) == len(THREADS) and all(
      r["harness_ok"] and r.get("n_samples") == N_EXPECTED for r in runs
    )
    by_threads = {r["threads_requested"]: r for r in runs if r["harness_ok"]}
    result["isolated_at_4"] = bool(by_threads.get(4, {}).get("cross_origin_isolated", False))
    result["control_tokens_detected"] = bool(runs) and all(r.get("control_tokens_detected") for r in runs)
    result["all_exact"] = bool(runs) and all(
      r.get("n_samples", 0) > 0 and r.get("n_pass") == r.get("n_samples") for r in runs
    )
  except Exception as exc:
    logger.exception("browser parity gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]

  # First matching branch wins (same order as the sidecar's [outcomes] declaration).
  if not (result["jax_arm_ok"] and result["artifacts_verified"] and result["browser_complete"]):
    result["outcome"] = "error"
  elif not result["isolated_at_4"]:
    result["outcome"] = "not_isolated"
  elif not result["control_tokens_detected"]:
    result["outcome"] = "instrument_unverified"
  elif not result["all_exact"]:
    result["outcome"] = "fail"
  else:
    result["outcome"] = "pass"
  result["float_rel_bar"] = FLOAT_REL_BAR
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("browser parity done: complete=%s isolated@4=%s control=%s exact=%s",
              result["browser_complete"], result["isolated_at_4"],
              result["control_tokens_detected"], result["all_exact"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
