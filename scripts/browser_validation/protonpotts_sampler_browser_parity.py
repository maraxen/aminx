"""X8c browser parity: the ProtonPotts sampler page in REAL headless Chromium vs the unpadded eager decoder, at 1 and 4 threads.

task_id 261009_protonpotts-onnx, backlog #5816 (B2). Sidecar: protonpotts_sampler_browser_parity.bth.toml (pre-registered before this
script existed). Reference arm, cases, streams and comparator are protonpotts_sampler_gate's (X8b); the page is served over HTTP with
COOP/COEP and driven through browser/layer_c/run_p07.mjs UNCHANGED. The page grades nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit  # noqa: E402
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report  # noqa: E402
from scripts.browser_validation.potts_browser_parity import git_state, run_harness  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _sha256  # noqa: E402
from scripts.browser_validation.protonpotts_sampler_gate import LOG_BAR, SEEDS, V, _rel, build_reference, node_case  # noqa: E402

logger = logging.getLogger("protonpotts_sampler_browser_parity")
SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
PAGE_FILES = ("sampler_page.mjs", "decoder_sampler.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs")
THREADS = (1, 4)


def assemble_site(site: Path, models: Path, ort_dir: Path, cells: list[dict[str, Any]], pdbs: dict[str, Path]) -> None:
  if site.exists():
    shutil.rmtree(site)
  for sub in ("models", "ort", "data"):
    (site / sub).mkdir(parents=True)
  shutil.copy(SCORER_DIR / "sampler_index.html", site / "index.html")
  for name in PAGE_FILES:
    shutil.copy(SCORER_DIR / name, site / name)
  manifest = json.loads((models / "MANIFEST.json").read_text())
  shutil.copy(models / "MANIFEST.json", site / "models" / "MANIFEST.json")
  for graph_file in sorted({g["file"] for b in manifest["buckets"] for g in b["graphs"].values()}):
    shutil.copy(models / graph_file, site / "models" / graph_file)
  for f in (ort_dir / "node_modules" / "onnxruntime-web" / "dist").iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)
  for src in pdbs.values():
    shutil.copy(src, site / "data" / src.name)
  out = []
  for cell in cells:
    pdb = Path(cell.pop("pdb_file"))
    out.append(cell | {"pdb": f"data/{pdb.name}"})
  (site / "cells.json").write_text(json.dumps({"cells": out}))


def grade(cases: list[dict[str, Any]], got: dict[str, Any]) -> dict[str, Any]:
  rows = []
  for c in cases:
    g = got.get(c["id"])
    if not g or not g.get("ok"):
      rows.append({"id": c["id"], "ran": False, "passed": False, "error": ((g or {}).get("error") or "missing")[:1500]})
      continue
    n = c["length"]
    ref_seq, ref_order, ref_lp = c["ref"]
    seq_eq = bool(np.array_equal(np.asarray(g["sequence"]), ref_seq[:n]))
    order_eq = bool(np.array_equal(np.asarray(g["decoding_order"]), ref_order[:n]))
    lp = _rel(ref_lp[:n], np.asarray(g["log_probs"]).reshape(n, V))
    rows.append({"id": c["id"], "ran": True, "fragile": bool(c["fragile"]), "sequence_exact": seq_eq, "order_exact": order_eq,
                 "log_probs_rel": lp, "passed": bool((seq_eq or c["fragile"]) and order_eq and lp <= LOG_BAR)})
  bias_cases = [c for c in cases if c["config"] == "bias_temp"]
  tokens = bool(bias_cases) and all(
    (g := got.get(c["id"] + "__rolled")) and g.get("ok") and not np.array_equal(np.asarray(g["sequence"]), c["ref"][0][: c["length"]])
    for c in bias_cases)
  floats = all((g := got.get(c["id"])) and g.get("ok")
               and _rel(c["bad_log_probs"][: c["length"]], np.asarray(g["log_probs"]).reshape(c["length"], V)) > LOG_BAR
               for c in bias_cases)
  repeat, distinct = [], []
  for name in STRUCTURES:
    a, b, d = (got.get(f"{name}__seed{s}__{k}") for s, k in ((SEEDS[0], 0), (SEEDS[0], 1), (SEEDS[1], 0)))
    ok = all(x and x.get("ok") for x in (a, b, d))
    repeat.append(bool(ok and a["sequence"] == b["sequence"] and a["decoding_order"] == b["decoding_order"] and a["log_probs"] == b["log_probs"]))
    distinct.append(bool(ok and a["sequence"] != d["sequence"]))
  return {"cases": rows, "n_cases": len(rows), "n_pass": sum(bool(r["passed"]) for r in rows),
          "control_tokens_detected": tokens, "control_float_detected": bool(floats),
          "seeded_ok": bool(all(repeat) and all(distinct))}


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True)
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path,
                      default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))))
  parser.add_argument("--timeout-ms", type=int, default=3_600_000)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean, "manifest_sha256": args.manifest_sha256, "artifacts_verified": False,
    "jax_arm_ok": False, "jax_version": "", "onnxruntime_web_version": "", "playwright_version": "", "node_version": "",
    "n_expected_per_thread": 0, "threads_requested": list(THREADS), "runs": [], "browser_complete": False, "isolated_at_4": False,
    "control_tokens_detected": False, "control_float_detected": False, "all_exact": False, "seeded_ok": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415

    result["jax_version"] = str(jax.__version__)
    if _sha256(args.models / "MANIFEST.json") != args.manifest_sha256:
      msg = "decoder MANIFEST.json sha256 does not match --manifest-sha256"
      raise RuntimeError(msg)  # noqa: TRY301
    manifest = json.loads((args.models / "MANIFEST.json").read_text())
    for b in manifest["buckets"]:
      for g in b["graphs"].values():
        if _sha256(args.models / g["file"]) != g["sha256"]:
          msg = f"{g['file']} sha256 does not match the manifest"
          raise RuntimeError(msg)  # noqa: TRY301
    result["artifacts_verified"] = True
    cases = build_reference(args.potts_root, args.state_npz)
    result["jax_arm_ok"] = True
    pdbs = {name: args.potts_root / "energy_benchmark_datasets" / rel for name, (rel, _s) in STRUCTURES.items()}
    page_cells = [node_case(c, pdbs[c["structure"]]) for c in cases]
    page_cells += [node_case(c, pdbs[c["structure"]], roll=True) for c in cases if c["config"] == "bias_temp"]
    for name in pdbs:
      bucket = next(c["bucket"] for c in cases if c["structure"] == name)
      for seed, k in ((SEEDS[0], 0), (SEEDS[0], 1), (SEEDS[1], 0)):
        page_cells.append({"id": f"{name}__seed{seed}__{k}", "bucket": bucket, "pdb_file": str(pdbs[name]), "designed": None,
                           "temperature": float(np.float32(0.1)), "bias": None, "uniforms": None, "noise": None, "seed": seed})
    result["n_expected_per_thread"] = len(page_cells)
    logger.info("reference arm: %d reference cases, %d page cells in %.1fs", len(cases), len(page_cells), time.perf_counter() - t0)

    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    site = args.work_dir / "site"
    assemble_site(site, args.models, args.ort_dir, [dict(c) for c in page_cells], pdbs)
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
        got = {c["id"]: c for c in payload.get("cells", [])}
        run["n_page_cells"] = sum(1 for c in got.values() if c.get("ok"))
        run.update(grade(cases, got))
        logger.info("threads %d (effective %d, isolated %s): %d/%d exact; controls tokens=%s float=%s; seeded=%s", threads,
                    run["num_threads_effective"], run["cross_origin_isolated"], run["n_pass"], run["n_cases"],
                    run["control_tokens_detected"], run["control_float_detected"], run["seeded_ok"])
      except Exception as exc:  # noqa: BLE001
        run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
        logger.exception("threads %d run failed", threads)

    runs = result["runs"]
    result["browser_complete"] = len(runs) == len(THREADS) and all(
      r["harness_ok"] and r.get("n_page_cells") == result["n_expected_per_thread"] for r in runs)
    by_threads = {r["threads_requested"]: r for r in runs if r["harness_ok"]}
    result["isolated_at_4"] = bool(by_threads.get(4, {}).get("cross_origin_isolated", False))
    result["control_tokens_detected"] = bool(runs) and all(r.get("control_tokens_detected") for r in runs)
    result["control_float_detected"] = bool(runs) and all(r.get("control_float_detected") for r in runs)
    result["all_exact"] = bool(runs) and all(r.get("n_cases", 0) > 0 and r.get("n_pass") == r.get("n_cases") for r in runs)
    result["seeded_ok"] = bool(runs) and all(r.get("seeded_ok") for r in runs)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("sampler browser parity gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("done: complete=%s isolated@4=%s controls=%s/%s exact=%s seeded=%s", result["browser_complete"], result["isolated_at_4"],
              result["control_tokens_detected"], result["control_float_detected"], result["all_exact"], result["seeded_ok"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
