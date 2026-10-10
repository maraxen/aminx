"""X9 browser bench: per-call wall time of the three ProtonPotts pages in REAL headless Chromium, with a browser timer control.

task_id 261009_protonpotts-onnx, backlog #2625 (item 2). Sidecar: protonpotts_browser_bench.bth.toml (pre-registered, committed before this script).

Arms (cells fixed here, before any run; all four graded structures 1EL1, 1BVC, 1OLR, 6m0j):
  scoring  : the four `__main` cells of the scoring parity gate (PDB text -> inputs -> table -> chunked energy graph).
  ph       : cases hpg_t0 and h_t005 on each structure (PhDesigner.design, samples_per_site=2, as the X7c gate).
  sampler  : seed 21, temperature float32(0.1), no bias, no designed mask (ProtonPottsSampler.sample, 30-token decode loop).
Each cell runs one warm-up and `reps` measured repetitions at 1 and at 4 ORT threads. A planted delay is timed once per page load through the
same wrapper as the real calls; nothing is publishable unless the page sees it. NO performance direction is pre-registered: the graded
pass/fail is measurement validity, the timings are result fields. This bench grades no output CORRECTNESS (X7c/X8c/X6c do), but it does require
every repetition to have returned a non-empty result so a fast-failing page cannot post a flattering time.

Site assembly comes from the parity gates (byte-identical sites); only cells.json is patched with reps/warmup/planted_ms.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation import protonpotts_browser_parity as scoring_gate  # noqa: E402
from scripts.browser_validation import protonpotts_ph_browser_parity as ph_gate  # noqa: E402
from scripts.browser_validation import protonpotts_sampler_browser_parity as sampler_gate  # noqa: E402
from scripts.browser_validation.layer_a_common import emit  # noqa: E402
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report  # noqa: E402
from scripts.browser_validation.potts_browser_parity import git_state, run_harness  # noqa: E402
from scripts.browser_validation.protonpotts_dump_inputs import auto_labels  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for  # noqa: E402
from scripts.browser_validation.protonpotts_ph_loop_gate import BLOCK_ROUNDS, CASES, MAX_MUTATIONS, SAMPLES, SEED as PH_SEED  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import CELLS as SCORING_CELLS  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import SEED as SCORE_SEED  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import _variants, _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_browser_bench")

THREADS = (1, 4)
BENCH_SEED = 21
PH_CASES = ("hpg_t0", "h_t005")
CTRL_LO = 0.9
CTRL_SLACK_MS = 2000.0
ARMS = ("scoring", "ph", "sampler")


def build_cells(potts_root: Path) -> dict[str, list[dict[str, Any]]]:
  """The fixed cell set per arm, with no reference arm (nothing here needs the model)."""
  from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues, loaded_residues  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415

  rng = np.random.default_rng(SCORE_SEED)
  ph_rng = np.random.default_rng(PH_SEED)
  scoring: list[dict[str, Any]] = []
  ph: list[dict[str, Any]] = []
  sampler: list[dict[str, Any]] = []
  for name, labelled in SCORING_CELLS.items():
    rel, _sha = STRUCTURES[name]
    pdb = potts_root / "energy_benchmark_datasets" / rel
    loaded = loaded_residues(pdb)
    labels = auto_labels(loaded) if labelled else None
    native = np.asarray(featurize_pdb(pdb, labels)["S"][0], dtype=np.int64)
    length = int(native.shape[0])
    bucket = _bucket_for(length)
    variants = _variants(native, rng, PROTONPOTTS_V6.size)
    names = [[PROTONPOTTS_V6.symbols[t] for t in row] for row in variants]
    scoring.append({"id": f"{name}__main", "pdb_path": str(pdb), "bucket": bucket, "labels": labels, "variants": names, "length": length})
    sampler.append({"id": f"{name}__s{BENCH_SEED}", "bucket": bucket, "pdb_file": str(pdb), "designed": None,
                    "temperature": float(np.float32(0.1)), "bias": None, "uniforms": None, "noise": None, "seed": BENCH_SEED,
                    "length": length})
    binder_chain = [k[0] for k in kept_residues(pdb)][-1]
    for case, types, temperature in CASES:
      if case not in PH_CASES:
        continue
      uniforms = ([ph_rng.uniform(size=BLOCK_ROUNDS * MAX_MUTATIONS).astype(np.float32).tolist() for _ in range(SAMPLES)]
                  if temperature > 0 else None)
      ph.append({"id": f"{name}__{case}", "bucket": bucket, "pdb_path": str(pdb), "labels": None, "length": length,
                 "config": {"binderChain": binder_chain, "centerTypes": list(types), "temperature": temperature,
                            "samplesPerSite": SAMPLES, "maxMutations": MAX_MUTATIONS, "blockMaxRounds": BLOCK_ROUNDS},
                 "uniforms": uniforms})
  return {"scoring": scoring, "ph": ph, "sampler": sampler}


def _patch_cells_json(site: Path, reps: int, warmup: int, planted_ms: float) -> None:
  path = site / "cells.json"
  data = json.loads(path.read_text())
  data.update(reps=reps, warmup=warmup, planted_ms=planted_ms)
  path.write_text(json.dumps(data))


def _complete(arm: str, cell: dict[str, Any], reps: int) -> bool:
  """A cell counts only if it returned every repetition AND a non-empty result."""
  if not cell.get("ok") or len(cell.get("wall_ms_all") or []) != reps:
    return False
  if arm == "scoring":
    return cell.get("energies") is not None and int(cell.get("l_total") or 0) > 0
  if arm == "ph":
    designs = cell.get("designs") or []
    return len(designs) == SAMPLES and all(len(d.get("sequence") or []) > 0 for d in designs)
  return len(cell.get("sequence") or []) > 0 and len(cell.get("log_probs") or []) > 0


def _summarize(arm: str, run: dict[str, Any], payload: dict[str, Any], n_cells: int, reps: int) -> None:
  cells = payload.get("cells", [])
  ok_cells = [c for c in cells if _complete(arm, c, reps)]
  walls = [w for c in ok_cells for w in c["wall_ms_all"]]
  planted = float(payload.get("planted_ms", 0.0))
  measured = float(payload.get("ctrl_measured_ms", -1.0))
  run.update(
    cross_origin_isolated=bool(payload.get("cross_origin_isolated")),
    num_threads_after_init=int(payload.get("num_threads_after_init", -1)),
    num_threads_effective=int(payload.get("num_threads_effective", -1)),
    ctrl_measured_ms=measured,
    ctrl_ratio=(measured / planted) if planted > 0 else -1.0,
    ctrl_in_window=bool(planted > 0 and CTRL_LO * planted <= measured <= planted + CTRL_SLACK_MS),
    cells_total=n_cells,
    cells_complete=len(ok_cells),
    n_walls=len(walls),
    median_ms=float(statistics.median(walls)) if walls else -1.0,
    min_ms=float(min(walls)) if walls else -1.0,
    max_ms=float(max(walls)) if walls else -1.0,
    session_create_ms={k: float(v) for k, v in payload.get("session_create_ms", {}).items()},
    per_cell_median_ms={c["id"]: float(statistics.median(c["wall_ms_all"])) for c in ok_cells},
    failed_cells=[{"id": c.get("id"), "error": str(c.get("error"))[:300]} for c in cells if not _complete(arm, c, reps)],
  )


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--scoring-models", type=Path, required=True)
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True)
  parser.add_argument("--ph-manifest-sha256", required=True)
  parser.add_argument("--decoder-models", type=Path, required=True)
  parser.add_argument("--decoder-manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))))
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--planted-ms", type=float, default=250.0)
  parser.add_argument("--timeout-ms", type=int, default=3_600_000)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean, "artifacts_verified": False,
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
    "decoder_manifest_sha256": args.decoder_manifest_sha256,
    "onnxruntime_web_version": "", "playwright_version": "", "node_version": "",
    "reps": args.reps, "warmup": args.warmup, "planted_ms": args.planted_ms,
    "arms_total": 0, "cells_total": 0, "bench_ok": False, "ctrl_measured_ms": -1.0, "ctrl_in_window": False,
    "threads_requested": list(THREADS), "arms": [], "failure": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    _verify_artifacts(args.scoring_models, args.scoring_manifest_sha256)
    _verify_artifacts(args.ph_models, args.ph_manifest_sha256)
    _verify_artifacts(args.decoder_models, args.decoder_manifest_sha256)
    result["artifacts_verified"] = True
    cells = build_cells(args.potts_root)
    result["arms_total"] = len(cells)
    result["cells_total"] = sum(len(v) for v in cells.values())

    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    sites: dict[str, Path] = {}
    for arm in ARMS:
      site = args.work_dir / f"site_{arm}"
      sites[arm] = site
      if arm == "scoring":
        scoring_gate.assemble_site(site, args.scoring_models, args.ort_dir, [dict(c) for c in cells[arm]])
      elif arm == "ph":
        ph_gate.assemble_site(site, args.scoring_models, args.ph_models, args.ort_dir, [dict(c) for c in cells[arm]])
      else:
        pdbs = {Path(c["pdb_file"]).stem: Path(c["pdb_file"]) for c in cells[arm]}
        sampler_gate.assemble_site(site, args.decoder_models, args.ort_dir, [dict(c) for c in cells[arm]], pdbs)
      _patch_cells_json(site, args.reps, args.warmup, args.planted_ms)
    logger.info("assembled %d sites, %d cells x (%d warm-up + %d reps)", len(sites), result["cells_total"], args.warmup, args.reps)

    for arm in ARMS:
      entry: dict[str, Any] = {"arm": arm, "n_cells": len(cells[arm]), "runs": []}
      result["arms"].append(entry)
      for threads in THREADS:
        run: dict[str, Any] = {"threads_requested": threads, "harness_ok": False}
        entry["runs"].append(run)
        try:
          harness = run_harness(node, sites[arm], args.work_dir / f"browser_{arm}_t{threads}", threads, args.timeout_ms)
          run["chromium_version"] = str(harness.get("chromiumVersion") or "")
          if not harness.get("harnessOk"):
            run["harness_error"] = str(harness.get("harnessError"))[:800]
            continue
          run["harness_ok"] = True
          _summarize(arm, run, harness["result"], len(cells[arm]), args.reps)
          logger.info("%s threads %d (effective %d, isolated %s): median %.0f ms over %d/%d complete cells | session %s | control %.1f ms",
                      arm, threads, run["num_threads_effective"], run["cross_origin_isolated"], run["median_ms"],
                      run["cells_complete"], run["cells_total"], run["session_create_ms"], run["ctrl_measured_ms"])
        except Exception as exc:  # noqa: BLE001
          run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
          logger.exception("%s threads %d bench run failed", arm, threads)

    all_runs = [r for e in result["arms"] for r in e["runs"]]
    result["bench_ok"] = len(result["arms"]) == len(ARMS) and all(
      len(e["runs"]) == len(THREADS) for e in result["arms"]) and all(
      r["harness_ok"] and r.get("cells_complete") == r.get("cells_total") for r in all_runs)
    result["ctrl_in_window"] = bool(all_runs) and all(r.get("ctrl_in_window") for r in all_runs)
    measured = [r["ctrl_measured_ms"] for r in all_runs if "ctrl_measured_ms" in r]
    result["ctrl_measured_ms"] = float(min(measured)) if measured else -1.0
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("browser bench failed")
    result["failure"] = f"{type(exc).__name__}: {exc}"[:4000]

  result["outcome"] = (
    "incomplete"
    if not result["bench_ok"] or result["arms_total"] < 3 or result["cells_total"] < 6 or result["reps"] < 3
    or not result["git_clean"] or result["failure"]
    else "timer_blind" if not result["ctrl_in_window"]
    else "pass"
  )
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("outcome=%s", result["outcome"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
