"""X6c browser parity: the ProtonPotts scoring page in REAL headless Chromium vs the real aminx driver.

task_id 261009_protonpotts-onnx, backlog #5816. Sidecar: protonpotts_browser_parity.bth.toml (pre-registered before this
script existed). Mirrors potts_browser_parity (X5b).

Serves the shipping page (browser/protonpotts-scorer/protonpotts_index.html + protonpotts_page.mjs) over HTTP with
COOP/COEP and drives it through browser/layer_c/run_p07.mjs UNCHANGED, at 1 and at 4 ORT threads. The reference is the REAL
user path, aminx.host.runner.score, with the cells, labels rule, variant generator and seed of the Node gate
protonpotts_score_gate (graded run 1a2355c5), so the two records grade the same inputs. The page grades nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report
from scripts.browser_validation.potts_browser_parity import git_state, run_harness
from scripts.browser_validation.protonpotts_dump_inputs import auto_labels
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model
from scripts.browser_validation.protonpotts_score_gate import CELLS, REL_BAR, SEED, _rel, _variants, _verify_artifacts

logger = logging.getLogger("protonpotts_browser_parity")

SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
RUNNER = LAYER_C_DIR / "run_p07.mjs"
PAGE_FILES = ("protonpotts_page.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs")
THREADS = (1, 4)


def driver_reference(
  potts_root: Path, state_npz: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
  """The real score path per cell, plus the page cells (main, vocab-shifted, and no-labels for labelled cells)."""
  import equinox as eqx  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.features import featurize_pdb, loaded_residues  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415
  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.run.options import ProtonPottsOptions  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  rng = np.random.default_rng(SEED)
  refs: dict[str, dict[str, Any]] = {}
  cases: list[dict[str, Any]] = []
  with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    model_path = tmp_dir / "model.eqx"
    eqx.tree_serialise_leaves(model_path, _load_model(state_npz))
    for name, labelled in CELLS.items():
      rel, _sha = STRUCTURES[name]
      pdb = potts_root / "energy_benchmark_datasets" / rel
      loaded = loaded_residues(pdb)
      labels = auto_labels(loaded) if labelled else None
      label_map = ({f"{c}:{n}{ic.strip()}": lab for (c, n, ic, _nm), lab in zip(loaded, labels, strict=True) if lab}
                   if labels else None)
      native = np.asarray(featurize_pdb(pdb, labels)["S"][0], dtype=np.int64)
      variants = _variants(native, rng, PROTONPOTTS_V6.size)
      names = [[PROTONPOTTS_V6.symbols[t] for t in row] for row in variants]
      variants_path = tmp_dir / f"{name}_variants.json"
      variants_path.write_text(json.dumps({f"v{i}": row for i, row in enumerate(names)}), encoding="utf-8")
      labels_path = None
      if label_map is not None:
        labels_path = tmp_dir / f"{name}_labels.json"
        labels_path.write_text(json.dumps(label_map), encoding="utf-8")
      spec = ScoringSpecification(
        inputs=str(pdb), model_family="protonpottsmpnn", model_local_path=model_path, output_kind="energy",
        protonpotts=ProtonPottsOptions(
          variants_json=str(variants_path), protonation_labels_json=str(labels_path) if labels_path else None),
      )
      arrays = score(spec)["structures"]["0"]["arrays"]
      refs[name] = {
        "energy": np.asarray(arrays["energy"], dtype=np.float64),
        "tokens0": np.asarray(arrays["candidate_tokens"])[0].astype(np.int64),
        "labelled": labelled, "l_total": int(native.shape[0]),
      }
      base = {"pdb_path": str(pdb), "bucket": _bucket_for(int(native.shape[0]))}
      cases.append({**base, "id": f"{name}__main", "labels": labels, "variants": names})
      cases.append({**base, "id": f"{name}__vocab", "labels": labels,
                    "variants": [[(t + 1) % PROTONPOTTS_V6.size for t in row] for row in variants]})
      if labelled:
        cases.append({**base, "id": f"{name}__nolabels", "labels": None, "variants": names})
  return refs, cases


def assemble_site(site: Path, models: Path, ort_dir: Path, cases: list[dict[str, Any]]) -> None:
  """Lay out exactly what the page fetches: the page, its modules, ORT, models, PDB data, cells.json."""
  if site.exists():
    shutil.rmtree(site)
  (site / "models").mkdir(parents=True)
  (site / "ort").mkdir()
  (site / "data").mkdir()
  shutil.copy(SCORER_DIR / "protonpotts_index.html", site / "index.html")
  for name in PAGE_FILES:
    shutil.copy(SCORER_DIR / name, site / name)
  manifest_path = models / "MANIFEST.json"
  shutil.copy(manifest_path, site / "models" / "MANIFEST.json")
  manifest = json.loads(manifest_path.read_text())
  for graph_file in sorted({g["file"] for b in manifest["buckets"] for g in b["graphs"].values()}):
    shutil.copy(models / graph_file, site / "models" / graph_file)
  dist = ort_dir / "node_modules" / "onnxruntime-web" / "dist"
  for f in dist.iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)
  site_cells = []
  for case in cases:
    src = Path(case["pdb_path"])
    if not (site / "data" / src.name).exists():
      shutil.copy(src, site / "data" / src.name)
    site_cells.append({"id": case["id"], "pdb": f"data/{src.name}", "bucket": case["bucket"],
                       "labels": case["labels"], "variants": case["variants"]})
  (site / "cells.json").write_text(json.dumps({"cells": site_cells, "reps": 1, "warmup": 0, "planted_ms": 0}))


def page_results(payload: dict[str, Any], out_dir: Path) -> dict[str, dict[str, Any]]:
  got: dict[str, dict[str, Any]] = {}
  for cell in payload.get("cells", []):
    if not cell.get("ok") or cell.get("energies") is None:
      got[cell["id"]] = {"ok": False, "error": cell.get("error") or "no energies", "energies": None, "tokens": None}
      continue
    got[cell["id"]] = {
      "ok": True, "error": None,
      "energies": np.fromfile(out_dir / cell["energies"]["file"], dtype=np.float32),
      "tokens": np.asarray(cell["tokens"], dtype=np.int64),
    }
  return got


def grade_run(refs: dict[str, dict[str, Any]], got: dict[str, dict[str, Any]]) -> dict[str, Any]:
  rows = []
  vocab = labels_hits = order = n_labelled = 0
  for name, ref in refs.items():
    g = got.get(f"{name}__main", {"ok": False, "error": "missing"})
    row: dict[str, Any] = {"cell": name, "l_total": ref["l_total"], "labelled": ref["labelled"], "ran": g["ok"]}
    if not g["ok"]:
      row.update(passed=False, error=str(g.get("error"))[:1500])
    else:
      row["tokens_exact"] = bool(np.array_equal(g["tokens"], ref["tokens0"]))
      row["max_rel"] = _rel(ref["energy"], g["energies"])
      row["passed"] = row["tokens_exact"] and row["max_rel"] <= REL_BAR
      order += int(_rel(ref["energy"], g["energies"][::-1]) > REL_BAR)
      v = got.get(f"{name}__vocab")
      vocab += int(bool(v and v["ok"]) and _rel(ref["energy"], v["energies"]) > REL_BAR)
      if ref["labelled"]:
        n_labelled += 1
        nl = got.get(f"{name}__nolabels")
        labels_hits += int(bool(nl and nl["ok"]) and _rel(ref["energy"], nl["energies"]) > REL_BAR)
    rows.append(row)
  n = len(refs)
  return {
    "cells": rows, "n_cells": n, "n_pass": sum(bool(r.get("passed")) for r in rows),
    "control_vocab_detected": vocab == n,
    "control_labels_detected": n_labelled > 0 and labels_hits == n_labelled,
    "control_order_detected": order == n,
  }


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
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
  parser.add_argument("--timeout-ms", type=int, default=1_800_000)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean, "manifest_sha256": "", "artifacts_verified": False,
    "jax_arm_ok": False, "jax_version": "", "onnxruntime_web_version": "", "playwright_version": "",
    "node_version": "", "n_expected_per_thread": 0, "threads_requested": list(THREADS), "runs": [],
    "browser_complete": False, "isolated_at_4": False, "control_vocab_detected": False,
    "control_labels_detected": False, "control_order_detected": False, "all_exact": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415

    result["jax_version"] = str(jax.__version__)
    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True
    result["manifest_sha256"] = args.manifest_sha256
    refs, cases = driver_reference(args.potts_root, args.state_npz)
    result["n_expected_per_thread"] = len(cases)
    result["jax_arm_ok"] = True
    logger.info("driver arm: %d cells, %d page cells", len(refs), len(cases))

    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    site = args.work_dir / "site"
    assemble_site(site, args.models, args.ort_dir, cases)
    for threads in THREADS:
      run: dict[str, Any] = {"threads_requested": threads, "harness_ok": False, "harness_error": ""}
      result["runs"].append(run)
      try:
        out_dir = args.work_dir / f"browser_t{threads}"
        harness = run_harness(node, site, out_dir, threads, args.timeout_ms)
        run["chromium_version"] = str(harness.get("chromiumVersion") or "")
        if not harness.get("harnessOk"):
          run["harness_error"] = str(harness.get("harnessError"))[:800]
          continue
        payload = harness["result"]
        run["harness_ok"] = True
        run["cross_origin_isolated"] = bool(payload.get("cross_origin_isolated"))
        run["num_threads_after_init"] = int(payload.get("num_threads_after_init", -1))
        run["num_threads_effective"] = int(payload.get("num_threads_effective", -1))
        page = page_results(payload, out_dir)
        run["n_page_cells"] = sum(1 for g in page.values() if g["ok"])
        run.update(grade_run(refs, page))
        logger.info("threads %d (effective %d, isolated %s): %d/%d cells exact; controls vocab=%s labels=%s order=%s",
                    threads, run["num_threads_effective"], run["cross_origin_isolated"], run["n_pass"],
                    run["n_cells"], run["control_vocab_detected"], run["control_labels_detected"],
                    run["control_order_detected"])
      except Exception as exc:  # noqa: BLE001
        run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
        logger.exception("threads %d run failed", threads)

    runs = result["runs"]
    result["browser_complete"] = len(runs) == len(THREADS) and all(
      r["harness_ok"] and r.get("n_page_cells") == result["n_expected_per_thread"] for r in runs)
    by_threads = {r["threads_requested"]: r for r in runs if r["harness_ok"]}
    result["isolated_at_4"] = bool(by_threads.get(4, {}).get("cross_origin_isolated", False))
    for ctl in ("control_vocab_detected", "control_labels_detected", "control_order_detected"):
      result[ctl] = bool(runs) and all(r.get(ctl) for r in runs)
    result["all_exact"] = bool(runs) and all(r.get("n_cells", 0) > 0 and r.get("n_pass") == r.get("n_cells")
                                             for r in runs)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("browser parity gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("done: complete=%s isolated@4=%s controls=%s/%s/%s exact=%s", result["browser_complete"],
              result["isolated_at_4"], result["control_vocab_detected"], result["control_labels_detected"],
              result["control_order_detected"], result["all_exact"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
