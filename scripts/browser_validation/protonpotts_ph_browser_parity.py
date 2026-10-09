"""X7c browser parity: the ProtonPotts pH design page in REAL headless Chromium vs aminx's design_structure.

task_id 261009_protonpotts-onnx, backlog #5816 (B1). Sidecar: protonpotts_ph_browser_parity.bth.toml (pre-registered before this
script existed). Serves the shipping page (browser/protonpotts-scorer/ph_index.html + ph_page.mjs) over HTTP with COOP/COEP and drives
it through browser/layer_c/run_p07.mjs UNCHANGED at 1 and 4 ORT threads. The reference arm, cases, seeds and comparator are the Node
loop gate's (protonpotts_ph_loop_gate), so the two records grade the same inputs. The page grades nothing.
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

from scripts.browser_validation.layer_a_common import emit
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report
from scripts.browser_validation.potts_browser_parity import git_state, run_harness
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model
from scripts.browser_validation.protonpotts_ph_loop_gate import (
  BLOCK_ROUNDS,
  CASES,
  ENERGY_BAR,
  MAX_MUTATIONS,
  SAMPLES,
  SEED,
  _design,
  _rel,
)
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts

logger = logging.getLogger("protonpotts_ph_browser_parity")

SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
PAGE_FILES = ("ph_page.mjs", "ph_design.mjs", "ph_plan.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs")
THREADS = (1, 4)


def build_reference(potts_root: Path, state_npz: Path):  # noqa: ANN201
  """design_structure per cell and case (as the Node loop gate), the rolled-uniform control references, and the page cells."""
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.energy import protonpotts_table  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415

  model = _load_model(state_npz)
  rng = np.random.default_rng(SEED)
  refs, rolled, cells, context = {}, {}, [], {}
  for name, (rel, _sha) in STRUCTURES.items():
    pdb = potts_root / "energy_benchmark_datasets" / rel
    feats = featurize_pdb(pdb, None)
    kept = kept_residues(pdb)
    native = np.asarray(feats["S"][0], dtype=np.int32)
    length = native.shape[0]
    chains = [k[0] for k in kept]
    binder_chain = chains[-1]
    binder = np.asarray([c == binder_chain for c in chains])
    res_id = np.asarray([k[1] for k in kept], dtype=np.int32)
    args_t = (jnp.asarray(feats["X"][0, :, :4, :], dtype=jnp.float32), jnp.ones(length, jnp.float32),
              jnp.asarray(feats["R_idx"][0]), jnp.asarray(feats["chain_labels"][0]), jnp.ones(length, bool))
    table, e_idx = jax.jit(lambda *a, _m=model: protonpotts_table(_m, *a))(*args_t)
    context[name] = {"args_t": args_t}
    for case, types, temperature in CASES:
      cid = f"{name}__{case}"
      config = PHDesignConfig(binder_chain=binder_chain, center_types=types, temperature=temperature,
                              samples_per_site=SAMPLES, max_mutations=MAX_MUTATIONS, block_max_rounds=BLOCK_ROUNDS)
      uniforms = ([rng.uniform(size=BLOCK_ROUNDS * MAX_MUTATIONS).astype(np.float32) for _ in range(SAMPLES)]
                  if temperature > 0 else None)
      refs[cid] = _design(table, e_idx, native, binder, res_id, config, uniforms)
      if temperature > 0:
        rolled[cid] = _design(table, e_idx, native, binder, res_id, config, [np.roll(u, 1) for u in uniforms])
      cells.append({"id": cid, "bucket": _bucket_for(length), "pdb_path": str(pdb), "labels": None,
                    "config": {"binderChain": binder_chain, "centerTypes": list(types), "temperature": temperature,
                               "samplesPerSite": SAMPLES, "maxMutations": MAX_MUTATIONS, "blockMaxRounds": BLOCK_ROUNDS},
                    "uniforms": None if uniforms is None else [u.tolist() for u in uniforms]})
  return model, refs, rolled, cells, context


def assemble_site(site: Path, scoring: Path, ph: Path, ort_dir: Path, cells: list[dict[str, Any]]) -> None:
  if site.exists():
    shutil.rmtree(site)
  for sub in ("models/scoring", "models/ph", "ort", "data"):
    (site / sub).mkdir(parents=True)
  shutil.copy(SCORER_DIR / "ph_index.html", site / "index.html")
  for name in PAGE_FILES:
    shutil.copy(SCORER_DIR / name, site / name)
  for src, dst in ((scoring, "scoring"), (ph, "ph")):
    manifest = json.loads((src / "MANIFEST.json").read_text())
    shutil.copy(src / "MANIFEST.json", site / "models" / dst / "MANIFEST.json")
    for graph_file in sorted({g["file"] for b in manifest["buckets"] for g in b["graphs"].values()}):
      shutil.copy(src / graph_file, site / "models" / dst / graph_file)
  dist = ort_dir / "node_modules" / "onnxruntime-web" / "dist"
  for f in dist.iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)
  site_cells = []
  for cell in cells:
    src = Path(cell["pdb_path"])
    if not (site / "data" / src.name).exists():
      shutil.copy(src, site / "data" / src.name)
    site_cells.append({k: cell[k] for k in ("id", "bucket", "labels", "config", "uniforms")} | {"pdb": f"data/{src.name}"})
  (site / "cells.json").write_text(json.dumps({"cells": site_cells}))


def grade(refs: dict[str, Any], rolled: dict[str, Any], page: dict[str, dict[str, Any]]) -> dict[str, Any]:
  rows, tok_hits, tok_n = [], 0, 0
  for cid, ref in refs.items():
    g = page.get(cid)
    if not g or not g.get("ok"):
      rows.append({"id": cid, "ran": False, "passed": False, "error": ((g or {}).get("error") or "missing")[:1500]})
      continue
    js = g["designs"]
    ok = len(js) == len(ref) > 0
    sequences = designable = pins = draws = True
    worst = 0.0
    for j, r in zip(js, ref, strict=False):
      sequences &= bool(np.array_equal(np.asarray(j["sequence"]), r.sequence))
      designable &= list(j["designable"]) == [int(d) for d in r.designable]
      pins &= [(p["position"], p["protIdx"], p["resId"]) for p in j["pins"]] == [
        (p.position, p.prot_idx, p.res_id) for p in r.pins] and j["label"] == r.label
      draws &= int(j["n_draws"]) == int(r.n_draws)
      worst = max(worst, _rel(r.final_potts_energy, j["final_potts_energy"]), _rel(r.selective_energy, j["selective_energy"]))
    rows.append({"id": cid, "ran": True, "sequences_exact": bool(sequences), "designable_equal": bool(designable),
                 "pins_equal": bool(pins), "draws_equal": bool(draws), "worst_energy_rel": worst,
                 "passed": bool(ok and sequences and designable and pins and draws and worst <= ENERGY_BAR)})
    if cid in rolled:
      tok_n += 1
      tok_hits += int(not all(np.array_equal(np.asarray(j["sequence"]), r.sequence) for j, r in zip(js, rolled[cid], strict=False)))
  return {"cases": rows, "n_cases": len(rows), "n_pass": sum(bool(r.get("passed")) for r in rows),
          "control_tokens_detected": tok_n > 0 and tok_hits == tok_n}


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--scoring-models", type=Path, required=True)
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True)
  parser.add_argument("--ph-manifest-sha256", required=True)
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
    "git_hash": git_hash, "git_clean": git_clean, "scoring_manifest_sha256": args.scoring_manifest_sha256,
    "ph_manifest_sha256": args.ph_manifest_sha256, "artifacts_verified": False, "jax_arm_ok": False, "jax_version": "",
    "onnxruntime_web_version": "", "playwright_version": "", "node_version": "", "n_expected_per_thread": 0,
    "threads_requested": list(THREADS), "runs": [], "browser_complete": False, "isolated_at_4": False,
    "control_tokens_detected": False, "control_energy_detected": False, "all_exact": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from aminx.families.protonpotts_mpnn.energy import protonpotts_energies  # noqa: PLC0415

    result["jax_version"] = str(jax.__version__)
    _verify_artifacts(args.scoring_models, args.scoring_manifest_sha256)
    _verify_artifacts(args.ph_models, args.ph_manifest_sha256)
    result["artifacts_verified"] = True
    model, refs, rolled, cells, context = build_reference(args.potts_root, args.state_npz)
    result["n_expected_per_thread"] = len(cells)
    result["jax_arm_ok"] = True
    logger.info("reference arm: %d cases in %.1fs", len(cells), time.perf_counter() - t0)

    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    env = node_env_report(node)
    result["node_version"] = env.get("node") or ""
    result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
    result["playwright_version"] = env.get("@playwright/test") or ""

    noise_key = jax.random.PRNGKey(1)
    perturbed = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.potts_head)
    bad_model = eqx.tree_at(lambda m: m.potts_head, model, perturbed)

    site = args.work_dir / "site"
    assemble_site(site, args.scoring_models, args.ph_models, args.ort_dir, cells)
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
        page = {c["id"]: c for c in payload.get("cells", [])}
        run["n_page_cells"] = sum(1 for c in page.values() if c.get("ok"))
        run.update(grade(refs, rolled, page))
        ctrl = next((cid for cid in refs if page.get(cid, {}).get("ok")), None)
        run["control_energy_detected"] = False
        if ctrl is not None:
          designs = page[ctrl]["designs"]
          seqs = jnp.asarray(np.stack([np.asarray(d["sequence"]) for d in designs]).astype(np.int32))
          p_energy = np.asarray(protonpotts_energies(bad_model, *context[ctrl.split("__", 1)[0]]["args_t"], seqs))
          run["control_energy_detected"] = all(
            _rel(p, d["final_potts_energy"]) > ENERGY_BAR for p, d in zip(p_energy, designs, strict=True))
        logger.info("threads %d (effective %d, isolated %s): %d/%d cases exact; controls tokens=%s energy=%s", threads,
                    run["num_threads_effective"], run["cross_origin_isolated"], run["n_pass"], run["n_cases"],
                    run["control_tokens_detected"], run["control_energy_detected"])
      except Exception as exc:  # noqa: BLE001
        run["harness_error"] = f"{type(exc).__name__}: {exc}"[:800]
        logger.exception("threads %d run failed", threads)

    runs = result["runs"]
    result["browser_complete"] = len(runs) == len(THREADS) and all(
      r["harness_ok"] and r.get("n_page_cells") == result["n_expected_per_thread"] for r in runs)
    by_threads = {r["threads_requested"]: r for r in runs if r["harness_ok"]}
    result["isolated_at_4"] = bool(by_threads.get(4, {}).get("cross_origin_isolated", False))
    result["control_tokens_detected"] = bool(runs) and all(r.get("control_tokens_detected") for r in runs)
    result["control_energy_detected"] = bool(runs) and all(r.get("control_energy_detected") for r in runs)
    result["all_exact"] = bool(runs) and all(r.get("n_cases", 0) > 0 and r.get("n_pass") == r.get("n_cases") for r in runs)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("ph browser parity gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("done: complete=%s isolated@4=%s controls=%s/%s exact=%s", result["browser_complete"], result["isolated_at_4"],
              result["control_tokens_detected"], result["control_energy_detected"], result["all_exact"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
