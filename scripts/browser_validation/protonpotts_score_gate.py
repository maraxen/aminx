"""X6b gate: the SHIPPING browser/protonpotts-scorer/protonpotts_scorer.mjs vs the real aminx driver, under Node/ORT-Web.

task_id 261009_protonpotts-onnx, backlog #5816. Sidecar: protonpotts_score_gate.bth.toml (pre-registered before this
script existed).

The X6a gate (run 40b4318e) proved the two exported graphs in isolation. This gate proves the JS that chains them AND
builds their inputs from a PDB file: ``protonpotts_scorer.mjs`` runs under Node with onnxruntime-web (wasm, 1 thread) on the
X6a artifacts (verified by sha256, not re-exported). The reference is the REAL user path, ``aminx.host.runner.score`` on the
structure file with the converted v6 checkpoint, ``ProtonPottsOptions(protonation_labels_json, variants_json)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.protonpotts_dump_inputs import auto_labels  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model, _sha256  # noqa: E402

logger = logging.getLogger("protonpotts_score_gate")

REL_BAR = 1.0e-4
N_VARIANTS = 19
CELLS = {"1EL1": True, "1BVC": True, "1OLR": False, "6m0j": True}  # name -> labelled
SEED = 7


def _rel(ref: np.ndarray, got: np.ndarray) -> float:
  ref = np.asarray(ref, dtype=np.float64)
  got = np.asarray(got, dtype=np.float64)
  if ref.shape != got.shape:
    return 1.0e9
  return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))


def _verify_artifacts(models: Path, want_sha256: str) -> dict[str, Any]:
  manifest_path = models / "MANIFEST.json"
  got = _sha256(manifest_path)
  if got != want_sha256:
    msg = f"MANIFEST sha256 {got} != expected X6a manifest {want_sha256}"
    raise RuntimeError(msg)
  manifest = json.loads(manifest_path.read_text())
  for bucket in manifest["buckets"]:
    for graph in bucket["graphs"].values():
      file_sha = _sha256(models / graph["file"])
      if file_sha != graph["sha256"]:
        msg = f"{graph['file']} sha256 {file_sha} != manifest {graph['sha256']}"
        raise RuntimeError(msg)
  return manifest


def _variants(native: np.ndarray, rng: np.random.Generator, n_tokens: int) -> list[list[int]]:
  rows = []
  for _ in range(N_VARIANTS):
    row = native.copy()
    for pos in rng.choice(native.shape[0], size=int(rng.integers(1, 4)), replace=False):
      row[pos] = (row[pos] + int(rng.integers(1, n_tokens))) % n_tokens
    rows.append([int(t) for t in row])
  return rows


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True, help="X6a artifact dir (MANIFEST.json + .onnx)")
  parser.add_argument("--manifest-sha256", required=True, help="sha256 of the X6a gate run's MANIFEST.json")
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path,
                      default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "jax_arm_ok": False, "artifacts_verified": False, "node_ok": False, "num_threads": 0, "cells": [],
    "n_cells": 0, "n_pass": 0, "tokens_exact": False, "worst_rel": 1.0e9,
    "control_vocab_detected": False, "control_labels_detected": False, "control_order_detected": False,
    "manifest_sha256": args.manifest_sha256, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)
    from aminx.families.protonpotts_mpnn.features import featurize_pdb, loaded_residues  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415
    from aminx.host.runner import score  # noqa: PLC0415
    from aminx.run.options import ProtonPottsOptions  # noqa: PLC0415
    from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True

    # ---- driver arm: the real score path, per cell ------------------------------------------------
    rng = np.random.default_rng(SEED)
    refs: dict[str, dict[str, Any]] = {}
    node_cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      model_path = tmp_dir / "model.eqx"
      eqx.tree_serialise_leaves(model_path, _load_model(args.state_npz))
      for name, labelled in CELLS.items():
        rel, _sha = STRUCTURES[name]
        pdb = args.potts_root / "energy_benchmark_datasets" / rel
        loaded = loaded_residues(pdb)
        labels = auto_labels(loaded) if labelled else None
        label_map = ({f"{c}:{n}{ic.strip()}": lab for (c, n, ic, _nm), lab in zip(loaded, labels, strict=True) if lab}
                     if labels else None)
        native = np.asarray(featurize_pdb(pdb, labels)["S"][0], dtype=np.int64)
        variants = _variants(native, rng, PROTONPOTTS_V6.size)
        names = {f"v{i}": [PROTONPOTTS_V6.symbols[t] for t in row] for i, row in enumerate(variants)}
        variants_path = tmp_dir / f"{name}_variants.json"
        variants_path.write_text(json.dumps(names), encoding="utf-8")
        labels_path = None
        if label_map is not None:
          labels_path = tmp_dir / f"{name}_labels.json"
          labels_path.write_text(json.dumps(label_map), encoding="utf-8")
        spec = ScoringSpecification(
          inputs=str(pdb), model_family="protonpottsmpnn", model_local_path=model_path, output_kind="energy",
          protonpotts=ProtonPottsOptions(
            variants_json=str(variants_path),
            protonation_labels_json=str(labels_path) if labels_path else None),
        )
        arrays = score(spec)["structures"]["0"]["arrays"]
        refs[name] = {
          "energy": np.asarray(arrays["energy"], dtype=np.float64),
          "tokens0": np.asarray(arrays["candidate_tokens"])[0].astype(np.int64),
          "labelled": labelled, "l_total": int(native.shape[0]),
        }
        bucket = _bucket_for(int(native.shape[0]))
        base = {"bucket": bucket, "pdb_file": str(pdb), "out_dir": str(args.work_dir / "out")}
        node_cases.append({**base, "id": f"{name}__main", "labels": labels, "variants": list(names.values())})
        node_cases.append({**base, "id": f"{name}__vocab", "labels": labels,
                           "variants": [[(t + 1) % PROTONPOTTS_V6.size for t in row] for row in variants]})
        if labelled:
          node_cases.append({**base, "id": f"{name}__nolabels", "labels": None, "variants": list(names.values())})
    result["jax_arm_ok"] = True
    logger.info("driver arm: %d cells in %.1fs", len(refs), time.perf_counter() - t0)

    # ---- shipping JS scorer under Node ---------------------------------------------------------------
    cases_path = args.work_dir / "score_cases.json"
    out_path = args.work_dir / "score_result.json"
    cases_path.write_text(json.dumps({"cases": node_cases}))
    runner = REPO / "browser" / "protonpotts-scorer" / "run_protonpotts_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--models", str(args.models),
       "--cases", str(cases_path), "--out", str(out_path)],
      capture_output=True, text=True, timeout=3600, check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
      msg = f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}"
      raise RuntimeError(msg)
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    result["num_threads"] = int(node.get("num_threads") or 0)
    got: dict[str, dict[str, Any]] = {}
    for rec in node["results"]:
      energies = np.fromfile(rec["energies"]["file"], dtype=np.float32) if rec["ok"] else None
      got[rec["id"]] = {"ok": rec["ok"], "error": rec.get("error"), "energies": energies,
                        "tokens": np.asarray(rec["tokens"], dtype=np.int64) if rec["ok"] else None}

    # ---- grade ---------------------------------------------------------------------------------------
    vocab_hits = labels_hits = order_hits = n_labelled = 0
    all_tokens_exact = True
    worst = 0.0
    for name, ref in refs.items():
      g = got.get(f"{name}__main", {"ok": False, "error": "missing"})
      row: dict[str, Any] = {"cell": name, "l_total": ref["l_total"], "labelled": ref["labelled"], "ran": g["ok"]}
      if not g["ok"]:
        row.update(passed=False, error=(g.get("error") or "")[:2000])
        all_tokens_exact = False
      else:
        row["tokens_exact"] = bool(np.array_equal(g["tokens"], ref["tokens0"]))
        row["max_rel"] = _rel(ref["energy"], g["energies"])
        row["n_rows"] = int(ref["energy"].shape[0])
        row["passed"] = row["tokens_exact"] and row["max_rel"] <= REL_BAR
        all_tokens_exact &= row["tokens_exact"]
        worst = max(worst, row["max_rel"])
        order_hits += int(_rel(ref["energy"], g["energies"][::-1]) > REL_BAR)
        v = got.get(f"{name}__vocab")
        vocab_hits += int(bool(v and v["ok"]) and _rel(ref["energy"], v["energies"]) > REL_BAR)
        if ref["labelled"]:
          n_labelled += 1
          nl = got.get(f"{name}__nolabels")
          labels_hits += int(bool(nl and nl["ok"]) and _rel(ref["energy"], nl["energies"]) > REL_BAR)
      result["cells"].append(row)
    result["control_vocab_detected"] = vocab_hits == len(refs)
    result["control_labels_detected"] = n_labelled > 0 and labels_hits == n_labelled
    result["control_order_detected"] = order_hits == len(refs)
    result["n_cells"] = len(result["cells"])
    result["n_pass"] = sum(bool(r.get("passed")) for r in result["cells"])
    result["tokens_exact"] = all_tokens_exact
    result["worst_rel"] = worst
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("score gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("cells pass %s/%s", result["n_pass"], result["n_cells"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
