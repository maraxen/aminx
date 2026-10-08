"""X0: PottsMPNN export spike -- does each Potts device path convert AND run (ORT-CPU, ORT-Web)?

task_id 261007_potts-onnx-export, backlog #5810. Sidecar: potts_export_spike.bth.toml
(pre-registered before any run). Scope spec:
.praxia/docs/specs/261007_potts-onnx-export-scope.md (§3, §5 lever 3).

Each probe is one real aminx Potts device function, traced as-is (no rewrite):

  score   driver.absolute_energies          encoder (k-NN top_k) -> potts_head ->
                                            merge_pair x2 (scatter-max) -> potts_energy
  sched   decode.schedule_groups            argsort x3, scatter mode="drop" with sentinels
  decode  decode.PottsARDecode (untied)     lax.scan over groups + lax.cond + drop scatters
  decode_tied  same, with tied groups
  refine  refine.PottsRefine mode="potts"   one sweep (the default optimization_mode)

Weights are a seeded random init: the spike measures operator coverage and backend
agreement, not model quality (layer-a parity against upstream is already sealed). The
geometry is the split exporter's helix fixture, with 8 pad rows and 4 fixed rows so the
pad and fixed-position branches run.

Every probe: convert with xtrax.export.convert_to_onnx (x64 refusal, in-graph RNG
refusal, JAX-namespace restore), make the file self-contained (embed_external_data, for
ORT-Web), run on ORT-CPU (xtrax run_onnx) and on onnxruntime-web wasm under Node
(browser/layer_c/run_onnx_io.mjs, 1 thread), and compare against JAX computed FIRST in
this process: integer outputs exact, float outputs max|d| / max|ref| <= FLOAT_REL_BAR.
Conversion alone is never a pass (layer-b lesson, 260927).

Controls (comparator must fire): a perturbed-weight score and an alternate-uniforms
decode, each compared to the reference with the same comparator, must FAIL it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

logger = logging.getLogger("potts_export_spike")

FLOAT_REL_BAR = 1.0e-4
BUCKET = 128
L_TOTAL = 120
N_SCORE_SEQS = 4
FIXED_ROWS = 4
TIED_POS = ((10, 40), (11, 41, 71), (90, 100))
WATCH_OPS = ("Loop", "If", "Scan", "ScatterND", "ScatterElements", "TopK", "ArgMax")
DTYPE_TAG = {np.dtype(np.float32): "float32", np.dtype(np.int32): "int32", np.dtype(np.bool_): "bool"}


def _fixture(seed: int = 0) -> dict[str, np.ndarray]:
  from scripts.browser_validation.p07_split_export import _helix_backbone  # noqa: PLC0415

  from aminx.families.potts_mpnn.sample_host import build_tie_groups_np  # noqa: PLC0415

  rng = np.random.default_rng(seed)
  geom = _helix_backbone(BUCKET, seed=seed)
  pad_valid = np.zeros(BUCKET, dtype=np.bool_)
  pad_valid[:L_TOTAL] = True
  present = pad_valid.astype(np.float32)
  chain_mask = present.copy()
  chain_mask[:FIXED_ROWS] = 0.0
  s_true = np.where(pad_valid, rng.integers(0, 20, BUCKET), 20).astype(np.int32)
  seqs = rng.integers(0, 20, (N_SCORE_SEQS, BUCKET)).astype(np.int32)
  seqs[:, L_TOTAL:] = 21  # etab X on pad rows
  return {
    "coords": geom["coords"],
    "present": present,
    "residue_idx": geom["residue_index"],
    "chain_index": geom["chain_index"],
    "pad_valid": pad_valid,
    "score_seqs": seqs,
    "s_true": s_true,
    "chain_mask": chain_mask,
    "chain_m_pos": present.copy(),
    "untied_groups": build_tie_groups_np((), L_TOTAL, BUCKET),
    "tied_groups": build_tie_groups_np(TIED_POS, L_TOTAL, BUCKET),
    "tied_beta": np.ones(BUCKET, dtype=np.float32),
    "randn": rng.standard_normal(BUCKET).astype(np.float32),
    "uniforms": rng.uniform(size=BUCKET).astype(np.float32),
    "refine_order": np.concatenate(
      [rng.permutation(L_TOTAL), np.arange(L_TOTAL, BUCKET)],
    ).astype(np.int32),
    "refine_uniforms": rng.uniform(size=(8, BUCKET)).astype(np.float32),
    "omit": np.zeros(21, np.float32),
    "bias": np.zeros(21, np.float32),
    "bias_by_res": np.zeros((BUCKET, 21), np.float32),
    "pssm_coef": np.zeros(BUCKET, np.float32),
    "pssm_bias": np.zeros((BUCKET, 21), np.float32),
    "pssm_log_odds_mask": np.zeros((BUCKET, 21), np.float32),
    "omit_aa_mask": np.zeros((BUCKET, 21), np.float32),
  }


def _build_probes(model: Any, fx: dict[str, np.ndarray]) -> dict[str, tuple[Any, list[np.ndarray]]]:
  """Probe callables (flat array args -> tuple of arrays) and their concrete inputs."""
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature, schedule_groups  # noqa: PLC0415
  from aminx.families.potts_mpnn.driver import absolute_energies  # noqa: PLC0415
  from aminx.families.potts_mpnn.etab import pad_etab_energy  # noqa: PLC0415
  from aminx.families.potts_mpnn.refine import PottsRefine  # noqa: PLC0415
  from aminx.families.potts_mpnn.sample_host import _dummy_binding, _encode  # noqa: PLC0415

  mods = {"layers": model.mpnn.decoder.layers, "w_s_embed": model.mpnn.w_s_embed, "w_out": model.mpnn.w_out}
  decode = PottsARDecode(**mods)
  refiner = PottsRefine(**mods)

  h_v, h_e, e_idx, forward, _table = (
    np.asarray(x)
    for x in _encode(
      model,
      jnp.asarray(fx["coords"]),
      jnp.asarray(fx["present"]),
      jnp.asarray(fx["residue_idx"]),
      jnp.asarray(fx["chain_index"]),
      jnp.asarray(fx["pad_valid"]),
    )
  )
  etab = np.asarray(pad_etab_energy(jnp.asarray(forward)))
  tables = _dummy_binding(BUCKET, etab.shape[-1], jnp.float32)

  def score(coords, present, residue_idx, chain_index, pad_valid, seqs):  # noqa: ANN001, ANN202
    return (absolute_energies(model, coords, present, residue_idx, chain_index, pad_valid, seqs),)

  def sched(tie_groups, pad_valid, randn, chain_mask, chain_m_pos, present):  # noqa: ANN001, ANN202
    return tuple(schedule_groups(tie_groups, pad_valid, randn, chain_mask, chain_m_pos, present, None))

  def dec(*a):  # noqa: ANN002, ANN202
    out = decode(*a, temperature=0.1, pssm_multi=0.0, pssm_bias_flag=False, pssm_log_odds_flag=False)
    return (out.sequence, out.rank_flat, out.decoding_order, out.h_v_stack)

  def ref(*a):  # noqa: ANN002, ANN202
    (seq, etab_, e_idx_, pad_valid, present, cm, cmp_, order, unif, omit, bias, bbr, pc, pb, plm, oam,
     tg, tb, hv, he) = a
    out = refiner(
      "potts", seq, etab_, e_idx_, pad_valid, present, cm, cmp_, order, unif, omit, bias, bbr, pc, pb,
      plm, oam, tg, tb, hv, he, tables,
      temperature=floor_temperature(0.0), pssm_multi=0.0, pssm_bias_flag=False,
      pssm_log_odds_flag=False, binding="none", tied=False, tied_epistasis=False, max_iters=1,
    )
    return (out.sequence, out.n_iters, out.ener_delta)

  def dec_inputs(groups: np.ndarray, uniforms: np.ndarray) -> list[np.ndarray]:
    return [
      h_v, h_e, e_idx, fx["present"], fx["pad_valid"], fx["s_true"], fx["chain_mask"],
      fx["chain_m_pos"], groups, fx["tied_beta"], fx["randn"], uniforms, fx["omit"], fx["bias"],
      fx["bias_by_res"], fx["pssm_coef"], fx["pssm_bias"], fx["pssm_log_odds_mask"], fx["omit_aa_mask"],
    ]

  start_seq = np.where(fx["pad_valid"], fx["s_true"], 20).astype(np.int32)
  return {
    "score": (score, [fx["coords"], fx["present"], fx["residue_idx"], fx["chain_index"], fx["pad_valid"],
                      fx["score_seqs"]]),
    "sched": (sched, [fx["tied_groups"], fx["pad_valid"], fx["randn"], fx["chain_mask"], fx["chain_m_pos"],
                      fx["present"]]),
    "decode": (dec, dec_inputs(fx["untied_groups"], fx["uniforms"])),
    "decode_tied": (dec, dec_inputs(fx["tied_groups"], fx["uniforms"])),
    "refine": (ref, [start_seq, etab, e_idx, fx["pad_valid"], fx["present"], fx["chain_mask"],
                     fx["chain_m_pos"], fx["refine_order"], fx["refine_uniforms"], fx["omit"], fx["bias"],
                     fx["bias_by_res"], fx["pssm_coef"], fx["pssm_bias"], fx["pssm_log_odds_mask"],
                     fx["omit_aa_mask"], fx["untied_groups"], fx["tied_beta"], h_v, h_e]),
  }


def compare(ref: list[np.ndarray], got: list[np.ndarray]) -> dict[str, Any]:
  """Ints/bools exact; floats max|d|/max|ref| <= FLOAT_REL_BAR. Dtype and shape must match."""
  if len(ref) != len(got):
    return {"ok": False, "why": f"{len(got)} outputs, expected {len(ref)}", "max_rel": None}
  worst = 0.0
  for i, (r, g) in enumerate(zip(ref, got, strict=True)):
    r, g = np.asarray(r), np.asarray(g)
    if r.shape != g.shape:
      return {"ok": False, "why": f"out{i} shape {g.shape} != {r.shape}", "max_rel": None}
    if np.issubdtype(r.dtype, np.floating):
      denom = max(float(np.max(np.abs(r))), 1e-30)
      rel = float(np.max(np.abs(g.astype(np.float64) - r.astype(np.float64)))) / denom
      worst = max(worst, rel)
      if not rel <= FLOAT_REL_BAR:
        return {"ok": False, "why": f"out{i} rel {rel:.3e} > {FLOAT_REL_BAR}", "max_rel": rel}
    else:
      if g.dtype != r.dtype and not (np.issubdtype(g.dtype, np.integer) and np.issubdtype(r.dtype, np.integer)):
        return {"ok": False, "why": f"out{i} dtype {g.dtype} != {r.dtype}", "max_rel": None}
      n_diff = int(np.sum(g.astype(np.int64) != r.astype(np.int64)))
      if n_diff:
        return {"ok": False, "why": f"out{i} {n_diff} integer entries differ", "max_rel": worst}
  return {"ok": True, "why": "", "max_rel": worst}


def _graph_facts(path: Path) -> dict[str, Any]:
  import onnx  # noqa: PLC0415

  model = onnx.load(str(path))
  ops: Counter[str] = Counter()

  def walk(graph: Any) -> None:
    for node in graph.node:
      ops[node.op_type] += 1
      for attr in node.attribute:
        if attr.g is not None and attr.g.node:
          walk(attr.g)
        for sub in attr.graphs:
          walk(sub)

  walk(model.graph)
  int64_io = [
    v.name for v in list(model.graph.input) + list(model.graph.output)
    if v.type.tensor_type.elem_type == onnx.TensorProto.INT64
  ]
  return {"bytes": path.stat().st_size, "watch_ops": {k: ops[k] for k in WATCH_OPS if ops[k]},
          "n_nodes": sum(ops.values()), "int64_io": int64_io}


def _run_web(cases: list[dict[str, Any]], work: Path, node_bin: str | None, ort_dir: str | None) -> dict[str, Any]:
  if not node_bin or not ort_dir or not cases:
    return {"available": False, "why": "node or onnxruntime-web dir not given", "results": {}}
  cases_path = work / "web_cases.json"
  out_path = work / "web_result.json"
  cases_path.write_text(json.dumps({"cases": cases}))
  runner = REPO / "browser" / "layer_c" / "run_onnx_io.mjs"
  proc = subprocess.run(  # noqa: S603 -- fixed argv we built
    [node_bin, str(runner), "--ort-dir", ort_dir, "--cases", str(cases_path), "--out", str(out_path)],
    capture_output=True, text=True, timeout=1800, check=False,
  )
  if proc.returncode != 0 or not out_path.exists():
    return {"available": False, "why": f"runner rc={proc.returncode}: {proc.stderr[-2000:]}", "results": {}}
  data = json.loads(out_path.read_text())
  return {"available": True, "why": "", "ort_web_version": data.get("ort_web_version"),
          "results": {r["id"]: r for r in data["results"]}}


def _read_web_outputs(record: dict[str, Any]) -> list[np.ndarray]:
  np_types = {"float32": np.float32, "int32": np.int32, "int64": np.int64, "bool": np.bool_, "uint8": np.uint8}
  return [np.fromfile(o["file"], dtype=np_types[o["dtype"]]).reshape(o["dims"]) for o in record["outputs"]]


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True, help="result JSON")
  parser.add_argument("--work-dir", type=Path, required=True, help="artifacts (onnx, raw io)")
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"),
                      help="directory containing node_modules/onnxruntime-web")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "versions": {}, "jax_arm_ok": False, "jax_enable_x64": False, "probes": {}, "n_probes": 0,
    "n_pass_cpu": 0, "n_pass_web": 0, "all_pass": False, "control_float_detected": False,
    "control_int_detected": False, "ort_web_available": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx imported before the JAX arm; the reference would use patched primitives"
      raise RuntimeError(msg)
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if result["jax_enable_x64"]:
      msg = "jax_enable_x64 is set; refusing (int64/f64 graphs are a different contract)"
      raise RuntimeError(msg)

    from aminx.families.potts_mpnn.model import PottsMPNN  # noqa: PLC0415

    model = PottsMPNN(key=jax.random.PRNGKey(args.seed))
    fx = _fixture(args.seed)
    probes = _build_probes(model, fx)

    # ---- JAX arm, entirely before any conversion -------------------------------------
    reference = {name: [np.asarray(x) for x in jax.jit(fn)(*ins)] for name, (fn, ins) in probes.items()}

    import equinox as eqx  # noqa: PLC0415

    noise_key = jax.random.PRNGKey(args.seed + 1)
    perturbed_head = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if _is_float_array(leaf) else leaf,
      model.potts_head,
    )
    perturbed = eqx.tree_at(lambda m: m.potts_head, model, perturbed_head)
    p_probes = _build_probes(perturbed, fx)
    p_score = [np.asarray(x) for x in jax.jit(p_probes["score"][0])(*p_probes["score"][1])]
    alt_inputs = list(probes["decode"][1])
    alt_inputs[11] = np.ascontiguousarray(fx["uniforms"][::-1])
    alt_decode = [np.asarray(x) for x in jax.jit(probes["decode"][0])(*alt_inputs)]
    result["control_float_detected"] = not compare(reference["score"], p_score)["ok"]
    result["control_int_detected"] = not compare(reference["decode"][:1], alt_decode[:1])["ok"]
    result["jax_arm_ok"] = True
    logger.info("JAX arm done in %.1fs", time.perf_counter() - t0)

    # ---- conversion + ORT-CPU ----------------------------------------------------------
    import xtrax  # noqa: PLC0415
    from xtrax.export import ONNX, convert_to_onnx, run_onnx  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

    web_cases: list[dict[str, Any]] = []
    for name, (fn, ins) in probes.items():
      entry: dict[str, Any] = {"converted": False, "cpu": "not_converted", "web": "not_converted",
                               "cpu_max_rel": None, "web_max_rel": None, "error": "", "graph": {}}
      result["probes"][name] = entry
      path = args.work_dir / f"potts_{name}_L{BUCKET}.onnx"
      try:
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in ins]
        convert_to_onnx(fn, specs, ONNX, out_path=path)
        embed_external_data(path)
        entry["converted"] = True
        entry["graph"] = _graph_facts(path)
      except Exception as exc:  # noqa: BLE001 -- record and continue to the next probe
        entry["error"] = f"convert: {type(exc).__name__}: {exc}"[:4000]
        continue
      try:
        got = run_onnx(path, *ins)
        cmp = compare(reference[name], got)
        entry["cpu"] = "pass" if cmp["ok"] else "mismatch"
        entry["cpu_max_rel"] = cmp["max_rel"]
        entry["cpu_why"] = cmp["why"]
      except Exception as exc:  # noqa: BLE001
        entry["cpu"] = "not_runnable"
        entry["error"] = f"ort-cpu: {type(exc).__name__}: {exc}"[:4000]
      io_dir = args.work_dir / f"io_{name}"
      io_dir.mkdir(exist_ok=True)
      specs_in = []
      for i, arr in enumerate(ins):
        arr = np.ascontiguousarray(arr)
        tag = DTYPE_TAG[arr.dtype]
        f = io_dir / f"in_{i}.bin"
        (arr.astype(np.uint8) if tag == "bool" else arr).tofile(f)
        specs_in.append({"file": str(f), "dtype": tag, "dims": list(arr.shape)})
      web_cases.append({"id": name, "model": str(path), "out_dir": str(io_dir), "inputs": specs_in})

    # ---- ORT-Web (wasm, Node) ----------------------------------------------------------
    web = _run_web(web_cases, args.work_dir, args.node_bin, args.ort_web_dir)
    result["ort_web_available"] = web["available"]
    result["ort_web_note"] = web["why"]
    for name, entry in result["probes"].items():
      if not entry["converted"]:
        continue
      rec = web["results"].get(name)
      if not web["available"] or rec is None:
        entry["web"] = "unavailable"
        continue
      if not rec["ok"]:
        entry["web"] = "not_runnable"
        entry["web_error"] = (rec.get("error") or "")[:4000]
        continue
      cmp = compare(reference[name], _read_web_outputs(rec))
      entry["web"] = "pass" if cmp["ok"] else "mismatch"
      entry["web_max_rel"] = cmp["max_rel"]
      entry["web_why"] = cmp["why"]
      entry["web_num_threads"] = rec.get("num_threads")

    import jax2onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    result["versions"] = {
      "jax": jax.__version__, "xtrax": xtrax.__version__, "jax2onnx": getattr(jax2onnx, "__version__", None),
      "onnxruntime": onnxruntime.__version__, "onnxruntime_web": web.get("ort_web_version"),
    }
    probes_out = result["probes"].values()
    result["n_probes"] = len(result["probes"])
    result["n_pass_cpu"] = sum(e["cpu"] == "pass" for e in probes_out)
    result["n_pass_web"] = sum(e["web"] == "pass" for e in probes_out)
    result["all_pass"] = result["n_pass_cpu"] == result["n_probes"] == result["n_pass_web"]
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("spike failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("probes: %s", {k: (v["cpu"], v["web"]) for k, v in result["probes"].items()})
  return 0


def _is_float_array(leaf: Any) -> bool:  # noqa: ANN401
  import equinox as eqx  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  return bool(eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating))


if __name__ == "__main__":
  raise SystemExit(main())
