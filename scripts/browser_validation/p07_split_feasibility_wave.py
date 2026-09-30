"""G0c: does the wave schedule + ar_mask convert to a standalone ONNX graph (Graph W)?

Graph W is the last unexported piece of the four-graph split
(`.praxia/docs/daily/260929_overnight-decisions.md` D1): it turns `decoding_order` and
`tie_group_map` into the wave schedule and the autoregressive mask, exactly as
``p07_bundle`` (``export/wrappers.py:340-374``) does today inside the monolith. It runs
ONCE PER SAMPLE, not per wave.

Why it matters that this exports. Graph W contains ``wave_from_decoding_order``'s
``jax.lax.sort`` (``wrappers.py:316-321``) -- one of only two index-producing sites in the
whole traced graph, and per the route research
(xtrax ``.praxia/docs/research/260914_browser-inference-routes-jaxjs-jax2onnx.md`` S5) the
more consequential one, because it determines the order residues are generated in: a
permutation there changes the whole design, not one neighbour list. The alternative to
exporting it is hand-porting a multi-key sort to JavaScript, which that same research names
as the likeliest way this route produces a green check over wrong output.

**Every output of this graph is discrete** -- group ids, group positions, validity masks,
and a 0/1 ar_mask. So this gate requires EXACT equality on all of them, not agreement
within a bound. A float tolerance would be the wrong instrument entirely: the failure mode
here is a permuted index, which is either right or wrong.

Decision record: D1/D2 in the overnight log. Sibling to G0 (E+D, run 8daaa978) and
G0b (F, run b8f413b6); it re-opens neither.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("p07_split_feasibility_wave")

_CODE_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")

_OUTPUT_NAMES = ("group_ids", "group_positions", "group_valid", "position_valid", "ar_mask")


def _git_state(repo: Path) -> tuple[str, bool, str]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo,
      capture_output=True,
      text=True,
      check=True,
      timeout=60,
    ).stdout.strip()

  sha = _run("rev-parse", "HEAD")
  dirty = _run("status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS)
  return sha, not dirty, dirty


def _synthetic_schedule(length: int, *, seed: int = 0) -> dict[str, np.ndarray]:
  """A decoding order and tie map that exercise the branches Graph W exists to carry.

  Deliberately NOT the identity/no-tie case, which would make the sort trivial and the
  wave schedule one-position-per-wave. Includes: a shuffled (non-identity) decoding order,
  several multi-member tie groups so waves genuinely carry more than one position, and
  unused group ids so the padded/inactive wave slots are exercised.
  """
  rng = np.random.default_rng(seed)
  decoding_order = rng.permutation(length).astype(np.int32)

  # Tie map: most positions are their own group; a few groups gather 2-3 members. Group
  # ids live in 0..L-1 and the unused ones must pad out to L waves.
  tie_group_map = np.arange(length, dtype=np.int32)
  tie_group_map[[5, 19, 61]] = 5  # 3-member group
  tie_group_map[[7, 40]] = 7  # 2-member group
  tie_group_map[[12, 33, 88, 90]] = 12  # 4-member group
  return {"decoding_order": decoding_order, "tie_group_map": tie_group_map}


def _fixture_stats(decoding_order: np.ndarray, tie_group_map: np.ndarray) -> dict[str, Any]:
  ids, counts = np.unique(tie_group_map, return_counts=True)
  return {
    "order_is_identity": bool(np.array_equal(decoding_order, np.arange(decoding_order.size))),
    "max_tie_group_size": int(counts.max()),
    "n_multi_member_groups": int((counts > 1).sum()),
    "n_unused_group_ids": int(decoding_order.size - ids.size),
  }


def _build_graph_w() -> Callable:
  """Graph W: exactly p07_bundle's schedule half, with no model and no bundle build."""
  import jax.numpy as jnp  # noqa: PLC0415
  from aminx.export.wrappers import wave_from_decoding_order  # noqa: PLC0415
  from aminx.utils.autoregression import generate_ar_mask  # noqa: PLC0415

  def graph_w(decoding_order: Any, tie_group_map: Any) -> tuple[Any, ...]:
    wave = wave_from_decoding_order(decoding_order, tie_group_map)
    ar_mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map).astype(jnp.float32)
    return (
      wave.group_ids,
      wave.group_positions,
      wave.group_valid,
      wave.position_valid,
      ar_mask,
    )

  return graph_w


def _convert(fn: Callable, arrays: list[np.ndarray], name: str, out_path: Path) -> None:
  import jax  # noqa: PLC0415
  import jax2onnx  # noqa: PLC0415

  from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

  specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  jax2onnx.to_onnx(fn, specs, model_name=name, output_path=str(out_path), return_mode="file")
  embed_external_data(out_path)


def _ort_run(onnx_path: Path, arrays: list[np.ndarray]) -> list[np.ndarray]:
  import onnxruntime as ort  # noqa: PLC0415

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  names = [i.name for i in session.get_inputs()]
  feeds = {n: np.asarray(a) for n, a in zip(names, arrays, strict=True)}
  return list(session.run(None, feeds))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--length", type=int, default=128)
  parser.add_argument("--work-dir", type=Path, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
  )

  repo = Path(__file__).resolve().parents[2]
  git_hash, git_clean, git_dirty_paths = _git_state(repo)
  work_dir = args.work_dir or (args.out.parent / "split_wave")
  work_dir.mkdir(parents=True, exist_ok=True)

  result: dict[str, Any] = {
    "length": args.length,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "git_dirty_paths": git_dirty_paths,
    "jax_enable_x64": False,
    "w_converted": False,
    "w_ran": False,
    "w_all_outputs_exact": False,
    "n_outputs_exact": 0,
    "n_outputs_total": len(_OUTPUT_NAMES),
    "mismatched_outputs": [],
    "controls_total": 2,
    "controls_detected": 0,
    "ctrl_index_perturb_detected": False,
    "ctrl_mask_flip_detected": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is True; the exported graph would be int64. Refusing."
      raise RuntimeError(msg)

    fixture = _synthetic_schedule(args.length)
    ordered = [fixture["decoding_order"], fixture["tie_group_map"]]
    result["fixture"] = _fixture_stats(*ordered)
    logger.info("fixture: %s", result["fixture"])

    graph_w = _build_graph_w()

    # JAX reference BEFORE importing jax2onnx (it patches jnp primitives at import with
    # ops that have no CPU MLIR lowering).
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx already imported; the JAX reference arm would use patched primitives."
      raise RuntimeError(msg)
    w_jax = [np.asarray(x) for x in jax.jit(graph_w)(*ordered)]
    logger.info("JAX reference arm complete")

    w_path = work_dir / f"p07_wave_L{args.length}.onnx"
    logger.info("converting Graph W (wave schedule + ar_mask) -> %s", w_path)
    _convert(graph_w, ordered, f"p07_wave_L{args.length}", w_path)
    result["w_converted"] = True
    result["w_onnx_bytes"] = w_path.stat().st_size

    w_onnx = _ort_run(w_path, ordered)
    result["w_ran"] = True

    mismatched = [
      name
      for name, j, o in zip(_OUTPUT_NAMES, w_jax, w_onnx, strict=True)
      if not np.array_equal(np.asarray(j), np.asarray(o))
    ]
    result["mismatched_outputs"] = mismatched
    result["n_outputs_exact"] = len(_OUTPUT_NAMES) - len(mismatched)
    result["w_all_outputs_exact"] = not mismatched
    logger.info(
      "Graph W: %d bytes, %d/%d outputs exact, mismatched=%s",
      result["w_onnx_bytes"],
      result["n_outputs_exact"],
      len(_OUTPUT_NAMES),
      mismatched,
    )

    # Controls. Exact-equality checks are pass-by-default against a self-comparison, so
    # plant a single-element index change and a single-element mask flip and require the
    # comparator to see each. These are the two output KINDS this graph produces.
    gp = np.array(w_onnx[1], copy=True)
    gp.flat[0] = gp.flat[0] + 1
    result["ctrl_index_perturb_detected"] = not np.array_equal(np.asarray(w_jax[1]), gp)

    pv = np.array(w_onnx[3], copy=True)
    pv.flat[0] = ~pv.flat[0] if pv.dtype == bool else (1 - pv.flat[0])
    result["ctrl_mask_flip_detected"] = not np.array_equal(np.asarray(w_jax[3]), pv)

    result["controls_detected"] = int(result["ctrl_index_perturb_detected"]) + int(
      result["ctrl_mask_flip_detected"]
    )
    logger.info("controls: %d/2", result["controls_detected"])

  except Exception as exc:  # noqa: BLE001 - a conversion refusal IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G0c probe failed")

  fx = result.get("fixture") or {}
  fixture_ok = bool(
    fx
    and not fx.get("order_is_identity", True)
    and fx.get("max_tie_group_size", 0) >= 2
    and fx.get("n_unused_group_ids", 0) > 0
  )
  result["fixture_ok"] = fixture_ok

  result["outcome"] = (
    "incomplete"
    if result["jax_enable_x64"]
    or not (result["w_converted"] and result["w_ran"])
    or not result["git_clean"]
    else "fixture_blind"
    if not fixture_ok
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["w_all_outputs_exact"]
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
