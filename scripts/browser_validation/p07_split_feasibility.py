"""G0: can jax2onnx convert the P07 encoder and per-step decoder as STANDALONE graphs?

The shipped P07 export (``make_p07_sample``) keeps the whole autoregressive loop inside
ONNX, and -- because ``AutoregressiveConfig.incremental`` defaults to ``"auto"`` -- it also
carries BOTH decoder arms behind ``jax.lax.cond(can_increment, run_incremental, run_full)``
(``inference/decode/autoregressive.py:809``). The split design
(``.praxia/docs/specs/260929_p07-split-export.md``) proposes exporting two loop-free graphs
instead and driving the loop from JavaScript.

This script is gate **G0** of that spec: a binary feasibility probe. It answers only
"do the two graphs convert, load, run, and agree with JAX on one fixture", NOT "are they
parity-exact" -- that is G1, which sets its own tight bound. G0's bound is deliberately
loose (see ``--agree-bound``); its job is to catch "converted but wrong", not to certify
parity.

Instrument controls (both must fire, or the run is ``ctrl_blind``):

* ``ctrl_nan``      -- a non-finite value injected into a graph output must be caught by
                       the finiteness check.
* ``ctrl_perturb``  -- a known offset added to an ONNX output must push the JAX-agreement
                       comparison over its bound.

A positive control that can only pass is not a check (``rules/BATHOS.md``).
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

# Run as a file (`python scripts/browser_validation/p07_split_feasibility.py`), sys.path[0]
# is this directory, so the `scripts.browser_validation.*` import below cannot resolve.
# `scripts/` has no __init__.py but `scripts/browser_validation/` does, so it resolves as a
# namespace package once the repo root is on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("p07_split_feasibility")

# Pre-registered: G0 is a feasibility gate, not the parity gate. 1e-3 max-abs is loose on
# purpose -- the whole-graph knobs gate already measures ORT-CPU vs JAX at 2e-4 on
# log-probs, and these subgraphs contain strictly fewer ops, so anything near this bound
# means a conversion defect rather than float noise. G1 sets the tight bound.
DEFAULT_AGREE_BOUND = 1e-3

# Pre-registered: large enough that no plausible float32 discrepancy reaches it, so the
# control proves the comparator fires rather than proving the graph is bad.
CTRL_PERTURB_MAGNITUDE = 1.0


# The paths whose cleanliness actually determines whether this run reproduces from git.
# Mirrors titanix_launch.sh's own rationale: the worktree permanently carries harness dirt
# (.praxia/*, .claude/*, .bth/refs/, *.ses) that "must never be committed or deleted", so a
# bare `git status --porcelain` can never be empty here and would gate every run to
# `incomplete` for a reason unrelated to the measurement.
_CODE_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")


def _git_state(repo: Path) -> tuple[str, bool, str]:
  """Return ``(HEAD sha, code paths clean, dirty listing)`` for ``repo``.

  Cleanliness is scoped to ``_CODE_PATHS`` and includes untracked files there, so an
  uncommitted source change still blocks the run while harness churn does not. The raw
  listing is recorded either way, so a reader can see exactly what was dirty.
  """

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


def _synthetic_backbone(length: int, *, seed: int = 0) -> dict[str, np.ndarray]:
  """A plausible alpha-helical backbone of ``length`` residues.

  G0 tests graph CONVERSION, not science, so a synthetic backbone is adequate and keeps
  the probe free of fixture-path dependencies. It is helical rather than random so the
  k-nearest-neighbour selection sees a realistic, non-degenerate neighbourhood structure.
  Atom order is N, CA, C, O -- the compact ``(L, 4, 3)`` convention documented in
  ``src/aminx/utils/coordinates.py``.
  """
  rng = np.random.default_rng(seed)
  t = np.arange(length, dtype=np.float64)
  # Ideal alpha helix: 1.5 A rise, 100 deg twist, 2.3 A radius.
  angle = np.deg2rad(100.0) * t
  ca = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * t], axis=-1)
  # Offset the other three backbone atoms off CA by small fixed vectors plus jitter, so
  # N/CA/C/O are distinct but the chain stays physically plausible.
  offsets = np.array([[-1.2, 0.3, -0.4], [0.0, 0.0, 0.0], [1.2, -0.3, 0.4], [1.9, 0.6, 1.0]])
  coords = ca[:, None, :] + offsets[None, :, :]
  coords = coords + rng.normal(scale=0.02, size=coords.shape)
  return {
    "coords": coords.astype(np.float32),
    "mask": np.ones(length, dtype=np.float32),
    "residue_index": np.arange(length, dtype=np.int32),
    "chain_index": np.zeros(length, dtype=np.int32),
  }


def _build_graphs(length: int) -> tuple[Callable, Callable, dict[str, Any]]:
  """Build the standalone encoder and per-step decoder as plain JAX callables.

  Mirrors ``make_p07_sample``'s encoder phase (``export/wrappers.py:425-455``) and the
  decoder step body (``inference/decode/_kernel.py:16-110``) exactly, but with the scan,
  the cond, and the sampling logic removed -- those move to JavaScript under the split.
  """
  import jax.numpy as jnp  # noqa: PLC0415
  from aminx.export.wrappers import EXPORT_TOP_K_ROW_CHUNK, zero_dropout  # noqa: PLC0415
  from aminx.inference.decode._kernel import _decode_one_step, _project_logits  # noqa: PLC0415
  from aminx.io.weights import load_model  # noqa: PLC0415
  from aminx.model.features import select_neighbors  # noqa: PLC0415
  from aminx.utils.coordinates import (  # noqa: PLC0415
    compute_backbone_coordinates,
    compute_backbone_distance,
  )
  from aminx.utils.radial_basis import compute_radial_basis  # noqa: PLC0415

  model = load_model(checkpoint_id="proteinmpnn_v_48_020")
  _, dropout_stats = zero_dropout(model)

  def graph_e(
    coords: Any,
    mask: Any,
    residue_index: Any,
    chain_index: Any,
  ) -> tuple[Any, Any, Any]:
    """Encoder: runs once per structure. No RNG, no scan, no cond."""
    backbone_coords = compute_backbone_coordinates(coords)
    distances = compute_backbone_distance(backbone_coords)
    neighbor_indices = select_neighbors(
      distances,
      mask,
      model.features.k_neighbors,
      row_chunk=EXPORT_TOP_K_ROW_CHUNK,
    )
    rbf = compute_radial_basis(backbone_coords, neighbor_indices)
    stages = model.features.forward_edge_stages(
      None,
      coords,
      mask,
      residue_index,
      chain_index,
      None,
      rbf_features=rbf,
      neighbor_indices=neighbor_indices,
    )
    node_features, edge_features = model.encoder(
      stages.final,
      stages.neighbor_indices,
      mask,
      key=None,
    )
    return (
      jnp.asarray(node_features, dtype=jnp.float32),
      jnp.asarray(edge_features, dtype=jnp.float32),
      jnp.asarray(stages.neighbor_indices, dtype=jnp.int32),
    )

  def graph_d(
    node_features: Any,
    edge_features: Any,
    neighbor_indices: Any,
    mask: Any,
    ar_mask: Any,
    sequence_oh: Any,
  ) -> Any:
    """One decoder step: all L rows, current sequence. No RNG, no scan, no cond.

    Of these six inputs only ``sequence_oh`` changes between waves -- ``ar_mask`` is
    constant for a whole sample (``autoregressive.py:516`` reads ``cond.ar_mask``, built
    once outside the scan), and the other four are encoder outputs.
    """
    decoded = _decode_one_step(
      model=model,
      node_features=node_features,
      edge_features=edge_features,
      neighbor_indices=neighbor_indices,
      mask=mask,
      ar_mask=ar_mask,
      sequence_oh=sequence_oh,
      key=None,
      inference=True,
    )
    return jnp.asarray(_project_logits(model, decoded), dtype=jnp.float32)

  return graph_e, graph_d, dict(dropout_stats)


def _convert(fn: Callable, arrays: list[np.ndarray], name: str, out_path: Path) -> None:
  """Convert ``fn`` to a self-contained ONNX file at ``out_path``."""
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
  if len(names) != len(arrays):
    msg = f"{onnx_path.name}: graph takes {len(names)} inputs, got {len(arrays)}"
    raise ValueError(msg)
  feeds = {name: np.asarray(a) for name, a in zip(names, arrays, strict=True)}
  return list(session.run(None, feeds))


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True, help="result JSON path")
  parser.add_argument("--bucket", type=int, default=128, help="sequence length bucket")
  parser.add_argument("--work-dir", type=Path, default=None, help="where to write .onnx files")
  parser.add_argument("--agree-bound", type=float, default=DEFAULT_AGREE_BOUND)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
  )

  repo = Path(__file__).resolve().parents[2]
  git_hash, git_clean, git_dirty_paths = _git_state(repo)
  work_dir = args.work_dir or (args.out.parent / f"split_L{args.bucket}")
  work_dir.mkdir(parents=True, exist_ok=True)

  result: dict[str, Any] = {
    "bucket": args.bucket,
    "agree_bound": args.agree_bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "git_dirty_paths": git_dirty_paths,
    "e_converted": False,
    "d_converted": False,
    "e_ran": False,
    "d_ran": False,
    "e_finite": False,
    "d_finite": False,
    "e_max_abs_diff": float("nan"),
    "d_max_abs_diff": float("nan"),
    "controls_total": 2,
    "controls_detected": 0,
    "ctrl_nan_detected": False,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from aminx.utils.autoregression import generate_ar_mask  # noqa: PLC0415

    length = args.bucket
    geom = _synthetic_backbone(length)
    graph_e, graph_d, dropout_stats = _build_graphs(length)
    result["dropout_stats"] = dropout_stats

    # ---- JAX reference arm: MUST run before jax2onnx is imported ---------------
    # jax2onnx monkey-patches jnp.stack (and friends) at import time with primitives
    # that have no MLIR lowering for the CPU platform, so a plain jax.jit of any code
    # using them dies with "MLIR translation rule for primitive 'jax.numpy.stack' not
    # found for platform cpu" once it is loaded. Measured 2026-09-29: converting first
    # and evaluating second poisoned this script's own reference arm. Evaluate every
    # JAX output first, materialise it to numpy, and only then convert.
    if "jax2onnx" in sys.modules:
      msg = (
        "jax2onnx is already imported; the JAX reference arm would run against its "
        "patched primitives. Evaluate JAX before converting."
      )
      raise RuntimeError(msg)

    e_inputs = [geom["coords"], geom["mask"], geom["residue_index"], geom["chain_index"]]
    e_jax = [np.asarray(x) for x in jax.jit(graph_e)(*e_inputs)]

    # ar_mask is (L, L) and CONSTANT for a whole sample; build it from a decoding order.
    decoding_order = jnp.arange(length, dtype=jnp.int32)
    ar_mask = np.asarray(generate_ar_mask(decoding_order), dtype=np.float32)
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 21, size=length)
    sequence_oh = np.eye(21, dtype=np.float32)[tokens]

    d_inputs = [
      np.asarray(e_jax[0], np.float32),
      np.asarray(e_jax[1], np.float32),
      np.asarray(e_jax[2], np.int32),
      geom["mask"],
      ar_mask,
      sequence_oh,
    ]
    d_jax = np.asarray(jax.jit(graph_d)(*d_inputs))
    logger.info("JAX reference arm complete; importing jax2onnx now")

    # ---- Graph E ---------------------------------------------------------------
    e_path = work_dir / f"p07_encoder_L{length}.onnx"
    logger.info("converting Graph E (encoder) -> %s", e_path)
    _convert(graph_e, e_inputs, f"p07_encoder_L{length}", e_path)
    result["e_converted"] = True
    result["e_onnx_bytes"] = e_path.stat().st_size

    e_onnx = _ort_run(e_path, e_inputs)
    result["e_ran"] = True
    result["e_finite"] = all(bool(np.all(np.isfinite(x))) for x in e_onnx[:2])
    result["e_max_abs_diff"] = max(
      _max_abs_diff(j, o) for j, o in zip(e_jax[:2], e_onnx[:2], strict=True)
    )
    result["e_neighbor_exact"] = bool(np.array_equal(e_jax[2], e_onnx[2]))
    logger.info(
      "Graph E: %d bytes, max_abs_diff=%.3e, neighbors_exact=%s",
      result["e_onnx_bytes"],
      result["e_max_abs_diff"],
      result["e_neighbor_exact"],
    )

    # ---- Graph D ---------------------------------------------------------------
    d_path = work_dir / f"p07_decoder_step_L{length}.onnx"
    logger.info("converting Graph D (decoder step) -> %s", d_path)
    _convert(graph_d, d_inputs, f"p07_decoder_step_L{length}", d_path)
    result["d_converted"] = True
    result["d_onnx_bytes"] = d_path.stat().st_size

    d_onnx = _ort_run(d_path, d_inputs)[0]
    result["d_ran"] = True
    result["d_finite"] = bool(np.all(np.isfinite(d_onnx)))
    result["d_max_abs_diff"] = _max_abs_diff(d_jax, d_onnx)
    logger.info(
      "Graph D: %d bytes, max_abs_diff=%.3e",
      result["d_onnx_bytes"],
      result["d_max_abs_diff"],
    )

    # ---- Controls ------------------------------------------------------------
    # (1) finiteness detector must fire on an injected non-finite value.
    poisoned = np.array(d_onnx, copy=True)
    poisoned.flat[0] = np.nan
    result["ctrl_nan_detected"] = not bool(np.all(np.isfinite(poisoned)))

    # (2) agreement comparator must fire on a known offset.
    offset = np.array(d_onnx, copy=True)
    offset.flat[0] += CTRL_PERTURB_MAGNITUDE
    result["ctrl_perturb_detected"] = _max_abs_diff(d_jax, offset) > args.agree_bound

    result["controls_detected"] = int(result["ctrl_nan_detected"]) + int(
      result["ctrl_perturb_detected"]
    )
    logger.info(
      "controls: nan=%s perturb=%s (%d/2)",
      result["ctrl_nan_detected"],
      result["ctrl_perturb_detected"],
      result["controls_detected"],
    )

  except Exception as exc:  # noqa: BLE001 - a conversion failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G0 probe failed")

  bound = args.agree_bound
  result["within_bound"] = bool(
    result["e_ran"]
    and result["d_ran"]
    and result["e_max_abs_diff"] <= bound
    and result["d_max_abs_diff"] <= bound
  )
  result["outcome"] = (
    "incomplete"
    if not (result["e_converted"] and result["d_converted"] and result["e_ran"] and result["d_ran"])
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["within_bound"] and result["e_finite"] and result["d_finite"]
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
