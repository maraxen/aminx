"""Export the four production split graphs (E, W, D, F) plus a sha256 manifest.

This is the artifact-producing counterpart to the G0 feasibility family. Those probes
answered "can each piece be exported at all" (G0 run 8daaa978, G0b b8f413b6, G0c a7f62e90);
this script emits the graphs the browser actually ships and records their hashes so a
downstream gate can prove which bytes it measured.

The cut (`.praxia/docs/specs/260929_p07-split-export.md` S2, decision D1):

    E  encoder          once per structure
    W  wave schedule    once per sample
    D  decoder step     once per wave
    F  fuse + sample    once per wave

Graph W here carries TWO outputs beyond `p07_bundle`'s own: `group_first_rank` (the
scatter-min at ``autoregressive.py:445-462``) and `pos_first_rank` (``:813``). Both depend
only on W's inputs, so emitting them removes the last nontrivial index computation from
the JavaScript side.

**A note on the loop's real shape, measured here rather than assumed.**
``wave_from_decoding_order`` returns ``group_ids`` of shape ``(L, 1)``, so
``max_groups_per_wave == 1`` and ``n_waves == L``: the driver makes **L sequential decoder
invocations**, each a full pass over all L positions. That is the O(L^2 k) full-recompute
arm (``mode.py`` docstring), and it is the cost G4 has to measure rather than predict. The
manifest records both numbers so no downstream reader has to infer them.

Export determinism: repeated runs at the same commit produce the same bytes, which is what
makes the manifest meaningful. This script does not assert that -- ``--verify-manifest``
re-hashes an existing manifest so a caller can check it explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
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

logger = logging.getLogger("p07_split_export")

BUCKETS = (128, 256)
N_TOKENS = 21


def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _git_sha(repo: Path) -> str:
  return subprocess.run(  # noqa: S603
    ["git", "rev-parse", "HEAD"],  # noqa: S607
    cwd=repo,
    capture_output=True,
    text=True,
    check=True,
    timeout=60,
  ).stdout.strip()


def _helix_backbone(length: int, *, seed: int = 0) -> dict[str, np.ndarray]:
  """Shape-only tracing fixture. Values never reach the exported weights."""
  rng = np.random.default_rng(seed)
  t = np.arange(length, dtype=np.float64)
  angle = np.deg2rad(100.0) * t
  ca = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * t], axis=-1)
  offsets = np.array([[-1.2, 0.3, -0.4], [0.0, 0.0, 0.0], [1.2, -0.3, 0.4], [1.9, 0.6, 1.0]])
  coords = ca[:, None, :] + offsets[None, :, :] + rng.normal(scale=0.02, size=(length, 4, 3))
  return {
    "coords": coords.astype(np.float32),
    "mask": np.ones(length, dtype=np.float32),
    "residue_index": np.arange(length, dtype=np.int32),
    "chain_index": np.zeros(length, dtype=np.int32),
  }


def _build_graphs(length: int) -> tuple[dict[str, Callable], dict[str, Any]]:
  """The four graphs, plus the shape facts a caller needs to feed them."""
  import jax.numpy as jnp  # noqa: PLC0415
  from aminx.export.wrappers import (  # noqa: PLC0415
    EXPORT_TOP_K_ROW_CHUNK,
    wave_from_decoding_order,
    zero_dropout,
  )
  from aminx.inference.decode._kernel import _decode_one_step, _project_logits  # noqa: PLC0415
  from aminx.inference.decode.autoregressive import _fuse_and_sample  # noqa: PLC0415
  from aminx.inference.logits import make_stage_set  # noqa: PLC0415, TID251
  from aminx.io.weights import load_model  # noqa: PLC0415
  from aminx.model.features import select_neighbors  # noqa: PLC0415
  from aminx.utils.autoregression import generate_ar_mask  # noqa: PLC0415
  from aminx.utils.coordinates import (  # noqa: PLC0415
    compute_backbone_coordinates,
    compute_backbone_distance,
  )
  from aminx.utils.radial_basis import compute_radial_basis  # noqa: PLC0415

  model = load_model(checkpoint_id="proteinmpnn_v_48_020")
  _, dropout_stats = zero_dropout(model)
  stage_set = make_stage_set()

  def graph_e(coords: Any, mask: Any, residue_index: Any, chain_index: Any) -> tuple[Any, ...]:
    backbone = compute_backbone_coordinates(coords)
    distances = compute_backbone_distance(backbone)
    neighbor_indices = select_neighbors(
      distances,
      mask,
      model.features.k_neighbors,
      row_chunk=EXPORT_TOP_K_ROW_CHUNK,
    )
    rbf = compute_radial_basis(backbone, neighbor_indices)
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
      jnp.asarray(node_features, jnp.float32),
      jnp.asarray(edge_features, jnp.float32),
      jnp.asarray(stages.neighbor_indices, jnp.int32),
    )

  def graph_w(decoding_order: Any, tie_group_map: Any) -> tuple[Any, ...]:
    wave = wave_from_decoding_order(decoding_order, tie_group_map)
    ar_mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map).astype(jnp.float32)

    # group_first_rank / pos_first_rank, transcribed from autoregressive.py:445-462,813.
    n_waves, g_per_wave = wave.group_ids.shape
    pos0_grid = wave.group_positions[:, :, 0]
    real_group_id_grid = tie_group_map[pos0_grid]
    wave_index_grid = jnp.broadcast_to(
      jnp.arange(n_waves, dtype=jnp.int32)[:, None],
      (n_waves, g_per_wave),
    )
    slot_grid = jnp.broadcast_to(
      jnp.arange(g_per_wave, dtype=jnp.int32)[None, :],
      (n_waves, g_per_wave),
    )
    combined_rank_grid = wave_index_grid * g_per_wave + slot_grid
    sentinel = n_waves * g_per_wave
    flat_group_id = jnp.where(wave.group_valid, real_group_id_grid, 0).reshape(-1)
    flat_rank = jnp.where(wave.group_valid, combined_rank_grid, sentinel).reshape(-1)
    length = tie_group_map.shape[0]
    group_first_rank = (
      jnp.full((length,), sentinel, dtype=jnp.int32).at[flat_group_id].min(flat_rank)
    )
    pos_first_rank = group_first_rank[tie_group_map]

    return (
      wave.group_ids,
      wave.group_positions,
      wave.group_valid,
      wave.position_valid,
      ar_mask,
      group_first_rank,
      pos_first_rank,
    )

  def graph_d(
    node_features: Any,
    edge_features: Any,
    neighbor_indices: Any,
    mask: Any,
    ar_mask: Any,
    sequence_oh: Any,
  ) -> Any:
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
    return jnp.asarray(_project_logits(model, decoded[None, ...])[0], jnp.float32)

  def graph_f(
    logits: Any,
    cond_bias: Any,
    mask_group: Any,
    fixed_mask: Any,
    fixed_tokens: Any,
    group_id: Any,
    temperature: Any,
    gumbel_noise: Any,
  ) -> tuple[Any, Any]:
    return _fuse_and_sample(
      logits=logits,
      cond_bias=cond_bias,
      mask_group=mask_group,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
      group_id=group_id,
      key=None,
      stage_set=stage_set,
      temperature=temperature,
      gumbel_noise=gumbel_noise,
    )

  graphs = {"encoder": graph_e, "wave": graph_w, "decoder": graph_d, "fuse": graph_f}
  return graphs, {"dropout_stats": dict(dropout_stats), "k_neighbors": model.features.k_neighbors}


def _convert(fn: Callable, arrays: list[np.ndarray], name: str, out_path: Path) -> None:
  import jax  # noqa: PLC0415
  import jax2onnx  # noqa: PLC0415

  from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

  specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  jax2onnx.to_onnx(fn, specs, model_name=name, output_path=str(out_path), return_mode="file")
  embed_external_data(out_path)


def _probe_bucket(bucket: int) -> dict[str, Any]:
  """Run the JAX shape probes for one bucket. MUST happen before any conversion.

  jax2onnx patches jnp primitives at import with ops that have no CPU MLIR lowering, and
  the patch is global and permanent for the process. So every bucket's probes must run
  before the FIRST conversion, not merely before its own -- exporting bucket A and then
  probing bucket B would trace B against patched primitives. The caller enforces that
  ordering; the guard below is the backstop that caught this exact bug.
  """
  import jax  # noqa: PLC0415

  if jax.config.jax_enable_x64:
    msg = "jax_enable_x64 is True; refusing to export an int64/float64 graph."
    raise RuntimeError(msg)
  if "jax2onnx" in sys.modules:
    msg = "jax2onnx already imported; shape probes would use patched primitives."
    raise RuntimeError(msg)

  geom = _helix_backbone(bucket)
  graphs, facts = _build_graphs(bucket)

  e_in = [geom["coords"], geom["mask"], geom["residue_index"], geom["chain_index"]]
  node_features, edge_features, neighbor_indices = (
    np.asarray(x) for x in jax.jit(graphs["encoder"])(*e_in)
  )

  decoding_order = np.arange(bucket, dtype=np.int32)
  tie_group_map = np.arange(bucket, dtype=np.int32)
  w_in = [decoding_order, tie_group_map]
  w_out = [np.asarray(x) for x in jax.jit(graphs["wave"])(*w_in)]
  group_ids, _gp, _gv, _pv, ar_mask = w_out[0], w_out[1], w_out[2], w_out[3], w_out[4]
  n_waves, g_per_wave = group_ids.shape

  sequence_oh = np.zeros((bucket, N_TOKENS), dtype=np.float32)
  d_in = [
    node_features,
    edge_features,
    neighbor_indices,
    geom["mask"],
    np.asarray(ar_mask, np.float32),
    sequence_oh,
  ]
  logits = np.asarray(jax.jit(graphs["decoder"])(*d_in))

  f_in = [
    logits[None, ...].astype(np.float32),
    np.zeros((bucket, N_TOKENS), np.float32),
    np.zeros((g_per_wave, bucket), bool),
    np.zeros(bucket, np.float32),
    np.zeros(bucket, np.int32),
    np.zeros(g_per_wave, np.int32),
    np.float32(0.1),
    np.zeros((bucket, N_TOKENS), np.float32),
  ]

  return {
    "bucket": bucket,
    "n_waves": int(n_waves),
    "max_groups_per_wave": int(g_per_wave),
    "decoder_calls_per_design": int(n_waves),
    "k_neighbors": int(facts["k_neighbors"]),
    "hidden": int(node_features.shape[-1]),
    "dropout_stats": facts["dropout_stats"],
    "specs": {
      "encoder": (graphs["encoder"], e_in),
      "wave": (graphs["wave"], w_in),
      "decoder": (graphs["decoder"], d_in),
      "fuse": (graphs["fuse"], f_in),
    },
  }


def _convert_bucket(probe: dict[str, Any], out_dir: Path) -> dict[str, Any]:
  """Convert one bucket's four graphs. Safe to run after jax2onnx is imported."""
  bucket = probe["bucket"]
  entries: dict[str, Any] = {}
  for key, (fn, arrays) in probe["specs"].items():
    path = out_dir / f"p07_{key}_L{bucket}.onnx"
    logger.info("exporting %s L=%d -> %s", key, bucket, path.name)
    _convert(fn, arrays, f"p07_{key}_L{bucket}", path)
    entries[key] = {
      "file": path.name,
      "bytes": path.stat().st_size,
      "sha256": _sha256(path),
      "input_shapes": [list(a.shape) for a in arrays],
      "input_dtypes": [str(a.dtype) for a in arrays],
    }
    logger.info("  %d bytes, sha256 %s", entries[key]["bytes"], entries[key]["sha256"][:16])

  return {k: v for k, v in probe.items() if k != "specs"} | {"graphs": entries}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=None, help="result JSON for the sidecar")
  parser.add_argument("--buckets", type=int, nargs="+", default=[128])
  parser.add_argument("--manifest", type=Path, default=None)
  parser.add_argument(
    "--verify-manifest",
    type=Path,
    default=None,
    help="re-hash the files named by an existing manifest and report drift; exports nothing",
  )
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
  )

  if args.verify_manifest is not None:
    manifest = json.loads(args.verify_manifest.read_text())
    drift = []
    for bucket in manifest["buckets"]:
      for key, entry in bucket["graphs"].items():
        path = args.out_dir / entry["file"]
        actual = _sha256(path) if path.exists() else None
        if actual != entry["sha256"]:
          drift.append({"graph": key, "file": entry["file"], "expected": entry["sha256"],
                        "actual": actual})
    print(json.dumps({"drift": drift, "ok": not drift}, indent=2))  # noqa: T201
    return 1 if drift else 0

  args.out_dir.mkdir(parents=True, exist_ok=True)
  git_hash = _git_sha(_REPO_ROOT)

  result: dict[str, Any] = {
    "buckets_requested": len(args.buckets),
    "buckets_exported": 0,
    "graphs_per_bucket": 4,
    "all_graphs_exported": False,
    "manifest_written": False,
    "manifest_sha256": "",
    "n_waves": -1,
    "max_groups_per_wave": -1,
    "decoder_calls_per_design": -1,
    "total_bytes": 0,
    "git_hash": git_hash,
    "jax_enable_x64": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)

    # ALL JAX shape probes first, across every bucket, then all conversions. jax2onnx's
    # jnp patch is global and irreversible once imported, so interleaving probe/convert
    # per bucket traces the second bucket against patched primitives. Caught by the
    # guard in _probe_bucket on the first two-bucket run.
    probes = [_probe_bucket(b) for b in args.buckets]
    exported = [_convert_bucket(p, args.out_dir) for p in probes]
    manifest = {
      "git_hash": git_hash,
      "checkpoint_id": "proteinmpnn_v_48_020",
      "alphabet": "ACDEFGHIKLMNPQRSTVWYX",
      "buckets": exported,
    }
    manifest_path = args.manifest or (args.out_dir / "MANIFEST.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("manifest -> %s", manifest_path)

    result["buckets_exported"] = len(exported)
    result["all_graphs_exported"] = all(len(b["graphs"]) == 4 for b in exported) and len(
      exported
    ) == len(args.buckets)
    result["manifest_written"] = manifest_path.exists()
    result["manifest_sha256"] = _sha256(manifest_path)
    result["total_bytes"] = sum(
      g["bytes"] for b in exported for g in b["graphs"].values()
    )
    # Reported for the LARGEST bucket exported: it is the binding cost case.
    largest = max(exported, key=lambda b: b["bucket"])
    result["n_waves"] = largest["n_waves"]
    result["max_groups_per_wave"] = largest["max_groups_per_wave"]
    result["decoder_calls_per_design"] = largest["decoder_calls_per_design"]

    for bucket in exported:
      logger.info(
        "L=%d: %d waves x %d slot(s) => %d decoder calls per design",
        bucket["bucket"],
        bucket["n_waves"],
        bucket["max_groups_per_wave"],
        bucket["decoder_calls_per_design"],
      )
  except Exception as exc:  # noqa: BLE001 - an export failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("split export failed")

  result["outcome"] = (
    "pass"
    if result["all_graphs_exported"]
    and result["manifest_written"]
    and not result["jax_enable_x64"]
    else "incomplete"
  )
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s", result["outcome"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
