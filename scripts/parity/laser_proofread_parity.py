"""Graded LASEr proofread parity (do not run from the fixer).

Five complexes, injected scalar-dropout masks, ``n_orders=2``, ``n_dropouts=1``.
Pass is max |delta mean| and max |delta std| <= 1e-4. Inconclusive is
(1e-4, 1e-3]. Fail is > 1e-3. Bands are the spec §7.3 line 1181
pre-registration.

Negative controls, each of which must land in fail:
  reduction swap (std over reps)
  ddof=0
  scalar dropout off
  vector dropout on

NEVER append ``scalar_both_off`` to a graded run's ``--mutants``. Redsox step
1c compares the RAW argv string against this slug's manifest rows by exact set
equality (``tests/knob_gate/_coverage.py:309``), and the positive control has
no manifest row -- every manifest row must report ``failed``, which a control
that must pass never will. This script drops the positive arm from
``n_listed``, ``n_failed`` and the controls ``mutants`` dict, but it cannot
rewrite its own argv, so appending it silently costs the run its ledger
eligibility. Run it as its own separate, ungraded invocation.

Positive control ``scalar_both_off`` (optional, not a negative, not part of
the pre-registered outcomes): upstream ``nn.Dropout`` stays in eval and the
aminx plan uses ``scalar=False``. It must report near-zero. A non-zero result
means instrument floor rather than a port defect. Run it on its own with
``--mutants scalar_both_off`` and no ``--controls-out``; it is excluded from
``n_listed`` and ``n_failed``, and must not ride along on a graded run.

The clean arm and each mutant arm are separate subprocesses. A missing
``$BTH_RESULTS_PATH`` is fatal.
"""

from __future__ import annotations

import os
import pathlib

# Only when run as a script. At module level it also fired when a test imported this module
# for constants, leaking x64 into every subprocess that pytest worker spawned afterwards
# (aminx #165 CI: a benchmark smoke test built an f64 skeleton for f32 weights).
if __name__ == "__main__":
  os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The parity dir on sys.path mirrors the convention in the potts scripts, so
# the shared stable_artifact_key helper imports at module scope rather than
# depending on the sys.path insert that main() does later.
_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

from artifact_key import stable_artifact_key

MUTANTS = ("reduction_swap", "ddof_0", "scalar_off", "vector_on")
# Optional arm. Not a member of MUTANTS: those four must fail, and this one
# must agree. Counting it in n_listed would change the pre-registered outcomes.
POSITIVE_ARM = "scalar_both_off"
PASS_MAX = 1e-4
INCONCLUSIVE_MAX = 1e-3
CHECKPOINT_NAME = "laser_weights_0p1A_nothing_heldout.pt"
CHECKPOINT_SHA256 = "304fe02a4807c310bdd9d68c988ae87619da3cf2025d5c223fb31030aa411173"
EXAMPLE_SHA256 = "0a933729c4915cfee14ff34408a57e1d65fd98845df189ba79ad4f4d8470ae7e"
EXAMPLE_NAME = "4jnj-1_prot.pdb"
N_COMPLEXES = 5


def _repo() -> Path:
  return Path(__file__).resolve().parents[2]


def _default_laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=_default_laser_root())
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--oracle-python", default=os.environ.get("LASER_ORACLE_PYTHON"))
  parser.add_argument("--mutants", default=",".join(MUTANTS))
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--upstream", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--arm-timeout", type=float, default=86400.0)
  parser.add_argument("--oracle-worker", action="store_true")
  parser.add_argument(
    "--scalar-eval",
    action="store_true",
    help="oracle only: leave nn.Dropout in eval and write the positive-control reference",
  )
  return parser.parse_args(argv)


@dataclass(frozen=True, slots=True)
class ArmPlan:
  """How one proofread arm reduces, and whether upstream dropout stays off."""

  scalar: bool
  vector: bool
  ddof: int
  std_over: str
  scalar_eval: bool


def scalar_both_off_plan() -> ArmPlan:
  """Positive control: both sides have scalar dropout genuinely disabled.

  Upstream ``nn.Dropout`` modules are left in eval (not switched to train)
  and the aminx plan uses ``scalar=False``, so the proofread computation
  reduces to the plain conditional decode.
  This arm must report near-zero max_abs_mean and max_abs_std.
  A non-zero result means instrument floor rather than a port defect.
  """
  return ArmPlan(
    scalar=False,
    vector=False,
    ddof=1,
    std_over="orders",
    scalar_eval=True,
  )


def arm_plan(arm: str | None) -> ArmPlan:
  """Reduction and dropout flags for ``arm``.

  ``scalar_off`` disables only the aminx plan. Upstream for that arm is still
  the train-mode reference, which is why it is a negative control.
  """
  if arm == POSITIVE_ARM:
    return scalar_both_off_plan()
  return ArmPlan(
    scalar=arm != "scalar_off",
    vector=arm == "vector_on",
    ddof=0 if arm == "ddof_0" else 1,
    std_over="reps" if arm == "reduction_swap" else "orders",
    scalar_eval=False,
  )


def split_mutants(raw: str) -> tuple[list[str], bool]:
  """Negative controls, and whether the positive-control arm was requested.

  The positive arm is not returned in the negative list, so ``n_listed`` stays
  the count of controls that must fail.
  """
  requested = [item for item in raw.split(",") if item]
  unknown = [item for item in requested if item not in (*MUTANTS, POSITIVE_ARM)]
  if unknown:
    msg = f"unknown mutant {unknown[0]}"
    raise SystemExit(msg)
  negatives = [item for item in requested if item in MUTANTS]
  return negatives, POSITIVE_ARM in requested


def _replay_paths(work: Path, *, scalar_eval: bool) -> tuple[Path, Path, Path, Path]:
  """Mask npz, mask catalog, draw npz, draw catalog.

  The train-mode names are the ones the negative controls already read.
  """
  if scalar_eval:
    return (
      work / "masks_eval.npz",
      work / "masks_eval.json",
      work / "draws_eval.npz",
      work / "draws_eval.json",
    )
  return (
    work / "masks.npz",
    work / "masks.json",
    work / "draws.npz",
    work / "draws.json",
  )


def _touch_scalar_dropout(model: Any, *, scalar_eval: bool) -> list[Any]:  # noqa: ANN401
  """Switch ``p > 0`` dropout to train, or require that it stay in eval.

  The positive control passes ``scalar_eval=True`` and gets an empty list.
  A module already in train mode is then an instrument error, not a port defect.
  """
  import torch

  if scalar_eval:
    for module in model.modules():
      if isinstance(module, torch.nn.Dropout) and module.p > 0 and module.training:
        msg = "positive control requires nn.Dropout to stay in eval"
        raise RuntimeError(msg)
    return []
  touched: list[Any] = []
  for module in model.modules():
    if isinstance(module, torch.nn.Dropout) and module.p > 0:
      module.train()
      touched.append(module)
  return touched


def _checkpoint(args: argparse.Namespace) -> Path:
  if args.checkpoint is not None:
    return args.checkpoint
  return args.laser_root / "model_weights" / CHECKPOINT_NAME


def _manifest() -> dict[str, Any]:
  path = _repo() / "tests" / "fixtures" / "laser" / "fixtures_manifest.toml"
  return tomllib.loads(path.read_text(encoding="utf-8"))


def _fixture_paths(root: Path) -> list[Path]:
  manifest = _manifest()
  folder = _repo() / "tests" / "fixtures" / "laser"
  paths = [root / "example_pdbs" / EXAMPLE_NAME]
  paths.extend(folder / f"{stem}.pdb" for stem in manifest["ordered_ids"][: N_COMPLEXES - 1])
  return paths


def _check_inputs(root: Path, checkpoint: Path) -> None:
  example = root / "example_pdbs" / EXAMPLE_NAME
  got = _sha256(example)
  if got != EXAMPLE_SHA256:
    msg = f"{example} sha256 {got} != {EXAMPLE_SHA256}"
    raise SystemExit(msg)
  manifest = _manifest()
  folder = _repo() / "tests" / "fixtures" / "laser"
  for stem in manifest["ordered_ids"][: N_COMPLEXES - 1]:
    name = f"{stem}.pdb"
    path = folder / name
    digest = manifest["files"][name]
    got = _sha256(path)
    if got != digest:
      msg = f"{path} sha256 {got} != {digest}"
      raise SystemExit(msg)
  got = _sha256(checkpoint)
  if got != CHECKPOINT_SHA256:
    msg = f"{checkpoint} sha256 {got} != {CHECKPOINT_SHA256}"
    raise SystemExit(msg)


def _results_path() -> Path:
  env = os.environ.get("BTH_RESULTS_PATH")
  if not env:
    msg = "laser_proofread_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  return Path(env)


def _band(max_mean: float, max_std: float) -> str:
  # Pre-registered at spec line 1181. Both mean and std must clear a band.
  # A non-finite std (the reduction-swap of one dropout rep) is a fail.
  if not (max_mean == max_mean and max_std == max_std):
    return "fail"
  worst = max(max_mean, max_std)
  if worst <= PASS_MAX:
    return "pass"
  if worst <= INCONCLUSIVE_MAX:
    return "inconclusive"
  return "fail"


def _work_dir(args: argparse.Namespace) -> Path:
  work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "laser_proofread_parity"
  work.mkdir(parents=True, exist_ok=True)
  return work


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_proofread_parity")
  logger.setLevel(logging.INFO)
  return logger


def _build_job(root: Path) -> dict[str, Any]:
  return {"structures": [{"name": path.stem, "pdb": str(path)} for path in _fixture_paths(root)]}


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import numpy as np

  job = json.loads(args.job.read_text(encoding="utf-8"))
  upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
  # The positive control reads the eval-mode capture. Every other arm reads
  # the train-mode files the negative controls already use.
  masks_path, catalog_path, draws_path, draw_catalog_path = _replay_paths(
    Path(args.work_dir),
    scalar_eval=arm_plan(args.arm).scalar_eval,
  )
  masks = np.load(masks_path)
  catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
  draws = np.load(draws_path)
  draw_catalog = json.loads(draw_catalog_path.read_text(encoding="utf-8"))
  checkpoint = _checkpoint(args)
  mean_gaps: list[float] = []
  std_gaps: list[float] = []
  cells: dict[str, Any] = {}
  for payload in job["structures"]:
    measured = _aminx_one(payload, checkpoint, args.arm, masks, catalog, draws, draw_catalog)
    cells[str(payload["name"])] = measured["cells"]
    ref = upstream[payload["name"]]
    mean_gaps.append(float(np.max(np.abs(np.asarray(measured["mean"]) - np.asarray(ref["mean"])))))
    std_delta = np.abs(np.asarray(measured["std"], dtype=np.float64) - np.asarray(ref["std"]))
    if not np.all(np.isfinite(std_delta)):
      std_gaps.append(float("nan"))
    else:
      std_gaps.append(float(np.max(std_delta)))
  max_mean = float(np.max(mean_gaps)) if mean_gaps else float("inf")
  max_std = float(np.nanmax(np.asarray(std_gaps, dtype=np.float64))) if std_gaps else float("inf")
  if any(gap != gap for gap in std_gaps):
    max_std = float("nan")
  _write_cells(args, cells, upstream)
  return {"band": _band(max_mean, max_std), "max_abs_mean": max_mean, "max_abs_std": max_std}


def _cell_draws(
  name: str,
  n_orders: int,
  draws: Any,  # noqa: ANN401
  draw_catalog: list[dict[str, Any]],
) -> list[Any]:
  """The uniform buffer each order consumed, in copy order.

  The oracle draws ``length * 5 + 4`` once per (complex, order) before that
  cell's dropout masks, and injects that array into ``sample``. Replaying a
  fresh generator here would diverge as soon as dropout consumed the shared
  RNG, so the arm reads the saved buffer instead.
  """
  cell: list[Any] = []
  for copy in range(n_orders):
    matched = [row for row in draw_catalog if row["name"] == name and int(row["copy"]) == copy]
    if len(matched) != 1:
      msg = f"{name} order {copy} has {len(matched)} uniform buffers"
      raise RuntimeError(msg)
    cell.append(draws[str(matched[0]["key"])])
  return cell


def _write_cells(
  args: argparse.Namespace,
  aminx_cells: dict[str, Any],
  upstream: dict[str, Any],
) -> None:
  """Per-complex, per-dropout, per-order probabilities from both arms.

  The graded payload keeps only the reduced mean and std. This file is what
  makes a later miss localizable to one decoding order without another run.
  """
  import numpy as np

  per_order: list[dict[str, Any]] = []
  upstream_cells: dict[str, Any] = {}
  for name, aminx in aminx_cells.items():
    ref = upstream.get(name, {})
    got = ref.get("cells") if isinstance(ref, dict) else None
    upstream_cells[name] = got
    if got is None:
      continue
    left = np.asarray(aminx, dtype=np.float64)
    right = np.asarray(got, dtype=np.float64)
    if left.shape != right.shape:
      per_order.append({"name": name, "agree": False, "detail": f"{left.shape} != {right.shape}"})
      continue
    # axes: dropout, order, class
    for dropout in range(left.shape[0]):
      for order in range(left.shape[1]):
        delta = float(np.max(np.abs(left[dropout, order] - right[dropout, order])))
        per_order.append(
          {
            "name": name,
            "dropout": dropout,
            "order": order,
            "max_abs": delta,
          },
        )
  payload = {
    "aminx": aminx_cells,
    "upstream": upstream_cells,
    "per_order_max_abs": per_order,
  }
  path = Path(args.work_dir) / f"cells_{args.arm}.json"
  path.write_text(json.dumps(payload), encoding="utf-8")


def _aminx_one(
  payload: dict[str, Any],
  checkpoint: Path,
  arm: str | None,
  masks: Any,  # noqa: ANN401
  catalog: list[dict[str, Any]],
  draws: Any,  # noqa: ANN401
  draw_catalog: list[dict[str, Any]],
) -> dict[str, list[float]]:
  from types import SimpleNamespace

  import jax.numpy as jnp
  import numpy as np

  from aminx.families.laser_mpnn.driver import LaserDriver, _working_dtype
  from aminx.families.laser_mpnn.featurize import featurize
  from aminx.model.laser.graphs import GraphStructure
  from aminx.model.laser.joint_decode import LaserJointDecode
  from aminx.model.laser.proofread import conditional_focus_probs, reduce_proofread

  plan = arm_plan(arm)
  model = LaserDriver().load(SimpleNamespace(model_local_path=str(checkpoint)))
  import jax

  from aminx.families.laser_mpnn.driver import _assign_state, _torch_blob

  state = _torch_blob(checkpoint)["model_state_dict"]
  joint = _assign_state(
    LaserJointDecode(key=jax.random.PRNGKey(0)),
    state,
    ("chi_offset_prediction_layers.",),
    _working_dtype(),
  )
  features = featurize(
    payload["pdb"],
    dtype=_working_dtype(),
    lig_pr_knn_k=int(model.lig_pr_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  length = int(features.sequence_indices.shape[0])
  orders = [np.arange(length, dtype=np.int32), np.arange(length, dtype=np.int32)[::-1]]
  order_masks: list[dict[str, list[np.ndarray]]] = []
  order_edges: list[dict[str, list[np.ndarray | None]]] = []
  for copy in range(2):
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    edge_grouped: dict[str, list[tuple[int, np.ndarray | None]]] = {}
    for row in catalog:
      if row["name"] != payload["name"] or int(row["copy"]) != copy:
        continue
      grouped.setdefault(str(row["path"]), []).append((int(row["call"]), masks[str(row["key"])]))
      edge_key = row.get("edge_key")
      edge_array = None if edge_key is None else np.asarray(masks[str(edge_key)])
      edge_grouped.setdefault(str(row["path"]), []).append((int(row["call"]), edge_array))
    order_masks.append(
      {path: [array for _call, array in sorted(rows)] for path, rows in grouped.items()},
    )
    order_edges.append(
      {path: [array for _call, array in sorted(rows)] for path, rows in edge_grouped.items()},
    )
  structure = GraphStructure(
    pr_pr_knn_graph_k=int(model.pr_pr_knn_graph_k),
    lig_pr_knn_graph_k=int(model.lig_pr_knn_graph_k),
    lig_lig_knn_graph_k=int(model.lig_lig_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  probs = conditional_focus_probs(
    model.encoder,
    model.decoder,
    joint,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    structure,
    np.asarray(features.sequence_indices),
    np.asarray(features.chi_angles),
    int(payload["focus"]),
    orders,
    [order_masks],
    [_cell_draws(str(payload["name"]), len(orders), draws, draw_catalog)],
    [order_edges],
    scalar=plan.scalar,
    vector=plan.vector,
    repack_all=True,
  )
  ddof = plan.ddof
  std_over = plan.std_over
  mean, std, _both = reduce_proofread(jnp.asarray(probs), ddof=ddof, std_over=std_over)
  return {
    "mean": np.asarray(mean).reshape(-1).tolist(),
    "std": np.asarray(std).reshape(-1).tolist(),
    # (n_dropouts, n_orders, 21), before the reduction the band is computed on.
    "cells": np.asarray(probs).tolist(),
  }


def _oracle_python(args: argparse.Namespace) -> str:
  if args.oracle_python:
    return str(args.oracle_python)
  try:
    import torch  # noqa: F401
  except ImportError:
    msg = "set LASER_ORACLE_PYTHON to the torch oracle interpreter"
    raise SystemExit(msg) from None
  return sys.executable


def _oracle_command(
  args: argparse.Namespace,
  *,
  script: str,
  checkpoint: Path,
  job_path: Path,
  upstream_path: Path,
  work: Path,
  scalar_eval: bool,
) -> list[str]:
  """Argv for one oracle worker. ``scalar_eval`` is the positive-control reference."""
  command = [
    _oracle_python(args),
    script,
    "--oracle-worker",
    "--laser-root",
    str(args.laser_root),
    "--checkpoint",
    str(checkpoint),
    "--job",
    str(job_path),
    "--upstream",
    str(upstream_path),
    "--work-dir",
    str(work),
  ]
  if scalar_eval:
    command.append("--scalar-eval")
  return command


def _arm_command(
  *,
  script: str,
  arm: str,
  checkpoint: Path,
  job_path: Path,
  upstream_path: Path,
  work: Path,
  laser_root: Path,
  payload_out: Path,
) -> list[str]:
  return [
    sys.executable,
    script,
    "--arm",
    arm,
    "--job",
    str(job_path),
    "--upstream",
    str(upstream_path),
    "--checkpoint",
    str(checkpoint),
    "--laser-root",
    str(laser_root),
    "--work-dir",
    str(work),
    "--payload-out",
    str(payload_out),
  ]


def _consume_arm(
  logger: logging.Logger,
  *,
  arm: str,
  command: list[str],
  payload_out: Path,
  timeout: float,
) -> dict[str, Any]:
  payload_out.unlink(missing_ok=True)
  try:
    launched = _launch(command, timeout)
  except subprocess.TimeoutExpired as exc:
    logger.error("arm %s timed out", arm)
    return {"band": "error", "detail": str(exc)}
  # An arm that errors must say why in the run log. Until 261004 the detail was
  # kept only in the returned dict, so graded run f7887e2f (its clean arm
  # OOM-killed at 20:19 on 261003) recorded clean = "error" with no cause
  # anywhere. Same fix the unconditional vehicle got in 37df972b.
  if launched.returncode != 0:
    logger.error("arm %s exited %s:\n%s", arm, launched.returncode, launched.stderr[-4000:])
    return {"band": "error", "detail": launched.stderr[-500:]}
  try:
    loaded: dict[str, Any] = json.loads(payload_out.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    logger.error("arm %s wrote no readable payload:\n%s", arm, launched.stderr[-4000:])
    return {"band": "error", "detail": launched.stderr[-500:] or "empty payload"}
  return loaded


def _positive_payload(row: dict[str, Any]) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "arm": POSITIVE_ARM,
    "band": row.get("band"),
    "max_abs_mean": row.get("max_abs_mean"),
    "max_abs_std": row.get("max_abs_std"),
  }
  if "detail" in row:
    payload["detail"] = row["detail"]
  return payload


def _launch(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
  # A fresh interpreter is load-bearing. A jit cache warmed by an earlier arm
  # would replay the unmutated trace, and the harness would report a kill it
  # never demonstrated. Dropout masks are trace-time inputs.
  return subprocess.run(
    command,
    check=False,
    capture_output=True,
    text=True,
    timeout=timeout,
  )


def _parent(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
  checkpoint = _checkpoint(args)
  _check_inputs(args.laser_root, checkpoint)
  if float(args.arm_timeout) <= 0:
    msg = "--arm-timeout must be positive"
    raise SystemExit(msg)
  negatives, run_positive = split_mutants(args.mutants)
  work = _work_dir(args)
  job = _build_job(args.laser_root)
  job_path = work / "job.json"
  upstream_path = work / "upstream.json"
  job_path.write_text(json.dumps(job), encoding="utf-8")
  script = str(Path(__file__).resolve())
  oracle = _launch(
    _oracle_command(
      args,
      script=script,
      checkpoint=checkpoint,
      job_path=job_path,
      upstream_path=upstream_path,
      work=work,
      scalar_eval=False,
    ),
    float(args.arm_timeout),
  )
  if oracle.returncode != 0:
    logger.error("oracle worker failed: %s", oracle.stderr[-500:])
    raise SystemExit(oracle.stderr[-500:])
  # The oracle writes the focus into the job so both sides share it.
  job_path.write_text(
    upstream_path.with_name("job.json").read_text(encoding="utf-8"), encoding="utf-8"
  )
  measured: dict[str, dict[str, Any]] = {}
  for arm in ("clean", *negatives):
    payload_out = work / f"payload_{arm}.json"
    measured[arm] = _consume_arm(
      logger,
      arm=arm,
      command=_arm_command(
        script=script,
        arm=arm,
        checkpoint=checkpoint,
        job_path=job_path,
        upstream_path=upstream_path,
        work=work,
        laser_root=args.laser_root,
        payload_out=payload_out,
      ),
      payload_out=payload_out,
      timeout=float(args.arm_timeout),
    )
  positive_row: dict[str, Any] | None = None
  if run_positive:
    # Separate capture. The train-mode upstream.json stays the reference for
    # clean and the four negatives.
    eval_upstream = work / "upstream_eval.json"
    logger.info("positive control oracle: nn.Dropout left in eval")
    eval_oracle = _launch(
      _oracle_command(
        args,
        script=script,
        checkpoint=checkpoint,
        job_path=job_path,
        upstream_path=eval_upstream,
        work=work,
        scalar_eval=True,
      ),
      float(args.arm_timeout),
    )
    if eval_oracle.returncode != 0:
      logger.error("positive-control oracle failed: %s", eval_oracle.stderr[-500:])
      positive_row = {"band": "error", "detail": eval_oracle.stderr[-500:]}
    else:
      payload_out = work / f"payload_{POSITIVE_ARM}.json"
      positive_row = _consume_arm(
        logger,
        arm=POSITIVE_ARM,
        command=_arm_command(
          script=script,
          arm=POSITIVE_ARM,
          checkpoint=checkpoint,
          job_path=job_path,
          upstream_path=eval_upstream,
          work=work,
          laser_root=args.laser_root,
          payload_out=payload_out,
        ),
        payload_out=payload_out,
        timeout=float(args.arm_timeout),
      )
    report = _positive_payload(positive_row)
    (work / "positive_control.json").write_text(
      json.dumps(report, indent=2) + "\n",
      encoding="utf-8",
    )
    logger.info(
      "positive control %s max_abs_mean=%s max_abs_std=%s band=%s",
      POSITIVE_ARM,
      report.get("max_abs_mean"),
      report.get("max_abs_std"),
      report.get("band"),
    )
  clean = str(measured["clean"]["band"])
  statuses = {
    mutant: "failed"
    if measured[mutant]["band"] == "fail"
    else "passed"
    if measured[mutant]["band"] in {"pass", "inconclusive"}
    else "error"
    for mutant in negatives
  }
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    controls: dict[str, Any] = {
      "clean": clean,
      "mutants": statuses,
      "weights": {stable_artifact_key(checkpoint): _sha256(checkpoint)},
    }
    if positive_row is not None:
      controls["positive_control"] = _positive_payload(positive_row)
    args.controls_out.write_text(json.dumps(controls, indent=2) + "\n", encoding="utf-8")
  n_failed = sum(status == "failed" for status in statuses.values())
  clean_row = measured["clean"]
  return {
    "clean": clean,
    "n_listed": len(negatives),
    "n_failed": n_failed,
    "max_abs_mean": float(clean_row.get("max_abs_mean", float("nan"))),
    "max_abs_std": float(clean_row.get("max_abs_std", float("nan"))),
  }


def _package_parent(root: Path) -> Path:
  if (root / "utils" / "model.py").is_file():
    return root.parent
  nested = root / "LASErMPNN"
  if (nested / "utils" / "model.py").is_file():
    return root
  msg = f"{root} does not contain LASErMPNN utils/model.py"
  raise SystemExit(msg)


def _cast_floats(value: object, dtype: Any) -> None:  # noqa: ANN401
  import torch

  if isinstance(value, torch.Tensor):
    return
  if isinstance(value, dict):
    for item in value.values():
      _cast_floats(item, dtype)
    return
  named = getattr(value, "__dict__", None)
  if not isinstance(named, dict):
    return
  for key, item in named.items():
    if isinstance(item, torch.Tensor) and item.is_floating_point():
      named[key] = item.to(dtype=dtype)
    elif item is not None and not isinstance(item, torch.Tensor):
      _cast_floats(item, dtype)


def _oracle_worker(args: argparse.Namespace) -> None:
  import numpy as np
  import torch

  sys.path.insert(0, str(_package_parent(args.laser_root)))
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  from LASErMPNN.run_inference import (  # noqa: PLC0415
    ProteinComplexData,
    get_protein_hierview,
    load_model_from_parameter_dict,
  )
  from LASErMPNN.utils.model import RBF_Encoding  # noqa: PLC0415
  from oracle_shims.laser import (  # noqa: PLC0415
    assert_vector_dropout_eval,
    f64_runtime,
    injected_scalar_dropout,
    injected_uniform_draws,
  )

  checkpoint = _checkpoint(args)
  torch.manual_seed(0)
  model, params = load_model_from_parameter_dict(str(checkpoint), "cpu", strict=False)
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
  model.load_state_dict(state, strict=False)
  model.double()
  model.eval()
  job = json.loads(args.job.read_text(encoding="utf-8"))
  stats: dict[str, dict[str, list[float]]] = {}
  blobs: dict[str, np.ndarray] = {}
  catalog: list[dict[str, Any]] = []
  draw_blobs: dict[str, np.ndarray] = {}
  draw_catalog: list[dict[str, Any]] = []
  model_params = params["model_params"]
  rng = np.random.default_rng(0)
  with f64_runtime(RBF_Encoding), torch.no_grad():
    for payload in job["structures"]:
      view = get_protein_hierview(payload["pdb"])
      data = ProteinComplexData(view, payload["pdb"], verbose=False)
      batch = data.output_batch_data(fix_beta=False)
      _cast_floats(batch, torch.float64)
      batch.construct_graphs(  # type: ignore[attr-defined]
        model.rotamer_builder,
        model.ligand_featurizer,
        **model_params["graph_structure"],
        protein_training_noise=0.0,
        ligand_training_noise=0.0,
        subgraph_only_dropout_rate=0.0,
        num_adjacent_residues_to_drop=0,
        build_hydrogens=bool(model_params["build_hydrogens"]),
      )
      _cast_floats(batch, torch.float64)
      shell = batch.first_shell_ligand_contact_mask  # type: ignore[attr-defined]
      focus = int(shell.nonzero()[0].item()) if bool(shell.any()) else 0
      payload["focus"] = focus
      length = int(batch.sequence_indices.shape[0])  # type: ignore[attr-defined]
      probs: list[torch.Tensor] = []
      for copy in range(2):
        order = torch.arange(length, dtype=torch.float64)
        if copy == 1:
          order = torch.flip(order, dims=(0,))
        batch.decoding_order = order.view(1, -1)  # type: ignore[attr-defined]
        batch.chain_mask = torch.ones_like(batch.chain_mask)  # type: ignore[attr-defined]
        batch.chain_mask[focus] = False  # type: ignore[attr-defined]
        draws = rng.random((1, length * 5 + 4))
        # Saved before dropout consumes the same generator, so the aminx arm
        # can inject this buffer rather than a later draw from the stream.
        draw_key = f"u{len(draw_blobs)}"
        draw_blobs[draw_key] = np.asarray(draws)
        draw_catalog.append({"key": draw_key, "name": payload["name"], "copy": copy})
        touched = _touch_scalar_dropout(model, scalar_eval=bool(args.scalar_eval))
        assert_vector_dropout_eval(model)
        try:
          with injected_scalar_dropout(model, rng=rng) as dropout, injected_uniform_draws(draws):
            sampled = model.sample(  # type: ignore[attr-defined]
              batch,
              sequence_sample_temperature=1.0,
              chi_angle_sample_temperature=1.0,
              disabled_residues=["X"],
              disable_pbar=True,
              repack_all=True,
            )
            # Eval mode records nothing: dropout never fires, and replaying an
            # empty catalog with scalar=False is the plain conditional decode.
            recorded = () if args.scalar_eval else tuple(dropout.recorded.items())
            for path, recorded_rows in recorded:
              identities = dropout.edges.get(path, [])
              for index, row in enumerate(recorded_rows):
                key = f"m{len(blobs)}"
                blobs[key] = np.asarray(row)
                identity = identities[index] if index < len(identities) else None
                edge_key = None
                if identity is not None:
                  edge_key = f"e{key[1:]}"
                  blobs[edge_key] = np.asarray(identity, dtype=np.int64)
                catalog.append(
                  {
                    "key": key,
                    "name": payload["name"],
                    "copy": copy,
                    "path": path,
                    "call": index,
                    "edge_key": edge_key,
                  },
                )
        finally:
          for module in touched:
            module.eval()
        if args.scalar_eval:
          _touch_scalar_dropout(model, scalar_eval=True)
        probs.append(torch.softmax(sampled.sequence_logits[focus], dim=-1))
      stacked = torch.stack(probs, dim=0).double()
      mean = stacked.mean(dim=0)
      std = stacked.std(dim=0, unbiased=True)
      # (n_dropouts, n_orders, 21). This vehicle uses one dropout.
      cells = stacked.detach().cpu().unsqueeze(0)
      stats[payload["name"]] = {
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "cells": cells.tolist(),
      }
  masks_path, catalog_path, draws_path, draw_catalog_path = _replay_paths(
    Path(args.work_dir),
    scalar_eval=bool(args.scalar_eval),
  )
  if blobs:
    np.savez_compressed(masks_path, **blobs)  # ty: ignore[invalid-argument-type]
  else:
    # Eval mode records no keeps. savez rejects an empty archive.
    np.savez_compressed(masks_path, _empty=np.zeros((0,), dtype=np.float64))
  catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
  np.savez_compressed(draws_path, **draw_blobs)  # ty: ignore[invalid-argument-type]
  draw_catalog_path.write_text(json.dumps(draw_catalog), encoding="utf-8")
  args.job.write_text(json.dumps(job), encoding="utf-8")
  args.upstream.write_text(json.dumps(stats), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  if args.oracle_worker:
    _oracle_worker(args)
    return
  if args.arm:
    payload = _run_arm(args)
    text = json.dumps(payload)
    if args.payload_out is not None:
      args.payload_out.write_text(text, encoding="utf-8")
    else:
      sys.stdout.write(text)
    return
  path = _results_path()
  logger = _logger()
  results = _parent(args, logger)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
