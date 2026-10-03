"""Graded LASEr unconditional proofread parity (do not run from the fixer).

Five complexes, injected scalar-dropout masks, one unconditional forward.
Pass is max |delta proofread_mean| <= 1e-4. Inconclusive is (1e-4, 1e-3].
Fail is > 1e-3. The band is the spec §7.3 line 1181 pre-registration, the
same one ``laser_proofread_parity`` uses for the conditional arm. There is
no standard deviation: upstream makes one forward pass.

Negative controls, each of which must land in fail:
  unconditional path with no dropout context (the #2424 pre-fix behaviour)
  scalar dropout off
  vector dropout on

NEVER append ``unc_both_off`` to a graded run's ``--mutants``. Redsox step
1c compares the RAW argv string against this slug's manifest rows by exact set
equality (``tests/knob_gate/_coverage.py:309``), and the positive control has
no manifest row -- every manifest row must report ``failed``, which a control
that must pass never will. This script drops the positive arm from
``n_listed``, ``n_failed`` and the controls ``mutants`` dict, but it cannot
rewrite its own argv, so appending it silently costs the run its ledger
eligibility. Run it as its own separate, ungraded invocation.

Positive control ``unc_both_off`` (optional, not a negative, not part of
the pre-registered outcomes): upstream ``nn.Dropout`` stays in eval and the
aminx plan runs without dropout. It must report near-zero. A non-zero result
means instrument floor rather than a port defect. Run it on its own with
``--mutants unc_both_off`` and no ``--controls-out``; it is excluded from
``n_listed`` and ``n_failed``, and must not ride along on a graded run.

The clean arm and each mutant arm are separate subprocesses. A missing
``$BTH_RESULTS_PATH`` is fatal.
"""

from __future__ import annotations

import os
import pathlib

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

MUTANTS = ("unc_no_dropout", "unc_scalar_off", "unc_vector_on")
# Optional arm. Not a member of MUTANTS: those three must fail, and this one
# must agree. Counting it in n_listed would change the pre-registered outcomes.
POSITIVE_ARM = "unc_both_off"
PASS_MAX = 1e-4
INCONCLUSIVE_MAX = 1e-3
#: Module-path prefix of upstream layers that run after sequence_logits.
_CHI_OFFSET = "chi_offset_prediction_layers."
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
  """Dropout flags for one unconditional arm, and whether upstream stays off."""

  scalar: bool
  vector: bool
  scalar_eval: bool
  force_no_dropout: bool
  inject_masks: bool


def unc_both_off_plan() -> ArmPlan:
  """Positive control: both sides have scalar dropout genuinely disabled.

  Upstream ``nn.Dropout`` modules are left in eval (not switched to train)
  and aminx runs the unconditional forward with no dropout plan.
  This arm must report near-zero max_abs_mean.
  A non-zero result means instrument floor rather than a port defect.
  """
  return ArmPlan(
    scalar=False,
    vector=False,
    scalar_eval=True,
    force_no_dropout=False,
    inject_masks=False,
  )


def arm_plan(arm: str | None) -> ArmPlan:
  """Dropout flags for ``arm``.

  ``unc_scalar_off`` disables only the aminx plan. Upstream for that arm is
  still the train-mode reference, which is why it is a negative control.
  ``unc_no_dropout`` is the #2424 path: the option stays on and the forward
  never enters the plan.
  """
  if arm == POSITIVE_ARM:
    return unc_both_off_plan()
  if arm == "unc_no_dropout":
    return ArmPlan(
      scalar=True,
      vector=False,
      scalar_eval=False,
      force_no_dropout=True,
      inject_masks=False,
    )
  return ArmPlan(
    scalar=arm != "unc_scalar_off",
    vector=arm == "unc_vector_on",
    scalar_eval=False,
    force_no_dropout=False,
    inject_masks=True,
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


def _replay_paths(work: Path, *, scalar_eval: bool) -> tuple[Path, Path]:
  """Mask npz and mask catalog.

  The train-mode names are the ones the negative controls already read.
  One forward records no uniform draws.
  """
  if scalar_eval:
    return work / "masks_eval.npz", work / "masks_eval.json"
  return work / "masks.npz", work / "masks.json"


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
    msg = "laser_proofread_unconditional_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  return Path(env)


def _band(max_mean: float) -> str:
  # Pre-registered at spec line 1181. Only the mean is compared: one forward
  # has no standard deviation. A non-finite mean is a fail.
  if not (max_mean == max_mean):
    return "fail"
  if max_mean <= PASS_MAX:
    return "pass"
  if max_mean <= INCONCLUSIVE_MAX:
    return "inconclusive"
  return "fail"


def _work_dir(args: argparse.Namespace) -> Path:
  work = args.work_dir or (
    Path(os.environ.get("TMPDIR", "/tmp")) / "laser_proofread_unconditional_parity"
  )
  work.mkdir(parents=True, exist_ok=True)
  return work


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_proofread_unconditional_parity")
  logger.setLevel(logging.INFO)
  return logger


def _build_job(root: Path) -> dict[str, Any]:
  return {"structures": [{"name": path.stem, "pdb": str(path)} for path in _fixture_paths(root)]}


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import jax.numpy as jnp
  import numpy as np

  from aminx.families.laser_mpnn.driver import canonical_logits

  job = json.loads(args.job.read_text(encoding="utf-8"))
  upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
  # The positive control reads the eval-mode capture. Every other arm reads
  # the train-mode files the negative controls already use.
  masks_path, catalog_path = _replay_paths(
    Path(args.work_dir),
    scalar_eval=arm_plan(args.arm).scalar_eval,
  )
  masks = np.load(masks_path)
  catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
  checkpoint = _checkpoint(args)
  mean_gaps: list[float] = []
  for payload in job["structures"]:
    measured = _aminx_one(payload, checkpoint, args.arm, masks, catalog)
    ref = upstream[payload["name"]]
    # proofread_mean is canonical. Upstream softmax is LASEr order; the same
    # map the purpose applies has to land on the reference before the delta.
    got = np.asarray(measured["mean"], dtype=np.float64)
    reference = np.asarray(canonical_logits(jnp.asarray(ref["mean"], dtype=jnp.float64)))
    if got.shape != reference.shape or not np.all(np.isfinite(got)):
      mean_gaps.append(float("nan"))
      continue
    mean_gaps.append(float(np.max(np.abs(got - reference))))
  max_mean = float(np.max(mean_gaps)) if mean_gaps else float("inf")
  if any(gap != gap for gap in mean_gaps):
    max_mean = float("nan")
  return {"band": _band(max_mean), "max_abs_mean": max_mean}


def _group_masks(
  name: str,
  masks: Any,  # noqa: ANN401
  catalog: list[dict[str, Any]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
  """One structure's keep catalog, in call order. One forward, so no copy axis."""
  import numpy as np

  grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
  edge_grouped: dict[str, list[tuple[int, np.ndarray | None]]] = {}
  for row in catalog:
    if row["name"] != name:
      continue
    grouped.setdefault(str(row["path"]), []).append((int(row["call"]), masks[str(row["key"])]))
    edge_key = row.get("edge_key")
    edge_array = None if edge_key is None else np.asarray(masks[str(edge_key)])
    edge_grouped.setdefault(str(row["path"]), []).append((int(row["call"]), edge_array))
  order_masks = {path: [array for _call, array in sorted(rows)] for path, rows in grouped.items()}
  order_edges = {
    path: [array for _call, array in sorted(rows)] for path, rows in edge_grouped.items()
  }
  return order_masks, order_edges


def _aminx_one(
  payload: dict[str, Any],
  checkpoint: Path,
  arm: str | None,
  masks: Any,  # noqa: ANN401
  catalog: list[dict[str, Any]],
) -> dict[str, list[float]]:
  from types import SimpleNamespace

  import jax
  import numpy as np

  from aminx.families.laser_mpnn.driver import (
    LaserDriver,
    _prepare,
    _proofread_batch,
    _working_dtype,
  )
  from aminx.families.laser_mpnn.featurize import featurize
  from aminx.run.options import LaserOptions

  plan = arm_plan(arm)
  model = LaserDriver().load(SimpleNamespace(model_local_path=str(checkpoint)))
  probe = featurize(payload["pdb"], dtype=_working_dtype())
  length = int(probe.sequence_indices.shape[0])
  # The unconditional forward reads the structure sequence, not this candidate.
  # _prepare still requires one string of the right length.
  spec = SimpleNamespace(sequences_to_score=("A" * length,), random_seed=0)
  # The option stays on for unc_no_dropout: that arm is the pre-fix path, which
  # ignores the option. scalar_off turns the plan's scalar flag off instead.
  honour_dropout = plan.scalar or plan.force_no_dropout
  options = LaserOptions(proofread_dropout=honour_dropout)
  prepared = _prepare(Path(payload["pdb"]), spec, options)
  catalog_masks = None
  catalog_edges = None
  if plan.inject_masks:
    catalog_masks, catalog_edges = _group_masks(str(payload["name"]), masks, catalog)
    # Upstream's unconditional model.forward also runs the chi-offset layers, so
    # masks are RECORDED for them, but aminx's unconditional_logits never runs
    # those layers and the strict replay then refuses the unconsumed masks.
    # Dropping them is sound, not convenient: upstream computes sequence_logits
    # at utils/model.py:420, BEFORE the chi loop that calls
    # chi_offset_prediction_layers (:430-436), so they cannot reach the one
    # output this vehicle compares. Every OTHER recorded mask is still demanded.
    catalog_masks = {k: v for k, v in catalog_masks.items() if not k.startswith(_CHI_OFFSET)}
    catalog_edges = {k: v for k, v in catalog_edges.items() if not k.startswith(_CHI_OFFSET)}
  # The host hands fold_in(PRNGKey(seed), 0). The stage folds the item index.
  result = _proofread_batch(
    model,
    (prepared,),
    options,
    purpose="score:proofread_unconditional",
    scalar_dropout=honour_dropout,
    seed=0,
    dropout_key=jax.random.fold_in(jax.random.PRNGKey(0), 0),
    masks=catalog_masks,
    edges=catalog_edges,
    vector=plan.vector,
    force_no_dropout=plan.force_no_dropout,
  )
  mean = np.asarray(result["proofread_mean"][0])
  return {"mean": mean.tolist()}


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
  # An arm that errors must say why in the run log. Before 261003 the detail
  # was kept only in the returned dict and never written anywhere, so a smoke
  # run reported clean = "error" with no cause at all.
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
    # clean and the three negatives.
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
      "positive control %s max_abs_mean=%s band=%s",
      POSITIVE_ARM,
      report.get("max_abs_mean"),
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


def _focus_probs(sequence_logits: Any, shell: Any) -> Any:  # noqa: ANN401
  """Softmax over the first-shell rows. An empty shell reports residue 0.

  Same row rule as ``focus_rows``: a selection is not used on this vehicle,
  and a structure with no ligand contact still has one row to compare.
  """
  import numpy as np
  import torch

  logits = sequence_logits.detach()
  while logits.ndim > 2:
    logits = logits[0]
  probs = torch.softmax(logits, dim=-1).cpu().numpy()
  mask = np.asarray(shell.detach().cpu().numpy(), dtype=bool).reshape(-1)
  rows = np.flatnonzero(mask)
  if rows.size == 0:
    rows = np.asarray([0], dtype=np.int64)
  return probs[rows]


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
      batch.generate_decoding_order(False)  # type: ignore[attr-defined]
      touched = _touch_scalar_dropout(model, scalar_eval=bool(args.scalar_eval))
      assert_vector_dropout_eval(model)
      try:
        with injected_scalar_dropout(model, rng=rng) as dropout:
          sequence_logits, *_rest = model.forward(  # type: ignore[attr-defined]
            batch,
            return_unconditional_probabilities=True,
          )
          # Eval mode records nothing: dropout never fires, and replaying an
          # empty catalog with scalar dropout off is the plain forward.
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
                  "copy": 0,
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
      shell = batch.first_shell_ligand_contact_mask  # type: ignore[attr-defined]
      chosen = _focus_probs(sequence_logits, shell)
      stats[payload["name"]] = {"mean": np.asarray(chosen, dtype=np.float64).tolist()}
  masks_path, catalog_path = _replay_paths(
    Path(args.work_dir),
    scalar_eval=bool(args.scalar_eval),
  )
  if blobs:
    np.savez_compressed(masks_path, **blobs)  # ty: ignore[invalid-argument-type]
  else:
    # Eval mode records no keeps. savez rejects an empty archive.
    np.savez_compressed(masks_path, _empty=np.zeros((0,), dtype=np.float64))
  catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
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
