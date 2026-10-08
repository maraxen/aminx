"""Confirmatory LASEr distributional protocol.

Pilot constants come from bathos run e0cefaa7-e027-4efe-bf63-07d78bcfabff.
``CELLS`` holds the two derived cells. The selector refuses ``min_p0@0.1``
and ``min_p0.05@0.3``; ``LASER_CELLS`` still names all three spec cells.

One resumable unit is one (cell, structure, run). Upstream units (U1 shimmed,
U2 unshimmed) call the pilot sampler. Aminx units (A at T, CTRL_m at m·T) call
``aminx.host.runner.sample`` with ``model_family='lasermpnn'``, which is
``LaserDriver`` purpose ``sample``, in float32 with ``jax.random`` keys derived
from the unit seed. Sequence temperature and ``LaserOptions.chi_temp`` both
scale by m, matching the pilot ``_sample_once`` pair. There is no whole-run
timeout; each aminx unit has its own timeout.

``sample_dist_stats`` grades sequence total variation. It has no χ1
confirmatory grader, so this driver builds that grade from
``tv_chi1_per_position`` / ``mean_tv_chi1``, a row bootstrap that shares the U1
resample, and ``confirmatory_verdict`` at ``delta_chi``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

import graded_resume
import laser_sample_dist_pilot as pilot
import sample_dist_stats as stats

PILOT_RUN_ID: str = "e0cefaa7-e027-4efe-bf63-07d78bcfabff"
CELLS: dict[str, dict[str, float | int]] = {
  "min_p0@0.3": {
    "delta": 0.01,
    "delta_chi": 0.05,
    "m": 1.25,
    "h_hat": 0.001560551948051945,
    "n": 1000,
  },
  "min_p0@1.0": {
    "delta": 0.01,
    "delta_chi": 0.05,
    "m": 1.1,
    "h_hat": 0.0017875000000000252,
    "n": 1000,
  },
}
LASER_CELLS = ("min_p0@0.3", "min_p0@1.0", "min_p0.05@0.3")
N_BOOT = 2000
# 4jnj-1_prot.pdb lives in the vendored LASEr repo. Entries 3-6 of
# fixtures_manifest.toml ordered_ids are the laser_score_parity fixtures.
STRUCTURES = {
  "example_pdbs/4jnj-1_prot.pdb": (
    "0a933729c4915cfee14ff34408a57e1d65fd98845df189ba79ad4f4d8470ae7e"
  ),
  "tests/fixtures/laser/103m_1.pdb": (
    "f8edd971b7754e0c30f7e3ae56fa64ad120338d50708873aa4019f180210201d"
  ),
  "tests/fixtures/laser/104m_1.pdb": (
    "44f0347958f06f13b8bd7f171d94cb54d3665c419539247eb21a127a1ef98ed8"
  ),
  "tests/fixtures/laser/105m_1.pdb": (
    "b5a7cc4df78e616eccd7ab40a9c0c4e834df3226fd006a3effe4a9927a7f68a1"
  ),
  "tests/fixtures/laser/106m_1.pdb": (
    "394e9198b3acf0ddaa8708befcb6c6f6aa5677c92f481460b4b49c00a4aea472"
  ),
}
_CELL_KEYS = ("delta", "delta_chi", "m", "h_hat", "n")
_EXCLUDED_CELLS = ("min_p0@0.1", "min_p0.05@0.3")
_EXCLUSION_REASONS = {
  "min_p0@0.1": (
    "min_p0@0.1 is excluded pending the user's decision doc "
    ".praxia/docs/decisions/261004_sample-dist-pilot-fixtures-and-low-t.md; "
    "this confirmatory run refuses it"
  ),
  "min_p0.05@0.3": (
    "min_p0.05@0.3 is excluded because UPSTREAM LASErMPNN cannot produce it "
    "at all (debt 2476: utils/model.py:897-903 feeds the min_p-masked -inf "
    "logits into chi_offset_prediction_layers, giving a NaN offset that "
    "poisons chi_prev and makes the next chi layer's logits all NaN, so "
    "torch Categorical raises; measured on 101m_1 at chi_min_p=0.05, T=0.3); "
    "this confirmatory run refuses it"
  ),
}
_WORK = "laser_sample_dist_confirm"
_WORK_SMOKE = "laser_sample_dist_confirm_smoke"
_SMOKE_CELL_KEY = "min_p0@1.0"
_SMOKE_N = 50
_SMOKE_BOOT = 50
_SMOKE_STRUCTURE = "103m_1"
_SMOKE_CELL: dict[str, float | int] = {
  "delta": 0.02,
  "delta_chi": 0.05,
  "m": 1.25,
  "h_hat": 0.0,
  "n": _SMOKE_N,
}
_RUNS = ("U1", "U2", "A", "CTRL_m")
_UPSTREAM = ("U1", "U2")
_VERDICT_RANK = {"pass": 0, "inconclusive": 1, "fail": 2}
_TOP_LEVEL = (
  "sidecar_verdict",
  "controls_all_fail",
  "n_controls",
  "cells",
  "cells_selected",
  "structures",
  "pilot_run_id",
  "n_reused",
  "n_computed",
  "script_sha256",
  "smoke",
)


class UnitSpec(TypedDict):
  unit_id: str
  cell: str
  condition: str
  cell_temperature: float
  sample_temperature: float
  min_p: float
  structure: str
  run: str
  m: float | None
  shim: bool
  n: int
  seed: int
  pdb: str
  pdb_sha256: str


def _parse(argv: list[str]) -> Any:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=pilot._default_laser_root())
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--oracle-python", default=os.environ.get("LASER_ORACLE_PYTHON"))
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--max-attempts", type=int, default=3)
  parser.add_argument("--arm-timeout", type=float, default=86400.0)
  parser.add_argument("--oracle-worker", action="store_true")
  parser.add_argument("--aminx-worker", action="store_true")
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument(
    "--cells",
    default=None,
    help=(
      "comma-separated subset of the LASEr cells; "
      "min_p0@0.1 and min_p0.05@0.3 are refused"
    ),
  )
  return parser.parse_args(argv)


def _split_cells(raw: str | None) -> list[str]:
  if raw is None:
    return []
  return [part.strip() for part in raw.split(",") if part.strip()]


def _require_cell_row(cell: str, row: dict[str, float | int]) -> None:
  missing = [key for key in _CELL_KEYS if key not in row]
  if missing:
    msg = f"CELLS[{cell}] is missing {missing}; expected {list(_CELL_KEYS)}"
    raise SystemExit(msg)


def selected_cells(raw: str | None, *, smoke: bool) -> list[str]:
  """Cells this process will run.

  An omitted ``--cells`` selects the derived cells. ``min_p0@0.1``,
  ``min_p0.05@0.3``, and unknown keys are refused.
  """
  if smoke and raw is None:
    chosen = [_SMOKE_CELL_KEY]
  elif raw is None:
    chosen = [cell for cell in LASER_CELLS if cell not in _EXCLUDED_CELLS]
  else:
    chosen = _split_cells(raw)
  if not chosen:
    msg = "--cells selected nothing; allowed: " + ",".join(LASER_CELLS)
    raise SystemExit(msg)
  for cell in chosen:
    if cell in _EXCLUDED_CELLS:
      raise SystemExit(_EXCLUSION_REASONS[cell])
    if cell not in LASER_CELLS:
      msg = f"unknown cell {cell}; allowed: {','.join(LASER_CELLS)}"
      raise SystemExit(msg)
  ordered = [cell for cell in LASER_CELLS if cell in set(chosen)]
  if smoke:
    if ordered != [_SMOKE_CELL_KEY]:
      msg = "smoke runs min_p0@1.0 on 103m_1 only"
      raise SystemExit(msg)
    return ordered
  if not CELLS:
    msg = (
      "refusing confirmatory run: CELLS is empty "
      "until the LASEr pilot constants are committed"
    )
    raise SystemExit(msg)
  missing = [cell for cell in ordered if cell not in CELLS]
  if missing:
    msg = (
      f"CELLS is missing {missing}; confirmatory constants are not committed "
      f"(PILOT_RUN_ID={PILOT_RUN_ID})"
    )
    raise SystemExit(msg)
  for cell in ordered:
    _require_cell_row(cell, CELLS[cell])
  return ordered


def _cell_temperature(cell: str) -> float:
  return float(cell.split("@", 1)[1])


def _cell_condition(cell: str) -> str:
  return cell.split("@", 1)[0]


def _min_p(cell: str) -> float:
  condition = _cell_condition(cell)
  if condition == "min_p0":
    return 0.0
  if condition == "min_p0.05":
    return 0.05
  msg = f"no min_p for {cell}"
  raise SystemExit(msg)


def _table(cell: str, *, smoke: bool) -> dict[str, float | int]:
  if smoke:
    return _SMOKE_CELL
  row = CELLS.get(cell)
  if row is None:
    msg = f"{cell} is not in CELLS"
    raise SystemExit(msg)
  _require_cell_row(cell, row)
  return row


def _unit_id(cell: str, structure: str, run: str, n: int) -> str:
  condition, temperature = cell.split("@", 1)
  return f"{condition}|{float(temperature):.1f}|{structure}|{run}|{n}"


def _assign_seeds(labels: list[str], used: set[int]) -> list[int]:
  """Pilot hash of ``confirm|label``, skipping the pilot seed of ``label``."""
  for label in labels:
    used.add(pilot._hash_seed(label))
  return [
    pilot._claim(pilot._hash_seed(f"confirm|{label}"), used) for label in labels
  ]


def unit_specs(
  structures: dict[str, dict[str, str]],
  cells: list[str],
  n: int,
  used: set[int],
  *,
  smoke: bool = False,
) -> list[UnitSpec]:
  """One spec per (cell, structure, run). Seeds are claimed into ``used``."""
  drafts: list[UnitSpec] = []
  labels: list[str] = []
  for cell in cells:
    condition = _cell_condition(cell)
    temperature = _cell_temperature(cell)
    factor = float(_table(cell, smoke=smoke)["m"])
    minimum = _min_p(cell)
    for structure, payload in structures.items():
      for run in _RUNS:
        unit_id = _unit_id(cell, structure, run, n)
        labels.append(unit_id)
        drafts.append(
          {
            "unit_id": unit_id,
            "cell": cell,
            "condition": condition,
            "cell_temperature": temperature,
            "sample_temperature": temperature * factor if run == "CTRL_m" else temperature,
            "min_p": minimum,
            "structure": structure,
            "run": run,
            "m": factor if run == "CTRL_m" else None,
            "shim": run == "U1",
            "n": n,
            "seed": 0,
            "pdb": payload["pdb"],
            "pdb_sha256": payload["sha256"],
          },
        )
  seeds = _assign_seeds(labels, used)
  for draft, seed in zip(drafts, seeds, strict=True):
    draft["seed"] = seed
  return drafts


def bootstrap_seeds(cells: list[str], used: set[int]) -> dict[str, int]:
  labels = [f"bootstrap:{cell}" for cell in cells]
  values = _assign_seeds(labels, used)
  return dict(zip(cells, values, strict=True))


def _require_confirm_entries() -> None:
  """Fail closed if ordered_ids[2:6] no longer names the pinned score-parity files."""
  manifest = pilot._manifest()
  expected = [Path(name).stem for name in STRUCTURES if name.startswith("tests/")]
  ordered = [str(item) for item in manifest["ordered_ids"]]
  if ordered[2:6] != expected:
    msg = (
      "laser_score_parity entries 3-6 moved: "
      f"ordered_ids[2:6]={ordered[2:6]} pin={expected}"
    )
    raise SystemExit(msg)
  files = manifest["files"]
  if not isinstance(files, dict):
    msg = "fixtures_manifest.toml [files] is not a table"
    raise SystemExit(msg)
  for relative, digest in STRUCTURES.items():
    if not relative.startswith("tests/"):
      continue
    name = Path(relative).name
    recorded = files.get(name)
    if recorded != digest:
      msg = f"manifest digest for {name} is {recorded}, driver pin is {digest}"
      raise SystemExit(msg)


def pinned_structures(
  laser_root: Path,
  *,
  smoke: bool,
) -> dict[str, dict[str, str]]:
  """Confirmatory PDBs checked against the pins before use."""
  _require_confirm_entries()
  repo = pilot._repo()
  pinned: dict[str, dict[str, str]] = {}
  for relative, digest in STRUCTURES.items():
    stem = Path(relative).stem
    if smoke and stem != _SMOKE_STRUCTURE:
      continue
    if relative.startswith("tests/"):
      path = repo / relative
    else:
      path = laser_root / relative
    if not path.is_file():
      msg = f"confirmatory structure {relative} is missing at {path}"
      raise SystemExit(msg)
    got = pilot._sha256(path)
    if got != digest:
      msg = f"{path} sha256 {got} != {digest}"
      raise SystemExit(msg)
    pinned[stem] = {"pdb": str(path), "relative": relative, "sha256": digest}
  return pinned


def _graded_units(
  specs: list[UnitSpec],
  checkpoint: Path,
  script: Path,
) -> list[graded_resume.Unit]:
  script_sha = pilot._sha256(script)
  digest = pilot._sha256(checkpoint)
  units: list[graded_resume.Unit] = []
  for spec in specs:
    arm = "upstream" if spec["run"] in _UPSTREAM else "aminx"
    units.append(
      graded_resume.Unit(
        unit_id=spec["unit_id"],
        arm_id=arm,
        checkpoint_sha256=digest,
        input_sha256=pilot._input_sha(spec["pdb_sha256"], spec["min_p"]),
        script_sha256=script_sha,
      ),
    )
  return units


def _pilot_spec(spec: UnitSpec) -> pilot._UnitSpec:
  return {
    "unit_id": spec["unit_id"],
    "condition": spec["condition"],
    "cell_temperature": spec["cell_temperature"],
    "sample_temperature": spec["sample_temperature"],
    "min_p": spec["min_p"],
    "structure": spec["structure"],
    "run": spec["run"],
    "m": spec["m"],
    "shim": spec["shim"],
    "n": spec["n"],
    "seed": spec["seed"],
    "pdb": spec["pdb"],
    "pdb_sha256": spec["pdb_sha256"],
  }


def _upstream_worker(args: Any) -> None:
  if args.job is None:
    msg = "upstream worker requires --job"
    raise SystemExit(msg)
  root = Path(args.laser_root)
  sys.path.insert(0, str(pilot._package_parent(root)))
  checkpoint = pilot._checkpoint(args)
  pilot._require_checkpoint(checkpoint)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[UnitSpec] = list(job["specs"])
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  work = pilot._work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  model: Any = None
  batches: dict[str, Any] = {}
  natives: dict[str, dict[str, Any]] = {}
  if remaining > 0:
    model, params = pilot._load_model(checkpoint)
    model_params = params["model_params"]
    for spec in specs:
      structure = spec["structure"]
      if structure in batches:
        continue
      batch = pilot._featurize(model, model_params, spec["pdb"])
      batches[structure] = batch
      natives[structure] = {
        "sequence": batch.sequence_indices.detach().clone(),
        "chi": batch.chi_angles.detach().clone(),
        "chain_mask": batch.chain_mask.detach().clone(),
      }

  def compute(unit: graded_resume.Unit) -> dict[str, Any]:
    if model is None:
      msg = "oracle model was not loaded"
      raise RuntimeError(msg)
    spec = by_id[unit.unit_id]
    return pilot._sample_unit(
      model,
      batches[spec["structure"]],
      natives[spec["structure"]],
      _pilot_spec(spec),
    )

  graded_resume.run_units(units, compute, cache, resume=bool(args.resume))


def _prng_seed(seed: int) -> int:
  """uint32 seed for ``jax.random.PRNGKey``, derived from the unit seed."""
  digest = hashlib.sha256(f"confirm-prng|{seed}".encode()).digest()
  return int.from_bytes(digest[:4], "little")


def _assert_float32(result: dict[str, Any]) -> None:
  import numpy as np

  for payload in result["structures"].values():
    arrays = payload["arrays"]
    for name, value in arrays.items():
      arr = np.asarray(value)
      if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.dtype(np.float32):
        msg = f"aminx array {name} is {arr.dtype}; this run requires float32"
        raise SystemExit(msg)


def _map_tokens(tokens: Any) -> list[list[int]]:
  """Map production canonical indices onto the pilot LASEr alphabet."""
  import numpy as np

  from aminx.families.laser_mpnn.driver import LASER_OF_CANONICAL
  from aminx.families.laser_mpnn.featurize import LASER_ALPHABET

  array = np.asarray(tokens, dtype=np.int32)
  if array.ndim != 2:
    msg = f"aminx sequences have ndim {array.ndim}, expected 2"
    raise SystemExit(msg)
  table = np.asarray(LASER_OF_CANONICAL, dtype=np.int32)
  mapped = table[array]
  alphabet = len(LASER_ALPHABET)
  if mapped.size and (int(mapped.min()) < 0 or int(mapped.max()) >= alphabet):
    msg = "aminx token outside the pilot alphabet"
    raise SystemExit(msg)
  return [[int(token) for token in row] for row in mapped.tolist()]


def _chi1_degrees(chi_deg: Any, chi_mask: Any) -> list[list[float | None]]:
  """χ1 where ``chi_mask[..., 0]`` is set; otherwise the pilot's None."""
  import numpy as np

  degrees = np.asarray(chi_deg)
  mask = np.asarray(chi_mask, dtype=np.bool_)
  if degrees.ndim != 3 or degrees.shape[-1] != 4:
    msg = f"chi_deg shape {degrees.shape}, expected (N, L, 4)"
    raise SystemExit(msg)
  if mask.shape != degrees.shape:
    msg = f"chi_mask shape {mask.shape} != chi_deg shape {degrees.shape}"
    raise SystemExit(msg)
  encoded: list[list[float | None]] = []
  for sample, sample_mask in zip(degrees, mask, strict=True):
    row: list[float | None] = []
    for angle, keep in zip(sample, sample_mask, strict=True):
      row.append(float(angle[0]) if bool(keep[0]) else None)
    encoded.append(row)
  return encoded


def _design_mask_from_pdb(pdb: Path) -> list[bool]:
  """Same designed-position predicate as the pilot, on production features."""
  import numpy as np

  from aminx.families.laser_mpnn.featurize import featurize

  features = featurize(pdb, dtype=np.float32)
  return pilot._design_mask(features.chain_mask)


def _laser_options(spec: UnitSpec) -> Any:
  from aminx.run.options import LaserOptions

  # Pilot ``_sample_once`` disables X only and sets both min_p knobs. χ temperature
  # tracks the scaled sample temperature, including CTRL_m at m·T.
  return LaserOptions(
    chi_temp=float(spec["sample_temperature"]),
    seq_min_p=float(spec["min_p"]),
    chi_min_p=float(spec["min_p"]),
    disabled_residues=("X",),
  )


def _sample_aminx(spec: UnitSpec, checkpoint: Path) -> dict[str, Any]:
  """Production sample path. Randomness comes from the unit seed only."""
  import jax

  from aminx.host.runner import sample as runner_sample
  from aminx.run.specs import SamplingSpecification

  jax.config.update("jax_enable_x64", False)
  options = _laser_options(spec)
  sampling = SamplingSpecification(
    inputs=spec["pdb"],
    model_family="lasermpnn",
    model_local_path=checkpoint,
    num_samples=spec["n"],
    samples_chunk_size=8,
    return_logits=False,
    random_seed=_prng_seed(spec["seed"]),
    temperature=spec["sample_temperature"],
    backbone_noise=0.0,
    laser=options,
  )
  result = runner_sample(sampling)
  _assert_float32(result)
  if "0" not in result["structures"]:
    msg = "production sample did not return structure 0"
    raise SystemExit(msg)
  arrays = result["structures"]["0"]["arrays"]
  for key in ("sequence", "chi_deg", "chi_mask"):
    if key not in arrays:
      msg = f"production sample did not emit {key}"
      raise SystemExit(msg)
  sequences = _map_tokens(arrays["sequence"])
  if len(sequences) != spec["n"]:
    msg = f"aminx drew {len(sequences)} sequences, expected {spec['n']}"
    raise SystemExit(msg)
  degrees = _chi1_degrees(arrays["chi_deg"], arrays["chi_mask"])
  mask = _design_mask_from_pdb(Path(spec["pdb"]))
  if len(mask) != len(sequences[0]) or len(degrees[0]) != len(sequences[0]):
    msg = "aminx design mask length does not match the sampled sequences"
    raise SystemExit(msg)
  return {
    "sequences": sequences,
    "chi1_degrees": degrees,
    "mask": mask,
    "seed": spec["seed"],
    "sample_temperature": spec["sample_temperature"],
    "min_p": spec["min_p"],
    "shim": False,
    "run": spec["run"],
    "m": spec["m"],
    "condition": spec["condition"],
    "cell_temperature": spec["cell_temperature"],
    "structure": spec["structure"],
    "n": spec["n"],
  }


def _aminx_worker(args: Any) -> None:
  if args.job is None:
    msg = "aminx worker requires --job"
    raise SystemExit(msg)
  checkpoint = pilot._checkpoint(args)
  pilot._require_checkpoint(checkpoint)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[UnitSpec] = list(job["specs"])
  if len(specs) != 1:
    msg = "aminx worker runs one unit per process"
    raise SystemExit(msg)
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  work = pilot._work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)

  def compute(unit: graded_resume.Unit) -> dict[str, Any]:
    return _sample_aminx(by_id[unit.unit_id], checkpoint)

  graded_resume.run_units(units, compute, cache, resume=bool(args.resume))


def _launch_no_timeout(
  args: Any,
  command: list[str],
  logger: logging.Logger,
) -> subprocess.CompletedProcess[str]:
  """Relaunch a non-zero exit. The attempt has no timeout."""
  last: subprocess.CompletedProcess[str] | None = None
  for attempt in range(1, int(args.max_attempts) + 1):
    logger.info("attempt %s", attempt)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    last = completed
    if completed.returncode == 0:
      return completed
    logger.warning("attempt %s exit %s", attempt, completed.returncode)
  if last is None:
    msg = "max_attempts must be >= 1"
    raise SystemExit(msg)
  return last


def _worker_command(
  args: Any,
  job_path: Path,
  work: Path,
  *,
  aminx: bool,
) -> list[str]:
  resume_flag = "--resume" if args.resume else "--no-resume"
  flag = "--aminx-worker" if aminx else "--oracle-worker"
  executable = sys.executable if aminx else pilot._oracle_python(args)
  return [
    executable,
    str(Path(__file__).resolve()),
    flag,
    "--laser-root",
    str(args.laser_root),
    "--checkpoint",
    str(pilot._checkpoint(args)),
    "--job",
    str(job_path),
    "--work-dir",
    str(work),
    resume_flag,
  ]


def _run_upstream(
  args: Any,
  logger: logging.Logger,
  specs: list[UnitSpec],
  work: Path,
) -> tuple[int, int]:
  if not specs:
    return 0, 0
  checkpoint = pilot._checkpoint(args)
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  cache = graded_resume.cache_dir(work)
  job_path = work / "job_upstream.json"
  job_path.write_text(json.dumps({"specs": specs}), encoding="utf-8")
  before = graded_resume.count_valid(cache, units)
  completed = _launch_no_timeout(
    args,
    _worker_command(args, job_path, work, aminx=False),
    logger,
  )
  if completed.returncode != 0:
    logger.error("upstream worker failed\n%s", completed.stderr[-2000:])
    raise SystemExit(completed.returncode)
  after = graded_resume.count_valid(cache, units)
  if bool(args.resume):
    return min(before, len(units)), max(0, after - before)
  return 0, len(units)


def _run_aminx_unit(
  args: Any,
  logger: logging.Logger,
  spec: UnitSpec,
  work: Path,
) -> tuple[int, int]:
  checkpoint = pilot._checkpoint(args)
  units = _graded_units([spec], checkpoint, Path(__file__).resolve())
  cache = graded_resume.cache_dir(work)
  if bool(args.resume) and graded_resume.load_unit(cache, units[0].cache_key) is not None:
    return 1, 0
  safe = spec["unit_id"].replace("|", "_")
  job_path = work / f"job_{safe}.json"
  job_path.write_text(json.dumps({"specs": [spec]}), encoding="utf-8")
  launched = graded_resume.relaunch_subprocess(
    _worker_command(args, job_path, work, aminx=True),
    directory=cache,
    units=units,
    resume=bool(args.resume),
    max_attempts=int(args.max_attempts),
    timeout_s=float(args.arm_timeout),
    logger=logger,
  )
  if launched.returncode != 0:
    logger.error("aminx unit %s failed\n%s", spec["unit_id"], launched.stderr[-2000:])
    raise SystemExit(launched.returncode)
  return launched.n_reused, launched.n_computed


def _load_cached(
  specs: list[UnitSpec],
  checkpoint: Path,
  work: Path,
) -> dict[str, Any]:
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  cache = graded_resume.cache_dir(work)
  loaded: dict[str, Any] = {}
  for unit in units:
    payload = graded_resume.load_unit(cache, unit.cache_key)
    if payload is None:
      msg = f"unit {unit.unit_id} missing after a zero exit"
      raise SystemExit(msg)
    loaded[unit.unit_id] = payload
  return loaded


def _nest_samples(
  loaded: dict[str, Any],
  specs: list[UnitSpec],
) -> dict[str, dict[str, dict[str, Any]]]:
  nested: dict[str, dict[str, dict[str, Any]]] = {}
  for spec in specs:
    cell = nested.setdefault(spec["cell"], {})
    structure = cell.setdefault(spec["structure"], {})
    structure[spec["run"]] = loaded[spec["unit_id"]]
  return nested


def _grade_fields(grade: stats.CellGrade) -> dict[str, float | str]:
  return {
    "verdict": grade.verdict,
    "mean_delta": float(grade.mean_delta),
    "upper_95": float(grade.upper_95),
    "lower_90": float(grade.lower_90),
    "max_delta": float(grade.max_delta),
    "delta": float(grade.delta),
  }


def _worst(left: str, right: str) -> stats.Verdict:
  if left not in _VERDICT_RANK or right not in _VERDICT_RANK:
    msg = f"unknown verdict {left!r} or {right!r}"
    raise SystemExit(msg)
  chosen = left if _VERDICT_RANK[left] >= _VERDICT_RANK[right] else right
  if chosen == "pass":
    return "pass"
  if chosen == "fail":
    return "fail"
  return "inconclusive"


def _as_sequences(payload: dict[str, Any]) -> Any:
  import numpy as np

  return np.asarray(payload["sequences"], dtype=np.int64)


def _as_chi(payload: dict[str, Any]) -> Any:
  return pilot._chi1_bins(pilot._degrees(payload["chi1_degrees"]))


def _tv_replicates(
  aa_left: Any,
  chi_left: Any,
  aa_right: Any,
  chi_right: Any,
  amino: int,
  idx_left: Any,
  idx_right: Any,
) -> Any:
  """(n_boot,) χ1 TV at one position. A replicate with no modal samples is NaN."""
  import numpy as np

  bins = stats.CHI_BINS
  left_aa = np.asarray(aa_left, dtype=np.int64)[idx_left]
  right_aa = np.asarray(aa_right, dtype=np.int64)[idx_right]
  left_chi = np.asarray(chi_left, dtype=np.int64)[idx_left]
  right_chi = np.asarray(chi_right, dtype=np.int64)[idx_right]
  n_boot = int(idx_left.shape[0])
  boot = np.arange(n_boot, dtype=np.int64)[:, None]
  keep_left = left_aa == amino
  keep_right = right_aa == amino
  if bool(keep_left.any()) and (
    int(left_chi[keep_left].min()) < 0 or int(left_chi[keep_left].max()) >= bins
  ):
    msg = f"chi bins must lie in [0, {bins})"
    raise SystemExit(msg)
  if bool(keep_right.any()) and (
    int(right_chi[keep_right].min()) < 0 or int(right_chi[keep_right].max()) >= bins
  ):
    msg = f"chi bins must lie in [0, {bins})"
    raise SystemExit(msg)
  flat_left = boot * bins + left_chi
  flat_right = boot * bins + right_chi
  counts_left = np.bincount(
    flat_left[keep_left],
    minlength=n_boot * bins,
  ).reshape(n_boot, bins)
  counts_right = np.bincount(
    flat_right[keep_right],
    minlength=n_boot * bins,
  ).reshape(n_boot, bins)
  n_left = counts_left.sum(axis=1)
  n_right = counts_right.sum(axis=1)
  valid = (n_left > 0) & (n_right > 0)
  freq_left = counts_left / np.maximum(n_left, 1)[:, None]
  freq_right = counts_right / np.maximum(n_right, 1)[:, None]
  tv = 0.5 * np.abs(freq_left - freq_right).sum(axis=1)
  return np.where(valid, tv, np.nan)


def _mean_tv_replicates(
  aa_left: Any,
  chi_left: Any,
  aa_right: Any,
  chi_right: Any,
  modal: Any,
  has_chi1: Any,
  mask: Any,
  idx_left: Any,
  idx_right: Any,
) -> Any:
  """Mean χ1 TV over the modal-χ1 screen, per bootstrap replicate."""
  import numpy as np

  designed = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
  table = np.asarray(has_chi1, dtype=np.bool_)
  n_boot = int(idx_left.shape[0])
  totals = np.zeros(n_boot, dtype=np.float64)
  counts = np.zeros(n_boot, dtype=np.float64)
  screened = False
  for position in designed:
    amino = int(modal[position])
    if amino >= int(table.shape[0]) or not bool(table[amino]):
      continue
    screened = True
    tv = _tv_replicates(
      aa_left[:, position],
      chi_left[:, position],
      aa_right[:, position],
      chi_right[:, position],
      amino,
      idx_left,
      idx_right,
    )
    finite = np.isfinite(tv)
    totals[finite] += tv[finite]
    counts[finite] += 1.0
  if not screened:
    msg = "no positions contribute to chi1 total variation"
    raise SystemExit(msg)
  if not bool(np.all(counts > 0)):
    msg = "a chi1 bootstrap replicate lost every chi1 position"
    raise SystemExit(msg)
  return totals / counts


def _draw_rows(n_rows: int, n_boot: int, rng: Any) -> Any:
  import numpy as np

  return rng.integers(0, n_rows, size=(n_boot, n_rows), dtype=np.int64)


def _grade_chi1(
  bundles: dict[str, list[Any]],
  has_chi1: Any,
  alphabet: int,
  *,
  delta_chi: float,
  n_boot: int,
  seed: int,
) -> dict[str, float | str]:
  """χ1 cell grade. No confirmatory χ1 grader exists in ``sample_dist_stats``."""
  import numpy as np

  rng = np.random.default_rng(seed)
  points: list[float] = []
  replicates: list[Any] = []
  for aa_a, chi_a, aa_u1, chi_u1, aa_u2, chi_u2, mask in zip(
    bundles["aa_A"],
    bundles["chi_A"],
    bundles["aa_U1"],
    bundles["chi_U1"],
    bundles["aa_U2"],
    bundles["chi_U2"],
    bundles["mask"],
    strict=True,
  ):
    modal = pilot._modal_aa(aa_u1, alphabet)
    try:
      left = stats.mean_tv_chi1(aa_a, chi_a, aa_u1, chi_u1, modal, has_chi1, mask)
      right = stats.mean_tv_chi1(aa_u2, chi_u2, aa_u1, chi_u1, modal, has_chi1, mask)
    except ValueError as exc:
      msg = f"chi1 grade: {exc}"
      raise SystemExit(msg) from exc
    points.append(left - right)
    n_rows = int(np.asarray(aa_u1).shape[0])
    idx_left = _draw_rows(n_rows, n_boot, rng)
    idx_u1 = _draw_rows(n_rows, n_boot, rng)
    idx_right = _draw_rows(n_rows, n_boot, rng)
    drawn_left = _mean_tv_replicates(
      aa_a, chi_a, aa_u1, chi_u1, modal, has_chi1, mask, idx_left, idx_u1,
    )
    drawn_right = _mean_tv_replicates(
      aa_u2, chi_u2, aa_u1, chi_u1, modal, has_chi1, mask, idx_right, idx_u1,
    )
    replicates.append(drawn_left - drawn_right)
  point = np.asarray(points, dtype=np.float64)
  pooled = np.stack(replicates, axis=0).mean(axis=0)
  upper_95 = stats._quantile(pooled, stats._Q_95_UPPER)
  lower_90 = stats._quantile(pooled, stats._Q_90_LO)
  mean_delta = float(point.mean())
  max_delta = float(point.max())
  return {
    "verdict": stats.confirmatory_verdict(upper_95, lower_90, max_delta, delta_chi),
    "mean_delta": mean_delta,
    "upper_95": upper_95,
    "lower_90": lower_90,
    "max_delta": max_delta,
    "delta": delta_chi,
    "delta_chi": delta_chi,
  }


def _control_failed(verdict: str) -> bool:
  return verdict == "fail"


def assemble_result(
  samples: dict[str, dict[str, dict[str, Any]]],
  structures: dict[str, dict[str, str]],
  cells_selected: list[str],
  *,
  n_boot: int,
  bootstrap_seeds: dict[str, int],
  n_reused: int,
  n_computed: int,
  script_sha256: str,
  smoke: bool,
) -> dict[str, Any]:
  """Grade sequence and χ1. A cell verdict is the worse of the two."""
  import numpy as np

  has_chi1, alphabet = pilot._chi_table()
  cell_grades: list[stats.CellGrade] = []
  cells: dict[str, Any] = {}
  control_verdicts: list[str] = []
  for cell in cells_selected:
    table = _table(cell, smoke=smoke)
    per_structure: dict[str, float] = {}
    bundles: dict[str, list[Any]] = {
      "A": [],
      "U1": [],
      "U2": [],
      "mask": [],
      "aa_A": [],
      "chi_A": [],
      "aa_U1": [],
      "chi_U1": [],
      "aa_U2": [],
      "chi_U2": [],
    }
    controls: dict[str, dict[str, list[Any]]] = {
      "CTRL_m": {"seq": [], "aa": [], "chi": []},
    }
    for structure in structures:
      runs = samples[cell][structure]
      mask = np.asarray(runs["U1"]["mask"], dtype=np.bool_)
      aminx_mask = np.asarray(runs["A"]["mask"], dtype=np.bool_)
      if mask.shape != aminx_mask.shape or not np.array_equal(mask, aminx_mask):
        msg = f"designed masks of A and U1 differ for {structure}"
        raise SystemExit(msg)
      columns = {run: _as_sequences(runs[run]) for run in ("U1", "U2", "A", "CTRL_m")}
      chi = {run: _as_chi(runs[run]) for run in ("U1", "U2", "A", "CTRL_m")}
      bundles["A"].append(columns["A"])
      bundles["U1"].append(columns["U1"])
      bundles["U2"].append(columns["U2"])
      bundles["mask"].append(mask)
      bundles["aa_A"].append(columns["A"])
      bundles["chi_A"].append(chi["A"])
      bundles["aa_U1"].append(columns["U1"])
      bundles["chi_U1"].append(chi["U1"])
      bundles["aa_U2"].append(columns["U2"])
      bundles["chi_U2"].append(chi["U2"])
      controls["CTRL_m"]["seq"].append(columns["CTRL_m"])
      controls["CTRL_m"]["aa"].append(columns["CTRL_m"])
      controls["CTRL_m"]["chi"].append(chi["CTRL_m"])
      per_structure[structure] = float(
        stats.delta_s(columns["A"], columns["U1"], columns["U2"], alphabet, mask),
      )
    seed = bootstrap_seeds[cell]
    sequence = stats.grade_condition(
      bundles["A"],
      bundles["U1"],
      bundles["U2"],
      bundles["mask"],
      alphabet,
      delta=float(table["delta"]),
      n_boot=n_boot,
      seed=seed,
    )
    chi1 = _grade_chi1(
      bundles,
      has_chi1,
      alphabet,
      delta_chi=float(table["delta_chi"]),
      n_boot=n_boot,
      seed=seed,
    )
    verdict = _worst(str(sequence.verdict), str(chi1["verdict"]))
    cell_grades.append(
      stats.CellGrade(
        verdict=verdict,
        mean_delta=float(sequence.mean_delta),
        upper_95=float(sequence.upper_95),
        lower_90=float(sequence.lower_90),
        max_delta=float(sequence.max_delta),
        delta=float(sequence.delta),
      ),
    )
    graded_controls: dict[str, Any] = {}
    for run, payload in controls.items():
      control_sequence = stats.grade_condition(
        payload["seq"],
        bundles["U1"],
        bundles["U2"],
        bundles["mask"],
        alphabet,
        delta=float(table["delta"]),
        n_boot=n_boot,
        seed=seed,
      )
      control_bundles = {
        "aa_A": payload["aa"],
        "chi_A": payload["chi"],
        "aa_U1": bundles["aa_U1"],
        "chi_U1": bundles["chi_U1"],
        "aa_U2": bundles["aa_U2"],
        "chi_U2": bundles["chi_U2"],
        "mask": bundles["mask"],
      }
      control_chi = _grade_chi1(
        control_bundles,
        has_chi1,
        alphabet,
        delta_chi=float(table["delta_chi"]),
        n_boot=n_boot,
        seed=seed,
      )
      control_verdict = _worst(str(control_sequence.verdict), str(control_chi["verdict"]))
      graded_controls[run] = {
        "verdict": control_verdict,
        "sequence": _grade_fields(control_sequence),
        "chi1": control_chi,
      }
      if _control_failed(control_verdict):
        control_verdicts.append("fail")
      else:
        control_verdicts.append(control_verdict)
    cells[cell] = {
      "verdict": verdict,
      "sequence": _grade_fields(sequence),
      "chi1": chi1,
      "m": float(table["m"]),
      "h_hat": float(table["h_hat"]),
      "n": int(table["n"]),
      "controls": graded_controls,
      "per_structure_delta": per_structure,
    }
  sidecar = stats.grade_sidecar(cell_grades)
  payload = {
    "sidecar_verdict": sidecar.verdict,
    "controls_all_fail": all(verdict == "fail" for verdict in control_verdicts),
    "n_controls": len(control_verdicts),
    "cells": cells,
    "cells_selected": list(cells_selected),
    "structures": structures,
    "pilot_run_id": PILOT_RUN_ID if PILOT_RUN_ID is not None else "none (smoke)",
    "n_reused": n_reused,
    "n_computed": n_computed,
    "script_sha256": script_sha256,
    "smoke": smoke,
  }
  if tuple(payload) != _TOP_LEVEL:
    msg = f"result keys {tuple(payload)} != {_TOP_LEVEL}"
    raise RuntimeError(msg)
  return payload


def _emit(path: Path, payload: dict[str, Any]) -> None:
  text = json.dumps(payload, indent=2) + "\n"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")
  env = os.environ.get("BTH_RESULTS_PATH")
  if not env:
    return
  dest = Path(env)
  if dest.resolve() == path.resolve():
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(text, encoding="utf-8")


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_sample_dist_confirm")
  logger.setLevel(logging.INFO)
  return logger


def _parent(args: Any) -> None:
  smoke = bool(args.smoke)
  if smoke and args.work_dir is None:
    args.work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / _WORK_SMOKE
  cells = selected_cells(None if args.cells is None else str(args.cells), smoke=smoke)
  if args.out is None:
    msg = "laser_sample_dist_confirm requires --out"
    raise SystemExit(msg)
  if int(args.max_attempts) < 1:
    msg = "--max-attempts must be >= 1"
    raise SystemExit(msg)
  if float(args.arm_timeout) <= 0:
    msg = "--arm-timeout must be positive"
    raise SystemExit(msg)
  logger = _logger()
  root = Path(args.laser_root)
  checkpoint = pilot._checkpoint(args)
  pilot._require_checkpoint(checkpoint)
  work = pilot._work_dir(args, _WORK_SMOKE if smoke else _WORK)
  structures = pinned_structures(root, smoke=smoke)
  n_boot = _SMOKE_BOOT if smoke else N_BOOT
  used: set[int] = set()
  specs: list[UnitSpec] = []
  for cell in cells:
    n = _SMOKE_N if smoke else int(_table(cell, smoke=False)["n"])
    specs.extend(unit_specs(structures, [cell], n, used, smoke=smoke))
  seeds = bootstrap_seeds(cells, used)
  upstream = [spec for spec in specs if spec["run"] in _UPSTREAM]
  aminx = [spec for spec in specs if spec["run"] not in _UPSTREAM]
  n_reused, n_computed = _run_upstream(args, logger, upstream, work)
  for spec in aminx:
    logger.info("aminx unit %s", spec["unit_id"])
    reused, computed = _run_aminx_unit(args, logger, spec, work)
    n_reused += reused
    n_computed += computed
  loaded = _load_cached(specs, checkpoint, work)
  payload = assemble_result(
    _nest_samples(loaded, specs),
    structures,
    cells,
    n_boot=n_boot,
    bootstrap_seeds=seeds,
    n_reused=n_reused,
    n_computed=n_computed,
    script_sha256=pilot._sha256(Path(__file__).resolve()),
    smoke=smoke,
  )
  _emit(Path(args.out), payload)


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  if args.oracle_worker:
    _upstream_worker(args)
    return
  if args.aminx_worker:
    _aminx_worker(args)
    return
  _parent(args)


if __name__ == "__main__":
  main()
