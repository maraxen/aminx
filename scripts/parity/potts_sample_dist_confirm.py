"""Confirmatory Potts distributional protocol for plain@1.0 and refine@0.3.

The pilot run ``PILOT_RUN_ID`` fixed δ, m, ĥ and n. Those values are the
module-level ``CELLS`` table. plain@0.3 is excluded: its pilot shim check
failed, which the spec grades ``instrument_invalid``.

One resumable unit is one (cell, structure, run). Upstream units (U1 shimmed,
U2 unshimmed) call the pilot sampler. Aminx units (A at T, CTRL_m at m·T, and
CTRL_ntoc for plain@1.0) call ``aminx.host.runner.sample``, which is
``PottsMPNNDriver`` purpose ``sample`` and therefore ``SampleStages``, in
float32 with ``jax.random`` keys derived from the unit seed. CTRL_ntoc
monkeypatches autoregressive order generation inside that unit's process only.
There is no whole-run timeout; each aminx unit has its own timeout.
"""

from __future__ import annotations

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
import potts_graded_common as common
import potts_sample_dist_pilot as pilot
import sample_dist_stats as stats

PILOT_RUN_ID = "6b47d40c-32e4-434b-b94f-fa2f8c077f2d"
CELLS: dict[str, dict[str, float | int]] = {
  "plain@1.0": {
    "delta": 0.01,
    "m": 1.1,
    "h_hat": 0.0022991985350076035,
    "n": 1000,
  },
  "refine@0.3": {
    "delta": 0.01,
    "m": 1.5,
    "h_hat": 0.0020500332952815914,
    "n": 1000,
  },
}
N_BOOT = 2000
STRUCTURES = {
  "inputs/example_pdbs/3gg7.pdb": (
    "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712"
  ),
  "inputs/example_pdbs/4jox.pdb": (
    "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8"
  ),
  "inputs/example_pdbs/6w25.pdb": (
    "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086"
  ),
  "inputs/example_pdbs/swe1_ligand.pdb": (
    "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6"
  ),
}
_EXCLUDED_CELL = "plain@0.3"
_WORK = "potts_sample_dist_confirm"
_WORK_SMOKE = "potts_sample_dist_confirm_smoke"
_SMOKE_N = 50
_SMOKE_BOOT = 50
_SMOKE_STRUCTURE = "3gg7"
_PILOT_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
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
  structure: str
  run: str
  m: float | None
  shim: bool
  kind: str
  n: int
  seed: int
  pdb: str
  pdb_sha256: str
  ntoc: bool


def _parse(argv: list[str]) -> Any:
  parser = common.build_parser(__doc__)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument(
    "--cells",
    default=None,
    help="comma-separated subset of plain@1.0,refine@0.3; plain@0.3 is refused",
  )
  parser.add_argument("--aminx-worker", action="store_true")
  return parser.parse_args(argv)


def selected_cells(raw: str | None, *, smoke: bool) -> list[str]:
  """Cells this process will run. Anything outside ``CELLS`` is refused."""
  if raw is None:
    chosen = ["plain@1.0"] if smoke else list(CELLS)
  else:
    chosen = [part.strip() for part in raw.split(",") if part.strip()]
  if not chosen:
    msg = "--cells selected nothing; allowed: " + ",".join(CELLS)
    raise SystemExit(msg)
  for cell in chosen:
    if cell == _EXCLUDED_CELL:
      msg = (
        "plain@0.3 is excluded: the pilot shim check failed "
        "(instrument_invalid); this confirmatory run refuses it"
      )
      raise SystemExit(msg)
    if cell not in CELLS:
      msg = f"unknown cell {cell}; allowed: {','.join(CELLS)}"
      raise SystemExit(msg)
  ordered = [cell for cell in CELLS if cell in set(chosen)]
  if smoke and ordered != ["plain@1.0"]:
    msg = "smoke runs plain@1.0 on 3gg7 only"
    raise SystemExit(msg)
  return ordered


def _cell_temperature(cell: str) -> float:
  return float(cell.split("@", 1)[1])


def _cell_kind(cell: str) -> str:
  return cell.split("@", 1)[0]


def _runs(cell: str) -> tuple[str, ...]:
  if cell == "refine@0.3":
    return ("U1", "U2", "A", "CTRL_m")
  if cell == "plain@1.0":
    return ("U1", "U2", "A", "CTRL_m", "CTRL_ntoc")
  msg = f"no run list for {cell}"
  raise SystemExit(msg)


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
) -> list[UnitSpec]:
  """One spec per (cell, structure, run). Seeds are claimed into ``used``."""
  drafts: list[UnitSpec] = []
  labels: list[str] = []
  for cell in cells:
    kind = _cell_kind(cell)
    temperature = _cell_temperature(cell)
    factor = float(CELLS[cell]["m"])
    for structure, payload in structures.items():
      for run in _runs(cell):
        unit_id = _unit_id(cell, structure, run, n)
        labels.append(unit_id)
        drafts.append(
          {
            "unit_id": unit_id,
            "cell": cell,
            "condition": kind,
            "cell_temperature": temperature,
            "sample_temperature": temperature * factor if run == "CTRL_m" else temperature,
            "structure": structure,
            "run": run,
            "m": factor if run == "CTRL_m" else None,
            "shim": run == "U1",
            "kind": kind,
            "n": n,
            "seed": 0,
            "pdb": payload["pdb"],
            "pdb_sha256": payload["sha256"],
            "ntoc": run == "CTRL_ntoc",
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


def pinned_structures(root: Path, *, smoke: bool) -> dict[str, dict[str, str]]:
  """Confirmatory PDBs, checked with ``common.check_pdb`` before use."""
  pinned: dict[str, dict[str, str]] = {}
  for relative, digest in STRUCTURES.items():
    stem = Path(relative).stem
    if smoke and stem != _SMOKE_STRUCTURE:
      continue
    path = root / relative
    common.check_pdb(path, digest)
    pinned[stem] = {"pdb": str(path), "relative": relative, "sha256": digest}
  return pinned


def _graded_units(
  specs: list[UnitSpec],
  checkpoint: Path,
  script: Path,
) -> list[graded_resume.Unit]:
  script_sha = common.sha256(script)
  digest = common.sha256(checkpoint)
  units: list[graded_resume.Unit] = []
  for spec in specs:
    arm = "upstream" if spec["run"] in ("U1", "U2") else "aminx"
    units.append(
      graded_resume.Unit(
        unit_id=spec["unit_id"],
        arm_id=arm,
        checkpoint_sha256=digest,
        input_sha256=pilot._input_sha(spec["pdb_sha256"], spec["kind"]),
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
    "structure": spec["structure"],
    "run": spec["run"],
    "m": spec["m"],
    "shim": spec["shim"],
    "kind": spec["kind"],
    "n": spec["n"],
    "seed": spec["seed"],
    "pdb": spec["pdb"],
    "pdb_sha256": spec["pdb_sha256"],
  }


def _upstream_worker(args: Any) -> None:
  if args.job is None:
    msg = "upstream worker requires --job"
    raise SystemExit(msg)
  checkpoint = common.checkpoint_path(args)
  common.require_checkpoint(checkpoint, pilot.CHECKPOINT_SHA256)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[UnitSpec] = list(job["specs"])
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  work = common.work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))[1]
  prepared: dict[str, Any] | None = None
  feats: dict[tuple[str, str], dict[str, Any]] = {}
  if remaining > 0:
    prepared = common.load_potts_oracle(
      Path(args.potts_root),
      checkpoint,
      double=False,
    )
    for spec in specs:
      key = (spec["kind"], spec["structure"])
      if key not in feats:
        feats[key] = pilot._featurize(
          Path(args.potts_root),
          Path(spec["pdb"]),
          spec["kind"],
        )

  def compute(unit: graded_resume.Unit) -> dict[str, Any]:
    if prepared is None:
      msg = "oracle model was not loaded"
      raise RuntimeError(msg)
    spec = by_id[unit.unit_id]
    return pilot._sample_unit(
      Path(args.potts_root),
      prepared["model"],
      feats[(spec["kind"], spec["structure"])],
      _pilot_spec(spec),
    )

  graded_resume.run_units(units, compute, cache, resume=bool(args.resume))


def _prng_seed(seed: int) -> int:
  """uint32 seed for ``jax.random.PRNGKey``, derived from the unit seed."""
  digest = hashlib.sha256(f"confirm-prng|{seed}".encode()).digest()
  return int.from_bytes(digest[:4], "little")


def _install_ntoc() -> None:
  """Force N-to-C among designed positions. Installed only in this process."""
  import jax.numpy as jnp

  from aminx.families.potts_mpnn import decode as decode_mod

  original = decode_mod.schedule_groups

  def _ntoc(
    tie_groups: Any,
    pad_valid: Any,
    randn: Any,
    chain_mask: Any,
    chain_m_pos: Any,
    present: Any,
    decoding_order: Any,
  ) -> Any:
    del decoding_order
    length = pad_valid.shape[0]
    index = jnp.arange(length, dtype=jnp.int32)
    designed = (chain_mask * chain_m_pos * present) > jnp.asarray(0, dtype=chain_mask.dtype)
    key = jnp.where(designed, index, index + jnp.asarray(length, dtype=jnp.int32))
    order = jnp.argsort(key).astype(jnp.int32)
    return original(
      tie_groups,
      pad_valid,
      randn,
      chain_mask,
      chain_m_pos,
      present,
      order,
    )

  decode_mod.schedule_groups = _ntoc  # ty: ignore[invalid-assignment]


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
  """Map production model indices onto the pilot TV alphabet."""
  import numpy as np

  from aminx.families.potts_mpnn.featurize import MODEL_ALPHABET

  array = np.asarray(tokens, dtype=np.int32)
  if array.ndim != 2:
    msg = f"aminx sequences have ndim {array.ndim}, expected 2"
    raise SystemExit(msg)
  if MODEL_ALPHABET != _PILOT_ALPHABET:
    table = np.asarray(
      [_PILOT_ALPHABET.index(letter) for letter in MODEL_ALPHABET],
      dtype=np.int32,
    )
    array = table[array]
  if array.size and (int(array.min()) < 0 or int(array.max()) >= len(_PILOT_ALPHABET)):
    msg = "aminx token outside the pilot alphabet"
    raise SystemExit(msg)
  return [[int(token) for token in row] for row in array.tolist()]


def _design_mask_from_pdb(pdb: Path, options: Any) -> list[bool]:
  """Same predicate as the pilot, on the features the driver will sample."""
  from aminx.families.potts_mpnn.driver import _featurize_one
  from aminx.families.potts_mpnn.featurize import parse_pdb_upstream

  parsed = parse_pdb_upstream(pdb, skip_gaps=bool(options.skip_gaps))[0]
  features = _featurize_one(parsed, options, str(parsed["name"]))
  feat = {
    "chain_m": features.chain_m,
    "chain_m_pos": features.chain_m_pos,
    "mask": features.present,
  }
  return pilot._design_mask(feat)


def _potts_options(kind: str) -> Any:
  from aminx.run.options import PottsMPNNOptions

  if kind == "plain":
    return PottsMPNNOptions(
      optimization_mode="none",
      pssm_multi=0.0,
      pssm_bias_flag=False,
      pssm_log_odds_flag=False,
    )
  if kind == "refine":
    return PottsMPNNOptions(
      optimization_mode="potts",
      optimization_temperature=float(pilot._OPTIMIZATION_TEMPERATURE),
      pssm_multi=0.0,
      pssm_bias_flag=False,
      pssm_log_odds_flag=False,
    )
  msg = f"no Potts options for kind {kind}"
  raise SystemExit(msg)


def _sample_aminx(spec: UnitSpec, checkpoint: Path) -> dict[str, Any]:
  """Production sample path. Randomness comes from the unit seed only."""
  import jax

  from aminx.host.runner import sample as runner_sample
  from aminx.run.specs import SamplingSpecification

  jax.config.update("jax_enable_x64", False)
  if spec["ntoc"]:
    _install_ntoc()
  options = _potts_options(spec["kind"])
  sampling = SamplingSpecification(
    inputs=spec["pdb"],
    model_family="pottsmpnn",
    model_local_path=checkpoint,
    num_samples=spec["n"],
    samples_chunk_size=8,
    return_logits=False,
    random_seed=_prng_seed(spec["seed"]),
    temperature=spec["sample_temperature"],
    backbone_noise=0.0,
    potts_mpnn=options,
  )
  result = runner_sample(sampling)
  _assert_float32(result)
  arrays = result["structures"]["0"]["arrays"]
  key = "refined_sequence" if spec["kind"] == "refine" else "sequence"
  if key not in arrays:
    msg = f"production sample did not emit {key}"
    raise SystemExit(msg)
  sequences = _map_tokens(arrays[key])
  if len(sequences) != spec["n"]:
    msg = f"aminx drew {len(sequences)} sequences, expected {spec['n']}"
    raise SystemExit(msg)
  mask = _design_mask_from_pdb(Path(spec["pdb"]), options)
  if len(mask) != len(sequences[0]):
    msg = "aminx design mask length does not match the sampled sequences"
    raise SystemExit(msg)
  return {
    "sequences": sequences,
    "mask": mask,
    "seed": spec["seed"],
    "sample_temperature": spec["sample_temperature"],
    "shim": False,
    "run": spec["run"],
    "m": spec["m"],
    "condition": spec["condition"],
    "cell_temperature": spec["cell_temperature"],
    "structure": spec["structure"],
    "kind": spec["kind"],
    "n": spec["n"],
  }


def _aminx_worker(args: Any) -> None:
  if args.job is None:
    msg = "aminx worker requires --job"
    raise SystemExit(msg)
  checkpoint = common.checkpoint_path(args)
  common.require_checkpoint(checkpoint, pilot.CHECKPOINT_SHA256)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[UnitSpec] = list(job["specs"])
  if len(specs) != 1:
    msg = "aminx worker runs one unit per process"
    raise SystemExit(msg)
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  work = common.work_dir(args, _WORK)
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
  executable = sys.executable if aminx else common.oracle_python(args)
  return [
    executable,
    str(Path(__file__).resolve()),
    flag,
    "--potts-root",
    str(args.potts_root),
    "--checkpoint",
    str(common.checkpoint_path(args)),
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
  checkpoint = common.checkpoint_path(args)
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
  checkpoint = common.checkpoint_path(args)
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
  }


def assemble_result(
  samples: dict[str, dict[str, dict[str, Any]]],
  structures: dict[str, dict[str, str]],
  cells_selected: list[str],
  *,
  n: int,
  n_boot: int,
  bootstrap_seeds: dict[str, int],
  n_reused: int,
  n_computed: int,
  script_sha256: str,
  smoke: bool,
) -> dict[str, Any]:
  """Grade A against U1/U2 and require every control verdict to be fail."""
  import numpy as np

  cell_grades: list[stats.CellGrade] = []
  cells: dict[str, Any] = {}
  control_verdicts: list[str] = []
  for cell in cells_selected:
    table = CELLS[cell]
    per_structure: dict[str, float] = {}
    bundles: dict[str, list[Any]] = {
      "A": [],
      "U1": [],
      "U2": [],
      "mask": [],
    }
    controls: dict[str, list[Any]] = {run: [] for run in _runs(cell) if run.startswith("CTRL_")}
    for structure in structures:
      runs = samples[cell][structure]
      mask = np.asarray(runs["U1"]["mask"], dtype=np.bool_)
      aminx_mask = np.asarray(runs["A"]["mask"], dtype=np.bool_)
      if mask.shape != aminx_mask.shape or not np.array_equal(mask, aminx_mask):
        msg = f"designed masks of A and U1 differ for {structure}"
        raise SystemExit(msg)
      columns = {
        run: np.asarray(runs[run]["sequences"])
        for run in ("U1", "U2", "A", *controls)
      }
      bundles["A"].append(columns["A"])
      bundles["U1"].append(columns["U1"])
      bundles["U2"].append(columns["U2"])
      bundles["mask"].append(mask)
      for run in controls:
        controls[run].append(columns[run])
      per_structure[structure] = float(
        stats.delta_s(
          columns["A"],
          columns["U1"],
          columns["U2"],
          common.VOCAB,
          mask,
        ),
      )
    seed = bootstrap_seeds[cell]
    main = stats.grade_condition(
      bundles["A"],
      bundles["U1"],
      bundles["U2"],
      bundles["mask"],
      common.VOCAB,
      delta=float(table["delta"]),
      n_boot=n_boot,
      seed=seed,
    )
    cell_grades.append(main)
    graded_controls: dict[str, dict[str, float | str]] = {}
    for run, arrays in controls.items():
      graded = stats.grade_condition(
        arrays,
        bundles["U1"],
        bundles["U2"],
        bundles["mask"],
        common.VOCAB,
        delta=float(table["delta"]),
        n_boot=n_boot,
        seed=seed,
      )
      graded_controls[run] = _grade_fields(graded)
      control_verdicts.append(graded.verdict)
    cells[cell] = {
      **_grade_fields(main),
      "delta": float(table["delta"]),
      "m": float(table["m"]),
      "h_hat": float(table["h_hat"]),
      "n": n,
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
    "pilot_run_id": PILOT_RUN_ID,
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


def _parent(args: Any) -> None:
  if args.out is None:
    msg = "potts_sample_dist_confirm requires --out"
    raise SystemExit(msg)
  if int(args.max_attempts) < 1:
    msg = "--max-attempts must be >= 1"
    raise SystemExit(msg)
  logger = common.logger_for("potts_sample_dist_confirm")
  smoke = bool(args.smoke)
  if smoke and args.work_dir is None:
    args.work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / _WORK_SMOKE
  cells = selected_cells(None if args.cells is None else str(args.cells), smoke=smoke)
  root = Path(args.potts_root)
  checkpoint = common.checkpoint_path(args)
  common.require_checkpoint(checkpoint, pilot.CHECKPOINT_SHA256)
  work = common.work_dir(args, _WORK)
  structures = pinned_structures(root, smoke=smoke)
  n = _SMOKE_N if smoke else int(CELLS[cells[0]]["n"])
  n_boot = _SMOKE_BOOT if smoke else N_BOOT
  used: set[int] = set()
  specs = unit_specs(structures, cells, n, used)
  seeds = bootstrap_seeds(cells, used)
  upstream = [spec for spec in specs if spec["run"] in ("U1", "U2")]
  aminx = [spec for spec in specs if spec["run"] not in ("U1", "U2")]
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
    n=n,
    n_boot=n_boot,
    bootstrap_seeds=seeds,
    n_reused=n_reused,
    n_computed=n_computed,
    script_sha256=common.sha256(Path(__file__).resolve()),
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
