"""LASEr pilot for the distributional protocol (do not run from the fixer).

Exploratory. It cites nothing. Per (condition, temperature) it draws upstream
runs U1, U2, U3 and ``C_m`` (m in {1.1, 1.25, 1.5}) on the pinned pilot
structures, then calls ``sample_dist_stats`` for ``δ``, ``m``, the shim check,
and the χ1 total variation. Those values are what the later confirmatory
sidecar commits. This script does not write a ``.bth.toml``. Sampling is
upstream only.

One resumable unit is one (condition, temperature, structure, run). The body
is written before the stamp, keyed by the script sha256 and the inputs, via
``graded_resume``. A crash loses at most the unit in flight. Attempts are
relaunched with no whole-run timeout.

Pilot structures are entries 1-2 of the ``laser_score_parity`` list: the first
two ``ordered_ids`` in ``tests/fixtures/laser/fixtures_manifest.toml``. The
confirmatory sentence names ``4jnj-1_prot.pdb`` separately from entries 3-6,
and ``b0_extras`` repeats those same two ids, so the pair is not the example
PDB. Each file is pinned by the path and SHA-256 that vehicle checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, TypedDict

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

import graded_resume
import sample_dist_stats as stats

CHECKPOINT_NAME = "laser_weights_0p1A_nothing_heldout.pt"
CHECKPOINT_SHA256 = "304fe02a4807c310bdd9d68c988ae87619da3cf2025d5c223fb31030aa411173"
# Entries 1-2 of fixtures_manifest.toml ordered_ids, 1-indexed. Digests are the
# [files] values laser_score_parity hashes for those names.
PILOT_PDB_SHA256: tuple[tuple[str, str], ...] = (
  ("101m_1.pdb", "5ee11918c0c71703859cc057cd3225583fb397909ba0e294c1f97f8e39448a9f"),
  ("102m_1.pdb", "cfb756a7e48b87344d4c9fb76ff93b29596f6b400c207a0e45f53066358e6ee4"),
)
_WORK = "laser_sample_dist_pilot"
_WORK_SMOKE = "laser_sample_dist_pilot_smoke"
_N = 1000
_N_ESCALATED = 4000
_M_GRID = (1.1, 1.25, 1.5)
# One sequence draw plus χ1..χ4. Upstream ``sample`` calls Categorical.sample
# once per decoding-order column for the sequence and once per χ angle.
_DRAWS_PER_RESIDUE = 5


class _Condition(TypedDict):
  name: str
  temperature: float
  min_p: float


class _UnitSpec(TypedDict):
  unit_id: str
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


def _parse(argv: list[str]) -> Any:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=_default_laser_root())
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--oracle-python", default=os.environ.get("LASER_ORACLE_PYTHON"))
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--max-attempts", type=int, default=3)
  parser.add_argument("--oracle-worker", action="store_true")
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument(
    "--cells",
    default=None,
    help=(
      "comma-separated cell keys (e.g. plain@0.3,refine@0.3) to run a SUBSET of "
      "the spec grid; all_cells_derived then covers only these, and the selection "
      "is recorded as cells_selected so a subset run cannot pass for the full grid"
    ),
  )
  return parser.parse_args(argv)


def _checkpoint(args: Any) -> Path:
  if args.checkpoint is not None:
    return Path(args.checkpoint)
  return Path(args.laser_root) / "model_weights" / CHECKPOINT_NAME


def _require_checkpoint(path: Path) -> None:
  digest = _sha256(path)
  if digest != CHECKPOINT_SHA256:
    msg = f"{path} sha256 {digest} != {CHECKPOINT_SHA256}"
    raise SystemExit(msg)


def _manifest() -> dict[str, Any]:
  path = _repo() / "tests" / "fixtures" / "laser" / "fixtures_manifest.toml"
  loaded = tomllib.loads(path.read_text(encoding="utf-8"))
  if not isinstance(loaded, dict):
    msg = f"{path} is not a table"
    raise SystemExit(msg)
  return loaded


def _require_pilot_entries() -> None:
  """Fail closed if the manifest no longer names these two files as entries 1-2."""
  manifest = _manifest()
  expected = [Path(name).stem for name, _digest in PILOT_PDB_SHA256]
  ordered = [str(item) for item in manifest["ordered_ids"]]
  extras = [str(item) for item in manifest["b0_extras"]]
  if ordered[:2] != expected or extras[:2] != expected:
    msg = (
      "laser_score_parity entries 1-2 are no longer unambiguous: "
      f"ordered_ids[:2]={ordered[:2]} b0_extras[:2]={extras[:2]} pin={expected}"
    )
    raise SystemExit(msg)
  files = manifest["files"]
  if not isinstance(files, dict):
    msg = "fixtures_manifest.toml [files] is not a table"
    raise SystemExit(msg)
  for name, digest in PILOT_PDB_SHA256:
    recorded = files.get(name)
    if recorded != digest:
      msg = f"manifest digest for {name} is {recorded}, driver pin is {digest}"
      raise SystemExit(msg)


def _hash_seed(label: str) -> int:
  digest = hashlib.sha256(f"laser-sample-dist-pilot-v1:{label}".encode()).digest()
  return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _claim(seed: int, used: set[int]) -> int:
  while seed in used:
    seed += 1
  used.add(seed)
  return seed


def _conditions(*, smoke: bool) -> list[_Condition]:
  """Spec LASEr grid: three temperatures at min_p=0, plus T=0.3 at min_p=0.05."""
  rows: list[_Condition] = [
    {"name": "min_p0", "temperature": 0.1, "min_p": 0.0},
    {"name": "min_p0", "temperature": 0.3, "min_p": 0.0},
    {"name": "min_p0", "temperature": 1.0, "min_p": 0.0},
    {"name": "min_p0.05", "temperature": 0.3, "min_p": 0.05},
  ]
  if smoke:
    return rows[:1]
  return rows


def _required_cells(*, smoke: bool) -> list[str]:
  if smoke:
    return [_cell_key("min_p0", 0.1)]
  cells = [_cell_key("min_p0", temperature) for temperature in (0.1, 0.3, 1.0)]
  cells.append(_cell_key("min_p0.05", 0.3))
  return cells


def _cell_key(name: str, temperature: float) -> str:
  return f"{name}@{temperature:.1f}"


def _unit_id(condition: str, temperature: float, structure: str, run: str, n: int) -> str:
  return f"{condition}|{temperature:.1f}|{structure}|{run}|{n}"


def _runs() -> list[tuple[str, bool, float | None]]:
  """(run, shim, m). ``m`` is None for the three upstream replicates."""
  rows: list[tuple[str, bool, float | None]] = [
    ("U1", True, None),
    ("U2", False, None),
    ("U3", True, None),
  ]
  rows.extend((f"C_{m}", True, m) for m in _M_GRID)
  return rows


def _structures(*, smoke: bool) -> dict[str, dict[str, str]]:
  """Pin ``tests/fixtures/laser/{101m_1,102m_1}.pdb`` by path and SHA-256."""
  _require_pilot_entries()
  folder = _repo() / "tests" / "fixtures" / "laser"
  rows = PILOT_PDB_SHA256[:1] if smoke else PILOT_PDB_SHA256
  pinned: dict[str, dict[str, str]] = {}
  for name, digest in rows:
    path = folder / name
    got = _sha256(path)
    if got != digest:
      msg = f"{path} sha256 {got} != {digest}"
      raise SystemExit(msg)
    pinned[path.stem] = {
      "pdb": str(path),
      "relative": f"tests/fixtures/laser/{name}",
      "sha256": digest,
    }
  return pinned


def _input_sha(pdb_sha256: str, min_p: float) -> str:
  blob = json.dumps(
    {"min_p": min_p, "pdb_sha256": pdb_sha256},
    sort_keys=True,
    separators=(",", ":"),
  )
  return hashlib.sha256(blob.encode()).hexdigest()


def _unit_specs(
  structures: dict[str, dict[str, str]],
  conditions: list[_Condition],
  n: int,
  used: set[int],
) -> list[_UnitSpec]:
  specs: list[_UnitSpec] = []
  for condition in conditions:
    for structure, payload in structures.items():
      for run, shim, factor in _runs():
        scale = 1.0 if factor is None else factor
        unit_id = _unit_id(
          condition["name"],
          condition["temperature"],
          structure,
          run,
          n,
        )
        specs.append(
          {
            "unit_id": unit_id,
            "condition": condition["name"],
            "cell_temperature": condition["temperature"],
            "sample_temperature": condition["temperature"] * scale,
            "min_p": condition["min_p"],
            "structure": structure,
            "run": run,
            "m": factor,
            "shim": shim,
            "n": n,
            "seed": _claim(_hash_seed(unit_id), used),
            "pdb": payload["pdb"],
            "pdb_sha256": payload["sha256"],
          },
        )
  return specs


def _graded_units(specs: list[_UnitSpec], checkpoint: Path) -> list[graded_resume.Unit]:
  script_sha = _sha256(Path(__file__).resolve())
  digest = _sha256(checkpoint)
  return [
    graded_resume.Unit(
      unit_id=spec["unit_id"],
      arm_id="upstream",
      checkpoint_sha256=digest,
      input_sha256=_input_sha(spec["pdb_sha256"], spec["min_p"]),
      script_sha256=script_sha,
    )
    for spec in specs
  ]


def _package_parent(root: Path) -> Path:
  if (root / "utils" / "model.py").is_file():
    return root.parent
  nested = root / "LASErMPNN"
  if (nested / "utils" / "model.py").is_file():
    return root
  msg = f"{root} does not contain LASErMPNN utils/model.py"
  raise SystemExit(msg)


def _oracle_python(args: Any) -> str:
  if args.oracle_python:
    return str(args.oracle_python)
  msg = "set --oracle-python or LASER_ORACLE_PYTHON to the torch oracle interpreter"
  raise SystemExit(msg)


def _work_dir(args: Any, name: str) -> Path:
  work = (
    Path(args.work_dir) if args.work_dir is not None else Path(os.environ.get("TMPDIR", "/tmp")) / name
  )
  work.mkdir(parents=True, exist_ok=True)
  return work


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_sample_dist_pilot")
  logger.setLevel(logging.INFO)
  return logger


def _design_mask(chain_mask: Any) -> list[bool]:
  """Designed positions. Upstream ``chain_mask`` 1 means fixed."""
  import numpy as np

  fixed = np.asarray(chain_mask).reshape(-1) != 0
  designed = ~fixed
  if not bool(designed.any()):
    msg = "condition has no designable positions"
    raise RuntimeError(msg)
  return [bool(value) for value in designed.tolist()]


def _json_degrees(values: Any) -> list[float | None]:
  import math

  import numpy as np

  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  encoded: list[float | None] = []
  for value in flat.tolist():
    number = float(value)
    encoded.append(None if math.isnan(number) else number)
  return encoded


def _load_model(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
  import torch
  from LASErMPNN.run_inference import load_model_from_parameter_dict

  model, params = load_model_from_parameter_dict(str(checkpoint), "cpu", strict=False)
  blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
  state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
  model.load_state_dict(state, strict=False)
  model.eval()
  if not isinstance(params, dict) or not isinstance(params.get("model_params"), dict):
    msg = "checkpoint params['model_params'] is not a dict"
    raise TypeError(msg)
  return model, params


def _featurize(model: Any, model_params: dict[str, Any], pdb: str) -> Any:
  from LASErMPNN.run_inference import ProteinComplexData, get_protein_hierview

  view = get_protein_hierview(pdb)
  data = ProteinComplexData(view, pdb, verbose=False)
  batch = data.output_batch_data(fix_beta=False)
  batch.construct_graphs(
    model.rotamer_builder,
    model.ligand_featurizer,
    **model_params["graph_structure"],
    protein_training_noise=0.0,
    ligand_training_noise=0.0,
    subgraph_only_dropout_rate=0.0,
    num_adjacent_residues_to_drop=0,
    build_hydrogens=bool(model_params["build_hydrogens"]),
  )
  return batch


def _restore(batch: Any, native: dict[str, Any]) -> None:
  batch.sequence_indices = native["sequence"].detach().clone()
  batch.chi_angles = native["chi"].detach().clone()
  batch.chain_mask = native["chain_mask"].detach().clone()


def _sample_once(model: Any, batch: Any, spec: _UnitSpec, length: int, rng: Any) -> tuple[list[int], list[float | None]]:
  import numpy as np
  from oracle_shims.laser import injected_uniform_draws

  batch.generate_decoding_order(stack_tensors=True)
  temperature = float(spec["sample_temperature"])
  kwargs = {
    "sequence_sample_temperature": temperature,
    "chi_angle_sample_temperature": temperature,
    "disabled_residues": ["X"],
    "disable_pbar": True,
    "seq_min_p": float(spec["min_p"]),
    "chi_min_p": float(spec["min_p"]),
    "ignore_chain_mask_zeros": False,
    "repack_all": False,
  }
  if spec["shim"]:
    uniforms = rng.random((1, _DRAWS_PER_RESIDUE * length), dtype=np.float64)
    with injected_uniform_draws(uniforms) as cursor:
      sampled = model.sample(batch, **kwargs)
    expected = _DRAWS_PER_RESIDUE * length
    if cursor.consumed_steps != expected:
      msg = f"shim consumed {cursor.consumed_steps} categorical draws, expected {expected}"
      raise RuntimeError(msg)
  else:
    sampled = model.sample(batch, **kwargs)
  sequence = [int(value) for value in sampled.sampled_sequence_indices.detach().cpu().reshape(-1).tolist()]
  chi1 = sampled.sampled_chi_degrees.detach().cpu().reshape(-1, 4)[:, 0]
  if len(sequence) != length:
    msg = f"sampled length {len(sequence)} != structure length {length}"
    raise RuntimeError(msg)
  return sequence, _json_degrees(chi1)


def _sample_unit(model: Any, batch: Any, native: dict[str, Any], spec: _UnitSpec) -> dict[str, Any]:
  """Draw ``n`` sequences. Shimmed runs inject a fresh iid uniform stream."""
  import numpy as np
  import torch

  length = int(np.asarray(native["sequence"].detach().cpu()).reshape(-1).shape[0])
  rng = np.random.default_rng(spec["seed"])
  torch.manual_seed(spec["seed"])
  sequences: list[list[int]] = []
  chi1: list[list[float | None]] = []
  with torch.no_grad():
    for _index in range(spec["n"]):
      _restore(batch, native)
      tokens, degrees = _sample_once(model, batch, spec, length, rng)
      sequences.append(tokens)
      chi1.append(degrees)
  return {
    "sequences": sequences,
    "chi1_degrees": chi1,
    "mask": _design_mask(native["chain_mask"].detach().cpu()),
    "seed": spec["seed"],
    "sample_temperature": spec["sample_temperature"],
    "min_p": spec["min_p"],
    "shim": spec["shim"],
    "run": spec["run"],
    "m": spec["m"],
    "condition": spec["condition"],
    "cell_temperature": spec["cell_temperature"],
    "structure": spec["structure"],
    "n": spec["n"],
  }


def _sample_worker(args: Any) -> None:
  if args.payload_out is None or args.job is None:
    msg = "sample worker requires --job and --payload-out"
    raise SystemExit(msg)
  root = Path(args.laser_root)
  sys.path.insert(0, str(_package_parent(root)))
  checkpoint = _checkpoint(args)
  _require_checkpoint(checkpoint)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[_UnitSpec] = list(job["specs"])
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint)
  work = _work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  model: Any = None
  batches: dict[str, Any] = {}
  natives: dict[str, dict[str, Any]] = {}
  if remaining > 0:
    model, params = _load_model(checkpoint)
    model_params = params["model_params"]
    for spec in specs:
      structure = spec["structure"]
      if structure in batches:
        continue
      batch = _featurize(model, model_params, spec["pdb"])
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
    return _sample_unit(model, batches[spec["structure"]], natives[spec["structure"]], spec)

  payloads, pass_stats = graded_resume.run_units(
    units,
    compute,
    cache,
    resume=bool(args.resume),
  )
  scored = {unit.unit_id: payload for unit, payload in zip(units, payloads, strict=True)}
  text = json.dumps(
    {
      "units": scored,
      "n_reused": pass_stats.n_reused,
      "n_computed": pass_stats.n_computed,
    },
  )
  Path(args.payload_out).write_text(text, encoding="utf-8")


def _launch(args: Any, command: list[str], logger: logging.Logger) -> subprocess.CompletedProcess[str]:
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


def _worker_command(args: Any, job_path: Path, payload_path: Path, work: Path) -> list[str]:
  resume_flag = "--resume" if args.resume else "--no-resume"
  return [
    _oracle_python(args),
    str(Path(__file__).resolve()),
    "--oracle-worker",
    "--laser-root",
    str(args.laser_root),
    "--checkpoint",
    str(_checkpoint(args)),
    "--job",
    str(job_path),
    "--payload-out",
    str(payload_path),
    "--work-dir",
    str(work),
    resume_flag,
  ]


def _run_pass(
  args: Any,
  logger: logging.Logger,
  specs: list[_UnitSpec],
  work: Path,
  tag: str,
) -> tuple[dict[str, Any], int, int]:
  checkpoint = _checkpoint(args)
  units = _graded_units(specs, checkpoint)
  cache = graded_resume.cache_dir(work)
  job_path = work / f"job_{tag}.json"
  payload_path = work / f"upstream_{tag}.json"
  job_path.write_text(json.dumps({"specs": specs}), encoding="utf-8")
  before = graded_resume.count_valid(cache, units)
  completed = _launch(args, _worker_command(args, job_path, payload_path, work), logger)
  if completed.returncode != 0:
    logger.error("sample worker failed\n%s", completed.stderr[-2000:])
    raise SystemExit(completed.returncode)
  after = graded_resume.count_valid(cache, units)
  if bool(args.resume):
    n_reused = min(before, len(units))
    n_computed = max(0, after - before)
  else:
    n_reused = 0
    n_computed = len(units)
  loaded: dict[str, Any] = {}
  for unit in units:
    payload = graded_resume.load_unit(cache, unit.cache_key)
    if payload is None:
      msg = f"unit {unit.unit_id} missing after a zero exit"
      raise SystemExit(msg)
    loaded[unit.unit_id] = payload
  return loaded, n_reused, n_computed


def _degrees(rows: list[list[float | None]]) -> Any:
  import numpy as np

  return np.asarray(
    [[np.nan if value is None else float(value) for value in row] for row in rows],
    dtype=np.float64,
  )


def _chi1_bins(degrees: Any) -> Any:
  """Map χ1 degrees onto ``CHI_BINS`` equal bins of the circle."""
  import numpy as np

  values = np.asarray(degrees, dtype=np.float64)
  bins = np.full(values.shape, stats.CHI_BINS, dtype=np.int64)
  finite = np.isfinite(values)
  if bool(finite.any()):
    width = 360.0 / float(stats.CHI_BINS)
    shifted = np.mod(values[finite] + 180.0, 360.0)
    drawn = np.floor(shifted / width).astype(np.int64)
    bins[finite] = np.clip(drawn, 0, stats.CHI_BINS - 1)
  return bins


def _arrays_for(
  loaded: dict[str, Any],
  specs: list[_UnitSpec],
  condition: _Condition,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[dict[float, Any]], list[Any], list[Any], list[Any], list[dict[float, Any]], dict[str, dict[str, int]]]:
  import numpy as np

  structures = list(dict.fromkeys(spec["structure"] for spec in specs))
  u1: list[Any] = []
  u2: list[Any] = []
  u3: list[Any] = []
  masks: list[Any] = []
  controls: list[dict[float, Any]] = []
  chi_u1: list[Any] = []
  chi_u2: list[Any] = []
  chi_u3: list[Any] = []
  chi_controls: list[dict[float, Any]] = []
  seeds: dict[str, dict[str, int]] = {}
  by_id = {spec["unit_id"]: spec for spec in specs}
  for structure in structures:
    seeds[structure] = {}
    table: dict[float, Any] = {}
    chi_table: dict[float, Any] = {}
    for run, _shim, factor in _runs():
      unit_id = _unit_id(
        condition["name"],
        condition["temperature"],
        structure,
        run,
        specs[0]["n"],
      )
      payload = loaded[unit_id]
      spec = by_id[unit_id]
      seeds[structure][run] = int(spec["seed"])
      array = np.asarray(payload["sequences"], dtype=np.int64)
      chi = _chi1_bins(_degrees(payload["chi1_degrees"]))
      if run == "U1":
        u1.append(array)
        chi_u1.append(chi)
        masks.append(np.asarray(payload["mask"], dtype=np.bool_))
      elif run == "U2":
        u2.append(array)
        chi_u2.append(chi)
      elif run == "U3":
        u3.append(array)
        chi_u3.append(chi)
      elif factor is not None:
        table[factor] = array
        chi_table[factor] = chi
    controls.append(table)
    chi_controls.append(chi_table)
  return u1, u2, u3, masks, controls, chi_u1, chi_u2, chi_u3, chi_controls, seeds


def _chi_table() -> tuple[Any, int]:
  """χ1 column of aminx's per-letter χ mask. X follows that table's own row."""
  import numpy as np

  from aminx.families.laser_mpnn.featurize import LASER_ALPHABET
  from aminx.model.laser.joint_decode import chi_position_mask

  alphabet = len(LASER_ALPHABET)
  letters = np.arange(alphabet, dtype=np.int64)
  mask = np.asarray(chi_position_mask(letters), dtype=np.bool_)
  if mask.shape != (alphabet, 4):
    msg = f"chi mask shape {mask.shape} != ({alphabet}, 4)"
    raise RuntimeError(msg)
  return mask[:, 0], alphabet


def _modal_aa(sequences: Any, alphabet: int) -> Any:
  import numpy as np

  tokens = np.asarray(sequences, dtype=np.int64)
  length = int(tokens.shape[1])
  modal = np.empty(length, dtype=np.int64)
  for position in range(length):
    counts = np.bincount(tokens[:, position], minlength=alphabet)
    modal[position] = int(np.argmax(counts[:alphabet]))
  return modal


def _chi_mean(
  aa_left: Any,
  chi_left: Any,
  aa_right: Any,
  chi_right: Any,
  modal: Any,
  has_chi1: Any,
  mask: Any,
) -> tuple[float, int]:
  import numpy as np

  per = stats.tv_chi1_per_position(aa_left, chi_left, aa_right, chi_right, modal, has_chi1)
  designed = np.asarray(mask, dtype=np.bool_)
  n_positions = int(np.count_nonzero(np.isfinite(per) & designed))
  distance = stats.mean_tv_chi1(aa_left, chi_left, aa_right, chi_right, modal, has_chi1, mask)
  return distance, n_positions


def _chi1_block(
  aa_u1: list[Any],
  aa_u2: list[Any],
  aa_u3: list[Any],
  chi_u1: list[Any],
  chi_u2: list[Any],
  chi_u3: list[Any],
  masks: list[Any],
  controls: list[dict[float, Any]],
  chi_controls: list[dict[float, Any]],
  has_chi1: Any,
  alphabet: int,
) -> dict[str, Any]:
  import numpy as np

  d_u2: list[float] = []
  d_u3: list[float] = []
  n_positions = 0
  control_values: dict[float, list[float]] = {}
  for index, mask in enumerate(masks):
    modal = _modal_aa(aa_u1[index], alphabet)
    distance, count = _chi_mean(
      aa_u2[index],
      chi_u2[index],
      aa_u1[index],
      chi_u1[index],
      modal,
      has_chi1,
      mask,
    )
    d_u2.append(distance)
    n_positions += count
    distance, _count = _chi_mean(
      aa_u3[index],
      chi_u3[index],
      aa_u1[index],
      chi_u1[index],
      modal,
      has_chi1,
      mask,
    )
    d_u3.append(distance)
    for factor in sorted(chi_controls[index]):
      distance, _count = _chi_mean(
        controls[index][factor],
        chi_controls[index][factor],
        aa_u1[index],
        chi_u1[index],
        modal,
        has_chi1,
        mask,
      )
      control_values.setdefault(factor, []).append(distance)
  return {
    "delta_chi": stats.DELTA_CHI,
    "n_positions": n_positions,
    "mean_d_u2_u1": float(np.mean(np.asarray(d_u2, dtype=np.float64))),
    "mean_d_u3_u1": float(np.mean(np.asarray(d_u3, dtype=np.float64))),
    "mean_d_controls": [
      [factor, float(np.mean(np.asarray(values, dtype=np.float64)))]
      for factor, values in sorted(control_values.items())
    ],
  }


def _grade_cell(
  loaded: dict[str, Any],
  specs: list[_UnitSpec],
  condition: _Condition,
  *,
  n_boot: int,
  bootstrap_seed: int,
  repilot: bool,
  has_chi1: Any,
  alphabet: int,
) -> dict[str, Any]:
  u1, u2, u3, masks, controls, chi_u1, chi_u2, chi_u3, chi_controls, seeds = _arrays_for(
    loaded,
    specs,
    condition,
  )
  derived = stats.derive_pilot(
    u1,
    u2,
    u3,
    masks,
    controls,
    alphabet,
    n_boot=n_boot,
    seed=bootstrap_seed,
  )
  checked = stats.shim_check(u1, u2, u3, masks, alphabet, derived.h_hat)
  status = stats.resolve_pilot(derived, repilot=repilot)
  return {
    "delta": derived.delta,
    "h_hat": derived.h_hat,
    "q_hat": derived.q_hat,
    "chosen_m": derived.chosen_m,
    "delta_neg": [[m, value] for m, value in derived.delta_neg],
    "escalate": derived.escalate,
    "status": status,
    "shim_check": {
      "mean_gap": checked.mean_gap,
      "h_hat": checked.h_hat,
      "instrument_invalid": checked.instrument_invalid,
    },
    "chi1": _chi1_block(
      u1,
      u2,
      u3,
      chi_u1,
      chi_u2,
      chi_u3,
      masks,
      controls,
      chi_controls,
      has_chi1,
      alphabet,
    ),
    "min_p": condition["min_p"],
    "n": specs[0]["n"],
    "bootstrap_seed": bootstrap_seed,
    "seeds": seeds,
    "derivation": derived,
  }


def _derived_ok(cell: dict[str, Any]) -> bool:
  delta = float(cell["delta"])
  return cell["status"] == "ok" and cell["chosen_m"] is not None and stats.DELTA_LO <= delta <= stats.DELTA_HI


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
    msg = "laser_sample_dist_pilot requires --out"
    raise SystemExit(msg)
  if int(args.max_attempts) < 1:
    msg = "--max-attempts must be >= 1"
    raise SystemExit(msg)
  logger = _logger()
  if args.smoke and args.work_dir is None:
    args.work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / _WORK_SMOKE
  checkpoint = _checkpoint(args)
  _require_checkpoint(checkpoint)
  work = _work_dir(args, _WORK)
  structures = _structures(smoke=bool(args.smoke))
  conditions = _conditions(smoke=bool(args.smoke))
  selected: list[str] | None = None
  if args.cells:
    selected = [cell.strip() for cell in str(args.cells).split(",") if cell.strip()]
    known = {_cell_key(c["name"], c["temperature"]) for c in conditions}
    unknown = sorted(set(selected) - known)
    if unknown:
      msg = f"--cells names cells this grid cannot run: {unknown}; runnable: {sorted(known)}"
      raise SystemExit(msg)
    conditions = [
      c for c in conditions if _cell_key(c["name"], c["temperature"]) in set(selected)
    ]
  n_boot = 50 if args.smoke else stats.DEFAULT_N_BOOT
  n_pilot = 50 if args.smoke else _N
  has_chi1, alphabet = _chi_table()
  used: set[int] = set()
  first = _unit_specs(structures, conditions, n_pilot, used)
  second = [] if args.smoke else _unit_specs(structures, conditions, _N_ESCALATED, used)
  bootstrap: dict[str, int] = {}
  for condition in conditions:
    for n_value in (n_pilot, _N_ESCALATED):
      label = f"{_cell_key(condition['name'], condition['temperature'])}:n{n_value}"
      bootstrap[label] = _claim(_hash_seed(f"bootstrap:{label}"), used)
  loaded, n_reused, n_computed = _run_pass(args, logger, first, work, f"n{n_pilot}")
  cells: dict[str, Any] = {}
  for condition in conditions:
    key = _cell_key(condition["name"], condition["temperature"])
    subset = [
      spec
      for spec in first
      if spec["condition"] == condition["name"] and spec["cell_temperature"] == condition["temperature"]
    ]
    graded = _grade_cell(
      loaded,
      subset,
      condition,
      n_boot=n_boot,
      bootstrap_seed=bootstrap[f"{key}:n{n_pilot}"],
      repilot=False,
      has_chi1=has_chi1,
      alphabet=alphabet,
    )
    seeds_n1000 = graded["seeds"]
    if graded["status"] == "escalate" and not args.smoke:
      escalated_specs = [
        spec
        for spec in second
        if spec["condition"] == condition["name"] and spec["cell_temperature"] == condition["temperature"]
      ]
      escalated, reused_n, computed_n = _run_pass(
        args,
        logger,
        escalated_specs,
        work,
        f"n{_N_ESCALATED}_{key}",
      )
      n_reused += reused_n
      n_computed += computed_n
      graded = _grade_cell(
        escalated,
        escalated_specs,
        condition,
        n_boot=n_boot,
        bootstrap_seed=bootstrap[f"{key}:n{_N_ESCALATED}"],
        repilot=True,
        has_chi1=has_chi1,
        alphabet=alphabet,
      )
    graded.pop("derivation")
    if graded["n"] != n_pilot:
      graded["seeds_n1000"] = seeds_n1000
    cells[key] = graded
  required = _required_cells(smoke=bool(args.smoke))
  if selected is not None:
    required = [key for key in required if key in set(selected)]
  missing = [key for key in required if key not in cells]
  shim_failures = [key for key, cell in cells.items() if bool(cell["shim_check"]["instrument_invalid"])]
  invalid = [key for key, cell in cells.items() if cell["status"] == "instrument_invalid"]
  payload = {
    "script_sha256": _sha256(Path(__file__).resolve()),
    "n_reused": n_reused,
    "n_computed": n_computed,
    "smoke": bool(args.smoke),
    "cells_selected": selected,
    "n_boot": n_boot,
    "structures": structures,
    "open_questions": [],
    "missing_cells": missing,
    "cells": cells,
    "all_cells_derived": not missing and all(_derived_ok(cells[key]) for key in required),
    "any_instrument_invalid": bool(invalid or shim_failures),
    "shim_ok": not shim_failures,
  }
  _emit(Path(args.out), payload)


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  if args.oracle_worker:
    _sample_worker(args)
    return
  _parent(args)


if __name__ == "__main__":
  main()
