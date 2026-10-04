"""Potts pilot for the distributional protocol (do not run from the fixer).

Exploratory. It cites nothing. Per (condition, temperature) it draws upstream
runs U1, U2, U3 and ``C_m`` (m in {1.1, 1.25, 1.5}) on the pinned pilot
structures, then calls ``sample_dist_stats`` for ``δ``, ``m`` and the shim
check. Those values are what the later confirmatory sidecar commits. This
script does not write a ``.bth.toml``.

One resumable unit is one (condition, temperature, structure, run). The body
is written before the stamp, keyed by the script sha256 and the inputs, via
``graded_resume``. A crash loses at most the unit in flight. Attempts are
relaunched with no whole-run timeout.

The PSSM fixture and the 2-member tied fixture are not pinned in the spec or
in this repo. Those conditions are listed in ``open_questions`` and are not
sampled. Nothing is substituted for them.
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
import sample_dist_stats as stats

CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
PILOT_PDB_SHA256 = {
  "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
  "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
}
_WORK = "potts_sample_dist_pilot"
_WORK_SMOKE = "potts_sample_dist_pilot_smoke"
_N = 1000
_N_ESCALATED = 4000
_M_GRID = (1.1, 1.25, 1.5)
_PSSM_MULTI = 0.5
_OPTIMIZATION_TEMPERATURE = 0.5
_X_INDEX = 20
_UNIFORM_STEPS_PER_LENGTH = 4

# Spec: a fixture PSSM, and a pinned 2-member tied fixture, each by path and
# SHA-256. Neither pin is in the spec text or in this repo. Leave these None.
# Filling them with upstream ``PSSM_inputs`` or ``example_tied_positions.json``
# would substitute a different fixture.
PSSM_PIN: tuple[str, str] | None = None
TIED_PIN: tuple[str, str] | None = None

_PSSM_QUESTION = (
  "PSSM fixture missing: spec requires a fixture PSSM (pssm_multi=0.5, bias flag "
  "on, mode none) pinned by path and SHA-256 for the pssm condition. The spec "
  "names neither path nor digest, and this repo has no such pin. Upstream "
  "inputs/PSSM_inputs/{4YOW,3HTN}.npz are other structures and are not used."
)
_TIED_QUESTION = (
  "Tied fixture missing: spec requires a pinned 2-member tied fixture (beta=1) "
  "by path and SHA-256. The spec names neither path nor digest, and this repo "
  "has no such pin. Upstream inputs/example_tied_positions.json is not a "
  "2-member group on the pilot structures (2yc3 is empty; 6w25 lists five A "
  "positions plus B1) and is not used."
)


class _Condition(TypedDict):
  name: str
  temperature: float
  kind: str


class _UnitSpec(TypedDict):
  unit_id: str
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


def _parse(argv: list[str]) -> Any:
  parser = common.build_parser(__doc__)
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


def _np(value: Any) -> Any:
  import numpy as np

  if hasattr(value, "detach"):
    return value.detach().cpu().numpy()
  return np.asarray(value)


def _hash_seed(label: str) -> int:
  digest = hashlib.sha256(f"potts-sample-dist-pilot-v1:{label}".encode()).digest()
  return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _claim(seed: int, used: set[int]) -> int:
  while seed in used:
    seed += 1
  used.add(seed)
  return seed


def _questions() -> list[str]:
  found: list[str] = []
  if PSSM_PIN is None:
    found.append(_PSSM_QUESTION)
  if TIED_PIN is None:
    found.append(_TIED_QUESTION)
  return found


def _conditions(*, smoke: bool) -> list[_Condition]:
  """Spec grid, dropping a condition whose pinned fixture is absent."""
  rows: list[_Condition] = [
    {"name": "plain", "temperature": 0.1, "kind": "plain"},
    {"name": "plain", "temperature": 0.3, "kind": "plain"},
    {"name": "plain", "temperature": 1.0, "kind": "plain"},
  ]
  if PSSM_PIN is not None:
    rows.append({"name": "pssm", "temperature": 0.3, "kind": "pssm"})
  if TIED_PIN is not None:
    rows.append({"name": "tied", "temperature": 0.3, "kind": "tied"})
  rows.append({"name": "refine", "temperature": 0.3, "kind": "refine"})
  if smoke:
    return rows[:1]
  return rows


def _required_cells(*, smoke: bool) -> list[str]:
  """Cells the spec lists, including ones that cannot run without a pin."""
  if smoke:
    return [_cell_key("plain", 0.1)]
  cells = [_cell_key("plain", temperature) for temperature in (0.1, 0.3, 1.0)]
  cells.append(_cell_key("pssm", 0.3))
  cells.append(_cell_key("refine", 0.3))
  cells.append(_cell_key("tied", 0.3))
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


def _structures(root: Path, *, smoke: bool) -> dict[str, dict[str, str]]:
  """Pin ``inputs/example_pdbs/{2yc3,3dkm}.pdb`` by path and SHA-256."""
  folder = root / "inputs" / "example_pdbs"
  names = ("2yc3.pdb",) if smoke else tuple(PILOT_PDB_SHA256)
  pinned: dict[str, dict[str, str]] = {}
  for name in names:
    path = folder / name
    digest = PILOT_PDB_SHA256[name]
    common.check_pdb(path, digest)
    pinned[path.stem] = {
      "pdb": str(path),
      "relative": f"inputs/example_pdbs/{name}",
      "sha256": digest,
    }
  return pinned


def _input_sha(pdb_sha256: str, kind: str) -> str:
  extra: str | None = None
  if kind == "pssm" and PSSM_PIN is not None:
    extra = PSSM_PIN[1]
  elif kind == "tied" and TIED_PIN is not None:
    extra = TIED_PIN[1]
  blob = json.dumps(
    {"fixture_sha256": extra, "pdb_sha256": pdb_sha256},
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
            "structure": structure,
            "run": run,
            "m": factor,
            "shim": shim,
            "kind": condition["kind"],
            "n": n,
            "seed": _claim(_hash_seed(unit_id), used),
            "pdb": payload["pdb"],
            "pdb_sha256": payload["sha256"],
          },
        )
  return specs


def _graded_units(
  specs: list[_UnitSpec],
  checkpoint: Path,
  script: Path,
) -> list[graded_resume.Unit]:
  script_sha = common.sha256(script)
  digest = common.sha256(checkpoint)
  return [
    graded_resume.Unit(
      unit_id=spec["unit_id"],
      arm_id="upstream",
      checkpoint_sha256=digest,
      input_sha256=_input_sha(spec["pdb_sha256"], spec["kind"]),
      script_sha256=script_sha,
    )
    for spec in specs
  ]


def _load_pin(root: Path, pin: tuple[str, str], label: str) -> Any:
  path = root / pin[0]
  digest = common.sha256(path)
  if digest != pin[1]:
    msg = f"{label} {path} sha256 {digest} != {pin[1]}"
    raise SystemExit(msg)
  text = path.read_text(encoding="utf-8")
  try:
    loaded = json.loads(text)
  except json.JSONDecodeError:
    loaded = None
  if isinstance(loaded, dict):
    return loaded
  merged: dict[str, Any] = {}
  for line in text.splitlines():
    if line.strip():
      merged.update(json.loads(line))
  if not merged:
    msg = f"{label} {path} is empty"
    raise SystemExit(msg)
  return merged


def _pssm_dict(root: Path) -> Any:
  if PSSM_PIN is None:
    return None
  return _load_pin(root, PSSM_PIN, "PSSM fixture")


def _tied_dict(root: Path) -> Any:
  if TIED_PIN is None:
    return None
  loaded = _load_pin(root, TIED_PIN, "tied fixture")
  for name, groups in loaded.items():
    if not isinstance(groups, list) or len(groups) != 1:
      msg = f"tied fixture group for {name} is not one 2-member group"
      raise SystemExit(msg)
    group = groups[0]
    members = [index for indexes in group.values() for index in indexes]
    if len(members) != 2:
      msg = f"tied fixture group for {name} has {len(members)} members, not 2"
      raise SystemExit(msg)
    for indexes in group.values():
      if indexes and isinstance(indexes[0], list):
        msg = f"tied fixture group for {name} carries betas other than the plain beta=1 form"
        raise SystemExit(msg)
  return loaded


def _featurize(root: Path, pdb: Path, kind: str) -> dict[str, Any]:
  """``parse_PDB`` + ``tied_featurize``. PSSM and ties only for those conditions."""
  import torch

  sys.path.insert(0, str(root))
  import potts_mpnn_utils as potts

  parsed = potts.parse_PDB(str(pdb), input_chain_list=None, ca_only=False, skip_gaps=False)
  stem = str(parsed[0]["name"])
  pssm = None if kind != "pssm" else _pssm_dict(root)
  tied = None if kind != "tied" else _tied_dict(root)
  if pssm is not None and stem not in pssm:
    msg = f"PSSM fixture has no entry for {stem}"
    raise SystemExit(msg)
  if tied is not None and stem not in tied:
    msg = f"tied fixture has no entry for {stem}"
    raise SystemExit(msg)
  (
    x,
    s,
    mask,
    _lengths,
    chain_m,
    chain_encoding,
    _letters,
    _visible,
    _masked,
    _masked_lengths,
    chain_m_pos,
    omit_aa_mask,
    residue_idx,
    _dihedral,
    tied_pos,
    pssm_coef,
    pssm_bias,
    pssm_log_odds,
    bias_by_res,
    tied_beta,
    _chain_lens,
  ) = potts.tied_featurize(
    [parsed[0]],
    torch.device("cpu"),
    None,
    None,
    None,
    None if tied is None else tied,
    None if pssm is None else pssm,
    None,
    ca_only=False,
    vocab=common.VOCAB,
  )
  import etab_utils

  seq = "".join(etab_utils.ints_to_seq(s[0].detach().cpu().tolist()))
  return {
    "x": _np(x),
    "s": _np(s),
    "mask": _np(mask),
    "chain_m": _np(chain_m),
    "chain_m_pos": _np(chain_m_pos),
    "chain_encoding": _np(chain_encoding),
    "residue_idx": _np(residue_idx),
    "omit_aa_mask": _np(omit_aa_mask),
    "bias_by_res": _np(bias_by_res),
    "pssm_coef": _np(pssm_coef),
    "pssm_bias": _np(pssm_bias),
    "pssm_log_odds": _np(pssm_log_odds),
    "tied_beta": _np(tied_beta),
    "tied_pos": tied_pos[0],
    "seq": seq,
    "length": int(s.shape[-1]),
  }


def _design_mask(feat: dict[str, Any]) -> list[bool]:
  """Designable positions: present, on a designed chain, and not fixed."""
  import numpy as np

  chain = np.asarray(feat["chain_m"]).reshape(-1)
  free = np.asarray(feat["chain_m_pos"]).reshape(-1)
  present = np.asarray(feat["mask"]).reshape(-1)
  designed = (chain != 0) & (free != 0) & (present != 0)
  if not bool(designed.any()):
    msg = "condition has no designable positions"
    raise RuntimeError(msg)
  return [bool(value) for value in designed.tolist()]


def _encode(model: Any, feat: dict[str, Any], dtype: Any) -> dict[str, Any]:
  """One encoder pass. Upstream reuses it for every sequence of the structure."""
  h_v, e_idx, h_e, etab = model.run_encoder(
    common.as_torch(feat["x"], floating=dtype),
    common.as_torch(feat["mask"], floating=dtype),
    common.as_torch(feat["residue_idx"], floating=None),
    common.as_torch(feat["chain_encoding"], floating=None),
  )
  return {"h_v": h_v, "e_idx": e_idx, "h_e": h_e, "etab": etab}


def _decode(
  model: Any,
  encoded: dict[str, Any],
  feat: dict[str, Any],
  randn: Any,
  *,
  kind: str,
  temperature: float,
  dtype: Any,
) -> dict[str, Any]:
  import numpy as np

  h_v = encoded["h_v"]
  e_idx = encoded["e_idx"]
  h_e = encoded["h_e"]
  etab = encoded["etab"]
  zeros = np.zeros(common.VOCAB, dtype=np.float64)
  bias_flag = kind == "pssm"
  kwargs = {
    "temperature": temperature,
    "omit_AAs_np": zeros,
    "bias_AAs_np": zeros,
    "chain_M_pos": common.as_torch(feat["chain_m_pos"], floating=dtype),
    "omit_AA_mask": common.as_torch(feat["omit_aa_mask"], floating=dtype),
    "pssm_coef": common.as_torch(feat["pssm_coef"], floating=dtype),
    "pssm_bias": common.as_torch(feat["pssm_bias"], floating=dtype),
    "pssm_multi": _PSSM_MULTI if bias_flag else 0.0,
    "pssm_log_odds_flag": False,
    "pssm_log_odds_mask": None,
    "pssm_bias_flag": bias_flag,
    "bias_by_res": common.as_torch(feat["bias_by_res"], floating=dtype),
  }
  chain_m = common.as_torch(feat["chain_m"], floating=dtype)
  mask = common.as_torch(feat["mask"], floating=dtype)
  s_true = common.as_torch(feat["s"], floating=None)
  residue_idx = common.as_torch(feat["residue_idx"], floating=None)
  chain_encoding = common.as_torch(feat["chain_encoding"], floating=None)
  if kind == "tied":
    output, _probs = model.tied_decoder(
      h_v,
      e_idx,
      h_e,
      randn,
      s_true,
      chain_m,
      chain_encoding,
      residue_idx,
      mask=mask,
      tied_pos=feat["tied_pos"],
      tied_beta=common.as_torch(feat["tied_beta"], floating=dtype),
      **kwargs,
    )
  else:
    output, _probs = model.decoder(
      h_v,
      e_idx,
      h_e,
      randn,
      s_true,
      chain_m,
      chain_encoding,
      residue_idx,
      mask=mask,
      **kwargs,
    )
  return {"output": output, "etab": etab, "h_v": h_v, "e_idx": e_idx, "h_e": h_e}


def _refine(root: Path, model: Any, feat: dict[str, Any], decoded: dict[str, Any], dtype: Any) -> list[int]:
  """One potts sweep at ``optimization_temperature=0.5`` on the AR sequence."""
  import torch

  sys.path.insert(0, str(root))
  import etab_utils
  import run_utils

  constant = torch.zeros(common.VOCAB, dtype=dtype)
  constant[_X_INDEX] = 1.0
  constant_bias = torch.zeros(common.VOCAB, dtype=dtype)
  h_ex = run_utils.cat_neighbors_nodes(torch.zeros_like(decoded["h_v"]), decoded["h_e"], decoded["e_idx"])
  h_exv = run_utils.cat_neighbors_nodes(decoded["h_v"], h_ex, decoded["e_idx"])
  order = [int(value) for value in decoded["output"]["decoding_order"][0].detach().cpu().tolist()]
  seq = "".join(etab_utils.ints_to_seq(decoded["output"]["S"][0].detach().cpu().tolist()))
  mask = common.as_torch(feat["mask"], floating=dtype) * common.as_torch(feat["chain_m_pos"], floating=dtype)
  chain_m = common.as_torch(feat["chain_m"], floating=dtype)
  shared = {
    "constant": constant,
    "constant_bias": constant_bias,
    "bias_by_res": common.as_torch(feat["bias_by_res"], floating=dtype),
    "pssm_bias_flag": False,
    "pssm_coef": common.as_torch(feat["pssm_coef"], floating=dtype),
    "pssm_bias": common.as_torch(feat["pssm_bias"], floating=dtype),
    "pssm_multi": 0.0,
    "pssm_log_odds_flag": False,
    "pssm_log_odds_mask": None,
    "omit_AA_mask": common.as_torch(feat["omit_aa_mask"], floating=dtype),
    "model": model,
    "h_E": decoded["h_e"],
    "h_EXV_encoder": h_exv,
    "h_V": decoded["h_v"],
    "decoding_order": order,
    "partition_etabs": None,
    "partition_index": None,
    "inter_mask": None,
    "binding_optimization": None,
    "vocab": common.VOCAB,
  }
  if feat["tied_pos"]:
    opt = run_utils.tied_optimize_sequence(
      seq,
      decoded["etab"],
      decoded["e_idx"],
      mask,
      chain_m,
      "potts",
      etab_utils.seq_to_ints,
      _OPTIMIZATION_TEMPERATURE,
      tied_pos=feat["tied_pos"],
      tied_beta=common.as_torch(feat["tied_beta"], floating=dtype),
      tied_epistasis=False,
      **shared,
    )
  else:
    opt = run_utils.optimize_sequence(
      seq,
      decoded["etab"],
      decoded["e_idx"],
      mask,
      chain_m,
      "potts",
      etab_utils.seq_to_ints,
      _OPTIMIZATION_TEMPERATURE,
      **shared,
    )
  return [int(value) for value in opt.detach().cpu().tolist()]


def _sample_unit(root: Path, model: Any, feat: dict[str, Any], spec: _UnitSpec) -> dict[str, Any]:
  """Draw ``n`` sequences. Shimmed runs inject a fresh iid uniform stream."""
  import numpy as np
  import torch
  from oracle_shims.potts import injected_uniform_draws

  dtype = torch.float32
  length = int(feat["length"])
  rng = np.random.default_rng(spec["seed"])
  torch.manual_seed(spec["seed"])
  sequences: list[list[int]] = []
  with torch.no_grad():
    encoded = _encode(model, feat, dtype)
    for _index in range(spec["n"]):
      if spec["shim"]:
        randn = torch.as_tensor(
          rng.standard_normal(np.asarray(feat["chain_m"]).shape),
          dtype=dtype,
        )
        uniforms = rng.random((1, _UNIFORM_STEPS_PER_LENGTH * length), dtype=np.float64)
      else:
        randn = torch.randn(np.asarray(feat["chain_m"]).shape)
        uniforms = None

      def _tokens(draw: Any) -> list[int]:
        decoded = _decode(
          model,
          encoded,
          feat,
          draw,
          kind=spec["kind"],
          temperature=spec["sample_temperature"],
          dtype=dtype,
        )
        if spec["kind"] == "refine":
          return _refine(root, model, feat, decoded, dtype)
        return [int(value) for value in decoded["output"]["S"][0].detach().cpu().tolist()]

      if uniforms is None:
        tokens = _tokens(randn)
      else:
        with injected_uniform_draws(uniforms):
          tokens = _tokens(randn)
      sequences.append(tokens)
  return {
    "sequences": sequences,
    "mask": _design_mask(feat),
    "seed": spec["seed"],
    "sample_temperature": spec["sample_temperature"],
    "shim": spec["shim"],
    "run": spec["run"],
    "m": spec["m"],
    "condition": spec["condition"],
    "cell_temperature": spec["cell_temperature"],
    "structure": spec["structure"],
    "kind": spec["kind"],
    "n": spec["n"],
  }


def _sample_worker(args: Any) -> None:
  if args.payload_out is None or args.job is None:
    msg = "sample worker requires --job and --payload-out"
    raise SystemExit(msg)
  checkpoint = common.checkpoint_path(args)
  common.require_checkpoint(checkpoint, CHECKPOINT_SHA256)
  job = json.loads(Path(args.job).read_text(encoding="utf-8"))
  specs: list[_UnitSpec] = list(job["specs"])
  by_id = {spec["unit_id"]: spec for spec in specs}
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  work = common.work_dir(args, _WORK)
  cache = graded_resume.cache_dir(work)
  _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
  prepared: dict[str, Any] | None = None
  feats: dict[tuple[str, str], dict[str, Any]] = {}
  if remaining > 0:
    prepared = common.load_potts_oracle(Path(args.potts_root), checkpoint, double=False)
    for spec in specs:
      key = (spec["kind"], spec["structure"])
      if key not in feats:
        feats[key] = _featurize(Path(args.potts_root), Path(spec["pdb"]), spec["kind"])

  def compute(unit: graded_resume.Unit) -> dict[str, Any]:
    if prepared is None:
      msg = "oracle model was not loaded"
      raise RuntimeError(msg)
    spec = by_id[unit.unit_id]
    return _sample_unit(
      Path(args.potts_root),
      prepared["model"],
      feats[(spec["kind"], spec["structure"])],
      spec,
    )

  payloads, pass_stats = graded_resume.run_units(
    units,
    compute,
    cache,
    resume=bool(args.resume),
  )
  scored = {unit.unit_id: payload for unit, payload in zip(units, payloads, strict=True)}
  common.write_payload(
    Path(args.payload_out),
    {
      "units": scored,
      "n_reused": pass_stats.n_reused,
      "n_computed": pass_stats.n_computed,
    },
  )


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
    common.oracle_python(args),
    str(Path(__file__).resolve()),
    "--oracle-worker",
    "--potts-root",
    str(args.potts_root),
    "--checkpoint",
    str(common.checkpoint_path(args)),
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
  checkpoint = common.checkpoint_path(args)
  units = _graded_units(specs, checkpoint, Path(__file__).resolve())
  cache = graded_resume.cache_dir(work)
  job_path = work / f"job_{tag}.json"
  payload_path = work / f"upstream_{tag}.json"
  job_path.write_text(json.dumps({"specs": specs}), encoding="utf-8")
  before = graded_resume.count_valid(cache, units)
  completed = _launch(
    args,
    _worker_command(args, job_path, payload_path, work),
    logger,
  )
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


def _arrays_for(
  loaded: dict[str, Any],
  specs: list[_UnitSpec],
  condition: _Condition,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[dict[float, Any]], dict[str, dict[str, int]]]:
  import numpy as np

  structures = list(dict.fromkeys(spec["structure"] for spec in specs))
  u1: list[Any] = []
  u2: list[Any] = []
  u3: list[Any] = []
  masks: list[Any] = []
  controls: list[dict[float, Any]] = []
  seeds: dict[str, dict[str, int]] = {}
  by_id = {spec["unit_id"]: spec for spec in specs}
  for structure in structures:
    seeds[structure] = {}
    table: dict[float, Any] = {}
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
      if run == "U1":
        u1.append(array)
        masks.append(np.asarray(payload["mask"], dtype=np.bool_))
      elif run == "U2":
        u2.append(array)
      elif run == "U3":
        u3.append(array)
      elif factor is not None:
        table[factor] = array
    controls.append(table)
  return u1, u2, u3, masks, controls, seeds


def _grade_cell(
  loaded: dict[str, Any],
  specs: list[_UnitSpec],
  condition: _Condition,
  *,
  n_boot: int,
  bootstrap_seed: int,
  repilot: bool,
) -> dict[str, Any]:
  u1, u2, u3, masks, controls, seeds = _arrays_for(loaded, specs, condition)
  derived = stats.derive_pilot(
    u1,
    u2,
    u3,
    masks,
    controls,
    common.VOCAB,
    n_boot=n_boot,
    seed=bootstrap_seed,
  )
  checked = stats.shim_check(u1, u2, u3, masks, common.VOCAB, derived.h_hat)
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
    "n": specs[0]["n"],
    "bootstrap_seed": bootstrap_seed,
    "seeds": seeds,
    "derivation": derived,
  }


def _derived_ok(cell: dict[str, Any]) -> bool:
  delta = float(cell["delta"])
  return (
    cell["status"] == "ok"
    and cell["chosen_m"] is not None
    and stats.DELTA_LO <= delta <= stats.DELTA_HI
  )


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
    msg = "potts_sample_dist_pilot requires --out"
    raise SystemExit(msg)
  if int(args.max_attempts) < 1:
    msg = "--max-attempts must be >= 1"
    raise SystemExit(msg)
  logger = common.logger_for("potts_sample_dist_pilot")
  if args.smoke and args.work_dir is None:
    args.work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / _WORK_SMOKE
  root = Path(args.potts_root)
  checkpoint = common.checkpoint_path(args)
  common.require_checkpoint(checkpoint, CHECKPOINT_SHA256)
  work = common.work_dir(args, _WORK)
  structures = _structures(root, smoke=bool(args.smoke))
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
    )
    seeds_n1000 = graded["seeds"]
    if graded["status"] == "escalate" and not args.smoke:
      escalated_specs = [
        spec
        for spec in second
        if spec["condition"] == condition["name"]
        and spec["cell_temperature"] == condition["temperature"]
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
      )
    graded.pop("derivation")
    if graded["n"] != n_pilot:
      graded["seeds_n1000"] = seeds_n1000
    cells[key] = graded
  required = _required_cells(smoke=bool(args.smoke))
  if selected is not None:
    required = [key for key in required if key in set(selected)]
  missing = [key for key in required if key not in cells]
  shim_failures = [
    key for key, cell in cells.items() if bool(cell["shim_check"]["instrument_invalid"])
  ]
  invalid = [
    key for key, cell in cells.items() if cell["status"] == "instrument_invalid"
  ]
  payload = {
    "script_sha256": common.sha256(Path(__file__).resolve()),
    "n_reused": n_reused,
    "n_computed": n_computed,
    "smoke": bool(args.smoke),
    "cells_selected": selected,
    "n_boot": n_boot,
    "structures": structures,
    "open_questions": _questions(),
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
