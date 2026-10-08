"""Shared launch, draw bank, and upstream load for the graded Potts vehicles.

Sampling parity is meaningless if each arm draws its own uniforms. The oracle
worker generates one stream, banks it, and the aminx arm replays that stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

from artifact_key import stable_artifact_key

import graded_resume

# Temperatures recorded by dump_potts_oracles. The graded vehicles sample the
# same conditionals those oracles were dumped at.
AR_TEMPERATURE = 0.1
REFINE_TEMPERATURE = 0.5
REFINE_SWEEPS = 1000
VOCAB = 21


class OracleFeat(NamedTuple):
  """One structure, batch dimension kept, ready for upstream ``decoder`` / ``optimize``."""

  x: Any
  s: Any
  mask: Any
  chain_m: Any
  chain_m_pos: Any
  chain_encoding: Any
  residue_idx: Any
  omit_aa_mask: Any
  bias_by_res: Any
  pssm_coef: Any
  pssm_bias: Any
  pssm_log_odds: Any
  tied_beta: Any
  seq: str
  length: int


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def default_potts_root() -> Path:
  return Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser()


def checkpoint_path(args: argparse.Namespace) -> Path:
  if args.checkpoint is not None:
    return Path(args.checkpoint)
  return Path(args.potts_root) / "vanilla_model_weights" / "pottsmpnn_20.pt"


def require_checkpoint(path: Path, expected: str) -> None:
  digest = sha256(path)
  if digest != expected:
    msg = f"{path} sha256 {digest} != {expected}"
    raise SystemExit(msg)


def band(match: float) -> str:
  if match == 1.0:
    return "pass"
  if match >= 0.99:
    return "inconclusive"
  return "fail"


def token_match(got: list[list[int]], ref: list[list[int]]) -> float:
  """Fraction of equal tokens. A shorter row cannot score 1."""
  if len(got) != len(ref):
    msg = f"{len(got)} rows != upstream {len(ref)}"
    raise SystemExit(msg)
  hits = 0
  total = 0
  for row, upstream in zip(got, ref, strict=True):
    total += max(len(row), len(upstream))
    hits += sum(int(a == b) for a, b in zip(row, upstream, strict=False))
  if total == 0:
    return 0.0
  return hits / total


def work_dir(args: argparse.Namespace, name: str) -> Path:
  work = (
    Path(args.work_dir)
    if args.work_dir is not None
    else Path(os.environ.get("TMPDIR", "/tmp")) / name
  )
  work.mkdir(parents=True, exist_ok=True)
  return work


def write_payload(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload), encoding="utf-8")


def results_path(script_name: str) -> Path:
  env = os.environ.get("BTH_RESULTS_PATH")
  if not env:
    msg = f"{script_name} requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  return Path(env)


def build_parser(description: str) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--potts-root", type=Path, default=default_potts_root())
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--oracle-python", default=os.environ.get("POTTS_ORACLE_PYTHON"))
  parser.add_argument("--mutants", default="")
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--upstream", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--max-attempts", type=int, default=3)
  parser.add_argument("--arm-timeout", type=float, default=86400.0)
  parser.add_argument("--oracle-worker", action="store_true")
  return parser


def units(
  job: dict[str, Any],
  arm_id: str,
  checkpoint: Path,
  script: Path,
  unit_ids: Callable[[str, dict[str, Any]], list[str]],
) -> list[graded_resume.Unit]:
  script_sha = sha256(script)
  digest = sha256(checkpoint)
  built: list[graded_resume.Unit] = []
  for name, payload in job["structures"].items():
    for unit_id in unit_ids(str(name), payload):
      built.append(
        graded_resume.Unit(
          unit_id=unit_id,
          arm_id=arm_id,
          checkpoint_sha256=digest,
          input_sha256=str(payload["sha256"]),
          script_sha256=script_sha,
        ),
      )
  return built


def check_pdb(path: Path, expected: str) -> None:
  digest = sha256(path)
  if digest != expected:
    msg = f"{path} sha256 {digest} != {expected}"
    raise SystemExit(msg)


# Every 5th chain-local slot. One fixed site left some draws at match 1.0 on
# the longer example chains, so the AR-mask control could still land in pass.
FIXED_POSITIONS_JSON = (
  Path(__file__).resolve().parent / "fixtures" / "potts_ar_fixed_positions.json"
)
FIXED_POSITIONS_SHA256 = "d4a3cd25332b65ced7a0874288336cacd952a992a6db9673a055b869712f8a0a"


def load_fixed_positions() -> tuple[dict[str, dict[str, list[int]]], str]:
  """Pinned chain-local indexes that make ``chain_M_pos`` non-degenerate.

  The example PDBs themselves have no fixed positions, so ``chain_M_pos`` is
  identically 1 and scaling the AR mask by it is the identity. The sidecar
  marks real slots on those PDBs; it does not add coordinates.
  """
  digest = sha256(FIXED_POSITIONS_JSON)
  if digest != FIXED_POSITIONS_SHA256:
    msg = f"{FIXED_POSITIONS_JSON} sha256 {digest} != {FIXED_POSITIONS_SHA256}"
    raise SystemExit(msg)
  raw = json.loads(FIXED_POSITIONS_JSON.read_text(encoding="utf-8"))
  parsed: dict[str, dict[str, list[int]]] = {}
  for name, chains in raw.items():
    parsed[str(name)] = {
      str(letter): [int(index) for index in indexes] for letter, indexes in chains.items()
    }
  return parsed, digest


def input_sha(pdb_sha256: str, fixed_sha256: str) -> str:
  """Unit identity covers the PDB and the fixed-position sidecar.

  A PDB-only key would reuse units featurized when ``chain_M_pos`` was identically 1.
  """
  blob = json.dumps(
    {"fixed_positions_sha256": fixed_sha256, "pdb_sha256": pdb_sha256},
    sort_keys=True,
    separators=(",", ":"),
  )
  return hashlib.sha256(blob.encode()).hexdigest()


def fixed_position_dict(
  stem: str,
  positions: dict[str, list[int]],
) -> dict[str, dict[str, list[int]]]:
  """``tied_featurize`` keys the dict by the PDB stem."""
  return {stem: positions}


def structure_fixed(payload: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
  """Fixed-position dict for one job entry."""
  stem = Path(str(payload["pdb"])).stem
  positions = payload["fixed_positions"]
  return fixed_position_dict(
    stem,
    {str(letter): [int(index) for index in indexes] for letter, indexes in positions.items()},
  )


def example_structures(root: Path, pdb_sha256: dict[str, str]) -> dict[str, dict[str, Any]]:
  """Example PDBs plus the pinned fixed-position sidecar.

  ``sha256`` is the unit-cache identity (PDB digest and sidecar digest). The
  PDB file is still checked against ``pdb_sha256`` on its own.
  """
  fixed, fixed_sha = load_fixed_positions()
  expected = {Path(name).stem for name in pdb_sha256}
  if set(fixed) != expected:
    msg = f"fixed-position keys {sorted(fixed)} != example PDBs {sorted(expected)}"
    raise SystemExit(msg)
  folder = root / "inputs" / "example_pdbs"
  structures: dict[str, dict[str, Any]] = {}
  for name, digest in pdb_sha256.items():
    path = folder / name
    check_pdb(path, digest)
    stem = path.stem
    structures[stem] = {
      "pdb": str(path),
      "sha256": input_sha(digest, fixed_sha),
      "pdb_sha256": digest,
      "fixed_positions": fixed[stem],
    }
  return structures


def draw_rng(cell: str) -> Any:
  """Deterministic stream for one cell. Re-banking after a resume does not move it."""
  import numpy as np

  digest = hashlib.sha256(f"potts-graded-v1:{cell}".encode()).digest()
  return np.random.default_rng(np.frombuffer(digest[:8], dtype=np.uint64))


def refine_width(mask: Any, chain_m: Any, chain_m_pos: Any) -> int:
  """Draws one refine sweep consumes. Inactive rows never call multinomial."""
  import numpy as np

  present = np.asarray(mask).reshape(-1) != 0
  designed = np.asarray(chain_m).reshape(-1) != 0
  free = np.asarray(chain_m_pos).reshape(-1) != 0
  count = int(np.count_nonzero(present & designed & free))
  return max(count, 1)


def write_draws(work: Path, cells: dict[str, dict[str, Any]]) -> None:
  """Bank ``draws.npz`` plus ``draws.json`` so the aminx arm can select a cell."""
  import numpy as np

  flat: dict[str, Any] = {}
  manifest: dict[str, dict[str, str]] = {}
  for cell, arrays in cells.items():
    manifest[cell] = {}
    for key, value in arrays.items():
      np_key = f"{cell}__{key}".replace(":", "_")
      flat[np_key] = np.asarray(value)
      manifest[cell][key] = np_key
  np.savez(work / "draws.npz", **flat)
  (work / "draws.json").write_text(json.dumps(manifest), encoding="utf-8")


def read_draws(work: Path, cell: str) -> dict[str, Any]:
  import numpy as np

  manifest = json.loads((work / "draws.json").read_text(encoding="utf-8"))
  if cell not in manifest:
    msg = f"draw cell {cell} is not in {work / 'draws.json'}"
    raise KeyError(msg)
  archive = np.load(work / "draws.npz")
  return {key: archive[name] for key, name in manifest[cell].items()}


def oracle_python(args: argparse.Namespace) -> str:
  """The oracle interpreter. Never the parent ``sys.executable``."""
  if args.oracle_python:
    return str(args.oracle_python)
  msg = "set --oracle-python or POTTS_ORACLE_PYTHON to the torch oracle interpreter"
  raise SystemExit(msg)


def load_potts_oracle(root: Path, checkpoint: Path, *, double: bool) -> dict[str, Any]:
  """Load upstream ``PottsMPNN`` from ``--potts-root``. Torch stays in this process."""
  from types import SimpleNamespace

  import torch

  sys.path.insert(0, str(root))
  from potts_mpnn_utils import PottsMPNN

  blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
  state = (
    blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
  )
  model = PottsMPNN(
    ca_only=False,
    num_letters=21,
    vocab=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    potts_dim=400,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    augment_eps=0.0,
  )
  model.load_state_dict(state, strict=False)
  model.eval()
  if double:
    model.double()
  return {
    "model": model,
    "double": double,
    "root": str(root),
    "cfg": SimpleNamespace(dev="cpu", model=SimpleNamespace(vocab=VOCAB)),
  }


def featurize_upstream(
  root: Path,
  pdb: Path,
  fixed_positions: dict[str, dict[str, list[int]]] | None = None,
) -> OracleFeat:
  """``parse_PDB`` + ``tied_featurize`` with no chain dict. All chains are designed.

  ``fixed_positions`` is the upstream dict (structure name → chain → 1-based
  indexes). Omit it and ``chain_M_pos`` stays 1 on every row of these PDBs.
  """
  import torch

  sys.path.insert(0, str(root))
  import potts_mpnn_utils as potts

  parsed = potts.parse_PDB(str(pdb), input_chain_list=None, ca_only=False, skip_gaps=False)
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
    _tied,
    pssm_coef,
    pssm_bias,
    pssm_log_odds,
    bias_by_res,
    _tied_beta,
    _chain_lens,
  ) = potts.tied_featurize(
    [parsed[0]],
    torch.device("cpu"),
    None,
    fixed_positions,
    None,
    None,
    None,
    None,
    ca_only=False,
    vocab=VOCAB,
  )
  import etab_utils

  seq = "".join(etab_utils.ints_to_seq(s[0].detach().cpu().tolist()))
  return OracleFeat(
    x=_np(x),
    s=_np(s),
    mask=_np(mask),
    chain_m=_np(chain_m),
    chain_m_pos=_np(chain_m_pos),
    chain_encoding=_np(chain_encoding),
    residue_idx=_np(residue_idx),
    omit_aa_mask=_np(omit_aa_mask),
    bias_by_res=_np(bias_by_res),
    pssm_coef=_np(pssm_coef),
    pssm_bias=_np(pssm_bias),
    pssm_log_odds=_np(pssm_log_odds),
    tied_beta=_np(_tied_beta),
    seq=seq,
    length=int(s.shape[-1]),
  )


def _np(value: Any) -> Any:
  import numpy as np

  array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
  return array


def prepare_aminx(
  pdb: Path,
  fixed_positions: dict[str, dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
  """Featurize one PDB with the aminx port. The arm replays draws; it does not reseed."""
  import numpy as np

  from aminx.families.potts_mpnn.featurize import pad, parse_pdb_upstream, tied_featurize_port
  from aminx.families.potts_mpnn.sample_host import build_tie_groups_np

  parsed = parse_pdb_upstream(pdb)[0]
  features = tied_featurize_port([parsed], None, fixed_position_dict=fixed_positions)[0]
  padded, pad_valid = pad(features, int(features.L_total))
  length = int(features.L_total)
  log_odds = np.asarray(padded.pssm_log_odds, dtype=np.float32)
  return {
    "coords": np.asarray(padded.x, dtype=np.float32),
    "present": np.asarray(padded.present, dtype=np.float32),
    "residue_idx": np.asarray(padded.residue_idx, dtype=np.int32),
    "chain_index": np.asarray(padded.chain_encoding, dtype=np.int32),
    "pad_valid": np.asarray(pad_valid, dtype=np.bool_),
    "s_true": np.asarray(padded.s, dtype=np.int32),
    "chain_mask": np.asarray(padded.chain_m, dtype=np.float32),
    "chain_m_pos": np.asarray(padded.chain_m_pos, dtype=np.float32),
    "tie_groups": build_tie_groups_np(features.tied_pos, length, length),
    "tied_beta": np.asarray(padded.tied_beta, dtype=np.float32),
    "bias_by_res": np.asarray(padded.bias_by_res, dtype=np.float32),
    "pssm_coef": np.asarray(padded.pssm_coef, dtype=np.float32),
    "pssm_bias": np.asarray(padded.pssm_bias, dtype=np.float32),
    "pssm_log_odds_mask": (log_odds > 0.0).astype(np.float32),
    "omit_aa_mask": np.asarray(padded.omit_aa_mask, dtype=np.float32),
    "omit": np.zeros((VOCAB,), dtype=np.float32),
    "bias": np.zeros((VOCAB,), dtype=np.float32),
    "length": length,
  }


def as_torch(array: Any, *, floating: Any) -> Any:
  import torch

  source = array
  if floating is not None and source.dtype.kind == "f":
    return torch.as_tensor(source, dtype=floating)
  if source.dtype.kind in "iu":
    return torch.as_tensor(source, dtype=torch.long)
  return torch.as_tensor(source)


def parent(
  args: argparse.Namespace,
  logger: logging.Logger,
  *,
  script: Path,
  work_name: str,
  checkpoint_sha: str,
  build_job: Callable[[Path], dict[str, Any]],
  units_for: Callable[[dict[str, Any], str, Path], list[graded_resume.Unit]],
) -> dict[str, Any]:
  """Launch the oracle interpreter, then each aminx arm, and record what they measured."""
  if int(args.max_attempts) < 1:
    msg = "--max-attempts must be >= 1"
    raise SystemExit(msg)
  if float(args.arm_timeout) <= 0:
    msg = "--arm-timeout must be positive"
    raise SystemExit(msg)
  checkpoint = checkpoint_path(args)
  require_checkpoint(checkpoint, checkpoint_sha)
  work = work_dir(args, work_name)
  cache = graded_resume.cache_dir(work)
  job = build_job(Path(args.potts_root))
  job_path = work / "job.json"
  upstream_path = work / "upstream.json"
  job_path.write_text(json.dumps(job), encoding="utf-8")
  resume_flag = "--resume" if args.resume else "--no-resume"
  oracle_units = units_for(job, graded_resume.ORACLE_ARM, checkpoint)
  oracle = _launch(
    args,
    logger,
    [
      oracle_python(args),
      str(script),
      "--oracle-worker",
      "--potts-root",
      str(args.potts_root),
      "--checkpoint",
      str(checkpoint),
      "--job",
      str(job_path),
      "--payload-out",
      str(upstream_path),
      "--work-dir",
      str(work),
      resume_flag,
    ],
    oracle_units,
    work,
  )
  if oracle.returncode != 0:
    logger.error("oracle worker failed after %s attempts", args.max_attempts)
    raise SystemExit(oracle.stderr[-500:])
  mutants = [item for item in str(args.mutants).split(",") if item]
  measured: dict[str, dict[str, Any]] = {}
  n_units = oracle.n_units
  n_reused = oracle.n_reused
  n_computed = oracle.n_computed
  for arm in ("clean", *mutants):
    arm_units = units_for(job, arm, checkpoint)
    payload_out = work / f"payload_{arm}.json"
    payload_out.unlink(missing_ok=True)
    launched = _launch(
      args,
      logger,
      [
        sys.executable,
        str(script),
        "--arm",
        arm,
        "--job",
        str(job_path),
        "--upstream",
        str(upstream_path),
        "--checkpoint",
        str(checkpoint),
        "--potts-root",
        str(args.potts_root),
        "--work-dir",
        str(work),
        "--payload-out",
        str(payload_out),
        resume_flag,
      ],
      arm_units,
      work,
    )
    n_units += launched.n_units
    n_reused += launched.n_reused
    n_computed += launched.n_computed
    if launched.returncode != 0:
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:]}
      continue
    try:
      measured[arm] = json.loads(payload_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      logger.error(
        "arm %s exited 0 but left no parseable payload at %s; stdout=%r stderr=%s",
        arm,
        payload_out,
        launched.stdout[:200],
        launched.stderr[-2000:] or "<empty>",
      )
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:] or "empty stdout"}
  clean = str(measured["clean"]["band"])
  statuses = {
    mutant: "failed"
    if measured[mutant]["band"] == "fail"
    else "passed"
    if measured[mutant]["band"] in {"pass", "inconclusive"}
    else "error"
    for mutant in mutants
  }
  if args.controls_out is not None:
    write_payload(
      Path(args.controls_out),
      {
        "clean": clean,
        "mutants": statuses,
        "weights": {stable_artifact_key(checkpoint): sha256(checkpoint)},
      },
    )
  n_failed = sum(status == "failed" for status in statuses.values())
  match = measured["clean"].get("match", float("nan"))
  return {
    "clean": clean,
    "n_listed": len(mutants),
    "n_failed": n_failed,
    "match": float(match),
    "n_units": n_units,
    "n_reused": n_reused,
    "n_computed": n_computed,
    "cache_dir": str(cache),
  }


def _launch(
  args: argparse.Namespace,
  logger: logging.Logger,
  command: list[str],
  unit_list: list[graded_resume.Unit],
  work: Path,
) -> graded_resume.LaunchResult:
  return graded_resume.relaunch_subprocess(
    command,
    directory=graded_resume.cache_dir(work),
    units=unit_list,
    resume=bool(args.resume),
    max_attempts=int(args.max_attempts),
    timeout_s=float(args.arm_timeout),
    logger=logger,
  )


def logger_for(name: str) -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  return logger


def ints(values: Any) -> list[int]:
  return [int(value) for value in values]
