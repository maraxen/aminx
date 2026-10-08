"""Graded LASEr score parity against upstream LASErMPNN (do not run from the fixer).

Clean arm: ``4jnj-1_prot.pdb`` plus the 20 complexes pinned by
``tests/fixtures/laser/fixtures_manifest.toml``. Pass when the maximum
absolute per-position log-prob gap is at most ``1e-4``. The open interval
above that through ``1e-3`` is inconclusive. Above ``1e-3`` is fail.

Negative control ``permute_decoder_layer`` reverses one decoder-layer
attention map and must land in the fail band. The clean arm and each mutant
arm are separate subprocesses. Results go to ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import os
import pathlib

# x64 before any JAX import. Parent and children share this so the decoding
# order drawn from random_seed is the same permutation on both sides. Only when run as a
# script, never at import: see laser_proofread_parity.py (aminx #165 CI).
if __name__ == "__main__":
  os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

# The parity dir on sys.path mirrors the convention in the potts scripts, so
# the shared stable_artifact_key helper imports at module scope rather than
# depending on the sys.path insert that main() does later.
_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

from artifact_key import stable_artifact_key

MUTANT_ID = "permute_decoder_layer"
SEED = 42
PASS_MAX = 1e-4
INCONCLUSIVE_MAX = 1e-3
CHECKPOINT_NAME = "laser_weights_0p1A_nothing_heldout.pt"
CHECKPOINT_SHA256 = "304fe02a4807c310bdd9d68c988ae87619da3cf2025d5c223fb31030aa411173"
EXAMPLE_SHA256 = "0a933729c4915cfee14ff34408a57e1d65fd98845df189ba79ad4f4d8470ae7e"
EXAMPLE_NAME = "4jnj-1_prot.pdb"


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
  parser.add_argument("--mutants", default=MUTANT_ID)
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--upstream", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--arm-timeout", type=float, default=86400.0)
  parser.add_argument("--oracle-worker", action="store_true")
  return parser.parse_args(argv)


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
  paths.extend(folder / f"{stem}.pdb" for stem in manifest["ordered_ids"])
  return paths


def _check_inputs(root: Path, checkpoint: Path) -> None:
  example = root / "example_pdbs" / EXAMPLE_NAME
  got = _sha256(example)
  if got != EXAMPLE_SHA256:
    msg = f"{example} sha256 {got} != {EXAMPLE_SHA256}"
    raise SystemExit(msg)
  manifest = _manifest()
  files = manifest["files"]
  folder = _repo() / "tests" / "fixtures" / "laser"
  for name, digest in files.items():
    path = folder / name
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
    msg = "laser_score_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  return Path(env)


def _band(max_abs: float) -> str:
  # Bands are pre-registered: pass <= 1e-4, inconclusive (1e-4, 1e-3], fail > 1e-3.
  if max_abs <= PASS_MAX:
    return "pass"
  if max_abs <= INCONCLUSIVE_MAX:
    return "inconclusive"
  return "fail"


def _install_mutant(arm: str | None) -> None:
  if arm in (None, "clean"):
    return
  if arm != MUTANT_ID:
    msg = f"unknown mutant {arm}"
    raise SystemExit(msg)
  import equinox as eqx
  import jax.numpy as jnp

  from aminx.families.laser_mpnn.driver import LASErMPNN, LaserDriver

  original = LaserDriver.load

  def _load(self: LaserDriver, spec: Any) -> LASErMPNN:
    model = original(self, spec)
    # Reversing gatW's output axis is a permutation of one decoder layer, and
    # it is not a symmetry of the trained map, so log-probs leave the 1e-3 band.
    layer = model.decoder.protein_decoder_layers[0]
    gat = layer.hetgat.subgats[0]
    weight = gat.gatW.weight
    index = jnp.arange(weight.shape[0] - 1, -1, -1)
    linear = eqx.tree_at(lambda item: item.weight, gat.gatW, weight[index])
    gat = eqx.tree_at(lambda item: item.gatW, gat, linear)
    subgats = (gat, *layer.hetgat.subgats[1:])
    hetgat = eqx.tree_at(lambda item: item.subgats, layer.hetgat, subgats)
    layer = eqx.tree_at(lambda item: item.hetgat, layer, hetgat)
    layers = (layer, *model.decoder.protein_decoder_layers[1:])
    decoder = eqx.tree_at(lambda item: item.protein_decoder_layers, model.decoder, layers)
    return eqx.tree_at(lambda item: item.decoder, model, decoder)

  # The mutant replaces load so the traced decoder sees the permuted layer.
  setattr(LaserDriver, "load", _load)


def _work_dir(args: argparse.Namespace) -> Path:
  work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "laser_score_parity"
  work.mkdir(parents=True, exist_ok=True)
  return work


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_score_parity")
  logger.setLevel(logging.INFO)
  return logger


def _native_sequence(indices: Any) -> str:  # noqa: ANN401
  import numpy as np

  from aminx.families.laser_mpnn.driver import CANONICAL_OF_LASER
  from aminx.host.family_driver import ALPHABET

  table = np.asarray(CANONICAL_OF_LASER)
  return "".join(ALPHABET[int(table[int(index)])] for index in np.asarray(indices))


def _build_job(root: Path) -> dict[str, Any]:
  import numpy as np

  from aminx.families.laser_mpnn.driver import decoding_order
  from aminx.families.laser_mpnn.featurize import featurize

  structures: dict[str, Any] = {}
  for path in _fixture_paths(root):
    features = featurize(path, dtype=np.float64)
    order = decoding_order(
      np.asarray(features.chain_mask),
      np.asarray(features.extra_atom_contact_mask),
      seed=SEED,
    )
    structures[path.stem] = {
      "pdb": str(path),
      "sequence": _native_sequence(features.sequence_indices),
      "order": [int(index) for index in order],
    }
  return {"seed": SEED, "structures": structures}


def _score_one(payload: dict[str, Any], checkpoint: Path) -> list[float]:
  from aminx.host.runner import score
  from aminx.run.specs import ScoringSpecification

  spec = ScoringSpecification(
    inputs=payload["pdb"],
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_nothing_heldout",
    model_local_path=checkpoint,
    output_kind="nll",
    sequences_to_score=[payload["sequence"]],
    random_seed=SEED,
  )
  # nll is -log p after the alphabet boundary, so the gap is on log-prob.
  nll = score(spec)["structures"]["0"]["arrays"]["nll"]
  return [-float(value) for value in nll[0]]


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import numpy as np

  _install_mutant(args.arm)
  job = json.loads(args.job.read_text(encoding="utf-8"))
  upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
  checkpoint = _checkpoint(args)
  gaps: list[float] = []
  for name, payload in job["structures"].items():
    log_prob = _score_one(payload, checkpoint)
    ref = upstream[name]
    if len(log_prob) != len(ref):
      msg = f"{name} length {len(log_prob)} != upstream {len(ref)}"
      raise SystemExit(msg)
    gaps.extend(abs(float(a) - float(b)) for a, b in zip(log_prob, ref, strict=True))
  max_abs = float(np.max(np.asarray(gaps, dtype=np.float64))) if gaps else float("inf")
  return {"band": _band(max_abs), "max_abs_delta": max_abs}


def _oracle_python(args: argparse.Namespace) -> str:
  if args.oracle_python:
    return str(args.oracle_python)
  try:
    import torch  # noqa: F401
  except ImportError:
    msg = "set LASER_ORACLE_PYTHON to the torch oracle interpreter"
    raise SystemExit(msg) from None
  return sys.executable


def _launch(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
  # A fresh interpreter is load-bearing. A jit/filter_jit cache warmed by an
  # earlier arm would replay the unmutated trace and the harness would report
  # a kill it never demonstrated.
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
  work = _work_dir(args)
  job = _build_job(args.laser_root)
  job_path = work / "job.json"
  upstream_path = work / "upstream.json"
  job_path.write_text(json.dumps(job), encoding="utf-8")
  script = str(Path(__file__).resolve())
  oracle = _launch(
    [
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
    ],
    float(args.arm_timeout),
  )
  if oracle.returncode != 0:
    logger.error("oracle worker failed: %s", oracle.stderr[-500:])
    raise SystemExit(oracle.stderr[-500:])
  mutants = [item for item in args.mutants.split(",") if item]
  measured: dict[str, dict[str, Any]] = {}
  for arm in ("clean", *mutants):
    payload_out = work / f"payload_{arm}.json"
    payload_out.unlink(missing_ok=True)
    try:
      launched = _launch(
        [
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
          str(args.laser_root),
          "--work-dir",
          str(work),
          "--payload-out",
          str(payload_out),
        ],
        float(args.arm_timeout),
      )
    except subprocess.TimeoutExpired as exc:
      logger.error("arm %s timed out", arm)
      measured[arm] = {"band": "error", "detail": str(exc)}
      continue
    if launched.returncode != 0:
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:]}
      continue
    try:
      measured[arm] = json.loads(payload_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:] or "empty payload"}
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
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(
      json.dumps(
        {
          "clean": clean,
          "mutants": statuses,
          "weights": {stable_artifact_key(checkpoint): _sha256(checkpoint)},
        },
        indent=2,
      )
      + "\n",
      encoding="utf-8",
    )
  n_failed = sum(status == "failed" for status in statuses.values())
  return {
    "clean": clean,
    "n_listed": len(mutants),
    "n_failed": n_failed,
    "max_abs_delta": float(measured["clean"].get("max_abs_delta", float("nan"))),
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
  import torch

  sys.path.insert(0, str(_package_parent(args.laser_root)))
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  from LASErMPNN.run_inference import (  # noqa: PLC0415
    ProteinComplexData,
    get_protein_hierview,
    load_model_from_parameter_dict,
  )
  from LASErMPNN.utils.model import RBF_Encoding  # noqa: PLC0415
  from oracle_shims.laser import f64_runtime  # noqa: PLC0415

  checkpoint = _checkpoint(args)
  torch.manual_seed(0)
  model, params = load_model_from_parameter_dict(str(checkpoint), "cpu", strict=False)
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
  model.load_state_dict(state, strict=False)
  model.double()
  model.eval()
  job = json.loads(args.job.read_text(encoding="utf-8"))
  scores: dict[str, list[float]] = {}
  model_params = params["model_params"]
  # Upstream's own float32 literals otherwise collide with the f64 batch inside
  # compute_first_shell_node_idces -> torch.cdist. dump_laser_oracles wraps the
  # identical work in this shim, so using it keeps this vehicle comparable to
  # the laser_score oracle. It cannot nest, hence one context for the whole loop.
  with f64_runtime(RBF_Encoding), torch.no_grad():
    for name, payload in job["structures"].items():
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
      order = torch.tensor(payload["order"], dtype=torch.long)
      sequence = batch.sequence_indices
      chi = batch.chi_angles
      logits, *_rest = model.get_logits_for_score(batch, order, sequence, chi)
      seq_log = torch.log_softmax(logits, dim=-1)
      gathered = seq_log.gather(-1, sequence.long().unsqueeze(-1)).squeeze(-1)
      scores[name] = [float(value) for value in gathered.detach().cpu()]
  args.upstream.write_text(json.dumps(scores), encoding="utf-8")


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
  # Missing results path is fatal before any arm runs. A pass-looking payload
  # written only to --controls-out still records outcome=unknown.
  path = _results_path()
  logger = _logger()
  results = _parent(args, logger)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
