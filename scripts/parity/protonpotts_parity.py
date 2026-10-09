"""The ``protonpotts_parity`` ledger vehicle: every graded ProtonPottsMPNN wave, behind the redsox row contract.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §50-§51. Launched by ``scripts/redsox/launch_wave.sh``::

    bth run --project-slug aminx --output-paths <dir>/branch_controls.json -- \\
        uv run --no-sync python3 scripts/parity/protonpotts_parity.py \\
        --mutants <wave>.<control>,... --work-dir <dir> --controls-out <dir>/branch_controls.json

Each of the nine waves (features, encoder, energy, driver, ph_block, ph_greedy, ph_driver, decoder, ddg) is a script that was graded on
its own before this vehicle existed. This vehicle runs them as subprocesses, unchanged, and reports the union of their
negative controls as ``<wave>.<control>``. The row passes only when every wave is clean, the set of controls the waves
reported equals the ``--mutants`` string exactly (step 1c compares that string to the manifest), and every control failed.

Resumable by unit: each wave's result is kept under ``--work-dir`` with a stamp over its script bytes, converter bytes,
argv and the commit under test; a wave is skipped only when its stamp matches, and the result records which waves were
reused. Each wave has its own timeout, so a stuck wave loses that wave and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_key import stable_artifact_key  # noqa: E402

logger = logging.getLogger("protonpotts_parity")

HERE = Path(__file__).resolve().parent
CHECKPOINT = Path("ProtonPottsMPNN/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt")
# name -> (structure path under the PottsMPNN checkout, binder chain used by the pH-design driver wave)
CELLS: dict[str, tuple[str, str]] = {
  "pkad_unlabelled": ("energy_benchmark_datasets/fireprot_pdbs/1BVC.pdb", "A"),
  "multichain": ("energy_benchmark_datasets/covid_pdbs/6m0j.pdb", "E"),
  "only_o_1olr": ("energy_benchmark_datasets/fireprot_pdbs/1OLR.pdb", "A"),
  "gap": ("energy_benchmark_datasets/fireprot_pdbs/1EL1.pdb", "A"),
}
WAVE_TIMEOUT_S = 3600


@dataclass(frozen=True)
class Roots:
  oracles: Path
  state_npz: Path
  potts: Path

  @property
  def p4e(self) -> Path:
    return self.oracles / "features_v6_p4e"

  @property
  def p4f(self) -> Path:
    return self.oracles / "features_v6_p4f"

  @property
  def p4g(self) -> Path:
    return self.oracles / "features_v6_p4g"

  @property
  def p4h(self) -> Path:
    return self.oracles / "features_v6_p4h2"


def _cells(roots: Roots, *, chains: bool) -> list[str]:
  out: list[str] = []
  for name, (rel, chain) in CELLS.items():
    out += ["--cell", f"{name}={roots.potts / rel}" + (f",{chain}" if chains else "")]
  return out


def wave_argv(wave: str, roots: Roots) -> list[str]:
  """The exact arguments each wave was graded with (see its sidecar's provenance note)."""
  state = ["--state-npz", str(roots.state_npz)]
  features = ["--features-dir", str(roots.oracles)]
  if wave == "features":
    return [*features, "--potts-repo", str(roots.potts)]
  if wave == "encoder":
    return [*state, "--encoder-dir", str(roots.p4e), *features]
  if wave == "energy":
    return [*state, "--encoder-dir", str(roots.p4e), "--energy-dir", str(roots.p4f), *features]
  if wave == "driver":
    return [*state, "--energy-dir", str(roots.p4f), *features, *_cells(roots, chains=False)]
  if wave in {"ph_block", "ph_greedy"}:
    return ["--encoder-dir", str(roots.p4e), "--ph-dir", str(roots.p4g), *features]
  if wave == "ph_driver":
    return [*state, "--ph-dir", str(roots.p4g), *_cells(roots, chains=True)]
  if wave == "decoder":
    return [*state, "--decoder-dir", str(roots.p4h), *features]
  if wave == "ddg":
    return [*state, "--energy-dir", str(roots.p4f), *features, *_cells(roots, chains=False)]
  msg = f"unknown wave {wave!r}"
  raise ValueError(msg)


WAVES = ("features", "encoder", "energy", "driver", "ph_block", "ph_greedy", "ph_driver", "decoder", "ddg")


def _script(wave: str) -> Path:
  return HERE / f"protonpotts_{wave}_parity.py"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _head() -> str:
  completed = subprocess.run(  # noqa: S603
    ["git", "-C", str(HERE), "rev-parse", "HEAD"],  # noqa: S607
    check=False,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def stamp_for(wave: str, argv: list[str], head: str) -> str:
  """Hash of everything that decides a wave's result besides the dumps (which the wave pins itself)."""
  digest = hashlib.sha256()
  for part in (
    _script(wave).read_bytes(),
    (HERE / "convert_protonpotts_checkpoint.py").read_bytes(),
    json.dumps(argv).encode(),
    head.encode(),
  ):
    digest.update(hashlib.sha256(part).digest())
  return digest.hexdigest()


def run_wave(wave: str, roots: Roots, work: Path, head: str) -> dict[str, object]:
  """Run one wave (or reuse its stamped result) and return ``{"results", "reused", "returncode", "stamp"}``."""
  argv = wave_argv(wave, roots)
  stamp = stamp_for(wave, argv, head)
  results_path = work / f"{wave}_results.json"
  stamp_path = work / f"{wave}.stamp"
  if results_path.is_file() and stamp_path.is_file() and stamp_path.read_text(encoding="utf-8").strip() == stamp:
    logger.info("wave %s: reused (stamp %s)", wave, stamp[:12])
    return {"results": json.loads(results_path.read_text(encoding="utf-8")), "reused": True, "returncode": 0, "stamp": stamp}
  stamp_path.unlink(missing_ok=True)
  results_path.unlink(missing_ok=True)
  env = {**os.environ, "BTH_RESULTS_PATH": str(results_path), "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")}
  log_path = work / f"{wave}.log"
  started = time.monotonic()
  try:
    with log_path.open("w", encoding="utf-8") as log:
      completed = subprocess.run(  # noqa: S603
        [sys.executable, str(_script(wave)), *argv],
        check=False,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=WAVE_TIMEOUT_S,
      )
    returncode = completed.returncode
  except subprocess.TimeoutExpired:
    logger.error("wave %s: timed out after %d s", wave, WAVE_TIMEOUT_S)
    returncode = -9
  logger.info("wave %s: rc=%s in %.0f s (log %s)", wave, returncode, time.monotonic() - started, log_path)
  if returncode != 0 or not results_path.is_file():
    return {"results": None, "reused": False, "returncode": returncode, "stamp": stamp}
  results = json.loads(results_path.read_text(encoding="utf-8"))
  stamp_path.write_text(stamp + "\n", encoding="utf-8")  # last: a stamp means the result file is complete
  return {"results": results, "reused": False, "returncode": 0, "stamp": stamp}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mutants", required=True, help="comma-separated <wave>.<control> ids; must equal the manifest rows")
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--oracle-root", type=Path, default=None)
  parser.add_argument("--state-npz", type=Path, default=None)
  parser.add_argument("--potts-root", type=Path, default=None)
  parser.add_argument("--checkpoint", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  oracles = (args.oracle_root or Path(os.environ.get("AMINX_PROTONPOTTS_ORACLES", "~/projects/aminx-oracles-protonpotts"))).expanduser()
  roots = Roots(
    oracles=oracles,
    state_npz=(args.state_npz or Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz"))).expanduser(),
    potts=(args.potts_root or Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN"))).expanduser(),
  )
  checkpoint = (args.checkpoint or oracles / CHECKPOINT).expanduser()
  listed = {part for part in args.mutants.split(",") if part}
  work = args.work_dir
  work.mkdir(parents=True, exist_ok=True)

  checkpoint_sha = _sha256(checkpoint)
  state_meta = json.loads(roots.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  state_matches = state_meta["checkpoint_sha256"] == checkpoint_sha
  head = _head()
  logger.info("checkpoint %s sha %s; converted state matches: %s; head %s", checkpoint, checkpoint_sha[:12], state_matches, head[:8])

  per_wave: dict[str, dict[str, object]] = {}
  statuses: dict[str, str] = {}
  for wave in WAVES:
    outcome = run_wave(wave, roots, work, head)
    results = outcome["results"]
    wave_clean = isinstance(results, dict) and results.get("clean") == "pass"
    controls = results.get("mutants", {}) if isinstance(results, dict) else {}
    for name, status in controls.items():
      statuses[f"{wave}.{name}"] = str(status)
    per_wave[wave] = {
      "clean": "pass" if wave_clean else "fail",
      "returncode": outcome["returncode"],
      "reused": outcome["reused"],
      "stamp": outcome["stamp"],
      "n_controls": len(controls),
      "n_controls_failed": sum(str(v) == "failed" for v in controls.values()),
    }

  n_waves_ok = sum(info["clean"] == "pass" for info in per_wave.values())
  listed_matches = set(statuses) == listed
  clean = "pass" if n_waves_ok == len(WAVES) and state_matches and listed_matches else "fail"
  if not listed_matches:
    logger.error("controls reported %s but --mutants listed %s", sorted(set(statuses) ^ listed), len(listed))
  weights = {stable_artifact_key(checkpoint): checkpoint_sha}
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(
      json.dumps({"clean": clean, "mutants": statuses, "weights": weights}, indent=2) + "\n", encoding="utf-8"
    )
  results = {
    "clean": clean,
    "n_waves": len(WAVES),
    "n_waves_ok": n_waves_ok,
    "state_matches_checkpoint": state_matches,
    "listed_matches": listed_matches,
    "n_listed": len(listed),
    "n_failed": sum(v == "failed" for v in statuses.values()),
    "undetected": sorted(k for k, v in statuses.items() if v != "failed"),
    "n_reused_waves": sum(bool(info["reused"]) for info in per_wave.values()),
    "per_wave": per_wave,
    "weights": weights,
    "head": head,
  }
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
  logger.info("clean=%s waves_ok=%d/%d controls_failed=%d/%d", clean, n_waves_ok, len(WAVES), results["n_failed"], len(listed))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
