"""Graded Potts energy parity against upstream PottsMPNN (do not run from the fixer).

Clean arm: example PDBs plus a 30-residue slice of chain A from ``2yc3.pdb``,
200 random sequences per structure. Pass when every ``|ΔE|`` is within
``1e-4 + 1e-5·|E_up|`` and a hand-built 3-residue table matches to ``1e-12``.
Ten times that bound is inconclusive; beyond that is fail.

Negative control ``permute_etab_out_rows`` must land in the fail band.
Each arm, including the oracle worker, is its own subprocess with its own
timeout and is re-launched from the per-PDB cache. Results go to
``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
    sys.path.insert(0, _PARITY_DIR)

from artifact_key import stable_artifact_key

import graded_resume

MUTANT_ID = "permute_etab_out_rows"
N_RANDOM = 200
SEED = 0
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
EXAMPLE_SHA256 = {
    "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
    "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
    "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
    "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
    "6w25.pdb": "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086",
    "swe1_ligand.pdb": "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6",
}
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"


def _default_potts_root() -> Path:
    """Upstream checkout, env-overridable.

    A hardcoded laptop path makes this script unrunnable on the box that holds the
    weights, which is the only box it runs on.
    """
    return Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potts-root", type=Path, default=_default_potts_root())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--oracle-python", default=os.environ.get("POTTS_ORACLE_PYTHON"))
    parser.add_argument("--mutants", default=MUTANT_ID)
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
    return parser.parse_args(argv)


def _checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    return args.potts_root / "vanilla_model_weights" / "pottsmpnn_20.pt"


def _check_inputs(root: Path, checkpoint: Path) -> None:
    folder = root / "inputs" / "example_pdbs"
    for name, digest in EXAMPLE_SHA256.items():
        path = folder / name
        got = _sha256(path)
        if got != digest:
            msg = f"{path} sha256 {got} != {digest}"
            raise SystemExit(msg)
    got = _sha256(checkpoint)
    if got != CHECKPOINT_SHA256:
        msg = f"{checkpoint} sha256 {got} != {CHECKPOINT_SHA256}"
        raise SystemExit(msg)


def _write_l30(source: Path, dest: Path) -> None:
    """First 30 residues of chain A, renumbered from 1. Pinned via ``2yc3.pdb``."""
    residues: list[tuple[int, str, list[str]]] = []
    seen: dict[tuple[int, str], int] = {}
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[21:22] != "A":
            continue
        token = line[22:27]
        icode = token[-1] if token[-1].isalpha() else " "
        resn = int(token[:-1]) if icode != " " else int(token)
        key = (resn, icode)
        if key not in seen:
            if len(seen) == 30:
                continue
            seen[key] = len(residues)
            residues.append((resn, icode, []))
        if len(seen) <= 30 and key in seen:
            residues[seen[key]][2].append(line)
    if len(residues) != 30:
        msg = f"expected 30 residues of chain A in {source}, got {len(residues)}"
        raise SystemExit(msg)
    lines: list[str] = []
    for new_index, (_resn, _icode, atoms) in enumerate(residues, start=1):
        for atom in atoms:
            lines.append(f"{atom[:22]}{new_index:4d} {atom[27:]}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _band(delta: Any, upstream: Any, *, analytic: float) -> str:
    import numpy as np

    bound = 1e-4 + 1e-5 * np.abs(upstream)
    gap = np.abs(delta)
    within = bool(np.all(gap <= bound))
    within_10 = bool(np.all(gap <= 10.0 * bound))
    if analytic > 1e-12:
        return "fail"
    if within:
        return "pass"
    if within_10:
        return "inconclusive"
    return "fail"


def _analytic_abs_err() -> float:
    import jax
    import jax.numpy as jnp

    from aminx.families.potts_mpnn.etab import potts_energy

    # jax.experimental.enable_x64 is absent on this JAX; tests/port/a1_compare.py:32 uses the
    # same fallback, which is why the A1 f64 tiers pass. Not imported from tests/ because
    # tests/ on the path shadows installed packages in this repo.
    enable_x64 = getattr(jax.experimental, "enable_x64", None) or jax.enable_x64
    with enable_x64():
        length = 3
        alphabet = 22
        table = jnp.zeros((length, length, alphabet, alphabet), dtype=jnp.float64)
        neighbors = jnp.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=jnp.int32)
        sequence = jnp.asarray([1, 2, 0], dtype=jnp.int32)
        # etab is [L K A A]: axis 1 is the NEIGHBOUR SLOT, and neighbors[i, k] is the
        # residue in that slot - not an absolute residue index. Each entry below is
        # (i, slot 1, a_i, a_partner) for the partner neighbors[i, 1] actually holds:
        #   i=0 slot1 -> residue 1, a=2   ->  0.5
        #   i=1 slot1 -> residue 2, a=0   -> -0.25
        #   i=2 slot1 -> residue 0, a=1   ->  1.25
        # summing to 1.5. Indexing axis 1 as a residue leaves two terms unselected and
        # yields 0.5, which is what this control reported before (analytic_abs_err 1.0).
        table = table.at[0, 1, 1, 2].set(0.5)
        table = table.at[1, 1, 2, 0].set(-0.25)
        table = table.at[2, 1, 0, 1].set(1.25)
        valid = jnp.ones((length,), dtype=jnp.bool_)
        got = potts_energy(table, neighbors, valid, sequence)
        return float(jnp.abs(got - jnp.asarray(1.5, dtype=jnp.float64)))


def _results_path() -> Path:
    env = os.environ.get("BTH_RESULTS_PATH")
    if not env:
        msg = "potts_energy_parity requires $BTH_RESULTS_PATH"
        raise SystemExit(msg)
    return Path(env)


def _install_mutant(arm: str | None) -> None:
    if arm in (None, "clean"):
        return
    if arm != MUTANT_ID:
        msg = f"unknown mutant {arm}"
        raise SystemExit(msg)
    import equinox as eqx
    import jax.numpy as jnp

    from aminx.families.potts_mpnn.driver import PottsMPNNDriver

    original = PottsMPNNDriver.load

    def _load(self: PottsMPNNDriver, spec: object) -> object:
        model = original(self, spec)
        weight = model.potts_head.linear.weight
        index = jnp.arange(weight.shape[0] - 1, -1, -1)
        linear = eqx.tree_at(lambda layer: layer.weight, model.potts_head.linear, weight[index])
        return eqx.tree_at(lambda module: module.potts_head.linear, model, linear)

    PottsMPNNDriver.load = _load  # type: ignore[method-assign]


def _work_dir(args: argparse.Namespace) -> Path:
    work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "potts_energy_parity"
    work.mkdir(parents=True, exist_ok=True)
    return work


def _structure_input_sha(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    with Path(payload["pdb"]).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(b"\0")
    digest.update(json.dumps(payload["random"], separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _units_from_job(
    job: dict[str, Any],
    arm_id: str,
    checkpoint: Path,
) -> list[graded_resume.Unit]:
    checkpoint_sha = _sha256(checkpoint)
    script_sha = _sha256(Path(__file__).resolve())
    units: list[graded_resume.Unit] = []
    structures = job["structures"]
    for name, payload in structures.items():
        units.append(
            graded_resume.Unit(
                unit_id=str(name),
                arm_id=arm_id,
                checkpoint_sha256=checkpoint_sha,
                input_sha256=_structure_input_sha(payload),
                script_sha256=script_sha,
            ),
        )
    return units


def _score_one(payload: dict[str, Any], checkpoint: Path) -> list[float]:
    from aminx.host.runner import score
    from aminx.run.specs import ScoringSpecification

    spec = ScoringSpecification(
        inputs=payload["pdb"],
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        model_local_path=checkpoint,
        output_kind="energy",
        sequences_to_score=payload["random"],
    )
    energy = score(spec)["structures"]["0"]["arrays"]["energy"]
    return [float(value) for value in energy]


def _resume_flag(args: argparse.Namespace) -> str:
    if args.resume:
        return "--resume"
    return "--no-resume"


def _logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("potts_energy_parity")
    logger.setLevel(logging.INFO)
    return logger


def _validate_launch(args: argparse.Namespace) -> None:
    if int(args.max_attempts) < 1:
        msg = "--max-attempts must be >= 1"
        raise SystemExit(msg)
    if float(args.arm_timeout) <= 0:
        msg = "--arm-timeout must be positive"
        raise SystemExit(msg)


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    _install_mutant(args.arm)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
    checkpoint = _checkpoint(args)
    arm_id = str(args.arm or "clean")
    units = _units_from_job(job, arm_id, checkpoint)
    cache = graded_resume.cache_dir(_work_dir(args))
    structures = job["structures"]

    def compute(unit: graded_resume.Unit) -> list[float]:
        return _score_one(structures[unit.unit_id], checkpoint)

    payloads, _stats = graded_resume.run_units(
        units,
        compute,
        cache,
        resume=bool(args.resume),
    )
    deltas: list[float] = []
    references: list[float] = []
    for unit, payload in zip(units, payloads, strict=True):
        energy = [float(value) for value in payload]
        ref = upstream[unit.unit_id]
        if len(energy) != len(ref):
            msg = f"{unit.unit_id} length {len(energy)} != upstream {len(ref)}"
            raise SystemExit(msg)
        deltas.extend(float(a) - float(b) for a, b in zip(energy, ref, strict=True))
        references.extend(float(value) for value in ref)
    analytic = 0.0 if args.arm not in (None, "clean") else _analytic_abs_err()
    delta = np.asarray(deltas, dtype=np.float64)
    ref = np.asarray(references, dtype=np.float64)
    return {
        "band": _band(delta, ref, analytic=analytic),
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else float("inf"),
        "analytic_abs_err": analytic,
    }


def _build_job(root: Path, work: Path) -> dict[str, Any]:
    import numpy as np

    from aminx.families.potts_mpnn.featurize import parse_pdb_upstream, tied_featurize_port

    work.mkdir(parents=True, exist_ok=True)
    l30 = work / "l30.pdb"
    _write_l30(root / "inputs" / "example_pdbs" / "2yc3.pdb", l30)
    paths = [*(root / "inputs" / "example_pdbs").glob("*.pdb"), l30]
    rng = np.random.default_rng(SEED)
    structures: dict[str, Any] = {}
    for path in paths:
        parsed = parse_pdb_upstream(path)[0]
        features = tied_featurize_port([parsed], None)[0]
        length = int(features.L_total)
        draws = rng.integers(0, len(CANONICAL), size=(N_RANDOM, length))
        random = ["".join(CANONICAL[int(index)] for index in row) for row in draws]
        structures[path.stem] = {"pdb": str(path), "random": random}
    return {"structures": structures}


def _oracle_python(args: argparse.Namespace) -> str:
    if args.oracle_python:
        return str(args.oracle_python)
    try:
        import torch  # noqa: F401
    except ImportError:
        msg = "set POTTS_ORACLE_PYTHON to the torch oracle interpreter"
        raise SystemExit(msg) from None
    return sys.executable


def _launch(
    args: argparse.Namespace,
    logger: logging.Logger,
    command: list[str],
    units: list[graded_resume.Unit],
    work: Path,
) -> graded_resume.LaunchResult:
    return graded_resume.relaunch_subprocess(
        command,
        directory=graded_resume.cache_dir(work),
        units=units,
        resume=bool(args.resume),
        max_attempts=int(args.max_attempts),
        timeout_s=float(args.arm_timeout),
        logger=logger,
    )


def _parent(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    checkpoint = _checkpoint(args)
    _check_inputs(args.potts_root, checkpoint)
    _validate_launch(args)
    work = _work_dir(args)
    cache = graded_resume.cache_dir(work)
    job = _build_job(args.potts_root, work)
    job_path = work / "job.json"
    upstream_path = work / "upstream.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    script = str(Path(__file__).resolve())
    resume_flag = _resume_flag(args)
    oracle_units = _units_from_job(job, graded_resume.ORACLE_ARM, checkpoint)
    oracle = _launch(
        args,
        logger,
        [
            _oracle_python(args),
            script,
            "--oracle-worker",
            "--potts-root",
            str(args.potts_root),
            "--checkpoint",
            str(checkpoint),
            "--job",
            str(job_path),
            "--upstream",
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
    mutants = [item for item in args.mutants.split(",") if item]
    measured: dict[str, dict[str, Any]] = {}
    n_units = oracle.n_units
    n_reused = oracle.n_reused
    n_computed = oracle.n_computed
    for arm in ("clean", *mutants):
        units = _units_from_job(job, arm, checkpoint)
        payload_out = work / f"payload_{arm}.json"
        # A payload left by an earlier attempt must never be read as this one's result.
        payload_out.unlink(missing_ok=True)
        launched = _launch(
            args,
            logger,
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
                "--potts-root",
                str(args.potts_root),
                "--work-dir",
                str(work),
                "--payload-out",
                str(payload_out),
                resume_flag,
            ],
            units,
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
            # An arm that exits 0 with unparseable stdout must not surface as a bare
            # JSONDecodeError in the parent with the child's stderr discarded.
            logger.error(
                "arm %s exited 0 but left no parseable payload at %s; stdout=%r stderr=%s",
                arm,
                payload_out,
                launched.stdout[:200],
                launched.stderr[-2000:] or "<empty>",
            )
            measured[arm] = {"band": "error", "detail": launched.stderr[-500:] or "empty stdout"}
    clean = measured["clean"]["band"]
    statuses = {
        mutant: "failed" if measured[mutant]["band"] == "fail" else "passed"
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
        "analytic_abs_err": float(measured["clean"].get("analytic_abs_err", float("nan"))),
        "n_units": n_units,
        "n_reused": n_reused,
        "n_computed": n_computed,
        "cache_dir": str(cache),
    }


def _oracle_worker(args: argparse.Namespace) -> None:
    checkpoint = _checkpoint(args)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    units = _units_from_job(job, graded_resume.ORACLE_ARM, checkpoint)
    work = _work_dir(args)
    cache = graded_resume.cache_dir(work)
    _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
    prepared: dict[str, Any] | None = None
    if remaining > 0:
        prepared = _load_oracle(args.potts_root, checkpoint)

    def compute(unit: graded_resume.Unit) -> list[float]:
        if prepared is None:
            msg = "oracle model was not loaded"
            raise RuntimeError(msg)
        return _oracle_one(prepared, job["structures"][unit.unit_id])

    payloads, _stats = graded_resume.run_units(
        units,
        compute,
        cache,
        resume=bool(args.resume),
    )
    scores = {
        unit.unit_id: [float(value) for value in payload]
        for unit, payload in zip(units, payloads, strict=True)
    }
    args.upstream.write_text(json.dumps(scores), encoding="utf-8")


def _load_oracle(root: Path, checkpoint: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(root))
    from potts_mpnn_utils import PottsMPNN

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
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
    cfg = SimpleNamespace(
        dev="cpu",
        model=SimpleNamespace(vocab=21),
        inference=SimpleNamespace(
            ddG=False,
            filter=False,
            mean_norm=False,
            max_tokens=10**12,
            skip_gaps=False,
            noise=0.0,
        ),
    )
    return {"model": model, "cfg": cfg, "np": np}


def _oracle_one(prepared: dict[str, Any], payload: dict[str, Any]) -> list[float]:
    from potts_mpnn_utils import parse_PDB
    from run_utils import score_seqs

    pdb = parse_PDB(payload["pdb"], skip_gaps=False)
    wt = pdb[0]["seq"]
    seqs = [wt, *payload["random"]]
    pred, _scored, _ref = score_seqs(
        prepared["model"],
        prepared["cfg"],
        pdb,
        prepared["np"].zeros(len(seqs), dtype=prepared["np"].float32),
        seqs,
    )
    return [float(value) for value in pred.reshape(-1).detach().cpu()]


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.oracle_worker:
        _oracle_worker(args)
        return
    if args.arm:
        payload = _run_arm(args)
        payload_json = json.dumps(payload)
        # Never hand the payload back on stdout: the checkpoint converter prints to
        # stdout, which silently corrupted every arm that had to build the model
        # (runs 03992b3d and 35f9a257 both reported "no parseable JSON").
        if args.payload_out is not None:
            args.payload_out.write_text(payload_json, encoding="utf-8")
        else:
            sys.stdout.write(payload_json)
        return
    logger = _logger()
    results = _parent(args, logger)
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
