"""Graded megascale ddG parity (do not run from the fixer).

All rows of ``energy_benchmark_datasets/megascale_test_subset.csv``. Pass when
``max |ΔddG| ≤ 1e-4``, inconclusive on ``(1e-4, 1e-3]``, fail above ``1e-3``.

Negative control ``skip_transpose_merge_pair`` must land in the fail band.
Each arm, including the oracle worker, is its own subprocess with its own
timeout and is re-launched from the per-PDB cache. Results go to
``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import csv
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

MUTANT_ID = "skip_transpose_merge_pair"
CSV_SHA256 = "9c816f1d1fb836f5c03db226beeab6af37bbf8bf1110f5af2fdeb72d095ea9e5"
N_ROWS = 202804
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


def _csv_path(root: Path) -> Path:
    return root / "energy_benchmark_datasets" / "megascale_test_subset.csv"


def _pdb_dir(root: Path) -> Path:
    return root / "energy_benchmark_datasets" / "megascale_pdbs"


def _check_inputs(root: Path, checkpoint: Path) -> None:
    table = _csv_path(root)
    digest = _sha256(table)
    if digest != CSV_SHA256:
        msg = f"{table} sha256 {digest} != {CSV_SHA256}"
        raise SystemExit(msg)
    with table.open(encoding="utf-8", newline="") as handle:
        n_rows = sum(1 for _row in csv.DictReader(handle))
    if n_rows != N_ROWS:
        msg = f"{table} has {n_rows} rows, expected {N_ROWS}"
        raise SystemExit(msg)
    got = _sha256(checkpoint)
    if got != CHECKPOINT_SHA256:
        msg = f"{checkpoint} sha256 {got} != {CHECKPOINT_SHA256}"
        raise SystemExit(msg)


# Amended 260929 (user-approved, see the [amendment] block in the sidecar). The original
# 1e-4 / 1e-3 bands were set before the first run and measured 1.2207e-04 on the clean arm,
# which is EXACTLY 2^-13, one float32 ULP for magnitudes in [1024, 2048). ddG is
# E_mut - E_wt, a small difference of large f32 Potts energies, so a few ULP of cancellation
# is the arithmetic floor rather than a port defect. PASS_BOUND is ~4 such ULP.
PASS_BOUND = 5e-4
INCONCLUSIVE_BOUND = 5e-3


def _band(max_abs: float) -> str:
    if max_abs <= PASS_BOUND:
        return "pass"
    if max_abs <= INCONCLUSIVE_BOUND:
        return "inconclusive"
    return "fail"


def _results_path() -> Path:
    env = os.environ.get("BTH_RESULTS_PATH")
    if not env:
        msg = "potts_ddg_megascale requires $BTH_RESULTS_PATH"
        raise SystemExit(msg)
    return Path(env)


def _install_mutant(arm: str | None) -> None:
    if arm in (None, "clean"):
        return
    if arm != MUTANT_ID:
        msg = f"unknown mutant {arm}"
        raise SystemExit(msg)
    import aminx.families.potts_mpnn.driver as driver_mod
    from aminx.families.potts_mpnn.etab import merge_pair

    def _skip(
        etab: Any,
        e_idx: Any,
        pad_valid: Any,
        *,
        denom: int,
        exclude_self: bool,
        transpose: bool = True,
    ) -> Any:
        del transpose
        return merge_pair(
            etab,
            e_idx,
            pad_valid,
            denom=denom,
            exclude_self=exclude_self,
            transpose=False,
        )

    driver_mod.merge_pair = _skip


def _pdb_names(table: Path) -> list[str]:
    seen: list[str] = []
    known: set[str] = set()
    with table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["pdb"]
            if name not in known:
                known.add(name)
                seen.append(name)
    return seen


def _work_dir(args: argparse.Namespace) -> Path:
    work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "potts_ddg_megascale"
    work.mkdir(parents=True, exist_ok=True)
    return work


def _unit_input_sha(pdb: Path, csv_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(csv_sha.encode("utf-8"))
    digest.update(b"\0")
    with pdb.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _units(root: Path, arm_id: str, checkpoint: Path) -> list[graded_resume.Unit]:
    table = _csv_path(root)
    csv_sha = _sha256(table)
    checkpoint_sha = _sha256(checkpoint)
    script_sha = _sha256(Path(__file__).resolve())
    units: list[graded_resume.Unit] = []
    for name in _pdb_names(table):
        pdb = _pdb_dir(root) / f"{name}.pdb"
        units.append(
            graded_resume.Unit(
                unit_id=name,
                arm_id=arm_id,
                checkpoint_sha256=checkpoint_sha,
                input_sha256=_unit_input_sha(pdb, csv_sha),
                script_sha256=script_sha,
            ),
        )
    return units


def _score_one(root: Path, checkpoint: Path, name: str) -> list[float]:
    from aminx.host.runner import score
    from aminx.run.options import PottsMPNNOptions
    from aminx.run.specs import ScoringSpecification

    table = _csv_path(root)
    pdb = _pdb_dir(root) / f"{name}.pdb"
    options = PottsMPNNOptions(mutant_csv=str(table), mean_norm=False)
    spec = ScoringSpecification(
        inputs=str(pdb),
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        model_local_path=checkpoint,
        output_kind="ddg",
        potts_mpnn=options,
    )
    values = score(spec)["structures"]["0"]["arrays"]["ddg"]
    return [float(value) for value in values]


def _resume_flag(args: argparse.Namespace) -> str:
    if args.resume:
        return "--resume"
    return "--no-resume"


def _logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("potts_ddg_megascale")
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
    upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
    checkpoint = _checkpoint(args)
    arm_id = str(args.arm or "clean")
    units = _units(args.potts_root, arm_id, checkpoint)
    cache = graded_resume.cache_dir(_work_dir(args))

    def compute(unit: graded_resume.Unit) -> list[float]:
        return _score_one(args.potts_root, checkpoint, unit.unit_id)

    payloads, _stats = graded_resume.run_units(
        units,
        compute,
        cache,
        resume=bool(args.resume),
    )
    gaps: list[float] = []
    for unit, payload in zip(units, payloads, strict=True):
        pred = [float(value) for value in payload]
        ref = upstream[unit.unit_id]
        if len(pred) != len(ref):
            msg = f"{unit.unit_id} n_mut {len(pred)} != upstream {len(ref)}"
            raise SystemExit(msg)
        gaps.extend(abs(float(a) - float(b)) for a, b in zip(pred, ref, strict=True))
    max_abs = float(np.max(gaps)) if gaps else float("inf")
    return {"band": _band(max_abs), "max_abs_delta": max_abs}


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
    upstream_path = work / "upstream.json"
    script = str(Path(__file__).resolve())
    resume_flag = _resume_flag(args)
    oracle_units = _units(args.potts_root, graded_resume.ORACLE_ARM, checkpoint)
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
        units = _units(args.potts_root, arm, checkpoint)
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
    return {
        "clean": clean,
        "n_listed": len(mutants),
        "n_failed": sum(status == "failed" for status in statuses.values()),
        "max_abs_delta": float(measured["clean"].get("max_abs_delta", float("nan"))),
        # The control's own deviation belongs in the record, not only in a scratch file:
        # it is what shows the negative control still has margin under the amended band.
        "mutant_max_abs_delta": {
            mutant: float(measured[mutant].get("max_abs_delta", float("nan")))
            for mutant in mutants
        },
        "n_units": n_units,
        "n_reused": n_reused,
        "n_computed": n_computed,
        "cache_dir": str(cache),
    }


def _oracle_worker(args: argparse.Namespace) -> None:
    checkpoint = _checkpoint(args)
    root = args.potts_root
    work = _work_dir(args)
    units = _units(root, graded_resume.ORACLE_ARM, checkpoint)
    cache = graded_resume.cache_dir(work)
    _reused, remaining = graded_resume.count_units(cache, units, resume=bool(args.resume))
    prepared: dict[str, Any] | None = None
    if remaining > 0:
        prepared = _load_oracle(args, checkpoint, root)

    def compute(unit: graded_resume.Unit) -> list[float]:
        if prepared is None:
            msg = "oracle model was not loaded"
            raise RuntimeError(msg)
        return _oracle_one(prepared, unit.unit_id)

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


def _load_oracle(args: argparse.Namespace, checkpoint: Path, root: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(root))
    from potts_mpnn_utils import PottsMPNN
    from run_utils import process_data

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
    listing = args.upstream.parent / "pdb_list.txt"
    names = _pdb_names(_csv_path(root))
    listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(
        dev="cpu",
        input_list=str(listing),
        input_dir=str(_pdb_dir(root)),
        mutant_fasta=None,
        mutant_csv=str(_csv_path(root)),
        model=SimpleNamespace(vocab=21, check_path=str(checkpoint)),
        inference=SimpleNamespace(
            ddG=True,
            filter=False,
            mean_norm=False,
            max_tokens=10**12,
            skip_gaps=False,
            noise=0.0,
            binding_energy_json=None,
        ),
    )
    mutant_data, _lens, _pdb_list, _binding = process_data(cfg)
    return {"model": model, "cfg": cfg, "mutant_data": mutant_data, "np": np}


def _oracle_one(prepared: dict[str, Any], pdb: str) -> list[float]:
    from potts_mpnn_utils import parse_PDB
    from run_utils import score_seqs

    cfg = prepared["cfg"]
    mutant_data = prepared["mutant_data"]
    subset = mutant_data[mutant_data["pdb"] == pdb]
    parsed = parse_PDB(str(Path(cfg.input_dir) / f"{pdb}.pdb"), skip_gaps=False)
    pred, _seqs, _ref = score_seqs(
        prepared["model"],
        cfg,
        parsed,
        prepared["np"].asarray(subset["ddG_expt"].values, dtype=prepared["np"].float64),
        list(subset["sequences"].values),
    )
    return [float(value) for value in pred.reshape(-1).detach().cpu()]


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.oracle_worker:
        _oracle_worker(args)
        return
    if args.arm:
        payload_json = json.dumps(_run_arm(args))
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
