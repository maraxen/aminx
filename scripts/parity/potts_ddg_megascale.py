"""Graded megascale ddG parity (do not run from the fixer).

All rows of ``energy_benchmark_datasets/megascale_test_subset.csv``. Pass when
``max |ΔddG| ≤ 1e-4``, inconclusive on ``(1e-4, 1e-3]``, fail above ``1e-3``.

Negative control ``skip_transpose_merge_pair`` must land in the fail band.
Each arm is a fresh subprocess. Results go to ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MUTANT_ID = "skip_transpose_merge_pair"
CSV_SHA256 = "9c816f1d1fb836f5c03db226beeab6af37bbf8bf1110f5af2fdeb72d095ea9e5"
N_ROWS = 202804
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potts-root", type=Path, default=Path("/home/marielle/repos/PottsMPNN"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--oracle-python", default=os.environ.get("POTTS_ORACLE_PYTHON"))
    parser.add_argument("--mutants", default=MUTANT_ID)
    parser.add_argument("--controls-out", type=Path, default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--upstream", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
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


def _band(max_abs: float) -> str:
    if max_abs <= 1e-4:
        return "pass"
    if max_abs <= 1e-3:
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


def _score_aminx(root: Path, checkpoint: Path) -> dict[str, list[float]]:
    from aminx.host.runner import score
    from aminx.run.options import PottsMPNNOptions
    from aminx.run.specs import ScoringSpecification

    table = _csv_path(root)
    found: dict[str, list[float]] = {}
    options = PottsMPNNOptions(mutant_csv=str(table), mean_norm=False)
    for name in _pdb_names(table):
        pdb = _pdb_dir(root) / f"{name}.pdb"
        spec = ScoringSpecification(
            inputs=str(pdb),
            model_family="pottsmpnn",
            checkpoint_id="pottsmpnn_vanilla_20",
            model_local_path=checkpoint,
            output_kind="ddg",
            potts_mpnn=options,
        )
        values = score(spec)["structures"]["0"]["arrays"]["ddg"]
        found[name] = [float(value) for value in values]
    return found


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    _install_mutant(args.arm)
    upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
    ours = _score_aminx(args.potts_root, _checkpoint(args))
    gaps: list[float] = []
    for name, pred in ours.items():
        ref = upstream[name]
        if len(pred) != len(ref):
            msg = f"{name} n_mut {len(pred)} != upstream {len(ref)}"
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


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _checkpoint(args)
    _check_inputs(args.potts_root, checkpoint)
    work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "potts_ddg_megascale"
    work.mkdir(parents=True, exist_ok=True)
    upstream_path = work / "upstream.json"
    subprocess.run(
        [
            _oracle_python(args),
            str(Path(__file__).resolve()),
            "--oracle-worker",
            "--potts-root",
            str(args.potts_root),
            "--checkpoint",
            str(checkpoint),
            "--upstream",
            str(upstream_path),
        ],
        check=True,
    )
    mutants = [item for item in args.mutants.split(",") if item]
    measured: dict[str, dict[str, Any]] = {}
    for arm in ("clean", *mutants):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--arm",
                arm,
                "--upstream",
                str(upstream_path),
                "--checkpoint",
                str(checkpoint),
                "--potts-root",
                str(args.potts_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            measured[arm] = {"band": "error", "detail": completed.stderr[-500:]}
            continue
        measured[arm] = json.loads(completed.stdout)
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
                    "weights": {str(checkpoint): _sha256(checkpoint)},
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
    }


def _oracle_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(args.potts_root))
    from potts_mpnn_utils import PottsMPNN
    from run_utils import process_data, score_seqs

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    root = args.potts_root
    listing = args.upstream.parent / "pdb_list.txt"
    names = _pdb_names(_csv_path(root))
    listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(
        dev="cpu",
        input_list=str(listing),
        input_dir=str(_pdb_dir(root)),
        mutant_fasta=None,
        mutant_csv=str(_csv_path(root)),
        model=SimpleNamespace(vocab=21, check_path=str(args.checkpoint)),
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
    mutant_data, _lens, pdb_list, _binding = process_data(cfg)
    from potts_mpnn_utils import parse_PDB

    scores: dict[str, list[float]] = {}
    for pdb in pdb_list:
        subset = mutant_data[mutant_data["pdb"] == pdb]
        parsed = parse_PDB(str(Path(cfg.input_dir) / f"{pdb}.pdb"), skip_gaps=False)
        pred, _seqs, _ref = score_seqs(
            model,
            cfg,
            parsed,
            np.asarray(subset["ddG_expt"].values, dtype=np.float64),
            list(subset["sequences"].values),
        )
        scores[pdb] = [float(value) for value in pred.reshape(-1).detach().cpu()]
    args.upstream.write_text(json.dumps(scores), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.oracle_worker:
        _oracle_worker(args)
        return
    if args.arm:
        sys.stdout.write(json.dumps(_run_arm(args)))
        return
    results = _parent(args)
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
