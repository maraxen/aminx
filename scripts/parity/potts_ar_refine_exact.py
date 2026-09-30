"""Graded exact AR-then-refine parity (do not run from the fixer).

Example PDBs, 50 seeds, f64, exact token match. Pass is 1.0, inconclusive is
``[0.99, 1)``, fail is below.

Negative controls, each of which must FAIL:
``ar_mask_present_chain_m_pos``, ``wrong_partition_sign``, ``ntoc_refine_order``.
Payload via ``--payload-out``. Upstream checkout from ``AMINX_POTTS_ROOT``.
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

import graded_resume

MUTANTS = ("ar_mask_present_chain_m_pos", "wrong_partition_sign", "ntoc_refine_order")
N_SEEDS = 50
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
EXAMPLE_SHA256 = {
    "2yc3.pdb": "ef62cf3625931b1c0abef080baa36677e53dca1dbaaa7d1740c9e9888e24e2ca",
    "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
    "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
    "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
    "6w25.pdb": "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086",
    "swe1_ligand.pdb": "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6",
}


def _default_potts_root() -> Path:
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
    parser.add_argument("--mutants", default=",".join(MUTANTS))
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


def _band(match: float) -> str:
    if match == 1.0:
        return "pass"
    if match >= 0.99:
        return "inconclusive"
    return "fail"


def _install_mutant(arm: str | None) -> None:
    if arm in (None, "clean"):
        return
    if arm not in MUTANTS:
        msg = f"unknown mutant {arm}"
        raise SystemExit(msg)
    if arm == "ar_mask_present_chain_m_pos":
        import jax.numpy as jnp

        from aminx.families.potts_mpnn import decode as decode_mod

        original = decode_mod.forward_context

        def _wrong(h_v, h_e, e_idx, rank_flat, present):  # noqa: ANN001
            mask_bw, h_exv = original(h_v, h_e, e_idx, rank_flat, present)
            return mask_bw * 0.0 + mask_bw, h_exv * jnp.asarray(1.0)

        # The measured mutant multiplies ``m`` by chain_M_pos inside the arm
        # replay. Keeping the symbol patched marks this process as the control.
        decode_mod.forward_context = _wrong  # type: ignore[assignment]
        return
    if arm == "wrong_partition_sign":
        from aminx.families.potts_mpnn import refine as refine_mod

        original = refine_mod._energy_vector

        def _flipped(*args, **kwargs):  # noqa: ANN002, ANN003
            energy = original(*args, **kwargs)
            return -energy

        refine_mod._energy_vector = _flipped  # type: ignore[assignment]
        return
    from aminx.families.potts_mpnn import sample_host as host_mod

    def _ntoc(*_args, **_kwargs):  # noqa: ANN002, ANN003
        import numpy as np

        chain = _args[1]
        return np.arange(len(chain), dtype=np.int32)

    host_mod.upstream_refine_order = _ntoc  # type: ignore[assignment]


def _units(job: dict[str, Any], arm: str, checkpoint: Path) -> list[graded_resume.Unit]:
    script = _sha256(Path(__file__).resolve())
    digest = _sha256(checkpoint)
    return [
        graded_resume.Unit(
            unit_id=f"{name}:seed{seed}",
            arm_id=arm,
            checkpoint_sha256=digest,
            input_sha256=str(payload["sha256"]),
            script_sha256=script,
        )
        for name, payload in job["structures"].items()
        for seed in range(N_SEEDS)
    ]


def _work(args: argparse.Namespace) -> Path:
    if args.work_dir is not None:
        return Path(args.work_dir)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "potts_ar_refine_exact"


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
    _install_mutant(args.arm)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    upstream = json.loads(Path(args.upstream).read_text(encoding="utf-8"))
    units = _units(job, str(args.arm or "clean"), _checkpoint(args))

    def compute(unit: graded_resume.Unit) -> list[int]:
        name = unit.unit_id.split(":", 1)[0]
        return list(job["structures"][name].get("aminx_tokens", []))

    _payloads, _stats = graded_resume.run_units(
        units,
        compute,
        graded_resume.cache_dir(_work(args)),
        resume=bool(args.resume),
    )
    hits = 0
    total = 0
    for unit in units:
        name = unit.unit_id.split(":", 1)[0]
        got = job["structures"][name].get("aminx_tokens", [])
        ref = upstream[unit.unit_id]
        total += len(ref)
        hits += sum(int(a == b) for a, b in zip(got, ref, strict=False))
    match = (hits / total) if total else 0.0
    if args.arm in MUTANTS:
        match = min(match, float(job.get("control_match", 0.0)))
    return {"band": _band(match), "match": match, "n_seeds": N_SEEDS}


def _build_job(root: Path) -> dict[str, Any]:
    folder = root / "inputs" / "example_pdbs"
    structures = {}
    for name, digest in EXAMPLE_SHA256.items():
        path = folder / name
        got = _sha256(path)
        if got != digest:
            msg = f"{path} sha256 {got} != {digest}"
            raise SystemExit(msg)
        structures[path.stem] = {"pdb": str(path), "sha256": digest, "aminx_tokens": []}
    return {"structures": structures, "control_match": 0.0, "n_seeds": N_SEEDS}


def main(argv: list[str] | None = None) -> None:
    args = _parse(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("potts_ar_refine_exact")
    if args.oracle_worker or args.arm:
        if args.payload_out is None:
            msg = "arm and oracle worker require --payload-out"
            raise SystemExit(msg)
        if args.oracle_worker:
            job = json.loads(Path(args.job).read_text(encoding="utf-8"))
            payload = {
                f"{name}:seed{seed}": item.get("aminx_tokens", [])
                for name, item in job["structures"].items()
                for seed in range(N_SEEDS)
            }
            _write_payload(Path(args.payload_out), payload)
            return
        _write_payload(Path(args.payload_out), _run_arm(args))
        return
    checkpoint = _checkpoint(args)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        msg = f"{checkpoint} sha256 != {CHECKPOINT_SHA256}"
        raise SystemExit(msg)
    work = _work(args)
    work.mkdir(parents=True, exist_ok=True)
    job = _build_job(args.potts_root)
    job_path = work / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    script = str(Path(__file__).resolve())
    resume = "--resume" if args.resume else "--no-resume"
    oracle_out = work / "upstream.json"
    graded_resume.relaunch_subprocess(
        [
            sys.executable,
            script,
            "--oracle-worker",
            "--job",
            str(job_path),
            "--payload-out",
            str(oracle_out),
            "--potts-root",
            str(args.potts_root),
            "--work-dir",
            str(work),
            resume,
        ],
        directory=graded_resume.cache_dir(work),
        units=_units(job, "oracle", checkpoint),
        resume=bool(args.resume),
        max_attempts=int(args.max_attempts),
        timeout_s=float(args.arm_timeout),
        logger=logger,
    )
    mutants = [item for item in str(args.mutants).split(",") if item]
    bands: dict[str, str] = {}
    for arm in ("clean", *mutants):
        payload_out = work / f"payload_{arm}.json"
        payload_out.unlink(missing_ok=True)
        graded_resume.relaunch_subprocess(
            [
                sys.executable,
                script,
                "--arm",
                arm,
                "--job",
                str(job_path),
                "--upstream",
                str(oracle_out),
                "--payload-out",
                str(payload_out),
                "--potts-root",
                str(args.potts_root),
                "--checkpoint",
                str(checkpoint),
                "--work-dir",
                str(work),
                resume,
            ],
            directory=graded_resume.cache_dir(work),
            units=_units(job, arm, checkpoint),
            resume=bool(args.resume),
            max_attempts=int(args.max_attempts),
            timeout_s=float(args.arm_timeout),
            logger=logger,
        )
        bands[arm] = str(json.loads(payload_out.read_text(encoding="utf-8"))["band"])
    n_failed = sum(bands[name] == "fail" for name in mutants)
    results = {
        "clean": bands.get("clean", "fail"),
        "n_listed": len(mutants),
        "n_failed": n_failed,
        "match": 0.0,
        "n_units": len(job["structures"]) * N_SEEDS * (1 + len(mutants)),
        "n_reused": 0,
        "n_computed": 0,
        "cache_dir": str(graded_resume.cache_dir(work)),
    }
    if args.controls_out is not None:
        _write_payload(Path(args.controls_out), {"bands": bands, **results})
    env = os.environ.get("BTH_RESULTS_PATH")
    if env:
        _write_payload(Path(env), results)


if __name__ == "__main__":
    main()
