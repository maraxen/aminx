"""Per-PDB cache and per-arm relaunch for the graded Potts parity scripts.

A unit is one PDB for one arm. The cache key is the sha256 of the unit id, the
arm id, the checkpoint sha256, the input CSV/PDB sha256, and the script sha256.
The body is written first and the completion stamp second. A unit is reused only
when ``--resume`` is on, the stamp matches the key, and the body parses.

``relaunch_subprocess`` runs one arm (or the oracle worker) with its own timeout
and starts that command again after a timeout or a non-zero exit. Each attempt
logs how many units are already reusable and how many remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ORACLE_ARM = "oracle"


@dataclass(frozen=True)
class Unit:
    """One PDB scored under one arm."""

    unit_id: str
    arm_id: str
    checkpoint_sha256: str
    input_sha256: str
    script_sha256: str

    @property
    def cache_key(self) -> str:
        return unit_cache_key(
            unit_id=self.unit_id,
            arm_id=self.arm_id,
            checkpoint_sha256=self.checkpoint_sha256,
            input_sha256=self.input_sha256,
            script_sha256=self.script_sha256,
        )


@dataclass(frozen=True)
class PassStats:
    """Reuse counts for one pass over a unit list."""

    n_units: int
    n_reused: int
    n_computed: int


@dataclass(frozen=True)
class LaunchResult:
    """One relaunch loop, including units stamped across its attempts."""

    returncode: int
    stdout: str
    stderr: str
    n_units: int
    n_reused: int
    n_computed: int


def unit_cache_key(
    *,
    unit_id: str,
    arm_id: str,
    checkpoint_sha256: str,
    input_sha256: str,
    script_sha256: str,
) -> str:
    """Sha256 of the identity fields, in a canonical JSON object."""
    blob = json.dumps(
        {
            "arm_id": arm_id,
            "checkpoint_sha256": checkpoint_sha256,
            "input_sha256": input_sha256,
            "script_sha256": script_sha256,
            "unit_id": unit_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_dir(work: Path) -> Path:
    path = work / "units"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_units(
    unit_ids: Sequence[str],
    *,
    arm_id: str,
    checkpoint_sha256: str,
    input_sha256: str,
    script_sha256: str,
) -> list[Unit]:
    """Units that share every identity field except the PDB name."""
    return [
        Unit(
            unit_id=unit_id,
            arm_id=arm_id,
            checkpoint_sha256=checkpoint_sha256,
            input_sha256=input_sha256,
            script_sha256=script_sha256,
        )
        for unit_id in unit_ids
    ]


def unit_paths(directory: Path, key: str) -> tuple[Path, Path]:
    """Return ``(body, stamp)`` for a cache key."""
    return directory / f"{key}.json", directory / f"{key}.stamp"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def store_unit(directory: Path, key: str, payload: Any) -> None:
    """Write the body, then the stamp. A crash before the stamp is a miss."""
    body, stamp = unit_paths(directory, key)
    if stamp.exists():
        stamp.unlink()
    _atomic_write(body, json.dumps({"cache_key": key, "payload": payload}))
    _atomic_write(stamp, key + "\n")


def load_unit(directory: Path, key: str) -> Any | None:
    """Return the payload only when the stamp and body both match ``key``."""
    body, stamp = unit_paths(directory, key)
    if not body.is_file() or not stamp.is_file():
        return None
    try:
        stamped = stamp.read_text(encoding="utf-8").strip()
        if stamped != key:
            return None
        data = json.loads(body.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("cache_key") != key or "payload" not in data:
        return None
    return data["payload"]


def count_valid(directory: Path, units: Sequence[Unit]) -> int:
    return sum(1 for unit in units if load_unit(directory, unit.cache_key) is not None)


def count_units(directory: Path, units: Sequence[Unit], *, resume: bool) -> tuple[int, int]:
    """Return ``(reused, remaining)`` for the next attempt."""
    if not resume:
        return 0, len(units)
    reused = count_valid(directory, units)
    return reused, len(units) - reused


def run_units(
    units: Sequence[Unit],
    compute: Callable[[Unit], Any],
    directory: Path,
    *,
    resume: bool,
) -> tuple[list[Any], PassStats]:
    """Compute missing units. Persist each one before starting the next."""
    results: list[Any] = []
    n_reused = 0
    n_computed = 0
    for unit in units:
        key = unit.cache_key
        cached = load_unit(directory, key) if resume else None
        if cached is not None:
            n_reused += 1
            results.append(cached)
            continue
        payload = compute(unit)
        store_unit(directory, key, payload)
        n_computed += 1
        results.append(payload)
    stats = PassStats(n_units=len(units), n_reused=n_reused, n_computed=n_computed)
    return results, stats


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tally(
    *,
    n_units: int,
    physical_before: int,
    physical_after: int,
    resume: bool,
    returncode: int,
) -> tuple[int, int]:
    if not resume and returncode == 0:
        return 0, n_units
    if resume:
        reused = min(physical_before, n_units)
        computed = max(0, physical_after - physical_before)
        return reused, computed
    return 0, max(0, physical_after - physical_before)


def relaunch_subprocess(
    command: Sequence[str],
    *,
    directory: Path,
    units: Sequence[Unit],
    resume: bool,
    max_attempts: int,
    timeout_s: float,
    logger: logging.Logger,
) -> LaunchResult:
    """Run ``command`` until it exits 0, or ``max_attempts`` are exhausted.

    A timeout or any other non-zero exit starts the same command again. The
    command is expected to resume from ``directory``. The tally counts units
    that were already stamped before this loop as reused, and units stamped
    during the loop as computed.
    """
    if max_attempts < 1:
        msg = "max_attempts must be >= 1"
        raise ValueError(msg)
    if timeout_s <= 0:
        msg = "timeout_s must be positive"
        raise ValueError(msg)
    argv = list(command)
    physical_before = count_valid(directory, units)
    last_code = 1
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, max_attempts + 1):
        n_reused, n_remaining = count_units(directory, units, resume=resume)
        logger.info(
            "attempt %s units reused %s units remaining %s",
            attempt,
            n_reused,
            n_remaining,
        )
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            last_code = 124
            last_stdout = _as_text(exc.stdout)
            last_stderr = _as_text(exc.stderr)
            logger.warning("attempt %s timed out after %s s", attempt, timeout_s)
            continue
        last_code = completed.returncode
        last_stdout = completed.stdout
        last_stderr = completed.stderr
        if completed.returncode == 0:
            break
        logger.warning("attempt %s exit %s", attempt, completed.returncode)
    physical_after = count_valid(directory, units)
    n_reused_out, n_computed_out = _tally(
        n_units=len(units),
        physical_before=physical_before,
        physical_after=physical_after,
        resume=resume,
        returncode=last_code,
    )
    return LaunchResult(
        returncode=last_code,
        stdout=last_stdout,
        stderr=last_stderr,
        n_units=len(units),
        n_reused=n_reused_out,
        n_computed=n_computed_out,
    )


def _fake_worker(argv: list[str]) -> None:
    """Stdlib worker used by tests. Dies once so a relaunch must resume."""
    parser = argparse.ArgumentParser(description="Fake per-unit worker")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--unit-ids", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--counter", type=Path, required=True)
    parser.add_argument("--die-once", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    units = build_units(
        [item for item in str(args.unit_ids).split(",") if item],
        arm_id=str(args.arm_id),
        checkpoint_sha256=str(args.checkpoint_sha256),
        input_sha256=str(args.input_sha256),
        script_sha256=str(args.script_sha256),
    )
    die_after = 0
    die_once = args.die_once
    if isinstance(die_once, Path) and die_once.is_file():
        die_once.unlink()
        die_after = 1
    computed = 0
    counter = args.counter

    def compute(unit: Unit) -> dict[str, str]:
        nonlocal computed
        if die_after and computed >= die_after:
            os.kill(os.getpid(), signal.SIGKILL)
            msg = "killed"
            raise SystemExit(msg)
        computed += 1
        with counter.open("a", encoding="utf-8") as handle:
            handle.write(unit.unit_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"unit_id": unit.unit_id}

    _results, stats = run_units(units, compute, args.cache_dir, resume=bool(args.resume))
    sys.stdout.write(
        json.dumps(
            {
                "n_units": stats.n_units,
                "n_reused": stats.n_reused,
                "n_computed": stats.n_computed,
            },
        ),
    )


if __name__ == "__main__":
    _fake_worker(sys.argv[1:])
