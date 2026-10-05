"""Regression gate: the extracted `psa` / `ensemble_tools` code must not be imported at runtime.

Both targets are **absent from this tree** -- there is no `src/aminx/psa`, and
`ensemble_tools` survives only in two docstrings (`run/specs.py`, `types/protocols.py`).
That is what makes a plain ``not in sys.modules`` check nearly toothless here: a failed
import leaves *nothing* in ``sys.modules``, so

    try:
        import ensemble_tools
    except ImportError:
        pass

at aminx import time would sail straight past it while being exactly the coupling this
gate exists to forbid. These tests therefore record import *attempts* through a
``sys.meta_path`` finder, which fires whether or not the target is installed and whether
or not the import is guarded.

`test_attempt_recorder_fires` is the negative control. Without it, the two gates could
pass by being incapable of failing rather than by the coupling being absent.
"""
from __future__ import annotations

import subprocess
import sys

# The import under test runs in a fresh interpreter, because "was this imported?" is only
# answerable before anything else in the test session has imported it. The forbidden name
# arrives via argv rather than string interpolation so this stays plain, un-formatted
# source.
_PROBE = """
import sys

forbidden = sys.argv[1]
attempted = []


class _AttemptRecorder:
    # Records attempts, not completed imports. Returning None means "not my job", so the
    # real finders still run and the import behaves exactly as it would unobserved.
    def find_spec(self, name, path=None, target=None):
        if name == forbidden or name.startswith(forbidden + "."):
            attempted.append(name)
        return None


sys.meta_path.insert(0, _AttemptRecorder())

import aminx
import aminx.run.specs

if attempted:
    raise AssertionError("runtime import attempt: " + repr(attempted))
if forbidden in sys.modules:
    raise AssertionError(forbidden + " in sys.modules after aminx core import")
print("OK: " + forbidden + " never attempted")
"""

# The negative control swaps the two aminx imports for the pattern being guarded against.
_IMPORT_UNDER_TEST = "import aminx\nimport aminx.run.specs"


def _run_probe(probe: str, forbidden: str) -> subprocess.CompletedProcess[str]:
    """Run `probe` in a fresh interpreter.

    No PYTHONPATH is set. An earlier version pointed it at
    /home/marielle/projects/tev_design/aminx/src -- a directory that exists on no
    machine, and assigned rather than prepended, so it would have discarded a real
    PYTHONPATH. `sys.executable` already resolves the project venv, which is how this
    ever worked.
    """
    return subprocess.run(
        [sys.executable, "-c", probe, forbidden],
        capture_output=True,
        text=True,
    )


def _run_isolation_check(module_path: str) -> None:
    result = _run_probe(_PROBE, module_path)
    assert result.returncode == 0, (
        f"Isolation check failed for {module_path!r}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_psa_not_imported_at_runtime() -> None:
    """aminx.psa must not be imported, or attempted, when aminx core is imported."""
    _run_isolation_check("aminx.psa")


def test_ensemble_tools_not_imported_at_runtime() -> None:
    """The sibling ensemble_tools package must not be eagerly imported at runtime."""
    _run_isolation_check("ensemble_tools")


def test_attempt_recorder_fires() -> None:
    """Negative control: the gate must FAIL on the exact pattern it exists to catch.

    A guarded import of an absent module -- which is what both targets are in this tree.
    If this test ever passes while reporting returncode 0, the two gates above have
    stopped being able to fail and are no longer evidence of anything.
    """
    probe = _PROBE.replace(
        _IMPORT_UNDER_TEST,
        "try:\n    import aminx_absent_probe_target\nexcept ImportError:\n    pass",
    )
    assert probe != _PROBE, "negative control failed to patch the probe source"

    result = _run_probe(probe, "aminx_absent_probe_target")

    assert result.returncode != 0, (
        "the attempt recorder did not fire on a guarded import of an absent module, "
        f"so the isolation gates cannot fail either:\nstdout: {result.stdout}"
    )
    assert "runtime import attempt" in result.stderr, result.stderr
