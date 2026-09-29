"""Test RS-6b flat-field ban gate via ast-grep.

Verifies that the rs6b-host-flat-field-ban YAML rule:
  (a) FIRES on a planted violation placed in src/aminx/host/ (matches files: glob).
  (b) Is CLEAN on the real src/aminx/host/ tree (0 violations) when no probe
      files are present.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


_RULE_PATH = ".ast-grep/rules/rs6b-host-flat-field-ban.yml"
_REPO_ROOT = Path(__file__).parent.parent.parent


def run_ast_grep_rule(rule_path: str, scan_path: str | None = None) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Mirrors the helper in tests/test_xtrax_boundary_lint.py.

    Args:
        rule_path: Path to the rule YAML, relative to repo root.
        scan_path: Optional explicit path to scan; overrides files: in the rule.

    Returns:
        (True, "") if rule passes (0 violations)
        (False, output) if rule reports violations (non-zero exit)

    Raises:
        pytest.skip: if ast-grep (sg) is not installed
    """
    cmd = ["sg", "scan", "--rule", rule_path]
    if scan_path is not None:
        cmd.append(scan_path)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        pytest.skip(
            "ast-grep (sg) not found in PATH. Install via: npm install -g @ast-grep/cli"
        )


def test_rs6b_gate_fires_on_planted_violation() -> None:
    """Assert the rule FIRES when src/aminx/host/ contains a sampling-exclusive flat read.

    Plants a temporary file directly inside src/aminx/host/ so the rule's
    'files: src/aminx/host/**/*.py' glob matches it, then asserts the rule
    reports at least one violation.  The file is removed in the finally block
    regardless of test outcome.
    """
    violation_code = """\
def sample(spec):
    # This should be flagged: direct flat read of sampling-exclusive field
    n = spec.num_samples
    return n
"""
    host_dir = _REPO_ROOT / "src" / "aminx" / "host"
    probe_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(host_dir),
            suffix="_rs6b_gate_probe.py",
            mode="w",
            delete=False,
        ) as f:
            f.write(violation_code)
            probe_path = f.name

        success, output = run_ast_grep_rule(_RULE_PATH)

    finally:
        if probe_path and os.path.exists(probe_path):
            os.unlink(probe_path)

    assert not success, (
        f"Rule should have fired on spec.num_samples in planted violation, but passed.\n"
        f"Rule output:\n{output}"
    )


def test_rs6b_gate_fires_on_omit_aa_planted_violation() -> None:
    """Assert the rule FIRES on a planted flat read of omit_aa."""
    violation_code = """\
def sample(spec):
    omitted = spec.omit_aa
    return omitted
"""
    host_dir = _REPO_ROOT / "src" / "aminx" / "host"
    probe_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(host_dir),
            suffix="_rs6b_omit_aa_probe.py",
            mode="w",
            delete=False,
        ) as f:
            f.write(violation_code)
            probe_path = f.name

        success, output = run_ast_grep_rule(_RULE_PATH)
    finally:
        if probe_path and os.path.exists(probe_path):
            os.unlink(probe_path)

    assert not success, (
        f"Rule should have fired on spec.omit_aa in planted violation, but passed.\n"
        f"Rule output:\n{output}"
    )


def test_rs6b_gate_clean_on_real_host_tree() -> None:
    """Assert the rule produces 0 violations on the real src/aminx/host/ tree.

    Uses the rule's own files:/ignores: configuration (no explicit scan_path),
    which excludes prep.py per the rule YAML.
    """
    success, output = run_ast_grep_rule(_RULE_PATH)

    assert success, (
        f"RS-6b flat-field ban gate found violations in src/aminx/host/:\n{output}\n\n"
        "Fix: migrate sampling-exclusive flat reads to run_spec sub-configs.\n"
        "Sampling-exclusive fields: num_samples, bias, fixed_positions, fixed_tokens,\n"
        "  use_unified_driver, compute_pseudo_perplexity, omit_aa, omit_aa_per_position.\n"
        "Exemption: prep.py is intentionally excluded."
    )
