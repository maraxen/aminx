"""Test RS-6b flat-field ban gate via ast-grep.

Verifies that the rs6b-host-flat-field-ban YAML rule:
  (a) FIRES on a planted violation placed in src/aminx/host/ (matches files: glob).
  (b) Is CLEAN on the real src/aminx/host/ tree (0 violations) when no probe
      files are present.
"""

import subprocess
from pathlib import Path

import pytest


_RULE_PATH = ".ast-grep/rules/rs6b-host-flat-field-ban.yml"
_REPO_ROOT = Path(__file__).parent.parent.parent


def run_ast_grep_rule(
    rule_path: str, scan_path: str | None = None, cwd: Path = _REPO_ROOT
) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Mirrors the helper in tests/test_xtrax_boundary_lint.py.

    Args:
        rule_path: Path to the rule YAML, relative to repo root (or absolute).
        scan_path: Optional explicit path to scan; overrides files: in the rule.
        cwd: Directory the scan runs from. The rule's files: globs resolve against it.

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
            cwd=str(cwd),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        pytest.skip(
            "ast-grep (sg) not found in PATH. Install via: npm install -g @ast-grep/cli"
        )


def _scan_probe(tmp_path: Path, stem: str, code: str) -> tuple[bool, str]:
    """Scan one planted probe with the real rule, outside the repo tree.

    The probe goes under tmp_path/src/aminx/host/ and sg runs from tmp_path, so the rule's
    'files: src/aminx/host/**/*.py' glob and its prep.py ignore still apply, while no file is
    written into the shared src/ tree. Parallel xdist workers that glob src/ therefore never see
    a probe (the earlier race with tests/lint/test_options_plumbing.py).
    """
    host_dir = tmp_path / "src" / "aminx" / "host"
    host_dir.mkdir(parents=True)
    (host_dir / f"probe_{stem}.py").write_text(code, encoding="utf-8")
    return run_ast_grep_rule(str(_REPO_ROOT / _RULE_PATH), cwd=tmp_path)


def test_rs6b_gate_fires_on_planted_violation(tmp_path: Path) -> None:
    """Assert the rule FIRES when src/aminx/host/ contains a sampling-exclusive flat read.

    Plants a probe under a tmp_path mirror of src/aminx/host/ so the rule's
    'files: src/aminx/host/**/*.py' glob matches it, then asserts the rule
    reports at least one violation.
    """
    violation_code = """\
def sample(spec):
    # This should be flagged: direct flat read of sampling-exclusive field
    n = spec.num_samples
    return n
"""
    success, output = _scan_probe(tmp_path, "rs6b_gate_probe", violation_code)

    assert not success, (
        f"Rule should have fired on spec.num_samples in planted violation, but passed.\n"
        f"Rule output:\n{output}"
    )


def test_rs6b_gate_fires_on_omit_aa_planted_violation(tmp_path: Path) -> None:
    """Assert the rule FIRES on a planted flat read of omit_aa."""
    violation_code = """\
def sample(spec):
    omitted = spec.omit_aa
    return omitted
"""
    success, output = _scan_probe(tmp_path, "rs6b_omit_aa_probe", violation_code)

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
