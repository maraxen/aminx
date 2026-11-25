"""Test potts import boundary gate via ast-grep.

Verifies that the potts-import-boundary YAML rule:
  (a) FIRES on a planted violation placed in src/aminx/potts/ (matches files: glob).
  (b) Is CLEAN on the real src/aminx/potts/ tree (0 violations).
  (c) EXEMPTS designer.py even when it contains a banned import (validates
      ignores: glob functionality).

(a) and (c) plant files in a scratch copy of the layout under ``tmp_path``, never
in the real ``src/``. Planting in ``src/`` raced other tests under pytest-xdist:
the browser-validation inventory test walks ``src/aminx`` and saw the probe.

The rule enforces the isolation between aminx.potts (parallel model) and
aminx (MPNN stageset): potts.model, potts.poe, and potts.sampling must NOT
import from aminx.inference.decode, aminx.host.plan, aminx.types.stages, or
aminx.inference.logits. Only potts.designer is exempt from this restriction.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


_RULE_PATH = ".ast-grep/rules/potts-import-boundary.yml"
_REPO_ROOT = Path(__file__).parent.parent.parent


def run_ast_grep_rule(
    rule_path: str, scan_path: str | None = None, cwd: Path | None = None
) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Mirrors the helper in tests/lint/test_rs6b_flat_field_gate.py.

    Args:
        rule_path: Path to the rule YAML, relative to repo root.
        scan_path: Optional explicit path to scan; overrides files: in the rule.
        cwd: Directory to run in; defaults to the repo root. The rule's files:
            and ignores: globs are relative to it.

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
            cwd=str(cwd if cwd is not None else _REPO_ROOT),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        pytest.skip(
            "ast-grep (sg) not found in PATH. Install via: npm install -g @ast-grep/cli"
        )


def _scratch_tree(tmp_path: Path) -> Path:
    """A copy of the rule plus an empty ``src/aminx/potts/`` under ``tmp_path``.

    The rule's ``files:``/``ignores:`` globs are relative, so running ast-grep
    with ``cwd=tmp_path`` applies them to this tree exactly as to the real one.
    """
    rule = tmp_path / _RULE_PATH
    rule.parent.mkdir(parents=True)
    shutil.copyfile(_REPO_ROOT / _RULE_PATH, rule)
    potts_dir = tmp_path / "src" / "aminx" / "potts"
    potts_dir.mkdir(parents=True)
    return potts_dir


def test_potts_boundary_fires_on_planted_violation(tmp_path: Path) -> None:
    """Assert the rule FIRES when src/aminx/potts/ contains a banned import.

    Plants a file at src/aminx/potts/ in a scratch tree so the rule's
    'files: src/aminx/potts/**/*.py' glob matches it, then asserts the rule
    reports at least one violation.
    """
    violation_code = """\
from aminx.inference.decode import decode_one

def potts_decode():
    return decode_one()
"""
    potts_dir = _scratch_tree(tmp_path)
    (potts_dir / "planted_potts_boundary_probe.py").write_text(violation_code)

    success, output = run_ast_grep_rule(_RULE_PATH, cwd=tmp_path)

    assert not success, (
        f"Rule should have fired on banned import from aminx.inference.decode in planted violation, but passed.\n"
        f"Rule output:\n{output}"
    )


def test_potts_boundary_clean_on_real_tree() -> None:
    """Assert the rule produces 0 violations on the real src/aminx/potts/ tree.

    Uses the rule's own files:/ignores: configuration (no explicit scan_path),
    which covers potts/**/*.py and excludes designer.py per the rule YAML.
    """
    success, output = run_ast_grep_rule(_RULE_PATH)

    assert success, (
        f"potts-import-boundary gate found violations in src/aminx/potts/:\n{output}\n\n"
        "Fix: potts.model, potts.poe, and potts.sampling must NOT import from:\n"
        "  aminx.inference.decode, aminx.host.plan, aminx.types.stages,\n"
        "  aminx.inference.logits.\n"
        "Exemption: designer.py is intentionally excluded."
    )


def test_potts_boundary_exempts_designer(tmp_path: Path) -> None:
    """Assert designer.py is EXEMPT from the rule even with a banned import.

    Copies the real src/aminx/potts/designer.py into a scratch tree, appends a
    banned import, and asserts the rule passes there (designer.py is in
    ignores:). A control file with the same import must still fire, so the
    pass cannot come from the rule matching nothing. The real source is never
    modified.
    """
    potts_dir = _scratch_tree(tmp_path)
    banned = "\nfrom aminx.inference.logits import compute_logits\n"
    real_designer = _REPO_ROOT / "src" / "aminx" / "potts" / "designer.py"
    (potts_dir / "designer.py").write_text(real_designer.read_text() + banned)

    success, output = run_ast_grep_rule(_RULE_PATH, cwd=tmp_path)

    (potts_dir / "not_designer.py").write_text(banned)
    control_success, _control_output = run_ast_grep_rule(_RULE_PATH, cwd=tmp_path)
    assert not control_success, "control: the same import outside designer.py must fire"

    assert success, (
        f"Rule should NOT fire on designer.py even with banned imports, but failed.\n"
        f"designer.py is in ignores: configuration.\n"
        f"Rule output:\n{output}"
    )
