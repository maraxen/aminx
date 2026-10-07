"""Library-surface lint: enforce composable-JAX boundary per REC-3.

This module AST-walks the library-side code (tiling/ and types/boundaries.py)
to verify it has NO dependencies on application-side modules (inference/,
model/, host/, sampling/, run/, io/, etc.).

The library-side consists of:
  - All files in src/aminx/tiling/
  - src/aminx/types/boundaries.py (library boundary)
  - src/aminx/types/stages.py is INTENTIONALLY EXCLUDED — it's a bridge
    between library and app and is allowed to reference both.

This lint enforces the composable-JAX architecture: the library (tiling/ +
boundaries.py) is domain-neutral and reusable across projects; the app layer
(inference/, model/, host/) uses the library for its specific use cases.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

LIBRARY_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "aminx"

# Allowed external packages
ALLOWED_EXTERNAL = {
    "jax",
    "equinox",
    "jaxtyping",
    "optax",
    "typing",
    "dataclasses",
    "abc",
    "collections",
    "functools",
    "warnings",
    "__future__",
    "enum",
    "numbers",
    "operator",
}

# Forbidden internal module prefixes (app-side modules)
FORBIDDEN_INTERNAL_PREFIXES = (
    "aminx.inference",
    "aminx.model",
    "aminx.host",
    "aminx.sampling",
    "aminx.run",
    "aminx.io",
    "aminx.score",  # for completeness
)

# Library-side files to lint
LIBRARY_FILES = (
    list((LIBRARY_ROOT / "tiling").rglob("*.py"))
    + [LIBRARY_ROOT / "types" / "boundaries.py"]
)


class ImportViolation(NamedTuple):
    """Records an import violation found in a file."""
    file_path: Path
    line_number: int
    import_kind: str  # "from" or "import"
    module_name: str
    reason: str


def extract_import_violations(source: str) -> list[ImportViolation]:
    """Parse source code and extract all import violations.

    Walks the AST including TYPE_CHECKING blocks per REC-3 mandate.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    list[ImportViolation]
        List of import violations found.
    """
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        # Handle 'from X import Y' statements
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # Relative import or from . import

            # Check if module name starts with forbidden prefix
            if any(node.module.startswith(prefix) for prefix in FORBIDDEN_INTERNAL_PREFIXES):
                violations.append(
                    ImportViolation(
                        file_path=Path("<input>"),
                        line_number=node.lineno,
                        import_kind="from",
                        module_name=node.module,
                        reason=f"App-side module {node.module} imported; library must not depend on app",
                    )
                )

        # Handle 'import X' statements
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                # Check if any forbidden prefix matches
                for forbidden in FORBIDDEN_INTERNAL_PREFIXES:
                    if name == forbidden or name.startswith(forbidden + "."):
                        violations.append(
                            ImportViolation(
                                file_path=Path("<input>"),
                                line_number=node.lineno,
                                import_kind="import",
                                module_name=name,
                                reason=f"App-side module {name} imported; library must not depend on app",
                            )
                        )
                        break

    return violations


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(LIBRARY_ROOT)))
def test_no_app_imports_in_library(path: Path):
    """Assert library-side file has no app-side imports (including TYPE_CHECKING).

    Per REC-3: enforce that library files (tiling/ + boundaries.py) do not
    import from app-side modules (inference/, model/, host/, sampling/, run/, io/).
    """
    source = path.read_text()
    violations = extract_import_violations(source)

    # Prepare a nice error message if violations found
    if violations:
        msg_lines = [f"Found {len(violations)} import violation(s) in {path.relative_to(LIBRARY_ROOT)}:"]
        for v in violations:
            msg_lines.append(f"  Line {v.line_number}: {v.import_kind} {v.module_name}")
            msg_lines.append(f"    {v.reason}")
        pytest.fail("\n".join(msg_lines))


def test_negative_control_catches_violation():
    """Explicit negative test: verify lint catches deliberately-bad imports.

    Per REC-3 mandate: prove the lint actually works by feeding it a
    deliberately-bad source and asserting it IS detected as a violation.
    """
    bad_src = (
        "from aminx.inference.encode import make_encode_fn\n"
        "import aminx.host.plan\n"
        "from aminx.model import SomeClass\n"
    )

    violations = extract_import_violations(bad_src)

    # We expect exactly 3 violations
    assert len(violations) == 3, f"Expected 3 violations, got {len(violations)}"

    # Verify they match the bad imports
    module_names = {v.module_name for v in violations}
    expected_modules = {
        "aminx.inference.encode",
        "aminx.host.plan",
        "aminx.model",
    }
    assert module_names == expected_modules, (
        f"Expected modules {expected_modules}, got {module_names}"
    )


def test_negative_control_allows_good_imports():
    """Verify lint allows good (external and library-side) imports."""
    good_src = (
        "import jax\n"
        "from jax import vmap\n"
        "import equinox as eqx\n"
        "from jaxtyping import PRNGKeyArray\n"
        "import functools\n"
        "from typing import Any\n"
        "from aminx.tiling.strategy import Vmap\n"
        "from aminx.types.boundaries import AxisBoundary\n"
    )

    violations = extract_import_violations(good_src)
    assert len(violations) == 0, f"Expected no violations, got {len(violations)}: {violations}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
