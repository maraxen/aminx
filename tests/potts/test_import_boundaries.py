"""Import boundary enforcement for aminx.potts modules.

Tests verify that aminx.potts.{model,poe,sampling} do NOT import from
forbidden modules (aminx.inference.decode, aminx.host.plan, aminx.types.stages,
aminx.inference.logits). The designer module is exempt from this restriction.

Reference: ADR 260605_potts-parallel-not-stageset.md
"""

import ast
from pathlib import Path
from typing import List, Set

import pytest


# Module import definitions
FORBIDDEN_IMPORTS = {
    "aminx.inference.decode",
    "aminx.host.plan",
    "aminx.types.stages",
    "aminx.inference.logits",
}

# Files that must NOT import from FORBIDDEN_IMPORTS
GUARDED_FILES = {
    "src/aminx/potts/model.py",
    "src/aminx/potts/poe.py",
    "src/aminx/potts/sampling.py",
}

# Files exempt from the restriction (may import anything)
EXEMPT_FILES = {
    "src/aminx/potts/designer.py",
}


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract all import names from a module."""

    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Handle: import a.b.c"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from a.b.c import x, y, z"""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


def extract_imports(filepath: Path) -> Set[str]:
    """Extract all module imports from a Python file via AST."""
    try:
        code = filepath.read_text()
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        pytest.fail(f"Failed to parse {filepath}: {e}")


def find_forbidden_imports(imports: Set[str], forbidden: Set[str]) -> Set[str]:
    """Find which imported modules are in the forbidden set."""
    violations = set()
    for imp in imports:
        # Check exact match and prefixes (e.g., "aminx.inference.decode" matches
        # both "aminx.inference.decode" and "aminx.inference.decode.submodule")
        for forbidden_module in forbidden:
            if imp == forbidden_module or imp.startswith(forbidden_module + "."):
                violations.add(imp)
                break
    return violations


def get_guarded_file_path(relative_path: str) -> Path:
    """Resolve a relative path from project root."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / relative_path


@pytest.mark.potts
class TestImportBoundaries:
    """Verify that potts modules respect import boundaries."""

    @pytest.mark.parametrize(
        "guarded_file",
        sorted(GUARDED_FILES),
        ids=lambda x: x.split("/")[-1],
    )
    def test_guarded_module_no_forbidden_imports(self, guarded_file: str) -> None:
        """Guarded modules must not import from forbidden set.

        Args:
            guarded_file: Relative path to a guarded module.

        Raises:
            AssertionError: If any forbidden import is found, with message
                citing the ADR.
        """
        filepath = get_guarded_file_path(guarded_file)

        # Skip if file doesn't exist yet (allows test to pass before all modules are created)
        if not filepath.exists():
            pytest.skip(f"Module not yet created: {guarded_file}")

        imports = extract_imports(filepath)
        violations = find_forbidden_imports(imports, FORBIDDEN_IMPORTS)

        assert not violations, (
            f"Module {guarded_file} violates import boundary (ADR 260605_potts-parallel-not-stageset.md):\n"
            f"  Forbidden imports found: {', '.join(sorted(violations))}\n"
            f"  Guarded modules must not import from: {', '.join(sorted(FORBIDDEN_IMPORTS))}"
        )

    def test_exempt_module_not_guarded(self) -> None:
        """Verify that exempt modules are not in the guarded list."""
        overlap = GUARDED_FILES & EXEMPT_FILES
        assert not overlap, (
            f"Exempt modules should not be guarded: {overlap}\n"
            f"If a module needs exemption, remove it from GUARDED_FILES."
        )

    def test_all_forbidden_modules_listed(self) -> None:
        """Verify forbidden modules are the expected set."""
        # This test ensures that future changes to the ADR are reflected here.
        expected = {
            "aminx.inference.decode",
            "aminx.host.plan",
            "aminx.types.stages",
            "aminx.inference.logits",
        }
        assert (
            FORBIDDEN_IMPORTS == expected
        ), f"Forbidden imports list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"

    def test_guarded_files_aligned_with_adr(self) -> None:
        """Verify guarded modules match ADR specification."""
        expected = {
            "src/aminx/potts/model.py",
            "src/aminx/potts/poe.py",
            "src/aminx/potts/sampling.py",
        }
        assert (
            GUARDED_FILES == expected
        ), f"Guarded files list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"

    def test_exempt_files_aligned_with_adr(self) -> None:
        """Verify exempt modules match ADR specification."""
        expected = {
            "src/aminx/potts/designer.py",
        }
        assert (
            EXEMPT_FILES == expected
        ), f"Exempt files list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"


@pytest.mark.potts
class TestPottsBoundary:
    """Verify aminx.potts respects domain boundaries.

    Potts is a pure mathematical layer (Potts model, TRW inference).
    Pure math modules (model, poe, sampling) must not import from protein-domain
    modules (aminx.model, aminx.host, aminx.run, aminx.cli). Integration layers
    (runner, cli, designer) are exempt as they sit at the boundary with protein code.

    Field names test: pure math modules must not reference protein-specific field
    names (atom_37, atom_37_mask, residue_index, chain_index).
    """

    # Pure math layers: must not import protein-domain modules
    POTTS_PURE_MATH_FILES = {
        "src/aminx/potts/model.py",
        "src/aminx/potts/poe.py",
        "src/aminx/potts/sampling.py",
    }

    POTTS_FORBIDDEN_IMPORTS = {
        "aminx.model",
        "aminx.host",
        "aminx.run",
        "aminx.cli",
    }

    POTTS_FORBIDDEN_FIELD_NAMES = {
        "atom_37",
        "atom_37_mask",
        "residue_index",
        "chain_index",
    }

    def test_potts_no_protein_domain_imports(self) -> None:
        """Verify pure math modules do not import from protein-domain packages.

        Only checks: model.py, poe.py, sampling.py (the mathematical core).
        Integration layers (runner, cli, designer) are exempt.

        Potts math layer must not import from:
        - aminx.model: protein structure/sequence models
        - aminx.host: host system integration
        - aminx.run: execution harness
        - aminx.cli: command-line interface

        Raises:
            AssertionError: If any forbidden import is found in pure math modules.
        """
        all_violations = {}

        for guarded_file in self.POTTS_PURE_MATH_FILES:
            project_root = Path(__file__).parent.parent.parent
            filepath = project_root / guarded_file

            if not filepath.exists():
                continue

            imports = extract_imports(filepath)
            violations = set()

            for imp in imports:
                # Check if import starts with any forbidden module
                for forbidden in self.POTTS_FORBIDDEN_IMPORTS:
                    if imp == forbidden or imp.startswith(forbidden + "."):
                        violations.add(imp)
                        break

            if violations:
                all_violations[filepath.name] = violations

        assert not all_violations, (
            f"Potts pure math modules import from protein-domain packages:\n"
            + "\n".join(
                f"  {filename}: {', '.join(sorted(viol))}"
                for filename, viol in sorted(all_violations.items())
            )
            + f"\nPure math modules (model, poe, sampling) must not couple to protein-domain code.\n"
            f"Integration (runner, cli, designer) may import from aminx.model as they sit at the boundary."
        )

    def test_potts_no_protein_field_names(self) -> None:
        """Verify potts source code does not reference protein field names in code.

        Field names like atom_37, residue_index, chain_index are protein-specific
        metadata from the protein structure domain. This test checks that pure math
        modules (model.py, poe.py, sampling.py) do not reference these in their
        implementation logic.

        Excludes:
        - Pure comment lines (lines starting with #)
        - Docstrings and function signatures (these are documentation, not logic)
        - Usage in integration layers (runner, cli, designer)

        Raises:
            AssertionError: If any field name is found in implementation logic.
        """
        # Only check pure math modules; integration layers are allowed these field names
        pure_math_files = {
            "model.py",
            "poe.py",
            "sampling.py",
        }

        potts_root = Path(__file__).parent.parent.parent / "src" / "aminx" / "potts"

        violations = {}
        for py_file in sorted(potts_root.glob("*.py")):
            if py_file.name not in pure_math_files:
                continue

            code = py_file.read_text()
            lines = code.split("\n")

            # Track docstring state
            in_docstring = False
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()

                # Skip empty lines and pure comment lines
                if not stripped or stripped.startswith("#"):
                    continue

                # Track docstring state (triple quotes)
                if '"""' in stripped or "'''" in stripped:
                    # Count quotes to handle opening/closing
                    triple_double = stripped.count('"""')
                    triple_single = stripped.count("'''")
                    if triple_double % 2 == 1:
                        in_docstring = not in_docstring
                    if triple_single % 2 == 1:
                        in_docstring = not in_docstring

                # Skip docstring lines
                if in_docstring:
                    continue

                # Check for forbidden field names (but not in function signatures)
                # Skip lines that are just type hints or parameter definitions
                if "def " in line or ": " in line and "=" not in line:
                    # This is likely a function signature or type hint
                    continue

                for field_name in self.POTTS_FORBIDDEN_FIELD_NAMES:
                    if field_name in line:
                        rel_path = py_file.relative_to(potts_root.parent)
                        if rel_path not in violations:
                            violations[rel_path] = []
                        violations[rel_path].append((line_num, line.rstrip()))

        assert not violations, (
            f"potts pure math modules reference protein-specific field names:\n"
            + "\n".join(
                f"  {filepath}:\n"
                + "\n".join(
                    f"    Line {line_num}: {line_text}"
                    for line_num, line_text in file_violations
                )
                for filepath, file_violations in sorted(violations.items())
            )
            + f"\nPure math modules (model, poe, sampling) must not reference protein field names in logic.\n"
            f"Integration layers (runner, cli, designer) are exempt.\n"
            f"Forbidden: {', '.join(sorted(self.POTTS_FORBIDDEN_FIELD_NAMES))}"
        )


# xtrax internal module boundary enforcement
XTRAX_BANNED_INTERNAL = {
    "xtrax.stages.bundle",
    "xtrax.stages.protocols",
    "xtrax.tiling.bucket",
    "xtrax.tiling.dedup",
    "xtrax.tiling.dispatch",
    "xtrax.tiling.plan",
    "xtrax.tiling.strategy",
    "xtrax.engine.engine",
    "xtrax.engine.io",
    "xtrax.safety.manager",
}

XTRAX_FIELD_NAMES = {
    "atom_37",
    "residue_index",
    "tie_group_map",
}


@pytest.mark.potts
class TestXtraxBoundaries:
    """Verify xtrax import boundary enforcement and aminx field separation."""

    def test_aminx_potts_no_xtrax_internals(self) -> None:
        """Verify aminx.potts modules only import from xtrax public APIs.

        Scans all .py files in src/aminx/potts/ and verifies no import
        matches the internal xtrax modules (implementation files that
        should only be accessed via public __init__ re-exports).

        Raises:
            AssertionError: If any internal xtrax import is found.
        """
        potts_root = Path(__file__).parent.parent.parent / "src" / "aminx" / "potts"

        # Collect all violations across all potts modules
        all_violations = {}
        for py_file in potts_root.glob("*.py"):
            imports = extract_imports(py_file)
            violations = find_forbidden_imports(imports, XTRAX_BANNED_INTERNAL)
            if violations:
                all_violations[py_file.name] = violations

        assert not all_violations, (
            f"aminx.potts modules import from xtrax internal modules (ADR 260605):\n"
            + "\n".join(
                f"  {filename}: {', '.join(sorted(viol))}"
                for filename, viol in sorted(all_violations.items())
            )
            + f"\nUse public __init__ re-exports instead, e.g., "
            f"`from xtrax.stages import ...` not `from xtrax.stages.bundle import ...`"
        )

    def test_xtrax_no_aminx_field_names(self) -> None:
        """Verify xtrax codebase does not reference aminx-specific field names.

        These field names (atom_37, residue_index, tie_group_map) are specific
        to aminx and should not appear in xtrax implementation. This enforces
        the separation that xtrax operates on generic structures.

        Resolved from the INSTALLED xtrax, not a checkout. A hardcoded
        /home/<user>/projects/xtrax path skipped this check on every other
        machine (titanix among them), and even where it existed it scanned a
        source tree that is not the xtrax aminx actually imports. xtrax is a
        core dependency, so there is nothing to skip for.

        Raises:
            AssertionError: If any forbidden field name is found in xtrax source.
        """
        import xtrax

        xtrax_root = Path(xtrax.__file__).parent

        violations = {}
        for py_file in xtrax_root.rglob("*.py"):
            code = py_file.read_text()
            for field_name in XTRAX_FIELD_NAMES:
                if field_name in code:
                    rel_path = py_file.relative_to(xtrax_root.parent)
                    if rel_path not in violations:
                        violations[rel_path] = []
                    violations[rel_path].append(field_name)

        assert not violations, (
            f"xtrax source contains aminx-specific field names (ADR 260605):\n"
            + "\n".join(
                f"  {filepath}: {', '.join(sorted(set(fields)))}"
                for filepath, fields in sorted(violations.items())
            )
            + f"\nxtrax must not reference aminx-specific structures. "
            f"Remove or parameterize these references."
        )
