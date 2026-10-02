# ruff: noqa: S101
"""Every family Options field must be read somewhere outside its own dataclass.

A field declared on ``PottsMPNNOptions`` or ``LaserOptions`` and read nowhere else
is accepted by the public surface and forwarded to nothing: a caller setting it
gets default behaviour and no error. Nine such fields exist today (aminx debt
2435), frozen in ``options_plumbing_allowlist.toml``.

The guard is the same shape as the x64-leak guard: lock in the measured state so a
tenth cannot appear unnoticed, and make the allowlist the unit of work so paying
the debt is a deletion. It fails in BOTH directions -- a new inert field, and an
allowlist row that is no longer inert -- so the list cannot rot into a permanent
excuse.

Reads are collected by an ast walk over exact identifier tokens, never a substring
search. That is load-bearing rather than fastidious: ``chi_temp`` is implemented
correctly in ``model/laser/`` under the parameter name ``chi_temperature``, and a
substring search credits the rename as a read, hiding the one case where the
plumbing stops at a rename (see the allowlist note).
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import fields
from pathlib import Path

from aminx.run.options import LaserOptions, PottsMPNNOptions

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "aminx"
_OPTIONS_FILE = (_SRC / "run" / "options.py").resolve()
_ALLOWLIST = Path(__file__).with_name("options_plumbing_allowlist.toml")
_CLASSES = {"LaserOptions": LaserOptions, "PottsMPNNOptions": PottsMPNNOptions}


def _names(path: Path) -> set[str]:
    """Identifiers, attribute names, keyword names and parameter names in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - a parse failure
        # would silently under-count reads and manufacture false positives, so it
        # is a hard error rather than a skipped file.
        msg = f"unparsable source in the tree under test: {path}"
        raise AssertionError(msg) from None
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


def _read_outside_options() -> set[str]:
    names: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == _OPTIONS_FILE:
            continue
        names |= _names(path)
    return names


def _allowlist() -> dict[str, dict[str, str]]:
    rows = tomllib.loads(_ALLOWLIST.read_text(encoding="utf-8")).get("entries", [])
    assert isinstance(rows, list), "options_plumbing_allowlist.toml entries must be a list"
    by_field: dict[str, dict[str, str]] = {}
    for row in rows:
        field = str(row["field"])
        assert field not in by_field, f"duplicate allowlist row for {field}"
        assert row["reason"] == "not_plumbed", (
            f"{field}: reason must be 'not_plumbed', got {row['reason']!r}. A field "
            f"that should never be read outside its dataclass does not belong in an "
            f"Options class."
        )
        assert row["options_class"] in _CLASSES, (
            f"{field}: unknown options_class {row['options_class']!r}"
        )
        by_field[field] = {str(k): str(v) for k, v in row.items()}
    return by_field


def _inert() -> dict[str, str]:
    """Field -> its Options class, for every field read nowhere outside options.py."""
    outside = _read_outside_options()
    return {
        f.name: name
        for name, cls in _CLASSES.items()
        for f in fields(cls)
        if f.name not in outside
    }


def test_allowlist_rows_name_real_fields() -> None:
    """An allowlist row for a field that no longer exists is stale."""
    allowed = _allowlist()
    declared = {
        f.name: name for name, cls in _CLASSES.items() for f in fields(cls)
    }
    unknown = {
        field: row["options_class"]
        for field, row in allowed.items()
        if declared.get(field) != row["options_class"]
    }
    assert not unknown, (
        f"{len(unknown)} allowlist rows do not match a declared field on the class "
        f"they name: {unknown}. Delete the row, or correct its options_class."
    )


def test_no_new_inert_option_fields() -> None:
    """No family Options field may go inert without being declared."""
    inert = _inert()
    allowed = _allowlist()
    new = {field: cls for field, cls in inert.items() if field not in allowed}
    assert not new, (
        f"{len(new)} family Options fields are read nowhere outside "
        f"src/aminx/run/options.py and are not in options_plumbing_allowlist.toml: "
        f"{new}. Either plumb the field through its family driver, or -- if this is "
        f"knowingly unfinished -- add a row with reason='not_plumbed' and a note "
        f"saying what consumes it once wired. See aminx debt 2435."
    )


def test_allowlist_has_no_fixed_entries() -> None:
    """A row whose field is now plumbed must be deleted, so the debt visibly shrinks."""
    inert = _inert()
    allowed = _allowlist()
    fixed = sorted(field for field in allowed if field not in inert)
    assert not fixed, (
        f"{len(fixed)} allowlist rows are no longer inert -- the field is now read "
        f"outside options.py: {fixed}. Delete those rows from "
        f"options_plumbing_allowlist.toml; the list is the remaining work, so it "
        f"must not keep paid-off entries."
    )
