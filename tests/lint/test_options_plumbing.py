# ruff: noqa: S101
"""Every family Options field must be read somewhere outside its own dataclass.

A field declared on ``PottsMPNNOptions`` or ``LaserOptions`` and read nowhere else
is accepted by the public surface and forwarded to nothing: a caller setting it
gets default behaviour and no error. Those fields are frozen in
``options_plumbing_allowlist.toml`` (aminx debt 2435).

The guard is the same shape as the x64-leak guard: lock in the measured state so a
new one cannot appear unnoticed, and make the allowlist the unit of work so paying
the debt is a deletion. It fails in BOTH directions -- a new inert field, and an
allowlist row that is no longer inert -- so the list cannot rot into a permanent
excuse.

A READ IS COUNTED ONLY OFF AN OPTIONS OBJECT, which is the whole difficulty. An
earlier revision collected every identifier token in the tree and matched field
names against it. Matching exact tokens rather than substrings is necessary --
``chi_temp`` is implemented in ``model/laser/`` under the name ``chi_temperature``
and a substring search credits the rename as a read -- but it is not sufficient,
because a token matches wherever it appears, on any object or none. Measured
261002, that hid nine inert fields behind same-named parameters and locals, and
reported one plumbed field as inert because its read goes through ``getattr`` with
a string constant. See ``_fields_read`` for both cases.
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


#: Variable names that hold a family Options object. A read counts only when it
#: comes off one of these, so an unrelated identifier that merely shares a field's
#: name cannot be mistaken for plumbing. Only the last dotted segment is compared,
#: so ``self._options`` matches via ``_options`` and ``spec.laser`` via ``laser``.
#: Adding a carrier is a deliberate edit; a read through some other variable shows
#: up as a false "inert", which fails loudly rather than passing silently.
_CARRIERS = frozenset({"options", "_options", "opts", "laser", "potts_mpnn"})


def _carrier(node: ast.expr) -> bool:
    """Is this expression one of the known Options carriers?"""
    if isinstance(node, ast.Name):
        return node.id in _CARRIERS
    if isinstance(node, ast.Attribute):
        return node.attr in _CARRIERS
    return False


def _fields_read(path: Path) -> set[str]:
    """Field names read OFF AN OPTIONS OBJECT in one file.

    Two shapes count, and nothing else:

    * ``<carrier>.<field>`` -- the ordinary read.
    * ``getattr(<carrier>, "<field>", ...)`` -- used where the caller may have
      passed a different Options class (e.g. ``laser_mpnn/driver.py:256``).

    Earlier this collected every ``Name``/``Attribute``/``keyword``/``arg`` token
    in the file, which matched a field's name regardless of what object it
    belonged to. That was wrong in BOTH directions and both were measured on
    261002:

    * ``tied_beta`` was credited as plumbed although ``options.tied_beta`` appears
      nowhere in ``src/`` -- the name belongs to a per-position array built by
      ``featurize._tied_groups`` and read off ``features``/``padded``/``ready``.
      Eight LaserOptions fields were hidden the same way, by parameters of the
      same name in ``model/laser/`` that nothing passes the Options field to.
      That is the ``chi_temp`` failure the allowlist already describes, repeated.
    * ``strict_load`` was reported inert although it IS read, because
      ``getattr(options, "strict_load", True)`` carries the name as a string
      constant, which no token kind above matches.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - a parse failure
        # would silently under-count reads and manufacture false positives, so it
        # is a hard error rather than a skipped file.
        msg = f"unparsable source in the tree under test: {path}"
        raise AssertionError(msg) from None
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _carrier(node.value):
            out.add(node.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= 2
                and _carrier(node.args[0])
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                out.add(node.args[1].value)
    return out


def _read_outside_options() -> set[str]:
    names: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == _OPTIONS_FILE:
            continue
        names |= _fields_read(path)
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
