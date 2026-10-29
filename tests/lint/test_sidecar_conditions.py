"""Sidecar condition lint (debt #2586): every [outcomes.*].condition must be
a machine-evaluable bathos expression over [result_schema] keys.

A prose condition (e.g. "speedup >= 1.5 at K=16") can never grade: bathos
evaluates the expression against the emitted result, so every bare
identifier must be a result_schema key or one of the keywords below.

Stdlib only (tomllib + a minimal tokenizer). Run as a script to list the
offenders across the repo:  python3 tests/lint/test_sidecar_conditions.py
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).with_name("sidecar_conditions_allowlist.toml")

# bathos outcome conditions are DuckDB SQL fragments, so the grammar is the
# SQL subset actually used: boolean/SQL keywords, comparisons, arithmetic,
# IN lists, and a fixed set of scalar functions called as name(...).
_KEYWORDS = frozenset({"AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE"})
_FUNCTIONS = frozenset({"min", "max", "count", "abs", "sum", "avg", "len", "round", "starts_with"})

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<str>'[^']*'|"[^"]*")
  | (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<op><=|>=|==|!=|<>|<|>|=|\+|-|\*|/|%)
  | (?P<punct>[()\[\],])
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


def tokenize(expr: str) -> list[tuple[str, str]]:
    """Split a condition into (kind, text) tokens. Raises ValueError on
    any character that is not part of the expression grammar."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if m is None:
            raise ValueError(f"untokenizable character {expr[pos]!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind != "ws":
            tokens.append((kind, m.group()))
    return tokens


def _next_is_open(tokens: list[tuple[str, str]], i: int) -> bool:
    return i + 1 < len(tokens) and tokens[i + 1] == ("punct", "(")


def condition_offenders(condition: str, schema_keys: set[str]) -> list[str]:
    """Return the reasons a condition is not machine-evaluable (empty = ok)."""
    try:
        tokens = tokenize(condition)
    except ValueError as exc:
        return [str(exc)]
    problems: list[str] = []
    depth = 0
    for i, (kind, text) in enumerate(tokens):
        if kind == "punct" and text in "([":
            depth += 1
        elif kind == "punct" and text in ")]":
            depth -= 1
            if depth < 0:
                problems.append("unbalanced closing bracket")
                break
        elif kind == "ident" and text.upper() in _KEYWORDS:
            continue
        elif kind == "ident" and text.lower() in _FUNCTIONS and _next_is_open(tokens, i):
            continue
        elif kind == "ident" and text not in schema_keys:
            problems.append(f"bare identifier {text!r} is not a result_schema key")
    if depth != 0:
        problems.append("unbalanced brackets")
    if not tokens:
        problems.append("empty condition")
    return problems


def sidecar_offenders(sidecar: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(outcome_name, condition, reason) for each non-evaluable condition."""
    schema_keys = set(sidecar.get("result_schema", {}) or {})
    found: list[tuple[str, str, str]] = []
    for name, outcome in (sidecar.get("outcomes", {}) or {}).items():
        if not isinstance(outcome, dict) or "condition" not in outcome:
            continue
        cond = outcome["condition"]
        for reason in condition_offenders(str(cond), schema_keys):
            found.append((name, str(cond), reason))
    return found


def scan_repo(root: Path = _REPO) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """Scan every *.bth.toml under root.

    Returns (condition_offenders as (path, outcome, condition, reason),
    unparseable sidecars as (path, error)).
    """
    offenders: list[tuple[str, str, str, str]] = []
    unparseable: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.bth.toml")):
        if ".git" in path.parts:
            continue
        rel = str(path.relative_to(root))
        try:
            sidecar = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            unparseable.append((rel, str(exc)))
            continue
        for name, cond, reason in sidecar_offenders(sidecar):
            offenders.append((rel, name, cond, reason))
    return offenders, unparseable


# --- tests -----------------------------------------------------------------

_VALID_SIDECAR = {
    "result_schema": {"n_bad": "int", "ok": "bool"},
    "outcomes": {
        "pass": {"condition": "n_bad = 0 AND ok = true"},
        "fail": {"condition": "NOT (n_bad = 0) OR ok = false"},
    },
}

_PROSE_SIDECAR = {
    "result_schema": {"speedup": "float", "K": "int"},
    "outcomes": {
        "pass": {"condition": "speedup >= 1.5 at K=16"},
    },
}


def test_negative_control_prose_condition_fails() -> None:
    offenders = sidecar_offenders(_PROSE_SIDECAR)
    assert offenders, "prose condition must be flagged"
    assert any("'at'" in reason for _, _, reason in offenders)


def test_positive_control_valid_condition_passes() -> None:
    assert sidecar_offenders(_VALID_SIDECAR) == []


def test_unknown_identifier_fails() -> None:
    sidecar = {"result_schema": {"a": "bool"}, "outcomes": {"x": {"condition": "a AND b"}}}
    assert [r for _, _, r in sidecar_offenders(sidecar)] == [
        "bare identifier 'b' is not a result_schema key"
    ]


def _load_allowlist() -> tuple[dict[str, str], dict[str, str]]:
    data = tomllib.loads(_ALLOWLIST.read_text(encoding="utf-8"))
    return dict(data.get("conditions", {})), dict(data.get("unparseable", {}))


def test_repo_sidecar_conditions_are_machine_evaluable() -> None:
    """Every sidecar outside the allowlist grades; allowlisted entries are
    known debt (#2586) and must be pruned as soon as they are fixed."""
    offenders, unparseable = scan_repo()
    allow_cond, allow_unparse = _load_allowlist()
    bad_paths = {p for p, _, _, _ in offenders}
    new_bad = sorted(
        f"{p} [{o}] {c!r}: {r}" for p, o, c, r in offenders if p not in allow_cond
    )
    stale_cond = sorted(set(allow_cond) - bad_paths)
    new_unparse = sorted(f"{p}: {e}" for p, e in unparseable if p not in allow_unparse)
    stale_unparse = sorted(set(allow_unparse) - {p for p, _ in unparseable})
    problems = (
        [f"NEW prose/unevaluable condition: {x}" for x in new_bad]
        + [f"NEW unparseable sidecar: {x}" for x in new_unparse]
        + [f"STALE allowlist entry (condition now clean): {p}" for p in stale_cond]
        + [f"STALE allowlist entry (now parses): {p}" for p in stale_unparse]
    )
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    off, bad = scan_repo()
    for p, o, c, r in off:
        print(f"OFFENDER {p} [{o}] {c!r}: {r}")
    for p, e in bad:
        print(f"UNPARSEABLE {p}: {e}")
    print(f"{len(off)} offending condition(s), {len(bad)} unparseable sidecar(s)")
