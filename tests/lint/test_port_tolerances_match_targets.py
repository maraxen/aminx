"""The tolerance a port test ASSERTS must be the one its target TOML REPORTS.

WHY THIS EXISTS (found 261003, debt #2445). Each ``tests/port/test_<wave>.py``
asserts against module constants ``_RTOL`` / ``_ATOL``, while
``tests/port/conftest.py`` separately reads ``[parity] tolerance_policy_f64/_f32``
from ``targets/<wave>.toml`` and stamps THAT string into every emitted tier
verdict (and hashes it into the verdict id). Nothing tied the two together. So:

* amending only the TOML changes no assertion -- the #2445 amendment to
  ``laser_decode_step.toml`` was about to land that way, and a CPU run of
  ``test_tier_3_f32`` still reported ``atol=1e-07``;
* and whenever they differ, every emitted verdict claims a band that is not the
  band the test applied.

This reads both statically (AST for the constants, tomllib for the policy), so
it costs nothing and runs in the ordinary suite rather than only in the gate.

A BAND HAS A FORM AS WELL AS A VALUE (added 261006 for laser_score option 3).
``laser_score``'s f32 tier is no longer an element-wise ``rtol``/``atol``
envelope; it is scale-relative, ``max|got-ref| <= tol * max|ref|``, carried by a
``_SCALE_REL`` constant and reported as ``scale_rel=4e-4``. So this guard
compares forms first and values second, and treats three further things as
failures: a policy whose key set matches no known form (which would otherwise
compare nothing and pass), a test still carrying a constant from the *other*
form for the same precision (a half-finished migration leaves two bands on one
tier), and of course a value that drifted.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

_PORT = Path(__file__).resolve().parents[1] / "port"
_TARGETS = _PORT / "targets"

#: The waves this guard compares, BY NAME.
#:
#: This replaced a bare ``_MIN_COVERED = 3`` count (261006). A count is not
#: coverage: five waves satisfied "at least three", so two could stop being
#: checked -- by losing their band dicts or their ``port_wave`` marker -- and
#: the guard would still certify itself non-vacuous while quietly comparing
#: less. Verified by probe: dropping the marker from test_laser_encoder.py took
#: the wave out of this guard entirely and the count check still passed.
#:
#: Equality, not a floor, and in both directions on purpose: a wave leaving is
#: a regression, and a wave arriving should be recorded here so the set stays a
#: true statement about what is checked rather than a stale lower bound.
_EXPECTED_COVERED = frozenset({
  "laser_decode_step",
  "laser_encoder",
  "laser_layers",
  "laser_rotamers",
  "laser_score",
})

#: Every module constant that can carry a band. Order is irrelevant; membership
#: is what makes a stale leftover from another form detectable.
_ALL_CONSTS = ("_RTOL", "_ATOL", "_SCALE_REL")

#: Sentinel for a band the module defines but does not define LITERALLY, so it
#: cannot be read statically. Compared by identity, never by value: an empty
#: dict would be indistinguishable from "the constant is absent", and the two
#: want different messages.
_UNREADABLE: dict[str, float] = {}

#: Policy key-set -> the (constant, policy key) pairs that form requires. A
#: policy whose keys match no entry is a failure, not a pass: an unrecognised
#: form would compare nothing at all.
_FORMS: dict[frozenset[str], tuple[tuple[str, str], ...]] = {
  frozenset({"rtol", "atol"}): (("_RTOL", "rtol"), ("_ATOL", "atol")),
  frozenset({"scale_rel"}): (("_SCALE_REL", "scale_rel"),),
}


def _parse_policy(policy: str) -> dict[str, float]:
  """``"rtol=1e-4,atol=1e-7"`` -> ``{"rtol": 1e-4, "atol": 1e-7}``."""
  out: dict[str, float] = {}
  for part in policy.split(","):
    key, _, value = part.strip().partition("=")
    out[key.strip()] = float(value)
  return out


def _module_facts(path: Path) -> tuple[str | None, dict[str, dict[str, float]]]:
  """The ``port_wave`` marker argument and the literal ``_RTOL``/``_ATOL`` dicts."""
  wave: str | None = None
  consts: dict[str, dict[str, float]] = {}
  for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "port_wave"
      and node.args
      and isinstance(node.args[0], ast.Constant)
    ):
      wave = str(node.args[0].value)
  for node in ast.parse(path.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target, expr = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
      target, expr = node.target, node.value
    else:
      continue
    if isinstance(target, ast.Name) and target.id in set(_ALL_CONSTS):
      # A band that is not a literal cannot be read statically. Do NOT let that
      # raise: this runs at import time to build the parametrize list, so a
      # ValueError here fails COLLECTION of the whole file and takes every test
      # in it down with the one module it could not parse (found 261006 by
      # probe, on `_RTOL = dict(_BAND["rtol"])`). Record it as unreadable
      # instead, and let _mismatches refuse it as a normal, legible failure --
      # unreadable must not become a quiet way to skip the comparison.
      try:
        value = ast.literal_eval(expr)
      except ValueError:
        consts[target.id] = _UNREADABLE
        continue
      if isinstance(value, dict):
        consts[target.id] = {str(k): float(v) for k, v in value.items()}
  return wave, consts


def _mismatches(
  consts: dict[str, dict[str, float]], parity: Mapping[str, object],
) -> list[str]:
  """Every precision where the asserted band differs from the reported one."""
  problems: list[str] = []
  for precision in ("f64", "f32"):
    policy = _parse_policy(str(parity[f"tolerance_policy_{precision}"]))
    form = _FORMS.get(frozenset(policy))
    if form is None:
      problems.append(
        f"{precision}: TOML policy {sorted(policy)} matches no known band form "
        f"{[sorted(keys) for keys in _FORMS]}",
      )
      continue
    for name, key in form:
      if consts.get(name) is _UNREADABLE:
        problems.append(
          f"{precision}: {name} is not a literal, so this guard cannot confirm "
          f"the asserted band matches the reported {key}. Keep the band literal.",
        )
        continue
      asserted = consts.get(name, {}).get(precision)
      reported = policy.get(key)
      if asserted != reported:
        problems.append(
          f"{precision} {key}: test asserts {asserted!r}, TOML reports {reported!r}",
        )
    # A constant belonging to a DIFFERENT form that still defines this precision
    # is a half-finished migration: the tier would carry two bands at once, and
    # which one the test applies is then an accident of control flow.
    named = {name for name, _ in form}
    for name in _ALL_CONSTS:
      if name not in named and consts.get(name, {}).get(precision) is not None:
        problems.append(
          f"{precision}: TOML declares the {sorted(policy)} form but the test "
          f"still defines {name}[{precision!r}]",
        )
  return problems


def test_mismatch_detector_fires() -> None:
  """Negative control: a known drift must be reported, a match must not."""
  parity = {"tolerance_policy_f64": "rtol=1e-8,atol=1e-11", "tolerance_policy_f32": "rtol=1e-4,atol=2e-3"}
  drifted = {"_RTOL": {"f64": 1e-8, "f32": 1e-4}, "_ATOL": {"f64": 1e-11, "f32": 1e-7}}
  agreed = {"_RTOL": {"f64": 1e-8, "f32": 1e-4}, "_ATOL": {"f64": 1e-11, "f32": 2e-3}}
  assert _mismatches(drifted, parity) == ["f32 atol: test asserts 1e-07, TOML reports 0.002"]
  assert _mismatches(agreed, parity) == []

  # --- the scale-relative form, and the three ways it can go wrong ---
  scale = {
    "tolerance_policy_f64": "rtol=1e-8,atol=1e-11",
    "tolerance_policy_f32": "scale_rel=4e-4",
  }
  ok = {"_RTOL": {"f64": 1e-8}, "_ATOL": {"f64": 1e-11}, "_SCALE_REL": {"f32": 4e-4}}
  assert _mismatches(ok, scale) == []

  # 1. the value drifted
  assert _mismatches({**ok, "_SCALE_REL": {"f32": 1e-3}}, scale) == [
    "f32 scale_rel: test asserts 0.001, TOML reports 0.0004",
  ]

  # 2. half-finished migration: an element-wise band left beside the new one.
  #    This is the case a form-blind guard would wave through.
  stale = {**ok, "_RTOL": {"f64": 1e-8, "f32": 1e-4}}
  assert _mismatches(stale, scale) == [
    "f32: TOML declares the ['scale_rel'] form but the test still defines _RTOL['f32']",
  ]

  # 3. a band the module defines but not literally: refused, not skipped, and
  #    above all not raised during collection.
  assert _mismatches({**ok, "_SCALE_REL": _UNREADABLE}, scale) == [
    "f32: _SCALE_REL is not a literal, so this guard cannot confirm the asserted "
    "band matches the reported scale_rel. Keep the band literal.",
  ]

  # 4. a form nobody has taught the guard must NOT pass silently
  unknown = {
    "tolerance_policy_f64": "rtol=1e-8,atol=1e-11",
    "tolerance_policy_f32": "ulps=4",
  }
  assert _mismatches(ok, unknown) != []


def _covered() -> list[tuple[str, Path, dict[str, dict[str, float]]]]:
  rows = []
  for path in sorted(_PORT.glob("test_*.py")):
    wave, consts = _module_facts(path)
    if wave is None or not consts:
      continue
    rows.append((wave, path, consts))
  return rows


def test_guard_covers_exactly_the_waves_it_claims() -> None:
  """Anti-vacuity by NAME. See ``_EXPECTED_COVERED`` for why not a count."""
  covered = {wave for wave, _, _ in _covered()}
  missing = sorted(_EXPECTED_COVERED - covered)
  arrived = sorted(covered - _EXPECTED_COVERED)
  assert not missing, (
    f"these waves were being checked and no longer are: {missing}. A wave leaves "
    f"this guard by losing its literal _RTOL/_ATOL dicts or its port_wave marker, "
    f"and either way its emitted verdict stops being tied to any assertion. "
    f"Still covered: {sorted(covered)}"
  )
  assert not arrived, (
    f"these waves newly expose literal band dicts and are now checked: {arrived}. "
    f"Add them to _EXPECTED_COVERED so it keeps stating what is really covered."
  )


@pytest.mark.parametrize(("wave", "path", "consts"), _covered(), ids=lambda v: v if isinstance(v, str) else "")
def test_asserted_tolerance_matches_reported(
  wave: str, path: Path, consts: dict[str, dict[str, float]],
) -> None:
  target = _TARGETS / f"{wave}.toml"
  assert target.exists(), f"{path.name} declares port_wave({wave!r}) but {target} is missing"
  parity = tomllib.loads(target.read_text(encoding="utf-8"))["parity"]
  problems = _mismatches(consts, parity)
  assert not problems, (
    f"{path.name} vs targets/{wave}.toml -- the verdict it emits would claim a "
    f"band it did not apply:\n  " + "\n  ".join(problems)
  )
