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
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

_PORT = Path(__file__).resolve().parents[1] / "port"
_TARGETS = _PORT / "targets"
#: How many waves must actually be compared. A guard that silently matches no
#: file would pass forever; raise this when a wave adopts the constants.
_MIN_COVERED = 3


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
    if isinstance(target, ast.Name) and target.id in {"_RTOL", "_ATOL"}:
      value = ast.literal_eval(expr)
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
    for name, key in (("_RTOL", "rtol"), ("_ATOL", "atol")):
      asserted = consts.get(name, {}).get(precision)
      reported = policy.get(key)
      if asserted != reported:
        problems.append(
          f"{precision} {key}: test asserts {asserted!r}, TOML reports {reported!r}",
        )
  return problems


def test_mismatch_detector_fires() -> None:
  """Negative control: a known drift must be reported, a match must not."""
  parity = {"tolerance_policy_f64": "rtol=1e-8,atol=1e-11", "tolerance_policy_f32": "rtol=1e-4,atol=2e-3"}
  drifted = {"_RTOL": {"f64": 1e-8, "f32": 1e-4}, "_ATOL": {"f64": 1e-11, "f32": 1e-7}}
  agreed = {"_RTOL": {"f64": 1e-8, "f32": 1e-4}, "_ATOL": {"f64": 1e-11, "f32": 2e-3}}
  assert _mismatches(drifted, parity) == ["f32 atol: test asserts 1e-07, TOML reports 0.002"]
  assert _mismatches(agreed, parity) == []


def _covered() -> list[tuple[str, Path, dict[str, dict[str, float]]]]:
  rows = []
  for path in sorted(_PORT.glob("test_*.py")):
    wave, consts = _module_facts(path)
    if wave is None or not consts:
      continue
    rows.append((wave, path, consts))
  return rows


def test_guard_is_not_vacuous() -> None:
  covered = _covered()
  assert len(covered) >= _MIN_COVERED, (
    f"only {len(covered)} port tests expose literal _RTOL/_ATOL dicts "
    f"({[w for w, _, _ in covered]}); the guard would be checking almost nothing"
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
