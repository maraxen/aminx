"""Every port wave's REPORTED band must be checked against the one it APPLIES.

WHY THIS EXISTS (found 261006). ``test_port_tolerances_match_targets.py`` solves
exactly the right problem -- debt #2445, "amending only the TOML changes no
assertion" -- but it can only see waves whose test module declares flat
``_RTOL`` / ``_ATOL`` / ``_SCALE_REL`` dicts at module level. That is five
modules, and ALL FIVE ARE LASEr. Every Potts wave is invisible to it, so on the
Potts side the #2445 defect is still live.

It does not notice, because its anti-vacuity check is
``len(covered) >= _MIN_COVERED`` with ``_MIN_COVERED = 3``. Five LASEr waves
satisfy a COUNT, so the guard certifies itself non-vacuous while checking no
Potts wave at all. A count is not coverage, and this file is the coverage half.

WHY POTTS WAS INVISIBLE: it uses a DIFFERENT CONVENTION, not a worse one.
LASEr writes ``_RTOL = {"f64": ..., "f32": ...}``; Potts writes a nested
``_TOL = {"f64": {"rtol": ..., "atol": ...}, ...}``, or inline literals at the
call site, or a precision-keyed ternary. Four shapes in one suite. This file
reads all four and then insists every wave be accounted for.

THE READING OF A POLICY STRING, stated once so it is arguable:
  * an OMITTED component means zero. ``"atol=1e-7"`` declares ``rtol=0``; this
    is what ``potts_merge_pair_d2`` actually applies, so the TOML is terse but
    honest rather than wrong.
  * ``"exact"`` means ``rtol=0, atol=0``, which is what the merge waves apply.
  * anything else is PROSE, not a machine-checkable band. Prose is not waved
    through: it must be named in ``_PROSE`` with a reason, so the set of
    unverifiable declarations is a list someone can read rather than a silence.

WHAT IT FOUND ON ARRIVAL. One real discrepancy, ``pottsmpnn_full`` f32: the
target reports ``"log-prob atol=1e-4"`` while the test applies ``rtol=1e-4``
as well (test_pottsmpnn_full.py:68-69). The reported band is not the applied
band -- the #2445 defect itself, on a wave its own guard could not see. It is
pinned in ``_KNOWN_UNREPORTED`` rather than hidden: the registry is asserted to
match EXACTLY, so a new discrepancy fails here, and so does FIXING this one
without removing its entry. Repairing it edits ``tests/port/targets/``, which is
scoped, so it rides the post-merge re-wave.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_PORT = Path(__file__).resolve().parents[1] / "port"
_TARGETS = _PORT / "targets"

#: LASEr convention: ``{precision: value}``, one constant per band component.
_FLAT = {"_RTOL": "rtol", "_ATOL": "atol", "_SCALE_REL": "scale_rel"}
#: Potts convention: ``{precision: {"rtol": ..., "atol": ...}}``.
_NESTED = ("_TOL",)
_PRECISIONS = ("f64", "f32")

#: Declarations that are prose, not a band, with the reason each is unverifiable.
#: Being on this list is a claim someone can dispute; being absent from every
#: list is impossible, which is the point of the whole file.
_PROSE: dict[tuple[str, str], str] = {
  ("declayer_f64", "f32"): "f64-only wave; there is no f32 tier to band",
  ("potts_energy", "f32"): "r15 condition bound |err| <= 1e-5*sum|terms|, not an envelope",
  ("pottsmpnn_full", "f32"): "prose form 'log-prob atol=1e-4'; see _KNOWN_UNREPORTED",
}

#: Waves that apply NO numeric band because they compare by equality. Their
#: ``"exact"`` declaration is machine-readable and correct, so they are not
#: prose -- but there is no rtol/atol to match it against either.
#:
#: Membership is VERIFIED, not taken on trust: the module must really compare
#: with array_equal and must carry no tolerance kwarg anywhere. A wave that
#: quietly grew a tolerance would stop qualifying and fail, which is the only
#: reason this is a category rather than an excuse.
_EXACT_BY_CONSTRUCTION: dict[str, str] = {
  "laser_order": "decoding order compared by np.array_equal",
  "potts_order": "neighbour order compared by np.array_equal / set equality",
}

#: Waves that legitimately have NO test module, so no band is applied anywhere
#: and there is nothing for this file to compare.
#:
#: THIS REGISTRY EXISTS BECAUSE ITS ABSENCE WAS A HOLE (found 261006 by probe).
#: The accounting test used to `continue` past any wave it could not find a
#: module for, with `port_selftest` in mind. But a wave is matched to its module
#: by the ``port_wave`` marker, so deleting that one marker argument -- from a
#: module that still exists and still asserts -- made the wave simply vanish
#: from the loop, and the test PASSED. Both guards keyed on the same marker, so
#: nothing else caught it either. A mutation probe that dropped the marker from
#: test_laser_encoder.py was waved through by both; dropping its BANDS was
#: correctly refused, which is what made the gap specific rather than theoretical.
#:
#: So membership is declared AND verified: an entry here must really have no
#: module, and a wave missing a module without an entry is now a failure.
_NO_TEST_MODULE: dict[str, str] = {
  "port_selftest": "the suite's own self-test target; no tests/port module declares it",
}

#: Waves whose APPLIED band is real but whose REPORTED policy does not state it
#: fully. This is the #2445 defect, pinned so it cannot grow silently. Asserted
#: to match exactly, so removing a defect without removing its entry also fails.
_KNOWN_UNREPORTED: dict[tuple[str, str], str] = {
  ("pottsmpnn_full", "f32"): (
    "target reports 'log-prob atol=1e-4' but test_pottsmpnn_full.py:68-69 applies "
    "rtol=1e-4 as well. Fixing it edits tests/port/targets/ (scoped) -> re-wave."
  ),
}


def _wave_of(tree: ast.Module) -> str | None:
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "port_wave"
      and node.args
      and isinstance(node.args[0], ast.Constant)
    ):
      return str(node.args[0].value)
  return None


def _module_level_dicts(tree: ast.Module) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for node in tree.body:
    target = value = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
      target, value = node.target, node.value
    if not isinstance(target, ast.Name):
      continue
    if target.id in _FLAT or target.id in _NESTED:
      try:
        literal = ast.literal_eval(value)
      except ValueError:
        continue
      if isinstance(literal, dict):
        out[target.id] = literal
  return out


def _inline_bands(tree: ast.Module) -> dict[str, float]:
  """Literal ``rtol=``/``atol=`` kwargs, accepted only when UNANIMOUS.

  Two call sites disagreeing means the module has no single band, so reporting
  either one as "the" applied band would be a guess. Disagreement yields
  nothing and the wave falls through to the accounting assertion.
  """
  seen: dict[str, set[float]] = {"rtol": set(), "atol": set()}
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    for keyword in node.keywords:
      if keyword.arg in seen and isinstance(keyword.value, ast.Constant):
        seen[keyword.arg].add(float(keyword.value.value))
  return {key: values.pop() for key, values in seen.items() if len(values) == 1}


def _ternary_bands(tree: ast.Module) -> dict[str, dict[str, float]]:
  """``atol = 1e-8 if precision == "f64" else 1e-4`` -> per-precision values."""
  out: dict[str, dict[str, float]] = {}
  for node in ast.walk(tree):
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
      continue
    target, value = node.targets[0], node.value
    if not (isinstance(target, ast.Name) and target.id in ("rtol", "atol")):
      continue
    if not isinstance(value, ast.IfExp):
      continue
    test = value.test
    if not (
      isinstance(test, ast.Compare)
      and isinstance(test.left, ast.Name)
      and test.left.id == "precision"
      and len(test.comparators) == 1
      and isinstance(test.comparators[0], ast.Constant)
    ):
      continue
    chosen = str(test.comparators[0].value)
    other = "f32" if chosen == "f64" else "f64"
    if isinstance(value.body, ast.Constant) and isinstance(value.orelse, ast.Constant):
      out.setdefault(chosen, {})[target.id] = float(value.body.value)
      out.setdefault(other, {})[target.id] = float(value.orelse.value)
  return out


def applied_bands(path: Path) -> dict[str, dict[str, float]]:
  """The band a module actually applies, per precision, from any of four shapes."""
  tree = ast.parse(path.read_text(encoding="utf-8"))
  consts = _module_level_dicts(tree)
  bands: dict[str, dict[str, float]] = {p: {} for p in _PRECISIONS}

  for name, component in _FLAT.items():
    for precision, value in consts.get(name, {}).items():
      if precision in bands:
        bands[precision][component] = float(value)
  for name in _NESTED:
    for precision, entry in consts.get(name, {}).items():
      if precision in bands and isinstance(entry, dict):
        for component, value in entry.items():
          bands[precision][str(component)] = float(value)

  for precision, entry in _ternary_bands(tree).items():
    if precision in bands:
      bands[precision].update({k: v for k, v in entry.items() if k not in bands[precision]})

  # Inline literals are the weakest source and apply to BOTH precisions, so they
  # only fill a precision no stronger source described.
  inline = _inline_bands(tree)
  for precision in _PRECISIONS:
    if not bands[precision]:
      bands[precision] = dict(inline)
  return {p: b for p, b in bands.items() if b}


def precision_keyed_bands(path: Path) -> dict[str, dict[str, float]]:
  """Only the bands whose precision is UNAMBIGUOUS from the source.

  Deliberately excludes inline call-site literals. A literal ``rtol=1e-10`` in a
  shared helper cannot be attributed to f64 or f32 without reasoning about which
  test reaches it, and guessing would produce exactly the false accusation this
  file is supposed to prevent: ``potts_energy`` applies ``rtol=1e-10`` on its f64
  path while its f32 tier uses the r15 condition bound, so attributing the
  literal to both would report a drift that does not exist.
  """
  tree = ast.parse(path.read_text(encoding="utf-8"))
  consts = _module_level_dicts(tree)
  bands: dict[str, dict[str, float]] = {p: {} for p in _PRECISIONS}
  for name, component in _FLAT.items():
    for precision, value in consts.get(name, {}).items():
      if precision in bands:
        bands[precision][component] = float(value)
  for name in _NESTED:
    for precision, entry in consts.get(name, {}).items():
      if precision in bands and isinstance(entry, dict):
        for component, value in entry.items():
          bands[precision][str(component)] = float(value)
  for precision, entry in _ternary_bands(tree).items():
    if precision in bands:
      bands[precision].update(entry)
  return {p: b for p, b in bands.items() if b}


def declared_band(policy: str) -> dict[str, float] | None:
  """A policy string as a band, or None when it is prose.

  An omitted component is zero; ``"exact"`` is zero on both.
  """
  text = policy.strip()
  if text == "exact":
    return {"rtol": 0.0, "atol": 0.0}
  out: dict[str, float] = {}
  for part in text.split(","):
    key, sep, value = part.strip().partition("=")
    if not sep:
      return None
    try:
      out[key.strip()] = float(value)
    except ValueError:
      return None
  if not out or not set(out) <= {"rtol", "atol", "scale_rel"}:
    return None
  if "scale_rel" not in out:
    out.setdefault("rtol", 0.0)
    out.setdefault("atol", 0.0)
  return out


def _modules() -> dict[str, Path]:
  out: dict[str, Path] = {}
  for path in sorted(_PORT.glob("test_*.py")):
    wave = _wave_of(ast.parse(path.read_text(encoding="utf-8")))
    if wave is not None:
      out[wave] = path
  return out


def test_policy_reader_handles_every_form() -> None:
  """Synthetic controls. A reader that cannot refuse would pass everything."""
  assert declared_band("exact") == {"rtol": 0.0, "atol": 0.0}
  assert declared_band("atol=1e-7") == {"rtol": 0.0, "atol": 1e-7}
  assert declared_band("rtol=1e-12") == {"rtol": 1e-12, "atol": 0.0}
  assert declared_band("rtol=1e-8,atol=1e-11") == {"rtol": 1e-8, "atol": 1e-11}
  assert declared_band("scale_rel=4e-4") == {"scale_rel": 4e-4}
  # Prose must REFUSE, not parse to something plausible.
  for prose in ("f64-only", "|err|<=1e-5*sum|terms|", "log-prob atol=1e-4", "ulps=4"):
    assert declared_band(prose) is None, prose


def test_every_target_is_accounted_for() -> None:
  """No wave may fall through: covered, prose, or without a test module."""
  modules = _modules()
  orphaned = []
  for target in sorted(_TARGETS.glob("*.toml")):
    wave = target.stem
    if wave not in modules:
      # NOT a free pass -- see _NO_TEST_MODULE. A wave losing its port_wave
      # marker looks identical here to one that never had a module, and that
      # is how a live, asserting module used to leave the guard unnoticed.
      if wave not in _NO_TEST_MODULE:
        orphaned.append(
          f"{wave}: targets/{wave}.toml exists but no tests/port module declares "
          f"port_wave({wave!r}). Either the marker was dropped from a module that "
          f"still asserts (a regression this guard must refuse), or the wave "
          f"genuinely has no module and belongs in _NO_TEST_MODULE with a reason.",
        )
      continue
    if wave in _EXACT_BY_CONSTRUCTION:
      continue
    bands = applied_bands(modules[wave])
    parity = tomllib.loads(target.read_text(encoding="utf-8"))["parity"]
    for precision in _PRECISIONS:
      if (wave, precision) in _PROSE:
        continue
      declared = declared_band(str(parity[f"tolerance_policy_{precision}"]))
      if declared is None or not bands.get(precision):
        orphaned.append(f"{wave} {precision}: no band read and not listed in _PROSE")
  assert not orphaned, (
    "these waves are neither machine-checked nor declared unverifiable, which is "
    "exactly the silent gap this file exists to close:\n  " + "\n  ".join(orphaned)
  )


def test_no_test_module_registry_is_earned_not_declared() -> None:
  """An entry must really have no module, or it is an opt-out from the guard.

  Without this, _NO_TEST_MODULE would be a way to silence any wave by naming
  it -- which is precisely the hole it was added to close, moved one level up.
  """
  modules = _modules()
  wrong = []
  for wave, reason in _NO_TEST_MODULE.items():
    if not (_TARGETS / f"{wave}.toml").is_file():
      wrong.append(f"{wave}: no such target; remove the entry")
    if wave in modules:
      wrong.append(
        f"{wave}: {modules[wave].name} DOES declare port_wave({wave!r}), so the "
        f"wave is checkable and must not be exempt",
      )
    assert reason.strip(), f"{wave}: _NO_TEST_MODULE entry needs a reason"
  assert not wrong, "\n  ".join(wrong)


def test_prose_list_is_not_a_dumping_ground() -> None:
  """Every _PROSE entry must name a real target that really is prose."""
  stale = []
  for (wave, precision), reason in _PROSE.items():
    target = _TARGETS / f"{wave}.toml"
    if not target.is_file():
      stale.append(f"{wave}: no such target")
      continue
    parity = tomllib.loads(target.read_text(encoding="utf-8"))["parity"]
    if declared_band(str(parity[f"tolerance_policy_{precision}"])) is not None:
      stale.append(f"{wave} {precision}: now machine-readable; remove from _PROSE")
    assert reason.strip(), f"{wave} {precision}: _PROSE entry needs a reason"
  assert not stale, "\n  ".join(stale)


@pytest.mark.parametrize("wave", sorted(_EXACT_BY_CONSTRUCTION))
def test_exact_by_construction_is_earned_not_declared(wave: str) -> None:
  """A wave claiming exactness must really compare by equality, with no band.

  Without this the category would be a way to opt out of the guard by writing a
  line in a dict. A wave that grows a tolerance kwarg stops qualifying and the
  claim fails here.
  """
  modules = _modules()
  assert wave in modules, f"{wave}: no test module"
  source = modules[wave].read_text(encoding="utf-8")
  assert "array_equal" in source, (
    f"{wave} claims exactness by construction but does not use array_equal"
  )
  bands = applied_bands(modules[wave])
  assert not bands, (
    f"{wave} claims exactness by construction but applies a numeric band {bands}; "
    f"it must be removed from _EXACT_BY_CONSTRUCTION and checked properly"
  )
  parity = tomllib.loads((_TARGETS / f"{wave}.toml").read_text(encoding="utf-8"))["parity"]
  for precision in _PRECISIONS:
    policy = str(parity[f"tolerance_policy_{precision}"])
    assert declared_band(policy) == {"rtol": 0.0, "atol": 0.0}, (
      f"{wave} {precision}: compares by equality but its target reports {policy!r}"
    )


@pytest.mark.parametrize("wave", sorted(_modules()))
def test_reported_band_is_the_applied_band(wave: str) -> None:
  """The band the target REPORTS must be the band the module APPLIES."""
  target = _TARGETS / f"{wave}.toml"
  if not target.is_file():
    pytest.skip(f"{wave} has no target file")
  parity = tomllib.loads(target.read_text(encoding="utf-8"))["parity"]
  bands = applied_bands(_modules()[wave])
  problems = []
  for precision in _PRECISIONS:
    if (wave, precision) in _PROSE or (wave, precision) in _KNOWN_UNREPORTED:
      continue
    declared = declared_band(str(parity[f"tolerance_policy_{precision}"]))
    applied = bands.get(precision)
    if declared is None or not applied:
      continue  # accounted for by test_every_target_is_accounted_for
    for component, value in declared.items():
      if component in applied and applied[component] != value:
        problems.append(
          f"{precision} {component}: target reports {value!r}, test applies "
          f"{applied[component]!r}"
        )
  assert not problems, (
    f"targets/{wave}.toml -- the verdict it emits would claim a band it did not "
    f"apply:\n  " + "\n  ".join(problems)
  )


def _scan_under_reported() -> set[tuple[str, str]]:
  """Every (wave, precision) applying a real band its target does not state.

  SCANS, rather than iterating the registry. An earlier version looped over
  ``_KNOWN_UNREPORTED`` itself, so emptying the registry made the loop vacuous
  and the check passed -- it could confirm a listed defect but could never
  DISCOVER an unlisted one, which is most of what a tripwire is for. A forced
  negative control caught that; this is the repair.
  """
  found: set[tuple[str, str]] = set()
  for wave, path in _modules().items():
    target = _TARGETS / f"{wave}.toml"
    if not target.is_file() or wave in _EXACT_BY_CONSTRUCTION:
      continue
    parity = tomllib.loads(target.read_text(encoding="utf-8"))["parity"]
    bands = precision_keyed_bands(path)
    for precision in _PRECISIONS:
      applied = bands.get(precision, {})
      if not applied:
        continue
      declared = declared_band(str(parity[f"tolerance_policy_{precision}"]))
      if declared is None:
        # Prose declaration, but a concrete per-precision band IS applied.
        if any(value for value in applied.values()):
          found.add((wave, precision))
      elif any(
        component not in declared and value for component, value in applied.items()
      ):
        found.add((wave, precision))
  return found


def test_known_unreported_registry_is_exact() -> None:
  """A tripwire in BOTH directions.

  A new under-reported band must fail here. So must FIXING a listed one without
  removing its entry -- otherwise the registry quietly becomes a list of things
  that used to be true, which is how a known-issues list stops being read.
  """
  for (wave, precision), reason in _KNOWN_UNREPORTED.items():
    assert (_TARGETS / f"{wave}.toml").is_file(), f"{wave}: no such target"
    assert reason.strip(), f"{wave} {precision}: entry needs a reason"
  measured = _scan_under_reported()
  assert measured == set(_KNOWN_UNREPORTED), (
    "the known-under-reported registry no longer matches reality. If a wave was "
    "fixed, delete its entry; if one appeared, fix it or justify it.\n"
    f"  registry: {sorted(_KNOWN_UNREPORTED)}\n  measured: {sorted(measured)}"
  )
