# ruff: noqa: S101
"""Root resolution (explicit > env var > default path) and read-error recording in the extractor.

Not run on the local box (narrow runs only, and the remote suite is the gate). The pre-fix
behaviour: ``FAMILY_REGISTRY`` declared env_var/default_paths that nothing read, and an unreadable
dataclass file returned ``[]`` with no message.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_extractor() -> ModuleType:
  path = _REPO / "scripts" / "redsox" / "extract_upstream_knobs.py"
  spec = importlib.util.spec_from_file_location("extract_upstream_knobs_under_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


EX = _load_extractor()


def _cfg() -> object:
  return EX.FAMILY_REGISTRY["potts"]


def test_explicit_argument_beats_env_and_default(tmp_path: Path) -> None:
  explicit = tmp_path / "explicit"
  decision = EX.resolve_family_root(
    _cfg(), explicit, {"AMINX_POTTS_ROOT": str(tmp_path / "env")},
  )
  assert decision is not None
  assert decision.path == explicit.resolve()
  assert decision.source == "argument"


def test_env_var_beats_default_path(tmp_path: Path) -> None:
  env_root = tmp_path / "from_env"
  decision = EX.resolve_family_root(_cfg(), None, {"AMINX_POTTS_ROOT": str(env_root)})
  assert decision is not None
  assert decision.path == env_root.resolve()
  assert decision.source == "env:AMINX_POTTS_ROOT"


def test_blank_env_var_falls_through_to_the_default_layer(tmp_path: Path) -> None:
  # Pre-fix this asserted only "not the env layer", which a None result also satisfies. Now the
  # check names the next layer down: a blank env var is unset, so the existing default is reported.
  checkout = tmp_path / "PottsMPNN"
  checkout.mkdir()
  cfg = EX.UpstreamFamilyConfig(
    family="potts",
    class_prefix="Potts",
    env_var="AMINX_POTTS_ROOT",
    default_paths=(str(checkout),),
    commit_var="POTTSMPNN_COMMIT",
    has_vendor_pin=True,
  )
  decision = EX.resolve_family_root(cfg, None, {"AMINX_POTTS_ROOT": "   "})
  assert decision is not None
  assert decision.source == f"default:{checkout}"
  assert decision.path == checkout.resolve()


def test_default_path_is_used_last_and_names_the_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  home = tmp_path / "home"
  checkout = home / "repos" / "PottsMPNN"
  checkout.mkdir(parents=True)
  monkeypatch.setenv("HOME", str(home))
  cfg = EX.UpstreamFamilyConfig(
    family="potts",
    class_prefix="Potts",
    env_var="AMINX_POTTS_ROOT",
    default_paths=("~/repos/Missing", "~/repos/PottsMPNN"),
    commit_var="POTTSMPNN_COMMIT",
    has_vendor_pin=True,
  )
  decision = EX.resolve_family_root(cfg, None, {})
  assert decision is not None
  assert decision.path == checkout.resolve()
  assert decision.source == "default:~/repos/PottsMPNN"


def test_nothing_resolves_returns_none(tmp_path: Path) -> None:
  cfg = EX.UpstreamFamilyConfig(
    family="potts",
    class_prefix="Potts",
    env_var="AMINX_POTTS_ROOT",
    default_paths=(str(tmp_path / "absent"),),
    commit_var="POTTSMPNN_COMMIT",
    has_vendor_pin=True,
  )
  assert EX.resolve_family_root(cfg, None, {}) is None


def test_unreadable_dataclass_file_is_recorded_not_swallowed(tmp_path: Path) -> None:
  # A directory where a file is expected raises an OSError on read. Pre-fix this returned [] with
  # no message, so the surface came out short and nothing said why.
  unresolved: list[str] = []
  entries = EX._extract_dataclass_entries("potts", tmp_path, tmp_path, frozenset({"X"}), unresolved)
  assert entries == []
  assert any("cannot read file" in item for item in unresolved), unresolved
