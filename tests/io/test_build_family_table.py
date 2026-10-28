"""The family weights table must be what its generator writes.

``src/aminx/io/family_weights.json`` says it is generated and must not be edited by hand. These tests
check the generator's table assembly without weights or network: regenerating from the checked-in rows
reproduces the file byte for byte, and a rebuild keeps the rows it does not build.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TABLE_PATH = REPO / "src/aminx/io/family_weights.json"
GENERATOR_PATH = REPO / "scripts/weights/build_family_weights.py"


def _load_generator() -> ModuleType:
  spec = importlib.util.spec_from_file_location("build_family_weights_under_test", GENERATOR_PATH)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _pinned() -> tuple[str, list[dict[str, Any]]]:
  text = TABLE_PATH.read_text(encoding="utf-8")
  return text, json.loads(text)["checkpoints"]


def test_regenerating_from_the_checked_in_rows_reproduces_the_file() -> None:
  generator = _load_generator()
  text, pinned = _pinned()
  catalog_ids = {entry.checkpoint_id for entry in generator.CATALOG}
  built = [row for row in pinned if row["checkpoint_id"] in catalog_ids]

  assert generator.render_table(generator.assemble_table(built, pinned)) == text


def test_rebuild_keeps_families_it_does_not_build() -> None:
  generator = _load_generator()
  _, pinned = _pinned()

  table = generator.assemble_table([], pinned)
  kept = {row["family"] for row in table["checkpoints"]}

  assert {"fampnn", "thermompnn"} <= kept
  assert [row["checkpoint_id"] for row in table["checkpoints"]] == [row["checkpoint_id"] for row in pinned]


def test_built_row_replaces_pinned_row_with_the_same_id() -> None:
  generator = _load_generator()
  _, pinned = _pinned()
  replaced = dict(pinned[0], sha256="0" * 64)

  table = generator.assemble_table([replaced], pinned)
  by_id = {row["checkpoint_id"]: row for row in table["checkpoints"]}

  assert by_id[replaced["checkpoint_id"]]["sha256"] == "0" * 64
  assert len(table["checkpoints"]) == len(pinned)


def test_header_keys_are_restored() -> None:
  generator = _load_generator()
  _, pinned = _pinned()

  table = generator.assemble_table([], pinned)

  assert table["schema_version"] == 1
  assert "do not edit by hand" in table["note"]
  assert list(table)[:2] == ["schema_version", "note"]
