# ruff: noqa: S101
"""Unit tests for LASEr fixture selection and manifest writing. No network."""

from __future__ import annotations

import importlib.util
import io
import sys
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "acquire_laser_fixtures.py"


def _load() -> ModuleType:
  spec = importlib.util.spec_from_file_location("acquire_laser_fixtures", _SCRIPT)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_SCRIPT}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  # dataclasses (3.14) look up cls.__module__ in sys.modules during class creation.
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


laser = _load()


def test_selection_predicate_positive_and_negative() -> None:
  """Qualifying complexes pass; water-only, ion-only, short, and small ligands do not."""
  qualifying = laser.synthetic_pdb(n_ca=80, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),))
  water_only = laser.synthetic_pdb(n_ca=80, ligands=(("HOH", ("O", "O", "O", "O", "O", "O", "O")),))
  ion_only = laser.synthetic_pdb(
    n_ca=80,
    ligands=(("SO4", ("S", "O", "O", "O", "O")), ("CA", ("CA",))),
  )
  too_short = laser.synthetic_pdb(n_ca=10, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),))
  too_long = laser.synthetic_pdb(n_ca=301, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),))
  five_heavy = laser.synthetic_pdb(n_ca=80, ligands=(("LIG", ("C", "C", "C", "C", "C")),))
  hydrogens_do_not_count = laser.synthetic_pdb(
    n_ca=80,
    ligands=(("LIG", ("C", "C", "C", "C", "C", "H", "H", "H")),),
  )
  lower_bound = laser.synthetic_pdb(n_ca=50, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),))
  upper_bound = laser.synthetic_pdb(n_ca=300, ligands=(("LIG", ("C", "C", "C", "C", "C", "C")),))

  assert laser.assess_pdb(io.BytesIO(qualifying)).ok
  assert laser.assess_pdb(io.BytesIO(lower_bound)).ok
  assert laser.assess_pdb(io.BytesIO(upper_bound)).ok
  assert not laser.assess_pdb(io.BytesIO(water_only)).ok
  assert not laser.assess_pdb(io.BytesIO(ion_only)).ok
  assert not laser.assess_pdb(io.BytesIO(too_short)).ok
  assert not laser.assess_pdb(io.BytesIO(too_long)).ok
  assert not laser.assess_pdb(io.BytesIO(five_heavy)).ok
  assert not laser.assess_pdb(io.BytesIO(hydrogens_do_not_count)).ok


def test_select_entries_keeps_only_qualifying_in_lex_order() -> None:
  """Lexicographic walk skips failures and keeps the qualifying id only."""
  qualifying = "qual_1-A-A"
  water_only = "water_1-A-A"
  too_short = "short_1-A-A"
  blobs = {
    qualifying: laser.synthetic_pdb(n_ca=80, ligands=(("LIG", ("C",) * 6),)),
    water_only: laser.synthetic_pdb(n_ca=80, ligands=(("WAT", ("O",) * 8),)),
    too_short: laser.synthetic_pdb(n_ca=49, ligands=(("LIG", ("C",) * 6),)),
  }
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    for entry_id, blob in blobs.items():
      archive.writestr(f"nested/{entry_id}.pdb", blob)
  buffer.seek(0)
  with zipfile.ZipFile(buffer) as archive:
    chosen = laser.select_entries(archive, list(blobs), limit=20)
  assert chosen == [qualifying]
  assert water_only not in chosen
  assert too_short not in chosen


def test_choose_split_prefers_heldout_val_30pct() -> None:
  """The LASEr zip has no test split; val held-out 30pct is the documented fallback."""
  names = [
    "ligandmpnn_train_data_30pct_main_clusters.json",
    "train_streptavidin_heldout_split_30pct_main_clusters.json",
    "val_streptavidin_heldout_split_70pct_subclusters.json",
    "val_streptavidin_heldout_split_30pct_main_clusters.json",
  ]
  member, fallback = laser.choose_split_member(names)
  assert member == "val_streptavidin_heldout_split_30pct_main_clusters.json"
  assert fallback is False

  test_member, test_fallback = laser.choose_split_member(
    [
      "train_a.json",
      "my_test_split.json",
      "val_streptavidin_heldout_split_30pct_main_clusters.json",
    ],
  )
  assert test_member == "my_test_split.json"
  assert test_fallback is False

  only_train, fell_through = laser.choose_split_member(
    [
      "ligandmpnn_train_data_70pct_subclusters.json",
      "ligandmpnn_train_data_30pct_main_clusters.json",
    ],
  )
  assert only_train == "ligandmpnn_train_data_30pct_main_clusters.json"
  assert fell_through is True


def test_collect_entry_ids_flattens_nested_split() -> None:
  """Cluster keys and nested member lists both count as entry ids."""
  payload = {
    "1aoc_1-A-A": {"1aoc_1-A-A": ["1aoc_1-A-A", "1aoc_1-A-B"]},
    "1bgd_1-A-A": ["1bgd_1-A-A"],
  }
  assert laser.collect_entry_ids(payload) == {"1aoc_1-A-A", "1aoc_1-A-B", "1bgd_1-A-A"}


def test_split_ids_map_to_assembly_stems() -> None:
  """Chain-suffixed split ids collapse onto the archive's one-file-per-bioassembly stems."""
  ids = {"1aoc_1-A-A", "1aoc_1-A-B", "1a19_2-B-A", "1bgd_1-A-A"}
  assert laser.split_ids_to_stems(ids) == ["1a19_2", "1aoc_1", "1bgd_1"]
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    archive.writestr(
      "reduce_filtered/ao/1aoc_1.pdb",
      laser.synthetic_pdb(n_ca=80, ligands=(("LIG", ("C",) * 6),)),
    )
  buffer.seek(0)
  with zipfile.ZipFile(buffer) as archive:
    # Raw chain-suffixed ids never match a member; the mapped stem does.
    assert laser.select_entries(archive, sorted(ids), limit=20) == []
    assert laser.select_entries(archive, laser.split_ids_to_stems(ids), limit=20) == ["1aoc_1"]


def test_manifest_writer_round_trip() -> None:
  """The manifest TOML keeps the rule, ordered ids, B0 extras, and per-file hashes."""
  ids = ("aaaa", "bbbb", "cccc", "dddd", "eeee")
  manifest = laser.FixtureManifest(
    source_urls=("https://zenodo.org/records/17990180/files/part.aa",),
    zenodo_record_ids=("17990180", "17990253"),
    archive_md5=laser.EXPECTED_MD5,
    archive_sha256="a" * 64,
    archive_size=50,
    selection_rule=laser.SELECTION_RULE,
    split_file="val_streptavidin_heldout_split_30pct_main_clusters.json",
    split_name="val_streptavidin_heldout_split_30pct_main_clusters",
    split_fallback=False,
    split_choice_note=laser.SPLIT_CHOICE_NOTE,
    ordered_ids=ids,
    file_sha256={f"{entry_id}.pdb": str(index) * 64 for index, entry_id in enumerate(ids)},
    b0_extras=ids[: laser.N_B0_EXTRAS],
  )
  parsed = tomllib.loads(laser.render_manifest(manifest))
  assert parsed["selection_rule"] == laser.SELECTION_RULE
  assert parsed["ordered_ids"] == list(ids)
  assert parsed["b0_extras"] == list(ids[:4])
  assert parsed["archive_md5"] == laser.EXPECTED_MD5
  assert parsed["zenodo_record_ids"] == ["17990180", "17990253"]
  assert parsed["split_file"].endswith("val_streptavidin_heldout_split_30pct_main_clusters.json")
  assert parsed["split_fallback"] is False
  assert parsed["files"]["aaaa.pdb"] == "0" * 64
  assert parsed["archive_size"] == 50
