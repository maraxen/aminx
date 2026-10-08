# ruff: noqa: S101
"""The ProtonPottsMPNN checkpoint is identifiable by the gate (spec §50 S1).

Three claims, each with the negative that shows it discriminates:

* ``_weights_required`` has a ``protonpottsmpnn_`` arm (and still does not claim an unrelated slug);
* ``stable_artifact_key`` anchors a path at ``ProtonPottsMPNN``, so a checkout anywhere keys the same,
  and a path under the sibling ``PottsMPNN`` keeps its own key (the two names must not alias);
* the registry lists the v6 checkpoint under exactly that key, with a well-formed digest.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from knob_gate import _coverage

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = Path(__file__).with_name("checkpoint_registry.json")
_CKPT = "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"


def _artifact_key():  # noqa: ANN202
  spec = importlib.util.spec_from_file_location("artifact_key", _REPO / "scripts/parity/artifact_key.py")
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.stable_artifact_key


def test_weights_arm_covers_protonpotts_and_only_its_own() -> None:
  assert _coverage._weights_required("protonpottsmpnn_energy")
  assert _coverage._weights_required("pottsmpnn_cfg")
  assert not _coverage._weights_required("protonpotts_parity")
  assert not _coverage._weights_required("unrelated_slug")


def test_key_is_anchored_at_the_upstream_root_wherever_it_is_cloned() -> None:
  key = _artifact_key()
  first = key(f"/home/a/repos/ProtonPottsMPNN/{_CKPT}")
  second = key(f"/scratch/b/clone/ProtonPottsMPNN/{_CKPT}")
  assert first == second == f"ProtonPottsMPNN/{_CKPT}"


def test_sibling_names_do_not_alias() -> None:
  key = _artifact_key()
  assert key("/x/PottsMPNN/vanilla_model_weights/pottsmpnn_20.pt") == "PottsMPNN/vanilla_model_weights/pottsmpnn_20.pt"
  assert not key(f"/x/ProtonPottsMPNN/{_CKPT}").startswith("PottsMPNN/")


def test_registry_lists_the_v6_checkpoint_under_the_stable_key() -> None:
  entries = json.loads(_REGISTRY.read_text(encoding="utf-8"))["entries"]
  by_path = {entry["artifact_path"]: entry for entry in entries}
  entry = by_path[f"ProtonPottsMPNN/{_CKPT}"]
  assert entry["family"] == "protonpottsmpnn"
  assert entry["upstream"] == "ProtonPottsMPNN"
  assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
  assert len(by_path) == len(entries), "duplicate artifact_path"
