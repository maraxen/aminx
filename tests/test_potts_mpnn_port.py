"""PottsMPNN port: name map coverage, pin refusal, merge symmetry, self-edge identity.

Uses the shape-only key manifest at ``tests/families/potts_mpnn/data`` so no weights are
needed. Parity against torch is NOT asserted here; that is a later gated unit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn import PottsMPNN
from aminx.families.potts_mpnn.etab import merge_pair
from aminx.families.potts_mpnn.potts_head import PottsHead
from aminx.io.weights import verify_pinned_file

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests/families/potts_mpnn/data/pottsmpnn_vanilla_keys.json"


def _load_converter():
  path = REPO / "scripts/convert_pottsmpnn_vanilla.py"
  spec = importlib.util.spec_from_file_location("convert_pottsmpnn_vanilla", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


converter = _load_converter()


@pytest.fixture(scope="module")
def torch_shapes() -> dict[str, tuple[int, ...]]:
  raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
  return {key: tuple(shape) for key, shape in raw.items()}


@pytest.fixture(scope="module")
def skeleton_shapes() -> dict[str, tuple[int, ...]]:
  return converter.skeleton_parameter_shapes(PottsMPNN(key=jax.random.PRNGKey(0)))


# The manifest is derived from the pinned file (sha256 77e797fd...aa5c, the converter's
# VANILLA_PIN_SHA256) by reading each tensor's shape only: a dump of
# {key: list(tensor.shape)} over model_state_dict. No tensor values are stored. The pin
# is what makes the derivation reproducible; the file itself is gitignored.
def test_manifest_is_the_vanilla_file_with_etab_out(torch_shapes) -> None:
  assert converter.VANILLA_PIN_SHA256.startswith("77e797fd")
  assert len(torch_shapes) == 120
  assert torch_shapes["etab_out.weight"] == (400, 128)
  assert torch_shapes["etab_out.bias"] == (400,)


def test_name_map_is_total_one_to_one_and_shape_preserving(torch_shapes, skeleton_shapes) -> None:
  name_map = converter.build_name_map()
  converter.check_name_map(torch_shapes, skeleton_shapes, name_map)
  assert len(name_map) == 120
  assert len(set(name_map.values())) == 120
  assert len(skeleton_shapes) == 120  # dropout scalars excluded, every weight leaf covered


def test_unmapped_torch_key_fails_loudly(torch_shapes, skeleton_shapes) -> None:
  name_map = converter.build_name_map()
  dropped = "encoder_layers.1.W12.weight"
  pruned = {k: v for k, v in name_map.items() if k != dropped}
  with pytest.raises(ValueError, match=dropped.replace(".", r"\.")):
    converter.check_name_map(torch_shapes, skeleton_shapes, pruned)


def test_wrong_shape_fails_loudly(torch_shapes, skeleton_shapes) -> None:
  name_map = converter.build_name_map()
  bad = dict(torch_shapes)
  bad["etab_out.weight"] = (900, 128)  # the 900-row head the converter must refuse
  with pytest.raises(ValueError, match="shape mismatches"):
    converter.check_name_map(bad, skeleton_shapes, name_map)


def test_etab_out_lands_on_the_potts_head(skeleton_shapes) -> None:
  assert skeleton_shapes[".potts_head.linear.weight"] == (400, 128)
  assert skeleton_shapes[".potts_head.linear.bias"] == (400,)


def test_pin_refuses_a_changed_file(tmp_path: Path) -> None:
  blob = tmp_path / "weights.pt"
  blob.write_bytes(b"not the vanilla checkpoint")
  with pytest.raises(ValueError, match="Refusing to load"):
    verify_pinned_file(blob, converter.VANILLA_PIN_SHA256, what="test checkpoint")


def test_pin_accepts_a_matching_file(tmp_path: Path) -> None:
  import hashlib

  payload = b"exact bytes"
  blob = tmp_path / "weights.pt"
  blob.write_bytes(payload)
  digest = hashlib.sha256(payload).hexdigest()
  assert verify_pinned_file(blob, digest) == blob


def _merge_property_violation(
  merged: np.ndarray,
  original: np.ndarray,
  e_idx: np.ndarray,
  denom: float,
) -> float:
  """Largest deviation from the defining relation of the reverse-edge merge.

  For each edge (i, k) whose neighbour j = e_idx[i, k] has a reverse slot r with
  e_idx[j, r] == i, the reverse entry must satisfy
  ``merged[j, r] == (original[j, r] + original[i, k].T) / denom``.
  """
  length, k = e_idx.shape
  worst = 0.0
  for i in range(length):
    for slot in range(k):
      j = int(e_idx[i, slot])
      reverse = [r for r in range(k) if int(e_idx[j, r]) == i]
      if not reverse:
        continue
      r = reverse[0]
      expected = (original[j, r] + original[i, slot].T) / denom
      worst = max(worst, float(np.max(np.abs(merged[j, r] - expected))))
  return worst


def _random_knn(rng: np.random.Generator, length: int, k: int) -> np.ndarray:
  e_idx = np.zeros((length, k), dtype=np.int64)
  for i in range(length):
    others = np.array([j for j in range(length) if j != i])
    e_idx[i, 0] = i
    e_idx[i, 1:] = rng.choice(others, size=k - 1, replace=False)
  return e_idx


def test_merge_satisfies_the_reverse_edge_relation() -> None:
  rng = np.random.default_rng(1)
  length, k = 9, 5
  e_idx = _random_knn(rng, length, k)
  original = rng.standard_normal((length, k, 20, 20)).astype(np.float32)
  original[:, 0] = original[:, 0] * np.eye(20, dtype=np.float32)
  merged = np.asarray(
    merge_pair(
      jnp.asarray(original),
      jnp.asarray(e_idx),
      jnp.ones(length, dtype=bool),
      denom=2,
      exclude_self=False,
    ),
  )
  assert _merge_property_violation(merged, original, e_idx, denom=2) < 1e-5


def test_merge_negative_control_permuted_neighbours_breaks_the_relation() -> None:
  rng = np.random.default_rng(2)
  length, k = 9, 5
  e_idx = _random_knn(rng, length, k)
  original = rng.standard_normal((length, k, 20, 20)).astype(np.float32)
  original[:, 0] = original[:, 0] * np.eye(20, dtype=np.float32)
  permuted = np.roll(e_idx, 1, axis=0)  # neighbour rows shifted: indices no longer describe the table
  merged_from_permuted = np.asarray(
    merge_pair(
      jnp.asarray(original),
      jnp.asarray(permuted),
      jnp.ones(length, dtype=bool),
      denom=2,
      exclude_self=False,
    ),
  )
  assert _merge_property_violation(merged_from_permuted, original, e_idx, denom=2) > 1e-2


def test_self_edge_slot_keeps_only_the_diagonal() -> None:
  key = jax.random.PRNGKey(3)
  head = PottsHead(128, key=key)
  edge_features = jax.random.normal(jax.random.PRNGKey(4), (6, 5, 128))
  e_idx = jnp.asarray(_random_knn(np.random.default_rng(5), 6, 5), dtype=jnp.int32)
  present = jnp.ones(6)
  pad_valid = jnp.ones(6, dtype=bool)
  tables = np.asarray(head(edge_features, e_idx, present, pad_valid))
  projected = np.asarray(edge_features @ head.linear.weight.T + head.linear.bias).reshape(6, 5, 20, 20)
  off_diagonal = tables[:, 0] * (1.0 - np.eye(20, dtype=np.float32))
  assert np.max(np.abs(off_diagonal)) == 0.0
  assert np.allclose(np.diagonal(tables[:, 0], axis1=-2, axis2=-1), np.diagonal(projected[:, 0], axis1=-2, axis2=-1))
  # Non-self slots are not zeroed.
  assert np.max(np.abs(tables[:, 1:] - projected[:, 1:])) < 1e-6


def test_duplicate_target_fails_loudly(torch_shapes, skeleton_shapes) -> None:
  name_map = converter.build_name_map()
  # Two torch tensors aimed at one aminx leaf: the second would silently overwrite the first.
  name_map["encoder_layers.0.W2.weight"] = name_map["encoder_layers.0.W1.weight"]
  with pytest.raises(ValueError, match="more than one torch key"):
    converter.check_name_map(torch_shapes, skeleton_shapes, name_map)


def test_unhit_skeleton_leaf_fails_loudly(torch_shapes, skeleton_shapes) -> None:
  name_map = converter.build_name_map()
  # Drop a torch tensor and its map entry together, so only the skeleton side is short.
  dropped = "encoder_layers.2.norm3.bias"
  pruned_map = {k: v for k, v in name_map.items() if k != dropped}
  pruned_shapes = {k: v for k, v in torch_shapes.items() if k != dropped}
  with pytest.raises(ValueError, match="no torch source"):
    converter.check_name_map(pruned_shapes, skeleton_shapes, pruned_map)


def test_same_shape_swapped_map_passes_the_structural_check(torch_shapes, skeleton_shapes) -> None:
  # W2 and W3 are both (128, 128) in encoder layer 0, so swapping their targets keeps every
  # structural invariant: total, one-to-one, shape-preserving. This test documents that the
  # structural check CANNOT catch this. Only value parity (converted output vs torch forward,
  # the gated G0 unit) would catch it. Do not read a green check here as correctness.
  name_map = converter.build_name_map()
  w2 = "encoder_layers.0.W2.weight"
  w3 = "encoder_layers.0.W3.weight"
  name_map[w2], name_map[w3] = name_map[w3], name_map[w2]
  converter.check_name_map(torch_shapes, skeleton_shapes, name_map)


def test_verify_pinned_file_missing_path_raises(tmp_path: Path) -> None:
  with pytest.raises(FileNotFoundError):
    verify_pinned_file(tmp_path / "absent.pt", converter.VANILLA_PIN_SHA256)


def test_convert_state_executes_and_writes_float32_leaves(torch_shapes) -> None:
  # Synthetic state with the real key/shape manifest and deterministic float64 values.
  # Executes convert_state end to end: dtype is float32, leaves change from the init.
  rng = np.random.default_rng(7)
  state = {k: rng.standard_normal(s).astype(np.float64) for k, s in torch_shapes.items()}
  init = converter.PottsMPNN(key=jax.random.PRNGKey(0))
  init_w_out = np.asarray(init.mpnn.w_out.weight)
  init_etab = np.asarray(init.potts_head.linear.weight)

  model = converter.convert_state(state, init)

  assert model.mpnn.w_out.weight.dtype == jnp.float32
  assert model.potts_head.linear.weight.dtype == jnp.float32
  assert not np.array_equal(np.asarray(model.mpnn.w_out.weight), init_w_out)
  assert not np.array_equal(np.asarray(model.potts_head.linear.weight), init_etab)
  np.testing.assert_array_equal(
    np.asarray(model.potts_head.linear.weight),
    state["etab_out.weight"].astype(np.float32),
  )


def test_load_vanilla_state_pin_then_torch_load(tmp_path: Path, monkeypatch) -> None:
  import hashlib

  torch = pytest.importorskip("torch")
  state = {"etab_out.weight": torch.arange(400 * 128, dtype=torch.float32).reshape(400, 128)}
  blob = tmp_path / "synthetic.pt"
  torch.save({"model_state_dict": state}, blob)

  # A wrong pin must refuse before torch.load runs.
  with pytest.raises(ValueError, match="Refusing to load"):
    converter.load_vanilla_state(blob)

  # With the matching pin, the same file loads and unwraps model_state_dict.
  digest = hashlib.sha256(blob.read_bytes()).hexdigest()
  monkeypatch.setattr(converter, "VANILLA_PIN_SHA256", digest)
  loaded = converter.load_vanilla_state(blob)
  assert set(loaded) == {"etab_out.weight"}
  np.testing.assert_array_equal(loaded["etab_out.weight"], state["etab_out.weight"].numpy())
