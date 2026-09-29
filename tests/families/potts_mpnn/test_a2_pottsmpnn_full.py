# ruff: noqa: S101
"""``pottsmpnn_full`` wave: encoder states, decoder states, log-probs, etab_forward.

Skips when the A1 oracle dump or torch is absent. A sha256 mismatch against
``tests/port/reference/a1_potts/oracles.sha256`` fails. Weights are converted
from the torch checkpoints under ``AMINX_POTTS_ROOT`` (default ``~/repos/PottsMPNN``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_REFERENCE = _REPO / "tests/port/reference/a1_potts"
_CONVERTER = _REPO / "scripts/recapture/pottsmpnn_model_to_eqx.py"

# §4.1a: knn_boundary_tie structures are outside the exact wave.
_EXCLUDED_FIXTURES = frozenset({"gap_gt48_gaps0", "gap_gt48_gaps1"})


def _converter():
  spec = importlib.util.spec_from_file_location("pottsmpnn_model_to_eqx", _CONVERTER)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_CONVERTER}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _oracle_dir() -> Path:
  raw = os.environ.get("AMINX_A1_ORACLE_DIR")
  if raw:
    return Path(raw)
  return Path.home() / "projects/aminx-oracles/dumps/a1_potts"


def _potts_root() -> Path:
  raw = os.environ.get("AMINX_POTTS_ROOT")
  if raw:
    return Path(raw)
  return Path.home() / "repos/PottsMPNN"


def _expected_sha(suffix: str) -> str:
  for line in (_REFERENCE / "oracles.sha256").read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    digest, path = line.split(maxsplit=1)
    if Path(path.strip()).as_posix().endswith(suffix):
      return digest
  msg = f"{suffix} is not listed in oracles.sha256"
  raise AssertionError(msg)


def _checkpoint_paths() -> dict[str, str]:
  manifest = tomllib.loads((_REFERENCE / "oracle_manifest.toml").read_text(encoding="utf-8"))
  paths: dict[str, str] = {}
  for row in manifest["checkpoint"]:
    checkpoint_id = str(row["id"])
    if checkpoint_id.endswith("_f64"):
      paths[checkpoint_id.removesuffix("_f64")] = str(row["path"])
  return paths


def _tolerances(precision: str, *, log_probs: bool) -> dict[str, float]:
  """§7.2 ``pottsmpnn_full``: f64 atol 1e-8; f32 log-prob 1e-4."""
  if precision == "f64":
    return {"rtol": 0.0, "atol": 1e-8}
  if log_probs:
    return {"rtol": 0.0, "atol": 1e-4}
  return {"rtol": 1e-4, "atol": 1e-4}


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
  if array.ndim >= 2 and array.shape[0] == 1:
    return array[0]
  return array


def _merge_pair():
  from aminx.families.potts_mpnn import etab

  return etab.merge_pair


def _slot_alignment(
  pred_idx: np.ndarray,
  ref_idx: np.ndarray,
  present: np.ndarray,
  label: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Map reference slots onto predicted slots by neighbour id.

  Gap rows are all-tie (§4.1a, §6.5b ``gap_row_knn_tiebreak``), so kNN slot order
  is not comparable once gap rows exist. On present rows the neighbour *sets* must
  match exactly; edge arrays are then compared per neighbour id on edges whose two
  endpoints are present. Returns ``(perm, keep)`` with ``perm[i, k]`` the predicted
  slot holding ``ref_idx[i, k]``.
  """
  length, k = ref_idx.shape
  perm = np.tile(np.arange(k), (length, 1))
  for row in np.flatnonzero(present):
    pred_row = [int(v) for v in pred_idx[row]]
    ref_row = [int(v) for v in ref_idx[row]]
    assert sorted(pred_row) == sorted(ref_row), f"{label}: neighbour set differs at row {row}"
    lookup = {neighbour: slot for slot, neighbour in enumerate(pred_row)}
    perm[row] = [lookup[neighbour] for neighbour in ref_row]
  keep = present[:, None] & present[ref_idx]
  return perm, keep


def _aligned(pred: np.ndarray, perm: np.ndarray) -> np.ndarray:
  index = perm.reshape(perm.shape + (1,) * (pred.ndim - 2))
  return np.take_along_axis(pred, np.broadcast_to(index, perm.shape + pred.shape[2:]), axis=1)


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_pottsmpnn_full_wave(precision: str) -> None:
  torch = pytest.importorskip("torch")
  oracle_dir = _oracle_dir()
  npz_path = oracle_dir / "pottsmpnn_full" / f"oracle_{precision}.npz"
  if not npz_path.is_file():
    pytest.skip(f"{npz_path} absent")
  digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
  expected = _expected_sha(f"pottsmpnn_full/oracle_{precision}.npz")
  if digest != expected:
    pytest.fail(f"{npz_path.name} sha256 {digest} != {expected}")
  root = _potts_root()
  if not root.is_dir():
    pytest.skip(f"AMINX_POTTS_ROOT {root} absent")
  payload = np.load(npz_path)
  keys_text = (_REFERENCE / "keys.txt").read_text(encoding="utf-8")
  for stem in ("enc_h_V_0", "enc_h_E_0", "dec_h_V_0", "dec_h_E_0", "log_probs", "etab_forward"):
    assert stem in keys_text
  converter = _converter()
  merge = _merge_pair()
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", precision == "f64")
  dtype = jnp.float64 if precision == "f64" else jnp.float32
  compared = 0
  try:
    fixtures = [str(name) for name in payload["fixture_names"].tolist()]
    for checkpoint_id, relative in _checkpoint_paths().items():
      source = root / relative
      if not source.is_file():
        pytest.skip(f"checkpoint {source} absent")
      blob = torch.load(source, map_location="cpu", weights_only=False)
      state = (
        blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
      )
      model = converter.convert_state_dict(state)
      for fixture in fixtures:
        if fixture in _EXCLUDED_FIXTURES:
          continue
        prefix = f"{checkpoint_id}__{fixture}__"
        coords = jnp.asarray(_squeeze_batch(payload[prefix + "X"]), dtype=dtype)
        mask = jnp.asarray(_squeeze_batch(payload[prefix + "mask"]), dtype=dtype)
        residue_index = jnp.asarray(_squeeze_batch(payload[prefix + "residue_idx"]))
        chain_index = jnp.asarray(_squeeze_batch(payload[prefix + "chain_encoding"]))
        sequence = jnp.asarray(_squeeze_batch(payload[prefix + "S"]))
        decoding_order = jnp.asarray(_squeeze_batch(payload[prefix + "decoding_order"]))
        output = model(
          coords,
          mask,
          residue_index,
          chain_index,
          sequence,
          decoding_order,
        )
        label = f"{prefix}{precision}"
        present = _squeeze_batch(payload[prefix + "mask"]) > 0
        ref_idx = _squeeze_batch(payload[prefix + "E_idx"]).astype(np.int64)
        pred_idx = np.asarray(output.neighbor_indices).astype(np.int64)
        perm, keep = _slot_alignment(pred_idx, ref_idx, present, label)
        node_tol = _tolerances(precision, log_probs=False)
        for index, predicted in enumerate(output.enc_h_v):
          np.testing.assert_allclose(
            np.asarray(predicted)[present],
            _squeeze_batch(payload[prefix + f"enc_h_V_{index}"])[present],
            err_msg=f"{label} enc_h_V_{index}",
            **node_tol,
          )
        for index, predicted in enumerate(output.enc_h_e):
          np.testing.assert_allclose(
            _aligned(np.asarray(predicted), perm)[keep],
            _squeeze_batch(payload[prefix + f"enc_h_E_{index}"])[keep],
            err_msg=f"{label} enc_h_E_{index}",
            **node_tol,
          )
        for index, predicted in enumerate(output.dec_h_v):
          np.testing.assert_allclose(
            np.asarray(predicted)[present],
            _squeeze_batch(payload[prefix + f"dec_h_V_{index}"])[present],
            err_msg=f"{label} dec_h_V_{index}",
            **node_tol,
          )
        for index, predicted in enumerate(output.dec_h_e):
          np.testing.assert_allclose(
            _aligned(np.asarray(predicted), perm)[keep],
            _squeeze_batch(payload[prefix + f"dec_h_E_{index}"])[keep],
            err_msg=f"{label} dec_h_E_{index}",
            **node_tol,
          )
        np.testing.assert_allclose(
          np.asarray(output.log_probs)[present],
          _squeeze_batch(payload[prefix + "log_probs"])[present],
          err_msg=f"{label} log_probs",
          **_tolerances(precision, log_probs=True),
        )
        pad_valid = jnp.ones((coords.shape[0],), dtype=bool)
        etab_forward = merge(
          output.etab_raw,
          output.neighbor_indices,
          pad_valid,
          denom=2,
          exclude_self=False,
        )
        np.testing.assert_allclose(
          _aligned(np.asarray(etab_forward), perm)[keep],
          _squeeze_batch(payload[prefix + "etab_forward"])[keep],
          err_msg=f"{label} etab_forward",
          **node_tol,
        )
        compared += 1
  finally:
    jax.config.update("jax_enable_x64", previous)
  assert compared > 0
