# ruff: noqa: S101
"""Oracle parity for the A3 Potts waves.

Reads sealed dumps from ``AMINX_A1_ORACLE_DIR`` (default
``~/projects/aminx-oracles/dumps/a1_potts``). Missing dumps skip. A sha256
mismatch against ``tests/port/reference/a1_potts/oracles.sha256`` fails.
``potts_head`` loads ``etab_out`` from the torch checkpoint under
``AMINX_POTTS_ROOT`` and skips when torch is unavailable.

``knn_boundary_tie`` fixtures are dropped entirely. Etabs are compared on edges
whose endpoints are both present; edges incident to gap rows are the
``gap_row_knn_tiebreak`` set.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import merge_pair, model_to_etab, potts_energy
from aminx.families.potts_mpnn.featurize import knn_boundary_tie
from aminx.families.potts_mpnn.potts_head import PottsHead

_REF = Path(__file__).resolve().parents[2] / "port" / "reference" / "a1_potts"
_WAVES = ("potts_head", "potts_merge_pair_d2", "potts_merge_pair_d4", "potts_energy")
_TOLERANCE: dict[tuple[str, str], dict[str, float]] = {
  ("potts_head", "f64"): {"rtol": 1e-10, "atol": 1e-12},
  ("potts_head", "f32"): {"rtol": 1e-5, "atol": 1e-6},
  ("potts_merge_pair_d2", "f64"): {"rtol": 0.0, "atol": 0.0},
  ("potts_merge_pair_d2", "f32"): {"rtol": 0.0, "atol": 1e-7},
  ("potts_merge_pair_d4", "f64"): {"rtol": 0.0, "atol": 0.0},
  ("potts_merge_pair_d4", "f32"): {"rtol": 0.0, "atol": 1e-7},
  ("potts_energy", "f64"): {"rtol": 1e-10, "atol": 0.0},
  ("potts_energy", "f32"): {"rtol": 1e-5, "atol": 0.0},
}
# Spec r15: an f32 energy is a sum of O(L*K) terms that can cancel to near zero, so a pure
# rtol on the total is ill-posed. f32 energies use the summation backward-error bound
# |got - ref| <= rtol * sum|terms|, where sum|terms| = potts_energy(|etab|, seq).
_ENERGY_F32_COND_RTOL = 1e-5


def _oracle_root() -> Path:
  raw = os.environ.get("AMINX_A1_ORACLE_DIR", "~/projects/aminx-oracles/dumps/a1_potts")
  return Path(raw).expanduser()


def _sha_table() -> dict[str, str]:
  table: dict[str, str] = {}
  for line in (_REF / "oracles.sha256").read_text().splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    digest, name = stripped.split(maxsplit=1)
    table[name.removeprefix("./")] = digest
  return table


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_oracle(wave: str, precision: str) -> np.lib.npyio.NpzFile:
  path = _oracle_root() / wave / f"oracle_{precision}.npz"
  if not path.is_file():
    pytest.skip(f"oracle dump absent: {path}")
  expected = _sha_table().get(f"{wave}/oracle_{precision}.npz")
  if expected is None:
    pytest.fail(f"no sha256 entry for {wave}/oracle_{precision}.npz")
  actual = _file_sha256(path)
  if actual != expected:
    pytest.fail(f"sha256 mismatch for {path}: {actual} != {expected}")
  return np.load(path)


def _names(values: np.ndarray) -> list[str]:
  return [str(item) for item in values.tolist()]


def _excluded_fixture(mask: np.ndarray) -> bool:
  present = np.asarray(mask[0])
  return knn_boundary_tie(present, int(present.shape[0]))


def _kept_edges(mask: np.ndarray, e_idx: np.ndarray) -> np.ndarray:
  present = np.asarray(mask[0]) > 0
  neighbours = np.asarray(e_idx[0])
  return present[:, None] & present[neighbours]


def _compare(got: np.ndarray, ref: np.ndarray, wave: str, precision: str, label: str) -> None:
  assert got.shape == ref.shape, f"{label}: {got.shape} vs {ref.shape}"
  assert got.dtype == ref.dtype, f"{label}: {got.dtype} vs {ref.dtype}"
  np.testing.assert_allclose(got, ref, err_msg=label, **_TOLERANCE[(wave, precision)])


def _compare_conditioned(got: np.ndarray, ref: np.ndarray, scale: np.ndarray, label: str) -> None:
  assert got.shape == ref.shape, f"{label}: {got.shape} vs {ref.shape}"
  assert got.dtype == ref.dtype, f"{label}: {got.dtype} vs {ref.dtype}"
  error = np.abs(got.astype(np.float64) - ref.astype(np.float64))
  bound = _ENERGY_F32_COND_RTOL * np.abs(scale.astype(np.float64))
  worst = int(np.argmax(error - bound))
  assert np.all(error <= bound), (
    f"{label}: |got-ref| {error.flat[worst]:.3e} > {bound.flat[worst]:.3e} (rtol*sum|terms|)"
  )


def _checkpoint_paths() -> dict[str, str]:
  manifest = tomllib.loads((_REF / "oracle_manifest.toml").read_text())
  return {str(row["id"]): str(row["path"]) for row in manifest["checkpoint"]}


def _etab_out(checkpoint_id: str, precision: str) -> tuple[jax.Array, jax.Array]:
  torch = pytest.importorskip("torch")
  root = Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser()
  if not root.is_dir():
    pytest.skip(f"PottsMPNN checkout absent: {root}")
  relative = _checkpoint_paths()[f"{checkpoint_id}_{precision}"]
  path = root / relative
  if not path.is_file():
    pytest.skip(f"checkpoint absent: {path}")
  blob = torch.load(path, map_location="cpu", weights_only=False)
  state = blob["model_state_dict"]
  np_dtype = np.float64 if precision == "f64" else np.float32
  weight = np.asarray(state["etab_out.weight"].detach().cpu().numpy(), dtype=np_dtype)
  bias = np.asarray(state["etab_out.bias"].detach().cpu().numpy(), dtype=np_dtype)
  return jnp.asarray(weight), jnp.asarray(bias)


def _pad_valid(length: int) -> jax.Array:
  return jnp.ones((length,), dtype=bool)


def _enable_x64(precision: str) -> contextlib.AbstractContextManager[None]:
  if precision != "f64":
    return contextlib.nullcontext()
  enable = getattr(jax.experimental, "enable_x64", None)
  if enable is None:
    enable = jax.enable_x64
  return enable()


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_potts_head_wave(precision: str) -> None:
  data = _load_oracle("potts_head", precision)
  compared = 0
  weights: dict[str, tuple[jax.Array, jax.Array]] = {}
  with _enable_x64(precision), jax.default_matmul_precision("highest"):
    for checkpoint in _names(data["checkpoint_ids"]):
      for fixture in _names(data["fixture_names"]):
        prefix = f"{checkpoint}__{fixture}__"
        mask = data[prefix + "mask"]
        e_idx = data[prefix + "E_idx"]
        if _excluded_fixture(mask):
          continue
        keep = _kept_edges(mask, e_idx)
        if not bool(keep.any()):
          continue
        if checkpoint not in weights:
          weights[checkpoint] = _etab_out(checkpoint, precision)
        weight, bias = weights[checkpoint]
        h_e = jnp.asarray(data[prefix + "h_E"][0])
        weight = weight.astype(h_e.dtype)
        bias = bias.astype(h_e.dtype)
        head = PottsHead(int(h_e.shape[-1]), key=jax.random.key(0)).with_weights(weight, bias)
        present = jnp.asarray(mask[0])
        pred = np.asarray(head(h_e, jnp.asarray(e_idx[0]), present, _pad_valid(h_e.shape[0])))
        ref = np.asarray(data[prefix + "etab_raw"][0])
        label = f"potts_head {precision} {checkpoint} {fixture}"
        _compare(pred[keep], ref[keep], "potts_head", precision, label)
        compared += 1
  assert compared > 0


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_potts_merge_pair_d2_wave(precision: str) -> None:
  _assert_merge(precision, wave="potts_merge_pair_d2", source="etab_raw", target="etab_forward")


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_potts_merge_pair_d4_wave(precision: str) -> None:
  _assert_merge(
    precision,
    wave="potts_merge_pair_d4",
    source="etab_forward",
    target="etab_energy",
  )


def _assert_merge(precision: str, *, wave: str, source: str, target: str) -> None:
  data = _load_oracle(wave, precision)
  denom = 2 if wave.endswith("d2") else 4
  exclude_self = wave.endswith("d4")
  compared = 0
  with _enable_x64(precision):
    for checkpoint in _names(data["checkpoint_ids"]):
      for fixture in _names(data["fixture_names"]):
        prefix = f"{checkpoint}__{fixture}__"
        mask = data[prefix + "mask"]
        e_idx = data[prefix + "E_idx"]
        if _excluded_fixture(mask):
          continue
        keep = _kept_edges(mask, e_idx)
        if not bool(keep.any()):
          continue
        table = jnp.asarray(data[prefix + source][0])
        pred = np.asarray(
          merge_pair(
            table,
            jnp.asarray(e_idx[0]),
            _pad_valid(table.shape[0]),
            denom=denom,
            exclude_self=exclude_self,
          ),
        )
        ref = np.asarray(data[prefix + target][0])
        _compare(
          pred[keep], ref[keep], wave, precision, f"{wave} {precision} {checkpoint} {fixture}"
        )
        compared += 1
  assert compared > 0


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_potts_energy_wave(precision: str) -> None:
  data = _load_oracle("potts_energy", precision)
  compared = 0
  with _enable_x64(precision):
    for checkpoint in _names(data["checkpoint_ids"]):
      for fixture in _names(data["fixture_names"]):
        prefix = f"{checkpoint}__{fixture}__"
        mask = data[prefix + "mask"]
        if _excluded_fixture(mask):
          continue
        e_idx = jnp.asarray(data[prefix + "E_idx"][0])
        etab = jnp.asarray(data[prefix + "etab_energy_padded"][0])
        pad_valid = _pad_valid(etab.shape[0])
        native = model_to_etab(jnp.asarray(data[prefix + "seq_native"]))
        random = model_to_etab(jnp.asarray(data[prefix + "seq_random"]))
        got_native = np.asarray(potts_energy(etab, e_idx, pad_valid, native))
        got_random = np.asarray(potts_energy(etab, e_idx, pad_valid, random))
        ref_native = np.asarray(data[prefix + "energy_native"])
        ref_random = np.asarray(data[prefix + "energy_random"])
        label = f"potts_energy {precision} {checkpoint} {fixture}"
        if precision == "f32":
          abs_etab = jnp.abs(etab)
          scale_native = np.asarray(potts_energy(abs_etab, e_idx, pad_valid, native))
          scale_random = np.asarray(potts_energy(abs_etab, e_idx, pad_valid, random))
          _compare_conditioned(got_native, ref_native, scale_native, label + " native")
          _compare_conditioned(
            got_random.reshape(ref_random.shape),
            ref_random,
            scale_random.reshape(ref_random.shape),
            label + " random",
          )
        else:
          _compare(got_native, ref_native, "potts_energy", precision, label + " native")
          _compare(
            got_random.reshape(ref_random.shape),
            ref_random,
            "potts_energy",
            precision,
            label + " random",
          )
        compared += 1
  assert compared > 0
