"""Contract tests for the FAMPNN driver that need no checkpoint weights.

These pin the driver's own decisions: which checkpoint id a run resolves, which spec fields it
refuses rather than dropping, and that the logits it returns belong to the sequence it returns.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aminx.families.fampnn.driver as driver_mod
from aminx.families.fampnn.constants import FAMPNN_TO_AMINX_ARR
from aminx.families.fampnn.driver import FAMPNNDriver, mar_sample
from aminx.run.options import FAMPNNOptions


def _spec(**fields: Any) -> SimpleNamespace:  # noqa: ANN401
  base: dict[str, Any] = {
    "checkpoint_id": None,
    "model_local_path": None,
    "checkpoint_registry_path": None,
    "fampnn": None,
    "fixed_positions": None,
    "fixed_mask": None,
    "fixed_tokens": None,
    "bias": None,
    "omit_aa": (),
    "omit_aa_per_position": None,
  }
  base.update(fields)
  return SimpleNamespace(**base)


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
  """Stub the artifact resolver and the loader; record the checkpoint id each one was given."""
  calls: dict[str, Any] = {}

  def fake_resolve(spec: Any, family: str, checkpoint_id: str | None = None) -> str:  # noqa: ANN401, ARG001
    calls["resolved_id"] = checkpoint_id
    return "/nowhere/fampnn.eqx.zst"

  def fake_load(path: Any, checkpoint_id: str) -> str:  # noqa: ANN401
    calls["loaded_id"] = checkpoint_id
    return "MODEL"

  monkeypatch.setattr(driver_mod, "_resolve_driver_checkpoint", fake_resolve)
  monkeypatch.setattr(driver_mod, "_load_fampnn_eqx", fake_load)
  return calls


def test_default_options_and_unset_spec_resolve_the_default_checkpoint(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(fampnn=FAMPNNOptions()))

  assert resolved["resolved_id"] == "fampnn_0_3"
  assert resolved["loaded_id"] == "fampnn_0_3"


def test_options_checkpoint_id_is_used_when_spec_is_unset(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_0")))

  assert resolved["resolved_id"] == "fampnn_0_0"
  assert resolved["loaded_id"] == "fampnn_0_0"


def test_spec_checkpoint_id_is_used_when_options_are_unset(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(checkpoint_id="fampnn_0_3_cath", fampnn=FAMPNNOptions(checkpoint_id=None)))

  assert resolved["resolved_id"] == "fampnn_0_3_cath"
  assert resolved["loaded_id"] == "fampnn_0_3_cath"


def test_equal_spec_and_options_checkpoint_ids_are_accepted(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(
    _spec(checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_0")),
  )

  assert resolved["loaded_id"] == "fampnn_0_0"


def test_conflicting_spec_and_options_checkpoint_ids_raise(resolved: dict[str, Any]) -> None:
  with pytest.raises(ValueError, match="conflicting FAMPNN checkpoint ids"):
    FAMPNNDriver().load(
      _spec(checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_3_cath")),
    )

  assert "loaded_id" not in resolved


@pytest.mark.parametrize(
  "fields",
  [
    {"fixed_positions": [1, 2]},
    {"fixed_tokens": [3]},
    {"fixed_mask": [0, 1]},
    {"bias": [[0.0] * 21]},
    {"omit_aa": ("C",)},
    {"omit_aa_per_position": {3: "W"}},
  ],
  ids=["fixed_positions", "fixed_tokens", "fixed_mask", "bias", "omit_aa", "omit_aa_per_position"],
)
def test_unsupported_constraint_fields_raise_instead_of_being_dropped(
  resolved: dict[str, Any], fields: dict[str, Any],
) -> None:
  with pytest.raises(ValueError, match="does not apply"):
    FAMPNNDriver().load(_spec(**fields))

  assert "loaded_id" not in resolved


def test_empty_constraint_fields_are_accepted(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(omit_aa=(), omit_aa_per_position=None, fixed_positions=None))

  assert resolved["loaded_id"] == "fampnn_0_3"


@pytest.mark.parametrize("purpose", ["score:nll", "score:logits"])
def test_score_purposes_refuse_sequences_to_score(purpose: str) -> None:
  """Scoring returns the wild-type sequence only, so candidates would be silently dropped.

  Parameters
  ----------
  purpose : str
    Scoring purpose that must refuse ``sequences_to_score``.

  Returns
  -------
  None
    Asserts the refusal.
  """
  with pytest.raises(ValueError, match="does not apply sequences_to_score"):
    FAMPNNDriver().stages(_spec(sequences_to_score=("ACDE",)), purpose, model=None)  # type: ignore[arg-type]


class _GetcwdPayload:
  """Pickles as a call to ``os.getcwd``, a harmless stand-in for a hostile ``__reduce__``."""

  def __reduce__(self) -> tuple[Any, ...]:
    """Return the callable and arguments the unpickler would invoke.

    Parameters
    ----------
    self : _GetcwdPayload
      Instance being pickled.

    Returns
    -------
    tuple[Any, ...]
      ``(os.getcwd, ())``.
    """
    import os  # noqa: PLC0415

    return (os.getcwd, ())


def test_fampnn_torch_checkpoint_refuses_a_reduce_payload(tmp_path: Path) -> None:
  """A FAMPNN ``.pt`` loads through the restricted unpickler, so a ``__reduce__`` global is refused.

  Parameters
  ----------
  tmp_path : Path
    Pytest temporary directory that holds the hostile checkpoint.

  Returns
  -------
  None
    Asserts the refusal.
  """
  torch = pytest.importorskip("torch")
  import pickle  # noqa: PLC0415

  path = tmp_path / "hostile.pt"
  torch.save({"state_dict": _GetcwdPayload()}, path)

  with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
    driver_mod._load_fampnn_torch(path)


class _FakeOut(NamedTuple):
  logits_fampnn: jax.Array
  logits_aminx: jax.Array
  h_v: jax.Array
  h_e: jax.Array
  e_idx: jax.Array


class _FakeFAMPNN:
  """Deterministic stand-in for the network: logits peak on the residue each position holds.

  A masked position holds the unknown token (index 20), so its logits carry no residue identity.
  """

  scn_diffusion = None
  scn_confidence = None

  def __call__(
    self,
    xyz: jax.Array,
    atom_mask: jax.Array,
    aatype: jax.Array,
    seq_mask: jax.Array,
    mask: jax.Array,
    residue_idx: jax.Array,
    chain_idx: jax.Array,
  ) -> _FakeOut:
    del xyz, atom_mask, mask, residue_idx, chain_idx
    held = jnp.where(seq_mask > 0, jnp.clip(aatype, 0, 20), 20)
    fampnn_logits = 50.0 * jax.nn.one_hot(held, 21, dtype=jnp.float32)
    aminx_logits = 50.0 * jax.nn.one_hot(jnp.asarray(FAMPNN_TO_AMINX_ARR)[held], 21, dtype=jnp.float32)
    zeros = jnp.zeros((held.shape[0], 1), dtype=jnp.float32)
    return _FakeOut(
      logits_fampnn=fampnn_logits,
      logits_aminx=aminx_logits,
      h_v=zeros,
      h_e=zeros,
      e_idx=zeros,
    )


def test_mar_sample_returns_logits_for_the_returned_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
  """The returned logits must describe the returned sequence, not the state before the last step.

  With two steps the cosine schedule unmasks every designable position on the last step, so the
  loop's own last forward pass saw a fully masked tail. Its argmax would not match the output.
  """
  monkeypatch.setattr(driver_mod, "edm_sample", lambda model, xyz, *args, **kwargs: xyz)  # noqa: ARG005
  length = 6
  seq_aminx, logits_aminx, _, _ = mar_sample(
    _FakeFAMPNN(),
    jnp.zeros((length, 37, 3), dtype=jnp.float32),
    jnp.zeros((length, 37), dtype=jnp.float32),
    jnp.zeros((length,), dtype=jnp.int32),
    jnp.zeros((length,), dtype=jnp.float32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.arange(1, length + 1, dtype=jnp.int32),
    jnp.zeros((length,), dtype=jnp.int32),
    num_seq_steps=2,
    temperature=1.0,
    pack_each_step=False,
    key=jax.random.PRNGKey(0),
  )

  np.testing.assert_array_equal(np.asarray(jnp.argmax(logits_aminx, axis=-1)), np.asarray(seq_aminx))
