"""FamilyDriver implementation for ThermoMPNN (v1) and ThermoMPNN-D (v2).

Supports ``score:ddg`` across single-checkpoint v1 (``thermo_mpnn_v1_default``)
and 3-member v2 ensembles (``thermo_mpnn_v2_ensemble`` for single-mutant SSM and
``thermo_mpnn_d_ensemble`` for epistatic double-mutant ΔΔG).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.thermo_mpnn.model import ThermoMPNN, ThermoMPNNDouble
from aminx.host.family_driver import FAMILY_DRIVERS

V2_SINGLE_ENSEMBLE_IDS: tuple[str, str, str] = (
  "thermo_mpnn_v2_ens1",
  "thermo_mpnn_v2_ens2",
  "thermo_mpnn_v2_ens3",
)

V2_DOUBLE_ENSEMBLE_IDS: tuple[str, str, str] = (
  "thermo_mpnn_d_ens1",
  "thermo_mpnn_d_ens2",
  "thermo_mpnn_d_ens3",
)


@eqx.filter_jit
def _score_single_ddg(
  model: ThermoMPNN,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
) -> Float[Array, "L 21"]:
  """Compute single-mutant ΔΔG matrix ``(L, 21)`` for a single ThermoMPNN model.

  Parameters
  ----------
  model : ThermoMPNN
    Single-mutant ThermoMPNN model instance.
  coords : Float[Array, "L 4 3"]
    Backbone ``(N, CA, C, O)`` coordinates.
  mask : Float[Array, " L"]
    Residue validity mask.
  residue_index : Int[Array, " L"]
    Residue index array.
  chain_index : Int[Array, " L"]
    Chain index array.
  sequence : Int[Array, " L"]
    Wild-type integer sequence in MPNN 21-letter alphabet.

  Returns
  -------
  Float[Array, "L 21"]
    Predicted ΔΔG matrix ``(L, 21)`` with wild-type self-mutations zeroed.
  """
  ddg_raw = model(coords, mask, residue_index, chain_index, sequence)
  wt_ddg = jnp.take_along_axis(ddg_raw, sequence[:, None], axis=-1)
  return (ddg_raw - wt_ddg) * mask[:, None]


@eqx.filter_jit
def _score_double_ddg(
  model: ThermoMPNNDouble,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
) -> tuple[Float[Array, "L 21"], Float[Array, "P 21 21"], Int[Array, "P 2"]]:
  """Compute single and double-mutant ΔΔG tensors for a ThermoMPNNDouble model.

  Parameters
  ----------
  model : ThermoMPNNDouble
    Double-mutant ThermoMPNN-D model instance.
  coords : Float[Array, "L 4 3"]
    Backbone ``(N, CA, C, O)`` coordinates.
  mask : Float[Array, " L"]
    Residue validity mask.
  residue_index : Int[Array, " L"]
    Residue index array.
  chain_index : Int[Array, " L"]
    Chain index array.
  sequence : Int[Array, " L"]
    Wild-type integer sequence in MPNN 21-letter alphabet.

  Returns
  -------
  tuple[Float[Array, "L 21"], Float[Array, "P 21 21"], Int[Array, "P 2"]]
    Tuple ``(ddg_single, ddg_double, pair_indices)``.
  """
  out = model(coords, mask, residue_index, chain_index, sequence)
  if isinstance(out, tuple):
    return out
  empty_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
  empty_double = jnp.zeros((0, 21, 21), dtype=jnp.float32)
  return out, empty_double, empty_pairs


class ThermoMPNNDriver:
  """Host ``FamilyDriver`` for ThermoMPNN (v1) and ThermoMPNN-D (v2).

  Attributes
  ----------
  family : str
    Registered family identifier (``"thermompnn"``).
  capabilities : tuple[str, ...]
    Supported capabilities (``("score:ddg",)``).
  """

  name: str = "thermompnn"
  family: str = "thermompnn"
  capabilities: tuple[str, ...] = ("score:ddg",)

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """Return True if purpose is supported by ThermoMPNNDriver.

    Parameters
    ----------
    spec : Any
      Run or sampling specification.
    purpose : str
      Requested capability identifier.

    Returns
    -------
    bool
      True when ``purpose`` is in ``self.capabilities``.
    """
    del spec
    return purpose in self.capabilities

  def sample(
    self,
    model: object,
    features: object,
    *,
    key: jax.Array | None = None,
    options: object | None = None,
  ) -> Mapping[str, Any]:
    """Refuse ``sample`` on ``ThermoMPNNDriver`` (ThermoMPNN is ``score:ddg`` only).

    Parameters
    ----------
    model : Any
      ThermoMPNN model or ensemble.
    features : Any
      Input structure features.
    key : jax.Array | None, optional
      PRNG key.
    options : Any | None, optional
      Driver options.

    Returns
    -------
    Mapping[str, Any]
      Always raises ``NotImplementedError``.
    """
    del model, features, key, options
    msg = "Family 'thermompnn' does not support 'sample'; supported capabilities: ('score:ddg',)."
    raise NotImplementedError(msg)

  def score(
    self,
    model_or_ensemble: ThermoMPNN | ThermoMPNNDouble | Sequence[ThermoMPNN | ThermoMPNNDouble],
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    *,
    mode: str = "score:ddg",
    double_mutant: bool = False,
  ) -> dict[str, jax.Array]:
    """Compute ΔΔG predictions for a single model or a 3-model ensemble.

    Parameters
    ----------
    model_or_ensemble : ThermoMPNN | ThermoMPNNDouble | Sequence[ThermoMPNN | ThermoMPNNDouble]
      Single model or sequence of ensemble member models.
    coords : Float[Array, "L 4 3"]
      Backbone ``(N, CA, C, O)`` coordinates.
    mask : Float[Array, " L"]
      Residue validity mask.
    residue_index : Int[Array, " L"]
      Residue index array.
    chain_index : Int[Array, " L"]
      Chain index array.
    sequence : Int[Array, " L"]
      Wild-type integer sequence ``(L,)``.
    mode : str, default="score:ddg"
      Scoring capability mode (must be ``"score:ddg"``).
    double_mutant : bool, default=False
      Whether to evaluate double-mutant epistatic ΔΔG.

    Returns
    -------
    dict[str, jax.Array]
      Dictionary containing ``"ddg"`` ``(L, 21)`` and, when ``double_mutant=True``,
      ``"ddg_double"`` ``(P, 21, 21)`` and ``"pair_indices"`` ``(P, 2)``.
    """
    if mode != "score:ddg":
      msg = f"Unsupported ThermoMPNN score mode: {mode!r}; expected 'score:ddg'."
      raise ValueError(msg)

    models: Sequence[Any] = (
      model_or_ensemble
      if isinstance(model_or_ensemble, Sequence)
      else (model_or_ensemble,)
    )
    if not models:
      msg = "ThermoMPNNDriver.score requires at least one model."
      raise ValueError(msg)

    inv_n = 1.0 / float(len(models))
    if double_mutant or isinstance(models[0], ThermoMPNNDouble):
      first_s, first_d, pair_idx = _score_double_ddg(
        models[0], coords, mask, residue_index, chain_index, sequence,
      )
      sum_single = first_s
      sum_double = first_d
      for m in models[1:]:
        s_ddg, d_ddg, pair_idx = _score_double_ddg(
          m, coords, mask, residue_index, chain_index, sequence,
        )
        sum_single = sum_single + s_ddg
        sum_double = sum_double + d_ddg
      return {
        "ddg": sum_single * inv_n,
        "ddg_double": sum_double * inv_n,
        "pair_indices": pair_idx,
      }

    sum_ddg = _score_single_ddg(
      models[0], coords, mask, residue_index, chain_index, sequence,
    )
    for m in models[1:]:
      sum_ddg = sum_ddg + _score_single_ddg(
        m, coords, mask, residue_index, chain_index, sequence,
      )
    return {"ddg": sum_ddg * inv_n}


def get_driver(*, family: str = "thermompnn") -> ThermoMPNNDriver:
  """Return a ``ThermoMPNNDriver`` instance for host dispatch.

  Parameters
  ----------
  family : str, default="thermompnn"
    Target family name.

  Returns
  -------
  ThermoMPNNDriver
    Instantiated driver for the ``"thermompnn"`` family.
  """
  del family
  return ThermoMPNNDriver()


FAMILY_DRIVERS.register("thermompnn")(ThermoMPNNDriver())
