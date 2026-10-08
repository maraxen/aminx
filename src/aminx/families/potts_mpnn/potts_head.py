"""Potts pair-energy head.

Ports the ``etab_out`` projection in ``potts_mpnn_utils.py`` (``run_encoder`` /
``forward``, lines 1282-1286 and 1793-1797) with the §4.1a edge mask.

Math:
    $$W \\in \\mathbb{R}^{400 \\times H},\\;
      T_{ik} = \\mathrm{reshape}_{20,20}(W h_{ik} + b)$$
    $$m_{ik} = \\mathrm{present}_i \\cdot \\mathrm{pad\\_valid}_{E_{ik}}$$
    $$\\tilde T_{ikab} = m_{ik} T_{ikab},\\quad
      \\tilde T_{i0} \\leftarrow \\tilde T_{i0} \\odot I_{20}$$

Pseudocode:
    tables = reshape(linear(edge_features), (L, K, 20, 20))
    tables *= present[:, None] * pad_valid[E_idx]
    tables[:, 0] *= eye(20)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import typing as jxtyping
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from aminx.families.potts_mpnn.alphabet import POTTS_MPNN

N_AA = POTTS_MPNN.pair_side
PAIR_DIM = POTTS_MPNN.pair_dim


class PottsHead(eqx.Module):
  """Linear map from edge features to a masked ``(L, K, 20, 20)`` pair table.

  Slot 0 is the self interaction: off-diagonal entries are cleared with
  ``eye(20)`` after the edge mask, matching upstream.
  """

  linear: eqx.nn.Linear

  def __init__(self, hidden_dim: int, *, key: PRNGKeyArray) -> None:
    """Initialize the ``H -> 400`` projection."""
    self.linear = eqx.nn.Linear(hidden_dim, PAIR_DIM, key=key)

  def with_weights(
    self,
    weight: Float[Array, "400 H"],
    bias: Float[Array, " 400"],
  ) -> PottsHead:
    """Return a copy whose projection matches a loaded ``etab_out`` layer.

    ``weight`` uses the PyTorch ``nn.Linear`` layout ``(400, H)``.
    """
    linear = eqx.tree_at(lambda layer: layer.weight, self.linear, weight)
    linear = eqx.tree_at(lambda layer: layer.bias, linear, bias)
    return eqx.tree_at(lambda head: head.linear, self, linear)

  def __call__(
    self,
    edge_features: Float[Array, "L K H"],
    e_idx: Int[Array, "L K"],
    present: Float[Array, " L"],
    pad_valid: Bool[Array, " L"],
  ) -> Float[Array, "L K 20 20"]:
    """Project edge features and apply the PottsHead mask.

    Parameters
    ----------
    edge_features:
        Encoder edge states ``h_E``, shape ``(L, K, H)``.
    e_idx:
        Neighbour indices, shape ``(L, K)``. Upstream name ``E_idx``.
    present:
        Row mask. Upstream multiplies by this alone; padding additionally
        requires ``pad_valid`` at the neighbour.
    pad_valid:
        True on real A0 rows, false on pad rows.

    Returns
    -------
    Raw pair table ``etab_raw`` of shape ``(L, K, 20, 20)``, before
    ``merge_pair``. Dtype matches ``edge_features``.
    """
    bias = self.linear.bias
    if bias is None:
      msg = "PottsHead requires a biased linear map"
      raise ValueError(msg)
    projected = edge_features @ self.linear.weight.T + bias
    length, k, _ = edge_features.shape
    tables = projected.reshape(length, k, N_AA, N_AA)
    scale = _edge_scale(e_idx, present, pad_valid, tables.dtype)
    tables = tables * scale[..., None, None]
    eye = jnp.eye(N_AA, dtype=tables.dtype)
    slot0 = tables[:, 0] * eye
    return tables.at[:, 0].set(slot0)


def _edge_scale(
  e_idx: Int[Array, "L K"],
  present: Float[Array, " L"],
  pad_valid: Bool[Array, " L"],
  dtype: jxtyping.DTypeLike,
) -> Float[Array, "L K"]:
  """``present[i] * pad_valid[E_idx[i, k]]`` in the table dtype."""
  neighbour_valid = pad_valid[e_idx].astype(dtype)
  return present.astype(dtype)[:, None] * neighbour_valid
