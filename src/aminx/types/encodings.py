"""Encoder output container."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array, Float, Int


class EncoderOutput(NamedTuple):
  """Container for encoder output features.

  A frozen, JAX-pytree-compatible named tuple holding the three key outputs
  from the encoder: node features, edge features, and neighbor indices.

  This structure enables clean composition in scoring and sampling kernels.
  """

  node_features: Float[Array, "L D"]
  edge_features: Float[Array, "L K D"]
  neighbor_indices: Int[Array, "L K"]
  mask: Float[Array, "L"] | None = None
