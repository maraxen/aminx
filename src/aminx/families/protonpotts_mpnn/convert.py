"""ProtonPottsMPNN v6 checkpoint -> aminx weight layout: renames, permutations, shape checks.

Spec: ``.praxia/docs/specs/261007_protonpottsmpnn-support.md`` §14, §15, §34.1, §41.

The v6 checkpoint was trained in FOUNDRY's native layout, and aminx uses ProteinMPNN's legacy layout, so
a key-complete conversion is numerically wrong (spec §34.1). Three orderings differ, none of which
raises a shape error:

1. ``graph_featurization_module.edge_embedding.weight`` ``(128, 16 + 25*16)``: the 25 atom-pair RBF blocks
   are in foundry's row-major order over atoms ``N, Ca, C, O, Cb`` (``5*i + j``); aminx's
   ``BACKBONE_PAIRS`` is ProteinMPNN's legacy order. aminx and foundry index the atoms identically
   (0=N, 1=Ca, 2=C, 3=O, 4=Cb), so aminx slot ``s`` holding pair ``(i, j)`` takes foundry block ``5*i + j``.
2. Token rows of ``W_s.weight`` and ``W_out.{weight,bias}``: foundry orders the 20 residues by three-letter
   name (``ARND...``), aminx by one-letter code (``ACDEF...``); the nine protonation tokens follow both.
3. ``etab_out.{weight,bias}`` ``(900, 128)`` / ``(900,)``: the 900 outputs are a ``30 x 30`` token-pair
   table (``a * 30 + b``), so BOTH token axes take the permutation in (2).

Nothing here reads a checkpoint or imports torch or atomworks: it works on numpy arrays, so it can be unit
tested without the oracle environment. The end-to-end check that the conversion is numerically right is
the ``protonpotts_encoder`` comparison against upstream, not these tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, upstream_to_aminx_index
from aminx.utils.radial_basis import BACKBONE_PAIRS

N_POSITIONAL = 16
N_RBF = 16
N_ATOM_PAIRS = 25
EDGE_EMBEDDING_SHAPE = (128, N_POSITIONAL + N_ATOM_PAIRS * N_RBF)
V = PROTONPOTTS_V6.size

# foundry key -> the name aminx's converter expects (spec §34.1: five renames)
RENAMES: dict[str, str] = {
  "graph_featurization_module.positional_embedding.embed_positional_features.weight": "features.embeddings.linear.weight",
  "graph_featurization_module.positional_embedding.embed_positional_features.bias": "features.embeddings.linear.bias",
  "graph_featurization_module.edge_embedding.weight": "features.edge_embedding.weight",
  "graph_featurization_module.edge_norm.weight": "features.norm_edges.weight",
  "graph_featurization_module.edge_norm.bias": "features.norm_edges.bias",
}

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
  "graph_featurization_module.positional_embedding.embed_positional_features.weight": (16, 66),
  "graph_featurization_module.positional_embedding.embed_positional_features.bias": (16,),
  "graph_featurization_module.edge_embedding.weight": EDGE_EMBEDDING_SHAPE,
  "graph_featurization_module.edge_norm.weight": (128,),
  "graph_featurization_module.edge_norm.bias": (128,),
  "W_s.weight": (V, 128),
  "W_out.weight": (V, 128),
  "W_out.bias": (V,),
  "etab_out.weight": (V * V, 128),
  "etab_out.bias": (V * V,),
}


def pair_permutation() -> np.ndarray:
  """``perm[s]`` is the FOUNDRY pair block that aminx's atom-pair slot ``s`` reads."""
  pairs = np.asarray(BACKBONE_PAIRS)
  return (5 * pairs[:, 0] + pairs[:, 1]).astype(np.int64)


def token_permutation() -> np.ndarray:
  """``take[k]`` is the FOUNDRY index of the token that sits at aminx index ``k``."""
  to_aminx = np.asarray([upstream_to_aminx_index(u) for u in range(V)], dtype=np.int64)
  take = np.empty(V, dtype=np.int64)
  take[to_aminx] = np.arange(V)
  return take


def permute_edge_embedding(weight: np.ndarray) -> np.ndarray:
  """Reorder the 25 atom-pair RBF column blocks from foundry's order to aminx's."""
  if weight.shape != EDGE_EMBEDDING_SHAPE:
    msg = f"edge_embedding.weight is {weight.shape}, expected {EDGE_EMBEDDING_SHAPE}"
    raise ValueError(msg)
  positional = weight[:, :N_POSITIONAL]
  rbf = weight[:, N_POSITIONAL:].reshape(weight.shape[0], N_ATOM_PAIRS, N_RBF)
  return np.concatenate(
    [positional, rbf[:, pair_permutation(), :].reshape(weight.shape[0], -1)], axis=1,
  )


def permute_token_rows(array: np.ndarray) -> np.ndarray:
  """Reorder a ``(V, ...)`` array whose first axis is tokens (``W_s``, ``W_out``)."""
  if array.shape[0] != V:
    msg = f"expected {V} token rows, got shape {array.shape}"
    raise ValueError(msg)
  return array[token_permutation()]


def permute_etab(array: np.ndarray) -> np.ndarray:
  """Reorder ``etab_out`` ``(V*V, ...)`` on BOTH token axes of the ``V x V`` pair table."""
  if array.shape[0] != V * V:
    msg = f"expected {V * V} etab rows, got shape {array.shape}"
    raise ValueError(msg)
  take = token_permutation()
  table = array.reshape(V, V, *array.shape[1:])
  return table[take][:, take].reshape(array.shape)


def to_aminx_layout(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
  """A foundry-layout state dict as numpy arrays -> the names and orderings aminx's converter expects.

  Every expected shape is checked first, because none of the three differences raises one.
  """
  for key, shape in EXPECTED_SHAPES.items():
    if key not in state:
      msg = f"checkpoint has no {key!r}"
      raise KeyError(msg)
    if tuple(state[key].shape) != shape:
      msg = f"{key} has shape {tuple(state[key].shape)}, expected {shape}"
      raise ValueError(msg)

  out = {RENAMES.get(key, key): np.asarray(value) for key, value in state.items()}
  out["features.edge_embedding.weight"] = permute_edge_embedding(
    out["features.edge_embedding.weight"],
  )
  out["W_s.weight"] = permute_token_rows(out["W_s.weight"])
  out["W_out.weight"] = permute_token_rows(out["W_out.weight"])
  out["W_out.bias"] = permute_token_rows(out["W_out.bias"])
  out["etab_out.weight"] = permute_etab(out["etab_out.weight"])
  out["etab_out.bias"] = permute_etab(out["etab_out.bias"])
  return out
