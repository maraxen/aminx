"""Scalar-dropout replay must fail closed when a recorded mask is missing.

Upstream ``_shim_dropout`` raises ``no injected dropout mask for {path} call
{call}``. The proofread cursor used to substitute ones and still divide by
``1 - p``, which scales the activation by ``1/0.9`` and looks like a replay.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.laser.decoder import LaserDecoder
from aminx.model.laser.encoders import LaserEncoder
from aminx.model.laser.graphs import GraphStructure
from aminx.model.laser.joint_decode import LaserJointDecode
from aminx.model.laser.proofread import (
    _Plan,
    _fit_keep,
    conditional_focus_probs,
    proofread_dropout,
)


@functools.cache
def _models() -> tuple[LaserEncoder, LaserDecoder, LaserJointDecode]:
  return (
    LaserEncoder(key=jax.random.PRNGKey(0)),
    LaserDecoder(key=jax.random.PRNGKey(1)),
    LaserJointDecode(key=jax.random.PRNGKey(2)),
  )


def _coords(n_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  backbone = np.zeros((n_res, 5, 3), dtype=np.float32)
  for index in range(n_res):
    backbone[index, :, 0] = index * 3.8
    backbone[index, 1, 1] = 1.5
  ligand = np.asarray([[0.0, 8.0, 0.0]], dtype=np.float32)
  atomic = np.asarray([6], dtype=np.int32)
  subbatch = np.asarray([0], dtype=np.int32)
  return backbone, ligand, atomic, subbatch


def _expected_calls(n_res: int) -> dict[str, int]:
  """Call counts upstream records: one shared module, no ``subgats`` paths.

  Hetero subgraphs are built with ``update_edges=False``, so
  ``HomoGATv2.dropout`` on ``subgats`` is never invoked. Ligand GAT layers 0
  and 1 apply that module twice (attention, then the edge residual). Each
  residual node update and each chi MLP dropout fires on its own module.
  """
  counts = {
    "ligand_encoder.gat_layers.0.dropout": 2,
    "ligand_encoder.gat_layers.1.dropout": 2,
    "ligand_encoder.gat_layers.2.dropout": 1,
    "ligand_encoder_output_gvp.dropout.sdropout": 2,
  }
  for index in range(3):
    counts[f"protein_encoder_layers.{index}.hetgat.dropout"] = 1
    counts[f"protein_encoder_layers.{index}.hetgat.dense_residual_node_update.dropout"] = 2
    counts[f"protein_decoder_layers.{index}.hetgat.dropout"] = n_res
    counts[f"protein_decoder_layers.{index}.hetgat.dense_residual_node_update.dropout"] = 2 * n_res
  for index in range(4):
    counts[f"chi_prediction_layers.{index}.layers.1"] = n_res
    counts[f"chi_prediction_layers.{index}.layers.4"] = n_res
    counts[f"chi_offset_prediction_layers.{index}.layers.1"] = n_res
    counts[f"chi_offset_prediction_layers.{index}.layers.4"] = n_res
  return counts


def test_incomplete_dropout_mask_raises() -> None:
  """A path-or-call miss must raise rather than scale by ones / 0.9.

  The observable that would have caught this defect is ``RuntimeError`` naming
  the missing path and call index. Before the change, ``_Plan.take`` returned
  ones and ``_Plan.scale`` still divided by ``1 - p``, so a total replay
  failure was indistinguishable from a correct decode.
  """
  encoder, decoder, joint = _models()
  value = jnp.ones((2, 3))
  offsets = tuple(joint.chi_offset_prediction_layers)
  with pytest.raises(RuntimeError, match="no injected dropout mask for missing call 0"):
    with proofread_dropout(
      encoder,
      decoder,
      offsets,
      scalar=True,
      masks={"known": [np.ones((2, 3), dtype=np.float32)]},
    ) as plan:
      plan.scale("missing", value)
  with pytest.raises(RuntimeError, match="known call 1: consumed 1, recorded 2"):
    with proofread_dropout(
      encoder,
      decoder,
      offsets,
      scalar=True,
      masks={
        "known": [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)],
      },
    ) as plan:
      plan.scale("known", value)


def test_dropout_replay_reconciles_one_decode(monkeypatch: pytest.MonkeyPatch) -> None:
  """One eager decode consumes each recorded mask once and never asks for subgats.

  The observable that would have caught this defect is ``RuntimeError: no
  injected dropout mask for homo.dropout call 0``. ``filter_jit`` rebuilds
  modules and used to drop the non-field path stamp, so the first ligand GAT
  dropout looked up a name upstream never recorded and the ones-fallback
  scaled that activation by ``1/0.9``.
  """
  n_res = 2
  encoder, decoder, joint = _models()
  backbone, ligand, atomic, sub = _coords(n_res)
  period = np.zeros((118,), dtype=np.int32)
  sequence = np.ones((n_res,), dtype=np.int32)
  chi = np.zeros((n_res, 4), dtype=np.float32)
  expected = _expected_calls(n_res)
  masks = {
    path: [np.ones((1,), dtype=np.float32) for _ in range(count)]
    for path, count in expected.items()
  }
  captured = _Capture()
  original = _Plan.reconcile

  def _spy(self: _Plan) -> None:
    captured.lookups = self.subgat_dropout_lookups
    captured.table = self.consumption()
    original(self)

  monkeypatch.setattr(_Plan, "reconcile", _spy)
  conditional_focus_probs(
    encoder,
    decoder,
    joint,
    backbone,
    ligand,
    atomic,
    sub,
    period,
    period,
    GraphStructure(
      pr_pr_knn_graph_k=4,
      lig_pr_knn_graph_k=4,
      lig_lig_knn_graph_k=2,
      lig_pr_distance_cutoff=20.0,
    ),
    sequence,
    chi,
    0,
    [np.arange(n_res, dtype=np.int32)],
    [[masks]],
    [[np.full((n_res * 5 + 4,), 0.5, dtype=np.float64)]],
    scalar=True,
    repack_all=True,
  )
  assert captured.lookups == 0
  got = {path: consumed for path, (consumed, _recorded) in captured.table.items()}
  assert got == expected


class _Capture:
  def __init__(self) -> None:
    self.lookups = -1
    self.table: dict[str, tuple[int, int]] = {}


def _prefix_fill(keep: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
  """The placement ``_fit_keep`` used to invent: leading C-order slots."""
  flat = np.ones(int(np.prod(shape)), dtype=np.float32)
  values = np.reshape(keep, (-1,)).astype(np.float32, copy=False)
  flat[: values.size] = values
  return flat.reshape(shape)


def test_fit_keep_rejects_a_shorter_edge_list() -> None:
  """A shorter keep is not a prefix of a padded neighbour axis.

  MLP masks match element-for-element, so they still reshape. A GAT mask
  whose edge count is the leading part of ``(node, slot)`` must not.
  """
  with pytest.raises(RuntimeError, match="element-for-element"):
    _fit_keep(jnp.asarray([0.0, 1.0, 0.0]), jnp.ones((3, 2)))
  reshaped = _fit_keep(jnp.asarray([[0.0, 1.0, 0.0, 1.0]]), jnp.ones((4,)))
  np.testing.assert_array_equal(np.asarray(reshaped), np.asarray([0.0, 1.0, 0.0, 1.0]))


def test_gat_keep_scatters_by_edge_identity_not_prefix() -> None:
  """Prefix fill and identity scatter disagree; a count check accepts both.

  Three nodes, width 2, three real edges. The sparse order is not the
  leading C-order prefix, and a padded slot sits inside that prefix.
  Consuming one mask of three values reconciles either placement.
  """
  neighbours = np.asarray([[1, 0], [0, 0], [0, 1]], dtype=np.int32)
  valid = np.asarray([[True, False], [False, False], [True, True]])
  # source, sink, block. Order is sink 2, then the early node — not C order.
  edge = np.asarray([[0, 1, 1], [2, 2, 0], [0, 0, 0]], dtype=np.int64)
  keep = np.asarray([0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 3, 1)
  value = jnp.ones((3, 2, 1), dtype=jnp.float32)
  plan = _Plan(
    scalar=True,
    vector=False,
    masks={"ligand_encoder.gat_layers.0.dropout": [keep]},
    edges={"ligand_encoder.gat_layers.0.dropout": [edge]},
    drop_p=0.1,
  )
  placed = plan.take(
    "ligand_encoder.gat_layers.0.dropout",
    value,
    neighbours=jnp.asarray(neighbours),
    valid=jnp.asarray(valid),
  )
  identity = np.ones((3, 2, 1), dtype=np.float32)
  identity[2, 0, 0] = 0.0
  identity[2, 1, 0] = 1.0
  identity[0, 0, 0] = 0.0
  prefix = _prefix_fill(keep, (3, 2, 1))
  assert not np.array_equal(identity, prefix)
  np.testing.assert_array_equal(np.asarray(placed), identity)
  # Both placements consume the one recorded mask. The array above is the check.
  assert plan.consumption()["ligand_encoder.gat_layers.0.dropout"] == (1, 1)


def test_edge_identity_stays_inside_its_neighbour_block() -> None:
  """Ligand atom 0 and protein residue 0 are different edges.

  Searching the concatenated slot axis would give the ligand edge the first
  protein slot, because that slot's stored source is also 0.
  """
  protein = np.asarray([[0, 0], [0, 0]], dtype=np.int32)
  ligand = np.asarray([[0], [0]], dtype=np.int32)
  protein_valid = np.asarray([[True, False], [False, False]])
  ligand_valid = np.asarray([[True], [False]])
  # Ligand edge first, so a block-blind search hits the protein slot.
  edge = np.asarray([[0, 0], [0, 0], [1, 0]], dtype=np.int64)
  keep = np.asarray([0.0, 1.0], dtype=np.float32).reshape(1, 2, 1)
  value = jnp.ones((2, 3, 1), dtype=jnp.float32)
  plan = _Plan(
    scalar=True,
    vector=False,
    masks={"protein_encoder_layers.0.hetgat.dropout": [keep]},
    edges={"protein_encoder_layers.0.hetgat.dropout": [edge]},
    drop_p=0.1,
  )
  placed = plan.take(
    "protein_encoder_layers.0.hetgat.dropout",
    value,
    neighbours=(jnp.asarray(protein), jnp.asarray(ligand)),
    valid=(jnp.asarray(protein_valid), jnp.asarray(ligand_valid)),
  )
  got = np.asarray(placed)
  np.testing.assert_array_equal(got[0, :, 0], np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
  np.testing.assert_array_equal(got[1, :, 0], np.ones((3,), dtype=np.float32))


def test_missing_dense_slot_raises() -> None:
  """An edge with no slot must fail the replay, not stay silently kept."""
  edge = np.asarray([[4], [0], [0]], dtype=np.int64)
  keep = np.ones((1, 1, 1), dtype=np.float32)
  plan = _Plan(
    scalar=True,
    vector=False,
    masks={"ligand_encoder.gat_layers.0.dropout": [keep]},
    edges={"ligand_encoder.gat_layers.0.dropout": [edge]},
    drop_p=0.1,
  )
  with pytest.raises(RuntimeError, match="no dense slot"):
    plan.take(
      "ligand_encoder.gat_layers.0.dropout",
      jnp.ones((1, 2, 1), dtype=jnp.float32),
      neighbours=jnp.asarray([[0, 1]], dtype=np.int32),
      valid=jnp.asarray([[True, True]]),
    )


def test_shim_records_gat_edge_identity() -> None:
  """The dropout shim banks ``(source, sink, block)`` beside a GAT mask."""
  import torch

  from scripts.parity.oracle_shims.laser import injected_scalar_dropout

  class _GAT(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.dropout = torch.nn.Dropout(0.5)

    def forward(
      self,
      x: torch.Tensor,
      e_idx: torch.Tensor,
      *,
      residual: bool = False,
    ) -> torch.Tensor:
      # The residual dropout's own frame has no edge list. The parent
      # forward does, which is where upstream HomoGATv2 keeps ``e_idx``.
      if residual:
        return self.compute_edge_update(x)
      return self.dropout(x)

    def compute_edge_update(self, x: torch.Tensor) -> torch.Tensor:
      return self.dropout(x)

  class _Het(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.dropout = torch.nn.Dropout(0.5)

    def forward(
      self,
      x: torch.Tensor,
      edge_index_list: list[torch.Tensor],
    ) -> torch.Tensor:
      return self.dropout(x)

  class _MLP(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
      del e_idx
      return self.dropout(x)

  gat = _GAT()
  het = _Het()
  mlp = _MLP()
  rng = np.random.default_rng(0)
  e_idx = torch.tensor([[0, 1, 1], [2, 2, 0]], dtype=torch.int64)
  with injected_scalar_dropout(gat, rng=rng) as cursor:
    gat.forward(torch.ones(1, 3, 1), e_idx)
    gat.forward(torch.ones(3, 4), e_idx, residual=True)
  attention = cursor.edges["dropout"][0]
  residual = cursor.edges["dropout"][1]
  assert attention is not None and residual is not None
  np.testing.assert_array_equal(attention[:2], e_idx.numpy())
  np.testing.assert_array_equal(attention[2], np.zeros((3,), dtype=np.int64))
  np.testing.assert_array_equal(residual[:2], e_idx.numpy())
  parts = [torch.tensor([[0, 1], [0, 0]]), torch.tensor([[2], [1]])]
  with injected_scalar_dropout(het, rng=rng) as cursor:
    het.forward(torch.ones(1, 3, 1), parts)
  hetero = cursor.edges["dropout"][0]
  assert hetero is not None
  np.testing.assert_array_equal(hetero[0], np.asarray([0, 1, 2]))
  np.testing.assert_array_equal(hetero[1], np.asarray([0, 0, 1]))
  np.testing.assert_array_equal(hetero[2], np.asarray([0, 0, 1]))
  with injected_scalar_dropout(mlp, rng=rng) as cursor:
    mlp.forward(torch.ones(5, 8), e_idx)
  assert cursor.edges["dropout"][0] is None


def test_decoder_attention_keep_follows_masked_first_slots() -> None:
    """A recorded edge lands on the weight slot, which is masked-first order.

    Upstream concatenates masked edges in front of unmasked ones and drops
    that list (``model.py`` edge cat, ``model_generics.py`` ``self.dropout``
    on the joint attention). ``_teacher_edges`` permutes the dense features
    to the same order. The neighbour ids have to move with those features:
    looking the source up in the knn grid writes the keep onto a different
    slot, and a count check accepts both placements.
    """
    _encoder, decoder, _joint = _models()
    n_res = 3
    width = 3
    # Sink 1. Sources [2, 0, 1]: 0 is already decoded, 2 and the self-edge are not.
    neighbours = np.zeros((n_res, width), dtype=np.int32)
    neighbours[1] = np.asarray([2, 0, 1], dtype=np.int32)
    mask = np.zeros((n_res, width), dtype=bool)
    mask[1] = True
    node = int(decoder.sequence_label_embedding.weight.shape[1])
    edge = jnp.zeros((n_res, width, 128), dtype=jnp.float32)
    chi = jnp.zeros((n_res, 4 * 72), dtype=jnp.float32)
    _features, row_mask, aligned = decoder._teacher_edges(
        jnp.zeros((n_res, node), dtype=jnp.float32),
        jnp.zeros((n_res, node), dtype=jnp.float32),
        edge,
        jnp.asarray(neighbours),
        jnp.asarray(mask),
        jnp.zeros((n_res,), dtype=jnp.int32),
        chi,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )
    # Masked sources 2 and 1, then the decoded source 0.
    np.testing.assert_array_equal(np.asarray(aligned[1]), np.asarray([2, 1, 0], dtype=np.int32))
    assert not np.array_equal(np.asarray(aligned[1]), neighbours[1])
    # One recorded edge: source 0 into sink 1. Its weight is the last real slot.
    keep = np.zeros((1, 1, 1), dtype=np.float32)
    identity = np.asarray([[0], [1], [0]], dtype=np.int64)
    value = jnp.ones((n_res, width, 1), dtype=jnp.float32)
    plan = _Plan(
        scalar=True,
        vector=False,
        masks={"protein_decoder_layers.0.hetgat.dropout": [keep]},
        edges={"protein_decoder_layers.0.hetgat.dropout": [identity]},
        drop_p=0.1,
    )
    placed = plan.take(
        "protein_decoder_layers.0.hetgat.dropout",
        value,
        neighbours=aligned,
        valid=row_mask,
    )
    got = np.asarray(placed[1, :, 0])
    np.testing.assert_array_equal(got, np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
