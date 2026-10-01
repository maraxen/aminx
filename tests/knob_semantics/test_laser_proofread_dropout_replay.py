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
from aminx.model.laser.proofread import _Plan, conditional_focus_probs, proofread_dropout


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
