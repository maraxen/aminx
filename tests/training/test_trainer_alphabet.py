"""Guard the trainer's alphabet boundary: labels it optimizes against must be MPNN-ordered.

Issue #109. ``aatype`` (proxide's parse, and the persisted ``"aatype"`` key of preprocessed
datasets) is **AF**-ordered; the model's token space -- logits, the decoder's one-hot embedding
input, CE targets -- is **MPNN**-ordered. ``train_step`` / ``eval_step`` used to receive
``batch.aatype`` raw, so they trained and scored against permuted labels with no error and no
loss anomaly. Every int 0-20 is legal in both alphabets, so no value/range/shape check can
catch this; only a comparison against ground truth can.

``tests/utils/test_alphabet_boundary.py`` pins the same fact at the *inference* boundary
(native recovery 0.592 through ``af_to_mpnn`` vs 0.118 raw). This module extends that gate to
the *training* boundary, in three layers:

1. cheap, always-on: ``training_labels`` is the AF->MPNN permutation and decodes 1UBQ's real
   sequence under MPNN (real parse, never a hand-written fixture);
2. cheap, always-on, structural: every ``.aatype`` read in the training code is routed through
   ``training_labels`` -- a revert at a call site is otherwise invisible, because the trainer
   functions take a plain int array;
3. heavy, real weights: one ``eval_step`` and one ``train_step`` on ``proteinmpnn_v_48_020``.
   Correctly converted labels must be decisively better than raw AF labels. The raw-label
   arm is the test's own negative control: it proves the measurement has power.

task_id: 260930_resolve-issues-debt
"""

from __future__ import annotations

import ast
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from aminx.utils.aa_convert import AF_ALPHABET, MPNN_ALPHABET, af_to_mpnn, training_labels

_TESTS = Path(__file__).resolve().parents[1]
_UBQ = _TESTS / "data" / "1ubq.pdb"
_SRC = Path(__file__).resolve().parents[2] / "src" / "aminx" / "training"

# 1UBQ's sequence from the deposition -- ground truth aminx did not produce.
_UBQ_SEQUENCE = (
  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

# ProteinMPNN's self-consistent recovery on a monomer is ~0.5-0.6 with correct labels; against
# permuted (AF) labels it sits at ~0.1. Thresholds sit in the empty gap, not near either side.
_RECOVERY_FLOOR = 0.35
_CHANCE_CEILING = 0.25


def _decode(tokens: np.ndarray, alphabet: str) -> str:
  return "".join(alphabet[int(i)] if 0 <= int(i) < len(alphabet) else "?" for i in tokens)


# ---------------------------------------------------------------------------------------
# Layer 1: the conversion helper (cheap, always on)
# ---------------------------------------------------------------------------------------


def test_training_labels_is_the_af_to_mpnn_permutation() -> None:
  """``training_labels`` agrees with ``af_to_mpnn`` on every legal residue type."""
  af = jnp.arange(len(AF_ALPHABET))
  got = np.asarray(training_labels(af))
  assert np.array_equal(got, np.asarray(af_to_mpnn(af)).astype(np.int32))
  # And it really permutes: the two alphabets differ at most positions, so a no-op fails here.
  assert not np.array_equal(got, np.asarray(af))
  for af_idx, aa in enumerate(AF_ALPHABET):
    assert MPNN_ALPHABET[int(got[af_idx])] == aa


def test_training_labels_preserves_shape_and_out_of_range_values() -> None:
  """Padding sentinels are not remapped onto a real residue; batch shape is preserved."""
  batch = jnp.array([[0, 1, 20, -1], [3, 4, 5, 99]])
  out = np.asarray(training_labels(batch))
  assert out.shape == batch.shape
  assert out.dtype == np.int32
  assert out[0, 3] == -1
  assert out[1, 3] == 99  # noqa: PLR2004
  assert out[0, 2] == 20  # noqa: PLR2004 -- X is a fixed point of the permutation


def test_loader_aatype_is_af_and_training_labels_is_mpnn() -> None:
  """On a real parse of 1UBQ through the trainer's own ingress: AF in, MPNN out."""
  from proxide.ops.dataset import create_protein_dataset  # noqa: PLC0415

  batch = next(iter(create_protein_dataset([str(_UBQ)], batch_size=1)))
  aatype = np.asarray(batch.aatype).reshape(-1)[: len(_UBQ_SEQUENCE)]
  labels = np.asarray(training_labels(batch.aatype)).reshape(-1)[: len(_UBQ_SEQUENCE)]

  assert _decode(aatype, AF_ALPHABET) == _UBQ_SEQUENCE
  assert _decode(aatype, MPNN_ALPHABET) != _UBQ_SEQUENCE, "test cannot discriminate"
  assert _decode(labels, MPNN_ALPHABET) == _UBQ_SEQUENCE, (
    "training_labels no longer yields MPNN-ordered labels for the loader's aatype."
  )


# ---------------------------------------------------------------------------------------
# Layer 2: no raw ``.aatype`` reaches a trainer step (cheap, always on, structural)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["trainer.py"])
def test_every_aatype_read_goes_through_training_labels(module: str) -> None:
  """Each ``<batch>.aatype`` in the training code is the sole argument of ``training_labels``.

  The trainer's step functions take a plain int array, so a call site that reverts to raw
  ``batch.aatype`` type-checks, runs, and trains on permuted labels. Layer 3 cannot see that
  either (it calls the step functions directly), so the call sites are pinned structurally.
  """
  tree = ast.parse((_SRC / module).read_text())
  wrapped: set[int] = set()
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "training_labels"
      and len(node.args) == 1
    ):
      wrapped.add(id(node.args[0]))

  reads = [
    n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "aatype"
  ]
  assert reads, f"{module}: expected at least one batch.aatype read; the guard is stale."
  unwrapped = [n.lineno for n in reads if id(n) not in wrapped]
  assert not unwrapped, (
    f"{module}: raw AF-ordered `.aatype` is used at line(s) {unwrapped} without "
    "`training_labels(...)`. The model's token space is MPNN; this trains against permuted "
    "labels (issue #109)."
  )


# ---------------------------------------------------------------------------------------
# Layer 3: one real step on real weights (heavy)
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _batch() -> dict:
  """1UBQ as the trainer's loader yields it. ``max_length=128`` pins the crop (no random crop)."""
  from proxide.ops.dataset import create_protein_dataset  # noqa: PLC0415

  batch = next(iter(create_protein_dataset([str(_UBQ)], batch_size=1, max_length=128)))
  return {
    "coordinates": jnp.asarray(np.asarray(batch.coordinates)),
    "mask": jnp.asarray(np.asarray(batch.mask)),
    "residue_index": jnp.asarray(np.asarray(batch.residue_index)),
    "chain_index": jnp.asarray(np.asarray(batch.chain_index)),
    "aatype": batch.aatype,
  }


@pytest.fixture(scope="module")
def _model() -> object:
  from aminx.io.weights import load_model  # noqa: PLC0415

  # dropout off: the comparison below must not hinge on which units dropout happened to drop.
  return load_model("proteinmpnn_v_48_020", dropout_rate=0.0)


def _eval(model: object, batch: dict, sequence: jax.Array) -> object:
  from aminx.training.trainer import eval_step  # noqa: PLC0415

  return eqx.filter_jit(eval_step)(
    model,
    batch["coordinates"],
    batch["mask"],
    batch["residue_index"],
    batch["chain_index"],
    sequence,
    jax.random.PRNGKey(0),
  )


def _train(model: object, batch: dict, sequence: jax.Array) -> object:
  from aminx.training.trainer import train_step  # noqa: PLC0415

  # A negligible learning rate: the reported metrics are computed before the update, and this
  # keeps the test about *which labels* the step sees, not about optimizer behaviour.
  optimizer = optax.sgd(1e-9)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
  _, _, metrics = eqx.filter_jit(train_step)(
    model,
    opt_state,
    optimizer,
    batch["coordinates"],
    batch["mask"],
    batch["residue_index"],
    batch["chain_index"],
    sequence,
    jax.random.PRNGKey(0),
    0.0,  # label_smoothing
    0,  # current_step
  )
  return metrics


@pytest.mark.alphabet_boundary
@pytest.mark.requires_weights
@pytest.mark.slow
def test_eval_step_scores_converted_labels_far_better_than_raw_af(
  _model: object, _batch: dict
) -> None:
  """``eval_step`` (recovery, perplexity, loss) only makes sense on MPNN-ordered labels."""
  converted = _eval(_model, _batch, training_labels(_batch["aatype"]))
  raw = _eval(_model, _batch, jnp.asarray(_batch["aatype"]).astype(jnp.int32))

  acc_ok, acc_raw = float(converted.val_accuracy), float(raw.val_accuracy)  # type: ignore[attr-defined]
  assert acc_ok > _RECOVERY_FLOOR, (
    f"eval_step recovery on correctly converted labels is {acc_ok:.3f}: the model/labels "
    "pairing is broken, so this test has no power."
  )
  assert acc_raw < _CHANCE_CEILING, (
    f"eval_step recovery on raw AF labels is {acc_raw:.3f}, not near chance. The negative "
    "control failed; do not relax the ceiling -- find out why the permutation stopped mattering."
  )
  assert float(converted.val_loss) < float(raw.val_loss)  # type: ignore[attr-defined]
  assert float(converted.val_perplexity) < float(raw.val_perplexity)  # type: ignore[attr-defined]


@pytest.mark.alphabet_boundary
@pytest.mark.requires_weights
@pytest.mark.slow
def test_train_step_loss_and_recovery_on_converted_vs_raw_labels(
  _model: object, _batch: dict
) -> None:
  """One ``train_step``: converted labels give low loss / high recovery, raw AF does not.

  This is the assertion that fails if the trainer goes back to feeding ``aatype`` unconverted:
  the loss is computed against ``sequence`` and ``sequence`` is also the decoder's embedding
  input, so both collapse together.
  """
  converted = _train(_model, _batch, training_labels(_batch["aatype"]))
  raw = _train(_model, _batch, jnp.asarray(_batch["aatype"]).astype(jnp.int32))

  acc_ok, acc_raw = float(converted.accuracy), float(raw.accuracy)  # type: ignore[attr-defined]
  assert acc_ok > _RECOVERY_FLOOR, (
    f"train_step recovery on converted labels is {acc_ok:.3f}; the model/labels pairing is "
    "broken, so this test has no power."
  )
  assert acc_raw < _CHANCE_CEILING, (
    f"train_step recovery on raw AF labels is {acc_raw:.3f}, not near chance; the negative "
    "control failed."
  )
  # The loss gap is the quantity gradient descent actually sees.
  assert float(raw.loss) - float(converted.loss) > 1.0, (  # type: ignore[attr-defined]
    f"raw-AF loss {float(raw.loss):.3f} is not decisively above converted "  # type: ignore[attr-defined]
    f"{float(converted.loss):.3f}."  # type: ignore[attr-defined]
  )
