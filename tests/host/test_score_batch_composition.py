"""A sequence's score must not depend on what else is in its batch (debt #3487).

Two checks, because one is not enough:

* **Content invariance (bitwise).** At a fixed number of sequences, the target's logits are
  identical whatever the partners contain or how long they are (shorter partners are X-padded).
  Shapes are equal, so float reassociation cannot move them: any difference is a cross-row leak.
* **Alone vs batched (bounded).** Across batch sizes the logits may differ by reassociation only,
  within ``SCORE_BATCH_COMPOSITION_ATOL``.

The planted positive control is a dispatcher that leaks other rows' PADDED positions into every
row at 1e-6, below the bound. The bound check alone passes it; the content-invariance check must
catch it. If it does not, a "clean" result here proves nothing.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aminx.tiling.dispatch as dispatch
from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification
from aminx.scoring.score import SCORE_BATCH_COMPOSITION_ATOL

_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_L = len(_UBIQUITIN)
_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
_PLANT = 1e-6


def _partners(n: int, length: int, seed: int) -> list[str]:
  rng = np.random.default_rng(seed)
  return ["".join(rng.choice(list(_LETTERS), size=length)) for _ in range(n)]


def _target_logits(partners: list[str]) -> np.ndarray:
  spec = ScoringSpecification(
    inputs=[_STRUCTURE], checkpoint_id=_CHECKPOINT, sequences_to_score=[_UBIQUITIN, *partners],
    max_length=96, return_logits=True, backbone_noise=0.0,
  )
  return np.asarray(score(spec)["logits"])[0, 0, :_L]


def _composition_report() -> dict[str, float]:
  """Max content deviation at fixed C, and max |delta| vs alone over C in {2, 4, 8}."""
  alone = _target_logits([])
  content_dev, vs_alone = 0.0, 0.0
  for c in (2, 4, 8):
    ref = None
    for seed, plen in ((0, _L), (1, _L - 20), (2, _L // 2)):
      row = _target_logits(_partners(c - 1, plen, seed))
      vs_alone = max(vs_alone, float(np.abs(row - alone).max()))
      if ref is None:
        ref = row
      else:
        content_dev = max(content_dev, float(np.abs(row - ref).max()))
  return {"content_dev": content_dev, "vs_alone": vs_alone}


def test_scores_are_content_invariant_and_within_the_bound() -> None:
  report = _composition_report()
  assert report["content_dev"] == 0.0, f"cross-row dependence: {report}"
  assert report["vs_alone"] <= SCORE_BATCH_COMPOSITION_ATOL, report


def test_planted_padding_leak_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
  """Positive control: a 1e-6 leak of other rows' padded positions must fail content invariance."""
  real = dispatch.make_axis_dispatch

  def leaky_make_axis_dispatch(strategy: Any, *, axis: str = "state") -> Callable[..., Any]:  # noqa: ANN401
    inner = real(strategy, axis=axis)

    def leaky(fn: Callable[..., Any], items: dict[str, jax.Array]) -> Any:  # noqa: ANN401
      nll, logits, order = inner(fn, items)
      is_pad = (items["seq"] == 20).astype(logits.dtype)[..., None]  # X-padded positions, every row
      leak = (logits * is_pad).sum((0, 1)) / jnp.maximum(is_pad.sum(), 1.0)
      return nll, logits + _PLANT * leak, order

    return leaky

  monkeypatch.setattr(dispatch, "make_axis_dispatch", leaky_make_axis_dispatch)
  report = _composition_report()
  assert report["vs_alone"] <= SCORE_BATCH_COMPOSITION_ATOL, "plant is meant to sit under the bound"
  assert report["content_dev"] > 0.0, "the content-invariance check missed a planted padding leak"
