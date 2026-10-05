"""``ScoringSpecification.ar_mask``: score under a caller-chosen visibility pattern.

``runner.score`` has always scored every position under full context minus self,
``p(s_i | s_{-i}, X)``. That is a pseudo-likelihood: its per-position conditionals do not multiply
into a normalised joint, so it cannot answer questions about an autoregressive factorisation
(KL between staged orders, fragment mutual information). ``ar_mask`` lets a caller choose the
visibility pattern; ``None`` -- the default -- must leave the old path exactly as it was.

What this file pins:

* the mask is validated (shape, binary, zero diagonal) before anything runs;
* the two scorers that would silently drop it refuse it;
* an explicit full-context mask reproduces the default, so the default really is "this mask";
* a causal mask is STRICTLY causal on real weights: changing the token at order position ``k``
  moves no logit at order positions ``<= k``, and does move later ones;
* the negative control: under the default mask the same edit DOES move the earlier logits, so
  the causality assertion above is capable of failing.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.runner import score
from aminx.run.spec_json import run_specification_from_json_dict, run_specification_to_json_dict
from aminx.run.specs import ScoringSpecification
from aminx.utils.autoregression import ar_mask_from_decoding_order, full_context_ar_mask

_PLACEHOLDER_PDB = "does/not/need/to/exist.pdb"
_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"
_MAX_LENGTH = 128
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_L = len(_UBIQUITIN)
# float32 reduction-order noise between two lowerings of the same program (measured in
# test_score_encode_once.py at <= 4.8e-6 on logits); 4x that.
_FP_ATOL = 2e-5
# A real dependence is orders of magnitude above float noise; this is the floor for "moved".
_MOVED = 1e-3


def _causal_mask(order: np.ndarray) -> np.ndarray:
  return np.asarray(ar_mask_from_decoding_order(jnp.asarray(order, dtype=jnp.int32)))


def _spec(sequences: list[str], **kwargs) -> ScoringSpecification:  # noqa: ANN003
  return ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=_CHECKPOINT,
    sequences_to_score=sequences,
    max_length=_MAX_LENGTH,
    return_logits=True,
    **kwargs,
  )


def _score(sequences: list[str], **kwargs) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN003
  result = score(_spec(sequences, **kwargs))
  return np.asarray(result["scores"])[0], np.asarray(result["logits"])[0][:, :_L, :]


def _edit(sequence: str, position: int) -> str:
  """Change one residue to a different one, deterministically."""
  replacement = "W" if sequence[position] != "W" else "G"
  return sequence[:position] + replacement + sequence[position + 1 :]


# ---------------------------------------------------------------------------------- no weights


@pytest.mark.parametrize(
  ("mask", "match"),
  [
    (np.zeros((3, 4)), "square"),
    (np.zeros((3,)), "square"),
    (np.full((3, 3), 0.5) - 0.5 * np.eye(3), "binary"),
    (np.ones((3, 3)), "zero diagonal"),
  ],
)
def test_invalid_ar_mask_is_refused_at_construction(mask: np.ndarray, match: str) -> None:
  with pytest.raises(ValueError, match=match):
    ScoringSpecification(inputs=[_PLACEHOLDER_PDB], sequences_to_score=["AAA"], ar_mask=mask)


def test_valid_ar_mask_is_kept_and_defaults_to_none() -> None:
  assert ScoringSpecification(inputs=[_PLACEHOLDER_PDB], sequences_to_score=["AAA"]).ar_mask is None
  mask = np.asarray(full_context_ar_mask(3))
  spec = ScoringSpecification(inputs=[_PLACEHOLDER_PDB], sequences_to_score=["AAA"], ar_mask=mask)
  np.testing.assert_array_equal(spec.ar_mask, mask)


def test_ar_mask_survives_a_json_round_trip() -> None:
  mask = _causal_mask(np.array([2, 0, 1]))
  spec = ScoringSpecification(inputs=[_PLACEHOLDER_PDB], sequences_to_score=["AAA"], ar_mask=mask)
  restored = run_specification_from_json_dict(
    json.loads(json.dumps(run_specification_to_json_dict(spec))),
  )
  np.testing.assert_array_equal(np.asarray(restored.ar_mask), mask)


@pytest.mark.parametrize(
  "field",
  [{"average_node_features": True}, {"state_position_map": np.zeros((1, 3), dtype=np.int32)}],
)
def test_ar_mask_is_refused_on_the_paths_that_would_drop_it(field: dict) -> None:
  spec = ScoringSpecification(
    inputs=[_PLACEHOLDER_PDB],
    sequences_to_score=["AAA"],
    max_length=3,
    ar_mask=np.asarray(full_context_ar_mask(3)),
    **field,
  )
  with pytest.raises(NotImplementedError, match="ar_mask"):
    score(spec)


# ------------------------------------------------------------------------------- real weights


@pytest.mark.slow
@pytest.mark.requires_weights
def test_explicit_full_context_mask_reproduces_the_default() -> None:
  """The default IS ``full_context_ar_mask``; spelling it out must not move a number."""
  default_nll, default_logits = _score([_UBIQUITIN, _edit(_UBIQUITIN, 30)])
  explicit_nll, explicit_logits = _score(
    [_UBIQUITIN, _edit(_UBIQUITIN, 30)], ar_mask=np.asarray(full_context_ar_mask(_L)),
  )
  # Vacuity guard: the two candidates must actually differ.
  assert abs(float(default_nll[0]) - float(default_nll[1])) > _MOVED
  np.testing.assert_allclose(explicit_nll, default_nll, rtol=0, atol=_FP_ATOL)
  np.testing.assert_allclose(explicit_logits, default_logits, rtol=0, atol=_FP_ATOL)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_causal_mask_is_strictly_causal_and_the_default_mask_is_not() -> None:
  order = np.random.default_rng(0).permutation(_L)
  cut = 40
  edited_position = int(order[cut])
  base, edited = _UBIQUITIN, _edit(_UBIQUITIN, edited_position)
  earlier_or_self = order[: cut + 1]
  later = order[cut + 1 :]

  # Strict causality under a causal mask.
  _, causal_logits = _score([base, edited], ar_mask=_causal_mask(order))
  moved = np.abs(causal_logits[0] - causal_logits[1]).max(axis=-1)  # (L,)
  assert moved[earlier_or_self].max() <= _FP_ATOL, "a logit at or before the edit moved"
  assert moved[later].max() > _MOVED, "no later logit moved -- the mask is dead"

  # Negative control: under the default full-context mask the same edit moves earlier logits,
  # so the invariance above is a property of the mask and not of the data.
  _, default_logits = _score([base, edited])
  moved_default = np.abs(default_logits[0] - default_logits[1]).max(axis=-1)
  assert moved_default[earlier_or_self[:-1]].max() > _MOVED


@pytest.mark.slow
@pytest.mark.requires_weights
def test_order_changes_the_score() -> None:
  """Forward and reversed orders are different factorisations; the number must show it."""
  forward_order = np.arange(_L)
  forward, _ = _score([_UBIQUITIN], ar_mask=_causal_mask(forward_order))
  reverse, _ = _score([_UBIQUITIN], ar_mask=_causal_mask(forward_order[::-1]))
  default, _ = _score([_UBIQUITIN])
  assert abs(float(forward[0]) - float(reverse[0])) > _MOVED
  assert abs(float(forward[0]) - float(default[0])) > _MOVED


@pytest.mark.slow
@pytest.mark.requires_weights
def test_score_is_the_mean_negative_log_prob_of_the_scored_tokens_under_the_mask() -> None:
  """The returned NLL and the returned logits agree, under a custom mask."""
  from aminx.utils.aa_convert import string_to_protein_sequence  # noqa: PLC0415

  order = np.random.default_rng(1).permutation(_L)
  nll, logits = _score([_UBIQUITIN], ar_mask=_causal_mask(order))
  tokens = np.asarray(string_to_protein_sequence(_UBIQUITIN))
  log_probs = np.asarray(jax.nn.log_softmax(jnp.asarray(logits[0]), axis=-1))
  recomputed = -float(log_probs[np.arange(_L), tokens].mean())
  np.testing.assert_allclose(float(nll[0]), recomputed, rtol=0, atol=_FP_ATOL)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_candidate_length_must_match_the_mask() -> None:
  with pytest.raises(ValueError, match="exactly"):
    score(_spec([_UBIQUITIN, _UBIQUITIN[:60]], ar_mask=np.asarray(full_context_ar_mask(_L))))
