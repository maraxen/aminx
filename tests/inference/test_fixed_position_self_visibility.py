"""No position may read its own slot through the ar_mask diagonal.

WHY THIS EXISTS
===============
Until 2026-08-27 every ar_mask constructor here set the diagonal. `generate_ar_mask`'s
untied branch was `row_indices >= col_indices` (non-strict); `generate_wave_ar_mask`
returns `earlier_wave | (same_wave & same_group)`, trivially true when i == j. The stated
rationale was that a not-yet-drawn position's own slot holds an inert placeholder.

That premise did not hold. `decode/autoregressive.py` initialized undrawn slots to token
index 0, and MPNN_ALPHABET is "ACDEFGHIKLMNPQRSTVWYX" -- index 0 is ALANINE, and the
unknown token X is index 20. `model/decoder.py:124` embeds it as
`one_hot_sequence @ w_s_weight`, a real nonzero row of a pretrained matrix. Because the
decoder gathers ar_mask into `attention_mask` to gate the SEQUENCE edge features
(`decoder.py:144-146`) and a residue is always among its own KNN neighbours (self-distance
0 is the global minimum), every position was fed the assertion "I am alanine" about itself
at the step it decided its own identity.

TWO INDEPENDENT REFERENCES DISAGREED WITH US, both verified against primary source:
  * LigandMPNN (dauparas), model_utils.py:227-231 + four more call sites:
        1 - torch.triu(torch.ones(L, L))        # triu defaults to diagonal=0
  * ColabDesign (sokrypton), colabdesign/mpnn/utils.py:19-26:
        jnp.tri(L, k=-1)[idx, :][:, idx]
Both are strictly triangular with a zero diagonal, preserved under any permutation.

Our own parity suite never caught it because `test_full_model_parity.py:90-95` hand-builds
its ar_mask with a STRICT comparison and passes it explicitly, so `generate_ar_mask` --
the function the production sampling path actually calls -- is never invoked there.

Measured downstream on TEV protease: mean JSD 0.0296 over designable positions between the
old and new masks, with an alanine-specific probability shift of -0.0049 against a mean of
0.0016 across the other twenty tokens.

SCOPE. Only the diagonal changes. Same-tie-group mutual visibility, and the
same-wave/different-group invisibility that gives the wave schedule its Jacobi
independence, are both preserved exactly. Whether tie visibility matches the reference's
tied decoding is a separate open question, deliberately not settled here.

Run narrowly:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 uv run pytest \
        tests/inference/test_fixed_position_self_visibility.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.autoregressive import UNDRAWN_TOKEN
from aminx.inference.schedule_selector import WaveScheduleBundle
from aminx.utils.autoregression import (
  full_context_ar_mask,
  generate_ar_mask,
  generate_wave_ar_mask,
)

L = 12


def _order(seed: int | None = None) -> jnp.ndarray:
  if seed is None:
    return jnp.arange(L, dtype=jnp.int32)
  return jnp.asarray(np.random.default_rng(seed).permutation(L), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# The invariant: zero diagonal, every constructor, every order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [None, 0, 1, 2, 3, 4])
def test_generate_ar_mask_is_self_excluding_under_every_order(seed):
  """A permutation must never reintroduce the diagonal."""
  m = np.asarray(generate_ar_mask(order=_order(seed)))
  assert int(np.diag(m).sum()) == 0, f"diagonal set under order seed={seed}: {np.diag(m)}"


def test_generate_ar_mask_tied_branch_is_self_excluding():
  """The tied branch takes a different code path and must also self-exclude."""
  tie = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=jnp.int32)
  m = np.asarray(generate_ar_mask(order=_order(), tie_group_map=tie))
  assert int(np.diag(m).sum()) == 0


def test_generate_wave_ar_mask_is_self_excluding():
  """The wave path is what production decodes on; it shares the defect and the fix."""
  tie = jnp.arange(L, dtype=jnp.int32)
  m = np.asarray(generate_wave_ar_mask(WaveScheduleBundle.empty(L), tie))
  assert int(np.diag(m).sum()) == 0


def test_full_context_ar_mask_remains_diagonal_free_and_non_causal():
  """Unchanged by this fix, and must stay distinguishable from a causal mask.

  It is the SCORING remedy: full context minus self. Substituting it for a causal decode
  would also destroy same-wave Jacobi independence, so the two must not converge.
  """
  m = np.asarray(full_context_ar_mask(L))
  assert int(np.diag(m).sum()) == 0
  assert not np.array_equal(m, np.tril(np.ones((L, L), dtype=m.dtype), k=-1))


# ---------------------------------------------------------------------------
# Non-degeneracy: a zero diagonal must not have been bought by emptying the mask
# ---------------------------------------------------------------------------


def test_masks_remain_non_degenerate_and_causal():
  """An all-zero mask also has a zero diagonal while removing ALL context.

  That is a separate and worse defect class this project has already hit, so a diagonal
  assertion alone is not sufficient evidence of correctness.
  """
  m = np.asarray(generate_ar_mask(order=_order()))
  assert m.sum() > 0, "mask is all-zero -- that removes all context"
  assert np.array_equal(m, np.tril(np.ones((L, L), dtype=m.dtype), k=-1)), (
    "under the identity decoding order the mask must be strictly lower-triangular"
  )

  wave = np.asarray(generate_wave_ar_mask(WaveScheduleBundle.empty(L), jnp.arange(L, dtype=jnp.int32)))
  assert wave.sum() > 0


def test_tie_visibility_is_preserved_and_not_silently_narrowed():
  """SCOPE GUARD: this change must alter the diagonal and nothing else.

  Same-tie-group positions stay mutually visible. If a future change to the diagonal also
  narrows tie semantics, that is a separate decision and must fail here rather than ride
  along unnoticed.
  """
  tie = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=jnp.int32)
  m = np.asarray(generate_ar_mask(order=_order(), tie_group_map=tie))
  assert m[0, 1] == 1 and m[1, 0] == 1, "same-tie-group mutual visibility was removed"


# ---------------------------------------------------------------------------
# Conformance against an independent implementation
# ---------------------------------------------------------------------------


def test_matches_colabdesign_bit_for_bit_on_the_untied_path():
  """Dev-dependency-only conformance, in the spirit of test_alphabet_conformance.py.

  ColabDesign is an independent reimplementation. On the untied path our mask must equal
  `jnp.tri(L, k=-1)` permuted by the decoding order, cell for cell -- not merely agree on
  the diagonal. If this fails, one of the two is wrong and WHICH ONE is the finding; do
  not edit this test to match.

  MIND THE CONVENTION -- the two libraries encode a decoding order differently, and the
  distinction is invisible in dtype and shape:

      ORDER array   order[k] = which position is decoded at step k
      RANK  array   rank[i]  = at which step position i is decoded   (rank = argsort(order))

  ColabDesign's `get_ar_mask` calls `order.argsort()` first, so it takes an ORDER array and
  inverts it internally. `generate_ar_mask` also takes an ORDER array (`order[t]` = position
  decoded at step t), so the same array is passed to both below. A RANK array (`argsort(order)`)
  handed to either one would silently compute the mask of the inverse order.
  """
  cd_utils = pytest.importorskip(
    "colabdesign.mpnn.utils",
    reason="colabdesign not installed; conformance check skipped",
  )
  for seed in (0, 1, 2, 3, 4):
    order = np.random.default_rng(seed).permutation(L)
    theirs = np.asarray(cd_utils.get_ar_mask(jnp.asarray(order, dtype=jnp.int32))).astype(int)
    ours = np.asarray(generate_ar_mask(order=jnp.asarray(order, dtype=jnp.int32))).astype(int)
    assert np.array_equal(ours, theirs), (
      f"diverges from ColabDesign at order seed={seed}; "
      f"differing cells: {np.argwhere(ours != theirs).tolist()[:10]}"
    )


def test_generate_ar_mask_consumes_an_order_array():
  """Pin the convention: Defect 9 retired the rank-vs-order hazard.

  `generate_ar_mask` now consistently consumes an ORDER array: `order[t]` = position
  decoded at step `t`.
  """
  # order[t] = position decoded at step t.
  # Position 2 decodes at step 0, pos 0 at step 1, pos 3 at step 2, pos 1 at step 3.
  order = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)
  m = np.asarray(generate_ar_mask(order=order)).astype(int)
  # Position 0 (step 1) must see position 2 (step 0) and nothing later.
  assert m[0, 2] == 1, "position decoded at step 1 cannot see the step-0 position"
  assert m[0, 1] == 0, "position decoded at step 1 sees a step-3 position"
  assert m[0, 3] == 0, "position decoded at step 1 sees a step-2 position"
  # The first-decoded position sees nobody.
  assert m[2].sum() == 0, "the step-0 position has visible context"
  # The last-decoded position (pos 1, step 3) sees all previously decoded positions.
  assert m[1, 2] == 1 and m[1, 0] == 1 and m[1, 3] == 1


# ---------------------------------------------------------------------------
# The undrawn sentinel
# ---------------------------------------------------------------------------


def test_partial_schedule_returns_undrawn_token_in_the_SEQUENCE_not_just_the_mask():
  """END-TO-END contract test: read `SampleResult.sequence`, not the mask.

  `tests/utils/test_autoregression.py::test_generate_wave_ar_mask_partial_schedule_does_not_leak_omitted_positions`
  already covers the MASK for a partial schedule. Nothing covered what the returned
  SEQUENCE holds at an omitted position -- which is exactly how the escape route for
  UNDRAWN_TOKEN went unnoticed until code review.

  A partial wave schedule is a supported feature: a caller schedules only the positions it
  needs and leaves the rest permanently undecided. Those positions used to come back as
  token 0, i.e. ALANINE, indistinguishable from a deliberately-sampled alanine. They now
  come back as -1, which is honest but is NOT a valid residue index -- see
  `SampleResult`'s docstring for the consumer contract.

  This test pins that. If someone later "fixes" the -1 by clamping it to 0, this fails and
  the plausible lie does not come back silently.
  """
  import equinox as eqx
  import jax

  from aminx.inference import sample_autoregressive
  from aminx.inference.logits import make_stage_set
  from aminx.model.mpnn import Aminx
  from aminx.types.bundles import WaveScheduleBundle as WSB

  seq_len = 6
  scheduled, omitted = [0, 1, 2], [3, 4, 5]

  key = jax.random.PRNGKey(0)
  model = eqx.tree_inference(
    Aminx(
      node_features=32, edge_features=32, hidden_features=32,
      num_encoder_layers=1, num_decoder_layers=1, k_neighbors=4, key=key,
    ),
    value=True,
  )

  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(seq_len, 4, 3)) * 3.0, dtype=jnp.float32)
  tie_group_map = jnp.arange(seq_len, dtype=jnp.int32)
  partial = WSB.from_tie_groups(tie_group_map, jnp.asarray(scheduled, dtype=jnp.int32))

  bundle, config = build_inference_bundle(
    coords=coords,
    mask=jnp.ones((seq_len,), dtype=jnp.float32),
    residue_index=jnp.arange(seq_len, dtype=jnp.int32),
    chain_index=jnp.zeros((seq_len,), dtype=jnp.int32),
    tie_group_map=tie_group_map,
    wave=partial,
    mode="sample_ar",
    inference=True,
  )
  result = sample_autoregressive.kernel(
    model, key, bundle, config,
    make_stage_set(strategy="arithmetic_mean", strategy_temperature=1.0, state_weights=None),
  )
  seq = np.asarray(result.sequence)
  seq = seq[0] if seq.ndim == 2 else seq

  assert all(seq[i] == UNDRAWN_TOKEN for i in omitted), (
    f"positions {omitted} were never scheduled, so they must come back as UNDRAWN_TOKEN "
    f"({UNDRAWN_TOKEN}); got {seq.tolist()}. If this now returns 0, an unscheduled position "
    "is again indistinguishable from a sampled alanine."
  )
  # Non-degeneracy: the scheduled positions must actually have been decoded, or this test
  # would pass just as well against a kernel that decoded nothing at all.
  assert all(0 <= seq[i] < 21 for i in scheduled), (
    f"scheduled positions {scheduled} were not decoded: {seq.tolist()}"
  )


def test_undrawn_sentinel_is_not_a_real_residue():
  """0 is alanine and 20 is X; neither can express "not drawn yet"."""
  from aminx.utils.aa_convert import MPNN_ALPHABET

  assert UNDRAWN_TOKEN < 0, f"sentinel {UNDRAWN_TOKEN} collides with a real token index"
  assert MPNN_ALPHABET[0] == "A", "index 0 is no longer alanine; re-check the sentinel rationale"
  assert MPNN_ALPHABET[20] == "X"


def test_undrawn_sentinel_is_signed_and_representable():
  """The carry dtype must be signed, or -1 wraps and stops being detectable.

  `CarryShape.dtype` is typed `Any` and enforces nothing; every construction site passes
  int32 today, so this invariant currently holds by caller convention. `AutoregressiveDecode`
  asserts it at decode time -- this pins the assumption the assertion encodes.
  """
  import jax.numpy as jnp_

  from aminx.inference.decode.factory import make_decode_fn  # noqa: F401  (import smoke)

  assert jnp_.issubdtype(jnp_.int32, jnp_.signedinteger)
  # And -1 must round-trip through the carry dtype unchanged.
  assert int(jnp_.full((1,), UNDRAWN_TOKEN, dtype=jnp_.int32)[0]) == UNDRAWN_TOKEN


def test_mbr_consensus_refuses_an_undrawn_token_rather_than_scoring_it_free():
  """A leaked -1 one-hots to zero, giving NLL 0.0 -- the BEST possible score.

  That is the one place the sentinel is more dangerous than the old token-0 default: an
  undecided position would score as a free perfect match and bias candidate selection
  toward whichever candidate carried it. It must fail loudly instead.
  """
  import jax.numpy as jnp_

  from aminx.sampling.mbr_consensus import _nll_from_logits

  logits = jnp_.zeros((4, 21))
  clean = jnp_.array([1, 2, 3, 4], dtype=jnp_.int32)
  # Sanity: the clean path works and is non-degenerate, so the failure below is about the
  # sentinel and not about the function being broken generally.
  assert float(_nll_from_logits(logits, clean)) > 0.0

  leaked = jnp_.array([1, 2, UNDRAWN_TOKEN, 4], dtype=jnp_.int32)
  with pytest.raises(ValueError, match="out-of-range token indices"):
    _nll_from_logits(logits, leaked)


def test_undrawn_sentinel_embeds_to_the_zero_vector():
  """This is the property that makes -1 equivalent to the reference's h_S = zeros.

  An undrawn position must contribute NO sequence signal. Token 0 contributes alanine's
  pretrained embedding row, which is what the old sentinel was silently asserting.
  """
  import jax

  assert float(jax.nn.one_hot(UNDRAWN_TOKEN, 21).sum()) == 0.0
  assert float(jax.nn.one_hot(0, 21).sum()) == 1.0, (
    "token 0 no longer one-hots to a real residue; the sentinel rationale needs revisiting"
  )


# ---------------------------------------------------------------------------
# The default-resolution branch, which nothing previously exercised
# ---------------------------------------------------------------------------


def _minimal_inputs(seq_len: int = L):
  rng = np.random.default_rng(0)
  return (
    jnp.asarray(rng.normal(size=(seq_len, 4, 3)) * 3.0, dtype=jnp.float32),
    jnp.ones((seq_len,), dtype=jnp.float32),
    jnp.arange(seq_len, dtype=jnp.int32),
    jnp.zeros((seq_len,), dtype=jnp.int32),
  )


@pytest.mark.parametrize("mode", ["sample_ar", "score_conditional"])
def test_bundle_resolved_mask_is_self_excluding(mode):
  """Reaches the `ar_mask is None` branch, which the parity suite never does.

  The parity tests hand-build a strict mask and pass it explicitly, so the constructors
  the production path actually calls were structurally untested. This closes that.
  """
  coords, mask, residue_index, chain_index = _minimal_inputs()
  bundle, _ = build_inference_bundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    ar_mask=None,
    fixed_mask=jnp.asarray(np.isin(np.arange(L), [2, 5, 7]).astype(np.float32)),
    fixed_tokens=jnp.zeros((L,), dtype=jnp.int32),
    mode=mode,
  )
  m = np.asarray(bundle.conditioning.ar_mask)
  m3 = m if m.ndim == 3 else m[None]
  for s in range(m3.shape[0]):
    assert int(np.diag(m3[s]).sum()) == 0, f"mode={mode} state={s} diagonal is set"
