"""``runner.score`` encodes each structure once; it must score exactly what the per-candidate core did (#147).

The hoist moved the backbone encoding out of the candidate map. Correctness is equivalence, not
speed: for every candidate the NLL and the logits must equal what
``aminx.scoring.score.make_score_fn`` -- the un-hoisted per-candidate core, which still runs
encode -> decode for each (structure, sequence) pair -- returns for the same inputs and the same
PRNG key. The reference is computed by calling that core directly, so no old ``runner.score`` code
has to be kept around.

No timing or memory figure is asserted or quoted here: this file is a correctness test only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _canonical_structure_ids_for_spec, _prepare_ligand_context
from aminx.host.prep import prep_protein_stream_and_model
from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification
from aminx.scoring.score import make_score_fn, make_score_split_fns
from aminx.utils.aa_convert import string_to_protein_sequence

_STRUCTURE = "tests/data/1ubq.pdb"
_PROTEIN_CHECKPOINT = "proteinmpnn_v_48_020"
_LIGAND_CHECKPOINT = "ligandmpnn_v_32_020_25"
# max_length >= chain length and a fixed value: the loader's random crop would otherwise make
# the two sides see different structures.
_MAX_LENGTH = 128
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_CANDIDATES = [
  _UBIQUITIN,
  "A" * len(_UBIQUITIN),  # homopolymer: very different NLL from native
  _UBIQUITIN[:40] + "GGGGGG" + _UBIQUITIN[46:],  # point-block mutant
  _UBIQUITIN[:60],  # shorter than the chain: exercises the X-padding of the candidate axis
]
# Tolerances (titanix, CPU, float32), measured rather than chosen. Two references are used:
#  - "batched": the un-hoisted core under the SAME batching the old runner used (vmap over the
#    structure axis, vmap over the candidate axis).
#  - "loop": the un-hoisted core called one candidate at a time, no vmap.
# proteinmpnn_v_48_020: runner.score is BITWISE equal to "batched" (the hoist changes no bit). Versus
#   "loop" it differs by max |d nll| 2.4e-7 and |d logits| 1.9e-6, and the two un-hoisted references
#   differ from each other by exactly that much -- batched-vs-unbatched float32 reduction order,
#   independent of the hoist.
# ligandmpnn_v_32_020_25 (with and without a ligand): NOT bitwise to "batched" -- max |d nll|
#   4.8e-7 and |d logits| 1.9e-6, with the un-hoisted references differing from each other by up
#   to 2.9e-6 in logits. Same ulp-scale reduction-order noise, here also moved by the encode no
#   longer sharing a batched lowering with the decode. It is tiny against the quantities the tests
#   must still resolve: dropping the ligand moves the NLL by ~7e-4, a wrong structure by O(1).
# _FP_ATOL is ~4x the largest observed logit gap (4.8e-6, ligand vs "loop").
_FP_ATOL = 2e-5


def _spec(checkpoint_id: str, **kwargs: Any) -> ScoringSpecification:  # noqa: ANN401
  return ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=checkpoint_id,
    sequences_to_score=list(_CANDIDATES),
    max_length=_MAX_LENGTH,
    return_logits=True,
    **kwargs,
  )


def _write_synthetic_ligand(path: Path, structure_id: str, *, n_atoms: int = 8) -> None:
  rng = np.random.default_rng(0)
  np.savez(
    path,
    **{
      f"{structure_id}::Y": rng.normal(scale=6.0, size=(_MAX_LENGTH, n_atoms, 3)).astype(np.float32),
      f"{structure_id}::Y_t": np.full((_MAX_LENGTH, n_atoms), 6, dtype=np.int32),
      f"{structure_id}::Y_m": np.ones((_MAX_LENGTH, n_atoms), dtype=np.float32),
    },
  )


def _reference_per_candidate_core(
  spec: ScoringSpecification,
  *,
  batched: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Score every candidate through ``make_score_fn`` -- encode + decode per candidate.

  ``batched=False`` calls the core once per candidate. ``batched=True`` reproduces how the old
  ``runner.score`` dispatched it: ``jax.vmap`` over the structure axis, ``jax.vmap`` over the
  candidate axis, ligand tensors mapped per structure.
  """
  iterator, model = prep_protein_stream_and_model(spec)
  (ensemble,) = list(iterator)
  assert ensemble.coordinates.shape[0] == 1, "test assumes one structure in one batch"
  struct_len = ensemble.coordinates.shape[1]

  ligand: tuple[Any, Any, Any] = (None, None, None)
  if spec.ligand_conditioning or spec.ligand_context_path is not None:
    (structure_id,) = _canonical_structure_ids_for_spec(spec)
    ctx = _prepare_ligand_context(
      spec,  # type: ignore[arg-type]
      ensemble,
      1,
      struct_len,
      canonical_structure_ids=[structure_id],
      batch_structure_ids=[structure_id],
    )
    ligand = (ctx["Y"], ctx["Y_t"], ctx["Y_m"])

  def _ligand_kwargs(lig: tuple[Any, Any, Any]) -> dict[str, Any]:
    if lig[0] is None:
      return {}
    return {"ligand_coords": lig[0], "ligand_atom_types": lig[1], "ligand_mask": lig[2]}

  core = make_score_fn(model)
  padded = []
  for seq in _CANDIDATES:
    idx = string_to_protein_sequence(seq)
    padded.append(jnp.concatenate([idx, jnp.full((struct_len - idx.shape[0],), 20, dtype=idx.dtype)]))
  sequences = jnp.stack(padded)
  # The runner's key tree: one sequential split per (structure, candidate), structure outer.
  prng_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed or 42)
  keys = []
  for _ in _CANDIDATES:
    prng_key, subkey = jax.random.split(prng_key)
    keys.append(subkey)
  keys = jnp.stack(keys)

  def _core(key, seq, c, m, r, ch, lig):  # noqa: ANN001, ANN202
    return core(
      key, jax.nn.one_hot(seq, 21), c, m, r, ch,
      multi_state_strategy=spec.multi_state_strategy, **_ligand_kwargs(lig),
    )

  if batched:

    def _per_structure(c, m, r, ch, ks, lig):  # noqa: ANN001, ANN202
      return jax.vmap(lambda k, s: _core(k, s, c, m, r, ch, lig))(ks, sequences)

    nll, logits, _ = jax.vmap(_per_structure)(
      ensemble.coordinates, ensemble.mask, ensemble.residue_index, ensemble.chain_index,
      keys[None], ligand,
    )
    return np.asarray(nll)[0], np.asarray(logits)[0]

  one_lig = jax.tree.map(lambda a: a[0], ligand)
  outs = [
    _core(
      keys[i], sequences[i], ensemble.coordinates[0], ensemble.mask[0],
      ensemble.residue_index[0], ensemble.chain_index[0], one_lig,
    )
    for i in range(len(_CANDIDATES))
  ]
  return np.stack([np.asarray(o[0]) for o in outs]), np.stack([np.asarray(o[1]) for o in outs])


def _assert_equivalent(spec: ScoringSpecification, *, bitwise_vs_batched: bool) -> np.ndarray:
  """``runner.score`` (encode once) vs the un-hoisted per-candidate core."""
  result = score(spec)
  got_nll = np.asarray(result["scores"])
  got_logits = np.asarray(result["logits"])
  assert got_nll.shape == (1, len(_CANDIDATES))
  assert got_logits.shape == (1, len(_CANDIDATES), _MAX_LENGTH, 21)

  batched_nll, batched_logits = _reference_per_candidate_core(spec, batched=True)
  atol = 0.0 if bitwise_vs_batched else _FP_ATOL
  np.testing.assert_allclose(got_nll[0], batched_nll, rtol=0, atol=atol)
  np.testing.assert_allclose(got_logits[0], batched_logits, rtol=0, atol=atol)

  loop_nll, loop_logits = _reference_per_candidate_core(spec, batched=False)
  np.testing.assert_allclose(got_nll[0], loop_nll, rtol=0, atol=_FP_ATOL)
  np.testing.assert_allclose(got_logits[0], loop_logits, rtol=0, atol=_FP_ATOL)
  return got_nll[0]


@pytest.mark.slow
@pytest.mark.requires_weights
def test_protein_checkpoint_encode_once_matches_per_candidate_core() -> None:
  got = _assert_equivalent(_spec(_PROTEIN_CHECKPOINT), bitwise_vs_batched=True)
  # The candidates must actually differ, or "equal" would be vacuous.
  assert len({round(float(v), 4) for v in got}) == len(_CANDIDATES)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_ligand_checkpoint_with_ligand_encode_once_matches_per_candidate_core(
  tmp_path: Path,
) -> None:
  """The ligand conditions the ENCODER, so the hoisted encode must carry it."""
  (structure_id,) = _canonical_structure_ids_for_spec(_spec(_LIGAND_CHECKPOINT))
  ligand_file = tmp_path / "ligand.npz"
  _write_synthetic_ligand(ligand_file, structure_id)
  with_ligand = _spec(
    _LIGAND_CHECKPOINT, ligand_conditioning=True, ligand_context_path=ligand_file,
  )
  got = _assert_equivalent(with_ligand, bitwise_vs_batched=False)

  # Guard against a vacuous pass: the reference with the ligand differs from ligand-free.
  free_nll, _ = _reference_per_candidate_core(_spec(_LIGAND_CHECKPOINT), batched=True)
  assert not np.allclose(got, free_nll, atol=1e-4), "ligand had no effect on the reference"


@pytest.mark.slow
@pytest.mark.requires_weights
def test_ligand_checkpoint_without_ligand_encode_once_matches_per_candidate_core() -> None:
  _assert_equivalent(_spec(_LIGAND_CHECKPOINT), bitwise_vs_batched=False)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_split_fns_match_fused_core_for_one_candidate() -> None:
  """Unbatched, no runner: encode_structure + score_candidate vs make_score_fn, same key."""
  spec = _spec(_PROTEIN_CHECKPOINT)
  iterator, model = prep_protein_stream_and_model(spec)
  (ensemble,) = list(iterator)
  coords, mask, residue_index, chain_index = (
    ensemble.coordinates[0], ensemble.mask[0], ensemble.residue_index[0], ensemble.chain_index[0],
  )
  idx = string_to_protein_sequence(_UBIQUITIN)
  idx = jnp.concatenate([idx, jnp.full((coords.shape[0] - idx.shape[0],), 20, dtype=idx.dtype)])
  seq = jax.nn.one_hot(idx, 21)
  key = jax.random.PRNGKey(7)

  ref_nll, ref_logits, ref_order = make_score_fn(model)(
    key, seq, coords, mask, residue_index, chain_index,
  )
  split = make_score_split_fns(model)
  enc = split.encode_structure(coords, mask, residue_index, chain_index)
  nll, logits, order = split.score_candidate(key, seq, enc, coords, mask, residue_index, chain_index)

  np.testing.assert_array_equal(np.asarray(nll), np.asarray(ref_nll))
  np.testing.assert_array_equal(np.asarray(logits), np.asarray(ref_logits))
  np.testing.assert_array_equal(np.asarray(order), np.asarray(ref_order))

  # Key-independence of the encoding is what licenses sharing it across candidates.
  enc_again = split.encode_structure(coords, mask, residue_index, chain_index)
  for a, b in zip(jax.tree.leaves(enc), jax.tree.leaves(enc_again), strict=True):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.slow
@pytest.mark.requires_weights
def test_candidate_activation_estimate_covers_the_encoding_not_just_the_logits() -> None:
  """The planner's per-candidate figure must exceed the old output-logits-only count.

  Structural bounds, not a measurement: the estimate counts the encoder output as an argument of
  the lowered candidate function, so it cannot be below the encoding's own bytes, and therefore
  cannot be below the (L, 21) float32 logits the old hand-typed constant counted.
  """
  from aminx.host.runner import _candidate_activation_bytes

  spec = _spec(_PROTEIN_CHECKPOINT)
  iterator, model = prep_protein_stream_and_model(spec)
  (ensemble,) = list(iterator)
  split = make_score_split_fns(model)
  struct_len = ensemble.coordinates.shape[1]
  encoding = jax.eval_shape(
    split.encode_structure,
    ensemble.coordinates[0], ensemble.mask[0], ensemble.residue_index[0], ensemble.chain_index[0],
  )
  encoding_bytes = sum(int(x.size) * x.dtype.itemsize for x in jax.tree.leaves(encoding))

  cache: dict[Any, int] = {}
  args = (
    split, "arithmetic_mean", ensemble, (None, None, None),
    jax.random.PRNGKey(0), jnp.zeros((struct_len,), dtype=jnp.int32),
  )
  one = _candidate_activation_bytes(*args, batch_size=1, estimate_cache=cache)
  assert one >= encoding_bytes
  assert one > struct_len * 21 * 4
  assert len(cache) == 1
  # The structure axis is vmapped outside the candidate axis: the figure scales with it.
  assert _candidate_activation_bytes(*args, batch_size=3, estimate_cache=cache) == 3 * one
  assert len(cache) == 1, "second call with identical shapes must hit the memo"
