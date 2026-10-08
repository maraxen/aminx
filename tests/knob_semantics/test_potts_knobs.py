# ruff: noqa: S101
"""PottsMPNN knob semantics.

The LASEr half of this directory covers ``LaserOptions``; every
``test_knob_semantics_*`` test was LASEr-side until now, so all 22
``PottsMPNNOptions`` fields were uncovered and
``tests/knob_gate/test_parity_ids_passed`` could not pass for any of them. This
file starts the Potts half, taking the knobs the spec pins to an exact upstream
anchor.

Each test pairs the aminx behaviour against a local oracle transcribed from the
pinned upstream (PottsMPNN ``0cb0a58``) and a negative control that must NOT
agree, so the test cannot pass by accident. Per ``~/.claude/rules/BATHOS.md``, a
positive control that can only pass is not a check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature
from aminx.families.potts_mpnn.refine import BindingTables, PottsRefine
from aminx.model.decoder import DecoderLayer
from aminx.families.potts_mpnn.featurize import _AA_N_1, _BACKBONE, _parse_biounits

_GAP_CHAR = _AA_N_1[20]


def _oracle_floor_temperature(temperature: float) -> float:
  """Upstream ``sample_seqs.py:38-39``, transcribed.

  Upstream rewrites the temperature in place before the draw::

      if cfg.inference.temperature == 0:
          cfg.inference.temperature = 1e-6

  and does the same for ``optimization_temperature``. The floor is not cosmetic:
  the value goes on to divide the logits, so leaving a literal zero there yields a
  degenerate distribution rather than the argmax the user asked for. That is the
  negative control below.
  """
  if temperature == 0:
    return 1e-6
  return float(temperature)


def test_knob_semantics_t0_floor() -> None:
  """``temperature`` and ``optimization_temperature`` floor 0 to 1e-6 before the draw.

  Spec 6.5, upstream ``sample_seqs.py:38-39``. The host applies this to every
  element of the ``temperatures`` axis and to ``optimization_temperature``
  separately (``sample_host.py:506-512``), so one semantic covers both knobs.

  Deliberately not parametrized: alias_map.toml rows cite nodeids, and a
  parametrized id carries the value in brackets, so adding a case would silently
  invalidate a ledger reference.
  """
  for value in (0.0, -0.0, 1e-6, 0.1, 1.0, 2.5):
    assert floor_temperature(value) == _oracle_floor_temperature(value), value


def test_knob_semantics_t0_floor_is_load_bearing() -> None:
  """Negative control: without the floor the draw degenerates instead of sharpening.

  Spec 2.3 reads ``temperature == 0`` as argmax. Dividing by a literal zero does
  not produce argmax -- it produces non-finite logits -- so a floor that returned
  0 unchanged would be a silently wrong answer, not a near-equivalent one. This
  pins why 1e-6 is the correct port and guards against the floor being removed as
  redundant.
  """
  logits = np.asarray([1.0, 3.0, 2.0], dtype=np.float64)

  floored = logits / floor_temperature(0.0)
  assert np.all(np.isfinite(floored))
  assert int(np.argmax(floored)) == int(np.argmax(logits))

  with np.errstate(divide="ignore", invalid="ignore"):
    unfloored = logits / 0.0
  assert not np.all(np.isfinite(unfloored)), (
    "dividing by an unfloored zero must be non-finite; if this ever becomes finite "
    "the negative control has stopped discriminating and this test is vacuous"
  )


def _write_pdb(path: Path, residues: tuple[tuple[int, str], ...]) -> Path:
  """Write one chain A with the given (residue number, residue name) pairs.

  The parser slices the fixed-width record directly, so the columns have to be
  exact. An earlier draft of this helper put resName at 16:19 -- one short of the
  real layout, which reserves column 16 for altLoc -- and that pushed chainID from
  21 to 20, so ``_parse_biounits`` matched no chain and returned ``'no_chain'``
  for every input. The layout below is spelled out to keep that from recurring::

      0-5 record  6-10 serial  11 blank  12-15 atom  16 altLoc  17-19 resName
      20 blank  21 chainID  22-25 resSeq  26 iCode  30-37 x  38-45 y  46-53 z
  """
  lines: list[str] = []
  serial = 1
  for resnum, resname in residues:
    for index, atom in enumerate(_BACKBONE):
      x = float(resnum) + index * 0.5
      lines.append(
        f"ATOM  {serial:>5} {atom:<4} {resname:>3} A{resnum:>4}    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00",
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def test_knob_semantics_skip_gaps(tmp_path: Path) -> None:
  """``skip_gaps`` drops unobserved residue numbers instead of emitting a gap row.

  Spec 6.1 / 6.3, upstream ``parse_PDB_biounits``. Unset, a number absent from the
  file still becomes a residue: alphabet slot 20 and all-NaN backbone coordinates,
  so the chain keeps its numbering. Set, the position is skipped and the chain
  shortens. The fixture numbers residues 1, 2 and 4, so exactly one gap exists.
  """
  pdb = _write_pdb(tmp_path / "gapped.pdb", ((1, "ALA"), (2, "GLY"), (4, "SER")))

  kept_coords, kept_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=False)
  skipped_coords, skipped_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=True)
  assert isinstance(kept_coords, np.ndarray)
  assert isinstance(skipped_coords, np.ndarray)

  # Unset: the gap is materialised as a residue.
  assert kept_seq == f"AG{_GAP_CHAR}S", kept_seq
  assert kept_coords.shape == (4, len(_BACKBONE), 3)
  assert np.all(np.isnan(kept_coords[2])), "the gap row's coordinates must be NaN"
  assert not np.any(np.isnan(kept_coords[[0, 1, 3]])), "observed rows must be finite"

  # Set: the gap is dropped and nothing else moves.
  assert skipped_seq == "AGS", skipped_seq
  assert skipped_coords.shape == (3, len(_BACKBONE), 3)
  assert not np.any(np.isnan(skipped_coords)), "no NaN row survives skip_gaps"
  np.testing.assert_array_equal(skipped_coords, kept_coords[[0, 1, 3]])

  # The knob is not a no-op: set and unset disagree in both outputs.
  assert kept_seq != skipped_seq
  assert kept_coords.shape != skipped_coords.shape


def test_knob_semantics_skip_gaps_inert_without_a_gap(tmp_path: Path) -> None:
  """Negative control: on contiguous numbering the knob must change nothing.

  A differential test that fired on every input would be measuring the parser
  rather than the knob, so the same comparison is run on a chain with no gap and
  must come out identical.
  """
  pdb = _write_pdb(
    tmp_path / "contiguous.pdb", ((1, "ALA"), (2, "GLY"), (3, "SER")),
  )

  kept_coords, kept_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=False)
  skipped_coords, skipped_seq = _parse_biounits(pdb, _BACKBONE, "A", skip_gaps=True)
  assert isinstance(kept_coords, np.ndarray)
  assert isinstance(skipped_coords, np.ndarray)

  assert kept_seq == skipped_seq == "AGS"
  np.testing.assert_array_equal(kept_coords, skipped_coords)
  assert _GAP_CHAR not in kept_seq


# ---------------------------------------------------------------------------
# PSSM knobs. These drive PottsARDecode rather than pssm_mix directly, because
# the knob under test is the GATE that decides whether mixing happens at all,
# and that expression lives inside __call__ (decode.py:297). Testing pssm_mix
# with the gate passed in by hand would assert my own reading of the gate.
# ---------------------------------------------------------------------------

_H = 4
_V = 21


def _zero_floats(module: eqx.Module) -> eqx.Module:
  def _leaf(leaf: object) -> object:
    if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating):
      return jnp.zeros(leaf.shape, leaf.dtype)
    return leaf

  return jax.tree.map(_leaf, module)


def _decoder(readout_bias: jax.Array) -> PottsARDecode:
  """A decoder whose logits are exactly ``readout_bias``, independent of input.

  Every float leaf of the message path is zeroed and the readout weight is zero,
  so ``w_out`` contributes only its bias. That makes the pre-PSSM distribution
  something the test states outright instead of something it has to measure.
  """
  layer = DecoderLayer(_H, 3 * _H, _H, dropout_rate=0.0, key=jax.random.PRNGKey(0))
  layer = eqx.tree_at(
    lambda item: (item.message_mlp, item.dense),
    layer,
    (_zero_floats(layer.message_mlp), _zero_floats(layer.dense)),
  )
  embed = eqx.nn.Embedding(_V, _H, key=jax.random.PRNGKey(1))
  readout = eqx.nn.Linear(_H, _V, key=jax.random.PRNGKey(2))
  readout = eqx.tree_at(
    lambda item: (item.weight, item.bias),
    readout,
    (jnp.zeros_like(readout.weight), readout_bias),
  )
  return PottsARDecode(layers=(layer,), w_s_embed=embed, w_out=readout)


def _decode_one(
  readout_bias: jax.Array,
  *,
  pssm_bias_row: jax.Array,
  pssm_coef_value: float,
  pssm_multi: float,
  pssm_bias_flag: bool,
  pssm_log_odds_row: jax.Array | None = None,
  pssm_log_odds_flag: bool = False,
  omit_row: jax.Array | None = None,
  omit_aa_mask_row: jax.Array | None = None,
) -> int:
  """Decode one residue and return the token drawn."""
  length = 1
  zeros_lv = jnp.zeros((length, _V), dtype=jnp.float32)
  decoded = _decoder(readout_bias)(
    jnp.zeros((length, _H), dtype=jnp.float32),
    jnp.zeros((length, 1, _H), dtype=jnp.float32),
    jnp.zeros((length, 1), dtype=jnp.int32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.ones((length,), dtype=jnp.bool_),
    jnp.zeros((length,), dtype=jnp.int32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.asarray([[0]], dtype=jnp.int32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.zeros((length,), dtype=jnp.float32),
    jnp.full((length,), 0.5, dtype=jnp.float32),
    jnp.zeros((_V,), dtype=jnp.float32) if omit_row is None else omit_row,
    jnp.zeros((_V,), dtype=jnp.float32),
    zeros_lv,
    jnp.full((length,), pssm_coef_value, dtype=jnp.float32),
    pssm_bias_row[None, :],
    zeros_lv if pssm_log_odds_row is None else pssm_log_odds_row[None, :],
    zeros_lv if omit_aa_mask_row is None else omit_aa_mask_row[None, :],
    temperature=1.0,
    pssm_multi=pssm_multi,
    pssm_bias_flag=pssm_bias_flag,
    pssm_log_odds_flag=pssm_log_odds_flag,
  )
  return int(decoded.sequence[0])


def _peaked(index: int, *, height: float = 50.0, floor: float = -50.0) -> jax.Array:
  return jnp.full((_V,), floor, dtype=jnp.float32).at[index].set(height)


def _onehot(index: int) -> jax.Array:
  return jnp.zeros((_V,), dtype=jnp.float32).at[index].set(1.0)


def test_knob_semantics_pssm_precedence() -> None:
  """PSSM mixing fires on a non-empty ``pssm_bias`` even with the flag unset.

  Spec 6.5 and upstream ``potts_mpnn_utils.py:1392``::

      if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):

  ``and`` binds tighter than ``or``, so that parses as
  ``(flag and coef_nonempty) or bias_nonempty`` -- a missing pair of parentheses.
  The author plainly meant ``flag and (coef_nonempty or bias_nonempty)``. aminx
  reproduces the quirk deliberately at decode.py:297, and because aminx always
  materialises ``pssm_bias`` as an ``(L, V)`` array the gate is in fact always
  open; what keeps it harmless by default is ``pssm_coef * pssm_multi == 0``,
  not the flag.

  The readout peaks at token 3 and the PSSM bias is one-hot on token 7, with
  ``coef * multi == 1``, so mixing replaces the distribution outright and the draw
  moves 3 -> 7. The intended reading would leave it at 3, so the two differ in the
  observable and this test discriminates between them.
  """
  intended_reading_token = 3
  drawn = _decode_one(
    _peaked(intended_reading_token),
    pssm_bias_row=_onehot(7),
    pssm_coef_value=1.0,
    pssm_multi=1.0,
    pssm_bias_flag=False,
  )
  assert drawn == 7, (
    f"mixing did not fire with pssm_bias_flag=False; got {drawn}. aminx must keep "
    f"the upstream precedence quirk, not the intended reading."
  )
  assert drawn != intended_reading_token


def test_knob_semantics_pssm_precedence_latent_without_weight() -> None:
  """Negative control: the open gate is unobservable while ``coef * multi`` is 0.

  Without this, the test above would pass for a port that mixed unconditionally
  AND one that ignored the weights. Same inputs, ``pssm_multi = 0``: the draw must
  stay on the readout peak, which shows the previous test turns on the weight
  rather than on mixing being hard-wired.
  """
  drawn = _decode_one(
    _peaked(3),
    pssm_bias_row=_onehot(7),
    pssm_coef_value=1.0,
    pssm_multi=0.0,
    pssm_bias_flag=False,
  )
  assert drawn == 3, f"an all-zero mixing weight must change nothing; got {drawn}"


def test_knob_semantics_pssm_multi() -> None:
  """``pssm_multi`` is the mixing weight, with ``pssm_coef``, on the PSSM bias.

  Upstream ``probs = (1 - pssm_multi*coef)*probs + pssm_multi*coef*pssm_bias``
  (``potts_mpnn_utils.py:1395``). Swept at the extremes with the flag set, so the
  knob alone moves the draw: 0 keeps the readout peak, 1 replaces it entirely.
  """
  kept = _decode_one(
    _peaked(3), pssm_bias_row=_onehot(7), pssm_coef_value=1.0,
    pssm_multi=0.0, pssm_bias_flag=True,
  )
  replaced = _decode_one(
    _peaked(3), pssm_bias_row=_onehot(7), pssm_coef_value=1.0,
    pssm_multi=1.0, pssm_bias_flag=True,
  )
  assert kept == 3
  assert replaced == 7
  assert kept != replaced, "pssm_multi must not be inert"


def test_knob_semantics_pssm_log_odds_flag() -> None:
  """``pssm_log_odds_flag`` renormalises by the mask with upstream's 0.001 floor.

  Upstream ``potts_mpnn_utils.py:1396-1400``: ``probs*mask + probs*0.001``, then
  renormalise -- so a masked-out token keeps a thousandth of its weight rather
  than vanishing. Two comparable peaks are used (tokens 3 and 7, one nat apart)
  because the floor means a mask cannot overcome an arbitrarily dominant token;
  with the usual 100-nat peak the renormalised argmax would not move and the test
  would be vacuous.
  """
  bias = jnp.full((_V,), -50.0, dtype=jnp.float32).at[3].set(1.0).at[7].set(0.0)
  mask = _onehot(7)

  without = _decode_one(
    bias, pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32), pssm_coef_value=0.0,
    pssm_multi=0.0, pssm_bias_flag=False,
    pssm_log_odds_row=mask, pssm_log_odds_flag=False,
  )
  with_flag = _decode_one(
    bias, pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32), pssm_coef_value=0.0,
    pssm_multi=0.0, pssm_bias_flag=False,
    pssm_log_odds_row=mask, pssm_log_odds_flag=True,
  )
  assert without == 3, f"unset, the readout peak wins; got {without}"
  assert with_flag == 7, f"set, the mask moves the draw; got {with_flag}"


def test_knob_semantics_omit_aa() -> None:
  """``omit_aa`` subtracts a large constant from the omitted letters' logits.

  Upstream ``potts_mpnn_utils.py:1392`` builds the softmax argument as
  ``logits - constant*1e8 + ...``, where ``constant`` is the omit indicator, so an
  omitted letter is pushed far below every other before the softmax rather than
  being removed afterwards. The global ``--omit_AAs`` list is this vector
  (spec 6.3: ``inference__omit_AAs`` -> ``omit_aa``).

  The readout peaks at token 3; omitting 3 must move the draw off it.
  """
  peak = 3
  kept = _decode_one(
    _peaked(peak), pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32),
    pssm_coef_value=0.0, pssm_multi=0.0, pssm_bias_flag=False,
  )
  omitted = _decode_one(
    _peaked(peak), pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32),
    pssm_coef_value=0.0, pssm_multi=0.0, pssm_bias_flag=False,
    omit_row=_onehot(peak),
  )
  assert kept == peak, f"without omit the peak wins; got {kept}"
  assert omitted != peak, (
    f"omitting the peak must move the draw off it; got {omitted}. If this ever "
    f"equals the peak, the omit vector is not reaching the softmax argument."
  )


def test_knob_semantics_omit_aa_per_position() -> None:
  """``omit_aa_per_position`` zeroes letters AFTER the softmax, then renormalises.

  A different mechanism from ``omit_aa``, and the ordering is the point. Upstream
  applies ``probs*(1 - mask)`` and renormalises at the end of the chain
  (``potts_mpnn_utils.py:1401-1405``), i.e. after PSSM mixing and log-odds, where
  ``omit_aa`` acts on the logits before the softmax. Spec 6.3:
  ``inference__omit_AA_json`` -> ``omit_aa_per_position``.

  aminx always materialises the mask as an ``(L, V)`` array, so the gate
  ``omit_aa_mask.size > 0`` (decode.py:299) is always open; an all-zero mask is
  what makes it a no-op, which the first leg below pins.
  """
  peak = 3
  kept = _decode_one(
    _peaked(peak), pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32),
    pssm_coef_value=0.0, pssm_multi=0.0, pssm_bias_flag=False,
    omit_aa_mask_row=jnp.zeros((_V,), dtype=jnp.float32),
  )
  masked = _decode_one(
    _peaked(peak), pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32),
    pssm_coef_value=0.0, pssm_multi=0.0, pssm_bias_flag=False,
    omit_aa_mask_row=_onehot(peak),
  )
  assert kept == peak, f"an all-zero mask must change nothing; got {kept}"
  assert masked != peak, f"masking the peak must move the draw; got {masked}"


# ---------------------------------------------------------------------------
# optimization_mode. PottsRefine dispatches on the mode string, so the knob is
# the dispatch itself: each value must select a visibly different algorithm, and
# a value outside the Literal must be refused rather than silently defaulted.
# ---------------------------------------------------------------------------

_A = 22


def _binding_tables(length: int, alphabet: int, etab: jax.Array | None = None) -> BindingTables:
  table = jnp.zeros((1, length, 1, alphabet, alphabet)) if etab is None else etab
  return BindingTables(
    etab=table,
    e_idx=jnp.zeros((1, length, 1), dtype=jnp.int32),
    pad_valid=jnp.ones((1, length), dtype=jnp.bool_),
    complex_index=jnp.zeros((1, length), dtype=jnp.int32),
    partition_of=jnp.zeros((length,), dtype=jnp.int32),
    local_of=jnp.arange(length, dtype=jnp.int32),
    inter_mask=jnp.ones((length,), dtype=jnp.bool_),
  )


def _refine(
  mode: str,
  etab: jax.Array,
  *,
  max_iters: int = 5,
  bias_index: int | None = None,
  chain_mask: np.ndarray | None = None,
  sequence: np.ndarray | None = None,
  tie_groups: np.ndarray | None = None,
  tied: bool = False,
  tied_epistasis: bool = False,
  e_idx: np.ndarray | None = None,
) -> object:
  """Refine with the zeroed-message decoder, so logits come only from the bias."""
  length = etab.shape[0]
  module = _decoder(jnp.zeros((_V,), dtype=jnp.float32))
  refiner = PottsRefine(
    layers=module.layers, w_s_embed=module.w_s_embed, w_out=module.w_out,
  )
  seq = np.zeros(length, dtype=np.int32) if sequence is None else sequence
  mask = np.ones(length, dtype=np.float32) if chain_mask is None else chain_mask
  bias = np.zeros(_V, dtype=np.float32)
  if bias_index is not None:
    bias[bias_index] = 50.0
  return refiner(
    mode,
    jnp.asarray(seq),
    etab,
    jnp.zeros((length, 1), dtype=jnp.int32)
    if e_idx is None
    else jnp.asarray(e_idx, dtype=jnp.int32),
    jnp.ones((length,), dtype=jnp.bool_),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.asarray(mask),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.arange(length, dtype=jnp.int32),
    jnp.full((max_iters, length), 0.5),
    jnp.zeros((_V,), dtype=jnp.float32),
    jnp.asarray(bias),
    jnp.zeros((length, _V), dtype=jnp.float32),
    jnp.zeros((length,), dtype=jnp.float32),
    jnp.zeros((length, _V), dtype=jnp.float32),
    jnp.zeros((length, _V), dtype=jnp.float32),
    jnp.zeros((length, _V), dtype=jnp.float32),
    jnp.arange(length, dtype=jnp.int32)[:, None]
    if tie_groups is None
    else jnp.asarray(tie_groups, dtype=jnp.int32),
    jnp.ones((length,), dtype=jnp.float32),
    jnp.zeros((length, _H), dtype=jnp.float32),
    jnp.zeros((length, 1, _H), dtype=jnp.float32),
    _binding_tables(length, etab.shape[-1]),
    temperature=1.0,
    pssm_multi=0.0,
    pssm_bias_flag=False,
    pssm_log_odds_flag=False,
    binding="none",
    tied=tied,
    tied_epistasis=tied_epistasis,
    max_iters=max_iters,
  )


def test_knob_semantics_optimization_mode() -> None:
  """Each ``optimization_mode`` value selects a visibly different algorithm.

  Spec 6.1 types the knob as ``Literal["none", "potts", "potts_converge",
  "nodes"]``. ``none`` is decided upstream of PottsRefine -- the host simply does
  not refine -- so the three values that reach the module are compared here on one
  input, and must not agree.

  ``potts`` sweeps a fixed number of times; ``potts_converge`` stops when the
  accumulated energy stops moving (upstream ``run_utils.py:116-120``, and note it
  accumulates the ABSOLUTE energy in the non-binding case, so a flat table
  converges on the first sweep); ``nodes`` is the Gibbs variant that honours
  ``chain_mask``, leaving a fixed row at its input residue.
  """
  length = 2
  flat = jnp.zeros((length, 1, _A, _A))

  swept = _refine("potts", flat, bias_index=3)
  assert np.array_equal(np.asarray(swept.sequence), np.asarray([3, 3]))

  converged = _refine("potts_converge", flat)
  assert int(converged.n_iters) == 1, (
    "a flat table has nothing to move, so absolute-energy accumulation must stop "
    "after one sweep"
  )
  assert float(converged.ener_delta) == 0.0

  identity = jnp.zeros((length, 1, _A, _A)).at[:, 0].set(jnp.eye(_A))
  capped = _refine("potts_converge", identity, max_iters=3)
  assert int(capped.n_iters) == 3, (
    "a table that keeps changing the energy must run to the iteration cap; if this "
    "equals 1 the convergence test is firing on everything and is not a check"
  )

  gibbs = _refine(
    "nodes",
    flat,
    bias_index=3,
    sequence=np.asarray([4, 4], dtype=np.int32),
    chain_mask=np.asarray([0.0, 1.0], dtype=np.float32),
  )
  assert np.array_equal(np.asarray(gibbs.sequence), np.asarray([4, 3])), (
    "nodes must leave the chain_mask=0 row at its input residue and redesign the "
    "other"
  )

  # The knob is not inert: converge stops earlier than the cap that potts uses,
  # and nodes alone respects the fixed row.
  assert int(converged.n_iters) != int(capped.n_iters)
  assert not np.array_equal(np.asarray(gibbs.sequence), np.asarray(swept.sequence))


def test_knob_semantics_optimization_mode_rejects_unknown() -> None:
  """Negative control: a mode outside the Literal raises instead of defaulting.

  ``refine.py:188``. A dispatch that fell through to a default would make the knob
  silently wrong for a typo, which is worse than an error.
  """
  with pytest.raises(ValueError, match="unknown optimization_mode"):
    _refine("potts_converg", jnp.zeros((2, 1, _A, _A)))


def _tied_pair_refine(first_member_pull: float, *, epistasis: bool) -> np.ndarray:
  """Refine one tied group of two rows, varying ONLY the first member's table.

  The energy a member contributes for candidate ``aa`` is
  ``etab[member, 0, aa, aa]``: ``positional_potts_energy`` takes
  ``jnp.diagonal(etab[pos, 0])`` for slot 0 and draws pair terms from
  ``e_idx[pos, 1:]`` (etab.py:173-174), so with ``K == 1`` there are no neighbour
  couplings and only the slot-0 DIAGONAL is read. Lower energy is preferred. The
  last member pulls towards token 5; the first member pulls towards token 7 by
  ``first_member_pull``, which is the only thing that varies between runs.

  Two earlier fixtures failed here and both failures were mine, not the code's:
  off-diagonal entries are never read with ``K == 1``, and ``eye(_A) * scale``
  perturbs a single candidate while leaving the other 21 exactly tied, so the draw
  lands on the tie midpoint however the scale moves.
  """
  length = 2
  etab = jnp.zeros((length, 1, _A, _A))
  etab = etab.at[0, 0, 7, 7].set(first_member_pull)
  etab = etab.at[1, 0, 5, 5].set(-10.0)
  result = _refine(
    "potts",
    etab,
    max_iters=1,
    tie_groups=np.asarray([[0, 1]], dtype=np.int32),
    tied=True,
    tied_epistasis=epistasis,
  )
  return np.asarray(result.sequence)


def test_knob_semantics_tied_epistasis_leaked_pos() -> None:
  """``tied_epistasis`` reads the group's energy at the LAST member only.

  Spec 6.5, upstream ``run_utils.py:349-361``. Mutants are set across the whole
  tied group jointly, but the positional energy is read at the loop variable left
  over from the member loop -- so it is the last listed member's, not the group's.
  aminx keeps this at ``refine.py:782``: set, the energy comes from
  ``_energy_vector(..., last_safe, ...)``; unset, from
  ``_average_member_energy(..., members, ...)``.

  The discriminator needs no prediction of which token is drawn. Vary ONLY the
  FIRST member's table between two runs:

    set    the first member's energy is never read, so the result must not move
    unset  the average includes it, so the result must move

  The second leg is the first one's control: together they rule out both a port
  that always averages and a port that always reads one member, without this test
  having to recompute an energy itself.
  """
  epistasis_low = _tied_pair_refine(0.0, epistasis=True)
  epistasis_high = _tied_pair_refine(-40.0, epistasis=True)
  assert np.array_equal(epistasis_low, epistasis_high), (
    f"with tied_epistasis the first member's table must be ignored, but changing "
    f"it moved the result: {epistasis_low} -> {epistasis_high}"
  )

  averaged_low = _tied_pair_refine(0.0, epistasis=False)
  averaged_high = _tied_pair_refine(-40.0, epistasis=False)
  assert not np.array_equal(averaged_low, averaged_high), (
    f"without tied_epistasis the energy is averaged over members, so changing the "
    f"first member's table must move the result; got {averaged_low} both times. If "
    f"this ever holds, the test above is passing for the wrong reason -- both "
    f"branches would be ignoring the member."
  )
