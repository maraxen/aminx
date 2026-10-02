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


import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature
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


def _decode_with_pssm(
  readout_bias: jax.Array,
  *,
  pssm_bias_row: jax.Array,
  pssm_coef_value: float,
  pssm_multi: float,
  pssm_bias_flag: bool,
  pssm_log_odds_row: jax.Array | None = None,
  pssm_log_odds_flag: bool = False,
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
    jnp.zeros((_V,), dtype=jnp.float32),
    jnp.zeros((_V,), dtype=jnp.float32),
    zeros_lv,
    jnp.full((length,), pssm_coef_value, dtype=jnp.float32),
    pssm_bias_row[None, :],
    zeros_lv if pssm_log_odds_row is None else pssm_log_odds_row[None, :],
    zeros_lv,
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
  drawn = _decode_with_pssm(
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
  drawn = _decode_with_pssm(
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
  kept = _decode_with_pssm(
    _peaked(3), pssm_bias_row=_onehot(7), pssm_coef_value=1.0,
    pssm_multi=0.0, pssm_bias_flag=True,
  )
  replaced = _decode_with_pssm(
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

  without = _decode_with_pssm(
    bias, pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32), pssm_coef_value=0.0,
    pssm_multi=0.0, pssm_bias_flag=False,
    pssm_log_odds_row=mask, pssm_log_odds_flag=False,
  )
  with_flag = _decode_with_pssm(
    bias, pssm_bias_row=jnp.zeros((_V,), dtype=jnp.float32), pssm_coef_value=0.0,
    pssm_multi=0.0, pssm_bias_flag=False,
    pssm_log_odds_row=mask, pssm_log_odds_flag=True,
  )
  assert without == 3, f"unset, the readout peak wins; got {without}"
  assert with_flag == 7, f"set, the mask moves the draw; got {with_flag}"
