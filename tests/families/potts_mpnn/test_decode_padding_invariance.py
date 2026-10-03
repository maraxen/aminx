# ruff: noqa: S101
"""Spec §4.2 (d): autoregressive decode does not depend on how far the input was padded.

THE PROPERTY IS UNEXERCISED IN PRODUCTION, which is why it needs a test rather
than merely an invariant comment. ``prepare_sample`` hardcoded ``l_pad`` to
``L_total``, and ``PottsARDecode`` only runs on the ~20 arrays that function
assembles, so no aminx run has decoded a graph with ``l_pad > L_total``.

WHY IT IS NOT TRIVIAL. The row kernel builds ``nbr_valid`` with
``neighbor_valid(e_idx, pad_valid)`` and passes it as
``DecoderLayer(attention_mask=...)``, which drops messages from pad rows.
Zeroing the gathered context is not enough: ``message_mlp([h_i, 0])`` is not
0, so an unmasked pad neighbour still moves the hidden state. Padding an L=30
chain to 128 also grows K from 30 to 48 (``features.py`` uses
``min(48, L_pad)``), and those extra neighbours are pad rows.

THE CONTROL IS WHAT MAKES THE TEST SHARP. Tokens and ``h_v_stack`` agreeing
across ``l_pad`` is also what you would see if pad rows never reached the
decoder, so the match on its own does not show the mask is load-bearing.
``test_decode_padding_control_fires`` replaces ``neighbor_valid`` with an
all-True mask and requires the l_pad=128 run to leave the rtol 1e-12 band, or
to change a real-row token.

TWO LENGTHS, TWO DIFFERENT MECHANISMS. For ``L_total >= 48`` K does not grow,
and ``top_k`` breaks a distance tie toward the lower index, so pad rows lose the
``D_max`` tie and never enter ``E_idx`` at all. Measured: padded to 128, 0 of 49
real rows have a pad neighbour at L=49, against 30 of 30 at L=30. So at L=49
the message mask has nothing to mask, an all-True mask is a no-op (max abs
0.0), and the control cannot fire there by construction. The L=49 invariance is
instead guarded by ``test_pad_rows_enter_knn_only_below_k``, which asserts the
mechanism directly and uses L=30 as its own control.

WHAT "rtol 1e-12" MEANS HERE, decided 261003 and stated so it is not mistaken
for a loosening. The spec gives a relative bar and no atol. Read elementwise
with ``atol=0`` it is unsatisfiable for any computation whose reduction length
changes: padding grows K from 30 to 48, XLA reassociates the neighbour sum, and
the 14 of 15360 elements with ``|ref| < 1e-3`` then fail on ~1e-16 absolute
noise. The bar is applied per ``h_v_stack`` layer against that layer's scale
instead: ``max|got - ref| <= 1e-12 * max|ref|``. The 1e-12 does not move.
Measured at L=30 -> 128: 8.8e-16 of layer scale, ~1100x inside the bar. The
L=30 control, unmasked, moves 8.3e-2 of layer scale -- ~1e14 separation, so
the reading is not what lets a real defect through.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.decode import ARResult, PottsARDecode
from aminx.families.potts_mpnn.featurize import (
  PottsFeatures,
  parse_pdb_upstream,
  tied_featurize_port,
)
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.potts_mpnn.sample_host import _encode, prepare_sample
from aminx.run.options import PottsMPNNOptions

_THREE = {
  "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
  "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
  "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
  "W": "TRP", "Y": "TYR",
}

#: Spec §4.2 (d), applied per h_v_stack layer against its scale (module docstring).
_RTOL = 1e-12
_WIDE = 128


@pytest.fixture
def x64() -> Iterator[None]:
  """f64 for the duration, restored afterwards.

  Restoring matters: ``jax_enable_x64`` is process-global, and a test that
  leaves it on silently re-types every later test in the session.
  """
  previous = bool(getattr(jax.config, "jax_enable_x64", False))
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", previous)


def _write_helix(path: Path, sequence: str) -> None:
  """A crude jittered helix, so the kNN graph is not degenerate.

  A straight line or a constant-coordinate chain would make many pair distances
  equal and let ``top_k`` break ties arbitrarily, which is the §6.5b
  ``knn_boundary_tie`` case this test must stay clear of.
  """
  lines: list[str] = []
  serial = 1
  rng = np.random.default_rng(0)
  for index, amino in enumerate(sequence, start=1):
    turn = index * 0.6
    base = np.array([np.cos(turn) * 2.3, np.sin(turn) * 2.3, index * 1.5])
    for offset, atom in enumerate(("N", "CA", "C", "O")):
      point = base + rng.normal(scale=0.15, size=3) + offset * 0.4
      lines.append(
        f"{'ATOM':<6.6}{serial:5d} {atom:>4.4} {_THREE[amino]:>3.3} A"
        f"{index:4d}    {point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}",
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _features(tmp_path: Path, l_total: int) -> PottsFeatures:
  sequence = ("ACDEFGHIKLMNPQRSTVWY" * 3)[:l_total]
  pdb = tmp_path / f"helix{l_total}.pdb"
  _write_helix(pdb, sequence)
  parsed = parse_pdb_upstream(str(pdb))[0]
  return tied_featurize_port([parsed], None)[0]


def _decoding_order(l_total: int, l_pad: int) -> np.ndarray:
  """Real rows in one fixed permutation, then ``arange(L_total, l_pad)``.

  ``schedule_groups`` treats an injected order as a permutation of
  ``0 .. L_pad-1``. The same real-row permutation at every pad length is what
  makes the two decodes draw the same residues in the same sequence.
  """
  real = np.random.default_rng(0).permutation(l_total).astype(np.int32)
  tail = np.arange(l_total, l_pad, dtype=np.int32)
  return np.concatenate([real, tail])


def _decode(
  features: PottsFeatures,
  model: PottsMPNN,
  l_pad: int,
  order: np.ndarray,
  uniforms: np.ndarray,
) -> ARResult:
  """Encode then ``PottsARDecode``, with the host's randomness replaced."""
  ready = prepare_sample(
    features,
    (("A", "A" * int(features.L_total)),),
    PottsMPNNOptions(),
    SimpleNamespace(),
    l_pad=l_pad,
  )
  dtype = jnp.float64
  h_v, h_e, e_idx, _forward, _energy = _encode(
    model,
    jnp.asarray(ready.coords, dtype=dtype),
    jnp.asarray(ready.present, dtype=dtype),
    jnp.asarray(ready.residue_idx, dtype=jnp.int32),
    jnp.asarray(ready.chain_index, dtype=jnp.int32),
    jnp.asarray(ready.pad_valid),
  )
  decoder = PottsARDecode(
    layers=model.mpnn.decoder.layers,
    w_s_embed=model.mpnn.w_s_embed,
    w_out=model.mpnn.w_out,
  )
  # Injected ``decoding_order`` is the visit order, so ``randn`` is unread.
  return decoder(
    h_v,
    h_e,
    e_idx,
    jnp.asarray(ready.present, dtype=dtype),
    jnp.asarray(ready.pad_valid),
    jnp.asarray(ready.s_true, dtype=jnp.int32),
    jnp.asarray(ready.chain_mask, dtype=dtype),
    jnp.asarray(ready.chain_m_pos, dtype=dtype),
    jnp.asarray(ready.tie_groups, dtype=jnp.int32),
    jnp.asarray(ready.tied_beta, dtype=dtype),
    jnp.zeros((l_pad,), dtype=dtype),
    jnp.asarray(uniforms, dtype=dtype),
    jnp.asarray(ready.omit, dtype=dtype),
    jnp.asarray(ready.bias, dtype=dtype),
    jnp.asarray(ready.bias_by_res, dtype=dtype),
    jnp.asarray(ready.pssm_coef, dtype=dtype),
    jnp.asarray(ready.pssm_bias, dtype=dtype),
    jnp.asarray(ready.pssm_log_odds_mask, dtype=dtype),
    jnp.asarray(ready.omit_aa_mask, dtype=dtype),
    temperature=0.1,
    pssm_multi=float(PottsMPNNOptions().pssm_multi),
    pssm_bias_flag=bool(PottsMPNNOptions().pssm_bias_flag),
    pssm_log_odds_flag=bool(PottsMPNNOptions().pssm_log_odds_flag),
    decoding_order=jnp.asarray(order, dtype=jnp.int32),
  )


def _real(decoded: ARResult, l_total: int) -> tuple[np.ndarray, np.ndarray]:
  """Tokens and every ``h_v_stack`` layer, restricted to ``pad_valid`` rows."""
  tokens = np.asarray(decoded.sequence[:l_total])
  states = np.asarray(decoded.h_v_stack[:, :l_total], dtype=np.float64)
  return tokens, states


def _layer_relative(got: np.ndarray, ref: np.ndarray) -> float:
  """Worst ``max|got - ref| / max|ref|`` over ``h_v_stack`` layers."""
  scale = np.max(np.abs(ref), axis=(1, 2))
  return float(np.max(np.max(np.abs(got - ref), axis=(1, 2)) / scale))


@pytest.mark.parametrize(("l_total", "l_pad"), [(30, 30), (30, 128), (49, 49), (49, 128)])
def test_decode_invariant_to_padding(
  x64: None, tmp_path: Path, l_total: int, l_pad: int,
) -> None:
  """Real-row tokens match exactly; ``pad_valid`` hidden states stay in rtol."""
  del x64
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  features = _features(tmp_path, l_total)
  assert int(features.L_total) == l_total
  draws = np.random.default_rng(1).random(_WIDE).astype(np.float64)
  base_tokens, base_states = _real(
    _decode(
      features, model, l_total, _decoding_order(l_total, l_total), draws[:l_total],
    ),
    l_total,
  )
  got_tokens, got_states = _real(
    _decode(features, model, l_pad, _decoding_order(l_total, l_pad), draws[:l_pad]),
    l_total,
  )
  assert np.array_equal(got_tokens, base_tokens), (
    f"L_total={l_total} padded to {l_pad} changed a real-row token"
  )
  relative = _layer_relative(got_states, base_states)
  assert relative <= _RTOL, (
    f"L_total={l_total} padded to {l_pad} moved pad_valid h_v_stack by "
    f"{relative:.3e} of layer scale, outside {_RTOL:.0e}"
  )


def test_decode_padding_control_fires(
  x64: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """An unmasked pad neighbour must move the padded decode.

  WITHOUT THIS, ``test_decode_invariant_to_padding`` IS VACUOUS at L=30.
  Agreement across ``l_pad`` is equally consistent with a correct message mask
  and with pad rows never appearing in ``E_idx``. Only L=30 is run: at L=49 pad
  rows never enter ``E_idx``, so there is nothing to unmask and the patch is a
  no-op by construction. That length is covered by
  ``test_pad_rows_enter_knn_only_below_k`` instead.
  """
  del x64
  l_total = 30

  def _all_true(e_idx: jax.Array, pad_valid: jax.Array) -> jax.Array:
    del pad_valid
    return jnp.ones(e_idx.shape, dtype=jnp.bool_)

  # ``PottsARDecode.__call__`` reads ``neighbor_valid`` from decode's globals.
  monkeypatch.setattr("aminx.families.potts_mpnn.decode.neighbor_valid", _all_true)
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  features = _features(tmp_path, l_total)
  assert int(features.L_total) == l_total
  draws = np.random.default_rng(1).random(_WIDE).astype(np.float64)
  base_tokens, base_states = _real(
    _decode(
      features, model, l_total, _decoding_order(l_total, l_total), draws[:l_total],
    ),
    l_total,
  )
  wide_tokens, wide_states = _real(
    _decode(
      features, model, _WIDE, _decoding_order(l_total, _WIDE), draws[:_WIDE],
    ),
    l_total,
  )
  tokens_same = bool(np.array_equal(wide_tokens, base_tokens))
  relative = _layer_relative(wide_states, base_states)
  assert not (tokens_same and relative <= _RTOL), (
    f"disabling neighbor_valid on L_total={l_total} padded to {_WIDE} left "
    f"real-row tokens identical and h_v_stack within {relative:.3e} of layer "
    f"scale (bar {_RTOL:.0e}). The mask is not reaching the decoder, so "
    f"test_decode_invariant_to_padding proves nothing."
  )


@pytest.mark.parametrize(("l_total", "expect_pad_neighbours"), [(30, True), (49, False)])
def test_pad_rows_enter_knn_only_below_k(
  x64: None, tmp_path: Path, l_total: int, expect_pad_neighbours: bool,
) -> None:
  """Why L=49 needs no message mask, asserted rather than assumed.

  Padded to 128, K is ``min(48, L_pad) = 48`` either way. At L=49 there are
  more real rows than K and pad rows sit at ``D_max``, so they lose every tie
  and no real row gets a pad neighbour. That, not the mask, is what keeps the
  L=49 decode invariant. L=30 is the control: with 30 real rows and K=48, every
  real row MUST pull in pad neighbours, or this test cannot tell the two
  mechanisms apart.
  """
  del x64
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  features = _features(tmp_path, l_total)
  ready = prepare_sample(
    features,
    (("A", "A" * l_total),),
    PottsMPNNOptions(),
    SimpleNamespace(),
    l_pad=_WIDE,
  )
  _h_v, _h_e, e_idx, _forward, _energy = _encode(
    model,
    jnp.asarray(ready.coords, dtype=jnp.float64),
    jnp.asarray(ready.present, dtype=jnp.float64),
    jnp.asarray(ready.residue_idx, dtype=jnp.int32),
    jnp.asarray(ready.chain_index, dtype=jnp.int32),
    jnp.asarray(ready.pad_valid),
  )
  real = np.asarray(e_idx)[:l_total]
  n_with_pad = int((real >= l_total).any(axis=1).sum())
  if expect_pad_neighbours:
    assert n_with_pad == l_total, (
      f"control: L={l_total} padded to {_WIDE} gave only {n_with_pad} real rows "
      f"a pad neighbour; K={real.shape[1]} > L should force every row to have one"
    )
  else:
    assert n_with_pad == 0, (
      f"L={l_total} padded to {_WIDE}: {n_with_pad} real rows have a pad "
      f"neighbour, so the L=49 invariance now depends on the message mask and "
      f"test_decode_padding_control_fires must cover this length too"
    )


def test_prepare_sample_rejects_short_l_pad(tmp_path: Path) -> None:
  """``l_pad < L_total`` is an error; the default stays ``L_total``."""
  features = _features(tmp_path, 30)
  chains = (("A", "A" * int(features.L_total)),)
  options = PottsMPNNOptions()
  with pytest.raises(ValueError, match="l_pad"):
    prepare_sample(features, chains, options, SimpleNamespace(), l_pad=int(features.L_total) - 1)
  ready = prepare_sample(features, chains, options, SimpleNamespace())
  assert ready.coords.shape[0] == int(features.L_total)
  assert ready.pad_valid.shape[0] == int(features.L_total)
