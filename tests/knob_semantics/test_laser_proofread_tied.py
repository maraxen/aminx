"""Proofread reduction and tied-sample knob semantics.

The ten section 5.3.1 tests stay in ``test_laser_joint_decode.py``. These cover
the B7 half: dropout reduction, the tied χ retention rule, and the LaserOptions
fields that half reads.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.laser_mpnn.driver import (
  LaserDriver,
  _assert_strict_keys,
  budget_mask_for,
)
from aminx.families.laser_mpnn.featurize import LaserInputError, featurize
from aminx.model.laser.decoder import LaserDecoder
from aminx.model.laser.encoders import LaserEncoder
from aminx.model.laser.graphs import GraphStructure
from aminx.model.laser.joint_decode import (
  LaserJointDecode,
  categorical_draw,
  decode_order,
  minp_warp_logits,
)
import aminx.model.laser.proofread as proofread
from aminx.model.laser.proofread import (
  closed_form_std,
  conditional_focus_probs,
  focus_rows,
  proofread_dropout,
  reduce_proofread,
)
import aminx.model.laser.tied as tied_mod
from aminx.model.laser.tied import (
  TiedDecodeResult,
  mix_sequence_probabilities,
  tied_decode,
)
from aminx.run.options import LaserOptions

_VAL = 19
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "laser" / "101m_1.pdb"


@functools.cache
def _models() -> tuple[LaserEncoder, LaserDecoder, LaserJointDecode]:
  return (
    LaserEncoder(key=jax.random.PRNGKey(0)),
    LaserDecoder(key=jax.random.PRNGKey(1)),
    LaserJointDecode(key=jax.random.PRNGKey(2)),
  )


def _structure() -> GraphStructure:
  return GraphStructure(
    pr_pr_knn_graph_k=4,
    lig_pr_knn_graph_k=4,
    lig_lig_knn_graph_k=2,
    lig_pr_distance_cutoff=20.0,
  )


def _coords(n_res: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  backbone = np.zeros((n_res, 5, 3), dtype=np.float32)
  for index in range(n_res):
    backbone[index, :, 0] = index * 3.8 + shift
    backbone[index, 1, 1] = 1.5
  ligand = np.asarray([[shift, 8.0, 0.0]], dtype=np.float32)
  atomic = np.asarray([6], dtype=np.int32)
  subbatch = np.asarray([0], dtype=np.int32)
  return backbone, ligand, atomic, subbatch


def test_knob_semantics_closed_form_proofread_std() -> None:
  # The identity is exact in f64. Production stays f32; this test opts in.
  previous = bool(getattr(jax.config, "jax_enable_x64", False))
  jax.config.update("jax_enable_x64", True)
  try:
    rng = np.random.default_rng(0)
    p1 = rng.random((3, 21))
    p2 = rng.random((3, 21))
    stacked = np.stack([p1, p2], axis=0)[None, ...]
    _mean, std, _both = reduce_proofread(jnp.asarray(stacked), ddof=1, std_over="orders")
    # The two formulas are the same real number. Float64 evaluates them a ulp apart.
    gap = np.max(np.abs(np.asarray(std) - closed_form_std(p1, p2)))
    assert gap <= np.finfo(np.float64).eps
  finally:
    jax.config.update("jax_enable_x64", previous)


def test_knob_semantics_proofread_uniforms_change_focus_probs(monkeypatch: pytest.MonkeyPatch) -> None:
  """A sampler handed a length-1 zero array cannot vary with its input.

  The conditional decode was called with ``np.zeros((1,))``. Inverse-CDF at
  u = 0 picks the first class at every sequence and chi site, so the focus
  probabilities did not depend on the caller's uniforms. This fails on that
  placeholder: the decode must receive the full ``length * 5 + 4`` buffer, and
  two buffers that share only the first draw must not yield the same focus row.
  """
  n_res = 3
  need = n_res * 5 + 4
  encoder, decoder, joint = _models()
  backbone, ligand, atomic, sub = _coords(n_res)
  period = np.zeros((118,), dtype=np.int32)
  # Arginine, so every chi is rotatable. Alanine would store NaN angles and
  # the focus residue would see zeros no matter which uniform it was handed.
  sequence = np.ones((n_res,), dtype=np.int32)
  chi = np.zeros((n_res, 4), dtype=np.float32)
  left = np.full((1, need), 1e-6, dtype=np.float64)
  right = left.copy()
  # The first draw is residue 0's sequence sample, which a fixed residue
  # discards. The remaining per-residue draws are the chi context the focus
  # residue reads. A length-1 consumer never sees them. The two tails sit
  # near 0 and near 1 so inverse-CDF cannot land in the same bin.
  right[0, 1 : n_res * 5] = 1.0 - 1e-6
  seen: list[np.ndarray] = []
  original = proofread.decode_order

  def _spy(*args: Any, **kwargs: Any) -> Any:
    seen.append(np.asarray(args[-1], dtype=np.float64).reshape(-1))
    return original(*args, **kwargs)

  monkeypatch.setattr(proofread, "decode_order", _spy)
  orders = [np.arange(n_res, dtype=np.int32)]
  structure = _structure()
  probs_left = conditional_focus_probs(
    encoder, decoder, joint,
    backbone, ligand, atomic, sub,
    period, period, structure,
    sequence, chi,
    n_res - 1,
    orders,
    [[None]],
    [[left]],
    scalar=False,
    repack_all=True,
  )
  probs_right = conditional_focus_probs(
    encoder, decoder, joint,
    backbone, ligand, atomic, sub,
    period, period, structure,
    sequence, chi,
    n_res - 1,
    orders,
    [[None]],
    [[right]],
    scalar=False,
    repack_all=True,
  )
  assert [int(row.shape[0]) for row in seen] == [need, need]
  np.testing.assert_array_equal(seen[0], left.reshape(-1))
  np.testing.assert_array_equal(seen[1], right.reshape(-1))
  assert not np.allclose(probs_left[0, 0], probs_right[0, 0])


def _focus_probs(
  uniforms: np.ndarray,
  order: np.ndarray,
  focus: int,
) -> np.ndarray:
  n_res = int(order.shape[0])
  encoder, decoder, joint = _models()
  backbone, ligand, atomic, sub = _coords(n_res)
  period = np.zeros((118,), dtype=np.int32)
  sequence = np.ones((n_res,), dtype=np.int32)
  chi = np.zeros((n_res, 4), dtype=np.float32)
  return conditional_focus_probs(
    encoder, decoder, joint,
    backbone, ligand, atomic, sub,
    period, period, _structure(),
    sequence, chi,
    focus,
    [order],
    [[None]],
    [[uniforms]],
    scalar=False,
    repack_all=True,
  )


def test_knob_semantics_proofread_draw_follows_upstream_call_order() -> None:
  """Upstream ``sample`` reads the flat buffer in call order, five draws per step.

  ``utils/model.py`` iterates columns of ``decoding_order`` and calls
  ``Categorical.sample`` once for the sequence and once per χ. The shim's
  cursor advances on those calls, so draw ``5 * step`` is the sequence sample
  of ``order[step]``, not of residue ``step``.

  The observable that would have caught the original defect is the reversed
  order's focus probabilities in ``laser_proofread_parity``: a residue-index
  read gives the last residue the block at ``5 * residue`` while upstream
  gives it the first five draws, and the proofread mean is taken over the two
  orders, so that one swapped order leaves a mean residual with a partly
  recovered std.
  """
  n_res = 3
  need = n_res * 5 + 4
  baseline = np.full((1, need), 1e-6, dtype=np.float64)
  # Order 0 is identity, so residue index and decode step name the same block.
  # The focus is last, and the block that builds its context is the first step.
  forward = np.arange(n_res, dtype=np.int32)
  forward_context = baseline.copy()
  forward_context[0, 0:5] = 1.0 - 1e-6
  forward_base = _focus_probs(baseline, forward, n_res - 1)
  forward_hit = _focus_probs(forward_context, forward, n_res - 1)
  assert not np.allclose(forward_base[0, 0], forward_hit[0, 0])
  # Order 1 decodes residue 2 first and the focus (residue 0) last.
  # Upstream's first five draws are residue 2's samples and change the focus.
  # Residue 2's index block sits at draws[10:15], which is the focus's own
  # step under call order and cannot move the focus logits.
  reverse = forward[::-1]
  call_order = baseline.copy()
  call_order[0, 0:5] = 1.0 - 1e-6
  residue_index = baseline.copy()
  residue_index[0, 2 * 5 : 2 * 5 + 5] = 1.0 - 1e-6
  reverse_base = _focus_probs(baseline, reverse, 0)
  reverse_call = _focus_probs(call_order, reverse, 0)
  reverse_residue = _focus_probs(residue_index, reverse, 0)
  assert not np.allclose(reverse_base[0, 0], reverse_call[0, 0])
  np.testing.assert_allclose(reverse_base[0, 0], reverse_residue[0, 0])


def test_knob_semantics_proofread_std_single_order() -> None:
  cell = np.arange(21, dtype=np.float64)
  stacked = cell.reshape(1, 1, 21)
  mean, std, _both = reduce_proofread(jnp.asarray(stacked), ddof=1)
  np.testing.assert_allclose(np.asarray(mean), cell)
  assert np.isnan(np.asarray(std)).all()


def test_knob_semantics_n_decoding_orders() -> None:
  """One order is the NaN std. Two orders are the closed form. ddof=0 is half of that gap."""
  previous = bool(getattr(jax.config, "jax_enable_x64", False))
  jax.config.update("jax_enable_x64", True)
  p1 = np.linspace(0.0, 1.0, 21)
  p2 = np.linspace(1.0, 0.0, 21)
  one = np.stack([p1], axis=0)[None, ...]
  two = np.stack([p1, p2], axis=0)[None, ...]
  try:
    _mean, std_one, _both = reduce_proofread(jnp.asarray(one), ddof=1)
    _mean, std_two, _both = reduce_proofread(jnp.asarray(two), ddof=1)
    _mean, std_pop, _both = reduce_proofread(jnp.asarray(two), ddof=0)
  finally:
    jax.config.update("jax_enable_x64", previous)
  assert np.isnan(np.asarray(std_one)).all()
  assert np.all(np.isfinite(np.asarray(std_two)))
  np.testing.assert_allclose(np.asarray(std_pop) * np.sqrt(2.0), np.asarray(std_two), rtol=0, atol=1e-12)
  assert int(LaserOptions(n_decoding_orders=1).n_decoding_orders) == 1
  assert int(LaserOptions().n_decoding_orders) == 10


def test_knob_semantics_n_dropouts() -> None:
  """The mean averages dropout reps after the per-rep order mean."""
  low = np.zeros((21,))
  high = np.ones((21,))
  one = np.stack([low, low], axis=0)[None, ...]
  two = np.stack(
    [np.stack([low, low], axis=0), np.stack([high, high], axis=0)],
    axis=0,
  )
  mean_one, _std, _both = reduce_proofread(jnp.asarray(one))
  mean_two, _std, _both = reduce_proofread(jnp.asarray(two))
  assert not np.allclose(np.asarray(mean_one), np.asarray(mean_two))
  swapped, _std, _both = reduce_proofread(jnp.asarray(two), std_over="reps")
  _mean, orders, _both = reduce_proofread(jnp.asarray(two), std_over="orders")
  assert not np.allclose(np.asarray(swapped), np.asarray(orders), equal_nan=True)


def test_knob_semantics_proofread_dropout() -> None:
  encoder, decoder, joint = _models()
  value = jnp.ones((2, 3))
  zeros = {"probe": [np.zeros((2, 3), dtype=np.float32)]}
  with proofread_dropout(
    encoder, decoder, tuple(joint.chi_offset_prediction_layers), scalar=True, masks=zeros,
  ) as plan:
    dropped = plan.scale("probe", value)
  with proofread_dropout(
    encoder, decoder, tuple(joint.chi_offset_prediction_layers), scalar=False, masks=zeros,
  ) as plan:
    kept = plan.scale("probe", value)
  assert float(dropped[0, 0]) == 0.0
  assert float(kept[0, 0]) == 1.0
  with proofread_dropout(
    encoder, decoder, tuple(joint.chi_offset_prediction_layers), scalar=True, vector=False,
  ) as plan:
    assert float(plan.take("layer.vdropout", value)[0, 0]) == 1.0
  with proofread_dropout(
    encoder, decoder, tuple(joint.chi_offset_prediction_layers), scalar=True, vector=True,
  ) as plan:
    assert float(plan.take("layer.vdropout", value)[0, 0]) == 0.0
  assert LaserOptions().proofread_dropout is True
  assert LaserOptions(proofread_dropout=False).proofread_dropout is False


def test_knob_semantics_selection_string() -> None:
  shell = np.asarray([True, False, True])
  resnums = np.asarray([10, 11, 12])
  assert list(focus_rows(shell, resnums, None)) == [0, 2]
  assert list(focus_rows(shell, resnums, "11")) == [1]
  with pytest.raises(ValueError, match="selection_string"):
    focus_rows(shell, resnums, "99")


def test_knob_semantics_strict_load() -> None:
  module = eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))

  class _Blob:
    def __init__(self, weight: jax.Array) -> None:
      self.weight = weight

  holder = _Blob(module.weight)
  _assert_strict_keys(holder, {"weight": module.weight}, ("weight",), np.dtype(np.float32))
  with pytest.raises(ValueError, match="strict_load"):
    _assert_strict_keys(holder, {"missing": module.weight}, ("missing",), np.dtype(np.float32))
  # f64 D_mu is rebuilt by the encoder, so a strict load must still accept it.
  _assert_strict_keys(holder, {"D_mu": module.weight}, ("D_mu",), np.dtype(np.float64))


def test_knob_semantics_output_fasta() -> None:
  driver = LaserDriver()
  off = driver.result_schema(SimpleNamespace(laser=LaserOptions()), "score:nll")
  on = driver.result_schema(SimpleNamespace(laser=LaserOptions(output_fasta=True)), "score:nll")
  assert "nll" in off
  assert "fasta" not in off
  assert "fasta" in on and "nll" in on


def test_knob_semantics_output_fasta_only() -> None:
  driver = LaserDriver()
  schema = driver.result_schema(
    SimpleNamespace(laser=LaserOptions(output_fasta_only=True)),
    "score:nll",
  )
  assert set(schema) == {"fasta"}


def test_knob_semantics_use_water() -> None:
  dry = featurize(_FIXTURE, use_water=False)
  wet = featurize(_FIXTURE, use_water=True)
  assert wet.ligand_atomic_numbers.shape[0] > dry.ligand_atomic_numbers.shape[0]


def test_knob_semantics_fix_from_bfactor() -> None:
  featurize(_FIXTURE, fix_from_bfactor=False)
  with pytest.raises(LaserInputError, match="fix_from_bfactor"):
    featurize(_FIXTURE, fix_from_bfactor=True)


def test_knob_semantics_noncanonical_aa_ligand(tmp_path: Path) -> None:
  path = tmp_path / "unk.pdb"
  lines = []
  for index, (name, element, x) in enumerate(
    (("N", "N", 0.0), ("CA", "C", 1.5), ("C", "C", 2.5), ("O", "O", 2.2)),
    start=1,
  ):
    y = 1.0 if name == "C" else 0.0
    lines.append(
      f"ATOM  {index:5d} {name:>4} UNK A   1    {x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2}",
    )
  lines.append(
    "ATOM  {serial:5d} {name:>4} ALA A   2    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}".format(
      serial=5, name="N", x=3.8, y=1.2, z=0.0, element="N",
    ),
  )
  for serial, name, element, x, y in (
    (6, "CA", "C", 5.0, 2.0),
    (7, "C", "C", 6.2, 1.2),
    (8, "O", "O", 6.4, 0.0),
    (9, "CB", "C", 5.2, 3.2),
  ):
    lines.append(
      f"ATOM  {serial:5d} {name:>4} ALA A   2    {x:8.3f}{y:8.3f}{0.2:8.3f}  1.00  0.00          {element:>2}",
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  plain = featurize(path, noncanonical_aa_ligand=False)
  ligand = featurize(path, noncanonical_aa_ligand=True)
  assert plain.sequence_indices.shape[0] > ligand.sequence_indices.shape[0]


def test_knob_semantics_chi_min_p() -> None:
  logits = jnp.asarray([0.0, 5.0, 1.0])
  warped = minp_warp_logits(logits, 0.4)
  assert not jnp.allclose(warped, logits)


def test_knob_semantics_budget_residue_selection() -> None:
  resnums = np.asarray([3, 4, 5])
  exposed = np.asarray([True, True, False])
  empty = budget_mask_for(resnums, ("-", "-", "H"), exposed, selection=None, constrain_to_exposed_non_ss=False)
  chosen = budget_mask_for(resnums, ("-", "-", "H"), exposed, selection="3,5", constrain_to_exposed_non_ss=False)
  assert not empty.any()
  assert list(chosen) == [True, False, True]


def test_knob_semantics_constrain_ala_gly_to_exposed_non_ss() -> None:
  resnums = np.asarray([3, 4, 5])
  exposed = np.asarray([True, True, False])
  loose = budget_mask_for(
    resnums, ("-", "H", "-"), exposed, selection="3,4,5", constrain_to_exposed_non_ss=False,
  )
  tight = budget_mask_for(
    resnums, ("-", "H", "-"), exposed, selection="3,4,5", constrain_to_exposed_non_ss=True,
  )
  assert list(loose) == [True, True, True]
  assert list(tight) == [True, False, False]


def _tied(  # noqa: PLR0913
  *,
  chain_mask: np.ndarray | None = None,
  chi1: np.ndarray | None = None,
  chi2: np.ndarray | None = None,
  sequence: np.ndarray | None = None,
  seq_u: np.ndarray | None = None,
  chi_u: np.ndarray | None = None,
  lambda_: float = 0.0,
  chi_temperature: float | None = None,
  sequence_temperature: float | None = 1.0,
  shift: float = 0.0,
  disabled: tuple[str, ...] = ("X",),
  repack_all: bool = False,
  order: np.ndarray | None = None,
  bias: np.ndarray | None = None,
) -> TiedDecodeResult:
  encoder, decoder, joint = _models()
  n_res = 2
  backbone, ligand, atomic, sub = _coords(n_res, shift)
  other, ligand2, atomic2, sub2 = _coords(n_res, shift + 5.0)
  if chain_mask is None:
    chain_mask = np.zeros((n_res,), dtype=bool)
  if sequence is None:
    sequence = np.full((n_res,), _VAL, dtype=np.int32)
  if chi1 is None:
    chi1 = np.zeros((n_res, 4), dtype=np.float32)
  if chi2 is None:
    chi2 = np.zeros((n_res, 4), dtype=np.float32)
  if seq_u is None:
    seq_u = np.full((n_res,), 0.5)
  if chi_u is None:
    chi_u = np.full((n_res, 4, 2), 0.5)
  period = np.zeros((118,), dtype=np.int32)
  group = np.zeros((118,), dtype=np.int32)
  with jax.disable_jit():
    return tied_decode(
      encoder,
      decoder,
      joint,
      backbone,
      other,
      ligand,
      ligand2,
      atomic,
      atomic2,
      sub,
      sub2,
      period,
      group,
      _structure(),
      sequence,
      chi1,
      chi2,
      chain_mask,
      np.arange(n_res, dtype=np.int32) if order is None else order,
      seq_u,
      chi_u,
      sequence_temperature=sequence_temperature,
      chi_temperature=chi_temperature,
      lambda_=lambda_,
      disabled_residues=disabled,
      repack_all=repack_all,
      bias=bias,
    )


def test_knob_semantics_tied_lambda() -> None:
  """λ = 1 is structure 1's softmax. λ = 0 is structure 2's. Logit mixing differs."""
  logits_1 = jnp.asarray([0.0, 5.0, 0.0])
  logits_2 = jnp.asarray([5.0, 0.0, 0.0])
  only_1 = mix_sequence_probabilities(logits_1, logits_2, 1.0, 1.0, on_logits=False)
  only_2 = mix_sequence_probabilities(logits_1, logits_2, 0.0, 1.0, on_logits=False)
  assert int(categorical_draw(only_1, jnp.asarray(0.5))) != int(categorical_draw(only_2, jnp.asarray(0.5)))
  mixed = mix_sequence_probabilities(logits_1, logits_2, 0.5, 1.0, on_logits=False)
  on_logits = mix_sequence_probabilities(logits_1, logits_2, 0.5, 1.0, on_logits=True)
  assert not jnp.allclose(mixed, on_logits)

def test_knob_semantics_tied_interpolation_lambda() -> None:
  with pytest.raises(ValueError, match="tied_interpolation_lambda"):
    _tied(lambda_=1.5)


def test_knob_semantics_tied_second_input() -> None:
  with pytest.raises(ValueError, match="seq_min_p"):
    encoder, decoder, joint = _models()
    backbone, ligand, atomic, sub = _coords(2)
    period = np.zeros((118,), dtype=np.int32)
    with jax.disable_jit():
      tied_decode(
        encoder, decoder, joint,
        backbone, backbone, ligand, ligand, atomic, atomic, sub, sub,
        period, period, _structure(),
        np.zeros((2,), dtype=np.int32),
        np.zeros((2, 4)),
        np.zeros((2, 4)),
        np.zeros((2,), dtype=bool),
        np.arange(2, dtype=np.int32),
        np.zeros((2,)),
        np.zeros((2, 4, 2)),
        seq_min_p=0.05,
      )


def test_knob_semantics_tied_chi_nan_fixed() -> None:
  chi = np.full((2, 4), np.nan, dtype=np.float32)
  chain = np.asarray([True, False])
  tied = _tied(chain_mask=chain, chi1=chi, chi2=chi, chi_temperature=None, sequence_temperature=None)
  # A NaN fixed χ becomes 0 degrees on the tied path.
  assert tied.chi_degrees_1[0, 0] == pytest.approx(0.0)
  encoder, decoder, joint = _models()
  backbone, ligand, atomic, sub = _coords(2)
  period = np.zeros((118,), dtype=np.int32)
  with jax.disable_jit():
    plain = decode_order(
      encoder, decoder, joint,
      backbone, ligand, atomic, sub, period, period, _structure(),
      np.full((2,), _VAL, dtype=np.int32),
      chi, chain, np.zeros((2,), dtype=bool),
      np.arange(2, dtype=np.int32),
      np.zeros((1,)),
      sequence_temperature=None,
      chi_temperature=None,
      disabled_residues=("X",),
    )
  # Non-tied sampling keeps input χ only where it is fixed and finite, so a NaN
  # fixed angle is replaced by the draw. Tied kept the nan_to_num 0 instead.
  assert plain.chi_degrees[0, 0] != pytest.approx(0.0)


def test_knob_semantics_tied_uniform_order() -> None:
  """Upstream ``tied_sample`` consumes χ in decoding-column order, not residue order.

  ``utils/model.py`` samples, per column of ``decoding_order``, the shared
  sequence and then χ index × structure (structure 1 before structure 2).
  The shim stores that as stream step = column. A ``(length, 4, 2)`` buffer
  is one row per column. Reading ``buffer[0]`` as a sample axis drops every
  row after the first.

  The observable that would have caught the original defect is the tied arm of
  ``laser_decode_e2e``: χ-bin agreement fell to 0.042 because later steps drew
  the zero-filled tail, and the wrong χ context then moved the sequence
  (agreement 0.374). Pinning ``tied_chi_index`` against itself cannot see that.
  """
  letters = "ARNDCEQGHILKMFPSTWYVX"
  disabled = tuple(letter for letter in letters if letter != "V")
  # Column 0 is residue 1. Its structure-1 χ0 draw is 0, so that bin is 0.
  # Residue 0 is column 1 and must see the 0.99 draw, not a zero-filled tail
  # and not column 0's draw.
  chi_u = np.full((2, 4, 2), 0.99)
  chi_u[0, 0, 0] = 0.0
  decoded = _tied(
    chi_u=chi_u,
    chi_temperature=1.0,
    sequence_temperature=1.0,
    disabled=disabled,
    seq_u=np.zeros((2,)),
    order=np.asarray([1, 0], dtype=np.int32),
  )
  assert int(decoded.sequence[1]) == _VAL
  assert int(decoded.chi_bins_1[1, 0]) == 0
  assert int(decoded.chi_bins_1[0, 0]) != 0
  assert int(decoded.chi_bins_2[1, 0]) != 0


def _sample_at(lineno: int) -> Any:
  """A ``Categorical.sample`` whose frame is ``model.py`` at ``lineno``.

  The uniform shim keys off that filename and the tied draw lines. A call from
  this test file would be ignored, which is the same miss the e2e recorder hits.
  """
  lines = ["import torch", "def site():"]
  while len(lines) < lineno - 1:
    lines.append("    pass")
  lines.append("    return torch.distributions.Categorical(probs=torch.ones(8) / 8).sample()")
  namespace: dict[str, Any] = {}
  exec(compile("\n".join(lines) + "\n", "model.py", "exec"), namespace)  # noqa: S102
  return namespace["site"]


def test_knob_semantics_tied_draw_order_matches_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
  """Upstream ``tied_sample`` draws sequence, then chi index by structure.

  ``utils/model.py`` calls ``Categorical.sample`` at :627 once per column, then
  at :676 and :677 inside the chi loop (structure 1, then structure 2). That is
  nine draws, not the non-tied five. The e2e recorder wraps ``sample`` before
  ``tied_sample`` runs. A one-frame lookback sees the wrapper, not :627, so
  upstream keeps torch's RNG.

  The observable that would have caught this defect is the tied arm of
  ``laser_decode_e2e``: probability rows match only for residues with no
  previously decoded neighbour. Under seed 42 that set has three members and
  the second decoded residue is index 1, which agrees to ~1e-16, while the
  token at index 1 mismatches.
  """
  torch = pytest.importorskip("torch")
  from scripts.parity.oracle_shims.laser import injected_uniform_draws

  n_res = 2
  seq_u = np.asarray([0.17, 0.83])
  chi_u = np.arange(n_res * 4 * 2, dtype=np.float64).reshape(n_res, 4, 2)
  chi_u = 0.05 + 0.9 * chi_u / float(chi_u.max())
  expected: list[float] = []
  for step in range(n_res):
    expected.append(float(seq_u[step]))
    for chi in range(4):
      expected.append(float(chi_u[step, chi, 0]))
      expected.append(float(chi_u[step, chi, 1]))

  consumed: list[float] = []
  real_draw = tied_mod.categorical_draw

  def _record(probs: jax.Array, uniform: jax.Array) -> jax.Array:
    consumed.append(float(uniform))
    return real_draw(probs, uniform)

  monkeypatch.setattr(tied_mod, "categorical_draw", _record)
  _tied(
    seq_u=seq_u,
    chi_u=chi_u,
    chi_temperature=1.0,
    sequence_temperature=1.0,
    order=np.asarray([1, 0], dtype=np.int32),
  )
  assert consumed == pytest.approx(expected)

  # The recorder is what laser_decode_e2e installs around the shim. Without a
  # stack walk, none of these calls consume the stream.
  stream = np.zeros((1, n_res, 9), dtype=np.float64)
  stream[0, :, 0] = seq_u
  for chi in range(4):
    stream[0, :, 1 + 2 * chi] = chi_u[:, chi, 0]
    stream[0, :, 2 + 2 * chi] = chi_u[:, chi, 1]
  sequence_site = _sample_at(627)
  chi_site_1 = _sample_at(676)
  chi_site_2 = _sample_at(677)
  drawn: list[int] = []
  with injected_uniform_draws(stream):
    shimmed = torch.distributions.Categorical.sample

    def _collect(self: Any, sample_shape: Any = None) -> Any:
      out = shimmed(self, torch.Size() if sample_shape is None else sample_shape)
      drawn.append(int(out.reshape(-1)[0]))
      return out

    torch.distributions.Categorical.sample = _collect  # type: ignore[method-assign]
    for _step in range(n_res):
      sequence_site()
      for _chi in range(4):
        chi_site_1()
        chi_site_2()
  # Equal mass over 8 bins: inverse-CDF of u lands in bin floor(u * 8), and a
  # missed injection draws from torch instead, which does not follow `expected`.
  bins = [min(int(np.floor(value * 8.0)), 7) for value in expected]
  assert drawn == bins


def test_knob_semantics_chi_temp() -> None:
  letters = "ARNDCEQGHILKMFPSTWYVX"
  disabled = tuple(letter for letter in letters if letter != "V")
  chi_u = np.full((2, 4, 2), 0.99)
  sampled = _tied(chi_u=chi_u, chi_temperature=1.0, disabled=disabled)
  peaked = _tied(chi_u=chi_u, chi_temperature=None, disabled=disabled)
  assert not np.array_equal(sampled.chi_bins_1, peaked.chi_bins_1)

def test_knob_semantics_gly_budget() -> None:
  """A zero GLY budget floors G on a budgeted row. The default mask is empty."""
  resnums = np.asarray([0, 1])
  exposed = np.asarray([True, True])
  masked = budget_mask_for(resnums, ("-", "-"), exposed, selection="0", constrain_to_exposed_non_ss=False)
  assert list(masked) == [True, False]
  assert LaserOptions().gly_budget == 0
  assert LaserOptions(gly_budget=2).gly_budget == 2


def test_knob_semantics_repack_only() -> None:
  chi = np.full((2, 4), np.nan, dtype=np.float32)
  chain = np.asarray([True, True])
  kept = _tied(chain_mask=chain, chi1=chi, chi2=chi, chi_temperature=None, repack_all=False)
  replaced = _tied(chain_mask=chain, chi1=chi, chi2=chi, chi_temperature=None, repack_all=True)
  assert kept.chi_degrees_1[0, 0] == pytest.approx(0.0)
  assert np.isfinite(replaced.chi_degrees_1[0, 0])
  assert LaserOptions(repack_only=True).repack_only is True


def _assert_bit_identical(left: TiedDecodeResult, right: TiedDecodeResult) -> None:
  names = (
    "sequence",
    "sequence_logits_1",
    "sequence_logits_2",
    "chi_degrees_1",
    "chi_degrees_2",
    "chi_logits_1",
    "chi_logits_2",
    "chi_bins_1",
    "chi_bins_2",
  )
  for name in names:
    got = np.asarray(getattr(left, name))
    exp = np.asarray(getattr(right, name))
    assert got.shape == exp.shape and got.dtype == exp.dtype
    np.testing.assert_array_equal(got.view(np.uint8), exp.view(np.uint8), err_msg=name)


def test_tied_decode_zero_bias_matches_omitted_bias() -> None:
  """(a) An omitted bias and an explicit zero row are the same decode."""
  omitted = _tied()
  zeros = _tied(bias=np.zeros((2, 21), dtype=np.float32))
  _assert_bit_identical(omitted, zeros)


def test_tied_decode_positive_bias_forces_token() -> None:
  """(b) A large positive bias on one designable site wins for every uniform."""
  token = 3
  position = 0
  bias = np.zeros((2, 21), dtype=np.float32)
  bias[position, token] = np.float32(1.0e6)
  for uniform in (0.0, 1.0e-12, 0.25, 0.5, 0.75, 1.0 - 1.0e-12):
    decoded = _tied(bias=bias, seq_u=np.asarray([uniform, 0.5], dtype=np.float64))
    assert int(decoded.sequence[position]) == token


def test_tied_decode_omit_bias_is_never_drawn() -> None:
  """(c) An omitted letter (-1e8) is never drawn at that position."""
  omitted = 4
  position = 0
  bias = np.zeros((2, 21), dtype=np.float32)
  bias[position, omitted] = np.float32(-1.0e8)
  for uniform in np.linspace(0.0, 1.0, num=11, endpoint=False):
    decoded = _tied(bias=bias, seq_u=np.asarray([uniform, 0.5], dtype=np.float64))
    assert int(decoded.sequence[position]) != omitted


def test_tied_decode_step_probabilities_match_biased_mix(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(d) Step 0 mixes ``λ softmax((l1+b)/T) + (1-λ) softmax((l2+b)/T)``."""
  lam = 0.35
  temperature = 1.0
  bias = np.zeros((2, 21), dtype=np.float32)
  bias[0, 1] = np.float32(1.5)
  bias[0, 5] = np.float32(-0.75)
  bias[0, 11] = np.float32(0.4)
  unbiased = _tied(lambda_=lam, sequence_temperature=temperature, disabled=())
  seen: list[np.ndarray] = []
  original = tied_mod.categorical_draw

  def _spy(probs: jax.Array, uniform: jax.Array) -> jax.Array:
    seen.append(np.asarray(probs, dtype=np.float64))
    return original(probs, uniform)

  monkeypatch.setattr(tied_mod, "categorical_draw", _spy)
  _tied(lambda_=lam, sequence_temperature=temperature, disabled=(), bias=bias)
  assert len(seen) == 2
  l1 = np.asarray(unbiased.sequence_logits_1[0], dtype=np.float32) + bias[0]
  l2 = np.asarray(unbiased.sequence_logits_2[0], dtype=np.float32) + bias[0]

  def _softmax(logits: np.ndarray) -> np.ndarray:
    scaled = jnp.asarray(logits, dtype=jnp.float32) / jnp.float32(temperature)
    return np.asarray(jax.nn.softmax(scaled), dtype=np.float64)

  expected = lam * _softmax(l1) + (1.0 - lam) * _softmax(l2)
  np.testing.assert_allclose(seen[0], expected, rtol=1e-5, atol=1e-5)
