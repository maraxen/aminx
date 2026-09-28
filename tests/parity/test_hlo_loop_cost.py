"""Tests for `scripts/browser_validation/hlo_loop_cost.py` (T7).

The `simple_hlo_text` fixture is a handwritten HLO snippet (real compiled-HLO syntax,
confirmed against the installed jax/jaxlib before writing this test -- see
`hlo_loop_cost.py`'s module docstring) containing a fusion, a dynamic-update-slice, an
unfused reduce-to-scalar, and a conditional with two branches. Its
`body_operand_elements` is hand-computed below; the three red-check tests each assert a
component of that hand computation directly, so each of the following mutations makes at
least one test fail:

1. counting the DUS full output instead of the update operand
   (`test_dus_counts_update_operand_not_full_output`);
2. descending into the fusion body instead of counting it once at the call site
   (`test_fusion_not_double_counted`);
3. counting the reduce by output elements only instead of operand + output
   (`test_reduce_counts_operand_plus_output`).
"""

from __future__ import annotations

import pytest

from scripts.browser_validation.hlo_loop_cost import (
  HLOLoopAnalyzer,
  _count_elements,
  _parse_shape,
  compute_scaling_factor,
)


class TestParseShape:
  """Shape parsing tests."""

  def test_parse_multidim_shape(self) -> None:
    dims = _parse_shape("f32[16,256]")
    assert dims == (16, 256)

  def test_parse_scalar_shape(self) -> None:
    dims = _parse_shape("f32[]")
    assert dims == ()

  def test_parse_single_dim_shape(self) -> None:
    dims = _parse_shape("f32[1024]")
    assert dims == (1024,)


class TestCountElements:
  """Element counting tests."""

  def test_count_multidim(self) -> None:
    assert _count_elements("f32[16,256]") == 4096

  def test_count_scalar(self) -> None:
    assert _count_elements("f32[]") == 1

  def test_count_single_dim(self) -> None:
    assert _count_elements("f32[1024]") == 1024


class TestComputeScalingFactor:
  """Scaling factor computation tests."""

  def test_linear_scaling(self) -> None:
    # Doubling cost when doubling size => alpha = 1
    alpha = compute_scaling_factor(m_2l=64, m_l=32)
    assert alpha == pytest.approx(1.0)

  def test_constant_scaling(self) -> None:
    # Same cost regardless of size => alpha = 0
    alpha = compute_scaling_factor(m_2l=100, m_l=100)
    assert alpha == pytest.approx(0.0)

  def test_zero_denominator_returns_none(self) -> None:
    assert compute_scaling_factor(m_2l=50, m_l=0) is None

  def test_negative_denominator_returns_none(self) -> None:
    assert compute_scaling_factor(m_2l=50, m_l=-5) is None

  def test_zero_numerator_returns_none(self) -> None:
    assert compute_scaling_factor(m_2l=0, m_l=50) is None


# HLO snippet in the exact syntax confirmed against the installed jax/jaxlib (`while(...)
# condition=%.., body=%..`; `conditional(...) branch_computations={%A, %B}`;
# `fusion(...), kind=kLoop, calls=%..`; `reduce(...), to_apply=%..`). Carried tuple:
# (buf: f32[256,16], upd: f32[16], idx: s32[], acc: f32[]).
#
# Hand-computed body_operand_elements (see class docstring below for the walk):
#   fused_mul_add fusion(%buf):        operand 256*16=4096 + output 4096       = 8192
#   conditional(branch_true, branch_false), max over branches:
#     branch_true:  dynamic-update-slice -> update operand only (%upd, 16 elems) = 16
#     branch_false: get-tuple-element (excluded)                                = 0
#     max(16, 0)                                                                = 16
#   reduce(%upd, %zero), UNFUSED:       operand (16 + 1) + output 1             = 18
#   add(%acc, %reduced):                output-only (scalar)                   = 1
#   copy(%idx):                         output-only (scalar), counted          = 1
#   -------------------------------------------------------------------------------
#   TOTAL                                                                      = 8228
_SIMPLE_HLO_TEXT = """
%fused_mul_add (p0: f32[256,16]) -> f32[256,16] {
  %p0 = f32[256,16] parameter(0)
  %c2 = f32[] constant(2)
  %bc2 = f32[256,16] broadcast(%c2), dimensions={}
  %mul = f32[256,16] multiply(%p0, %bc2)
  %c1 = f32[] constant(1)
  %bc1 = f32[256,16] broadcast(%c1), dimensions={}
  ROOT %add_ = f32[256,16] add(%mul, %bc1)
}

%add_computation (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%bt (arg: (f32[256,16], f32[16], s32[])) -> f32[256,16] {
  %arg = (f32[256,16], f32[16], s32[]) parameter(0)
  %buf = f32[256,16] get-tuple-element(%arg), index=0
  %upd = f32[16] get-tuple-element(%arg), index=1
  %idx = s32[] get-tuple-element(%arg), index=2
  %zero = s32[] constant(0)
  ROOT %dus = f32[256,16] dynamic-update-slice(%buf, %upd, %idx, %zero)
}

%bf (arg: (f32[256,16], f32[16], s32[])) -> f32[256,16] {
  %arg = (f32[256,16], f32[16], s32[]) parameter(0)
  ROOT %buf = f32[256,16] get-tuple-element(%arg), index=0
}

%body (carry: (f32[256,16], f32[16], s32[], f32[])) -> (f32[256,16], f32[16], s32[], f32[]) {
  %carry = (f32[256,16], f32[16], s32[], f32[]) parameter(0)
  %buf = f32[256,16] get-tuple-element(%carry), index=0
  %upd = f32[16] get-tuple-element(%carry), index=1
  %idx = s32[] get-tuple-element(%carry), index=2
  %acc = f32[] get-tuple-element(%carry), index=3
  %fused = f32[256,16] fusion(%buf), kind=kLoop, calls=%fused_mul_add
  %branch_arg = (f32[256,16], f32[16], s32[]) tuple(%fused, %upd, %idx)
  %pred = pred[] constant(true)
  %cond = f32[256,16] conditional(%pred, %branch_arg, %branch_arg), branch_computations={%bt, %bf}
  %zero = f32[] constant(0)
  %reduced = f32[] reduce(%upd, %zero), dimensions={0}, to_apply=%add_computation
  %acc2 = f32[] add(%acc, %reduced)
  %copy_idx = s32[] copy(%idx)
  ROOT %out = (f32[256,16], f32[16], s32[], f32[]) tuple(%cond, %upd, %copy_idx, %acc2)
}

%cond_comp (carry: (f32[256,16], f32[16], s32[], f32[])) -> pred[] {
  %carry = (f32[256,16], f32[16], s32[], f32[]) parameter(0)
  ROOT %true = pred[] constant(true)
}

ENTRY %main (p) -> (f32[256,16], f32[16], s32[], f32[]) {
  %buf0 = f32[256,16] parameter(0)
  %upd0 = f32[16] parameter(1)
  %idx0 = s32[] parameter(2)
  %acc0 = f32[] constant(0)
  %init = (f32[256,16], f32[16], s32[], f32[]) tuple(%buf0, %upd0, %idx0, %acc0)
  ROOT %while = (f32[256,16], f32[16], s32[], f32[]) while(%init), condition=%cond_comp, body=%body
}
"""

_EXPECTED_TOTAL = 8228


class TestHLOLoopAnalyzer:
  """HLO loop analyzer tests, against the hand-computed fixture above."""

  def test_parser_structure(self) -> None:
    """The parser finds exactly the one while loop, keyed by the while instruction's name."""
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    assert len(loops) == 1
    assert loops[0].name == "while"
    assert loops[0].has_nested_while is False

  def test_total_body_operand_elements_matches_hand_count(self) -> None:
    """The full walk (fusion + conditional-max + unfused-reduce + copy) sums to 8228."""
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    assert loops[0].body_operand_elements == _EXPECTED_TOTAL

  def test_fusion_not_double_counted(self) -> None:
    """RED-CHECK: fusion is counted once, at the call site, never descended into.

    If the walker descended into `%fused_mul_add` (`calls=`), it would separately emit
    entries for `%mul`/`%bc2`/`%bc1`/`%add_` (double-counting on top of the fusion's own
    8192), inflating the total past 8228 and introducing 'multiply'/'broadcast' op
    entries that must never appear in the top-10 (they live inside the fused
    computation, which this walker never visits).
    """
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    assert loops[0].body_operand_elements == _EXPECTED_TOTAL
    ops_seen = {entry["op"] for entry in loops[0].top_10_instructions}
    assert "multiply" not in ops_seen
    assert "broadcast" not in ops_seen
    fusion_entries = [e for e in loops[0].top_10_instructions if e["op"] == "fusion"]
    assert len(fusion_entries) == 1
    assert fusion_entries[0]["operand_elements"] == 4096
    assert fusion_entries[0]["output_elements"] == 4096
    assert fusion_entries[0]["total_weight"] == 8192

  def test_dus_counts_update_operand_not_full_output(self) -> None:
    """RED-CHECK: dynamic-update-slice costs its update operand (16), not the full
    output (256*16 = 4096). A full-output regression would flip the conditional's
    max(branch_true, branch_false) from 16 to 4096, and the total from 8228 to 12308.
    """
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    assert loops[0].body_operand_elements == _EXPECTED_TOTAL
    dus_entries = [e for e in loops[0].top_10_instructions if e["op"] == "dynamic-update-slice"]
    assert len(dus_entries) == 1
    assert dus_entries[0]["operand_elements"] == 16
    assert dus_entries[0]["output_elements"] == 0
    assert dus_entries[0]["total_weight"] == 16

  def test_reduce_counts_operand_plus_output(self) -> None:
    """RED-CHECK: unfused reduce(%upd[16], %zero[]) costs operand(16+1) + output(1) =
    18, not output-only (1). An output-only regression would blind alpha to L entirely
    for reduce-shaped bodies (the exact blindness #1983/this control exists to catch).
    """
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    assert loops[0].body_operand_elements == _EXPECTED_TOTAL
    reduce_entries = [e for e in loops[0].top_10_instructions if e["op"] == "reduce"]
    assert len(reduce_entries) == 1
    assert reduce_entries[0]["operand_elements"] == 17
    assert reduce_entries[0]["output_elements"] == 1
    assert reduce_entries[0]["total_weight"] == 18

  def test_conditional_takes_max_over_branches(self) -> None:
    """The conditional picks branch_true's DUS (16), not branch_false's 0."""
    analyzer = HLOLoopAnalyzer(_SIMPLE_HLO_TEXT)
    loops = analyzer.analyze()
    dus_entries = [e for e in loops[0].top_10_instructions if e["op"] == "dynamic-update-slice"]
    assert len(dus_entries) == 1  # branch_false's get-tuple-element never contributes


def test_synthetic_control_expectation_linear() -> None:
  """Verify that a linear body (x * 2 + 1) has alpha ~= 1."""
  # A fusion with operands (L*16 in) and output (L*16 out) has weight 32*L.
  # Doubling L doubles the weight, so alpha = log2(2) = 1.
  m_l = 32 * 128  # Weight at L=128
  m_2l = 32 * 256  # Weight at L=256
  alpha = compute_scaling_factor(m_2l, m_l)
  assert alpha == pytest.approx(1.0)


def test_synthetic_control_expectation_reduce() -> None:
  """Verify that reduce to scalar has alpha ~= 1."""
  # Reduce body: operand (L) + (1 scalar acc input) + output (1).
  # Weight ~= L + 1 + 1 = L + 2.
  # When L >> 2: log2((2L+2)/(L+2)) -> log2(2) = 1.
  m_l = 128 + 2
  m_2l = 256 + 2
  alpha = compute_scaling_factor(m_2l, m_l)
  assert alpha == pytest.approx(1.0, abs=0.05)


def test_synthetic_control_expectation_constant() -> None:
  """Verify that constant body (matmul independent of L) has alpha ~= 0."""
  # Matmul (48*16) @ (16*16) is constant regardless of L.
  # log2(const / const) = 0.
  m_l = 48 * 16 + 16 * 16
  m_2l = 48 * 16 + 16 * 16  # Same cost
  alpha = compute_scaling_factor(m_2l, m_l)
  assert alpha == pytest.approx(0.0)


def test_ar_kernel_measurement_returns_non_null_alpha_on_tiny_l() -> None:
  """The placeholder AR measurement (imported a nonexistent module, never built a model,
  returned `alpha: None` behind a blanket `except Exception`) can never return: on a
  tiny L pair the real, HLO-based `measure_ar_kernel` must yield a concrete float alpha
  for both AR modes (off, force), not None.
  """
  from scripts.browser_validation.layer_b_loop_cost import measure_ar_kernel

  result = measure_ar_kernel(l_vals=(16, 32), repeats=1)

  assert result["off"]["alpha"] is not None
  assert isinstance(result["off"]["alpha"], float)
  assert result["off"]["measurements"][16] is not None
  assert result["off"]["measurements"][32] is not None

  assert result["force"]["alpha"] is not None
  assert isinstance(result["force"]["alpha"], float)
  assert result["force"]["measurements"][16] is not None
  assert result["force"]["measurements"][32] is not None
