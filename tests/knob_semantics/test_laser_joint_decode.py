"""Section 5.3.1 knob semantics, non-tied subset.

Item 4's tied half is B7 and is not covered here. Upstream equality on the 21
fixtures is the ``laser_decode_e2e`` sidecar; these tests lock the carry rules
that sidecar depends on.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.laser_mpnn.featurize import featurize
from aminx.model.laser.decoder import LaserDecoder
from aminx.model.laser.encoders import LaserEncoder
from aminx.model.laser.graphs import GraphStructure
from aminx.model.laser.joint_decode import (
  JointStepInputs,
  LaserJointDecode,
  _Carry,
  _commit_residue,
  categorical_draw,
  JointDecodeResult,
  chi_position_mask,
  circular_abs_delta_deg,
  decode_order,
)

_ALA = 0
_LYS = 11
_VAL = 19
_X = 20


def _quadratic_shapes(closed: Any, n_res: int) -> list[str]:
  hits: list[str] = []

  def walk(jaxpr: Any) -> None:
    for eqn in jaxpr.eqns:
      for var in (*eqn.invars, *eqn.outvars):
        aval = getattr(var, "aval", None)
        shape = getattr(aval, "shape", None)
        if shape is not None and sum(dim == n_res for dim in shape) >= 2:
          hits.append(f"{eqn.primitive} {tuple(shape)}")
      for value in eqn.params.values():
        inner = getattr(value, "jaxpr", None)
        if inner is None:
          continue
        if hasattr(inner, "eqns"):
          walk(inner)
        else:
          nested = getattr(inner, "jaxpr", None)
          if nested is not None and hasattr(nested, "eqns"):
            walk(nested)

  walk(closed.jaxpr)
  return hits


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


def _coords(n_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  backbone = np.zeros((n_res, 5, 3), dtype=np.float32)
  for index in range(n_res):
    backbone[index, :, 0] = index * 3.8
    backbone[index, 1, 1] = 1.5
  ligand = np.asarray([[0.0, 8.0, 0.0]], dtype=np.float32)
  atomic = np.asarray([6], dtype=np.int32)
  subbatch = np.asarray([0], dtype=np.int32)
  return backbone, ligand, atomic, subbatch


def _decode(
  *,
  n_res: int = 2,
  order: np.ndarray | None = None,
  chain_mask: np.ndarray | None = None,
  sequence: np.ndarray | None = None,
  chi: np.ndarray | None = None,
  first_shell: np.ndarray | None = None,
  budget_mask: np.ndarray | None = None,
  ala_budget: int = 4,
  gly_budget: int = 0,
  ignore_chain_mask_zeros: bool = False,
  repack_all: bool = False,
  repack_only: bool = False,
) -> JointDecodeResult:
  encoder, decoder, joint = _models()
  backbone, ligand, atomic, subbatch = _coords(n_res)
  period = np.zeros((118,), dtype=np.int32)
  group = np.zeros((118,), dtype=np.int32)
  if order is None:
    order = np.arange(n_res, dtype=np.int32)
  if chain_mask is None:
    chain_mask = np.zeros((n_res,), dtype=bool)
  if sequence is None:
    sequence = np.zeros((n_res,), dtype=np.int32)
  if chi is None:
    chi = np.full((n_res, 4), np.nan, dtype=np.float32)
  if first_shell is None:
    first_shell = np.zeros((n_res,), dtype=bool)
  return decode_order(
    encoder,
    decoder,
    joint,
    backbone,
    ligand,
    atomic,
    subbatch,
    period,
    group,
    _structure(),
    sequence,
    chi,
    chain_mask,
    first_shell,
    order,
    np.asarray([0.5], dtype=np.float32),
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=repack_all,
    repack_only=repack_only,
    budget_mask=budget_mask,
    disabled_residues=("X",),
  )


def _step_inputs(
  *,
  n_res: int = 4,
  chain_mask: bool = False,
  first_shell: bool = False,
  charged: bool = False,
  bias_x: float = 0.0,
  bias_k: float = 0.0,
  sequence_index: int = 0,
  chi0: float | None = None,
) -> JointStepInputs:
  dtype = jnp.float32
  length = n_res
  disabled = jnp.zeros((21,), dtype=jnp.bool_).at[_X].set(True)
  bias = jnp.zeros((length, 21), dtype=dtype)
  bias = bias.at[:, _X].set(bias_x)
  bias = bias.at[:, _LYS].set(bias_k)
  chi = jnp.full((length, 4), jnp.nan, dtype=dtype)
  if chi0 is not None:
    chi = chi.at[0, 0].set(chi0)
  node_s = jnp.zeros((4, length, 256), dtype=dtype)
  node_s = node_s.at[0].set(jax.random.normal(jax.random.PRNGKey(3), (length, 256), dtype=dtype))
  node_v = jnp.zeros((4, length, 10, 3), dtype=dtype)
  return JointStepInputs(
    ligand_scalars=jnp.zeros((2, 256), dtype=dtype),
    pr_edges=jnp.zeros((length, 5, 128), dtype=dtype),
    pr_neighbours=jnp.zeros((length, 5), dtype=jnp.int32),
    pr_mask=jnp.ones((length, 5), dtype=jnp.bool_),
    lp_edges=jnp.zeros((length, 4, 128), dtype=dtype),
    lp_neighbours=jnp.zeros((length, 4), dtype=jnp.int32),
    lp_mask=jnp.ones((length, 4), dtype=jnp.bool_),
    decoding_order=jnp.arange(length, dtype=jnp.int32),
    sequence=jnp.full((length,), 21, dtype=jnp.int32),
    chi_degrees=jnp.full((length, 4), jnp.nan, dtype=dtype),
    input_sequence=jnp.full((length,), sequence_index, dtype=jnp.int32),
    input_chi=chi,
    chain_mask=jnp.full((length,), chain_mask, dtype=jnp.bool_),
    first_shell=jnp.full((length,), first_shell, dtype=jnp.bool_),
    budget_mask=jnp.zeros((length,), dtype=jnp.bool_),
    charged_mask=jnp.full((length,), charged, dtype=jnp.bool_),
    disabled=disabled,
    bias=bias,
    uniforms=jnp.full((8,), 0.5, dtype=dtype),
    step_index=jnp.int32(0),
    ala_count=jnp.int32(0),
    gly_count=jnp.int32(0),
    node_scalars=node_s,
    node_vectors=node_v,
  )


def _call(
  inputs: JointStepInputs,
  *,
  sequence_temperature: float | None = None,
  fs_sequence_temp: float | None = None,
  seq_min_p: float = 0.0,
  ignore_chain_mask_zeros: bool = False,
  repack_all: bool = False,
  uniforms: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  _encoder, decoder, joint = _models()
  if uniforms is not None:
    inputs = eqx.tree_at(lambda item: item.uniforms, inputs, uniforms)
  stored, chosen, chi_logits, chi_degrees, _consumed = joint(
    decoder,
    inputs,
    sequence_temperature=sequence_temperature,
    chi_temperature=None,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=0.0,
    ala_budget=4,
    gly_budget=0,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=repack_all,
  )
  return stored, chosen, chi_logits, chi_degrees


def test_knob_semantics_budget_counts_fixed() -> None:
  """A fixed ALA counts toward the budget even when it sits outside the region."""
  sequence = np.asarray([_ALA, 1], dtype=np.int32)
  chain = np.asarray([True, False])
  region = np.asarray([False, True])
  order = np.asarray([0, 1], dtype=np.int32)
  tight = _decode(
    sequence=sequence,
    chain_mask=chain,
    budget_mask=region,
    order=order,
    ala_budget=1,
    gly_budget=4,
  )
  loose = _decode(
    sequence=sequence,
    chain_mask=chain,
    budget_mask=region,
    order=order,
    ala_budget=2,
    gly_budget=4,
  )
  floor = np.finfo(np.float32).min
  assert int(tight.sequence[0]) == _ALA
  assert tight.sequence_logits[1, _ALA] == floor
  assert loose.sequence_logits[1, _ALA] != floor


def test_knob_semantics_disabled_floor_vs_charged_neginf() -> None:
  """Disabled letters use finfo.min. Charged letters use -inf. The fills differ."""
  stored, _chosen, _chi_logits, _chi_degrees = _call(
    _step_inputs(charged=True, bias_x=50.0, bias_k=50.0),
  )
  floor = float(jnp.finfo(stored.dtype).min)
  assert float(stored[_X]) == floor
  assert float(stored[_LYS]) == -np.inf
  assert floor != -np.inf


def test_knob_semantics_minp_only_when_temperature_set() -> None:
  """T_eff None is argmax with finite stored logits. A set T warps before the draw."""
  inputs = _step_inputs()
  cold, choice, _chi_logits, _chi_degrees = _call(inputs, sequence_temperature=None)
  warm, _warm_choice, _chi_logits, _chi_degrees = _call(
    inputs,
    sequence_temperature=0.5,
    seq_min_p=1.0,
  )
  assert not np.any(np.isinf(np.asarray(cold)))
  assert int(choice) == int(jnp.argmax(cold))
  assert np.any(np.isneginf(np.asarray(warm)))


def test_knob_semantics_ignore_chain_mask_zeros_writes_x() -> None:
  """An unsampled row becomes X with zero logits, and its decoder row stays zero."""
  decoded = _decode(
    chain_mask=np.asarray([False, True]),
    order=np.asarray([0, 1], dtype=np.int32),
    ignore_chain_mask_zeros=True,
  )
  assert int(decoded.sequence[0]) == _X
  assert np.all(decoded.sequence_logits[0] == 0)
  assert np.all(decoded.node_scalars[1, 0] == 0)
  assert not np.all(decoded.node_scalars[0, 0] == 0)


def _separating_uniform(stored: jax.Array, left: float, right: float) -> float | None:
  for unit in np.linspace(0.02, 0.98, 49):
    draw = jnp.asarray(unit, dtype=stored.dtype)
    soft = categorical_draw(jax.nn.softmax(stored / left), draw)
    hard = categorical_draw(jax.nn.softmax(stored / right), draw)
    if int(soft) != int(hard):
      return float(unit)
  return None


def test_knob_semantics_fs_sequence_temp() -> None:
  """First-shell rows use fs_sequence_temp. Other rows use T, or 1e-6 when T is absent."""
  _encoder, decoder, joint = _models()
  for temperature in (None, 0.3):
    for min_p in (0.0, 0.05):
      for shell in (False, True):
        inputs = _step_inputs(first_shell=shell)
        probe, _choice, _chi_logits, _chi_degrees, _consumed = joint(
          decoder,
          inputs,
          sequence_temperature=temperature,
          chi_temperature=None,
          fs_sequence_temp=0.2,
          seq_min_p=min_p,
          chi_min_p=0.0,
          ala_budget=4,
          gly_budget=0,
          ignore_chain_mask_zeros=False,
          repack_all=False,
        )
        expected_t = 0.2 if shell else (1e-6 if temperature is None else temperature)
        uniform = _separating_uniform(probe, expected_t, 1e-6) if not shell and temperature == 0.3 else 0.5
        if uniform is None:
          uniform = 0.5
        draw = jnp.asarray(uniform, dtype=probe.dtype)
        _stored, chosen, _chi_logits, _chi_degrees, _consumed = joint(
          decoder,
          eqx.tree_at(lambda item: item.uniforms, inputs, jnp.full((8,), uniform, dtype=probe.dtype)),
          sequence_temperature=temperature,
          chi_temperature=None,
          fs_sequence_temp=0.2,
          seq_min_p=min_p,
          chi_min_p=0.0,
          ala_budget=4,
          gly_budget=0,
          ignore_chain_mask_zeros=False,
          repack_all=False,
        )
        routed = categorical_draw(jax.nn.softmax(probe / expected_t), draw)
        assert int(chosen) == int(routed)
        if not shell and temperature == 0.3 and _separating_uniform(probe, 0.3, 1e-6) is not None:
          wrong = categorical_draw(jax.nn.softmax(probe / 1e-6), draw)
          assert int(chosen) != int(wrong)


def test_knob_semantics_ignore_ligand() -> None:
  """ignore_ligand drops ligand atoms and matches a ligand-free copy of the same PDB."""
  source = Path(__file__).resolve().parents[1] / "fixtures" / "laser" / "101m_1.pdb"
  kept = [
    line
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True)
    if line.startswith("ATOM")
  ]
  stripped = source.with_name("101m_1.protein_only.pdb")
  stripped.write_text("".join(kept), encoding="utf-8")
  try:
    full = featurize(source, ignore_ligand=False)
    ignored = featurize(source, ignore_ligand=True)
    empty = featurize(stripped, ignore_ligand=False)
  finally:
    stripped.unlink(missing_ok=True)
  assert full.ligand_coords.shape[0] > 0
  assert ignored.ligand_coords.shape == (0, 3)
  assert empty.ligand_coords.shape == (0, 3)
  assert not np.any(ignored.first_shell_ligand_contact_mask)
  assert np.array_equal(
    ignored.first_shell_ligand_contact_mask,
    empty.first_shell_ligand_contact_mask,
  )
  assert np.array_equal(ignored.sequence_indices, empty.sequence_indices)


def test_knob_semantics_repack_all() -> None:
  """repack_all drops input χ on a fixed row. repack_only also fixes the sequence."""
  chi = np.full((2, 4), np.nan, dtype=np.float32)
  chi[0, 0] = 12.5
  sequence = np.asarray([_VAL, _ALA], dtype=np.int32)
  kept = _decode(sequence=sequence, chi=chi, chain_mask=np.asarray([True, False]))
  repacked = _decode(
    sequence=sequence,
    chi=chi,
    chain_mask=np.asarray([True, False]),
    repack_all=True,
  )
  only = _decode(
    sequence=sequence,
    chi=chi,
    chain_mask=np.asarray([False, False]),
    repack_only=True,
  )
  assert kept.chi_degrees[0, 0] == np.float32(12.5)
  assert int(repacked.sequence[0]) == _VAL
  assert repacked.chi_degrees[0, 0] != np.float32(12.5)
  assert np.array_equal(only.sequence, sequence)
  assert only.chi_degrees[0, 0] != np.float32(12.5)


def test_knob_semantics_deterministic_e2e_argmax() -> None:
  """Injected order plus argmax is a pure function of that order.

  Exact agreement with upstream on the 21 fixtures is the laser_decode_e2e
  sidecar, not this test.
  """
  order = np.asarray([1, 0], dtype=np.int32)
  first = _decode(order=order)
  second = _decode(order=order)
  reversed_order = _decode(order=order[::-1])
  assert np.array_equal(first.sequence, second.sequence)
  assert np.array_equal(first.chi_bins, second.chi_bins)
  assert np.allclose(first.chi_degrees, second.chi_degrees, equal_nan=True)
  assert not np.array_equal(first.sequence_logits, reversed_order.sequence_logits)


def test_knob_semantics_ignore_chain_mask_zeros_inverts_disabled() -> None:
  """The disabled fill follows the rows being sampled, which the flag swaps."""
  floor = float(jnp.finfo(jnp.float32).min)
  designed, _chosen, _chi_logits, _chi_degrees = _call(
    _step_inputs(chain_mask=False, bias_x=40.0),
    ignore_chain_mask_zeros=False,
  )
  fixed, _chosen, _chi_logits, _chi_degrees = _call(
    _step_inputs(chain_mask=True, bias_x=40.0),
    ignore_chain_mask_zeros=False,
  )
  designed_flag, _chosen, _chi_logits, _chi_degrees = _call(
    _step_inputs(chain_mask=False, bias_x=40.0),
    ignore_chain_mask_zeros=True,
  )
  fixed_flag, _chosen, _chi_logits, _chi_degrees = _call(
    _step_inputs(chain_mask=True, bias_x=40.0),
    ignore_chain_mask_zeros=True,
  )
  assert float(designed[_X]) == floor
  assert float(fixed[_X]) != floor
  assert float(designed_flag[_X]) != floor
  assert float(fixed_flag[_X]) == floor


def test_knob_semantics_fixed_after_designed_and_skipped_predecessor() -> None:
  """A fixed label is visible only after its own step. A skipped row still is."""
  sequence = np.asarray([_VAL, _ALA], dtype=np.int32)
  chain = np.asarray([True, False])
  designed_first = _decode(
    sequence=sequence,
    chain_mask=chain,
    order=np.asarray([1, 0], dtype=np.int32),
  )
  fixed_first = _decode(
    sequence=sequence,
    chain_mask=chain,
    order=np.asarray([0, 1], dtype=np.int32),
  )
  # Negative control: the fixed letter is already a predecessor of the designed row.
  assert not np.allclose(designed_first.sequence_logits[1], fixed_first.sequence_logits[1])

  skipped_first = _decode(
    chain_mask=np.asarray([False, True]),
    order=np.asarray([0, 1], dtype=np.int32),
    ignore_chain_mask_zeros=True,
  )
  skipped_last = _decode(
    chain_mask=np.asarray([False, True]),
    order=np.asarray([1, 0], dtype=np.int32),
    ignore_chain_mask_zeros=True,
  )
  assert int(skipped_first.sequence[0]) == _X
  assert np.all(skipped_first.node_scalars[1, 0] == 0)
  # Negative control: dropping the skipped row from the predecessor set changes the neighbour.
  assert not np.allclose(skipped_first.sequence_logits[1], skipped_last.sequence_logits[1])


def test_circular_chi_delta_wraps_at_180() -> None:
  """179.9999 versus -179.9999 is a fraction of a degree, not a full turn."""
  delta = circular_abs_delta_deg(np.asarray([179.9999]), np.asarray([-179.9999]))
  assert float(delta[0]) < 1e-3
  assert float(delta[0]) != 360.0 - 1e-4
  mask = chi_position_mask(np.asarray([7, _X, _VAL]))
  assert not np.any(mask[0])
  assert not np.any(mask[1])
  assert bool(mask[2, 0])


def test_scan_loop_body_has_no_quadratic_residue_op() -> None:
  """The scan body is one step. No residue-by-residue tensor is built inside it."""
  n_res = 7
  _encoder, decoder, joint = _models()
  inputs = _step_inputs(n_res=n_res)
  carry = _Carry(
    sequence=inputs.sequence,
    chi_degrees=inputs.chi_degrees,
    node_scalars=inputs.node_scalars,
    node_vectors=inputs.node_vectors,
    ala_count=inputs.ala_count,
    gly_count=inputs.gly_count,
    sequence_logits=jnp.zeros((n_res, 21), dtype=jnp.float32),
    chi_logits=jnp.zeros((n_res, 4, 72), dtype=jnp.float32),
    cursor=jnp.int32(0),
  )

  def body(current: _Carry, residue: jax.Array) -> _Carry:
    return _commit_residue(
      current,
      residue,
      joint,
      decoder,
      ligand_scalars=inputs.ligand_scalars,
      pr_edges=inputs.pr_edges,
      pr_neighbours=inputs.pr_neighbours,
      pr_mask=inputs.pr_mask,
      lp_edges=inputs.lp_edges,
      lp_neighbours=inputs.lp_neighbours,
      lp_mask=inputs.lp_mask,
      decoding_order=inputs.decoding_order,
      input_sequence=inputs.input_sequence,
      input_chi=inputs.input_chi,
      chain_mask=inputs.chain_mask,
      first_shell=inputs.first_shell,
      budget_mask=inputs.budget_mask,
      charged_mask=inputs.charged_mask,
      disabled=inputs.disabled,
      bias=inputs.bias,
      uniforms=inputs.uniforms,
      sequence_temperature=None,
      chi_temperature=None,
      fs_sequence_temp=None,
      seq_min_p=0.0,
      chi_min_p=0.0,
      ala_budget=4,
      gly_budget=0,
      ignore_chain_mask_zeros=False,
      repack_all=False,
    )

  closed = jax.make_jaxpr(body)(carry, jnp.int32(0))
  hits = _quadratic_shapes(closed, n_res)
  assert hits == [], f"O(L^2) shapes in the scan body: {hits[:8]}"
