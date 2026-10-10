# ruff: noqa: S101, PLR2004
"""The host-loop pH methods (``ph_methods``): autoregressive, converged_mcmc(+combined), two_phase, gibbs.

Random-init weights on a synthetic helix. What is checked here needs no upstream: the draw machinery (replay and forced replay
reproduce a fresh run), the decoder field against ``teacher_forced`` itself, each method's structural contract (pins stay protonated,
only designable positions move, determinism), the scan-placement formulas against an independent numpy computation, and the
fixed-point property of the converged and Gibbs methods at low temperature. Agreement with upstream is settled by the
``protonpotts_ph_methods`` wave against the P4j dump, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ph_methods
from aminx.families.protonpotts_mpnn.decode import teacher_forced
from aminx.families.protonpotts_mpnn.driver import _JIT_TABLE, _graph_args, _prepare
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
from aminx.families.protonpotts_mpnn.ph_methods import (
  UNK_INDEX,
  DecoderField,
  Draws,
  MethodContext,
  converge,
  design_methods,
  gibbs,
  masked_infill,
  pick,
  plan_for_methods,
  score_at,
  selective_reward_decoder,
  softmax_probs,
  two_phase,
)
from aminx.families.protonpotts_mpnn.ph_plan import Pin, placement_scores, valid_token_mask
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, token_index
from aminx.run.options import ProtonPottsOptions

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR", "VAL", "LEU"]
HIS_P, HIS_S = token_index("HIS-P"), token_index("HIS-S")
V = PROTONPOTTS_V6.size


def _atom(serial: int, name: str, resname: str, chain: str, number: int, xyz: tuple[float, float, float]) -> str:
  padded = f" {name:<3}"
  return (
    f"ATOM  {serial:>5} {padded} {resname:>3} {chain}{number:>4}    "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2}"
  )


def _write_pdb(path: Path, chains: list[tuple[str, list[str], float]]) -> Path:
  rng = np.random.default_rng(0)
  lines: list[str] = []
  for chain, names, offset in chains:
    for i, resname in enumerate(names, start=1):
      centre = np.array([3.8 * i + offset, 2.0 * np.sin(i), 2.0 * np.cos(i)])
      for k, atom in enumerate(("N", "CA", "C", "O")):
        xyz = centre + rng.normal(scale=0.4, size=3) + np.array([0.6 * k, 0.0, 0.0])
        lines.append(_atom(len(lines) + 1, atom, resname, chain, i, (xyz[0], xyz[1], xyz[2])))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


class Structure(NamedTuple):
  model: PottsMPNN
  graph_args: tuple[Any, ...]
  table: jax.Array
  e_idx: jax.Array
  native: np.ndarray
  binder: np.ndarray
  res_id: np.ndarray


@pytest.fixture(scope="module")
def structure(tmp_path_factory: pytest.TempPathFactory) -> Structure:
  """Binder chain A (12 residues) and a target chain B (12 residues), random-init model."""
  path = _write_pdb(tmp_path_factory.mktemp("ph_methods") / "toy.pdb", [("A", RESIDUES, 0.0), ("B", RESIDUES, 50.0)])
  prepared = _prepare(path, ProtonPottsOptions(), "sample", ())
  model = PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)
  args = _graph_args(prepared.graph)
  table, e_idx = _JIT_TABLE(model, *args)
  native = np.asarray(prepared.graph.sequences[0], dtype=np.int32)
  binder = np.asarray([chain == "A" for chain, *_rest in prepared.kept], dtype=bool)
  res_id = np.asarray([number for _chain, number, *_rest in prepared.kept], dtype=np.int32)
  return Structure(model, args, table, e_idx, native, binder, res_id)


def _config(**overrides: Any) -> PHDesignConfig:  # noqa: ANN401
  values: dict[str, Any] = {
    "method": "converged_mcmc", "backend": "potts", "binder_chain": "A", "center_types": ("HIS-P",), "temperature": 0.1,
    "samples_per_site": 2, "neighbour_k": 0, "max_mutations": 6, "cv_patience": 1, "cv_max": 2, "combined_lambda": 0.3,
  }
  values.update(overrides)
  return PHDesignConfig(**values)


def _context(s: Structure, *, decoder: bool = True) -> MethodContext:
  field = DecoderField.from_model(s.model, s.graph_args) if decoder else None
  return MethodContext(s.table, s.e_idx, valid_token_mask(("HIS-A", "ASP-A", "GLU-A", "UNK")), np.arange(V), field)


def _pin(s: Structure, position: int) -> Pin:
  return Pin(position, "HIS-P", HIS_P, (HIS_S,), int(s.res_id[position]))


# --- draws ------------------------------------------------------------------------------------------------------------


def test_draws_replay_consumes_each_kind_in_order_and_runs_out() -> None:
  draws = Draws(uniforms=[0.1, 0.9], randints=[2, 0], perms=[[1, 0, 2]])
  assert draws.randint(3) == 2
  assert draws.randint(3) == 0
  np.testing.assert_array_equal(draws.permutation(3), [1, 0, 2])
  probs = np.zeros(V)
  probs[:2] = 0.5
  assert draws.pick_probs(probs) == 0  # u = 0.1 falls in the first half
  assert draws.pick_probs(probs) == 1  # u = 0.9 falls in the second
  with pytest.raises(IndexError, match="ran out of uniform"):
    draws.pick_probs(probs)
  with pytest.raises(ValueError, match="outside"):
    Draws(randints=[5], seed=0).randint(3)


def test_draws_fresh_is_seeded() -> None:
  a, b, c = Draws(seed=3), Draws(seed=3), Draws(seed=4)
  assert [a.randint(1000) for _ in range(5)] == [b.randint(1000) for _ in range(5)]
  assert [Draws(seed=3).randint(10**6)] != [c.randint(10**6)]


def test_pick_accumulates_the_inverse_cdf_in_cdf_order() -> None:
  probs = np.zeros(V)
  probs[0] = probs[1] = 0.5
  order = np.arange(V)
  order[[0, 1]] = order[[1, 0]]  # token 1 comes first
  assert Draws(uniforms=[0.25]).pick_probs(probs) == 0
  assert Draws(uniforms=[0.25], cdf_order=order).pick_probs(probs) == 1


def test_forced_draws_return_the_recorded_choice_and_keep_the_probabilities() -> None:
  draws = Draws(choices=[7, 3])
  probs = np.full(V, 1.0 / V)
  assert draws.pick_probs(probs) == 7
  assert draws.pick_probs(probs) == 3
  assert [e.choice for e in draws.events] == [7, 3]
  assert np.isnan(draws.events[0].uniform)  # no uniform was given


def test_pick_at_temperature_zero_is_the_first_minimum_and_draws_nothing() -> None:
  score = np.array([3.0, 1.0, 1.0, 2.0] + [9.0] * (V - 4), dtype=np.float32)
  draws = Draws(seed=0)
  assert pick(score, 0.0, draws) == 1
  assert not draws.events


def test_softmax_probs_zeroes_infinity_and_sharpens_with_low_temperature() -> None:
  score = np.array([0.0, 1.0, np.inf] + [2.0] * (V - 3), dtype=np.float32)
  warm, cold = softmax_probs(score, 1.0), softmax_probs(score, 0.1)
  assert warm.dtype == np.float32
  assert warm[2] == 0.0 == cold[2]
  np.testing.assert_allclose([warm.sum(), cold.sum()], 1.0, rtol=1e-6)
  assert cold[0] > warm[0]


# --- the decoder field --------------------------------------------------------------------------------------------------


def test_decoder_field_is_negative_log_probs_of_the_conditional_minus_self_pass(structure: Structure) -> None:
  s = structure
  field = DecoderField.from_model(s.model, s.graph_args)
  out = field(s.native)
  assert out.shape == (s.native.shape[0], V)
  np.testing.assert_allclose(np.exp(-out).sum(axis=-1), 1.0, rtol=1e-4)  # log-probs: each position is a distribution
  # an independent call of the graded primitive with another decoding order gives the same field (the pattern has no order)
  coords, present, residue_idx, chain_index, _pad = s.graph_args
  h_v, h_e, nbr = ph_methods._JIT_STATES(s.model, coords, present, residue_idx, chain_index)  # noqa: SLF001
  from aminx.families.potts_mpnn.model import cast_floating  # noqa: PLC0415

  scored = cast_floating(s.model, h_v.dtype)
  length = h_v.shape[0]
  _logits, log_probs = teacher_forced(
    scored.mpnn.decoder, scored.mpnn.w_s_embed, scored.mpnn.w_out, h_v, h_e, nbr, present, jnp.asarray(s.native),
    jnp.full((length,), ph_methods.UPSTREAM_SAMPLE_TEMPERATURE, dtype=h_v.dtype), jnp.zeros((length, V), dtype=h_v.dtype),
    pattern="conditional_minus_self", decoding_order=jnp.asarray(np.random.default_rng(1).permutation(length)),
  )  # fmt: skip
  np.testing.assert_allclose(out, -np.asarray(log_probs), rtol=1e-5, atol=1e-5)


def test_decoder_field_is_driven_by_the_other_residues_far_more_than_by_the_residue_itself(structure: Structure) -> None:
  s = structure
  field = DecoderField.from_model(s.model, s.graph_args)
  base = field(s.native)
  i = 3
  self_changed = s.native.copy()
  self_changed[i] = (s.native[i] + 1) % 20
  other_changed = s.native.copy()
  other_changed[i + 1] = (s.native[i + 1] + 1) % 20
  # conditional_minus_self: row i does not see token i directly, but sees its neighbours. A residue's own token still reaches its row
  # INDIRECTLY (a neighbour's hidden state carries it back in the later decoder layers), so the self effect is small, not zero.
  self_effect = float(np.abs(field(self_changed)[i] - base[i]).max())
  other_effect = float(np.abs(field(other_changed)[i] - base[i]).max())
  assert other_effect > 1e-6
  assert self_effect < other_effect


# --- placement ----------------------------------------------------------------------------------------------------------


def test_non_selective_placement_scores_subtract_the_scanned_tokens_energy() -> None:
  rng = np.random.default_rng(0)
  field = rng.normal(size=(6, V))
  base_seq = np.array([0, 1, 2, 3, 4, 5])
  mask = np.array([True, True, True, False, True, True])
  got = placement_scores(field, HIS_P, (HIS_S,), mask, selective=False, base_sequence=base_seq)
  want = field[:, HIS_P] - field[np.arange(6), base_seq]
  np.testing.assert_allclose(got[mask], want[mask])
  assert np.isinf(got[~mask]).all()
  with pytest.raises(ValueError, match="needs base_sequence"):
    placement_scores(field, HIS_P, (HIS_S,), mask, selective=False)
  # the selective score does not depend on the scanned tokens (the base cancels)
  np.testing.assert_allclose(
    placement_scores(field, HIS_P, (HIS_S,), mask)[mask], (field[:, HIS_P] - field[:, HIS_S])[mask]
  )


def test_scan_mpnn_places_by_the_decoder_field_of_the_binder_masked_native(structure: Structure) -> None:
  s = structure
  ctx = _context(s)
  config = _config(placement_by="scan_mpnn", method="converged_mcmc", backend="potts")
  plan = plan_for_methods(ctx, s.native, s.binder, s.res_id, np.asarray(s.e_idx), config)
  assert plan is not None
  masked = s.native.copy()
  masked[s.binder] = UNK_INDEX
  field = ctx.need_decoder()(masked).astype(np.float64)
  gap = np.where(s.binder, field[:, HIS_P] - field[:, HIS_S], np.inf)
  assert plan.pins[0].position == int(np.argmin(gap))
  unmasked = ctx.need_decoder()(s.native).astype(np.float64)
  assert np.abs(unmasked - field).max() > 1e-3  # the mask matters: scanning the unmasked native is a different field


def test_scan_potts_and_scan_mpnn_use_different_fields(structure: Structure) -> None:
  s = structure
  ctx = _context(s)
  e_idx = np.asarray(s.e_idx)
  potts = plan_for_methods(ctx, s.native, s.binder, s.res_id, e_idx, _config(placement_by="scan_potts"))
  field = np.asarray(ph_methods._CANDIDATES(s.table, s.e_idx, jnp.asarray(s.native)))  # noqa: SLF001
  gap = np.where(s.binder, field[:, HIS_P] - field[:, HIS_S], np.inf)
  assert potts is not None
  assert potts.pins[0].position == int(np.argmin(gap))


# --- autoregressive -----------------------------------------------------------------------------------------------------


def _infill_inputs(s: Structure) -> tuple[Pin, list[int]]:
  pin = _pin(s, 1)
  order = [int(i) for i in (3, 5, 2, 7, 4)]
  return pin, order


def test_masked_infill_keeps_the_centre_and_everything_outside_the_order(structure: Structure) -> None:
  s = structure
  ctx = _context(s)
  pin, order = _infill_inputs(s)
  config = _config(method="autoregressive", backend="mpnn")
  result = masked_infill(ctx, config, [pin], order, s.native, Draws(seed=0))
  outside = np.ones(s.native.shape[0], dtype=bool)
  outside[order] = False
  outside[pin.position] = False
  np.testing.assert_array_equal(result.sequence[outside], s.native[outside])
  assert result.sequence[pin.position] == HIS_P
  assert not np.any(result.sequence[order] == UNK_INDEX)  # every visited position was decoded
  assert ctx.valid[result.sequence[order]].all()
  assert len(result.events) == len(order)  # one draw per visited position
  sd_nat, sd_sel = result.zscales  # type: ignore[misc]
  assert sd_nat >= ph_methods.STABILITY_EPS
  assert sd_sel >= ph_methods.STABILITY_EPS


def test_masked_infill_replay_and_forced_replay_reproduce_a_fresh_run(structure: Structure) -> None:
  s = structure
  ctx = _context(s)
  pin, order = _infill_inputs(s)
  config = _config(method="autoregressive", backend="mpnn")
  fresh = Draws(seed=5, cdf_order=ctx.cdf_order)
  first = masked_infill(ctx, config, [pin], order, s.native, fresh)
  replay = Draws(uniforms=[e.uniform for e in fresh.events], cdf_order=ctx.cdf_order)
  forced = Draws(choices=[e.choice for e in fresh.events])
  np.testing.assert_array_equal(masked_infill(ctx, config, [pin], order, s.native, replay).sequence, first.sequence)
  again = masked_infill(ctx, config, [pin], order, s.native, forced)
  np.testing.assert_array_equal(again.sequence, first.sequence)
  for a, b in zip(again.events, first.events, strict=True):  # the forced run sees the same probabilities at every step
    np.testing.assert_allclose(a.probs, b.probs, rtol=1e-6, atol=1e-7)


def test_masked_infill_decoder_selectivity_uses_no_zscales_and_a_pure_probability_reward(structure: Structure) -> None:
  s = structure
  ctx = _context(s)
  pin, order = _infill_inputs(s)
  config = _config(method="autoregressive", backend="mpnn", selective_source="decoder")
  result = masked_infill(ctx, config, [pin], order, s.native, Draws(seed=1))
  assert result.zscales is None
  assert result.sequence[pin.position] == HIS_P
  # lam = 0: the reward is the plain target-state probability, on the valid tokens only
  seq = s.native.copy()
  seq[pin.position] = HIS_P
  valid_idx = [int(a) for a in np.flatnonzero(ctx.valid)]
  reward = selective_reward_decoder(ctx, valid_idx, seq, 4, [pin], 0.0, np.float32)
  want = np.exp(-ctx.need_decoder()(seq)[4])
  np.testing.assert_allclose(reward[valid_idx], want[valid_idx], rtol=1e-5)
  assert np.isneginf(reward[~ctx.valid]).all()
  assert int(seq[pin.position]) == HIS_P  # restored


# --- converged_mcmc, two_phase ------------------------------------------------------------------------------------------


def test_score_at_combines_the_stability_and_deprotonated_rows(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  pin = _pin(s, 1)
  field_at = ctx.field_at("potts")
  seq = s.native.copy()
  seq[pin.position] = HIS_P
  stab = ctx.rows(seq, [4])[0]
  dep = seq.copy()
  dep[pin.position] = HIS_S
  off = ctx.rows(dep, [4])[0]
  plain = score_at(ctx, field_at, seq.copy(), 4, [pin], selective=False, lam=None)
  selective = score_at(ctx, field_at, seq.copy(), 4, [pin], selective=True, lam=None)
  blended = score_at(ctx, field_at, seq.copy(), 4, [pin], selective=False, lam=0.3)
  valid = ctx.valid
  np.testing.assert_allclose(plain[valid], stab[valid], rtol=1e-6)
  np.testing.assert_allclose(selective[valid], (stab - off)[valid], rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(blended[valid], (1.3 * stab - 0.3 * off)[valid], rtol=1e-5, atol=1e-6)
  assert np.isinf(plain[~valid]).all()


def test_converge_at_temperature_zero_ends_at_a_fixed_point(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  pin = _pin(s, 1)
  neigh = [2, 3, 4, 5, 6]
  config = _config(temperature=0.0, cv_patience=1, cv_max=50)
  # visit the neighbours cyclically so one unchanged pass proves the fixed point
  draws = Draws(randints=list(range(len(neigh))) * 50)
  result = converge(ctx, config, [pin], neigh, s.native, draws)
  field_at = ctx.field_at("potts")
  for i in neigh:
    row = score_at(ctx, field_at, result.sequence.copy(), i, [pin], selective=True, lam=None)
    assert int(np.argmin(row)) == int(result.sequence[i])
  assert result.sequence[pin.position] == HIS_P
  assert not result.events  # temperature 0 draws no multinomial


def test_converge_replay_reproduces_a_fresh_run_and_stops_at_the_cap(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  pin = _pin(s, 1)
  neigh = [2, 3, 4, 5]
  config = _config(temperature=0.5, cv_patience=100, cv_max=3)  # never patient enough: the cap ends it
  fresh = Draws(seed=2, cdf_order=ctx.cdf_order)
  first = converge(ctx, config, [pin], neigh, s.native, fresh)
  assert len(fresh.randint_log) == 3 * len(neigh)
  replay = Draws(uniforms=[e.uniform for e in fresh.events], randints=fresh.randint_log, cdf_order=ctx.cdf_order)
  np.testing.assert_array_equal(converge(ctx, config, [pin], neigh, s.native, replay).sequence, first.sequence)


def test_converged_mcmc_combined_uses_the_blend_not_the_selective_flag(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  pin = _pin(s, 1)
  neigh = [2, 3, 4, 5]
  base = {"temperature": 0.0, "cv_patience": 1, "cv_max": 4}
  draws = list(range(len(neigh))) * 10
  blended = converge(ctx, _config(method="converged_mcmc_combined", selective=False, **base), [pin], neigh, s.native, Draws(randints=draws))
  selective = converge(ctx, _config(method="converged_mcmc", selective=True, **base), [pin], neigh, s.native, Draws(randints=draws))
  plain = converge(ctx, _config(method="converged_mcmc", selective=False, **base), [pin], neigh, s.native, Draws(randints=draws))
  assert len({tuple(blended.sequence), tuple(selective.sequence), tuple(plain.sequence)}) > 1  # the three objectives differ


def test_two_phase_commits_the_least_disruptive_picks_first(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  pin = _pin(s, 1)
  neigh = [2, 3, 4, 5, 6, 7]
  config = _config(method="two_phase", temperature=0.0, two_phase_frac=0.5, cv_patience=1, cv_max=0)
  field_at = ctx.field_at("potts")
  start = s.native.copy()
  start[pin.position] = HIS_P
  picks = {i: int(np.argmin(score_at(ctx, field_at, start.copy(), i, [pin], selective=True, lam=None))) for i in neigh}
  disruption = {i: float(field_at(start, i)[picks[i]] - field_at(start, i)[int(start[i])]) for i in neigh}
  ranked = sorted(neigh, key=lambda i: disruption[i])
  k = max(1, min(len(neigh) - 1, int(round(0.5 * len(neigh)))))
  result = two_phase(ctx, config, [pin], neigh, s.native, Draws(randints=[0]))  # cv_max = 0: phase 2 changes nothing
  for i in ranked[:k]:
    assert int(result.sequence[i]) == picks[i]
  for i in ranked[k:]:
    assert int(result.sequence[i]) == int(start[i])


# --- gibbs ----------------------------------------------------------------------------------------------------------------


def test_gibbs_at_low_temperature_lowers_the_energy_and_ends_at_a_conditional_minimum(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  config = _config(method="gibbs", temperature=0.0)  # clamped to 1e-3: effectively greedy
  draws = Draws(seed=3, cdf_order=ctx.cdf_order)
  (out,) = gibbs(ctx, config, s.native[None], s.binder, draws)
  valid = jnp.ones(s.native.shape[0], dtype=bool)
  before = float(potts_energy(s.table, s.e_idx, valid, jnp.asarray(s.native)))
  after = float(potts_energy(s.table, s.e_idx, valid, jnp.asarray(out)))
  assert after <= before + 1e-6
  np.testing.assert_array_equal(out[~s.binder], s.native[~s.binder])  # only the free positions move
  for position in np.flatnonzero(s.binder):
    row = np.where(ctx.valid, ctx.rows(out, [int(position)])[0], np.inf)
    best = np.sort(row)[:2]
    if best[1] - best[0] > 0.05:  # skip near-ties, where a 1e-3-temperature draw can still pick the runner-up
      assert int(out[position]) == int(np.argmin(row))
  assert len(draws.perm_log) >= 2  # at least one sweep that moved and one that proved convergence
  assert all(sorted(p.tolist()) == list(range(int(s.binder.sum()))) for p in draws.perm_log)


def test_gibbs_rows_are_independent_and_replayable(structure: Structure) -> None:
  s = structure
  ctx = _context(s, decoder=False)
  config = _config(method="gibbs", temperature=0.3)
  fresh = Draws(seed=4, cdf_order=ctx.cdf_order)
  rows = np.broadcast_to(s.native, (2, s.native.shape[0]))
  first = gibbs(ctx, config, rows, s.binder, fresh)
  replay = Draws(uniforms=[e.uniform for e in fresh.events], perms=fresh.perm_log, cdf_order=ctx.cdf_order)
  again = gibbs(ctx, config, rows, s.binder, replay)
  for a, b in zip(first, again, strict=True):
    np.testing.assert_array_equal(a, b)
  assert not np.array_equal(first[0], first[1])  # two rows, two chains


# --- the structure-level driver -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
  "overrides",
  [
    {"method": "autoregressive", "backend": "mpnn"},
    {"method": "autoregressive", "backend": "mpnn", "selective_source": "decoder"},
    {"method": "converged_mcmc", "backend": "potts"},
    {"method": "converged_mcmc", "backend": "mpnn", "cv_max": 1},
    {"method": "converged_mcmc_combined", "backend": "potts"},
    {"method": "two_phase", "backend": "potts"},
    {"method": "converged_mcmc", "backend": "potts", "placement_by": "scan_mpnn"},
    {"method": "converged_mcmc", "backend": "potts", "selective": False},
  ],
  ids=["ar", "ar_decoder", "mcmc", "mcmc_mpnn", "combined", "two_phase", "scan_mpnn", "nonselective"],
)
def test_design_methods_contract(structure: Structure, overrides: dict[str, Any]) -> None:
  s = structure
  config = _config(**overrides)
  key = jax.random.PRNGKey(7)
  args = (s.model, s.graph_args, s.table, s.e_idx, s.native, s.binder, s.res_id, config)
  designs = design_methods(*args, key=key)
  assert len(designs) == config.samples_per_site
  assert [d.final_potts_energy for d in designs] == sorted(d.final_potts_energy for d in designs)
  valid = jnp.ones(s.native.shape[0], dtype=bool)
  for design in designs:
    assert design.method == config.method
    pin = design.pins[0]
    assert design.sequence[pin.position] == pin.prot_idx  # the centre stays protonated
    assert s.binder[pin.position]
    outside = np.ones(s.native.shape[0], dtype=bool)
    outside[list(design.designable)] = False
    outside[pin.position] = False
    np.testing.assert_array_equal(design.sequence[outside], s.native[outside])  # only the designable set moves
    assert np.isclose(design.final_potts_energy, float(potts_energy(s.table, s.e_idx, valid, jnp.asarray(design.sequence))))
    assert np.isclose(design.selective_energy, sum(design.selective_energies))
  again = design_methods(*args, key=key)
  for a, b in zip(designs, again, strict=True):
    np.testing.assert_array_equal(a.sequence, b.sequence)
  other = design_methods(*args, key=jax.random.PRNGKey(8))
  assert any(not np.array_equal(a.sequence, b.sequence) for a, b in zip(designs, other, strict=True))


def test_design_methods_gibbs_designs_the_binder_chain_and_nothing_else(structure: Structure) -> None:
  s = structure
  config = _config(method="gibbs", temperature=0.3, samples_per_site=2, center_types=())
  designs = design_methods(s.model, s.graph_args, s.table, s.e_idx, s.native, s.binder, s.res_id, config, key=jax.random.PRNGKey(1))
  assert len(designs) == 2
  for design in designs:
    assert design.pins == ()
    assert design.selective_energy == 0.0
    np.testing.assert_array_equal(design.sequence[~s.binder], s.native[~s.binder])
    assert design.designable == tuple(int(i) for i in np.flatnonzero(s.binder))


def test_design_methods_replay_order_is_used_for_autoregressive(structure: Structure) -> None:
  s = structure
  config = _config(method="autoregressive", backend="mpnn", samples_per_site=1)
  args = (s.model, s.graph_args, s.table, s.e_idx, s.native, s.binder, s.res_id, config)
  first = design_methods(*args, key=jax.random.PRNGKey(2))[0]
  order = list(first.designable)[::-1]
  replay = [{"uniforms": np.random.default_rng(0).random(len(order)), "order": order}]
  a = design_methods(*args, replay=replay)[0]
  b = design_methods(*args, replay=replay)[0]
  np.testing.assert_array_equal(a.sequence, b.sequence)  # replay is deterministic
  assert a.designable == first.designable  # the supplied order is a visiting order, not a different designable set


def test_design_methods_needs_a_key_or_replay_and_the_decoder_inputs(structure: Structure) -> None:
  s = structure
  config = _config(method="autoregressive", backend="mpnn")
  with pytest.raises(ValueError, match="PRNG key or replayed draws"):
    design_methods(s.model, s.graph_args, s.table, s.e_idx, s.native, s.binder, s.res_id, config)
  with pytest.raises(ValueError, match="needs the model and graph arguments"):
    design_methods(None, None, s.table, s.e_idx, s.native, s.binder, s.res_id, config, key=jax.random.PRNGKey(0))
  with pytest.raises(ValueError, match="no True entries"):
    design_methods(
      s.model, s.graph_args, s.table, s.e_idx, s.native, np.zeros_like(s.binder), s.res_id, config, key=jax.random.PRNGKey(0)
    )
