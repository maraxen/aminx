# ruff: noqa: S101
"""Potts autoregressive decode, PSSMMix, refine modes, and optimize knobs."""

from __future__ import annotations

import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.decode import PottsARDecode, mask_refine_x, pssm_mix
from aminx.families.potts_mpnn.refine import BindingTables, PottsRefine, nodes_attention
from aminx.families.potts_mpnn.sample_host import (
    build_tie_groups_np,
    force_optimize_num_samples,
    load_optimize_fasta,
    sample_schema,
    seq_to_ints,
    upstream_refine_order,
)
from aminx.model.decoder import DecoderLayer
from aminx.run.options import PottsMPNNOptions
from aminx.run.specs import SamplingSpecification

_H = 4
_V = 21
_A = 22


def _zero_floats(module: eqx.Module) -> eqx.Module:
    def _leaf(leaf: object) -> object:
        if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return jnp.zeros(leaf.shape, leaf.dtype)
        return leaf

    return jax.tree.map(_leaf, module)


def _decoder() -> PottsARDecode:
    layer = DecoderLayer(_H, 3 * _H, _H, dropout_rate=0.0, key=jax.random.PRNGKey(0))
    layer = eqx.tree_at(
        lambda item: (item.message_mlp, item.dense),
        layer,
        (_zero_floats(layer.message_mlp), _zero_floats(layer.dense)),
    )
    embed = eqx.nn.Embedding(_V, _H, key=jax.random.PRNGKey(1))
    readout = eqx.nn.Linear(_H, _V, key=jax.random.PRNGKey(2))
    weight = jnp.zeros_like(readout.weight)
    bias = jnp.full((_V,), -50.0)
    bias = bias.at[3].set(50.0)
    readout = eqx.tree_at(lambda item: (item.weight, item.bias), readout, (weight, bias))
    return PottsARDecode(layers=(layer,), w_s_embed=embed, w_out=readout)


def _decode(
    module: PottsARDecode,
    *,
    present: np.ndarray,
    s_true: np.ndarray,
    chain_m_pos: np.ndarray,
    tie_groups: np.ndarray,
    h_v: np.ndarray | None = None,
    bias_peak: int = 3,
) -> object:
    length = present.shape[0]
    hidden = np.zeros((length, _H), dtype=np.float32)
    if h_v is not None:
        hidden[:] = h_v
    module = eqx.tree_at(
        lambda item: item.w_out.bias,
        module,
        jnp.full((_V,), -50.0).at[bias_peak].set(50.0),
    )
    return module(
        jnp.asarray(hidden),
        jnp.zeros((length, 1, _H), dtype=jnp.float32),
        jnp.zeros((length, 1), dtype=jnp.int32),
        jnp.asarray(present, dtype=jnp.float32),
        jnp.ones((length,), dtype=jnp.bool_),
        jnp.asarray(s_true, dtype=jnp.int32),
        jnp.ones((length,), dtype=jnp.float32),
        jnp.asarray(chain_m_pos, dtype=jnp.float32),
        jnp.asarray(tie_groups, dtype=jnp.int32),
        jnp.ones((length,), dtype=jnp.float32),
        jnp.zeros((length,), dtype=jnp.float32),
        jnp.full((length,), 0.5, dtype=jnp.float32),
        jnp.zeros((_V,), dtype=jnp.float32),
        jnp.zeros((_V,), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length,), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        temperature=1.0,
        pssm_multi=0.0,
        pssm_bias_flag=False,
        pssm_log_odds_flag=False,
    )


def test_single_ar_step_follows_injected_uniform() -> None:
    decoded = _decode(
        _decoder(),
        present=np.ones(1),
        s_true=np.zeros(1, dtype=np.int32),
        chain_m_pos=np.ones(1),
        tie_groups=np.asarray([[0]], dtype=np.int32),
    )
    assert int(decoded.sequence[0]) == 3


def test_tied_last_member_selects_fixed_or_designed_token() -> None:
    module = _decoder()
    fixed = _decode(
        module,
        present=np.ones(2),
        s_true=np.asarray([1, 5]),
        chain_m_pos=np.asarray([1.0, 0.0]),
        tie_groups=np.asarray([[0, 1]], dtype=np.int32),
    )
    assert np.array_equal(np.asarray(fixed.sequence), np.asarray([5, 5]))
    designed = _decode(
        module,
        present=np.ones(2),
        s_true=np.asarray([1, 5]),
        chain_m_pos=np.asarray([0.0, 1.0]),
        tie_groups=np.asarray([[0, 1]], dtype=np.int32),
    )
    assert np.array_equal(np.asarray(designed.sequence), np.asarray([3, 3]))


def test_masked_member_freezes_token_and_stops_later_updates() -> None:
    hidden = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    decoded = _decode(
        _decoder(),
        present=np.asarray([1.0, 0.0, 1.0]),
        s_true=np.asarray([1, 7, 2]),
        chain_m_pos=np.ones(3),
        tie_groups=np.asarray([[0, 1, 2]], dtype=np.int32),
        h_v=hidden,
    )
    assert np.array_equal(np.asarray(decoded.sequence), np.asarray([7, 7, 7]))
    assert float(jnp.max(jnp.abs(decoded.h_v_stack[1, 0]))) > 0.0
    assert float(jnp.max(jnp.abs(decoded.h_v_stack[1, 2]))) == 0.0


def test_pssm_mix_applies_bias_before_log_odds() -> None:
    logits = jnp.asarray([2.0, 0.0, 0.0])
    temperature = jnp.asarray(1.0)
    zeros = jnp.zeros((3,))
    coef = jnp.asarray(1.0)
    multi = jnp.asarray(0.5)
    pssm_bias = jnp.asarray([0.0, 1.0, 0.0])
    log_mask = jnp.asarray([1.0, 0.0, 0.0])
    got = pssm_mix(
        logits,
        temperature,
        zeros,
        zeros,
        zeros,
        coef,
        pssm_bias,
        multi,
        log_mask,
        zeros,
        apply_pssm_bias=jnp.bool_(True),
        apply_log_odds=jnp.bool_(True),
        apply_omit_aa=jnp.bool_(False),
    )
    raw = jax.nn.softmax(logits)
    mixed = (1.0 - coef * multi) * raw + (coef * multi) * pssm_bias
    logged = mixed * (log_mask + 0.001)
    logged = logged / jnp.sum(logged)
    swapped = raw * (log_mask + 0.001)
    swapped = swapped / jnp.sum(swapped)
    swapped = (1.0 - coef * multi) * swapped + (coef * multi) * pssm_bias
    assert np.allclose(got, logged)
    assert not np.allclose(got, swapped)


def _binding(length: int, alphabet: int, *, etab: jax.Array | None = None) -> BindingTables:
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
    max_iters: int = 3,
    bias_index: int | None = None,
    chain_mask: np.ndarray | None = None,
    sequence: np.ndarray | None = None,
    binding: str = "none",
    tables: BindingTables | None = None,
    uniforms: jax.Array | None = None,
) -> object:
    length = etab.shape[0]
    module = _decoder()
    refiner = PottsRefine(layers=module.layers, w_s_embed=module.w_s_embed, w_out=module.w_out)
    seq = np.zeros(length, dtype=np.int32) if sequence is None else sequence
    mask = np.ones(length, dtype=np.float32) if chain_mask is None else chain_mask
    bias = np.zeros(_V, dtype=np.float32)
    if bias_index is not None:
        bias[bias_index] = 50.0
    draws = uniforms if uniforms is not None else jnp.full((max_iters, length), 0.5)
    return refiner(
        mode,
        jnp.asarray(seq),
        etab,
        jnp.zeros((length, 1), dtype=jnp.int32),
        jnp.ones((length,), dtype=jnp.bool_),
        jnp.ones((length,), dtype=jnp.float32),
        jnp.asarray(mask),
        jnp.ones((length,), dtype=jnp.float32),
        jnp.arange(length, dtype=jnp.int32),
        draws,
        jnp.zeros((_V,), dtype=jnp.float32),
        jnp.asarray(bias),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length,), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.zeros((length, _V), dtype=jnp.float32),
        jnp.arange(length, dtype=jnp.int32)[:, None],
        jnp.ones((length,), dtype=jnp.float32),
        jnp.zeros((length, _H), dtype=jnp.float32),
        jnp.zeros((length, 1, _H), dtype=jnp.float32),
        _binding(length, etab.shape[-1]) if tables is None else tables,
        temperature=1.0,
        pssm_multi=0.0,
        pssm_bias_flag=False,
        pssm_log_odds_flag=False,
        binding=binding,
        tied=False,
        tied_epistasis=False,
        max_iters=max_iters,
    )


def test_refine_modes_and_converge_stop() -> None:
    length = 2
    flat = jnp.zeros((length, 1, _A, _A))
    peaked = _refine("potts", flat, bias_index=3, max_iters=5)
    assert int(peaked.n_iters) == 1
    assert np.array_equal(np.asarray(peaked.sequence), np.asarray([3, 3]))

    stopped = _refine("potts_converge", flat, max_iters=5)
    assert int(stopped.n_iters) == 1
    assert float(stopped.ener_delta) == 0.0

    eye = jnp.zeros((length, 1, _A, _A)).at[:, 0].set(jnp.eye(_A))
    capped = _refine("potts_converge", eye, max_iters=3)
    assert int(capped.n_iters) == 3

    nodes = _refine(
        "nodes",
        flat,
        sequence=np.asarray([4, 4], dtype=np.int32),
        chain_mask=np.asarray([0.0, 1.0]),
    )
    assert np.array_equal(np.asarray(nodes.sequence), np.asarray([4, 3]))
    mask_bw, _mask_fw = nodes_attention(jnp.ones(2), jnp.ones(2), 2)
    assert np.array_equal(np.asarray(mask_bw[:, 0]), np.asarray([0.0, 0.0]))

    shared = jnp.zeros((1, length, 1, _A, _A)).at[:, :, 0].set(jnp.eye(_A) * 5)
    complex_etab = jnp.zeros((length, 1, _A, _A)).at[:, 0].set(jnp.eye(_A) * 5)
    bound = _refine(
        "potts_converge",
        complex_etab,
        max_iters=4,
        binding="both",
        tables=_binding(length, _A, etab=shared),
    )
    assert int(bound.n_iters) == 1
    assert float(bound.ener_delta) == 0.0


def test_refine_never_draws_x_when_its_logit_is_largest() -> None:
    raw = jnp.zeros((_V,)).at[20].set(100.0)
    masked = mask_refine_x(raw)
    assert float(masked[20]) < float(jnp.max(masked[:20]))
    length = 1
    etab = jnp.zeros((length, 1, _A, _A))
    etab = etab.at[:, 0, 20, 20].set(-1000.0)
    for uniform in (0.1, 0.5, 0.9):
        refined = _refine(
            "potts",
            etab,
            uniforms=jnp.full((1, length), uniform),
        )
        assert int(refined.sequence[0]) != 20


def test_optimize_fasta_and_num_samples_knobs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fasta = tmp_path / "opt.fasta"
    fasta.write_text(">other\nAC\n>toy\nA:W\n>toy:B\nACDE\n", encoding="utf-8")
    assert load_optimize_fasta(fasta, "toy") == ("AW", "ACDE")
    assert np.array_equal(seq_to_ints("A-X"), np.asarray([0, 20, 21]))
    with pytest.raises(ValueError, match="outside"):
        seq_to_ints("A!")
    groups = build_tie_groups_np(((1, 0),), 3, 4)
    assert groups.shape[1] == 2
    assert np.array_equal(groups[0, :2], np.asarray([1, 0]))
    with pytest.raises(ValueError, match="overlapping"):
        build_tie_groups_np(((0, 1), (1, 2)), 3, 3)
    chain = np.asarray([1.0, 1.0, 0.0])
    # |randn[0]| > |randn[1]| so the spec fresh key is not the stored [2, 0, 1].
    randn = np.asarray([0.4, -0.2, 0.1])
    stored = upstream_refine_order(
        np.asarray([2, 0, 1]),
        chain,
        randn,
        num_samples=2,
        chain_suffix="",
        stored_orders_present=True,
    )
    assert np.array_equal(stored, np.arange(3))
    fresh = upstream_refine_order(
        np.asarray([2, 0, 1]),
        chain,
        randn,
        num_samples=1,
        chain_suffix="_A",
        stored_orders_present=True,
    )
    assert not np.array_equal(fresh, np.asarray([2, 0, 1]))
    spec = SamplingSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        num_samples=4,
        return_logits=False,
        potts_mpnn=PottsMPNNOptions(optimize_pdb=True, optimization_mode="none"),
    )
    with caplog.at_level(logging.WARNING):
        force_optimize_num_samples(spec)
    assert spec.num_samples == 1
    assert spec.run_spec.sampling.num_samples == 1
    assert "forcing num_samples=1" in caplog.text
    refining = SamplingSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        num_samples=1,
        return_logits=False,
        potts_mpnn=PottsMPNNOptions(optimize_fasta="opt.fasta", optimization_mode="potts"),
    )
    assert set(sample_schema(refining)) == {"refined_sequence"}
    designing = SamplingSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        num_samples=1,
        return_logits=False,
        potts_mpnn=PottsMPNNOptions(optimize_pdb=True, optimization_mode="none"),
    )
    assert "sequence" in sample_schema(designing)
    assert "sample_rank" in sample_schema(designing)
