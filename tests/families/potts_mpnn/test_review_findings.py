# ruff: noqa: S101
"""Regressions for the potts review findings. Each test failed on the previous behaviour."""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import _ChainSeq, _converter_script, _from_fasta
from aminx.families.potts_mpnn.featurize import PottsFeatures
from aminx.families.potts_mpnn.sample_host import (
    _require_refine_uniforms,
    load_optimize_fasta,
    prepare_sample,
    sample_binding_tables,
)
from aminx.host.family_runner import _dropout_key
from aminx.host.omit_aa_bias import compile_omit_aa_bias
from aminx.run.options import PottsMPNNOptions
from tests.conftest import _hf_hub_reachable

_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _features(length: int) -> PottsFeatures:
    vocab = len(_ALPHABET)
    return PottsFeatures(
        x=np.zeros((length, 4, 3), np.float32),
        s=np.zeros((length,), np.int32),
        present=np.ones((length,), np.float32),
        lengths=np.asarray(length, dtype=np.int32),
        chain_m=np.ones((length,), np.float32),
        chain_m_pos=np.ones((length,), np.float32),
        chain_encoding=np.zeros((length,), np.int32),
        residue_idx=np.arange(length, dtype=np.int32),
        omit_aa_mask=np.zeros((length, vocab), np.float32),
        dihedral_mask=np.zeros((length, 3), np.float32),
        tied_beta=np.ones((length,), np.float32),
        pssm_coef=np.zeros((length,), np.float32),
        pssm_bias=np.zeros((length, vocab), np.float32),
        pssm_log_odds=np.zeros((length, vocab), np.float32),
        bias_by_res=np.zeros((length, vocab), np.float32),
        letter_list=("A",),
        visible_list=(),
        masked_list=("A",),
        masked_chain_lengths=(length,),
        tied_pos=(),
        chain_lens=(length,),
        name="toy",
        L_total=length,
    )


def _spec(**kwargs: object) -> SimpleNamespace:
    base: dict[str, object] = {"omit_aa": (), "bias": None, "fixed_positions": None}
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_converge_refuses_a_short_uniform_table() -> None:
    """Iterations past the drawn rows must not be filled with u=0."""
    with pytest.raises(ValueError, match="uniform"):
        _require_refine_uniforms(
            jnp.ones((8, 2), dtype=jnp.float32),
            "potts_converge",
        )


def test_sample_key_does_not_depend_on_chunk_size() -> None:
    """Global sample 3 is the same stream from one chunk or from a chunk that starts at 3."""
    base = jax.random.PRNGKey(1)

    def produced(chunk_start: int, offset: int) -> jax.Array:
        dropout = _dropout_key(base, "sample", chunk_start)
        return jax.random.fold_in(dropout, chunk_start + offset)

    assert jnp.array_equal(produced(0, 3), produced(3, 0))


def test_score_dropout_key_still_folds_chunk_start() -> None:
    base = jax.random.PRNGKey(1)
    assert not jnp.array_equal(
        _dropout_key(base, "score:ddg", 0),
        _dropout_key(base, "score:ddg", 4),
    )


def test_binding_optimization_does_not_use_a_zero_interface() -> None:
    with pytest.raises(ValueError, match="binding_energy_optimization"):
        sample_binding_tables(
            PottsMPNNOptions(binding_energy_optimization="both"),
            length=4,
            alphabet=22,
            dtype=jnp.float32,
        )
    with pytest.raises(ValueError, match="binding_energy_optimization"):
        sample_binding_tables(
            PottsMPNNOptions(binding_energy_optimization="only"),
            length=4,
            alphabet=22,
            dtype=jnp.float32,
        )
    tables = sample_binding_tables(
        PottsMPNNOptions(binding_energy_optimization="none"),
        length=4,
        alphabet=22,
        dtype=jnp.float32,
    )
    assert not bool(jnp.any(tables.inter_mask))


def test_omit_token_expands_and_unknown_letters_raise() -> None:
    features = _features(5)
    chains = (("A", "ACDEF"),)
    ready = prepare_sample(
        features,
        chains,
        PottsMPNNOptions(),
        _spec(omit_aa=("CW",)),
    )
    assert ready.omit[_ALPHABET.index("C")] == 1.0
    assert ready.omit[_ALPHABET.index("W")] == 1.0
    assert ready.omit[_ALPHABET.index("A")] == 0.0
    with pytest.raises(ValueError, match="omit amino acid"):
        prepare_sample(features, chains, PottsMPNNOptions(), _spec(omit_aa=("Z",)))


def test_per_position_bias_is_not_collapsed_to_row_zero() -> None:
    length = 4
    features = _features(length)
    grid = np.arange(length * len(_ALPHABET), dtype=np.float32).reshape(length, len(_ALPHABET))
    ready = prepare_sample(
        features,
        (("A", "ACDE"),),
        PottsMPNNOptions(),
        _spec(bias=grid),
    )
    assert np.allclose(ready.bias_by_res[:length], grid)
    assert np.all(ready.bias == 0.0)


def test_fixed_positions_do_not_write_through_into_features() -> None:
    features = _features(4)
    before = features.chain_m_pos.copy()
    ready = prepare_sample(
        features,
        (("A", "ACDE"),),
        PottsMPNNOptions(),
        _spec(fixed_positions=np.asarray([1], dtype=np.int32)),
    )
    assert ready.chain_m_pos[1] == 0.0
    assert np.array_equal(features.chain_m_pos, before)


def test_optimize_fasta_joins_wrapped_lines_and_rejects_a_leading_comment(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.fasta"
    wrapped.write_text(">toy\nAC\nDE\n\n>toy:B\nWWWW\n", encoding="utf-8")
    assert load_optimize_fasta(wrapped, "toy") == ("ACDE", "WWWW")
    comment = tmp_path / "comment.fasta"
    comment.write_text("# note\n>toy\nAAAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FASTA"):
        load_optimize_fasta(comment, "toy")


def test_mutant_fasta_keeps_the_record_after_a_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "mut.fasta"
    path.write_text(">prot\nABCD\n\n>prot|0.5\nWXYZ\n", encoding="utf-8")
    found = _from_fasta(path, "prot", (_ChainSeq("A", "ABCD"),))
    assert [item.sequence for item in found] == ["ABCD", "WXYZ"]
    assert found[1].expt == pytest.approx(0.5)


def test_converter_script_is_found_deeper_than_parents_four(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "recapture" / "pottsmpnn_model_to_eqx.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    anchor = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "driver.py"
    anchor.parent.mkdir(parents=True)
    found = _converter_script(anchor)
    assert found == script.resolve()


def test_length_21_vector_is_per_position_bias() -> None:
    bias = jnp.arange(21, dtype=jnp.float32)
    compiled = compile_omit_aa_bias(
        bias,
        fixed_mask_row=None,
        omit_aa=("W",),
        omit_aa_per_position=None,
        seq_len=21,
    )
    # Column 1 is C, not the omitted W. Per-position stores bias[i] on every letter.
    assert float(compiled[0, 1]) == float(bias[0])
    assert float(compiled[3, 1]) == float(bias[3])


def test_alphabet_vector_still_broadcasts_when_the_chain_is_not_length_21() -> None:
    bias = jnp.arange(21, dtype=jnp.float32)
    compiled = compile_omit_aa_bias(
        bias,
        fixed_mask_row=None,
        omit_aa=("W",),
        omit_aa_per_position=None,
        seq_len=4,
    )
    assert float(compiled[0, 1]) == float(bias[1])
    assert float(compiled[2, 1]) == float(bias[1])


def test_hf_probe_does_not_change_the_process_socket_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    socket.setdefaulttimeout(None)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("blocked")

    monkeypatch.setattr(socket, "create_connection", _boom)
    assert _hf_hub_reachable() is False
    assert socket.getdefaulttimeout() is None


def test_dead_potts_options_are_not_declared() -> None:
    names = {field.name for field in dataclasses.fields(PottsMPNNOptions)}
    assert "binding_energy_cutoff" not in names
    assert "filter_nan" not in names
    assert "write_pdb" not in names
