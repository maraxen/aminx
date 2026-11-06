"""Unit tests for aminx domain plugin (P5-04).

Verifies:
1. DomainPlugin protocol conformance and metadata.
2. Sequence token decoding and MPNN alphabet mapping.
3. Derived NLL fallback calculation from logits and perplexity.
4. Designable masking (real_mask & ~fixed_mask) excluding padding.
5. Diversity, positional Shannon entropy, and composition entropy.
6. Target native recovery, shuffled baseline, and degeneracy margin guard.
7. Zero-copy array immutability and metric schema validation.
8. Entry point declaration in pyproject.toml and callable target resolution.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import tomllib
import numpy as np
import pytest

from aminx.eda.plugin import (
    AminxDomainPlugin,
    DomainMetricResult,
    DomainPlugin,
    LibraryCell,
    MPNN_ALPHABET,
    ResidueTable,
    _compute_derived_nll,
    _decode_tokens_to_seqs,
    _encode_seqs_to_tokens,
)


def test_plugin_metadata():
    plugin = AminxDomainPlugin()
    assert plugin.domain_name == "aminx"
    assert isinstance(plugin.version, str)
    assert isinstance(plugin, DomainPlugin)


def test_alphabet_mapping_and_token_encoding():
    # MPNN alphabet has 21 characters
    assert len(MPNN_ALPHABET) == 21
    assert MPNN_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"

    test_seqs = ("ACDEF", "GHIKL", "MNPQR")
    tokens = _encode_seqs_to_tokens(test_seqs, max_len=5)
    assert tokens.shape == (3, 5)
    assert np.array_equal(tokens[0], [0, 1, 2, 3, 4])  # A, C, D, E, F

    decoded = _decode_tokens_to_seqs(tokens)
    assert decoded == test_seqs


def test_derived_nll_fallback_and_perplexity():
    # 2 samples, length 3, 21 tokens
    rng = np.random.default_rng(42)
    tokens = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    logits = rng.normal(size=(2, 3, 21)).astype(np.float32)
    mask = np.array([True, True, False])

    nll = _compute_derived_nll(tokens, logits, mask)
    assert nll.shape == (2,)
    assert np.all(nll >= 0.0)

    # Manual verification for sample 0 at position 0
    # LSE - logit[tokens[0, 0]]
    z0 = logits[0, 0]
    lse0 = np.log(np.sum(np.exp(z0 - np.max(z0)))) + np.max(z0)
    expected_pos0 = lse0 - z0[tokens[0, 0]]

    z1 = logits[0, 1]
    lse1 = np.log(np.sum(np.exp(z1 - np.max(z1)))) + np.max(z1)
    expected_pos1 = lse1 - z1[tokens[0, 1]]

    expected_sample0 = (expected_pos0 + expected_pos1) / 2.0
    assert np.isclose(nll[0], expected_sample0, rtol=1e-5)


def test_designable_mask_application():
    plugin = AminxDomainPlugin()
    seq_len = 10
    n_samples = 4

    tokens = np.zeros((n_samples, seq_len), dtype=np.int64)
    # Position 2 is fixed
    fixed_mask = np.zeros(seq_len, dtype=bool)
    fixed_mask[2] = True

    sink_data = {
        "sequences": tokens,
        "fixed_mask": fixed_mask,
    }
    manifest_row = {
        "manifest_row_hash": "test_hash_001",
        "seed": 1001,
        "target_input": "",
        "parameters": {"temperature": 0.3},
    }

    # Simulate extraction with custom sink data
    cells = plugin.extract_cell(sink_data, manifest_row)
    assert len(cells) == 1
    cell = cells[0]

    # Verify mask: real & ~fixed
    assert cell.mask[2] == False  # fixed position must be False
    assert cell.mask[0] == True
    assert cell.tokens.flags.writeable == False
    assert cell.mask.flags.writeable == False


def test_compute_metrics_diversity_and_entropy():
    plugin = AminxDomainPlugin()

    # Case 1: Diversity with N > 1
    tokens = np.array([
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [6, 7, 8, 9],
    ], dtype=np.int64)
    mask = np.array([True, True, True, True])
    residues = ResidueTable(
        native_tokens="ACDE",
        chain_ids=("A", "A", "A", "A"),
        res_numbers=(1, 2, 3, 4),
        insertion_codes=("", "", "", ""),
        real_mask=mask,
        parser_name="test",
        parser_version="1.0",
    )
    cell = LibraryCell(
        knobs={"temperature": 0.5},
        seed=123,
        target="test_target",
        tokens=tokens,
        sequences=_decode_tokens_to_seqs(tokens),
        logits=None,
        scores=np.array([1.2, 1.4, 1.8]),
        is_score_derived=False,
        residues=residues,
        mask=mask,
    )

    result = plugin.compute_metrics(cell)
    assert isinstance(result, DomainMetricResult)
    result.validate_schema()

    assert "diversity" in result.metrics
    assert 0.0 <= result.metrics["diversity"] <= 1.0
    assert result.metrics["positional_entropy"] >= 0.0
    assert result.metrics["composition_entropy"] >= 0.0
    assert "single_sample" not in result.flags

    # Case 2: Single sample boundary guard (BLK-04)
    cell_single = LibraryCell(
        knobs={"temperature": 0.5},
        seed=123,
        target="test_target",
        tokens=tokens[:1],
        sequences=_decode_tokens_to_seqs(tokens[:1]),
        logits=None,
        scores=np.array([1.2]),
        is_score_derived=False,
        residues=residues,
        mask=mask,
    )
    res_single = plugin.compute_metrics(cell_single)
    assert res_single.metrics["diversity"] == 0.0
    assert res_single.flags["single_sample"] == True


def test_native_recovery_and_degeneracy_margin():
    plugin = AminxDomainPlugin()
    mask = np.array([True, True, True, True])

    # Sequence identical to native -> 100% recovery
    tokens_perfect = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ], dtype=np.int64)  # "ACDE"

    residues = ResidueTable(
        native_tokens="ACDE",  # 0, 1, 2, 3
        chain_ids=("A", "A", "A", "A"),
        res_numbers=(1, 2, 3, 4),
        insertion_codes=("", "", "", ""),
        real_mask=mask,
        parser_name="test",
        parser_version="1.0",
    )

    cell = LibraryCell(
        knobs={"temperature": 0.1},
        seed=42,
        target="perfect",
        tokens=tokens_perfect,
        sequences=_decode_tokens_to_seqs(tokens_perfect),
        logits=None,
        scores=np.array([0.5, 0.5]),
        is_score_derived=False,
        residues=residues,
        mask=mask,
    )
    res = plugin.compute_metrics(cell)
    assert res.metrics["recovery"] == 1.0
    assert res.metrics["recovery_margin"] > 0.10
    assert res.flags["degenerate"] == False

    # Low recovery random sequences -> near chance
    rng = np.random.default_rng(99)
    tokens_random = rng.integers(0, 20, size=(10, 20), dtype=np.int64)
    mask_long = np.ones(20, dtype=bool)
    residues_long = ResidueTable(
        native_tokens="A" * 20,
        chain_ids=tuple(["A"] * 20),
        res_numbers=tuple(range(1, 21)),
        insertion_codes=tuple([""] * 20),
        real_mask=mask_long,
        parser_name="test",
        parser_version="1.0",
    )
    cell_rand = LibraryCell(
        knobs={"temperature": 1.0},
        seed=42,
        target="random",
        tokens=tokens_random,
        sequences=_decode_tokens_to_seqs(tokens_random),
        logits=None,
        scores=np.ones(10),
        is_score_derived=False,
        residues=residues_long,
        mask=mask_long,
    )
    res_rand = plugin.compute_metrics(cell_rand)
    assert res_rand.flags["degenerate"] == True


def test_schema_validation_catches_non_finite():
    bad_result = DomainMetricResult(
        metrics={"bad_nan": float("nan"), "valid": 1.0},
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="must be strictly finite"):
        bad_result.validate_schema()

    bad_inf = DomainMetricResult(
        metrics={"bad_inf": float("inf")},
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="must be strictly finite"):
        bad_inf.validate_schema()


def test_entry_point_registration_and_loading():
    # 1. Verify pyproject.toml declaration
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            cfg = tomllib.load(f)
        ep_group = cfg.get("project", {}).get("entry-points", {}).get("xtrax.eda.domains", {})
        assert "aminx" in ep_group
        assert ep_group["aminx"] == "aminx.eda.plugin:AminxDomainPlugin"

    # 2. Verify target is resolvable and callable
    mod_name, cls_name = "aminx.eda.plugin:AminxDomainPlugin".split(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    inst = cls()
    assert isinstance(inst, DomainPlugin)
    assert inst.domain_name == "aminx"
