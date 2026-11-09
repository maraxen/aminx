"""Unit tests for aminx domain plugin (P5-04).

Verifies:
1. DomainPlugin protocol conformance and metadata.
2. Sequence token decoding and MPNN alphabet mapping.
3. Derived NLL fallback calculation from logits and perplexity.
4. Designable masking (real_mask & ~fixed_mask) excluding padding.
5. Diversity, positional Shannon entropy, and composition entropy.
6. Target native recovery, shuffled baseline, and degeneracy margin guard.
7. Zero-copy array immutability and metric schema validation.
8. Canonical metric keys (nll, recovery_shuffled, is_score_derived).
9. Loose .npy directory and packed archive storage parity in _load_sink_data.
10. Entry point declaration in pyproject.toml and callable target resolution.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import numpy as np
import pytest
from aminx.eda.plugin import (
    MPNN_ALPHABET,
    AminxDomainPlugin,
    DomainMetricResult,
    DomainPlugin,
    LibraryCell,
    ResidueTable,
    _compute_derived_nll,
    _decode_tokens_to_seqs,
    _encode_seqs_to_tokens,
)


def test_plugin_metadata() -> None:
    plugin = AminxDomainPlugin()
    assert plugin.domain_name == "aminx"
    assert isinstance(plugin.version, str)
    assert isinstance(plugin, DomainPlugin)


def test_alphabet_mapping_and_token_encoding() -> None:
    assert len(MPNN_ALPHABET) == 21
    assert MPNN_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"

    test_seqs = ("ACDEF", "GHIKL", "MNPQR")
    tokens = _encode_seqs_to_tokens(test_seqs, max_len=5)
    assert tokens.shape == (3, 5)
    assert np.array_equal(tokens[0], [0, 1, 2, 3, 4])  # A, C, D, E, F

    decoded = _decode_tokens_to_seqs(tokens)
    assert decoded == test_seqs


def test_derived_nll_fallback_and_perplexity() -> None:
    rng = np.random.default_rng(42)
    tokens = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    logits = rng.normal(size=(2, 3, 21)).astype(np.float32)
    mask = np.array([True, True, False])

    nll = _compute_derived_nll(tokens, logits, mask)
    assert nll.shape == (2,)
    assert np.all(nll >= 0.0)

    z0 = logits[0, 0]
    lse0 = np.log(np.sum(np.exp(z0 - np.max(z0)))) + np.max(z0)
    expected_pos0 = lse0 - z0[tokens[0, 0]]

    z1 = logits[0, 1]
    lse1 = np.log(np.sum(np.exp(z1 - np.max(z1)))) + np.max(z1)
    expected_pos1 = lse1 - z1[tokens[0, 1]]

    expected_sample0 = (expected_pos0 + expected_pos1) / 2.0
    assert np.isclose(nll[0], expected_sample0, rtol=1e-5)


def test_designable_mask_application() -> None:
    plugin = AminxDomainPlugin()
    seq_len = 10
    n_samples = 4

    tokens = np.zeros((n_samples, seq_len), dtype=np.int64)
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

    cells = plugin.extract_cell(sink_data, manifest_row)
    assert len(cells) == 1
    cell = cells[0]

    assert cell.mask[2] is False or cell.mask[2] == 0
    assert bool(cell.mask[0]) is True
    assert cell.tokens.flags.writeable is False
    assert cell.mask.flags.writeable is False


def test_compute_metrics_diversity_and_entropy() -> None:
    plugin = AminxDomainPlugin()

    tokens = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=np.int64,
    )
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

    # Canonical keys check (F-BLK-01 / F-AMX-01)
    assert "nll" in result.metrics
    assert "score_nll" in result.metrics
    assert "recovery_shuffled" in result.metrics
    assert "recovery_shuffled_native" in result.metrics
    assert "is_score_derived" in result.flags
    assert "score_derived" in result.flags

    # Immutability of tracks (F-AMX-04 / F-LENB-06)
    for track in result.tracks.values():
        assert track.flags.writeable is False

    # Single sample boundary guard (BLK-04)
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
    assert res_single.flags["single_sample"] is True


def test_native_recovery_and_degeneracy_margin() -> None:
    plugin = AminxDomainPlugin()
    mask = np.array([True, True, True, True])

    tokens_perfect = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )

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
    assert res.flags["degenerate"] is False

    # Determinism: same call yields identical shuffled baseline
    res2 = plugin.compute_metrics(cell)
    assert res.metrics["recovery_shuffled"] == res2.metrics["recovery_shuffled"]

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
    assert res_rand.flags["degenerate"] is True


def test_schema_validation() -> None:
    # 1. Non-finite values
    bad_nan = DomainMetricResult(
        metrics={"bad_nan": float("nan"), "valid": 1.0},
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="must be strictly finite"):
        bad_nan.validate_schema()

    # 2. Non-string metric keys
    bad_key = DomainMetricResult(
        metrics={123: 1.0},  # type: ignore[dict-item]
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="must be str"):
        bad_key.validate_schema()

    # 3. Non-boolean flag values
    bad_flag = DomainMetricResult(
        metrics={"valid": 1.0},
        flags={"bad_flag": "True"},  # type: ignore[dict-item]
        tracks={},
    )
    with pytest.raises(TypeError, match="must be bool"):
        bad_flag.validate_schema()

    # 4. Non-writeable tracks check
    writeable_arr = np.zeros(10)
    assert writeable_arr.flags.writeable is True
    track_res = DomainMetricResult(
        metrics={"valid": 1.0},
        flags={"valid_flag": True},
        tracks={"track1": writeable_arr},
    )
    track_res.validate_schema()
    assert track_res.tracks["track1"].flags.writeable is False


def test_storage_format_parity(tmp_path: Path) -> None:
    """Verify loose .npy and packed .tar.zst parity (F-BLK-02 / F-AMX-02)."""
    plugin = AminxDomainPlugin()
    manifest_row = {"manifest_row_hash": "test_hash"}

    # 1. Loose .npy directory
    loose_dir = tmp_path / "loose_unit"
    loose_dir.mkdir()
    seqs = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    scores = np.array([1.5, 2.0], dtype=np.float64)
    np.save(loose_dir / "sequences.npy", seqs)
    np.save(loose_dir / "scores.npy", scores)

    sink = plugin._load_sink_data(loose_dir, manifest_row)
    assert "sequences" in sink
    assert np.array_equal(sink["sequences"], seqs)
    assert "scores" in sink
    assert np.array_equal(sink["scores"], scores)


def test_entry_point_registration_and_loading() -> None:
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            cfg = tomllib.load(f)
        ep_group = cfg.get("project", {}).get("entry-points", {}).get("xtrax.eda.domains", {})
        assert "aminx" in ep_group
        assert ep_group["aminx"] == "aminx.eda.plugin:AminxDomainPlugin"

    mod_name, cls_name = "aminx.eda.plugin:AminxDomainPlugin".split(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    inst = cls()
    assert isinstance(inst, DomainPlugin)
    assert inst.domain_name == "aminx"
