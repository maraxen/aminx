"""The runner warns once when structures are padded to far more than their chain's span (aminx debt #2358).

Autoregressive decode cost grows with the padded length, and the default ``max_length`` is 512, so a short chain pays a
large factor with no signal. The warning changes no numbers. Its suggestion must never be below the chain's SPAN (index
of the last valid residue + 1), because structures with unresolved residues have gaps in the mask and a ``max_length``
below the span makes the loader crop at random or raise (measured on 1BC8: 93 valid residues, span 113).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aminx.host.runner import _PADDING_WARN_RATIO, _PaddingCheck


def _batch(real_lengths: list[int], padded: int) -> SimpleNamespace:
    """Contiguous masks: row i has real_lengths[i] valid residues from index 0."""
    mask = np.zeros((len(real_lengths), padded), dtype=np.float32)
    for i, n in enumerate(real_lengths):
        mask[i, :n] = 1.0
    return SimpleNamespace(mask=mask)


def _gap_batch(valid_runs: list[tuple[int, int]], padded: int) -> SimpleNamespace:
    """One row whose valid residues are the half-open index runs given, i.e. a mask with unresolved gaps."""
    mask = np.zeros((1, padded), dtype=np.float32)
    for start, stop in valid_runs:
        mask[0, start:stop] = 1.0
    return SimpleNamespace(mask=mask)


def _suggested(message: str) -> int:
    return int(re.search(r"for example max_length=(\d+)", message).group(1))


# --- the check itself ------------------------------------------------------------------------------------------


def test_warns_with_the_numbers_a_user_needs():
    with pytest.warns(UserWarning, match=r"max_length=512.*spans 93 residues.*max_length=96") as rec:
        _PaddingCheck("sample")(_batch([93], 512))
    text = str(rec[0].message)
    assert "30x" in text, "should state the approximate extra work: (512/93)^2 ~ 30"
    assert "#2358" not in text, "internal tracker ids are not for end users"


def test_uses_the_longest_chain_in_a_batch():
    with pytest.warns(UserWarning, match=r"spans 200 residues.*max_length=224"):
        _PaddingCheck("sample")(_batch([40, 200, 75], 512))


def test_gaps_in_the_mask_count_toward_the_span_not_just_the_valid_residues():
    """1BC8-like: 93 valid residues but the last valid index is 112 (span 113). Suggesting 96 would crop at random."""
    runs = [(0, 40), (50, 80), (90, 113)]  # 40 + 30 + 23 = 93 valid, span 113
    batch = _gap_batch(runs, 512)
    assert int(batch.mask.sum()) == 93
    with pytest.warns(UserWarning, match=r"spans 113 residues") as rec:
        _PaddingCheck("sample")(batch)
    suggested = _suggested(str(rec[0].message))
    assert suggested >= 113, f"suggestion {suggested} is below the span and would crop the structure"
    assert suggested == 128


def test_message_tells_the_user_about_truncation_and_the_whole_input_set():
    with pytest.warns(UserWarning) as rec:
        _PaddingCheck("sample")(_batch([93], 512))
    text = str(rec[0].message)
    assert "ALL your inputs" in text
    assert "truncation_strategy" in text


def test_sample_and_score_messages_differ_where_their_physics_differs():
    with pytest.warns(UserWarning) as sampled:
        _PaddingCheck("sample")(_batch([93], 512))
    with pytest.warns(UserWarning) as scored:
        _PaddingCheck("score")(_batch([93], 512))
    s, c = str(sampled[0].message), str(scored[0].message)
    assert "square of the padded length" in s and "same seed" in s
    assert "single forward pass" in c and "do not depend on max_length" in c
    assert "same seed" not in c and "square" not in c, "scoring must not be told sampling's cost model"


@pytest.mark.parametrize(
    ("real", "padded"),
    [(100, 128), (96, 192), (256, 512), (512, 512), (300, 512)],
    ids=["1.28x", "exactly 2x", "exactly 2x big", "no padding", "1.7x"],
)
def test_silent_at_or_below_the_threshold(real, padded):
    assert padded <= _PADDING_WARN_RATIO * real
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PaddingCheck("sample")(_batch([real], padded))


def test_warns_just_above_the_threshold():
    with pytest.warns(UserWarning):
        _PaddingCheck("sample")(_batch([100], 201))


def test_never_suggests_more_than_the_current_padding():
    """A 9-residue peptide at max_length=20: 20 > 2*9 but the 32-multiple would be MORE padding, so say nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PaddingCheck("sample")(_batch([9], 20))


def test_warns_only_once_per_run():
    check = _PaddingCheck("sample")
    with pytest.warns(UserWarning):
        check(_batch([93], 512))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(_batch([93], 512))
        check(_batch([50], 512))


def test_a_new_run_warns_again():
    for _ in range(2):
        with pytest.warns(UserWarning):
            _PaddingCheck("sample")(_batch([93], 512))


def test_warning_is_attributed_to_the_callers_code():
    with pytest.warns(UserWarning) as rec:
        _PaddingCheck("sample")(_batch([93], 512))
    assert rec[0].filename == __file__, "the warning should point at the code that called into aminx"


def test_never_raises_from_its_own_computation():
    check = _PaddingCheck("sample")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(SimpleNamespace())  # no mask attribute
        check(SimpleNamespace(mask=None))
        check(SimpleNamespace(mask=np.ones((5,), dtype=np.float32)))  # not (batch, length)
        check(SimpleNamespace(mask=np.zeros((0, 64), dtype=np.float32)))  # zero rows
        check(SimpleNamespace(mask=np.zeros((2, 0), dtype=np.float32)))  # zero length
        check(SimpleNamespace(mask=np.zeros((1, 64), dtype=np.float32)))  # nothing valid: no real length to compare
        check(SimpleNamespace(mask=np.array([["a", "b"]], dtype=object)))  # non-numeric


def test_suggestion_rounds_up_to_a_multiple_of_32_and_never_below_the_span():
    for real in (1, 31, 32, 33, 93, 214, 250):
        padded = 1024
        with pytest.warns(UserWarning) as rec:
            _PaddingCheck("sample")(_batch([real], padded))
        suggested = _suggested(str(rec[0].message))
        if real < 32 and suggested >= padded:
            continue
        assert suggested >= real
        assert suggested % 32 == 0
        assert suggested - real < 32


# --- end-to-end through the real runner ------------------------------------------------------------------------

_STRUCTURE = str(Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb")  # 76 residues, contiguous
_CHECKPOINT = "proteinmpnn_v_48_020"
_UBQ_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # 76 residues


def _sample(max_length: int) -> None:
    from aminx.host.runner import sample

    sample(
        inputs=[_STRUCTURE],
        checkpoint_id=_CHECKPOINT,
        num_samples=1,
        temperature=0.5,
        random_seed=0,
        max_length=max_length,
    )


def _score(max_length: int) -> None:
    from aminx.host.runner import score

    score(
        inputs=[_STRUCTURE],
        checkpoint_id=_CHECKPOINT,
        sequences_to_score=[_UBQ_SEQUENCE],
        random_seed=0,
        max_length=max_length,
    )


def _padding_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if "padded to max_length" in str(w.message)]


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_sample_warns_when_padding_dominates():
    with pytest.warns(UserWarning, match=r"max_length=256.*spans 76 residues.*max_length=96"):
        _sample(256)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_sample_is_silent_with_modest_padding():
    """Control: 128 / 76 = 1.7x is below the threshold, so the same call must not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sample(128)
    assert not _padding_warnings(caught), "padding warning fired at 1.7x padding"


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_score_warns_when_padding_dominates():
    assert len(_UBQ_SEQUENCE) == 76
    with pytest.warns(UserWarning, match=r"max_length=256.*spans 76 residues.*single forward pass"):
        _score(256)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_score_is_silent_with_modest_padding():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _score(128)
    assert not _padding_warnings(caught), "padding warning fired at 1.7x padding"
