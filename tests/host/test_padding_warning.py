"""The runner warns once when structures are padded to far more than their real length (aminx #2358).

Autoregressive decode cost grows roughly with the square of the padded length, and the default ``max_length`` is 512,
so a short chain pays a large factor with no signal. The warning changes no numbers.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from aminx.host.runner import _PADDING_WARN_RATIO, _PaddingCheck


def _batch(real_lengths: list[int], padded: int) -> SimpleNamespace:
    mask = np.zeros((len(real_lengths), padded), dtype=np.float32)
    for i, n in enumerate(real_lengths):
        mask[i, :n] = 1.0
    return SimpleNamespace(mask=mask)


def test_warns_with_the_numbers_a_user_needs():
    with pytest.warns(UserWarning, match=r"max_length=512.*93 residues.*max_length=96") as rec:
        _PaddingCheck()(_batch([93], 512))
    text = str(rec[0].message)
    assert "30x" in text, "should state the approximate extra work: (512/93)^2 ~ 30"
    assert "#2358" in text


def test_uses_the_longest_chain_in_a_batch():
    with pytest.warns(UserWarning, match=r"has 200 residues.*max_length=224"):
        _PaddingCheck()(_batch([40, 200, 75], 512))


@pytest.mark.parametrize(
    ("real", "padded"),
    [(100, 128), (96, 192), (256, 512), (512, 512), (300, 512)],
    ids=["1.28x", "exactly 2x", "exactly 2x big", "no padding", "1.7x"],
)
def test_silent_at_or_below_the_threshold(real, padded):
    assert padded <= _PADDING_WARN_RATIO * real
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PaddingCheck()(_batch([real], padded))


def test_warns_just_above_the_threshold():
    with pytest.warns(UserWarning):
        _PaddingCheck()(_batch([100], 201))


def test_warns_only_once_per_run():
    check = _PaddingCheck()
    with pytest.warns(UserWarning):
        check(_batch([93], 512))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(_batch([93], 512))
        check(_batch([50], 512))


def test_a_new_run_warns_again():
    for _ in range(2):
        with pytest.warns(UserWarning):
            _PaddingCheck()(_batch([93], 512))


def test_never_raises_on_unexpected_batches():
    check = _PaddingCheck()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(SimpleNamespace())  # no mask attribute
        check(SimpleNamespace(mask=None))
        check(SimpleNamespace(mask=np.ones((5,), dtype=np.float32)))  # not (batch, length)
        check(_batch([0], 64))  # an empty structure has no real length to compare against


def test_suggestion_rounds_up_to_a_multiple_of_32_and_never_below_the_chain():
    for real in (1, 31, 32, 33, 93, 214, 250):
        with pytest.warns(UserWarning) as rec:
            _PaddingCheck()(_batch([real], 1024))
        suggested = int(str(rec[0].message).split("(for example max_length=")[1].split(")")[0])
        assert suggested >= real
        assert suggested % 32 == 0
        assert suggested - real < 32


# --- end-to-end through the real runner ------------------------------------------------------------------------

_STRUCTURE = "tests/data/1ubq.pdb"  # 76 residues
_CHECKPOINT = "proteinmpnn_v_48_020"


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


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_sample_warns_when_padding_dominates():
    with pytest.warns(UserWarning, match=r"max_length=256.*76 residues.*max_length=96"):
        _sample(256)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_sample_is_silent_with_modest_padding():
    """Control: 128 / 76 = 1.7x is below the threshold, so the same call must not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sample(128)
    assert not [w for w in caught if "#2358" in str(w.message)], "padding warning fired at 1.7x padding"


_UBQ_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # 76 residues


def _score(max_length: int) -> None:
    from aminx.host.runner import score

    score(
        inputs=[_STRUCTURE],
        checkpoint_id=_CHECKPOINT,
        sequences_to_score=[_UBQ_SEQUENCE],
        random_seed=0,
        max_length=max_length,
    )


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_score_warns_when_padding_dominates():
    assert len(_UBQ_SEQUENCE) == 76
    with pytest.warns(UserWarning, match=r"max_length=256.*76 residues.*max_length=96"):
        _score(256)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_runner_score_is_silent_with_modest_padding():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _score(128)
    assert not [w for w in caught if "#2358" in str(w.message)], "padding warning fired at 1.7x padding"
