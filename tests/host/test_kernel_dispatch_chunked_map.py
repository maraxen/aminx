"""_dispatch_axis dispatches on type(strategy).__name__, so xtrax 0.4.0a11's rename of
SafeMap to ChunkedMap (xtrax #3644) must not drop xtrax plans to the unchunked fallback.

xtrax's deprecated ``SafeMap`` alias IS ``ChunkedMap``, so every xtrax plan now reports
the name "ChunkedMap". Matching only "SafeMap" sent it to the fallback branch, which
calls ``_safe_map(..., batch_size=batch_size_fallback)`` (default 0 = no chunking) --
silently losing the memory bound with no error and no warning.
"""

import jax.numpy as jnp
import pytest
from xtrax.tiling import ChunkedMap

from aminx.host import kernel_dispatch
from aminx.tiling.strategy import SafeMap as AminxSafeMap


class _Unrecognised:
    """A strategy whose name matches no branch: exercises the fallback path."""

    batch_size = 3


@pytest.fixture
def seen_batch_sizes(monkeypatch):
    seen: list[object] = []

    def _spy(body, xs, batch_size=None):
        seen.append(batch_size)
        return xs

    monkeypatch.setattr(kernel_dispatch, "_safe_map", _spy)
    return seen


@pytest.mark.parametrize(
    "strategy",
    [ChunkedMap(batch_size=3), AminxSafeMap(tile=3)],
    ids=["xtrax-ChunkedMap", "aminx-SafeMap"],
)
def test_chunked_strategies_keep_their_tile(strategy, seen_batch_sizes):
    kernel_dispatch._dispatch_axis(strategy, lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [3]


def test_unrecognised_name_takes_the_unchunked_fallback(seen_batch_sizes):
    """Control: the fallback really does lose the tile, so the test above can fail."""
    kernel_dispatch._dispatch_axis(_Unrecognised(), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [0]
