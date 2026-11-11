"""_dispatch_axis dispatches on type(strategy).__name__, so ChunkedMap
must use chunked_map with the strategy's batch_size.
"""

import jax.numpy as jnp
import pytest
from xtrax.tiling import ChunkedMap

from aminx.host import kernel_dispatch


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


def test_chunked_strategies_keep_their_tile(seen_batch_sizes):
    kernel_dispatch._dispatch_axis(ChunkedMap(batch_size=3), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [3]


def test_unrecognised_name_takes_the_unchunked_fallback(seen_batch_sizes):
    """Control: the fallback really does lose the tile, so the test above can fail."""
    kernel_dispatch._dispatch_axis(_Unrecognised(), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [None]
