"""_dispatch_axis dispatches via isinstance(strategy, ChunkedMap), forwarding
strategy.batch_size (normalizing batch_size=0 to None) to chunked_map.
"""

import jax.numpy as jnp
import pytest
from xtrax.tiling import ChunkedMap

from aminx.host import kernel_dispatch


class _Unrecognised:
    """A strategy matching no AxisStrategy branch: exercises the fallback path."""

    batch_size = 3


@pytest.fixture
def seen_batch_sizes(monkeypatch):
    seen: list[object] = []

    def _spy(body, xs, batch_size=None):
        seen.append(batch_size)
        return xs

    monkeypatch.setattr(kernel_dispatch, "chunked_map", _spy)
    return seen


def test_chunked_strategies_keep_their_tile(seen_batch_sizes):
    kernel_dispatch._dispatch_axis(ChunkedMap(batch_size=3), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [3]


def test_chunked_map_zero_batch_size_normalizes_to_none(seen_batch_sizes):
    kernel_dispatch._dispatch_axis(ChunkedMap(batch_size=0), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [None]


def test_unrecognised_name_takes_the_unchunked_fallback(seen_batch_sizes):
    """Control: the fallback really does lose the tile, so the test above can fail."""
    kernel_dispatch._dispatch_axis(_Unrecognised(), lambda x: x, jnp.zeros((6,)))
    assert seen_batch_sizes == [None]
