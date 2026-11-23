"""``aminx.host.campaign.main`` applies xtrax's persistent JAX cache before dispatch (debt #3527)."""

from __future__ import annotations

import pytest
from xtrax import jax_cache

import aminx.host.campaign as campaign


pytestmark = pytest.mark.usefixtures("jax_cache_isolated")


def test_main_configures_jax_cache_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
  events: list[str] = []
  monkeypatch.setattr(jax_cache, "configure", lambda *a, **k: events.append("configure"))
  monkeypatch.setattr(
    campaign, "_handle_gates_command", lambda args: events.append("gates") or 0
  )

  campaign.main(["gates", "--manifest-path", "unused.json"])

  assert events == ["configure", "gates"]
