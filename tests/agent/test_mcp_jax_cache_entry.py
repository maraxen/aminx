"""``aminx.agent.mcp.main`` applies xtrax's persistent JAX cache before serving (debt #3527)."""

from __future__ import annotations

import pytest
from xtrax import jax_cache


pytestmark = pytest.mark.usefixtures("jax_cache_isolated")


def test_main_configures_jax_cache_before_server_run(monkeypatch: pytest.MonkeyPatch) -> None:
  cisternal = pytest.importorskip("cisternal")
  pytest.importorskip("fastmcp")
  import aminx.agent.mcp as mcp  # noqa: PLC0415

  events: list[str] = []

  class _Server:
    def run(self) -> None:
      events.append("run")

  monkeypatch.setattr(cisternal, "init", lambda *a, **k: events.append("cisternal.init"))
  monkeypatch.setattr(jax_cache, "configure", lambda *a, **k: events.append("configure"))
  monkeypatch.setattr(mcp, "build_server", lambda: events.append("build") or _Server())

  mcp.main()

  assert events == ["cisternal.init", "configure", "build", "run"]
