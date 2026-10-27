"""Initialization for the io module, providing functions to parse and load protein structures."""

from __future__ import annotations

from typing import Any

from aminx.io.designs import DesignMetadata, DesignPayload, DesignZarrWriter

__all__ = [
  "CAMPAIGN_METRICS_COLUMNS",
  "CampaignSinkInspector",
  "CampaignSinkSummary",
  "DesignMetadata",
  "DesignPayload",
  "DesignZarrWriter",
  "UnifiedCampaignSink",
  "inspect_campaign_sink",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
  if name in (
    "CAMPAIGN_METRICS_COLUMNS",
    "CampaignSinkInspector",
    "CampaignSinkSummary",
    "UnifiedCampaignSink",
    "inspect_campaign_sink",
  ):
    from aminx.io import campaign_sink  # noqa: PLC0415

    return getattr(campaign_sink, name)
  msg = f"module {__name__!r} has no attribute {name!r}"
  raise AttributeError(msg)
