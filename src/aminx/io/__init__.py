"""Initialization for the io module, providing functions to parse and load protein structures."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from aminx.io.campaign_sink import (
    CAMPAIGN_METRICS_COLUMNS,
    CampaignSinkInspector,
    CampaignSinkSummary,
    UnifiedCampaignSink,
    inspect_campaign_sink,
  )
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


def _lazy_import(mod_name: str) -> Any:  # noqa: ANN401
  return getattr(importlib, "import_module")(mod_name)  # noqa: B009


def __getattr__(name: str) -> Any:  # noqa: ANN401
  if name in (
    "CAMPAIGN_METRICS_COLUMNS",
    "CampaignSinkInspector",
    "CampaignSinkSummary",
    "UnifiedCampaignSink",
    "inspect_campaign_sink",
  ):
    return getattr(_lazy_import("aminx.io.campaign_sink"), name)
  if name in (
    "DesignMetadata",
    "DesignPayload",
    "DesignZarrWriter",
  ):
    return getattr(_lazy_import("aminx.io.designs"), name)
  msg = f"module {__name__!r} has no attribute {name!r}"
  raise AttributeError(msg)


def __dir__() -> list[str]:
  return __all__
