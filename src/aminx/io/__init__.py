"""Consolidated I/O package for aminx."""

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
  from aminx.io.designs import (
    DesignMetadata,
    DesignPayload,
    DesignZarrWriter,
  )

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

_LAZY_SUBMODULES: dict[str, str] = {
  "CAMPAIGN_METRICS_COLUMNS": "aminx.io.campaign_sink",
  "CampaignSinkInspector": "aminx.io.campaign_sink",
  "CampaignSinkSummary": "aminx.io.campaign_sink",
  "UnifiedCampaignSink": "aminx.io.campaign_sink",
  "inspect_campaign_sink": "aminx.io.campaign_sink",
  "DesignMetadata": "aminx.io.designs",
  "DesignPayload": "aminx.io.designs",
  "DesignZarrWriter": "aminx.io.designs",
}


def __getattr__(name: str) -> Any:
  """Lazily resolve public symbols from ``aminx.io`` submodules.

  Parameters
  ----------
  name : str
      Attribute name to resolve.

  Returns
  -------
  Any
      Resolved module attribute.
  """
  mod_name = _LAZY_SUBMODULES.get(name)
  if mod_name is not None:
    module = importlib.import_module(mod_name)
    val = getattr(module, name)
    globals()[name] = val
    return val
  msg = f"module {__name__!r} has no attribute {name!r}"
  raise AttributeError(msg)


def __dir__() -> list[str]:
  """Return available module attributes including lazily exported symbols.

  Parameters
  ----------
  self : None
      Module scope.

  Returns
  -------
  list[str]
      Sorted attribute names.
  """
  return sorted({*globals().keys(), *__all__})
