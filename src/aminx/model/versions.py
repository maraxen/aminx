"""Model version constants. This module is import-free to prevent circular dependencies."""

from __future__ import annotations

from typing import Literal

MODEL_WEIGHTS = Literal[
  "original",
  "soluble",
  "ligand",
  "sc",
  "membrane",
  "abmpnn",
  "hypermpnn",
  "cyclicmpnn",
  "ca",
]
MODEL_VERSION = Literal[
  "v_48_002",
  "v_48_010",
  "v_48_020",
  "v_48_030",
  "v_48_v2",
  "v_32_010_25",
  "v_32_002_16",
]
