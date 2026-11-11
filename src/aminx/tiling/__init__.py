"""aminx.tiling — axis strategy vocabulary (planning/carry/dedup declaration moved to xtrax.tiling, EPIC #1541)."""

from aminx.tiling.strategy import (
  AxisStrategy,
  ChunkedMap,
  DedupGather,
  Scan,
  Vmap,
)
from aminx.types.tiling_protocols import DedupFn, GatherFn, ScanTransition

__all__ = [
  "AxisStrategy",
  "ChunkedMap",
  "DedupFn",
  "DedupGather",
  "GatherFn",
  "Scan",
  "ScanTransition",
  "Vmap",
]
