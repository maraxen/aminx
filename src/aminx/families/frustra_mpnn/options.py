"""Options for FrustraMPNN frustration index scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_CHECKPOINT_ID = "megascale_default_la-true_esm-False_split1"


@dataclass(frozen=True)
class FrustraMPNNOptions:
  """Options for FrustraMPNN local energetic frustration index scoring.

  Attributes
  ----------
  checkpoint_id : str | None
      Checkpoint identifier (default: "megascale_default_la-true_esm-False_split1").
  chains : str | tuple[str, ...] | None
      Target chains to evaluate (e.g. "A" or ("A", "B")). None evaluates all chains.
  positions : str | tuple[int, ...] | None
      Target residue positions (e.g. "1,2,3" or (1, 2, 3)). None evaluates all positions.
  subtract_mut : bool
      Whether to subtract the wild-type residue prediction score (default: False).
  dtype : Literal["float32", "float64"]
      Floating-point compute precision: "float32" (default) or "float64".
  """

  checkpoint_id: str | None = DEFAULT_CHECKPOINT_ID
  chains: str | tuple[str, ...] | None = None
  positions: str | tuple[int, ...] | None = None
  subtract_mut: bool = False
  dtype: Literal["float32", "float64"] = "float32"

  def __post_init__(self) -> None:
    """Validate configuration parameters."""
    if self.dtype not in ("float32", "float64"):
      msg = f"Unknown FrustraMPNN dtype {self.dtype!r}. Use 'float32' or 'float64'."
      raise ValueError(msg)
