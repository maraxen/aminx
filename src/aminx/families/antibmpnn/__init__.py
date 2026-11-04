"""AntiBMPNN family: design post-processing (descriptor scoring of a designed segment).

AntiBMPNN has no family driver. The sampling CLI is ProteinMPNN's flag set and the checkpoints
(``antibmpnn_v_48_000`` ...) are ordinary ProteinMPNN-architecture weights, so sampling and
scoring go through the generic ``aminx.host.runner.sample`` / ``score`` with
``checkpoint_id="antibmpnn_v_48_000"``. What is AntiBMPNN-specific upstream is
``helper_scripts/parsing_design_result.py``: slice each design at ``--start``/``--end`` and report the
``peptides`` physical descriptors PD1/PD2 and the Lehninger net charge at pH 7.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from aminx.model import cdr_bias
from aminx.run.options import AntiBMPNNOptions

if TYPE_CHECKING:
  from collections.abc import Sequence

#: Net charge is reported at this pH (parsing_design_result.py:51 ``peptide.charge(pH=7.4)``).
DESCRIPTOR_PH = 7.4
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class DesignDescriptors(NamedTuple):
  """Per-design segment and its PD1 / PD2 / net-charge values."""

  segment: str
  pd1_volume: float
  pd2_hydro: float
  pd3_charge: float


def design_segment(sequence: str, options: AntiBMPNNOptions) -> str:
  """Slice ``sequence`` like upstream ``seq[start-1:end]`` (1-based, inclusive ``descriptor_end``)."""
  start = 0 if options.descriptor_start is None else options.descriptor_start - 1
  if start < 0:
    msg = f"descriptor_start must be >= 1, got {options.descriptor_start}"
    raise ValueError(msg)
  end = len(sequence) if options.descriptor_end is None else options.descriptor_end
  return sequence[start:end]


def design_descriptors(sequences: Sequence[str], options: AntiBMPNNOptions | None = None) -> list[DesignDescriptors]:
  """PD1, PD2 and Lehninger charge (pH 7.4) of each design's selected segment."""
  options = options or AntiBMPNNOptions()
  out: list[DesignDescriptors] = []
  for sequence in sequences:
    segment = design_segment(sequence, options)
    tokens = jnp.asarray([_ALPHABET.index(letter) for letter in segment], dtype=jnp.int32)
    got = cdr_bias.compute_cdr_physicochemical_descriptors(tokens, ph=DESCRIPTOR_PH)
    out.append(DesignDescriptors(segment, float(got.pd1_volume), float(got.pd2_hydro), float(got.pd3_charge)))
  return out


__all__ = ["DESCRIPTOR_PH", "DesignDescriptors", "design_descriptors", "design_segment"]
