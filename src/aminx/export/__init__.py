"""Public API for aminx.export (D-A).

Browser/native export wrappers around the P03 (edge-feature) and P04
(unconditional-logit) computations. Wrappers compose existing public functions and do
not reimplement numerics (D-A); the one non-wrapper source change this package's task
makes is extracting ``aminx.model.features.ProteinFeatures.forward_edge_stages``'s k-NN
selection block into the public ``aminx.model.features.select_neighbors``.

Public contract (everything in ``__all__``):
- Length bucketing (D-C): ``EXPORT_BUCKETS``, ``select_export_bucket``, ``pad_inputs``,
  and their refusal errors.
- Wrappers (D-B, D-H): ``make_p03_featurize``, ``make_p04_unconditional``,
  ``make_p07_sample``, ``zero_dropout``, ``PINNED_CHECKPOINT_ID``.
- RNG audit: ``RNG_PRIMITIVES``, ``find_rng_primitives``.
"""

from __future__ import annotations

from .buckets import (
  EXPORT_BUCKETS,
  ExportLengthError,
  LengthAboveMaxBucketError,
  LengthBelowNeighborsError,
  pad_inputs,
  select_export_bucket,
)
from .rng_audit import RNG_PRIMITIVES, find_rng_primitives
from .wrappers import (
  PINNED_CHECKPOINT_ID,
  make_p03_featurize,
  make_p04_unconditional,
  make_p07_sample,
  zero_dropout,
)

__all__ = [
  "EXPORT_BUCKETS",
  "PINNED_CHECKPOINT_ID",
  "RNG_PRIMITIVES",
  "ExportLengthError",
  "LengthAboveMaxBucketError",
  "LengthBelowNeighborsError",
  "find_rng_primitives",
  "make_p03_featurize",
  "make_p04_unconditional",
  "make_p07_sample",
  "pad_inputs",
  "select_export_bucket",
  "zero_dropout",
]
