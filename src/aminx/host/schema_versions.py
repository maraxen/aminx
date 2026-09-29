"""Schema version constants for all host output types.

This module is the single source of truth for all schema version labels.
All other modules import from here to ensure consistency.
"""

from __future__ import annotations

from typing import Final

# Grid-mode sampling schema version. Feeds manifest_row_hash -> output path -> resume markers.
# NEVER feeds PRNG seed -- see _SEED_HASH_SCHEMA_PIN in _sampling_grid_lineage.py for details.
#
# REVERTED to "grid_v1" (audit finding A, task_id `260910_aminx-sink-provenance-schema`,
# code-review round on PR #154): bumping this changes every future row's output directory,
# which silently triggers a full campaign recompute on resume. The new fields added in that
# task are purely additive; use SinkProvenanceVersion in aminx.io.sink_provenance to
# distinguish stores instead.
GRID_SCHEMA_VERSION: Final[str] = "grid_v1"

# Non-grid sampling schema version. Used when grid_mode=False.
#
# REVERTED to "sampling_v1" (audit finding A, task_id `260910_aminx-sink-provenance-schema`,
# code-review round on PR #154): this constant is bumped together with GRID_SCHEMA_VERSION
# at every call site (GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION),
# so keeping them in lockstep prevents accidental re-introduction of schema-version/resume-safety
# coupling. The new fields (logits_bias_semantics, prng_seed, aminx_version) are purely additive.
# Use SinkProvenanceVersion in aminx.io.sink_provenance for the hash-free way to distinguish
# stores with these new fields.
SAMPLING_SCHEMA_VERSION: Final[str] = "sampling_v1"

# Scoring (full-context inference) schema version.
SCORING_SCHEMA_VERSION: Final[str] = "scoring_v1"

# Inspection (detailed model internals export) schema version.
INSPECTION_SCHEMA_VERSION: Final[str] = "inspection_v1"

# Jacobian (gradient computation) schema version.
JACOBIAN_SCHEMA_VERSION: Final[str] = "jacobian_v1"

__all__ = [
  "GRID_SCHEMA_VERSION",
  "INSPECTION_SCHEMA_VERSION",
  "JACOBIAN_SCHEMA_VERSION",
  "SAMPLING_SCHEMA_VERSION",
  "SCORING_SCHEMA_VERSION",
]
