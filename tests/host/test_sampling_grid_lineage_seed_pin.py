"""Regression coverage for the GRID_SCHEMA_VERSION / PRNG-seed decoupling.

task_id `260910_aminx-sink-provenance-schema`, FINDING 1 (CRITICAL). Before this fix,
`_grid_job_seed_hash`'s payload keyed its "schema_version" entry off `GRID_SCHEMA_VERSION`
-- the SAME constant used to label stores for readers -- and that hash folds directly into
`_base_sampling_key` (`jax.random.fold_in`). Bumping `GRID_SCHEMA_VERSION` for a purely
cosmetic/labelling reason (as commit 7b1bc3b did, `grid_v1` -> `grid_v2`) therefore silently
reseeded every grid-mode run: different sampled sequences, different logits at later
autoregressive positions.

The fix introduces `_SEED_HASH_SCHEMA_PIN`, a frozen literal used ONLY by
`_grid_job_seed_hash`, decoupled from `GRID_SCHEMA_VERSION`. These tests pin the exact
digest for a fixed synthetic spec+lineage and prove a `GRID_SCHEMA_VERSION` mutation cannot
move it (or the derived base sampling key) -- while also proving the test harness is not
vacuous by checking `_grid_manifest_row_hash` (which is SUPPOSED to keep tracking
`GRID_SCHEMA_VERSION`, see that function's own docstring) DOES change under the same
mutation.
"""

from __future__ import annotations

import jax

from aminx.host import _sampling_grid_lineage as gl
from aminx.run.specs import SamplingSpecification

# Fixed synthetic spec+lineage, chosen arbitrarily but held constant. If this hash ever
# needs to change, that is a deliberate, documented reproducibility break of every stored
# grid-mode run to date -- not something a routine refactor should cause silently.
_EXPECTED_SEED_HASH = (
  "5ad27cee01bab5fba3134676106ca96951d3b89ee276225e9041ed1d27728f93"
)
_EXPECTED_BASE_KEY_DATA = [1075959580, 493978054]


def _make_synthetic_grid_spec() -> SamplingSpecification:
  return SamplingSpecification(
    inputs=[],
    grid_mode=True,
    job_id="synthetic_job_260910",
    chunk_id=0,
    sample_start=0,
    sample_count=5,
    num_samples=5,
    temperature=0.5,
    backbone_noise=0.1,
    model_family="proteinmpnn",
    random_seed=42,
  )


def test_grid_job_seed_hash_is_pinned() -> None:
  """Pin `_grid_job_seed_hash`'s digest for a fixed synthetic spec+lineage.

  A change to this value that isn't a deliberate, called-out reproducibility break is a
  regression -- fails loudly rather than silently reseeding every stored grid-mode run.
  """
  spec = _make_synthetic_grid_spec()
  lineage = gl._resolve_grid_lineage(spec)
  assert lineage is not None
  seed_hash = gl._grid_job_seed_hash(spec, lineage)
  assert seed_hash == _EXPECTED_SEED_HASH


def test_base_sampling_key_is_pinned() -> None:
  """Pin the actual derived PRNG key, one level below the hash -- the thing sampling uses."""
  spec = _make_synthetic_grid_spec()
  lineage = gl._resolve_grid_lineage(spec)
  key = gl._base_sampling_key(spec, grid_lineage=lineage)
  assert jax.random.key_data(key).tolist() == _EXPECTED_BASE_KEY_DATA


def test_grid_schema_version_bump_does_not_move_seed_hash_or_base_key() -> None:
  """THE regression test: a `GRID_SCHEMA_VERSION` bump must not reseed sampling.

  Simulates the exact defect commit 7b1bc3b introduced (grid_v1 -> grid_v2) by mutating
  the module-level constant directly, then asserts the seed hash and the derived base
  sampling key are BOTH unaffected. Restores the original value in a `finally` so this
  test cannot leak state into others.
  """
  spec = _make_synthetic_grid_spec()
  lineage = gl._resolve_grid_lineage(spec)
  assert lineage is not None

  seed_hash_before = gl._grid_job_seed_hash(spec, lineage)
  key_before = gl._base_sampling_key(spec, grid_lineage=lineage)

  original_schema_version = gl.GRID_SCHEMA_VERSION
  try:
    gl.GRID_SCHEMA_VERSION = "grid_v999_simulated_future_bump"

    seed_hash_after = gl._grid_job_seed_hash(spec, lineage)
    key_after = gl._base_sampling_key(spec, grid_lineage=lineage)

    assert seed_hash_after == seed_hash_before, (
      "_grid_job_seed_hash moved when GRID_SCHEMA_VERSION changed -- the seed hash payload "
      "must use _SEED_HASH_SCHEMA_PIN, not GRID_SCHEMA_VERSION."
    )
    assert jax.random.key_data(key_after).tolist() == jax.random.key_data(key_before).tolist(), (
      "_base_sampling_key moved when GRID_SCHEMA_VERSION changed -- a schema-version bump "
      "must never change sampled output."
    )
  finally:
    gl.GRID_SCHEMA_VERSION = original_schema_version


def test_grid_schema_version_bump_still_moves_manifest_row_hash() -> None:
  """Non-vacuousness check: mutating GRID_SCHEMA_VERSION DOES move the OTHER hash.

  `_grid_manifest_row_hash` is documented (see its own docstring) to intentionally keep
  tracking `GRID_SCHEMA_VERSION` for manifest-row identity/dedupe/done-marker matching --
  it does not feed the PRNG. This proves the previous test's harness actually exercises a
  live wire (mutating the module attribute has a real, observable effect somewhere) rather
  than testing a no-op mutation that happened to look like a passing assertion.
  """
  spec = _make_synthetic_grid_spec()
  lineage = gl._resolve_grid_lineage(spec)
  assert lineage is not None

  row_hash_before = gl._grid_manifest_row_hash(spec, lineage)

  original_schema_version = gl.GRID_SCHEMA_VERSION
  try:
    gl.GRID_SCHEMA_VERSION = "grid_v999_simulated_future_bump"
    row_hash_after = gl._grid_manifest_row_hash(spec, lineage)
    assert row_hash_after != row_hash_before
  finally:
    gl.GRID_SCHEMA_VERSION = original_schema_version
