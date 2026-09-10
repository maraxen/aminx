"""Regression coverage for FINDING 2: bias array silently dropped in grid mode.

task_id `260910_aminx-sink-provenance-schema`. `host/streaming.py::_sample_streaming` and
`sampling/multistate_poe.py::sample_multistate_poe_campaign_row` both build up a
`root_arrays` dict via `.update(bias_arrays)` (staging the persisted ``"bias"`` array), then
-- ONLY when `grid_lineage is not None` -- reassigned `root_arrays = {...}` instead of
`.update(...)`, silently discarding whatever was staged before. Net effect: with
`grid_mode=True` + `return_logits=True` + a nonzero bias, the `bias` array never reached the
store while `root_attrs["logits_bias_semantics"]["bias_persisted"]` still claimed `True` --
a lying attr, worse than a missing one.

Both tests below drive the REAL staging code path (no stubbing of the bug site itself) with
everything upstream of it (model, real protein I/O) faked out, so they exercise the exact
dict-construction bug and would have failed against 7b1bc3b before the fix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import zarr

from aminx.host._sampling_grid_lineage import _grid_job_seed_hash, _resolve_grid_lineage
from aminx.host.streaming import _sample_streaming
from aminx.io.sink_provenance import SINK_PROVENANCE_VERSION
from aminx.run.specs import SamplingSpecification
from aminx.sampling import multistate_poe

_SEQ_LEN = 8


def test_sample_streaming_grid_mode_persists_bias_array() -> None:
  """`_sample_streaming` must stage `bias` even when `grid_lineage is not None`.

  Uses an EMPTY protein iterator so the batch-sampling loop body never executes (and no
  model is ever loaded) -- the root-level attrs/arrays staging under test happens
  unconditionally before that loop, so this still exercises the real bug site.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    bias = np.ones((_SEQ_LEN, 21), dtype=np.float32)
    spec = SamplingSpecification(
      inputs=[],
      grid_mode=True,
      job_id="synthetic_job_260910",
      chunk_id=0,
      sample_start=0,
      sample_count=3,
      num_samples=3,
      return_logits=True,
      bias=bias,
      output_h5_path=str(output_dir),
    )

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
      msg = "sample_batch_fn must not be called with an empty protein_iterator"
      raise AssertionError(msg)

    _sample_streaming(spec, [], None, _fail_if_called)

    root = zarr.open_group(str(output_dir), mode="r")
    array_keys = set(root.array_keys())
    assert "bias" in array_keys, (
      f"'bias' array missing from grid-mode store; got arrays {sorted(array_keys)}. "
      "root_arrays.update(...) must not be replaced by a bare reassignment in the "
      "grid_lineage branch."
    )
    np.testing.assert_array_equal(np.asarray(root["bias"]), bias)

    semantics = root.attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True
    assert semantics["bias_is_nonzero"] is True
    # The lying-attr failure mode: bias_persisted=True but no "bias" array. Assert the
    # attr and the array agree, not just that each looks right in isolation.
    assert semantics["bias_persisted"] == ("bias" in array_keys)


def test_multistate_poe_campaign_row_grid_mode_persists_bias_array() -> None:
  """`sample_multistate_poe_campaign_row` must stage `bias` even when grid-lineage applies.

  Fakes out `sample_multistate_poe_bead` (the actual AR sampling/model call) with a
  zero-tensor stand-in so this stays CPU-only and fast while still exercising the real
  root_arrays/root_attrs staging logic that had the bug.
  """

  def _fake_bead(
    _cell_spec: SamplingSpecification,
    _cell_key: object,
    n_samples: int,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
      jnp.zeros((n_samples, _SEQ_LEN), dtype=jnp.int32),
      jnp.zeros((n_samples, _SEQ_LEN, 21), dtype=jnp.float32),
    )

  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    bias = np.ones((_SEQ_LEN, 21), dtype=np.float32)
    spec = SamplingSpecification(
      inputs=["state_a.pdb", "state_b.pdb"],
      batch_size=2,
      grid_mode=True,
      job_id="synthetic_poe_job_260910",
      chunk_id=0,
      sample_start=0,
      sample_count=3,
      num_samples=3,
      return_logits=True,
      bias=bias,
      output_h5_path=str(output_dir),
      multi_state_strategy="product",
    )

    with patch.object(multistate_poe, "sample_multistate_poe_bead", _fake_bead):
      multistate_poe.sample_multistate_poe_campaign_row(spec)

    root = zarr.open_group(str(output_dir), mode="r")
    array_keys = set(root.array_keys())
    assert "bias" in array_keys, (
      f"'bias' array missing from grid-mode PoE store; got arrays {sorted(array_keys)}. "
      "root_arrays.update(...) must not be replaced by a bare reassignment in the "
      "grid_lineage branch."
    )
    np.testing.assert_array_equal(np.asarray(root["bias"]), bias)

    semantics = root.attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True
    assert semantics["bias_persisted"] == ("bias" in array_keys)


# ---------------------------------------------------------------------------
# Code-review round on PR #154: findings A (hash-free marker actually staged),
# B (non-AR decoder must not get AR bias semantics), C (grid seed hash recorded),
# D (bias_array_group makes bias_persisted honest).
#
# Same empty-protein-iterator harness as above -- the root-level staging under test runs
# unconditionally before the batch loop, so no model is ever loaded.
# ---------------------------------------------------------------------------


def _grid_spec(output_dir: Path, bias: np.ndarray | None = None) -> SamplingSpecification:
  """The shared grid-mode spec for the tests below."""
  return SamplingSpecification(
    inputs=[],
    grid_mode=True,
    job_id="synthetic_job_260910",
    chunk_id=0,
    sample_start=0,
    sample_count=3,
    num_samples=3,
    return_logits=bias is not None,
    bias=bias,
    output_h5_path=str(output_dir),
  )


def _fail_if_sampled(*_args: object, **_kwargs: object) -> None:
  msg = "sample_batch_fn must not be called with an empty protein_iterator"
  raise AssertionError(msg)


def test_grid_mode_records_grid_job_seed_hash() -> None:
  """FINDING C: `prng_seed` alone under-determines the key that drove a grid-mode run.

  `_base_sampling_key` starts at `jax.random.key(random_seed)` then folds in four words
  derived from `_grid_job_seed_hash(spec, lineage)`. That hash keys off `job_id` plus
  strategy/conditioning fields, so a reader holding only `prng_seed` cannot re-derive the
  actual sampling key without reconstructing the exact grid lineage. Recording it is what
  makes the store's reproducibility claim meetable.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    spec = _grid_spec(output_dir)
    _sample_streaming(spec, [], None, _fail_if_sampled)

    root = zarr.open_group(str(output_dir), mode="r")
    assert "grid_job_seed_hash" in root.attrs, (
      f"grid-mode store records prng_seed={root.attrs.get('prng_seed')!r} but no "
      "'grid_job_seed_hash' -- the seed int alone does not reproduce the sampling key. "
      f"Got attrs {sorted(root.attrs.keys())}."
    )

    lineage = _resolve_grid_lineage(spec)
    assert lineage is not None
    assert root.attrs["grid_job_seed_hash"] == _grid_job_seed_hash(spec, lineage)

    # Non-vacuousness: prove this is a DISTINCT value, not an alias of the hash already
    # stored. `manifest_row_hash` keys off GRID_SCHEMA_VERSION and addresses the output
    # path; the seed hash is pinned to `_SEED_HASH_SCHEMA_PIN` and addresses the PRNG. If
    # the two ever became equal, the deliberate decoupling of the schema constants that
    # finding A's revert exists to preserve would have been undone.
    assert root.attrs["grid_job_seed_hash"] != root.attrs["manifest_row_hash"], (
      "grid_job_seed_hash equals manifest_row_hash -- the seed-hash/row-hash decoupling "
      "(the whole point of _SEED_HASH_SCHEMA_PIN) has been undone."
    )


def test_root_store_stages_hash_free_sink_provenance_version() -> None:
  """FINDING A: the hash-free marker must actually be STAGED, not merely defined.

  Reverting the `GRID_SCHEMA_VERSION` bump left `SINK_PROVENANCE_VERSION` as the only way
  a reader can distinguish a store carrying the new optional attrs from one predating
  them. A constant no writer stamps provides exactly none of that.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    _sample_streaming(_grid_spec(output_dir), [], None, _fail_if_sampled)

    root = zarr.open_group(str(output_dir), mode="r")
    assert root.attrs.get("sink_provenance_version") == SINK_PROVENANCE_VERSION, (
      "root store does not stamp 'sink_provenance_version'; a reader cannot tell this "
      f"store carries the new provenance attrs. Got attrs {sorted(root.attrs.keys())}."
    )
    # It must not have come at the cost of re-bumping the HASHED label (finding A's
    # original defect). Pin both here so a future edit cannot trade one for the other.
    assert root.attrs["schema_version"] == "grid_v1"


def test_bias_array_group_locates_the_persisted_bias_array() -> None:
  """FINDING D: `bias_persisted: true` must say WHERE the array is.

  The same `logits_bias_semantics` dict is merged into the root group AND every
  per-structure group, but the `bias` array is staged ONCE, at the root. Without
  `bias_array_group` a reader of `structure_3` sees `bias_persisted: true` and looks for a
  `bias` array in that group, where none exists -- true about the store, false-reading
  about the group.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    bias = np.ones((_SEQ_LEN, 21), dtype=np.float32)
    _sample_streaming(_grid_spec(output_dir, bias=bias), [], None, _fail_if_sampled)

    root = zarr.open_group(str(output_dir), mode="r")
    semantics = root.attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True
    assert semantics["bias_array_group"] == "/", (
      "bias_persisted=True but bias_array_group="
      f"{semantics.get('bias_array_group')!r} -- a reader cannot locate the array."
    )
    assert "bias" in set(root.array_keys())


def test_non_autoregressive_strategy_refuses_to_stage_ar_bias_semantics() -> None:
  """FINDING B: AR bias semantics must not be stamped on non-AR logits.

  `logits_bias_semantics` asserts the stored/sampling bias split from
  `inference/decode/autoregressive.py`. Decode mode is a pure function of
  `sampling_strategy` (`host/plan.py::resolve_decode_mode`: "straight_through" -> STEMode
  -> `decode/ste.py`). `SamplingSpecification.__post_init__` already rejects
  straight_through in GRID mode -- so this test necessarily uses grid_mode=False, which is
  precisely the reachable hole the guard closes.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    spec = SamplingSpecification(
      inputs=[],
      grid_mode=False,
      num_samples=3,
      return_logits=True,
      sampling_strategy="straight_through",
      iterations=1,
      learning_rate=0.1,
      output_h5_path=str(output_dir),
    )

    with pytest.raises(NotImplementedError, match="logits_bias_semantics"):
      _sample_streaming(spec, [], None, _fail_if_sampled)


# ---------------------------------------------------------------------------
# Code-review round 2 on PR #154: the four assertions above covered only
# `_sample_streaming`. `sample_multistate_poe_campaign_row` stages the SAME three attrs
# and needs the same guard, so it gets the same coverage rather than relying on the two
# writers being kept in sync by inspection.
# ---------------------------------------------------------------------------


def _fake_bead(
  _cell_spec: SamplingSpecification,
  _cell_key: object,
  n_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Zero-tensor stand-in for the real AR sampling/model call -- keeps these CPU-only."""
  return (
    jnp.zeros((n_samples, _SEQ_LEN), dtype=jnp.int32),
    jnp.zeros((n_samples, _SEQ_LEN, 21), dtype=jnp.float32),
  )


def _poe_spec(output_dir: Path, bias: np.ndarray | None = None) -> SamplingSpecification:
  return SamplingSpecification(
    inputs=["state_a.pdb", "state_b.pdb"],
    batch_size=2,
    grid_mode=True,
    job_id="synthetic_poe_job_260910",
    chunk_id=0,
    sample_start=0,
    sample_count=3,
    num_samples=3,
    return_logits=bias is not None,
    bias=bias,
    output_h5_path=str(output_dir),
    multi_state_strategy="product",
  )


def test_poe_writer_records_seed_hash_marker_and_bias_group() -> None:
  """The PoE writer must stage the same three provenance attrs `_sample_streaming` does.

  Findings A, C and D were originally asserted only against `host/streaming.py`, leaving
  `sampling/multistate_poe.py`'s identical root staging unverified -- the two writers are
  separate code, so "the other one does it" is not coverage.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    bias = np.ones((_SEQ_LEN, 21), dtype=np.float32)
    spec = _poe_spec(output_dir, bias=bias)

    with patch.object(multistate_poe, "sample_multistate_poe_bead", _fake_bead):
      multistate_poe.sample_multistate_poe_campaign_row(spec)

    root = zarr.open_group(str(output_dir), mode="r")

    # FINDING A -- hash-free marker staged, and the hashed label NOT re-bumped.
    assert root.attrs.get("sink_provenance_version") == SINK_PROVENANCE_VERSION
    assert root.attrs["schema_version"] == "grid_v1"

    # FINDING C -- the seed hash is recorded, matches the function `_base_sampling_key`
    # itself calls, and is a genuinely distinct value from the row hash.
    lineage = _resolve_grid_lineage(spec)
    assert lineage is not None
    assert root.attrs["grid_job_seed_hash"] == _grid_job_seed_hash(spec, lineage)
    assert root.attrs["grid_job_seed_hash"] != root.attrs["manifest_row_hash"]

    # FINDING D -- bias_persisted says WHERE, and the named group really holds it.
    semantics = root.attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True
    assert semantics["bias_array_group"] == "/"
    assert "bias" in set(root.array_keys())


def test_poe_writer_refuses_to_stage_ar_bias_semantics_for_non_ar_strategy() -> None:
  """FINDING B, second writer: the guard must be EXPLICIT here, not incidental.

  Two unrelated checks happen to reject "straight_through" before the staging block today
  (`sample_multistate_poe_bead`'s AutoregressiveMode assertion, and
  `SamplingSpecification.__post_init__`'s grid_mode rule). Both exist for other reasons, so
  this test pins the dedicated guard rather than the incidental protection -- it patches
  the bead out entirely, which removes the first of those two checks, and uses
  grid_mode=False, which removes the second.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "store"
    spec = SamplingSpecification(
      inputs=["state_a.pdb", "state_b.pdb"],
      batch_size=2,
      grid_mode=False,
      num_samples=3,
      return_logits=True,
      sampling_strategy="straight_through",
      iterations=1,
      learning_rate=0.1,
      output_h5_path=str(output_dir),
      multi_state_strategy="product",
    )

    with (
      patch.object(multistate_poe, "sample_multistate_poe_bead", _fake_bead),
      pytest.raises(NotImplementedError, match="logits_bias_semantics"),
    ):
      multistate_poe.sample_multistate_poe_campaign_row(spec)
