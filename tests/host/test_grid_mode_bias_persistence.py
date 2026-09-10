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
import zarr

from aminx.host.streaming import _sample_streaming
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
