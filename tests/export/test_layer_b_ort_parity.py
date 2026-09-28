"""ORT-vs-native-JAX parity for the P04 export wrapper (260926_browser-export-loop, T2b).

Builds the real P04 (unconditional-logit) export wrapper on the pinned checkpoint,
converts it through `jax2onnx.to_onnx` at one small bucket, and compares ONNX
Runtime's (`CPUExecutionProvider`) output against the wrapper's own native JAX output
on the SAME synthetic inputs. This is the end-to-end regression test for the T2b fix
(`aminx.utils.coordinates.compute_backbone_coordinates`'s out-of-range atom-axis index,
which previously failed ONNX Runtime execution outright -- `ShapeInferenceError`, node
`node_Squeeze_24` -- before ever reaching a numeric comparison): before the fix, this
test cannot even construct an `InferenceSession`; after it, ORT and JAX must agree
within `P04_LOGPROB_BAR` (1e-4, T3a's own log-prob bar, `layer_b_ort_calibrate.py`).

Marked `parity_heavy` (loads the real pinned checkpoint + a real jax2onnx conversion):
excluded from the default `-m 'not parity_heavy and not slow'` addopts, run explicitly
via `-k`/`-m parity_heavy` with `$ORTW` (jax2onnx/onnx/onnxruntime are not base
project dependencies).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.browser_validation.layer_b_build import _synthetic_inputs, ort_execution_check
from scripts.browser_validation.layer_b_ort_calibrate import P04_LOGPROB_BAR

pytestmark = pytest.mark.parity_heavy

#: Small bucket: fast to convert/execute; the defect (and its fix) are bucket-independent
#: (the atom axis is always 4-wide regardless of sequence length).
_BUCKET = 32


def test_p04_ort_matches_native_jax_log_probs(tmp_path: Path) -> None:
  import jax2onnx
  import onnxruntime as ort

  from aminx.export import PINNED_CHECKPOINT_ID, make_p04_unconditional
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  model = load_model(checkpoint_id=PINNED_CHECKPOINT_ID)
  stage_set = make_stage_set()
  wrapper = make_p04_unconditional(model, stage_set)

  inputs = _synthetic_inputs(_BUCKET)
  jax_inputs = tuple(jnp.asarray(a) for a in inputs)

  logits_jax, _idx_jax = wrapper(*jax_inputs)
  log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1), dtype=np.float64)
  assert np.all(np.isfinite(log_probs_jax))

  onnx_path = tmp_path / f"p04_L{_BUCKET}_parity.onnx"
  specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in inputs]
  jax2onnx.to_onnx(
    wrapper, specs, model_name="p04_parity", output_path=str(onnx_path), return_mode="file"
  )

  # The T2b gate itself: this is exactly what would have failed before the fix
  # (InferenceSession construction: ShapeInferenceError, node_Squeeze_24).
  gate = ort_execution_check(onnx_path, inputs)
  assert gate["ort_ok"] is True, gate["error"]

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  feeds = dict(zip([i.name for i in session.get_inputs()], inputs, strict=True))
  logits_ort, _idx_ort = session.run(None, feeds)
  log_probs_ort = np.asarray(jax.nn.log_softmax(jnp.asarray(logits_ort), axis=-1), dtype=np.float64)
  assert np.all(np.isfinite(log_probs_ort))

  max_abs = float(np.max(np.abs(log_probs_jax - log_probs_ort)))
  assert max_abs <= P04_LOGPROB_BAR, (
    f"ORT vs native-JAX log-prob max-abs {max_abs!r} exceeds P04_LOGPROB_BAR={P04_LOGPROB_BAR!r}"
  )
