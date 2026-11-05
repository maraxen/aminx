"""aminx's FAMPNN encoder on a padded input against upstream's, real ``fampnn_0_3`` weights, float32.

Reference: ``tests/data/fampnn_padded_oracle.json.gz`` from ``scripts/parity/dump_fampnn_padded_oracle.py`` (real upstream
``SeqDenoiser.score`` on 1ubq plus 11 pad residues with ``seq_mask = 0`` and junk coordinates, two mask states).

Finding (261105, titanix): upstream is *not* pad-invariant on the real residues (logits move by up to 0.078, node states by
0.047 when the same 76 residues are padded), so "padded equals unpadded" is the wrong criterion; the criterion is that
aminx on the padded input equals upstream on the padded input, which is what this test grades (``sampler.py`` passes an
all-ones ``seq_mask`` and the pad mask as ``mask``; upstream passes the pad mask for both).
Heavy (model on CPU): run on titanix, not locally. Skipped without the checkpoint.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.fampnn.convert import convert_fampnn_state_dict
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path(os.environ.get("AMINX_FAMPNN_ROOT", str(Path.home() / "repos/fampnn"))) / "weights" / "fampnn_0_3.pt"
ORACLE = ROOT / "tests/data/fampnn_padded_oracle.json.gz"
BAND = 2e-4  # float32 encoder band of the registered vehicle (max(1e-4, 10 x floor))

pytestmark = pytest.mark.skipif(not CKPT.is_file(), reason="needs the released fampnn_0_3 checkpoint")


@pytest.fixture(scope="module")
def oracle() -> dict:
  """The gzipped upstream reference."""
  with gzip.open(ORACLE, "rt") as fh:
    return json.load(fh)


@pytest.fixture(scope="module")
def model():  # noqa: ANN201
  """The converted real checkpoint."""
  return convert_fampnn_state_dict(load_restricted_torch_checkpoint(CKPT))


@pytest.mark.parametrize("state", ["all_masked", "half"])
@pytest.mark.parametrize("seq_mask_mode", ["ones", "pad_mask"])
def test_padded_encoder_matches_upstream(model, oracle: dict, state: str, seq_mask_mode: str) -> None:  # noqa: ANN001
  """Real-residue logits and node states of the padded run equal upstream's padded run."""
  n = oracle["n"]
  r = oracle[state]["padded"]
  pad_mask = jnp.asarray(oracle["padded_inputs"]["seq_mask"], jnp.float32)
  seq_mask = jnp.ones_like(pad_mask) if seq_mask_mode == "ones" else pad_mask
  out = model(
    jnp.asarray(r["x_in"], jnp.float32),
    jnp.asarray(r["atom_mask_noised"], jnp.float32),
    jnp.asarray(r["aatype_noised"], jnp.int32),
    seq_mask,
    pad_mask,
    jnp.asarray(oracle["padded_inputs"]["residue_index"], jnp.int32),
    jnp.asarray(oracle["padded_inputs"]["chain_index"], jnp.int32),
  )
  np.testing.assert_array_equal(np.asarray(out.e_idx)[:n], np.asarray(r["E_idx"])[:n])
  np.testing.assert_allclose(np.asarray(out.logits_fampnn)[:n], np.asarray(r["logits"])[:n], atol=BAND)
  np.testing.assert_allclose(np.asarray(out.h_v)[:n], np.asarray(r["h_V"])[:n], atol=BAND)
