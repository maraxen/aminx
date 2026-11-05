"""Padded input must not change the real residues' result (upstream batches pad with ``seq_mask = 0``).

Runs the real ``fampnn_0_3`` weights (skipped when the checkpoint is absent: set ``AMINX_FAMPNN_ROOT`` or keep
``~/repos/fampnn``) on 1ubq twice: as is, and with trailing padding residues whose coordinates, residue indices,
chain index and *noise draws* are junk. The full MAR sampler (``mar_sample_upstream``;
``sampler.py`` passes an all-ones ``seq_mask`` to the encoder and the pad mask as ``mask``) must agree on the real
residues. Heavy (fampnn model on CPU): run on titanix, not locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.fampnn import sampler
from aminx.families.fampnn.convert import convert_fampnn_state_dict
from aminx.families.fampnn.driver import _prepare_structure
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path(os.environ.get("AMINX_FAMPNN_ROOT", str(Path.home() / "repos/fampnn"))) / "weights" / "fampnn_0_3.pt"
PAD = 11
STEPS = 3
SCN_STEPS = 4

pytestmark = pytest.mark.skipif(not CKPT.is_file(), reason="needs the released fampnn_0_3 checkpoint")


@pytest.fixture(scope="module")
def model():  # noqa: ANN201
  return convert_fampnn_state_dict(load_restricted_torch_checkpoint(CKPT))


def _inputs(pad: int):  # noqa: ANN202
  prep = _prepare_structure(ROOT / "tests/data/1ubq.pdb")
  num_l = prep.aatype_fampnn.shape[0]
  rng = np.random.default_rng(7)
  x = np.zeros((num_l + pad, 37, 3), np.float32)
  x[:num_l] = prep.atom37_xyz
  # junk (not zero) coordinates at pad positions: they must be masked, not merely harmless
  x[num_l:] = rng.normal(size=(pad, 37, 3)).astype(np.float32) * 3.0
  aa = np.full(num_l + pad, 20, np.int32)
  aa[:num_l] = prep.aatype_fampnn
  valid = np.zeros(num_l + pad, np.float32)
  valid[:num_l] = 1.0
  missing = np.zeros((num_l + pad, 37), np.float32)
  missing[:num_l] = prep.missing_atom_mask
  ridx = np.zeros(num_l + pad, np.int32)
  ridx[:num_l] = prep.residue_idx
  cidx = np.zeros(num_l + pad, np.int32)
  cidx[:num_l] = prep.chain_idx
  return num_l, dict(x=x, aa=aa, valid=valid, missing=missing, ridx=ridx, cidx=cidx)


def _run(model, d, noise):  # noqa: ANN001, ANN202
  ts = sampler.timestep_schedule("linear", STEPS, 0.0, 1.0, jnp.float32)
  ts_scn = sampler.timestep_schedule("linear", SCN_STEPS, 0.0, 1.0, jnp.float32)
  return sampler.mar_sample_upstream(
    model, jnp.asarray(d["x"]), jnp.asarray(d["aa"]), jnp.asarray(d["valid"]), jnp.asarray(d["missing"]),
    jnp.asarray(d["ridx"]), jnp.asarray(d["cidx"]), ts, noise,
    temperature=0.1, seq_only=False, repack_last=True, psce_threshold=None, timesteps_scn=ts_scn,
    step_scale=1.5, s_churn=0.0,
  )  # fmt: skip


def test_mar_sampler_is_pad_invariant(model) -> None:  # noqa: ANN001
  n, a = _inputs(0)
  _, b = _inputs(PAD)
  rng = np.random.default_rng(11)
  calls = STEPS + 1  # repack_last
  dec = rng.random(n + PAD).astype(np.float32)
  cat = rng.random((STEPS, n + PAD)).astype(np.float32)
  prior = rng.normal(size=(calls, n + PAD, 33, 3)).astype(np.float32)
  noise_b = sampler.SamplerNoise(jnp.asarray(dec), jnp.asarray(cat), jnp.asarray(prior), None)
  noise_a = sampler.SamplerNoise(jnp.asarray(dec[:n]), jnp.asarray(cat[:, :n]), jnp.asarray(prior[:, :n]), None)
  ra, rb = _run(model, a, noise_a), _run(model, b, noise_b)
  # decoding order of the real residues must not depend on the pad draws
  np.testing.assert_array_equal(np.asarray(rb.aatype_final)[:n], np.asarray(ra.aatype_final))
  np.testing.assert_allclose(np.asarray(rb.x_final)[:n], np.asarray(ra.x_final), atol=5e-3)
  np.testing.assert_allclose(np.asarray(rb.seq_probs)[:n], np.asarray(ra.seq_probs), atol=1e-3)
  assert jax.numpy.isfinite(rb.x_final).all()
