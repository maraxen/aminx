"""Campaign reuse and the sampler both treat seed 0 as seed 0."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from xtrax.run import canonical_json_bytes

from aminx.host.campaign import compute_unit_input_hash

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"


def test_unit_input_hash_adds_seed0_semantics_only_for_seed_zero(tmp_path: Path) -> None:
  pdb = tmp_path / "s.pdb"
  pdb.write_text("x")
  base = {"inputs": [str(pdb)], "output_h5_path": "x", "temperature": [0.1]}

  hash_seven, components_seven = compute_unit_input_hash({**base, "random_seed": 7})
  assert "seed0_semantics" not in components_seven

  hash_zero, components_zero = compute_unit_input_hash({**base, "random_seed": 0})
  assert components_zero["seed0_semantics"] == 2
  without_semantics = {
    key: value for key, value in components_zero.items() if key != "seed0_semantics"
  }
  previous = hashlib.sha256(canonical_json_bytes(without_semantics)).hexdigest()
  assert hash_zero != previous
  assert hash_seven != hash_zero


@pytest.mark.slow
@pytest.mark.requires_weights
def test_sample_seed_zero_reaches_the_sampler() -> None:
  """Seed 0 and seed 42 differ; repeating seed 7 is identical."""
  pytest.importorskip("aminx")
  from aminx.host import runner  # noqa: PLC0415
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  if not STRUCTURE.is_file():
    pytest.skip(f"structure fixture missing: {STRUCTURE}")

  def sequences(seed: int) -> np.ndarray:
    result = runner.sample(
      SamplingSpecification(
        inputs=str(STRUCTURE),
        chain_id="A",
        checkpoint_id="proteinmpnn_v_48_020",
        num_samples=1,
        temperature=0.1,
        backbone_noise=0.0,
        random_seed=seed,
        batch_size=1,
        max_length=76,
        return_logits=False,
      ),
    )
    return np.asarray(result["sequences"])

  assert not np.array_equal(sequences(0), sequences(42))
  assert np.array_equal(sequences(7), sequences(7))
