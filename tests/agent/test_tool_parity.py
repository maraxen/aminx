"""G-PARITY: MCP sample, score, and jacobian match host.runner on 1ubq."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("cisternal")

from fastmcp import Client

from aminx.agent.mcp import build_server
from aminx.agent.requests import build_spec, spec_from_json, structure_lengths
from aminx.host import runner
from aminx.utils.aa_convert import protein_sequence_to_string

_PDB = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"


def _data(result: object) -> dict[str, Any]:
  """Return the structured payload of a tool call."""
  data = getattr(result, "data", result)
  if isinstance(data, dict):
    return data
  dump = getattr(data, "model_dump", None)
  if callable(dump):
    dumped = dump()
    if isinstance(dumped, dict):
      return dumped
  msg = f"unexpected tool payload type {type(data).__name__}"
  raise AssertionError(msg)


def _tool_sequences(payload: Mapping[str, Any]) -> list[str]:
  """Collect decoded sequences in sample-grid order."""
  sequences: list[str] = []
  structures = payload["structures"]
  if not isinstance(structures, list):
    msg = "sample result has no structures list"
    raise AssertionError(msg)
  for structure in structures:
    samples = structure["samples"]
    for sample in samples:
      sequences.append(str(sample["sequence"]))
  return sequences


def _direct_sequences(
  results: Mapping[str, Any],
  lengths: Mapping[str, int],
  structure_ids: Sequence[str],
) -> list[str]:
  """Decode a runner sample array and trim each row to the parsed length."""
  array = np.asarray(results["sequences"])
  batch, n_samples, n_noise, n_temp, _length = (int(dim) for dim in array.shape)
  decoded: list[str] = []
  for batch_index in range(batch):
    known = lengths[structure_ids[batch_index]]
    for sample_slot in range(n_samples):
      for noise_index in range(n_noise):
        for temperature_index in range(n_temp):
          text = protein_sequence_to_string(
            array[batch_index, sample_slot, noise_index, temperature_index],
          )
          decoded.append(text[:known])
  return decoded


def _tool_nlls(payload: Mapping[str, Any]) -> list[float]:
  """Collect negative log-likelihoods in structure then sequence order."""
  values: list[float] = []
  structures = payload["structures"]
  if not isinstance(structures, list):
    msg = "score result has no structures list"
    raise AssertionError(msg)
  for structure in structures:
    for scored in structure["scores"]:
      values.append(float(scored["nll"]))
  return values


@pytest.mark.slow
def test_sample_matches_runner_and_seed_changes(tmp_path: Path) -> None:
  """Sampled sequences match runner.sample, and a new seed changes them."""
  pdb = str(_PDB)
  options = {"num_samples": 2, "temperature": 0.1, "random_seed": 7}

  async def body() -> tuple[dict[str, Any], dict[str, Any]]:
    async with Client(build_server()) as client:
      seeded = await client.call_tool(
        "sample",
        {"inputs": [pdb], "output_dir": str(tmp_path), **options},
      )
      other = await client.call_tool(
        "sample",
        {
          "inputs": [pdb],
          "output_dir": str(tmp_path),
          "num_samples": 2,
          "temperature": 0.1,
          "random_seed": 8,
        },
      )
      return _data(seeded), _data(other)

  payload, other_payload = asyncio.run(body())
  # The runner buckets sample itself (76 residues run at 128), so the tool keeps the
  # specification default; the returned spec must reproduce the run exactly.
  assert build_spec("sample", [pdb], options).max_length == 512
  spec = spec_from_json(payload["spec"])
  assert spec.max_length == 512
  assert spec.length_bucketing is True
  direct = runner.sample(spec=spec)
  lengths = structure_lengths(spec)
  structure_ids = list(lengths)
  tool_sequences = _tool_sequences(payload)
  direct_sequences = _direct_sequences(direct, lengths, structure_ids)
  assert tool_sequences == direct_sequences
  residue_count = lengths[structure_ids[0]]
  assert tool_sequences
  assert all(len(sequence) == residue_count for sequence in tool_sequences)
  changed = _tool_sequences(other_payload)
  assert any(left != right for left, right in zip(tool_sequences, changed, strict=True))


@pytest.mark.slow
def test_score_matches_runner(tmp_path: Path) -> None:
  """Score negative log-likelihoods match runner.score on the sampled sequences."""
  pdb = str(_PDB)

  async def sample_sequences() -> list[str]:
    async with Client(build_server()) as client:
      result = await client.call_tool(
        "sample",
        {
          "inputs": [pdb],
          "num_samples": 2,
          "temperature": 0.1,
          "random_seed": 7,
          "output_dir": str(tmp_path),
        },
      )
      return _tool_sequences(_data(result))

  sequences = asyncio.run(sample_sequences())
  assert len(sequences) == 2

  async def score_sequences() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool(
        "score",
        {
          "inputs": [pdb],
          "sequences": sequences,
          "random_seed": 42,
          "output_dir": str(tmp_path),
        },
      )
      return _data(result)

  payload = asyncio.run(score_sequences())
  spec = spec_from_json(payload["spec"])
  assert spec.max_length == 512  # runner-bucketed: the tool does not fit
  assert list(spec.sequences_to_score) == sequences
  direct = runner.score(spec=spec)
  direct_nlls = [float(value) for value in np.asarray(direct["scores"]).reshape(-1)]
  assert _tool_nlls(payload) == pytest.approx(direct_nlls, rel=1e-6)


@pytest.mark.slow
def test_reverse_jacobian_side_file(tmp_path: Path) -> None:
  """Reverse-mode jacobian stores score_gradients for 1ubq with shape (L, 21)."""
  pdb = str(_PDB)

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool(
        "jacobian",
        {
          "inputs": [pdb],
          "mode": "reverse",
          "options": {"compute_apc": False},
          "output_dir": str(tmp_path),
          "inline_cap": 0,
        },
      )
      return _data(result)

  payload = asyncio.run(body())
  side_file = payload["side_file"]
  assert isinstance(side_file, dict)
  arrays = side_file["arrays"]
  matched = {
    key: info
    for key, info in arrays.items()
    if "score_gradients" in key and "1ubq" in key
  }
  assert len(matched) == 1
  info = next(iter(matched.values()))
  spec = spec_from_json(payload["spec"])
  assert spec.max_length == 128  # jacobian is not runner-bucketed, so the tool still fits
  length = structure_lengths(spec)["1ubq"]
  assert info["shape"] == [length, 21]

  # The runner pads to max_length (fitted to 128 here); the tool trims to the parsed residue count. Trimming must
  # keep every real row (equal to the direct runner's first `length` rows) and drop only padding
  # (the direct output's rows past `length` are all zero).
  key = next(iter(matched)).replace("/", "__")
  tool_gradients = np.load(side_file["path"])[key]
  direct = np.asarray(runner.jacobian(spec=spec)["score_gradients"][0])
  assert direct.shape[0] >= length
  np.testing.assert_allclose(tool_gradients, direct[:length], rtol=1e-6, atol=0)
  assert not np.any(direct[length:])
