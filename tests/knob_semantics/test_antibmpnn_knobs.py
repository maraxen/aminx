# ruff: noqa: S101, PLR2004
"""AntiBMPNN knob semantics (P4 redo 261104).

AntiBMPNN's sampling CLI (``Running_AntiBMPNN_run.py:405-452``) is the ProteinMPNN flag set, so these
tests drive aminx's real ``sample`` / ``score`` on the real ``antibmpnn_v_48_000`` weights (Hub) and
check what each upstream flag MEANS, not that a dataclass stores a value. The only AntiBMPNN-specific
upstream logic is ``helper_scripts/parsing_design_result.py`` (descriptor scoring of a design segment).

Numerical parity with the real upstream is NOT claimed here; that is ``scripts/parity/antibmpnn_parity_v2.py``.
The earlier revision of this file tested ``cdr_scheme`` / ``heavy_chain_id`` / ``light_chain_id`` /
``cdr_bias_weight`` and a ``checkpoint_id`` option, none of which exists upstream (rg 261104 over the
AntiBMPNN repository); those options were removed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from aminx.families.antibmpnn import design_descriptors, design_segment
from aminx.run.options import AntiBMPNNOptions
from aminx.utils.aa_convert import MPNN_ALPHABET

_PDB = Path(__file__).resolve().parents[2] / "tests" / "data" / "1ubq.pdb"
_CHECKPOINT = "antibmpnn_v_48_000"
_MAX_LENGTH = 128
_L = 76
_NATIVE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

weights = pytest.mark.requires_weights


def _sample(**kwargs) -> np.ndarray:  # noqa: ANN003
  from aminx.host.runner import sample

  kwargs.setdefault("random_seed", 0)
  result = sample(inputs=[str(_PDB)], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH, **kwargs)
  return np.asarray(result["sequences"]).reshape(-1, _MAX_LENGTH)[:, :_L]


def _score(sequence: str = _NATIVE, **kwargs) -> float:  # noqa: ANN003
  from aminx.host.runner import score

  kwargs.setdefault("random_seed", 0)
  result = score(inputs=[str(_PDB)], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH, sequences_to_score=[sequence], output_kind="nll", **kwargs)
  return float(np.asarray(result["scores"]).reshape(-1)[0])


def _letters(tokens: np.ndarray) -> str:
  return "".join(MPNN_ALPHABET[int(t)] for t in tokens)


def _native_tokens() -> np.ndarray:
  return np.asarray([MPNN_ALPHABET.index(c) for c in _NATIVE])


def test_options_carry_only_upstream_fields() -> None:
  """The invented CDR / chain / bias-weight fields are gone; only parsing_design_result's --start/--end remain."""
  names = {field.name for field in dataclasses.fields(AntiBMPNNOptions)}
  assert names == {"descriptor_start", "descriptor_end"}
  for invented in ("cdr_scheme", "heavy_chain_id", "light_chain_id", "cdr_bias_weight", "compute_descriptors", "checkpoint_id"):
    with pytest.raises(TypeError):
      AntiBMPNNOptions(**{invented: 1})


@weights
def test_knob_semantics_model_name() -> None:
  """model_name -> checkpoint_id: different noise-level checkpoints are different weights; 040..100 are not on the Hub."""
  from aminx.io.weights import LEGACY_ALIAS_MAP, load_model

  for version in ("v_48_000", "v_48_010", "v_48_020", "v_48_030"):
    assert ("antibmpnn", version) in LEGACY_ALIAS_MAP
  assert ("antibmpnn", "v_48_040") not in LEGACY_ALIAS_MAP  # upstream ships antibmpnn_040..100.pt; debt 3247
  first = load_model(checkpoint_id="antibmpnn_v_48_000")
  second = load_model(checkpoint_id="antibmpnn_v_48_010")
  assert not np.array_equal(np.asarray(first.w_out.weight), np.asarray(second.w_out.weight))


@weights
def test_knob_semantics_score_only() -> None:
  """score_only: the AR-order score (ar_mask) is a different estimand from the full-context default."""
  full_context = _score()
  causal = np.tril(np.ones((_L, _L), dtype=np.float32), -1)
  autoregressive = _score(ar_mask=causal)
  assert np.isfinite(full_context)
  assert np.isfinite(autoregressive)
  assert abs(full_context - autoregressive) > 1e-3


@weights
def test_knob_semantics_path_to_fasta() -> None:
  """path_to_fasta -> sequences_to_score: a shuffled sequence scores worse than the native one."""
  rng = np.random.default_rng(0)
  shuffled = "".join(rng.permutation(list(_NATIVE)))
  assert _score(_NATIVE) < _score(shuffled)


def _logp(sequence: str = _NATIVE, **kwargs) -> np.ndarray:  # noqa: ANN003
  """Per-position log-softmax of the logits ``score(return_logits=True)`` returns, shape ``(L, 21)``."""
  from aminx.host.runner import score

  result = score(
    inputs=[str(_PDB)], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH, random_seed=0,
    sequences_to_score=[sequence], output_kind="nll", return_logits=True, **kwargs,
  )
  logits = np.asarray(result["logits"], dtype=np.float64).reshape(-1, 21)[:_L]
  shifted = logits - logits.max(axis=-1, keepdims=True)
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@weights
def test_knob_semantics_conditional_probs_only() -> None:
  """conditional_probs_only: p(s_i | rest of sequence, backbone) -- a normalised row per position that sees the
  other residues but never its own (upstream ``ProteinMPNN.conditional_probs``, protein_mpnn_utils.py:1292;
  ``conditional_probs_only_backbone`` -> p(s_i | backbone), an all-zero ``ar_mask``)."""
  from aminx.host.runner import score

  native = _native_tokens()
  full = _logp()
  assert full.shape == (_L, 21)
  # (1) each row is a normalised log-distribution (log-softmax of the returned logits)
  np.testing.assert_allclose(np.exp(full).sum(axis=-1), 1.0, atol=1e-9)
  assert np.all(np.isfinite(full))
  # (2) the returned logits are the ones the NLL is made of: score == -mean_i log p(native_i | rest)
  nll = float(np.asarray(score(
    inputs=[str(_PDB)], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH, random_seed=0,
    sequences_to_score=[_NATIVE], output_kind="nll",
  )["scores"]).reshape(-1)[0])
  assert nll == pytest.approx(-float(np.mean(full[np.arange(_L), native])), abs=1e-3)
  # (3) conditional on the REST: changing residue j changes the row at some other position, but never row j itself
  j = 30
  mutated = _NATIVE[:j] + ("A" if _NATIVE[j] != "A" else "G") + _NATIVE[j + 1 :]
  other = _logp(mutated)
  assert np.array_equal(other[j], full[j]), "row j must not see its own residue"
  assert np.max(np.abs(np.delete(other, j, axis=0) - np.delete(full, j, axis=0))) > 1e-3, "other rows must see residue j"
  # (4) backbone-only: with nothing visible, the rows do not depend on the sequence at all
  zero = np.zeros((_L, _L), dtype=np.float32)
  bb = _logp(ar_mask=zero)
  np.testing.assert_allclose(np.exp(bb).sum(axis=-1), 1.0, atol=1e-9)
  assert np.array_equal(_logp(mutated, ar_mask=zero), bb)
  assert np.max(np.abs(bb - full)) > 1e-3, "backbone-only differs from full-context conditionals"


@weights
def test_knob_semantics_backbone_noise() -> None:
  """backbone_noise: zero is deterministic; 1.0 A of backbone noise changes (worsens) the score."""
  clean = _score(backbone_noise=0.0)
  assert clean == _score(backbone_noise=0.0)
  assert _score(backbone_noise=1.0) > clean + 0.1


@weights
def test_knob_semantics_num_seq_per_target() -> None:
  """num_seq_per_target -> num_samples: exactly that many sequences (upstream floors to a batch multiple)."""
  assert _sample(num_samples=3, temperature=0.5).shape[0] == 3


@weights
def test_knob_semantics_batch_size() -> None:
  """batch_size -> samples_batch_size: chunking does not change how many sequences come back."""
  assert _sample(num_samples=4, temperature=0.5, samples_batch_size=1).shape[0] == 4
  assert _sample(num_samples=4, temperature=0.5, samples_batch_size=4).shape[0] == 4


@weights
def test_knob_semantics_max_length() -> None:
  """max_length: aminx pads / crops to max_length (upstream drops longer proteins instead)."""
  from aminx.host.runner import sample

  result = sample(
    inputs=[str(_PDB)], checkpoint_id=_CHECKPOINT, max_length=40, truncation_strategy="center_crop",
    num_samples=1, temperature=0.5, random_seed=0,
  )
  assert np.asarray(result["sequences"]).shape[-1] == 40


def _recovery(designs: np.ndarray) -> float:
  return float(np.mean(designs == _native_tokens()[None, :]))


@weights
def test_knob_semantics_sampling_temp() -> None:
  """sampling_temp -> temperature: probability is softmax((logits+bias)/T), so low T concentrates on the model's mode."""
  cold = _recovery(_sample(num_samples=4, temperature=0.1))
  hot = _recovery(_sample(num_samples=4, temperature=5.0))
  assert cold > hot + 0.15


@weights
def test_knob_semantics_pdb_path() -> None:
  """pdb_path -> inputs: the structure is read, so a cold design recovers far more than chance (~6%) of its native sequence."""
  assert _recovery(_sample(num_samples=3, temperature=0.1)) > 0.2


@weights
def test_knob_semantics_fixed_positions() -> None:
  """fixed_positions_jsonl / chain_id_jsonl / pdb_path_chains: frozen positions keep the native residue, the rest are designed."""
  fixed = np.zeros(_MAX_LENGTH, dtype=np.float32)
  fixed[:20] = 1.0
  tokens = np.zeros(_MAX_LENGTH, dtype=np.int32)
  tokens[:_L] = _native_tokens()
  out = _sample(num_samples=2, temperature=1.0, fixed_mask=fixed, fixed_tokens=tokens)
  native = _native_tokens()
  assert np.all(out[:, :20] == native[:20])
  assert np.any(out[:, 20:] != native[20:])
  with pytest.raises(ValueError, match="fixed_tokens"):
    _sample(num_samples=1, temperature=1.0, fixed_mask=fixed)


@weights
def test_knob_semantics_omit_aa() -> None:
  """omit_AAs -> omit_aa: omitted letters are never drawn (leaving one letter forces it); X is never drawn by default."""
  keep_y = "".join(c for c in "ACDEFGHIKLMNPQRSTVW")
  assert set(_letters(_sample(num_samples=2, temperature=1.0, omit_aa=(keep_y,)).reshape(-1))) == {"Y"}
  assert MPNN_ALPHABET.index("X") not in set(_sample(num_samples=2, temperature=5.0).reshape(-1).tolist())


@weights
def test_knob_semantics_bias_aa() -> None:
  """bias_AA_jsonl -> bias: a large per-letter bias added to every position forces that letter."""
  bias = np.zeros((_MAX_LENGTH, 21), dtype=np.float32)
  bias[:, MPNN_ALPHABET.index("W")] = 100.0
  assert set(_letters(_sample(num_samples=1, temperature=0.1, bias=bias).reshape(-1))) == {"W"}


@weights
def test_knob_semantics_bias_by_res() -> None:
  """bias_by_res_jsonl -> bias: a per-position bias moves only the biased positions."""
  bias = np.zeros((_MAX_LENGTH, 21), dtype=np.float32)
  bias[:5, MPNN_ALPHABET.index("W")] = 100.0
  out = _sample(num_samples=1, temperature=0.1, bias=bias)[0]
  assert _letters(out[:5]) == "WWWWW"
  assert set(_letters(out[5:])) != {"W"}


@weights
def test_knob_semantics_omit_aa_per_position() -> None:
  """omit_AA_jsonl -> omit_aa_per_position: only the named position loses the named letters."""
  keep_k = "ACDEFGHILMNPQRSTVWY"
  out = _sample(num_samples=2, temperature=1.0, omit_aa_per_position={3: keep_k})
  assert set(_letters(out[:, 3])) == {"K"}
  assert len(set(_letters(out[:, 4:30].reshape(-1)))) > 1


@weights
def test_knob_semantics_tied_positions() -> None:
  """tied_positions_jsonl -> tie_group_map: positions in one group receive the same letter.

  The ``tied_positions`` spec field is NOT the working channel: ``utils.autoregression.resolve_tie_groups``
  (which would turn it into a map) has no caller in src/ (verified 261104), so setting it changes nothing
  (debt 3248). The per-position ``tie_group_map`` array is what reaches the decoder.
  """
  groups = np.arange(_MAX_LENGTH, dtype=np.int32)
  groups[1] = groups[2] = 0
  groups[10] = 9
  out = _sample(num_samples=3, temperature=1.0, tie_group_map=groups)
  assert np.all(out[:, 0] == out[:, 1])
  assert np.all(out[:, 0] == out[:, 2])
  assert np.all(out[:, 9] == out[:, 10])
  assert np.any(out[:, 3] != out[:, 4])


def test_knob_semantics_descriptor_segment() -> None:
  """parsing_design_result --start/--end: seq[start-1:end] (1-based, inclusive), then PD1/PD2/charge(pH 7.4).

  Reference values are the AntiBMPNN README results table (README.md:107), produced upstream with
  ``peptides``; the segment boundaries are the example script's ``DESIGN_ONLY_POSITIONS``.
  """
  full = "EVQLVESGGGLVQPGGSLRLSCAASGSIFNQYAMAWFRQAPGKEREFVAGMGAVPHYGEFVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCARSKSTYISYNSNGYDYWGQGTLVTVSS"
  options = AntiBMPNNOptions(descriptor_start=102, descriptor_end=106)
  assert design_segment(full, options) == "SYNSN"  # the example's designed window (parsing_design_result.py:8,27-28)
  assert design_segment(full, AntiBMPNNOptions()) == full
  assert design_segment(full, AntiBMPNNOptions(descriptor_start=3)) == full[2:]
  assert design_segment(full, AntiBMPNNOptions(descriptor_end=4)) == full[:4]
  readme = design_descriptors(["ARSKSTYLSRDSSGYDY"])[0]
  assert readme.pd1_volume == pytest.approx(-0.4271, abs=1e-4)
  assert readme.pd2_hydro == pytest.approx(0.3253, abs=1e-4)
  assert readme.pd3_charge == pytest.approx(0.9881, abs=1e-4)
  windowed = design_descriptors([full], options)[0]
  assert windowed.segment == "SYNSN"
  assert windowed.pd3_charge == pytest.approx(-0.0072, abs=1e-4)  # peptides 0.5.0, parsing_design_result CSV
  with pytest.raises(ValueError, match="descriptor_start"):
    design_segment(full, AntiBMPNNOptions(descriptor_start=0))
