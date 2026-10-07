"""S8 gates for runner length bucketing.

Weight-loading tests follow ``tests/host/test_score_padding_invariance.py``: they are
marked ``slow`` and ``requires_weights``, and they load the ProteinMPNN checkpoint
through ``runner.sample`` / ``runner.score``.

No ligand structure fixture lives under ``tests/`` (searched ``tests/data`` and the
host tests). G-INVARIANCE therefore has no ligand case; that gap is recorded rather
than filled with a fabricated file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_DATA = Path(__file__).resolve().parents[1] / "data"
_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "length_bucketing"
_CHECKPOINT = "proteinmpnn_v_48_020"
# 1UBQ chain A is 76 residues, contiguous, so the span is 76 and the rung is 128.
_UBQ_SPAN = 76
_UBQ_RUNG = 128


def _skip_if_golden_runtime_differs(meta: dict[str, object]) -> None:
  import jax  # noqa: PLC0415

  backend = jax.default_backend()
  version = jax.__version__
  if backend != meta["backend"] or version != meta["jax_version"]:
    pytest.skip(
      "opt-out golden was captured on "
      f"jax {meta['jax_version']} backend {meta['backend']}; "
      f"this process is jax {version} backend {backend}"
    )


def _crop_first_residues(src: Path, dest: Path, n_residues: int) -> None:
  """Write a PDB containing the first ``n_residues`` ATOM residues of ``src``."""
  seen: list[tuple[str, str]] = []
  kept: list[str] = []
  for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.startswith("ATOM") or len(line) < 27:
      continue
    key = (line[21], line[22:27])
    if key not in seen:
      if len(seen) >= n_residues:
        break
      seen.append(key)
    kept.append(line)
  if len(seen) < n_residues:
    msg = f"{src.name} has {len(seen)} residues, wanted {n_residues}"
    raise AssertionError(msg)
  dest.write_text("\n".join(kept) + "\nEND\n", encoding="utf-8")


@pytest.mark.slow
@pytest.mark.requires_weights
def test_g_optout_matches_golden() -> None:
  """length_bucketing=False reproduces the pre-change sample and score arrays."""
  pytest.importorskip("aminx")
  from aminx.host.runner import sample, score  # noqa: PLC0415

  meta_path = _GOLDEN_DIR / "optout_golden.json"
  npz_path = _GOLDEN_DIR / "optout_golden.npz"
  if not meta_path.is_file() or not npz_path.is_file():
    pytest.skip(f"opt-out golden missing under {_GOLDEN_DIR}")
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  _skip_if_golden_runtime_differs(meta)
  golden = np.load(npz_path)
  produced: dict[str, np.ndarray] = {}
  for case in meta["cases"]:
    name = str(case["name"])
    pdb = str(_DATA / str(case["pdb"]))
    sampled = sample(
      inputs=[pdb],
      num_samples=case["num_samples"],
      temperature=case["temperature"],
      random_seed=case["random_seed"],
      max_length=meta["max_length"],
      length_bucketing=False,
    )
    scored = score(
      inputs=[pdb],
      sequences_to_score=case["scored_sequences"],
      random_seed=meta["score_seed"],
      max_length=meta["max_length"],
      length_bucketing=False,
    )
    produced[f"{name}__sample__sequences"] = np.asarray(sampled["sequences"])
    if sampled.get("logits") is not None:
      produced[f"{name}__sample__logits"] = np.asarray(sampled["logits"])
    produced[f"{name}__score__scores"] = np.asarray(scored["scores"])
    if scored.get("logits") is not None:
      produced[f"{name}__score__logits"] = np.asarray(scored["logits"])
  for key in golden.files:
    assert key in produced, key
    assert np.array_equal(produced[key], golden[key]), key


@pytest.mark.slow
@pytest.mark.requires_weights
def test_g_shape_bucketed_sample_keeps_padded_shape_and_x_tail() -> None:
  """Bucketed 1ubq keeps the golden shapes; past the rung, tokens are X (20) and logits 0.

  The opt-out golden samples every padded position (tokens and logits are nonzero
  out to 511), so masked positions inside the rung (76-127) are sampled too. Only
  the re-pad past the rung is fixed: X, not token 0, which is Alanine.
  """
  pytest.importorskip("aminx")
  from aminx.host.runner import sample  # noqa: PLC0415

  meta = json.loads((_GOLDEN_DIR / "optout_golden.json").read_text(encoding="utf-8"))
  case = next(item for item in meta["cases"] if item["name"] == "1ubq")
  result = sample(
    inputs=[str(_DATA / "1ubq.pdb")],
    num_samples=case["num_samples"],
    temperature=case["temperature"],
    random_seed=case["random_seed"],
    max_length=meta["max_length"],
  )
  sequences = np.asarray(result["sequences"])
  logits = np.asarray(result["logits"])
  assert list(sequences.shape) == meta["arrays"]["1ubq__sample__sequences"]
  assert list(logits.shape) == meta["arrays"]["1ubq__sample__logits"]
  assert np.all(sequences[..., _UBQ_RUNG:] == 20)
  assert np.all(logits[..., _UBQ_RUNG:, :] == 0)
  assert np.any(logits[..., :_UBQ_SPAN, :] != 0)
  real = sequences[..., :_UBQ_SPAN]
  assert real.size > 0
  assert int(real.min()) >= 0
  assert int(real.max()) <= 20


def _score_pair(
  pdb: Path,
  *,
  backbone_noise: float,
  chain_id: str | None = None,
  sequences: tuple[str, str] = ("A" * 10, "A" * 10),
) -> tuple[np.ndarray, np.ndarray]:
  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  def once(length_bucketing: bool, sequence: str) -> np.ndarray:
    kwargs: dict[str, object] = {}
    if chain_id is not None:
      kwargs["chain_id"] = chain_id
    result = score(
      ScoringSpecification(
        inputs=str(pdb),
        checkpoint_id=_CHECKPOINT,
        sequences_to_score=[sequence],
        backbone_noise=backbone_noise,
        random_seed=42,
        max_length=512,
        length_bucketing=length_bucketing,
        return_logits=False,
        **kwargs,
      ),
    )
    return np.asarray(result["scores"])

  return once(True, sequences[0]), once(False, sequences[1])


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize(
  ("pdb_name", "chain_id"),
  [
    ("1ubq.pdb", "A"),
    ("5awl.pdb", "A"),
    # 1BC8.cif is the multi-chain file in tests/data (protein plus DNA, chains A-H).
    ("1BC8.cif", None),
    # Chain C is the gapped protein: span 113, 93 valid residues.
    ("1BC8.cif", "C"),
  ],
  ids=["1ubq", "5awl", "1bc8-multichain", "1bc8-gapped"],
)
def test_g_invariance_score_noise0(pdb_name: str, chain_id: str | None) -> None:
  pytest.importorskip("aminx")
  pdb = _DATA / pdb_name
  if not pdb.is_file():
    pytest.skip(f"structure fixture missing: {pdb}")
  bucketed, opted_out = _score_pair(pdb, backbone_noise=0.0, chain_id=chain_id)
  np.testing.assert_allclose(bucketed, opted_out, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_averaged_score_path_is_not_bucketed() -> None:
  """average_node_features=True keeps the padded length (S8 v1 non-goal), so it is bit-identical.

  This path draws backbone noise at the coordinate shape, so a trim would change the
  noise draw and the score. Before the fix it was trimmed despite being documented as
  unbucketed. One noise level: with two or more, this path fails its R3 check under
  vmap (TracerBoolConversionError) with bucketing on or off alike (aminx debt #2517).
  """
  pytest.importorskip("aminx")
  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  def once(length_bucketing: bool) -> np.ndarray:
    result = score(
      ScoringSpecification(
        inputs=str(_DATA / "1ubq.pdb"),
        checkpoint_id=_CHECKPOINT,
        sequences_to_score=["A" * 10],
        average_node_features=True,
        backbone_noise=0.1,
        random_seed=42,
        max_length=512,
        length_bucketing=length_bucketing,
        return_logits=False,
      ),
    )
    return np.asarray(result["scores"])

  assert np.array_equal(once(True), once(False))


@pytest.mark.slow
@pytest.mark.requires_weights
def test_g_invariance_comparison_can_fail() -> None:
  """Negative control: the rtol=1e-6 comparison detects a real change.

  Backbone noise is not a usable control: the plain score path forwards no noise
  level (runner.score, encode-once path #147), so 1ubq scores 4.4554 at noise 0,
  0.1 and 1.0 alike (aminx debt #2509). Scoring a different sequence through the
  bucketed path does change the score, and the comparison must catch it.
  """
  pytest.importorskip("aminx")
  bucketed, opted_out = _score_pair(
    _DATA / "1ubq.pdb", backbone_noise=0.0, chain_id="A", sequences=("G" * 10, "A" * 10)
  )
  assert not np.allclose(bucketed, opted_out, rtol=1e-6)


def _sample_controls(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
  from aminx.host.runner import sample  # noqa: PLC0415
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  result = sample(
    SamplingSpecification(
      inputs=str(_DATA / "1ubq.pdb"),
      chain_id="A",
      checkpoint_id=_CHECKPOINT,
      num_samples=1,
      temperature=0.1,
      random_seed=0,
      max_length=512,
      **kwargs,
    ),
  )
  return np.asarray(result["sequences"]), np.asarray(result["logits"])


@pytest.mark.slow
@pytest.mark.requires_weights
def test_g_controls_fixed_position_and_bias() -> None:
  pytest.importorskip("aminx")
  padded = 512
  fixed_mask = np.zeros(padded, dtype=np.float32)
  fixed_tokens = np.zeros(padded, dtype=np.int32)
  fixed_mask[0] = 1.0
  fixed_tokens[0] = 7
  sequences, _logits = _sample_controls(fixed_mask=fixed_mask, fixed_tokens=fixed_tokens)
  assert np.all(sequences[..., 0] == 7)

  plain_seq, plain_logits = _sample_controls()
  bias = np.zeros((padded, 21), dtype=np.float32)
  bias[0, 3] = 50.0
  _biased_seq, biased_logits = _sample_controls(bias=bias)
  assert not np.array_equal(biased_logits, plain_logits)

  # The 1ubq rung is 128, so an index at or past 128 is sliced off. 200 is past the
  # span (76) and past the rung, and still inside the padded length.
  tail_bias = np.zeros((padded, 21), dtype=np.float32)
  tail_bias[200, 3] = 50.0
  tail_seq, tail_logits = _sample_controls(bias=tail_bias)
  assert np.array_equal(tail_seq, plain_seq)
  assert np.array_equal(tail_logits, plain_logits)


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.xfail(
  raises=ValueError,
  strict=True,
  reason=(
    "aminx debt #2508: proxide concatenate_proteins_for_inter_mode cannot concatenate "
    "the 0-d source-path leaf, so runner inter mode fails before sampling. The inter "
    "guard itself is covered in test_length_bucketing_helper.py."
  ),
)
def test_g_controls_inter_is_not_trimmed() -> None:
  """Ties expressed as tied_positions auto/direct require pass_mode='inter'.

  ``inter`` disables length bucketing, so an inter run is not trimmed: it matches
  the same call with length_bucketing=False.
  """
  pytest.importorskip("aminx")
  inter_seq, inter_logits = _sample_controls(pass_mode="inter")
  opt_seq, opt_logits = _sample_controls(pass_mode="inter", length_bucketing=False)
  assert np.array_equal(inter_seq, opt_seq)
  assert np.array_equal(inter_logits, opt_logits)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_g_compile_two_rungs_trace_decode_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Three lengths in two rungs (5awl -> 64, 1ubq and a 100-residue crop -> 128).

  The counter wraps ``InferencePlan.decode_fn``. ``decode`` is ``eqx.filter_jit``,
  so its Python body runs only while a new signature is being traced, and that body
  calls ``decode_fn`` once. A cache hit does not re-enter Python, so the count is
  the number of decode traces rather than the number of executions. ``batch_size=1``
  puts each structure in its own batch; the two structures that share rung 128
  must not trace a second time.
  """
  pytest.importorskip("aminx")
  import aminx.host.runner as runner_mod  # noqa: PLC0415
  from aminx.host.runner import sample  # noqa: PLC0415
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  cropped = tmp_path / "1mbn_100.pdb"
  _crop_first_residues(_DATA / "1mbn.pdb", cropped, 100)
  counts = {"n": 0}
  original = runner_mod.make_inference_plan

  def counting_make(*args: object, **kwargs: object) -> object:
    plan = original(*args, **kwargs)
    orig_fn = plan.decode_fn

    def counting_fn(*fn_args: object, **fn_kwargs: object) -> object:
      counts["n"] += 1
      return orig_fn(*fn_args, **fn_kwargs)

    return plan.with_decode_fn(counting_fn)

  monkeypatch.setattr(runner_mod, "make_inference_plan", counting_make)
  sample(
    SamplingSpecification(
      inputs=[str(_DATA / "5awl.pdb"), str(_DATA / "1ubq.pdb"), str(cropped)],
      chain_id="A",
      checkpoint_id=_CHECKPOINT,
      num_samples=1,
      temperature=0.1,
      random_seed=0,
      max_length=512,
      batch_size=1,
    ),
  )
  assert counts["n"] == 2
