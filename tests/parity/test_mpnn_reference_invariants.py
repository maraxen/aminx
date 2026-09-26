"""Orchestrator-owned MPNN invariants, each run against BOTH the reference and aminx (T11 step 1).

Literature-parity Constraint 1 (bathos-literature-parity): the decisive properties of the
method are re-derived here as runnable synthetic-ground-truth checks instead of being taken
from any agent's reconciliation. Every invariant asserts the property on the pinned
LigandMPNN reference (26ec57ac) AND on aminx, on a real fixture from the browser-validation
manifest. The injected-defect RED evidence for each test is recorded in the T11 verdict doc.

Heavy (marker ``parity_heavy``): needs ``REFERENCE_PATH`` and the reference weights.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.parity_heavy

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"

SMALL_FIXTURE = "6MRR"  # L=68, set B, protein-only
TIED_FIXTURE = "3HTN"  # L=416, set B, has k-NN-disjoint multi-member tie groups
MEMBRANE_FIXTURE = "6MRR"
LOG_PROB_BAR = 1e-4  # spec bars table, conditional/unconditional log-probs
CAUSAL_TOL = 1e-5  # "unchanged" tolerance for f32 re-runs of the same forward pass
MOVED_MIN = 1e-3  # "changed": well above f32 noise
OMIT_P_MAX = 1e-30
N_ONE_STEP_DRAWS = 4_000
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
X_INDEX = 20


# --------------------------------------------------------------------------------------
# Shared loaders (module-cached; one model load per session)
# --------------------------------------------------------------------------------------


@cache
def _lac() -> Any:
  import scripts.browser_validation.layer_a_common as lac

  return lac


@cache
def _lae() -> Any:
  import scripts.browser_validation.layer_a_exact as lae

  return lae


@cache
def _las() -> Any:
  import scripts.browser_validation.layer_a_sampling as las

  return las


@cache
def _fixture(name: str) -> dict[str, Any]:
  manifest = json.loads(_MANIFEST.read_text())
  fixtures = manifest["fixtures"] if isinstance(manifest, dict) else manifest
  for fixture in fixtures:
    if fixture["name"] == name:
      return fixture
  msg = f"fixture {name!r} not in {_MANIFEST}"
  raise KeyError(msg)


@cache
def _full_model() -> tuple[Any, Any, Any, Any]:
  return _lac().load_full_model("eqx")


@cache
def _batch(name: str) -> Any:
  return _lae().build_exact_batch(_fixture(name), None)


def _ref_score(batch: Any, *, use_sequence: bool) -> np.ndarray:
  _jax_model, pt_model, torch, _mu = _full_model()
  fd = _lae()._reference_feature_dict(torch, batch)  # noqa: SLF001
  return _lae()._score_reference(pt_model, torch, fd, use_sequence=use_sequence)  # noqa: SLF001


def _aminx_conditional(batch: Any) -> np.ndarray:
  return _lae()._score_aminx_conditional(_full_model()[0], batch)  # noqa: SLF001


def _with_token(batch: Any, position: int, token: int) -> Any:
  """`batch` with residue `position` set to MPNN token `token` on both sides."""
  aatype = batch.aatype_aminx.copy()
  aatype[position] = token
  seq_ref = batch.seq_ref.copy()
  seq_ref[position] = token
  return dataclasses.replace(batch, aatype_aminx=aatype, seq_ref=seq_ref)


def _other_token(token: int) -> int:
  return (int(token) + 7) % 20


# --------------------------------------------------------------------------------------
# 1. Autoregressive causality
# --------------------------------------------------------------------------------------


def test_invariant_1_ar_causality() -> None:
  """log p(i) is unchanged by mutating a residue decoded AFTER i and changed by one BEFORE."""
  batch = _batch(SMALL_FIXTURE)
  order = batch.decoding_order
  rank = np.empty(batch.length, dtype=np.int64)
  rank[order] = np.arange(batch.length)
  i = int(order[len(order) // 2])
  # Both mutated residues are k-NN neighbours of i, so either could reach i through an
  # edge; only the decoding order decides whether it may.
  _jax_model, pt_model, torch, _mu = _full_model()
  with torch.no_grad():
    _e, e_idx = pt_model.features(_lae()._reference_feature_dict(torch, batch))  # noqa: SLF001
  neighbours = [int(j) for j in e_idx.numpy()[0][i] if int(j) != i]
  j_before = next(j for j in neighbours if rank[j] < rank[i])
  j_after = next(j for j in neighbours if rank[j] > rank[i])

  for side, score in (
    ("reference", lambda b: _ref_score(b, use_sequence=True)),
    ("aminx", _aminx_conditional),
  ):
    base = score(batch)[i]
    after = score(_with_token(batch, j_after, _other_token(batch.seq_ref[j_after])))[i]
    before = score(_with_token(batch, j_before, _other_token(batch.seq_ref[j_before])))[i]
    d_after = float(np.abs(after - base).max())
    d_before = float(np.abs(before - base).max())
    assert d_after < CAUSAL_TOL, f"{side}: future residue {j_after} leaked into {i} ({d_after})"
    assert d_before > MOVED_MIN, f"{side}: past residue {j_before} invisible to {i} ({d_before})"


# --------------------------------------------------------------------------------------
# 2. Unconditional logits are sequence-invariant
# --------------------------------------------------------------------------------------


def _aminx_unconditional_with_sequence(batch: Any) -> np.ndarray:
  """aminx's unconditional path, HANDED the batch's sequence (it must not use it)."""
  import jax

  from aminx.inference import score_unconditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  kwargs = _lae()._aminx_bundle_kwargs(batch, sequence_one_hot=True)  # noqa: SLF001
  kwargs.pop("ar_mask")
  bundle, config = build_inference_bundle(mode="score_unconditional", **kwargs)
  logits = score_unconditional.kernel(
    _full_model()[0], jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  return np.asarray(jax.nn.log_softmax(logits, axis=-1))


def test_invariant_2_unconditional_sequence_invariant() -> None:
  batch = _batch(SMALL_FIXTURE)
  rng = np.random.default_rng(2)
  scrambled = dataclasses.replace(
    batch,
    aatype_aminx=rng.integers(0, 20, size=batch.length).astype(batch.aatype_aminx.dtype),
    seq_ref=rng.integers(0, 20, size=batch.length).astype(batch.seq_ref.dtype),
  )
  scrambled = dataclasses.replace(scrambled, aatype_aminx=scrambled.seq_ref.copy())
  for side, score in (
    ("reference", lambda b: _ref_score(b, use_sequence=False)),
    ("aminx", _aminx_unconditional_with_sequence),
  ):
    delta = float(np.abs(score(batch) - score(scrambled)).max())
    assert delta < CAUSAL_TOL, f"{side}: unconditional log-probs depend on the sequence ({delta})"


# --------------------------------------------------------------------------------------
# 3. Fixed-first decoding order
# --------------------------------------------------------------------------------------


def test_invariant_3_fixed_first_order() -> None:
  """Every fixed (chain_mask = 0) residue is decoded before every designable one."""
  las = _las()
  lane_batch = las.build_lane_batch(_fixture(SMALL_FIXTURE), "P07@1.0", None)
  fixed = (lane_batch.mask * lane_batch.chain_mask) == 0
  assert 0 < fixed.sum() < lane_batch.length

  # reference: the order its own sample() returns
  _jax_model, pt_model, torch, _mu = _full_model()
  for seed in range(3):
    _seq, _lp, ref_order, _randn = las.reference_sample_one(
      pt_model, torch, lane_batch, 100 + seed, temperature=1.0
    )
    rank = np.empty(lane_batch.length, dtype=np.int64)
    rank[ref_order] = np.arange(lane_batch.length)
    assert rank[fixed].max() < rank[~fixed].min(), (
      "reference: a designable residue precedes a fixed one"
    )

  # aminx: its native default schedule (`fixed_n_to_c`) is a designed DEVIATION (clause
  # decoding_order_default), so the rule is checked on the kernel path the validation
  # harness drives (host reference-formula order -> WaveScheduleBundle). Its semantic
  # consequence is asserted directly: the FIRST designable step already sees every fixed
  # residue, so changing one fixed token (even one far later in sequence index) moves that
  # step's logits, and the fixed position itself returns its fixed token.
  import jax

  _randn, order = las._draw_order_for(lane_batch, 300)  # noqa: SLF001
  first = int(order[np.isin(order, np.flatnonzero(~fixed))][0])
  jax_model = _full_model()[0]
  _ef, e_idx, _nf, _k = jax_model.features(
    jax.random.PRNGKey(0),
    jax.numpy.asarray(lane_batch.x4),
    jax.numpy.asarray(lane_batch.mask),
    jax.numpy.asarray(lane_batch.residue_index, dtype=jax.numpy.int32),
    jax.numpy.asarray(lane_batch.chain_index, dtype=jax.numpy.int32),
    backbone_noise=0.0,
  )
  # a fixed k-NN neighbour of `first` (so an edge exists), the latest one in sequence index
  far_fixed = max(int(j) for j in np.asarray(e_idx)[first] if fixed[int(j)] and int(j) != first)
  seq0, logits0 = las.aminx_sample_one(
    jax_model, lane_batch, order, jax.random.PRNGKey(3), temperature=1.0
  )
  seq_ref = lane_batch.seq_ref.copy()
  seq_ref[far_fixed] = _other_token(seq_ref[far_fixed])
  mutated = dataclasses.replace(lane_batch, seq_ref=seq_ref)
  seq1, logits1 = las.aminx_sample_one(
    jax_model, mutated, order, jax.random.PRNGKey(3), temperature=1.0
  )
  assert int(seq0[far_fixed]) == int(lane_batch.seq_ref[far_fixed]), (
    "aminx: fixed position not held"
  )
  assert int(seq1[far_fixed]) == int(seq_ref[far_fixed]), "aminx: fixed position not held"
  moved = float(np.abs(logits0[first] - logits1[first]).max())
  assert moved > MOVED_MIN, (
    f"aminx: first designable step {first} blind to fixed residue {far_fixed} ({moved})"
  )


# --------------------------------------------------------------------------------------
# 4. Omitted residues are never sampled; X is never sampled under the X-omit column
# --------------------------------------------------------------------------------------


def _omit_bias(length: int, *, x_column: bool) -> np.ndarray:
  bias = np.zeros((length, 21), dtype=np.float32)
  for aa in "CW":
    bias[:, ALPHABET.index(aa)] = -1e8
  if x_column:
    bias[:, X_INDEX] = -1e8
  return bias


def test_invariant_4_omit_and_x() -> None:
  las = _las()
  base = las.build_lane_batch(_fixture(SMALL_FIXTURE), "P07@1.0", None)
  omitted = [ALPHABET.index("C"), ALPHABET.index("W")]

  # reference: omit via bias (run.py formula), X hard-omitted by its own probs[:, :20] slice
  ref_batch = dataclasses.replace(base, reference_bias=_omit_bias(base.length, x_column=False))
  _jax_model, pt_model, torch, _mu = _full_model()
  randn, _order = las._draw_order_for(ref_batch, 400)  # noqa: SLF001
  fd = las._reference_feature_dict_lane(torch, ref_batch, np.stack([randn] * 8), 1.0, batch_size=8)  # noqa: SLF001
  with torch.no_grad():
    out = pt_model.sample(fd)
  ref_tokens = out["S"].numpy()
  probs = out["sampling_probs"].numpy()  # (B, L, 20), zero on fixed positions
  designable = (ref_batch.mask * ref_batch.chain_mask) > 0
  assert float(probs[:, designable][..., omitted].max()) < OMIT_P_MAX, (
    "reference: omitted AA has p >= 1e-30"
  )
  assert not np.isin(ref_tokens[:, designable], [*omitted, X_INDEX]).any(), (
    "reference: sampled an omitted AA or X"
  )

  # aminx: omit via bias, X via the -1e8 X column
  am_batch = dataclasses.replace(base, bias=_omit_bias(base.length, x_column=True))
  am_tokens = las.aminx_sample_batch(_full_model()[0], am_batch, 8, 400, temperature=1.0)
  assert not np.isin(am_tokens[:, designable], [*omitted, X_INDEX]).any(), (
    "aminx: sampled an omitted AA or X"
  )
  sampling = (
    am_batch.bias[None] + _aminx_teacher_forced(am_batch, am_tokens[0], _order)[None]
  ) / 1.0
  p = _softmax(sampling[0])
  assert float(p[designable][:, [*omitted, X_INDEX]].max()) < OMIT_P_MAX, (
    "aminx: omitted AA has p >= 1e-30"
  )


def _softmax(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  z = x - x.max(axis=-1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=-1, keepdims=True)


def _aminx_teacher_forced(batch: Any, tokens: np.ndarray, order: np.ndarray) -> np.ndarray:
  """aminx bias-free conditional logits for `tokens` under decoding `order` (L, 21)."""
  return np.asarray(_las().aminx_conditional_logits(_full_model()[0], batch, tokens, order))


# --------------------------------------------------------------------------------------
# 5. One-step distribution == softmax((l + b) / T)
# --------------------------------------------------------------------------------------


def test_invariant_5_one_step_distribution() -> None:
  temperature = 0.5
  las = _las()
  base = las.build_lane_batch(_fixture(SMALL_FIXTURE), "P07@1.0", None)
  rng = np.random.default_rng(5)
  bias = rng.normal(0.0, 1.0, size=(base.length, 21)).astype(np.float32)
  bias[:, X_INDEX] = -1e8  # keep X out on both sides so the 20-way comparison is exact

  # reference: sampling_probs must equal numpy softmax((l + b)/T) with l its own logits
  ref_batch = dataclasses.replace(base, reference_bias=bias)
  _jax_model, pt_model, torch, _mu = _full_model()
  randn, order = las._draw_order_for(ref_batch, 500)  # noqa: SLF001
  fd = las._reference_feature_dict_lane(torch, ref_batch, randn, temperature, batch_size=1)  # noqa: SLF001
  with torch.no_grad():
    out = pt_model.sample(fd)
  designable = np.flatnonzero((ref_batch.mask * ref_batch.chain_mask) > 0)
  l_ref = out["log_probs"].numpy()[0]  # log_softmax(logits): softmax is shift-invariant
  expected = _softmax((l_ref + bias) / temperature)[:, :20]
  expected = expected / expected.sum(axis=-1, keepdims=True)
  got = out["sampling_probs"].numpy()[0]
  delta = float(np.abs(got[designable] - expected[designable]).max())
  assert delta < 1e-5, f"reference: sampling distribution != softmax((l+b)/T) ({delta})"

  # aminx: schedule ONLY the first designable position (a supported partial schedule), draw
  # N times and compare the empirical frequencies to softmax((l+b)/T) with l the kernel's
  # own bias-free logits at that position.
  am_batch = dataclasses.replace(base, bias=bias)
  first = int(order[np.isin(order, designable)][0])
  freq, logits_first = _aminx_one_step_frequencies(am_batch, order, first, temperature)
  expected_am = _softmax((logits_first + bias[first]) / temperature)
  tv = 0.5 * float(np.abs(freq - expected_am).sum())
  # TV of an N-draw empirical 21-way distribution: E[TV] <= sqrt(K/N)/2 ~ 0.035 at N=4000
  assert tv < 0.06, f"aminx: one-step frequencies != softmax((l+b)/T) (TV={tv:.4f})"


def _aminx_one_step_frequencies(
  batch: Any, order: np.ndarray, position: int, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
  import equinox as eqx
  import jax
  import jax.numpy as jnp

  from aminx.inference import sample_autoregressive
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  las = _las()
  wave = las.wave_from_tie_groups_np(batch.tie_group_map, order)
  keep = np.asarray(
    [int(np.flatnonzero(np.asarray(wave.group_ids)[:, 0] == batch.tie_group_map[position])[0])]
  )
  wave = dataclasses.replace(
    wave,
    group_ids=wave.group_ids[keep],
    group_positions=wave.group_positions[keep],
    group_valid=wave.group_valid[keep],
    position_valid=wave.position_valid[keep],
  )
  kw: dict[str, Any] = {
    "coords": jnp.asarray(batch.x4),
    "mask": jnp.asarray(batch.mask),
    "residue_index": jnp.asarray(batch.residue_index, dtype=jnp.int32),
    "chain_index": jnp.asarray(batch.chain_index, dtype=jnp.int32),
    "chain_mask": jnp.asarray(batch.chain_mask),
    "bias": jnp.asarray(batch.bias),
    "fixed_mask": jnp.zeros(batch.length),
    "fixed_tokens": jnp.asarray(batch.seq_ref, dtype=jnp.int32),
    "tie_group_map": jnp.asarray(batch.tie_group_map),
    "wave": wave,
    "temperature": float(temperature),
    "mode": "sample",
  }
  bundle, config = build_inference_bundle(**kw)
  stage_set = make_stage_set()
  model = _full_model()[0]

  @eqx.filter_jit
  def draw(keys: Any) -> Any:
    def one(k: Any) -> Any:
      res = sample_autoregressive.kernel(model, k, bundle, config, stage_set, inference_only=True)
      return res.sequence[position], res.logits[position]

    return jax.vmap(one)(keys)

  counts = np.zeros(21)
  logits_first = None
  chunk = 200
  for c in range(N_ONE_STEP_DRAWS // chunk):
    keys = jax.random.split(jax.random.PRNGKey(55 + c), chunk)
    tokens, logits = draw(keys)
    counts += np.bincount(np.asarray(tokens), minlength=21)[:21]
    if logits_first is None:
      logits_first = np.asarray(logits[0])
  return counts / counts.sum(), logits_first


# --------------------------------------------------------------------------------------
# 6. Tied positions: identical tokens; fused logits == reference unit-weight sum (+ const)
# --------------------------------------------------------------------------------------


def test_invariant_6_tied_positions() -> None:
  las = _las()
  lae = _lae()
  fixture = _fixture(TIED_FIXTURE)
  lane_batch = las.build_lane_batch(fixture, "P09-s@1.0", None)
  assert lane_batch.groups, f"{TIED_FIXTURE}: no qualifying tie groups"
  groups = lane_batch.groups[:8]
  small = dataclasses.replace(
    lane_batch,
    groups=groups,
    tie_group_map=_tie_map(lane_batch.length, groups),
  )
  _jax_model, pt_model, torch, _mu = _full_model()

  ref_seq, _lp, _o, _r = las.reference_sample_one(pt_model, torch, small, 600, temperature=1.0)
  am_seq = las.aminx_sample_batch(_full_model()[0], small, 1, 600, temperature=1.0)[0]
  for side, seq in (("reference", ref_seq), ("aminx", am_seq)):
    for g in groups:
      assert len({int(seq[m]) for m in g}) == 1, f"{side}: tie group {g} got different tokens"

  rows = lae.row_p09(fixture, _batch(TIED_FIXTURE), _full_model(), "eqx")
  (row,) = [r for r in rows if r["path"] == "P09.log_probs"]
  assert row["value"] <= 10 * LOG_PROB_BAR, (
    f"fused tied log-probs differ from the reference unit-weight zero-bias sum: {row['value']}"
  )


def _tie_map(length: int, groups: list[list[int]]) -> np.ndarray:
  from tests.parity.test_full_model_parity import _build_tie_group_map

  return _build_tie_group_map(length, groups).astype(np.int64)


# --------------------------------------------------------------------------------------
# 7. Membrane label channels are live
# --------------------------------------------------------------------------------------


def test_invariant_7_membrane_labels_live() -> None:
  import jax
  import jax.numpy as jnp

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  lac, lae = _lac(), _lae()
  models = lac.load_soluble_membrane_models("eqx")
  batch = _batch(MEMBRANE_FIXTURE)
  torch = models.torch
  for kind, pt_model in (
    ("membrane_per_residue", models.pt_membrane_per_residue),
    ("membrane_global", models.pt_membrane_global),
  ):
    jax_model = lac.jax_model_for_source(models, "eqx", kind)
    ref_lp, am_lp = [], []
    for label in (0, 1, 2):
      labels = np.full(batch.length, label, dtype=np.int64)
      fd = lae._reference_feature_dict(torch, batch)  # noqa: SLF001
      fd["membrane_per_residue_labels"] = torch.from_numpy(labels[None].copy())
      ref_lp.append(lae._score_reference(pt_model, torch, fd, use_sequence=True))  # noqa: SLF001
      kw = lae._aminx_bundle_kwargs(batch, sequence_one_hot=True)  # noqa: SLF001
      kw["physics_features"] = jax.nn.one_hot(jnp.asarray(labels), 3)
      bundle, config = build_inference_bundle(mode="score_conditional", **kw)
      logits = score_conditional.kernel(
        jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
      )
      am_lp.append(np.asarray(jax.nn.log_softmax(logits, axis=-1)))
    for side, lps in (("reference", ref_lp), ("aminx", am_lp)):
      for a in range(3):
        for b in range(a + 1, 3):
          delta = float(np.abs(lps[a] - lps[b]).max())
          assert delta > 10 * LOG_PROB_BAR, (
            f"{side}/{kind}: labels {a} vs {b} barely move log-probs ({delta})"
          )


# --------------------------------------------------------------------------------------
# 8. k-NN neighbour set == numpy brute force on no-tie residues
# --------------------------------------------------------------------------------------


def _brute_force_knn(ca: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
  d = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1) + 1e-6)
  m2 = mask[:, None] * mask[None, :]
  d_adj = d + (1.0 - m2) * d.max()
  return np.argsort(d_adj, axis=-1, kind="stable")[:, :k]


def test_invariant_8_knn_brute_force() -> None:
  import jax

  fixture = _fixture(SMALL_FIXTURE)
  batch = _batch(SMALL_FIXTURE)
  k = min(48, batch.length)
  expected = _brute_force_knn(
    batch.x4[:, 1, :].astype(np.float64), batch.mask.astype(np.float64), k
  )
  near_tie = set((fixture.get("near_tie_residues") or {}).get(str(k)) or [])
  rows = [i for i in range(batch.length) if batch.mask[i] > 0 and i not in near_tie]
  assert len(rows) > batch.length // 2

  jax_model, pt_model, torch, _mu = _full_model()
  fd = _lae()._reference_feature_dict(torch, batch)  # noqa: SLF001
  with torch.no_grad():
    _e, ref_idx = pt_model.features(fd)
  ref_idx = ref_idx.numpy()[0][:, :k]
  _ef, am_idx, _nf, _key = jax_model.features(
    jax.random.PRNGKey(0),
    jax.numpy.asarray(batch.x4),
    jax.numpy.asarray(batch.mask),
    jax.numpy.asarray(batch.residue_index, dtype=jax.numpy.int32),
    jax.numpy.asarray(batch.chain_index, dtype=jax.numpy.int32),
    backbone_noise=0.0,
  )
  am_idx = np.asarray(am_idx)[:, :k]
  for side, idx in (("reference", ref_idx), ("aminx", am_idx)):
    bad = [i for i in rows if set(idx[i].tolist()) != set(expected[i].tolist())]
    assert not bad, (
      f"{side}: neighbour set != brute force at {len(bad)} no-tie residues, e.g. {bad[:5]}"
    )
