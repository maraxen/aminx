r"""Layer (a) sampling-tier engine (T8): teacher-forced exact tier + IUT statistical tier.

Companion to `layer_a_exact.py` (T7): where that module compares deterministic
log-probs/NLL on real fixtures, this module compares SAMPLED sequences across five
lanes -- P07@T0.1, P07@T1.0, P08, P09-s, P11-s -- at two tiers per the bars table's
"Teacher-forced per-step log-probs" and "Sampled sequences" rows:

- **(a) Teacher-forced.** The reference's OWN sampled sequence and decoding order are
  fed into aminx conditional decoding (`score_conditional.kernel`) with the matching
  AR mask; per-step log-probs are compared (`tf_max_abs`). For P09-s the compared
  quantity is the FUSED per-group distribution (`p09_fused_tf_max_abs`), per R3-C6(2).
- **(b) Statistical.** Each arm draws `2n` sequences (`A1, A2` aminx; `R1, R2`
  reference); per lane, BOTH a recovery TOST (delta=0.01) and an excess-JS bootstrap
  one-sided 95% upper bound (`aminx.parity.compare.excess_js_upper`, reused -- never
  reimplemented) against the committed margin `m_l` must pass (IUT, no multiplicity
  adjustment: `aminx.parity.compare.iut_equivalent`).

This module is the ENGINE only (no CLI) -- `layer_a_sampling_calibrate.py` (T8b, set
A) and `layer_a_sampling_validate.py` (set B) are its two callers, mirroring T7/T7b's
split. It reuses `layer_a_exact.py` (`lae`) and `layer_a_common.py` (`lac`) rather than
re-deriving fixture parsing, model loading, order formulas, tie-group helpers, or the
X/omit-bias convention -- see each function's docstring for its specific reuse.

**Sequence-token convention.** Every lane feeds `af_to_mpnn`-converted MPNN-alphabet
tokens on both sides via `lae.parse_canonical_fixture`/`lae.build_exact_batch` (never
aminx's raw AlphaFold-order `aatype`) -- see `layer_a_exact`'s module docstring for why
this is load-bearing. `X_INDEX = 20` in this alphabet.

**Order per draw `i` (spec "T8 step 1").** `randn_i` is drawn from a seeded RNG; the
host computes `order_i = argsort((mask*chain_mask + 1e-4) * |randn_i|)`
(`lac.reference_formula_order`, reused). P07, P08 and P11-s run the reference BATCHED
(one `randn` row per draw, `model_utils.py:262-263`); P09-s runs the reference at
batch 1 per draw (its symmetric branch uses `decoding_order[0]` for the whole batch,
`model_utils.py:358`) and flattens tie groups at first occurrence internally -- the
reference's own RETURNED `decoding_order` is used downstream for P09-s (verified
against the host formula for the non-P09-s lanes; row_p09's own precedent in
`layer_a_exact.py` does the same for the scoring lane). aminx runs
`inference.sample_autoregressive.kernel` with
`WaveScheduleBundle.from_tie_groups(tie_group_map, order_i)` -- verified end-to-end
against a real fixture during this task's own development (5L33, P07 lane): the
resulting `SampleResult.sequence` is fully drawn (no `-1` UNDRAWN_TOKEN) and the
X-omit bias column correctly hard-omits X. `fixed_mask = 1 - mask*chain_mask` and
`fixed_tokens = S_true` (`seq_ref`), mirroring the reference's own `chain_mask`
convention (LigandMPNN `model_utils.py:334`: `S_t = S_t*chain_mask_t +
S_true_t*(1-chain_mask_t)`).

**P14 (packer sampling lane) is NOT IMPLEMENTED.** See `P14_SAMPLING_NOT_IMPLEMENTED_REASON`:
`grep -rln "vonmises\|von_mises" src/ tests/` returns ZERO matches anywhere in this
repository -- there is no chi-angle sampling routine for the packer at all (not even a
synthetic/parameter-only one), which is a strictly stronger gap than T7a's finding
(that only a real-structure -> PackerBundle adapter is missing). Nothing exists to draw
the 10,000 chi samples from, so this lane cannot be faked into existence.

**Sized fusion control (P09-s, R3-C6(3)) implementation note.** Rather than
intercepting `TieGroupProductOfExperts` inside the AR kernel (which fuses internally,
mid-decode), this module calls `aminx.inference.logits.TieGroupProductOfExperts`
DIRECTLY (the exact same registered strategy `make_stage_set()` wires as
`tie_group_fuse` by default) on per-member RAW logits obtained from an UNGATED
`score_conditional` call (`tie_group_map = arange(L)`, i.e. no fusion applied by the
kernel), scaling the LAST group member's logit row by `(1 + eps)` before re-fusing.
This reuses the model's own fusion function unmodified rather than re-deriving its
math, and needs no kernel-internal hook.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_a_exact as lae  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants (Pre-registered layer-(a) bars / Sampling statistics / P09-s restrictions)
# --------------------------------------------------------------------------------------

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
X_INDEX = 20  # spec: "X is index 20 in MPNN order"
ALANINE_INDEX = 0

TF_BAR = 1e-4  # bars table: "Teacher-forced per-step log-probs ... TOL ... <= 1e-4 nats"
RECOVERY_DELTA = 0.01  # bars table: "recovery TOST delta = 0.01"
ALPHA = 0.05
N_REQUIRED_FLOOR = 1500
DEFAULT_N_BOOT = 1000
NULL_REPLICATES = 20
NULL_REPLICATES_PASS_FLOOR = 18  # ">= 18/20"
POSCTL_MIN_DETECTED = 18  # ">= 18/20"
NEGCTL_MAX_FP = 3  # "<= 3/20"
MAX_N_DOUBLINGS = 3  # bounded: full doubling-until-18/20 is a titanix-only cost

# spec "Lane temperatures (pre-registered, R2-C11)".
LANE_KEYS: tuple[str, ...] = ("P07@0.1", "P07@1.0", "P08@1.0", "P09-s@1.0", "P11-s@1.0")
DEFAULT_LANE_TEMPERATURES: dict[str, float] = {
  "P07@0.1": 0.1,
  "P07@1.0": 1.0,
  "P08@1.0": 1.0,
  "P09-s@1.0": 1.0,
  "P11-s@1.0": 1.0,
}
MARGIN_TEMPERATURE_RATIO = 1.05  # "T vs 1.05*T" margin rule

_LANE_BASE: dict[str, str] = {
  "P07@0.1": "P07",
  "P07@1.0": "P07",
  "P08@1.0": "P08",
  "P09-s@1.0": "P09-s",
  "P11-s@1.0": "P11-s",
}

OMIT_AA_CHARS = "CW"  # spec "the run.py omit formula for omit_AA='CW'"
P08_PER_RESIDUE_OMIT_POSITION = 0  # "plus one per-residue omit" -- fixed, documented position
P08_PER_RESIDUE_OMIT_CHAR = "D"
P07_FIXED_FRACTION = 0.20  # "fixed positions on 20% of residues"

DEFAULT_BETA = 5.0  # smoke-path default; run_full searches BETA_CANDIDATES for real
BETA_CANDIDATES: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
DEFAULT_FUSION_CTRL_EPS = 0.1  # smoke-path default; run_full searches FUSION_EPS_CANDIDATES
FUSION_EPS_CANDIDATES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SIZING_RATIO_RANGE: tuple[float, float] = (2.0, 10.0)  # "[2x, 10x] the bar"

# A finite "could not compute" sentinel, never literal inf/nan -- pinned bathos
# (0.13.0a4) renders result fields as DuckDB SQL literals and a bare `inf`/`nan` fails
# to bind (measured on titanix, 2026-09-25: a clean run reported catalog outcome=error
# for exactly this reason). 1e18 is comfortably outside any real margin/ratio value.
UNCOMPUTED_SENTINEL = 1.0e18

_SEED_BASE = 20260924  # arbitrary fixed base, distinct from layer_a_exact's own base


def _seed_for(name: str) -> int:
  """Identity-derived seed: a PURE function of `name` alone, never of call order or of
  how many earlier draws/stages ran. This is what makes calibrate's checkpoint/resume
  (T10g) bit-identical -- a resumed unit re-derives the SAME seed a never-interrupted
  run would have used for that same identity tag.

  T10g fix (pre-registration note, made BEFORE any run used it): this used to be
  `hash(name) % 10_000` -- Python's builtin `str.__hash__` is SipHash-randomized per
  PROCESS (`PYTHONHASHSEED` defaults to `random`), so the SAME tag produced a
  DIFFERENT seed in every fresh interpreter (measured: three `python3 -c` invocations
  of `hash('P07@1.0testA1') % 10000` returned 9142, 9751, 8451). Within one
  uninterrupted process this was internally self-consistent (one hash seed for the
  whole run), but a checkpoint/resume by construction starts a NEW process -- so
  every unit computed after a resume would have silently drawn from different
  randomness than an uninterrupted run, breaking the "resuming gives the SAME
  results" requirement even for units that were never checkpointed. Replaced with
  `hashlib.sha256`, which is stable across processes/machines by construction. No
  caller's numeric threshold, grid, replicate count, or n changes -- only the
  seed-derivation primitive underneath an already identity-shaped `name -> int`
  mapping.
  """
  digest = hashlib.sha256(name.encode("utf-8")).digest()
  return (_SEED_BASE + (int.from_bytes(digest[:8], "big") % 10_000)) & 0xFFFFFFFF


def _mpnn_index(char: str) -> int:
  return MPNN_ALPHABET.index(char)


def _omit_indices(chars: str) -> list[int]:
  return [_mpnn_index(c) for c in chars]


# --------------------------------------------------------------------------------------
# Bias construction (X-omit column, P08's omit_AA formula)
# --------------------------------------------------------------------------------------


def _x_omit_bias(length: int) -> np.ndarray:
  """`(L, 21)` bias with `-1e8` at `X_INDEX` -- applied on every lane, both arms (host side;
  the reference achieves the equivalent hard-omit by truncating `probs[:, :20]` before
  `multinomial`, `model_utils.py:320-323`, so this column is aminx-arm-only, matching the
  task text: "Every lane applies the X-omit bias column (-1e8 at index 20) on the aminx
  arm.")."""
  bias = np.zeros((length, 21), dtype=np.float32)
  bias[:, X_INDEX] = -1e8
  return bias


def _omit_aa_bias(length: int) -> np.ndarray:
  """P08's bias: `run.py`'s `omit_AA` formula (`-1e8 * omit_AA[None,None,:]`, `run.py:404-408`)
  for `OMIT_AA_CHARS`, PLUS one per-residue omit at `P08_PER_RESIDUE_OMIT_POSITION`, PLUS the
  X-omit column (aminx-arm bias only; see `_x_omit_bias`)."""
  bias = np.zeros((length, 21), dtype=np.float32)
  for idx in _omit_indices(OMIT_AA_CHARS):
    bias[:, idx] = -1e8
  if P08_PER_RESIDUE_OMIT_POSITION < length:
    bias[P08_PER_RESIDUE_OMIT_POSITION, _mpnn_index(P08_PER_RESIDUE_OMIT_CHAR)] = -1e8
  bias[:, X_INDEX] = -1e8
  return bias


def _reference_omit_aa_bias(length: int) -> np.ndarray:
  """The matching REFERENCE-side bias (no X-omit column -- the reference hard-omits X by
  slicing `probs[:, :20]`, never via bias; see `_x_omit_bias`'s docstring)."""
  bias = np.zeros((length, 21), dtype=np.float32)
  for idx in _omit_indices(OMIT_AA_CHARS):
    bias[:, idx] = -1e8
  if P08_PER_RESIDUE_OMIT_POSITION < length:
    bias[P08_PER_RESIDUE_OMIT_POSITION, _mpnn_index(P08_PER_RESIDUE_OMIT_CHAR)] = -1e8
  return bias


def _chain_mask_fixed_fraction(
  length: int, seed: int, frac: float = P07_FIXED_FRACTION
) -> np.ndarray:
  """P07: chain_mask with `frac` of residues fixed (0.0), the rest designable (1.0)."""
  rng = np.random.default_rng(seed)
  n_fixed = max(1, round(length * frac)) if length > 1 else 0
  chain_mask = np.ones(length, dtype=np.float32)
  if n_fixed:
    idx = rng.choice(length, size=n_fixed, replace=False)
    chain_mask[idx] = 0.0
  return chain_mask


# --------------------------------------------------------------------------------------
# LaneBatch -- per-lane geometry, built ONCE from lae.build_exact_batch/parse_canonical_fixture
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LaneBatch:
  """Per-lane sampling inputs, sharing the same real-fixture geometry every exact-tier
  row uses (`lae.build_exact_batch`, never reparsed here)."""

  lane: str
  fixture_name: str
  length: int
  x4: np.ndarray
  atom37: np.ndarray
  atom37_mask: np.ndarray
  seq_ref: np.ndarray  # S_true, MPNN-alphabet tokens (af_to_mpnn'd, cross-checked)
  mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  chain_mask: np.ndarray
  fixed_mask: np.ndarray
  bias: np.ndarray  # aminx-side (L, 21), includes X-omit column
  reference_bias: np.ndarray  # reference-side (L, 21), no X-omit column
  tie_group_map: np.ndarray  # (L,) int64; arange(L) unless P09-s
  groups: list[list[int]] | None  # only for P09-s
  use_side_chain_context: bool
  comparison_positions: np.ndarray  # (n_pos,) int64: designable positions, or tie members for P09-s


def build_lane_batch(fixture: dict[str, Any], lane: str, data_utils_module: Any) -> LaneBatch:
  """Build the per-lane geometry for `fixture` (real structure, `lae.build_exact_batch` reused)."""
  base = lae.build_exact_batch(fixture, data_utils_module)
  length = base.length
  lane_base = _LANE_BASE[lane]

  groups: list[list[int]] | None = None
  tie_group_map = np.arange(length, dtype=np.int64)
  use_sc = False
  bias = _x_omit_bias(length)
  reference_bias = np.zeros((length, 21), dtype=np.float32)

  if lane_base == "P07":
    chain_mask = _chain_mask_fixed_fraction(length, _seed_for(fixture["name"] + lane) + 1)
  elif lane_base == "P08":
    chain_mask = np.ones(length, dtype=np.float32)
    bias = _omit_aa_bias(length)
    reference_bias = _reference_omit_aa_bias(length)
  elif lane_base == "P09-s":
    groups = lae._qualifying_groups(fixture)  # noqa: SLF001 (reuse, not reimplement)
    if groups:
      lae._reverify_knn_disjoint(base, groups)  # noqa: SLF001 -- exit 2 on any violation (R2-C3)
      from tests.parity.test_full_model_parity import _build_tie_group_map

      tie_group_map = _build_tie_group_map(length, groups).astype(np.int64)
    chain_mask = np.ones(length, dtype=np.float32)  # "no mixed fixed/designable groups"
  elif lane_base == "P11-s":
    chain_mask = np.ones(length, dtype=np.float32)
    chain_mask[::2] = 0.0  # partial fixed, mirrors row_p11's exact-tier wiring
    use_sc = True
  else:
    msg = f"unknown lane {lane!r}"
    raise ValueError(msg)

  fixed_mask = 1.0 - base.mask * chain_mask

  if lane_base == "P09-s":
    comparison_positions = (
      np.asarray(sorted({m for g in groups for m in g}), dtype=np.int64)
      if groups
      else np.asarray([], dtype=np.int64)
    )
  else:
    comparison_positions = np.where((base.mask > 0) & (chain_mask > 0))[0].astype(np.int64)

  return LaneBatch(
    lane=lane,
    fixture_name=fixture["name"],
    length=length,
    x4=base.x4,
    atom37=base.atom37,
    atom37_mask=base.atom37_mask,
    seq_ref=base.seq_ref,
    mask=base.mask,
    residue_index=base.residue_index,
    chain_index=base.chain_index,
    chain_mask=chain_mask,
    fixed_mask=fixed_mask,
    bias=bias,
    reference_bias=reference_bias,
    tie_group_map=tie_group_map,
    groups=groups,
    use_side_chain_context=use_sc,
    comparison_positions=comparison_positions,
  )


def full_model_bundle_for_lane(lane: str, weight_source: str) -> tuple[Any, Any, Any, Any]:
  """P11-s uses the ligand/side-chain-context checkpoint pair (`lac.load_sidechain_context_models`,
  always pinned to `weight_source="eqx"` per that loader's own docstring); every other lane uses
  the full-model pair (`lac.load_full_model`, reused)."""
  if _LANE_BASE[lane] == "P11-s":
    reference_model, aminx_model = lac.load_sidechain_context_models(
      weight_source, use_side_chain_context=True
    )
    torch = __import__("torch")
    return aminx_model, reference_model, torch, None
  return lac.load_full_model(weight_source)


# --------------------------------------------------------------------------------------
# Reference feature dict, order draw, single-sequence reference sample
# --------------------------------------------------------------------------------------


def _reference_feature_dict_lane(
  torch: Any,
  batch: LaneBatch,
  randn: np.ndarray,
  temperature: float,
  *,
  batch_size: int,
) -> dict[str, Any]:
  randn_arr = randn if randn.ndim == 2 else randn[None]
  actx = 16
  fd: dict[str, Any] = {
    "X": torch.from_numpy(batch.x4[None].copy()),
    "S": torch.from_numpy(batch.seq_ref[None].astype(np.int64).copy()),
    "mask": torch.from_numpy(batch.mask[None].copy()),
    "chain_mask": torch.from_numpy(batch.chain_mask[None].copy()),
    "R_idx": torch.from_numpy(batch.residue_index[None].copy()),
    "chain_labels": torch.from_numpy(batch.chain_index[None].copy()),
    "randn": torch.from_numpy(np.ascontiguousarray(randn_arr)),
    "bias": torch.from_numpy(batch.reference_bias[None].copy()),
    "temperature": float(temperature),
    "batch_size": batch_size,
    "symmetry_residues": [list(g) for g in batch.groups] if batch.groups else [[]],
    "symmetry_weights": [[1.0] * len(g) for g in batch.groups] if batch.groups else [[]],
    # P11-s (ligand/side-chain-context) fields, harmless zeros for every other lane's
    # plain ProteinMPNN class (mirrors layer_a_exact.row_p11's own fd construction).
    "Y": torch.zeros((1, batch.length, actx, 3), dtype=torch.float32),
    "Y_t": torch.zeros((1, batch.length, actx), dtype=torch.int32),
    "Y_m": torch.zeros((1, batch.length, actx), dtype=torch.int32),
    "xyz_37": torch.from_numpy(batch.atom37[None].copy()),
    "xyz_37_m": torch.from_numpy(batch.atom37_mask[None].copy()),
  }
  return fd


def _draw_order_for(batch: LaneBatch, seed_i: int) -> tuple[np.ndarray, np.ndarray]:
  """`(randn_i, order_i)` from the reference-formula host order (`lac.reference_formula_order`,
  reused), seeded per draw `i`."""
  return lac.reference_formula_order(batch.mask, batch.chain_mask, seed=seed_i)


def reference_sample_one(
  pt_model: Any, torch: Any, batch: LaneBatch, seed_i: int, *, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """One reference `.sample()` call at batch 1. Returns `(S, log_probs, decoding_order, randn)`.

  For non-P09-s lanes, asserts the reference's own returned `decoding_order` equals the
  host formula (spec T8 step 1: "asserts it equals the reference's returned
  decoding_order"). P09-s's symmetric branch reorders/flattens internally
  (`model_utils.py:357-368`) so no such equality is asserted there -- its RETURNED order
  is authoritative, matching `layer_a_exact.row_p09`'s precedent.
  """
  randn, host_order = _draw_order_for(batch, seed_i)
  fd = _reference_feature_dict_lane(torch, batch, randn, temperature, batch_size=1)
  with torch.no_grad():
    out = pt_model.sample(fd)
  seq = out["S"].numpy()[0]
  log_probs = out["log_probs"].numpy()[0]
  decoding_order = out["decoding_order"].numpy()[0]
  if batch.groups is None:
    if not np.array_equal(decoding_order, host_order):
      msg = (
        f"{batch.fixture_name}/{batch.lane}: reference decoding_order disagrees with the "
        "host reference-formula order (non-P09-s lane; these must match by construction)"
      )
      raise AssertionError(msg)
  return seq, log_probs, decoding_order, randn


def reference_sample_batch(
  pt_model: Any, torch: Any, batch: LaneBatch, n: int, seed_base: int, *, temperature: float
) -> np.ndarray:
  """`n` reference draws, FULL length, shape `(n, L)` -- NOT pre-restricted to
  `batch.comparison_positions` (callers that need the restricted view call `restrict_to_comparison`
  themselves; token/X-omission counting needs the FULL sequence, see V4).

  P09-s (symmetric branch) runs at batch 1 per draw, looped (spec "Pooling, bootstrap
  and compute budget": "only P09-s ... runs at batch 1"); every other lane runs the
  reference BATCHED, one `randn` row per draw, in ONE `.sample()` call.
  """
  if batch.groups is not None:
    rows = [
      reference_sample_one(pt_model, torch, batch, seed_base + i, temperature=temperature)[0]
      for i in range(n)
    ]
    return np.stack(rows, axis=0)
  randn_rows = np.stack([_draw_order_for(batch, seed_base + i)[0] for i in range(n)], axis=0)
  fd = _reference_feature_dict_lane(torch, batch, randn_rows, temperature, batch_size=n)
  with torch.no_grad():
    out = pt_model.sample(fd)
  return out["S"].numpy()


def restrict_to_comparison(seqs: np.ndarray, batch: LaneBatch) -> np.ndarray:
  """`(n, L)` full-length sequences -> `(n, n_pos)` restricted to `batch.comparison_positions`
  (designable positions, or tie-group members for P09-s)."""
  if batch.comparison_positions.size == 0:
    return seqs
  return seqs[:, batch.comparison_positions]


# --------------------------------------------------------------------------------------
# aminx side: single sample draw, conditional (teacher-forced) score, batched draws
# --------------------------------------------------------------------------------------


def aminx_sample_one(
  jax_model: Any, batch: LaneBatch, order_i: np.ndarray, prng_key: Any, *, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
  """One `inference.sample_autoregressive.kernel` draw. Returns `(sequence, logits)`.

  `WaveScheduleBundle.from_tie_groups(tie_group_map, order_i)` groups tied positions at
  their first occurrence in `order_i` internally -- `order_i` need not be pre-flattened
  (verified against the class's own source, `types/bundles.py:212-277`).
  """
  import jax

  from aminx.inference import sample_autoregressive
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.types.bundles import WaveScheduleBundle

  wave = WaveScheduleBundle.from_tie_groups(
    jax.numpy.asarray(batch.tie_group_map), jax.numpy.asarray(order_i)
  )
  kw: dict[str, Any] = {
    "coords": jax.numpy.asarray(batch.x4),
    "mask": jax.numpy.asarray(batch.mask),
    "residue_index": jax.numpy.asarray(batch.residue_index, dtype=jax.numpy.int32),
    "chain_index": jax.numpy.asarray(batch.chain_index, dtype=jax.numpy.int32),
    "chain_mask": jax.numpy.asarray(batch.chain_mask),
    "bias": jax.numpy.asarray(batch.bias),
    "fixed_mask": jax.numpy.asarray(batch.fixed_mask),
    "fixed_tokens": jax.numpy.asarray(batch.seq_ref, dtype=jax.numpy.int32),
    "tie_group_map": jax.numpy.asarray(batch.tie_group_map),
    "wave": wave,
    "temperature": float(temperature),
    "mode": "sample",
  }
  if batch.use_side_chain_context:
    kw["atom_37"] = jax.numpy.asarray(batch.atom37)
    kw["atom_37_mask"] = jax.numpy.asarray(batch.atom37_mask)
    actx = 16
    kw["ligand_coords"] = jax.numpy.zeros((batch.length, actx, 3))
    kw["ligand_atom_types"] = jax.numpy.zeros((batch.length, actx), jax.numpy.int32)
    kw["ligand_mask"] = jax.numpy.zeros((batch.length, actx))
  bundle, config = build_inference_bundle(**kw)
  result = sample_autoregressive.kernel(jax_model, prng_key, bundle, config, make_stage_set())
  return np.asarray(result.sequence), np.asarray(result.logits)


def aminx_conditional_logits(
  jax_model: Any, batch: LaneBatch, seq_tokens: np.ndarray, order_i: np.ndarray
) -> np.ndarray:
  """RAW (pre-`log_softmax`) `score_conditional.kernel` logits for `seq_tokens` under the AR
  mask derived from `order_i`, at `batch.tie_group_map` (fused if P09-s)."""
  import jax

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  ar_mask = lac.ar_mask_from_order(order_i)
  kw: dict[str, Any] = {
    "coords": jax.numpy.asarray(batch.x4),
    "mask": jax.numpy.asarray(batch.mask),
    "residue_index": jax.numpy.asarray(batch.residue_index, dtype=jax.numpy.int32),
    "chain_index": jax.numpy.asarray(batch.chain_index, dtype=jax.numpy.int32),
    "chain_mask": jax.numpy.asarray(batch.chain_mask),
    "sequence": jax.nn.one_hot(jax.numpy.asarray(seq_tokens), 21),
    "ar_mask": jax.numpy.asarray(ar_mask),
    "tie_group_map": jax.numpy.asarray(batch.tie_group_map),
    "mode": "score_conditional",
  }
  if batch.use_side_chain_context:
    kw["atom_37"] = jax.numpy.asarray(batch.atom37)
    kw["atom_37_mask"] = jax.numpy.asarray(batch.atom37_mask)
    actx = 16
    kw["ligand_coords"] = jax.numpy.zeros((batch.length, actx, 3))
    kw["ligand_atom_types"] = jax.numpy.zeros((batch.length, actx), jax.numpy.int32)
    kw["ligand_mask"] = jax.numpy.zeros((batch.length, actx))
  bundle, config = build_inference_bundle(**kw)
  logits = score_conditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  return np.asarray(logits)


def wave_from_tie_groups_np(tie_group_map: np.ndarray, order_i: np.ndarray) -> Any:
  """`WaveScheduleBundle.from_tie_groups`, built in NumPy (D10).

  Same layout as the class method (one wave per tie group, groups in order of first
  appearance in `order_i`, each group's positions ascending and padded with -1 then
  zeroed under `position_valid`), without its per-position JAX ops -- on a GPU backend
  those dispatch one tiny device op each, measured at 0.5-3.3 s per wave on titanix.
  """
  import jax.numpy as jnp

  from aminx.types.bundles import WaveScheduleBundle

  tgm = np.asarray(tie_group_map)
  order = np.asarray(order_i)
  groups_in_order = tgm[order]
  _uniq, first = np.unique(groups_in_order, return_index=True)
  present = groups_in_order[np.sort(first)]
  max_positions = int(np.bincount(tgm).max())
  positions = np.full((present.size, max_positions), -1, dtype=np.int32)
  for w, g in enumerate(present):
    idx = np.flatnonzero(tgm == g)
    positions[w, : idx.size] = idx
  valid = positions != -1
  return WaveScheduleBundle(
    group_ids=jnp.asarray(present.astype(np.int32))[:, None],
    group_positions=jnp.asarray(np.where(valid, positions, 0))[:, None, :],
    group_valid=jnp.ones((present.size, 1), dtype=jnp.bool_),
    position_valid=jnp.asarray(valid)[:, None, :],
  )


_STAGE_SET: Any = None
_VMAPPED_SAMPLE: Any = None


def _vmapped_sample() -> Any:
  """One module-level jitted, vmapped kernel call (compiled once per fixture/lane shape).

  `eqx.filter_jit` treats `config` and the stage set as static; both are built once per
  call site and the stage set is a module singleton, so repeated chunks reuse the compile.
  """
  global _STAGE_SET, _VMAPPED_SAMPLE  # noqa: PLW0603
  if _VMAPPED_SAMPLE is None:
    import equinox as eqx
    import jax

    from aminx.inference import sample_autoregressive
    from aminx.inference.logits import make_stage_set

    _STAGE_SET = make_stage_set()

    @eqx.filter_jit
    def run(model: Any, keys: Any, waves: Any, bundle: Any, config: Any) -> Any:
      def one(key: Any, wave: Any) -> Any:
        b = eqx.tree_at(lambda x: x.wave, bundle, wave)
        return sample_autoregressive.kernel(
          model, key, b, config, _STAGE_SET, inference_only=True
        ).sequence

      return jax.vmap(one)(keys, waves)

    _VMAPPED_SAMPLE = run
  return _VMAPPED_SAMPLE


def sample_chunk_size(length: int) -> int:
  """Draws per vmapped call: bounded by device memory, which grows ~L^2 (measured on a
  24 GiB TITAN RTX: L=693 fits 16 draws but not 64). Deterministic in `length`."""
  return int(min(32, max(1, 1_500_000 // max(1, length * length))))


def aminx_sample_batch(
  jax_model: Any,
  batch: LaneBatch,
  n: int,
  seed_base: int,
  *,
  temperature: float,
  beta_alanine: float = 0.0,
) -> np.ndarray:
  """`n` aminx draws, FULL length, shape `(n, L)` -- NOT pre-restricted (see
  `reference_sample_batch`'s docstring for why; use `restrict_to_comparison` at the call site).

  `beta_alanine` adds an additive bias to the alanine column (positive-control sizing,
  "aminx arm with +beta on alanine") -- never applied to the reference arm.

  D10: draws run as one jitted `vmap` over `(key, wave)` per chunk of
  `sample_chunk_size(L)` rather than one un-jitted kernel call each. Draw `i` still uses
  order `_draw_order_for(batch, seed_base + i)` and key `PRNGKey(seed_base + i)`; the
  fused XLA program can round differently from the eager one, so a draw may differ from
  the per-call path at a few positions (measured 0/85, 15/85, 10/85 on 5L33) -- the two are
  the same sampler, not bit-identical.
  """
  import jax
  import jax.numpy as jnp

  from aminx.inference.bundle_builder import build_inference_bundle

  if n <= 0:
    return np.zeros((0, batch.length), dtype=np.int32)
  eff_batch = batch
  if beta_alanine:
    perturbed_bias = batch.bias.copy()
    perturbed_bias[:, ALANINE_INDEX] += beta_alanine
    eff_batch = dataclasses.replace(batch, bias=perturbed_bias)

  waves = [
    wave_from_tie_groups_np(eff_batch.tie_group_map, _draw_order_for(eff_batch, seed_base + i)[1])
    for i in range(n)
  ]
  kw: dict[str, Any] = {
    "coords": jnp.asarray(eff_batch.x4),
    "mask": jnp.asarray(eff_batch.mask),
    "residue_index": jnp.asarray(eff_batch.residue_index, dtype=jnp.int32),
    "chain_index": jnp.asarray(eff_batch.chain_index, dtype=jnp.int32),
    "chain_mask": jnp.asarray(eff_batch.chain_mask),
    "bias": jnp.asarray(eff_batch.bias),
    "fixed_mask": jnp.asarray(eff_batch.fixed_mask),
    "fixed_tokens": jnp.asarray(eff_batch.seq_ref, dtype=jnp.int32),
    "tie_group_map": jnp.asarray(eff_batch.tie_group_map),
    "wave": waves[0],
    "temperature": float(temperature),
    "mode": "sample",
  }
  if eff_batch.use_side_chain_context:
    kw["atom_37"] = jnp.asarray(eff_batch.atom37)
    kw["atom_37_mask"] = jnp.asarray(eff_batch.atom37_mask)
    actx = 16
    kw["ligand_coords"] = jnp.zeros((eff_batch.length, actx, 3))
    kw["ligand_atom_types"] = jnp.zeros((eff_batch.length, actx), jnp.int32)
    kw["ligand_mask"] = jnp.zeros((eff_batch.length, actx))
  bundle, config = build_inference_bundle(**kw)
  run = _vmapped_sample()
  chunk = min(sample_chunk_size(eff_batch.length), n)
  rows: list[np.ndarray] = []
  for lo in range(0, n, chunk):
    idx = list(range(lo, min(n, lo + chunk)))
    # Pad a short final chunk to the full chunk width so every call reuses one compile.
    padded = idx + [idx[-1]] * (chunk - len(idx))
    keys = jnp.stack([jax.random.PRNGKey(seed_base + i) for i in padded])
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[waves[i] for i in padded])
    out = np.asarray(run(jax_model, keys, stacked, bundle, config))
    rows.append(out[: len(idx)])
  return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------------------
# Teacher-forced comparison (step a) + sized fusion control (P09-s, R3-C6(3))
# --------------------------------------------------------------------------------------


def teacher_forced_lane(
  fixture: dict[str, Any],
  lane: str,
  weight_source: str,
  full_model_bundle: tuple[Any, Any, Any, Any],
  data_utils_module: Any,
  *,
  temperature: float,
  fusion_eps: float = DEFAULT_FUSION_CTRL_EPS,
) -> dict[str, Any]:
  """(a) Teacher-forced: feed the reference's own sampled sequence + order into aminx
  conditional decoding, compare per-step log-probs. Also runs the sized fusion control
  for P09-s (R3-C6(3))."""
  jax_model, pt_model, torch, _model_utils = full_model_bundle
  batch = build_lane_batch(fixture, lane, data_utils_module)
  seed_i = _seed_for(fixture["name"] + lane)

  if batch.groups is None and _LANE_BASE[lane] == "P09-s":
    # P09-s but no qualifying groups on this fixture: nothing to teacher-force.
    return {
      "lane": lane,
      "fixture": fixture["name"],
      "tf_max_abs": None,
      "status_note": "no k-NN-disjoint multi-member tie group on this fixture",
    }

  seq, ref_log_probs, decoding_order, _randn = reference_sample_one(
    pt_model, torch, batch, seed_i, temperature=temperature
  )
  aminx_raw_logits = aminx_conditional_logits(jax_model, batch, seq, decoding_order)
  aminx_log_probs = np.asarray(_log_softmax(aminx_raw_logits))

  omitted, x_aminx, x_ref = _count_omitted_and_x(seq, lane, aminx_seq=None)

  result: dict[str, Any] = {
    "lane": lane,
    "fixture": fixture["name"],
    "reference_sequence": seq,
    "reference_log_probs": ref_log_probs,
    "aminx_log_probs": aminx_log_probs,
    "decoding_order": decoding_order,
    "omitted_aa_count": omitted,
    "x_token_count_reference": x_ref,
    "x_token_count_aminx_sentinel": x_aminx,  # from this SAME reference-fed sequence, sentinel only
  }

  if batch.groups:
    from tests.parity.test_full_model_parity import _combine_reference_tied_log_probs

    ref_fused = _combine_reference_tied_log_probs(
      ref_log_probs, tie_groups=batch.groups, tie_weights=[[1.0] * len(g) for g in batch.groups]
    )
    member_idx = batch.comparison_positions
    tf_max_abs = lae._max_abs(ref_fused[member_idx], aminx_log_probs[member_idx])  # noqa: SLF001
    result["tf_max_abs"] = tf_max_abs
    result["p09_fused_tf_max_abs"] = tf_max_abs
    result["p09_tied_positions"] = int(member_idx.size)
    result["fusion_control"] = _fusion_sized_control(
      jax_model, batch, decoding_order, eps=fusion_eps
    )
  else:
    pos = batch.comparison_positions if batch.comparison_positions.size else np.arange(batch.length)
    result["tf_max_abs"] = lae._max_abs(ref_log_probs[pos], aminx_log_probs[pos])  # noqa: SLF001

  return result


def _log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _fusion_sized_control(
  jax_model: Any, batch: LaneBatch, decoding_order: np.ndarray, *, eps: float
) -> dict[str, Any]:
  """R3-C6(3): the aminx fusion with the LAST group member's logits scaled by `(1+eps)` must
  move the fused max-abs into `SIZING_RATIO_RANGE` of `TF_BAR`. Reuses
  `TieGroupProductOfExperts` (`aminx.inference.logits`) directly, never re-derived."""
  import jax

  from aminx.inference.logits import TieGroupProductOfExperts

  if not batch.groups:
    return {"eps": eps, "effect": 0.0, "ratio_to_bar": 0.0, "detected": False}
  group = next((g for g in batch.groups if len(g) >= 2), None)
  if group is None:
    return {"eps": eps, "effect": 0.0, "ratio_to_bar": 0.0, "detected": False}

  ungated_batch = dataclasses.replace(batch, tie_group_map=np.arange(batch.length, dtype=np.int64))
  raw_logits = aminx_conditional_logits(jax_model, ungated_batch, batch.seq_ref, decoding_order)

  fuse = TieGroupProductOfExperts()
  mask = np.zeros(batch.length, dtype=bool)
  mask[group] = True
  normal_fused = np.asarray(fuse(jax.numpy.asarray(raw_logits), jax.numpy.asarray(mask)))

  perturbed_logits = raw_logits.copy()
  perturbed_logits[group[-1]] = perturbed_logits[group[-1]] * (1.0 + eps)
  perturbed_fused = np.asarray(fuse(jax.numpy.asarray(perturbed_logits), jax.numpy.asarray(mask)))

  effect = float(np.max(np.abs(normal_fused - perturbed_fused)))
  ratio = effect / TF_BAR
  return {
    "eps": eps,
    "effect": effect,
    "ratio_to_bar": ratio,
    "detected": bool(effect > TF_BAR),
  }


def _count_omitted_and_x(
  tokens: np.ndarray, lane: str, *, aminx_seq: np.ndarray | None
) -> tuple[int, int, int]:
  """`(omitted_aa_count, x_token_count_aminx, x_token_count_reference)` on a full-length
  (unrestricted) token array. Only P08 has a nonzero omit vocabulary; every lane's X-omit
  bias should make `x_token_count_*` 0 by construction (a nonzero count is a real finding,
  not expected noise)."""
  x_reference = int(np.sum(tokens == X_INDEX))
  x_aminx = int(np.sum(aminx_seq == X_INDEX)) if aminx_seq is not None else 0
  omitted = count_omitted(tokens, lane)
  return omitted, x_aminx, x_reference


def count_omitted(tokens: np.ndarray, lane: str) -> int:
  """Count P08's omitted-AA vocabulary (`OMIT_AA_CHARS` everywhere, PLUS the per-residue
  omit char at `P08_PER_RESIDUE_OMIT_POSITION`) in `tokens` -- 1-D `(n_pos,)` or 2-D
  `(n_draws, n_pos)`, ASSUMED to be indexed identically to the FULL sequence (true for
  P08, whose `comparison_positions` covers every position since its `chain_mask` is
  all-designable -- see `build_lane_batch`). Every other lane has an empty omit
  vocabulary and always returns 0."""
  if _LANE_BASE[lane] != "P08":
    return 0
  arr = np.atleast_2d(tokens)
  omitted = int(np.isin(arr, _omit_indices(OMIT_AA_CHARS)).sum())
  per_residue_idx = _mpnn_index(P08_PER_RESIDUE_OMIT_CHAR)
  if P08_PER_RESIDUE_OMIT_POSITION < arr.shape[1]:
    omitted += int(np.sum(arr[:, P08_PER_RESIDUE_OMIT_POSITION] == per_residue_idx))
  return omitted


def count_x(tokens: np.ndarray) -> int:
  """Count `X_INDEX` occurrences in `tokens` (1-D or 2-D) -- should be 0 by construction
  (the X-omit bias column) on every lane, both arms."""
  return int(np.sum(np.asarray(tokens) == X_INDEX))


# --------------------------------------------------------------------------------------
# Statistical tier (step b): margin/excess-JS/TOST, reusing aminx.parity.compare
# --------------------------------------------------------------------------------------


def pooled_excess_js(
  a1: np.ndarray | list[np.ndarray],
  a2: np.ndarray | list[np.ndarray],
  r1: np.ndarray | list[np.ndarray],
  r2: np.ndarray | list[np.ndarray],
  *,
  k: int = 21,
) -> float:
  """Point-estimate `E = D(a1,r1) - 1/2*[D(a1,a2)+D(r1,r2)]` (`aminx.parity.compare.excess_js`,
  reused). `excess_js`/`mean_positional_js` operate on per-position token COUNT tables, not raw
  sampled-sequence arrays -- this pools each arm's (possibly per-fixture-stratified) sequences
  into counts via `aminx.parity.compare.token_counts` first (the same pooling
  `excess_js_upper`'s bootstrap does internally per-resample), so the point estimate and the
  bootstrap upper bound are computed from the SAME representation."""
  from aminx.parity.compare import excess_js, token_counts

  a1_counts = token_counts(a1, k=k)
  a2_counts = token_counts(a2, k=k)
  r1_counts = token_counts(r1, k=k)
  r2_counts = token_counts(r2, k=k)
  return float(excess_js(a1_counts, a2_counts, r1_counts, r2_counts))


def pooled_js(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray], *, k: int = 21
) -> float:
  """Pooled-composition JS divergence between two arms (`main_js_vs_ref`'s own metric: no
  excess-JS bias correction, just `mean_positional_js` on pooled token counts)."""
  from aminx.parity.compare import mean_positional_js, token_counts

  return float(mean_positional_js(token_counts(a, k=k), token_counts(b, k=k)))


def lane_equivalence(
  a1: np.ndarray | list[np.ndarray],
  a2: np.ndarray | list[np.ndarray],
  r1: np.ndarray | list[np.ndarray],
  r2: np.ndarray | list[np.ndarray],
  *,
  margin: float,
  n_boot: int,
  rng: np.random.Generator,
  k: int = 21,
) -> dict[str, Any]:
  """One lane's statistical-tier verdict: recovery TOST AND excess-JS upper bound < margin
  (IUT). `k=21` (the full MPNN alphabet incl. X) since sampling lanes may, in principle,
  produce X at a nonzero rate on a mis-wired run -- the omitted-AA/X EXACT row is the
  dedicated check for that, but `excess_js_upper`'s token-count math must not silently
  discard a real divergence by assuming a 20-letter alphabet.

  Each of `a1, a2, r1, r2` is either one fixture's `(n_seq, L)` token array or a list of such
  arrays (one per fixture, fixture-stratified -- `excess_js_upper`/`token_counts` both accept
  this natively).

  Returns `{tost_pass, excess_js, excess_js_ub, margin, equiv, equiv_js, n_boot}`. `tost_pass`
  is always `False` here (callers compute recovery TOST separately, over the FULL arms per
  V3, and merge it in) -- kept in the dict only for backward-compatible key presence.
  """
  from aminx.parity.compare import excess_js_upper

  excess_js_point = pooled_excess_js(a1, a2, r1, r2, k=k)
  excess_js_ub = excess_js_upper(a1, a2, r1, r2, n_boot=n_boot, rng=rng, k=k)
  equiv_js = bool(excess_js_ub < margin)
  return {
    "excess_js": excess_js_point,
    "excess_js_ub": excess_js_ub,
    "margin": margin,
    "equiv_js": equiv_js,
    "n_boot": n_boot,
  }


def recovery_tost(
  arm_recovery: np.ndarray, reference_recovery: np.ndarray, *, delta: float = RECOVERY_DELTA
) -> dict[str, Any]:
  """Recovery TOST between two per-sequence recovery-fraction arrays."""
  from aminx.parity.compare import tost_mean_diff

  passed, p_lower, p_upper = tost_mean_diff(arm_recovery, reference_recovery, delta)
  return {"tost_pass": bool(passed), "p_lower": p_lower, "p_upper": p_upper, "delta": delta}


def per_sequence_recovery(tokens: np.ndarray, seq_ref_restricted: np.ndarray) -> np.ndarray:
  """`(n,)` per-sequence recovery fraction against the restricted native sequence."""
  return np.mean(tokens == seq_ref_restricted[None, :], axis=-1)


# --------------------------------------------------------------------------------------
# Shared IUT-arm draw: main lane measurement, margin computation, null-replicate check,
# and the positive/negative controls all funnel through this ONE function so the draw
# semantics (allocation, seeding, restriction) cannot drift between calibrate and validate.
# --------------------------------------------------------------------------------------


def draw_iut_arms(
  jax_model: Any,
  pt_model: Any,
  torch: Any,
  fixture_batches: list[tuple[dict[str, Any], LaneBatch]],
  allocation: dict[str, int],
  *,
  temperature: float,
  temperature_r: float | None = None,
  beta_a: float = 0.0,
  reference_is_aminx: bool = False,
  seed_tag: str,
) -> tuple[
  list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
  """Draw `(a1_list, a2_list, r1_list, r2_list, seq_ref_list)`, each a list of one
  `(k, n_pos)` array per fixture (fixture-stratified, restricted to `comparison_positions`),
  `k = allocation[fixture_name]`.

  - `a1`/`a2`: two INDEPENDENTLY-seeded aminx draws (the aminx arm; `+beta_a` on alanine
    when nonzero -- positive-control sizing).
  - `r1`/`r2`: the reference arm (`reference_is_aminx=False`, the default -- REAL LigandMPNN
    draws) OR two more independently-seeded, UNPERTURBED aminx draws
    (`reference_is_aminx=True` -- used for the margin rule's "aminx at T vs aminx at
    1.05*T" (`temperature_r` overrides `temperature` for r1/r2), the null-replicate
    criterion, and the negative control -- all three are "aminx vs aminx" by spec).

  Fixtures with a zero/missing allocation entry, or whose `comparison_positions` is empty
  (e.g. P09-s on a fixture with no qualifying tie groups), are skipped.
  """
  temp_r = temperature if temperature_r is None else temperature_r
  a1_list: list[np.ndarray] = []
  a2_list: list[np.ndarray] = []
  r1_list: list[np.ndarray] = []
  r2_list: list[np.ndarray] = []
  seq_ref_list: list[np.ndarray] = []
  for fixture, batch in fixture_batches:
    k = allocation.get(fixture["name"], 0)
    if k <= 0 or batch.comparison_positions.size == 0:
      continue
    tag = f"{fixture['name']}{batch.lane}{seed_tag}"
    a1 = restrict_to_comparison(
      aminx_sample_batch(
        jax_model, batch, k, _seed_for(tag + "A1"), temperature=temperature, beta_alanine=beta_a
      ),
      batch,
    )
    a2 = restrict_to_comparison(
      aminx_sample_batch(jax_model, batch, k, _seed_for(tag + "A2"), temperature=temperature),
      batch,
    )
    if reference_is_aminx:
      r1 = restrict_to_comparison(
        aminx_sample_batch(jax_model, batch, k, _seed_for(tag + "R1"), temperature=temp_r), batch
      )
      r2 = restrict_to_comparison(
        aminx_sample_batch(jax_model, batch, k, _seed_for(tag + "R2"), temperature=temp_r), batch
      )
    else:
      r1 = restrict_to_comparison(
        reference_sample_batch(
          pt_model, torch, batch, k, _seed_for(tag + "R1"), temperature=temp_r
        ),
        batch,
      )
      r2 = restrict_to_comparison(
        reference_sample_batch(
          pt_model, torch, batch, k, _seed_for(tag + "R2"), temperature=temp_r
        ),
        batch,
      )
    a1_list.append(a1)
    a2_list.append(a2)
    r1_list.append(r1)
    r2_list.append(r2)
    seq_ref_list.append(batch.seq_ref[batch.comparison_positions])
  return a1_list, a2_list, r1_list, r2_list, seq_ref_list


def full_arm_recovery(
  arm1_list: list[np.ndarray], arm2_list: list[np.ndarray], seq_ref_list: list[np.ndarray]
) -> np.ndarray:
  """Per-sequence recovery over the FULL arm (`arm1 UNION arm2`, spec V3: "Recovery TOST
  over the full arms"), pooled across fixtures."""
  parts = []
  for a1, a2, ref in zip(arm1_list, arm2_list, seq_ref_list, strict=True):
    parts.append(per_sequence_recovery(a1, ref))
    parts.append(per_sequence_recovery(a2, ref))
  return np.concatenate(parts) if parts else np.asarray([])


# --------------------------------------------------------------------------------------
# Per-draw wall-clock cost measurement (C4's budget formula inputs)
# --------------------------------------------------------------------------------------


def measure_aminx_draw_cost_s(
  jax_model: Any, batch: LaneBatch, temperature: float, *, seed_base: int = 999_000
) -> float:
  """Seconds per aminx draw on the batched path, STEADY-STATE: one untimed chunk first
  (compile), then one timed chunk of `sample_chunk_size(L)` draws, host-side wave
  construction included (it is part of every real draw's cost)."""
  import time

  chunk = sample_chunk_size(batch.length)
  aminx_sample_batch(jax_model, batch, chunk, seed_base, temperature=temperature)
  start = time.monotonic()
  aminx_sample_batch(jax_model, batch, chunk, seed_base + chunk, temperature=temperature)
  return (time.monotonic() - start) / chunk


def measure_native_x_frequency(
  jax_model: Any, batch: LaneBatch, temperature: float, n: int = 20, seed_base: int = 777_000
) -> float:
  """AC-11: a small, SEPARATE, non-gating diagnostic -- aminx's own X-sampling frequency
  WITHOUT the X-omit bias column (i.e. what the model would do if nothing hard-omitted X),
  reported alongside `x_token_count_*` for context, never used in any pass/fail decision."""
  no_omit_bias = batch.bias.copy()
  no_omit_bias[:, X_INDEX] = 0.0
  eff_batch = dataclasses.replace(batch, bias=no_omit_bias)
  seqs = aminx_sample_batch(jax_model, eff_batch, n, seed_base, temperature=temperature)
  restricted = restrict_to_comparison(seqs, batch)
  return float(np.mean(restricted == X_INDEX)) if restricted.size else 0.0


def measure_reference_draw_cost_s(
  pt_model: Any, torch: Any, batch: LaneBatch, temperature: float, n_sample: int
) -> float:
  """Seconds per reference draw, amortized over one batched (or batch-1-looped, for P09-s)
  `.sample()` call of `n_sample` draws."""
  import time

  start = time.monotonic()
  reference_sample_batch(pt_model, torch, batch, n_sample, 999_200, temperature=temperature)
  return (time.monotonic() - start) / n_sample


# --------------------------------------------------------------------------------------
# Fixture-proportional draw allocation (R2-C11: "allocated across fixtures proportionally
# to their designable-position counts")
# --------------------------------------------------------------------------------------


def allocate_draws(
  fixtures_for_lane: list[tuple[dict[str, Any], LaneBatch]], n_total: int
) -> dict[str, int]:
  """Allocate `n_total` per-arm draws across fixtures proportionally to
  `len(comparison_positions)`, largest remainder method (deterministic, sums to `n_total`)."""
  weights = [max(1, len(batch.comparison_positions)) for _fixture, batch in fixtures_for_lane]
  total_weight = sum(weights)
  raw = [n_total * w / total_weight for w in weights]
  floors = [int(x) for x in raw]
  remainder = n_total - sum(floors)
  order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
  for i in order[:remainder]:
    floors[i] += 1
  return {
    fixture["name"]: count
    for (fixture, _batch), count in zip(fixtures_for_lane, floors, strict=True)
  }


# --------------------------------------------------------------------------------------
# P14 sampling lane (NOT IMPLEMENTED -- see module docstring)
# --------------------------------------------------------------------------------------

P14_SAMPLING_NOT_IMPLEMENTED_REASON = (
  "P14 sampling lane (10,000 chi draws from fixed real mixture parameters vs the analytic "
  "von Mises mixture CDF, KS test, x1.2 concentration control) needs a chi-angle SAMPLING "
  "routine for the packer. Searched (per the fixer brief's own directive) for 'packer' and "
  "'von_mises'/'vonmises' across the whole repository: `grep -rln \"vonmises\\|von_mises\" "
  "src/ tests/` returns ZERO matches anywhere in aminx -- src/aminx/model/packer.py "
  "(PackerProteinFeatures, Packer) defines only the mixture PARAMETER heads (mixture_logits, "
  "mu/loc, kappa/concentration -- see types/bundles.py:501-518's PackerResult fields), with "
  "no `jax.random.vonmises` call site, no CDF, and no other sampling routine of any kind "
  "for chi angles anywhere in src/ or tests/. This is a STRICTLY STRONGER gap than T7a's "
  "finding (which was only about a real-structure -> PackerBundle adapter): even with FIXED, "
  "already-real mixture parameters handed to it directly (no real-structure adapter needed "
  "at all), there is nothing in this codebase to draw a chi sample from, or an analytic CDF "
  "to compare against. Recorded as not_implemented rather than fabricated; the "
  "weight-perturbation concentration x1.2 control is correspondingly not implemented "
  "(nothing to perturb without a working sampler)."
)


def packer_lane() -> dict[str, Any]:
  """P14 sampling lane: NOT IMPLEMENTED. See `P14_SAMPLING_NOT_IMPLEMENTED_REASON`."""
  return {
    "path": "P14.packer_mixture_sampling",
    "metric": "not_implemented",
    "status": "not_implemented",
    "rejection_rate": None,
    "reason": P14_SAMPLING_NOT_IMPLEMENTED_REASON,
  }
