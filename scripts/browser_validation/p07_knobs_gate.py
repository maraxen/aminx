"""Sampling-knobs gate for the P07 browser ProteinMPNN sampler.

Check A feeds RunSpec tensors from ``browser/layer_c/runspec.mjs`` to JAX
``make_p07_sample`` and to the jax2onnx ORT-CPU graph (and, with ``--browser``,
to onnxruntime-web through ``run_p07.mjs``). Liveness requires each non-default
knob to change tokens on at least one seed; a bias-frozen wrapper must stay
not-live on bias+omit. B1 is the teacher-forced log-prob comparison against
reference ProteinMPNN. B2 is synthetic Gumbel-max, Fisher-Yates uniformity,
and fixed-first decoding order, including the ``-log(u)``, biased-shuffle,
and uniform-shuffle controls.

Graded failures exit 0. Exit 3 is reserved for an integrity refusal.
``temp0.1`` repeats the default temperature, so it is an alias of baseline and
is not part of the liveness count.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

RUNSPEC_MJS = _ROOT / "browser" / "layer_c" / "runspec.mjs"
RUN_P07_MJS = _ROOT / "browser" / "layer_c" / "run_p07.mjs"
LAYER_C_DIR = _ROOT / "browser" / "layer_c"
MANIFEST_PATH = _ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
OMIT_BIAS = np.float32(-1e8)
LOGPROB_BAR = 2e-4  # a_max_logprob_abs_diff bound; amended 1e-4 -> 2e-4 (run e15b06ec, 260929)
TV_BAR = 0.03
P_BAR = 1e-3
N_AA = 21
FULL_BUCKETS = (128, 256)
FULL_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
SMOKE_SEEDS = (11, 22)
FULL_KNOBS = (
  "baseline",
  "temp0.1",
  "temp1.0",
  "bias+omit",
  "fixed_positions",
  "tied_pairs",
  "chains_to_design",
  "explicit_order",
  "all_combined",
)
SMOKE_KNOBS = ("baseline", "temp1.0", "bias+omit")
LIVE_SKIP = frozenset({"baseline", "temp0.1"})
B2_N_FULL = 200_000
ORDER_N_FULL = 240_000
B2_N_SMOKE = 20_000
ORDER_LENGTH = 4
OMIT_CLASS = 3

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_1 = 0xBF58476D1CE4E5B9
_MIX_2 = 0x94D049BB133111EB
_U32_SPAN = 0x100000000
_MANTISSA_DENOM = 16777217.0

OUTCOMES: tuple[tuple[str, str], ...] = (
  (
    "incomplete",
    "NOT checks_complete OR budget_exceeded OR smoke OR NOT browser_enabled "
    "OR knobs_total < 7 OR a_cases_total < 144 OR b2_n < 200000 OR order_n < 240000",
  ),
  (
    "ctrl_blind",
    "NOT ctrl_frozen_detected OR NOT ctrl_bad_gumbel_detected OR NOT ctrl_bad_shuffle_detected "
    "OR NOT ctrl_uniform_order_detected",
  ),
  (
    "a_mismatch",
    "a_cases_bitwise_cpu < a_cases_total OR a_cases_bitwise_web < a_cases_total "
    "OR a_max_logprob_abs_diff > 0.0002 OR NOT a_web_ok OR NOT alias_ok",
  ),
  ("not_live", "knobs_live < knobs_total"),
  ("b1_mismatch", "b1_max_abs_nats > 0.0001"),
  (
    "b2_fail",
    "b2_gumbel_tv_max > 0.03 OR b2_gumbel_p_min < 0.001 OR b2_order_p < 0.001 "
    "OR b2_omitted_draws > 0 OR b2_order_fixed_first_violations > 0 OR b2_order_within_p < 0.001",
  ),
  (
    "pass",
    "checks_complete AND NOT budget_exceeded AND NOT smoke AND browser_enabled "
    "AND knobs_total >= 7 AND a_cases_total >= 144 AND b2_n >= 200000 AND order_n >= 240000 "
    "AND a_cases_bitwise_web = a_cases_total "
    "AND ctrl_frozen_detected AND ctrl_bad_gumbel_detected "
    "AND ctrl_bad_shuffle_detected AND a_web_ok AND alias_ok AND knobs_live = knobs_total "
    "AND a_cases_bitwise_cpu = a_cases_total AND a_max_logprob_abs_diff <= 0.0002 "
    "AND b1_max_abs_nats <= 0.0001 AND b2_gumbel_tv_max <= 0.03 AND b2_gumbel_p_min >= 0.001 "
    "AND b2_order_p >= 0.001 AND b2_omitted_draws = 0 "
    "AND b2_order_fixed_first_violations = 0 AND b2_order_within_p >= 0.001 "
    "AND ctrl_uniform_order_detected",
  ),
)

_CMP = re.compile(r"^(\w+)\s*(>=|<=|>|<|=)\s*(\S+)$")

P07_KEYS = (
  "coords",
  "mask",
  "residue_index",
  "chain_index",
  "gumbel_noise",
  "decoding_order",
  "bias",
  "fixed_mask",
  "fixed_tokens",
  "temperature",
  "tie_group_map",
)


class IntegrityRefusalError(RuntimeError):
  """Raised when the run cannot be graded because an input pin or file is missing."""


class SplitMix64:
  """Counter-based SplitMix64. ``seed`` is a uint32, zero-extended."""

  def __init__(self, seed: int) -> None:
    self.state = int(seed) & 0xFFFFFFFF

  def next_u64(self) -> int:
    self.state = (self.state + _GOLDEN) & _MASK64
    mixed = self.state
    mixed = ((mixed ^ (mixed >> 30)) * _MIX_1) & _MASK64
    mixed = ((mixed ^ (mixed >> 27)) * _MIX_2) & _MASK64
    return (mixed ^ (mixed >> 31)) & _MASK64

  def next_u32(self) -> int:
    return self.next_u64() >> 32


def to_float32(value: float) -> np.float32:
  return np.float32(value)


def uniform_from_u32(u32: int) -> np.float32:
  mant = (int(u32) >> 8) & 0xFFFFFF
  return np.float32((mant + 1) / _MANTISSA_DENOM)


def gumbel_from_uniform(u: float) -> np.float32:
  """``-log(-log(u))`` with the same float32 stores as ``runspec.mjs``."""
  uu = np.float32(u)
  inner = np.float32(-math.log(float(uu)))
  return np.float32(-math.log(float(inner)))


def bad_gumbel_from_uniform(u: float) -> np.float32:
  """Negative control: ``-log(u)`` instead of a Gumbel variate."""
  uu = np.float32(u)
  return np.float32(-math.log(float(uu)))


def next_below(prng: SplitMix64, bound: int) -> int:
  limit = _U32_SPAN - (_U32_SPAN % bound)
  draw = prng.next_u32()
  while draw >= limit:
    draw = prng.next_u32()
  return draw % bound


def shuffle(length: int, prng: SplitMix64) -> np.ndarray:
  order = np.arange(length, dtype=np.int32)
  for i in range(length - 1, 0, -1):
    j = next_below(prng, i + 1)
    order[i], order[j] = order[j], order[i]
  return order


def designed_flags(mask: np.ndarray, fixed_mask: np.ndarray, ties: np.ndarray) -> np.ndarray:
  """Positions decoded with the designed group (1), after tie expansion.

  Unmasked and unfixed positions are designed. A tie group that contains any
  designed position is entirely designed.
  """
  base = (mask != 0) & (fixed_mask == 0)
  group_hit = np.zeros((mask.shape[0],), dtype=bool)
  group_hit[ties[base]] = True
  return group_hit[ties]


def fixed_first_shuffle(length: int, prng: SplitMix64, designed: np.ndarray) -> np.ndarray:
  """Fisher-Yates, then a stable partition (not-designed, then designed).

  Within-group uniformity follows because ``shuffle`` is uniform on S_L and the
  relative order of a subset of a uniform permutation is uniform on that subset.
  The partition draws no further random numbers.
  """
  order = shuffle(length, prng)
  flags = np.asarray(designed, dtype=bool)
  return np.concatenate([order[~flags[order]], order[flags[order]]]).astype(np.int32, copy=False)


def biased_shuffle(length: int, prng: SplitMix64) -> np.ndarray:
  """Negative control: swap ``i`` with a uniform index in ``0..L-1``."""
  order = np.arange(length, dtype=np.int32)
  for i in range(length):
    j = next_below(prng, length)
    order[i], order[j] = order[j], order[i]
  return order


def tie_group_map(length: int, groups: list[list[int]] | None) -> np.ndarray:
  """Map each tied member to the smallest index in its group. Untied positions keep their index."""
  ids = np.arange(length, dtype=np.int32)
  for group in groups or []:
    if not group:
      continue
    members = [int(member) for member in group]
    gid = min(members)  # canonical: smallest member, always inside 0..L-1
    for member in members:
      ids[member] = np.int32(gid)
  return ids


def _letter_index(letter: str) -> int:
  try:
    return MPNN_ALPHABET.index(letter)
  except ValueError as exc:
    msg = f"letter {letter!r} is not in MPNN_ALPHABET {MPNN_ALPHABET!r}"
    raise ValueError(msg) from exc


def _as_f32(values: object, shape: tuple[int, ...]) -> np.ndarray:
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  return flat.astype(np.float32).reshape(shape)


def _as_i32(values: object, shape: tuple[int, ...]) -> np.ndarray:
  return np.asarray(values, dtype=np.int32).reshape(shape)


def _position_items(mapping: object) -> list[tuple[int, object]]:
  if not mapping:
    return []
  if not isinstance(mapping, dict):
    msg = "per-residue knob must be a mapping"
    raise TypeError(msg)
  return [(int(key), value) for key, value in mapping.items()]


def _coords_shape(structure: dict[str, object]) -> tuple[int, int, int]:
  declared = structure.get("coords_shape")
  if isinstance(declared, list) and len(declared) == 3:
    return (int(declared[0]), int(declared[1]), int(declared[2]))
  coords = np.asarray(structure["coords"])
  if coords.ndim == 3:
    return (int(coords.shape[0]), int(coords.shape[1]), int(coords.shape[2]))
  msg = "structure.coords needs coords_shape or a rank-3 array"
  raise ValueError(msg)


def build_p07_inputs(  # noqa: PLR0915
  structure: dict[str, object],
  runspec: dict[str, object],
) -> dict[str, np.ndarray]:
  """Python twin of ``buildP07Inputs`` in ``runspec.mjs``."""
  length, _, _ = _coords_shape(structure)
  coords = _as_f32(structure["coords"], (length, 4, 3))
  mask = _as_f32(structure["mask"], (length,))
  residue_index = _as_i32(structure["residue_index"], (length,))
  chain_index = _as_i32(structure["chain_index"], (length,))
  chain_ids_obj = structure.get("chain_ids") or []
  chain_ids = [str(item) for item in chain_ids_obj] if isinstance(chain_ids_obj, list) else []
  if structure.get("native_tokens") is None:
    native = np.zeros((length,), dtype=np.int32)
  else:
    native = _as_i32(structure["native_tokens"], (length,))

  temperature = to_float32(
    0.1 if runspec.get("temperature") is None else float(runspec["temperature"]),
  )  # type: ignore[arg-type]
  bias = np.zeros((length, N_AA), dtype=np.float32)
  bias_aa = runspec.get("bias_AA") or {}
  if isinstance(bias_aa, dict):
    for letter, value in bias_aa.items():
      column = _letter_index(str(letter))
      delta = to_float32(float(value))
      bias[:, column] = bias[:, column] + delta
  for pos, letters in _position_items(runspec.get("bias_AA_per_residue")):
    if isinstance(letters, dict):
      for letter, value in letters.items():
        column = _letter_index(str(letter))
        bias[pos, column] = np.float32(bias[pos, column] + to_float32(float(value)))
  omit_aa = str(runspec.get("omit_AA") or "")
  for letter in omit_aa:
    bias[:, _letter_index(letter)] = OMIT_BIAS
  for pos, letters in _position_items(runspec.get("omit_AA_per_residue")):
    for letter in str(letters):
      bias[pos, _letter_index(letter)] = OMIT_BIAS

  fixed_mask = np.zeros((length,), dtype=np.float32)
  fixed_tokens = np.zeros((length,), dtype=np.int32)
  chains = runspec.get("chains_to_design")
  if isinstance(chains, list):
    design = {str(item) for item in chains}
    if len(chain_ids) != length:
      msg = "chains_to_design requires structure.chain_ids of length L"
      raise ValueError(msg)
    for pos, chain_id in enumerate(chain_ids):
      if chain_id not in design:
        fixed_mask[pos] = np.float32(1.0)
        fixed_tokens[pos] = native[pos]
  for pos, letter in _position_items(runspec.get("fixed_positions")):
    fixed_mask[pos] = np.float32(1.0)
    fixed_tokens[pos] = np.int32(_letter_index(str(letter)))

  groups_obj = runspec.get("tied_positions") or []
  groups = (
    [[int(member) for member in group] for group in groups_obj]
    if isinstance(groups_obj, list)
    else []
  )
  ties = tie_group_map(length, groups)

  seed = int(runspec.get("seed") or 0) & 0xFFFFFFFF
  prng = SplitMix64(seed)
  requested = runspec.get("decoding_order", "random")
  if requested == "random" or requested is None:
    decoding_order = fixed_first_shuffle(length, prng, designed_flags(mask, fixed_mask, ties))
  else:
    decoding_order = np.asarray(requested, dtype=np.int32)
    if decoding_order.shape != (length,) or len({int(v) for v in decoding_order}) != length:
      msg = "explicit decoding_order is not a permutation of 0..L-1"
      raise ValueError(msg)

  gumbel = np.empty((length, N_AA), dtype=np.float32)
  flat = gumbel.reshape(-1)
  for i in range(flat.shape[0]):
    flat[i] = gumbel_from_uniform(uniform_from_u32(prng.next_u32()))

  return {
    "coords": coords,
    "mask": mask,
    "residue_index": residue_index,
    "chain_index": chain_index,
    "gumbel_noise": gumbel,
    "decoding_order": decoding_order,
    "bias": bias,
    "fixed_mask": fixed_mask,
    "fixed_tokens": fixed_tokens,
    "temperature": np.asarray(temperature, dtype=np.float32),
    "tie_group_map": ties,
  }


def logit_bank() -> list[np.ndarray]:
  """Three fixed logit rows: a peak, a hard omit, and a spread."""
  peaked = np.zeros((N_AA,), dtype=np.float32)
  peaked[0] = np.float32(2.0)
  omitted = np.zeros((N_AA,), dtype=np.float32)
  omitted[OMIT_CLASS] = OMIT_BIAS
  spread = np.linspace(-2.0, 2.0, N_AA).astype(np.float32)
  return [peaked, omitted, spread]


def softmax_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
  scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
  scaled -= np.max(scaled)
  weights = np.exp(scaled)
  return weights / np.sum(weights)


def multinomial_p(counts: np.ndarray, probs: np.ndarray) -> float:
  """Chi-square p, pooling bins with expected count below 5. Degenerate cases return 1."""
  from scipy.stats import chisquare  # noqa: PLC0415

  expected = np.asarray(probs, dtype=np.float64) * float(np.sum(counts))
  keep = expected >= 1e-8
  observed = np.asarray(counts, dtype=np.float64)[keep]
  expected = expected[keep]
  small = expected < 5
  if np.any(small):
    if not np.any(~small):
      return 1.0
    observed = np.concatenate([observed[~small], [float(np.sum(observed[small]))]])
    expected = np.concatenate([expected[~small], [float(np.sum(expected[small]))]])
  if expected.size < 2 or float(np.min(expected)) <= 0:
    return 1.0
  return float(chisquare(observed, expected).pvalue)


def gumbel_max_tokens(logits: np.ndarray, temperature: float, noise: np.ndarray) -> np.ndarray:
  scaled = np.asarray(logits, dtype=np.float32) / np.float32(temperature)
  return np.argmax(noise + scaled[None, :], axis=1).astype(np.int32)


def summarize_gumbel(noise_of: object, n_draws: int, seed0: int) -> dict[str, float | int | bool]:
  """Total variation, chi-square p, and omitted-class draws over the logit bank."""
  tv_max = 0.0
  p_min = 1.0
  omitted = 0
  for index, logits in enumerate(logit_bank()):
    for temp_i, temperature in enumerate((0.1, 1.0)):
      noise = noise_of(seed0 + index * 10 + temp_i, n_draws, N_AA)  # type: ignore[operator]
      tokens = gumbel_max_tokens(logits, temperature, np.asarray(noise))
      probs = softmax_probs(logits, temperature)
      counts = np.bincount(tokens, minlength=N_AA)
      tv = 0.5 * float(np.sum(np.abs(counts / float(n_draws) - probs)))
      p_value = multinomial_p(counts, probs)
      tv_max = max(tv_max, tv)
      p_min = min(p_min, p_value)
      if index == 1:
        omitted += int(counts[OMIT_CLASS])
  ok = tv_max <= TV_BAR and p_min >= P_BAR and omitted == 0
  return {"tv": tv_max, "p": p_min, "omitted": omitted, "ok": ok}


def order_counts(n_draws: int, length: int, seed: int, *, kind: str) -> np.ndarray:
  from itertools import permutations  # noqa: PLC0415

  keys = list(permutations(range(length)))
  index = {perm: i for i, perm in enumerate(keys)}
  counts = np.zeros((len(keys),), dtype=np.int64)
  draw = biased_shuffle if kind == "biased" else shuffle
  for i in range(n_draws):
    perm = tuple(int(v) for v in draw(length, SplitMix64((seed + i) & 0xFFFFFFFF)))
    counts[index[perm]] += 1
  return counts


def summarize_orders(counts: np.ndarray) -> dict[str, float | bool]:
  from scipy.stats import chisquare  # noqa: PLC0415

  expected = np.full(counts.shape, float(np.sum(counts)) / float(counts.shape[0]))
  p_value = float(chisquare(counts.astype(np.float64), expected).pvalue)
  return {"p": p_value, "ok": p_value >= P_BAR}


def _lookup(atom: str, result: dict[str, object]) -> object:
  if atom in result:
    return result[atom]
  if atom == "true":
    return True
  if atom == "false":
    return False
  if re.fullmatch(r"-?\d+", atom):
    return int(atom)
  return float(atom)


def _compare(left: object, op: str, right: object) -> bool:
  if op == "=":
    return left == right
  if op == ">":
    return float(left) > float(right)  # type: ignore[arg-type]
  if op == "<":
    return float(left) < float(right)  # type: ignore[arg-type]
  if op == ">=":
    return float(left) >= float(right)  # type: ignore[arg-type]
  if op == "<=":
    return float(left) <= float(right)  # type: ignore[arg-type]
  msg = f"unknown comparator {op}"
  raise ValueError(msg)


def _eval_atom(expr: str, result: dict[str, object]) -> bool:
  text = expr.strip()
  negated = text.startswith("NOT ")
  if negated:
    text = text[4:].strip()
  matched = _CMP.match(text)
  if matched:
    left = _lookup(matched.group(1), result)
    right = _lookup(matched.group(3), result)
    value = _compare(left, matched.group(2), right)
  else:
    value = bool(result[text])
  return not value if negated else value


def _eval_and(expr: str, result: dict[str, object]) -> bool:
  return all(_eval_atom(part, result) for part in expr.split(" AND "))


def _eval_expr(expr: str, result: dict[str, object]) -> bool:
  return any(_eval_and(part, result) for part in expr.split(" OR "))


def evaluate_outcome(result: dict[str, object]) -> str:
  """First matching sidecar outcome. Local SQL-free stand-in for bathos ``evaluate_outcome``."""
  for name, expr in OUTCOMES:
    if _eval_expr(expr, result):
      return name
  msg = "no outcome matched"
  raise RuntimeError(msg)


def result_template() -> dict[str, object]:
  return {
    "a_cases_total": 0,
    "a_cases_bitwise_cpu": 0,
    "a_cases_bitwise_web": 0,
    "a_max_logprob_abs_diff": 0.0,
    "a_web_ok": True,
    "alias_ok": True,
    "knobs_live": 0,
    "knobs_total": 0,
    "ctrl_frozen_detected": False,
    "ctrl_bad_gumbel_detected": False,
    "ctrl_bad_shuffle_detected": False,
    "ctrl_uniform_order_detected": False,
    "b1_max_abs_nats": 0.0,
    "b2_gumbel_tv_max": 0.0,
    "b2_gumbel_p_min": 1.0,
    "b2_order_p": 1.0,
    "b2_order_fixed_first_violations": 0,
    "b2_order_within_p": 1.0,
    "b2_omitted_draws": 0,
    "b2_n": 0,
    "order_n": 0,
    "browser_enabled": False,
    "budget_exceeded": False,
    "checks_complete": False,
    "smoke": False,
    "git_hash": "",
    "git_clean": False,
    "fixture_ids": [],
    "buckets": [],
    "versions": {},
    "status_note": "",
    "export_ok": True,
  }


def passing_result() -> dict[str, object]:
  result = result_template()
  result.update(
    {
      "a_cases_total": 144,
      "a_cases_bitwise_cpu": 144,
      "a_cases_bitwise_web": 144,
      "a_max_logprob_abs_diff": 0.0,
      "a_web_ok": True,
      "alias_ok": True,
      "knobs_live": 7,
      "knobs_total": 7,
      "browser_enabled": True,
      "smoke": False,
      "ctrl_frozen_detected": True,
      "ctrl_bad_gumbel_detected": True,
      "ctrl_bad_shuffle_detected": True,
      "ctrl_uniform_order_detected": True,
      "b1_max_abs_nats": 0.0,
      "b2_gumbel_tv_max": 0.01,
      "b2_gumbel_p_min": 0.2,
      "b2_order_p": 0.2,
      "b2_order_fixed_first_violations": 0,
      "b2_order_within_p": 0.2,
      "b2_omitted_draws": 0,
      "b2_n": B2_N_FULL,
      "order_n": ORDER_N_FULL,
      "checks_complete": True,
      "git_hash": "0" * 40,
      "git_clean": True,
      "fixture_ids": ["5L33"],
      "buckets": [128],
      "versions": {"jax": "0"},
    },
  )
  return result


def _json_default(value: object) -> object:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.floating):
    return float(value)
  if isinstance(value, np.integer):
    return int(value)
  msg = f"not JSON serializable: {type(value).__name__}"
  raise TypeError(msg)


def _node_bin(explicit: str | None) -> str:
  import layer_c_common as lcc  # noqa: PLC0415

  found = lcc.find_node(explicit)
  if found is None:
    msg = "no node binary found"
    raise IntegrityRefusalError(msg)
  return found


def _run_node(spec: dict[str, object], node: str, *, timeout: float) -> dict[str, object]:
  with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
    json.dump(spec, handle, default=_json_default)
    spec_path = handle.name
  try:
    proc = subprocess.run(  # noqa: S603
      [node, str(RUNSPEC_MJS), spec_path],
      capture_output=True,
      text=True,
      check=False,
      timeout=timeout,
    )
  finally:
    Path(spec_path).unlink()
  if proc.returncode != 0:
    msg = f"runspec.mjs failed: {proc.stderr[-2000:]}"
    raise RuntimeError(msg)
  lines = [line for line in proc.stdout.splitlines() if line.strip()]
  loaded = json.loads(lines[-1])
  if not isinstance(loaded, dict):
    msg = "runspec.mjs did not return an object"
    raise TypeError(msg)
  return loaded


def _unpack_tensor(spec: dict[str, object]) -> np.ndarray:
  dtype = np.float32 if spec["dtype"] == "float32" else np.int32
  shape = tuple(int(dim) for dim in spec["shape"])  # type: ignore[union-attr]
  return np.asarray(spec["data"], dtype=dtype).reshape(shape)


def js_build(
  structure: dict[str, object],
  runspec: dict[str, object],
  node: str,
) -> dict[str, np.ndarray]:
  payload = _run_node({"structure": structure, "runspec": runspec}, node, timeout=120)
  return {key: _unpack_tensor(payload[key]) for key in P07_KEYS}  # type: ignore[arg-type]


def gumbel_noise_js(
  *,
  node: str,
  seed: int,
  rows: int,
  cols: int,
  transform: str,
  out_path: Path,
) -> np.ndarray:
  _run_node(
    {
      "op": "gumbel",
      "seed": seed,
      "rows": rows,
      "cols": cols,
      "transform": transform,
      "out": str(out_path),
    },
    node,
    timeout=600,
  )
  return np.fromfile(out_path, dtype=np.float32).reshape(rows, cols)


def order_counts_js(*, node: str, seed: int, n_draws: int, length: int, kind: str) -> np.ndarray:
  payload = _run_node(
    {"op": "orders", "seed": seed, "n": n_draws, "length": length, "kind": kind},
    node,
    timeout=600,
  )
  return np.asarray(payload["counts"], dtype=np.int64)


def fixed_order_stats_js(*, node: str, seed: int, n_draws: int) -> dict[str, float | int | bool]:
  """Fixed-first violations, designed-group chi-square p, and the uniform-shuffle control.

  L=6 with positions 0 and 1 fixed. The control is the old uniform Fisher-Yates,
  which must put a designed position before a fixed one on at least one draw.
  """
  payload = _run_node(
    {"op": "fixed_order", "seed": seed, "n": n_draws, "length": 6, "fixed": [0, 1]},
    node,
    timeout=600,
  )
  counts = np.asarray(payload["counts"], dtype=np.int64)
  within_p = float(summarize_orders(counts)["p"])
  if not math.isfinite(within_p):
    within_p = 0.0
  violations = int(payload["violations"])
  uniform_violations = int(payload["uniform_violations"])
  return {
    "violations": violations,
    "within_p": within_p,
    "uniform_detected": uniform_violations > 0,
  }


def _git_state(root: Path) -> tuple[str, bool]:
  try:
    rev = subprocess.check_output(
      ["git", "rev-parse", "HEAD"],  # noqa: S607
      cwd=root,
      text=True,
    ).strip()
    porcelain = subprocess.check_output(
      ["git", "status", "--porcelain"],  # noqa: S607
      cwd=root,
      text=True,
    )
  except (OSError, subprocess.CalledProcessError) as exc:
    msg = f"git state unavailable: {exc}"
    raise IntegrityRefusalError(msg) from exc
  return rev, porcelain.strip() == ""


def _versions(node: str) -> dict[str, str]:
  from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
  from importlib.metadata import version as pkg_version  # noqa: PLC0415

  versions: dict[str, str] = {}
  for name in ("jax", "jax2onnx", "onnx", "onnxruntime", "numpy", "equinox", "scipy"):
    try:
      versions[name] = pkg_version(name)
    except PackageNotFoundError:
      versions[name] = "not-installed"
  try:
    proc = subprocess.run(  # noqa: S603
      [node, "--version"],
      capture_output=True,
      text=True,
      check=False,
      timeout=30,
    )
    versions["node"] = proc.stdout.strip() if proc.returncode == 0 else "unknown"
  except OSError:
    versions["node"] = "unknown"
  return versions


def _load_manifest() -> dict[str, object]:
  if not MANIFEST_PATH.is_file():
    msg = f"fixture manifest missing: {MANIFEST_PATH}"
    raise IntegrityRefusalError(msg)
  with MANIFEST_PATH.open() as handle:
    loaded = json.load(handle)
  if not isinstance(loaded, dict):
    msg = "fixture manifest is not an object"
    raise IntegrityRefusalError(msg)
  return loaded


def _parse_geometry(fixture: dict[str, object], cache_dir: Path) -> dict[str, object]:
  from layer_b_ort_calibrate import _build_fixture_inputs  # noqa: PLC0415

  if fixture.get("kind") == "tie_lattice":
    coords, mask, residue_index, chain_index, n_real = _build_fixture_inputs(fixture)
    return {
      "name": str(fixture["name"]),
      "n_real": int(n_real),
      "n_chains": 1,
      "coords": np.asarray(coords, dtype=np.float32),
      "mask": np.asarray(mask, dtype=np.float32),
      "residue_index": np.asarray(residue_index, dtype=np.int32),
      "chain_index": np.asarray(chain_index, dtype=np.int32),
      "chain_ids": ["A"] * int(n_real),
      "native_tokens": np.zeros((int(n_real),), dtype=np.int32),
      "atom37": np.zeros((int(n_real), 37, 3), dtype=np.float32),
      "atom37_mask": np.zeros((int(n_real), 37), dtype=np.float32),
    }

  import fixtures as fixture_mod  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415
  import layer_a_exact as lae  # noqa: PLC0415
  from parse_parity import _canonical_copy  # noqa: PLC0415
  from proxide.chem.residues import atom_order  # noqa: PLC0415

  from aminx.io.parsing import parse_structure  # noqa: PLC0415
  from aminx.utils.aa_convert import af_to_mpnn  # noqa: PLC0415

  try:
    path = fixture_mod.resolve_fixture_path(fixture)
  except (FileNotFoundError, ValueError) as exc:
    msg = str(exc)
    raise IntegrityRefusalError(msg) from exc
  canonical = _canonical_copy(path, cache_dir)
  protein = parse_structure(str(canonical))
  atom37 = np.asarray(protein.coordinates, dtype=np.float32)
  atom37_mask = np.asarray(protein.atom_mask, dtype=np.float32)
  n_idx, ca_idx, c_idx, o_idx = (int(atom_order[name]) for name in ("N", "CA", "C", "O"))
  coords = atom37[:, [n_idx, ca_idx, c_idx, o_idx], :]
  mask = np.asarray(protein.mask, dtype=np.float32)
  chain_index = np.asarray(protein.chain_index, dtype=np.int32)
  residue_index = np.asarray(protein.residue_index, dtype=np.int32)
  # Protein.chain_ids is one label per chain, indexed by chain_index.
  # RunSpec chains_to_design needs one label per residue.
  unique_chain_ids = (
    [str(item) for item in protein.chain_ids] if protein.chain_ids is not None else ["A"]
  )
  chain_ids = [
    unique_chain_ids[int(ordinal)] if int(ordinal) < len(unique_chain_ids) else str(int(ordinal))
    for ordinal in chain_index.tolist()
  ]
  native = np.asarray(af_to_mpnn(jnp.asarray(protein.aatype)), dtype=np.int32)
  reference_module = lae._load_reference_data_utils()  # noqa: SLF001
  lae._assert_tokens_match_reference_parser(  # noqa: SLF001
    str(fixture["name"]),
    canonical,
    native,
    chain_index,
    residue_index,
    unique_chain_ids,
    reference_module,
  )
  return {
    "name": str(fixture["name"]),
    "n_real": int(native.shape[0]),
    "n_chains": len(set(chain_ids)),
    "coords": np.asarray(coords, dtype=np.float32),
    "mask": mask,
    "residue_index": residue_index,
    "chain_index": chain_index,
    "chain_ids": chain_ids,
    "native_tokens": native,
    "atom37": atom37,
    "atom37_mask": atom37_mask,
  }


def _pad_geometry(geom: dict[str, object], bucket: int) -> dict[str, object]:
  from aminx.export.buckets import pad_inputs  # noqa: PLC0415

  padded = pad_inputs(
    np.asarray(geom["coords"]),
    np.asarray(geom["mask"]),
    np.asarray(geom["residue_index"]),
    np.asarray(geom["chain_index"]),
    bucket,
  )
  n_real = int(geom["n_real"])
  n_pad = bucket - n_real
  chain_ids = list(geom["chain_ids"])  # type: ignore[arg-type]
  chain_ids = chain_ids + [chain_ids[-1]] * n_pad
  native = np.pad(np.asarray(geom["native_tokens"], dtype=np.int32), (0, n_pad))
  return {
    "name": geom["name"],
    "n_real": n_real,
    "n_chains": geom["n_chains"],
    "coords": padded["coords"],
    "mask": padded["mask"],
    "residue_index": padded["residue_index"],
    "chain_index": padded["chain_index"],
    "chain_ids": chain_ids,
    "native_tokens": native,
    "coords_shape": [bucket, 4, 3],
  }


def _with_chain_split(geom: dict[str, object]) -> dict[str, object]:
  if int(geom["n_chains"]) > 1:
    return geom
  n_real = int(geom["n_real"])
  mid = n_real // 2
  chain_ids = ["A"] * mid + ["B"] * (n_real - mid)
  chain_index = np.asarray([0] * mid + [1] * (n_real - mid), dtype=np.int32)
  out = dict(geom)
  out["chain_ids"] = chain_ids
  out["chain_index"] = chain_index
  out["n_chains"] = 2
  return out


def _design_chain(geom: dict[str, object]) -> str:
  chain_ids = list(geom["chain_ids"])  # type: ignore[arg-type]
  return str(chain_ids[0])


def _tied_pair(n_real: int) -> list[list[int]]:
  other = min(40, n_real - 1)
  if other <= 1:
    other = n_real - 1
  return [[1, other]]


def _fixed_map(n_real: int) -> dict[str, str]:
  return {str(pos): "W" for pos in (3, 8, 15) if pos < n_real}


def runspec_for(name: str, seed: int, geom: dict[str, object]) -> dict[str, object]:  # noqa: PLR0911
  length = int(np.asarray(geom["mask"]).shape[0])
  n_real = int(geom["n_real"])
  base: dict[str, object] = {
    "seed": int(seed) & 0xFFFFFFFF,
    "decoding_order": "random",
    "temperature": 0.1,
  }
  if name in ("baseline", "temp0.1"):
    return base
  if name == "temp1.0":
    return {**base, "temperature": 1.0}
  if name == "bias+omit":
    return {
      **base,
      "bias_AA": {"A": 1.5},
      "omit_AA": "CX",
      "omit_AA_per_residue": {"4": "W"},
    }
  if name == "fixed_positions":
    return {**base, "fixed_positions": _fixed_map(n_real)}
  if name == "tied_pairs":
    return {**base, "tied_positions": _tied_pair(n_real)}
  if name == "chains_to_design":
    return {**base, "chains_to_design": [_design_chain(geom)]}
  if name == "explicit_order":
    return {**base, "decoding_order": list(range(length - 1, -1, -1))}
  if name == "all_combined":
    return {
      "seed": int(seed) & 0xFFFFFFFF,
      "temperature": 1.0,
      "bias_AA": {"A": 1.5},
      "omit_AA": "CX",
      "omit_AA_per_residue": {"4": "W"},
      "fixed_positions": _fixed_map(n_real),
      "chains_to_design": [_design_chain(geom)],
      "tied_positions": _tied_pair(n_real),
      "decoding_order": list(range(length - 1, -1, -1)),
    }
  msg = f"unknown knob {name}"
  raise ValueError(msg)


def _structure_payload(geom: dict[str, object]) -> dict[str, object]:
  coords = np.asarray(geom["coords"], dtype=np.float32)
  return {
    "coords": coords,
    "coords_shape": [int(coords.shape[0]), 4, 3],
    "mask": np.asarray(geom["mask"], dtype=np.float32),
    "residue_index": np.asarray(geom["residue_index"], dtype=np.int32),
    "chain_index": np.asarray(geom["chain_index"], dtype=np.int32),
    "chain_ids": list(geom["chain_ids"]),  # type: ignore[arg-type]
    "native_tokens": np.asarray(geom["native_tokens"], dtype=np.int32),
  }


def _call_p07(fn: object, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  import jax.numpy as jnp  # noqa: PLC0415

  tokens, log_probs = fn(  # type: ignore[operator]
    jnp.asarray(arrays["coords"]),
    jnp.asarray(arrays["mask"]),
    jnp.asarray(arrays["residue_index"]),
    jnp.asarray(arrays["chain_index"]),
    jnp.asarray(arrays["gumbel_noise"]),
    jnp.asarray(arrays["decoding_order"]),
    jnp.asarray(arrays["bias"]),
    jnp.asarray(arrays["fixed_mask"]),
    jnp.asarray(arrays["fixed_tokens"]),
    jnp.asarray(arrays["temperature"]),
    jnp.asarray(arrays["tie_group_map"]),
  )
  return np.asarray(tokens), np.asarray(log_probs)


def _bias_frozen(fn: object) -> object:
  import jax.numpy as jnp  # noqa: PLC0415

  def wrapped(
    coords: object,
    mask: object,
    residue_index: object,
    chain_index: object,
    gumbel_noise: object,
    decoding_order: object,
    bias: object,
    fixed_mask: object,
    fixed_tokens: object,
    temperature: object,
    tie_group_map_arg: object,
  ) -> object:
    return fn(  # type: ignore[operator]
      coords,
      mask,
      residue_index,
      chain_index,
      gumbel_noise,
      decoding_order,
      jnp.zeros_like(jnp.asarray(bias)),
      fixed_mask,
      fixed_tokens,
      temperature,
      tie_group_map_arg,
    )

  return wrapped


def _real_tokens_differ(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> bool:
  real = np.asarray(mask) > 0
  return not np.array_equal(left[real], right[real])


def _convert_onnx(fn: object, sample: dict[str, np.ndarray], bucket: int, out_path: Path) -> None:
  import jax  # noqa: PLC0415
  import jax2onnx  # noqa: PLC0415

  inputs = tuple(np.asarray(sample[key]) for key in P07_KEYS)
  specs = [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in inputs]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  jax2onnx.to_onnx(
    fn,
    specs,
    model_name=f"p07_sample_L{bucket}",
    output_path=str(out_path),
    return_mode="file",
  )
  embed_external_data(out_path)


def _graph_tensors(graph: object) -> list[object]:
  """Every TensorProto in ``graph``, including Loop/If/Scan subgraph initializers."""
  import onnx  # noqa: PLC0415

  tensors: list[object] = list(graph.initializer)  # type: ignore[attr-defined]
  for node in graph.node:  # type: ignore[attr-defined]
    for attr in node.attribute:
      if attr.type == onnx.AttributeProto.TENSOR:
        tensors.append(attr.t)
      elif attr.type == onnx.AttributeProto.TENSORS:
        tensors.extend(attr.tensors)
      elif attr.type == onnx.AttributeProto.GRAPH:
        tensors.extend(_graph_tensors(attr.g))
      elif attr.type == onnx.AttributeProto.GRAPHS:
        for sub in attr.graphs:
          tensors.extend(_graph_tensors(sub))
  return tensors


def embed_external_data(onnx_path: Path) -> None:
  """Rewrite ``onnx_path`` as one self-contained file.

  jax2onnx stores some Loop-subgraph constants as external data next to the model.
  onnxruntime (CPU) resolves those from disk, but the browser site serves only the
  ``.onnx`` file, so ORT-Web fails session creation ("external data path could not be
  canonicalized"). Both arms load the rewritten file, so they run the same bytes.
  """
  import onnx  # noqa: PLC0415

  model = onnx.load(str(onnx_path), load_external_data=True)
  for tensor in _graph_tensors(model.graph):
    if tensor.data_location == onnx.TensorProto.EXTERNAL:  # type: ignore[attr-defined]
      tensor.data_location = onnx.TensorProto.DEFAULT  # type: ignore[attr-defined]
      del tensor.external_data[:]  # type: ignore[attr-defined]
  onnx.save_model(model, str(onnx_path), save_as_external_data=False)
  reloaded = onnx.load(str(onnx_path), load_external_data=False)
  leftover = [
    tensor.name  # type: ignore[attr-defined]
    for tensor in _graph_tensors(reloaded.graph)
    if tensor.data_location == onnx.TensorProto.EXTERNAL  # type: ignore[attr-defined]
  ]
  if leftover:
    msg = f"P07 ONNX still references external data after embedding: {leftover[:5]}"
    raise ValueError(msg)


def _ort_run(session: object, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  inputs = [np.asarray(arrays[key]) for key in P07_KEYS]
  names = [item.name for item in session.get_inputs()]  # type: ignore[attr-defined]
  feeds = dict(zip(names, inputs, strict=True))
  tokens, log_probs = session.run(None, feeds)  # type: ignore[attr-defined]
  return np.asarray(tokens), np.asarray(log_probs)


def _run_p07_browser(
  *,
  site_dir: Path,
  out_dir: Path,
  node_bin: str,
  timeout_s: float,
) -> dict[str, object]:
  out_dir.mkdir(parents=True, exist_ok=True)
  cmd = [
    node_bin,
    str(RUN_P07_MJS),
    "--site",
    str(site_dir),
    "--out-dir",
    str(out_dir),
    "--num-threads",
    "1",
    "--timeout-ms",
    str(int(timeout_s * 1000)),
  ]
  proc = subprocess.run(  # noqa: S603
    cmd,
    cwd=LAYER_C_DIR,
    capture_output=True,
    text=True,
    timeout=timeout_s + 30.0,
    check=False,
  )
  lines = [line for line in proc.stdout.splitlines() if line.strip()]
  if lines:
    try:
      loaded = json.loads(lines[-1])
      if isinstance(loaded, dict):
        return loaded
    except json.JSONDecodeError:
      pass
  harness_path = out_dir / "harness.json"
  if harness_path.is_file():
    with harness_path.open() as handle:
      loaded = json.load(handle)
    if isinstance(loaded, dict):
      return loaded
  msg = f"run_p07.mjs produced no harness JSON: {proc.stderr[-2000:]}"
  raise RuntimeError(msg)


def _raise_on_failed_cells(harness: dict[str, object], names: list[str]) -> None:
  """Raise if any of ``names`` is missing from ``harness["result"]["cells"]`` or not ok.

  T11d: ``harnessOk`` only proves the browser PAGE ran to completion, not that every
  individual cell inside it succeeded -- a leaked-session ``bad_alloc`` (or the bare
  numeric wasm abort that precedes it) fails a cell without failing the harness, and
  the caller previously discovered this only later, via a confusing
  ``FileNotFoundError`` on a missing ``__tokens.bin``. Fail fast with the real cause.
  """
  result = harness.get("result")
  cells = (result or {}).get("cells", {}) if isinstance(result, dict) else {}
  failed: list[tuple[str, str]] = []
  for name in names:
    cell = cells.get(name) if isinstance(cells, dict) else None
    if not isinstance(cell, dict) or cell.get("ok") is not True:
      error = cell.get("error") if isinstance(cell, dict) else None
      if error is None:
        error = "missing from browser harness result"
      failed.append((name, str(error)[:500]))
  if failed:
    first_name, first_error = failed[0]
    msg = (
      f"browser harness reported {len(failed)}/{len(names)} failed cell(s); "
      f"first failure: {first_name!r}: {first_error}"
    )
    raise RuntimeError(msg)


def _browser_compare(
  cases: list[dict[str, object]],
  onnx_by_bucket: dict[int, Path],
  work: Path,
  node: str,
  timeout_s: float,
) -> tuple[int, float]:
  import layer_c_common as lcc  # noqa: PLC0415

  matched = 0
  max_diff = 0.0
  by_bucket: dict[int, list[dict[str, object]]] = {}
  for case in cases:
    by_bucket.setdefault(int(case["bucket"]), []).append(case)
  dtypes = {
    "coords": "float32",
    "mask": "float32",
    "residue_index": "int32",
    "chain_index": "int32",
    "gumbel_noise": "float32",
    "decoding_order": "int32",
    "bias": "float32",
    "fixed_mask": "float32",
    "fixed_tokens": "int32",
    "temperature": "float32",
    "tie_group_map": "int32",
  }
  for bucket, group in by_bucket.items():
    site = work / f"site_L{bucket}"
    data_dir = site / "data"
    cells = []
    model_name = f"p07_L{bucket}.onnx"
    for case in group:
      arrays = case["arrays"]
      if not isinstance(arrays, dict):
        continue
      inputs = []
      for index, key in enumerate(P07_KEYS):
        inputs.append(
          lcc.write_raw_input(
            np.asarray(arrays[key]),
            data_dir / f"{case['name']}_{index}.bin",
            dtypes[key],
          ),
        )
      cells.append(
        {
          "name": str(case["name"]),
          "onnx": f"models/{model_name}",
          "inputs": inputs,
          "outputs": [
            {"label": "tokens", "dtype": "int32"},
            {"label": "log_probs", "dtype": "float32"},
          ],
        },
      )
    lcc.assemble_site(cells, {model_name: onnx_by_bucket[bucket]}, site)
    harness = _run_p07_browser(
      site_dir=site,
      out_dir=work / f"browser_L{bucket}",
      node_bin=node,
      timeout_s=timeout_s,
    )
    if not harness.get("harnessOk"):
      msg = f"browser harness failed: {harness.get('harnessError')}"
      raise RuntimeError(msg)
    group_names = [str(case["name"]) for case in group if isinstance(case["arrays"], dict)]
    _raise_on_failed_cells(harness, group_names)
    for case in group:
      arrays = case["arrays"]
      if not isinstance(arrays, dict):
        continue
      length = int(np.asarray(arrays["mask"]).shape[0])
      name = str(case["name"])
      out_dir = work / f"browser_L{bucket}"
      tokens = lcc.read_raw_output(out_dir / f"{name}__tokens.bin", "int32", (length,))
      log_probs = lcc.read_raw_output(
        out_dir / f"{name}__log_probs.bin",
        "float32",
        (length, N_AA),
      )
      jax_tokens = np.asarray(case["jax_tokens"])
      jax_lp = np.asarray(case["jax_log_probs"])
      if np.array_equal(tokens, jax_tokens):
        matched += 1
      max_diff = max(max_diff, float(np.max(np.abs(log_probs - jax_lp))))
  return matched, max_diff


def _lane_batch(geom: dict[str, object], lane: str) -> object:
  import layer_a_sampling as las  # noqa: PLC0415

  length = int(geom["n_real"])
  lane_base = las._LANE_BASE[lane]  # noqa: SLF001
  groups = None
  ties = np.arange(length, dtype=np.int64)
  bias = las._x_omit_bias(length)  # noqa: SLF001
  reference_bias = np.zeros((length, N_AA), dtype=np.float32)
  chain_mask = np.ones((length,), dtype=np.float32)
  if lane_base == "P07":
    fixed_seed = las._seed_for(str(geom["name"]) + lane) + 1  # noqa: SLF001
    chain_mask = las._chain_mask_fixed_fraction(length, fixed_seed)  # noqa: SLF001
  elif lane_base == "P08":
    bias = las._omit_aa_bias(length)  # noqa: SLF001
    reference_bias = las._reference_omit_aa_bias(length)  # noqa: SLF001
  elif lane_base == "P09-s":
    pair = _tied_pair(length)[0]
    groups = [pair]
    ties = tie_group_map(length, groups).astype(np.int64)
  fixed_mask = 1.0 - np.asarray(geom["mask"], dtype=np.float32)[:length] * chain_mask
  if groups:
    comparison = np.asarray(
      sorted({member for group in groups for member in group}),
      dtype=np.int64,
    )
  else:
    mask = np.asarray(geom["mask"], dtype=np.float32)[:length]
    comparison = np.where((mask > 0) & (chain_mask > 0))[0].astype(np.int64)
  return las.LaneBatch(
    lane=lane,
    fixture_name=str(geom["name"]),
    length=length,
    x4=np.asarray(geom["coords"])[:length],
    atom37=np.asarray(geom["atom37"])[:length],
    atom37_mask=np.asarray(geom["atom37_mask"])[:length],
    seq_ref=np.asarray(geom["native_tokens"], dtype=np.int64)[:length],
    mask=np.asarray(geom["mask"], dtype=np.float32)[:length],
    residue_index=np.asarray(geom["residue_index"])[:length],
    chain_index=np.asarray(geom["chain_index"])[:length],
    chain_mask=chain_mask,
    fixed_mask=fixed_mask,
    bias=bias,
    reference_bias=reference_bias,
    tie_group_map=ties,
    groups=groups,
    use_side_chain_context=False,
    comparison_positions=comparison,
  )


def _tf_max(
  jax_model: object,
  pt_model: object,
  torch: object,
  batch: object,
  temperature: float,
) -> float:
  import layer_a_exact as lae  # noqa: PLC0415
  import layer_a_sampling as las  # noqa: PLC0415

  seed_i = las._seed_for(batch.fixture_name + batch.lane)  # type: ignore[attr-defined]  # noqa: SLF001
  seq, ref_lp, order, _randn = las.reference_sample_one(
    pt_model,
    torch,
    batch,
    seed_i,
    temperature=temperature,
  )
  raw = las.aminx_conditional_logits(jax_model, batch, seq, order)
  aminx_lp = np.asarray(las._log_softmax(raw))  # noqa: SLF001
  if batch.groups:  # type: ignore[attr-defined]
    from tests.parity.test_full_model_parity import (  # noqa: PLC0415
      _combine_reference_tied_log_probs,
    )

    groups = batch.groups  # type: ignore[attr-defined]
    ref_fused = _combine_reference_tied_log_probs(
      ref_lp,
      tie_groups=groups,
      tie_weights=[[1.0] * len(group) for group in groups],
    )
    member_idx = batch.comparison_positions  # type: ignore[attr-defined]
    return float(lae._max_abs(ref_fused[member_idx], aminx_lp[member_idx]))  # noqa: SLF001
  pos = batch.comparison_positions  # type: ignore[attr-defined]
  if pos.size == 0:
    pos = np.arange(batch.length)  # type: ignore[attr-defined]
  return float(lae._max_abs(ref_lp[pos], aminx_lp[pos]))  # noqa: SLF001


def _explicit_order_max(
  jax_model: object,
  pt_model: object,
  torch: object,
  geom: dict[str, object],
) -> float:
  import layer_a_exact as lae  # noqa: PLC0415
  import layer_a_sampling as las  # noqa: PLC0415

  batch = _lane_batch(geom, "P07@0.1")
  length = batch.length  # type: ignore[attr-defined]
  order = np.arange(length - 1, -1, -1, dtype=np.int64)
  chain_mask = np.ones((length,), dtype=np.float32)
  rank = np.empty((length,), dtype=np.float32)
  rank[order] = np.arange(length, dtype=np.float32)
  scale = np.asarray(batch.mask, dtype=np.float32) * chain_mask + np.float32(1e-4)  # type: ignore[attr-defined]
  randn = (rank / scale).astype(np.float32)
  import dataclasses  # noqa: PLC0415

  replaced = dataclasses.replace(
    batch,  # type: ignore[arg-type]
    chain_mask=chain_mask,
    fixed_mask=np.zeros((length,), dtype=np.float32),
    comparison_positions=np.arange(length, dtype=np.int64),
    groups=None,
    tie_group_map=np.arange(length, dtype=np.int64),
  )
  fd = las._reference_feature_dict_lane(torch, replaced, randn, 0.1, batch_size=1)  # noqa: SLF001
  with torch.no_grad():  # type: ignore[attr-defined]
    out = pt_model.sample(fd)  # type: ignore[attr-defined]
  seq = out["S"].numpy()[0]
  ref_lp = out["log_probs"].numpy()[0]
  got_order = out["decoding_order"].numpy()[0]
  raw = las.aminx_conditional_logits(jax_model, replaced, seq, got_order)
  aminx_lp = np.asarray(las._log_softmax(raw))  # noqa: SLF001
  return float(lae._max_abs(ref_lp, aminx_lp))  # noqa: SLF001


def run_b1(geom: dict[str, object]) -> float:
  import layer_a_common as lac  # noqa: PLC0415

  jax_model, pt_model, torch, _utils = lac.load_full_model("eqx")
  values = [
    _tf_max(jax_model, pt_model, torch, _lane_batch(geom, "P08@1.0"), 1.0),
    _tf_max(jax_model, pt_model, torch, _lane_batch(geom, "P07@0.1"), 0.1),
    _tf_max(jax_model, pt_model, torch, _lane_batch(geom, "P07@1.0"), 1.0),
    _tf_max(jax_model, pt_model, torch, _lane_batch(geom, "P09-s@1.0"), 1.0),
    _explicit_order_max(jax_model, pt_model, torch, geom),
  ]
  logger.info("B1 per-setting max-abs %s", [float(v) for v in values])
  return float(max(values))


def _select_fixtures(
  manifest: dict[str, object],
  buckets: tuple[int, ...],
) -> list[dict[str, object]]:
  from layer_b_ort_calibrate import FIXTURE_BUCKETS  # noqa: PLC0415

  from aminx.io.weights import get_topology_for_checkpoint  # noqa: PLC0415

  k_neighbors = int(get_topology_for_checkpoint("proteinmpnn_v_48_020")["k_neighbors"])
  rows = manifest.get("fixtures")
  if not isinstance(rows, list):
    msg = "manifest has no fixtures list"
    raise IntegrityRefusalError(msg)
  by_name = {str(row["name"]): row for row in rows if isinstance(row, dict)}
  chosen: list[dict[str, object]] = []
  for name in FIXTURE_BUCKETS:
    row = by_name.get(name)
    if row is None:
      continue
    length = int(row["L"])
    if length < k_neighbors:
      continue
    if any(length <= bucket for bucket in buckets):
      chosen.append(row)
  if not chosen:
    msg = "no set-A fixture fits the requested buckets"
    raise IntegrityRefusalError(msg)
  return chosen


def _noise_factory(node: str, transform: str, work: Path) -> object:
  def _noise(seed: int, rows: int, cols: int) -> np.ndarray:
    path = work / f"gumbel_{transform}_{seed}_{rows}.bin"
    return gumbel_noise_js(
      node=node,
      seed=seed,
      rows=rows,
      cols=cols,
      transform=transform,
      out_path=path,
    )

  return _noise


def run(args: argparse.Namespace) -> dict[str, object]:  # noqa: PLR0911, PLR0915
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  started = time.monotonic()
  deadline = started + float(args.budget_s)
  result = result_template()
  result["smoke"] = bool(args.smoke)
  result["browser_enabled"] = bool(args.browser)
  git_hash, git_clean = _git_state(_ROOT)
  result["git_hash"] = git_hash
  result["git_clean"] = git_clean

  buckets = (128,) if args.smoke else FULL_BUCKETS
  seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
  knobs = SMOKE_KNOBS if args.smoke else FULL_KNOBS
  b2_n = B2_N_SMOKE if args.smoke else B2_N_FULL
  order_n = B2_N_SMOKE if args.smoke else ORDER_N_FULL
  result["buckets"] = list(buckets)
  result["b2_n"] = b2_n
  result["order_n"] = order_n

  node = _node_bin(args.node_bin)
  result["versions"] = _versions(node)
  manifest = _load_manifest()
  fixture_rows = _select_fixtures(manifest, buckets)
  result["fixture_ids"] = [str(row["name"]) for row in fixture_rows]
  logger.info("fixtures %s buckets %s knobs %s", result["fixture_ids"], buckets, knobs)

  work = Path(tempfile.mkdtemp(prefix="p07_knobs_"))
  geometries = [_parse_geometry(row, work / "canonical") for row in fixture_rows]

  def expired() -> bool:
    return time.monotonic() > deadline

  gumbel_ok = summarize_gumbel(_noise_factory(node, "gumbel", work), b2_n, 50_000)
  gumbel_bad = summarize_gumbel(_noise_factory(node, "bad", work), b2_n, 50_000)
  fair_counts = order_counts_js(
    node=node,
    seed=7,
    n_draws=order_n,
    length=ORDER_LENGTH,
    kind="fisher",
  )
  bad_counts = order_counts_js(
    node=node,
    seed=7,
    n_draws=order_n,
    length=ORDER_LENGTH,
    kind="biased",
  )
  fair_order = summarize_orders(fair_counts)
  bad_order = summarize_orders(bad_counts)
  fixed_order = fixed_order_stats_js(node=node, seed=7, n_draws=order_n)
  result["b2_gumbel_tv_max"] = float(gumbel_ok["tv"])
  result["b2_gumbel_p_min"] = float(gumbel_ok["p"])
  result["b2_omitted_draws"] = int(gumbel_ok["omitted"])
  result["b2_order_p"] = float(fair_order["p"])
  result["b2_order_fixed_first_violations"] = int(fixed_order["violations"])
  result["b2_order_within_p"] = float(fixed_order["within_p"])
  result["ctrl_bad_gumbel_detected"] = not bool(gumbel_bad["ok"])
  result["ctrl_bad_shuffle_detected"] = not bool(bad_order["ok"])
  result["ctrl_uniform_order_detected"] = bool(fixed_order["uniform_detected"])
  logger.info(
    "B2 gumbel tv=%s p=%s omitted=%s order_p=%s fixed_first=%s within_p=%s "
    "bad_gumbel_ok=%s bad_shuffle_ok=%s uniform_order_detected=%s",
    result["b2_gumbel_tv_max"],
    result["b2_gumbel_p_min"],
    result["b2_omitted_draws"],
    result["b2_order_p"],
    result["b2_order_fixed_first_violations"],
    result["b2_order_within_p"],
    gumbel_bad["ok"],
    bad_order["ok"],
    result["ctrl_uniform_order_detected"],
  )
  if expired():
    result["budget_exceeded"] = True
    result["status_note"] = "budget exhausted after B2"
    return result

  from aminx.export.wrappers import make_p07_sample  # noqa: PLC0415
  from aminx.inference.logits import make_stage_set  # noqa: PLC0415, TID251
  from aminx.io.weights import load_model  # noqa: PLC0415

  model = load_model(checkpoint_id="proteinmpnn_v_48_020")
  p07 = make_p07_sample(model, make_stage_set())
  frozen = _bias_frozen(p07)

  cases: list[dict[str, object]] = []
  baseline_tokens: dict[tuple[str, int, int], np.ndarray] = {}
  live_hits: dict[str, bool] = {name: False for name in knobs if name not in LIVE_SKIP}
  alias_ok = True
  frozen_differed = False
  for bucket in buckets:
    for geom in geometries:
      if int(geom["n_real"]) > bucket:
        continue
      padded = _pad_geometry(geom, bucket)
      split = _pad_geometry(_with_chain_split(geom), bucket)
      for knob in knobs:
        host = split if knob in ("chains_to_design", "all_combined") else padded
        for seed in seeds:
          if expired():
            result["budget_exceeded"] = True
            result["status_note"] = "budget exhausted during check A"
            return result
          spec = runspec_for(knob, seed, host)
          payload = _structure_payload(host)
          arrays = js_build(payload, spec, node)
          tokens, log_probs = _call_p07(p07, arrays)
          case_name = f"{geom['name']}_L{bucket}_{knob}_s{seed}".replace("+", "p").replace(".", "p")
          cases.append(
            {
              "name": case_name,
              "bucket": bucket,
              "knob": knob,
              "seed": seed,
              "fixture": geom["name"],
              "arrays": arrays,
              "jax_tokens": tokens,
              "jax_log_probs": log_probs,
            },
          )
          key = (str(geom["name"]), bucket, seed)
          if knob == "baseline":
            baseline_tokens[key] = tokens
          elif (
            knob == "temp0.1"
            and key in baseline_tokens
            and _real_tokens_differ(tokens, baseline_tokens[key], arrays["mask"])
          ):
            alias_ok = False
          elif (
            knob in live_hits
            and key in baseline_tokens
            and _real_tokens_differ(tokens, baseline_tokens[key], arrays["mask"])
          ):
            live_hits[knob] = True
          if knob == "bias+omit" and key in baseline_tokens:
            frozen_tokens, _frozen_lp = _call_p07(frozen, arrays)
            base_arrays = js_build(payload, runspec_for("baseline", seed, host), node)
            frozen_base, _ = _call_p07(frozen, base_arrays)
            if _real_tokens_differ(frozen_tokens, frozen_base, arrays["mask"]):
              frozen_differed = True

  result["alias_ok"] = alias_ok
  result["knobs_total"] = len(live_hits)
  result["knobs_live"] = sum(1 for hit in live_hits.values() if hit)
  result["ctrl_frozen_detected"] = not frozen_differed and result["knobs_total"] > 0
  logger.info("liveness %s frozen_differed=%s", live_hits, frozen_differed)

  protein = next((geom for geom in geometries if geom["name"] == "5L33"), None)
  if protein is None:
    protein = next(
      (
        geom
        for geom in geometries
        if int(geom["n_chains"]) >= 1 and geom["name"] != "tie_lattice_L96"
      ),
      geometries[0],
    )
  b1_failed = False
  if not expired():
    try:
      result["b1_max_abs_nats"] = run_b1(protein)
    except IntegrityRefusalError:
      raise
    except Exception as exc:
      logger.exception("B1 failed")
      result["status_note"] = f"B1 failed: {type(exc).__name__}: {exc}"
      result["checks_complete"] = False
      b1_failed = True
    else:
      b1_failed = False
  else:
    result["budget_exceeded"] = True
    result["status_note"] = "budget exhausted before B1"
    return result

  import onnxruntime as ort  # noqa: PLC0415

  onnx_by_bucket: dict[int, Path] = {}
  bitwise = 0
  max_diff = 0.0
  sessions: dict[int, object] = {}
  try:
    for bucket in buckets:
      sample = next(case["arrays"] for case in cases if int(case["bucket"]) == bucket)
      if not isinstance(sample, dict):
        continue
      onnx_path = work / f"p07_L{bucket}.onnx"
      logger.info("converting P07 L=%s", bucket)
      _convert_onnx(p07, sample, bucket, onnx_path)
      onnx_by_bucket[bucket] = onnx_path
      options = ort.SessionOptions()
      options.intra_op_num_threads = 4
      sessions[bucket] = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
      )
    for case in cases:
      if expired():
        result["budget_exceeded"] = True
        result["status_note"] = "budget exhausted during ORT"
        return result
      arrays = case["arrays"]
      if not isinstance(arrays, dict):
        continue
      ort_tokens, ort_lp = _ort_run(sessions[int(case["bucket"])], arrays)
      if np.array_equal(ort_tokens, np.asarray(case["jax_tokens"])):
        bitwise += 1
      max_diff = max(
        max_diff,
        float(np.max(np.abs(ort_lp - np.asarray(case["jax_log_probs"])))),
      )
  except Exception as exc:
    logger.exception("export/ORT failed")
    result["export_ok"] = False
    result["status_note"] = f"export/ORT failed: {type(exc).__name__}: {exc}"
    result["a_cases_total"] = len(cases)
    return result

  result["a_cases_total"] = len(cases)
  result["a_cases_bitwise_cpu"] = bitwise
  result["a_max_logprob_abs_diff"] = max_diff
  if args.browser:
    try:
      web_match, web_diff = _browser_compare(
        cases,
        onnx_by_bucket,
        work,
        node,
        float(args.timeout_s),
      )
      result["a_cases_bitwise_web"] = web_match
      result["a_max_logprob_abs_diff"] = max(max_diff, web_diff)
      result["a_web_ok"] = web_match == len(cases)
    except Exception as exc:
      logger.exception("browser arm failed")
      result["status_note"] = f"browser arm failed: {type(exc).__name__}: {exc}"
      result["a_web_ok"] = False
      return result
  else:
    result["a_cases_bitwise_web"] = 0
    result["a_web_ok"] = True

  result["checks_complete"] = not bool(result["budget_exceeded"]) and not b1_failed
  logger.info(
    "A bitwise %s/%s max_logprob %s live %s/%s b1 %s",
    result["a_cases_bitwise_cpu"],
    result["a_cases_total"],
    result["a_max_logprob_abs_diff"],
    result["knobs_live"],
    result["knobs_total"],
    result["b1_max_abs_nats"],
  )
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Result JSON path.")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Bucket 128, 2 seeds, 3 knobs, B2 N=20000.",
  )
  parser.add_argument("--browser", action="store_true", help="Also run ORT-Web via run_p07.mjs.")
  parser.add_argument("--budget-s", type=float, default=3600.0, help="Wall-clock budget.")
  parser.add_argument("--node-bin", default=None, help="Explicit node binary.")
  parser.add_argument("--timeout-s", type=float, default=300.0, help="Browser harness timeout.")
  args = parser.parse_args(argv)

  import layer_a_common as lac  # noqa: PLC0415

  exit_code = 0
  try:
    result = run(args)
    if result.pop("_integrity", False):
      exit_code = 3
  except IntegrityRefusalError as exc:
    logger.exception("integrity refusal")
    result = result_template()
    result["status_note"] = str(exc)
    try:
      git_hash, git_clean = _git_state(_ROOT)
      result["git_hash"] = git_hash
      result["git_clean"] = git_clean
    except IntegrityRefusalError:
      pass
    exit_code = 3
  except SystemExit as exc:
    if exc.code != 3:
      raise
    logger.exception("integrity refusal from a pinned loader")
    result = result_template()
    result["status_note"] = "pinned loader refused"
    exit_code = 3
  lac.emit(result, args.out)
  logger.info("outcome=%s exit=%s out=%s", evaluate_outcome(result), exit_code, args.out)
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
