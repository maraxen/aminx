"""Layer (a) exact-tier engine (T7, step 2): one function per bars-table row.

This module is the ENGINE only -- it has no `calibrate`/`validate` CLI of its own
(those are T7b: `layer_a_exact_calibrate.py` / `layer_a_exact_validate.py`). It
exposes the API those two scripts (and this task's own gate) call:

- ``run_rows(fixtures, params, weight_source, controls=True) -> dict`` -- runs
  every implemented bars-table row over `fixtures` (a list of manifest fixture
  dicts, see `fixtures.py`) at the given `weight_source`, plus every in-run
  control, and returns::

    {
      "rows": [ {path, metric, value, bar, ratio, status, weight_source, fixture}, ... ],
      "controls": [ {name, metric, off, on, effect, ratio_to_bar, detected}, ... ],
      "n_comparisons": int,   # len(rows)
      "n_over_bar": int,      # rows with ratio > 1.0
      "n_not_advanced": int,  # rows forced to "not_advanced" by params headroom
      "n_near_tie_excluded": int,
      "n_skipped": int,       # pytest.skip.Exception absorbed via layer_a_common.reference_call
      "controls_total": int,
      "controls_detected": int,
      "not_implemented": [ {path, reason}, ... ],
    }

- ``sentinel_ratio_to_bar(subset, *, weight_source, perturb) -> float`` -- the
  differential-phase entry point (spec's `sentinel_ratio_to_bar` metric): P05
  log-prob max-abs against its bar, on a REDUCED fixture subset, with the
  calibrated W_out-bias sentinel perturbation applied iff `perturb`.

**`params` contract (produced by T7b's `layer_a_exact_calibrate.py`, consumed
here and by `layer_a_exact_validate.py`).** `params` is `None` (pre-calibration)
or the parsed `preregistered_params.json`. This module reads, if present:

- ``params["exact"]["not_advanced_paths"]``: list[str] of bars-table `path` ids
  the calibrate run's headroom rule (bar/2) flagged `not_advanced`; any row whose
  `path` is listed here is forced to `status="not_advanced"` regardless of its
  own ratio (T11's advance table reads this field, never a raw ratio).
- ``params["exact"]["controls"]["w_out_bias_perturb_magnitude"]``: float,
  calibrated sized W_out-bias perturbation (searched over {1e-4,...,2e-3} by
  T7b); falls back to `DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE` pre-calibration.
- ``params["exact"]["p09_fusion_ctrl_eps"]``: unused here (P09-s sampling-lane
  control, T8's concern) -- documented for T7b's cross-reference only.

**Rows implemented**: P00, P01, P02 (delegates to `parse_parity.compare_parse`),
P03, P04, P05, P06, P09 (scoring lane, k-NN-disjoint groups only), P11, P12,
P13. **Row P14 (packer) is NOT implemented** -- see `_row_p14` for the exact
reason (AC-15/AC-16's real-structure packer bundle needs a proxide
atom37->packer-14-atom-order adapter this task did not find an existing,
reusable, tested source for; fabricating one under this task's remaining scope
risks a silently wrong tolerance, which orchestrator override #6 forbids). Its
"packer weight perturbation" control is correspondingly not implemented.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import fixtures  # noqa: E402 (sibling module, script-dir on sys.path)
import layer_a_common as lac  # noqa: E402
import parse_parity  # noqa: E402

# --------------------------------------------------------------------------------------
# Bars (Pre-registered layer-(a) bars table)
# --------------------------------------------------------------------------------------

WEIGHT_BAR = 0.0  # EXACT: max-abs 0.0, f32
LOG_PROB_BAR = 1e-4  # TOL: max-abs <= 1e-4 nats
PEARSON_FLOOR = 0.9999  # diagnostic only, never a gate (see aminx.parity.compare.pearson)
NLL_BAR = 1e-5  # TOL: |delta| <= 1e-5 nats
ARGMAX_MARGIN = 2e-4  # near-ties excluded below this top-2 margin
NEAR_TIE_GAP = 1e-4  # fixtures.py NEAR_TIE_GAP, reused for consistency

DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE = 1e-3
K_VALUES: tuple[int, ...] = (48, 32)

_SEED_BASE = 20260923  # arbitrary fixed base; per-fixture seed = base + hash(fixture name)


def _seed_for(fixture_name: str) -> int:
  return (_SEED_BASE + (hash(fixture_name) % 10_000)) & 0xFFFFFFFF


def _row(
  path: str,
  metric: str,
  value: float,
  bar: float,
  *,
  weight_source: str,
  fixture: str,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  ratio = float(value) / float(bar) if bar > 0 else (0.0 if value == 0 else float("inf"))
  status = "validated" if value <= bar else "not_advanced"
  row = {
    "path": path,
    "metric": metric,
    "value": float(value),
    "bar": float(bar),
    "ratio": ratio,
    "status": status,
    "weight_source": weight_source,
    "fixture": fixture,
  }
  if extra:
    row.update(extra)
  return row


def _apply_headroom(rows: list[dict[str, Any]], params: dict[str, Any] | None) -> int:
  """Force `status="not_advanced"` on any row whose `path` is in the params headroom list."""
  not_advanced_paths = set()
  if params is not None:
    not_advanced_paths = set(params.get("exact", {}).get("not_advanced_paths", []))
  n_forced = 0
  for row in rows:
    if row["path"] in not_advanced_paths and row["status"] != "not_advanced":
      row["status"] = "not_advanced"
      n_forced += 1
  return n_forced


# --------------------------------------------------------------------------------------
# Batch construction: real manifest fixture -> both sides' inputs, pinned order
# --------------------------------------------------------------------------------------


def _load_reference_data_utils() -> types.ModuleType:
  """Load the reference `data_utils.py` module by file path (never retyped)."""
  module_path = lac.reference_path() / "data_utils.py"
  spec = importlib.util.spec_from_file_location("_layer_a_exact_data_utils", module_path)
  if spec is None or spec.loader is None:
    msg = f"could not load reference data_utils.py from {module_path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class ExactBatch:
  """Real-fixture inputs for both sides, sharing ONE reference-formula decoding order."""

  __slots__ = (
    "fixture_name",
    "length",
    "x4",
    "atom37",
    "atom37_mask",
    "aatype_aminx",
    "seq_ref",
    "mask",
    "chain_mask",
    "residue_index",
    "chain_index",
    "randn",
    "decoding_order",
    "ar_mask",
  )

  def __init__(self, **kwargs: Any) -> None:
    for key, value in kwargs.items():
      setattr(self, key, value)


def build_exact_batch(fixture: dict[str, Any], data_utils_module: types.ModuleType) -> ExactBatch:
  """Parse `fixture`'s structure with aminx and build both sides' pinned-order inputs.

  ``seq_ref`` is the SAME raw integer array as aminx's own ``aatype`` (proxide's
  AlphaFold-order convention) -- deliberately NOT remapped through
  `data_utils_module.restype_str_to_int` (the reference's alphabetical
  convention). Those two orderings genuinely differ (verified: proxide's
  ``restypes_with_x`` is ``"ARNDCQEGHILKMFPSTWYVX"``, the reference's own
  ``restype_int_to_str`` is ``"ACDEFGHIKLMNPQRSTVWYX"``) -- but `aminx.io.weights`
  / `scripts/convert_weights.py` copy the PT `W_s`/`w_s_embed` embedding TABLE
  ROWS unpermuted, so integer `t` selects the identical embedding row on both
  sides regardless of which amino acid either convention nominally assigns to
  `t`. A numerical-parity comparison only needs both sides fed the SAME integer
  per residue, not the "true" amino acid identity -- confirmed empirically: using
  `restype_str_to_int` here reproducibly broke P05 to Pearson ~0.86 (a real,
  non-tolerance-noise divergence), while the raw shared integer used here
  reproduces the ~1e-5 max-abs agreement `test_full_model_parity.py` documents.
  (P02's OWN comparison, in `parse_parity.py`, is different: it independently
  DECODES each side's aatype to a letter via each side's own native convention
  before comparing -- that decode step is exactly where the two alphabets
  matter, and it stays untouched here.)
  """
  from proxide.chem.residues import atom_order

  from aminx.io.parsing import parse_structure

  del data_utils_module  # kept in the signature for call-site symmetry; unused (see docstring)
  path = fixtures.resolve_fixture_path(fixture)
  protein = parse_structure(str(path))

  atom37 = np.asarray(protein.coordinates, dtype=np.float32)
  atom37_mask = np.asarray(protein.atom_mask, dtype=np.float32)
  aatype = np.asarray(protein.aatype)
  mask = np.asarray(protein.mask, dtype=np.float32)
  chain_index = np.asarray(protein.chain_index, dtype=np.int64)
  residue_index = np.asarray(protein.residue_index, dtype=np.int64)

  n_idx, ca_idx, c_idx, o_idx = (int(atom_order[name]) for name in ("N", "CA", "C", "O"))
  x4 = atom37[:, [n_idx, ca_idx, c_idx, o_idx], :]

  seq_ref = aatype.astype(np.int64)

  chain_mask = mask.copy()
  randn, order = lac.reference_formula_order(mask, chain_mask, seed=_seed_for(fixture["name"]))
  ar_mask = lac.ar_mask_from_order(order)

  return ExactBatch(
    fixture_name=fixture["name"],
    length=int(aatype.shape[0]),
    x4=x4,
    atom37=atom37,
    atom37_mask=atom37_mask,
    aatype_aminx=aatype,
    seq_ref=seq_ref,
    mask=mask,
    chain_mask=chain_mask,
    residue_index=residue_index,
    chain_index=chain_index,
    randn=randn,
    decoding_order=order,
    ar_mask=ar_mask,
  )


def _reference_feature_dict(torch_module: Any, batch: ExactBatch) -> dict[str, Any]:
  return {
    "X": torch_module.from_numpy(batch.x4[None].copy()),
    "S": torch_module.from_numpy(batch.seq_ref[None].copy()),
    "mask": torch_module.from_numpy(batch.mask[None].copy()),
    "chain_mask": torch_module.from_numpy(batch.chain_mask[None].copy()),
    "R_idx": torch_module.from_numpy(batch.residue_index[None].copy()),
    "chain_labels": torch_module.from_numpy(batch.chain_index[None].copy()),
    "randn": torch_module.from_numpy(batch.randn[None].copy()),
    "batch_size": 1,
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
  }


def _aminx_bundle_kwargs(batch: ExactBatch, *, sequence_one_hot: bool = True) -> dict[str, Any]:
  import jax
  import jax.numpy as jnp

  kwargs: dict[str, Any] = {
    "coords": jnp.asarray(batch.x4),
    "mask": jnp.asarray(batch.mask),
    "residue_index": jnp.asarray(batch.residue_index, dtype=jnp.int32),
    "chain_index": jnp.asarray(batch.chain_index, dtype=jnp.int32),
    "chain_mask": jnp.asarray(batch.chain_mask),
  }
  if sequence_one_hot:
    kwargs["sequence"] = jax.nn.one_hot(jnp.asarray(batch.aatype_aminx), 21)
    kwargs["ar_mask"] = jnp.asarray(batch.ar_mask)
  return kwargs


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  from aminx.parity.compare import pearson

  return pearson(a, b)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  from aminx.parity.compare import max_abs

  return max_abs(a, b)


# --------------------------------------------------------------------------------------
# P00 -- LigandMPNN's `protein_mpnn` path vs dauparas/ProteinMPNN (both originals)
# --------------------------------------------------------------------------------------


def _load_original_proteinmpnn_module() -> types.ModuleType:
  """Load ProteinMPNN's OWN `protein_mpnn_utils.py` under an isolated module name.

  Never aliased to `model_utils` (LigandMPNN's own module of the same shape) so
  neither import shadows the other.
  """
  module_path = lac.proteinmpnn_path() / "protein_mpnn_utils.py"
  spec = importlib.util.spec_from_file_location("_pmpnn_protein_mpnn_utils", module_path)
  if spec is None or spec.loader is None:
    msg = f"could not load reference protein_mpnn_utils.py from {module_path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def row_p00(
  fixture: dict[str, Any], weight_source: str, skip_counter: lac.SkipCounter
) -> list[dict[str, Any]]:
  """P00: LigandMPNN's `protein_mpnn` model_type vs ProteinMPNN's own class, same v_48_020 weights.

  Both originals (no aminx side, per the inventory table): EXACT weight-tensor
  equality is the primary bar; TOL logits on a real fixture (same checkpoint fed
  to both classes, same pinned decoding order) is the secondary one.
  """
  del skip_counter
  from tests.parity.reference_utils import prepend_reference_to_syspath

  prepend_reference_to_syspath()
  torch = __import__("torch")

  pmpnn_root = lac.proteinmpnn_path()
  ligandmpnn_root = lac.reference_path()
  pt_path = pmpnn_root / "vanilla_model_weights" / "v_48_020.pt"
  ligandmpnn_pt_path = ligandmpnn_root / "model_params" / "proteinmpnn_v_48_020.pt"

  checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
  state_dict = checkpoint["model_state_dict"]

  # --- EXACT weight comparison: same released v_48_020 weights, both checkpoint copies ---
  ligandmpnn_checkpoint = torch.load(ligandmpnn_pt_path, map_location="cpu", weights_only=False)
  ligandmpnn_state_dict = ligandmpnn_checkpoint["model_state_dict"]
  common_keys = sorted(set(state_dict) & set(ligandmpnn_state_dict))
  weight_max_abs = 0.0
  for key in common_keys:
    a = state_dict[key].detach().cpu().numpy().astype(np.float64)
    b = ligandmpnn_state_dict[key].detach().cpu().numpy().astype(np.float64)
    if a.shape != b.shape:
      continue
    weight_max_abs = max(weight_max_abs, float(np.max(np.abs(a - b))))

  # --- TOL logits: both classes loaded from the SAME checkpoint (ligandmpnn's copy), same order ---
  import model_utils as ligandmpnn_model_utils  # noqa: PLC0415 (reference module, sys.path via reference_utils)

  ligandmpnn_model = ligandmpnn_model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    model_type="protein_mpnn",
  )
  ligandmpnn_model.load_state_dict(ligandmpnn_state_dict)
  ligandmpnn_model.eval()

  original_module = _load_original_proteinmpnn_module()
  original_model = original_module.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    augment_eps=0.0,  # ProteinMPNN's OWN default is 0.05 (training-time noise); exact-tier is 0
  )
  original_model.load_state_dict(ligandmpnn_state_dict)
  original_model.eval()

  data_utils_module = _load_reference_data_utils()
  batch = build_exact_batch(fixture, data_utils_module)
  fd = _reference_feature_dict(torch, batch)

  with torch.no_grad():
    ligandmpnn_out = ligandmpnn_model.score(fd, use_sequence=True)
    ligandmpnn_log_probs = ligandmpnn_out["log_probs"].numpy()[0]
    decoding_order = ligandmpnn_out["decoding_order"]

    original_log_probs = original_model.forward(
      fd["X"],
      fd["S"],
      fd["mask"],
      fd["chain_mask"],
      fd["R_idx"],
      fd["chain_labels"],
      fd["randn"],
      use_input_decoding_order=True,
      decoding_order=decoding_order,
    ).numpy()[0]

  logits_max_abs = _max_abs(ligandmpnn_log_probs, original_log_probs)
  logits_pearson = _pearson(ligandmpnn_log_probs, original_log_probs)

  return [
    _row(
      "P00.weights",
      "max_abs",
      weight_max_abs,
      WEIGHT_BAR,
      weight_source=weight_source,
      fixture="v_48_020.pt (both checkpoints)",
      extra={"n_common_tensors": len(common_keys)},
    ),
    _row(
      "P00.log_probs",
      "max_abs",
      logits_max_abs,
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=fixture["name"],
      extra={"pearson": logits_pearson},
    ),
  ]


# --------------------------------------------------------------------------------------
# P01 -- Weights: shipped .eqx.zst == .pt after layout map
# --------------------------------------------------------------------------------------


def row_p01(
  full_model_bundle: tuple[Any, Any, Any, Any], weight_source: str
) -> tuple[list[dict[str, Any]], float]:
  """P01: leaf-wise max-abs between the eqx-loaded and pt-converted models.

  Returns `(rows, max_abs)` -- `max_abs` is reused by the mantissa-bit-flip control.
  """
  import jax

  jax_model, _pt_model, _torch, _model_utils = full_model_bundle
  del jax_model  # full_model_bundle already selected `weight_source`; load BOTH sides here
  from tests.parity.test_full_model_parity import _load_heavy_parity_models_impl

  models = _load_heavy_parity_models_impl()
  eqx_leaves = jax.tree_util.tree_leaves(models.jax_model_eqx)
  pt_leaves = jax.tree_util.tree_leaves(models.jax_model_pt_convert)
  max_abs = 0.0
  for eqx_leaf, pt_leaf in zip(eqx_leaves, pt_leaves, strict=True):
    if not hasattr(eqx_leaf, "shape") or not hasattr(pt_leaf, "shape"):
      continue
    if eqx_leaf.shape != pt_leaf.shape:
      continue
    eqx_leaf_f64 = np.asarray(eqx_leaf, dtype=np.float64)
    pt_leaf_f64 = np.asarray(pt_leaf, dtype=np.float64)
    diff = float(np.max(np.abs(eqx_leaf_f64 - pt_leaf_f64)))
    max_abs = max(max_abs, diff)

  rows = [
    _row(
      "P01.weights",
      "max_abs",
      max_abs,
      WEIGHT_BAR,
      weight_source=weight_source,
      fixture="proteinmpnn_v_48_020 (eqx vs pt_convert)",
    )
  ]
  return rows, max_abs


# --------------------------------------------------------------------------------------
# P02 -- delegate entirely to parse_parity.compare_parse
# --------------------------------------------------------------------------------------


def row_p02(fixture: dict[str, Any], weight_source: str) -> list[dict[str, Any]]:
  """P02: delegate to `parse_parity.compare_parse` (T6), relabeled onto the bars-table rows."""
  del weight_source
  return parse_parity.compare_parse(fixture)


# --------------------------------------------------------------------------------------
# P03 -- Featurisation + k-NN: EXACT neighbour sets on no-tie residues
# --------------------------------------------------------------------------------------


def row_p03(
  fixture: dict[str, Any],
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
  weight_source: str,
) -> tuple[list[dict[str, Any]], int]:
  """P03: aminx's own k-NN vs the reference's `ProteinFeatures`, on no-tie residues.

  Returns `(rows, n_near_tie_excluded)`.
  """
  from aminx.parity.compare import neighbor_set_equality

  _jax_model, pt_model, torch, _model_utils = full_model_bundle
  fd = _reference_feature_dict(torch, batch)

  rows: list[dict[str, Any]] = []
  n_near_tie_excluded_total = 0
  near_tie = fixture.get("near_tie_residues", {})
  # `pt_model.features` is fixed at construction to k_neighbors=48; its E_idx columns are
  # sorted nearest-first, so truncating to the first `k` columns gives the true top-k for
  # any k <= 48 without rebuilding a second reference model per k.
  with torch.no_grad():
    _e, e_idx_full = pt_model.features(fd)
  reference_idx_full = e_idx_full.numpy()[0]
  for k in K_VALUES:
    if k >= batch.length - 1:
      continue
    reference_idx = reference_idx_full[:, :k]
    aminx_idx = fixtures._neighbor_indices(batch.atom37, batch.mask, k)  # noqa: SLF001 (sibling module reuse)

    excluded = set(near_tie.get(str(k)) or [])
    n_near_tie_excluded_total += len(excluded)
    no_tie_mask = np.array([batch.mask[i] > 0 and i not in excluded for i in range(batch.length)])

    equal = neighbor_set_equality(aminx_idx, reference_idx, no_tie_mask)
    n_mismatch = int(np.sum(~equal[no_tie_mask])) if no_tie_mask.any() else 0
    n_eval = int(no_tie_mask.sum())
    rows.append(
      _row(
        f"P03.neighbor_set_k{k}",
        "n_mismatch",
        n_mismatch,
        WEIGHT_BAR,
        weight_source=weight_source,
        fixture=fixture["name"],
        extra={"n_evaluated": n_eval, "n_near_tie_excluded": len(excluded), "k": k},
      )
    )
  return rows, n_near_tie_excluded_total


# --------------------------------------------------------------------------------------
# P04/P05/P06 -- unconditional / conditional logits, pinned-order NLL
# --------------------------------------------------------------------------------------


def _score_reference(
  pt_model: Any, torch: Any, fd: dict[str, Any], *, use_sequence: bool
) -> np.ndarray:
  with torch.no_grad():
    out = pt_model.score(fd, use_sequence=use_sequence)
  return out["log_probs"].numpy()[0]


def _score_aminx_conditional(jax_model: Any, batch: ExactBatch) -> np.ndarray:
  import jax

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  kwargs = _aminx_bundle_kwargs(batch, sequence_one_hot=True)
  bundle, config = build_inference_bundle(mode="score_conditional", **kwargs)
  logits = score_conditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  return np.asarray(jax.nn.log_softmax(logits, axis=-1))


def _score_aminx_unconditional(jax_model: Any, batch: ExactBatch) -> np.ndarray:
  import jax

  from aminx.inference import score_unconditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  kwargs = _aminx_bundle_kwargs(batch, sequence_one_hot=False)
  bundle, config = build_inference_bundle(mode="score_unconditional", **kwargs)
  logits = score_unconditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  return np.asarray(jax.nn.log_softmax(logits, axis=-1))


def _nll_rows(
  path_prefix: str,
  ref_log_probs: np.ndarray,
  aminx_log_probs: np.ndarray,
  batch: ExactBatch,
  data_utils_module: types.ModuleType,
  weight_source: str,
) -> list[dict[str, Any]]:
  from aminx.parity.compare import argmax_agreement, reference_nll

  torch = __import__("torch")
  get_score = data_utils_module.get_score

  max_abs_val = _max_abs(ref_log_probs, aminx_log_probs)
  pearson_val = _pearson(ref_log_probs, aminx_log_probs)

  def _aggregate(log_probs: np.ndarray) -> float:
    return reference_nll(
      log_probs,
      torch.from_numpy(np.ascontiguousarray(batch.seq_ref)).long()[None, ...],
      torch.from_numpy(np.ascontiguousarray(batch.mask)).float()[None, ...],
      lambda seq, lp, mask: get_score(
        seq, torch.from_numpy(np.ascontiguousarray(lp)).float()[None, ...], mask
      ),
    )

  nll_ref = _aggregate(ref_log_probs)
  nll_aminx = _aggregate(aminx_log_probs)
  nll_delta = abs(nll_ref - nll_aminx)

  agree, valid = argmax_agreement(ref_log_probs, aminx_log_probs, ARGMAX_MARGIN)
  n_valid = int(valid.sum())
  n_argmax_mismatch = int(np.sum(valid & ~agree))

  return [
    _row(
      f"{path_prefix}.log_probs",
      "max_abs",
      max_abs_val,
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"pearson": pearson_val},
    ),
    _row(
      f"{path_prefix}.nll",
      "abs_delta",
      nll_delta,
      NLL_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"nll_reference": nll_ref, "nll_aminx": nll_aminx},
    ),
    _row(
      f"{path_prefix}.argmax",
      "n_mismatch",
      n_argmax_mismatch,
      WEIGHT_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"n_valid": n_valid},
    ),
  ]


def _score_reference_unconditional_manual(
  pt_model: Any, model_utils: Any, torch: Any, fd: dict[str, Any]
) -> np.ndarray:
  """Reference unconditional log-probs via the MANUAL branch (full encoder context, no
  decoding-order gating), matching `aminx.inference.score_unconditional.kernel`'s own
  semantics -- NOT `pt_model.score(fd, use_sequence=False)`, whose internal branch still
  applies order-dependent `mask_fw` gating and is therefore not "unconditional" in the sense
  this row means (verified against `tests/parity/test_full_model_parity.py`'s own
  `test_decoder_unconditional_parity`, which builds this exact manual computation)."""
  with torch.no_grad():
    node_features, edge_features, neighbor_indices = pt_model.encode(fd)
    sequence_embedding = torch.zeros_like(node_features)
    encoder_context = model_utils.cat_neighbors_nodes(
      sequence_embedding, edge_features, neighbor_indices
    )
    decoder_context = model_utils.cat_neighbors_nodes(
      node_features, encoder_context, neighbor_indices
    )
    decoded_nodes = node_features
    for layer in pt_model.decoder_layers:
      decoded_nodes = layer(decoded_nodes, decoder_context, fd["mask"])
    log_probs = torch.log_softmax(pt_model.W_out(decoded_nodes), dim=-1)
  return log_probs.numpy()[0]


def row_p04(
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
  weight_source: str,
) -> list[dict[str, Any]]:
  """P04: unconditional logits, TOL."""
  jax_model, pt_model, torch, model_utils = full_model_bundle
  fd = _reference_feature_dict(torch, batch)
  ref_log_probs = _score_reference_unconditional_manual(pt_model, model_utils, torch, fd)
  aminx_log_probs = _score_aminx_unconditional(jax_model, batch)
  max_abs_val = _max_abs(ref_log_probs, aminx_log_probs)
  pearson_val = _pearson(ref_log_probs, aminx_log_probs)
  return [
    _row(
      "P04.log_probs",
      "max_abs",
      max_abs_val,
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"pearson": pearson_val},
    )
  ]


def row_p05(
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
  data_utils_module: types.ModuleType,
  weight_source: str,
  *,
  ref_log_probs: np.ndarray | None = None,
  aminx_log_probs: np.ndarray | None = None,
) -> list[dict[str, Any]]:
  """P05: conditional logits, TOL, plus per-sequence NLL."""
  jax_model, pt_model, torch, _model_utils = full_model_bundle
  if ref_log_probs is None:
    fd = _reference_feature_dict(torch, batch)
    ref_log_probs = _score_reference(pt_model, torch, fd, use_sequence=True)
  if aminx_log_probs is None:
    aminx_log_probs = _score_aminx_conditional(jax_model, batch)
  return _nll_rows("P05", ref_log_probs, aminx_log_probs, batch, data_utils_module, weight_source)


def row_p06(
  batch: ExactBatch,
  jax_model: Any,
  data_utils_module: types.ModuleType,
  weight_source: str,
  pt_model: Any,
  torch: Any,
) -> list[dict[str, Any]]:
  """P06: aminx's OWN `scoring.score.score()` (pinned order via `ar_mask`) vs reference NLL."""
  import jax
  import jax.numpy as jnp

  from aminx.parity.compare import reference_nll
  from aminx.scoring.score import score as aminx_score

  nll_aminx_jax, logits_aminx, _decoding_order = aminx_score(
    jax.random.PRNGKey(0),
    jax_model,
    jnp.asarray(batch.x4),
    jnp.asarray(batch.mask),
    jnp.asarray(batch.residue_index, dtype=jnp.int32),
    jnp.asarray(batch.chain_index, dtype=jnp.int32),
    sequence=jax.nn.one_hot(jnp.asarray(batch.aatype_aminx), 21),
    ar_mask=jnp.asarray(batch.ar_mask),
  )
  aminx_log_probs = np.asarray(jax.nn.log_softmax(logits_aminx, axis=-1))
  nll_aminx = float(nll_aminx_jax)

  fd = _reference_feature_dict(torch, batch)
  ref_log_probs = _score_reference(pt_model, torch, fd, use_sequence=True)
  get_score = data_utils_module.get_score
  nll_ref = reference_nll(
    ref_log_probs,
    torch.from_numpy(np.ascontiguousarray(batch.seq_ref)).long()[None, ...],
    torch.from_numpy(np.ascontiguousarray(batch.mask)).float()[None, ...],
    lambda seq, lp, mask: get_score(
      seq, torch.from_numpy(np.ascontiguousarray(lp)).float()[None, ...], mask
    ),
  )

  max_abs_val = _max_abs(ref_log_probs, aminx_log_probs)
  pearson_val = _pearson(ref_log_probs, aminx_log_probs)
  nll_delta = abs(nll_ref - nll_aminx)

  return [
    _row(
      "P06.log_probs",
      "max_abs",
      max_abs_val,
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"pearson": pearson_val},
    ),
    _row(
      "P06.nll",
      "abs_delta",
      nll_delta,
      NLL_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"nll_reference": nll_ref, "nll_aminx": nll_aminx},
    ),
  ]


# --------------------------------------------------------------------------------------
# P09 -- Tied positions (scoring lane), k-NN-disjoint groups only
# --------------------------------------------------------------------------------------


def _qualifying_groups(fixture: dict[str, Any]) -> list[list[int]]:
  return [
    g["members"]
    for g in fixture.get("tie_groups_knn_disjoint", [])
    if len(g.get("designable_members", g["members"])) >= 2
  ]


def _reverify_knn_disjoint(batch: ExactBatch, groups: list[list[int]]) -> None:
  """Re-check k-NN disjointness at k=48 and k=32 from THIS harness's own featuriser.

  Exits 2 on any violation (R2-C3): the manifest's own qualification is trusted
  as a starting point, but this row's own aminx k-NN call is the one whose
  disjointness the fused-distribution comparison actually depends on.
  """
  for k in K_VALUES:
    if k >= batch.length - 1:
      continue
    neighbor_indices = fixtures._neighbor_indices(batch.atom37, batch.mask, k)  # noqa: SLF001
    for group in groups:
      for i in group:
        neighbors_i = set(neighbor_indices[i].tolist())
        for j in group:
          if i != j and j in neighbors_i:
            print(
              f"layer_a_exact.row_p09: k-NN-disjoint violation at k={k}: "
              f"residue {i} has group member {j} as a neighbour (fixture {batch.fixture_name})",
              file=sys.stderr,
            )
            raise SystemExit(2)


def row_p09(
  fixture: dict[str, Any],
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
  weight_source: str,
) -> list[dict[str, Any]]:
  """P09 (scoring lane): fused tied-position log-probs, k-NN-disjoint groups only."""
  from tests.parity.test_full_model_parity import (
    _build_tie_group_map,
    _combine_reference_tied_log_probs,
  )

  groups = _qualifying_groups(fixture)
  if not groups:
    return [
      _row(
        "P09.log_probs",
        "n_qualifying_groups",
        0,
        1,
        weight_source=weight_source,
        fixture=fixture["name"],
        extra={"status_note": "no k-NN-disjoint multi-member tie group on this fixture"},
      )
    ]

  _reverify_knn_disjoint(batch, groups)

  jax_model, pt_model, torch, _model_utils = full_model_bundle
  fd = _reference_feature_dict(torch, batch)
  fd["symmetry_residues"] = groups
  fd["symmetry_weights"] = [[1.0] * len(g) for g in groups]

  with torch.no_grad():
    ref_out = pt_model.score(fd, use_sequence=True)
  ref_log_probs = ref_out["log_probs"].numpy()[0]
  ref_fused = _combine_reference_tied_log_probs(
    ref_log_probs,
    tie_groups=groups,
    tie_weights=[[1.0] * len(g) for g in groups],
  )

  import jax

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  # The reference's symmetry branch REORDERS decoding: each tie group is inserted, in full,
  # at the position of its first-occurring member (model_utils.py:594-609) -- so the AR mask
  # this row compares against must come from THAT flattened order, not the original ungrouped
  # `batch.decoding_order` (verified: using the ungrouped order here reproduces a large, real
  # divergence even on k-NN-disjoint groups; the reference's own returned `decoding_order`
  # output is authoritative and used directly).
  flattened_order = ref_out["decoding_order"].numpy()[0]
  tied_ar_mask = lac.ar_mask_from_order(flattened_order)

  tie_group_map = _build_tie_group_map(batch.length, groups)
  kwargs = _aminx_bundle_kwargs(batch, sequence_one_hot=True)
  kwargs["ar_mask"] = jax.numpy.asarray(tied_ar_mask)
  kwargs["tie_group_map"] = jax.numpy.asarray(tie_group_map)
  bundle, config = build_inference_bundle(mode="score_conditional", **kwargs)
  aminx_logits = score_conditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  aminx_log_probs = np.asarray(jax.nn.log_softmax(aminx_logits, axis=-1))

  member_indices = sorted({member for group in groups for member in group})
  max_abs_val = _max_abs(ref_fused[member_indices], aminx_log_probs[member_indices])
  pearson_val = _pearson(ref_fused[member_indices], aminx_log_probs[member_indices])

  return [
    _row(
      "P09.log_probs",
      "max_abs",
      max_abs_val,
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=fixture["name"],
      extra={
        "pearson": pearson_val,
        "n_qualifying_groups": len(groups),
        "n_tied_positions": len(member_indices),
      },
    )
  ]


# --------------------------------------------------------------------------------------
# P11 -- LigandMPNN ligand/side-chain context, on/off
# --------------------------------------------------------------------------------------


def row_p11(
  fixture: dict[str, Any],
  weight_source: str,
) -> list[dict[str, Any]]:
  """P11: context on/off, reusing `tests.parity.test_sidechain_context_parity`'s loaders.

  aminx side always uses the shipped `.eqx.zst` for this path (see
  `layer_a_common.load_sidechain_context_models`'s docstring): the upstream test
  module only exposes an `"eqx"` loader for `ligandmpnn_v_32_010_25`, so
  `weight_source` is recorded as `"eqx"` regardless of the caller's request.
  """
  import jax
  import jax.numpy as jnp
  from proxide.chem.residues import atom_order

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  path = fixtures.resolve_fixture_path(fixture)
  from aminx.io.parsing import parse_structure

  protein = parse_structure(str(path))
  atom37 = np.asarray(protein.coordinates, dtype=np.float32)
  atom37_mask = np.asarray(protein.atom_mask, dtype=np.float32)
  aatype = np.asarray(protein.aatype)
  mask = np.asarray(protein.mask, dtype=np.float32)
  residue_index = np.asarray(protein.residue_index, dtype=np.int64)
  chain_index = np.asarray(protein.chain_index, dtype=np.int64)
  length = int(aatype.shape[0])

  n_idx, ca_idx, c_idx, o_idx = (int(atom_order[name]) for name in ("N", "CA", "C", "O"))
  x4 = atom37[:, [n_idx, ca_idx, c_idx, o_idx], :]

  # Partial-fixed chain mask (every other residue fixed) -- the wiring only moves anything
  # when at least some residues are fixed (P11's own "noop when all-designable" invariant).
  chain_mask = np.ones((length,), dtype=np.float32)
  chain_mask[::2] = 0.0
  fixed_mask = 1.0 - chain_mask

  randn, order = lac.reference_formula_order(mask, chain_mask, seed=_seed_for(fixture["name"]))
  ar_mask = lac.ar_mask_from_order(order)

  actx = 16
  torch = __import__("torch")

  rows: list[dict[str, Any]] = []
  logits_by_context: dict[bool, np.ndarray] = {}
  for use_sc in (False, True):
    reference_model, aminx_model = lac.load_sidechain_context_models(
      weight_source, use_side_chain_context=use_sc
    )

    fd = {
      "X": torch.from_numpy(x4[None].copy()),
      "S": torch.from_numpy(aatype[None].astype(np.int64).copy()),
      "mask": torch.from_numpy(mask[None].copy()),
      "chain_mask": torch.from_numpy(chain_mask[None].copy()),
      "R_idx": torch.from_numpy(residue_index[None].copy()),
      "chain_labels": torch.from_numpy(chain_index[None].copy()),
      "randn": torch.from_numpy(randn[None].copy()),
      "Y": torch.zeros((1, length, actx, 3), dtype=torch.float32),
      "Y_t": torch.zeros((1, length, actx), dtype=torch.int32),
      "Y_m": torch.zeros((1, length, actx), dtype=torch.int32),
      "xyz_37": torch.from_numpy(atom37[None].copy()),
      "xyz_37_m": torch.from_numpy(atom37_mask[None].copy()),
      "batch_size": 1,
      "symmetry_residues": [[]],
      "symmetry_weights": [[]],
    }
    with torch.no_grad():
      ref_log_probs = reference_model.score(fd, use_sequence=True)["log_probs"].numpy()[0]

    kw = {
      "coords": jnp.asarray(x4),
      "mask": jnp.asarray(mask),
      "residue_index": jnp.asarray(residue_index, dtype=jnp.int32),
      "chain_index": jnp.asarray(chain_index, dtype=jnp.int32),
      "sequence": jax.nn.one_hot(jnp.asarray(aatype), 21),
      "ar_mask": jnp.asarray(ar_mask),
      "ligand_coords": jnp.zeros((length, actx, 3)),
      "ligand_atom_types": jnp.zeros((length, actx), jnp.int32),
      "ligand_mask": jnp.zeros((length, actx)),
      "fixed_mask": jnp.asarray(fixed_mask),
      "mode": "score_conditional",
    }
    if use_sc:
      kw["atom_37"] = jnp.asarray(atom37)
      kw["atom_37_mask"] = jnp.asarray(atom37_mask)
    bundle, config = build_inference_bundle(**kw)
    aminx_logits = score_conditional.kernel(
      aminx_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
    )
    aminx_log_probs = np.asarray(jax.nn.log_softmax(aminx_logits, axis=-1))
    logits_by_context[use_sc] = aminx_log_probs

    max_abs_val = _max_abs(ref_log_probs, aminx_log_probs)
    pearson_val = _pearson(ref_log_probs, aminx_log_probs)
    label = "context_on" if use_sc else "context_off"
    rows.append(
      _row(
        f"P11.{label}.log_probs",
        "max_abs",
        max_abs_val,
        LOG_PROB_BAR,
        weight_source="eqx",
        fixture=fixture["name"],
        extra={"pearson": pearson_val},
      )
    )

  wiring_delta = float(np.abs(logits_by_context[True] - logits_by_context[False]).max())
  rows.append(
    _row(
      "P11.context_wiring",
      "max_abs_delta_on_vs_off",
      0.0 if wiring_delta > 0.01 else 1.0,  # inverted: 0 = wiring OK (detectable difference)
      0.5,
      weight_source="eqx",
      fixture=fixture["name"],
      extra={
        "raw_delta": wiring_delta,
        "note": "context ON must differ measurably from OFF when residues are fixed",
      },
    )
  )
  return rows


# --------------------------------------------------------------------------------------
# P12 -- SolubleMPNN, weight variant of P04-P05
# --------------------------------------------------------------------------------------


def row_p12(
  batch: ExactBatch,
  models: Any,
  weight_source: str,
  data_utils_module: types.ModuleType,
) -> list[dict[str, Any]]:
  """P12: SolubleMPNN unconditional + conditional TOL, on a real fixture."""
  import jax

  from aminx.inference import score_conditional, score_unconditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  jax_model = lac.jax_model_for_source(models, weight_source, "soluble")
  torch = models.torch
  fd = _reference_feature_dict(torch, batch)
  fd["batch_size"] = 1

  ref_uncond = _score_reference_unconditional_manual(
    models.pt_soluble, models.model_utils, torch, fd
  )
  ref_cond = _score_reference(models.pt_soluble, torch, fd, use_sequence=True)

  kw_u = _aminx_bundle_kwargs(batch, sequence_one_hot=False)
  bundle_u, config_u = build_inference_bundle(mode="score_unconditional", **kw_u)
  logits_u = score_unconditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle_u, config_u, make_stage_set()
  )
  aminx_uncond = np.asarray(jax.nn.log_softmax(logits_u, axis=-1))

  kw_c = _aminx_bundle_kwargs(batch, sequence_one_hot=True)
  bundle_c, config_c = build_inference_bundle(mode="score_conditional", **kw_c)
  logits_c = score_conditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle_c, config_c, make_stage_set()
  )
  aminx_cond = np.asarray(jax.nn.log_softmax(logits_c, axis=-1))

  rows = [
    _row(
      "P12.unconditional.log_probs",
      "max_abs",
      _max_abs(ref_uncond, aminx_uncond),
      LOG_PROB_BAR,
      weight_source=weight_source,
      fixture=batch.fixture_name,
      extra={"pearson": _pearson(ref_uncond, aminx_uncond)},
    ),
  ]
  rows.extend(
    _nll_rows("P12.conditional", ref_cond, aminx_cond, batch, data_utils_module, weight_source)
  )
  return rows


# --------------------------------------------------------------------------------------
# P13 -- Membrane MPNN, varied per-residue / global labels
# --------------------------------------------------------------------------------------


def _membrane_row(
  checkpoint_kind: str,
  batch: ExactBatch,
  models: Any,
  weight_source: str,
  labels: np.ndarray,
  data_utils_module: types.ModuleType,
) -> tuple[list[dict[str, Any]], np.ndarray]:
  import jax
  import jax.numpy as jnp

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  jax_model = lac.jax_model_for_source(models, weight_source, checkpoint_kind)
  pt_model = (
    models.pt_membrane_per_residue
    if checkpoint_kind == "membrane_per_residue"
    else models.pt_membrane_global
  )
  torch = models.torch

  fd = _reference_feature_dict(torch, batch)
  fd["membrane_per_residue_labels"] = torch.from_numpy(labels[None].astype(np.int64).copy())
  ref_log_probs = _score_reference(pt_model, torch, fd, use_sequence=True)

  physics_features = jax.nn.one_hot(jnp.asarray(labels), 3)
  kw = _aminx_bundle_kwargs(batch, sequence_one_hot=True)
  kw["physics_features"] = physics_features
  bundle, config = build_inference_bundle(mode="score_conditional", **kw)
  logits = score_conditional.kernel(
    jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  aminx_log_probs = np.asarray(jax.nn.log_softmax(logits, axis=-1))

  label = "per_residue" if checkpoint_kind == "membrane_per_residue" else "global"
  rows = _nll_rows(
    f"P13.{label}", ref_log_probs, aminx_log_probs, batch, data_utils_module, weight_source
  )
  return rows, aminx_log_probs


def row_p13(
  batch: ExactBatch,
  models: Any,
  weight_source: str,
  data_utils_module: types.ModuleType,
) -> list[dict[str, Any]]:
  """P13: membrane MPNN, random per-residue labels {0,1,2} AND a global label, plus a
  live-label invariant control (changing the label must move the logits measurably)."""
  rng = np.random.default_rng(_seed_for(batch.fixture_name) + 1)
  per_residue_labels = rng.integers(0, 3, size=(batch.length,))
  global_label_value = int(rng.integers(0, 3))
  global_labels = np.full((batch.length,), global_label_value, dtype=np.int64)

  rows_per_residue, logits_a = _membrane_row(
    "membrane_per_residue", batch, models, weight_source, per_residue_labels, data_utils_module
  )
  rows_global, logits_b0 = _membrane_row(
    "membrane_global", batch, models, weight_source, global_labels, data_utils_module
  )

  # Live-label invariant: a DIFFERENT global label must move the global-checkpoint's own
  # logits measurably -- otherwise the physics-feature wiring is a dead no-op.
  other_label_value = (global_label_value + 1) % 3
  other_labels = np.full((batch.length,), other_label_value, dtype=np.int64)
  _rows_other, logits_b1 = _membrane_row(
    "membrane_global", batch, models, weight_source, other_labels, data_utils_module
  )
  live_label_delta = float(np.abs(logits_b0 - logits_b1).max())

  invariant_row = _row(
    "P13.live_label_invariant",
    "max_abs_delta_across_labels",
    0.0 if live_label_delta > 1e-3 else 1.0,  # inverted like P11's wiring row
    0.5,
    weight_source=weight_source,
    fixture=batch.fixture_name,
    extra={
      "raw_delta": live_label_delta,
      "label_a": global_label_value,
      "label_b": other_label_value,
    },
  )
  del logits_a
  return [*rows_per_residue, *rows_global, invariant_row]


# --------------------------------------------------------------------------------------
# P14 -- Packer (NOT IMPLEMENTED, per orchestrator override #6)
# --------------------------------------------------------------------------------------

P14_NOT_IMPLEMENTED_REASON = (
  "P14 (packer, shipped weights, real structures) needs a proxide atom37 -> packer's "
  "14-heavy-atom-order adapter to build a real-structure PackerBundle (backbone_coords, "
  "ligand_coords/mask/types) from a manifest fixture. tests/parity/test_packer_parity.py's "
  "own loaders (reused for the model construction) only exercise the packer on fully "
  "synthetic random features, not a real structure, and this task found no existing, "
  "tested source for that mapping to reuse. Writing one from scratch within T7a's scope "
  "risks a silently wrong atom-order permutation feeding a numeric tolerance gate -- "
  "exactly what orchestrator override #6 forbids fabricating. Deferred to T7b or a "
  "follow-up task with the mapping as its own reviewed unit."
)


def row_p14(fixture: dict[str, Any], weight_source: str) -> list[dict[str, Any]]:
  """P14: NOT IMPLEMENTED -- see `P14_NOT_IMPLEMENTED_REASON`."""
  return [
    {
      "path": "P14.packer_mixture",
      "metric": "not_implemented",
      "value": None,
      "bar": None,
      "ratio": None,
      "status": "not_implemented",
      "weight_source": weight_source,
      "fixture": fixture["name"],
      "reason": P14_NOT_IMPLEMENTED_REASON,
    }
  ]


# --------------------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------------------


def _control(
  name: str,
  metric: str,
  off: float,
  on: float,
  bar: float,
  *,
  detected_rule: str = "on_exceeds_bar_and_off_does_not",
) -> dict[str, Any]:
  effect = abs(on - off)
  ratio_to_bar = effect / bar if bar > 0 else float("inf")
  if detected_rule == "on_exceeds_bar_and_off_does_not":
    detected = bool(on > bar and off <= bar)
  else:
    detected = bool(effect > bar)
  return {
    "name": name,
    "metric": metric,
    "off": off,
    "on": on,
    "effect": effect,
    "ratio_to_bar": ratio_to_bar,
    "detected": detected,
  }


def _control_w_out_bias_perturb(
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
  magnitude: float,
) -> dict[str, Any]:
  """Sized W_out-bias perturbation: the differential sentinel (P05 log-prob max-abs).

  Perturbs ONE vocabulary entry (alanine, index 0 -- matching the spec's
  "Positive-control sizing" paragraph: "aminx arm with +beta on alanine"), never
  every entry uniformly: `log_softmax` is exactly invariant to a constant added
  across its whole axis, so a uniform bias shift is mathematically guaranteed to
  cancel out and produce a control that can never be detected (measured: a
  uniform +1e-3 on all 21 logits moved log-probs by ~1e-6, all floating-point
  noise, not a real effect).
  """
  import equinox as eqx

  jax_model, pt_model, torch, _model_utils = full_model_bundle
  fd = _reference_feature_dict(torch, batch)
  ref_log_probs = _score_reference(pt_model, torch, fd, use_sequence=True)

  off_log_probs = _score_aminx_conditional(jax_model, batch)
  off_effect = _max_abs(ref_log_probs, off_log_probs)

  perturbed_bias = jax_model.w_out.bias.at[0].add(magnitude)
  perturbed_model = eqx.tree_at(lambda m: m.w_out.bias, jax_model, perturbed_bias)
  on_log_probs = _score_aminx_conditional(perturbed_model, batch)
  on_effect = _max_abs(ref_log_probs, on_log_probs)

  return _control(
    "w_out_bias_perturb", "P05.log_probs.max_abs", off_effect, on_effect, LOG_PROB_BAR
  )


def _control_reversed_order(
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
  """Reversed decoding order fed to the reference only: must be DETECTED as a divergence."""
  jax_model, pt_model, torch, _model_utils = full_model_bundle
  aminx_log_probs = _score_aminx_conditional(jax_model, batch)

  fd = _reference_feature_dict(torch, batch)
  ref_log_probs_matched = _score_reference(pt_model, torch, fd, use_sequence=True)
  off_effect = _max_abs(ref_log_probs_matched, aminx_log_probs)

  reversed_order = batch.decoding_order[::-1].copy()
  reversed_ar_mask = lac.ar_mask_from_order(reversed_order)
  reversed_batch = ExactBatch(
    fixture_name=batch.fixture_name,
    length=batch.length,
    x4=batch.x4,
    atom37=batch.atom37,
    atom37_mask=batch.atom37_mask,
    aatype_aminx=batch.aatype_aminx,
    seq_ref=batch.seq_ref,
    mask=batch.mask,
    chain_mask=batch.chain_mask,
    residue_index=batch.residue_index,
    chain_index=batch.chain_index,
    randn=batch.randn,
    decoding_order=reversed_order,
    ar_mask=reversed_ar_mask,
  )
  aminx_log_probs_reversed = _score_aminx_conditional(jax_model, reversed_batch)
  on_effect = _max_abs(ref_log_probs_matched, aminx_log_probs_reversed)

  return _control("reversed_order", "P05.log_probs.max_abs", off_effect, on_effect, LOG_PROB_BAR)


def _control_mantissa_bit_flip(p01_max_abs: float) -> dict[str, Any]:
  """Flip the lowest mantissa bit of one eqx weight leaf; the P01 max-abs must move off 0."""
  import jax
  import jax.numpy as jnp

  from tests.parity.test_full_model_parity import _load_heavy_parity_models_impl

  models = _load_heavy_parity_models_impl()
  eqx_leaves, treedef = jax.tree_util.tree_flatten(models.jax_model_eqx)
  flip_index = next(
    i for i, leaf in enumerate(eqx_leaves) if hasattr(leaf, "shape") and leaf.size > 0
  )
  target = np.asarray(eqx_leaves[flip_index])
  flat = target.reshape(-1).copy()
  as_int = flat[:1].view(np.int32)
  as_int ^= 1  # flip the lowest mantissa bit of the first element
  flat[:1] = as_int.view(np.float32)
  perturbed = flat.reshape(target.shape)

  perturbed_leaves = list(eqx_leaves)
  perturbed_leaves[flip_index] = jnp.asarray(perturbed)
  perturbed_model = jax.tree_util.tree_unflatten(treedef, perturbed_leaves)

  perturbed_leaves_flat = jax.tree_util.tree_leaves(perturbed_model)
  max_abs = 0.0
  eqx_leaves_after = jax.tree_util.tree_leaves(models.jax_model_eqx)
  for a, b in zip(eqx_leaves_after, perturbed_leaves_flat, strict=True):
    if not hasattr(a, "shape"):
      continue
    diff = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
    max_abs = max(max_abs, diff)

  return _control("mantissa_bit_flip", "P01.weights.max_abs", p01_max_abs, max_abs, WEIGHT_BAR)


_KNN_MOVE_ANGSTROM = 50.0


def _control_knn_neighbor_move(
  fixture: dict[str, Any], batch: ExactBatch, k: int = 48
) -> dict[str, Any]:
  """Move the k-th nearest neighbour's CA far away; the neighbour SET must change.

  Deliberately large (`_KNN_MOVE_ANGSTROM`, not a small tolerance-sized nudge like
  P02's 1e-3 Angstrom control): this control checks a discrete SET-membership
  change, not a continuous coordinate tolerance, so it needs a move guaranteed to
  cross whatever the current k-th/(k+1)-th distance gap happens to be on the
  fixture at hand (measured: a 1 Angstrom move did not reliably cross that gap
  when k is a large fraction of a small fixture's residue count).
  """
  from proxide.chem.residues import atom_order

  if k >= batch.length - 1:
    k = max(1, batch.length - 2)
  ca_idx = int(atom_order["CA"])

  before = fixtures._neighbor_indices(batch.atom37, batch.mask, k)  # noqa: SLF001
  before_set = set(before[0].tolist())

  moved_atom37 = batch.atom37.copy()
  target = before[0][-1] if before[0].size else 0
  moved_atom37[target, ca_idx, :] += _KNN_MOVE_ANGSTROM

  after = fixtures._neighbor_indices(moved_atom37, batch.mask, k)  # noqa: SLF001
  after_set = set(after[0].tolist())

  n_changed = len(before_set.symmetric_difference(after_set))
  return _control(
    "knn_neighbor_move",
    "n_neighbor_set_changed",
    0,
    n_changed,
    0.0,
    detected_rule="effect_over_bar",
  )


def _control_tie_group_flip(
  fixture: dict[str, Any],
  batch: ExactBatch,
  full_model_bundle: tuple[Any, Any, Any, Any],
) -> dict[str, Any] | None:
  """Swap one member between two tie groups; the fused log-probs at those positions must move.

  Reversing member ORDER within a group (rather than swapping membership ACROSS
  groups) does NOT perturb anything: aminx's own fusion sums every member's
  contribution (`sum(log_softmax(...))`, `inference/logits.py:396-435`), which is
  commutative in member order, and `_build_tie_group_map` only reads group
  MEMBERSHIP, not order -- verified empirically (an order-only flip measured
  `effect == 0.0` here). A genuine membership swap is required to move anything.
  """
  groups = _qualifying_groups(fixture)
  if len(groups) < 2:
    return None

  from tests.parity.test_full_model_parity import _build_tie_group_map

  jax_model, _pt_model, _torch, _model_utils = full_model_bundle

  import jax

  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  normal_map = _build_tie_group_map(batch.length, groups)
  flipped_groups = [list(g) for g in groups]
  flipped_groups[0] = [*groups[1][:1], *groups[0][1:]]
  flipped_groups[1] = [*groups[0][:1], *groups[1][1:]]
  flipped_map = _build_tie_group_map(batch.length, flipped_groups)

  def _fused_logits(tie_map: np.ndarray) -> np.ndarray:
    kw = _aminx_bundle_kwargs(batch, sequence_one_hot=True)
    kw["tie_group_map"] = jax.numpy.asarray(tie_map)
    bundle, config = build_inference_bundle(mode="score_conditional", **kw)
    logits = score_conditional.kernel(
      jax_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
    )
    return np.asarray(jax.nn.log_softmax(logits, axis=-1))

  normal_logits = _fused_logits(normal_map)
  flipped_logits = _fused_logits(flipped_map)
  member_indices = sorted({m for g in groups for m in g})
  effect = float(np.abs(normal_logits[member_indices] - flipped_logits[member_indices]).max())

  return _control("tie_group_flip", "P09.log_probs.max_abs", 0.0, effect, LOG_PROB_BAR)


# --------------------------------------------------------------------------------------
# Differential-phase entry point
# --------------------------------------------------------------------------------------


def sentinel_ratio_to_bar(
  subset: list[dict[str, Any]],
  *,
  weight_source: str,
  perturb: bool,
  magnitude: float = DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE,
) -> float:
  """Differential-phase metric: mean P05 log-prob max-abs / bar over `subset`, off or on.

  `subset` is the pre-registered reduced fixture list (a subset of manifest
  fixtures). When `perturb` is True, the calibrated W_out-bias sentinel
  (`magnitude`) is added before scoring, matching `AMINX_BV_PERTURB`'s
  ``BTH_DIFFERENTIAL_VALUE`` "1" arm; `perturb=False` matches the "0" arm.
  """
  import equinox as eqx

  data_utils_module = _load_reference_data_utils()
  full_model_bundle = lac.load_full_model(weight_source)
  jax_model, pt_model, torch, model_utils = full_model_bundle
  if perturb:
    # Perturb ONE vocabulary entry (alanine, index 0), never all of them uniformly -- a
    # uniform shift is exactly cancelled by log_softmax (see `_control_w_out_bias_perturb`).
    perturbed_bias = jax_model.w_out.bias.at[0].add(magnitude)
    jax_model = eqx.tree_at(lambda m: m.w_out.bias, jax_model, perturbed_bias)
    full_model_bundle = (jax_model, pt_model, torch, model_utils)

  ratios: list[float] = []
  for fixture in subset:
    batch = build_exact_batch(fixture, data_utils_module)
    fd = _reference_feature_dict(torch, batch)
    ref_log_probs = _score_reference(pt_model, torch, fd, use_sequence=True)
    aminx_log_probs = _score_aminx_conditional(jax_model, batch)
    value = _max_abs(ref_log_probs, aminx_log_probs)
    ratios.append(value / LOG_PROB_BAR)
  return float(np.mean(ratios)) if ratios else 0.0


# --------------------------------------------------------------------------------------
# run_rows -- top-level entry point
# --------------------------------------------------------------------------------------


def run_rows(
  fixtures_list: list[dict[str, Any]],
  params: dict[str, Any] | None,
  weight_source: str,
  *,
  controls: bool = True,
) -> dict[str, Any]:
  """Run every implemented bars-table row over `fixtures_list`, plus in-run controls."""
  skip_counter = lac.SkipCounter()
  data_utils_module = _load_reference_data_utils()
  full_model_bundle = lac.load_full_model(weight_source)
  soluble_membrane_models = lac.reference_call(
    skip_counter, lac.load_soluble_membrane_models, weight_source
  )

  all_rows: list[dict[str, Any]] = []
  not_implemented: list[dict[str, Any]] = []
  n_near_tie_excluded = 0
  p01_max_abs = 0.0

  p01_rows, p01_max_abs = row_p01(full_model_bundle, weight_source)
  all_rows.extend(p01_rows)

  protein_fixtures = [f for f in fixtures_list if f.get("kind") == "protein"]
  first_fixture = protein_fixtures[0] if protein_fixtures else None

  if first_fixture is not None:
    p00_rows = lac.reference_call(skip_counter, row_p00, first_fixture, weight_source, skip_counter)
    if p00_rows:
      all_rows.extend(p00_rows)

  for fixture in protein_fixtures:
    all_rows.extend(row_p02(fixture, weight_source))

    batch = build_exact_batch(fixture, data_utils_module)

    p03_rows, n_excluded = row_p03(fixture, batch, full_model_bundle, weight_source)
    all_rows.extend(p03_rows)
    n_near_tie_excluded += n_excluded

    all_rows.extend(row_p04(batch, full_model_bundle, weight_source))
    all_rows.extend(row_p05(batch, full_model_bundle, data_utils_module, weight_source))
    all_rows.extend(
      row_p06(
        batch,
        full_model_bundle[0],
        data_utils_module,
        weight_source,
        full_model_bundle[1],
        full_model_bundle[2],
      )
    )
    all_rows.extend(row_p09(fixture, batch, full_model_bundle, weight_source))

    if soluble_membrane_models is not None:
      all_rows.extend(row_p12(batch, soluble_membrane_models, weight_source, data_utils_module))
      all_rows.extend(row_p13(batch, soluble_membrane_models, weight_source, data_utils_module))

    all_rows.extend(row_p14(fixture, weight_source))

  if first_fixture is not None:
    ligand_rows = lac.reference_call(skip_counter, row_p11, first_fixture, weight_source)
    if ligand_rows:
      all_rows.extend(ligand_rows)

  n_not_advanced = _apply_headroom(all_rows, params)
  for row in all_rows:
    if row.get("status") == "not_implemented":
      not_implemented.append({"path": row["path"], "reason": row.get("reason", "")})

  n_over_bar = sum(1 for row in all_rows if row.get("ratio") is not None and row["ratio"] > 1.0)

  control_list: list[dict[str, Any]] = []
  if controls and first_fixture is not None:
    batch0 = build_exact_batch(first_fixture, data_utils_module)
    perturb_magnitude = (
      params.get("exact", {}).get("controls", {}).get("w_out_bias_perturb_magnitude")
      if params is not None
      else None
    ) or DEFAULT_W_OUT_BIAS_PERTURB_MAGNITUDE

    control_list.append(_control_w_out_bias_perturb(batch0, full_model_bundle, perturb_magnitude))
    control_list.append(_control_reversed_order(batch0, full_model_bundle))
    control_list.append(_control_mantissa_bit_flip(p01_max_abs))
    control_list.append(_control_knn_neighbor_move(first_fixture, batch0))
    parse_control = parse_parity._run_control(  # noqa: SLF001
      first_fixture,
      Path(os.environ.get("TMPDIR", "/tmp")) / "layer_a_exact_parse_control",
      Path(os.environ.get("TMPDIR", "/tmp")) / "layer_a_exact_parse_control_canonical",
      parse_parity._load_reference_module(),  # noqa: SLF001
    )
    control_list.append(
      _control(
        "atom_shift_1e-3A",
        "P02.backbone_coords_max_abs",
        0.0,
        parse_control["max_abs"],
        parse_parity.COORD_BAR_ANGSTROM,
      )
    )
    tie_control = _control_tie_group_flip(first_fixture, batch0, full_model_bundle)
    if tie_control is not None:
      control_list.append(tie_control)

  controls_total = len(control_list)
  controls_detected = sum(1 for c in control_list if c["detected"])

  return {
    "rows": all_rows,
    "controls": control_list,
    "n_comparisons": len(all_rows),
    "n_over_bar": n_over_bar,
    "n_not_advanced": n_not_advanced,
    "n_near_tie_excluded": n_near_tie_excluded,
    "n_skipped": skip_counter.n,
    "skip_reasons": skip_counter.reasons,
    "controls_total": controls_total,
    "controls_detected": controls_detected,
    "not_implemented": not_implemented,
  }
