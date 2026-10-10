"""ProtonPottsMPNN family driver: ``score:energy``, ``score:ddg``, ``score:selectivity`` and ``sample``.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §42-§47. Registration runs when this module is
imported (``aminx.families.protonpotts_mpnn`` imports it).

HOW A SEQUENCE IS GIVEN. Upstream has no text format: a sequence is an integer token tensor ``S`` (v6 indices),
read off the structure's pre-assigned protonation labels, and a design variant is a set of pinned centres
``{res_id, protonation_type}``. This driver follows that model, not a one-letter string (which could not
carry ``HIS-P``):

* the REFERENCE is the structure's own ``S``, with ``protonation_labels_json`` applied (none means every
  residue is a standard residue);
* a VARIANT is pins ``{"A:42": "HIS"}`` applied on top of the reference, or a full per-residue token list;
* ``score:energy`` returns the reference energy then one per variant (row 0 is the reference);
  ``score:ddg`` returns ``E(variant) - E(reference)`` and requires variants (no DMS enumeration: protonation
  tokens are not mutation targets, spec §35.1);
* ``score:selectivity`` takes the same candidates and reports, per candidate, the selectivity gap of the
  centres in ``ProtonPottsOptions.explicit_centers``, evaluated with those centres forced to their protonated
  token;
* ``sample`` is pH design (ports ``ph_design.design_structure``): one structure per call, the number of
  designs set by ``ProtonPottsOptions.samples_per_site``, not by ``spec.num_samples``.

Output sequences are token INDICES into the vocabulary stamped on the result (``vocabulary`` attribute), never
one-letter strings: upstream's ``decode_sequences`` collapses ``HIS-P`` to ``H``, which loses the state.

Not handled: the four generic-MPNN fallback purposes, which REFUSE rather than load a 21-wide model (spec §35.1).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec, BatchPlanner

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn.energy import protonpotts_energies, protonpotts_table
from aminx.families.protonpotts_mpnn.features import (
  PdbFeatureError,
  featurize_pdb,
  kept_residues,
  loaded_residues,
)
from aminx.families.protonpotts_mpnn.ph_config import DEFAULT_DEP_MAP, HOST_METHODS, config_from_options
from aminx.families.protonpotts_mpnn.ph_design import (
  PHDesign,
  design_structure,
  force_pins,
  selectivity_gaps,
)
from aminx.families.protonpotts_mpnn.ph_plan import Pin, pins_at_positions
from aminx.families.protonpotts_mpnn.sequence import encode_sequence
from aminx.families.protonpotts_mpnn.vocab import (
  PROTONPOTTS_V6,
  THREE_TO_ONE,
  V6_PROTONATION_TOKENS,
  canonical_letter,
  token_index,
)
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, FamilyStages, SinkArraySpec
from aminx.host.prep import _resolve_local_checkpoint_from_registry
from aminx.run.options import ProtonPottsOptions

log = logging.getLogger(__name__)

_HANDLED = frozenset({"score:energy", "score:ddg", "score:selectivity", "sample"})
_REFERENCE_PURPOSES = frozenset({"score:energy", "score:selectivity"})
_KEY = re.compile(r"^(?P<chain>[^:\s]+):(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$")
_VOCABULARY = ",".join(PROTONPOTTS_V6.symbols)


class _Graph(NamedTuple):
  """Device inputs for one featurized structure."""

  coords: np.ndarray
  present: np.ndarray
  residue_idx: np.ndarray
  chain_index: np.ndarray
  pad_valid: np.ndarray
  sequences: np.ndarray


class _Prepared(NamedTuple):
  graph: _Graph
  include_reference_in_output: bool
  kept: tuple[tuple[str, int, str, str], ...] = ()


class ProtonPottsInputError(ValueError):
  """The structure cannot be scored (it is skipped, as the shipped Potts driver skips its own)."""


_JIT_ENERGIES = eqx.filter_jit(protonpotts_energies)
_JIT_TABLE = eqx.filter_jit(protonpotts_table)


def _graph_args(graph: _Graph) -> tuple[jax.Array, ...]:
  return (
    jnp.asarray(graph.coords),
    jnp.asarray(graph.present),
    jnp.asarray(graph.residue_idx),
    jnp.asarray(graph.chain_index),
    jnp.asarray(graph.pad_valid),
  )


class _ScoreStages:
  """Host loop over a batch: reference and variants of each structure are one broadcast."""

  def __init__(self, model: PottsMPNN) -> None:
    self._model = model

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del chunk_start, chunk_count, scalar_dropout, dropout_key
    prepared = cast("tuple[_Prepared, ...]", batch.arrays["prepared"])
    energies: list[jax.Array] = []
    ids: list[jax.Array] = []
    tokens: list[jax.Array] = []
    expt: list[jax.Array] = []
    reference_mode = prepared[0].include_reference_in_output
    for item in prepared:
      graph = item.graph
      block = _JIT_ENERGIES(self._model, *_graph_args(graph), jnp.asarray(graph.sequences))
      rows = jnp.asarray(graph.sequences, dtype=jnp.int32)
      if reference_mode:
        values, shown = block.astype(jnp.float32), rows
      else:
        values, shown = (block[1:] - block[0]).astype(jnp.float32), rows[1:]
      energies.append(values)
      ids.append(jnp.arange(values.shape[0], dtype=jnp.int32))
      tokens.append(shown)
      expt.append(jnp.full(values.shape, jnp.nan, dtype=jnp.float32))
    if reference_mode:
      return {
        "energy": jnp.stack(energies),
        "candidate_ids": jnp.stack(ids),
        "candidate_tokens": jnp.stack(tokens),
      }
    return {
      "ddg": jnp.stack(energies),
      "mutant_ids": jnp.stack(ids),
      "ddg_expt": jnp.stack(expt),
      "mutant_tokens": jnp.stack(tokens),
    }


class _SelectivityStages:
  """Host loop over a batch for ``score:selectivity``: each candidate with the centres forced.

  The merged table is computed here, where the model is. The pins were resolved in ``batches``.
  """

  def __init__(self, model: PottsMPNN) -> None:
    self._model = model

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del chunk_start, chunk_count, scalar_dropout, dropout_key
    prepared = cast("tuple[_Prepared, ...]", batch.arrays["prepared"])
    pins_by_structure = cast("tuple[tuple[Pin, ...], ...]", batch.arrays["pins"])
    totals: list[np.ndarray] = []
    per_centre: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    tokens: list[np.ndarray] = []
    for item, pins in zip(prepared, pins_by_structure, strict=True):
      table, e_idx = _JIT_TABLE(self._model, *_graph_args(item.graph))
      forced = np.stack([force_pins(row, pins) for row in item.graph.sequences])
      gaps = [selectivity_gaps(table, e_idx, row, pins) for row in forced]
      totals.append(np.asarray([total for total, _ in gaps], dtype=np.float32))
      per_centre.append(
        np.asarray([per for _, per in gaps], dtype=np.float32).reshape(len(gaps), -1),
      )
      ids.append(np.arange(forced.shape[0], dtype=np.int32))
      tokens.append(forced)
    return {
      "selectivity": jnp.asarray(np.stack(totals)),
      "selectivity_per_centre": jnp.asarray(np.stack(per_centre)),
      "candidate_ids": jnp.asarray(np.stack(ids)),
      "candidate_tokens": jnp.asarray(np.stack(tokens).astype(np.int32)),
    }


def _design_arrays(designs: Sequence[PHDesign]) -> dict[str, np.ndarray]:
  """Per-design arrays of one structure's designs (all from one placement plan, so one pin layout)."""
  pins = designs[0].pins
  n_centre = max(1, len(pins))
  n_design = len(designs)
  centre_positions = np.full((n_design, n_centre), -1, dtype=np.int32)
  centre_types = np.full((n_design, n_centre), -1, dtype=np.int32)
  for c, pin in enumerate(pins):
    centre_positions[:, c] = pin.position
    centre_types[:, c] = pin.prot_idx
  return {
    "sequence": np.stack([d.sequence for d in designs]).astype(np.int32),
    "final_potts_energy": np.asarray([d.final_potts_energy for d in designs], dtype=np.float32),
    "selective_energy": np.asarray([d.selective_energy for d in designs], dtype=np.float32),
    "design_sample": np.asarray([d.sample for d in designs], dtype=np.int32),
    "center_positions": centre_positions,
    "center_types": centre_types,
  }


class _SampleStages:
  """Host body for ``sample``: packs the designs of each structure.

  The designs are computed in ``ProtonPottsDriver.batches``, which must know whether a structure has a plan
  before the host sees the batch (a structure with no plan is a skipped input, not zero-length arrays).
  """

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del chunk_start, chunk_count, scalar_dropout, dropout_key
    designs_by_structure = cast("tuple[list[PHDesign], ...]", batch.arrays["designs"])
    per_structure = [_design_arrays(designs) for designs in designs_by_structure]
    return {
      name: jnp.asarray(np.stack([arrays[name] for arrays in per_structure]))
      for name in per_structure[0]
    }


class ProtonPottsDriver:
  """Score and design ProtonPottsMPNN through the family-driver seam."""

  name = "protonpottsmpnn"
  options_type = ProtonPottsOptions
  mpnn_fallback_purposes = frozenset({"jacobian", "inspect", "score:nll", "score:logits"})
  root_alphabet = _VOCABULARY

  def __init__(self) -> None:
    # Single-entry cache: ``run_family_driver`` loads the model, and ``batches`` (pH design) needs it
    # again to build the merged table. Keyed by path, mtime and size, so a rewritten artifact reloads.
    self._cache: tuple[tuple[str, int, int], PottsMPNN] | None = None

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """True for ``score:energy``, ``score:ddg``, ``score:selectivity`` and ``sample``."""
    del spec
    return purpose in _HANDLED

  def load(self, spec: Any) -> PottsMPNN:  # noqa: ANN401
    """Load a converted ProtonPottsMPNN (``convert_protonpotts_checkpoint.py --out``).

    ``model_local_path`` wins; otherwise the checkpoint registry entry is used and must carry ``sha256``.
    A manifest next to the artifact, when present, must name the v6 alphabet.
    """
    local = getattr(spec, "model_local_path", None)
    if local is None:
      artifact = _resolve_local_checkpoint_from_registry(spec)
      if artifact is None:
        msg = (
          "ProtonPottsDriver requires model_local_path or a checkpoint registry entry with sha256"
        )
        raise ValueError(msg)
      path = Path(artifact)
    else:
      path = Path(str(local))
    if path.suffix.lower() != ".eqx":
      msg = (
        f"{path}: expected a converted .eqx artifact; convert the checkpoint with "
        "scripts/parity/convert_protonpotts_checkpoint.py"
      )
      raise ValueError(msg)
    manifest = path.with_suffix(".json")
    if manifest.exists():
      alphabet = json.loads(manifest.read_text(encoding="utf-8")).get("alphabet")
      if alphabet != PROTONPOTTS_V6.name:
        msg = f"{manifest}: alphabet {alphabet!r}, expected {PROTONPOTTS_V6.name!r}"
        raise ValueError(msg)
    stat = path.stat()
    stamp = (str(path), stat.st_mtime_ns, stat.st_size)
    if self._cache is not None and self._cache[0] == stamp:
      return self._cache[1]
    skeleton = PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)
    model = cast("PottsMPNN", eqx.tree_deserialise_leaves(path, skeleton))
    self._cache = (stamp, model)
    return model

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """Refuse: the generic MPNN path reads a 21-token alphabet and would misread a 30-token model."""
    del model
    msg = (
      f"{self.name}: fallback purposes ({', '.join(sorted(self.mpnn_fallback_purposes))}) assume the "
      f"21-token ProteinMPNN alphabet; this model uses {PROTONPOTTS_V6.name!r} "
      f"(size {PROTONPOTTS_V6.size})"
    )
    raise ValueError(msg)

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """One structure per batch, featurized by the v6 host port."""
    purpose = _purpose(spec)
    options = _options(spec)
    if purpose == "sample":
      _check_sample_count(spec)
    if purpose == "score:selectivity" and not options.explicit_centers:
      msg = (
        "score:selectivity needs ProtonPottsOptions.explicit_centers: a non-empty list of "
        "(residue number, protonation type) pairs naming the centres to score"
      )
      raise ValueError(msg)
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(_inputs(spec)):
      if not isinstance(item, (str, Path)):
        pending.append((index, "protonpottsmpnn_requires_pdb"))
        continue
      path = Path(item)
      try:
        prepared = _prepare(path, options, purpose, _spec_sequences(spec))
      except PdbFeatureError as exc:
        pending.append((index, str(exc)))
        continue
      length = int(prepared.graph.coords.shape[0])
      if purpose == "sample":
        outcome = self._design_one(spec, options, path, prepared, index)
        if isinstance(outcome, str):
          pending.append((index, outcome))
          continue
        arrays: dict[str, Any] = {"prepared": (prepared,), "designs": (outcome,)}
      elif purpose == "score:selectivity":
        arrays = {"prepared": (prepared,), "pins": (_selectivity_pins(prepared, options, path),)}
      else:
        arrays = {"prepared": (prepared,)}
      yield FamilyBatch(
        input_indices=(index,),
        arrays=arrays,
        skipped=tuple(pending),
        lengths=(length,),
      )
      pending = []
    if pending:
      yield FamilyBatch(
        input_indices=(),
        arrays={"prepared": ()},
        skipped=tuple(pending),
        lengths=(),
      )

  def _design_one(
    self,
    spec: Any,  # noqa: ANN401
    options: ProtonPottsOptions,
    path: Path,
    prepared: _Prepared,
    index: int,
  ) -> list[PHDesign] | str:
    """pH designs of one structure, or the skip reason when its placement plan is empty.

    ``design_structure`` returns ``[]`` where upstream returns no plan: no designable binder positions
    after the centres, or fewer free binder positions than centre types.
    """
    binder = _require_binder_chain(options.binder_chain, prepared.kept, path)
    config = config_from_options(options)
    config.validate_for_design()
    binder_mask = np.asarray([chain == binder for chain, *_rest in prepared.kept], dtype=bool)
    res_id = np.asarray([number for _chain, number, *_rest in prepared.kept], dtype=np.int32)
    model = self.load(spec)
    table, e_idx = _JIT_TABLE(model, *_graph_args(prepared.graph))
    native = np.asarray(prepared.graph.sequences[0], dtype=np.int32)
    key = jax.random.fold_in(jax.random.PRNGKey(int(spec.random_seed)), index)
    if config.method == "mpnn_sample":
      # Lazy: only this method needs the decoder (debt #2617; graded by the protonpotts_sample wave).
      from aminx.families.protonpotts_mpnn.ph_sample import mpnn_sample_designs  # noqa: PLC0415

      designs = mpnn_sample_designs(
        model, _graph_args(prepared.graph), table, e_idx, native, binder_mask, config, key=key,
      )
    elif config.method in HOST_METHODS:
      # Lazy: the host-loop methods (specs §56; graded by the protonpotts_ph_methods wave).
      from aminx.families.protonpotts_mpnn.ph_methods import design_methods  # noqa: PLC0415

      designs = design_methods(
        model, _graph_args(prepared.graph), table, e_idx, native, binder_mask, res_id, config, key=key,
      )
    else:
      designs = design_structure(table, e_idx, native, binder_mask, res_id, config, key=key)
    if not designs:
      return (
        f"no pH design plan for {path.name}: no placement plan with designable binder positions "
        f"(binder chain {binder!r}, center_types={list(config.center_types)}, "
        f"explicit_centers={list(config.explicit_centers)})"
      )
    return designs

  def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
    """Structure and candidate axes."""
    del spec
    if purpose == "sample":
      designs = cast("tuple[list[PHDesign], ...]", batch.arrays.get("designs", ()))
      n_rows = max((len(d) for d in designs), default=1)
    else:
      prepared = cast("tuple[_Prepared, ...]", batch.arrays.get("prepared", ()))
      n_rows = int(prepared[0].graph.sequences.shape[0]) if prepared else 1
    specs = [
      AxisSpec(
        name="structures",
        cardinality=max(len(batch.input_indices), 1),
        default_batch_size=1,
        heterogeneous=True,
        bucket_boundaries=bucket_ladder(),
      ),
      AxisSpec(name="mutants", cardinality=max(n_rows, 1), default_batch_size=max(n_rows, 1)),
    ]
    BatchPlanner().plan(specs)
    return specs

  def stages(self, spec: Any, purpose: str, model: eqx.Module) -> FamilyStages:  # noqa: ANN401
    """Per-batch energy, ddG, selectivity or design-packing stage."""
    del spec
    if purpose == "sample":
      return _SampleStages()
    if purpose == "score:selectivity":
      return _SelectivityStages(cast("PottsMPNN", model))
    return _ScoreStages(cast("PottsMPNN", model))

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Arrays one chunk returns. ``*_tokens`` and ``center_types`` are indices into ``vocabulary``.

    The scoring id arrays carry ``variant_names`` (a JSON list indexed by id) when the spec names its variants: id ``k`` is
    ``variant_names[k]``. Energy and selectivity rows start with the reference, ddg rows do not (debt #2618, item 5).
    """
    vocab = {"vocabulary": _VOCABULARY}
    names = _id_attrs(spec, purpose)
    if purpose == "score:energy":
      return {
        "energy": SinkArraySpec(dims=("N_cand",), dtype="float32", attrs={}),
        "candidate_ids": SinkArraySpec(dims=("N_cand",), dtype="int32", attrs=names),
        "candidate_tokens": SinkArraySpec(dims=("N_cand", "L_total"), dtype="int32", attrs=vocab),
      }
    if purpose == "score:ddg":
      return {
        "ddg": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
        "mutant_ids": SinkArraySpec(dims=("N_mut",), dtype="int32", attrs=names),
        "ddg_expt": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
        "mutant_tokens": SinkArraySpec(dims=("N_mut", "L_total"), dtype="int32", attrs=vocab),
      }
    if purpose == "score:selectivity":
      return {
        "selectivity": SinkArraySpec(dims=("N_cand",), dtype="float32", attrs={}),
        "selectivity_per_centre": SinkArraySpec(
          dims=("N_cand", "N_centre"),
          dtype="float32",
          attrs={},
        ),
        "candidate_ids": SinkArraySpec(dims=("N_cand",), dtype="int32", attrs=names),
        "candidate_tokens": SinkArraySpec(dims=("N_cand", "L_total"), dtype="int32", attrs=vocab),
      }
    if purpose == "sample":
      return {
        "sequence": SinkArraySpec(dims=("N_design", "L_total"), dtype="int32", attrs=vocab),
        "final_potts_energy": SinkArraySpec(dims=("N_design",), dtype="float32", attrs={}),
        "selective_energy": SinkArraySpec(dims=("N_design",), dtype="float32", attrs={}),
        "design_sample": SinkArraySpec(dims=("N_design",), dtype="int32", attrs={}),
        "center_positions": SinkArraySpec(dims=("N_design", "N_centre"), dtype="int32", attrs={}),
        "center_types": SinkArraySpec(dims=("N_design", "N_centre"), dtype="int32", attrs=vocab),
      }
    msg = f"protonpottsmpnn does not support {purpose} in v1"
    raise ValueError(msg)


# --- purposes and sample/selectivity inputs ----------------------------------------------------


def _purpose(spec: Any) -> str:  # noqa: ANN401
  """``score:<output_kind>`` for a scoring spec; ``sample`` for a sampling spec (which has no output kind)."""
  kind = getattr(spec, "output_kind", None)
  if kind is None:
    return "sample"
  return f"score:{kind}"


def _check_sample_count(spec: Any) -> None:  # noqa: ANN401
  """``sample`` designs are set by ``samples_per_site``; ``num_samples`` > 1 would repeat each design."""
  count = int(getattr(spec, "num_samples", 1))
  if count != 1:
    msg = (
      f"protonpottsmpnn sample runs one pH design call per structure: set num_samples=1 (got {count}). "
      "The number of designs is ProtonPottsOptions.samples_per_site per placement plan."
    )
    raise ValueError(msg)


def _require_binder_chain(
  binder: str | None,
  kept: Sequence[tuple[str, int, str, str]],
  path: Path,
) -> str:
  """The binder chain, which must be set and present among the kept residues."""
  chains = sorted({chain for chain, *_rest in kept})
  if not binder or binder not in chains:
    msg = (
      f"{path.name}: ProtonPottsOptions.binder_chain is {binder!r}; pH design needs a binder chain "
      f"present in the structure. Chains present: {chains}"
    )
    raise ValueError(msg)
  return binder


def _selectivity_pins(
  prepared: _Prepared,
  options: ProtonPottsOptions,
  path: Path,
) -> tuple[Pin, ...]:
  """Pins of the explicit centres, resolved against the kept residues (restricted to the binder chain if set)."""
  binder = options.binder_chain
  if binder is not None:
    _require_binder_chain(binder, prepared.kept, path)
  placements: list[tuple[int, str]] = []
  for residue, ptype in options.explicit_centers:
    hits = [
      j
      for j, (chain, number, _icode, _name) in enumerate(prepared.kept)
      if number == residue and (binder is None or chain == binder)
    ]
    where = f" on chain {binder!r}" if binder is not None else ""
    if not hits:
      msg = f"{path.name}: explicit centre residue {residue} ({ptype}) is not a kept residue{where}"
      raise ValueError(msg)
    if len(hits) > 1:
      chains = sorted({prepared.kept[j][0] for j in hits})
      msg = (
        f"{path.name}: residue number {residue} occurs on chains {chains}; "
        "set ProtonPottsOptions.binder_chain to choose one"
      )
      raise ValueError(msg)
    placements.append((hits[0], ptype))
  dep_map = dict(options.dep_map) or dict(DEFAULT_DEP_MAP)
  res_id = np.asarray([number for _chain, number, _icode, _name in prepared.kept], dtype=np.int32)
  return pins_at_positions(placements, res_id, dep_map)


# --- input reading -----------------------------------------------------------------------------


def _options(spec: Any) -> ProtonPottsOptions:  # noqa: ANN401
  options = getattr(spec, "protonpotts", None)
  return ProtonPottsOptions() if options is None else cast("ProtonPottsOptions", options)


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = spec.inputs
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _spec_sequences(spec: Any) -> tuple[str, ...]:  # noqa: ANN401
  return tuple(str(seq) for seq in (getattr(spec, "sequences_to_score", ()) or ()))


def _id_attrs(spec: Any, purpose: str) -> dict[str, str]:  # noqa: ANN401
  """``{"variant_names": <JSON list>}`` for a scoring spec that names variants, else no attrs.

  Ids count the rows in the order ``_variant_rows`` builds them: the variants_json entries in file order, then each
  ``sequences_to_score`` entry as ``sequences_to_score[i]``. Energy and selectivity prepend ``reference`` (id 0).
  """
  if spec is None or purpose not in {"score:energy", "score:ddg", "score:selectivity"}:
    return {}
  names = [*_read_json(_options(spec).variants_json, "variants_json"), *(f"sequences_to_score[{i}]" for i, _ in enumerate(_spec_sequences(spec)))]
  if purpose in _REFERENCE_PURPOSES:
    names = ["reference", *names]
  return {"variant_names": json.dumps(names)} if names else {}


def _read_json(path: str | None, what: str) -> dict[str, Any]:
  if not path:
    return {}
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    msg = f"{what} {path}: expected a JSON object, got {type(payload).__name__}"
    raise ValueError(msg)  # noqa: TRY004 -- a user input error, reported as ValueError
  return payload


def _residue_lookup(
  residues: Sequence[tuple[str, int, str, str]],
) -> dict[tuple[str, int, str], int]:
  return {
    (chain, number, icode.strip()): i for i, (chain, number, icode, _name) in enumerate(residues)
  }


def _resolve_key(
  key: str,
  lookup: Mapping[tuple[str, int, str], int],
  what: str,
) -> int:
  match = _KEY.match(key)
  if match is None:
    msg = f"{what}: {key!r} is not '<chain>:<number>' or '<chain>:<number><insertion code>'"
    raise ValueError(msg)
  found = lookup.get((match["chain"], int(match["number"]), match["icode"]))
  if found is None:
    msg = f"{what}: residue {key!r} is not in the structure's loaded residues"
    raise ValueError(msg)
  return found


def _reference_labels(
  loaded: Sequence[tuple[str, int, str, str]],
  options: ProtonPottsOptions,
) -> list[str] | None:
  """Per-loaded-residue label list (``""`` for none), or ``None`` when no labels were given."""
  given = _read_json(options.protonation_labels_json, "protonation_labels_json")
  if not given:
    return None
  lookup = _residue_lookup(loaded)
  labels = [""] * len(loaded)
  for key, token in given.items():
    labels[_resolve_key(key, lookup, "protonation_labels_json")] = str(token)
  return labels


def _pin(
  row: np.ndarray,
  kept: Sequence[tuple[str, int, str, str]],
  pins: Mapping[str, Any],
  what: str,
) -> np.ndarray:
  lookup = _residue_lookup(kept)
  out = row.copy()
  for key, raw in pins.items():
    index = _resolve_key(key, lookup, what)
    token = str(raw)
    new = token_index(token)
    name = kept[index][3]
    if token in V6_PROTONATION_TOKENS and THREE_TO_ONE.get(name) != canonical_letter(new):
      msg = (
        f"{what}: {key!r} is {name}, so {token!r} (a {canonical_letter(new)} state) cannot apply"
      )
      raise ValueError(msg)
    out[index] = new
  return out


def _variant_rows(
  reference: np.ndarray,
  kept: Sequence[tuple[str, int, str, str]],
  options: ProtonPottsOptions,
  sequences: Sequence[str],
) -> list[np.ndarray]:
  rows: list[np.ndarray] = []
  for name, body in _read_json(options.variants_json, "variants_json").items():
    what = f"variants_json[{name!r}]"
    if isinstance(body, dict):
      rows.append(_pin(reference, kept, body, what))
    elif isinstance(body, list):
      if len(body) != reference.shape[0]:
        msg = f"{what}: {len(body)} tokens for {reference.shape[0]} kept residues"
        raise ValueError(msg)
      rows.append(np.asarray([token_index(str(token)) for token in body], dtype=np.int64))
    else:
      msg = f"{what}: expected pins (object) or a token list, got {type(body).__name__}"
      raise ValueError(msg)  # noqa: TRY004
  for index, letters in enumerate(sequences):
    if len(letters) != reference.shape[0]:
      msg = f"sequences_to_score[{index}]: {len(letters)} letters for {reference.shape[0]} kept residues"
      raise ValueError(msg)
    rows.append(np.asarray([token_index(letter) for letter in letters], dtype=np.int64))
  return rows


def _prepare(
  path: Path,
  options: ProtonPottsOptions,
  purpose: str,
  sequences: Sequence[str],
) -> _Prepared:
  loaded = loaded_residues(path)
  labels = _reference_labels(loaded, options)
  if labels is not None:
    # Same validation the featurizer applies (vocabulary and parent residue), before any drop.
    encode_sequence([name for *_rest, name in loaded], labels)
  features = featurize_pdb(path, labels)
  kept = kept_residues(path)
  reference = features["S"][0]
  if len(kept) != reference.shape[0]:
    msg = f"{path}: {len(kept)} kept residues but {reference.shape[0]} feature rows"
    raise ProtonPottsInputError(msg)
  # pH design takes only the reference: its candidates are designs, not the scored variants.
  variants = [] if purpose == "sample" else _variant_rows(reference, kept, options, sequences)
  if purpose == "score:ddg" and not variants:
    msg = (
      f"score:ddg needs variants (ProtonPottsOptions.variants_json or sequences_to_score) for {path.name}; "
      "protonation states are not mutation targets, so there is no DMS enumeration"
    )
    raise ValueError(msg)
  rows = np.stack([reference, *variants]).astype(np.int32)
  length = reference.shape[0]
  graph = _Graph(
    coords=np.asarray(features["X"][0, :, :4, :], dtype=np.float32),
    present=np.ones(length, dtype=np.float32),
    residue_idx=np.asarray(features["R_idx"][0], dtype=np.int32),
    chain_index=np.asarray(features["chain_labels"][0], dtype=np.int32),
    pad_valid=np.ones(length, dtype=np.bool_),
    sequences=rows,
  )
  return _Prepared(
    graph=graph,
    include_reference_in_output=purpose in _REFERENCE_PURPOSES,
    kept=tuple((chain, int(number), icode, name) for chain, number, icode, name in kept),
  )


FAMILY_DRIVERS.register("protonpottsmpnn")(ProtonPottsDriver())
