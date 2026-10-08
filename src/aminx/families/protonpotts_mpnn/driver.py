"""ProtonPottsMPNN family driver for ``score:energy`` and ``score:ddg``.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §42-§43. Registration runs when this module is
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
  tokens are not mutation targets, spec §35.1).

Output sequences are token INDICES into the vocabulary stamped on the result (``vocabulary`` attribute), never
one-letter strings: upstream's ``decode_sequences`` collapses ``HIS-P`` to ``H``, which loses the state.

Not handled: ``sample`` (P9, the pH design engine), and the four generic-MPNN fallback purposes, which REFUSE
rather than load a 21-wide model (spec §35.1).
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
from aminx.families.protonpotts_mpnn.energy import protonpotts_energies
from aminx.families.protonpotts_mpnn.features import (
  PdbFeatureError,
  featurize_pdb,
  kept_residues,
  loaded_residues,
)
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

_HANDLED = frozenset({"score:energy", "score:ddg"})
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


class ProtonPottsInputError(ValueError):
  """The structure cannot be scored (it is skipped, as the shipped Potts driver skips its own)."""


_JIT_ENERGIES = eqx.filter_jit(protonpotts_energies)


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
      block = _JIT_ENERGIES(
        self._model,
        jnp.asarray(graph.coords),
        jnp.asarray(graph.present),
        jnp.asarray(graph.residue_idx),
        jnp.asarray(graph.chain_index),
        jnp.asarray(graph.pad_valid),
        jnp.asarray(graph.sequences),
      )
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


class ProtonPottsDriver:
  """Score ProtonPottsMPNN through the family-driver seam."""

  name = "protonpottsmpnn"
  options_type = ProtonPottsOptions
  mpnn_fallback_purposes = frozenset({"jacobian", "inspect", "score:nll", "score:logits"})
  root_alphabet = _VOCABULARY

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """True for ``score:energy`` and ``score:ddg``."""
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
        "scripts/protonpotts/convert_protonpotts_checkpoint.py"
      )
      raise ValueError(msg)
    manifest = path.with_suffix(".json")
    if manifest.exists():
      alphabet = json.loads(manifest.read_text(encoding="utf-8")).get("alphabet")
      if alphabet != PROTONPOTTS_V6.name:
        msg = f"{manifest}: alphabet {alphabet!r}, expected {PROTONPOTTS_V6.name!r}"
        raise ValueError(msg)
    skeleton = PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)
    return cast("PottsMPNN", eqx.tree_deserialise_leaves(path, skeleton))

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
    purpose = f"score:{spec.output_kind}"
    options = _options(spec)
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(_inputs(spec)):
      if not isinstance(item, (str, Path)):
        pending.append((index, "protonpottsmpnn_requires_pdb"))
        continue
      try:
        prepared = _prepare(Path(item), options, purpose, _spec_sequences(spec))
      except PdbFeatureError as exc:
        pending.append((index, str(exc)))
        continue
      yield FamilyBatch(
        input_indices=(index,),
        arrays={"prepared": (prepared,)},
        skipped=tuple(pending),
        lengths=(int(prepared.graph.coords.shape[0]),),
      )
      pending = []
    if pending:
      yield FamilyBatch(
        input_indices=(),
        arrays={"prepared": ()},
        skipped=tuple(pending),
        lengths=(),
      )

  def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
    """Structure and candidate axes."""
    del spec, purpose
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
    """Per-batch energy or ddG stage."""
    del spec, purpose
    return _ScoreStages(cast("PottsMPNN", model))

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Arrays one chunk returns. ``*_tokens`` are indices into the stamped ``vocabulary``."""
    del spec
    vocab = {"vocabulary": _VOCABULARY}
    if purpose == "score:energy":
      return {
        "energy": SinkArraySpec(dims=("N_cand",), dtype="float32", attrs={}),
        "candidate_ids": SinkArraySpec(dims=("N_cand",), dtype="int32", attrs={}),
        "candidate_tokens": SinkArraySpec(dims=("N_cand", "L_total"), dtype="int32", attrs=vocab),
      }
    if purpose == "score:ddg":
      return {
        "ddg": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
        "mutant_ids": SinkArraySpec(dims=("N_mut",), dtype="int32", attrs={}),
        "ddg_expt": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
        "mutant_tokens": SinkArraySpec(dims=("N_mut", "L_total"), dtype="int32", attrs=vocab),
      }
    msg = f"protonpottsmpnn does not support {purpose} in v1"
    raise ValueError(msg)


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
  variants = _variant_rows(reference, kept, options, sequences)
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
  return _Prepared(graph=graph, include_reference_in_output=purpose == "score:energy")


FAMILY_DRIVERS.register("protonpottsmpnn")(ProtonPottsDriver())
