"""Host-side placement planning for the ProtonPottsMPNN pH-design optimiser (numpy, no JAX).

Port of the ``backend="potts"`` planning stage of upstream ``PottsMPNNPHEngine``
(ProtonPottsMPNN foundry ``potts_mpnn_ph.py``). Only two placement modes are ported:
``placement_by="scan_potts"`` (ranked by the caller-supplied native-sequence field) and explicit
centres, with ``infill_scope`` ``"neighbourhood"`` or ``"chain"``, plus the centre-free plan of
``greedy_energy_block`` (``center_count=0``).

Upstream anchors are given per function. Several upstream docstrings are wrong; these functions
follow the code, not the docstrings.

Indices are aminx's v6 token order (``vocab.PROTONPOTTS_V6``). Upstream's ``UNK`` is ``X``.

Ties: upstream ranks with ``torch.argsort``, which is not stable. Where upstream's tie order is
not specified, this module uses ``np.argsort(kind="stable")`` and ties resolve to ascending
position. Upstream's ``_knn_rank`` and ``_finalize_plan`` sort on explicit (rank, index) keys,
so their ties are deterministic and are mirrored exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

CENTRE_TYPES: tuple[str, ...] = ("HIS-P", "ASP-P", "GLU-P")
"""Protonation tokens that may pin a centre (spec §ph_plan; upstream accepts any dep_map key)."""

INFILL_SCOPES: tuple[str, ...] = ("neighbourhood", "chain")

_UNK_SYMBOL = "X"
_VOCAB: tuple[str, ...] = PROTONPOTTS_V6.symbols
_INDEX: dict[str, int] = {symbol: i for i, symbol in enumerate(_VOCAB)}

# Rank given to a designable position that appears in no centre's outgoing kNN row
# (incoming-only neighbours). Matches upstream ``_knn_rank``; any value above K works.
_UNRANKED = 10**6


class Pin(NamedTuple):
  position: int
  protonation_type: str
  prot_idx: int
  dep_idxs: tuple[int, ...]
  res_id: int


class Plan(NamedTuple):
  pins: tuple[Pin, ...]
  designable: tuple[int, ...]
  label: str


def valid_token_mask(forbidden_tokens: Sequence[str]) -> np.ndarray:
  """Boolean (30,) mask of tokens the sampler may emit.

  Port of upstream ``valid_aa_mask`` (``potts_mpnn_ph.py`` 1702-1715): every token is valid
  except the unknown token ``X`` (upstream ``UNK``) and each forbidden token that exists in the
  vocabulary. A forbidden name that is not in the vocabulary (upstream's default ``HIS-D``, which
  v6 lacks) is ignored, as upstream does.
  """
  mask = np.ones(len(_VOCAB), dtype=bool)
  mask[_INDEX[_UNK_SYMBOL]] = False
  for name in forbidden_tokens:
    symbol = _UNK_SYMBOL if name == "UNK" else name
    idx = _INDEX.get(symbol)
    if idx is not None:
      mask[idx] = False
  return mask


def neighbour_mask(
  e_idx: np.ndarray,
  centre: int,
  binder_mask: np.ndarray,
  neighbour_k: int,
) -> np.ndarray:
  """Designable neighbourhood of one centre: coupled kNN in both graph directions.

  Port of upstream ``neighbour_mask`` (``potts_mpnn_ph.py`` 1717-1734). Slot 0 of ``e_idx`` is
  the position itself and is excluded. With ``neighbour_k > 0`` the kNN is truncated to the
  ``neighbour_k`` nearest partners, i.e. slots ``1..neighbour_k`` (``K' = min(K, neighbour_k+1)``),
  applied to both the outgoing row and the incoming check. ``neighbour_k <= 0`` keeps all K slots.

  The result includes outgoing neighbours (``e_idx[centre, 1:K']``) and incoming positions ``i``
  whose slots ``1:K'`` contain the centre, intersected with ``binder_mask``. The centre itself
  is never included.
  """
  e_idx = np.asarray(e_idx)
  binder_mask = np.asarray(binder_mask, dtype=bool)
  n_pos, n_slots = e_idx.shape
  k_eff = min(n_slots, neighbour_k + 1) if neighbour_k > 0 else n_slots
  mask = np.zeros(n_pos, dtype=bool)
  for j in e_idx[centre, 1:k_eff].tolist():
    if j != centre and binder_mask[j]:
      mask[j] = True
  incoming = np.flatnonzero((e_idx[:, 1:k_eff] == centre).any(axis=1))
  for i in incoming.tolist():
    if i != centre and binder_mask[i]:
      mask[i] = True
  return mask


def knn_rank(e_idx: np.ndarray, designable: Sequence[int], pins: Sequence[Pin]) -> list[int]:
  """Designable positions ordered closest-coupled first.

  Port of upstream ``_knn_rank`` (``potts_mpnn_ph.py`` 2180-2190). A position's rank is its best
  (smallest) index in any centre's outgoing kNN row, where that row is ``e_idx[centre, 1:]``
  (upstream ``nbr_idx = E_idx[:, 1:]``, line 1523). Positions in no centre's row get
  ``_UNRANKED`` and sort after every ranked one. Ties break on ascending position, so the
  result is deterministic.
  """
  e_idx = np.asarray(e_idx)
  best = {int(j): _UNRANKED for j in designable}
  for pin in pins:
    for rank, j in enumerate(e_idx[pin.position, 1:].tolist()):
      if j in best:
        best[j] = min(best[j], rank)
  return sorted(best, key=lambda j: (best[j], j))


def block_partners(
  nbr_row: Sequence[int] | np.ndarray,
  p: int,
  designable_set: set[int] | frozenset[int],
  block_size: int,
) -> list[int]:
  """Block for position ``p``: ``p`` plus its nearest designable partners, at most ``block_size``.

  Port of upstream ``_block_partners`` (``potts_mpnn_ph.py`` 2041-2052), with ``nbr_row`` =
  ``e_idx[p, 1:]``. Partners are taken in kNN order from ``nbr_row`` and only if they are in
  ``designable_set``. The block can be SHORTER than ``block_size`` when fewer designable partners
  exist, and it is ``[p]`` when ``block_size <= 1``.
  """
  block = [int(p)]
  if block_size <= 1:
    return block
  for j in np.asarray(nbr_row).tolist():
    if j != p and j in designable_set and j not in block:
      block.append(int(j))
      if len(block) == block_size:
        break
  return block


def block_table(
  e_idx: np.ndarray,
  designable: Sequence[int],
  block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Padded blocks for every designable position, in the given order.

  Returns ``(blocks, block_valid)``, both shaped ``(N, block_size)``. ``blocks`` holds positions
  padded with ``-1``. ``block_valid`` is True where a block entry is real. Each row comes from
  ``block_partners`` (upstream ``_block_partners``, 2041-2052), so a block can be shorter than
  ``block_size`` and is padded on the right.
  """
  if block_size < 1:
    raise ValueError(f"block_size must be >= 1, got {block_size}")
  e_idx = np.asarray(e_idx)
  order = [int(p) for p in designable]
  designable_set = set(order)
  blocks = np.full((len(order), block_size), -1, dtype=np.int64)
  block_valid = np.zeros((len(order), block_size), dtype=bool)
  for n, p in enumerate(order):
    block = block_partners(e_idx[p, 1:], p, designable_set, block_size)
    blocks[n, : len(block)] = block
    block_valid[n, : len(block)] = True
  return blocks, block_valid


def placement_scores(
  field: np.ndarray,
  prot_idx: int,
  dep_idxs: Sequence[int],
  binder_free_mask: np.ndarray,
) -> np.ndarray:
  """Selective placement score per position, lower is better. Shape ``(L,)``.

  Port of upstream ``_ranked_candidates`` (``potts_mpnn_ph.py`` 2563-2582) with
  ``selective=True``. Upstream computes ``(ef[prot] - base) - min_d (ef[d] - base)``, where
  ``base`` is the native residue's energy at that position. ``base`` cancels, so this computes
  ``field[j, prot] - min_d field[j, d]``. ``field`` is the ``(L, V)`` conditional-energy table
  of the NATIVE sequence, supplied by the caller. It is not computed here.

  Positions outside ``binder_free_mask`` get ``+inf``.

  Any free binder residue can become a centre. There is NO residue-type restriction, as upstream.
  """
  field = np.asarray(field, dtype=np.float64)
  deps = list(dep_idxs)
  if not deps:
    raise ValueError("placement_scores needs at least one deprotonated contrast token")
  score = field[:, prot_idx] - field[:, deps].min(axis=1)
  return np.where(np.asarray(binder_free_mask, dtype=bool), score, np.inf)


def _ranked_positions(scores: np.ndarray) -> list[int]:
  """Finite-score positions, best (lowest) first. Ties resolve to ascending position."""
  order = np.argsort(scores, kind="stable")
  return [int(p) for p in order if np.isfinite(scores[p])]


def _pin_indices(
  protonation_type: str,
  dep_map: Mapping[str, Sequence[str]],
) -> tuple[int, tuple[int, ...]]:
  """Vocabulary indices for a centre and its contrast tokens.

  Port of upstream ``_pin_idxs`` (``potts_mpnn_ph.py`` 1293-1309). Upstream raises KeyError here,
  and this module raises ValueError, naming the tokens. A v3/v4 map (``HID``, ``HIE``, ``HIS-D``)
  is refused because v6 lacks those tokens.
  """
  if protonation_type not in CENTRE_TYPES:
    raise ValueError(
      f"unknown centre protonation type {protonation_type!r}; expected one of {CENTRE_TYPES}",
    )
  if protonation_type not in dep_map:
    raise ValueError(f"dep_map has no contrast tokens for centre {protonation_type!r}")
  deps = tuple(dep_map[protonation_type])
  if not deps:
    raise ValueError(f"dep_map[{protonation_type!r}] is empty")
  missing = [t for t in deps if t not in _INDEX]
  if missing:
    raise ValueError(
      f"dep_map[{protonation_type!r}] names {missing}, which are absent from the "
      f"{PROTONPOTTS_V6.name} vocabulary. v3/v4 tokens (HID, HIE, HIS-D) are refused; "
      "v6 uses HIS-P as the charged contrast.",
    )
  return _INDEX[protonation_type], tuple(_INDEX[t] for t in deps)


def _make_pin(
  position: int,
  protonation_type: str,
  dep_map: Mapping[str, Sequence[str]],
  res_id: np.ndarray,
) -> Pin:
  """Port of upstream ``_pin`` (``potts_mpnn_ph.py`` 1311-1314)."""
  prot_idx, dep_idxs = _pin_indices(protonation_type, dep_map)
  return Pin(
    position=int(position),
    protonation_type=protonation_type,
    prot_idx=prot_idx,
    dep_idxs=dep_idxs,
    res_id=int(np.asarray(res_id)[position]),
  )


def finalize_plan(
  e_idx: np.ndarray,
  binder_mask: np.ndarray,
  pins: Sequence[Pin],
  *,
  infill_scope: str,
  neighbour_k: int,
  max_mutations: int,
) -> Plan | None:
  """Pins plus their designable set, or ``None`` when nothing is designable.

  Port of upstream ``_finalize_plan`` (``potts_mpnn_ph.py`` 2643-2663). Pins are sorted by
  position. The designable set is either every free binder position minus the pins
  (``"chain"``) or the union of the pins' neighbourhoods minus the pins (``"neighbourhood"``).
  When ``max_mutations > 0`` and the set is larger, it keeps the ``max_mutations``
  closest-coupled positions by ``knn_rank``.
  """
  if infill_scope not in INFILL_SCOPES:
    raise ValueError(f"infill_scope must be one of {INFILL_SCOPES}, got {infill_scope!r}")
  if max_mutations < 0:
    raise ValueError(f"max_mutations must be >= 0 (0 = no cap), got {max_mutations}")
  binder_mask = np.asarray(binder_mask, dtype=bool)
  pins = tuple(sorted(pins, key=lambda p: p.position))
  pin_pos = {p.position for p in pins}
  if infill_scope == "chain":
    designable = set(np.flatnonzero(binder_mask).tolist()) - pin_pos
  else:
    designable = set()
    for pin in pins:
      designable.update(
        np.flatnonzero(neighbour_mask(e_idx, pin.position, binder_mask, neighbour_k)).tolist(),
      )
    designable -= pin_pos
  if not designable:
    return None
  ordered = sorted(int(p) for p in designable)
  if max_mutations and len(ordered) > max_mutations:
    ordered = sorted(knn_rank(e_idx, ordered, pins)[:max_mutations])
  label = "+".join(sorted(p.protonation_type for p in pins))
  return Plan(pins=pins, designable=tuple(ordered), label=label)


def plan_from_center_types(
  field: np.ndarray,
  e_idx: np.ndarray,
  binder_mask: np.ndarray,
  res_id: np.ndarray,
  center_types: Sequence[str],
  dep_map: Mapping[str, Sequence[str]],
  *,
  infill_scope: str,
  neighbour_k: int,
  max_mutations: int,
) -> Plan | None:
  """One plan with one distinct, best-ranked centre per requested type.

  Port of upstream ``enumerate_placement_plans``'s ``center_types`` branch
  (``potts_mpnn_ph.py`` 1339-1356) with ``placement_by="scan_potts"``. Types are processed in
  list order. Each takes the best-scoring free position not already used (``placement_scores``),
  is pinned, and the plan is finalised. Returns ``None`` where upstream returns ``[]``: fewer
  free binder positions than types, no unused candidate, or an empty designable set.
  """
  binder_mask = np.asarray(binder_mask, dtype=bool)
  if int(binder_mask.sum()) < len(center_types):
    return None
  used: set[int] = set()
  pins: list[Pin] = []
  for ptype in center_types:
    prot_idx, dep_idxs = _pin_indices(ptype, dep_map)
    scores = placement_scores(field, prot_idx, dep_idxs, binder_mask)
    pos = next((p for p in _ranked_positions(scores) if p not in used), None)
    if pos is None:
      return None
    used.add(pos)
    pins.append(_make_pin(pos, ptype, dep_map, res_id))
  return finalize_plan(
    e_idx,
    binder_mask,
    pins,
    infill_scope=infill_scope,
    neighbour_k=neighbour_k,
    max_mutations=max_mutations,
  )


def plan_from_explicit_centers(
  e_idx: np.ndarray,
  binder_mask: np.ndarray,
  res_id: np.ndarray,
  centers: Sequence[tuple[int, str]],
  dep_map: Mapping[str, Sequence[str]],
  *,
  infill_scope: str,
  neighbour_k: int,
  max_mutations: int,
) -> Plan | None:
  """Pins at hand-chosen residue numbers, given as ``(res_id, protonation_type)`` pairs.

  Port of upstream ``enumerate_placement_plans``'s ``explicit_centers`` branch
  (``potts_mpnn_ph.py`` 1333-1337) and ``binder_pos_of_res_id`` (1749-1754). Upstream looks the
  residue up among all chain-A positions. This module requires a FREE binder position (a
  ``binder_mask`` True entry), as the spec asks, and raises ``ValueError`` otherwise. If several
  binder positions share a residue number, the last one wins, as upstream does.
  """
  binder_mask = np.asarray(binder_mask, dtype=bool)
  res_id = np.asarray(res_id)
  pos_of = {int(res_id[p]): int(p) for p in np.flatnonzero(binder_mask).tolist()}
  pins: list[Pin] = []
  for rid, ptype in centers:
    pos = pos_of.get(int(rid))
    if pos is None:
      raise ValueError(f"res_id {rid} is not a free binder position")
    pins.append(_make_pin(pos, ptype, dep_map, res_id))
  return finalize_plan(
    e_idx,
    binder_mask,
    pins,
    infill_scope=infill_scope,
    neighbour_k=neighbour_k,
    max_mutations=max_mutations,
  )


def plan_centre_free(binder_mask: np.ndarray) -> Plan | None:
  """Plan with no pins and every free binder position designable.

  Port of upstream's ``center_count == 0`` branch (``potts_mpnn_ph.py`` 1321-1331) for
  ``greedy_energy_block``, with ``infill_scope="chain"``. The caller passes the already
  restricted binder mask. Upstream's ``placement_region`` filter is not reproduced here.
  Returns ``None`` when the mask is empty, as upstream returns ``[]``.
  """
  designable = tuple(int(p) for p in np.flatnonzero(np.asarray(binder_mask, dtype=bool)))
  if not designable:
    return None
  return Plan(pins=(), designable=designable, label="")
