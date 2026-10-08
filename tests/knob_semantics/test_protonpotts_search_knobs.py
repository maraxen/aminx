# ruff: noqa: S101
"""ProtonPottsMPNN optimiser (search) knobs, driven through ``sample`` on a toy PDB.

Each ``test_knob_semantics_<field>`` checks one ``ProtonPottsOptions`` search knob against its own meaning.
Every test has a counterfactual: the default run is repeated and must reproduce itself, then the changed value
must change an observable output in the way the knob's meaning says.

The model is random-init and the structures are toy backbones, so the regimes were chosen by reading
``ph_descent.py``, ``ph_greedy.py``, ``ph_plan.py`` and ``ph_config.py``. Observable outputs are the designed
``sequence`` rows, ``center_positions`` and ``selective_energy``. Step counts are not public, so the ``cv_*``
tests assert through the sequence.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ProtonPottsDriver
from aminx.families.protonpotts_mpnn.energy import protonpotts_table
from aminx.families.protonpotts_mpnn.ph_descent import rep_class_mask
from aminx.families.protonpotts_mpnn.ph_plan import neighbour_mask
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample
from aminx.run.options import ProtonPottsOptions
from aminx.run.specs import SamplingSpecification

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR"]
# Charged and repetitive-window classes in many slots, so the window term has residues to act on.
_PATTERN = (
  "LYS", "GLU", "ARG", "ASP", "HIS", "ALA", "GLY", "SER",
  "LYS", "GLU", "ARG", "ASP", "HIS", "THR", "LEU", "VAL",
)  # fmt: skip
_WINDOW_PARENTS = ("ARG", "LYS", "HIS", "ASP", "GLU")


def _atom(
  serial: int,
  name: str,
  resname: str,
  chain: str,
  number: int,
  xyz: tuple[float, float, float],
) -> str:
  padded = f" {name:<3}"
  return (
    f"ATOM  {serial:>5} {padded} {resname:>3} {chain}{number:>4}    "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2}"
  )


def _write_pdb(path: Path, chains: list[tuple[str, list[str], float]]) -> Path:
  """Backbone-only residues. Each chain is ``(letter, residue names, x offset)``, numbered from 1."""
  rng = np.random.default_rng(0)
  lines: list[str] = []
  for chain, names, offset in chains:
    for i, resname in enumerate(names, start=1):
      centre = np.array([3.8 * i + offset, 2.0 * np.sin(i), 2.0 * np.cos(i)])
      for k, atom in enumerate(("N", "CA", "C", "O")):
        xyz = centre + rng.normal(scale=0.4, size=3) + np.array([0.6 * k, 0.0, 0.0])
        lines.append(_atom(len(lines) + 1, atom, resname, chain, i, (xyz[0], xyz[1], xyz[2])))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  return _write_pdb(tmp_path / "toy.pdb", [("A", RESIDUES, 0.0)])


@pytest.fixture
def registered() -> Iterator[ProtonPottsDriver]:
  driver = ProtonPottsDriver()
  FAMILY_DRIVERS.register("protonpottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("protonpottsmpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6))
  return path


def _design_options(**overrides: Any) -> ProtonPottsOptions:  # noqa: ANN401
  values: dict[str, Any] = {
    "binder_chain": "A",
    "center_types": ("HIS-P",),
    "block_size": 2,
    "samples_per_site": 2,
    "neighbour_k": 0,
    "max_mutations": 0,
    "block_max_rounds": 3,
    "temperature": 0.05,
  }
  values.update(overrides)
  return ProtonPottsOptions(**values)


def _sample_spec(
  inputs: object,
  model_path: Path,
  options: ProtonPottsOptions,
  seed: int = 42,
) -> SamplingSpecification:
  return SamplingSpecification(
    inputs=inputs,  # type: ignore[arg-type]
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    random_seed=seed,
    protonpotts=options,
  )


def _designs(pdb: Path, model_path: Path, options: ProtonPottsOptions, seed: int = 42) -> dict[str, Any]:
  spec = _sample_spec(str(pdb), model_path, options, seed)
  return sample(spec)["structures"]["0"]["arrays"]


def _chain_pdb(tmp_path: Path, length: int) -> Path:
  names = [_PATTERN[i % len(_PATTERN)] for i in range(length)]
  return _write_pdb(tmp_path / f"chain_{length}.pdb", [("A", names, 0.0)])


def _long_options(**overrides: Any) -> ProtonPottsOptions:  # noqa: ANN401
  """Block descent at T=0 on a 32-residue chain, with the neighbourhood left wide open."""
  values: dict[str, Any] = {
    "temperature": 0.0,
    "neighbour_k": 16,
    "max_mutations": 0,
    "block_size": 3,
    "block_max_rounds": 10,
    "samples_per_site": 1,
  }
  values.update(overrides)
  return _design_options(**values)


def _greedy_options(**overrides: Any) -> ProtonPottsOptions:  # noqa: ANN401
  """Centre-free greedy energy block at T=0: the only regime in which the ``cv_*`` knobs apply."""
  values: dict[str, Any] = {
    "design_method": "greedy_energy_block",
    "infill_scope": "chain",
    "center_types": (),
    "temperature": 0.0,
    "samples_per_site": 1,
    "block_size": 3,
    "cv_patience": 3,
    "cv_max": 50,
    "repetitive_window_weight": 1.0,
  }
  values.update(overrides)
  return _design_options(**values)


def _binder_context(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  options: ProtonPottsOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Native tokens, binder mask and kNN ``e_idx`` of the structure, built the way the driver builds them."""
  spec = _sample_spec(str(pdb), model_path, options)
  prepared = next(registered.batches(spec)).arrays["prepared"][0]
  graph = prepared.graph
  model = registered.load(spec)
  _table, e_idx = protonpotts_table(
    model,
    jnp.asarray(graph.coords),
    jnp.asarray(graph.present),
    jnp.asarray(graph.residue_idx),
    jnp.asarray(graph.chain_index),
    jnp.asarray(graph.pad_valid),
  )
  native = np.asarray(graph.sequences[0], dtype=np.int32)
  binder = np.asarray([chain == "A" for chain, *_rest in prepared.kept], dtype=bool)
  return native, binder, np.asarray(e_idx)


def _window_pairs(sequence: np.ndarray, radius: int) -> int:
  """Pairs of repetitive-window-class tokens within ``radius`` residues: what the window term penalises.

  The chain is one consecutive run numbered 1..L, so residue distance is index distance.
  """
  cls = rep_class_mask(_WINDOW_PARENTS)[sequence] > 0
  return sum(int(np.sum(cls[:-gap] & cls[gap:])) for gap in range(1, radius + 1))


# --- block_size --------------------------------------------------------------------------------


def test_knob_semantics_block_size(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues, one HIS-P centre. Size 1 is single-site descent; size 3 commits
  # joint moves that include the pair terms inside the block, so the fixed points differ.
  pdb = _chain_pdb(tmp_path, 32)
  default = _designs(pdb, model_path, _long_options(block_size=3))
  again = _designs(pdb, model_path, _long_options(block_size=3))
  assert np.array_equal(default["sequence"], again["sequence"])

  single = _designs(pdb, model_path, _long_options(block_size=1))
  assert not np.array_equal(default["sequence"], single["sequence"])


# --- combined_lambda ---------------------------------------------------------------------------


def test_knob_semantics_combined_lambda(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues. ``block_objective`` weights the Potts stability by (1 - lambda) / sdH
  # and the selective term by lambda / sdSel. Lambda 0 optimises stability only; lambda 1 the selective gap only.
  pdb = _chain_pdb(tmp_path, 32)
  default = _designs(pdb, model_path, _long_options(combined_lambda=0.3))
  again = _designs(pdb, model_path, _long_options(combined_lambda=0.3))
  assert np.array_equal(default["sequence"], again["sequence"])

  stability_only = _designs(pdb, model_path, _long_options(combined_lambda=0.0))
  selective_only = _designs(pdb, model_path, _long_options(combined_lambda=1.0))
  assert not np.array_equal(stability_only["sequence"], selective_only["sequence"])
  assert not np.array_equal(default["sequence"], selective_only["sequence"])
  # Lambda 1 descends the selective term alone, so its final gap should be no larger than the gap of the
  # stability-only design. This rests on a descent tendency rather than an identity: if it fails, look at the
  # descent before at this line.
  assert float(selective_only["selective_energy"][0]) <= float(stability_only["selective_energy"][0]) + 1e-6


# --- temperature -------------------------------------------------------------------------------


def test_knob_semantics_temperature(registered: ProtonPottsDriver, pdb: Path, model_path: Path) -> None:
  del registered
  # Regime: block descent on the 10-residue toy PDB, the same regime as the seed test in test_driver_design.
  base = _designs(pdb, model_path, _design_options(temperature=0.05), seed=7)
  again = _designs(pdb, model_path, _design_options(temperature=0.05), seed=7)
  assert np.array_equal(base["sequence"], again["sequence"])
  others = [_designs(pdb, model_path, _design_options(temperature=0.05), seed=s)["sequence"] for s in range(8)]
  assert any(not np.array_equal(base["sequence"], other) for other in others)

  # Counterfactual: at T=0 the seed is never read, so seeds 1 and 2 must agree exactly.
  cold_a = _designs(pdb, model_path, _design_options(temperature=0.0), seed=1)
  cold_b = _designs(pdb, model_path, _design_options(temperature=0.0), seed=2)
  assert np.array_equal(cold_a["sequence"], cold_b["sequence"])


# --- samples_per_site --------------------------------------------------------------------------


def test_knob_semantics_samples_per_site(registered: ProtonPottsDriver, pdb: Path, model_path: Path) -> None:
  del registered
  # Regime: one placement plan (one centre type), so the design count is samples_per_site times one plan.
  default = _designs(pdb, model_path, _design_options(samples_per_site=2))
  assert default["sequence"].shape[0] == 2
  again = _designs(pdb, model_path, _design_options(samples_per_site=2))
  assert np.array_equal(default["sequence"], again["sequence"])

  wider = _designs(pdb, model_path, _design_options(samples_per_site=5))
  assert wider["sequence"].shape[0] == 5
  assert sorted(wider["design_sample"].tolist()) == list(range(5))


# --- neighbour_k -------------------------------------------------------------------------------


def test_knob_semantics_neighbour_k(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  # Regime: infill_scope "neighbourhood", T=0, 32 residues. The centre's designable set is
  # neighbour_mask(e_idx, centre, binder, k), which shrinks as k shrinks.
  pdb = _chain_pdb(tmp_path, 32)
  wide_options = _long_options(neighbour_k=16)
  wide = _designs(pdb, model_path, wide_options)
  again = _designs(pdb, model_path, wide_options)
  assert np.array_equal(wide["sequence"], again["sequence"])
  narrow = _designs(pdb, model_path, _long_options(neighbour_k=2))

  native, binder, e_idx = _binder_context(registered, pdb, model_path, wide_options)
  centre = int(narrow["center_positions"][0, 0])
  near_narrow = neighbour_mask(e_idx, centre, binder, 2)
  near_wide = neighbour_mask(e_idx, centre, binder, 16)
  near_narrow[centre] = True  # the pinned centre may itself change to its protonated token
  near_wide[centre] = True
  assert near_narrow.sum() < near_wide.sum()

  narrow_allowed = set(np.flatnonzero(near_narrow).tolist())
  moved_narrow = set(np.flatnonzero(narrow["sequence"][0] != native).tolist())
  moved_wide = set(np.flatnonzero(wide["sequence"][0] != native).tolist())
  assert moved_narrow <= narrow_allowed
  # The wide neighbourhood reaches positions the narrow one forbids, and the optimiser changes some of them.
  assert moved_wide - narrow_allowed


# --- max_mutations -----------------------------------------------------------------------------


def test_knob_semantics_max_mutations(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  # Regime: block descent, T=0, 32 residues. ``finalize_plan`` keeps the max_mutations closest-coupled designable
  # positions when the set is larger, and 0 means no cap. The pinned centre is not counted against the cap.
  pdb = _chain_pdb(tmp_path, 32)
  default_options = _long_options(max_mutations=20)
  default = _designs(pdb, model_path, default_options)
  again = _designs(pdb, model_path, default_options)
  assert np.array_equal(default["sequence"], again["sequence"])
  native, _binder, _e_idx = _binder_context(registered, pdb, model_path, default_options)
  centre = int(default["center_positions"][0, 0])

  def moved(row: np.ndarray) -> list[int]:
    return [p for p in np.flatnonzero(row != native).tolist() if p != centre]

  assert len(moved(default["sequence"][0])) <= 20

  capped = _designs(pdb, model_path, _long_options(max_mutations=3))
  assert len(moved(capped["sequence"][0])) <= 3

  uncapped = _designs(pdb, model_path, _long_options(max_mutations=0))
  assert len(moved(uncapped["sequence"][0])) > 3  # with no cap the optimiser moves more than three positions


# --- repetitive_window_weight ------------------------------------------------------------------


def test_knob_semantics_repetitive_window_weight(
  registered: ProtonPottsDriver, tmp_path: Path, model_path: Path
) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues with window-class residues in most slots. The window term adds ``rw``
  # per pair of class tokens within the radius, so a large weight should leave fewer such pairs.
  pdb = _chain_pdb(tmp_path, 32)
  default_options = _long_options(repetitive_window_weight=1.0)
  default = _designs(pdb, model_path, default_options)
  again = _designs(pdb, model_path, default_options)
  assert np.array_equal(default["sequence"], again["sequence"])

  off = _designs(pdb, model_path, _long_options(repetitive_window_weight=0.0))
  strong = _designs(pdb, model_path, _long_options(repetitive_window_weight=20.0))
  assert not np.array_equal(off["sequence"][0], strong["sequence"][0])
  assert _window_pairs(strong["sequence"][0], 2) < _window_pairs(off["sequence"][0], 2)


# --- repetitive_window_radius ------------------------------------------------------------------


def test_knob_semantics_repetitive_window_radius(
  registered: ProtonPottsDriver, tmp_path: Path, model_path: Path
) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues, weight 20 so the radius decides which neighbours are penalised.
  pdb = _chain_pdb(tmp_path, 32)

  def run(radius: int) -> np.ndarray:
    options = _long_options(repetitive_window_weight=20.0, repetitive_window_radius=radius)
    return _designs(pdb, model_path, options)["sequence"][0]

  default = run(2)
  assert np.array_equal(default, run(2))
  adjacent = run(1)
  # The radius is clamped to at least 1 (``max(1, radius)`` in ``block_objective``), so radius 0 equals radius 1.
  assert np.array_equal(run(0), adjacent)
  assert not np.array_equal(adjacent, run(4))


# --- repetitive_window_parents -----------------------------------------------------------------


def test_knob_semantics_repetitive_window_parents(
  registered: ProtonPottsDriver, tmp_path: Path, model_path: Path
) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues, weight 20. An empty parent set gives an all-zero class mask, so the
  # window term vanishes exactly and the run equals a run with weight 0 under the default parents.
  pdb = _chain_pdb(tmp_path, 32)
  default_options = _long_options(repetitive_window_weight=20.0)
  default = _designs(pdb, model_path, default_options)
  again = _designs(pdb, model_path, default_options)
  assert np.array_equal(default["sequence"], again["sequence"])

  no_parents = _designs(
    pdb,
    model_path,
    _long_options(repetitive_window_weight=20.0, repetitive_window_parents=()),
  )
  zero_weight = _designs(pdb, model_path, _long_options(repetitive_window_weight=0.0))
  assert np.array_equal(no_parents["sequence"], zero_weight["sequence"])
  assert not np.array_equal(default["sequence"], no_parents["sequence"])


# --- block_max_rounds --------------------------------------------------------------------------


def test_knob_semantics_block_max_rounds(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  del registered
  # Regime: block descent, T=0, 32 residues from the native start. One sweep is not a fixed point here: the loop
  # runs sweeps while the previous one changed something, up to block_max_rounds (``_sweep_to_convergence``).
  pdb = _chain_pdb(tmp_path, 32)
  converged_options = _long_options(block_max_rounds=10)
  converged = _designs(pdb, model_path, converged_options)
  again = _designs(pdb, model_path, converged_options)
  assert np.array_equal(converged["sequence"], again["sequence"])

  one_sweep = _designs(pdb, model_path, _long_options(block_max_rounds=1))
  assert not np.array_equal(converged["sequence"], one_sweep["sequence"])


# --- cv_patience (greedy_energy_block only) ----------------------------------------------------


def test_knob_semantics_cv_patience(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  # Regime: centre-free greedy, T=0, 32 residues, repetitive weight 20. The run stops after cv_patience * N steps
  # without a new best Potts energy. Patience 0 never enters the loop, so the native comes back exactly. Patience
  # 1 versus 3 is the spec's comparison. The strong window weight makes energy-rising steps frequent, which is what
  # lets the patience window bind on this chain.
  pdb = _chain_pdb(tmp_path, 32)
  options = _greedy_options(cv_patience=3, repetitive_window_weight=20.0)
  default = _designs(pdb, model_path, options)
  again = _designs(pdb, model_path, options)
  assert np.array_equal(default["sequence"], again["sequence"])
  native, _binder, _e_idx = _binder_context(registered, pdb, model_path, options)

  no_patience = _designs(pdb, model_path, _greedy_options(cv_patience=0, repetitive_window_weight=20.0))
  assert np.array_equal(no_patience["sequence"][0], native)

  # At T=0 greedy descent reaches its best sequence at once, so patience cannot matter there. It matters when sampling
  # can find a better sequence after a non-improving step: at a high temperature a short patience stops earlier than a
  # long one for at least one seed.
  def finals(patience: int) -> list[np.ndarray]:
    options = _greedy_options(cv_patience=patience, temperature=1.0)
    return [_designs(pdb, model_path, options, seed=seed)["sequence"][0] for seed in range(6)]

  short, long_ = finals(1), finals(6)
  assert any(not np.array_equal(a, b) for a, b in zip(short, long_, strict=True))


# --- cv_max (greedy_energy_block only) ---------------------------------------------------------


def test_knob_semantics_cv_max(registered: ProtonPottsDriver, tmp_path: Path, model_path: Path) -> None:
  del registered
  # Regime: centre-free greedy, T=0, 40 residues, block_size 1 so each step commits one position. The step cap is
  # cv_max * N, so cv_max 1 stops after 40 steps, which the default cap of 50 * 40 does not reach on this chain.
  pdb = _chain_pdb(tmp_path, 40)
  default_options = _greedy_options(block_size=1, cv_max=50)
  default = _designs(pdb, model_path, default_options)
  again = _designs(pdb, model_path, default_options)
  assert np.array_equal(default["sequence"], again["sequence"])
  capped = _designs(pdb, model_path, _greedy_options(block_size=1, cv_max=1))
  assert not np.array_equal(default["sequence"][0], capped["sequence"][0])
