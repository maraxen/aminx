"""pH-design criteria for the ProtonPottsMPNN family: the in-scope subset of upstream ``PHDesignCriteria``.

Upstream: ``ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py``. The
dataclass is ``PHDesignCriteria`` (lines ~93-330); the ``__post_init__`` checks mirrored here are its
lines ~238-283. Production defaults come from the example ``ProtonPottsMPNN/inference/design_ph.py``
lines 66-95, NOT from the dataclass defaults (e.g. upstream ``block_size=2``, ``temperature=0.1``).

Scope: ``block_descent`` and ``greedy_energy_block`` on the Potts energy backend only. Anything else is
refused at construction, naming the debt item that tracks it:

- debt #2616: MCMC / two-phase / combined samplers and Gibbs (``converged_mcmc``, ``two_phase``,
  ``converged_mcmc_combined``, ``gibbs``).
- debt #2617: decoder-backed paths (``autoregressive``, ``mpnn_sample``, ``backend="mpnn"``,
  ``selective_source="decoder"``, ``placement_by="scan_mpnn"``).

Representation: every collection is a tuple so the config is hashable. ``dep_map`` is a tuple of
``(centre_type, (deprotonated_type, ...))`` pairs; use ``PHDesignConfig.dep_map_dict()`` for a dict view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from aminx.run.options import ProtonPottsOptions

MCMC_DEBT = 2616
DECODER_DEBT = 2617

IN_SCOPE_METHODS: tuple[str, ...] = ("block_descent", "greedy_energy_block")
_DEFERRED_MCMC_METHODS: tuple[str, ...] = (
  "converged_mcmc",
  "two_phase",
  "converged_mcmc_combined",
  "gibbs",
)
_DEFERRED_DECODER_METHODS: tuple[str, ...] = ("autoregressive", "mpnn_sample")

# Upstream DEFAULT_DEP_MAP maps HIS-P to (HID, HIE); v6 has no HID/HIE, so the production example uses HIS-S.
DEFAULT_DEP_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("HIS-P", ("HIS-S",)),
  ("ASP-P", ("ASP-D",)),
  ("GLU-P", ("GLU-D",)),
)


def _deferred(what: str, debt: int) -> ValueError:
  return ValueError(
    f"{what} is deferred (debt #{debt}). Only method 'block_descent' or 'greedy_energy_block' "
    "on backend 'potts' is supported.",
  )


@dataclass(frozen=True)
class PHDesignConfig:
  """One pH-design run's criteria. Validated on construction; see the module docstring for scope.

  Production defaults are the values in upstream ``inference/design_ph.py``. ``binder_chain`` may be
  None so the config can be built before the chain is known; ``validate_for_design`` rejects None.
  """

  method: str = "block_descent"
  binder_chain: str | None = None
  backend: str = "potts"
  selective_source: str = "potts"
  seed_source: str = "native"
  center_types: tuple[str, ...] = ("HIS-P", "ASP-P", "GLU-P")
  explicit_centers: tuple[tuple[int, str], ...] = ()
  center_count: int = 1
  dep_map: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_DEP_MAP
  forbidden_tokens: tuple[str, ...] = ("HIS-A", "ASP-A", "GLU-A", "UNK")
  placement_by: str = "scan_potts"
  placement_region: tuple[str, ...] = ("all",)
  temperature: float = 0.05
  samples_per_site: int = 2
  combined_lambda: float = 0.3
  block_size: int = 3
  global_weight: float = 0.0
  zscale_mode: str = "block"
  adjacent_repeat_weight: float = 0.0
  self_weight: float = 1.0
  block_max_rounds: int = 10
  sweep_order: str = "position"
  infill_scope: str = "neighbourhood"
  neighbour_k: int = 16
  max_mutations: int = 20
  cv_patience: int = 3
  cv_max: int = 50
  repetitive_window_parents: tuple[str, ...] = ("ARG", "LYS", "HIS", "ASP", "GLU")
  repetitive_window_radius: int = 2
  repetitive_window_weight: float = 1.0
  record_trajectory: bool = True

  def __post_init__(self) -> None:
    dep = self.dep_map.items() if isinstance(self.dep_map, Mapping) else self.dep_map
    set_ = object.__setattr__
    set_(self, "dep_map", tuple((str(k), tuple(str(t) for t in v)) for k, v in dep))
    set_(self, "center_types", tuple(str(t) for t in self.center_types))
    set_(self, "explicit_centers", tuple((int(r), str(t)) for r, t in self.explicit_centers))
    set_(self, "placement_region", tuple(str(r) for r in self.placement_region))
    set_(self, "forbidden_tokens", tuple(str(t) for t in self.forbidden_tokens))
    set_(self, "repetitive_window_parents", tuple(str(p) for p in self.repetitive_window_parents))
    self._check_scope()
    self._check_upstream()

  def _check_scope(self) -> None:
    if self.method in _DEFERRED_MCMC_METHODS:
      raise _deferred(f"method={self.method!r}", MCMC_DEBT)
    if self.method in _DEFERRED_DECODER_METHODS:
      raise _deferred(f"method={self.method!r}", DECODER_DEBT)
    if self.method not in IN_SCOPE_METHODS:
      raise ValueError(f"Unknown method {self.method!r}. Supported: {IN_SCOPE_METHODS}.")
    if self.backend != "potts":
      raise _deferred(f"backend={self.backend!r}", DECODER_DEBT)
    if self.selective_source != "potts":
      raise _deferred(f"selective_source={self.selective_source!r}", DECODER_DEBT)
    if self.placement_by == "scan_mpnn":
      raise _deferred("placement_by='scan_mpnn'", DECODER_DEBT)

  def _check_upstream(self) -> None:
    # Mirrors upstream PHDesignCriteria.__post_init__ for the in-scope fields.
    if self.seed_source not in ("inverse", "native"):
      raise ValueError(f"Unknown seed_source {self.seed_source!r}. Use 'inverse' or 'native'.")
    if self.placement_by not in ("random", "scan_potts"):
      raise ValueError(f"Unknown placement_by {self.placement_by!r}. Use 'random' or 'scan_potts'.")
    # Centre-free design (center_count == 0) is allowed only for greedy_energy_block on the whole chain
    # with no pinned centres, exactly as upstream.
    centre_free = (
      self.method == "greedy_energy_block"
      and self.infill_scope == "chain"
      and not self.center_types
      and not self.explicit_centers
    )
    if self.center_count < 1 and not (self.center_count == 0 and centre_free):
      raise ValueError(
        f"center_count must be >= 1 (got {self.center_count}); center_count=0 is allowed only for "
        "method='greedy_energy_block' with infill_scope='chain' and no center_types/explicit_centers.",
      )
    regions = {"interface", "core", "surface", "all"}
    bad_regions = set(self.placement_region) - regions
    if bad_regions:
      raise ValueError(
        f"Unknown placement_region {sorted(bad_regions)}. Allowed: {sorted(regions)}.",
      )
    if self.sweep_order not in ("position", "knn", "energy"):
      raise ValueError(
        f"Unknown sweep_order {self.sweep_order!r}. Use 'position', 'knn' or 'energy'.",
      )
    if self.infill_scope not in ("neighbourhood", "chain"):
      raise ValueError(
        f"Unknown infill_scope {self.infill_scope!r}. Use 'neighbourhood' or 'chain'.",
      )
    if self.zscale_mode not in ("single_mutation", "block"):
      raise ValueError(
        f"Unknown zscale_mode {self.zscale_mode!r}. Use 'single_mutation' or 'block'.",
      )
    if self.self_weight < 0:
      raise ValueError(f"self_weight must be >= 0 (got {self.self_weight}).")
    if self.explicit_centers and self.center_count not in (1, len(self.explicit_centers)):
      raise ValueError(
        f"center_count must equal len(explicit_centers) ({len(self.explicit_centers)}) when "
        f"explicit_centers is set (got {self.center_count}). center_count is taken from center_types "
        "when center_types is given, so clear center_types to use explicit_centers.",
      )
    if self.center_types:
      unknown = set(self.center_types) - set(self.dep_map_dict())
      if unknown:
        raise ValueError(
          f"center_types {sorted(unknown)} not in dep_map keys {sorted(self.dep_map_dict())}.",
        )
      # Upstream keeps center_count consistent with the pinned composition.
      object.__setattr__(self, "center_count", len(self.center_types))

  def dep_map_dict(self) -> dict[str, tuple[str, ...]]:
    """The dep_map as a plain dict, for lookups."""
    return dict(self.dep_map)

  def validate_for_design(self) -> None:
    """Raise ``ValueError`` unless the config can drive a design run (needs a binder chain)."""
    if not self.binder_chain:
      raise ValueError("binder_chain is required for pH design (got None or empty).")


def config_from_options(options: ProtonPottsOptions) -> PHDesignConfig:
  """Map the flat ``ProtonPottsOptions`` design fields onto a validated ``PHDesignConfig``.

  ``None`` (binder_chain) and an empty ``dep_map`` mean the config default. Fields absent from
  ``ProtonPottsOptions`` (e.g. ``seed_source``, ``placement_by``) keep their production defaults.
  """
  default = PHDesignConfig()
  # Centre-free design (greedy, whole chain, no centres named) is upstream's center_count == 0 (lines 257-262).
  centre_free = (
    options.design_method == "greedy_energy_block"
    and options.infill_scope == "chain"
    and not options.center_types
    and not options.explicit_centers
  )
  return PHDesignConfig(
    center_count=0 if centre_free else 1,  # 1 is the dataclass default; center_types then sets it
    method=options.design_method,
    binder_chain=options.binder_chain,
    center_types=tuple(options.center_types),
    explicit_centers=tuple(options.explicit_centers),
    dep_map=tuple(options.dep_map) or default.dep_map,
    forbidden_tokens=tuple(options.forbidden_tokens),
    temperature=options.temperature,
    samples_per_site=options.samples_per_site,
    combined_lambda=options.combined_lambda,
    block_size=options.block_size,
    neighbour_k=options.neighbour_k,
    max_mutations=options.max_mutations,
    infill_scope=options.infill_scope,
    repetitive_window_weight=options.repetitive_window_weight,
    repetitive_window_radius=options.repetitive_window_radius,
    repetitive_window_parents=tuple(options.repetitive_window_parents),
    block_max_rounds=options.block_max_rounds,
    cv_patience=options.cv_patience,
    cv_max=options.cv_max,
    record_trajectory=options.record_trajectory,
  )
