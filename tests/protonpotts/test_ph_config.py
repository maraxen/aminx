"""Tests for the pH-design config (in-scope subset of upstream PHDesignCriteria) and its options mapping."""

from __future__ import annotations

import dataclasses
import json

import pytest

from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig, config_from_options
from aminx.run.options import ProtonPottsOptions
from aminx.run.spec_json import options_from_json_value


def test_production_defaults_match_upstream_example() -> None:
  # Literal values from ProtonPottsMPNN/inference/design_ph.py lines 66-95.
  cfg = PHDesignConfig()
  assert cfg.method == "block_descent"
  assert cfg.backend == "potts"
  assert cfg.temperature == 0.05
  assert cfg.samples_per_site == 2
  assert cfg.block_size == 3
  assert cfg.combined_lambda == 0.3
  assert cfg.seed_source == "native"
  assert cfg.center_types == ("HIS-P", "ASP-P", "GLU-P")
  assert cfg.dep_map == (("HIS-P", ("HIS-S",)), ("ASP-P", ("ASP-D",)), ("GLU-P", ("GLU-D",)))
  assert cfg.forbidden_tokens == ("HIS-A", "ASP-A", "GLU-A", "UNK")
  assert cfg.placement_by == "scan_potts"
  assert cfg.placement_region == ("all",)
  assert cfg.repetitive_window_parents == ("ARG", "LYS", "HIS", "ASP", "GLU")
  assert cfg.repetitive_window_radius == 2
  assert cfg.repetitive_window_weight == 1.0
  assert cfg.neighbour_k == 16
  assert cfg.max_mutations == 20
  assert cfg.record_trajectory is True
  # Remaining fields keep the upstream dataclass defaults.
  assert cfg.global_weight == 0.0
  assert cfg.zscale_mode == "block"
  assert cfg.adjacent_repeat_weight == 0.0
  assert cfg.self_weight == 1.0
  assert cfg.block_max_rounds == 10
  assert cfg.sweep_order == "position"
  assert cfg.infill_scope == "neighbourhood"
  assert cfg.cv_patience == 3
  assert cfg.cv_max == 50
  assert cfg.center_count == 3  # set from center_types by validation
  assert cfg.binder_chain is None


def test_config_is_hashable() -> None:
  assert hash(PHDesignConfig(binder_chain="B")) == hash(PHDesignConfig(binder_chain="B"))


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"method": "not_a_method"}, "Unknown method"),
    ({"zscale_mode": "bogus"}, "zscale_mode"),
    ({"infill_scope": "bogus"}, "infill_scope"),
    ({"sweep_order": "bogus"}, "sweep_order"),
    ({"seed_source": "bogus"}, "seed_source"),
    ({"placement_region": ("nowhere",)}, "placement_region"),
    ({"self_weight": -0.1}, "self_weight"),
    ({"center_types": ("CYS-P",)}, "not in dep_map"),
    ({"center_types": (), "center_count": 0}, "center_count must be >= 1"),
    (
      {"center_types": (), "center_count": 3, "explicit_centers": ((1, "HIS-P"), (2, "ASP-P"))},
      "explicit_centers",
    ),
  ],
)
def test_upstream_validation_analogues_raise(kwargs: dict, match: str) -> None:
  with pytest.raises(ValueError, match=match):
    PHDesignConfig(**kwargs)


def test_centre_free_allowed_only_for_greedy_chain() -> None:
  cfg = PHDesignConfig(
    method="greedy_energy_block", infill_scope="chain", center_types=(), center_count=0
  )
  assert cfg.center_count == 0
  with pytest.raises(ValueError, match="center_count must be >= 1"):
    PHDesignConfig(method="block_descent", infill_scope="chain", center_types=(), center_count=0)
  with pytest.raises(ValueError, match="center_count must be >= 1"):
    PHDesignConfig(method="greedy_energy_block", infill_scope="neighbourhood", center_types=(), center_count=0)


def test_explicit_centers_match_count_is_accepted() -> None:
  cfg = PHDesignConfig(center_types=(), explicit_centers=[(1, "HIS-P"), (2, "ASP-P")], center_count=2)
  assert cfg.explicit_centers == ((1, "HIS-P"), (2, "ASP-P"))


@pytest.mark.parametrize(
  ("kwargs", "debt"),
  [
    ({"method": "converged_mcmc"}, "#2616"),
    ({"method": "two_phase"}, "#2616"),
    ({"method": "converged_mcmc_combined"}, "#2616"),
    ({"method": "autoregressive"}, "#2617"),
    ({"method": "mpnn_sample"}, "#2617"),
    ({"backend": "mpnn"}, "#2617"),
    ({"selective_source": "decoder"}, "#2617"),
    ({"placement_by": "scan_mpnn"}, "#2617"),
  ],
)
def test_deferred_methods_and_backends_refuse_with_debt_number(kwargs: dict, debt: str) -> None:
  with pytest.raises(ValueError, match=debt):
    PHDesignConfig(**kwargs)


def test_validate_for_design_requires_binder_chain() -> None:
  with pytest.raises(ValueError, match="binder_chain"):
    PHDesignConfig().validate_for_design()
  PHDesignConfig(binder_chain="B").validate_for_design()


def test_config_from_options_defaults_match_config() -> None:
  assert config_from_options(ProtonPottsOptions()) == PHDesignConfig(binder_chain=None)
  with pytest.raises(ValueError, match="binder_chain"):
    config_from_options(ProtonPottsOptions()).validate_for_design()


def test_options_json_round_trip_with_design_fields() -> None:
  options = ProtonPottsOptions(
    protonation_labels_json="labels.json",
    variants_json="variants.json",
    binder_chain="B",
    design_method="greedy_energy_block",
    block_size=2,
    combined_lambda=0.5,
    temperature=0.1,
    samples_per_site=1,
    center_types=(),
    explicit_centers=((12, "HIS-P"), (40, "GLU-P")),
    dep_map=(("HIS-P", ("HIS-S",)), ("GLU-P", ("GLU-D",))),
    forbidden_tokens=("UNK",),
    neighbour_k=0,
    max_mutations=5,
    infill_scope="chain",
    repetitive_window_weight=0.0,
    repetitive_window_radius=3,
    repetitive_window_parents=("ASP",),
    block_max_rounds=4,
    cv_patience=2,
    cv_max=9,
    record_trajectory=False,
  )
  payload = json.loads(json.dumps(dataclasses.asdict(options)))
  rebuilt = options_from_json_value(ProtonPottsOptions, payload)
  assert rebuilt == options
  assert rebuilt.protonation_labels_json == "labels.json"
  assert rebuilt.variants_json == "variants.json"
  assert rebuilt.explicit_centers == ((12, "HIS-P"), (40, "GLU-P"))
  cfg = config_from_options(rebuilt)
  assert cfg.binder_chain == "B"
  assert cfg.method == "greedy_energy_block"
  assert cfg.dep_map == (("HIS-P", ("HIS-S",)), ("GLU-P", ("GLU-D",)))
  assert cfg.explicit_centers == ((12, "HIS-P"), (40, "GLU-P"))


def test_options_accept_dict_dep_map() -> None:
  options = ProtonPottsOptions(dep_map={"HIS-P": ["HIS-S"]})  # type: ignore[arg-type]
  assert options.dep_map == (("HIS-P", ("HIS-S",)),)
  hash(options)


def test_centre_free_options_map_to_center_count_zero() -> None:
  options = ProtonPottsOptions(
    binder_chain="A", design_method="greedy_energy_block", infill_scope="chain", center_types=()
  )
  config = config_from_options(options)
  assert config.center_count == 0
  # centres named, or a non-chain scope, are not centre-free
  named = config_from_options(
    ProtonPottsOptions(binder_chain="A", design_method="greedy_energy_block", infill_scope="chain")
  )
  assert named.center_count == len(named.center_types)
