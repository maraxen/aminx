"""Run model pipelines."""

# Jacobian functionality temporarily disabled during Equinox migration
# Will be re-enabled after refactoring conditional_logits module
from aminx.host.sampling_driver import SamplingDriver

from ._exports import sample, score
from ._runspec_coverage import (
  RunSpecCoverageError,
  assert_runspec_coverage_is_exhaustive,
)
from .run_spec_portable_json import (
  PORTABLE_RUN_SPEC_VERSION,
  run_spec_portable_from_dict,
  run_spec_portable_to_dict,
)
from .spec import RunSpec, build_run_spec, topology_hash
from .spec_json import (
  SpecJSONDecodeError,
  SpecJSONEncodeError,
  run_specification_from_json,
  run_specification_from_json_dict,
  run_specification_to_json,
  run_specification_to_json_dict,
)
from .specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

__all__ = [
  "PORTABLE_RUN_SPEC_VERSION",
  "InspectionSpecification",
  "JacobianSpecification",
  "RunSpec",
  "RunSpecCoverageError",
  "RunSpecification",
  "SamplingDriver",
  "SamplingSpecification",
  "ScoringSpecification",
  "SpecJSONDecodeError",
  "SpecJSONEncodeError",
  "assert_runspec_coverage_is_exhaustive",
  "build_run_spec",
  "run_spec_portable_from_dict",
  "run_spec_portable_to_dict",
  "run_specification_from_json",
  "run_specification_from_json_dict",
  "run_specification_to_json",
  "run_specification_to_json_dict",
  "sample",
  "score",
  "topology_hash",
]
