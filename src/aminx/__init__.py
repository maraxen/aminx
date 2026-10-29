"""Aminx: a functional interface for ProteinMPNN.

Importing this package does not configure the multiprocessing start method. Call
``configure_multiprocessing()`` once at process start in notebooks, standalone scripts,
or any entrypoint that uses ``multiprocessing`` worker pools (the campaign CLI already
does this); see ``aminx.runtime``.

Submodules:
  - ``aminx.potts``: Potts model family with TRW inference and Gibbs sampling
    (see ``aminx.potts.__init__.py`` for PottsModel, PoeModel, etc.)
"""

from .io.parsing import parse_structure
from .io.weights import load_model
from .run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)
from .runtime import configure_multiprocessing
from .sampling import make_conditional_logits_fn, make_sample_sequences, sample
from .scoring import make_score_fn, score

try:
  from importlib.metadata import PackageNotFoundError as _PkgNFE
  from importlib.metadata import version as _meta_version

  __version__ = _meta_version("aminx")
except _PkgNFE:
  # Build-time fallback: package metadata not yet installed (e.g. editable install
  # race or direct source import before `pip install`).  The release tooling writes
  # the canonical version into pyproject.toml; callers that need the authoritative
  # string should always prefer `importlib.metadata.version("aminx")` directly.
  __version__ = "0.1.0"
__author__ = "Marielle Russo"
__description__ = "Aminx: A functional interface for ProteinMPNN"
__license__ = "MIT"
__url__ = "https://github.com/maraxen/aminx"

# TODO(tech-debt): `.agents/TECHNICAL_DEBT.md` §13 — repository hygiene (stale docs, dead code, CI warning budget).

__all__ = [
  "InspectionSpecification",
  "JacobianSpecification",
  "RunSpecification",
  "SamplingSpecification",
  "ScoringSpecification",
  "configure_multiprocessing",
  "load_model",
  "make_conditional_logits_fn",
  "make_sample_sequences",
  "make_score_fn",
  "parse_structure",
  "sample",
  "score",
]
