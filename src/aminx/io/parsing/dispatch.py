"""Parsing module - all parsing handled by proxide.

What these entry points actually support:
- ``parse_structure`` / ``parse_input``: single structure files (PDB, mmCIF).
- ``parse_trajectory``: an AMBER topology (parm7/prmtop) plus selected XTC frames, via
  proxide's ``parse_amber_trajectory``. A trajectory alone has no residue identities, so
  the topology is required.
- ``parse_protein`` / ``parse_mdcath`` / ``parse_mdtraj_h5``: aliases of ``parse_input``
  (single structure file); they do NOT read MD-CATH or MDTraj HDF5 datasets.

Legacy biotite/mdtraj-based parsers have been removed; do not add mdtraj back.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from proxide import OutputSpec
from proxide.core.containers import Protein


def _output_spec(k_neighbors: int, kwargs: dict[str, Any]) -> OutputSpec:
  """Build the aminx OutputSpec: RBF on, Atom37-compatible, optional physics/solvent flags."""
  compute_physics = kwargs.get("compute_physics", False)
  compute_vdw = kwargs.get("compute_vdw", compute_physics)
  compute_estat = kwargs.get("compute_electrostatics", compute_physics)

  spec_args = {
    "compute_rbf": True,
    "rbf_num_neighbors": k_neighbors,
    "compute_vdw": compute_vdw,
    "compute_electrostatics": compute_estat,
    "parameterize_md": compute_vdw or compute_estat,
  }
  for key in ("remove_solvent", "add_hydrogens", "force_field"):
    if key in kwargs:
      spec_args[key] = kwargs[key]
  return OutputSpec(**spec_args)


def parse_structure(
  file_path: str | Path,
  k_neighbors: int = 48,
  **kwargs: Any,  # noqa: ANN401
) -> Protein:
  """Parse structure using proxide with customized OutputSpec.

  This wrapper enables RBF compilation and ensures Atom37-compatible
  Atom ordering.
  """
  spec = _output_spec(k_neighbors, kwargs)

  # Prefer legacy rust module when present, then proxide's current backend path,
  # then older top-level parsing export.
  _parse_structure = None
  try:
    from proxide.io.parsing.rust import parse_structure as _parse_structure  # noqa: PLC0415
  except (ModuleNotFoundError, ImportError):
    pass

  if _parse_structure is None:
    try:
      from proxide.io.parsing.backend import parse_structure as _parse_structure  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError):
      pass

  if _parse_structure is None:
    try:
      from proxide.io.parsing import parse_structure as _parse_structure  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError) as exc:
      msg = (
        "No proxide parsing backend available. Install proxide with parsing support "
        "or provide a runtime that includes either proxide.io.parsing.backend "
        "(preferred) or proxide.io.parsing.rust."
      )
      raise ModuleNotFoundError(msg) from exc

  return _parse_structure(file_path, spec=spec)


def parse_trajectory(
  topology: str | Path,
  trajectory: str | Path,
  frames: Sequence[int] | None = None,
  k_neighbors: int = 48,
  **kwargs: Any,  # noqa: ANN401
) -> Iterator[Protein]:
  """Load selected frames of an MD trajectory, with residue identities from its topology.

  Args:
    topology: AMBER ``.parm7``/``.prmtop`` file (atom names, residues, chains).
    trajectory: ``.xtc`` trajectory whose atom count matches the topology.
    frames: Frame indices to load (negative counts from the end). ``None`` loads frame 0
      only; loading a whole trajectory must be requested explicitly.
    k_neighbors: Neighbours for the RBF features, as in :func:`parse_structure`.
    **kwargs: The same physics/solvent flags :func:`parse_structure` accepts.

  Yields:
    One ``Protein`` per requested frame, in the order given (the same iterator shape as
    :func:`parse_input`). Each frame is parsed on its own, so its features belong to it.
    Water and ions are excluded; AMBER variants (``HIE``, ``CYX``, ...) map to their
    parent amino acids; chains follow the topology's molecules (disulfides excluded).

  Raises:
    ImportError: If the installed proxide predates ``parse_amber_trajectory``.
  """
  # Resolved eagerly so a too-old proxide fails at the call, not at first iteration.
  try:
    from proxide.io.parsing.backend import parse_amber_trajectory  # noqa: PLC0415
  except ImportError as exc:
    import proxide  # noqa: PLC0415

    # proxide's version string is not bumped between builds, so name the capability,
    # not a version number.
    msg = (
      "aminx.io.parsing.parse_trajectory needs proxide.io.parsing.backend."
      f"parse_amber_trajectory, which the installed proxide ({proxide.__file__}) lacks. "
      "Install a proxide build that includes the AMBER parm7 reader "
      "(proxide branch feat/parm7-amber-trajectory)."
    )
    raise ImportError(msg) from exc

  spec = _output_spec(k_neighbors, kwargs)
  selected = [0] if frames is None else [int(f) for f in frames]
  return iter(parse_amber_trajectory(topology, trajectory, frames=selected, spec=spec))


def parse_input(file_path: str | Path, **kwargs: Any) -> Iterator[Protein]:  # noqa: ANN401
  """Unified entry point that defaults to our configured parse_structure."""
  # For now, just route to parse_structure which returns a single Protein
  # The original parse_input yielded generator, so to maintain compat
  # we yield
  yield parse_structure(file_path, **kwargs)


# Aliases for specific formats
parse_protein = parse_input
parse_mdcath = parse_input
parse_mdtraj_h5 = parse_input

__all__ = [
  "parse_input",
  "parse_mdcath",
  "parse_mdtraj_h5",
  "parse_protein",
  "parse_structure",
  "parse_trajectory",
]
