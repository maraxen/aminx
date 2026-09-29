"""Command-line interface (Typer) for Aminx."""

from __future__ import annotations

import dataclasses
import glob as _glob_module
import json
import os
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

from aminx.io.input_uri import (
  get_dedup_key,
  parse_input_uri,
)
from aminx.io.proxide_fetch import (
  InputResolutionError,
  fetch_afdb,
  fetch_md_cath,
  fetch_pdb,
)
from aminx.run.options import LaserOptions, PottsMPNNOptions
from aminx.run.run_spec_portable_json import (
  run_spec_portable_from_dict,
  run_spec_portable_to_dict,
)
from aminx.run.spec_json import (
  options_from_json_value,
  run_specification_from_json,
  run_specification_to_json,
  run_specification_to_json_dict,
)

app = typer.Typer(
  name="aminx",
  help="Aminx CLI — configuration and run helpers.",
  no_args_is_help=True,
)

spec_app = typer.Typer(
  name="spec",
  help="Run specification JSON (see aminx.run.spec_json).",
  no_args_is_help=True,
)
app.add_typer(spec_app, name="spec")

run_app = typer.Typer(
  name="run",
  help="Build and dispatch run specifications.",
  no_args_is_help=True,
)
app.add_typer(run_app, name="run")

campaign_app = typer.Typer(
  name="campaign",
  help="Campaign manifest planner and worker operations.",
  no_args_is_help=True,
)
app.add_typer(campaign_app, name="campaign")

# potts sub-app registered at end of file (after other imports)


class LockBackend(StrEnum):
  """Lock backend choices for campaign worker/run commands."""

  local_fs = "local_fs"
  distributed = "distributed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
  """Parse a comma-separated float string into a tuple, or return None."""
  if value is None:
    return None
  return tuple(float(x.strip()) for x in value.split(","))


def _parse_options_json(raw: str | None) -> Any:
  if raw is None:
    return None
  return json.loads(raw)


def _sampling_temperature_kwargs(temperature: str | None) -> dict[str, tuple[float, ...]]:
  """Omit the temperature kwarg when the CLI flag was left unset.

  ``SamplingSpecification.temperature`` then stays ``UNSET`` and the family
  default resolves in ``__post_init__`` (``(0.1,)`` for MPNN families).
  """
  parsed = _parse_float_tuple(temperature)
  if not parsed:
    return {}
  return {"temperature": parsed}


_SCORE_SEQUENCE_KINDS = frozenset({"nll", "logits"})


def _require_sequences_for_output_kind(output_kind: str, sequences: list[str]) -> None:
  if output_kind in _SCORE_SEQUENCE_KINDS and not sequences:
    typer.echo("--sequences-to-score is required", err=True)
    raise typer.Exit(code=2)


def _parse_tied_positions(
  tied_position: list[str],
) -> list[tuple[int, int]] | str | None:
  """Parse --tied-position values into a list of (int, int) pairs or a mode string."""
  if not tied_position:
    return None
  if len(tied_position) == 1 and tied_position[0] in ("auto", "direct"):
    return tied_position[0]
  pairs: list[tuple[int, int]] = []
  for token in tied_position:
    parts = token.split(":")
    if len(parts) != 2:
      msg = f"--tied-position value {token!r} must be 'A:B' (two integers) or 'auto'/'direct'"
      raise typer.BadParameter(msg)
    pairs.append((int(parts[0]), int(parts[1])))
  return pairs


_STRUCTURE_EXTS = frozenset({".pdb", ".cif"})
_GLOB_CHARS = frozenset("*?[")


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
  """Resolve cache directory with precedence: arg -> AMINX_CACHE_DIR -> XDG default.

  Args:
    cache_dir: Explicit cache dir from CLI arg (highest priority).

  Returns:
    Resolved cache directory Path.
  """
  if cache_dir is not None:
    return Path(cache_dir)

  # Check AMINX_CACHE_DIR env var
  env_cache = os.environ.get("AMINX_CACHE_DIR")
  if env_cache:
    return Path(env_cache)

  # XDG-respecting default: $XDG_CACHE_HOME/aminx/inputs or ~/.cache/aminx/inputs
  xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
  if xdg_cache_home:
    return Path(xdg_cache_home) / "aminx" / "inputs"

  return Path.home() / ".cache" / "aminx" / "inputs"


def resolve_inputs(
  inputs: list[str],
  *,
  input_type: str = "auto",
  cache_dir: Path | str | None = None,
  fail_fast: bool = True,
) -> list[str]:
  """Resolve URI and local path entries to local file paths.

  Dispatches based on URI scheme (pdb://, afdb://, mdcath://, file://, bare local path).
  Fetches remote sources to cache_dir with pre-resolution deduplication. Local paths are
  expanded via the existing dir/glob logic. Resolves local paths first-seen order,
  respecting current _expand_inputs semantics.

  Args:
    inputs: Raw list of path/directory/glob/URI strings from CLI.
    input_type: Override scheme detection: 'auto' (default, use scheme), 'file' (force all local),
      or 'pdb'/'afdb'/'mdcath' (apply to schemeless tokens only).
    cache_dir: Explicit cache directory. Defaults to precedence: AMINX_CACHE_DIR env
      -> XDG_CACHE_HOME/aminx/inputs or ~/.cache/aminx/inputs.
    fail_fast: If True, re-raise on first fetch/resolution failure. If False, warn and skip.

  Returns:
    Flat, deduped list of local path strings (mixed from local and fetched sources).

  Raises:
    typer.BadParameter: For invalid URI syntax, unknown scheme, or input_type conflicts.
    InputResolutionError: On fetch failure (wrapped from proxide), cache I/O error, or offline.
    typer.Exit(code=1): On empty final result or fail_fast mode error.
  """
  # Resolve the cache directory once
  resolved_cache_dir = _resolve_cache_dir(cache_dir)

  # Step 1: Pre-fetch deduplication by (scheme, accession, fmt)
  deduped_entries = _dedup_input_entries(inputs)

  # Step 2: Classify and dispatch each deduped entry
  seen_paths: dict[str, None] = {}  # ordered set of resolved paths

  for entry in deduped_entries:
    _dispatch_and_resolve_entry(
      entry,
      input_type,
      resolved_cache_dir,
      seen_paths,
      fail_fast,
    )

  result = list(seen_paths.keys())
  if not result:
    typer.echo("--inputs: no valid structure files found after resolution", err=True)
    raise typer.Exit(code=1)

  return result


def _dedup_input_entries(inputs: list[str]) -> list[str]:
  """Deduplicate input entries by (scheme, accession, fmt) before resolution (M2).

  Args:
    inputs: Raw list of input entries.

  Returns:
    Deduplicated list preserving first-seen order.
  """
  seen_entries: dict[tuple[str, str, str | None], str] = {}
  deduped: list[str] = []

  for entry in inputs:
    parsed = parse_input_uri(entry)
    dedup_key = get_dedup_key(parsed)

    if dedup_key not in seen_entries:
      seen_entries[dedup_key] = entry
      deduped.append(entry)

  return deduped


def _dispatch_and_resolve_entry(
  entry: str,
  input_type: str,
  cache_dir: Path,
  seen_paths: dict[str, None],
  fail_fast: bool,
) -> None:
  """Dispatch and resolve a single entry (remote fetch or local expansion).

  Args:
    entry: Single input entry to resolve.
    input_type: Input type flag ('auto', 'file', 'pdb', 'afdb', 'mdcath').
    cache_dir: Resolved cache directory.
    seen_paths: Dict to accumulate resolved paths (modified in-place).
    fail_fast: If True, re-raise on error; if False, warn and skip.

  Raises:
    typer.BadParameter: On invalid input_type/scheme conflict.
    InputResolutionError: On fetch failure (if fail_fast=True).
    typer.Exit: On local expansion failure (if fail_fast=True).
  """
  parsed = parse_input_uri(entry)
  scheme = parsed.scheme

  # Apply input_type override
  if input_type == "file":
    scheme = "local"
  elif input_type in ("pdb", "afdb", "mdcath"):
    if parsed.scheme not in ("local", input_type):
      msg = (
        f"--input-type {input_type} conflicts with explicit scheme {parsed.scheme}:// in {entry!r}"
      )
      raise typer.BadParameter(msg)
    scheme = input_type

  try:
    if scheme == "local":
      _resolve_and_add_local_paths(entry, seen_paths, fail_fast)
    elif scheme == "file":
      _resolve_and_add_local_paths(parsed.accession, seen_paths, fail_fast)
    elif scheme == "pdb":
      path = fetch_pdb(parsed.accession, cache_dir, fmt=parsed.fmt or "mmcif")
      seen_paths.setdefault(str(path), None)
    elif scheme == "afdb":
      path = fetch_afdb(parsed.accession, cache_dir)
      seen_paths.setdefault(str(path), None)
    elif scheme == "mdcath":
      path = fetch_md_cath(parsed.accession, cache_dir)
      seen_paths.setdefault(str(path), None)
    else:
      msg = f"unknown scheme {scheme}:// (internal error)"
      raise typer.BadParameter(msg)
  except InputResolutionError:
    if fail_fast:
      raise
    warnings.warn(f"--inputs: skipping {entry!r} due to fetch error", stacklevel=2)
  except typer.Exit:
    raise


def _resolve_and_add_local_paths(
  entry: str,
  seen_paths: dict[str, None],
  fail_fast: bool,
) -> None:
  """Expand a local path entry and add to seen_paths (internal helper).

  Handles directories (iterdir for structure files), globs, and concrete file paths.
  Mirrors the existing _expand_inputs logic exactly.

  Args:
    entry: Local path, directory, or glob pattern.
    seen_paths: Dict to accumulate resolved paths (modified in-place).
    fail_fast: If True, exit on empty dir/glob; if False, warn and skip.

  Raises:
    typer.Exit(code=1): If fail_fast and no matches found.
  """
  p = Path(entry)

  if p.is_dir():
    # Expand all structure files directly inside the directory (non-recursive, sorted)
    found: list[str] = [
      str(candidate)
      for candidate in sorted(p.iterdir())
      if candidate.suffix.lower() in _STRUCTURE_EXTS
    ]
    if not found:
      msg = f"--inputs: directory {entry!r} contains no .pdb or .cif files"
      if fail_fast:
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)
      warnings.warn(msg, stacklevel=2)
      return
    for f in found:
      seen_paths.setdefault(f, None)

  elif any(c in entry for c in _GLOB_CHARS):
    # Glob expansion — keep only structure files
    matches = sorted(
      m
      for m in _glob_module.glob(entry)  # noqa: PTH207
      if Path(m).suffix.lower() in _STRUCTURE_EXTS
    )
    if not matches:
      msg = f"--inputs: glob {entry!r} matched no .pdb or .cif files"
      if fail_fast:
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)
      warnings.warn(msg, stacklevel=2)
      return
    for f in matches:
      seen_paths.setdefault(f, None)

  else:
    # Concrete path — pass through unchanged (backward-compatible behaviour).
    # Existence is not checked here; the downstream runner validates files.
    seen_paths.setdefault(entry, None)


def _expand_inputs(inputs: list[str], *, fail_fast: bool = False) -> list[str]:
  """Expand --inputs entries to a flat, deduped list of concrete structure file paths.

  BACKWARD-COMPATIBLE SHIM: This function is now a wrapper around resolve_inputs()
  to preserve the existing behavior for local paths only (local input_type).
  The resolver handles remote URIs and cache management; this shim ensures existing
  call sites continue to work unchanged.

  Each entry is expanded as follows:
  - Directory: all ``*.pdb`` / ``*.cif`` files directly inside it (case-insensitive, sorted).
  - Glob (contains ``*``, ``?``, or ``[``): expanded via glob, keeping only ``.pdb``/``.cif`` matches.
  - Otherwise: treated as a concrete file path and passed through unchanged.

  Deduplication preserves first-seen order.

  Args:
    inputs: Raw list of path/directory/glob strings from the CLI.
    fail_fast: If True, exit 1 on the first invalid/empty entry. If False (default),
      warn and skip non-existent / non-structure entries.

  Returns:
    Flat, deduped list of concrete file path strings.

  Raises:
    typer.Exit(code=1): If expansion is empty or (in fail_fast mode) on first bad entry.
  """
  # Delegate to resolve_inputs with input_type="file" (force all local, no scheme detection)
  # This preserves the byte-identical behavior of the original _expand_inputs.
  return resolve_inputs(inputs, input_type="file", fail_fast=fail_fast)


def _emit_or_run(
  spec: Any,
  emit_json: bool,
  out: Path | None,
  runner_name: str,
  runner_available: bool,
) -> None:
  """Either emit the spec as JSON or invoke the runner."""
  if emit_json:
    blob = json.dumps(run_specification_to_json_dict(spec), indent=2, sort_keys=True)
    if out is not None:
      out.write_text(blob, encoding="utf-8")
    else:
      typer.echo(blob)
    return
  if not runner_available:
    typer.echo(
      f"aminx run {runner_name} runner not wired; use --emit-json",
      err=True,
    )
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Shared base options — declared once on the ``run`` group callback.
#
# All four ``run`` subcommands (sample, score, jacobian, inspect) share the
# same ~38 model/infra parameters.  Rather than copy-pasting the declarations
# into each function signature, we collect them on the parent group via
# ``@run_app.callback()`` and store the result in ``ctx.obj`` as a
# ``_RunBase`` dataclass.  Subcommands receive ``ctx: typer.Context`` and
# unpack ``ctx.obj`` to access the shared fields.
#
# CLI interface is unchanged: base options still appear after ``run`` but
# before the subcommand name, e.g.:
#   aminx run --backbone-noise 0.1 sample --inputs foo.pdb --emit-json
# ---------------------------------------------------------------------------

_OPT = typer.Option


@dataclasses.dataclass
class _RunBase:
  """Carrier for the shared model/infra options parsed by the ``run`` callback."""

  topology: str | None
  model_weights: str
  model_version: str
  model_family: str | None
  checkpoint_id: str | None
  model_local_path: Path | None
  checkpoint_registry_path: Path | None
  ligand_mpnn_use_side_chain_context: bool | None
  batch_size: int | None
  backbone_noise: str
  backbone_noise_mode: str
  estat_noise: str | None
  estat_noise_mode: str
  vdw_noise: str | None
  vdw_noise_mode: str
  use_electrostatics: bool
  use_vdw: bool
  random_seed: int
  chain_id: str | None
  model: int | None
  altloc: str
  cache_path: Path | None
  output_dir: Path | None
  max_buffer_size: int | None
  overwrite_cache: bool
  max_length: int | None
  truncation_strategy: str
  host_resource_allocation_strategy: str
  ram_budget_mb: int | None
  max_workers: int | None
  n_devices: int | None
  use_preprocessed: bool
  preprocessed_index_path: Path | None
  split: str
  tied_position: list[str]
  pass_mode: str
  multi_state_temperature: float
  input_type: str
  input_cache_dir: Path | None
  potts_options_json: str | None = None
  laser_options_json: str | None = None


@run_app.callback()
def _run_base(
  ctx: typer.Context,
  topology: Annotated[str | None, _OPT(help="Topology file path")] = None,
  model_weights: Annotated[str, _OPT(help="Model weights name")] = "original",
  model_version: Annotated[str, _OPT(help="Model version")] = "v_48_020",
  model_family: Annotated[
    str | None,
    _OPT(
      help="Model family: proteinmpnn or ligandmpnn. Leave unset to auto-derive from "
      "checkpoint_id (RunSpecification.__post_init__); an explicit value here is always "
      "respected even if it disagrees with checkpoint_id.",
    ),
  ] = None,
  checkpoint_id: Annotated[str | None, _OPT(help="Checkpoint identifier")] = None,
  model_local_path: Annotated[Path | None, _OPT(help="Local model checkpoint path")] = None,
  checkpoint_registry_path: Annotated[Path | None, _OPT(help="Checkpoint registry path")] = None,
  ligand_mpnn_use_side_chain_context: Annotated[
    bool | None,
    _OPT(help="LigandMPNN side-chain context"),
  ] = None,
  batch_size: Annotated[int | None, _OPT(help="Batch size (None = subclass default)")] = None,
  backbone_noise: Annotated[str, _OPT(help="Comma-separated backbone noise levels")] = "0.0",
  backbone_noise_mode: Annotated[
    str,
    _OPT(help="Backbone noise mode: direct or thermal"),
  ] = "direct",
  estat_noise: Annotated[
    str | None,
    _OPT(help="Comma-separated electrostatic noise levels"),
  ] = None,
  estat_noise_mode: Annotated[str, _OPT(help="Electrostatic noise mode")] = "direct",
  vdw_noise: Annotated[str | None, _OPT(help="Comma-separated vdw noise levels")] = None,
  vdw_noise_mode: Annotated[str, _OPT(help="VDW noise mode")] = "direct",
  use_electrostatics: Annotated[bool, _OPT(help="Enable electrostatics")] = False,
  use_vdw: Annotated[bool, _OPT(help="Enable VDW")] = False,
  random_seed: Annotated[int, _OPT(help="Random seed")] = 42,
  chain_id: Annotated[str | None, _OPT(help="Chain ID")] = None,
  model: Annotated[int | None, _OPT(help="Model index")] = None,
  altloc: Annotated[str, _OPT(help="Alternate location: first or all")] = "first",
  cache_path: Annotated[Path | None, _OPT(help="Cache path")] = None,
  output_dir: Annotated[Path | None, _OPT(help="Output directory")] = None,
  max_buffer_size: Annotated[int | None, _OPT(help="Max buffer size (bytes)")] = None,
  overwrite_cache: Annotated[bool, _OPT(help="Overwrite cache")] = False,
  max_length: Annotated[int | None, _OPT(help="Max sequence length")] = 512,
  truncation_strategy: Annotated[
    str,
    _OPT(help="Truncation strategy: none, random_crop, center_crop"),
  ] = "random_crop",
  host_resource_allocation_strategy: Annotated[
    str,
    _OPT(help="Resource allocation: auto or full"),
  ] = "auto",
  ram_budget_mb: Annotated[int | None, _OPT(help="RAM budget in MB")] = None,
  max_workers: Annotated[int | None, _OPT(help="Max data loading workers")] = None,
  n_devices: Annotated[int | None, _OPT(help="Device count override")] = None,
  use_preprocessed: Annotated[bool, _OPT(help="Use preprocessed data")] = False,
  preprocessed_index_path: Annotated[Path | None, _OPT(help="Preprocessed index path")] = None,
  split: Annotated[str, _OPT(help="Data split")] = "inference",
  tied_position: Annotated[
    list[str],
    _OPT("--tied-position", help="Tied position pair A:B (repeatable) or 'auto'/'direct'"),
  ] = [],  # noqa: B006
  pass_mode: Annotated[str, _OPT(help="Pass mode: inter or intra")] = "intra",
  multi_state_temperature: Annotated[float, _OPT(help="Multi-state temperature")] = 1.0,
  input_type: Annotated[
    str,
    _OPT(
      help=(
        "Override URI scheme detection: 'auto' (default, detect scheme per entry), "
        "'file' (treat all entries as local paths, suppress scheme detection), "
        "or 'pdb'/'afdb'/'mdcath' (apply to schemeless entries only, hard error on conflicting explicit scheme). "
        "Accepted URI forms: bare path (e.g., /data/1ubq.pdb), "
        "file:///path, pdb://<RCSB-id>, afdb://<UniProt-id>, mdcath://<id>. "
        "Remote sources are fetched to --cache-dir at CLI time; compute nodes remain offline-safe."
      ),
    ),
  ] = "auto",
  input_cache_dir: Annotated[
    Path | None,
    _OPT(
      help=(
        "Cache directory for remotely-fetched structures. "
        "Precedence: --cache-dir (explicit), AMINX_CACHE_DIR (env), "
        "$XDG_CACHE_HOME/aminx/inputs, or ~/.cache/aminx/inputs (XDG default). "
        "Each remote URI is fetched to a per-accession subdir and cached for reuse. "
        "Cache hits skip network access entirely (cluster compute nodes remain offline-safe)."
      ),
    ),
  ] = None,
  potts_options_json: Annotated[
    str | None,
    _OPT("--potts-options-json", help="JSON object of PottsMPNNOptions fields"),
  ] = None,
  laser_options_json: Annotated[
    str | None,
    _OPT("--laser-options-json", help="JSON object of LaserOptions fields"),
  ] = None,
) -> None:
  """Build and dispatch run specifications.

  Base model/infra options are shared across all subcommands and must be
  placed before the subcommand name, e.g.::

      aminx run --backbone-noise 0.1,0.2 sample --inputs foo.pdb --emit-json
  """
  ctx.obj = _RunBase(
    topology=topology,
    model_weights=model_weights,
    model_version=model_version,
    model_family=model_family,
    checkpoint_id=checkpoint_id,
    model_local_path=model_local_path,
    checkpoint_registry_path=checkpoint_registry_path,
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    batch_size=batch_size,
    backbone_noise=backbone_noise,
    backbone_noise_mode=backbone_noise_mode,
    estat_noise=estat_noise,
    estat_noise_mode=estat_noise_mode,
    vdw_noise=vdw_noise,
    vdw_noise_mode=vdw_noise_mode,
    use_electrostatics=use_electrostatics,
    use_vdw=use_vdw,
    random_seed=random_seed,
    chain_id=chain_id,
    model=model,
    altloc=altloc,
    cache_path=cache_path,
    output_dir=output_dir,
    max_buffer_size=max_buffer_size,
    overwrite_cache=overwrite_cache,
    max_length=max_length,
    truncation_strategy=truncation_strategy,
    host_resource_allocation_strategy=host_resource_allocation_strategy,
    ram_budget_mb=ram_budget_mb,
    max_workers=max_workers,
    n_devices=n_devices,
    use_preprocessed=use_preprocessed,
    preprocessed_index_path=preprocessed_index_path,
    split=split,
    tied_position=list(tied_position),
    pass_mode=pass_mode,
    multi_state_temperature=multi_state_temperature,
    input_type=input_type,
    input_cache_dir=input_cache_dir,
    potts_options_json=potts_options_json,
    laser_options_json=laser_options_json,
  )


def _base_spec_kwargs(b: _RunBase) -> dict[str, Any]:
  """Translate a ``_RunBase`` into the kwargs shared by all spec constructors."""
  backbone_noise_parsed = _parse_float_tuple(b.backbone_noise)
  estat_noise_parsed = _parse_float_tuple(b.estat_noise)
  vdw_noise_parsed = _parse_float_tuple(b.vdw_noise)
  tied_positions_parsed = _parse_tied_positions(b.tied_position)
  return {
    "topology": b.topology,
    "model_weights": b.model_weights,
    "model_version": b.model_version,
    "model_family": b.model_family,
    "checkpoint_id": b.checkpoint_id,
    "model_local_path": b.model_local_path,
    "checkpoint_registry_path": b.checkpoint_registry_path,
    "ligand_mpnn_use_side_chain_context": b.ligand_mpnn_use_side_chain_context,
    **({} if b.batch_size is None else {"batch_size": b.batch_size}),
    "backbone_noise": backbone_noise_parsed or (0.0,),
    "backbone_noise_mode": b.backbone_noise_mode,
    "estat_noise": estat_noise_parsed,
    "estat_noise_mode": b.estat_noise_mode,
    "vdw_noise": vdw_noise_parsed,
    "vdw_noise_mode": b.vdw_noise_mode,
    "use_electrostatics": b.use_electrostatics,
    "use_vdw": b.use_vdw,
    "random_seed": b.random_seed,
    "chain_id": b.chain_id,
    "model": b.model,
    "altloc": b.altloc,
    "cache_path": b.cache_path,
    "output_dir": b.output_dir,
    "max_buffer_size": b.max_buffer_size,
    "overwrite_cache": b.overwrite_cache,
    "max_length": b.max_length,
    "truncation_strategy": b.truncation_strategy,
    "host_resource_allocation_strategy": b.host_resource_allocation_strategy,
    "ram_budget_mb": b.ram_budget_mb,
    "max_workers": b.max_workers,
    "n_devices": b.n_devices,
    "use_preprocessed": b.use_preprocessed,
    "preprocessed_index_path": b.preprocessed_index_path,
    "split": b.split,
    "tied_positions": tied_positions_parsed,
    "pass_mode": b.pass_mode,
    "multi_state_temperature": b.multi_state_temperature,
    "potts_mpnn": options_from_json_value(
      PottsMPNNOptions,
      _parse_options_json(b.potts_options_json),
    ),
    "laser": options_from_json_value(LaserOptions, _parse_options_json(b.laser_options_json)),
  }


# ---------------------------------------------------------------------------
# run sample
# ---------------------------------------------------------------------------


@run_app.command("sample")
def run_sample(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  emit_json: Annotated[
    bool,
    _OPT("--emit-json", help="Write spec JSON to stdout or --out; exit 0"),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON file (only with --emit-json)")] = None,
  # Sample-specific
  num_samples: Annotated[int, _OPT(help="Number of samples")] = 1,
  sampling_strategy: Annotated[
    str,
    _OPT(help="Sampling strategy: temperature or straight_through"),
  ] = "temperature",
  temperature: Annotated[
    str | None,
    _OPT(help="Comma-separated temperature values; omit to use the family default"),
  ] = None,
  use_unified_driver: Annotated[bool, _OPT(help="Use unified driver")] = True,
  iterations: Annotated[int | None, _OPT(help="Straight-through iterations")] = None,
  learning_rate: Annotated[float | None, _OPT(help="Straight-through learning rate")] = None,
  use_concrete: Annotated[bool, _OPT(help="Use concrete relaxation")] = False,
  concrete_tau_start: Annotated[float, _OPT(help="Concrete tau start")] = 1.0,
  concrete_tau_end: Annotated[float, _OPT(help="Concrete tau end")] = 0.1,
  output_h5_path: Annotated[Path | None, _OPT(help="Zarr output path")] = None,
  return_logits: Annotated[bool, _OPT(help="Return logits")] = True,
  samples_batch_size: Annotated[int, _OPT(help="Samples batch size")] = 16,
  samples_chunk_size: Annotated[int | None, _OPT(help="Samples chunk size")] = None,
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 1,
  temperature_batch_size: Annotated[int, _OPT(help="Temperature batch size")] = 1,
  average_node_features: Annotated[bool, _OPT(help="Average node features")] = False,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  multi_state_strategy: Annotated[
    str,
    _OPT(help="Multi-state aggregation strategy"),
  ] = "arithmetic_mean",
  compute_pseudo_perplexity: Annotated[
    bool,
    _OPT(help="Compute pseudo-perplexity (requires return-logits)"),
  ] = False,
  ligand_conditioning: Annotated[bool, _OPT(help="Ligand conditioning")] = False,
  sidechain_conditioning: Annotated[bool, _OPT(help="Sidechain conditioning")] = False,
  campaign_mode: Annotated[bool, _OPT(help="Campaign mode")] = False,
  allow_logits_in_campaign: Annotated[bool, _OPT(help="Allow logits in campaign")] = False,
  logits_memory_budget_mb: Annotated[int | None, _OPT(help="Logits memory budget (MB)")] = None,
  ligand_context_path: Annotated[Path | None, _OPT(help="Ligand context path")] = None,
  grid_mode: Annotated[bool, _OPT(help="Grid mode")] = False,
  job_id: Annotated[str | None, _OPT(help="Grid job ID")] = None,
  chunk_id: Annotated[int | None, _OPT(help="Grid chunk ID")] = None,
  sample_start: Annotated[int | None, _OPT(help="Grid sample start")] = None,
  sample_count: Annotated[int | None, _OPT(help="Grid sample count")] = None,
) -> None:
  """Sample sequences from the model (non-serializable fields: bias, fixed_positions, fixed_mask, fixed_tokens, state_weights, decode_fn)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  try:
    spec = SamplingSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      num_samples=num_samples,
      sampling_strategy=sampling_strategy,  # type: ignore[arg-type]
      **_sampling_temperature_kwargs(temperature),
      use_unified_driver=use_unified_driver,
      iterations=iterations,
      learning_rate=learning_rate,
      use_concrete=use_concrete,
      concrete_tau_start=concrete_tau_start,
      concrete_tau_end=concrete_tau_end,
      output_h5_path=output_h5_path,
      return_logits=return_logits,
      samples_batch_size=samples_batch_size,
      samples_chunk_size=samples_chunk_size,
      noise_batch_size=noise_batch_size,
      temperature_batch_size=temperature_batch_size,
      average_node_features=average_node_features,
      multi_state_strategy=multi_state_strategy,  # type: ignore[arg-type]
      compute_pseudo_perplexity=compute_pseudo_perplexity,
      ligand_conditioning=ligand_conditioning,
      sidechain_conditioning=sidechain_conditioning,
      campaign_mode=campaign_mode,
      allow_logits_in_campaign=allow_logits_in_campaign,
      logits_memory_budget_mb=logits_memory_budget_mb,
      ligand_context_path=ligand_context_path,
      grid_mode=grid_mode,
      job_id=job_id,
      chunk_id=chunk_id,
      sample_start=sample_start,
      sample_count=sample_count,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_or_run(spec, emit_json, out, "sample", runner_available=True)
  if not emit_json:
    from aminx.host.runner import sample  # noqa: PLC0415

    sample(spec)


# ---------------------------------------------------------------------------
# run score
# ---------------------------------------------------------------------------


@run_app.command("score")
def run_score(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  sequences_to_score: Annotated[
    list[str],
    _OPT(
      "--sequences-to-score",
      help="Amino acid sequences to score (repeatable; required for nll and logits)",
    ),
  ] = [],  # noqa: B006
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  emit_json: Annotated[
    bool,
    _OPT("--emit-json", help="Write spec JSON to stdout or --out; exit 0"),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON file (only with --emit-json)")] = None,
  # Score-specific
  output_kind: Annotated[
    str,
    _OPT(
      "--output-kind",
      help="nll, logits, energy, ddg, proofread_unconditional, or proofread_conditional",
    ),
  ] = "nll",
  temperature: Annotated[str, _OPT(help="Comma-separated temperature values")] = "1.0",
  return_logits: Annotated[bool, _OPT(help="Return logits")] = False,
  return_decoding_orders: Annotated[bool, _OPT(help="Return decoding orders")] = False,
  return_all_scores: Annotated[bool, _OPT(help="Return all scores")] = False,
  output_h5_path: Annotated[Path | None, _OPT(help="HDF5 output path")] = None,
  average_node_features: Annotated[bool, _OPT(help="Average node features")] = False,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 4,
  multi_state_strategy: Annotated[
    str,
    _OPT(help="Multi-state aggregation strategy"),
  ] = "arithmetic_mean",
) -> None:
  """Score sequences against structure inputs (non-serializable fields: conformational_states, decoding_order_fn)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )
  _require_sequences_for_output_kind(output_kind, sequences_to_score)

  b: _RunBase = ctx.obj
  temperature_parsed = _parse_float_tuple(temperature)

  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  try:
    spec = ScoringSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      sequences_to_score=sequences_to_score,
      output_kind=cast(
        "Literal['nll', 'logits', 'energy', 'ddg', 'proofread_unconditional', 'proofread_conditional']",
        output_kind,
      ),
      temperature=temperature_parsed[0]
      if temperature_parsed and len(temperature_parsed) == 1
      else (temperature_parsed[0] if temperature_parsed else 1.0),
      return_logits=return_logits,
      return_decoding_orders=return_decoding_orders,
      return_all_scores=return_all_scores,
      output_h5_path=output_h5_path,
      average_node_features=average_node_features,
      noise_batch_size=noise_batch_size,
      multi_state_strategy=multi_state_strategy,  # type: ignore[arg-type]
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_or_run(spec, emit_json, out, "score", runner_available=True)
  if not emit_json:
    from aminx.host.runner import score  # noqa: PLC0415

    score(spec)


# ---------------------------------------------------------------------------
# run jacobian
# ---------------------------------------------------------------------------


@run_app.command("jacobian")
def run_jacobian(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  emit_json: Annotated[
    bool,
    _OPT("--emit-json", help="Write spec JSON to stdout or --out; exit 0"),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON file (only with --emit-json)")] = None,
  # Jacobian-specific
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 1,
  jacobian_batch_size: Annotated[int, _OPT(help="Jacobian batch size")] = 16,
  average_encodings: Annotated[bool, _OPT(help="Average encodings")] = True,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  combine: Annotated[bool, _OPT(help="Combine Jacobians")] = False,
  combine_batch_size: Annotated[int, _OPT(help="Combine batch size")] = 8,
  output_h5_path: Annotated[
    Path | None,
    _OPT(help="Large-tensor output path (Zarr store)"),
  ] = None,
  jacobian_mode: Annotated[
    str,
    _OPT(help="categorical: full CatJac (L,21,L,21); reverse: fast score gradient (L,21)"),
  ] = "categorical",
  compute_apc: Annotated[bool, _OPT(help="Compute APC correction (categorical mode only)")] = True,
  apc_batch_size: Annotated[int, _OPT(help="APC batch size")] = 8,
  apc_residue_batch_size: Annotated[int, _OPT(help="APC residue batch size")] = 1000,
) -> None:
  """Compute Jacobians (categorical CatJac or reverse-mode score gradient)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )

  from aminx.run.specs import JacobianSpecification  # noqa: PLC0415

  try:
    spec = JacobianSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      noise_batch_size=noise_batch_size,
      jacobian_batch_size=jacobian_batch_size,
      average_encodings=average_encodings,
      combine=combine,
      combine_batch_size=combine_batch_size,
      output_h5_path=output_h5_path,
      jacobian_mode=jacobian_mode,  # type: ignore[arg-type]
      compute_apc=compute_apc,
      apc_batch_size=apc_batch_size,
      apc_residue_batch_size=apc_residue_batch_size,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_or_run(spec, emit_json, out, "jacobian", runner_available=True)
  if not emit_json:
    from aminx.host.runner import jacobian  # noqa: PLC0415

    jacobian(spec)


# ---------------------------------------------------------------------------
# run inspect
# ---------------------------------------------------------------------------


@run_app.command("inspect")
def run_inspect(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  emit_json: Annotated[
    bool,
    _OPT("--emit-json", help="Write spec JSON to stdout or --out; exit 0"),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON file (only with --emit-json)")] = None,
  # Inspect-specific
  output_h5_path: Annotated[Path | None, _OPT(help="HDF5 output path")] = None,
  inspection_features: Annotated[
    list[str],
    _OPT(
      "--inspection-features",
      help="Features to inspect: unconditional_logits, encoded_node_features, edge_features, decoded_node_features, conditional_logits",
    ),
  ] = ["unconditional_logits"],  # noqa: B006
  distance_matrix: Annotated[bool, _OPT(help="Compute distance matrix")] = False,
  distance_matrix_method: Annotated[
    str,
    _OPT(help="Distance matrix method: ca, cb, backbone_average, closest_atom"),
  ] = "ca",
  cross_input_similarity: Annotated[
    bool,
    _OPT(help="Compute cross-input similarity (requires >=2 inputs)"),
  ] = False,
  similarity_metric: Annotated[
    str,
    _OPT(help="Similarity metric: rmsd, tm-score, gdt_ts, gdt_ha, cosine"),
  ] = "rmsd",
) -> None:
  """Inspect model encodings and features (non-serializable fields: conformational_states)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )

  from aminx.run.specs import InspectionSpecification  # noqa: PLC0415

  try:
    spec = InspectionSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      output_h5_path=output_h5_path,
      inspection_features=inspection_features,  # type: ignore[arg-type]
      distance_matrix=distance_matrix,
      distance_matrix_method=distance_matrix_method,  # type: ignore[arg-type]
      cross_input_similarity=cross_input_similarity,
      similarity_metric=similarity_metric,  # type: ignore[arg-type]
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_or_run(spec, emit_json, out, "inspect", runner_available=True)
  if not emit_json:
    from aminx.host.runner import inspect  # noqa: PLC0415

    inspect(spec)


@spec_app.callback()
def _spec_base(
  ctx: typer.Context,
  topology: Annotated[str | None, _OPT(help="Topology file path")] = None,
  model_weights: Annotated[str, _OPT(help="Model weights name")] = "original",
  model_version: Annotated[str, _OPT(help="Model version")] = "v_48_020",
  model_family: Annotated[
    str | None,
    _OPT(
      help="Model family: proteinmpnn or ligandmpnn. Leave unset to auto-derive from "
      "checkpoint_id (RunSpecification.__post_init__); an explicit value here is always "
      "respected even if it disagrees with checkpoint_id.",
    ),
  ] = None,
  checkpoint_id: Annotated[str | None, _OPT(help="Checkpoint identifier")] = None,
  model_local_path: Annotated[Path | None, _OPT(help="Local model checkpoint path")] = None,
  checkpoint_registry_path: Annotated[Path | None, _OPT(help="Checkpoint registry path")] = None,
  ligand_mpnn_use_side_chain_context: Annotated[
    bool | None,
    _OPT(help="LigandMPNN side-chain context"),
  ] = None,
  batch_size: Annotated[int | None, _OPT(help="Batch size (None = subclass default)")] = None,
  backbone_noise: Annotated[str, _OPT(help="Comma-separated backbone noise levels")] = "0.0",
  backbone_noise_mode: Annotated[
    str,
    _OPT(help="Backbone noise mode: direct or thermal"),
  ] = "direct",
  estat_noise: Annotated[
    str | None,
    _OPT(help="Comma-separated electrostatic noise levels"),
  ] = None,
  estat_noise_mode: Annotated[str, _OPT(help="Electrostatic noise mode")] = "direct",
  vdw_noise: Annotated[str | None, _OPT(help="Comma-separated vdw noise levels")] = None,
  vdw_noise_mode: Annotated[str, _OPT(help="VDW noise mode")] = "direct",
  use_electrostatics: Annotated[bool, _OPT(help="Enable electrostatics")] = False,
  use_vdw: Annotated[bool, _OPT(help="Enable VDW")] = False,
  random_seed: Annotated[int, _OPT(help="Random seed")] = 42,
  chain_id: Annotated[str | None, _OPT(help="Chain ID")] = None,
  model: Annotated[int | None, _OPT(help="Model index")] = None,
  altloc: Annotated[str, _OPT(help="Alternate location: first or all")] = "first",
  cache_path: Annotated[Path | None, _OPT(help="Cache path")] = None,
  output_dir: Annotated[Path | None, _OPT(help="Output directory")] = None,
  max_buffer_size: Annotated[int | None, _OPT(help="Max buffer size (bytes)")] = None,
  overwrite_cache: Annotated[bool, _OPT(help="Overwrite cache")] = False,
  max_length: Annotated[int | None, _OPT(help="Max sequence length")] = 512,
  truncation_strategy: Annotated[
    str,
    _OPT(help="Truncation strategy: none, random_crop, center_crop"),
  ] = "random_crop",
  host_resource_allocation_strategy: Annotated[
    str,
    _OPT(help="Resource allocation: auto or full"),
  ] = "auto",
  ram_budget_mb: Annotated[int | None, _OPT(help="RAM budget in MB")] = None,
  max_workers: Annotated[int | None, _OPT(help="Max data loading workers")] = None,
  n_devices: Annotated[int | None, _OPT(help="Device count override")] = None,
  use_preprocessed: Annotated[bool, _OPT(help="Use preprocessed data")] = False,
  preprocessed_index_path: Annotated[Path | None, _OPT(help="Preprocessed index path")] = None,
  split: Annotated[str, _OPT(help="Data split")] = "inference",
  tied_position: Annotated[
    list[str],
    _OPT("--tied-position", help="Tied position pair A:B (repeatable) or 'auto'/'direct'"),
  ] = [],  # noqa: B006
  pass_mode: Annotated[str, _OPT(help="Pass mode: inter or intra")] = "intra",
  multi_state_temperature: Annotated[float, _OPT(help="Multi-state temperature")] = 1.0,
  input_type: Annotated[
    str,
    _OPT(
      help="Input type override: 'auto' (detect scheme), 'file' (force local), or 'pdb'/'afdb'/'mdcath'",
    ),
  ] = "auto",
  input_cache_dir: Annotated[
    Path | None,
    _OPT(help="Cache directory for fetched structures"),
  ] = None,
  potts_options_json: Annotated[
    str | None,
    _OPT("--potts-options-json", help="JSON object of PottsMPNNOptions fields"),
  ] = None,
  laser_options_json: Annotated[
    str | None,
    _OPT("--laser-options-json", help="JSON object of LaserOptions fields"),
  ] = None,
) -> None:
  """Run specification JSON (see aminx.run.spec_json).

  Base model/infra options are shared across emit-* subcommands and must be
  placed before the subcommand name, e.g.::

      aminx spec --backbone-noise 0.1,0.2 emit-sample --inputs foo.pdb
  """
  ctx.obj = _RunBase(
    topology=topology,
    model_weights=model_weights,
    model_version=model_version,
    model_family=model_family,
    checkpoint_id=checkpoint_id,
    model_local_path=model_local_path,
    checkpoint_registry_path=checkpoint_registry_path,
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    batch_size=batch_size,
    backbone_noise=backbone_noise,
    backbone_noise_mode=backbone_noise_mode,
    estat_noise=estat_noise,
    estat_noise_mode=estat_noise_mode,
    vdw_noise=vdw_noise,
    vdw_noise_mode=vdw_noise_mode,
    use_electrostatics=use_electrostatics,
    use_vdw=use_vdw,
    random_seed=random_seed,
    chain_id=chain_id,
    model=model,
    altloc=altloc,
    cache_path=cache_path,
    output_dir=output_dir,
    max_buffer_size=max_buffer_size,
    overwrite_cache=overwrite_cache,
    max_length=max_length,
    truncation_strategy=truncation_strategy,
    host_resource_allocation_strategy=host_resource_allocation_strategy,
    ram_budget_mb=ram_budget_mb,
    max_workers=max_workers,
    n_devices=n_devices,
    use_preprocessed=use_preprocessed,
    preprocessed_index_path=preprocessed_index_path,
    split=split,
    tied_position=list(tied_position),
    pass_mode=pass_mode,
    multi_state_temperature=multi_state_temperature,
    input_type=input_type,
    input_cache_dir=input_cache_dir,
    potts_options_json=potts_options_json,
    laser_options_json=laser_options_json,
  )


def _emit_spec_json(spec: Any, compact: bool, out: Path | None) -> None:
  """Serialise *spec* to JSON and write to *out* or stdout."""
  blob = json.dumps(
    run_specification_to_json_dict(spec),
    indent=None if compact else 2,
    sort_keys=True,
  )
  if out is not None:
    out.write_text(blob, encoding="utf-8")
  else:
    typer.echo(blob)


# ---------------------------------------------------------------------------
# spec emit-sample
# ---------------------------------------------------------------------------


@spec_app.command("emit-sample")
def spec_emit_sample(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON to this file instead of stdout")] = None,
  compact: Annotated[bool, _OPT("--compact", help="Single-line JSON")] = False,
  # Sample-specific
  num_samples: Annotated[int, _OPT(help="Number of samples")] = 1,
  sampling_strategy: Annotated[
    str,
    _OPT(help="Sampling strategy: temperature or straight_through"),
  ] = "temperature",
  temperature: Annotated[
    str | None,
    _OPT(help="Comma-separated temperature values; omit to use the family default"),
  ] = None,
  use_unified_driver: Annotated[bool, _OPT(help="Use unified driver")] = True,
  iterations: Annotated[int | None, _OPT(help="Straight-through iterations")] = None,
  learning_rate: Annotated[float | None, _OPT(help="Straight-through learning rate")] = None,
  use_concrete: Annotated[bool, _OPT(help="Use concrete relaxation")] = False,
  concrete_tau_start: Annotated[float, _OPT(help="Concrete tau start")] = 1.0,
  concrete_tau_end: Annotated[float, _OPT(help="Concrete tau end")] = 0.1,
  output_h5_path: Annotated[Path | None, _OPT(help="Zarr output path")] = None,
  return_logits: Annotated[bool, _OPT(help="Return logits")] = True,
  samples_batch_size: Annotated[int, _OPT(help="Samples batch size")] = 16,
  samples_chunk_size: Annotated[int | None, _OPT(help="Samples chunk size")] = None,
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 1,
  temperature_batch_size: Annotated[int, _OPT(help="Temperature batch size")] = 1,
  average_node_features: Annotated[bool, _OPT(help="Average node features")] = False,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  multi_state_strategy: Annotated[
    str,
    _OPT(help="Multi-state aggregation strategy"),
  ] = "arithmetic_mean",
  compute_pseudo_perplexity: Annotated[
    bool,
    _OPT(help="Compute pseudo-perplexity (requires return-logits)"),
  ] = False,
  ligand_conditioning: Annotated[bool, _OPT(help="Ligand conditioning")] = False,
  sidechain_conditioning: Annotated[bool, _OPT(help="Sidechain conditioning")] = False,
  campaign_mode: Annotated[bool, _OPT(help="Campaign mode")] = False,
  allow_logits_in_campaign: Annotated[bool, _OPT(help="Allow logits in campaign")] = False,
  logits_memory_budget_mb: Annotated[int | None, _OPT(help="Logits memory budget (MB)")] = None,
  ligand_context_path: Annotated[Path | None, _OPT(help="Ligand context path")] = None,
  grid_mode: Annotated[bool, _OPT(help="Grid mode")] = False,
  job_id: Annotated[str | None, _OPT(help="Grid job ID")] = None,
  chunk_id: Annotated[int | None, _OPT(help="Grid chunk ID")] = None,
  sample_start: Annotated[int | None, _OPT(help="Grid sample start")] = None,
  sample_count: Annotated[int | None, _OPT(help="Grid sample count")] = None,
) -> None:
  """Emit SamplingSpecification as JSON (non-serializable fields excluded)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415

  try:
    spec = SamplingSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      num_samples=num_samples,
      sampling_strategy=sampling_strategy,  # type: ignore[arg-type]
      **_sampling_temperature_kwargs(temperature),
      use_unified_driver=use_unified_driver,
      iterations=iterations,
      learning_rate=learning_rate,
      use_concrete=use_concrete,
      concrete_tau_start=concrete_tau_start,
      concrete_tau_end=concrete_tau_end,
      output_h5_path=output_h5_path,
      return_logits=return_logits,
      samples_batch_size=samples_batch_size,
      samples_chunk_size=samples_chunk_size,
      noise_batch_size=noise_batch_size,
      temperature_batch_size=temperature_batch_size,
      average_node_features=average_node_features,
      multi_state_strategy=multi_state_strategy,  # type: ignore[arg-type]
      compute_pseudo_perplexity=compute_pseudo_perplexity,
      ligand_conditioning=ligand_conditioning,
      sidechain_conditioning=sidechain_conditioning,
      campaign_mode=campaign_mode,
      allow_logits_in_campaign=allow_logits_in_campaign,
      logits_memory_budget_mb=logits_memory_budget_mb,
      ligand_context_path=ligand_context_path,
      grid_mode=grid_mode,
      job_id=job_id,
      chunk_id=chunk_id,
      sample_start=sample_start,
      sample_count=sample_count,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_spec_json(spec, compact, out)


# ---------------------------------------------------------------------------
# spec emit-score
# ---------------------------------------------------------------------------


@spec_app.command("emit-score")
def spec_emit_score(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  sequences_to_score: Annotated[
    list[str],
    _OPT(
      "--sequences-to-score",
      help="Amino acid sequences to score (repeatable; required for nll and logits)",
    ),
  ] = [],  # noqa: B006,
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON to this file instead of stdout")] = None,
  compact: Annotated[bool, _OPT("--compact", help="Single-line JSON")] = False,
  # Score-specific
  output_kind: Annotated[
    str,
    _OPT(
      "--output-kind",
      help="nll, logits, energy, ddg, proofread_unconditional, or proofread_conditional",
    ),
  ] = "nll",
  temperature: Annotated[str, _OPT(help="Comma-separated temperature values")] = "1.0",
  return_logits: Annotated[bool, _OPT(help="Return logits")] = False,
  return_decoding_orders: Annotated[bool, _OPT(help="Return decoding orders")] = False,
  return_all_scores: Annotated[bool, _OPT(help="Return all scores")] = False,
  output_h5_path: Annotated[Path | None, _OPT(help="HDF5 output path")] = None,
  average_node_features: Annotated[bool, _OPT(help="Average node features")] = False,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 4,
  multi_state_strategy: Annotated[
    str,
    _OPT(help="Multi-state aggregation strategy"),
  ] = "arithmetic_mean",
) -> None:
  """Emit ScoringSpecification as JSON (non-serializable fields excluded)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )
  _require_sequences_for_output_kind(output_kind, sequences_to_score)

  b: _RunBase = ctx.obj
  temperature_parsed = _parse_float_tuple(temperature)

  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  try:
    spec = ScoringSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      sequences_to_score=sequences_to_score,
      output_kind=cast(
        "Literal['nll', 'logits', 'energy', 'ddg', 'proofread_unconditional', 'proofread_conditional']",
        output_kind,
      ),
      temperature=temperature_parsed[0]
      if temperature_parsed and len(temperature_parsed) == 1
      else (temperature_parsed[0] if temperature_parsed else 1.0),
      return_logits=return_logits,
      return_decoding_orders=return_decoding_orders,
      return_all_scores=return_all_scores,
      output_h5_path=output_h5_path,
      average_node_features=average_node_features,
      noise_batch_size=noise_batch_size,
      multi_state_strategy=multi_state_strategy,  # type: ignore[arg-type]
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_spec_json(spec, compact, out)


# ---------------------------------------------------------------------------
# spec emit-jacobian
# ---------------------------------------------------------------------------


@spec_app.command("emit-jacobian")
def spec_emit_jacobian(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON to this file instead of stdout")] = None,
  compact: Annotated[bool, _OPT("--compact", help="Single-line JSON")] = False,
  # Jacobian-specific
  noise_batch_size: Annotated[int, _OPT(help="Noise batch size")] = 1,
  jacobian_batch_size: Annotated[int, _OPT(help="Jacobian batch size")] = 16,
  average_encodings: Annotated[bool, _OPT(help="Average encodings")] = True,
  average_encoding_mode: Annotated[str, _OPT(help="Encoding averaging mode")] = "inputs_and_noise",
  combine: Annotated[bool, _OPT(help="Combine Jacobians")] = False,
  combine_batch_size: Annotated[int, _OPT(help="Combine batch size")] = 8,
  output_h5_path: Annotated[
    Path | None,
    _OPT(help="Large-tensor output path (Zarr store)"),
  ] = None,
  jacobian_mode: Annotated[
    str,
    _OPT(help="categorical: full CatJac (L,21,L,21); reverse: fast score gradient (L,21)"),
  ] = "categorical",
  compute_apc: Annotated[bool, _OPT(help="Compute APC correction (categorical mode only)")] = True,
  apc_batch_size: Annotated[int, _OPT(help="APC batch size")] = 8,
  apc_residue_batch_size: Annotated[int, _OPT(help="APC residue batch size")] = 1000,
) -> None:
  """Emit JacobianSpecification as JSON (non-serializable fields excluded)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )

  from aminx.run.specs import JacobianSpecification  # noqa: PLC0415

  try:
    spec = JacobianSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      noise_batch_size=noise_batch_size,
      jacobian_batch_size=jacobian_batch_size,
      average_encodings=average_encodings,
      combine=combine,
      combine_batch_size=combine_batch_size,
      output_h5_path=output_h5_path,
      jacobian_mode=jacobian_mode,  # type: ignore[arg-type]
      compute_apc=compute_apc,
      apc_batch_size=apc_batch_size,
      apc_residue_batch_size=apc_residue_batch_size,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_spec_json(spec, compact, out)


# ---------------------------------------------------------------------------
# spec emit-inspect
# ---------------------------------------------------------------------------


@spec_app.command("emit-inspect")
def spec_emit_inspect(
  ctx: typer.Context,
  inputs: Annotated[
    list[str],
    _OPT("--inputs", help="Input PDB/structure paths (repeatable, required)"),
  ],
  inputs_fail_fast: Annotated[
    bool,
    _OPT(
      "--inputs-fail-fast",
      help="Exit 1 on first invalid --inputs entry instead of warning and skipping",
    ),
  ] = False,
  out: Annotated[Path | None, _OPT(help="Write JSON to this file instead of stdout")] = None,
  compact: Annotated[bool, _OPT("--compact", help="Single-line JSON")] = False,
  # Inspect-specific
  output_h5_path: Annotated[Path | None, _OPT(help="HDF5 output path")] = None,
  inspection_features: Annotated[
    list[str],
    _OPT(
      "--inspection-features",
      help="Features to inspect: unconditional_logits, encoded_node_features, edge_features, decoded_node_features, conditional_logits",
    ),
  ] = ["unconditional_logits"],  # noqa: B006
  distance_matrix: Annotated[bool, _OPT(help="Compute distance matrix")] = False,
  distance_matrix_method: Annotated[
    str,
    _OPT(help="Distance matrix method: ca, cb, backbone_average, closest_atom"),
  ] = "ca",
  cross_input_similarity: Annotated[
    bool,
    _OPT(help="Compute cross-input similarity (requires >=2 inputs)"),
  ] = False,
  similarity_metric: Annotated[
    str,
    _OPT(help="Similarity metric: rmsd, tm-score, gdt_ts, gdt_ha, cosine"),
  ] = "rmsd",
) -> None:
  """Emit InspectionSpecification as JSON (non-serializable fields excluded)."""
  if not inputs:
    typer.echo("--inputs is required", err=True)
    raise typer.Exit(code=2)

  b: _RunBase = ctx.obj
  inputs = resolve_inputs(
    inputs,
    input_type=b.input_type,
    cache_dir=b.input_cache_dir,
    fail_fast=inputs_fail_fast,
  )

  from aminx.run.specs import InspectionSpecification  # noqa: PLC0415

  try:
    spec = InspectionSpecification(
      inputs=inputs,
      **_base_spec_kwargs(b),  # type: ignore[arg-type]
      output_h5_path=output_h5_path,
      inspection_features=inspection_features,  # type: ignore[arg-type]
      distance_matrix=distance_matrix,
      distance_matrix_method=distance_matrix_method,  # type: ignore[arg-type]
      cross_input_similarity=cross_input_similarity,
      similarity_metric=similarity_metric,  # type: ignore[arg-type]
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  _emit_spec_json(spec, compact, out)


@spec_app.command("emit-potts")
def spec_emit_potts(
  k_neighbors: Annotated[
    int,
    _OPT(help="Number of nearest neighbours used during training"),
  ] = 48,
  n_backbones: Annotated[
    int,
    _OPT(help="Number of backbones in ensemble"),
  ] = 1,
  weights_path: Annotated[
    str,
    _OPT(help="Path to Potts model checkpoint"),
  ] = "",
  out: Annotated[Path | None, _OPT(help="Write JSON to this file instead of stdout")] = None,
  compact: Annotated[bool, _OPT("--compact", help="Single-line JSON")] = False,
) -> None:
  """Emit PottsRunSpec as JSON.

  Generates a PottsRunSpec configuration for Potts model inference with
  customizable k_neighbors, n_backbones, and weights_path.

  The output JSON can be round-tripped via PottsRunSpec.from_json().

  Args:
      k_neighbors: Number of nearest neighbours (default 48).
      n_backbones: Number of backbones in ensemble (default 1, must be >= 1).
      weights_path: Path to Potts model checkpoint (default empty string).
      out: Output JSON file path (default stdout).
      compact: Single-line JSON output (default False).

  Raises:
      typer.Exit: If validation fails.
  """
  from aminx.potts.spec import PottsRunSpec  # noqa: PLC0415

  try:
    spec = PottsRunSpec(
      k_neighbors=k_neighbors,
      n_backbones=n_backbones,
      weights_path=weights_path,
      caliby_path=None,
      trw_spec=None,  # Will use default in __post_init__
      training=False,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  # Serialize and emit
  blob = spec.to_json()
  if not compact:
    # Pretty-print if not compact
    data = json.loads(blob)
    blob = json.dumps(data, indent=2, sort_keys=True)
  if out is not None:
    out.write_text(blob, encoding="utf-8")
  else:
    typer.echo(blob)


@spec_app.command("validate")
def spec_validate(
  path: Annotated[
    Path,
    typer.Argument(exists=True, readable=True, help="JSON file from run_specification_to_json"),
  ],
) -> None:
  """Load a JSON specification and exit 0 if it constructs a valid spec."""
  text = path.read_text(encoding="utf-8")
  loaded = run_specification_from_json(text)
  typer.secho(f"OK: {type(loaded).__name__}", fg=typer.colors.GREEN)


@spec_app.command("roundtrip")
def spec_roundtrip(
  path: Annotated[Path, typer.Argument(exists=True, readable=True, help="JSON spec file")],
  out: Annotated[Path | None, typer.Option(help="Write re-serialized JSON here")] = None,
  *,
  compact: Annotated[bool, typer.Option(help="Single-line JSON")] = False,
) -> None:
  """Load JSON, re-encode through the same codec (checks round-trip)."""
  spec = run_specification_from_json(path.read_text(encoding="utf-8"))
  blob = run_specification_to_json(spec, indent=None if compact else 2)
  if out is not None:
    out.write_text(blob, encoding="utf-8")
    typer.secho(f"Wrote {out}", fg=typer.colors.GREEN)
  else:
    typer.echo(blob)


@spec_app.command("portable-roundtrip")
def spec_portable_roundtrip(
  path: Annotated[
    Path,
    typer.Argument(exists=True, readable=True, help="JSON dict: portable RunSpec subset"),
  ],
  *,
  compact: Annotated[bool, typer.Option(help="Single-line JSON")] = False,
) -> None:
  """Load portable subset JSON, rebuild :class:`~aminx.run.spec.RunSpec`, re-encode subset."""
  raw = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    typer.secho("Top-level JSON must be an object", fg=typer.colors.RED)
    raise typer.Exit(code=1)
  rs = run_spec_portable_from_dict(cast("dict[str, Any]", raw))
  out = run_spec_portable_to_dict(rs)
  typer.echo(json.dumps(out, indent=None if compact else 2))


# ---------------------------------------------------------------------------
# campaign subcommands
# ---------------------------------------------------------------------------


@campaign_app.command("plan")
def campaign_plan(
  inputs: Annotated[str, _OPT("--inputs", help="Comma-separated input paths (required)")] = ...,  # ty: ignore[invalid-parameter-default]
  campaign_id: Annotated[str, _OPT("--campaign-id", help="Campaign identifier (required)")] = ...,  # ty: ignore[invalid-parameter-default]
  manifest_path: Annotated[
    Path,
    _OPT("--manifest-path", help="Path to write manifest JSON (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  output_root: Annotated[
    Path,
    _OPT("--output-root", help="Root directory for output HDF5 files (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  designs_per_library_type: Annotated[
    int,
    _OPT("--designs-per-library-type", help="Designs per library type (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  samples_chunk_size: Annotated[
    int,
    _OPT("--samples-chunk-size", help="Samples chunk size (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  fixed_arm: Annotated[
    list[str] | None,
    _OPT(
      "--fixed-arm",
      help=(
        "Arm LABEL=DECLARATION, e.g. catalytic_triad=H38|D73|C143 -- residue letter plus "
        "0-based canonical index, pipe-separated. The letters set fixed_tokens, so the arm "
        "states WHAT to hold, not just where. A path to a 1-D .npy mask also works for large "
        "sets. Repeatable; each arm is its own row-set. Omit to fix nothing."
      ),
    ),
  ] = None,
  state_weight_profiles: Annotated[
    str,
    _OPT("--state-weight-profiles", help="Comma-separated state weight profile names"),
  ] = "equal",
  checkpoint_id: Annotated[
    str | None,
    _OPT("--checkpoint-id", help="Checkpoint identifier for manifest rows"),
  ] = None,
  chain_id: Annotated[
    str | None,
    _OPT(
      "--chain-id",
      help=(
        "Chain(s) to keep per input, forwarded to SamplingSpecification.chain_id "
        "(see aminx.run.specs.RunSpecification.chain_id / "
        "proxide.ops.processing.frame_iterator_from_inputs for accepted forms). "
        "JSON only, to support the full range that field already accepts: "
        "a bare string ('\"A\"') applies to every input; a JSON array "
        '(\'["A","B"]\') is a filter applied to every input; a JSON object '
        "mapping each --inputs path to its own chain list "
        '(\'{"a.cif": ["A","B"], "b.cif": ["A","C"]}\') selects '
        "per-input, e.g. to discard extra crystallographic copies that differ "
        "in chain layout across a multi-input PoE bead."
      ),
    ),
  ] = None,
  ligand_context_path: Annotated[
    Path | None,
    _OPT(
      "--ligand-context-path",
      help=(
        "Path to a keyed npz file supplying real Y/Y_t/Y_m ligand-atom-cloud "
        "tensors, forwarded to SamplingSpecification.ligand_context_path (see "
        "aminx.run.specs.SamplingSpecification.ligand_context_path / "
        "aminx.host._sampling_helper._load_ligand_context_file for the expected "
        "'<structure_id>::Y', '<structure_id>::Y_t', '<structure_id>::Y_m' keying "
        "convention, structure_id = Path(input_path).stem)."
      ),
    ),
  ] = None,
) -> None:
  """Generate campaign manifest JSON."""
  from aminx.host.campaign import (  # noqa: PLC0415
    SamplingSpecification,
    _parse_csv,
    parse_fixed_arms,
    write_campaign_manifest,
  )

  parsed_chain_id = json.loads(chain_id) if chain_id is not None else None
  base_spec = SamplingSpecification(
    inputs=_parse_csv(inputs),
    return_logits=False,
    **({"checkpoint_id": checkpoint_id} if checkpoint_id is not None else {}),
    **({"chain_id": parsed_chain_id} if parsed_chain_id is not None else {}),
    **({"ligand_context_path": ligand_context_path} if ligand_context_path is not None else {}),
  )
  write_campaign_manifest(
    base_spec=base_spec,
    campaign_id=campaign_id,
    manifest_path=manifest_path,
    designs_per_library_type=designs_per_library_type,
    samples_chunk_size=samples_chunk_size,
    output_root=output_root,
    fixed_arms=parse_fixed_arms(fixed_arm),
    state_weight_profiles=_parse_csv(state_weight_profiles),
  )


@campaign_app.command("worker")
def campaign_worker(
  manifest_path: Annotated[
    Path,
    _OPT("--manifest-path", help="Path to campaign manifest JSON (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  row_index: Annotated[int | None, _OPT("--row-index", help="Row index to execute")] = None,
  row_hash: Annotated[str | None, _OPT("--row-hash", help="Row hash to execute")] = None,
  lock_backend: Annotated[
    LockBackend,
    _OPT("--lock-backend", help="Lock backend: local_fs or distributed"),
  ] = LockBackend.local_fs,
  lock_lease_seconds: Annotated[
    int | None,
    _OPT("--lock-lease-seconds", help="Lock lease duration in seconds"),
  ] = None,
  heartbeat_interval_seconds: Annotated[
    int | None,
    _OPT("--heartbeat-interval-seconds", help="Heartbeat interval in seconds"),
  ] = None,
) -> None:
  """Execute one manifest row."""
  from aminx.host.campaign import (  # noqa: PLC0415
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_LOCK_LEASE_SECONDS,
    run_manifest_row,
  )

  if lock_backend == LockBackend.distributed:
    raise typer.BadParameter(
      "DistributedLockBackend is not available from the CLI; use lock_backend='local_fs' or "
      "call run_manifest_row() directly with a DistributedLockBackend instance.",
      param_hint="--lock-backend",
    )

  run_manifest_row(
    manifest_path,
    row_index=row_index,
    row_hash=row_hash,
    lock_backend=lock_backend.value,
    lock_lease_seconds=lock_lease_seconds
    if lock_lease_seconds is not None
    else DEFAULT_LOCK_LEASE_SECONDS,
    heartbeat_interval_seconds=heartbeat_interval_seconds
    if heartbeat_interval_seconds is not None
    else DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )


@campaign_app.command("run")
def campaign_run(
  manifest_path: Annotated[
    Path,
    _OPT("--manifest-path", help="Path to campaign manifest JSON (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  row_hash: Annotated[list[str], _OPT("--row-hash", help="Row hash filter (repeatable)")] = [],  # noqa: B006
  continue_on_error: Annotated[
    bool,
    _OPT("--continue-on-error", help="Continue executing rows after a failure"),
  ] = False,
  summary_path: Annotated[
    Path | None,
    _OPT("--summary-path", help="Path to write execution summary JSON"),
  ] = None,
  lock_backend: Annotated[
    LockBackend,
    _OPT("--lock-backend", help="Lock backend: local_fs or distributed"),
  ] = LockBackend.local_fs,
  lock_lease_seconds: Annotated[
    int | None,
    _OPT("--lock-lease-seconds", help="Lock lease duration in seconds"),
  ] = None,
  heartbeat_interval_seconds: Annotated[
    int | None,
    _OPT("--heartbeat-interval-seconds", help="Heartbeat interval in seconds"),
  ] = None,
) -> None:
  """Execute manifest rows sequentially."""
  from aminx.host.campaign import (  # noqa: PLC0415
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_LOCK_LEASE_SECONDS,
    _emit_json,
    execute_manifest,
  )

  if lock_backend == LockBackend.distributed:
    raise typer.BadParameter(
      "DistributedLockBackend is not available from the CLI; use lock_backend='local_fs' or "
      "call execute_manifest() directly with a DistributedLockBackend instance.",
      param_hint="--lock-backend",
    )

  summary = execute_manifest(
    manifest_path,
    row_hashes=tuple(row_hash) if row_hash else None,
    continue_on_error=continue_on_error,
    lock_backend=lock_backend.value,
    lock_lease_seconds=lock_lease_seconds
    if lock_lease_seconds is not None
    else DEFAULT_LOCK_LEASE_SECONDS,
    heartbeat_interval_seconds=heartbeat_interval_seconds
    if heartbeat_interval_seconds is not None
    else DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )
  _emit_json(summary, str(summary_path) if summary_path else None)
  if summary["failed_rows"] > 0:
    raise typer.Exit(code=1)


@campaign_app.command("gates")
def campaign_gates(
  manifest_path: Annotated[
    Path,
    _OPT("--manifest-path", help="Path to campaign manifest JSON (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  rerun_manifest_path: Annotated[
    list[Path],
    _OPT("--rerun-manifest-path", help="Rerun manifest path (repeatable)"),
  ] = [],  # noqa: B006
  report_path: Annotated[
    Path | None,
    _OPT("--report-path", help="Path to write gate report JSON"),
  ] = None,
  allow_missing_rerun: Annotated[
    bool,
    _OPT("--allow-missing-rerun", help="Allow determinism gate without full-cell rerun"),
  ] = False,
) -> None:
  """Evaluate pilot campaign gates."""
  from aminx.host.campaign import _emit_json, evaluate_campaign_gates  # noqa: PLC0415

  report = evaluate_campaign_gates(
    manifest_path,
    rerun_manifest_paths=tuple(rerun_manifest_path),
    require_full_cell_rerun=not allow_missing_rerun,
  )
  _emit_json(report, str(report_path) if report_path else None)
  if not report["promote"]:
    raise typer.Exit(code=2)


@campaign_app.command("ramp-plan")
def campaign_ramp_plan(
  inputs: Annotated[str, _OPT("--inputs", help="Comma-separated input paths (required)")] = ...,  # ty: ignore[invalid-parameter-default]
  campaign_id: Annotated[str, _OPT("--campaign-id", help="Campaign identifier (required)")] = ...,  # ty: ignore[invalid-parameter-default]
  manifest_dir: Annotated[
    Path,
    _OPT("--manifest-dir", help="Directory for staged manifest files (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  output_root: Annotated[
    Path,
    _OPT("--output-root", help="Root directory for output HDF5 files (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  stage_designs_per_library_type: Annotated[
    str,
    _OPT("--stage-designs-per-library-type", help="Comma-separated stage design counts (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  samples_chunk_size: Annotated[
    int,
    _OPT("--samples-chunk-size", help="Samples chunk size (required)"),
  ] = ...,  # ty: ignore[invalid-parameter-default]
  fixed_arm: Annotated[
    list[str] | None,
    _OPT(
      "--fixed-arm",
      help=(
        "Arm LABEL=DECLARATION, e.g. catalytic_triad=H38|D73|C143 -- residue letter plus "
        "0-based canonical index, pipe-separated. The letters set fixed_tokens, so the arm "
        "states WHAT to hold, not just where. A path to a 1-D .npy mask also works for large "
        "sets. Repeatable; each arm is its own row-set. Omit to fix nothing."
      ),
    ),
  ] = None,
  state_weight_profiles: Annotated[
    str,
    _OPT("--state-weight-profiles", help="Comma-separated state weight profile names"),
  ] = "equal",
  plan_path: Annotated[
    Path | None,
    _OPT("--plan-path", help="Path to write scale ramp plan JSON"),
  ] = None,
  checkpoint_id: Annotated[
    str | None,
    _OPT("--checkpoint-id", help="Checkpoint identifier for manifest rows"),
  ] = None,
) -> None:
  """Generate staged ramp manifests."""
  from aminx.host.campaign import (  # noqa: PLC0415
    SamplingSpecification,
    _emit_json,
    _parse_csv,
    _parse_int_csv,
    parse_fixed_arms,
    plan_scale_ramp,
  )

  base_spec = SamplingSpecification(
    inputs=_parse_csv(inputs),
    return_logits=False,
    **({"checkpoint_id": checkpoint_id} if checkpoint_id is not None else {}),
  )
  plan_payload = plan_scale_ramp(
    base_spec=base_spec,
    campaign_id=campaign_id,
    manifest_dir=manifest_dir,
    output_root=output_root,
    stage_designs_per_library_type=_parse_int_csv(stage_designs_per_library_type),
    samples_chunk_size=samples_chunk_size,
    fixed_arms=parse_fixed_arms(fixed_arm),
    state_weight_profiles=_parse_csv(state_weight_profiles),
  )
  _emit_json(plan_payload, str(plan_path) if plan_path else None)


@campaign_app.command("ramp-evaluate")
def campaign_ramp_evaluate(
  report_path: Annotated[
    list[Path],
    _OPT("--report-path", help="Stage gate report path (repeatable, required, min=1)"),
  ] = [],  # noqa: B006
  summary_path: Annotated[
    Path | None,
    _OPT("--summary-path", help="Path to write ramp evaluation summary JSON"),
  ] = None,
) -> None:
  """Evaluate staged gate reports."""
  from aminx.host.campaign import _emit_json, evaluate_scale_ramp_reports  # noqa: PLC0415

  if not report_path:
    typer.echo("--report-path is required (pass at least one)", err=True)
    raise typer.Exit(code=2)

  summary = evaluate_scale_ramp_reports(tuple(report_path))
  _emit_json(summary, str(summary_path) if summary_path else None)
  if not summary["promote"]:
    raise typer.Exit(code=2)


# Register potts sub-app
from aminx.potts.cli import potts_app

app.add_typer(potts_app, name="potts")


def main() -> None:
  """Console script entrypoint."""
  app()


if __name__ == "__main__":
  main()
