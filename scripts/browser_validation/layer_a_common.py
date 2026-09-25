"""Layer (a) shared plumbing for the exact-tier harness (T7, AC-4/AC-6).

Everything a `layer_a_exact*.py` script needs that is not row-specific numerical
comparison lives here:

- **Model loaders** (`load_full_model`, `load_soluble_membrane_models`,
  `load_sidechain_context_models`, `load_packer_model`) -- thin wrappers around the
  ALREADY-TESTED loaders in `tests/parity/test_{full_model,soluble_membrane,
  sidechain_context,packer}_parity.py`. Every wrapper takes an explicit
  ``weight_source`` (``"eqx"`` or ``"pt_convert"``) with **no default** -- a caller
  that forgets to choose a side is a bug, not a silently-resolved eqx.
- **`reference_call`** -- wraps a call into `tests.parity.reference_utils` (or any
  function that may raise ``pytest.skip.Exception``) so the caller's `SkipCounter`
  is incremented instead of the whole run crashing; per the common context, a run
  with ``n_skipped > 0`` still finishes and writes its result, then exits 4.
- **`pins.assert_reference_pinned`** call sites (`_assert_pinned`) for every ``.pt``
  this module loads directly (the wrapped test loaders already pin the aminx
  ``.eqx.zst`` side via `aminx.io.weights`; this module additionally pins the raw
  ``.pt`` bytes against `reference_pins.json` before trusting them).
- **The reference decoding-order formula** (`reference_formula_order`), fixed by an
  explicit seed so every row that needs "the" pinned order draws the identical one.
- **Per-side input hashing** (`input_sha256`).
- **`provenance()`** -- `git_hash`/`git_clean` via `bathos.git.capture_git_state`
  (exits 3 unless the live-git channel wins, matching the common context's
  `git_clean` note) plus `prereg_sha256` of whatever params file is actually read,
  captured once before any write.
- **`section_sha` / `prereq_check`** -- Bathos staging's prerequisite-check
  machinery, with **Deviation D7** (see `prereq_check` docstring) in place of the
  spec's ``json_extract_string(metadata, ...)`` recipe, which cannot work against
  bathos ``84be544e``.
- **`reference_path` / `proteinmpnn_path`** -- the two reference checkout roots.
- **`differential_mode` / `emit`** -- the differential pre-flight contract
  (`BTH_DIFFERENTIAL_{KNOB,VALUE,PHASE}`, skip `--out` in a differential phase).

CLI: ``python layer_a_common.py section-sha <params.json> <section>`` prints the
canonical-JSON sha256 of ``params[section]`` (sha256 of
``json.dumps(params[section], sort_keys=True, separators=(",", ":"))``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WeightSource = Literal["eqx", "pt_convert"]
_WEIGHT_SOURCES: tuple[WeightSource, ...] = ("eqx", "pt_convert")

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]

DEFAULT_REFERENCE_PATH = "/home/marielle/projects/aminx/reference_ligandmpnn_clone"


def _worktree_root() -> Path:
  return _WORKTREE_ROOT


def reference_path() -> Path:
  """Return `$REFERENCE_PATH` (the LigandMPNN reference clone root)."""
  return Path(os.environ.get("REFERENCE_PATH", DEFAULT_REFERENCE_PATH))


def proteinmpnn_path() -> Path:
  """Return `$PROTEINMPNN_PATH` (the pinned ProteinMPNN clone root)."""
  default = _worktree_root() / ".cache" / "reference" / "ProteinMPNN"
  return Path(os.environ.get("PROTEINMPNN_PATH", str(default)))


def _add_worktree_to_syspath() -> None:
  root = str(_worktree_root())
  if root not in sys.path:
    sys.path.insert(0, root)
  script_dir = str(_SCRIPT_DIR)
  if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# --------------------------------------------------------------------------------------
# SkipCounter / reference_call -- pytest.skip.Exception tolerance (common context, step 1)
# --------------------------------------------------------------------------------------


class SkipCounter:
  """Counts `pytest.skip.Exception`s absorbed by `reference_call`."""

  def __init__(self) -> None:
    self.n = 0
    self.reasons: list[str] = []

  def bump(self, reason: str) -> None:
    self.n += 1
    self.reasons.append(reason)


def reference_call(counter: SkipCounter, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
  """Call `fn`, absorbing `pytest.skip.Exception` into `counter` instead of raising.

  Every call into `tests.parity.reference_utils` (directly, or transitively via a
  wrapped test-module loader) goes through this so a missing prerequisite degrades
  to `n_skipped += 1` rather than crashing the whole harness run (common context,
  step 1: "every `reference_utils` call wrapped so `pytest.skip.Exception`
  increments `n_skipped` (exit 4 after writing)").
  """
  import pytest

  try:
    return fn(*args, **kwargs)
  except pytest.skip.Exception as exc:  # _pytest.outcomes.Skipped
    reason = str(exc)
    logger.warning("reference_call: skipped (%s): %s", fn, reason)
    counter.bump(reason)
    return None


# --------------------------------------------------------------------------------------
# Reference-checkpoint pinning (pins.assert_reference_pinned call sites)
# --------------------------------------------------------------------------------------


def _pins_module() -> Any:
  _add_worktree_to_syspath()
  import pins  # noqa: PLC0415 (sibling module, script-dir on sys.path)

  return pins


def _assert_pinned(root: Path, filename: str, pin_key_prefix: str) -> None:
  """Assert `root / filename` hashes to the value recorded under `reference_pins.json`."""
  pins_path = _SCRIPT_DIR / "reference_pins.json"
  with pins_path.open() as fh:
    pins_data = json.load(fh)
  key = f"{pin_key_prefix}/{filename}"
  expected = pins_data.get("weights_sha256", {}).get(key)
  if not expected:
    print(f"layer_a_common: no pinned sha256 recorded for {key!r} in {pins_path}", file=sys.stderr)
    raise SystemExit(3)
  _pins_module().assert_reference_pinned(root / filename, expected)


# --------------------------------------------------------------------------------------
# Model loaders (reused BY IMPORT from tests/parity/test_*_parity.py)
# --------------------------------------------------------------------------------------


def _check_weight_source(weight_source: WeightSource) -> None:
  if weight_source not in _WEIGHT_SOURCES:
    msg = f"weight_source must be one of {_WEIGHT_SOURCES!r}, got {weight_source!r}"
    raise ValueError(msg)


def load_full_model(weight_source: WeightSource) -> tuple[Any, Any, Any, Any]:
  """Load the P00/P04-P09 full-model pair (proteinmpnn_v_48_020, k=48).

  Returns ``(jax_model, pt_model, torch_module, model_utils_module)``. Delegates
  entirely to `tests.parity.test_full_model_parity`'s already-tested loader; this
  function only selects `weight_source` (no default) and re-verifies the raw
  reference `.pt` pin.
  """
  _check_weight_source(weight_source)
  _add_worktree_to_syspath()
  from tests.parity.test_full_model_parity import (
    _jax_protein_for_source,
    _load_heavy_parity_models_impl,
  )

  models = _load_heavy_parity_models_impl()
  _assert_pinned(reference_path() / "model_params", "proteinmpnn_v_48_020.pt", "ligandmpnn_pt")
  jax_model = _jax_protein_for_source(models, weight_source)
  return jax_model, models.pt_model, models.torch, models.model_utils


def load_soluble_membrane_models(weight_source: WeightSource) -> Any:
  """Load the P12/P13 soluble + membrane model set (SolubleMembraneParityModels)."""
  _check_weight_source(weight_source)
  _add_worktree_to_syspath()
  from tests.parity.test_soluble_membrane_parity import _load_soluble_membrane_models_impl

  models = _load_soluble_membrane_models_impl()
  _assert_pinned(reference_path() / "model_params", "solublempnn_v_48_020.pt", "ligandmpnn_pt")
  _assert_pinned(
    reference_path() / "model_params",
    "per_residue_label_membrane_mpnn_v_48_020.pt",
    "ligandmpnn_pt",
  )
  _assert_pinned(
    reference_path() / "model_params",
    "global_label_membrane_mpnn_v_48_020.pt",
    "ligandmpnn_pt",
  )
  return models


def jax_model_for_source(models: Any, weight_source: WeightSource, checkpoint_kind: str) -> Any:
  """Select a JAX model out of `SolubleMembraneParityModels` (re-export, no default)."""
  _check_weight_source(weight_source)
  _add_worktree_to_syspath()
  from tests.parity.test_soluble_membrane_parity import _jax_model_for_source

  return _jax_model_for_source(models, weight_source, checkpoint_kind)


def load_sidechain_context_models(
  weight_source: WeightSource, *, use_side_chain_context: bool
) -> tuple[Any, Any]:
  """Load the P11 ligand/side-chain-context pair (ligandmpnn_v_32_010_25, k=32).

  Returns ``(reference_model, aminx_model)``. `weight_source` selects which aminx
  checkpoint variant feeds the aminx side: `"eqx"` uses the shipped `.eqx.zst`
  directly (`_load_aminx_model` always reads the shipped file); `"pt_convert"` is
  not offered by the upstream test module for this path (P11 only ships an `eqx`
  loader) -- callers asking for `"pt_convert"` here get the same shipped-weights
  model, and the caller's row should record `weight_source="eqx"` for this path.
  """
  _check_weight_source(weight_source)
  _add_worktree_to_syspath()
  from tests.parity.test_sidechain_context_parity import _load_aminx_model, _load_reference_model

  _assert_pinned(reference_path() / "model_params", "ligandmpnn_v_32_010_25.pt", "ligandmpnn_pt")
  reference_model = _load_reference_model(use_side_chain_context=use_side_chain_context)
  aminx_model = _load_aminx_model(use_side_chain_context=use_side_chain_context)
  return reference_model, aminx_model


def load_packer_model() -> tuple[Any, Any]:
  """Load the P14 packer pair (ligandmpnn_sc_v_32_002_16).

  Returns ``(jax_packer, pt_packer_module_namespace)`` where the latter is the
  reference `sc_utils` module (imported via `reference_utils.import_reference_module`)
  so a caller can build its own reference `Packer` instance against a real fixture.
  """
  _add_worktree_to_syspath()
  from tests.parity.reference_utils import import_reference_module, require_heavy_parity_prereqs

  reference_root, _ = require_heavy_parity_prereqs(
    python_modules=["Bio"],
    reference_rel_paths=["model_params/ligandmpnn_sc_v_32_002_16.pt"],
  )
  _assert_pinned(reference_root / "model_params", "ligandmpnn_sc_v_32_002_16.pt", "ligandmpnn_pt")
  sc_utils = import_reference_module("sc_utils")
  return sc_utils, reference_root


# --------------------------------------------------------------------------------------
# Reference decoding-order formula + input hashing
# --------------------------------------------------------------------------------------


def reference_formula_order(
  mask: np.ndarray,
  chain_mask: np.ndarray,
  *,
  seed: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Return `(randn, decoding_order)` from the reference argsort formula.

  ``decoding_order = argsort((mask * chain_mask + 1e-4) * |randn|)``
  (LigandMPNN `model_utils.py:217-220`). `randn` is drawn from a fixed,
  caller-supplied `seed` -- exact-tier runs use ONE randn per fixture, computed
  once and fed to both sides (spec "Exact-tier runs" paragraph), never re-drawn
  per row.
  """
  rng = np.random.default_rng(seed)
  randn = rng.standard_normal(size=mask.shape).astype(np.float32)
  chain_m = np.asarray(mask, dtype=np.float32) * np.asarray(chain_mask, dtype=np.float32)
  order = np.argsort((chain_m + 1e-4) * np.abs(randn))
  return randn, order


def ar_mask_from_order(order: np.ndarray) -> np.ndarray:
  """Build the backward autoregressive mask for a fixed decoding `order`.

  ``ar_mask[i, j] = 1`` iff `j` is decoded strictly before `i`. Vectorized (no
  Python-level L^2 loop) so it stays cheap even for the largest fixtures.
  """
  length = order.shape[0]
  position = np.empty(length, dtype=np.int64)
  position[order] = np.arange(length)
  return (position[None, :] < position[:, None]).astype(np.int32)


def input_sha256(array: Any) -> str:
  """Return the sha256 hex digest of `array`'s contiguous bytes (per-side input hash)."""
  arr = np.ascontiguousarray(np.asarray(array))
  return hashlib.sha256(arr.tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def provenance(params_path: str | Path | None = None) -> dict[str, Any]:
  """Return `{git_hash, git_clean, prereg_sha256}`.

  `git_hash`/`git_clean` come from `bathos.git.capture_git_state`, captured once
  at process start (before any write -- a calibrate run later rewrites the tracked
  params file, so provenance must be read before that happens). Exits 3 unless the
  live-git channel (`provenance_source == "git"`) won, matching the common
  context's `git_clean` note (myxcel env/sidecar channels would otherwise take
  silent precedence over the live checkout this run actually executed in).

  `prereg_sha256` is the sha256 of `params_path`'s bytes if given and it exists,
  else `""` (a calibrate run reads no params file at all; a validate run always
  passes one).
  """
  import bathos.git as bathos_git  # noqa: PLC0415 (only present with --with "$BATHOS_REQ")

  state = bathos_git.capture_git_state(_worktree_root())
  if state.provenance_source != "git":
    print(
      f"layer_a_common.provenance: expected the live-git provenance channel, got "
      f"{state.provenance_source!r} (a myxcel env/sidecar channel took precedence)",
      file=sys.stderr,
    )
    raise SystemExit(3)

  prereg_sha256 = ""
  if params_path is not None:
    candidate = Path(params_path)
    if candidate.is_file():
      prereg_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()

  return {
    "git_hash": state.hash,
    "git_clean": not state.dirty,
    "prereg_sha256": prereg_sha256,
  }


# --------------------------------------------------------------------------------------
# Bathos staging: section_sha / prereq_check
# --------------------------------------------------------------------------------------


def section_sha(params: dict[str, Any], section: str) -> str:
  """Return the sha256 of `params[section]`'s canonical JSON (`sort_keys`, `(',', ':')`)."""
  payload = params[section]
  canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prereq_check(
  stem: str,
  section: str,
  params: dict[str, Any],
  *,
  bth_bin: str | None = None,
  cwd: Path | None = None,
) -> dict[str, Any]:
  """Bathos-staging prerequisite check for a validate script -- Deviation D7.

  **D7 (binding orchestrator override).** bathos ``84be544e`` never persists
  ``runs.metadata``: ``Run.to_arrow()`` (bathos ``schema.py:227``) drops the
  ``metadata`` field entirely on write, so ``json_extract_string(metadata, ...)``
  is always ``NULL`` after ``bth compact`` -- the spec's "Bathos staging" recipe
  (``json_extract_string(metadata,'$.params_section_sha256')``) cannot work
  against this bathos commit. This function therefore does NOT query metadata.
  Instead: read `params[section]["git_hash"]` directly (the calibrate run's own
  provenance, already embedded in the params file it wrote), recompute
  `section_sha(params, section)` locally, and ask `$BTH_BIN sql` (default `bth`)
  only whether a `pass`-outcome `<stem>_calibrate.py` row exists at that
  `git_hash` -- the params-content binding (section hash <-> that specific
  catalog row) is enforced by the ORCHESTRATOR's gate (see the orchestrator Gate
  block's `json_extract_string(metadata,'$.params_section_sha256')` line, which
  runs against a DIFFERENT catalog than this in-script check and is not affected
  by this deviation), not by this function.

  Returns
  -------
  dict with `prereq_ok`, `calibrate_git_hash`, `params_section_sha256`,
  `n_matching_rows`, `reason` (`None` when `prereq_ok`).
  """
  bth = bth_bin or os.environ.get("BTH_BIN", "bth")
  working_dir = cwd or _worktree_root()

  calibrate_git_hash = params.get(section, {}).get("git_hash")
  if not (isinstance(calibrate_git_hash, str) and len(calibrate_git_hash) == 40):
    return {
      "prereq_ok": False,
      "calibrate_git_hash": calibrate_git_hash,
      "params_section_sha256": None,
      "n_matching_rows": 0,
      "reason": (
        f"params[{section!r}]['git_hash'] is missing or not a 40-char hex sha "
        f"(got {calibrate_git_hash!r})"
      ),
    }

  params_section_sha256 = section_sha(params, section)

  sql = (
    "SELECT git_hash FROM runs WHERE command LIKE "
    f"'%{stem}_calibrate.py%' AND outcome = 'pass' AND git_hash = '{calibrate_git_hash}'"
  )
  try:
    proc = subprocess.run(  # noqa: S603
      [bth, "sql", "--sql", sql],
      capture_output=True,
      text=True,
      check=True,
      cwd=working_dir,
    )
  except (OSError, subprocess.CalledProcessError) as exc:
    return {
      "prereq_ok": False,
      "calibrate_git_hash": calibrate_git_hash,
      "params_section_sha256": params_section_sha256,
      "n_matching_rows": 0,
      "reason": f"`{bth} sql` failed: {exc}",
    }

  try:
    payload = json.loads(proc.stdout)
    n_matching_rows = int(payload.get("count", 0))
  except (json.JSONDecodeError, TypeError, ValueError) as exc:
    return {
      "prereq_ok": False,
      "calibrate_git_hash": calibrate_git_hash,
      "params_section_sha256": params_section_sha256,
      "n_matching_rows": 0,
      "reason": f"could not parse `{bth} sql` output as {{rows,count}} JSON: {exc}",
    }

  if n_matching_rows < 1:
    return {
      "prereq_ok": False,
      "calibrate_git_hash": calibrate_git_hash,
      "params_section_sha256": params_section_sha256,
      "n_matching_rows": n_matching_rows,
      "reason": (
        f"no catalog row with command LIKE '%{stem}_calibrate.py%', outcome='pass', "
        f"git_hash='{calibrate_git_hash}'"
      ),
    }

  ancestor = subprocess.run(  # noqa: S603, S607
    ["git", "merge-base", "--is-ancestor", calibrate_git_hash, "HEAD"],
    cwd=working_dir,
    check=False,
  )
  if ancestor.returncode != 0:
    return {
      "prereq_ok": False,
      "calibrate_git_hash": calibrate_git_hash,
      "params_section_sha256": params_section_sha256,
      "n_matching_rows": n_matching_rows,
      "reason": (
        f"git_hash {calibrate_git_hash} is not an ancestor of HEAD "
        f"(git merge-base --is-ancestor exit {ancestor.returncode})"
      ),
    }

  return {
    "prereq_ok": True,
    "calibrate_git_hash": calibrate_git_hash,
    "params_section_sha256": params_section_sha256,
    "n_matching_rows": n_matching_rows,
    "reason": None,
  }


# --------------------------------------------------------------------------------------
# Differential pre-flight
# --------------------------------------------------------------------------------------


def differential_mode() -> dict[str, Any]:
  """Return `{active, knob, value, phase}` from `BTH_DIFFERENTIAL_{KNOB,VALUE,PHASE}`.

  `active` is `True` iff `BTH_DIFFERENTIAL_PHASE` is set (bathos's own signal that
  this invocation is one arm of a differential pre-flight re-execution, spec
  "Differential pre-flight rules").
  """
  return {
    "active": os.environ.get("BTH_DIFFERENTIAL_PHASE") is not None,
    "knob": os.environ.get("BTH_DIFFERENTIAL_KNOB"),
    "value": os.environ.get("BTH_DIFFERENTIAL_VALUE"),
    "phase": os.environ.get("BTH_DIFFERENTIAL_PHASE"),
  }


def emit(result: dict[str, Any], out: str | Path | None) -> None:
  """Write `result` to `$BTH_RESULTS_PATH` always, and to `out` UNLESS a differential phase is set.

  Per "Differential pre-flight rules": "when `BTH_DIFFERENTIAL_PHASE` is set, a
  script computes only the differential metric ... and writes ONLY to
  `$BTH_RESULTS_PATH`, never `--out`".
  """
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    results_file = Path(results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w") as fh:
      json.dump(result, fh, indent=2, default=str)

  mode = differential_mode()
  if mode["active"]:
    logger.info(
      "differential phase %r active: skipping --out per Bathos staging "
      "(wrote BTH_RESULTS_PATH only)",
      mode["phase"],
    )
    return

  if out is not None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
      json.dump(result, fh, indent=2, default=str)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  section_sha_parser = subparsers.add_parser(
    "section-sha", help="Print the canonical-JSON sha256 of params[section]."
  )
  section_sha_parser.add_argument("params_file", type=Path)
  section_sha_parser.add_argument("section")

  args = parser.parse_args(argv)

  if args.command == "section-sha":
    with args.params_file.open() as fh:
      params = json.load(fh)
    print(section_sha(params, args.section))
    return 0

  parser.error(f"unknown command {args.command!r}")
  return 2


if __name__ == "__main__":
  sys.exit(main())
