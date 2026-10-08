"""Compute + profile a categorical Jacobian, persisting everything the comparison needs.

Exists for two reasons the CLI cannot serve:

1. **The CLI discards the APC matrix.** ``host.runner.jacobian`` computes
   ``results["apc_frobenius_norm"]`` but ``cli.py`` throws the returned dict away, so the
   ``(L, L)`` map -- the thing a coupling analysis actually wants -- was unobtainable from
   the command line. (The streaming path now persists it; this script also keeps it in the
   result JSON.)
2. **The residue mapping must travel with the tensor.** A Jacobian row is a *structure*
   residue, not an MSA column, and for TEV those are neither equal nor contiguous. Losing
   ``residue_index`` makes the tensor unmappable after the fact.

Also records wall time and peak RSS per length, which is what backlog #4145 needs in order
to decide whether Hutchinson-style probe estimation is worth building.

    # profile ladder entry
    uv run python scripts/analysis/jacobian_profile.py \
        --inputs tests/data/1ubq.pdb --out /tmp/1ubq.json

    # TEV, chain A only, catalytic Cys restored
    uv run python scripts/analysis/jacobian_profile.py \
        --inputs ../tev_design/reference_states/1LVB.cif --chain-id A \
        --restore-residue 158:C --zarr-out data/tev/jacobian --out /tmp/tev.json
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("jacobian_profile")

KIB_PER_MIB = 1024


def peak_rss_mib() -> float:
  """Peak resident set size so far, in MiB (Linux reports ru_maxrss in KiB)."""
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB


def parse_restore(spec: str | None) -> tuple[int, str] | None:
  """Parse ``--restore-residue`` as ``<residue_index>:<one-letter>``."""
  if spec is None:
    return None
  index_text, _, letter = spec.partition(":")
  if not letter or len(letter) != 1:
    msg = f"--restore-residue must look like '158:C', got {spec!r}"
    raise ValueError(msg)
  return int(index_text), letter.upper()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--inputs", required=True, help="structure path (PDB/CIF)")
  parser.add_argument("--chain-id", default=None, help="restrict to one chain (e.g. A)")
  parser.add_argument("--tangent-batch-size", type=int, default=None, help="SafeMap tile; None = planner")
  parser.add_argument(
    "--restore-residue",
    default=None,
    help=(
      "Override one residue before computing the Jacobian, as '<residue_index>:<letter>'. "
      "For TEV/1LVB use '158:C': the crystal is the C151A catalytic-null mutant, and the "
      "MSA query is WT, so conditioning on Ala would put a non-native residue at exactly "
      "the catalytic position the comparison cares about."
    ),
  )
  parser.add_argument("--zarr-out", type=Path, default=None, help="persist tensors here")
  parser.add_argument("--out", type=Path, default=None, help="result JSON path")
  parser.add_argument("--skip-apc", action="store_true", help="skip the (L, L) APC reduction")
  args = parser.parse_args()

  import jax
  import jax.numpy as jnp

  from aminx.host.prep import prep_protein_stream_and_model
  from aminx.run.specs import JacobianSpecification
  from aminx.utils.aa_convert import MPNN_ALPHABET, af_to_mpnn
  from aminx.utils.apc import apc_corrected_frobenius_norm
  from aminx.utils.forward_jac import make_categorical_jacobian_fn

  spec = JacobianSpecification(inputs=[args.inputs], max_length=None, chain_id=args.chain_id)
  protein_iterator, model = prep_protein_stream_and_model(spec)
  batch = next(iter(protein_iterator))

  if getattr(batch, "aatype", None) is None:
    msg = f"{args.inputs} carries no sequence; a Jacobian needs one"
    raise ValueError(msg)

  coords = batch.coordinates[0]
  mask = batch.mask[0]
  residue_index = np.asarray(batch.residue_index[0]).astype(int)
  chain_index = batch.chain_index[0]
  sequence = np.asarray(af_to_mpnn(batch.aatype[0])).astype(int)
  seq_len = int(coords.shape[0])

  restored = parse_restore(args.restore_residue)
  restore_record = None
  if restored is not None:
    target_index, letter = restored
    positions = np.flatnonzero(residue_index == target_index)
    if positions.size != 1:
      msg = (
        f"--restore-residue {args.restore_residue}: residue_index {target_index} matched "
        f"{positions.size} rows; expected exactly 1"
      )
      raise ValueError(msg)
    row = int(positions[0])
    was = MPNN_ALPHABET[sequence[row]]
    sequence[row] = MPNN_ALPHABET.index(letter)
    restore_record = {"residue_index": target_index, "row": row, "from": was, "to": letter}
    logger.info("restored residue_index %d (row %d): %s -> %s", target_index, row, was, letter)

  jacobian_fn = make_categorical_jacobian_fn(model, tangent_batch_size=args.tangent_batch_size)

  logger.info(
    "computing Jacobian: L=%d tangents=%d tile=%s",
    seq_len,
    seq_len * len(MPNN_ALPHABET),
    args.tangent_batch_size or "planner",
  )
  started = time.monotonic()
  jacobian = jacobian_fn(
    jax.random.PRNGKey(0),
    coords,
    mask,
    jnp.asarray(residue_index),
    chain_index,
    jnp.asarray(sequence),
  )
  jacobian.block_until_ready()
  jacobian_seconds = time.monotonic() - started

  jacobian_np = np.asarray(jacobian)
  payload: dict[str, object] = {
    "input": args.inputs,
    "chain_id": args.chain_id,
    "seq_len": seq_len,
    "n_tangents": seq_len * len(MPNN_ALPHABET),
    "tangent_batch_size": args.tangent_batch_size,
    "jacobian_seconds": jacobian_seconds,
    "peak_rss_mib": peak_rss_mib(),
    "jacobian_shape": list(jacobian_np.shape),
    "jacobian_abs_max": float(np.abs(jacobian_np).max()),
    "jacobian_nonzero_fraction": float((jacobian_np != 0).mean()),
    "jacobian_all_finite": bool(np.all(np.isfinite(jacobian_np))),
    # Carried so the tensor stays mappable: rows are STRUCTURE residues, and for TEV they
    # are neither 0-based nor contiguous.
    "residue_index_min": int(residue_index.min()),
    "residue_index_max": int(residue_index.max()),
    "residue_index_contiguous": bool(np.all(np.diff(residue_index) == 1)),
    "n_residue_gaps": int((np.diff(residue_index) != 1).sum()),
    "restored_residue": restore_record,
  }

  # A zero Jacobian is the failure this whole path shipped with; refuse to report one as a
  # successful profile.
  if payload["jacobian_abs_max"] == 0.0:
    msg = (
      "Jacobian is identically zero -- the decoder received no sequence information. "
      "Check ar_mask (must be 1 - I, not zeros); see aminx.utils.forward_jac."
    )
    raise ValueError(msg)

  if not args.skip_apc:
    started = time.monotonic()
    apc = np.asarray(apc_corrected_frobenius_norm(jacobian))
    payload["apc_seconds"] = time.monotonic() - started
    payload["apc_shape"] = list(apc.shape)
    payload["apc_max"] = float(apc.max())
    payload["apc_mean"] = float(apc.mean())
    payload["peak_rss_mib"] = peak_rss_mib()
  else:
    apc = None

  if args.zarr_out is not None:
    from xtrax.run import (
      SinkSpec,
      ZarrStagingSink,
      fsync_tree,
      new_run_id,
      zarr_content_digest,
    )

    # run_id is REQUIRED and has no default: SinkSpec.__init__ raises
    # "missing 1 required positional argument: 'run_id'", so --zarr-out could
    # never have worked. It is the join key tying a store to the run that wrote
    # it, and __post_init__ enforces str even outside the test env.
    #
    # new_run_id() is the right source here rather than anything derived: this
    # script has no RunSpec to take an id from, and a fresh id per invocation
    # is what makes xtrax's re-run policy do the useful thing -- pointing
    # --zarr-out at an existing store raises ValueError instead of silently
    # interleaving two profiles in one tree.
    run_id = new_run_id()
    sink = ZarrStagingSink(
      SinkSpec(run_id=run_id, output_dir=args.zarr_out, format="zarr", flush_every=1),
    )
    arrays: dict[str, np.ndarray] = {
      "categorical_jacobian": jacobian_np,
      "residue_index": residue_index.astype(np.int32),
      "sequence": sequence.astype(np.int32),
    }
    if apc is not None:
      arrays["apc_frobenius_norm"] = apc
    sink.stage(("0",), **arrays)
    sink.drain()
    fsync_tree(args.zarr_out)
    payload["zarr_path"] = str(args.zarr_out)
    payload["zarr_run_id"] = run_id
    payload["zarr_digest"] = zarr_content_digest(args.zarr_out)

  text = json.dumps(payload, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    logger.info("wrote %s", args.out)
  else:
    sys.stdout.write(text + "\n")

  logger.info(
    "L=%d  jacobian %.1fs  peak RSS %.0f MiB  max|J| %.4f",
    seq_len,
    jacobian_seconds,
    payload["peak_rss_mib"],
    payload["jacobian_abs_max"],
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
