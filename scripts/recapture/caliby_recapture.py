#!/usr/bin/env python3
"""
Recapture caliby (calibration y) corrections from training data.

Caliby is a learned post-hoc correction to TRW marginals, capturing systematic
deviations from target distributions (e.g., native sequence distributions,
experimental folding data). This script trains a LearnedCalibration module
on a calibration dataset and saves it to caliby_<id>.eqx.zst.

Usage:
  uv run python scripts/recapture/caliby_recapture.py \\
    --dataset-path /path/to/calibration/data \\
    --potts-model /path/to/potts_model.eqx.zst \\
    --out /path/to/caliby_<id>.eqx.zst \\
    --dry-run

The dataset should contain native sequences and corresponding TRW marginals
from a reference (e.g., mistypotts training data). Correction is computed
as the systematic gap between native log-likelihood and model predictions.

NOTE (Debt #3072 — Historical Misattribution Correction):
  This script trains the internal Potts marginal calibration sidecar (`aminx.potts.calibration`)
  and is UNRELATED to the upstream `ProteinDesignLab/caliby` model family
  (Shuai et al., 2025/2026; Zenodo DOI 10.5281/zenodo.18644985), which is ported in
  `aminx.families.caliby`. To prevent collisions with upstream Caliby checkpoints,
  default output filenames should use `potts_calibration_<id>.eqx.zst`.

References:
  - ADR 260605_potts-parallel-not-stageset.md: Potts as parallel model family
  - Task 260605_multistate-potts Track J (#1298)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
  """Main entry point for caliby recapture.

  Parameters
  ----------
  argv : list[str] | None, optional
    Command-line arguments passed to ``parser.parse_args``.

  Returns
  -------
  int
    Process exit code (``0`` on success, ``1`` on validation failure).
  """
  parser = argparse.ArgumentParser(
      description="Recapture caliby corrections from calibration dataset.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--dataset-path",
      type=str,
      required=False,
      default=None,
      help="Path to calibration dataset (Parquet or NPZ). Required for non-dry-run mode.",
  )
  parser.add_argument(
      "--potts-model",
      type=str,
      required=False,
      default=None,
      help="Path to PottsModel checkpoint (eqx.zst). Required for non-dry-run mode.",
  )
  parser.add_argument(
      "--out",
      type=str,
      required=True,
      help="Output path for potts_calibration_<id>.eqx.zst checkpoint (legacy caliby_<id>.eqx.zst also accepted).",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Validate inputs and exit without training. Useful for pre-flight checks.",
  )
  parser.add_argument(
      "--log-level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Logging verbosity.",
  )

  args = parser.parse_args(argv)

  # Configure logging
  logging.basicConfig(
      level=getattr(logging, args.log_level),
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )

  logger.info("Caliby recapture script started (pid=%d)", __import__("os").getpid())

  # Dry-run: validate paths only
  if args.dry_run:
    logger.info("Dry-run mode: validating inputs only.")
    if args.dataset_path:
      dataset_path = Path(args.dataset_path)
      if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        return 1
      logger.info("Dataset path found: %s", dataset_path)

    if args.potts_model:
      model_path = Path(args.potts_model)
      if not model_path.exists():
        logger.error("Potts model path does not exist: %s", model_path)
        return 1
      logger.info("Potts model path found: %s", model_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Output path writable: %s", out_path)
    logger.info("Dry-run completed successfully.")
    return 0

  # Real run: check dataset and model are provided
  if not args.dataset_path or not args.potts_model:
    logger.error(
        "Non-dry-run mode requires both --dataset-path and --potts-model. "
        "Dataset path: %s, Potts model: %s",
        args.dataset_path,
        args.potts_model,
    )
    return 1

  # Validate paths exist
  dataset_path = Path(args.dataset_path)
  model_path = Path(args.potts_model)
  out_path = Path(args.out)

  if not dataset_path.exists():
    logger.error("Dataset path does not exist: %s", dataset_path)
    return 1

  if not model_path.exists():
    logger.error("Potts model path does not exist: %s", model_path)
    return 1

  out_path.parent.mkdir(parents=True, exist_ok=True)

  # Main implementation: NOT IMPLEMENTED
  # This is a scaffold for future implementation. The caliby recapture pipeline
  # requires a calibration dataset (location TBD), a reference TRW marginal
  # computation, and a training loop to learn the correction. This is deferred
  # pending upstream mistypotts caliby dataset preparation.
  logger.info("Caliby recapture training pipeline not yet implemented.")
  logger.info("Required: upstream mistypotts caliby dataset location and format spec.")
  logger.info("Deferring full implementation to next phase.")

  raise NotImplementedError(
      "Caliby recapture training pipeline not implemented. "
      "Awaiting calibration dataset availability and format specification. "
      "See .praxia/docs/research/260605_caliby-nature-and-dataset.md for status."
  )


if __name__ == "__main__":
  exit(main())
