"""Batch-level grouping of variable-length sequences into fixed-size XLA buckets.

This module groups a batch of sequences by bucket assignment and calls
BatchPlanner.plan() once per bucket, preventing recompilation on every
novel sequence length. Bucket selection is delegated to
``xtrax.tiling.select_bucket``.

JAX constraint: bucket boundaries are static (config-loaded at init);
individual sequence lengths are dynamic but map to static bucket keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xtrax.tiling import BatchPlan


@dataclass(frozen=True)
class BucketingConfig:
    """Configuration for sequence length bucketing.

    Attributes
    ----------
    buckets : tuple[int, ...]
        Sorted bucket ceiling lengths. Default (64, 128, 256, 512).

    """

    buckets: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        """Validate bucket configuration."""
        if not self.buckets:
            raise ValueError("buckets tuple cannot be empty")
        sorted_buckets = tuple(sorted(self.buckets))
        if self.buckets != sorted_buckets:
            raise ValueError(f"buckets must be sorted; got {self.buckets}")


def select_bucket(seq_len: int, config: BucketingConfig) -> int:
    """Select the appropriate bucket for a sequence length.

    Parameters
    ----------
    seq_len : int
        Sequence length to bucket.
    config : BucketingConfig
        Bucketing configuration with sorted bucket ceilings.

    Returns
    -------
    int
        Bucket ceiling for this sequence length.

    Raises
    ------
    ValueError
        If seq_len exceeds all configured buckets.

    """
    from xtrax.tiling import select_bucket as xtrax_select_bucket  # noqa: PLC0415

    try:
        return xtrax_select_bucket(seq_len, boundaries=tuple(config.buckets))
    except ValueError as exc:
        if seq_len >= 0 and config.buckets and seq_len > config.buckets[-1]:
            msg = f"{seq_len} exceeds all buckets {config.buckets}"
            raise ValueError(msg) from exc
        raise


def group_by_bucket(
    seq_lens: list[int], config: BucketingConfig,
) -> dict[int, list[int]]:
    """Group sequence indices by bucket assignment.

    Parameters
    ----------
    seq_lens : list[int]
        Sequence lengths for each position in the batch.
    config : BucketingConfig
        Bucketing configuration.

    Returns
    -------
    dict[int, list[int]]
        Mapping from bucket ceiling to list of original sequence indices
        assigned to that bucket, in encounter order.

    Raises
    ------
    ValueError
        If any sequence length exceeds all configured buckets.

    """
    result: dict[int, list[int]] = {}
    for idx, seq_len in enumerate(seq_lens):
        bucket = select_bucket(seq_len, config)  # May raise ValueError
        if bucket not in result:
            result[bucket] = []
        result[bucket].append(idx)
    return result


@dataclass(frozen=True)
class BucketAssignment:
    """Result of bucketing a batch and planning per bucket.

    Attributes
    ----------
    bucket_boundaries : tuple[int, ...]
        Sorted tuple of bucket ceilings used by this batch (subset of config.buckets).
    bucket_groups : dict[int, list[int]]
        Mapping from bucket ceiling to list of original sequence indices.
    per_bucket_plans : dict[int, BatchPlan]
        Mapping from bucket ceiling to its computed BatchPlan.

    """

    bucket_boundaries: tuple[int, ...]
    bucket_groups: dict[int, list[int]]
    per_bucket_plans: dict[int, BatchPlan]

    def __post_init__(self) -> None:
        """Ensure bucket_boundaries is sorted."""
        sorted_boundaries = tuple(sorted(self.bucket_boundaries))
        if self.bucket_boundaries != sorted_boundaries:
            object.__setattr__(self, "bucket_boundaries", sorted_boundaries)
