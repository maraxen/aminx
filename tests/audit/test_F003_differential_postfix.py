"""Tier A closure assertion for F003 (task_id: 260826_aminx-invariant-audit).

CLOSURE GATE (a+b+c):
  Under the parent's F003 intent-B decision, runner.sample with multistate spec
  fields (same spec as the pre-fix differential baseline) must now RAISE.
  The pre-fix baseline showed byte-identical logits across strategies (TIER-A-PRE-FIX-INERT),
  confirming the field was inert. The post-fix adequacy test is that this call raises.

Finding summary: runner.sample() had spec.multi_state_strategy declared but silently
deleted via `del use_rolling_state, multi_state_strategy` in the inner kernel, making
the field completely inert. The fix makes this failure explicit via NotImplementedError.
"""

from pathlib import Path

import numpy as np
import pytest


# Resolved from this file, not from an absolute path into the main checkout:
# a worktree is an isolated copy, so an absolute path silently reads main's
# fixture while the tree under test is the worktree's.
PDB = str(Path(__file__).resolve().parents[2] / "tests/data/1ubq.pdb")


def _build_spec(strategy: str, L: int = 76, seed: int = 42):
    """Replicate the exact spec from F003_differential.py (pinned-wheel baseline)."""
    from aminx.run.specs import SamplingSpecification

    return SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=seed,
        multi_state_strategy=strategy,
        state_position_map=np.arange(L, dtype=np.int32)[None, :],
        model=1,
        max_length=L,
        return_logits=True,
    )


def test_differential_arithmetic_mean_now_raises() -> None:
    """The pre-fix arithmetic_mean differential call must now raise.

    F003 Tier A closure gate (a) — adequacy under intent-B:
    The spec that previously returned byte-identical logits (confirming field inertness)
    must now raise NotImplementedError at the runner.sample entry point.
    """
    from aminx.host import runner

    spec = _build_spec("arithmetic_mean")
    with pytest.raises(NotImplementedError, match="runner.sample"):
        runner.sample(spec)


def test_differential_product_now_raises() -> None:
    """The pre-fix product differential call must now raise.

    F003 Tier A closure gate (a) — adequacy under intent-B:
    Both strategies that proved field inertness (byte-identical outputs) must now raise.
    """
    from aminx.host import runner

    spec = _build_spec("product")
    with pytest.raises(NotImplementedError, match="runner.sample"):
        runner.sample(spec)


def test_error_message_names_fields_and_campaign_path() -> None:
    """The NotImplementedError must name the bad fields and point to campaign verbs.

    F003 Tier A closure gate (b) — adequacy: the error message must be informative;
    'campaign' or 'aminx campaign run' must appear in the message.
    """
    from aminx.host import runner

    spec = _build_spec("arithmetic_mean")
    with pytest.raises(NotImplementedError) as exc_info:
        runner.sample(spec)
    msg = str(exc_info.value)
    assert "state_position_map" in msg, f"Error message must name state_position_map; got: {msg[:300]}"
    assert "campaign" in msg.lower(), f"Error message must mention campaign path; got: {msg[:300]}"
