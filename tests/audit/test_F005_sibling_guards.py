"""F005 closure assertions — task_id 260826_aminx-invariant-audit.

Tier R: runner.jacobian and runner.inspect silently accepted
spec.state_position_map (field declared on the spec classes, never read).
Post-fix both entry points RAISE NotImplementedError at the entry point,
naming the field and pointing at the honouring paths (runner.score /
campaign verbs) — mirroring the F002 guard shape from 7460516a.
"""
from __future__ import annotations

import importlib.metadata as im
from pathlib import Path

import numpy as np
import pytest

# Resolved from this file, not from an absolute path into the main checkout:
# a worktree is an isolated copy, so an absolute path silently reads main's
# fixture while the tree under test is the worktree's.
PDB = str(Path(__file__).resolve().parents[2] / "tests/data/1ubq.pdb")


def _stamp() -> str:
    return im.version("aminx")


def test_version_stamp_is_post_fix():
    assert _stamp() >= "0.1.0a27"


@pytest.mark.parametrize("spec_name", ["JacobianSpecification", "InspectionSpecification"])
def test_state_position_map_raises_at_entry_point(spec_name):
    import aminx.run.specs as specs
    from aminx.host import runner

    spec_cls = getattr(specs, spec_name)
    spec = spec_cls(
        inputs=[PDB],
        batch_size=1,
        random_seed=42,
        model=1,
        max_length=76,
        state_position_map=np.arange(76, dtype=np.int32)[None, :],
    )
    entry = runner.jacobian if spec_name == "JacobianSpecification" else runner.inspect
    with pytest.raises(NotImplementedError) as ei:
        entry(spec)
    msg = str(ei.value)
    assert "state_position_map" in msg
    assert "_score_fused_multistate" in msg or "campaign verbs" in msg


def test_error_names_the_honouring_paths():
    """Adequacy: the refusal message redirects to score + campaign verbs."""
    from aminx.host import runner
    from aminx.run.specs import InspectionSpecification

    spec = InspectionSpecification(
        inputs=[PDB],
        batch_size=1,
        random_seed=42,
        model=1,
        max_length=76,
        state_position_map=np.arange(76, dtype=np.int32)[None, :],
    )
    with pytest.raises(NotImplementedError) as ei:
        runner.inspect(spec)
    assert "runner.score" in str(ei.value)


def test_specs_declare_the_field():
    """The field is declared on both classes (the silent-accept surface)."""
    import dataclasses

    from aminx.run.specs import InspectionSpecification, JacobianSpecification

    for cls in (JacobianSpecification, InspectionSpecification):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "state_position_map" in names
