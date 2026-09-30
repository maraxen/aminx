"""`ligand_conditioning` is a two-way switch, not a one-way assertion (#114).

`SamplingSpecification.ligand_conditioning` was only consulted to decide whether to RAISE when
ligand tensors were absent. Real tensors -- on the batch, or loaded from `ligand_context_path`
-- were injected regardless of its value, so `ligand_conditioning=False` (a campaign's
`ligand_on=False` row, or a caller asking for an ablation) silently got full ligand
conditioning: a "no ligand" arm that was byte-for-byte the "ligand" arm.

Now tri-state:  None = use whatever is present (legacy default) | True = require real tensors |
False = ablate (ignore tensors, feed the zero placeholder).

task_id: 260930_resolve-issues-debt
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from aminx.cli import app
from aminx.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _prepare_ligand_context,
  ligand_conditioning_mode,
)
from aminx.host.campaign import plan_campaign_manifest
from aminx.run.specs import SamplingSpecification
from tests.host.test_sampling_helper import (
  _make_fake_protein,
  _make_ligand_tensors,
  _write_keyed_npz,
)

_CKPT = "ligandmpnn_v_32_020_25"
_B, _L, _A = 1, 4, 3


def _ligand_file(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
  tensors = _make_ligand_tensors(np.random.default_rng(0), _L, _A)
  path = tmp_path / "ligand_context.npz"
  _write_keyed_npz(path, {"struct_a": tensors})
  return path, tensors


def _spec(**kwargs: Any) -> SamplingSpecification:
  return SamplingSpecification(inputs=["/tmp/struct_a.pdb"], checkpoint_id=_CKPT, **kwargs)


def _prepare(spec: SamplingSpecification, protein: Any = None) -> dict[str, Any]:
  ids = _canonical_structure_ids_for_spec(spec)
  return _prepare_ligand_context(
    spec,
    batched_ensemble=protein if protein is not None else _make_fake_protein(_B, _L),
    batch_size=_B,
    seq_len=_L,
    canonical_structure_ids=ids,
    batch_structure_ids=ids,
  )


def _batch_with_ligand() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
  tensors = _make_ligand_tensors(np.random.default_rng(1), _L, _A)
  protein = SimpleNamespace(
    Y=jnp.asarray(tensors["Y"])[None],
    Y_t=jnp.asarray(tensors["Y_t"], dtype=jnp.int32)[None],
    Y_m=jnp.asarray(tensors["Y_m"])[None],
  )
  return protein, tensors


def _is_placeholder(ctx: dict[str, Any]) -> bool:
  return not np.any(np.asarray(ctx["Y_m"])) and not np.any(np.asarray(ctx["Y"]))


def test_default_is_unset_not_false() -> None:
  assert _spec().ligand_conditioning is None


def test_false_IGNORES_ligand_context_path(tmp_path: Path) -> None:
  """THE bug. An explicit False with a real ligand file must NOT inject the file."""
  path, tensors = _ligand_file(tmp_path)
  ctx = _prepare(_spec(ligand_conditioning=False, ligand_context_path=path))
  assert _is_placeholder(ctx), "ligand_conditioning=False still injected the ligand file"
  assert not np.array_equal(np.asarray(ctx["Y"])[0], tensors["Y"])


def test_false_IGNORES_ligand_tensors_on_the_batch() -> None:
  protein, _ = _batch_with_ligand()
  ctx = _prepare(_spec(ligand_conditioning=False), protein)
  assert _is_placeholder(ctx)


def test_false_does_not_even_read_a_declared_path(tmp_path: Path) -> None:
  """A campaign hands every row the same base_spec, path included; False must not choke on it."""
  ctx = _prepare(_spec(ligand_conditioning=False, ligand_context_path=tmp_path / "absent.npz"))
  assert _is_placeholder(ctx)


def test_unset_keeps_legacy_behaviour_file_is_used(tmp_path: Path) -> None:
  """Back-compat: None (every caller that never set the field) still uses what is present."""
  path, tensors = _ligand_file(tmp_path)
  ctx = _prepare(_spec(ligand_context_path=path))
  np.testing.assert_array_equal(np.asarray(ctx["Y"])[0], tensors["Y"])


def test_unset_keeps_legacy_behaviour_batch_tensors_are_used() -> None:
  protein, tensors = _batch_with_ligand()
  ctx = _prepare(_spec(), protein)
  np.testing.assert_array_equal(np.asarray(ctx["Y"])[0], tensors["Y"])


def test_unset_without_tensors_is_the_zero_placeholder() -> None:
  assert _is_placeholder(_prepare(_spec()))


def test_true_uses_the_file_and_still_raises_when_nothing_is_present(tmp_path: Path) -> None:
  path, tensors = _ligand_file(tmp_path)
  ctx = _prepare(_spec(ligand_conditioning=True, ligand_context_path=path))
  np.testing.assert_array_equal(np.asarray(ctx["Y"])[0], tensors["Y"])
  with pytest.raises(ValueError, match="ligand_conditioning=True requires ligand context"):
    _prepare(_spec(ligand_conditioning=True))


def test_false_without_any_tensors_does_not_raise() -> None:
  assert _is_placeholder(_prepare(_spec(ligand_conditioning=False)))


def test_provenance_mode_distinguishes_unset_from_ablated() -> None:
  assert ligand_conditioning_mode(None) == "auto"
  assert ligand_conditioning_mode(True) == "required"
  assert ligand_conditioning_mode(False) == "ablated"


def test_run_spec_keeps_the_tristate() -> None:
  assert _spec().run_spec.ligand.ligand_conditioning is None
  assert _spec(ligand_conditioning=False).run_spec.ligand.ligand_conditioning is False
  assert _spec(ligand_conditioning=True).run_spec.ligand.ligand_conditioning is True


def test_campaign_false_row_is_a_real_ablation_end_to_end(tmp_path: Path) -> None:
  """Plan with a ligand_context_path -> reconstruct each row as the worker does -> prepare.

  The grid's ligand_on=False rows share base_spec's ligand_context_path. Before the fix they
  were byte-identical to the ligand_on=True rows (the duplicate work tev_design had to patch).
  """
  path, tensors = _ligand_file(tmp_path)
  rows = plan_campaign_manifest(
    base_spec=_spec(ligand_context_path=path),
    campaign_id="lig",
    output_root=str(tmp_path / "out"),
    designs_per_library_type=1,
    samples_chunk_size=1,
  )
  by_flag: dict[bool, dict[str, Any]] = {}
  for row in rows:
    payload = json.loads(json.dumps(row["sampling_spec"]))
    worker = SamplingSpecification(**payload)
    by_flag[bool(worker.ligand_conditioning)] = _prepare(worker)
    assert row["ligand_conditioning"] == bool(worker.ligand_conditioning)
  assert _is_placeholder(by_flag[False])
  np.testing.assert_array_equal(np.asarray(by_flag[True]["Y"])[0], tensors["Y"])


def _emit(*extra: str) -> dict[str, Any]:
  result = CliRunner().invoke(app, ["run", "sample", "--inputs", "fake.pdb", *extra, "--emit-json"])
  assert result.exit_code == 0, result.output
  return json.loads(result.output)


def test_cli_default_is_unset_and_flags_are_tristate() -> None:
  assert _emit()["ligand_conditioning"] is None
  assert _emit("--ligand-conditioning")["ligand_conditioning"] is True
  assert _emit("--no-ligand-conditioning")["ligand_conditioning"] is False
