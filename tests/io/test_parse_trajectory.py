"""aminx.io.parsing.parse_trajectory: AMBER topology + XTC frames via proxide (debt #2016).

``parse_trajectory`` used to be an alias of ``parse_input`` (one file in, one structure
out), so an ``.xtc`` could never be read with residue identities. It now takes the
topology explicitly and delegates to proxide's ``parse_amber_trajectory``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import proxide.io.parsing.backend as backend
from aminx.io.parsing import parse_trajectory

AMBER_DIR = Path(
  os.environ.get(
    "AMINX_AMBER_TEST_DIR",
    Path.home() / "projects/nanobody/data/toxins_peptide_mhc/peptide_MHC/150_2_96",
  ),
)


def test_old_proxide_fails_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delattr(backend, "parse_amber_trajectory", raising=False)
  with pytest.raises(ImportError, match="parse_amber_trajectory"):
    parse_trajectory("x.parm7", "x.xtc")  # not iterated: the check must be eager


def test_delegates_frames_and_spec(monkeypatch: pytest.MonkeyPatch) -> None:
  calls = []

  def fake(topology, trajectory, frames, spec):
    calls.append((topology, trajectory, frames, spec))
    return [f"protein-{f}" for f in frames]

  monkeypatch.setattr(backend, "parse_amber_trajectory", fake, raising=False)

  assert list(parse_trajectory("t.parm7", "t.xtc", frames=(0, -1), k_neighbors=16)) == [
    "protein-0",
    "protein--1",
  ]
  topology, trajectory, frames, spec = calls[-1]
  assert (topology, trajectory, frames) == ("t.parm7", "t.xtc", [0, -1])
  assert spec.compute_rbf is True
  assert spec.rbf_num_neighbors == 16

  assert list(parse_trajectory("t.parm7", "t.xtc")) == ["protein-0"], "default is frame 0 only"


@pytest.mark.skipif(
  not hasattr(backend, "parse_amber_trajectory"),
  reason="installed proxide lacks parse_amber_trajectory",
)
@pytest.mark.skipif(
  not (AMBER_DIR / "150_2_96.parm7").is_file(),
  reason=f"AMBER test system not found under {AMBER_DIR}",
)
def test_real_parm7_xtc_frames() -> None:
  first, last = parse_trajectory(
    AMBER_DIR / "150_2_96.parm7", AMBER_DIR / "150_2_96.xtc", frames=[0, -1],
  )
  a, b = np.asarray(first.coordinates), np.asarray(last.coordinates)
  assert a.shape == b.shape and a.shape[1:] == (37, 3)
  np.testing.assert_array_equal(np.asarray(first.aatype), np.asarray(last.aatype))
  assert not np.allclose(a, b), "frame 0 and the last frame must differ"
  # 4 protein molecules, matching the backbone breaks mdtraj's coordinates imply
  # (chain starts 0/101/377/477; proxide validation run a91fe079).
  assert len(np.unique(np.asarray(first.chain_index))) == 4  # noqa: PLR2004
  assert int(np.asarray(first.aatype).max()) < 20, "no unknown residues (AMBER variants resolved)"  # noqa: PLR2004


def test_single_path_call_still_parses_a_structure_with_a_warning(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import aminx.io.parsing.dispatch as dispatch  # noqa: PLC0415

  seen = []

  def fake_parse_structure(path, **kwargs):
    seen.append((path, kwargs))
    return "structure"

  monkeypatch.setattr(dispatch, "parse_structure", fake_parse_structure)
  with pytest.warns(DeprecationWarning, match="parse_input"):
    result = list(parse_trajectory("x.pdb", k_neighbors=12))
  assert result == ["structure"]
  assert seen == [("x.pdb", {"k_neighbors": 12})]


def test_frames_without_trajectory_is_an_error() -> None:
  with pytest.raises(TypeError, match="requires a `trajectory`"):
    parse_trajectory("x.pdb", frames=[0])
