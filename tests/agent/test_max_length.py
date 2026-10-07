"""S7-08: MCP tools fit ``max_length`` to the parsed inputs on xtrax's bucket ladder."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("cisternal")
pytest.importorskip("fastmcp")

from aminx.agent.tools import fitted_max_length, max_length_ladder


def _spec(inputs: list[str], *, max_length: int = 512, pass_mode: str = "intra") -> SimpleNamespace:
  """Stand-in carrying the three fields the fitter reads."""
  return SimpleNamespace(inputs=inputs, max_length=max_length, pass_mode=pass_mode)


def test_ladder_is_xtrax_bucket_ladder() -> None:
  """The fit uses xtrax's pinned ladder, not a local copy."""
  from xtrax.export.rings import BUCKET_LADDER  # noqa: PLC0415

  assert max_length_ladder() == tuple(BUCKET_LADDER)
  assert max_length_ladder()[:4] == (64, 128, 256, 512)


def test_single_structure_takes_smallest_bucket() -> None:
  """76 residues pad to 128, not the 512 default."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 76}, explicit=False) == 128


def test_exact_bucket_is_kept() -> None:
  """A length already on a bucket boundary is not bumped to the next one."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 128}, explicit=False) == 128


def test_intra_uses_longest_structure() -> None:
  """Independent structures share one padded shape sized by the longest."""
  spec = _spec(["a.pdb", "b.pdb"])
  assert fitted_max_length(spec, {"a": 40, "b": 130}, explicit=False) == 256


def test_inter_uses_summed_length() -> None:
  """``inter`` joins inputs, so the fit covers their sum and truncates nothing."""
  spec = _spec(["a.pdb", "b.pdb"], pass_mode="inter")
  assert fitted_max_length(spec, {"a": 100, "b": 100}, explicit=False) == 256


def test_never_below_measured_residues_and_always_on_ladder() -> None:
  """Every fitted value covers the residues it measured and is a ladder rung.

  Counts above 256 land in the 512 bucket, which equals the default, so the
  spec is kept (covered by test_never_grows_past_spec_value).
  """
  ladder = set(max_length_ladder())
  for count in range(1, 257, 7):
    fitted = fitted_max_length(_spec(["a.pdb"]), {"a": count}, explicit=False)
    assert fitted is not None
    assert fitted >= count
    assert fitted in ladder


def test_explicit_max_length_is_kept() -> None:
  """A caller-set ``max_length`` is never replaced."""
  assert fitted_max_length(_spec(["a.pdb"], max_length=512), {"a": 76}, explicit=True) is None


def test_unparsed_input_keeps_spec_value() -> None:
  """If any input has no measured length the spec is left alone."""
  spec = _spec(["a.pdb", "b.pdb"])
  assert fitted_max_length(spec, {"a": 76}, explicit=False) is None
  assert fitted_max_length(spec, {}, explicit=False) is None


def test_never_grows_past_spec_value() -> None:
  """A structure longer than the default keeps the default (existing truncation behaviour)."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 600}, explicit=False) is None


def test_past_the_ladder_keeps_spec_value() -> None:
  """A length beyond the largest bucket leaves the spec alone instead of raising."""
  big = max_length_ladder()[-1] + 1
  assert fitted_max_length(_spec(["a.pdb"], max_length=4096), {"a": big}, explicit=False) is None


def test_fitting_does_not_load_the_onnx_toolchain() -> None:
  """Reading the ladder from xtrax.export must not import jax2onnx or onnx.

  jax2onnx patches jnp at import with primitives that lack CPU lowering, so it
  must never load in the server process. Runs in a fresh interpreter.
  """
  code = (
    "import sys\n"
    "from aminx.agent.tools import fitted_max_length\n"
    "from types import SimpleNamespace\n"
    "s = SimpleNamespace(inputs=['a.pdb'], max_length=512, pass_mode='intra')\n"
    "assert fitted_max_length(s, {'a': 76}, explicit=False) == 128\n"
    "bad = sorted(m for m in ('jax2onnx', 'onnx', 'onnxruntime') if m in sys.modules)\n"
    "print(','.join(bad))\n"
  )
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
  assert out.stdout.strip() == ""
