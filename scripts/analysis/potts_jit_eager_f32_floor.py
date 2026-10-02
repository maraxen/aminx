"""Measure the jit-vs-eager float32 reproducibility floor for PottsMPNN.

WHY THIS EXISTS

``tests/families/potts_mpnn/test_model.py::test_jit_matches_eager`` asserts
``atol=1e-6, rtol=1e-6`` between an eager and an ``eqx.filter_jit`` forward pass
on the SAME float32 inputs. It was green for months and began failing the moment
a leaked ``jax_enable_x64`` was fixed (debt #2419 -> #2432): the leak had been
promoting the intermediates to float64, where 1e-6 is ~4.5e9 eps and the
assertion is unfalsifiable. At float32 it is ~8 eps.

So the test had never checked this path at the precision production uses, and
float64 is not available as a fix -- it would cost the GPU performance the whole
port exists to get.

The one observed violation was 8.99e-06 relative on a value of 0.126, i.e. ~75x
float32 eps. That is the ordinary cost of XLA fusing and reassociating
reductions differently from the eager path, not a correctness defect. But "75
eps looks normal" is an assertion, and a tolerance chosen after seeing the
number it has to admit is exactly how a band gets fitted to its data.

WHAT MAKES THE RESULTING TOLERANCE LEGITIMATE

The VALUE is not chosen here; a RULE fixed before the run derives it:

    floor    F = max relative deviation over N independently seeded models
    proposed rtol = the next power of ten at or above 10 * F

and the run then has to clear three bars, two of which can fail:

  1. F > 1e-6, or the current tolerance was fine and the single failure was
     seed-specific -- in which case do NOT touch the test. This is a real
     possible outcome, not a formality.
  2. Every clean seed is admitted at the proposed rtol.
  3. A perturbation of relative size 100 * F injected into one element is
     REJECTED at the proposed rtol. Without this the "fix" is indistinguishable
     from deleting the assertion: a tolerance that admits everything is not a
     test, and a positive control that can only pass is not a check.

Each seed is written as its own JSON plus a completion stamp before the next
starts, and a re-run reuses a seed only when the script hash and shape key
match, so an interruption costs one seed rather than the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.model import PottsMPNN, PottsMPNNOutput

logger = logging.getLogger("potts_jit_eager_f32_floor")

# The test's current tolerance, and the threshold outcome 1 is measured against.
_CURRENT_RTOL = 1e-6
# Fixed before the run: proposed rtol is the next power of ten >= this * floor.
_HEADROOM = 10.0
# Fixed before the run: the relative size of the defect the tolerance must still
# catch. NOT derived from the measured floor -- a control scaled to the floor is
# very nearly circular, since proposed_rtol is also derived from it, and
# next_power_of_ten(10*F) <= 100*F makes "100*F is rejected" true by
# construction with as little as 1x margin. 1e-3 is instead the scale of a
# defect we would actually care about: three orders above float32 eps, far below
# the 1218-absolute deviation that permute_etab_out_rows produces, and the
# region where a genuine numerical error would live. If the measured floor ever
# forces proposed_rtol up to 1e-3, that is a finding in its own right -- it
# would mean jit-vs-eager noise is the size of a real defect -- and this run is
# designed to FAIL rather than quietly widen past it.
_CONTROL_RELATIVE = 1e-3


def _backbone(length: int, dtype: jnp.dtype) -> jax.Array:
  """The test's own synthetic backbone, copied verbatim so the floor is ITS floor.

  Deliberately duplicated rather than imported: importing from a test module
  would make this script's number depend on a file that may be edited in
  response to this script's number.
  """
  index = jnp.arange(length, dtype=dtype)
  ca = jnp.stack(
    (
      index * jnp.asarray(3.8, dtype=dtype),
      jnp.zeros(length, dtype),
      jnp.zeros(length, dtype),
    ),
    axis=-1,
  )
  nitrogen = ca + jnp.asarray([-0.5, 1.0, 0.0], dtype=dtype)
  carbon = ca + jnp.asarray([1.5, 0.0, 0.0], dtype=dtype)
  oxygen = carbon + jnp.asarray([0.0, 1.2, 0.0], dtype=dtype)
  return jnp.stack((nitrogen, ca, carbon, oxygen), axis=1)


def _call(model: PottsMPNN, coords: jax.Array, mask: jax.Array) -> PottsMPNNOutput:
  length = coords.shape[0]
  return model(
    coords,
    mask,
    jnp.arange(length, dtype=jnp.int32),
    jnp.zeros((length,), dtype=jnp.int32),
    jnp.zeros((length,), dtype=jnp.int32),
    jnp.arange(length, dtype=jnp.int32),
  )


@dataclass(frozen=True)
class SeedResult:
  """One seed's deviation between the eager and compiled forward pass."""

  seed: int
  etab_max_abs: float
  etab_max_rel: float
  logp_max_abs: float
  logp_max_rel: float


def _deviation(compiled: np.ndarray, eager: np.ndarray) -> tuple[float, float]:
  """Max absolute and max relative deviation, relative to the EAGER value.

  numpy's assert_allclose compares against ``atol + rtol*|desired|`` with the
  eager array as ``desired``, so the relative denominator has to be the eager
  magnitude to mean the same thing the test means. Elements whose eager value is
  exactly zero are excluded from the relative figure and are covered by the
  absolute one, because a relative deviation from zero is not defined.
  """
  delta = np.abs(compiled - eager)
  max_abs = float(delta.max()) if delta.size else 0.0
  nonzero = np.abs(eager) > 0.0
  if not nonzero.any():
    return max_abs, 0.0
  max_rel = float((delta[nonzero] / np.abs(eager[nonzero])).max())
  return max_abs, max_rel


def _measure(seed: int, length: int) -> SeedResult:
  """Eager vs compiled on identical float32 inputs for one seeded model."""
  coords = _backbone(length, jnp.float32)
  mask = jnp.ones((length,), dtype=jnp.float32)
  model = PottsMPNN(key=jax.random.key(seed))

  @eqx.filter_jit
  def run(module: PottsMPNN, xyz: jax.Array, residue_mask: jax.Array) -> PottsMPNNOutput:
    return _call(module, xyz, residue_mask)

  eager = _call(model, coords, mask)
  compiled = run(model, coords, mask)
  etab_abs, etab_rel = _deviation(
    np.asarray(compiled.etab_raw),
    np.asarray(eager.etab_raw),
  )
  logp_abs, logp_rel = _deviation(
    np.asarray(compiled.log_probs),
    np.asarray(eager.log_probs),
  )
  return SeedResult(seed, etab_abs, etab_rel, logp_abs, logp_rel)


def _next_power_of_ten(value: float) -> float:
  """Smallest power of ten >= ``value``; 0 maps to 0."""
  if value <= 0.0:
    return 0.0
  return float(10.0 ** math.ceil(math.log10(value)))


def _admits(seed: int, length: int, rtol: float, atol: float) -> bool:
  """Whether ``assert_allclose(rtol, atol)`` admits this seed. Re-runs it.

  An earlier version of this function reconstructed numpy's verdict from the
  recorded maxima, which is NOT sound: numpy requires every element to satisfy
  ``|diff| <= atol + rtol*|eager|``, and the element with the largest relative
  deviation need not be the one with the largest absolute deviation, so neither
  maximum constrains the other's element. The only honest check is the real
  one. The forward pass is deterministic in the seed, so re-running it is exact
  rather than merely close.
  """
  coords = _backbone(length, jnp.float32)
  mask = jnp.ones((length,), dtype=jnp.float32)
  model = PottsMPNN(key=jax.random.key(seed))

  @eqx.filter_jit
  def run(module: PottsMPNN, xyz: jax.Array, residue_mask: jax.Array) -> PottsMPNNOutput:
    return _call(module, xyz, residue_mask)

  eager = _call(model, coords, mask)
  compiled = run(model, coords, mask)
  for got, want in (
    (compiled.etab_raw, eager.etab_raw),
    (compiled.log_probs, eager.log_probs),
  ):
    try:
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=rtol, atol=atol)
    except AssertionError:
      return False
  return True


def _control_rejected(length: int, rtol: float, atol: float) -> dict:
  """Inject a ``_CONTROL_RELATIVE`` defect in one element; the band must reject it.

  This is the half that makes the proposed tolerance a test rather than a
  deletion: a tolerance that admits everything is not a test. The perturbation
  goes into the single largest-magnitude element so the injected relative size
  is unambiguous, and the verdict comes from the SAME ``assert_allclose`` call
  the test uses -- not from hand-computing the band at one element, which would
  only check the element I happened to pick.
  """
  coords = _backbone(length, jnp.float32)
  mask = jnp.ones((length,), dtype=jnp.float32)
  model = PottsMPNN(key=jax.random.key(0))
  eager = np.asarray(_call(model, coords, mask).etab_raw)

  index = int(np.abs(eager).argmax())
  target = float(eager.flat[index])
  perturbed = eager.copy()
  perturbed.flat[index] = target * (1.0 + _CONTROL_RELATIVE)

  rejected = False
  try:
    np.testing.assert_allclose(perturbed, eager, rtol=rtol, atol=atol)
  except AssertionError:
    rejected = True

  delta = abs(float(perturbed.flat[index]) - target)
  return {
    "element_value": target,
    "injected_relative": _CONTROL_RELATIVE,
    "injected_absolute": delta,
    "band_at_element": atol + rtol * abs(target),
    "margin_over_band": delta / (atol + rtol * abs(target)),
    "rejected": rejected,
  }


def _script_sha256() -> str:
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _unit_key(seed: int, length: int) -> str:
  payload = f"{_script_sha256()}|{seed}|{length}".encode()
  return hashlib.sha256(payload).hexdigest()


def _load_unit(work_dir: Path, seed: int, length: int) -> SeedResult | None:
  stamp = work_dir / f"seed_{seed}.stamp"
  body = work_dir / f"seed_{seed}.json"
  if not (stamp.is_file() and body.is_file()):
    return None
  if stamp.read_text(encoding="utf-8").strip() != _unit_key(seed, length):
    return None
  data = json.loads(body.read_text(encoding="utf-8"))
  return SeedResult(**data)


def _store_unit(work_dir: Path, result: SeedResult, length: int) -> None:
  body = work_dir / f"seed_{result.seed}.json"
  body.write_text(json.dumps(result.__dict__), encoding="utf-8")
  # Body first, stamp second: a crash between them leaves an unstamped body,
  # which is recomputed rather than trusted.
  (work_dir / f"seed_{result.seed}.stamp").write_text(
    _unit_key(result.seed, length),
    encoding="utf-8",
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seeds", type=int, default=24, help="number of seeded models")
  parser.add_argument("--length", type=int, default=5, help="residues, as the test uses")
  parser.add_argument("--work-dir", type=Path, required=True, help="per-seed artifacts")
  parser.add_argument("--out", type=Path, required=True, help="result JSON")
  parser.add_argument("--no-resume", action="store_true", help="recompute every seed")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

  work_dir = args.work_dir
  work_dir.mkdir(parents=True, exist_ok=True)

  results: list[SeedResult] = []
  reused = 0
  for seed in range(args.seeds):
    cached = None if args.no_resume else _load_unit(work_dir, seed, args.length)
    if cached is not None:
      reused += 1
      results.append(cached)
      logger.info("seed %d reused (etab_max_rel %.3e)", seed, cached.etab_max_rel)
      continue
    result = _measure(seed, args.length)
    _store_unit(work_dir, result, args.length)
    results.append(result)
    logger.info(
      "seed %d etab_max_rel %.3e etab_max_abs %.3e logp_max_rel %.3e",
      seed,
      result.etab_max_rel,
      result.etab_max_abs,
      result.logp_max_rel,
    )

  # The floor is the worst case across BOTH arrays and every seed: the test
  # asserts on etab_raw and log_probs with one shared tolerance, so a band
  # derived from either alone would not cover what the test checks.
  floor_rel = max(max(r.etab_max_rel, r.logp_max_rel) for r in results)
  floor_abs = max(max(r.etab_max_abs, r.logp_max_abs) for r in results)

  # The RULE, fixed before the run. The value falls out of it.
  proposed_rtol = _next_power_of_ten(_HEADROOM * floor_rel)
  proposed_atol = _next_power_of_ten(_HEADROOM * floor_abs)

  floor_above_current = floor_rel > _CURRENT_RTOL
  not_admitted = [
    r.seed for r in results if not _admits(r.seed, args.length, proposed_rtol, proposed_atol)
  ]
  control = _control_rejected(args.length, proposed_rtol, proposed_atol)

  # The derived tolerance must stay BELOW the defect scale it has to catch.
  # If the floor ever pushes it up to _CONTROL_RELATIVE, the honest answer is
  # "f32 jit-vs-eager noise is the size of a real defect here", which needs a
  # decision, not a wider band.
  below_control_scale = proposed_rtol < _CONTROL_RELATIVE

  # Named to match the sidecar's four outcomes, which are mutually exclusive
  # and exhaustive, so a reader is not left deciding whether "fail" meant the
  # hypothesis was refuted or the instrument misbehaved.
  if not floor_above_current:
    verdict = "tolerance_was_fine"
  elif not below_control_scale:
    verdict = "noise_reaches_defect_scale"
  elif not_admitted or not control["rejected"]:
    verdict = "fail"
  else:
    verdict = "pass"

  payload = {
    "verdict": verdict,
    "n_seeds": len(results),
    "n_reused": reused,
    "n_computed": len(results) - reused,
    "current_rtol": _CURRENT_RTOL,
    "floor_rel": floor_rel,
    "floor_abs": floor_abs,
    "float32_eps": float(np.finfo(np.float32).eps),
    "floor_in_eps": floor_rel / float(np.finfo(np.float32).eps),
    "headroom": _HEADROOM,
    "proposed_rtol": proposed_rtol,
    "proposed_atol": proposed_atol,
    "floor_above_current": floor_above_current,
    "clean_seeds_admitted": not not_admitted,
    "seeds_not_admitted": not_admitted,
    "control": control,
    "control_rejected": control["rejected"],
    "control_relative": _CONTROL_RELATIVE,
    "below_control_scale": below_control_scale,
    "per_seed": [r.__dict__ for r in results],
    "script_sha256": _script_sha256(),
    "work_dir": str(work_dir),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  logger.info(json.dumps({k: v for k, v in payload.items() if k != "per_seed"}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
