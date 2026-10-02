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

REVISION 2, AND WHY THE FIRST RUN DOES NOT COUNT

Revision 1 derived a tolerance from the maximum RELATIVE deviation. It returned
``noise_reaches_defect_scale`` -- the outcome that forbids widening -- on a
measured relative floor of 5.1e-02, i.e. 430,185x float32 eps, while the
absolute floor was only 1.43e-06. Those two cannot both describe reassociation
noise, and the pre-registered scale guard refusing to proceed is the only reason
a proposed rtol of 1.0 did not get written into a test.

The cause was a defect in the MEASUREMENT, not in the model. Relative deviation
divides by the eager magnitude, and ``etab_raw`` holds many near-zero entries,
so the statistic was dominated by its denominator. Measured on seed 0: the
worst-relative element is |diff| 5.29e-07 over |eager| 4.22e-04, and restricting
the denominator collapses the figure monotonically --

    |eager| > 0      1.25e-03
    |eager| > 0.01   4.22e-05
    |eager| > 0.1    6.54e-06

-- which is the signature of an ill-conditioned ratio, not of a model that
disagrees with itself by 5%.

The first run's verdict stands as recorded: its criteria were NOT met, and none
of its numbers are reused to choose a tolerance. What it legitimately
established is the FORM of the metric, which is what this revision changes.

WHAT ACTUALLY FAILS, AND SO WHAT ACTUALLY CHANGES

numpy's band is ``atol + rtol*|eager|``. For the element that failed --
0.1259099543094635 vs 0.12590882182121277 -- the band was
1e-6 + 1e-6*0.126 = 1.126e-06 against a deviation of 1.1325e-06. The rtol term
contributes 0.1% of that band. **atol is the binding constraint**, and it sits
below the measured absolute floor of 1.43e-06.

So only atol moves. rtol stays at 1e-6: changing a parameter that is not
violated would be widening for its own sake.

WHAT MAKES THE RESULTING TOLERANCE LEGITIMATE

The VALUE is not chosen here; a RULE fixed before this run derives it:

    floor    A = max ABSOLUTE deviation over N independently seeded models
    proposed atol = 10 * A, rounded up to one significant figure
    proposed rtol = unchanged

Rounding up to one significant figure rather than to the next power of ten is
deliberate: the power-of-ten form turned a 1.43e-06 floor into 1e-04, a 70x
jump, which would have left every element below 1e-04 effectively unchecked.
The headroom factor is fixed at 10 either way.

and the run then has to clear four bars, three of which can fail:

  1. A > the current atol, or the current tolerance was fine and the single
     failure was seed-specific -- in which case do NOT touch the test. This is
     a real possible outcome, not a formality.
  2. Every clean seed is admitted at the proposed band.
  3. A fixed 1e-3 relative defect injected into one element is REJECTED at the
     proposed band. Without this the "fix" is indistinguishable from deleting
     the assertion: a tolerance that admits everything is not a test, and a
     positive control that can only pass is not a check.
  4. The proposed atol stays well below that defect's ABSOLUTE size, so the
     band cannot swallow the thing it is supposed to catch.

The conditioned relative floor (|eager| > 0.1) is reported for context and is
deliberately NOT used to set rtol -- having been burned once by a statistic
chosen for availability rather than for meaning.

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

# The test's current tolerances. atol is the one outcome 1 is measured against,
# because atol is the term that actually binds: for the element that failed, the
# rtol term contributed 0.1% of the 1.126e-06 band.
_CURRENT_RTOL = 1e-6
_CURRENT_ATOL = 1e-6
# Fixed before the run: proposed atol = this * absolute floor, rounded up to one
# significant figure. Revision 1 rounded to the next power of ten, which turned a
# 1.43e-06 floor into 1e-04 -- a 70x jump that would leave every element below
# 1e-04 unchecked. The headroom factor is 10 either way; only the rounding moved.
_HEADROOM = 10.0
# Denominator floor for the CONTEXT-ONLY conditioned relative statistic. Revision
# 1 divided by every nonzero eager value and produced 5.1e-02, which is an
# artifact of near-zero denominators rather than a property of the model. This
# figure is reported, never used to derive a tolerance.
_CONDITION_THRESHOLD = 0.1
# Fixed before the run: the relative size of the defect the tolerance must still
# catch. NOT derived from the measured floor -- a control scaled to the floor is
# very nearly circular, since the proposed band is also derived from it. 1e-3 is
# instead the scale of a defect we would actually care about: three orders above
# float32 eps, far below the 1218-absolute deviation that permute_etab_out_rows
# produces, and the region where a genuine numerical error would live. If the
# measured floor ever forces the band up to this scale, that is a finding in its
# own right -- jit-vs-eager noise as large as a real defect -- and this run is
# designed to FAIL rather than quietly widen past it. That is exactly what
# happened in revision 1, and it is why revision 1's numbers are not reused.
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
  etab_rel_conditioned: float
  logp_max_abs: float
  logp_max_rel: float
  logp_rel_conditioned: float


def _deviation(compiled: np.ndarray, eager: np.ndarray) -> tuple[float, float, float]:
  """Max absolute, max relative, and max CONDITIONED relative deviation.

  numpy's assert_allclose compares against ``atol + rtol*|desired|`` with the
  eager array as ``desired``, so a relative denominator has to be the eager
  magnitude to mean the same thing the test means.

  The unconditioned relative figure is kept only so the two can be compared:
  it divides by every nonzero eager value, and ``etab_raw`` is full of near-zero
  entries, so it measures its own denominator more than it measures the model.
  The conditioned figure restricts to ``|eager| > _CONDITION_THRESHOLD``, where
  the ratio is actually informative. Neither drives the proposed tolerance --
  only the absolute deviation does.
  """
  delta = np.abs(compiled - eager)
  max_abs = float(delta.max()) if delta.size else 0.0

  nonzero = np.abs(eager) > 0.0
  max_rel = float((delta[nonzero] / np.abs(eager[nonzero])).max()) if nonzero.any() else 0.0

  well_conditioned = np.abs(eager) > _CONDITION_THRESHOLD
  conditioned = (
    float((delta[well_conditioned] / np.abs(eager[well_conditioned])).max())
    if well_conditioned.any()
    else 0.0
  )
  return max_abs, max_rel, conditioned


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
  etab_abs, etab_rel, etab_cond = _deviation(
    np.asarray(compiled.etab_raw),
    np.asarray(eager.etab_raw),
  )
  logp_abs, logp_rel, logp_cond = _deviation(
    np.asarray(compiled.log_probs),
    np.asarray(eager.log_probs),
  )
  return SeedResult(
    seed,
    etab_abs,
    etab_rel,
    etab_cond,
    logp_abs,
    logp_rel,
    logp_cond,
  )


def _ceil_one_significant_figure(value: float) -> float:
  """Round ``value`` UP to one significant figure; 0 maps to 0.

  Used instead of rounding to the next power of ten, which inflates by up to
  10x. On the revision-1 floor that turned 1.43e-06 into 1e-04 and would have
  left everything below 1e-04 unchecked; this gives 2e-05.
  """
  if value <= 0.0:
    return 0.0
  exponent = math.floor(math.log10(value))
  scale = 10.0**exponent
  return float(math.ceil(value / scale) * scale)


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
  floor_rel_conditioned = max(max(r.etab_rel_conditioned, r.logp_rel_conditioned) for r in results)

  # The RULE, fixed before the run. Only atol moves: it is the term that binds,
  # and rtol is not violated. floor_rel plays no part in the derivation -- it is
  # reported so the ill-conditioning that sank revision 1 stays visible.
  proposed_atol = _ceil_one_significant_figure(_HEADROOM * floor_abs)
  proposed_rtol = _CURRENT_RTOL

  floor_above_current = floor_abs > _CURRENT_ATOL
  not_admitted = [
    r.seed for r in results if not _admits(r.seed, args.length, proposed_rtol, proposed_atol)
  ]
  control = _control_rejected(args.length, proposed_rtol, proposed_atol)

  # The derived band must stay well BELOW the defect it has to catch. The band
  # here is atol-dominated, so the comparison is against the control's ABSOLUTE
  # size, not its relative one. If atol ever reaches that, the honest answer is
  # "f32 jit-vs-eager noise is the size of a real defect here", which needs a
  # decision, not a wider band.
  below_control_scale = proposed_atol < control["injected_absolute"]

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
    "current_atol": _CURRENT_ATOL,
    "floor_abs": floor_abs,
    "floor_rel": floor_rel,
    "floor_rel_conditioned": floor_rel_conditioned,
    "condition_threshold": _CONDITION_THRESHOLD,
    "float32_eps": float(np.finfo(np.float32).eps),
    "floor_rel_in_eps": floor_rel / float(np.finfo(np.float32).eps),
    "floor_rel_conditioned_in_eps": floor_rel_conditioned / float(np.finfo(np.float32).eps),
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
